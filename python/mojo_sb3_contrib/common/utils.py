"""Port of `sb3_contrib.common.utils`.

The two compute-bound helpers upstream keeps there are `quantile_huber_loss`,
which every QR-DQN and TQC critic step calls, and `conjugate_gradient_solver`,
which ARS uses to invert its Hessian-vector product. `flat_grad` is not ported:
it is a thin wrapper over `torch.autograd.grad` and contains no arithmetic of
its own.
"""

from __future__ import annotations

import math
import shutil
import subprocess

import numpy as np

from .._lib import lib, scratch
from .._tensors import as_f64, restore, scalar

__all__ = ["quantile_huber_loss", "conjugate_gradient_solver"]

# Must match `DW` in src/device.mojo: the kernel's tile width, and therefore
# the size of the per-thread partial array it writes.
_GPU_WIDTH = 4

# The device path costs about 0.2 ms of fixed overhead (context, four buffer
# allocations, one launch, one copy back). Below roughly half a million
# pairwise terms the CPU kernel is faster; above it the device wins by 1.8x to
# 20x. Measured on the shapes in bench/bench.py.
_GPU_MIN_TERMS = 500_000

_gpu_usable_cache = None


def _gpu_usable() -> bool:
    """Whether the device is worth calling into. Checked once per process."""
    global _gpu_usable_cache
    if _gpu_usable_cache is None:
        _gpu_usable_cache = False
        try:
            if shutil.which("nvidia-smi"):
                probe = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                free_mib = int(probe.stdout.split(",")[0].strip())
                _gpu_usable_cache = free_mib >= 4000
        except Exception:
            _gpu_usable_cache = False
    return _gpu_usable_cache


def _quantile_huber_loss_gpu(cur, tgt, cp, batch_size, n_critics, n_quantiles,
                             n_target_quantiles, ndim, sum_over_quantiles):
    """The device kernel, or NaN if it could not run."""
    partials = np.empty(
 max(1, batch_size * -(-n_target_quantiles // _GPU_WIDTH)), dtype=np.float64
    )
    return lib().sb3c_quantile_huber_loss_gpu(
        cur.ctypes.data,
        tgt.ctypes.data,
        cp.ctypes.data,
        partials.ctypes.data,
        batch_size,
        n_critics,
        n_quantiles,
        n_target_quantiles,
        ndim,
        bool(sum_over_quantiles),
    )


def quantile_huber_loss(
    current_quantiles,
    target_quantiles,
    cum_prob=None,
    sum_over_quantiles: bool = True,
):
    """The quantile-regression loss, as described in the QR-DQN and TQC papers.

    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    cur = as_f64(current_quantiles)
    tgt = as_f64(target_quantiles)
    if cur.ndim != tgt.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({cur.ndim}) needs to match "
            f"the dimension of target_quantiles ({tgt.ndim})."
        )
    if cur.shape[0] != tgt.shape[0]:
        raise ValueError(
            f"Error: The batch size of curremt_quantile ({cur.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({tgt.shape[0]})."
        )
    if cur.ndim not in (2, 3):
        raise ValueError(
            f"Error: The dimension of current_quantiles ({cur.ndim}) needs to be either 2 or 3."
        )

    batch_size = cur.shape[0]
    n_critics = cur.shape[1] if cur.ndim == 3 else 1
    n_quantiles = cur.shape[-1]
    n_target_quantiles = tgt.shape[-1]

    # Upstream right-aligns a caller-supplied cum_prob against pairwise_delta,
    # whose axes are (batch, [critics], n_quantiles, n_target_quantiles). The
    # kernel only needs the three axes that survive: batch, critic, target.
    cp_batch_stride, cp_critic_stride, cp_target_stride = 0, 0, 0
    # The kernel materialises both of upstream's th.float weights per current
    # quantile here, so the default path needs two entries per quantile.
    cp = scratch(2 * n_quantiles)
    if cum_prob is not None:
        cp = as_f64(cum_prob)
        rank = 4 if cur.ndim == 3 else 3
        shape = (1,) * (rank - cp.ndim) + cp.shape
        expected = (
            (batch_size, n_critics, n_quantiles, n_target_quantiles)
            if cur.ndim == 3
            else (batch_size, n_quantiles, n_target_quantiles)
        )
        for got, want in zip(shape, expected):
            if got != 1 and got != want:
                raise ValueError(
                    f"Error: cum_prob of shape {cp.shape} is not broadcastable to {expected}."
                )
        # element stride of each axis, right-aligned
        strides = [1] * rank
        for axis in range(rank - 2, -1, -1):
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        if shape[0] == batch_size:
            cp_batch_stride = strides[0]
        if cur.ndim == 3 and shape[1] == n_critics:
            cp_critic_stride = strides[1]
        if shape[-1] == n_target_quantiles:
            cp_target_stride = 1

    # The device kernel covers upstream's default `cum_prob` only, and only
    # once the term count is past its fixed cost. Anything it declines - a
    # caller-supplied cum_prob, a small batch, a busy or absent device - comes
    # back as NaN and lands on the CPU kernel below.
    terms = batch_size * n_critics * n_quantiles * n_target_quantiles
    if cum_prob is None and terms >= _GPU_MIN_TERMS and _gpu_usable():
        loss = _quantile_huber_loss_gpu(
            cur, tgt, cp, batch_size, n_critics, n_quantiles,
            n_target_quantiles, cur.ndim, sum_over_quantiles,
        )
        if not math.isnan(loss):
            return scalar(current_quantiles, loss)

    loss = lib().sb3c_quantile_huber_loss(
        cur.ctypes.data,
        tgt.ctypes.data,
        cp.ctypes.data,
        batch_size,
        n_critics,
        n_quantiles,
        n_target_quantiles,
        cur.ndim,
        cum_prob is not None,
        bool(sum_over_quantiles),
        cp_batch_stride,
        cp_critic_stride,
        cp_target_stride,
    )
    return scalar(current_quantiles, loss)


def conjugate_gradient_solver(
    matrix_vector_dot_fn,
    b,
    max_iter: int = 10,
    residual_tol: float = 1e-10,
):
    """Finds an approximate solution to a set of linear equations Ax = b

    :param matrix_vector_dot_fn:
        a function that right multiplies a matrix A by a vector v
    :param b:
        the right hand term in the set of linear equations Ax = b
    :param max_iter:
        the maximum number of iterations (default is 10)
    :param residual_tol:
        residual tolerance for early stopping of the solving (default is 1e-10)
    :return x:
        the approximate solution to the system of equations defined by `matrix_vector_dot_fn`
        and b
    """
    import torch

    is_torch = hasattr(b, "detach")
    shape = as_f64(b).shape
    rhs = as_f64(b).reshape(-1).copy()
    n = rhs.size
    if n == 0:
        return restore(b, rhs.reshape(shape))

    # The kernel works on a dense matrix, so materialise A by right-multiplying
    # it with the unit vectors. This is the one place the port diverges from
    # upstream: the callable cannot cross the C ABI.
    matrix = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        col = as_f64(matrix_vector_dot_fn(_unit_vector(i, n, is_torch, b))).reshape(-1)
        if col.size != n:
            raise ValueError(
                f"matrix_vector_dot_fn returned {col.size} elements, expected {n}."
            )
        matrix[:, i] = col
    matrix = np.ascontiguousarray(matrix)

    # The vector is not initialized at 0 because of the instability issues when
    # the gradient becomes small. A small random gaussian noise is used for the
    # initialization.
    if is_torch:
        x = as_f64(1e-4 * torch.randn_like(b.to(torch.float64))).reshape(-1)
    else:
        x = 1e-4 * np.random.standard_normal(n)
    x = np.ascontiguousarray(x)

    work = scratch(3 * n)
    lib().sb3c_conjugate_gradient_solver(
        matrix.ctypes.data,
        rhs.ctypes.data,
        x.ctypes.data,
        work.ctypes.data,
        n,
        int(max_iter),
        float(residual_tol),
    )
    return restore(b, x.reshape(shape))


def _unit_vector(i: int, n: int, is_torch: bool, reference):
    if is_torch:
        import torch

        basis = torch.zeros(n, dtype=torch.float64, device=reference.device)
        basis[i] = 1.0
        return basis
    basis = np.zeros(n, dtype=np.float64)
    basis[i] = 1.0
    return basis
