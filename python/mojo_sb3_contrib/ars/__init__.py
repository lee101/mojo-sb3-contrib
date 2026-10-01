"""Port of the numeric core of `sb3_contrib.ars.ars.ARS._do_one_update`.

Upstream performs the update in three inline steps: take the top-`n_top`
candidates, gather the selected returns and noise directions, and apply the
step. The names here (`top_returns`, `gather`, `update_weights`) are new; the
arithmetic is line-for-line what `_do_one_update` computes. The Gaussian noise
sampling and the environment rollouts upstream does around these blocks are not
arithmetic the port replaces.
"""

from __future__ import annotations

import numpy as np

from .._lib import lib, scratch
from .._tensors import as_f64, restore, scalar

__all__ = ["top_returns", "gather", "update_weights"]


def top_returns(plus_returns, minus_returns, n_top: int):
    """Keep only the top performing candidates for update.

    :param plus_returns: (n_delta,) returns for weights + deltas
    :param minus_returns: (n_delta,) returns for weights - deltas
    :param n_top: how many candidates survive
    :return: (top_idx, return_std), top_idx being (n_top,) and return_std the
        unbiased standard deviation of the concatenated selected returns
    """
    plus = as_f64(plus_returns).reshape(-1)
    minus = as_f64(minus_returns).reshape(-1)
    if plus.size != minus.size:
        raise ValueError(
            f"plus_returns has {plus.size} elements but minus_returns has {minus.size}"
        )
    n_delta = plus.size
    if not 0 < n_top <= n_delta:
        raise ValueError(f"n_top={n_top} must be in (0, n_delta={n_delta}]")

    idx = np.zeros(n_top, dtype=np.int64)
    std = np.zeros(1, dtype=np.float64)
    got = lib().sb3c_ars_top_returns(
        plus.ctypes.data,
        minus.ctypes.data,
        idx.ctypes.data,
        std.ctypes.data,
        n_delta,
        n_top,
    )
    return idx[:got], float(std[0])


def gather(plus_returns, minus_returns, deltas, top_idx):
    """plus_returns = plus_returns[top_idx]; deltas = deltas[top_idx].

    :param plus_returns: (n_delta,)
    :param minus_returns: (n_delta,)
    :param deltas: (n_delta, n_params)
    :param top_idx: (n_top,) indices from `top_returns`
    :return: (plus_returns, minus_returns, deltas), all narrowed to n_top
    """
    plus = as_f64(plus_returns).reshape(-1)
    minus = as_f64(minus_returns).reshape(-1)
    d = as_f64(deltas)
    if d.ndim != 2:
        raise ValueError(f"deltas must be 2D, got {d.ndim}D")
    n_delta, n_params = d.shape
    d = np.ascontiguousarray(d)
    idx = as_f64(top_idx).reshape(-1).astype(np.int64)
    n_top = idx.size

    out_plus = np.empty(n_top, dtype=np.float64)
    out_minus = np.empty(n_top, dtype=np.float64)
    out_deltas = np.empty((n_top, n_params), dtype=np.float64)
    lib().sb3c_ars_gather(
        plus.ctypes.data,
        minus.ctypes.data,
        d.ctypes.data,
        idx.ctypes.data,
        out_plus.ctypes.data,
        out_minus.ctypes.data,
        out_deltas.ctypes.data,
        n_top,
        n_params,
    )
    return (
        restore(plus_returns, out_plus),
        restore(minus_returns, out_minus),
        restore(deltas, out_deltas),
    )


def update_weights(weights, plus_returns, minus_returns, deltas, return_std, learning_rate: float):
    """self.weights = self.weights + step_size * ((plus - minus) @ deltas)

    :param weights: (n_params,) the current flattened policy weights
    :param plus_returns: (n_top,) the gathered plus returns
    :param minus_returns: (n_top,) the gathered minus returns
    :param deltas: (n_top, n_params) the gathered noise directions
    :param return_std: the value `top_returns` returned
    :param learning_rate: the current learning rate
    :return: (new_weights, step_size)
    """
    w = as_f64(weights).reshape(-1)
    plus = as_f64(plus_returns).reshape(-1)
    minus = as_f64(minus_returns).reshape(-1)
    d = as_f64(deltas)
    if d.ndim != 2:
        raise ValueError(f"deltas must be 2D, got {d.ndim}D")
    n_top, n_params = d.shape
    d = np.ascontiguousarray(d)
    if plus.size != n_top or minus.size != n_top:
        raise ValueError(
            f"expected {n_top} plus/minus returns, got {plus.size}/{minus.size}"
        )
    if w.size != n_params:
        raise ValueError(f"weights has {w.size} elements, expected {n_params}")
    w = np.ascontiguousarray(w)

    out = np.empty(n_params, dtype=np.float64)
    work = scratch(n_params)
    step_size = lib().sb3c_ars_update_weights(
        w.ctypes.data,
        plus.ctypes.data,
        minus.ctypes.data,
        d.ctypes.data,
        out.ctypes.data,
        work.ctypes.data,
        float(learning_rate),
        float(return_std),
        n_top,
        n_params,
    )
    return restore(weights, out), scalar(learning_rate, step_size)
