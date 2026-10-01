"""Port of `sb3_contrib.common.torch_layers`.

Only `BatchRenorm`/`BatchRenorm1d` are here. Upstream's class is a
`torch.nn.Module`; the arithmetic in `forward` is what is compute-bound, so that
is what this port keeps, with the module's state living in plain arrays.
"""

from __future__ import annotations

import numpy as np

from .._lib import lib, scratch
from .._tensors import as_f64, restore

__all__ = ["BatchRenorm", "BatchRenorm1d"]


class BatchRenorm:
    """BatchRenorm Module (https://arxiv.org/abs/1702.03275).

    :param num_features: Number of features in the input tensor.
    :param eps: A value added to the variance for numerical stability.
    :param momentum: The value used for the ra_mean and ra_var (running average) computation.
        It controls the rate of convergence for the batch renormalization statistics.
    :param affine: A boolean value that when set to True, this module has learnable
            affine parameters. Default: True
    :param warmup_steps: Number of steps that are performed before the running statistics
        are used for normalization. During the warmup phase, the batch statistics are used.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 100_000,
    ):
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.warmup_steps = int(warmup_steps)
        # Clip scale and bias of the affine transform
        self.rmax = 3.0
        self.dmax = 5.0
        self.ra_mean = np.zeros(self.num_features, dtype=np.float64)
        self.ra_var = np.ones(self.num_features, dtype=np.float64)
        self.steps = 0
        self.scale = np.ones(self.num_features, dtype=np.float64)
        self.bias = np.zeros(self.num_features, dtype=np.float64)
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def _check_input_dim(self, x) -> None:
        raise NotImplementedError()

    def forward(self, x):
        """
        Normalize the input tensor.

        :param x: Input tensor, (batch_size, num_features)
        :return: Normalized tensor, same shape.
        """
        # Upstream never calls _check_input_dim from forward either; a 1-D input
        # broadcasts there and is rejected here. See README, "Known divergences".
        data = as_f64(x)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input (got {data.ndim}D input)")
        batch_size, num_features = data.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features (got {num_features})"
            )

        out = np.empty_like(data)
        work = scratch(4 * self.num_features)
        self.steps = lib().sb3c_batch_renorm_forward(
            data.ctypes.data,
            out.ctypes.data,
            work.ctypes.data,
            self.ra_mean.ctypes.data,
            self.ra_var.ctypes.data,
            self.scale.ctypes.data,
            self.bias.ctypes.data,
            batch_size,
            num_features,
            bool(self.training),
            int(self.steps),
            self.eps,
            self.momentum,
            bool(self.affine),
            self.rmax,
            self.dmax,
            self.warmup_steps,
        )
        return restore(x, out)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, momentum={self.momentum}, "
            f"warmup_steps={self.warmup_steps}, affine={self.affine}"
        )


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x) -> None:
        if as_f64(x).ndim == 1:
            raise ValueError(f"Expected 2D or 3D input (got {as_f64(x).ndim}D input)")
