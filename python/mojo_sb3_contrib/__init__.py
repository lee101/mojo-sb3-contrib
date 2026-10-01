"""mojo-sb3-contrib: the compute-bound kernels of sb3-contrib, in Mojo.

The module layout mirrors upstream, so a reader with the Python package open can
find the same code under the same path:

    mojo_sb3_contrib.common.utils                 -> sb3_contrib.common.utils
    mojo_sb3_contrib.common.torch_layers          -> sb3_contrib.common.torch_layers
    mojo_sb3_contrib.common.maskable.distributions
    mojo_sb3_contrib.tqc / .qrdqn / .crossq / .ars
"""

from . import ars, common, crossq, qrdqn, tqc
from .common.maskable.distributions import MaskableCategorical, apply_masking, entropy
from .common.torch_layers import BatchRenorm, BatchRenorm1d
from .common.utils import conjugate_gradient_solver, quantile_huber_loss

__all__ = [
    "ars",
    "common",
    "crossq",
    "qrdqn",
    "tqc",
    "quantile_huber_loss",
    "conjugate_gradient_solver",
    "BatchRenorm",
    "BatchRenorm1d",
    "MaskableCategorical",
    "apply_masking",
    "entropy",
]
__version__ = "0.1.0"
