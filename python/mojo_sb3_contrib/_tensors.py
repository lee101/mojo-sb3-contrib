"""Shared helpers for handing torch tensors and numpy arrays to the kernels."""

from __future__ import annotations

import numpy as np

from ._lib import f64



def as_f64(a) -> np.ndarray:
    return f64(a)


def _is_torch(x) -> bool:
    return hasattr(x, "detach") and hasattr(x, "device")


def restore(reference, values: np.ndarray):
    """Return `values` shaped/dtyped like `reference` (torch in, torch out)."""
    if _is_torch(reference):
        import torch

        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if isinstance(reference, np.ndarray):
        return values.astype(reference.dtype, copy=False)
    return float(values)


def scalar(reference, value: float):
    if _is_torch(reference):
        import torch

        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return value
