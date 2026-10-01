"""Port of `sb3_contrib.common.maskable.distributions`.

The compute-bound part is `MaskableCategorical.apply_masking` and its `entropy`
override, which MaskablePPO calls on every policy step. The surrounding
`torch.distributions` plumbing is not arithmetic and is not ported; this module
exposes the two kernels plus a thin `MaskableCategorical` holder with the same
attribute and method names.
"""

from __future__ import annotations

import numpy as np

from ..._lib import lib, scratch
from ..._tensors import as_f64, restore, scalar

__all__ = ["MaskableCategorical", "apply_masking", "entropy"]

HUGE_NEG = -1e8


def apply_masking(logits, masks):
    """Eliminate ("mask out") chosen categorical outcomes.

    :param logits: the unmasked logits
    :param masks: a boolean array of the same shape. If True, the corresponding
        choice's logit value is preserved. If False, it is set to a large
        negative value, resulting in near 0 probability.
    :return: the re-normalised log-probabilities torch's Categorical holds
    """
    lg = as_f64(logits)
    mk = as_f64(masks).reshape(-1).astype(np.int64)
    if mk.size != lg.size:
        raise ValueError(f"masks has {mk.size} elements, expected {lg.size}")
    out = np.empty_like(lg)
    lib().sb3c_apply_masking(lg.ctypes.data, mk.ctypes.data, out.ctypes.data, lg.size)
    return restore(logits, out)


def entropy(logits, masks=None):
    """The entropy of a masked categorical distribution.

    Masked-out choices do not contribute, because a masked action possesses no
    uncertainty.
    """
    lg = as_f64(logits)
    if lg.ndim != 1:
        raise ValueError(f"Expected 1D logits (got {lg.ndim}D)")
    n = lg.size
    if masks is None:
        mk = np.ones(n, dtype=np.int64)
        use_masks = False
    else:
        mk = as_f64(masks).reshape(-1).astype(np.int64)
        if mk.size != n:
            raise ValueError(f"masks has {mk.size} elements, expected {n}")
        use_masks = True
    work = scratch(n)
    value = lib().sb3c_entropy(lg.ctypes.data, mk.ctypes.data, work.ctypes.data, n, use_masks)
    return scalar(logits, value)


class MaskableCategorical:
    """`sb3_contrib.common.maskable.distributions.MaskableCategorical`.

    Upstream subclasses `torch.distributions.Categorical`; this keeps the same
    constructor arguments and the same `logits` / `probs` / `masks` / `entropy`
    surface, backed by numpy arrays.
    """

    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified, but not both.")
        if probs is not None:
            self._original_logits = _probs_to_logits(probs)
        else:
            self._original_logits = as_f64(logits).reshape(-1).copy()
        # torch's Categorical normalises its logits to log-probabilities, and
        # upstream stores that normalised copy as `_original_logits`.
        self._original_logits = self._normalise(self._original_logits)
        self.logits = self._original_logits.copy()
        self.masks = None
        self.apply_masking(masks)

    @staticmethod
    def _normalise(logits: np.ndarray) -> np.ndarray:
        keep = np.ones(logits.size, dtype=np.int64)
        return np.asarray(apply_masking(logits, keep), dtype=np.float64).reshape(-1)

    def apply_masking(self, masks):
        """See `apply_masking`. `masks=None` removes any previous masking."""
        if masks is None:
            self.masks = None
            keep = np.ones(self._original_logits.size, dtype=np.int64)
        else:
            self.masks = as_f64(masks).reshape(-1).astype(bool)
            keep = self.masks.astype(np.int64)
        self.logits = np.asarray(
            apply_masking(self._original_logits, keep), dtype=np.float64
        ).reshape(-1)
        return self

    @property
    def probs(self) -> np.ndarray:
        shifted = self.logits - self.logits.max()
        e = np.exp(shifted)
        return e / e.sum()

    def entropy(self):
        if self.masks is None:
            return entropy(self.logits)
        return entropy(self.logits, self.masks.astype(np.int64))

    def log_prob(self, value):
        return float(np.log(self.probs[value]))


def _probs_to_logits(probs) -> np.ndarray:
    p = as_f64(probs).reshape(-1)
    if np.any(p < 0):
        raise ValueError("The parameter probs must be non-negative.")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("The parameter probs must sum to 1.")
    # torch.distributions.utils.probs_to_logits: take the log relative to the
    # largest positive entry. softmax() of the result reproduces `probs`, and a
    # zero probability stays -inf.
    with np.errstate(divide="ignore"):
        base = np.log(p)
    return (base - base[p > 0].max()).astype(np.float64)
