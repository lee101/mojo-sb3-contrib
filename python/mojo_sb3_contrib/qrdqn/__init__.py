"""Port of the numeric core of `sb3_contrib.qrdqn.qrdqn.QRDQN.train`.

Upstream writes this block inline in `train()`; the name `target_quantiles` is
new, the arithmetic is line-for-line what `QRDQN.train` computes.
"""

from __future__ import annotations

import numpy as np

from .._lib import lib, scratch
from .._tensors import as_f64, restore

__all__ = ["target_quantiles"]


def target_quantiles(next_quantiles, rewards, dones, discounts):
    """The `with th.no_grad():` block of `QRDQN.train`.

    :param next_quantiles: (batch_size, n_quantiles, n_actions) output of the target network
    :param rewards: (batch_size,)
    :param dones: (batch_size,)
    :param discounts: (batch_size,) the per-sample discount factor
    :return: (batch_size, n_quantiles)
    """
    nq = as_f64(next_quantiles)
    if nq.ndim != 3:
        raise ValueError(f"next_quantiles must be 3D, got {nq.ndim}D")
    batch_size, n_quantiles, n_actions = nq.shape

    rew = as_f64(rewards).reshape(-1)
    done = as_f64(dones).reshape(-1)
    disc = as_f64(discounts).reshape(-1)
    for name, arr in (("rewards", rew), ("dones", done), ("discounts", disc)):
        if arr.size != batch_size:
            raise ValueError(f"{name} has {arr.size} elements, expected {batch_size}")

    out = np.empty((batch_size, n_quantiles), dtype=np.float64)
    work = scratch(n_actions)
    lib().sb3c_qrdqn_target_quantiles(
        nq.ctypes.data,
        rew.ctypes.data,
        done.ctypes.data,
        disc.ctypes.data,
        out.ctypes.data,
        work.ctypes.data,
        batch_size,
        n_quantiles,
        n_actions,
    )
    return restore(next_quantiles, out)
