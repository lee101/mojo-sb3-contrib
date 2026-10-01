"""Port of the numeric core of `sb3_contrib.tqc.tqc.TQC.train`.

Upstream writes this block inline in `train()` rather than in a named function,
so the names here (`target_quantiles`, `critic_loss`) are new; everything they
compute is line-for-line what `TQC.train` computes.
"""

from __future__ import annotations

import numpy as np

from .._lib import lib, scratch
from .._tensors import as_f64, restore
from ..common.utils import quantile_huber_loss

__all__ = ["target_quantiles", "critic_loss"]


def target_quantiles(
    next_quantiles,
    next_log_prob,
    rewards,
    dones,
    discounts,
    ent_coef,
    top_quantiles_to_drop_per_net: int,
    n_critics: int,
    n_quantiles: int,
):
    """The `with th.no_grad():` block of `TQC.train`.

    :param next_quantiles: (batch_size, n_critics, n_quantiles) target critic output
    :param next_log_prob: (batch_size,) log-probabilities of the next action
    :param rewards: (batch_size,)
    :param dones: (batch_size,)
    :param discounts: (batch_size,) the per-sample discount factor
    :param ent_coef: entropy temperature
    :param top_quantiles_to_drop_per_net: quantiles dropped per network
    :param n_critics: number of critics
    :param n_quantiles: quantiles per critic
    :return: (batch_size, n_target_quantiles)
    """
    nq = as_f64(next_quantiles)
    if nq.ndim != 3 or nq.shape[1] != n_critics or nq.shape[2] != n_quantiles:
        raise ValueError(
            f"next_quantiles must be (batch, {n_critics}, {n_quantiles}), got {nq.shape}"
        )
    batch_size = nq.shape[0]
    n_target_quantiles = n_critics * n_quantiles - top_quantiles_to_drop_per_net * n_critics
    if n_target_quantiles <= 0:
        raise ValueError(
            "top_quantiles_to_drop_per_net leaves no quantiles: "
            f"{n_critics}x{n_quantiles} - {top_quantiles_to_drop_per_net}x{n_critics}"
        )

    log_prob = as_f64(next_log_prob).reshape(-1)
    rew = as_f64(rewards).reshape(-1)
    done = as_f64(dones).reshape(-1)
    disc = as_f64(discounts).reshape(-1)
    for name, arr in (("next_log_prob", log_prob), ("rewards", rew), ("dones", done), ("discounts", disc)):
        if arr.size != batch_size:
            raise ValueError(f"{name} has {arr.size} elements, expected {batch_size}")

    out = np.empty((batch_size, n_target_quantiles), dtype=np.float64)
    work = scratch(n_critics * n_quantiles)
    lib().sb3c_tqc_target_quantiles(
        nq.ctypes.data,
        log_prob.ctypes.data,
        rew.ctypes.data,
        done.ctypes.data,
        disc.ctypes.data,
        work.ctypes.data,
        out.ctypes.data,
        batch_size,
        n_critics,
        n_quantiles,
        n_target_quantiles,
        float(ent_coef),
    )
    return restore(next_quantiles, out)


def critic_loss(current_quantiles, target_quantiles, cum_prob=None):
    """`TQC.train`'s critic loss: `quantile_huber_loss(..., sum_over_quantiles=False)`.

    :param current_quantiles: (batch_size, n_critics, n_quantiles)
    :param target_quantiles: (batch_size, n_target_quantiles), as returned above
    """
    target = as_f64(target_quantiles)
    if target.ndim == 2:
        # (batch_size, 1, n_target_quantiles): upstream unsqueezes here so the
        # target broadcasts against (batch_size, n_critics, n_quantiles).
        target = target[:, None, :]
    return quantile_huber_loss(
        current_quantiles, target, cum_prob=cum_prob, sum_over_quantiles=False
    )
