"""Port of the numeric core of `sb3_contrib.crossq.crossq.CrossQ.train`.

The distinguishing feature of CrossQ is the joint forward pass over
`th.cat([observations, next_observations])` and
`th.cat([actions, next_actions])`; the two reductions that follow are what this
module ports. The names `target_q_values` and `critic_loss` are new, the
arithmetic is line-for-line what `CrossQ.train` computes.
"""

from __future__ import annotations

import numpy as np

from .._lib import lib
from .._tensors import as_f64, restore, scalar

__all__ = ["target_q_values", "critic_loss"]


def target_q_values(
    all_q_values,
    next_log_prob,
    rewards,
    dones,
    discounts,
    ent_coef: float,
):
    """The target-Q half of `CrossQ.train`.

    :param all_q_values: (2 * batch_size, n_critics), the joint obs/next_obs
        forward pass
    :param next_log_prob: (batch_size,)
    :param rewards: (batch_size,)
    :param dones: (batch_size,)
    :param discounts: (batch_size,) the per-sample discount factor
    :param ent_coef: entropy temperature
    :return: (batch_size,)
    """
    q = as_f64(all_q_values)
    if q.ndim != 2:
        raise ValueError(f"all_q_values must be 2D, got {q.ndim}D")
    n_critics = q.shape[1]
    if q.shape[0] % 2 != 0:
        raise ValueError(f"all_q_values has {q.shape[0]} rows, expected an even number")
    batch_size = q.shape[0] // 2

    log_prob = as_f64(next_log_prob).reshape(-1)
    rew = as_f64(rewards).reshape(-1)
    done = as_f64(dones).reshape(-1)
    disc = as_f64(discounts).reshape(-1)
    for name, arr in (("next_log_prob", log_prob), ("rewards", rew), ("dones", done), ("discounts", disc)):
        if arr.size != batch_size:
            raise ValueError(f"{name} has {arr.size} elements, expected {batch_size}")

    out = np.empty(batch_size, dtype=np.float64)
    lib().sb3c_crossq_target_q_values(
        q.ctypes.data,
        log_prob.ctypes.data,
        rew.ctypes.data,
        done.ctypes.data,
        disc.ctypes.data,
        out.ctypes.data,
        batch_size,
        n_critics,
        float(ent_coef),
    )
    return restore(all_q_values, out)


def critic_loss(current_q_values, target_q_values):
    """`critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values.detach()) ...)`

    :param current_q_values: (n_critics, batch_size) or (batch_size, n_critics)
    :param target_q_values: (batch_size,)
    """
    cur = as_f64(current_q_values)
    tgt = as_f64(target_q_values).reshape(-1)
    if cur.ndim == 3:
        # upstream keeps (n_critics, batch_size, 1) after the .T[..., None]
        if cur.shape[-1] != 1:
            raise ValueError(f"expected a trailing axis of 1, got shape {cur.shape}")
        cur = cur.reshape(cur.shape[0], cur.shape[1])
    if cur.ndim != 2:
        raise ValueError(f"current_q_values must be 2D or 3D, got {cur.ndim}D")
    if cur.shape[-1] == tgt.size:
        n_critics, batch_size = cur.shape
    elif cur.shape[0] == tgt.size:
        batch_size, n_critics = cur.shape
        cur = np.ascontiguousarray(cur.T)
    else:
        raise ValueError(
            f"current_q_values of shape {cur.shape} does not match {tgt.size} targets"
        )
    value = lib().sb3c_crossq_critic_loss(
        cur.ctypes.data, tgt.ctypes.data, batch_size, n_critics
    )
    return scalar(current_q_values, value)
