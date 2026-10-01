"""Runs the TQC critic loss through both implementations and prints the delta.

    pixi run python examples/quickstart.py
"""

import numpy as np
import torch as th

from mojo_sb3_contrib import tqc
from mojo_sb3_contrib.common.utils import quantile_huber_loss

from sb3_contrib.common.utils import quantile_huber_loss as upstream_quantile_huber_loss

rng = np.random.default_rng(0)

batch_size, n_critics, n_quantiles = 256, 2, 25
top_quantiles_to_drop_per_net = 2

next_quantiles = th.as_tensor(rng.normal(size=(batch_size, n_critics, n_quantiles)), dtype=th.float64)
current_quantiles = th.as_tensor(rng.normal(size=(batch_size, n_critics, n_quantiles)), dtype=th.float64)
next_log_prob = th.as_tensor(rng.normal(size=batch_size), dtype=th.float64)
rewards = th.as_tensor(rng.normal(size=(batch_size, 1)), dtype=th.float64)
dones = th.zeros(batch_size, 1, dtype=th.float64)
discounts = th.full((batch_size, 1), 0.99, dtype=th.float64)
ent_coef = 0.2

# 1. the target-quantile block of TQC.train
target_quantiles = tqc.target_quantiles(
    next_quantiles, next_log_prob, rewards, dones, discounts, ent_coef,
    top_quantiles_to_drop_per_net, n_critics, n_quantiles,
)
print("target quantiles", tuple(target_quantiles.shape), float(target_quantiles[0, 0]))

# the same block, written out of TQC.train by hand
flat, _ = th.sort(next_quantiles.reshape(batch_size, -1))
kept = flat[:, : n_critics * n_quantiles - top_quantiles_to_drop_per_net * n_critics]
reference = rewards + (1 - dones) * discounts * (kept - ent_coef * next_log_prob.reshape(-1, 1))
print("max |target delta| vs upstream:", np.abs(target_quantiles.numpy() - reference.numpy()).max())

# 2. the critic loss
ours = tqc.critic_loss(current_quantiles, target_quantiles)
theirs = upstream_quantile_huber_loss(
    current_quantiles, target_quantiles[:, None, :], sum_over_quantiles=False
)
print(f"critic loss  mojo {float(ours):.12f}   upstream {float(theirs):.12f}")
print("relative delta:", abs(float(ours) - float(theirs)) / abs(float(theirs)))
