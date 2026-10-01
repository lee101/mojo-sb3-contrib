"""Parity tests against the real `sb3_contrib` package.

Every reference here is upstream code: either a function imported from
`sb3_contrib`, a `torch.nn.Module` from `sb3_contrib`, or the exact expression
from the upstream source, run in float64. Nothing is compared against a
hand-written formula written for this test.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th

import mojo_sb3_contrib as m
from mojo_sb3_contrib import ars, crossq, qrdqn, tqc
from mojo_sb3_contrib.common.torch_layers import BatchRenorm1d
from mojo_sb3_contrib.common.utils import conjugate_gradient_solver, quantile_huber_loss

import sb3_contrib.qrdqn.qrdqn as qrdqn_upstream
import sb3_contrib.tqc.tqc as tqc_upstream
from sb3_contrib.common.maskable.distributions import MaskableCategorical as UpstreamMaskableCategorical
from sb3_contrib.common.torch_layers import BatchRenorm1d as UpstreamBatchRenorm1d
from sb3_contrib.common.utils import conjugate_gradient_solver as upstream_conjugate_gradient_solver
from sb3_contrib.common.utils import quantile_huber_loss as upstream_quantile_huber_loss

th.set_num_threads(4)

F64 = th.float64
RNG = np.random.default_rng(0)


def rand(*shape):
    return th.as_tensor(RNG.normal(size=shape), dtype=F64)


def rand_dones(batch_size):
    return th.as_tensor(RNG.integers(0, 2, size=batch_size), dtype=F64)


# ---------------------------------------------------------------------------
# sb3_contrib.common.utils.quantile_huber_loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,n_quantiles,n_target",
    [(1, 1, 1), (7, 5, 5), (64, 200, 200), (33, 17, 40)],
)
def test_quantile_huber_loss_qrdqn_matches_upstream(batch_size, n_quantiles, n_target):
    """QR-DQN calls it with (batch, n_quantiles) and sum_over_quantiles=True."""
    current = rand(batch_size, n_quantiles)
    target = rand(batch_size, n_target)
    expected = upstream_quantile_huber_loss(current, target, sum_over_quantiles=True)
    got = quantile_huber_loss(current, target, sum_over_quantiles=True)
    assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    "batch_size,n_critics,n_quantiles,n_target",
    [(1, 1, 2, 1), (256, 2, 25, 50), (8, 3, 9, 12)],
)
def test_quantile_huber_loss_tqc_matches_upstream(batch_size, n_critics, n_quantiles, n_target):
    """TQC calls it with (batch, n_critics, n_quantiles) and sum_over_quantiles=False."""
    current = rand(batch_size, n_critics, n_quantiles)
    target = rand(batch_size, 1, n_target)
    expected = upstream_quantile_huber_loss(current, target, sum_over_quantiles=False)
    got = quantile_huber_loss(current, target, sum_over_quantiles=False)
    assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


def test_quantile_huber_loss_cum_prob_matches_upstream():
    batch_size, n_quantiles = 16, 11
    current = rand(batch_size, n_quantiles)
    target = rand(batch_size, n_quantiles)
    cum_prob = th.as_tensor(RNG.uniform(size=(batch_size, 1, n_quantiles)), dtype=F64)
    expected = upstream_quantile_huber_loss(current, target, cum_prob=cum_prob)
    got = quantile_huber_loss(current, target, cum_prob=cum_prob)
    assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


def test_quantile_huber_loss_accepts_float32_like_upstream():
    current = th.as_tensor(RNG.normal(size=(32, 20)), dtype=th.float32)
    target = th.as_tensor(RNG.normal(size=(32, 20)), dtype=th.float32)
    expected = upstream_quantile_huber_loss(current, target)
    got = quantile_huber_loss(current, target)
    assert got.dtype == th.float32
    assert float(got) == pytest.approx(float(expected), rel=1e-5, abs=1e-7)


def test_quantile_huber_loss_rejects_the_shapes_upstream_rejects():
    current = rand(4, 3)
    with pytest.raises(ValueError, match="needs to match the dimension"):
        quantile_huber_loss(current, rand(4, 3, 3))
    with pytest.raises(ValueError, match="batch size"):
        quantile_huber_loss(current, rand(5, 3))
    with pytest.raises(ValueError, match="either 2 or 3"):
        quantile_huber_loss(rand(4, 1, 1, 3), rand(4, 1, 1, 3))


def test_tqc_critic_loss_matches_upstream():
    """`tqc.critic_loss` is exactly upstream's `quantile_huber_loss(..., False)` call."""
    current = rand(64, 2, 25)
    target = rand(64, 50)[:, None, :]
    expected = upstream_quantile_huber_loss(current, target, sum_over_quantiles=False)
    got = tqc.critic_loss(current, target)
    assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


# ---------------------------------------------------------------------------
# sb3_contrib.common.utils.conjugate_gradient_solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 5, 24])
def test_conjugate_gradient_solver_matches_upstream(n):
    matrix = th.as_tensor(
        np.diag(np.linspace(1.0, 4.0, n)) + RNG.normal(scale=0.05, size=(n, n)), dtype=F64
    )
    matrix = matrix @ matrix.T + n * th.eye(n, dtype=F64)
    b = rand(n)
    th.manual_seed(1234)
    expected = upstream_conjugate_gradient_solver(lambda v: matrix @ v, b, max_iter=10, residual_tol=1e-10)
    th.manual_seed(1234)
    got = conjugate_gradient_solver(lambda v: matrix @ v, b, max_iter=10, residual_tol=1e-10)
    assert th.allclose(th.as_tensor(got), expected, rtol=1e-9, atol=1e-11)
    # and the residual is no worse than upstream's own after 10 CG steps
    got_residual = (matrix @ th.as_tensor(got) - b).norm()
    ref_residual = (matrix @ expected - b).norm()
    assert got_residual <= 2 * ref_residual + 1e-12


def test_conjugate_gradient_solver_stops_on_tiny_gradient():
    """Upstream returns the initial guess when the residual is already tiny."""
    th.manual_seed(7)
    b = th.full((4,), 1e-9, dtype=F64)
    th.manual_seed(7)
    expected = upstream_conjugate_gradient_solver(lambda v: v, b)
    th.manual_seed(7)
    got = conjugate_gradient_solver(lambda v: v, b)
    assert th.allclose(th.as_tensor(got), expected, rtol=1e-12, atol=1e-14)
    assert th.isfinite(th.as_tensor(got)).all()


# ---------------------------------------------------------------------------
# sb3_contrib.tqc.tqc.TQC.train
# ---------------------------------------------------------------------------


def tqc_target_reference(
    next_quantiles, next_log_prob, rewards, dones, discounts, ent_coef, n_target_quantiles
):
    """The `with th.no_grad():` block of upstream TQC.train, verbatim."""
    batch_size = next_quantiles.shape[0]
    next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
    next_quantiles = next_quantiles[:, :n_target_quantiles]
    target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
    target_quantiles = rewards + (1 - dones) * discounts * target_quantiles
    return target_quantiles


@pytest.mark.parametrize(
    "batch_size,n_critics,n_quantiles,drop_per_net,ent_coef",
    [(1, 2, 25, 2, 0.1), (256, 2, 25, 2, 0.2), (64, 3, 25, 5, 1.0)],
)
def test_tqc_target_quantiles_matches_upstream_expressions(
    batch_size, n_critics, n_quantiles, drop_per_net, ent_coef
):
    next_quantiles = rand(batch_size, n_critics, n_quantiles)
    next_log_prob = rand(batch_size)
    rewards = rand(batch_size, 1)
    dones = rand_dones(batch_size).reshape(-1, 1)
    discounts = th.full((batch_size, 1), 0.99, dtype=F64)

    expected = tqc_target_reference(
        next_quantiles, next_log_prob, rewards, dones, discounts, ent_coef,
        n_critics * n_quantiles - drop_per_net * n_critics,
    )
    got = tqc.target_quantiles(
        next_quantiles, next_log_prob, rewards, dones, discounts, ent_coef, drop_per_net, n_critics, n_quantiles
    )
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=1e-12)


def test_tqc_target_quantiles_with_a_real_critic():
    """Feed the kernel a real upstream TQC Critic's output."""
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from sb3_contrib.tqc.policies import Critic

    obs_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    critic = Critic(
        obs_space,
        spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        net_arch=[32, 32],
        features_extractor=FlattenExtractor(obs_space),
        features_dim=4,
    ).double()
    obs = th.as_tensor(RNG.normal(size=(17, 4)), dtype=F64)
    action = th.as_tensor(RNG.normal(size=(17, 2)), dtype=F64)
    with th.no_grad():
        next_quantiles = critic(obs, action)
    assert tuple(next_quantiles.shape) == (17, 2, 25)

    rewards = rand(17, 1)
    dones = th.zeros(17, 1, dtype=F64)
    discounts = th.full((17, 1), 0.99, dtype=F64)
    next_log_prob = rand(17)

    expected = tqc_target_reference(next_quantiles, next_log_prob, rewards, dones, discounts, 0.2, 40)
    got = tqc.target_quantiles(next_quantiles, next_log_prob, rewards, dones, discounts, 0.2, 5, 2, 25)
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=1e-12)


def _fill_buffer(model, env, n_steps=200):
    """Fill a replay buffer deterministically, in the shape SB3 2.9 expects."""
    obs, _ = env.reset(seed=0)
    model.replay_buffer.seed = 0
    for _ in range(n_steps):
        action = np.asarray([env.action_space.sample()])
        next_obs, reward, terminated, truncated, _ = env.step(action[0])
        model.replay_buffer.add(obs, next_obs, action, np.array([reward]), np.array([terminated]), [{}])
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()


def _spy_on_loss(module, upstream_fn, our_fn, records, use_ours: bool, monkeypatch=None):
    """Wrap upstream's loss so each call is checked and (optionally) replaced."""

    def spy(current_quantiles, target_quantiles, cum_prob=None, sum_over_quantiles=True):
        reference = upstream_fn(current_quantiles, target_quantiles, cum_prob, sum_over_quantiles)
        ours = our_fn(current_quantiles, target_quantiles, cum_prob, sum_over_quantiles)
        records.append((current_quantiles, target_quantiles, cum_prob, sum_over_quantiles,
                        float(reference.detach()), float(ours)))
        if not use_ours:
            return reference
        # Feed the training loop our value while keeping a grad_fn, so
        # loss.backward() still works: the constant offset carries no gradient
        # and the graph is upstream's.
        return reference + (float(ours) - reference.detach())

    monkeypatch.setattr(module, "quantile_huber_loss", spy)
    return spy


def test_tqc_train_loss_agrees_with_upstream(monkeypatch):
    """Per-call parity inside a real TQC.train, on the real TQC data shapes."""
    from stable_baselines3.common.utils import set_random_seed
    from sb3_contrib.common.utils import quantile_huber_loss as reference_loss

    records = []
    set_random_seed(0)
    env = gym.make("Pendulum-v1")
    model = tqc_upstream.TQC(
        "MlpPolicy", env, learning_starts=0, buffer_size=1000, verbose=0, seed=0,
        policy_kwargs=dict(net_arch=[16, 16], n_quantiles=10, n_critics=2),
    )
    _fill_buffer(model, env)
    model._setup_learn(0, None)
    monkeypatch.setattr(
        tqc_upstream,
        "quantile_huber_loss",
        _spy_on_loss(tqc_upstream, reference_loss, quantile_huber_loss, records, True, monkeypatch),
    )
    model.train(gradient_steps=3, batch_size=32)

    assert len(records) == 3
    recorded = float(model.logger.name_to_value["train/critic_loss"])
    for current, target, cum_prob, sum_over, reference, ours in records:
        assert current.shape == (32, 2, 10)
        assert target.shape == (32, 1, 16)  # 2 critics x 10 quantiles - 2 dropped x 2 critics
        assert cum_prob is None
        assert sum_over is False
        assert ours == pytest.approx(reference, rel=1e-6, abs=1e-9)
    assert recorded == pytest.approx(np.mean([r[4] for r in records]), rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# sb3_contrib.qrdqn.qrdqn.QRDQN.train
# ---------------------------------------------------------------------------


def qrdqn_target_reference(next_quantiles, rewards, dones, discounts, n_quantiles):
    """The `with th.no_grad():` block of upstream QRDQN.train, verbatim."""
    greedy = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
    greedy = greedy.expand(next_quantiles.shape[0], n_quantiles, 1)
    taken = next_quantiles.gather(dim=2, index=greedy).squeeze(dim=2)
    return rewards + (1 - dones) * discounts * taken


@pytest.mark.parametrize("batch_size,n_quantiles,n_actions", [(1, 5, 2), (100, 200, 6), (13, 7, 3)])
def test_qrdqn_target_quantiles_matches_upstream_expressions(batch_size, n_quantiles, n_actions):
    next_quantiles = rand(batch_size, n_quantiles, n_actions)
    rewards = rand(batch_size, 1)
    dones = rand_dones(batch_size).reshape(-1, 1)
    discounts = th.full((batch_size, 1), 0.99, dtype=F64)

    expected = qrdqn_target_reference(next_quantiles, rewards, dones, discounts, n_quantiles)
    got = qrdqn.target_quantiles(next_quantiles, rewards, dones, discounts)
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-13, atol=1e-15)


def test_qrdqn_target_quantiles_breaks_ties_like_argmax():
    """Two actions with identical mean Q: argmax keeps the lower index."""
    n_quantiles, n_actions = 4, 3
    next_quantiles = th.zeros(2, n_quantiles, n_actions, dtype=F64)
    next_quantiles[0, :, 2] = 1.0  # only the last action is best for row 0
    next_quantiles[1, :, 0] = 1.0  # row 1 ties between actions 0 and 1
    next_quantiles[1, :, 2] = 1.0
    rewards = th.zeros(2, 1, dtype=F64)
    dones = th.zeros(2, 1, dtype=F64)
    discounts = th.ones(2, 1, dtype=F64)
    expected = qrdqn_target_reference(next_quantiles, rewards, dones, discounts, n_quantiles)
    got = qrdqn.target_quantiles(next_quantiles, rewards, dones, discounts)
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-13, atol=1e-15)
    assert np.asarray(got)[1, 0] == 1.0


def test_qrdqn_train_loss_agrees_with_upstream(monkeypatch):
    """Per-call parity inside a real QRDQN.train, on the real QR-DQN data shapes."""
    from stable_baselines3.common.utils import set_random_seed
    from sb3_contrib.common.utils import quantile_huber_loss as reference_loss

    records = []
    set_random_seed(0)
    env = gym.make("CartPole-v1")
    model = qrdqn_upstream.QRDQN(
        "MlpPolicy", env, learning_starts=0, buffer_size=1000, verbose=0, seed=0,
        policy_kwargs=dict(net_arch=[16, 16], n_quantiles=10),
    )
    obs, _ = env.reset(seed=0)
    model.replay_buffer.seed = 0
    for _ in range(200):
        action = np.array([env.action_space.sample()])
        next_obs, reward, terminated, truncated, _ = env.step(int(action[0]))
        model.replay_buffer.add(obs, next_obs, action, np.array([reward]), np.array([terminated]), [{}])
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    model._setup_learn(0, None)
    monkeypatch.setattr(
        qrdqn_upstream,
        "quantile_huber_loss",
        _spy_on_loss(qrdqn_upstream, reference_loss, quantile_huber_loss, records, True, monkeypatch),
    )
    model.train(gradient_steps=3, batch_size=32)

    assert len(records) == 3
    recorded = float(model.logger.name_to_value["train/loss"])
    for current, target, cum_prob, sum_over, reference, ours in records:
        assert current.shape == (32, 10)
        assert target.shape == (32, 10)
        assert cum_prob is None
        assert sum_over is True
        assert ours == pytest.approx(reference, rel=1e-6, abs=1e-9)
    assert recorded == pytest.approx(np.mean([r[4] for r in records]), rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# sb3_contrib.crossq.crossq.CrossQ.train
# ---------------------------------------------------------------------------


def crossq_target_reference(all_q_values, next_log_prob, rewards, dones, discounts, ent_coef, batch_size):
    """The reduction half of upstream CrossQ.train, verbatim."""
    current_q_values, next_q_values = th.split(all_q_values, batch_size, dim=0)
    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    target_q_values = rewards + (1 - dones) * discounts * next_q_values
    return current_q_values.T[..., None], target_q_values


@pytest.mark.parametrize("batch_size,n_critics,ent_coef", [(1, 2, 0.2), (256, 4, 0.2), (9, 1, 1.5)])
def test_crossq_target_q_values_matches_upstream_expressions(batch_size, n_critics, ent_coef):
    all_q_values = rand(2 * batch_size, n_critics)
    next_log_prob = rand(batch_size)
    rewards = rand(batch_size, 1)
    dones = rand_dones(batch_size).reshape(-1, 1)
    discounts = th.full((batch_size, 1), 0.99, dtype=F64)

    current_q, target_q = crossq_target_reference(
        all_q_values, next_log_prob, rewards, dones, discounts, ent_coef, batch_size
    )
    got = crossq.target_q_values(all_q_values, next_log_prob, rewards, dones, discounts, ent_coef)
    assert np.allclose(np.asarray(got), np.asarray(target_q).reshape(-1), rtol=1e-13, atol=1e-14)

    ref_loss = 0.5 * sum(th.nn.functional.mse_loss(cq, target_q.detach()) for cq in current_q)
    got_loss = crossq.critic_loss(current_q.squeeze(-1), target_q)
    assert float(got_loss) == pytest.approx(float(ref_loss), rel=1e-12, abs=1e-14)


def test_crossq_target_q_values_with_a_real_critic():
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from sb3_contrib.crossq.policies import CrossQCritic

    obs_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
    critic = CrossQCritic(
        obs_space,
        spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        net_arch=[32, 32],
        features_extractor=FlattenExtractor(obs_space),
        features_dim=3,
        n_critics=3,
    ).double()
    # the joint forward pass stacks obs and next_obs along the batch axis
    obs = th.as_tensor(RNG.normal(size=(24, 3)), dtype=F64)
    action = th.as_tensor(RNG.normal(size=(24, 1)), dtype=F64)
    with th.no_grad():
        all_q = th.cat(critic(obs, action), dim=1)
    assert tuple(all_q.shape) == (24, 3)
    rewards = rand(12, 1)
    dones = th.zeros(12, 1, dtype=F64)
    discounts = th.full((12, 1), 0.99, dtype=F64)
    next_log_prob = rand(12)
    _, target_q = crossq_target_reference(all_q, next_log_prob, rewards, dones, discounts, 0.2, 12)
    got = crossq.target_q_values(all_q, next_log_prob, rewards, dones, discounts, 0.2)
    assert th.allclose(th.as_tensor(got), target_q.reshape(-1), rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# sb3_contrib.ars.ars.ARS._do_one_update
# ---------------------------------------------------------------------------


def ars_update_reference(plus_returns, minus_returns, deltas, weights, learning_rate, n_top):
    """The tail of upstream ARS._do_one_update, verbatim."""
    top_returns, _ = th.max(th.vstack((plus_returns, minus_returns)), dim=0)
    top_idx = th.argsort(top_returns, descending=True)[:n_top]
    sel_plus = plus_returns[top_idx]
    sel_minus = minus_returns[top_idx]
    sel_deltas = deltas[top_idx]
    return_std = th.cat([sel_plus, sel_minus]).std()
    step_size = learning_rate / (n_top * return_std + 1e-6)
    return top_idx, sel_plus, sel_minus, sel_deltas, return_std, step_size, \
        weights + step_size * ((sel_plus - sel_minus) @ sel_deltas)


@pytest.mark.parametrize("n_delta,n_top,n_params", [(4, 2, 50), (20, 10, 5000), (7, 7, 13)])
def test_ars_update_matches_upstream_expressions(n_delta, n_top, n_params):
    plus = rand(n_delta)
    minus = rand(n_delta)
    deltas = rand(n_delta, n_params)
    weights = rand(n_params)
    learning_rate = 0.01

    top_idx, sel_plus, sel_minus, sel_deltas, return_std, step_size, expected_weights = ars_update_reference(
        plus, minus, deltas, weights, learning_rate, n_top
    )

    got_idx, got_std = ars.top_returns(plus, minus, n_top)
    assert set(np.asarray(got_idx).tolist()) == set(np.asarray(top_idx).tolist())
    assert got_std == pytest.approx(float(return_std), rel=1e-12)

    g_plus, g_minus, g_deltas = ars.gather(plus, minus, deltas, got_idx)
    assert np.array_equal(np.asarray(g_plus), np.asarray(sel_plus))
    assert np.array_equal(np.asarray(g_minus), np.asarray(sel_minus))
    assert np.array_equal(np.asarray(g_deltas), np.asarray(sel_deltas))

    got_weights, got_step = ars.update_weights(weights, g_plus, g_minus, g_deltas, got_std, learning_rate)
    assert got_step == pytest.approx(float(step_size), rel=1e-12)
    assert np.allclose(np.asarray(got_weights), np.asarray(expected_weights), rtol=1e-10, atol=1e-14)


def test_ars_update_moves_the_weights_uphill():
    """A perfect candidate pair must move the weights along +delta."""
    n_delta, n_top, n_params = 4, 2, 9
    plus = th.as_tensor([3.0, 2.0, 0.0, 0.0], dtype=F64)
    minus = th.as_tensor([0.0, 0.0, 1.0, 2.0], dtype=F64)
    deltas = th.eye(n_params, dtype=F64)[th.arange(n_delta)]
    weights = th.zeros(n_params, dtype=F64)
    idx, std = ars.top_returns(plus, minus, n_top)
    g_plus, g_minus, g_deltas = ars.gather(plus, minus, deltas, idx)
    new_weights, step = ars.update_weights(weights, g_plus, g_minus, g_deltas, std, 0.1)
    assert th.as_tensor(new_weights)[:2].min() > 0.0
    assert step > 0.0


# ---------------------------------------------------------------------------
# sb3_contrib.common.torch_layers.BatchRenorm1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "training,steps,warmup,affine",
    [
        (True, 0, 100_000, True),
        (True, 200_000, 100_000, True),
        (True, 200_000, 100_000, False),
        (False, 200_000, 100_000, True),
    ],
)
def test_batch_renorm_matches_upstream(training, steps, warmup, affine):
    batch_size, num_features = 64, 8
    x = th.as_tensor(RNG.normal(size=(batch_size, num_features)), dtype=F64) * 3.0 + 1.5
    scale = th.as_tensor(RNG.normal(size=num_features) + 2.0, dtype=F64)
    bias = th.as_tensor(RNG.normal(size=num_features), dtype=F64)
    ra_mean = th.as_tensor(RNG.normal(size=num_features), dtype=F64)
    ra_var = th.as_tensor(np.abs(RNG.normal(size=num_features)) + 0.5, dtype=F64)

    ref = UpstreamBatchRenorm1d(num_features, eps=0.001, momentum=0.01, affine=affine, warmup_steps=warmup)
    # upstream stores ra_mean/ra_var as float32 buffers and scale/bias as
    # float32 parameters, which silently rounds the eval-mode normalisation.
    # .double() puts both sides in the dtype the kernel actually uses.
    ref = ref.double()
    with th.no_grad():
        ref.ra_mean.copy_(ra_mean)
        ref.ra_var.copy_(ra_var)
        ref.scale.copy_(scale)
        ref.bias.copy_(bias)
        ref.steps.copy_(th.as_tensor(steps, dtype=th.int64))
    ref.train(training)
    expected = ref(x)

    ours = BatchRenorm1d(num_features, eps=0.001, momentum=0.01, affine=affine, warmup_steps=warmup)
    ours.ra_mean[:] = ra_mean.numpy()
    ours.ra_var[:] = ra_var.numpy()
    ours.scale[:] = scale.numpy()
    ours.bias[:] = bias.numpy()
    ours.steps = steps
    ours.training = training
    got = ours(x)

    assert np.allclose(np.asarray(got), expected.detach().numpy(), rtol=1e-12, atol=1e-14)
    assert ours.steps == int(ref.steps)
    assert np.allclose(ours.ra_mean, ref.ra_mean.detach().numpy(), rtol=1e-12, atol=1e-14)
    assert np.allclose(ours.ra_var, ref.ra_var.detach().numpy(), rtol=1e-12, atol=1e-14)


def test_batch_renorm_step_accumulates_running_statistics():
    """`self.steps` counts calls in training mode and is frozen in eval."""
    layer = BatchRenorm1d(3, warmup_steps=0)
    x = th.as_tensor(RNG.normal(size=(8, 3)), dtype=F64)
    for expected_steps in (1, 2, 3):
        layer.forward(x)
        assert layer.steps == expected_steps
    layer.training = False
    layer.forward(x)
    assert layer.steps == 3


def test_batch_renorm_1d_input_is_a_documented_divergence():
    """Upstream broadcasts a 1-D input through `custom_mean[None]`; the kernel
    has no broadcasting and rejects it. See README, "Known divergences"."""
    assert UpstreamBatchRenorm1d(4)(th.zeros(4, dtype=F64)).shape == (4,)
    with pytest.raises(ValueError, match="Expected 2D input"):
        BatchRenorm1d(4).forward(th.zeros(4, dtype=F64))


# ---------------------------------------------------------------------------
# sb3_contrib.common.maskable.distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_masks", [False, True])
def test_maskable_entropy_matches_upstream(use_masks):
    n = 17
    logits = th.as_tensor(RNG.normal(size=(n,)), dtype=F64)
    masks = th.as_tensor(RNG.integers(0, 2, size=(n,)), dtype=th.bool) if use_masks else None
    expected = float(UpstreamMaskableCategorical(logits=logits, masks=masks).entropy())
    got = m.entropy(logits, masks)
    assert float(got) == pytest.approx(expected, rel=1e-10, abs=1e-13)


@pytest.mark.parametrize("n_legal", [1, 3, 17])
def test_maskable_entropy_with_every_mask_pattern(n_legal):
    """Explicitly set at least one legal action, as MaskablePPO does."""
    n = 17
    logits = th.as_tensor(RNG.normal(size=(n,)), dtype=F64)
    masks = th.zeros(n, dtype=th.bool)
    masks[th.as_tensor(RNG.choice(n, size=n_legal, replace=False))] = True
    expected = float(UpstreamMaskableCategorical(logits=logits, masks=masks).entropy())
    got = m.entropy(logits, masks)
    assert float(got) == pytest.approx(expected, rel=1e-10, abs=1e-13)


def test_apply_masking_matches_upstream():
    logits = th.as_tensor(RNG.normal(size=(23,)), dtype=F64)
    masks = th.as_tensor(RNG.integers(0, 2, size=(23,)), dtype=th.bool)
    masks[0] = True
    expected = UpstreamMaskableCategorical(logits=logits, masks=masks).logits
    got = m.apply_masking(logits, masks)
    assert np.allclose(np.asarray(got), expected.detach().numpy(), rtol=1e-10, atol=1e-10)


def test_apply_masking_removes_the_mask_when_none():
    logits = th.as_tensor(RNG.normal(size=(11,)), dtype=F64)
    expected = UpstreamMaskableCategorical(logits=logits, masks=None).logits
    got = m.apply_masking(logits, np.ones(11, dtype=np.int64))
    assert np.allclose(np.asarray(got), expected.detach().numpy(), rtol=1e-10, atol=1e-10)


def test_maskable_categorical_probs_match_upstream():
    logits = th.as_tensor(RNG.normal(size=(11,)), dtype=F64)
    masks = th.as_tensor(RNG.integers(0, 2, size=(11,)), dtype=th.bool)
    masks[0] = True
    ref = UpstreamMaskableCategorical(logits=logits, masks=masks)
    ours = m.MaskableCategorical(logits=logits, masks=masks)
    assert np.allclose(ours.probs, ref.probs.numpy(), rtol=1e-10, atol=1e-12)
    assert float(ours.entropy()) == pytest.approx(float(ref.entropy()), rel=1e-10)


def test_maskable_categorical_from_probs_matches_upstream():
    probs = th.as_tensor(RNG.uniform(size=(9,)), dtype=F64)
    probs = probs / probs.sum()
    ref = UpstreamMaskableCategorical(probs=probs)
    ours = m.MaskableCategorical(probs=probs)
    assert np.allclose(ours.probs, ref.probs.numpy(), rtol=1e-10, atol=1e-12)
    assert float(ours.entropy()) == pytest.approx(float(ref.entropy()), rel=1e-10)


def test_maskable_categorical_preserves_original_logits():
    logits = th.as_tensor(RNG.normal(size=(9,)), dtype=F64)
    masks = th.as_tensor(RNG.integers(0, 2, size=(9,)), dtype=th.bool)
    masks[0] = True
    cat = m.MaskableCategorical(logits=logits, masks=masks)
    assert not np.array_equal(cat.logits, cat._original_logits)
    cat.apply_masking(None)
    assert np.allclose(cat.logits, cat._original_logits, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# SIMD tails, block boundaries and fused passes
#
# Every kernel now runs its contiguous axis W doubles at a time with a scalar
# tail, and several walk memory in blocks. The widths below are chosen so each
# one lands on a different side of a vector boundary (W == 4), of the per-row
# sort's insertion/heapsort threshold, and of the ARS accumulator block.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_target", [1, 2, 3, 4, 5, 7, 8, 9, 17])
def test_quantile_huber_loss_vector_tail(n_target):
    """n_target_quantiles either side of, and well inside, one SIMD block."""
    current = rand(9, 6)
    target = rand(9, n_target)
    expected = upstream_quantile_huber_loss(current, target, sum_over_quantiles=True)
    got = quantile_huber_loss(current, target, sum_over_quantiles=True)
    assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("n_quantiles", [1, 2, 3, 5])
def test_quantile_huber_loss_tail_covers_the_current_quantile_axis(n_quantiles):
    current = rand(5, n_quantiles)
    target = rand(5, n_quantiles + 1)
    for sum_over in (True, False):
        expected = upstream_quantile_huber_loss(current, target, sum_over_quantiles=sum_over)
        got = quantile_huber_loss(current, target, sum_over_quantiles=sum_over)
        assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("num_features", [1, 2, 3, 4, 5, 7, 9, 13])
def test_batch_renorm_vector_tail(num_features):
    """`num_features` either side of a SIMD block, in both modes."""
    batch_size = 33
    x = th.as_tensor(RNG.normal(size=(batch_size, num_features)), dtype=F64) * 3.0 + 1.5
    scale = th.as_tensor(RNG.normal(size=num_features) + 2.0, dtype=F64)
    bias = th.as_tensor(RNG.normal(size=num_features), dtype=F64)
    ra_mean = th.as_tensor(RNG.normal(size=num_features), dtype=F64)
    ra_var = th.as_tensor(np.abs(RNG.normal(size=num_features)) + 0.5, dtype=F64)
    for training in (True, False):
        ref = UpstreamBatchRenorm1d(num_features, eps=0.001, momentum=0.01, warmup_steps=0).double()
        with th.no_grad():
            ref.ra_mean.copy_(ra_mean)
            ref.ra_var.copy_(ra_var)
            ref.scale.copy_(scale)
            ref.bias.copy_(bias)
        ref.train(training)
        expected = ref(x)

        ours = BatchRenorm1d(num_features, eps=0.001, momentum=0.01, warmup_steps=0)
        ours.ra_mean[:] = ra_mean.numpy()
        ours.ra_var[:] = ra_var.numpy()
        ours.scale[:] = scale.numpy()
        ours.bias[:] = bias.numpy()
        ours.training = training
        got = ours(x)
        assert np.allclose(np.asarray(got), expected.detach().numpy(), rtol=1e-12, atol=1e-14)
        assert np.allclose(ours.ra_mean, ref.ra_mean.detach().numpy(), rtol=1e-12, atol=1e-14)
        assert np.allclose(ours.ra_var, ref.ra_var.detach().numpy(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n_params", [1, 3, 4, 5, 4095, 4096, 4097, 8193])
def test_ars_update_weight_block_boundary(n_params):
    """`n_params` either side of the accumulator block and of a SIMD block."""
    n_delta, n_top = 9, 3
    plus = rand(n_delta)
    minus = rand(n_delta)
    deltas = rand(n_delta, n_params)
    weights = rand(n_params)
    learning_rate = 0.01

    top_idx, sel_plus, sel_minus, sel_deltas, return_std, step_size, expected = ars_update_reference(
        plus, minus, deltas, weights, learning_rate, n_top
    )
    got_idx, got_std = ars.top_returns(plus, minus, n_top)
    g_plus, g_minus, g_deltas = ars.gather(plus, minus, deltas, got_idx)
    got_weights, got_step = ars.update_weights(weights, g_plus, g_minus, g_deltas, got_std, learning_rate)
    assert got_step == pytest.approx(float(step_size), rel=1e-12)
    assert np.allclose(np.asarray(got_weights), np.asarray(expected), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("n_actions", [1, 2, 3, 4, 5, 7, 9])
def test_qrdqn_target_vector_tail(n_actions):
    """`n_actions` either side of a SIMD block in the per-action mean."""
    batch_size, n_quantiles = 6, 9
    next_quantiles = rand(batch_size, n_quantiles, n_actions)
    rewards = rand(batch_size, 1)
    dones = rand_dones(batch_size).reshape(-1, 1)
    discounts = th.full((batch_size, 1), 0.99, dtype=F64)

    expected = qrdqn_target_reference(next_quantiles, rewards, dones, discounts, n_quantiles)
    got = qrdqn.target_quantiles(next_quantiles, rewards, dones, discounts)
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("n_critics,n_quantiles", [(1, 1), (1, 127), (1, 128), (1, 129), (3, 43), (2, 64)])
def test_tqc_target_sorts_across_the_threshold(n_critics, n_quantiles):
    """The per-row sort switches algorithm at 128; both sides must match."""
    batch_size = 5
    drop_per_net = 0 if n_quantiles == 1 else 1
    next_quantiles = rand(batch_size, n_critics, n_quantiles)
    next_log_prob = rand(batch_size)
    rewards = rand(batch_size, 1)
    dones = th.zeros(batch_size, 1, dtype=F64)
    discounts = th.full((batch_size, 1), 0.99, dtype=F64)

    n_target = n_critics * (n_quantiles - drop_per_net)
    expected = tqc_target_reference(
        next_quantiles, next_log_prob, rewards, dones, discounts, 0.2, n_target
    )
    got = tqc.target_quantiles(
        next_quantiles, next_log_prob, rewards, dones, discounts, 0.2, drop_per_net, n_critics, n_quantiles
    )
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 17])
def test_conjugate_gradient_matrix_vector_tail(n):
    """The matvec/dot/axpy fusion with n either side of a SIMD block."""
    matrix = th.as_tensor(
        np.diag(np.linspace(1.0, 4.0, n)) + RNG.normal(scale=0.05, size=(n, n)), dtype=F64
    )
    matrix = matrix @ matrix.T + n * th.eye(n, dtype=F64)
    b = rand(n)
    th.manual_seed(11)
    expected = upstream_conjugate_gradient_solver(lambda v: matrix @ v, b, max_iter=10, residual_tol=1e-10)
    th.manual_seed(11)
    got = conjugate_gradient_solver(lambda v: matrix @ v, b, max_iter=10, residual_tol=1e-10)
    assert th.allclose(th.as_tensor(got), expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("n_critics", [1, 3, 5, 8])
def test_crossq_critic_loss_vector_tail(n_critics):
    """The squared-error reduction with a batch that is not a multiple of W."""
    batch_size = 11
    current = rand(n_critics, batch_size)
    target = rand(batch_size)
    expected = 0.5 * sum(
        float(th.mean((current[c] - target) ** 2)) for c in range(n_critics)
    )
    got = crossq.critic_loss(current, target)
    assert float(got) == pytest.approx(expected, rel=1e-12, abs=1e-14)


def _gpu_enabled(monkeypatch):
    """Force the device path on, or skip if this machine has no usable GPU."""
    from mojo_sb3_contrib.common import utils

    if not utils._gpu_usable():
        pytest.skip("no GPU with at least 4000 MiB free")
    monkeypatch.setattr(utils, "_GPU_MIN_TERMS", 0)
    return utils


@pytest.mark.parametrize(
    "batch_size,n_critics,n_quantiles,n_target",
    [
        (256, 1, 200, 200),   # the QR-DQN shape the device path exists for
        (1024, 1, 200, 200),
        (256, 2, 25, 50),     # the TQC shape
        (7, 1, 5, 5),         # fewer target elements than one tile
        (5, 1, 3, 7),         # a scalar tail inside the last tile
        (3, 1, 9, 13),        # a partial trailing tile
        (1, 1, 1, 1),
        (2, 3, 6, 2),         # 3-D layout, several critics
    ],
)
def test_quantile_huber_loss_gpu_path_matches_upstream(
    monkeypatch, batch_size, n_critics, n_quantiles, n_target
):
    """The device kernel is held to the same upstream reference and tolerance."""
    _gpu_enabled(monkeypatch)
    if n_critics == 1:
        current = rand(batch_size, n_quantiles)
        target = rand(batch_size, n_target)
    else:
        current = rand(batch_size, n_critics, n_quantiles)
        target = rand(batch_size, 1, n_target)
    for sum_over in (True, False):
        expected = upstream_quantile_huber_loss(
            current, target, sum_over_quantiles=sum_over
        )
        got = quantile_huber_loss(current, target, sum_over_quantiles=sum_over)
        assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)


def test_quantile_huber_loss_gpu_and_cpu_agree(monkeypatch):
    """The term-count threshold is the only thing that decides, so both agree."""
    utils = _gpu_enabled(monkeypatch)
    current = rand(96, 1, 40)
    target = rand(96, 1, 40)
    on_gpu = quantile_huber_loss(current, target, sum_over_quantiles=True)
    monkeypatch.setattr(utils, "_GPU_MIN_TERMS", 1 << 62)
    on_cpu = quantile_huber_loss(current, target, sum_over_quantiles=True)
    assert float(on_gpu) == pytest.approx(float(on_cpu), rel=1e-12, abs=1e-14)


def test_quantile_huber_loss_skips_the_gpu_for_a_caller_cum_prob(monkeypatch):
    """A caller-supplied `cum_prob` has no device path and must still be right."""
    _gpu_enabled(monkeypatch)
    batch_size, n_quantiles = 16, 11
    current = rand(batch_size, n_quantiles)
    target = rand(batch_size, n_quantiles)
    cum_prob = th.as_tensor(RNG.uniform(size=(batch_size, 1, n_quantiles)), dtype=F64)
    expected = upstream_quantile_huber_loss(current, target, cum_prob=cum_prob)
    got = quantile_huber_loss(current, target, cum_prob=cum_prob)
    assert float(got) == pytest.approx(float(expected), rel=1e-10, abs=1e-12)
