"""mojo-sb3-contrib vs upstream sb3-contrib on the same data.

Every row is the same arithmetic on the same inputs, timed with the same
harness. Run it through `pixi run bench`, which holds a machine-wide lock so a
concurrent factory job cannot distort the numbers.
"""

import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_sb3_contrib as m  # noqa: E402
from mojo_sb3_contrib import ars, crossq, qrdqn, tqc  # noqa: E402
from mojo_sb3_contrib.common.torch_layers import BatchRenorm1d  # noqa: E402
from mojo_sb3_contrib.common.utils import (  # noqa: E402
    conjugate_gradient_solver,
    quantile_huber_loss,
)

from sb3_contrib.common.torch_layers import BatchRenorm1d as UpstreamBatchRenorm1d  # noqa: E402
from sb3_contrib.common.utils import (  # noqa: E402
    conjugate_gradient_solver as upstream_cg,
    quantile_huber_loss as upstream_qhl,
)

F64 = th.float64
RNG = np.random.default_rng(0)


def timeit(fn, reps=5, warmup=1):
    """Best-of-`reps` wall time, plus whatever the call returned."""
    for _ in range(warmup):
        out = fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def table(rows):
    print("| kernel | shape | mojo | upstream | speedup | note |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for name, shape, mojo_s, ref_s, note in rows:
        speed = ref_s / mojo_s if mojo_s else float("inf")
        print(f"| {name} | {shape} | {mojo_s * 1e3:.3f} ms | {ref_s * 1e3:.3f} ms | "
              f"{speed:.2f}x | {note} |")


def rand(*shape):
    return th.as_tensor(RNG.normal(size=shape), dtype=F64)


def bench_quantile_huber_loss():
    rows = []
    cases = [
        ("QR-DQN  batch 256, n 200", (256, 200), (256, 200), True),
        ("QR-DQN  batch 1024, n 200", (1024, 200), (1024, 200), True),
        ("TQC     batch 256, 2x50", (256, 2, 25), (256, 1, 50), False),
        ("TQC     batch 1024, 2x50", (1024, 2, 25), (1024, 1, 50), False),
    ]
    for name, cur_shape, tgt_shape, sum_over in cases:
        cur = rand(*cur_shape)
        tgt = rand(*tgt_shape)
        mojo_s, mojo_out = timeit(lambda: quantile_huber_loss(cur, tgt, sum_over_quantiles=sum_over))
        ref_s, ref_out = timeit(lambda: upstream_qhl(cur, tgt, sum_over_quantiles=sum_over))
        assert abs(float(mojo_out) - float(ref_out)) <= 1e-9 * max(1.0, abs(float(ref_out)))
        rows.append((name, f"{cur_shape} x {tgt_shape[-1]}", mojo_s, ref_s, "SIMD over the target axis; never materialised"))
    return rows

def bench_quantile_huber_loss_gpu():
    """Device against CPU, same inputs, with the term-count threshold forced
    both ways. Prints nothing and returns no rows when there is no usable
    device, so the table is absent rather than fabricated on a CPU-only box."""
    from mojo_sb3_contrib.common import utils

    if not utils._gpu_usable():
        print("(no GPU with at least 4000 MiB free; the device table is skipped)")
        return []
    saved = utils._GPU_MIN_TERMS
    rows = []
    cases = [
        ("QR-DQN  batch 256, n 200", (256, 200), (256, 200), True),
        ("QR-DQN  batch 1024, n 200", (1024, 200), (1024, 200), True),
        ("TQC     batch 256, 2x50", (256, 2, 25), (256, 1, 50), False),
        ("TQC     batch 1024, 2x50", (1024, 2, 25), (1024, 1, 50), False),
        ("QR-DQN  batch 64, n 50", (64, 50), (64, 50), True),
        ("QR-DQN  batch 4, n 10", (4, 10), (4, 10), True),
    ]
    try:
        for name, cur_shape, tgt_shape, sum_over in cases:
            cur = rand(*cur_shape)
            tgt = rand(*tgt_shape)
            utils._GPU_MIN_TERMS = 1 << 62
            cpu_s, cpu_out = timeit(
                lambda: quantile_huber_loss(cur, tgt, sum_over_quantiles=sum_over)
            )
            utils._GPU_MIN_TERMS = 0
            gpu_s, gpu_out = timeit(
                lambda: quantile_huber_loss(cur, tgt, sum_over_quantiles=sum_over)
            )
            assert abs(float(gpu_out) - float(cpu_out)) <= 1e-9 * max(
                1.0, abs(float(cpu_out))
            ), f"device and CPU disagree on {name}"
            rows.append((name, f"{cur_shape} x {tgt_shape[-1]}", cpu_s, gpu_s))
    finally:
        utils._GPU_MIN_TERMS = saved
    return rows


def bench_tqc_target():
    rows = []
    for batch_size, n_critics, n_quantiles in [(256, 2, 25), (1024, 2, 25), (256, 5, 50)]:
        nq = rand(batch_size, n_critics, n_quantiles)
        lp = rand(batch_size)
        rew = rand(batch_size, 1)
        done = th.zeros(batch_size, 1, dtype=F64)
        disc = th.full((batch_size, 1), 0.99, dtype=F64)
        drop = 2
        n_target = n_critics * (n_quantiles - drop)

        def reference():
            flat, _ = th.sort(nq.reshape(batch_size, -1))
            flat = flat[:, :n_target]
            target = flat - 0.2 * lp.reshape(-1, 1)
            return rew + (1 - done) * disc * target

        mojo_s, mojo_out = timeit(
            lambda: tqc.target_quantiles(nq, lp, rew, done, disc, 0.2, drop, n_critics, n_quantiles)
        )
        ref_s, ref_out = timeit(reference)
        assert np.allclose(np.asarray(mojo_out), ref_out.numpy(), rtol=1e-12, atol=1e-12)
        rows.append(("TQC target (sort + drop)", f"batch {batch_size}, {n_critics}x{n_quantiles}",
                     mojo_s, ref_s, "per-row insertion sort under 128, heapsort above"))
    return rows


def bench_qrdqn_target():
    rows = []
    for batch_size, n_quantiles, n_actions in [(256, 200, 6), (1024, 200, 10)]:
        nq = rand(batch_size, n_quantiles, n_actions)
        rew = rand(batch_size, 1)
        done = th.zeros(batch_size, 1, dtype=F64)
        disc = th.full((batch_size, 1), 0.99, dtype=F64)

        def reference():
            greedy = nq.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            greedy = greedy.expand(batch_size, n_quantiles, 1)
            return rew + (1 - done) * disc * nq.gather(dim=2, index=greedy).squeeze(dim=2)

        mojo_s, mojo_out = timeit(lambda: qrdqn.target_quantiles(nq, rew, done, disc))
        ref_s, ref_out = timeit(reference)
        assert np.allclose(np.asarray(mojo_out), ref_out.numpy(), rtol=1e-12, atol=1e-13)
        rows.append(("QR-DQN target (mean+argmax)", f"batch {batch_size}, {n_quantiles}x{n_actions}",
                     mojo_s, ref_s, "row-major mean, gather fused into the TD target"))
    return rows


def bench_crossq():
    rows = []
    for batch_size, n_critics in [(256, 2), (1024, 4)]:
        allq = rand(2 * batch_size, n_critics)
        lp = rand(batch_size)
        rew = rand(batch_size, 1)
        done = th.zeros(batch_size, 1, dtype=F64)
        disc = th.full((batch_size, 1), 0.99, dtype=F64)

        def reference():
            _, nxt = th.split(allq, batch_size, dim=0)
            nxt, _ = th.min(nxt, dim=1, keepdim=True)
            nxt = nxt - 0.2 * lp.reshape(-1, 1)
            return rew + (1 - done) * disc * nxt

        mojo_s, mojo_out = timeit(lambda: crossq.target_q_values(allq, lp, rew, done, disc, 0.2))
        ref_s, ref_out = timeit(reference)
        assert np.allclose(np.asarray(mojo_out), ref_out.numpy().reshape(-1), rtol=1e-12, atol=1e-13)
        rows.append(("CrossQ target Q (min + td)", f"batch {batch_size}, {n_critics} critics",
                     mojo_s, ref_s, "memory bound; both are launch-latency dominated"))
    return rows


def bench_batch_renorm():
    rows = []
    batch_size, num_features = 4096, 256
    x = rand(batch_size, num_features) * 3.0 + 1.5
    for training in (True, False):
        ref = UpstreamBatchRenorm1d(num_features, warmup_steps=0).double()
        ref.train(training)
        state = (
            ref.ra_mean.detach().numpy().copy(),
            ref.ra_var.detach().numpy().copy(),
            ref.scale.detach().numpy().copy(),
            ref.bias.detach().numpy().copy(),
        )
        ours = BatchRenorm1d(num_features, warmup_steps=0)
        ours.training = training

        def seed(layer):
            layer.ra_mean[:] = state[0]
            layer.ra_var[:] = state[1]
            layer.scale[:] = state[2]
            layer.bias[:] = state[3]

        def run_ours():
            seed(ours)
            return ours(x)

        def run_ref():
            with th.no_grad():
                ref.ra_mean.copy_(th.as_tensor(state[0]))
                ref.ra_var.copy_(th.as_tensor(state[1]))
            return ref(x)

        mojo_s, mojo_out = timeit(run_ours)
        ref_s, ref_out = timeit(run_ref)
        assert np.allclose(np.asarray(mojo_out), ref_out.detach().numpy(), rtol=1e-9, atol=1e-10)
        rows.append((f"BatchRenorm1d ({'train' if training else 'eval'})",
                     f"{batch_size} x {num_features}", mojo_s, ref_s,
                     "row-major passes; per-feature scale folded into one multiply-add"))
    return rows


def bench_ars():
    rows = []
    for n_delta, n_top, n_params in [(32, 8, 5_000), (64, 16, 50_000)]:
        plus = rand(n_delta)
        minus = rand(n_delta)
        deltas = rand(n_delta, n_params)
        weights = rand(n_params)
        learning_rate = 0.01

        def reference():
            top, _ = th.max(th.vstack((plus, minus)), dim=0)
            idx = th.argsort(top, descending=True)[:n_top]
            sp, sm, sd = plus[idx], minus[idx], deltas[idx]
            std = th.cat([sp, sm]).std()
            step = learning_rate / (n_top * std + 1e-6)
            return weights + step * ((sp - sm) @ sd), float(step)

        def run_ours():
            idx, std = ars.top_returns(plus, minus, n_top)
            gp, gm, gd = ars.gather(plus, minus, deltas, idx)
            return ars.update_weights(weights, gp, gm, gd, std, learning_rate)

        mojo_s, mojo_out = timeit(run_ours)
        ref_s, ref_out = timeit(reference)
        assert np.allclose(np.asarray(mojo_out[0]), ref_out[0].numpy(), rtol=1e-9, atol=1e-12)
        rows.append(("ARS top-k + update", f"n_delta {n_delta}, n_top {n_top}, {n_params} params",
                     mojo_s, ref_s, "L1-blocked accumulation; torch hits BLAS for the matvec"))
    return rows


def bench_cg():
    rows = []
    for n in (64, 256):
        matrix = th.as_tensor(
            np.diag(np.linspace(1.0, 4.0, n)) + RNG.normal(scale=0.05, size=(n, n)), dtype=F64
        )
        matrix = matrix @ matrix.T + n * th.eye(n, dtype=F64)
        b = rand(n)
        th.manual_seed(0)
        ref_s, _ = timeit(lambda: upstream_cg(lambda v: matrix @ v, b, max_iter=10, residual_tol=1e-10))
        # upstream never materialises A; the port does, so time the kernel alone
        # against ten torch matvecs to compare like with like.
        x = np.ascontiguousarray(1e-4 * RNG.standard_normal(n))
        from mojo_sb3_contrib._lib import lib, scratch

        work = scratch(3 * n)
        m_data = matrix.numpy()
        k_s, _ = timeit(
            lambda: lib().sb3c_conjugate_gradient_solver(
                m_data.ctypes.data, b.numpy().ctypes.data, x.ctypes.data, work.ctypes.data,
                n, 10, 1e-10,
            )
        )
        th.manual_seed(0)
        w_s, _ = timeit(lambda: conjugate_gradient_solver(lambda v: matrix @ v, b, 10, 1e-10))
        rows.append(("ARS CG, kernel only", f"n {n}, 10 iterations", k_s, ref_s,
                     "upstream gets a closure and never builds A"))
        rows.append(("ARS CG, full wrapper", f"n {n}, 10 iterations", w_s, ref_s,
                     "wrapper materialises A with n matvecs first"))
    return rows


def main():
    torch_threads = th.get_num_threads()
    print(f"machine: {cpu_name()}")
    print(f"torch {th.__version__}, {torch_threads} threads, numpy {np.__version__}")
    print(f"python {platform.python_version()}, {platform.system()} {platform.machine()}")
    print()

    rows = []
    rows += bench_quantile_huber_loss()
    print("### quantile_huber_loss")
    table(rows)

    rows = bench_quantile_huber_loss_gpu()
    if rows:
        print("### quantile_huber_loss, device against CPU")
        print("| kernel | shape | CPU | GPU | device speedup |")
        print("| --- | --- | ---: | ---: | ---: |")
        for name, shape, cpu_s, gpu_s in rows:
            print(f"| {name} | {shape} | {cpu_s * 1e3:.3f} ms | {gpu_s * 1e3:.3f} ms | "
                  f"{cpu_s / gpu_s:.2f}x |")
        print()

    rows = bench_tqc_target()
    print("### TQC target quantiles")
    table(rows)
    print()

    rows = bench_qrdqn_target()
    print("### QR-DQN target quantiles")
    table(rows)
    print()

    rows = bench_crossq()
    print("### CrossQ target Q values")
    table(rows)
    print()

    rows = bench_batch_renorm()
    print("### BatchRenorm1d")
    table(rows)
    print()

    rows = bench_ars()
    print("### ARS update")
    table(rows)
    print()

    rows = bench_cg()
    print("### ARS conjugate gradient")
    table(rows)


if __name__ == "__main__":
    main()
