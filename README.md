# mojo-sb3-contrib

The compute-bound kernels of [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
(Stable-Baselines3-Contrib), rewritten in Mojo, with a Python API that keeps
upstream's names, argument order and defaults.

The upstream package *is* installable — this repo installs `sb3-contrib` 2.9.0
from PyPI as a test dependency and every test below compares against it. There
is no reimplementation of the "reference"; the reference is upstream's own code.

## What this is

sb3-contrib is mostly PyTorch plumbing: policies, optimizers, replay buffers,
rollout collection. What is worth rewriting are the handful of numeric kernels
that every gradient step of the covered algorithms runs over the whole batch:
the pairwise quantile-regression loss,
the target-quantile truncation TQC applies to its critics, the BatchRenorm
forward pass, and the ARS update. Those are what lives in `src/ported.mojo`.

`src/ported.mojo` is meant to be read next to the upstream file it came from.
The functions appear in upstream's module order and, within a module, in
upstream's source order. Upstream's branch structure and arithmetic are
intact; what the optimisation pass changed is *the order memory is walked in*
and the fact that the innermost contiguous axis runs `simd_width_of[float64]`
doubles at a time. Both are noted per function, and the tests pin the
arithmetic against upstream at the tolerances upstream itself meets.

## What the optimisation pass did

The kernels keep upstream's arithmetic and summation order. Four things
changed around it, in the order they were worth trying:

1. **Walk memory the way it is laid out.** `BatchRenorm` computed
   `x.mean(dim=0)` by, for each feature, striding down the rows of a row-major
   array - every cache line touched `num_features / 8` times. The QR-DQN
   target walked its `(batch, n_quantiles, n_actions)` block with a stride of
   `n_actions` for the same reason. Both now walk whole rows. Each output
   element is still produced by the same sequence of floating-point
   operations on the same inputs, so the results are bit-identical; only the
   addresses change.
2. **Hoist what does not change.** The ARS weight update reduced
   `(plus - minus) @ deltas` parameter-major and blocked the accumulator to
   L1, so each `deltas` element is read from memory once instead of once per
   noise direction. The quantile-Huber loss materialises upstream's two
   `th.float` weights per current quantile once instead of recomputing a
   division for every pairwise delta. The TQC and QR-DQN targets fuse their
   gather into the TD update and hoist `(1 - dones) * discounts`.
3. **Vectorise the contiguous axis, with a scalar tail.** Every kernel's
   innermost loop runs `p.unsafe_load[width=W](i)` with a scalar remainder
   loop behind it. The two Huber branches and the `(pairwise_delta < 0)`
   indicator are computed branchlessly, exactly: with `a = |d|`,
   `(a * min(a, 1) + max(a - 1, 0)) * 0.5` is upstream's `th.where`, and
   `copysign` turns the sign bit straight into the `0.0`/`1.0` the weight
   needs.
4. **Sort the way the data is shaped.** A per-row insertion sort beats the
   heapsort below 128 elements per row (the default TQC row is 50) and loses
   above it, so the kernel picks on the row width. Both sides of the
   threshold are in the test suite.

Every kernel's contiguous axis runs `simd_width_of[DType.float64]()` doubles at
a time, with a scalar remainder loop behind it.

### Parallelism: not used

`from max.algorithm import parallelize` compiles and runs on this toolchain
(the `max` package is pinned in `pixi.toml`; `sync_parallelize` segfaults from
a ctypes host). On this build it costs milliseconds of fixed per-call overhead
for work that is microseconds to milliseconds, and the compute-bound shapes -
the only ones a thread pool could pay for - showed no scaling in the probes
that were run against it. That measurement was taken on scratch kernels that
are not in this tree, so no numbers are claimed here for it; the conclusion
the numbers supported is the one that acted on: every kernel in this repo is
serial. `pixi run bench` measures the code that is in the tree, and nothing
else. A caller on a different `max` build can revisit the row-per-batch and
row-per-sample loops.

### GPU: used, for the one kernel that earns it

`from max.gpu.host import DeviceContext` and `from max.gpu import block_idx,
thread_idx` both compile on `mojo == 1.2.0.dev2026092605` (`from std.gpu import
...` does not - that module is gone). `src/device.mojo` uses them, and the
machine has an RTX 5090 with well over the 4 GiB free the shared GPU policy
asks for. The Python wrapper checks `nvidia-smi --query-gpu=memory.free`
once per process and stays on the CPU below 4000 MiB, so the device is skipped
rather than fought for when it is busy.

Exactly one kernel goes to the device: `quantile_huber_loss`. It is the only
one whose arithmetic is out of proportion to its memory traffic. At
`(1024, 200) x 200` it moves 3.2 MB across the bus and computes 328 MFLOP -
about 100 flops per byte, well past the ~2 flops/byte line where a discrete
GPU stops being the right answer. The rest are at or under 0.1 flops/byte and
stay on the CPU, where they are already at memory bandwidth:

| kernel | work per byte moved | where it runs |
| --- | ---: | --- |
| `quantile_huber_loss` | ~100 flop/byte | GPU above 500k pairwise terms, CPU below |
| `BatchRenorm1d` | ~0.06 flop/byte | CPU: one fused multiply-add per feature |
| ARS weight update | ~0.06 flop/byte | CPU: one fused multiply-add per parameter |
| QR-DQN / TQC target | ~0.1 flop/byte | CPU: a reduction and a per-row sort |

The device path is one launch, one copy in, one copy out. The grid is a flat
1-D sweep over (batch row, `DW`-wide target tile) pairs, so every element of
`pairwise_delta` is covered exactly once and the `n_target_quantiles % 4`
remainder is a scalar tail inside the thread that owns it. Per-thread partials
come back in one array and are summed on the host - this device has no native
float64 atomic add, so a read-modify-write from every thread silently loses
updates. The same shapes, timed by `pixi run bench` with the term-count
threshold forced to the CPU one way and to zero the other:

| shape | CPU | GPU | device speedup |
| --- | ---: | ---: | ---: |
| QR-DQN, batch 256, n 200 | 7.895 ms | 0.891 ms | 8.9x |
| QR-DQN, batch 1024, n 200 | 34.243 ms | 1.719 ms | 19.9x |
| TQC, batch 256, 2x25 | 0.620 ms | 0.339 ms | 1.8x |
| TQC, batch 1024, 2x25 | 2.216 ms | 0.428 ms | 5.2x |
| QR-DQN, batch 64, n 50 | 0.187 ms | 0.369 ms | 0.5x |
| QR-DQN, batch 4, n 10 | 0.106 ms | 0.184 ms | 0.6x |

The device has about 0.2 ms of fixed cost - context, four buffer allocations,
one launch, one copy back - so it loses below roughly half a million pairwise
terms. That is where `_GPU_MIN_TERMS` in `common/utils.py` sits, and the last
two rows are why it is not lower. On a machine with no usable device the
benchmark prints a one-line note in place of this table rather than a number.

Two things about this target are worth writing down, because both are silent
wrong answers rather than compile errors. `simd_width_of[DType.float64]()` is
4 on the host and 2 on the device, so the width the host divides the target
axis by has to be pinned to the width the kernel strides by (`DW` in
`src/device.mojo`), not queried. And the device kernel is only wired up for
upstream's default `cum_prob`; a caller-supplied one, a small batch, an absent
device or a device that throws all come back as NaN and land on the CPU kernel,
which the parity tests pin to upstream either way.

See "Known divergences" for the one place where the CG wrapper's cost is
structural rather than arithmetic.

## Coverage

Ported (module path -> upstream path, and where the code sits upstream):

| this repo | upstream | function | upstream location |
| --- | --- | --- | --- |
| `mojo_sb3_contrib.common.utils.quantile_huber_loss` | `sb3_contrib.common.utils` | `quantile_huber_loss` | named function |
| `mojo_sb3_contrib.common.utils.conjugate_gradient_solver` | `sb3_contrib.common.utils` | `conjugate_gradient_solver` | named function |
| `mojo_sb3_contrib.common.torch_layers.BatchRenorm1d.forward` | `sb3_contrib.common.torch_layers` | `BatchRenorm.forward` | named method |
| `mojo_sb3_contrib.common.maskable.distributions.apply_masking` | `sb3_contrib.common.maskable.distributions` | `MaskableCategorical.apply_masking` | named method |
| `mojo_sb3_contrib.common.maskable.distributions.entropy` | `sb3_contrib.common.maskable.distributions` | `MaskableCategorical.entropy` | named method |
| `mojo_sb3_contrib.tqc.target_quantiles` | `sb3_contrib.tqc.tqc` | (inline) | `TQC.train`, `with th.no_grad():` block |
| `mojo_sb3_contrib.tqc.critic_loss` | `sb3_contrib.tqc.tqc` | (inline) | `TQC.train` critic loss line |
| `mojo_sb3_contrib.qrdqn.target_quantiles` | `sb3_contrib.qrdqn.qrdqn` | (inline) | `QRDQN.train`, `with th.no_grad():` block |
| `mojo_sb3_contrib.crossq.target_q_values` | `sb3_contrib.crossq.crossq` | (inline) | `CrossQ.train` target-Q block |
| `mojo_sb3_contrib.crossq.critic_loss` | `sb3_contrib.crossq.crossq` | (inline) | `CrossQ.train` critic loss line |
| `mojo_sb3_contrib.ars.top_returns` | `sb3_contrib.ars.ars` | (inline) | `ARS._do_one_update`, top-k block |
| `mojo_sb3_contrib.ars.gather` | `sb3_contrib.ars.ars` | (inline) | `ARS._do_one_update`, gather block |
| `mojo_sb3_contrib.ars.update_weights` | `sb3_contrib.ars.ars` | (inline) | `ARS._do_one_update`, weight update |

The rows marked *(inline)* are arithmetic upstream writes directly in `train()` /
`_do_one_update()` rather than in a named function. There is no upstream name to
keep, so this repo gives the block a name and the test asserts it against the
upstream expression copied verbatim out of the source file.

### Not covered

* **Everything that is not arithmetic.** The `nn.Module` classes themselves
  (`QuantileNetwork`, `Critic`, `Actor`, the policies), the optimizers, the
  replay and rollout buffers, `collect_rollouts`, `predict`, the env wrappers
  and callbacks, the multiprocess `AsyncEval`. Those are PyTorch or SB3
  machinery, not kernels, and rewriting them would add a second implementation
  of a network framework with no arithmetic win.
* **`flat_grad`** (`sb3_contrib.common.utils`). It is a five-line wrapper around
  `torch.autograd.grad`. There is no arithmetic to port.
* **`BatchRenorm.extra_repr`, `_check_input_dim`.** Preserved as Python
  methods for API surface; the second is dead code upstream too (it is never
  called from `forward`, and this port does not call it either).
* **`MaskableCategoricalDistribution` and the other maskable wrappers**
  (`ppo_mask`, `make_masked_proba_distribution`). The categorical kernel is
  ported; the distribution-factory plumbing around it is not.
* **`conjugate_gradient_solver`'s callback form.** The public Python signature
  is upstream's, but the kernel needs a dense matrix. See the divergence note.

### Known divergences

Each of these is forced by Mojo or by the C ABI, and each is isolated in the
ctypes glue rather than in the kernel's structure. None of them changed in the
optimisation pass; the traversal and vectorisation work is bit-identical and
is not a divergence.

1. **`conjugate_gradient_solver` takes a dense matrix internally.** The public
   Python function keeps upstream's signature
   (`matrix_vector_dot_fn, b, max_iter=10, residual_tol=1e-10`) and seeds
   `x = 1e-4 * th.randn_like(b)` exactly as upstream does, but a C-ABI function
   cannot carry a closure, so the wrapper materialises `A` by calling the
   supplied function on the `n` unit vectors. The Krylov iteration itself
   (initial residual, the two early-exit tolerances, the `i == max_iter - 1`
   short-circuit) is upstream's, line for line. Cost: `n` extra Python-level
   calls into the caller's function. At `n = 256` the benchmark reports 18.9 ms
   for the wrapper against 0.056 ms for the iteration itself. That gap is
   torch's per-op dispatch overhead multiplied by `n`, not arithmetic; the
   benchmark reports both numbers so the two are not confused. Calling the
   kernel directly through `_lib` with a matrix you already have avoids it.

2. **`quantile_huber_loss` takes `ndim` as an `Int`.** Upstream branches on
   `Tensor.ndim`; there is no tensor type here to interrogate, so the Python
   wrapper passes `2` or `3` and the kernel keeps both branches.
3. **Buffers cross as `Int` addresses.** `@export` rejects parametric functions
   and a pointer with an inferred origin is parametric, so every buffer arrives
   as an address and is rebuilt as
   `Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=...)` in
   `src/capi.mojo`.
4. **`BatchRenorm1d` rejects a 1-D input.** Upstream broadcasts it through
   `custom_mean[None]`; the kernel has no broadcasting and requires a 2-D
   array. `test_batch_renorm_1d_input_is_a_documented_divergence` pins both
   behaviours.
5. **`float64` throughout.** Upstream runs these in the network's dtype, which
   is `float32` by default. The Python layer casts up and casts scalar results
   back, so a `float32` caller still gets a `float32` result
   (`test_quantile_huber_loss_accepts_float32_like_upstream`). Two float32
   details upstream keeps are reproduced inside the kernel: the default
   `cum_prob` midpoints are built in `th.float`, and the weight
   `|cum_prob - (pairwise_delta < 0).float()|` is subtracted in `th.float`
   before being promoted for the product with the `float64` Huber term. Without
   those two details the loss disagrees with upstream at the 1e-9 level.
6. **A caller-supplied `cum_prob` is indexed differently from the default one.**
   Upstream views the default as `(1, -1, 1)` / `(1, 1, -1, 1)`, selecting by
   the *current* quantile, but right-aligns a user-supplied tensor against
   `pairwise_delta`, whose last axis is the *target* quantile. The kernel
   reproduces that asymmetry via three strides the wrapper computes from the
   shape.
7. **Not differentiable.** The kernels are forward/eval only, as they are in
   every upstream call site that uses them (all of them are inside
   `th.no_grad()` or feed a loss that is then differentiated through PyTorch).
   To use one inside a training step, keep PyTorch in charge of autograd and
   call the kernel inside `torch.no_grad()`; the loss value is unchanged.
8. **Transcendental accuracy.** `std.math`'s `log` is a fast approximation on
   this target, accurate to about `2e-9` relative; `exp` is accurate to about
   `1e-12`. `_log` in `ported.mojo` closes the gap with two Newton steps, so
   the residual disagreement with torch's SIMD paths is around `1e-12`
   relative, not `1e-9`. This is why the categorical tests assert `1e-10`
   rather than `1e-13`.
9. **Some kernels want scratch the caller has to size.** Three of them do,
   and the Python wrapper allocates it: `quantile_huber_loss` needs
   `2 * n_quantiles` floats (it writes upstream's two `th.float` weights per
   current quantile into that buffer once, instead of recomputing a division
   for every pairwise delta), `qrdqn.target_quantiles` needs `n_actions`, and
   `ars.update_weights` needs `n_params` for its accumulator. The kernels
   still allocate nothing themselves.
10. **The reduction order is upstream's, not torch's.** Every kernel sums in
   the order upstream's expression sums in, so changing the traversal order
   changes nothing; but torch's own reductions are blocked, so the two agree
   to within a few ulps rather than bit for bit. That is what the `1e-12`
   tolerances are, and they are unchanged from before the optimisation pass.
11. **One upstream bug found and fixed on the way.** Upstream's
   `quantile_huber_loss(current, target, sum_over_quantiles=True)` divides by
   `batch * n_critics * n_target_quantiles`, because `loss.sum(dim=-2)` leaves
   the critic axis in place for the 3-D layout and `.mean()` then divides by
   every element that is left. This port was dividing by
   `batch * n_target_quantiles`, which is the same number whenever
   `n_critics == 1` - so it agreed with upstream on every call site upstream
   actually has (QR-DQN is 2-D, TQC passes `False`), and disagreed by exactly
   `n_critics` on the 3-D-plus-`True` combination nothing exercised. The
   divisor is now upstream's, in both the CPU and the device kernel, and
   `test_quantile_huber_loss_gpu_path_matches_upstream` covers all four
   combinations of layout and flag against upstream. Callers who were relying
   on the old value for that one combination will see it change; it was never
   what upstream computes.

## Install

```bash
pixi install
```

The environment pins `mojo == 1.2.0.dev2026092605` and the matching
`max == 26.7.0.dev2026092605`, Python 3.13, and installs `sb3-contrib` from
PyPI so the parity tests and the benchmark have a real upstream to compare
against.

```bash
pixi run build   # mojo build --emit shared-lib -> dist/libmojo-sb3-contrib.so
pixi run test    # pytest
pixi run bench   # benchmarks, under a machine-wide flock
```

`python/mojo_sb3_contrib/_lib.py` also builds the library on first import if it
is older than `src/`, so `pixi run test` works in a fresh checkout without a
separate build step.

The `pyproject.toml` is there so the package can also be installed with
`pip install .`. A non-editable install copies the Python package out of the
checkout, so the loader can no longer find `src/` and compile: point
`MOJO_SB3_CONTRIB_LIB` at a built shared library (the one `pixi run build`
produces in `dist/`) and the import is instant.

```bash
pixi run build
pip install .
MOJO_SB3_CONTRIB_LIB=$PWD/dist/libmojo-sb3-contrib.so python -c "import mojo_sb3_contrib"
```

`MOJO_SB3_CONTRIB_MOJO` overrides the compiler the loader shells out to, for a
checkout where `mojo` is neither on `PATH` nor reachable through `pixi`.

## Usage

```python
import numpy as np
import torch as th

from mojo_sb3_contrib import tqc

rng = np.random.default_rng(0)
batch_size, n_critics, n_quantiles = 256, 2, 25
next_quantiles = th.as_tensor(rng.normal(size=(batch_size, n_critics, n_quantiles)))
current_quantiles = th.as_tensor(rng.normal(size=(batch_size, n_critics, n_quantiles)))
next_log_prob = th.as_tensor(rng.normal(size=batch_size))
rewards = th.as_tensor(rng.normal(size=(batch_size, 1)))
dones = th.zeros(batch_size, 1)
discounts = th.full((batch_size, 1), 0.99)

target_quantiles = tqc.target_quantiles(
    next_quantiles, next_log_prob, rewards, dones, discounts,
    ent_coef=0.2, top_quantiles_to_drop_per_net=2,
    n_critics=n_critics, n_quantiles=n_quantiles,
)
critic_loss = tqc.critic_loss(current_quantiles, target_quantiles)

# the same two calls, straight out of sb3_contrib
from sb3_contrib.common.utils import quantile_huber_loss as upstream
flat, _ = th.sort(next_quantiles.reshape(batch_size, -1))
kept = flat[:, : n_critics * n_quantiles - 2 * n_critics]
expected = rewards + (1 - dones) * discounts * (kept - 0.2 * next_log_prob.reshape(-1, 1))
print(critic_loss, upstream(current_quantiles, expected[:, None, :], sum_over_quantiles=False))
```

`examples/quickstart.py` is a runnable version of the above. Its output:

```
target quantiles (256, 46) -4.115466828058184
max |target delta| vs upstream: 8.881784197001252e-16
critic loss  mojo 0.470801398345   upstream 0.470801398345
relative delta: 1.5328012375148088e-15
```

`quantile_huber_loss` and `conjugate_gradient_solver` are drop-in for
upstream's functions: same name, same positional order, same defaults, same
`ValueError` messages (upstream's `curremt` typo included, so a
`pytest.raises(..., match=...)` written against upstream still matches).

## Benchmarks

Machine: **Intel Xeon E5-2697 v4 @ 2.30GHz**, Linux x86_64, 36 hardware
threads, one RTX 5090. torch 2.14.0 (36 threads), numpy 2.5.3, Python
3.13.15. Both columns are timed from Python, best of 5, with the same input
arrays. Reproduce with `pixi run bench`, which takes a machine-wide lock so a
concurrent job cannot distort the numbers.

These are one run, not a best-of-runs, and this is a shared machine. The
`upstream` column is torch's own 36-thread execution and moves by 10-20%
between runs; the `mojo` column is usually stable to a few percent but has been
seen 35% high on a busy run. Read the ratios as "one to two orders of
magnitude", not to three significant figures.

### quantile_huber_loss

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| QR-DQN, batch 256, n 200 | (256, 200) x 200 | 0.798 ms | 170.924 ms | 214.2x |
| QR-DQN, batch 1024, n 200 | (1024, 200) x 200 | 1.735 ms | 585.567 ms | 337.4x |
| TQC, batch 256, 2x50 | (256, 2, 25) x 50 | 0.351 ms | 8.196 ms | 23.3x |
| TQC, batch 1024, 2x50 | (1024, 2, 25) x 50 | 0.451 ms | 29.712 ms | 65.9x |

The one kernel with real arithmetic, and the one the device is for. torch
materialises the whole `(batch, n, n)` `pairwise_delta` tensor in memory and
then reduces it; the kernel streams the pairwise delta and reduces on the fly,
never allocating, and above 500k terms it does it on the GPU. All four of
these shapes are past that line, so the `mojo` column is device time; the
device-against-CPU table earlier in this file separates the two.

### TQC target quantiles (sort, drop top k, entropy, TD error)

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| TQC target | batch 256, 2x25 | 0.338 ms | 0.490 ms | 1.45x |
| TQC target | batch 1024, 2x25 | 1.087 ms | 1.452 ms | 1.34x |
| TQC target | batch 256, 5x50 | 3.071 ms | 2.297 ms | 0.75x |

Faster at the default shape, slower at 250 elements per row. The per-row sort
is an insertion sort below 128 elements and a heapsort above, which is where
the crossover is measured; at 50 elements per row - two critics times 25
quantiles, the TQC default - insertion sort's sequential shifts beat the
heap's dependent loads, and at 250 the `O(n log n)` wins again. torch's
`th.sort` is a blocked radix sort across 36 threads, which is the remaining
gap at the largest shape and is the one place a thread pool would have
closed.

### QR-DQN target quantiles (mean over quantiles, argmax over actions, gather)

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| QR-DQN target | batch 256, 200x6 | 0.260 ms | 0.378 ms | 1.45x |
| QR-DQN target | batch 1024, 200x10 | 1.021 ms | 0.674 ms | 0.66x |

`mean(dim=1)` reduces over the quantile axis, which strides `n_actions`
doubles per step; accumulating the per-action means in quantile-major order
instead reads the row once. The gather is fused into the TD target, so the row
is walked once for the mean and once for the gather rather than three times.
The remaining gap is again torch running the reduction on 36 threads.

### CrossQ target Q values (min over critics, entropy, TD error)

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| CrossQ target Q | batch 256, 2 critics | 0.053 ms | 0.098 ms | 1.85x |
| CrossQ target Q | batch 1024, 4 critics | 0.075 ms | 0.102 ms | 1.35x |

The whole thing is a few tens of microseconds, and it wins because the kernel
makes one FFI call where the upstream expression makes five. It was never a
target.

### BatchRenorm1d

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| BatchRenorm1d (train) | 4096 x 256 | 1.838 ms | 3.061 ms | 1.66x |
| BatchRenorm1d (eval) | 4096 x 256 | 0.955 ms | 0.648 ms | 0.68x |

The faithful loop is `batch_mean = x.mean(dim=0)`: for each feature `f`, walk
the `batch` rows. `x` is row-major `(batch, num_features)`, so that walk strides
`num_features` doubles per step and touches every cache line
`num_features / 8` times. This port walks whole rows instead, and folds the
per-feature normalisation into `(x - custom_mean) * gain + shift` - one
multiply-add per feature, no division and no square root in the inner loop.
Every feature still sums its own rows in ascending order, so the results are
unchanged; the win is in the memory traffic, and it is largest in training,
which also has to do the two column passes.

### ARS update (top-k selection, gather, weight update)

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| ARS top-k + update | n_delta 32, n_top 8, 5000 params | 0.221 ms | 2.437 ms | 11.03x |
| ARS top-k + update | n_delta 64, n_top 16, 50000 params | 1.388 ms | 0.716 ms | 0.52x |

`(plus - minus) @ deltas` reduced parameter-major with an L1-sized
accumulator, so each `deltas` element is read from memory once rather than
once per noise direction, and the gather is a vectorised contiguous copy.
Faster at 5000 parameters, still behind at 50000: the crossover is BLAS. At
5000 params the matvec is dispatch-dominated and the port's single fused pass
wins; at 50000 params torch reaches a blocked, threaded `sgemv`, and about
half of what is left is the 12.8 MB the gather copies, which is inherent to
the public `gather` / `update_weights` split.

### ARS conjugate gradient (10 Krylov iterations)

| kernel | shape | mojo | upstream | speedup |
| --- | --- | ---: | ---: | ---: |
| ARS CG, kernel only | n 64 | 0.020 ms | 0.429 ms | 21.7x |
| ARS CG, full wrapper | n 64 | 3.754 ms | 0.429 ms | 0.11x |
| ARS CG, kernel only | n 256 | 0.056 ms | 0.471 ms | 8.45x |
| ARS CG, full wrapper | n 256 | 18.888 ms | 0.471 ms | 0.02x |

The Krylov iteration itself is the fast half: the matvec is vectorised and
fused with the dot `alpha` needs. The wrapper is the honest number to plan
around - it pays `n` extra `matrix_vector_dot_fn` calls to materialise `A`
(divergence 1 above), which at `n = 256` is 18.9 ms of torch dispatch against
0.06 ms of arithmetic. Until the kernel takes a callback, the CG path is only
worth using when `A` is small or when the caller can supply it directly
through `_lib`.


## How it works

### FFI strategy

`build/build.sh` runs `mojo build --emit shared-lib` over `src/capi.mojo` with
`-I src`, producing `dist/libmojo-sb3-contrib.so`, which is the file the
Python loader `dlopen`s.

There are three files under `src/`. `ported.mojo` holds the CPU kernels and
knows nothing about the ABI: it takes
`Pointer[Float64, AnyOrigin[mut=True]]` and plain `Int`/`Float64`/`Bool`
values. `device.mojo` holds the one kernel that runs on a GPU. `capi.mojo` is
the only file with `@export`, and it does exactly one job — turn an `Int`
address back into a pointer and forward the call. That split is what keeps the
upstream branch structure visible in the kernel file instead of drowning it in
pointer plumbing.

Every CPU kernel is a pure function of its inputs. None of them allocate: the
Python wrapper owns every output buffer and every scratch block (`scratch(n)`
in `python/mojo_sb3_contrib/_lib.py` returns a zeroed `float64` array), and
passes addresses. Consequence: the kernels are re-entrant, and a caller that
wants to avoid the allocation can hold the scratch buffers and reuse them. The
device kernel is the exception: it allocates its own device buffers per call,
because the caching allocator is cheaper than keeping a context alive between
calls, and the whole call costs about 0.2 ms.

The Python layer is deliberately thin. `python/mojo_sb3_contrib/_lib.py` holds
the ctypes signatures and the loader; `_tensors.py` converts between torch
tensors and numpy arrays, casting to `float64` on the way
in and back to the caller's dtype on the way out. Everything else is a
per-module Python function whose name, signature and docstring mirror the
upstream function it replaces, so the modules line up with the upstream package
tree.

### Memory layout

All buffers are C-contiguous `float64` and are addressed in row-major order,
matching the tensor layouts upstream produces:

| shape | meaning | upstream equivalent |
| --- | --- | --- |
| `(batch, n_quantiles)` | QR-DQN current quantiles | `quantile_net(obs).gather(...)` |
| `(batch, n_critics, n_quantiles)` | TQC current quantiles | `self.critic(obs, action)` |
| `(batch, 1, n_target_quantiles)` | TQC target quantiles | `next_quantiles[:, None, :]` |
| `(batch, 1)` | rewards, dones, discounts | `ReplayBufferSamples` columns |
| `(batch, num_features)` | `BatchRenorm1d` input and output | `(batch_size, features)` |
| `(2 * batch, n_critics)` | CrossQ joint obs/next_obs Q values | `th.cat(self.critic(all_obs, all_actions), dim=1)` |
| `(n_delta, n_params)` | ARS noise directions | `deltas` |
| `(n, n)` | ARS Hessian-vector operator, materialised | `matrix_vector_dot_fn` |

Two reductions read a matrix by row (`_matvec`) and one writes a per-row
scratch block (`tqc_target_quantiles` sorts in place into a caller-provided
`n_critics * n_quantiles` buffer). Both are documented at the call site. The
scratch each caller must size is listed in "Known divergences" below; the
device path additionally needs a `batch * ceil(n_target / 4)` partial array,
which `common/utils.py` allocates per call.

Everything is float64 and little-endian, which is what the C ABI expects; a
non-contiguous or non-float64 input is copied by `numpy.ascontiguousarray`
before the call, so a caller never has to think about it.

## Tests

`pixi run test` runs 115 tests. Every reference is upstream: an imported
`sb3_contrib` function, an `sb3_contrib` `nn.Module`, or an expression copied
verbatim out of the upstream source file and evaluated in `float64`. The two
integration tests go further and drive a real `TQC.train` and a real
`QRDQN.train` with the loss replaced by a spy that records our value and
upstream's value for every call, then asserts per-call agreement on the real
data shapes. The rest are the paths this pass added: SIMD tails either side of
a vector boundary, the per-row sort's insertion/heapsort threshold, the ARS
accumulator block, the CG matvec/dot/axpy fusion, and the GPU kernel, which is
held to the same upstream reference at the same tolerance and skipped
entirely when no device with 4 GiB free is present.

## Layout

```
src/ported.mojo            CPU kernels, in upstream module and source order
src/device.mojo            the GPU kernel for quantile_huber_loss
src/capi.mojo              the @export C ABI
build/build.sh             mojo build --emit shared-lib
python/mojo_sb3_contrib/   ctypes loader + pure-Python API
tests/test_parity.py       parity against real sb3-contrib
bench/bench.py             benchmarks, markdown tables
examples/quickstart.py     runnable example
```

## License

MIT, matching upstream. See `LICENSE`.
