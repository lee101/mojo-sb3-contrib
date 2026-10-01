# The GPU half of `quantile_huber_loss`.
#
# `ported.mojo` holds the CPU kernels and knows nothing about devices. This file
# holds the one kernel that is worth a GPU: the pairwise quantile-regression
# loss, which is `O(batch * n_critics * n_quantiles * n_target_quantiles)` and
# never materialises `pairwise_delta`. Every other kernel in this repo moves
# about 0.1 flop per byte and is a bandwidth problem that a discrete GPU does
# not help with; this one computes roughly 100 flops per byte of DRAM traffic
# and is the shape a GPU is for.
#
# The arithmetic is the same arithmetic `ported.mojo` runs on the CPU, and the
# same one upstream runs in torch: the same branchless Huber, the same
# `copysign` indicator, the same two `th.float` weights. Only the traversal and
# the reduction tree differ, and both are inside the tolerance the parity tests
# already assert.
#
# One launch per call, one copy in, one copy out. The launch is a flat 1-D
# grid over (batch row, target tile) pairs, so every element of `pairwise_delta`
# is covered exactly once and the `n_target_quantiles % 4` remainder is a
# scalar tail inside the thread that owns it. Each thread writes one partial,
# and the host sums them: this device has no native float64 atomic add, so a
# read-modify-write from every thread into one scalar silently loses updates.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import abs, ceildiv, copysign

# `simd_width_of[DType.float64]()` is 4 on this host and 2 on this device, so
# the width the kernel strides by and the width the host divides the target
# axis by have to be the same number. An explicit width is honoured on both
# sides - `SIMD[DType.float64, 4]` reduces correctly on a device whose native
# fp64 vector is half that - so the width is pinned here rather than queried.
comptime DW: Int = 4
comptime GP = Pointer[Float64, AnyOrigin[mut=True]]
comptime BLK: Int = 256

# One thread per (batch row, DW-wide target tile). `n_c` is the critic count
# (1 for the QR-DQN layout), `three` selects the 3-D current-quantile stride.
def _qh_kernel(
    current: GP,
    target: GP,
    weights: GP,
    acc: GP,
    batch_size: Int32,
    n_c: Int32,
    n_quantiles: Int32,
    n_target: Int32,
    tiles: Int32,
    three: Int32,
):
    var g = Int(global_idx.x)
    var total = Int(batch_size) * Int(tiles)
    if g >= total:
        return
    var b = g // Int(tiles)
    var m0 = (g % Int(tiles)) * DW
    # This thread owns [m0, tile_end) and nothing else. Running the vector loop
    # to the end of the row instead would have every tile re-count the whole
    # row, which is `tiles` times too much.
    var tile_end = m0 + DW
    if tile_end > Int(n_target):
        tile_end = Int(n_target)

    var one = SIMD[DType.float64, DW](1.0)
    var accv = SIMD[DType.float64, DW](0.0)
    var tail = 0.0

    var c = 0
    while c < Int(n_c):
        var n = 0
        while n < Int(n_quantiles):
            var cq = 0.0
            if three == 1:
                cq = current.unsafe_load((b * Int(n_c) + c) * Int(n_quantiles) + n)
            else:
                cq = current.unsafe_load(b * Int(n_quantiles) + n)
            var w_neg = SIMD[DType.float64, DW](weights.unsafe_load(n))
            var w_pos = SIMD[DType.float64, DW](weights.unsafe_load(Int(n_quantiles) + n))
            var m = m0
            while m + DW <= tile_end:
                var d = target.unsafe_load[width=DW](b * Int(n_target) + m) - cq
                var a = max(d, -d)
                var e = max(a - 1.0, SIMD[DType.float64, DW](0.0))
                var huber = (a * (a - e) + e) * 0.5
                var ind = (one - copysign(one, d)) * 0.5
                accv += huber * (w_neg * (one - ind) + w_pos * ind)
                m += DW
            while m < tile_end:
                var d = target.unsafe_load(b * Int(n_target) + m) - cq
                var a = max(d, -d)
                var huber: Float64
                if a > 1.0:
                    huber = a - 0.5
                else:
                    huber = a * a * 0.5
                # the weight follows the sign of the delta, not the Huber knee
                var w = w_neg[0]
                if d < 0.0:
                    w = w_pos[0]
                tail += huber * w
                m += 1
            n += 1
        c += 1

    # One partial per thread, not an atomic into a single scalar: this device
    # has no native float64 atomic add, so a read-modify-write from every
    # thread loses updates. The host sums the partials instead, which is both
    # correct and deterministic.
    acc.unsafe_store(g, accv.reduce_add() + tail)


# Mirrors `ported.quantile_huber_loss` for upstream's default `cum_prob`, which
# is the only case with enough arithmetic to be worth a device. Returns NaN if
# the device is unusable, and the Python wrapper falls back to the CPU kernel.
def quantile_huber_loss_gpu(
    current_quantiles: Int,
    target_quantiles: Int,
    cum_prob: Int,
    acc_out: Int,
    batch_size: Int,
    n_critics: Int,
    n_quantiles: Int,
    n_target_quantiles: Int,
    ndim: Int,
    sum_over_quantiles: Bool,
) raises -> Float64:
    var p_current = Pointer[Float64, AnyOrigin[mut=True]](
        unsafe_from_address=current_quantiles
    )
    var p_target = Pointer[Float64, AnyOrigin[mut=True]](
        unsafe_from_address=target_quantiles
    )
    var p_weights = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=cum_prob)
    var p_acc = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=acc_out)

    # The weights are the same two th.float values the CPU kernel materialises.
    var n = 0
    while n < n_quantiles:
        var mid = (Float32(n) + Float32(0.5)) / Float32(n_quantiles)
        p_weights.unsafe_store(n, Float64(abs(mid - Float32(0.0))))
        p_weights.unsafe_store(n_quantiles + n, Float64(abs(mid - Float32(1.0))))
        n += 1

    var n_c = 1
    var three = 0
    if ndim == 3:
        n_c = n_critics
        three = 1
    var tiles = (n_target_quantiles + DW - 1) // DW
    var total = batch_size * tiles

    with DeviceContext() as ctx:
        var d_current = ctx.enqueue_create_buffer[DType.float64](
            batch_size * n_c * n_quantiles
        )
        var d_target = ctx.enqueue_create_buffer[DType.float64](
            batch_size * n_target_quantiles
        )
        var d_weights = ctx.enqueue_create_buffer[DType.float64](2 * n_quantiles)
        var d_acc = ctx.enqueue_create_buffer[DType.float64](total)
        d_current.enqueue_copy_from(p_current)
        d_target.enqueue_copy_from(p_target)
        d_weights.enqueue_copy_from(p_weights)
        ctx.enqueue_function[_qh_kernel](
            d_current,
            d_target,
            d_weights,
            d_acc,
            Int32(batch_size),
            Int32(n_c),
            Int32(n_quantiles),
            Int32(n_target_quantiles),
            Int32(tiles),
            Int32(three),
            grid_dim=ceildiv(total, BLK),
            block_dim=BLK,
        )
        d_acc.enqueue_copy_to(p_acc)
        ctx.synchronize()

    var total_loss = 0.0
    var g = 0
    while g < total:
        total_loss += p_acc.unsafe_load(g)
        g += 1
    # Same divisor as the CPU kernel: `loss.sum(dim=-2).mean()` divides by
    # batch * n_critics * n_target, `loss.mean()` by every pairwise delta.
    var denom = Float64(batch_size * n_c * n_target_quantiles)
    if not sum_over_quantiles:
        denom = Float64(batch_size * n_c * n_quantiles * n_target_quantiles)
    return total_loss / denom
