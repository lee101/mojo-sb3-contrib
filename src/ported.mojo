# Kernels ported from sb3-contrib (Stable-Baselines-Team/stable-baselines3-contrib).
#
# Sections below follow upstream's own module order, and every function keeps
# upstream's name, argument order and branch structure. The only rewrites are
# the ones Mojo forces on us, and they are listed per-function:
#
#   * buffers arrive as raw pointers because `@export` cannot be parametric;
#     the Int-address -> Pointer construction lives in `capi.mojo`.
#   * `quantile_huber_loss` takes `ndim` as an Int instead of branching on
#     `Tensor.ndim`, because there is no tensor type to interrogate.
#   * `conjugate_gradient_solver` takes a dense matrix instead of a Python
#     callable, because a C-ABI function cannot carry a closure.
#   * scratch buffers are passed in rather than allocated, so the kernels do
#     no heap traffic at all.
#
# Everything is float64. Upstream runs these in the network's dtype (usually
# float32); the Python layer upcasts, so the arithmetic is identical.
#
# The reduction kernels keep upstream's arithmetic and summation order, but
# walk memory in the order the data is laid out instead of the order upstream
# names its axes in, and run the innermost contiguous axis through SIMD with a
# scalar tail. The arithmetic, and the order each reduction sums in, are the
# ones upstream's expressions use.

from std.math import abs, copysign, exp, fma, log, sqrt
from std.sys import simd_width_of

comptime FP = Pointer[Float64, AnyOrigin[mut=True]]
comptime IP = Pointer[Int, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


# ---------------------------------------------------------------------------
# sb3_contrib/common/utils.py
# ---------------------------------------------------------------------------


def quantile_huber_loss(
    current_quantiles: FP,
    target_quantiles: FP,
    cum_prob: FP,
    batch_size: Int,
    n_critics: Int,
    n_quantiles: Int,
    n_target_quantiles: Int,
    ndim: Int,
    use_cum_prob: Bool,
    sum_over_quantiles: Bool,
    cp_batch_stride: Int,
    cp_critic_stride: Int,
    cp_target_stride: Int,
) -> Float64:
    """`sb3_contrib.common.utils.quantile_huber_loss`.

    `ndim` is 2 for QR-DQN ((batch, n_quantiles)) and 3 for TQC
    ((batch, n_critics, n_quantiles)). `use_cum_prob` selects upstream's
    `cum_prob is None` branch, which builds the unit quantiles inline. In that
    branch `cum_prob` is a writable scratch buffer of `2 * n_quantiles` floats
    the kernel fills with the two possible weights (see below). For a
    caller-supplied `cum_prob`, `cp_batch_stride` / `cp_critic_stride` /
    `cp_target_stride` address it. Upstream right-aligns it against
    `pairwise_delta`, whose last axis is the target axis, so an explicit
    `cum_prob` is indexed by the target quantile while the default
    (constructed inline) is indexed by the current one. That asymmetry is
    upstream's and is reproduced here.

    `loss.sum(dim=-2).mean()` and `loss.mean()` sum the same pairwise terms and
    differ only in the divisor, so a single fused pass covers both; only the
    denominator below still branches. The pairwise delta is streamed over the
    target axis, which is the contiguous one for both layouts, so the term is
    evaluated a vector at a time and never materialised.
    """
    # n_critics is 1 in the QR-DQN case, which is what the (batch, 1,
    # n_quantiles) unsqueeze broadcasts the 2D layout up to.
    var n_c = 1
    if ndim == 3:
        n_c = n_critics

    if not use_cum_prob:
        # Upstream's default cum_prob is th.float and the weight
        # `abs(cum_prob - (pairwise_delta < 0).float())` is evaluated in
        # th.float before it is promoted, so both possible weights are
        # materialised here, in th.float, exactly once: entry n is the weight
        # for a non-negative pairwise delta, entry n_quantiles + n the one for
        # a negative delta.
        var n = 0
        while n < n_quantiles:
            var mid = (Float32(n) + Float32(0.5)) / Float32(n_quantiles)
            cum_prob.unsafe_store(n, Float64(abs(mid - Float32(0.0))))
            cum_prob.unsafe_store(n_quantiles + n, Float64(abs(mid - Float32(1.0))))
            n += 1

    var total = 0.0
    var b = 0
    if use_cum_prob:
        # A caller-supplied cum_prob is indexed by the target quantile, which is
        # not the contiguous axis of this layout, so this path stays scalar.
        while b < batch_size:
            var c = 0
            while c < n_c:
                var n = 0
                while n < n_quantiles:
                    var cq = _current_at(current_quantiles, b, c, n, ndim, n_c, n_quantiles)
                    var m = 0
                    while m < n_target_quantiles:
                        var d = target_quantiles.unsafe_load(b * n_target_quantiles + m) - cq
                        total += _huber(d) * abs(
                            cum_prob.unsafe_load(
                                b * cp_batch_stride + c * cp_critic_stride + m * cp_target_stride
                            ) - _indicator(d)
                        )
                        m += 1
                    n += 1
                c += 1
            b += 1
    else:
        var one = SIMD[DType.float64, W](1.0)
        while b < batch_size:
            var acc = SIMD[DType.float64, W](0.0)
            var c = 0
            while c < n_c:
                var n = 0
                while n < n_quantiles:
                    var cq = _current_at(current_quantiles, b, c, n, ndim, n_c, n_quantiles)
                    var w_neg = SIMD[DType.float64, W](cum_prob.unsafe_load(n))
                    var w_pos = SIMD[DType.float64, W](cum_prob.unsafe_load(n_quantiles + n))
                    var m = 0
                    while m + W <= n_target_quantiles:
                        var d = target_quantiles.unsafe_load[width=W](b * n_target_quantiles + m) - cq
                        var ind = _indicator(d)
                        acc += _huber(d) * (w_neg * (one - ind) + w_pos * ind)
                        m += W
                    while m < n_target_quantiles:
                        var d = target_quantiles.unsafe_load(b * n_target_quantiles + m) - cq
                        total += _huber(d) * _default_weight(cum_prob, n, n_quantiles, d)
                        m += 1
                    n += 1
                c += 1
            total += acc.reduce_add()
            b += 1

    # `loss.sum(dim=-2)` leaves (batch, n_critics, n_target) for the 3-D layout
    # and (batch, n_target) for the 2-D one, and `.mean()` then divides by every
    # element that is left, so the critic axis counts in both cases (it is 1 in
    # the 2-D layout, which is what the unsqueeze broadcasts up from).
    # `loss.mean()` divides by every element of pairwise_delta.
    var denom = Float64(batch_size * n_c * n_target_quantiles)
    if not sum_over_quantiles:
        denom = Float64(batch_size * n_c * n_quantiles * n_target_quantiles)
    return total / denom


def _current_at(
    current_quantiles: FP,
    b: Int,
    c: Int,
    n: Int,
    ndim: Int,
    n_c: Int,
    n_quantiles: Int,
) -> Float64:
    if ndim == 2:
        return current_quantiles.unsafe_load(b * n_quantiles + n)
    return current_quantiles.unsafe_load((b * n_c + c) * n_quantiles + n)


def _default_weight(cum_prob: FP, n: Int, n_quantiles: Int, d: Float64) -> Float64:
    # scalar tail of the vectorised weight above; the indicator is 0 or 1, so
    # picking between the two materialised weights reproduces upstream's
    # `abs(cum_prob - (pairwise_delta < 0).float())` bit for bit.
    if d < 0.0:
        return cum_prob.unsafe_load(n_quantiles + n)
    return cum_prob.unsafe_load(n)


# pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
# abs_pairwise_delta = th.abs(pairwise_delta)
# huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5,
#                       pairwise_delta ** 2 * 0.5)
# loss = th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
#
# The vector form drops both branches. With `e = max(|d| - 1, 0)` and
# `u = |d| - e = min(|d|, 1)`, `(a * u + e) * 0.5` is `a * a * 0.5` below the
# knee (bit-identical, because `e` is then exactly zero) and `a - 0.5` above
# it, where the two summands are both positive so nothing cancels.
def _huber(pairwise_delta: Float64) -> Float64:
    var a = max(pairwise_delta, -pairwise_delta)
    if a > 1.0:
        return a - 0.5
    return a * a * 0.5


def _huber(pairwise_delta: SIMD[DType.float64, W]) -> SIMD[DType.float64, W]:
    var a = max(pairwise_delta, -pairwise_delta)
    var e = max(a - 1.0, SIMD[DType.float64, W](0.0))
    return (a * (a - e) + e) * 0.5


# (pairwise_delta < 0).float(). A SIMD comparison yields a mask this target
# cannot turn back into a number, but `copysign` is exact, so the sign bit goes
# straight to the 0.0/1.0 the kernel needs.
def _indicator(pairwise_delta: Float64) -> Float64:
    if pairwise_delta < 0.0:
        return 1.0
    return 0.0


def _indicator(pairwise_delta: SIMD[DType.float64, W]) -> SIMD[DType.float64, W]:
    var one = SIMD[DType.float64, W](1.0)
    return (one - copysign(one, pairwise_delta)) * 0.5


# `matrix_vector_dot_fn` becomes a dense row-major matrix: ARS only ever forms
# A = Hessian-vector products out of the same flat parameter vector, and a
# C-ABI function cannot carry a closure anyway. `x` arrives already holding
# upstream's `1e-4 * th.randn_like(b)` seed.
def conjugate_gradient_solver(
    matrix: FP,
    b: FP,
    x: FP,
    scratch: FP,
    n: Int,
    max_iter: Int,
    residual_tol: Float64,
) -> Int:
    """`sb3_contrib.common.utils.conjugate_gradient_solver`.

    `scratch` is 3 * n floats: residual, p and A_dot_p. Returns the number of
    iterations actually run; the caller reads the solution out of `x`.
    """
    var residual = scratch
    var p = scratch.unsafe_offset(n)
    var a_dot_p = scratch.unsafe_offset(2 * n)

    # residual = b - matrix_vector_dot_fn(x), and upstream's `p = residual`
    # copy, in one pass.
    _matvec_diff(b, matrix, x, residual, p, n)

    # Equivalent to th.linalg.norm(residual) ** 2
    var residual_squared_norm = _dot(residual, residual, n)

    if residual_squared_norm < residual_tol:
        # The denominator in alpha would be zero.
        return 0

    var it = 0
    while it < max_iter:
        # A @ p, fused with the p . (A @ p) the alpha needs.
        var pap = _matvec_dot(matrix, p, a_dot_p, n)

        var alpha = residual_squared_norm / pap
        _axpy(x, x, alpha, p, n)

        if it == max_iter - 1:
            return it + 1

        _axpy(residual, residual, -alpha, a_dot_p, n)
        var new_residual_squared_norm = _dot(residual, residual, n)

        if new_residual_squared_norm < residual_tol:
            return it + 1

        var beta = new_residual_squared_norm / residual_squared_norm
        residual_squared_norm = new_residual_squared_norm
        _axpy(p, residual, beta, p, n)
        it += 1
    # Only reached when max_iter == 0.
    return it


# out_ptr = b - matrix @ v, and `p` receives the same values.
def _matvec_diff(b: FP, matrix: FP, v: FP, out_ptr: FP, p: FP, n: Int):
    var row = 0
    while row < n:
        var m = matrix.unsafe_offset(row * n)
        var acc = SIMD[DType.float64, W](0.0)
        var col = 0
        while col + W <= n:
            acc = fma(m.unsafe_load[width=W](col), v.unsafe_load[width=W](col), acc)
            col += W
        var tail = 0.0
        while col < n:
            tail += m.unsafe_load(col) * v.unsafe_load(col)
            col += 1
        var r = b.unsafe_load(row) - (acc.reduce_add() + tail)
        out_ptr.unsafe_store(row, r)
        p.unsafe_store(row, r)
        row += 1


# out_ptr = matrix @ v, returning p . out_ptr: the matvec and the dot alpha
# needs in one pass over the matrix row.
def _matvec_dot(matrix: FP, v: FP, out_ptr: FP, n: Int) -> Float64:
    var total = 0.0
    var row = 0
    while row < n:
        var m = matrix.unsafe_offset(row * n)
        var acc = SIMD[DType.float64, W](0.0)
        var col = 0
        while col + W <= n:
            acc = fma(m.unsafe_load[width=W](col), v.unsafe_load[width=W](col), acc)
            col += W
        var tail = 0.0
        while col < n:
            tail += m.unsafe_load(col) * v.unsafe_load(col)
            col += 1
        var s = acc.reduce_add() + tail
        out_ptr.unsafe_store(row, s)
        total += v.unsafe_load(row) * s
        row += 1
    return total


def _dot(a: FP, b: FP, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        i += W
    var tail = 0.0
    while i < n:
        tail += a.unsafe_load(i) * b.unsafe_load(i)
        i += 1
    return acc.reduce_add() + tail


def _axpy(out_ptr: FP, base: FP, alpha: Float64, v: FP, n: Int):
    # out_ptr = base + alpha * v, elementwise; out_ptr may alias base or v.
    var av = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        out_ptr.unsafe_store(i, base.unsafe_load[width=W](i) + av * v.unsafe_load[width=W](i))
        i += W
    while i < n:
        out_ptr.unsafe_store(i, base.unsafe_load(i) + alpha * v.unsafe_load(i))
        i += 1


# ---------------------------------------------------------------------------
# sb3_contrib/common/maskable/distributions.py
# ---------------------------------------------------------------------------


# torch's Categorical normalises its logits to log-probabilities, and upstream
# builds a Categorical twice: once with the raw logits (which becomes
# `_original_logits`) and once more with the masked ones.
#   logits = th.where(self.masks, self._original_logits, HUGE_NEG)
#   super().__init__(logits=logits)   ->   logits - logits.logsumexp(dim=-1)
# with HUGE_NEG = th.tensor(-1e8, ...). `masks` is 1 where the choice is legal.
def apply_masking(original_logits: FP, masks: IP, logits: FP, n: Int):
    """`sb3_contrib.common.maskable.distributions.MaskableCategorical.apply_masking`."""
    _copy(logits, original_logits, n)
    _renormalise(logits, n)

    var i = 0
    while i < n:
        if masks.unsafe_load(i) == 0:
            logits.unsafe_store(i, -1e8)
        i += 1
    _renormalise(logits, n)


# logits - logits.logsumexp(dim=-1, keepdim=True)
def _renormalise(a: FP, n: Int):
    var log_z = _logsumexp(a, n)
    var lz = SIMD[DType.float64, W](log_z)
    var i = 0
    while i + W <= n:
        a.unsafe_store(i, a.unsafe_load[width=W](i) - lz)
        i += W
    while i < n:
        a.unsafe_store(i, a.unsafe_load(i) - log_z)
        i += 1


# p_log_p = self.logits * self.probs
# p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, ...))
# return -p_log_p.sum(-1)
def entropy(logits: FP, masks: IP, scratch: FP, n: Int, use_masks: Bool) -> Float64:
    """`sb3_contrib.common.maskable.distributions.MaskableCategorical.entropy`.

    Upstream reads `self.logits`, which is the *masked and re-normalised* copy
    Categorical holds, and `self.probs` is its softmax. Both are rebuilt here
    from the unmasked logits so the caller can pass what it was constructed
    with.
    """
    var masked = scratch
    _copy(masked, logits, n)
    _renormalise(masked, n)
    if use_masks:
        var i = 0
        while i < n:
            if masks.unsafe_load(i) == 0:
                masked.unsafe_store(i, -1e8)
            i += 1
        _renormalise(masked, n)

    # self.probs is Categorical's softmax of the normalised logits
    var denom = 0.0
    var i = 0
    while i < n:
        denom += exp(masked.unsafe_load(i))
        i += 1

    # p_log_p = self.logits * self.probs
    # p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, ...))
    # return -p_log_p.sum(-1)
    var acc = 0.0
    i = 0
    while i < n:
        var p_log_p = masked.unsafe_load(i) * exp(masked.unsafe_load(i)) / denom
        if use_masks:
            # Highly negative logits don't result in 0 probs, so we must replace
            # with 0s to ensure 0 contribution to the distribution's entropy.
            if masks.unsafe_load(i) == 0:
                p_log_p = 0.0
        acc += p_log_p
        i += 1
    return -acc


# std.math's `log` is a fast approximation on this target with about 2e-9
# relative error, against ~1e-12 for `exp`. Two Newton steps on log(x) using exp
# close the gap, so the float64 kernels stay accurate to the last few bits.
def _log(x: Float64) -> Float64:
    var y = log(x)
    y = y + (x * exp(-y) - 1.0)
    y = y + (x * exp(-y) - 1.0)
    return y


def _logsumexp(a: FP, n: Int) -> Float64:
    var m = a.unsafe_load(0)
    var i = 1
    while i < n:
        if a.unsafe_load(i) > m:
            m = a.unsafe_load(i)
        i += 1
    var s = 0.0
    i = 0
    while i < n:
        s += exp(a.unsafe_load(i) - m)
        i += 1
    return m + _log(s)


def _copy(out_ptr: FP, v: FP, n: Int):
    var i = 0
    while i + W <= n:
        out_ptr.unsafe_store(i, v.unsafe_load[width=W](i))
        i += W
    while i < n:
        out_ptr.unsafe_store(i, v.unsafe_load(i))
        i += 1


# ---------------------------------------------------------------------------
# sb3_contrib/common/torch_layers.py
# ---------------------------------------------------------------------------


# Scratch layout is [batch_mean, batch_var, custom_mean, custom_var], each
# num_features wide.
def batch_renorm_forward(
    x: FP,
    out_ptr: FP,
    scratch: FP,
    ra_mean: FP,
    ra_var: FP,
    scale: FP,
    bias: FP,
    batch_size: Int,
    num_features: Int,
    training: Bool,
    steps: Int,
    eps: Float64,
    momentum: Float64,
    affine: Bool,
    rmax: Float64,
    dmax: Float64,
    warmup_steps: Int,
) -> Int:
    """`sb3_contrib.common.torch_layers.BatchRenorm1d.forward`.

    `x` is (batch_size, num_features) row-major, as is the output.
    `ra_mean`, `ra_var`, `scale` and `bias` are read and, in training mode,
    the running statistics are updated in place exactly as upstream does.
    Returns the value `self.steps` holds after the call.

    Upstream writes `x.mean(dim=0)`, which is a reduction over the rows of a
    row-major array. Summing the columns of that reduction in feature order
    instead of row order touches every cache line once rather than once per
    feature; each feature still accumulates its own rows in the same order, so
    the sums are bit-identical to the row-major-per-feature walk. The
    normalisation pass then runs along the contiguous feature axis.
    """
    var batch_mean = scratch
    var batch_var = scratch.unsafe_offset(num_features)
    var custom_mean = scratch.unsafe_offset(2 * num_features)
    var custom_var = scratch.unsafe_offset(3 * num_features)

    var new_steps = steps
    var f = 0

    if training:
        # batch_mean = x.mean(dim=0)
        _col_mean(x, batch_mean, batch_size, num_features)
        # batch_var = x.var(dim=0)   (torch's default correction=1)
        _col_var(x, batch_mean, batch_var, batch_size, num_features)

        # Use batch statistics during the initial warm up phase.
        if steps > warmup_steps:
            f = 0
            while f < num_features:
                var batch_std = sqrt(batch_var.unsafe_load(f) + eps)
                var running_std = sqrt(ra_var.unsafe_load(f) + eps)
                # scale
                var r = batch_std / running_std
                r = min(max(r, 1.0 / rmax), rmax)
                # bias
                var d = (batch_mean.unsafe_load(f) - ra_mean.unsafe_load(f)) / running_std
                d = min(max(d, -dmax), dmax)
                # BatchNorm normalization, using minibatch stats and running
                # average stats
                custom_mean.unsafe_store(f, batch_mean.unsafe_load(f) - d * sqrt(batch_var.unsafe_load(f)) / r)
                custom_var.unsafe_store(f, batch_var.unsafe_load(f) / (r * r))
                f += 1
        else:
            _copy(custom_mean, batch_mean, num_features)
            _copy(custom_var, batch_var, num_features)

        # Update Running Statistics
        f = 0
        while f < num_features:
            ra_mean.unsafe_store(f, ra_mean.unsafe_load(f) + momentum * (batch_mean.unsafe_load(f) - ra_mean.unsafe_load(f)))
            ra_var.unsafe_store(f, ra_var.unsafe_load(f) + momentum * (batch_var.unsafe_load(f) - ra_var.unsafe_load(f)))
            f += 1
        new_steps = steps + 1
    else:
        # Use running statistics during evaluation mode
        _copy(custom_mean, ra_mean, num_features)
        _copy(custom_var, ra_var, num_features)

    # The per-feature normalisation collapses to `y = (x - custom_mean) * gain
    # + shift`, with `gain = scale / sqrt(custom_var + eps)` when affine and
    # `1 / sqrt(custom_var + eps)` when not. Folding it here takes the square
    # root, the division and the affine pair out of the pass over the batch,
    # which then costs four L1-resident loads and a multiply-add per feature.
    # `batch_mean` and `batch_var` are dead by now: the running statistics were
    # already updated, and the evaluation path never wrote them.
    var gain = batch_mean
    var shift = batch_var
    f = 0
    while f < num_features:
        var inv = 1.0 / sqrt(custom_var.unsafe_load(f) + eps)
        var g = inv
        var s = 0.0
        if affine:
            g = scale.unsafe_load(f) * inv
            s = bias.unsafe_load(f)
        gain.unsafe_store(f, g)
        shift.unsafe_store(f, s)
        f += 1

    # Normalize, one row at a time: `x` and the output are both row-major, so
    # this reads and writes each cache line exactly once.
    var r = 0
    while r < batch_size:
        var off = r * num_features
        f = 0
        while f + W <= num_features:
            out_ptr.unsafe_store(
                off + f,
                (x.unsafe_load[width=W](off + f) - custom_mean.unsafe_load[width=W](f))
                * gain.unsafe_load[width=W](f)
                + shift.unsafe_load[width=W](f),
            )
            f += W
        while f < num_features:
            out_ptr.unsafe_store(
                off + f,
                (x.unsafe_load(off + f) - custom_mean.unsafe_load(f)) * gain.unsafe_load(f)
                + shift.unsafe_load(f),
            )
            f += 1
        r += 1

    return new_steps

# batch_mean = x.mean(dim=0). One row at a time, accumulating W features at a
# time in `mean_out` itself, so `x` is read once and each feature still sums
# its rows in ascending order.
def _col_mean(x: FP, mean_out: FP, batch_size: Int, num_features: Int):
    _col_zero(mean_out, num_features)
    var r = 0
    while r < batch_size:
        var off = r * num_features
        var f = 0
        while f + W <= num_features:
            mean_out.unsafe_store(
                f, mean_out.unsafe_load[width=W](f) + x.unsafe_load[width=W](off + f)
            )
            f += W
        while f < num_features:
            mean_out.unsafe_store(f, mean_out.unsafe_load(f) + x.unsafe_load(off + f))
            f += 1
        r += 1
    var f = 0
    while f + W <= num_features:
        mean_out.unsafe_store(f, mean_out.unsafe_load[width=W](f) / Float64(batch_size))
        f += W
    while f < num_features:
        mean_out.unsafe_store(f, mean_out.unsafe_load(f) / Float64(batch_size))
        f += 1


# batch_var = x.var(dim=0), correction=1: the same walk, squared about the mean.
def _col_var(x: FP, mean: FP, var_out: FP, batch_size: Int, num_features: Int):
    _col_zero(var_out, num_features)
    var r = 0
    while r < batch_size:
        var off = r * num_features
        var f = 0
        while f + W <= num_features:
            var d = x.unsafe_load[width=W](off + f) - mean.unsafe_load[width=W](f)
            var_out.unsafe_store(f, var_out.unsafe_load[width=W](f) + d * d)
            f += W
        while f < num_features:
            var d = x.unsafe_load(off + f) - mean.unsafe_load(f)
            var_out.unsafe_store(f, var_out.unsafe_load(f) + d * d)
            f += 1
        r += 1
    var denom = batch_size - 1
    var f = 0
    while f + W <= num_features:
        if denom <= 0:
            var_out.unsafe_store(f, SIMD[DType.float64, W](_nan()))
        else:
            var_out.unsafe_store(f, var_out.unsafe_load[width=W](f) / Float64(denom))
        f += W
    while f < num_features:
        if denom <= 0:
            var_out.unsafe_store(f, _nan())
        else:
            var_out.unsafe_store(f, var_out.unsafe_load(f) / Float64(denom))
        f += 1


def _col_zero(a: FP, n: Int):
    var f = 0
    while f + W <= n:
        a.unsafe_store(f, SIMD[DType.float64, W](0.0))
        f += W
    while f < n:
        a.unsafe_store(f, 0.0)
        f += 1


def _nan() -> Float64:
    return 0.0 / 0.0


# ---------------------------------------------------------------------------
# sb3_contrib/qrdqn/qrdqn.py  -- QRDQN.train, the `with th.no_grad():` block
# ---------------------------------------------------------------------------


# next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
# next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
# target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles
def qrdqn_target_quantiles(
    next_quantiles: FP,
    rewards: FP,
    dones: FP,
    discounts: FP,
    out_ptr: FP,
    scratch: FP,
    batch_size: Int,
    n_quantiles: Int,
    n_actions: Int,
) -> Int:
    """The numeric block of `sb3_contrib.qrdqn.qrdqn.QRDQN.train`.

    `next_quantiles` is (batch_size, n_quantiles, n_actions); the output is
    (batch_size, n_quantiles). `scratch` holds n_actions doubles. Returns the
    greedy action chosen per row.

    `mean(dim=1)` is a reduction over the quantile axis, but accumulating the
    per-action means in quantile-major order walks the row contiguously, which
    is the whole difference between reading every cache line once and reading it
    once per action. Each action still sums its quantiles in the same order, so
    the means are bit-identical. The gather and the 1-step TD target are then
    one pass: `(1 - dones) * discounts` is loop-invariant, and the fused
    expression is the one upstream writes.
    """
    var b = 0
    while b < batch_size:
        var means = scratch
        var a = 0
        while a + W <= n_actions:
            means.unsafe_store(a, SIMD[DType.float64, W](0.0))
            a += W
        while a < n_actions:
            means.unsafe_store(a, 0.0)
            a += 1
        var n = 0
        while n < n_quantiles:
            var off = (b * n_quantiles + n) * n_actions
            a = 0
            while a + W <= n_actions:
                means.unsafe_store(
                    a,
                    means.unsafe_load[width=W](a)
                    + next_quantiles.unsafe_load[width=W](off + a),
                )
                a += W
            while a < n_actions:
                means.unsafe_store(a, means.unsafe_load(a) + next_quantiles.unsafe_load(off + a))
                a += 1
            n += 1
        a = 0
        while a + W <= n_actions:
            means.unsafe_store(a, means.unsafe_load[width=W](a) / Float64(n_quantiles))
            a += W
        while a < n_actions:
            means.unsafe_store(a, means.unsafe_load(a) / Float64(n_quantiles))
            a += 1

        var best_action = 0
        var best_mean = _nan()
        a = 0
        while a < n_actions:
            var m = means.unsafe_load(a)
            if a == 0 or m > best_mean:
                best_mean = m
                best_action = a
            a += 1

        # 1-step TD target, gathered straight out of the row
        var td = (1.0 - dones.unsafe_load(b)) * discounts.unsafe_load(b)
        var rew = rewards.unsafe_load(b)
        n = 0
        while n < n_quantiles:
            out_ptr.unsafe_store(
                b * n_quantiles + n,
                rew + td * next_quantiles.unsafe_load((b * n_quantiles + n) * n_actions + best_action),
            )
            n += 1
        b += 1

    return b


# ---------------------------------------------------------------------------
# sb3_contrib/tqc/tqc.py  -- TQC.train, the `with th.no_grad():` block
# ---------------------------------------------------------------------------


# next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
# next_quantiles = next_quantiles[:, :n_target_quantiles]
# target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
# target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * target_quantiles
def tqc_target_quantiles(
    next_quantiles: FP,
    next_log_prob: FP,
    rewards: FP,
    dones: FP,
    discounts: FP,
    scratch: FP,
    out_ptr: FP,
    batch_size: Int,
    n_critics: Int,
    n_quantiles: Int,
    n_target_quantiles: Int,
    ent_coef: Float64,
) -> Int:
    """The numeric block of `sb3_contrib.tqc.tqc.TQC.train`.

    `next_quantiles` is (batch_size, n_critics, n_quantiles); the output is
    (batch_size, n_target_quantiles). `scratch` holds n_critics*n_quantiles
    floats per row.

    The per-row scalars (`next_log_prob`, `rewards`, `dones`, `discounts`) are
    loop-invariant over the quantile axis upstream writes them inside, so they
    are hoisted out; the row is then copied in one contiguous block and sorted
    in place, as before.
    """
    var width = n_critics * n_quantiles
    var b = 0
    while b < batch_size:
        var lp = ent_coef * next_log_prob.unsafe_load(b)
        var td = (1.0 - dones.unsafe_load(b)) * discounts.unsafe_load(b)
        var rew = rewards.unsafe_load(b)
        var src = next_quantiles.unsafe_offset(b * width)
        var c = 0
        while c + W <= width:
            scratch.unsafe_store(c, src.unsafe_load[width=W](c))
            c += W
        while c < width:
            scratch.unsafe_store(c, src.unsafe_load(c))
            c += 1
        _sort_ascending(scratch, width)

        var m = 0
        while m < n_target_quantiles:
            out_ptr.unsafe_store(
                b * n_target_quantiles + m,
                rew + td * (scratch.unsafe_load(m) - lp),
            )
            m += 1
        b += 1
    return b


# th.sort's ascending order, in place. torch's sort is not stable for the
# float64 keys used here either, and only the values matter downstream. Below
# the threshold a straight insertion sort wins: at these widths its shifts are
# sequential and cache-resident, where the heap's sift-downs are dependent
# loads behind an unpredictable branch at every level. Above it the O(n log n)
# heapsort takes over; the two measure the same around n = 200.
def _sort_ascending(a: FP, n: Int):
    if n < 2:
        return
    if n <= 128:
        _insertion_sort(a, n)
        return
    var i = n // 2 - 1
    while i >= 0:
        _sift_down(a, i, n)
        i -= 1
    var end = n - 1
    while end > 0:
        var tmp = a.unsafe_load(0)
        var last = a.unsafe_load(end)
        a.unsafe_store(0, last)
        a.unsafe_store(end, tmp)
        _sift_down(a, 0, end)
        end -= 1


def _insertion_sort(a: FP, n: Int):
    var i = 1
    while i < n:
        var v = a.unsafe_load(i)
        var j = i - 1
        while j >= 0:
            if a.unsafe_load(j) <= v:
                break
            a.unsafe_store(j + 1, a.unsafe_load(j))
            j -= 1
        a.unsafe_store(j + 1, v)
        i += 1


def _sift_down(a: FP, start: Int, size: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= size:
            return
        var right = child + 1
        if right < size:
            var cv = a.unsafe_load(child)
            if a.unsafe_load(right) > cv:
                child = right
        var cv = a.unsafe_load(child)
        var rv = a.unsafe_load(root)
        if rv >= cv:
            return
        a.unsafe_store(root, cv)
        a.unsafe_store(child, rv)
        root = child


# ---------------------------------------------------------------------------
# sb3_contrib/crossq/crossq.py  -- CrossQ.train
# ---------------------------------------------------------------------------


# current_q_values, next_q_values = th.split(all_q_values, batch_size, dim=0)
# next_q_values, _ = th.min(next_q_values.detach(), dim=1, keepdim=True)
# next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
# target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
def crossq_target_q_values(
    all_q_values: FP,
    next_log_prob: FP,
    rewards: FP,
    dones: FP,
    discounts: FP,
    out_ptr: FP,
    batch_size: Int,
    n_critics: Int,
    ent_coef: Float64,
) -> Int:
    """The numeric block of `sb3_contrib.crossq.crossq.CrossQ.train`.

    `all_q_values` is the (2 * batch_size, n_critics) stack of the joint
    obs/next_obs forward pass; the output is (batch_size,).
    """
    var b = 0
    while b < batch_size:
        var row = all_q_values.unsafe_offset((batch_size + b) * n_critics)
        var min_q = row.unsafe_load(0)
        var c = 1
        while c < n_critics:
            if row.unsafe_load(c) < min_q:
                min_q = row.unsafe_load(c)
            c += 1
        var next_q = min_q - ent_coef * next_log_prob.unsafe_load(b)
        out_ptr.unsafe_store(
            b,
            rewards.unsafe_load(b) + (1.0 - dones.unsafe_load(b)) * discounts.unsafe_load(b) * next_q,
        )
        b += 1
    return b


# critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values.detach())
#                        for current_q in current_q_values)
def crossq_critic_loss(
    current_q_values: FP,
    target_q_values: FP,
    batch_size: Int,
    n_critics: Int,
) -> Float64:
    """`sb3_contrib.crossq.crossq.CrossQ.train`'s critic loss.

    `current_q_values` is (n_critics, batch_size); `target_q_values` is
    (batch_size,).
    """
    var total = 0.0
    var c = 0
    while c < n_critics:
        var row = current_q_values.unsafe_offset(c * batch_size)
        var acc = SIMD[DType.float64, W](0.0)
        var b = 0
        while b + W <= batch_size:
            var d = row.unsafe_load[width=W](b) - target_q_values.unsafe_load[width=W](b)
            acc += d * d
            b += W
        var sq = acc.reduce_add()
        while b < batch_size:
            var d = row.unsafe_load(b) - target_q_values.unsafe_load(b)
            sq += d * d
            b += 1
        total += sq / Float64(batch_size)
        c += 1
    return 0.5 * total


# ---------------------------------------------------------------------------
# sb3_contrib/ars/ars.py  -- ARS._do_one_update
# ---------------------------------------------------------------------------


# top_returns, _ = th.max(th.vstack((plus_returns, minus_returns)), dim=0)
# top_idx = th.argsort(top_returns, descending=True)[:n_top]
# plus_returns = plus_returns[top_idx]; minus_returns = minus_returns[top_idx]
# return_std = th.cat([plus_returns, minus_returns]).std()
def ars_top_returns(
    plus_returns: FP,
    minus_returns: FP,
    out_idx: IP,
    out_return_std: FP,
    n_delta: Int,
    n_top: Int,
) -> Int:
    """The top-k selection of `sb3_contrib.ars.ars.ARS._do_one_update`.

    Selection by repeated maximum, skipping what has already been taken. Ties
    keep the lower index first, as `th.argsort` does here. Testing membership of
    the taken set is only reached by a candidate that beats the running best,
    which is what keeps this O(n_delta) per pick rather than O(n_delta * n_top).
    """
    var taken = 0
    while taken < n_top:
        var best = -1
        var best_r = 0.0
        var i = 0
        while i < n_delta:
            var r = max(plus_returns.unsafe_load(i), minus_returns.unsafe_load(i))
            if best < 0 or r > best_r:
                var already = False
                var t = 0
                while t < taken:
                    if out_idx.unsafe_load(t) == i:
                        already = True
                    t += 1
                if not already:
                    best = i
                    best_r = r
            i += 1
        out_idx.unsafe_store(taken, best)
        taken += 1

    # th.cat([plus_returns, minus_returns]).std(), unbiased.
    var acc = 0.0
    var m = 0
    while m < 2 * n_top:
        var v = 0.0
        if m < n_top:
            v = plus_returns.unsafe_load(out_idx.unsafe_load(m))
        else:
            v = minus_returns.unsafe_load(out_idx.unsafe_load(m - n_top))
        acc += v
        m += 1
    var mean = acc / Float64(2 * n_top)
    var sq = 0.0
    m = 0
    while m < 2 * n_top:
        var v = 0.0
        if m < n_top:
            v = plus_returns.unsafe_load(out_idx.unsafe_load(m))
        else:
            v = minus_returns.unsafe_load(out_idx.unsafe_load(m - n_top))
        sq += (v - mean) * (v - mean)
        m += 1
    out_return_std.unsafe_store(0, sqrt(sq / Float64(2 * n_top - 1)))
    return n_top


# plus_returns = plus_returns[top_idx]
# minus_returns = minus_returns[top_idx]
# deltas = deltas[top_idx]
def ars_gather(
    plus_returns: FP,
    minus_returns: FP,
    deltas: FP,
    idx: IP,
    out_plus: FP,
    out_minus: FP,
    out_deltas: FP,
    n_delta: Int,
    n_params: Int,
) -> Int:
    """The top-k gather of `sb3_contrib.ars.ars.ARS._do_one_update`."""
    var t = 0
    while t < n_delta:
        var i = idx.unsafe_load(t)
        out_plus.unsafe_store(t, plus_returns.unsafe_load(i))
        out_minus.unsafe_store(t, minus_returns.unsafe_load(i))
        var src = deltas.unsafe_offset(i * n_params)
        var dst = out_deltas.unsafe_offset(t * n_params)
        var p = 0
        while p + W <= n_params:
            dst.unsafe_store(p, src.unsafe_load[width=W](p))
            p += W
        while p < n_params:
            dst.unsafe_store(p, src.unsafe_load(p))
            p += 1
        t += 1
    return n_delta


# step_size = learning_rate / (self.n_top * return_std + 1e-6)
# self.weights = self.weights + step_size * ((plus_returns - minus_returns) @ deltas)
def ars_update_weights(
    weights: FP,
    plus_returns: FP,
    minus_returns: FP,
    deltas: FP,
    out_weights: FP,
    scratch: FP,
    learning_rate: Float64,
    return_std: Float64,
    n_top: Int,
    n_params: Int,
) -> Float64:
    """The parameter-noise update of `sb3_contrib.ars.ars.ARS._do_one_update`.

    Returns the step size, which upstream records as `train/step_size`.

    `(plus - minus) @ deltas` is a (n_top, n_params) row-major product reduced
    over the rows. Reducing it parameter-major instead accumulates a whole
    contiguous run of parameters at a time. That run is kept small enough to
    stay in L1, so every element of `deltas` is read from memory exactly once
    instead of once per noise direction. The accumulation order over t is
    unchanged, so the result is bit-identical.
    """
    var step_size = learning_rate / (Float64(n_top) * return_std + 1e-6)

    # A block of parameters small enough that the accumulator stays in L1.
    var block = 4096 - 4096 % W
    var lo = 0
    while lo < n_params:
        var hi = lo + block
        if hi > n_params:
            hi = n_params
        var p = lo
        while p + W <= hi:
            scratch.unsafe_store(p, SIMD[DType.float64, W](0.0))
            p += W
        while p < hi:
            scratch.unsafe_store(p, 0.0)
            p += 1
        var t = 0
        while t < n_top:
            var coef = plus_returns.unsafe_load(t) - minus_returns.unsafe_load(t)
            var row = deltas.unsafe_offset(t * n_params)
            p = lo
            while p + W <= hi:
                scratch.unsafe_store(
                    p, scratch.unsafe_load[width=W](p) + coef * row.unsafe_load[width=W](p)
                )
                p += W
            while p < hi:
                scratch.unsafe_store(p, scratch.unsafe_load(p) + coef * row.unsafe_load(p))
                p += 1
            t += 1
        lo = hi

    var ss = SIMD[DType.float64, W](step_size)
    var p = 0
    while p + W <= n_params:
        out_weights.unsafe_store(
            p, weights.unsafe_load[width=W](p) + ss * scratch.unsafe_load[width=W](p)
        )
        p += W
    while p < n_params:
        out_weights.unsafe_store(p, weights.unsafe_load(p) + step_size * scratch.unsafe_load(p))
        p += 1
    return step_size
