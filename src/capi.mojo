"""The C ABI the Python bindings call through.

`@export` refuses parametric functions and a pointer with an inferred origin is
parametric, so every buffer crosses as an `Int` address and is rebuilt here as a
`Pointer[Float64, AnyOrigin[mut=True]]`. Nothing in this file allocates; all
scratch space is passed in from Python.
"""

from ported import (
    apply_masking,
    ars_gather,
    ars_top_returns,
    ars_update_weights,
    batch_renorm_forward,
    conjugate_gradient_solver,
    crossq_critic_loss,
    crossq_target_q_values,
    entropy,
    qrdqn_target_quantiles,
    quantile_huber_loss,
    tqc_target_quantiles,
)

comptime FP = Pointer[Float64, AnyOrigin[mut=True]]
comptime IP = Pointer[Int, AnyOrigin[mut=True]]


from device import quantile_huber_loss_gpu


@export("sb3c_quantile_huber_loss_gpu")
def sb3c_quantile_huber_loss_gpu(
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
) abi("C") -> Float64:
    # abi("C") cannot be `raises`, and a device that is busy, absent or out of
    # memory must not take the caller down with it. NaN is the sentinel: the
    # only way a correct result is NaN is if the inputs are, and then the CPU
    # fallback returns NaN too.
    try:
        return quantile_huber_loss_gpu(
            current_quantiles,
            target_quantiles,
            cum_prob,
            acc_out,
            batch_size,
            n_critics,
            n_quantiles,
            n_target_quantiles,
            ndim,
            sum_over_quantiles,
        )
    except:
        return 0.0 / 0.0



@export("sb3c_quantile_huber_loss")
def sb3c_quantile_huber_loss(
    current_quantiles: Int,
    target_quantiles: Int,
    cum_prob: Int,
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
) abi("C") -> Float64:
    return quantile_huber_loss(
        FP(unsafe_from_address=current_quantiles),
        FP(unsafe_from_address=target_quantiles),
        FP(unsafe_from_address=cum_prob),
        batch_size,
        n_critics,
        n_quantiles,
        n_target_quantiles,
        ndim,
        use_cum_prob,
        sum_over_quantiles,
        cp_batch_stride,
        cp_critic_stride,
        cp_target_stride,
    )


@export("sb3c_conjugate_gradient_solver")
def sb3c_conjugate_gradient_solver(
    matrix: Int,
    b: Int,
    x: Int,
    scratch: Int,
    n: Int,
    max_iter: Int,
    residual_tol: Float64,
) abi("C") -> Int:
    return conjugate_gradient_solver(
        FP(unsafe_from_address=matrix),
        FP(unsafe_from_address=b),
        FP(unsafe_from_address=x),
        FP(unsafe_from_address=scratch),
        n,
        max_iter,
        residual_tol,
    )


@export("sb3c_apply_masking")
def sb3c_apply_masking(original_logits: Int, masks: Int, logits: Int, n: Int) abi("C"):
    apply_masking(
        FP(unsafe_from_address=original_logits),
        IP(unsafe_from_address=masks),
        FP(unsafe_from_address=logits),
        n,
    )


@export("sb3c_entropy")
def sb3c_entropy(logits: Int, masks: Int, scratch: Int, n: Int, use_masks: Bool) abi("C") -> Float64:
    return entropy(
        FP(unsafe_from_address=logits),
        IP(unsafe_from_address=masks),
        FP(unsafe_from_address=scratch),
        n,
        use_masks,
    )


@export("sb3c_batch_renorm_forward")
def sb3c_batch_renorm_forward(
    x: Int,
    out_ptr: Int,
    scratch: Int,
    ra_mean: Int,
    ra_var: Int,
    scale: Int,
    bias: Int,
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
) abi("C") -> Int:
    return batch_renorm_forward(
        FP(unsafe_from_address=x),
        FP(unsafe_from_address=out_ptr),
        FP(unsafe_from_address=scratch),
        FP(unsafe_from_address=ra_mean),
        FP(unsafe_from_address=ra_var),
        FP(unsafe_from_address=scale),
        FP(unsafe_from_address=bias),
        batch_size,
        num_features,
        training,
        steps,
        eps,
        momentum,
        affine,
        rmax,
        dmax,
        warmup_steps,
    )


@export("sb3c_qrdqn_target_quantiles")
def sb3c_qrdqn_target_quantiles(
    next_quantiles: Int,
    rewards: Int,
    dones: Int,
    discounts: Int,
    out_ptr: Int,
    scratch: Int,
    batch_size: Int,
    n_quantiles: Int,
    n_actions: Int,
) abi("C") -> Int:
    return qrdqn_target_quantiles(
        FP(unsafe_from_address=next_quantiles),
        FP(unsafe_from_address=rewards),
        FP(unsafe_from_address=dones),
        FP(unsafe_from_address=discounts),
        FP(unsafe_from_address=out_ptr),
        FP(unsafe_from_address=scratch),
        batch_size,
        n_quantiles,
        n_actions,
    )


@export("sb3c_tqc_target_quantiles")
def sb3c_tqc_target_quantiles(
    next_quantiles: Int,
    next_log_prob: Int,
    rewards: Int,
    dones: Int,
    discounts: Int,
    scratch: Int,
    out_ptr: Int,
    batch_size: Int,
    n_critics: Int,
    n_quantiles: Int,
    n_target_quantiles: Int,
    ent_coef: Float64,
) abi("C") -> Int:
    return tqc_target_quantiles(
        FP(unsafe_from_address=next_quantiles),
        FP(unsafe_from_address=next_log_prob),
        FP(unsafe_from_address=rewards),
        FP(unsafe_from_address=dones),
        FP(unsafe_from_address=discounts),
        FP(unsafe_from_address=scratch),
        FP(unsafe_from_address=out_ptr),
        batch_size,
        n_critics,
        n_quantiles,
        n_target_quantiles,
        ent_coef,
    )


@export("sb3c_crossq_target_q_values")
def sb3c_crossq_target_q_values(
    all_q_values: Int,
    next_log_prob: Int,
    rewards: Int,
    dones: Int,
    discounts: Int,
    out_ptr: Int,
    batch_size: Int,
    n_critics: Int,
    ent_coef: Float64,
) abi("C") -> Int:
    return crossq_target_q_values(
        FP(unsafe_from_address=all_q_values),
        FP(unsafe_from_address=next_log_prob),
        FP(unsafe_from_address=rewards),
        FP(unsafe_from_address=dones),
        FP(unsafe_from_address=discounts),
        FP(unsafe_from_address=out_ptr),
        batch_size,
        n_critics,
        ent_coef,
    )


@export("sb3c_crossq_critic_loss")
def sb3c_crossq_critic_loss(
    current_q_values: Int,
    target_q_values: Int,
    batch_size: Int,
    n_critics: Int,
) abi("C") -> Float64:
    return crossq_critic_loss(
        FP(unsafe_from_address=current_q_values),
        FP(unsafe_from_address=target_q_values),
        batch_size,
        n_critics,
    )


@export("sb3c_ars_top_returns")
def sb3c_ars_top_returns(
    plus_returns: Int,
    minus_returns: Int,
    out_idx: Int,
    out_return_std: Int,
    n_delta: Int,
    n_top: Int,
) abi("C") -> Int:
    return ars_top_returns(
        FP(unsafe_from_address=plus_returns),
        FP(unsafe_from_address=minus_returns),
        IP(unsafe_from_address=out_idx),
        FP(unsafe_from_address=out_return_std),
        n_delta,
        n_top,
    )


@export("sb3c_ars_gather")
def sb3c_ars_gather(
    plus_returns: Int,
    minus_returns: Int,
    deltas: Int,
    idx: Int,
    out_plus: Int,
    out_minus: Int,
    out_deltas: Int,
    n_delta: Int,
    n_params: Int,
) abi("C") -> Int:
    return ars_gather(
        FP(unsafe_from_address=plus_returns),
        FP(unsafe_from_address=minus_returns),
        FP(unsafe_from_address=deltas),
        IP(unsafe_from_address=idx),
        FP(unsafe_from_address=out_plus),
        FP(unsafe_from_address=out_minus),
        FP(unsafe_from_address=out_deltas),
        n_delta,
        n_params,
    )


@export("sb3c_ars_update_weights")
def sb3c_ars_update_weights(
    weights: Int,
    plus_returns: Int,
    minus_returns: Int,
    deltas: Int,
    out_weights: Int,
    scratch: Int,
    learning_rate: Float64,
    return_std: Float64,
    n_top: Int,
    n_params: Int,
) abi("C") -> Float64:
    return ars_update_weights(
        FP(unsafe_from_address=weights),
        FP(unsafe_from_address=plus_returns),
        FP(unsafe_from_address=minus_returns),
        FP(unsafe_from_address=deltas),
        FP(unsafe_from_address=out_weights),
        FP(unsafe_from_address=scratch),
        learning_rate,
        return_std,
        n_top,
        n_params,
    )
