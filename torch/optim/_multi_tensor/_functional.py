import math
import torch
from torch import Tensor
from typing import List, Optional


def _has_sparse(tenosrs: List[Tensor]) -> bool:
    for tensor in tensors:
        if tensor.is_sparse:
            return True
    else:
        return False


def adagrad(
        params: List[Tensor],
        grads: List[Tensor],
        state_sums: List[Tensor],
        state_steps: List[int],
        *,
        lr: float,
        weight_decay: float,
        lr_decay: float,
        eps: float,
):
    """Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    if weight_decay != 0:
        torch._foreach_add(grads, params, alpha=weight_decay)

    clr = lr / (1 + (step - 1) * lr_decay)
    torch._foreach_addcmul_(state_sums, grads, grads, value=1)
    stds = torch._foreach_add(torch._foreach_sqrt(state_sums), eps)
    torch._foreach_addcdiv_(params, grads, stds, value=-clr)


def adam(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
):
    """Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    step_sizes = [-lr / (1 - beta1 ** step) for step in state_steps]
    sqrt_bias_correction2s = [math.sqrt(1 - beta2 ** step) for step in state_steps]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avgs = torch._foreach_add(torch._foreach_mul(exp_avgs, beta1), grads, alpha=1 - beta1)
    exp_avg_sqs = torch._foreach_addcmul(torch._foreach_mul(exp_avg_sqs, beta2), grads, grads, value=1 - beta2)

    if amsgrad:
        # Maintains the maximum of all 2nd moment running average till now
        max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)
        # Use the maximum for normalizing running average of gradient
        denom = torch._foreach_add(torch._foreach_div(torch._foreach_sqrt(max_exp_avg_sqs), sqrt_bias_correction2s), eps)
    else:
        denom = torch._foreach_add(torch._foreach_div(torch._foreach_sqrt(exp_avg_sqs), sqrt_bias_correction2s), eps)

    torch._foreach_addcdiv_(params, exp_avgs, denom, scalars=step_sizes)


def adamw(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
):
    """Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    step_sizes = [-lr / (1 - beta1 ** step) for step in state_steps]
    sqrt_bias_correction2s = [math.sqrt(1 - beta2 ** step) for step in state_steps]

    # Perform step weight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avgs = torch._foreach_add(torch._foreach_mul(exp_avgs, beta1), grads, alpha=1 - beta1)
    exp_avg_sqs = torch._foreach_addcmul(torch._foreach_mul(exp_avg_sqs, beta2), grads, grads, value=1 - beta2)
    if amsgrad:
        # Maintains the maximum of all 2nd moment running average till now
        max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)
        # Use the maximum for normalizing running average of gradient
        denom = torch._foreach_add(torch._foreach_div(torch._foreach_sqrt(max_exp_avg_sqs), sqrt_bias_correction2s), eps)
    else:
        denom = torch._foreach_add(torch._foreach_div(torch._foreach_sqrt(exp_avg_sqs), sqrt_bias_correction2s), eps)

    torch._foreach_addcdiv_(params, exp_avgs, denom, scalars=step_sizes)


def sgd(
        params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
):
    """Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if weight_decay != 0:
        d_p_list = torch._foreach_add(d_p_list, params, alpha=weight_decay)

    if momentum != 0:
        if momentum_buffer_list[0] is None:
            for i, d_p in enumerate(d_p_list):
                momentum_buffer_list[i] = d_p.clone().detach()
        else:
            torch._foreach_mul_(momentum_buffer_list, momentum)
            torch._foreach_add_(momentum_buffer_list, d_p_list, alpha=1 - dampening)

        if nesterov:
            d_p_list = torch._foreach_add(d_p_list, momentum_buffer_list, alpha=momentum)
        else:
            d_p_list = momentum_buffer_list
    torch._foreach_add_(params, d_p_list, alpha=-lr)


def adadelta(
        params: List[Tensor],
        grads: List[Tensor],
        square_avgs: List[Tensor],
        acc_deltas: List[Tensor],
        *,
        lr: float,
        rho: float,
        eps: float,
        weight_decay: float,
):
    """Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    torch._foreach_mul_(square_avgs, rho)
    torch._foreach_addcmul_(square_avgs, grads, grads, value=1 - rho)
    stds = torch._foreach_sqrt(torch._foreach_add(square_avgs, eps))
    # In native pytorch: `delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)`
    deltas = torch._foreach_mul(
        torch._foreach_div(torch._foreach_sqrt(torch._foreach_add(acc_delta, eps)), stds), grads)
    torch._foreach_add_(params, deltas, alpha=-lr)
    # update acc_deltas
    torch._foreach_mul_(acc_deltas, rho)
    torch._foreach_addcmul_(acc_deltas, deltas, deltas, value=1 - rho)


def rmsprop(
        params: List[Tensor],
        grads: List[Tensor],
        square_avgs: List[Tensor],
        grad_avgs: List[Tensor],
        momentum_buffer_list: List[Tensor],
        *,
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
        centered: bool,
):
    """Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    torch._foreach_mul_(square_avgs, alpha)
    torch._foreach_addcmul_(square_avgs, grads, grads, alpha=1 - alpha)

    if centered:
        torch._foreach_mul_(grad_avgs, alpha)
        torch._foreach_add_(grad_avgs, grads, alpha=1 - alpha)
        avgs = torch._foreach_add(
            torch._foreach_sqrt(
                torch._foreach_addcmul(square_avgs, grad_avgs, grad_avgs, value=-1)), eps)
    else:
        avgs = torch._foreach_add(torch._foreach_sqrt(square_avgs), eps)

    if momentum > 0:
        torch._foreach_mul(momentum_buffer_list, momentum)
        torch._foreach_addcdiv_(momentum_buffer_list, grads, avgs)
        torch._foreach_add_(params, momentum_buffer_list, alpha=-lr)
    else:
        torch._foreach_addcdiv_(params, grads, avgs, value=-lr)


# NOTE(mkozuki): Need more functions in foreach to make this better
def rprop(
        params: List[Tensor],
        grads: List[Tensor],
        prevs: List[Tensor],
        step_sizes: List[Tensor],
        *,
        step_size_min: float,
        step_size_max: float,
        etaminus: float,
        etaplus: float,
):
    """Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """

    signs = torch._foreach_sign(torch._foreach_mul(grads, prevs))
    for sign in signs:
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1

    # update step sizes with step size updates
    torch._foreach_mul_(step_sizes, signs)
    torch._foreach_clmp_(step_sizes, step_size_min, step_size_max)

    # dfdx=0 if dir<0, dfdx, otherwise
    for i, g in enumerate(grads):
        g = g.clone(memory_format=torch.preserve_format)
        g[signs[i].eq(etaminus)] = 0
        grads[i] = g

    # update parameters
    torch._foreach_addcmul_(params, torch._foreach_sign(grads), step_sizes, value=-1)

    for i, prev in enumerate(prevs):
        prev.copy_(grads[i])


def adamax(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_infs: List[Tensor],
        state_steps: List[int],
        *,
        eps: float,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
):
    """Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    # Update biased first moment estimate.
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    # Update the exponentially weighted infinity norm.
    norm_bufs = [
        torch.cat([exp_infs.mul_(beta2).unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)], 0)
        for exp_inf, grad in zip(exp_infs, grads)
    ]
    for nomr_buf, exp_inf in zip(norm_bufs, exp_infs):
        torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

    bias_corrections = [1 - beta1 ** step for step in state_steps]
    clrs = [-lr / bias_correction for bias_correction in bias_corrections]

    torch._foreach_addcdiv_(params, exp_avgs, exp_infs, scalars=clrs)
