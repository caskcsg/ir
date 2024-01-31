#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Add Min LR to Linear & Cosine Scheduler

@Author  :   Ma (Ma787639046@outlook.com)
'''

import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _get_linear_schedule_with_warmup_lr_lambda_minlr(current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float):
    # 1) linear warmup for warmup_iters steps
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 2) if current_step > lr_decay_iters, return min learning rate
    if current_step > num_training_steps:
        return min_lr_ratio
    # 3) in between, use linear decay down to min learning rate
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))) * (1 - min_lr_ratio) + min_lr_ratio


def get_linear_schedule_with_warmup_minlr(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_lr_ratio=0.):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The final learning rate at the end of the linear decay will be `init_lr * min_lr_ratio`.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda_minlr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_cosine_schedule_with_warmup_lr_lambda_minlr(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float
):
    # 1) linear warmup for warmup_iters steps
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 2) if current_step > lr_decay_iters, return min learning rate
    if current_step > num_training_steps:
        return min_lr_ratio
    # 3) in between, use cosine decay down to min learning rate
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (1 - min_lr_ratio) + min_lr_ratio

def get_cosine_schedule_with_warmup_minlr(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio=0.
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda_minlr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

