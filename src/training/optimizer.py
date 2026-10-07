from __future__ import annotations

import jax
import optax


def build_lr_schedule(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:

    warmup = optax.linear_schedule(
        init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps
    )

    decay = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=max(total_steps - warmup_steps, 1),
        alpha=min_lr / peak_lr,
    )

    return optax.join_schedules(schedules=[warmup, decay], boundaries=[warmup_steps])


def _should_decay(value) -> bool:
    return value.ndim >= 2


def build_optimizer(
    params,
    peak_lr=6e-4,
    warmup_steps=2000,
    total_steps=600000,
    min_lr=6e-5,
    weight_decay=0.1,
    grad_clip_norm=1.0,
    b1=0.9,
    b2=0.95,
    eps=1e-8,
):
    schedule = build_lr_schedule(peak_lr, min_lr, warmup_steps, total_steps)

    decay_mask = jax.tree_util.tree_map(_should_decay, params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            mask=decay_mask,
        ),
    )

    return optimizer, schedule
