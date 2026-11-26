"""Teacher activity and credit schedule, using MeSD's one-indexed steps."""


def teacher_schedule(config, step):
    target = float(config.stgca_lambda)
    warmup = config.stgca_lambda_warmup_steps
    decay = config.stgca_lambda_decay_steps
    if getattr(config, 'ar_alignment_mode', 'legacy') == 'mesd':
        if step < 1:
            raise ValueError('MeSD-aligned teacher schedule requires a one-indexed update step')
        cutoff = config.ar_teacher_disable_after_step
        active = not cutoff or step <= cutoff
        weight = target * max(1.0 - (step - 1) / decay, 0.0) if decay > 0 else target
        return active, weight if active else 0.0
    if warmup > 0 and step < warmup:
        return True, target * step / warmup
    if decay > 0 and step >= warmup:
        return True, target * max(1.0 - (step - warmup) / decay, 0.0)
    return True, target
