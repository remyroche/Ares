"""Minimal torch.optim.lr_scheduler stub for tests."""

class _LRScheduler:
    """Placeholder LR scheduler."""

    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min


class ReduceLROnPlateau(_LRScheduler):
    def step(self, metrics=None):
        self.last_epoch += 1


__all__ = ["_LRScheduler", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
