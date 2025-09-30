"""Minimal torch.optim stub for tests."""

class Optimizer:
    """Placeholder optimizer."""

    def __init__(self, params, lr: float = 0.001):
        self.params = list(params)
        self.lr = lr

    def step(self):
        return None

    def zero_grad(self):
        return None


class Adam(Optimizer):
    """Placeholder Adam optimizer."""

    pass


class SGD(Optimizer):
    """Placeholder SGD optimizer."""

    pass


class AdamW(Optimizer):
    """Placeholder AdamW optimizer."""

    pass


__all__ = ["Optimizer", "Adam", "SGD", "AdamW"]
