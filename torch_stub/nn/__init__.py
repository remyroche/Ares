"""Minimal torch.nn stub for tests."""

class Module:
    """Placeholder base module."""

    def __init__(self):
        self.training = True

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    """Placeholder linear layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features


class ReLU(Module):
    """Placeholder ReLU activation."""

    def forward(self, x):
        return x


__all__ = ["Module", "Linear", "ReLU"]
