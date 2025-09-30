"""Minimal torch stub for test environment."""

from types import SimpleNamespace

__all__ = ["nn", "optim", "tensor", "Tensor", "device", "cuda"]


class Tensor:
    """Simple placeholder for torch.Tensor."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def numpy(self):
        return self.args


def tensor(*args, **kwargs):
    """Create a placeholder tensor."""
    return Tensor(*args, **kwargs)


class device(str):
    """Minimal placeholder for torch.device."""

    def __new__(cls, value: str = "cpu"):
        return str.__new__(cls, value)


def _cuda_is_available() -> bool:
    return False


def _cuda_device_count() -> int:
    return 0


cuda = SimpleNamespace(is_available=_cuda_is_available, device_count=_cuda_device_count)

nn = SimpleNamespace()
optim = SimpleNamespace()
