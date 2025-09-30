"""Minimal torch.nn.utils.prune stub."""

class BasePruningMethod:
    """Placeholder base pruning method."""

    def apply(self, module, name):
        return None


def l1_unstructured(module, name, amount):
    return None


__all__ = ["BasePruningMethod", "l1_unstructured"]
