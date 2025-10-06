import sys
import types

import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _DummyTensor:
        def __init__(self, *args, **kwargs):
            self.value = 0.0

        def unsqueeze(self, *args, **kwargs):
            return self

        def item(self):
            return self.value

    class _DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class _DummyNNModule:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def parameters(self):
            return []

        def __call__(self, *args, **kwargs):
            return _DummyTensor()

    class _DummyOptimizer:
        def zero_grad(self):
            return None

        def step(self):
            return None

    torch_stub.FloatTensor = lambda *args, **kwargs: _DummyTensor()
    torch_stub.Tensor = _DummyTensor
    torch_stub.argmax = lambda *args, **kwargs: _DummyTensor()
    torch_stub.max = lambda *args, **kwargs: _DummyTensor()
    torch_stub.no_grad = lambda: _DummyContext()
    torch_stub.device = lambda *args, **kwargs: None
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.nn = types.ModuleType("torch.nn")
    torch_stub.nn.__path__ = []
    torch_stub.nn.Module = _DummyNNModule
    torch_stub.nn.Sequential = lambda *args, **kwargs: _DummyNNModule()
    torch_stub.nn.Linear = _DummyNNModule
    torch_stub.nn.ReLU = _DummyNNModule
    torch_stub.nn.Softmax = _DummyNNModule
    torch_stub.nn.functional = types.ModuleType("torch.nn.functional")
    torch_stub.nn.utils = types.ModuleType("torch.nn.utils")
    torch_stub.nn.utils.__path__ = []
    torch_stub.nn.utils.prune = types.ModuleType("torch.nn.utils.prune")
    torch_stub.optim = types.ModuleType("torch.optim")
    torch_stub.optim.Optimizer = _DummyOptimizer
    torch_stub.optim.Adam = lambda *args, **kwargs: _DummyOptimizer()

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_stub.nn
    sys.modules["torch.nn.functional"] = torch_stub.nn.functional
    sys.modules["torch.nn.utils"] = torch_stub.nn.utils
    sys.modules["torch.nn.utils.prune"] = torch_stub.nn.utils.prune
    sys.modules["torch.optim"] = torch_stub.optim

from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
    FeatureLookbackOptimizationComponent,
)


def test_multi_target_consensus_uniform_fallback_when_weights_zero():
    component = FeatureLookbackOptimizationComponent.__new__(FeatureLookbackOptimizationComponent)

    feature_results = {
        "target_a": {"score": 0.0, "target_type": "composite_leverage", "lookback": 10},
        "target_b": {"score": 0.0, "target_type": "short_probability", "lookback": 20},
    }

    result = component._calculate_multi_target_consensus(feature_results, target_scores={})

    assert result["method"] == "uniform_consensus"
    assert result["lookback"] == 15
    assert pytest.approx(result["weighted_score"], abs=1e-9) == 0.0
