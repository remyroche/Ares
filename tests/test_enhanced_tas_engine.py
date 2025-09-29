import numpy as np
import runpy
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TreeModelConfig:
    model_type: str = "dummy"


@dataclass
class TreeModelResult:
    model: Any
    train_score: float
    val_score: float
    test_score: Optional[float] = None
    model_type: str = "dummy"
    config: TreeModelConfig = field(default_factory=TreeModelConfig)
    success: bool = True


@dataclass
class AutoMLResult:
    best_model: Any = None
    best_config: TreeModelConfig = field(default_factory=TreeModelConfig)
    best_score: float = 0.0
    model_results: List[TreeModelResult] = field(default_factory=list)
    success: bool = False


@dataclass
class AdvancedEvaluationResult:
    success: bool = False


@dataclass
class Individual:
    parameters: Dict[str, Any]


@dataclass
class EvolutionaryResult:
    best_individuals: List[Individual]
    pareto_front: List[Individual]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    success: bool


class TreeModelEvaluator:
    def evaluate_model(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError


class EnhancedTreeModelFactory:
    def create_model(self, config: TreeModelConfig):  # pragma: no cover - stub
        raise NotImplementedError


class TreeAutoMLManager:
    def optimize(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError


class AdvancedEvaluator:
    def evaluate(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError


class EvolutionaryAlgorithmManager:
    def optimize_with_algorithm(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError


class TASConfig:
    pass


class TASSearchConfig:
    pass


class TASOptimizationConfig:
    pass


class TASResult:
    pass


class TreeArchitectureSearchEngine:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        pass


def _create_stub_packages() -> Dict[str, types.ModuleType]:
    repo_root = Path(__file__).resolve().parents[1]
    packages = {
        "src": repo_root / "src",
        "src.utils": repo_root / "src/utils",
        "src.utils.ml_common": repo_root / "src/utils/ml_common",
        "src.utils.ml_common.optimization": repo_root / "src/utils/ml_common/optimization",
        "src.utils.ml_common.optimization.tas": repo_root / "src/utils/ml_common/optimization/tas",
        "src.utils.ml_common.optimization.tas.models": repo_root
        / "src/utils/ml_common/optimization/tas/models",
        "src.utils.ml_common.optimization.tas.automl": repo_root
        / "src/utils/ml_common/optimization/tas/automl",
        "src.utils.ml_common.optimization.shared_utils": repo_root
        / "src/utils/ml_common/optimization/shared_utils",
        "src.utils.ml_common.optimization.tas.core": repo_root
        / "src/utils/ml_common/optimization/tas/core",
    }

    stubs: Dict[str, types.ModuleType] = {}
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        stubs[name] = module
    return stubs


def _install_stub_modules(stubs: Dict[str, types.ModuleType]) -> Dict[str, Optional[types.ModuleType]]:
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    return saved


def _restore_modules(saved: Dict[str, Optional[types.ModuleType]]):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_engine_module():
    stubs = _create_stub_packages()
    saved = _install_stub_modules(stubs)
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/utils/ml_common/optimization/tas/enhanced_tas_engine.py"
    )

    init_globals = {
        "__name__": "src.utils.ml_common.optimization.tas.enhanced_tas_engine",
        "__package__": "src.utils.ml_common.optimization.tas",
        "__file__": str(module_path),
        "EnhancedTreeModelFactory": EnhancedTreeModelFactory,
        "TreeModelConfig": TreeModelConfig,
        "TreeModelResult": TreeModelResult,
        "TreeModelEvaluator": TreeModelEvaluator,
        "create_model_ensemble": lambda configs: [],
        "TreeAutoMLManager": TreeAutoMLManager,
        "AutoMLConfig": type("AutoMLConfig", (), {}),
        "AutoMLResult": AutoMLResult,
        "create_tree_automl_manager": lambda *args, **kwargs: None,
        "AdvancedEvaluator": AdvancedEvaluator,
        "AdvancedEvaluationResult": AdvancedEvaluationResult,
        "create_advanced_evaluator": lambda *args, **kwargs: None,
        "EvolutionaryAlgorithmManager": EvolutionaryAlgorithmManager,
        "EvolutionaryConfig": type("EvolutionaryConfig", (), {}),
        "EvolutionaryResult": EvolutionaryResult,
        "create_evolutionary_algorithm_manager": lambda *args, **kwargs: None,
        "TASConfig": TASConfig,
        "TASSearchConfig": TASSearchConfig,
        "TASOptimizationConfig": TASOptimizationConfig,
        "TASResult": TASResult,
        "TreeArchitectureSearchEngine": TreeArchitectureSearchEngine,
    }

    module_globals = runpy.run_path(str(module_path), init_globals=init_globals)
    module = types.ModuleType(init_globals["__name__"])
    module.__dict__.update(module_globals)
    sys.modules[module.__name__] = module
    _restore_modules(saved)
    return module


engine_module = _load_engine_module()
EnhancedTASEngine = engine_module.EnhancedTASEngine
EnhancedTASConfig = engine_module.EnhancedTASConfig


class DummyModel:
    def __init__(self, config: TreeModelConfig):
        self.config = config
        self.fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.fitted = True
        return self

    def predict(self, X):
        if X is None:
            return None
        return np.zeros(len(X))

    def get_feature_importance(self):
        return {}


class DummyFactory:
    def create_model(self, config: TreeModelConfig):
        return DummyModel(config)


class DummyEvaluator:
    def __init__(self, val_score: float = 0.75):
        self.val_score = val_score
        self.last_call = None

    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        model.fit(X_train, y_train, X_val, y_val)
        self.last_call = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }
        return TreeModelResult(
            model=model,
            train_score=self.val_score,
            val_score=self.val_score,
            test_score=self.val_score,
            model_type=model.config.model_type,
            config=model.config,
            success=True,
        )


def test_evolutionary_selection_uses_provided_data():
    config = EnhancedTASConfig(
        enable_enhanced_models=False,
        enable_automl=False,
        enable_evolutionary_search=False,
        enable_ensemble=False,
        enable_advanced_metrics=False,
    )
    engine = EnhancedTASEngine(config)
    engine.model_factory = DummyFactory()
    evaluator = DummyEvaluator(val_score=0.42)
    engine.model_evaluator = evaluator

    X_train = np.ones((4, 2))
    y_train = np.arange(4)
    X_val = np.ones((2, 2)) * 2
    y_val = np.arange(2)
    X_test = np.ones((1, 2)) * 3
    y_test = np.arange(1)

    candidate = Individual(parameters={"model_type": "dummy"})
    evolutionary_result = EvolutionaryResult(
        best_individuals=[candidate],
        pareto_front=[candidate],
        optimization_history=[],
        convergence_info={},
        execution_time=0.0,
        success=True,
    )

    best_model, best_config, best_score = engine._select_best_model(
        automl_result=None,
        evolutionary_result=evolutionary_result,
        ensemble_models=[],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    assert best_model is not None
    assert isinstance(best_config, TreeModelConfig)
    assert best_config.model_type == "dummy"
    assert best_score == evaluator.val_score
    assert engine.best_result is not None
    assert engine.best_result.model is best_model
    assert evaluator.last_call["X_train"] is X_train
    assert evaluator.last_call["X_test"] is X_test
