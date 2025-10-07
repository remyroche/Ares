import sys
from types import SimpleNamespace, ModuleType
import importlib.util
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

import pandas as pd
import pytest


class _DummyModel:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _dummy_distribution(*args, **kwargs):
    return SimpleNamespace()


def _dummy_advi(*args, **kwargs):
    return SimpleNamespace(fit=lambda *a, **k: None)


def _dummy_sample(*args, **kwargs):
    return SimpleNamespace()


# Provide lightweight stubs for optional dependencies so the module can be imported
pymc_stub = ModuleType("pymc")
pymc_stub.Model = _DummyModel
pymc_stub.Normal = _dummy_distribution
pymc_stub.HalfNormal = _dummy_distribution
pymc_stub.ADVI = _dummy_advi
pymc_stub.sample = _dummy_sample
sys.modules.setdefault("pymc", pymc_stub)

aesara_stub = ModuleType("aesara")
aesara_tensor_stub = ModuleType("aesara.tensor")
aesara_stub.tensor = aesara_tensor_stub
sys.modules.setdefault("aesara", aesara_stub)
sys.modules.setdefault("aesara.tensor", aesara_tensor_stub)

sys.modules.setdefault(
    "cvxpy",
    SimpleNamespace(
        Variable=object,
        Parameter=object,
        Problem=object,
        Minimize=object,
        Constraint=object,
        sum=lambda *args, **kwargs: None,
    ),
)


# Provide lightweight stubs for feature_engineering modules expected by phase1_probe
feature_engineering_stub = ModuleType("feature_engineering")
feature_registry_stub = ModuleType("feature_engineering.feature_registry")
transforms_stub = ModuleType("feature_engineering.transforms")


class _FeatureFamily(Enum):
    PRICE_RETURNS = "price_returns"
    VOLATILITY = "volatility"
    MEAN_REVERSION = "mean_reversion"
    ANCHORS_TOD = "anchors_tod"


@dataclass
class _FeatureMetadata:
    family: _FeatureFamily


class _FeatureRegistry:
    def __init__(self):
        self._family_map = {
            _FeatureFamily.PRICE_RETURNS: ["p/price_ema10_pct"],
            _FeatureFamily.VOLATILITY: ["v/volatility_stub"],
            _FeatureFamily.MEAN_REVERSION: ["m/mean_stub"],
            _FeatureFamily.ANCHORS_TOD: ["a/anchor_stub"],
        }
        self._feature_family = {
            feature: family
            for family, features in self._family_map.items()
            for feature in features
        }

    def get_features_by_family(self, family: _FeatureFamily):
        return list(self._family_map.get(family, []))

    def get_feature_metadata(self, base_feature: str):
        family = self._feature_family.get(base_feature, _FeatureFamily.PRICE_RETURNS)
        return _FeatureMetadata(family=family)

    def compute_feature(self, base_feature: str, data: pd.DataFrame):
        if base_feature in data:
            return data[base_feature]
        return pd.Series(index=data.index, dtype=float)


class _TransformType(Enum):
    IDENTITY = "id"


@dataclass
class _TransformConfig:
    transform_type: _TransformType
    params: dict


class _TransformRouter:
    def __init__(self, config):
        self.config = config

    def fit_transform(self, train_df, valid_df):
        return {
            feature: {"train": train_df[[feature]]}
            for feature in train_df.columns
        }


def _create_default_transform_config(features):
    return {
        feature: _TransformConfig(transform_type=_TransformType.IDENTITY, params={})
        for feature in features
    }


feature_registry_stub.FeatureFamily = _FeatureFamily
feature_registry_stub.FeatureRegistry = _FeatureRegistry
feature_registry_stub.FeatureMetadata = _FeatureMetadata

transforms_stub.TransformType = _TransformType
transforms_stub.TransformConfig = _TransformConfig
transforms_stub.TransformRouter = _TransformRouter
transforms_stub.create_default_transform_config = _create_default_transform_config

feature_engineering_stub.FeatureFamily = _FeatureFamily
feature_engineering_stub.FeatureRegistry = _FeatureRegistry
feature_engineering_stub.TransformRouter = _TransformRouter
feature_engineering_stub.create_default_transform_config = _create_default_transform_config

feature_engineering_stub.feature_registry = feature_registry_stub
feature_engineering_stub.transforms = transforms_stub

sys.modules.setdefault("feature_engineering", feature_engineering_stub)
sys.modules.setdefault("feature_engineering.feature_registry", feature_registry_stub)
sys.modules.setdefault("feature_engineering.transforms", transforms_stub)
sys.modules.setdefault(
    "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_engineering",
    feature_engineering_stub,
)


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[4]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    parts = relative_path.replace("/", ".").split(".")
    package = ".".join(parts[:-2])
    package_parts = Path(relative_path).with_suffix("").parts
    for idx in range(len(package_parts) - 1):
        package_name = ".".join(package_parts[: idx + 1])
        package_dir = root / Path(*package_parts[: idx + 1])
        if package_name not in sys.modules:
            pkg_module = ModuleType(package_name)
            pkg_module.__path__ = [str(package_dir)]
            sys.modules[package_name] = pkg_module
    module.__package__ = package
    spec.loader.exec_module(module)
    return module


_phase1_probe = _load_module(
    "phase1_probe_module",
    "src/training/steps/pre_training/interaction_feature_generator/cross_timeframe_generation/phase1_probe.py",
)

FamilyProcessingError = _phase1_probe.FamilyProcessingError
Phase1HTFProbe = _phase1_probe.Phase1HTFProbe
AdaptiveScoringSystem = _phase1_probe.AdaptiveScoringSystem


@pytest.fixture
def phase1_with_scoring():
    config = SimpleNamespace(
        coarse_grid_min=15,
        coarse_grid_max=298,
        meta_learning_range=0.05,
        base_timeframe_minutes=5,
    )
    scoring_system = AdaptiveScoringSystem(config)
    probe = Phase1HTFProbe(config, scoring_system=scoring_system)
    return probe, scoring_system


def test_phase1_probe_uses_adaptive_scoring_penalties(phase1_with_scoring):
    probe, scoring_system = phase1_with_scoring

    metrics = {
        "ic_oos": 0.12,
        "se_wild_bootstrap": 0.05,
        "cpu_p95": 1.5,
        "staleness": 0.3,
    }

    baseline_score = probe._calculate_utility_score(**metrics)

    # Change penalty configuration through the adaptive scoring system
    scoring_system.meta_learner.lambda_unc = 0.2
    scoring_system.meta_learner.lambda_cost = 0.1
    scoring_system.meta_learner.lambda_stale = 0.15

    updated_score = probe._calculate_utility_score(**metrics)
    expected_score = scoring_system.calculate_utility_score(**metrics)

    assert updated_score == pytest.approx(expected_score)
    assert updated_score != pytest.approx(baseline_score)


def test_run_probe_stage_raises_when_all_candidates_fail(monkeypatch, phase1_with_scoring):
    probe, _ = phase1_with_scoring

    probe.htf_generator.htf_families = {"trend_level_vol": ["p/price_ema10_pct"]}
    monkeypatch.setattr(
        probe.grid_generator,
        "generate_adaptive_grid",
        lambda *args, **kwargs: [45],
    )

    def _fail_generate(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(probe.htf_generator, "generate_htf_feature", _fail_generate)

    sessionized_data = {
        "aligned_data": pd.DataFrame(
            {"p/price_ema10_pct": [1.0, 2.0, 3.0]},
            index=pd.date_range("2021-01-01", periods=3, freq="5min"),
        )
    }

    with pytest.raises(FamilyProcessingError) as excinfo:
        probe.run_probe_stage(sessionized_data, regime_segments={}, targets=None)

    assert "Failed to produce any valid candidates" in str(excinfo.value)
    assert excinfo.value.family == "trend_level_vol"
