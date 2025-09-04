#!/usr/bin/env python3
"""
Lightweight import smoke test.

Usage:
  python tools/import_smoke_test.py            # regular import test
  python tools/import_smoke_test.py --stub     # stub heavy deps (numpy, pandas, scipy, numba, optuna, sklearn)

Exits with non-zero if any import or YAML parse fails.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from types import SimpleNamespace


def _maybe_stub_heavy_deps() -> None:
    import types

    def stub(name: str, obj: object | None = None) -> None:
        if name in sys.modules:
            return
        sys.modules[name] = obj if obj is not None else types.SimpleNamespace()

    # numpy
    import math
    stub(
        "numpy",
        SimpleNamespace(
            array=lambda x, dtype=None: x,
            mean=lambda x, axis=None: (sum(x) / len(x) if hasattr(x, "__len__") and len(x) > 0 else 0.0),
            std=lambda x, axis=None: 0.0,
            clip=lambda x, a, b: max(a, min(b, x)),
            log=lambda x: math.log(x) if x else 0.0,
            exp=lambda x: math.e ** x if x else 1.0,
            min=lambda seq: min(seq) if seq else 0.0,
            max=lambda seq: max(seq) if seq else 0.0,
        ),
    )
    # pandas
    stub("pandas", SimpleNamespace(DataFrame=object, Series=object))
    # numba
    stub("numba", SimpleNamespace(njit=(lambda *a, **k: (lambda f: f))))
    # optuna
    stub("optuna", SimpleNamespace(create_study=lambda **k: SimpleNamespace(optimize=lambda *a, **k: None)))
    # scipy.stats
    stats = SimpleNamespace(norm=SimpleNamespace(cdf=lambda x: 0.5))
    sys.modules["scipy"] = SimpleNamespace(stats=stats)
    sys.modules["scipy.stats"] = stats
    # sklearn
    sys.modules.setdefault("sklearn", SimpleNamespace())
    sys.modules.setdefault("sklearn.metrics", SimpleNamespace(accuracy_score=lambda *_a, **_k: 0.0))
    sys.modules.setdefault("sklearn.model_selection", SimpleNamespace(train_test_split=lambda *a, **k: a))
    sys.modules.setdefault(
        "sklearn.preprocessing",
        SimpleNamespace(StandardScaler=type("StandardScaler", (), {"fit_transform": lambda self, X: X})),
    )
    sys.modules.setdefault(
        "sklearn.linear_model",
        SimpleNamespace(
            LogisticRegression=type(
                "LogisticRegression",
                (),
                {"fit": (lambda self, X, y: None), "predict_proba": (lambda self, X: [[0.5, 0.5] for _ in range(len(X))])},
            )
        ),
    )
    sys.modules.setdefault(
        "sklearn.ensemble",
        SimpleNamespace(
            RandomForestClassifier=type(
                "RandomForestClassifier",
                (),
                {"fit": (lambda self, X, y: None), "predict_proba": (lambda self, X: [[0.5, 0.5] for _ in range(len(X))])},
            )
        ),
    )
    sys.modules.setdefault(
        "sklearn.svm",
        SimpleNamespace(SVC=type("SVC", (), {"fit": (lambda self, X, y: None), "decision_function": (lambda self, X: [0.0 for _ in range(len(X))])})),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="Stub heavy dependencies to allow imports without installing")
    parser.add_argument("--base", default="/workspace", help="Project base path (default: /workspace)")
    args = parser.parse_args()

    if args.stub:
        _maybe_stub_heavy_deps()

    # Prefer workspace base in sys.path
    if args.base and os.path.isdir(args.base):
        sys.path.insert(0, args.base)

    print("Import smoke tests:")

    errs = 0
    # YAML parsing
    yaml_paths = [
        "config.yaml",
        "src/config/step12_confidence_optimization.yaml",
        "src/config/step17_optimization_structure.yaml",
    ]
    try:
        import yaml as yaml_mod  # type: ignore
        have_yaml = True
    except Exception:
        have_yaml = False

    for rel in yaml_paths:
        try:
            p = rel if os.path.isabs(rel) else os.path.join(args.base, rel)
            if not os.path.exists(p):
                print("YAML MISSING:", rel)
                errs += 1
                continue
            if have_yaml:
                with open(p, "r", encoding="utf-8") as f:
                    yaml_mod.safe_load(f)
                print("YAML OK:", rel)
            else:
                print("YAML EXISTS (parser missing):", rel)
        except Exception as e:
            print("YAML FAIL:", rel, e)
            errs += 1

    # Modules to import
    modules = [
        "src.analyst.analyst",
        "src.analyst.feature_engineering_orchestrator",
        "src.analyst.ml_confidence_predictor",
        "src.analytics.bayesian_probability_updates",
        "src.analytics.copula_dependency_models",
        "src.analytics.limited_microstructure_features",
        "src.analytics.performance_attribution",
        "src.config.config_tpsl",
        "src.optimization.hmm_regime_ab_testing",
        "src.optimization.ml_optimized_barriers",
        "src.tactician.leverage_sizer",
        "src.tactician.ml_tactics_manager",
        "src.tactician.position_sizer",
        "src.tactician.tactician",
        "src.training.core",
        "src.training.core.training_manager",
        "src.tactician.async_order_executor",
        "src.tactician.enhanced_order_manager",
        "src.supervisor.performance_reporter",
        "src.config",
        "src.config.system",
    ]

    for mod in modules:
        try:
            importlib.import_module(mod)
            print("OK:", mod)
        except Exception as e:  # noqa: BLE001
            print("FAIL:", mod, e)
            errs += 1

    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())

