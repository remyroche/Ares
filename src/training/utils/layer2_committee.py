"""Layer 2 committee wrapper for managing multiple base classifiers.

This module defines a small, self-contained `Layer2Committee` that owns a
fixed set of base models (e.g., KNN + LightGBM variants) and exposes a
unified interface:

    - fit(X, y)
    - predict_proba(X)
    - predict(X)
    - save(path) / load(path)

It intentionally avoids pulling in the full enhanced training stack; it
is meant to be a lightweight building block that higher-level training
steps can orchestrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from src.utils.logger import system_logger


logger = system_logger.getChild("Layer2Committee")


@dataclass
class Layer2CommitteeConfig:
    """Configuration for the Layer 2 committee.

    `base_model_defs` is a mapping from model name to a tuple of
    (estimator_class, init_kwargs).
    """

    base_model_defs: Dict[str, Tuple[type, Dict]] = field(
        default_factory=lambda: {
            "knn_short": (KNeighborsClassifier, {"n_neighbors": 25, "weights": "distance"}),
            "knn_long": (KNeighborsClassifier, {"n_neighbors": 50, "weights": "distance"}),
            "lgbm_short": (
                LGBMClassifier,
                {
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "num_leaves": 48,
                    "max_depth": -1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
            ),
            "lgbm_long": (
                LGBMClassifier,
                {
                    "n_estimators": 600,
                    "learning_rate": 0.03,
                    "num_leaves": 64,
                    "max_depth": -1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                },
            ),
        }
    )


class Layer2Committee:
    """Small committee of base classifiers with simple aggregation.

    Aggregation rule is currently a simple average of per-model
    `predict_proba` outputs for the positive class.
    """

    def __init__(self, config: Optional[Layer2CommitteeConfig] = None) -> None:
        self.config = config or Layer2CommitteeConfig()
        self.models: Dict[str, object] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Layer2Committee":
        """Fit all committee members on the same (X, y)."""
        self.models.clear()
        for name, (cls, kwargs) in self.config.base_model_defs.items():
            est = cls(**kwargs)
            logger.info(f"[Layer2Committee] Fitting member '{name}' ({cls.__name__}) on {len(X)} samples")
            est.fit(X.values, y.values)
            self.models[name] = est
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return aggregated class probabilities.

        Returns array of shape (n_samples, 2) for binary classification.
        """
        if not self._fitted or not self.models:
            raise RuntimeError("Layer2Committee must be fitted before calling predict_proba")

        probas: List[np.ndarray] = []
        for name, est in self.models.items():
            if not hasattr(est, "predict_proba"):
                raise TypeError(f"Committee member '{name}' does not implement predict_proba")
            p = est.predict_proba(X.values)
            if p.shape[1] != 2:
                raise ValueError(f"Committee member '{name}' must be binary classifier (got {p.shape[1]} classes)")
            probas.append(p)

        # Simple average ensemble
        stacked = np.stack(probas, axis=0)  # (n_models, n_samples, 2)
        avg = stacked.mean(axis=0)
        return avg

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions from aggregated probabilities."""
        proba = self.predict_proba(X)
        pos = proba[:, 1]
        return (pos >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, base_path: Path) -> None:
        """Save all committee members and config under `base_path`.

        Layout:
            base_path/
                config.joblib
                member_knn_short.joblib
                member_knn_long.joblib
                member_lgbm_short.joblib
                member_lgbm_long.joblib
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.config, base_path / "config.joblib")
        for name, est in self.models.items():
            joblib.dump(est, base_path / f"member_{name}.joblib")

    @classmethod
    def load(cls, base_path: Path) -> "Layer2Committee":
        """Load committee and members from `base_path`."""
        base_path = Path(base_path)
        config_path = base_path / "config.joblib"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing committee config at {config_path}")

        config: Layer2CommitteeConfig = joblib.load(config_path)
        committee = cls(config=config)

        for name in config.base_model_defs.keys():
            member_path = base_path / f"member_{name}.joblib"
            if not member_path.exists():
                raise FileNotFoundError(f"Missing committee member '{name}' at {member_path}")
            committee.models[name] = joblib.load(member_path)

        committee._fitted = True
        return committee
