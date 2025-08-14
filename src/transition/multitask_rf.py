# src/transition/multitask_rf.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

from src.utils.logger import system_logger


@dataclass
class MTRFConfig:
    enabled: bool
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    random_state: int
    max_train_samples: int
    enable_regression: bool


class MultiTaskRandomForest:
    """
    Simple multi-head trainer built on RF:
    - path_class head: multiclass {beginning_of_trend, continuation, reversal, end_of_trend}
    - onset_beginning head: binary
    - end_trend head: binary
    - direction heads: one per horizon H (binary up/down)
    - return heads: one per horizon H (regression, optional)
    """

    def __init__(self, config: dict[str, Any], horizons: List[int]) -> None:
        self.logger = system_logger.getChild("MultiTaskRandomForest")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        mt = tm.get("multitask_rf", {}) if isinstance(tm.get("multitask_rf", {}), dict) else {}
        self.cfg = MTRFConfig(
            enabled=bool(mt.get("enabled", True)),
            n_estimators=int(mt.get("n_estimators", 400)),
            max_depth=int(mt.get("max_depth", 14)),
            min_samples_leaf=int(mt.get("min_samples_leaf", 5)),
            random_state=int(mt.get("random_state", 42)),
            max_train_samples=int(mt.get("max_train_samples", 300000)),
            enable_regression=bool(mt.get("enable_regression", True)),
        )
        self.horizons = list(horizons)
        self.models: Dict[str, Any] = {}

    def _assemble_X(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for s in samples:
            rf = dict(s.get("rf_features", {}))
            # include path-class one-hot from previous bar if available (simple lag context)
            pc = str(s.get("path_class", "end_of_trend"))
            rf[f"pc_is_{pc}"] = 1.0
            rows.append(rf)
        X = pd.DataFrame(rows).fillna(0.0)
        return X

    def _cap(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) > self.cfg.max_train_samples:
            return X.iloc[: self.cfg.max_train_samples], y.iloc[: self.cfg.max_train_samples]
        return X, y

    def fit(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.cfg.enabled or not samples:
            return {"trained": False}
        X = self._assemble_X(samples)

        results: Dict[str, Any] = {"trained": True}

        # 1) Path class
        y_pc = pd.Series([str(s.get("path_class", "end_of_trend")) for s in samples])
        X_pc, y_pc = self._cap(X, y_pc)
        Xtr, Xva, ytr, yva = train_test_split(X_pc, y_pc, test_size=0.2, random_state=self.cfg.random_state, stratify=y_pc)
        pc_model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=-1,
        )
        pc_model.fit(Xtr, ytr)
        results["path_class"] = {
            "report": classification_report(yva, pc_model.predict(Xva), output_dict=True, zero_division=0),
            "classes": list(pc_model.classes_),
        }
        self.models["path_class"] = pc_model

        # 2) Onset / End heads
        for head in ("onset_beginning", "end_trend"):
            y = pd.Series([int(s.get(head, 0)) for s in samples])
            Xh, yh = self._cap(X, y)
            Xtr, Xva, ytr, yva = train_test_split(Xh, yh, test_size=0.2, random_state=self.cfg.random_state, stratify=yh)
            clf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                min_samples_leaf=self.cfg.min_samples_leaf,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )
            clf.fit(Xtr, ytr)
            self.models[head] = clf
            results[head] = {"report": classification_report(yva, clf.predict(Xva), output_dict=True, zero_division=0)}

        # 3) Direction heads per horizon
        for H in self.horizons:
            head = f"direction_up_{H}"
            y = pd.Series([int(s.get(head, 0)) for s in samples])
            Xh, yh = self._cap(X, y)
            Xtr, Xva, ytr, yva = train_test_split(Xh, yh, test_size=0.2, random_state=self.cfg.random_state, stratify=yh)
            clf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                min_samples_leaf=self.cfg.min_samples_leaf,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )
            clf.fit(Xtr, ytr)
            self.models[head] = clf
            results[head] = {"report": classification_report(yva, clf.predict(Xva), output_dict=True, zero_division=0)}

        # 4) Optional return regressors
        if self.cfg.enable_regression:
            for H in self.horizons:
                head = f"return_{H}"
                y = pd.Series([float(s.get(head, 0.0)) for s in samples])
                Xh, yh = self._cap(X, y)
                Xtr, Xva, ytr, yva = train_test_split(Xh, yh, test_size=0.2, random_state=self.cfg.random_state)
                reg = RandomForestRegressor(
                    n_estimators=max(200, self.cfg.n_estimators // 2),
                    max_depth=self.cfg.max_depth,
                    min_samples_leaf=self.cfg.min_samples_leaf,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )
                reg.fit(Xtr, ytr)
                self.models[head] = reg
                pred = reg.predict(Xva)
                results[head] = {"mae": float(mean_absolute_error(yva, pred))}

        return results

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    classes = getattr(model, "classes_", [])
                    out[name] = {str(c): proba[:, i].tolist() for i, c in enumerate(classes)}
                else:
                    out[name] = model.predict(X).tolist()
            except Exception:
                out[name] = []
        return out