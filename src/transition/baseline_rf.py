# src/transition/baseline_rf.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

from src.utils.logger import system_logger


@dataclass
class RFConfig:
    enabled: bool
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    random_state: int
    max_train_samples: int
    enable_shap: bool


class TransitionRandomForest:
    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = system_logger.getChild("TransitionRandomForest")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        rfc = tm.get("baseline_random_forest", {})
        self.cfg = RFConfig(
            enabled=bool(rfc.get("enabled", True)),
            n_estimators=int(rfc.get("n_estimators", 300)),
            max_depth=int(rfc.get("max_depth", 12)),
            min_samples_leaf=int(rfc.get("min_samples_leaf", 5)),
            random_state=int(rfc.get("random_state", 42)),
            max_train_samples=int(rfc.get("max_train_samples", 200000)),
            enable_shap=bool(tm.get("enable_shap", True)),
        )
        self.model: RandomForestClassifier | None = None
        self.label_names: List[str] = []
        self.feature_names_: List[str] = []

    def _assemble_features(self, samples: list[dict[str, Any]], label_index: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        rows: list[dict[str, Any]] = []
        y: list[str] = []
        for s in samples:
            rf = dict(s.get("rf_features", {}))
            # attach multi-hot context as features
            mh = np.array(s.get("multi_hot_labels"), dtype=float)
            for i, lab in enumerate(label_index):
                rf[f"ctx_label_{lab}"] = float(mh[i] if i < len(mh) else 0.0)
            # add event anchor
            rf["anchor_label"] = s.get("event_label", "")
            # encode anchor as one-hot sparse
            rf[f"anchor_is_{s.get('event_label','')}"] = 1.0
            rows.append(rf)
        # Targets stored temporarily in each sample under 'path_class'
        for s in samples:
            y.append(str(s.get("path_class", "end_of_trend")))
        X = pd.DataFrame(rows)
        # fill missing with 0 for RF
        X = X.fillna(0)
        return X, pd.Series(y)

    def fit(self, samples: list[dict[str, Any]], label_index: list[str]) -> dict[str, Any]:
        if not self.cfg.enabled or not samples:
            return {"trained": False}
        X, y = self._assemble_features(samples, label_index)
        # cap size for speed
        if len(X) > self.cfg.max_train_samples:
            X = X.iloc[: self.cfg.max_train_samples]
            y = y.iloc[: self.cfg.max_train_samples]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.cfg.random_state, stratify=y)
        mdl = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=-1,
        )
        mdl.fit(X_train, y_train)
        self.model = mdl
        self.feature_names_ = list(X.columns)
        # Eval
        y_pred = mdl.predict(X_val)
        rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        result = {"trained": True, "report": rep}
        # SHAP (optional)
        if self.cfg.enable_shap and shap is not None:
            try:
                explainer = shap.TreeExplainer(mdl)
                # Sample a subset for SHAP speed
                ns = min(2000, len(X_val))
                shap_vals = explainer.shap_values(X_val.iloc[:ns])
                # summarize mean |shap| per feature
                if isinstance(shap_vals, list):
                    # multiclass returns list per class
                    abs_mean = np.mean([np.abs(v).mean(axis=0) for v in shap_vals], axis=0)
                else:
                    abs_mean = np.abs(shap_vals).mean(axis=0)
                top_idx = np.argsort(-abs_mean)[:50]
                top_features = {self.feature_names_[i]: float(abs_mean[i]) for i in top_idx}
                result["shap_top_features"] = top_features
            except Exception as e:
                self.logger.warning(f"SHAP computation failed/skipped: {e}")
        return result