# src/transition/multitask_rf.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, f1_score

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
        self.feature_names_: List[str] = []
        self.thresholds_: Dict[str, Any] = {}
        self.reliability_: Dict[str, Any] = {}

    def _assemble_X(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for s in samples:
            rf = dict(s.get("rf_features", {}))
            rows.append(rf)
        X = pd.DataFrame(rows).fillna(0.0)
        return X

    def _cap(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) > self.cfg.max_train_samples:
            return X.iloc[: self.cfg.max_train_samples], y.iloc[: self.cfg.max_train_samples]
        return X, y

    def _best_f1_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        if y_true.size == 0 or y_score.size == 0:
            return 0.5
        candidates = np.linspace(0.05, 0.95, 19)
        best_thr, best_f1 = 0.5, -1.0
        for thr in candidates:
            y_pred = (y_score >= thr).astype(int)
            try:
                f1 = f1_score(y_true, y_pred)
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        return float(best_thr)

    def fit(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.cfg.enabled or not samples:
            return {"trained": False}
        X = self._assemble_X(samples)
        self.feature_names_ = list(X.columns)

        results: Dict[str, Any] = {"trained": True}
        thresholds: Dict[str, Any] = {}
        reliability: Dict[str, Any] = {}

        # 1) Path class (multiclass)
        y_pc = pd.Series([str(s.get("path_class", "end_of_trend")) for s in samples])
        X_pc, y_pc = self._cap(X, y_pc)
        Xtr, Xva, ytr, yva = train_test_split(
            X_pc, y_pc, test_size=0.2, random_state=self.cfg.random_state, stratify=y_pc
        )
        pc_model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=-1,
        )
        pc_model.fit(Xtr, ytr)
        self.models["path_class"] = pc_model
        # Eval
        pc_pred = pc_model.predict(Xva)
        results["path_class"] = {
            "report": classification_report(yva, pc_pred, output_dict=True, zero_division=0),
            "classes": list(pc_model.classes_),
        }
        # Reliability + thresholds per class (one-vs-rest)
        try:
            proba = pc_model.predict_proba(Xva)
            classes = list(pc_model.classes_)
            val_true = yva.values
            scales: Dict[str, float] = {}
            thrs: Dict[str, float] = {}
            for i, c in enumerate(classes):
                p = proba[:, i].astype(float)
                y_bin = (val_true == c).astype(int)
                mean_p = float(np.clip(np.mean(p), 1e-6, 1.0))
                mean_y = float(np.mean(y_bin))
                scales[str(c)] = float(np.clip(mean_y / mean_p, 0.5, 1.5))
                thrs[str(c)] = self._best_f1_threshold(y_bin, p)
            reliability["path_class"] = scales
            thresholds["path_class"] = thrs
        except Exception:
            pass

        # 2) Onset / End heads (binary)
        for head in ("onset_beginning", "end_trend"):
            y = pd.Series([int(s.get(head, 0)) for s in samples])
            if y.nunique() < 2:
                continue
            Xh, yh = self._cap(X, y)
            Xtr, Xva, ytr, yva = train_test_split(
                Xh, yh, test_size=0.2, random_state=self.cfg.random_state, stratify=yh
            )
            clf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                min_samples_leaf=self.cfg.min_samples_leaf,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )
            clf.fit(Xtr, ytr)
            self.models[head] = clf
            y_pred = clf.predict(Xva)
            results[head] = {"report": classification_report(yva, y_pred, output_dict=True, zero_division=0)}
            try:
                p1 = clf.predict_proba(Xva)[:, 1]
                mean_p = float(np.clip(np.mean(p1), 1e-6, 1.0))
                mean_y = float(np.mean(yva.values))
                reliability[head] = {"positive_scale": float(np.clip(mean_y / mean_p, 0.5, 1.5))}
                thresholds[head] = float(self._best_f1_threshold(yva.values.astype(int), p1))
            except Exception:
                pass

        # 3) Direction heads per horizon (binary)
        for H in self.horizons:
            head = f"direction_up_{H}"
            y = pd.Series([int(s.get(head, 0)) for s in samples])
            if y.nunique() < 2:
                continue
            Xh, yh = self._cap(X, y)
            Xtr, Xva, ytr, yva = train_test_split(
                Xh, yh, test_size=0.2, random_state=self.cfg.random_state, stratify=yh
            )
            clf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                min_samples_leaf=self.cfg.min_samples_leaf,
                random_state=self.cfg.random_state,
                n_jobs=-1,
            )
            clf.fit(Xtr, ytr)
            self.models[head] = clf
            y_pred = clf.predict(Xva)
            results[head] = {"report": classification_report(yva, y_pred, output_dict=True, zero_division=0)}
            try:
                p1 = clf.predict_proba(Xva)[:, 1]
                mean_p = float(np.clip(np.mean(p1), 1e-6, 1.0))
                mean_y = float(np.mean(yva.values))
                reliability[head] = {"positive_scale": float(np.clip(mean_y / mean_p, 0.5, 1.5))}
                thresholds[head] = float(self._best_f1_threshold(yva.values.astype(int), p1))
            except Exception:
                pass

        # 4) Optional return regressors
        if self.cfg.enable_regression:
            for H in self.horizons:
                head = f"return_{H}"
                y = pd.Series([float(s.get(head, 0.0)) for s in samples])
                if y.empty:
                    continue
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

        self.thresholds_ = thresholds
        self.reliability_ = reliability
        return results

    def save(self, models_dir: str, prefix: str = "rolling_mtrf") -> dict[str, Any]:
        os.makedirs(models_dir, exist_ok=True)
        saved: Dict[str, str] = {}
        # Save each model
        for name, model in self.models.items():
            path = os.path.join(models_dir, f"{prefix}_{name}.pkl")
            try:
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                saved[name] = path
            except Exception as e:
                self.logger.warning(f"Failed to save model {name}: {e}")
        # Save metadata
        meta = {
            "feature_names": self.feature_names_,
            "heads": list(self.models.keys()),
        }
        try:
            with open(os.path.join(models_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save meta: {e}")
        # Save thresholds and reliability for inference
        try:
            with open(os.path.join(models_dir, "thresholds.json"), "w") as f:
                json.dump(self.thresholds_, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save thresholds: {e}")
        try:
            with open(os.path.join(models_dir, "reliability.json"), "w") as f:
                json.dump(self.reliability_, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save reliability: {e}")
        return {"models": saved, "meta_path": os.path.join(models_dir, f"{prefix}_meta.json")}

    @staticmethod
    def load(models_dir: str, prefix: str = "rolling_mtrf") -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        models: Dict[str, Any] = {}
        # Load models
        for fname in os.listdir(models_dir):
            if fname.startswith(prefix + "_") and fname.endswith(".pkl"):
                head = fname[len(prefix) + 1 : -4]
                try:
                    with open(os.path.join(models_dir, fname), "rb") as f:
                        models[head] = pickle.load(f)
                except Exception:
                    continue
        # Load thresholds and reliability
        thresholds: Dict[str, Any] = {}
        reliability: Dict[str, Any] = {}
        try:
            with open(os.path.join(models_dir, "thresholds.json"), "r") as f:
                thresholds = json.load(f)
        except Exception:
            pass
        try:
            with open(os.path.join(models_dir, "reliability.json"), "r") as f:
                reliability = json.load(f)
        except Exception:
            pass
        # Load feature names
        feature_names: List[str] = []
        try:
            with open(os.path.join(models_dir, f"{prefix}_meta.json"), "r") as f:
                meta = json.load(f)
                feature_names = list(meta.get("feature_names", []))
        except Exception:
            pass
        return models, {"thresholds": thresholds, "reliability": reliability}, feature_names

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