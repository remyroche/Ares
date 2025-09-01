# src/transition/multitask_rf.py

from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
from src.utils.logger import system_logger
import json
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle

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

    def __init__(self, config: dict[str, Any], horizons: list[int]) -> None:
    pass
    pass
        self.logger = system_logger.getChild("MultiTaskRandomForest")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        mt = (
            tm.get("multitask_rf", {})
            if isinstance(tm.get("multitask_rf", {}), dict)
            else {}
        )
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
        self.models: dict[str, Any] = {}
        self.feature_names_: list[str] = []
        self.thresholds_: dict[str, Any] = {}
        self.reliability_: dict[str, Any] = {}

    def _assemble_X(self, samples: list[dict[str, Any]]) -> pd.DataFrame:
    pass
    pass
        rows: list[dict[str, float]] = []
        for s in samples:
    pass
    pass
            rf = dict(s.get("rf_features", {}))
            rows.append(rf)
        return pd.DataFrame(rows).fillna(0.0)

    def _cap(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    pass
    pass
        if len(X) > self.cfg.max_train_samples:
    pass
    pass
            return X.iloc[-self.cfg.max_train_samples :], y.iloc[
                -self.cfg.max_train_samples :
            ]
        return X, y

    def _best_f1_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
    pass
    pass
        if y_true.size == 0 or y_score.size == 0:
    pass
    pass
            return 0.5
        candidates = np.linspace(0.05, 0.95, 19)
        best_thr, best_f1 = 0.5, -1.0
        for thr in candidates:
    pass
    pass
            y_pred = (y_score >= thr).astype(int)
            try:
                f1 = f1_score(y_true, y_pred)
    except Exception as e:
        pass
    except Exception as e:
        pass
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
    pass
    pass
                best_f1, best_thr = f1, thr
        return float(best_thr)

    def fit(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
    pass
    pass
        if not self.cfg.enabled or not samples:
    pass
    pass
            return {"trained": False}
        X = self._assemble_X(samples)
        self.feature_names_ = list(X.columns)

        results: dict[str, Any] = {"trained": True}
        thresholds: dict[str, Any] = {}
        reliability: dict[str, Any] = {}

        # 1) Path class (multiclass)
        y_pc = pd.Series([str(s.get("path_class", "end_of_trend")) for s in samples])
        X_pc, y_pc = self._cap(X, y_pc)
        # FIXED: Use time-based split to prevent lookahead bias
        split_idx = int(len(X_pc) * 0.8)
        Xtr = X_pc.iloc[:split_idx]
        Xva = X_pc.iloc[split_idx:]
        ytr = y_pc.iloc[:split_idx]
        yva = y_pc.iloc[split_idx:]
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
            "report": classification_report(
                yva, pc_pred,
                output_dict=True, zero_division=0,
            ),
            "classes": list(pc_model.classes_),
        }
        # Reliability + thresholds per class (one-vs-rest)
        try:
            proba = pc_model.predict_proba(Xva)
    except Exception as e:
        pass
    except Exception as e:
        pass
            classes = list(pc_model.classes_)
            val_true = yva.values
            scales: dict[str , float] = {}
            thrs: dict[str , float] = {}
            for i, c in enumerate(classes):
    pass
    pass
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
    pass
    pass
            y = pd.Series([int(s.get(head, 0)) for s in samples])
            if y.nunique() < 2:
    pass
    pass
                continue
            Xh, yh = self._cap(X, y)
            # FIXED: Use time-based split to prevent lookahead bias
            split_idx = int(len(Xh) * 0.8)
            Xtr = Xh.iloc[:split_idx]
            Xva = Xh.iloc[split_idx:]
            ytr = yh.iloc[:split_idx]
            yva = yh.iloc[split_idx:]
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
            results[head] = {
                "report": classification_report(
                    yva, y_pred,
                    output_dict=True, zero_division=0,
                ),
            }
            try:
                p1 = clf.predict_proba(Xva)[:, 1]
    except Exception as e:
        pass
    except Exception as e:
        pass
                mean_p = float(np.clip(np.mean(p1), 1e-6, 1.0))
                mean_y = float(np.mean(yva.values))
                reliability[head] = {
                    "positive_scale": float(np.clip(mean_y / mean_p, 0.5, 1.5)),
                }
                thresholds[head] = float(
                    self._best_f1_threshold(yva.values.astype(int), p1),
                )
            except Exception:
                pass

        # 2b) Next regime head (multiclass): majority regime in Y_post_states
        try:
            regimes = []
    except Exception as e:
        pass
    except Exception as e:
        pass
            for s in samples:
    pass
    pass
                y_states = s.get("Y_post_states")
                if isinstance(y_states, pd.DataFrame) and "regime" in y_states.columns:
    pass
    pass
                    vals = [
                        str(v)
                        for v in y_states["regime"].tolist()
                        if isinstance(v, (str, int))
                    ]
                    if vals:
    pass
    pass
                        # majority label
                        regimes.append(Counter(vals).most_common(1)[0][0])
                    else:
                        regimes.append("SIDEWAYS")
                else:
                    regimes.append("SIDEWAYS")
            y_nr = pd.Series(regimes)
            if y_nr.nunique() >= 2:
    pass
    pass
                X_nr, y_nr = self._cap(X, y_nr)
                # FIXED: Use time-based split to prevent lookahead bias
                split_idx = int(len(X_nr) * 0.8)
                Xtr = X_nr.iloc[:split_idx]
                Xva = X_nr.iloc[split_idx:]
                ytr = y_nr.iloc[:split_idx]
                yva = y_nr.iloc[split_idx:]
                nr_model = RandomForestClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    min_samples_leaf=self.cfg.min_samples_leaf,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )
                nr_model.fit(Xtr, ytr)
                self.models["next_regime"] = nr_model
                results["next_regime"] = {
                    "report": classification_report(
                        yva, nr_model.predict(Xva),
                        output_dict=True, zero_division=0,
                    ),
                    "classes": list(nr_model.classes_),
                }
                # thresholds are not used for multiclass here; reliability scale
                try:
                    proba = nr_model.predict_proba(Xva)
    except Exception as e:
        pass
    except Exception as e:
        pass
                    classes = list(nr_model.classes_)
                    val_true = yva.values
                    scales: dict[str, float] = {}
                    for i, c in enumerate(classes):
    pass
    pass
                        p = proba[:, i].astype(float)
                        y_bin = (val_true == c).astype(int)
                        mean_p = float(np.clip(np.mean(p), 1e-6, 1.0))
                        mean_y = float(np.mean(y_bin))
                        scales[str(c)] = float(np.clip(mean_y / mean_p, 0.5, 1.5))
                    reliability["next_regime"] = scales
                except Exception:
                    pass
        except Exception:
            pass

        # 3) Direction heads per horizon (binary)
        for H in self.horizons:
    pass
    pass
            head = f"direction_up_{H}"
            y = pd.Series([int(s.get(head, 0)) for s in samples])
            if y.nunique() < 2:
    pass
    pass
                continue
            Xh, yh = self._cap(X, y)
            # FIXED: Use time-based split to prevent lookahead bias
            split_idx = int(len(Xh) * 0.8)
            Xtr = Xh.iloc[:split_idx]
            Xva = Xh.iloc[split_idx:]
            ytr = yh.iloc[:split_idx]
            yva = yh.iloc[split_idx:]
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
            results[head] = {
                "report": classification_report(
                    yva, y_pred,
                    output_dict=True, zero_division=0,
                ),
            }
            try:
                p1 = clf.predict_proba(Xva)[:, 1]
    except Exception as e:
        pass
    except Exception as e:
        pass
                mean_p = float(np.clip(np.mean(p1), 1e-6, 1.0))
                mean_y = float(np.mean(yva.values))
                reliability[head] = {
                    "positive_scale": float(np.clip(mean_y / mean_p, 0.5, 1.5)),
                }
                thresholds[head] = float(
                    self._best_f1_threshold(yva.values.astype(int), p1),
                )
            except Exception:
                pass

        # 4) Optional return regressors
        if self.cfg.enable_regression:
    pass
    pass
            for H in self.horizons:
    pass
    pass
                head = f"return_{H}"
                y = pd.Series([float(s.get(head, 0.0)) for s in samples])
                if y.empty:
    pass
    pass
                    continue
                Xh, yh = self._cap(X, y)
                # FIXED: Use time-based split to prevent lookahead bias
                split_idx = int(len(Xh) * 0.8)
                Xtr = Xh.iloc[:split_idx]
                Xva = Xh.iloc[split_idx:]
                ytr = yh.iloc[:split_idx]
                yva = yh.iloc[split_idx:]
                reg = RandomForestRegressor(
                    n_estimators=max(200, self.cfg.n_estimators // 2),
                    max_depth=self.cfg.max_depth, min_samples_leaf = self.cfg.min_samples_leaf,
                    random_state=self.cfg.random_state, n_jobs = -1,
                )
                reg.fit(Xtr, ytr)
                self.models[head] = reg
                pred = reg.predict(Xva)
                results[head] = {"mae": float(mean_absolute_error(yva, pred))}

        self.thresholds_ = thresholds
        self.reliability_ = reliability
        return results

    def save(self, models_dir: str, prefix: str = "rolling_mtrf") -> dict[str, Any]:
    pass
    pass
        os.makedirs(models_dir, exist_ok=True)
        saved: dict[str, str] = {}
        # Save each model
        for name, model in self.models.items():
    pass
    pass
            path = os.path.join(models_dir, f"{prefix}_{name}.pkl")
            try:
                with open(path, "wb") as f:
                    pickle.dump(model, f)
    except Exception as e:
        pass
    except Exception as e:
        pass
                saved[name] = path
            except Exception as e:
                self.logger.warning(f"Failed to save model {name}: {e}")
        # Save metadata
        meta = {
            "feature_names": self.feature_names_,
            "heads": list(self.models.keys()),
        }
        try:
            meta_path = os.path.join(models_dir, f"{prefix}_meta.json")
    except Exception as e:
        pass
    except Exception as e:
        pass
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save meta: {e}")
            meta_path = os.path.join(models_dir, f"{prefix}_meta.json")
        # Save thresholds and reliability for inference
        try:
            thr_path = os.path.join(models_dir, "thresholds.json")
    except Exception as e:
        pass
    except Exception as e:
        pass
            with open(thr_path, "w", encoding="utf-8") as f:
                json.dump(self.thresholds_, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save thresholds: {e}")
        try:
            rel_path = os.path.join(models_dir, "reliability.json")
    except Exception as e:
        pass
    except Exception as e:
        pass
            with open(rel_path, "w", encoding="utf-8") as f:
                json.dump(self.reliability_, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save reliability: {e}")
        return {
            "models": saved,
            "meta_path": os.path.join(models_dir, f"{prefix}_meta.json"),
        }

    @staticmethod
    def load(models_dir: str, prefix: str = "rolling_mtrf") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    pass
    pass
        models: dict[str, Any] = {}
        # Load models
        for fname in os.listdir(models_dir):
    pass
    pass
            if fname.startswith(prefix + "_") and fname.endswith(".pkl"):
    pass
    pass
                head = fname[len(prefix) + 1 : -4]
                try:
                    with open(os.path.join(models_dir, fname), "rb") as f:
                        models[head] = pickle.load(f)
    except Exception as e:
        pass
    except Exception as e:
        pass
                except Exception:
                    continue
        # Load thresholds and reliability
        thresholds: dict[str, Any] = {}
        reliability: dict[str, Any] = {}
        try:
            with open(os.path.join(models_dir, "thresholds.json"), encoding="utf-8") as f:
                thresholds = json.load(f)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        try:
            with open(os.path.join(models_dir, "reliability.json"), encoding="utf-8") as f:
                reliability = json.load(f)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        # Load feature names
        feature_names: list[str] = []
        try:
            with open(os.path.join(models_dir, f"{prefix}_meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
                feature_names = list(meta.get("feature_names", []))
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        return models, thresholds, reliability, feature_names

    def predict(self, X: pd.DataFrame) -> dict[str, Any]:
    pass
    pass
        out: dict[str, Any] = {}
        for name, model in self.models.items():
    pass
    pass
            try:
                if hasattr(model, "predict_proba"):
    pass
    except Exception as e:
        pass
    pass
                    proba = model.predict_proba(X)
                    classes = getattr(model, "classes_", [])
                    out[name] = {str(c): proba[:, i].tolist() for i, c in enumerate(classes)}
    except Exception as e:
        pass
                else:
                    out[name] = list(map(float, model.predict(X).tolist()))
            except Exception as e:
                self.logger.warning(
                    f"Prediction failed for model '{name}': {e}",
                    exc_info=True,
                )
                out[name] = []
        return out
