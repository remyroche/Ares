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
class PlaceholderDataClass:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="mtrfconfig initialization",
    )
    async def initialize(self) -> bool:
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="multitaskrandomforest initialization",
    )
    async def initialize(self) -> bool:
        """Initialize MultiTaskRandomForest."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    """Initialize MTRFConfig."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpass  # TODO: Add implementation
class MTRFConfig:
    passpass  # TODO: Add implementation
class MTRFConfig:
    passpass  # TODO: Add implementation
class MTRFConfig:
    passenabled: bool
n_estimators: int
max_depth: int | None
min_samples_leaf: int
random_state: int
max_train_samples: int
enable_regression: bool

class MultiTaskRandomForest:
    passpass  # TODO: Add implementation
class MultiTaskRandomForest:
    passpass  # TODO: Add implementation
class MultiTaskRandomForest:
    pass"""
Simple multi-head trainer built on RF:
    - path_class head: multiclass {beginning_of_trend, continuation, reversal, end_of_trend}
- onset_beginning head: binary
- end_trend head: binary
- direction heads: one per horizon H (binary up/down)
- return heads: one per horizon H (regression, optional)
"""

def __init__(self, config: dict[str, Any], horizons: list[int]) -> None:
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
        rows: list[dict[str, float]] = []
for s in samples:
    passrf = dict(s.get("rf_features", {}))
rows.append(rf)
return pd.DataFrame(rows).fillna(0.0)

def _cap(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if len(X) > self.cfg.max_train_samples:
    passreturn X.iloc[-self.cfg.max_train_samples :], y.iloc[
-self.cfg.max_train_samples :
            ]
return X, y

def _best_f1_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        if y_true.size == 0 or y_score.size == 0:
    passreturn 0.5
candidates = np.linspace(0.05, 0.95, 19)
best_thr, best_f1 = 0.5, -1.0
for thr in candidates:
    passy_pred = (y_score >= thr).astype(int)
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
f1 = f1_score(y_true, y_pred)
except Exception:
    passpassf1 = 0.0
if f1 > best_f1:
    passbest_f1, best_thr = f1, thr
return float(best_thr)

def fit(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.cfg.enabled or not samples:
    passreturn {"trained": False}
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
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
proba = pc_model.predict_proba(Xva)
classes = list(pc_model.classes_)
val_true = yva.values
scales: dict[str , float] = {}
thrs: dict[str , float] = {}
for i, c in enumerate(classes):
    passp = proba[:, i].astype(float)
y_bin = (val_true == c).astype(int)
mean_p = float(np.clip(np.mean(p), 1e-6, 1.0))
mean_y = float(np.mean(y_bin))
scales[str(c)] = float(np.clip(mean_y / mean_p, 0.5, 1.5))
thrs[str(c)] = self._best_f1_threshold(y_bin, p)
reliability["path_class"] = scales
thresholds["path_class"] = thrs
except Exception:
    passpasspass

# 2) Onset / End heads (binary)
for head in ("onset_beginning", "end_trend"):
    passy = pd.Series([int(s.get(head, 0)) for s in samples])
if y.nunique() < 2:
    passpasscontinue
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
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
p1 = clf.predict_proba(Xva)[:, 1]
mean_p = float(np.clip(np.mean(p1), 1e-6, 1.0))
mean_y = float(np.mean(yva.values))
reliability[head] = {
"positive_scale": float(np.clip(mean_y / mean_p, 0.5, 1.5)),
}
thresholds[head] = float(
self._best_f1_threshold(yva.values.astype(int), p1),
)
except Exception:
    passpasspass

# 2b) Next regime head (multiclass): majority regime in Y_post_states
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
regimes = []
for s in samples:
    passy_states = s.get("Y_post_states")
if isinstance(y_states, pd.DataFrame) and "regime" in y_states.columns:
    passvals = [
str(v)
for v in y_states["regime"].tolist()
if isinstance(v, (str, int))
]
if vals:
    passpass# majority label
regimes.append(Counter(vals).most_common(1)[0][0])
else:
    passregimes.append("SIDEWAYS")
else:
    passregimes.append("SIDEWAYS")
y_nr = pd.Series(regimes)
if y_nr.nunique() >= 2:
    passX_nr, y_nr = self._cap(X, y_nr)
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
    passpasspass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
proba = nr_model.predict_proba(Xva)
classes = list(nr_model.classes_)
val_true = yva.values
scales: dict[str, float] = {}
for i, c in enumerate(classes):
    passp = proba[:, i].astype(float)
y_bin = (val_true == c).astype(int)
mean_p = float(np.clip(np.mean(p), 1e-6, 1.0))
mean_y = float(np.mean(y_bin))
scales[str(c)] = float(np.clip(mean_y / mean_p, 0.5, 1.5))
reliability["next_regime"] = scales
except Exception:
    passpasspass
except Exception:
    passpasspass

# 3) Direction heads per horizon (binary)
for H in self.horizons:
    passhead = f"direction_up_{H}"
y = pd.Series([int(s.get(head, 0)) for s in samples])
if y.nunique() < 2:
    passpasscontinue
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
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
p1 = clf.predict_proba(Xva)[:, 1]
mean_p = float(np.clip(np.mean(p1), 1e-6, 1.0))
mean_y = float(np.mean(yva.values))
reliability[head] = {
"positive_scale": float(np.clip(mean_y / mean_p, 0.5, 1.5)),
}
thresholds[head] = float(
self._best_f1_threshold(yva.values.astype(int), p1),
)
except Exception:
    passpasspass

# 4) Optional return regressors
if self.cfg.enable_regression:
    passfor H in self.horizons:
    passhead = f"return_{H}"
y = pd.Series([float(s.get(head, 0.0)) for s in samples])
if y.empty:
    passpasscontinue
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
        os.makedirs(models_dir, exist_ok=True)
saved: dict[str, str] = {}
# Save each model
for name, model in self.models.items():
    passpath = os.path.join(models_dir, f"{prefix}_{name}.pkl")
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
with open(path, "wb") as f:
    passpickle.dump(model, f)
saved[name] = path
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Failed to save model {name}: {e}")
# Save metadata
meta = {
"feature_names": self.feature_names_,
"heads": list(self.models.keys()),
}
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
meta_path = os.path.join(models_dir, f"{prefix}_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    passjson.dump(meta, f, indent=2)
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Failed to save meta: {e}")
meta_path = os.path.join(models_dir, f"{prefix}_meta.json")
# Save thresholds and reliability for inference
try:
    passpasspass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
thr_path = os.path.join(models_dir, "thresholds.json")
with open(thr_path, "w", encoding="utf-8") as f:
    passjson.dump(self.thresholds_, f, indent=2)
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Failed to save thresholds: {e}")
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
rel_path = os.path.join(models_dir, "reliability.json")
with open(rel_path, "w", encoding="utf-8") as f:
    passjson.dump(self.reliability_, f, indent=2)
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Failed to save reliability: {e}")
return {
"models": saved,
"meta_path": os.path.join(models_dir, f"{prefix}_meta.json"),
}

@staticmethod
def load(models_dir: str, prefix: str = "rolling_mtrf") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
        models: dict[str, Any] = {}
# Load models
for fname in os.listdir(models_dir):
    passif fname.startswith(prefix + "_") and fname.endswith(".pkl"):
    passhead = fname[len(prefix) + 1 : -4]
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
with open(os.path.join(models_dir, fname), "rb") as f:
    passmodels[head] = pickle.load(f)
except Exception:
    passpasscontinue
# Load thresholds and reliability
thresholds: dict[str, Any] = {}
reliability: dict[str, Any] = {}
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
with open(os.path.join(models_dir, "thresholds.json"), encoding="utf-8") as f:
    passthresholds = json.load(f)
except Exception:
    passpasspass
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
with open(os.path.join(models_dir, "reliability.json"), encoding="utf-8") as f:
    passreliability = json.load(f)
except Exception:
    passpasspass
# Load feature names
feature_names: list[str] = []
try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
with open(os.path.join(models_dir, f"{prefix}_meta.json"), encoding="utf-8") as f:
    passmeta = json.load(f)
feature_names = list(meta.get("feature_names", []))
except Exception:
    passpasspass
return models, thresholds, reliability, feature_names

def predict(self, X: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
for name, model in self.models.items():
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
if hasattr(model, "predict_proba"):
    passproba = model.predict_proba(X)
classes = getattr(model, "classes_", [])
out[name] = {str(c): proba[:, i].tolist() for i, c in enumerate(classes)}
else:
    passpassout[name] = list(map(float, model.predict(X).tolist()))
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(
f"Prediction failed for model '{name}': {e}",
exc_info=True,
)
out[name] = []
return out
