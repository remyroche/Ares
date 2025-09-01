# src/transition/rolling_inference.py

from src.transition.multitask_rf import MultiTaskRandomForest
from src.utils.logger import system_logger
from typing import Any
import contextlib
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PlaceholderDataClass:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        """Initialize Placeh
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="rollinginferenceconfig initialization",
    )
    async def initializ
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="rollingmtinference initialization",
    )
    async def initialize(self) -> bool:
        """Initialize RollingMTInference."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
e(self) -> bool:
        """Initialize RollingInferenceConfig."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
olderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passself.logger.info("Implementation placeholder - needs specific logic")
class RollingInferenceConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RollingInferenceConfig:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RollingInferenceConfig:
    passpre_window: int
horizons: list[int]
path_class_priority: list[str]


class RollingMTInference:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RollingMTInference:
    passself.logger.info("Implementation placeholder - needs specific logic")
class RollingMTInference:
    pass"""
Runtime helper for the rolling MultiTask RF.
- Loads per-head models, thresholds, and reliability
- Builds a single-row feature vector for the latest pre-window
- Produces entry/exit decisions and supporting probabilities
"""

def __init__(
self,
config: dict[str, Any],
models_dir: str,
symbol: str,
timeframe: str,
) -> None:
        self.logger = system_logger.getChild("RollingMTInference")
tm = (config or {}).get("TRANSITION_MODELING", {})
r = tm.get("rolling", {}) if isinstance(tm.get("rolling", {}), dict) else {}
self.cfg = RollingInferenceConfig(
pre_window=int(r.get("pre_window", tm.get("pre_window", 60))),
horizons=list(r.get("direction_horizons", [5, 15])),
path_class_priority=[
"beginning_of_trend",
"continuation",
"reversal",
"end_of_trend",
],
)
self.models_dir = models_dir
self.prefix = f"{symbol}_{timeframe}_rolling_mtrf"
self.models: dict[str, Any] = {}
self.thresholds: dict[str, Any] = {}
self.reliability: dict[str, Any] = {}
self.feature_names: list[str] = []

def load(self) -> bool:
        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
models, meta, feat = MultiTaskRandomForest.load(
self.models_dir, prefix=self.prefix,
)
self.models = models
self.thresholds = meta.get("thresholds", {})
self.reliability = meta.get("reliability", {})
self.feature_names = feat
if not self.models:
    passself.logger.warning("No models loaded for rolling inference")
return False
return True
except Exception as e:
    passpasspasspasspasspasspasspassself.logger.warning(f"Failed to load rolling models: {e}")
return False

def _rf_pooled_features(self, seq_df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
for col in [
"log_returns",
"volatility_20",
"volume_ratio",
"rsi",
"macd",
"macd_signal",
"macd_histogram",
"bb_position",
"bb_width",
"atr",
"volatility_regime",
"volatility_acceleration",
]:
    passif col in seq_df.columns:
    passs = pd.to_numeric(seq_df[col], errors="coerce")
out[f"mean_{col}"] = float(np.nanmean(s.values))
out[f"std_{col}"] = float(np.nanstd(s.values))
return out

def _build_X_last(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        if combined_df is None or combined_df.empty:
    passreturn pd.DataFrame(columns=self.feature_names)
pre = self.cfg.pre_window
if len(combined_df) < pre + 1:
    passreturn pd.DataFrame(columns=self.feature_names)
seq = combined_df.iloc[-pre:]
rf = self._rf_pooled_features(seq)
# Build DataFrame with known feature columns; fill missing
x = {name: float(rf.get(name, 0.0)) for name in self.feature_names}
return pd.DataFrame([x])

def _apply_reliability(
self,
head: str,
value: float,
cls: str | None = None,
) -> float:
        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if head == "path_class" and cls is not None:
    passscale = float(self.reliability.get("path_class", {}).get(cls, 1.0))
return float(np.clip(value * scale, 0.0, 1.0))
scale = float(self.reliability.get(head, {}).get("positive_scale", 1.0))
return float(np.clip(value * scale, 0.0, 1.0))
except Exception:
    passpassreturn float(np.clip(value, 0.0, 1.0))

def _get_threshold(
self,
head: str,
cls: str | None = None,
default: float = 0.6,
) -> float:
        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if head == "path_class" and cls is not None:
    passreturn float(self.thresholds.get("path_class", {}).get(cls, default))
return float(self.thresholds.get(head, default))
except Exception:
    passpassreturn float(default)

def predict_latest(self, combined_df: pd.DataFrame) -> dict[str, Any]:
        X = self._build_X_last(combined_df)
if X.empty:
    passreturn {"ready": False}
out: dict[str, Any] = {"ready": True}

# Path class probabilities with reliability scaling
pc = self.models.get("path_class")
p_path: dict[str, float] = {}
if pc is not None:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
proba = pc.predict_proba(X)[0]
classes = list(getattr(pc, "classes_", []))
for i, c in enumerate(classes):
    passp_adj = self._apply_reliability(
"path_class",
float(proba[i]),
cls=str(c),
)
p_path[str(c)] = p_adj
# Optionally normalize
s = float(sum(p_path.values()))
if s > 0:
    passp_path = {k: v / s for k, v in p_path.items()}
except Exception:
    passpasspasspass
out["p_path_class"] = p_path

# Heads: onset / end
for head in ("onset_beginning", "end_trend"):
    passmdl = self.models.get(head)
if mdl is None:
    passcontinue
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
p = float(mdl.predict_proba(X)[0, 1])
out[f"p_{head}"] = self._apply_reliability(head, p)
except Exception:
    passpasscontinue

# Direction and returns per first horizon
if self.cfg.horizons:
    passH = int(self.cfg.horizons[0])
head = f"direction_up_{H}"
mdl = self.models.get(head)
if mdl is not None:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
p = float(mdl.predict_proba(X)[0, 1])
out[f"p_direction_up_{H}"] = self._apply_reliability(head, p)
except Exception:
    passpasspass
reg_head = f"return_{H}"
rmdl = self.models.get(reg_head)
if rmdl is not None:
    passwith contextlib.suppress(Exception):
    passout[f"return_{H}"] = float(rmdl.predict(X)[0])
out["horizon"] = H

# Next regime (multiclass)
nr = self.models.get("next_regime")
if nr is not None:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
proba = nr.predict_proba(X)[0]
classes = list(getattr(nr, "classes_", []))
p_nr = {}
for i, c in enumerate(classes):
    passp_adj = self._apply_reliability(
"next_regime",
float(proba[i]),
cls=str(c),
)
p_nr[str(c)] = p_adj
s = float(sum(p_nr.values()))
if s > 0:
    passp_nr = {k: v / s for k, v in p_nr.items()}
out["p_next_regime"] = p_nr
except Exception:
    passpasspasspass

# Decisions
# Entry: favor beginning/continuation crossings; fallback to onset
allow = False
trigger = None
fav = 0.0
fav_thr = 0.6
for cls in ["beginning_of_trend", "continuation"]:
    passp = float(p_path.get(cls, 0.0))
thr = self._get_threshold("path_class", cls, default=0.6)
if p >= thr and p > fav:
    passallow = True
trigger = cls
fav = p
fav_thr = thr
if not allow and "p_onset_beginning" in out:
    passp_onset = float(out.get("p_onset_beginning", 0.0))
thr_onset = self._get_threshold("onset_beginning", default=0.6)
if p_onset >= thr_onset:
    passallow = True
trigger = "onset_beginning"
fav = p_onset
fav_thr = thr_onset
out["allow_trade"] = allow
out["trigger"] = trigger

# Side and reinforcement
side = None
mult = 1.0
H = out.get("horizon")
if H is not None:
    passp_up = float(out.get(f"p_direction_up_{H}", 0.0))
thr_up = self._get_threshold(f"direction_up_{H}", default=0.6)
side = "long" if p_up >= thr_up else "short"
# reinforcement: scale between 0.5 and 2.0 based on how far above threshold fav is
if allow and fav_thr < 1.0:
    passmult = float(
np.clip(
0.5 + 1.5 * (fav - fav_thr) / max(1e-6, (1.0 - fav_thr)),
0.5,
2.0,
),
)
out["side"] = side
out["position_multiplier"] = mult

# Exit logic
p_rev = float(p_path.get("reversal", 0.0))
thr_rev = self._get_threshold("path_class", "reversal", default=0.6)
p_end = float(out.get("p_end_trend", 0.0))
thr_end = self._get_threshold("end_trend", default=0.6)
favorable = max(
float(p_path.get("continuation", 0.0)),
float(p_path.get("beginning_of_trend", 0.0)),
)
exit_bias = float(p_rev - favorable)
exit_flag = bool(p_rev >= thr_rev or p_end >= thr_end or exit_bias > 0)
out["exit_flag"] = exit_flag
out["exit_bias"] = exit_bias

return out
