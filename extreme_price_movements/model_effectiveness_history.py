from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_EFFECTIVENESS_HISTORY_EXACT = {
    "prob_error",
    "recent_prob_error_20",
    "recent_hit_rate_20",
    "base_model_abs_error_roll20",
    "recent_effectiveness_available",
    "base_meta_model_disagreement",
    "abs_base_meta_diff",
}

MODEL_EFFECTIVENESS_HISTORY_PREFIXES = (
    "recent_meta_",
    "recent_global_",
    "recent_side_horizon_",
    "recent_bucket_",
    "recent_regime_",
    "recent_asset_",
    "recent_base_meta_disagreement_",
    "recent_base_internal_disagreement_",
    "meta_recent_",
    "symbol_recent_",
    "rank_bin_",
)

MODEL_EFFECTIVENESS_HISTORY_TOKENS = (
    "rank_ic",
    "rolling_ic",
    "hit_rate",
    "expected_hit_rate",
    "hit_rate_surprise",
    "confidence_surprise",
    "cal_error",
    "calibration_error",
    "brier",
    "ece",
    "accept_ev",
    "accept_hit_rate",
    "false_accept",
    "false_reject",
    "reject_opportunity_cost",
    "top15_ev",
    "top15_hit_rate",
)


def is_model_effectiveness_history_feature(name: str) -> bool:
    low = str(name).strip().lower()
    if not low:
        return False
    if low in MODEL_EFFECTIVENESS_HISTORY_EXACT:
        return True
    if low.startswith(MODEL_EFFECTIVENESS_HISTORY_PREFIXES):
        return True
    return any(
        re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", low) is not None
        for token in MODEL_EFFECTIVENESS_HISTORY_TOKENS
    )


def _latest_finite(values: np.ndarray, ts: np.ndarray | None = None) -> tuple[float, str]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.any():
        if ts is not None and len(ts) == len(arr):
            ts_ns = pd.to_datetime(ts, utc=True, errors="coerce").astype("int64")
            valid_ts = np.asarray(ts_ns, dtype=np.int64) != pd.NaT.value
            ok = finite & valid_ts
            if ok.any():
                idx = int(np.where(ok)[0][np.argmax(np.asarray(ts_ns)[ok])])
                return float(arr[idx]), "latest_training_finite_by_timestamp"
        idx = int(np.where(finite)[0][-1])
        return float(arr[idx]), "latest_training_finite_by_row_order"
    return 0.0, "fallback_zero_no_training_finite"


def build_model_effectiveness_history_defaults(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    timestamps: Any = None,
) -> dict[str, Any]:
    """Build compact train-history defaults for model-effectiveness features.

    The default is the latest finite causal feature value seen in training, not
    a neutral constant. It is only emitted for explicitly recognized
    model-effectiveness/history columns.
    """
    if frame is None or frame.empty:
        return {
            "defaults": {},
            "sources": {},
            "rows": 0,
            "feature_count": 0,
            "policy": "latest_finite_training_value_for_model_effectiveness_features",
        }
    ts_arr = None
    if timestamps is not None:
        try:
            ts_arr = np.asarray(timestamps)
        except Exception:
            ts_arr = None
    defaults: dict[str, float] = {}
    sources: dict[str, str] = {}
    for feature in dict.fromkeys(str(c) for c in features):
        if feature not in frame.columns:
            continue
        if not is_model_effectiveness_history_feature(feature):
            continue
        vals = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)
        value, source = _latest_finite(vals, ts_arr)
        defaults[feature] = float(value)
        sources[feature] = source
    return {
        "defaults": defaults,
        "sources": sources,
        "rows": int(len(frame)),
        "feature_count": int(len(defaults)),
        "policy": "latest_finite_training_value_for_model_effectiveness_features",
    }


def extract_model_effectiveness_history_defaults(owner: Any) -> dict[str, float]:
    """Read historical defaults from a model, wrapper, or contract dict."""
    candidates: list[Any] = [owner]
    if not isinstance(owner, Mapping):
        for attr in ("best_model", "model", "estimator", "clf", "classifier"):
            child = getattr(owner, attr, None)
            if child is not None:
                candidates.append(child)
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            raw = candidate.get("model_effectiveness_history_defaults")
        else:
            raw = getattr(candidate, "model_effectiveness_history_defaults_", None)
            if raw is None:
                raw = getattr(candidate, "model_effectiveness_history_defaults", None)
            if raw is None:
                contract = getattr(candidate, "meta_feature_contract_", None)
                if isinstance(contract, Mapping):
                    raw = contract.get("model_effectiveness_history_defaults")
        if not isinstance(raw, Mapping):
            continue
        out: dict[str, float] = {}
        for key, value in raw.items():
            if not is_model_effectiveness_history_feature(str(key)):
                continue
            try:
                val = float(value)
            except Exception:
                continue
            if np.isfinite(val):
                out[str(key)] = val
        if out:
            return out
    return {}


def apply_model_effectiveness_history_defaults(
    frame: pd.DataFrame,
    features: Sequence[str],
    defaults: Mapping[str, float] | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Fill only recognized model-effectiveness columns from train history."""
    if frame is None:
        frame = pd.DataFrame()
    out = frame.copy()
    defaults = defaults or {}
    added: list[str] = []
    filled: list[str] = []
    for feature in dict.fromkeys(str(c) for c in features):
        if not is_model_effectiveness_history_feature(feature):
            continue
        if feature not in defaults:
            continue
        try:
            value = float(defaults[feature])
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        if feature not in out.columns:
            out[feature] = np.full(len(out), value, dtype=np.float32)
            added.append(feature)
            continue
        vals = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=np.float32)
        bad = ~np.isfinite(vals)
        if bad.any():
            vals[bad] = np.float32(value)
            out[feature] = vals.astype(np.float32, copy=False)
            filled.append(feature)
    return out, added, filled
