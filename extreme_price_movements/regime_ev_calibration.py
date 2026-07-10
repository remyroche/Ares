"""Regime EV score calibration shared by replay and live inference.

The calibration artifact is intentionally small and deterministic.  It stores
per side x archetype regime effects learned offline; application only reads
pre-entry/live-predictable feature columns and produces:

    score_regime_calibrated = source_score - clipped(sum(effects))

Positive effects therefore lower the trading score, while negative effects
raise it.  The module does not fit anything; fitting belongs in reporting/HPO
scripts so live and replay paths can consume a frozen artifact.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SOURCE_SCORE_DEFAULT = "score_meta_base_soft_label"
ADJUSTED_SCORE_DEFAULT = "score_regime_calibrated"
RISK_SCORE_DEFAULT = "regime_ev_risk_score"
EFFECT_COUNT_DEFAULT = "regime_ev_effect_count"
CALIBRATION_POLICY_ID = "per_regime_archetype_calibration_v1"
_MODEL_CACHE: dict[str, Any] = {}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGIME_EV_CALIBRATION_ARTIFACT = REPO_ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "meta_oos_regime_calibration_rolling60d_oos15_20260708/regime_ev_calibration.json"
)
DEFAULT_REGIME_EV_FEATURE_HANDOFF = REPO_ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706/"
    "train_meta_regime_handoff.parquet"
)


def default_regime_ev_calibration_artifact() -> Path | None:
    """Return the current default calibration artifact unless explicitly disabled."""
    enabled = str(
        os.environ.get("EPM_REGIME_EV_CALIBRATION_ENABLED", "1")
    ).strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    raw = os.environ.get("EPM_REGIME_EV_CALIBRATION_ARTIFACT", "").strip()
    path = Path(raw) if raw else DEFAULT_REGIME_EV_CALIBRATION_ARTIFACT
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def default_regime_ev_feature_handoff() -> Path | None:
    """Return the optional feature handoff used to fill missing calibration inputs."""
    raw = os.environ.get("EPM_REGIME_EV_FEATURE_HANDOFF", "").strip()
    path = Path(raw) if raw else DEFAULT_REGIME_EV_FEATURE_HANDOFF
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _as_numeric(values: Any, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values.reindex(index), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return pd.Series(float("nan"), index=index, dtype="float64")


def _standardized(values: pd.Series, params: Mapping[str, Any]) -> pd.Series:
    med = _safe_float(params.get("median"), 0.0)
    scale = max(_safe_float(params.get("scale"), 1.0), 1e-9)
    return ((values - med) / scale).clip(-6.0, 6.0)


def _apply_shape(values: pd.Series, shape: str, params: Mapping[str, Any]) -> pd.Series:
    if shape == "flat":
        return pd.Series(0.0, index=values.index, dtype="float32")
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if shape == "bucketed":
        qs = np.asarray(params.get("quantiles", []), dtype=float)
        raw = x.to_numpy(dtype=float)
        effects = {int(k): _safe_float(v, 0.0) for k, v in dict(params.get("effects", {})).items()}
        bins = np.digitize(raw, qs, right=True)
        out = np.asarray([effects.get(int(b), 0.0) for b in bins], dtype="float32")
        out[~np.isfinite(raw)] = 0.0
        return pd.Series(out, index=values.index)
    z = _standardized(x, params).fillna(0.0)
    if shape == "monotone":
        xs = np.asarray(params.get("x_thresholds", []), dtype=float)
        ys = np.asarray(params.get("y_thresholds", []), dtype=float)
        if xs.size == 0 or ys.size == 0:
            return pd.Series(0.0, index=values.index, dtype="float32")
        out = np.interp(z.to_numpy(dtype=float), xs, ys, left=ys[0], right=ys[-1])
    elif shape == "linear":
        coef = list(params.get("coef", [0.0]))
        out = z.to_numpy(dtype=float) * _safe_float(coef[0] if coef else 0.0, 0.0)
    elif shape == "quadratic":
        coef = list(params.get("coef", [0.0, 0.0]))
        zz = z.to_numpy(dtype=float)
        out = zz * _safe_float(coef[0] if len(coef) > 0 else 0.0, 0.0)
        out += zz * zz * _safe_float(coef[1] if len(coef) > 1 else 0.0, 0.0)
    elif shape in {"ushape", "u_shaped", "u-shaped", "spline"}:
        coef = list(params.get("coef", [0.0]))
        out = np.abs(z.to_numpy(dtype=float)) * _safe_float(coef[0] if coef else 0.0, 0.0)
    else:
        out = np.zeros(len(values), dtype=float)
    return pd.Series(np.clip(out, -0.06, 0.06).astype("float32"), index=values.index)


def _derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    if (
        "__derived_score_dispersion__" not in out.columns
        and "score_meta_base_soft_label" in out.columns
        and "score_base" in out.columns
    ):
        out["__derived_score_dispersion__"] = (
            pd.to_numeric(out["score_meta_base_soft_label"], errors="coerce")
            - pd.to_numeric(out["score_base"], errors="coerce")
        ).abs().astype("float32")
    if "__derived_meta_uncertainty__" not in out.columns:
        score_col = "score_meta_base_soft_label" if "score_meta_base_soft_label" in out.columns else "calibrated_score"
        if score_col in out.columns:
            p = pd.to_numeric(out[score_col], errors="coerce").clip(1e-6, 1.0 - 1e-6)
            entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
            out["__derived_meta_uncertainty__"] = entropy.astype("float32")
    if "__derived_gmm_entropy__" not in out.columns:
        posterior_cols = [col for col in out.columns if str(col).startswith("gmm_cluster_posterior_")]
        if posterior_cols:
            probs = out[posterior_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0.0, 1.0)
            denom = probs.sum(axis=1).replace(0.0, np.nan)
            probs = probs.div(denom, axis=0).fillna(0.0)
            entropy = -(probs.where(probs.gt(0.0), 1.0).apply(np.log) * probs).sum(axis=1)
            out["__derived_gmm_entropy__"] = (entropy / math.log(max(2, len(posterior_cols)))).astype("float32")
        elif "gmm_posterior_max" in out.columns:
            out["__derived_gmm_entropy__"] = (
                1.0 - pd.to_numeric(out["gmm_posterior_max"], errors="coerce").clip(0.0, 1.0)
            ).astype("float32")
    return out


def load_regime_ev_calibration(path: str | Path | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["_artifact_base_dir"] = str(p.parent)
        return payload
    return {}


def _resolve_model_path(path_value: Any, artifact: Mapping[str, Any]) -> Path:
    p = Path(str(path_value or ""))
    if not p.is_absolute():
        base = Path(str(artifact.get("_artifact_base_dir") or "."))
        p = base / p
    return p


def _load_calibration_model(path_value: Any, artifact: Mapping[str, Any]) -> Any:
    path = _resolve_model_path(path_value, artifact)
    key = str(path)
    if key not in _MODEL_CACHE:
        import joblib

        _MODEL_CACHE[key] = joblib.load(path)
    return _MODEL_CACHE[key]


def _apply_pickled_model(
    frame: pd.DataFrame,
    effect: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> pd.Series:
    feature_cols = [str(c) for c in effect.get("feature_cols") or []]
    if not feature_cols or not all(col in frame.columns for col in feature_cols):
        return pd.Series(0.0, index=frame.index, dtype="float32")
    model = _load_calibration_model(effect.get("model_path"), artifact)
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    fill_values = effect.get("fill_values") or {}
    for col in feature_cols:
        x[col] = x[col].fillna(_safe_float(fill_values.get(col), 0.0))
    try:
        pred = model.predict(x)
    except Exception:
        pred = np.zeros(len(frame), dtype=float)
    return pd.Series(np.asarray(pred, dtype="float32"), index=frame.index).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


def _apply_archetype_aliases(values: pd.Series, artifact: Mapping[str, Any]) -> pd.Series:
    out = values.astype(str)
    aliases = {
        str(k): str(v)
        for k, v in dict(artifact.get("archetype_aliases") or {}).items()
    }
    if aliases:
        out = out.replace(aliases)
    for item in artifact.get("archetype_prefix_aliases") or []:
        if not isinstance(item, Mapping):
            continue
        prefix = str(item.get("prefix") or "")
        alias = str(item.get("alias") or "")
        if prefix and alias:
            out = out.mask(out.str.startswith(prefix, na=False), alias)
    return out


def _timestamp_series(frame: pd.DataFrame) -> pd.Series | None:
    for col in ("__ts__", "timestamp", "entry_timestamp", "bar_ts"):
        if col in frame.columns:
            ts = pd.to_datetime(frame[col], utc=True, errors="coerce")
            return pd.Series(ts, index=frame.index)
    return None


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _effect_time_mask(
    effect: Mapping[str, Any],
    index: pd.Index,
    ts: pd.Series | None,
    artifact: Mapping[str, Any],
) -> pd.Series:
    valid_from = _parse_ts(effect.get("valid_from"))
    valid_to = _parse_ts(effect.get("valid_to"))
    latest_valid_to = _parse_ts(artifact.get("latest_valid_to"))
    is_time_windowed = bool(
        artifact.get("time_windowed_effects")
        or valid_from is not None
        or valid_to is not None
        or effect.get("latest")
    )
    if ts is None:
        if is_time_windowed and any(e.get("latest") for e in artifact.get("effects") or [] if isinstance(e, Mapping)):
            return pd.Series(bool(effect.get("latest")), index=index)
        return pd.Series(True, index=index)
    aligned = ts.reindex(index)
    mask = pd.Series(True, index=index)
    if valid_from is not None:
        mask &= aligned.ge(valid_from)
    if valid_to is not None:
        mask &= aligned.lt(valid_to)
    if bool(effect.get("latest")) and latest_valid_to is not None:
        mask |= aligned.ge(latest_valid_to)
    return mask.fillna(False)


def _effect_match_mask(
    effect: Mapping[str, Any],
    index: pd.Index,
    side: pd.Series,
    arch: pd.Series,
    ts: pd.Series | None,
    artifact: Mapping[str, Any],
) -> pd.Series:
    side_key = str(effect.get("side_name") or effect.get("side") or "")
    arch_key = str(effect.get("archetype_policy_key") or effect.get("archetype") or "")
    mask = pd.Series(True, index=index)
    if side_key and side_key != "*":
        mask &= side.reindex(index).eq(side_key)
    if arch_key and arch_key != "*":
        mask &= arch.reindex(index).eq(arch_key)
    mask &= _effect_time_mask(effect, index, ts, artifact)
    return mask.fillna(False)


def apply_regime_ev_calibration(
    frame: pd.DataFrame,
    artifact: Mapping[str, Any] | None,
    *,
    source_score_col: str | None = None,
    adjusted_score_col: str | None = None,
    side_col: str = "side_name",
    archetype_col: str = "archetype_policy_key",
    copy: bool = True,
) -> pd.DataFrame:
    """Apply frozen regime effects and return a frame with adjusted scores."""
    out = frame.copy() if copy else frame
    artifact = artifact or {}
    source_col = str(source_score_col or artifact.get("source_score_col") or SOURCE_SCORE_DEFAULT)
    adjusted_col = str(adjusted_score_col or artifact.get("adjusted_score_col") or ADJUSTED_SCORE_DEFAULT)
    risk_col = str(artifact.get("risk_score_col") or RISK_SCORE_DEFAULT)
    count_col = str(artifact.get("effect_count_col") or EFFECT_COUNT_DEFAULT)
    risk_cap = max(_safe_float(artifact.get("risk_cap"), 0.06), 0.0)
    risk_cap_positive = max(
        _safe_float(artifact.get("risk_cap_positive"), risk_cap),
        0.0,
    )
    risk_cap_negative = max(
        _safe_float(artifact.get("risk_cap_negative"), risk_cap),
        0.0,
    )
    effects = artifact.get("effects") or []
    if source_col not in out.columns and "calibrated_score" in out.columns:
        source_col = "calibrated_score"
    if source_col not in out.columns:
        raise ValueError(f"source score column missing for regime EV calibration: {source_col}")
    out = _derive_features(out)
    risk = pd.Series(0.0, index=out.index, dtype="float32")
    count = pd.Series(0, index=out.index, dtype="int16")
    side = out[side_col].astype(str) if side_col in out.columns else pd.Series("", index=out.index)
    arch = out[archetype_col].astype(str) if archetype_col in out.columns else pd.Series("", index=out.index)
    arch = _apply_archetype_aliases(arch, artifact)
    ts = _timestamp_series(out)
    for effect in effects:
        if not isinstance(effect, Mapping):
            continue
        if str(effect.get("shape") or "") in {
            "sklearn_pickle",
            "gam_pickle",
            "ebm_pickle",
            "spline_pickle",
        }:
            mask = _effect_match_mask(effect, out.index, side, arch, ts, artifact)
            if not bool(mask.any()):
                continue
            eff = _apply_pickled_model(out.loc[mask], effect, artifact)
            risk.loc[mask] = risk.loc[mask].add(eff.astype("float32"), fill_value=0.0)
            count.loc[mask] = (
                count.loc[mask] + eff.ne(0.0).astype("int16")
            ).astype("int16")
            continue
        feature_col = str(effect.get("feature_col") or "")
        if not feature_col or feature_col not in out.columns:
            continue
        mask = _effect_match_mask(effect, out.index, side, arch, ts, artifact)
        if not bool(mask.any()):
            continue
        values = _as_numeric(out[feature_col], out.index)
        eff = _apply_shape(values.loc[mask], str(effect.get("shape") or "flat"), effect.get("params") or {})
        risk.loc[mask] = risk.loc[mask].add(eff.astype("float32"), fill_value=0.0)
        count.loc[mask] = (count.loc[mask] + eff.ne(0.0).astype("int16")).astype("int16")
    risk = risk.clip(-risk_cap_negative, risk_cap_positive).astype("float32")
    raw = pd.to_numeric(out[source_col], errors="coerce")
    out[risk_col] = risk
    out[count_col] = count
    out[adjusted_col] = (raw - risk).clip(0.0, 1.0).astype("float32")
    out["regime_ev_calibration_source_score_col"] = source_col
    out["regime_ev_calibration_source"] = str(
        artifact.get("policy_id")
        or artifact.get("artifact_id")
        or artifact.get("source")
        or CALIBRATION_POLICY_ID
    )
    return out


def required_feature_columns(artifact: Mapping[str, Any] | None) -> list[str]:
    cols: set[str] = set()
    for effect in (artifact or {}).get("effects") or []:
        if isinstance(effect, Mapping):
            for col in effect.get("feature_cols") or []:
                col = str(col or "")
                if col and not col.startswith("__derived_"):
                    cols.add(col)
            col = str(effect.get("feature_col") or "")
            if col and not col.startswith("__derived_"):
                cols.add(col)
    return sorted(cols)
