"""MR/TF hard-route mask generation and specialist comparison helpers.

The mask layer is deliberately upstream of model training: it only reads columns
already present in the feature/training frame, writes internal route columns, and
leaves the downstream base/meta fit functions unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Optional acceleration for larger MR/TF gate searches.
    from numba import njit
except Exception:  # pragma: no cover - numba is optional at runtime.
    njit = None  # type: ignore[assignment]


ROUTE_COL = "__mr_tf_route__"
MR_MASK_COL = "__mr_mask__"
TF_MASK_COL = "__tf_mask__"
MIXED_MASK_COL = "__mixed_mask__"
PARAMS_HASH_COL = "__mr_tf_params_hash__"
MASK_COLUMNS = (ROUTE_COL, MR_MASK_COL, TF_MASK_COL, MIXED_MASK_COL, PARAMS_HASH_COL)
ROUTE_FEATURE_COLUMNS = (
    "mr_tf_route_mr",
    "mr_tf_route_tf",
    "mr_tf_route_mixed",
    "mr_tf_route_known",
    "mr_tf_specialist_active",
    "mr_tf_general_fallback",
)
ROUTE_SCORE_SOURCE_COL = "mr_tf_policy_score_source"
GENERAL_SCORE_PREFIX = "mr_tf_general_"
DEFAULT_ROUTE_SCORE_COLUMNS = (
    "oof_meta_clf",
    "oof_pred",
    "oof_p_move",
    "clf",
    "calibrated_score",
    "raw_meta_prediction",
)


@dataclass(frozen=True)
class MRTFMaskParams:
    q_adx_tf: float = 0.65
    q_adx_mr: float = 0.45
    q_stretch_mr: float = 0.75
    q_persist_tf: float = 0.70
    q_persist_mr: float = 0.30
    q_tf_quality: float = 0.60
    q_mr_quality: float = 0.60
    N_tf: int = 3
    N_mr: int = 2
    ema_gap_min_tf: float = 0.0
    mom_min_tf: float = 0.0
    stretch_min_mr: float = 0.75
    reversal_min_mr: float = 0.0
    persistence_axis: str = "none"
    tf_quality_axis: str = "none"
    mr_quality_axis: str = "none"
    thresholds: Mapping[str, float] | None = None


DEFAULT_PARAMS = MRTFMaskParams()
PERSISTENCE_AXIS_CHOICES = {"none", "hurst", "autocorr", "efficiency", "composite"}
PERSISTENCE_OPTUNA_AXIS_CHOICES = ("none", "hurst", "autocorr", "efficiency")
TF_QUALITY_AXIS_CHOICES = {
    "none",
    "trend",
    "path",
    "autocorr",
    "breakout",
}
MR_QUALITY_AXIS_CHOICES = {
    "none",
    "chop",
    "range_position",
    "rsi_reversion",
    "compression",
    "reversion_target",
}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def mr_tf_masks_enabled(cfg: Mapping[str, Any] | None) -> bool:
    section = (cfg or {}).get("mr_tf_masks", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(section, Mapping) and "enabled" in section:
        return _truthy(section.get("enabled"))
    return _truthy((cfg or {}).get("mr_tf_masks_enabled") if isinstance(cfg, Mapping) else False)


def _section(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        return {}
    raw = cfg.get("mr_tf_masks", {})
    out = dict(raw) if isinstance(raw, Mapping) else {}
    for key in asdict(DEFAULT_PARAMS):
        flat = f"mr_tf_masks_{key}"
        if flat in cfg and key not in out:
            out[key] = cfg[flat]
    return out


def mask_params_from_cfg(
    cfg: Mapping[str, Any] | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> MRTFMaskParams:
    raw = {**asdict(DEFAULT_PARAMS)}
    raw.update(_section(cfg))
    raw.update(dict(overrides or {}))
    allowed = set(asdict(DEFAULT_PARAMS))
    clean = {k: raw[k] for k in allowed if k in raw}
    clean["N_tf"] = int(clean.get("N_tf", DEFAULT_PARAMS.N_tf))
    clean["N_mr"] = int(clean.get("N_mr", DEFAULT_PARAMS.N_mr))
    for key in (
        "q_adx_tf",
        "q_adx_mr",
        "q_stretch_mr",
        "q_persist_tf",
        "q_persist_mr",
        "q_tf_quality",
        "q_mr_quality",
        "ema_gap_min_tf",
        "mom_min_tf",
        "stretch_min_mr",
        "reversal_min_mr",
    ):
        clean[key] = float(clean.get(key, getattr(DEFAULT_PARAMS, key)))
    axis = str(clean.get("persistence_axis", DEFAULT_PARAMS.persistence_axis) or "none").strip().lower()
    if axis not in PERSISTENCE_AXIS_CHOICES:
        axis = DEFAULT_PARAMS.persistence_axis
    clean["persistence_axis"] = axis
    tf_axis = str(clean.get("tf_quality_axis", DEFAULT_PARAMS.tf_quality_axis) or "none").strip().lower()
    if tf_axis not in TF_QUALITY_AXIS_CHOICES:
        tf_axis = DEFAULT_PARAMS.tf_quality_axis
    clean["tf_quality_axis"] = tf_axis
    mr_axis = str(clean.get("mr_quality_axis", DEFAULT_PARAMS.mr_quality_axis) or "none").strip().lower()
    if mr_axis not in MR_QUALITY_AXIS_CHOICES:
        mr_axis = DEFAULT_PARAMS.mr_quality_axis
    clean["mr_quality_axis"] = mr_axis
    thresholds = clean.get("thresholds")
    clean["thresholds"] = (
        {str(k): float(v) for k, v in dict(thresholds).items() if _is_finite(v)}
        if isinstance(thresholds, Mapping)
        else None
    )
    return MRTFMaskParams(**clean)


def params_to_dict(params: MRTFMaskParams | Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return asdict(DEFAULT_PARAMS)
    if isinstance(params, MRTFMaskParams):
        out = asdict(params)
    else:
        out = dict(params)
    if isinstance(out.get("thresholds"), Mapping):
        out["thresholds"] = {
            str(k): float(v) for k, v in dict(out["thresholds"]).items() if _is_finite(v)
        }
    return out


def mr_tf_params_hash(params: MRTFMaskParams | Mapping[str, Any] | None) -> str:
    blob = json.dumps(params_to_dict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def canonical_route(value: Any) -> str:
    route = str(value or "").strip().lower()
    if route in {"mr", "mean_reversion", "mean-reversion"}:
        return "mr"
    if route in {"tf", "trend_follow", "trend-follow", "trend_following"}:
        return "tf"
    if route == "mixed":
        return "mixed"
    return "unknown"


def mr_tf_route_from_path(path: str | Path) -> str | None:
    stem = Path(path).stem
    for suffix in (
        "_tbm_clf",
        "_correctness_clf",
        "_clf",
        "_reg",
    ):
        for route in ("mr", "tf"):
            if stem.endswith(f"_{route}{suffix}"):
                return route
    return None


def strip_mr_tf_route_suffix(stem: str) -> str:
    out = str(stem)
    for suffix in (
        "_tbm_clf",
        "_correctness_clf",
        "_clf",
        "_reg",
    ):
        for route in ("mr", "tf"):
            routed_suffix = f"_{route}{suffix}"
            if out.endswith(routed_suffix):
                return out[: -len(routed_suffix)] + suffix
    return out


def route_series_from_frame(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return canonical MR/TF route and whether it came from explicit route state."""
    n = len(frame)
    idx = frame.index
    route = pd.Series("unknown", index=idx, dtype=object)
    known = pd.Series(False, index=idx, dtype=bool)
    for col in (ROUTE_COL, "mr_tf_route"):
        if col in frame.columns:
            raw = frame[col].map(canonical_route)
            explicit = raw.isin(["mr", "tf", "mixed"])
            route.loc[explicit] = raw.loc[explicit]
            known.loc[explicit] = True
            break
    if not bool(known.all()):
        mr_col = MR_MASK_COL if MR_MASK_COL in frame.columns else "mr_mask"
        tf_col = TF_MASK_COL if TF_MASK_COL in frame.columns else "tf_mask"
        mixed_col = MIXED_MASK_COL if MIXED_MASK_COL in frame.columns else "mixed_mask"
        mr = (
            pd.to_numeric(frame[mr_col], errors="coerce").fillna(0.0).astype(float) > 0
            if mr_col in frame.columns
            else pd.Series(False, index=idx)
        )
        tf = (
            pd.to_numeric(frame[tf_col], errors="coerce").fillna(0.0).astype(float) > 0
            if tf_col in frame.columns
            else pd.Series(False, index=idx)
        )
        mixed = (
            pd.to_numeric(frame[mixed_col], errors="coerce").fillna(0.0).astype(float) > 0
            if mixed_col in frame.columns
            else pd.Series(False, index=idx)
        )
        inferred_known = (~known) & (mr | tf | mixed)
        route.loc[inferred_known & mr & ~tf] = "mr"
        route.loc[inferred_known & tf & ~mr] = "tf"
        route.loc[inferred_known & (mixed | (mr & tf))] = "mixed"
        known.loc[inferred_known] = True
    if n == 0:
        route = pd.Series(dtype=object, index=idx)
        known = pd.Series(dtype=bool, index=idx)
    return route, known


def append_mr_tf_route_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach numeric route features for regime adaptor/reporting consumers."""
    out = frame.copy()
    route, known = route_series_from_frame(out)
    route = route.reindex(out.index).fillna("unknown")
    known = known.reindex(out.index).fillna(False)
    out["mr_tf_route_mr"] = (route == "mr").astype(np.float32)
    out["mr_tf_route_tf"] = (route == "tf").astype(np.float32)
    out["mr_tf_route_mixed"] = (route == "mixed").astype(np.float32)
    out["mr_tf_route_known"] = known.astype(np.float32)
    source = (
        out[ROUTE_SCORE_SOURCE_COL].astype(str).map(canonical_route)
        if ROUTE_SCORE_SOURCE_COL in out.columns
        else pd.Series("unknown", index=out.index)
    )
    specialist = source.isin(["mr", "tf"])
    if not bool(specialist.any()) and "mr_tf_specialist_active" in out.columns:
        specialist = (
            pd.to_numeric(out["mr_tf_specialist_active"], errors="coerce")
            .fillna(0.0)
            .astype(float)
            > 0
        )
    out["mr_tf_specialist_active"] = specialist.astype(np.float32)
    out["mr_tf_general_fallback"] = (
        known.astype(bool) & ~specialist.astype(bool)
    ).astype(np.float32)
    return out


def _row_key_frame(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    candidates = [
        ["timestamp", "symbol"],
        ["source_row_index"],
        ["index"],
        ["timestamp"],
    ]
    for keys in candidates:
        if all(k in df.columns for k in keys):
            key_df = df[keys].copy()
            if "timestamp" in key_df.columns:
                key_df["timestamp"] = pd.to_datetime(
                    key_df["timestamp"], utc=True, errors="coerce"
                )
            if "symbol" in key_df.columns:
                key_df["symbol"] = key_df["symbol"].astype(str)
            key_df = key_df.reset_index(drop=True)
            if bool(key_df.notna().any().any()):
                return key_df, keys
    return None, []


def overlay_mr_tf_route_predictions(
    base_df: pd.DataFrame,
    route_df: pd.DataFrame,
    *,
    route: str,
    score_columns: Sequence[str] = DEFAULT_ROUTE_SCORE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Overlay route-specialist OOF scores onto canonical general OOF rows.

    The policy layer should still see one strategy. Route heads only replace the
    canonical score for rows whose persisted/recomputed route matches.
    """
    canonical = canonical_route(route)
    if canonical not in {"mr", "tf"} or base_df.empty or route_df.empty:
        return base_df, {
            "route": canonical,
            "overlay_rows": 0,
            "reason": "empty_or_unknown_route",
        }
    out = base_df.copy()
    base_route, base_known = route_series_from_frame(out)
    route_mask = (base_route == canonical) & base_known
    if not bool(route_mask.any()):
        return out, {
            "route": canonical,
            "overlay_rows": 0,
            "reason": "no_matching_base_route_rows",
        }
    usable_cols = [c for c in score_columns if c in route_df.columns]
    if not usable_cols:
        return out, {
            "route": canonical,
            "overlay_rows": 0,
            "reason": "missing_route_score_columns",
        }

    base_key, keys = _row_key_frame(out)
    route_key, _ = _row_key_frame(route_df)
    overlay_positions: np.ndarray
    route_positions: np.ndarray
    if base_key is not None and route_key is not None and keys:
        right = route_df.reset_index(drop=True).copy()
        right_key = route_key.copy()
        right_key["_route_pos"] = np.arange(len(right_key), dtype=np.int64)
        left = base_key.copy()
        left["_base_pos"] = np.arange(len(left), dtype=np.int64)
        matched = left.merge(
            right_key.drop_duplicates(subset=keys, keep="first"),
            on=keys,
            how="left",
            sort=False,
        )
        valid = matched["_route_pos"].notna().to_numpy()
        base_pos = matched.loc[valid, "_base_pos"].to_numpy(dtype=np.int64)
        route_pos = matched.loc[valid, "_route_pos"].to_numpy(dtype=np.int64)
        keep = route_mask.to_numpy()[base_pos]
        overlay_positions = base_pos[keep]
        route_positions = route_pos[keep]
    else:
        base_pos = np.flatnonzero(route_mask.to_numpy())
        n = min(len(base_pos), len(route_df))
        overlay_positions = base_pos[:n]
        route_positions = np.arange(n, dtype=np.int64)

    if len(overlay_positions) <= 0:
        return out, {
            "route": canonical,
            "overlay_rows": 0,
            "reason": "no_aligned_route_rows",
        }

    for col in usable_cols:
        if col not in out.columns:
            out[col] = np.nan
        general_col = f"{GENERAL_SCORE_PREFIX}{col}"
        if general_col not in out.columns:
            out[general_col] = out[col].values
        vals = pd.to_numeric(route_df.iloc[route_positions][col], errors="coerce")
        finite = np.isfinite(vals.to_numpy(dtype=np.float64, copy=False))
        if not bool(finite.any()):
            continue
        out.iloc[overlay_positions[finite], out.columns.get_loc(col)] = vals.iloc[
            np.flatnonzero(finite)
        ].to_numpy()
    if ROUTE_SCORE_SOURCE_COL not in out.columns:
        out[ROUTE_SCORE_SOURCE_COL] = "general"
    out.iloc[
        overlay_positions,
        out.columns.get_loc(ROUTE_SCORE_SOURCE_COL),
    ] = canonical
    out = append_mr_tf_route_features(out)
    return out, {
        "route": canonical,
        "overlay_rows": int(len(overlay_positions)),
        "score_columns": list(usable_cols),
        "alignment_keys": list(keys),
    }


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _first_existing(frame: pd.DataFrame, aliases: Sequence[str], prefixes: Sequence[str] = ()) -> str | None:
    cols = [str(c) for c in frame.columns]
    by_lower = {c.lower(): c for c in cols}
    for alias in aliases:
        hit = by_lower.get(str(alias).lower())
        if hit is not None:
            return hit
    for prefix in prefixes:
        p = str(prefix).lower()
        for col in cols:
            if col.lower().startswith(p):
                return col
    return None


def _contains_existing(frame: pd.DataFrame, terms: Sequence[str]) -> str | None:
    cols = [str(c) for c in frame.columns]
    for term in terms:
        t = str(term).lower()
        for col in cols:
            if t in col.lower():
                return col
    return None


def _numeric_series(frame: pd.DataFrame, col: str | None, *, default: float = 0.0) -> pd.Series:
    if col is None or col not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").astype(np.float64)


def _safe_quantile(values: pd.Series, q: float, default: float = 0.0) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.nanquantile(arr, float(np.clip(q, 0.0, 1.0))))


def _momentum_aliases(n: int) -> list[str]:
    return [
        f"return{n}h",
        f"return_{n}h",
        f"ret{n}h",
        f"ret_{n}h",
        f"ret_eq_{n}h",
        f"mkt_ret_eq_{n}h",
        f"mkt_ret_{n}h",
        f"return_{n}",
        f"ret_{n}",
        f"mom{n}h",
        f"mom_{n}",
        f"momentum{n}h",
        f"momentum_{n}",
    ]


def _persistence_source_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    hurst_col = _first_existing(
        frame,
        [
            "hurst_exponent",
            "hurst",
            "hurst_100",
            "hurst_200",
            "hurst_proxy_24",
            "hurst_proxy_x_regime_trend_48h",
        ],
    ) or _contains_existing(frame, ["hurst_proxy", "hurst"])
    autocorr_col = _first_existing(
        frame,
        [
            "return_autocorr_48",
            "return_autocorr_24",
            "ret_autocorr_48",
            "ret_autocorr_24",
            "autocorr_ret_48",
            "autocorr_ret_24",
            "autocorr_24h",
            "autocorr_6h",
        ],
    ) or _contains_existing(frame, ["return_autocorr", "ret_autocorr", "autocorr_"])
    efficiency_col = _first_existing(
        frame,
        [
            "ker_24",
            "ker_16",
            "kaufman_efficiency_ratio",
            "efficiency_ratio",
            "efficiency_ratio_20",
            "path_efficiency_24",
            "path_efficiency_12",
            "z_path_efficiency_24",
            "trend_efficiency",
            "prog_eff_12",
        ],
    ) or _contains_existing(frame, ["ker_", "path_efficiency", "efficiency"])
    choppiness_col = _first_existing(
        frame,
        ["choppiness_index_20", "choppiness_index", "choppiness"],
    ) or _contains_existing(frame, ["choppiness"])
    entropy_col = _first_existing(
        frame,
        [
            "direction_entropy_20",
            "perm_entropy_ret_24",
            "perm_entropy_ret_12",
            "spectral_entropy_ret_48",
            "spectral_entropy_ret_24",
            "shannon_entropy_ret_16",
            "shannon_entropy_ret_8",
            "volume_entropy_24",
            "volume_entropy_12",
        ],
    ) or _contains_existing(frame, ["entropy_ret", "direction_entropy", "entropy"])
    return {
        "hurst": hurst_col,
        "autocorr": autocorr_col,
        "efficiency": efficiency_col,
        "choppiness": choppiness_col,
        "entropy": entropy_col,
    }


def _quality_source_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    trend_col = _first_existing(
        frame,
        [
            "trend_snr",
            "trend_regime_score",
            "regime_trend_score",
            "trend_strength_vs_reversion",
            "price_trend_7d_vol_norm",
            "price_trend_10d_vol_norm",
            "ema50_ema200_spread_continuous",
            "adx_14_slope",
            "adx_10_slope",
            "adx_7_slope",
        ],
    ) or _contains_existing(frame, ["trend_snr", "trend_regime", "trend_strength"])
    path_col = _first_existing(
        frame,
        [
            "path_efficiency_24",
            "path_efficiency_12",
            "efficiency_ratio_20",
            "z_path_efficiency_24",
        ],
    ) or _contains_existing(frame, ["path_efficiency", "efficiency_ratio"])
    autocorr_col = _first_existing(
        frame,
        [
            "return_autocorr_48",
            "return_autocorr_24",
            "autocorr_24h",
            "autocorr_6h",
            "volatility_autocorr_48",
        ],
    ) or _contains_existing(frame, ["return_autocorr", "autocorr_"])
    breakout_col = _first_existing(
        frame,
        [
            "post_impulse_persistence",
            "post_impulse_volume_persistence",
            "flow_persistence",
            "vol_regime_shift",
            "vol_high",
            "spike_score_surprise",
            "volatility_ratio_short_long",
        ],
    ) or _contains_existing(frame, ["post_impulse", "flow_persistence", "vol_regime_shift"])
    chop_col = _first_existing(
        frame,
        [
            "chop_score",
            "choppiness_index_20",
            "choppiness_index",
            "complexity_regime_24h",
        ],
    ) or _contains_existing(frame, ["chop", "choppiness"])
    range_position_col = _first_existing(
        frame,
        [
            "oiw_entry_zone_1d_atr",
            "dist_oiw_intensity_12h_atr",
            "dist_oiw_intensity_96h_atr",
            "dist_oiw_z_delta_12h_atr",
            "dist_oiw_z_delta_96h_atr",
            "dist_oiw_signed_delta_12h_atr",
            "dist_oiw_signed_delta_96h_atr",
            "dist_oiw_abs_delta_12h_atr",
            "dist_oiw_abs_delta_96h_atr",
            "loc_bb_channel_pos_48",
            "loc_bb_channel_pos_24",
            "loc_prev_week_range_pos_48",
            "loc_prev_week_range_pos_24",
            "loc_prev_day_range_pos_48",
            "loc_prev_day_range_pos_24",
            "loc_range_pos_48",
            "loc_range_pos_24",
            "loc_swing_range_pos_48",
            "loc_swing_range_pos_24",
        ],
    ) or _contains_existing(frame, ["range_pos", "bb_channel_pos"])
    rsi_col = _first_existing(
        frame,
        [
            "rsi",
            "rsi_base",
            "rsi_z",
            "rsi_ts_resid",
            "rsi_lag1",
            "rsi_slope",
            "rsi_1h_slope",
        ],
    ) or _contains_existing(frame, ["rsi"])
    compression_col = _first_existing(
        frame,
        [
            "vol_compression",
            "compression_ratio",
            "atr_compression_ratio",
            "vol_compression_ratio",
            "bollinger_band_width",
            "bars_in_high_vol_state_log_norm",
        ],
    ) or _contains_existing(frame, ["compression", "bollinger_band_width"])
    reversion_target_col = _first_existing(
        frame,
        [
            "reversion_target_distance",
            "premium_mean_reversion_halflife_24h",
            "t_be_proxy",
        ],
    ) or _contains_existing(frame, ["reversion_target", "mean_reversion"])
    return {
        "trend": trend_col,
        "path": path_col,
        "autocorr": autocorr_col,
        "breakout": breakout_col,
        "chop": chop_col,
        "range_position": range_position_col,
        "rsi_reversion": rsi_col,
        "compression": compression_col,
        "reversion_target": reversion_target_col,
    }


def _compression_quality_from_values(values: pd.Series, col: str | None) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(np.float64)
    name = str(col or "").lower()
    if "band_width" in name or name.endswith("_width"):
        return -vals
    return vals


def _range_position_quality_from_values(
    values: pd.Series,
    *,
    side_sign: float,
    col: str | None,
) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(np.float64)
    name = str(col or "").lower()
    if "pos" in name or bool(vals.dropna().between(0.0, 1.0).mean() > 0.80):
        centered = vals - 0.5
    else:
        centered = vals
    return (-float(side_sign) * centered).astype(np.float64)


def _rsi_reversion_quality_from_values(
    values: pd.Series,
    *,
    side_sign: float,
) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(np.float64)
    finite = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty and float(finite.quantile(0.90)) > 2.0:
        centered = (vals - 50.0) / 50.0
    elif not finite.empty and float(finite.quantile(0.90)) <= 1.0 and float(finite.quantile(0.10)) >= 0.0:
        centered = vals - 0.5
    else:
        centered = vals
    return (-float(side_sign) * centered).astype(np.float64)


def _quality_score_series(
    frame: pd.DataFrame,
    *,
    axis: str,
    route: str,
    side_sign: float,
    sources: Mapping[str, str | None] | None = None,
) -> pd.Series:
    axis_clean = str(axis or "none").strip().lower()
    if axis_clean == "none":
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    src = dict(sources or _quality_source_columns(frame))
    col = src.get(axis_clean)
    vals = _numeric_series(frame, col, default=np.nan)
    if route == "mr" and axis_clean == "range_position":
        return _range_position_quality_from_values(vals, side_sign=side_sign, col=col)
    if route == "mr" and axis_clean == "rsi_reversion":
        return _rsi_reversion_quality_from_values(vals, side_sign=side_sign)
    if route == "mr" and axis_clean == "compression":
        return _compression_quality_from_values(vals, col)
    return vals.astype(np.float64)


def _scaled_unit_series(values: pd.Series, *, mode: str) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(np.float64)
    if mode == "autocorr":
        return ((vals.clip(-1.0, 1.0) + 1.0) * 0.5).astype(np.float64)
    if mode in {"choppiness", "entropy"}:
        finite = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty and float(finite.max()) > 2.0:
            return (1.0 - (vals / 100.0)).clip(0.0, 1.0).astype(np.float64)
        return (1.0 - vals.clip(0.0, 1.0)).astype(np.float64)
    return vals.clip(0.0, 1.0).astype(np.float64)


def _persistence_score_series(
    frame: pd.DataFrame,
    *,
    axis: str,
    sources: Mapping[str, str | None] | None = None,
) -> pd.Series:
    axis_clean = str(axis or "none").strip().lower()
    src = dict(sources or _persistence_source_columns(frame))
    if axis_clean == "none":
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    if axis_clean in {"hurst", "autocorr", "efficiency"}:
        return _numeric_series(frame, src.get(axis_clean), default=np.nan)
    if axis_clean != "composite":
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)

    components: list[pd.Series] = []
    for name, mode in (
        ("hurst", "bounded"),
        ("autocorr", "autocorr"),
        ("efficiency", "bounded"),
        ("choppiness", "choppiness"),
        ("entropy", "entropy"),
    ):
        col = src.get(name)
        if col is None or col not in frame.columns:
            continue
        if mode == "bounded":
            comp = _scaled_unit_series(_numeric_series(frame, col, default=np.nan), mode="bounded")
        else:
            comp = _scaled_unit_series(_numeric_series(frame, col, default=np.nan), mode=mode)
        components.append(comp)
    if not components:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    stacked = pd.concat(components, axis=1)
    return stacked.mean(axis=1, skipna=True).astype(np.float64)


def _source_columns(frame: pd.DataFrame, params: MRTFMaskParams) -> dict[str, Any]:
    adx_col = _first_existing(frame, ["adx", "adx_14", "adx_24"], prefixes=["adx_"])
    ema_col = _first_existing(
        frame,
        [
            "ema_gap",
            "ema_gap_atr",
            "ema_fast_minus_slow_atr",
            "ema50_ema200_spread_atr",
            "ema50_ema200_spread_continuous",
            "ffd_ema_spread_04",
            "ffd_ema_spread_05",
            "ffd_ema_spread_06",
            "dist_ema_fast",
            "dist_ema20_atr",
            "dist_ema50_atr",
            "dist_ema200_atr",
            "loc_ema_stack_pos_24",
            "loc_ema_stack_pos",
        ],
    ) or _contains_existing(frame, ["ema_stack_pos", "ema_gap"])
    mom_tf_col = _first_existing(frame, _momentum_aliases(params.N_tf))
    rev_mr_col = _first_existing(frame, _momentum_aliases(params.N_mr))
    stretch_col = _first_existing(
        frame,
        [
            "oiw_entry_zone_1d_atr",
            "oiw_intensity_entry_dist_1d_atr",
            "oiw_intensity_entry_dist_7d_atr",
            "oiw_z_delta_entry_dist_1d_atr",
            "oiw_z_delta_entry_dist_7d_atr",
            "oiw_pos_delta_entry_dist_1d_atr",
            "oiw_pos_delta_entry_dist_7d_atr",
            "dist_oiw_intensity_12h_atr",
            "dist_oiw_intensity_96h_atr",
            "dist_oiw_z_delta_12h_atr",
            "dist_oiw_z_delta_96h_atr",
            "dist_oiw_signed_delta_12h_atr",
            "dist_oiw_signed_delta_96h_atr",
            "dist_oiw_abs_delta_12h_atr",
            "dist_oiw_abs_delta_96h_atr",
            "stretch_atr",
            "vwap_dev_atr",
            "dist_vwap_norm",
            "loc_vwap_dev_z_24",
            "loc_vwap_dev_z_48",
            "zscore_price_50",
            "dist_weekly_vwap",
            "dist_rolling_7d_high",
            "loc_prev_week_range_pos_48",
        ],
    ) or _contains_existing(frame, ["vwap_dev", "zscore_price", "dist_weekly_vwap"])
    return {
        "adx": adx_col,
        "ema_gap": ema_col,
        "mom_tf": mom_tf_col,
        "stretch": stretch_col,
        "reversal_mr": rev_mr_col,
        "persistence_axis": params.persistence_axis,
        "persistence": _persistence_source_columns(frame),
        "tf_quality_axis": params.tf_quality_axis,
        "mr_quality_axis": params.mr_quality_axis,
        "quality": _quality_source_columns(frame),
    }


def _side_sign(side: str | None) -> float:
    return -1.0 if str(side or "").lower().startswith("short") else 1.0


def apply_mr_tf_masks(
    frame: pd.DataFrame,
    *,
    side: str | None = None,
    params: MRTFMaskParams | Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach hard MR/TF/mixed route columns to ``frame``.

    If no explicit threshold values are supplied, finite thresholds are derived
    from the provided frame. Training artifacts persist those thresholds so
    inference/replay can call the same function with identical route behavior.
    """
    out = frame.copy()
    p = mask_params_from_cfg(cfg, overrides=params_to_dict(params) if params is not None else None)
    src = _source_columns(out, p)
    sign = _side_sign(side)

    adx = _numeric_series(out, src["adx"])
    ema_gap = _numeric_series(out, src["ema_gap"])
    mom_tf = _numeric_series(out, src["mom_tf"])
    stretch = _numeric_series(out, src["stretch"])
    reversal_mr = _numeric_series(out, src["reversal_mr"])
    abs_stretch = stretch.abs()
    persistence_axis = str(p.persistence_axis or "none").strip().lower()
    persistence_score = _persistence_score_series(
        out,
        axis=persistence_axis,
        sources=src.get("persistence") if isinstance(src.get("persistence"), Mapping) else None,
    )
    tf_quality_axis = str(p.tf_quality_axis or "none").strip().lower()
    mr_quality_axis = str(p.mr_quality_axis or "none").strip().lower()
    quality_sources = src.get("quality") if isinstance(src.get("quality"), Mapping) else None
    tf_quality_score = _quality_score_series(
        out,
        axis=tf_quality_axis,
        route="tf",
        side_sign=sign,
        sources=quality_sources,
    )
    mr_quality_score = _quality_score_series(
        out,
        axis=mr_quality_axis,
        route="mr",
        side_sign=sign,
        sources=quality_sources,
    )

    thresholds = dict(p.thresholds or {})
    if "adx_tf" not in thresholds:
        thresholds["adx_tf"] = _safe_quantile(adx, p.q_adx_tf)
    if "adx_mr" not in thresholds:
        thresholds["adx_mr"] = _safe_quantile(adx, p.q_adx_mr)
    if "stretch_mr" not in thresholds:
        thresholds["stretch_mr"] = _safe_quantile(abs_stretch, p.q_stretch_mr)
    if persistence_axis != "none":
        if "persist_tf" not in thresholds:
            thresholds["persist_tf"] = _safe_quantile(persistence_score, p.q_persist_tf)
        if "persist_mr" not in thresholds:
            thresholds["persist_mr"] = _safe_quantile(persistence_score, p.q_persist_mr)
    if tf_quality_axis != "none" and "tf_quality" not in thresholds:
        thresholds["tf_quality"] = _safe_quantile(tf_quality_score, p.q_tf_quality)
    if mr_quality_axis != "none" and "mr_quality" not in thresholds:
        thresholds["mr_quality"] = _safe_quantile(mr_quality_score, p.q_mr_quality)

    finite_tf = adx.notna() & ema_gap.notna() & mom_tf.notna()
    finite_mr = adx.notna() & stretch.notna() & reversal_mr.notna()
    if persistence_axis != "none":
        finite_persistence = persistence_score.notna()
        tf_persistence_gate = finite_persistence & (
            persistence_score > float(thresholds["persist_tf"])
        )
        mr_persistence_gate = finite_persistence & (
            persistence_score < float(thresholds["persist_mr"])
        )
    else:
        tf_persistence_gate = pd.Series(True, index=out.index, dtype=bool)
        mr_persistence_gate = pd.Series(True, index=out.index, dtype=bool)
    if tf_quality_axis != "none":
        tf_quality_gate = tf_quality_score.notna() & (
            tf_quality_score > float(thresholds["tf_quality"])
        )
    else:
        tf_quality_gate = pd.Series(True, index=out.index, dtype=bool)
    if mr_quality_axis != "none":
        mr_quality_gate = mr_quality_score.notna() & (
            mr_quality_score > float(thresholds["mr_quality"])
        )
    else:
        mr_quality_gate = pd.Series(True, index=out.index, dtype=bool)
    tf_gate = (
        finite_tf
        & (adx > float(thresholds["adx_tf"]))
        & ((sign * ema_gap) > p.ema_gap_min_tf)
        & ((sign * mom_tf) > p.mom_min_tf)
        & tf_persistence_gate
        & tf_quality_gate
    )
    mr_gate = (
        finite_mr
        & (adx < float(thresholds["adx_mr"]))
        & (abs_stretch > max(float(thresholds["stretch_mr"]), p.stretch_min_mr))
        & ((sign * stretch) < -p.stretch_min_mr)
        & ((sign * reversal_mr) > p.reversal_min_mr)
        & mr_persistence_gate
        & mr_quality_gate
    )
    tf_only = np.asarray(tf_gate & ~mr_gate, dtype=bool)
    mr_only = np.asarray(mr_gate & ~tf_gate, dtype=bool)
    mixed = ~(tf_only | mr_only)
    route = np.full(len(out), "mixed", dtype=object)
    route[tf_only] = "tf"
    route[mr_only] = "mr"

    persisted_params = params_to_dict(p)
    persisted_params["thresholds"] = {str(k): float(v) for k, v in thresholds.items()}
    digest = mr_tf_params_hash(persisted_params)
    out[ROUTE_COL] = route
    out[MR_MASK_COL] = mr_only.astype(np.int8)
    out[TF_MASK_COL] = tf_only.astype(np.int8)
    out[MIXED_MASK_COL] = mixed.astype(np.int8)
    out[PARAMS_HASH_COL] = digest
    counts = {
        "mr": int(np.sum(mr_only)),
        "tf": int(np.sum(tf_only)),
        "mixed": int(np.sum(mixed)),
    }
    n = max(int(len(out)), 1)
    diagnostics = {
        "enabled": True,
        "params": persisted_params,
        "params_hash": digest,
        "source_columns": dict(src),
        "thresholds": dict(persisted_params["thresholds"]),
        "counts": counts,
        "fractions": {k: float(v) / float(n) for k, v in counts.items()},
        "missing_source_columns": [
            k
            for k, v in src.items()
            if v is None and k not in {"persistence_axis", "persistence", "tf_quality_axis", "mr_quality_axis", "quality"}
        ],
        "persistence_axis": persistence_axis,
        "persistence_source_columns": dict(src.get("persistence") or {}),
        "tf_quality_axis": tf_quality_axis,
        "mr_quality_axis": mr_quality_axis,
        "quality_source_columns": dict(src.get("quality") or {}),
    }
    return out, diagnostics


def route_support_diagnostics(
    y: Sequence[float],
    mask: Sequence[bool],
    *,
    min_train_samples: int,
) -> dict[str, Any]:
    mask_arr = np.asarray(mask, dtype=bool)
    y_arr = np.asarray(y, dtype=np.float64)
    n = int(np.sum(mask_arr))
    y_route = y_arr[mask_arr] if len(y_arr) == len(mask_arr) else np.asarray([])
    hard = (np.clip(y_route, 0.0, 1.0) >= 0.5).astype(np.int8) if len(y_route) else np.asarray([], dtype=np.int8)
    classes = int(len(np.unique(hard))) if len(hard) else 0
    ok = n >= int(min_train_samples) and classes >= 2
    reason = None
    if n < int(min_train_samples):
        reason = "too_few_rows"
    elif classes < 2:
        reason = "single_class_route"
    return {
        "ok": bool(ok),
        "reason": reason,
        "n": n,
        "min_train_samples": int(min_train_samples),
        "n_positive": int(np.sum(hard == 1)) if len(hard) else 0,
        "n_negative": int(np.sum(hard == 0)) if len(hard) else 0,
        "class_count": classes,
    }


def topk_return_metrics(
    pred: Sequence[float],
    returns: Sequence[float],
    *,
    top_fracs: Sequence[float] = (0.10, 0.20, 0.30),
) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    n = min(len(p), len(r))
    if n <= 0:
        return {}
    p = p[:n]
    r = r[:n]
    finite = np.isfinite(p) & np.isfinite(r)
    if int(np.sum(finite)) <= 0:
        return {}
    p = p[finite]
    r = r[finite]
    out: dict[str, float] = {"n": float(len(p)), "mean_return_all": float(np.mean(r))}
    for frac in top_fracs:
        k = max(1, int(np.ceil(float(frac) * len(p))))
        idx = np.argsort(p)[-k:]
        tag = int(round(float(frac) * 100))
        ret_top = r[idx]
        out[f"mean_return_top{tag}"] = float(np.mean(ret_top))
        out[f"profit_rate_top{tag}"] = float(np.mean(ret_top > 0.0))
    return out


def compare_specialist_to_baseline(
    *,
    specialist_pred: Sequence[float],
    baseline_pred: Sequence[float],
    returns: Sequence[float],
    margin: float = 0.0,
    top_frac: float = 0.30,
) -> dict[str, Any]:
    spec = topk_return_metrics(specialist_pred, returns)
    base = topk_return_metrics(baseline_pred, returns)
    tag = int(round(float(top_frac) * 100))
    metric = f"mean_return_top{tag}"
    spec_val = float(spec.get(metric, float("nan")))
    base_val = float(base.get(metric, float("nan")))
    uplift = spec_val - base_val if np.isfinite(spec_val) and np.isfinite(base_val) else float("nan")
    return {
        "specialist": spec,
        "baseline": base,
        "promotion_metric": metric,
        "uplift": float(uplift) if np.isfinite(uplift) else float("nan"),
        "margin": float(margin),
        "promoted": bool(np.isfinite(uplift) and uplift > float(margin)),
    }


def suggest_mr_tf_mask_params(
    trial: Any,
    *,
    persistence_axis_choices: Sequence[str] | None = None,
    tf_quality_axis_choices: Sequence[str] | None = None,
    mr_quality_axis_choices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Optuna search space for the upstream MR/TF hard-route gate."""
    persistence_choices = list(persistence_axis_choices or PERSISTENCE_OPTUNA_AXIS_CHOICES)
    tf_quality_choices = list(tf_quality_axis_choices or sorted(TF_QUALITY_AXIS_CHOICES))
    mr_quality_choices = list(mr_quality_axis_choices or sorted(MR_QUALITY_AXIS_CHOICES))
    if "none" not in persistence_choices:
        persistence_choices.insert(0, "none")
    if "none" not in tf_quality_choices:
        tf_quality_choices.insert(0, "none")
    if "none" not in mr_quality_choices:
        mr_quality_choices.insert(0, "none")
    persistence_axis = str(
        trial.suggest_categorical(
            "persistence_axis",
            persistence_choices,
        )
    )
    tf_quality_axis = str(trial.suggest_categorical("tf_quality_axis", tf_quality_choices))
    mr_quality_axis = str(trial.suggest_categorical("mr_quality_axis", mr_quality_choices))
    params = {
        "q_adx_tf": float(trial.suggest_float("q_adx_tf", 0.40, 0.80)),
        "q_adx_mr": float(trial.suggest_float("q_adx_mr", 0.30, 0.75)),
        "q_stretch_mr": float(trial.suggest_float("q_stretch_mr", 0.50, 0.90)),
        "q_tf_quality": float(trial.suggest_float("q_tf_quality", 0.50, 0.85)),
        "q_mr_quality": float(trial.suggest_float("q_mr_quality", 0.50, 0.85)),
        "N_tf": int(trial.suggest_categorical("N_tf", [2, 3, 4, 5, 8])),
        "N_mr": int(trial.suggest_categorical("N_mr", [1, 2, 3])),
        "ema_gap_min_tf": float(trial.suggest_float("ema_gap_min_tf", -0.50, 0.50)),
        "mom_min_tf": float(trial.suggest_float("mom_min_tf", -0.50, 0.50)),
        "stretch_min_mr": float(trial.suggest_float("stretch_min_mr", 0.10, 2.50)),
        "reversal_min_mr": float(
            trial.suggest_float("reversal_min_mr", -0.50, 0.25)
        ),
        "persistence_axis": persistence_axis,
        "tf_quality_axis": tf_quality_axis,
        "mr_quality_axis": mr_quality_axis,
        "q_persist_tf": float(trial.suggest_float("q_persist_tf", 0.55, 0.85)),
        "q_persist_mr": float(trial.suggest_float("q_persist_mr", 0.15, 0.45)),
    }
    return params


def _finite_values(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32, copy=False)


def _fast_quantile(finite_values: np.ndarray, q: float, default: float = 0.0) -> float:
    if finite_values.size == 0:
        return float(default)
    return float(np.nanquantile(finite_values, float(np.clip(q, 0.0, 1.0))))


def _extract_numeric_array(
    frame: pd.DataFrame,
    col: str | None,
    *,
    n: int,
    default: float = 0.0,
) -> np.ndarray:
    if col is None or col not in frame.columns:
        return np.full(n, float(default), dtype=np.float32)
    return pd.to_numeric(frame[col].iloc[:n], errors="coerce").to_numpy(
        dtype=np.float32,
        copy=True,
    )


def _scaled_unit_array(values: np.ndarray, *, mode: str) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = vals.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    if mode == "autocorr":
        out[finite] = (np.clip(out[finite], -1.0, 1.0) + 1.0) * 0.5
        return out
    if mode in {"choppiness", "entropy"}:
        finite_vals = out[finite]
        if finite_vals.size > 0 and float(np.nanmax(finite_vals)) > 2.0:
            out[finite] = np.clip(1.0 - (out[finite] / 100.0), 0.0, 1.0)
        else:
            out[finite] = 1.0 - np.clip(out[finite], 0.0, 1.0)
        return out
    out[finite] = np.clip(out[finite], 0.0, 1.0)
    return out


def _compression_quality_array(values: np.ndarray, col: str | None) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    name = str(col or "").lower()
    if "band_width" in name or name.endswith("_width"):
        return (-vals).astype(np.float32, copy=False)
    return vals


def _range_position_quality_array(
    values: np.ndarray,
    *,
    side_sign: float,
    col: str | None,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    name = str(col or "").lower()
    if "pos" in name or (finite.size > 0 and float(np.mean((finite >= 0.0) & (finite <= 1.0))) > 0.80):
        centered = vals - np.float32(0.5)
    else:
        centered = vals
    return (-float(side_sign) * centered).astype(np.float32, copy=False)


def _rsi_reversion_quality_array(values: np.ndarray, *, side_sign: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size > 0 and float(np.nanquantile(finite, 0.90)) > 2.0:
        centered = (vals - np.float32(50.0)) / np.float32(50.0)
    elif (
        finite.size > 0
        and float(np.nanquantile(finite, 0.90)) <= 1.0
        and float(np.nanquantile(finite, 0.10)) >= 0.0
    ):
        centered = vals - np.float32(0.5)
    else:
        centered = vals
    return (-float(side_sign) * centered).astype(np.float32, copy=False)


def _quality_score_from_arrays(
    arrays: Mapping[str, Any],
    *,
    axis: str,
    route: str,
    side_sign: float,
) -> np.ndarray:
    adx = np.asarray(arrays["adx"], dtype=np.float32)
    axis_clean = str(axis or "none").strip().lower()
    if axis_clean == "none":
        return np.full_like(adx, np.nan, dtype=np.float32)
    quality_arrays = arrays.get("quality_arrays") or {}
    quality_sources = arrays.get("source_columns", {}).get("quality", {}) or {}
    raw = np.asarray(
        quality_arrays.get(axis_clean, np.full_like(adx, np.nan)),
        dtype=np.float32,
    )
    col = quality_sources.get(axis_clean) if isinstance(quality_sources, Mapping) else None
    if route == "mr" and axis_clean == "range_position":
        return _range_position_quality_array(raw, side_sign=side_sign, col=col)
    if route == "mr" and axis_clean == "rsi_reversion":
        return _rsi_reversion_quality_array(raw, side_sign=side_sign)
    if route == "mr" and axis_clean == "compression":
        return _compression_quality_array(raw, col)
    return raw


def _available_quality_axes(
    arrays: Mapping[str, Any],
    *,
    choices: set[str],
    route: str,
    side_sign: float,
    min_rows: int,
) -> list[str]:
    out = ["none"]
    for axis in sorted(choices - {"none"}):
        score = _quality_score_from_arrays(
            arrays,
            axis=axis,
            route=route,
            side_sign=side_sign,
        )
        if int(np.isfinite(score).sum()) >= int(min_rows):
            out.append(axis)
    return out


def _persistence_scores_from_arrays(
    persistence_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    empty = None
    for arr in persistence_arrays.values():
        empty = np.full_like(np.asarray(arr, dtype=np.float32), np.nan, dtype=np.float32)
        break
    if empty is None:
        return {}
    scores["none"] = empty.copy()
    for axis in ("hurst", "autocorr", "efficiency"):
        arr = persistence_arrays.get(axis)
        scores[axis] = (
            np.asarray(arr, dtype=np.float32)
            if arr is not None
            else empty.copy()
        )
    components: list[np.ndarray] = []
    mode_by_axis = {
        "hurst": "bounded",
        "autocorr": "autocorr",
        "efficiency": "bounded",
        "choppiness": "choppiness",
        "entropy": "entropy",
    }
    for axis, mode in mode_by_axis.items():
        arr = persistence_arrays.get(axis)
        if arr is None:
            continue
        scaled = _scaled_unit_array(np.asarray(arr, dtype=np.float32), mode=mode)
        if np.isfinite(scaled).any():
            components.append(scaled)
    if components:
        stacked = np.vstack(components).astype(np.float32, copy=False)
        valid = np.isfinite(stacked)
        counts = valid.sum(axis=0).astype(np.float32)
        summed = np.where(valid, stacked, 0.0).sum(axis=0)
        comp = np.full(stacked.shape[1], np.nan, dtype=np.float32)
        nz = counts > 0
        comp[nz] = summed[nz] / counts[nz]
        scores["composite"] = comp
    else:
        scores["composite"] = empty.copy()
    return scores


def _prepare_mr_tf_optuna_arrays(
    frame: pd.DataFrame,
    *,
    n: int,
) -> dict[str, Any]:
    base_params = DEFAULT_PARAMS
    base_src = _source_columns(frame, base_params)
    adx = _extract_numeric_array(frame, base_src["adx"], n=n)
    ema_gap = _extract_numeric_array(frame, base_src["ema_gap"], n=n)
    stretch = _extract_numeric_array(frame, base_src["stretch"], n=n)
    mom_by_n: dict[int, np.ndarray] = {}
    rev_by_n: dict[int, np.ndarray] = {}
    persistence_sources = _persistence_source_columns(frame)
    persistence_arrays: dict[str, np.ndarray] = {
        key: _extract_numeric_array(frame, col, n=n, default=np.nan)
        for key, col in persistence_sources.items()
    }
    persistence_scores = _persistence_scores_from_arrays(persistence_arrays)
    quality_sources = _quality_source_columns(frame)
    quality_arrays: dict[str, np.ndarray] = {
        key: _extract_numeric_array(frame, col, n=n, default=np.nan)
        for key, col in quality_sources.items()
    }
    source_columns: dict[str, Any] = {
        "adx": base_src["adx"],
        "ema_gap": base_src["ema_gap"],
        "stretch": base_src["stretch"],
        "mom_tf": {},
        "reversal_mr": {},
        "persistence": dict(persistence_sources),
        "quality": dict(quality_sources),
    }
    for lookback in (2, 3, 4, 5, 8):
        col = _first_existing(frame, _momentum_aliases(lookback))
        mom_by_n[int(lookback)] = _extract_numeric_array(frame, col, n=n)
        source_columns["mom_tf"][str(lookback)] = col
    for lookback in (1, 2, 3):
        col = _first_existing(frame, _momentum_aliases(lookback))
        rev_by_n[int(lookback)] = _extract_numeric_array(frame, col, n=n)
        source_columns["reversal_mr"][str(lookback)] = col
    abs_stretch = np.abs(stretch).astype(np.float32, copy=False)
    return {
        "adx": adx,
        "ema_gap": ema_gap,
        "stretch": stretch,
        "abs_stretch": abs_stretch,
        "mom_by_n": mom_by_n,
        "rev_by_n": rev_by_n,
        "persistence_scores": persistence_scores,
        "quality_arrays": quality_arrays,
        "finite_persistence_values": {
            axis: _finite_values(values)
            for axis, values in persistence_scores.items()
            if axis != "none"
        },
        "finite_adx_values": _finite_values(adx),
        "finite_abs_stretch_values": _finite_values(abs_stretch),
        "source_columns": source_columns,
        "missing_source_columns": [
            key
            for key, value in base_src.items()
            if value is None and key in {"adx", "ema_gap", "stretch"}
        ],
    }


def _route_masks_from_arrays(
    arrays: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    side_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    adx = np.asarray(arrays["adx"], dtype=np.float32)
    ema_gap = np.asarray(arrays["ema_gap"], dtype=np.float32)
    stretch = np.asarray(arrays["stretch"], dtype=np.float32)
    abs_stretch = np.asarray(arrays["abs_stretch"], dtype=np.float32)
    n_tf = int(params.get("N_tf", DEFAULT_PARAMS.N_tf))
    n_mr = int(params.get("N_mr", DEFAULT_PARAMS.N_mr))
    mom_tf = np.asarray(arrays["mom_by_n"].get(n_tf), dtype=np.float32)
    reversal_mr = np.asarray(arrays["rev_by_n"].get(n_mr), dtype=np.float32)
    persistence_axis = str(
        params.get("persistence_axis", DEFAULT_PARAMS.persistence_axis) or "none"
    ).strip().lower()
    if persistence_axis not in PERSISTENCE_AXIS_CHOICES:
        persistence_axis = DEFAULT_PARAMS.persistence_axis
    persistence_scores = arrays.get("persistence_scores") or {}
    persistence_score = np.asarray(
        persistence_scores.get(persistence_axis, np.full_like(adx, np.nan)),
        dtype=np.float32,
    )
    tf_quality_axis = str(
        params.get("tf_quality_axis", DEFAULT_PARAMS.tf_quality_axis) or "none"
    ).strip().lower()
    if tf_quality_axis not in TF_QUALITY_AXIS_CHOICES:
        tf_quality_axis = DEFAULT_PARAMS.tf_quality_axis
    mr_quality_axis = str(
        params.get("mr_quality_axis", DEFAULT_PARAMS.mr_quality_axis) or "none"
    ).strip().lower()
    if mr_quality_axis not in MR_QUALITY_AXIS_CHOICES:
        mr_quality_axis = DEFAULT_PARAMS.mr_quality_axis
    tf_quality_score = _quality_score_from_arrays(
        arrays,
        axis=tf_quality_axis,
        route="tf",
        side_sign=side_sign,
    )
    mr_quality_score = _quality_score_from_arrays(
        arrays,
        axis=mr_quality_axis,
        route="mr",
        side_sign=side_sign,
    )

    thresholds_raw = params.get("thresholds")
    thresholds = dict(thresholds_raw) if isinstance(thresholds_raw, Mapping) else {}
    if "adx_tf" not in thresholds:
        thresholds["adx_tf"] = _fast_quantile(
            np.asarray(arrays["finite_adx_values"], dtype=np.float32),
            float(params.get("q_adx_tf", DEFAULT_PARAMS.q_adx_tf)),
        )
    if "adx_mr" not in thresholds:
        thresholds["adx_mr"] = _fast_quantile(
            np.asarray(arrays["finite_adx_values"], dtype=np.float32),
            float(params.get("q_adx_mr", DEFAULT_PARAMS.q_adx_mr)),
        )
    if "stretch_mr" not in thresholds:
        thresholds["stretch_mr"] = _fast_quantile(
            np.asarray(arrays["finite_abs_stretch_values"], dtype=np.float32),
            float(params.get("q_stretch_mr", DEFAULT_PARAMS.q_stretch_mr)),
        )
    if persistence_axis != "none":
        finite_persist_values = (arrays.get("finite_persistence_values") or {}).get(
            persistence_axis,
            np.asarray([], dtype=np.float32),
        )
        if "persist_tf" not in thresholds:
            thresholds["persist_tf"] = _fast_quantile(
                np.asarray(finite_persist_values, dtype=np.float32),
                float(params.get("q_persist_tf", DEFAULT_PARAMS.q_persist_tf)),
            )
        if "persist_mr" not in thresholds:
            thresholds["persist_mr"] = _fast_quantile(
                np.asarray(finite_persist_values, dtype=np.float32),
                float(params.get("q_persist_mr", DEFAULT_PARAMS.q_persist_mr)),
            )
    if tf_quality_axis != "none" and "tf_quality" not in thresholds:
        thresholds["tf_quality"] = _fast_quantile(
            _finite_values(tf_quality_score),
            float(params.get("q_tf_quality", DEFAULT_PARAMS.q_tf_quality)),
        )
    if mr_quality_axis != "none" and "mr_quality" not in thresholds:
        thresholds["mr_quality"] = _fast_quantile(
            _finite_values(mr_quality_score),
            float(params.get("q_mr_quality", DEFAULT_PARAMS.q_mr_quality)),
        )

    finite_adx = np.isfinite(adx)
    finite_tf = finite_adx & np.isfinite(ema_gap) & np.isfinite(mom_tf)
    finite_mr = finite_adx & np.isfinite(stretch) & np.isfinite(reversal_mr)
    if persistence_axis != "none":
        finite_persistence = np.isfinite(persistence_score)
        tf_persistence_gate = finite_persistence & (
            persistence_score > float(thresholds["persist_tf"])
        )
        mr_persistence_gate = finite_persistence & (
            persistence_score < float(thresholds["persist_mr"])
        )
    else:
        tf_persistence_gate = np.ones_like(finite_tf, dtype=bool)
        mr_persistence_gate = np.ones_like(finite_mr, dtype=bool)
    if tf_quality_axis != "none":
        tf_quality_gate = np.isfinite(tf_quality_score) & (
            tf_quality_score > float(thresholds["tf_quality"])
        )
    else:
        tf_quality_gate = np.ones_like(finite_tf, dtype=bool)
    if mr_quality_axis != "none":
        mr_quality_gate = np.isfinite(mr_quality_score) & (
            mr_quality_score > float(thresholds["mr_quality"])
        )
    else:
        mr_quality_gate = np.ones_like(finite_mr, dtype=bool)
    tf_gate = (
        finite_tf
        & (adx > float(thresholds["adx_tf"]))
        & ((float(side_sign) * ema_gap) > float(params.get("ema_gap_min_tf", 0.0)))
        & ((float(side_sign) * mom_tf) > float(params.get("mom_min_tf", 0.0)))
        & tf_persistence_gate
        & tf_quality_gate
    )
    stretch_floor = max(
        float(thresholds["stretch_mr"]),
        float(params.get("stretch_min_mr", DEFAULT_PARAMS.stretch_min_mr)),
    )
    mr_gate = (
        finite_mr
        & (adx < float(thresholds["adx_mr"]))
        & (abs_stretch > stretch_floor)
        & (
            (float(side_sign) * stretch)
            < -float(params.get("stretch_min_mr", DEFAULT_PARAMS.stretch_min_mr))
        )
        & (
            (float(side_sign) * reversal_mr)
            > float(params.get("reversal_min_mr", DEFAULT_PARAMS.reversal_min_mr))
        )
        & mr_persistence_gate
        & mr_quality_gate
    )
    tf_only = np.asarray(tf_gate & ~mr_gate, dtype=bool)
    mr_only = np.asarray(mr_gate & ~tf_gate, dtype=bool)
    mixed = ~(tf_only | mr_only)
    return mr_only, tf_only, mixed, {str(k): float(v) for k, v in thresholds.items()}


def _route_support_fast(
    hard_labels: np.ndarray,
    mask: np.ndarray,
    *,
    min_train_samples: int,
) -> dict[str, Any]:
    mask_arr = np.asarray(mask, dtype=bool)
    hard = np.asarray(hard_labels, dtype=np.int8)
    n = int(mask_arr.sum())
    if n <= 0 or len(hard) != len(mask_arr):
        n_pos = 0
    else:
        n_pos = int(np.sum(hard[mask_arr] == 1))
    n_neg = int(n - n_pos)
    class_count = int((n_pos > 0) + (n_neg > 0))
    ok = n >= int(min_train_samples) and class_count >= 2
    reason = None
    if n < int(min_train_samples):
        reason = "too_few_rows"
    elif class_count < 2:
        reason = "single_class_route"
    return {
        "ok": bool(ok),
        "reason": reason,
        "n": n,
        "min_train_samples": int(min_train_samples),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "class_count": class_count,
    }


def _route_score_numpy(
    route_pred: np.ndarray,
    route_ret: np.ndarray,
    *,
    min_rows: int,
) -> float:
    if len(route_ret) < int(min_rows):
        return float("nan")
    p = np.asarray(route_pred, dtype=np.float32)
    r = np.asarray(route_ret, dtype=np.float32)
    finite = np.isfinite(p) & np.isfinite(r)
    if int(finite.sum()) < int(min_rows):
        return float("nan")
    p = p[finite]
    r = r[finite]
    n = len(r)
    total = 0.0
    for weight, frac in ((0.50, 0.30), (0.30, 0.20), (0.20, 0.10)):
        k = max(1, int(np.ceil(float(frac) * n)))
        idx = np.argpartition(p, n - k)[n - k :]
        top_ret = r[idx]
        total += float(weight) * float(np.mean(top_ret)) * max(
            float(np.mean(top_ret > 0.0)),
            1e-6,
        )
    return float(total)


if njit is not None:

    @njit(cache=True)  # type: ignore[misc]
    def _route_score_numba_impl(
        route_pred: np.ndarray,
        route_ret: np.ndarray,
        min_rows: int,
    ) -> float:
        finite_count = 0
        for i in range(route_ret.shape[0]):
            if np.isfinite(route_pred[i]) and np.isfinite(route_ret[i]):
                finite_count += 1
        if finite_count < min_rows:
            return np.nan
        p = np.empty(finite_count, dtype=np.float32)
        r = np.empty(finite_count, dtype=np.float32)
        j = 0
        for i in range(route_ret.shape[0]):
            if np.isfinite(route_pred[i]) and np.isfinite(route_ret[i]):
                p[j] = route_pred[i]
                r[j] = route_ret[i]
                j += 1
        order = np.argsort(p)
        total = 0.0
        fracs = (0.30, 0.20, 0.10)
        weights = (0.50, 0.30, 0.20)
        n = finite_count
        for ix in range(3):
            k = int(np.ceil(fracs[ix] * n))
            if k < 1:
                k = 1
            mean_ret = 0.0
            hit = 0.0
            for pos in range(n - k, n):
                val = r[order[pos]]
                mean_ret += val
                if val > 0.0:
                    hit += 1.0
            mean_ret /= k
            hit /= k
            if hit < 1e-6:
                hit = 1e-6
            total += weights[ix] * mean_ret * hit
        return total

else:
    _route_score_numba_impl = None


def _route_score_fast(
    route_pred: np.ndarray,
    route_ret: np.ndarray,
    *,
    min_rows: int,
    use_numba: bool,
) -> float:
    if use_numba and _route_score_numba_impl is not None:
        return float(
            _route_score_numba_impl(
                np.asarray(route_pred, dtype=np.float32),
                np.asarray(route_ret, dtype=np.float32),
                int(min_rows),
            )
        )
    return _route_score_numpy(route_pred, route_ret, min_rows=int(min_rows))


def optimize_mr_tf_mask_params(
    frame: pd.DataFrame,
    *,
    y: Sequence[float],
    returns: Sequence[float],
    side: str | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the optional upstream mask-only Optuna search.

    The objective is intentionally a cheap route-quality proxy. It does not tune
    specialist model internals; those remain governed by the existing
    train_base/train_meta LGBM search spaces once rows are routed.
    """
    section = _section(cfg)
    n_trials = int(section.get("optuna_trials", section.get("trials", 40)) or 40)
    min_rows = int(section.get("min_train_samples", 400) or 400)
    patience = int(
        section.get(
            "optuna_patience",
            section.get("no_improvement_trials", section.get("patience", 20)),
        )
        or 20
    )
    if n_trials <= 0:
        params = params_to_dict(mask_params_from_cfg(cfg))
        return params, {"enabled": False, "reason": "no_trials"}
    try:
        import optuna
    except Exception as exc:
        params = params_to_dict(mask_params_from_cfg(cfg))
        return params, {"enabled": False, "reason": f"optuna_unavailable:{exc}"}

    y_arr = np.asarray(y, dtype=np.float32)
    ret_arr = np.asarray(returns, dtype=np.float32)
    n = min(len(frame), len(y_arr), len(ret_arr))
    y_arr = y_arr[:n]
    ret_arr = ret_arr[:n]
    hard_labels = (np.clip(y_arr, 0.0, 1.0) >= 0.5).astype(np.int8)
    frame_head = frame.iloc[:n]
    arrays = _prepare_mr_tf_optuna_arrays(frame_head, n=n)
    side_sign = _side_sign(side)
    use_numba = bool(
        section.get("optuna_use_numba", section.get("use_numba", True))
    ) and n >= int(section.get("optuna_numba_min_rows", 20_000) or 20_000)
    persistence_axis_choices = ["none"] + [
        axis
        for axis in PERSISTENCE_OPTUNA_AXIS_CHOICES
        if axis != "none"
        and int(
            np.isfinite(
                np.asarray((arrays.get("persistence_scores") or {}).get(axis, []), dtype=np.float32)
            ).sum()
        )
        >= min_rows
    ]
    tf_quality_axis_choices = _available_quality_axes(
        arrays,
        choices=TF_QUALITY_AXIS_CHOICES,
        route="tf",
        side_sign=side_sign,
        min_rows=min_rows,
    )
    mr_quality_axis_choices = _available_quality_axes(
        arrays,
        choices=MR_QUALITY_AXIS_CHOICES,
        route="mr",
        side_sign=side_sign,
        min_rows=min_rows,
    )
    baseline_score = _route_score_fast(
        y_arr,
        ret_arr,
        min_rows=min_rows,
        use_numba=use_numba,
    )
    if not np.isfinite(baseline_score):
        baseline_score = 0.0
    support_loss_hurdle_abs = float(section.get("support_loss_hurdle", 0.0) or 0.0)
    support_loss_hurdle_ratio = float(
        section.get("support_loss_hurdle_ratio", 0.0015) or 0.0
    )
    support_loss_hurdle_floor = float(
        section.get("support_loss_hurdle_floor", 0.0) or 0.0
    )
    support_loss_hurdle = max(
        support_loss_hurdle_abs,
        support_loss_hurdle_floor,
        abs(float(baseline_score)) * max(support_loss_hurdle_ratio, 0.0),
    )
    support_value_power = float(section.get("support_value_power", 0.5) or 0.0)
    min_coverage = float(section.get("min_coverage", 0.0) or 0.0)
    support_loss_quadratic_multiplier = float(
        section.get("support_loss_quadratic_multiplier", 0.0) or 0.0
    )
    support_loss_hard_veto = _truthy(section.get("support_loss_hard_veto", False))
    min_earned_quality_uplift = float(
        section.get("min_earned_quality_uplift", 0.0) or 0.0
    )

    def _required_uplift_for_coverage(route_coverage: float) -> float:
        support_loss = max(0.0, 1.0 - float(route_coverage))
        return float(
            support_loss
            * support_loss_hurdle
            * (
                1.0
                + max(0.0, support_loss_quadratic_multiplier) * support_loss
            )
        )

    def objective(trial: Any) -> float:
        trial_params = suggest_mr_tf_mask_params(
            trial,
            persistence_axis_choices=persistence_axis_choices,
            tf_quality_axis_choices=tf_quality_axis_choices,
            mr_quality_axis_choices=mr_quality_axis_choices,
        )
        mr_mask, tf_mask, mixed_mask, thresholds = _route_masks_from_arrays(
            arrays,
            trial_params,
            side_sign=side_sign,
        )
        mr_support = _route_support_fast(
            hard_labels, mr_mask, min_train_samples=min_rows
        )
        tf_support = _route_support_fast(
            hard_labels, tf_mask, min_train_samples=min_rows
        )
        counts = {
            "mr": int(np.sum(mr_mask)),
            "tf": int(np.sum(tf_mask)),
            "mixed": int(np.sum(mixed_mask)),
        }
        route_specs = {
            "mr": {"mask": mr_mask, "support": mr_support},
            "tf": {"mask": tf_mask, "support": tf_support},
        }
        supported_routes = [
            name for name, spec in route_specs.items() if bool(spec["support"].get("ok"))
        ]
        if not supported_routes:
            trial.set_user_attr("mr_support", dict(mr_support))
            trial.set_user_attr("tf_support", dict(tf_support))
            trial.set_user_attr("counts", dict(counts))
            trial.set_user_attr("supported_routes", [])
            raise optuna.TrialPruned(
                f"no_supported_routes: mr={mr_support}, tf={tf_support}"
            )

        route_scores: dict[str, float] = {}
        route_counts: dict[str, int] = {}
        route_values: dict[str, float] = {}
        routed_mask = np.zeros(n, dtype=bool)
        for route_name in supported_routes:
            route_mask = np.asarray(route_specs[route_name]["mask"], dtype=bool)
            route_score = _route_score_fast(
                y_arr[route_mask],
                ret_arr[route_mask],
                min_rows=min_rows,
                use_numba=use_numba,
            )
            if not np.isfinite(route_score):
                continue
            routed_mask |= route_mask
            route_count = int(np.sum(route_mask))
            route_coverage = float(route_count) / max(float(n), 1.0)
            route_required_uplift = _required_uplift_for_coverage(route_coverage)
            route_scores[route_name] = float(route_score)
            route_counts[route_name] = int(route_count)
            route_values[route_name] = (
                float(route_score) - float(baseline_score) - route_required_uplift
            ) * (max(route_coverage, 1e-6) ** max(support_value_power, 0.0))

        if not route_scores:
            trial.set_user_attr("mr_support", dict(mr_support))
            trial.set_user_attr("tf_support", dict(tf_support))
            trial.set_user_attr("counts", dict(counts))
            trial.set_user_attr("supported_routes", list(supported_routes))
            raise optuna.TrialPruned("no_finite_supported_route_scores")

        coverage = float(np.sum(routed_mask)) / max(float(n), 1.0)
        trial.set_user_attr("counts", dict(counts))
        trial.set_user_attr("supported_routes", list(route_scores.keys()))
        trial.set_user_attr("route_support", {"mr": dict(mr_support), "tf": dict(tf_support)})
        trial.set_user_attr("route_counts_supported", dict(route_counts))
        trial.set_user_attr("route_values", dict(route_values))
        trial.set_user_attr("coverage", float(coverage))
        if coverage < max(0.0, min_coverage):
            raise optuna.TrialPruned(
                f"coverage_below_minimum: coverage={coverage:.4f}, "
                f"min_coverage={min_coverage:.4f}"
            )

        trial.report(float(max(route_values.values())), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned("median_pruner_after_best_route_score")
        union_mask = np.asarray(routed_mask, dtype=bool)
        union_score = _route_score_fast(
            y_arr[union_mask],
            ret_arr[union_mask],
            min_rows=min_rows,
            use_numba=use_numba,
        )
        if not np.isfinite(union_score):
            raise optuna.TrialPruned("non-finite routed-union score")
        routed_count = max(int(sum(route_counts.values())), 1)
        route_weighted_score = sum(
            float(route_scores[name]) * float(route_counts.get(name, 0))
            for name in route_scores
        ) / float(routed_count)
        quality_score = 0.60 * float(union_score) + 0.40 * float(route_weighted_score)
        quality_uplift = float(quality_score) - float(baseline_score)
        support_loss = max(0.0, 1.0 - float(coverage))
        required_uplift = _required_uplift_for_coverage(coverage)
        earned_quality_uplift = quality_uplift - required_uplift
        trial.set_user_attr("baseline_score", float(baseline_score))
        trial.set_user_attr("mr_score", float(route_scores.get("mr", np.nan)))
        trial.set_user_attr("tf_score", float(route_scores.get("tf", np.nan)))
        trial.set_user_attr("union_score", float(union_score))
        trial.set_user_attr("route_weighted_score", float(route_weighted_score))
        trial.set_user_attr("quality_score", float(quality_score))
        trial.set_user_attr("quality_uplift", float(quality_uplift))
        trial.set_user_attr("required_uplift", float(required_uplift))
        trial.set_user_attr("earned_quality_uplift", float(earned_quality_uplift))
        trial.set_user_attr("support_loss", float(support_loss))
        if support_loss_hard_veto and earned_quality_uplift <= min_earned_quality_uplift:
            raise optuna.TrialPruned(
                "support_reduction_not_earned:"
                f"coverage={coverage:.4f},"
                f"quality_uplift={quality_uplift:.8g},"
                f"required_uplift={required_uplift:.8g},"
                f"support_loss_quadratic_multiplier={support_loss_quadratic_multiplier:.4g}"
            )
        value = float(
            earned_quality_uplift * (max(coverage, 1e-6) ** max(support_value_power, 0.0))
            + 1e-9 * quality_score
        )
        trial.report(value, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned("median_pruner_after_combined_score")
        persisted = params_to_dict(mask_params_from_cfg(cfg, overrides=trial_params))
        persisted["thresholds"] = dict(thresholds)
        trial.set_user_attr("thresholds", dict(thresholds))
        trial.set_user_attr("params_hash", mr_tf_params_hash(persisted))
        trial.set_user_attr("support_loss_hurdle", float(support_loss_hurdle))
        trial.set_user_attr("support_loss_hurdle_ratio", float(support_loss_hurdle_ratio))
        trial.set_user_attr("support_loss_hurdle_abs", float(support_loss_hurdle_abs))
        trial.set_user_attr("support_loss_hurdle_floor", float(support_loss_hurdle_floor))
        trial.set_user_attr("support_value_power", float(support_value_power))
        trial.set_user_attr("min_coverage", float(min_coverage))
        trial.set_user_attr(
            "support_loss_quadratic_multiplier",
            float(support_loss_quadratic_multiplier),
        )
        trial.set_user_attr(
            "persistence_axis",
            str(trial_params.get("persistence_axis", DEFAULT_PARAMS.persistence_axis)),
        )
        trial.set_user_attr(
            "tf_quality_axis",
            str(trial_params.get("tf_quality_axis", DEFAULT_PARAMS.tf_quality_axis)),
        )
        trial.set_user_attr(
            "mr_quality_axis",
            str(trial_params.get("mr_quality_axis", DEFAULT_PARAMS.mr_quality_axis)),
        )
        return value

    best_seen = {"value": -np.inf, "stale": 0}

    def _early_stop_callback(study: Any, trial: Any) -> None:
        if patience <= 0:
            return
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        try:
            current_best = float(study.best_value)
        except Exception:
            return
        if current_best > float(best_seen["value"]) + 1e-12:
            best_seen["value"] = current_best
            best_seen["stale"] = 0
        else:
            best_seen["stale"] = int(best_seen["stale"]) + 1
        if int(best_seen["stale"]) >= int(patience):
            study.stop()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(section.get("seed", 42))),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=int(section.get("optuna_n_startup_trials", 8) or 8),
            n_warmup_steps=int(section.get("optuna_n_warmup_steps", 0) or 0),
        ),
    )
    default_trial = {
        k: v
        for k, v in params_to_dict(mask_params_from_cfg(cfg)).items()
        if k != "thresholds"
    }
    if str(default_trial.get("persistence_axis", "none")) not in persistence_axis_choices:
        default_trial["persistence_axis"] = "none"
    if str(default_trial.get("tf_quality_axis", "none")) not in tf_quality_axis_choices:
        default_trial["tf_quality_axis"] = "none"
    if str(default_trial.get("mr_quality_axis", "none")) not in mr_quality_axis_choices:
        default_trial["mr_quality_axis"] = "none"
    seed_trials = [default_trial]
    seed_trials.extend(
        [
            {
                **default_trial,
                "q_adx_tf": 0.50,
                "q_adx_mr": 0.65,
                "q_stretch_mr": 0.60,
                "N_tf": 3,
                "N_mr": 2,
                "ema_gap_min_tf": -0.10,
                "mom_min_tf": -0.10,
                "stretch_min_mr": 0.25,
                "reversal_min_mr": -0.25,
                "persistence_axis": "none",
                "tf_quality_axis": "none",
                "mr_quality_axis": "none",
            },
            {
                **default_trial,
                "q_adx_tf": 0.45,
                "q_adx_mr": 0.70,
                "q_stretch_mr": 0.55,
                "N_tf": 3,
                "N_mr": 2,
                "ema_gap_min_tf": -0.25,
                "mom_min_tf": -0.25,
                "stretch_min_mr": 0.15,
                "reversal_min_mr": -0.35,
                "persistence_axis": "none",
                "tf_quality_axis": "none",
                "mr_quality_axis": "none",
            },
            {
                **default_trial,
                "q_adx_tf": 0.40,
                "q_adx_mr": 0.75,
                "q_stretch_mr": 0.50,
                "N_tf": 3,
                "N_mr": 2,
                "ema_gap_min_tf": -0.50,
                "mom_min_tf": -0.50,
                "stretch_min_mr": 0.10,
                "reversal_min_mr": -0.50,
                "persistence_axis": "none",
                "tf_quality_axis": "none",
                "mr_quality_axis": "none",
            },
        ]
    )
    for _seed_trial in seed_trials:
        study.enqueue_trial(_seed_trial)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_early_stop_callback],
    )

    def _trial_snapshot(trial: Any) -> dict[str, Any]:
        attrs = dict(getattr(trial, "user_attrs", {}) or {})
        value = getattr(trial, "value", None)
        state = getattr(getattr(trial, "state", None), "name", str(getattr(trial, "state", "")))
        out = {
            "number": int(getattr(trial, "number", -1)),
            "state": str(state),
            "value": float(value) if value is not None and np.isfinite(value) else None,
            "coverage": attrs.get("coverage"),
            "quality_uplift": attrs.get("quality_uplift"),
            "required_uplift": attrs.get("required_uplift"),
            "earned_quality_uplift": attrs.get("earned_quality_uplift"),
            "supported_routes": attrs.get("supported_routes"),
            "counts": attrs.get("counts"),
            "route_support": attrs.get("route_support"),
            "route_counts_supported": attrs.get("route_counts_supported"),
        }
        return out

    pruned_snapshots = [_trial_snapshot(t) for t in study.trials]
    pruned_snapshots.sort(
        key=lambda row: (
            -float(row.get("coverage") or 0.0),
            -float(row.get("quality_uplift") or -1e9),
        )
    )
    pruned_snapshots = pruned_snapshots[:10]
    try:
        best_trial = study.best_trial
        best_value = study.best_value
    except Exception:
        params = params_to_dict(mask_params_from_cfg(cfg))
        return params, {
            "enabled": True,
            "selected": False,
            "reason": "no_best_trial",
            "n_trials": int(len(study.trials)),
            "patience": int(patience),
            "objective": "route_quality_proxy_support_cost_adjusted",
            "baseline_score": float(baseline_score),
            "support_loss_hurdle": float(support_loss_hurdle),
            "support_loss_hurdle_ratio": float(support_loss_hurdle_ratio),
            "support_loss_hurdle_abs": float(support_loss_hurdle_abs),
            "support_loss_hurdle_floor": float(support_loss_hurdle_floor),
            "support_value_power": float(support_value_power),
            "min_coverage": float(min_coverage),
            "support_loss_quadratic_multiplier": float(
                support_loss_quadratic_multiplier
            ),
            "support_loss_hard_veto": bool(support_loss_hard_veto),
            "min_earned_quality_uplift": float(min_earned_quality_uplift),
            "array_eval": True,
            "dtype_policy": "float32_features_returns_labels_bool_masks",
            "numba_available": bool(_route_score_numba_impl is not None),
            "numba_used": bool(use_numba and _route_score_numba_impl is not None),
            "source_columns": dict(arrays.get("source_columns") or {}),
            "available_persistence_axes": list(persistence_axis_choices),
            "available_tf_quality_axes": list(tf_quality_axis_choices),
            "available_mr_quality_axes": list(mr_quality_axis_choices),
            "missing_source_columns": list(arrays.get("missing_source_columns") or []),
            "best_pruned_candidates": pruned_snapshots,
            "n_pruned": int(
                sum(
                    1
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                )
            ),
        }
    routed, diag = apply_mr_tf_masks(
        frame_head.reset_index(drop=True),
        side=side,
        cfg=cfg,
        params=best_trial.params,
    )
    best_params = dict(diag.get("params") or best_trial.params)
    return best_params, {
        "enabled": True,
        "selected": True,
        "objective": "route_quality_proxy_support_cost_adjusted",
        "best_value": float(best_value),
        "best_trial": int(best_trial.number),
        "best_params": dict(best_params),
        "best_counts": dict((diag or {}).get("counts") or {}),
        "best_params_hash": diag.get("params_hash", ""),
        "best_objective_diagnostics": dict(best_trial.user_attrs),
        "baseline_score": float(baseline_score),
        "support_loss_hurdle": float(support_loss_hurdle),
        "support_loss_hurdle_ratio": float(support_loss_hurdle_ratio),
        "support_loss_hurdle_abs": float(support_loss_hurdle_abs),
        "support_loss_hurdle_floor": float(support_loss_hurdle_floor),
        "support_value_power": float(support_value_power),
        "min_coverage": float(min_coverage),
        "support_loss_quadratic_multiplier": float(
            support_loss_quadratic_multiplier
        ),
        "support_loss_hard_veto": bool(support_loss_hard_veto),
        "min_earned_quality_uplift": float(min_earned_quality_uplift),
        "n_trials": int(len(study.trials)),
        "patience": int(patience),
        "stopped_by_patience": bool(int(best_seen["stale"]) >= int(patience) if patience > 0 else False),
        "array_eval": True,
        "dtype_policy": "float32_features_returns_labels_bool_masks",
        "numba_available": bool(_route_score_numba_impl is not None),
        "numba_used": bool(use_numba and _route_score_numba_impl is not None),
        "source_columns": dict(arrays.get("source_columns") or {}),
        "available_persistence_axes": list(persistence_axis_choices),
        "available_tf_quality_axes": list(tf_quality_axis_choices),
        "available_mr_quality_axes": list(mr_quality_axis_choices),
        "missing_source_columns": list(arrays.get("missing_source_columns") or []),
        "best_pruned_candidates": pruned_snapshots,
        "n_pruned": int(
            sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        ),
    }
