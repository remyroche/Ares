#!/usr/bin/env python3
"""Performance-first regime break diagnostics for OOF base/meta predictions.

This script is deliberately diagnostic-only. It does not mutate model artifacts,
training configs, or policy outputs.

The workflow is:
  1. Load base/meta OOF prediction files for a training run.
  2. Detect all meaningfully bad 3-calendar-day hit-rate-surprise windows.
  3. Merge overlapping bad windows into performance-break episodes.
  4. For safe feature columns, score Shift, Relevance, and Harmfulness.
  5. Compute feature-matrix covariance/correlation/autocorrelation diagnostics.

The useful regime candidates are the intersection:
  shifted features AND predictively relevant features AND harmful shifted direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - scipy is available in the project env
    ks_2samp = None


DEFAULT_RUN_ID = "20260617_090000_no_mkt4_labelhpo_final_fit"
DEFAULT_PARQUET_CACHE_COMPRESSION = "zstd"
DEFAULT_GENERATED_TRANSFORM_CACHE_MAX_ROWS = 500_000
DEFAULT_GENERATED_TRANSFORM_CACHE_MAX_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class AnalysisConfig:
    data_root: Path
    artifact_run_id: str
    output_dir: Path
    feature_frame: Path | None = None
    regime_feature_artifact_dir: Path | None = None
    feature_store_dir: Path | None = None
    feature_columns_json: Path | None = None
    include_all_feature_store_columns: bool = False
    include_config_liquidity_features: bool = False
    stream_feature_generation: bool = True
    transform_cache_enabled: bool = True
    generated_transform_cache_enabled: bool = True
    transform_cache_dir: Path | None = None
    refresh_transform_cache: bool = False
    parquet_cache_compression: str = DEFAULT_PARQUET_CACHE_COMPRESSION
    generated_transform_cache_ttl_days: float = 0.0
    generated_transform_cache_keep_last_n: int = 1
    generated_transform_cache_max_rows: int = DEFAULT_GENERATED_TRANSFORM_CACHE_MAX_ROWS
    generated_transform_cache_max_bytes: int = DEFAULT_GENERATED_TRANSFORM_CACHE_MAX_BYTES
    previous_meta_parent_report: Path | None = None
    previous_meta_parent_top_n: int = 50
    previous_meta_parent_slice: str = "top30"
    previous_meta_parent_transforms_enabled: bool = True
    generate_url_composites: bool = True
    top_rank_slice_only: bool = False
    analysis_start_day: str | None = None
    min_episode_end_day: str | None = None
    min_candidate_score_for_explanation: float = 0.005
    breakout_exploration_enabled: bool = True
    raw_exploration_max_features: int = 0
    raw_exploration_min_score: float = 0.015
    raw_exploration_min_pass_count: int = 1
    raw_candidate_min_score: float = 0.001
    composite_candidate_min_score: float = 0.005
    breakout_generate_svd_knn: bool = False
    advanced_transform_enabled: bool = True
    advanced_transform_windows: tuple[int, ...] = (24, 72)
    advanced_transform_extreme_z: float = 2.0
    advanced_covariance_enabled: bool = True
    max_precision_features: int = 30
    max_nonlinear_dependence_features: int = 12
    ebm_interaction_enabled: bool = True
    ebm_max_episodes: int = 6
    ebm_max_features: int = 10
    ebm_max_pairs: int = 15
    ebm_max_rows_per_episode: int = 1200
    ebm_max_control_rows: int = 6000
    ebm_max_rounds: int = 250
    ebm_min_rows: int = 400
    ebm_threshold_registry: Path | None = None
    ebm_threshold_state_features_enabled: bool = True
    ebm_threshold_min_selection_frequency: float = 0.50
    ebm_threshold_max_false_alarm_rate: float = 0.20
    ebm_threshold_min_episode_rows: int = 50
    ebm_threshold_require_positive_lift: bool = True
    redundancy_filter_enabled: bool = True
    redundancy_abs_spearman_threshold: float = 0.94
    redundancy_max_rows: int = 80_000
    timestamp_aggregate_row_threshold: int = 250_000
    ebm_min_recurrence_episodes: int = 2
    mixed_effects_enabled: bool = True
    mixed_effects_max_features: int = 40
    baseline_max_rows_per_episode: int = 50_000
    episode_max_rows_per_episode: int = 200_000
    window_days: int = 3
    secondary_window_days: int = 5
    embargo_days: int = 1
    min_window_rows: int = 0
    min_window_rows_per_day: float = 10.0
    surprise_z_threshold: float = -10.0
    hit_rate_delta_threshold: float = -0.22
    secondary_surprise_z_threshold: float = -10.0
    secondary_hit_rate_delta_threshold: float = -0.22
    bad_window_calibration_enabled: bool = False
    target_bad_day_share: float = 0.20
    bad_window_calibration_grid_size: int = 45
    rank_frac: float = 0.30
    min_feature_coverage: float = 0.70
    max_dominant_fraction: float = 0.985
    min_unique_values: int = 8
    max_features: int = 180
    max_rows_per_side: int = 200_000
    max_cov_features: int = 60
    random_seed: int = 1729
    include_diagnostic_features: bool = False


ID_COLUMNS = {
    "timestamp",
    "symbol",
    "dataset",
    "index",
    "__index_level_0__",
}

LABEL_TARGET_COLUMNS = {
    "y_bin",
    "y_bin_baseline",
    "y_move",
    "y_move_soft",
    "y_ret",
    "return",
    "mfe_ret",
    "mae_ret",
    "mfe",
    "mae",
    "barrier_pct",
    "bars_to_mfe",
    "is_timeout",
    "exit_code",
    "label_code",
    "label_weight_hpo_winner",
    "label_weight_hpo_selected",
    "label_weight_hpo_soft_label",
    "label_weight_hpo_sample_weight",
    "move_threshold",
}

PREDICTION_OR_MODEL_COLUMNS = {
    "oof_prob",
    "oof_pred",
    "oof_p_move",
    "oof_base_clf",
    "oof_meta_clf",
    "base_clf_centered",
    "clf_center",
    "clf_entropy",
    "diag_mean_pred",
    "diag_std_pred",
    "clf_reg_direction_score",
    "clf_prob_x_reg_snr",
    "reg_snr_x_clf_vote_margin",
    "clf_entropy_x_reg_sign_entropy",
    "clf_std_x_reg_std",
}

LEARNED_OR_ARTIFACT_PREFIXES = (
    "oof_lgbm_",
    "oof_gmm_",
    "oof_dae_",
    "oof_score_",
    "oof_rank_",
    "oof_raw_score_",
    "oof_prob_",
    "oof_tree_",
    "oof_leaf_",
    "oof_base_error_",
    "oof_archetype_",
    "reg_pred_",
    "reg_q",
    "reg_sign_",
    "reg_pos_vote_",
    "reg_leaf_",
    "reg_rare_leaf_",
    "reg_uncertainty",
)

DIAGNOSTIC_PREFIXES = (
    "oof_feature_drift_",
    "oof_row_drift_",
    "oof_inference_drift_",
    "oof_uncertainty_",
    "oof_rare_leaf_",
    "oof_contribution_drift_",
    "oof_mahalanobis_",
    "oof_frobenius_",
)

FEATURE_NAME_HINTS = (
    "url_",
    "url_asset__",
    "url_market__",
    "url_xs_z__",
    "url_sigreg__",
    "q_",
    "eig_",
    "autocorr_",
    "roll_slope_",
    "roll_accel_",
    "extreme_exposure_",
    "pair_",
    "cov_w",
    "corr_w",
    "xs_cov_",
    "svd_",
    "knn_",
    "trend",
    "vol",
    "rv",
    "rvol",
    "amihud",
    "efficiency",
    "compression",
    "liquidity",
    "volume",
    "funding",
    "oi",
    "open_interest",
    "atr",
    "range",
    "momentum",
    "autocorr",
    "variance",
    "basis",
    "spread",
    "price",
    "return_",
    "loc_",
    "meta_en_",
)

LIQUIDITY_EXECUTION_FEATURE_TOKENS = (
    "amihud",
    "liquidity",
    "spread",
    "depth",
    "turnover",
    "stale",
    "available",
    "snapshot_age",
    "update_gap",
    "quote_volume",
    "volume_z",
    "volume_percentile",
    "relative_volume",
    "dollar_vol",
    "top_liquidity",
    "to_qv",
    "notional_to_depth",
    "trade_size_to_l1_depth",
    "volume_depth_risk",
)

BREAKOUT_STRUCTURE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # Existing 48/120-bar features are the codebase's 50/100-bar equivalents.
    "distance_to_high_low": ("dist_from_high_48h", "dist_from_low_48h"),
    "donchian_channel_position": ("loc_range_pos_24", "loc_range_pos_48"),
    "bb_keltner_squeeze": ("bollinger_band_width", "compression_score"),
    "atr_breakout_distance": ("donch_dist_48", "donch_dist_120"),
    "range_compression_expansion": ("range_expansion_ratio", "range_decay"),
    "higher_high_lower_low": ("higher_highs_count_48h", "lower_lows_count_48h"),
    "close_location_recent_range": ("loc_swing_range_pos_24", "loc_swing_range_pos_48"),
}
BREAKOUT_STRUCTURE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        feature
        for features in BREAKOUT_STRUCTURE_FEATURE_GROUPS.values()
        for feature in features
    )
)

REGIME_PANEL_FILES = (
    "regime_context_features.parquet",
    "signal_regime_interaction_features.parquet",
)

REGIME_PICKLE_PANEL_FILES = (
    "model_regime_features.pkl",
    "regime_transition_features.pkl",
)

PAIR_FEATURE_RE = re.compile(r"^(?:cov|corr)_w(?P<window>\d+)__(?P<left>.+)__(?P<right>.+)$")
QUANTILE_FEATURE_RE = re.compile(r"^q_(?:iqr|tail_width|upper_tail|lower_tail|tail_asym|percentile_rank)__(?P<source>.+)$")
AUTOCORR_FEATURE_RE = re.compile(r"^autocorr_lag(?P<lag>\d+)_w(?P<window>\d+)__(?P<source>.+)$")
EIG_FEATURE_RE = re.compile(r"^eig_(?:largest_share|top\d+_share|effective_rank|participation_ratio|turnover)__(?P<group>.+)$")


def _log(message: str) -> None:
    print(message, flush=True)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_hash_payload(payload: object, *, digest_size: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=digest_size).hexdigest()


def _row_universe_hash(frame: pd.DataFrame) -> str:
    if "timestamp" not in frame.columns or "symbol" not in frame.columns:
        return "missing_keys"
    keys = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            .astype("int64", copy=False)
            .to_numpy(dtype=np.int64, copy=False),
            "symbol": frame["symbol"].astype(str).to_numpy(copy=False),
        }
    )
    row_hash = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([len(frame)], dtype=np.int64).tobytes())
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _clip01(value: float) -> float:
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


def _strategy_from_path(path: Path, layer: str) -> str:
    stem = path.stem
    if layer == "base":
        stem = re.sub(r"^oof_", "", stem)
        stem = re.sub(r"_H\d+$", "", stem)
    else:
        stem = re.sub(r"^meta_oof_", "", stem)
        stem = re.sub(r"_tbm_clf$", "", stem)
    return stem


def _artifact_root(config: AnalysisConfig) -> Path:
    return config.data_root / "artifacts" / config.artifact_run_id


def _discover_oof_files(config: AnalysisConfig) -> list[tuple[str, str, Path]]:
    root = _artifact_root(config)
    files: list[tuple[str, str, Path]] = []
    for path in sorted((root / "oof").glob("oof_*_H*.parquet")):
        files.append(("base", _strategy_from_path(path, "base"), path))
    for path in sorted((root / "meta_oof").glob("meta_oof_*_tbm_clf.parquet")):
        files.append(("meta", _strategy_from_path(path, "meta"), path))
    return files


def _prediction_columns(frame: pd.DataFrame, layer: str) -> tuple[str, str, str]:
    if layer == "base":
        pred_candidates = ("oof_prob", "oof_lgbm_prob")
        pnl_candidates = ("y_ret", "return")
    else:
        pred_candidates = ("oof_pred", "oof_meta_clf", "oof_p_move", "oof_base_clf")
        pnl_candidates = ("return", "y_ret")
    pred_col = next((col for col in pred_candidates if col in frame.columns), "")
    label_col = "y_bin" if "y_bin" in frame.columns else ""
    pnl_col = next((col for col in pnl_candidates if col in frame.columns), "")
    if not pred_col or not label_col:
        raise ValueError(
            f"Could not identify prediction/label columns for layer={layer}: "
            f"pred={pred_col!r}, label={label_col!r}"
        )
    return pred_col, label_col, pnl_col


def _load_optional_feature_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"timestamp", "symbol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"--feature-frame must include {sorted(required)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    return frame


def _parse_utc_day(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(str(value).strip())
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _filter_frame_by_analysis_period(frame: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    start = _parse_utc_day(config.analysis_start_day)
    if start is None or frame.empty or "timestamp" not in frame.columns:
        return frame
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    before = len(out)
    out = out.loc[out["timestamp"].ge(start)].copy()
    if before != len(out):
        _log(
            f"[filter] analysis_start_day={start.date()} "
            f"rows={len(out)}/{before}"
        )
    return out


def _read_pickle_frame(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    return obj


def _load_regime_feature_artifact_dir(path: Path | None) -> pd.DataFrame | None:
    """Load unsupervised-regime feature panels with persisted row keys.

    The unsupervised_regime_learning POC stores context/interactions without
    timestamp/symbol columns because their contract is row alignment. The
    advanced artifact persists row_keys.pkl; use that to make a keyed frame.
    """

    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    frames_dir = path / "advanced_regime_learning" / "advanced_regime_learning_frames"
    row_keys_path = frames_dir / "row_keys.pkl"
    row_keys: pd.DataFrame | None = None
    if row_keys_path.exists():
        row_keys = _read_pickle_frame(row_keys_path)
        required = {"timestamp", "symbol"}
        if not required.issubset(row_keys.columns):
            raise ValueError(f"{row_keys_path} must include {sorted(required)}")
        row_keys = row_keys[["timestamp", "symbol"]].copy()
        row_keys["timestamp"] = pd.to_datetime(row_keys["timestamp"], utc=True, errors="coerce")
        row_keys["symbol"] = row_keys["symbol"].astype(str)

    parts: list[pd.DataFrame] = []
    for filename in REGIME_PANEL_FILES:
        panel_path = path / filename
        if not panel_path.exists():
            continue
        panel = pd.read_parquet(panel_path)
        if {"timestamp", "symbol"}.issubset(panel.columns):
            keyed = panel.copy()
            keyed["timestamp"] = pd.to_datetime(keyed["timestamp"], utc=True, errors="coerce")
            keyed["symbol"] = keyed["symbol"].astype(str)
            parts.append(keyed)
            continue
        if row_keys is None:
            raise ValueError(
                f"{panel_path} has no timestamp/symbol columns and no row_keys.pkl "
                "was found in the advanced artifact."
            )
        if len(panel) != len(row_keys):
            raise ValueError(
                f"{panel_path} row count {len(panel)} does not match row_keys {len(row_keys)}"
            )
        keyed = pd.concat([row_keys.reset_index(drop=True), panel.reset_index(drop=True)], axis=1)
        parts.append(keyed)

    for filename in REGIME_PICKLE_PANEL_FILES:
        panel_path = frames_dir / filename
        if not panel_path.exists():
            continue
        if row_keys is None:
            raise ValueError(f"{panel_path} requires row_keys.pkl for safe alignment")
        panel = _read_pickle_frame(panel_path)
        if len(panel) != len(row_keys):
            raise ValueError(
                f"{panel_path} row count {len(panel)} does not match row_keys {len(row_keys)}"
            )
        keyed = pd.concat([row_keys.reset_index(drop=True), panel.reset_index(drop=True)], axis=1)
        parts.append(keyed)

    if not parts:
        raise FileNotFoundError(
            f"No supported regime feature panels found in {path}. Expected one of "
            f"{list(REGIME_PANEL_FILES)} or {list(REGIME_PICKLE_PANEL_FILES)}."
        )

    out: pd.DataFrame | None = None
    for part in parts:
        feature_cols = [col for col in part.columns if col not in {"timestamp", "symbol"}]
        part = part[["timestamp", "symbol", *feature_cols]].copy()
        for col in feature_cols:
            part[col] = pd.to_numeric(part[col], errors="coerce").astype(np.float32, copy=False)
        if out is None:
            out = part
        else:
            existing = set(out.columns)
            add_cols = [col for col in feature_cols if col not in existing]
            if add_cols:
                out = out.merge(part[["timestamp", "symbol", *add_cols]], on=["timestamp", "symbol"], how="outer", copy=False)
    if out is None:
        return None
    out = out.loc[out["timestamp"].notna()].copy()
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").drop_duplicates(
        ["timestamp", "symbol"],
        keep="last",
    )
    return out.reset_index(drop=True)


def _feature_path_for_symbol(feature_dir: Path, symbol: str) -> Path:
    return feature_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return [str(name) for name in pq.ParquetFile(path).schema.names]
    except Exception:
        return [str(col) for col in pd.read_parquet(path).columns]


def _load_feature_columns_json(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        values = obj.get("features") or obj.get("feature_columns") or obj.get("columns") or []
    else:
        values = obj
    return list(dict.fromkeys(str(col) for col in values if str(col)))


def _available_safe_feature_store_columns(feature_dir: Path | None) -> list[str]:
    if feature_dir is None or not feature_dir.exists():
        return []
    raw_panel_columns = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "mark_price",
        "index_price",
        "canonical_index",
    }
    columns: set[str] = set()
    for path in sorted(feature_dir.glob("symbol=*.parquet")):
        columns.update(_parquet_columns(path))
    out: list[str] = []
    for col in sorted(columns):
        name = str(col)
        lower = name.lower()
        if lower in raw_panel_columns:
            continue
        if lower.startswith("__"):
            continue
        if _is_feature_artifact(name, include_diagnostic_features=False):
            continue
        out.append(name)
    return out


def _is_url_generated_feature(name: str) -> bool:
    value = str(name)
    return bool(
        QUANTILE_FEATURE_RE.match(value)
        or AUTOCORR_FEATURE_RE.match(value)
        or EIG_FEATURE_RE.match(value)
        or PAIR_FEATURE_RE.match(value)
        or value.startswith("svd8_")
        or value.startswith("svd16_")
        or value.startswith("roll_slope_")
        or value.startswith("roll_accel_")
        or value.startswith("extreme_exposure_")
        or value.startswith("xs_cov_")
    )


def _is_composite_operator_feature(name: str) -> bool:
    value = str(name)
    lower = value.lower()
    return bool(
        _is_url_generated_feature(value)
        or lower.startswith(("url_", "url_asset__", "url_market__", "url_xs_z__", "url_sigreg__"))
    )


def _is_previous_meta_parent_raw_feature(name: str) -> bool:
    """True only for raw, safe columns that can seed regime-state transforms.

    Previous reports often contain generated operator columns. Feeding those
    back into the operator generator would create hard-to-interpret features
    such as covariance-of-covariance or KNN-on-KNN. Keep this filter stricter
    than the ordinary feature candidate filter.
    """

    value = str(name).strip()
    if not value:
        return False
    lower = value.lower()
    if _is_feature_artifact(value, include_diagnostic_features=False):
        return False
    if _is_composite_operator_feature(value):
        return False
    if "__" in value:
        return False
    blocked_prefixes = (
        "xs_",
        "q_",
        "autocorr_",
        "roll_slope_",
        "roll_accel_",
        "extreme_exposure_",
        "cov_w",
        "corr_w",
        "xs_cov_",
        "eig_",
        "svd",
        "knn",
        "ebm_state_",
        "url_",
        "meta_",
        "regime_",
        "pred_",
        "target_",
        "future_",
    )
    if lower.startswith(blocked_prefixes):
        return False
    if lower in ID_COLUMNS or lower in LABEL_TARGET_COLUMNS or lower in PREDICTION_OR_MODEL_COLUMNS:
        return False
    return True


def _resolve_previous_meta_parent_report(path: Path | None) -> Path | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return path
    preferred = (
        "feature_breakout_explanatory_strength_by_head.csv",
        "selected_features_by_head.csv",
        "feature_breakout_explanatory_strength_by_head.parquet",
        "selected_features_by_head.parquet",
    )
    for name in preferred:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No previous meta parent feature report found in {path}. Expected one of {list(preferred)}."
    )


def _read_report_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _previous_parent_sort_columns(frame: pd.DataFrame) -> tuple[list[str], list[bool]]:
    if "breakout_explanatory_strength" in frame.columns:
        return (
            [
                "breakout_explanatory_strength",
                "explained_breakout_weight_share",
                "explained_breakout_count",
            ],
            [False, False, False],
        )
    if "feature_rank" in frame.columns:
        return (["feature_rank"], [True])
    return (["__source_order"], [True])


def _load_previous_meta_parent_features(
    path: Path | None,
    *,
    config: AnalysisConfig,
) -> tuple[dict[tuple[str, str, str], list[str]], pd.DataFrame]:
    """Load top-N raw parent features from a previous run, meta heads only."""

    resolved = _resolve_previous_meta_parent_report(path)
    if resolved is None or not bool(config.previous_meta_parent_transforms_enabled):
        return {}, pd.DataFrame()
    frame = _read_report_frame(resolved)
    required = {"layer", "strategy", "slice", "feature"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{resolved} must include {sorted(required)}")
    if frame.empty:
        return {}, pd.DataFrame()

    work = frame.copy()
    work["__source_order"] = np.arange(len(work), dtype=np.int64)
    work["layer"] = work["layer"].astype(str)
    work["strategy"] = work["strategy"].astype(str)
    work["slice"] = work["slice"].astype(str)
    work["feature"] = work["feature"].astype(str)
    work = work.loc[work["layer"].str.lower().eq("meta")].copy()
    wanted_slice = str(config.previous_meta_parent_slice).strip()
    if wanted_slice:
        work = work.loc[work["slice"].eq(wanted_slice)].copy()
    if work.empty:
        _log(
            f"[previous-meta-parents] no meta rows found in {resolved} "
            f"for slice={wanted_slice!r}"
        )
        return {}, pd.DataFrame()

    work["raw_parent_candidate"] = work["feature"].map(_is_previous_meta_parent_raw_feature)
    rejected_count = int((~work["raw_parent_candidate"]).sum())
    work = work.loc[work["raw_parent_candidate"]].copy()
    if work.empty:
        _log(
            f"[previous-meta-parents] all candidate rows rejected by raw-parent filter "
            f"path={resolved}"
        )
        return {}, pd.DataFrame()

    sort_cols, ascending = _previous_parent_sort_columns(work)
    for col in sort_cols:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "feature_rank" in work.columns:
        work["previous_parent_score"] = 1.0 / np.maximum(
            pd.to_numeric(work["feature_rank"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64),
            1.0,
        )
    elif "breakout_explanatory_strength" in work.columns:
        work["previous_parent_score"] = pd.to_numeric(
            work["breakout_explanatory_strength"],
            errors="coerce",
        ).fillna(0.0)
    else:
        work["previous_parent_score"] = 1.0

    rows: list[pd.DataFrame] = []
    parent_map: dict[tuple[str, str, str], list[str]] = {}
    top_n = max(int(config.previous_meta_parent_top_n), 0)
    for key, group in work.groupby(["layer", "strategy", "slice"], sort=False):
        group = group.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        group = group.drop_duplicates("feature", keep="first")
        if top_n > 0:
            group = group.head(top_n)
        if group.empty:
            continue
        group = group.copy()
        group["previous_parent_rank"] = np.arange(1, len(group) + 1, dtype=np.int32)
        group["previous_parent_source"] = str(resolved)
        group["feature_family"] = group["feature"].map(_feature_family)
        portability = group["feature"].map(_feature_portability)
        group["portability_kind"] = [item[0] for item in portability]
        group["portability_reason"] = [item[1] for item in portability]
        parent_map[(str(key[0]), str(key[1]), str(key[2]))] = group["feature"].astype(str).tolist()
        rows.append(group)

    report = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _log(
        f"[previous-meta-parents] loaded heads={len(parent_map)} "
        f"parents={sum(len(v) for v in parent_map.values())} "
        f"rejected_non_raw={rejected_count} source={resolved}"
    )
    keep_cols = [
        col
        for col in [
            "layer",
            "strategy",
            "slice",
            "previous_parent_rank",
            "feature",
            "feature_family",
            "previous_parent_score",
            "portability_kind",
            "portability_reason",
            "previous_parent_source",
        ]
        if col in report.columns
    ]
    return parent_map, report[keep_cols].copy() if keep_cols else report


def _previous_meta_parent_features_for_head(
    parent_map: dict[tuple[str, str, str], list[str]],
    *,
    layer: str,
    strategy: str,
    config: AnalysisConfig,
) -> list[str]:
    if str(layer).lower() != "meta" or not parent_map:
        return []
    wanted_slice = str(config.previous_meta_parent_slice).strip()
    if wanted_slice:
        direct = parent_map.get(("meta", str(strategy), wanted_slice), [])
        if direct:
            return list(direct)
    out: list[str] = []
    for (map_layer, map_strategy, _slice), features in parent_map.items():
        if map_layer == "meta" and map_strategy == str(strategy):
            out.extend(features)
    return list(dict.fromkeys(out))


def _infer_url_primitive_sources(feature_columns: Sequence[str]) -> list[str]:
    sources: list[str] = []
    for feature in feature_columns:
        name = str(feature)
        if _is_url_generated_feature(name):
            continue
        if name.startswith(("url_", "oof_", "meta_", "regime_")):
            continue
        sources.append(name)
    for feature in feature_columns:
        name = str(feature)
        match = QUANTILE_FEATURE_RE.match(name)
        if match:
            sources.append(match.group("source"))
            continue
        match = AUTOCORR_FEATURE_RE.match(name)
        if match:
            sources.append(match.group("source"))
            continue
        match = PAIR_FEATURE_RE.match(name)
        if match:
            sources.extend([match.group("left"), match.group("right")])
    return list(dict.fromkeys(source for source in sources if source))


def _selected_pair_scores(feature_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for feature in feature_columns:
        match = PAIR_FEATURE_RE.match(str(feature))
        if not match:
            continue
        left = str(match.group("left"))
        right = str(match.group("right"))
        window = int(match.group("window"))
        key = (left, right, window)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "feature_i": left,
                "feature_j": right,
                "pair_score": 1.0,
                "rho_variation": 1.0,
                "rho_persistence": 1.0,
                "reliability": 1.0,
                "window": window,
            }
        )
    return pd.DataFrame(rows)


def _url_feature_groups_for_eigen(feature_columns: Sequence[str]) -> dict[str, list[str]]:
    wanted_groups = {
        match.group("group")
        for feature in feature_columns
        if (match := EIG_FEATURE_RE.match(str(feature)))
    }
    if not wanted_groups:
        return {}
    try:
        from extreme_price_movements.config import CFG
    except Exception:
        return {}
    raw_groups = CFG.get("UNSUPERVISED_REGIME_LEARNING", {}).get("primitive_feature_groups", {})
    groups: dict[str, list[str]] = {}
    if isinstance(raw_groups, dict):
        for key, values in raw_groups.items():
            name = str(key)
            if name.endswith("_features"):
                name = name[: -len("_features")]
            if name in wanted_groups and isinstance(values, (list, tuple, set)):
                groups[name] = [str(value) for value in values if str(value)]
    return groups


def _generate_selected_url_composites(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    primitive_sources: Sequence[str],
) -> pd.DataFrame:
    requested = set(str(col) for col in feature_columns if str(col))
    generated_parts: list[pd.DataFrame] = []
    if panel.empty or not primitive_sources:
        return pd.DataFrame(index=panel.index)
    try:
        from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
            prepare_frame_context,
        )
        from extreme_price_movements.unsupervised_regime_learning.operators import (
            generate_autocorr_operator_features,
            generate_pair_operator_features,
            generate_quantile_operator_features,
        )
    except Exception as exc:
        _log(f"[features] URL composite generation unavailable: {type(exc).__name__}: {exc}")
        return pd.DataFrame(index=panel.index)

    t0 = time.perf_counter()
    context = prepare_frame_context(panel, symbol_col="symbol", timestamp_col="timestamp")
    q_sources = [
        match.group("source")
        for feature in requested
        if (match := QUANTILE_FEATURE_RE.match(str(feature)))
    ]
    q_sources = [source for source in dict.fromkeys(q_sources) if source in panel.columns]
    _log(
        f"[features] generating selected URL composites: requested={len(requested)} "
        f"primitive_sources={len(primitive_sources)} q_sources={len(q_sources)}"
    )
    if q_sources:
        part_t0 = time.perf_counter()
        q = generate_quantile_operator_features(
            panel,
            q_sources,
            window=24,
            min_periods=8,
            symbol_col="symbol",
            timestamp_col="timestamp",
            context=context,
        )
        q_cols = [col for col in q.columns if col in requested]
        if q_cols:
            generated_parts.append(q[q_cols].astype(np.float32, copy=False))
        _log(
            f"[features] URL quantile operators kept={len(q_cols)} "
            f"elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    autocorr_specs: dict[tuple[int, int], list[str]] = {}
    for feature in requested:
        match = AUTOCORR_FEATURE_RE.match(str(feature))
        if match:
            key = (int(match.group("lag")), int(match.group("window")))
            autocorr_specs.setdefault(key, []).append(match.group("source"))
    for (lag, window), sources in autocorr_specs.items():
        sources = [source for source in dict.fromkeys(sources) if source in panel.columns]
        if not sources:
            continue
        part_t0 = time.perf_counter()
        ac = generate_autocorr_operator_features(
            panel,
            sources,
            window=window,
            lag=lag,
            min_periods=8,
            symbol_col="symbol",
            timestamp_col="timestamp",
            context=context,
        )
        ac_cols = [col for col in ac.columns if col in requested]
        if ac_cols:
            generated_parts.append(ac[ac_cols].astype(np.float32, copy=False))
        _log(
            f"[features] URL autocorr operators lag={lag} window={window} "
            f"sources={len(sources)} kept={len(ac_cols)} elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    pair_scores = _selected_pair_scores(feature_columns)
    if not pair_scores.empty:
        pair_scores = pair_scores[
            pair_scores["feature_i"].astype(str).isin(panel.columns)
            & pair_scores["feature_j"].astype(str).isin(panel.columns)
        ].reset_index(drop=True)
    if not pair_scores.empty:
        for window, group in pair_scores.groupby("window", sort=False):
            part_t0 = time.perf_counter()
            pair = generate_pair_operator_features(
                panel,
                group,
                window=int(window),
                min_periods=8,
                symbol_col="symbol",
                timestamp_col="timestamp",
                context=context,
            )
            pair_cols = [col for col in pair.columns if col in requested]
            if pair_cols:
                generated_parts.append(pair[pair_cols].astype(np.float32, copy=False))
            _log(
                f"[features] URL pair operators window={int(window)} pairs={len(group)} "
                f"kept={len(pair_cols)} elapsed={time.perf_counter() - part_t0:.1f}s"
            )

    eigen_groups = _url_feature_groups_for_eigen(feature_columns)
    eigen_groups = {
        name: [feature for feature in features if feature in panel.columns]
        for name, features in eigen_groups.items()
    }
    eigen_groups = {name: features for name, features in eigen_groups.items() if len(features) >= 2}
    if eigen_groups:
        part_t0 = time.perf_counter()
        eig = generate_eigenvalue_summary_features(
            panel,
            eigen_groups,
            window=24,
            min_periods=8,
            top_k=3,
            symbol_col="symbol",
            timestamp_col="timestamp",
            context=context,
        )
        eig_cols = [col for col in eig.columns if col in requested]
        if eig_cols:
            generated_parts.append(eig[eig_cols].astype(np.float32, copy=False))
        _log(
            f"[features] URL eigen operators groups={len(eigen_groups)} "
            f"kept={len(eig_cols)} elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    svd_requested = [
        feature
        for feature in requested
        if str(feature).startswith("svd8_") or str(feature).startswith("svd16_")
    ]
    svd_sources = [source for source in primitive_sources if source in panel.columns]
    if svd_requested and len(svd_sources) >= 2:
        part_t0 = time.perf_counter()
        _log(
            f"[features] URL SVD/KNN operators requested={len(svd_requested)} "
            f"sources={len(svd_sources)}"
        )
        svd, _state = fit_transform_svd_knn_features_walk_forward(
            panel,
            svd_sources,
            svd_components=[8, 16],
            knn_svd_components=16,
            knn_neighbors=10,
            timestamp_col="timestamp",
            symbol_col="symbol",
            block_hours=24 * 14,
            min_prior_rows=64,
            max_reference_rows=3000,
            knn_max_reference_rows=1500,
            sample_time_bins=8,
        )
        svd_cols = [col for col in svd.columns if col in requested]
        if svd_cols:
            generated_parts.append(svd[svd_cols].astype(np.float32, copy=False))
        _log(
            f"[features] URL SVD/KNN operators kept={len(svd_cols)} "
            f"elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    if not generated_parts:
        return pd.DataFrame(index=panel.index)
    out = pd.concat(generated_parts, axis=1)
    _log(
        f"[features] selected URL composite generation complete cols={out.shape[1]} "
        f"elapsed={time.perf_counter() - t0:.1f}s"
    )
    return out.loc[:, ~out.columns.duplicated()].astype(np.float32, copy=False)


def _breakout_pair_scores_from_raw_screen(
    raw_screen: pd.DataFrame,
    *,
    max_pairs: int,
) -> pd.DataFrame:
    if raw_screen.empty:
        return pd.DataFrame()
    selected = raw_screen.loc[raw_screen["selected_for_operator_generation"].astype(bool)].copy()
    if selected.empty:
        selected = raw_screen.copy()
    selected = selected.sort_values("raw_breakout_link_score", ascending=False, kind="mergesort")
    features = selected["feature"].astype(str).tolist()
    scores = {
        str(row.feature): float(max(_safe_float(row.raw_breakout_link_score, 0.0), 1e-6))
        for row in selected.itertuples(index=False)
    }
    families = {feature: _feature_family(feature) for feature in features}
    rows: list[dict[str, object]] = []
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            diversity = 1.15 if families.get(left) != families.get(right) else 0.85
            pair_score = math.sqrt(scores.get(left, 1e-6) * scores.get(right, 1e-6)) * diversity
            rows.append(
                {
                    "feature_i": left,
                    "feature_j": right,
                    "pair_score": pair_score,
                    "rho_variation": pair_score,
                    "rho_persistence": 1.0,
                    "reliability": 1.0,
                    "mechanism_i": families.get(left, "unknown"),
                    "mechanism_j": families.get(right, "unknown"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pair_score", ascending=False, kind="mergesort").head(int(max_pairs))


def _safe_operator_group_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "group"


def _local_eigen_summary(matrix: np.ndarray, top_k: int = 3) -> tuple[float, float, float, float, np.ndarray]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    finite_cols = np.isfinite(arr).any(axis=0)
    if int(finite_cols.sum()) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    arr = arr[:, finite_cols]
    finite_rows = np.isfinite(arr).any(axis=1)
    if int(finite_rows.sum()) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    arr = arr[finite_rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(arr)
    if bool(bad.any()):
        arr = arr.copy()
        rows, cols = np.where(bad)
        arr[rows, cols] = med[cols]
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    cov = np.cov(arr, rowvar=False)
    if cov.ndim != 2:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    vals, vecs = np.linalg.eigh(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    total = float(vals.sum())
    if total <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    shares = vals / total
    positive = shares[shares > 1e-12]
    largest_share = float(shares[0])
    top_share = float(np.sum(shares[: max(1, int(top_k))]))
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if positive.size else np.nan
    participation = float((np.sum(vals) ** 2) / max(np.sum(vals * vals), 1e-12))
    return largest_share, top_share, effective_rank, participation, vecs[:, 0]


def _generate_fast_timestamp_eigen_features(
    panel: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    window: int,
    min_periods: int,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Fast global rolling eigen summaries, broadcast from timestamp to rows."""

    groups = {
        _safe_operator_group_name(group): [feature for feature in features if feature in panel.columns]
        for group, features in feature_groups.items()
    }
    groups = {group: features for group, features in groups.items() if len(features) >= 2}
    if not groups or timestamp_col not in panel.columns:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    all_features = list(dict.fromkeys(feature for features in groups.values() for feature in features))
    work = panel[[timestamp_col, *all_features]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = work.dropna(subset=[timestamp_col])
    if work.empty:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    for col in all_features:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(np.float32, copy=False)
    by_time = work.groupby(timestamp_col, sort=True)[all_features].mean()
    if by_time.empty:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    win = max(3, int(window))
    minp = max(3, int(min_periods))
    cols: dict[str, np.ndarray] = {}
    for group, features in groups.items():
        values = by_time[features].to_numpy(dtype=np.float64, copy=True)
        names = [
            f"eig_largest_share__{group}",
            f"eig_top3_share__{group}",
            f"eig_effective_rank__{group}",
            f"eig_participation_ratio__{group}",
            f"eig_turnover__{group}",
        ]
        for name in names:
            cols[name] = np.full(len(by_time), np.nan, dtype=np.float32)
        prev_vec: np.ndarray | None = None
        for pos in range(len(by_time)):
            start = max(0, pos - win + 1)
            sample = values[start : pos + 1]
            finite_rows = np.isfinite(sample).any(axis=1)
            if int(finite_rows.sum()) < minp:
                continue
            largest, top, effective_rank, participation, vec = _local_eigen_summary(sample[finite_rows], top_k=3)
            turnover = np.nan
            if prev_vec is not None and vec.size == prev_vec.size and vec.size:
                turnover = float(1.0 - abs(float(np.dot(prev_vec, vec))))
            if vec.size:
                prev_vec = vec
            cols[names[0]][pos] = largest
            cols[names[1]][pos] = top
            cols[names[2]][pos] = effective_rank
            cols[names[3]][pos] = participation
            cols[names[4]][pos] = turnover
    time_features = pd.DataFrame(cols, index=by_time.index, dtype=np.float32)
    left = pd.DataFrame(
        {
            timestamp_col: pd.to_datetime(panel[timestamp_col], utc=True, errors="coerce"),
            "__row_pos": np.arange(len(panel), dtype=np.int64),
        }
    )
    merged = left.merge(time_features.reset_index(), on=timestamp_col, how="left", copy=False)
    merged = merged.sort_values("__row_pos", kind="mergesort")
    out = merged.drop(columns=[timestamp_col, "__row_pos"], errors="ignore")
    out.index = panel.index
    return out.astype(np.float32, copy=False)


def _generate_cross_sectional_regime_features(
    panel: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    min_assets: int = 8,
) -> pd.DataFrame:
    """Cross-sectional regime summaries for portable asset-level primitives."""

    portable = [feature for feature in _asset_portable_features(raw_features) if feature in panel.columns]
    if not portable or timestamp_col not in panel.columns or symbol_col not in panel.columns:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    work = panel[[timestamp_col, symbol_col, *portable]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = work.dropna(subset=[timestamp_col])
    if work.empty:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    for col in portable:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(np.float32, copy=False)
    grouped_items = list(work.groupby(timestamp_col, sort=True))
    if not grouped_items:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    time_index = pd.Index([ts for ts, _group in grouped_items], name=timestamp_col)
    n_time = len(grouped_items)
    n_features = len(portable)
    mean_arr = np.full((n_time, n_features), np.nan, dtype=np.float32)
    median_arr = np.full((n_time, n_features), np.nan, dtype=np.float32)
    std_arr = np.full((n_time, n_features), np.nan, dtype=np.float32)
    iqr_arr = np.full((n_time, n_features), np.nan, dtype=np.float32)

    family_groups: dict[str, list[str]] = {}
    for feature in portable:
        family_groups.setdefault(_feature_family(feature), []).append(feature)
    family_groups = {family: cols for family, cols in family_groups.items() if len(cols) >= 2}
    if len(portable) >= 2:
        family_groups["asset_portable_all"] = portable
    eig_cols: dict[str, np.ndarray] = {}
    group_output_names: dict[str, list[str]] = {}
    for group_name in family_groups:
        safe_group = _safe_operator_group_name(f"xs_{group_name}")
        names = [
            f"xs_cov_pc1_concentration__{safe_group}",
            f"xs_cov_top3_share__{safe_group}",
            f"xs_cov_effective_rank__{safe_group}",
            f"xs_cov_participation_ratio__{safe_group}",
            f"xs_cov_mean_abs_corr__{safe_group}",
        ]
        for name in names:
            eig_cols[name] = np.full(n_time, np.nan, dtype=np.float32)
        group_output_names[group_name] = names

    for pos, (_ts, group) in enumerate(grouped_items):
        values_all = group[portable].to_numpy(dtype=np.float64, copy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_arr[pos] = np.nanmean(values_all, axis=0).astype(np.float32)
            median_arr[pos] = np.nanmedian(values_all, axis=0).astype(np.float32)
            std_arr[pos] = np.nanstd(values_all, axis=0).astype(np.float32)
            q25 = np.nanpercentile(values_all, 25, axis=0)
            q75 = np.nanpercentile(values_all, 75, axis=0)
        iqr_arr[pos] = (q75 - q25).astype(np.float32)
        if len(group) < int(min_assets):
            continue
        for group_name, features in family_groups.items():
            names = group_output_names[group_name]
            values = group[features].to_numpy(dtype=np.float64, copy=True)
            finite_rows = np.isfinite(values).any(axis=1)
            if int(finite_rows.sum()) < int(min_assets):
                continue
            largest, top, effective_rank, participation, _vec = _local_eigen_summary(
                values[finite_rows],
                top_k=3,
            )
            if not np.isfinite(largest):
                continue
            corr = _corr_matrix(values[finite_rows])
            tri = corr[np.triu_indices_from(corr, k=1)] if corr.shape[0] > 1 else np.asarray([], dtype=np.float64)
            eig_cols[names[0]][pos] = largest
            eig_cols[names[1]][pos] = top
            eig_cols[names[2]][pos] = effective_rank
            eig_cols[names[3]][pos] = participation
            eig_cols[names[4]][pos] = float(np.nanmean(np.abs(tri))) if tri.size else np.nan

    dispersion_arr = std_arr / (np.abs(mean_arr) + 1e-6)
    time_parts = [
        pd.DataFrame(mean_arr, index=time_index, columns=[f"xs_mean__{feature}" for feature in portable]),
        pd.DataFrame(median_arr, index=time_index, columns=[f"xs_median__{feature}" for feature in portable]),
        pd.DataFrame(std_arr, index=time_index, columns=[f"xs_std__{feature}" for feature in portable]),
        pd.DataFrame(iqr_arr, index=time_index, columns=[f"xs_iqr__{feature}" for feature in portable]),
        pd.DataFrame(dispersion_arr, index=time_index, columns=[f"xs_dispersion__{feature}" for feature in portable]),
    ]
    if eig_cols:
        time_parts.append(pd.DataFrame(eig_cols, index=time_index, dtype=np.float32))
    time_features = pd.concat(time_parts, axis=1)
    left = pd.DataFrame(
        {
            timestamp_col: pd.to_datetime(panel[timestamp_col], utc=True, errors="coerce"),
            "__row_pos": np.arange(len(panel), dtype=np.int64),
        }
    )
    merged = left.merge(time_features.reset_index(), on=timestamp_col, how="left", copy=False)
    merged = merged.sort_values("__row_pos", kind="mergesort")
    out = merged.drop(columns=[timestamp_col, "__row_pos"], errors="ignore")
    out.index = panel.index
    _log(
        f"[features] breakout cross-sectional regime features portable_sources={len(portable)} "
        f"cols={out.shape[1]}"
    )
    return out.astype(np.float32, copy=False)


def _rolling_slope_exposure_group(
    values: np.ndarray,
    extreme: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling equal-spaced slope, slope acceleration, and extreme exposure.

    Uses cumulative sums over a single sorted symbol group. The x-axis is the
    row position within the symbol, which is appropriate for hourly aligned
    OOF/feature rows and avoids expensive pandas rolling regressions.
    """

    n = int(values.size)
    slope = np.full(n, np.nan, dtype=np.float32)
    accel = np.full(n, np.nan, dtype=np.float32)
    exposure = np.full(n, np.nan, dtype=np.float32)
    if n <= 0:
        return slope, accel, exposure
    win = max(2, int(window))
    minp = max(2, int(min_periods))
    y = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(y)
    x = np.arange(n, dtype=np.float64)
    y0 = np.where(finite, y, 0.0)
    x0 = np.where(finite, x, 0.0)
    xy0 = np.where(finite, x * y0, 0.0)
    x20 = np.where(finite, x * x, 0.0)
    ext0 = np.where(finite & extreme.astype(bool), 1.0, 0.0)
    count = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
    sy = np.concatenate([[0.0], np.cumsum(y0)])
    sx = np.concatenate([[0.0], np.cumsum(x0)])
    sxy = np.concatenate([[0.0], np.cumsum(xy0)])
    sx2 = np.concatenate([[0.0], np.cumsum(x20)])
    sext = np.concatenate([[0.0], np.cumsum(ext0)])
    for pos in range(n):
        left = max(0, pos - win + 1)
        right = pos + 1
        c = count[right] - count[left]
        if c < minp:
            continue
        sum_x = sx[right] - sx[left]
        sum_y = sy[right] - sy[left]
        sum_xy = sxy[right] - sxy[left]
        sum_x2 = sx2[right] - sx2[left]
        denom = c * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-12:
            slope[pos] = float((c * sum_xy - sum_x * sum_y) / denom)
        exposure[pos] = float((sext[right] - sext[left]) / max(c, 1.0))
    prev = np.roll(slope, 1)
    accel[1:] = slope[1:] - prev[1:]
    accel[~np.isfinite(slope) | ~np.isfinite(prev)] = np.nan
    return slope, accel, exposure


def _generate_rolling_state_transform_features(
    panel: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    windows: Sequence[int],
    extreme_z: float,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    if panel.empty or timestamp_col not in panel.columns or symbol_col not in panel.columns:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    features = [feature for feature in dict.fromkeys(str(f) for f in raw_features if str(f)) if feature in panel.columns]
    if not features:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    ts = pd.to_datetime(panel[timestamp_col], utc=True, errors="coerce")
    ts_ns = ts.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
    symbols = panel[symbol_col].astype(str)
    symbol_codes, _uniques = pd.factorize(symbols, sort=True)
    valid_order_mask = ts.notna().to_numpy(dtype=bool) & (symbol_codes >= 0)
    order = np.lexsort((ts_ns, symbol_codes))
    order = order[valid_order_mask[order]]
    if order.size == 0:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    sorted_codes = symbol_codes[order]
    breaks = np.flatnonzero(np.diff(sorted_codes) != 0) + 1
    group_bounds = np.split(np.arange(order.size, dtype=np.int64), breaks)
    clean_windows = tuple(dict.fromkeys(max(2, int(w)) for w in windows if int(w) > 1))
    if not clean_windows:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    cols: dict[str, np.ndarray] = {}
    z_threshold = max(float(extreme_z), 0.25)
    for feature in features:
        raw = pd.to_numeric(panel[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(raw)
        if int(finite.sum()) < 50:
            continue
        center, scale = _robust_center_scale(raw[finite])
        z = np.clip((raw.astype(np.float64, copy=False) - center) / scale, -12.0, 12.0).astype(np.float32)
        extreme = np.abs(z) >= z_threshold
        z_sorted = z[order]
        extreme_sorted = extreme[order]
        for window in clean_windows:
            minp = min(window, max(3, window // 3))
            slope_sorted = np.full(order.size, np.nan, dtype=np.float32)
            accel_sorted = np.full(order.size, np.nan, dtype=np.float32)
            exposure_sorted = np.full(order.size, np.nan, dtype=np.float32)
            for locs in group_bounds:
                if locs.size < minp:
                    continue
                s, a, e = _rolling_slope_exposure_group(
                    z_sorted[locs],
                    extreme_sorted[locs],
                    window=window,
                    min_periods=minp,
                )
                slope_sorted[locs] = s
                accel_sorted[locs] = a
                exposure_sorted[locs] = e
            for prefix, values_sorted in (
                ("roll_slope", slope_sorted),
                ("roll_accel", accel_sorted),
                ("extreme_exposure", exposure_sorted),
            ):
                arr = np.full(len(panel), np.nan, dtype=np.float32)
                arr[order] = values_sorted
                cols[f"{prefix}_w{window}__{feature}"] = arr
    if not cols:
        return pd.DataFrame(index=panel.index, dtype=np.float32)
    out = pd.DataFrame(cols, index=panel.index, dtype=np.float32)
    _log(
        f"[features] breakout rolling state transforms sources={len(features)} "
        f"windows={list(clean_windows)} cols={out.shape[1]}"
    )
    return out


def _safe_feature_slug(value: str, *, max_prefix: int = 72) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    clean = re.sub(r"_+", "_", clean)
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=5).hexdigest()
    return f"{clean[:max_prefix].strip('_')}_{digest}" if clean else digest


def _rolling_rate_group(values: np.ndarray, *, window: int, min_periods: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    out = np.full(n, np.nan, dtype=np.float32)
    if n <= 0:
        return out
    finite = np.isfinite(arr)
    vals = np.where(finite, arr, 0.0)
    counts = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
    sums = np.concatenate([[0.0], np.cumsum(vals)])
    win = max(2, int(window))
    minp = max(2, int(min_periods))
    for pos in range(n):
        left = max(0, pos - win + 1)
        right = pos + 1
        c = counts[right] - counts[left]
        if c >= minp:
            out[pos] = float((sums[right] - sums[left]) / max(c, 1.0))
    return out


def _load_ebm_threshold_registry(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"feature", "threshold_robust_z_median"}
    if not required.issubset(frame.columns):
        raise ValueError(f"EBM threshold registry must include {sorted(required)}: {path}")
    return frame.copy()


def _stable_ebm_threshold_rows(thresholds: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    if thresholds.empty:
        return pd.DataFrame()
    work = thresholds.copy()
    for col in [
        "loeo_selection_frequency",
        "false_alarm_rate_control_mean",
        "harmful_region_episode_row_count",
        "delta_logloss_mean",
        "delta_brier_mean",
        "threshold_robust_z_median",
        "high_direction_share",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    mask = (
        work.get("loeo_selection_frequency", pd.Series(0.0, index=work.index)).ge(
            float(config.ebm_threshold_min_selection_frequency)
        )
        & work.get("false_alarm_rate_control_mean", pd.Series(np.inf, index=work.index)).le(
            float(config.ebm_threshold_max_false_alarm_rate)
        )
        & work.get("harmful_region_episode_row_count", pd.Series(0.0, index=work.index)).ge(
            int(config.ebm_threshold_min_episode_rows)
        )
        & work.get("threshold_robust_z_median", pd.Series(np.nan, index=work.index)).notna()
    )
    if bool(config.ebm_threshold_require_positive_lift):
        mask &= (
            work.get("delta_logloss_mean", pd.Series(-np.inf, index=work.index)).ge(0.0)
            | work.get("delta_brier_mean", pd.Series(-np.inf, index=work.index)).ge(0.0)
        )
    work = work.loc[mask].copy()
    if work.empty:
        return work
    work["threshold_direction"] = np.where(
        work.get("high_direction_share", pd.Series(1.0, index=work.index)).fillna(1.0).ge(0.5),
        "high",
        "low",
    )
    work["threshold_registry_score"] = (
        work["loeo_selection_frequency"].fillna(0.0)
        * np.maximum(work["delta_logloss_mean"].fillna(0.0), 0.0).add(
            np.maximum(work["delta_brier_mean"].fillna(0.0), 0.0),
            fill_value=0.0,
        )
        * (1.0 - work["false_alarm_rate_control_mean"].fillna(1.0).clip(0.0, 1.0))
        * np.log1p(work["harmful_region_episode_row_count"].fillna(0.0).clip(lower=0.0))
    )
    return work.sort_values(
        ["threshold_registry_score", "loeo_selection_frequency", "delta_logloss_mean"],
        ascending=False,
        kind="mergesort",
    )


def _threshold_rows_for_head(
    registry: pd.DataFrame,
    *,
    layer: str,
    strategy: str,
    slice_name: str,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    work = _stable_ebm_threshold_rows(registry, config)
    if work.empty:
        return work
    mask = pd.Series(True, index=work.index)
    if "layer" in work.columns:
        mask &= work["layer"].astype(str).eq(str(layer))
    if "strategy" in work.columns:
        mask &= work["strategy"].astype(str).eq(str(strategy))
    if "slice" in work.columns:
        mask &= work["slice"].astype(str).eq(str(slice_name))
    selected = work.loc[mask].copy()
    if selected.empty and "strategy" in work.columns:
        # Fall back to layer/slice-level thresholds only when no exact strategy
        # match exists. This lets broad stable thresholds seed new heads without
        # letting one strategy dominate when a head-specific row is available.
        mask = pd.Series(True, index=work.index)
        if "layer" in work.columns:
            mask &= work["layer"].astype(str).eq(str(layer))
        if "slice" in work.columns:
            mask &= work["slice"].astype(str).eq(str(slice_name))
        selected = work.loc[mask].copy()
    return selected


def _generate_ebm_threshold_state_features(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    layer: str,
    strategy: str,
    slice_name: str,
    config: AnalysisConfig,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    if (
        frame.empty
        or registry.empty
        or not bool(config.ebm_threshold_state_features_enabled)
        or timestamp_col not in frame.columns
        or symbol_col not in frame.columns
    ):
        return pd.DataFrame(index=frame.index, dtype=np.float32)
    rows = _threshold_rows_for_head(
        registry,
        layer=layer,
        strategy=strategy,
        slice_name=slice_name,
        config=config,
    )
    if rows.empty:
        return pd.DataFrame(index=frame.index, dtype=np.float32)
    rows = rows.drop_duplicates(["feature", "threshold_direction"], keep="first")
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    ts_ns = ts.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
    symbol_codes, _uniques = pd.factorize(frame[symbol_col].astype(str), sort=True)
    valid_order_mask = ts.notna().to_numpy(dtype=bool) & (symbol_codes >= 0)
    order = np.lexsort((ts_ns, symbol_codes))
    order = order[valid_order_mask[order]]
    if order.size == 0:
        return pd.DataFrame(index=frame.index, dtype=np.float32)
    sorted_codes = symbol_codes[order]
    breaks = np.flatnonzero(np.diff(sorted_codes) != 0) + 1
    group_bounds = np.split(np.arange(order.size, dtype=np.int64), breaks)
    clean_windows = tuple(dict.fromkeys(max(2, int(w)) for w in config.advanced_transform_windows if int(w) > 1))
    if not clean_windows:
        clean_windows = (24, 72)
    cols: dict[str, np.ndarray] = {}
    used = 0
    for row in rows.itertuples(index=False):
        feature = str(getattr(row, "feature"))
        if feature not in frame.columns:
            continue
        threshold = _safe_float(getattr(row, "threshold_robust_z_median", np.nan))
        if not np.isfinite(threshold):
            continue
        direction = str(getattr(row, "threshold_direction", "high")).lower()
        raw = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(raw)
        if int(finite.sum()) < 50:
            continue
        center, scale = _robust_center_scale(raw[finite])
        z = np.clip((raw.astype(np.float64, copy=False) - center) / scale, -12.0, 12.0).astype(np.float32)
        state = z >= float(threshold) if direction == "high" else z <= float(threshold)
        state &= np.isfinite(z)
        slug = _safe_feature_slug(f"{layer}_{strategy}_{slice_name}_{feature}_{direction}_{threshold:.4f}")
        base_name = f"ebm_state_{direction}__{slug}"
        state_f = state.astype(np.float32)
        cols[f"{base_name}__flag"] = state_f
        share = (
            pd.DataFrame({timestamp_col: ts, "__state": state_f})
            .groupby(timestamp_col, sort=True)["__state"]
            .mean()
            .rename(f"{base_name}__xs_share")
        )
        xs = pd.DataFrame({timestamp_col: ts, "__row_pos": np.arange(len(frame), dtype=np.int64)})
        xs = xs.merge(share.reset_index(), on=timestamp_col, how="left", copy=False).sort_values("__row_pos", kind="mergesort")
        cols[f"{base_name}__xs_share"] = xs[f"{base_name}__xs_share"].to_numpy(dtype=np.float32, copy=False)

        z_sorted = np.where(state[order], z[order], np.nan).astype(np.float32)
        state_sorted = state[order]
        entry_sorted = np.zeros(order.size, dtype=np.float32)
        exit_sorted = np.zeros(order.size, dtype=np.float32)
        for locs in group_bounds:
            flags = state_sorted[locs]
            prev = np.r_[False, flags[:-1]]
            entry_sorted[locs] = (flags & ~prev).astype(np.float32)
            exit_sorted[locs] = (~flags & prev).astype(np.float32)
        for window in clean_windows:
            minp = min(window, max(3, window // 3))
            slope_sorted = np.full(order.size, np.nan, dtype=np.float32)
            accel_sorted = np.full(order.size, np.nan, dtype=np.float32)
            exposure_sorted = np.full(order.size, np.nan, dtype=np.float32)
            entry_rate_sorted = np.full(order.size, np.nan, dtype=np.float32)
            exit_rate_sorted = np.full(order.size, np.nan, dtype=np.float32)
            for locs in group_bounds:
                if locs.size < minp:
                    continue
                s, a, e = _rolling_slope_exposure_group(
                    z_sorted[locs],
                    state_sorted[locs],
                    window=window,
                    min_periods=minp,
                )
                slope_sorted[locs] = s
                accel_sorted[locs] = a
                exposure_sorted[locs] = e
                entry_rate_sorted[locs] = _rolling_rate_group(entry_sorted[locs], window=window, min_periods=minp)
                exit_rate_sorted[locs] = _rolling_rate_group(exit_sorted[locs], window=window, min_periods=minp)
            for suffix, values_sorted in (
                (f"exposure_w{window}", exposure_sorted),
                (f"slope_w{window}", slope_sorted),
                (f"accel_w{window}", accel_sorted),
                (f"entry_intensity_w{window}", entry_rate_sorted),
                (f"exit_intensity_w{window}", exit_rate_sorted),
            ):
                arr = np.full(len(frame), np.nan, dtype=np.float32)
                arr[order] = values_sorted
                cols[f"{base_name}__{suffix}"] = arr
        used += 1
    if not cols:
        return pd.DataFrame(index=frame.index, dtype=np.float32)
    out = pd.DataFrame(cols, index=frame.index, dtype=np.float32)
    _log(
        f"[features] EBM threshold state features {layer} {strategy} {slice_name}: "
        f"thresholds_used={used} cols={out.shape[1]}"
    )
    return out


def _breakout_raw_features_for_generation(
    panel: pd.DataFrame,
    raw_screen: pd.DataFrame,
    *,
    config: AnalysisConfig,
) -> list[str]:
    if panel.empty or raw_screen.empty:
        return []
    selected = raw_screen.loc[raw_screen["selected_for_operator_generation"].astype(bool)].copy()
    if selected.empty:
        return []
    selected = selected.sort_values("raw_breakout_link_score", ascending=False, kind="mergesort")
    raw_limit = int(config.raw_exploration_max_features)
    selected_features = selected["feature"].astype(str).tolist()
    if raw_limit > 0:
        selected_features = selected_features[:raw_limit]
    return [
        feature
        for feature in selected_features
        if feature in panel.columns
    ]


def _previous_meta_parent_screen(
    features: Sequence[str],
    *,
    base_score: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, feature in enumerate(dict.fromkeys(str(f) for f in features if str(f)), start=1):
        if not _is_previous_meta_parent_raw_feature(feature):
            continue
        rows.append(
            {
                "feature": feature,
                "feature_family": _feature_family(feature),
                "raw_breakout_link_score": float(base_score) / float(max(rank, 1)),
                "weighted_raw_breakout_link_score": float(base_score) / float(max(rank, 1)),
                "raw_exploration_pass_count": 1,
                "raw_exploration_pass_weight_share": 1.0,
                "selected_for_operator_generation": True,
                "operator_generation_source": "previous_meta_parent",
            }
        )
    return pd.DataFrame(rows)


def _merge_previous_meta_parent_screen(
    raw_screen: pd.DataFrame,
    previous_parent_features: Sequence[str],
) -> pd.DataFrame:
    parent_screen = _previous_meta_parent_screen(previous_parent_features)
    if parent_screen.empty:
        return raw_screen.copy() if isinstance(raw_screen, pd.DataFrame) else pd.DataFrame()
    if raw_screen is None or raw_screen.empty:
        return parent_screen
    work = raw_screen.copy()
    if "operator_generation_source" not in work.columns:
        work["operator_generation_source"] = "breakout_screen"
    for col in parent_screen.columns:
        if col not in work.columns:
            if col == "selected_for_operator_generation":
                work[col] = False
            elif col == "feature":
                work[col] = ""
            else:
                work[col] = np.nan
    combined = pd.concat([work, parent_screen], ignore_index=True, sort=False)
    combined["feature"] = combined["feature"].astype(str)
    combined["selected_for_operator_generation"] = combined["selected_for_operator_generation"].fillna(False).astype(bool)
    combined["raw_breakout_link_score"] = pd.to_numeric(
        combined.get("raw_breakout_link_score", pd.Series(0.0, index=combined.index)),
        errors="coerce",
    ).fillna(0.0)
    combined["__source_priority"] = np.where(
        combined.get("operator_generation_source", "").astype(str).eq("previous_meta_parent"),
        1,
        0,
    )
    combined = combined.sort_values(
        ["selected_for_operator_generation", "__source_priority", "raw_breakout_link_score"],
        ascending=[False, False, False],
        kind="mergesort",
    ).drop_duplicates("feature", keep="first")
    return combined.drop(columns=["__source_priority"], errors="ignore").reset_index(drop=True)


def _generate_breakout_exploration_composites(
    panel: pd.DataFrame,
    raw_screen: pd.DataFrame,
    *,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Generate URL-style operators from raw features loosely linked to breakouts."""

    raw_features = _breakout_raw_features_for_generation(panel, raw_screen, config=config)
    if not raw_features:
        return pd.DataFrame(index=panel.index)
    selected = raw_screen.loc[raw_screen["feature"].astype(str).isin(raw_features)].copy()
    try:
        from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
            prepare_frame_context,
        )
        from extreme_price_movements.unsupervised_regime_learning.operators import (
            generate_autocorr_operator_features,
            generate_pair_operator_features,
            generate_quantile_operator_features,
        )
    except Exception as exc:
        _log(f"[features] breakout URL composite generation unavailable: {type(exc).__name__}: {exc}")
        return pd.DataFrame(index=panel.index)

    t0 = time.perf_counter()
    _log(
        f"[features] breakout exploration operator generation raw_sources={len(raw_features)} "
        f"families={selected['feature_family'].nunique() if 'feature_family' in selected else 0}"
    )
    context = prepare_frame_context(panel, symbol_col="symbol", timestamp_col="timestamp")
    parts: list[pd.DataFrame] = []

    part_t0 = time.perf_counter()
    xs = _generate_cross_sectional_regime_features(
        panel,
        raw_features,
        timestamp_col="timestamp",
        symbol_col="symbol",
    )
    if not xs.empty:
        parts.append(xs.astype(np.float32, copy=False))
    _log(f"[features] breakout cross-sectional operators cols={xs.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s")

    if bool(config.advanced_transform_enabled):
        part_t0 = time.perf_counter()
        transforms = _generate_rolling_state_transform_features(
            panel,
            raw_features,
            windows=config.advanced_transform_windows,
            extreme_z=float(config.advanced_transform_extreme_z),
            timestamp_col="timestamp",
            symbol_col="symbol",
        )
        if not transforms.empty:
            parts.append(transforms.astype(np.float32, copy=False))
        _log(
            f"[features] breakout rolling slope/accel/extreme transforms "
            f"cols={transforms.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    part_t0 = time.perf_counter()
    quantile = generate_quantile_operator_features(
        panel,
        raw_features,
        window=72,
        min_periods=12,
        symbol_col="symbol",
        timestamp_col="timestamp",
        context=context,
    )
    if not quantile.empty:
        parts.append(quantile.astype(np.float32, copy=False))
    _log(f"[features] breakout quantile operators cols={quantile.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s")

    for window in (24, 72):
        part_t0 = time.perf_counter()
        autocorr = generate_autocorr_operator_features(
            panel,
            raw_features,
            window=window,
            lag=1,
            min_periods=max(8, window // 4),
            symbol_col="symbol",
            timestamp_col="timestamp",
            context=context,
        )
        if not autocorr.empty:
            parts.append(autocorr.astype(np.float32, copy=False))
        _log(
            f"[features] breakout autocorr operators window={window} "
            f"cols={autocorr.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    pair_scores = _breakout_pair_scores_from_raw_screen(
        selected.loc[selected["feature"].astype(str).isin(raw_features)],
        max_pairs=max(20, len(raw_features) * 3),
    )
    for window in (24, 72):
        if pair_scores.empty:
            break
        part_t0 = time.perf_counter()
        pair = generate_pair_operator_features(
            panel,
            pair_scores,
            window=window,
            min_periods=max(8, window // 4),
            symbol_col="symbol",
            timestamp_col="timestamp",
            context=context,
        )
        if not pair.empty:
            parts.append(pair.astype(np.float32, copy=False))
        _log(
            f"[features] breakout cov/corr pair operators window={window} "
            f"pairs={len(pair_scores)} cols={pair.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    family_groups: dict[str, list[str]] = {}
    for feature in raw_features:
        family_groups.setdefault(_feature_family(feature), []).append(feature)
    family_groups = {family: cols for family, cols in family_groups.items() if len(cols) >= 2}
    if len(raw_features) >= 2:
        family_groups["breakout_all"] = raw_features
    if family_groups:
        part_t0 = time.perf_counter()
        eigen = _generate_fast_timestamp_eigen_features(
            panel,
            family_groups,
            window=72,
            min_periods=12,
            timestamp_col="timestamp",
        )
        if not eigen.empty:
            parts.append(eigen.astype(np.float32, copy=False))
        _log(
            f"[features] breakout eigen operators groups={len(family_groups)} "
            f"cols={eigen.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s"
        )

    if bool(config.breakout_generate_svd_knn) and len(raw_features) >= 4:
        from extreme_price_movements.unsupervised_regime_learning.operators import (
            fit_transform_svd_knn_features_walk_forward,
        )

        part_t0 = time.perf_counter()
        svd, _state = fit_transform_svd_knn_features_walk_forward(
            panel,
            raw_features,
            svd_components=[8, 16],
            knn_svd_components=min(16, max(2, len(raw_features))),
            knn_neighbors=10,
            timestamp_col="timestamp",
            symbol_col="symbol",
            block_hours=24 * 14,
            min_prior_rows=64,
            max_reference_rows=3000,
            knn_max_reference_rows=1500,
            sample_time_bins=8,
        )
        if not svd.empty:
            parts.append(svd.astype(np.float32, copy=False))
        _log(f"[features] breakout SVD/KNN operators cols={svd.shape[1]} elapsed={time.perf_counter() - part_t0:.1f}s")
    elif len(raw_features) >= 4:
        _log("[features] breakout SVD/KNN operators skipped; pass --enable-breakout-svd-knn to include them")

    if not parts:
        return pd.DataFrame(index=panel.index)
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].astype(np.float32, copy=False)
    _log(
        f"[features] breakout exploration operator generation complete cols={out.shape[1]} "
        f"elapsed={time.perf_counter() - t0:.1f}s"
    )
    return out


def _collect_oof_row_keys(files: Sequence[tuple[str, str, Path]], *, config: AnalysisConfig) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _layer, _strategy, path in files:
        cols = set(_parquet_columns(path))
        if "timestamp" not in cols or "symbol" not in cols:
            continue
        keys = pd.read_parquet(path, columns=["timestamp", "symbol"])
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
        keys["symbol"] = keys["symbol"].astype(str)
        keys = _filter_frame_by_analysis_period(keys, config)
        parts.append(keys.loc[keys["timestamp"].notna() & keys["symbol"].ne("")])
    if not parts:
        return pd.DataFrame(columns=["timestamp", "symbol"])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["timestamp", "symbol"], keep="last")
    return out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def _hydrate_feature_store_for_keys(
    feature_dir: Path | None,
    keys: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    generate_url_composites: bool = True,
    cache_dir: Path | None = None,
    cache_enabled: bool = True,
    refresh_cache: bool = False,
    cache_compression: str = DEFAULT_PARQUET_CACHE_COMPRESSION,
) -> pd.DataFrame | None:
    if feature_dir is None or not feature_columns:
        return None
    if not feature_dir.exists():
        raise FileNotFoundError(feature_dir)
    if keys.empty:
        return None
    requested_final = list(dict.fromkeys(str(col) for col in feature_columns if str(col)))
    primitive_sources = _infer_url_primitive_sources(requested_final)
    requested = list(dict.fromkeys(primitive_sources + requested_final))
    cache_path: Path | None = None
    if bool(cache_enabled) and cache_dir is not None:
        row_hash = _row_universe_hash(keys)
        column_hash = _stable_hash_payload(
            {
                "feature_dir": str(feature_dir.resolve()),
                "requested_columns": requested,
                "generate_url_composites": bool(generate_url_composites),
            },
            digest_size=12,
        )
        cache_path = cache_dir / f"hydrated_feature_store_{row_hash}_{column_hash}.parquet"
        if cache_path.exists() and not bool(refresh_cache):
            cached = pd.read_parquet(cache_path)
            if {"timestamp", "symbol"}.issubset(cached.columns):
                cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
                cached["symbol"] = cached["symbol"].astype(str)
            _log(
                f"[features] loaded hydrated feature-store cache rows={len(cached)} "
                f"cols={max(len(cached.columns) - 2, 0)} path={cache_path}"
            )
            return cached
    rows: list[pd.DataFrame] = []
    available_counts: list[int] = []
    for symbol, group in keys.groupby("symbol", sort=False):
        path = _feature_path_for_symbol(feature_dir, str(symbol))
        if not path.exists():
            continue
        available = set(_parquet_columns(path))
        present = [col for col in requested if col in available]
        available_counts.append(len(present))
        if not present:
            continue
        values = pd.read_parquet(path, columns=present)
        idx = pd.to_datetime(values.index, utc=True, errors="coerce")
        idx_ns = idx.to_numpy(dtype="datetime64[ns]", copy=False)
        wanted_ns = pd.to_datetime(
            group["timestamp"],
            utc=True,
            errors="coerce",
        ).dropna().to_numpy(dtype="datetime64[ns]", copy=False)
        if wanted_ns.size == 0:
            continue
        mask = np.isin(idx_ns, wanted_ns, assume_unique=False)
        if not bool(mask.any()):
            continue
        values = values.loc[mask, present].copy()
        values["timestamp"] = pd.to_datetime(values.index, utc=True, errors="coerce")
        values["symbol"] = str(symbol)
        rows.append(values.reset_index(drop=True))
    if not rows:
        _log(
            f"[features] feature-store hydration found no matching rows. "
            f"feature_dir={feature_dir} requested_cols={len(requested)}"
        )
        return None
    features = pd.concat(rows, ignore_index=True)
    for col in [c for c in features.columns if c not in {"timestamp", "symbol"}]:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(np.float32, copy=False)
    features = features.sort_values(["timestamp", "symbol"], kind="mergesort").drop_duplicates(
        ["timestamp", "symbol"],
        keep="last",
    )
    merged = keys.merge(features, on=["timestamp", "symbol"], how="left", copy=False)
    raw_feature_cols = [col for col in merged.columns if col not in {"timestamp", "symbol"}]
    coverage = float(merged[raw_feature_cols].notna().any(axis=1).mean()) if raw_feature_cols and len(merged) else 0.0
    _log(
        f"[features] hydrated feature-store panel rows={len(merged)} "
        f"cols={len(raw_feature_cols)} row_overlap={coverage:.3f} "
        f"requested_cols={len(requested_final)} primitive_source_cols={len(primitive_sources)} mean_available_per_symbol="
        f"{(float(np.mean(available_counts)) if available_counts else 0.0):.1f}"
    )
    if generate_url_composites:
        generated = _generate_selected_url_composites(
            merged,
            requested_final,
            primitive_sources,
        )
        if not generated.empty:
            add_cols = [col for col in generated.columns if col not in merged.columns]
            merged = pd.concat([merged, generated[add_cols]], axis=1)
            _log(
                f"[features] generated URL composite columns={len(add_cols)} "
                f"total_feature_cols={len(merged.columns) - 2}"
            )
        missing_selected = [
            col
            for col in requested_final
            if col not in {"timestamp", "symbol"} and col not in merged.columns
        ]
        if missing_selected:
            _log(
                f"[features] selected URL columns still unavailable={len(missing_selected)} "
                f"sample={missing_selected[:12]}"
            )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False, compression=cache_compression)
        _log(
            f"[features] saved hydrated feature-store cache rows={len(merged)} "
            f"cols={max(len(merged.columns) - 2, 0)} compression={cache_compression} path={cache_path}"
        )
    return merged


def _combine_feature_frames(*frames: pd.DataFrame | None) -> pd.DataFrame | None:
    valid = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid:
        return None
    out: pd.DataFrame | None = None
    for frame in valid:
        part = frame.copy()
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True, errors="coerce")
        part["symbol"] = part["symbol"].astype(str)
        feature_cols = [col for col in part.columns if col not in {"timestamp", "symbol"}]
        if out is None:
            out = part[["timestamp", "symbol", *feature_cols]]
            continue
        existing = set(out.columns)
        add_cols = [col for col in feature_cols if col not in existing]
        if add_cols:
            out = out.merge(part[["timestamp", "symbol", *add_cols]], on=["timestamp", "symbol"], how="outer", copy=False)
    return out


def _merge_feature_frame(frame: pd.DataFrame, feature_frame: pd.DataFrame | None) -> pd.DataFrame:
    if feature_frame is None:
        return frame
    left = frame.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    left["symbol"] = left["symbol"].astype(str)
    existing = set(left.columns)
    feature_cols = [
        col
        for col in feature_frame.columns
        if col not in {"timestamp", "symbol"} and col not in existing
    ]
    if not feature_cols:
        return left
    right = feature_frame[["timestamp", "symbol", *feature_cols]]
    merged = left.merge(right, on=["timestamp", "symbol"], how="left", copy=False)
    if feature_cols:
        overlap = float(merged[feature_cols].notna().any(axis=1).mean()) if len(merged) else 0.0
        _log(
            f"[features] external feature overlap rows={overlap:.3f} "
            f"cols_added={len(feature_cols)}"
        )
    return merged


def _expected_breakout_generated_columns(
    panel: pd.DataFrame,
    raw_features: Sequence[str],
    raw_screen: pd.DataFrame,
    *,
    config: AnalysisConfig,
) -> list[str]:
    cols: list[str] = []
    portable = [feature for feature in _asset_portable_features(raw_features) if feature in panel.columns]
    for feature in portable:
        cols.extend(
            [
                f"xs_mean__{feature}",
                f"xs_median__{feature}",
                f"xs_std__{feature}",
                f"xs_iqr__{feature}",
                f"xs_dispersion__{feature}",
            ]
        )
    family_groups: dict[str, list[str]] = {}
    for feature in portable:
        family_groups.setdefault(_feature_family(feature), []).append(feature)
    family_groups = {family: features for family, features in family_groups.items() if len(features) >= 2}
    if len(portable) >= 2:
        family_groups["asset_portable_all"] = portable
    for family in family_groups:
        safe_group = _safe_operator_group_name(f"xs_{family}")
        cols.extend(
            [
                f"xs_cov_pc1_concentration__{safe_group}",
                f"xs_cov_top3_share__{safe_group}",
                f"xs_cov_effective_rank__{safe_group}",
                f"xs_cov_participation_ratio__{safe_group}",
                f"xs_cov_mean_abs_corr__{safe_group}",
            ]
        )
    if bool(config.advanced_transform_enabled):
        clean_windows = tuple(dict.fromkeys(max(2, int(w)) for w in config.advanced_transform_windows if int(w) > 1))
        for feature in raw_features:
            for window in clean_windows:
                cols.extend(
                    [
                        f"roll_slope_w{window}__{feature}",
                        f"roll_accel_w{window}__{feature}",
                        f"extreme_exposure_w{window}__{feature}",
                    ]
                )
    for feature in raw_features:
        cols.extend(
            [
                f"q_iqr__{feature}",
                f"q_tail_width__{feature}",
                f"q_upper_tail__{feature}",
                f"q_lower_tail__{feature}",
                f"q_tail_asym__{feature}",
                f"q_percentile_rank__{feature}",
            ]
        )
    for window in (24, 72):
        for feature in raw_features:
            cols.append(f"autocorr_lag1_w{window}__{feature}")
    pair_scores = _breakout_pair_scores_from_raw_screen(
        raw_screen.loc[raw_screen["feature"].astype(str).isin(raw_features)],
        max_pairs=max(20, len(raw_features) * 3),
    )
    if not pair_scores.empty:
        for window in (24, 72):
            for row in pair_scores.itertuples(index=False):
                left = str(row.feature_i)
                right = str(row.feature_j)
                cols.extend(
                    [
                        f"cov_w{window}__{left}__{right}",
                        f"corr_w{window}__{left}__{right}",
                    ]
                )
    operator_groups: dict[str, list[str]] = {}
    for feature in raw_features:
        operator_groups.setdefault(_feature_family(feature), []).append(feature)
    operator_groups = {family: values for family, values in operator_groups.items() if len(values) >= 2}
    if len(raw_features) >= 2:
        operator_groups["breakout_all"] = list(raw_features)
    for group in operator_groups:
        safe_group = _safe_operator_group_name(group)
        cols.extend(
            [
                f"eig_largest_share__{safe_group}",
                f"eig_top3_share__{safe_group}",
                f"eig_effective_rank__{safe_group}",
                f"eig_participation_ratio__{safe_group}",
                f"eig_turnover__{safe_group}",
            ]
        )
    # SVD/KNN names are produced by the upstream operator module and may change;
    # cache reuse still works for them after the first generation, but we do not
    # use a hand-maintained expected-name list to decide completeness.
    return list(dict.fromkeys(cols))


def _expected_selected_generated_columns(feature_columns: Sequence[str], frame: pd.DataFrame) -> list[str]:
    generated_like = []
    for col in dict.fromkeys(str(c) for c in feature_columns if str(c)):
        if col in frame.columns or col in {"timestamp", "symbol"}:
            continue
        if (
            QUANTILE_FEATURE_RE.match(col)
            or AUTOCORR_FEATURE_RE.match(col)
            or PAIR_FEATURE_RE.match(col)
            or EIG_FEATURE_RE.match(col)
            or col.startswith("svd8_")
            or col.startswith("svd16_")
            or col.startswith("knn_")
        ):
            generated_like.append(col)
    return generated_like


def _transform_cache_paths(
    frame: pd.DataFrame,
    *,
    config: AnalysisConfig,
    feature_columns: Sequence[str],
    raw_features: Sequence[str],
    mode: str,
) -> tuple[Path | None, dict[str, object]]:
    metadata: dict[str, object] = {
        "schema_version": 3,
        "cache_layout": "single_appendable_generated_transform_file",
        "mode": mode,
        "artifact_run_id": config.artifact_run_id,
        "feature_store_dir": str(config.feature_store_dir) if config.feature_store_dir is not None else "",
        "feature_columns_json": str(config.feature_columns_json) if config.feature_columns_json is not None else "",
        "row_universe_hash": _row_universe_hash(frame),
        "row_count": int(len(frame)),
        "raw_features": list(raw_features),
        "feature_columns": list(dict.fromkeys(str(col) for col in feature_columns if str(col))),
        "breakout_exploration_enabled": bool(config.breakout_exploration_enabled),
        "advanced_transform_enabled": bool(config.advanced_transform_enabled),
        "advanced_transform_windows": list(config.advanced_transform_windows),
        "advanced_transform_extreme_z": float(config.advanced_transform_extreme_z),
        "breakout_generate_svd_knn": bool(config.breakout_generate_svd_knn),
        "raw_exploration_max_features": int(config.raw_exploration_max_features),
        "previous_meta_parent_report": str(config.previous_meta_parent_report)
        if config.previous_meta_parent_report is not None
        else "",
        "previous_meta_parent_top_n": int(config.previous_meta_parent_top_n),
        "previous_meta_parent_slice": str(config.previous_meta_parent_slice),
        "analysis_start_day": str(config.analysis_start_day or ""),
        "top_rank_slice_only": bool(config.top_rank_slice_only),
        "rank_frac": float(config.rank_frac),
        "generated_transform_cache_max_rows": int(config.generated_transform_cache_max_rows),
        "generated_transform_cache_max_bytes": int(config.generated_transform_cache_max_bytes),
    }
    if (
        not bool(config.transform_cache_enabled)
        or not bool(config.generated_transform_cache_enabled)
        or config.transform_cache_dir is None
    ):
        return None, metadata
    identity = {
        key: metadata[key]
        for key in [
            "schema_version",
            "cache_layout",
            "mode",
            "artifact_run_id",
            "feature_store_dir",
            "feature_columns_json",
            "breakout_exploration_enabled",
            "advanced_transform_enabled",
            "advanced_transform_windows",
            "advanced_transform_extreme_z",
            "breakout_generate_svd_knn",
            "raw_exploration_max_features",
            "previous_meta_parent_report",
            "previous_meta_parent_top_n",
            "previous_meta_parent_slice",
            "analysis_start_day",
            "top_rank_slice_only",
            "rank_frac",
        ]
    }
    digest = _stable_hash_payload(identity, digest_size=16)
    path = config.transform_cache_dir / f"generated_transforms_single_{digest}.parquet"
    return path, metadata


def _cleanup_generated_transform_caches(config: AnalysisConfig) -> dict[str, object]:
    cache_dir = config.transform_cache_dir
    if cache_dir is None or not cache_dir.exists():
        return {"deleted_count": 0, "deleted_bytes": 0}
    keep_n = max(int(config.generated_transform_cache_keep_last_n), 0)
    ttl_days = float(config.generated_transform_cache_ttl_days)
    paths = sorted(
        cache_dir.glob("generated_transforms_*.parquet"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
        reverse=True,
    )
    keep = set(paths[:keep_n])
    cutoff = time.time() - max(ttl_days, 0.0) * 86400.0
    deleted_count = 0
    deleted_bytes = 0
    for path in paths:
        if path in keep:
            continue
        if path.stat().st_mtime >= cutoff:
            continue
        sidecar = path.with_suffix(".json")
        try:
            size = path.stat().st_size
            path.unlink()
            deleted_count += 1
            deleted_bytes += size
            if sidecar.exists():
                deleted_bytes += sidecar.stat().st_size
                sidecar.unlink()
        except FileNotFoundError:
            continue
    if deleted_count:
        _log(
            f"[cache] cleaned generated transform caches deleted={deleted_count} "
            f"bytes={deleted_bytes} keep_last_n={keep_n} ttl_days={ttl_days:g}"
        )
    return {"deleted_count": deleted_count, "deleted_bytes": deleted_bytes}


def _refresh_generated_transform_caches(config: AnalysisConfig) -> dict[str, object]:
    """Remove generated-transform caches once at the start of a refreshed run."""

    cache_dir = config.transform_cache_dir
    if cache_dir is None or not cache_dir.exists():
        return {"deleted_count": 0, "deleted_bytes": 0}
    deleted_count = 0
    deleted_bytes = 0
    for path in sorted(cache_dir.glob("generated_transforms_*.parquet")):
        sidecar = path.with_suffix(".json")
        try:
            size = path.stat().st_size
            path.unlink()
            deleted_count += 1
            deleted_bytes += size
            if sidecar.exists():
                deleted_bytes += sidecar.stat().st_size
                sidecar.unlink()
        except FileNotFoundError:
            continue
    if deleted_count:
        _log(
            f"[cache] refreshed generated transform caches deleted={deleted_count} "
            f"bytes={deleted_bytes}"
        )
    return {"deleted_count": deleted_count, "deleted_bytes": deleted_bytes}


def _cache_scope_from_metadata(metadata: dict[str, object]) -> str:
    return str(metadata.get("row_universe_hash") or "")


def _read_transform_cache(path: Path, frame: pd.DataFrame, metadata: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=frame.index)
    cache = pd.read_parquet(path)
    if not {"timestamp", "symbol"}.issubset(cache.columns):
        return pd.DataFrame(index=frame.index)
    scope = _cache_scope_from_metadata(metadata or {})
    if scope and "__cache_scope" in cache.columns:
        cache = cache.loc[cache["__cache_scope"].astype(str).eq(scope)].copy()
    elif "__cache_scope" in cache.columns:
        cache = cache.iloc[0:0].copy()
    if cache.empty:
        return pd.DataFrame(index=frame.index)
    keys = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "symbol": frame["symbol"].astype(str),
            "__row_pos": np.arange(len(frame), dtype=np.int64),
        }
    )
    cache = cache.copy()
    cache["timestamp"] = pd.to_datetime(cache["timestamp"], utc=True, errors="coerce")
    cache["symbol"] = cache["symbol"].astype(str)
    cache = cache.drop_duplicates(["timestamp", "symbol"], keep="last")
    feature_cols = [col for col in cache.columns if col not in {"timestamp", "symbol", "__cache_scope"}]
    if not feature_cols:
        return pd.DataFrame(index=frame.index)
    merged = keys.merge(cache[["timestamp", "symbol", *feature_cols]], on=["timestamp", "symbol"], how="left", copy=False)
    merged = merged.sort_values("__row_pos", kind="mergesort")
    out = merged.drop(columns=["timestamp", "symbol", "__row_pos"], errors="ignore")
    out.index = frame.index
    return out.astype(np.float32, copy=False)


def _write_transform_cache(
    path: Path,
    frame: pd.DataFrame,
    generated: pd.DataFrame,
    metadata: dict[str, object],
    *,
    compression: str = DEFAULT_PARQUET_CACHE_COMPRESSION,
) -> bool:
    if generated.empty:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_cols = [col for col in generated.columns if col not in {"timestamp", "symbol"}]
    if not feature_cols:
        return False
    scope = _cache_scope_from_metadata(metadata)
    keys = pd.DataFrame(
        {
            "__cache_scope": scope,
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "symbol": frame["symbol"].astype(str),
        },
        index=frame.index,
    )
    cache = pd.concat([keys.reset_index(drop=True), generated[feature_cols].reset_index(drop=True)], axis=1)
    cache = cache.dropna(subset=["timestamp"])
    cache["__cache_scope"] = cache["__cache_scope"].astype(str)
    cache["symbol"] = cache["symbol"].astype(str)
    cache = cache.drop_duplicates(["__cache_scope", "timestamp", "symbol"], keep="last")
    write_cols = feature_cols
    append_row_count = int(len(cache))
    append_feature_count = int(len(write_cols))
    if path.exists():
        existing_row_count = 0
        existing_size = 0
        try:
            existing_size = int(path.stat().st_size)
        except OSError:
            existing_size = 0
        try:
            import pyarrow.parquet as pq

            existing_row_count = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            existing_row_count = 0
        max_rows = max(int(metadata.get("generated_transform_cache_max_rows", 0) or 0), 0)
        max_bytes = max(int(metadata.get("generated_transform_cache_max_bytes", 0) or 0), 0)
        projected_rows = existing_row_count + append_row_count
        if (max_rows and projected_rows > max_rows) or (max_bytes and existing_size > max_bytes):
            _log(
                f"[features] transform cache append skipped projected_rows={projected_rows} "
                f"append_rows={append_row_count} existing_rows={existing_row_count} "
                f"existing_bytes={existing_size} max_rows={max_rows} max_bytes={max_bytes} "
                f"path={path}"
            )
            return False
        try:
            existing = pd.read_parquet(path)
        except Exception:
            existing = pd.DataFrame()
        if {"timestamp", "symbol"}.issubset(existing.columns):
            existing = existing.copy()
            if "__cache_scope" not in existing.columns:
                existing["__cache_scope"] = ""
            existing["__cache_scope"] = existing["__cache_scope"].astype(str)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
            existing["symbol"] = existing["symbol"].astype(str)
            existing = existing.dropna(subset=["timestamp"]).drop_duplicates(
                ["__cache_scope", "timestamp", "symbol"],
                keep="last",
            )
            existing_idx = existing.set_index(["__cache_scope", "timestamp", "symbol"], drop=True)
            cache_idx = cache.set_index(["__cache_scope", "timestamp", "symbol"], drop=True)
            cache_idx = cache_idx.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
            combined = cache_idx.combine_first(existing_idx)
            cache = combined.reset_index()
            write_cols = [col for col in cache.columns if col not in {"timestamp", "symbol", "__cache_scope"}]
    if write_cols:
        cache[write_cols] = cache[write_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
    cache.to_parquet(path, index=False, compression=compression)
    manifest = dict(metadata)
    manifest.update(
        {
            "path": str(path),
            "feature_count": int(len(write_cols)),
            "append_row_count": append_row_count,
            "append_feature_count": append_feature_count,
            "cache_row_count": int(len(cache)),
            "compression": compression,
            "timestamp_min": str(keys["timestamp"].min()),
            "timestamp_max": str(keys["timestamp"].max()),
        }
    )
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return True


def _append_generated_columns(frame: pd.DataFrame, generated: pd.DataFrame, *, source: str) -> pd.DataFrame:
    add_cols = [
        col
        for col in generated.columns
        if col not in frame.columns and col not in {"timestamp", "symbol"}
    ]
    if not add_cols:
        return frame
    out = pd.concat([frame, generated[add_cols].astype(np.float32, copy=False)], axis=1)
    _log(
        f"[features] streamed generated columns={len(add_cols)} "
        f"frame_feature_cols={len(out.columns) - 2} source={source}"
    )
    return out


def _append_streamed_generated_features(
    frame: pd.DataFrame,
    *,
    config: AnalysisConfig,
    feature_columns: Sequence[str],
    raw_breakout_screen: pd.DataFrame,
    previous_meta_parent_features: Sequence[str] = (),
) -> pd.DataFrame:
    if not bool(config.generate_url_composites) or not feature_columns:
        return frame
    if not bool(config.stream_feature_generation):
        return frame
    requested_final = list(dict.fromkeys(str(col) for col in feature_columns if str(col)))
    if bool(config.breakout_exploration_enabled):
        generation_screen = _merge_previous_meta_parent_screen(
            raw_breakout_screen,
            previous_meta_parent_features,
        )
        raw_features = _breakout_raw_features_for_generation(frame, generation_screen, config=config)
        expected_cols = _expected_breakout_generated_columns(
            frame,
            raw_features,
            generation_screen,
            config=config,
        )
        mode = "breakout"
    else:
        raw_features = [
            col
            for col in _infer_url_primitive_sources(requested_final)
            if col in frame.columns
        ]
        expected_cols = _expected_selected_generated_columns(requested_final, frame)
        mode = "selected"
    cache_path, cache_meta = _transform_cache_paths(
        frame,
        config=config,
        feature_columns=requested_final,
        raw_features=raw_features,
        mode=mode,
    )
    cached = pd.DataFrame(index=frame.index)
    if cache_path is not None and cache_path.exists():
        t0 = time.perf_counter()
        cached = _read_transform_cache(cache_path, frame, cache_meta)
        cached_cols = [col for col in cached.columns if col not in {"timestamp", "symbol"}]
        expected_set = set(expected_cols)
        if cached_cols:
            cached_numeric = cached[cached_cols]
            cached_row_coverage = float(cached_numeric.notna().any(axis=1).mean()) if len(cached_numeric) else 0.0
        else:
            cached_row_coverage = 0.0
        if (
            cached_cols
            and cached_row_coverage >= 0.95
            and (not expected_set or expected_set.issubset(cached_cols))
        ):
            _log(
                f"[features] transform cache hit cols={len(cached_cols)} "
                f"row_coverage={cached_row_coverage:.3f} "
                f"path={cache_path} elapsed={time.perf_counter() - t0:.1f}s"
            )
            return _append_generated_columns(frame, cached, source="cache")
        if cached_cols:
            missing = sorted(expected_set - set(cached_cols)) if expected_set else []
            _log(
                f"[features] transform cache partial cols={len(cached_cols)} "
                f"row_coverage={cached_row_coverage:.3f} "
                f"missing_expected={len(missing)} path={cache_path}"
            )
    if bool(config.breakout_exploration_enabled):
        generated = _generate_breakout_exploration_composites(
            frame,
            generation_screen,
            config=config,
        )
    else:
        primitive_sources = [
            col
            for col in _infer_url_primitive_sources(requested_final)
            if col in frame.columns
        ]
        generated = _generate_selected_url_composites(
            frame,
            requested_final,
            primitive_sources,
        )
    if generated is None or generated.empty:
        if not cached.empty:
            return _append_generated_columns(frame, cached, source="partial-cache")
        return frame
    if not cached.empty:
        combined = cached.copy()
        for col in generated.columns:
            if col not in combined.columns:
                combined[col] = generated[col].astype(np.float32, copy=False)
        generated = combined
    if cache_path is not None:
        wrote_cache = _write_transform_cache(
            cache_path,
            frame,
            generated.astype(np.float32, copy=False),
            cache_meta,
            compression=config.parquet_cache_compression,
        )
        if wrote_cache:
            _log(
                f"[features] transform cache saved cols={generated.shape[1]} "
                f"compression={config.parquet_cache_compression} path={cache_path}"
            )
            _cleanup_generated_transform_caches(config)
    return _append_generated_columns(frame, generated, source="generated")


def _latest_regime_feature_artifact(data_root: Path) -> Path | None:
    base = data_root / "artifacts" / "unsupervised_regime_learning_poc"
    if not base.exists():
        return None
    candidates: list[Path] = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        frames = path / "advanced_regime_learning" / "advanced_regime_learning_frames"
        if (frames / "row_keys.pkl").exists() and (
            (path / "regime_context_features.parquet").exists()
            or (frames / "model_regime_features.pkl").exists()
        ):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _config_feature_name_set() -> set[str]:
    names: set[str] = set(BREAKOUT_STRUCTURE_FEATURE_COLUMNS)
    try:
        from extreme_price_movements.config import CFG
    except Exception:
        return names
    for key, value in CFG.items():
        lower = str(key).lower()
        if "feature" not in lower:
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item:
                    names.add(item)
    return names


def _configured_liquidity_execution_feature_columns() -> list[str]:
    """Return registered liquidity/execution feature names for diagnostic hydration.

    These are not used to define labels or predictions; they only expand the
    candidate diagnostic pool when the caller opts in. Keep this broad enough
    to include configured spread/depth/volume proxies, but still name-based and
    deterministic so it does not pull arbitrary feature-store columns.
    """

    try:
        from extreme_price_movements import config as epm_config
    except Exception:
        return []

    names: set[str] = set()
    preferred_groups = (
        "ORDERBOOK_BASE_FEATURE_KEYS",
        "ORDERBOOK_RAW_META_FEATURE_KEYS",
        "ORDERBOOK_NORMALIZED_META_FEATURE_KEYS",
        "ORDERBOOK_META_FEATURE_KEYS",
        "ORDERBOOK_DIAGNOSTIC_ONLY_FEATURE_KEYS",
        "ORDERBOOK_EXCLUDED_STALE_FEATURE_KEYS",
        "REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS",
        "SPREAD_PROXY_FEATURE_KEYS",
        "ROLLING_ALPHA_FEATURE_KEYS",
        "RESIDUAL_BASE_FEATURE_KEYS",
        "RESIDUAL_META_FEATURE_KEYS",
        "REGIME_ADAPTOR_ASSET_FEATURE_KEYS",
    )
    for attr in preferred_groups:
        value = getattr(epm_config, attr, None)
        if isinstance(value, (list, tuple, set)):
            names.update(str(item) for item in value if isinstance(item, str) and item)

    cfg = getattr(epm_config, "CFG", {})
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if "feature" not in str(key).lower():
                continue
            if not isinstance(value, (list, tuple, set)):
                continue
            for item in value:
                if not isinstance(item, str) or not item:
                    continue
                lower = item.lower()
                if any(token in lower for token in LIQUIDITY_EXECUTION_FEATURE_TOKENS):
                    names.add(item)

    out = []
    for name in sorted(names):
        if _is_feature_artifact(name, include_diagnostic_features=False):
            continue
        lower = name.lower()
        if lower.endswith("_features") or "ev_spread" in lower:
            continue
        if any(token in lower for token in LIQUIDITY_EXECUTION_FEATURE_TOKENS):
            out.append(name)
    return out


def _is_feature_artifact(name: str, *, include_diagnostic_features: bool) -> bool:
    lower = str(name).lower()
    if lower in ID_COLUMNS or lower in LABEL_TARGET_COLUMNS or lower in PREDICTION_OR_MODEL_COLUMNS:
        return True
    if lower.startswith(LEARNED_OR_ARTIFACT_PREFIXES):
        return True
    if not include_diagnostic_features and lower.startswith(DIAGNOSTIC_PREFIXES):
        return True
    if "archetype" in lower or "distillation" in lower:
        return True
    if lower.endswith("_id") or lower.endswith("_source"):
        return True
    return False


def _looks_like_composite_feature(name: str) -> bool:
    lower = str(name).lower()
    return any(hint in lower for hint in FEATURE_NAME_HINTS)


def _numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=False)


def _dominant_fraction(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    if values.size > 20_000:
        rng = np.random.default_rng(123)
        values = values[rng.choice(values.size, size=20_000, replace=False)]
    rounded = np.round(values.astype(np.float64), 8)
    _, counts = np.unique(rounded, return_counts=True)
    return float(counts.max() / max(len(rounded), 1))


def _select_candidate_features(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    cfg_features: set[str],
) -> tuple[list[str], pd.DataFrame]:
    numeric_cols = list(frame.select_dtypes(include=["number", "bool"]).columns)
    rows: list[dict[str, object]] = []
    candidates: list[str] = []
    for col in numeric_cols:
        if _is_feature_artifact(col, include_diagnostic_features=config.include_diagnostic_features):
            continue
        if cfg_features and col not in cfg_features and not _looks_like_composite_feature(col):
            continue
        arr = _numeric_array(frame[col])
        finite = np.isfinite(arr)
        coverage = float(finite.mean()) if len(arr) else 0.0
        if coverage < float(config.min_feature_coverage):
            continue
        finite_arr = arr[finite]
        if finite_arr.size < int(config.min_window_rows):
            continue
        unique_count = int(min(np.unique(finite_arr[: min(finite_arr.size, 50_000)]).size, 50_000))
        dominant = _dominant_fraction(finite_arr)
        if unique_count < int(config.min_unique_values) or dominant > float(config.max_dominant_fraction):
            continue
        q25, q75 = np.nanpercentile(finite_arr, [25, 75])
        robust_scale = float(max(q75 - q25, np.nanmedian(np.abs(finite_arr - np.nanmedian(finite_arr))) * 1.4826, 1e-12))
        rows.append(
            {
                "feature": col,
                "coverage": coverage,
                "unique_count": unique_count,
                "dominant_fraction": dominant,
                "robust_scale": robust_scale,
                "cfg_feature": bool(col in cfg_features),
                "composite_name_hint": bool(_looks_like_composite_feature(col)),
            }
        )
        candidates.append(col)
    quality = pd.DataFrame(rows)
    if not candidates:
        return [], quality
    quality["selection_score"] = (
        quality["coverage"].astype(float)
        * np.log1p(quality["unique_count"].astype(float))
        * np.clip(quality["robust_scale"].astype(float), 1e-6, np.inf)
    )
    quality = quality.sort_values(
        ["composite_name_hint", "cfg_feature", "selection_score", "coverage"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    selected = quality["feature"].head(int(config.max_features)).astype(str).tolist()
    return selected, quality


def _rank_matrix_for_spearman(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32, copy=False)
    ranks = pd.DataFrame(mat).rank(axis=0, method="average", pct=True).to_numpy(
        dtype=np.float32,
        copy=False,
    )
    med = np.nanmedian(ranks, axis=0).astype(np.float32, copy=False)
    med[~np.isfinite(med)] = 0.5
    missing = ~np.isfinite(ranks)
    if bool(missing.any()):
        rows, cols = np.where(missing)
        ranks[rows, cols] = med[cols]
    ranks -= np.nanmean(ranks, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    scale = np.nanstd(ranks, axis=0).astype(np.float32, copy=False)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0).astype(
        np.float32,
        copy=False,
    )
    ranks /= scale
    return ranks.astype(np.float32, copy=False)


def _spearman_redundancy_filter(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_quality: pd.DataFrame,
    *,
    layer: str,
    strategy: str,
    config: AnalysisConfig,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    features = [str(col) for col in feature_cols if str(col) in frame.columns]
    if (
        not bool(config.redundancy_filter_enabled)
        or len(features) < 2
        or int(config.redundancy_max_rows) <= 1
    ):
        quality = feature_quality.copy()
        if not quality.empty and "feature" in quality:
            quality["redundancy_kept"] = quality["feature"].astype(str).isin(features)
            quality["redundancy_representative"] = quality["feature"].where(
                quality["redundancy_kept"],
                "",
            )
            quality["redundancy_abs_spearman_to_representative"] = np.nan
            quality["redundancy_cluster_size"] = np.where(
                quality["redundancy_kept"],
                1,
                0,
            )
        return features, quality, pd.DataFrame()

    n_rows = len(frame)
    if n_rows == 0:
        return features, feature_quality, pd.DataFrame()
    max_rows = int(max(2, config.redundancy_max_rows))
    if n_rows > max_rows:
        seed_payload = {
            "layer": layer,
            "strategy": strategy,
            "artifact_run_id": config.artifact_run_id,
            "random_seed": int(config.random_seed),
        }
        seed = int(_stable_hash_payload(seed_payload, digest_size=4), 16) % (2**32)
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    else:
        sample_idx = np.arange(n_rows, dtype=np.int64)

    values = np.empty((len(sample_idx), len(features)), dtype=np.float32)
    for j, feature in enumerate(features):
        values[:, j] = pd.to_numeric(frame[feature].iloc[sample_idx], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=False,
        )
    ranks = _rank_matrix_for_spearman(values)
    corr = np.corrcoef(ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 1.0)

    threshold = float(np.clip(config.redundancy_abs_spearman_threshold, 0.0, 1.0))
    adjacency = abs_corr >= threshold
    quality_scores = (
        feature_quality.set_index("feature")["selection_score"].astype(float).to_dict()
        if not feature_quality.empty and {"feature", "selection_score"}.issubset(feature_quality.columns)
        else {}
    )
    original_rank = {feature: rank for rank, feature in enumerate(features)}
    visited = np.zeros(len(features), dtype=bool)
    keep_features: list[str] = []
    rows: list[dict[str, object]] = []

    for start in range(len(features)):
        if visited[start]:
            continue
        stack = [start]
        component: list[int] = []
        visited[start] = True
        while stack:
            idx = stack.pop()
            component.append(idx)
            neighbors = np.flatnonzero(adjacency[idx] & ~visited)
            for nb in neighbors:
                visited[nb] = True
                stack.append(int(nb))
        component_features = [features[idx] for idx in component]
        representative = max(
            component_features,
            key=lambda name: (
                float(quality_scores.get(name, -np.inf)),
                -int(original_rank.get(name, 10**9)),
            ),
        )
        rep_idx = original_rank[representative]
        keep_features.append(representative)
        cluster_id = len(keep_features) - 1
        cluster_size = len(component_features)
        for member in component_features:
            member_idx = original_rank[member]
            rows.append(
                {
                    "layer": layer,
                    "strategy": strategy,
                    "redundancy_cluster_id": cluster_id,
                    "feature": member,
                    "representative_feature": representative,
                    "dropped_for_redundancy": bool(member != representative),
                    "cluster_size": cluster_size,
                    "abs_spearman_to_representative": float(abs_corr[member_idx, rep_idx]),
                    "selection_score": float(quality_scores.get(member, np.nan)),
                    "sample_rows": int(len(sample_idx)),
                    "threshold": threshold,
                }
            )

    keep_order = {feature: idx for idx, feature in enumerate(features)}
    keep_features = sorted(keep_features, key=lambda name: keep_order.get(name, 10**9))
    redundancy = pd.DataFrame(rows)
    quality = feature_quality.copy()
    if not quality.empty and "feature" in quality:
        representative_map = (
            redundancy.set_index("feature")["representative_feature"].astype(str).to_dict()
            if not redundancy.empty
            else {}
        )
        corr_map = (
            redundancy.set_index("feature")["abs_spearman_to_representative"].astype(float).to_dict()
            if not redundancy.empty
            else {}
        )
        size_map = (
            redundancy.set_index("feature")["cluster_size"].astype(int).to_dict()
            if not redundancy.empty
            else {}
        )
        kept_set = set(keep_features)
        feature_names = quality["feature"].astype(str)
        quality["redundancy_kept"] = feature_names.isin(kept_set)
        quality["redundancy_representative"] = feature_names.map(representative_map).fillna("")
        quality["redundancy_abs_spearman_to_representative"] = feature_names.map(corr_map)
        quality["redundancy_cluster_size"] = feature_names.map(size_map).fillna(0).astype(int)

    return keep_features, quality, redundancy


def _rank_slice_mask(
    frame: pd.DataFrame,
    pred_col: str,
    *,
    rank_frac: float,
) -> np.ndarray:
    if rank_frac <= 0.0 or rank_frac >= 1.0:
        return np.ones(len(frame), dtype=bool)
    work = frame[["timestamp", pred_col]].copy()
    pred = pd.to_numeric(work[pred_col], errors="coerce")
    ranks = pred.groupby(work["timestamp"]).rank(method="first", pct=True)
    return ranks.ge(1.0 - float(rank_frac)).fillna(False).to_numpy(dtype=bool)


def _analysis_slices(frame: pd.DataFrame, pred_col: str, config: AnalysisConfig) -> list[tuple[str, np.ndarray]]:
    top_name = f"top{int(round(float(config.rank_frac) * 100))}"
    top_mask = _rank_slice_mask(frame, pred_col, rank_frac=float(config.rank_frac))
    if bool(config.top_rank_slice_only):
        return [(top_name, top_mask)]
    slices: list[tuple[str, np.ndarray]] = [("all", np.ones(len(frame), dtype=bool))]
    if top_mask.any() and not np.array_equal(top_mask, slices[0][1]):
        slices.append((top_name, top_mask))
    return slices


def _slice_support_summary(
    frame: pd.DataFrame,
    *,
    strategy: str,
    layer: str,
    slice_name: str,
    pred_col: str,
    label_col: str,
    mask: np.ndarray,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    pred = pd.to_numeric(frame[pred_col], errors="coerce") if pred_col in frame.columns else pd.Series(np.nan, index=frame.index)
    label = pd.to_numeric(frame[label_col], errors="coerce") if label_col in frame.columns else pd.Series(np.nan, index=frame.index)
    valid = (
        ts.notna().to_numpy(dtype=bool)
        & mask
        & pred.notna().to_numpy(dtype=bool)
        & label.notna().to_numpy(dtype=bool)
    )
    if not bool(valid.any()):
        return pd.DataFrame(
            [
                {
                    "strategy": strategy,
                    "layer": layer,
                    "slice": slice_name,
                    "support_days": 0,
                    "eligible_support_days": 0,
                    "support_rows": 0,
                    "mean_rows_per_support_day": np.nan,
                    "median_rows_per_support_day": np.nan,
                    "min_rows_per_day_threshold": float(max(config.min_window_rows_per_day, 1.0)),
                    "eligible_support_day_list": "",
                }
            ]
        )
    days = ts.loc[valid].dt.floor("D")
    day_counts = days.value_counts(sort=False)
    min_rows_per_day = max(float(config.min_window_rows_per_day), 1.0)
    eligible = day_counts.loc[day_counts.ge(min_rows_per_day)]
    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "layer": layer,
                "slice": slice_name,
                "support_days": int(day_counts.size),
                "eligible_support_days": int(eligible.size),
                "support_rows": int(day_counts.sum()),
                "mean_rows_per_support_day": float(day_counts.mean()) if day_counts.size else np.nan,
                "median_rows_per_support_day": float(day_counts.median()) if day_counts.size else np.nan,
                "min_rows_per_day_threshold": float(min_rows_per_day),
                "support_start_day": str(day_counts.index.min()) if day_counts.size else "",
                "support_end_day": str(day_counts.index.max()) if day_counts.size else "",
                "eligible_support_day_list": "|".join(
                    day.strftime("%Y-%m-%d") for day in sorted(eligible.index)
                ),
            }
        ]
    )


def _daily_performance(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    label_col: str,
    pnl_col: str,
    mask: np.ndarray,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if not len(frame):
        return pd.DataFrame()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    valid = ts.notna().to_numpy(dtype=bool) & mask
    if not valid.any():
        return pd.DataFrame()
    pred = pd.to_numeric(frame[pred_col], errors="coerce").clip(1e-5, 1.0 - 1e-5)
    y = pd.to_numeric(frame[label_col], errors="coerce")
    pnl = (
        pd.to_numeric(frame[pnl_col], errors="coerce")
        if pnl_col and pnl_col in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    valid &= pred.notna().to_numpy(dtype=bool) & y.notna().to_numpy(dtype=bool)
    if not valid.any():
        return pd.DataFrame()
    work = pd.DataFrame(
        {
            "day": ts.dt.floor("D"),
            "pred": pred.astype(float),
            "y": y.astype(float),
            "pnl": pnl.astype(float),
        }
    ).loc[valid]
    work["var"] = np.clip(work["pred"].to_numpy() * (1.0 - work["pred"].to_numpy()), 1e-6, np.inf)
    daily = work.groupby("day", sort=True).agg(
        n=("y", "size"),
        hits=("y", "sum"),
        expected_hits=("pred", "sum"),
        variance=("var", "sum"),
        pnl_sum=("pnl", "sum"),
        pnl_mean=("pnl", "mean"),
        mean_pred=("pred", "mean"),
        actual_hit_rate=("y", "mean"),
    )
    if daily.empty:
        return daily
    index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(index)
    fill_zero = ["n", "hits", "expected_hits", "variance", "pnl_sum"]
    daily[fill_zero] = daily[fill_zero].fillna(0.0)
    daily["pnl_mean"] = daily["pnl_mean"].astype(float)
    daily["mean_pred"] = daily["mean_pred"].astype(float)
    daily["actual_hit_rate"] = daily["actual_hit_rate"].astype(float)
    win = max(1, int(config.window_days))
    rolling = pd.DataFrame(index=daily.index)
    for col in fill_zero:
        rolling[col] = daily[col].rolling(win, min_periods=win).sum()
    rolling["start_day"] = rolling.index - pd.Timedelta(days=win - 1)
    rolling["end_day"] = rolling.index
    rolling["actual_hit_rate"] = rolling["hits"] / rolling["n"].replace(0.0, np.nan)
    rolling["expected_hit_rate"] = rolling["expected_hits"] / rolling["n"].replace(0.0, np.nan)
    rolling["hit_rate_delta"] = rolling["actual_hit_rate"] - rolling["expected_hit_rate"]
    rolling["hit_rate_surprise"] = rolling["hits"] - rolling["expected_hits"]
    rolling["hit_rate_surprise_z"] = rolling["hit_rate_surprise"] / np.sqrt(
        rolling["variance"].clip(lower=1e-6)
    )
    rolling["mean_pnl"] = rolling["pnl_sum"] / rolling["n"].replace(0.0, np.nan)
    rolling = rolling.reset_index(names="window_end_day")
    rolling["window_days"] = win
    rolling["surprise_z_threshold"] = float(config.surprise_z_threshold)
    rolling["hit_rate_delta_threshold"] = float(config.hit_rate_delta_threshold)
    return rolling


def _effective_min_window_rows(config: AnalysisConfig) -> float:
    absolute = max(float(config.min_window_rows), 0.0)
    per_day = max(float(config.min_window_rows_per_day), 0.0) * max(float(config.window_days), 1.0)
    if absolute > 0.0 and per_day > 0.0:
        return max(absolute, per_day)
    return max(absolute, per_day, 1.0)


def _detect_bad_windows(
    rolling: pd.DataFrame,
    *,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if rolling.empty:
        return rolling
    min_rows = _effective_min_window_rows(config)
    mask = (
        rolling["n"].ge(float(min_rows))
        & rolling["hit_rate_surprise_z"].le(float(config.surprise_z_threshold))
        & rolling["hit_rate_delta"].le(float(config.hit_rate_delta_threshold))
    )
    out = rolling.loc[mask].copy()
    if config.min_episode_end_day:
        min_end = pd.Timestamp(config.min_episode_end_day)
        if min_end.tzinfo is None:
            min_end = min_end.tz_localize("UTC")
        out = out.loc[pd.to_datetime(out["end_day"], utc=True, errors="coerce").ge(min_end)]
    return out.sort_values(["start_day", "end_day"], kind="mergesort")


def _eligible_support_day_set(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    label_col: str,
    mask: np.ndarray,
    config: AnalysisConfig,
) -> set[pd.Timestamp]:
    if frame.empty or "timestamp" not in frame.columns:
        return set()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    pred = pd.to_numeric(frame[pred_col], errors="coerce") if pred_col in frame.columns else pd.Series(np.nan, index=frame.index)
    label = pd.to_numeric(frame[label_col], errors="coerce") if label_col in frame.columns else pd.Series(np.nan, index=frame.index)
    valid = (
        ts.notna().to_numpy(dtype=bool)
        & mask
        & pred.notna().to_numpy(dtype=bool)
        & label.notna().to_numpy(dtype=bool)
    )
    if not bool(valid.any()):
        return set()
    days = ts.loc[valid].dt.floor("D")
    day_counts = days.value_counts(sort=False)
    min_rows_per_day = max(float(config.min_window_rows_per_day), 1.0)
    eligible = day_counts.loc[day_counts.ge(min_rows_per_day)]
    return {pd.Timestamp(day).floor("D") for day in eligible.index}


def _eligible_rolling_mask(rolling: pd.DataFrame, config: AnalysisConfig) -> pd.Series:
    if rolling.empty:
        return pd.Series(False, index=rolling.index)
    min_rows = pd.to_numeric(rolling.get("window_days", config.window_days), errors="coerce").fillna(config.window_days)
    min_rows = min_rows.map(
        lambda window_days: max(
            float(config.min_window_rows),
            float(config.min_window_rows_per_day) * max(float(window_days), 1.0),
            1.0,
        )
    )
    mask = (
        pd.to_numeric(rolling.get("n", np.nan), errors="coerce").ge(min_rows)
        & pd.to_numeric(rolling.get("hit_rate_surprise_z", np.nan), errors="coerce").lt(0.0)
        & pd.to_numeric(rolling.get("hit_rate_delta", np.nan), errors="coerce").lt(0.0)
    )
    if config.min_episode_end_day:
        min_end = pd.Timestamp(config.min_episode_end_day)
        if min_end.tzinfo is None:
            min_end = min_end.tz_localize("UTC")
        else:
            min_end = min_end.tz_convert("UTC")
        mask &= pd.to_datetime(rolling.get("end_day"), utc=True, errors="coerce").ge(min_end)
    return mask.fillna(False)


def _bad_window_day_share(rolling: pd.DataFrame, mask: pd.Series | np.ndarray, eligible_days: set[pd.Timestamp]) -> float:
    if rolling.empty or not eligible_days:
        return 0.0
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.size != len(rolling) or not bool(mask_arr.any()):
        return 0.0
    covered = _covered_days_set(rolling.loc[mask_arr]) & eligible_days
    return float(len(covered) / max(len(eligible_days), 1))


def _calibrated_threshold_mask(
    rolling: pd.DataFrame,
    *,
    surprise_z_threshold: float,
    hit_rate_delta_threshold: float,
    config: AnalysisConfig,
) -> pd.Series:
    eligible = _eligible_rolling_mask(rolling, config)
    z = pd.to_numeric(rolling["hit_rate_surprise_z"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    delta = pd.to_numeric(rolling["hit_rate_delta"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return pd.Series(
        eligible.to_numpy(dtype=bool, copy=False)
        & np.isfinite(z)
        & np.isfinite(delta)
        & (z <= float(surprise_z_threshold))
        & (delta <= float(hit_rate_delta_threshold)),
        index=rolling.index,
    )


def _calibrate_bad_window_thresholds(
    rolling: pd.DataFrame,
    *,
    eligible_days: set[pd.Timestamp],
    config: AnalysisConfig,
) -> tuple[dict[str, float | str | int], pd.Series]:
    empty_mask = pd.Series(False, index=rolling.index)
    target = float(np.clip(config.target_bad_day_share, 0.0, 1.0))
    eligible = _eligible_rolling_mask(rolling, config)
    pool = rolling.loc[eligible].copy()
    if pool.empty or not eligible_days or target <= 0.0:
        diagnostics: dict[str, float | str | int] = {
            "calibration_status": "insufficient_windows",
            "calibration_target_bad_day_share": target,
            "calibration_realized_bad_day_share": 0.0,
            "calibration_candidate_windows": int(len(pool)),
            "calibrated_surprise_z_threshold": np.nan,
            "calibrated_hit_rate_delta_threshold": np.nan,
        }
        return diagnostics, empty_mask

    z = pd.to_numeric(pool["hit_rate_surprise_z"], errors="coerce")
    delta = pd.to_numeric(pool["hit_rate_delta"], errors="coerce")
    z_values = z.loc[z.lt(0.0) & delta.lt(0.0) & z.notna() & delta.notna()].to_numpy(dtype=np.float64, copy=False)
    delta_values = delta.loc[z.lt(0.0) & delta.lt(0.0) & z.notna() & delta.notna()].to_numpy(dtype=np.float64, copy=False)
    if z_values.size == 0 or delta_values.size == 0:
        diagnostics = {
            "calibration_status": "no_negative_underperformance_windows",
            "calibration_target_bad_day_share": target,
            "calibration_realized_bad_day_share": 0.0,
            "calibration_candidate_windows": int(len(pool)),
            "calibrated_surprise_z_threshold": np.nan,
            "calibrated_hit_rate_delta_threshold": np.nan,
        }
        return diagnostics, empty_mask

    grid_size = int(np.clip(config.bad_window_calibration_grid_size, 8, 200))
    quantiles = np.linspace(0.0, 1.0, grid_size)
    z_candidates = sorted(set(float(v) for v in np.quantile(z_values, quantiles) if np.isfinite(v)))
    delta_candidates = sorted(set(float(v) for v in np.quantile(delta_values, quantiles) if np.isfinite(v)))
    if not z_candidates or not delta_candidates:
        diagnostics = {
            "calibration_status": "empty_candidate_grid",
            "calibration_target_bad_day_share": target,
            "calibration_realized_bad_day_share": 0.0,
            "calibration_candidate_windows": int(len(pool)),
            "calibrated_surprise_z_threshold": np.nan,
            "calibrated_hit_rate_delta_threshold": np.nan,
        }
        return diagnostics, empty_mask

    best: dict[str, float | str | int] | None = None
    best_mask = empty_mask
    for z_thr in z_candidates:
        for delta_thr in delta_candidates:
            candidate_mask = _calibrated_threshold_mask(
                rolling,
                surprise_z_threshold=z_thr,
                hit_rate_delta_threshold=delta_thr,
                config=config,
            )
            share = _bad_window_day_share(rolling, candidate_mask, eligible_days)
            selected_count = int(candidate_mask.sum())
            if selected_count <= 0:
                continue
            under_target = share <= target + 1e-12
            score = (
                0 if under_target else 1,
                abs(target - share),
                -share if under_target else share,
                selected_count,
            )
            if best is None or score < best["_selection_score"]:  # type: ignore[index]
                best = {
                    "_selection_score": score,
                    "calibration_status": "ok",
                    "calibration_target_bad_day_share": target,
                    "calibration_realized_bad_day_share": float(share),
                    "calibration_candidate_windows": int(len(pool)),
                    "calibration_selected_windows": selected_count,
                    "calibrated_surprise_z_threshold": float(z_thr),
                    "calibrated_hit_rate_delta_threshold": float(delta_thr),
                }
                best_mask = candidate_mask

    if best is None:
        diagnostics = {
            "calibration_status": "no_nonempty_threshold_pair",
            "calibration_target_bad_day_share": target,
            "calibration_realized_bad_day_share": 0.0,
            "calibration_candidate_windows": int(len(pool)),
            "calibrated_surprise_z_threshold": np.nan,
            "calibrated_hit_rate_delta_threshold": np.nan,
        }
        return diagnostics, empty_mask
    best.pop("_selection_score", None)
    return best, best_mask


def _window_effective_min_rows(row: pd.Series, config: AnalysisConfig) -> float:
    window_days = max(_safe_float(row.get("window_days"), float(config.window_days)), 1.0)
    absolute = max(float(config.min_window_rows), 0.0)
    per_day = max(float(config.min_window_rows_per_day), 0.0) * window_days
    if absolute > 0.0 and per_day > 0.0:
        return max(absolute, per_day)
    return max(absolute, per_day, 1.0)


def _bad_window_severity(row: pd.Series, config: AnalysisConfig) -> dict[str, float]:
    z_threshold = abs(_safe_float(row.get("surprise_z_threshold"), float(config.surprise_z_threshold)))
    delta_threshold = abs(_safe_float(row.get("hit_rate_delta_threshold"), float(config.hit_rate_delta_threshold)))
    window_days = max(_safe_float(row.get("window_days"), float(config.window_days)), 1.0)
    z_depth = max(0.0, -_safe_float(row.get("hit_rate_surprise_z"), 0.0) / max(z_threshold, 1e-6))
    delta_depth = max(0.0, -_safe_float(row.get("hit_rate_delta"), 0.0) / max(delta_threshold, 1e-6))
    rows = max(_safe_float(row.get("n"), 0.0), 1.0)
    row_component = float(np.clip(math.sqrt(rows / _window_effective_min_rows(row, config)), 1.0, 4.0))
    duration_component = float(np.clip(math.sqrt(window_days), 1.0, 3.0))
    depth_component = float(np.clip(0.65 * z_depth + 0.35 * delta_depth, 0.1, 8.0))
    severity = float(duration_component * row_component * depth_component)
    return {
        "window_z_depth": float(z_depth),
        "window_hit_rate_delta_depth": float(delta_depth),
        "window_row_support_ratio": float(rows / _window_effective_min_rows(row, config)),
        "window_severity": severity,
    }


def _add_bad_window_severity(bad_windows: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    if bad_windows.empty:
        return bad_windows
    out = bad_windows.copy()
    rows = [_bad_window_severity(row, config) for _, row in out.iterrows()]
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _detect_bad_windows_for_slice(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    label_col: str,
    pnl_col: str,
    mask: np.ndarray,
    config: AnalysisConfig,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    configs = [config]
    if int(config.secondary_window_days) > 0:
        configs.append(
            replace(
                config,
                window_days=int(config.secondary_window_days),
                surprise_z_threshold=float(config.secondary_surprise_z_threshold),
                hit_rate_delta_threshold=float(config.secondary_hit_rate_delta_threshold),
            )
        )
    rolling_parts: list[pd.DataFrame] = []
    for local_config in configs:
        rolling = _daily_performance(
            frame,
            pred_col=pred_col,
            label_col=label_col,
            pnl_col=pnl_col,
            mask=mask,
            config=local_config,
        )
        if not rolling.empty:
            rolling_parts.append(rolling)
    if not rolling_parts:
        return pd.DataFrame()

    calibration = pd.DataFrame()
    if bool(config.bad_window_calibration_enabled):
        rolling_all = pd.concat(rolling_parts, ignore_index=True)
        rolling_all = rolling_all.sort_values(
            ["start_day", "end_day", "window_days", "hit_rate_surprise_z"],
            kind="mergesort",
        ).drop_duplicates(["start_day", "end_day", "window_days"], keep="first")
        eligible_days = _eligible_support_day_set(
            frame,
            pred_col=pred_col,
            label_col=label_col,
            mask=mask,
            config=config,
        )
        diagnostics, calibrated_mask = _calibrate_bad_window_thresholds(
            rolling_all,
            eligible_days=eligible_days,
            config=config,
        )
        out = rolling_all.loc[calibrated_mask].copy()
        if not out.empty:
            out["default_surprise_z_threshold"] = pd.to_numeric(
                out.get("surprise_z_threshold", config.surprise_z_threshold),
                errors="coerce",
            ).fillna(config.surprise_z_threshold)
            out["default_hit_rate_delta_threshold"] = pd.to_numeric(
                out.get("hit_rate_delta_threshold", config.hit_rate_delta_threshold),
                errors="coerce",
            ).fillna(config.hit_rate_delta_threshold)
            z_thr = float(diagnostics.get("calibrated_surprise_z_threshold", config.surprise_z_threshold))
            delta_thr = float(diagnostics.get("calibrated_hit_rate_delta_threshold", config.hit_rate_delta_threshold))
            out["calibrated_surprise_z_threshold"] = z_thr
            out["calibrated_hit_rate_delta_threshold"] = delta_thr
            out["surprise_z_threshold"] = z_thr
            out["hit_rate_delta_threshold"] = delta_thr
            out["bad_window_detection_mode"] = "calibrated"
        diagnostics = dict(diagnostics)
        diagnostics["calibration_eligible_support_days"] = int(len(eligible_days))
        calibration = pd.DataFrame([diagnostics])
    else:
        bad_parts: list[pd.DataFrame] = []
        for rolling, local_config in zip(rolling_parts, configs):
            bad = _detect_bad_windows(rolling, config=local_config)
            if not bad.empty:
                bad["bad_window_detection_mode"] = "fixed"
                bad_parts.append(bad)
        if not bad_parts:
            return pd.DataFrame()
        out = pd.concat(bad_parts, ignore_index=True)

    if out.empty:
        if bool(config.bad_window_calibration_enabled):
            return out, calibration
        return out
    out = out.sort_values(
        ["start_day", "end_day", "window_days", "hit_rate_surprise_z"],
        kind="mergesort",
    )
    out = out.drop_duplicates(
        ["start_day", "end_day", "window_days"],
        keep="first",
    ).reset_index(drop=True)
    out = _add_bad_window_severity(out, config)
    if bool(config.bad_window_calibration_enabled):
        return out, calibration
    return out


def _merge_bad_windows(
    bad_windows: pd.DataFrame,
    *,
    strategy: str,
    layer: str,
    slice_name: str,
) -> pd.DataFrame:
    if bad_windows.empty:
        return pd.DataFrame()
    windows = bad_windows.sort_values(["start_day", "end_day"], kind="mergesort")
    episodes: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for row in windows.to_dict("records"):
        start = pd.Timestamp(row["start_day"])
        end = pd.Timestamp(row["end_day"])
        if current is None:
            current = {
                "strategy": strategy,
                "layer": layer,
                "slice": slice_name,
                "episode_id": 0,
                "start_day": start,
                "end_day": end,
                "window_count": 1,
                "min_hit_rate_surprise_z": _safe_float(row.get("hit_rate_surprise_z")),
                "min_hit_rate_delta": _safe_float(row.get("hit_rate_delta")),
                "total_rows_in_bad_windows": _safe_float(row.get("n"), 0.0),
                "window_severity_sum": _safe_float(row.get("window_severity"), 0.0),
                "max_window_severity": _safe_float(row.get("window_severity"), 0.0),
            }
            continue
        if start <= pd.Timestamp(current["end_day"]) + pd.Timedelta(days=1):
            current["end_day"] = max(pd.Timestamp(current["end_day"]), end)
            current["window_count"] = int(current["window_count"]) + 1
            current["min_hit_rate_surprise_z"] = min(
                _safe_float(current["min_hit_rate_surprise_z"]),
                _safe_float(row.get("hit_rate_surprise_z")),
            )
            current["min_hit_rate_delta"] = min(
                _safe_float(current["min_hit_rate_delta"]),
                _safe_float(row.get("hit_rate_delta")),
            )
            current["total_rows_in_bad_windows"] = _safe_float(current["total_rows_in_bad_windows"], 0.0) + _safe_float(
                row.get("n"),
                0.0,
            )
            current["window_severity_sum"] = _safe_float(current.get("window_severity_sum"), 0.0) + _safe_float(
                row.get("window_severity"),
                0.0,
            )
            current["max_window_severity"] = max(
                _safe_float(current.get("max_window_severity"), 0.0),
                _safe_float(row.get("window_severity"), 0.0),
            )
        else:
            current["episode_id"] = len(episodes)
            episodes.append(current)
            current = {
                "strategy": strategy,
                "layer": layer,
                "slice": slice_name,
                "episode_id": len(episodes),
                "start_day": start,
                "end_day": end,
                "window_count": 1,
                "min_hit_rate_surprise_z": _safe_float(row.get("hit_rate_surprise_z")),
                "min_hit_rate_delta": _safe_float(row.get("hit_rate_delta")),
                "total_rows_in_bad_windows": _safe_float(row.get("n"), 0.0),
                "window_severity_sum": _safe_float(row.get("window_severity"), 0.0),
                "max_window_severity": _safe_float(row.get("window_severity"), 0.0),
            }
    if current is not None:
        current["episode_id"] = len(episodes)
        episodes.append(current)
    return pd.DataFrame(episodes)


def _collect_breakout_episodes_for_generation(
    files: Sequence[tuple[str, str, Path]],
    *,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Detect performance-break episodes before feature generation.

    This is intentionally prediction-only. It gives the exploratory operator
    stage a set of bad periods without using generated regime features.
    """

    parts: list[pd.DataFrame] = []
    for layer, strategy, path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            _log(f"[breakout-screen] skipped {layer} {strategy}: {type(exc).__name__}: {exc}")
            continue
        if "timestamp" not in frame.columns or "symbol" not in frame.columns:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["symbol"] = frame["symbol"].astype(str)
        frame = _filter_frame_by_analysis_period(frame, config)
        if frame.empty:
            continue
        try:
            pred_col, label_col, pnl_col = _prediction_columns(frame, layer)
        except Exception as exc:
            _log(f"[breakout-screen] skipped {layer} {strategy}: {type(exc).__name__}: {exc}")
            continue
        for slice_name, mask in _analysis_slices(frame, pred_col, config):
            detected = _detect_bad_windows_for_slice(
                frame,
                pred_col=pred_col,
                label_col=label_col,
                pnl_col=pnl_col,
                mask=mask,
                config=config,
            )
            bad = detected[0] if isinstance(detected, tuple) else detected
            episodes = _merge_bad_windows(bad, strategy=strategy, layer=layer, slice_name=slice_name)
            if episodes.empty:
                continue
            parts.append(episodes)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if not out.empty:
        out["breakout_weight"] = [
            _episode_breakout_weight(row, config)
            for _, row in out.iterrows()
        ]
    _log(
        f"[breakout-screen] detected episodes for operator generation: "
        f"episodes={len(out)} files={len(files)}"
    )
    return out


def _sample_positions(mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    pos = np.flatnonzero(mask)
    if len(pos) <= max_rows:
        return pos
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pos, size=max_rows, replace=False))


def _baseline_episode_sample_limits(config: AnalysisConfig) -> tuple[int, int]:
    global_cap = max(1, int(config.max_rows_per_side))
    baseline_cap = min(global_cap, max(1, int(config.baseline_max_rows_per_episode)))
    episode_cap = min(global_cap, max(1, int(config.episode_max_rows_per_episode)))
    return baseline_cap, episode_cap


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(finite))
    q25, q75 = np.nanpercentile(finite, [25, 75])
    iqr = float(q75 - q25)
    mad = float(np.nanmedian(np.abs(finite - center)) * 1.4826)
    scale = max(iqr, mad, 1e-8)
    return center, scale


def _safe_corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = True) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 30:
        return 0.0
    xs = x[mask].astype(np.float64, copy=False)
    ys = y[mask].astype(np.float64, copy=False)
    if spearman:
        xs = pd.Series(xs).rank(method="average").to_numpy(dtype=np.float64)
        ys = pd.Series(ys).rank(method="average").to_numpy(dtype=np.float64)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = float(np.sqrt(np.sum(xs * xs) * np.sum(ys * ys)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(xs * ys) / denom, -1.0, 1.0))


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 30 or b.size < 30:
        return 0.0
    if a.size > 20_000:
        rng = np.random.default_rng(11)
        a = a[rng.choice(a.size, size=20_000, replace=False)]
    if b.size > 20_000:
        rng = np.random.default_rng(17)
        b = b[rng.choice(b.size, size=20_000, replace=False)]
    if ks_2samp is not None:
        return float(ks_2samp(a, b, alternative="two-sided", mode="auto").statistic)
    grid = np.unique(np.concatenate([np.quantile(a, np.linspace(0, 1, 200)), np.quantile(b, np.linspace(0, 1, 200))]))
    if grid.size == 0:
        return 0.0
    return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a) - np.searchsorted(np.sort(b), grid, side="right") / len(b))))


def _binary_auc_lift(negative: np.ndarray, positive: np.ndarray) -> tuple[float, float]:
    negative = negative[np.isfinite(negative)]
    positive = positive[np.isfinite(positive)]
    if negative.size < 30 or positive.size < 30:
        return 0.5, 0.0
    rng = np.random.default_rng(23)
    if negative.size > 20_000:
        negative = negative[rng.choice(negative.size, size=20_000, replace=False)]
    if positive.size > 20_000:
        positive = positive[rng.choice(positive.size, size=20_000, replace=False)]
    values = np.concatenate([negative, positive]).astype(np.float64, copy=False)
    labels = np.concatenate(
        [
            np.zeros(negative.size, dtype=np.int8),
            np.ones(positive.size, dtype=np.int8),
        ]
    )
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    pos_ranks = ranks[labels == 1]
    auc = (float(pos_ranks.sum()) - positive.size * (positive.size + 1.0) / 2.0) / max(
        float(positive.size * negative.size),
        1.0,
    )
    auc = float(np.clip(auc, 0.0, 1.0))
    return auc, float(abs(auc - 0.5) * 2.0)


def _episode_duration_days(episode: pd.Series) -> int:
    start = pd.Timestamp(episode["start_day"])
    end = pd.Timestamp(episode["end_day"])
    return max(1, int((end.normalize() - start.normalize()).days) + 1)


def _episode_breakout_weight(episode: pd.Series, config: AnalysisConfig) -> float:
    duration_days = _episode_duration_days(episode)
    z_depth = max(
        0.0,
        -_safe_float(episode.get("min_hit_rate_surprise_z"), 0.0)
        / max(abs(float(config.surprise_z_threshold)), 1e-6),
    )
    delta_depth = max(
        0.0,
        -_safe_float(episode.get("min_hit_rate_delta"), 0.0)
        / max(abs(float(config.hit_rate_delta_threshold)), 1e-6),
    )
    rows = max(_safe_float(episode.get("total_rows_in_bad_windows"), 0.0), _safe_float(episode.get("episode_rows"), 0.0))
    row_component = float(np.clip(math.sqrt(max(rows, 1.0) / _effective_min_window_rows(config)), 1.0, 4.0))
    duration_component = float(np.clip(math.sqrt(float(duration_days)), 1.0, 4.0))
    depth_component = float(np.clip(0.65 * z_depth + 0.35 * delta_depth, 0.1, 6.0))
    window_component = float(np.clip(math.sqrt(max(_safe_float(episode.get("window_count"), 1.0), 1.0)), 1.0, 4.0))
    return float(duration_component * depth_component * row_component * window_component)


def _score_raw_features_against_breakouts(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Loose pre-screen for primitive features before operator generation."""

    features = [str(col) for col in dict.fromkeys(raw_features) if str(col) in panel.columns]
    if panel.empty or episodes.empty or not features:
        return pd.DataFrame()
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    day = ts.dt.floor("D")
    arrays = {feature: _numeric_array(panel[feature]) for feature in features}
    rows: list[dict[str, object]] = []
    baseline_sample_cap, episode_sample_cap = _baseline_episode_sample_limits(config)
    baseline_sample_cap = min(baseline_sample_cap, 50_000)
    for episode_idx, episode in episodes.reset_index(drop=True).iterrows():
        start = pd.Timestamp(episode["start_day"])
        end = pd.Timestamp(episode["end_day"])
        baseline_end = start - pd.Timedelta(days=max(0, int(config.embargo_days)))
        baseline_mask = day.lt(baseline_end).fillna(False).to_numpy(dtype=bool)
        episode_mask = day.ge(start).fillna(False).to_numpy(dtype=bool) & day.le(end).fillna(False).to_numpy(dtype=bool)
        baseline_pos = _sample_positions(
            baseline_mask,
            baseline_sample_cap,
            int(config.random_seed) + int(episode_idx) * 17,
        )
        episode_pos = _sample_positions(
            episode_mask,
            episode_sample_cap,
            int(config.random_seed) + int(episode_idx) * 17 + 1,
        )
        if len(baseline_pos) < 100 or len(episode_pos) < 100:
            continue
        breakout_weight = _episode_breakout_weight(episode, config)
        for feature in features:
            arr = arrays[feature]
            base = arr[baseline_pos]
            bad = arr[episode_pos]
            base = base[np.isfinite(base)]
            bad = bad[np.isfinite(bad)]
            if base.size < 100 or bad.size < 50:
                continue
            center, scale = _robust_center_scale(base)
            base_z = np.clip((base - center) / scale, -8.0, 8.0)
            bad_z = np.clip((bad - center) / scale, -8.0, 8.0)
            ks = _ks_stat(base_z, bad_z)
            auc, auc_lift = _binary_auc_lift(base_z, bad_z)
            med_shift = float(np.nanmedian(bad_z) - np.nanmedian(base_z))
            q25_b, q75_b = np.nanpercentile(base_z, [25, 75])
            q25_e, q75_e = np.nanpercentile(bad_z, [25, 75])
            iqr_ratio_log = float(np.log((q75_e - q25_e + 1e-6) / (q75_b - q25_b + 1e-6)))
            link_score = _clip01(
                0.40 * (ks / 0.25)
                + 0.35 * (auc_lift / 0.25)
                + 0.20 * (abs(med_shift) / 1.00)
                + 0.05 * (abs(iqr_ratio_log) / 0.75)
            )
            pass_loose = bool(
                link_score >= float(config.raw_exploration_min_score)
                or ks >= 0.05
                or auc_lift >= 0.05
                or abs(med_shift) >= 0.10
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "strategy": episode.get("strategy", ""),
                    "layer": episode.get("layer", ""),
                    "slice": episode.get("slice", ""),
                    "episode_id": int(_safe_float(episode.get("episode_id"), 0.0)),
                    "episode_start_day": start,
                    "episode_end_day": end,
                    "episode_duration_days": _episode_duration_days(episode),
                    "episode_min_hit_rate_surprise_z": _safe_float(episode.get("min_hit_rate_surprise_z")),
                    "episode_min_hit_rate_delta": _safe_float(episode.get("min_hit_rate_delta")),
                    "episode_breakout_weight": breakout_weight,
                    "ks_shift": ks,
                    "period_auc": auc,
                    "period_auc_lift": auc_lift,
                    "median_shift_robust_z": med_shift,
                    "iqr_ratio_log": iqr_ratio_log,
                    "raw_breakout_link_score": link_score,
                    "raw_exploration_pass": pass_loose,
                    "weighted_raw_breakout_link_score": link_score * breakout_weight,
                }
            )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    agg_rows: list[dict[str, object]] = []
    for feature, group in detail.groupby("feature", sort=False):
        total_weight = float(pd.to_numeric(group["episode_breakout_weight"], errors="coerce").fillna(0.0).sum())
        passed = group.loc[group["raw_exploration_pass"].astype(bool)]
        passed_weight = float(pd.to_numeric(passed["episode_breakout_weight"], errors="coerce").fillna(0.0).sum())
        weighted_score = float(pd.to_numeric(group["weighted_raw_breakout_link_score"], errors="coerce").fillna(0.0).sum())
        raw_score = weighted_score / max(total_weight, 1e-12)
        pass_count = int(passed[["strategy", "layer", "slice", "episode_id"]].drop_duplicates().shape[0]) if not passed.empty else 0
        agg_rows.append(
            {
                "feature": feature,
                "feature_family": _feature_family(feature),
                "portability_kind": _feature_portability(feature)[0],
                "portability_reason": _feature_portability(feature)[1],
                "raw_breakout_link_score": raw_score,
                "weighted_raw_breakout_link_score": weighted_score,
                "episode_count_scored": int(group[["strategy", "layer", "slice", "episode_id"]].drop_duplicates().shape[0]),
                "raw_exploration_pass_count": pass_count,
                "raw_exploration_pass_weight": passed_weight,
                "raw_exploration_pass_weight_share": passed_weight / max(total_weight, 1e-12),
                "mean_ks_shift": float(pd.to_numeric(group["ks_shift"], errors="coerce").mean()),
                "mean_period_auc_lift": float(pd.to_numeric(group["period_auc_lift"], errors="coerce").mean()),
                "mean_abs_median_shift": float(np.nanmean(np.abs(pd.to_numeric(group["median_shift_robust_z"], errors="coerce").to_numpy(dtype=np.float64)))),
                "median_shift_direction_consistency": _direction_consistency(passed["median_shift_robust_z"]) if not passed.empty else 0.0,
                "selected_for_operator_generation": bool(
                    pass_count >= int(config.raw_exploration_min_pass_count)
                    and raw_score >= float(config.raw_exploration_min_score)
                ),
            }
        )
    out = pd.DataFrame(agg_rows).sort_values(
        [
            "selected_for_operator_generation",
            "raw_breakout_link_score",
            "raw_exploration_pass_count",
            "raw_exploration_pass_weight_share",
        ],
        ascending=False,
        kind="mergesort",
    )
    selected_count = int(out["selected_for_operator_generation"].sum()) if "selected_for_operator_generation" in out else 0
    if selected_count == 0 and not out.empty:
        raw_limit = int(config.raw_exploration_max_features)
        fallback_n = min(raw_limit, len(out)) if raw_limit > 0 else len(out)
        out.loc[out.index[:fallback_n], "selected_for_operator_generation"] = True
        selected_count = fallback_n
    _log(
        f"[breakout-screen] raw feature pre-screen scored={len(out)} "
        f"selected_for_operator_generation={selected_count}"
    )
    return out


def _feature_scores_for_episode(
    frame: pd.DataFrame,
    *,
    strategy: str,
    layer: str,
    slice_name: str,
    episode: pd.Series,
    features: Sequence[str],
    pred_col: str,
    label_col: str,
    config: AnalysisConfig,
    feature_arrays: dict[str, np.ndarray] | None = None,
    pred_all: np.ndarray | None = None,
    y_all: np.ndarray | None = None,
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    day = ts.dt.floor("D")
    start = pd.Timestamp(episode["start_day"])
    end = pd.Timestamp(episode["end_day"])
    baseline_end = start - pd.Timedelta(days=max(0, int(config.embargo_days)))
    baseline_mask = day.lt(baseline_end).fillna(False).to_numpy(dtype=bool)
    episode_mask = day.ge(start).fillna(False).to_numpy(dtype=bool) & day.le(end).fillna(False).to_numpy(dtype=bool)
    baseline_sample_cap, episode_sample_cap = _baseline_episode_sample_limits(config)
    baseline_pos = _sample_positions(baseline_mask, baseline_sample_cap, int(config.random_seed))
    episode_pos = _sample_positions(episode_mask, episode_sample_cap, int(config.random_seed) + 1)
    if len(baseline_pos) < 100 or len(episode_pos) < 100:
        return pd.DataFrame()

    if pred_all is None:
        pred_all = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    if y_all is None:
        y_all = pd.to_numeric(frame[label_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    residual_all = y_all - np.clip(pred_all, 1e-5, 1.0 - 1e-5)
    abs_error_all = np.abs(residual_all)
    rows: list[dict[str, object]] = []
    combined_pos = np.concatenate([baseline_pos, episode_pos])
    residual_scale = max(float(np.nanstd(residual_all[baseline_pos])), 1e-6)
    episode_duration_days = _episode_duration_days(episode)
    breakout_weight = _episode_breakout_weight(episode, config)
    breakout_key = f"{layer}|{strategy}|{slice_name}|{int(episode['episode_id'])}"

    for feature in features:
        arr = feature_arrays.get(feature) if feature_arrays is not None else None
        if arr is None:
            arr = _numeric_array(frame[feature])
        base = arr[baseline_pos]
        bad = arr[episode_pos]
        base_finite = base[np.isfinite(base)]
        bad_finite = bad[np.isfinite(bad)]
        if base_finite.size < 100 or bad_finite.size < 50:
            continue
        center, scale = _robust_center_scale(base_finite)
        base_z = np.clip((base_finite - center) / scale, -8.0, 8.0)
        bad_z = np.clip((bad_finite - center) / scale, -8.0, 8.0)
        ks = _ks_stat(base_z, bad_z)
        auc, auc_lift = _binary_auc_lift(base_z, bad_z)
        med_shift = float(np.nanmedian(bad_z) - np.nanmedian(base_z))
        q25_b, q75_b = np.nanpercentile(base_z, [25, 75])
        q25_e, q75_e = np.nanpercentile(bad_z, [25, 75])
        iqr_ratio_log = float(np.log((q75_e - q25_e + 1e-6) / (q75_b - q25_b + 1e-6)))
        lo, hi = np.nanpercentile(base_z, [10, 90])
        base_tail = float(np.mean((base_z <= lo) | (base_z >= hi)))
        bad_tail = float(np.mean((bad_z <= lo) | (bad_z >= hi)))
        tail_share_shift = bad_tail - base_tail
        shift_score = _clip01(
            0.30 * (ks / 0.35)
            + 0.25 * (auc_lift / 0.35)
            + 0.25 * (abs(med_shift) / 1.50)
            + 0.10 * (abs(iqr_ratio_log) / 1.00)
            + 0.10 * (abs(tail_share_shift) / 0.30)
        )

        x_rel = np.clip((arr[combined_pos] - center) / scale, -8.0, 8.0)
        pred_corr = _safe_corr(x_rel, pred_all[combined_pos], spearman=False)
        residual_corr = _safe_corr(x_rel, residual_all[combined_pos], spearman=False)
        abs_error_corr = _safe_corr(x_rel, abs_error_all[combined_pos], spearman=False)
        prediction_relevance = abs(pred_corr)
        residual_relevance = max(abs(residual_corr), abs(abs_error_corr))
        relevance_score = _clip01(max(0.60 * prediction_relevance, residual_relevance) / 0.25)

        direction = float(np.sign(med_shift))
        directional_harm = 0.0
        if direction != 0.0:
            directional_harm = max(0.0, -direction * residual_corr)

        x_combined = arr[combined_pos]
        resid_combined = residual_all[combined_pos]
        q_tail = np.nanpercentile(base_finite, 75 if direction >= 0 else 25)
        tail_mask = x_combined >= q_tail if direction >= 0 else x_combined <= q_tail
        finite_tail = tail_mask & np.isfinite(resid_combined) & np.isfinite(x_combined)
        finite_rest = (~tail_mask) & np.isfinite(resid_combined) & np.isfinite(x_combined)
        tail_harm = 0.0
        if int(finite_tail.sum()) >= 30 and int(finite_rest.sum()) >= 30:
            tail_mean = float(np.nanmean(resid_combined[finite_tail]))
            rest_mean = float(np.nanmean(resid_combined[finite_rest]))
            tail_harm = max(0.0, (rest_mean - tail_mean) / residual_scale)
        harmfulness_score = _clip01(max(directional_harm / 0.20, tail_harm / 1.00))
        candidate_score = float(shift_score * relevance_score * harmfulness_score)
        episode_explanation_score = float(candidate_score * breakout_weight)
        is_composite = _is_composite_operator_feature(feature)
        if is_composite:
            min_shift = 0.25
            min_relevance = 0.15
            min_harm = 0.10
            candidate_floor = float(config.composite_candidate_min_score)
            threshold_kind = "composite_strict"
        else:
            min_shift = 0.10
            min_relevance = 0.04
            min_harm = 0.03
            candidate_floor = float(config.raw_candidate_min_score)
            threshold_kind = "raw_exploratory"
        regime_candidate = bool(
            shift_score >= min_shift
            and relevance_score >= min_relevance
            and harmfulness_score >= min_harm
            and candidate_score >= candidate_floor
        )
        rows.append(
            {
                "strategy": strategy,
                "layer": layer,
                "slice": slice_name,
                "breakout_key": breakout_key,
                "episode_id": int(episode["episode_id"]),
                "episode_start_day": start,
                "episode_end_day": end,
                "episode_duration_days": int(episode_duration_days),
                "episode_window_count": int(_safe_float(episode.get("window_count"), 1.0)),
                "episode_min_hit_rate_surprise_z": _safe_float(episode.get("min_hit_rate_surprise_z")),
                "episode_min_hit_rate_delta": _safe_float(episode.get("min_hit_rate_delta")),
                "episode_breakout_weight": breakout_weight,
                "feature": feature,
                "feature_family": _feature_family(feature),
                "feature_is_composite": bool(is_composite),
                "candidate_threshold_kind": threshold_kind,
                "candidate_score_floor": candidate_floor,
                "baseline_rows": int(len(base_finite)),
                "episode_rows": int(len(bad_finite)),
                "ks_shift": ks,
                "period_auc": auc,
                "period_auc_lift": auc_lift,
                "median_shift_robust_z": med_shift,
                "iqr_ratio_log": iqr_ratio_log,
                "tail_share_shift": tail_share_shift,
                "shift_score": shift_score,
                "pred_corr": pred_corr,
                "residual_corr": residual_corr,
                "abs_error_corr": abs_error_corr,
                "prediction_relevance": prediction_relevance,
                "residual_relevance": residual_relevance,
                "relevance_score": relevance_score,
                "directional_harm": directional_harm,
                "tail_harm": tail_harm,
                "harmfulness_score": harmfulness_score,
                "candidate_score": candidate_score,
                "episode_explanation_score": episode_explanation_score,
                "regime_candidate": regime_candidate,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["candidate_score", "shift_score", "relevance_score", "harmfulness_score"],
        ascending=False,
        kind="mergesort",
    )


def _effective_rank(eigvals: np.ndarray) -> float:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = np.clip(eigvals[np.isfinite(eigvals)], 0.0, np.inf)
    total = float(eigvals.sum())
    if total <= 1e-12:
        return 0.0
    p = eigvals / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    if matrix.shape[0] < 5 or matrix.shape[1] < 2:
        return {
            "pc1_concentration": np.nan,
            "top3_concentration": np.nan,
            "effective_rank": np.nan,
            "participation_ratio": np.nan,
            "fragmentation": np.nan,
            "mean_abs_corr": np.nan,
        }
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.cov(matrix, rowvar=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals_pos = np.clip(eigvals, 0.0, np.inf)
    total = float(eigvals_pos.sum())
    ordered = np.sort(eigvals_pos)[::-1]
    pc1 = float(max(ordered[0] / total, 0.0)) if total > 1e-12 and ordered.size else np.nan
    top3 = float(np.sum(ordered[:3]) / total) if total > 1e-12 and ordered.size else np.nan
    eff_rank = _effective_rank(eigvals)
    participation = float((np.sum(eigvals_pos) ** 2) / max(np.sum(eigvals_pos * eigvals_pos), 1e-12)) if total > 1e-12 else np.nan
    if corr.shape[0] > 1:
        tri = corr[np.triu_indices_from(corr, k=1)]
        mean_abs_corr = float(np.mean(np.abs(tri))) if tri.size else np.nan
    else:
        mean_abs_corr = np.nan
    return {
        "pc1_concentration": pc1,
        "top3_concentration": top3,
        "effective_rank": eff_rank,
        "participation_ratio": participation,
        "fragmentation": float(eff_rank / max(matrix.shape[1], 1)) if np.isfinite(eff_rank) else np.nan,
        "mean_abs_corr": mean_abs_corr,
    }


def _dominant_eigenvector(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 5 or matrix.shape[1] < 2:
        return np.asarray([], dtype=np.float64)
    arr = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    cov = np.cov(arr, rowvar=False)
    if cov.ndim != 2:
        return np.asarray([], dtype=np.float64)
    vals, vecs = np.linalg.eigh(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
    if vals.size == 0:
        return np.asarray([], dtype=np.float64)
    vec = vecs[:, int(np.argmax(vals))]
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else np.asarray([], dtype=np.float64)


def _matrix_break_reference_stats(
    base_z: np.ndarray,
    bad_z: np.ndarray,
    *,
    kind: str,
) -> dict[str, float]:
    if kind == "cov":
        ref = _cov_matrix(base_z)
        bad = _cov_matrix(bad_z)
        maker = _cov_matrix
    else:
        ref = _corr_matrix(base_z)
        bad = _corr_matrix(bad_z)
        maker = _corr_matrix
    ref_norm = max(float(np.linalg.norm(ref, ord="fro")), 1e-8)
    bad_distance = float(np.linalg.norm(bad - ref, ord="fro") / ref_norm)
    n = int(len(base_z))
    win = int(max(len(bad_z), 12))
    if n < win * 3:
        return {
            f"historical_{kind}_break_distance": bad_distance,
            f"historical_{kind}_break_z": np.nan,
            f"historical_{kind}_break_percentile": np.nan,
            f"historical_{kind}_break_reference_n": 0,
        }
    stride = max(1, win // 2)
    distances: list[float] = []
    for start in range(0, n - win + 1, stride):
        sample = base_z[start : start + win]
        if sample.shape[0] < 12:
            continue
        dist = float(np.linalg.norm(maker(sample) - ref, ord="fro") / ref_norm)
        if np.isfinite(dist):
            distances.append(dist)
    if len(distances) < 3:
        return {
            f"historical_{kind}_break_distance": bad_distance,
            f"historical_{kind}_break_z": np.nan,
            f"historical_{kind}_break_percentile": np.nan,
            f"historical_{kind}_break_reference_n": len(distances),
        }
    arr = np.asarray(distances, dtype=np.float64)
    center = float(np.nanmedian(arr))
    scale = float(np.nanmedian(np.abs(arr - center)) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.nanstd(arr))
    z = (bad_distance - center) / max(scale, 1e-8)
    percentile = float(np.mean(arr <= bad_distance))
    return {
        f"historical_{kind}_break_distance": bad_distance,
        f"historical_{kind}_break_z": float(z),
        f"historical_{kind}_break_percentile": percentile,
        f"historical_{kind}_break_reference_n": len(distances),
    }


def _partial_corr_from_precision(precision: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(precision), 1e-12, np.inf))
    denom = np.outer(diag, diag)
    partial = -precision / np.maximum(denom, 1e-12)
    np.fill_diagonal(partial, 0.0)
    return np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)


def _precision_shift_diagnostics(base_z: np.ndarray, bad_z: np.ndarray, *, max_features: int) -> dict[str, float | str]:
    p = min(int(max_features), base_z.shape[1], bad_z.shape[1])
    if p < 2:
        return {"precision_status": "too_few_features"}
    base = base_z[:, :p]
    bad = bad_z[:, :p]
    min_obs = max(30, p * 3)
    if base.shape[0] < min_obs or bad.shape[0] < max(20, p * 2):
        return {"precision_status": "insufficient_observations"}
    def _rows_from_precision(base_precision: np.ndarray, bad_precision: np.ndarray, status: str) -> dict[str, float | str]:
        base_partial = _partial_corr_from_precision(base_precision)
        bad_partial = _partial_corr_from_precision(bad_precision)
        threshold = 1e-3
        base_edges = np.abs(base_partial) > threshold
        bad_edges = np.abs(bad_partial) > threshold
        tri = np.triu_indices(p, k=1)
        base_edge_vec = base_edges[tri]
        bad_edge_vec = bad_edges[tri]
        base_norm = max(float(np.linalg.norm(base_partial, ord="fro")), 1e-8)
        return {
            "precision_status": status,
            "precision_feature_count": p,
            "precision_partial_corr_frobenius_shift": float(np.linalg.norm(bad_partial - base_partial, ord="fro") / base_norm),
            "precision_base_edge_density": float(np.mean(base_edge_vec)) if base_edge_vec.size else np.nan,
            "precision_episode_edge_density": float(np.mean(bad_edge_vec)) if bad_edge_vec.size else np.nan,
            "precision_edge_turnover": float(np.mean(base_edge_vec != bad_edge_vec)) if base_edge_vec.size else np.nan,
            "precision_mean_abs_partial_corr_delta": float(np.mean(np.abs(bad_partial[tri])) - np.mean(np.abs(base_partial[tri]))) if base_edge_vec.size else np.nan,
        }

    def _ridge_precision(arr: np.ndarray, shrinkage: float = 0.15) -> np.ndarray:
        cov = _cov_matrix(arr)
        diag_scale = float(np.trace(cov) / max(cov.shape[0], 1))
        target = np.eye(cov.shape[0], dtype=np.float64) * max(diag_scale, 1e-6)
        shrunk = (1.0 - float(shrinkage)) * cov + float(shrinkage) * target
        return np.linalg.pinv(shrunk)

    # GraphicalLasso was too unstable for these per-episode matrices in practice
    # and fell back on nearly every row. Use the deterministic ridge precision
    # proxy directly so this diagnostic remains cheap and reproducible.
    try:
        return _rows_from_precision(
            _ridge_precision(base),
            _ridge_precision(bad),
            "ridge_precision",
        )
    except Exception as exc:
        return {"precision_status": f"failed:{type(exc).__name__}"}


def _tail_coexceedance_diagnostics(base_z: np.ndarray, bad_z: np.ndarray, *, threshold: float = 1.5) -> dict[str, float]:
    def _matrix(arr: np.ndarray) -> np.ndarray:
        extreme = np.abs(np.asarray(arr, dtype=np.float64)) >= float(threshold)
        if extreme.shape[0] == 0:
            return np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float64)
        mat = extreme.T @ extreme.astype(np.float64) / max(extreme.shape[0], 1)
        np.fill_diagonal(mat, 0.0)
        return mat

    base = _matrix(base_z)
    bad = _matrix(bad_z)
    tri = np.triu_indices_from(base, k=1) if base.shape[0] > 1 else ([], [])
    base_norm = max(float(np.linalg.norm(base, ord="fro")), 1e-8)
    return {
        "tail_coexceedance_frobenius_shift": float(np.linalg.norm(bad - base, ord="fro") / base_norm),
        "base_mean_tail_coexceedance": float(np.mean(base[tri])) if len(tri[0]) else np.nan,
        "episode_mean_tail_coexceedance": float(np.mean(bad[tri])) if len(tri[0]) else np.nan,
        "mean_tail_coexceedance_delta": float(np.mean(bad[tri]) - np.mean(base[tri])) if len(tri[0]) else np.nan,
    }


def _distance_correlation_1d(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    n = int(x.size)
    if n < 20:
        return np.nan
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(a * b))
    dvarx = float(np.mean(a * a))
    dvary = float(np.mean(b * b))
    if dvarx <= 1e-12 or dvary <= 1e-12 or dcov2 <= 0.0:
        return 0.0
    return float(np.sqrt(dcov2 / np.sqrt(dvarx * dvary)))


def _distance_correlation_matrix(arr: np.ndarray, *, max_rows: int = 400) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if values.shape[0] > max_rows:
        idx = np.linspace(0, values.shape[0] - 1, int(max_rows)).astype(int)
        values = values[idx]
    p = values.shape[1]
    out = np.eye(p, dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            value = _distance_correlation_1d(values[:, i], values[:, j])
            out[i, j] = out[j, i] = 0.0 if not np.isfinite(value) else value
    return out


def _nonlinear_dependence_diagnostics(base_z: np.ndarray, bad_z: np.ndarray, *, max_features: int) -> dict[str, float]:
    p = min(int(max_features), base_z.shape[1], bad_z.shape[1])
    if p < 2 or base_z.shape[0] < 30 or bad_z.shape[0] < 20:
        return {
            "distance_corr_feature_count": p,
            "distance_corr_frobenius_shift": np.nan,
            "distance_corr_mean_abs_delta": np.nan,
        }
    base = _distance_correlation_matrix(base_z[:, :p])
    bad = _distance_correlation_matrix(bad_z[:, :p])
    tri = np.triu_indices(p, k=1)
    base_norm = max(float(np.linalg.norm(base, ord="fro")), 1e-8)
    return {
        "distance_corr_feature_count": p,
        "distance_corr_frobenius_shift": float(np.linalg.norm(bad - base, ord="fro") / base_norm),
        "distance_corr_mean_abs_delta": float(np.mean(np.abs(bad[tri] - base[tri]))) if tri[0].size else np.nan,
    }


def _corr_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 5 or matrix.shape[1] < 2:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    matrix = np.nan_to_num(matrix.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _cov_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 5 or matrix.shape[1] < 2:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    matrix = np.nan_to_num(matrix.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.cov(matrix, rowvar=False)
    return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)


def _autocorr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= lag + 5:
        return np.nan
    x = values[:-lag]
    y = values[lag:]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < max(5, lag + 2):
        return np.nan
    return _safe_corr(x[mask], y[mask], spearman=False)


def _build_timestamp_feature_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    selected = [str(feature) for feature in features if str(feature) in frame.columns]
    if not selected or "timestamp" not in frame.columns:
        return pd.DataFrame()
    work = frame[["timestamp", *selected]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return pd.DataFrame()
    for col in selected:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(np.float32, copy=False)
    return work.groupby("timestamp", sort=True)[selected].mean()


def _episode_covariance_autocorr(
    frame: pd.DataFrame,
    *,
    strategy: str,
    layer: str,
    slice_name: str,
    episode: pd.Series,
    feature_scores: pd.DataFrame,
    features: Sequence[str],
    config: AnalysisConfig,
    timestamp_feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    if feature_scores.empty:
        selected = list(features[: int(config.max_cov_features)])
    else:
        selected = (
            feature_scores.sort_values("candidate_score", ascending=False, kind="mergesort")["feature"]
            .drop_duplicates()
            .head(int(config.max_cov_features))
            .astype(str)
            .tolist()
        )
    if len(selected) < 2:
        return pd.DataFrame()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    day = ts.dt.floor("D")
    start = pd.Timestamp(episode["start_day"])
    end = pd.Timestamp(episode["end_day"])
    baseline_end = start - pd.Timedelta(days=max(0, int(config.embargo_days)))
    baseline_mask = day.lt(baseline_end).fillna(False)
    episode_mask = day.ge(start).fillna(False) & day.le(end).fillna(False)
    if timestamp_feature_frame is None or timestamp_feature_frame.empty:
        timestamp_feature_frame = _build_timestamp_feature_frame(frame, features)
    if timestamp_feature_frame.empty:
        return pd.DataFrame()
    selected = [col for col in selected if col in timestamp_feature_frame.columns]
    if len(selected) < 2:
        return pd.DataFrame()
    ts_index = pd.to_datetime(timestamp_feature_frame.index, utc=True, errors="coerce")
    base_time = timestamp_feature_frame.loc[ts_index.floor("D") < baseline_end, selected]
    bad_time = timestamp_feature_frame.loc[(ts_index.floor("D") >= start) & (ts_index.floor("D") <= end), selected]
    if len(base_time) < 24 or len(bad_time) < 12:
        return pd.DataFrame()
    base_arr = base_time.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    bad_arr = bad_time.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    centers = np.nanmedian(base_arr, axis=0)
    q25 = np.nanpercentile(base_arr, 25, axis=0)
    q75 = np.nanpercentile(base_arr, 75, axis=0)
    scales = np.maximum(q75 - q25, 1e-6)
    base_z = np.clip((base_arr - centers) / scales, -8.0, 8.0)
    bad_z = np.clip((bad_arr - centers) / scales, -8.0, 8.0)
    base_z = np.nan_to_num(base_z, nan=0.0, posinf=0.0, neginf=0.0)
    bad_z = np.nan_to_num(bad_z, nan=0.0, posinf=0.0, neginf=0.0)
    cov_base = _cov_matrix(base_z)
    cov_bad = _cov_matrix(bad_z)
    corr_base = _corr_matrix(base_z)
    corr_bad = _corr_matrix(bad_z)
    base_diag = _matrix_diagnostics(base_z)
    bad_diag = _matrix_diagnostics(bad_z)
    cov_frob = float(np.linalg.norm(cov_bad - cov_base, ord="fro") / max(np.linalg.norm(cov_base, ord="fro"), 1e-8))
    corr_frob = float(np.linalg.norm(corr_bad - corr_base, ord="fro") / max(np.linalg.norm(corr_base, ord="fro"), 1e-8))
    base_vec = _dominant_eigenvector(base_z)
    bad_vec = _dominant_eigenvector(bad_z)
    factor_rotation = (
        float(1.0 - abs(float(np.dot(base_vec, bad_vec))))
        if base_vec.size and bad_vec.size and base_vec.size == bad_vec.size
        else np.nan
    )
    extra_diag: dict[str, object] = {}
    if bool(config.advanced_covariance_enabled):
        extra_diag.update(_matrix_break_reference_stats(base_z, bad_z, kind="cov"))
        extra_diag.update(_matrix_break_reference_stats(base_z, bad_z, kind="corr"))
        extra_diag.update(
            _precision_shift_diagnostics(
                base_z,
                bad_z,
                max_features=int(config.max_precision_features),
            )
        )
        extra_diag.update(_tail_coexceedance_diagnostics(base_z, bad_z))
        extra_diag.update(
            _nonlinear_dependence_diagnostics(
                base_z,
                bad_z,
                max_features=int(config.max_nonlinear_dependence_features),
            )
        )
    lag_rows: list[dict[str, object]] = []
    for lag in (1, 3, 6, 24):
        base_ac = np.asarray([_autocorr(base_z[:, i], lag) for i in range(base_z.shape[1])], dtype=np.float64)
        bad_ac = np.asarray([_autocorr(bad_z[:, i], lag) for i in range(bad_z.shape[1])], dtype=np.float64)
        delta = bad_ac - base_ac
        finite = np.isfinite(delta)
        row = {
            "strategy": strategy,
            "layer": layer,
            "slice": slice_name,
            "episode_id": int(episode["episode_id"]),
            "episode_start_day": start,
            "episode_end_day": end,
            "feature_count": len(selected),
            "baseline_timestamps": int(len(base_time)),
            "episode_timestamps": int(len(bad_time)),
            "lag": lag,
            "feature_covariance_frobenius_shift": cov_frob,
            "feature_correlation_frobenius_shift": corr_frob,
            "factor_rotation": factor_rotation,
            "base_pc1_concentration": base_diag["pc1_concentration"],
            "episode_pc1_concentration": bad_diag["pc1_concentration"],
            "pc1_concentration_delta": bad_diag["pc1_concentration"] - base_diag["pc1_concentration"],
            "base_top3_concentration": base_diag["top3_concentration"],
            "episode_top3_concentration": bad_diag["top3_concentration"],
            "top3_concentration_delta": bad_diag["top3_concentration"] - base_diag["top3_concentration"],
            "base_effective_rank": base_diag["effective_rank"],
            "episode_effective_rank": bad_diag["effective_rank"],
            "effective_rank_delta": bad_diag["effective_rank"] - base_diag["effective_rank"],
            "base_participation_ratio": base_diag["participation_ratio"],
            "episode_participation_ratio": bad_diag["participation_ratio"],
            "participation_ratio_delta": bad_diag["participation_ratio"] - base_diag["participation_ratio"],
            "base_fragmentation": base_diag["fragmentation"],
            "episode_fragmentation": bad_diag["fragmentation"],
            "fragmentation_delta": bad_diag["fragmentation"] - base_diag["fragmentation"],
            "base_mean_abs_corr": base_diag["mean_abs_corr"],
            "episode_mean_abs_corr": bad_diag["mean_abs_corr"],
            "mean_abs_corr_delta": bad_diag["mean_abs_corr"] - base_diag["mean_abs_corr"],
            "mean_abs_autocorr_delta": float(np.nanmean(np.abs(delta[finite]))) if finite.any() else np.nan,
            "max_abs_autocorr_delta": float(np.nanmax(np.abs(delta[finite]))) if finite.any() else np.nan,
            "median_autocorr_delta": float(np.nanmedian(delta[finite])) if finite.any() else np.nan,
        }
        row.update(extra_diag)
        lag_rows.append(row)
    return pd.DataFrame(lag_rows)


def _performance_metrics_for_episode(
    frame: pd.DataFrame,
    *,
    episode: pd.Series,
    pred_col: str,
    label_col: str,
    pnl_col: str,
) -> dict[str, float]:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    day = ts.dt.floor("D")
    start = pd.Timestamp(episode["start_day"])
    end = pd.Timestamp(episode["end_day"])
    mask = day.ge(start).fillna(False) & day.le(end).fillna(False)
    if not mask.any():
        return {}
    pred = pd.to_numeric(frame.loc[mask, pred_col], errors="coerce").clip(1e-5, 1.0 - 1e-5)
    y = pd.to_numeric(frame.loc[mask, label_col], errors="coerce")
    valid = pred.notna() & y.notna()
    if not valid.any():
        return {}
    p = pred.loc[valid].to_numpy(dtype=np.float64)
    yy = y.loc[valid].to_numpy(dtype=np.float64)
    out = {
        "episode_rows": float(len(yy)),
        "episode_actual_hit_rate": float(np.mean(yy)),
        "episode_expected_hit_rate": float(np.mean(p)),
        "episode_hit_rate_delta": float(np.mean(yy) - np.mean(p)),
        "episode_hit_rate_surprise_z": float(np.sum(yy - p) / np.sqrt(np.sum(np.clip(p * (1.0 - p), 1e-6, np.inf)))),
    }
    if pnl_col and pnl_col in frame.columns:
        pnl = pd.to_numeric(frame.loc[mask, pnl_col], errors="coerce").to_numpy(dtype=np.float64)
        out["episode_mean_pnl"] = float(np.nanmean(pnl))
        out["episode_sum_pnl"] = float(np.nansum(pnl))
    return out


def _binary_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    prob = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    mask = np.isfinite(y) & np.isfinite(prob)
    if int(mask.sum()) == 0:
        return np.nan
    y = y[mask]
    prob = prob[mask]
    return float(-np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))


def _brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(p, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(prob)
    if int(mask.sum()) == 0:
        return np.nan
    return float(np.mean((prob[mask] - y[mask]) ** 2))


def _sample_feature_matrix(
    frame: pd.DataFrame,
    positions: np.ndarray,
    features: Sequence[str],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    kept: list[str] = []
    columns: list[np.ndarray] = []
    centers: list[float] = []
    scales: list[float] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        arr = _numeric_array(frame[feature])
        values = arr[positions].astype(np.float32, copy=False)
        train_values = values[train_mask]
        finite = np.isfinite(train_values)
        if int(finite.sum()) < 50:
            continue
        center, scale = _robust_center_scale(train_values[finite])
        scaled = np.clip((values.astype(np.float64, copy=False) - center) / scale, -8.0, 8.0).astype(np.float32)
        scaled[~np.isfinite(scaled)] = 0.0
        columns.append(scaled)
        kept.append(str(feature))
        centers.append(center)
        scales.append(scale)
    if not columns:
        return np.empty((len(positions), 0), dtype=np.float32), [], np.asarray([]), np.asarray([])
    return (
        np.column_stack(columns).astype(np.float32, copy=False),
        kept,
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def _top_features_for_ebm_fold(
    feature_scores: pd.DataFrame,
    *,
    train_breakout_keys: set[str],
    max_features: int,
    config: AnalysisConfig,
) -> list[str]:
    if feature_scores.empty or "breakout_key" not in feature_scores:
        return []
    work = feature_scores.loc[feature_scores["breakout_key"].astype(str).isin(train_breakout_keys)].copy()
    if work.empty:
        return []
    if "regime_candidate" in work.columns:
        candidate = work.loc[work["regime_candidate"].astype(bool)].copy()
        if not candidate.empty:
            work = candidate
    if "episode_breakout_weight" not in work.columns:
        work["episode_breakout_weight"] = 1.0
    grouped = (
        work.groupby("feature", sort=False)
        .agg(
            explanation_score_sum=("episode_explanation_score", "sum"),
            candidate_score_mean=("candidate_score", "mean"),
            candidate_score_max=("candidate_score", "max"),
            recurrence_episodes=("breakout_key", "nunique"),
            recurrence_weight=("episode_breakout_weight", "sum"),
        )
        .reset_index()
    )
    min_recurrence = int(max(1, config.ebm_min_recurrence_episodes))
    recurrent = grouped.loc[grouped["recurrence_episodes"].astype(int).ge(min_recurrence)].copy()
    if not recurrent.empty:
        grouped = recurrent
    grouped["recurrence_rank_score"] = (
        grouped["explanation_score_sum"].astype(float)
        * np.log1p(grouped["recurrence_episodes"].astype(float))
        * np.sqrt(np.maximum(grouped["recurrence_weight"].astype(float), 1e-12))
    )
    grouped = grouped.sort_values(
        ["recurrence_rank_score", "recurrence_episodes", "candidate_score_mean"],
        ascending=False,
        kind="mergesort",
    )
    return grouped["feature"].astype(str).head(int(max_features)).tolist()


def _candidate_pairs_for_ebm(features: Sequence[str], scores: pd.DataFrame, *, max_pairs: int) -> list[tuple[str, str]]:
    values = list(dict.fromkeys(str(f) for f in features if str(f)))
    if len(values) < 2:
        return []
    score_map: dict[str, float] = {}
    if not scores.empty and "feature" in scores:
        grouped = scores.groupby("feature")["candidate_score"].mean()
        score_map = {str(k): float(v) for k, v in grouped.items()}
    rows: list[tuple[float, str, str]] = []
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            diversity = 1.15 if _feature_family(left) != _feature_family(right) else 0.85
            pair_score = math.sqrt(max(score_map.get(left, 1e-4), 1e-4) * max(score_map.get(right, 1e-4), 1e-4)) * diversity
            rows.append((pair_score, left, right))
    rows.sort(reverse=True)
    return [(left, right) for _score, left, right in rows[: int(max_pairs)]]


def _control_folds_by_timestamp(
    timestamps: pd.Series,
    *,
    n_folds: int,
) -> np.ndarray:
    if n_folds <= 1:
        return np.zeros(len(timestamps), dtype=np.int32)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    unique = pd.Index(ts.dropna().drop_duplicates().sort_values())
    fold_map = {value: idx % int(n_folds) for idx, value in enumerate(unique)}
    return np.asarray([fold_map.get(value, -1) for value in ts], dtype=np.int32)


def _aggregate_design_by_timestamp(
    X: np.ndarray,
    timestamps: pd.Series,
    groups: np.ndarray,
    weights: np.ndarray,
    control_folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, groups, weights, control_folds
    work = pd.DataFrame(X.astype(np.float32, copy=False))
    work["timestamp"] = pd.to_datetime(timestamps, utc=True, errors="coerce").to_numpy()
    work["group"] = np.asarray(groups, dtype=np.int32)
    work["weight"] = np.asarray(weights, dtype=np.float32)
    work["control_fold"] = np.asarray(control_folds, dtype=np.int32)
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return X, groups, weights, control_folds
    feature_cols = [col for col in work.columns if isinstance(col, int)]
    grouped = (
        work.groupby(["timestamp", "group"], sort=True, observed=True)
        .agg(
            {**{col: "mean" for col in feature_cols}, "weight": "mean", "control_fold": "first"}
        )
        .reset_index()
    )
    X_out = grouped[feature_cols].to_numpy(dtype=np.float32, copy=False)
    groups_out = grouped["group"].to_numpy(dtype=np.int32, copy=False)
    weights_out = grouped["weight"].to_numpy(dtype=np.float32, copy=False)
    folds_out = grouped["control_fold"].to_numpy(dtype=np.int32, copy=False)
    return X_out, groups_out, weights_out, folds_out


def _episode_ebm_interaction_diagnostics(
    frame: pd.DataFrame,
    *,
    strategy: str,
    layer: str,
    slice_name: str,
    episodes: pd.DataFrame,
    feature_scores: pd.DataFrame,
    slice_mask: np.ndarray,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not bool(config.ebm_interaction_enabled) or episodes.empty or feature_scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except Exception as exc:
        _log(f"[ebm] unavailable: {type(exc).__name__}: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    ep = episodes.copy()
    ep["__breakout_weight"] = [_episode_breakout_weight(row, config) for _, row in ep.iterrows()]
    ep = ep.sort_values(["__breakout_weight", "min_hit_rate_surprise_z"], ascending=[False, True], kind="mergesort")
    ep = ep.head(max(1, int(config.ebm_max_episodes))).reset_index(drop=True)
    if ep.empty:
        return pd.DataFrame(), pd.DataFrame()

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    day = ts.dt.floor("D")
    slice_mask = np.asarray(slice_mask, dtype=bool)
    any_episode_mask = np.zeros(len(frame), dtype=bool)
    episode_positions: list[np.ndarray] = []
    episode_keys: list[str] = []
    episode_weights: list[float] = []
    rng = np.random.default_rng(int(config.random_seed) + 991)
    for _, row in ep.iterrows():
        start = pd.Timestamp(row["start_day"])
        end = pd.Timestamp(row["end_day"])
        mask = (
            slice_mask
            & day.ge(start).fillna(False).to_numpy(dtype=bool)
            & day.le(end).fillna(False).to_numpy(dtype=bool)
        )
        pos = np.flatnonzero(mask)
        if pos.size < 50:
            continue
        if pos.size > int(config.ebm_max_rows_per_episode):
            pos = np.sort(rng.choice(pos, size=int(config.ebm_max_rows_per_episode), replace=False))
        any_episode_mask[pos] = True
        episode_positions.append(pos)
        episode_keys.append(f"{layer}|{strategy}|{slice_name}|{int(row['episode_id'])}")
        episode_weights.append(_episode_breakout_weight(row, config))
    if len(episode_positions) < 2:
        return pd.DataFrame(), pd.DataFrame()
    control_mask = slice_mask & ~any_episode_mask & ts.notna().to_numpy(dtype=bool)
    control_pos = np.flatnonzero(control_mask)
    if control_pos.size < max(100, int(config.ebm_min_rows // 2)):
        return pd.DataFrame(), pd.DataFrame()
    if control_pos.size > int(config.ebm_max_control_rows):
        control_pos = np.sort(rng.choice(control_pos, size=int(config.ebm_max_control_rows), replace=False))
    selected_episode_count = len(episode_positions)
    median_ep_weight = max(float(np.median(episode_weights)), 1e-12)
    all_positions = [control_pos, *episode_positions]
    positions = np.concatenate(all_positions).astype(np.int64, copy=False)
    groups = np.full(len(positions), -1, dtype=np.int32)
    weights = np.ones(len(positions), dtype=np.float32)
    offset = len(control_pos)
    for idx, pos in enumerate(episode_positions):
        n = len(pos)
        groups[offset : offset + n] = idx
        weights[offset : offset + n] = float(np.clip(episode_weights[idx] / median_ep_weight, 0.25, 10.0))
        offset += n
    y = (groups >= 0).astype(np.int8)
    if len(positions) < int(config.ebm_min_rows) or y.sum() < 100:
        return pd.DataFrame(), pd.DataFrame()
    position_timestamps = pd.Series(ts.iloc[positions].to_numpy())
    control_folds = np.full(len(positions), -1, dtype=np.int32)
    control_folds[: len(control_pos)] = _control_folds_by_timestamp(
        pd.Series(ts.iloc[control_pos].to_numpy()),
        n_folds=selected_episode_count,
    )
    use_timestamp_aggregate = (
        int(config.timestamp_aggregate_row_threshold) > 0
        and len(positions) >= int(config.timestamp_aggregate_row_threshold)
    )
    if use_timestamp_aggregate:
        _log(
            f"[ebm] {layer} {strategy} {slice_name}: using timestamp-level "
            f"interaction design rows={len(positions)} "
            f"threshold={config.timestamp_aggregate_row_threshold}"
        )

    pair_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for heldout_idx, heldout_key in enumerate(episode_keys):
        train_episode_keys = {key for idx, key in enumerate(episode_keys) if idx != heldout_idx}
        fold_features = _top_features_for_ebm_fold(
            feature_scores,
            train_breakout_keys=train_episode_keys,
            max_features=int(config.ebm_max_features),
            config=config,
        )
        if len(fold_features) < 2:
            continue
        row_test_mask = (groups == heldout_idx) | ((groups < 0) & (control_folds == heldout_idx))
        row_train_mask = (~row_test_mask) & (groups != heldout_idx)
        if int(row_train_mask.sum()) < int(config.ebm_min_rows) or int(row_test_mask.sum()) < 100:
            continue
        X, kept_features, centers, scales = _sample_feature_matrix(frame, positions, fold_features, row_train_mask)
        if X.shape[1] < 2:
            continue
        design_groups = groups
        design_weights = weights
        design_control_folds = control_folds
        if use_timestamp_aggregate:
            X, design_groups, design_weights, design_control_folds = _aggregate_design_by_timestamp(
                X,
                position_timestamps,
                groups,
                weights,
                control_folds,
            )
        design_y = (design_groups >= 0).astype(np.int8)
        test_mask = (design_groups == heldout_idx) | (
            (design_groups < 0) & (design_control_folds == heldout_idx)
        )
        train_mask = (~test_mask) & (design_groups != heldout_idx)
        if int(train_mask.sum()) < int(config.ebm_min_rows) or int(test_mask.sum()) < 100:
            continue
        fold_scores = feature_scores.loc[feature_scores["breakout_key"].astype(str).isin(train_episode_keys)].copy()
        pairs_named = _candidate_pairs_for_ebm(kept_features, fold_scores, max_pairs=int(config.ebm_max_pairs))
        pairs = [
            (kept_features.index(left), kept_features.index(right))
            for left, right in pairs_named
            if left in kept_features and right in kept_features
        ]
        if not pairs:
            continue
        train_y = design_y[train_mask]
        test_y = design_y[test_mask]
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            continue
        try:
            main = ExplainableBoostingClassifier(
                feature_names=kept_features,
                interactions=0,
                max_bins=64,
                max_interaction_bins=32,
                outer_bags=2,
                inner_bags=0,
                learning_rate=0.04,
                max_rounds=int(config.ebm_max_rounds),
                early_stopping_rounds=25,
                validation_size=0.15,
                max_leaves=2,
                n_jobs=1,
                random_state=int(config.random_seed) + heldout_idx,
            )
            main.fit(X[train_mask], train_y, sample_weight=design_weights[train_mask])
            inter = ExplainableBoostingClassifier(
                feature_names=kept_features,
                interactions=pairs,
                max_bins=64,
                max_interaction_bins=32,
                outer_bags=2,
                inner_bags=0,
                learning_rate=0.04,
                max_rounds=int(config.ebm_max_rounds),
                early_stopping_rounds=25,
                validation_size=0.15,
                max_leaves=3,
                n_jobs=1,
                random_state=int(config.random_seed) + heldout_idx + 1000,
            )
            inter.fit(X[train_mask], train_y, sample_weight=design_weights[train_mask], init_score=main)
            p_main = main.predict_proba(X[test_mask])[:, 1]
            p_inter = inter.predict_proba(X[test_mask])[:, 1]
        except Exception as exc:
            _log(f"[ebm] fold skipped {layer} {strategy} {slice_name} episode={heldout_idx}: {type(exc).__name__}: {exc}")
            continue
        main_logloss = _binary_log_loss(test_y, p_main)
        inter_logloss = _binary_log_loss(test_y, p_inter)
        main_brier = _brier_score(test_y, p_main)
        inter_brier = _brier_score(test_y, p_inter)
        delta_logloss = main_logloss - inter_logloss
        delta_brier = main_brier - inter_brier
        term_importance = np.asarray(inter.term_importances(), dtype=np.float64)
        pair_term_indices = [
            idx
            for idx, term in enumerate(inter.term_features_)
            if len(term) == 2
        ]
        total_pair_importance = max(float(np.nansum(np.maximum(term_importance[pair_term_indices], 0.0))), 1e-12)
        test_groups = design_groups[test_mask]
        control_test = test_groups < 0
        episode_test = test_groups == heldout_idx
        for f_idx, feature in enumerate(kept_features):
            train_controls = train_mask & (design_groups < 0)
            train_episodes = train_mask & (design_groups >= 0)
            if int(train_controls.sum()) < 50 or int(train_episodes.sum()) < 50:
                continue
            cvals = X[train_controls, f_idx]
            evals = X[train_episodes, f_idx]
            direction = 1.0 if float(np.nanmedian(evals) - np.nanmedian(cvals)) >= 0 else -1.0
            threshold = float(np.nanpercentile(cvals, 85 if direction > 0 else 15))
            vals = X[test_mask, f_idx]
            harmful = vals >= threshold if direction > 0 else vals <= threshold
            threshold_rows.append(
                {
                    "strategy": strategy,
                    "layer": layer,
                    "slice": slice_name,
                    "heldout_breakout_key": heldout_key,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "direction": "high" if direction > 0 else "low",
                    "threshold_robust_z": threshold,
                    "harmful_region_row_count": int(harmful.sum()),
                    "harmful_region_episode_row_count": int(np.sum(harmful & episode_test)),
                    "harmful_region_control_row_count": int(np.sum(harmful & control_test)),
                    "false_alarm_rate_control": float(np.mean(harmful[control_test])) if bool(control_test.any()) else np.nan,
                    "main_logloss": main_logloss,
                    "interaction_logloss": inter_logloss,
                    "delta_logloss": delta_logloss,
                    "main_brier": main_brier,
                    "interaction_brier": inter_brier,
                    "delta_brier": delta_brier,
                }
            )
        for term_idx in pair_term_indices:
            i, j = inter.term_features_[term_idx]
            left = kept_features[i]
            right = kept_features[j]
            imp = float(max(term_importance[term_idx], 0.0))
            share = imp / total_pair_importance
            train_controls = train_mask & (design_groups < 0)
            train_episodes = train_mask & (design_groups >= 0)
            li_direction = 1.0 if float(np.nanmedian(X[train_episodes, i]) - np.nanmedian(X[train_controls, i])) >= 0 else -1.0
            rj_direction = 1.0 if float(np.nanmedian(X[train_episodes, j]) - np.nanmedian(X[train_controls, j])) >= 0 else -1.0
            li_threshold = float(np.nanpercentile(X[train_controls, i], 85 if li_direction > 0 else 15))
            rj_threshold = float(np.nanpercentile(X[train_controls, j], 85 if rj_direction > 0 else 15))
            test_x = X[test_mask]
            left_harm = test_x[:, i] >= li_threshold if li_direction > 0 else test_x[:, i] <= li_threshold
            right_harm = test_x[:, j] >= rj_threshold if rj_direction > 0 else test_x[:, j] <= rj_threshold
            harmful = left_harm & right_harm
            scores = np.asarray(inter.term_scores_[term_idx], dtype=np.float64)
            finite_scores = scores[np.isfinite(scores)]
            surface_sign = float(np.sign(np.nanmean(finite_scores))) if finite_scores.size else 0.0
            topology = float(np.nanstd(finite_scores) / max(abs(float(np.nanmean(finite_scores))), 1e-6)) if finite_scores.size else np.nan
            pair_rows.append(
                {
                    "strategy": strategy,
                    "layer": layer,
                    "slice": slice_name,
                    "heldout_breakout_key": heldout_key,
                    "pair": f"{left}__x__{right}",
                    "feature_i": left,
                    "feature_j": right,
                    "family_i": _feature_family(left),
                    "family_j": _feature_family(right),
                    "selected_in_fold": True,
                    "term_importance": imp,
                    "term_importance_share": share,
                    "main_logloss": main_logloss,
                    "interaction_logloss": inter_logloss,
                    "delta_logloss_model": delta_logloss,
                    "delta_logloss_pair_weighted": delta_logloss * share,
                    "main_brier": main_brier,
                    "interaction_brier": inter_brier,
                    "delta_brier_model": delta_brier,
                    "delta_brier_pair_weighted": delta_brier * share,
                    "harmful_region": (
                        f"{left} {'>=' if li_direction > 0 else '<='} {li_threshold:.3f}; "
                        f"{right} {'>=' if rj_direction > 0 else '<='} {rj_threshold:.3f}"
                    ),
                    "harmful_region_row_count": int(harmful.sum()),
                    "harmful_region_episode_row_count": int(np.sum(harmful & episode_test)),
                    "harmful_region_episode_count": int(np.any(harmful & episode_test)),
                    "harmful_region_control_row_count": int(np.sum(harmful & control_test)),
                    "false_alarm_rate_control": float(np.mean(harmful[control_test])) if bool(control_test.any()) else np.nan,
                    "surface_sign": surface_sign,
                    "surface_topology": topology,
                    "test_rows": int(test_mask.sum()),
                    "test_episode_rows": int(episode_test.sum()),
                    "test_control_rows": int(control_test.sum()),
                }
            )
    pair_frame = pd.DataFrame(pair_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    if not pair_frame.empty:
        grouped_rows: list[dict[str, object]] = []
        total_folds = max(1, len(set(pair_frame["heldout_breakout_key"].astype(str))))
        for pair, group in pair_frame.groupby("pair", sort=False):
            signs = np.sign(pd.to_numeric(group["surface_sign"], errors="coerce").to_numpy(dtype=np.float64))
            signs = signs[np.isfinite(signs) & (signs != 0)]
            sign_stability = max(float(np.mean(signs > 0)), float(np.mean(signs < 0))) if signs.size else 0.0
            grouped_rows.append(
                {
                    "strategy": strategy,
                    "layer": layer,
                    "slice": slice_name,
                    "pair": pair,
                    "feature_i": group["feature_i"].iloc[0],
                    "feature_j": group["feature_j"].iloc[0],
                    "family_i": group["family_i"].iloc[0],
                    "family_j": group["family_j"].iloc[0],
                    "loeo_selection_frequency": float(group["heldout_breakout_key"].nunique() / total_folds),
                    "loeo_delta_logloss_mean": float(pd.to_numeric(group["delta_logloss_pair_weighted"], errors="coerce").mean()),
                    "loeo_delta_brier_mean": float(pd.to_numeric(group["delta_brier_pair_weighted"], errors="coerce").mean()),
                    "term_importance_mean": float(pd.to_numeric(group["term_importance"], errors="coerce").mean()),
                    "harmful_region_example": group.sort_values("term_importance", ascending=False, kind="mergesort")["harmful_region"].iloc[0],
                    "harmful_region_row_count": int(pd.to_numeric(group["harmful_region_row_count"], errors="coerce").fillna(0).sum()),
                    "harmful_region_episode_row_count": int(pd.to_numeric(group["harmful_region_episode_row_count"], errors="coerce").fillna(0).sum()),
                    "harmful_region_episode_count": int(pd.to_numeric(group["harmful_region_episode_count"], errors="coerce").fillna(0).sum()),
                    "surface_sign_stability": sign_stability,
                    "surface_topology_stability": float(1.0 / (1.0 + pd.to_numeric(group["surface_topology"], errors="coerce").fillna(0.0).std())),
                    "false_alarm_rate_control_mean": float(pd.to_numeric(group["false_alarm_rate_control"], errors="coerce").mean()),
                    "fold_rows": int(len(group)),
                }
            )
        pair_frame = pd.DataFrame(grouped_rows).sort_values(
            ["loeo_selection_frequency", "loeo_delta_logloss_mean", "term_importance_mean"],
            ascending=False,
            kind="mergesort",
        )
    if not threshold_frame.empty:
        threshold_frame = (
            threshold_frame.groupby(["strategy", "layer", "slice", "feature", "feature_family"], sort=False)
            .agg(
                loeo_selection_frequency=("heldout_breakout_key", lambda s: float(s.nunique() / max(selected_episode_count, 1))),
                threshold_robust_z_median=("threshold_robust_z", "median"),
                high_direction_share=("direction", lambda s: float(np.mean(pd.Series(s).astype(str).eq("high")))),
                harmful_region_row_count=("harmful_region_row_count", "sum"),
                harmful_region_episode_row_count=("harmful_region_episode_row_count", "sum"),
                harmful_region_control_row_count=("harmful_region_control_row_count", "sum"),
                false_alarm_rate_control_mean=("false_alarm_rate_control", "mean"),
                delta_logloss_mean=("delta_logloss", "mean"),
                delta_brier_mean=("delta_brier", "mean"),
            )
            .reset_index()
            .sort_values(["loeo_selection_frequency", "delta_logloss_mean"], ascending=False, kind="mergesort")
        )
    _log(
        f"[ebm] {layer} {strategy} {slice_name}: "
        f"pairs={len(pair_frame)} thresholds={len(threshold_frame)} episodes={selected_episode_count}"
    )
    return pair_frame, threshold_frame


def _feature_family(name: str) -> str:
    lower = str(name).lower()
    if lower.startswith("q_"):
        return "url_quantile"
    if lower.startswith("autocorr_"):
        return "url_autocorr"
    if lower.startswith(("roll_slope_", "roll_accel_", "extreme_exposure_")):
        return "state_dynamics"
    if lower.startswith("ebm_state_"):
        return "ebm_threshold_state"
    if lower.startswith(("cov_w", "corr_w")):
        return "url_pair"
    if lower.startswith("xs_cov_"):
        return "cross_sectional_covariance"
    if lower.startswith("eig_"):
        return "url_eigen"
    if lower.startswith(("svd", "knn")):
        return "url_svd_knn"
    if lower.startswith(("url_", "url_asset__", "url_market__", "url_xs_z__", "url_sigreg__")):
        return "url_context"
    if "breadth" in lower:
        return "market_breadth"
    if "oi" in lower or "open_interest" in lower:
        return "open_interest"
    if "funding" in lower:
        return "funding"
    if "vol" in lower or "rv" in lower or "atr" in lower or "range" in lower:
        return "volatility_range"
    if "ret" in lower or "return" in lower or "momentum" in lower:
        return "return_momentum"
    if "liquidity" in lower or "volume" in lower or "amihud" in lower:
        return "liquidity_volume"
    return "other"


def _feature_portability(name: str) -> tuple[str, str]:
    lower = str(name).lower()
    global_prefixes = (
        "mkt_",
        "market_breadth_",
        "market_dispersion_",
        "mkt_ret_",
        "global_liquidity_",
        "xasset_mkt_",
        "pct_assets_",
        "median_",
    )
    if lower.startswith(global_prefixes):
        return "global_regime_broadcast", "already_cross_sectional_or_market_wide"
    raw_scale_tokens = (
        "log_quote_volume",
        "asset_volume_30d",
        "ob_top_liquidity_usd",
        "ob_depth_usd",
        "ob_buy_notional",
        "ob_sell_notional",
        "ob_notional_z",
        "qv",
    )
    if any(token in lower for token in raw_scale_tokens) and not any(
        token in lower
        for token in (
            "_z",
            "z_",
            "_rank",
            "rank_",
            "percentile",
            "to_qv",
            "_ratio",
            "_bps",
            "robust_z",
        )
    ):
        return "not_portable", "raw_quote_notional_or_depth_scale"
    portable_tokens = (
        "_z",
        "z_",
        "_resid",
        "peer_resid",
        "mkt_resid",
        "_rank",
        "rank_",
        "_pct",
        "percentile",
        "to_volume",
        "price_x_oi",
        "_x_funding",
        "asset_minus_mkt",
        "funding_per_hour",
        "bollinger_band_width",
        "ema50_slope",
        "efficiency",
        "high_vol_state",
        "ret",
        "amihud",
        "rvol",
        "vol_",
        "spread_bps",
        "spread_proxy",
        "spread_to_expected_move",
        "liquidity_ratio",
        "liquidity_stress_score",
        "liquidity_shock_z",
        "liquidity_divergence",
        "ob_available",
        "ob_stale_flag",
        "ob_update_gap_flag",
        "ob_snapshot_age",
        "ob_depth_l10_to_qv",
        "ob_depth_l20_to_qv",
        "ob_top_liquidity_to_qv",
        "depth_to_qv",
        "depth_ratio",
        "depth_decay",
        "notional_to_depth",
        "trade_size_to_l1_depth",
        "quote_volume_z",
        "relative_volume",
        "volume_percentile",
        "volume_depth_risk",
        "asset_p75_spread_bps",
        "asset_spread_decile",
    )
    if any(token in lower for token in portable_tokens):
        return "asset_portable", "normalized_residual_rate_or_dimensionless"
    return "not_portable", "raw_scale_or_unknown_asset_comparability"


def _asset_portable_features(features: Sequence[str]) -> list[str]:
    out: list[str] = []
    for feature in dict.fromkeys(str(col) for col in features if str(col)):
        kind, _reason = _feature_portability(feature)
        if kind == "asset_portable":
            out.append(feature)
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _direction_consistency(values: pd.Series) -> float:
    signs = np.sign(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64))
    signs = signs[np.isfinite(signs) & (signs != 0)]
    if signs.size == 0:
        return 0.0
    pos = float(np.mean(signs > 0))
    neg = float(np.mean(signs < 0))
    return max(pos, neg)


def _aggregate_feature_breakout_strength(
    feature_scores: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    config: AnalysisConfig,
) -> pd.DataFrame:
    if feature_scores.empty:
        return pd.DataFrame()
    required = {
        "feature",
        "breakout_key",
        "candidate_score",
        "episode_explanation_score",
        "episode_breakout_weight",
    }
    if not required.issubset(feature_scores.columns):
        return pd.DataFrame()
    work = feature_scores.copy()
    work["regime_candidate"] = work.get("regime_candidate", False).astype(bool)
    if "candidate_score_floor" in work.columns:
        score_floor = pd.to_numeric(work["candidate_score_floor"], errors="coerce").fillna(
            float(config.min_candidate_score_for_explanation)
        )
    else:
        score_floor = pd.Series(float(config.min_candidate_score_for_explanation), index=work.index)
    work["explains_breakout"] = work["regime_candidate"] & pd.to_numeric(
        work["candidate_score"],
        errors="coerce",
    ).ge(score_floor)
    rows: list[dict[str, object]] = []
    for key, group in work.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group = group.copy()
        episode_base = group.drop_duplicates("breakout_key")
        explained = group.loc[group["explains_breakout"]].copy()
        explained_episode = explained.drop_duplicates("breakout_key")
        total_breakouts = int(episode_base["breakout_key"].nunique())
        explained_breakouts = int(explained_episode["breakout_key"].nunique())
        total_weight = float(pd.to_numeric(episode_base["episode_breakout_weight"], errors="coerce").fillna(0.0).sum())
        explained_weight = float(pd.to_numeric(explained_episode["episode_breakout_weight"], errors="coerce").fillna(0.0).sum())
        weighted_sum = float(pd.to_numeric(explained["episode_explanation_score"], errors="coerce").fillna(0.0).sum())
        weighted_mean_candidate = _weighted_mean(group["candidate_score"], group["episode_breakout_weight"])
        weighted_mean_explained = (
            _weighted_mean(explained["candidate_score"], explained["episode_breakout_weight"])
            if not explained.empty
            else 0.0
        )
        breakout_weight_share = explained_weight / max(total_weight, 1e-12)
        repeat_factor = math.log1p(float(explained_breakouts))
        strength = (weighted_sum / max(total_weight, 1e-12)) * repeat_factor
        best = group.sort_values("episode_explanation_score", ascending=False, kind="mergesort").head(1)
        best_row = best.iloc[0] if not best.empty else pd.Series(dtype=object)
        out = {str(col): value for col, value in zip(group_cols, key, strict=False)}
        feature_name = str(out.get("feature", group["feature"].iloc[0]))
        out.update(
            {
                "feature_family": _feature_family(feature_name),
                "total_breakout_count_scored": total_breakouts,
                "explained_breakout_count": explained_breakouts,
                "explained_breakout_fraction": explained_breakouts / max(total_breakouts, 1),
                "total_breakout_weight_scored": total_weight,
                "explained_breakout_weight": explained_weight,
                "explained_breakout_weight_share": breakout_weight_share,
                "weighted_explanation_sum": weighted_sum,
                "weighted_candidate_score_mean": weighted_mean_candidate,
                "weighted_explained_candidate_score_mean": weighted_mean_explained,
                "breakout_explanatory_strength": strength,
                "median_shift_direction_consistency": _direction_consistency(explained["median_shift_robust_z"])
                if not explained.empty and "median_shift_robust_z" in explained
                else 0.0,
                "mean_ks_shift_explained": float(pd.to_numeric(explained.get("ks_shift"), errors="coerce").mean())
                if not explained.empty and "ks_shift" in explained
                else np.nan,
                "mean_period_auc_lift_explained": float(pd.to_numeric(explained.get("period_auc_lift"), errors="coerce").mean())
                if not explained.empty and "period_auc_lift" in explained
                else np.nan,
                "mean_relevance_score_explained": float(pd.to_numeric(explained.get("relevance_score"), errors="coerce").mean())
                if not explained.empty and "relevance_score" in explained
                else np.nan,
                "mean_harmfulness_score_explained": float(pd.to_numeric(explained.get("harmfulness_score"), errors="coerce").mean())
                if not explained.empty and "harmfulness_score" in explained
                else np.nan,
                "explained_duration_days": float(
                    pd.to_numeric(explained_episode.get("episode_duration_days"), errors="coerce").fillna(0.0).sum()
                )
                if not explained_episode.empty and "episode_duration_days" in explained_episode
                else 0.0,
                "deepest_hit_rate_surprise_z_explained": float(
                    pd.to_numeric(explained_episode.get("episode_min_hit_rate_surprise_z"), errors="coerce").min()
                )
                if not explained_episode.empty and "episode_min_hit_rate_surprise_z" in explained_episode
                else np.nan,
                "deepest_hit_rate_delta_explained": float(
                    pd.to_numeric(explained_episode.get("episode_min_hit_rate_delta"), errors="coerce").min()
                )
                if not explained_episode.empty and "episode_min_hit_rate_delta" in explained_episode
                else np.nan,
                "best_episode_explanation_score": _safe_float(best_row.get("episode_explanation_score")),
                "best_episode_candidate_score": _safe_float(best_row.get("candidate_score")),
                "best_episode_key": best_row.get("breakout_key", ""),
                "best_episode_start_day": best_row.get("episode_start_day", ""),
                "best_episode_end_day": best_row.get("episode_end_day", ""),
            }
        )
        if "strategy" not in group_cols:
            out["strategy_count"] = int(group.loc[group["explains_breakout"], "strategy"].nunique()) if "strategy" in group else 0
        if "layer" not in group_cols:
            out["layer_count"] = int(group.loc[group["explains_breakout"], "layer"].nunique()) if "layer" in group else 0
        if "slice" not in group_cols:
            out["slice_count"] = int(group.loc[group["explains_breakout"], "slice"].nunique()) if "slice" in group else 0
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "breakout_explanatory_strength",
            "explained_breakout_count",
            "explained_breakout_weight_share",
            "weighted_explained_candidate_score_mean",
        ],
        ascending=False,
        kind="mergesort",
    )


def _mixed_effect_feature_recurrence(
    feature_scores: pd.DataFrame,
    breakout_strength_global: pd.DataFrame,
    *,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if (
        not bool(config.mixed_effects_enabled)
        or feature_scores.empty
        or breakout_strength_global.empty
        or "breakout_key" not in feature_scores
    ):
        return pd.DataFrame()
    top_features = (
        breakout_strength_global.sort_values("breakout_explanatory_strength", ascending=False, kind="mergesort")[
            "feature"
        ]
        .astype(str)
        .head(int(config.mixed_effects_max_features))
        .tolist()
    )
    if not top_features:
        return pd.DataFrame()
    work = feature_scores.loc[feature_scores["feature"].astype(str).isin(top_features)].copy()
    if work.empty:
        return pd.DataFrame()
    work["candidate_score"] = pd.to_numeric(work["candidate_score"], errors="coerce")
    work = work.loc[work["candidate_score"].notna()]
    if work["breakout_key"].nunique() < 3 or work["feature"].nunique() < 2:
        return pd.DataFrame()
    grouped_fallback = (
        work.groupby("feature", sort=False)
        .agg(
            mixed_effect_status=("candidate_score", lambda _s: "fallback_insufficient_groups"),
            candidate_score_mean=("candidate_score", "mean"),
            candidate_score_median=("candidate_score", "median"),
            episode_count=("breakout_key", "nunique"),
            row_count=("candidate_score", "size"),
        )
        .reset_index()
    )
    try:
        import statsmodels.api as sm

        y = work["candidate_score"].to_numpy(dtype=np.float64)
        exog = pd.get_dummies(work["feature"].astype(str), dtype=float)
        if exog.shape[1] < 2:
            return grouped_fallback
        model = sm.MixedLM(y, exog.to_numpy(dtype=np.float64), groups=work["breakout_key"].astype(str).to_numpy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        params = np.asarray(fit.params[: exog.shape[1]], dtype=np.float64)
        bse = np.asarray(fit.bse[: exog.shape[1]], dtype=np.float64)
        rows: list[dict[str, object]] = []
        for idx, feature in enumerate(exog.columns.astype(str)):
            sub = work.loc[work["feature"].astype(str).eq(feature)]
            rows.append(
                {
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "mixed_effect_status": "fit",
                    "mixed_effect_coef": float(params[idx]),
                    "mixed_effect_se": float(bse[idx]) if idx < bse.size else np.nan,
                    "mixed_effect_z": float(params[idx] / bse[idx]) if idx < bse.size and bse[idx] > 1e-12 else np.nan,
                    "candidate_score_mean": float(sub["candidate_score"].mean()),
                    "candidate_score_median": float(sub["candidate_score"].median()),
                    "episode_count": int(sub["breakout_key"].nunique()),
                    "row_count": int(len(sub)),
                    "random_intercept_var": float(np.asarray(fit.cov_re).ravel()[0]) if np.asarray(fit.cov_re).size else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["mixed_effect_coef", "episode_count"],
            ascending=False,
            kind="mergesort",
        )
    except Exception as exc:
        grouped_fallback["mixed_effect_status"] = f"fallback:{type(exc).__name__}"
        grouped_fallback["feature_family"] = grouped_fallback["feature"].map(_feature_family)
        return grouped_fallback.sort_values(
            ["candidate_score_mean", "episode_count"],
            ascending=False,
            kind="mergesort",
        )


def _build_ebm_threshold_registry(ebm_thresholds: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    stable = _stable_ebm_threshold_rows(ebm_thresholds, config)
    if stable.empty:
        return pd.DataFrame()
    out = stable.copy()
    key_cols = [col for col in ["strategy", "layer", "slice", "feature", "feature_family"] if col in out.columns]
    metric_cols = [
        "loeo_selection_frequency",
        "threshold_robust_z_median",
        "high_direction_share",
        "harmful_region_row_count",
        "harmful_region_episode_row_count",
        "harmful_region_control_row_count",
        "false_alarm_rate_control_mean",
        "delta_logloss_mean",
        "delta_brier_mean",
        "threshold_registry_score",
    ]
    keep = [*key_cols, "threshold_direction", *[col for col in metric_cols if col in out.columns]]
    out = out[keep].copy()
    rank_groups = [col for col in ["layer", "strategy", "slice"] if col in out.columns]
    if rank_groups:
        out["strategy_threshold_rank"] = (
            out.sort_values(["threshold_registry_score"], ascending=False, kind="mergesort")
            .groupby(rank_groups)
            .cumcount()
            + 1
        )
    else:
        out["strategy_threshold_rank"] = np.arange(len(out), dtype=np.int64) + 1
    return out.sort_values(
        ["threshold_registry_score", "loeo_selection_frequency", "delta_logloss_mean"],
        ascending=False,
        kind="mergesort",
    )


def _feature_strategy_recurrence_scores(breakout_strength_by_head: pd.DataFrame) -> pd.DataFrame:
    if breakout_strength_by_head.empty:
        return pd.DataFrame()
    work = breakout_strength_by_head.copy()
    work["breakout_explanatory_strength"] = pd.to_numeric(
        work["breakout_explanatory_strength"],
        errors="coerce",
    ).fillna(0.0)
    work["explained_breakout_count"] = pd.to_numeric(
        work["explained_breakout_count"],
        errors="coerce",
    ).fillna(0.0)
    relevant = work.loc[
        work["breakout_explanatory_strength"].gt(0.0)
        & work["explained_breakout_count"].gt(0.0)
    ].copy()
    if relevant.empty:
        return pd.DataFrame()
    relevant["head_key"] = (
        relevant.get("layer", "").astype(str)
        + "|"
        + relevant.get("strategy", "").astype(str)
        + "|"
        + relevant.get("slice", "").astype(str)
    )
    rows: list[dict[str, object]] = []
    for feature, group in relevant.groupby("feature", sort=False):
        strength_sum = float(group["breakout_explanatory_strength"].sum())
        strategy_count = int(group["strategy"].astype(str).nunique()) if "strategy" in group else 0
        layer_count = int(group["layer"].astype(str).nunique()) if "layer" in group else 0
        slice_count = int(group["slice"].astype(str).nunique()) if "slice" in group else 0
        head_count = int(group["head_key"].nunique())
        explained_count = float(pd.to_numeric(group["explained_breakout_count"], errors="coerce").fillna(0.0).sum())
        recurrence_factor = math.log1p(strategy_count) * math.log1p(layer_count) * math.log1p(head_count)
        single_head_penalty = 0.35 if head_count <= 1 else 1.0
        rows.append(
            {
                "feature": feature,
                "feature_family": _feature_family(str(feature)),
                "strategy_count": strategy_count,
                "layer_count": layer_count,
                "slice_count": slice_count,
                "head_count": head_count,
                "explained_breakout_count_sum": explained_count,
                "strength_sum": strength_sum,
                "strength_mean": float(group["breakout_explanatory_strength"].mean()),
                "best_head_strength": float(group["breakout_explanatory_strength"].max()),
                "recurrence_factor": recurrence_factor,
                "single_head_penalty": single_head_penalty,
                "global_regime_gate_score": strength_sum * recurrence_factor * single_head_penalty,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["global_regime_gate_score", "strategy_count", "head_count", "strength_sum"],
        ascending=False,
        kind="mergesort",
    )


def _process_oof_file(
    *,
    layer: str,
    strategy: str,
    path: Path,
    config: AnalysisConfig,
    cfg_features: set[str],
    optional_feature_frame: pd.DataFrame | None,
    feature_columns: Sequence[str] = (),
    raw_breakout_screen: pd.DataFrame | None = None,
    ebm_threshold_registry: pd.DataFrame | None = None,
    previous_meta_parent_features: Sequence[str] = (),
) -> dict[str, pd.DataFrame | dict[str, object]]:
    _log(f"[load] {layer} {strategy}: {path}")
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns or "symbol" not in frame.columns:
        raise ValueError(f"{path} must include timestamp and symbol")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    frame = _filter_frame_by_analysis_period(frame, config)
    if frame.empty:
        raise ValueError(f"{path} has no rows after analysis_start_day={config.analysis_start_day!r}")
    frame = _merge_feature_frame(frame, optional_feature_frame)
    if bool(config.stream_feature_generation) and feature_columns:
        frame = _append_streamed_generated_features(
            frame,
            config=config,
            feature_columns=feature_columns,
            raw_breakout_screen=raw_breakout_screen
            if isinstance(raw_breakout_screen, pd.DataFrame)
            else pd.DataFrame(),
            previous_meta_parent_features=previous_meta_parent_features,
        )
    pred_col, label_col, pnl_col = _prediction_columns(frame, layer)
    if isinstance(ebm_threshold_registry, pd.DataFrame) and not ebm_threshold_registry.empty:
        threshold_parts: list[pd.DataFrame] = []
        for slice_name, _mask in _analysis_slices(frame, pred_col, config):
            threshold_features = _generate_ebm_threshold_state_features(
                frame,
                ebm_threshold_registry,
                layer=layer,
                strategy=strategy,
                slice_name=slice_name,
                config=config,
            )
            if not threshold_features.empty:
                threshold_parts.append(threshold_features)
        if threshold_parts:
            threshold_frame = pd.concat(threshold_parts, axis=1)
            threshold_frame = threshold_frame.loc[:, ~threshold_frame.columns.duplicated()]
            add_cols = [col for col in threshold_frame.columns if col not in frame.columns]
            if add_cols:
                frame = pd.concat([frame, threshold_frame[add_cols].astype(np.float32, copy=False)], axis=1)
                _log(
                    f"[features] appended EBM threshold state columns={len(add_cols)} "
                    f"{layer} {strategy}"
                )
    feature_cols, feature_quality = _select_candidate_features(frame, config, cfg_features)
    _log(
        f"[features] {layer} {strategy}: selected {len(feature_cols)} safe feature columns "
        f"from {len(feature_quality)} quality-passing candidates"
    )
    feature_cols, feature_quality, feature_redundancy = _spearman_redundancy_filter(
        frame,
        feature_cols,
        feature_quality,
        layer=layer,
        strategy=strategy,
        config=config,
    )
    if not feature_redundancy.empty:
        dropped_count = int(feature_redundancy["dropped_for_redundancy"].astype(bool).sum())
        cluster_count = int(feature_redundancy["redundancy_cluster_id"].nunique())
        _log(
            f"[features] {layer} {strategy}: redundancy filter kept={len(feature_cols)} "
            f"dropped={dropped_count} clusters={cluster_count} "
            f"threshold={config.redundancy_abs_spearman_threshold:.3f}"
        )
    feature_arrays = {
        feature: _numeric_array(frame[feature])
        for feature in feature_cols
        if feature in frame.columns
    }
    pred_all = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    y_all = pd.to_numeric(frame[label_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    timestamp_feature_frame = _build_timestamp_feature_frame(frame, feature_cols)
    if not timestamp_feature_frame.empty:
        _log(
            f"[features] {layer} {strategy}: timestamp feature matrix "
            f"timestamps={len(timestamp_feature_frame)} cols={timestamp_feature_frame.shape[1]}"
        )

    all_outputs: dict[str, list[pd.DataFrame]] = {
        "bad_windows": [],
        "episodes": [],
        "slice_support": [],
        "feature_scores": [],
        "covariance_autocorr": [],
        "ebm_pair_interactions": [],
        "ebm_feature_thresholds": [],
        "threshold_calibration": [],
    }
    for slice_name, mask in _analysis_slices(frame, pred_col, config):
        support = _slice_support_summary(
            frame,
            strategy=strategy,
            layer=layer,
            slice_name=slice_name,
            pred_col=pred_col,
            label_col=label_col,
            mask=mask,
            config=config,
        )
        all_outputs["slice_support"].append(support)
        detected = _detect_bad_windows_for_slice(
            frame,
            pred_col=pred_col,
            label_col=label_col,
            pnl_col=pnl_col,
            mask=mask,
            config=config,
        )
        if isinstance(detected, tuple):
            bad, calibration = detected
        else:
            bad = detected
            calibration = pd.DataFrame()
        if not calibration.empty:
            calibration = calibration.copy()
            calibration.insert(0, "strategy", strategy)
            calibration.insert(1, "layer", layer)
            calibration.insert(2, "slice", slice_name)
        all_outputs["threshold_calibration"].append(calibration)
        if not bad.empty:
            bad = bad.copy()
            bad.insert(0, "strategy", strategy)
            bad.insert(1, "layer", layer)
            bad.insert(2, "slice", slice_name)
        all_outputs["bad_windows"].append(bad)
        episodes = _merge_bad_windows(bad, strategy=strategy, layer=layer, slice_name=slice_name)
        if not episodes.empty:
            metrics = [
                _performance_metrics_for_episode(
                    frame,
                    episode=row,
                    pred_col=pred_col,
                    label_col=label_col,
                    pnl_col=pnl_col,
                )
                for _, row in episodes.iterrows()
            ]
            metric_frame = pd.DataFrame(metrics)
            episodes = pd.concat([episodes.reset_index(drop=True), metric_frame.reset_index(drop=True)], axis=1)
            episodes["episode_breakout_weight"] = [
                _episode_breakout_weight(row, config)
                for _, row in episodes.iterrows()
            ]
        all_outputs["episodes"].append(episodes)
        _log(
            f"[breaks] {layer} {strategy} {slice_name}: "
            f"{len(bad)} bad windows, {len(episodes)} merged episodes"
        )
        slice_score_parts: list[pd.DataFrame] = []
        for _, episode in episodes.iterrows():
            scores = _feature_scores_for_episode(
                frame,
                strategy=strategy,
                layer=layer,
                slice_name=slice_name,
                episode=episode,
                features=feature_cols,
                pred_col=pred_col,
                label_col=label_col,
                config=config,
                feature_arrays=feature_arrays,
                pred_all=pred_all,
                y_all=y_all,
            )
            if not scores.empty:
                all_outputs["feature_scores"].append(scores)
                slice_score_parts.append(scores)
            cov = _episode_covariance_autocorr(
                frame,
                strategy=strategy,
                layer=layer,
                slice_name=slice_name,
                episode=episode,
                feature_scores=scores,
                features=feature_cols,
                config=config,
                timestamp_feature_frame=timestamp_feature_frame,
            )
            if not cov.empty:
                all_outputs["covariance_autocorr"].append(cov)
        if slice_score_parts and not episodes.empty:
            slice_scores = pd.concat(slice_score_parts, ignore_index=True)
            ebm_pairs, ebm_thresholds = _episode_ebm_interaction_diagnostics(
                frame,
                strategy=strategy,
                layer=layer,
                slice_name=slice_name,
                episodes=episodes,
                feature_scores=slice_scores,
                slice_mask=mask,
                config=config,
            )
            if not ebm_pairs.empty:
                all_outputs["ebm_pair_interactions"].append(ebm_pairs)
            if not ebm_thresholds.empty:
                all_outputs["ebm_feature_thresholds"].append(ebm_thresholds)

    manifest = {
        "layer": layer,
        "strategy": strategy,
        "path": str(path),
        "rows": int(len(frame)),
        "timestamp_min": str(frame["timestamp"].min()),
        "timestamp_max": str(frame["timestamp"].max()),
        "prediction_col": pred_col,
        "label_col": label_col,
        "pnl_col": pnl_col,
        "selected_feature_count": int(len(feature_cols)),
        "selected_features": list(feature_cols),
        "candidate_quality_rows": int(len(feature_quality)),
        "redundancy_rows": int(len(feature_redundancy)),
        "redundancy_dropped_count": int(feature_redundancy["dropped_for_redundancy"].astype(bool).sum())
        if not feature_redundancy.empty and "dropped_for_redundancy" in feature_redundancy
        else 0,
    }
    return {
        "manifest": manifest,
        "feature_quality": feature_quality.assign(strategy=strategy, layer=layer)
        if not feature_quality.empty
        else pd.DataFrame(),
        "feature_redundancy": feature_redundancy,
        **{
            name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            for name, parts in all_outputs.items()
        },
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame.to_csv(path, index=False)
        return str(path)
    frame.to_csv(path, index=False)
    return str(path)


def _covered_days_count(frame: pd.DataFrame, *, start_col: str = "start_day", end_col: str = "end_day") -> int:
    return len(_covered_days_set(frame, start_col=start_col, end_col=end_col))


def _covered_days_set(
    frame: pd.DataFrame,
    *,
    start_col: str = "start_day",
    end_col: str = "end_day",
) -> set[pd.Timestamp]:
    if frame.empty or start_col not in frame or end_col not in frame:
        return set()
    days: set[pd.Timestamp] = set()
    for _, row in frame.iterrows():
        start = pd.Timestamp(row[start_col])
        end = pd.Timestamp(row[end_col])
        if pd.isna(start) or pd.isna(end):
            continue
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        for day in pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC"):
            days.add(day)
    return days


def _parse_day_set(value: object) -> set[pd.Timestamp]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    days: set[pd.Timestamp] = set()
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        day = pd.to_datetime(part, utc=True, errors="coerce")
        if pd.isna(day):
            continue
        days.add(pd.Timestamp(day).floor("D"))
    return days


def _severity_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not bool(mask.any()):
        return np.nan
    return float(np.average(vals[mask], weights=w[mask]))


def _aggregate_breakout_head_summary(
    slice_support: pd.DataFrame,
    bad_windows: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    if slice_support.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["layer", "strategy", "slice"]
    bad_grouped = {key: group for key, group in bad_windows.groupby(group_cols, sort=False)} if not bad_windows.empty else {}
    episode_grouped = {key: group for key, group in episodes.groupby(group_cols, sort=False)} if not episodes.empty else {}
    for _, support in slice_support.drop_duplicates(group_cols).iterrows():
        key = tuple(support[col] for col in group_cols)
        bad = bad_grouped.get(key, pd.DataFrame())
        ep = episode_grouped.get(key, pd.DataFrame())
        eligible_days = int(_safe_float(support.get("eligible_support_days"), 0.0))
        support_days = int(_safe_float(support.get("support_days"), 0.0))
        eligible_day_set = _parse_day_set(support.get("eligible_support_day_list", ""))
        if not eligible_day_set and eligible_days > 0:
            # Backward-compatible fallback for older support frames that only
            # have the count. The share remains count-based, but new reports
            # include the explicit eligible day list and use true intersections.
            bad_day_count = min(_covered_days_count(bad), eligible_days)
            episode_day_count = min(_covered_days_count(ep), eligible_days)
        else:
            bad_day_count = len(_covered_days_set(bad) & eligible_day_set)
            episode_day_count = len(_covered_days_set(ep) & eligible_day_set)
        window_severity = pd.to_numeric(
            bad.get("window_severity", pd.Series(dtype=float)),
            errors="coerce",
        )
        if len(window_severity) != len(bad):
            window_severity = pd.Series(np.ones(len(bad), dtype=np.float64), index=bad.index)
        episode_weight = pd.to_numeric(
            ep.get("episode_breakout_weight", ep.get("window_severity_sum", pd.Series(dtype=float))),
            errors="coerce",
        )
        if len(episode_weight) != len(ep):
            episode_weight = pd.Series(np.ones(len(ep), dtype=np.float64), index=ep.index)
        rows.append(
            {
                "layer": support["layer"],
                "strategy": support["strategy"],
                "slice": support["slice"],
                "support_days": support_days,
                "eligible_support_days": eligible_days,
                "support_rows": int(_safe_float(support.get("support_rows"), 0.0)),
                "bad_window_count": int(len(bad)),
                "episode_count": int(len(ep)),
                "bad_day_count": int(bad_day_count),
                "episode_day_count": int(episode_day_count),
                "bad_day_share": bad_day_count / max(eligible_days, 1),
                "episode_day_share": episode_day_count / max(eligible_days, 1),
                "window_severity_sum": float(window_severity.fillna(0.0).sum()) if not bad.empty else 0.0,
                "window_severity_mean": float(window_severity.mean()) if not bad.empty else np.nan,
                "window_severity_p95": float(window_severity.quantile(0.95)) if not bad.empty else np.nan,
                "window_severity_weighted_hr_delta": _severity_weighted_mean(
                    bad.get("hit_rate_delta", pd.Series(dtype=float)),
                    window_severity,
                ),
                "window_severity_weighted_surprise_z": _severity_weighted_mean(
                    bad.get("hit_rate_surprise_z", pd.Series(dtype=float)),
                    window_severity,
                ),
                "episode_breakout_weight_sum": float(episode_weight.fillna(0.0).sum()) if not ep.empty else 0.0,
                "episode_breakout_weight_mean": float(episode_weight.mean()) if not ep.empty else np.nan,
                "episode_breakout_weight_p95": float(episode_weight.quantile(0.95)) if not ep.empty else np.nan,
                "episode_weighted_hr_delta": _severity_weighted_mean(
                    ep.get("episode_hit_rate_delta", pd.Series(dtype=float)),
                    episode_weight,
                ),
                "episode_weighted_surprise_z": _severity_weighted_mean(
                    ep.get("episode_hit_rate_surprise_z", pd.Series(dtype=float)),
                    episode_weight,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "strategy", "slice"], kind="mergesort")


def _selected_features_long_frame(manifests: Sequence[dict[str, object]], config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    slice_name = f"top{int(round(float(config.rank_frac) * 100))}" if bool(config.top_rank_slice_only) else "all_selected_slices"
    for manifest in manifests:
        features = manifest.get("selected_features", [])
        if not isinstance(features, list):
            continue
        for rank, feature in enumerate(features, start=1):
            rows.append(
                {
                    "layer": manifest.get("layer", ""),
                    "strategy": manifest.get("strategy", ""),
                    "slice": slice_name,
                    "feature_rank": rank,
                    "feature": str(feature),
                    "feature_family": _feature_family(str(feature)),
                }
            )
    return pd.DataFrame(rows)


def run_analysis(config: AnalysisConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache_cleanup_start = _cleanup_generated_transform_caches(config)
    files = _discover_oof_files(config)
    if not files:
        raise FileNotFoundError(f"No OOF files found under {_artifact_root(config)}")
    _log(f"[start] processing {len(files)} OOF files from {_artifact_root(config)}")
    cfg_features = _config_feature_name_set()
    feature_store_frame = None
    ebm_threshold_registry = _load_ebm_threshold_registry(config.ebm_threshold_registry)
    if not ebm_threshold_registry.empty:
        stable_registry = _stable_ebm_threshold_rows(ebm_threshold_registry, config)
        _log(
            f"[ebm-registry] loaded {len(ebm_threshold_registry)} rows "
            f"stable={len(stable_registry)} path={config.ebm_threshold_registry}"
        )
        ebm_threshold_registry = stable_registry
    raw_breakout_screen = pd.DataFrame()
    breakout_generation_episodes = pd.DataFrame()
    feature_columns = _load_feature_columns_json(config.feature_columns_json)
    previous_meta_parent_map, previous_meta_parent_report = _load_previous_meta_parent_features(
        config.previous_meta_parent_report,
        config=config,
    )
    before_breakout_structure = len(feature_columns)
    feature_columns = list(
        dict.fromkeys([*feature_columns, *BREAKOUT_STRUCTURE_FEATURE_COLUMNS])
    )
    _log(
        f"[features] included compact breakout/range primitives: "
        f"added={len(feature_columns) - before_breakout_structure} "
            f"total_requested={len(feature_columns)} "
            f"groups={len(BREAKOUT_STRUCTURE_FEATURE_GROUPS)}"
    )
    if previous_meta_parent_map:
        previous_parent_union = list(
            dict.fromkeys(
                feature
                for features in previous_meta_parent_map.values()
                for feature in features
                if _is_previous_meta_parent_raw_feature(feature)
            )
        )
        before = len(feature_columns)
        feature_columns = list(dict.fromkeys([*feature_columns, *previous_parent_union]))
        _log(
            f"[previous-meta-parents] added raw parent hydration columns="
            f"{len(feature_columns) - before} union={len(previous_parent_union)} "
            f"heads={len(previous_meta_parent_map)}"
        )
    if bool(config.include_all_feature_store_columns):
        store_columns = _available_safe_feature_store_columns(config.feature_store_dir)
        before = len(feature_columns)
        feature_columns = list(dict.fromkeys([*feature_columns, *store_columns]))
        _log(
            f"[features] included all safe feature-store columns: "
            f"added={len(feature_columns) - before} total_requested={len(feature_columns)}"
        )
    if bool(config.include_config_liquidity_features):
        liquidity_columns = _configured_liquidity_execution_feature_columns()
        before = len(feature_columns)
        feature_columns = list(dict.fromkeys([*feature_columns, *liquidity_columns]))
        _log(
            f"[features] included configured liquidity/execution features: "
            f"added={len(feature_columns) - before} total_requested={len(feature_columns)}"
        )
    if feature_columns and config.feature_store_dir is not None:
        _log(
            f"[features] hydrating {len(feature_columns)} selected unsupervised-regime "
            f"feature columns from {config.feature_store_dir}"
        )
        oof_keys = _collect_oof_row_keys(files, config=config)
        _log(
            f"[features] collected OOF row keys rows={len(oof_keys)} "
            f"timestamps={oof_keys['timestamp'].nunique() if not oof_keys.empty else 0} "
            f"symbols={oof_keys['symbol'].nunique() if not oof_keys.empty else 0}"
        )
        feature_store_frame = _hydrate_feature_store_for_keys(
            config.feature_store_dir,
            oof_keys,
            feature_columns,
            generate_url_composites=False,
            cache_dir=config.transform_cache_dir,
            cache_enabled=bool(config.transform_cache_enabled),
            refresh_cache=bool(config.refresh_transform_cache),
            cache_compression=config.parquet_cache_compression,
        )
        if (
            feature_store_frame is not None
            and bool(config.generate_url_composites)
        ):
            generated = pd.DataFrame(index=feature_store_frame.index)
            requested_final = list(dict.fromkeys(str(col) for col in feature_columns if str(col)))
            primitive_sources = [
                col
                for col in _infer_url_primitive_sources(requested_final)
                if col in feature_store_frame.columns
            ]
            if bool(config.breakout_exploration_enabled):
                breakout_generation_episodes = _collect_breakout_episodes_for_generation(files, config=config)
                raw_breakout_screen = _score_raw_features_against_breakouts(
                    feature_store_frame,
                    breakout_generation_episodes,
                    primitive_sources,
                    config=config,
                )
            if bool(config.stream_feature_generation):
                _log(
                    "[features] streaming generated URL/breakout operators per OOF head; "
                    "global feature-store panel remains narrow"
                )
            elif bool(config.breakout_exploration_enabled):
                generated = _generate_breakout_exploration_composites(
                    feature_store_frame,
                    raw_breakout_screen,
                    config=config,
                )
            else:
                generated = _generate_selected_url_composites(
                    feature_store_frame,
                    requested_final,
                    primitive_sources,
                )
            if generated is not None and not generated.empty:
                add_cols = [col for col in generated.columns if col not in feature_store_frame.columns]
                feature_store_frame = pd.concat([feature_store_frame, generated[add_cols]], axis=1)
                _log(
                    f"[features] appended generated URL columns={len(add_cols)} "
                    f"total_feature_cols={len(feature_store_frame.columns) - 2}"
                )
            if not bool(config.breakout_exploration_enabled):
                missing_selected = [
                    col
                    for col in requested_final
                    if col not in {"timestamp", "symbol"} and col not in feature_store_frame.columns
                ]
                if missing_selected:
                    _log(
                        f"[features] selected URL columns still unavailable={len(missing_selected)} "
                        f"sample={missing_selected[:12]}"
                    )
    optional_feature_frame = _load_optional_feature_frame(config.feature_frame)
    if optional_feature_frame is not None:
        _log(f"[features] loaded optional feature frame {config.feature_frame} rows={len(optional_feature_frame)}")
    regime_feature_frame = _load_regime_feature_artifact_dir(config.regime_feature_artifact_dir)
    if regime_feature_frame is not None:
        _log(
            f"[features] loaded unsupervised-regime feature artifact "
            f"{config.regime_feature_artifact_dir} rows={len(regime_feature_frame)} "
            f"cols={len(regime_feature_frame.columns) - 2}"
        )
    external_feature_frame = _combine_feature_frames(optional_feature_frame, regime_feature_frame, feature_store_frame)
    if bool(config.refresh_transform_cache) and bool(config.stream_feature_generation):
        _refresh_generated_transform_caches(config)

    manifests: list[dict[str, object]] = []
    feature_quality_parts: list[pd.DataFrame] = []
    feature_redundancy_parts: list[pd.DataFrame] = []
    bad_window_parts: list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []
    slice_support_parts: list[pd.DataFrame] = []
    feature_score_parts: list[pd.DataFrame] = []
    cov_parts: list[pd.DataFrame] = []
    ebm_pair_parts: list[pd.DataFrame] = []
    ebm_threshold_parts: list[pd.DataFrame] = []
    threshold_calibration_parts: list[pd.DataFrame] = []
    for layer, strategy, path in files:
        previous_parent_features = _previous_meta_parent_features_for_head(
            previous_meta_parent_map,
            layer=layer,
            strategy=strategy,
            config=config,
        )
        if previous_parent_features:
            _log(
                f"[previous-meta-parents] {layer} {strategy}: "
                f"using parents={len(previous_parent_features)}"
            )
        result = _process_oof_file(
            layer=layer,
            strategy=strategy,
            path=path,
            config=config,
            cfg_features=cfg_features,
            optional_feature_frame=external_feature_frame,
            feature_columns=feature_columns,
            raw_breakout_screen=raw_breakout_screen,
            ebm_threshold_registry=ebm_threshold_registry,
            previous_meta_parent_features=previous_parent_features,
        )
        manifests.append(result["manifest"])  # type: ignore[arg-type]
        for key, target in (
            ("feature_quality", feature_quality_parts),
            ("feature_redundancy", feature_redundancy_parts),
            ("bad_windows", bad_window_parts),
            ("episodes", episode_parts),
            ("slice_support", slice_support_parts),
            ("feature_scores", feature_score_parts),
            ("covariance_autocorr", cov_parts),
            ("ebm_pair_interactions", ebm_pair_parts),
            ("ebm_feature_thresholds", ebm_threshold_parts),
            ("threshold_calibration", threshold_calibration_parts),
        ):
            frame = result[key]
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                target.append(frame)

    feature_quality = pd.concat(feature_quality_parts, ignore_index=True) if feature_quality_parts else pd.DataFrame()
    feature_redundancy = (
        pd.concat(feature_redundancy_parts, ignore_index=True)
        if feature_redundancy_parts
        else pd.DataFrame()
    )
    bad_windows = pd.concat(bad_window_parts, ignore_index=True) if bad_window_parts else pd.DataFrame()
    episodes = pd.concat(episode_parts, ignore_index=True) if episode_parts else pd.DataFrame()
    slice_support = pd.concat(slice_support_parts, ignore_index=True) if slice_support_parts else pd.DataFrame()
    feature_scores = pd.concat(feature_score_parts, ignore_index=True) if feature_score_parts else pd.DataFrame()
    cov = pd.concat(cov_parts, ignore_index=True) if cov_parts else pd.DataFrame()
    ebm_pairs = pd.concat(ebm_pair_parts, ignore_index=True) if ebm_pair_parts else pd.DataFrame()
    ebm_thresholds = pd.concat(ebm_threshold_parts, ignore_index=True) if ebm_threshold_parts else pd.DataFrame()
    threshold_calibration = (
        pd.concat(threshold_calibration_parts, ignore_index=True)
        if threshold_calibration_parts
        else pd.DataFrame()
    )
    breakout_strength_by_head = _aggregate_feature_breakout_strength(
        feature_scores,
        group_cols=["layer", "strategy", "slice", "feature"],
        config=config,
    )
    breakout_strength_global = _aggregate_feature_breakout_strength(
        feature_scores,
        group_cols=["feature"],
        config=config,
    )
    mixed_effects = _mixed_effect_feature_recurrence(
        feature_scores,
        breakout_strength_global,
        config=config,
    )
    ebm_threshold_registry_out = _build_ebm_threshold_registry(ebm_thresholds, config)
    strategy_recurrence = _feature_strategy_recurrence_scores(breakout_strength_by_head)
    breakout_head_summary = _aggregate_breakout_head_summary(slice_support, bad_windows, episodes)
    selected_features_by_head = _selected_features_long_frame(manifests, config)
    top_candidates = (
        feature_scores.loc[feature_scores["regime_candidate"].astype(bool)]
        .sort_values(["episode_explanation_score", "candidate_score", "shift_score"], ascending=False, kind="mergesort")
        if not feature_scores.empty and "regime_candidate" in feature_scores
        else pd.DataFrame()
    )

    paths = {
        "manifest": _write_frame(pd.DataFrame(manifests), config.output_dir / "file_manifest.csv"),
        "feature_quality": _write_frame(feature_quality, config.output_dir / "feature_quality.csv"),
        "feature_redundancy": _write_frame(
            feature_redundancy,
            config.output_dir / "feature_redundancy_clusters.csv",
        ),
        "bad_3d_windows": _write_frame(bad_windows, config.output_dir / "bad_3d_windows.csv"),
        "bad_window_threshold_calibration": _write_frame(
            threshold_calibration,
            config.output_dir / "bad_window_threshold_calibration.csv",
        ),
        "bad_performance_episodes": _write_frame(episodes, config.output_dir / "bad_performance_episodes.csv"),
        "breakout_head_summary": _write_frame(
            breakout_head_summary,
            config.output_dir / "breakout_head_summary.csv",
        ),
        "selected_features_by_head": _write_frame(
            selected_features_by_head,
            config.output_dir / "selected_features_by_head.csv",
        ),
        "feature_shift_relevance_harmfulness": _write_frame(
            feature_scores,
            config.output_dir / "feature_shift_relevance_harmfulness.csv",
        ),
        "top_regime_candidates": _write_frame(top_candidates, config.output_dir / "top_regime_candidates.csv"),
        "feature_breakout_explanatory_strength_by_head": _write_frame(
            breakout_strength_by_head,
            config.output_dir / "feature_breakout_explanatory_strength_by_head.csv",
        ),
        "feature_breakout_explanatory_strength_global": _write_frame(
            breakout_strength_global,
            config.output_dir / "feature_breakout_explanatory_strength_global.csv",
        ),
        "breakout_operator_generation_episodes": _write_frame(
            breakout_generation_episodes,
            config.output_dir / "breakout_operator_generation_episodes.csv",
        ),
        "raw_breakout_feature_screen": _write_frame(
            raw_breakout_screen,
            config.output_dir / "raw_breakout_feature_screen.csv",
        ),
        "previous_meta_parent_features": _write_frame(
            previous_meta_parent_report,
            config.output_dir / "previous_meta_parent_features.csv",
        ),
        "episode_feature_covariance_autocorr": _write_frame(
            cov,
            config.output_dir / "episode_feature_covariance_autocorr.csv",
        ),
        "ebm_pair_interaction_diagnostics": _write_frame(
            ebm_pairs,
            config.output_dir / "ebm_pair_interaction_diagnostics.csv",
        ),
        "ebm_feature_threshold_diagnostics": _write_frame(
            ebm_thresholds,
            config.output_dir / "ebm_feature_threshold_diagnostics.csv",
        ),
        "ebm_threshold_registry": _write_frame(
            ebm_threshold_registry_out,
            config.output_dir / "ebm_threshold_registry.csv",
        ),
        "feature_strategy_recurrence": _write_frame(
            strategy_recurrence,
            config.output_dir / "feature_strategy_recurrence.csv",
        ),
        "mixed_effect_feature_recurrence": _write_frame(
            mixed_effects,
            config.output_dir / "mixed_effect_feature_recurrence.csv",
        ),
    }
    summary = {
        "config": asdict(config),
        "files_processed": len(files),
        "bad_window_count": int(len(bad_windows)),
        "bad_window_threshold_calibration_rows": int(len(threshold_calibration)),
        "episode_count": int(len(episodes)),
        "feature_score_rows": int(len(feature_scores)),
        "feature_redundancy_rows": int(len(feature_redundancy)),
        "feature_redundancy_dropped_rows": int(feature_redundancy["dropped_for_redundancy"].astype(bool).sum())
        if not feature_redundancy.empty and "dropped_for_redundancy" in feature_redundancy
        else 0,
        "top_regime_candidate_rows": int(len(top_candidates)),
        "breakout_head_summary_rows": int(len(breakout_head_summary)),
        "selected_features_by_head_rows": int(len(selected_features_by_head)),
        "feature_breakout_strength_by_head_rows": int(len(breakout_strength_by_head)),
        "feature_breakout_strength_global_rows": int(len(breakout_strength_global)),
        "breakout_operator_generation_episode_rows": int(len(breakout_generation_episodes)),
        "raw_breakout_feature_screen_rows": int(len(raw_breakout_screen)),
        "previous_meta_parent_feature_rows": int(len(previous_meta_parent_report)),
        "previous_meta_parent_head_count": int(len(previous_meta_parent_map)),
        "covariance_autocorr_rows": int(len(cov)),
        "ebm_pair_interaction_rows": int(len(ebm_pairs)),
        "ebm_feature_threshold_rows": int(len(ebm_thresholds)),
        "ebm_threshold_registry_rows": int(len(ebm_threshold_registry_out)),
        "feature_strategy_recurrence_rows": int(len(strategy_recurrence)),
        "mixed_effect_feature_recurrence_rows": int(len(mixed_effects)),
        "cache_cleanup_start": cache_cleanup_start,
        "cache_cleanup_end": _cleanup_generated_transform_caches(config),
        "paths": paths,
    }
    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _log(f"[done] wrote diagnostics to {config.output_dir}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> AnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--artifact-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--feature-frame", default=None, help="Optional parquet with timestamp/symbol plus feature columns to merge.")
    parser.add_argument(
        "--feature-store-dir",
        default="",
        help="Optional data_perp/features/<run_id> directory used to hydrate selected feature columns for OOF row keys.",
    )
    parser.add_argument(
        "--feature-run-id",
        default="",
        help="Shortcut for --feature-store-dir data_perp/features/<feature-run-id>.",
    )
    parser.add_argument(
        "--feature-columns-json",
        default="",
        help="JSON list or {'features': [...]} from unsupervised_regime_learning final_feature_columns.json.",
    )
    parser.add_argument(
        "--include-all-feature-store-columns",
        action="store_true",
        help=(
            "Hydrate every safe available column from the feature-store parquet schemas. "
            "This is broader than the URL/config-liquidity set and can be substantially heavier."
        ),
    )
    parser.add_argument(
        "--include-config-liquidity-features",
        action="store_true",
        help=(
            "Also hydrate registered liquidity/execution features from config.py "
            "so spread/depth/volume proxies can participate in breakout diagnostics."
        ),
    )
    parser.add_argument(
        "--disable-streamed-feature-generation",
        action="store_true",
        help=(
            "Generate URL/breakout operator columns on the global OOF-key panel "
            "instead of streaming them per OOF head. Higher memory; kept for reproducibility."
        ),
    )
    parser.add_argument(
        "--disable-transform-cache",
        action="store_true",
        help="Disable local Parquet caching of streamed generated transform columns.",
    )
    parser.add_argument(
        "--disable-generated-transform-cache",
        action="store_true",
        help=(
            "Keep the hydrated feature-store cache, but do not read/write the large "
            "per-head generated_transforms_* Parquet caches."
        ),
    )
    parser.add_argument(
        "--refresh-transform-cache",
        action="store_true",
        help="Regenerate streamed transform columns even when a matching local cache exists.",
    )
    parser.add_argument(
        "--transform-cache-dir",
        default="",
        help=(
            "Directory for reusable streamed transform Parquet cache. Defaults to "
            "data_root/reports/performance_regime_break_transform_cache."
        ),
    )
    parser.add_argument(
        "--parquet-cache-compression",
        default=DEFAULT_PARQUET_CACHE_COMPRESSION,
        help="Compression codec for future Parquet cache writes. Defaults to zstd.",
    )
    parser.add_argument(
        "--generated-transform-cache-ttl-days",
        type=float,
        default=0.0,
        help=(
            "Delete generated_transforms_* caches older than this many days, except the latest kept files. "
            "0 deletes all non-kept generated caches on cleanup."
        ),
    )
    parser.add_argument(
        "--generated-transform-cache-keep-last-n",
        type=int,
        default=1,
        help="Always keep this many newest generated_transforms_* cache files even if older than TTL.",
    )
    parser.add_argument(
        "--generated-transform-cache-max-rows",
        type=int,
        default=DEFAULT_GENERATED_TRANSFORM_CACHE_MAX_ROWS,
        help=(
            "Maximum projected rows allowed in the single generated-transform cache before "
            "new appends are skipped. 0 disables this row cap."
        ),
    )
    parser.add_argument(
        "--generated-transform-cache-max-bytes",
        type=int,
        default=DEFAULT_GENERATED_TRANSFORM_CACHE_MAX_BYTES,
        help=(
            "Maximum existing generated-transform cache size in bytes before new appends "
            "are skipped. 0 disables this size cap."
        ),
    )
    parser.add_argument(
        "--previous-meta-parent-report",
        default="",
        help=(
            "Previous report file or directory used to seed the next run with top raw parent "
            "features from meta heads only. Accepts feature_breakout_explanatory_strength_by_head "
            "or selected_features_by_head reports."
        ),
    )
    parser.add_argument(
        "--previous-meta-parent-top-n",
        type=int,
        default=50,
        help="Top raw parent features to keep per previous meta head for regime-transform generation.",
    )
    parser.add_argument(
        "--previous-meta-parent-slice",
        default="top30",
        help="Previous report slice used for meta parent feature selection. Default: top30.",
    )
    parser.add_argument(
        "--disable-previous-meta-parent-transforms",
        action="store_true",
        help="Do not generate regime-level transforms from --previous-meta-parent-report.",
    )
    parser.add_argument(
        "--disable-url-composite-generation",
        action="store_true",
        help="Only hydrate raw feature-store columns; do not generate selected URL operator/SVD composites.",
    )
    parser.add_argument(
        "--top-rank-slice-only",
        action="store_true",
        help="Analyze only the top rank slice, e.g. top30, for both base and meta heads.",
    )
    parser.add_argument(
        "--analysis-start-day",
        default="",
        help="Only analyze OOF rows on/after this UTC date, e.g. 2024-06-21.",
    )
    parser.add_argument(
        "--min-episode-end-day",
        default="",
        help="Only score bad windows whose end_day is on/after this UTC date, e.g. 2026-06-01.",
    )
    parser.add_argument(
        "--min-candidate-score-for-explanation",
        type=float,
        default=0.005,
        help=(
            "Minimum per-episode Shift x Relevance x Harmfulness score required for a feature "
            "to count as explaining that breakout in aggregate reports."
        ),
    )
    parser.add_argument(
        "--disable-breakout-exploration",
        action="store_true",
        help="Use the static selected URL feature list instead of breakout-aware raw pre-screened operator generation.",
    )
    parser.add_argument(
        "--raw-exploration-max-features",
        type=int,
        default=0,
        help="Max screened raw primitives used for operator generation. 0 means all screened primitives.",
    )
    parser.add_argument("--raw-exploration-min-score", type=float, default=0.015)
    parser.add_argument("--raw-exploration-min-pass-count", type=int, default=1)
    parser.add_argument("--raw-candidate-min-score", type=float, default=0.001)
    parser.add_argument("--composite-candidate-min-score", type=float, default=0.005)
    parser.add_argument(
        "--enable-breakout-svd-knn",
        action="store_true",
        help="Include SVD/KNN operators in breakout exploration. Off by default because it is the slowest operator family.",
    )
    parser.add_argument(
        "--disable-advanced-transforms",
        action="store_true",
        help="Disable rolling slope, acceleration, and extreme-exposure transform generation.",
    )
    parser.add_argument(
        "--advanced-transform-windows",
        default="24,72",
        help="Comma-separated row windows for rolling slope/acceleration/extreme exposure transforms.",
    )
    parser.add_argument("--advanced-transform-extreme-z", type=float, default=2.0)
    parser.add_argument(
        "--disable-advanced-covariance",
        action="store_true",
        help="Disable precision/tail/distance-correlation/historical covariance break diagnostics.",
    )
    parser.add_argument("--max-precision-features", type=int, default=30)
    parser.add_argument("--max-nonlinear-dependence-features", type=int, default=12)
    parser.add_argument(
        "--disable-ebm-interactions",
        action="store_true",
        help="Disable leave-one-episode-out EBM pair interaction and threshold diagnostics.",
    )
    parser.add_argument("--ebm-max-episodes", type=int, default=6)
    parser.add_argument("--ebm-max-features", type=int, default=10)
    parser.add_argument("--ebm-max-pairs", type=int, default=15)
    parser.add_argument("--ebm-max-rows-per-episode", type=int, default=1200)
    parser.add_argument("--ebm-max-control-rows", type=int, default=6000)
    parser.add_argument("--ebm-max-rounds", type=int, default=250)
    parser.add_argument("--ebm-min-rows", type=int, default=400)
    parser.add_argument(
        "--ebm-threshold-registry",
        default="",
        help=(
            "Optional prior ebm_threshold_registry.csv or ebm_feature_threshold_diagnostics.csv "
            "used to generate threshold-derived state features before scoring."
        ),
    )
    parser.add_argument(
        "--disable-ebm-threshold-state-features",
        action="store_true",
        help="Do not generate threshold-derived regime-state features from the EBM threshold registry.",
    )
    parser.add_argument("--ebm-threshold-min-selection-frequency", type=float, default=0.50)
    parser.add_argument("--ebm-threshold-max-false-alarm-rate", type=float, default=0.20)
    parser.add_argument("--ebm-threshold-min-episode-rows", type=int, default=50)
    parser.add_argument(
        "--allow-negative-ebm-threshold-lift",
        action="store_true",
        help="Allow registry thresholds even when held-out log-loss/Brier lift is not positive.",
    )
    parser.add_argument(
        "--disable-redundancy-filter",
        action="store_true",
        help="Disable per-head absolute Spearman redundancy filtering before feature scoring.",
    )
    parser.add_argument("--redundancy-abs-spearman-threshold", type=float, default=0.94)
    parser.add_argument("--redundancy-max-rows", type=int, default=80_000)
    parser.add_argument(
        "--timestamp-aggregate-row-threshold",
        type=int,
        default=250_000,
        help=(
            "Use timestamp-level aggregate design matrices for heavy covariance/interaction diagnostics "
            "when sampled row count reaches this threshold. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--ebm-min-recurrence-episodes",
        type=int,
        default=2,
        help="Prefer EBM features that recur in at least this many training episodes when available.",
    )
    parser.add_argument(
        "--disable-mixed-effects",
        action="store_true",
        help="Disable statsmodels mixed-effect recurrence summary.",
    )
    parser.add_argument("--mixed-effects-max-features", type=int, default=40)
    parser.add_argument(
        "--regime-feature-artifact-dir",
        default="",
        help=(
            "Optional unsupervised_regime_learning artifact directory. Use 'latest' "
            "to load the latest POC artifact with row_keys.pkl."
        ),
    )
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--secondary-window-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument(
        "--min-window-rows",
        type=int,
        default=0,
        help=(
            "Absolute minimum rows per rolling hit-rate window. 0 means use only "
            "--min-window-rows-per-day * window_days."
        ),
    )
    parser.add_argument(
        "--min-window-rows-per-day",
        type=float,
        default=10.0,
        help=(
            "Minimum rows per day inside the analyzed rank slice. Default gives "
            "30 rows for 3d windows and 50 rows for 5d windows."
        ),
    )
    parser.add_argument("--surprise-z-threshold", type=float, default=-10.0)
    parser.add_argument("--hit-rate-delta-threshold", type=float, default=-0.22)
    parser.add_argument("--secondary-surprise-z-threshold", type=float, default=-10.0)
    parser.add_argument("--secondary-hit-rate-delta-threshold", type=float, default=-0.22)
    parser.add_argument(
        "--calibrate-bad-window-thresholds",
        action="store_true",
        help=(
            "Choose per-head hit-rate surprise/HR-delta thresholds from historical rolling-window "
            "distributions instead of using the fixed thresholds directly."
        ),
    )
    parser.add_argument(
        "--target-bad-day-share",
        type=float,
        default=0.20,
        help="Target support-aware bad-day share for calibrated per-head thresholds.",
    )
    parser.add_argument(
        "--bad-window-calibration-grid-size",
        type=int,
        default=45,
        help="Quantile grid size used when searching calibrated surprise/HR-delta threshold pairs.",
    )
    parser.add_argument("--rank-frac", type=float, default=0.30)
    parser.add_argument("--min-feature-coverage", type=float, default=0.70)
    parser.add_argument("--max-dominant-fraction", type=float, default=0.985)
    parser.add_argument("--min-unique-values", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=180)
    parser.add_argument("--max-rows-per-side", type=int, default=200_000)
    parser.add_argument("--baseline-max-rows-per-episode", type=int, default=50_000)
    parser.add_argument("--episode-max-rows-per-episode", type=int, default=200_000)
    parser.add_argument("--max-cov-features", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument(
        "--include-diagnostic-features",
        action="store_true",
        help="Allow deterministic OOF drift/uncertainty diagnostic columns in feature scoring.",
    )
    args = parser.parse_args(argv)
    regime_feature_artifact_dir: Path | None = None
    if str(args.regime_feature_artifact_dir).strip():
        raw = str(args.regime_feature_artifact_dir).strip()
        if raw.lower() == "latest":
            regime_feature_artifact_dir = _latest_regime_feature_artifact(Path(args.data_root))
            if regime_feature_artifact_dir is None:
                raise FileNotFoundError("No unsupervised_regime_learning POC artifact with row_keys.pkl found")
        else:
            regime_feature_artifact_dir = Path(raw)
    feature_store_dir: Path | None = None
    if str(args.feature_store_dir).strip():
        feature_store_dir = Path(str(args.feature_store_dir).strip())
    elif str(args.feature_run_id).strip():
        feature_store_dir = Path(args.data_root) / "features" / str(args.feature_run_id).strip()
    feature_columns_json: Path | None = Path(args.feature_columns_json) if str(args.feature_columns_json).strip() else None
    if feature_columns_json is None and regime_feature_artifact_dir is not None:
        candidate = regime_feature_artifact_dir / "final_feature_columns.json"
        if candidate.exists():
            feature_columns_json = candidate
    transform_windows: tuple[int, ...] = tuple(
        dict.fromkeys(
            max(2, int(item.strip()))
            for item in str(args.advanced_transform_windows).split(",")
            if item.strip()
        )
    )
    if not transform_windows:
        transform_windows = (24, 72)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.data_root)
        / "reports"
        / f"performance_regime_break_analysis_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    transform_cache_dir = (
        Path(str(args.transform_cache_dir).strip())
        if str(args.transform_cache_dir).strip()
        else Path(args.data_root) / "reports" / "performance_regime_break_transform_cache"
    )
    return AnalysisConfig(
        data_root=Path(args.data_root),
        artifact_run_id=str(args.artifact_run_id),
        output_dir=output_dir,
        feature_frame=Path(args.feature_frame) if args.feature_frame else None,
        regime_feature_artifact_dir=regime_feature_artifact_dir,
        feature_store_dir=feature_store_dir,
        feature_columns_json=feature_columns_json,
        include_all_feature_store_columns=bool(args.include_all_feature_store_columns),
        include_config_liquidity_features=bool(args.include_config_liquidity_features),
        stream_feature_generation=not bool(args.disable_streamed_feature_generation),
        transform_cache_enabled=not bool(args.disable_transform_cache),
        generated_transform_cache_enabled=not bool(args.disable_generated_transform_cache),
        transform_cache_dir=transform_cache_dir,
        refresh_transform_cache=bool(args.refresh_transform_cache),
        parquet_cache_compression=str(args.parquet_cache_compression).strip() or DEFAULT_PARQUET_CACHE_COMPRESSION,
        generated_transform_cache_ttl_days=float(args.generated_transform_cache_ttl_days),
        generated_transform_cache_keep_last_n=int(args.generated_transform_cache_keep_last_n),
        generated_transform_cache_max_rows=int(args.generated_transform_cache_max_rows),
        generated_transform_cache_max_bytes=int(args.generated_transform_cache_max_bytes),
        previous_meta_parent_report=Path(args.previous_meta_parent_report)
        if str(args.previous_meta_parent_report).strip()
        else None,
        previous_meta_parent_top_n=int(args.previous_meta_parent_top_n),
        previous_meta_parent_slice=str(args.previous_meta_parent_slice).strip(),
        previous_meta_parent_transforms_enabled=not bool(args.disable_previous_meta_parent_transforms),
        generate_url_composites=not bool(args.disable_url_composite_generation),
        top_rank_slice_only=bool(args.top_rank_slice_only),
        analysis_start_day=str(args.analysis_start_day).strip() or None,
        min_episode_end_day=str(args.min_episode_end_day).strip() or None,
        min_candidate_score_for_explanation=float(args.min_candidate_score_for_explanation),
        breakout_exploration_enabled=not bool(args.disable_breakout_exploration),
        raw_exploration_max_features=int(args.raw_exploration_max_features),
        raw_exploration_min_score=float(args.raw_exploration_min_score),
        raw_exploration_min_pass_count=int(args.raw_exploration_min_pass_count),
        raw_candidate_min_score=float(args.raw_candidate_min_score),
        composite_candidate_min_score=float(args.composite_candidate_min_score),
        breakout_generate_svd_knn=bool(args.enable_breakout_svd_knn),
        advanced_transform_enabled=not bool(args.disable_advanced_transforms),
        advanced_transform_windows=transform_windows,
        advanced_transform_extreme_z=float(args.advanced_transform_extreme_z),
        advanced_covariance_enabled=not bool(args.disable_advanced_covariance),
        max_precision_features=int(args.max_precision_features),
        max_nonlinear_dependence_features=int(args.max_nonlinear_dependence_features),
        ebm_interaction_enabled=not bool(args.disable_ebm_interactions),
        ebm_max_episodes=int(args.ebm_max_episodes),
        ebm_max_features=int(args.ebm_max_features),
        ebm_max_pairs=int(args.ebm_max_pairs),
        ebm_max_rows_per_episode=int(args.ebm_max_rows_per_episode),
        ebm_max_control_rows=int(args.ebm_max_control_rows),
        ebm_max_rounds=int(args.ebm_max_rounds),
        ebm_min_rows=int(args.ebm_min_rows),
        ebm_threshold_registry=Path(args.ebm_threshold_registry) if str(args.ebm_threshold_registry).strip() else None,
        ebm_threshold_state_features_enabled=not bool(args.disable_ebm_threshold_state_features),
        ebm_threshold_min_selection_frequency=float(args.ebm_threshold_min_selection_frequency),
        ebm_threshold_max_false_alarm_rate=float(args.ebm_threshold_max_false_alarm_rate),
        ebm_threshold_min_episode_rows=int(args.ebm_threshold_min_episode_rows),
        ebm_threshold_require_positive_lift=not bool(args.allow_negative_ebm_threshold_lift),
        redundancy_filter_enabled=not bool(args.disable_redundancy_filter),
        redundancy_abs_spearman_threshold=float(args.redundancy_abs_spearman_threshold),
        redundancy_max_rows=int(args.redundancy_max_rows),
        timestamp_aggregate_row_threshold=int(args.timestamp_aggregate_row_threshold),
        ebm_min_recurrence_episodes=int(args.ebm_min_recurrence_episodes),
        mixed_effects_enabled=not bool(args.disable_mixed_effects),
        mixed_effects_max_features=int(args.mixed_effects_max_features),
        baseline_max_rows_per_episode=int(args.baseline_max_rows_per_episode),
        episode_max_rows_per_episode=int(args.episode_max_rows_per_episode),
        window_days=int(args.window_days),
        secondary_window_days=int(args.secondary_window_days),
        embargo_days=int(args.embargo_days),
        min_window_rows=int(args.min_window_rows),
        min_window_rows_per_day=float(args.min_window_rows_per_day),
        surprise_z_threshold=float(args.surprise_z_threshold),
        hit_rate_delta_threshold=float(args.hit_rate_delta_threshold),
        secondary_surprise_z_threshold=float(args.secondary_surprise_z_threshold),
        secondary_hit_rate_delta_threshold=float(args.secondary_hit_rate_delta_threshold),
        bad_window_calibration_enabled=bool(args.calibrate_bad_window_thresholds),
        target_bad_day_share=float(args.target_bad_day_share),
        bad_window_calibration_grid_size=int(args.bad_window_calibration_grid_size),
        rank_frac=float(args.rank_frac),
        min_feature_coverage=float(args.min_feature_coverage),
        max_dominant_fraction=float(args.max_dominant_fraction),
        min_unique_values=int(args.min_unique_values),
        max_features=int(args.max_features),
        max_rows_per_side=int(args.max_rows_per_side),
        max_cov_features=int(args.max_cov_features),
        random_seed=int(args.random_seed),
        include_diagnostic_features=bool(args.include_diagnostic_features),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    run_analysis(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
