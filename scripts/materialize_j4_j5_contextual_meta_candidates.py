"""Materialize frozen contextual meta candidates and blind-score baseline ledger rows.

The research freeze names the arms to test; this script turns those arms into
physical full-fit LightGBM artifacts trained only through a fixed cutoff, then
scores the already-collected baseline ledger rows without reading labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_one_head_contextual_meta_ablation as ctx
from scripts.diagnose_meta_recent_failures import (
    _base_models_for_head,
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _known_export_features,
    _merge_feature_candidates,
    _normalise_keys,
    _prepare_model_matrix,
    lgb,
)
from scripts.run_j4_j5_contextual_meta_prospective_dual_scoring import (
    _collect_baseline_scores,
    _discover_ledgers,
    _infer_head,
)


DEFAULT_FREEZE_MANIFEST = Path(
    "data_perp/reports/j4_j5_contextual_meta_all_head_freeze_20260623/"
    "j4_j5_contextual_meta_all_head_freeze_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/artifacts/j4_j5_contextual_meta_candidate_freeze_20260623")
DEFAULT_LEDGER_ROOT = Path("data_perp/exchanges/krakenfutures/live_state")

FROZEN_ARMS = {
    "long_bars": ctx.ARM_B,
    "long_dist": ctx.ARM_E,
    "short_asset": ctx.ARM_B,
    "short_boll": ctx.ARM_B,
}
TIME_COLUMNS = ("timestamp", "signal_bar_ts", "decision_ts", "__ts__", "ts")
JSON_VALUE_COLUMNS = ("meta_model_feature_values_json", "base_model_feature_values_json")
DIRECT_LEDGER_FEATURE_COLUMNS = (
    "inference_drift_score",
    "uncertainty_score",
    "prob_uncertainty",
    "feature_drift_psi_core",
    "feature_drift_ks_core",
    "feature_drift_cov_shift",
    "regime_centroid_similarity_train",
    "rare_leaf_fraction",
    "leaf_count_p10",
    "leaf_count_min",
    "leaf_weight_p10",
    "base_lgbm_regime_centroid_similarity_train",
    "base_lgbm_feature_drift_psi_core",
    "base_lgbm_feature_drift_ks_core",
    "base_lgbm_feature_drift_cov_shift",
    "base_lgbm_rare_leaf_fraction",
    "base_lgbm_leaf_count_p10",
    "base_lgbm_leaf_count_min",
    "base_lgbm_leaf_weight_p10",
    "base_lgbm_prob_uncertainty",
    "meta_lgbm_regime_centroid_similarity_train",
    "meta_lgbm_feature_drift_psi_core",
    "meta_lgbm_feature_drift_ks_core",
    "meta_lgbm_feature_drift_cov_shift",
    "meta_lgbm_rare_leaf_fraction",
    "meta_lgbm_leaf_count_p10",
    "meta_lgbm_leaf_count_min",
    "meta_lgbm_leaf_weight_p10",
    "meta_lgbm_prob_uncertainty",
)


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@dataclass(frozen=True)
class CandidateArtifact:
    head: str
    selected_arm: str
    model_path: Path
    model_hash: str
    feature_contract_hash: str
    transformer_hash: str
    feature_count: int
    train_rows: int
    train_start: str
    train_end: str
    max_depth: int
    min_child_samples: int
    feature_count_before_live_filter: int = 0
    live_feature_filter_dropped: int = 0


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_status_hash(status: str) -> str:
    return hashlib.sha256(status.encode("utf-8")).hexdigest() if status else ""


def _as_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ledger_feature_aliases(name: str) -> list[str]:
    raw = str(name)
    aliases: list[str] = []

    def add(value: str) -> None:
        value = str(value or "").strip()
        if value and value not in aliases:
            aliases.append(value)

    queue = [raw]
    while queue:
        item = queue.pop(0)
        if item in aliases:
            continue
        add(item)
        for prefix in ("export__", "oof_", "pred_H5_", "base_H5_"):
            if item.startswith(prefix):
                queue.append(item[len(prefix) :])
        if "_H5_" in item:
            tail = item.split("_H5_", 1)[1]
            queue.extend(
                [
                    tail,
                    f"pred_H5_{tail}",
                    f"base_H5_{tail}",
                    f"oof_{tail}",
                    f"export__oof_{tail}",
                ]
            )
        if item.startswith("export__oof_"):
            tail = item[len("export__oof_") :]
            queue.extend([tail, f"oof_{tail}", f"pred_H5_{tail}", f"base_H5_{tail}"])

    semantic = {
        "vol_z": ("volume_zscore", "volume_zscore_48h", "volatility_zscore"),
        "vol_z24": ("volume_zscore_24h", "volume_z_24h", "volatility_zscore"),
        "vol_z_4h": ("volume_zscore_4h", "volume_z_4h", "volatility_zscore"),
        "rvol_z": ("realized_volatility_zscore", "rv_24h", "realized_volatility_24h"),
        "clf_entropy": ("prediction_entropy", "prob_uncertainty", "uncertainty_score"),
        "trend_slope_48h": ("ema50_slope", "trend_slope_48h", "trend_pct"),
        "volatility_zscore": ("volatility_zscore", "vol_z", "vol_z24"),
        "rank_margin_top10": ("rank_margin_top10", "pred_H5_rank_margin_top10"),
        "rank_margin_top20": ("rank_margin_top20", "pred_H5_rank_margin_top20"),
        "rank_margin_top30": ("rank_margin_top30", "pred_H5_rank_margin_top30"),
    }
    for alias in list(aliases):
        for candidate in semantic.get(alias, ()):
            add(candidate)
    return aliases


def _finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    values[~np.isfinite(values)] = np.nan
    return values


def _first_finite_source(
    column: str,
    source_frames: list[pd.DataFrame],
) -> np.ndarray | None:
    aliases = _ledger_feature_aliases(column)
    for frame in source_frames:
        if frame is None or frame.empty:
            continue
        for alias in aliases:
            if alias not in frame.columns:
                continue
            values = _finite_values(frame[alias])
            if bool(np.isfinite(values).any()):
                return values
    return None


def _constant_values(length: int, value: float) -> np.ndarray:
    return np.full(int(length), np.float32(value), dtype=np.float32)


def _squash01(values: np.ndarray, *, default: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.full(len(arr), np.float32(default), dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return out
    clipped = np.clip(arr[finite], -8.0, 8.0)
    out[finite] = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)
    return out


def _first_source_any(names: list[str], source_frames: list[pd.DataFrame]) -> np.ndarray | None:
    for name in names:
        values = _first_finite_source(name, source_frames)
        if values is not None:
            return values
    return None


def _live_proxy_values(
    column: str,
    source_frames: list[pd.DataFrame],
    length: int,
) -> np.ndarray | None:
    """Causal live proxies for canonical context fields absent from OOS rows."""
    col = str(column)
    if "__x__" in col:
        left, right = col.split("__x__", 1)
        lv = _live_proxy_values(left, source_frames, length)
        rv = _live_proxy_values(right, source_frames, length)
        if lv is not None and rv is not None:
            return (lv * rv).astype(np.float32, copy=False)
        return None

    if col == "prediction_support_quality":
        support = _first_source_any(
            [
                "leaf_count_p10",
                "leaf_count_min",
                "leaf_weight_p10",
                "pred_H5_leaf_count_p10",
                "pred_H5_leaf_count_min",
                "base_H5_leaf_count_min",
            ],
            source_frames,
        )
        if support is not None:
            return _squash01(support, default=0.5)
        rare = _first_source_any(["rare_leaf_fraction", "meta_lgbm_rare_leaf_fraction"], source_frames)
        if rare is not None:
            return np.clip(1.0 - _squash01(rare, default=0.5), 0.0, 1.0).astype(np.float32, copy=False)
        return _constant_values(length, 0.5)

    if col == "prediction_reconstruction_anomaly":
        values = _first_source_any(
            [
                "pred_H5_dae_reconstruction_error_zscore",
                "dae_reconstruction_error_zscore",
                "uncertainty_score",
                "prob_uncertainty",
                "inference_drift_score",
            ],
            source_frames,
        )
        return _squash01(values, default=0.0) if values is not None else _constant_values(length, 0.0)

    if col == "prediction_path_instability":
        values = _first_source_any(
            [
                "pred_H5_score_path_std",
                "score_path_std",
                "feature_drift_psi_core",
                "feature_drift_ks_core",
                "inference_drift_score",
            ],
            source_frames,
        )
        return _squash01(values, default=0.0) if values is not None else _constant_values(length, 0.0)

    if col == "regime_similarity_or_novelty":
        similarity = _first_source_any(
            [
                "regime_centroid_similarity_train",
                "meta_lgbm_regime_centroid_similarity_train",
                "base_lgbm_regime_centroid_similarity_train",
            ],
            source_frames,
        )
        if similarity is not None:
            return np.clip(1.0 - _squash01(similarity, default=0.5), 0.0, 1.0).astype(np.float32, copy=False)
        drift = _first_source_any(["inference_drift_score", "feature_drift_psi_core"], source_frames)
        return _squash01(drift, default=0.5) if drift is not None else _constant_values(length, 0.5)

    market_proxy_sources = {
        "leverage_funding_crowding": [
            "funding_z",
            "funding_abs_z",
            "funding_rate",
            "oi_value_z_90d",
            "oi_value_log_z_90d",
            "leverage_build_score",
        ],
        "liquidity_participation_stress": [
            "amihud_z",
            "amihud_illiq",
            "volume_zscore",
            "volume_zscore_48h",
            "volume_percentile",
            "dv_z",
        ],
        "tail_volatility_stress": [
            "volatility_zscore",
            "realized_volatility_24h",
            "rv_24h",
            "atr_percentile",
            "true_range_percentile",
        ],
        "relative_value_dislocation": [
            "symbol_minus_mkt_ret_24h",
            "symbol_minus_mkt_ret_4h",
            "cs_rank_ret_24h",
            "asset_ret_vs_universe_24h",
        ],
        "breadth_market_state": [
            "market_breadth_24h",
            "market_breadth_4h",
            "market_dispersion_24h",
            "market_dispersion_4h",
        ],
        "network_concentration": [
            "avg_pair_corr_24h",
            "market_dispersion_24h",
            "mkt_oi_dispersion_24h",
            "mkt_oi_breadth_rising_24h",
        ],
    }
    if col in market_proxy_sources:
        values = _first_source_any(market_proxy_sources[col], source_frames)
        return _squash01(values, default=0.0) if values is not None else _constant_values(length, 0.0)

    return None


def _fill_missing_from_sources(
    frame: pd.DataFrame,
    columns: list[str],
    source_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resolve live aliases for selected model columns without using labels."""
    out = frame.copy()
    stats = {"alias_created_columns": 0, "alias_filled_values": 0, "proxy_filled_values": 0}
    for col in columns:
        existing = (
            _finite_values(out[col])
            if col in out.columns
            else np.full(len(out), np.nan, dtype=np.float32)
        )
        missing = ~np.isfinite(existing)
        if not bool(missing.any()):
            continue
        source = _first_finite_source(col, source_frames)
        if source is None:
            source = _live_proxy_values(col, source_frames, len(out))
        used_proxy = source is not None and _first_finite_source(col, source_frames) is None
        if source is None:
            if col not in out.columns:
                out[col] = existing
            continue
        if len(source) != len(out):
            continue
        fill_mask = missing & np.isfinite(source)
        if not bool(fill_mask.any()):
            if col not in out.columns:
                out[col] = existing
            continue
        if col not in out.columns:
            stats["alias_created_columns"] += 1
        existing[fill_mask] = source[fill_mask]
        if used_proxy:
            stats["proxy_filled_values"] += int(fill_mask.sum())
        else:
            stats["alias_filled_values"] += int(fill_mask.sum())
        out[col] = existing
    return out, stats


def _finite_rate_by_column(frame: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    n = max(int(len(frame)), 1)
    for col in frame.columns:
        values = _finite_values(frame[col])
        rates[str(col)] = float(np.isfinite(values).sum() / n)
    return rates


def _build_score_arm_frame(
    *,
    arm: str,
    score_rows: pd.DataFrame,
    current_score: pd.DataFrame,
    raw_score: pd.DataFrame,
    raw_train: pd.DataFrame,
    leaf_transform: dict[str, Any],
    canonical_defs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat([raw_train, raw_score], axis=0, ignore_index=True, copy=False)
    train_idx = np.arange(len(raw_train), dtype=np.int64)
    score_idx = np.arange(len(raw_train), len(raw_train) + len(raw_score), dtype=np.int64)
    canonical_all, canonical_diag = ctx._fresh_oos_canonical_features(
        combined,
        train_idx=train_idx,
        test_idx=score_idx,
        definitions=canonical_defs,
        trailing_window=int(args.trailing_window),
        min_periods=int(args.min_periods),
        min_resolved_features=int(args.min_resolved_features),
    )
    canonical_score = canonical_all.iloc[score_idx].reset_index(drop=True)
    canonical_score["leaf_occupancy_novelty"] = _transform_leaf_occupancy(raw_score, leaf_transform)
    canonical_score = canonical_score.loc[:, list(ctx.CANONICAL_CONTEXT)]
    fake_panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(score_rows["timestamp"], utc=True, errors="coerce").to_numpy(),
            "symbol": score_rows["symbol"].astype(str).to_numpy(),
        }
    )
    arms = ctx._arm_frames(
        fake_panel,
        current_score.reset_index(drop=True),
        canonical_score.reset_index(drop=True),
    )
    x_score = arms[arm]
    if x_score is None:
        x_score = pd.DataFrame(index=score_rows.index)
    x_score, _duplicate_feature_columns = _deduplicate_columns(x_score)
    return x_score, canonical_diag


def _feature_store_symbol(symbol: Any) -> str:
    return str(symbol).replace("/", "_")


def _parse_feature_json(value: Any) -> dict[str, float]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return {}
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in payload.items():
        try:
            val = float(raw)
        except Exception:
            continue
        if np.isfinite(val):
            out[str(key)] = val
    return out


def _row_feature_map(row: pd.Series) -> dict[str, float]:
    features: dict[str, float] = {}
    for col in JSON_VALUE_COLUMNS:
        if col in row.index:
            features.update(_parse_feature_json(row[col]))
    for col in DIRECT_LEDGER_FEATURE_COLUMNS:
        if col not in row.index:
            continue
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.notna(val) and np.isfinite(float(val)):
            features[col] = float(val)
            features[f"oof_{col}"] = float(val)
            features[f"export__{col}"] = float(val)
            features[f"export__oof_{col}"] = float(val)
            if col.startswith("base_lgbm_"):
                tail = col.removeprefix("base_lgbm_")
                features[tail] = float(val)
                features[f"base_H5_{tail}"] = float(val)
                features[f"export__oof_{tail}"] = float(val)
            if col.startswith("meta_lgbm_"):
                tail = col.removeprefix("meta_lgbm_")
                features[tail] = float(val)
                features[f"pred_H5_{tail}"] = float(val)
                features[f"export__oof_{tail}"] = float(val)
    for score_col in ("base_pred", "meta_pred", "calibrated_score", "raw_prediction_score", "policy_rank_pct", "auction_rank_pct"):
        if score_col in row.index:
            val = pd.to_numeric(pd.Series([row[score_col]]), errors="coerce").iloc[0]
            if pd.notna(val) and np.isfinite(float(val)):
                features[score_col] = float(val)
                features[f"oof_{score_col}"] = float(val)
                features[f"export__{score_col}"] = float(val)
                features[f"export__oof_{score_col}"] = float(val)
                if score_col in {"meta_pred", "calibrated_score", "raw_prediction_score"}:
                    for alias in ("oof_pred", "oof_p_move", "oof_meta_clf", "clf_center"):
                        features[alias] = float(val)
                        features[f"export__{alias}"] = float(val)
                if score_col == "base_pred":
                    for alias in ("oof_base_clf", "base_clf_centered"):
                        features[alias] = float(val)
                        features[f"export__{alias}"] = float(val)
                if score_col in {"policy_rank_pct", "auction_rank_pct"}:
                    features["oof_rank_pct"] = float(val)
                    features["export__oof_rank_pct"] = float(val)
    for rank_col in ("base_train_rank_pct", "meta_train_rank_pct", "historical_rank_pct", "batch_rank_pct"):
        if rank_col not in row.index:
            continue
        val = pd.to_numeric(pd.Series([row[rank_col]]), errors="coerce").iloc[0]
        if pd.notna(val) and np.isfinite(float(val)):
            features[rank_col] = float(val)
            features["oof_rank_pct"] = float(val)
            features["export__oof_rank_pct"] = float(val)
    return features


def _matrix_from_feature_maps(
    rows: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data: dict[str, np.ndarray] = {}
    source = pd.Series("", index=rows.index, dtype=object)
    missing_count = pd.Series(0, index=rows.index, dtype=np.int32)
    maps = [_row_feature_map(row) for _, row in rows.iterrows()]
    for col in columns:
        values = np.full(len(rows), np.nan, dtype=np.float32)
        for i, fmap in enumerate(maps):
            found = False
            for alias in _ledger_feature_aliases(col):
                if alias in fmap:
                    values[i] = np.float32(fmap[alias])
                    found = True
                    break
            if not found:
                missing_count.iloc[i] += 1
        data[col] = values
    return _downcast_numeric(pd.DataFrame(data, index=rows.index)), source, missing_count


def _missing_feature_summary(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    head: str,
    source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not columns:
        return pd.DataFrame(rows)
    n = max(int(len(frame)), 1)
    for col in columns:
        if col not in frame.columns:
            rows.append(
                {
                    "head": head,
                    "source": source,
                    "feature": col,
                    "present": False,
                    "missing_rows": int(len(frame)),
                    "missing_rate": 1.0,
                    "finite_rate": 0.0,
                }
            )
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        finite = np.isfinite(vals.to_numpy(dtype=np.float64, copy=False))
        rows.append(
            {
                "head": head,
                "source": source,
                "feature": col,
                "present": True,
                "missing_rows": int((~finite).sum()),
                "missing_rate": float((~finite).sum() / n),
                "finite_rate": float(finite.sum() / n),
            }
        )
    return pd.DataFrame(rows)


def _collect_full_ledger_rows(
    ledger_root: Path,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _discover_ledgers(ledger_root):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if not {"strategy_id", "symbol"}.issubset(df.columns):
            continue
        time_col = "signal_bar_ts" if "signal_bar_ts" in df.columns else next((c for c in TIME_COLUMNS if c in df.columns), None)
        if time_col is None:
            continue
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        mask = ts.ge(start) & ts.lt(end_exclusive)
        if not bool(mask.any()):
            continue
        local = df.loc[mask].copy()
        local["head"] = local["strategy_id"].map(_infer_head)
        local = local[local["head"].notna()].copy()
        if local.empty:
            continue
        local["timestamp"] = ts.loc[local.index].to_numpy()
        local["decision_ts"] = pd.to_datetime(local.get("decision_ts", local["timestamp"]), utc=True, errors="coerce")
        local["ledger_run_id"] = path.parent.name if path.parent.name != "live_state" else "root_live_state"
        local["ledger_path"] = str(path)
        frames.append(local)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True, errors="coerce")
    out = out.sort_values(["head", "timestamp", "symbol", "strategy_id", "decision_ts"], kind="mergesort")
    return out.drop_duplicates(["head", "timestamp", "symbol", "strategy_id"], keep="last").reset_index(drop=True)


def _fit_leaf_occupancy_transform(raw_train: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    source = pd.DataFrame(index=raw_train.index)
    candidates = {
        "low_leaf_frequency": (
            -1.0,
            [
                "oof_leaf_train_freq_p10",
                "leaf_count_p10",
                "meta_lgbm_leaf_count_p10",
                "pred_H5_leaf_count_p10",
                "pred_H5_leaf_weight_p10",
            ],
        ),
        "low_freq_fraction": (
            1.0,
            [
                "oof_leaf_low_freq_fraction",
                "rare_leaf_fraction",
                "meta_lgbm_rare_leaf_fraction",
                "pred_H5_rare_leaf_low_support_score",
            ],
        ),
        "leaf_surprisal": (1.0, ["oof_leaf_surprisal_mean", "leaf_surprisal", "pred_H5_leaf_surprisal_mean"]),
        "support_gap": (1.0, ["oof_support_gap", "support_gap", "pred_H5_support_gap"]),
    }
    resolved: dict[str, str] = {}
    for name, (sign, aliases) in candidates.items():
        col = next((alias for alias in aliases if alias in raw_train.columns), None)
        if col is None:
            continue
        values = pd.to_numeric(raw_train[col], errors="coerce").astype("float32")
        source[name] = float(sign) * values
        resolved[name] = col
    if source.empty:
        return np.full(len(raw_train), np.nan, dtype=np.float32), {"resolved_sources": {}, "lo": np.nan, "hi": np.nan}
    raw = source.mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    if finite.size < 20:
        return np.full(len(raw_train), np.nan, dtype=np.float32), {"resolved_sources": resolved, "lo": np.nan, "hi": np.nan}
    lo = float(np.nanquantile(finite, 0.05))
    hi = float(np.nanquantile(finite, 0.95))
    out = np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32, copy=False)
    out[~np.isfinite(raw)] = np.nan
    return out, {"resolved_sources": resolved, "lo": lo, "hi": hi}


def _transform_leaf_occupancy(raw: pd.DataFrame, transform: dict[str, Any]) -> np.ndarray:
    resolved = dict(transform.get("resolved_sources", {}) or {})
    lo = float(transform.get("lo", np.nan))
    hi = float(transform.get("hi", np.nan))
    if not resolved or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(len(raw), np.nan, dtype=np.float32)
    source = pd.DataFrame(index=raw.index)
    signs = {
        "low_leaf_frequency": -1.0,
        "low_freq_fraction": 1.0,
        "leaf_surprisal": 1.0,
        "support_gap": 1.0,
    }
    for name, col in resolved.items():
        chosen = next((alias for alias in _ledger_feature_aliases(col) if alias in raw.columns), None)
        if chosen is None and col in raw.columns:
            chosen = col
        if chosen is not None:
            source[name] = signs.get(name, 1.0) * pd.to_numeric(raw[chosen], errors="coerce").astype("float32")
    if source.empty:
        return np.full(len(raw), np.nan, dtype=np.float32)
    values = source.mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    out = np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32, copy=False)
    out[~np.isfinite(values)] = np.nan
    return out


def _candidate_raw_from_ledger(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix, _source, _missing = _matrix_from_feature_maps(rows, columns)
    raw = pd.concat(
        [
            rows[["timestamp", "symbol"]].reset_index(drop=True),
            matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    return raw.loc[:, ~raw.columns.duplicated()]


def _deduplicate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_count = int(frame.columns.duplicated().sum())
    if duplicate_count:
        frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    return frame, duplicate_count


def _fit_full_candidate(
    *,
    head: Any,
    arm: str,
    panel_train: pd.DataFrame,
    current_x: pd.DataFrame,
    raw_train: pd.DataFrame,
    canonical_defs: dict[str, dict[str, Any]],
    live_feature_finite_rate: dict[str, float] | None,
    args: argparse.Namespace,
    out_dir: Path,
) -> CandidateArtifact:
    train_idx = np.arange(len(raw_train), dtype=np.int64)
    canonical, canonical_diag = ctx._fresh_oos_canonical_features(
        raw_train,
        train_idx=train_idx,
        test_idx=np.array([], dtype=np.int64),
        definitions=canonical_defs,
        trailing_window=int(args.trailing_window),
        min_periods=int(args.min_periods),
        min_resolved_features=int(args.min_resolved_features),
    )
    leaf_values, leaf_transform = _fit_leaf_occupancy_transform(raw_train)
    canonical["leaf_occupancy_novelty"] = leaf_values
    canonical = canonical.loc[:, list(ctx.CANONICAL_CONTEXT)]
    arms = ctx._arm_frames(panel_train, current_x.reset_index(drop=True), canonical.reset_index(drop=True))
    x = arms[arm]
    if x is None or x.empty:
        raise RuntimeError(f"{head.head}: selected arm {arm} has no feature matrix")
    x, duplicate_feature_columns = _deduplicate_columns(x)
    y = ctx._meta_target(panel_train)
    train_mask = y >= 0
    x = x.loc[train_mask].reset_index(drop=True)
    y_fit = y[train_mask]
    x = x.replace([np.inf, -np.inf], np.nan)
    train_finite_rate = {
        str(col): float(pd.to_numeric(x[col], errors="coerce").notna().mean())
        for col in x.columns
    }
    keep_cols_before_live = [
        col for col in x.columns if train_finite_rate.get(str(col), 0.0) > 0.02
    ]
    live_feature_finite_rate = dict(live_feature_finite_rate or {})
    min_live = float(getattr(args, "min_live_feature_coverage", 1.0))
    if bool(getattr(args, "enforce_live_feature_contract", True)):
        keep_cols = [
            col
            for col in keep_cols_before_live
            if float(live_feature_finite_rate.get(str(col), 0.0)) >= min_live
        ]
    else:
        keep_cols = keep_cols_before_live
    if not keep_cols:
        raise RuntimeError(f"{head.head}: no trainable candidate columns")
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    if len(np.unique(y_fit)) < 2:
        raise RuntimeError(f"{head.head}: training labels have one class")
    max_depth = 2 if arm == ctx.ARM_E else 3
    min_child = max(50, int(math.ceil(0.025 * len(y_fit))))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.035,
        max_depth=int(max_depth),
        num_leaves=max(4, min(16, 2 ** int(max_depth))),
        min_child_samples=min_child,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=int(args.seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_prepared, y_fit)
    ts = pd.to_datetime(panel_train.loc[train_mask, "timestamp"], utc=True, errors="coerce")
    feature_contract = {
        "head": head.head,
        "selected_arm": arm,
        "feature_columns": keep_cols,
        "feature_count_before_live_filter": int(len(keep_cols_before_live)),
        "feature_count_after_live_filter": int(len(keep_cols)),
        "live_feature_filter_dropped": int(len(keep_cols_before_live) - len(keep_cols)),
        "min_live_feature_coverage": min_live,
        "enforce_live_feature_contract": bool(getattr(args, "enforce_live_feature_contract", True)),
        "live_feature_finite_rate": {
            str(col): float(live_feature_finite_rate.get(str(col), 0.0))
            for col in keep_cols_before_live
        },
        "train_feature_finite_rate": {
            str(col): float(train_finite_rate.get(str(col), 0.0))
            for col in keep_cols_before_live
        },
        "duplicate_feature_columns_dropped": int(duplicate_feature_columns),
        "training_cutoff": args.training_cutoff,
        "imputation_policy": "live-equivalent feature contract; LightGBM native missing only for residual within-column missingness",
        "model_params": model.get_params(),
    }
    transformer_payload = {
        "canonical_definitions_hash": _sha256_json(canonical_defs),
        "canonical_diagnostics": canonical_diag,
        "leaf_occupancy_transform": leaf_transform,
        "trailing_window": int(args.trailing_window),
        "min_periods": int(args.min_periods),
        "min_resolved_features": int(args.min_resolved_features),
    }
    model_dir = out_dir / "models" / head.head
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": keep_cols,
        "selected_arm": arm,
        "head": head.head,
        "feature_contract": feature_contract,
        "transformer": transformer_payload,
        "current_feature_columns": list(dict.fromkeys(str(c) for c in current_x.columns)),
        "train_columns_all": list(x.columns),
    }
    model_path = model_dir / "candidate_model.joblib"
    joblib.dump(payload, model_path)
    (model_dir / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2, sort_keys=True, default=_json_default))
    (model_dir / "transformer.json").write_text(json.dumps(transformer_payload, indent=2, sort_keys=True, default=_json_default))
    return CandidateArtifact(
        head=head.head,
        selected_arm=arm,
        model_path=model_path,
        model_hash=_sha256_file(model_path),
        feature_contract_hash=_sha256_json(feature_contract),
        transformer_hash=_sha256_json(transformer_payload),
        feature_count=int(len(keep_cols)),
        train_rows=int(len(y_fit)),
        train_start="" if ts.dropna().empty else ts.min().isoformat(),
        train_end="" if ts.dropna().empty else ts.max().isoformat(),
        max_depth=int(max_depth),
        min_child_samples=int(min_child),
        feature_count_before_live_filter=int(len(keep_cols_before_live)),
        live_feature_filter_dropped=int(len(keep_cols_before_live) - len(keep_cols)),
    )


def _score_head_rows(
    *,
    head: str,
    candidate: dict[str, Any],
    score_rows: pd.DataFrame,
    current_score: pd.DataFrame,
    raw_score: pd.DataFrame,
    raw_train: pd.DataFrame,
    canonical_defs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    matrix_output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = list(candidate["feature_columns"])
    model = candidate["model"]
    arm = str(candidate["selected_arm"])
    current_feature_cols = list(candidate.get("current_feature_columns", []))
    current_score = current_score.loc[:, ~current_score.columns.duplicated(keep="first")]
    current_missing_value_count = np.zeros(len(score_rows), dtype=np.int32)
    for col in current_feature_cols:
        if col not in current_score.columns:
            current_missing_value_count += 1
            continue
        vals = pd.to_numeric(current_score[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        current_missing_value_count += (~np.isfinite(vals)).astype(np.int32)
    x_score, canonical_diag = _build_score_arm_frame(
        arm=arm,
        score_rows=score_rows,
        current_score=current_score.reset_index(drop=True),
        raw_score=raw_score.reset_index(drop=True),
        raw_train=raw_train,
        leaf_transform=candidate["transformer"]["leaf_occupancy_transform"],
        canonical_defs=canonical_defs,
        args=args,
    )
    x_score, duplicate_feature_columns = _deduplicate_columns(x_score)
    raw_feature_coverage_before_alias = _finite_rate_by_column(x_score)
    x_score, alias_stats = _fill_missing_from_sources(
        x_score,
        feature_cols,
        [
            x_score.reset_index(drop=True),
            current_score.reset_index(drop=True),
            raw_score.reset_index(drop=True),
        ],
    )
    present_before_fill = set(x_score.columns)
    unresolved_columns = [col for col in feature_cols if col not in present_before_fill]
    for col in feature_cols:
        if col not in x_score.columns:
            x_score[col] = np.nan
    x_score = x_score.loc[:, feature_cols]
    feature_availability = _missing_feature_summary(x_score, columns=feature_cols, head=head, source=f"candidate_score_{arm}")
    if matrix_output_path is not None:
        matrix_output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_keys = score_rows[["head", "timestamp", "decision_ts", "symbol", "strategy_id"]].reset_index(drop=True)
        matrix_payload = pd.concat(
            [matrix_keys, _downcast_numeric(x_score.reset_index(drop=True))],
            axis=1,
        )
        matrix_payload.to_parquet(matrix_output_path, index=False)
    x_score = _prepare_model_matrix(x_score)
    missing_value_count = (~np.isfinite(x_score.to_numpy(dtype=np.float32, copy=False))).sum(axis=1).astype(np.int32)
    unresolved_required = np.full(len(score_rows), len(unresolved_columns), dtype=np.int32)
    score = np.full(len(score_rows), np.nan, dtype=np.float32)
    eligible = unresolved_required == 0
    if bool(eligible.any()):
        score[eligible] = model.predict_proba(x_score.loc[eligible])[:, 1].astype(np.float32, copy=False)
    out = pd.DataFrame(
        {
            "head": head,
            "timestamp": pd.to_datetime(score_rows["timestamp"], utc=True, errors="coerce").to_numpy(),
            "decision_ts": pd.to_datetime(score_rows["decision_ts"], utc=True, errors="coerce").to_numpy(),
            "symbol": score_rows["symbol"].astype(str).to_numpy(),
            "strategy_id": score_rows["strategy_id"].astype(str).to_numpy(),
            "candidate_score": score,
            "candidate_feature_coverage": 1.0 - missing_value_count.astype(np.float32) / max(float(len(feature_cols)), 1.0),
            "candidate_fallback_count": unresolved_required,
            "candidate_missing_value_count": missing_value_count,
            "candidate_current_missing_count": current_missing_value_count,
            "candidate_alias_created_columns": int(alias_stats.get("alias_created_columns", 0)),
            "candidate_alias_filled_values": int(alias_stats.get("alias_filled_values", 0)),
            "candidate_proxy_filled_values": int(alias_stats.get("proxy_filled_values", 0)),
            "candidate_raw_feature_coverage_before_alias": np.float32(
                np.nanmean(
                    [
                        raw_feature_coverage_before_alias.get(str(col), np.nan)
                        for col in feature_cols
                    ]
                )
            )
            if feature_cols
            else np.float32(np.nan),
            "candidate_duplicate_feature_columns_dropped": int(duplicate_feature_columns),
            "candidate_context_diag_hash": _sha256_json(canonical_diag),
            "candidate_feature_matrix_path": str(matrix_output_path) if matrix_output_path is not None else "",
        }
    )
    return out, feature_availability


def _rank_within_timestamp(df: pd.DataFrame, score_col: str, out_col: str) -> pd.Series:
    ranks = pd.Series(np.nan, index=df.index, dtype="float32")
    for _, idx in df.groupby(["head", "timestamp"], sort=False).groups.items():
        local = df.loc[idx, score_col]
        valid = pd.to_numeric(local, errors="coerce")
        if valid.notna().sum() == 0:
            continue
        ranks.loc[idx] = valid.rank(method="average", pct=True).astype("float32")
    ranks.name = out_col
    return ranks


def run(args: argparse.Namespace) -> Path:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = _as_utc(args.training_cutoff)
    score_start = _as_utc(args.score_start)
    score_end_exclusive = _as_utc(args.score_end) + pd.Timedelta(days=1)

    freeze = pd.read_csv(args.freeze_manifest)
    selected_by_head = {str(row["head"]): FROZEN_ARMS.get(str(row["head"]), str(row.get("selected_contextual_feature_arm", ""))) for _, row in freeze.iterrows()}
    selected_by_head = {head: arm for head, arm in selected_by_head.items() if head in FROZEN_ARMS}

    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    feature_dir = Path(args.feature_dir)
    report_dir = Path(args.report_dir)
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    regime_context = Path(args.regime_context) if args.regime_context else None
    canonical_defs = ctx._load_canonical_definitions(Path(args.canonical_reduction))
    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = [h for h in _discover_heads(meta_artifact_dir, report_dir, meta_models) if h.head in selected_by_head]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)

    baseline, ledger_audit = _collect_baseline_scores(Path(args.ledger_root), start=score_start, end_exclusive=score_end_exclusive)
    full_ledger = _collect_full_ledger_rows(Path(args.ledger_root), start=score_start, end_exclusive=score_end_exclusive)
    full_ledger = full_ledger.merge(
        baseline[["head", "timestamp", "symbol", "strategy_id", "baseline_score"]].drop_duplicates(
            ["head", "timestamp", "symbol", "strategy_id"]
        ),
        on=["head", "timestamp", "symbol", "strategy_id"],
        how="inner",
    )

    artifacts: list[CandidateArtifact] = []
    raw_train_by_head: dict[str, pd.DataFrame] = {}
    score_context_by_head: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    score_symbol_columns: set[str] | None = None
    with _temporary_env("EPM_RECENT_FAILURE_INCLUDE_FEATURE_DELTAS", "0"):
        train_symbol_columns = _feature_store_union(feature_dir)
        for head in heads:
            print(f"[materialize_candidates] training head={head.head} arm={selected_by_head[head.head]}", flush=True)
            panel = _normalise_keys(pd.read_parquet(head.meta_oof_path))
            panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
            panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
            panel = panel.loc[panel["timestamp"].le(cutoff)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
            if panel.empty:
                raise RuntimeError(f"{head.head}: no training rows through {cutoff}")
            race = meta_models[head.meta_key]
            current_x, raw = ctx._assemble_head_context(
                head=head,
                panel=panel,
                race=race,
                base_bundle=base_bundle,
                feature_dir=feature_dir,
                transform_cache=transform_cache,
                symbol_columns=train_symbol_columns,
                regime_context=regime_context,
                max_regime_columns=int(args.max_regime_columns),
            )
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
            raw_train_by_head[head.head] = raw.reset_index(drop=True)
            live_feature_finite_rate: dict[str, float] | None = None
            rows = full_ledger.loc[full_ledger["head"].astype(str).eq(head.head)].reset_index(drop=True)
            if not rows.empty:
                with _temporary_env("EPM_RECENT_FAILURE_INCLUDE_FEATURE_DELTAS", "1"):
                    if score_symbol_columns is None:
                        score_symbol_columns = _feature_store_union(feature_dir)
                    rows_for_context = _normalise_keys(rows.copy())
                    rows_for_context["timestamp"] = pd.to_datetime(rows_for_context["timestamp"], utc=True, errors="coerce")
                    rows_for_context["symbol"] = rows_for_context["symbol"].map(_feature_store_symbol)
                    current_score, raw_score = ctx._assemble_head_context(
                        head=head,
                        panel=rows_for_context,
                        race=meta_models[head.meta_key],
                        base_bundle=base_bundle,
                        feature_dir=feature_dir,
                        transform_cache=transform_cache,
                        symbol_columns=score_symbol_columns,
                        regime_context=regime_context,
                        max_regime_columns=int(args.max_regime_columns),
                    )
                    ledger_raw = _candidate_raw_from_ledger(rows, list(raw.columns))
                    ledger_features = ledger_raw.drop(columns=["timestamp", "symbol"], errors="ignore").reset_index(drop=True)
                    for frame in (current_score, raw_score):
                        frame.reset_index(drop=True, inplace=True)
                        for col in ledger_features.columns:
                            vals = pd.to_numeric(ledger_features[col], errors="coerce")
                            if col not in frame.columns:
                                frame[col] = vals.to_numpy(dtype=np.float32, copy=False)
                                continue
                            existing = pd.to_numeric(frame[col], errors="coerce")
                            missing = ~np.isfinite(existing.to_numpy(dtype=np.float64, copy=False))
                            if bool(missing.any()):
                                out_vals = existing.to_numpy(dtype=np.float32, copy=True)
                                src_vals = vals.to_numpy(dtype=np.float32, copy=False)
                                fill = missing & np.isfinite(src_vals)
                                out_vals[fill] = src_vals[fill]
                                frame[col] = out_vals
                    _leaf_values, leaf_transform = _fit_leaf_occupancy_transform(raw.reset_index(drop=True))
                    x_live, _canonical_diag = _build_score_arm_frame(
                        arm=selected_by_head[head.head],
                        score_rows=rows,
                        current_score=current_score.reset_index(drop=True),
                        raw_score=raw_score.reset_index(drop=True),
                        raw_train=raw.reset_index(drop=True),
                        leaf_transform=leaf_transform,
                        canonical_defs=canonical_defs,
                        args=args,
                    )
                    x_live, _ = _deduplicate_columns(x_live)
                    all_cols = list(dict.fromkeys([*x_live.columns, *current_score.columns, *raw_score.columns]))
                    x_live, alias_stats = _fill_missing_from_sources(
                        x_live,
                        all_cols,
                        [
                            x_live.reset_index(drop=True),
                            current_score.reset_index(drop=True),
                            raw_score.reset_index(drop=True),
                        ],
                    )
                    live_feature_finite_rate = _finite_rate_by_column(x_live)
                    score_context_by_head[head.head] = (
                        rows,
                        current_score.reset_index(drop=True),
                        raw_score.reset_index(drop=True),
                    )
                    print(
                        "[materialize_candidates] live feature contract "
                        f"head={head.head} cols={len(live_feature_finite_rate)} "
                        f"alias_filled={int(alias_stats.get('alias_filled_values', 0))} "
                        f"proxy_filled={int(alias_stats.get('proxy_filled_values', 0))}",
                        flush=True,
                    )
            artifact = _fit_full_candidate(
                head=head,
                arm=selected_by_head[head.head],
                panel_train=panel,
                current_x=current_x,
                raw_train=raw,
                canonical_defs=canonical_defs,
                live_feature_finite_rate=live_feature_finite_rate,
                args=args,
                out_dir=out_dir,
            )
            artifacts.append(artifact)

    score_frames: list[pd.DataFrame] = []
    missing_feature_frames: list[pd.DataFrame] = []
    model_payloads = {a.head: joblib.load(a.model_path) for a in artifacts}
    heads_by_name = {head.head: head for head in heads}
    with _temporary_env("EPM_RECENT_FAILURE_INCLUDE_FEATURE_DELTAS", "1"):
        if score_symbol_columns is None:
            score_symbol_columns = _feature_store_union(feature_dir)
        for artifact in artifacts:
            rows = full_ledger.loc[full_ledger["head"].astype(str).eq(artifact.head)].reset_index(drop=True)
            if rows.empty:
                continue
            print(f"[materialize_candidates] scoring head={artifact.head} rows={len(rows)}", flush=True)
            head_ctx = heads_by_name[artifact.head]
            if artifact.head in score_context_by_head:
                _cached_rows, current_score, raw_score = score_context_by_head[artifact.head]
            else:
                rows_for_context = _normalise_keys(rows.copy())
                rows_for_context["timestamp"] = pd.to_datetime(rows_for_context["timestamp"], utc=True, errors="coerce")
                rows_for_context["symbol"] = rows_for_context["symbol"].map(_feature_store_symbol)
                current_score, raw_score = ctx._assemble_head_context(
                    head=head_ctx,
                    panel=rows_for_context,
                    race=meta_models[head_ctx.meta_key],
                    base_bundle=base_bundle,
                    feature_dir=feature_dir,
                    transform_cache=transform_cache,
                    symbol_columns=score_symbol_columns,
                    regime_context=regime_context,
                    max_regime_columns=int(args.max_regime_columns),
                )
            ledger_raw = _candidate_raw_from_ledger(
                rows,
                list(model_payloads[artifact.head].get("train_columns_all", [])),
            )
            ledger_features = ledger_raw.drop(columns=["timestamp", "symbol"], errors="ignore").reset_index(drop=True)
            for frame in (current_score, raw_score):
                frame.reset_index(drop=True, inplace=True)
                for col in ledger_features.columns:
                    vals = pd.to_numeric(ledger_features[col], errors="coerce")
                    if col not in frame.columns:
                        frame[col] = vals.to_numpy(dtype=np.float32, copy=False)
                        continue
                    existing = pd.to_numeric(frame[col], errors="coerce")
                    if not bool(np.isfinite(existing.to_numpy(dtype=np.float64, copy=False)).any()):
                        frame[col] = vals.to_numpy(dtype=np.float32, copy=False)
            scored, missing_features = _score_head_rows(
                head=artifact.head,
                candidate=model_payloads[artifact.head],
                score_rows=rows,
                current_score=current_score.reset_index(drop=True),
                raw_score=raw_score.reset_index(drop=True),
                raw_train=raw_train_by_head[artifact.head],
                canonical_defs=canonical_defs,
                args=args,
                matrix_output_path=out_dir / "candidate_score_matrices" / f"{artifact.head}_{artifact.selected_arm}_score_matrix.parquet",
            )
            scored["model_hash"] = artifact.model_hash
            scored["feature_contract_hash"] = artifact.feature_contract_hash
            scored["transformer_hash"] = artifact.transformer_hash
            score_frames.append(scored)
            missing_feature_frames.append(missing_features)
    candidate_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    dual = baseline.merge(
        candidate_scores,
        on=["head", "timestamp", "symbol", "strategy_id"],
        how="left",
        suffixes=("", "_candidate"),
    )
    dual["candidate_timestamp_rank"] = _rank_within_timestamp(dual, "candidate_score", "candidate_timestamp_rank")
    dual["baseline_timestamp_rank"] = _rank_within_timestamp(dual, "baseline_score", "baseline_timestamp_rank")
    universe = dual.groupby(["head", "timestamp"], sort=False)["symbol"].transform("count")
    dual["eligible_universe_size"] = universe.astype("int32")
    manifest_rows = [artifact.__dict__ for artifact in artifacts]
    manifest = pd.DataFrame([{k: str(v) if isinstance(v, Path) else v for k, v in row.items()} for row in manifest_rows])
    git_status = _git_status_short()
    manifest["code_commit"] = _git_commit()
    manifest["git_dirty"] = bool(git_status)
    manifest["git_status_hash"] = _git_status_hash(git_status)
    manifest["training_cutoff"] = cutoff.isoformat()
    manifest["label_contract"] = "unchanged_y_bin"
    manifest["training_loss"] = "hard_label_binary_logloss"
    manifest["sample_weight_contract"] = "uniform"
    manifest["hpo_contract"] = "none"
    manifest["feature_selection_contract"] = "none_new; frozen_arm_feature_definition"
    manifest["imputation_policy"] = "LightGBM native missing for materialized columns; live unresolved required columns scored NaN"

    parity_rows: list[dict[str, Any]] = []
    for head in sorted(selected_by_head):
        group = dual.loc[dual["head"].astype(str).eq(head)].copy()
        candidate_ok = group["candidate_score"].notna()
        key_dupes = int(group.duplicated(["head", "timestamp", "symbol", "strategy_id"]).sum())
        parity_rows.append(
            {
                "head": head,
                "baseline_rows": int(len(group)),
                "candidate_rows": int(candidate_ok.sum()),
                "candidate_rows_equal_baseline": bool(int(candidate_ok.sum()) == int(len(group)) and len(group) > 0),
                "baseline_timestamps": int(group["timestamp"].nunique()) if not group.empty else 0,
                "candidate_timestamps": int(group.loc[candidate_ok, "timestamp"].nunique()) if not group.empty else 0,
                "candidate_timestamps_equal_baseline": bool(
                    int(group["timestamp"].nunique()) == int(group.loc[candidate_ok, "timestamp"].nunique()) and len(group) > 0
                ),
                "duplicate_row_keys": key_dupes,
                "no_duplicate_row_keys": bool(key_dupes == 0),
                "mean_candidate_feature_coverage": float(pd.to_numeric(group.get("candidate_feature_coverage"), errors="coerce").mean())
                if not group.empty
                else np.nan,
                "max_candidate_fallback_count": int(pd.to_numeric(group.get("candidate_fallback_count"), errors="coerce").max())
                if not group.empty and group.get("candidate_fallback_count") is not None
                else 0,
            }
        )
    parity = pd.DataFrame(parity_rows)
    missing_feature_audit = (
        pd.concat(missing_feature_frames, ignore_index=True)
        if missing_feature_frames
        else pd.DataFrame(columns=["head", "source", "feature", "present", "missing_rows", "missing_rate", "finite_rate"])
    )
    audit = {
        "status": "sealed_scores_ready_for_label_merge"
        if not parity.empty
        and parity["candidate_rows_equal_baseline"].all()
        and parity["candidate_timestamps_equal_baseline"].all()
        and parity["no_duplicate_row_keys"].all()
        else "candidate_scoring_incomplete",
        "training_cutoff": cutoff.isoformat(),
        "score_window_start": score_start.isoformat(),
        "score_window_end_exclusive": score_end_exclusive.isoformat(),
        "labels_read_or_merged": False,
        "ledger_audit": ledger_audit,
        "parity": parity.to_dict(orient="records"),
        "feature_availability": (
            missing_feature_audit.sort_values(["head", "missing_rate", "feature"], ascending=[True, False, True])
            .head(200)
            .to_dict(orient="records")
        ),
        "manifest_rows": manifest.to_dict(orient="records"),
    }
    manifest.to_csv(out_dir / "candidate_freeze_manifest.csv", index=False)
    (out_dir / "candidate_freeze_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=_json_default) + "\n")
    dual.to_parquet(out_dir / "blind_candidate_dual_scores.parquet", index=False)
    dual.to_csv(out_dir / "blind_candidate_dual_scores.csv", index=False)
    parity.to_csv(out_dir / "blind_candidate_parity_audit.csv", index=False)
    missing_feature_audit.to_csv(out_dir / "candidate_feature_availability_audit.csv", index=False)
    report_lines = [
        "# J4/J5 Contextual Meta Candidate Materialization",
        "",
        f"- Status: `{audit['status']}`",
        f"- Training cutoff: `{cutoff.isoformat()}`",
        f"- Score window: `{score_start.isoformat()}` to `{score_end_exclusive.isoformat()}` exclusive",
        f"- Labels read or merged: `{audit['labels_read_or_merged']}`",
        "",
        "## Candidate Freeze Manifest",
        "",
        manifest[[
            "head",
            "selected_arm",
            "model_hash",
            "feature_contract_hash",
            "transformer_hash",
            "feature_count",
            "feature_count_before_live_filter",
            "live_feature_filter_dropped",
            "train_rows",
            "train_start",
            "train_end",
            "max_depth",
        ]].to_markdown(index=False),
        "",
        "## Blind Score Parity",
        "",
        parity.to_markdown(index=False),
        "",
        "## Worst Feature Availability",
        "",
        missing_feature_audit.sort_values(["head", "missing_rate", "feature"], ascending=[True, False, True])
        .groupby("head", sort=False)
        .head(15)
        .to_markdown(index=False),
        "",
    ]
    (out_dir / "candidate_materialization_report.md").write_text("\n".join(report_lines))
    print(json.dumps({"status": audit["status"], "output_dir": str(out_dir)}, default=_json_default))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER_ROOT)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument(
        "--transform-cache",
        default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet",
    )
    parser.add_argument(
        "--canonical-reduction",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv",
    )
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--training-cutoff", default="2026-06-15T04:00:00+00:00")
    parser.add_argument("--score-start", default="2026-06-16")
    parser.add_argument("--score-end", default="2026-06-22")
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-regime-columns", type=int, default=80)
    parser.add_argument(
        "--min-live-feature-coverage",
        type=float,
        default=1.0,
        help="Minimum label-blind OOS finite coverage required for a candidate input column.",
    )
    parser.add_argument(
        "--no-enforce-live-feature-contract",
        action="store_true",
        help="Disable dropping candidate input columns that are not finite on OOS score rows.",
    )
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()
    args.enforce_live_feature_contract = not bool(args.no_enforce_live_feature_contract)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
