#!/usr/bin/env python3
"""Run a small, explicitly-enabled unsupervised regime-learning POC."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.unsupervised_regime_learning.context_features import (
    build_regime_context_feature_frame,
    generate_signal_regime_interaction_features,
    regime_outputs_from_artifact,
)
from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    stratified_period_sample_positions,
)
from extreme_price_movements.unsupervised_regime_learning.lgbm_feature_filter import (
    RegimeFeatureLGBMFilterConfig,
    extract_lgbm_reuse_contract,
    select_regime_lgbm_addon_features,
)
from extreme_price_movements.unsupervised_regime_learning.pipeline import (
    fit_unsupervised_regime_learning_features,
)
from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    save_advanced_regime_learning_artifact,
)
from extreme_price_movements.unsupervised_regime_learning.validation import (
    regime_pipeline_validation_summary,
    validate_regime_learning_artifact,
)


def _env_bool(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(data)), indent=2, sort_keys=True))


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        empty = frame if len(frame.columns) else pd.DataFrame({"__empty__": pd.Series(dtype=np.float32)})
        csv_path = path.with_suffix(".csv")
        empty.to_csv(csv_path, index=False)
        if path != csv_path and path.exists():
            path.unlink()
        return
    try:
        frame.to_parquet(path, index=True)
        csv_path = path.with_suffix(".csv")
        if csv_path != path and csv_path.exists():
            csv_path.unlink()
    except Exception:
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=True)
        if path != csv_path and path.exists():
            path.unlink()


def _latest_dir_with(root: Path, pattern: str) -> str:
    if not root.exists():
        return ""
    candidates = [p for p in root.iterdir() if p.is_dir() and list(p.glob(pattern))]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda p: p.name)[-1].name


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return [str(name) for name in pq.ParquetFile(path).schema.names]
    except Exception:
        return [str(col) for col in pd.read_parquet(path).columns]


def _symbol_from_feature_path(path: Path) -> str:
    raw = path.name.removeprefix("symbol=").removesuffix(".parquet")
    left, sep, right = raw.partition(":")
    left = left.replace("_", "/")
    return f"{left}:{right}" if sep else left


def _feature_path_for_symbol(feature_dir: Path, symbol: str) -> Path:
    return feature_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _load_saved_contract(artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.exists():
        return {"selected_features": [], "params": {}, "stage": "train_base", "source": "missing"}
    try:
        with artifact_path.open("rb") as fh:
            artifact = pickle.load(fh)
        return extract_lgbm_reuse_contract(artifact, stage="train_base")
    except Exception as exc:
        return {
            "selected_features": [],
            "params": {},
            "stage": "train_base",
            "source": f"load_error:{type(exc).__name__}:{exc}",
        }


def _label_head_terms(label_path: Path | str | None) -> dict[str, Any]:
    name = Path(label_path).stem if label_path is not None else ""
    for prefix in ("train_", "test_", "labels_", "label_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    horizon = ""
    base = name
    match = re.match(r"^(?P<base>.+)_(?P<horizon>\d+)$", name)
    if match:
        base = str(match.group("base"))
        horizon = str(match.group("horizon"))
    normalized_base = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    terms = [normalized_base] if normalized_base else []
    if horizon:
        terms.append(f"{normalized_base}_h{horizon}")
    return {
        "label_stem": Path(label_path).stem if label_path is not None else "",
        "base_slug": normalized_base,
        "horizon": horizon,
        "terms": terms,
    }


def _select_oof_prediction_columns(
    columns: Sequence[str],
    label_path: Path | str | None,
) -> tuple[list[str], dict[str, Any]]:
    pred_cols = [
        str(col)
        for col in columns
        if str(col).lower() not in {"timestamp", "ts", "symbol", "asset"}
        and "sigma" not in str(col).lower()
    ]
    terms = _label_head_terms(label_path)
    base_slug = str(terms.get("base_slug") or "")
    horizon = str(terms.get("horizon") or "")
    if not pred_cols:
        return [], {**terms, "match_type": "none", "candidate_columns": 0}
    normalized = {
        col: re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")
        for col in pred_cols
    }
    strict: list[str] = []
    if base_slug and horizon:
        suffix = f"{base_slug}_h{horizon}"
        strict = [
            col
            for col, norm in normalized.items()
            if base_slug in norm and suffix in norm
        ]
    if strict:
        return strict, {
            **terms,
            "match_type": "label_head_and_horizon",
            "candidate_columns": int(len(pred_cols)),
            "selected_columns": int(len(strict)),
        }
    base_matches = [
        col
        for col, norm in normalized.items()
        if base_slug and base_slug in norm
    ]
    if base_matches:
        return base_matches, {
            **terms,
            "match_type": "label_head",
            "candidate_columns": int(len(pred_cols)),
            "selected_columns": int(len(base_matches)),
        }
    return pred_cols, {
        **terms,
        "match_type": "fallback_all_oof_heads",
        "candidate_columns": int(len(pred_cols)),
        "selected_columns": int(len(pred_cols)),
    }


def _frame_row_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns or "symbol" not in frame.columns:
        return pd.DataFrame(columns=["timestamp", "symbol"])
    keys = frame[["timestamp", "symbol"]].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
    keys["symbol"] = keys["symbol"].astype(str)
    return keys.loc[keys["timestamp"].notna() & keys["symbol"].ne("")].reset_index(drop=True)


def _score_base_oof_candidate(
    oof_path: Path,
    sample_keys: pd.DataFrame,
    *,
    label_path: Path | str | None,
) -> dict[str, Any]:
    run_id = oof_path.parent.parent.name
    try:
        cols = _parquet_columns(oof_path)
        ts_col = "timestamp" if "timestamp" in cols else "ts" if "ts" in cols else ""
        symbol_col = "symbol" if "symbol" in cols else "asset" if "asset" in cols else ""
        pred_cols, pred_diag = _select_oof_prediction_columns(cols, label_path)
        if not ts_col or not symbol_col or not pred_cols or sample_keys.empty:
            return {
                "run_id": run_id,
                "path": str(oof_path),
                "status": "missing_required_columns",
                "key_coverage": 0.0,
                "finite_coverage": 0.0,
                "prediction_column_match": pred_diag,
            }
        oof = pd.read_parquet(oof_path, columns=[ts_col, symbol_col, *pred_cols])
        oof = oof.rename(columns={ts_col: "timestamp", symbol_col: "symbol"})
        oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True, errors="coerce")
        oof["symbol"] = oof["symbol"].astype(str)
        numeric = oof[pred_cols].apply(pd.to_numeric, errors="coerce")
        oof["_finite_oof"] = numeric.notna().any(axis=1)
        grouped = (
            oof[["timestamp", "symbol", "_finite_oof"]]
            .dropna(subset=["timestamp", "symbol"])
            .groupby(["timestamp", "symbol"], sort=False)["_finite_oof"]
            .max()
            .reset_index()
        )
        merged = sample_keys.merge(grouped.assign(_key_hit=1), on=["timestamp", "symbol"], how="left")
        key_hit = merged["_key_hit"].notna().to_numpy(dtype=bool)
        finite = merged["_finite_oof"].eq(True).to_numpy(dtype=bool)
        return {
            "run_id": run_id,
            "path": str(oof_path),
            "status": "completed",
            "rows": int(len(oof)),
            "unique_keys": int(len(grouped)),
            "sample_rows": int(len(sample_keys)),
            "key_overlap_count": int(key_hit.sum()),
            "finite_overlap_count": int(finite.sum()),
            "key_coverage": float(np.mean(key_hit)) if len(sample_keys) else 0.0,
            "finite_coverage": float(np.mean(finite)) if len(sample_keys) else 0.0,
            "prediction_columns": int(len(pred_cols)),
            "prediction_column_match": pred_diag,
        }
    except Exception as exc:
        return {
            "run_id": run_id,
            "path": str(oof_path),
            "status": "error",
            "key_coverage": 0.0,
            "finite_coverage": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _select_base_run_by_oof_overlap(
    data_root: Path,
    sample_keys: pd.DataFrame,
    *,
    label_path: Path | str | None,
) -> tuple[str, dict[str, Any]]:
    artifact_root = data_root / "artifacts"
    candidates = sorted(artifact_root.glob("*/oof/base_oof_all.parquet"))
    scored: list[dict[str, Any]] = []
    for path in candidates:
        run_dir = path.parent.parent
        if not (run_dir / "base_models_intermediate.pkl").exists():
            continue
        scored.append(_score_base_oof_candidate(path, sample_keys, label_path=label_path))
    if not scored:
        return "", {"status": "no_base_oof_candidates", "candidate_count": 0}
    def sort_key(row: Mapping[str, Any]) -> tuple[float, int, float, str]:
        match_type = str((row.get("prediction_column_match") or {}).get("match_type") or "")
        match_score = 2 if match_type == "label_head_and_horizon" else 1 if match_type == "label_head" else 0
        return (
            float(row.get("finite_coverage") or 0.0),
            int(match_score),
            float(row.get("key_coverage") or 0.0),
            str(row.get("run_id") or ""),
        )
    ordered = sorted(scored, key=sort_key, reverse=True)
    best = ordered[0]
    return str(best.get("run_id") or ""), {
        "status": "completed",
        "selected_run_id": str(best.get("run_id") or ""),
        "candidate_count": int(len(scored)),
        "selected": best,
        "top_candidates": ordered[:8],
    }


def _load_aligned_base_oof_predictions(
    artifact_dir: Path,
    sample: pd.DataFrame,
    *,
    label_path: Path | str | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    path = artifact_dir / "oof" / "base_oof_all.parquet"
    empty = pd.Series(np.nan, index=sample.index, dtype=np.float32)
    if not path.exists():
        return empty, {"status": "missing", "path": str(path), "coverage": 0.0}
    try:
        cols = _parquet_columns(path)
        ts_col = "timestamp" if "timestamp" in cols else "ts" if "ts" in cols else ""
        symbol_col = "symbol" if "symbol" in cols else "asset" if "asset" in cols else ""
        pred_cols, pred_diag = _select_oof_prediction_columns(cols, label_path)
        if not ts_col or not symbol_col or not pred_cols:
            return empty, {
                "status": "missing_required_columns",
                "path": str(path),
                "coverage": 0.0,
                "columns": cols[:20],
                "prediction_column_match": pred_diag,
            }
        oof = pd.read_parquet(path, columns=[ts_col, symbol_col, *pred_cols])
        oof = oof.rename(columns={ts_col: "timestamp", symbol_col: "symbol"})
        oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True, errors="coerce")
        oof["symbol"] = oof["symbol"].astype(str)
        numeric = oof[pred_cols].apply(pd.to_numeric, errors="coerce")
        oof["base_oof_mean"] = numeric.mean(axis=1, skipna=True).astype(np.float32)
        oof = oof[["timestamp", "symbol", "base_oof_mean"]].dropna(subset=["timestamp", "symbol"])
        oof = oof.groupby(["timestamp", "symbol"], sort=False)["base_oof_mean"].mean().reset_index()
        keys = sample[["timestamp", "symbol"]].copy()
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True, errors="coerce")
        keys["symbol"] = keys["symbol"].astype(str)
        merged = keys.merge(oof, on=["timestamp", "symbol"], how="left")
        out = pd.Series(merged["base_oof_mean"].to_numpy(dtype=np.float32), index=sample.index)
        coverage = float(np.mean(np.isfinite(out.to_numpy(dtype=np.float32)))) if len(out) else 0.0
        return out, {
            "status": "completed",
            "path": str(path),
            "rows": int(len(oof)),
            "prediction_columns": int(len(pred_cols)),
            "selected_prediction_columns": list(pred_cols),
            "prediction_column_match": pred_diag,
            "coverage": coverage,
        }
    except Exception as exc:
        return empty, {
            "status": "load_error",
            "path": str(path),
            "coverage": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _find_label_file(artifact_dir: Path) -> Path | None:
    label_dir = artifact_dir / "labels"
    if not label_dir.exists():
        return None
    best: tuple[int, Path] | None = None
    for path in sorted(label_dir.glob("*.parquet")):
        if path.name == "labels_manifest.parquet":
            continue
        cols = set(_parquet_columns(path))
        if {"__ts__", "__symbol__", "__y_bin__"}.issubset(cols):
            try:
                rows = int(pd.read_parquet(path, columns=["__y_bin__"]).shape[0])
            except Exception:
                rows = 0
            if best is None or rows > best[0]:
                best = (rows, path)
    return best[1] if best else None


def _sample_labels(
    label_path: Path,
    *,
    feature_symbols: set[str],
    max_rows: int,
    sampling_mode: str = "contiguous_panels",
) -> pd.DataFrame:
    cols = [col for col in ["__ts__", "__symbol__", "__y_bin__", "__y_ret__", "__w__"] if col in _parquet_columns(label_path)]
    labels = pd.read_parquet(label_path, columns=cols)
    labels = labels.rename(
        columns={
            "__ts__": "timestamp",
            "__symbol__": "symbol",
            "__y_bin__": "target",
            "__y_ret__": "target_return",
            "__w__": "sample_weight",
        }
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True, errors="coerce")
    labels["symbol"] = labels["symbol"].astype(str)
    labels = labels.loc[
        labels["timestamp"].notna()
        & labels["symbol"].isin(feature_symbols)
        & pd.to_numeric(labels["target"], errors="coerce").notna()
    ].copy()
    labels = labels.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    if len(labels) > int(max_rows):
        if str(sampling_mode).strip().lower() == "contiguous_panels":
            positions = _stratified_contiguous_panel_positions(
                labels,
                max_rows=int(max_rows),
                timestamp_col="timestamp",
                n_periods=12,
            )
        else:
            positions = stratified_period_sample_positions(
                labels,
                np.arange(len(labels), dtype=np.int64),
                max_rows=int(max_rows),
                timestamp_col="timestamp",
                symbol_col="symbol",
                n_periods=12,
            )
        labels = labels.iloc[positions].sort_values(["timestamp", "symbol"], kind="mergesort")
    labels["target"] = pd.to_numeric(labels["target"], errors="coerce").fillna(0).astype(np.int8)
    if "sample_weight" in labels.columns:
        labels["sample_weight"] = pd.to_numeric(labels["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32)
    return labels.reset_index(drop=True)


def _stratified_contiguous_panel_positions(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    timestamp_col: str,
    n_periods: int = 12,
) -> np.ndarray:
    """Select representative contiguous timestamp panels.

    Sparse row sampling is cheap, but it breaks the temporal texture needed by
    HMMs, dwell features, autocorrelation, covariance geometry, and transition
    hazards. This sampler keeps whole timestamp panels inside each broad period
    so regime fitting sees contiguous local histories and cross-sections.
    """

    n = int(len(frame))
    cap = int(max_rows or 0)
    if cap <= 0 or n <= cap or timestamp_col not in frame.columns:
        return np.arange(n, dtype=np.int64)
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    valid = ts.notna().to_numpy(dtype=bool)
    if not bool(valid.any()):
        return np.arange(min(n, cap), dtype=np.int64)
    valid_pos = np.flatnonzero(valid).astype(np.int64, copy=False)
    unique_ts = pd.Index(pd.unique(ts.iloc[valid_pos])).sort_values()
    if len(unique_ts) == 0:
        return np.arange(min(n, cap), dtype=np.int64)
    row_counts = ts.groupby(ts).size().reindex(unique_ts).fillna(0).astype(int)
    periods = [pd.Index(block) for block in np.array_split(unique_ts, max(1, int(n_periods))) if len(block)]
    total_valid = int(row_counts.sum())
    selected_ts: set[pd.Timestamp] = set()
    for period_i, period_ts in enumerate(periods):
        period_rows = int(row_counts.reindex(period_ts).sum())
        if period_rows <= 0:
            continue
        target_rows = max(1, int(np.floor(float(cap) * float(period_rows) / max(float(total_valid), 1.0))))
        median_panel_rows = max(1, int(np.nanmedian(row_counts.reindex(period_ts).to_numpy(dtype=np.float64))))
        target_panels = max(1, int(np.floor(float(target_rows) / float(median_panel_rows))))
        target_panels = min(int(target_panels), len(period_ts))
        center = len(period_ts) // 2
        start = max(0, center - target_panels // 2)
        end = min(len(period_ts), start + target_panels)
        start = max(0, end - target_panels)
        for stamp in period_ts[start:end]:
            selected_ts.add(pd.Timestamp(stamp))
    if not selected_ts:
        selected_ts.add(pd.Timestamp(unique_ts[len(unique_ts) // 2]))
    ordered_ts = [pd.Timestamp(stamp) for stamp in unique_ts if pd.Timestamp(stamp) in selected_ts]
    kept_ts: list[pd.Timestamp] = []
    rows_used = 0
    for stamp in ordered_ts:
        panel_rows = int(row_counts.get(stamp, 0))
        if panel_rows <= 0:
            continue
        if kept_ts and rows_used + panel_rows > cap:
            continue
        kept_ts.append(stamp)
        rows_used += panel_rows
        if rows_used >= cap:
            break
    if not kept_ts:
        kept_ts = [ordered_ts[0]]
    mask = ts.isin(kept_ts).to_numpy(dtype=bool)
    out = np.flatnonzero(mask).astype(np.int64)
    return out if out.size else np.arange(min(n, cap), dtype=np.int64)


def _load_feature_sample(
    feature_dir: Path,
    labels: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    requested = list(dict.fromkeys(str(col) for col in feature_columns if str(col)))
    for symbol, group in labels.groupby("symbol", sort=False):
        path = _feature_path_for_symbol(feature_dir, str(symbol))
        if not path.exists():
            continue
        available = set(_parquet_columns(path))
        present = [col for col in requested if col in available]
        if not present:
            continue
        values = pd.read_parquet(path, columns=present)
        idx = pd.to_datetime(values.index, utc=True, errors="coerce")
        values = values.loc[idx.isin(set(group["timestamp"]))].copy()
        if values.empty:
            continue
        values["timestamp"] = pd.to_datetime(values.index, utc=True, errors="coerce")
        values["symbol"] = str(symbol)
        rows.append(values.reset_index(drop=True))
    if not rows:
        raise RuntimeError("No feature rows matched the sampled labels.")
    features = pd.concat(rows, axis=0, ignore_index=True)
    merged = labels.merge(features, on=["timestamp", "symbol"], how="inner")
    merged = merged.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("Feature/label merge produced no rows.")
    return merged


def _poc_regime_cfg(base_cfg: Mapping[str, Any], output_dir: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    poc = dict(cfg.get("proof_of_concept") or {})
    row_cap = int(max_rows or poc.get("max_rows", 3000) or 3000)
    classifier_rows = int(poc.get("poc_classifier_rows", min(row_cap, 12000)))
    ae_train_rows = int(poc.get("poc_ae_train_rows", min(row_cap, 12000)))
    assessment_auc_rows = int(poc.get("poc_assessment_auc_rows", min(row_cap, 12000)))
    embedding_fit_rows = int(poc.get("poc_embedding_fit_rows", min(row_cap, 3000)))
    cfg.setdefault("quality", {})
    cfg["quality"].update({"warmup_rows": 5, "min_good_row_fraction": 0.75})
    cfg.setdefault("primitive_selection", {})
    cfg["primitive_selection"].update(
        {
            "target_features": int(poc.get("poc_feature_target_primitives", 32)),
            "spearman_max_corr_rows": 3000,
            "spearman_corr_time_bins": 8,
            "block_hours": 24,
            "min_block_rows": 8,
        }
    )
    cfg.setdefault("operator_selection", {})
    cfg["operator_selection"].update(
        {
            "target_features": int(poc.get("poc_operator_target_features", 64)),
            "spearman_max_corr_rows": 3000,
            "spearman_corr_time_bins": 8,
            "max_pair_features_for_spearman": 256,
        }
    )
    cfg.setdefault("operators", {})
    cfg["operators"].update(
        {
            "quantile_window": 24,
            "autocorr_window": 24,
            "pair_window": 24,
            "eigen_window": 24,
            "min_periods": 8,
            "max_pair_candidates_for_generation": int(poc.get("poc_pair_candidates", 96)),
            "svd_walk_forward_block_hours": 24 * 14,
            "svd_min_prior_rows": 64,
            "svd_max_reference_rows": 3000,
            "knn_max_reference_rows": 1500,
            "svd_sample_time_bins": 8,
            "svd_components": [8, 16],
            "knn_svd_components": 16,
            "knn_neighbors": 10,
        }
    )
    cfg.setdefault("regime_models", {})
    cfg["regime_models"].update(
        {
            "enabled": True,
            "max_rows": row_cap,
            "sample_time_bins": 8,
            "stability_bootstraps": int(poc.get("poc_stability_bootstraps", 3)),
            "stability_top_m": int(poc.get("poc_stability_top_m", 24)),
            "n_estimators": int(poc.get("poc_n_estimators", 24)),
            "leaf_embedding_max_trees": int(poc.get("poc_n_estimators", 24)),
            "max_classifier_rows": classifier_rows,
            "n_regimes": int(poc.get("poc_n_regimes", 3)),
            "primary_trading_horizon_hours": 6,
            "transition_change_horizons": (1, 4, 6, 12, 24),
            "umap_embedding_max_rows": embedding_fit_rows,
            "spectral_embedding_max_rows": embedding_fit_rows,
            "spectral_clustering_max_rows": embedding_fit_rows,
            "mfa_regimes": int(poc.get("poc_n_regimes", 3)),
            "mfa_factors": 2,
            "mfa_max_iter": int(poc.get("poc_mfa_max_iter", 4)),
            "ae_epochs": int(poc.get("poc_ae_epochs", 2)),
            "ae_latent_dim": 4,
            "ae_hidden_dim": 12,
            "ae_batch_size": 128,
            "ae_max_train_rows": ae_train_rows,
            "min_regime_duration": 2,
            "regime_assessment_bootstraps": 2,
            "regime_assessment_windows": 3,
            "regime_assessment_null_repeats": 1,
            "regime_assessment_max_auc_rows": assessment_auc_rows,
            "regime_assessment_max_robustness_rows": 1000,
            "regime_assessment_max_geometry_rows_per_regime": 256,
            "artifact_output_dir": str(output_dir / "advanced_regime_learning"),
        }
    )
    return cfg


def _summarize_metrics(
    *,
    sample: pd.DataFrame,
    result: Any,
    validation_report: pd.DataFrame,
    context_diag: Mapping[str, Any],
    interaction_diag: Mapping[str, Any],
    lgbm_diag: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = result.regime_models
    regime_diag = getattr(artifact, "regime_diagnostics", pd.DataFrame()) if artifact is not None else pd.DataFrame()
    top_methods: list[dict[str, Any]] = []
    score_col = "UsefulRegimeScore" if "UsefulRegimeScore" in regime_diag.columns else "TotalScore"
    if isinstance(regime_diag, pd.DataFrame) and not regime_diag.empty and score_col in regime_diag.columns:
        cols = [
            col
            for col in [
                "method",
                "regime_family",
                "regime_objective",
                "assessment_cluster_method",
                "UsefulRegimeScore",
                "ModelHelpfulness",
                "ConditionalSignalLearnability",
                "RegimeConditionedSignalAUC",
                "SignalOnlyFutureStructureAUC",
                "IncrementalConditionalSignalAUC",
                "FutureStructureAUC",
                "TrendVolFutureStructureAUC",
                "IncrementalFutureStructureAUC",
                "TotalScore",
                "NonTriviality",
                "OOS_Stability",
                "Dwell_Quality",
                "Transition_Stability",
                "Feature_Stability",
                "Null_Robustness",
                "Window_Robustness",
                "Geometry_Separation",
                "min_support",
                "regime_count",
            ]
            if col in regime_diag.columns
        ]
        top_methods = (
            regime_diag.sort_values(score_col, ascending=False, kind="mergesort")
            .head(8)[cols]
            .to_dict("records")
        )
    return {
        "sample": {
            "rows": int(len(sample)),
            "symbols": int(sample["symbol"].nunique()) if "symbol" in sample.columns else 0,
            "start": str(sample["timestamp"].min()) if "timestamp" in sample.columns else None,
            "end": str(sample["timestamp"].max()) if "timestamp" in sample.columns else None,
            "target_mean": float(pd.to_numeric(sample.get("target", pd.Series(dtype=float)), errors="coerce").mean()),
        },
        "feature_selection": {
            "primitive_selected": int(len(result.primitives.selected_features)),
            "operator_selected": int(len(result.operators.selected_operator_features)),
            "pair_selected": int(len(result.operators.selected_pair_features)),
            "svd_knn_features": int(len(result.operators.svd_knn_features)),
            "final_feature_count": int(len(result.final_feature_columns)),
        },
        "pipeline_steps": result.pipeline_steps.to_dict("records"),
        "advanced_validation": regime_pipeline_validation_summary(validation_report),
        "context_features": dict(context_diag),
        "signal_regime_interactions": dict(interaction_diag),
        "top_regime_methods": top_methods,
        "lgbm_addon_filter": dict(lgbm_diag),
    }


def _write_analysis(path: Path, summary: Mapping[str, Any]) -> None:
    lines = ["# Unsupervised Regime Learning POC Analysis", ""]
    sample = dict(summary.get("sample") or {})
    lines.append(
        f"Sample: {sample.get('rows', 0)} rows, {sample.get('symbols', 0)} symbols, "
        f"{sample.get('start')} to {sample.get('end')}, target_mean={sample.get('target_mean'):.4f}."
    )
    fs = dict(summary.get("feature_selection") or {})
    lines.append(
        "Feature set: "
        f"{fs.get('primitive_selected', 0)} primitives, {fs.get('operator_selected', 0)} regular operators, "
        f"{fs.get('pair_selected', 0)} pair operators, {fs.get('svd_knn_features', 0)} SVD/KNN, "
        f"{fs.get('final_feature_count', 0)} final features."
    )
    val = dict(summary.get("advanced_validation") or {})
    lines.append(
        f"Validation: passed={val.get('passed')} checks={val.get('check_count')} failed={val.get('failed_count')}."
    )
    lines.append("")
    lines.append("## Top Regime Methods")
    for row in summary.get("top_regime_methods", []) or []:
        lines.append(
            "- "
            f"{row.get('method')}: UsefulRegimeScore={float(row.get('UsefulRegimeScore', row.get('TotalScore', 0.0))):.4f}, "
            f"ModelHelpfulness={float(row.get('ModelHelpfulness', 0.0)):.4f}, "
            f"ConditionalSignalLearnability={float(row.get('ConditionalSignalLearnability', 0.0)):.4f}, "
            f"IncrementalConditionalSignalAUC={float(row.get('IncrementalConditionalSignalAUC', 0.0)):.4f}, "
            f"IncrementalFutureStructureAUC={float(row.get('IncrementalFutureStructureAUC', 0.0)):.4f}, "
            f"TotalScore={float(row.get('TotalScore', 0.0)):.4f}, "
            f"NonTriviality={float(row.get('NonTriviality', 0.0)):.4f}, "
            f"OOS_Stability={float(row.get('OOS_Stability', 0.0)):.4f}, "
            f"Feature_Stability={float(row.get('Feature_Stability', 0.0)):.4f}, "
            f"Geometry={float(row.get('Geometry_Separation', 0.0)):.4f}, "
            f"min_support={float(row.get('min_support', 0.0)):.4f}."
        )
    context = dict(summary.get("context_features") or {})
    lines.append("")
    lines.append(
        "Context features: "
        f"{context.get('output_feature_count', 0)} columns across groups {context.get('groups', {})}."
    )
    interactions = dict(summary.get("signal_regime_interactions") or {})
    if interactions:
        lines.append(
            "Signal x regime interactions: "
            f"status={interactions.get('status')}, output_features={interactions.get('output_feature_count', 0)}, "
            f"signals={interactions.get('selected_signal_features', 0)}, "
            f"regimes={interactions.get('selected_regime_features', 0)}."
        )
    lgbm = dict(summary.get("lgbm_addon_filter") or {})
    lines.append(
        "LGBM add-on filter: "
        f"status={lgbm.get('status')}, candidates={lgbm.get('candidate_regime_feature_count', 0)}, "
        f"selected={lgbm.get('selected_feature_count', 0)}, folds={lgbm.get('fold_count', 0)}."
    )
    if val.get("failed_checks"):
        lines.append("")
        lines.append("## Failed Validation Checks")
        for row in val.get("failed_checks", []):
            lines.append(f"- {row.get('step')} / {row.get('check')}: {row.get('message')}")
    path.write_text("\n".join(lines) + "\n")


def _feature_records(frame: pd.DataFrame, *, max_rows: int = 30) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    preferred = [
        "feature",
        "source",
        "context_role",
        "rank_score",
        "risk_gate_acceptance_score",
        "risk_gate_acceptance_pass",
        "risk_budget_scaler_score",
        "risk_budget_scaler_pass",
        "median_risk_budget_scaled_hr_lift",
        "median_risk_budget_high_low_hr_lift",
        "median_risk_budget_failure_avoidance",
        "risk_budget_monotonicity_mean",
        "signal_uplift_mean_abs",
        "signal_uplift_context_pass",
        "opportunity_context_pass",
        "context_helper_candidate_pass",
        "context_helper_reason",
        "median_oof_failure_lift",
        "oof_failure_alignment_pass",
        "effective_structural_pass_rate",
        "pre_redundancy_keep",
        "redundancy_keep",
        "source_keep",
    ]
    cols = [col for col in preferred if col in frame.columns]
    return frame.head(int(max_rows)).reindex(columns=cols).to_dict("records")


def _split_lgbm_feature_buckets(
    feature_metrics: pd.DataFrame,
    selected_features: Sequence[str],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Split add-on outputs by intended downstream use."""

    if not isinstance(feature_metrics, pd.DataFrame) or feature_metrics.empty:
        empty = pd.DataFrame()
        return {
            "selected_additive_features": [],
            "production_risk_gates": [],
            "context_portfolio_scalers": [],
            "production_opportunity_gates": [],
            "candidate_context_helpers": [],
            "exploratory_context_interactions": [],
            "diagnostic_only_regime_features": [],
            "counts": {
                "selected_additive_features": 0,
                "accepted_oof_aligned_risk_gates": 0,
                "production_risk_gates": 0,
                "context_portfolio_scalers": 0,
                "production_opportunity_gates": 0,
                "candidate_context_helpers": 0,
                "exploratory_context_interactions": 0,
                "diagnostic_only_regime_features": 0,
            },
        }, {
            "selected_additive_features": empty,
            "production_risk_gates": empty,
            "context_portfolio_scalers": empty,
            "production_opportunity_gates": empty,
            "candidate_context_helpers": empty,
            "exploratory_context_interactions": empty,
            "diagnostic_only_regime_features": empty,
        }

    metrics = feature_metrics.copy()
    for col, default in {
        "risk_budget_scaler_score": 0.0,
        "risk_budget_scaler_pass": False,
    }.items():
        if col not in metrics.columns:
            metrics[col] = default
    selected_set = set(str(feature) for feature in selected_features)
    metrics["selected_final"] = metrics["feature"].astype(str).isin(selected_set)
    source = metrics.get("source", pd.Series("", index=metrics.index)).astype(str)
    role = metrics.get("context_role", pd.Series("", index=metrics.index)).astype(str)
    risk_gate_pass = metrics.get("risk_gate_acceptance_pass", pd.Series(False, index=metrics.index)).astype(bool)
    risk_budget_pass = metrics.get("risk_budget_scaler_pass", pd.Series(False, index=metrics.index)).astype(bool)
    oof_aligned = metrics.get("oof_failure_alignment_pass", pd.Series(False, index=metrics.index)).astype(bool)
    signal_uplift = metrics.get("signal_uplift_context_pass", pd.Series(False, index=metrics.index)).astype(bool)
    opportunity_pass = metrics.get("opportunity_context_pass", pd.Series(False, index=metrics.index)).astype(bool)
    context_helper_pass = metrics.get("context_helper_candidate_pass", pd.Series(False, index=metrics.index)).astype(bool)
    redundancy_keep = metrics.get("redundancy_keep", pd.Series(False, index=metrics.index)).astype(bool)
    source_keep = metrics.get("source_keep", pd.Series(True, index=metrics.index)).astype(bool)

    production_risk = metrics.loc[
        risk_gate_pass
        & oof_aligned
        & role.eq("risk_gate")
    ].sort_values(
        ["risk_gate_acceptance_score", "rank_score"],
        ascending=False,
        kind="mergesort",
    )
    production_risk_features = set(production_risk["feature"].astype(str))

    context_portfolio_scalers = metrics.loc[
        risk_budget_pass
        & redundancy_keep
        & source_keep
    ].sort_values(
        ["risk_budget_scaler_score", "rank_score"],
        ascending=False,
        kind="mergesort",
    )
    context_portfolio_features = set(context_portfolio_scalers["feature"].astype(str))

    production_opportunity = metrics.loc[
        metrics["selected_final"].astype(bool)
        & opportunity_pass
        & redundancy_keep
        & source_keep
        & role.eq("opportunity_gate")
    ].sort_values("rank_score", ascending=False, kind="mergesort")
    production_opportunity_features = set(production_opportunity["feature"].astype(str))

    additive = metrics.loc[
        metrics["selected_final"].astype(bool)
        & ~metrics["feature"].astype(str).isin(production_risk_features)
        & ~metrics["feature"].astype(str).isin(context_portfolio_features)
        & ~metrics["feature"].astype(str).isin(production_opportunity_features)
        & ~source.eq("signal_regime_interaction")
        & ~role.eq("risk_gate")
    ].sort_values("rank_score", ascending=False, kind="mergesort")

    context_helpers = metrics.loc[
        context_helper_pass
        & ~metrics["feature"].astype(str).isin(production_risk_features)
        & ~metrics["feature"].astype(str).isin(context_portfolio_features)
        & ~metrics["feature"].astype(str).isin(production_opportunity_features)
    ].sort_values(
        ["rank_score", "signal_uplift_mean_abs"],
        ascending=False,
        kind="mergesort",
    )

    exploratory = metrics.loc[
        source.eq("signal_regime_interaction")
        & signal_uplift
        & ~metrics["feature"].astype(str).isin(production_risk_features)
        & ~metrics["feature"].astype(str).isin(context_portfolio_features)
        & ~metrics["feature"].astype(str).isin(production_opportunity_features)
    ].sort_values(
        ["rank_score", "signal_uplift_mean_abs"],
        ascending=False,
        kind="mergesort",
    )

    assigned = set(additive["feature"].astype(str))
    assigned.update(production_risk_features)
    assigned.update(context_portfolio_features)
    assigned.update(production_opportunity_features)
    assigned.update(context_helpers["feature"].astype(str))
    assigned.update(exploratory["feature"].astype(str))
    diagnostic = metrics.loc[
        ~metrics["feature"].astype(str).isin(assigned)
    ].sort_values("rank_score", ascending=False, kind="mergesort")

    summary = {
        "criteria": {
            "selected_additive_features": "final selected non-risk-gate features excluding signal-regime interactions",
            "production_risk_gates": "risk_gate_acceptance_pass and oof_failure_alignment_pass",
            "context_portfolio_scalers": "ctx_portfolio features accepted by monotonic risk-budget scaler diagnostics after redundancy/source pruning",
            "production_opportunity_gates": "selected opportunity_gate features with conditional signal-uplift after redundancy/source pruning",
            "candidate_context_helpers": "strict signal-uplift context helpers surfaced for controlled model comparison even when not production-selected",
            "exploratory_context_interactions": "signal_regime_interaction with strict signal_uplift_context_pass but not accepted as OOF-aligned risk gate",
            "diagnostic_only_regime_features": "evaluated regime features not assigned to production or candidate context buckets",
        },
        "counts": {
            "selected_additive_features": int(len(additive)),
            "accepted_oof_aligned_risk_gates": int(len(production_risk)),
            "production_risk_gates": int(len(production_risk)),
            "context_portfolio_scalers": int(len(context_portfolio_scalers)),
            "production_opportunity_gates": int(len(production_opportunity)),
            "candidate_context_helpers": int(len(context_helpers)),
            "exploratory_context_interactions": int(len(exploratory)),
            "diagnostic_only_regime_features": int(len(diagnostic)),
        },
        "selected_additive_features": additive["feature"].astype(str).head(30).tolist(),
        "accepted_oof_aligned_risk_gates": production_risk["feature"].astype(str).head(30).tolist(),
        "production_risk_gates": production_risk["feature"].astype(str).head(30).tolist(),
        "context_portfolio_scalers": context_portfolio_scalers["feature"].astype(str).head(30).tolist(),
        "production_opportunity_gates": production_opportunity["feature"].astype(str).head(30).tolist(),
        "candidate_context_helpers": context_helpers["feature"].astype(str).head(30).tolist(),
        "exploratory_context_interactions": exploratory["feature"].astype(str).head(30).tolist(),
        "diagnostic_only_regime_features": diagnostic["feature"].astype(str).head(30).tolist(),
        "top_records": {
            "selected_additive_features": _feature_records(additive),
            "accepted_oof_aligned_risk_gates": _feature_records(production_risk),
            "production_risk_gates": _feature_records(production_risk),
            "context_portfolio_scalers": _feature_records(context_portfolio_scalers),
            "production_opportunity_gates": _feature_records(production_opportunity),
            "candidate_context_helpers": _feature_records(context_helpers),
            "exploratory_context_interactions": _feature_records(exploratory),
            "diagnostic_only_regime_features": _feature_records(diagnostic),
        },
    }
    return summary, {
        "selected_additive_features": additive,
        "accepted_oof_aligned_risk_gates": production_risk,
        "production_risk_gates": production_risk,
        "context_portfolio_scalers": context_portfolio_scalers,
        "production_opportunity_gates": production_opportunity,
        "candidate_context_helpers": context_helpers,
        "exploratory_context_interactions": exploratory,
        "diagnostic_only_regime_features": diagnostic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-poc", action="store_true", help="Required explicit opt-in.")
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--feature-run-id", default="")
    parser.add_argument("--label-artifact-run-id", default="")
    parser.add_argument("--base-artifact-run-id", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--skip-lgbm-addon", action="store_true")
    parser.add_argument(
        "--sampling-mode",
        default="contiguous_panels",
        choices=["contiguous_panels", "sparse_stratified"],
        help="POC row sampling. contiguous_panels preserves temporal panels for regime fitting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = copy.deepcopy(CFG["UNSUPERVISED_REGIME_LEARNING"])
    poc_cfg = dict(cfg.get("proof_of_concept") or {})
    enabled = bool(poc_cfg.get("enabled", False)) or bool(args.enable_poc) or _env_bool("EPM_UNSUPERVISED_REGIME_POC")
    if not enabled:
        print("POC disabled. Re-run with --enable-poc or EPM_UNSUPERVISED_REGIME_POC=1.")
        return 0

    data_root = Path(args.data_root)
    feature_run_id = args.feature_run_id or str(poc_cfg.get("feature_run_id") or "")
    label_run_id = args.label_artifact_run_id or str(poc_cfg.get("label_artifact_run_id") or "")
    requested_base_run_id = args.base_artifact_run_id or str(poc_cfg.get("base_artifact_run_id") or "")
    base_run_id = requested_base_run_id
    if not feature_run_id:
        feature_run_id = _latest_dir_with(data_root / "features", "*.parquet")
    if not label_run_id:
        label_run_id = _latest_dir_with(data_root / "artifacts", "labels/*.parquet")
    max_rows = int(args.max_rows or poc_cfg.get("max_rows", 3000) or 3000)
    output_base = Path(args.output_dir or str(poc_cfg.get("output_dir") or "data_perp/artifacts/unsupervised_regime_learning_poc"))
    output_dir = output_base / datetime.now(timezone.utc).strftime("poc_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    feature_dir = data_root / "features" / feature_run_id
    label_dir = data_root / "artifacts" / label_run_id
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature run not found: {feature_dir}")
    label_path = _find_label_file(label_dir)
    if label_path is None:
        raise FileNotFoundError(f"No label parquet with __ts__/__symbol__/__y_bin__ in {label_dir / 'labels'}")

    feature_symbols = {_symbol_from_feature_path(path) for path in feature_dir.glob("symbol=*.parquet")}
    labels = _sample_labels(
        label_path,
        feature_symbols=feature_symbols,
        max_rows=max_rows,
        sampling_mode=str(args.sampling_mode),
    )
    base_run_selection: dict[str, Any]
    if base_run_id:
        base_run_selection = {
            "status": "explicit",
            "selected_run_id": base_run_id,
            "requested_run_id": base_run_id,
        }
    else:
        base_run_id, base_run_selection = _select_base_run_by_oof_overlap(
            data_root,
            _frame_row_keys(labels),
            label_path=label_path,
        )
        if not base_run_id:
            base_run_id = _latest_dir_with(data_root / "artifacts", "base_models_intermediate.pkl")
            base_run_selection = {
                **base_run_selection,
                "fallback_run_id": base_run_id,
                "status": "fallback_latest_base_artifact",
            }
    base_artifact = data_root / "artifacts" / base_run_id / "base_models_intermediate.pkl"
    contract = _load_saved_contract(base_artifact)
    base_contract_features = [str(col) for col in contract.get("selected_features", [])]
    requested_features = list(
        dict.fromkeys(
            list(cfg.get("primitive_feature_keys", []))
            + base_contract_features
        )
    )
    sample = _load_feature_sample(feature_dir, labels, requested_features)
    if len(sample) > max_rows:
        if str(args.sampling_mode) == "contiguous_panels":
            pos = _stratified_contiguous_panel_positions(
                sample,
                max_rows=max_rows,
                timestamp_col="timestamp",
                n_periods=12,
            )
        else:
            pos = stratified_period_sample_positions(
                sample,
                np.arange(len(sample), dtype=np.int64),
                max_rows=max_rows,
                timestamp_col="timestamp",
                symbol_col="symbol",
                n_periods=12,
            )
        sample = sample.iloc[pos].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

    base_oof_pred, base_oof_diag = _load_aligned_base_oof_predictions(
        base_artifact.parent,
        sample,
        label_path=label_path,
    )
    run_cfg = _poc_regime_cfg(cfg, output_dir, max_rows=max_rows)
    result = fit_unsupervised_regime_learning_features(
        sample,
        cfg=run_cfg,
        feature_columns=list(cfg.get("primitive_feature_keys", [])),
        regime_assessment_target=sample["target"].to_numpy(dtype=np.float32) if "target" in sample.columns else None,
        regime_assessment_oof_pred=base_oof_pred,
    )
    _write_frame(output_dir / "pipeline_steps.parquet", result.pipeline_steps)
    _write_frame(output_dir / "primitive_quality_report.parquet", result.primitives.quality_report)
    _write_frame(output_dir / "primitive_diagnostics.parquet", result.primitives.diagnostics)
    _write_frame(output_dir / "operator_quality_report.parquet", result.operators.quality_report)
    _write_frame(output_dir / "operator_diagnostics.parquet", result.operators.diagnostics)
    _write_frame(output_dir / "pair_scores.parquet", result.operators.pair_scores)
    _write_json(output_dir / "final_feature_columns.json", {"features": result.final_feature_columns})

    validation_report = pd.DataFrame()
    context_features = pd.DataFrame(index=sample.index)
    context_diag: dict[str, Any] = {"status": "skipped_no_regime_artifact"}
    interaction_features = pd.DataFrame(index=sample.index)
    interaction_diag: dict[str, Any] = {"status": "skipped_no_context_features"}
    if result.regime_models is not None:
        save_advanced_regime_learning_artifact(result.regime_models, output_dir / "advanced_regime_learning")
        validation_report = validate_regime_learning_artifact(result.regime_models)
        _write_frame(output_dir / "advanced_validation_report.parquet", validation_report)
        outputs = regime_outputs_from_artifact(result.regime_models)
        context_features, context_diag = build_regime_context_feature_frame(sample, outputs)
        _write_frame(output_dir / "regime_context_features.parquet", context_features)
        _write_json(output_dir / "regime_context_diagnostics.json", context_diag)

    lgbm_diag: dict[str, Any] = {"status": "skipped"}
    if bool(poc_cfg.get("run_lgbm_addon_filter", True)) and not args.skip_lgbm_addon and not context_features.empty:
        model_frame = pd.concat([sample, result.operators.feature_frame, context_features], axis=1)
        base_features = [feature for feature in base_contract_features if feature in model_frame.columns]
        base_feature_source = "saved_contract"
        if len(base_features) < 5:
            base_features = [feature for feature in result.final_feature_columns if feature in model_frame.columns][:64]
            base_feature_source = "poc_final_feature_fallback"
        interaction_features, interaction_diag = generate_signal_regime_interaction_features(
            model_frame,
            base_features,
            list(context_features.columns),
        )
        _write_frame(output_dir / "signal_regime_interaction_features.parquet", interaction_features)
        _write_json(output_dir / "signal_regime_interaction_diagnostics.json", interaction_diag)
        if not interaction_features.empty:
            model_frame = pd.concat([model_frame, interaction_features], axis=1)
        candidate_regime_features = list(context_features.columns) + list(interaction_features.columns)
        source_map = {col: "signal_regime_interaction" for col in interaction_features.columns}
        lgbm_result = select_regime_lgbm_addon_features(
            model_frame,
            sample["target"].to_numpy(dtype=np.int8),
            base_features=base_features,
            regime_features=candidate_regime_features,
            timestamps=sample["timestamp"].to_numpy(),
            sample_weight=sample["sample_weight"].to_numpy(dtype=np.float32) if "sample_weight" in sample.columns else None,
            base_oof_pred=base_oof_pred,
            reused_model_params=contract.get("params", {}),
            source_map=source_map,
            config=RegimeFeatureLGBMFilterConfig(
                n_folds=7,
                fold_sample_fraction=0.50,
                max_rows=max_rows,
                max_trees=80,
                min_child_samples=20,
                route_max_rows=min(1500, max_rows),
                stratified_period_bins=12,
            ),
        )
        lgbm_diag = dict(lgbm_result.diagnostics)
        lgbm_diag["base_feature_source"] = base_feature_source
        lgbm_diag["base_feature_overlap"] = int(len(base_features))
        lgbm_diag["contract_source"] = contract.get("source")
        lgbm_diag["signal_regime_interaction_feature_count"] = int(interaction_features.shape[1])
        lgbm_diag["base_oof_predictions"] = base_oof_diag
        lgbm_diag["base_run_selection"] = base_run_selection
        bucket_summary, bucket_frames = _split_lgbm_feature_buckets(
            lgbm_result.feature_metrics,
            lgbm_result.selected_features,
        )
        lgbm_diag["feature_buckets"] = bucket_summary
        _write_frame(output_dir / "lgbm_addon_feature_metrics.parquet", lgbm_result.feature_metrics)
        _write_frame(output_dir / "lgbm_addon_fold_metrics.parquet", lgbm_result.fold_metrics)
        _write_frame(output_dir / "lgbm_addon_source_metrics.parquet", lgbm_result.source_metrics)
        _write_frame(
            output_dir / "lgbm_addon_selected_additive_features.parquet",
            bucket_frames["selected_additive_features"],
        )
        _write_frame(
            output_dir / "lgbm_addon_accepted_oof_aligned_risk_gates.parquet",
            bucket_frames["accepted_oof_aligned_risk_gates"],
        )
        _write_frame(
            output_dir / "lgbm_addon_production_risk_gates.parquet",
            bucket_frames["production_risk_gates"],
        )
        _write_frame(
            output_dir / "lgbm_addon_context_portfolio_scalers.parquet",
            bucket_frames["context_portfolio_scalers"],
        )
        _write_frame(
            output_dir / "lgbm_addon_production_opportunity_gates.parquet",
            bucket_frames["production_opportunity_gates"],
        )
        _write_frame(
            output_dir / "lgbm_addon_candidate_context_helpers.parquet",
            bucket_frames["candidate_context_helpers"],
        )
        _write_frame(
            output_dir / "lgbm_addon_exploratory_context_interactions.parquet",
            bucket_frames["exploratory_context_interactions"],
        )
        _write_frame(
            output_dir / "lgbm_addon_diagnostic_only_regime_features.parquet",
            bucket_frames["diagnostic_only_regime_features"],
        )
        _write_json(output_dir / "lgbm_addon_feature_buckets.json", bucket_summary)
        _write_json(output_dir / "lgbm_addon_diagnostics.json", lgbm_diag)

    summary = _summarize_metrics(
        sample=sample,
        result=result,
        validation_report=validation_report,
        context_diag=context_diag,
        interaction_diag=interaction_diag,
        lgbm_diag=lgbm_diag,
    )
    summary["inputs"] = {
        "feature_run_id": feature_run_id,
        "label_artifact_run_id": label_run_id,
        "base_artifact_run_id": base_run_id,
        "requested_base_artifact_run_id": requested_base_run_id,
        "label_path": str(label_path),
        "base_artifact": str(base_artifact),
        "output_dir": str(output_dir),
        "sampling_mode": str(args.sampling_mode),
    }
    _write_json(output_dir / "metrics_summary.json", summary)
    _write_analysis(output_dir / "metrics_analysis.md", summary)
    print(json.dumps(_jsonable({"status": "completed", "output_dir": str(output_dir), **summary["sample"]}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
