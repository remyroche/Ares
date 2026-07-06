#!/usr/bin/env python3
"""Materialize broad cross-asset/meta-context features for train_meta.

This is the non-stability route for the cross-asset layer.  It uses
decision-time ledger columns and optional pre-entry feature-store columns,
learns month-forward latent context dimensions from prior months only, and
evaluates whether those context dimensions improve top-k executable selection
inside side x archetype cells.

No prior-month cell stability features are created here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_LEDGER = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "cumulative_ledger/cumulative_flat_candidates.parquet"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260617_090000")
DEFAULT_BASE_OOF_DIR = Path("data_perp/artifacts/20260629_050000_lgbm_mda/oof")
DEFAULT_OUT_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1"
)

CROSS_MARKET_PREFIXES: tuple[str, ...] = (
    "q_tail_",
    "q_iqr_",
    "q_width_",
    "width_",
    "tail_",
    "asym_",
    "iqr_",
    "pct_assets_",
    "cs_",
    "cs_rank_",
    "btc_",
    "eth_",
    "eth_btc_",
    "xs_dispersion_",
    "trend_dispersion_",
    "spectral_",
    "state_spectral_",
    "xasset_",
    "mkt_",
    "eig_",
    "market_breadth_",
    "market_dispersion_",
    "xasset_mkt_",
    "market_index_",
    "cross_asset_",
    "median_asset_",
    "top_decile_asset_",
    "cross_asset_correlation_",
    "avg_pairwise_corr_",
)

DIRECT_CONTEXT_PREFIXES: tuple[str, ...] = (
    "oof_",
    "oofctx_",
    "meta_lgbm_",
    "base_lgbm_",
    "feature_drift_",
    "row_drift_",
    "regime_centroid_",
    "rare_leaf_",
    "leaf_",
    "contrib_",
    "prob_uncertainty",
    "entropy",
    "mahalanobis_",
    "inference_drift_",
    "uncertainty_",
    "generated_",
    "simple_policy_calibrated_",
    "estimated_",
    "ev_adjusted_",
    "volatility_zscore",
    "auction_rank_",
    "contextual_",
)

AE_GMM_OOF_PREFIXES: tuple[str, ...] = (
    "oof_gmm_prob_",
    "oof_gmm_dist_center_",
    "oof_gmm_mahal_",
    "oof_dae_",
    "oof_cluster_",
    "oof_time_since_cluster_change",
    "oof_rolling_cluster_stability",
    "oof_raw_state_min_cluster_distance",
    "oof_regime_centroid_",
    "oof_min_mahalanobis",
    "oof_expected_mahalanobis",
    "oof_latent_mahalanobis_drift",
)

KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
TOP_FRACS = (0.30, 0.20, 0.10)
OUTCOME_COLUMNS = {
    "net_return",
    "gross_return",
    "simple_policy_exit_reason",
    "exec_net_return",
    "exec_ev_after_1pct_cost",
    "positive_ev_after_1pct",
    "full_sl",
    "timeout",
    "clean_exec_proxy",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _feature_file_for_symbol(feature_dir: Path, symbol: str) -> Path:
    return Path(feature_dir) / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _context_columns(columns: Iterable[str]) -> list[str]:
    blocked = {"symbol", "__symbol__", "side", "side_name", "target", "label"}
    out: list[str] = []
    for col in columns:
        name = str(col)
        lower = name.lower()
        if name in blocked or name.startswith("__"):
            continue
        if any(token in lower for token in ("future", "target", "label", "outcome", "forward")):
            continue
        if name.startswith(CROSS_MARKET_PREFIXES):
            out.append(name)
    return sorted(set(out))


def _read_feature_symbol(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    schema_cols = set(pq.read_schema(path).names)
    read_cols = [c for c in columns if c in schema_cols]
    for key in ("ts", "timestamp", "__symbol__"):
        if key in schema_cols and key not in read_cols:
            read_cols.append(key)
    frame = pd.read_parquet(path, columns=read_cols if read_cols else None)
    if "ts" in frame.columns:
        frame["__ts__"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    elif "timestamp" in frame.columns:
        frame["__ts__"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    else:
        idx_name = str(getattr(frame.index, "name", "")).lower()
        if idx_name in {"ts", "timestamp"}:
            frame = frame.reset_index()
            frame["__ts__"] = pd.to_datetime(frame.iloc[:, 0], utc=True, errors="coerce").dt.tz_convert(None)
        else:
            raise ValueError(f"Feature file has no timestamp column: {path}")
    if "__symbol__" not in frame.columns:
        frame["__symbol__"] = path.name.removeprefix("symbol=").removesuffix(".parquet").replace("_USD:USD", "/USD:USD")
    keep = ["__ts__", "__symbol__"] + [c for c in columns if c in frame.columns]
    return frame[keep].dropna(subset=["__ts__", "__symbol__"]).drop_duplicates(["__ts__", "__symbol__"], keep="last")


def _join_feature_store_asof(
    ledger: pd.DataFrame,
    feature_dir: Path | None,
    *,
    max_staleness_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_dir is None or not Path(feature_dir).exists():
        return ledger, {"status": "missing", "feature_dir": str(feature_dir) if feature_dir else None}
    if "__ts__" not in ledger.columns or "__symbol__" not in ledger.columns:
        return ledger, {"status": "missing_keys", "required_keys": list(KEY_COLUMNS[:2])}

    import pyarrow.parquet as pq

    feature_dir = Path(feature_dir)
    symbols = sorted(ledger["__symbol__"].dropna().astype(str).unique().tolist())
    available: dict[str, Path] = {}
    schema_columns: set[str] = set()
    missing = 0
    for symbol in symbols:
        path = _feature_file_for_symbol(feature_dir, symbol)
        if not path.exists():
            missing += 1
            continue
        available[symbol] = path
        if not schema_columns:
            schema_columns = set(pq.read_schema(path).names)
    context_cols = _context_columns(schema_columns)
    if not available or not context_cols:
        return ledger, {
            "status": "no_joinable_context",
            "feature_dir": str(feature_dir),
            "available_symbol_files": len(available),
            "missing_symbol_files": missing,
            "context_column_count": len(context_cols),
        }

    left = ledger.copy()
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="coerce").dt.tz_convert(None)
    left["__row_id__"] = np.arange(len(left), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    matched_row_ids: list[np.ndarray] = []
    loaded_symbols = 0
    for symbol, path in available.items():
        rows = left[left["__symbol__"].astype(str).eq(symbol)].copy()
        if rows.empty:
            continue
        features = _read_feature_symbol(path, context_cols)
        if features.empty:
            continue
        min_ts = rows["__ts__"].min() - pd.Timedelta(minutes=max_staleness_minutes)
        max_ts = rows["__ts__"].max()
        features = features[features["__ts__"].between(min_ts, max_ts)].copy()
        if features.empty:
            continue
        rows = rows.sort_values("__ts__")
        features = features.sort_values("__ts__")
        joined = pd.merge_asof(
            rows,
            features,
            on="__ts__",
            by="__symbol__",
            direction="backward",
            tolerance=pd.Timedelta(minutes=max_staleness_minutes),
            suffixes=("", "__feature_store"),
        )
        parts.append(joined)
        matched_row_ids.append(joined["__row_id__"].to_numpy())
        loaded_symbols += 1
    if not parts:
        return left.drop(columns=["__row_id__"]), {
            "status": "no_matching_rows",
            "feature_dir": str(feature_dir),
            "available_symbol_files": len(available),
            "missing_symbol_files": missing,
            "context_column_count": len(context_cols),
        }
    matched_ids = np.concatenate(matched_row_ids) if matched_row_ids else np.array([], dtype=np.int64)
    unmatched = left[~left["__row_id__"].isin(matched_ids)]
    if not unmatched.empty:
        parts.append(unmatched)
    out = pd.concat(parts, ignore_index=True)
    rename: dict[str, str] = {}
    for col in context_cols:
        if col in out.columns:
            rename[col] = f"ctx_{col}"
        suffixed = f"{col}__feature_store"
        if suffixed in out.columns:
            rename[suffixed] = f"ctx_{col}"
    out = out.rename(columns=rename)
    out = out.sort_values("__row_id__").drop(columns=["__row_id__"]).reset_index(drop=True)
    loaded_cols = sorted(rename.values())
    matched = out[loaded_cols].notna().any(axis=1) if loaded_cols else pd.Series(False, index=out.index)
    return out, {
        "status": "joined_asof",
        "feature_dir": str(feature_dir),
        "available_symbol_files": len(available),
        "loaded_symbol_files": loaded_symbols,
        "missing_symbol_files": missing,
        "loaded_column_count": len(loaded_cols),
        "loaded_columns": loaded_cols,
        "matched_rows": int(matched.sum()),
        "row_count": int(len(out)),
        "match_rate": float(matched.mean()) if len(out) else float("nan"),
        "max_staleness_minutes": int(max_staleness_minutes),
        "leakage_contract": "asof backward join by symbol; feature timestamp must be <= decision timestamp within tolerance",
    }


def _parse_oof_strategy(path: Path, horizon: str) -> str | None:
    stem = path.stem
    prefix = "oof_"
    suffix = f"_{horizon}"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None
    return stem[len(prefix) : -len(suffix)]


def _oof_context_columns(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        name = str(col)
        lower = name.lower()
        if any(token in lower for token in ("future", "target", "label", "outcome", "forward")):
            continue
        if name.startswith(AE_GMM_OOF_PREFIXES):
            out.append(name)
    return sorted(set(out))


def _join_base_oof_context(
    ledger: pd.DataFrame,
    base_oof_dir: Path | None,
    *,
    horizon: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if base_oof_dir is None or not Path(base_oof_dir).exists():
        return ledger, {"status": "missing", "base_oof_dir": str(base_oof_dir) if base_oof_dir else None}
    if not {"__ts__", "__symbol__", "strategy_id"}.issubset(ledger.columns):
        return ledger, {
            "status": "missing_keys",
            "required_keys": ["__ts__", "__symbol__", "strategy_id"],
            "base_oof_dir": str(base_oof_dir),
        }

    import pyarrow.parquet as pq

    base_oof_dir = Path(base_oof_dir)
    wanted_strategies = set(ledger["strategy_id"].dropna().astype(str).unique().tolist())
    parts: list[pd.DataFrame] = []
    scanned_files = 0
    used_files = 0
    loaded_columns: set[str] = set()
    for path in sorted(base_oof_dir.glob(f"oof_*_{horizon}.parquet")):
        scanned_files += 1
        strategy = _parse_oof_strategy(path, horizon)
        if strategy is None or strategy not in wanted_strategies:
            continue
        schema_cols = set(pq.read_schema(path).names)
        context_cols = _oof_context_columns(schema_cols)
        key_cols = [c for c in ("timestamp", "__ts__", "symbol", "__symbol__") if c in schema_cols]
        if not context_cols or not {"timestamp", "symbol"}.issubset(schema_cols):
            continue
        read_cols = ["timestamp", "symbol"] + context_cols
        part = pd.read_parquet(path, columns=read_cols)
        part["__ts__"] = pd.to_datetime(part["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        part["__symbol__"] = part["symbol"].astype(str)
        part["strategy_id"] = strategy
        rename = {col: f"oofctx_{col.removeprefix('oof_')}" for col in context_cols}
        part = part.rename(columns=rename)
        keep = ["__ts__", "__symbol__", "strategy_id"] + list(rename.values())
        parts.append(part[keep].drop_duplicates(["__ts__", "__symbol__", "strategy_id"], keep="last"))
        loaded_columns.update(rename.values())
        used_files += 1
    if not parts:
        return ledger, {
            "status": "no_joinable_oof_files",
            "base_oof_dir": str(base_oof_dir),
            "horizon": horizon,
            "scanned_files": scanned_files,
            "wanted_strategies": len(wanted_strategies),
        }
    context = pd.concat(parts, ignore_index=True)
    left = ledger.copy()
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="coerce").dt.tz_convert(None)
    before = len(left)
    out = left.merge(context, on=["__ts__", "__symbol__", "strategy_id"], how="left", validate="many_to_one")
    loaded = sorted(loaded_columns)
    matched = out[loaded].notna().any(axis=1) if loaded else pd.Series(False, index=out.index)
    return out, {
        "status": "joined",
        "base_oof_dir": str(base_oof_dir),
        "horizon": horizon,
        "scanned_files": scanned_files,
        "used_files": used_files,
        "loaded_column_count": len(loaded),
        "loaded_columns": loaded,
        "matched_rows": int(matched.sum()),
        "row_count": int(len(out)),
        "rows_before": int(before),
        "match_rate": float(matched.mean()) if len(out) else float("nan"),
        "leakage_contract": "joined by timestamp, symbol, and strategy_id from base OOF files; columns are OOF AE/GMM/regime context outputs",
    }


def _normalize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    if "__ts__" not in out.columns:
        if "timestamp" not in out.columns:
            raise ValueError("ledger must contain timestamp or __ts__")
        out["__ts__"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    else:
        out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce").dt.tz_convert(None)
    if "__symbol__" not in out.columns:
        symbol_col = "symbol" if "symbol" in out.columns else None
        if symbol_col is None:
            raise ValueError("ledger must contain symbol or __symbol__")
        out["__symbol__"] = out[symbol_col].astype(str)
    if "side_name" not in out.columns:
        if "side" not in out.columns:
            raise ValueError("ledger must contain side or side_name")
        out["side_name"] = out["side"].astype(str)
    out["month"] = out["__ts__"].dt.to_period("M").astype(str)
    if "head" in out.columns:
        source = out["head"].fillna("").astype(str)
    else:
        source = pd.Series("", index=out.index, dtype="object")
    fallback = out.get("strategy_id", pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)
    out["source_archetype"] = np.where(source.str.len() > 0, source, fallback.str.split("_").str[:2].str.join("_"))
    out["exec_net_return"] = pd.to_numeric(out.get("net_return"), errors="coerce").astype("float32")
    out["exec_ev_after_1pct_cost"] = (out["exec_net_return"] - 0.01).astype("float32")
    exit_reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    out["full_sl"] = exit_reason.eq("full_sl").astype("int8")
    out["timeout"] = exit_reason.eq("timeout").astype("int8")
    out["positive_ev_after_1pct"] = (out["exec_ev_after_1pct_cost"] > 0).astype("int8")
    out["clean_exec_proxy"] = (
        out["positive_ev_after_1pct"].eq(1) & out["full_sl"].eq(0) & out["timeout"].eq(0)
    ).astype("int8")
    return out


def _is_numeric_context(col: str, frame: pd.DataFrame) -> bool:
    if col in KEY_COLUMNS or col in OUTCOME_COLUMNS:
        return False
    if col in {"timestamp", "symbol", "side", "month", "strategy_id", "head", "source_archetype"}:
        return False
    if col.endswith("_json") or col.endswith("_hash") or col.endswith("_source"):
        return False
    if not pd.api.types.is_numeric_dtype(frame[col]):
        return False
    if col.startswith("ctx_"):
        return True
    if col.startswith(DIRECT_CONTEXT_PREFIXES):
        return True
    return False


def _select_context_features(frame: pd.DataFrame, *, max_features: int) -> list[str]:
    candidates = [c for c in frame.columns if _is_numeric_context(c, frame)]
    scored: list[tuple[float, str]] = []
    for col in candidates:
        s = pd.to_numeric(frame[col], errors="coerce")
        coverage = float(s.notna().mean())
        if coverage < 0.20:
            continue
        std = float(s.std(skipna=True))
        if not np.isfinite(std) or std <= 1e-12:
            continue
        scored.append((coverage * min(std, 10.0), col))
    scored.sort(reverse=True)
    return [col for _, col in scored[:max_features]]


def _add_month_forward_latents(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_components: int,
    n_clusters: int,
    max_fit_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        return frame, {"status": f"skipped_dependency:{type(exc).__name__}"}

    out = frame.copy()
    months = sorted(out["month"].dropna().astype(str).unique().tolist())
    latent_cols = [f"xctx_latent_{i}" for i in range(n_components)]
    for col in latent_cols + ["xctx_cluster_id", "xctx_cluster_distance", "xctx_cluster_entropy"]:
        out[col] = np.nan

    fit_events: list[dict[str, Any]] = []
    rng = np.random.default_rng(17)
    for month in months[1:]:
        train_idx = out.index[out["month"].astype(str) < month]
        val_idx = out.index[out["month"].astype(str).eq(month)]
        if len(train_idx) < max(500, len(feature_cols) * 5) or len(val_idx) == 0:
            continue
        if len(train_idx) > max_fit_rows:
            train_idx = pd.Index(rng.choice(train_idx.to_numpy(), size=max_fit_rows, replace=False))
        x_train = out.loc[train_idx, feature_cols]
        x_val = out.loc[val_idx, feature_cols]
        pca_n = min(n_components, len(feature_cols), max(1, len(train_idx) - 1))
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=pca_n, random_state=17),
        )
        z_train = pipe.fit_transform(x_train)
        z_val = pipe.transform(x_val)
        for i in range(pca_n):
            out.loc[val_idx, f"xctx_latent_{i}"] = z_val[:, i].astype("float32")
        k = min(n_clusters, max(2, int(math.sqrt(max(len(train_idx), 2)))))
        km = MiniBatchKMeans(n_clusters=k, random_state=17, batch_size=2048, n_init=5)
        km.fit(z_train[:, :pca_n])
        distances = km.transform(z_val[:, :pca_n])
        labels = np.argmin(distances, axis=1)
        min_dist = distances[np.arange(len(labels)), labels]
        # Entropy over inverse-distance soft assignments.  This is uncertainty,
        # not a stability prior.
        logits = -distances
        logits = logits - np.nanmax(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / math.log(k)
        out.loc[val_idx, "xctx_cluster_id"] = labels.astype("float32")
        out.loc[val_idx, "xctx_cluster_distance"] = min_dist.astype("float32")
        out.loc[val_idx, "xctx_cluster_entropy"] = entropy.astype("float32")
        fit_events.append(
            {
                "month": month,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "feature_count": int(len(feature_cols)),
                "pca_components": int(pca_n),
                "kmeans_clusters": int(k),
            }
        )
    contract = {
        "status": "month_forward_latents_materialized",
        "fit_events": fit_events,
        "feature_columns": feature_cols,
        "latent_columns": latent_cols + ["xctx_cluster_id", "xctx_cluster_distance", "xctx_cluster_entropy"],
        "leakage_contract": "each validation month is transformed by PCA/KMeans fit only on strictly earlier months",
    }
    return out, contract


def _fit_month_forward_ev_score(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_fit_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as exc:  # pragma: no cover
        return frame, {"status": f"skipped_dependency:{type(exc).__name__}"}

    out = frame.copy()
    out["xctx_ev_score_oof"] = np.nan
    months = sorted(out["month"].dropna().astype(str).unique().tolist())
    rng = np.random.default_rng(23)
    events: list[dict[str, Any]] = []
    for month in months[1:]:
        train_idx = out.index[out["month"].astype(str) < month]
        val_idx = out.index[out["month"].astype(str).eq(month)]
        y = pd.to_numeric(out.loc[train_idx, "exec_ev_after_1pct_cost"], errors="coerce")
        valid = y.notna()
        train_idx = train_idx[valid.to_numpy()]
        if len(train_idx) < 1000 or len(val_idx) == 0:
            continue
        if len(train_idx) > max_fit_rows:
            train_idx = pd.Index(rng.choice(train_idx.to_numpy(), size=max_fit_rows, replace=False))
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=96,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                min_samples_leaf=120,
                random_state=23,
            ),
        )
        model.fit(out.loc[train_idx, feature_cols], out.loc[train_idx, "exec_ev_after_1pct_cost"].astype(float))
        preds = model.predict(out.loc[val_idx, feature_cols])
        out.loc[val_idx, "xctx_ev_score_oof"] = preds.astype("float32")
        events.append({"month": month, "train_rows": int(len(train_idx)), "validation_rows": int(len(val_idx))})
    if "normalized_rank_score" in out.columns:
        base = pd.to_numeric(out["normalized_rank_score"], errors="coerce")
    elif "calibrated_score" in out.columns:
        base = pd.to_numeric(out["calibrated_score"], errors="coerce")
    else:
        base = pd.Series(np.nan, index=out.index)
    out["xctx_baseline_score"] = base.astype("float32")
    out["xctx_blend_score"] = np.nan
    for _, idx in out.groupby(["month", "side_name"], dropna=False).groups.items():
        idx = pd.Index(idx)
        base_rank = out.loc[idx, "xctx_baseline_score"].rank(pct=True)
        ctx_rank = out.loc[idx, "xctx_ev_score_oof"].rank(pct=True)
        out.loc[idx, "xctx_blend_score"] = (0.5 * base_rank + 0.5 * ctx_rank).astype("float32")
    return out, {
        "status": "month_forward_ev_score_materialized",
        "fit_events": events,
        "feature_columns": feature_cols,
        "target": "exec_ev_after_1pct_cost",
        "leakage_contract": "each validation month is scored by model fit only on strictly earlier months",
    }


def _topk_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    group_cols: list[str],
    top_fracs: tuple[float, ...],
    min_group_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    for keys, grp in valid.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) < min_group_rows:
            continue
        ordered = grp.sort_values(score_col, ascending=False)
        for frac in top_fracs:
            n = max(1, int(math.ceil(len(ordered) * frac)))
            sel = ordered.head(n)
            ev = pd.to_numeric(sel["exec_ev_after_1pct_cost"], errors="coerce")
            abs_ev = ev.abs().sum()
            rec = {col: key for col, key in zip(group_cols, keys)}
            rec.update(
                {
                    "score_col": score_col,
                    "top_frac": float(frac),
                    "rows": int(len(grp)),
                    "selected_rows": int(len(sel)),
                    "precision_positive_ev": float((ev > 0).mean()),
                    "ev_weighted_precision": float(ev.clip(lower=0).sum() / abs_ev) if abs_ev > 0 else float("nan"),
                    "mean_ev_after_1pct": float(ev.mean()),
                    "sum_ev_after_1pct": float(ev.sum()),
                    "full_sl_rate": float(pd.to_numeric(sel["full_sl"], errors="coerce").mean()),
                    "timeout_rate": float(pd.to_numeric(sel["timeout"], errors="coerce").mean()),
                    "clean_exec_proxy_rate": float(pd.to_numeric(sel["clean_exec_proxy"], errors="coerce").mean()),
                }
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def _delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    key_cols = [c for c in ["month", "side_name", "source_archetype", "top_frac"] if c in metrics.columns]
    base = metrics[metrics["score_col"].eq("xctx_baseline_score")]
    rows: list[dict[str, Any]] = []
    for _, cur in metrics[~metrics["score_col"].eq("xctx_baseline_score")].iterrows():
        mask = pd.Series(True, index=base.index)
        for col in key_cols:
            mask &= base[col].eq(cur[col])
        if not mask.any():
            continue
        ref = base[mask].iloc[0]
        rec = {col: cur[col] for col in key_cols}
        rec["score_col"] = cur["score_col"]
        rec["rows"] = int(cur["rows"])
        rec["selected_rows"] = int(cur["selected_rows"])
        for metric in (
            "precision_positive_ev",
            "ev_weighted_precision",
            "mean_ev_after_1pct",
            "full_sl_rate",
            "timeout_rate",
            "clean_exec_proxy_rate",
        ):
            rec[f"delta_{metric}"] = float(cur[metric] - ref[metric])
            rec[f"{metric}"] = float(cur[metric])
            rec[f"baseline_{metric}"] = float(ref[metric])
        rows.append(rec)
    return pd.DataFrame(rows)


def _write_report(path: Path, manifest: dict[str, Any], summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lines = [
        "# Direct Cross-Asset Meta Context",
        "",
        "## Status",
        "",
        f"- Source rows: `{manifest['rows']}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Context feature count: `{manifest['context_feature_count']}`",
        f"- Latent feature count: `{manifest['latent_feature_count']}`",
        f"- Feature-store join: `{manifest['feature_store_contract']['status']}`",
        "",
        "This artifact intentionally excludes prior-month cell stability features.",
        "",
        "## Aggregate Top-k",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No aggregate metrics.",
        "",
        "## Best Side x Archetype Deltas",
        "",
        deltas.sort_values("delta_mean_ev_after_1pct", ascending=False)
        .head(25)
        .to_markdown(index=False)
        if not deltas.empty
        else "No delta rows.",
        "",
        "## Metric Notes",
        "",
        "- `exec_ev_after_1pct_cost` is `net_return - 0.01` for a conservative 1% round-trip cost objective.",
        "- `full_sl_rate` is the available bad-path proxy in this broad ledger; first-touch bad-MAE is not present.",
        "- Month-forward latent/scoring models are fit on strictly earlier months only.",
        "- Raw direct context features are live-predictable decision-time columns and optional as-of feature-store columns.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    ledger_path: Path,
    output_dir: Path,
    feature_dir: Path | None,
    base_oof_dir: Path | None,
    base_oof_horizon: str,
    max_asof_staleness_minutes: int,
    max_context_features: int,
    n_components: int,
    n_clusters: int,
    max_fit_rows: int,
    min_group_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(ledger_path)
    ledger = _normalize_ledger(ledger)
    ledger, feature_contract = _join_feature_store_asof(
        ledger,
        feature_dir,
        max_staleness_minutes=max_asof_staleness_minutes,
    )
    ledger, oof_contract = _join_base_oof_context(ledger, base_oof_dir, horizon=base_oof_horizon)
    context_features = _select_context_features(ledger, max_features=max_context_features)
    handoff, latent_contract = _add_month_forward_latents(
        ledger,
        context_features,
        n_components=n_components,
        n_clusters=n_clusters,
        max_fit_rows=max_fit_rows,
    )
    latent_cols = [c for c in handoff.columns if c.startswith("xctx_latent_") or c in {
        "xctx_cluster_id",
        "xctx_cluster_distance",
        "xctx_cluster_entropy",
    }]
    score_features = context_features + latent_cols
    score_features = [c for c in score_features if c in handoff.columns]
    handoff, score_contract = _fit_month_forward_ev_score(handoff, score_features, max_fit_rows=max_fit_rows)

    score_cols = [c for c in ("xctx_baseline_score", "xctx_ev_score_oof", "xctx_blend_score") if c in handoff.columns]
    metrics = pd.concat(
        [
            _topk_metrics(
                handoff,
                score_col=score_col,
                group_cols=["month", "side_name", "source_archetype"],
                top_fracs=TOP_FRACS,
                min_group_rows=min_group_rows,
            )
            for score_col in score_cols
        ],
        ignore_index=True,
    )
    aggregate = pd.concat(
        [
            _topk_metrics(
                handoff,
                score_col=score_col,
                group_cols=["month"],
                top_fracs=TOP_FRACS,
                min_group_rows=min_group_rows,
            )
            for score_col in score_cols
        ],
        ignore_index=True,
    )
    deltas = _delta_table(metrics)
    useful = deltas[
        (deltas["top_frac"].eq(0.10))
        & (deltas["delta_mean_ev_after_1pct"] > 0)
        & (deltas["delta_precision_positive_ev"] >= 0)
    ].copy() if not deltas.empty else pd.DataFrame()

    outputs = {
        "handoff": output_dir / "direct_cross_asset_meta_context_handoff.parquet",
        "topk_metrics": output_dir / "direct_cross_asset_topk_metrics_by_cell.csv",
        "aggregate_metrics": output_dir / "direct_cross_asset_topk_metrics_by_month.csv",
        "deltas": output_dir / "direct_cross_asset_side_archetype_deltas.csv",
        "useful_cells": output_dir / "direct_cross_asset_useful_side_archetype_cells.csv",
        "feature_columns": output_dir / "direct_cross_asset_context_feature_columns.json",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "direct_cross_asset_meta_context_report.md",
    }
    handoff.to_parquet(outputs["handoff"], index=False)
    metrics.to_csv(outputs["topk_metrics"], index=False)
    aggregate.to_csv(outputs["aggregate_metrics"], index=False)
    deltas.to_csv(outputs["deltas"], index=False)
    useful.to_csv(outputs["useful_cells"], index=False)
    outputs["feature_columns"].write_text(
        json.dumps(
            {
                "context_features": context_features,
                "latent_features": latent_cols,
                "score_features": score_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "scope": "direct_cross_asset_meta_context",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "rows": int(len(handoff)),
        "months": sorted(handoff["month"].dropna().astype(str).unique().tolist()),
        "context_feature_count": int(len(context_features)),
        "latent_feature_count": int(len(latent_cols)),
        "score_feature_count": int(len(score_features)),
        "feature_store_contract": feature_contract,
        "base_oof_context_contract": oof_contract,
        "latent_contract": latent_contract,
        "score_contract": score_contract,
        "outcome_contract": {
            "exec_net_return": "ledger net_return",
            "exec_ev_after_1pct_cost": "net_return minus 1% round-trip cost objective",
            "full_sl": "simple_policy_exit_reason == full_sl; bad-path proxy, not first-touch bad-MAE",
            "timeout": "simple_policy_exit_reason == timeout",
        },
        "stability_features": "excluded_by_user_request",
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    summary = aggregate.groupby(["score_col", "top_frac"], as_index=False).agg(
        months=("month", "nunique"),
        mean_precision_positive_ev=("precision_positive_ev", "mean"),
        mean_ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        mean_full_sl_rate=("full_sl_rate", "mean"),
        mean_timeout_rate=("timeout_rate", "mean"),
        mean_clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )
    _write_report(outputs["report"], manifest, summary, deltas)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--base-oof-dir", type=Path, default=DEFAULT_BASE_OOF_DIR)
    parser.add_argument("--base-oof-horizon", default="H5")
    parser.add_argument("--no-feature-store", action="store_true")
    parser.add_argument("--no-base-oof", action="store_true")
    parser.add_argument("--max-asof-staleness-minutes", type=int, default=90)
    parser.add_argument("--max-context-features", type=int, default=160)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--max-fit-rows", type=int, default=120_000)
    parser.add_argument("--min-group-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        feature_dir=None if args.no_feature_store else args.feature_dir,
        base_oof_dir=None if args.no_base_oof else args.base_oof_dir,
        base_oof_horizon=str(args.base_oof_horizon),
        max_asof_staleness_minutes=int(args.max_asof_staleness_minutes),
        max_context_features=int(args.max_context_features),
        n_components=int(args.n_components),
        n_clusters=int(args.n_clusters),
        max_fit_rows=int(args.max_fit_rows),
        min_group_rows=int(args.min_group_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
