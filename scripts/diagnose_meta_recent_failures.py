#!/usr/bin/env python3
"""Diagnose recent high-confidence meta-model failures.

The script is designed for the final-fit/meta-feature-selection artifacts in
``data_perp``.  It writes separate artifacts for:

1. high-confidence failure classifiers,
2. normal-vs-bad-week adversarial validation,
3. meta/base LGBM leaf instability.

The diagnostics are deliberately post-hoc and explanatory.  They do not train
or modify production models.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

try:
    from extreme_price_movements.data_store import _feature_schema_names, read_symbol_features
except Exception:  # pragma: no cover - diagnostics can still run on plain parquet stores.
    _feature_schema_names = None
    read_symbol_features = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - the repo normally depends on LightGBM.
    lgb = None


OUTCOME_COLUMNS = {
    "y_move",
    "y_move_soft",
    "y_bin",
    "target",
    "return",
    "barrier_pct",
    "mae_ret",
    "mfe_ret",
    "mae",
    "mfe",
    "bars_to_mfe",
    "is_timeout",
    "exit_code",
    "label_code",
}

FORBIDDEN_FEATURE_TOKENS = (
    "leaf_target",
    "barrier_pct",
    "rank_bin_",
    "rank_bin_win_rate",
    "rank_bin_lift",
    "rank_bin_net_ret",
    "rank_bin_se",
    "net_ret_oof",
    "policy_result",
    "post_trade",
)

KEY_COLUMNS = {"timestamp", "symbol", "index", "row_index", "strategy_id"}

NON_DEPLOYABLE_EXPORT_PREFIXES = ("diag_",)

DEPLOYABLE_EXACT_EXPORTS = {
    "oof_pred",
    "oof_p_move",
    "oof_meta_clf",
    "oof_base_clf",
    "oof_rank_pct",
    "oof_rank_margin_top10",
    "oof_rank_margin_top20",
    "oof_rank_margin_top30",
    "clf_center",
    "base_clf_centered",
}

DEPLOYABLE_EXPORT_PREFIXES = (
    "oof_score_early_",
    "oof_rank_100_minus_50",
    "oof_rank_path_std",
    "oof_leaf_proximity_",
    "oof_leaf_count_",
    "oof_leaf_depth_",
    "oof_leaf_pred_",
    "oof_dae_",
    "oof_gmm_",
    "oof_cluster_entropy",
    "oof_regime_centroid",
    "meta_en_",
)

MARKET_OBSERVABLE_TOKENS = (
    "ema",
    "sma",
    "vwap",
    "adx",
    "rsi",
    "boll",
    "bb_",
    "loc_",
    "dist_",
    "zscore",
    "draw",
    "wick",
    "retest",
    "ker_",
    "fund",
    "carry",
    "oi_",
    "_oi",
    "open_interest",
    "leverage",
    "liquid",
    "spread",
    "depth",
    "amihud",
    "volume",
    "rvol",
    "vol",
    "rv",
    "atr",
    "range",
    "trend",
    "slope",
    "compression",
    "efficiency",
    "chop",
    "entropy",
    "mkt_ret",
    "btc_ret",
    "eth_ret",
    "breadth",
    "dispersion",
    "cs_rank",
    "symbol_minus_mkt",
    "asset_minus_mkt",
    "ret",
    "return_autocorr",
    "tail",
    "coherence",
)

CONTEXT_PATTERNS = (
    "fund",
    "oi_",
    "_oi",
    "liquidity",
    "spread",
    "vol",
    "rv",
    "atr",
    "amihud",
    "depth",
    "breadth",
    "dispersion",
)


@dataclass(frozen=True)
class HeadContext:
    head: str
    strategy_id: str
    meta_key: str
    meta_oof_path: Path


def _log(message: str) -> None:
    print(f"[diagnose_meta_failures] {message}", flush=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


def _include_feature_delta_store() -> bool:
    return str(os.getenv("EPM_RECENT_FAILURE_INCLUDE_FEATURE_DELTAS", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _feature_store_columns(path: Path) -> set[str]:
    if _include_feature_delta_store() and _feature_schema_names is not None:
        try:
            return set(_feature_schema_names(str(path)))
        except Exception:
            pass
    return set(_parquet_columns(path))


def _symbol_aliases(symbol: str) -> list[str]:
    raw = str(symbol)
    aliases = [raw, raw.replace("/", "_")]
    if "/" not in raw and "_" in raw:
        base, rest = raw.split("_", 1)
        aliases.append(f"{base}/{rest}")
    return list(dict.fromkeys(aliases))


def _feature_path_for_symbol(feature_dir: Path, symbol: str) -> Path | None:
    for alias in _symbol_aliases(symbol):
        path = feature_dir / f"symbol={alias}.parquet"
        if path.exists():
            return path
    return None


def _downcast_numeric(df: pd.DataFrame, *, exclude: Iterable[str] = ()) -> pd.DataFrame:
    excluded = set(exclude)
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32, copy=False)
        elif pd.api.types.is_integer_dtype(df[col]):
            cmin = df[col].min(skipna=True)
            cmax = df[col].max(skipna=True)
            if pd.notna(cmin) and pd.notna(cmax) and cmin >= np.iinfo(np.int32).min and cmax <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32, copy=False)
    return df


def _normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    return out


def _feature_name_variants(name: str) -> list[str]:
    lowered = str(name).lower()
    variants = [lowered]
    changed = True
    while changed:
        changed = False
        for prefix in ("export__", "url__", "oof_", "pred_h5_", "base_h5_", "diag_"):
            for value in list(variants):
                if value.startswith(prefix):
                    stripped = value[len(prefix) :]
                    if stripped and stripped not in variants:
                        variants.append(stripped)
                        changed = True
    return variants


def _is_forbidden_feature_name(name: str) -> bool:
    raw = str(name)
    lowered = raw.lower()
    variants = _feature_name_variants(name)
    if any(value.startswith(NON_DEPLOYABLE_EXPORT_PREFIXES) for value in variants):
        return True
    if raw in OUTCOME_COLUMNS or any(variant in OUTCOME_COLUMNS for variant in variants):
        return True
    return any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS)


def _is_deployable_export_feature(name: str) -> bool:
    lowered = str(name).lower()
    variants = _feature_name_variants(name)
    if _is_forbidden_feature_name(name):
        return False
    if lowered.startswith("url__") and any(
        token in lowered for token in ("regime", "gmm", "dae", "cluster", "centroid", "archetype", "latent")
    ):
        return True
    if any(value in DEPLOYABLE_EXACT_EXPORTS for value in variants):
        return True
    if any(any(value.startswith(prefix) for prefix in DEPLOYABLE_EXPORT_PREFIXES) for value in variants):
        return True
    if lowered.startswith(("export__oof_", "oof_")):
        return False
    if any(token in lowered for token in MARKET_OBSERVABLE_TOKENS):
        return True
    return False


def _feature_contract_row(feature: str, *, present: bool = True) -> dict[str, Any]:
    lowered = str(feature).lower()
    variants = _feature_name_variants(feature)
    if lowered.startswith("url__"):
        source_family = "latent_regime_context"
    elif lowered.startswith("export__"):
        source_family = "meta_oof_export"
    elif lowered.startswith(("oof_", "meta_en_")):
        source_family = "model_oof_path"
    else:
        source_family = "selected_or_market_feature"
    forbidden = _is_forbidden_feature_name(feature)
    if source_family == "selected_or_market_feature":
        deployable = not forbidden
    else:
        deployable = _is_deployable_export_feature(feature)
    is_prediction_path = any(
        value in DEPLOYABLE_EXACT_EXPORTS or any(value.startswith(prefix) for prefix in DEPLOYABLE_EXPORT_PREFIXES)
        for value in variants
    )
    is_latent = lowered.startswith("url__") or any(
        token in lowered for token in ("regime", "gmm", "dae", "cluster", "centroid", "archetype")
    )
    is_market = any(token in lowered for token in MARKET_OBSERVABLE_TOKENS)
    return {
        "feature": feature,
        "source_family": source_family,
        "present_in_matrix": bool(present),
        "allowed_by_clean_contract": bool(deployable and not forbidden),
        "available_before_trade": bool(deployable and not forbidden),
        "outcome_independent": bool(not forbidden),
        "fold_fitted": bool(is_prediction_path or source_family == "model_oof_path"),
        "live_equivalent": bool(deployable and not forbidden),
        "train_live_parity_validated": False,
        "causal_availability": (
            "deployable_prediction_or_model_path"
            if is_prediction_path
            else "causal_latent_state_transform"
            if is_latent and deployable
            else "market_feature_available_at_decision_time"
            if is_market and deployable
            else "forbidden_or_unclassified"
        ),
        "contract_reason": (
            "forbidden_target_or_non_deployable_diagnostic"
            if forbidden
            else "allowed"
            if deployable
            else "unclassified_not_allowed"
        ),
    }


def _candidate_feature_contract(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_feature_contract_row(str(col), present=True) for col in x.columns])


def _strategy_from_meta_key(meta_key: str) -> str:
    suffix = "_tbm_clf"
    return meta_key[: -len(suffix)] if meta_key.endswith(suffix) else meta_key


def _head_from_strategy(strategy_id: str) -> str:
    if strategy_id.startswith("long_dist"):
        return "long_dist"
    if strategy_id.startswith("long_bars"):
        return "long_bars"
    if strategy_id.startswith("short_boll"):
        return "short_boll"
    if strategy_id.startswith("short_asset"):
        return "short_asset"
    if strategy_id.startswith("long_"):
        return "long"
    if strategy_id.startswith("short_"):
        return "short"
    return strategy_id[:32]


def _discover_heads(meta_artifact_dir: Path, report_dir: Path, meta_models: dict[str, Any]) -> list[HeadContext]:
    selected_path = report_dir / "selected_feature_importance_all.csv"
    report_map: dict[str, str] = {}
    if selected_path.exists():
        selected = pd.read_csv(selected_path, usecols=["head", "strategy_id"])
        report_map = dict(zip(selected["strategy_id"].astype(str), selected["head"].astype(str)))

    out: list[HeadContext] = []
    meta_oof_dir = meta_artifact_dir / "meta_oof"
    for meta_key in sorted(meta_models):
        strategy_id = _strategy_from_meta_key(meta_key)
        path = meta_oof_dir / f"meta_oof_{meta_key}.parquet"
        if not path.exists():
            matches = sorted(meta_oof_dir.glob(f"meta_oof_{strategy_id}*.parquet"))
            if not matches:
                raise FileNotFoundError(f"Missing meta OOF parquet for {meta_key} in {meta_oof_dir}")
            path = matches[0]
        out.append(
            HeadContext(
                head=report_map.get(strategy_id, _head_from_strategy(strategy_id)),
                strategy_id=strategy_id,
                meta_key=meta_key,
                meta_oof_path=path,
            )
        )
    return out


def _feature_store_union(feature_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not feature_dir.exists():
        return out
    for path in sorted(feature_dir.glob("symbol=*.parquet")):
        symbol = path.name.removeprefix("symbol=").removesuffix(".parquet")
        try:
            cols = _feature_store_columns(path)
        except Exception:
            continue
        for alias in _symbol_aliases(symbol):
            out[alias] = cols
    return out


def _hydrate_feature_store(
    feature_dir: Path,
    keys: pd.DataFrame,
    columns: Iterable[str],
    symbol_columns: dict[str, set[str]],
) -> pd.DataFrame:
    requested = list(dict.fromkeys(str(c) for c in columns if str(c)))
    base = keys[["timestamp", "symbol"]].copy()
    base["__row_id"] = np.arange(len(base), dtype=np.int64)
    if not requested or not symbol_columns:
        return base

    def _read_symbol_group(item: tuple[str, pd.DataFrame]):
        symbol, group = item
        available = symbol_columns.get(str(symbol), set())
        present = [c for c in requested if c in available]
        if not present:
            return None
        path = _feature_path_for_symbol(feature_dir, str(symbol))
        if path is None:
            return None
        try:
            if _include_feature_delta_store() and read_symbol_features is not None:
                start_ts = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").min()
                end_ts = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").max()
                values = read_symbol_features(
                    str(path),
                    columns=present,
                    start_ts=start_ts if pd.notna(start_ts) else None,
                    end_ts=end_ts if pd.notna(end_ts) else None,
                )
            else:
                # Read the immutable training feature snapshot directly. This
                # is faster for post-hoc diagnostics, while the opt-in project
                # reader above is needed for prospective backfills that are
                # stored as append-only DuckDB deltas.
                schema_cols = set(_parquet_columns(path))
                read_cols = ["ts", *present] if "ts" in schema_cols else present
                values = pq.read_table(path, columns=read_cols).to_pandas()
        except Exception:
            return None
        if values.empty:
            return None
        if not values.index.is_unique:
            values = values[~values.index.duplicated(keep="last")]
        if "ts" in values.columns:
            idx = pd.to_datetime(values.pop("ts"), utc=True, errors="coerce")
        else:
            idx = pd.to_datetime(values.index, utc=True, errors="coerce")
        values = values.assign(timestamp=idx)
        wanted = group[["timestamp", "__row_id"]].copy()
        merged = wanted.merge(values.reset_index(drop=True), on="timestamp", how="left", copy=False)
        merged["symbol"] = str(symbol)
        return merged

    grouped = [(str(symbol), group.copy()) for symbol, group in base.groupby("symbol", sort=False)]
    workers = max(1, int(os.getenv("EPM_RECENT_FAILURE_HYDRATE_WORKERS", "8") or "8"))
    workers = min(workers, max(1, len(grouped)))
    rows: list[pd.DataFrame] = []
    completed = 0
    print(
        f"[diagnose_meta_failures] hydrate feature_store symbols={len(grouped)} "
        f"requested={len(requested)} workers={workers}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_read_symbol_group, item) for item in grouped]
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                print(
                    f"[diagnose_meta_failures] hydrate progress {completed}/{len(futures)}",
                    flush=True,
                )
            result = fut.result()
            if isinstance(result, pd.DataFrame) and not result.empty:
                rows.append(result)

    if not rows:
        return base
    hydrated = pd.concat(rows, ignore_index=True)
    hydrated = hydrated.sort_values("__row_id", kind="mergesort")
    hydrated = base.merge(
        hydrated.drop(columns=["timestamp", "symbol"], errors="ignore"),
        on="__row_id",
        how="left",
        copy=False,
    )
    return _downcast_numeric(hydrated, exclude=["timestamp", "symbol", "__row_id"])


def _read_keyed_parquet_subset(path: Path | None, keys: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    requested = list(dict.fromkeys(str(c) for c in columns if str(c)))
    base = keys[["timestamp", "symbol"]].copy()
    base["__row_id"] = np.arange(len(base), dtype=np.int64)
    if path is None or not path.exists() or not requested:
        return base
    available = set(_parquet_columns(path))
    present = [c for c in requested if c in available]
    if not present:
        return base
    frame = pd.read_parquet(path, columns=["timestamp", "symbol", *present])
    frame = _normalise_keys(frame)
    frame = frame.sort_values(["timestamp", "symbol"], kind="mergesort").drop_duplicates(
        ["timestamp", "symbol"],
        keep="last",
    )
    merged = base.merge(frame, on=["timestamp", "symbol"], how="left", copy=False)
    return _downcast_numeric(merged, exclude=["timestamp", "symbol", "__row_id"])


def _latest_regime_context(root: Path) -> Path | None:
    candidates = sorted(root.glob("poc_*/regime_context_features.parquet"))
    return candidates[-1] if candidates else None


def _read_regime_features(path: Path | None, keys: pd.DataFrame, max_columns: int) -> pd.DataFrame:
    base = keys[["timestamp", "symbol"]].copy()
    base["__row_id"] = np.arange(len(base), dtype=np.int64)
    if path is None or not path.exists():
        return base
    cols = _parquet_columns(path)
    if not {"timestamp", "symbol"}.issubset(cols):
        return base
    feature_cols = [c for c in cols if c not in {"timestamp", "symbol"}][:max_columns]
    if not feature_cols:
        return base
    frame = pd.read_parquet(path, columns=["timestamp", "symbol", *feature_cols])
    frame = _normalise_keys(frame)
    frame = frame.sort_values(["timestamp", "symbol"], kind="mergesort").drop_duplicates(
        ["timestamp", "symbol"],
        keep="last",
    )
    frame = frame.rename(columns={c: f"url__{c}" for c in feature_cols})
    merged = base.merge(frame, on=["timestamp", "symbol"], how="left", copy=False)
    return _downcast_numeric(merged, exclude=["timestamp", "symbol", "__row_id"])


def _suffix_candidates(feature: str) -> list[str]:
    out: list[str] = []
    if "_H5_" in feature:
        out.append(feature.split("_H5_", 1)[1])
    for prefix in ("pred_H5_", "base_H5_", "base_lgbm_"):
        if feature.startswith(prefix):
            out.append(feature[len(prefix) :])
    if feature.startswith("pred_") and "_" in feature:
        tail = feature.rsplit("_", 1)[-1]
        if tail:
            out.append(tail)
    return list(dict.fromkeys(out))


def _assemble_selected_matrix(
    *,
    panel: pd.DataFrame,
    race: Any,
    feature_dir: Path,
    transform_cache: Path | None,
    symbol_columns: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = [
        str(feature)
        for feature in (getattr(race.best_model, "selected_features", []) or [])
        if not _is_forbidden_feature_name(str(feature))
    ]
    keys = panel[["timestamp", "symbol"]].copy()
    store_union = set().union(*symbol_columns.values()) if symbol_columns else set()
    cache_cols = set(_parquet_columns(transform_cache)) if transform_cache and transform_cache.exists() else set()

    store_request = [c for c in selected if c in store_union]
    cache_request = [c for c in selected if c in cache_cols]
    store = _hydrate_feature_store(feature_dir, keys, store_request, symbol_columns)
    generated = _read_keyed_parquet_subset(transform_cache, keys, cache_request)

    try:
        meta_features = race.best_model.get_training_meta_features()
    except Exception:
        meta_features = pd.DataFrame(index=panel.index)
    if len(meta_features) != len(panel):
        meta_features = pd.DataFrame(index=panel.index)

    defaults = dict(getattr(race.best_model, "model_effectiveness_history_defaults_", {}) or {})
    feature_stats = dict(getattr(race.best_model, "feature_stats_train", {}) or {})
    x = pd.DataFrame(index=panel.index)
    source_rows: list[dict[str, Any]] = []

    def assign(feature: str, values: Any, source: str) -> bool:
        ser = pd.Series(values, index=panel.index)
        ser = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x[feature] = ser.astype(np.float32, copy=False)
        source_rows.append(
            {
                "feature": feature,
                "source": source,
                "finite_fraction": float(np.isfinite(x[feature].to_numpy(dtype=np.float64, copy=False)).mean()),
                "missing_fraction": float(x[feature].isna().mean()),
            }
        )
        return True

    for feature in selected:
        if feature in panel.columns:
            assign(feature, panel[feature], "meta_oof_export_exact")
            continue
        if feature in meta_features.columns:
            assign(feature, meta_features[feature].to_numpy(copy=False), "training_meta_exact")
            continue
        if f"oof_{feature}" in panel.columns:
            assign(feature, panel[f"oof_{feature}"], "meta_oof_export_oof_prefix")
            continue
        assigned = False
        for suffix in _suffix_candidates(feature):
            if suffix in meta_features.columns:
                assigned = assign(feature, meta_features[suffix].to_numpy(copy=False), f"training_meta_suffix:{suffix}")
                break
            oof_suffix = f"oof_{suffix}"
            if oof_suffix in panel.columns:
                assigned = assign(feature, panel[oof_suffix], f"meta_oof_suffix:{oof_suffix}")
                break
        if assigned:
            continue
        if feature in generated.columns:
            generated_values = pd.to_numeric(generated[feature], errors="coerce")
            if bool(np.isfinite(generated_values.to_numpy(dtype=np.float64, copy=False)).any()):
                assign(feature, generated_values, "generated_transform_cache")
                continue
        if feature in store.columns:
            assign(feature, store[feature], "feature_store")
            continue
        if feature in defaults:
            assign(feature, np.full(len(panel), float(defaults[feature]), dtype=np.float32), "model_default")
            continue
        if feature in feature_stats and isinstance(feature_stats[feature], dict):
            median = feature_stats[feature].get("median", feature_stats[feature].get("mean", np.nan))
            if median is not None and np.isfinite(float(median)):
                assign(feature, np.full(len(panel), float(median), dtype=np.float32), "feature_stat_default")
                continue
        assign(feature, np.full(len(panel), np.nan, dtype=np.float32), "missing")

    coverage = pd.DataFrame(source_rows)
    summary = {
        "selected_features": int(len(selected)),
        "features_non_missing_source": int((coverage["source"] != "missing").sum()) if not coverage.empty else 0,
        "features_missing_source": int((coverage["source"] == "missing").sum()) if not coverage.empty else 0,
        "mean_finite_fraction": float(coverage["finite_fraction"].mean()) if not coverage.empty else 0.0,
        "source_counts": coverage["source"].value_counts().to_dict() if not coverage.empty else {},
    }
    return _downcast_numeric(x), coverage, summary


def _known_export_features(panel: pd.DataFrame) -> pd.DataFrame:
    cols: list[str] = []
    for col in panel.columns:
        if col in KEY_COLUMNS or _is_forbidden_feature_name(col):
            continue
        if pd.api.types.is_numeric_dtype(panel[col]) and _is_deployable_export_feature(col):
            cols.append(col)
    export = panel[cols].copy()
    rename = {c: f"export__{c}" for c in export.columns}
    return _downcast_numeric(export.rename(columns=rename))


def _merge_feature_candidates(
    selected_x: pd.DataFrame,
    export_x: pd.DataFrame,
    regime_x: pd.DataFrame,
    *,
    max_missing: float = 0.98,
) -> pd.DataFrame:
    selected_x = selected_x.loc[:, [c for c in selected_x.columns if not _is_forbidden_feature_name(c)]]
    export_x = export_x.loc[:, [c for c in export_x.columns if not _is_forbidden_feature_name(c)]]
    regime_x = regime_x.loc[:, [c for c in regime_x.columns if not _is_forbidden_feature_name(c)]]
    parts = [selected_x]
    if not export_x.empty:
        parts.append(export_x.loc[:, [c for c in export_x.columns if c not in selected_x.columns]])
    if not regime_x.empty:
        extra = [
            c
            for c in regime_x.columns
            if c not in {"timestamp", "symbol", "__row_id"}
            and c not in selected_x.columns
            and not _is_forbidden_feature_name(c)
        ]
        if extra:
            parts.append(regime_x[extra])
    x = pd.concat(parts, axis=1, copy=False)
    x = x.replace([np.inf, -np.inf], np.nan)
    keep: list[str] = []
    for col in x.columns:
        ser = pd.to_numeric(x[col], errors="coerce")
        missing = float(ser.isna().mean())
        if missing > max_missing:
            continue
        arr = ser.to_numpy(dtype=np.float64, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0 or float(np.nanstd(finite)) <= 1e-12:
            continue
        keep.append(col)
    return _downcast_numeric(x[keep])


def _period_stratified_sample(
    frame: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    *,
    seed: int = 7,
    period: str = "W",
) -> np.ndarray:
    n = len(frame)
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    tmp = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int64),
            "y": np.asarray(y, dtype=np.int8),
            "period": pd.to_datetime(frame["timestamp"], utc=True).dt.to_period(period).astype(str).to_numpy(),
        }
    )
    samples: list[np.ndarray] = []
    target_frac = max_rows / max(n, 1)
    for _, group in tmp.groupby(["period", "y"], sort=False):
        take = max(1, int(round(len(group) * target_frac)))
        take = min(take, len(group))
        samples.append(rng.choice(group["idx"].to_numpy(), size=take, replace=False))
    idx = np.concatenate(samples) if samples else np.arange(n, dtype=np.int64)
    if idx.size > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _prepare_model_matrix(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return _downcast_numeric(out)


def _fit_lgbm_cv(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    cv_kind: str,
    max_rows: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for these diagnostics")
    y = np.asarray(y, dtype=np.int8)
    valid = np.isfinite(y)
    x = x.loc[valid].reset_index(drop=True)
    ts = pd.to_datetime(timestamps.loc[valid], utc=True, errors="coerce").reset_index(drop=True)
    y = y[valid]
    if len(np.unique(y)) < 2 or len(y) < 200:
        return {
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "folds": 0,
            "rows": int(len(y)),
            "positive_rate": float(np.mean(y)) if len(y) else np.nan,
            "reason": "insufficient_classes_or_rows",
        }, pd.DataFrame()

    frame = pd.DataFrame({"timestamp": ts})
    sample_idx = _period_stratified_sample(frame, y, max_rows=max_rows, seed=seed)
    x = x.iloc[sample_idx].reset_index(drop=True)
    y = y[sample_idx]
    ts = ts.iloc[sample_idx].reset_index(drop=True)
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    x = x.iloc[order].reset_index(drop=True)
    y = y[order]

    if cv_kind == "temporal":
        n_splits = min(5, max(2, len(y) // 5000))
        splitter: Iterable[tuple[np.ndarray, np.ndarray]] = TimeSeriesSplit(n_splits=n_splits).split(x)
    else:
        n_splits = 5 if min(np.bincount(y)) >= 5 else 3
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y)

    aucs: list[float] = []
    imps: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splitter):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        min_child = max(50, int(math.ceil(0.025 * len(train_idx))))
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=8,
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(
                x.iloc[train_idx],
                y[train_idx],
                eval_set=[(x.iloc[test_idx], y[test_idx])],
                eval_metric="auc",
                callbacks=callbacks,
            )
        pred = clf.predict_proba(x.iloc[test_idx])[:, 1]
        auc = float(roc_auc_score(y[test_idx], pred))
        aucs.append(auc)
        booster = clf.booster_
        imps.append(
            pd.DataFrame(
                {
                    "feature": booster.feature_name(),
                    "gain": booster.feature_importance(importance_type="gain"),
                    "split": booster.feature_importance(importance_type="split"),
                    "fold": fold,
                }
            )
        )

    if not aucs:
        return {
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "folds": 0,
            "rows": int(len(y)),
            "positive_rate": float(np.mean(y)) if len(y) else np.nan,
            "reason": "no_valid_cv_folds",
        }, pd.DataFrame()
    imp = pd.concat(imps, ignore_index=True)
    agg = (
        imp.groupby("feature", as_index=False)
        .agg(gain_mean=("gain", "mean"), split_mean=("split", "mean"), fold_count=("fold", "nunique"))
        .sort_values(["gain_mean", "split_mean"], ascending=False)
    )
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "folds": int(len(aucs)),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "reason": "",
    }, agg


def _adversarial_nuisance_controls(panel: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy(dtype=np.float64, copy=False)
    finite_ts = np.isfinite(ts_ns)
    if finite_ts.any():
        min_ts = float(np.nanmin(ts_ns[finite_ts]))
        denom = max(float(np.nanmax(ts_ns[finite_ts]) - min_ts), 1.0)
        time_ordinal = (ts_ns - min_ts) / denom
    else:
        time_ordinal = np.zeros(len(panel), dtype=np.float64)

    rows_per_ts = ts.groupby(ts).transform("size").to_numpy(dtype=np.float32, copy=False)
    hour = ts.dt.hour.fillna(0).to_numpy(dtype=np.float32, copy=False)
    dow = ts.dt.dayofweek.fillna(0).to_numpy(dtype=np.float32, copy=False)
    symbols = panel["symbol"].astype(str) if "symbol" in panel.columns else pd.Series("", index=panel.index)
    first_seen = ts.groupby(symbols).transform("min")
    symbol_age_hours = ((ts - first_seen).dt.total_seconds() / 3600.0).fillna(0.0).to_numpy(dtype=np.float32)
    symbol_obs = symbols.map(symbols.value_counts()).fillna(0).to_numpy(dtype=np.float32)
    missing_fraction = x.isna().mean(axis=1).to_numpy(dtype=np.float32) if not x.empty else np.zeros(len(panel), dtype=np.float32)
    out = pd.DataFrame(
        {
            "nuisance_time_ordinal": np.asarray(time_ordinal, dtype=np.float32),
            "nuisance_session_sin": np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32, copy=False),
            "nuisance_session_cos": np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32, copy=False),
            "nuisance_dow_sin": np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32, copy=False),
            "nuisance_dow_cos": np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32, copy=False),
            "nuisance_rows_per_timestamp": np.log1p(rows_per_ts).astype(np.float32, copy=False),
            "nuisance_symbol_age_hours": np.log1p(np.maximum(symbol_age_hours, 0.0)).astype(np.float32, copy=False),
            "nuisance_symbol_observation_count": np.log1p(np.maximum(symbol_obs, 0.0)).astype(np.float32, copy=False),
            "nuisance_feature_missing_fraction": missing_fraction.astype(np.float32, copy=False),
        },
        index=panel.index,
    )
    return _downcast_numeric(out)


def _timestamp_representative_indices(panel: pd.DataFrame, seed: int) -> np.ndarray:
    if panel.empty or "timestamp" not in panel.columns:
        return np.arange(len(panel), dtype=np.int64)
    rng = np.random.default_rng(seed)
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    tmp = pd.DataFrame({"timestamp": ts, "idx": np.arange(len(panel), dtype=np.int64)})
    picks: list[int] = []
    for _timestamp, group in tmp.groupby("timestamp", sort=False):
        values = group["idx"].to_numpy(dtype=np.int64, copy=False)
        if values.size:
            picks.append(int(rng.choice(values)))
    return np.asarray(sorted(picks), dtype=np.int64)


def _residualize_against_nuisance(
    x: pd.DataFrame,
    nuisance: pd.DataFrame,
    *,
    max_features: int,
) -> pd.DataFrame:
    if x.empty or nuisance.empty:
        return pd.DataFrame(index=x.index)
    x = x.loc[:, [c for c in x.columns if not _is_forbidden_feature_name(c)]].replace([np.inf, -np.inf], np.nan)
    keep: list[str] = []
    for col in x.columns:
        ser = pd.to_numeric(x[col], errors="coerce")
        if ser.notna().mean() < 0.02:
            continue
        arr = ser.to_numpy(dtype=np.float64, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size < 50 or float(np.nanstd(finite)) <= 1e-12:
            continue
        keep.append(col)
    if max_features > 0 and len(keep) > max_features:
        variance = x[keep].var(numeric_only=True).sort_values(ascending=False)
        keep = list(variance.head(max_features).index)
    if not keep:
        return pd.DataFrame(index=x.index)

    n = nuisance.replace([np.inf, -np.inf], np.nan).copy()
    for col in n.columns:
        n[col] = pd.to_numeric(n[col], errors="coerce")
    n = n.loc[:, [c for c in n.columns if n[c].notna().mean() > 0.02]]
    if n.empty:
        return x[keep].add_prefix("resid__")
    n = n.fillna(n.median(numeric_only=True)).fillna(0.0)
    n_arr = n.to_numpy(dtype=np.float64, copy=False)
    n_mean = np.nanmean(n_arr, axis=0)
    n_std = np.nanstd(n_arr, axis=0)
    n_std = np.where(n_std <= 1e-12, 1.0, n_std)
    z = (n_arr - n_mean) / n_std
    design = np.column_stack([np.ones(z.shape[0], dtype=np.float64), z])

    x_num = x[keep].copy()
    for col in x_num.columns:
        x_num[col] = pd.to_numeric(x_num[col], errors="coerce")
    x_filled = x_num.fillna(x_num.median(numeric_only=True)).fillna(0.0)
    x_arr = x_filled.to_numpy(dtype=np.float64, copy=False)
    x_mean = np.nanmean(x_arr, axis=0)
    x_std = np.nanstd(x_arr, axis=0)
    x_std = np.where(x_std <= 1e-12, 1.0, x_std)
    x_z = (x_arr - x_mean) / x_std
    beta, *_ = np.linalg.lstsq(design, x_z, rcond=None)
    resid = x_z - design @ beta
    return _downcast_numeric(pd.DataFrame(resid.astype(np.float32, copy=False), columns=[f"resid__{c}" for c in keep]))


def _fit_adversarial_variants(
    *,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    max_rows: int,
    max_residual_features: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    idx = _timestamp_representative_indices(panel, seed=seed)
    if idx.size == 0:
        return {"reason": "empty_timestamp_balanced_panel"}, pd.DataFrame()
    panel_b = panel.iloc[idx].reset_index(drop=True)
    x_b = x.iloc[idx].reset_index(drop=True)
    y_b = np.asarray(y, dtype=np.int8)[idx]
    ts_b = timestamps.iloc[idx].reset_index(drop=True)
    nuisance = _adversarial_nuisance_controls(panel_b, x_b)
    residualized = _residualize_against_nuisance(x_b, nuisance, max_features=max_residual_features)

    raw_summary, raw_imp = _fit_lgbm_cv(
        x=x_b,
        y=y_b,
        timestamps=ts_b,
        cv_kind="stratified",
        max_rows=max_rows,
        seed=seed,
    )
    nuisance_summary, nuisance_imp = _fit_lgbm_cv(
        x=nuisance,
        y=y_b,
        timestamps=ts_b,
        cv_kind="stratified",
        max_rows=max_rows,
        seed=seed + 1,
    )
    residual_summary, residual_imp = _fit_lgbm_cv(
        x=residualized,
        y=y_b,
        timestamps=ts_b,
        cv_kind="stratified",
        max_rows=max_rows,
        seed=seed + 2,
    )
    stacked = pd.concat([nuisance.reset_index(drop=True), residualized.reset_index(drop=True)], axis=1, copy=False)
    stacked_summary, stacked_imp = _fit_lgbm_cv(
        x=stacked,
        y=y_b,
        timestamps=ts_b,
        cv_kind="stratified",
        max_rows=max_rows,
        seed=seed + 3,
    )
    raw_auc = float(raw_summary.get("auc_mean", np.nan))
    nuisance_auc = float(nuisance_summary.get("auc_mean", np.nan))
    residual_auc = float(residual_summary.get("auc_mean", np.nan))
    stacked_auc = float(stacked_summary.get("auc_mean", np.nan))
    summary = {
        "timestamp_balanced_rows": int(len(y_b)),
        "raw_auc": raw_auc,
        "nuisance_auc": nuisance_auc,
        "residualized_auc": residual_auc,
        "nuisance_plus_residualized_auc": stacked_auc,
        "raw_minus_nuisance_auc": raw_auc - nuisance_auc if np.isfinite(raw_auc) and np.isfinite(nuisance_auc) else np.nan,
        "residualized_minus_nuisance_auc": residual_auc - nuisance_auc
        if np.isfinite(residual_auc) and np.isfinite(nuisance_auc)
        else np.nan,
        "incremental_auc_beyond_nuisance": stacked_auc - nuisance_auc
        if np.isfinite(stacked_auc) and np.isfinite(nuisance_auc)
        else np.nan,
        "raw_folds": raw_summary.get("folds", 0),
        "nuisance_folds": nuisance_summary.get("folds", 0),
        "residualized_folds": residual_summary.get("folds", 0),
        "nuisance_plus_residualized_folds": stacked_summary.get("folds", 0),
        "raw_reason": raw_summary.get("reason", ""),
        "nuisance_reason": nuisance_summary.get("reason", ""),
        "residualized_reason": residual_summary.get("reason", ""),
        "nuisance_plus_residualized_reason": stacked_summary.get("reason", ""),
        "raw_feature_count": int(x_b.shape[1]),
        "nuisance_feature_count": int(nuisance.shape[1]),
        "residualized_feature_count": int(residualized.shape[1]),
        "nuisance_plus_residualized_feature_count": int(stacked.shape[1]),
    }
    imps: list[pd.DataFrame] = []
    if not raw_imp.empty:
        raw_imp = raw_imp.copy()
        raw_imp.insert(0, "variant", "raw_clean")
        imps.append(raw_imp)
    if not nuisance_imp.empty:
        nuisance_imp = nuisance_imp.copy()
        nuisance_imp.insert(0, "variant", "nuisance_only")
        imps.append(nuisance_imp)
    if not residual_imp.empty:
        residual_imp = residual_imp.copy()
        residual_imp.insert(0, "variant", "residualized")
        imps.append(residual_imp)
    if not stacked_imp.empty:
        stacked_imp = stacked_imp.copy()
        stacked_imp.insert(0, "variant", "nuisance_plus_residualized")
        imps.append(stacked_imp)
    return summary, pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()


def _weekly_high_conf_metrics(panel: pd.DataFrame, rank_threshold: float, min_week_rows: int) -> pd.DataFrame:
    df = panel.loc[pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= rank_threshold].copy()
    if df.empty:
        return pd.DataFrame()
    df["week"] = pd.to_datetime(df["timestamp"], utc=True).dt.to_period("W").dt.start_time
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["__rows_per_timestamp"] = ts.groupby(ts).transform("size").astype("float32")
    if "symbol" in df.columns:
        symbols = df["symbol"].astype(str)
        first_seen = ts.groupby(symbols).transform("min")
        df["__asset_age_hours"] = ((ts - first_seen).dt.total_seconds() / 3600.0).fillna(0.0).astype("float32")
    else:
        df["__asset_age_hours"] = 0.0
    observable_match_cols = [
        col
        for col in (
            "vol_z",
            "vol_z24",
            "rvol_z",
            "amihud_z",
            "spread_bps",
            "xasset_mkt_spread_bps",
            "volume_percentile",
            "market_dispersion_24h",
            "market_breadth_24h",
            "mkt_ret_eq_24h",
        )
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not _is_forbidden_feature_name(col)
    ]
    agg_spec: dict[str, tuple[str, str]] = {
        "rows": ("y_bin", "size"),
        "hit_rate": ("y_bin", "mean"),
        "mean_return": ("return", "mean"),
        "pred_mean": ("oof_pred", "mean"),
        "pred_std": ("oof_pred", "std"),
        "timestamp_count": ("timestamp", "nunique"),
        "rows_per_timestamp_mean": ("__rows_per_timestamp", "mean"),
        "asset_age_hours_mean": ("__asset_age_hours", "mean"),
    }
    if "symbol" in df.columns:
        agg_spec["symbol_count"] = ("symbol", "nunique")
    for col in observable_match_cols:
        agg_spec[f"match__{col}"] = (col, "mean")
    out = (
        df.groupby("week", as_index=False)
        .agg(**agg_spec)
        .sort_values("week")
    )
    if "symbol_count" not in out.columns:
        out["symbol_count"] = np.nan
    out["pred_std"] = pd.to_numeric(out["pred_std"], errors="coerce").fillna(0.0)
    out["usable_week"] = out["rows"] >= int(min_week_rows)
    return out


def _matched_baseline_weeks(
    weekly: pd.DataFrame,
    *,
    bad_week: str,
    usable_week_set: set[str],
    all_week_labels: set[str],
    max_weeks: int = 4,
) -> tuple[list[str], dict[str, Any]]:
    """Pick prior usable baseline weeks by observable similarity to a bad week."""
    if weekly.empty:
        return [], {"baseline_match_method": "none", "baseline_match_reason": "empty_weekly_metrics"}
    work = weekly.copy()
    work["week_label"] = pd.to_datetime(work["week"]).dt.strftime("%Y-%m-%d")
    bad_label = pd.Timestamp(bad_week).strftime("%Y-%m-%d")
    bad_rows = work.loc[work["week_label"].eq(bad_label)]
    prior_labels = sorted(
        w
        for w in all_week_labels
        if pd.Timestamp(w) < pd.Timestamp(bad_label) and w in usable_week_set and w != bad_label
    )
    if not prior_labels:
        return [], {"baseline_match_method": "prior_observable_similarity", "baseline_match_reason": "no_prior_usable_weeks"}
    if bad_rows.empty:
        return prior_labels[-max_weeks:], {
            "baseline_match_method": "last_prior_usable_weeks",
            "baseline_match_reason": "bad_week_missing_from_weekly_metrics",
            "baseline_match_candidate_weeks": int(len(prior_labels)),
            "baseline_match_score_mean": np.nan,
        }
    candidate = work.loc[work["week_label"].isin(prior_labels)].copy()
    if candidate.empty:
        return [], {"baseline_match_method": "prior_observable_similarity", "baseline_match_reason": "empty_prior_candidates"}
    match_cols = [
        c
        for c in (
            "rows",
            "symbol_count",
            "timestamp_count",
            "rows_per_timestamp_mean",
            "asset_age_hours_mean",
            "pred_mean",
            "pred_std",
        )
        if c in work.columns
    ]
    match_cols.extend([c for c in work.columns if c.startswith("match__")])
    usable_cols: list[str] = []
    bad = bad_rows.iloc[0]
    for col in match_cols:
        values = pd.to_numeric(work[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.size < 3 or float(np.nanstd(finite)) <= 1e-12 or not np.isfinite(float(bad.get(col, np.nan))):
            continue
        usable_cols.append(col)
    if not usable_cols:
        return prior_labels[-max_weeks:], {
            "baseline_match_method": "last_prior_usable_weeks",
            "baseline_match_reason": "no_observable_match_columns",
            "baseline_match_candidate_weeks": int(len(prior_labels)),
            "baseline_match_score_mean": np.nan,
        }
    scores = np.zeros(len(candidate), dtype=np.float64)
    for col in usable_cols:
        values = pd.to_numeric(work[col], errors="coerce")
        center = float(values.median(skipna=True))
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        scale = max((q75 - q25) / 1.349, float(values.std(skipna=True) or 0.0), 1e-8)
        bad_z = (float(bad[col]) - center) / scale
        cand_z = (pd.to_numeric(candidate[col], errors="coerce").fillna(center).to_numpy(dtype=np.float64) - center) / scale
        scores += np.abs(cand_z - bad_z)
    candidate = candidate.assign(__match_score=scores / max(float(len(usable_cols)), 1.0))
    selected = candidate.sort_values(["__match_score", "week_label"], ascending=[True, False]).head(max_weeks)
    baseline = sorted(selected["week_label"].astype(str).tolist())
    return baseline, {
        "baseline_match_method": "prior_observable_similarity",
        "baseline_match_reason": "",
        "baseline_match_candidate_weeks": int(len(candidate)),
        "baseline_match_feature_count": int(len(usable_cols)),
        "baseline_match_features": ",".join(usable_cols),
        "baseline_match_score_mean": float(selected["__match_score"].mean()) if not selected.empty else np.nan,
    }


def _bad_recent_weeks(
    weekly: pd.DataFrame,
    *,
    recent_weeks: int,
    min_week_rows: int,
) -> tuple[list[pd.Timestamp], dict[str, Any]]:
    if weekly.empty:
        return [], {"reason": "no_weekly_metrics"}
    usable = weekly.loc[weekly["rows"] >= int(min_week_rows)].copy()
    if usable.empty:
        usable = weekly.copy()
    max_week = pd.to_datetime(usable["week"]).max()
    cutoff = max_week - pd.Timedelta(weeks=recent_weeks)
    recent = usable.loc[pd.to_datetime(usable["week"]) >= cutoff].copy()
    hist = usable.loc[pd.to_datetime(usable["week"]) < cutoff].copy()
    if recent.empty:
        return [], {"reason": "no_recent_weeks", "cutoff": cutoff}
    if hist.empty:
        hist = usable.loc[pd.to_datetime(usable["week"]) < max_week].copy()
    hit_q25 = float(hist["hit_rate"].quantile(0.25)) if not hist.empty else float(usable["hit_rate"].quantile(0.25))
    ret_q25 = float(hist["mean_return"].quantile(0.25)) if not hist.empty else float(usable["mean_return"].quantile(0.25))
    bad = recent.loc[(recent["hit_rate"] <= hit_q25) | (recent["mean_return"] <= ret_q25)].copy()
    reason = "threshold"
    if bad.empty:
        recent = recent.assign(
            composite_rank=recent["hit_rate"].rank(method="first") + recent["mean_return"].rank(method="first")
        )
        bad = recent.nsmallest(1, "composite_rank").copy()
        reason = "fallback_worst_recent_week"
    weeks = [pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x) for x in bad["week"]]
    return weeks, {
        "reason": reason,
        "recent_cutoff_week": cutoff,
        "hit_q25_reference": hit_q25,
        "return_q25_reference": ret_q25,
        "bad_week_count": int(len(weeks)),
    }


def _high_conf_failure_diagnostic(
    head: HeadContext,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    out_dir: Path,
    *,
    rank_threshold: float,
    max_rows: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    mask = pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= rank_threshold
    data = panel.loc[mask].reset_index(drop=True)
    xh = x.loc[mask].reset_index(drop=True)
    y = (pd.to_numeric(data["y_bin"], errors="coerce").fillna(0).to_numpy() <= 0).astype(np.int8)
    summary, imp = _fit_lgbm_cv(
        x=xh,
        y=y,
        timestamps=data["timestamp"],
        cv_kind="temporal",
        max_rows=max_rows,
        seed=seed,
    )
    summary.update(
        {
            "head": head.head,
            "strategy_id": head.strategy_id,
            "diagnostic": "high_conf_failure",
            "rank_threshold": rank_threshold,
            "candidate_feature_count": int(xh.shape[1]),
            "high_conf_rows": int(len(data)),
            "high_conf_failures": int(y.sum()),
            "date_min": data["timestamp"].min(),
            "date_max": data["timestamp"].max(),
        }
    )
    if not imp.empty:
        imp.insert(0, "head", head.head)
        imp.to_csv(out_dir / f"{head.head}_high_conf_failure_importance.csv", index=False)
    return summary, imp


def _adversarial_diagnostics(
    head: HeadContext,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    out_dir: Path,
    *,
    rank_threshold: float,
    recent_weeks: int,
    min_week_rows: int,
    max_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    high = panel.loc[pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= rank_threshold].copy()
    xh = x.loc[high.index].reset_index(drop=True)
    high = high.reset_index(drop=True)
    weekly = _weekly_high_conf_metrics(panel, rank_threshold, min_week_rows)
    weekly.insert(0, "head", head.head)
    weekly.to_csv(out_dir / f"{head.head}_weekly_high_conf_metrics.csv", index=False)
    bad_weeks, bad_meta = _bad_recent_weeks(weekly, recent_weeks=recent_weeks, min_week_rows=min_week_rows)
    if high.empty or not bad_weeks:
        return [
            {
                "head": head.head,
                "diagnostic": "adversarial_global_bad_weeks",
                "auc_mean": np.nan,
                "reason": "no_high_conf_rows_or_bad_weeks",
                **bad_meta,
            }
        ], pd.DataFrame(), pd.DataFrame()

    weeks = pd.to_datetime(high["timestamp"], utc=True).dt.to_period("W").dt.start_time
    week_labels = pd.Series(
        pd.to_datetime(weeks, utc=True).dt.strftime("%Y-%m-%d"),
        index=high.index,
        dtype="string",
    )
    bad_week_set = {pd.Timestamp(w).strftime("%Y-%m-%d") for w in bad_weeks}
    y_global = week_labels.isin(bad_week_set).astype(np.int8).to_numpy()
    max_bad_week = max(pd.Timestamp(w) for w in bad_week_set)
    older = pd.to_datetime(week_labels) < max_bad_week
    keep = older | (y_global == 1)
    global_summary, global_imp = _fit_lgbm_cv(
        x=xh.loc[keep].reset_index(drop=True),
        y=y_global[keep],
        timestamps=high.loc[keep, "timestamp"].reset_index(drop=True),
        cv_kind="stratified",
        max_rows=max_rows,
        seed=seed + 101,
    )
    global_summary.update(
        {
            "head": head.head,
            "strategy_id": head.strategy_id,
            "diagnostic": "adversarial_global_bad_weeks",
            "rank_threshold": rank_threshold,
            "bad_weeks": ",".join(sorted(bad_week_set)),
            "bad_rows": int(y_global[keep].sum()),
            "normal_rows": int((1 - y_global[keep]).sum()),
            **bad_meta,
        }
    )
    if not global_imp.empty:
        global_imp.insert(0, "head", head.head)
        global_imp.insert(1, "diagnostic", "adversarial_global_bad_weeks")
        global_imp.to_csv(out_dir / f"{head.head}_adversarial_global_importance.csv", index=False)

    local_rows: list[dict[str, Any]] = []
    local_imps: list[pd.DataFrame] = []
    usable_weeks = weekly.loc[weekly["rows"] >= min_week_rows, "week"]
    usable_week_set = set(pd.to_datetime(usable_weeks).dt.strftime("%Y-%m-%d"))
    all_week_labels = set(week_labels.astype(str))
    for i, bad_week in enumerate(sorted(bad_week_set)):
        baseline, baseline_meta = _matched_baseline_weeks(
            weekly,
            bad_week=bad_week,
            usable_week_set=usable_week_set,
            all_week_labels=all_week_labels,
            max_weeks=4,
        )
        bad_mask = week_labels.eq(str(bad_week))
        baseline_mask = week_labels.isin([str(w) for w in baseline])
        local_keep = (bad_mask | baseline_mask).to_numpy()
        if not baseline or int(local_keep.sum()) < 100:
            local_rows.append(
                {
                    "head": head.head,
                    "diagnostic": "adversarial_local_bad_week",
                    "bad_week": bad_week,
                    "auc_mean": np.nan,
                    "reason": "insufficient_matched_baseline",
                    **baseline_meta,
                }
            )
            continue
        local_y = bad_mask.astype(np.int8).to_numpy()
        summary, imp = _fit_lgbm_cv(
            x=xh.loc[local_keep].reset_index(drop=True),
            y=local_y[local_keep],
            timestamps=high.loc[local_keep, "timestamp"].reset_index(drop=True),
            cv_kind="stratified",
            max_rows=max_rows,
            seed=seed + 200 + i,
        )
        summary.update(
            {
                "head": head.head,
                "strategy_id": head.strategy_id,
                "diagnostic": "adversarial_local_bad_week",
                "rank_threshold": rank_threshold,
                "bad_week": bad_week,
                "baseline_weeks": ",".join(baseline),
                "bad_rows": int(local_y[local_keep].sum()),
                "normal_rows": int((1 - local_y[local_keep]).sum()),
                **baseline_meta,
            }
        )
        local_rows.append(summary)
        if not imp.empty:
            imp.insert(0, "head", head.head)
            imp.insert(1, "bad_week", bad_week)
            local_imps.append(imp)

    local_imp_df = pd.concat(local_imps, ignore_index=True) if local_imps else pd.DataFrame()
    if not local_imp_df.empty:
        local_imp_df.to_csv(out_dir / f"{head.head}_adversarial_local_importance.csv", index=False)
    return [global_summary, *local_rows], global_imp, local_imp_df


def _residualized_adversarial_diagnostics(
    head: HeadContext,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    out_dir: Path,
    *,
    rank_threshold: float,
    recent_weeks: int,
    min_week_rows: int,
    max_rows: int,
    max_residual_features: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    high = panel.loc[pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= rank_threshold].copy()
    xh = x.loc[high.index].reset_index(drop=True)
    high = high.reset_index(drop=True)
    weekly = _weekly_high_conf_metrics(panel, rank_threshold, min_week_rows)
    bad_weeks, bad_meta = _bad_recent_weeks(weekly, recent_weeks=recent_weeks, min_week_rows=min_week_rows)
    if high.empty or not bad_weeks:
        return [
            {
                "head": head.head,
                "diagnostic": "residualized_adversarial_global_bad_weeks",
                "reason": "no_high_conf_rows_or_bad_weeks",
                **bad_meta,
            }
        ], pd.DataFrame()

    weeks = pd.to_datetime(high["timestamp"], utc=True).dt.to_period("W").dt.start_time
    week_labels = pd.Series(pd.to_datetime(weeks, utc=True).dt.strftime("%Y-%m-%d"), index=high.index, dtype="string")
    bad_week_set = {pd.Timestamp(w).strftime("%Y-%m-%d") for w in bad_weeks}
    y_global = week_labels.isin(bad_week_set).astype(np.int8).to_numpy()
    max_bad_week = max(pd.Timestamp(w) for w in bad_week_set)
    older = pd.to_datetime(week_labels) < max_bad_week
    keep = (older | (y_global == 1)).to_numpy()
    rows: list[dict[str, Any]] = []
    imps: list[pd.DataFrame] = []

    global_summary, global_imp = _fit_adversarial_variants(
        panel=high.loc[keep].reset_index(drop=True),
        x=xh.loc[keep].reset_index(drop=True),
        y=y_global[keep],
        timestamps=high.loc[keep, "timestamp"].reset_index(drop=True),
        max_rows=max_rows,
        max_residual_features=max_residual_features,
        seed=seed + 501,
    )
    global_summary.update(
        {
            "head": head.head,
            "strategy_id": head.strategy_id,
            "diagnostic": "residualized_adversarial_global_bad_weeks",
            "rank_threshold": rank_threshold,
            "bad_weeks": ",".join(sorted(bad_week_set)),
            "bad_rows": int(y_global[keep].sum()),
            "normal_rows": int((1 - y_global[keep]).sum()),
            **bad_meta,
        }
    )
    rows.append(global_summary)
    if not global_imp.empty:
        global_imp.insert(0, "head", head.head)
        global_imp.insert(1, "diagnostic", "residualized_adversarial_global_bad_weeks")
        imps.append(global_imp)

    usable_weeks = weekly.loc[weekly["rows"] >= min_week_rows, "week"]
    usable_week_set = set(pd.to_datetime(usable_weeks).dt.strftime("%Y-%m-%d"))
    all_week_labels = set(week_labels.astype(str))
    for i, bad_week in enumerate(sorted(bad_week_set)):
        baseline, baseline_meta = _matched_baseline_weeks(
            weekly,
            bad_week=bad_week,
            usable_week_set=usable_week_set,
            all_week_labels=all_week_labels,
            max_weeks=4,
        )
        bad_mask = week_labels.eq(str(bad_week))
        baseline_mask = week_labels.isin([str(w) for w in baseline])
        local_keep = (bad_mask | baseline_mask).to_numpy()
        if not baseline or int(local_keep.sum()) < 100:
            rows.append(
                {
                    "head": head.head,
                    "strategy_id": head.strategy_id,
                    "diagnostic": "residualized_adversarial_local_bad_week",
                    "bad_week": bad_week,
                    "reason": "insufficient_matched_baseline",
                    **baseline_meta,
                }
            )
            continue
        local_y = bad_mask.astype(np.int8).to_numpy()
        summary, imp = _fit_adversarial_variants(
            panel=high.loc[local_keep].reset_index(drop=True),
            x=xh.loc[local_keep].reset_index(drop=True),
            y=local_y[local_keep],
            timestamps=high.loc[local_keep, "timestamp"].reset_index(drop=True),
            max_rows=max_rows,
            max_residual_features=max_residual_features,
            seed=seed + 600 + i,
        )
        summary.update(
            {
                "head": head.head,
                "strategy_id": head.strategy_id,
                "diagnostic": "residualized_adversarial_local_bad_week",
                "rank_threshold": rank_threshold,
                "bad_week": bad_week,
                "baseline_weeks": ",".join(baseline),
                "bad_rows": int(local_y[local_keep].sum()),
                "normal_rows": int((1 - local_y[local_keep]).sum()),
                **baseline_meta,
            }
        )
        rows.append(summary)
        if not imp.empty:
            imp.insert(0, "head", head.head)
            imp.insert(1, "bad_week", bad_week)
            imp.insert(2, "diagnostic", "residualized_adversarial_local_bad_week")
            imps.append(imp)

    out = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
    if not out.empty:
        out.to_csv(out_dir / f"{head.head}_residualized_adversarial_importance.csv", index=False)
    return rows, out


def _pick_context_columns(panel: pd.DataFrame, x: pd.DataFrame, max_cols: int = 20) -> list[str]:
    all_cols: list[str] = []
    for source in (panel, x):
        all_cols.extend(
            c
            for c in source.columns
            if c not in KEY_COLUMNS
            and not _is_forbidden_feature_name(c)
            and pd.api.types.is_numeric_dtype(source[c])
        )
    all_cols = list(dict.fromkeys(all_cols))

    groups: list[tuple[str, tuple[str, ...], int]] = [
        ("funding", ("fund", "carry"), 4),
        ("oi", ("oi_", "_oi", "open_interest", "leverage"), 5),
        ("liquidity_spread", ("liquid", "spread", "depth", "amihud", "volume", "rvol"), 6),
        ("volatility", ("vol", "rv", "atr", "range", "barrier"), 6),
        ("trend_path", ("trend", "slope", "compression", "efficiency", "chop", "entropy"), 5),
        ("model_state", ("gmm", "dae", "cluster", "mahal", "leaf", "support", "score_path"), 6),
    ]
    selected: list[str] = []
    for _, patterns, limit in groups:
        matches = [c for c in all_cols if any(p in c.lower() for p in patterns)]
        selected.extend(matches[:limit])
    if len(selected) < max_cols:
        selected.extend([c for c in all_cols if any(pat in c.lower() for pat in CONTEXT_PATTERNS)])
    return list(dict.fromkeys(selected))[:max_cols]


def _pick_archetype_columns(panel: pd.DataFrame, x: pd.DataFrame, max_cols: int = 12) -> list[str]:
    """Pick latent/context columns for residual ~ leaf * archetype diagnostics."""
    all_cols: list[str] = []
    for source in (panel, x):
        all_cols.extend(
            c
            for c in source.columns
            if c not in KEY_COLUMNS
            and not _is_forbidden_feature_name(c)
            and pd.api.types.is_numeric_dtype(source[c])
        )
    all_cols = list(dict.fromkeys(all_cols))
    priority_groups: list[tuple[tuple[str, ...], int]] = [
        (("gmm", "dae", "cluster", "mahal", "regime", "archetype"), 5),
        (("support", "leaf", "score_path", "score_early", "rank_100_minus_50"), 3),
        (("fund", "carry", "oi_", "_oi", "leverage"), 4),
        (("liquid", "spread", "depth", "amihud", "volume", "rvol"), 3),
        (("vol", "rv", "atr", "range"), 3),
    ]
    selected: list[str] = []
    for patterns, limit in priority_groups:
        matches = [c for c in all_cols if any(p in c.lower() for p in patterns)]
        selected.extend(matches[:limit])
    return list(dict.fromkeys(selected))[:max_cols]


def _leaf_row_sample(panel: pd.DataFrame, rank_threshold: float, max_rows: int, seed: int) -> np.ndarray:
    rank = pd.to_numeric(panel["oof_rank_pct"], errors="coerce")
    high = (rank >= rank_threshold).to_numpy()
    if not high.any():
        high = np.ones(len(panel), dtype=bool)
    y = (pd.to_numeric(panel["y_bin"], errors="coerce").fillna(0).to_numpy() <= 0).astype(np.int8)
    idx_all = np.flatnonzero(high)
    sampled = _period_stratified_sample(panel.iloc[idx_all].reset_index(drop=True), y[idx_all], max_rows=max_rows, seed=seed)
    return idx_all[sampled]


def _tree_indices(n_trees: int, stride: int, max_trees: int) -> np.ndarray:
    stride = max(1, int(stride))
    idx = np.arange(0, n_trees, stride, dtype=np.int32)
    if max_trees > 0 and idx.size > max_trees:
        positions = np.linspace(0, idx.size - 1, max_trees).round().astype(np.int32)
        idx = idx[positions]
    return idx


def _leaf_stats_for_models(
    *,
    head: HeadContext,
    model_kind: str,
    models: list[Any],
    x: pd.DataFrame,
    panel: pd.DataFrame,
    rank_threshold: float,
    max_rows: int,
    min_support: int,
    tree_stride: int,
    max_trees_per_model: int,
    seed: int,
) -> pd.DataFrame:
    if not models:
        return pd.DataFrame()
    row_idx = _leaf_row_sample(panel, rank_threshold, max_rows=max_rows, seed=seed)
    if row_idx.size == 0:
        return pd.DataFrame()
    sample = panel.iloc[row_idx].reset_index(drop=True)
    y = pd.to_numeric(sample["y_bin"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    pred = pd.to_numeric(sample["oof_pred"], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
    recent_cutoff = pd.to_datetime(sample["timestamp"], utc=True).max() - pd.Timedelta(days=28)
    recent = (pd.to_datetime(sample["timestamp"], utc=True) >= recent_cutoff).to_numpy()
    hist = ~recent
    n_recent_total = int(recent.sum())
    n_hist_total = int(hist.sum())
    if n_recent_total == 0 or n_hist_total == 0:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for model_idx, model in enumerate(models):
        feature_names = list(getattr(model, "feature_name_", []) or [])
        if not feature_names and hasattr(model, "booster_"):
            try:
                feature_names = [str(name) for name in model.booster_.feature_name()]
            except Exception:
                feature_names = []
        feature_names = feature_names or list(x.columns)
        missing = [c for c in feature_names if c not in x.columns]
        if missing:
            xm = x.copy()
            for col in missing:
                xm[col] = np.nan
            x_use = xm[feature_names]
        else:
            x_use = x[feature_names]
        x_sample = _prepare_model_matrix(x_use.iloc[row_idx].reset_index(drop=True))
        try:
            leaves = model.booster_.predict(x_sample, pred_leaf=True)
        except Exception as exc:
            _log(f"[leaf] failed leaf extraction head={head.head} kind={model_kind} model={model_idx}: {exc}")
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        for tree_idx in _tree_indices(leaves.shape[1], tree_stride, max_trees_per_model):
            leaf = leaves[:, int(tree_idx)].astype(np.int32, copy=False)
            codes, inv = np.unique(leaf, return_inverse=True)
            total = np.bincount(inv)
            recent_count = np.bincount(inv, weights=recent.astype(np.float32))
            hist_count = total - recent_count
            valid = (total >= min_support) & (recent_count >= max(5, min_support // 10)) & (hist_count >= max(5, min_support // 10))
            if not valid.any():
                continue
            recent_y = np.bincount(inv, weights=np.where(recent, y, 0.0))
            hist_y = np.bincount(inv, weights=np.where(hist, y, 0.0))
            resid = y - pred
            resid = np.where(np.isfinite(resid), resid, 0.0)
            recent_resid = np.bincount(inv, weights=np.where(recent, resid, 0.0))
            hist_resid = np.bincount(inv, weights=np.where(hist, resid, 0.0))
            recent_pred = np.bincount(inv, weights=np.where(recent, np.nan_to_num(pred, nan=0.0), 0.0))
            hist_pred = np.bincount(inv, weights=np.where(hist, np.nan_to_num(pred, nan=0.0), 0.0))
            safe_recent = np.maximum(recent_count, 1.0)
            safe_hist = np.maximum(hist_count, 1.0)
            reliability = np.minimum(1.0, np.minimum(recent_count, hist_count) / max(float(min_support), 1.0))
            occ_recent = recent_count / max(n_recent_total, 1)
            occ_hist = hist_count / max(n_hist_total, 1)
            outcome_shift = recent_y / safe_recent - hist_y / safe_hist
            calibration_shift = recent_resid / safe_recent - hist_resid / safe_hist
            score = reliability * (
                np.abs(outcome_shift) + np.abs(calibration_shift) + 5.0 * np.abs(occ_recent - occ_hist)
            )
            part = pd.DataFrame(
                {
                    "head": head.head,
                    "strategy_id": head.strategy_id,
                    "model_kind": model_kind,
                    "model_idx": model_idx,
                    "tree_idx": int(tree_idx),
                    "leaf_id": codes,
                    "n_total": total,
                    "n_recent": recent_count,
                    "n_history": hist_count,
                    "occupancy_recent": occ_recent,
                    "occupancy_history": occ_hist,
                    "occupancy_shift": occ_recent - occ_hist,
                    "outcome_recent": recent_y / safe_recent,
                    "outcome_history": hist_y / safe_hist,
                    "outcome_shift": outcome_shift,
                    "pred_recent": recent_pred / safe_recent,
                    "pred_history": hist_pred / safe_hist,
                    "calibration_shift": calibration_shift,
                    "shrinkage_reliability": reliability,
                    "instability_score": score,
                }
            )
            rows.append(part.loc[valid])
    if not rows:
        return pd.DataFrame()
    stats = pd.concat(rows, ignore_index=True)
    stats = stats.sort_values("instability_score", ascending=False).reset_index(drop=True)
    return stats


def _enrich_leaf_context(
    leaf_stats: pd.DataFrame,
    *,
    models: list[Any],
    x: pd.DataFrame,
    panel: pd.DataFrame,
    context_cols: list[str],
    top_n: int,
    rank_threshold: float,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if leaf_stats.empty:
        return leaf_stats
    top = leaf_stats.head(top_n).copy()
    row_idx = _leaf_row_sample(panel, rank_threshold, max_rows=max_rows, seed=seed)
    sample = panel.iloc[row_idx].reset_index(drop=True)
    recent_cutoff = pd.to_datetime(sample["timestamp"], utc=True).max() - pd.Timedelta(days=28)
    recent = (pd.to_datetime(sample["timestamp"], utc=True) >= recent_cutoff).to_numpy()
    panel_context_cols = [c for c in context_cols if c in panel.columns]
    x_context_cols = [c for c in context_cols if c in x.columns]
    context_parts: list[pd.DataFrame] = []
    if panel_context_cols:
        context_parts.append(panel[panel_context_cols])
    if x_context_cols:
        context_parts.append(x[x_context_cols])
    context_frame = pd.concat(context_parts, axis=1) if context_parts else pd.DataFrame(index=panel.index)
    context_frame = context_frame.loc[:, ~context_frame.columns.duplicated()]
    context_sample = context_frame.iloc[row_idx].reset_index(drop=True)
    enriched: list[dict[str, Any]] = []
    for (model_idx, tree_idx), group in top.groupby(["model_idx", "tree_idx"], sort=False):
        model = models[int(model_idx)]
        feature_names = list(getattr(model, "feature_name_", []) or [])
        if not feature_names and hasattr(model, "booster_"):
            try:
                feature_names = [str(name) for name in model.booster_.feature_name()]
            except Exception:
                feature_names = []
        feature_names = feature_names or list(x.columns)
        xm = x.copy()
        for col in feature_names:
            if col not in xm.columns:
                xm[col] = np.nan
        x_sample = _prepare_model_matrix(xm[feature_names].iloc[row_idx].reset_index(drop=True))
        try:
            leaves = model.booster_.predict(x_sample, pred_leaf=True)
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        leaf_col = leaves[:, int(tree_idx)].astype(np.int32, copy=False)
        for _, stat in group.iterrows():
            mask = leaf_col == int(stat["leaf_id"])
            recent_mask = mask & recent
            hist_mask = mask & (~recent)
            row = stat.to_dict()
            symbols = sample.loc[mask, "symbol"].astype(str)
            weeks = pd.to_datetime(sample.loc[mask, "timestamp"], utc=True).dt.to_period("W").astype(str)
            row["asset_top"] = symbols.value_counts().index[0] if len(symbols) else ""
            row["asset_top_share"] = float(symbols.value_counts(normalize=True).iloc[0]) if len(symbols) else np.nan
            row["week_top"] = weeks.value_counts().index[0] if len(weeks) else ""
            row["week_top_share"] = float(weeks.value_counts(normalize=True).iloc[0]) if len(weeks) else np.nan
            for col in context_cols[:24]:
                if col not in context_sample.columns:
                    continue
                values = pd.to_numeric(context_sample[col], errors="coerce")
                row[f"context_recent_mean__{col}"] = float(values.loc[recent_mask].mean()) if recent_mask.any() else np.nan
                row[f"context_history_mean__{col}"] = float(values.loc[hist_mask].mean()) if hist_mask.any() else np.nan
                row[f"context_leaf_q10__{col}"] = float(values.loc[mask].quantile(0.10)) if mask.any() else np.nan
                row[f"context_leaf_q90__{col}"] = float(values.loc[mask].quantile(0.90)) if mask.any() else np.nan
            enriched.append(row)
    return pd.DataFrame(enriched) if enriched else top


def _leaf_archetype_interactions(
    leaf_stats: pd.DataFrame,
    *,
    head: HeadContext,
    model_kind: str,
    models: list[Any],
    x: pd.DataFrame,
    panel: pd.DataFrame,
    archetype_cols: list[str],
    context_cols: list[str] | None,
    top_leaves: int,
    rank_threshold: float,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if leaf_stats.empty or not models or not archetype_cols:
        return pd.DataFrame()
    row_idx = _leaf_row_sample(panel, rank_threshold, max_rows=max_rows, seed=seed)
    if row_idx.size == 0:
        return pd.DataFrame()
    sample = panel.iloc[row_idx].reset_index(drop=True)
    y = pd.to_numeric(sample["y_bin"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    pred = pd.to_numeric(sample["oof_pred"], errors="coerce").to_numpy(dtype=np.float32)
    residual = y - np.nan_to_num(pred, nan=np.nanmean(pred) if np.isfinite(pred).any() else 0.0)
    residual = np.where(np.isfinite(residual), residual, 0.0).astype(np.float32, copy=False)

    panel_cols = [c for c in archetype_cols if c in panel.columns]
    x_cols = [c for c in archetype_cols if c in x.columns]
    parts: list[pd.DataFrame] = []
    if panel_cols:
        parts.append(panel[panel_cols])
    if x_cols:
        parts.append(x[x_cols])
    if not parts:
        return pd.DataFrame()
    archetype = pd.concat(parts, axis=1)
    archetype = archetype.loc[:, ~archetype.columns.duplicated()]
    archetype = archetype.iloc[row_idx].reset_index(drop=True)
    available_cols = [c for c in archetype_cols if c in archetype.columns]
    if not available_cols:
        return pd.DataFrame()

    timestamps = pd.to_datetime(sample["timestamp"], utc=True, errors="coerce")
    week_labels = timestamps.dt.to_period("W").astype(str).to_numpy()
    realized_return = (
        pd.to_numeric(sample["return"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if "return" in sample.columns
        else np.full(len(sample), np.nan, dtype=np.float32)
    )

    def _component_slope(signal: np.ndarray, row_mask: np.ndarray, *, min_count: int) -> float:
        finite_signal = np.isfinite(signal) & np.isfinite(residual) & row_mask
        if int(finite_signal.sum()) < min_count:
            return np.nan
        variance = float(np.nanmean(signal[finite_signal] ** 2))
        if not np.isfinite(variance) or variance <= 1e-8:
            return np.nan
        return float(np.nanmean(residual[finite_signal] * signal[finite_signal]) / variance)

    def _episode_sign_stats(signal: np.ndarray, leaf_mask: np.ndarray, base_mask: np.ndarray) -> dict[str, float]:
        deltas: list[float] = []
        for week in pd.unique(week_labels):
            week_mask = week_labels == week
            leaf_week = leaf_mask & week_mask
            base_week = base_mask & week_mask
            if int(leaf_week.sum()) < 10 or int(base_week.sum()) < 30:
                continue
            global_week_slope = _component_slope(signal, base_week, min_count=30)
            leaf_week_slope = _component_slope(signal, leaf_week, min_count=10)
            if np.isfinite(global_week_slope) and np.isfinite(leaf_week_slope):
                deltas.append(float(leaf_week_slope - global_week_slope))
        if not deltas:
            return {
                "episode_count": 0.0,
                "episode_sign_stability": np.nan,
                "episode_interaction_mean": np.nan,
                "episode_interaction_std": np.nan,
            }
        arr = np.asarray(deltas, dtype=np.float32)
        nonzero = arr[np.abs(arr) > 1e-8]
        if nonzero.size == 0:
            sign_stability = 0.0
        else:
            sign_stability = float(abs(np.nanmean(np.sign(nonzero))))
        return {
            "episode_count": float(arr.size),
            "episode_sign_stability": sign_stability,
            "episode_interaction_mean": float(np.nanmean(arr)),
            "episode_interaction_std": float(np.nanstd(arr)),
        }

    def _safe_mean(values: np.ndarray, row_mask: np.ndarray) -> float:
        mask = row_mask & np.isfinite(values)
        return float(np.nanmean(values[mask])) if bool(mask.any()) else np.nan

    archetype_components: dict[str, dict[str, np.ndarray]] = {}
    for col in available_cols:
        z = pd.to_numeric(archetype[col], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(z) & np.isfinite(residual)
        if int(finite.sum()) < 50:
            continue
        z_mean = float(np.nanmean(z[finite]))
        z_std = float(np.nanstd(z[finite]))
        if not np.isfinite(z_std) or z_std <= 1e-8:
            continue
        z_norm = ((z - z_mean) / z_std).astype(np.float32, copy=False)
        period_state = (
            pd.Series(z_norm)
            .groupby(timestamps, sort=False)
            .transform("mean")
            .to_numpy(dtype=np.float32, copy=False)
        )
        within_timestamp_state = (z_norm - period_state).astype(np.float32, copy=False)
        archetype_components[col] = {
            "raw": z,
            "norm": z_norm,
            "period": period_state,
            "within": within_timestamp_state,
            "finite": finite,
        }
    if not archetype_components:
        return pd.DataFrame()

    context_sample = pd.DataFrame(index=sample.index)
    if context_cols:
        context_parts: list[pd.DataFrame] = []
        panel_context_cols = [c for c in context_cols if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
        x_context_cols = [c for c in context_cols if c in x.columns and pd.api.types.is_numeric_dtype(x[c])]
        if panel_context_cols:
            context_parts.append(panel[panel_context_cols])
        if x_context_cols:
            context_parts.append(x[x_context_cols])
        if context_parts:
            context_sample = pd.concat(context_parts, axis=1)
            context_sample = context_sample.loc[:, ~context_sample.columns.duplicated()]
            context_sample = context_sample.iloc[row_idx].reset_index(drop=True)
            context_sample = context_sample.loc[:, list(context_sample.columns[:8])]
    recent_cutoff = timestamps.max() - pd.Timedelta(days=28)
    recent_sample_mask = (timestamps >= recent_cutoff).to_numpy(dtype=bool, copy=False)
    history_sample_mask = ~recent_sample_mask

    top = leaf_stats.head(top_leaves).copy()
    rows: list[dict[str, Any]] = []
    for (model_idx, tree_idx), group in top.groupby(["model_idx", "tree_idx"], sort=False):
        model = models[int(model_idx)]
        feature_names = list(getattr(model, "feature_name_", []) or [])
        if not feature_names and hasattr(model, "booster_"):
            try:
                feature_names = [str(name) for name in model.booster_.feature_name()]
            except Exception:
                feature_names = []
        feature_names = feature_names or list(x.columns)
        xm = x.copy()
        for col in feature_names:
            if col not in xm.columns:
                xm[col] = np.nan
        x_sample = _prepare_model_matrix(xm[feature_names].iloc[row_idx].reset_index(drop=True))
        try:
            leaves = model.booster_.predict(x_sample, pred_leaf=True)
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        leaf_col = leaves[:, int(tree_idx)].astype(np.int32, copy=False)
        for _, stat in group.iterrows():
            mask = leaf_col == int(stat["leaf_id"])
            n_leaf = int(mask.sum())
            if n_leaf < 20:
                continue
            recent_support = float(stat.get("n_recent", np.nan))
            historical_support = float(stat.get("n_history", np.nan))
            for col, components in archetype_components.items():
                z = components["raw"]
                z_norm = components["norm"]
                period_state = components["period"]
                within_timestamp_state = components["within"]
                finite = components["finite"]
                leaf_finite = mask & finite
                if leaf_finite.sum() < 20:
                    continue
                global_var = float(np.nanmean(z_norm[finite] ** 2))
                leaf_var = float(np.nanmean(z_norm[leaf_finite] ** 2))
                if global_var <= 1e-8 or leaf_var <= 1e-8:
                    continue
                global_slope = float(np.nanmean(residual[finite] * z_norm[finite]) / global_var)
                leaf_slope = float(np.nanmean(residual[leaf_finite] * z_norm[leaf_finite]) / leaf_var)
                q25, q75 = np.nanquantile(z_norm[finite], [0.25, 0.75])
                global_hi = finite & (z_norm >= q75)
                global_lo = finite & (z_norm <= q25)
                leaf_hi = leaf_finite & (z_norm >= q75)
                leaf_lo = leaf_finite & (z_norm <= q25)
                if leaf_hi.sum() < 5 or leaf_lo.sum() < 5 or global_hi.sum() < 10 or global_lo.sum() < 10:
                    high_low_interaction = np.nan
                else:
                    global_hilo = float(np.nanmean(residual[global_hi]) - np.nanmean(residual[global_lo]))
                    leaf_hilo = float(np.nanmean(residual[leaf_hi]) - np.nanmean(residual[leaf_lo]))
                    high_low_interaction = leaf_hilo - global_hilo
                ret_finite = np.isfinite(realized_return)
                if (
                    leaf_hi.sum() >= 5
                    and leaf_lo.sum() >= 5
                    and global_hi.sum() >= 10
                    and global_lo.sum() >= 10
                    and int((ret_finite & (global_hi | global_lo)).sum()) >= 20
                ):
                    global_ret_hilo = float(
                        np.nanmean(realized_return[global_hi & ret_finite])
                        - np.nanmean(realized_return[global_lo & ret_finite])
                    )
                    leaf_ret_hilo = float(
                        np.nanmean(realized_return[leaf_hi & ret_finite])
                        - np.nanmean(realized_return[leaf_lo & ret_finite])
                    )
                    economic_effect = leaf_ret_hilo - global_ret_hilo
                else:
                    global_ret_hilo = np.nan
                    leaf_ret_hilo = np.nan
                    economic_effect = np.nan
                global_period_slope = _component_slope(period_state, finite, min_count=50)
                leaf_period_slope = _component_slope(period_state, leaf_finite, min_count=20)
                global_within_slope = _component_slope(within_timestamp_state, finite, min_count=50)
                leaf_within_slope = _component_slope(within_timestamp_state, leaf_finite, min_count=20)
                period_interaction_delta = (
                    leaf_period_slope - global_period_slope
                    if np.isfinite(leaf_period_slope) and np.isfinite(global_period_slope)
                    else np.nan
                )
                within_interaction_delta = (
                    leaf_within_slope - global_within_slope
                    if np.isfinite(leaf_within_slope) and np.isfinite(global_within_slope)
                    else np.nan
                )
                episode_stats = _episode_sign_stats(period_state, leaf_finite, finite)
                reliability = min(1.0, n_leaf / 100.0)
                interaction_slope = leaf_slope - global_slope
                hilo_term = 0.0 if not np.isfinite(high_low_interaction) else abs(float(high_low_interaction))
                period_term = 0.0 if not np.isfinite(period_interaction_delta) else abs(float(period_interaction_delta))
                within_term = 0.0 if not np.isfinite(within_interaction_delta) else abs(float(within_interaction_delta))
                stability_term = (
                    0.0
                    if not np.isfinite(episode_stats["episode_sign_stability"])
                    else 0.25 * float(episode_stats["episode_sign_stability"])
                )
                economic_term = 0.0 if not np.isfinite(economic_effect) else min(abs(float(economic_effect)), 0.10)
                score = reliability * (abs(float(interaction_slope)) + hilo_term + period_term + within_term + stability_term + economic_term)
                context_payload: dict[str, float] = {}
                for context_col in context_sample.columns:
                    context_values = pd.to_numeric(context_sample[context_col], errors="coerce").to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                    context_payload[f"context_leaf_mean__{context_col}"] = _safe_mean(context_values, leaf_finite)
                    context_payload[f"context_recent_leaf_mean__{context_col}"] = _safe_mean(
                        context_values,
                        leaf_finite & recent_sample_mask,
                    )
                    context_payload[f"context_history_leaf_mean__{context_col}"] = _safe_mean(
                        context_values,
                        leaf_finite & history_sample_mask,
                    )
                rows.append(
                    {
                        "head": head.head,
                        "strategy_id": head.strategy_id,
                        "model_kind": model_kind,
                        "model_idx": int(model_idx),
                        "tree_id": int(tree_idx),
                        "tree_idx": int(tree_idx),
                        "leaf_id": int(stat["leaf_id"]),
                        "archetype_feature": col,
                        "n_leaf": n_leaf,
                        "historical_support": historical_support,
                        "recent_support": recent_support,
                        "residual_mean_leaf": float(np.nanmean(residual[leaf_finite])),
                        "return_mean_leaf": _safe_mean(realized_return, leaf_finite),
                        "return_mean_global": _safe_mean(realized_return, finite),
                        "archetype_mean_leaf": float(np.nanmean(z[leaf_finite])),
                        "period_state_mean_leaf": _safe_mean(period_state, leaf_finite),
                        "period_state_mean_global": _safe_mean(period_state, finite),
                        "within_timestamp_state_mean_leaf": _safe_mean(within_timestamp_state, leaf_finite),
                        "within_timestamp_state_mean_global": _safe_mean(within_timestamp_state, finite),
                        "global_slope": global_slope,
                        "leaf_slope": leaf_slope,
                        "leaf_x_archetype_slope": float(interaction_slope),
                        "global_period_state_slope": global_period_slope,
                        "leaf_period_state_slope": leaf_period_slope,
                        "leaf_x_period_state_slope": period_interaction_delta,
                        "global_within_timestamp_slope": global_within_slope,
                        "leaf_within_timestamp_slope": leaf_within_slope,
                        "leaf_x_within_timestamp_slope": within_interaction_delta,
                        "high_low_residual_interaction": float(high_low_interaction)
                        if np.isfinite(high_low_interaction)
                        else np.nan,
                        "global_high_low_return_effect": global_ret_hilo,
                        "leaf_high_low_return_effect": leaf_ret_hilo,
                        "economic_effect": economic_effect,
                        "episode_count": int(episode_stats["episode_count"]),
                        "episode_sign_stability": episode_stats["episode_sign_stability"],
                        "episode_interaction_mean": episode_stats["episode_interaction_mean"],
                        "episode_interaction_std": episode_stats["episode_interaction_std"],
                        "interaction_reliability": float(reliability),
                        "interaction_score": float(score),
                        **context_payload,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("interaction_score", ascending=False).reset_index(drop=True)


def _base_models_for_head(base_bundle: dict[str, Any], head: HeadContext) -> tuple[list[Any], list[str]]:
    side = "long" if head.strategy_id.startswith("long_") else "short"
    entry = (((base_bundle.get("alpha_models") or {}).get(side) or {}).get(head.strategy_id) or {})
    model_race = entry.get("model")
    if model_race is None or not hasattr(model_race, "best_model"):
        return [], []
    best = model_race.best_model
    return list(getattr(best, "models", []) or []), list(getattr(best, "selected_features", []) or entry.get("selected_features") or [])


def _leaf_diagnostic(
    head: HeadContext,
    panel: pd.DataFrame,
    selected_x: pd.DataFrame,
    base_selected_x: pd.DataFrame,
    race: Any,
    base_bundle: dict[str, Any] | None,
    out_dir: Path,
    *,
    rank_threshold: float,
    max_rows: int,
    min_support: int,
    tree_stride: int,
    max_trees_per_model: int,
    seed: int,
    archetype_top_leaves: int,
    archetype_max_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context_cols = _pick_context_columns(panel, selected_x)
    archetype_cols = _pick_archetype_columns(panel, selected_x, max_cols=archetype_max_features)
    all_stats: list[pd.DataFrame] = []
    meta_interactions = pd.DataFrame()
    base_interactions = pd.DataFrame()
    meta_models = list(getattr(race.best_model, "models", []) or [])
    meta_stats = _leaf_stats_for_models(
        head=head,
        model_kind="meta",
        models=meta_models,
        x=selected_x,
        panel=panel,
        rank_threshold=rank_threshold,
        max_rows=max_rows,
        min_support=min_support,
        tree_stride=tree_stride,
        max_trees_per_model=max_trees_per_model,
        seed=seed,
    )
    if not meta_stats.empty:
        meta_top = _enrich_leaf_context(
            meta_stats,
            models=meta_models,
            x=selected_x,
            panel=panel,
            context_cols=context_cols,
            top_n=100,
            rank_threshold=rank_threshold,
            max_rows=max_rows,
            seed=seed,
        )
        meta_top.to_csv(out_dir / f"{head.head}_meta_leaf_instability_top.csv", index=False)
        meta_interactions = _leaf_archetype_interactions(
            meta_stats,
            head=head,
            model_kind="meta",
            models=meta_models,
            x=selected_x,
            panel=panel,
            archetype_cols=archetype_cols,
            context_cols=context_cols,
            top_leaves=archetype_top_leaves,
            rank_threshold=rank_threshold,
            max_rows=max_rows,
            seed=seed,
        )
        if not meta_interactions.empty:
            meta_interactions.head(500).to_csv(out_dir / f"{head.head}_meta_leaf_archetype_interactions.csv", index=False)
        all_stats.append(meta_stats)

    base_stats = pd.DataFrame()
    base_models: list[Any] = []
    if base_bundle is not None:
        base_models, _ = _base_models_for_head(base_bundle, head)
        if base_models and not base_selected_x.empty:
            base_stats = _leaf_stats_for_models(
                head=head,
                model_kind="base",
                models=base_models,
                x=base_selected_x,
                panel=panel,
                rank_threshold=rank_threshold,
                max_rows=max_rows,
                min_support=min_support,
                tree_stride=tree_stride,
                max_trees_per_model=max_trees_per_model,
                seed=seed + 17,
            )
            if not base_stats.empty:
                base_top = _enrich_leaf_context(
                    base_stats,
                    models=base_models,
                    x=base_selected_x,
                    panel=panel,
                    context_cols=context_cols,
                    top_n=100,
                    rank_threshold=rank_threshold,
                    max_rows=max_rows,
                    seed=seed + 17,
                )
                base_top.to_csv(out_dir / f"{head.head}_base_leaf_instability_top.csv", index=False)
                base_archetype_cols = _pick_archetype_columns(panel, base_selected_x, max_cols=archetype_max_features)
                base_interactions = _leaf_archetype_interactions(
                    base_stats,
                    head=head,
                    model_kind="base",
                    models=base_models,
                    x=base_selected_x,
                    panel=panel,
                    archetype_cols=base_archetype_cols,
                    context_cols=context_cols,
                    top_leaves=archetype_top_leaves,
                    rank_threshold=rank_threshold,
                    max_rows=max_rows,
                    seed=seed + 17,
                )
                if not base_interactions.empty:
                    base_interactions.head(500).to_csv(
                        out_dir / f"{head.head}_base_leaf_archetype_interactions.csv", index=False
                    )
                all_stats.append(base_stats)

    stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    if not stats.empty:
        stats.head(5000).to_csv(out_dir / f"{head.head}_leaf_instability_summary.csv", index=False)

    def _top_leaf_payload(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
        if frame.empty:
            return {
                f"{prefix}_top_instability_score": np.nan,
                f"{prefix}_top_tree_id": np.nan,
                f"{prefix}_top_leaf_id": np.nan,
                f"{prefix}_top_occupancy_shift": np.nan,
                f"{prefix}_top_outcome_shift": np.nan,
                f"{prefix}_top_calibration_shift": np.nan,
                f"{prefix}_top_recent_support": np.nan,
                f"{prefix}_top_history_support": np.nan,
            }
        top = frame.sort_values("instability_score", ascending=False).iloc[0]
        return {
            f"{prefix}_top_instability_score": float(top.get("instability_score", np.nan)),
            f"{prefix}_top_tree_id": int(top.get("tree_idx", -1)),
            f"{prefix}_top_leaf_id": int(top.get("leaf_id", -1)),
            f"{prefix}_top_occupancy_shift": float(top.get("occupancy_shift", np.nan)),
            f"{prefix}_top_outcome_shift": float(top.get("outcome_shift", np.nan)),
            f"{prefix}_top_calibration_shift": float(top.get("calibration_shift", np.nan)),
            f"{prefix}_top_recent_support": float(top.get("n_recent", np.nan)),
            f"{prefix}_top_history_support": float(top.get("n_history", np.nan)),
        }

    def _top_interaction_payload(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
        if frame.empty:
            return {
                f"{prefix}_leaf_archetype_interaction_rows": 0,
                f"{prefix}_top_interaction_feature": "",
                f"{prefix}_top_interaction_tree_id": np.nan,
                f"{prefix}_top_interaction_leaf_id": np.nan,
                f"{prefix}_top_interaction_score": np.nan,
                f"{prefix}_top_interaction_delta": np.nan,
                f"{prefix}_top_period_interaction_delta": np.nan,
                f"{prefix}_top_within_interaction_delta": np.nan,
                f"{prefix}_top_episode_sign_stability": np.nan,
                f"{prefix}_top_economic_effect": np.nan,
            }
        top = frame.sort_values("interaction_score", ascending=False).iloc[0]
        return {
            f"{prefix}_leaf_archetype_interaction_rows": int(len(frame)),
            f"{prefix}_top_interaction_feature": str(top.get("archetype_feature", "")),
            f"{prefix}_top_interaction_tree_id": int(top.get("tree_id", -1)),
            f"{prefix}_top_interaction_leaf_id": int(top.get("leaf_id", -1)),
            f"{prefix}_top_interaction_score": float(top.get("interaction_score", np.nan)),
            f"{prefix}_top_interaction_delta": float(top.get("leaf_x_archetype_slope", np.nan)),
            f"{prefix}_top_period_interaction_delta": float(top.get("leaf_x_period_state_slope", np.nan)),
            f"{prefix}_top_within_interaction_delta": float(top.get("leaf_x_within_timestamp_slope", np.nan)),
            f"{prefix}_top_episode_sign_stability": float(top.get("episode_sign_stability", np.nan)),
            f"{prefix}_top_economic_effect": float(top.get("economic_effect", np.nan)),
        }

    summary = {
        "head": head.head,
        "strategy_id": head.strategy_id,
        "meta_leaf_rows": int(len(meta_stats)),
        "base_leaf_rows": int(len(base_stats)),
        "meta_model_count": int(len(meta_models)),
        "base_model_count": int(len(base_models)),
        "leaf_max_rows": int(max_rows),
        "leaf_min_support": int(min_support),
        "leaf_tree_stride": int(tree_stride),
        "leaf_max_trees_per_model": int(max_trees_per_model),
        "context_columns": context_cols,
        "archetype_columns": archetype_cols,
        **_top_leaf_payload(meta_stats, "meta"),
        **_top_leaf_payload(base_stats, "base"),
        **_top_interaction_payload(meta_interactions, "meta"),
        **_top_interaction_payload(base_interactions, "base"),
    }
    return stats, summary


def _summarise_report(
    out_dir: Path,
    failure_rows: list[dict[str, Any]],
    adversarial_rows: list[dict[str, Any]],
    residualized_adversarial_rows: list[dict[str, Any]],
    leaf_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Meta Recent-Failure Diagnostics")
    lines.append("")
    lines.append("## High-confidence failure classifier")
    if failure_rows:
        for row in failure_rows:
            auc = row.get("auc_mean")
            interpretation = "not enough signal"
            if isinstance(auc, (int, float)) and np.isfinite(auc):
                if auc >= 0.70:
                    interpretation = "learnable missing regime interaction likely"
                elif auc >= 0.60:
                    interpretation = "weak but usable filtering/sizing signal"
                else:
                    interpretation = "little learnable failure structure in this feature set"
            lines.append(
                f"- {row['head']}: AUC={auc:.3f} rows={row.get('rows')} "
                f"fail_rate={row.get('positive_rate', np.nan):.3f} -> {interpretation}"
            )
    lines.append("")
    lines.append("## Adversarial validation")
    global_adv_rows = [row for row in adversarial_rows if row.get("diagnostic") == "adversarial_global_bad_weeks"]
    for row in global_adv_rows:
        if row.get("diagnostic") != "adversarial_global_bad_weeks":
            continue
        auc = row.get("auc_mean")
        interpretation = "feature shift inconclusive"
        if isinstance(auc, (int, float)) and np.isfinite(auc):
            interpretation = "recent/bad weeks are feature-distribution different" if auc >= 0.70 else "no strong distribution break"
        lines.append(
            f"- {row['head']}: global bad-week AUC={auc:.3f} bad_rows={row.get('bad_rows')} "
            f"normal_rows={row.get('normal_rows')} -> {interpretation}"
        )
    lines.append("")
    local_adv_rows = [
        row
        for row in adversarial_rows
        if row.get("diagnostic") == "adversarial_local_bad_week" and not row.get("reason")
    ]
    if local_adv_rows:
        local_df = pd.DataFrame(local_adv_rows)
        sort_cols = [c for c in ("auc_mean", "bad_rows") if c in local_df.columns]
        if sort_cols:
            local_df = local_df.sort_values(sort_cols, ascending=False)
        lines.append("## Local adversarial validation")
        lines.append("")
        lines.append(
            local_df[
                [
                    c
                    for c in (
                        "head",
                        "bad_week",
                        "baseline_weeks",
                        "auc_mean",
                        "bad_rows",
                        "normal_rows",
                        "baseline_match_method",
                        "baseline_match_score_mean",
                    )
                    if c in local_df.columns
                ]
            ]
            .head(20)
            .to_markdown(index=False, floatfmt=".3f")
        )
        lines.append("")
    if residualized_adversarial_rows:
        lines.append("## Residualized adversarial validation")
        for row in residualized_adversarial_rows:
            if row.get("diagnostic") != "residualized_adversarial_global_bad_weeks":
                continue
            lines.append(
                f"- {row['head']}: raw_auc={row.get('raw_auc', np.nan):.3f} "
                f"nuisance_auc={row.get('nuisance_auc', np.nan):.3f} "
                f"residualized_auc={row.get('residualized_auc', np.nan):.3f} "
                f"raw_minus_nuisance={row.get('raw_minus_nuisance_auc', np.nan):.3f} "
                f"incremental_beyond_nuisance={row.get('incremental_auc_beyond_nuisance', np.nan):.3f}"
            )
        lines.append("")
        local_resid_rows = [
            row
            for row in residualized_adversarial_rows
            if row.get("diagnostic") == "residualized_adversarial_local_bad_week" and not row.get("reason")
        ]
        if local_resid_rows:
            local_resid_df = pd.DataFrame(local_resid_rows)
            sort_cols = [c for c in ("incremental_auc_beyond_nuisance", "residualized_auc") if c in local_resid_df.columns]
            if sort_cols:
                local_resid_df = local_resid_df.sort_values(sort_cols, ascending=False)
            lines.append("## Local Residualized Adversarial Validation")
            lines.append("")
            lines.append(
                local_resid_df[
                    [
                        c
                        for c in (
                            "head",
                            "bad_week",
                            "baseline_weeks",
                            "raw_auc",
                            "nuisance_auc",
                            "residualized_auc",
                            "incremental_auc_beyond_nuisance",
                            "bad_rows",
                            "normal_rows",
                            "baseline_match_method",
                            "baseline_match_score_mean",
                        )
                        if c in local_resid_df.columns
                    ]
                ]
                .head(20)
                .to_markdown(index=False, floatfmt=".3f")
            )
            lines.append("")
    lines.append("## Leaf instability")
    for row in leaf_rows:
        lines.append(
            f"- {row['head']}: meta_leaf_clusters={row.get('meta_leaf_rows')} "
            f"base_leaf_clusters={row.get('base_leaf_rows')} "
            f"models(meta/base)={row.get('meta_model_count')}/{row.get('base_model_count')}"
        )
    lines.append("")
    if leaf_rows:
        leaf_df = pd.DataFrame(leaf_rows)
        top_shift_cols = [
            c
            for c in (
                "head",
                "meta_top_instability_score",
                "meta_top_occupancy_shift",
                "meta_top_outcome_shift",
                "meta_top_calibration_shift",
                "meta_top_recent_support",
                "base_top_instability_score",
                "base_top_occupancy_shift",
                "base_top_outcome_shift",
                "base_top_calibration_shift",
                "base_top_recent_support",
            )
            if c in leaf_df.columns
        ]
        if len(top_shift_cols) > 1:
            lines.append("## Top Base/Meta Leaf Shifts")
            lines.append("")
            lines.append(leaf_df[top_shift_cols].to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
        interaction_cols = [
            c
            for c in (
                "head",
                "meta_top_interaction_feature",
                "meta_top_interaction_score",
                "meta_top_interaction_delta",
                "meta_top_period_interaction_delta",
                "meta_top_within_interaction_delta",
                "meta_top_episode_sign_stability",
                "meta_top_economic_effect",
                "base_top_interaction_feature",
                "base_top_interaction_score",
                "base_top_interaction_delta",
                "base_top_period_interaction_delta",
                "base_top_within_interaction_delta",
                "base_top_episode_sign_stability",
                "base_top_economic_effect",
            )
            if c in leaf_df.columns
        ]
        if len(interaction_cols) > 1:
            lines.append("## Top Residual x Archetype Leaf Interactions")
            lines.append("")
            lines.append(leaf_df[interaction_cols].to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
    lines.append("## Feature reconstruction coverage")
    for row in coverage_rows:
        lines.append(
            f"- {row['head']} {row['matrix_kind']}: selected={row.get('selected_features')} "
            f"non_missing_sources={row.get('features_non_missing_source')} "
            f"missing_sources={row.get('features_missing_source')} "
            f"mean_finite_fraction={row.get('mean_finite_fraction'):.3f}"
        )
    lines.append("")
    lines.append("See CSV files in this directory for feature importances, weekly metrics, and top leaf contexts.")
    (out_dir / "diagnostic_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    report_dir = Path(args.report_dir)
    feature_dir = Path(args.feature_dir)
    out_dir = _ensure_dir(Path(args.output_dir))
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    regime_context = Path(args.regime_context) if args.regime_context else _latest_regime_context(Path(args.regime_root))

    _log(f"loading meta state from {meta_artifact_dir}")
    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    if args.only_head:
        wanted = {str(h).strip() for h in args.only_head if str(h).strip()}
        heads = [
            h
            for h in heads
            if h.head in wanted
            or h.strategy_id in wanted
            or h.meta_key in wanted
            or any(token in h.head or token in h.strategy_id or token in h.meta_key for token in wanted)
        ]
    _log(f"found {len(heads)} meta heads")

    base_bundle: dict[str, Any] | None = None
    if (not args.skip_classifiers) or (not args.skip_leaves and not args.skip_base_leaves):
        base_path = baseline_artifact_dir / "base_models_intermediate.pkl"
        if base_path.exists():
            _log(f"loading base bundle from {base_path}")
            with base_path.open("rb") as fh:
                base_bundle = pickle.load(fh)

    symbol_columns = _feature_store_union(feature_dir)
    _log(f"feature store symbols={len(symbol_columns)}")

    failure_rows: list[dict[str, Any]] = []
    adversarial_rows: list[dict[str, Any]] = []
    residualized_adversarial_rows: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    all_failure_imps: list[pd.DataFrame] = []
    all_adversarial_imps: list[pd.DataFrame] = []
    all_residualized_adversarial_imps: list[pd.DataFrame] = []

    for head in heads:
        _log(f"processing head={head.head}")
        panel = pd.read_parquet(head.meta_oof_path)
        panel = _normalise_keys(panel)
        panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
        race = meta_models[head.meta_key]
        selected_x, coverage, coverage_summary = _assemble_selected_matrix(
            panel=panel,
            race=race,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
        coverage.insert(0, "head", head.head)
        coverage.to_csv(out_dir / f"{head.head}_meta_selected_feature_coverage.csv", index=False)
        coverage_summary.update({"head": head.head, "matrix_kind": "meta_selected"})
        coverage_rows.append(coverage_summary)

        base_selected_x = pd.DataFrame(index=panel.index)
        if base_bundle is not None:
            _, base_features = _base_models_for_head(base_bundle, head)
            if base_features:
                fake_race = type("FakeRace", (), {})()
                fake_best = type("FakeBest", (), {})()
                fake_best.selected_features = list(base_features)
                fake_best.get_training_meta_features = lambda: pd.DataFrame(index=panel.index)
                fake_best.model_effectiveness_history_defaults_ = {}
                fake_best.feature_stats_train = {}
                fake_race.best_model = fake_best
                base_selected_x, base_cov, base_cov_summary = _assemble_selected_matrix(
                    panel=panel,
                    race=fake_race,
                    feature_dir=feature_dir,
                    transform_cache=transform_cache,
                    symbol_columns=symbol_columns,
                )
                base_cov.insert(0, "head", head.head)
                base_cov.to_csv(out_dir / f"{head.head}_base_selected_feature_coverage.csv", index=False)
                base_cov_summary.update({"head": head.head, "matrix_kind": "base_selected"})
                coverage_rows.append(base_cov_summary)

        if not args.skip_classifiers:
            export_x = _known_export_features(panel)
            regime_x = _read_regime_features(regime_context, panel[["timestamp", "symbol"]], args.max_regime_columns)
            selected_parts = [selected_x]
            if not base_selected_x.empty:
                base_extra = [c for c in base_selected_x.columns if c not in selected_x.columns]
                if base_extra:
                    selected_parts.append(base_selected_x[base_extra])
            classifier_selected_x = pd.concat(selected_parts, axis=1, copy=False)
            candidate_x = _merge_feature_candidates(classifier_selected_x, export_x, regime_x)
            feature_contract = _candidate_feature_contract(candidate_x)
            feature_contract.insert(0, "head", head.head)
            feature_contract.insert(1, "strategy_id", head.strategy_id)
            feature_contract.to_csv(out_dir / f"{head.head}_candidate_feature_contract.csv", index=False)
            _log(f"head={head.head} candidate_features={candidate_x.shape[1]} rows={len(candidate_x)}")

            fail_summary, fail_imp = _high_conf_failure_diagnostic(
                head,
                panel,
                candidate_x,
                out_dir,
                rank_threshold=args.rank_threshold,
                max_rows=args.classifier_max_rows,
                seed=args.seed,
            )
            failure_rows.append(fail_summary)
            if not fail_imp.empty:
                all_failure_imps.append(fail_imp)

            adv_summaries, adv_imp, local_imp = _adversarial_diagnostics(
                head,
                panel,
                candidate_x,
                out_dir,
                rank_threshold=args.rank_threshold,
                recent_weeks=args.recent_weeks,
                min_week_rows=args.min_week_rows,
                max_rows=args.adversarial_max_rows,
                seed=args.seed,
            )
            adversarial_rows.extend(adv_summaries)
            if not adv_imp.empty:
                all_adversarial_imps.append(adv_imp)
            if not local_imp.empty:
                all_adversarial_imps.append(local_imp)
            if not args.skip_residualized_adversarial:
                resid_summaries, resid_imp = _residualized_adversarial_diagnostics(
                    head,
                    panel,
                    candidate_x,
                    out_dir,
                    rank_threshold=args.rank_threshold,
                    recent_weeks=args.recent_weeks,
                    min_week_rows=args.min_week_rows,
                    max_rows=args.residualized_adversarial_max_rows,
                    max_residual_features=args.residualized_max_features,
                    seed=args.seed,
                )
                residualized_adversarial_rows.extend(resid_summaries)
                if not resid_imp.empty:
                    all_residualized_adversarial_imps.append(resid_imp)
        else:
            _log(f"head={head.head} classifier/adversarial diagnostics skipped")

        if args.skip_leaves:
            leaf_summary = {
                "head": head.head,
                "strategy_id": head.strategy_id,
                "meta_leaf_rows": 0,
                "base_leaf_rows": 0,
                "meta_model_count": 0,
                "base_model_count": 0,
                "leaf_skipped": True,
            }
        else:
            _, leaf_summary = _leaf_diagnostic(
                head,
                panel,
                selected_x,
                base_selected_x,
                race,
                base_bundle,
                out_dir,
                rank_threshold=args.rank_threshold,
                max_rows=args.leaf_max_rows,
                min_support=args.leaf_min_support,
                tree_stride=args.leaf_tree_stride,
                max_trees_per_model=args.leaf_max_trees_per_model,
                seed=args.seed,
                archetype_top_leaves=args.leaf_archetype_top_leaves,
                archetype_max_features=args.leaf_archetype_max_features,
            )
        leaf_rows.append(leaf_summary)

    pd.DataFrame(failure_rows).to_csv(out_dir / "high_conf_failure_summary.csv", index=False)
    pd.DataFrame(adversarial_rows).to_csv(out_dir / "adversarial_validation_summary.csv", index=False)
    pd.DataFrame(residualized_adversarial_rows).to_csv(
        out_dir / "adversarial_residualized_validation_summary.csv", index=False
    )
    pd.DataFrame(leaf_rows).to_csv(out_dir / "leaf_instability_manifest.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "feature_reconstruction_coverage_summary.csv", index=False)
    if all_failure_imps:
        pd.concat(all_failure_imps, ignore_index=True).to_csv(out_dir / "high_conf_failure_importance_all.csv", index=False)
    if all_adversarial_imps:
        pd.concat(all_adversarial_imps, ignore_index=True).to_csv(out_dir / "adversarial_importance_all.csv", index=False)
    if all_residualized_adversarial_imps:
        pd.concat(all_residualized_adversarial_imps, ignore_index=True).to_csv(
            out_dir / "adversarial_residualized_importance_all.csv", index=False
        )
    _write_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "meta_artifact_dir": str(meta_artifact_dir),
            "baseline_artifact_dir": str(baseline_artifact_dir),
            "report_dir": str(report_dir),
            "feature_dir": str(feature_dir),
            "transform_cache": str(transform_cache) if transform_cache else "",
            "regime_context": str(regime_context) if regime_context else "",
        },
    )
    _summarise_report(out_dir, failure_rows, adversarial_rows, residualized_adversarial_rows, leaf_rows, coverage_rows)
    _log(f"wrote diagnostics to {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument(
        "--transform-cache",
        default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet",
    )
    parser.add_argument("--regime-root", default="data_perp/artifacts/unsupervised_regime_learning_poc")
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/meta_recent_failure_diagnostics_20260622")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--recent-weeks", type=int, default=8)
    parser.add_argument("--min-week-rows", type=int, default=30)
    parser.add_argument("--classifier-max-rows", type=int, default=80000)
    parser.add_argument("--adversarial-max-rows", type=int, default=70000)
    parser.add_argument("--residualized-adversarial-max-rows", type=int, default=30000)
    parser.add_argument("--residualized-max-features", type=int, default=250)
    parser.add_argument("--leaf-max-rows", type=int, default=18000)
    parser.add_argument("--leaf-min-support", type=int, default=120)
    parser.add_argument("--leaf-tree-stride", type=int, default=2)
    parser.add_argument("--leaf-max-trees-per-model", type=int, default=500)
    parser.add_argument("--leaf-archetype-top-leaves", type=int, default=80)
    parser.add_argument("--leaf-archetype-max-features", type=int, default=12)
    parser.add_argument("--max-regime-columns", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--only-head", action="append", default=[])
    parser.add_argument("--skip-classifiers", action="store_true")
    parser.add_argument("--skip-residualized-adversarial", action="store_true")
    parser.add_argument("--skip-leaves", action="store_true")
    parser.add_argument("--skip-base-leaves", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
