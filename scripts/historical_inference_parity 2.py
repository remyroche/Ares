#!/usr/bin/env python3
"""Replay inference from historical local data and compare to training artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import exchange_data_component  # noqa: E402
from extreme_price_movements.inference.config import get_inference_defaults  # noqa: E402
from extreme_price_movements.inference.config import load_inference_config  # noqa: E402
from extreme_price_movements.inference.config import load_trained_symbol_universe  # noqa: E402
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    _required_tail_warmup_hours,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    load_or_compute_features,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator  # noqa: E402
from extreme_price_movements.inference.parity import (  # noqa: E402
    calibrated_score_and_threshold,
    strategy_core_id,
    strategy_id_matches,
)
from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    policy_rank_pct_from_sorted_scores,
    strategy_rank_reference_aliases,
)
from extreme_price_movements.inference.run_inference import (  # noqa: E402
    _lgbm_mask_required_feature_keys,
    _load_lgbm_strategy_mask_rows,
    _select_candidates_and_load_features,
)
from extreme_price_movements.inference.training_live_parity_contract import (  # noqa: E402
    load_training_live_parity_contract,
)
from extreme_price_movements.model_loader import load_full_state  # noqa: E402
from extreme_price_movements.pipeline_steps import (  # noqa: E402
    _load_external_kraken_spot_panels,
)
from extreme_price_movements.simple_position_sizer import load_calibration_curves  # noqa: E402
from extreme_price_movements.utils import tprint  # noqa: E402
from scripts.replay_live_signal_predictions import (  # noqa: E402
    _load_panel,
    _local_quote_symbols,
    _normalise_symbol,
)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _runtime_only_feature_key(feature: str) -> bool:
    key = str(feature)
    return bool(
        key == "G_VOL"
        or key == "barrier_pct"
        or key.startswith("ret1h_G_VOL_")
    )


def _requires_benchmark_residual(required_keys: set[str]) -> bool:
    return any(str(key).endswith("_bench_resid") for key in required_keys or set())


def _benchmark_context_symbols(
    required_keys: set[str],
    *,
    market_mode: str,
    live_quote_currency: str = "USDC",
) -> list[str]:
    if not _requires_benchmark_residual(required_keys):
        return []
    quote = str(live_quote_currency or "USDC").upper()
    if str(market_mode).lower() == "perps":
        return ["BTC/USD:USD"]
    return [f"BTC/{quote}", "BTC/USDT", "BTC/USD"]


def _add_required_context_symbols(
    symbols: list[str],
    required_keys: set[str],
    *,
    market_mode: str,
    live_quote_currency: str = "USDC",
) -> list[str]:
    if not symbols:
        return symbols
    out = list(dict.fromkeys(_normalise_symbol(sym) for sym in symbols))
    present = set(out)
    for sym in _benchmark_context_symbols(
        required_keys,
        market_mode=market_mode,
        live_quote_currency=live_quote_currency,
    ):
        norm = _normalise_symbol(sym)
        if norm not in present:
            out.append(norm)
            present.add(norm)
    return sorted(out)


def _meta_oof_path(data_root: Path, run_id: str, strategy: str | None) -> Path:
    root = data_root / "artifacts" / run_id / "meta_oof"
    if strategy:
        path = root / f"meta_oof_{strategy}_clf.parquet"
        if path.exists():
            return path
    paths = sorted(root.glob("meta_oof_*_clf.parquet"))
    if not paths:
        raise FileNotFoundError(f"No meta_oof parquet files found in {root}")
    return paths[0]


def _strategy_from_meta_oof_path(path: Path) -> str:
    name = path.stem
    if name.startswith("meta_oof_"):
        name = name[len("meta_oof_") :]
    if name.endswith("_clf"):
        name = name[: -len("_clf")]
    return name


def _historical_market_data_root(data_root: Path, market_mode: str) -> Path:
    exchange_id = (
        os.environ.get("EPM_EXCHANGE")
        or os.environ.get("EXCHANGE_NAME")
        or os.environ.get("PRIMARY_EXCHANGE")
        or ""
    )
    component = exchange_data_component(exchange_id, market_mode)
    exchange_root = Path(data_root) / "exchanges" / component
    if (exchange_root / "ohlcv").exists():
        return exchange_root
    return Path(data_root)


def _sample_oof_rows(
    path: Path,
    *,
    sample_rows: int,
    samples_per_symbol: int,
    warmup_rows: int,
    min_timestamp: str | None,
) -> pd.DataFrame:
    tprint(f"Loading OOF rows from {path}")
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol"]).sort_values("timestamp")
    tprint(
        "Loaded OOF rows: "
        f"rows={len(df):,} symbols={df['symbol'].nunique():,} "
        f"ts={df['timestamp'].min()}..{df['timestamp'].max()}"
    )
    if min_timestamp:
        start = pd.Timestamp(min_timestamp)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        df = df[df["timestamp"] >= start]
        tprint(
            "Applied min timestamp filter: "
            f"min_timestamp={start} rows={len(df):,} symbols={df['symbol'].nunique():,}"
        )
    if df.empty:
        return df
    if samples_per_symbol and samples_per_symbol > 0:
        parts = []
        for _, group in df.groupby("symbol", sort=True):
            group = group.sort_values("timestamp")
            if warmup_rows and warmup_rows > 0:
                group = group.iloc[int(warmup_rows) :]
            if group.empty:
                continue
            n = min(int(samples_per_symbol), len(group))
            idx = np.linspace(0, len(group) - 1, n).round().astype(int)
            parts.append(group.iloc[idx])
        if not parts:
            return df.iloc[0:0].copy()
        sampled = (
            pd.concat(parts, axis=0)
            .drop_duplicates(["timestamp", "symbol"])
            .sort_values(["symbol", "timestamp"])
            .reset_index(drop=True)
        )
        if sample_rows and sample_rows > 0 and len(sampled) > int(sample_rows):
            idx = np.linspace(0, len(sampled) - 1, int(sample_rows)).round().astype(int)
            sampled = sampled.iloc[idx].drop_duplicates(["timestamp", "symbol"]).reset_index(drop=True)
        tprint(
            "Selected sampled OOF rows by symbol: "
            f"rows={len(sampled):,} symbols={sampled['symbol'].nunique():,} "
            f"samples_per_symbol={samples_per_symbol} global_cap={sample_rows if sample_rows > 0 else 'none'}"
        )
        return sampled
    # Keep rows spread across the latest available OOF segment rather than
    # adjacent rows from one burst.
    tail = df.tail(max(int(sample_rows) * 20, int(sample_rows)))
    idx = np.linspace(0, len(tail) - 1, min(int(sample_rows), len(tail))).round().astype(int)
    sampled = tail.iloc[idx].drop_duplicates(["timestamp", "symbol"]).reset_index(drop=True)
    tprint(
        "Selected sampled OOF rows from tail: "
        f"rows={len(sampled):,} symbols={sampled['symbol'].nunique():,} "
        f"global_cap={sample_rows}"
    )
    return sampled


def _safe_strategy_filename(strategy_id: str) -> str:
    sid = str(strategy_id or "").strip()
    return "".join(ch if ch.isalnum() or ch in "_.=-" else "_" for ch in sid) or "unknown_strategy"


def _policy_rank_reference_path(
    data_root: Path,
    run_id: str,
    strategy_id: str,
    rank_reference_dir: Path | None = None,
) -> Path:
    root = (
        Path(rank_reference_dir)
        if rank_reference_dir is not None
        else data_root / "artifacts" / run_id / "simple_policy_optimiser" / "rank_reference"
    )
    for alias in strategy_rank_reference_aliases(strategy_id):
        path = root / f"{_safe_strategy_filename(alias)}.parquet"
        if path.exists():
            return path
    matches = sorted(root.glob(f"*{strategy_core_id(strategy_id)}*.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No policy rank reference found for {strategy_id} in {root}")


def _sample_policy_rank_reference_rows(
    data_root: Path,
    run_id: str,
    strategy_id: str,
    *,
    sample_rows: int,
    min_timestamp: str | None,
    rank_reference_dir: Path | None = None,
) -> pd.DataFrame:
    path = _policy_rank_reference_path(
        data_root,
        run_id,
        strategy_id,
        rank_reference_dir=rank_reference_dir,
    )
    tprint(f"Loading policy rank-reference rows from {path}")
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    if min_timestamp:
        start = pd.Timestamp(min_timestamp)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        df = df[df["timestamp"] >= start]
    if sample_rows and sample_rows > 0 and len(df) > int(sample_rows):
        # Keep dense timestamp batches so this exercises live cross-sectional
        # ranking semantics instead of scoring one symbol per timestamp.
        chosen_ts: list[pd.Timestamp] = []
        total = 0
        for ts, count in df.groupby("timestamp", sort=True).size().items():
            chosen_ts.append(ts)
            total += int(count)
            if total >= int(sample_rows):
                break
        df = df[df["timestamp"].isin(chosen_ts)].copy()
        if len(df) > int(sample_rows):
            df = df.sort_values(["timestamp", "symbol"]).head(int(sample_rows))
    tprint(
        "Selected policy rank-reference rows: "
        f"rows={len(df):,} symbols={df['symbol'].nunique():,} "
        f"timestamps={df['timestamp'].nunique():,} global_cap={sample_rows if sample_rows > 0 else 'none'}"
    )
    return df.reset_index(drop=True)


def _sample_policy_candidate_rows(
    data_root: Path,
    run_id: str,
    strategy_id: str,
    *,
    sample_rows: int,
    min_timestamp: str | None,
    candidate_path: Path | None = None,
) -> pd.DataFrame:
    path = (
        Path(candidate_path)
        if candidate_path is not None
        else data_root
        / "artifacts"
        / run_id
        / "simple_policy_optimiser"
        / "simple_policy_candidates.parquet"
    )
    tprint(f"Loading simple-policy candidate rows from {path}")
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    allowed = {str(strategy_id), strategy_core_id(str(strategy_id))}
    strategy_col = df.get("strategy_id")
    if strategy_col is None:
        raise ValueError(f"Policy candidate file is missing strategy_id: {path}")
    strategy_query = str(strategy_id)
    mask = strategy_col.astype(str).map(
        lambda value: strategy_id_matches(value, allowed)
        or value == strategy_query
        or value.startswith(f"{strategy_query}_")
    )
    df = df[mask].copy()
    if min_timestamp:
        start = pd.Timestamp(min_timestamp)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        df = df[df["timestamp"] >= start]
    if sample_rows and sample_rows > 0 and len(df) > int(sample_rows):
        df = df.tail(int(sample_rows)).copy()
    tprint(
        "Selected simple-policy candidate rows: "
        f"rows={len(df):,} symbols={df['symbol'].nunique():,} "
        f"timestamps={df['timestamp'].nunique():,} global_cap={sample_rows if sample_rows > 0 else 'none'}"
    )
    return df.reset_index(drop=True)


def _matches_allowed(model_key: str, allowed: set[str] | None) -> bool:
    if allowed is None:
        return True
    key = str(model_key or "")
    if strategy_id_matches(key, allowed):
        return True
    return any(
        key == allowed_key
        or key.startswith(f"{allowed_key}_")
        or allowed_key.startswith(f"{key}_")
        for allowed_key in allowed
    )


def _actual_estimator_feature_names(item: Any) -> set[str]:
    names: set[str] = set()
    seen: set[int] = set()

    def visit(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 5 or id(obj) in seen:
            return
        seen.add(id(obj))
        for attr in ("feature_name_", "feature_names_in_"):
            vals = getattr(obj, attr, None)
            if vals is not None:
                try:
                    names.update(map(str, vals))
                except Exception:
                    pass
        booster = getattr(obj, "booster_", None)
        if booster is not None:
            try:
                names.update(map(str, booster.feature_name()))
            except Exception:
                pass
        for attr in ("best_model", "model", "clf", "estimator", "base_model"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None:
                visit(child, depth + 1)
        for attr in ("models", "estimators_", "calibrated_classifiers_"):
            try:
                children = getattr(obj, attr, None)
            except Exception:
                children = None
            if isinstance(children, dict):
                for child in children.values():
                    visit(child, depth + 1)
            elif isinstance(children, (list, tuple)):
                for child in children:
                    visit(child, depth + 1)

    visit(item)
    return names


def _effective_selected_feature_names(item: Any) -> set[str]:
    inner = getattr(item, "best_model", item)
    selected = [str(c) for c in (getattr(inner, "selected_features", []) or [])]
    if not selected:
        return set()
    input_features = [
        str(c) for c in (getattr(inner, "input_feature_names", []) or [])
    ]
    if len(input_features) == len(selected) and input_features != selected:
        return set(input_features)
    return set(selected)


def _legacy_feature_columns_for_state(
    state: dict[str, Any],
    strategy_id: str | None = None,
) -> set[str]:
    allowed = None
    if strategy_id:
        allowed = {str(strategy_id), strategy_core_id(str(strategy_id))}
    keys: set[str] = set()
    bundle = state.get("bundle", {}) if isinstance(state.get("bundle"), dict) else {}
    for family in ("alpha_models", "meta_models"):
        obj = bundle.get(family, {})
        if not isinstance(obj, dict):
            continue
        stack = [
            (str(model_key), model_value)
            for model_key, model_value in obj.items()
            if _matches_allowed(str(model_key), allowed)
        ]
        while stack:
            path, item = stack.pop()
            if isinstance(item, dict):
                for candidate in (
                    item.get("feat_cols"),
                    item.get("feature_columns"),
                    item.get("columns"),
                ):
                    if candidate:
                        keys.update(map(str, candidate))
                for nested_key, nested_value in item.items():
                    nested_path = f"{path}_{nested_key}"
                    if path in {"long", "short"}:
                        keep_nested = _matches_allowed(nested_path, allowed)
                    else:
                        keep_nested = _matches_allowed(path, allowed) or _matches_allowed(
                            nested_path,
                            allowed,
                        )
                    if keep_nested:
                        stack.append((nested_path, nested_value))
            else:
                effective_selected = _effective_selected_feature_names(item)
                if effective_selected:
                    keys.update(effective_selected)
                else:
                    for attr in ("feature_columns", "feature_names_in_"):
                        vals = getattr(item, attr, None)
                        if vals is not None:
                            keys.update(map(str, vals))
                actual_names = _actual_estimator_feature_names(item)
                if actual_names:
                    keys.update(actual_names)
    return keys


def _feature_columns_for_state(
    state: dict[str, Any],
    strategy_id: str | None = None,
) -> set[str]:
    """Return decision-used feature keys for replay.

    Live inference already knows how to scope a deployed head to the raw
    features used by its base/meta models.  Historical replay should use that
    same resolver first; the older recursive collector is intentionally kept as
    a fallback only, because it can pull in the full union feature contract and
    force unnecessary historical feature materialization.
    """

    allowed = None
    if strategy_id:
        allowed = [str(strategy_id), strategy_core_id(str(strategy_id))]
    legacy_keys = _legacy_feature_columns_for_state(state, strategy_id)
    try:
        live_keys = set(
            get_inference_required_feature_keys(
                state,
                accepted_strategies=allowed,
            )
        )
    except Exception as exc:
        tprint(
            "Live decision feature resolver failed; falling back to legacy "
            f"state scan: {exc}"
        )
        live_keys = set()
    live_keys = {str(k) for k in live_keys if str(k)}
    if live_keys:
        dropped = len(legacy_keys - live_keys)
        added = len(live_keys - legacy_keys)
        tprint(
            "Using live decision feature resolver: "
            f"keys={len(live_keys):,} legacy_union={len(legacy_keys):,} "
            f"legacy_only_dropped={dropped:,} live_only_added={added:,}"
        )
        return live_keys
    tprint(
        "Using legacy replay feature resolver: "
        f"keys={len(legacy_keys):,}"
    )
    return legacy_keys


def _filter_lgbm_mask_rows_for_strategy(
    rows: Any,
    strategy_id: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, dict):
        return {}
    allowed = {str(strategy_id), strategy_core_id(str(strategy_id))}
    out: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        if not isinstance(row, dict):
            continue
        candidates = {
            str(key),
            str(row.get("strategy_id") or ""),
            str(row.get("strategy") or ""),
            str(row.get("selected_strategy") or ""),
            str(row.get("strategy_for_inference") or ""),
        }
        if any(candidate and strategy_id_matches(candidate, allowed) for candidate in candidates):
            out[str(key)] = row
    return out


def _load_reference_feature_rows(
    data_root: Path,
    run_id: str,
    samples: pd.DataFrame,
    feature_keys: set[str],
) -> dict[tuple[str, pd.Timestamp], pd.Series]:
    out: dict[tuple[str, pd.Timestamp], pd.Series] = {}
    features_root = data_root / "features" / run_id
    groups = list(samples.groupby("symbol"))
    tprint(
        "Loading stored training feature rows: "
        f"symbols={len(groups):,} requested_features={len(feature_keys):,}"
    )
    for idx, (symbol, group) in enumerate(groups, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(groups):
            tprint(f"  reference feature rows progress: {idx}/{len(groups)} symbols")
        norm_symbol = _normalise_symbol(symbol)
        path_keys = [
            norm_symbol.replace("/", "_"),
            norm_symbol.replace("/", "_").replace(":", "_"),
        ]
        path = next(
            (features_root / f"symbol={key}.parquet" for key in path_keys if (features_root / f"symbol={key}.parquet").exists()),
            features_root / f"symbol={path_keys[0]}.parquet",
        )
        if not path.exists():
            continue
        try:
            cols = sorted(c for c in feature_keys if c)
            df = pd.read_parquet(path, columns=cols)
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in df.columns if str(c) in feature_keys]
            df = df[keep]
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[pd.notna(df.index)]
        for ts in group["timestamp"]:
            ts = pd.Timestamp(ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            if ts in df.index:
                out[(norm_symbol, ts)] = df.loc[ts]
    return out


def _build_runtime_cfg(
    *,
    data_root: Path,
    artifact_data_root: Path,
    run_id: str,
    market_mode: str,
    state: dict[str, Any],
    feature_source_run_id: str | None = None,
    disable_rolling_cache: bool = False,
    disable_offline_cache: bool = False,
    offline_allowed_periods=None,
) -> dict[str, Any]:
    try:
        feature_cfg = load_inference_config(
            data_root=str(artifact_data_root),
            run_id=run_id,
            market_mode=market_mode,
        )
    except Exception:
        feature_cfg = dict(CFG)
    runtime_cfg = dict(feature_cfg.get("runtime_cfg") or feature_cfg)
    runtime_cfg.update(
        {
            "use_perps": market_mode == "perps",
            "market_mode": market_mode,
            "data_root": str(data_root),
            "artifact_data_root": str(artifact_data_root),
            "live_data_root": str(data_root),
            "offline_feature_data_root": str(artifact_data_root),
            "live_feature_memory_cache_enabled": False,
            "live_feature_return_latest_only": False,
            "live_feature_snapshot_cache_enabled": False,
            "live_feature_rolling_cache_enabled": True,
            "live_feature_offline_cache_enabled": True,
            "feature_transform_cache_enabled": False,
            # Historical parity must replay the feature contract that the
            # selected model was trained with, even when today's deployable
            # config is stricter about portable feature admission.
            "feature_portability_mode": "legacy",
            "feature_portability_strict": False,
            "historical_inference_parity_allow_legacy_deleted_keys": True,
            "historical_inference_parity_allow_missing_live_sources": True,
            "historical_inference_parity_preserve_cached_features": True,
            # Final-fit replay only needs strict feature contract checks and
            # predictions. Internal LGBM diagnostics are expensive historical
            # enrichments and are not part of the 1:1 feature-vector parity
            # contract being verified here.
            "inference_lgbm_internal_diagnostics_enabled": False,
        }
    )
    parity_contract: dict[str, Any] = {}
    try:
        parity_contract = load_training_live_parity_contract(
            data_root=str(artifact_data_root),
            run_id=run_id,
        )
    except Exception as exc:
        tprint(f"Training-live parity contract load failed for replay; continuing without it: {exc}")
    if isinstance(parity_contract, dict) and parity_contract:
        feature_cfg.setdefault("training_live_parity_contract", parity_contract)
        runtime_cfg.setdefault("training_live_parity_contract", parity_contract)
        feature_source = (
            parity_contract.get("feature_source")
            if isinstance(parity_contract.get("feature_source"), dict)
            else {}
        )
        contract_feature_source_run_id = feature_source.get("run_id")
        if contract_feature_source_run_id:
            runtime_cfg.setdefault(
                "live_feature_source_run_id",
                str(contract_feature_source_run_id),
            )
            runtime_cfg.setdefault(
                "feature_source_run_id",
                str(contract_feature_source_run_id),
            )
            tprint(
                "Historical replay loaded training-live parity contract: "
                f"feature_source_run_id={contract_feature_source_run_id} "
                f"path={parity_contract.get('_contract_path', '')}"
            )
    if feature_source_run_id:
        runtime_cfg["offline_feature_run_id"] = str(feature_source_run_id)
    if disable_rolling_cache:
        runtime_cfg["live_feature_rolling_cache_enabled"] = False
    if disable_offline_cache:
        runtime_cfg["live_feature_offline_cache_enabled"] = False
    if offline_allowed_periods:
        runtime_cfg["live_feature_offline_allowed_periods"] = offline_allowed_periods
    bundle = state.get("bundle", {}) if isinstance(state.get("bundle"), dict) else {}
    runtime_cfg.setdefault("bundle", bundle)
    for key in (
        "feature_transform_contract",
        "feature_transform_contract_hash",
        "feature_transform_manifest",
    ):
        value = state.get(key)
        if value is None and isinstance(bundle, dict):
            value = bundle.get(key)
        if value is not None:
            feature_cfg[key] = value
            runtime_cfg[key] = value
    feature_cfg["runtime_cfg"] = runtime_cfg
    return feature_cfg


def _reference_feature_run_id(
    feature_cfg: dict[str, Any],
    *,
    active_run_id: str,
    override_run_id: str | None = None,
) -> str:
    if override_run_id:
        return str(override_run_id)
    runtime_cfg = feature_cfg.get("runtime_cfg") if isinstance(feature_cfg, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    parity_contract = (
        runtime_cfg.get("training_live_parity_contract")
        if isinstance(runtime_cfg.get("training_live_parity_contract"), dict)
        else feature_cfg.get("training_live_parity_contract")
        if isinstance(feature_cfg.get("training_live_parity_contract"), dict)
        else {}
    )
    feature_source = (
        parity_contract.get("feature_source")
        if isinstance(parity_contract.get("feature_source"), dict)
        else {}
    )
    source_run_id = feature_source.get("run_id")
    if source_run_id:
        return str(source_run_id)
    for key in (
        "live_feature_source_run_id",
        "offline_feature_run_id",
        "feature_source_run_id",
    ):
        value = runtime_cfg.get(key)
        if value:
            return str(value)
    return str(active_run_id)


def _root_for_exchange_scoped_data(data_root: Path) -> Path:
    parts = data_root.parts
    if len(parts) >= 2 and parts[-2] == "exchanges":
        return Path(*parts[:-2])
    return data_root


def _attach_external_kraken_spot_panels(
    panel: dict[str, pd.DataFrame],
    *,
    data_root: Path,
    market_mode: str,
) -> None:
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return
    if str(market_mode).lower() not in {"perp", "perps"}:
        return
    if "kraken" not in str(data_root).lower():
        return
    cfg = {
        "exchange_id": "kraken",
        "exchange": "kraken",
        "market_mode": "perps",
        "data_root": str(_root_for_exchange_scoped_data(data_root)),
    }
    external_spot = _load_external_kraken_spot_panels(
        cfg,
        list(close.columns),
        close.index,
    )
    if external_spot:
        tprint(f"Loaded external Kraken spot panels for parity: fields={len(external_spot)}")
    for key, frame in external_spot.items():
        existing = panel.get(key)
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            existing_count = int(existing.gt(0.0).sum().sum())
            external_count = int(frame.gt(0.0).sum().sum())
            if external_count <= existing_count:
                continue
        panel[key] = frame


def _compare_features(
    samples: pd.DataFrame,
    fresh_feats: dict[str, pd.DataFrame],
    reference_rows: dict[tuple[str, pd.Timestamp], pd.Series],
    feature_keys: set[str],
) -> pd.DataFrame:
    started = time.monotonic()
    lazy_lookup = hasattr(fresh_feats, "latest_values_at")
    fresh_by_feature: dict[str, pd.DataFrame] = {}
    if not lazy_lookup:
        fresh_by_feature = {
            str(feature): frame
            for feature, frame in fresh_feats.items()
            if str(feature) in feature_keys and isinstance(frame, pd.DataFrame) and not frame.empty
        }
    rows = []
    total = len(samples)
    tprint(
        "Comparing feature values: "
        f"samples={total:,} comparable_features={len(feature_keys):,} "
        f"fresh_feature_matrices={len(fresh_by_feature):,}"
    )
    for i, (_, sample) in enumerate(samples.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            elapsed = time.monotonic() - started
            tprint(f"  feature parity progress: {i:,}/{total:,} samples elapsed={elapsed:.1f}s")
        symbol = _normalise_symbol(sample["symbol"])
        ts = pd.Timestamp(sample["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        ref = reference_rows.get((symbol, ts), pd.Series(dtype=float))
        for feature in sorted(feature_keys):
            matrix = fresh_by_feature.get(feature)
            fresh_has_value = (
                feature in fresh_feats
                if lazy_lookup
                else matrix is not None and ts in matrix.index and symbol in matrix.columns
            )
            ref_has_value = feature in ref.index
            if not fresh_has_value and not ref_has_value:
                continue
            if lazy_lookup and feature in fresh_feats:
                values = fresh_feats.latest_values_at(feature, [symbol], ts)
                fval = _safe_float(values.get(symbol, np.nan))
                fresh_has_value = np.isfinite(fval)
            else:
                fval = _safe_float(matrix.at[ts, symbol] if fresh_has_value else np.nan)
            rval = _safe_float(ref.get(feature, np.nan))
            fresh_finite = np.isfinite(fval)
            ref_finite = np.isfinite(rval)
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "feature": feature,
                    "inference_value": fval,
                    "training_value": rval,
                    "abs_diff": abs(fval - rval)
                    if fresh_finite and ref_finite
                    else np.nan,
                    "inference_missing": (not fresh_finite) and ref_finite,
                    "training_missing": fresh_finite and (not ref_finite),
                    "both_missing": (not fresh_finite) and (not ref_finite),
                }
            )
    out = pd.DataFrame(rows)
    tprint(f"Feature value comparison complete: rows={len(out):,} elapsed={time.monotonic() - started:.1f}s")
    return out


def _canonical_hash_value(value: float) -> str:
    value = _safe_float(value)
    if not np.isfinite(value):
        return "missing"
    return float(value).hex()


def _feature_vector_hash_report(
    samples: pd.DataFrame,
    fresh_feats: dict[str, pd.DataFrame],
    reference_rows: dict[tuple[str, pd.Timestamp], pd.Series],
    feature_keys: set[str],
    *,
    tolerance: float,
) -> pd.DataFrame:
    import hashlib

    started = time.monotonic()
    lazy_lookup = hasattr(fresh_feats, "latest_values_at")
    fresh_by_feature: dict[str, pd.DataFrame] = {}
    if not lazy_lookup:
        fresh_by_feature = {
            str(feature): frame
            for feature, frame in fresh_feats.items()
            if str(feature) in feature_keys and isinstance(frame, pd.DataFrame) and not frame.empty
        }
    ordered_features = sorted(str(feature) for feature in feature_keys)
    rows: list[dict[str, Any]] = []
    total = len(samples)
    tprint(
        "Hashing feature vectors: "
        f"samples={total:,} comparable_features={len(ordered_features):,}"
    )
    for i, (_, sample) in enumerate(samples.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            elapsed = time.monotonic() - started
            tprint(f"  vector hash progress: {i:,}/{total:,} samples elapsed={elapsed:.1f}s")
        symbol = _normalise_symbol(sample["symbol"])
        ts = pd.Timestamp(sample["timestamp"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        ref = reference_rows.get((symbol, ts), pd.Series(dtype=float))
        inference_payload: list[tuple[str, str]] = []
        training_payload: list[tuple[str, str]] = []
        mismatched: list[str] = []
        missing_inference: list[str] = []
        missing_training: list[str] = []
        max_abs = float("nan")
        worst_feature = ""
        both_finite = 0
        inference_finite = 0
        training_finite = 0
        for feature in ordered_features:
            matrix = fresh_by_feature.get(feature)
            fresh_has_value = (
                feature in fresh_feats
                if lazy_lookup
                else matrix is not None and ts in matrix.index and symbol in matrix.columns
            )
            if lazy_lookup and feature in fresh_feats:
                values = fresh_feats.latest_values_at(feature, [symbol], ts)
                fval = _safe_float(values.get(symbol, np.nan))
            else:
                fval = _safe_float(matrix.at[ts, symbol] if fresh_has_value else np.nan)
            rval = _safe_float(ref.get(feature, np.nan))
            fresh_finite = np.isfinite(fval)
            ref_finite = np.isfinite(rval)
            if fresh_finite:
                inference_finite += 1
            if ref_finite:
                training_finite += 1
            if fresh_finite and ref_finite:
                both_finite += 1
                delta = abs(float(fval) - float(rval))
                if not np.isfinite(max_abs) or delta > max_abs:
                    max_abs = float(delta)
                    worst_feature = feature
                if delta > float(tolerance):
                    mismatched.append(feature)
            elif ref_finite and not fresh_finite:
                missing_inference.append(feature)
            elif fresh_finite and not ref_finite:
                missing_training.append(feature)
            inference_payload.append((feature, _canonical_hash_value(fval)))
            training_payload.append((feature, _canonical_hash_value(rval)))
        inference_hash = hashlib.sha256(
            json.dumps(inference_payload, separators=(",", ":"), sort_keys=False).encode("utf-8")
        ).hexdigest()
        training_hash = hashlib.sha256(
            json.dumps(training_payload, separators=(",", ":"), sort_keys=False).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "strategy_id": sample.get("strategy_id", ""),
                "feature_count": len(ordered_features),
                "inference_finite_count": inference_finite,
                "training_finite_count": training_finite,
                "both_finite_count": both_finite,
                "missing_inference_count": len(missing_inference),
                "missing_training_count": len(missing_training),
                "mismatch_count_gt_tolerance": len(mismatched),
                "max_abs_diff": max_abs,
                "worst_feature": worst_feature,
                "inference_vector_hash": inference_hash,
                "training_vector_hash": training_hash,
                "exact_hash_equal": inference_hash == training_hash,
                "parity_ok": (
                    inference_hash == training_hash
                    and not missing_inference
                    and not missing_training
                    and not mismatched
                ),
                "mismatched_features_json": json.dumps(mismatched, separators=(",", ":")),
                "missing_inference_features_json": json.dumps(
                    missing_inference,
                    separators=(",", ":"),
                ),
                "missing_training_features_json": json.dumps(
                    missing_training,
                    separators=(",", ":"),
                ),
            }
        )
    out = pd.DataFrame(rows)
    tprint(
        "Feature vector hash report complete: "
        f"rows={len(out):,} elapsed={time.monotonic() - started:.1f}s"
    )
    return out


def _score_predictions(
    samples: pd.DataFrame,
    fresh_feats: dict[str, pd.DataFrame],
    orchestrator: ModelOrchestrator,
    strategy_id: str,
    *,
    calibration_data: dict[str, dict[str, Any]] | None = None,
    policy_rank_scores: np.ndarray | None = None,
    prediction_universe_symbols: list[str] | None = None,
) -> pd.DataFrame:
    started = time.monotonic()
    rows = []
    side_default = "short" if strategy_id.startswith("short_") else "long"
    total = len(samples)
    tprint(f"Scoring final-fit predictions: samples={total:,} strategy={strategy_id}")
    work = samples.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["symbol_norm"] = work["symbol"].map(_normalise_symbol)
    work["__row_id"] = [f"row_{i}" for i in range(len(work))]
    universe_symbols = (
        list(dict.fromkeys(_normalise_symbol(symbol) for symbol in prediction_universe_symbols))
        if prediction_universe_symbols
        else None
    )
    model_strategy_id = strategy_id
    core_strategy_id = strategy_core_id(strategy_id)
    if universe_symbols is None:
        feature_frames: list[pd.DataFrame] = []
        row_lookup: dict[str, pd.Series] = {}
        for ts, group in work.groupby("timestamp", sort=True):
            sample_symbols = list(
                dict.fromkeys(group["symbol_norm"].dropna().astype(str).tolist())
            )
            feature_rows = get_features_for_candidates(fresh_feats, sample_symbols, ts=ts)
            if feature_rows.empty:
                continue
            for _, sample in group.iterrows():
                symbol = str(sample["symbol_norm"])
                row_id = str(sample["__row_id"])
                if symbol not in feature_rows.index:
                    continue
                row = feature_rows.loc[symbol].copy()
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                frame = row.to_frame().T
                frame.index = pd.Index([row_id], name="row_id")
                frame["__symbol__"] = symbol
                frame["__ts__"] = ts
                feature_frames.append(frame)
                row_lookup[row_id] = sample
        if feature_frames:
            all_feature_rows = pd.concat(feature_frames, axis=0, copy=False)
        else:
            all_feature_rows = pd.DataFrame()
        if "is_long" in work.columns:
            side = "long" if bool(work["is_long"].astype(bool).mode().iloc[0]) else "short"
        else:
            side = side_default
        alpha = pd.Series(dtype=float)
        meta = pd.Series(dtype=float)
        base_error = None
        meta_error = None
        if not all_feature_rows.empty:
            try:
                alpha_parts: list[pd.Series] = []
                chunk_size = int(
                    os.environ.get("EPM_HISTORICAL_PARITY_PREDICT_CHUNK_ROWS", "512")
                    or "512"
                )
                chunk_size = max(1, chunk_size)
                for start in range(0, len(all_feature_rows), chunk_size):
                    chunk = all_feature_rows.iloc[start : start + chunk_size]
                    pred = orchestrator.predict_alpha(chunk, side, model_strategy_id)
                    alpha_parts.append(pred.reindex(chunk.index))
                    if len(all_feature_rows) > chunk_size:
                        tprint(
                            "  alpha prediction chunk: "
                            f"{min(start + chunk_size, len(all_feature_rows)):,}/"
                            f"{len(all_feature_rows):,}"
                        )
                alpha = (
                    pd.concat(alpha_parts).reindex(all_feature_rows.index)
                    if alpha_parts
                    else pd.Series(index=all_feature_rows.index, data=np.nan, dtype=float)
                )
            except Exception as exc:
                base_error = str(exc)
                alpha = pd.Series(index=all_feature_rows.index, data=np.nan, dtype=float)
            try:
                if isinstance(alpha, pd.Series) and not alpha.empty:
                    meta_base = all_feature_rows.copy()
                    meta_base[model_strategy_id] = alpha.reindex(meta_base.index)
                    meta_parts: list[pd.Series] = []
                    for start in range(0, len(meta_base), chunk_size):
                        chunk = meta_base.iloc[start : start + chunk_size]
                        pred = orchestrator.predict_meta(
                            chunk,
                            side,
                            model_strategy_id or core_strategy_id,
                        )
                        meta_parts.append(pred.reindex(chunk.index))
                        if len(meta_base) > chunk_size:
                            tprint(
                                "  meta prediction chunk: "
                                f"{min(start + chunk_size, len(meta_base)):,}/"
                                f"{len(meta_base):,}"
                            )
                    meta = (
                        pd.concat(meta_parts).reindex(meta_base.index)
                        if meta_parts
                        else pd.Series(index=meta_base.index, data=np.nan, dtype=float)
                    )
                else:
                    meta = pd.Series(index=all_feature_rows.index, data=np.nan, dtype=float)
            except Exception as exc:
                meta_error = str(exc)
                meta = pd.Series(index=all_feature_rows.index, data=np.nan, dtype=float)
        for _, sample in work.iterrows():
            row_id = str(sample["__row_id"])
            symbol = str(sample["symbol_norm"])
            has_features = row_id in row_lookup
            out = {
                "timestamp": sample["timestamp"],
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "feature_cols": int(all_feature_rows.shape[1]) if has_features else 0,
                "oof_base_clf": _safe_float(sample.get("oof_base_clf")),
                "oof_meta_clf": _safe_float(sample.get("oof_meta_clf")),
                "oof_pred": _safe_float(sample.get("oof_pred")),
            }
            if not has_features:
                out["prediction_error"] = "missing_feature_row"
                out["final_fit_base_pred"] = np.nan
                out["final_fit_meta_pred"] = np.nan
            else:
                if base_error:
                    out["base_prediction_error"] = base_error
                if meta_error:
                    out["meta_prediction_error"] = meta_error
                out["final_fit_base_pred"] = _safe_float(alpha.get(row_id, np.nan))
                out["final_fit_meta_pred"] = _safe_float(meta.get(row_id, np.nan))
                out["chain_action"] = (
                    "meta_prediction"
                    if np.isfinite(out["final_fit_meta_pred"])
                    else "no_meta_prediction"
                )
                out["chain_reason"] = "batched_meta_replay"
            if np.isfinite(out.get("final_fit_meta_pred", np.nan)):
                calibrated, _ = calibrated_score_and_threshold(
                    raw_score=float(out["final_fit_meta_pred"]),
                    strategy_id=strategy_id,
                    calibration_data=calibration_data or {},
                    default_threshold=1.0,
                )
                out["final_fit_calibrated_score"] = _safe_float(calibrated)
                if policy_rank_scores is not None:
                    out["final_fit_policy_rank_pct"] = _safe_float(
                        policy_rank_pct_from_sorted_scores(policy_rank_scores, float(calibrated))
                    )
            out["base_pred_abs_diff_vs_oof"] = (
                abs(out["final_fit_base_pred"] - out["oof_base_clf"])
                if np.isfinite(out.get("final_fit_base_pred", np.nan))
                and np.isfinite(out.get("oof_base_clf", np.nan))
                else np.nan
            )
            oof_meta = out["oof_meta_clf"] if np.isfinite(out["oof_meta_clf"]) else out["oof_pred"]
            out["meta_pred_abs_diff_vs_oof"] = (
                abs(out["final_fit_meta_pred"] - oof_meta)
                if np.isfinite(out.get("final_fit_meta_pred", np.nan))
                and np.isfinite(oof_meta)
                else np.nan
            )
            out["policy_calibrated_score_ref"] = _safe_float(sample.get("calibrated_score"))
            out["policy_rank_pct_ref"] = _safe_float(
                sample.get(
                    "rank_pct",
                    sample.get("strategy_rank_pct", sample.get("normalized_rank_score")),
                )
            )
            out["policy_calibrated_score_abs_diff"] = (
                abs(
                    out.get("final_fit_calibrated_score", np.nan)
                    - out["policy_calibrated_score_ref"]
                )
                if np.isfinite(out.get("final_fit_calibrated_score", np.nan))
                and np.isfinite(out["policy_calibrated_score_ref"])
                else np.nan
            )
            out["policy_rank_pct_abs_diff"] = (
                abs(out.get("final_fit_policy_rank_pct", np.nan) - out["policy_rank_pct_ref"])
                if np.isfinite(out.get("final_fit_policy_rank_pct", np.nan))
                and np.isfinite(out["policy_rank_pct_ref"])
                else np.nan
            )
            rows.append(out)
        out = pd.DataFrame(rows)
        tprint(
            "Prediction scoring complete: "
            f"rows={len(out):,} elapsed={time.monotonic() - started:.1f}s "
            "mode=batched"
        )
        return out
    done = 0
    for ts, group in work.groupby("timestamp", sort=True):
        ts_started = time.monotonic()
        sample_symbols = list(dict.fromkeys(group["symbol_norm"].dropna().astype(str).tolist()))
        symbols = universe_symbols or sample_symbols
        side_values = group.get("is_long")
        if side_values is not None:
            side = "long" if bool(side_values.astype(bool).mode().iloc[0]) else "short"
        else:
            side = side_default
        feature_rows = get_features_for_candidates(fresh_feats, symbols, ts=ts)
        alpha = pd.Series(dtype=float)
        meta = pd.Series(dtype=float)
        base_error = None
        meta_error = None
        if not feature_rows.empty:
            try:
                alpha = orchestrator.predict_alpha(feature_rows, side, model_strategy_id)
            except Exception as exc:
                base_error = str(exc)
                alpha = pd.Series(index=feature_rows.index, data=np.nan, dtype=float)
            try:
                if isinstance(alpha, pd.Series) and not alpha.empty:
                    meta_base = feature_rows.copy()
                    meta_base[model_strategy_id] = alpha.reindex(meta_base.index)
                    meta = orchestrator.predict_meta(
                        meta_base,
                        side,
                        model_strategy_id or core_strategy_id,
                    )
                else:
                    meta = pd.Series(index=feature_rows.index, data=np.nan, dtype=float)
            except Exception as exc:
                meta_error = str(exc)
                meta = pd.Series(index=feature_rows.index, data=np.nan, dtype=float)
        for _, sample in group.iterrows():
            symbol = str(sample["symbol_norm"])
            out = {
                "timestamp": ts,
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "feature_cols": int(feature_rows.shape[1]) if symbol in feature_rows.index else 0,
                "oof_base_clf": _safe_float(sample.get("oof_base_clf")),
                "oof_meta_clf": _safe_float(sample.get("oof_meta_clf")),
                "oof_pred": _safe_float(sample.get("oof_pred")),
            }
            if feature_rows.empty or symbol not in feature_rows.index:
                out["prediction_error"] = "missing_feature_row"
                out["final_fit_base_pred"] = np.nan
                out["final_fit_meta_pred"] = np.nan
            else:
                if base_error:
                    out["base_prediction_error"] = base_error
                if meta_error:
                    out["meta_prediction_error"] = meta_error
                out["final_fit_base_pred"] = _safe_float(alpha.get(symbol, np.nan))
                out["final_fit_meta_pred"] = _safe_float(meta.get(symbol, np.nan))
                out["chain_action"] = "meta_prediction" if np.isfinite(out["final_fit_meta_pred"]) else "no_meta_prediction"
                out["chain_reason"] = "batched_meta_replay"
            if np.isfinite(out.get("final_fit_meta_pred", np.nan)):
                calibrated, _ = calibrated_score_and_threshold(
                    raw_score=float(out["final_fit_meta_pred"]),
                    strategy_id=strategy_id,
                    calibration_data=calibration_data or {},
                    default_threshold=1.0,
                )
                out["final_fit_calibrated_score"] = _safe_float(calibrated)
                if policy_rank_scores is not None:
                    out["final_fit_policy_rank_pct"] = _safe_float(
                        policy_rank_pct_from_sorted_scores(policy_rank_scores, float(calibrated))
                    )
            out["base_pred_abs_diff_vs_oof"] = (
                abs(out["final_fit_base_pred"] - out["oof_base_clf"])
                if np.isfinite(out.get("final_fit_base_pred", np.nan))
                and np.isfinite(out.get("oof_base_clf", np.nan))
                else np.nan
            )
            oof_meta = out["oof_meta_clf"] if np.isfinite(out["oof_meta_clf"]) else out["oof_pred"]
            out["meta_pred_abs_diff_vs_oof"] = (
                abs(out["final_fit_meta_pred"] - oof_meta)
                if np.isfinite(out.get("final_fit_meta_pred", np.nan))
                and np.isfinite(oof_meta)
                else np.nan
            )
            out["policy_calibrated_score_ref"] = _safe_float(sample.get("calibrated_score"))
            out["policy_rank_pct_ref"] = _safe_float(
                sample.get(
                    "rank_pct",
                    sample.get("strategy_rank_pct", sample.get("normalized_rank_score")),
                )
            )
            out["policy_calibrated_score_abs_diff"] = (
                abs(out.get("final_fit_calibrated_score", np.nan) - out["policy_calibrated_score_ref"])
                if np.isfinite(out.get("final_fit_calibrated_score", np.nan))
                and np.isfinite(out["policy_calibrated_score_ref"])
                else np.nan
            )
            out["policy_rank_pct_abs_diff"] = (
                abs(out.get("final_fit_policy_rank_pct", np.nan) - out["policy_rank_pct_ref"])
                if np.isfinite(out.get("final_fit_policy_rank_pct", np.nan))
                and np.isfinite(out["policy_rank_pct_ref"])
                else np.nan
            )
            rows.append(out)
        done += len(group)
        if done == len(group) or done % 5000 < len(group) or done == total:
            elapsed = time.monotonic() - started
            tprint(
                "  prediction parity progress: "
                f"{done:,}/{total:,} samples elapsed={elapsed:.1f}s "
                f"last_ts_rows={len(group):,} pred_symbols={len(symbols):,} "
                f"last_ts_time={time.monotonic() - ts_started:.2f}s"
            )
    out = pd.DataFrame(rows)
    tprint(f"Prediction scoring complete: rows={len(out):,} elapsed={time.monotonic() - started:.1f}s")
    return out


def _summary(
    features: pd.DataFrame,
    preds: pd.DataFrame,
    vector_report: pd.DataFrame | None = None,
) -> dict[str, Any]:
    both_missing = (
        features["both_missing"].astype(bool)
        if "both_missing" in features
        else pd.Series(False, index=features.index)
    )
    common = features[
        ~features["inference_missing"].astype(bool)
        & ~features["training_missing"].astype(bool)
        & ~both_missing
    ]
    mismatched = common[pd.to_numeric(common["abs_diff"], errors="coerce") > 1e-6]
    summary = {
        "feature_rows": int(len(features)),
        "feature_common_rows": int(len(common)),
        "feature_missing_inference": int(features["inference_missing"].sum()) if not features.empty else 0,
        "feature_missing_training": int(features["training_missing"].sum()) if not features.empty else 0,
        "feature_mismatches_gt_1e_6": int(len(mismatched)),
        "feature_max_abs_diff": float(common["abs_diff"].max()) if not common.empty else None,
        "feature_mean_abs_diff": float(common["abs_diff"].mean()) if not common.empty else None,
        "prediction_rows": int(len(preds)),
        "base_pred_max_abs_diff_vs_oof": float(preds["base_pred_abs_diff_vs_oof"].max())
        if "base_pred_abs_diff_vs_oof" in preds and preds["base_pred_abs_diff_vs_oof"].notna().any()
        else None,
        "meta_pred_max_abs_diff_vs_oof": float(preds["meta_pred_abs_diff_vs_oof"].max())
        if "meta_pred_abs_diff_vs_oof" in preds and preds["meta_pred_abs_diff_vs_oof"].notna().any()
        else None,
        "policy_calibrated_score_max_abs_diff": float(preds["policy_calibrated_score_abs_diff"].max())
        if "policy_calibrated_score_abs_diff" in preds and preds["policy_calibrated_score_abs_diff"].notna().any()
        else None,
        "policy_rank_pct_max_abs_diff": float(preds["policy_rank_pct_abs_diff"].max())
        if "policy_rank_pct_abs_diff" in preds and preds["policy_rank_pct_abs_diff"].notna().any()
        else None,
    }
    if vector_report is not None and not vector_report.empty:
        summary.update(
            {
                "feature_vector_rows": int(len(vector_report)),
                "feature_vector_parity_ok_rows": int(
                    vector_report["parity_ok"].astype(bool).sum()
                    if "parity_ok" in vector_report
                    else 0
                ),
                "feature_vector_exact_hash_mismatch_rows": int(
                    (~vector_report["exact_hash_equal"].astype(bool)).sum()
                    if "exact_hash_equal" in vector_report
                    else len(vector_report)
                ),
                "feature_vector_tolerance_mismatch_rows": int(
                    pd.to_numeric(
                        vector_report.get(
                            "mismatch_count_gt_tolerance",
                            pd.Series(0, index=vector_report.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                ),
                "feature_vector_missing_inference_rows": int(
                    pd.to_numeric(
                        vector_report.get(
                            "missing_inference_count",
                            pd.Series(0, index=vector_report.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                ),
                "feature_vector_missing_training_rows": int(
                    pd.to_numeric(
                        vector_report.get(
                            "missing_training_count",
                            pd.Series(0, index=vector_report.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                ),
                "feature_vector_max_abs_diff": float(
                    pd.to_numeric(vector_report["max_abs_diff"], errors="coerce").max()
                )
                if "max_abs_diff" in vector_report
                and pd.to_numeric(vector_report["max_abs_diff"], errors="coerce").notna().any()
                else None,
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-data-root", type=Path, default=None)
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument(
        "--sample-source",
        choices=("oof", "policy_rank_reference", "policy_candidates"),
        default="oof",
        help=(
            "Replay OOF rows, policy rank-reference rows, or selected "
            "simple_policy_optimiser candidate rows."
        ),
    )
    parser.add_argument("--market-mode", choices=("spot", "perps"), default="spot")
    parser.add_argument("--live-quote-currency", default="USDC")
    parser.add_argument("--sample-rows", type=int, default=12)
    parser.add_argument(
        "--samples-per-symbol",
        type=int,
        default=0,
        help="Sample up to this many OOF rows per eligible symbol before optional global --sample-rows cap.",
    )
    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=0,
        help="Skip this many earliest OOF rows per symbol before sampling.",
    )
    parser.add_argument("--lookback-hours", type=int, default=24 * 90)
    parser.add_argument(
        "--feature-source-run-id",
        default=None,
        help="Override the offline selected-feature run id used for parity reference loading.",
    )
    parser.add_argument(
        "--disable-rolling-cache",
        action="store_true",
        help="Do not merge persisted live rolling feature cache into the replay feature set.",
    )
    parser.add_argument(
        "--disable-offline-cache",
        action="store_true",
        help="Disable selected offline feature cache and force live-style feature recompute.",
    )
    parser.add_argument(
        "--feature-load-path",
        choices=("direct", "inference_candidate"),
        default="direct",
        help=(
            "direct calls load_or_compute_features; inference_candidate routes "
            "through run_inference._select_candidates_and_load_features."
        ),
    )
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--min-timestamp", default=None)
    parser.add_argument(
        "--rank-reference-dir",
        type=Path,
        default=None,
        help=(
            "Override simple_policy_optimiser rank-reference directory for "
            "policy_rank_reference sampling and score comparison."
        ),
    )
    parser.add_argument(
        "--policy-candidates-path",
        type=Path,
        default=None,
        help="Override simple_policy_optimiser candidate parquet path.",
    )
    parser.add_argument(
        "--policy-artifact-root",
        type=Path,
        default=None,
        help=(
            "Override the inference policy artifact root used for strategy_for_inference "
            "and best_policy_params lookups during replay."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--basket-mode",
        choices=("full", "sample"),
        default="full",
        help=(
            "Use full local/trained basket for exact cross-asset feature parity, "
            "or only sampled symbols for faster debugging with changed cross-asset values."
        ),
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Only compare training vs inference feature values; skip model scoring.",
    )
    parser.add_argument(
        "--skip-feature-comparison",
        action="store_true",
        help="Skip per-cell feature parity rows; useful after full feature parity is already proven.",
    )
    parser.add_argument(
        "--dump-feature-vectors",
        action="store_true",
        help=(
            "Write per-row selected-feature vector hashes and strict parity "
            "status in feature_vector_parity.csv."
        ),
    )
    parser.add_argument(
        "--feature-vector-tolerance",
        default=1e-9,
        type=float,
        help="Absolute tolerance for per-row vector mismatch counts.",
    )
    parser.add_argument(
        "--fail-on-feature-mismatch",
        action="store_true",
        help=(
            "Exit non-zero if feature comparison or vector-hash parity finds "
            "missing/asymmetric/tolerance mismatches."
        ),
    )
    parser.add_argument(
        "--score-full-universe",
        action="store_true",
        help=(
            "Score every symbol in the feature basket at each sampled timestamp. "
            "By default, replay scores only sampled rows while still computing "
            "features with the requested basket context."
        ),
    )
    args = parser.parse_args()
    if args.policy_artifact_root is not None:
        os.environ["EPM_INFERENCE_POLICY_ARTIFACT_ROOT"] = str(args.policy_artifact_root)

    started = time.monotonic()
    artifact_data_root = args.artifact_data_root or args.data_root
    tprint(
        "Starting historical inference parity: "
        f"run_id={args.run_id} market_mode={args.market_mode} "
        f"data_root={args.data_root} artifact_data_root={artifact_data_root}"
    )
    if args.sample_source == "policy_rank_reference":
        if not args.strategy_id:
            raise SystemExit("--strategy-id is required with --sample-source policy_rank_reference")
        strategy_id = str(args.strategy_id)
        tprint(f"Resolved strategy={strategy_id} sample_source=policy_rank_reference")
        samples = _sample_policy_rank_reference_rows(
            artifact_data_root,
            args.run_id,
            strategy_id,
            sample_rows=args.sample_rows,
            min_timestamp=args.min_timestamp,
            rank_reference_dir=args.rank_reference_dir,
        )
    elif args.sample_source == "policy_candidates":
        if not args.strategy_id:
            raise SystemExit("--strategy-id is required with --sample-source policy_candidates")
        strategy_id = str(args.strategy_id)
        tprint(f"Resolved strategy={strategy_id} sample_source=policy_candidates")
        samples = _sample_policy_candidate_rows(
            artifact_data_root,
            args.run_id,
            strategy_id,
            sample_rows=args.sample_rows,
            min_timestamp=args.min_timestamp,
            candidate_path=args.policy_candidates_path,
        )
        concrete_strategy_ids = sorted(
            str(value)
            for value in samples.get("strategy_id", pd.Series(dtype=str)).dropna().unique()
            if str(value)
        )
        if len(concrete_strategy_ids) == 1 and concrete_strategy_ids[0] != strategy_id:
            tprint(
                "Resolved policy candidate alias to concrete strategy id: "
                f"{strategy_id} -> {concrete_strategy_ids[0]}"
            )
            strategy_id = concrete_strategy_ids[0]
    else:
        meta_path = _meta_oof_path(artifact_data_root, args.run_id, args.strategy_id)
        strategy_id = _strategy_from_meta_oof_path(meta_path)
        tprint(f"Resolved strategy={strategy_id} meta_oof={meta_path}")
        samples = _sample_oof_rows(
            meta_path,
            sample_rows=args.sample_rows,
            samples_per_symbol=args.samples_per_symbol,
            warmup_rows=args.warmup_rows,
            min_timestamp=args.min_timestamp,
        )
    if samples.empty:
        raise SystemExit("No OOF rows selected for parity replay.")

    min_ts = pd.Timestamp(samples["timestamp"].min())
    max_ts = pd.Timestamp(samples["timestamp"].max())
    sample_symbols = sorted({_normalise_symbol(s) for s in samples["symbol"]})
    if args.basket_mode == "sample":
        symbols = sample_symbols
        tprint(
            "Using sample-only basket. Cross-asset features can differ from training; "
            "use this mode only for debugging speed."
        )
    else:
        if args.market_mode == "perps":
            trained_symbols = load_trained_symbol_universe(str(artifact_data_root), str(args.run_id))
            symbols = sorted({_normalise_symbol(s) for s in trained_symbols})
            tprint(
                "Using trained perp universe for full-basket parity: "
                f"symbols={len(symbols):,}"
            )
        else:
            symbols = _local_quote_symbols(
                args.data_root,
                run_id=args.run_id,
                live_quote_currency=args.live_quote_currency,
                market_mode=args.market_mode,
            )
        if args.max_symbols and args.max_symbols > 0:
            extras = [s for s in symbols if s not in sample_symbols]
            symbols = sorted(set(sample_symbols + extras[: max(0, args.max_symbols - len(sample_symbols))]))
        if not symbols:
            symbols = sample_symbols
    sample_span_hours = int(
        np.ceil((max_ts - min_ts) / pd.Timedelta(hours=1))
    )
    inference_defaults = get_inference_defaults()
    min_warmup_hours = _required_tail_warmup_hours(
        lookback_hours=int(args.lookback_hours),
        trend_sma_hours=int(inference_defaults["trend_sma_hours"]),
        gate_vol_lookback_hours=int(inference_defaults["gate_vol_lookback_hours"]),
    )
    effective_lookback_hours = max(
        int(args.lookback_hours),
        sample_span_hours + 1,
        int(min_warmup_hours),
    )
    start_ts = min_ts - pd.Timedelta(hours=int(effective_lookback_hours))
    market_data_root = _historical_market_data_root(args.data_root, args.market_mode)
    tprint(
        f"Historical inference parity: strategy={strategy_id} samples={len(samples)} "
        f"sample_symbols={len(sample_symbols)} basket_symbols={len(symbols)} "
        f"window={start_ts}..{max_ts} "
        f"effective_lookback_hours={effective_lookback_hours} basket_mode={args.basket_mode} "
        f"market_data_root={market_data_root}"
    )
    panel_started = time.monotonic()
    panel = _load_panel(
        data_root=market_data_root,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=max_ts,
    )
    if not panel or "close" not in panel:
        raise SystemExit("No historical OHLCV panel loaded.")
    tprint(
        "Loaded historical panel: "
        f"fields={len(panel)} close_shape={panel['close'].shape} "
        f"elapsed={time.monotonic() - panel_started:.1f}s"
    )
    _attach_external_kraken_spot_panels(
        panel,
        data_root=market_data_root,
        market_mode=args.market_mode,
    )

    tprint("Loading trained model state and transform contract")
    state = load_full_state(args.run_id, str(artifact_data_root))
    unique_sample_timestamps = sorted(pd.to_datetime(samples["timestamp"], utc=True).dropna().unique())
    offline_allowed_periods = None
    if unique_sample_timestamps:
        offline_allowed_periods = [
            (
                pd.Timestamp(sample_ts),
                pd.Timestamp(sample_ts) + pd.Timedelta(microseconds=1),
            )
            for sample_ts in unique_sample_timestamps
        ]
        tprint(
            "Historical inference parity: using exact-timestamp selected-feature "
            "cache windows for offline feature load "
            f"timestamps={len(offline_allowed_periods)} "
            f"first={pd.Timestamp(unique_sample_timestamps[0])} "
            f"last={pd.Timestamp(unique_sample_timestamps[-1])}"
        )
    feature_cfg = _build_runtime_cfg(
        data_root=args.data_root,
        artifact_data_root=artifact_data_root,
        run_id=args.run_id,
        market_mode=args.market_mode,
        state=state,
        feature_source_run_id=args.feature_source_run_id,
        disable_rolling_cache=bool(args.disable_rolling_cache),
        disable_offline_cache=bool(args.disable_offline_cache),
        offline_allowed_periods=offline_allowed_periods,
    )
    required_keys = _feature_columns_for_state(state, strategy_id)
    tprint(f"Initial required feature contract keys: {len(required_keys):,}")
    lgbm_strategy_mask_rows: dict[str, dict[str, Any]] = {}
    try:
        mask_rows = _load_lgbm_strategy_mask_rows(
            str(artifact_data_root),
            args.run_id,
            market_mode=args.market_mode,
        )
        lgbm_strategy_mask_rows = _filter_lgbm_mask_rows_for_strategy(
            mask_rows,
            strategy_id,
        )
        required_keys |= set(
            _lgbm_mask_required_feature_keys(lgbm_strategy_mask_rows)
        )
        tprint(f"Required keys after strategy-mask features: {len(required_keys):,}")
    except Exception:
        tprint("Strategy-mask feature key load failed; continuing with model feature contract only")
        pass
    before_context_symbols = len(symbols)
    symbols = _add_required_context_symbols(
        symbols,
        required_keys,
        market_mode=args.market_mode,
        live_quote_currency=args.live_quote_currency,
    )
    if len(symbols) != before_context_symbols:
        tprint(
            "Added required benchmark context symbols for residual feature parity: "
            f"{before_context_symbols}->{len(symbols)}"
        )
        close_cols = set(str(c) for c in panel.get("close", pd.DataFrame()).columns)
        missing_panel_symbols = [sym for sym in symbols if sym not in close_cols]
        if missing_panel_symbols:
            panel_started = time.monotonic()
            tprint(
                "Reloading historical panel with required benchmark context: "
                f"missing={missing_panel_symbols[:5]}"
            )
            panel = _load_panel(
                data_root=market_data_root,
                symbols=symbols,
                start_ts=start_ts,
                end_ts=max_ts,
            )
            if not panel or "close" not in panel:
                raise SystemExit("No historical OHLCV panel loaded after context expansion.")
            tprint(
                "Reloaded historical panel: "
                f"fields={len(panel)} close_shape={panel['close'].shape} "
                f"elapsed={time.monotonic() - panel_started:.1f}s"
            )
            _attach_external_kraken_spot_panels(
                panel,
                data_root=market_data_root,
                market_mode=args.market_mode,
            )
    feature_started = time.monotonic()
    tprint(
        "Loading/computing inference features: "
        f"required_keys={len(required_keys):,} symbols={len(symbols):,} "
        f"lookback_hours={effective_lookback_hours} "
        f"feature_load_path={args.feature_load_path}"
    )
    feature_path_audit: dict[str, Any] = {
        "feature_load_path": args.feature_load_path,
        "disable_offline_cache": bool(args.disable_offline_cache),
        "disable_rolling_cache": bool(args.disable_rolling_cache),
        "feature_source_run_id": args.feature_source_run_id,
    }
    if args.feature_load_path == "inference_candidate":
        (
            thresholds,
            long_candidates,
            short_candidates,
            fresh_feats,
            strategy_candidate_masks,
        ) = _select_candidates_and_load_features(
            panel=panel,
            symbols=symbols,
            run_id=args.run_id,
            data_root=str(market_data_root),
            cfg=feature_cfg,
            lookback_hours=effective_lookback_hours,
            required_feature_keys=required_keys,
            lgbm_strategy_mask_rows=lgbm_strategy_mask_rows,
            feature_context_symbols=symbols,
            model_features_required=not bool(args.skip_predictions),
        )
        feature_path_audit.update(
            {
                "thresholds": thresholds,
                "long_candidates": int(len(long_candidates)),
                "short_candidates": int(len(short_candidates)),
                "strategy_masks": {
                    str(k): int(len(v or []))
                    for k, v in (strategy_candidate_masks or {}).items()
                },
            }
        )
    else:
        fresh_feats = load_or_compute_features(
            panel,
            list(panel["close"].columns),
            args.run_id,
            str(market_data_root),
            feature_cfg,
            lookback_hours=effective_lookback_hours,
            required_feature_keys=required_keys,
        )
    tprint(
        "Inference feature load/compute complete: "
        f"features={len(fresh_feats):,} elapsed={time.monotonic() - feature_started:.1f}s"
    )
    comparable_feature_keys = {
        key for key in required_keys if not _runtime_only_feature_key(key)
    }
    reference_feature_run_id = _reference_feature_run_id(
        feature_cfg,
        active_run_id=str(args.run_id),
        override_run_id=args.feature_source_run_id,
    )
    feature_path_audit["reference_feature_run_id"] = reference_feature_run_id
    tprint(
        "Reference feature rows will be loaded from feature source run: "
        f"{reference_feature_run_id}"
    )
    reference_rows: dict[tuple[str, pd.Timestamp], pd.Series] = {}
    if not args.skip_feature_comparison or args.dump_feature_vectors:
        reference_rows = _load_reference_feature_rows(
            artifact_data_root,
            reference_feature_run_id,
            samples,
            comparable_feature_keys,
        )
    if args.skip_feature_comparison:
        tprint("Skipping feature value comparison by request")
        feature_report = pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "feature",
                "inference_value",
                "training_value",
                "abs_diff",
                "inference_missing",
                "training_missing",
                "both_missing",
            ]
        )
    else:
        feature_report = _compare_features(
            samples,
            fresh_feats,
            reference_rows,
            comparable_feature_keys,
        )
    if args.dump_feature_vectors:
        vector_report = _feature_vector_hash_report(
            samples,
            fresh_feats,
            reference_rows,
            comparable_feature_keys,
            tolerance=float(args.feature_vector_tolerance),
        )
    else:
        vector_report = pd.DataFrame()
    if args.skip_predictions:
        tprint("Skipping prediction parity by request")
        prediction_report = pd.DataFrame()
    else:
        runtime_cfg = feature_cfg.get("runtime_cfg", feature_cfg)
        prediction_runtime_cfg = dict(runtime_cfg or {})
        prediction_runtime_cfg["inference_lgbm_internal_diagnostics_enabled"] = False
        tprint("Initializing ModelOrchestrator for prediction parity")
        orchestrator = ModelOrchestrator(
            state,
            runtime_cfg={"model_bundle": state.get("bundle", {}), **prediction_runtime_cfg},
        )
        calibration_data = load_calibration_curves(str(artifact_data_root), args.run_id)
        policy_rank_scores = None
        if args.sample_source == "policy_rank_reference":
            ref_path = _policy_rank_reference_path(
                artifact_data_root,
                args.run_id,
                strategy_id,
                rank_reference_dir=args.rank_reference_dir,
            )
            ref = pd.read_parquet(ref_path, columns=["calibrated_score"])
            policy_rank_scores = pd.to_numeric(
                ref["calibrated_score"], errors="coerce"
            ).to_numpy(dtype=np.float64)
        prediction_report = _score_predictions(
            samples,
            fresh_feats,
            orchestrator,
            strategy_id,
            calibration_data=calibration_data,
            policy_rank_scores=policy_rank_scores,
            prediction_universe_symbols=symbols if args.score_full_universe else None,
        )
    summary = _summary(feature_report, prediction_report, vector_report)
    out_dir = args.output_dir or (
        artifact_data_root / "artifacts" / args.run_id / "historical_inference_parity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_report.to_csv(out_dir / "feature_parity.csv", index=False)
    if args.dump_feature_vectors:
        vector_report.to_csv(out_dir / "feature_vector_parity.csv", index=False)
    prediction_report.to_csv(out_dir / "prediction_parity.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "feature_path_audit.json").write_text(
        json.dumps(feature_path_audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tprint(json.dumps(summary, indent=2, sort_keys=True, default=str))
    tprint(f"Wrote {out_dir}")
    tprint(f"Historical inference parity complete: elapsed={time.monotonic() - started:.1f}s")
    if args.fail_on_feature_mismatch:
        failures: list[str] = []
        if not feature_report.empty:
            feature_missing_inference = int(feature_report["inference_missing"].sum())
            feature_missing_training = int(feature_report["training_missing"].sum())
            feature_mismatches = int(
                pd.to_numeric(feature_report["abs_diff"], errors="coerce")
                .fillna(0.0)
                .gt(float(args.feature_vector_tolerance))
                .sum()
            )
            if feature_missing_inference:
                failures.append(f"feature_missing_inference={feature_missing_inference}")
            if feature_missing_training:
                failures.append(f"feature_missing_training={feature_missing_training}")
            if feature_mismatches:
                failures.append(f"feature_mismatches_gt_tolerance={feature_mismatches}")
        if args.dump_feature_vectors:
            if vector_report.empty:
                failures.append("feature_vector_rows=0")
            else:
                bad_vectors = int((~vector_report["parity_ok"].astype(bool)).sum())
                if bad_vectors:
                    failures.append(f"feature_vector_parity_failed_rows={bad_vectors}")
        if failures:
            print(
                "Feature parity failed: " + ", ".join(failures),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
