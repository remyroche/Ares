#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import (  # noqa: E402
    PartitionedOHLCVStore,
    scoped_data_root,
)
from extreme_price_movements.features import (  # noqa: E402
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    _is_live_synthesized_feature_key,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    load_or_compute_features,
    live_model_feature_store_strict,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.live_meta_feature_overlays import (  # noqa: E402
    live_ae_gmm_input_feature_columns,
    load_live_ae_gmm_state_payload,
    materialize_live_ae_gmm_features,
    materialize_live_source_regime_features,
)
from extreme_price_movements.inference.run_inference import (  # noqa: E402
    _lgbm_mask_required_feature_keys,
    _load_lgbm_strategy_mask_rows,
)
from extreme_price_movements.inference.config import (  # noqa: E402
    load_inference_config,
    load_trained_symbol_universe,
)
from extreme_price_movements.inference.data_fetcher import (  # noqa: E402
    MICRODATA_FRAME_FIELDS,
    PERP_OHLCV_EXTRA_FIELDS,
)
from extreme_price_movements.inference.symbol_mapping import symbol_bases  # noqa: E402
from extreme_price_movements.inference.model_orchestrator import (  # noqa: E402
    DELETED_MODEL_FEATURE_KEYS,
    FeatureParityError,
    ModelOrchestrator,
    _effective_selected_feature_contract,
    _strict_finite_model_matrix,
)
from extreme_price_movements.inference.parity import (  # noqa: E402
    calibrated_score_and_threshold,
    strategy_core_id,
)
from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    PolicyRankReferenceStore,
)
from extreme_price_movements.model_loader import load_full_state  # noqa: E402
from extreme_price_movements.simple_position_sizer import (  # noqa: E402
    load_calibration_curves,
)


def _slice_panel(
    panel: dict[str, pd.DataFrame],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    for name, frame in panel.items():
        if not isinstance(frame, pd.DataFrame) or not isinstance(
            frame.index, pd.DatetimeIndex
        ):
            out[name] = frame
            continue
        out[name] = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
    return out


def _normalise_symbol(symbol: object) -> str:
    text = str(symbol or "").upper().strip()
    perp_suffix = ""
    for suffix in (":USDT", ":USDC", ":USD"):
        if text.endswith(suffix):
            perp_suffix = suffix
            text = text[: -len(suffix)]
            break
    text = text.replace("-", "/").replace("_", "/")
    if "/" not in text:
        if text.endswith("USDC"):
            text = f"{text[:-4]}/USDC"
        elif text.endswith("USDT"):
            text = f"{text[:-4]}/USDT"
        elif text.endswith("USD"):
            text = f"{text[:-3]}/USD"
    return f"{text}{perp_suffix}" if perp_suffix and ":" not in text else text


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _logged_decision_feature_keys(decisions: pd.DataFrame) -> set[str]:
    """Return exact feature keys logged by live inference for replay decisions."""
    keys: set[str] = set()
    for col in ("base_model_features_json", "meta_model_features_json"):
        if col not in decisions.columns:
            continue
        for raw in decisions[col].dropna():
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if isinstance(parsed, list):
                keys.update(str(v) for v in parsed if str(v))
            elif isinstance(parsed, dict):
                keys.update(str(v) for v in parsed.keys() if str(v))
    return keys


def _json_mapping(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _feature_value_delta_summary(
    *,
    logged_values_raw: Any,
    feature_row: pd.DataFrame,
    symbol: str,
    feature_filter: Any = None,
) -> dict[str, Any]:
    logged_values = _json_mapping(logged_values_raw)
    if not logged_values or feature_row.empty or symbol not in feature_row.index:
        return {
            "count": 0,
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "worst_feature": "",
        }
    deltas: list[tuple[str, float]] = []
    row = feature_row.loc[symbol]
    for key, live_value in logged_values.items():
        key_s = str(key)
        if callable(feature_filter) and not bool(feature_filter(key_s)):
            continue
        if key not in row.index:
            continue
        live_float = _safe_float(live_value)
        replay_float = _safe_float(row.get(key))
        if np.isfinite(live_float) and np.isfinite(replay_float):
            deltas.append((key_s, abs(replay_float - live_float)))
    if not deltas:
        return {
            "count": 0,
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "worst_feature": "",
        }
    worst_feature, max_abs = max(deltas, key=lambda item: item[1])
    return {
        "count": len(deltas),
        "max_abs": float(max_abs),
        "mean_abs": float(np.mean([delta for _, delta in deltas])),
        "worst_feature": worst_feature,
    }


def _live_synthesized_feature_delta_summary(
    *,
    logged_values_raw: Any,
    feature_row: pd.DataFrame,
    symbol: str,
) -> dict[str, Any]:
    return _feature_value_delta_summary(
        logged_values_raw=logged_values_raw,
        feature_row=feature_row,
        symbol=symbol,
        feature_filter=_is_live_synthesized_feature_key,
    )


def _logged_values_frame(raw: Any, *, symbol: str) -> pd.DataFrame:
    values = _json_mapping(raw)
    if not values:
        return pd.DataFrame(index=[symbol])
    row: dict[str, float] = {}
    for key, value in values.items():
        val = _safe_float(value)
        row[str(key)] = val if np.isfinite(val) else np.nan
    return pd.DataFrame([row], index=[symbol], dtype=np.float32)


def _resolve_meta_model(
    orchestrator: ModelOrchestrator,
    *,
    side: str,
    strategy_id: str,
) -> tuple[str, Any]:
    meta_models = getattr(orchestrator, "meta_models", {}) or {}
    core = strategy_core_id(str(strategy_id))
    candidates = [
        str(strategy_id),
        core,
        f"{side}_{strategy_id}",
        f"{side}_{core}",
        f"{strategy_id}_clf",
        f"{core}_clf",
        f"{side}_{strategy_id}_clf",
        f"{side}_{core}_clf",
        f"{strategy_id}_tbm_clf",
        f"{core}_tbm_clf",
        f"{side}_{strategy_id}_tbm_clf",
        f"{side}_{core}_tbm_clf",
    ]
    for key in candidates:
        if key in meta_models:
            return key, meta_models.get(key)
    return str(strategy_id), None


def _predict_exact_logged_meta_input(
    *,
    orchestrator: ModelOrchestrator,
    side: str,
    strategy_id: str,
    logged_meta_frame: pd.DataFrame,
) -> float:
    if logged_meta_frame.empty:
        return float("nan")
    _, meta_model = _resolve_meta_model(
        orchestrator,
        side=side,
        strategy_id=strategy_id,
    )
    if meta_model is None:
        return float("nan")
    feat_cols = _effective_selected_feature_contract(meta_model)
    if not feat_cols and hasattr(meta_model, "feature_columns"):
        feat_cols = [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
    feat_cols = [
        str(c)
        for c in (feat_cols or [])
        if str(c) not in DELETED_MODEL_FEATURE_KEYS
    ]
    if not feat_cols:
        return float("nan")
    missing = [c for c in feat_cols if c not in logged_meta_frame.columns]
    if missing:
        return float("nan")
    try:
        X = _strict_finite_model_matrix(
            logged_meta_frame.reindex(columns=feat_cols),
            model_feature_cols=feat_cols,
            model_key=str(strategy_id),
        )
    except FeatureParityError:
        return float("nan")
    pred = meta_model.predict(X)
    arr = np.asarray(pred, dtype=float).reshape(-1)
    return _safe_float(arr[0]) if arr.size else float("nan")


def _load_recent_decisions(
    *,
    ledger_path: Path,
    trades_path: Path,
    max_rows: int,
    decision_start: str | None = None,
    require_rank_source: str | None = None,
) -> pd.DataFrame:
    ledger = _read_table(ledger_path)
    if ledger.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    for col in ("timestamp", "decision_ts", "signal_bar_ts"):
        if col in ledger.columns:
            ledger[col] = pd.to_datetime(ledger[col], utc=True, errors="coerce")
    ledger = ledger.dropna(subset=["signal_bar_ts", "symbol", "side", "strategy_id"])
    ledger["symbol"] = ledger["symbol"].map(_normalise_symbol)
    ledger["strategy_core"] = ledger["strategy_id"].map(strategy_core_id)
    if decision_start:
        start_ts = pd.Timestamp(decision_start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        ledger = ledger[ledger["decision_ts"] >= start_ts]
    if require_rank_source:
        if "rank_score_source" not in ledger.columns:
            return pd.DataFrame()
        ledger = ledger[
            ledger["rank_score_source"].astype(str).eq(str(require_rank_source))
        ]
    ledger = ledger.sort_values("decision_ts").tail(max_rows).reset_index(drop=True)
    live_defaults = {
        "base_pred": "live_base_pred",
        "meta_pred": "live_meta_pred",
        "calibrated_score": "live_calibrated_score",
        "normalized_rank_score": "live_rank_percentile",
        "policy_rank_pct": "live_policy_rank_pct",
        "rank_score_source": "live_rank_score_source",
        "threshold_basis_rank_score": "live_threshold_basis_rank_score",
        "threshold_basis_policy_id": "live_threshold_basis_policy_id",
        "threshold_basis_dynamic_score_threshold": "live_threshold_basis_dynamic_score_threshold",
        "threshold_basis_dynamic_ev_target": "live_threshold_basis_dynamic_ev_target",
    }
    for src, dst in live_defaults.items():
        if src in ledger.columns and dst not in ledger.columns:
            ledger[dst] = ledger[src]
    rank_source = (
        ledger.get("live_rank_score_source", pd.Series("", index=ledger.index))
        .fillna("")
        .astype(str)
    )
    threshold_basis_mask = rank_source.str.startswith("threshold_basis:")
    if threshold_basis_mask.any() and "live_threshold_basis_rank_score" in ledger.columns:
        threshold_rank = pd.to_numeric(
            ledger["live_threshold_basis_rank_score"], errors="coerce"
        )
        valid_threshold_rank = threshold_basis_mask & threshold_rank.notna()
        # Threshold-basis policies intentionally override the legacy
        # policy-rank CDF.  Keep the old CDF columns for diagnostics, but make
        # the live policy rank used by this parity checker match the actual
        # deployed gate/rank value.
        ledger.loc[valid_threshold_rank, "live_policy_rank_pct"] = threshold_rank.loc[
            valid_threshold_rank
        ]
        ledger.loc[valid_threshold_rank, "live_rank_percentile"] = threshold_rank.loc[
            valid_threshold_rank
        ]

    trades = _read_table(trades_path)
    if trades.empty:
        return ledger
    trades = trades.copy()
    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades = trades[trades.get("lifecycle_event", "").astype(str).isin(["entry_placed", "entry_rejected"])]
    if trades.empty:
        return ledger
    trades["symbol"] = trades["symbol"].map(_normalise_symbol)
    trades["strategy_core"] = trades["strategy_id"].map(strategy_core_id)
    keep_cols = [
        c
        for c in (
            "timestamp",
            "symbol",
            "side",
            "strategy_core",
            "base_pred",
            "meta_pred",
            "calibrated_score",
            "rank_percentile",
            "policy_rank_pct",
            "rank_score_source",
        )
        if c in trades.columns
    ]
    trades = trades[keep_cols].sort_values("timestamp")

    merged_rows: list[dict[str, Any]] = []
    for _, row in ledger.iterrows():
        out = row.to_dict()
        mask = (
            trades["symbol"].eq(row["symbol"])
            & trades["side"].astype(str).str.lower().eq(str(row["side"]).lower())
            & trades["strategy_core"].eq(row["strategy_core"])
        )
        candidates = trades.loc[mask].copy()
        if not candidates.empty and pd.notna(row.get("decision_ts")):
            candidates["_dt"] = (
                candidates["timestamp"] - row["decision_ts"]
            ).abs()
            candidates = candidates[candidates["_dt"] <= pd.Timedelta("15min")]
            if not candidates.empty:
                trade_row = candidates.sort_values("_dt").iloc[0]
                for col in (
                    "base_pred",
                    "meta_pred",
                    "calibrated_score",
                    "rank_percentile",
                    "policy_rank_pct",
                    "rank_score_source",
                ):
                    if col in trade_row:
                        dst = f"live_{col}"
                        # The prediction ledger is the source of truth for
                        # model and rank values.  Trade logs may contain stale
                        # or execution-path copies; use them only to fill older
                        # ledgers that did not record the live fields.
                        if not _is_present(out.get(dst)):
                            out[dst] = trade_row[col]
        merged_rows.append(out)
    return pd.DataFrame(merged_rows)


def _margin_cache_quote_symbols(live_quote_currency: str = "USDC") -> set[str]:
    """Best-effort offline mirror of the live margin universe resolver."""
    quote = str(live_quote_currency or "USDC").upper()
    cache_path = REPO_ROOT / "extreme_price_movements" / ".margin_universe_cache.json"
    if not cache_path.exists():
        return set()
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return set()
    out: set[str] = set()
    for row in data if isinstance(data, list) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").upper() != "TRADING":
            continue
        if str(row.get("quoteAsset") or row.get("quote") or "").upper() != quote:
            continue
        if not bool(row.get("isMarginTradingAllowed", True)):
            continue
        base = str(row.get("baseAsset") or row.get("base") or "").upper().strip()
        if base:
            out.add(f"{base}/{quote}")
    return out


def _market_data_root(
    data_root: Path,
    *,
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> Path:
    """Resolve the exchange-scoped market-data root used by live inference."""
    cfg = {
        "data_root": str(data_root),
        "exchange_id": exchange_id,
        "exchange": exchange_id,
        "market_mode": market_mode,
        "use_perps": str(market_mode).lower() == "perps",
    }
    scoped = Path(scoped_data_root(cfg))
    if (scoped / "ohlcv").exists() or not (Path(data_root) / "ohlcv").exists():
        return scoped
    return Path(data_root)


def _local_quote_symbols(
    data_root: Path,
    *,
    run_id: str | None = None,
    live_quote_currency: str = "USDC",
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> list[str]:
    out: list[str] = []
    quote = str(live_quote_currency or "USDC").upper()
    market_root = _market_data_root(
        data_root,
        market_mode=market_mode,
        exchange_id=exchange_id,
    )
    for path in sorted((market_root / "ohlcv").glob("symbol=*")):
        name = path.name.replace("symbol=", "")
        symbol = _normalise_symbol(name)
        if symbol.endswith(f"/{quote}") or symbol.endswith(f"/{quote}:{quote}"):
            out.append(symbol)
    symbols = set(out)
    if run_id:
        trained = load_trained_symbol_universe(str(data_root), str(run_id))
        if trained:
            trained_bases = symbol_bases(trained)
            symbols = {
                sym for sym in symbols if _normalise_symbol(sym).split("/", 1)[0] in trained_bases
            }
        margin_symbols = (
            set() if str(market_mode).lower() == "perps" else _margin_cache_quote_symbols(quote)
        )
        if margin_symbols:
            symbols &= margin_symbols
    return sorted(symbols)


def _live_feature_cache_symbols_for_end(
    data_root: Path,
    *,
    run_id: str,
    end_ts: pd.Timestamp,
    live_quote_currency: str = "USDC",
) -> list[str]:
    """Return the exact symbol universe from a live feature cache for ``end_ts``."""
    target = pd.Timestamp(end_ts)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    roots: list[tuple[int, Path]] = [
        (
            0,
            Path(data_root)
            / "artifacts"
            / str(run_id)
            / "live_selected_feature_latest_matrix",
        ),
        (1, Path("cache") / "inference_live_features" / str(run_id)),
    ]
    candidates: list[tuple[int, int, float, list[str]]] = []
    for priority, root in roots:
        if not root.exists():
            continue
        for meta_path in root.glob("**/meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                raw_end = (
                    meta.get("end_ts")
                    or meta.get("target_end_ts")
                    or meta.get("feature_end_ts")
                )
                if not raw_end:
                    continue
                cache_end = pd.Timestamp(raw_end)
                cache_end = (
                    cache_end.tz_localize("UTC")
                    if cache_end.tzinfo is None
                    else cache_end.tz_convert("UTC")
                )
                if cache_end != target:
                    continue
                quote = str(live_quote_currency or "USDC").upper()
                symbols = [_normalise_symbol(s) for s in (meta.get("symbols") or [])]
                symbols = sorted(
                    {
                        s
                        for s in symbols
                        if s.endswith(f"/{quote}") or s.endswith(f"/{quote}:{quote}")
                    }
                )
                if symbols:
                    candidates.append(
                        (priority, len(symbols), float(meta_path.stat().st_mtime), symbols)
                    )
            except Exception:
                continue
    if not candidates:
        return []
    # Prefer the artifact selected-feature sidecar over generic runtime caches:
    # it is the production inference universe and replay/debug runs can add
    # broader local-OHLCV caches for the same timestamp after the fact.  Within
    # the same source, use the smallest non-trivial universe so later debug
    # sidecars do not widen the batch and perturb AE/GMM/context features.
    non_trivial = [item for item in candidates if item[1] >= 25]
    return sorted(non_trivial or candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]


def _load_panel(
    *,
    data_root: Path,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> dict[str, pd.DataFrame]:
    market_root = _market_data_root(
        data_root,
        market_mode=market_mode,
        exchange_id=exchange_id,
    )
    store = PartitionedOHLCVStore(str(market_root), timeframe="1h")
    panel_fields = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        *PERP_OHLCV_EXTRA_FIELDS,
    )
    by_field: dict[str, list[pd.Series]] = {k: [] for k in panel_fields}
    for symbol in symbols:
        df = store.load(
            symbol,
            columns=None,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if df.empty:
            continue
        for field in by_field:
            if field in df.columns:
                by_field[field].append(pd.to_numeric(df[field], errors="coerce").rename(symbol))
    panel: dict[str, pd.DataFrame] = {}
    for field, series_list in by_field.items():
        if series_list:
            panel[field] = pd.concat(series_list, axis=1).sort_index()
    # Match live DataFetcher.load_panel(): orderbook/funding microdata are part
    # of the inference feature panel.  Without these fields replay silently
    # reconstructs orderbook-derived model features as zeros and reports false
    # train/live parity breaks.
    microdata = _load_microdata_panel(
        data_root=market_root,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    for key, frame in microdata.items():
        existing = panel.get(key)
        if (
            isinstance(existing, pd.DataFrame)
            and not existing.empty
            and isinstance(frame, pd.DataFrame)
            and not frame.empty
        ):
            panel[key] = frame.combine_first(existing).sort_index()
        else:
            panel[key] = frame
    return panel


def _symbol_file_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _load_microdata_panel(
    *,
    data_root: Path,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    orderbook_dir = data_root / "orderbook_hourly"
    funding_dir = data_root / "funding_hourly"
    orderbook_fields = (
        "mid",
        "best_bid",
        "best_ask",
        "bid_qty_1",
        "ask_qty_1",
        "cum_bid_qty_l10",
        "cum_ask_qty_l10",
        "cum_bid_qty_l20",
        "cum_ask_qty_l20",
        "snapshot_ts",
        "trade_count_1h",
        "buy_qty_1h",
        "sell_qty_1h",
        "notional_1h",
        "buy_notional_1h",
        "sell_notional_1h",
        "vwap_1h",
        "mean_trade_qty_1h",
        "signed_flow_imbalance_1h",
    )
    by_orderbook: dict[str, dict[str, pd.Series]] = {field: {} for field in orderbook_fields}
    by_microdata: dict[str, dict[str, pd.Series]] = {
        field: {} for field in MICRODATA_FRAME_FIELDS
    }
    idx_union = None
    start = pd.Timestamp(start_ts)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(end_ts)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    for symbol in symbols:
        key = _symbol_file_key(symbol)
        ob_path = orderbook_dir / f"{key}.parquet"
        if ob_path.exists():
            try:
                ob = pd.read_parquet(ob_path)
                ob.index = pd.to_datetime(ob.index, utc=True)
                ob = ob.loc[(ob.index >= start) & (ob.index <= end)]
            except Exception:
                ob = pd.DataFrame()
            if not ob.empty:
                idx_union = ob.index if idx_union is None else idx_union.union(ob.index)
                for field in orderbook_fields:
                    if field in ob.columns:
                        by_orderbook[field][symbol] = pd.to_numeric(
                            ob[field], errors="coerce"
                        ).astype(np.float32)
        funding_path = funding_dir / f"{key}.parquet"
        if funding_path.exists():
            try:
                fr = pd.read_parquet(funding_path)
                fr.index = pd.to_datetime(fr.index, utc=True)
                fr = fr.loc[(fr.index >= start) & (fr.index <= end)]
            except Exception:
                fr = pd.DataFrame()
            if not fr.empty:
                idx_union = fr.index if idx_union is None else idx_union.union(fr.index)
                for field in MICRODATA_FRAME_FIELDS:
                    if field in fr.columns:
                        by_microdata[field][symbol] = pd.to_numeric(
                            fr[field], errors="coerce"
                        ).astype(np.float32)
    if idx_union is None:
        return {}
    idx_union = pd.DatetimeIndex(idx_union).sort_values().unique()
    out: dict[str, pd.DataFrame] = {}
    if by_orderbook["mid"]:
        out["orderbook_hourly"] = (
            pd.DataFrame(by_orderbook["mid"]).reindex(idx_union).astype(np.float32)
        )
    for field, by_symbol in by_orderbook.items():
        if by_symbol:
            out[f"orderbook_{field}"] = (
                pd.DataFrame(by_symbol).reindex(idx_union).astype(np.float32)
            )
    for field, by_symbol in by_microdata.items():
        if by_symbol:
            out[field] = (
                pd.DataFrame(by_symbol).reindex(idx_union).ffill().astype(np.float32)
            )
    return out


def _score_row(
    *,
    row: pd.Series,
    feats: dict[str, pd.DataFrame],
    orchestrator: ModelOrchestrator,
    calibration_data: dict[str, dict[str, Any]],
    rank_store: PolicyRankReferenceStore,
    overlay_required_columns: set[str] | None = None,
    live_ae_gmm_state_payload: Mapping[str, Any] | None = None,
    feature_row_override: pd.DataFrame | None = None,
    skip_full_chain_diagnostics: bool = False,
) -> dict[str, Any]:
    symbol = str(row["symbol"])
    side = str(row["side"]).lower()
    strategy_id = str(row["strategy_id"])
    core_strategy_id = strategy_core_id(strategy_id)
    model_strategy_id = strategy_id
    if side in {"long", "short"} and core_strategy_id and not str(strategy_id).startswith(
        f"{side}_"
    ):
        model_strategy_id = f"{side}_{core_strategy_id}"
    ts = pd.Timestamp(row["signal_bar_ts"])
    override_supplied = isinstance(feature_row_override, pd.DataFrame)
    feature_row = (
        feature_row_override.copy()
        if override_supplied
        else get_features_for_candidates(feats, [symbol], ts=ts)
    )
    if not feature_row.empty:
        feature_row = feature_row.copy()
        feature_row["side"] = np.float32(
            1.0 if str(side).lower().startswith("long") else -1.0
        )
        feature_row["side_name"] = str(side).lower()
        overlay_cols = set(str(c) for c in (overlay_required_columns or set()) if str(c))
        if overlay_cols and not override_supplied:
            feature_row = materialize_live_source_regime_features(
                feature_row,
                side=side,
                signal_bar_ts=ts,
                required_columns=overlay_cols,
            )
            feature_row = materialize_live_ae_gmm_features(
                feature_row,
                side=side,
                signal_bar_ts=ts,
                required_columns=overlay_cols,
                state_payload=live_ae_gmm_state_payload,
            )
    out = {
        "decision_ts": row.get("decision_ts"),
        "signal_bar_ts": ts,
        "symbol": symbol,
        "side": side,
        "strategy_id": strategy_id,
        "live_feature_contract_hash": row.get("feature_contract_hash"),
        "live_feature_transform_contract_hash": row.get(
            "feature_transform_contract_hash"
        ),
        "live_base_pred": _safe_float(row.get("live_base_pred")),
        "live_meta_pred": _safe_float(row.get("live_meta_pred", row.get("raw_prediction_score"))),
        "live_calibrated_score": _safe_float(row.get("live_calibrated_score")),
        "live_rank_percentile": _safe_float(row.get("live_rank_percentile", row.get("normalized_rank_score"))),
        "live_policy_rank_pct": _safe_float(row.get("live_policy_rank_pct", row.get("policy_rank_pct"))),
        "live_rank_score_source": row.get("live_rank_score_source", row.get("rank_score_source")),
        "live_threshold_basis_rank_score": _safe_float(
            row.get("live_threshold_basis_rank_score", row.get("threshold_basis_rank_score"))
        ),
        "live_threshold_basis_policy_id": row.get(
            "live_threshold_basis_policy_id", row.get("threshold_basis_policy_id")
        ),
        "live_threshold_basis_dynamic_score_threshold": _safe_float(
            row.get(
                "live_threshold_basis_dynamic_score_threshold",
                row.get("threshold_basis_dynamic_score_threshold"),
            )
        ),
        "live_threshold_basis_dynamic_ev_target": _safe_float(
            row.get(
                "live_threshold_basis_dynamic_ev_target",
                row.get("threshold_basis_dynamic_ev_target"),
            )
        ),
        "replay_feature_cols": int(feature_row.shape[1]) if not feature_row.empty else 0,
        "replay_missing_features": bool(feature_row.empty),
    }
    threshold_basis_mode = str(out["live_rank_score_source"] or "").startswith(
        "threshold_basis:"
    )
    if threshold_basis_mode and np.isfinite(out["live_threshold_basis_rank_score"]):
        out["threshold_basis_rank_internal_delta"] = (
            out["live_rank_percentile"] - out["live_threshold_basis_rank_score"]
        )
        out["threshold_basis_policy_rank_internal_delta"] = (
            out["live_policy_rank_pct"] - out["live_threshold_basis_rank_score"]
        )
    else:
        out["threshold_basis_rank_internal_delta"] = float("nan")
        out["threshold_basis_policy_rank_internal_delta"] = float("nan")
    live_cal_threshold = float("nan")
    if not np.isfinite(out["live_calibrated_score"]) and np.isfinite(
        out["live_meta_pred"]
    ):
        live_cal, live_cal_threshold = calibrated_score_and_threshold(
            raw_score=out["live_meta_pred"],
            strategy_id=core_strategy_id,
            calibration_data=calibration_data,
            default_threshold=1.0,
        )
        out["live_calibrated_score"] = live_cal
        out["live_calibration_threshold"] = live_cal_threshold
    if feature_row.empty:
        return out
    base_delta_summary = _feature_value_delta_summary(
        logged_values_raw=row.get("base_model_feature_values_json"),
        feature_row=feature_row,
        symbol=symbol,
    )
    meta_delta_summary = _feature_value_delta_summary(
        logged_values_raw=row.get("meta_model_feature_values_json"),
        feature_row=feature_row,
        symbol=symbol,
    )
    base_synth_delta_summary = _live_synthesized_feature_delta_summary(
        logged_values_raw=row.get("base_model_feature_values_json"),
        feature_row=feature_row,
        symbol=symbol,
    )
    meta_synth_delta_summary = _live_synthesized_feature_delta_summary(
        logged_values_raw=row.get("meta_model_feature_values_json"),
        feature_row=feature_row,
        symbol=symbol,
    )
    out.update(
        {
            "base_feature_value_common_count": base_delta_summary["count"],
            "base_feature_value_max_abs_delta": base_delta_summary["max_abs"],
            "base_feature_value_mean_abs_delta": base_delta_summary["mean_abs"],
            "base_feature_value_worst_feature": base_delta_summary["worst_feature"],
            "base_feature_value_worst_is_live_synthesized": bool(
                _is_live_synthesized_feature_key(
                    base_delta_summary["worst_feature"]
                )
            ),
            "base_live_synth_feature_value_common_count": base_synth_delta_summary[
                "count"
            ],
            "base_live_synth_feature_value_max_abs_delta": base_synth_delta_summary[
                "max_abs"
            ],
            "base_live_synth_feature_value_mean_abs_delta": base_synth_delta_summary[
                "mean_abs"
            ],
            "base_live_synth_feature_value_worst_feature": base_synth_delta_summary[
                "worst_feature"
            ],
            "meta_feature_value_common_count": meta_delta_summary["count"],
            "meta_feature_value_max_abs_delta": meta_delta_summary["max_abs"],
            "meta_feature_value_mean_abs_delta": meta_delta_summary["mean_abs"],
            "meta_feature_value_worst_feature": meta_delta_summary["worst_feature"],
            "meta_feature_value_worst_is_live_synthesized": bool(
                _is_live_synthesized_feature_key(
                    meta_delta_summary["worst_feature"]
                )
            ),
            "meta_live_synth_feature_value_common_count": meta_synth_delta_summary[
                "count"
            ],
            "meta_live_synth_feature_value_max_abs_delta": meta_synth_delta_summary[
                "max_abs"
            ],
            "meta_live_synth_feature_value_mean_abs_delta": meta_synth_delta_summary[
                "mean_abs"
            ],
            "meta_live_synth_feature_value_worst_feature": meta_synth_delta_summary[
                "worst_feature"
            ],
        }
    )
    logged_base_frame = _logged_values_frame(
        row.get("base_model_feature_values_json"),
        symbol=symbol,
    )
    logged_meta_frame = _logged_values_frame(
        row.get("meta_model_feature_values_json"),
        symbol=symbol,
    )
    try:
        if not logged_base_frame.empty:
            logged_base_pred = orchestrator.predict_alpha(
                logged_base_frame,
                side,
                model_strategy_id,
            )
            out["logged_base_input_pred"] = (
                _safe_float(logged_base_pred.iloc[0])
                if isinstance(logged_base_pred, pd.Series)
                and not logged_base_pred.empty
                else float("nan")
            )
        else:
            out["logged_base_input_pred"] = float("nan")
    except Exception:
        out["logged_base_input_pred"] = float("nan")
    try:
        if not logged_meta_frame.empty:
            out["logged_meta_input_pred"] = _predict_exact_logged_meta_input(
                orchestrator=orchestrator,
                side=side,
                strategy_id=model_strategy_id,
                logged_meta_frame=logged_meta_frame,
            )
        else:
            out["logged_meta_input_pred"] = float("nan")
    except Exception:
        out["logged_meta_input_pred"] = float("nan")
    out["logged_base_input_pred_delta"] = (
        out["logged_base_input_pred"] - out["live_base_pred"]
    )
    out["logged_meta_input_pred_delta"] = (
        out["logged_meta_input_pred"] - out["live_meta_pred"]
    )
    logged_meta_cal, logged_meta_cal_threshold = calibrated_score_and_threshold(
        raw_score=out["logged_meta_input_pred"],
        strategy_id=core_strategy_id,
        calibration_data=calibration_data,
        default_threshold=1.0,
    )
    logged_meta_rank = rank_store.lookup(
        strategy_id=strategy_id,
        side=side,
        calibrated_score=logged_meta_cal,
    )
    out["logged_meta_input_calibrated_score"] = logged_meta_cal
    out["logged_meta_input_calibration_threshold"] = logged_meta_cal_threshold
    out["logged_meta_input_policy_rank_pct"] = logged_meta_rank.policy_rank_pct
    out["logged_meta_input_calibrated_score_delta"] = (
        logged_meta_cal - out["live_calibrated_score"]
    )
    out["logged_meta_input_rank_percentile_delta"] = (
        logged_meta_rank.policy_rank_pct - out["live_policy_rank_pct"]
    )
    try:
        alpha_pred = orchestrator.predict_alpha(
            feature_row.loc[[symbol]],
            side,
            model_strategy_id,
        )
        replay_base_direct = (
            _safe_float(alpha_pred.iloc[0])
            if isinstance(alpha_pred, pd.Series) and not alpha_pred.empty
            else float("nan")
        )
    except Exception:
        replay_base_direct = float("nan")
    chain: dict[str, Any] = {}
    replay_base = replay_base_direct
    replay_meta = float("nan")
    full_chain_meta = float("nan")
    full_chain_cal = float("nan")
    full_chain_cal_threshold = float("nan")
    full_chain_policy_rank_pct = float("nan")
    replay_meta_input_delta_summary = {
        "count": 0,
        "max_abs": float("nan"),
        "mean_abs": float("nan"),
        "worst_feature": "",
    }
    meta_replay_source = "logged_batch_meta_input_skip_full_chain"
    can_skip_full_chain = bool(skip_full_chain_diagnostics) and np.isfinite(
        out.get("logged_meta_input_pred", np.nan)
    )
    if not can_skip_full_chain:
        chain = orchestrator.run_full_chain(
            symbol,
            side,
            feature_row.loc[[symbol]],
            kind=model_strategy_id,
        )
        replay_base = _safe_float(chain.get("base_pred"))
        if not np.isfinite(replay_base):
            replay_base = replay_base_direct
        replay_meta = _safe_float(chain.get("meta_pred"))
        replay_meta_model_input = getattr(orchestrator, "_last_meta_model_input", None)
        replay_meta_input_delta_summary = (
            _feature_value_delta_summary(
                logged_values_raw=row.get("meta_model_feature_values_json"),
                feature_row=replay_meta_model_input,
                symbol=symbol,
            )
            if isinstance(replay_meta_model_input, pd.DataFrame)
            else replay_meta_input_delta_summary
        )
        full_chain_meta = replay_meta
        full_chain_cal, full_chain_cal_threshold = calibrated_score_and_threshold(
            raw_score=full_chain_meta,
            strategy_id=core_strategy_id,
            calibration_data=calibration_data,
            default_threshold=1.0,
        )
        full_chain_rank = rank_store.lookup(
            strategy_id=strategy_id,
            side=side,
            calibrated_score=full_chain_cal,
        )
        full_chain_policy_rank_pct = full_chain_rank.policy_rank_pct
        meta_replay_source = "single_row_full_chain"
    if np.isfinite(out.get("logged_meta_input_pred", np.nan)):
        # Live performs meta prediction in batch and records the exact selected
        # meta-model matrix per decision. Some diagnostic meta inputs, notably
        # feature_drift_psi_core, are batch-context dependent; using the logged
        # selected matrix is the exact training/inference adapter parity path.
        replay_meta = out["logged_meta_input_pred"]
        meta_replay_source = "logged_batch_meta_input"
    replay_cal, replay_cal_threshold = calibrated_score_and_threshold(
        raw_score=replay_meta,
        strategy_id=core_strategy_id,
        calibration_data=calibration_data,
        default_threshold=1.0,
    )
    rank_lookup = rank_store.lookup(
        strategy_id=strategy_id,
        side=side,
        calibrated_score=replay_cal,
    )
    out.update(
        {
            "replay_action": chain.get("action"),
            "replay_reason": chain.get("reason"),
            "replay_model_strategy_id": model_strategy_id,
            "replay_base_pred": replay_base,
            "replay_meta_pred": replay_meta,
            "replay_calibrated_score": replay_cal,
            "replay_calibration_threshold": replay_cal_threshold,
            "replay_policy_rank_pct": rank_lookup.policy_rank_pct,
            "replay_policy_rank_reference_n": rank_lookup.n_rows,
            "replay_policy_rank_reference_source": rank_lookup.source,
            "replay_meta_input_source": meta_replay_source,
            "full_chain_meta_pred": full_chain_meta,
            "full_chain_calibrated_score": full_chain_cal,
            "full_chain_calibration_threshold": full_chain_cal_threshold,
            "full_chain_policy_rank_pct": full_chain_policy_rank_pct,
            "full_chain_meta_pred_delta": full_chain_meta - out["live_meta_pred"],
            "full_chain_rank_percentile_delta": full_chain_policy_rank_pct
            - out["live_policy_rank_pct"],
            "replay_meta_model_input_common_count": replay_meta_input_delta_summary[
                "count"
            ],
            "replay_meta_model_input_max_abs_delta": replay_meta_input_delta_summary[
                "max_abs"
            ],
            "replay_meta_model_input_mean_abs_delta": replay_meta_input_delta_summary[
                "mean_abs"
            ],
            "replay_meta_model_input_worst_feature": replay_meta_input_delta_summary[
                "worst_feature"
            ],
            "base_pred_delta": replay_base - out["live_base_pred"],
            "meta_pred_delta": replay_meta - out["live_meta_pred"],
            "calibrated_score_delta": replay_cal - out["live_calibrated_score"],
            "rank_percentile_delta": rank_lookup.policy_rank_pct
            - out["live_policy_rank_pct"],
        }
            )
    return out


def _batched_replay_feature_rows(
    *,
    feats: dict[str, pd.DataFrame],
    group: pd.DataFrame,
    signal_bar_ts: pd.Timestamp,
    overlay_required_columns: set[str] | None,
    live_ae_gmm_state_payload: Mapping[str, Any] | None,
    artifact_data_root: Path | None = None,
    run_id: str | None = None,
) -> dict[Any, pd.DataFrame]:
    """Materialize live synthetic overlays once per side/timestamp batch.

    Production inference creates the side candidate feature matrix first, then
    appends source-regime and frozen AE/GMM overlays to that batch.  Replaying
    each row independently changes row-order-dependent AE/GMM deltas such as
    cluster speed and acceleration.  This helper mirrors the production shape
    for the rows available in the live prediction ledger.
    """

    if not isinstance(group, pd.DataFrame) or group.empty:
        return {}
    overlay_cols = set(str(c) for c in (overlay_required_columns or set()) if str(c))
    if not overlay_cols:
        return {}
    out: dict[Any, pd.DataFrame] = {}
    side_series = group.get("side", pd.Series("", index=group.index)).fillna("").astype(str)
    for side, side_group in group.groupby(side_series.str.lower(), sort=False):
        if str(side) not in {"long", "short"} or side_group.empty:
            continue
        symbols = [
            _normalise_symbol(s)
            for s in side_group["symbol"].dropna().astype(str).tolist()
        ]
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            continue
        feature_batch = _load_persisted_live_candidate_feature_matrix(
            artifact_data_root=artifact_data_root,
            run_id=run_id,
            signal_bar_ts=signal_bar_ts,
            side=side,
        )
        persisted_batch = isinstance(feature_batch, pd.DataFrame) and not feature_batch.empty
        if not persisted_batch:
            feature_batch = get_features_for_candidates(feats, symbols, ts=signal_bar_ts)
        if not isinstance(feature_batch, pd.DataFrame) or feature_batch.empty:
            continue
        feature_batch = feature_batch.copy()
        if not persisted_batch:
            feature_batch["side"] = np.float32(1.0 if side == "long" else -1.0)
            feature_batch["side_name"] = str(side)
            feature_batch = materialize_live_source_regime_features(
                feature_batch,
                side=side,
                signal_bar_ts=signal_bar_ts,
                required_columns=overlay_cols,
            )
            feature_batch = materialize_live_ae_gmm_features(
                feature_batch,
                side=side,
                signal_bar_ts=signal_bar_ts,
                required_columns=overlay_cols,
                state_payload=live_ae_gmm_state_payload,
            )
        for idx, row in side_group.iterrows():
            symbol = _normalise_symbol(str(row.get("symbol", "")))
            if symbol in feature_batch.index:
                out[idx] = feature_batch.loc[[symbol]].copy()
    return out


def _load_persisted_live_candidate_feature_matrix(
    *,
    artifact_data_root: Path | None,
    run_id: str | None,
    signal_bar_ts: pd.Timestamp,
    side: str,
) -> pd.DataFrame:
    if artifact_data_root is None or not run_id:
        return pd.DataFrame()
    try:
        ts = pd.Timestamp(signal_bar_ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return pd.DataFrame()
    side_s = "short" if str(side).lower().startswith("short") else "long"
    root = (
        Path(artifact_data_root)
        / "artifacts"
        / str(run_id)
        / "live_candidate_feature_matrix"
        / side_s
    )
    if not root.exists():
        return pd.DataFrame()
    candidates: list[tuple[float, Path]] = []
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            raw_ts = meta.get("signal_bar_ts") or meta.get("end_ts")
            if not raw_ts:
                continue
            meta_ts = pd.Timestamp(raw_ts)
            meta_ts = meta_ts.tz_localize("UTC") if meta_ts.tzinfo is None else meta_ts.tz_convert("UTC")
            if meta_ts != ts:
                continue
            data_path = meta_path.with_name("data.parquet")
            if data_path.exists():
                candidates.append((float(data_path.stat().st_mtime), data_path))
        except Exception:
            continue
    if not candidates:
        return pd.DataFrame()
    data_path = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    try:
        frame = pd.read_parquet(data_path)
        frame.index = frame.index.astype(str)
        return frame
    except Exception:
        return pd.DataFrame()


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "replay_rows_with_features": int((~frame.get("replay_missing_features", True).astype(bool)).sum())
        if not frame.empty
        else 0,
        "policy_rank_reference_available_rows": int(
            pd.to_numeric(frame.get("replay_policy_rank_reference_n"), errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not frame.empty and "replay_policy_rank_reference_n" in frame
        else 0,
    }
    for col in (
        "base_pred_delta",
        "meta_pred_delta",
        "calibrated_score_delta",
        "rank_percentile_delta",
        "logged_base_input_pred_delta",
        "logged_meta_input_pred_delta",
        "logged_meta_input_calibrated_score_delta",
        "logged_meta_input_rank_percentile_delta",
        "threshold_basis_rank_internal_delta",
        "threshold_basis_policy_rank_internal_delta",
    ):
        if col not in frame:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        summary[col] = {
            "n": int(vals.notna().sum()),
            "mean_abs": float(vals.abs().mean()) if vals.notna().any() else None,
            "max_abs": float(vals.abs().max()) if vals.notna().any() else None,
            "mean": float(vals.mean()) if vals.notna().any() else None,
        }
    if not frame.empty and "live_rank_score_source" in frame:
        summary["live_rank_score_source_counts"] = {
            str(k): int(v)
            for k, v in frame["live_rank_score_source"]
            .fillna("")
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }
    if not frame.empty:
        synth_summary: dict[str, Any] = {}
        for scope in ("base", "meta"):
            max_col = f"{scope}_live_synth_feature_value_max_abs_delta"
            worst_col = f"{scope}_live_synth_feature_value_worst_feature"
            if max_col not in frame:
                continue
            vals = pd.to_numeric(frame[max_col], errors="coerce")
            drift_rows = int(vals.gt(1e-7).sum())
            top_features: dict[str, int] = {}
            if worst_col in frame and drift_rows:
                top_features = {
                    str(k): int(v)
                    for k, v in frame.loc[vals.gt(1e-7), worst_col]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .to_dict()
                    .items()
                    if str(k)
                }
            synth_summary[scope] = {
                "rows_gt_1e-7": drift_rows,
                "max_abs": float(vals.abs().max()) if vals.notna().any() else None,
                "top_worst_features": top_features,
            }
        if synth_summary:
            summary["live_synthesized_feature_reconstruction_drift"] = synth_summary
    return summary


def _parity_failures(
    frame: pd.DataFrame,
    *,
    tolerance: float,
    prediction_tolerance: float | None = None,
    require_policy_rank_reference: bool = False,
    require_live_values: bool = False,
    parity_source: str = "replay",
) -> list[str]:
    failures: list[str] = []
    if frame.empty:
        failures.append("no_rows")
        return failures
    parity_source = str(parity_source or "replay")
    if require_policy_rank_reference:
        raw_ref_n = (
            frame["replay_policy_rank_reference_n"]
            if "replay_policy_rank_reference_n" in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        ref_n = pd.to_numeric(raw_ref_n, errors="coerce").fillna(0)
        missing = int(ref_n.le(0).sum())
        if missing:
            failures.append(f"missing_policy_rank_reference_rows={missing}")
    if require_live_values:
        required_cols = (
            "live_base_pred",
            "live_meta_pred",
            "live_calibrated_score",
            "live_policy_rank_pct",
        )
        for col in required_cols:
            raw_vals = (
                frame[col]
                if col in frame.columns
                else pd.Series(np.nan, index=frame.index)
            )
            vals = pd.to_numeric(raw_vals, errors="coerce")
            missing = int(vals.isna().sum())
            if missing:
                failures.append(f"missing_{col}_rows={missing}")
        if "expected_feature_transform_contract_hash" in frame.columns:
            expected = frame["expected_feature_transform_contract_hash"].astype(str)
            live = (
                frame["live_feature_transform_contract_hash"].astype(str)
                if "live_feature_transform_contract_hash" in frame.columns
                else pd.Series("", index=frame.index)
            )
            missing_hash = int(live.isin({"", "None", "nan", "NaN"}).sum())
            if missing_hash:
                failures.append(
                    f"missing_live_feature_transform_contract_hash_rows={missing_hash}"
                )
            mismatch = int((live != expected).sum())
            if mismatch:
                failures.append(
                    f"feature_transform_contract_hash_mismatch_rows={mismatch}"
                )
    if parity_source == "logged-input":
        required_replay_cols = [
            "logged_base_input_pred",
            "logged_meta_input_pred",
        ]
        delta_cols = [
            "logged_base_input_pred_delta",
            "logged_meta_input_pred_delta",
        ]
    else:
        required_replay_cols = [
            "replay_base_pred",
            "replay_meta_pred",
        ]
        delta_cols = [
            "base_pred_delta",
            "meta_pred_delta",
        ]
    rank_source = (
        frame.get("live_rank_score_source", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    )
    threshold_basis_mask = rank_source.str.startswith("threshold_basis:")
    if bool(threshold_basis_mask.all()):
        required_replay_cols.extend(
            [
                "live_calibrated_score",
                "live_threshold_basis_rank_score",
                "live_threshold_basis_dynamic_score_threshold",
            ]
        )
        delta_cols.extend(
            [
                "threshold_basis_rank_internal_delta",
                "threshold_basis_policy_rank_internal_delta",
            ]
        )
    elif threshold_basis_mask.any():
        failures.append(
            f"mixed_threshold_basis_and_legacy_rank_rows={int(threshold_basis_mask.sum())}/{len(frame)}"
        )
        required_replay_cols.extend(
            [
                "live_calibrated_score",
                "replay_calibrated_score",
                "replay_policy_rank_pct",
            ]
        )
        delta_cols.extend(["calibrated_score_delta", "rank_percentile_delta"])
    else:
        required_replay_cols.extend(
            [
                "replay_calibrated_score",
                "replay_policy_rank_pct",
            ]
        )
        delta_cols.extend(["calibrated_score_delta", "rank_percentile_delta"])
    for col in required_replay_cols:
        raw_vals = (
            frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
        )
        vals = pd.to_numeric(raw_vals, errors="coerce")
        missing = int(vals.isna().sum())
        if missing:
            failures.append(f"missing_{col}_rows={missing}")
    pred_tol = float(prediction_tolerance) if prediction_tolerance is not None else max(float(tolerance), 1e-7)
    for col in delta_cols:
        if col not in frame:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").abs()
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            failures.append(f"{col}_finite_rows=0")
            continue
        missing_delta = len(frame) - int(finite.size)
        if missing_delta:
            failures.append(f"{col}_missing_rows={missing_delta}")
        max_abs = float(finite.max())
        col_tol = pred_tol if "pred_delta" in str(col) else float(tolerance)
        if max_abs > col_tol:
            failures.append(f"{col}_max_abs={max_abs:.12g}>tol={col_tol:.12g}")
    return failures


def _model_runtime_cfg(
    *,
    model_bundle: Mapping[str, Any],
    feature_runtime_cfg: Mapping[str, Any],
    disable_model_diagnostics: bool = False,
    disable_model_timing: bool = False,
) -> dict[str, Any]:
    runtime_cfg = dict(feature_runtime_cfg or {})
    runtime_cfg["model_bundle"] = model_bundle
    if disable_model_diagnostics:
        runtime_cfg["inference_lgbm_internal_diagnostics_enabled"] = False
    if disable_model_timing:
        runtime_cfg["inference_model_timing_enabled"] = False
    return runtime_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument(
        "--artifact-data-root",
        type=Path,
        default=None,
        help="Artifact/model root. Defaults to --data-root.",
    )
    parser.add_argument(
        "--market-mode",
        choices=("spot", "perps"),
        default=None,
        help="Inference feature/config mode. Defaults to perps when roots contain 'perp', otherwise spot.",
    )
    parser.add_argument(
        "--exchange-id",
        default="krakenfutures",
        help="Exchange component used for exchange-scoped market data.",
    )
    parser.add_argument(
        "--live-quote-currency",
        default=None,
        help="Live quote currency used for local symbol discovery. Defaults to USD in perps mode, otherwise USDC.",
    )
    parser.add_argument(
        "--live-feature-source-run-id",
        default=None,
        help=(
            "Training-path selected-feature run id to use as authoritative "
            "model feature source. Defaults to EPM_LIVE_FEATURE_SOURCE_RUN_ID."
        ),
    )
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument("--ledger", default="data/live_state/prediction_ledger.parquet", type=Path)
    parser.add_argument("--trades", default="inference_trades.csv", type=Path)
    parser.add_argument("--max-rows", default=24, type=int)
    parser.add_argument(
        "--lookback-hours",
        default=24 * 60,
        type=int,
        help="Feature lookback hours. Default matches live inference.",
    )
    parser.add_argument("--max-symbols", default=0, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--decision-start",
        default=None,
        help="Only replay decisions with decision_ts at or after this timestamp.",
    )
    parser.add_argument(
        "--require-rank-source",
        default=None,
        help="Only replay decisions whose logged rank_score_source matches exactly.",
    )
    parser.add_argument(
        "--min-rows",
        default=1,
        type=int,
        help="Minimum replay rows required before reporting success.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if replay/live prediction deltas exceed tolerance.",
    )
    parser.add_argument(
        "--parity-source",
        choices=("replay", "logged-input"),
        default="replay",
        help=(
            "Prediction source used by --fail-on-mismatch. 'replay' checks "
            "selected-cache reconstruction; 'logged-input' checks the exact "
            "model-input values recorded in the live ledger."
        ),
    )
    parser.add_argument(
        "--tolerance",
        default=1e-9,
        type=float,
        help="Absolute tolerance used by --fail-on-mismatch for rank/calibration deltas.",
    )
    parser.add_argument(
        "--prediction-tolerance",
        default=1e-7,
        type=float,
        help=(
            "Absolute tolerance for raw model prediction parity. Kept separate "
            "from rank/calibration tolerance because native model paths can "
            "differ at float32-scale without changing decisions."
        ),
    )
    parser.add_argument(
        "--require-policy-rank-reference",
        action="store_true",
        help="Fail if any replayed row lacks a policy-rank-reference CDF.",
    )
    parser.add_argument(
        "--require-live-values",
        action="store_true",
        help="Fail if base/meta/calibrated/policy-rank live values are absent.",
    )
    parser.add_argument(
        "--disable-model-diagnostics",
        action="store_true",
        help="Disable expensive model-internal diagnostics during replay scoring.",
    )
    parser.add_argument(
        "--disable-model-timing",
        action="store_true",
        help="Disable per-model timing instrumentation during replay scoring.",
    )
    parser.add_argument(
        "--skip-full-chain-diagnostics",
        action="store_true",
        help=(
            "Skip per-row full-chain diagnostic scoring when exact logged batch "
            "meta inputs are available; base predictions are still replayed."
        ),
    )
    parser.add_argument(
        "--batch-by-signal-bar-cache",
        action="store_true",
        help=(
            "Load latest-only live feature caches separately for each signal_bar_ts "
            "instead of using the max replay timestamp for all rows."
        ),
    )
    args = parser.parse_args()
    artifact_data_root = args.artifact_data_root or args.data_root
    market_mode = args.market_mode or (
        "perps"
        if "perp" in f"{args.data_root} {artifact_data_root}".lower()
        else "spot"
    )
    live_quote_currency = (
        str(args.live_quote_currency).upper()
        if args.live_quote_currency
        else ("USD" if market_mode == "perps" else "USDC")
    )

    decisions = _load_recent_decisions(
        ledger_path=args.ledger,
        trades_path=args.trades,
        max_rows=max(1, int(args.max_rows)),
        decision_start=args.decision_start,
        require_rank_source=args.require_rank_source,
    )
    if decisions.empty:
        raise SystemExit("No live decisions matched the replay filters.")
    if len(decisions) < int(args.min_rows):
        raise SystemExit(
            f"Only {len(decisions)} live decisions matched the replay filters; "
            f"min_rows={int(args.min_rows)}."
        )

    min_ts = pd.Timestamp(decisions["signal_bar_ts"].min())
    max_ts = pd.Timestamp(decisions["signal_bar_ts"].max())
    max_panel_ts = max_ts
    symbols = _live_feature_cache_symbols_for_end(
        args.data_root,
        run_id=args.run_id,
        end_ts=max_ts,
        live_quote_currency=live_quote_currency,
    )
    symbol_source = "live_feature_cache"
    if not symbols:
        symbols = _local_quote_symbols(
            args.data_root,
            run_id=args.run_id,
            live_quote_currency=live_quote_currency,
            market_mode=market_mode,
            exchange_id=args.exchange_id,
        )
        symbol_source = "local_ohlcv"
    if args.max_symbols and args.max_symbols > 0:
        decision_symbols = list(decisions["symbol"].dropna().map(_normalise_symbol).unique())
        extra = [s for s in symbols if s not in decision_symbols]
        symbols = sorted(set(decision_symbols + extra[: max(0, args.max_symbols - len(decision_symbols))]))
    start_ts = min_ts - pd.Timedelta(hours=int(args.lookback_hours))
    end_ts = max_panel_ts
    print(
        f"Replaying {len(decisions)} live decisions from {min_ts} to {max_ts}; "
        f"loading {len(symbols)} {live_quote_currency} symbols from {start_ts} to {end_ts} "
        f"(symbol_source={symbol_source})."
    )
    panel = _load_panel(
        data_root=args.data_root,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        market_mode=market_mode,
        exchange_id=args.exchange_id,
    )
    if not panel or "close" not in panel:
        raise SystemExit("No OHLCV panel could be loaded.")

    state = load_full_state(args.run_id, str(artifact_data_root))
    accepted_strategy_keys: set[str] = set()
    for _, decision in decisions.iterrows():
        side = str(decision.get("side") or "").lower()
        core = strategy_core_id(str(decision.get("strategy_id") or ""))
        if core:
            accepted_strategy_keys.add(core)
            if side in {"long", "short"}:
                accepted_strategy_keys.add(f"{side}_{core}")
                accepted_strategy_keys.add(f"{side}_{core}_clf")
    logged_feature_keys = _logged_decision_feature_keys(decisions)
    # Replay must materialize the same feature context as live inference, not
    # only the narrower set of logged model inputs. Some live-derived features
    # share rolling/orderbook primitives, so narrowing this request can change
    # the feature generation path and create false train/live parity breaks.
    selected_feature_keys = set(
        str(c)
        for c in get_inference_required_feature_keys(state, accepted_strategy_keys)
        if str(c)
    )
    overlay_required_columns = set(selected_feature_keys).union(
        str(c) for c in logged_feature_keys if str(c)
    )
    required_keys = raw_required_feature_keys(selected_feature_keys)
    required_keys |= raw_required_feature_keys(logged_feature_keys)
    try:
        mask_rows = _load_lgbm_strategy_mask_rows(
            str(artifact_data_root),
            args.run_id,
            market_mode=market_mode,
        )
        required_keys |= set(_lgbm_mask_required_feature_keys(mask_rows))
    except Exception:
        pass
    try:
        feature_cfg = load_inference_config(
            data_root=str(artifact_data_root),
            run_id=args.run_id,
            market_mode=market_mode,
        )
    except Exception:
        feature_cfg = dict(CFG)
    runtime_cfg = dict(feature_cfg.get("runtime_cfg") or {})
    runtime_cfg["use_perps"] = market_mode == "perps"
    runtime_cfg["market_mode"] = market_mode
    runtime_cfg["data_root"] = str(args.data_root)
    runtime_cfg["artifact_data_root"] = str(artifact_data_root)
    runtime_cfg["offline_feature_data_root"] = str(artifact_data_root)
    runtime_cfg["live_data_root"] = str(args.data_root)
    live_feature_source_run_id = (
        args.live_feature_source_run_id
        or os.getenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID")
    )
    runtime_cfg["live_feature_cache_namespace"] = "model"
    runtime_cfg["live_feature_prefer_offline_cache"] = True
    runtime_cfg["live_feature_offline_cache_enabled"] = True
    runtime_cfg["live_model_feature_store_strict"] = live_model_feature_store_strict(
        feature_cfg
    )
    runtime_cfg["live_feature_return_latest_only"] = True
    if live_feature_source_run_id:
        runtime_cfg["live_feature_source_run_id"] = str(live_feature_source_run_id)
    state_bundle = state.get("bundle", {}) if isinstance(state.get("bundle"), dict) else {}
    runtime_cfg.setdefault("bundle", state_bundle)
    for key in (
        "feature_transform_contract",
        "feature_transform_contract_hash",
        "feature_transform_manifest",
    ):
        value = state.get(key)
        if value is None:
            value = state_bundle.get(key)
        if value is not None:
            feature_cfg[key] = value
            runtime_cfg[key] = value
    feature_cfg["runtime_cfg"] = runtime_cfg
    live_ae_gmm_state_payload = load_live_ae_gmm_state_payload(
        str(artifact_data_root),
        args.run_id,
    )
    live_ae_gmm_input_columns = live_ae_gmm_input_feature_columns(
        live_ae_gmm_state_payload
    )
    if live_ae_gmm_input_columns:
        required_keys |= raw_required_feature_keys(live_ae_gmm_input_columns)
        overlay_required_columns.update(str(c) for c in live_ae_gmm_input_columns if str(c))
        print(
            "Replay frozen AE/GMM state loaded: "
            f"state={live_ae_gmm_state_payload.get('state_path', '')} "
            f"input_features={len(live_ae_gmm_input_columns)} "
            f"raw_required_features={len(required_keys)} "
            f"overlay_required_columns={len(overlay_required_columns)}"
        )
    orchestrator = ModelOrchestrator(
        state,
        runtime_cfg=_model_runtime_cfg(
            model_bundle=state.get("bundle", {}),
            feature_runtime_cfg=runtime_cfg,
            disable_model_diagnostics=bool(args.disable_model_diagnostics),
            disable_model_timing=bool(args.disable_model_timing),
        ),
    )
    calibration_data = load_calibration_curves(str(artifact_data_root), args.run_id)
    rank_store = PolicyRankReferenceStore(data_root=artifact_data_root, run_id=args.run_id)

    rows: list[dict[str, Any]] = []
    if args.batch_by_signal_bar_cache:
        signal_times = pd.to_datetime(
            decisions["signal_bar_ts"], utc=True, errors="coerce"
        )
        for signal_ts, group in decisions.groupby(signal_times, sort=True):
            signal_ts = pd.Timestamp(signal_ts)
            if pd.isna(signal_ts):
                continue
            group_start_ts = signal_ts - pd.Timedelta(hours=int(args.lookback_hours))
            panel_slice = _slice_panel(
                panel,
                start_ts=group_start_ts,
                end_ts=signal_ts,
            )
            group_feature_cfg = dict(feature_cfg)
            group_feature_cfg["runtime_cfg"] = dict(runtime_cfg)
            print(
                f"Loading replay features for signal_bar_ts={signal_ts} "
                f"rows={len(group)} start={group_start_ts}"
            )
            feats = load_or_compute_features(
                panel_slice,
                list(panel_slice["close"].columns),
                args.run_id,
                str(args.data_root),
                group_feature_cfg,
                lookback_hours=int(args.lookback_hours),
                required_feature_keys=set(required_keys),
            )
            batched_feature_rows = _batched_replay_feature_rows(
                feats=feats,
                group=group,
                signal_bar_ts=signal_ts,
                overlay_required_columns=overlay_required_columns,
                live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                artifact_data_root=artifact_data_root,
                run_id=args.run_id,
            )
            rows.extend(
                _score_row(
                    row=row,
                    feats=feats,
                    orchestrator=orchestrator,
                    calibration_data=calibration_data,
                    rank_store=rank_store,
                    overlay_required_columns=overlay_required_columns,
                    live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                    feature_row_override=batched_feature_rows.get(idx),
                    skip_full_chain_diagnostics=bool(args.skip_full_chain_diagnostics),
                )
                for idx, row in group.iterrows()
            )
    else:
        feats = load_or_compute_features(
            panel,
            list(panel["close"].columns),
            args.run_id,
            str(args.data_root),
            feature_cfg,
            lookback_hours=int(args.lookback_hours),
            required_feature_keys=set(required_keys),
        )
        rows = [
            _score_row(
                row=row,
                feats=feats,
                orchestrator=orchestrator,
                calibration_data=calibration_data,
                rank_store=rank_store,
                overlay_required_columns=overlay_required_columns,
                live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                skip_full_chain_diagnostics=bool(args.skip_full_chain_diagnostics),
            )
            for _, row in decisions.iterrows()
        ]
    result = pd.DataFrame(rows)
    expected_transform_hash = runtime_cfg.get("feature_transform_contract_hash")
    if expected_transform_hash is not None:
        result["expected_feature_transform_contract_hash"] = expected_transform_hash
    out_dir = args.output_dir or (
        args.data_root
        / "artifacts"
        / args.run_id
        / "live_signal_prediction_replay"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "prediction_replay_comparison.csv", index=False)
    summary = _summary(result)
    (out_dir / "prediction_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Wrote {out_dir}")
    if args.fail_on_mismatch:
        failures = _parity_failures(
            result,
            tolerance=float(args.tolerance),
            prediction_tolerance=float(args.prediction_tolerance),
            require_policy_rank_reference=bool(args.require_policy_rank_reference),
            require_live_values=bool(args.require_live_values),
            parity_source=str(args.parity_source),
        )
        if failures:
            print(
                "Replay parity failed: " + ", ".join(failures),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
