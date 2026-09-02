#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.features import (  # noqa: E402
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    get_features_for_candidates,
    get_inference_required_feature_keys,
    load_or_compute_features,
    raw_required_feature_keys,
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
    ModelOrchestrator,
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


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


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
    }
    for src, dst in live_defaults.items():
        if src in ledger.columns and dst not in ledger.columns:
            ledger[dst] = ledger[src]

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
                        out[f"live_{col}"] = trade_row[col]
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


def _local_quote_symbols(
    data_root: Path,
    *,
    run_id: str | None = None,
    live_quote_currency: str = "USDC",
    market_mode: str = "spot",
) -> list[str]:
    out: list[str] = []
    quote = str(live_quote_currency or "USDC").upper()
    for path in sorted((data_root / "ohlcv").glob("symbol=*")):
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
    root = Path("cache") / "inference_live_features" / str(run_id)
    target = pd.Timestamp(end_ts)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    candidates: list[tuple[int, list[str]]] = []
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            raw_end = meta.get("end_ts")
            if not raw_end:
                continue
            cache_end = pd.Timestamp(raw_end)
            cache_end = cache_end.tz_localize("UTC") if cache_end.tzinfo is None else cache_end.tz_convert("UTC")
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
                candidates.append((len(symbols), symbols))
        except Exception:
            continue
    if not candidates:
        return []
    # Prefer the smallest non-trivial live universe. Replay/debug runs can add
    # broader local-OHLCV caches for the same timestamp after the fact.
    non_trivial = [item for item in candidates if item[0] >= 25]
    return sorted(non_trivial or candidates, key=lambda item: item[0])[0][1]


def _load_panel(
    *,
    data_root: Path,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    store = PartitionedOHLCVStore(str(data_root), timeframe="1h")
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
        data_root=data_root,
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
    feature_row = get_features_for_candidates(feats, [symbol], ts=ts)
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
        "replay_feature_cols": int(feature_row.shape[1]) if not feature_row.empty else 0,
        "replay_missing_features": bool(feature_row.empty),
    }
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
            "base_pred_delta": replay_base - out["live_base_pred"],
            "meta_pred_delta": replay_meta - out["live_meta_pred"],
            "calibrated_score_delta": replay_cal - out["live_calibrated_score"],
            "rank_percentile_delta": rank_lookup.policy_rank_pct
            - out["live_policy_rank_pct"],
        }
    )
    return out


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
    return summary


def _parity_failures(
    frame: pd.DataFrame,
    *,
    tolerance: float,
    require_policy_rank_reference: bool = False,
    require_live_values: bool = False,
) -> list[str]:
    failures: list[str] = []
    if frame.empty:
        failures.append("no_rows")
        return failures
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
    required_replay_cols = (
        "replay_base_pred",
        "replay_meta_pred",
        "replay_calibrated_score",
        "replay_policy_rank_pct",
    )
    for col in required_replay_cols:
        raw_vals = (
            frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
        )
        vals = pd.to_numeric(raw_vals, errors="coerce")
        missing = int(vals.isna().sum())
        if missing:
            failures.append(f"missing_{col}_rows={missing}")
    for col in (
        "base_pred_delta",
        "meta_pred_delta",
        "calibrated_score_delta",
        "rank_percentile_delta",
    ):
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
        if max_abs > float(tolerance):
            failures.append(f"{col}_max_abs={max_abs:.12g}")
    return failures


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
        "--live-quote-currency",
        default=None,
        help="Live quote currency used for local symbol discovery. Defaults to USD in perps mode, otherwise USDC.",
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
        "--tolerance",
        default=1e-9,
        type=float,
        help="Absolute tolerance used by --fail-on-mismatch.",
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
    )
    if not panel or "close" not in panel:
        raise SystemExit("No OHLCV panel could be loaded.")

    state = load_full_state(args.run_id, str(artifact_data_root))
    required_keys = raw_required_feature_keys(get_inference_required_feature_keys(state, None))
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
    runtime_cfg["live_data_root"] = str(args.data_root)
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
    feats = load_or_compute_features(
        panel,
        list(panel["close"].columns),
        args.run_id,
        str(args.data_root),
        feature_cfg,
        lookback_hours=int(args.lookback_hours),
        required_feature_keys=set(required_keys),
    )
    orchestrator = ModelOrchestrator(state, runtime_cfg={"model_bundle": state.get("bundle", {})})
    calibration_data = load_calibration_curves(str(artifact_data_root), args.run_id)
    rank_store = PolicyRankReferenceStore(data_root=artifact_data_root, run_id=args.run_id)

    rows = [
        _score_row(
            row=row,
            feats=feats,
            orchestrator=orchestrator,
            calibration_data=calibration_data,
            rank_store=rank_store,
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
            require_policy_rank_reference=bool(args.require_policy_rank_reference),
            require_live_values=bool(args.require_live_values),
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
