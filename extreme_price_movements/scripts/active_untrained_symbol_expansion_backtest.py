from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG as DEFAULT_CFG
from extreme_price_movements.data_store import PartitionedOHLCVStore, scoped_data_root
from extreme_price_movements.holdout_strategy_eval import (
    _has_deployed_meta_model,
    _resolve_alpha_strategy_id,
)
from extreme_price_movements.pipeline_steps import _load_saved_microdata_for_symbols
from extreme_price_movements.pipeline_steps import _compute_features_hourly_runtime
from extreme_price_movements.features import add_regime_gates, compute_market_features
from extreme_price_movements.features_residual import RESIDUAL_FEATURE_KEYS
from extreme_price_movements.inference.candidate_selector import _build_mask_for_mode
from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
)
from extreme_price_movements.model_loader import load_model_bundle
from extreme_price_movements.utils import tprint


_COMPARE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|<|>)")


@dataclass(frozen=True)
class SymbolRecord:
    live_symbol: str
    base: str
    cache_symbol: str
    group: str


def _base_from_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "/" in raw:
        return raw.split("/", 1)[0]
    if "_" in raw:
        return raw.split("_", 1)[0]
    for suffix in ("USDT", "USDC", "USD"):
        if raw.endswith(suffix) and len(raw) > len(suffix):
            return raw[: -len(suffix)]
    return raw


def _live_symbol_for_base(base: str) -> str:
    return f"{str(base).upper()}/USD:USD"


def _perp_data_cfg(data_root: str) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "timeframe": "1h",
        "exchange_id": "krakenfutures",
        "exchange": "krakenfutures",
        "market_mode": "perps",
        "use_perps": True,
    }


def _market_data_root(data_root: str) -> Path:
    return Path(scoped_data_root(_perp_data_cfg(data_root)))


def _local_cache_symbol_for_base(data_root: str, base: str) -> str | None:
    root = _market_data_root(data_root) / "ohlcv"
    base = str(base).upper()
    candidates = [
        f"{base}_USDT",
        f"{base}_USD:USD",
        f"{base}_USD",
        f"{base}_USDC",
    ]
    for candidate in candidates:
        if (root / f"symbol={candidate}").exists():
            return candidate
    return None


def _context_records_for_feature_computation(
    *,
    data_root: str,
    cfg: dict[str, Any],
    existing_bases: set[str],
) -> tuple[list[SymbolRecord], dict[str, str]]:
    """Return benchmark/basket symbols needed only to compute contextual features."""

    bases: set[str] = set()
    for key in ("primary_benchmark", "benchmark_1", "bench1_symbol"):
        value = cfg.get(key)
        if value:
            bases.add(_base_from_symbol(str(value)))
    for value in cfg.get("market_basket", []) or []:
        if value:
            bases.add(_base_from_symbol(str(value)))
    # Benchmark residuals default to BTC if no explicit benchmark is configured.
    bases.add("BTC")

    context: list[SymbolRecord] = []
    missing: dict[str, str] = {}
    for base in sorted(b for b in bases if b and b not in existing_bases):
        cache_symbol = _local_cache_symbol_for_base(data_root, base)
        live_symbol = _live_symbol_for_base(base)
        if not cache_symbol:
            missing[live_symbol] = "no_local_hourly_cache"
            continue
        context.append(
            SymbolRecord(
                live_symbol=live_symbol,
                base=base,
                cache_symbol=cache_symbol,
                group="feature_context_only",
            )
        )
    return context, missing


def _load_active_symbols(path: str | None, data_root: str) -> list[str]:
    if path:
        payload = json.loads(Path(path).read_text())
        if isinstance(payload, dict):
            raw = payload.get("symbols") or payload.get("active_symbols") or []
        else:
            raw = payload
        return sorted({_live_symbol_for_base(_base_from_symbol(s)) for s in raw if str(s)})

    root = _market_data_root(data_root) / "ohlcv"
    bases: set[str] = set()
    for sym_dir in root.glob("symbol=*"):
        if not sym_dir.is_dir():
            continue
        raw = sym_dir.name.replace("symbol=", "")
        base = _base_from_symbol(raw)
        if base:
            bases.add(base)
    return sorted(_live_symbol_for_base(base) for base in bases)


def _trained_bases(data_root: str, run_id: str) -> set[str]:
    path = Path(data_root) / "artifacts" / run_id / "oof" / "base_oof_all.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["symbol"])
    return {_base_from_symbol(s) for s in df["symbol"].dropna().unique()}


def _load_policy_strategies(data_root: str, run_id: str) -> list[dict[str, Any]]:
    root = Path(data_root) / "artifacts" / run_id
    for rel in (
        "policy_params/strategy_for_inference.json",
        "strategy_for_inference.json",
    ):
        path = root / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("strategies") if isinstance(payload, dict) else None
        if isinstance(rows, list):
            return [dict(row) for row in rows if row.get("selected", True)]
    raise FileNotFoundError(f"No strategy_for_inference.json found under {root}")


def _mask_feature_keys(strategies: Iterable[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in strategies:
        mask = row.get("lgbm_regime_mask") or {}
        params = mask.get("mask_params") if isinstance(mask, dict) else {}
        text = ""
        if isinstance(params, dict):
            text = str(params.get("canonical_key") or params.get("base_event_trigger") or "")
        text += " " + str(mask.get("canonical_key") or "")
        for name in _COMPARE_RE.findall(text):
            keys.add(name)
    return keys


def _load_panel(
    store: PartitionedOHLCVStore,
    records: list[SymbolRecord],
    *,
    data_root: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], list[SymbolRecord], dict[str, str]]:
    cols = ["open", "high", "low", "close", "volume", "open_interest", "funding_rate"]
    loaded: dict[str, pd.DataFrame] = {}
    kept: list[SymbolRecord] = []
    rejected: dict[str, str] = {}
    for rec in records:
        df = store.load(rec.cache_symbol, columns=cols, start_ts=start_ts, end_ts=end_ts)
        if df.empty:
            rejected[rec.live_symbol] = "no_cached_rows"
            continue
        if len(df) < 24 * 7:
            rejected[rec.live_symbol] = f"too_few_rows:{len(df)}"
            continue
        df = df.sort_index()
        kept.append(rec)
        loaded[rec.live_symbol] = df

    if not kept:
        raise RuntimeError("No symbols had enough cached hourly rows for expansion evaluation")

    union_index = pd.DatetimeIndex(
        sorted(set().union(*(df.index for df in loaded.values()))), tz="UTC"
    )
    panel: dict[str, pd.DataFrame] = {}
    for col in cols:
        series_by_symbol = {}
        for rec in kept:
            df = loaded[rec.live_symbol]
            if col in df.columns:
                series_by_symbol[rec.live_symbol] = pd.to_numeric(
                    df[col], errors="coerce"
                ).astype(np.float32)
        if series_by_symbol:
            panel[col] = pd.DataFrame(series_by_symbol, index=union_index)
    sidecars, orderbook = _load_saved_microdata_for_symbols(
        data_root=str(data_root),
        symbols=[rec.live_symbol for rec in kept],
        index=union_index,
        cfg=_perp_data_cfg(str(data_root)),
    )
    for key, frame in sidecars.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            panel[key] = frame.reindex(index=union_index, columns=[rec.live_symbol for rec in kept])
    if orderbook:
        panel["orderbook_hourly"] = orderbook
        for field in sorted(
            {
                str(col)
                for ob in orderbook.values()
                if isinstance(ob, pd.DataFrame)
                for col in ob.columns
            }
        ):
            by_symbol: dict[str, pd.Series] = {}
            for rec in kept:
                ob = orderbook.get(rec.live_symbol)
                if isinstance(ob, pd.DataFrame) and field in ob.columns:
                    by_symbol[rec.live_symbol] = pd.to_numeric(
                        ob[field], errors="coerce"
                    ).astype(np.float32)
            if by_symbol:
                panel[f"orderbook_{field}"] = (
                    pd.DataFrame(by_symbol)
                    .reindex(index=union_index, columns=[rec.live_symbol for rec in kept])
                    .ffill()
                    .astype(np.float32)
                )
    return panel, kept, rejected


def _latest_common_end(
    store: PartitionedOHLCVStore, records: list[SymbolRecord], min_rows: int
) -> pd.Timestamp:
    maxes: list[pd.Timestamp] = []
    probe_start = pd.Timestamp("2024-01-01", tz="UTC")
    for rec in records:
        df = store.load(rec.cache_symbol, columns=["close"], start_ts=probe_start)
        if len(df) >= min_rows:
            maxes.append(pd.Timestamp(df.index.max()).tz_convert("UTC"))
    if not maxes:
        raise RuntimeError("No records have enough cached history to choose an evaluation end")
    return max(maxes).floor("h")


def _feature_frame_for_rows(
    feats: dict[str, Any],
    feature_keys: Iterable[str],
    row_index: pd.MultiIndex,
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    ts = row_index.get_level_values("timestamp")
    syms = row_index.get_level_values("symbol")
    for key in feature_keys:
        obj = feats.get(key)
        if obj is None:
            continue
        if isinstance(obj, pd.Series):
            vals = obj.reindex(ts).to_numpy(dtype=np.float32)
        elif isinstance(obj, pd.DataFrame):
            vals = np.empty(len(row_index), dtype=np.float32)
            for symbol in pd.Index(syms).unique():
                mask = syms == symbol
                if symbol in obj.columns:
                    vals[mask] = obj[symbol].reindex(ts[mask]).to_numpy(dtype=np.float32)
                elif obj.shape[1] == 1:
                    vals[mask] = obj.iloc[:, 0].reindex(ts[mask]).to_numpy(dtype=np.float32)
                else:
                    vals[mask] = np.nan
        else:
            continue
        data[str(key)] = vals
    return pd.DataFrame(data, index=row_index)


def _rank_lookup(
    store: PolicyRankReferenceStore,
    strategy_id: str,
    side: str,
    score: float,
) -> tuple[float, float]:
    policy = store.lookup(strategy_id=strategy_id, side=side, calibrated_score=score)
    auction = store.lookup_auction(calibrated_score=score)
    policy_rank = float(policy.policy_rank_pct) if np.isfinite(policy.policy_rank_pct) else np.nan
    auction_rank = float(auction.policy_rank_pct) if np.isfinite(auction.policy_rank_pct) else np.nan
    return policy_rank, auction_rank


_RESIDUAL_BASE_FEATURE_DEPS: dict[str, tuple[str, ...]] = {
    "ret4h_bench_resid": ("ret4h",),
    "ret24h_bench_resid": ("ret24h",),
    "ret48h_bench_resid": ("ret48h",),
    "ret4h_peer_resid": ("ret4h",),
    "ret24h_peer_resid": ("ret24h",),
    "rv_24h_peer_resid": ("rv_24h",),
    "vol_z_peer_resid": ("vol_z",),
    "rvol_z_peer_resid": ("rvol_z",),
    "amihud_z_peer_resid": ("amihud_z",),
    "liquidity_ratio_peer_resid": ("liquidity_ratio",),
    "dist_vwap_norm_mkt_resid": ("dist_vwap_norm",),
    "dist_ema_fast_mkt_resid": ("dist_ema_fast",),
    "trend_pct_mkt_resid": ("trend_pct",),
    "dist_vwap_norm_ts_resid": ("dist_vwap_norm",),
    "dist_ema_fast_ts_resid": ("dist_ema_fast",),
    "rsi_ts_resid": ("rsi",),
    "flow_persistence_ts_resid": ("flow_persistence",),
    "excess_6h_ts_resid": ("excess_6h",),
    "atr_expansion_ts_resid": ("atr_expansion",),
    "coherence_24_ts_resid": ("coherence_24",),
    "basis_pct_mkt_resid": ("basis_pct",),
    "funding_per_hour_mkt_resid": ("funding_per_hour",),
    "fund_abs_z_mkt_resid": ("fund_abs_z",),
    "basis_fund_div_mkt_resid": ("basis_fund_div",),
    "xasset_funding_ts_resid": ("xasset_asset_minus_mkt_funding",),
    "xasset_funding_peer_resid": ("xasset_asset_minus_mkt_funding",),
    "funding_1d_chg_ts_resid": ("funding_1d_chg_z_90d",),
    "funding_1d_chg_peer_resid": ("funding_1d_chg_z_90d",),
    "oi_chg_8h_mkt_resid": ("oi_chg_8h",),
    "oi_rel_vol_8h_peer_resid": ("oi_rel_vol_8h",),
    "oi_chg_8h_robust_z_peer_resid": ("oi_chg_8h_robust_z",),
    "asset_minus_mkt_oi_1d_ts_resid": ("asset_minus_mkt_oi_1d_z_90d",),
    "asset_minus_mkt_oi_7d_ts_resid": ("asset_minus_mkt_oi_7d_z_180d",),
    "asset_minus_mkt_oi_1d_peer_resid": ("asset_minus_mkt_oi_1d_z_90d",),
    "asset_minus_mkt_oi_7d_peer_resid": ("asset_minus_mkt_oi_7d_z_180d",),
    "squeeze_prob_mkt_resid": ("squeeze_prob",),
    "ob_pressure_mkt_resid": ("ob_pressure",),
    "ob_spread_mkt_resid": ("ob_spread",),
    "ob_depth_mkt_resid": ("ob_depth",),
    "ob_imbalance_mkt_resid": ("ob_imbalance",),
    "xasset_ob_pressure_ts_resid": ("xasset_asset_minus_mkt_ob_pressure_z_24h",),
    "xasset_ob_pressure_peer_resid": ("xasset_asset_minus_mkt_ob_pressure_z_24h",),
    "xasset_ob_liquidity_ts_resid": ("xasset_ob_liquidity_divergence_z_24h",),
    "xasset_ob_liquidity_peer_resid": ("xasset_ob_liquidity_divergence_z_24h",),
    "volume_price_corr_ts_resid": ("volume_price_corr",),
    "path_efficiency_24_ts_resid": ("path_efficiency_24",),
    "entry_quality_composite_ts_resid": ("entry_quality_composite",),
}


def _expand_feature_compute_keys(feature_keys: Iterable[str]) -> set[str]:
    """Add hidden dependencies needed to construct requested residual features.

    Model scoring still receives the original decision-used feature list. This
    expansion is only for the historical feature construction pass, where
    residual features are derived after their base inputs have been built.
    """

    expanded = {str(k) for k in feature_keys if str(k)}
    residual_names = set(RESIDUAL_FEATURE_KEYS)
    for key in list(expanded):
        if key in residual_names:
            expanded.update(_RESIDUAL_BASE_FEATURE_DEPS.get(key, ()))
    return expanded


def _historical_training_path_features(
    *,
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    run_id: str,
    data_root: str,
    cfg: dict[str, Any],
    required_feature_keys: set[str],
) -> dict[str, pd.DataFrame]:
    """Compute full-history features through the training-path feature builder.

    This expansion audit evaluates multi-week historical windows. The live
    inference feature loader is intentionally latest-row oriented for latency,
    so using it here can produce one-row mask features and a misleading
    zero-pass historical result.
    """

    compute_cfg = {**dict(DEFAULT_CFG), **dict(cfg or {})}
    compute_cfg["data_root"] = str(data_root)
    compute_cfg["run_id"] = str(run_id)
    compute_cfg["market_mode"] = "perps"
    compute_cfg["use_perps"] = True
    compute_cfg["exchange_id"] = "krakenfutures"
    compute_cfg["exchange"] = "krakenfutures"
    if required_feature_keys:
        compute_cfg["enable_gated_features"] = True

    mkt_df = compute_market_features(
        panel,
        symbols,
        trend_sma_hours=int(compute_cfg.get("trend_sma_hours", 24 * 14) or 24 * 14),
    )
    mkt_gates = add_regime_gates(
        mkt_df,
        int(compute_cfg.get("gate_vol_lookback_hours", 24 * 7) or 24 * 7),
        float(compute_cfg.get("gate_trend_thr", 0.0) or 0.0),
    )
    compute_feature_keys = _expand_feature_compute_keys(required_feature_keys)
    feats, _, _ = _compute_features_hourly_runtime(
        panel,
        mkt_gates,
        compute_cfg,
        panel.get("orderbook_hourly") if isinstance(panel.get("orderbook_hourly"), dict) else {},
        requested_feature_keys=sorted(compute_feature_keys),
    )
    return {
        str(k): v
        for k, v in feats.items()
        if str(k) in required_feature_keys and isinstance(v, pd.DataFrame)
    }


def _feature_spans_eval_window(
    feats: dict[str, Any],
    feature_keys: Iterable[str],
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> tuple[bool, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    ok = True
    for key in sorted({str(k) for k in feature_keys if str(k)}):
        obj = feats.get(key)
        if not isinstance(obj, pd.DataFrame) or obj.empty:
            diagnostics.append({"feature": key, "status": "missing_or_empty"})
            ok = False
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(obj.index, utc=True, errors="coerce"))
        finite_rows = int(
            np.isfinite(obj.loc[(idx >= eval_start) & (idx <= eval_end)].to_numpy(dtype=np.float64)).any(axis=1).sum()
        ) if len(idx) else 0
        status = "ok" if finite_rows > 0 else "no_eval_finite_rows"
        diagnostics.append(
            {
                "feature": key,
                "status": status,
                "rows": int(len(obj)),
                "start": str(idx.min()) if len(idx) else "",
                "end": str(idx.max()) if len(idx) else "",
                "eval_finite_rows": finite_rows,
            }
        )
        if finite_rows <= 0:
            ok = False
    return ok, diagnostics


def _score_strategy(
    *,
    strategy: dict[str, Any],
    feats: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    bundle: dict[str, Any],
    orchestrator: ModelOrchestrator,
    rank_store: PolicyRankReferenceStore,
    required_keys: set[str],
    fee_bps: float,
    spread_bps: float,
    horizon_hours: int,
    diagnostics: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    strategy_id = str(strategy.get("strategy_for_inference") or strategy.get("strategy_id") or "")
    side = str(strategy.get("side") or "").lower()
    threshold = float(strategy.get("deployment_rank_threshold", np.nan))
    mask_cfg = ((strategy.get("lgbm_regime_mask") or {}).get("mask_params") or {})
    mask_df = _build_mask_for_mode(panel, feats, mask_cfg)
    diag: dict[str, Any] = {
        "strategy_id": strategy_id,
        "side": side,
        "mask_rows_total": 0,
        "mask_pass_total": 0,
        "mask_rows_eval": 0,
        "mask_pass_eval": 0,
        "scored_rows": 0,
        "selected_rows": 0,
    }
    if mask_df.empty:
        if diagnostics is not None:
            diagnostics.append({**diag, "reason": "empty_mask"})
        return pd.DataFrame()
    mask_df = mask_df.reindex(index=panel["close"].index, columns=symbols, fill_value=False)
    diag["mask_rows_total"] = int(mask_df.size)
    diag["mask_pass_total"] = int(mask_df.to_numpy(dtype=bool, copy=False).sum())
    mask_df = mask_df.loc[(mask_df.index >= eval_start) & (mask_df.index <= eval_end)]
    diag["mask_rows_eval"] = int(mask_df.size)
    diag["mask_pass_eval"] = int(mask_df.to_numpy(dtype=bool, copy=False).sum())
    stacked = mask_df.stack(dropna=False)
    stacked = stacked[stacked.astype(bool)]
    if stacked.empty:
        if diagnostics is not None:
            diagnostics.append({**diag, "reason": "no_mask_pass"})
        return pd.DataFrame()
    row_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(ts), str(sym)) for ts, sym in stacked.index],
        names=["timestamp", "symbol"],
    )
    feature_df = _feature_frame_for_rows(feats, required_keys, row_index)
    if feature_df.empty:
        if diagnostics is not None:
            diagnostics.append({**diag, "reason": "empty_feature_frame"})
        return pd.DataFrame()
    alpha_id = _resolve_alpha_strategy_id(bundle, strategy_id, side)
    alpha = orchestrator.predict_alpha(feature_df, side, alpha_id)
    if alpha.empty:
        if diagnostics is not None:
            diagnostics.append({**diag, "reason": "empty_alpha_score", "feature_rows": int(len(feature_df))})
        return pd.DataFrame()
    score = alpha.reindex(feature_df.index)
    score_source = "alpha"
    if _has_deployed_meta_model(orchestrator, side=side, strategy_id=strategy_id):
        meta_base = feature_df.copy()
        meta_base[strategy_id] = score
        meta_base[alpha_id] = score
        meta_base["__symbol__"] = meta_base.index.get_level_values("symbol").astype(str)
        meta_base["__ts__"] = pd.DatetimeIndex(meta_base.index.get_level_values("timestamp"))
        meta = orchestrator.predict_meta(meta_base, side, strategy_id)
        if not meta.empty:
            score = meta.reindex(feature_df.index)
            score_source = "meta"
    out = pd.DataFrame(
        {
            "timestamp": feature_df.index.get_level_values("timestamp"),
            "symbol": feature_df.index.get_level_values("symbol").astype(str),
            "strategy_id": strategy_id,
            "side": side,
            "calibrated_score": pd.to_numeric(score, errors="coerce").to_numpy(dtype=np.float64),
            "score_source": score_source,
            "deployment_rank_threshold": threshold,
        },
        index=feature_df.index,
    )
    ranks = np.array(
        [
            _rank_lookup(rank_store, strategy_id, side, s)
            for s in out["calibrated_score"].to_numpy(dtype=np.float64)
        ],
        dtype=np.float64,
    )
    out["policy_rank_pct"] = ranks[:, 0] if ranks.size else np.nan
    out["auction_rank_score"] = ranks[:, 1] if ranks.size else np.nan
    out["threshold_rank_score"] = out["auction_rank_score"].where(
        np.isfinite(out["auction_rank_score"]), out["policy_rank_pct"]
    )
    out["selected"] = out["threshold_rank_score"] >= threshold
    diag["scored_rows"] = int(len(out))
    diag["selected_rows"] = int(out["selected"].sum())
    diag["finite_score_rows"] = int(np.isfinite(out["calibrated_score"]).sum())
    diag["finite_rank_rows"] = int(np.isfinite(out["threshold_rank_score"]).sum())
    diag["max_threshold_rank_score"] = (
        float(np.nanmax(out["threshold_rank_score"].to_numpy(dtype=np.float64)))
        if len(out) and np.isfinite(out["threshold_rank_score"]).any()
        else np.nan
    )

    close = panel["close"]
    entry_ts = pd.to_datetime(out["timestamp"], utc=True) + pd.Timedelta(hours=1)
    exit_ts = pd.to_datetime(out["timestamp"], utc=True) + pd.Timedelta(hours=horizon_hours)
    entry = np.empty(len(out), dtype=np.float64)
    exitp = np.empty(len(out), dtype=np.float64)
    for symbol in out["symbol"].unique():
        mask = out["symbol"].to_numpy() == symbol
        s_close = close[symbol]
        entry[mask] = s_close.reindex(entry_ts[mask], method="nearest", tolerance=pd.Timedelta(hours=1)).to_numpy(dtype=np.float64)
        exitp[mask] = s_close.reindex(exit_ts[mask], method="nearest", tolerance=pd.Timedelta(hours=1)).to_numpy(dtype=np.float64)
    out["proxy_entry_ts"] = entry_ts.to_numpy()
    out["proxy_exit_ts"] = exit_ts.to_numpy()
    out["proxy_entry_price"] = entry
    out["proxy_exit_price"] = exitp
    diag["finite_entry_price_rows"] = int(np.isfinite(entry).sum())
    diag["finite_exit_price_rows"] = int(np.isfinite(exitp).sum())
    gross = (exitp / entry) - 1.0
    if side == "short":
        gross = -gross
    cost = (float(fee_bps) + float(spread_bps)) / 10000.0
    out["gross_return_proxy"] = gross
    out["net_return_proxy"] = gross - cost
    diag["finite_net_return_rows"] = int(np.isfinite(out["net_return_proxy"]).sum())
    out = out[np.isfinite(out["calibrated_score"]) & np.isfinite(out["net_return_proxy"])]
    if diagnostics is not None:
        diagnostics.append({**diag, "reason": "ok", "finite_rows": int(len(out))})
    return out.reset_index(drop=True)


def _summarise(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=group_cols + ["n_scored", "n_selected", "gross_hit_rate", "net_hit_rate", "mean_gross_return", "mean_net_return"])
    def agg(g: pd.DataFrame) -> pd.Series:
        sel = g[g["selected"]].copy()
        return pd.Series(
            {
                "n_scored": int(len(g)),
                "n_selected": int(len(sel)),
                "gross_hit_rate": float((sel["gross_return_proxy"] > 0).mean()) if len(sel) else np.nan,
                "net_hit_rate": float((sel["net_return_proxy"] > 0).mean()) if len(sel) else np.nan,
                "mean_gross_return": float(sel["gross_return_proxy"].mean()) if len(sel) else np.nan,
                "mean_net_return": float(sel["net_return_proxy"].mean()) if len(sel) else np.nan,
            }
        )
    return rows.groupby(group_cols, dropna=False).apply(agg).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest active but not trained/OOS-covered symbols for tradeable-universe expansion.")
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", default="20260525_010004_nopenalty")
    parser.add_argument("--active-symbols-json", default="")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--warmup-hours", type=int, default=24 * 120)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--fee-bps", type=float, default=7.0)
    parser.add_argument("--spread-bps", type=float, default=66.7)
    parser.add_argument("--tolerance-pp", type=float, default=0.05)
    parser.add_argument("--max-symbols-per-group", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    data_root = str(args.data_root)
    run_id = str(args.run_id)
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_root) / "artifacts" / run_id / "symbol_expansion"
    output_dir.mkdir(parents=True, exist_ok=True)

    active = _load_active_symbols(args.active_symbols_json or None, data_root)
    trained = _trained_bases(data_root, run_id)
    records: list[SymbolRecord] = []
    missing_cache: dict[str, str] = {}
    for sym in active:
        base = _base_from_symbol(sym)
        cache_symbol = _local_cache_symbol_for_base(data_root, base)
        if not cache_symbol:
            missing_cache[sym] = "no_local_hourly_cache"
            continue
        group = "baseline_trained" if base in trained else "active_untrained"
        records.append(SymbolRecord(sym, base, cache_symbol, group))
    if args.max_symbols_per_group > 0:
        limited: list[SymbolRecord] = []
        for group in ("baseline_trained", "active_untrained"):
            limited.extend([r for r in records if r.group == group][: args.max_symbols_per_group])
        records = limited

    min_eval_rows = int(args.weeks * 7 * 24 * 0.75)
    cfg = _perp_data_cfg(data_root)
    cfg = {**dict(DEFAULT_CFG), **cfg}
    eval_records = list(records)
    context_records, missing_context_cache = _context_records_for_feature_computation(
        data_root=data_root,
        cfg=cfg,
        existing_bases={r.base for r in eval_records},
    )
    records_for_feature_computation = eval_records + context_records
    market_data_root = _market_data_root(data_root)
    store = PartitionedOHLCVStore(root_dir=str(market_data_root), timeframe="1h")
    eval_end = (
        _latest_common_end(store, eval_records, min_rows=min_eval_rows).floor("h")
        - pd.Timedelta(hours=int(args.horizon_hours))
    )
    eval_start = eval_end - pd.Timedelta(days=7 * int(args.weeks))
    load_start = eval_start - pd.Timedelta(hours=int(args.warmup_hours))
    tprint(f"Expansion eval window: eval_start={eval_start} eval_end={eval_end} load_start={load_start}")

    panel, kept, rejected = _load_panel(
        store,
        records_for_feature_computation,
        data_root=data_root,
        start_ts=load_start,
        end_ts=eval_end + pd.Timedelta(hours=args.horizon_hours + 1),
    )
    symbols = [r.live_symbol for r in kept]
    eval_symbol_set = {r.live_symbol for r in eval_records}
    scoring_symbols = [r.live_symbol for r in kept if r.live_symbol in eval_symbol_set]
    group_by_symbol = {r.live_symbol: r.group for r in kept if r.live_symbol in eval_symbol_set}
    base_by_symbol = {r.live_symbol: r.base for r in kept if r.live_symbol in eval_symbol_set}

    strategies = _load_policy_strategies(data_root, run_id)
    accepted_ids = {str(s.get("strategy_for_inference") or s.get("strategy_id")) for s in strategies}
    bundle = load_model_bundle(run_id, data_root)
    required = get_inference_required_feature_keys(bundle, accepted_ids)
    required.update(_mask_feature_keys(strategies))
    cfg["historical_inference_parity_allow_missing_live_sources"] = True
    cfg["live_materialize_orderbook_model_features"] = bool(
        cfg.get("live_materialize_orderbook_model_features", True)
    )
    feats = _historical_training_path_features(
        panel=panel,
        symbols=symbols,
        run_id=run_id,
        data_root=data_root,
        cfg=cfg,
        required_feature_keys=required,
    )
    mask_ok, mask_feature_diagnostics = _feature_spans_eval_window(
        feats,
        _mask_feature_keys(strategies),
        eval_start=eval_start,
        eval_end=eval_end,
    )
    if not mask_ok:
        diag_path = output_dir / "active_untrained_symbol_expansion_feature_window_diagnostics.csv"
        pd.DataFrame(mask_feature_diagnostics).to_csv(diag_path, index=False)
        raise RuntimeError(
            "Historical expansion audit cannot continue: one or more mask "
            "features do not span the eval window. This usually means the "
            "audit accidentally received latest-only live feature frames. "
            f"diagnostics={diag_path}"
        )
    orchestrator = ModelOrchestrator(bundle, {"disable_spike_filter": True, "inference_model_timing_enabled": False})
    rank_store = PolicyRankReferenceStore(data_root=data_root, run_id=run_id)

    frames = []
    mask_diagnostics: list[dict[str, Any]] = []
    for strat in strategies:
        sid = str(strat.get("strategy_for_inference") or strat.get("strategy_id"))
        tprint(f"Scoring expansion strategy: {sid[:100]}")
        frame = _score_strategy(
            strategy=strat,
            feats=feats,
            panel=panel,
            symbols=scoring_symbols,
            eval_start=eval_start,
            eval_end=eval_end,
            bundle=bundle,
            orchestrator=orchestrator,
            rank_store=rank_store,
            required_keys=required,
            fee_bps=float(args.fee_bps),
            spread_bps=float(args.spread_bps),
            horizon_hours=int(args.horizon_hours),
            diagnostics=mask_diagnostics,
        )
        if not frame.empty:
            frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_rows.empty:
        all_rows["group"] = all_rows["symbol"].map(group_by_symbol)
        all_rows["base"] = all_rows["symbol"].map(base_by_symbol)

    group_summary = _summarise(all_rows, ["group"])
    strategy_summary = _summarise(all_rows, ["group", "strategy_id", "side"])
    symbol_summary = _summarise(all_rows, ["group", "symbol", "base"])

    baseline = group_summary[group_summary["group"] == "baseline_trained"]
    eligible_symbols: list[str] = []
    if not baseline.empty and not symbol_summary.empty:
        b_hit = float(baseline["gross_hit_rate"].iloc[0])
        b_net = float(baseline["mean_net_return"].iloc[0])
        sym_untrained = symbol_summary[symbol_summary["group"] == "active_untrained"].copy()
        sym_untrained["baseline_gross_hit_rate"] = b_hit
        sym_untrained["baseline_mean_net_return"] = b_net
        sym_untrained["gross_hit_delta"] = sym_untrained["gross_hit_rate"] - b_hit
        sym_untrained["mean_net_return_delta"] = sym_untrained["mean_net_return"] - b_net
        sym_untrained["eligible_for_tradeable_expansion"] = (
            (sym_untrained["n_selected"] > 0)
            & (sym_untrained["gross_hit_delta"] >= -float(args.tolerance_pp))
            & (sym_untrained["mean_net_return_delta"] >= -float(args.tolerance_pp))
            & (sym_untrained["mean_net_return"] > 0.0)
        )
        symbol_summary = pd.concat(
            [symbol_summary[symbol_summary["group"] != "active_untrained"], sym_untrained],
            ignore_index=True,
        )
        eligible_symbols = sorted(sym_untrained.loc[sym_untrained["eligible_for_tradeable_expansion"], "symbol"].astype(str).tolist())

    all_rows.to_parquet(output_dir / "active_untrained_symbol_expansion_rows.parquet", index=False)
    group_summary.to_csv(output_dir / "active_untrained_symbol_expansion_group_summary.csv", index=False)
    strategy_summary.to_csv(output_dir / "active_untrained_symbol_expansion_strategy_summary.csv", index=False)
    symbol_summary.to_csv(output_dir / "active_untrained_symbol_expansion_symbol_summary.csv", index=False)
    pd.DataFrame(mask_diagnostics).to_csv(output_dir / "active_untrained_symbol_expansion_mask_diagnostics.csv", index=False)
    payload = {
        "schema_version": "active_untrained_symbol_expansion_backtest_v1",
        "run_id": run_id,
        "data_root": data_root,
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "weeks": int(args.weeks),
        "horizon_hours": int(args.horizon_hours),
        "fee_bps": float(args.fee_bps),
        "spread_bps": float(args.spread_bps),
        "tolerance_pp": float(args.tolerance_pp),
        "active_symbols": len(active),
        "trained_bases": len(trained),
        "records_with_local_cache": len(eval_records),
        "context_records_with_local_cache": len(context_records),
        "kept_symbols": len(kept),
        "scored_symbol_count": len(scoring_symbols),
        "missing_local_cache": missing_cache,
        "missing_context_cache": missing_context_cache,
        "rejected_cached_symbols": rejected,
        "eligible_symbols": eligible_symbols,
        "eligible_bases": sorted({_base_from_symbol(s) for s in eligible_symbols}),
        "group_summary": group_summary.to_dict(orient="records"),
        "mask_diagnostics": mask_diagnostics,
        "notes": [
            "This is a cross-asset unseen-symbol, inference-like evaluation using deployed final-fit models and policy rank references.",
            "It is not policy-OOS for model training rows; the expansion symbols were absent from training/OOS coverage.",
            "Returns use an hourly proxy from next-hour entry to configured horizon with configured fee/spread drag because historical t+10 1m paths are not available for all expansion symbols.",
            "Historical missing live-source model features are allowed through the feature generator parity switch; symbols with non-finite selected model matrices still fail downstream scoring.",
        ],
    }
    (output_dir / "tradeable_symbol_expansion.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({k: payload[k] for k in ("eval_start", "eval_end", "active_symbols", "records_with_local_cache", "kept_symbols", "eligible_symbols", "group_summary")}, indent=2, default=str))


if __name__ == "__main__":
    main()
