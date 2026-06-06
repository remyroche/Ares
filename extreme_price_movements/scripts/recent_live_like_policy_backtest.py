"""Recent live-like policy backtest for deployed inference strategies.

Diagnostic-only. This evaluates recent windows with:
- deployed final-fit models,
- training-path feature generation,
- deployed mask contracts, rank references, and thresholds,
- current trained + active-untrained expansion tradable universe,
- t+5 delayed entry using cached 1m execution bars where available,
- p66 spread, fees, and simple-policy stop/trailing simulation.

The historical path source is hourly OHLCV. Cached 1m bars are used only to
replace the entry price at t+5 when available. This keeps the test executable on
current local data while clearly reporting delayed-entry coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

# This script must not download during diagnostics unless explicitly requested.
os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")
os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
os.environ.setdefault("EXCHANGE_NAME", "krakenfutures")

from extreme_price_movements.config import CFG as DEFAULT_CFG
from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.inference.config import (
    load_active_untrained_symbol_expansion_allowlist,
)
from extreme_price_movements.inference.feature_generator import (
    _compute_policy_barrier_pct,
    get_inference_required_feature_keys,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.policy_rank_reference import PolicyRankReferenceStore
from extreme_price_movements.inference.simple_policy_stop import (
    load_simple_policy_stop_params_by_strategy,
)
from extreme_price_movements.model_loader import load_model_bundle
from extreme_price_movements.scripts.active_untrained_symbol_expansion_backtest import (
    SymbolRecord,
    _base_from_symbol,
    _context_records_for_feature_computation,
    _expand_feature_compute_keys,
    _feature_spans_eval_window,
    _historical_training_path_features,
    _live_symbol_for_base,
    _load_active_symbols,
    _load_panel,
    _load_policy_strategies,
    _local_cache_symbol_for_base,
    _market_data_root,
    _mask_feature_keys,
    _perp_data_cfg,
    _score_strategy,
    _trained_bases,
)
from extreme_price_movements.simple_policy_optimiser import (
    _apply_delayed_entry_execution_model,
    _fetch_policy_paths,
    _policy_market_data_root,
    simulate_and_score,
)
from extreme_price_movements.utils import tprint


DEFAULT_RUN_ID = "20260525_010004_nopenalty"
DEFAULT_OUT_DIR = (
    Path("extreme_price_movements")
    / "reports"
    / "inference_mismatch_investigation"
    / "recent_live_like_policy_backtest"
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _normalise_live_symbol(symbol: Any) -> str:
    return _live_symbol_for_base(_base_from_symbol(str(symbol)))


def _stable_expansion_symbols(data_root: str, run_id: str) -> set[str]:
    return {
        _normalise_live_symbol(sym)
        for sym in load_active_untrained_symbol_expansion_allowlist(data_root, run_id)
    }


def _current_tradable_records(data_root: str, run_id: str) -> Tuple[List[SymbolRecord], Dict[str, Any]]:
    active = _load_active_symbols(None, data_root)
    trained = _trained_bases(data_root, run_id)
    expansion = _stable_expansion_symbols(data_root, run_id)
    expansion_bases = {_base_from_symbol(s) for s in expansion}
    allowed_bases = set(trained).union(expansion_bases)
    records: List[SymbolRecord] = []
    missing_cache: Dict[str, str] = {}
    for sym in active:
        base = _base_from_symbol(sym)
        if base not in allowed_bases:
            continue
        cache_symbol = _local_cache_symbol_for_base(data_root, base)
        if not cache_symbol:
            missing_cache[sym] = "no_local_hourly_cache"
            continue
        group = "baseline_trained" if base in trained else "active_untrained_expansion"
        records.append(SymbolRecord(sym, base, cache_symbol, group))
    meta = {
        "active_download_symbols": len(active),
        "trained_bases": len(trained),
        "stable_expansion_symbols": len(expansion),
        "stable_expansion_bases": len(expansion_bases),
        "tradable_records_with_cache": len(records),
        "missing_hourly_cache": missing_cache,
    }
    return records, meta


def _latest_common_end(store: PartitionedOHLCVStore, records: list[SymbolRecord], min_rows: int) -> pd.Timestamp:
    maxes: list[pd.Timestamp] = []
    probe_start = pd.Timestamp("2025-01-01", tz="UTC")
    for rec in records:
        df = store.load(rec.cache_symbol, columns=["close"], start_ts=probe_start)
        if len(df) >= min_rows:
            maxes.append(pd.Timestamp(df.index.max()).tz_convert("UTC"))
    if not maxes:
        raise RuntimeError("No tradable records have enough cached history")
    return max(maxes).floor("h")


def _attach_barrier_pct(
    rows: pd.DataFrame,
    *,
    panel: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    symbols = sorted(out["symbol"].dropna().astype(str).unique())
    barrier = _compute_policy_barrier_pct(dict(panel), symbols, dict(cfg))
    values = np.full(len(out), np.nan, dtype=np.float64)
    if isinstance(barrier, pd.DataFrame) and not barrier.empty:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        syms = out["symbol"].astype(str).to_numpy()
        for symbol in pd.Index(syms).unique():
            mask = syms == symbol
            if symbol not in barrier.columns:
                continue
            values[mask] = (
                pd.to_numeric(barrier[symbol], errors="coerce")
                .reindex(ts[mask])
                .to_numpy(dtype=np.float64)
            )
    out["barrier_pct"] = values
    out["barrier_pct_source"] = np.where(
        np.isfinite(values) & (values > 0.0),
        "raw_policy_barrier_pct",
        "fallback_0p02",
    )
    out["barrier_pct"] = pd.to_numeric(out["barrier_pct"], errors="coerce").fillna(0.02).clip(lower=0.005)
    return out


def _simulate_selected_rows(
    *,
    rows: pd.DataFrame,
    cache_by_live_symbol: Mapping[str, str],
    data_root: str,
    stop_params_by_strategy: Mapping[str, Mapping[str, Any]],
    path_len_hours: int,
    fee_bps: float,
    spread_bps: float,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if rows.empty:
        return pd.DataFrame(), []
    market_mode = "perps"
    ds = PartitionedOHLCVStore(_policy_market_data_root(data_root, market_mode), timeframe="1h")
    simulated_frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, Any]] = []
    cost_pct = (float(fee_bps) + float(spread_bps)) / 2.0 / 10000.0

    for strategy_id, grp in rows.groupby("strategy_id", sort=False):
        params = dict(stop_params_by_strategy.get(str(strategy_id), {}))
        if not params:
            diagnostics.append(
                {"strategy_id": str(strategy_id), "reason": "missing_stop_params", "rows": int(len(grp))}
            )
            continue
        sim_rows = grp.copy().reset_index(drop=True)
        live_symbols = sim_rows["symbol"].astype(str).copy()
        sim_rows["live_symbol"] = live_symbols
        sim_rows["cache_symbol"] = live_symbols.map(cache_by_live_symbol)
        sim_rows = sim_rows.dropna(subset=["cache_symbol"]).copy()
        if sim_rows.empty:
            diagnostics.append(
                {"strategy_id": str(strategy_id), "reason": "no_cache_symbols", "rows": int(len(grp))}
            )
            continue

        # Optimiser paths expect entry timestamp, while scoring timestamp is the
        # completed signal bar. Live policy entry reference is next-hour open.
        sim_rows["signal_timestamp"] = pd.to_datetime(sim_rows["timestamp"], utc=True)
        sim_rows["timestamp"] = sim_rows["signal_timestamp"] + pd.Timedelta(hours=1)
        sim_rows["rank_pct"] = pd.to_numeric(
            sim_rows.get("threshold_rank_score", sim_rows.get("policy_rank_pct")),
            errors="coerce",
        )
        side_name = str(sim_rows["side"].iloc[0]).lower()
        sim_rows["side_name"] = side_name
        sim_rows["side"] = 1.0 if side_name == "long" else -1.0
        path_rows = sim_rows.copy()
        path_rows["symbol"] = path_rows["cache_symbol"]
        paths = _fetch_policy_paths(path_rows, ds, path_len=max(2, int(path_len_hours) + 1))
        path_rows, delayed_paths = _apply_delayed_entry_execution_model(
            path_rows,
            paths,
            data_root=data_root,
            market_mode=market_mode,
        )
        finite_path = np.isfinite(delayed_paths[0]).all(axis=1)
        for arr in delayed_paths[1:]:
            finite_path &= np.isfinite(arr).all(axis=1)
        if not finite_path.any():
            diagnostics.append(
                {"strategy_id": str(strategy_id), "reason": "no_finite_paths", "rows": int(len(sim_rows))}
            )
            continue
        path_rows = path_rows.loc[finite_path].copy().reset_index(drop=True)
        delayed_paths = tuple(arr[finite_path] for arr in delayed_paths)
        metrics = simulate_and_score(
            path_rows,
            *delayed_paths,
            cost_pct=cost_pct,
            size_power=1.0,
            sl_mult=float(params.get("sl_mult", 1.0)),
            trailing_activation_mult=float(params.get("trailing_activation_mult", 1.0)),
            trailing_power=float(params.get("trailing_power", 1.5)),
            trailing_squash_divisor=float(params.get("trailing_squash_divisor", 2.0)),
            giveback_beta=float(params.get("giveback_beta", 0.5)),
            capital_protect_mfe_mult=float(params.get("capital_protect_mfe_mult", 0.0)),
            capital_protect_regression_frac=float(params.get("capital_protect_regression_frac", 0.45)),
            adverse_exit_enabled=bool(params.get("adverse_exit_enabled", False)),
            adverse_exit_min_mae_atr=float(params.get("adverse_exit_min_mae_atr", 1.0)),
            adverse_exit_min_speed=float(params.get("adverse_exit_min_speed", 0.3)),
            adverse_exit_theta_quantile=float(params.get("adverse_exit_theta_quantile", 0.75)),
            adverse_exit_theta=(
                _safe_float(params.get("adverse_exit_theta"))
                if np.isfinite(_safe_float(params.get("adverse_exit_theta")))
                else None
            ),
            adverse_exit_alpha=float(params.get("adverse_exit_alpha", 1.0)),
            adverse_exit_beta=float(params.get("adverse_exit_beta", 1.0)),
            adverse_exit_delta=float(params.get("adverse_exit_delta", 1.0)),
            adverse_exit_fast_bars=int(float(params.get("adverse_exit_fast_bars", 4))),
            adverse_exit_max_mfe_atr=float(params.get("adverse_exit_max_mfe_atr", 0.25)),
            # Hourly bars make optimiser's 15m concurrency accounting invalid.
            # Keep this as pre-portfolio edge diagnostics.
            max_concurrent_trades=1_000_000,
            max_concurrent_per_asset=1_000_000,
        )
        selected_mask = np.asarray(metrics.get("selected_mask", np.ones(len(path_rows), dtype=bool)), dtype=bool)
        selected_rows = path_rows.loc[selected_mask].copy().reset_index(drop=True)
        raw_gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
        gross_gains = np.asarray(metrics.get("gross_gains", []), dtype=np.float64)
        exit_bars = np.asarray(metrics.get("exit_bars", []), dtype=np.float64)
        exit_reason = list(metrics.get("exit_reason", []))
        n = min(len(selected_rows), len(raw_gains), len(gross_gains), len(exit_bars), len(exit_reason))
        selected_rows = selected_rows.iloc[:n].copy()
        selected_rows["net_return_live_like"] = raw_gains[:n]
        selected_rows["gross_return_live_like_before_cost"] = gross_gains[:n]
        selected_rows["exit_bars"] = exit_bars[:n]
        selected_rows["exit_reason"] = exit_reason[:n]
        selected_rows["fee_bps"] = float(fee_bps)
        selected_rows["spread_bps"] = float(spread_bps)
        selected_rows["round_trip_cost_bps"] = float(fee_bps) + float(spread_bps)
        simulated_frames.append(selected_rows)
        sources = path_rows.get("entry_execution_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        diagnostics.append(
            {
                "strategy_id": str(strategy_id),
                "reason": "ok",
                "input_selected_rows": int(len(grp)),
                "path_rows": int(len(path_rows)),
                "simulated_rows": int(len(selected_rows)),
                "delayed_1m_rows": int((path_rows.get("entry_execution_source") == "delayed_1m_intraminute_proxy").sum())
                if "entry_execution_source" in path_rows
                else 0,
                "entry_execution_sources": sources,
                "total_trades_after_sim_selection": int(metrics.get("total_trades", 0)),
                "mean_net_trade": float(metrics.get("mean_net_trade", np.nan)),
                "win_rate": float(metrics.get("win_rate", np.nan)),
                "full_sl_exit_count": int(metrics.get("full_sl_exit_count", 0)),
                "capital_protect_exit_count": int(metrics.get("capital_protect_exit_count", 0)),
                "trailing_exit_count": int(metrics.get("trailing_exit_count", 0)),
                "adverse_exit_count": int(metrics.get("adverse_exit_count", 0)),
            }
        )
    out = pd.concat(simulated_frames, ignore_index=True) if simulated_frames else pd.DataFrame()
    return out, diagnostics


def _summary(rows: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    cols = group_cols + [
        "n_trades",
        "trades_per_day",
        "gross_hit_rate",
        "net_hit_rate",
        "mean_gross_bps",
        "median_gross_bps",
        "mean_net_bps",
        "median_net_bps",
        "stop_loss_rate",
        "capital_protect_rate",
        "trailing_rate",
        "adverse_exit_rate",
        "delayed_1m_entry_coverage",
    ]
    if rows.empty:
        return pd.DataFrame(columns=cols)

    def agg(g: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(g["signal_timestamp"], utc=True, errors="coerce")
        days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0) if ts.notna().any() else 1.0
        gross = pd.to_numeric(g["gross_return_live_like_before_cost"], errors="coerce")
        net = pd.to_numeric(g["net_return_live_like"], errors="coerce")
        reasons = g["exit_reason"].astype(str)
        source = g.get("entry_execution_source", pd.Series("", index=g.index)).astype(str)
        return pd.Series(
            {
                "n_trades": int(len(g)),
                "trades_per_day": float(len(g) / days),
                "gross_hit_rate": float((gross > 0).mean()) if len(g) else np.nan,
                "net_hit_rate": float((net > 0).mean()) if len(g) else np.nan,
                "mean_gross_bps": float(gross.mean() * 10000.0),
                "median_gross_bps": float(gross.median() * 10000.0),
                "mean_net_bps": float(net.mean() * 10000.0),
                "median_net_bps": float(net.median() * 10000.0),
                "stop_loss_rate": float(reasons.eq("full_sl").mean()),
                "capital_protect_rate": float(reasons.eq("capital_protect").mean()),
                "trailing_rate": float(reasons.eq("trailing").mean()),
                "adverse_exit_rate": float(reasons.eq("adverse_exit").mean()),
                "delayed_1m_entry_coverage": float(source.eq("delayed_1m_intraminute_proxy").mean()),
            }
        )

    return rows.groupby(group_cols, dropna=False).apply(agg).reset_index()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = str(args.data_root)
    run_id = str(args.run_id)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records, universe_meta = _current_tradable_records(data_root, run_id)
    if args.max_symbols > 0:
        records = records[: int(args.max_symbols)]
    cfg = {**dict(DEFAULT_CFG), **_perp_data_cfg(data_root)}
    eval_records = list(records)
    context_records, missing_context_cache = _context_records_for_feature_computation(
        data_root=data_root,
        cfg=cfg,
        existing_bases={r.base for r in eval_records},
    )
    all_records_for_features = eval_records + context_records
    store = PartitionedOHLCVStore(root_dir=str(_market_data_root(data_root)), timeframe="1h")
    min_rows = int((args.warmup_hours or 24 * 120) + 24 * 7)
    latest_end = _latest_common_end(store, eval_records, min_rows=24 * 7)
    eval_end = latest_end - pd.Timedelta(hours=int(args.path_len_hours))
    may_start = pd.Timestamp(args.may_start, tz="UTC")
    june_start = pd.Timestamp(args.june_start, tz="UTC")
    eval_start = min(may_start, june_start)
    load_start = eval_start - pd.Timedelta(hours=int(args.warmup_hours))
    tprint(
        f"Recent live-like eval: load_start={load_start} eval_start={eval_start} "
        f"eval_end={eval_end} symbols={len(eval_records)}"
    )
    panel, kept, rejected = _load_panel(
        store,
        all_records_for_features,
        data_root=data_root,
        start_ts=load_start,
        end_ts=eval_end + pd.Timedelta(hours=int(args.path_len_hours) + 2),
    )
    eval_symbol_set = {r.live_symbol for r in eval_records}
    scoring_symbols = [r.live_symbol for r in kept if r.live_symbol in eval_symbol_set]
    group_by_symbol = {r.live_symbol: r.group for r in kept if r.live_symbol in eval_symbol_set}
    cache_by_live_symbol = {r.live_symbol: r.cache_symbol for r in kept if r.live_symbol in eval_symbol_set}
    base_by_symbol = {r.live_symbol: r.base for r in kept if r.live_symbol in eval_symbol_set}
    strategies = _load_policy_strategies(data_root, run_id)
    accepted_ids = {str(s.get("strategy_for_inference") or s.get("strategy_id")) for s in strategies}
    bundle = load_model_bundle(run_id, data_root)
    required = get_inference_required_feature_keys(bundle, accepted_ids)
    mask_keys = _mask_feature_keys(strategies)
    required.update(mask_keys)
    cfg["historical_inference_parity_allow_missing_live_sources"] = True
    feats = _historical_training_path_features(
        panel=panel,
        symbols=[r.live_symbol for r in kept],
        run_id=run_id,
        data_root=data_root,
        cfg=cfg,
        required_feature_keys=required,
    )
    mask_ok, feature_diagnostics = _feature_spans_eval_window(
        feats,
        mask_keys,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    pd.DataFrame(feature_diagnostics).to_csv(out_dir / "feature_window_diagnostics.csv", index=False)
    if not mask_ok:
        raise RuntimeError(f"Mask feature coverage failed; see {out_dir / 'feature_window_diagnostics.csv'}")

    orchestrator = ModelOrchestrator(bundle, {"disable_spike_filter": True, "inference_model_timing_enabled": False})
    rank_store = PolicyRankReferenceStore(data_root=data_root, run_id=run_id)
    scored_frames: List[pd.DataFrame] = []
    mask_diagnostics: List[Dict[str, Any]] = []
    for strat in strategies:
        sid = str(strat.get("strategy_for_inference") or strat.get("strategy_id"))
        tprint(f"Scoring recent strategy: {sid[:100]}")
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
            horizon_hours=int(args.path_len_hours),
            diagnostics=mask_diagnostics,
        )
        if not frame.empty:
            scored_frames.append(frame)
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    if not scored.empty:
        scored["group"] = scored["symbol"].map(group_by_symbol)
        scored["base"] = scored["symbol"].map(base_by_symbol)
        scored = _attach_barrier_pct(scored, panel=panel, cfg=cfg)
    selected = scored[scored["selected"]].copy() if not scored.empty else pd.DataFrame()
    stop_params = load_simple_policy_stop_params_by_strategy(data_root, run_id)
    simulated, sim_diag = _simulate_selected_rows(
        rows=selected,
        cache_by_live_symbol=cache_by_live_symbol,
        data_root=data_root,
        stop_params_by_strategy=stop_params,
        path_len_hours=int(args.path_len_hours),
        fee_bps=float(args.fee_bps),
        spread_bps=float(args.spread_bps),
    )
    if not simulated.empty:
        sim_ts = pd.to_datetime(simulated["signal_timestamp"], utc=True, errors="coerce")
        simulated["window"] = np.where(sim_ts >= june_start, "june_to_now", "may_to_now")
        simulated.loc[sim_ts < may_start, "window"] = "pre_may"
        simulated = simulated[simulated["window"].isin(["may_to_now", "june_to_now"])].copy()

    scored.to_parquet(out_dir / "recent_live_like_scored_rows.parquet", index=False)
    selected.to_parquet(out_dir / "recent_live_like_selected_rows_pre_exit.parquet", index=False)
    simulated.to_parquet(out_dir / "recent_live_like_simulated_trades.parquet", index=False)
    pd.DataFrame(mask_diagnostics).to_csv(out_dir / "mask_diagnostics.csv", index=False)
    pd.DataFrame(sim_diag).to_csv(out_dir / "simulation_diagnostics.csv", index=False)
    overall = _summary(simulated, ["window"])
    by_strategy = _summary(simulated, ["window", "strategy_id", "side_name"])
    by_group = _summary(simulated, ["window", "group"])
    overall.to_csv(out_dir / "summary_overall.csv", index=False)
    by_strategy.to_csv(out_dir / "summary_by_strategy.csv", index=False)
    by_group.to_csv(out_dir / "summary_by_symbol_group.csv", index=False)
    payload = {
        "schema": "recent_live_like_policy_backtest_v1",
        "run_id": run_id,
        "data_root": data_root,
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "may_start": str(may_start),
        "june_start": str(june_start),
        "path_len_hours": int(args.path_len_hours),
        "fee_bps": float(args.fee_bps),
        "spread_bps": float(args.spread_bps),
        "round_trip_cost_bps": float(args.fee_bps) + float(args.spread_bps),
        "universe": universe_meta,
        "scoring_symbols": len(scoring_symbols),
        "feature_context_symbols": len(kept),
        "missing_context_cache": missing_context_cache,
        "rejected_cached_symbols": rejected,
        "scored_rows": int(len(scored)),
        "selected_rows_pre_exit": int(len(selected)),
        "simulated_trades": int(len(simulated)),
        "notes": [
            "Uses deployed final-fit models, deployed rank references, deployed thresholds, and training-path feature generation.",
            "Entry is next-hour open, then t+5 delayed entry is applied from cached 1m execution bars when present.",
            "No network download is attempted unless EPM_SIMPLE_POLICY_1M_DOWNLOAD is overridden.",
            "Exit policy uses simple_policy_optimiser.simulate_and_score with deployed simple-policy stop/trailing params.",
            "Forward paths are hourly OHLCV, so this is a conservative recent-window live-like diagnostic, not a 15m-perfect portfolio replay.",
            "Portfolio concurrency is disabled because optimiser concurrency assumes 15m bars; this is pre-portfolio edge decay evidence.",
        ],
        "overall": overall.to_dict(orient="records"),
        "by_group": by_group.to_dict(orient="records"),
    }
    (out_dir / "recent_live_like_policy_backtest_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    print(json.dumps(payload, indent=2, default=str))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--may-start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--june-start", default="2026-06-01T00:00:00Z")
    parser.add_argument("--warmup-hours", type=int, default=24 * 120)
    parser.add_argument("--path-len-hours", type=int, default=24)
    parser.add_argument("--fee-bps", type=float, default=7.0)
    parser.add_argument("--spread-bps", type=float, default=97.32886619027215)
    parser.add_argument("--max-symbols", type=int, default=0)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
