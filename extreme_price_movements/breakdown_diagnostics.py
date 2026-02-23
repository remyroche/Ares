from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extreme_price_movements.engine import simulate_trade_hourly
from extreme_price_movements.policy_ml import load_best_policy_params_from_optimise


@dataclass
class EventBuildResult:
    events: pd.DataFrame
    row_event_id: pd.Series


def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out = out.set_index("timestamp")
        elif "ts" in out.columns:
            out = out.set_index("ts")
        else:
            raise ValueError("DataFrame must have DatetimeIndex or timestamp/ts column")
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def build_breakdown_events(
    df: pd.DataFrame,
    lookback_h: int = 12,
    trigger: float = 0.08,
    decluster_h: int = 6,
    max_event_h: int = 72,
) -> EventBuildResult:
    """Build de-clustered breakdown events from abs(lookback return) threshold crossings."""
    dfx = _to_ts_index(df)
    if "close" not in dfx.columns:
        raise ValueError("build_breakdown_events requires 'close' column")

    ret_lb = dfx["close"].pct_change(int(lookback_h)).fillna(0.0)
    active = ret_lb.abs() >= float(trigger)
    event_start_mask = active & (~active.shift(1, fill_value=False))

    row_event_id = pd.Series(-1, index=dfx.index, dtype=np.int64)
    events: List[Dict[str, Any]] = []

    trigger_exit = float(trigger) * 0.8
    starts = np.flatnonzero(event_start_mask.values)
    evt_id = 0
    i = 0
    n = len(dfx)

    while i < len(starts):
        s = int(starts[i])
        # skip if already inside existing event assignment
        if row_event_id.iloc[s] != -1:
            i += 1
            continue

        end = min(n - 1, s + int(max_event_h))
        calm_run = 0
        for j in range(s, min(n, s + int(max_event_h) + 1)):
            if abs(float(ret_lb.iloc[j])) < trigger_exit:
                calm_run += 1
            else:
                calm_run = 0
            if calm_run >= int(decluster_h):
                end = j
                break

        sub = ret_lb.iloc[s : end + 1]
        peak_idx = int(sub.abs().idxmax().value) if len(sub) else int(dfx.index[s].value)
        peak_move = float(sub.abs().max()) if len(sub) else 0.0
        trough_idx = int(sub.idxmin().value) if len(sub) else int(dfx.index[s].value)

        shock_sign = 1 if float(ret_lb.iloc[s]) >= 0 else -1
        side_hint = "follow" if shock_sign > 0 else "fade"

        row_event_id.iloc[s : end + 1] = evt_id
        events.append(
            {
                "event_id": evt_id,
                "start_idx": int(dfx.index[s].value),
                "end_idx": int(dfx.index[end].value),
                "side_hint": side_hint,
                "peak_move": peak_move,
                "peak_idx": peak_idx,
                "trough_idx": trough_idx,
                "shock_sign": shock_sign,
                "start_ts": dfx.index[s],
                "end_ts": dfx.index[end],
            }
        )

        # de-cluster by skipping starts inside [s, end + decluster]
        skip_until = end + int(decluster_h)
        while i < len(starts) and int(starts[i]) <= skip_until:
            i += 1
        evt_id += 1

        events_df = pd.DataFrame(events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=[
            "event_id", "start_idx", "end_idx", "side_hint", "peak_move",
            "peak_idx", "trough_idx", "shock_sign", "start_ts", "end_ts",
        ])
    return EventBuildResult(events=events_df, row_event_id=row_event_id)


def _exit_code_from_reason(reason: str, ret: float) -> int:
    r = str(reason or "")
    if r in {"stop_loss", "early_invalidation"}:
        return 0
    if r in {"trailing_stop", "take_profit", "tp", "giveback_exit"}:
        return 2
    if r in {"time_exit", "timeout", "no_entry", "limit_not_filled"}:
        return 1
    return 2 if ret > 0 else (0 if ret < 0 else 1)


def _simulate_trade_for_event(
    ohlc: pd.DataFrame,
    atr_pct: pd.Series,
    entry_ts: pd.Timestamp,
    direction: int,
    policy_params: Dict[str, Any],
    max_hold_hours: int,
) -> Dict[str, Any]:
    side = "long" if int(direction) > 0 else "short"
    if entry_ts not in ohlc.index:
        return {
            "r_policy": np.nan,
            "u_policy": np.nan,
            "exit_code": -1,
            "mae_ret": np.nan,
            "mfe_ret": np.nan,
            "duration": np.nan,
            "reason": "missing_entry",
            "gap_through_stop": False,
            "gap_slippage_beyond_stop": 0.0,
            "entry_ts": entry_ts,
            "exit_ts": pd.NaT,
        }
    entry_px = float(ohlc.loc[entry_ts, "open"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return {
            "r_policy": np.nan,
            "u_policy": np.nan,
            "exit_code": -1,
            "mae_ret": np.nan,
            "mfe_ret": np.nan,
            "duration": np.nan,
            "reason": "bad_entry",
            "gap_through_stop": False,
            "gap_slippage_beyond_stop": 0.0,
            "entry_ts": entry_ts,
            "exit_ts": pd.NaT,
        }

    ret, exit_ts, reason, extras = simulate_trade_hourly(
        o_s=ohlc["open"],
        h_s=ohlc["high"],
        l_s=ohlc["low"],
        c_s=ohlc["close"],
        feats_s=atr_pct,
        ts_entry=entry_ts,
        entry_px=entry_px,
        side=side,
        cfg=dict(policy_params),
        max_hold_hours=int(max_hold_hours),
        exchange=None,
        symbol=None,
        cost=None,
    )

    ret = float(ret)
    exit_code = _exit_code_from_reason(reason, ret)
    u_policy = float(np.log1p(max(-0.999999, ret)))
    dur_h = float(max(0.0, (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 3600.0))
    extras = extras or {}
    mae = float(extras.get("mae_pct", np.nan))
    mfe = float(extras.get("mfe_pct", np.nan))

    # Gap-through-stop approximation at first bar after entry
    gap_through_stop = False
    gap_slip = 0.0
    sl_pct = float(extras.get("sl_pct", np.nan)) if np.isfinite(extras.get("sl_pct", np.nan)) else np.nan
    loc = ohlc.index.get_indexer([entry_ts])[0]
    if loc >= 1 and loc < len(ohlc) and np.isfinite(sl_pct):
        prev_close = float(ohlc["close"].iloc[loc - 1])
        open_now = float(ohlc["open"].iloc[loc])
        if side == "long":
            stop_px = entry_px * (1.0 - sl_pct)
            gap_through_stop = bool(open_now < stop_px)
            if gap_through_stop:
                gap_slip = float((stop_px - open_now) / max(entry_px, 1e-12))
        else:
            stop_px = entry_px * (1.0 + sl_pct)
            gap_through_stop = bool(open_now > stop_px)
            if gap_through_stop:
                gap_slip = float((open_now - stop_px) / max(entry_px, 1e-12))

    return {
        "r_policy": ret,
        "u_policy": u_policy,
        "exit_code": exit_code,
        "mae_ret": mae,
        "mfe_ret": mfe,
        "duration": dur_h,
        "reason": str(reason),
        "gap_through_stop": bool(gap_through_stop),
        "gap_slippage_beyond_stop": float(gap_slip),
        "entry_ts": pd.Timestamp(entry_ts),
        "exit_ts": pd.Timestamp(exit_ts),
    }


def policy_profitability_sweep(
    df: pd.DataFrame,
    events: pd.DataFrame,
    policy_params: Dict[str, Any],
    entry_offsets: Iterable[int],
    directions: Iterable[str],
    cost_modes: Iterable[float],
    lookback_h: int = 12,
    max_event_h: int = 72,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep timing/direction/cost over event starts using engine-identical simulation."""
    dfx = _to_ts_index(df)
    if "atr_pct" in dfx.columns:
        atr = dfx["atr_pct"].astype(float).ffill().fillna(0.02)
    else:
        atr = pd.Series(0.02, index=dfx.index)

    trade_rows: List[Dict[str, Any]] = []

    for _, ev in events.iterrows():
        start_ts = pd.Timestamp(ev["start_ts"])
        shock_sign = int(ev["shock_sign"])
        for off in entry_offsets:
            entry_ts = start_ts + pd.Timedelta(hours=int(off))
            for dname in directions:
                if dname == "follow":
                    direction = shock_sign
                elif dname == "fade":
                    direction = -shock_sign
                else:
                    continue

                for cm in cost_modes:
                    p = dict(policy_params)
                    if "fee_bps" in p:
                        p["fee_bps"] = float(p.get("fee_bps", 0.0)) * float(cm)
                    if "slippage_bps" in p:
                        p["slippage_bps"] = float(p.get("slippage_bps", 0.0)) * float(cm)
                    out = _simulate_trade_for_event(
                        ohlc=dfx[["open", "high", "low", "close"]],
                        atr_pct=atr,
                        entry_ts=entry_ts,
                        direction=direction,
                        policy_params=p,
                        max_hold_hours=int(max_event_h),
                    )
                    trade_rows.append(
                        {
                            "event_id": int(ev["event_id"]),
                            "shock_sign": shock_sign,
                            "offset_h": int(off),
                            "direction": str(dname),
                            "cost_mult": float(cm),
                            **out,
                        }
                    )

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        return trades, pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        ex = g["exit_code"].values
        tp = np.mean(ex == 2) if len(ex) else np.nan
        sl = np.mean(ex == 0) if len(ex) else np.nan
        to = np.mean(ex == 1) if len(ex) else np.nan
        return pd.Series(
            {
                "n": len(g),
                "mean_u": float(np.nanmean(g["u_policy"])),
                "median_u": float(np.nanmedian(g["u_policy"])),
                "mean_r": float(np.nanmean(g["r_policy"])),
                "median_r": float(np.nanmedian(g["r_policy"])),
                "tp_rate": float(tp),
                "sl_rate": float(sl),
                "to_rate": float(to),
                "mean_mae": float(np.nanmean(g["mae_ret"])),
                "mean_mfe": float(np.nanmean(g["mfe_ret"])),
                "mean_duration_h": float(np.nanmean(g["duration"])),
            }
        )

    summary = (
        trades.groupby(["offset_h", "direction", "cost_mult"], dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
        .sort_values(["cost_mult", "direction", "offset_h"])
    )
    return trades, summary


def trigger_threshold_sweep(
    df: pd.DataFrame,
    lookback_h: int,
    triggers: Iterable[float],
    decluster_h: int,
    max_event_h: int,
    baseline_trigger: float,
    policy_params: Dict[str, Any],
    entry_offsets_h: Iterable[int],
    directions: Iterable[str],
    cost_mult: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[float, EventBuildResult]]:
    baseline = build_breakdown_events(df, lookback_h, baseline_trigger, decluster_h, max_event_h)
    base_set = set(baseline.events["event_id"].tolist())

    rows = []
    event_map: Dict[float, EventBuildResult] = {float(baseline_trigger): baseline}

    for tr in triggers:
        evb = build_breakdown_events(df, lookback_h, float(tr), decluster_h, max_event_h)
        event_map[float(tr)] = evb
        cur_set = set(evb.events["event_id"].tolist())
        # event IDs are per build, so use interval overlap approximation by start timestamps
        bstarts = set(pd.to_datetime(baseline.events.get("start_ts", pd.Series([], dtype="datetime64[ns, UTC]"))).astype("int64").tolist())
        cstarts = set(pd.to_datetime(evb.events.get("start_ts", pd.Series([], dtype="datetime64[ns, UTC]"))).astype("int64").tolist())
        inter = len(bstarts & cstarts)
        uni = len(bstarts | cstarts)
        jacc = float(inter / uni) if uni > 0 else np.nan

        _, ssum = policy_profitability_sweep(
            df=df,
            events=evb.events,
            policy_params=policy_params,
            entry_offsets=entry_offsets_h,
            directions=directions,
            cost_modes=[float(cost_mult)],
            lookback_h=lookback_h,
            max_event_h=max_event_h,
        )
        if len(ssum):
            best = ssum.loc[ssum["mean_u"].idxmax()]
            best_u = float(best["mean_u"])
            best_offset = int(best["offset_h"])
            best_direction = str(best["direction"])
        else:
            best_u = np.nan
            best_offset = 0
            best_direction = "na"

        rows.append(
            {
                "trigger": float(tr),
                "n_events": int(len(evb.events)),
                "jaccard_vs_baseline": jacc,
                "best_mean_u": best_u,
                "earliest_profitable_offset": best_offset,
                "best_direction": best_direction,
            }
        )

    out = pd.DataFrame(rows).sort_values("trigger")
    return out, event_map


def direction_confusion_report(sweep_summary: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if sweep_summary.empty:
        return out
    s = sweep_summary.copy()
    base = s[s["cost_mult"] == s["cost_mult"].min()]
    g = base.groupby("direction")["mean_u"].mean().to_dict()
    out["direction_mean_u"] = {str(k): float(v) for k, v in g.items()}
    if g:
        out["best_direction_overall"] = max(g.items(), key=lambda kv: kv[1])[0]
        vals = list(g.values())
        out["direction_gap"] = float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0
    return out


def gap_slippage_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["is_sl"] = t["exit_code"] == 0
    grp = t.groupby("exit_code", dropna=False).agg(
        n=("exit_code", "size"),
        gap_through_rate=("gap_through_stop", "mean"),
        mean_gap_slip=("gap_slippage_beyond_stop", "mean"),
        mean_u=("u_policy", "mean"),
    )
    grp = grp.reset_index()
    return grp


def _plot_offset_direction_mean_u(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    if summary.empty:
        plt.title("offset vs mean_u (no data)")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        return
    for d in sorted(summary["direction"].unique()):
        g = summary[(summary["direction"] == d) & (summary["cost_mult"] == summary["cost_mult"].min())]
        g = g.sort_values("offset_h")
        plt.plot(g["offset_h"], g["mean_u"], marker="o", label=str(d))
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel("Entry offset (h)")
    plt.ylabel("Mean u_policy")
    plt.title("offset_direction_mean_u")
    plt.legend()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_offset_direction_rates(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    if summary.empty:
        plt.title("TP/SL/TO rates (no data)")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        return
    g = summary[(summary["cost_mult"] == summary["cost_mult"].min()) & (summary["direction"] == "follow")].sort_values("offset_h")
    if len(g):
        plt.plot(g["offset_h"], g["tp_rate"], label="TP")
        plt.plot(g["offset_h"], g["sl_rate"], label="SL")
        plt.plot(g["offset_h"], g["to_rate"], label="TO")
    plt.xlabel("Entry offset (h)")
    plt.ylabel("Rate")
    plt.title("offset_direction_tp_sl_to_rates (follow, baseline cost)")
    plt.legend()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_trigger_jaccard(trigger_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    if len(trigger_df):
        plt.plot(trigger_df["trigger"], trigger_df["jaccard_vs_baseline"], marker="o")
    plt.xlabel("Trigger")
    plt.ylabel("Jaccard vs baseline")
    plt.title("trigger_uniqueness_jaccard")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_surface(trigger_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    if len(trigger_df):
        sc = plt.scatter(
            trigger_df["trigger"],
            trigger_df["earliest_profitable_offset"],
            c=trigger_df["best_mean_u"],
            cmap="viridis",
            s=80,
        )
        plt.colorbar(sc, label="best_mean_u")
    plt.xlabel("Trigger")
    plt.ylabel("Earliest profitable offset (h)")
    plt.title("trigger_vs_best_offset_surface")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_gap_rate(gap_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    if len(gap_df):
        plt.bar(gap_df["exit_code"].astype(str), gap_df["gap_through_rate"])
    plt.xlabel("Exit code")
    plt.ylabel("Gap-through-stop rate")
    plt.title("gap_through_stop_rate")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_cost_stress(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    if len(summary):
        x = []
        y = []
        for cm, g in summary.groupby("cost_mult"):
            x.append(float(cm))
            y.append(float(g["mean_u"].max()))
        ord_idx = np.argsort(x)
        x = np.array(x)[ord_idx]
        y = np.array(y)[ord_idx]
        plt.plot(x, y, marker="o")
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel("Cost stress multiplier")
    plt.ylabel("Best mean_u")
    plt.title("cost_stress_sweep")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _read_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = p.read_text()
    if p.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception as e:
        raise RuntimeError("YAML config requires pyyaml installed, or use JSON config") from e


def run_breakdown_diagnostics(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    runp = Path(run_dir)
    out_dir = runp / "breakdown_diagnostics"
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ohlc_path = Path(cfg.get("ohlc_path", runp / "ohlc.parquet"))
    if not ohlc_path.exists():
        raise FileNotFoundError(f"OHLC path not found: {ohlc_path}")
    df = pd.read_parquet(ohlc_path)
    df = _to_ts_index(df)

    lookback_h = int(cfg.get("lookback_h", 12))
    baseline_trigger = float(cfg.get("baseline_trigger", 0.08))
    trigger_sweep = list(cfg.get("trigger_sweep", [0.06, 0.07, 0.08, 0.09, 0.10]))
    decluster_h = int(cfg.get("decluster_h", 6))
    max_event_h = int(cfg.get("max_event_h", 72))
    entry_offsets_h = list(cfg.get("entry_offsets_h", [-12, -6, -4, -2, -1, 0, 1, 2, 4, 6, 12]))
    directions = list(cfg.get("directions", ["follow", "fade"]))
    cost_mults = list(cfg.get("cost_stress_multipliers", [1.0, 1.25, 1.5, 2.0]))

    optimise_run_dir = str(cfg.get("optimise_run_dir", run_dir))
    _rp = Path(optimise_run_dir)
    data_root = str(_rp.parent.parent) if _rp.name == "artifacts" else str(_rp.parent.parent)
    run_id = _rp.name
    policy_blob = load_best_policy_params_from_optimise(data_root=data_root, run_id=run_id)
    if isinstance(policy_blob, dict) and "buckets" in policy_blob and policy_blob["buckets"]:
        # choose first bucket config for diagnostics unless overridden
        first_key = sorted(policy_blob["buckets"].keys())[0]
        policy_params = dict(policy_blob["buckets"][first_key])
    else:
        policy_params = dict(policy_blob) if isinstance(policy_blob, dict) else {}
    if not policy_params:
        policy_params = {
            "tp_mult": 1.0,
            "sl_mult": 0.5,
            "trail_mult": 0.25,
            "fee_bps": 25.0,
            "vol_lo": 0.03,
            "vol_hi": 0.06,
        }

    ev = build_breakdown_events(
        df=df,
        lookback_h=lookback_h,
        trigger=baseline_trigger,
        decluster_h=decluster_h,
        max_event_h=max_event_h,
    )
    events = ev.events
    events.to_parquet(tables_dir / "events.parquet", index=False)

    trades, sweep = policy_profitability_sweep(
        df=df,
        events=events,
        policy_params=policy_params,
        entry_offsets=entry_offsets_h,
        directions=directions,
        cost_modes=cost_mults,
        lookback_h=lookback_h,
        max_event_h=max_event_h,
    )
    trades.to_parquet(tables_dir / "trades_policy_sweep.parquet", index=False)
    sweep.to_parquet(tables_dir / "sweep_by_offset.parquet", index=False)

    trig_df, _ = trigger_threshold_sweep(
        df=df,
        lookback_h=lookback_h,
        triggers=trigger_sweep,
        decluster_h=decluster_h,
        max_event_h=max_event_h,
        baseline_trigger=baseline_trigger,
        policy_params=policy_params,
        entry_offsets_h=entry_offsets_h,
        directions=directions,
        cost_mult=1.0,
    )
    trig_df.to_parquet(tables_dir / "sweep_by_trigger.parquet", index=False)

    gap_df = gap_slippage_audit(trades)
    gap_df.to_parquet(tables_dir / "gap_audit.parquet", index=False)

    _plot_offset_direction_mean_u(sweep, plots_dir / "offset_direction_mean_u.png")
    _plot_offset_direction_rates(sweep, plots_dir / "offset_direction_tp_sl_to_rates.png")
    _plot_trigger_jaccard(trig_df, plots_dir / "trigger_uniqueness_jaccard.png")
    _plot_surface(trig_df, plots_dir / "trigger_vs_best_offset_surface.png")
    _plot_gap_rate(gap_df, plots_dir / "gap_through_stop_rate.png")
    _plot_cost_stress(sweep, plots_dir / "cost_stress_sweep.png")

    dir_rep = direction_confusion_report(sweep)

    # verdict flags
    policy_profitable_at_any_offset = bool((sweep["mean_u"] > 0).any()) if len(sweep) else False
    timing_sensitive = False
    if len(sweep):
        best = sweep.loc[sweep["mean_u"].idxmax()]
        timing_sensitive = abs(int(best["offset_h"])) > 1
    direction_ambiguous = bool(abs(float(dir_rep.get("direction_gap", 0.0))) < 0.0005)
    gap_risk_material = False
    if len(gap_df):
        sl_row = gap_df[gap_df["exit_code"] == 0]
        if len(sl_row):
            gap_risk_material = bool(float(sl_row["gap_through_rate"].iloc[0]) > 0.15 or float(sl_row["mean_gap_slip"].iloc[0]) > 0.002)
    cost_sensitivity_high = False
    if len(sweep):
        by_cost = sweep.groupby("cost_mult")["mean_u"].max().sort_index()
        if len(by_cost) >= 2:
            cost_sensitivity_high = bool((by_cost.iloc[0] > 0) and (by_cost.iloc[-1] <= 0))

    verdict = {
        "policy_profitable_at_any_offset": policy_profitable_at_any_offset,
        "timing_sensitive": bool(timing_sensitive),
        "direction_ambiguous": direction_ambiguous,
        "gap_risk_material": gap_risk_material,
        "cost_sensitivity_high": cost_sensitivity_high,
        "recommendations": {
            "policy_profitable_at_any_offset": "If false, re-check policy params / optimize overfit under cost stress.",
            "timing_sensitive": "If true, move entry timing earlier/later by best offset and retest OOS.",
            "direction_ambiguous": "If true, add regime-conditioned direction model (fade vs follow).",
            "gap_risk_material": "If true, include gap-through-stop repricing and widen buffers/limits.",
            "cost_sensitivity_high": "If true, reduce turnover and tighten cost assumptions in training objective.",
        },
    }

    report = {
        "config": {
            "lookback_h": lookback_h,
            "baseline_trigger": baseline_trigger,
            "trigger_sweep": trigger_sweep,
            "decluster_h": decluster_h,
            "max_event_h": max_event_h,
            "entry_offsets_h": entry_offsets_h,
            "directions": directions,
            "cost_stress_multipliers": cost_mults,
        },
        "counts": {
            "n_events": int(len(events)),
            "n_policy_trades": int(len(trades)),
        },
        "direction_report": dir_rep,
        "verdict": verdict,
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="BreakdownDiagnostics")
    ap.add_argument("--config", type=str, default=None, help="JSON/YAML config path")
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory for outputs and data")
    args = ap.parse_args()

    cfg = _read_config(args.config)
    report = run_breakdown_diagnostics(cfg=cfg, run_dir=args.run_dir)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
