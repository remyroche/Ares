#!/usr/bin/env python3
"""Backtest loss-streak kill switches and hit-rate surprise overlays.

This is intentionally an overlay on an existing portfolio replay decision file:
it preserves the candidate universe, ranking, thresholds, sizing, and accepted
execution outcomes from the reference replay, then tests whether additional
loss-streak or hit-rate-surprise guards would have blocked later accepted rows.
It is conservative: a blocked trade does not free capital to admit a previously
rejected candidate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DECISIONS = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_may_july_combined_20260708/"
    "threshold_basis_ablation_may_june_july_weighted_gmm_posterior_8d_v2_overlay_parity/"
    "live_compatible_overlay_vs_hard8d/"
    "ev_target_archetype_reachable_match_current_activity_8d_decisions.parquet"
)


@dataclass(frozen=True)
class GuardConfig:
    arm: str
    global_trigger: int = 0
    global_cooldown_hours: float = 0.0
    archetype_trigger: int = 0
    archetype_cooldown_hours: float = 0.0
    hr_enabled: bool = False
    hr_lookback_trades: int = 0
    hr_min_trades: int = 0
    hr_negative_surprise_threshold: float = 0.0


@dataclass
class OpenTrade:
    exit_timestamp: pd.Timestamp
    symbol: str
    side: str
    policy_archetype: str
    position_size: float
    net_return: float
    gross_return: float
    exit_reason: str


def _manual_block_until() -> pd.Timestamp:
    return pd.Timestamp("2262-01-01", tz="UTC")


def _block_until(timestamp: pd.Timestamp, hours: float) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if float(hours) > 0.0:
        return ts + pd.Timedelta(hours=float(hours))
    return _manual_block_until()


def _week_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return (ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.date.astype(str)


def _load_decisions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "accepted" not in df.columns:
        raise ValueError(f"{path} has no accepted column")
    required = {
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "policy_archetype",
        "position_size",
        "position_net_return",
        "position_gross_return",
        "position_exit_timestamp",
        "position_exit_reason",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["position_exit_timestamp"] = pd.to_datetime(
        out["position_exit_timestamp"], utc=True, errors="coerce"
    )
    out["accepted"] = out["accepted"].astype(bool)
    out["policy_archetype"] = out["policy_archetype"].fillna("missing").astype(str)
    out["side"] = out["side"].fillna(out.get("side_name", "")).astype(str)
    out["position_size"] = pd.to_numeric(out["position_size"], errors="coerce").fillna(0.0)
    out["position_net_return"] = pd.to_numeric(
        out["position_net_return"], errors="coerce"
    )
    out["position_gross_return"] = pd.to_numeric(
        out["position_gross_return"], errors="coerce"
    )
    out["position_exit_reason"] = out["position_exit_reason"].fillna("").astype(str)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    out["_row_order"] = np.arange(len(out), dtype=np.int64)
    return out.sort_values(["timestamp", "_row_order"]).reset_index(drop=True)


def _hit_rate_block(
    history: Dict[str, List[bool]],
    global_history: List[bool],
    archetype: str,
    cfg: GuardConfig,
) -> Tuple[bool, Dict[str, Any]]:
    if not cfg.hr_enabled:
        return False, {}
    lookback = max(1, int(cfg.hr_lookback_trades))
    min_trades = max(1, int(cfg.hr_min_trades))
    arch_hist = history.get(archetype, [])
    recent = arch_hist[-lookback:]
    if len(recent) < min_trades:
        return False, {
            "hr_recent_n": len(recent),
            "hr_recent_hit_rate": np.nan,
            "hr_baseline_hit_rate": np.nan,
            "hr_surprise": np.nan,
        }
    prior_baseline = arch_hist[: -len(recent)] if len(arch_hist) > len(recent) else []
    baseline_source = prior_baseline if prior_baseline else global_history
    if not baseline_source:
        return False, {
            "hr_recent_n": len(recent),
            "hr_recent_hit_rate": float(np.mean(recent)),
            "hr_baseline_hit_rate": np.nan,
            "hr_surprise": np.nan,
        }
    recent_hit = float(np.mean(recent))
    baseline_hit = float(np.mean(baseline_source))
    surprise = recent_hit - baseline_hit
    return bool(surprise <= -float(cfg.hr_negative_surprise_threshold)), {
        "hr_recent_n": len(recent),
        "hr_recent_hit_rate": recent_hit,
        "hr_baseline_hit_rate": baseline_hit,
        "hr_surprise": surprise,
    }


def simulate_overlay(decisions: pd.DataFrame, cfg: GuardConfig) -> pd.DataFrame:
    open_trades: List[OpenTrade] = []
    consecutive_losses = 0
    archetype_losses: Dict[str, int] = {}
    global_block_until: Optional[pd.Timestamp] = None
    archetype_block_until: Dict[str, pd.Timestamp] = {}
    history: Dict[str, List[bool]] = {}
    global_history: List[bool] = []
    rows: List[Dict[str, Any]] = []

    def close_due(ts: pd.Timestamp) -> None:
        nonlocal open_trades, consecutive_losses, global_block_until
        still_open: List[OpenTrade] = []
        for trade in open_trades:
            if trade.exit_timestamp <= ts:
                pnl = float(trade.position_size) * float(trade.net_return)
                was_win = pnl > 0.0
                history.setdefault(trade.policy_archetype, []).append(bool(was_win))
                global_history.append(bool(was_win))
                if was_win:
                    consecutive_losses = 0
                    archetype_losses[trade.policy_archetype] = 0
                else:
                    consecutive_losses += 1
                    archetype_losses[trade.policy_archetype] = (
                        int(archetype_losses.get(trade.policy_archetype, 0)) + 1
                    )
                    if (
                        int(cfg.archetype_trigger) > 0
                        and archetype_losses[trade.policy_archetype]
                        >= int(cfg.archetype_trigger)
                    ):
                        archetype_block_until[trade.policy_archetype] = _block_until(
                            trade.exit_timestamp, float(cfg.archetype_cooldown_hours)
                        )
                    if (
                        int(cfg.global_trigger) > 0
                        and consecutive_losses >= int(cfg.global_trigger)
                    ):
                        global_block_until = _block_until(
                            trade.exit_timestamp, float(cfg.global_cooldown_hours)
                        )
            else:
                still_open.append(trade)
        open_trades = still_open
        if global_block_until is not None and global_block_until <= ts:
            global_block_until = None
            consecutive_losses = 0
        expired = [
            key for key, until in archetype_block_until.items() if pd.Timestamp(until) <= ts
        ]
        for key in expired:
            archetype_block_until.pop(key, None)
            archetype_losses[key] = 0

    for _, row in decisions.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        close_due(ts)
        baseline_accepted = bool(row["accepted"])
        accepted = False
        reason = str(row.get("rejection_reason", ""))
        hr_info: Dict[str, Any] = {}
        archetype = str(row.get("policy_archetype", "missing") or "missing")
        if baseline_accepted:
            if global_block_until is not None and global_block_until > ts:
                reason = "global_loss_streak_block"
            elif archetype_block_until.get(archetype) is not None and archetype_block_until[archetype] > ts:
                reason = "archetype_loss_streak_block"
            else:
                hr_blocked, hr_info = _hit_rate_block(
                    history, global_history, archetype, cfg
                )
                if hr_blocked:
                    reason = "hit_rate_surprise_block"
                else:
                    accepted = True
                    reason = "accepted"
                    exit_ts = pd.Timestamp(row["position_exit_timestamp"])
                    if pd.notna(exit_ts):
                        open_trades.append(
                            OpenTrade(
                                exit_timestamp=exit_ts,
                                symbol=str(row.get("symbol", "")),
                                side=str(row.get("side", "")),
                                policy_archetype=archetype,
                                position_size=float(row.get("position_size", 0.0) or 0.0),
                                net_return=float(row.get("position_net_return", 0.0) or 0.0),
                                gross_return=float(
                                    row.get("position_gross_return", 0.0) or 0.0
                                ),
                                exit_reason=str(row.get("position_exit_reason", "")),
                            )
                        )
        out = row.to_dict()
        out.update(
            {
                "arm": cfg.arm,
                "overlay_accepted": bool(accepted),
                "overlay_rejection_reason": reason,
                "baseline_accepted": baseline_accepted,
                "global_loss_streak": int(consecutive_losses),
                "archetype_loss_streak": int(archetype_losses.get(archetype, 0)),
                "global_block_until": (
                    global_block_until.isoformat() if global_block_until is not None else ""
                ),
                "archetype_block_until": (
                    archetype_block_until[archetype].isoformat()
                    if archetype in archetype_block_until
                    else ""
                ),
                **hr_info,
            }
        )
        rows.append(out)

    if len(decisions):
        close_due(pd.Timestamp(decisions["timestamp"].max()) + pd.Timedelta(days=365))
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame, cfg: GuardConfig) -> Dict[str, Any]:
    accepted = frame.loc[frame["overlay_accepted"].astype(bool)].copy()
    ts_all = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    span_days = max((ts_all.max() - ts_all.min()).total_seconds() / 86400.0, 1.0)
    if accepted.empty:
        return {
            **asdict(cfg),
            "accepted_rows": 0,
            "trades_per_day": 0.0,
            "net_pnl": 0.0,
            "mean_net_return_per_trade": 0.0,
            "hit_rate": 0.0,
            "full_sl_rate": 0.0,
            "timeout_rate": 0.0,
            "mean_week_net_pnl": 0.0,
            "std_week_net_pnl": 0.0,
            "worst_week_net_pnl": 0.0,
            "positive_weeks": 0,
            "objective": float("-inf"),
        }
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    pnl = size * net
    accepted["_net_pnl"] = pnl
    accepted["week_start"] = _week_start(accepted["timestamp"])
    weekly = accepted.groupby("week_start", dropna=False)["_net_pnl"].sum()
    mean_week = float(weekly.mean()) if len(weekly) else 0.0
    std_week = float(weekly.std(ddof=0)) if len(weekly) else 0.0
    worst_week = float(weekly.min()) if len(weekly) else 0.0
    objective = mean_week - 0.5 * std_week + 0.25 * worst_week
    reasons = frame["overlay_rejection_reason"].astype(str).value_counts().to_dict()
    exit_reason = accepted["position_exit_reason"].astype(str).str.lower()
    return {
        **asdict(cfg),
        "accepted_rows": int(len(accepted)),
        "trades_per_day": float(len(accepted) / span_days),
        "net_pnl": float(pnl.sum()),
        "gross_pnl": float(
            (
                size
                * pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
            ).sum()
        ),
        "mean_net_return_per_trade": float(net.mean()),
        "hit_rate": float((net > 0.0).mean()),
        "full_sl_rate": float(exit_reason.str.contains("full_sl|stop").mean()),
        "timeout_rate": float(exit_reason.str.contains("timeout").mean()),
        "mean_week_net_pnl": mean_week,
        "std_week_net_pnl": std_week,
        "worst_week_net_pnl": worst_week,
        "positive_weeks": int((weekly > 0.0).sum()),
        "week_count": int(len(weekly)),
        "objective": float(objective),
        "global_loss_blocks": int(reasons.get("global_loss_streak_block", 0)),
        "archetype_loss_blocks": int(reasons.get("archetype_loss_streak_block", 0)),
        "hit_rate_surprise_blocks": int(reasons.get("hit_rate_surprise_block", 0)),
    }


def _breakdown(frame: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    accepted = frame.loc[frame["overlay_accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=keys)
    accepted["week_start"] = _week_start(accepted["timestamp"])
    accepted["month"] = pd.to_datetime(
        accepted["timestamp"], utc=True, errors="coerce"
    ).dt.to_period("M").astype(str)
    accepted["_net_pnl"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    )
    accepted["_gross_pnl"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    )
    accepted["_hit"] = pd.to_numeric(
        accepted["position_net_return"], errors="coerce"
    ).fillna(0.0) > 0.0
    accepted["_full_sl"] = (
        accepted["position_exit_reason"].astype(str).str.lower().str.contains("full_sl|stop")
    )
    accepted["_timeout"] = (
        accepted["position_exit_reason"].astype(str).str.lower().str.contains("timeout")
    )
    grouped = accepted.groupby(keys, dropna=False)
    out = grouped.agg(
        trades=("overlay_accepted", "size"),
        net_pnl=("_net_pnl", "sum"),
        gross_pnl=("_gross_pnl", "sum"),
        avg_net_return=("position_net_return", "mean"),
        hit_rate=("_hit", "mean"),
        full_sl_rate=("_full_sl", "mean"),
        timeout_rate=("_timeout", "mean"),
    ).reset_index()
    return out


def _configs() -> Iterable[GuardConfig]:
    yield GuardConfig(arm="baseline_no_extra_guard")
    for global_trigger in [0, 8, 10, 12]:
        global_cooldowns = [0.0] if global_trigger == 0 else [0.0, 12.0, 24.0, 48.0]
        for global_cd in global_cooldowns:
            for arch_trigger in [0, 3, 4, 5, 6]:
                arch_cooldowns = [0.0] if arch_trigger == 0 else [0.0, 12.0, 24.0, 48.0]
                for arch_cd in arch_cooldowns:
                    arm = (
                        f"kill_g{global_trigger}_cd{global_cd:g}"
                        f"_a{arch_trigger}_cd{arch_cd:g}_hr_off"
                    )
                    yield GuardConfig(
                        arm=arm,
                        global_trigger=global_trigger,
                        global_cooldown_hours=global_cd,
                        archetype_trigger=arch_trigger,
                        archetype_cooldown_hours=arch_cd,
                    )
    for lookback in [3, 5, 8, 10, 15, 20, 30]:
        for min_trades in [3, 5, 8, 10, 15]:
            if min_trades > lookback:
                continue
            for surprise in [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]:
                yield GuardConfig(
                    arm=f"hr_l{lookback}_min{min_trades}_s{surprise:.2f}",
                    hr_enabled=True,
                    hr_lookback_trades=lookback,
                    hr_min_trades=min_trades,
                    hr_negative_surprise_threshold=surprise,
                )
                yield GuardConfig(
                    arm=(
                        f"kill_live_g10_a5_cd24"
                        f"_hr_l{lookback}_min{min_trades}_s{surprise:.2f}"
                    ),
                    global_trigger=10,
                    global_cooldown_hours=24.0,
                    archetype_trigger=5,
                    archetype_cooldown_hours=24.0,
                    hr_enabled=True,
                    hr_lookback_trades=lookback,
                    hr_min_trades=min_trades,
                    hr_negative_surprise_threshold=surprise,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_perp/reports/loss_streak_kill_switch_backtest_20260710"),
    )
    parser.add_argument("--top-n-detail", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decisions = _load_decisions(args.decisions)
    summary_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    baseline_frame: Optional[pd.DataFrame] = None
    top_candidates: List[Tuple[float, GuardConfig, pd.DataFrame]] = []

    for cfg in _configs():
        sim = simulate_overlay(decisions, cfg)
        metrics = _metrics(sim, cfg)
        summary_rows.append(metrics)
        if cfg.arm == "baseline_no_extra_guard":
            baseline_frame = sim
        top_candidates.append((float(metrics.get("objective", -np.inf)), cfg, sim))
        top_candidates = sorted(top_candidates, key=lambda x: x[0], reverse=True)[
            : max(int(args.top_n_detail), 1)
        ]

    summary = pd.DataFrame(summary_rows).sort_values(
        ["objective", "net_pnl"], ascending=[False, False]
    )
    if baseline_frame is not None:
        base_metrics = _metrics(
            baseline_frame, GuardConfig(arm="baseline_no_extra_guard")
        )
        for col in ["net_pnl", "mean_net_return_per_trade", "accepted_rows", "objective"]:
            summary[f"delta_{col}_vs_baseline"] = (
                pd.to_numeric(summary[col], errors="coerce")
                - float(base_metrics.get(col, 0.0))
            )
    summary.to_csv(args.out_dir / "summary_metrics.csv", index=False)

    selected_arms = set(summary.head(int(args.top_n_detail))["arm"].astype(str))
    selected_arms.add("baseline_no_extra_guard")
    for _, cfg, sim in top_candidates:
        selected_arms.add(cfg.arm)
    for cfg in _configs():
        if cfg.arm not in selected_arms:
            continue
        sim = simulate_overlay(decisions, cfg)
        detail_frames.append(_breakdown(sim, ["arm", "week_start"]))
        _breakdown(sim, ["arm", "month"]).to_csv(
            args.out_dir / f"{cfg.arm}_monthly.csv", index=False
        )
        _breakdown(sim, ["arm", "side", "policy_archetype"]).to_csv(
            args.out_dir / f"{cfg.arm}_side_archetype.csv", index=False
        )
        sim.to_parquet(args.out_dir / f"{cfg.arm}_decisions.parquet", index=False)
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            args.out_dir / "selected_weekly_metrics.csv", index=False
        )

    manifest = {
        "schema": "loss_streak_kill_switch_backtest_v1",
        "decisions": str(args.decisions),
        "out_dir": str(args.out_dir),
        "rows": int(len(decisions)),
        "timestamp_min": pd.to_datetime(
            decisions["timestamp"], utc=True, errors="coerce"
        ).min().isoformat(),
        "timestamp_max": pd.to_datetime(
            decisions["timestamp"], utc=True, errors="coerce"
        ).max().isoformat(),
        "overlay_limitations": [
            "uses existing portfolio replay decisions as the candidate stream",
            "blocked trades do not free capacity to admit previously rejected rows",
            "metrics remain useful for kill-switch/HR guard comparison on current policy",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.head(20).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
