#!/usr/bin/env python3
"""Causal weekly head-gate ablation for a frozen candidate parquet.

This script is intentionally narrow: it applies a simple prior-week gate to a
single already-materialized candidate table, then replays the same portfolio
policy.  It is a diagnostic for whether recent per-head degradation can reduce
weekly tail damage without throwing away too much PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)


HEAD_ORDER = ("long_bars", "long_dist", "short_asset", "short_bollinger")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _head_name(strategy_id: Any) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"timestamp", "strategy_id", "symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].copy()
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    if "portfolio_rank_adjustment" not in frame.columns:
        frame["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        frame["portfolio_rank_adjustment"] = (
            pd.to_numeric(frame["portfolio_rank_adjustment"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
    return frame.sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)


def _accepted_table(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["week"] = accepted["timestamp"].dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(accepted.get("position_size", 0.0), errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    reason = accepted.get("position_exit_reason", pd.Series("", index=accepted.index))
    accepted["is_full_sl"] = reason.astype(str).str.contains("full_sl|sl", case=False, na=False)
    accepted["is_timeout"] = reason.astype(str).str.contains("timeout", case=False, na=False)
    return accepted


def _weekly_reference(decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = _accepted_table(decisions)
    if accepted.empty:
        raise ValueError("Reference decisions contain no accepted trades")
    out = (
        accepted.groupby(["week", "head"], as_index=False)
        .agg(
            reference_net_pnl=("net_pnl_amount", "sum"),
            reference_trades=("accepted", "size"),
            reference_hit_rate=("is_win", "mean"),
            reference_full_sl_rate=("is_full_sl", "mean"),
            reference_timeout_rate=("is_timeout", "mean"),
        )
        .sort_values(["week", "head"])
    )
    return out


def _weighted_average(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(np.average(values, weights=weights))


def _should_gate(rule: str, trailing: pd.DataFrame, min_trades: int) -> bool:
    trades = float(pd.to_numeric(trailing["reference_trades"], errors="coerce").fillna(0.0).sum())
    if trades < float(min_trades):
        return False
    net = float(pd.to_numeric(trailing["reference_net_pnl"], errors="coerce").fillna(0.0).sum())
    full_sl = _weighted_average(trailing, "reference_full_sl_rate", "reference_trades")
    hit = _weighted_average(trailing, "reference_hit_rate", "reference_trades")
    if rule == "net_lt_0":
        return net < 0.0
    if rule == "net_lt_500":
        return net < 500.0
    if rule == "net_lt_1000":
        return net < 1000.0
    if rule == "hit_lt_40":
        return hit < 0.40
    if rule == "hit_lt_45":
        return hit < 0.45
    if rule == "fullsl_gt_45":
        return full_sl > 0.45
    if rule == "fullsl_gt_50":
        return full_sl > 0.50
    if rule == "fullsl_gt_55":
        return full_sl > 0.55
    if rule == "net_lt_0_or_fullsl_gt_50":
        return net < 0.0 or full_sl > 0.50
    if rule == "hit_lt_45_or_fullsl_gt_50":
        return hit < 0.45 or full_sl > 0.50
    raise ValueError(f"Unknown gate rule: {rule}")


def _build_gate_schedule(
    candidates: pd.DataFrame,
    weekly_ref: pd.DataFrame,
    *,
    gate_heads: list[str],
    lookback_weeks: int,
    rule: str,
    min_trailing_trades: int,
) -> pd.DataFrame:
    weeks = sorted(pd.to_datetime(candidates["timestamp"], utc=True).dt.to_period("W").astype(str).dropna().unique())
    ref = weekly_ref.set_index(["week", "head"])
    rows: list[dict[str, Any]] = []
    for pos, week in enumerate(weeks):
        prior = weeks[max(0, pos - int(lookback_weeks)) : pos]
        for head in gate_heads:
            if prior:
                idx = pd.MultiIndex.from_product([prior, [head]], names=["week", "head"])
                trailing = ref.reindex(idx).dropna(how="all").reset_index()
            else:
                trailing = pd.DataFrame()
            closed = False if trailing.empty else _should_gate(rule, trailing, min_trailing_trades)
            rows.append(
                {
                    "week": week,
                    "head": head,
                    "lookback_weeks": int(lookback_weeks),
                    "rule": rule,
                    "min_trailing_trades": int(min_trailing_trades),
                    "gate_closed": bool(closed),
                    "trailing_week_count": int(len(trailing)),
                    "trailing_trades": int(pd.to_numeric(trailing.get("reference_trades", 0), errors="coerce").sum())
                    if not trailing.empty
                    else 0,
                    "trailing_net_pnl": float(
                        pd.to_numeric(trailing.get("reference_net_pnl", 0.0), errors="coerce").sum()
                    )
                    if not trailing.empty
                    else 0.0,
                    "trailing_hit_rate": _weighted_average(trailing, "reference_hit_rate", "reference_trades")
                    if not trailing.empty
                    else 0.0,
                    "trailing_full_sl_rate": _weighted_average(
                        trailing, "reference_full_sl_rate", "reference_trades"
                    )
                    if not trailing.empty
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _apply_gate(candidates: pd.DataFrame, schedule: pd.DataFrame, *, priority_multiplier: float) -> pd.DataFrame:
    out = candidates.copy()
    out["week"] = pd.to_datetime(out["timestamp"], utc=True).dt.to_period("W").astype(str)
    out["head"] = out["strategy_id"].astype(str).map(_head_name)
    closed_index = set(
        zip(
            schedule.loc[schedule["gate_closed"].astype(bool), "week"].astype(str),
            schedule.loc[schedule["gate_closed"].astype(bool), "head"].astype(str),
        )
    )
    if "portfolio_priority_multiplier" not in out.columns:
        out["portfolio_priority_multiplier"] = np.float32(1.0)
    current = pd.to_numeric(out["portfolio_priority_multiplier"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    keys = list(zip(out["week"].astype(str), out["head"].astype(str)))
    mask = np.fromiter((key in closed_index for key in keys), dtype=bool, count=len(out))
    current[mask] = current[mask] * max(float(priority_multiplier), 0.0)
    out["portfolio_priority_multiplier"] = current.astype("float32")
    out["gate_closed"] = mask
    return out.drop(columns=["week", "head"], errors="ignore").reset_index(drop=True)


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = _accepted_table(decisions)
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    accepted["day"] = accepted["timestamp"].dt.date.astype(str)
    frames: list[pd.DataFrame] = []
    for cols in (["week"], ["week", "head"], ["day"], ["day", "head"]):
        cur = (
            accepted.groupby(cols, as_index=False)
            .agg(
                net_pnl=("net_pnl_amount", "sum"),
                gross_pnl=("gross_pnl_amount", "sum"),
                trades=("accepted", "size"),
                hit_rate=("is_win", "mean"),
                full_sl_rate=("is_full_sl", "mean"),
                timeout_rate=("is_timeout", "mean"),
            )
            .sort_values(cols)
        )
        cur.insert(0, "period_type", "_".join(cols))
        frames.append(cur)
    return pd.concat(frames[2:], ignore_index=True), pd.concat(frames[:2], ignore_index=True)


def _summary_delta(base: dict[str, Any], cur: dict[str, Any]) -> dict[str, float]:
    keys = ["net_pnl", "gross_pnl", "trade_count", "hit_rate", "full_sl_rate", "timeout_rate", "max_drawdown"]
    return {f"delta_{key}": float(cur.get(key, 0.0) or 0.0) - float(base.get(key, 0.0) or 0.0) for key in keys}


def _parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _safe_label(heads: list[str], lookback: int, rule: str, multiplier: float) -> str:
    head_label = "allheads" if set(heads) == set(HEAD_ORDER) else "_".join(heads)
    return f"{head_label}_{rule}_lb{int(lookback)}_priorityx{int(round(float(multiplier) * 1000)):03d}"


def _run_one(
    *,
    label: str,
    out_dir: Path,
    train: pd.DataFrame,
    eval_candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    priority_multiplier: float,
    market_mode: str,
) -> dict[str, Any]:
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    gated = _apply_gate(eval_candidates, schedule, priority_multiplier=priority_multiplier)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(train)
    decisions, equity, metrics = replay_candidates(
        gated,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    decisions.to_parquet(run_dir / "decisions.parquet", index=False)
    equity.to_parquet(run_dir / "equity.parquet", index=False)
    daily, weekly = _period_tables(decisions)
    daily.to_csv(run_dir / "daily.csv", index=False)
    weekly.to_csv(run_dir / "weekly.csv", index=False)
    schedule.to_csv(run_dir / "gate_schedule.csv", index=False)
    return {
        "variant": label,
        "priority_multiplier": float(priority_multiplier),
        "gate_closed_head_weeks": int(schedule["gate_closed"].sum()),
        "gate_total_head_weeks": int(len(schedule)),
        "gated_candidate_rows": int(gated["gate_closed"].sum()),
        **dict(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-start", default="2026-02-01T00:00:00+00:00")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--gate-heads", default="long_bars,long_dist,short_asset,short_bollinger")
    parser.add_argument("--lookbacks", default="1,2")
    parser.add_argument("--rules", default="net_lt_0,hit_lt_45,fullsl_gt_50")
    parser.add_argument("--priority-multipliers", default="0.25,0.50")
    parser.add_argument("--min-trailing-trades", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC") if args.eval_end else None
    baseline_all = _load_candidates(args.baseline)
    candidate_all = _load_candidates(args.candidate)
    if len(baseline_all) != len(candidate_all):
        raise ValueError(f"candidate row count mismatch: {len(candidate_all)} != {len(baseline_all)}")
    key_cols = ["timestamp", "strategy_id", "symbol"]
    if not baseline_all[key_cols].reset_index(drop=True).equals(candidate_all[key_cols].reset_index(drop=True)):
        raise ValueError("candidate key ordering does not match baseline")

    train = baseline_all[baseline_all["timestamp"].lt(eval_start)].copy().reset_index(drop=True)
    eval_mask = baseline_all["timestamp"].ge(eval_start)
    if eval_end is not None:
        eval_mask &= baseline_all["timestamp"].le(eval_end)
    eval_candidates = candidate_all.loc[eval_mask].copy().reset_index(drop=True)
    if train.empty or eval_candidates.empty:
        raise ValueError("empty train or evaluation candidate slice")

    reference_decisions = pd.read_parquet(args.reference_decisions)
    weekly_ref = _weekly_reference(reference_decisions)
    weekly_ref.to_csv(args.output_dir / "reference_weekly_by_head.csv", index=False)

    gate_heads = _parse_csv_strings(args.gate_heads)
    unknown_heads = sorted(set(gate_heads) - set(HEAD_ORDER))
    if unknown_heads:
        raise ValueError(f"unknown gate heads: {unknown_heads}")
    lookbacks = _parse_csv_ints(args.lookbacks)
    rules = _parse_csv_strings(args.rules)
    multipliers = _parse_csv_floats(args.priority_multipliers)

    rows: list[dict[str, Any]] = []
    for lookback in lookbacks:
        for rule in rules:
            schedule = _build_gate_schedule(
                eval_candidates,
                weekly_ref,
                gate_heads=gate_heads,
                lookback_weeks=int(lookback),
                rule=rule,
                min_trailing_trades=int(args.min_trailing_trades),
            )
            for multiplier in multipliers:
                label = _safe_label(gate_heads, int(lookback), rule, float(multiplier))
                print(f"RUN {label}", flush=True)
                rows.append(
                    _run_one(
                        label=label,
                        out_dir=args.output_dir,
                        train=train,
                        eval_candidates=eval_candidates,
                        schedule=schedule,
                        priority_multiplier=float(multiplier),
                        market_mode=str(args.market_mode),
                    )
                )

    summary = pd.DataFrame(rows).sort_values(["net_pnl", "max_drawdown"], ascending=[False, False])
    summary.to_csv(args.output_dir / "weekly_head_gate_summary.csv", index=False)
    manifest = {
        "generated_by": "ablate_frozen_candidate_weekly_head_gate",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "reference_decisions": str(args.reference_decisions),
        "baseline_sha256": _sha256(args.baseline),
        "candidate_sha256": _sha256(args.candidate),
        "reference_decisions_sha256": _sha256(args.reference_decisions),
        "eval_start": eval_start.isoformat(),
        "eval_end": eval_end.isoformat() if eval_end is not None else "",
        "train_rows_for_ev": int(len(train)),
        "eval_rows": int(len(eval_candidates)),
        "gate_heads": gate_heads,
        "lookbacks": lookbacks,
        "rules": rules,
        "priority_multipliers": multipliers,
        "min_trailing_trades": int(args.min_trailing_trades),
        "market_mode": str(args.market_mode),
        "policy_params": asdict(PortfolioPolicyParams(global_threshold_floor=0.0)),
        "run_count": int(len(summary)),
    }
    (args.output_dir / "weekly_head_gate_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen Candidate Weekly Head Gate",
        "",
        "Current-week outcomes are not used for current-week gating; decisions use prior weekly reference replay metrics.",
        "",
        summary.to_markdown(index=False),
    ]
    (args.output_dir / "weekly_head_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "run_count": int(len(summary))}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
