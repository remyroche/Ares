#!/usr/bin/env python3
"""Run causal weekly head-gating ablations for contextual TP/SL combos.

The gate suppresses one strategy head for a week when its reference replay
performance over prior weeks is poor.  It is deliberately simple and causal at
week granularity: current-week outcomes are never used to decide current-week
participation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
ARM_LABELS = {
    "S": "static",
    "R": "rank_only",
    "P": "performance_only",
    "J": "joint_all",
    "I": "independent_all",
}


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


def _head_name(strategy_id: str) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _parse_combo(combo_id: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    chunks = str(combo_id).split("_")
    i = 0
    while i < len(chunks):
        if i + 1 < len(chunks) and ":" not in chunks[i] and ":" in chunks[i + 1]:
            head_prefix = f"{chunks[i]}_{chunks[i + 1].split(':', 1)[0]}"
            label = chunks[i + 1].split(":", 1)[1]
            if head_prefix in HEAD_ORDER and label in ARM_LABELS:
                mapping[head_prefix] = ARM_LABELS[label]
                i += 2
                continue
        if ":" in chunks[i]:
            head, label = chunks[i].split(":", 1)
            if head in HEAD_ORDER and label in ARM_LABELS:
                mapping[head] = ARM_LABELS[label]
        i += 1
    missing = [head for head in HEAD_ORDER if head not in mapping]
    if missing:
        raise ValueError(f"Combo {combo_id!r} is missing heads: {missing}")
    return mapping


def _load_arm_tables(source_dir: Path, arms: Sequence[str]) -> tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    tables: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, Path] = {}
    for arm in sorted(set(arms)):
        path = source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate table: {path}")
        frame = pd.read_parquet(path)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        frame["head"] = frame["strategy_id"].map(_head_name)
        tables[arm] = frame
        paths[arm] = path
    return tables, paths


def _combo_candidates(source_dir: Path, combo_id: str) -> tuple[pd.DataFrame, Dict[str, str], Dict[str, Path]]:
    mapping = _parse_combo(combo_id)
    tables, paths = _load_arm_tables(source_dir, tuple(mapping.values()))
    frames = []
    for head, arm in mapping.items():
        frames.append(tables[arm].loc[tables[arm]["head"].eq(head)].copy())
    candidates = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )
    return candidates, mapping, paths


def _accepted_reference_weekly(reference_replay_dir: Path, gate_head: str) -> pd.DataFrame:
    path = reference_replay_dir / "combo_replay_decisions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference decisions: {path}")
    decisions = pd.read_parquet(path)
    if decisions.empty or "accepted" not in decisions.columns:
        raise ValueError(f"Reference replay has no accepted decisions: {path}")
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    accepted["head"] = accepted["strategy_id"].astype(str).map(_head_name)
    accepted = accepted.loc[accepted["head"].eq(gate_head)].copy()
    if accepted.empty:
        raise ValueError(f"Reference replay has no accepted decisions for head {gate_head!r}")
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["week"] = ts.dt.to_period("W").astype(str)
    size = pd.to_numeric(accepted.get("position_size", 0.0), errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["is_win"] = net > 0.0
    exit_reason = accepted.get("position_exit_reason", pd.Series("", index=accepted.index))
    accepted["is_full_sl"] = exit_reason.astype(str).str.contains("full_sl|sl", case=False, na=False)
    out = (
        accepted.groupby("week", as_index=False)
        .agg(
            reference_net_pnl=("net_pnl_amount", "sum"),
            reference_trades=("accepted", "size"),
            reference_hit_rate=("is_win", "mean"),
            reference_full_sl_rate=("is_full_sl", "mean"),
        )
        .sort_values("week")
    )
    return out


def _candidate_weeks(candidates: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    return ts.dt.to_period("W").astype(str)


def _rule_decision(rule: str, trailing: pd.DataFrame, min_trades: int) -> bool:
    """Return True when the gate should be closed."""
    trades = float(pd.to_numeric(trailing["reference_trades"], errors="coerce").fillna(0.0).sum())
    if trades < float(min_trades):
        return False
    net = float(pd.to_numeric(trailing["reference_net_pnl"], errors="coerce").fillna(0.0).sum())
    full_sl = float(
        np.average(
            pd.to_numeric(trailing["reference_full_sl_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            weights=pd.to_numeric(trailing["reference_trades"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        )
    )
    hit = float(
        np.average(
            pd.to_numeric(trailing["reference_hit_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            weights=pd.to_numeric(trailing["reference_trades"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        )
    )
    if rule == "net_lt_0":
        return net < 0.0
    if rule == "net_lt_1000":
        return net < 1000.0
    if rule == "net_lt_minus_1000":
        return net < -1000.0
    if rule == "fullsl_gt_55":
        return full_sl > 0.55
    if rule == "fullsl_gt_60":
        return full_sl > 0.60
    if rule == "hit_lt_45":
        return hit < 0.45
    if rule == "net_lt_0_or_fullsl_gt_55":
        return net < 0.0 or full_sl > 0.55
    if rule == "net_lt_0_and_fullsl_gt_55":
        return net < 0.0 and full_sl > 0.55
    raise ValueError(f"Unknown rule: {rule}")


def _gate_table(
    candidates: pd.DataFrame,
    reference_weekly: pd.DataFrame,
    *,
    gate_head: str,
    lookback_weeks: int,
    rule: str,
    min_trailing_trades: int,
) -> pd.DataFrame:
    weeks = pd.DataFrame({"week": sorted(_candidate_weeks(candidates).dropna().unique())})
    ref = reference_weekly.set_index("week")
    records: List[Dict[str, Any]] = []
    for pos, week in enumerate(weeks["week"].tolist()):
        prior = weeks["week"].iloc[max(0, pos - int(lookback_weeks)) : pos]
        trailing = ref.reindex(prior).dropna(how="all").reset_index()
        close_gate = False if trailing.empty else _rule_decision(rule, trailing, int(min_trailing_trades))
        records.append(
            {
                "week": week,
                "gate_head": gate_head,
                "lookback_weeks": int(lookback_weeks),
                "rule": rule,
                "min_trailing_trades": int(min_trailing_trades),
                "gate_closed": bool(close_gate),
                "trailing_week_count": int(len(trailing)),
                "trailing_trades": int(pd.to_numeric(trailing.get("reference_trades", 0), errors="coerce").sum())
                if not trailing.empty
                else 0,
                "trailing_net_pnl": float(
                    pd.to_numeric(trailing.get("reference_net_pnl", 0.0), errors="coerce").sum()
                )
                if not trailing.empty
                else 0.0,
            }
        )
    return pd.DataFrame(records)


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame(), pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    exit_reason = accepted.get("position_exit_reason", pd.Series("", index=accepted.index))
    accepted["is_full_sl"] = exit_reason.astype(str).str.contains("full_sl|sl", case=False, na=False)
    accepted["is_timeout"] = exit_reason.astype(str).str.contains("timeout", case=False, na=False)

    frames = []
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


def _filter_candidates(
    candidates: pd.DataFrame,
    gate: pd.DataFrame,
    gate_head: str,
    *,
    closed_rank_threshold: float | None,
    closed_size_multiplier: float | None,
    closed_priority_multiplier: float | None,
    rank_column: str,
) -> pd.DataFrame:
    out = candidates.copy()
    out["week"] = _candidate_weeks(out)
    out["head"] = out["strategy_id"].astype(str).map(_head_name)
    gate_map = gate.set_index("week")["gate_closed"].astype(bool)
    closed = out["week"].map(gate_map).fillna(False).astype(bool)
    gated_head = out["head"].eq(gate_head) & closed
    if closed_priority_multiplier is not None:
        if "portfolio_priority_multiplier" not in out.columns:
            out["portfolio_priority_multiplier"] = 1.0
        current = pd.to_numeric(out["portfolio_priority_multiplier"], errors="coerce").fillna(1.0)
        out.loc[gated_head, "portfolio_priority_multiplier"] = (
            current.loc[gated_head] * max(float(closed_priority_multiplier), 0.0)
        )
        keep = pd.Series(True, index=out.index)
    elif closed_size_multiplier is not None:
        if "portfolio_size_multiplier" not in out.columns:
            out["portfolio_size_multiplier"] = 1.0
        current = pd.to_numeric(out["portfolio_size_multiplier"], errors="coerce").fillna(1.0)
        out.loc[gated_head, "portfolio_size_multiplier"] = (
            current.loc[gated_head] * max(float(closed_size_multiplier), 0.0)
        )
        keep = pd.Series(True, index=out.index)
    elif closed_rank_threshold is None:
        keep = ~gated_head
    else:
        if rank_column not in out.columns:
            raise KeyError(f"Rank column {rank_column!r} not present in candidates")
        ranks = pd.to_numeric(out[rank_column], errors="coerce").fillna(-np.inf)
        keep = (~gated_head) | ranks.ge(float(closed_rank_threshold))
    filtered = out.loc[keep].copy()
    return filtered.drop(columns=["week", "head"], errors="ignore").reset_index(drop=True)


def _safe_label(
    rule: str,
    lookback_weeks: int,
    closed_rank_threshold: float | None,
    closed_size_multiplier: float | None,
    closed_priority_multiplier: float | None,
) -> str:
    if closed_priority_multiplier is not None:
        action = f"priorityx{int(round(float(closed_priority_multiplier) * 1000)):03d}"
    elif closed_size_multiplier is not None:
        action = f"sizex{int(round(float(closed_size_multiplier) * 1000)):03d}"
    elif closed_rank_threshold is None:
        action = "drop"
    else:
        action = f"rankgte{int(round(float(closed_rank_threshold) * 1000)):03d}"
    return f"{rule}_lb{int(lookback_weeks)}_{action}"


def _run_one(
    *,
    candidates: pd.DataFrame,
    mapping: Mapping[str, str],
    input_paths: Mapping[str, Path],
    source_dir: Path,
    combo_id: str,
    out_dir: Path,
    gate: pd.DataFrame,
    gate_head: str,
    market_mode: str,
    closed_rank_threshold: float | None,
    closed_size_multiplier: float | None,
    closed_priority_multiplier: float | None,
    rank_column: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = _filter_candidates(
        candidates,
        gate,
        gate_head,
        closed_rank_threshold=closed_rank_threshold,
        closed_size_multiplier=closed_size_multiplier,
        closed_priority_multiplier=closed_priority_multiplier,
        rank_column=rank_column,
    )
    if filtered.empty:
        raise ValueError(f"Gate removed all candidate rows for {out_dir}")
    filtered.to_parquet(out_dir / "combo_candidates.parquet", index=False)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(filtered)
    decisions, equity, metrics = replay_candidates(
        filtered,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    decisions.to_parquet(out_dir / "combo_replay_decisions.parquet", index=False)
    equity.to_parquet(out_dir / "combo_replay_equity.parquet", index=False)
    daily, weekly = _period_tables(decisions)
    daily.to_csv(out_dir / "combo_replay_daily_metrics.csv", index=False)
    weekly.to_csv(out_dir / "combo_replay_weekly_metrics.csv", index=False)
    gate.to_csv(out_dir / "head_gate_weeks.csv", index=False)
    ts = pd.to_datetime(filtered["timestamp"], utc=True, errors="coerce")
    manifest = {
        "generated_by": "ablate_contextual_tp_sl_dynamic_head_gate",
        "source_dir": str(source_dir),
        "combo_id": combo_id,
        "head_arm_mapping": dict(mapping),
        "gate_head": gate_head,
        "gate_rule": str(gate["rule"].iloc[0]) if not gate.empty else "",
        "lookback_weeks": int(gate["lookback_weeks"].iloc[0]) if not gate.empty else 0,
        "gate_action": (
            "priority_multiplier"
            if closed_priority_multiplier is not None
            else "size_multiplier"
            if closed_size_multiplier is not None
            else "drop"
            if closed_rank_threshold is None
            else "rank_threshold"
        ),
        "closed_rank_threshold": closed_rank_threshold,
        "closed_size_multiplier": closed_size_multiplier,
        "closed_priority_multiplier": closed_priority_multiplier,
        "rank_column": rank_column,
        "gate_closed_weeks": int(gate["gate_closed"].sum()) if not gate.empty else 0,
        "gate_total_weeks": int(len(gate)),
        "market_mode": market_mode,
        "candidate_rows": int(len(filtered)),
        "candidate_start": ts.min(),
        "candidate_end": ts.max(),
        "policy_params": asdict(params),
        "input_hashes": {arm: _sha256(path) for arm, path in sorted(input_paths.items())},
        "metrics": dict(metrics),
    }
    (out_dir / "combo_replay_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    return {
        "label": out_dir.name,
        "out_dir": str(out_dir),
        "rule": manifest["gate_rule"],
        "lookback_weeks": manifest["lookback_weeks"],
        "gate_action": manifest["gate_action"],
        "closed_rank_threshold": manifest["closed_rank_threshold"],
        "closed_size_multiplier": manifest["closed_size_multiplier"],
        "closed_priority_multiplier": manifest["closed_priority_multiplier"],
        "rank_column": manifest["rank_column"],
        "gate_closed_weeks": manifest["gate_closed_weeks"],
        "gate_total_weeks": manifest["gate_total_weeks"],
        "candidate_rows": manifest["candidate_rows"],
        "net_pnl": metrics.get("net_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "trade_count": metrics.get("trade_count"),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
    }


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_thresholds(value: str) -> List[float | None]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return [None]
    out: List[float | None] = []
    for part in parts:
        if part.lower() in {"drop", "none"}:
            out.append(None)
        else:
            out.append(float(part))
    return out


def _parse_size_multipliers(value: str) -> List[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-replay-dir", type=Path, required=True)
    parser.add_argument("--combo-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate-head", default="long_bars", choices=HEAD_ORDER)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--lookbacks", default="1,2,4")
    parser.add_argument(
        "--rules",
        default="net_lt_0,net_lt_1000,fullsl_gt_55,hit_lt_45,net_lt_0_or_fullsl_gt_55",
    )
    parser.add_argument("--min-trailing-trades", type=int, default=10)
    parser.add_argument(
        "--closed-rank-thresholds",
        default="drop",
        help="Comma-separated actions for closed gate: `drop` or rank thresholds such as 0.80,0.85.",
    )
    parser.add_argument(
        "--closed-size-multipliers",
        default="",
        help=(
            "Comma-separated size multipliers for closed gate, e.g. 0.25,0.50. "
            "When provided, these are tested instead of rank/drop actions."
        ),
    )
    parser.add_argument(
        "--closed-priority-multipliers",
        default="",
        help=(
            "Comma-separated priority multipliers for closed gate, e.g. 0.10,0.25. "
            "When provided, these are tested before size/rank/drop actions."
        ),
    )
    parser.add_argument("--rank-column", default="rank_pct")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates, mapping, input_paths = _combo_candidates(args.source_dir, str(args.combo_id))
    reference_weekly = _accepted_reference_weekly(args.reference_replay_dir, str(args.gate_head))
    reference_weekly.to_csv(args.out_dir / "reference_gate_head_weekly_metrics.csv", index=False)

    rows: List[Dict[str, Any]] = []
    size_multipliers = _parse_size_multipliers(str(args.closed_size_multipliers))
    priority_multipliers = _parse_size_multipliers(str(args.closed_priority_multipliers))
    action_pairs = (
        [(None, None, multiplier) for multiplier in priority_multipliers]
        if priority_multipliers
        else [(None, multiplier, None) for multiplier in size_multipliers]
        if size_multipliers
        else [(threshold, None, None) for threshold in _parse_thresholds(str(args.closed_rank_thresholds))]
    )
    for lookback in _parse_csv_ints(str(args.lookbacks)):
        for rule in _parse_csv_strings(str(args.rules)):
            for threshold, size_multiplier, priority_multiplier in action_pairs:
                gate = _gate_table(
                    candidates,
                    reference_weekly,
                    gate_head=str(args.gate_head),
                    lookback_weeks=int(lookback),
                    rule=str(rule),
                    min_trailing_trades=int(args.min_trailing_trades),
                )
                run_dir = args.out_dir / "materialized" / _safe_label(
                    rule,
                    int(lookback),
                    threshold,
                    size_multiplier,
                    priority_multiplier,
                )
                print(f"RUN {run_dir.name}", flush=True)
                rows.append(
                    _run_one(
                        candidates=candidates,
                        mapping=mapping,
                        input_paths=input_paths,
                        source_dir=args.source_dir,
                        combo_id=str(args.combo_id),
                        out_dir=run_dir,
                        gate=gate,
                        gate_head=str(args.gate_head),
                        market_mode=str(args.market_mode),
                        closed_rank_threshold=threshold,
                        closed_size_multiplier=size_multiplier,
                        closed_priority_multiplier=priority_multiplier,
                        rank_column=str(args.rank_column),
                    )
                )
    summary = pd.DataFrame(rows).sort_values(["net_pnl", "max_drawdown"], ascending=[False, False])
    summary.to_csv(args.out_dir / "dynamic_head_gate_summary.csv", index=False)
    payload = {
        "generated_by": "ablate_contextual_tp_sl_dynamic_head_gate",
        "source_dir": str(args.source_dir),
        "reference_replay_dir": str(args.reference_replay_dir),
        "combo_id": str(args.combo_id),
        "out_dir": str(args.out_dir),
        "gate_head": str(args.gate_head),
        "lookbacks": _parse_csv_ints(str(args.lookbacks)),
        "rules": _parse_csv_strings(str(args.rules)),
        "min_trailing_trades": int(args.min_trailing_trades),
        "closed_rank_thresholds": _parse_thresholds(str(args.closed_rank_thresholds)),
        "closed_size_multipliers": size_multipliers,
        "closed_priority_multipliers": priority_multipliers,
        "rank_column": str(args.rank_column),
        "run_count": int(len(summary)),
    }
    (args.out_dir / "dynamic_head_gate_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Dynamic Head Gate Ablation",
        "",
        f"Source: `{args.source_dir}`",
        f"Reference replay: `{args.reference_replay_dir}`",
        f"Combo: `{args.combo_id}`",
        f"Gate head: `{args.gate_head}`",
        "",
        "Current-week outcomes are not used for current-week gating; each decision uses prior weekly reference replay metrics.",
        "",
        summary.to_markdown(index=False),
    ]
    (args.out_dir / "dynamic_head_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "run_count": int(len(summary))}), indent=2))


if __name__ == "__main__":
    main()
