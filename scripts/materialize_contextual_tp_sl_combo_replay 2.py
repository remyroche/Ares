#!/usr/bin/env python3
"""Materialize a named contextual TP/SL head-arm combo replay.

The exhaustive sweep is intentionally summary-oriented.  This script turns a
specific combo ID into concrete artifacts: combined candidate rows, replay
decisions, equity curve, global/per-head period metrics, and a manifest with
input hashes.  It is meant for frozen champion comparisons and cheap forward
replays without rerunning every combination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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


def _parse_active_heads(values: Sequence[str]) -> set[str]:
    if not values:
        return set(HEAD_ORDER)
    heads: set[str] = set()
    for raw in values:
        for part in str(raw).split(","):
            head = part.strip()
            if not head:
                continue
            if head not in HEAD_ORDER:
                raise ValueError(f"Unknown active head {head!r}; expected one of {HEAD_ORDER}")
            heads.add(head)
    if not heads:
        raise ValueError("At least one active head is required")
    return heads


def _parse_combo(combo_id: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in str(combo_id).split("_"):
        if ":" not in part:
            continue
        head, label = part.split(":", 1)
        if head not in HEAD_ORDER:
            # Heads contain underscores, so reconstruct from adjacent pieces below.
            continue
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


def _load_arm_tables(source_dir: Path, arms: Sequence[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for arm in sorted(set(arms)):
        path = source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate table: {path}")
        frame = pd.read_parquet(path)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        frame["head"] = frame["strategy_id"].map(_head_name)
        tables[arm] = frame
    return tables


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame(), pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    if "position_exit_reason" in accepted.columns:
        exit_reason = accepted["position_exit_reason"]
    elif "exit_reason" in accepted.columns:
        exit_reason = accepted["exit_reason"]
    else:
        exit_reason = pd.Series("", index=accepted.index)
    accepted["is_full_sl"] = exit_reason.astype(str).str.contains(
        "sl", case=False, na=False
    )
    accepted["is_timeout"] = exit_reason.astype(str).str.contains(
        "timeout", case=False, na=False
    )

    group_cols = [["week"], ["week", "head"], ["day"], ["day", "head"]]
    frames = []
    for cols in group_cols:
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
    weekly = pd.concat(frames[:2], ignore_index=True)
    daily = pd.concat(frames[2:], ignore_index=True)
    return daily, weekly


def _write_report(out_dir: Path, manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Contextual TP/SL Materialized Combo Replay",
        "",
        f"Combo: `{manifest['combo_id']}`",
        f"Source: `{manifest['source_dir']}`",
        f"Period: `{manifest['candidate_start']}` to `{manifest['candidate_end']}`",
        "Costs included via `portfolio_policy_replay.replay_candidates`.",
        "",
        "## Head Arms",
        "",
        pd.DataFrame(
            [{"head": head, "arm": arm} for head, arm in manifest["head_arm_mapping"].items()]
        ).to_markdown(index=False),
        "",
        "## Global Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Artifacts",
        "",
        "- `combo_candidates.parquet`",
        "- `combo_replay_decisions.parquet`",
        "- `combo_replay_equity.parquet`",
        "- `combo_replay_daily_metrics.csv`",
        "- `combo_replay_weekly_metrics.csv`",
        "- `combo_replay_manifest.json`",
    ]
    (out_dir / "combo_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--combo-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--start", default="", help="Optional inclusive UTC timestamp filter")
    parser.add_argument("--end", default="", help="Optional inclusive UTC timestamp filter")
    parser.add_argument(
        "--active-head",
        action="append",
        default=[],
        help="Restrict replay to one or more heads. Accepts repeated values or comma-separated heads.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = _parse_combo(args.combo_id)
    active_heads = _parse_active_heads(args.active_head)
    tables = _load_arm_tables(args.source_dir, tuple(mapping.values()))
    frames = []
    input_paths = {}
    for head, arm in mapping.items():
        if head not in active_heads:
            continue
        source = tables[arm]
        frames.append(source.loc[source["head"].eq(head)].copy())
        input_paths[arm] = args.source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
    if not frames:
        raise ValueError(f"No frames selected for active heads: {sorted(active_heads)}")
    candidates = (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["head"], errors="ignore")
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )
    ts_all = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    if args.start:
        start_ts = pd.Timestamp(args.start, tz="UTC")
        candidates = candidates.loc[ts_all.ge(start_ts)].copy()
        ts_all = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    if args.end:
        end_ts = pd.Timestamp(args.end, tz="UTC")
        candidates = candidates.loc[ts_all.le(end_ts)].copy()
        ts_all = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    if candidates.empty:
        raise ValueError("No candidate rows after applying combo and timestamp filters")
    candidates = candidates.reset_index(drop=True)
    candidates.to_parquet(args.out_dir / "combo_candidates.parquet", index=False)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(candidates)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )
    decisions.to_parquet(args.out_dir / "combo_replay_decisions.parquet", index=False)
    equity.to_parquet(args.out_dir / "combo_replay_equity.parquet", index=False)
    daily, weekly = _period_tables(decisions)
    daily.to_csv(args.out_dir / "combo_replay_daily_metrics.csv", index=False)
    weekly.to_csv(args.out_dir / "combo_replay_weekly_metrics.csv", index=False)

    ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    manifest = {
        "generated_by": "materialize_contextual_tp_sl_combo_replay",
        "source_dir": str(args.source_dir),
        "combo_id": str(args.combo_id),
        "head_arm_mapping": mapping,
        "active_heads": sorted(active_heads),
        "market_mode": str(args.market_mode),
        "start_filter": str(args.start),
        "end_filter": str(args.end),
        "candidate_rows": int(len(candidates)),
        "candidate_start": ts.min(),
        "candidate_end": ts.max(),
        "policy_params": asdict(params),
        "input_hashes": {arm: _sha256(path) for arm, path in sorted(input_paths.items())},
        "metrics": dict(metrics),
    }
    (args.out_dir / "combo_replay_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    _write_report(args.out_dir, _json_safe(manifest), _json_safe(dict(metrics)))
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "combo_id": str(args.combo_id),
                    "candidate_rows": int(len(candidates)),
                    "net_pnl": metrics.get("net_pnl"),
                    "trade_count": metrics.get("trade_count"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
