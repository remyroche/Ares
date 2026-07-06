#!/usr/bin/env python3
"""Replay active-head subsets from materialized candidate tables.

This is a small adapter around ``portfolio_policy_replay.replay_candidates``. It
lets us test real auction/capacity effects for predeclared head subsets without
regenerating model predictions or TP/SL candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _head_name(strategy_id: str) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    if text.startswith("short_asset"):
        return "short_asset"
    if text.startswith("long_bars"):
        return "long_bars"
    if text.startswith("long_dist"):
        return "long_dist"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _parse_heads(values: Iterable[str]) -> set[str]:
    heads: set[str] = set()
    for raw in values:
        for part in str(raw).split(","):
            head = part.strip()
            if not head:
                continue
            if head not in HEAD_ORDER:
                raise ValueError(f"Unknown head {head!r}; expected one of {HEAD_ORDER}")
            heads.add(head)
    return heads or set(HEAD_ORDER)


def _triggered_noop_weeks(path: Path | None) -> set[str]:
    if path is None:
        return set()
    selections = pd.read_csv(path)
    if not {"eval_week", "triggered", "action_label"}.issubset(selections.columns):
        raise ValueError(f"Selection file is missing required columns: {path}")
    mask = selections["triggered"].astype(bool) & selections["action_label"].astype(str).eq("__noop__")
    return set(selections.loc[mask, "eval_week"].astype(str))


def _add_week(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["__week"] = ts.dt.to_period("W-SUN").astype(str)
    return out


def _load_candidate_mix(default_candidates: Path, noop_candidates: Path | None, selections: Path | None) -> pd.DataFrame:
    default = _add_week(pd.read_parquet(default_candidates))
    noop_weeks = _triggered_noop_weeks(selections)
    if noop_candidates is None or not noop_weeks:
        return default.drop(columns=["__week"], errors="ignore")
    noop = _add_week(pd.read_parquet(noop_candidates))
    mixed = pd.concat(
        [
            default.loc[~default["__week"].isin(noop_weeks)],
            noop.loc[noop["__week"].isin(noop_weeks)],
        ],
        ignore_index=True,
    )
    return mixed.drop(columns=["__week"], errors="ignore")


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
    accepted["net_pnl"] = size * net
    accepted["gross_pnl"] = size * gross
    accepted["is_win"] = net > 0.0
    reason = accepted.get("position_exit_reason", pd.Series("", index=accepted.index)).astype(str)
    accepted["is_full_sl"] = reason.str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = reason.str.contains("timeout", case=False, na=False)
    frames: list[pd.DataFrame] = []
    for cols in (["week"], ["week", "head"], ["day"], ["day", "head"]):
        cur = (
            accepted.groupby(cols, as_index=False)
            .agg(
                net_pnl=("net_pnl", "sum"),
                gross_pnl=("gross_pnl", "sum"),
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


def _write_report(out_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# Head-Subset Candidate-Mix Replay",
        "",
        f"Label: `{manifest['label']}`",
        f"Active heads: `{', '.join(manifest['active_heads'])}`",
        f"Candidate rows: `{manifest['candidate_rows']}`",
        f"Candidate period: `{manifest['candidate_start']}` to `{manifest['candidate_end']}`",
        "Costs included via `portfolio_policy_replay.replay_candidates`.",
        "",
        "## Metrics",
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
    (out_dir / "combo_replay_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-candidates", type=Path, required=True)
    parser.add_argument("--noop-candidates", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--active-head", action="append", default=[])
    parser.add_argument("--label", default="")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    active_heads = _parse_heads(args.active_head)
    label = args.label or "_".join(sorted(active_heads))
    candidates = _load_candidate_mix(args.default_candidates, args.noop_candidates, args.selections)
    ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    if args.start:
        candidates = candidates.loc[ts.ge(pd.Timestamp(args.start, tz="UTC"))].copy()
        ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    if args.end:
        candidates = candidates.loc[ts.le(pd.Timestamp(args.end, tz="UTC"))].copy()
        ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates["__head"] = candidates["strategy_id"].astype(str).map(_head_name)
    candidates = candidates.loc[candidates["__head"].isin(active_heads)].drop(columns=["__head"]).copy()
    if candidates.empty:
        raise ValueError("No candidate rows remain after timestamp/head filtering")
    candidates = candidates.sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates.to_parquet(args.out_dir / "combo_candidates.parquet", index=False)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(candidates)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(args.market_mode),
    )
    decisions.to_parquet(args.out_dir / "combo_replay_decisions.parquet", index=False)
    equity.to_parquet(args.out_dir / "combo_replay_equity.parquet", index=False)
    daily, weekly = _period_tables(decisions)
    daily.to_csv(args.out_dir / "combo_replay_daily_metrics.csv", index=False)
    weekly.to_csv(args.out_dir / "combo_replay_weekly_metrics.csv", index=False)
    ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    manifest = {
        "generated_by": "replay_head_subset_candidate_mix",
        "label": label,
        "active_heads": sorted(active_heads),
        "default_candidates": str(args.default_candidates),
        "noop_candidates": str(args.noop_candidates) if args.noop_candidates else None,
        "selections": str(args.selections) if args.selections else None,
        "triggered_noop_weeks": sorted(_triggered_noop_weeks(args.selections)),
        "market_mode": str(args.market_mode),
        "start_filter": str(args.start),
        "end_filter": str(args.end),
        "candidate_rows": int(len(candidates)),
        "candidate_start": ts.min(),
        "candidate_end": ts.max(),
        "policy_params": asdict(params),
        "input_hashes": {
            "default_candidates": _sha256(args.default_candidates),
            **({"noop_candidates": _sha256(args.noop_candidates)} if args.noop_candidates else {}),
        },
        "metrics": dict(metrics),
    }
    (args.out_dir / "combo_replay_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    _write_report(args.out_dir, _json_safe(manifest), _json_safe(dict(metrics)))
    print(json.dumps(_json_safe({"label": label, "out_dir": str(args.out_dir), **dict(metrics)}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
