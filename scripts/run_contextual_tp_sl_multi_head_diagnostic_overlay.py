#!/usr/bin/env python3
"""Replay contextual TP/SL candidates with multiple head-specific overlays."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

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
from scripts.ablate_contextual_tp_sl_diagnostic_head_gate import (  # noqa: E402
    _combo_candidates,
    _diagnostic_scores,
    _head_name,
    _json_safe,
    _period_tables,
    _sha256,
)


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_rule(value: str) -> Dict[str, Any]:
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) != 4:
        raise ValueError(
            "Rules must be head:diagnostic_family:threshold:size_multiplier, "
            f"got {value!r}"
        )
    head, family, threshold, multiplier = parts
    risk_column = family if family.startswith("diagnostic_") else f"diagnostic_{family}_risk"
    return {
        "head": head,
        "diagnostic_family": family,
        "risk_column": risk_column,
        "threshold": float(threshold),
        "size_multiplier": float(multiplier),
    }


def _timestamp_mask(frame: pd.DataFrame, *, start: str, end: str) -> pd.Series:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= ts.ge(pd.Timestamp(start, tz="UTC"))
    if end:
        mask &= ts.le(pd.Timestamp(end, tz="UTC"))
    return mask


def _apply_rules(candidates: pd.DataFrame, rules: List[Dict[str, Any]]) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    out = candidates.copy()
    out["head"] = out["strategy_id"].astype(str).map(_head_name)
    if "portfolio_size_multiplier" not in out.columns:
        out["portfolio_size_multiplier"] = 1.0
    size = pd.to_numeric(out["portfolio_size_multiplier"], errors="coerce").fillna(1.0).astype("float64")
    records: List[Dict[str, Any]] = []
    for rule in rules:
        risk_column = str(rule["risk_column"])
        if risk_column not in out.columns:
            raise KeyError(f"Rule requires missing risk column {risk_column!r}")
        risk = pd.to_numeric(out[risk_column], errors="coerce")
        head_mask = out["head"].eq(str(rule["head"]))
        bind = head_mask & risk.ge(float(rule["threshold"]))
        before = size.copy()
        size.loc[bind] = size.loc[bind] * float(rule["size_multiplier"])
        head_rows = int(head_mask.sum())
        records.append(
            {
                **rule,
                "bound_rows": int(bind.sum()),
                "head_rows": head_rows,
                "bound_row_share_within_head": float(bind.sum() / head_rows) if head_rows else np.nan,
                "mean_size_before_bound": float(before.loc[bind].mean()) if bind.any() else np.nan,
                "mean_size_after_bound": float(size.loc[bind].mean()) if bind.any() else np.nan,
            }
        )
    out["portfolio_size_multiplier"] = size.astype("float32")
    return out.drop(columns=["head"], errors="ignore").reset_index(drop=True), records


def _rule_label(rules: List[Dict[str, Any]]) -> str:
    chunks = []
    for rule in rules:
        family = str(rule["diagnostic_family"]).replace("diagnostic_", "").replace("_risk", "")
        threshold = int(round(float(rule["threshold"]) * 1000))
        multiplier = int(round(float(rule["size_multiplier"]) * 1000))
        chunks.append(f"{rule['head']}__{family}_gte{threshold}_sizex{multiplier:03d}")
    return "__".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--combo-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--rule",
        action="append",
        required=True,
        help="head:diagnostic_family:threshold:size_multiplier. May be repeated.",
    )
    parser.add_argument("--groups", default="uncertainty,drift,ood,recent_hr_surprise")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument(
        "--risk-reference-end",
        default="",
        help="Optional inclusive end timestamp for percentile-reference rows. Defaults to --start when provided.",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = [_parse_rule(rule) for rule in args.rule]
    candidates, mapping, input_paths = _combo_candidates(args.source_dir, str(args.combo_id))
    groups = _parse_csv(args.groups)

    reference_candidates = None
    reference_end = str(args.risk_reference_end or args.start or "")
    if reference_end:
        ref_ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
        reference_candidates = candidates.loc[ref_ts.lt(pd.Timestamp(reference_end, tz="UTC"))].copy()
        if reference_candidates.empty:
            raise ValueError(f"No percentile-reference rows before {reference_end!r}")

    scored = _diagnostic_scores(candidates, groups, reference_candidates=reference_candidates)
    if args.start or args.end:
        scored = scored.loc[_timestamp_mask(scored, start=str(args.start), end=str(args.end))].copy()
        if scored.empty:
            raise ValueError(f"No candidate rows remain after start/end filtering: {args.start!r}, {args.end!r}")

    missing = sorted({str(rule["risk_column"]) for rule in rules} - set(scored.columns))
    if missing:
        raise KeyError(f"Missing rule risk columns after scoring: {missing}")

    label = str(args.label or _rule_label(rules))
    run_dir = args.out_dir / "multi_overlay" / "materialized" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    filtered, rule_records = _apply_rules(scored, rules)
    filtered.to_parquet(run_dir / "combo_candidates.parquet", index=False)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(filtered)
    decisions, equity, metrics = replay_candidates(
        filtered,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(args.market_mode),
    )
    decisions.to_parquet(run_dir / "combo_replay_decisions.parquet", index=False)
    equity.to_parquet(run_dir / "combo_replay_equity.parquet", index=False)
    daily, weekly = _period_tables(decisions)
    daily.to_csv(run_dir / "combo_replay_daily_metrics.csv", index=False)
    weekly.to_csv(run_dir / "combo_replay_weekly_metrics.csv", index=False)
    pd.DataFrame(rule_records).to_csv(run_dir / "multi_overlay_rule_coverage.csv", index=False)

    manifest = {
        "generated_by": "run_contextual_tp_sl_multi_head_diagnostic_overlay",
        "source_dir": str(args.source_dir),
        "combo_id": str(args.combo_id),
        "head_arm_mapping": dict(mapping),
        "gate_head": "multi_head",
        "rules": rule_records,
        "groups": groups,
        "start_filter": str(args.start),
        "end_filter": str(args.end),
        "risk_reference_end": reference_end or None,
        "risk_reference_rows": int(len(reference_candidates)) if reference_candidates is not None else None,
        "market_mode": str(args.market_mode),
        "candidate_rows": int(len(filtered)),
        "policy_params": asdict(params),
        "input_hashes": {arm: _sha256(path) for arm, path in sorted(input_paths.items())},
        "metrics": dict(metrics),
    }
    (run_dir / "combo_replay_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "label": label,
        "out_dir": str(run_dir),
        "net_pnl": metrics.get("net_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "trade_count": metrics.get("trade_count"),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "rule_count": len(rule_records),
        "candidate_rows": int(len(filtered)),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "multi_overlay_summary.csv", index=False)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
