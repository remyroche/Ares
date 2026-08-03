#!/usr/bin/env python3
"""Evaluate OOF candidate score streams under the expanded roadmap contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.candidate_evaluation import FRACTIONS, TailGate, evaluate_global_book, paired_day_block_bootstrap, stable_global_top_k, tail_gates


SCHEMA = "expanded_roadmap_candidate_evaluation_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover(frame: pd.DataFrame, choices: tuple[str, ...]) -> str | None:
    return next((name for name in choices if name in frame), None)


def _unit(column: str, requested: str) -> str:
    return ("bps" if column.endswith("_bps") else "return") if requested == "auto" else requested


def _groups(frame: pd.DataFrame, supplied: str | None) -> list[str]:
    if supplied:
        columns = [item.strip() for item in supplied.split(",") if item.strip()]
    else:
        columns = [column for column in ("target_arm", "support_stage", "model_family", "seed") if column in frame]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"group columns unavailable: {sorted(missing)}")
    return columns


def evaluate(frame: pd.DataFrame, *, score: str, net: str, net_unit: str, gross: str | None, gross_unit: str | None, cost: str | None, cost_unit: str | None, group_columns: list[str], regime: str | None, liquidity: str | None, hurdle: str | None, gate: TailGate, bootstrap_baseline_column: str | None, bootstrap_baseline_value: str | None) -> dict[str, pd.DataFrame]:
    result: dict[str, list[pd.DataFrame]] = {"global_tail_metrics": [], "side_attribution": [], "month_attribution": [], "regime_attribution": [], "cost_attribution": [], "liquidity_attribution": [], "hurdle_attribution": [], "tail_gates": []}
    selected_top10: dict[tuple[object, ...], pd.DataFrame] = {}
    iterator = frame.groupby(group_columns, observed=True, sort=True) if group_columns else [((), frame)]
    for key, local in iterator:
        key = key if isinstance(key, tuple) else (key,)
        prefix = dict(zip(group_columns, key, strict=True))
        tail, diagnostics = evaluate_global_book(local, score_column=score, net_column=net, net_unit=net_unit, gross_column=gross, gross_unit=gross_unit, cost_column=cost, cost_unit=cost_unit, regime_column=regime, liquidity_column=liquidity, hurdle_column=hurdle)
        for table_name, table in (("global_tail_metrics", tail), ("side_attribution", diagnostics.get("side", pd.DataFrame())), ("month_attribution", diagnostics.get("month", pd.DataFrame())), ("regime_attribution", diagnostics.get("regime", pd.DataFrame())), ("cost_attribution", diagnostics.get("cost", pd.DataFrame())), ("liquidity_attribution", diagnostics.get("liquidity", pd.DataFrame())), ("hurdle_attribution", diagnostics.get("hurdle", pd.DataFrame()))):
            if not table.empty:
                result[table_name].append(table.assign(**prefix))
        result["tail_gates"].append(tail_gates(tail, diagnostics["side"], gate=gate).assign(**prefix))
        selected_top10[key] = stable_global_top_k(local, score, .10)
    tables = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in result.items()}
    bootstrap: list[dict[str, object]] = []
    if bootstrap_baseline_column and bootstrap_baseline_value:
        if bootstrap_baseline_column not in group_columns:
            raise ValueError("bootstrap baseline column must be one of group columns")
        baseline_position = group_columns.index(bootstrap_baseline_column)
        for key, selected in selected_top10.items():
            if str(key[baseline_position]) == bootstrap_baseline_value:
                continue
            counterpart = list(key); counterpart[baseline_position] = bootstrap_baseline_value
            baseline = selected_top10.get(tuple(counterpart))
            if baseline is None:
                continue
            bootstrap.append({**dict(zip(group_columns, key, strict=True)), "baseline_value": bootstrap_baseline_value, **paired_day_block_bootstrap(baseline, selected, net_column=net, net_unit=net_unit)})
    tables["paired_day_block_bootstrap"] = pd.DataFrame(bootstrap)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--net-column", default=None)
    parser.add_argument("--gross-column", default=None)
    parser.add_argument("--cost-column", default=None)
    parser.add_argument("--outcome-unit", choices=("auto", "return", "bps"), default="auto")
    parser.add_argument("--group-columns", default=None)
    parser.add_argument("--regime-column", default=None)
    parser.add_argument("--liquidity-column", default=None)
    parser.add_argument("--hurdle-column", default=None)
    parser.add_argument("--bootstrap-baseline-column", default=None)
    parser.add_argument("--bootstrap-baseline-value", default=None)
    parser.add_argument("--filter", action="append", default=[], metavar="COLUMN=VALUE", help="Repeatable exact pre-evaluation population filter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    frame = pd.read_parquet(args.predictions)
    applied_filters: dict[str, str] = {}
    for item in args.filter:
        if "=" not in item:
            raise ValueError("--filter must use COLUMN=VALUE")
        column, value = item.split("=", 1)
        if column not in frame:
            raise ValueError(f"filter column is absent: {column}")
        frame = frame.loc[frame[column].astype(str).eq(value)].copy()
        applied_filters[column] = value
    if frame.empty:
        raise ValueError("filters left no candidate rows")
    if args.score_column not in frame:
        raise ValueError(f"score column is absent: {args.score_column}")
    net = args.net_column or _discover(frame, ("execution_net_ev_12h", "net_h12_bps"))
    gross = args.gross_column or _discover(frame, ("execution_gross_ev_12h", "gross_h12_bps"))
    cost = args.cost_column or _discover(frame, ("execution_cost_return", "cost_bps"))
    if net is None:
        raise ValueError("no exact net outcome column found; pass --net-column")
    groups = _groups(frame, args.group_columns)
    regime = args.regime_column or _discover(frame, ("regime", "regime_label", "market_regime"))
    liquidity = args.liquidity_column or _discover(frame, ("liquidity_bucket", "liquidity_score", "amihud_illiq"))
    hurdle = args.hurdle_column or _discover(frame, ("hurdle_probability", "hurdle_score", "p_hurdle"))
    gate = TailGate()
    tables = evaluate(frame, score=args.score_column, net=net, net_unit=_unit(net, args.outcome_unit), gross=gross, gross_unit=_unit(gross, args.outcome_unit) if gross else None, cost=cost, cost_unit=_unit(cost, args.outcome_unit) if cost else None, group_columns=groups, regime=regime, liquidity=liquidity, hurdle=hurdle, gate=gate, bootstrap_baseline_column=args.bootstrap_baseline_column, bootstrap_baseline_value=args.bootstrap_baseline_value)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        output_hashes = {}
        for name, table in tables.items():
            path = stage / f"{name}.parquet"; table.to_parquet(path, index=False); output_hashes[path.name] = _sha256(path)
        manifest: dict[str, Any] = {"schema": SCHEMA, "status": "RESEARCH_ONLY_DIAGNOSTIC_NO_POLICY_OR_PROMOTION_CHANGE", "input": {"path": str(args.predictions), "sha256": _sha256(args.predictions), "filters": applied_filters, "filtered_rows": int(len(frame))}, "score_column": args.score_column, "outcomes": {"net": net, "gross": gross, "cost": cost, "unit": args.outcome_unit}, "group_columns": groups, "diagnostic_columns": {"regime": regime, "liquidity": liquidity, "hurdle": hurdle}, "selection": "one pooled global top 1/5/10/20 percent per configuration, score descending/candidate-id ascending ties; no side/timestamp/asset/regime quota, replacement or backfill", "gates": gate.manifest(), "bootstrap": "paired circular UTC-day-block bootstrap only when an explicit baseline group is supplied; selection is frozen before resampling", "limitations": ["Attribution tables do not establish causality or alter selection.", "No portfolio constraints, sizing, timing, wait action or threshold policy is replayed.", "Tail gates are diagnostic and never authorize promotion."], "outputs_sha256": output_hashes}
        (stage / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise


if __name__ == "__main__":
    main()
