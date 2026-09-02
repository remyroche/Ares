#!/usr/bin/env python3
"""Fail-closed gate audit for the frozen direct exact-net challenger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256
from scripts.run_cross_era_direct_net_quantile_challenger import (
    SIDES,
    _assert_complete_current_labels,
    _binding,
    _probability_calibration_metrics,
    _tail_economics_by_period,
    _write_json,
)


SCHEMA = "cross_era_direct_net_quantile_challenger_gate_audit_v1"


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="spearman")) if len(x) >= 3 else float("nan")


def predictive_metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    work["net_bps"] = pd.to_numeric(work["execution_net_ev_12h"], errors="raise") * 1e4
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        side_frame = work.loc[work["side_name"].astype(str).eq(side)]
        periods = [("aggregate", "all", side_frame)]
        periods.extend(("month", str(month), local) for month, local in side_frame.groupby("month", sort=True))
        for level, period, local in periods:
            actual = local["net_bps"].to_numpy(float)
            for quantile in (10, 25, 50, 75):
                column = f"q{quantile:02d}_net_bps"
                prediction = local[column].to_numpy(float)
                alpha = quantile / 100.0
                error = actual - prediction
                pinball = np.maximum(alpha * error, (alpha - 1.0) * error)
                rows.append({
                    "split": split,
                    "level": level,
                    "period": period,
                    "side_name": side,
                    "head": column,
                    "rows": int(len(local)),
                    "rank_ic": _rank_ic(local[column], local["net_bps"]),
                    "pinball_loss_bps": float(pinball.mean()),
                    "mean_prediction_bps": float(prediction.mean()),
                    "mean_actual_bps": float(actual.mean()),
                })
    return pd.DataFrame(rows)


def promotion_gates(
    historical_economics: pd.DataFrame,
    current_economics: pd.DataFrame,
) -> dict[str, Any]:
    def value(table: pd.DataFrame, level: str, period: str, scope: str) -> float:
        row = table.loc[
            table["level"].eq(level)
            & table["period"].eq(period)
            & table["scope"].eq(scope)
        ]
        if len(row) != 1:
            raise ValueError(f"missing unique gate row: {level}/{period}/{scope}")
        return float(row.iloc[0]["net_ev_bps"])

    months = historical_economics.loc[
        historical_economics["level"].eq("month")
        & historical_economics["scope"].eq("global")
    ].sort_values("period")
    latest = months.iloc[-1]
    gates = {
        "historical_global_top10_positive": value(
            historical_economics, "aggregate", "all", "global"
        ) > 0.0,
        "historical_latest_month_positive": float(latest["net_ev_bps"]) > 0.0,
        "historical_all_months_positive": bool((months["net_ev_bps"] > 0.0).all()),
        "current_global_top10_positive": value(
            current_economics, "aggregate", "all", "global"
        ) > 0.0,
        "current_long_local_top10_positive": value(
            current_economics, "aggregate", "all", "side_local_long"
        ) > 0.0,
        "current_short_local_top10_positive": value(
            current_economics, "aggregate", "all", "side_local_short"
        ) > 0.0,
    }
    gates["pre_portfolio_gate_passed"] = bool(all(gates.values()))
    return gates


def _verify_source(source_dir: Path) -> dict[str, Any]:
    report_path = source_dir / "report.json"
    manifest_path = source_dir / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256(report_path) != manifest["report"]["sha256"]:
        raise ValueError("source report hash mismatch")
    for name, record in report["outputs"].items():
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"source output hash mismatch: {name}")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = _verify_source(args.source_dir)
    history = pd.read_parquet(source["outputs"]["historical_oof_winner"]["path"])
    current = pd.read_parquet(source["outputs"]["current_scored_exact"]["path"])
    pack = pd.read_parquet(args.current_pack, columns=list(IDENTITY))
    labels = pd.read_parquet(args.current_labels, columns=list(IDENTITY))
    coverage = _assert_complete_current_labels(pack, labels)
    coverage["scored_rows"] = int(len(current))
    if len(current) != coverage["prediction_rows"]:
        raise ValueError("source current scored rows do not match declared pack")

    score_column = str(source["winner"]["mapped_column"])
    historical_economics = _tail_economics_by_period(history, score_column, "historical_oof")
    current_economics = _tail_economics_by_period(current, score_column, "current")
    raw_historical_economics = _tail_economics_by_period(history, source["winner"]["score_column"], "historical_oof_raw")
    raw_current_economics = _tail_economics_by_period(current, source["winner"]["score_column"], "current_raw")
    prediction_metrics = pd.concat(
        [predictive_metrics(history, "historical_oof"), predictive_metrics(current, "current")],
        ignore_index=True,
    )
    probability_metrics = pd.concat(
        [
            _probability_calibration_metrics(history, "historical_oof"),
            _probability_calibration_metrics(current, "current"),
        ],
        ignore_index=True,
    )
    gates = promotion_gates(historical_economics, current_economics)

    args.output_dir.mkdir(parents=True)
    tables = {
        "historical_mapped_economics": historical_economics,
        "historical_raw_economics": raw_historical_economics,
        "current_mapped_economics": current_economics,
        "current_raw_economics": raw_current_economics,
        "predictive_metrics": prediction_metrics,
        "probability_metrics": probability_metrics,
    }
    outputs: dict[str, Any] = {}
    for name, table in tables.items():
        path = args.output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = {**_binding(path), "rows": int(len(table))}

    report = {
        "schema": SCHEMA,
        "status": "completed_research_only_no_promotion",
        "promotion_eligible": False,
        "portfolio_replay_authorized": bool(gates["pre_portfolio_gate_passed"]),
        "source": {
            "report": _binding(args.source_dir / "report.json"),
            "manifest": _binding(args.source_dir / "manifest.json"),
            "current_pack": _binding(args.current_pack),
            "current_labels": _binding(args.current_labels),
        },
        "coverage": coverage,
        "winner": source["winner"],
        "gates": gates,
        "calibration_contract": (
            "The v1 frozen source persisted calibrated severe probabilities only; "
            "raw-versus-calibrated parity requires the strengthened future runner."
        ),
        "outputs": outputs,
    }
    _write_json(args.output_dir / "report.json", report)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "status": report["status"],
            "promotion_eligible": False,
            "report": _binding(args.output_dir / "report.json"),
            "outputs": outputs,
        },
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"),
    )
    result.add_argument(
        "--current-pack",
        type=Path,
        default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/packb/packb_forward_context.parquet"),
    )
    result.add_argument(
        "--current-labels",
        type=Path,
        default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_12h/execution_ev_policy_labels.parquet"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/artifacts/cross_era_direct_net_quantile_challenger_gate_audit_20260730_v1"),
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
