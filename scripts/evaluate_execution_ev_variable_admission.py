#!/usr/bin/env python3
"""Evaluate fail-closed global admission on existing strict-OOF EV scores.

This script does not fit or tune a model.  It applies fixed predicted-net-EV
floors to already materialized causal mapped scores and retains at most the
same one-pooled-global top-k capacity as the baseline.  An arm may therefore
select fewer than k rows, including zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "execution_ev_variable_admission_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_PREDICTIONS = ROOT / (
    "data_perp/artifacts/exact_policy_capture_hurdle_ablation_20260727_v1/"
    "hurdle_predictions.parquet"
)
DEFAULT_SOURCE_MANIFEST = ROOT / (
    "data_perp/artifacts/exact_policy_capture_hurdle_ablation_20260727_v1/"
    "manifest.json"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/execution_ev_variable_admission_20260727_v1"
)
TARGET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def select_variable_admission(
    score: np.ndarray,
    *,
    capacity_fraction: float,
    predicted_net_floor: float | None,
) -> np.ndarray:
    """Return selected positions under one global capacity and optional floor."""
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("score must be a finite one-dimensional array")
    if not 0.0 < float(capacity_fraction) <= 1.0:
        raise ValueError("capacity_fraction must be in (0, 1]")
    capacity = int(math.ceil(len(values) * float(capacity_fraction)))
    eligible = np.ones(len(values), dtype=bool)
    if predicted_net_floor is not None:
        eligible &= values > float(predicted_net_floor)
    positions = np.flatnonzero(eligible)
    if len(positions) > capacity:
        local_order = np.argsort(-values[positions], kind="mergesort")[:capacity]
        positions = positions[local_order]
    return positions


def _metrics(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    window: str,
    arm: str,
    rule: str,
    floor: float | None,
    capacity: int,
) -> dict[str, Any]:
    selected = frame.iloc[positions]
    rows = int(len(selected))
    net = pd.to_numeric(selected[TARGET], errors="raise").to_numpy(np.float64)
    gross = pd.to_numeric(selected[GROSS], errors="raise").to_numpy(np.float64)
    cost = pd.to_numeric(selected[COST], errors="raise").to_numpy(np.float64)
    return {
        "window": window,
        "arm": arm,
        "admission_rule": rule,
        "predicted_net_floor_bps": (
            None if floor is None else float(floor * 10_000.0)
        ),
        "candidate_rows": int(len(frame)),
        "global_capacity_rows": int(capacity),
        "selected_rows": rows,
        "capacity_fill_rate": float(rows / capacity) if capacity else 0.0,
        "candidate_admission_rate": float(rows / len(frame)) if len(frame) else 0.0,
        "mean_predicted_net_bps": (
            float(selected["canonical_recent_ev_score"].mean() * 10_000.0)
            if rows
            else float("nan")
        ),
        "mean_gross_bps": float(gross.mean() * 10_000.0) if rows else float("nan"),
        "mean_cost_bps": float(cost.mean() * 10_000.0) if rows else float("nan"),
        "mean_net_bps": float(net.mean() * 10_000.0) if rows else float("nan"),
        "sum_net_return": float(net.sum()) if rows else 0.0,
        "positive_net_rate": float(np.mean(net > 0.0)) if rows else float("nan"),
        "long_rows": int((selected["side_name"] == "long").sum()) if rows else 0,
        "short_rows": int((selected["side_name"] == "short").sum()) if rows else 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--floors-bps",
        type=float,
        nargs="+",
        default=(0.0, 25.0, 50.0),
        help="Fixed predicted net-EV admission floors; no evaluation tuning.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = json.loads(args.source_manifest.read_text())
    source = ROOT / manifest["inputs"]["data"]["path"]
    expected_source_sha = manifest["inputs"]["data"]["sha256"]
    if _sha256(source) != expected_source_sha:
        raise ValueError("canonical exact-policy input hash does not match source manifest")

    predictions = pd.read_parquet(args.predictions)
    required_prediction = {*IDENTITY, "window", "arm", "canonical_recent_ev_score"}
    missing_prediction = sorted(required_prediction - set(predictions.columns))
    if missing_prediction:
        raise ValueError(f"prediction columns missing: {missing_prediction}")
    if predictions.duplicated([*IDENTITY, "window", "arm"]).any():
        raise ValueError("prediction identity is not unique within window and arm")

    labels = pd.read_parquet(source, columns=[*IDENTITY, TARGET, GROSS, COST])
    if labels.duplicated(list(IDENTITY)).any():
        raise ValueError("canonical target identity is not unique")
    accounting_error = np.abs(
        labels[GROSS].to_numpy(np.float64)
        - labels[COST].to_numpy(np.float64)
        - labels[TARGET].to_numpy(np.float64)
    )
    if not np.isfinite(accounting_error).all() or accounting_error.max() > 1e-7:
        raise ValueError("canonical gross-cost-net accounting does not reconcile")

    joined = predictions.merge(
        labels, on=list(IDENTITY), how="left", validate="many_to_one", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("some predictions lack canonical exact-policy targets")
    joined = joined.drop(columns="_merge")

    records: list[dict[str, Any]] = []
    selections: list[pd.DataFrame] = []
    floors = [float(value) / 10_000.0 for value in args.floors_bps]
    for (window, arm), group in joined.groupby(["window", "arm"], sort=True):
        group = group.reset_index(drop=True)
        score = group["canonical_recent_ev_score"].to_numpy(np.float64)
        capacity = int(math.ceil(len(group) * float(args.top_fraction)))
        rules: list[tuple[str, float | None]] = [("forced_global_top10", None)]
        rules.extend((f"predicted_net_above_{bps:g}bps", floor) for bps, floor in zip(args.floors_bps, floors))
        for rule, floor in rules:
            selected = select_variable_admission(
                score,
                capacity_fraction=float(args.top_fraction),
                predicted_net_floor=floor,
            )
            records.append(
                _metrics(
                    group,
                    selected,
                    window=str(window),
                    arm=str(arm),
                    rule=rule,
                    floor=floor,
                    capacity=capacity,
                )
            )
            if len(selected):
                materialized = group.iloc[selected][
                    [*IDENTITY, "window", "arm", "canonical_recent_ev_score", TARGET, GROSS, COST]
                ].copy()
                materialized["admission_rule"] = rule
                selections.append(materialized)

    metrics = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "admission_metrics.csv", index=False)
    selected_frame = (
        pd.concat(selections, ignore_index=True)
        if selections
        else pd.DataFrame(
            columns=[
                *IDENTITY,
                "window",
                "arm",
                "canonical_recent_ev_score",
                TARGET,
                GROSS,
                COST,
                "admission_rule",
            ]
        )
    )
    selected_frame.to_parquet(args.output_dir / "admitted_rows.parquet", index=False)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "status": "completed_fixed_rule_oos_diagnostic_not_promotion_evidence",
            "contract": {
                "model_fit": "none; consumes existing strict-OOF/forward scores",
                "ranking": "one pooled global capacity per window and arm",
                "admission": (
                    "fixed predicted-net floor after causal mapping; may select fewer "
                    "than k, including zero; no evaluation threshold search"
                ),
                "floors_bps": list(args.floors_bps),
                "top_fraction_capacity": float(args.top_fraction),
                "accounting": "canonical exact gross minus exact row cost equals exact net",
            },
            "inputs": {
                "predictions": {
                    "path": _repo_relative(args.predictions),
                    "sha256": _sha256(args.predictions),
                },
                "source_manifest": {
                    "path": _repo_relative(args.source_manifest),
                    "sha256": _sha256(args.source_manifest),
                },
                "canonical_targets": {
                    "path": _repo_relative(source),
                    "sha256": expected_source_sha,
                },
            },
            "outputs": {
                "metrics": "admission_metrics.csv",
                "admitted_rows": "admitted_rows.parquet",
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
