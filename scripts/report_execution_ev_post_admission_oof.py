#!/usr/bin/env python3
"""Rank completed execution-EV OOF scores after the causal 21-day correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import FeatureProvenance  # noqa: E402
from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ExecutionEVModelAblationConfig,
    _ranking_metrics,
    apply_execution_ev_causal_recent_ev_correction,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SCHEMA = "execution_ev_post_admission_global_oof_report_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _provenance(path: Path) -> dict[str, FeatureProvenance]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(name): FeatureProvenance(
            family=str(spec["family"]),
            source=str(spec["source"]),
            pre_entry=bool(spec.get("pre_entry", True)),
            available_at_col=spec.get("available_at_col"),
            oof_or_frozen=bool(spec.get("oof_or_frozen", True)),
            model_input=bool(spec.get("model_input", True)),
            class_order=spec.get("class_order"),
            class_order_sha256=spec.get("class_order_sha256"),
        )
        for name, spec in payload["features"].items()
    }


def _selected_breakdown(
    frame: pd.DataFrame,
    actual: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    local = frame.loc[selected, ["side_name", "execution_decision_utc"]].copy()
    local["actual"] = actual[selected]
    local["month"] = pd.to_datetime(
        local["execution_decision_utc"], utc=True
    ).dt.strftime("%Y-%m")

    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(group)),
            "mean_net_ev": float(group["actual"].mean()),
            "sum_net_ev": float(group["actual"].sum()),
            "positive_rate": float((group["actual"] > 0.0).mean()),
        }

    return {
        "by_side": {
            str(key): summarize(group)
            for key, group in local.groupby("side_name", sort=True)
        },
        "by_month": {
            str(key): summarize(group)
            for key, group in local.groupby("month", sort=True)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.output_dir}")
    frame = pd.read_parquet(args.oof)
    missing = sorted(
        set(
            (
                *IDENTITY,
                "execution_decision_utc",
                "execution_label_end_utc",
                "execution_net_ev_12h",
                "execution_gross_ev_12h",
                "catboost_archetype",
            )
        )
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"OOF ledger is missing columns: {missing}")
    prediction_columns = [
        column
        for column in frame.columns
        if "__" in column
        and not column.endswith("__is_oof")
        and (
            column.startswith(("direct__", "residual__"))
            or column == "baseline__existing_alpha"
        )
    ]
    model_columns = [
        column for column in prediction_columns if column != "baseline__existing_alpha"
    ]
    if not model_columns:
        raise ValueError("OOF ledger has no model prediction columns")
    masks = [
        np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
        for column in model_columns
    ]
    shared_oof = masks[0]
    if any(not np.array_equal(shared_oof, mask) for mask in masks[1:]):
        raise ValueError("model arms do not share identical OOF rows")
    actual = pd.to_numeric(frame["execution_net_ev_12h"], errors="coerce").to_numpy(
        dtype=float
    )
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="coerce").to_numpy(
        dtype=float
    )
    shared_oof &= np.isfinite(actual) & np.isfinite(gross)
    provenance = _provenance(args.provenance_json)
    config = replace(
        ExecutionEVModelAblationConfig(),
        decision_time_col="execution_decision_utc",
        label_end_time_col="execution_label_end_utc",
        side_col="side_name",
        catboost_archetype_col="catboost_archetype",
        recent_ev_window_days=int(args.window_days),
        recent_ev_trim_fraction=float(args.trim_fraction),
        top_k_fraction=float(args.top_fraction),
    )
    corrected_table = frame.loc[:, list(IDENTITY)].copy()
    rows: list[dict[str, Any]] = []
    calibration: dict[str, Any] = {}
    for column in prediction_columns:
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        raw = np.where(shared_oof, raw, np.nan)
        corrected, route = apply_execution_ev_causal_recent_ev_correction(
            frame,
            raw,
            actual,
            provenance,
            route="catboost_predicted_archetype",
            config=config,
        )
        corrected_name = f"{column}__post_admission_21d"
        corrected_table[corrected_name] = corrected
        metrics = _ranking_metrics(
            actual[shared_oof],
            gross[shared_oof],
            corrected[shared_oof],
            top_k_fraction=float(args.top_fraction),
        )
        valid_positions = np.flatnonzero(shared_oof & np.isfinite(corrected))
        top_count = max(1, int(np.ceil(len(valid_positions) * args.top_fraction)))
        top_positions = valid_positions[
            np.argsort(corrected[valid_positions], kind="stable")[-top_count:]
        ]
        selected = np.zeros(len(frame), dtype=bool)
        selected[top_positions] = True
        admitted = (
            shared_oof
            & np.isfinite(corrected)
            & (corrected >= float(args.admission_threshold))
        )
        rows.append(
            {
                "prediction": column,
                "corrected_prediction": corrected_name,
                "oof_rows": int(shared_oof.sum()),
                "ranking_scope": "global_shared_outer_oof",
                "ranking_stage": "after_causal_21d_admission_calibrator",
                "top_fraction": float(args.top_fraction),
                "admission_threshold": float(args.admission_threshold),
                "admitted_rows": int(admitted.sum()),
                "admitted_mean_net_ev": (
                    float(actual[admitted].mean()) if admitted.any() else np.nan
                ),
                "admitted_sum_net_ev": (
                    float(actual[admitted].sum()) if admitted.any() else np.nan
                ),
                **metrics,
                "global_top_breakdown": _selected_breakdown(frame, actual, selected),
            }
        )
        calibration[column] = {
            key: value for key, value in route.items() if key != "daily_snapshots"
        }
    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "top_k_mean_net_ev",
            "top_k_mean_gross_ev",
            "spearman",
        ],
        ascending=False,
        kind="stable",
    )
    args.output_dir.mkdir(parents=True)
    leaderboard_path = args.output_dir / "post_admission_global_leaderboard.csv"
    json_path = args.output_dir / "report.json"
    corrected_path = args.output_dir / "corrected_oof_predictions.parquet"
    csv = leaderboard.drop(columns=["global_top_breakdown"])
    csv.to_csv(leaderboard_path, index=False)
    corrected_table.to_parquet(corrected_path, index=False, compression="zstd")
    report = {
        "schema": SCHEMA,
        "status": "evaluation_only_post_calibrator",
        "ranking_contract": {
            "scope": "global_shared_outer_oof",
            "stage": "after_causal_21d_admission_calibrator",
            "selection": "one pooled top fraction across all shared OOF rows",
            "top_fraction": float(args.top_fraction),
            "window_days": int(args.window_days),
            "trim_fraction": float(args.trim_fraction),
            "fixed_admission_threshold": float(args.admission_threshold),
        },
        "source": {
            "oof": str(args.oof),
            "oof_sha256": _sha256(args.oof),
            "provenance": str(args.provenance_json),
            "provenance_sha256": _sha256(args.provenance_json),
        },
        "shared_oof_rows": int(shared_oof.sum()),
        "calibration": calibration,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "output": {
            "corrected_oof": str(corrected_path),
            "corrected_oof_sha256": _sha256(corrected_path),
            "leaderboard": str(leaderboard_path),
            "leaderboard_sha256": _sha256(leaderboard_path),
        },
    }
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "report": json_path,
        "leaderboard": leaderboard_path,
        "corrected_oof": corrected_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=21)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--admission-threshold", type=float, default=0.007)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.window_days < 1:
        raise ValueError("--window-days must be positive")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in [0, 0.5)")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1]")
    paths = run(args)
    print(json.dumps(_json_safe(paths), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
