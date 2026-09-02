#!/usr/bin/env python3
"""Compare two Pack-B outer-OOF routes on exact shared rows and costs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from scripts.run_packb_side_local_outer_oof import (
    ECONOMIC_COLUMN,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    _metrics,
)

SCHEMA = "packb_outer_oof_feature_route_gate_v1"
SIDES = ("long", "short")
IDENTITY_COLUMNS = (
    "side_name",
    "__ts__",
    "__symbol__",
    "outer_fold",
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    ECONOMIC_COLUMN,
)


class PackBOuterRouteGateError(RuntimeError):
    """Raised when the paired outer-OOF comparison is not exact."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PackBOuterRouteGateError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _load_arm(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = root / "summary.json"
    predictions_path = root / "oof_predictions.parquet"
    summary = _json(summary_path)
    if (
        summary.get("status") != "COMPLETE_STRICT_SIDE_LOCAL_OUTER_OOF_AND_FINAL_REFITS"
        or summary.get("validation_sampling") != "full_authorized_outer_rows"
        or summary.get("final_refit_predictions_used_in_oof") is not False
    ):
        raise PackBOuterRouteGateError(f"promotion-grade outer OOF required: {root}")
    frame = pd.read_parquet(predictions_path)
    required = {"candidate_id", "prediction", *IDENTITY_COLUMNS}
    if required.difference(frame.columns):
        raise PackBOuterRouteGateError(f"OOF schema is incomplete: {root}")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise PackBOuterRouteGateError(f"OOF candidate IDs are duplicated: {root}")
    if len(frame) != int(summary.get("oof_rows", -1)):
        raise PackBOuterRouteGateError(f"OOF row count changed: {root}")
    return summary, frame


def _exact_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    paired = left.merge(
        right,
        on="candidate_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if paired.empty:
        raise PackBOuterRouteGateError("outer-OOF arms have no shared candidate IDs")
    for column in IDENTITY_COLUMNS:
        left_value = paired[f"{column}_left"]
        right_value = paired[f"{column}_right"]
        if pd.api.types.is_numeric_dtype(left_value):
            equal = np.allclose(
                left_value,
                right_value,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            equal = left_value.astype(str).equals(right_value.astype(str))
        if not equal:
            raise PackBOuterRouteGateError(
                f"paired rows disagree on identity or cost column {column!r}"
            )
    return paired


def _score(paired: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    ledger = pd.DataFrame(
        {
            "__ts__": paired["__ts___left"],
            "__symbol__": paired["__symbol___left"],
        }
    )
    labels = pd.DataFrame(
        {
            TARGET_COLUMN: paired[f"{TARGET_COLUMN}_left"],
            WEIGHT_COLUMN: paired[f"{WEIGHT_COLUMN}_left"],
            ECONOMIC_COLUMN: paired[f"{ECONOMIC_COLUMN}_left"],
        }
    )
    return _metrics(paired[prediction_column].to_numpy(), ledger, labels)


def _winner(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    metrics = (
        "objective",
        "weighted_rank_ic",
        "top10_net_return_lift",
        "relative_rmse_gain",
    )
    right_higher = all(float(right[name]) > float(left[name]) for name in metrics)
    left_higher = all(float(left[name]) > float(right[name]) for name in metrics)
    if right_higher:
        return "right"
    if left_higher:
        return "left"
    return "mixed_requires_review"


def run(
    *,
    left_root: Path,
    right_root: Path,
    output_path: Path,
    left_name: str = "hist55_37",
    right_name: str = "fresh31_8",
) -> dict[str, Any]:
    left_summary, left = _load_arm(Path(left_root))
    right_summary, right = _load_arm(Path(right_root))
    if (
        left_summary["outer_population_manifest_sha256"]
        != right_summary["outer_population_manifest_sha256"]
        or left_summary["fixed_calendar_sha256"]
        != right_summary["fixed_calendar_sha256"]
    ):
        raise PackBOuterRouteGateError("outer population or calendar differs by arm")
    paired = _exact_pair(left, right)
    sides: dict[str, Any] = {}
    winners: list[str] = []
    for side in SIDES:
        side_rows = paired.loc[paired["side_name_left"].eq(side)].reset_index(drop=True)
        if side_rows.empty:
            raise PackBOuterRouteGateError(f"paired side is empty: {side}")
        aggregate = {
            left_name: _score(side_rows, "prediction_left"),
            right_name: _score(side_rows, "prediction_right"),
        }
        folds: dict[str, Any] = {}
        for fold in sorted(side_rows["outer_fold_left"].astype(str).unique()):
            fold_rows = side_rows.loc[
                side_rows["outer_fold_left"].astype(str).eq(fold)
            ].reset_index(drop=True)
            folds[fold] = {
                left_name: _score(fold_rows, "prediction_left"),
                right_name: _score(fold_rows, "prediction_right"),
            }
        winner = _winner(aggregate[left_name], aggregate[right_name])
        winners.append(winner)
        sides[side] = {
            "paired_rows": int(len(side_rows)),
            "aggregate": aggregate,
            "folds": folds,
            "winner": right_name
            if winner == "right"
            else left_name
            if winner == "left"
            else winner,
            "right_minus_left": {
                name: float(aggregate[right_name][name])
                - float(aggregate[left_name][name])
                for name in (
                    "objective",
                    "weighted_rank_ic",
                    "top10_net_return_lift",
                    "relative_rmse_gain",
                )
            },
        }
    status = (
        f"RETAIN_{right_name.upper()}_HIGHER_PAIRED_METRICS"
        if winners == ["right", "right"]
        else "MIXED_SIDE_RESULT_REQUIRES_REVIEW"
    )
    result = {
        "schema": SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_contract": (
            "exact candidate-ID intersection; exact side/timestamp/symbol/fold/"
            "target/weight/net-return equality; within-timestamp side ranking"
        ),
        "selection_rule": (
            "55/37 is default unless 31/8 is higher on paired aggregate "
            "objective, rank IC, top-10 net-return lift, and relative RMSE gain"
        ),
        "arms": {
            left_name: {
                "root": str(Path(left_root)),
                "summary_sha256": stage_manifest.sha256_file(
                    Path(left_root) / "summary.json"
                ),
                "predictions_sha256": stage_manifest.sha256_file(
                    Path(left_root) / "oof_predictions.parquet"
                ),
                "raw_oof_rows": int(len(left)),
            },
            right_name: {
                "root": str(Path(right_root)),
                "summary_sha256": stage_manifest.sha256_file(
                    Path(right_root) / "summary.json"
                ),
                "predictions_sha256": stage_manifest.sha256_file(
                    Path(right_root) / "oof_predictions.parquet"
                ),
                "raw_oof_rows": int(len(right)),
            },
        },
        "paired_rows": int(len(paired)),
        "sides": sides,
        "outer_population_manifest_sha256": left_summary[
            "outer_population_manifest_sha256"
        ],
        "fixed_calendar_sha256": left_summary["fixed_calendar_sha256"],
        "final_refit_predictions_used": False,
    }
    _atomic_json(Path(output_path), result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-name", default="hist55_37")
    parser.add_argument("--right-name", default="fresh31_8")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(
            left_root=args.left_root,
            right_root=args.right_root,
            output_path=args.output,
            left_name=args.left_name,
            right_name=args.right_name,
        )
    except (PackBOuterRouteGateError, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_PRECONDITION_FAILED", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
