#!/usr/bin/env python3
"""Choose between matched LDF feature contracts without touching score ranking.

The LDF is a strictly post-admission *sizing* overlay.  This utility therefore
only blends two independently trained, chronological OOF size multipliers on
identical candidate IDs.  It never blends or recalibrates ``final_score`` and
cannot alter candidate selection, rank, or causal EV admission.

Selection is made on the 2025 development replay only.  The 2026 table is
written as a frozen confirmation and is deliberately excluded from the winner
decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_n5_canonical_selection import (  # noqa: E402
    TAILS,
    _objective,
    _period_tail_metrics,
    _stability,
)


SCHEMA = "strict_r3_ldf_contract_blend_v1"
WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "candidate_id",
        "__decision_ts__",
        "final_score",
        "policy_net_bps",
        "trust_size_multiplier",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required LDF OOF columns: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate candidate IDs")
    return frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def blend_outputs(
    compact: pd.DataFrame,
    full: pd.DataFrame,
    *,
    compact_weight: float,
) -> pd.DataFrame:
    """Return an identity-preserving convex blend of two size multipliers."""

    if not 0.0 <= float(compact_weight) <= 1.0:
        raise ValueError("compact weight must be in [0, 1]")
    if len(compact) != len(full) or not compact["candidate_id"].equals(full["candidate_id"]):
        raise ValueError("LDF blend requires exactly the same ordered candidate IDs")
    for column in ("__decision_ts__", "final_score", "policy_net_bps"):
        left = compact[column]
        right = full[column]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.allclose(left.to_numpy(float), right.to_numpy(float), equal_nan=True)
        else:
            equal = left.equals(right)
        if not equal:
            raise ValueError(f"LDF contracts disagree on frozen upstream {column}")
    output = compact.copy()
    weight = float(compact_weight)
    output["trust_size_multiplier"] = np.clip(
        weight * pd.to_numeric(compact["trust_size_multiplier"], errors="coerce").to_numpy(float)
        + (1.0 - weight)
        * pd.to_numeric(full["trust_size_multiplier"], errors="coerce").to_numpy(float),
        0.25,
        1.75,
    ).astype(np.float32)
    output["ldf_compact_weight"] = np.float32(weight)
    output["ldf_full_weight"] = np.float32(1.0 - weight)
    return output


def _selection_row(output: pd.DataFrame, *, arm: str) -> Mapping[str, Any]:
    _score, metrics = _objective(output, arm=arm)
    return {"selection_score": _score, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    selection_rows: list[dict[str, Any]] = []
    global_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    weekly_rows: list[pd.DataFrame] = []
    stability_rows: list[pd.DataFrame] = []
    for year in (2025, 2026):
        compact = _load(args.compact_dir / f"oof_predictions_{year}.parquet")
        full = _load(args.full_dir / f"oof_predictions_{year}.parquet")
        for weight in WEIGHTS:
            arm = f"compact{int(round(weight * 100)):03d}_full{int(round((1.0 - weight) * 100)):03d}"
            output = blend_outputs(compact, full, compact_weight=weight)
            global_metric = _period_tail_metrics(output, arm=arm, period_kind="global").assign(year=year)
            monthly = _period_tail_metrics(output, arm=arm, period_kind="month").assign(year=year)
            weekly = _period_tail_metrics(output, arm=arm, period_kind="week").assign(year=year)
            stability = _stability(monthly.drop(columns="year")).assign(year=year)
            global_rows.append(global_metric)
            monthly_rows.append(monthly)
            weekly_rows.append(weekly)
            stability_rows.append(stability)
            if year == 2025:
                row = dict(_selection_row(output, arm=arm))
                row.update(
                    arm=arm,
                    compact_weight=weight,
                    full_weight=1.0 - weight,
                    selection_data="2025 development OOF only",
                )
                selection_rows.append(row)
    selection = pd.DataFrame(selection_rows).sort_values(
        [
            "selection_score",
            "mean_portability_top1_2_5",
            "worst_month_top1_2_5",
            "top1_net_bps",
        ],
        ascending=False,
        kind="stable",
    )
    selection.to_parquet(args.out_dir / "selection_2025.parquet", index=False)
    pd.concat(global_rows, ignore_index=True).to_parquet(
        args.out_dir / "metrics_global.parquet", index=False,
    )
    pd.concat(monthly_rows, ignore_index=True).to_parquet(
        args.out_dir / "metrics_monthly.parquet", index=False,
    )
    pd.concat(weekly_rows, ignore_index=True).to_parquet(
        args.out_dir / "metrics_weekly.parquet", index=False,
    )
    pd.concat(stability_rows, ignore_index=True).to_parquet(
        args.out_dir / "stability.parquet", index=False,
    )
    winner = selection.iloc[0].to_dict()
    (args.out_dir / "winner.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "selection_era": "2025 development OOF only",
                "confirmation_era": "2026; excluded from selection",
                "winner": winner,
                "invariants": {
                    "final_score_blended": False,
                    "admission_changed": False,
                    "candidate_identity_changed": False,
                    "size_bounds": [0.25, 1.75],
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
