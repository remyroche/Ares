#!/usr/bin/env python3
"""Measure whether OOS detector alerts recognize frozen adverse episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.residual_event_block_taxonomy import (
    detector_recognized_missed_contrasts,
)


ARMS = {
    "same_day": ("negative_ev_onset", 0, 0),
    "lead_1d": ("next1d_negative_ev_onset", -1, -1),
    "lead_3d": ("next3d_negative_ev_onset", -3, -1),
}


def _technical_mode(assignments: pd.DataFrame) -> pd.Series:
    semantic = assignments.get(
        "semantic_label", pd.Series("unresolved", index=assignments.index)
    ).fillna("unresolved").astype(str)
    return (
        semantic
        + "::"
        + assignments["method"].astype(str)
        + "__d"
        + assignments["latent_dim"].astype(str)
        + "__k"
        + assignments["clusters"].astype(str)
        + "__c"
        + assignments["cluster_id"].astype(str)
    )


def _attach_arm(
    blocks: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    arm: str,
    target_name: str,
    start_offset: int,
    end_offset: int,
) -> pd.DataFrame:
    detector = predictions.loc[predictions["failure_mode"].eq(target_name)].copy()
    detector["day"] = pd.to_datetime(detector["day"], utc=True).dt.floor("D")
    detector["alert"] = detector["alert"].astype(bool)
    detector["risk"] = pd.to_numeric(detector["risk"], errors="coerce")
    output = blocks.copy()
    maximum: list[float] = []
    recognized: list[bool] = []
    assessable: list[bool] = []
    for block in output.itertuples(index=False):
        start = pd.Timestamp(block.event_start) + pd.Timedelta(days=start_offset)
        end = pd.Timestamp(block.event_start) + pd.Timedelta(days=end_offset)
        local = detector.loc[
            detector["side_name"].eq(block.side_name)
            & detector["archetype_policy_key"].eq(block.archetype_policy_key)
            & detector["day"].between(start, end)
        ]
        assessable.append(bool(len(local)))
        recognized.append(bool(local["alert"].any()) if len(local) else False)
        maximum.append(float(local["risk"].max()) if len(local) else np.nan)
    output[f"{arm}_assessable"] = assessable
    output[f"{arm}_recognized"] = recognized
    output[f"{arm}_max_risk"] = maximum
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy, detector, output = (
        Path(args.taxonomy),
        Path(args.detector),
        Path(args.output),
    )
    output.mkdir(parents=True, exist_ok=True)
    blocks = pd.read_parquet(taxonomy / "local_failure_block_taxonomy.parquet")
    assignments = pd.read_parquet(
        taxonomy / "local_frozen_failure_mode_semantic_assignments.parquet"
    ).copy()
    assignments["frozen_failure_mode"] = _technical_mode(assignments)
    assignment_columns = [
        "side_name",
        "archetype_policy_key",
        "event_block",
        "frozen_failure_mode",
    ]
    blocks = blocks.merge(
        assignments.loc[:, assignment_columns].drop_duplicates(
            ["side_name", "archetype_policy_key", "event_block"]
        ),
        on=["side_name", "archetype_policy_key", "event_block"],
        how="left",
        validate="one_to_one",
    )
    predictions = pd.read_parquet(detector / "local_oos_predictions.parquet")
    for arm, (target, start_offset, end_offset) in ARMS.items():
        blocks = _attach_arm(
            blocks,
            predictions,
            arm=arm,
            target_name=target,
            start_offset=start_offset,
            end_offset=end_offset,
        )
    blocks.to_parquet(output / "local_episode_detector_coverage.parquet", index=False)

    rows: list[dict[str, Any]] = []
    group_keys = ["side_name", "archetype_policy_key", "frozen_failure_mode"]
    for values, local in blocks.groupby(group_keys, observed=True, sort=True):
        row = dict(zip(group_keys, values))
        row["episodes"] = int(len(local))
        for arm in ARMS:
            assessable = local[f"{arm}_assessable"].astype(bool)
            recognized = local[f"{arm}_recognized"].astype(bool)
            row[f"{arm}_assessable_episodes"] = int(assessable.sum())
            row[f"{arm}_recognized_episodes"] = int((assessable & recognized).sum())
            row[f"{arm}_recall"] = float(recognized[assessable].mean()) if assessable.any() else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "local_mode_episode_coverage.csv", index=False)

    contrast_input = blocks.rename(
        columns={
            "same_day_assessable": "detector_assessable",
            "same_day_recognized": "detector_recognized",
        }
    )
    contrasts = detector_recognized_missed_contrasts(contrast_input)
    contrasts.to_csv(output / "recognized_vs_missed_feature_contrasts.csv", index=False)
    manifest = {
        "schema": "failure_detector_episode_coverage_v1",
        "taxonomy": str(taxonomy.resolve()),
        "detector": str(detector.resolve()),
        "episodes": int(len(blocks)),
        "oos_assessable_episodes": {
            arm: int(blocks[f"{arm}_assessable"].sum()) for arm in ARMS
        },
        "oos_recognized_episodes": {
            arm: int(blocks[f"{arm}_recognized"].sum()) for arm in ARMS
        },
        "contract": (
            "Coverage is read-only OOS evidence. Same-day uses day-open state; "
            "lead arms use only alerts emitted before episode onset."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
