#!/usr/bin/env python3
"""Combine local and shared residual-episode recognition into one calendar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["day", "side_name", "archetype_policy_key"]


def _read(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
    frame["recognized"] = frame["recognized"].fillna(False).astype(bool)
    return frame[KEYS + ["recognized", "matching_composites", "best_composite_score"]].assign(
        recognition_source=source
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.loc[calendar["adverse_event_rows"].gt(0)].drop_duplicates(KEYS)
    evidence = pd.concat(
        [_read(args.local, "side_archetype_local"), _read(args.shared, "side_global_shared")],
        ignore_index=True,
    )

    rows: list[dict[str, object]] = []
    for key, group in evidence.groupby(KEYS, observed=True):
        active = group.loc[group["recognized"]]
        rows.append(
            {
                **dict(zip(KEYS, key)),
                "recognized": not active.empty,
                "recognition_sources": "|".join(sorted(active["recognition_source"].unique())),
                "matching_composites": "|".join(
                    sorted(
                        {
                            value
                            for values in active["matching_composites"].fillna("")
                            for value in str(values).split("|")
                            if value
                        }
                    )
                ),
                "best_composite_score": pd.to_numeric(
                    active["best_composite_score"], errors="coerce"
                ).max(),
            }
        )
    recognition = pd.DataFrame(rows)
    result = calendar.merge(recognition, on=KEYS, how="left", validate="one_to_one")
    result["recognized"] = result["recognized"].fillna(False).astype(bool)
    result["status"] = np.where(result["recognized"], "recognized", "ignored")
    result["evidence_scope"] = np.where(
        result["day"].ge(pd.Timestamp(args.oos_start, tz="UTC")),
        "final_oos",
        "discovery_period",
    )
    result = result.sort_values(KEYS, kind="stable")
    result.to_csv(args.output / "calendar_recognized_vs_ignored.csv", index=False)
    summary = (
        result.groupby(["evidence_scope", "side_name", "archetype_policy_key"], observed=True)
        .agg(episodes=("day", "nunique"), recognized=("recognized", "sum"), recognition_rate=("recognized", "mean"))
        .reset_index()
    )
    summary.to_csv(args.output / "calendar_recognition_summary.csv", index=False)
    manifest = {
        "schema": "residual_episode_recognition_calendar_v1",
        "target": "adverse high-residual-autocorrelation calendar cells only",
        "episodes": int(len(result)),
        "recognized": int(result["recognized"].sum()),
        "ignored": int((~result["recognized"]).sum()),
        "final_oos_episodes": int(result["evidence_scope"].eq("final_oos").sum()),
        "final_oos_recognized": int(
            (result["evidence_scope"].eq("final_oos") & result["recognized"]).sum()
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--local", type=Path, default=Path("data_perp/reports/residual_episode_composite_discovery_20260712_v2_leakagesafe/episode_coverage.csv"))
    parser.add_argument("--shared", type=Path, default=Path("data_perp/reports/residual_side_global_episode_composite_discovery_20260712_v1/episode_coverage_local_cells.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/residual_episode_recognition_calendar_20260712_v1"))
    parser.add_argument("--oos-start", default="2026-04-01")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
