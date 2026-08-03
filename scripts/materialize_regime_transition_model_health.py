#!/usr/bin/env python3
"""Aggregate outcome-free old55 model-health distributions by source hour."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SHARDS = Path(
    "data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/"
    "prediction_shards"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_model_health_20260726_v1"
)
HEALTH_COLUMNS = (
    "score_base",
    "score_meta_base_soft_label",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "base_score_rank_pct_train_prior",
    "support_min_log_count",
    "support_mean_log_count",
    "support_unseen_bucket_share",
    "support_rare_bucket_share",
    "base_arch_hit_recent_rate_hl3d",
    "base_arch_hit_expected_rate_hl3d",
    "base_arch_hit_surprise_z_hl3d",
    "base_arch_hit_surprise_z_hl7d",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    columns = ["__ts__", "__symbol__", "side_name", *HEALTH_COLUMNS]
    parts = [
        pd.read_parquet(path, columns=columns)
        for path in sorted(Path(args.shards).glob("*.parquet"))
    ]
    if not parts:
        raise ValueError("no prediction shards")
    rows = pd.concat(parts, ignore_index=True)
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["__ts__", "__symbol__", "side_name"])
    rows = rows.drop_duplicates(
        ["__ts__", "__symbol__", "side_name"], keep="last"
    )
    for name in HEALTH_COLUMNS:
        rows[name] = pd.to_numeric(rows[name], errors="coerce").astype(np.float32)
    grouped = rows.groupby("__ts__", sort=True, observed=True)
    frames: list[pd.DataFrame] = [
        grouped.agg(
            health__candidate_rows=("__symbol__", "size"),
            health__distinct_assets=("__symbol__", "nunique"),
        )
    ]
    for name in HEALTH_COLUMNS:
        summary = grouped[name].agg(
            ["mean", "std", lambda values: values.quantile(0.10), lambda values: values.quantile(0.90)]
        )
        summary.columns = [
            f"health__{name}__mean",
            f"health__{name}__std",
            f"health__{name}__p10",
            f"health__{name}__p90",
        ]
        frames.append(summary)
    side = (
        rows.groupby(["__ts__", "side_name"], observed=True)[
            [
                "score_base",
                "score_meta_base_soft_label",
                "base_margin_to_cutoff_z",
                "base_arch_hit_surprise_z_hl3d",
            ]
        ]
        .mean()
        .unstack("side_name")
    )
    for name in side.columns.levels[0]:
        if (name, "long") in side and (name, "short") in side:
            frames.append(
                (
                    side[(name, "long")] - side[(name, "short")]
                ).rename(f"health__{name}__long_minus_short")
            )
    hourly = pd.concat(frames, axis=1).reset_index().rename(
        columns={"__ts__": "source_utc"}
    )
    hourly["execution_decision_utc"] = hourly["source_utc"] + pd.Timedelta(
        hours=1
    )
    hourly.to_parquet(output / "hourly_model_health.parquet", index=False)
    report = {
        "schema": "old55_outcome_free_hourly_model_health_v1",
        "lineage": "old55; not current-model parity",
        "forbidden_outcomes_read": False,
        "source_shards": len(parts),
        "source_rows": len(rows),
        "hourly_rows": len(hourly),
        "start": str(hourly["source_utc"].min()),
        "end": str(hourly["source_utc"].max()),
        "health_features": len(hourly.columns) - 2,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
