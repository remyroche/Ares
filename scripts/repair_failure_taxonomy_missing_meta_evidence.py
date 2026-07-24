#!/usr/bin/env python3
"""Remove fabricated zero-valued meta diagnostics from a frozen taxonomy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_failure_episode_taxonomy import (
    _mixture_profiles,
    _negative_day_mode_catalog,
    _semantic_failure_assignments,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _meta_evidence_columns(columns: list[str]) -> list[str]:
    return sorted(
        name
        for name in columns
        if "meta_" in name.casefold() or "base_meta_" in name.casefold()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    shutil.copytree(source, output)

    calendar_path = output / "local_adverse_calendar.parquet"
    calendar = pd.read_parquet(calendar_path)
    calendar_columns = _meta_evidence_columns(list(calendar.columns))
    for name in calendar_columns:
        calendar[name] = np.nan
    calendar.to_parquet(calendar_path, index=False)

    taxonomy_path = output / "local_failure_block_taxonomy.parquet"
    taxonomy = pd.read_parquet(taxonomy_path)
    taxonomy_columns = _meta_evidence_columns(list(taxonomy.columns))
    for name in taxonomy_columns:
        taxonomy[name] = np.nan
    taxonomy.to_parquet(taxonomy_path, index=False)

    descriptive_assignments = pd.read_parquet(
        output / "local_failure_mixture_assignments.parquet"
    )
    descriptive_profiles = _mixture_profiles(taxonomy, descriptive_assignments)
    descriptive_semantic = _semantic_failure_assignments(
        descriptive_assignments, descriptive_profiles
    )
    descriptive_profiles.to_csv(
        output / "local_failure_mixture_profiles.csv", index=False
    )
    descriptive_semantic.to_parquet(
        output / "local_failure_semantic_assignments.parquet", index=False
    )

    frozen_assignments = pd.read_parquet(
        output / "local_frozen_failure_mode_assignments.parquet"
    )
    frozen_profiles = _mixture_profiles(taxonomy, frozen_assignments)
    frozen_semantic = _semantic_failure_assignments(
        frozen_assignments, frozen_profiles
    )
    frozen_profiles.to_csv(
        output / "local_frozen_failure_mode_profiles.csv", index=False
    )
    frozen_semantic.to_parquet(
        output / "local_frozen_failure_mode_semantic_assignments.parquet",
        index=False,
    )

    daily_global = pd.read_parquet(output / "daily_global_health.parquet")
    parent_calendar = pd.read_parquet(output / "parent_adverse_calendar.parquet")
    parent_frozen_assignments = pd.read_parquet(
        output / "parent_frozen_failure_mode_assignments.parquet"
    )
    parent_frozen_profiles = pd.read_csv(
        output / "parent_frozen_failure_mode_profiles.csv"
    )
    parent_descriptive_assignments = pd.read_parquet(
        output / "parent_failure_mixture_assignments.parquet"
    )
    parent_descriptive_profiles = pd.read_csv(
        output / "parent_failure_mixture_profiles.csv"
    )
    frozen_negative_days = _negative_day_mode_catalog(
        daily_global,
        parent_calendar,
        parent_frozen_assignments,
        parent_frozen_profiles,
        calendar,
        frozen_assignments,
        frozen_profiles,
    )
    descriptive_negative_days = _negative_day_mode_catalog(
        daily_global,
        parent_calendar,
        parent_descriptive_assignments,
        parent_descriptive_profiles,
        calendar,
        descriptive_assignments,
        descriptive_profiles,
    )
    frozen_negative_days.to_csv(
        output / "negative_pnl_day_failure_modes.csv", index=False
    )
    descriptive_negative_days.to_csv(
        output / "negative_pnl_day_descriptive_failure_modes.csv", index=False
    )

    previous_frozen_labels = pd.read_parquet(
        source / "local_frozen_failure_mode_semantic_assignments.parquet"
    )
    keys = ["side_name", "archetype_policy_key", "event_block"]
    label_comparison = previous_frozen_labels.loc[
        :, [*keys, "semantic_label"]
    ].merge(
        frozen_semantic.loc[:, [*keys, "semantic_label"]],
        on=keys,
        how="outer",
        suffixes=("_before", "_after"),
        indicator=True,
    )
    changed = label_comparison.loc[
        label_comparison["_merge"].ne("both")
        | label_comparison["semantic_label_before"].ne(
            label_comparison["semantic_label_after"]
        )
    ]
    label_comparison.to_csv(output / "missing_meta_label_comparison.csv", index=False)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_stability = pd.read_csv(output / "local_failure_mode_temporal_stability.csv")
    parent_stability = pd.read_csv(output / "parent_failure_mode_temporal_stability.csv")
    local_nonredundancy = pd.read_csv(
        output / "local_failure_mixture_nonredundancy.csv"
    )
    parent_nonredundancy = pd.read_csv(
        output / "parent_failure_mixture_nonredundancy.csv"
    )
    manifest.update(
        {
            "missing_meta_evidence_contract": "unavailable_preserved_as_nan",
            "historical_meta_score_available": False,
            "sanitized_local_calendar_meta_columns": calendar_columns,
            "sanitized_local_taxonomy_meta_columns": taxonomy_columns,
            "frozen_semantic_label_changes_after_meta_missingness_repair": int(
                len(changed)
            ),
            "missing_meta_repair_source": str(source.resolve()),
            # The source manifest may predate a stability/profile refresh. Always
            # derive summary counts from the tables copied into this artifact.
            "mixture_temporal_stability_rows": int(len(local_stability)),
            "mixture_temporal_stability_warnings": int(
                local_stability.get(
                    "temporal_stability_warning", pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            ),
            "parent_temporal_stability_rows": int(len(parent_stability)),
            "parent_temporal_stability_warnings": int(
                parent_stability.get(
                    "temporal_stability_warning", pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            ),
            "mixture_nonredundancy_rows": int(len(local_nonredundancy)),
            "mixture_redundancy_warnings": int(
                local_nonredundancy.get(
                    "calendar_redundancy_warning", pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            ),
            "parent_mixture_nonredundancy_rows": int(len(parent_nonredundancy)),
            "parent_mixture_redundancy_warnings": int(
                parent_nonredundancy.get(
                    "calendar_redundancy_warning", pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, default=_json_default), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
