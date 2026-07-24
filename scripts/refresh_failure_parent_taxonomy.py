#!/usr/bin/env python3
"""Refresh capped global failure modes while reusing a frozen local taxonomy.

The full candidate ledger is intentionally not materialized.  Local failure
blocks and daily observable state are copied from a validated taxonomy run;
only four row-identity columns are streamed for the parent composition audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.residual_event_block_taxonomy import (
    BlockTaxonomyConfig,
    annotate_onset_mechanism_profiles,
    attach_event_blocks,
    block_family_profiles,
    build_block_taxonomy,
)
from extreme_price_movements.unsupervised_regime_learning.failure_taxonomy_models import (
    FailureTaxonomyModelConfig,
    failure_taxonomy_nonredundancy,
    failure_taxonomy_temporal_stability,
    fit_failure_taxonomy_models,
    fit_frozen_consensus_taxonomy,
)
from scripts.run_failure_episode_taxonomy import (
    PARENT_MAX_EVENT_DAYS,
    _failure_mode_composition_audit,
    _mixture_profiles,
    _negative_day_mode_catalog,
    _semantic_failure_assignments,
)

IDENTITY_COLUMNS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _identity_ledger(candidate_root: Path) -> pd.DataFrame:
    shards = sorted(candidate_root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No candidate shards under {candidate_root}")
    return pd.concat(
        [pd.read_parquet(path, columns=IDENTITY_COLUMNS) for path in shards],
        ignore_index=True,
        copy=False,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_taxonomy)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    shutil.copytree(source, output)

    reference_end = pd.Timestamp(args.reference_end)
    if reference_end.tzinfo is None:
        raise ValueError("--reference-end must be timezone-aware")
    reference_end = reference_end.tz_convert("UTC")
    config = FailureTaxonomyModelConfig(
        min_cluster_episodes=int(args.min_cluster_episodes)
    )

    parent_calendar = pd.read_parquet(source / "parent_adverse_calendar.parquet")
    parent_calendar = parent_calendar.drop(columns="event_block", errors="ignore")
    if "adverse_event" in parent_calendar:
        parent_calendar = parent_calendar.rename(
            columns={"adverse_event": "adverse_calendar_cell"}
        )
    parent_state = pd.read_parquet(source / "daily_parent_market_state.parquet")
    parent_taxonomy, _ = build_block_taxonomy(
        parent_calendar,
        parent_state,
        config=BlockTaxonomyConfig(max_event_days=PARENT_MAX_EVENT_DAYS),
    )
    parent_taxonomy = annotate_onset_mechanism_profiles(parent_taxonomy)
    parent_profiles = block_family_profiles(parent_taxonomy)
    parent_assignments, parent_diagnostics = fit_failure_taxonomy_models(
        parent_taxonomy,
        config=config,
    )
    parent_mixture_profiles = _mixture_profiles(
        parent_taxonomy, parent_assignments
    )
    parent_nonredundancy = failure_taxonomy_nonredundancy(
        parent_taxonomy, parent_assignments
    )
    parent_stability = failure_taxonomy_temporal_stability(
        parent_taxonomy, parent_assignments
    )
    frozen_assignments, frozen_diagnostics, frozen_state = (
        fit_frozen_consensus_taxonomy(
            parent_taxonomy,
            reference_end=reference_end,
            config=config,
        )
    )
    if frozen_assignments.empty:
        raise RuntimeError("Capped parent taxonomy still has no frozen assignments")
    frozen_profiles = _mixture_profiles(parent_taxonomy, frozen_assignments)
    frozen_semantic = _semantic_failure_assignments(
        frozen_assignments, frozen_profiles
    )
    descriptive_semantic = _semantic_failure_assignments(
        parent_assignments, parent_mixture_profiles
    )

    parent_event_calendar = attach_event_blocks(
        parent_calendar,
        max_event_days=PARENT_MAX_EVENT_DAYS,
    )
    local_calendar = pd.read_parquet(source / "local_adverse_calendar.parquet")
    local_frozen_assignments = pd.read_parquet(
        source / "local_frozen_failure_mode_assignments.parquet"
    )
    local_frozen_profiles = pd.read_csv(
        source / "local_frozen_failure_mode_profiles.csv"
    )
    local_descriptive_assignments = pd.read_parquet(
        source / "local_failure_mixture_assignments.parquet"
    )
    local_descriptive_profiles = pd.read_csv(
        source / "local_failure_mixture_profiles.csv"
    )
    local_taxonomy = pd.read_parquet(
        source / "local_failure_block_taxonomy.parquet"
    )
    local_stability = failure_taxonomy_temporal_stability(
        local_taxonomy, local_descriptive_assignments
    )
    daily_global = pd.read_parquet(source / "daily_global_health.parquet")
    frozen_negative_days = _negative_day_mode_catalog(
        daily_global,
        parent_event_calendar,
        frozen_assignments,
        frozen_profiles,
        local_calendar,
        local_frozen_assignments,
        local_frozen_profiles,
    )
    descriptive_negative_days = _negative_day_mode_catalog(
        daily_global,
        parent_event_calendar,
        parent_assignments,
        parent_mixture_profiles,
        local_calendar,
        local_descriptive_assignments,
        local_descriptive_profiles,
    )

    identity = _identity_ledger(Path(args.candidate_root))
    parent_composition = _failure_mode_composition_audit(
        identity,
        parent_event_calendar,
        parent_assignments,
        parent_mixture_profiles,
        parent_scope=True,
    )

    parent_event_calendar.to_parquet(
        output / "parent_adverse_calendar.parquet", index=False
    )
    parent_taxonomy.to_parquet(
        output / "parent_failure_block_taxonomy.parquet", index=False
    )
    parent_profiles.to_csv(output / "parent_failure_family_profiles.csv", index=False)
    parent_assignments.to_parquet(
        output / "parent_failure_mixture_assignments.parquet", index=False
    )
    parent_diagnostics.to_csv(
        output / "parent_failure_mixture_diagnostics.csv", index=False
    )
    parent_mixture_profiles.to_csv(
        output / "parent_failure_mixture_profiles.csv", index=False
    )
    parent_nonredundancy.to_csv(
        output / "parent_failure_mixture_nonredundancy.csv", index=False
    )
    parent_stability.to_csv(
        output / "parent_failure_mode_temporal_stability.csv", index=False
    )
    local_stability.to_csv(
        output / "local_failure_mode_temporal_stability.csv", index=False
    )
    descriptive_semantic.to_parquet(
        output / "parent_failure_semantic_assignments.parquet", index=False
    )
    parent_composition.to_csv(
        output / "parent_failure_mode_composition.csv", index=False
    )
    frozen_assignments.to_parquet(
        output / "parent_frozen_failure_mode_assignments.parquet", index=False
    )
    frozen_diagnostics.to_csv(
        output / "parent_frozen_failure_mode_diagnostics.csv", index=False
    )
    frozen_profiles.to_csv(
        output / "parent_frozen_failure_mode_profiles.csv", index=False
    )
    frozen_semantic.to_parquet(
        output / "parent_frozen_failure_mode_semantic_assignments.parquet",
        index=False,
    )
    (output / "parent_frozen_failure_taxonomy_state.json").write_text(
        json.dumps(frozen_state, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    frozen_negative_days.to_csv(
        output / "negative_pnl_day_failure_modes.csv", index=False
    )
    descriptive_negative_days.to_csv(
        output / "negative_pnl_day_descriptive_failure_modes.csv", index=False
    )

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "parent_taxonomy_blocks": int(len(parent_taxonomy)),
            "parent_mixture_assignment_rows": int(len(parent_assignments)),
            "parent_mixture_diagnostic_rows": int(len(parent_diagnostics)),
            "parent_mixture_nonredundancy_rows": int(len(parent_nonredundancy)),
            "parent_temporal_stability_rows": int(len(parent_stability)),
            "frozen_parent_mode_assignment_rows": int(len(frozen_assignments)),
            "frozen_parent_mode_groups": int(len(frozen_diagnostics)),
            "frozen_parent_semantic_modes": int(
                frozen_semantic.get("semantic_label", pd.Series(dtype=str)).nunique()
            ),
            "negative_day_mode_assignment_contract": "frozen_reference_prototypes",
            "descriptive_negative_day_mode_rows": int(
                len(descriptive_negative_days)
            ),
            "parent_max_event_days": PARENT_MAX_EVENT_DAYS,
            "parent_taxonomy_refresh_source": str(source.resolve()),
            "status": (
                "complete"
                if len(frozen_negative_days)
                and frozen_negative_days["parent_mode_assigned"].all()
                else "incomplete_negative_day_taxonomy_coverage"
            ),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, default=_json_default), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-taxonomy", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-end", default="2025-01-01T00:00:00Z")
    parser.add_argument("--min-cluster-episodes", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
