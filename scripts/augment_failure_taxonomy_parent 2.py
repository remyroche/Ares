#!/usr/bin/env python3
"""Augment a completed local failure taxonomy with detector/parent artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.residual_event_block_taxonomy import (
    annotate_onset_mechanism_profiles,
    attach_event_blocks,
    block_family_profiles,
    build_block_taxonomy,
)
from extreme_price_movements.unsupervised_regime_learning.failure_taxonomy_models import (
    FailureTaxonomyModelConfig,
    fit_failure_taxonomy_models,
)
from scripts.run_failure_episode_taxonomy import _mixture_profiles


def _local_calendar(root: Path) -> pd.DataFrame:
    cells = pd.read_parquet(root / "daily_side_archetype_health.parquet")
    events = pd.read_parquet(root / "local_failure_episodes.parquet")
    membership = pd.read_parquet(root / "local_failure_membership.parquet")
    cells["day"] = pd.to_datetime(cells["day"], utc=True).dt.floor("D")
    membership["day"] = pd.to_datetime(membership["day"], utc=True).dt.floor("D")
    adverse_ids = set(
        events.loc[
            events.get("event_class", pd.Series("", index=events.index)).isin(
                ["adverse", "payoff_disagreement"]
            ),
            "event_id",
        ].astype(str)
    )
    flags = membership.assign(
        adverse_calendar_cell=membership["event_id"].astype(str).isin(adverse_ids)
    )
    flags = flags.groupby(
        ["day", "side_name", "archetype_policy_key"], observed=True, as_index=False
    )["adverse_calendar_cell"].max()
    calendar = cells.merge(
        flags, on=["day", "side_name", "archetype_policy_key"], how="left"
    )
    calendar["adverse_calendar_cell"] = calendar["adverse_calendar_cell"].fillna(False)
    return calendar.rename(
        columns={
            "mean_ev_after_cost": "mean_ev_after_1pct",
            "clean_rate": "clean_exec_rate",
            "signed_hit_surprise": "signed_surprise",
        }
    )


def _parent_calendar(root: Path) -> pd.DataFrame:
    daily = pd.read_parquet(root / "daily_global_health.parquet")
    membership = pd.read_parquet(root / "parent_failure_membership.parquet")
    daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.floor("D")
    active = set(pd.to_datetime(membership["day"], utc=True).dt.floor("D"))
    return pd.DataFrame(
        {
            "day": daily["day"],
            "side_name": "global",
            "archetype_policy_key": "global_market",
            "adverse_calendar_cell": daily["day"].isin(active),
            "negative_pnl_day": daily["negative_pnl_day"],
            "selected_rows": daily["selected_rows"],
            "mean_ev_after_1pct": daily["mean_ev"],
            "signed_surprise": daily["expost__signed_residual"],
        }
    )


def _parent_state(root: Path) -> pd.DataFrame:
    daily = pd.read_parquet(root / "daily_observable_state.parquet")
    features = [
        name for name in daily if name not in {"day", "side_name", "archetype_policy_key"}
    ]
    result = daily.groupby("day", observed=True, as_index=False)[features].median(numeric_only=True)
    result["side_name"] = "global"
    result["archetype_policy_key"] = "global_market"
    return result.loc[:, ["day", "side_name", "archetype_policy_key", *features]]


def run(root: Path, *, min_cluster_episodes: int) -> dict[str, object]:
    local_calendar = attach_event_blocks(_local_calendar(root))
    parent_calendar_raw = _parent_calendar(root)
    parent_calendar = attach_event_blocks(parent_calendar_raw)
    parent_state = _parent_state(root)
    taxonomy, _ = build_block_taxonomy(parent_calendar_raw, parent_state)
    taxonomy = annotate_onset_mechanism_profiles(taxonomy)
    profiles = block_family_profiles(taxonomy)
    assignments, diagnostics = fit_failure_taxonomy_models(
        taxonomy,
        config=FailureTaxonomyModelConfig(
            min_cluster_episodes=int(min_cluster_episodes),
        ),
    )
    mixture_profiles = _mixture_profiles(taxonomy, assignments)
    local_calendar.to_parquet(root / "local_adverse_calendar.parquet", index=False)
    parent_calendar.to_parquet(root / "parent_adverse_calendar.parquet", index=False)
    parent_state.to_parquet(root / "daily_parent_market_state.parquet", index=False)
    taxonomy.to_parquet(root / "parent_failure_block_taxonomy.parquet", index=False)
    profiles.to_csv(root / "parent_failure_family_profiles.csv", index=False)
    assignments.to_parquet(root / "parent_failure_mixture_assignments.parquet", index=False)
    diagnostics.to_csv(root / "parent_failure_mixture_diagnostics.csv", index=False)
    mixture_profiles.to_csv(root / "parent_failure_mixture_profiles.csv", index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "local_adverse_calendar_rows": int(len(local_calendar)),
            "parent_taxonomy_blocks": int(len(taxonomy)),
            "parent_mixture_assignment_rows": int(len(assignments)),
            "parent_mixture_diagnostic_rows": int(len(diagnostics)),
            "parent_mixture_profile_rows": int(len(mixture_profiles)),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-cluster-episodes", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.root, min_cluster_episodes=args.min_cluster_episodes), indent=2))
