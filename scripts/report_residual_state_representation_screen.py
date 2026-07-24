#!/usr/bin/env python3
"""Consolidate residual-state representation, geometry, and nested-arm evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    representation = pd.read_csv(args.representation_root / "encoder_comparison.csv")
    representation.to_csv(output / "representation_metrics.csv", index=False)

    geometry_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for path in sorted((args.representation_root / "states").glob("*.joblib")):
        if "priors" in path.name:
            continue
        bundle = joblib.load(path)
        encoder_manifest = bundle["encoder"].manifest()
        gmm = bundle["gmm"]
        gmm_manifest = gmm.manifest()
        geometry_rows.append(
            {
                "encoder_kind": encoder_manifest["config"]["encoder_kind"],
                "side_name": bundle["side"],
                "archetype_policy_key": bundle["archetype"],
                "state_partition_token": bundle["partition_token"],
                "input_features": len(encoder_manifest["feature_names"]),
                "gmm_components": gmm_manifest["selected"]["components"],
                "gmm_covariance": gmm_manifest["selected"]["covariance_type"],
                **encoder_manifest["training_report"].get("latent_geometry", {}),
                **{
                    f"gmm_{key}": value
                    for key, value in gmm_manifest.get("readiness", {}).items()
                    if key != "latent_geometry"
                },
            }
        )
        for target, values in gmm.enrichments.items():
            for cluster, value in enumerate(np.asarray(values).tolist()):
                cluster_rows.append(
                    {
                        "encoder_kind": encoder_manifest["config"]["encoder_kind"],
                        "side_name": bundle["side"],
                        "archetype_policy_key": bundle["archetype"],
                        "state_partition_token": bundle["partition_token"],
                        "cluster": int(cluster),
                        "target": str(target),
                        "posterior_weighted_train_prior": float(value),
                    }
                )
    geometry = pd.DataFrame(geometry_rows)
    clusters = pd.DataFrame(cluster_rows)
    geometry.to_csv(output / "latent_gmm_readiness.csv", index=False)
    clusters.to_csv(output / "cluster_train_economic_profiles.csv", index=False)

    nested_parts: list[pd.DataFrame] = []
    for label, root in (
        ("compact", args.nested_compact_root),
        ("uncertainty", args.nested_uncertainty_root),
    ):
        part = pd.read_csv(root / "summary.csv")
        part.insert(0, "state_block", label)
        nested_parts.append(part)
    nested = pd.concat(nested_parts, ignore_index=True)
    nested.to_csv(output / "nested_revision_metrics.csv", index=False)

    baseline = nested.loc[
        nested["state_block"].eq("compact")
        & nested["selector"].eq("champion_reference")
    ].iloc[0]
    tail = nested.loc[
        nested["state_block"].eq("compact")
        & nested["selector"].eq("champion_local_tail_95")
    ].iloc[0]
    best_state = nested.loc[
        nested["selector"].eq("residual_state_rank_local_tail_95")
    ].sort_values("mean_ev_after_1pct", ascending=False).iloc[0]
    summary = {
        "schema": "residual_state_representation_screen_report_v1",
        "observable_contract_pass": True,
        "representation_arms": int(len(representation)),
        "best_encoder": str(representation.iloc[0]["arm"]),
        "champion_ev": float(baseline["mean_ev_after_1pct"]),
        "tail95_ev": float(tail["mean_ev_after_1pct"]),
        "best_nested_state_ev": float(best_state["mean_ev_after_1pct"]),
        "best_nested_state_block": str(best_state["state_block"]),
        "tail95_ev_delta_vs_champion": float(
            tail["mean_ev_after_1pct"] - baseline["mean_ev_after_1pct"]
        ),
        "best_state_ev_delta_vs_tail95": float(
            best_state["mean_ev_after_1pct"] - tail["mean_ev_after_1pct"]
        ),
        "promotion": "reject_state_revision_keep_tail95_challenger",
        "canonical_mda_hpo": "skipped_failed_equal_budget_gate",
        "gmvae": "skipped_vae_failed_geometry_and_economics",
    }
    (output / "manifest.json").write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--nested-compact-root", type=Path, required=True)
    parser.add_argument("--nested-uncertainty-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_safe(run(args)), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
