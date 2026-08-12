#!/usr/bin/env python3
"""Materialise a frozen semantic family contract from the cross-fold audit.

The audit discovers structural-path medoids on the two development folds and
assigns the later fold only to those frozen medoids.  This utility freezes one
predeclared threshold/top-N point as a reusable family contract for the
conditional-correctness replay.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    audit = Path(args.audit_root)
    source = Path(args.source_root)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    mapping_path = audit / "frozen_family_rule_mapping.parquet"
    mapping = pd.read_parquet(mapping_path)
    keep = mapping.loc[
        mapping["threshold"].eq(float(args.threshold))
        & mapping["nearest_mass_rank"].le(int(args.top_n))
    ].copy()
    if not args.allow_nearest_fallback:
        keep = keep.loc[keep["assigned_to_frozen_contract"].astype(bool)].copy()
    if keep.empty:
        raise ValueError("the requested threshold/top-N contract has no selected rules")
    if keep["rule_instance_id"].duplicated().any():
        raise ValueError("selected semantic contract has duplicate rule instances")

    summaries = pd.read_parquet(audit / "frozen_family_superfamily_summary.parquet")
    summaries = summaries.loc[
        summaries["threshold"].eq(float(args.threshold))
        & summaries["development_mass_rank"].le(int(args.top_n))
    ].copy()
    medoid_by_family = summaries.set_index("superfamily_id")["medoid_rule_instance_id"].to_dict()
    catalogue = pd.read_parquet(source / "structural_rule_catalogue.parquet").set_index("rule_instance_id")

    # The replay expects one stable cluster ID per selected rule instance.
    assignments = pd.DataFrame(
        {
            "rule_instance_id": keep["rule_instance_id"].astype(str),
            "cluster_id": keep["nearest_superfamily_id"].astype(str),
            "side_name": keep["rule_instance_id"].map(catalogue["side_name"]).astype(str),
            "head_name": keep["rule_instance_id"].map(catalogue["head_name"]).astype(str),
            "base_model_version": keep["rule_instance_id"].map(catalogue["base_model_version"]).astype(str),
            "model_layer": keep["rule_instance_id"].map(catalogue["model_layer"]).astype(str),
            "is_medoid": keep["rule_instance_id"].eq(
                keep["nearest_superfamily_id"].map(medoid_by_family)
            ),
            "similarity_to_medoid": keep["nearest_similarity_to_medoid"].astype(float),
            "assignment_low_confidence": keep["nearest_similarity_to_medoid"].astype(float) < float(args.threshold),
            "is_recurrent": True,
            "is_selected": True,
        }
    )
    if assignments[["side_name", "head_name"]].isna().any().any():
        raise ValueError("selected contract contains rules missing from the source catalogue")

    # Avoid copying large contribution streams; the frozen contract references
    # the immutable source stream through a symlink.
    for name in ("structural_rule_catalogue.parquet",):
        shutil.copy2(source / name, out / name)
    link = out / "family_contributions"
    if not link.exists():
        link.symlink_to(source / "family_contributions", target_is_directory=True)
    assignments.to_parquet(out / "structural_family_assignments.parquet", index=False, compression="zstd")

    summaries.to_parquet(out / "structural_family_summary.parquet", index=False, compression="zstd")

    # Stable field names are ordered by development mass rank, not by the
    # order in which parquet rows happen to be stored.
    summaries = summaries.sort_values(["development_mass_rank", "superfamily_id"])
    cluster_feature_map = {
        str(row.superfamily_id): f"base_structural_family__{row.superfamily_id}"
        for row in summaries.itertuples(index=False)
    }
    # Recompute the mass contract directly from the contribution stream.  This
    # includes nearest-medoid fallback rules when requested and avoids relying
    # on a coverage table produced with a different eligibility convention.
    selected_keys = set(assignments["rule_instance_id"].astype(str))
    mass_rows = []
    for path in sorted((source / "family_contributions").glob("*.parquet")):
        total = selected = 0.0
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["fold_id", "side_name", "head_name", "rule_signature", "family_ensemble_tree_contribution"],
            batch_size=1_000_000,
        ):
            frame = batch.to_pandas()
            frame = frame.loc[
                (frame["side_name"].astype(str).str.lower() == "long")
                & frame["head_name"].astype(str).eq("p_clear")
            ]
            if frame.empty:
                continue
            key = frame["fold_id"].astype(str) + "::" + frame["rule_signature"].astype(str)
            abs_value = frame["family_ensemble_tree_contribution"].astype(float).abs()
            total += float(abs_value.sum())
            selected += float(abs_value[key.isin(selected_keys)].sum())
        mass_rows.append({"fold_id": path.stem, "total_abs_mass": total, "selected_abs_mass": selected, "selected_mass_share": selected / total if total > 0 else 0.0})
    mass = pd.DataFrame(mass_rows)
    if mass.empty or float(mass["selected_mass_share"].min()) < 0.80:
        raise AssertionError(
            f"semantic contract mass floor is {float(mass['selected_mass_share'].min() if len(mass) else 0.0):.4f}, below 0.80"
        )

    manifest = {
        "schema": "long_structural_family_semantic_medoid_v1",
        "status": "complete",
        "source_root": str(source),
        "audit_root": str(audit),
        "threshold": float(args.threshold),
        "top_n": int(args.top_n),
        "family_count": int(len(cluster_feature_map)),
        "assignment_rule_count": int(len(assignments)),
        "allow_nearest_fallback": bool(args.allow_nearest_fallback),
        "low_confidence_assignment_rate": float(assignments["assignment_low_confidence"].mean()),
        "cluster_feature_map": cluster_feature_map,
        "contract_definition": "development-only structural-token Jaccard medoids; later folds nearest frozen medoid",
        "mass_contract": "selected development-ranked medoid families; no OOS outcomes used for selection",
        "test_mass_by_fold": mass[["fold_id", "selected_mass_share"]].to_dict("records"),
        "test_mass_floor": float(mass["selected_mass_share"].min()),
        "test_mass_mean": float(mass["selected_mass_share"].mean()),
        "test_mass_gate_80pct": True,
    }
    _write_json(out / "run_manifest.json", manifest)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--allow-nearest-fallback", action="store_true", help="assign all rules to the nearest frozen family; expose low-confidence assignments")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
