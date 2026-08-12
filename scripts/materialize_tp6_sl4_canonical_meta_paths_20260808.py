#!/usr/bin/env python3
"""Materialise strict canonical TP6/SL4 residual-ranker path evidence.

The canonical downstream replay deliberately discards fitted ranker objects
after scoring.  This companion materialiser refits the *same* residual head
before each held 2025 month and persists its native LightGBM leaf/path and
additive-contribution evidence for the conditional cluster layer.

Only the long side is materialised.  The fitted model sees rows whose labels
are available strictly before the held month; the persisted path partition is
always ``test``.  Raw leaf tokens remain inside each strict artifact and are
converted to token-free family contributions through the existing lineage
bridge.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_family_contributions import materialize_leaf_family_contributions
from extreme_price_movements.strict_oof_base_reasoning import (
    StrictOOFBaseReasoningConfig,
    materialize_strict_oof_base_reasoning,
)
from extreme_price_movements.structural_rule_families import (
    StructuralRuleFamilyConfig,
    cluster_structural_rule_families,
    materialize_structural_family_posteriors,
)
from scripts.run_tp6_sl4_downstream_retrain_2025 import (
    MONTHS,
    SEED,
    _load,
    _map_base,
    _rank_fit,
)


SIDE = "long"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260808_v1"
DISCOVERY_RULES_PER_MONTH = 48
DEFAULT_STRUCTURAL_THRESHOLD = 0.30
MAX_SELECTED_FAMILIES = 100


def _digest(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _residual_fields(context: list[str]) -> list[str]:
    return ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context]


def _month_frames(x: pd.DataFrame, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(month, tz="UTC")
    held = x.loc[x.month.eq(month) & x.side_name.eq(SIDE)].copy()
    train = x.loc[
        x.__ts__.lt(start)
        & x.label_available_ts.lt(start)
        & x.label_valid
        & x.side_name.eq(SIDE)
    ].copy()
    return train, held


def _materialize_month(
    *,
    train: pd.DataFrame,
    held: pd.DataFrame,
    context: list[str],
    month: str,
    out: Path,
    max_trees: int,
    contribution_components: int,
    threshold_bands: int,
) -> dict[str, object]:
    if len(train) < 300 or held.empty:
        return {"month": month, "status": "skipped", "train_rows": len(train), "held_rows": len(held)}

    train = train.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    held = held.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    train_anchor, held_anchor = _map_base(train, held)
    train = train.copy(); held = held.copy()
    train["base_anchor"] = train_anchor
    held["base_anchor"] = held_anchor
    residual = train.exact_net_bps.to_numpy(float) - train.base_anchor.to_numpy(float)
    grade = np.digitize(residual, [-100.0, -25.0, 25.0, 100.0]).astype(np.int32)
    fields = _residual_fields(context)

    result = _rank_fit(
        train,
        held,
        fields,
        grade,
        equal_month=True,
        seed=SEED + int(month[-2:]) * 1000 + 99,
        return_model=True,
    )
    train_raw, held_raw, model, train_matrix, held_matrix = result
    artifact_dir = out / "strict_base_reasoning" / f"month={month}"
    identity = held.loc[:, ["candidate_id", "__ts__", "side_name"]].copy()
    reasoning = materialize_strict_oof_base_reasoning(
        [model],
        train_matrix,
        held_matrix,
        head_name="canonical_residual",
        side_name=SIDE,
        fold_id=month,
        train_timestamps=train["__ts__"],
        eval_timestamps=held["__ts__"],
        eval_identity=identity,
        eval_predictions=held_raw,
        train_targets=grade,
        class_index=1,
        artifact_dir=artifact_dir,
        config=StrictOOFBaseReasoningConfig(
            max_trees_per_model=int(max_trees),
            rule_threshold_band_count=int(threshold_bands),
            contribution_components=int(contribution_components),
            contribution_batch_rows=20_000,
        ),
    )
    contribution_path = out / "family_contributions" / f"month={month}.parquet"
    contribution_path.parent.mkdir(parents=True, exist_ok=True)
    materialize_leaf_family_contributions(artifact_dir, contribution_path)

    evaluation = held.loc[:, [
        "candidate_id", "__ts__", "side_name", "label_available_ts",
        "exact_net_bps", "exact_gross_bps",
    ]].copy()
    evaluation["base_expected_bps"] = held_anchor.astype(np.float32)
    evaluation["meta_raw"] = np.asarray(held_raw, dtype=np.float32)
    evaluation["fold"] = month
    evaluation["meta_partition"] = "test"
    evaluation_dir = out / "fold_evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(evaluation_dir / f"month={month}.parquet", index=False, compression="zstd")

    manifest = {
        "month": month,
        "status": "MATERIALIZED_STRICT_OOF",
        "side": SIDE,
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "feature_count": int(len(fields)),
        "feature_contract": fields,
        "feature_contract_sha256": _digest(fields),
        "query": "4-hour UTC x side",
        "target": "exact TP6/SL4 net bps - train-only isotonic base anchor; ordinal thresholds [-100,-25,25,100]",
        "path_partition": "test",
        "model_layer": "meta_residual",
        "model_head": "canonical_residual",
        "artifact_dir": str(artifact_dir),
        "family_contribution_path": str(contribution_path),
        "reasoning_manifest": reasoning.manifest,
    }
    audit_dir = out / "fold_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"month={month}.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    del model, train_matrix, held_matrix, train_raw, held_raw, reasoning
    gc.collect()
    return manifest


def _catalogue_from_month(catalogue: pd.DataFrame, *, month: str) -> pd.DataFrame:
    """Convert one strict local catalogue to the cross-month contract."""
    required = {"rule_signature", "rule_structural_path_json", "train_leaf_frequency"}
    missing = sorted(required.difference(catalogue.columns))
    if missing:
        raise ValueError(f"strict catalogue misses structural fields: {missing}")
    grouped = catalogue.groupby("rule_signature", observed=True, sort=False).agg(
        rule_structural_path_json=("rule_structural_path_json", "first"),
        train_leaf_frequency=("train_leaf_frequency", "sum"),
    ).reset_index()
    grouped["rule_instance_id"] = month + "::" + grouped["rule_signature"].astype(str)
    grouped["fold_id"] = month
    grouped["model_id"] = month
    grouped["side_name"] = SIDE
    grouped["head_name"] = "canonical_residual"
    grouped["base_model_version"] = "tp6_sl4_canonical_meta_residual_v1"
    grouped["model_layer"] = "meta"
    return grouped.loc[:, [
        "rule_instance_id", "fold_id", "model_id", "side_name", "head_name",
        "base_model_version", "model_layer", "rule_signature",
        "rule_structural_path_json", "train_leaf_frequency",
    ]]


def _finalize_family_matrix(
    out: Path,
    months: tuple[str, ...],
    *,
    structural_threshold: float = DEFAULT_STRUCTURAL_THRESHOLD,
) -> dict[str, object]:
    """Discover recurrent meta-path families and emit the wide handoff."""
    catalogues: list[pd.DataFrame] = []
    activations: dict[str, pd.DataFrame] = {}
    evaluations: dict[str, pd.DataFrame] = {}
    for month in months:
        artifact = out / "strict_base_reasoning" / f"month={month}"
        catalogue_path = artifact / "leaf_rule_catalog.parquet"
        contribution_path = out / "family_contributions" / f"month={month}.parquet"
        evaluation_path = out / "fold_evaluations" / f"month={month}.parquet"
        if not all(path.exists() for path in (catalogue_path, contribution_path, evaluation_path)):
            continue
        local_catalogue = _catalogue_from_month(pd.read_parquet(catalogue_path), month=month)
        # Structural alignment is intentionally bounded.  The priority is
        # train-fold support only; no realised outcome, residual, or health
        # field participates in this shortlist.  Raw per-row activations are
        # retained below and are filtered to the discovered rule instances.
        local_catalogue = local_catalogue.sort_values(
            ["train_leaf_frequency", "rule_instance_id"],
            ascending=[False, True],
            kind="stable",
        ).head(DISCOVERY_RULES_PER_MONTH)
        catalogues.append(local_catalogue)
        raw = pd.read_parquet(contribution_path)
        raw["rule_instance_id"] = raw["fold_id"].astype(str) + "::" + raw["rule_signature"].astype(str)
        activations[month] = raw.loc[:, ["candidate_id", "rule_instance_id", "family_ensemble_tree_contribution"]]
        evaluations[month] = pd.read_parquet(evaluation_path)
    if not catalogues:
        return {"status": "not_ready", "reason": "no completed strict month artifacts"}

    catalogue = pd.concat(catalogues, ignore_index=True)
    family_result = cluster_structural_rule_families(
        catalogue,
        config=StructuralRuleFamilyConfig(
            threshold=float(structural_threshold),
            min_distinct_folds=2,
            min_distinct_models=2,
            max_rule_instances_per_cell=DISCOVERY_RULES_PER_MONTH * max(len(activations), 1),
            max_selected_families_per_cell=MAX_SELECTED_FAMILIES,
        ),
    )
    matrix_rows: list[pd.DataFrame] = []
    feature_map: dict[str, str] = {}
    for month, activation in activations.items():
        known = set(family_result.assignments["rule_instance_id"].astype(str))
        activation = activation.loc[activation["rule_instance_id"].isin(known)].copy()
        if activation.empty:
            continue
        posterior = materialize_structural_family_posteriors(
            activation,
            family_result.assignments,
            contribution_column="family_ensemble_tree_contribution",
        )
        post = posterior.features.copy()
        rename = {
            name: name.replace("base_structural_family__", "meta_structural_family__", 1)
            for name in post.columns if name.startswith("base_structural_family__")
        }
        post = post.rename(columns=rename)
        feature_map.update({cluster: name.replace("base_structural_family__", "meta_structural_family__", 1) for cluster, name in posterior.cluster_id_to_feature.items()})
        membership_fields = [name for name in post.columns if name.startswith("meta_structural_family__") and not name.endswith("unassigned_mass")]
        # Build the companion views in one concat rather than repeatedly
        # inserting columns into the posterior frame (which fragments a wide
        # 80+ family matrix and materially increases memory use).
        membership = post.loc[:, membership_fields].astype(np.float32, copy=False)
        abs_share = membership.copy()
        abs_share.columns = [f"family_abs_share__{field}" for field in membership_fields]
        confidence_share = membership.copy()
        confidence_share.columns = [f"family_confidence_share__{field}" for field in membership_fields]
        post = pd.concat([post, abs_share, confidence_share], axis=1)
        ev = evaluations[month].loc[:, ["candidate_id", "__ts__", "side_name", "fold", "meta_partition"]].copy()
        merged = ev.merge(post, on="candidate_id", how="left", validate="one_to_one")
        for field in [c for c in post.columns if c != "candidate_id"]:
            if field not in merged:
                continue
            if field.startswith(("meta_structural_family__", "family_abs_share__", "family_confidence_share__")):
                merged[field] = pd.to_numeric(merged[field], errors="coerce").fillna(0.0).astype(np.float32)
        matrix_rows.append(merged)
    matrix = pd.concat(matrix_rows, ignore_index=True) if matrix_rows else pd.DataFrame()
    matrix_path = out / "meta_family_contribution_matrix.parquet"
    matrix.to_parquet(matrix_path, index=False, compression="zstd")
    catalogue.to_parquet(out / "meta_structural_rule_catalogue.parquet", index=False, compression="zstd")
    family_result.assignments.to_parquet(out / "meta_structural_family_assignments.parquet", index=False, compression="zstd")
    family_result.family_summary.to_parquet(out / "meta_structural_family_summary.parquet", index=False, compression="zstd")
    family_result.pairwise_similarity.to_parquet(out / "meta_structural_family_similarity.parquet", index=False, compression="zstd")
    contract = {
        "schema": "tp6_sl4_canonical_meta_family_contract_v1",
        "status": "complete",
        "side": SIDE,
        "model_layer": "meta_residual",
        "head": "canonical_residual",
        "threshold": float(structural_threshold),
        "max_selected_families": int(MAX_SELECTED_FAMILIES),
        "months": sorted(activations),
        "selected_cluster_count": int(len(family_result.selected_cluster_ids)),
        "selected_cluster_ids": list(family_result.selected_cluster_ids),
        "cluster_id_to_feature": feature_map,
        "feature_prefix": "meta_structural_family__",
        "path_partition": "test",
        "matrix": str(matrix_path),
        "catalogue": str(out / "meta_structural_rule_catalogue.parquet"),
        "structural_alignment": "rule paths/signatures only; no outcomes/economics/health during clustering",
    }
    (out / "meta_family_contract.json").write_text(json.dumps(contract, indent=2, default=str) + "\n")
    return {
        "status": "complete",
        "matrix": str(matrix_path),
        "rows": int(len(matrix)),
        "family_fields": int(len([c for c in matrix.columns if c.startswith("meta_structural_family__")])),
        "selected_cluster_count": int(len(family_result.selected_cluster_ids)),
    }


def run(
    out: Path = DEFAULT_OUT,
    *,
    months: tuple[str, ...] = MONTHS,
    max_trees: int = 64,
    contribution_components: int = 8,
    threshold_bands: int = 4,
    structural_threshold: float = DEFAULT_STRUCTURAL_THRESHOLD,
    resume: bool = False,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    x, context, context_digest = _load()
    audits: list[dict[str, object]] = []
    for month in months:
        audit_path = out / "fold_audits" / f"month={month}.json"
        contribution_path = out / "family_contributions" / f"month={month}.parquet"
        evaluation_path = out / "fold_evaluations" / f"month={month}.parquet"
        if resume and all(path.exists() for path in (audit_path, contribution_path, evaluation_path)):
            audits.append(json.loads(audit_path.read_text()))
            continue
        train, held = _month_frames(x, month)
        audit = _materialize_month(
            train=train,
            held=held,
            context=context,
            month=month,
            out=out,
            max_trees=max_trees,
            contribution_components=contribution_components,
            threshold_bands=threshold_bands,
        )
        audits.append(audit)
        print(json.dumps({"month": month, "status": audit.get("status"), "train_rows": audit.get("train_rows"), "held_rows": audit.get("held_rows")}), flush=True)

    completed = [a for a in audits if a.get("status") == "MATERIALIZED_STRICT_OOF"]
    family_status = _finalize_family_matrix(out, months, structural_threshold=structural_threshold) if len(completed) == len(months) else {"status": "deferred_until_all_months_complete"}
    manifest = {
        "schema": "tp6_sl4_canonical_meta_paths_v1",
        "status": "complete" if len(completed) == len(months) else "partial",
        "side": SIDE,
        "months_requested": list(months),
        "months_completed": [str(a["month"]) for a in completed],
        "feature_contract": context,
        "feature_contract_sha256": context_digest,
        "model_layer": "meta_residual",
        "path_partition": "test",
        "query": "4-hour UTC x side",
        "target": "exact TP6/SL4 net bps - train-only isotonic base anchor; ordinal thresholds [-100,-25,25,100]",
        "max_trees_per_model": int(max_trees),
        "contribution_components": int(contribution_components),
        "threshold_bands": int(threshold_bands),
        "structural_threshold": float(structural_threshold),
        "strict_oof": True,
        "raw_leaf_tokens_persisted": "only inside per-month strict artifacts",
        "family_contribution_artifacts": [str(out / "family_contributions" / f"month={m}.parquet") for m in months if (out / "family_contributions" / f"month={m}.parquet").exists()],
        "fold_audits": audits,
        "family_materialization": family_status,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--month", action="append", dest="months", help="restrict to one or more YYYY-MM held months")
    parser.add_argument("--max-trees", type=int, default=64)
    parser.add_argument("--contribution-components", type=int, default=8)
    parser.add_argument("--threshold-bands", type=int, default=4)
    parser.add_argument("--structural-threshold", type=float, default=DEFAULT_STRUCTURAL_THRESHOLD)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    requested = tuple(args.months) if args.months else MONTHS
    print(run(args.out, months=requested, max_trees=args.max_trees, contribution_components=args.contribution_components, threshold_bands=args.threshold_bands, structural_threshold=args.structural_threshold, resume=args.resume))


if __name__ == "__main__":
    main()
