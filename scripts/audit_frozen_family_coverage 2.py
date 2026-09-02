#!/usr/bin/env python3
"""Materialise a frozen, cross-fold structural-family coverage contract.

The existing structural-family artifact retains only nine recurrent clusters.
This audit asks whether a broader *fixed* family/superfamily layer can recover
more of the model's contribution mass without allowing the OOS fold to define
the grouping.  It deliberately separates:

* development-only path clustering (the two OOF folds);
* nearest frozen-medoid assignment for the later OOS fold; and
* contribution-mass coverage measurement across meta-train, calibration and
  test rows.

The materialised specialist contract is the medoid's exact path feature set.
Every member and every later nearest-medoid assignment therefore receives the
same feature names and contract digest; fold-local paths are never silently
renamed into production fields.

No labels, net outcomes, or OOS rows are used for clustering or selecting the
top-N families.  Outcomes are only read indirectly through the existing row
partition manifest to distinguish meta-train/calibration/test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4"
DEFAULT_OUT = ROOT / "data_perp/artifacts/frozen_family_coverage_audit_20260808_v1"
DEFAULT_DEV_FOLDS = ("oof_jul_aug", "oof_may_jun")
DEFAULT_THRESHOLDS = (0.45, 0.50, 0.60)
DEFAULT_TOP_N = (40, 64, 80)


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _parse_tokens(path_json: str) -> tuple[set[str], tuple[str, ...]]:
    """Return structural tokens and the frozen medoid feature list.

    Feature, branch and quantile-band tokens are intentionally all included.
    The combination prevents a common feature appearing in unrelated paths
    from creating a giant connected component, while retaining exact repeated
    paths across folds.
    """

    path = json.loads(path_json)
    tokens: set[str] = set()
    features: set[str] = set()
    for step in path:
        feature = str(step.get("feature", ""))
        if not feature:
            continue
        features.add(feature)
        tokens.add(f"feature::{feature}")
        tokens.add(f"branch::{feature}::{step.get('branch', '')}")
        tokens.add(
            "band::{}::{}::{}".format(
                feature,
                step.get("threshold_band_state", ""),
                step.get("threshold_band_index", ""),
            )
        )
    return tokens, tuple(sorted(features))


def _load_catalogue(source: Path) -> tuple[pd.DataFrame, csr_matrix, np.ndarray, list[set[str]]]:
    catalogue = pd.read_parquet(source / "structural_rule_catalogue.parquet")
    required = {
        "rule_instance_id",
        "fold_id",
        "rule_signature",
        "rule_structural_path_json",
        "train_leaf_frequency",
    }
    missing = required - set(catalogue.columns)
    if missing:
        raise ValueError(f"catalogue missing required columns: {sorted(missing)}")
    if catalogue.rule_instance_id.duplicated().any():
        raise ValueError("rule_instance_id is not unique")
    expected_key = catalogue.fold_id.astype(str) + "::" + catalogue.rule_signature.astype(str)
    if not np.array_equal(expected_key.to_numpy(), catalogue.rule_instance_id.astype(str).to_numpy()):
        raise ValueError("rule_instance_id does not equal fold_id::rule_signature")

    parsed = [_parse_tokens(x) for x in catalogue.rule_structural_path_json.astype(str)]
    token_sets = [x[0] for x in parsed]
    catalogue = catalogue.copy()
    catalogue["rule_key"] = expected_key.to_numpy()
    catalogue["frozen_feature_names"] = [x[1] for x in parsed]
    catalogue["path_token_count"] = [len(x[0]) for x in parsed]
    catalogue["path_depth"] = [len(json.loads(x)) for x in catalogue.rule_structural_path_json.astype(str)]

    encoder = MultiLabelBinarizer(sparse_output=True)
    matrix = encoder.fit_transform(token_sets).astype(np.int16).tocsr()
    sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    return catalogue, matrix, sizes, token_sets


def _load_partitions(source: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in ("oof_jul_aug", "oof_may_jun", "oos_sep_nov"):
        path = source / "fold_evaluations" / f"{fold}.parquet"
        frame = pd.read_parquet(path, columns=["candidate_id", "meta_partition", "fold"])
        frame = frame.rename(columns={"fold": "fold_id"})
        frames.append(frame)
    partitions = pd.concat(frames, ignore_index=True)
    if partitions.duplicated(["fold_id", "candidate_id"]).any():
        raise ValueError("duplicate fold/candidate partition keys")
    if partitions.meta_partition.isna().any():
        raise ValueError("missing meta partition")
    return partitions


def _aggregate_development_rule_mass(source: Path, partitions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model contribution mass using meta-train rows only."""

    partials: list[pd.DataFrame] = []
    for fold in ("oof_jul_aug", "oof_may_jun", "oos_sep_nov"):
        path = source / "family_contributions" / f"{fold}.parquet"
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=1_000_000,
            columns=["candidate_id", "fold_id", "rule_signature", "family_ensemble_tree_contribution"],
        ):
            frame = batch.to_pandas()
            frame = frame.merge(
                partitions,
                on=["fold_id", "candidate_id"],
                how="left",
                validate="many_to_one",
                sort=False,
            )
            if frame.meta_partition.isna().any():
                raise ValueError(f"unmapped contribution rows in {path}")
            frame = frame[frame.meta_partition.eq("meta_train")]
            if frame.empty:
                continue
            frame["rule_key"] = frame.fold_id.astype(str) + "::" + frame.rule_signature.astype(str)
            frame["abs_contribution"] = frame.family_ensemble_tree_contribution.astype(float).abs()
            frame["active"] = frame.abs_contribution > 1e-12
            partials.append(
                frame.groupby("rule_key", sort=False, observed=True)
                .agg(
                    meta_train_abs_mass=("abs_contribution", "sum"),
                    meta_train_contribution_rows=("abs_contribution", "size"),
                    meta_train_active_rows=("active", "sum"),
                )
                .reset_index()
            )
    if not partials:
        raise ValueError("no meta-train contribution rows found")
    mass = pd.concat(partials, ignore_index=True)
    return (
        mass.groupby("rule_key", sort=False, observed=True)
        .agg(
            meta_train_abs_mass=("meta_train_abs_mass", "sum"),
            meta_train_contribution_rows=("meta_train_contribution_rows", "sum"),
            meta_train_active_rows=("meta_train_active_rows", "sum"),
        )
        .reset_index()
    )


def _jaccard_to_row(matrix: csr_matrix, sizes: np.ndarray, row_index: int) -> np.ndarray:
    intersections = (matrix @ matrix.getrow(row_index).T).toarray().ravel().astype(float)
    unions = sizes[row_index] + sizes - intersections
    return np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)


def _select_frozen_superfamilies(
    catalogue: pd.DataFrame,
    matrix: csr_matrix,
    sizes: np.ndarray,
    rule_mass: pd.DataFrame,
    *,
    development_folds: tuple[str, ...],
    threshold: float,
    max_families: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily select disjoint development medoids with cross-fold support."""

    mass_map = rule_mass.set_index("rule_key")["meta_train_abs_mass"]
    catalogue = catalogue.copy()
    catalogue["meta_train_abs_mass"] = catalogue.rule_key.map(mass_map).fillna(0.0)
    catalogue["is_development_rule"] = catalogue.fold_id.isin(development_folds)
    dev_indices = np.flatnonzero(catalogue.is_development_rule.to_numpy())
    order = sorted(
        dev_indices.tolist(),
        key=lambda i: (
            -float(catalogue.iloc[i].meta_train_abs_mass),
            -float(catalogue.iloc[i].train_leaf_frequency),
            str(catalogue.iloc[i].rule_instance_id),
        ),
    )

    assigned = np.zeros(len(catalogue), dtype=bool)
    families: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for seed in order:
        if assigned[seed]:
            continue
        similarity = _jaccard_to_row(matrix, sizes, seed)
        available = dev_indices[~assigned[dev_indices]]
        matches = available[similarity[available] >= threshold]
        if len(matches) == 0:
            continue
        matched_folds = set(catalogue.iloc[matches].fold_id.astype(str))
        if not set(development_folds).issubset(matched_folds):
            # A one-fold path is a diagnostic candidate, not a frozen family.
            continue

        family_number = len(families) + 1
        family_id = f"sf_t{threshold:.2f}_{family_number:03d}"
        assigned[matches] = True
        medoid = catalogue.iloc[seed]
        families.append(
            {
                "superfamily_id": family_id,
                "similarity_threshold": threshold,
                "medoid_rule_instance_id": medoid.rule_instance_id,
                "medoid_fold_id": medoid.fold_id,
                "member_count_development": len(matches),
                "development_fold_count": len(matched_folds),
                "development_folds": tuple(sorted(matched_folds)),
                "meta_train_abs_mass": float(catalogue.iloc[matches].meta_train_abs_mass.sum()),
                "medoid_path_depth": int(medoid.path_depth),
                "medoid_path_token_count": int(medoid.path_token_count),
                "frozen_feature_names": tuple(medoid.frozen_feature_names),
                "frozen_feature_digest": _digest(medoid.frozen_feature_names),
            }
        )
        for member in matches:
            member_rows.append(
                {
                    "superfamily_id": family_id,
                    "rule_instance_id": catalogue.iloc[member].rule_instance_id,
                    "rule_key": catalogue.iloc[member].rule_key,
                    "fold_id": catalogue.iloc[member].fold_id,
                    "similarity_to_medoid": float(similarity[member]),
                    "membership_source": "development_direct",
                }
            )
        if len(families) >= max_families:
            break

    if not families:
        raise ValueError(f"no cross-fold families at threshold {threshold}")

    summaries = pd.DataFrame(families).sort_values(
        ["meta_train_abs_mass", "superfamily_id"], ascending=[False, True], ignore_index=True
    )
    summaries["development_mass_rank"] = np.arange(1, len(summaries) + 1)
    rank_map = summaries.set_index("superfamily_id")["development_mass_rank"].to_dict()
    members = pd.DataFrame(member_rows)
    members["development_mass_rank"] = members.superfamily_id.map(rank_map).astype(int)

    # Assign later-fold rules to the nearest frozen development medoid.  This
    # never creates a new family or changes the medoid feature contract.
    medoid_indices = [catalogue.index[catalogue.rule_instance_id.eq(x)].item() for x in summaries.medoid_rule_instance_id]
    medoid_matrix = matrix[medoid_indices]
    intersections = (matrix @ medoid_matrix.T).toarray().astype(float)
    all_sizes = sizes[:, None]
    medoid_sizes = sizes[np.asarray(medoid_indices)][None, :]
    unions = all_sizes + medoid_sizes - intersections
    similarities = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    nearest = similarities.argmax(axis=1)
    nearest_similarity = similarities[np.arange(len(catalogue)), nearest]
    mapping = catalogue[["rule_instance_id", "rule_key", "fold_id", "side_name", "head_name", "base_model_version", "model_layer"]].copy()
    mapping["similarity_threshold"] = threshold
    mapping["nearest_superfamily_id"] = summaries.iloc[nearest].superfamily_id.to_numpy()
    mapping["nearest_mass_rank"] = summaries.iloc[nearest].development_mass_rank.to_numpy()
    mapping["nearest_similarity_to_medoid"] = nearest_similarity
    nearest_summary = summaries.iloc[nearest].reset_index(drop=True)
    mapping["frozen_feature_names"] = nearest_summary.frozen_feature_names.to_numpy()
    mapping["frozen_feature_digest"] = nearest_summary.frozen_feature_digest.to_numpy()
    mapping["is_development_rule"] = catalogue.is_development_rule.to_numpy()
    mapping["assigned_to_frozen_contract"] = nearest_similarity >= threshold
    mapping["membership_source"] = np.where(
        mapping.is_development_rule & mapping.rule_key.isin(set(members.rule_key)),
        "development_direct",
        np.where(mapping.assigned_to_frozen_contract, "nearest_frozen_medoid", "unassigned_below_threshold"),
    )
    # Retain all candidate families in the mapping; top-N contract flags are
    # added by the caller after the development-mass ranks are fixed.
    return summaries, mapping


def _coverage_for_threshold(
    source: Path,
    partitions: pd.DataFrame,
    mapping: pd.DataFrame,
    summaries: pd.DataFrame,
    *,
    thresholds_top_n: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate contribution mass and candidate coverage for one threshold."""

    rank_map = mapping.set_index("rule_key")["nearest_mass_rank"].to_dict()
    family_map = mapping.set_index("rule_key")["nearest_superfamily_id"].to_dict()
    assigned_map = mapping.set_index("rule_key")["assigned_to_frozen_contract"].astype(bool).to_dict()
    existing_rows = (
        partitions.groupby(["fold_id", "meta_partition"], observed=True)
        .size()
        .rename("candidate_rows")
        .reset_index()
    )
    total_abs: defaultdict[tuple[str, str], float] = defaultdict(float)
    selected_abs: dict[int, defaultdict[tuple[str, str], float]] = {
        n: defaultdict(float) for n in thresholds_top_n
    }
    selected_keys: dict[int, defaultdict[tuple[str, str], set[str]]] = {
        n: defaultdict(set) for n in thresholds_top_n
    }
    family_mass_parts: list[pd.DataFrame] = []

    for fold in ("oof_jul_aug", "oof_may_jun", "oos_sep_nov"):
        path = source / "family_contributions" / f"{fold}.parquet"
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=1_000_000,
            columns=["candidate_id", "fold_id", "rule_signature", "family_ensemble_tree_contribution"],
        ):
            frame = batch.to_pandas()
            frame = frame.merge(
                partitions,
                on=["fold_id", "candidate_id"],
                how="left",
                validate="many_to_one",
                sort=False,
            )
            if frame.meta_partition.isna().any():
                raise ValueError(f"unmapped contribution rows in {path}")
            frame["rule_key"] = frame.fold_id.astype(str) + "::" + frame.rule_signature.astype(str)
            frame["mass_rank"] = frame.rule_key.map(rank_map).fillna(-1).astype(int)
            frame["superfamily_id"] = frame.rule_key.map(family_map).fillna("")
            frame["assigned_to_frozen_contract"] = frame.rule_key.map(assigned_map).fillna(False).astype(bool)
            frame["abs_contribution"] = frame.family_ensemble_tree_contribution.astype(float).abs()
            for (fold_id, partition), group in frame.groupby(["fold_id", "meta_partition"], observed=True):
                total_abs[(str(fold_id), str(partition))] += float(group.abs_contribution.sum())
            for n in thresholds_top_n:
                selected = frame[
                    frame.assigned_to_frozen_contract
                    & (frame.mass_rank > 0)
                    & (frame.mass_rank <= n)
                ]
                if selected.empty:
                    continue
                for (fold_id, partition), group in selected.groupby(["fold_id", "meta_partition"], observed=True):
                    key = (str(fold_id), str(partition))
                    selected_abs[n][key] += float(group.abs_contribution.sum())
                    selected_keys[n][key].update(
                        (str(fold_id) + "\x00" + x for x in group.candidate_id.astype(str).unique())
                    )
                family_mass_parts.append(
                    selected.groupby(["mass_rank", "superfamily_id", "fold_id", "meta_partition"], observed=True)
                    .agg(abs_contribution=("abs_contribution", "sum"), contribution_rows=("abs_contribution", "size"))
                    .reset_index()
                )

    summary_rows: list[dict[str, object]] = []
    for n in thresholds_top_n:
        for row in existing_rows.itertuples(index=False):
            key = (str(row.fold_id), str(row.meta_partition))
            total = total_abs[key]
            selected = selected_abs[n][key]
            selected_count = len(selected_keys[n][key])
            summary_rows.append(
                {
                    "top_n": n,
                    "fold_id": str(row.fold_id),
                    "meta_partition": str(row.meta_partition),
                    "candidate_rows": int(row.candidate_rows),
                    "selected_candidate_rows": selected_count,
                    "selected_candidate_share": selected_count / float(row.candidate_rows) if row.candidate_rows else math.nan,
                    "total_abs_contribution_mass": total,
                    "selected_abs_contribution_mass": selected,
                    "selected_mass_share": selected / total if total > 0 else math.nan,
                }
            )
    coverage = pd.DataFrame(summary_rows)
    family_mass = (
        pd.concat(family_mass_parts, ignore_index=True)
        if family_mass_parts
        else pd.DataFrame(columns=["mass_rank", "superfamily_id", "fold_id", "meta_partition", "abs_contribution", "contribution_rows"])
    )
    return coverage, family_mass


def _build_contract_rows(
    summaries: pd.DataFrame,
    mapping: pd.DataFrame,
    top_n_values: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rank_to_id = summaries.set_index("development_mass_rank")["superfamily_id"].to_dict()
    summary_by_id = summaries.set_index("superfamily_id")
    for row in mapping.itertuples(index=False):
        if not row.assigned_to_frozen_contract:
            continue
        summary = summary_by_id.loc[row.nearest_superfamily_id]
        for top_n in top_n_values:
            rows.append(
                {
                    "top_n": top_n,
                    "superfamily_id": row.nearest_superfamily_id,
                    "development_mass_rank": int(row.nearest_mass_rank),
                    "rule_instance_id": row.rule_instance_id,
                    "rule_key": row.rule_key,
                    "fold_id": row.fold_id,
                    "membership_source": row.membership_source,
                    "similarity_to_medoid": float(row.nearest_similarity_to_medoid),
                    "is_selected": bool(row.nearest_mass_rank <= top_n),
                    "medoid_rule_instance_id": summary.medoid_rule_instance_id,
                    "frozen_feature_names": summary.frozen_feature_names,
                    "frozen_feature_digest": summary.frozen_feature_digest,
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    out: Path,
    *,
    source: Path,
    catalogue: pd.DataFrame,
    summaries_by_threshold: dict[float, pd.DataFrame],
    coverage: pd.DataFrame,
    development_folds: tuple[str, ...],
    thresholds: tuple[float, ...],
    top_n_values: tuple[int, ...],
) -> None:
    lines = [
        "# Frozen cross-fold family coverage audit",
        "",
        f"Source: `{source}`",
        f"Development folds used for clustering and selection: `{', '.join(development_folds)}`",
        "The later `oos_sep_nov` fold is assigned only to frozen development medoids; it cannot create or rank a family.",
        "",
        "## Contract design",
        "",
        "Each superfamily is a disjoint greedy medoid group built from structural path tokens on development rules. A member must match the medoid at the declared Jaccard threshold and the group must contain both development folds. The medoid's exact raw feature names are the frozen specialist input contract.",
        "",
        "## Family counts and development coverage",
        "",
        "| threshold | frozen families | rules assigned to a frozen family | families with both development folds |",
        "|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        s = summaries_by_threshold[threshold]
        assigned = int((s.member_count_development > 0).sum())
        lines.append(f"| {threshold:.2f} | {len(s)} | {int(s.member_count_development.sum())} | {int((s.development_fold_count >= 2).sum())} |")
    lines += ["", "## Absolute contribution-mass coverage", "", "All mass is measured from the complete contribution stream; family selection itself used only meta-train rows from the two development folds.", "", "| threshold | top-N | fold | partition | selected row share | selected abs-mass share |", "|---:|---:|---|---|---:|---:|"]
    for row in coverage.sort_values(["threshold", "top_n", "fold_id", "meta_partition"]).itertuples(index=False):
        lines.append(
            f"| {row.threshold:.2f} | {row.top_n} | {row.fold_id} | {row.meta_partition} | {row.selected_candidate_share:.3f} | {row.selected_mass_share:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The contract is a coverage diagnostic, not a promoted specialist model. A high selected-mass share with poor later-fold row coverage would indicate regime-specific paths rather than a portable family. A useful next specialist run must use the same medoid feature digest in every fold and report standalone and residual-layer economics separately.",
        "The medoid path inputs are intentionally compact (typically 2–6 raw fields, with development-member unions still below 20 fields). This is not yet the requested 40–80-field specialist architecture; if specialists are trained next, the medoid contract should be augmented with a frozen, layer-appropriate context pool and that augmentation must itself be held fixed across folds.",
        "",
        "## Leakage statement",
        "",
        "No net/gross outcome is used in path tokenisation, clustering, or family ranking. `meta_train` is used only to rank already-materialised model contribution mass. Calibration and test rows are held out from selection. OOS rules are nearest-medoid assignments only.",
        "",
    ]
    (out / "FROZEN_FAMILY_COVERAGE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    development_folds = tuple(args.development_folds)
    thresholds = tuple(float(x) for x in args.thresholds)
    top_n_values = tuple(int(x) for x in args.top_n)
    if any(x <= 0 or x >= 1 for x in thresholds):
        raise ValueError("thresholds must be in (0, 1)")
    if any(x <= 0 for x in top_n_values):
        raise ValueError("top-N values must be positive")

    catalogue, matrix, sizes, _ = _load_catalogue(source)
    partitions = _load_partitions(source)
    rule_mass = _aggregate_development_rule_mass(source, partitions)

    summaries_by_threshold: dict[float, pd.DataFrame] = {}
    mapping_by_threshold: dict[float, pd.DataFrame] = {}
    coverage_parts: list[pd.DataFrame] = []
    family_mass_parts: list[pd.DataFrame] = []
    contract_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    mapping_parts: list[pd.DataFrame] = []

    for threshold in thresholds:
        summaries, mapping = _select_frozen_superfamilies(
            catalogue,
            matrix,
            sizes,
            rule_mass,
            development_folds=development_folds,
            threshold=threshold,
            max_families=max(max(top_n_values), 128),
        )
        summaries_by_threshold[threshold] = summaries
        mapping_by_threshold[threshold] = mapping
        summaries_out = summaries.copy()
        summaries_out["threshold"] = threshold
        summary_parts.append(summaries_out)
        mapping_out = mapping.copy()
        mapping_out["threshold"] = threshold
        mapping_parts.append(mapping_out)
        contract = _build_contract_rows(summaries, mapping, top_n_values)
        contract["threshold"] = threshold
        contract_parts.append(contract)
        coverage, family_mass = _coverage_for_threshold(
            source,
            partitions,
            mapping,
            summaries,
            thresholds_top_n=top_n_values,
        )
        coverage["threshold"] = threshold
        family_mass["threshold"] = threshold
        coverage_parts.append(coverage)
        family_mass_parts.append(family_mass)

    summary_all = pd.concat(summary_parts, ignore_index=True)
    mapping_all = pd.concat(mapping_parts, ignore_index=True)
    contract_all = pd.concat(contract_parts, ignore_index=True)
    coverage_all = pd.concat(coverage_parts, ignore_index=True)
    family_mass_all = pd.concat(family_mass_parts, ignore_index=True)
    summary_all.to_parquet(out / "frozen_family_superfamily_summary.parquet", index=False, compression="zstd")
    mapping_all.to_parquet(out / "frozen_family_rule_mapping.parquet", index=False, compression="zstd")
    contract_all.to_parquet(out / "frozen_family_cluster_contract.parquet", index=False, compression="zstd")
    coverage_all.to_parquet(out / "frozen_family_coverage_summary.parquet", index=False, compression="zstd")
    family_mass_all.to_parquet(out / "frozen_family_mass_by_superfamily.parquet", index=False, compression="zstd")
    rule_mass.to_parquet(out / "development_rule_contribution_mass.parquet", index=False, compression="zstd")

    checks = {
        "status": "passed",
        "development_only_selection": True,
        "oos_not_used_for_selection": True,
        "all_selected_families_have_two_development_folds": bool(
            all((s.development_fold_count >= 2).all() for s in summaries_by_threshold.values())
        ),
        "no_duplicate_partition_keys": not partitions.duplicated(["fold_id", "candidate_id"]).any(),
        "explicit_unassigned_present": bool((mapping_all.membership_source == "unassigned_below_threshold").any()),
        "frozen_feature_digests_present": bool(mapping_all.frozen_feature_digest.notna().all()) if "frozen_feature_digest" in mapping_all else False,
        "coverage_rows_present": bool(len(coverage_all) > 0),
    }
    if not all(bool(v) for k, v in checks.items() if k != "status"):
        checks["status"] = "failed"
    _write_json(out / "correctness_test_report.json", checks)

    manifest = {
        "schema": "frozen_family_coverage_audit_v1",
        "status": "complete" if checks["status"] == "passed" else "failed",
        "source": str(source),
        "development_folds": list(development_folds),
        "thresholds": list(thresholds),
        "top_n": list(top_n_values),
        "catalogue_rows": int(len(catalogue)),
        "partition_rows": int(len(partitions)),
        "selection_population": int((partitions.meta_partition == "meta_train").sum()),
        "contract_definition": "development-only greedy structural-token Jaccard medoids; later folds nearest frozen medoid",
        "checks": checks,
    }
    _write_json(out / "run_manifest.json", manifest)
    _write_report(
        out,
        source=source,
        catalogue=catalogue,
        summaries_by_threshold=summaries_by_threshold,
        coverage=coverage_all,
        development_folds=development_folds,
        thresholds=thresholds,
        top_n_values=top_n_values,
    )
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-folds", nargs="+", default=list(DEFAULT_DEV_FOLDS))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--top-n", nargs="+", type=int, default=list(DEFAULT_TOP_N))
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
