#!/usr/bin/env python3
"""Materialise strict base-tree explanations for the long-side meta ablation.

The base ranker is fitted only before each scored partition.  Tree rule
alignment is then structural-only; historical correctness, portability, and
AE/GMM compatibility are materialised in a separate strictly prior-resolved
pass.  The output is a token-free candidate sidecar for later residual arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_family_contributions import materialize_leaf_family_contributions
from extreme_price_movements.structural_family_health import build_structural_family_historical_health
from extreme_price_movements.structural_rule_families import (
    StructuralRuleFamilyConfig,
    cluster_structural_rule_families,
    materialize_structural_family_posteriors,
)
from extreme_price_movements.strict_oof_base_reasoning import (
    StrictOOFBaseReasoningConfig,
    materialize_strict_oof_base_reasoning,
)
from scripts.run_long_only_executable_net_lambdarank import _fit_bps_map, _fit_ranker, _folds, _predict_ranker


SIDE = "long"
BASE_VERSION = "long_barrier_rank_aegmm_v1"


def _digest(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _columns(path: Path) -> list[str]:
    return list(map(str, pq.ParquetFile(path).schema.names))


def _model_columns(path: Path) -> list[str]:
    """Return DuckDB-addressable, case-insensitively unique model fields.

    The frozen AE/GMM store has one legacy pair which differs only by case and
    contains identical values.  DuckDB resolves quoted identifiers
    case-insensitively, so projecting both would silently create an ambiguous
    pandas contract.  Keep its first canonical spelling; this removes a true
    duplicate rather than a signal.
    """
    result: list[str] = []
    seen: set[str] = set()
    for name in _columns(path):
        if name == "candidate_id" or name.casefold() in seen:
            continue
        result.append(name)
        seen.add(name.casefold())
    return result


def _feature_contract(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    raw = _model_columns(args.raw_features)
    ae = _model_columns(args.aegmm)
    fields = [*raw, *ae]
    if len(fields) != len(set(fields)) or len(fields) < 30:
        raise ValueError("raw/AE-GMM contract is unexpectedly small or duplicated")
    return fields, ae


def _population_clause(*, lower: pd.Timestamp | None, upper: pd.Timestamp | None) -> tuple[str, list[object]]:
    clauses = [
        "lower(l.side_name) = 'long'",
        "coalesce(l.m6_contract_complete, false)",
        "coalesce(l.shared_regime_contract_complete, false)",
        "isfinite(l.net_bps)",
        "isfinite(l.gross_bps)",
    ]
    params: list[object] = []
    if lower is not None:
        clauses.append("l.__ts__ > ?")
        params.append(pd.Timestamp(lower).to_pydatetime())
    if upper is not None:
        clauses.append("l.__ts__ <= ?")
        params.append(pd.Timestamp(upper).to_pydatetime())
    return " AND ".join(clauses), params


def _load_slice(
    args: argparse.Namespace, fields: list[str], *, lower: pd.Timestamp | None, upper: pd.Timestamp | None,
) -> pd.DataFrame:
    raw_fields = _model_columns(args.raw_features)
    ae_fields = _model_columns(args.aegmm)
    if fields != [*raw_fields, *ae_fields]:
        raise ValueError("feature contract changed while loading a fold")
    where, bounds = _population_clause(lower=lower, upper=upper)
    raw_select = ", ".join(f"r.{_quoted(name)} AS {_quoted(name)}" for name in raw_fields)
    ae_select = ", ".join(f"a.{_quoted(name)} AS {_quoted(name)}" for name in ae_fields)
    labels = ", ".join(
        f"p.{_quoted(name)} AS {_quoted(name)}"
        for name in ("label_valid", "barrier_relevance_0_5", "mfe_mae_label_valid", "atr_bps")
    )
    sql = f"""
        SELECT l.candidate_id, l.__ts__, l.side_name, l.gross_bps, l.net_bps,
               {labels}, {raw_select}, {ae_select}
        FROM read_parquet(?) AS l
        INNER JOIN read_parquet(?) AS r USING (candidate_id)
        INNER JOIN read_parquet(?) AS a USING (candidate_id)
        INNER JOIN read_parquet(?) AS p USING (candidate_id)
        WHERE {where}
        ORDER BY l.__ts__, l.candidate_id
    """
    path_glob = str(args.path_labels / "parts" / "*.parquet")
    with duckdb.connect(database=":memory:") as connection:
        result = connection.execute(
            sql, [str(args.ledger), str(args.raw_features), str(args.aegmm), path_glob, *bounds]
        ).fetch_df()
    if result.empty or result["candidate_id"].duplicated().any():
        raise ValueError("empty or non-unique exact candidate slice")
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["label_available_ts"] = result["__ts__"] + pd.Timedelta(hours=12)
    result["query_id"] = result["__ts__"].astype(str) + "|long"
    # LightGBM has an explicit missing-value branch.  Preserve genuinely absent
    # causal inputs as NaN rather than dropping rows or treating an unavailable
    # value as zero; only infinities are invalid numeric encodings and are
    # normalised to that missing branch.  Coverage is audited per fold below.
    values = result.loc[:, fields].to_numpy(float)
    if np.isinf(values).any():
        result.loc[:, fields] = result.loc[:, fields].replace([np.inf, -np.inf], np.nan)
    return result


def _history_cutoffs(args: argparse.Namespace, start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    where, params = _population_clause(lower=None, upper=None)
    where += " AND l.__ts__ < ?"
    params.append((start - pd.Timedelta(hours=12)).to_pydatetime())
    sql = f"""
        SELECT quantile_cont(__ts__, .50), quantile_cont(__ts__, .70), quantile_cont(__ts__, .90)
        FROM read_parquet(?) AS l WHERE {where}
    """
    with duckdb.connect(database=":memory:") as connection:
        values = connection.execute(sql, [str(args.ledger), *params]).fetchone()
    cuts = tuple(pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC") for value in values)
    if any(pd.isna(cut) for cut in cuts) or not (cuts[0] < cuts[1] < cuts[2] < start):
        raise ValueError("could not form strict chronological history cutoffs")
    return cuts


def _base_labels(frame: pd.DataFrame) -> np.ndarray:
    labels = pd.to_numeric(frame["barrier_relevance_0_5"], errors="coerce").to_numpy(float)
    if not np.isfinite(labels).all() or (labels < 0).any() or (labels > 5).any():
        raise ValueError("base barrier relevance must be complete 0--5 labels")
    return labels.astype(np.int8)


def _coverage_audit(frame: pd.DataFrame, fields: list[str]) -> dict[str, object]:
    coverage = frame.loc[:, fields].notna().mean(axis=0)
    below = coverage.loc[coverage < .90]
    return {
        "minimum_feature_coverage": float(coverage.min()),
        "features_below_90pct_coverage": {str(name): float(value) for name, value in below.items()},
    }


def _catalogue_from_fold(catalogue: pd.DataFrame, *, fold: str) -> pd.DataFrame:
    required = {"rule_signature", "rule_structural_path_json", "train_leaf_frequency"}
    missing = sorted(required.difference(catalogue.columns))
    if missing:
        raise ValueError(f"strict catalogue misses structural fields: {missing}")
    grouped = catalogue.groupby("rule_signature", observed=True, sort=False).agg(
        rule_structural_path_json=("rule_structural_path_json", "first"),
        train_leaf_frequency=("train_leaf_frequency", "sum"),
    ).reset_index()
    grouped["rule_instance_id"] = fold + "::" + grouped["rule_signature"].astype(str)
    grouped["fold_id"] = fold
    grouped["model_id"] = fold  # one independently fitted frozen base per outer fold
    grouped["side_name"] = SIDE
    grouped["head_name"] = "p_clear"
    grouped["base_model_version"] = BASE_VERSION
    grouped["model_layer"] = "base"
    return grouped.loc[:, [
        "rule_instance_id", "fold_id", "model_id", "side_name", "head_name",
        "base_model_version", "model_layer", "rule_signature",
        "rule_structural_path_json", "train_leaf_frequency",
    ]]


def _posterior_for_fold(
    frame: pd.DataFrame, contributions: pd.DataFrame, assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    activation = contributions.loc[:, ["candidate_id", "fold_id", "rule_signature", "family_ensemble_tree_contribution"]].copy()
    activation["rule_instance_id"] = activation["fold_id"].astype(str) + "::" + activation["rule_signature"].astype(str)
    known = set(assignments["rule_instance_id"].astype(str))
    known_activation = activation.loc[activation["rule_instance_id"].isin(known)].copy()
    if known_activation.empty:
        result = frame.loc[:, ["candidate_id"]].copy()
        result["base_structural_family__unassigned_mass"] = np.float32(1.0)
        return result, {}
    posterior = materialize_structural_family_posteriors(
        known_activation.loc[:, ["candidate_id", "rule_instance_id", "family_ensemble_tree_contribution"]],
        assignments,
        contribution_column="family_ensemble_tree_contribution",
    )
    known_mass = known_activation.assign(
        __mass__=known_activation["family_ensemble_tree_contribution"].abs()
    ).groupby("candidate_id", observed=True)["__mass__"].sum()
    full_mass = activation.assign(
        __mass__=activation["family_ensemble_tree_contribution"].abs()
    ).groupby("candidate_id", observed=True)["__mass__"].sum()
    result = frame.loc[:, ["candidate_id"]].merge(posterior.features, on="candidate_id", how="left", validate="one_to_one")
    scale = result["candidate_id"].map(known_mass).fillna(0.).to_numpy(float) / np.maximum(result["candidate_id"].map(full_mass).fillna(0.).to_numpy(float), 1e-12)
    feature_names = list(posterior.cluster_id_to_feature.values())
    for name in feature_names:
        result[name] = (result[name].fillna(0.).to_numpy(float) * scale).astype(np.float32)
    result["base_structural_family__unassigned_mass"] = (
        1.0 - result.loc[:, feature_names].sum(axis=1).to_numpy(float)
        if feature_names else np.ones(len(result), dtype=float)
    ).clip(0., 1.).astype(np.float32)
    return result, dict(posterior.cluster_id_to_feature)


def run(args: argparse.Namespace) -> Path:
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}; use --resume for a partial sidecar")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields, ae_fields = _feature_contract(args)
    feature_contract = _digest(fields)
    # Frozen AE/GMM fields are explicitly permitted here as a representation
    # exception.  They are used only to describe historical family-context
    # compatibility, never for structural rule alignment.
    context_columns = list(ae_fields[: min(10, len(ae_fields))])
    catalogues: list[pd.DataFrame] = []
    contribution_paths: dict[str, Path] = {}
    evaluation_paths: dict[str, Path] = {}
    expected_folds = {name for name, _, _ in _folds()}
    requested_folds = set(args.fold) if args.fold else set(expected_folds)
    unknown = sorted(requested_folds.difference(expected_folds))
    if unknown:
        raise ValueError(f"unknown requested folds: {unknown}")
    for fold, start, end in _folds():
        catalogue_path = args.output_dir / "fold_catalogues" / f"{fold}.parquet"
        contribution_path = args.output_dir / "family_contributions" / f"{fold}.parquet"
        evaluation_path = args.output_dir / "fold_evaluations" / f"{fold}.parquet"
        audit_path = args.output_dir / "fold_audits" / f"{fold}.json"
        if fold not in requested_folds:
            continue
        if args.resume and all(path.exists() for path in (catalogue_path, contribution_path, evaluation_path, audit_path)):
            continue
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        base_cut, calibration_cut, meta_cut = _history_cutoffs(args, start_ts)
        base = _load_slice(args, fields, lower=None, upper=base_cut)
        calibration = _load_slice(args, fields, lower=base_cut, upper=calibration_cut)
        model, model_audit = _fit_ranker(base, fields, _base_labels(base), seed=20260840 + len(catalogues))
        calibration_raw = _predict_ranker(model, calibration, fields)
        mapper = _fit_bps_map(calibration_raw, calibration["net_bps"].to_numpy(float))
        del calibration_raw, calibration
        meta_train = _load_slice(args, fields, lower=calibration_cut, upper=meta_cut)
        history_limit = start_ts - pd.Timedelta(hours=12)
        meta_calibration = _load_slice(args, fields, lower=meta_cut, upper=history_limit - pd.Timedelta(microseconds=1))
        test = _load_slice(args, fields, lower=start_ts - pd.Timedelta(microseconds=1), upper=end_ts - pd.Timedelta(microseconds=1))
        evaluation = pd.concat([
            meta_train.assign(meta_partition="meta_train"),
            meta_calibration.assign(meta_partition="meta_calibration"),
            test.assign(meta_partition="test"),
        ], ignore_index=True)
        raw = _predict_ranker(model, evaluation, fields)
        evaluation["base_raw_score"] = raw.astype(np.float32)
        evaluation["base_expected_bps"] = mapper.predict(raw).astype(np.float32)
        artifact = args.output_dir / "strict_base_reasoning" / fold
        reasoning = materialize_strict_oof_base_reasoning(
            [model], base.loc[:, fields], evaluation.loc[:, fields],
            head_name="p_clear", side_name=SIDE, fold_id=fold,
            train_timestamps=base["__ts__"], eval_timestamps=evaluation["__ts__"],
            eval_identity=evaluation.loc[:, ["candidate_id", "__ts__", "side_name"]],
            eval_predictions=evaluation["base_expected_bps"].to_numpy(float),
            train_targets=_base_labels(base), artifact_dir=artifact,
            config=StrictOOFBaseReasoningConfig(
                max_trees_per_model=args.max_trees, contribution_components=16,
            ),
        )
        reason_features = reasoning.features.drop(columns=["head_name", "fold_id", "side_name", "__ts__"])
        evaluation = evaluation.merge(reason_features, on="candidate_id", how="left", validate="one_to_one")
        evaluation["fold"] = fold
        evaluation["feature_contract_sha256"] = feature_contract
        evaluation_path.parent.mkdir(exist_ok=True)
        evaluation.to_parquet(evaluation_path, index=False, compression="zstd")
        evaluation_paths[fold] = evaluation_path
        fold_catalogue = _catalogue_from_fold(reasoning.leaf_rule_catalog, fold=fold)
        catalogue_path.parent.mkdir(exist_ok=True)
        fold_catalogue.to_parquet(catalogue_path, index=False, compression="zstd")
        catalogues.append(fold_catalogue)
        contribution_path.parent.mkdir(exist_ok=True)
        materialize_leaf_family_contributions(artifact, contribution_path)
        contribution_paths[fold] = contribution_path
        audit_path.parent.mkdir(exist_ok=True)
        audit_path.write_text(json.dumps({
            "fold": fold, "base_train_rows": len(base), "meta_train_rows": len(meta_train),
            "meta_calibration_rows": len(meta_calibration), "test_rows": len(test),
            "base_feature_coverage": _coverage_audit(base, fields),
            "scored_feature_coverage": _coverage_audit(evaluation, fields), **model_audit,
        }, indent=2) + "\n")
        del base, meta_train, meta_calibration, test, evaluation, reasoning

    missing_folds = [fold for fold in sorted(expected_folds) if not all((args.output_dir / directory / f"{fold}.{suffix}").exists() for directory, suffix in (("fold_catalogues", "parquet"), ("family_contributions", "parquet"), ("fold_evaluations", "parquet"), ("fold_audits", "json")))]
    if missing_folds:
        (args.output_dir / "partial_manifest.json").write_text(json.dumps({
            "schema": "long_structural_tree_meta_sidecar_v1", "status": "partial",
            "completed_folds": sorted(expected_folds.difference(missing_folds)), "missing_folds": missing_folds,
            "resume": "rerun with --resume --fold <missing fold>; after all folds exist, rerun once with --resume to finalize",
        }, indent=2) + "\n")
        return args.output_dir

    catalogues = [pd.read_parquet(args.output_dir / "fold_catalogues" / f"{fold}.parquet") for fold in sorted(expected_folds)]
    contribution_paths = {fold: args.output_dir / "family_contributions" / f"{fold}.parquet" for fold in expected_folds}
    evaluation_paths = {fold: args.output_dir / "fold_evaluations" / f"{fold}.parquet" for fold in expected_folds}
    strict_audit = [json.loads((args.output_dir / "fold_audits" / f"{fold}.json").read_text()) for fold in sorted(expected_folds)]

    catalogue = pd.concat(catalogues, ignore_index=True)
    family_result = cluster_structural_rule_families(
        catalogue,
        config=StructuralRuleFamilyConfig(
            threshold=args.structural_threshold, min_distinct_folds=2,
            min_distinct_models=2, max_rule_instances_per_cell=args.max_rule_instances,
            max_selected_families_per_cell=args.max_families,
        ),
    )
    catalogue.to_parquet(args.output_dir / "structural_rule_catalogue.parquet", index=False, compression="zstd")
    family_result.assignments.to_parquet(args.output_dir / "structural_family_assignments.parquet", index=False, compression="zstd")
    family_result.family_summary.to_parquet(args.output_dir / "structural_family_summary.parquet", index=False, compression="zstd")
    family_result.pairwise_similarity.to_parquet(args.output_dir / "structural_family_similarity.parquet", index=False, compression="zstd")

    output_rows: list[pd.DataFrame] = []
    health_columns: list[str] = []
    feature_map: dict[str, str] = {}
    for fold in sorted(expected_folds):
        evaluation_path = evaluation_paths[fold]
        evaluation = pd.read_parquet(evaluation_path)
        contribution = pd.read_parquet(contribution_paths[fold])
        posterior, current_map = _posterior_for_fold(evaluation, contribution, family_result.assignments)
        feature_map.update(current_map)
        sidecar = evaluation.merge(posterior, on="candidate_id", how="left", validate="one_to_one")
        posterior_fields = [name for name in sidecar if name.startswith("base_structural_family__")]
        sidecar.loc[:, posterior_fields] = sidecar.loc[:, posterior_fields].fillna(0.0).astype(np.float32)
        health_path = args.output_dir / "historical_health" / f"{fold}.parquet"
        if health_path.exists():
            # Finalisation is deterministic; a prior successful health
            # checkpoint must be reused on a serialization-only resume rather
            # than overwritten.
            health = pd.read_parquet(health_path)
            if set(health["candidate_id"]) != set(sidecar["candidate_id"]):
                raise ValueError("existing health checkpoint does not match the fold candidate identity")
        else:
            health = build_structural_family_historical_health(
                sidecar.rename(columns={"__ts__": "decision_ts"}), context_columns=context_columns,
                output_path=health_path,
            )
        health_columns = [name for name in health if name.startswith("structural_health__")]
        sidecar = sidecar.merge(health.drop(columns="decision_ts"), on="candidate_id", how="left", validate="one_to_one")
        output_rows.append(sidecar)
    output = pd.concat(output_rows, ignore_index=True)
    if output[[*health_columns]].isna().any().any():
        raise ValueError("historical structural health did not cover every scored candidate")
    output.to_parquet(args.output_dir / "tree_meta_candidate_sidecar.parquet", index=False, compression="zstd")
    # Coverage details are a sparse mapping (often empty).  Store it as JSON
    # so Arrow receives a stable scalar schema across folds.
    audit_frame = pd.DataFrame(strict_audit)
    for column in ("base_feature_coverage", "scored_feature_coverage"):
        if column in audit_frame:
            audit_frame[column] = audit_frame[column].map(
                lambda value: json.dumps(value, sort_keys=True)
            )
    audit_frame.to_parquet(args.output_dir / "fold_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "long_structural_tree_meta_sidecar_v1", "status": "complete", "side": SIDE,
        "feature_contract_sha256": feature_contract, "base_model_version": BASE_VERSION,
        "base_features": len(fields), "context_features": context_columns,
        "folds": [row["fold"] for row in strict_audit],
        "structural_alignment": "rule paths/signatures + fold/model recurrence only; no outcomes/economics/health during clustering",
        "posterior_contract": "token-free selected-family absolute base-contribution shares plus unassigned mass",
        "health_contract": "label_available_ts < decision_ts; same-timestamp labels excluded; completed-period portability and causal AE/GMM compatibility",
        "selected_cluster_count": len(family_result.selected_cluster_ids), "cluster_feature_map": feature_map,
        "strict_artifacts": "raw local leaf tokens remain only in per-fold strict artifacts and never in candidate sidecar",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet")
    parser.add_argument("--raw-features", type=Path, default=ROOT / "data_perp/artifacts/long_only_executable_net_lambdarank_20260804_v1/long_mda92_features.parquet")
    parser.add_argument("--aegmm", type=Path, default=ROOT / "data_perp/artifacts/long_pairwise_aegmm_20260804_v1.parquet")
    parser.add_argument("--path-labels", type=Path, default=ROOT / "data_perp/artifacts/long_pairwise_path_labels_20260804_v1")
    parser.add_argument("--max-trees", type=int, default=64)
    parser.add_argument("--max-rule-instances", type=int, default=512)
    parser.add_argument("--max-families", type=int, default=20)
    parser.add_argument("--structural-threshold", type=float, default=.70)
    parser.add_argument("--fold", action="append", choices=[name for name, _, _ in _folds()], help="materialise only one deterministic outer fold; repeatable")
    parser.add_argument("--resume", action="store_true", help="continue a partial per-fold sidecar and/or finalize when all folds exist")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
