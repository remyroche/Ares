#!/usr/bin/env python3
"""Run the standalone strict chronological leaf-reasoning meta funnel.

The input ledger must already contain same-side base OOF rows and materialized
candidate reasoning features.  This runner intentionally has no code path that
creates base scores or derives raw tree leaf identifiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_meta_funnel import (  # noqa: E402
    ClusterTaxonomyContract,
    FrozenMetaModelSpec,
    MetaFunnelConfig,
    MetaTransportGateConfig,
    NestedPredecessorOOFContract,
    run_leaf_reasoning_meta_funnel,
    write_immutable_meta_funnel_output,
)
from extreme_price_movements.leaf_reasoning_cluster_materializer import (  # noqa: E402
    LeafReasoningClusterMaterializerError,
    load_finalized_leaf_reasoning_cluster_artifact,
    load_leaf_reasoning_cluster_candidate_artifact,
    merge_strict_cluster_candidate_features,
)
from extreme_price_movements.strict_predecessor_meta_oof import (  # noqa: E402
    StrictPredecessorMetaOOFError,
    load_immutable_strict_predecessor_meta_oof,
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("input must be a .parquet or .csv ledger")


def _groups(path: Path) -> dict[str, tuple[str, ...]]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not all(isinstance(key, str) and isinstance(value, list) for key, value in parsed.items()):
        raise ValueError(f"{path} must be a JSON object mapping group names to feature-name lists")
    return {key: tuple(map(str, value)) for key, value in parsed.items()}


def _frozen_model(path: Path) -> FrozenMetaModelSpec:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("--frozen-model-spec must be a JSON object")
    expected = {"family", "contract_id", "params"}
    missing = sorted(expected.difference(parsed))
    if missing:
        raise ValueError(f"--frozen-model-spec is missing {missing}")
    if not isinstance(parsed["params"], dict):
        raise ValueError("--frozen-model-spec.params must be an object")
    return FrozenMetaModelSpec(str(parsed["family"]), dict(parsed["params"]), str(parsed["contract_id"]))


def _cluster_taxonomy_value(parsed: object) -> ClusterTaxonomyContract:
    if not isinstance(parsed, dict):
        raise ValueError("cluster taxonomy contract must be a JSON object")
    required = {"linkage", "cluster_ids_by_arm"}
    missing = sorted(required.difference(parsed))
    if missing:
        raise ValueError(f"cluster taxonomy contract is missing {missing}")
    ids = parsed["cluster_ids_by_arm"]
    if not isinstance(ids, dict) or not all(isinstance(key, str) and isinstance(value, list) for key, value in ids.items()):
        raise ValueError("cluster_ids_by_arm must be a JSON object of string lists")
    kwargs = {
        name: parsed[name]
        for name in (
            "threshold_by_arm", "c5_source_arm", "c6_source_arm",
            "top_decile_coverage_target", "top_decile_coverage_by_arm",
            "portable_top_decile_coverage_by_arm", "production_soft_cap",
            "exploratory_hard_cap", "c6_best_cross_era_score",
            "c6_best_cross_era_standard_error", "c6_compact_cross_era_score",
            "selection_phase",
        )
        if name in parsed
    }
    return ClusterTaxonomyContract(
        linkage=str(parsed["linkage"]),
        cluster_ids_by_arm={str(key): tuple(map(str, value)) for key, value in ids.items()},
        **kwargs,
    )


def _cluster_taxonomy(path: Path | None) -> ClusterTaxonomyContract | None:
    if path is None:
        return None
    return _cluster_taxonomy_value(json.loads(path.read_text(encoding="utf-8")))


def _validate_cluster_ledger_lineage(ledger_path: Path, cluster_manifest: dict[str, object]) -> None:
    """Require the C candidate root to descend from this immutable meta ledger.

    Exact candidate identity prevents row substitution, while the strict
    manifest hash prevents subtle reuse of C features from a different base
    generation that happened to share candidate IDs.
    """

    path = ledger_path.parent / "meta_ledger_manifest.json"
    if not path.is_file():
        raise ValueError(
            "--cluster-root requires an immutable ledger directory containing "
            "meta_ledger_manifest.json"
        )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or parsed.get("status") != "STRICT_BASE_TO_META_LEDGER_ASSEMBLED":
        raise ValueError("--cluster-root requires a completed immutable strict base-to-meta ledger")
    source_hashes = parsed.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("immutable ledger manifest lacks strict source hashes")
    strict_hashes = {
        str(value)
        for name, value in source_hashes.items()
        if str(name).startswith("strict_manifest_") and isinstance(value, str)
    }
    cluster_hash = cluster_manifest.get("strict_manifest_sha256")
    if not isinstance(cluster_hash, str) or cluster_hash not in strict_hashes:
        raise ValueError(
            "cluster candidate root was not built from one of this ledger's strict source manifests"
        )
    ledger_transports = sorted(map(str, parsed.get("transports", ())))
    cluster_transports = sorted(map(str, cluster_manifest.get("transports", ())))
    if ledger_transports != cluster_transports:
        raise ValueError("cluster candidate and ledger transport contracts differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--feature-groups", required=True, type=Path, help="JSON mapping L0--L4/H0--H6 to feature lists")
    parser.add_argument("--cluster-groups", type=Path, help="optional JSON mapping C0 (frozen upstream compact features) and C1--C6 cluster additions")
    parser.add_argument("--cluster-taxonomy-contract", type=Path, help="required with --cluster-groups: frozen average/complete-linkage threshold/coverage/count metadata")
    parser.add_argument("--cluster-root", type=Path, help="completed immutable C-stage root; verifies hashes/lineage and joins its candidate features exactly")
    parser.add_argument("--cluster-finalization-root", type=Path, help="immutable post-C1--C4 C5/C6 finalization overlay; consumes its hashed threshold-sweep candidate root")
    parser.add_argument("--stages", default="L", help="comma-separated sequential stages from L,H,C; default L avoids fake empty health/cluster ablations")
    parser.add_argument("--successor", choices=("S0", "S1", "S2"), default="S0")
    parser.add_argument(
        "--predecessor-root", type=Path,
        help="required for S2: completed immutable strict predecessor-meta OOF root",
    )
    parser.add_argument("--frozen-model-spec", required=True, type=Path, help="JSON {family, contract_id, params}; objective must be fixed huber")
    parser.add_argument("--fit-protocol", choices=("prequential_batched", "transport_outer_frozen"), default="transport_outer_frozen")
    parser.add_argument("--refit-interval-hours", type=int, default=24)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--min-train-rows", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--required-transport-count", type=int, default=2)
    parser.add_argument("--max-worst-month-net-drop-bps", type=float, help="pre-declared maximum allowed decline versus the stage control; must be <= 0")
    args = parser.parse_args()

    predecessor = None
    predecessor_artifact = None
    if args.successor == "S2":
        if args.predecessor_root is None:
            parser.error("S2 is fail-closed: --predecessor-root must be a completed strict nested predecessor OOF artifact")
        try:
            predecessor_artifact = load_immutable_strict_predecessor_meta_oof(args.predecessor_root)
        except StrictPredecessorMetaOOFError as exc:
            parser.error(str(exc))
        if args.ledger.resolve() != predecessor_artifact.ledger_path.resolve():
            parser.error(
                "S2 --ledger must be the joined ledger inside --predecessor-root; "
                "a manually joined table is not accepted"
            )
        predecessor = NestedPredecessorOOFContract(predecessor_artifact.feature_columns)

    if (args.cluster_groups is None) != (args.cluster_taxonomy_contract is None):
        parser.error("--cluster-groups and --cluster-taxonomy-contract must be supplied together")
    cluster_sources = sum(value is not None for value in (args.cluster_root, args.cluster_finalization_root))
    if cluster_sources and (args.cluster_groups is not None or args.cluster_taxonomy_contract is not None):
        parser.error("--cluster-root/--cluster-finalization-root are mutually exclusive with --cluster-groups/--cluster-taxonomy-contract")
    if cluster_sources > 1:
        parser.error("--cluster-root and --cluster-finalization-root are mutually exclusive")
    stages = tuple(part.strip().upper() for part in args.stages.split(",") if part.strip())
    config = MetaFunnelConfig(
        min_train_rows=args.min_train_rows, ridge_alpha=args.ridge_alpha,
        fit_protocol=args.fit_protocol, refit_interval_hours=args.refit_interval_hours,
    )
    gate_config = MetaTransportGateConfig(
        required_transport_count=args.required_transport_count,
        max_worst_month_net_drop_bps=args.max_worst_month_net_drop_bps,
    )
    ledger = predecessor_artifact.ledger if predecessor_artifact is not None else _read_table(args.ledger)
    feature_groups = _groups(args.feature_groups)
    if predecessor_artifact is not None:
        # S2 has no free-form feature bucket.  Its compact predecessor
        # component hand-off is therefore an explicit addition to the L3
        # contribution-bundle representation; L4/H/C inherit it only through
        # their normal declared sequential contracts.  The validation inside
        # ``run_leaf_reasoning_meta_funnel`` still requires a genuine G2/G3
        # base representation in addition to these six fields.
        feature_groups["L3"] = tuple(dict.fromkeys([
            *feature_groups.get("L3", ()), *predecessor_artifact.feature_columns,
        ]))
    cluster_groups = _groups(args.cluster_groups) if args.cluster_groups else None
    cluster_taxonomy = _cluster_taxonomy(args.cluster_taxonomy_contract)
    if args.cluster_root is not None or args.cluster_finalization_root is not None:
        try:
            cluster = (
                load_leaf_reasoning_cluster_candidate_artifact(args.cluster_root)
                if args.cluster_root is not None
                else load_finalized_leaf_reasoning_cluster_artifact(args.cluster_finalization_root)
            )
            _validate_cluster_ledger_lineage(args.ledger, dict(cluster.manifest))
            ledger = merge_strict_cluster_candidate_features(ledger, cluster.candidate_features)
            cluster_groups = {name: tuple(fields) for name, fields in cluster.groups.items()}
            cluster_taxonomy = _cluster_taxonomy_value(dict(cluster.taxonomy))
        except (LeafReasoningClusterMaterializerError, ValueError) as exc:
            parser.error(str(exc))
    # A multi-arm H/C run can emit millions of rows.  Stream its compact
    # prediction ledger to a hidden sibling cache and move it into the writer's
    # atomic staging directory after the final arm has been evaluated.  The
    # cache is removed on either success or failure; it is never a result by
    # itself because it has no manifest.
    args.output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".leaf_reasoning_meta_predictions-", dir=args.output_root) as cache_dir:
        result = run_leaf_reasoning_meta_funnel(
            ledger,
            feature_groups=feature_groups,
            cluster_groups=cluster_groups,
            cluster_taxonomy=cluster_taxonomy,
            stages=stages,
            successor=args.successor,
            predecessor_contract=predecessor,
            model_spec=_frozen_model(args.frozen_model_spec),
            config=config,
            gate_config=gate_config,
            prediction_cache_path=Path(cache_dir) / "predictions.parquet",
        )
        output = write_immutable_meta_funnel_output(
            result,
            args.output_root,
            config=config,
            gate_config=gate_config,
            consume_prediction_cache=True,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
