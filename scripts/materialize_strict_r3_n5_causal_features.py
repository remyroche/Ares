#!/usr/bin/env python3
"""Materialize target-free causal features for canonical N5 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_n5_canonical import load_n5_contract  # noqa: E402
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    apply_current_admission_by_geometry,
)
from extreme_price_movements.n5_context_features import (  # noqa: E402
    build_cluster_recent_correctness,
    build_residual_head_state,
    cluster_recent_correctness_fields,
    residual_head_state_fields,
)
from scripts.run_strict_r3_c3_window_cadence_ablation import _causal_reliability_context  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _by_geometry_bundle(
    frame: pd.DataFrame,
    builder: object,
) -> pd.DataFrame:
    """Build a causal state independently inside every immutable geometry ID.

    This is used only by the periodic-K9 research arm.  It prevents a recent
    correctness or covariance state produced under one K9 representation from
    becoming an input to an LDF fitted under another representation.  The
    frozen canonical path remains the single-bundle fast path.
    """

    output: list[pd.DataFrame] = []
    for _bundle, positions in frame.groupby("geometry_bundle_sha256", sort=False).groups.items():
        block = frame.loc[positions].copy()
        built = builder(block)  # type: ignore[operator]
        built.index = block.index
        output.append(built)
    if not output:
        raise ValueError("geometry-isolated materialisation received no geometry bundles")
    return pd.concat(output, axis=0).reindex(frame.index)


def _causal_admission_provenance(
    ledger: pd.DataFrame,
    *,
    geometry_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the live EV map without leaking outcome fields into its artifact.

    The periodic-K9 arm deliberately resets the 21/42/84-day admission map at
    each immutable geometry identity.  That reduces support at an episode
    boundary, but preserves the requested same-representation rule and fails
    closed rather than borrowing incompatible score/outcome history.
    """

    return apply_current_admission_by_geometry(ledger, geometry_mode=geometry_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=None,
        help="Optional challenger JSON contract; canonical v1 remains the default.",
    )
    parser.add_argument(
        "--head-ranks", type=Path, default=None,
        help="Optional candidate-keyed ledger containing the ten conditional-head ranks.",
    )
    parser.add_argument(
        "--supplemental-causal-ledger", type=Path, default=None,
        help=(
            "Optional candidate-keyed ledger containing only supplemental causal "
            "Geometry/K9, leaf, or committee fields.  Core strict-R3 scores in "
            "--scored-label-ledger remain authoritative."
        ),
    )
    parser.add_argument(
        "--geometry-mode", choices=("frozen", "episode-isolated"), default="frozen",
        help=(
            "frozen requires the canonical one-geometry contract; "
            "episode-isolated resets every causal reliability input at an "
            "immutable periodic geometry boundary for the research ablation."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable N5 feature output already exists: {args.out_dir}")
    contract = (
        load_n5_contract()
        if args.contract is None
        else json.loads(args.contract.read_text())
    )
    fields = list(contract["features"])
    source_schema = set(pq.ParquetFile(args.scored_label_ledger).schema.names)
    # The repaired lockstep producer intentionally persists the unambiguous
    # serving names (``base_rank42`` and ``conditional_consensus_rank``),
    # whereas the frozen v3 N5 contract predates that rename and calls the
    # same two quantities ``base_rank`` and ``consensus_rank``.  Do not read
    # a stale sidecar just to satisfy those legacy names: materialise explicit
    # in-memory aliases from the authoritative producer instead.  These are
    # exact semantic aliases, not a recalibration or a rank computed over the
    # held window.
    rank_alias_sources = {
        "base_rank": "base_rank42",
        "consensus_rank": "conditional_consensus_rank",
    }
    for legacy, source in rank_alias_sources.items():
        if legacy not in source_schema and source not in source_schema:
            raise ValueError(
                f"scored label ledger lacks {legacy} and its causal serving alias {source}",
            )
    required_primary = {
        "candidate_id", "__decision_ts__", "side_name", "policy_net_bps",
        "policy_label_available_ts", "policy_path_valid", "geometry_bundle_sha256",
        "base_score", "base_rank", "base_anchor_bps", "consensus_rank", "final_score",
        # Required to prove that the only tolerated missing core LDF field is
        # an explicit upstream-only episodic conversion warm-up.
        "correctness_raw", "correctness_gate_active",
    }
    # A repaired current-v5 ledger partitions the EV map by its exact score
    # producer.  The feature sidecar must retain that lineage while it
    # materialises admission provenance; otherwise it would silently pool
    # score CDFs from different conversion/upstream vintages and produce a
    # non-reproducible, less fail-closed map.
    vintage_lineage = {
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "ev_score_family_id", "stack_is_prequential",
    }
    present_vintage = vintage_lineage.intersection(source_schema)
    if present_vintage and present_vintage != vintage_lineage:
        raise ValueError(
            "scored label ledger has incomplete score-vintage lineage: "
            f"missing {sorted(vintage_lineage.difference(source_schema))}",
        )
    required_primary.update(present_vintage)
    # Cluster-health features require current memberships, but raw membership
    # slots never leave this materialiser as model inputs.
    required_primary.update(
        column for column in source_schema if column.startswith("k09__cluster_")
    )
    requested_primary = sorted((required_primary | set(fields)).intersection(source_schema))
    # Include canonical serving aliases even though the older N5 feature
    # contract references their legacy field names.
    requested_primary = sorted(set(requested_primary).union(
        source for legacy, source in rank_alias_sources.items()
        if legacy in fields and legacy not in source_schema and source in source_schema
    ))
    ledger = pd.read_parquet(args.scored_label_ledger, columns=requested_primary)
    for legacy, source in rank_alias_sources.items():
        if legacy in fields and legacy not in ledger.columns:
            ledger[legacy] = pd.to_numeric(ledger[source], errors="coerce")
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("scored label ledger is duplicated by candidate_id")
    geometry_identities = sorted(
        ledger["geometry_bundle_sha256"].dropna().astype(str).unique().tolist(),
    )
    if args.geometry_mode == "frozen" and len(geometry_identities) != 1:
        raise ValueError(
            "canonical N5 materialisation requires one frozen Geometry/K9 identity; "
            f"found {geometry_identities}",
        )
    if args.geometry_mode == "episode-isolated" and not geometry_identities:
        raise ValueError("episode-isolated N5 materialisation requires geometry identities")
    if args.geometry_mode == "episode-isolated":
        required_warmup_provenance = {"correctness_raw", "correctness_gate_active"}
        missing_warmup_provenance = sorted(required_warmup_provenance.difference(ledger.columns))
        if missing_warmup_provenance:
            raise ValueError(
                "episode-isolated N5 materialisation lacks explicit conversion "
                f"warm-up provenance: {missing_warmup_provenance}",
            )
    if args.head_ranks is not None:
        head_schema = set(pq.ParquetFile(args.head_ranks).schema.names)
        head_fields = sorted(
            column for column in head_schema
            if column.startswith("conditional_head__") and column.endswith("__rank")
        )
        if len(head_fields) != 10:
            raise ValueError(
                f"ten-head state requires exactly ten head ranks, found {len(head_fields)}"
            )
        head_frame = pd.read_parquet(args.head_ranks, columns=["candidate_id", *head_fields])
        if head_frame["candidate_id"].duplicated().any():
            raise ValueError("ten-head rank ledger is duplicated by candidate_id")
        already_present = [field for field in head_fields if field in ledger]
        if already_present:
            ledger = ledger.drop(columns=already_present)
        ledger = ledger.merge(
            head_frame.loc[:, ["candidate_id", *head_fields]],
            on="candidate_id", how="left", validate="one_to_one",
        )
    # The map is materialised here as target-free provenance for the OOF LDF
    # runner.  Only its causal score/map outputs are persisted; policy outcomes
    # remain in the scored-label ledger and never become LDF inference inputs.
    if args.supplemental_causal_ledger is not None:
        supplemental_schema = set(
            pq.ParquetFile(args.supplemental_causal_ledger).schema.names
        )
        internally_derived_prefixes = (
            "reliability_", "cluster_recent_", "residual_heads_",
        )
        supplemental_fields = [
            field for field in fields
            if field not in ledger.columns
            and not field.startswith(internally_derived_prefixes)
        ]
        invariant_candidates = [
            field for field in ledger.columns
            if field in supplemental_schema
            and (
                field.startswith("k09__")
                or field in {
                    "k9_entropy", "k9_top2_margin", "k9_ood_distance",
                    "k9_path_support_effective_28d", "k9_model_ood_marginal",
                    "k9_model_drift_psi", "leaf_support_effective",
                    "leaf_support_p05", "leaf_support_p50", "leaf_support_p95",
                    "leaf_ood_marginal",
                }
            )
        ]
        requested_supplemental = [
            "candidate_id", *sorted(set(supplemental_fields + invariant_candidates))
        ]
        supplemental = pd.read_parquet(
            args.supplemental_causal_ledger, columns=requested_supplemental,
        )
        if supplemental["candidate_id"].duplicated().any():
            raise ValueError("supplemental causal ledger is duplicated by candidate_id")
        missing = sorted(set(supplemental_fields).difference(supplemental.columns))
        if missing:
            raise ValueError(
                "supplemental causal ledger lacks requested derived fields: "
                f"{missing}"
            )
        # The two ledgers may have independently persisted Geometry/K9 bundle
        # hashes.  Never accept that as semantic evidence.  The stable K9
        # memberships and invariant summaries must match exactly before fields
        # derived from the supplemental representation can enter MDA.
        invariant_fields = invariant_candidates
        if not invariant_fields:
            raise ValueError("no shared Geometry/K9 invariants available for supplemental lineage check")
        check = ledger.loc[:, ["candidate_id", *invariant_fields]].merge(
            supplemental.loc[:, ["candidate_id", *invariant_fields]],
            on="candidate_id", how="left", validate="one_to_one", suffixes=("__primary", "__supplemental"),
        )
        if check[[f"{field}__supplemental" for field in invariant_fields]].isna().any().any():
            raise ValueError("supplemental causal ledger does not cover every primary candidate")
        for field in invariant_fields:
            primary = pd.to_numeric(check[f"{field}__primary"], errors="coerce").to_numpy(float)
            candidate = pd.to_numeric(check[f"{field}__supplemental"], errors="coerce").to_numpy(float)
            if not np.allclose(primary, candidate, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError(
                    f"supplemental Geometry/K9 invariant differs from primary ledger: {field}"
                )
        if supplemental_fields:
            ledger = ledger.merge(
                supplemental.loc[:, ["candidate_id", *supplemental_fields]],
                on="candidate_id", how="left", validate="one_to_one",
            )
    if args.geometry_mode == "episode-isolated":
        context = _by_geometry_bundle(
            ledger, lambda block: _causal_reliability_context(block)[0],
        )
    else:
        context, _groups = _causal_reliability_context(ledger)
    context.index = ledger.index
    enriched = pd.concat([ledger, context], axis=1)
    cluster_fields = set(cluster_recent_correctness_fields())
    if cluster_fields.intersection(fields):
        cluster_context = (
            _by_geometry_bundle(enriched, build_cluster_recent_correctness)
            if args.geometry_mode == "episode-isolated"
            else build_cluster_recent_correctness(enriched)
        )
        enriched = pd.concat([enriched, cluster_context], axis=1)
    head_state_fields = set(residual_head_state_fields())
    if head_state_fields.intersection(fields):
        head_rank_fields = sorted(
            column for column in enriched.columns
            if column.startswith("conditional_head__") and column.endswith("__rank")
        )
        if len(head_rank_fields) != 10:
            raise ValueError(
                "challenger contract requests ten-head state without ten causal head ranks"
            )
        head_context = (
            _by_geometry_bundle(
                enriched,
                lambda block: build_residual_head_state(block, head_rank_fields),
            )
            if args.geometry_mode == "episode-isolated"
            else build_residual_head_state(enriched, head_rank_fields)
        )
        enriched = pd.concat([enriched, head_context], axis=1)
    missing = sorted(set(fields).difference(enriched.columns))
    if missing:
        raise ValueError(f"N5 causal feature materialization lacks: {missing}")
    output = enriched.loc[:, ["candidate_id", "__decision_ts__", *fields]].copy()
    if output["candidate_id"].duplicated().any():
        raise ValueError("N5 causal feature sidecar changed candidate identity")
    coverage = output.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().mean()
    audit = pd.DataFrame(
        {
            "feature": fields,
            "coverage": [float(coverage[field]) for field in fields],
            "passes_90pct": [bool(coverage[field] >= 0.90) for field in fields],
            "raw_k9_membership": [field.startswith("k09__cluster_") for field in fields],
        }
    )
    # An episode starts with an explicitly upstream-only conversion block.
    # ``correctness_raw`` is intentionally absent there: inventing a neutral
    # value would make a cold K9 representation look trainable.  The episodic
    # LDF runner consumes this exact missingness to emit unit-size warm-up
    # output and excludes those rows from fitting.  Every other inference
    # field still has to meet the canonical 90% coverage gate.
    audit["episodic_warmup_missing_allowed"] = False
    episodic_missing_correctness = (
        args.geometry_mode == "episode-isolated"
        and "correctness_raw" in fields
        and "correctness_gate_active" in ledger
        and audit.loc[audit["feature"].eq("correctness_raw"), "coverage"].iloc[0] < 0.90
    )
    if episodic_missing_correctness:
        missing_raw = pd.to_numeric(ledger["correctness_raw"], errors="coerce").isna()
        if ledger.loc[missing_raw, "correctness_gate_active"].fillna(False).astype(bool).any():
            raise ValueError("episodic missing correctness_raw is not confined to upstream-only warm-up")
        audit.loc[audit["feature"].eq("correctness_raw"), "episodic_warmup_missing_allowed"] = True
    required_coverage = audit["passes_90pct"] | audit["episodic_warmup_missing_allowed"]
    if not required_coverage.all():
        failed = audit.loc[~required_coverage, "feature"].tolist()
        raise ValueError(f"canonical N5 feature coverage below 90%: {failed}")
    if audit["raw_k9_membership"].any():
        raise ValueError("pooled canonical N5 sidecar contains raw K9 memberships")
    admitted, admission_audit = _causal_admission_provenance(
        ledger, geometry_mode=args.geometry_mode,
    )
    mapped = pd.to_numeric(
        admitted["causal_21d_side_expected_net_bps"], errors="coerce",
    )
    provenance_columns = [
        "candidate_id", "__decision_ts__", "side_name", "geometry_bundle_sha256",
        "final_score", "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "causal_21d_side_mapping_status",
        "causal_21d_side_reference_rows", "causal_42d_side_reference_rows",
        "causal_84d_side_reference_rows",
    ]
    provenance = admitted.loc[:, [name for name in provenance_columns if name in admitted]].copy()
    # Admission maps are part of the inference contract.  Persist their native
    # float64 values: float32 quantisation is economically tiny, but it breaks
    # the strict replay-versus-provenance equality check and can move a row
    # exactly at the +50-bps admission boundary.
    provenance["raw_expected_bps"] = mapped.astype(np.float64)
    provenance["mapped_ev_available"] = mapped.notna()
    if provenance["candidate_id"].duplicated().any() or len(provenance) != len(ledger):
        raise AssertionError("causal admission provenance changed candidate identity")
    # Compute all causal state before the immutable directory is created so an
    # exception cannot leave a plausible-looking, incomplete feature artifact.
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "n5_causal_features.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "feature_coverage.parquet", index=False)
    provenance.to_parquet(
        args.out_dir / "causal_admission_provenance.parquet", index=False, compression="zstd",
    )
    admission_audit.to_parquet(
        args.out_dir / "admission_materialization_audit.parquet", index=False,
    )
    manifest = {
        "schema": "strict_r3_n5_causal_feature_sidecar_v1",
        "contract": str(args.contract) if args.contract is not None else "canonical_v1",
        "source": str(args.scored_label_ledger),
        "source_sha256": _sha(args.scored_label_ledger),
        "rows": len(output),
        "geometry_bundle_sha256": (
            geometry_identities[0] if len(geometry_identities) == 1 else geometry_identities
        ),
        "geometry_mode": args.geometry_mode,
        "geometry_refit_cadence": (
            "never" if args.geometry_mode == "frozen" else "episode_boundary_only"
        ),
        "fields": fields,
        "feature_count": len(fields),
        "episodic_warmup_missingness": {
            "field": "correctness_raw" if episodic_missing_correctness else None,
            "allowed": bool(episodic_missing_correctness),
            "handling": (
                "LDF emits unit size during upstream-only blocks and excludes "
                "those rows from same-geometry fitting"
                if episodic_missing_correctness else None
            ),
        },
        "causality": (
            "recent outcomes enter only after policy_label_available_ts via the "
            "prior-resolved reliability builder; all such state is reset at "
            "each geometry boundary" if args.geometry_mode == "episode-isolated" else
            "recent outcomes enter only after policy_label_available_ts via the prior-resolved reliability builder"
        ),
        "admission_materialization": (
            "causal prior-resolved 21/42/84-day expected-net provenance; no outcome "
            "columns are persisted"
        ),
        "outcome_columns_persisted": [],
        "causal_admission_provenance": {
            "path": "causal_admission_provenance.parquet",
            "outcome_columns_persisted": [],
            "raw_expected_bps": "causal_21d_side_expected_net_bps",
            "raw_expected_bps_dtype": "float64",
            "mapped_ev_available": "expected-net map has resolved prior support",
        },
        "raw_k9_memberships_used": False,
        "raw_k9_memberships_used_transiently": bool(cluster_fields.intersection(fields)),
        "ten_head_rank_source": None if args.head_ranks is None else str(args.head_ranks),
        "supplemental_causal_ledger": (
            None if args.supplemental_causal_ledger is None else str(args.supplemental_causal_ledger)
        ),
        "supplemental_lineage": (
            "strict-R3 primary scores retained; supplemental Geometry/K9 invariants "
            "matched exactly by candidate_id before derived fields were joined"
            if args.supplemental_causal_ledger is not None else None
        ),
        "ten_head_surprise_definition": (
            "per-head hit minus expected hit; adjusted rank equals rank minus strictly "
            "prior-resolved 3d/7d surprise"
            if head_state_fields.intersection(fields) else None
        ),
        "ten_head_threshold_lineage": (
            "p90/p95/p99 are applied to each head's prequential training-reference "
            "percentile rank; no held-period threshold fit"
            if head_state_fields.intersection(fields) else None
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(output)}))


if __name__ == "__main__":
    main()
