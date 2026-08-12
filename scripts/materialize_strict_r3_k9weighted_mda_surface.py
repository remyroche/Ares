#!/usr/bin/env python3
"""Materialise a target-safe MDA surface for frozen-K9 reliability features.

Raw K9 membership columns are joined from the internal sidecar.  By default
they are transient: they create soft-membership-weighted 3/7/14-day cluster
health features and are then omitted from the output contract.  The explicit
``--include-frozen-k9-membership-posterior`` ablation persists only the nine
posterior coordinates, provided every row belongs to the identical frozen
Geometry/K9 bundle.  Outcome history enters derived fields only when its
policy label was resolved strictly before the candidate's decision timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_context_features import (
    build_cluster_recent_correctness,
    build_cluster_score_conditioned_correctness,
    build_residual_head_state,
    cluster_recent_correctness_fields,
    cluster_score_conditioned_correctness_fields,
    k9_membership_columns,
    residual_head_state_fields,
)
from extreme_price_movements.causal_market_regime_systems import (
    CONTINUOUS_CONTEXT_FEATURE_KEYS,
    RELATIONSHIP_BREAK_FEATURE_KEYS,
)
from extreme_price_movements.strict_r3_canonical_v2 import GEOMETRY_SCHEMA, load_geometry_bundle
from scripts.run_strict_r3_c3_window_cadence_ablation import _causal_reliability_context


IDENTITY = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "bundle_sha256", "geometry_bundle_sha256",
)
OUTCOME = (
    "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
    "policy_gross_bps", "policy_exit_reason", "policy_exit_bar_15m",
    "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_dirs(root: Path) -> list[Path]:
    paths = sorted(path for path in root.glob("month=20??-??") if path.is_dir())
    if not paths:
        # Canonical schema-v2 walk-forward runs place score folds at
        # ``<run>/scores/month=YYYY-MM``; legacy runs wrapped them one level
        # deeper.  The surface is identical under either layout.
        paths = sorted(path for path in (root / "scores").glob("month=20??-??") if path.is_dir())
    if not paths:
        raise FileNotFoundError(f"no month=YYYY-MM directories under {root}")
    return paths


def _score_dir(path: Path) -> Path:
    """Return the immutable score directory for either supported fold layout."""

    month = path.name.removeprefix("month=")
    nested = path / "scores" / f"month={month}"
    if nested.is_dir():
        return nested
    if (path / "predictions.parquet").exists():
        return path
    raise FileNotFoundError(f"no score directory for {month}")


def _read_month(path: Path) -> pd.DataFrame:
    month = path.name.removeprefix("month=")
    score_dir = _score_dir(path)
    predictions = score_dir / "predictions.parquet"
    memberships = score_dir / "k9_membership_internal.parquet"
    if not predictions.exists() or not memberships.exists():
        raise FileNotFoundError(f"{month} needs predictions and internal K9 memberships")
    score = pd.read_parquet(predictions)
    membership = pd.read_parquet(memberships)
    fields = list(k9_membership_columns())
    required = {"candidate_id", "__decision_ts__", "side_name", "geometry_bundle_sha256", *fields}
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise KeyError(f"{month} membership sidecar missing {missing}")
    keys = ["candidate_id", "__decision_ts__", "side_name"]
    if len(score) != len(membership) or not score["candidate_id"].is_unique or not membership["candidate_id"].is_unique:
        raise ValueError(f"{month} does not have one-to-one candidate identities")
    joined = score.merge(
        membership.loc[:, [*keys, "geometry_bundle_sha256", *fields]],
        on=keys,
        how="left",
        suffixes=("", "__sidecar"),
        validate="one_to_one",
    )
    sidecar_hash = "geometry_bundle_sha256__sidecar"
    if sidecar_hash in joined:
        mismatch = joined[sidecar_hash].astype("string") != joined["geometry_bundle_sha256"].astype("string")
        if mismatch.any():
            raise AssertionError(f"{month} geometry identity differs between score and membership sidecar")
        joined = joined.drop(columns=sidecar_hash)
    if joined[fields].isna().any().any():
        raise AssertionError(f"{month} has missing raw memberships")
    total = joined[fields].sum(axis=1)
    if (total.sub(1.0).abs() > 1e-5).any():
        raise AssertionError(f"{month} membership rows do not sum to one")
    return joined


def _frozen_geometry_nonraw_surface(
    *,
    months: list[Path],
    source_panel: Path,
    geometry_bundle: Path,
    temperature_scale: float = 1.0,
    geometry_schema: str = GEOMETRY_SCHEMA,
) -> pd.DataFrame:
    """Regenerate the established support/OOD surface under this same bundle.

    This deliberately never borrows geometry values from a prior scorer.  A
    repaired schema-v2 score may share candidate IDs with a legacy scorer but
    not its score or Geometry/K9 lineage.
    """

    geometry = load_geometry_bundle(geometry_bundle, expected_schema=str(geometry_schema))
    columns = ["candidate_id", "__decision_ts__", "side_name", *geometry.encoder_fields]
    parts: list[pd.DataFrame] = []
    for month_dir in months:
        month = month_dir.name.removeprefix("month=")
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        source = pd.read_parquet(
            source_panel,
            columns=list(dict.fromkeys(columns)),
            filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
        )
        source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True)
        source = source.loc[source["side_name"].astype(str).str.lower().eq("long")].copy()
        score_path = _score_dir(month_dir) / "predictions.parquet"
        ids = pd.read_parquet(score_path, columns=["candidate_id"])
        if not source["candidate_id"].is_unique or set(source["candidate_id"]) != set(ids["candidate_id"]):
            raise AssertionError(f"{month} geometry source does not exactly match scored candidates")
        source = source.set_index("candidate_id").loc[ids["candidate_id"]].reset_index()
        state = geometry.transform(source, temperature_scale=float(temperature_scale))
        membership = state.loc[:, list(k9_membership_columns())].to_numpy(float)
        ordered = np.sort(membership, axis=1)
        entropy = -np.sum(membership * np.log(np.maximum(membership, 1e-12)), axis=1) / np.log(9.0)
        selected = pd.DataFrame({
            "candidate_id": source["candidate_id"].to_numpy(),
            # Transient reconstruction-only slots.  They replace any upstream
            # membership before cluster history is calculated, then are
            # removed from the persisted MDA surface.
            **{column: state[column].to_numpy() for column in k9_membership_columns()},
            "__recomputed_geometry_bundle_sha256": str(geometry.bundle_sha256),
            "leaf_support_effective": state["rule_support_effective"].to_numpy(),
            "leaf_support_p05": state["rule_support_p05"].to_numpy(),
            "leaf_support_p50": state["rule_support_p50"].to_numpy(),
            "leaf_support_p95": state["rule_support_p95"].to_numpy(),
            "leaf_support_contribution_weighted": state["rule_support_contribution_weighted"].to_numpy(),
            "leaf_support_contribution_weighted_log": np.log1p(state["rule_support_contribution_weighted"].to_numpy()),
            "leaf_support_adequate_fraction": state["rule_support_adequate_fraction"].to_numpy(),
            "leaf_support_leaf_coverage": state["rule_support_leaf_coverage"].to_numpy(),
            "leaf_ood_marginal": state["rule_ood_marginal"].to_numpy(),
            "leaf_ood_joint": state["rule_ood_joint_factorised"].to_numpy(),
            "k9_entropy": entropy,
            "k9_top2_margin": ordered[:, -1] - ordered[:, -2],
            "k9_ood_distance": state["k9_cluster_weighted_ood"].to_numpy(),
            "k9_path_support_effective_28d": state["path_support_effective_28d"].to_numpy(),
            "k9_path_support_adequate_fraction": state["path_support_adequate_fraction"].to_numpy(),
            "k9_path_ood_marginal": state["path_ood_marginal"].to_numpy(),
            "k9_model_ood_marginal": state["model_ood_marginal"].to_numpy(),
            "k9_model_ood_mahalanobis_diag": state["model_ood_mahalanobis_diag"].to_numpy(),
            "k9_model_drift_psi": state["model_drift_prototype_psi"].to_numpy(),
            "k9_model_drift_ks": state["model_drift_prototype_ks"].to_numpy(),
        })
        # These are not a covariance of the membership simplex.  They are
        # candidate-specific, soft-membership-weighted breaks of each frozen
        # K9 cluster's encoder-input geometry versus its Oct--Dec reference.
        # Older immutable bundles legitimately lack them, while the versioned
        # structural-break bundle must expose all three.
        within_cluster = [
            column for column in geometry.structural_fields
            if column.startswith("k9_cluster_activation_weighted_within_")
        ]
        if within_cluster:
            if len(within_cluster) != 3 or any(column not in state for column in within_cluster):
                raise AssertionError("frozen within-cluster geometry-break contract is incomplete")
            for column in within_cluster:
                selected[column] = state[column].to_numpy()
        parts.append(selected)
    combined = pd.concat(parts, ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise AssertionError("geometry surface duplicate candidate identity")
    return combined


def _replace_geometry_state(
    frame: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    """Atomically replace memberships, summaries and their frozen identity."""

    raw = list(k9_membership_columns())
    # The recomputed state owns the posterior values.  They are retained or
    # dropped only at final surface construction; retaining upstream values
    # here would create suffixes and could silently use a stale sidecar.
    base = frame.drop(columns=[column for column in raw if column in frame])
    merged = base.merge(structural, on="candidate_id", how="left", validate="one_to_one")
    required = structural.columns.drop("candidate_id")
    if merged[required].isna().any().any():
        raise AssertionError("recomputed frozen geometry state did not cover every scored candidate")
    if "geometry_bundle_sha256" in merged:
        merged["upstream_geometry_bundle_sha256"] = merged["geometry_bundle_sha256"]
    merged["geometry_bundle_sha256"] = merged.pop("__recomputed_geometry_bundle_sha256")
    return merged


def _apply_policy_label_overlay(frame: pd.DataFrame, overlay_dir: Path) -> pd.DataFrame:
    """Replace only evaluation/history policy labels from a versioned overlay.

    The score, identities, Geometry/K9 representation and all model inputs
    remain owned by ``frame``.  The overlay is permitted to alter only the
    already-resolved policy outcome ledger used by causal history and held-out
    evaluation; its identity must exactly match the scored universe.
    """

    parts = sorted(overlay_dir.glob("month=20??-??.parquet"))
    if not parts:
        raise FileNotFoundError(f"no monthly policy overlay parts under {overlay_dir}")
    columns = [
        "candidate_id", "policy_path_valid", "policy_label_available_ts",
        "policy_net_bps", "policy_gross_bps", "policy_exit_reason",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_cost_bps", "policy_atr_source", "policy_outcome_source",
        "policy_market_data_quality",
    ]
    overlay = pd.concat([pd.read_parquet(path, columns=columns) for path in parts], ignore_index=True)
    if overlay["candidate_id"].duplicated().any():
        raise AssertionError("policy overlay contains duplicate candidate identity")
    # A versioned overlay may contain later scored months while a causal
    # diagnostic intentionally materializes a strict earlier prefix.  Narrow
    # the overlay to the immutable score population first, then require exact
    # one-to-one coverage of *that* population.  This does not allow a missing
    # target identity or a foreign identity to affect labels/history.
    target_ids = set(frame["candidate_id"])
    overlay = overlay.loc[overlay["candidate_id"].isin(target_ids)].copy()
    if len(overlay) != len(frame) or set(overlay["candidate_id"]) != target_ids:
        raise AssertionError("policy overlay must cover exactly the scored candidate universe")
    result = frame.drop(columns=[column for column in columns if column != "candidate_id" and column in frame]).merge(
        overlay, on="candidate_id", how="left", validate="one_to_one",
    )
    if result["policy_path_valid"].isna().any():
        raise AssertionError("policy overlay join left an unlabelled scored identity")
    valid = result["policy_path_valid"].astype(bool)
    net = pd.to_numeric(result["policy_net_bps"], errors="coerce")
    gross = pd.to_numeric(result["policy_gross_bps"], errors="coerce")
    if valid.any() and not np.allclose(net.loc[valid], gross.loc[valid] - 100.0, rtol=0.0, atol=1e-12):
        raise AssertionError("policy overlay cost was not applied exactly once")
    return result


def _join_continuous_market_context(frame: pd.DataFrame, sidecar_path: Path) -> pd.DataFrame:
    """Exactly join the target-free stable continuous-context sidecar.

    Fold-local latent posterior coordinates are expressly not accepted: their
    component semantics can differ between fitted blocks.  This contract is
    restricted to continuous decision-time observables and causal relationship
    breaks, whose names and meanings remain stable at inference.
    """

    sidecar_path = Path(sidecar_path)
    schema = set(pq.ParquetFile(sidecar_path).schema.names)
    stable = [*CONTINUOUS_CONTEXT_FEATURE_KEYS, *RELATIONSHIP_BREAK_FEATURE_KEYS]
    identity = ["candidate_id", "__ts__", "__symbol__", "side_name"]
    provenance = ["source_utc", "continuous_context_available_utc"]
    missing = sorted(set(identity + stable + provenance).difference(schema))
    if missing:
        raise KeyError(f"continuous context sidecar is incomplete: {missing}")
    forbidden = sorted(
        column for column in schema
        if column.startswith((
            "market_regime__state_p_", "regime_state_p__", "transition_state_p__", "geometry_regime__",
        ))
    )
    if forbidden:
        raise AssertionError(
            "continuous reliability join must not expose fold-local latent-state coordinates: "
            f"{forbidden[:8]}"
        )
    context = pd.read_parquet(sidecar_path, columns=[*identity, *provenance, *stable])
    for column in ("__ts__", "source_utc", "continuous_context_available_utc"):
        context[column] = pd.to_datetime(context[column], utc=True, errors="raise")
    if context["candidate_id"].duplicated().any():
        raise AssertionError("continuous context sidecar has duplicate candidate IDs")
    target_identity = frame.loc[:, identity].copy()
    target_identity["__ts__"] = pd.to_datetime(target_identity["__ts__"], utc=True, errors="raise")
    joined = target_identity.merge(
        context, on="candidate_id", how="left", suffixes=("", "__context"), validate="one_to_one",
    )
    for column in ("__ts__", "__symbol__", "side_name"):
        actual = joined.pop(f"{column}__context")
        if actual.isna().any() or not actual.astype(str).eq(joined[column].astype(str)).all():
            raise AssertionError(f"continuous context identity mismatch for {column}")
    # Individual continuous fields have legitimate causal warm-up/source gaps;
    # their per-field >=90% coverage gate belongs to MDA.  What this exact
    # identity join must prohibit is a candidate with no state at all.
    unmatched = int(joined[stable].isna().all(axis=1).sum())
    if unmatched:
        raise AssertionError(f"continuous context does not cover every scored candidate ({unmatched} unmatched)")
    if (joined["source_utc"] > joined["__ts__"]).any() or (
        joined["continuous_context_available_utc"] > joined["__ts__"]
    ).any():
        raise AssertionError("continuous context join looks ahead of candidate decision time")
    if set(stable).intersection(frame.columns):
        raise AssertionError("continuous context fields already exist on score frame")
    result = pd.concat([frame.reset_index(drop=True), joined.loc[:, stable].reset_index(drop=True)], axis=1)
    if len(result) != len(frame):
        raise AssertionError("continuous context join changed candidate cardinality")
    return result


def _join_candidate_meta_context(frame: pd.DataFrame, sidecar_path: Path) -> pd.DataFrame:
    """Exactly join target-free candidate-specific fields owned by meta keys.

    Unlike the continuous market-state sidecar, these fields can vary within a
    decision timestamp.  They are nevertheless identity- and timestamp-bound
    to the same PIT candidate universe, and enter the MDA as ordinary removal
    candidates—not as a protected context tier.
    """

    sidecar_path = Path(sidecar_path)
    schema = set(pq.ParquetFile(sidecar_path).schema.names)
    identity = ["candidate_id", "__ts__", "__symbol__", "side_name"]
    provenance = ["exact170_context_source_utc", "exact170_context_available_utc"]
    fields = sorted(column for column in schema if column.startswith("meta_context__"))
    if len(fields) < 40:
        raise ValueError(f"candidate meta context sidecar is unexpectedly small ({len(fields)} fields)")
    missing = sorted(set(identity + provenance).difference(schema))
    if missing:
        raise KeyError(f"candidate meta context sidecar lacks {missing}")
    if set(fields).intersection(frame.columns):
        raise AssertionError("candidate meta context fields already exist on score frame")
    context = pd.read_parquet(sidecar_path, columns=[*identity, *provenance, *fields])
    for column in ("__ts__", *provenance):
        context[column] = pd.to_datetime(context[column], utc=True, errors="raise")
    if context["candidate_id"].duplicated().any():
        raise AssertionError("candidate meta context sidecar has duplicate candidate IDs")
    target = frame.loc[:, identity].copy()
    target["__ts__"] = pd.to_datetime(target["__ts__"], utc=True, errors="raise")
    joined = target.merge(context, on="candidate_id", how="left", suffixes=("", "__context"), validate="one_to_one")
    for column in ("__ts__", "__symbol__", "side_name"):
        actual = joined.pop(f"{column}__context")
        if actual.isna().any() or not actual.astype(str).eq(joined[column].astype(str)).all():
            raise AssertionError(f"candidate meta context identity mismatch for {column}")
    # A sidecar may retain an explicit causal source warm-up gap (for example
    # the first decision hour before its feature panel starts).  It is not a
    # candidate-universe failure and must not filter or reclassify that row.
    # Per-field MDA coverage/variance gates and train-only model imputation
    # decide whether any affected field can be used.
    if (joined["exact170_context_source_utc"] > joined["__ts__"]).any() or (
        joined["exact170_context_available_utc"] > joined["__ts__"]
    ).any():
        raise AssertionError("candidate meta context looks ahead of candidate decision time")
    result = pd.concat([frame.reset_index(drop=True), joined.loc[:, fields].reset_index(drop=True)], axis=1)
    if len(result) != len(frame):
        raise AssertionError("candidate meta context join changed candidate cardinality")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--end-month-exclusive",
        help=(
            "Optional exclusive YYYY-MM upper bound for the emitted target months. "
            "Earlier context folds are still retained solely to seed causal state."
        ),
    )
    parser.add_argument("--source-panel", type=Path)
    parser.add_argument("--geometry-bundle", type=Path)
    parser.add_argument(
        "--geometry-schema", type=str, default=GEOMETRY_SCHEMA,
        help="Expected immutable geometry schema; an ablation must opt in explicitly.",
    )
    parser.add_argument(
        "--policy-label-overlay", type=Path,
        help="Versioned evaluation/history-only policy label overlay with exact scored identities.",
    )
    parser.add_argument(
        "--context-fold-root", type=Path,
        help=(
            "Optional strictly earlier scored folds. They seed causal rolling "
            "reliability/cluster history but are excluded from this output."
        ),
    )
    parser.add_argument(
        "--context-policy-label-overlay", type=Path,
        help="Required causal policy-label overlay for --context-fold-root.",
    )
    parser.add_argument(
        "--continuous-context-sidecar", type=Path,
        help=(
            "Exact candidate-keyed, strictly prequential continuous market-context "
            "sidecar. Only stable continuous and relationship-break fields are "
            "accepted; fold-local latent coordinates are rejected."
        ),
    )
    parser.add_argument(
        "--candidate-meta-context-sidecar", type=Path,
        help=(
            "Exact candidate-keyed target-free panel of meta-owned, potentially "
            "within-timestamp-varying context fields. They join as ordinary MDA inputs."
        ),
    )
    parser.add_argument(
        "--cluster-membership-power", type=float, default=1.0,
        help=(
            "Power applied to the frozen K9 posterior before its soft cluster-history "
            "aggregate. One is unmodified; values above one are an explicit ablation."
        ),
    )
    parser.add_argument(
        "--geometry-temperature-scale", type=float, default=1.0,
        help=(
            "Explicit frozen-K9 soft-assignment temperature multiplier for a "
            "representation ablation. It never refits the encoder, centres, "
            "ordering, or bundle identity."
        ),
    )
    parser.add_argument(
        "--include-frozen-k9-membership-posterior", action="store_true",
        help=(
            "Persist the nine frozen K9 posterior coordinates as ordinary MDA "
            "candidates. This is allowed only under the one-bundle identity "
            "invariant; raw distances and confidence slots remain excluded."
        ),
    )
    args = parser.parse_args()

    months = _month_dirs(args.fold_root)
    if args.end_month_exclusive is not None:
        cutoff = pd.Timestamp(f"{str(args.end_month_exclusive)}-01", tz="UTC")
        months = [
            path for path in months
            if pd.Timestamp(f"{path.name.removeprefix('month=')}-01", tz="UTC") < cutoff
        ]
        if not months:
            raise ValueError("--end-month-exclusive excluded every target month")
    target = pd.concat([_read_month(path) for path in months], ignore_index=True)
    target["__decision_ts__"] = pd.to_datetime(target["__decision_ts__"], utc=True, errors="raise")
    target = target.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if target["candidate_id"].duplicated().any():
        raise AssertionError("candidate ids must remain unique across the MDA surface")
    if target["geometry_bundle_sha256"].nunique(dropna=False) != 1:
        raise AssertionError("this MDA surface requires exactly one frozen Geometry/K9 identity")
    if args.policy_label_overlay is not None:
        target = _apply_policy_label_overlay(target, args.policy_label_overlay)
        target = target.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if (args.source_panel is None) != (args.geometry_bundle is None):
        raise ValueError("--source-panel and --geometry-bundle must be supplied together")
    if args.source_panel is not None:
        if not np.isfinite(float(args.geometry_temperature_scale)) or float(args.geometry_temperature_scale) <= 0.0:
            raise ValueError("--geometry-temperature-scale must be finite and positive")
        structural = _frozen_geometry_nonraw_surface(
            months=months, source_panel=args.source_panel, geometry_bundle=args.geometry_bundle,
            temperature_scale=float(args.geometry_temperature_scale),
            geometry_schema=str(args.geometry_schema),
        )
        target = _replace_geometry_state(target, structural)
    target_ids = target["candidate_id"].copy()
    context_months: list[Path] = []
    if (args.context_fold_root is None) != (args.context_policy_label_overlay is None):
        raise ValueError("--context-fold-root and --context-policy-label-overlay must be supplied together")
    if args.context_fold_root is not None:
        context_months = _month_dirs(args.context_fold_root)
        context = pd.concat([_read_month(path) for path in context_months], ignore_index=True)
        context["__decision_ts__"] = pd.to_datetime(context["__decision_ts__"], utc=True, errors="raise")
        context = _apply_policy_label_overlay(context, args.context_policy_label_overlay)
        if args.source_panel is not None:
            context_structural = _frozen_geometry_nonraw_surface(
                months=context_months, source_panel=args.source_panel, geometry_bundle=args.geometry_bundle,
                temperature_scale=float(args.geometry_temperature_scale),
                geometry_schema=str(args.geometry_schema),
            )
            context = _replace_geometry_state(context, context_structural)
        if context["candidate_id"].duplicated().any() or context["candidate_id"].isin(target_ids).any():
            raise AssertionError("context rows must be distinct from target scored identities")
        if context["geometry_bundle_sha256"].nunique(dropna=False) != 1 or (
            context["geometry_bundle_sha256"].iloc[0] != target["geometry_bundle_sha256"].iloc[0]
        ):
            raise AssertionError("context and target must share the identical frozen Geometry/K9 bundle")
        if context["__decision_ts__"].max() >= target["__decision_ts__"].min():
            raise AssertionError("context folds must be strictly earlier than the target population")
        frame = pd.concat([context, target], ignore_index=True, sort=False)
    else:
        frame = target
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("context-plus-target candidate identities must remain unique")
    if args.continuous_context_sidecar is not None:
        frame = _join_continuous_market_context(frame, args.continuous_context_sidecar)
    if args.candidate_meta_context_sidecar is not None:
        frame = _join_candidate_meta_context(frame, args.candidate_meta_context_sidecar)
    # Alias the repaired, same-model 42-day rank into the established feature
    # name.  The historical legacy sidecar cannot be used because its score
    # lineage differs materially from this strict schema-v2 score.
    frame["base_rank"] = pd.to_numeric(frame["base_rank42"], errors="coerce")
    global_context, _ = _causal_reliability_context(frame)
    global_context.index = frame.index
    frame = pd.concat([frame, global_context], axis=1)

    rank_fields = sorted(
        column for column in frame.columns
        if column.startswith("residual_head__") and column.endswith("__rank")
    )
    if len(rank_fields) != 10:
        raise AssertionError(f"expected ten residual rank heads, received {len(rank_fields)}")
    cluster = build_cluster_recent_correctness(
        frame, membership_power=float(args.cluster_membership_power),
    )
    score_conditioned_cluster = build_cluster_score_conditioned_correctness(
        frame, membership_power=float(args.cluster_membership_power),
    )
    committee = build_residual_head_state(frame, rank_fields)
    raw_memberships = list(k9_membership_columns())
    frame_for_output = frame if args.include_frozen_k9_membership_posterior else frame.drop(columns=raw_memberships)
    output = pd.concat([frame_for_output, cluster, score_conditioned_cluster, committee], axis=1)
    # Context rows solely seed prior-resolved rolling state.  They are never
    # emitted as target evidence or reused as held evaluation rows.
    output = output.loc[output["candidate_id"].isin(target_ids)].copy()
    output = output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    posterior_fields = list(k9_membership_columns())
    present_posterior = [column for column in posterior_fields if column in output]
    if args.include_frozen_k9_membership_posterior:
        if present_posterior != posterior_fields:
            raise AssertionError("the explicit frozen K9 posterior contract is incomplete")
        posterior = output.loc[:, posterior_fields].apply(pd.to_numeric, errors="coerce")
        if posterior.isna().any().any() or not np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError("persisted frozen K9 posterior is not a finite probability simplex")
        if output["geometry_bundle_sha256"].nunique(dropna=False) != 1:
            raise AssertionError("K9 posterior requires exactly one frozen geometry identity")
    elif present_posterior:
        raise AssertionError(f"raw K9 slots escaped the MDA surface: {present_posterior}")
    expected = (
        set(cluster_recent_correctness_fields())
        | set(cluster_score_conditioned_correctness_fields())
        | set(residual_head_state_fields())
    )
    missing = sorted(expected.difference(output.columns))
    if missing:
        raise AssertionError(f"missing derived causal state fields: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "mda_surface.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    selected = [
        *IDENTITY,
        *OUTCOME,
        *[column for column in output if column.startswith("k9_") or column.startswith("leaf_")],
        *present_posterior,
        *[column for column in output if column.startswith("continuous_regime__")],
        *[column for column in output if column.startswith("meta_context__")],
        *cluster_recent_correctness_fields(),
        *cluster_score_conditioned_correctness_fields(),
        *residual_head_state_fields(),
    ]
    coverage = pd.DataFrame({
        "feature": selected,
        "coverage": [float(output[column].notna().mean()) for column in selected],
        "finite_coverage": [
            float(pd.to_numeric(output[column], errors="coerce").notna().mean())
            if column not in IDENTITY and column not in OUTCOME else None
            for column in selected
        ],
    })
    coverage.to_parquet(args.out_dir / "feature_coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_k9weighted_mda_surface_v1",
        "rows": int(len(output)),
        "months": [path.name.removeprefix("month=") for path in months],
        "geometry_bundle_sha256": str(output["geometry_bundle_sha256"].iloc[0]),
        "residual_rank_heads": rank_fields,
        "raw_k9_memberships": (
            "nine frozen posterior coordinates persisted under one immutable Geometry/K9 identity; "
            "raw distances and confidence slots excluded"
            if args.include_frozen_k9_membership_posterior
            else "transient only; excluded from output"
        ),
        "frozen_k9_membership_posterior_fields": present_posterior,
        "cluster_history": "soft K9-membership weighted; same frozen bundle; resolved labels strictly before decision timestamp",
        "cluster_score_conditioned_history": "same soft K9 aggregate restricted to the candidate's fixed final-score CDF band; prior-resolved labels only",
        "cluster_membership_power": float(args.cluster_membership_power),
        "geometry_temperature_scale": float(args.geometry_temperature_scale),
        "geometry_schema": str(args.geometry_schema),
        "cluster_history_horizons_days": [3, 7, 14],
        "policy_outcomes": "evaluation and causal-state history only; never model scoring inputs",
        "policy_label_overlay": str(args.policy_label_overlay) if args.policy_label_overlay else None,
        "context_fold_root": str(args.context_fold_root) if args.context_fold_root else None,
        "context_policy_label_overlay": str(args.context_policy_label_overlay) if args.context_policy_label_overlay else None,
        "context_months": [path.name.removeprefix("month=") for path in context_months],
        "continuous_context_sidecar": str(args.continuous_context_sidecar) if args.continuous_context_sidecar else None,
        "continuous_context_contract": (
            "strict-prequential stable continuous observables and relationship breaks; "
            "fold-local latent state coordinates prohibited"
            if args.continuous_context_sidecar else None
        ),
        "candidate_meta_context_sidecar": (
            str(args.candidate_meta_context_sidecar) if args.candidate_meta_context_sidecar else None
        ),
        "candidate_meta_context_contract": (
            "target-free exact170 candidate-specific fields with declared meta-key ownership; "
            "identity/timestamp matched and equal-status MDA candidates"
            if args.candidate_meta_context_sidecar else None
        ),
        "existing_geometry_features": (
            "all transient memberships and structural fields regenerated from the declared frozen Geometry/K9 bundle" if args.source_panel
            else "not materialized; source-panel/geometry-bundle omitted"
        ),
        "source_hashes": {
            # Hash the exact per-month immutable score input.  Fold roots have
            # two supported layouts, so a root-level aggregate file is not a
            # reliable per-fold provenance identifier.
            path.name: _sha(_score_dir(path) / "predictions.parquet") for path in months
        },
        "context_source_hashes": {
            path.name: _sha(_score_dir(path) / "predictions.parquet") for path in context_months
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "correctness_report.json").write_text(json.dumps({
        "one_frozen_geometry_bundle": True,
        "raw_k9_output_columns": present_posterior,
        "one_frozen_geometry_identity_for_posterior": bool(
            args.include_frozen_k9_membership_posterior
        ),
        "cluster_recent_fields": list(cluster_recent_correctness_fields()),
        "cluster_score_conditioned_fields": list(cluster_score_conditioned_correctness_fields()),
        "residual_committee_fields": list(residual_head_state_fields()),
        "future_outcomes_in_scoring_inputs": False,
        "continuous_context_join": bool(args.continuous_context_sidecar),
        "candidate_meta_context_join": bool(args.candidate_meta_context_sidecar),
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
