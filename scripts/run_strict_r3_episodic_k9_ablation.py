#!/usr/bin/env python3
"""Causal ablation: periodically refit K9 only as isolated representations.

This is intentionally *not* a canonical scorer.  It contrasts the canonical
single October--December 2024 geometry with a stronger version of periodic
refitting than the rejected rolling-K9 experiment:

* a geometry parent is fitted once on the three full months before an episode;
* it is immutable for the full scoring episode;
* its definition rows never train the episode correctness model;
* leaf-trust and C3 correctness train only on rows scored under that exact
  geometry hash; and
* a new episode starts cold (upstream-only) rather than borrowing a previous
  K9 representation just because its cluster labels were aligned.

Monthly strict-R3 base/map/conditional-consensus models remain shared and
prequential because they do not consume K9 fields.  Reference rows are
rescored with the current episode bundle only for the causal prior-42-day CDF;
they are never used to train the episode's K9-dependent layers.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    BASE_BLEND_WEIGHT,
    CONSENSUS_BLEND_WEIGHT,
    CORRECTNESS_FLOOR,
    CORRECTNESS_SPAN,
    FOUR_WEEK_DAYS,
    K9_TEMPERATURE_SCALE,
    META_TRAIN_MONTHS,
    MODEL_CAP,
    CorrectnessHead,
    FrozenGeometryK9View,
    LeafTrustBundle,
    _aggregate_state_fields,
    _equal_month_sample,
    _fit_correctness,
    _fit_leaf_trust,
    load_monthly_upstream_bundle,
    persist_monthly_upstream_bundle,
    score_monthly_upstream_bundle,
    train_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    FrozenGeometryK9,
    ScoreReference,
    _file_hash,
    _json_hash,
    _numeric_matrix,
    _require_columns,
    assert_scoring_frame_is_target_free,
    fit_frozen_geometry_k9,
    persist_geometry_bundle,
)


SCHEMA = "strict_r3_episodic_k9_isolated_ablation_v1"
GEOMETRY_SCHEMA = "strict_r3_geometry_k9_episode_isolated_v1"
CONVERSION_SCHEMA = "strict_r3_episodic_k9_conversion_v1"
SIDE = "long"
REFERENCE_DAYS = 42
DEFAULT_DEFINITION_MONTHS = 3
DEFAULT_EPISODE_MONTHS = 3


@dataclass
class EpisodicConversionBundle:
    """All K9-consuming state fitted inside one immutable geometry episode."""

    cutoff: pd.Timestamp
    end_exclusive: pd.Timestamp
    episode_start: pd.Timestamp
    episode_end_exclusive: pd.Timestamp
    base_fields: tuple[str, ...]
    geometry: FrozenGeometryK9View
    leaf_trust: LeafTrustBundle | None
    correctness: CorrectnessHead | None
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cutoff = _utc(self.cutoff)
        self.end_exclusive = _utc(self.end_exclusive)
        self.episode_start = _utc(self.episode_start)
        self.episode_end_exclusive = _utc(self.episode_end_exclusive)
        if self.end_exclusive > self.episode_end_exclusive:
            raise ValueError("episodic conversion crosses its immutable geometry episode")
        if not self.episode_start <= self.cutoff < self.episode_end_exclusive:
            raise ValueError("conversion cutoff is outside its geometry episode")
        if len(self.base_fields) != 120 or len(set(self.base_fields)) != 120:
            raise ValueError("episodic conversion requires the frozen 120-field base contract")
        if not isinstance(self.geometry, FrozenGeometryK9View):
            raise ValueError("episodic conversion requires an immutable persisted K9 geometry")
        if self.correctness is not None and any(
            "k09__cluster_" in name for name in self.correctness.fields
        ):
            raise ValueError("raw K9 memberships are prohibited from episodic correctness")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _month_start(value: pd.Timestamp) -> pd.Timestamp:
    return value.normalize().replace(day=1)


def _month_add(value: pd.Timestamp, months: int) -> pd.Timestamp:
    return (value.tz_convert(None).to_period("M") + months).to_timestamp().tz_localize("UTC")


def _fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"][SIDE]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("episodic K9 ablation requires the frozen 120-field long contract")
    return fields


def _read_target_free(
    source: Path | pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.loc[
            source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)
        ].copy()
    else:
        frame = pd.read_parquet(
            source,
            columns=["candidate_id", "__decision_ts__", "__symbol__", "side_name", *fields],
            filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
        )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq(SIDE)].copy()
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError(f"target-free source is empty or duplicated for {start} to {end}")
    assert_scoring_frame_is_target_free(frame)
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def _initialise_working_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    output = ledger.copy()
    output["teacher_base_rank42"] = output["prequential_base_rank42"]
    aliases = {
        "base_score": "prequential_base_score",
        "base_rank42": "prequential_base_rank42",
        "base_anchor_bps": "prequential_base_anchor_bps",
        "conditional_consensus_rank": "prequential_consensus_rank",
        "upstream": "prequential_upstream",
        "ordinary_shadow_consensus_rank": "prequential_consensus_rank",
        "ordinary_shadow_upstream": "prequential_upstream",
    }
    for target, source in aliases.items():
        output[target] = output[source]
    return output.set_index("candidate_id", drop=False)


def _apply_upstream_scores(working: pd.DataFrame, score: pd.DataFrame) -> None:
    indexed = score.set_index("candidate_id", drop=False)
    missing = indexed.index.difference(working.index)
    if len(missing):
        raise ValueError(f"monthly upstream scores contain {len(missing)} unknown identities")
    mapping = {
        "base_score": "base_score",
        "base_rank42": "base_rank42",
        "prequential_base_rank42": "base_rank42",
        "base_anchor_bps": "base_anchor_bps",
        "conditional_consensus_rank": "conditional_consensus_rank",
        "upstream": "upstream",
        "ordinary_shadow_consensus_rank": "ordinary_shadow_consensus_rank",
        "ordinary_shadow_upstream": "ordinary_shadow_upstream",
    }
    for target, source in mapping.items():
        working.loc[indexed.index, target] = indexed[source].to_numpy()
    working.loc[indexed.index, "stack_is_prequential"] = True


def _attach_scores(raw: pd.DataFrame, upstream: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    output = raw.merge(upstream.loc[:, columns], on="candidate_id", how="left", validate="one_to_one")
    if output[columns[1:]].isna().any().any():
        raise ValueError("upstream score ledger does not fully cover an episodic conversion frame")
    return output


def _build_geometry_population(
    source_cache: pd.DataFrame,
    working: pd.DataFrame,
    *,
    definition_start: pd.Timestamp,
    definition_end: pd.Timestamp,
    fields: Sequence[str],
) -> pd.DataFrame:
    """Join labels after selecting the complete point-in-time definition universe."""

    raw = _read_target_free(
        source_cache, start=definition_start, end=definition_end, fields=fields,
    )
    labels = working.reindex(raw["candidate_id"]).copy()
    required = [
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        "base_anchor_bps", "stack_is_prequential",
    ]
    _require_columns(labels, required, "episodic geometry label ledger")
    for name in required:
        raw[name if name != "base_anchor_bps" else "prequential_base_anchor_bps"] = labels[name].to_numpy()
    raw["geometry_definition_population_complete"] = True
    # Labels are deliberately not candidate inputs.  They are visible only to
    # the supervised geometry fit after the complete decision-time population
    # has been materialised and after their availability predicate is applied
    # by ``fit_frozen_geometry_k9``.
    return raw


def _persist_episode_bundle(bundle: EpisodicConversionBundle, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / "episodic_conversion_bundle.joblib"
    joblib.dump(bundle, payload, compress=3)
    digest = _file_hash(payload)
    manifest = {**bundle.manifest, "schema": CONVERSION_SCHEMA, "bundle_file": payload.name, "bundle_sha256": digest}
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    bundle.manifest["bundle_sha256"] = digest
    return digest


def _fit_episode_bundle(
    *,
    cutoff: pd.Timestamp,
    held_end: pd.Timestamp,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    upstream_ledger: pd.DataFrame,
    geometry: FrozenGeometryK9View,
    base_fields: Sequence[str],
    geometry_parent_sha256: str,
) -> EpisodicConversionBundle:
    fields = tuple(map(str, base_fields))
    _require_columns(
        upstream_ledger,
        [
            "candidate_id", "__decision_ts__", "r3_class", "r3_label_available_ts",
            "policy_net_bps", "policy_label_available_ts", "h12_label_valid",
            "h12_label_available_ts", "h12_tp6_sl4_net_bps", "base_score",
            "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
            "stack_is_prequential", *fields,
        ],
        "episodic K9 conversion ledger",
    )
    ledger = upstream_ledger.copy()
    for name in (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
        "h12_label_available_ts",
    ):
        ledger[name] = pd.to_datetime(ledger[name], utc=True, errors="raise")
    if not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("episodic conversion received non-prequential upstream rows")
    # This is the essential isolation condition: definitions lie before
    # episode_start, and no training row may have another geometry identity.
    episode_rows = ledger.loc[
        ledger["__decision_ts__"].ge(episode_start)
        & ledger["__decision_ts__"].lt(cutoff)
    ].copy()
    meta = episode_rows.loc[
        episode_rows["policy_label_available_ts"].lt(cutoff)
        & np.isfinite(pd.to_numeric(episode_rows["policy_net_bps"], errors="coerce"))
    ].copy()
    state = None
    leaf: LeafTrustBundle | None = None
    correctness: CorrectnessHead | None = None
    warmup_reason: str | None = None
    try:
        if len(episode_rows) < 1_000 or len(meta) < 1_000:
            raise ValueError("insufficient same-geometry episode support")
        leaf = _fit_leaf_trust(episode_rows, fields, cutoff)
        state = pd.concat([geometry.transform(meta), leaf.transform(meta)], axis=1)
        aggregate = _aggregate_state_fields(state)
        meta = pd.concat([meta.reset_index(drop=True), state.loc[:, list(aggregate)].reset_index(drop=True)], axis=1)
        meta = _equal_month_sample(meta, MODEL_CAP, seed=20260817 + 5001)
        correctness_fields = (
            "base_score", "base_anchor_bps", "base_rank42",
            "conditional_consensus_rank", "upstream", *aggregate,
        )
        correctness = _fit_correctness(meta, correctness_fields)
    except ValueError as exc:
        # We do not borrow old-geometry examples just to make the first block
        # look fully trained.  The upstream-only state is an explicit and
        # auditable valid warm-up.
        leaf = None
        correctness = None
        warmup_reason = str(exc)
    manifest = {
        "schema": CONVERSION_SCHEMA,
        "cutoff": cutoff.isoformat(),
        "end_exclusive": held_end.isoformat(),
        "episode_start": episode_start.isoformat(),
        "episode_end_exclusive": episode_end.isoformat(),
        "geometry_parent_bundle_sha256": geometry_parent_sha256,
        "geometry_bundle_sha256": geometry.bundle_sha256,
        "geometry_temperature_scale": K9_TEMPERATURE_SCALE,
        "geometry_refit_cadence": "episode_boundary_only",
        "geometry_training_identity": geometry.bundle_sha256,
        "definition_rows_excluded_from_downstream_training": True,
        "downstream_training_geometry_scope": "same_geometry_bundle_only",
        "reference_geometry_scope": "same_current_bundle_rescore_only",
        "raw_k9_used_by_correctness": False,
        "correctness_status": "complete" if correctness is not None else "upstream_only_warmup",
        "correctness_warmup_reason": warmup_reason,
    }
    return EpisodicConversionBundle(
        cutoff, held_end, episode_start, episode_end, fields, geometry, leaf, correctness, manifest,
    )


def _score_episode_bundle(
    bundle: EpisodicConversionBundle,
    *,
    reference: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    upstream_fields = [
        "base_score", "base_rank42", "base_anchor_bps", "conditional_consensus_rank",
        "upstream", "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    for role, frame in (("reference", reference), ("held", held)):
        _require_columns(
            frame, ["candidate_id", "__decision_ts__", "side_name", *upstream_fields, *bundle.base_fields],
            f"episodic conversion {role}",
        )
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"episodic conversion {role} has duplicate identities")
        assert_scoring_frame_is_target_free(frame)
    reference = reference.copy()
    held = held.copy()
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    held["__decision_ts__"] = pd.to_datetime(held["__decision_ts__"], utc=True)
    if not reference["__decision_ts__"].between(
        bundle.cutoff - pd.Timedelta(days=REFERENCE_DAYS), bundle.cutoff, inclusive="left",
    ).all():
        raise ValueError("episodic reference is not the preceding 42 days")
    if not held["__decision_ts__"].between(bundle.cutoff, bundle.end_exclusive, inclusive="left").all():
        raise ValueError("episodic held rows fall outside their conversion block")
    combined = pd.concat(
        [reference.assign(__score_role__="reference"), held.assign(__score_role__="held")],
        ignore_index=True,
    ).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if bundle.correctness is None or bundle.leaf_trust is None:
        combined["correctness_raw"] = np.nan
        combined["correctness_rank"] = np.float32(0.5)
        combined["correctness_gate_active"] = False
        combined["raw_correctness_demote"] = combined["upstream"].to_numpy(float)
    else:
        state = pd.concat([bundle.geometry.transform(combined), bundle.leaf_trust.transform(combined)], axis=1)
        aggregate = _aggregate_state_fields(state)
        combined = pd.concat([combined, state.loc[:, list(aggregate)].reset_index(drop=True)], axis=1)
        raw = bundle.correctness.model.predict(
            _numeric_matrix(combined, bundle.correctness.fields, bundle.correctness.medians),
        )
        combined["correctness_raw"] = raw
        combined["correctness_rank"] = bundle.correctness.score_reference.cdf(raw)
        gate = combined["upstream"].ge(bundle.correctness.training_score_floor).to_numpy(bool)
        combined["correctness_gate_active"] = gate
        multiplier = CORRECTNESS_FLOOR + CORRECTNESS_SPAN * combined["correctness_rank"].to_numpy(float)
        combined["raw_correctness_demote"] = combined["upstream"].to_numpy(float) * np.where(gate, multiplier, 1.0)
    reference_mask = combined["__score_role__"].eq("reference").to_numpy()
    cdf = ScoreReference.fit(
        combined.loc[reference_mask, "raw_correctness_demote"],
        source="same_episode_conversion_model_prior42",
    )
    combined["final_score"] = cdf.cdf(combined["raw_correctness_demote"])
    combined["geometry_bundle_sha256"] = bundle.geometry.bundle_sha256
    combined["geometry_parent_bundle_sha256"] = bundle.geometry.parent_bundle_sha256
    combined["geometry_episode_start"] = bundle.episode_start
    combined["geometry_episode_end_exclusive"] = bundle.episode_end_exclusive
    combined["conversion_bundle_sha256"] = bundle.manifest.get("bundle_sha256", "unpersisted")
    columns = [
        "candidate_id", "__decision_ts__", *( ["__symbol__"] if "__symbol__" in combined else []),
        "side_name", *upstream_fields, "correctness_raw", "correctness_rank",
        "correctness_gate_active", "raw_correctness_demote", "final_score",
        "conversion_bundle_sha256", "geometry_bundle_sha256", "geometry_parent_bundle_sha256",
        "geometry_episode_start", "geometry_episode_end_exclusive", "__score_role__",
    ]
    audit = {
        "reference_rows": int(reference_mask.sum()),
        "held_rows": int((~reference_mask).sum()),
        "same_current_geometry_for_reference_and_held": True,
        "held_percentile_operations": 0,
        "downstream_training_geometry_scope": "same_geometry_bundle_only",
        "correctness_status": bundle.manifest["correctness_status"],
    }
    return combined.loc[:, columns].copy(), audit


def _attach_outcomes_after_scoring(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price", "policy_label_available_ts",
        "policy_outcome_source",
    ]
    available = [name for name in columns if name in outcomes]
    required = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    if required.difference(available):
        raise ValueError(f"outcome ledger lacks: {sorted(required.difference(available))}")
    joined = predictions.merge(
        outcomes.loc[:, ["candidate_id", *available]], on="candidate_id", how="left", validate="one_to_one",
    )
    if len(joined) != len(predictions) or joined["candidate_id"].duplicated().any():
        raise AssertionError("post-score outcome join changed episodic prediction identities")
    return joined, available


def _acquire_run_lock(directory: Path) -> Any:
    handle = (directory / ".episodic_k9.run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(f"episodic K9 output already has an active writer: {owner}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "out_dir": str(directory)}) + "\n")
    handle.flush()
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument(
        "--reuse-upstream-dir", type=Path,
        help=(
            "Optional completed frozen-control upstream directory. Only the "
            "strict-R3 base/map/ten-head producer may be reused because it "
            "does not consume K9. Hashes are verified; all K9-dependent "
            "conversion/reliability/admission state is still rebuilt per episode."
        ),
    )
    parser.add_argument("--outcome-ledger", type=Path)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--episode-months", type=int, default=DEFAULT_EPISODE_MONTHS, choices=(3, 6))
    parser.add_argument("--geometry-definition-months", type=int, default=DEFAULT_DEFINITION_MONTHS, choices=(3,))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"episodic-K9 output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    lock = _acquire_run_lock(args.out_dir)
    atexit.register(lock.close)

    fields = _fields(args.feature_contract)
    evaluation_start, evaluation_end = _utc(args.evaluation_start), _utc(args.evaluation_end)
    if evaluation_start != _month_start(evaluation_start):
        raise ValueError("episodic K9 ablation must start on a UTC calendar month")
    # Never materialise later history merely to discard it.  It cannot enter
    # a prequential fit or reference before this evaluation end.
    ledger = pd.read_parquet(
        args.prequential_ledger,
        filters=[("__decision_ts__", "<", evaluation_end)],
    )
    ledger = ledger.loc[ledger["side_name"].astype(str).str.lower().eq(SIDE)].copy()
    for name in ("__decision_ts__", "r3_label_available_ts", "policy_label_available_ts", "h12_label_available_ts"):
        ledger[name] = pd.to_datetime(ledger[name], utc=True)
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("prequential source ledger has duplicate candidate IDs")
    working = _initialise_working_ledger(ledger)
    source_hashes = {
        "source_panel": _sha(args.source_panel),
        "prequential_ledger": _sha(args.prequential_ledger),
        "feature_contract": _sha(args.feature_contract),
    }

    upstream_start = _month_add(_month_start(evaluation_start), -META_TRAIN_MONTHS - 1)
    final_upstream_month = _month_start(evaluation_end - pd.Timedelta(nanoseconds=1))
    first_definition = _month_add(evaluation_start, -args.geometry_definition_months)
    source_floor = min(upstream_start - pd.Timedelta(days=REFERENCE_DAYS), first_definition)
    source_cache = _read_target_free(args.source_panel, start=source_floor, end=evaluation_end, fields=fields)

    upstream_scores: list[pd.DataFrame] = []
    # The conversion prior-42 distribution is always rescored by the current
    # monthly upstream producer.  Older monthly OOF scores remain training
    # inputs only; they are never a mixed-model calibration reference.
    upstream_bundles: dict[pd.Timestamp, object] = {}
    upstream_audit: list[dict[str, Any]] = []
    upstream_months = pd.date_range(
        upstream_start, final_upstream_month, freq="MS", inclusive="both",
    )
    reused_upstream: dict[str, Any] | None = None
    if args.reuse_upstream_dir is not None:
        reuse_root = args.reuse_upstream_dir
        reuse_manifest_path = reuse_root / "run_manifest.json"
        reuse_manifest = json.loads(reuse_manifest_path.read_text())
        reuse_hashes = reuse_manifest.get("source_hashes", {})
        for name, expected in source_hashes.items():
            if reuse_hashes.get(name) != expected:
                raise ValueError(
                    f"reused upstream source hash mismatch for {name}: "
                    f"{reuse_hashes.get(name)} != {expected}",
                )
        upstream_path = reuse_root / "monthly_upstream_predictions.parquet"
        if not upstream_path.exists():
            raise FileNotFoundError("reused upstream directory lacks monthly_upstream_predictions.parquet")
        upstream = pd.read_parquet(upstream_path)
        upstream["__decision_ts__"] = pd.to_datetime(upstream["__decision_ts__"], utc=True)
        expected_start, expected_end = upstream_start, final_upstream_month + pd.offsets.MonthBegin(1)
        upstream = upstream.loc[
            upstream["__decision_ts__"].ge(expected_start)
            & upstream["__decision_ts__"].lt(expected_end)
        ].copy()
        if upstream.empty or upstream["candidate_id"].duplicated().any():
            raise ValueError("reused upstream predictions are empty or duplicate")
        for month in upstream_months:
            bundle_dir = reuse_root / "upstream_bundles" / f"month={month:%Y-%m}"
            bundle = load_monthly_upstream_bundle(bundle_dir)
            if tuple(bundle.base_fields) != tuple(fields):
                raise ValueError("reused upstream bundle has a different frozen base feature contract")
            upstream_bundles[pd.Timestamp(month)] = bundle
            held = upstream.loc[
                upstream["__decision_ts__"].ge(month)
                & upstream["__decision_ts__"].lt(month + pd.offsets.MonthBegin(1))
            ]
            if held.empty:
                raise ValueError(f"reused upstream scores lack held month {month:%Y-%m}")
            upstream_audit.append({
                "month": month, "rows": len(held),
                "bundle_sha256": bundle.manifest["bundle_sha256"], "status": "reused_hash_verified",
            })
        _apply_upstream_scores(working, upstream)
        reused_upstream = {
            "path": str(reuse_root),
            "run_manifest_sha256": _sha(reuse_manifest_path),
            "monthly_predictions_sha256": _sha(upstream_path),
            "semantics": "base/map/consensus only; no K9-dependent state reused",
        }
    else:
        for month in upstream_months:
            month_end = month + pd.offsets.MonthBegin(1)
            held_raw = _read_target_free(source_cache, start=month, end=month_end, fields=fields)
            prior42 = _read_target_free(source_cache, start=month - pd.Timedelta(days=REFERENCE_DAYS), end=month, fields=fields)
            prior = working.loc[working["__decision_ts__"].lt(month)].copy().reset_index(drop=True)
            bundle = train_monthly_upstream_bundle(
                cutoff=month, training_ledger=prior, prior42_features=prior42,
                base_fields=fields, source_hashes=source_hashes,
            )
            bundle_dir = args.out_dir / "upstream_bundles" / f"month={month:%Y-%m}"
            manifest = persist_monthly_upstream_bundle(bundle, bundle_dir)
            score = score_monthly_upstream_bundle(bundle, held_raw)
            upstream_bundles[pd.Timestamp(month)] = bundle
            _apply_upstream_scores(working, score)
            upstream_scores.append(score)
            upstream_audit.append({"month": month, "rows": len(score), "bundle_sha256": manifest["bundle_sha256"], "status": "fit"})
            print(json.dumps({"event": "upstream_month_complete", **upstream_audit[-1]}, default=str), flush=True)
        upstream = pd.concat(upstream_scores, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        upstream.to_parquet(args.out_dir / "monthly_upstream_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(upstream_audit).to_parquet(args.out_dir / "monthly_upstream_audit.parquet", index=False)

    predictions: list[pd.DataFrame] = []
    geometry_audit: list[dict[str, Any]] = []
    block_audit: list[dict[str, Any]] = []
    episode_start = evaluation_start
    episode_index = 0
    while episode_start < evaluation_end:
        episode_end = min(_month_add(episode_start, args.episode_months), evaluation_end)
        definition_start = _month_add(episode_start, -args.geometry_definition_months)
        definition_end = episode_start
        warmup = _build_geometry_population(
            source_cache, working, definition_start=definition_start, definition_end=definition_end, fields=fields,
        )
        geometry = fit_frozen_geometry_k9(
            warmup, encoder_fields=fields, seed=20260817 + 10_000 + episode_index,
            definition_start=definition_start, definition_end_exclusive=definition_end,
        )
        geometry.fit_audit.update({
            "schema_role": "episodic_k9_isolated_ablation",
            "episode_start": episode_start.isoformat(),
            "episode_end_exclusive": episode_end.isoformat(),
            "downstream_training_scope": "same_geometry_bundle_only",
            "definition_rows_excluded_from_downstream_training": True,
            "source_hashes": source_hashes,
        })
        geometry_dir = args.out_dir / "geometry_bundles" / f"episode={episode_start:%Y%m%d}"
        geometry_manifest = persist_geometry_bundle(geometry, geometry_dir, schema=GEOMETRY_SCHEMA)
        view = FrozenGeometryK9View(geometry)
        geometry_audit.append({
            "episode_index": episode_index, "episode_start": episode_start,
            "episode_end_exclusive": episode_end, "definition_start": definition_start,
            "definition_end_exclusive": definition_end, "parent_bundle_sha256": geometry.bundle_sha256,
            "view_bundle_sha256": view.bundle_sha256, "definition_rows": geometry.fit_audit["complete_warmup_rows"],
            "geometry_manifest_sha256": _sha(geometry_dir / "run_manifest.json"),
        })
        print(json.dumps({"event": "geometry_episode_complete", **geometry_audit[-1]}, default=str), flush=True)

        for cutoff in pd.date_range(episode_start, episode_end, freq=f"{FOUR_WEEK_DAYS}D", inclusive="left"):
            held_end = min(cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS), episode_end, evaluation_end)
            prior = working.loc[working["__decision_ts__"].lt(cutoff)].copy().reset_index(drop=True)
            bundle = _fit_episode_bundle(
                cutoff=cutoff, held_end=held_end, episode_start=episode_start, episode_end=episode_end,
                upstream_ledger=prior, geometry=view, base_fields=fields,
                geometry_parent_sha256=geometry.bundle_sha256,
            )
            bundle_dir = args.out_dir / "conversion_bundles" / f"cutoff={cutoff:%Y%m%d}"
            bundle_hash = _persist_episode_bundle(bundle, bundle_dir)
            reference_raw = _read_target_free(source_cache, start=cutoff - pd.Timedelta(days=REFERENCE_DAYS), end=cutoff, fields=fields)
            held_raw = _read_target_free(source_cache, start=cutoff, end=held_end, fields=fields)
            current_month = _month_start(cutoff)
            current_upstream = upstream_bundles.get(current_month)
            if current_upstream is None:
                raise AssertionError(f"missing current monthly upstream bundle for {current_month}")
            reference_score = score_monthly_upstream_bundle(
                current_upstream, reference_raw, allow_prior_reference=True,
            )
            reference = _attach_scores(reference_raw, reference_score)
            held = _attach_scores(held_raw, upstream)
            scored, score_audit = _score_episode_bundle(bundle, reference=reference, held=held)
            predictions.append(scored.loc[scored["__score_role__"].eq("held")].drop(columns="__score_role__"))
            block_audit.append({
                "episode_index": episode_index, "cutoff": cutoff, "held_end_exclusive": held_end,
                "episode_start": episode_start, "episode_end_exclusive": episode_end,
                "conversion_bundle_sha256": bundle_hash, "geometry_bundle_sha256": view.bundle_sha256,
                "geometry_parent_bundle_sha256": geometry.bundle_sha256,
                "geometry_refit_cadence": "episode_boundary_only",
                "reference_upstream_bundle_sha256": current_upstream.manifest["bundle_sha256"],
                "reference_upstream_rescored_same_current_bundle": True,
                "downstream_training_geometry_scope": "same_geometry_bundle_only",
                "definition_rows_excluded_from_downstream_training": True,
                **score_audit,
            })
            print(json.dumps({"event": "conversion_block_complete", **block_audit[-1]}, default=str), flush=True)
        episode_start = episode_end
        episode_index += 1

    final = pd.concat(predictions, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if final["candidate_id"].duplicated().any():
        raise AssertionError("episodic K9 output duplicated candidate IDs")
    final["base_rank"] = pd.to_numeric(final["base_rank42"], errors="coerce")
    final["consensus_rank"] = pd.to_numeric(final["conditional_consensus_rank"], errors="coerce")
    final["stack_is_prequential"] = True
    # This ablation has no active Severe-200 scorer.  State this explicitly so
    # the common current-v5 portfolio producer can reject an accidental active
    # demotion, rather than inferring the contract from a missing column.
    final["severe200_probability_shadow"] = np.nan
    final["severe_affects_final_score"] = False
    final.to_parquet(args.out_dir / "walkforward_predictions.parquet", index=False, compression="zstd")
    outcome_columns: list[str] = []
    if args.outcome_ledger is not None:
        outcome = pd.read_parquet(args.outcome_ledger)
        scored_labels, outcome_columns = _attach_outcomes_after_scoring(final, outcome)
        scored_labels.to_parquet(args.out_dir / "walkforward_scored_label_ledger.parquet", index=False, compression="zstd")
    pd.DataFrame(geometry_audit).to_parquet(args.out_dir / "geometry_episode_audit.parquet", index=False)
    pd.DataFrame(block_audit).to_parquet(args.out_dir / "conversion_block_audit.parquet", index=False)
    manifest = {
        "schema": SCHEMA, "side": SIDE, "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(), "rows": len(final),
        "geometry_definition_months": args.geometry_definition_months,
        "geometry_episode_months": args.episode_months,
        "geometry_refit_cadence": "episode_boundary_only",
        "geometry_compatibility": "K9-dependent downstream fitting is same-geometry-only",
        "definition_rows_excluded_from_downstream_training": True,
        "base_and_consensus": "monthly strict-prequential shared; neither consumes K9 outputs",
        "upstream_reuse": reused_upstream,
        "conversion": "episode-isolated leaf/C3; explicit upstream-only warm-up where same-geometry support is inadequate",
        "normalization": (
            "same current episode conversion-model prior-42-day CDF; reference "
            "base/consensus scores rescored by the current monthly producer"
        ),
        "held_percentile_operations": 0,
        "raw_k9_used_by_correctness": False,
        "severe200": "not fitted in the episodic-K9 ablation; explicit inactive shadow field",
        "outcomes_consumed_during_scoring": [],
        "outcome_join": {"performed_after_scoring": args.outcome_ledger is not None, "columns": outcome_columns},
        "source_hashes": source_hashes,
        "comparison_role": "research ablation only; cannot replace the frozen schema-v5 control without matched metrics",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}, default=str), flush=True)


if __name__ == "__main__":
    main()
