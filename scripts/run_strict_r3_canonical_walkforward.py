#!/usr/bin/env python3
"""Produce the exact current strict-R3 walk-forward cadence, long only.

Monthly D2/conditional-consensus upstream bundles are fitted first.  Their
strict-prequential scores feed a separate C3 correctness bundle refit every
four weeks.  This matches the validated research topology; no monolithic
four-week refit of the base/consensus layer is permitted.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    CALIBRATION_RESERVE_DAYS,
    CORRECTNESS_TRAIN_FRACTION,
    FOUR_WEEK_DAYS,
    K9_TEMPERATURE_SCALE,
    META_TRAIN_MONTHS,
    REFERENCE_DAYS,
    SCHEMA,
    persist_four_week_conversion_bundle,
    persist_monthly_upstream_bundle,
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    score_four_week_conversion_by_upstream_vintage,
    score_monthly_upstream_bundle,
    train_four_week_conversion_bundle,
    train_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    assert_scoring_frame_is_target_free,
    load_geometry_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _month_start(value: pd.Timestamp) -> pd.Timestamp:
    return value.normalize().replace(day=1)


def _month_add(value: pd.Timestamp, months: int) -> pd.Timestamp:
    return (value.tz_convert(None).to_period("M") + months).to_timestamp().tz_localize("UTC")


def _fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"]["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("schema-v4 requires the frozen 120-field long contract")
    return fields


def _read_target_free(
    source: Path | pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: list[str],
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
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq("long")].copy()
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


def _attach_scores(raw: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
    ]
    output = raw.merge(score[columns], on="candidate_id", how="left", validate="one_to_one")
    if output[columns[1:]].isna().any().any():
        raise ValueError("upstream score ledger does not fully cover a conversion score frame")
    return output


def _attach_outcomes_after_scoring(
    predictions: pd.DataFrame,
    outcome_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Join evaluation labels only after all prediction artifacts are frozen.

    This producer is used for reproducible OOF evaluation, not live scoring.
    Keeping the join at this boundary makes it impossible for policy outcomes
    to enter the monthly base, conversion, reference CDF, or candidate inputs.
    """

    if "candidate_id" not in outcome_ledger or outcome_ledger["candidate_id"].duplicated().any():
        raise ValueError("outcome ledger requires unique candidate_id")
    columns = [
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    ]
    available = [column for column in columns if column in outcome_ledger.columns]
    required = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"outcome ledger lacks required policy fields: {missing}")
    joined = predictions.merge(
        outcome_ledger.loc[:, ["candidate_id", *available]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    if len(joined) != len(predictions) or joined["candidate_id"].duplicated().any():
        raise AssertionError("post-score outcome join changed prediction identities")
    return joined, available


def _load_resumable_bundle(
    directory: Path,
    *,
    loader: object,
    cutoff: pd.Timestamp,
    source_hashes: dict[str, str],
    kind: str,
) -> object:
    """Load an immutable checkpoint only when its lineage exactly matches."""

    manifest_path = directory / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {kind} resume manifest: {manifest_path}")
    bundle = loader(directory)  # type: ignore[operator]
    observed_cutoff = pd.Timestamp(bundle.cutoff)
    if observed_cutoff != cutoff:
        raise ValueError(
            f"{kind} resume bundle cutoff {observed_cutoff} does not match {cutoff}",
        )
    if dict(bundle.manifest.get("source_hashes", {})) != dict(source_hashes):
        raise ValueError(f"{kind} resume bundle source hashes do not match this run")
    return bundle


def _acquire_run_lock(directory: Path) -> object:
    """Prevent two writers from mutating one immutable replay root.

    ``flock`` is kernel-scoped, so it is released even after an interrupted
    process.  The tiny owner record is diagnostic only; it is deliberately
    outside the immutable monthly/conversion bundle directories.
    """

    lock_path = directory / ".walkforward.run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"walk-forward output is already actively owned: {directory} ({owner})",
        ) from exc
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
        "--geometry-bundle", type=Path, required=True,
        help="One persisted target-free Oct-Dec 2024 geometry/K9 definition.",
    )
    parser.add_argument(
        "--outcome-ledger", type=Path,
        help=(
            "Optional evaluation-only policy label ledger. It is joined only "
            "after the target-free OOF score artifact has been completed."
        ),
    )
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-reserve-days", type=int, default=CALIBRATION_RESERVE_DAYS,
        help=(
            "Pre-cutoff target-free tail reserved from every supervised fit. "
            "The newly fitted full bundle scores this OOS tail immediately for "
            "its per-refit EV calibration. Zero is legacy-only."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume only hash-matched immutable monthly/four-week bundle checkpoints.",
    )
    args = parser.parse_args()
    if not 0 <= args.calibration_reserve_days <= 42:
        raise ValueError("--calibration-reserve-days must lie in [0, 42]")
    if args.out_dir.exists():
        if not args.resume:
            raise FileExistsError(
                f"walk-forward output already exists: {args.out_dir}; use --resume only "
                "for hash-matched partial checkpoints",
            )
        if (args.out_dir / "run_manifest.json").exists():
            raise FileExistsError("cannot resume a finalized immutable walk-forward output")
    else:
        args.out_dir.mkdir(parents=True)

    # Keep the handle reachable for the entire producer lifetime.  This closes
    # on ordinary exits and is automatically released by the kernel on a crash.
    run_lock = _acquire_run_lock(args.out_dir)
    atexit.register(run_lock.close)

    fields = _fields(args.feature_contract)
    evaluation_start, evaluation_end = _utc(args.evaluation_start), _utc(args.evaluation_end)
    frozen_geometry = load_geometry_bundle(args.geometry_bundle)
    geometry_audit = frozen_geometry.fit_audit
    if (
        geometry_audit.get("definition_start") != "2024-10-01T00:00:00+00:00"
        or geometry_audit.get("definition_end_exclusive") != "2025-01-01T00:00:00+00:00"
    ):
        raise ValueError("walk-forward requires the frozen Oct-Dec 2024 geometry/K9 definition")
    # Later rows cannot contribute to any training, reference, or evaluation
    # operation in this replay.  Predicate-push this bound into Parquet rather
    # than materialising the whole multi-year ledger and filtering afterwards.
    ledger = pd.read_parquet(
        args.prequential_ledger,
        filters=[("__decision_ts__", "<", evaluation_end)],
    )
    ledger = ledger.loc[ledger["side_name"].astype(str).str.lower().eq("long")].copy()
    for column in (
        "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts",
        "h12_label_available_ts",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True)
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("prequential source ledger has duplicate candidate IDs")
    working = _initialise_working_ledger(ledger)
    source_hashes = {
        "source_panel": _sha(args.source_panel),
        "prequential_ledger": _sha(args.prequential_ledger),
        "feature_contract": _sha(args.feature_contract),
        "geometry_manifest": _sha(args.geometry_bundle / "run_manifest.json"),
        "calibration_reserve_days": str(args.calibration_reserve_days),
    }

    # Six complete months are needed before the first conversion fit; an
    # additional prior month supplies the earliest monthly base's canonical
    # reference window
    # reference without using the evaluation window.
    upstream_start = _month_add(_month_start(evaluation_start), -META_TRAIN_MONTHS - 1)
    final_upstream_month = _month_start(evaluation_end - pd.Timedelta(nanoseconds=1))
    # The prior producer read the same monolithic source parquet once for each
    # monthly/reference/geometry slice.  Load the bounded range once instead:
    # this is byte-identical to those slices but removes repeated 120-column
    # scans and is essential for a practical production-parity replay.
    source_floor = upstream_start - pd.Timedelta(days=REFERENCE_DAYS)
    source_cache = _read_target_free(
        args.source_panel, start=source_floor, end=evaluation_end, fields=fields,
    )
    upstream_months = pd.date_range(
        upstream_start, final_upstream_month, freq="MS", inclusive="both",
    )
    upstream_scores: list[pd.DataFrame] = []
    # Retain each monthly producer so that every conversion CDF reference can
    # be rescored by the *current* monthly base/consensus bundle.  Joining the
    # old monthly score ledger here would mix score domains across the active
    # reference window
    # reference window even though no labels leak.
    upstream_bundles: dict[pd.Timestamp, object] = {}
    upstream_audit: list[dict[str, object]] = []
    for month in upstream_months:
        month_end = month + pd.offsets.MonthBegin(1)
        held_raw = _read_target_free(
            source_cache, start=month, end=month_end, fields=fields,
        )
        prior42 = _read_target_free(
            source_cache,
            start=month - pd.Timedelta(days=REFERENCE_DAYS),
            end=month,
            fields=fields,
        )
        prior = working.loc[working["__decision_ts__"].lt(month)].copy().reset_index(drop=True)
        bundle_dir = args.out_dir / "upstream_bundles" / f"month={month:%Y-%m}"
        if args.resume and (bundle_dir / "run_manifest.json").exists():
            bundle = _load_resumable_bundle(
                bundle_dir, loader=load_monthly_upstream_bundle, cutoff=month,
                source_hashes=source_hashes, kind="monthly upstream",
            )
            upstream_status = "resumed"
        else:
            bundle = train_monthly_upstream_bundle(
                cutoff=month,
                training_ledger=prior,
                prior42_features=prior42,
                base_fields=fields,
                source_hashes=source_hashes,
                calibration_reserve_days=args.calibration_reserve_days,
            )
            persist_monthly_upstream_bundle(bundle, bundle_dir)
            upstream_status = "complete"
        score = score_monthly_upstream_bundle(bundle, held_raw)
        upstream_bundles[pd.Timestamp(month)] = bundle
        _apply_upstream_scores(working, score)
        upstream_scores.append(score)
        upstream_audit.append({
            "month": month.strftime("%Y-%m"),
            "rows": len(score),
            "bundle_sha256": bundle.manifest["bundle_sha256"],
            "status": upstream_status,
        })
        print(json.dumps({"event": "upstream_month_complete", **upstream_audit[-1]}), flush=True)

    upstream = pd.concat(upstream_scores, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    upstream.to_parquet(
        args.out_dir / "monthly_upstream_predictions.parquet", index=False, compression="zstd",
    )
    pd.DataFrame(upstream_audit).to_parquet(
        args.out_dir / "monthly_upstream_audit.parquet", index=False,
    )

    predictions: list[pd.DataFrame] = []
    calibration_reference_scores: list[pd.DataFrame] = []
    conversion_audit: list[pd.DataFrame] = []
    block_rows: list[dict[str, object]] = []
    cutoffs = pd.date_range(
        evaluation_start, evaluation_end, freq=f"{FOUR_WEEK_DAYS}D", inclusive="left",
    )
    for index, cutoff in enumerate(cutoffs):
        held_end = min(cutoff + pd.Timedelta(days=FOUR_WEEK_DAYS), evaluation_end)
        prior = working.loc[working["__decision_ts__"].lt(cutoff)].copy().reset_index(drop=True)
        bundle_dir = args.out_dir / "conversion_bundles" / f"cutoff={cutoff:%Y%m%d}"
        if args.resume and (bundle_dir / "run_manifest.json").exists():
            bundle = _load_resumable_bundle(
                bundle_dir, loader=load_four_week_conversion_bundle, cutoff=cutoff,
                source_hashes=source_hashes, kind="four-week conversion",
            )
            conversion_status = "resumed"
        else:
            bundle = train_four_week_conversion_bundle(
                cutoff=cutoff,
                upstream_ledger=prior,
                frozen_geometry=frozen_geometry,
                base_fields=fields,
                source_hashes=source_hashes,
                calibration_reserve_days=args.calibration_reserve_days,
            )
            persist_four_week_conversion_bundle(bundle, bundle_dir)
            conversion_status = "complete"
        reference_raw = _read_target_free(
            source_cache,
            start=cutoff - pd.Timedelta(days=REFERENCE_DAYS),
            end=cutoff,
            fields=fields,
        )
        held_raw = _read_target_free(
            source_cache, start=cutoff, end=held_end, fields=fields,
        )
        held_months = sorted(held_raw["__decision_ts__"].dt.strftime("%Y-%m").unique().tolist())
        block_upstreams = {
            token: upstream_bundles[_month_start(pd.Timestamp(token + "-01", tz="UTC"))]
            for token in held_months
        }
        if len(block_upstreams) != len(held_months):
            raise AssertionError("conversion block is missing a held-month upstream bundle")
        scored, audit = score_four_week_conversion_by_upstream_vintage(
            bundle,
            reference=reference_raw,
            held=held_raw,
            upstream_bundles=block_upstreams,
        )
        held_score = scored.loc[scored["__score_role__"].eq("held")].drop(columns="__score_role__")
        calibration_reference_scores.append(
            scored.loc[scored["__score_role__"].eq("reference")].copy(),
        )
        predictions.append(held_score)
        conversion_audit.append(audit.assign(block_index=index))
        block_rows.append({
            "block_index": index,
            "cutoff": cutoff,
            "held_end_exclusive": held_end,
            "held_rows": len(held_score),
            "reference_rows": len(reference_raw),
            "conversion_bundle_sha256": bundle.manifest["bundle_sha256"],
            "geometry_bundle_sha256": bundle.geometry.bundle_sha256,
            "geometry_parent_bundle_sha256": bundle.geometry.parent_bundle_sha256,
            "geometry_refit_cadence": "never",
            "reference_upstream_bundle_sha256_by_held_month": {
                token: str(model.manifest["bundle_sha256"])
                for token, model in block_upstreams.items()
            },
            "same_upstream_bundle_for_reference_and_held_per_producer": True,
            "status": conversion_status,
        })
        print(json.dumps({"event": "conversion_block_complete", **block_rows[-1]}, default=str), flush=True)

    final = pd.concat(predictions, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    if final["candidate_id"].duplicated().any():
        raise AssertionError("canonical walk-forward duplicated candidate IDs")
    # Stable aliases are the compact LDF/reliability contract.  The aliases do
    # not change score semantics: they make the current schema-v4 producer
    # consumable by the same target-free feature materialiser at replay and
    # inference time.
    final["base_rank"] = pd.to_numeric(final["base_rank42"], errors="coerce")
    final["consensus_rank"] = pd.to_numeric(
        final["conditional_consensus_rank"], errors="coerce",
    )
    final["stack_is_prequential"] = True
    final.to_parquet(args.out_dir / "walkforward_predictions.parquet", index=False, compression="zstd")
    calibration_reference = pd.concat(
        calibration_reference_scores, ignore_index=True,
    ).sort_values(
        ["calibration_activation_ts", "upstream_bundle_sha256", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    # A candidate can legitimately appear in several reference cohorts.  Its
    # producer pair is part of the calibration identity, so duplicate raw
    # candidate IDs are expected and never used as a global score ledger.
    calibration_reference.to_parquet(
        args.out_dir / "immediate_calibration_reference_scores.parquet",
        index=False,
        compression="zstd",
    )
    outcome_columns: list[str] = []
    if args.outcome_ledger is not None:
        outcomes = pd.read_parquet(args.outcome_ledger)
        scored_labels, outcome_columns = _attach_outcomes_after_scoring(final, outcomes)
        scored_labels.to_parquet(
            args.out_dir / "walkforward_scored_label_ledger.parquet",
            index=False, compression="zstd",
        )
    pd.concat(conversion_audit, ignore_index=True).to_parquet(
        args.out_dir / "conversion_reference_audit.parquet", index=False,
    )
    pd.DataFrame(block_rows).to_parquet(args.out_dir / "conversion_block_audit.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_walkforward_long",
        "side": "long",
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(),
        "upstream_cadence": "monthly UTC calendar",
        "conversion_cadence_days": FOUR_WEEK_DAYS,
        "upstream_months": len(upstream_audit),
        "conversion_blocks": len(block_rows),
        "calibration_reserve": {
            "days": int(args.calibration_reserve_days),
            "contract": (
                "OOS calibration tail excluded from every supervised fit at "
                "its refit; a producer-specific calibration reference must "
                "retain only rows outside every active component's fit tail"
                if args.calibration_reserve_days else "disabled_legacy_contract"
            ),
        },
        "rows": len(final),
        "canonical_consensus": "conditional_usefulness_ten_head_v1",
        "ordinary_consensus": "shadow_rollback_only",
        "conversion": (
            "top-30%-trained policy-residual correctness; aggregate C3/leaf trust; "
            "one frozen Oct-Dec 2024 K9 view at temperature scale 0.25; "
            "no raw K9 memberships"
        ),
        "correctness_training_fraction": CORRECTNESS_TRAIN_FRACTION,
        "correctness_gate_domain": "pooled-global training upstream score only",
        "k9_temperature_scale": K9_TEMPERATURE_SCALE,
        "geometry": {
            "refit_cadence": "never",
            "parent_bundle_sha256": frozen_geometry.bundle_sha256,
            "definition_start": geometry_audit["definition_start"],
            "definition_end_exclusive": geometry_audit["definition_end_exclusive"],
        },
        "severe": "exact-H12 diagnostic only; no score effect",
        "normalization": (
            f"same-conversion-model prior-{REFERENCE_DAYS}-day CDF; every held monthly "
            f"upstream producer rescored its own prior-{REFERENCE_DAYS}-day reference"
        ),
        "reference_window_days": REFERENCE_DAYS,
        "same_upstream_bundle_for_reference_and_held_per_producer": True,
        "immediate_calibration_reference": {
            "target_free_scores": "immediate_calibration_reference_scores.parquet",
            "rows": int(len(calibration_reference)),
            "oos_to_all_active_fit_rows": int(
                calibration_reference[
                    "calibration_reference_oos_to_all_active_fits"
                ].fillna(False).astype(bool).sum()
            ),
            "identity": (
                "candidate_id x conversion_bundle_sha256 x upstream_bundle_sha256; "
                "labels may be joined only after this artifact is frozen"
            ),
        },
        "held_percentile_operations": 0,
        "outcomes_consumed_during_scoring": [],
        "outcome_join": {
            "performed_after_scoring": args.outcome_ledger is not None,
            "outcome_ledger": None if args.outcome_ledger is None else str(args.outcome_ledger),
            "outcome_ledger_sha256": None if args.outcome_ledger is None else _sha(args.outcome_ledger),
            "columns": outcome_columns,
        },
        "source_hashes": source_hashes,
        "target_free_source_cache": {
            "start": source_floor.isoformat(),
            "end_exclusive": evaluation_end.isoformat(),
            "rows": int(len(source_cache)),
            "semantics": "bounded target-free source cache; individual slices retain exact timestamp predicates",
        },
        "resumed_from_hash_matched_checkpoints": bool(args.resume),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
    )
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}), flush=True)


if __name__ == "__main__":
    main()
