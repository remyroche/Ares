#!/usr/bin/env python3
"""Score the canonical long-only stack on shifted hourly feature panels.

This research producer intentionally recomputes each phase's current-v5 base,
ten conditional consensus heads, frozen Geometry/K9 conversion state and BCF
score from target-free point-in-time inputs.  It never consumes outcomes or
historical score columns.  Admission and portfolio replay are separate,
chronological stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    LIVE_BASE_ROUTE_FRACTION,
    advance_lockstep_geometry_k9_state,
    initialize_lockstep_geometry_k9_state,
    load_four_week_conversion_bundle,
    load_monthly_upstream_bundle,
    score_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    assert_scoring_frame_is_target_free,
)


CURRENT_BUNDLE_ROOT = (
    ROOT
    / "data_perp/artifacts/strict_r3_score_family_current_v5_"
    "canonical_policy_reconstruction_2025_2026_20260816_v4/bundles"
)
BCF_BUNDLE_ROOT = (
    ROOT
    / "data_perp/artifacts/strict_r3_schema_v2_walkforward_targetfree_"
    "long_2025_aug7_2026_20260809_v1/bundles"
)
CURRENT_BLOCKS = (
    "20260422T000000Z",
    "20260520T000000Z",
    "20260617T000000Z",
    "20260715T000000Z_finalcoverage",
)
HISTORICAL_CURRENT_BLOCKS = (
    "20260225T000000Z",
    "20260325T000000Z",
    "20260422T000000Z",
)
# The final archived producer had a 4 July internal conversion cutoff so its
# 28-day reserve could be established, but its declared out-of-sample emission
# begins on 15 July.  Score the intervening target-free hours to advance frozen
# Geometry/K9 state, then retain only the producer's declared live interval.
CURRENT_BLOCK_EMISSION_START = {
    "20260422T000000Z": "2026-05-01T00:00:00Z",
    "20260520T000000Z": "2026-05-20T00:00:00Z",
    "20260617T000000Z": "2026-06-17T00:00:00Z",
    "20260715T000000Z_finalcoverage": "2026-07-15T00:00:00Z",
}
BCF_MONTHS = ("2026-05", "2026-06", "2026-07")
HISTORICAL_BCF_MONTHS = ("2026-03", "2026-04")
KEYS = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _range_mask(values: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    return parsed.ge(start) & parsed.lt(end)


def _phase_sources(
    root: Path, phase: int, kind: str, *, stream_tag: str,
) -> list[Path]:
    """Return one explicitly versioned phase stream's immutable shards.

    The feature producer is deliberately versioned because a repaired source
    contract must never silently substitute its inputs for an older research
    replay.  Callers name the tag (for example ``v3_completed_h1``) and the
    resulting raw-score manifest preserves it.
    """
    stream = root / f"phase{phase}_streamed_{stream_tag}"
    names = (
        "reference_feature_shards" if kind == "reference" else "replay_feature_shards"
    )
    return sorted((stream / names).glob("block_*/canonical120_features.parquet"))


def _candidate_sources(root: Path, phase: int) -> tuple[Path, Path]:
    return (
        root / f"warmup_grid_phase{phase}" / "target_free_candidate_population.parquet",
        root / f"grid_phase{phase}" / "target_free_candidate_population.parquet",
    )


def _read_features(
    root: Path, phase: int, start: pd.Timestamp, end: pd.Timestamp,
    *, closure_root: Path | None = None, stream_tag: str = "v2",
) -> pd.DataFrame:
    if closure_root is not None:
        path = closure_root / f"phase={phase:02d}" / "features" / "canonical120_features.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"phase={phase} exact full-history closure is missing")
        # Predicate pushdown keeps an exact closure reusable across all
        # same-model reserves without loading its full March--July matrix for
        # each frozen model vintage.
        result = pd.read_parquet(path, filters=[
            ("__decision_ts__", ">=", start.to_pydatetime()),
            ("__decision_ts__", "<", end.to_pydatetime()),
        ])
        result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True)
        result = result.loc[_range_mask(result["__decision_ts__"], start, end)].copy()
        if result.empty:
            raise ValueError(f"phase={phase} exact closure has no feature rows in {start} -> {end}")
        if result["candidate_id"].duplicated().any():
            raise ValueError(f"phase={phase} exact closure has duplicate identities")
        assert_scoring_frame_is_target_free(result)
        return result
    pieces: list[pd.DataFrame] = []
    for kind in ("reference", "replay"):
        for path in _phase_sources(root, phase, kind, stream_tag=stream_tag):
            probe = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__"])
            mask = _range_mask(probe["__decision_ts__"], start, end)
            if not mask.any():
                continue
            # Each three-day feature shard is small enough to preserve every
            # frozen source field; avoid re-generating or imputing anything.
            full = pd.read_parquet(path)
            full["__decision_ts__"] = pd.to_datetime(full["__decision_ts__"], utc=True)
            pieces.append(full.loc[_range_mask(full["__decision_ts__"], start, end)].copy())
    if not pieces:
        raise ValueError(f"phase={phase} has no feature rows in {start} -> {end}")
    result = pd.concat(pieces, ignore_index=True)
    # The warm-up and replay materialisations intentionally meet on their
    # shared boundary decision.  They may therefore contain the *same*
    # candidate once each.  Preserve one copy only after proving that the
    # identity metadata agrees; never collapse genuinely different rows.
    duplicate = result["candidate_id"].duplicated(keep=False)
    if duplicate.any():
        identity = ["__decision_ts__", "__symbol__", "side_name"]
        if (result.loc[duplicate].groupby("candidate_id", sort=False)[identity]
                .nunique(dropna=False).gt(1).any(axis=None)):
            raise ValueError(f"phase={phase} feature shards contain conflicting identities")
        result = result.drop_duplicates("candidate_id", keep="last").reset_index(drop=True)
    return result


def _read_candidates(
    root: Path, phase: int, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in _candidate_sources(root, phase):
        raw = pd.read_parquet(path, columns=list(KEYS))
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True)
        piece = raw.loc[_range_mask(raw["__decision_ts__"], start, end)].copy()
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        raise ValueError(f"phase={phase} has no candidate rows in {start} -> {end}")
    result = pd.concat(pieces, ignore_index=True)
    duplicate = result["candidate_id"].duplicated(keep=False)
    if duplicate.any():
        identity = ["__decision_ts__", "__symbol__", "side_name"]
        if (result.loc[duplicate].groupby("candidate_id", sort=False)[identity]
                .nunique(dropna=False).gt(1).any(axis=None)):
            raise ValueError(f"phase={phase} candidate grids contain conflicting identities")
        result = result.drop_duplicates("candidate_id", keep="last").reset_index(drop=True)
    return result


def _frame(
    root: Path, candidate_root: Path, phase: int, start: pd.Timestamp, end: pd.Timestamp,
    *, closure_root: Path | None = None, stream_tag: str = "v2",
) -> pd.DataFrame:
    candidates = _read_candidates(candidate_root, phase, start, end)
    features = _read_features(
        root, phase, start, end, closure_root=closure_root,
        stream_tag=stream_tag,
    )
    result = candidates.merge(
        features.drop(columns=[column for column in KEYS if column != "candidate_id"], errors="ignore"),
        on="candidate_id", how="left", validate="one_to_one",
    )
    if result["candidate_id"].duplicated().any() or len(result) != len(candidates):
        raise AssertionError("candidate-feature identity changed")
    for column in ("__decision_ts__", "__symbol__", "side_name"):
        if result[column].isna().any():
            raise AssertionError(f"candidate feature join lost {column}")
    assert_scoring_frame_is_target_free(result)
    return result


def _availability(frame: pd.DataFrame, fields: Iterable[str]) -> pd.DataFrame:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, copy=False))
    output = frame.loc[:, list(KEYS)].copy()
    output["frozen_base_feature_count"] = finite.sum(axis=1).astype("int16")
    output["frozen_base_feature_fraction"] = finite.mean(axis=1).astype("float32")
    output["frozen_base_contract_complete"] = finite.all(axis=1)
    return output


def _score_current(
    root: Path, candidate_root: Path, out: Path, phase: int, block_name: str,
    *, closure_root: Path | None = None, stream_tag: str = "v2",
    historical_native_ledger: bool = False, score_end_exclusive: pd.Timestamp | None = None,
) -> dict[str, object]:
    bundle_root = CURRENT_BUNDLE_ROOT / f"block={block_name}"
    conversion = load_four_week_conversion_bundle(bundle_root / "conversion")
    upstream = load_monthly_upstream_bundle(bundle_root / "upstream")
    cutoff = _utc(conversion.cutoff)
    end = _utc(conversion.end_exclusive)
    if score_end_exclusive is not None:
        end = min(end, score_end_exclusive)
    if end <= cutoff:
        raise ValueError(f"current block={block_name} has no requested held interval")
    reference_start = cutoff - pd.Timedelta(days=28)
    reference = _frame(
        root, candidate_root, phase, reference_start, cutoff, closure_root=closure_root,
        stream_tag=stream_tag,
    )
    held = _frame(
        root, candidate_root, phase, cutoff, end, closure_root=closure_root,
        stream_tag=stream_tag,
    )
    reference_score = score_monthly_upstream_bundle(
        upstream, reference, allow_prior_reference=True,
        prior_reference_start=reference_start, route_top_fraction=None,
    )
    held_score = score_monthly_upstream_bundle(
        upstream, held, route_top_fraction=LIVE_BASE_ROUTE_FRACTION,
    )
    score_keys = ["candidate_id", "__decision_ts__", "side_name"]
    reference_input = reference.merge(reference_score, on=score_keys, how="left", validate="one_to_one")
    held_input = held.merge(held_score, on=score_keys, how="left", validate="one_to_one")
    state, scored_reference = initialize_lockstep_geometry_k9_state(
        conversion, reference=reference_input,
        upstream_bundle_sha256=str(upstream.manifest["bundle_sha256"]),
        chunk_hours=72,
        first_live_decision=pd.to_datetime(held_input["__decision_ts__"], utc=True).min(),
    )
    _next, scored_held = advance_lockstep_geometry_k9_state(
        conversion, state=state, held=held_input,
    )
    availability = _availability(held, conversion.base_fields)
    held_out = scored_held.merge(availability, on=list(KEYS), how="left", validate="one_to_one")
    held_out["base_rank"] = pd.to_numeric(held_out["base_rank42"], errors="coerce")
    held_out["consensus_rank"] = pd.to_numeric(held_out["conditional_consensus_rank"], errors="coerce")
    held_out["stack_is_prequential"] = True
    held_out["calibration_activation_ts"] = cutoff
    target_start = cutoff if historical_native_ledger else _utc(CURRENT_BLOCK_EMISSION_START[block_name])
    target_end = min(end, _utc("2026-08-01T00:00:00Z"))
    held_out = held_out.loc[_range_mask(held_out["__decision_ts__"], target_start, target_end)].copy()
    destination = out / "current" / f"block={block_name}" / f"phase={phase:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    held_out.to_parquet(destination / "predictions.parquet", index=False, compression="zstd")
    scored_reference.to_parquet(destination / "same_model_reference_scores.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_phase_h1_current_v5_full_stack_score_v1",
        "phase_minutes": phase, "block": block_name,
        "conversion_bundle": str(bundle_root / "conversion"),
        "upstream_bundle": str(bundle_root / "upstream"),
        "conversion_bundle_sha256": conversion.manifest["bundle_sha256"],
        "upstream_bundle_sha256": upstream.manifest["bundle_sha256"],
        "geometry_bundle_sha256": conversion.geometry.bundle_sha256,
        "geometry_refit_cadence": "never",
        "reference_start": reference_start.isoformat(), "cutoff": cutoff.isoformat(),
        "emission_start": target_start.isoformat(),
        "end_exclusive": end.isoformat(), "reference_rows": int(len(scored_reference)),
        "held_rows": int(len(held_out)), "outcome_columns_consumed": [],
        "held_percentile_operations": 0,
        "base_route": "timestamp-local top-30% from current phase complete universe",
        "phase_offset_geometry_state": "causal one-hour recurrence from same phase prior-28-day reserve",
        "historical_native_ledger": historical_native_ledger,
        "feature_source": "exact_full_history_closure" if closure_root else "incremental_phase_shards",
        "phase_stream_tag": stream_tag,
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def _score_bcf(
    root: Path, candidate_root: Path, out: Path, phase: int, month: str,
    *, closure_root: Path | None = None, stream_tag: str = "v2",
    historical_native_ledger: bool = False, score_end_exclusive: pd.Timestamp | None = None,
) -> dict[str, object]:
    bundle = BCF_BUNDLE_ROOT / f"month={month}"
    cutoff = _utc(f"{month}-01T00:00:00Z")
    month_end = cutoff + pd.offsets.MonthBegin(1)
    if score_end_exclusive is not None:
        month_end = min(_utc(month_end), score_end_exclusive)
    if month_end <= cutoff:
        raise ValueError(f"BCF month={month} has no requested held interval")
    reference = _frame(
        root, candidate_root, phase, cutoff - pd.Timedelta(days=42), cutoff,
        closure_root=closure_root, stream_tag=stream_tag,
    )
    held = _frame(
        root, candidate_root, phase, cutoff, _utc(month_end), closure_root=closure_root,
        stream_tag=stream_tag,
    )
    destination = out / "bcf" / f"month={month}" / f"phase={phase:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    reference_path = destination / "same_model_reference_features.parquet"
    held_path = destination / "held_features.parquet"
    reference.to_parquet(reference_path, index=False, compression="zstd")
    held.to_parquet(held_path, index=False, compression="zstd")
    score_dir = destination / "score"
    command = [
        sys.executable, str(ROOT / "scripts/score_strict_r3_bcf_forward.py"),
        "--monthly-bundle-dir", str(bundle), "--reference-ledger", str(reference_path),
        "--held-features", str(held_path), "--decision-ts", "all",
        "--reference-cache-dir", str(out / "bcf_reference_cache"),
        "--out-dir", str(score_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    (destination / "score.stdout.log").write_text(completed.stdout)
    manifest = json.loads((score_dir / "run_manifest.json").read_text())
    manifest.update({
        "phase_minutes": phase, "month": month, "reference_features_sha256": _sha(reference_path),
        "held_features_sha256": _sha(held_path), "outcome_columns_consumed": [],
        "phase_stream_tag": stream_tag,
        "historical_native_ledger": historical_native_ledger,
    })
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-root", type=Path,
        help=(
            "Immutable target-free candidate root. Defaults to --feature-root; "
            "provide it when repaired feature output is stored separately."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phases", default="0,15,30,45")
    parser.add_argument(
        "--phase-stream-tag", default="v2",
        help=(
            "Explicit immutable phase feature stream tag, for example v2 or "
            "v3_completed_h1. Never silently substitutes a repaired stream."
        ),
    )
    parser.add_argument(
        "--closure-root", type=Path,
        help="Optional immutable exact full-history phase feature closures.",
    )
    parser.add_argument(
        "--current-blocks",
        help=(
            "Comma-separated frozen current-v5 bundle blocks. Defaults to the "
            "matching production or historical-native immutable ledger set."
        ),
    )
    parser.add_argument(
        "--bcf-months",
        help=(
            "Comma-separated frozen BCF months. Defaults to the matching "
            "production or historical-native immutable ledger set."
        ),
    )
    parser.add_argument(
        "--historical-native-ledger", action="store_true",
        help=(
            "Enable only the predeclared February--April current and March--April "
            "BCF bundles for a phase-native historical MC1 ledger.  This never "
            "changes the default production-score interval."
        ),
    )
    parser.add_argument(
        "--score-end-exclusive",
        help="Optional exclusive score horizon for an immutable bounded historical ledger.",
    )
    parser.add_argument("--skip-bcf", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    candidate_root = args.candidate_root or args.feature_root
    phases = tuple(int(value) for value in args.phases.split(",") if value.strip())
    if not phases or any(value not in {0, 15, 30, 45} for value in phases):
        raise ValueError("phases must be a non-empty subset of 0,15,30,45")
    default_current_blocks = (
        HISTORICAL_CURRENT_BLOCKS if args.historical_native_ledger else CURRENT_BLOCKS
    )
    default_bcf_months = (
        HISTORICAL_BCF_MONTHS if args.historical_native_ledger else BCF_MONTHS
    )
    current_blocks = tuple(
        value
        for value in (args.current_blocks or ",".join(default_current_blocks)).split(",")
        if value
    )
    bcf_months = tuple(
        value
        for value in (args.bcf_months or ",".join(default_bcf_months)).split(",")
        if value
    )
    allowed_current = (
        set(HISTORICAL_CURRENT_BLOCKS) if args.historical_native_ledger else set(CURRENT_BLOCKS)
    )
    allowed_bcf = (
        set(HISTORICAL_BCF_MONTHS) if args.historical_native_ledger else set(BCF_MONTHS)
    )
    if not set(current_blocks).issubset(allowed_current):
        raise ValueError("--current-blocks contains a bundle outside the selected immutable ledger mode")
    if not set(bcf_months).issubset(allowed_bcf):
        raise ValueError("--bcf-months contains a bundle outside the selected immutable ledger mode")
    score_end_exclusive = _utc(args.score_end_exclusive) if args.score_end_exclusive else None
    args.out_dir.mkdir(parents=True)
    current: list[dict[str, object]] = []
    bcf: list[dict[str, object]] = []
    for phase in phases:
        for block in current_blocks:
            current.append(_score_current(
                args.feature_root, candidate_root, args.out_dir, phase, block,
                closure_root=args.closure_root, stream_tag=args.phase_stream_tag,
                historical_native_ledger=args.historical_native_ledger,
                score_end_exclusive=score_end_exclusive,
            ))
        if not args.skip_bcf:
            for month in bcf_months:
                bcf.append(_score_bcf(
                    args.feature_root, candidate_root, args.out_dir, phase, month,
                    closure_root=args.closure_root, stream_tag=args.phase_stream_tag,
                    historical_native_ledger=args.historical_native_ledger,
                    score_end_exclusive=score_end_exclusive,
                ))
    manifest = {
        "schema": "strict_r3_phase_h1_full_stack_raw_scores_v1",
        "feature_root": str(args.feature_root), "candidate_root": str(candidate_root), "phases": list(phases),
        "phase_stream_tag": args.phase_stream_tag,
        "requested_current_blocks": list(current_blocks), "requested_bcf_months": list(bcf_months),
        "historical_native_ledger": args.historical_native_ledger,
        "score_end_exclusive": score_end_exclusive.isoformat() if score_end_exclusive is not None else None,
        "current_runs": current, "bcf_runs": bcf,
        "outcome_columns_consumed": [],
        "status": "complete",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "current_runs": len(current), "bcf_runs": len(bcf)}))


if __name__ == "__main__":
    main()
