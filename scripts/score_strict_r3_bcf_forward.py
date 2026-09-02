#!/usr/bin/env python3
"""Score one live BCF snapshot against its same-model prior-42-day reserve.

The script consumes only the frozen BCF 120+73 feature contract.  The reserve
and held rows are scored by the exact same immutable monthly bundle; no held
scores enter any percentile or CDF reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import derive_bcf_mc1_features
from extreme_price_movements.strict_r3_canonical_v2 import (
    assert_scoring_frame_is_target_free,
    load_monthly_bundle,
    score_held_from_same_model_reference_cache,
    score_same_model_reference,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _frame(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    required = ["candidate_id", "__decision_ts__", "side_name", *fields]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"BCF source misses frozen contract fields: {missing}")
    out = frame.loc[:, required].copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("BCF score frame has duplicate candidate IDs")
    assert_scoring_frame_is_target_free(out)
    return out


def _json_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    """Expose an immutable cached reference under the per-run wire name."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _reference_cache(
    *,
    cache_root: Path,
    cache_key: str,
    expected: dict[str, object],
) -> tuple[Path, Path, dict[str, object]] | None:
    directory = cache_root / cache_key
    manifest_path = directory / "manifest.json"
    scores_path = directory / "same_model_prior42_reference_scores.parquet"
    geometry_state_path = directory / "same_model_prior42_geometry_state.parquet"
    if not manifest_path.is_file() or not scores_path.is_file() or not geometry_state_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if manifest.get("status") != "complete" or manifest.get("identity") != expected:
        return None
    if str(manifest.get("scores_sha256") or "") != _sha(scores_path):
        return None
    if str(manifest.get("geometry_state_sha256") or "") != _sha(geometry_state_path):
        return None
    return scores_path, geometry_state_path, manifest


def _reference_geometry_state(bundle: object, reference: pd.DataFrame) -> pd.DataFrame:
    """Persist only timestamp K9 mass needed by dynamic held-state features."""
    _leaves, _distances, membership = bundle.geometry._leaves_membership(reference)
    names = [f"k{index}" for index in range(membership.shape[1])]
    state = pd.DataFrame(membership, columns=names)
    state["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True).to_numpy()
    return state.groupby("__decision_ts__", sort=True)[names].sum().reset_index()


def _persist_reference_cache(
    *,
    cache_root: Path,
    cache_key: str,
    expected: dict[str, object],
    reference_scores: pd.DataFrame,
    reference_geometry_state: pd.DataFrame,
    reference_ledger: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Create an immutable, hash-bound cache once for a BCF bundle/reserve."""
    directory = cache_root / cache_key
    if directory.exists():
        cached = _reference_cache(
            cache_root=cache_root, cache_key=cache_key, expected=expected,
        )
        if cached is None:
            raise ValueError(f"incomplete or incompatible BCF reference cache: {directory}")
        return cached
    directory.mkdir(parents=True, exist_ok=False)
    scores_path = directory / "same_model_prior42_reference_scores.parquet"
    geometry_state_path = directory / "same_model_prior42_geometry_state.parquet"
    reference_scores.to_parquet(scores_path, index=False, compression="zstd")
    reference_geometry_state.to_parquet(geometry_state_path, index=False, compression="zstd")
    manifest: dict[str, object] = {
        "schema": "strict_r3_bcf_same_model_reference_cache_v1",
        "status": "complete",
        "identity": expected,
        "reference_rows": int(len(reference_scores)),
        "reference_ledger_sha256": _sha(reference_ledger),
        "scores_sha256": _sha(scores_path),
        "geometry_state_sha256": _sha(geometry_state_path),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scores_path, geometry_state_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-bundle-dir", type=Path, required=True)
    parser.add_argument("--reference-ledger", type=Path, required=True)
    parser.add_argument("--held-features", type=Path, required=True)
    parser.add_argument(
        "--reference-cache-dir", type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_bcf_same_model_reference_cache_v1",
        help="Immutable cache root for same-bundle prior-42 BCF score domains.",
    )
    parser.add_argument("--decision-ts", required=True,
                        help="One UTC decision timestamp, or 'all' to score every held timestamp.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    bundle = load_monthly_bundle(args.monthly_bundle_dir)
    # The BCF base and Severe context contracts intentionally share causal
    # primitives.  A physical frame may contain each field once; the scorer
    # restores each model's declared ordering when it selects its matrix.
    fields = list(dict.fromkeys([*bundle.base_fields, *bundle.context_fields]))
    start = bundle.cutoff - pd.Timedelta(days=42)
    stat = args.reference_ledger.stat()
    cache_identity: dict[str, object] = {
        "cache_schema": "strict_r3_bcf_same_model_reference_cache_v2",
        "scorer_implementation_sha256": _sha(Path(__file__).resolve()),
        "canonical_stack_implementation_sha256": _sha(
            ROOT / "extreme_price_movements/strict_r3_canonical_v2.py"
        ),
        "bundle_sha256": str(bundle.manifest["bundle_sha256"]),
        "geometry_bundle_sha256": str(bundle.geometry.bundle_sha256),
        "reference_ledger_path": str(args.reference_ledger.resolve()),
        "reference_ledger_size": int(stat.st_size),
        "reference_ledger_mtime_ns": int(stat.st_mtime_ns),
        "reference_start": start.isoformat(),
        "reference_end_exclusive": bundle.cutoff.isoformat(),
        "field_contract_sha256": _json_hash({"fields": fields}),
    }
    cache_key = _json_hash(cache_identity)
    cached = _reference_cache(
        cache_root=args.reference_cache_dir, cache_key=cache_key, expected=cache_identity,
    )
    held_raw = pd.read_parquet(args.held_features)
    held_raw["__decision_ts__"] = pd.to_datetime(held_raw["__decision_ts__"], utc=True)
    if str(args.decision_ts).strip().lower() == "all":
        held_slice = held_raw.loc[held_raw["__decision_ts__"].ge(bundle.cutoff)].copy()
        decision_label = "all_held_timestamps"
    else:
        decision = _utc(args.decision_ts)
        if decision < bundle.cutoff:
            raise ValueError("BCF live decision precedes the monthly bundle cutoff")
        held_slice = held_raw.loc[held_raw["__decision_ts__"].eq(decision)].copy()
        decision_label = decision.isoformat()
    held = _frame(held_slice, fields)
    if held.empty:
        raise ValueError("BCF held snapshot has no current decision rows")
    reference_scores: pd.DataFrame
    reference_geometry_state: pd.DataFrame
    cache_hit = cached is not None
    if cached is None:
        reference_raw = pd.read_parquet(
            args.reference_ledger,
            columns=["candidate_id", "__decision_ts__", "side_name", *fields],
        )
        reference_raw["__decision_ts__"] = pd.to_datetime(
            reference_raw["__decision_ts__"], utc=True,
        )
        reference = _frame(reference_raw.loc[
            reference_raw["__decision_ts__"].ge(start)
            & reference_raw["__decision_ts__"].lt(bundle.cutoff)
        ], fields)
        if reference.empty:
            raise ValueError("BCF same-model prior-42 reserve is empty")
        scored, audit = score_same_model_reference(bundle, reference=reference, held=held)
        held_scored = scored.loc[
            scored["__score_role__"].eq("held")
        ].drop(columns="__score_role__").copy()
        reference_scores = scored.loc[
            scored["__score_role__"].eq("reference")
        ].drop(columns="__score_role__").copy()
        reference_geometry_state = _reference_geometry_state(bundle, reference)
        cached = _persist_reference_cache(
            cache_root=args.reference_cache_dir, cache_key=cache_key,
            expected=cache_identity, reference_scores=reference_scores,
            reference_geometry_state=reference_geometry_state,
            reference_ledger=args.reference_ledger,
        )
    else:
        reference_scores = pd.read_parquet(cached[0])
        reference_geometry_state = pd.read_parquet(cached[1])
        held_scored, audit = score_held_from_same_model_reference_cache(
            bundle, reference_scores=reference_scores,
            reference_geometry_state=reference_geometry_state, held=held,
        )
    native = derive_bcf_mc1_features(held_scored)
    # ``final_score``, ``base_rank42`` and ``upstream`` are already canonical
    # BCF outputs.  Merge only the three derived native-MC1 coordinates so
    # pandas cannot silently suffix and hide the scorer's authoritative values.
    derived = [
        "candidate_id", "conditional_consensus_rank",
        "ordinary_shadow_consensus_rank", "correctness_rank",
    ]
    held_scored = held_scored.merge(
        native.loc[:, derived], on="candidate_id", how="inner", validate="one_to_one"
    )
    args.out_dir.mkdir(parents=True)
    held_scored.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    _link_or_copy(cached[0], args.out_dir / "same_model_prior42_reference_scores.parquet")
    _link_or_copy(cached[1], args.out_dir / "same_model_prior42_geometry_state.parquet")
    audit.to_parquet(args.out_dir / "same_model_reference_replay_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_bcf_live_score_v1",
        "decision_ts": decision_label, "bundle_cutoff": bundle.cutoff.isoformat(),
        "bundle_sha256": bundle.manifest["bundle_sha256"],
        "geometry_bundle_sha256": bundle.geometry.bundle_sha256,
        "reference_rows": int(len(reference_scores)), "held_rows": int(len(held_scored)),
        "same_bundle_reference_and_held": True, "held_percentile_operations": 0,
        "outcome_columns_consumed": [],
        "native_mc1_feature_contract": "native_bcf_ten_head_agreement_v1",
        "reference_ledger": {"path": str(args.reference_ledger), "sha256": _sha(args.reference_ledger)},
        "held_features": {"path": str(args.held_features), "sha256": _sha(args.held_features)},
        "reference_cache": {
            "hit": bool(cache_hit), "key": cache_key,
            "path": str(cached[0]), "scores_sha256": _sha(cached[0]),
            "geometry_state_path": str(cached[1]),
            "geometry_state_sha256": _sha(cached[1]),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
