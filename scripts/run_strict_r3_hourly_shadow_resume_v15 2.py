#!/usr/bin/env python3
"""Materialize and run one fail-closed strict-R3 hourly shadow cycle.

This command has no network client and no exchange/order authority.  Source
refresh remains an explicit prior operation.  It builds the complete frozen
universe first, computes every feature on that complete point-in-time panel,
scores only candidates passing the contemporaneous spread/entry gate, and
then delegates to the sealed shadow-only score/admission/portfolio runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import (  # noqa: E402
    StrictR3InferenceBundle,
    validate_live_feature_frame,
)
from scripts.bridge_strict_r3_live_to_shadow_state import (  # noqa: E402
    LIVE_SCHEMA,
    bridge_state,
)


SCHEMA = "strict_r3_hourly_shadow_orchestration_v1"
def _persisted_feature_state_contract(
    bundle: StrictR3InferenceBundle,
) -> dict[str, object] | None:
    """Return and validate the sealed persisted-state runtime contract."""
    runtime = dict(bundle.payload.get("runtime") or {})
    raw = runtime.get("feature_state")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("runtime.feature_state must be a mapping")
    contract = dict(raw)
    if contract.get("mode") != "persisted_state_only":
        raise ValueError("canonical feature state must use persisted_state_only")
    if contract.get("schema") != "strict_r3_causal_feature_state_bundle_v2":
        raise ValueError("canonical feature state requires a self-contained v2 bundle")
    if contract.get("full_reconstruction_allowed") is not False:
        raise ValueError("canonical feature state must disable full reconstruction")
    if contract.get("hybrid_exact_long_memory_allowed") is not False:
        raise ValueError("canonical feature state must disable the exact fallback")
    if contract.get("runtime_checkpoint_required") is not True:
        raise ValueError("canonical feature state requires a runtime checkpoint")
    if contract.get("runtime_checkpoint_before_order_submission") is not True:
        raise ValueError("runtime checkpoint must precede order submission")
    families = list(contract.get("stateful_exact_families") or [])
    if families != ["final14", "orderbook_precomposite"]:
        raise ValueError(
            "canonical persisted state requires the frozen final14 and "
            "orderbook_precomposite families"
        )
    required = list(contract.get("required_state_kinds") or [])
    if "strict_r3_final14" not in required:
        raise ValueError("canonical state inventory must require strict_r3_final14")
    if "orderbook_precomposite" not in required:
        raise ValueError(
            "canonical state inventory must require orderbook_precomposite"
        )
    if not contract.get("contract_sha256"):
        raise ValueError("canonical feature-state contract hash is absent")
    if not contract.get("final14_contract_sha256"):
        raise ValueError("canonical final14 contract hash is absent")
    if not contract.get("orderbook_precomposite_contract_sha256"):
        raise ValueError("canonical orderbook pre-composite contract hash is absent")
    if int(contract.get("panel_tail_hours", 0)) < 72:
        raise ValueError("canonical persisted state needs at least 72 panel-tail hours")
    return contract


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _preserve_current_hour_scorer_inputs(
    *,
    out_dir: Path,
    decision: pd.Timestamp,
) -> dict[str, object]:
    """Freeze the exact files consumed by the incremental scorer.

    The orchestration appends immutable historical rows to the public candidate
    and feature ledgers after scoring.  Preserve the pre-append files so an
    independent audit can reproduce the live scorer without rebuilding or
    replaying that historical prefix.
    """
    source_candidates = out_dir / "candidate_grid" / "eligible_candidates.parquet"
    source_features = out_dir / "features" / "canonical120_features.parquet"
    frozen_dir = out_dir / "current_hour_inputs"
    if frozen_dir.exists():
        raise FileExistsError(f"immutable scorer-input snapshot exists: {frozen_dir}")
    frozen_dir.mkdir(parents=True)
    frozen_candidates = frozen_dir / "eligible_candidates.parquet"
    frozen_features = frozen_dir / "canonical120_features.parquet"
    shutil.copy2(source_candidates, frozen_candidates)
    shutil.copy2(source_features, frozen_features)

    candidates = pd.read_parquet(frozen_candidates)
    features = pd.read_parquet(frozen_features)
    for frame, role in ((candidates, "candidates"), (features, "features")):
        timestamps = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if not timestamps.eq(decision).all():
            raise AssertionError(
                f"current-hour scorer {role} contain a non-current decision"
            )
        if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
            raise AssertionError(f"current-hour scorer {role} identities are invalid")

    if not set(candidates["candidate_id"].astype(str)).issubset(
        set(features["candidate_id"].astype(str))
    ):
        raise AssertionError(
            "one or more current-hour scorer candidates lack complete-universe features"
        )
    receipt = {
        "schema": "strict_r3_current_hour_scorer_inputs_v1",
        "decision_ts": decision.isoformat(),
        "candidate_rows": int(len(candidates)),
        "feature_rows": int(len(features)),
        "eligible_candidates_sha256": _sha(frozen_candidates),
        "canonical120_features_sha256": _sha(frozen_features),
        "source_paths": {
            "eligible_candidates": str(source_candidates),
            "canonical120_features": str(source_features),
        },
    }
    (frozen_dir / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2, default=str) + "\n"
    )
    return receipt


def _decision_age_seconds(*, decision: pd.Timestamp, now: pd.Timestamp) -> float:
    return float((now - decision).total_seconds())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_payload_digest(bundle_dir: Path) -> str:
    """Hash immutable operator-state payloads, excluding bundle receipts."""
    inventory_path = bundle_dir / "operator_state_inventory.parquet"
    if not inventory_path.is_file():
        raise FileNotFoundError("state re-seal bundle lacks operator inventory")
    inventory = pd.read_parquet(inventory_path)
    required = {"relative_path", "sha256"}
    if not required.issubset(inventory.columns):
        raise ValueError("state re-seal inventory lacks payload hashes")
    digest = hashlib.sha256()
    rows = inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    )
    for row in rows.itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_one_time_feature_state_reseal(
    *,
    contract: dict[str, object],
    source_bundle: Path,
    predecessor_bundle: Path,
) -> dict[str, object]:
    """Permit only an explicitly sealed, byte-identical state re-receipt."""
    raw = contract.get("one_time_state_reseal")
    if not isinstance(raw, dict):
        raise ValueError(
            "recurring persisted state must descend from the exact predecessor"
        )
    reseal = dict(raw)
    expected_old = (ROOT / str(reseal.get("superseded_bundle") or "")).resolve()
    expected_new = (ROOT / str(reseal.get("resealed_bundle") or "")).resolve()
    if predecessor_bundle.resolve() != expected_old or source_bundle.resolve() != expected_new:
        raise ValueError("feature-state re-seal lineage does not match predecessor")
    old_manifest = predecessor_bundle / "state_bundle_manifest.json"
    new_manifest = source_bundle / "state_bundle_manifest.json"
    if not old_manifest.is_file() or not new_manifest.is_file():
        raise FileNotFoundError("feature-state re-seal manifests are incomplete")
    if _sha(old_manifest) != str(reseal.get("superseded_manifest_sha256") or ""):
        raise ValueError("feature-state re-seal superseded manifest hash mismatch")
    if _sha(new_manifest) != str(reseal.get("resealed_manifest_sha256") or ""):
        raise ValueError("feature-state re-seal manifest hash mismatch")
    old_payload = _state_payload_digest(predecessor_bundle)
    new_payload = _state_payload_digest(source_bundle)
    if old_payload != new_payload or old_payload != str(
        reseal.get("operator_state_payload_sha256") or ""
    ):
        raise ValueError("feature-state re-seal changed an operator-state payload")
    return {
        "schema": "strict_r3_feature_state_reseal_audit_v1",
        "superseded_bundle": str(predecessor_bundle),
        "resealed_bundle": str(source_bundle),
        "operator_state_payload_sha256": old_payload,
        "superseded_manifest_sha256": _sha(old_manifest),
        "resealed_manifest_sha256": _sha(new_manifest),
    }


def _resolve_reconciliation_state(
    *,
    supplied_state_path: Path,
    predecessor_state_path: Path,
    decision: pd.Timestamp,
    out_dir: Path,
) -> tuple[Path, dict[str, object] | None]:
    """Use an exact bridge when the caller supplies canonical live state.

    The live ledger is deliberately an exchange-facing state, whereas the
    portfolio runner consumes a shadow-policy state.  A recurring flat ledger
    may be older than the decision boundary; deriving the bridge from the
    immutable predecessor makes its portfolio timestamp exact without creating
    positions or changing any historical score/policy state.
    """
    payload = json.loads(supplied_state_path.read_text())
    if payload.get("schema") != LIVE_SCHEMA:
        return supplied_state_path, None
    shadow = json.loads(predecessor_state_path.read_text())
    bridge_path = out_dir / "portfolio_reconciliation_state.json"
    if bridge_path.exists():
        raise FileExistsError(f"immutable reconciliation bridge exists: {bridge_path}")
    bridge = bridge_state(
        live=payload,
        shadow=shadow,
        decision_ts=decision,
        live_state_path=supplied_state_path,
        shadow_reference_path=predecessor_state_path,
    )
    bridge_path.write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n")
    return bridge_path, {
        "generated_from_live_state": True,
        "path": str(bridge_path),
        "sha256": _sha(bridge_path),
        "live_state_sha256": _sha(supplied_state_path),
        "shadow_reference_sha256": _sha(predecessor_state_path),
    }


def _utc_hour(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )
    if timestamp != timestamp.floor("h"):
        raise ValueError("strict-R3 decision timestamp must be an exact UTC hour")
    return timestamp


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"hourly shadow stage failed; see {log_path}")


def _skip_incomplete_current_rows(
    *,
    decision: pd.Timestamp,
    fields: list[str],
    eligible_path: Path,
    population_path: Path,
    rejection_path: Path,
    features: pd.DataFrame,
) -> dict[str, object]:
    """Remove only current timestamp/symbol rows missing a frozen input.

    Features are still generated on the complete point-in-time universe so
    cross-sectional values do not depend on the actionable subset.  This gate
    changes scoring eligibility only after feature generation; it never fills
    a missing value and never removes another timestamp or symbol.
    """
    feature_decision = pd.to_datetime(features["__decision_ts__"], utc=True)
    current = features.loc[feature_decision.eq(decision)].copy()
    numeric = current.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    complete = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
        index=current.index,
    )
    skipped_ids = set(current.loc[~complete, "candidate_id"].astype(str))
    missing_fields_by_id = {
        str(row["candidate_id"]): [
            field for field in fields
            if not np.isfinite(pd.to_numeric(row[field], errors="coerce"))
        ]
        for _, row in current.loc[~complete].iterrows()
    }
    if not skipped_ids:
        return {
            "decision_ts": decision.isoformat(),
            "current_feature_rows": int(len(current)),
            "skipped_rows": 0,
            "retained_rows": int(len(current)),
            "reason": "feature_unavailable_at_decision",
            "missing_field_counts": {},
        }

    eligible = pd.read_parquet(eligible_path)
    eligible["candidate_id"] = eligible["candidate_id"].astype(str)
    actually_skipped = eligible["candidate_id"].isin(skipped_ids)
    skipped = eligible.loc[actually_skipped].copy()
    eligible.loc[~actually_skipped].to_parquet(
        eligible_path, index=False, compression="zstd",
    )

    if not skipped.empty:
        skipped["eligible"] = False
        skipped["eligibility_reason"] = "feature_unavailable_at_decision"
        skipped["missing_frozen_fields"] = skipped["candidate_id"].map(
            lambda value: json.dumps(missing_fields_by_id.get(str(value), []))
        )
        rejected = pd.read_parquet(rejection_path)
        rejected = pd.concat([rejected, skipped], ignore_index=True, sort=False)
        rejected.to_parquet(rejection_path, index=False, compression="zstd")

        population = pd.read_parquet(population_path)
        population["candidate_id"] = population["candidate_id"].astype(str)
        mask = population["candidate_id"].isin(set(skipped["candidate_id"]))
        population.loc[mask, "eligible"] = False
        population.loc[mask, "eligibility_reason"] = (
            "feature_unavailable_at_decision"
        )
        population.to_parquet(population_path, index=False, compression="zstd")

    field_counts: dict[str, int] = {}
    for candidate_id in skipped["candidate_id"].astype(str):
        for field in missing_fields_by_id.get(candidate_id, []):
            field_counts[field] = field_counts.get(field, 0) + 1
    return {
        "decision_ts": decision.isoformat(),
        "current_feature_rows": int(len(current)),
        "skipped_rows": int(len(skipped)),
        "retained_rows": int(len(current) - len(skipped)),
        "reason": "feature_unavailable_at_decision",
        "missing_field_counts": dict(sorted(field_counts.items())),
    }


def _assert_append_only_overlap(
    *, previous_run: Path, current_run: Path,
    include_predictions: bool = True,
) -> dict[str, object]:
    """Prove that extending the live prefix cannot rewrite earlier rows."""
    contracts = {
        "candidate_population": (
            "candidate_grid/target_free_candidate_population.parquet", "candidate_id",
        ),
        "eligible_candidates": (
            "candidate_grid/eligible_candidates.parquet", "candidate_id",
        ),
        "features": ("features/canonical120_features.parquet", "candidate_id"),
        "predictions": ("cycle/score/predictions.parquet", "candidate_id"),
    }
    if not include_predictions:
        contracts.pop("predictions")
    audit: dict[str, object] = {}
    for role, (relative, key) in contracts.items():
        old_path = previous_run / relative
        new_path = current_run / relative
        if not old_path.exists() or not new_path.exists():
            raise FileNotFoundError(
                f"append-only parity requires {relative} in both hourly runs",
            )
        old = pd.read_parquet(old_path)
        new = pd.read_parquet(new_path)
        if old[key].isna().any() or old[key].duplicated().any():
            raise ValueError(f"previous {role} identities are not unique")
        if new[key].isna().any() or new[key].duplicated().any():
            raise ValueError(f"current {role} identities are not unique")
        old[key] = old[key].astype(str)
        new[key] = new[key].astype(str)
        missing = set(old[key]).difference(new[key])
        if missing:
            raise AssertionError(
                f"hourly prefix removed {len(missing)} prior {role} rows",
            )
        overlap = old.merge(
            new, on=key, how="left", validate="one_to_one",
            suffixes=("__old", "__new"),
        )
        changed_fields: list[str] = []
        max_numeric_delta = 0.0
        for field in sorted(set(old.columns).intersection(new.columns).difference({key})):
            left = overlap[f"{field}__old"]
            right = overlap[f"{field}__new"]
            try:
                left_num = pd.to_numeric(left, errors="raise").to_numpy(float)
                right_num = pd.to_numeric(right, errors="raise").to_numpy(float)
            except (TypeError, ValueError):
                equal = left.astype(str).eq(right.astype(str)) | (left.isna() & right.isna())
                if not bool(equal.all()):
                    changed_fields.append(field)
                continue
            equal = np.isclose(
                left_num, right_num, atol=1e-9, rtol=0.0, equal_nan=True,
            )
            if not bool(equal.all()):
                changed_fields.append(field)
                finite = np.isfinite(left_num) & np.isfinite(right_num)
                if finite.any():
                    max_numeric_delta = max(
                        max_numeric_delta,
                        float(np.max(np.abs(left_num[finite] - right_num[finite]))),
                    )
        if changed_fields:
            raise AssertionError(
                f"hourly prefix rewrote prior {role} fields: {changed_fields[:12]}",
            )
        audit[role] = {
            "previous_rows": int(len(old)),
            "current_rows": int(len(new)),
            "overlap_rows": int(len(overlap)),
            "new_rows": int(len(new) - len(overlap)),
            "changed_fields": [],
            "max_numeric_delta": max_numeric_delta,
        }
    return audit


def _assemble_immutable_prefix(
    *, previous_run: Path, current_run: Path,
) -> dict[str, object]:
    """Append the current hour to the preceding immutable input prefix.

    Historical target-free eligibility and features are never recomputed from
    a mutable market-data cache. Only the current complete-universe signal
    hour is freshly materialized, then concatenated after the sealed prefix.
    """
    contracts = {
        "candidate_population": "candidate_grid/target_free_candidate_population.parquet",
        "eligible_candidates": "candidate_grid/eligible_candidates.parquet",
        "candidate_rejections": "candidate_grid/candidate_rejection_audit.parquet",
        "features": "features/canonical120_features.parquet",
    }
    audit: dict[str, object] = {}
    for role, relative in contracts.items():
        old_path = previous_run / relative
        new_path = current_run / relative
        old = pd.read_parquet(old_path)
        current = pd.read_parquet(new_path)
        for frame, label in ((old, "previous"), (current, "current")):
            if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
                raise ValueError(f"{label} {role} identities are not unique")
            frame["candidate_id"] = frame["candidate_id"].astype(str)
        overlap = set(old["candidate_id"]).intersection(current["candidate_id"])
        if overlap:
            raise ValueError(f"current-hour {role} overlaps immutable prefix")
        combined = pd.concat([old, current], ignore_index=True, sort=False)
        combined["__decision_ts__"] = pd.to_datetime(
            combined["__decision_ts__"], utc=True, errors="raise",
        )
        combined = combined.sort_values(
            ["__decision_ts__", "candidate_id"], kind="stable",
        ).reset_index(drop=True)
        combined.to_parquet(new_path, index=False, compression="zstd")
        audit[role] = {
            "previous_rows": int(len(old)),
            "current_rows": int(len(current)),
            "combined_rows": int(len(combined)),
        }
    population = pd.read_parquet(
        current_run / contracts["candidate_population"],
    )
    summary = population.groupby(
        ["side_name", "eligibility_reason"], as_index=False, dropna=False,
    ).agg(rows=("candidate_id", "size"))
    summary.to_parquet(
        current_run / "candidate_grid/candidate_rejection_reason_summary.parquet",
        index=False,
    )
    grid_manifest_path = current_run / "candidate_grid/run_manifest.json"
    grid_manifest = json.loads(grid_manifest_path.read_text())
    previous_grid_manifest = json.loads(
        (previous_run / "candidate_grid/run_manifest.json").read_text(),
    )
    if grid_manifest.get("source_map") != previous_grid_manifest.get("source_map"):
        raise ValueError("current-hour universe differs from immutable prefix")
    grid_manifest.update({
        "start": previous_grid_manifest["start"],
        "population_rows": int(len(population)),
        "eligible_rows": int(audit["eligible_candidates"]["combined_rows"]),
        "rejected_rows": int(audit["candidate_rejections"]["combined_rows"]),
        "immutable_prefix_assembly": audit,
        "immutable_prefix_source": str(previous_run),
    })
    grid_manifest_path.write_text(json.dumps(grid_manifest, indent=2, default=str))
    feature_manifest_path = current_run / "features/feature_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())
    feature_manifest["immutable_prefix_assembly"] = audit["features"]
    feature_manifest["immutable_prefix_source"] = str(previous_run)
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2, default=str))
    return audit


def _reuse_current_inputs(
    *, source_run: Path, previous_run: Path | None, current_run: Path,
    decision: pd.Timestamp,
) -> dict[str, object]:
    """Reuse an already materialized current-hour input checkpoint.

    This is deliberately narrower than a general cache: only the exact
    decision cross-section is copied from ``source_run``.  The immutable
    historical prefix is subsequently restored from ``previous_run`` by
    :func:`_assemble_immutable_prefix`.  Every identity and universe contract
    is checked before the expensive scorer is allowed to run.
    """
    forbidden = {current_run.resolve()}
    if previous_run is not None:
        forbidden.add(previous_run.resolve())
    if source_run.resolve() in forbidden:
        raise ValueError("reused input source must be distinct from predecessor/output")
    source_grid = source_run / "candidate_grid"
    source_features = source_run / "features"
    required = (
        source_grid / "run_manifest.json",
        source_grid / "target_free_candidate_population.parquet",
        source_grid / "eligible_candidates.parquet",
        source_grid / "candidate_rejection_audit.parquet",
        source_features / "feature_manifest.json",
        source_features / "canonical120_features.parquet",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"reused input checkpoint is incomplete: {missing}")

    source_grid_manifest = json.loads(required[0].read_text())
    source_feature_manifest = json.loads(required[4].read_text())
    previous_grid_manifest = (
        json.loads(
            (previous_run / "candidate_grid/run_manifest.json").read_text()
        )
        if previous_run is not None else None
    )
    if source_grid_manifest.get("future_path_columns_consumed") not in ([], None):
        raise ValueError("reused inputs consumed future-path columns")
    if previous_grid_manifest is not None:
        if source_grid_manifest.get("source_map") != previous_grid_manifest.get("source_map"):
            raise ValueError("reused input universe/source map differs from predecessor")
        if source_grid_manifest.get("universe_sha256") != previous_grid_manifest.get("universe_sha256"):
            raise ValueError("reused input universe hash differs from predecessor")

    current_grid = current_run / "candidate_grid"
    current_features = current_run / "features"
    shutil.copytree(source_grid, current_grid)
    shutil.copytree(source_features, current_features)
    source_feature_state = source_run / "feature_state" / "bundle"
    if source_feature_state.is_dir():
        shutil.copytree(
            source_feature_state,
            current_run / "feature_state" / "bundle",
        )

    roles = {
        "candidate_population": (
            current_grid / "target_free_candidate_population.parquet", True,
        ),
        "eligible_candidates": (
            current_grid / "eligible_candidates.parquet", False,
        ),
        "candidate_rejections": (
            current_grid / "candidate_rejection_audit.parquet", False,
        ),
        "features": (
            current_features / "canonical120_features.parquet", True,
        ),
    }
    audit: dict[str, object] = {}
    identities: dict[str, set[str]] = {}
    for role, (path, require_universe) in roles.items():
        frame = pd.read_parquet(path)
        if "__decision_ts__" not in frame:
            raise ValueError(f"reused {role} lacks __decision_ts__")
        timestamps = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        current = frame.loc[timestamps.eq(decision)].copy()
        if current["candidate_id"].isna().any() or current["candidate_id"].duplicated().any():
            raise ValueError(f"reused current-hour {role} identities are invalid")
        current["candidate_id"] = current["candidate_id"].astype(str)
        if require_universe and len(current) != int(source_grid_manifest["universe_rows"]):
            raise ValueError(
                f"reused current-hour {role} does not cover frozen universe: "
                f"rows={len(current)} expected={source_grid_manifest['universe_rows']}"
            )
        # A first-run resume has no predecessor from which to reconstruct the
        # dynamic Geometry/K9 prefix. Preserve the source's exact continuous
        # candidate *and feature* ledgers so the one-time bootstrap has full
        # frozen-contract coverage. With a predecessor, all roles stay
        # current-only here and are appended by _assemble_immutable_prefix.
        preserve_source_prefix = previous_run is None
        if not preserve_source_prefix:
            current.to_parquet(path, index=False, compression="zstd")
        identities[role] = set(current["candidate_id"])
        audit[role] = {
            "source_rows": int(len(frame)),
            "current_rows": int(len(current)),
            "preserved_source_prefix": bool(preserve_source_prefix),
            "sha256": _sha(path),
        }
    if identities["features"] != identities["candidate_population"]:
        raise ValueError("reused features do not match complete current population")
    if not identities["eligible_candidates"].issubset(identities["candidate_population"]):
        raise ValueError("reused eligible identities are outside current population")
    if identities["eligible_candidates"].intersection(identities["candidate_rejections"]):
        raise ValueError("reused eligible and rejected identities overlap")
    if identities["eligible_candidates"] | identities["candidate_rejections"] != identities["candidate_population"]:
        raise ValueError("reused eligibility partition is incomplete")

    population = pd.read_parquet(roles["candidate_population"][0])
    summary = population.groupby(
        ["side_name", "eligibility_reason"], as_index=False, dropna=False,
    ).agg(rows=("candidate_id", "size"))
    summary.to_parquet(
        current_grid / "candidate_rejection_reason_summary.parquet",
        index=False, compression="zstd",
    )
    persisted_population = pd.read_parquet(
        current_grid / "target_free_candidate_population.parquet"
    )
    persisted_eligible = pd.read_parquet(
        current_grid / "eligible_candidates.parquet"
    )
    persisted_rejected = pd.read_parquet(
        current_grid / "candidate_rejection_audit.parquet"
    )
    source_grid_manifest.update({
        "start": (
            source_grid_manifest.get("start")
            if previous_run is None
            else (decision - pd.Timedelta(hours=1)).isoformat()
        ),
        "end_exclusive": decision.isoformat(),
        "population_rows": int(len(persisted_population)),
        "eligible_rows": int(len(persisted_eligible)),
        "rejected_rows": int(len(persisted_rejected)),
        "checkpoint_reuse": audit,
        "checkpoint_source": str(source_run),
    })
    (current_grid / "run_manifest.json").write_text(
        json.dumps(source_grid_manifest, indent=2, default=str) + "\n"
    )
    source_feature_manifest.update({
        "rows": len(identities["features"]),
        "checkpoint_reuse": audit["features"],
        "checkpoint_source": str(source_run),
    })
    (current_features / "feature_manifest.json").write_text(
        json.dumps(source_feature_manifest, indent=2, default=str) + "\n"
    )
    return {
        "source_run": str(source_run),
        "decision_ts": decision.isoformat(),
        "source_grid_manifest_sha256": _sha(required[0]),
        "source_feature_manifest_sha256": _sha(required[4]),
        "roles": audit,
        "feature_state_bundle_reused": source_feature_state.is_dir(),
    }


def _stateful_input_commands(
    *,
    bundle: StrictR3InferenceBundle,
    decision: pd.Timestamp,
    out_dir: Path,
    source_state_bundle: Path,
    state_contract_hash: str,
    state_tail_hours: int,
    stateful_exact_families: list[str],
    final14_contract_hash: str | None,
    orderbook_precomposite_contract_hash: str | None,
    required_state_kinds: list[str],
    grid_command: list[str],
) -> list[tuple[str, list[str]]]:
    """Build current inputs by advancing one immutable causal-state bundle."""
    state_manifest_path = source_state_bundle / "state_bundle_manifest.json"
    if not state_manifest_path.is_file():
        raise FileNotFoundError("stateful input bundle lacks its manifest")
    state_manifest = json.loads(state_manifest_path.read_text())
    if state_manifest.get("schema") != "strict_r3_causal_feature_state_bundle_v2":
        raise ValueError("hourly stateful inputs require a self-contained v2 bundle")
    if state_manifest.get("feature_contract_sha256") != state_contract_hash:
        raise ValueError("feature-state bundle has the wrong frozen contract")
    manifest_kinds = set(map(str, state_manifest.get("required_state_kinds") or []))
    missing_kinds = sorted(set(required_state_kinds).difference(manifest_kinds))
    if missing_kinds:
        raise ValueError(
            f"feature-state bundle lacks sealed required kinds: {missing_kinds}"
        )
    implementation = state_manifest.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("feature-state bundle lacks implementation hashes")
    for relative, expected_hash in implementation.items():
        source = (ROOT / str(relative)).resolve()
        if ROOT.resolve() not in source.parents or not source.is_file():
            raise ValueError(f"feature-state implementation path is invalid: {relative}")
        if _sha(source) != str(expected_hash):
            raise ValueError(f"feature-state implementation hash changed: {relative}")
    if "final14" in stateful_exact_families:
        final14_path = source_state_bundle / "states" / "strict_r3_final14.state"
        if not final14_path.is_file():
            raise FileNotFoundError("feature-state bundle lacks strict_r3_final14.state")
    embedded_panel = source_state_bundle / str(state_manifest["panel_state"])
    if not embedded_panel.is_file():
        raise FileNotFoundError("feature-state bundle lacks its embedded panel")
    runtime = dict(bundle.payload.get("runtime") or {})
    grid_dir = out_dir / "candidate_grid"
    feature_dir = out_dir / "features"
    state_root = out_dir / "feature_state"
    panel_dir = state_root / "source_panel_update"
    panel_path = panel_dir / "feature_panel_state.joblib"
    cache_dir = state_root / "cache"
    next_bundle = state_root / "bundle"
    return [
        ("candidate_grid", grid_command),
        (
            "source_panel_state",
            [
                sys.executable,
                str(ROOT / "scripts/update_strict_r3_feature_panel_state.py"),
                "--candidates", str(grid_dir / "target_free_candidate_population.parquet"),
                "--history-start", str(state_manifest["panel_start"]),
                "--end-exclusive", decision.isoformat(),
                "--state-in", str(embedded_panel),
                "--preserve-sealed-overlap",
                "--out-dir", str(panel_dir),
            ],
        ),
        (
            "features",
            [
                sys.executable,
                str(ROOT / "scripts/materialize_strict_r3_forward_features_incremental_v13.py"),
                "--candidates", str(grid_dir / "target_free_candidate_population.parquet"),
                "--panel-state", str(panel_path),
                "--cache-dir", str(cache_dir),
                "--restore-state-bundle", str(source_state_bundle),
                "--expected-state-contract-hash", state_contract_hash,
                "--stateful-tail-hours", str(state_tail_hours),
                *[
                    value
                    for family in stateful_exact_families
                    for value in ("--stateful-exact-family", family)
                ],
                *(
                    ["--expected-final14-contract-hash", str(final14_contract_hash)]
                    if "final14" in stateful_exact_families else []
                ),
                *(
                    [
                        "--expected-orderbook-precomposite-contract-hash",
                        str(orderbook_precomposite_contract_hash),
                    ]
                    if "orderbook_precomposite" in stateful_exact_families else []
                ),
                "--side", "long",
                "--out-dir", str(feature_dir),
            ],
        ),
        (
            "feature_state_snapshot",
            [
                sys.executable,
                str(ROOT / "scripts/snapshot_strict_r3_feature_state_bundle.py"),
                "--cache-dir", str(cache_dir),
                "--panel-state", str(panel_path),
                "--out-dir", str(next_bundle),
                "--contract-hash", state_contract_hash,
                "--scope", "strict_r3_hourly_canonical120_stateful",
                "--panel-tail-hours", str(state_tail_hours),
                *[
                    value
                    for kind in required_state_kinds
                    for value in ("--required-state-kind", kind)
                ],
                "--expected-state-timestamp",
                (decision - pd.Timedelta(hours=1)).isoformat(),
            ],
        ),
    ]


def _commands(
    *,
    bundle_path: Path,
    bundle: StrictR3InferenceBundle,
    state_path: Path,
    decision: pd.Timestamp,
    out_dir: Path,
    previous_run: Path | None = None,
    portfolio_state_activation: bool = False,
) -> list[tuple[str, list[str]]]:
    signal = decision - pd.Timedelta(hours=1)
    activation = pd.Timestamp(bundle.payload["activation_ts"])
    activation = (
        activation.tz_localize("UTC")
        if activation.tzinfo is None else activation.tz_convert("UTC")
    )
    # A signal at producer activation becomes a decision at activation + 1h.
    # This exactly matches the canonical next-bar-open batch ledger; including
    # an activation-hour decision would inject one extra cross-section into
    # rolling K9 state.
    prefix_signal = signal if previous_run is not None else activation
    prefix_decision = decision if previous_run is not None else activation + pd.Timedelta(hours=1)
    runtime = dict(bundle.payload.get("runtime") or {})
    history_start = runtime.get("feature_history_start")
    if not history_start:
        raise ValueError("sealed inference bundle lacks feature_history_start")
    grid_dir = out_dir / "candidate_grid"
    feature_dir = out_dir / "features"
    cycle_dir = out_dir / "cycle"
    chained_state = (
        previous_run / "cycle" / "next_portfolio_state.json"
        if previous_run is not None
        else None
    )
    if chained_state is not None and not chained_state.exists():
        raise FileNotFoundError(
            "previous hourly run lacks cycle/next_portfolio_state.json; "
            "portfolio state chaining fails closed"
        )
    effective_state_path = (
        chained_state if chained_state is not None and not portfolio_state_activation
        else state_path
    )
    intraday_resolved_ledger: Path | None = None
    if previous_run is not None:
        previous_manifest = json.loads((previous_run / "run_manifest.json").read_text())
        previous_decision = _utc_hour(previous_manifest["decision_ts"])
        if previous_decision.normalize() == decision.normalize():
            intraday_resolved_ledger = (
                previous_run / "cycle" / "runtime_resolved_state" /
                "walkforward_scored_label_ledger.parquet"
            )
            if not intraday_resolved_ledger.exists():
                raise FileNotFoundError(
                    "same-day predecessor lacks its frozen calibration ledger"
                )
    immutable_prediction_prefix = (
        previous_run / "cycle" / "score" / "predictions.parquet"
        if previous_run is not None
        else bundle.path("resolved_score_label_ledger")
    )
    if not immutable_prediction_prefix.exists():
        raise FileNotFoundError(
            "hourly successor lacks its immutable prediction-prefix source"
        )
    geometry_k9_state = (
        previous_run / "cycle" / "score" / "geometry_k9_state"
        if previous_run is not None else None
    )
    if geometry_k9_state is not None and not (
        geometry_k9_state / "run_manifest.json"
    ).is_file():
        # The first successor after this optimization deliberately bootstraps
        # from activation once.  A later run may never silently fall back to
        # replaying the full prefix: it must receive the exact predecessor
        # state or fail closed upstream in the orchestrator.
        geometry_k9_state = None
    return [
        (
            "candidate_grid",
            [
                sys.executable,
                str(ROOT / runtime["candidate_materializer"]),
                "--universe-manifest", str(bundle.path("frozen_universe_manifest")),
                # The complete activation-to-current prefix is required to
                # reconstruct the conversion layer's causal rolling K9 state
                # after any process restart. Only the current decision is
                # admitted/auctioned downstream.
                "--start", prefix_signal.isoformat(),
                "--end-exclusive", decision.isoformat(),
                "--sides", "long",
                "--spread-limit-bps", "100",
                "--policy-bar-root", str(
                    ROOT / runtime.get("policy_bar_root", "15m_ohlcv_perp")
                ),
                "--out-dir", str(grid_dir),
            ],
        ),
        (
            "features",
            [
                sys.executable,
                str(ROOT / runtime["feature_materializer"]),
                # The complete population, including spread rejects, defines
                # the causal cross-section.  Scoring is filtered separately.
                "--candidates", str(grid_dir / "target_free_candidate_population.parquet"),
                "--out-dir", str(feature_dir),
                "--candidate-start", prefix_decision.isoformat(),
                "--history-start", str(history_start),
                "--end-exclusive", (decision + pd.Timedelta(hours=1)).isoformat(),
                "--side", "long",
            ],
        ),
        (
            "shadow_cycle",
            [
                sys.executable,
                str(ROOT / runtime["shadow_cycle"]),
                "--inference-bundle", str(bundle_path),
                "--held-candidates", str(grid_dir / "eligible_candidates.parquet"),
                "--held-features", str(feature_dir / "canonical120_features.parquet"),
                *(
                    [
                        "--policy-label-candidates",
                        str(
                            previous_run / "candidate_grid" /
                            "eligible_candidates.parquet"
                        ),
                    ]
                    if previous_run is not None else []
                ),
                "--portfolio-state-json", str(effective_state_path),
                "--decision-ts", decision.isoformat(),
                "--out-dir", str(cycle_dir),
                "--mode", "shadow-only",
                "--immutable-prediction-prefix",
                str(immutable_prediction_prefix),
                # The incremental scorer deliberately emits only the new
                # decision hour.  The predecessor prediction ledger is
                # immutable and is restored below; requiring the current
                # scorer to reproduce that full prefix would reintroduce the
                # expensive historical replay this path is designed to avoid.
                "--allow-missing-current-prefix-rows",
                *(
                    [
                        "--sealed-bootstrap-prediction-prefix",
                    ]
                    if previous_run is None else []
                ),
                *(
                    [
                        "--intraday-frozen-resolved-ledger",
                        str(intraday_resolved_ledger),
                    ]
                    if intraday_resolved_ledger is not None else []
                ),
                *(
                    [
                        "--lockstep-geometry-k9-state-in",
                        str(geometry_k9_state),
                    ]
                    if geometry_k9_state is not None else []
                ),
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-bundle", type=Path, required=True,
        help=(
            "Exact immutable schema-v6 Robust-21/MC1 bundle. This is required "
            "so a mutable or self-referential canonical default can never be used."
        ),
    )
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--previous-shadow-run", type=Path,
        help=(
            "Previous immutable successful hourly run. When supplied, every "
            "overlapping candidate, feature, and prediction must be identical."
        ),
    )
    parser.add_argument(
        "--reuse-current-inputs-from", type=Path,
        help=(
            "Resume from a persisted candidate/feature checkpoint for this "
            "exact decision. The current cross-section is revalidated and "
            "the immutable history is restored from --previous-shadow-run."
        ),
    )
    parser.add_argument(
        "--feature-state-bundle",
        type=Path,
        help=(
            "Enable the stateful feature challenger by advancing this exact "
            "self-contained v2 bundle. The incumbent full-history path is "
            "unchanged when omitted."
        ),
    )
    parser.add_argument("--feature-state-contract-hash")
    parser.add_argument("--feature-state-tail-hours", type=int, default=1536)
    parser.add_argument(
        "--portfolio-state-activation", action="store_true",
        help=(
            "One-time live activation: retain the prior immutable score prefix "
            "but use the explicitly supplied exact-decision flat state."
        ),
    )
    parser.add_argument(
        "--portfolio-state-reconciliation", action="store_true",
        help=(
            "Recurring live operation: use the supplied exact-decision "
            "actual-fill bridge while retaining the predecessor's immutable "
            "candidate/feature/prediction prefix."
        ),
    )
    parser.add_argument("--mode", choices=("shadow-only",), default="shadow-only")
    parser.add_argument(
        "--enforce-live-wall-clock", action="store_true",
        help=(
            "Require both orchestration start and completed manifest to fall "
            "inside the bundle's live decision-freshness window."
        ),
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable hourly shadow output exists: {args.out_dir}")
    decision = _utc_hour(args.decision_ts)
    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    bundle_audit = bundle.validate(decision_ts=decision)
    feature_state_contract = _persisted_feature_state_contract(bundle)
    started_at = _utc_now()
    freshness_seconds = int(bundle.payload["live_decision_freshness_seconds"])
    age_at_start = _decision_age_seconds(decision=decision, now=started_at)
    if args.enforce_live_wall_clock and not 0.0 <= age_at_start <= freshness_seconds:
        raise RuntimeError(
            "hourly shadow started outside the sealed live decision window: "
            f"age={age_at_start:.3f}s limit={freshness_seconds}s"
        )
    chained_state_path = (
        args.previous_shadow_run / "cycle" / "next_portfolio_state.json"
        if args.previous_shadow_run is not None else None
    )
    if chained_state_path is not None and not chained_state_path.exists():
        raise FileNotFoundError(
            "previous hourly run lacks cycle/next_portfolio_state.json; "
            "portfolio state chaining fails closed"
        )
    if args.portfolio_state_activation and args.portfolio_state_reconciliation:
        raise ValueError("portfolio activation and reconciliation are mutually exclusive")
    use_supplied_state = bool(
        args.portfolio_state_activation or args.portfolio_state_reconciliation
    )
    effective_state_path = (
        chained_state_path
        if chained_state_path is not None and not use_supplied_state
        else args.portfolio_state_json
    )
    reconciliation_bridge_audit = None
    if args.portfolio_state_activation:
        if args.previous_shadow_run is None:
            raise ValueError("portfolio activation requires an immutable predecessor")
        activation_state = json.loads(args.portfolio_state_json.read_text())
        state_ts = pd.Timestamp(activation_state["as_of_ts"])
        state_ts = (
            state_ts.tz_localize("UTC") if state_ts.tzinfo is None
            else state_ts.tz_convert("UTC")
        )
        if state_ts != decision or activation_state.get("open_positions") != []:
            raise ValueError(
                "portfolio activation requires an exact-decision flat shadow state"
            )
    if args.portfolio_state_reconciliation:
        if chained_state_path is None:
            raise ValueError("portfolio reconciliation requires an immutable predecessor")
        # The canonical executor persists Kraken-facing live state.  Convert it
        # to the exact-decision shadow-policy state here, before any candidate
        # materialization, when a raw live ledger is supplied by the hourly
        # scheduler.  A prebuilt immutable bridge remains accepted unchanged.
        args.out_dir.mkdir(parents=True)
        effective_state_path, reconciliation_bridge_audit = (
            _resolve_reconciliation_state(
                supplied_state_path=args.portfolio_state_json,
                predecessor_state_path=chained_state_path,
                decision=decision,
                out_dir=args.out_dir,
            )
        )
        reconciled = json.loads(effective_state_path.read_text())
        state_ts = pd.Timestamp(reconciled["as_of_ts"])
        state_ts = (
            state_ts.tz_localize("UTC") if state_ts.tzinfo is None
            else state_ts.tz_convert("UTC")
        )
        provenance = dict(reconciled.get("bridge_provenance") or {})
        matched = int(provenance.get("matched_positions", -1))
        overlays = int(provenance.get("live_execution_state_overlays", -2))
        if state_ts != decision:
            raise ValueError("portfolio reconciliation state is not exact-decision")
        if str(provenance.get("shadow_reference_sha256")) != _sha(chained_state_path):
            raise ValueError("portfolio reconciliation does not descend from predecessor")
        if matched != overlays or matched != len(reconciled.get("open_positions") or []):
            raise ValueError("portfolio reconciliation lacks exact actual-fill overlays")
        if not provenance.get("live_state_sha256"):
            raise ValueError("portfolio reconciliation lacks canonical live-state lineage")
    feature_state_reseal_audit = None
    if feature_state_contract is not None:
        if args.feature_state_bundle is None:
            raise ValueError(
                "sealed persisted_state_only runtime requires --feature-state-bundle"
            )
        sealed_hash = str(feature_state_contract["contract_sha256"])
        if (
            args.feature_state_contract_hash is not None
            and str(args.feature_state_contract_hash) != sealed_hash
        ):
            raise ValueError("CLI feature-state contract differs from sealed bundle")
        args.feature_state_contract_hash = sealed_hash
        sealed_tail = int(feature_state_contract["panel_tail_hours"])
        if int(args.feature_state_tail_hours) != sealed_tail:
            raise ValueError("CLI feature-state tail differs from sealed bundle")
        if args.previous_shadow_run is not None:
            expected_source = (
                args.previous_shadow_run / "feature_state" / "bundle"
            ).resolve()
            if args.feature_state_bundle.resolve() != expected_source:
                feature_state_reseal_audit = _validate_one_time_feature_state_reseal(
                    contract=feature_state_contract,
                    source_bundle=args.feature_state_bundle,
                    predecessor_bundle=expected_source,
                )
    elif args.feature_state_bundle is not None and not args.feature_state_contract_hash:
        raise ValueError("stateful features require --feature-state-contract-hash")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stages = _commands(
        bundle_path=args.inference_bundle,
        bundle=bundle,
        state_path=effective_state_path,
        decision=decision,
        out_dir=args.out_dir,
        previous_run=args.previous_shadow_run,
        portfolio_state_activation=use_supplied_state,
    )
    input_stages = (
        _stateful_input_commands(
            bundle=bundle,
            decision=decision,
            out_dir=args.out_dir,
            source_state_bundle=args.feature_state_bundle,
            state_contract_hash=str(args.feature_state_contract_hash),
            state_tail_hours=int(args.feature_state_tail_hours),
            stateful_exact_families=(
                list(feature_state_contract["stateful_exact_families"])
                if feature_state_contract is not None else []
            ),
            final14_contract_hash=(
                str(feature_state_contract["final14_contract_sha256"])
                if feature_state_contract is not None else None
            ),
            orderbook_precomposite_contract_hash=(
                str(feature_state_contract[
                    "orderbook_precomposite_contract_sha256"
                ])
                if feature_state_contract is not None else None
            ),
            required_state_kinds=(
                list(feature_state_contract["required_state_kinds"])
                if feature_state_contract is not None else []
            ),
            grid_command=stages[0][1],
        )
        if args.feature_state_bundle is not None else stages[:2]
    )
    # Candidate and feature parity are fail-before-score gates.  Never allow
    # an invalid live matrix to reach a model, admission map, or auction.
    input_reuse_audit = None
    if args.reuse_current_inputs_from is not None:
        input_reuse_audit = _reuse_current_inputs(
            source_run=args.reuse_current_inputs_from,
            previous_run=args.previous_shadow_run,
            current_run=args.out_dir,
            decision=decision,
        )
        reused_state = args.out_dir / "feature_state" / "bundle"
        if not reused_state.exists() and args.feature_state_bundle is not None:
            state_manifest_path = (
                args.feature_state_bundle / "state_bundle_manifest.json"
            )
            state_manifest = json.loads(state_manifest_path.read_text())
            expected_signal = decision - pd.Timedelta(hours=1)
            state_ts = pd.Timestamp(state_manifest["latest_state_timestamp"])
            state_ts = (
                state_ts.tz_localize("UTC")
                if state_ts.tzinfo is None else state_ts.tz_convert("UTC")
            )
            if state_ts != expected_signal:
                raise ValueError(
                    "reused current inputs require an exact-decision feature "
                    "state checkpoint"
                )
            reused_state.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(args.feature_state_bundle, reused_state)
            input_reuse_audit["feature_state_bundle_reused"] = True
            input_reuse_audit["feature_state_manifest_sha256"] = _sha(
                reused_state / "state_bundle_manifest.json"
            )
    else:
        for name, command in input_stages:
            _run(command, log_path=args.out_dir / f"{name}.log")
    predecessor_geometry_state = (
        args.previous_shadow_run / "cycle" / "score" / "geometry_k9_state"
        if args.previous_shadow_run is not None else None
    )
    use_persisted_geometry_state = bool(
        predecessor_geometry_state is not None
        and (predecessor_geometry_state / "run_manifest.json").is_file()
    )
    # A one-time migration/bootstrap has no predecessor score-state yet and
    # must reconstruct the exact activation prefix once.  Once the state is
    # present, do not concatenate the prefix before scoring: the state instead
    # supplies the strict-prior K9 history for the single current hour.
    immutable_prefix_assembly = (
        _assemble_immutable_prefix(
            previous_run=args.previous_shadow_run,
            current_run=args.out_dir,
        )
        if args.previous_shadow_run is not None and not use_persisted_geometry_state
        else None
    )
    pre_score_overlap_audit = (
        _assert_append_only_overlap(
            previous_run=args.previous_shadow_run,
            current_run=args.out_dir,
            include_predictions=False,
        )
        if args.previous_shadow_run is not None and not use_persisted_geometry_state
        else None
    )

    grid_manifest = json.loads(
        (args.out_dir / "candidate_grid" / "run_manifest.json").read_text(),
    )
    feature_manifest = json.loads(
        (args.out_dir / "features" / "feature_manifest.json").read_text(),
    )
    population = pd.read_parquet(
        args.out_dir / "candidate_grid" / "target_free_candidate_population.parquet",
    )
    eligible = pd.read_parquet(
        args.out_dir / "candidate_grid" / "eligible_candidates.parquet",
    )
    features = pd.read_parquet(
        args.out_dir / "features" / "canonical120_features.parquet",
    )
    feature_contract = json.loads(bundle.path("feature_contract").read_text())
    row_local_feature_skip_audit = _skip_incomplete_current_rows(
        decision=decision,
        fields=list(feature_contract["base_fields_by_side"]["long"]),
        eligible_path=(
            args.out_dir / "candidate_grid" / "eligible_candidates.parquet"
        ),
        population_path=(
            args.out_dir / "candidate_grid" /
            "target_free_candidate_population.parquet"
        ),
        rejection_path=(
            args.out_dir / "candidate_grid" / "candidate_rejection_audit.parquet"
        ),
        features=features,
    )
    grid_manifest_path = args.out_dir / "candidate_grid" / "run_manifest.json"
    grid_manifest = json.loads(grid_manifest_path.read_text())
    population = pd.read_parquet(
        args.out_dir / "candidate_grid" / "target_free_candidate_population.parquet",
    )
    eligible = pd.read_parquet(
        args.out_dir / "candidate_grid" / "eligible_candidates.parquet",
    )
    rejected = pd.read_parquet(
        args.out_dir / "candidate_grid" / "candidate_rejection_audit.parquet",
    )
    grid_manifest.update({
        "eligible_rows": int(len(eligible)),
        "rejected_rows": int(len(rejected)),
        "row_local_feature_skip_audit": row_local_feature_skip_audit,
    })
    grid_manifest_path.write_text(
        json.dumps(grid_manifest, indent=2, default=str) + "\n"
    )
    rejection_summary = population.groupby(
        ["side_name", "eligibility_reason"], as_index=False, dropna=False,
    ).agg(rows=("candidate_id", "size"))
    rejection_summary.to_parquet(
        args.out_dir / "candidate_grid" /
        "candidate_rejection_reason_summary.parquet",
        index=False, compression="zstd",
    )
    scoring_features = features.loc[
        features["candidate_id"].isin(set(eligible["candidate_id"])),
    ].copy()
    scoring_decision_ts = pd.to_datetime(
        scoring_features["__decision_ts__"], utc=True,
    )
    current_scoring_features = scoring_features.loc[
        scoring_decision_ts.eq(decision)
    ].copy()
    if scoring_features.empty:
        # Row-local missingness may legitimately remove the entire current
        # cross-section. No row reaches a model in that case, so retain an
        # explicit fail-closed receipt instead of applying percentage gates
        # to an empty matrix (whose means are undefined).
        feature_parity_audit = {
            "fields": int(len(feature_contract["base_fields_by_side"]["long"])),
            "rows": 0,
            "minimum_row_finite_fraction": None,
            "rows_meeting_minimum_fraction": None,
            "all_fields_complete_fraction": None,
            "minimum_per_field_finite_fraction": None,
            "checks": {
                "all_frozen_fields_present": True,
                "row_coverage_fraction_meets_cycle_gate": True,
                "complete_row_fraction_meets_cycle_gate": True,
                "every_field_meets_finite_gate": True,
                "empty_scoring_set_failed_closed": True,
            },
        }
    else:
        feature_parity_audit = validate_live_feature_frame(
            scoring_features,
            fields=list(feature_contract["base_fields_by_side"]["long"]),
            requirements=dict(bundle.payload["feature_parity"]),
        )
    if current_scoring_features.empty:
        # A market-wide delayed primitive must never be repaired after the
        # executable decision window and then scored retrospectively.  Keep
        # the complete point-in-time population and frozen features for audit,
        # route no new entries, and still run the portfolio cycle below so
        # protective/trailing/timeout exits and state advance causally.
        current_entry_data_state = "no_actionable_rows_fail_closed"
        current_feature_parity_audit = {
            "fields": int(len(feature_contract["base_fields_by_side"]["long"])),
            "rows": 0,
            "minimum_row_finite_fraction": None,
            "rows_meeting_minimum_fraction": None,
            "all_fields_complete_fraction": None,
            "minimum_per_field_finite_fraction": None,
            "checks": {
                "all_frozen_fields_present": True,
                "row_coverage_fraction_meets_cycle_gate": True,
                "complete_row_fraction_meets_cycle_gate": True,
                "every_field_meets_finite_gate": True,
                "empty_current_entry_set_failed_closed": True,
            },
        }
    else:
        current_entry_data_state = "actionable"
        current_feature_parity_audit = validate_live_feature_frame(
            current_scoring_features,
            fields=list(feature_contract["base_fields_by_side"]["long"]),
            requirements=dict(bundle.payload["feature_parity"]),
        )
    current_hour_scorer_inputs = _preserve_current_hour_scorer_inputs(
        out_dir=args.out_dir,
        decision=decision,
    )
    name, command = stages[2]
    _run(command, log_path=args.out_dir / f"{name}.log")
    cycle_manifest = json.loads(
        (args.out_dir / "cycle" / "run_manifest.json").read_text(),
    )
    if args.previous_shadow_run is not None and use_persisted_geometry_state:
        immutable_prefix_assembly = _assemble_immutable_prefix(
            previous_run=args.previous_shadow_run,
            current_run=args.out_dir,
        )
        pre_score_overlap_audit = _assert_append_only_overlap(
            previous_run=args.previous_shadow_run,
            current_run=args.out_dir,
            include_predictions=False,
        )
    append_only_overlap_audit = (
        _assert_append_only_overlap(
            previous_run=args.previous_shadow_run,
            current_run=args.out_dir,
        )
        if args.previous_shadow_run is not None else None
    )
    # The grid deliberately contains the complete activation-to-current prefix
    # so the conversion layer can rebuild rolling K9 state.  Validate the
    # frozen universe per signal cross-section; comparing the whole prefix row
    # count with the symbol count incorrectly rejects every run after the
    # activation hour.
    expected_universe_rows = int(grid_manifest["universe_rows"])
    population_signal_ts = pd.to_datetime(population["__ts__"], utc=True)
    current_signal = decision - pd.Timedelta(hours=1)
    current_population = population.loc[population_signal_ts.eq(current_signal)]
    current_symbols = current_population["__symbol__"].astype(str)
    if (
        len(current_population) != expected_universe_rows
        or current_symbols.nunique() != expected_universe_rows
    ):
        raise AssertionError(
            "current-hour feature population does not cover the frozen universe: "
            f"rows={len(current_population)} unique_symbols={current_symbols.nunique()} "
            f"expected={expected_universe_rows}"
        )
    if len(features) != len(population):
        raise AssertionError("features were not generated on the complete universe")
    if not set(eligible["candidate_id"]).issubset(set(features["candidate_id"])):
        raise AssertionError("one or more actionable identities lack complete-universe features")
    if grid_manifest.get("spread_gate") != (
        "official_kraken_signal_hour_bid_ask_bps_before_signal_plus_1h_entry"
    ):
        raise AssertionError("hourly grid did not use the contemporaneous spread gate")
    if not all(cycle_manifest.get("checks", {}).values()):
        raise AssertionError("nested strict-R3 shadow cycle failed its invariant set")

    completed_at = _utc_now()
    age_at_completion = _decision_age_seconds(
        decision=decision, now=completed_at,
    )
    completed_within_window = 0.0 <= age_at_completion <= freshness_seconds
    manifest = {
        "schema": SCHEMA,
        "mode": "shadow-only",
        "decision_ts": decision.isoformat(),
        "signal_ts": (decision - pd.Timedelta(hours=1)).isoformat(),
        "orchestration_started_at": started_at.isoformat(),
        "orchestration_completed_at": completed_at.isoformat(),
        "decision_age_at_start_seconds": age_at_start,
        "decision_age_at_completion_seconds": age_at_completion,
        "live_decision_freshness_seconds": freshness_seconds,
        "live_wall_clock_enforced": bool(args.enforce_live_wall_clock),
        "completed_within_live_decision_window": completed_within_window,
        "inference_bundle_audit": bundle_audit,
        "population_rows": int(len(population)),
        "current_population_rows": int(len(current_population)),
        "current_population_unique_symbols": int(current_symbols.nunique()),
        "eligible_rows": int(len(eligible)),
        "rejected_rows": int(len(population) - len(eligible)),
        "feature_rows": int(len(features)),
        "feature_parity_rows": int(len(scoring_features)),
        "feature_parity_audit": feature_parity_audit,
        "current_feature_parity_rows": int(len(current_scoring_features)),
        "current_entry_data_state": current_entry_data_state,
        "current_hour_scorer_inputs": current_hour_scorer_inputs,
        "row_local_feature_skip_audit": row_local_feature_skip_audit,
        "current_feature_parity_audit": current_feature_parity_audit,
        "mapped_rows": int(cycle_manifest["mapped_rows"]),
        "admitted_rows": int(cycle_manifest["admitted_rows"]),
        "portfolio_accepted_rows": int(cycle_manifest["portfolio_accepted_rows"]),
        "portfolio_state_input": str(effective_state_path),
        "portfolio_state_chained_from_previous": bool(
            chained_state_path is not None and chained_state_path.exists()
            and not use_supplied_state
        ),
        "portfolio_state_activation": bool(args.portfolio_state_activation),
        "portfolio_state_reconciliation": bool(
            args.portfolio_state_reconciliation
        ),
        "portfolio_reconciliation_bridge": reconciliation_bridge_audit,
        "portfolio_open_positions_before": int(cycle_manifest["portfolio_open_positions_before"]),
        "portfolio_open_positions_after": int(cycle_manifest["next_portfolio_open_positions"]),
        "realized_exit_rows": int(cycle_manifest["realized_exit_rows"]),
        "next_portfolio_state": str(args.out_dir / "cycle" / "next_portfolio_state.json"),
        "complete_universe_features_before_actionability_filter": True,
        "conversion_state_prefix_start": bundle.payload["activation_ts"],
        "conversion_state_replayed_from_activation": bool(
            cycle_manifest.get("geometry_k9_state_mode")
            == "bootstrap_activation_to_current"
        ),
        "conversion_state_mode": cycle_manifest.get("geometry_k9_state_mode"),
        "conversion_state_input": cycle_manifest.get("geometry_k9_state_input"),
        "conversion_state_output": cycle_manifest.get("geometry_k9_state_output"),
        "current_spread_gate": True,
        "append_only_overlap_audit": append_only_overlap_audit,
        "pre_score_overlap_audit": pre_score_overlap_audit,
        "immutable_prefix_assembly": immutable_prefix_assembly,
        "input_reuse_audit": input_reuse_audit,
        "stateful_feature_bundle_input": (
            str(args.feature_state_bundle)
            if args.feature_state_bundle is not None else None
        ),
        "stateful_feature_bundle_output": (
            str(args.out_dir / "feature_state" / "bundle")
            if (args.out_dir / "feature_state" / "bundle").is_dir() else None
        ),
        "stateful_feature_contract_hash": args.feature_state_contract_hash,
        "feature_state_reseal_audit": feature_state_reseal_audit,
        "stateful_feature_tail_hours": (
            int(args.feature_state_tail_hours)
            if args.feature_state_bundle is not None else None
        ),
        "persisted_feature_state_only": bool(feature_state_contract is not None),
        "persisted_feature_state_contract": feature_state_contract,
        "future_paths_consumed": [],
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "hashes": {
            "inference_bundle": _sha(args.inference_bundle),
            "portfolio_state": _sha(effective_state_path),
            "next_portfolio_state": _sha(
                args.out_dir / "cycle" / "next_portfolio_state.json"
            ),
            "candidate_population": _sha(args.out_dir / "candidate_grid" / "target_free_candidate_population.parquet"),
            "eligible_candidates": _sha(args.out_dir / "candidate_grid" / "eligible_candidates.parquet"),
            "features": _sha(args.out_dir / "features" / "canonical120_features.parquet"),
            "shadow_decisions": _sha(args.out_dir / "cycle" / "shadow_decisions.parquet"),
        },
        "feature_source_contract": feature_manifest.get("bar_source_contract"),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    if args.enforce_live_wall_clock and not completed_within_window:
        raise RuntimeError(
            "hourly shadow completed outside the sealed live decision window; "
            "artifact is reconciliation-only and cannot extend promotion"
        )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
