#!/usr/bin/env python3
"""Run and seal the frozen execution-EV forward pre-outcome pipeline.

The runner authenticates the successor source lock, validates a raw-stage
coverage manifest, scores all frozen final refits, applies causal resolved-only
mapping updates, recomputes one deterministic pooled global book, and publishes
only a readiness-passing population.  It never reads unresolved outcomes or
changes a model, feature list, interaction, threshold, or mapping rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.audit_execution_ev_forward_confirmation_readiness import (
    LOCK_SCHEMA,
    build_readiness,
)
from scripts.materialize_execution_ev_forward_preentry import (
    DEFAULT_CATBOOST_ROOT,
    DEFAULT_HEAD_CONTRACT,
    DEFAULT_ROLE_ROOT,
    run as materialize_preentry,
)
from scripts.score_execution_ev_forward_population import (
    DEFAULT_HEAD_ROOT,
    DEFAULT_STATE,
    run as score_population,
)
from scripts.score_packb_final_refits_forward import (
    DEFAULT_ALPHA_MANIFEST,
    DEFAULT_RESIDUAL_ROOT,
    DEFAULT_SUPPORT_CONTEXT,
    run as score_packb,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(
    "configs/execution_ev_forward_confirmation_candidate_20260728_v1.json"
)
DEFAULT_LOCK = Path(
    "data_perp/artifacts/execution_ev_forward_source_lock_20260728_v5/"
    "contract.json"
)
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
COVERAGE_SCHEMA = "execution_ev_forward_raw_coverage_v1"
UPDATE_SCHEMA = "execution_ev_forward_resolved_updates_v1"
SEAL_SCHEMA = "execution_ev_forward_preoutcome_seal_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path


def _verify_file(record: Mapping[str, Any], *, name: str) -> Path:
    path = _resolve(record.get("path"))
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    expected = str(record.get("sha256", ""))
    if not expected or _sha256(path) != expected:
        raise ValueError(f"{name} hash mismatch")
    return path


def _strict_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be true/false or 1/0")


def _identity_hash(frame: pd.DataFrame) -> str:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError("candidate identity columns missing: " + ", ".join(missing))
    ordered = frame.loc[:, list(IDENTITY)].astype(str).sort_values(
        list(IDENTITY), kind="mergesort"
    )
    payload = "\n".join("\x1f".join(row) for row in ordered.itertuples(
        index=False, name=None
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_coverage_manifest(
    manifest_path: Path,
    *,
    candidate_path: Path,
    candidates: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Authenticate raw coverage and derive the only accepted complete-day CSV."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != COVERAGE_SCHEMA:
        raise ValueError(f"coverage manifest must use {COVERAGE_SCHEMA}")
    candidate_record = manifest.get("candidate_features", {})
    if _verify_file(candidate_record, name="coverage candidate features").resolve() != (
        candidate_path.resolve()
    ):
        raise ValueError("coverage manifest points to a different candidate matrix")
    for index, record in enumerate(manifest.get("source_manifests", [])):
        _verify_file(record, name=f"raw source manifest {index}")
    if not manifest.get("source_manifests"):
        raise ValueError("coverage manifest requires at least one raw source manifest")

    decision_column = "execution_decision_utc"
    decisions = pd.to_datetime(
        candidates[decision_column], utc=True, errors="raise"
    )
    first = pd.Timestamp(spec["first_decision_exclusive_utc"])
    last = pd.Timestamp(spec["requested_last_decision_utc"])
    if decisions.min() <= first or decisions.max() > last:
        raise ValueError("candidate decisions fall outside the frozen fixed window")
    observed = candidates.assign(
        __utc_date__=decisions.dt.normalize()
    ).groupby("__utc_date__").agg(
        rows=("candidate_id", "size"),
        both_sides=(
            "side_name",
            lambda values: {"long", "short"}.issubset(
                set(values.astype(str).str.lower())
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    seen: set[pd.Timestamp] = set()
    for record in manifest.get("days", []):
        date = pd.Timestamp(record["utc_date"])
        if date.tzinfo is None:
            raise ValueError("coverage utc_date must be timezone-aware")
        date = date.tz_convert("UTC").normalize()
        if date in seen:
            raise ValueError("coverage manifest contains duplicate UTC days")
        seen.add(date)
        source_complete = _strict_bool(
            record.get("all_required_point_in_time_features_complete"),
            field="all_required_point_in_time_features_complete",
        )
        declared_both = _strict_bool(
            record.get("both_sides_complete"), field="both_sides_complete"
        )
        observed_rows = int(observed.loc[date, "rows"]) if date in observed.index else 0
        observed_both = (
            bool(observed.loc[date, "both_sides"]) if date in observed.index else False
        )
        declared_rows = int(record.get("candidate_rows", observed_rows))
        if declared_rows != observed_rows:
            raise ValueError(f"coverage candidate row mismatch on {date.date()}")
        if declared_both != observed_both:
            raise ValueError(f"coverage side completeness mismatch on {date.date()}")
        complete = source_complete and declared_both and observed_rows > 0
        rows.append(
            {
                "utc_date": date,
                "rows": observed_rows,
                "both_sides": observed_both,
                "complete": complete,
            }
        )
    coverage = pd.DataFrame(rows).sort_values("utc_date").reset_index(drop=True)
    observed_dates = set(observed.index)
    if set(coverage["utc_date"]) != observed_dates:
        raise ValueError("coverage manifest must declare every observed decision day")
    return coverage


def validate_update_manifest(
    manifest_path: Path,
    *,
    candidate_path: Path,
    candidates: pd.DataFrame,
) -> Path:
    """Authenticate exact-policy update provenance and required resolved coverage."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != UPDATE_SCHEMA:
        raise ValueError(f"update manifest must use {UPDATE_SCHEMA}")
    if manifest.get("score_binding") != "generated_or_verified_by_locked_scorer":
        raise ValueError("update manifest score_binding contract mismatch")
    if str(manifest.get("candidate_features_sha256")) != _sha256(candidate_path):
        raise ValueError("update manifest candidate-feature hash mismatch")
    updates_path = _verify_file(
        manifest.get("updates", {}), name="resolved update ledger"
    )
    _verify_file(
        manifest.get("exact_policy_label_manifest", {}),
        name="exact-policy label manifest",
    )
    _verify_file(
        manifest.get("source_manifest", {}), name="resolved update source manifest"
    )
    updates = pd.read_parquet(updates_path)
    required = {*IDENTITY, "execution_label_end_utc"}
    missing = sorted(required.difference(updates.columns))
    if missing:
        raise ValueError("resolved update ledger missing: " + ", ".join(missing))
    if updates.duplicated(list(IDENTITY)).any():
        raise ValueError("resolved update ledger contains duplicate identities")
    decisions = pd.to_datetime(
        candidates["execution_decision_utc"], utc=True, errors="raise"
    )
    required_mask = (
        decisions + pd.Timedelta(hours=12)
    ).lt(decisions.max())
    required_keys = set(
        candidates.loc[required_mask, list(IDENTITY)].itertuples(
            index=False, name=None
        )
    )
    update_keys = set(
        updates.loc[:, list(IDENTITY)].itertuples(index=False, name=None)
    )
    if update_keys != required_keys:
        raise ValueError(
            "resolved updates must exactly cover candidates resolved before "
            "the final decision"
        )
    return updates_path


def _source_lock_check(
    spec: Mapping[str, Any],
    source_lock_path: Path,
) -> Mapping[str, Any]:
    if (source_lock_path.parent / "SUPERSEDED.json").exists():
        raise ValueError("source lock is explicitly superseded")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if (
        source_lock.get("schema") != LOCK_SCHEMA
        or source_lock.get("status") != "frozen_before_forward_outcomes"
    ):
        raise ValueError("invalid source lock")
    current = build_readiness(spec, root=ROOT, stage="source_lock")
    if not current["ready"]:
        raise ValueError(f"current source contract is not ready: {current['blockers']}")
    if current["lock_fingerprint"] != source_lock.get("lock_fingerprint"):
        raise ValueError("source-lock fingerprint mismatch")
    return source_lock


def _prepublish_spec(
    spec: Mapping[str, Any],
    *,
    scored_dir: Path,
) -> dict[str, Any]:
    candidate = deepcopy(spec)
    scored = candidate["scored_population"]
    population = scored_dir / "scored_population.parquet"
    coverage = scored_dir / "daily_coverage.csv"
    scored["path"] = str(population)
    scored["sha256"] = _sha256(population)
    scored["daily_coverage"]["path"] = str(coverage)
    scored["daily_coverage"]["sha256"] = _sha256(coverage)
    seal = scored_dir / "preoutcome_seal.json"
    if seal.is_file():
        scored["preoutcome_seal"]["path"] = str(seal)
        scored["preoutcome_seal"]["sha256"] = _sha256(seal)
    return candidate


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    source_lock = _source_lock_check(spec, args.source_lock)
    candidates = pd.read_parquet(args.candidate_features)
    if candidates.duplicated(list(IDENTITY)).any():
        raise ValueError("candidate feature matrix contains duplicate identities")
    coverage = validate_coverage_manifest(
        args.coverage_manifest,
        candidate_path=args.candidate_features,
        candidates=candidates,
        spec=spec,
    )
    updates_path = validate_update_manifest(
        args.resolved_updates_manifest,
        candidate_path=args.candidate_features,
        candidates=candidates,
    )
    run_key = hashlib.sha256(
        (
            _sha256(args.candidate_features)
            + _sha256(args.coverage_manifest)
            + _sha256(args.resolved_updates_manifest)
            + str(source_lock["lock_fingerprint"])
        ).encode("utf-8")
    ).hexdigest()
    staging_root = args.staging_parent / f"execution_ev_forward_staging_{run_key[:16]}"
    if staging_root.exists():
        raise FileExistsError(staging_root)
    staging_root.mkdir(parents=True)
    coverage_path = staging_root / "verified_daily_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    packb_dir = staging_root / "packb"
    preentry_dir = staging_root / "preentry"
    scored_dir = staging_root / "scored"
    score_packb(
        Namespace(
            candidate_features=args.candidate_features,
            contract_spec=args.spec,
            alpha_manifest=args.alpha_manifest,
            support_context=args.support_context,
            residual_root=args.residual_root,
            output_dir=packb_dir,
        )
    )
    materialize_preentry(
        Namespace(
            packb_context=packb_dir / "packb_forward_context.parquet",
            contract_spec=args.spec,
            role_root=args.role_root,
            catboost_root=args.catboost_root,
            head_feature_contract=args.head_feature_contract,
            output_dir=preentry_dir,
        )
    )
    score_population(
        Namespace(
            preentry=preentry_dir / "preentry.parquet",
            head_root=args.head_root,
            calibrator_state=args.calibrator_state,
            resolved_updates=updates_path,
            complete_days=coverage_path,
            output_dir=scored_dir,
        )
    )
    final_dir = _resolve(spec["scored_population"]["path"]).parent
    if final_dir.exists():
        raise FileExistsError(final_dir)
    scored_manifest_path = scored_dir / "manifest.json"
    scored_manifest = json.loads(scored_manifest_path.read_text(encoding="utf-8"))
    scored_manifest["outputs"]["scored_population"]["path"] = str(
        final_dir / "scored_population.parquet"
    )
    scored_manifest["outputs"]["daily_coverage"]["path"] = str(
        final_dir / "daily_coverage.csv"
    )
    _write_json(scored_manifest_path, scored_manifest)
    scored = pd.read_parquet(scored_dir / "scored_population.parquet")
    seal_core = {
        "schema": SEAL_SCHEMA,
        "status": "sealed_preoutcome_population_not_performance_evidence",
        "source_lock": {
            "path": args.source_lock,
            "sha256": _sha256(args.source_lock),
            "fingerprint": source_lock["lock_fingerprint"],
        },
        "run_key": run_key,
        "candidate_identity_sha256": _identity_hash(scored),
        "candidate_features": {
            "path": args.candidate_features,
            "sha256": _sha256(args.candidate_features),
        },
        "coverage_manifest": {
            "path": args.coverage_manifest,
            "sha256": _sha256(args.coverage_manifest),
        },
        "resolved_updates_manifest": {
            "path": args.resolved_updates_manifest,
            "sha256": _sha256(args.resolved_updates_manifest),
        },
        "intermediates": {
            "packb_manifest": {
                "path": packb_dir / "manifest.json",
                "sha256": _sha256(packb_dir / "manifest.json"),
            },
            "preentry_manifest": {
                "path": preentry_dir / "manifest.json",
                "sha256": _sha256(preentry_dir / "manifest.json"),
            },
            "scored_manifest": {
                "path": final_dir / "manifest.json",
                "sha256": _sha256(scored_manifest_path),
            },
        },
        "outputs": {
            "scored_population": {
                "path": final_dir / "scored_population.parquet",
                "sha256": _sha256(scored_dir / "scored_population.parquet"),
            },
            "daily_coverage": {
                "path": final_dir / "daily_coverage.csv",
                "sha256": _sha256(scored_dir / "daily_coverage.csv"),
            },
        },
    }
    seal_core["seal_fingerprint"] = hashlib.sha256(
        json.dumps(
            _safe(seal_core), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    _write_json(scored_dir / "preoutcome_seal.json", seal_core)

    candidate_spec = _prepublish_spec(spec, scored_dir=scored_dir)
    readiness = build_readiness(
        candidate_spec,
        root=ROOT,
        stage="preoutcome",
        source_lock=source_lock,
    )
    readiness_path = staging_root / "readiness_preoutcome.json"
    _write_json(readiness_path, readiness)
    if not readiness["ready"]:
        rejected = {
            "schema": SEAL_SCHEMA,
            "status": "not_ready_not_published",
            "run_key": run_key,
            "blockers": readiness["blockers"],
        }
        _write_json(staging_root / "PREOUTCOME_NOT_READY.json", rejected)
        return rejected

    os.replace(scored_dir, final_dir)
    runtime_spec = _prepublish_spec(spec, scored_dir=final_dir)
    _write_json(final_dir / "runtime_spec.json", runtime_spec)
    final_readiness = build_readiness(
        runtime_spec,
        root=ROOT,
        stage="preoutcome",
        source_lock=source_lock,
    )
    _write_json(args.readiness_report, final_readiness)
    if not final_readiness["ready"]:
        raise RuntimeError(
            f"published population failed final readiness: "
            f"{final_readiness['blockers']}"
        )
    return seal_core


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--resolved-updates-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--alpha-manifest", type=Path, default=DEFAULT_ALPHA_MANIFEST)
    parser.add_argument("--support-context", type=Path, default=DEFAULT_SUPPORT_CONTEXT)
    parser.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL_ROOT)
    parser.add_argument("--role-root", type=Path, default=DEFAULT_ROLE_ROOT)
    parser.add_argument("--catboost-root", type=Path, default=DEFAULT_CATBOOST_ROOT)
    parser.add_argument(
        "--head-feature-contract", type=Path, default=DEFAULT_HEAD_CONTRACT
    )
    parser.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    parser.add_argument("--calibrator-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--staging-parent",
        type=Path,
        default=Path("data_perp/artifacts"),
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_forward_source_lock_20260728_v5/"
            "readiness_preoutcome.json"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(_safe(run(_parser())), indent=2, sort_keys=True))
