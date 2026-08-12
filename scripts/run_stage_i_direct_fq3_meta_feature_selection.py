#!/usr/bin/env python3
"""Select and tune a promotable direct-FQ3 meta contract for R3/S/O bases.

The input is a completed side-local base selector OOF.  Its native same-side
score and probability states enter meta without an EV conversion.  Every MDA
and HPO refit derives outcome-CDF/q33 labels from its own training rows.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import lgbm_pipeline
from extreme_price_movements.config import CFG
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_CORRELATION_POLICIES,
    STAGE_I_CORRELATION_POLICY_GROUPED_PRESERVE,
    STAGE_I_SELECTOR_SCHEMA,
    StageIHeadContract,
    run_stage_i_head_selection,
)
from extreme_price_movements.stage_i_model_hpo import run_stage_i_model_hpo
from extreme_price_movements.stage_i_mda_support import (
    MDA_SUPPORT_MODES,
    build_stage_i_mda_training_support,
    restrict_stage_i_mda_training_support,
)
from extreme_price_movements.stage_i_ranking import RANKING_POLICY, stable_stage_i_topk_positions
from extreme_price_movements.stage_i_target_adapter import (
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
    StageITargetContract,
    bind_target_contract,
    file_sha256,
)
from extreme_price_movements.stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
    _direct_trust,
)


SCHEMA = "stage_i_adapter_meta_feature_selection_v2"
IDENTITY = ("candidate_id", "__ts__", "__symbol__")
RESUME_ATTEMPT_SCHEMA = "stage_i_direct_fq3_resume_attempt_v1"
RESUME_COMPLETE_SCHEMA = "stage_i_direct_fq3_resume_complete_v1"


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sha(value: Any) -> str:
    return sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bounded_relief_checkpoint_root(
    root_destination: Path, *, feature_chunk_size: int, anchor_chunk_size: int,
) -> Path:
    """Version sliced Relief state by its immutable geometry/chunk contract."""
    feature_chunk_size, anchor_chunk_size = int(feature_chunk_size), int(anchor_chunk_size)
    if feature_chunk_size < 1 or anchor_chunk_size < 1:
        raise ValueError("bounded Relief chunk sizes must be positive")
    return root_destination / f"_bounded_relief_v2_f{feature_chunk_size}_a{anchor_chunk_size}"


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected JSON object")
    return raw


def _align(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    left = pd.MultiIndex.from_frame(source.loc[:, list(IDENTITY)])
    right = pd.MultiIndex.from_frame(target.loc[:, list(IDENTITY)])
    positions = left.get_indexer(right)
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise ValueError("direct-FQ3 identity alignment failed")
    return positions


def _load_feature_sidecar(
    path: Path | None,
    *,
    fields: tuple[str, ...],
    selector_features: pd.DataFrame,
    min_coverage: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a declared latent/archetype sidecar against the immutable selector ids.

    Stage-I's raw selector panel deliberately excludes realised-path archetypes
    and optional frozen latent-state projections.  This bridge makes such a
    meta-only addition explicit, identity-bound and auditable.  It never
    silently uses all columns from a sidecar: callers must name every field.
    """
    if path is None:
        if fields:
            raise ValueError("feature-sidecar fields require --feature-sidecar")
        return pd.DataFrame(index=selector_features.index), {"enabled": False, "fields": []}
    if not fields:
        raise ValueError("--feature-sidecar requires one or more --feature-sidecar-field values")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError("feature-sidecar minimum coverage must lie in (0,1]")
    if not path.is_file():
        raise FileNotFoundError(f"feature sidecar does not exist: {path}")
    sidecar = pd.read_parquet(path)
    required = set(IDENTITY).union(fields)
    if missing := required.difference(sidecar.columns):
        raise ValueError(f"feature sidecar lacks declared identity/fields: {sorted(missing)}")
    if sidecar.duplicated(list(IDENTITY)).any():
        raise ValueError("feature sidecar has duplicate candidate identities")
    positions = _align(sidecar, selector_features)
    work = sidecar.iloc[positions].loc[:, list(fields)].reset_index(drop=True).copy()
    if any(name in selector_features.columns for name in fields):
        overlap = sorted(set(fields).intersection(selector_features.columns))
        raise ValueError(f"feature sidecar collides with selector feature names: {overlap}")
    coverage: dict[str, float] = {}
    for name in fields:
        values = pd.to_numeric(work[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work[name] = values.astype(np.float32)
        observed = float(values.notna().mean())
        coverage[str(name)] = observed
        if observed < float(min_coverage):
            raise ValueError(
                f"feature sidecar field {name!r} has coverage {observed:.4f}, below {float(min_coverage):.4f}"
            )
    return work, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "fields": list(fields),
        "coverage": coverage,
        "minimum_coverage": float(min_coverage),
        "identity_contract": list(IDENTITY),
    }


def _validate_pristine_orchestrator_directory(destination: Path, *, side: str) -> dict[str, Any]:
    """Accept only the request-only directory created by the bounded parent."""
    contents = {path.name for path in destination.iterdir()}
    request_path = destination / "orchestrator_request.json"
    if contents != {"orchestrator_request.json"} or not request_path.is_file():
        raise FileExistsError(
            f"{side}: partial direct-FQ3 selector artifacts are not resumable: {sorted(contents)}"
        )
    receipt = _read_json(request_path)
    if str(receipt.get("schema", "")) != "stage_i_bounded_side_orchestrator_v2":
        raise ValueError(f"{side}: unsupported orchestrator request receipt schema")
    if str(receipt.get("side", "")).lower() != side:
        raise ValueError(f"{side}: orchestrator request receipt is cross-side")
    request_sha = str(receipt.get("request_sha256", ""))
    command = receipt.get("command")
    if len(request_sha) != 64 or not isinstance(command, list) or "--resume" not in command:
        raise ValueError(f"{side}: orchestrator request receipt lacks immutable resume lineage")
    return receipt


def _validate_orchestrator_receipt(destination: Path, *, side: str) -> dict[str, Any]:
    """Validate the bounded parent's receipt without making the root pristine.

    This is deliberately separate from ``_validate_pristine...``: an
    interrupted selector root is evidence, not a scratch directory.  A later
    attempt may live below it only after this receipt has bound the side to the
    parent request.
    """
    request_path = destination / "orchestrator_request.json"
    if not request_path.is_file():
        raise FileExistsError(f"{side}: unbound partial direct-FQ3 selector root")
    receipt = _read_json(request_path)
    if str(receipt.get("schema", "")) != "stage_i_bounded_side_orchestrator_v2":
        raise ValueError(f"{side}: unsupported orchestrator request receipt schema")
    if str(receipt.get("side", "")).lower() != side:
        raise ValueError(f"{side}: orchestrator request receipt is cross-side")
    if len(str(receipt.get("request_sha256", ""))) != 64:
        raise ValueError(f"{side}: orchestrator request receipt lacks immutable request lineage")
    return receipt


def _bootstrap_direct_runner_receipt(
    destination: Path,
    *,
    side: str,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a root created by this CLI before its first bounded return.

    The bounded side orchestrator normally creates the receipt.  A user may
    also invoke this CLI directly, however; in that case the first bounded
    univariate/Relief/MDA return used to leave a valid checkpoint below an
    *unbound* root, which the next ``--resume`` invocation correctly refused.

    Only the literal root-level files/directories this runner creates before a
    completed manifest are eligible for this bootstrap.  We do not adopt a
    generic partial selector directory, and the receipt is bound to the
    already-derived immutable request lineage.
    """
    request_path = destination / "orchestrator_request.json"
    if request_path.exists():
        return _validate_orchestrator_receipt(destination, side=side)
    allowed = {
        "base_candidate_handoff.parquet",
        "mda",
        "_bounded_univariate",
        "_bounded_relief",
        "_bounded_relief_v2",
        "_bounded_mda",
        "_stability_grid",
        "_hpo_halving",
        "_resume_attempts",
    }
    contents = {path.name for path in destination.iterdir()}
    unknown = contents.difference(allowed)
    if unknown:
        raise FileExistsError(
            f"{side}: unbound partial direct-FQ3 selector root contains "
            f"unknown artifacts: {sorted(unknown)}"
        )
    # Geometry-versioned Relief roots are created as sibling directories.
    if any(
        not name.startswith("_bounded_relief_v2_f")
        for name in contents
        if name.startswith("_bounded_relief_v2_f")
    ):
        raise FileExistsError(f"{side}: unbound direct-FQ3 Relief namespace drift")
    receipt = {
        "schema": "stage_i_bounded_side_orchestrator_v2",
        "side": side,
        "request_sha256": _sha(
            {
                "schema": "stage_i_direct_fq3_self_bootstrap_v1",
                "side": side,
                "attempt_lineage": dict(lineage),
            }
        ),
        "command": ["python3", __file__, "--side", side, "--resume"],
        "origin": "direct_runner_self_bootstrap_v1",
        "attempt_lineage": _safe(dict(lineage)),
    }
    request_path.write_text(
        json.dumps(_safe(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _partial_evidence_inventory(root: Path) -> dict[str, str]:
    """Hash existing interrupted evidence, never including later attempts."""
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "_resume_attempts" in path.parts:
            continue
        inventory[str(path.relative_to(root))] = file_sha256(path)
    return inventory


def _attempt_request_lineage(
    *, selector_manifest_sha: str, selector_feature_contract_sha: str,
    base_manifest_sha: str, target_contract_sha: str, correlation_policy: str,
    hpo_trials: int, hpo_patience: int, base_candidate_fraction: float,
    mda_support_mode: str,
) -> dict[str, Any]:
    """The immutable inputs an isolated restart must bind before it can run."""
    return {
        "selector_sample_manifest_sha256": selector_manifest_sha,
        "selector_feature_contract_sha256": selector_feature_contract_sha,
        "base_selector_manifest_sha256": base_manifest_sha,
        "target_contract_sha256": target_contract_sha,
        "correlation_policy": correlation_policy,
        "hpo_trials": int(hpo_trials), "hpo_patience": int(hpo_patience),
        "base_candidate_fraction": float(base_candidate_fraction),
        "mda_support_mode": str(mda_support_mode),
    }


def _safe_attempt_child(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("resume completion lacks an attempt path")
    candidate = (root / value).resolve()
    attempts = (root / "_resume_attempts").resolve()
    if attempts not in candidate.parents or candidate.parent != attempts:
        raise ValueError("resume completion points outside its attempt root")
    return candidate


def _prepare_clean_resume_attempt(
    root: Path, *, side: str, lineage: dict[str, Any],
) -> tuple[Path, bool]:
    """Return a clean isolated attempt, or a validated completed attempt.

    The lower selector has no durable, validated round-restart API.  We
    therefore do not pretend that arbitrary ``mda/`` files can be resumed.
    Every interrupted attempt is retained read-only and a new attempt starts
    from immutable source contracts.  Only a complete child manifest plus its
    output hashes is reusable.
    """
    receipt = _validate_orchestrator_receipt(root, side=side)
    completion_path = root / "resume_complete.json"
    if completion_path.is_file():
        completion = _read_json(completion_path)
        if completion.get("schema") != RESUME_COMPLETE_SCHEMA or completion.get("side") != side:
            raise ValueError(f"{side}: resume completion lineage is invalid")
        if completion.get("attempt_lineage") != lineage:
            raise ValueError(f"{side}: completed resume attempt lineage drift")
        attempt = _safe_attempt_child(root, completion.get("attempt_relative_path"))
        manifest_path = attempt / "manifest.json"
        if not manifest_path.is_file() or completion.get("attempt_manifest_sha256") != file_sha256(manifest_path):
            raise ValueError(f"{side}: completed resume attempt manifest hash drift")
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("side") != side:
            raise ValueError(f"{side}: resume attempt is not a complete side artifact")
        for relative, expected in dict(manifest.get("artifact_sha256", {})).items():
            if file_sha256(attempt / relative) != expected:
                raise ValueError(f"{side}: resumed attempt artifact hash drift: {relative}")
        return attempt, True

    attempts = root / "_resume_attempts"
    attempts.mkdir(exist_ok=True)
    existing = {path.name for path in attempts.iterdir()}
    index = 1
    while f"attempt-{index:04d}" in existing:
        index += 1
    attempt = attempts / f"attempt-{index:04d}"
    attempt.mkdir()
    evidence = _partial_evidence_inventory(root)
    request = {
        "schema": RESUME_ATTEMPT_SCHEMA, "side": side,
        "parent_orchestrator_request_sha256": receipt["request_sha256"],
        "attempt_lineage": lineage, "attempt_lineage_sha256": _sha(lineage),
        "prior_evidence_inventory": evidence,
        "prior_evidence_inventory_sha256": _sha(evidence),
        "resume_policy": "clean_restart_no_partial_mda_checkpoint_reuse",
    }
    (attempt / "attempt_request.json").write_text(
        json.dumps(_safe(request), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return attempt, False


def _publish_completed_resume_attempt(
    root: Path, *, side: str, attempt: Path, lineage: dict[str, Any],
) -> None:
    """Publish a pointer only after the attempt wrote a valid complete manifest."""
    manifest_path = attempt / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("side") != side:
        raise ValueError(f"{side}: cannot publish an incomplete resume attempt")
    for relative, expected in dict(manifest.get("artifact_sha256", {})).items():
        if file_sha256(attempt / relative) != expected:
            raise ValueError(f"{side}: cannot publish resume attempt with hash drift: {relative}")
    relative = attempt.relative_to(root)
    completion = {
        "schema": RESUME_COMPLETE_SCHEMA, "side": side,
        "attempt_relative_path": str(relative),
        "attempt_manifest_sha256": file_sha256(manifest_path),
        "attempt_lineage": lineage,
    }
    pointer = root / "resume_complete.json"
    if pointer.exists():
        raise FileExistsError(f"{side}: refusing to overwrite existing resume completion pointer")
    pointer.write_text(json.dumps(_safe(completion), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_base(root: Path, side: str) -> tuple[pd.DataFrame, dict[str, Any], StageITargetContract, str]:
    manifest_path, oof_path = root / side / "manifest.json", root / side / "selector_base_oof.parquet"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("side") != side:
        raise ValueError(f"{side}: base selector is incomplete or cross-side")
    if manifest.get("selector_base_oof_sha256") != file_sha256(oof_path):
        raise ValueError(f"{side}: base OOF hash drift")
    audit = manifest.get("hpo_oof_regeneration_fold_audit")
    if not isinstance(audit, list) or not audit or any(not bool(row.get("strict_prior_resolved")) for row in audit):
        raise ValueError(f"{side}: base OOF lacks strict prior-resolved provenance")
    contract = StageITargetContract.from_dict(manifest["target_contract"])
    if contract.family not in {LEGACY_R3_MULTICLASS3, SOFT_SCALAR_S, CUMULATIVE_ORDINAL5_O}:
        raise ValueError(f"{side}: unsupported direct base family {contract.family}")
    frame = pd.read_parquet(oof_path)
    required = {*IDENTITY, "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps", "base_raw_score"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{side}: base OOF lacks {sorted(missing)}")
    if not frame.side_name.astype(str).str.lower().eq(side).all():
        raise ValueError(f"{side}: base OOF contains cross-side rows")
    return frame, manifest, contract, file_sha256(manifest_path)


def _native_handoff(base: pd.DataFrame, family: str) -> tuple[pd.DataFrame, tuple[str, ...], tuple[float, float]]:
    score = pd.to_numeric(base.base_raw_score, errors="coerce").to_numpy(np.float32)
    if family == LEGACY_R3_MULTICLASS3:
        source_states = ("r3_p_adverse", "r3_p_weak", "r3_p_clear")
        domain = (-1.0, 1.0)
    else:
        source_states = tuple(sorted(
            (name for name in base.columns if str(name).startswith("base_state_p")),
            key=lambda name: int(str(name).rsplit("p", 1)[1]),
        ))
        expected = 2 if family == SOFT_SCALAR_S else 5
        if len(source_states) != expected:
            raise ValueError(f"direct base state-width drift: expected {expected}, got {len(source_states)}")
        domain = (0.0, 1.0)
    simplex = base.loc[:, list(source_states)].to_numpy(np.float32)
    finite = np.isfinite(score) & np.isfinite(simplex).all(axis=1)
    if finite.any() and (
        (simplex[finite] < 0).any() or not np.allclose(simplex[finite].sum(axis=1), 1.0, atol=1e-5)
        or (score[finite] < domain[0] - 1e-6).any() or (score[finite] > domain[1] + 1e-6).any()
    ):
        raise ValueError("direct base OOF is not a finite native score/simplex")
    if family == LEGACY_R3_MULTICLASS3 and finite.any() and not np.allclose(
        score[finite], simplex[finite, 2] - simplex[finite, 0], atol=1e-6,
    ):
        raise ValueError("R3 native score must equal P(clear)-P(adverse)")
    output = pd.DataFrame({"base_raw_score": score})
    state_names = tuple(f"base_state_p{index}" for index in range(len(source_states)))
    for index, name in enumerate(state_names):
        output[name] = simplex[:, index]
    trust = _direct_trust(simplex[finite])
    for name in trust:
        values = np.full(len(base), np.nan, dtype=np.float32)
        values[finite] = trust[name].to_numpy(np.float32)
        output[name] = values
    return output, state_names, domain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), action="append", default=[])
    parser.add_argument("--required-regime-feature", action="append", required=True)
    parser.add_argument("--required-context-feature", action="append", required=True)
    parser.add_argument("--hpo-trials", type=int, default=60)
    parser.add_argument("--hpo-patience", type=int, default=15)
    parser.add_argument("--base-candidate-fraction", type=float, default=1.0)
    parser.add_argument(
        "--feature-sidecar", type=Path,
        help="Identity-bound meta-only latent/archetype feature parquet; fields must be declared below.",
    )
    parser.add_argument(
        "--feature-sidecar-field", action="append", default=[],
        help="One permitted numeric latent/archetype feature from --feature-sidecar (repeatable).",
    )
    parser.add_argument("--feature-sidecar-min-coverage", type=float, default=0.90)
    parser.add_argument("--target-neutral-cache-dir", type=Path)
    parser.add_argument("--dedicated-mda-reference", choices=("full-selector-side",), default="full-selector-side")
    parser.add_argument(
        "--mda-support-mode", choices=MDA_SUPPORT_MODES, default="full",
        help="Use full realised-path MDA support or the target-only negative control.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--bounded-univariate", action="store_true",
        help=(
            "Persist and resume the normal univariate selector in one deterministic "
            "feature chunk per invocation. Incomplete chunks are not selectable."
        ),
    )
    parser.add_argument(
        "--bounded-univariate-chunk-features", type=int, default=25,
        help="Feature columns per bounded univariate invocation (default: 25).",
    )
    parser.add_argument(
        "--bounded-mda", action="store_true",
        help=(
            "Persist one complete dedicated chronological first-pass MDA cohort per "
            "invocation. Incomplete cohort evidence is never selectable."
        ),
    )
    parser.add_argument(
        "--bounded-relief", action="store_true",
        help=(
            "Persist and resume the normal ReliefF rescue one deterministic "
            "archetype/repeat task per invocation. Incomplete rescue evidence is not selectable."
        ),
    )
    parser.add_argument(
        "--bounded-relief-feature-chunk-size", type=int, default=64,
        help="Feature columns per durable ReliefF score block (default: 64; cache-mode only).",
    )
    parser.add_argument(
        "--bounded-relief-anchor-chunk-size", type=int, default=128,
        help="Anchor rows per durable ReliefF neighbour-geometry block (default: 128).",
    )
    parser.add_argument(
        "--correlation-policy", choices=sorted(STAGE_I_CORRELATION_POLICIES),
        default=STAGE_I_CORRELATION_POLICY_GROUPED_PRESERVE,
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.base_candidate_fraction <= 1.0:
        raise ValueError("base candidate fraction must lie in (0,1]")
    if args.bounded_univariate and int(args.bounded_univariate_chunk_features) < 1:
        raise ValueError("--bounded-univariate-chunk-features must be positive")
    if args.bounded_relief and (
        int(args.bounded_relief_feature_chunk_size) < 1
        or int(args.bounded_relief_anchor_chunk_size) < 1
    ):
        raise ValueError("bounded Relief feature and anchor chunk sizes must be positive")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite direct-FQ3 selector: {args.output_dir}")
    selector_manifest = _read_json(args.selector_dir / "manifest.json")
    integrity = selector_manifest.get("artifact_integrity")
    if not isinstance(integrity, dict) or integrity.get("schema") != "stage_i_selector_artifact_integrity_v1":
        raise ValueError("selector manifest lacks immutable artifact integrity")
    ledger_path = args.selector_dir / "selector_ledger.parquet"
    features_path = args.selector_dir / "selector_features.parquet"
    for key, path in (("selector_ledger_sha256", ledger_path), ("selector_features_sha256", features_path)):
        if integrity.get(key) != file_sha256(path):
            raise ValueError(f"selector artifact hash drift: {path.name}")
    ledger = pd.read_parquet(ledger_path)
    feature_panel = pd.read_parquet(features_path)
    if not ledger.loc[:, list(IDENTITY)].equals(feature_panel.loc[:, list(IDENTITY)]):
        raise ValueError("selector ledger/features identity drift")
    sidecar_fields = tuple(dict.fromkeys(map(str, args.feature_sidecar_field)))
    sidecar_panel, sidecar_lineage = _load_feature_sidecar(
        args.feature_sidecar,
        fields=sidecar_fields,
        selector_features=feature_panel,
        min_coverage=float(args.feature_sidecar_min_coverage),
    )
    raw_panel = pd.concat(
        [feature_panel.drop(columns=list(IDENTITY)).reset_index(drop=True), sidecar_panel],
        axis=1,
    )
    selector_manifest_sha = file_sha256(args.selector_dir / "manifest.json")
    selector_feature_contract_sha = file_sha256(args.selector_dir / "selector_feature_contract.json")
    cache_dir = args.target_neutral_cache_dir or Path(str(args.selector_dir) + "_target_neutral_cache_v1")
    cache_manifest = _read_json(cache_dir / "manifest.json")
    cache_request_sha = str(cache_manifest.get("request_sha256", ""))
    if len(cache_request_sha) != 64:
        raise ValueError("direct-FQ3 selector requires the immutable target-neutral cache lineage")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    summaries = []
    for side in list(dict.fromkeys(args.side or ["long", "short"])):
        base, base_manifest, base_contract, base_manifest_sha = _load_base(args.base_selection_dir, side)
        source_mask = ledger.side_name.astype(str).str.lower().eq(side)
        side_ledger = ledger.loc[source_mask].reset_index(drop=True)
        side_raw = raw_panel.loc[source_mask].reset_index(drop=True)
        positions = _align(side_ledger, base)
        side_ledger, side_raw = side_ledger.iloc[positions].reset_index(drop=True), side_raw.iloc[positions].reset_index(drop=True)
        handoff, state_names, domain = _native_handoff(base, base_contract.family)
        pre_finite_filter_rows = int(len(base))
        generated_handoff_finite_counts = {
            str(column): int(np.isfinite(pd.to_numeric(handoff[column], errors="coerce")).sum())
            for column in handoff.columns
        }
        valid = (
            np.isfinite(handoff.to_numpy(np.float32)).all(axis=1)
            & np.isfinite(pd.to_numeric(base.exact_net_bps, errors="coerce"))
            & np.isfinite(pd.to_numeric(base.exact_gross_bps, errors="coerce"))
        )
        if int(valid.sum()) < 500:
            raise ValueError(f"{side}: insufficient finite direct-base OOF support")
        base, side_ledger, side_raw, handoff = (
            item.loc[valid].reset_index(drop=True) for item in (base, side_ledger, side_raw, handoff)
        )
        if not np.allclose(
            pd.to_numeric(base.exact_gross_bps, errors="raise").to_numpy(float) - 100.0,
            pd.to_numeric(base.exact_net_bps, errors="raise").to_numpy(float),
            atol=2e-3, rtol=0.0,
        ):
            raise ValueError(f"{side}: base economics do not apply the 100bps cost exactly once")
        design = pd.concat([side_raw, handoff], axis=1)
        full_rows = len(base)
        candidate_count = max(1, int(np.ceil(float(args.base_candidate_fraction) * full_rows)))
        candidate_positions = stable_stage_i_topk_positions(
            handoff.base_raw_score.to_numpy(np.float32),
            candidate_ids=base.candidate_id,
            side_names=base.side_name,
            decision_timestamps=base.decision_ts,
            signal_timestamps=base["__ts__"], symbols=base["__symbol__"],
            count=candidate_count,
        )
        # Ranking selects membership only. Training rows retain their original
        # chronological order for every strict fold.
        candidate_positions = np.sort(candidate_positions.astype(np.int32))
        candidate_audit = base.loc[:, [*IDENTITY, "side_name", "decision_ts"]].copy()
        candidate_audit["base_raw_score"] = handoff.base_raw_score.to_numpy(np.float32)
        candidate_audit["selected_base_candidate"] = False
        candidate_audit.loc[candidate_positions, "selected_base_candidate"] = True
        candidate_audit["candidate_ranking_scope"] = "side_local_global_across_all_timestamps"
        candidate_audit["ranking_policy"] = RANKING_POLICY
        base, side_ledger, design, handoff = (
            item.iloc[candidate_positions].reset_index(drop=True)
            for item in (base, side_ledger, design, handoff)
        )
        required_regime = tuple(dict.fromkeys(map(str, args.required_regime_feature)))
        required_context = tuple(dict.fromkeys(map(str, args.required_context_feature)))
        trust_names = ("base_output_entropy", "base_output_top2_margin", "base_output_max_probability")
        required_handoff = ("base_raw_score", *state_names, *trust_names)
        if missing := set((*required_regime, *required_context, *required_handoff)).difference(design.columns):
            raise ValueError(f"{side}: direct meta required fields absent: {sorted(missing)}")
        contract_frame = side_ledger.loc[:, list(IDENTITY)].copy()
        contract_frame["side_name"] = side
        contract_frame["direct_fq3_exact_net_basis"] = pd.to_numeric(base.exact_net_bps, errors="raise").to_numpy(np.float32)
        contract_frame["gross_bps"] = pd.to_numeric(base.exact_gross_bps, errors="raise").to_numpy(np.float32)
        contract_frame["net_bps"] = pd.to_numeric(base.exact_net_bps, errors="raise").to_numpy(np.float32)
        contract_frame["target_valid"], contract_frame["sample_weight"] = True, 1.0
        meta_contract = bind_target_contract(
            contract_frame, family=FOLD_QUANTILE_RESIDUAL3, layer="meta", target_name="FQ3_direct_correctness",
            geometry=base_contract.geometry, target_columns=("direct_fq3_exact_net_basis",),
            metadata={
                "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
                "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
                "native_score_domain": list(domain),
                "required_regime_features": list(required_regime),
                "required_context_features": list(required_context),
                "required_trust_features": list(trust_names),
                "base_target_contract_sha256": base_contract.sha256,
                "feature_sidecar": sidecar_lineage,
                "mda_support_mode": args.mda_support_mode,
            },
        )
        decision = pd.to_datetime(base.decision_ts, utc=True, errors="raise")
        available = pd.to_datetime(base.label_available_ts, utc=True, errors="raise")
        signal_time = pd.to_datetime(base["__ts__"], utc=True, errors="raise")
        if not (decision.reset_index(drop=True) - signal_time.reset_index(drop=True)).eq(pd.Timedelta(hours=1)).all() or not (
            available.reset_index(drop=True) - decision.reset_index(drop=True)
        ).eq(pd.Timedelta(hours=12)).all():
            raise ValueError(f"{side}: direct-FQ3 timing must be signal close -> +1h decision -> +12h label")
        net = contract_frame.net_bps.to_numpy(np.float32)
        direct = handoff.base_raw_score.to_numpy(np.float32)
        root_destination = args.output_dir / side
        destination = root_destination
        resumed_clean_attempt = False
        attempt_lineage = _attempt_request_lineage(
            selector_manifest_sha=selector_manifest_sha,
            selector_feature_contract_sha=selector_feature_contract_sha,
            base_manifest_sha=base_manifest_sha,
            target_contract_sha=meta_contract.sha256,
            correlation_policy=args.correlation_policy,
            hpo_trials=args.hpo_trials, hpo_patience=args.hpo_patience,
            base_candidate_fraction=args.base_candidate_fraction,
            mda_support_mode=args.mda_support_mode,
        )
        if root_destination.exists():
            manifest_path = root_destination / "manifest.json"
            if not args.resume:
                raise FileExistsError(f"{side}: existing direct-FQ3 selector is not resumable")
            if manifest_path.is_file():
                prior = _read_json(manifest_path)
                if prior.get("status") == "complete":
                    if (
                        prior.get("selector_sample_manifest_sha256") != selector_manifest_sha
                        or prior.get("selector_feature_contract_sha256") != selector_feature_contract_sha
                        or prior.get("base_selector_manifest_sha256") != base_manifest_sha
                        or prior.get("correlation_policy") != args.correlation_policy
                        or int(prior.get("hpo_trials", -1)) != int(args.hpo_trials)
                        or int(prior.get("hpo_patience", -1)) != int(args.hpo_patience)
                        or float(prior.get("base_candidate_fraction", -1.0)) != float(args.base_candidate_fraction)
                        or prior.get("mda_support_mode", "full") != args.mda_support_mode
                    ):
                        raise ValueError(f"{side}: completed direct-FQ3 resume lineage drift")
                    for relative, expected in dict(prior.get("artifact_sha256", {})).items():
                        if file_sha256(destination / relative) != expected:
                            raise ValueError(f"{side}: resumed direct-FQ3 artifact hash drift: {relative}")
                    summaries.append({"side": side, "rows": prior["rows"], "selected_feature_count": prior["selected_feature_count"]})
                    continue
                # An incomplete root manifest is evidence of a killed process,
                # never a substitute for a complete selector contract.
                destination, completed_attempt = _prepare_clean_resume_attempt(
                    root_destination, side=side, lineage=attempt_lineage,
                )
                if completed_attempt:
                    prior = _read_json(destination / "manifest.json")
                    summaries.append({"side": side, "rows": prior["rows"], "selected_feature_count": prior["selected_feature_count"]})
                    continue
                resumed_clean_attempt = True
            # The bounded orchestrator creates the isolated side directory and
            # writes this one request receipt before spawning the child.  That
            # exact pristine state is not a partial selector and is safe to
            # consume.  A non-pristine root becomes an isolated, clean
            # attempt; no arbitrary lower-level checkpoint is adopted.
            elif {path.name for path in root_destination.iterdir()} == {"orchestrator_request.json"}:
                _validate_pristine_orchestrator_directory(root_destination, side=side)
            else:
                _bootstrap_direct_runner_receipt(
                    root_destination,
                    side=side,
                    lineage=attempt_lineage,
                )
                destination, completed_attempt = _prepare_clean_resume_attempt(
                    root_destination, side=side, lineage=attempt_lineage,
                )
                if completed_attempt:
                    prior = _read_json(destination / "manifest.json")
                    summaries.append({"side": side, "rows": prior["rows"], "selected_feature_count": prior["selected_feature_count"]})
                    continue
                resumed_clean_attempt = True
        else:
            root_destination.mkdir()
            _bootstrap_direct_runner_receipt(
                root_destination,
                side=side,
                lineage=attempt_lineage,
            )
        candidate_audit_path = destination / "base_candidate_handoff.parquet"
        candidate_audit.to_parquet(candidate_audit_path, index=False, compression="zstd")
        support = build_stage_i_mda_training_support(
            side_ledger.assign(
                exact_net_bps=net,
            ), side=side, identity_columns=IDENTITY, decision_timestamps=decision,
        )
        mda_label_context, mda_support_audit = restrict_stage_i_mda_training_support(
            support, mode=args.mda_support_mode,
        )
        try:
            result = run_stage_i_head_selection(
                design, net, contract=StageIHeadContract("meta", side, FOLD_QUANTILE_RESIDUAL3),
            cfg=CFG, report_root=destination / "mda",
            train_candidate=lgbm_pipeline.train_lgbm_stability_candidate,
                candidate_kwargs={
                "timestamps": decision, "label_available_timestamps": available,
                "exact_net_bps": net, "exact_net_units": "bps",
                "frozen_base_direct_score": direct,
                "frozen_base_direct_score_units": "native_score",
                "base_oof_provenance": {
                    "side": side, "strict_oof": True, "source": "completed_base_selector_native_oof",
                    "base_selector_manifest_sha256": base_manifest_sha,
                },
                "assets": base["__symbol__"].astype(str).to_numpy(),
                "candidate_ids": base.candidate_id.to_numpy(dtype=object),
                "label_context": mda_label_context,
                "sample_weight": np.ones(len(base), np.float32),
                # Direct semantics are hash-bound by the target contract; the
                # lower generic pipeline keeps its established meta enum.
                "mode": "regressor", "hpo_objective_mode": "train_meta",
                "mda_reference": {
                    "source": "full_finite_direct_base_oof_reference", "side": side,
                    "X": design, "target": net, "metric_target": net,
                    "sample_weight": np.ones(len(base), np.float32),
                    "timestamps": decision, "label_available_timestamps": available,
                    "exact_net_bps": net, "prediction_offset": direct,
                    "assets": base["__symbol__"].astype(str).to_numpy(),
                    "candidate_ids": base.candidate_id.to_numpy(dtype=object),
                    "identity": base.loc[:, [*IDENTITY, "decision_ts"]],
                    "archetype_labels": (
                        support["archetype_labels"]
                        if args.mda_support_mode == "full" else None
                    ),
                    "archetype_label_audit": support["audit"],
                },
                "stage_i_declared_single_side_scope": side,
                "reference_artifact_dir": destination / "reference",
                "cfg": {
                    # Use the shared Stage-I readiness gate.  It deliberately
                    # evaluates all rows after the declared leading warm-up,
                    # so a legitimate long-lookback feature is retained while
                    # fields missing throughout a material part of the panel
                    # cannot enter a frozen OOS contract.
                    "lgbm_feature_min_coverage": 0.90,
                    "lgbm_feature_coverage_scope": "all_post_warmup",
                    "lgbm_joint_complete_case_filter_enabled": False,
                    "stage_i_exact_readiness_coverage_prevalidated": False,
                    "stage_i_target_neutral_cache_root": str(cache_dir),
                    "stage_i_target_neutral_cache_request_sha256": cache_request_sha,
                    "stage_i_target_neutral_relief_cache_root": str(cache_dir / "relief_geometry"),
                    "mda_config": {
                        "pre_mda_bypass_features": [*required_regime, *required_context],
                        "force_include_features": [*required_regime, *required_context],
                        "archetype_conditioned_enabled": bool(args.mda_support_mode == "full"),
                        "redundancy_use_archetype_labels": bool(args.mda_support_mode == "full"),
                        "archetype_univariate_prescreen_enabled": bool(args.mda_support_mode == "full"),
                        "archetype_relief_prescreen_enabled": bool(args.mda_support_mode == "full"),
                        **(
                            {
                                "stage_i_bounded_univariate_checkpoint": {
                                    # Deliberately rooted outside a clean resume
                                    # attempt: each short invocation creates a
                                    # new attempt, while this checkpoint is bound
                                    # to the immutable parent and the literal
                                    # inner fit split before use.
                                    "root": str(root_destination / "_bounded_univariate"),
                                    "chunk_features": int(args.bounded_univariate_chunk_features),
                                    "input_sha256": _sha({
                                        "attempt_lineage": attempt_lineage,
                                        "side": side,
                                        "target_contract_sha256": meta_contract.sha256,
                                    }),
                                },
                            }
                            if args.bounded_univariate else {}
                        ),
                        **(
                            {
                                "stage_i_bounded_relief_checkpoint": {
                                    # Rescue runs after the hash-bound completed
                                    # univariate frontier. Its lower contract
                                    # binds the exact fit rows, archetype tasks,
                                    # feature matrix, target and cached geometry.
                                    # v2 stores feature-sliced geometry and
                                    # score evidence.  Never adopt or mutate
                                    # an older v1 repeat-only checkpoint.
                                    "root": str(_bounded_relief_checkpoint_root(
                                        root_destination,
                                        feature_chunk_size=args.bounded_relief_feature_chunk_size,
                                        anchor_chunk_size=args.bounded_relief_anchor_chunk_size,
                                    )),
                                    "feature_chunk_size": int(args.bounded_relief_feature_chunk_size),
                                    "anchor_chunk_size": int(args.bounded_relief_anchor_chunk_size),
                                    "input_sha256": _sha({
                                        "attempt_lineage": attempt_lineage,
                                        "side": side,
                                        "target_contract_sha256": meta_contract.sha256,
                                        "stage": "relief",
                                    }),
                                },
                            }
                            if args.bounded_relief else {}
                        ),
                        **(
                            {
                                # The stability grid is fitted before the
                                # dedicated MDA cohorts.  Keep its completed
                                # config/seed units across short bounded-MDA
                                # invocations; the lower cache contract binds
                                # the literal matrix, target, identities,
                                # timestamps, CV chronology and fit grid.
                                "stage_i_stability_grid_checkpoint": {
                                    "root": str(root_destination / "_stability_grid"),
                                    "input_sha256": _sha({
                                        "attempt_lineage": attempt_lineage,
                                        "side": side,
                                        "target_contract_sha256": meta_contract.sha256,
                                        "stage": "pre_mda_stability_grid",
                                    }),
                                },
                            }
                            if args.bounded_mda else {}
                        ),
                        **(
                            {
                                "stage_i_bounded_mda_checkpoint": {
                                    # Like the univariate state, this survives
                                    # clean isolated resume attempts while its
                                    # lower contract binds the exact realised
                                    # reference arrays, plan and parameters.
                                    "root": str(root_destination / "_bounded_mda"),
                                    "input_sha256": _sha({
                                        "attempt_lineage": attempt_lineage,
                                        "side": side,
                                        "target_contract_sha256": meta_contract.sha256,
                                        "stage": "dedicated_mda_first_pass",
                                    }),
                                },
                            }
                            if args.bounded_mda else {}
                        ),
                    },
                },
            },
                correlation_policy=args.correlation_policy, target_contract=meta_contract,
                required_base_handoff_features=required_handoff,
            )
        except lgbm_pipeline.StageIBoundedSelectionPending as pending:
            print(json.dumps({
                "status": f"bounded_{pending.stage}_pending", "side": side,
                "checkpoint_dir": pending.checkpoint_dir,
                "completed_chunks": pending.completed_chunks,
                "total_chunks": pending.total_chunks,
                "promotion": "forbidden_until_full_selection_and_hpo_complete",
            }, indent=2))
            return 75
        if result is None:
            raise RuntimeError(f"{side}: direct-FQ3 selector returned no result")
        selected = tuple(map(str, result["selected_feature_names"]))
        if missing := set((*required_regime, *required_context, *required_handoff)).difference(selected):
            raise ValueError(f"{side}: selected direct-FQ3 contract dropped required fields: {sorted(missing)}")
        hpo = run_stage_i_model_hpo(
            design, net, selected_feature_names=selected,
            candidate_ids=base.candidate_id, exact_net_bps=net,
            decision_timestamps=decision, label_available_timestamps=available,
            side=side, layer="meta", target_contract=meta_contract,
            prediction_offset_native_score=direct,
            hpo_trials=args.hpo_trials, hpo_patience=args.hpo_patience,
            # Resume attempts intentionally isolate partial selector evidence;
            # the independently hash-bound HPO rungs can safely survive them.
            successive_halving_checkpoint_dir=root_destination / "_hpo_halving",
        )
        probability = np.asarray(hpo.oof_probabilities, dtype=np.float32)
        correction = np.asarray(hpo.oof_score, dtype=np.float32)
        reconstructed = np.clip(direct + correction, domain[0], domain[1])
        output = base.loc[:, [*IDENTITY, "side_name", "decision_ts", "label_available_ts", "exact_gross_bps", "exact_net_bps"]].copy()
        output["base_raw_score"] = direct
        for name in state_names:
            output[name] = handoff[name].to_numpy(np.float32)
        output[["meta_p_overestimating", "meta_p_approximately_right", "meta_p_underestimating"]] = probability
        output[["meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"]] = probability
        output["meta_direct_correction"], output["meta_direct_score"] = correction, reconstructed
        oof_path = destination / "selector_meta_oof.parquet"
        output.to_parquet(oof_path, index=False, compression="zstd")
        request = {
            "schema": SCHEMA, "side": side, "selector_manifest_sha256": selector_manifest_sha,
            # Side orchestration validates this immutable MDA contract before
            # accepting a completed child on resume.  It is deliberately
            # distinct from this runner's own adapter schema above.
            "stage_i_selector_schema": STAGE_I_SELECTOR_SCHEMA,
            "selector_sample_manifest_sha256": selector_manifest_sha,
            "selector_feature_contract_sha256": selector_feature_contract_sha,
            "base_selector_manifest_sha256": base_manifest_sha,
            "base_target_contract_sha256": base_contract.sha256,
            "target_contract_sha256": meta_contract.sha256,
            "correlation_policy": args.correlation_policy,
            "base_candidate_fraction": float(args.base_candidate_fraction),
            "candidate_ranking_scope": "side_local_global_across_all_timestamps_never_per_timestamp",
            "dedicated_mda_reference_mode": args.dedicated_mda_reference,
            "hpo_trials": args.hpo_trials, "hpo_patience": args.hpo_patience,
            "feature_sidecar": sidecar_lineage,
            "mda_support_mode": args.mda_support_mode,
        }
        manifest = {
            **request, "request_sha256": _sha(request), "status": "complete", "rows": len(base),
            "selected_features": list(selected), "selected_feature_contract": list(selected),
            "selected_feature_count": len(selected), "best_params": _safe(hpo.best_params),
            # Exact selector-time input universe.  This is distinct from the
            # physical source parquet schema and lets later OOS materializers
            # prove every retained raw field came from the proper meta keys.
            "input_feature_contract": list(map(str, result.get("stage_i_input_features", ()))),
            "feature_sidecar": sidecar_lineage,
            "required_same_side_base_oof_handoff_features": list(required_handoff),
            "target_contract": meta_contract.to_dict(), "base_target_contract": base_contract.to_dict(),
            "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
            "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
            "meta_probability_semantics": (
                "neutral_error_terciles_0_1_2; legacy overestimating/approximately_right/"
                "underestimating aliases are literal only when q33<0<=q67"
            ),
            "selector_parent_calibration": result.get(
                "stage_i_direct_fq3_selector_parent_state", {}
            ),
            "native_score_domain": list(domain),
            "terminal_economics": "joint_reconstructed_meta_only_after_causal_common_bps_mapping",
            "meta_training_stream": (
                "all_finite_base_oof_rows_per_row" if float(args.base_candidate_fraction) == 1.0
                else "predeclared_side_local_global_base_score_tail"
            ),
            "mda_economic_metric": "top10_global_within_declared_side_stream_never_per_timestamp",
            "target_neutral_cache_provenance": {
                "root": str(cache_dir), "request_sha256": cache_request_sha,
            },
            # These labels guide only training-time archetype-aware MDA.  They
            # are deliberately not model inputs; retaining the audit makes
            # that separation and the resolved-path support reproducible.
            "mda_training_only_archetype_support": _safe(support["audit"]),
            "mda_support_audit": _safe(mda_support_audit),
            "timestamp_contract": {
                "schema": "stage_i_signal_decision_label_timing_v1",
                "signal_identity_column": "__ts__", "decision_column": "decision_ts",
                "label_available_column": "label_available_ts",
                "signal_to_decision_hours": 1, "decision_to_label_available_hours": 12,
                "rows": len(base),
                "strict_offsets_verified": bool(
                    (decision.reset_index(drop=True) - pd.to_datetime(base["__ts__"], utc=True).reset_index(drop=True)).eq(pd.Timedelta(hours=1)).all()
                    and (available.reset_index(drop=True) - decision.reset_index(drop=True)).eq(pd.Timedelta(hours=12)).all()
                ),
            },
            "hpo_actual_trials": hpo.actual_trials, "hpo_completed_trials": hpo.completed_trials,
            "hpo_best_metrics": _safe(hpo.best_metrics), "hpo_fold_audit": _safe(hpo.fold_audit),
            "hpo_oof_fold_audit": _safe(hpo.oof_fold_audit),
            "full_finite_base_oof_rows": full_rows,
            "generated_handoff_coverage_audit": {
                "pre_finite_filter_rows": pre_finite_filter_rows,
                "post_finite_filter_rows": int(full_rows),
                "finite_rows_by_feature_before_filter": generated_handoff_finite_counts,
                "finite_fraction_by_feature_before_filter": {
                    name: float(count / max(pre_finite_filter_rows, 1))
                    for name, count in generated_handoff_finite_counts.items()
                },
                "finite_fraction_by_feature_in_selector": {
                    str(column): float(
                        np.isfinite(pd.to_numeric(design[column], errors="coerce")).mean()
                    )
                    for column in required_handoff
                },
                "selector_exact_coverage_audit_sha256": integrity.get(
                    "exact_coverage_audit_sha256"
                ),
            },
            "feature_selection_reuse_exception": {
                "approved": True,
                "scope": "selected_feature_list_only",
                "selection_reference_start_utc": str(decision.min()),
                "selection_reference_end_utc": str(decision.max()),
                "selected_feature_contract_sha256": _sha(list(selected)),
                "rationale": (
                    "User-approved exception: select features once on the full available reference "
                    "and reuse the frozen list backward; HPO, causal mapping, admission and final "
                    "economics remain outside this exception."
                ),
            },
            "candidate_ranking_policy": RANKING_POLICY,
            "final_ranking_scope": "pooled_global_only_after_causal_common_bps_mapping",
            "artifact_sha256": {
                oof_path.name: file_sha256(oof_path),
                candidate_audit_path.name: file_sha256(candidate_audit_path),
            },
        }
        for key in (
            "stage_i_selector_schema", "stage_i_pruning_contract", "stage_i_pruning_contract_sha256",
            "stage_i_timestamp_contract", "stage_i_feature_universe_lineage",
            "stage_i_input_feature_count", "stage_i_input_features",
            "iterative_mda_feature_transition_ledger_csv",
            "iterative_mda_feature_transition_ledger_json",
            "iterative_mda_feature_transition_steps",
        ):
            if key in result:
                manifest[key] = _safe(result[key])
        (destination / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n")
        if resumed_clean_attempt:
            _publish_completed_resume_attempt(
                root_destination, side=side, attempt=destination, lineage=attempt_lineage,
            )
        summaries.append({"side": side, "rows": len(base), "selected_feature_count": len(selected)})
    print(json.dumps({"status": "complete", "cells": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
