#!/usr/bin/env python3
"""Materialize evidence gates for reconstructed historical exact-policy labels.

This is intentionally a readiness report rather than an economics report.  It
keeps only the newly reconstructed deployed-policy label lineage eligible and
records why a month is unavailable or restricted before downstream policy work
can select on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_historical_execution_ev_policy_inputs import (  # noqa: E402
    load_archived_path_inputs,
    load_historical_candidates,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_FEBAPR_LABEL_MANIFEST = ROOT / (
    "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/"
    "manifest.json"
)
DEFAULT_DEC_STAGE_MANIFEST = ROOT / (
    "data_perp/artifacts/dec2025_execution_ev_current_spread_12h_stage_20260727_v1/"
    "manifest.json"
)
DEFAULT_PARITY_GATE = ROOT / (
    "data_perp/artifacts/deployed_policy_label_parity_20260727_v1/evidence_gate.json"
)
DEFAULT_ACTIVE_OOF = ROOT / (
    "data_perp/artifacts/regime_transition_active_head_20260726_v1/grouped_oof.parquet"
)
DEFAULT_HISTORICAL_OOF = ROOT / (
    "data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_20260727_v2/"
    "two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/historical_exact_policy_readiness_20260727_v1"
DEFAULT_PATH_INPUTS = (
    ROOT / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels/train_global_long_3.parquet",
    ROOT / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels/train_global_short_3.parquet",
)
DEFAULT_LABELS_ROOT = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"


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
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def period_gate(
    *,
    period: str,
    candidate_rows: int,
    canonical_path_rows: int,
    exact_1m_rows: int,
    minimum_exact_coverage: float,
    parity_pass: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return an explicit acceptance decision without hiding attrition."""

    canonical_coverage = canonical_path_rows / max(candidate_rows, 1)
    exact_coverage = exact_1m_rows / max(canonical_path_rows, 1)
    exact_over_all = exact_1m_rows / max(candidate_rows, 1)
    accepted = bool(
        parity_pass
        and canonical_path_rows > 0
        and exact_coverage >= float(minimum_exact_coverage)
    )
    blockers: list[str] = []
    if not parity_pass:
        blockers.append("current_overlap_deployed_policy_parity_failed")
    if canonical_path_rows == 0:
        blockers.append("no_joinable_canonical_path_inputs")
    elif exact_coverage < float(minimum_exact_coverage):
        blockers.append("insufficient_exact_1m_coverage_of_canonical_candidates")
    if reason:
        blockers.append(reason)
    return {
        "period": str(period),
        "candidate_rows": int(candidate_rows),
        "canonical_path_rows": int(canonical_path_rows),
        "canonical_path_coverage": float(canonical_coverage),
        "exact_1m_rows": int(exact_1m_rows),
        "exact_1m_coverage_of_canonical_candidates": float(exact_coverage),
        "exact_1m_coverage_of_original_candidates": float(exact_over_all),
        "minimum_exact_coverage": float(minimum_exact_coverage),
        "new_exact_policy_labels_accepted": accepted,
        "blockers": blockers,
    }


def _jan_canonical_input_audit(labels_root: Path, path_inputs: Sequence[Path]) -> dict[str, Any]:
    candidates, _ = load_historical_candidates(
        labels_root, start_month="2025-01", end_month="2025-01"
    )
    archived = load_archived_path_inputs(
        path_inputs, start_month="2025-01", end_month="2025-01"
    )
    joined = candidates.loc[:, list(IDENTITY)].merge(
        archived.loc[:, list(IDENTITY)], on=list(IDENTITY), how="inner", validate="one_to_one"
    )
    return {
        "candidate_rows": int(len(candidates)),
        "canonical_path_rows": int(len(joined)),
    }


def _active_overlap(
    rows: pd.DataFrame,
    active: pd.DataFrame,
    *,
    period: str,
    complete_column: str | None = None,
) -> list[dict[str, Any]]:
    work = rows.copy()
    timestamp = "__ts__" if "__ts__" in work.columns else "source_utc"
    work[timestamp] = pd.to_datetime(work[timestamp], utc=True, errors="raise")
    if complete_column is None:
        work["__is_exact__"] = True
    else:
        work["__is_exact__"] = work[complete_column].fillna(False).astype(bool)
    joined = work.merge(
        active,
        left_on=timestamp,
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    active_rows = joined.loc[joined["target__transition_active"].eq(1)].copy()
    all_events = active_rows["target__event_id"].dropna().astype(str).nunique()
    exact = active_rows.loc[active_rows["__is_exact__"]]
    return [
        {
            "period": period,
            "scope": "all_candidates",
            "active_candidate_rows": int(len(active_rows)),
            "active_hours": int(active_rows[timestamp].nunique()),
            "active_events": int(all_events),
        },
        {
            "period": period,
            "scope": "exact_1m_complete",
            "active_candidate_rows": int(len(exact)),
            "active_hours": int(exact[timestamp].nunique()),
            "active_events": int(exact["target__event_id"].dropna().astype(str).nunique()),
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    parity = json.loads(args.parity_gate.read_text(encoding="utf-8"))
    parity_pass = bool(parity.get("comparison", {}).get("parity_pass"))
    febapr = json.loads(args.febapr_label_manifest.read_text(encoding="utf-8"))
    dec = json.loads(args.dec_stage_manifest.read_text(encoding="utf-8"))
    # The raw policy-label artifact preserves the exchange's slash symbol
    # notation.  The strict OOF reconstruction normalizes symbols before its
    # identity join and writes a matching exact-label ledger beside the score.
    # That derived ledger is the only historical economics source paired with
    # a score stream; do not cross a raw slash-symbol ledger with it.
    labels_path = args.historical_oof.parent / "exact_1m_execution_ev_12h_labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"strict historical OOF exact label ledger is unavailable: {labels_path}")
    labels = pd.read_parquet(labels_path, columns=list(IDENTITY))
    active = pd.read_parquet(
        args.active_oof,
        columns=["source_utc", "target__event_id", "target__transition_active"],
    )
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True, errors="raise")
    jan = _jan_canonical_input_audit(args.labels_root, args.path_inputs)
    febapr_source = febapr["source"]["exact_join"]
    febapr_coverage = febapr["coverage"]["overall"]
    dec_source = dec["source"]["exact_join"]
    dec_coverage = dec["coverage"]["overall"]
    readiness = [
        period_gate(
            period="2025-01",
            candidate_rows=jan["candidate_rows"],
            canonical_path_rows=jan["canonical_path_rows"],
            exact_1m_rows=0,
            minimum_exact_coverage=args.minimum_exact_coverage,
            parity_pass=parity_pass,
        ),
        period_gate(
            period="2025-02..2025-04",
            candidate_rows=int(febapr_source["input_candidate_rows"]),
            canonical_path_rows=int(febapr_source["admitted_rows"]),
            exact_1m_rows=int(febapr_coverage["complete"]),
            minimum_exact_coverage=args.minimum_exact_coverage,
            parity_pass=parity_pass,
            reason="frozen_canonical_path_subset_only; do_not_compare_to_unrestricted_candidate_book",
        ),
        period_gate(
            period="2025-12",
            candidate_rows=int(dec_source["input_candidate_rows"]),
            canonical_path_rows=int(dec_source["admitted_rows"]),
            exact_1m_rows=int(dec_coverage["complete"]),
            minimum_exact_coverage=args.minimum_exact_coverage,
            parity_pass=parity_pass,
        ),
    ]
    stage_audit = pd.read_csv(ROOT / str(dec["coverage_csv"]["path"]))
    transition_rows = [
        *_active_overlap(labels, active, period="2025-02..2025-04"),
        *_active_overlap(stage_audit, active, period="2025-12", complete_column="complete"),
    ]
    historical_oof = pd.read_parquet(args.historical_oof, columns=list(IDENTITY))
    oof_overlap = labels.merge(historical_oof, on=list(IDENTITY), how="inner", validate="one_to_one")
    readiness_table = pd.DataFrame(readiness)
    transition_table = pd.DataFrame(transition_rows)
    args.output_dir.mkdir(parents=True)
    readiness_path = args.output_dir / "period_readiness.csv"
    transition_path = args.output_dir / "transition_event_overlap.csv"
    gate_path = args.output_dir / "evidence_gate.json"
    readiness_table.to_csv(readiness_path, index=False)
    transition_table.to_csv(transition_path, index=False)
    gate = {
        "schema": "historical_exact_policy_readiness_gate_v1",
        "invalidated_lineage": "exact_history_state_recurrence_20260727_v1 is excluded from all economics",
        "parity_prerequisite": {
            "path": str(args.parity_gate),
            "sha256": _sha256(args.parity_gate),
            "passed": parity_pass,
        },
        "deployed_contract": {
            "simulator": febapr["accounting"]["simulator"],
            "policy_sha256": febapr["source"]["policy_sha256"],
            "spread_baseline_sha256": febapr["accounting"]["spread_baseline_sha256"],
            "geometry": "side_x_policy_archetype contract; current observable candidates resolve 100% to side-parent fallback",
            "horizon_minutes": int(febapr["exit_policy_contract"]["horizon_minutes"]),
        },
        "periods": readiness,
        "transition_overlap": transition_rows,
        "historical_score_evidence": {
            "febapr_local_two_layer_oof_rows": int(len(historical_oof)),
            "exact_label_identity_overlap": int(len(oof_overlap)),
            "interpretation": "historical side-local OOF score only; not a parity claim with the current production ranker",
        },
        "allowed_economic_use": {
            "febapr_exact_policy_candidate_local": bool(readiness[1]["new_exact_policy_labels_accepted"]),
            "jan_exact_policy": False,
            "dec_exact_policy_pooled_global": False,
            "reason": "December exact-1m coverage is below the minimum; January has no canonical path-input join",
        },
        "promotion_or_overlay_ready": False,
        "remaining_blockers": [
            "historical labels are a current-spread counterfactual, not historical realized spread evidence",
            "active-transition head evidence is grouped OOF rather than chronological policy OOS",
            "Feb-Apr exact labels retain a frozen canonical path-input subset; any score/policy comparison must use the same identities",
            "No current production causal global-score history exists on the accepted Feb-Apr exact-label identities",
        ],
        "artifacts": {
            "period_readiness": str(readiness_path),
            "transition_event_overlap": str(transition_path),
            "febapr_score_paired_exact_labels": str(labels_path),
        },
    }
    _write_json(gate_path, gate)
    return {"gate": gate_path, "periods": readiness_path, "transitions": transition_path}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--febapr-label-manifest", type=Path, default=DEFAULT_FEBAPR_LABEL_MANIFEST)
    result.add_argument("--dec-stage-manifest", type=Path, default=DEFAULT_DEC_STAGE_MANIFEST)
    result.add_argument("--parity-gate", type=Path, default=DEFAULT_PARITY_GATE)
    result.add_argument("--active-oof", type=Path, default=DEFAULT_ACTIVE_OOF)
    result.add_argument("--historical-oof", type=Path, default=DEFAULT_HISTORICAL_OOF)
    result.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    result.add_argument("--path-inputs", type=Path, nargs="+", default=list(DEFAULT_PATH_INPUTS))
    result.add_argument("--minimum-exact-coverage", type=float, default=0.70)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    if not 0.0 < options.minimum_exact_coverage <= 1.0:
        raise ValueError("minimum exact coverage must lie in (0, 1]")
    outputs = run(options)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
