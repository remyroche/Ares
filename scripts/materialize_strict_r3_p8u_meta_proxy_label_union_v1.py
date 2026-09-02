#!/usr/bin/env python3
"""Recompute one strict learned-proxy label ledger across completed MC1 roots.

Each source is a completed ``downstream_labels_v1`` artifact.  The sources
share the frozen control/policy/MC1 contract but may cover different Meta
feature contracts.  This utility never reruns score production or MC1.  It
concatenates only their already-audited primitive weekly/trial labels and
recomputes the robust normalisation plus block-bootstrap reliability once over
the combined population.  That avoids comparing separately scaled targets in
leave-feature-contract-out validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import build_strict_r3_p8u_meta_proxy_downstream_labels_v1 as labels_parent  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_proxy_downstream_label_union_v1"
REQUIRED_FLAGS = {
    "all_candidate_mc1_roots_have_target_free_before_policy_join_receipts",
    "all_candidate_mc1_roots_have_six_complete_month_prequential_training",
    "candidate_and_control_identity_and_policy_are_exact",
    "bcf_score_and_mc1_map_are_identical_for_matched_meta_replacements",
    "priority_uses_matched_real_dual_admission_budget",
    "gate_uses_real_dual_50bps_contract",
    "gate_bootstrap_uses_same_per_timestamp_scale_as_aggregate_label",
    "portfolio_is_confirmation_not_the_only_label",
    "no_live_or_exchange_mutation",
}
PRIOR21_RECEIPT_ALIASES = (
    "all_candidate_mc1_roots_use_prior21_shift_causal",
    "all_candidate_mc1_roots_use_prior21_resolved_shift",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    correctness = json.loads((root / "correctness_report.json").read_text())
    if manifest.get("schema") != labels_parent.SCHEMA:
        raise AssertionError(f"{root}: not a strict downstream label artifact")
    missing = sorted(key for key in REQUIRED_FLAGS if correctness.get(key) is not True)
    # The pre-existing parent ledger and the append ledger use two explicit
    # names for the same invariant: every prior-21-day shift is computed from
    # resolved labels only.  Require at least one affirmative receipt rather
    # than weakening this check or rewriting an immutable parent artifact.
    if not any(correctness.get(key) is True for key in PRIOR21_RECEIPT_ALIASES):
        missing.append("one_of(" + ",".join(PRIOR21_RECEIPT_ALIASES) + ")")
    if missing:
        raise AssertionError(f"{root}: downstream label causality receipt failed {missing}")
    trial = pd.read_parquet(root / "downstream_trial_labels.parquet")
    weekly = pd.read_parquet(root / "downstream_weekly_labels.parquet")
    monthly = pd.read_parquet(root / "downstream_monthly_labels.parquet")
    audit = pd.read_parquet(root / "correctness_audit.parquet")
    if trial.trial.duplicated().any() or weekly.duplicated(["trial", "era", "week"]).any() or monthly.duplicated(["trial", "era"]).any():
        raise AssertionError(f"{root}: duplicate downstream label identity")
    return trial, weekly, monthly, audit, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    roots = tuple(path.resolve() for path in args.label_root)
    if len(roots) < 2 or len(roots) != len(set(roots)):
        raise ValueError("at least two unique --label-root values are required")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    parts = [_load(root) for root in roots]
    raw_trial = pd.concat([part[0] for part in parts], ignore_index=True)
    weekly = pd.concat([part[1] for part in parts], ignore_index=True)
    audits = pd.concat([part[3].assign(source_label_root=str(root)) for root, part in zip(roots, parts)], ignore_index=True)
    if raw_trial.trial.duplicated().any() or weekly.duplicated(["trial", "era", "week"]).any():
        raise AssertionError("trial identity overlaps across label roots")
    # Recalculate every scaled/shrunk outcome from the common primitive
    # components, including reliability weights, under one fixed weekly block
    # bootstrap.  Existing source-level d* columns never survive as targets.
    combined, normalisation = labels_parent._bootstrap_labels(
        raw_trial,
        weekly,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
    )
    monthly = labels_parent._monthly_from_weekly(weekly)
    out.mkdir(parents=True)
    combined.to_parquet(out / "downstream_trial_labels.parquet", index=False, compression="zstd")
    weekly.to_parquet(out / "downstream_weekly_labels.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "downstream_monthly_labels.parquet", index=False, compression="zstd")
    audits.to_parquet(out / "correctness_audit.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline downstream-label union for learned-proxy portability validation; no score/MC1/admission/portfolio/live/exchange mutation",
        "source_label_roots": [str(root) for root in roots],
        "source_manifests_sha256": {str(root): _sha256(root / "run_manifest.json") for root in roots},
        "trials": int(len(combined)), "weekly_rows": int(len(weekly)), "monthly_rows": int(len(monthly)),
        "normalisation": normalisation,
        "bootstrap": {"unit": "week", "iterations": int(args.bootstrap_iterations), "seed": int(args.bootstrap_seed)},
        "normalisation_scope": "recomputed jointly across every source trial before proxy fitting",
        "selection_authority": "none; this is a label-scale reconciliation only",
    })
    _once(out / "correctness_report.json", {
        "all_source_label_roots_passed_strict_mc1_causality": True,
        "trial_identities_are_disjoint_across_sources": True,
        "primitive_weekly_labels_are_preserved": True,
        "normalisation_and_reliability_recomputed_jointly": True,
        "portfolio_is_not_the_sole_proxy_training_target": True,
        "no_score_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })


if __name__ == "__main__":
    main()
