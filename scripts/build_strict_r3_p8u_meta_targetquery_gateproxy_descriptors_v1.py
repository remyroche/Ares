#!/usr/bin/env python3
"""Build GateProxy descriptors from completed P8u Meta screening receipts.

This adapter keeps the cheap target/query stage and the learned downstream
GateProxy cleanly separated.  It opens completed, target-free candidate and
frozen-control score receipts first, proves identity/Base-rank equality, and
only then joins held policy/path outcomes to compute descriptors.  It neither
fits nor opens MC1 maps, admission, portfolio, live, or exchange state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_strict_r3_p8u_meta_downstream_proxy_descriptors_v1 as descriptor  # noqa: E402
import reblend_strict_r3_p8u_meta_hpo_authority_v1 as authority  # noqa: E402
import run_strict_r3_p8u_meta_target_query_grid_v1 as screen  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_targetquery_gateproxy_descriptors_v1"
IDENTITY = screen.IDENTITY
HELD_MONTHS = tuple(screen._utc_month(value) for value in (
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
))


def _once(path: Path, payload: object) -> None:
    descriptor_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor_fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, **payload: object) -> None:
    """Append fold-level observability without changing selection semantics."""
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*.parquet")) if path.is_dir() else (path,):
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _screen_arms(root: Path) -> pd.DataFrame:
    """Return target-free score descriptors from either supported screen shape.

    ``target_query_grid`` stores one model per named target/query arm.  The
    later cross-model screen instead stores one selected target/query arm and
    several model-family trials beneath it.  GateProxy needs to compare the
    latter trials independently, while retaining the *shared* true target
    family for its diversity proposal.  This adapter makes that distinction
    explicit: ``arm`` is the unique score-directory/trial name; ``family``,
    ``scale``, and ``query`` remain the frozen common target contract.
    """
    manifest = _read_json(root / "run_manifest.json")
    correctness = _read_json(root / "correctness_report.json")
    if not all(value is True for value in correctness.values() if isinstance(value, bool)):
        raise AssertionError(f"{root}: incomplete screen correctness receipt")
    schema = str(manifest.get("schema", ""))
    source_feature_contract = manifest.get("meta_feature_contract") or manifest.get("feature_contract")
    if not isinstance(source_feature_contract, str) or not source_feature_contract:
        raise AssertionError(f"{root}: screen manifest lacks exact feature-contract provenance")
    if schema == "strict_r3_p8u_meta_target_query_grid_v1":
        summary_path = root / "target_query_summary.parquet"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        table = pd.read_parquet(summary_path, columns=["arm", "family", "scale", "query"])
        listed = {str(item["name"]) for item in manifest.get("arms", ())}
        if set(table.arm.astype(str)) != listed or table.arm.duplicated().any():
            raise AssertionError(f"{root}: target/query arm summary mismatch")
    elif schema == "strict_r3_p8u_meta_crossmodel_v1":
        summary_path = root / "cross_model_summary.parquet"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        common = manifest.get("arm")
        if not isinstance(common, dict):
            raise AssertionError(f"{root}: cross-model manifest lacks common arm")
        required = {"trial", "model_family", "arm"}
        available = set(pd.read_parquet(summary_path).columns)
        if not required.issubset(available):
            raise AssertionError(f"{root}: cross-model summary lacks {sorted(required - available)}")
        summary = pd.read_parquet(summary_path, columns=sorted(required))
        if summary.empty or summary.trial.isna().any() or summary.trial.astype(str).duplicated().any():
            raise AssertionError(f"{root}: invalid cross-model trial summary")
        if not summary.arm.astype(str).eq(str(common.get("name"))).all():
            raise AssertionError(f"{root}: cross-model target/query arm mismatch")
        table = pd.DataFrame({
            "arm": summary.trial.astype(str),
            "family": str(common.get("family")),
            "scale": str(common.get("scale")),
            "query": str(common.get("query")),
            "model_family": summary.model_family.astype(str),
        })
        if table[["family", "scale", "query"]].isna().any().any():
            raise AssertionError(f"{root}: incomplete common target/query contract")
    else:
        raise AssertionError(f"{root}: unexpected screen schema {schema!r}")
    table["score_root"] = str(root)
    table["source_feature_contract"] = source_feature_contract
    return table


def _candidate(path: Path) -> pd.DataFrame:
    screen._assert_target_free(path)
    required = [*IDENTITY, "base_score", "base_rank_ts", "meta_raw_score", "meta_rank_ts", "target_free"]
    score = pd.read_parquet(path, columns=required)
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
    if score.duplicated(IDENTITY).any() or not score.side_name.eq("long").all() or not score.target_free.fillna(False).astype(bool).all():
        raise AssertionError(f"{path}: invalid target-free candidate receipt")
    return score.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _trial(row: pd.Series) -> dict[str, Any]:
    arm = str(row.arm)
    return {
        "name": arm,
        "target": f"{row.family}_{row.scale}",
        # The shared descriptor helper derives its grouping family from the
        # token before ``__``.  Preserve the true target family there instead
        # of making every geometry look like a distinct family to GateProxy's
        # diversity-control proposal.
        "arm_name": f"{row.family}__{arm}",
        "parent_contract": "P8U_ROUTED_F72_UNDERF120_RESEARCH_CANONICAL_20260828",
        "additive_feature_family": "f120_shap_kalman_transition_innovation_synergy",
        "feature_mode": "all_237_preselection",
        "sample_weight": {"profile": "uniform"},
        "model": {"objective": "rank_xendcg"},
        "gain": "common_screen_ordinal",
        "truncation": None,
        "sigmoid": None,
    }


def run(
    *, config: Path, canonical_root: Path, screen_roots: Sequence[Path], out: Path,
    bootstrap_iterations: int, source_override: Path | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    raw, applied_source_override = screen._apply_source_override(
        _read_json(config), source_override,
    )
    source = raw["source"]
    tables = [_screen_arms(root) for root in screen_roots]
    arms = pd.concat(tables, ignore_index=True)
    if arms.arm.duplicated().any() or len(arms) < 4:
        raise AssertionError("target/query descriptor input has duplicate or inadequate arms")
    control_root = canonical_root.resolve()
    policy_path = (ROOT / str(source["policy_labels"])).resolve()
    path_root = (ROOT / str(source["path_labels"])).resolve()
    base_root = (ROOT / str(source["base_target_free_root"])).resolve()
    policy = screen._read_policy(policy_path)
    anchors = {
        month: authority._held_anchor(raw=raw, base_root=base_root, policy=policy, path_root=path_root, held_month=month)
        for month in HELD_MONTHS
    }
    out.mkdir(parents=True)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF GateProxy descriptor materialization only; no MC1, admission, portfolio, live, or exchange input/mutation",
        "config": str(config), "config_sha256": _sha(config),
        "target_query_roots": [str(root) for root in screen_roots],
        "canonical_control": str(control_root),
        "feature_contract": str(ROOT / str(source["base_f72_contract"])),
        # The successor target/query screen intentionally uses only frozen
        # F72.  Earlier staged configurations declared a wider F120/F237
        # count, whereas the current concise contract need not repeat it.
        "feature_count": int(raw.get("meta_feature_count", 72)),
        "held_months": [f"{month:%Y-%m}" for month in HELD_MONTHS],
        "source": source,
        "source_override": str(source_override) if source_override else None,
        "source_override_sha256": _sha(source_override) if source_override else None,
        "source_override_payload": applied_source_override,
        "causality": "candidate/control score receipts are validated target-free before held outcomes open; anchors use only earlier resolved policy labels",
    })
    labels: dict[pd.Timestamp, pd.DataFrame] = {}
    fold_rows: list[dict[str, Any]] = []
    weekly_rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    _progress(out, event="started", arms=int(len(arms)), held_months=int(len(HELD_MONTHS)))
    for item in arms.sort_values(["family", "arm"], kind="stable").itertuples(index=False):
        root, trial = Path(item.score_root), _trial(pd.Series(item._asdict()))
        for month in HELD_MONTHS:
            candidate_path = root / "target_free_scores" / str(item.arm) / f"month={month:%Y-%m}.parquet"
            candidate = _candidate(candidate_path)
            control = descriptor._read_control_score(control_root / "target_free_scores" / "current" / f"month={month:%Y-%m}.parquet")
            descriptor._validate_target_free_pair(candidate, control, trial=str(item.arm), month=month)
            if month not in labels:
                labels[month] = screen._labelled(control, policy, path_root, month, screen._month_end(month))
            row, weekly = descriptor._metric_row(
                candidate=candidate, control=control, labelled=labels[month], anchor=anchors[month], trial=trial,
                root_name=root.name,
                # Keep GateProxy's descriptor provenance tied to the exact
                # selected F120/F72 source contract, rather than a generic
                # stage name.  The MC1-plan materializer rechecks this exact
                # value against the immutable score manifest.
                feature_contract=str(item.source_feature_contract),
                feature_count=int(raw.get("meta_feature_count", 72)),
                held_month=month,
            )
            fold_rows.append(row); weekly_rows.append(weekly)
            audits.append({
                "score_root": root.name, "trial": str(item.arm), "held_month": f"{month:%Y-%m}",
                "candidate_target_free_validated_before_outcome_join": True,
                "control_target_free_validated_before_outcome_join": True,
                "candidate_control_identity_exact": True,
                "candidate_base_rank_matches_control": True,
                "held_anchor_uses_only_prior_resolved_labels": True,
                "held_outcomes_are_descriptor_only": True,
            })
            _progress(
                out, event="fold_complete", arm=str(item.arm),
                held_month=f"{month:%Y-%m}", rows=int(len(candidate)),
            )
    fold = pd.DataFrame(fold_rows)
    weekly = pd.concat(weekly_rows, ignore_index=True)
    summary = descriptor._bootstrap_summary(fold, iterations=bootstrap_iterations, seed=1729)
    summary = descriptor._attach_cross_fold_stability(summary, fold, weekly)
    fold.to_parquet(out / "trial_fold_descriptors.parquet", index=False, compression="zstd")
    weekly.to_parquet(out / "trial_weekly_descriptors.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "trial_descriptor_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "correctness_audit.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "completed_target_query_inputs_are_receipted_strict_oof": True,
        "candidate_and_control_scores_are_target_free_before_outcome_join": True,
        "candidate_control_identity_and_base_rank_are_exact": True,
        "held_anchor_is_prior_resolved_only": True,
        "held_outcomes_are_used_only_for_post_score_descriptors": True,
        "no_mc1_admission_portfolio_live_or_exchange_input_opened": True,
        "descriptor_outputs_have_no_selection_or_promotion_authority": True,
    })
    _progress(out, event="completed", trials=int(len(summary)), fold_rows=int(len(fold)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--source-override", type=Path, help="immutable source-only binding receipt")
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    print(run(
        config=args.config.resolve(), canonical_root=args.canonical_root.resolve(),
        screen_roots=tuple(path.resolve() for path in args.screen_root), out=args.out.resolve(),
        bootstrap_iterations=args.bootstrap_iterations,
        source_override=args.source_override.resolve() if args.source_override else None,
    ))


if __name__ == "__main__":
    main()
