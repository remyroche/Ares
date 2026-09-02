#!/usr/bin/env python3
"""Create full strict-MC1 ground-truth labels for selected Meta trials.

This is intentionally downstream of the descriptor sampler.  It joins an
earlier target-free continuation to the existing target-free OOF Meta scores,
then rebuilds the fixed dual BCF/Current MC1 maps and constrained portfolio for
each selected trial.  It does not rank candidates or alter any live contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import materialize_strict_r3_p8u_meta_score_union_v1 as union_parent  # noqa: E402
import run_strict_r3_p8u_dual_mc1_prequential_v2 as mc1_parent  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_proxy_selected_mc1_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*")) if path.is_dir() else [path]
    for member in members:
        if not member.is_file():
            continue
        digest.update(str(member.relative_to(path) if path.is_dir() else member.name).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in raw.split(",") if item.strip())
    expected = tuple(pd.date_range(months[0], months[-1], freq="MS", tz="UTC")) if months else ()
    if len(months) <= mc1_parent.TRAIN_MONTHS or months != expected:
        raise ValueError("months must be a complete chronological sequence longer than the frozen MC1 window")
    return months


def _parse_prehistory(raw: list[str]) -> dict[str, tuple[Path, ...]]:
    result: dict[str, list[Path]] = {}
    for value in raw:
        pieces = value.split("::", 1)
        if len(pieces) != 2:
            raise ValueError("--prehistory-root must be ARM::ROOT")
        arm, root = str(pieces[0]), Path(pieces[1]).resolve()
        if not arm:
            raise ValueError("prehistory arm keys must be non-empty")
        roots = result.setdefault(arm, [])
        if root in roots:
            raise ValueError(f"duplicate prehistory root for arm {arm!r}")
        roots.append(root)
    return {arm: tuple(roots) for arm, roots in result.items()}


def _materialize_union(*, trial: str, arm: str, historical: tuple[Path, ...], existing: Path, months: tuple[pd.Timestamp, ...], out: Path) -> dict[str, Any]:
    target = out / "target_free_scores" / trial
    target.mkdir(parents=True)
    audit: list[dict[str, Any]] = []
    for month in months:
        source = union_parent._source((*historical, existing), trial, month)
        frame = union_parent._read(source, trial, month)
        destination = target / f"month={month:%Y-%m}.parquet"
        frame.to_parquet(destination, index=False, compression="zstd")
        audit.append({
            "month": f"{month:%Y-%m}",
            "rows": int(len(frame)),
            "timestamps": int(frame.__decision_ts__.nunique()),
            "source": str(source),
            "source_sha256": union_parent._sha([source]),
            "target_free": True,
        })
    _once(out / "run_manifest.json", {
        "schema": union_parent.SCHEMA,
        "scope": "offline selected-Meta target-free score union; no labels, MC1, admission, portfolio, live, or exchange mutation",
        "trial": trial,
        "arm": arm,
        "months": [f"{month:%Y-%m}" for month in months],
        "sources": [*(str(root) for root in historical), str(existing)],
        "audit": audit,
        "correctness": {
            "single_source_per_month": True,
            "target_free": True,
            "long_unique_identity": True,
            "earlier_continuation_and_existing_oof_scores_are_disjoint_by_month": True,
        },
    })
    pd.DataFrame(audit).to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    return {"trial": trial, "arm": arm, "union_root": str(out), "union_sha256": _sha(out), "months": len(audit)}


def _assert_base_alignment(*, union_root: Path, base_root: Path, trial: str, months: tuple[pd.Timestamp, ...]) -> dict[str, Any]:
    """Prove every joined Meta receipt shares the frozen Base coordinate."""
    max_delta = 0.0
    rows = 0
    for month in months:
        meta = pd.read_parquet(
            union_root / "target_free_scores" / trial / f"month={month:%Y-%m}.parquet",
            columns=["candidate_id", "__decision_ts__", "side_name", "base_rank_ts"],
        )
        base = pd.read_parquet(
            base_root / f"month={month:%Y-%m}.parquet",
            columns=["candidate_id", "__decision_ts__", "side_name", "base_rank_ts"],
        )
        for frame in (meta, base):
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        joined = base.merge(meta, on=["candidate_id", "__decision_ts__", "side_name"], how="left", suffixes=("_base", "_meta"), validate="one_to_one")
        if len(joined) != len(base) or joined.base_rank_ts_meta.isna().any():
            raise AssertionError(f"{trial} {month:%Y-%m}: Meta/Base identity mismatch")
        delta = float((joined.base_rank_ts_base - joined.base_rank_ts_meta).abs().max())
        if delta > 1e-6:
            raise AssertionError(f"{trial} {month:%Y-%m}: Meta/Base rank mismatch {delta}")
        max_delta = max(max_delta, delta)
        rows += len(joined)
    return {"base_identity_exact": True, "base_rank_max_abs_delta": max_delta, "base_alignment_rows": rows}


def _run_one(
    *, record: dict[str, Any], prehistory: dict[str, tuple[Path, ...]], base_root: Path, policy: Path,
    months: tuple[pd.Timestamp, ...], threshold_bps: float, out: Path,
) -> dict[str, Any]:
    trial = str(record["trial"])
    trial_config = dict(record["trial_config"])
    arm = str(trial_config["arm_name"])
    if str(trial_config["name"]) != trial:
        raise AssertionError(f"{trial}: trial configuration identity mismatch")
    # A selected score root sometimes already contains the entire requested
    # six-month MC1 history.  In that case no synthetic or unrelated
    # "prehistory" root may be supplied merely to satisfy a CLI guard.
    # ``_source`` below still proves that each month has exactly one source.
    historical = prehistory.get(arm, ())
    existing = Path(str(record["source_score_root"])).resolve()
    union_root = out / "target_free_score_unions" / trial
    union_receipt = _materialize_union(
        trial=trial, arm=arm, historical=historical, existing=existing, months=months, out=union_root,
    )
    base_alignment = _assert_base_alignment(
        union_root=union_root, base_root=base_root, trial=trial, months=months,
    )
    mc1_out = out / "candidate_mc1" / trial
    mc1_parent.run(
        base_root=base_root,
        metas=((union_root, trial, 1.0),),
        policy_path=policy,
        months=months,
        out=mc1_out,
        threshold_bps=threshold_bps,
    )
    correctness = json.loads((mc1_out / "correctness_report.json").read_text())
    if not all(bool(value) for value in correctness.values()):
        raise AssertionError(f"{trial}: candidate MC1 correctness failure")
    metrics = pd.read_parquet(mc1_out / "portfolio_metrics.parquet").iloc[0].to_dict()
    return {**union_receipt, **base_alignment, "mc1_root": str(mc1_out), "mc1_sha256": _sha(mc1_out), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--prehistory-root", action="append", default=[], help="ARM::ROOT; repeat")
    parser.add_argument(
        "--score-root-override",
        type=Path,
        help=("Use one explicitly supplied target-free score root for every retained "
              "trial. Intended only for a matched full-history rescoring of a selected trial."),
    )
    parser.add_argument(
        "--only-candidate",
        action="append",
        default=[],
        help="Retain only this selected trial; repeatable. The filtered plan is still receipt-bound.",
    )
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    if args.threshold_bps <= 0:
        raise ValueError("threshold must be positive")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    selection = args.selection_root.resolve()
    plan_path = selection / "selected_trial_plan.json"
    records = json.loads(plan_path.read_text())
    if not isinstance(records, list) or not records:
        raise AssertionError("selected-trial plan must be non-empty")
    names = [str(record.get("trial")) for record in records]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate selected trial")
    requested = tuple(dict.fromkeys(str(name) for name in args.only_candidate))
    if requested:
        unknown = sorted(set(requested).difference(names))
        if unknown:
            raise ValueError(f"--only-candidate is absent from selected plan: {unknown}")
        records = [record for record in records if str(record["trial"]) in requested]
        if not records:
            raise AssertionError("candidate filter removed every selected trial")
    score_root_override = args.score_root_override.resolve() if args.score_root_override else None
    if score_root_override:
        if not score_root_override.is_dir():
            raise FileNotFoundError(score_root_override)
        records = [{**record, "source_score_root": str(score_root_override)} for record in records]
    prehistory = _parse_prehistory(list(args.prehistory_root))
    months = _months(args.months)
    base_root, policy = args.base_root.resolve(), args.policy.resolve()
    out.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            _run_one, record=record, prehistory=prehistory, base_root=base_root, policy=policy,
            months=months, threshold_bps=float(args.threshold_bps), out=out,
        ) for record in records]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results = sorted(results, key=lambda item: str(item["trial"]))
    pd.DataFrame(results).to_parquet(out / "candidate_mc1_summary.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "all_trial_score_unions_target_free_and_month_disjoint": True,
        "all_trial_meta_scores_match_the_frozen_base_coordinate": True,
        "all_candidate_mc1_replays_passed_strict_correctness": True,
        "base_and_policy_contracts_fixed_across_trials": True,
        "no_trial_ranking_or_promotion_authority": True,
        "no_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline selected Meta -> fixed dual MC1 -> constrained portfolio ground-truth labelling; no HPO selection, promotion, live, or exchange mutation",
        "selection_root": str(selection),
        "selection_plan_sha256": _sha(plan_path),
        "score_root_override": str(score_root_override) if score_root_override else None,
        "score_root_override_sha256": _sha(score_root_override) if score_root_override else None,
        "only_candidates": list(requested),
        "prehistory_roots": {arm: [str(path) for path in paths] for arm, paths in sorted(prehistory.items())},
        "base_root": str(base_root),
        "policy": str(policy),
        "months": [f"{month:%Y-%m}" for month in months],
        "threshold_bps": float(args.threshold_bps),
        "workers": int(args.workers),
        "selected_trials": len(records),
        "output": "candidate_mc1_summary.parquet",
        "selection_authority": "none; these are noisy downstream labels used only to fit/falsify the later PriorityProxy and GateProxy",
    })
    print(out)


if __name__ == "__main__":
    main()
