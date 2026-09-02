#!/usr/bin/env python3
"""Strict P8u Meta-combination -> independent dual-MC1 -> portfolio replay.

Inputs are frozen P8u Base and Meta *target-free* score receipts.  The script
persists a Base-only BCF coordinate and one declared Base+Meta current
coordinate before policy labels are joined.  It then fits independent,
strict-prequential MC1 maps and applies their shared dual gate to one global
chronological portfolio replay.  This is research only and has no live or
exchange-writing path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_mc1_combination_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
POLICY_FORBIDDEN = frozenset({
    "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_exit_bar_15m",
    "policy_exit_price", "policy_entry_price", "policy_label_available_ts", "policy_exit_reason",
})


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        for member in sorted(path.rglob("*.parquet")) if path.is_dir() else (path,):
            digest.update(str(member).encode())
            with member.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in raw.split(",") if value.strip())
    if len(values) < 4 or tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError("need at least four chronological unique months")
    return values


def _base_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _meta_path(root: Path, arm: str, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _rank_desc(frame: pd.DataFrame, column: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", column]].copy()
    work["row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(work), dtype=np.float32)
    result[work.row.to_numpy(np.int64)] = (1.0 - (ordinal - .5) / count).astype(np.float32)
    return result


def _parse_meta(raw: Sequence[str], *, base_only: bool) -> tuple[tuple[Path, str, float], ...]:
    """Parse ROOT::ARM::WEIGHT declarations without an implicit score owner."""
    items: list[tuple[Path, str, float]] = []
    if base_only:
        if raw:
            raise ValueError("--base-only cannot be combined with --meta")
        return tuple()
    for value in raw:
        parts = value.split("::")
        if len(parts) not in {2, 3}:
            raise ValueError("--meta must be ROOT::ARM or ROOT::ARM::WEIGHT")
        root, arm = Path(parts[0]).resolve(), str(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else 1.0
        if weight <= 0.0:
            raise ValueError("Meta blend weights must be positive")
        items.append((root, arm, weight))
    if not items:
        raise ValueError("at least one --meta receipt is required")
    names = [(str(root), arm) for root, arm, _ in items]
    if len(names) != len(set(names)):
        raise ValueError("duplicate Meta receipt")
    return tuple(items)


def _load_month(
    base_root: Path, metas: Sequence[tuple[Path, str, float]], month: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_path = _base_path(base_root, month)
    base_names = set(pq.ParquetFile(base_path).schema_arrow.names)
    if POLICY_FORBIDDEN.intersection(base_names):
        raise AssertionError(f"{base_path}: Base score receipt leaks outcomes")
    base = pd.read_parquet(base_path, columns=[*IDENTITY, "base_rank_ts"])
    base["__decision_ts__"] = pd.to_datetime(base["__decision_ts__"], utc=True, errors="raise")
    if base.duplicated(list(IDENTITY)).any() or not base.side_name.eq("long").all():
        raise AssertionError(f"{month:%Y-%m}: invalid target-free Base identity")
    ranks: list[np.ndarray] = []
    names: list[str] = []
    weights: list[float] = []
    for root, arm, weight in metas:
        path = _meta_path(root, arm, month)
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        if POLICY_FORBIDDEN.intersection(schema):
            raise AssertionError(f"{path}: Meta score receipt leaks outcomes")
        meta = pd.read_parquet(path, columns=[*IDENTITY, "meta_rank_ts"])
        meta["__decision_ts__"] = pd.to_datetime(meta["__decision_ts__"], utc=True, errors="raise")
        if meta.duplicated(list(IDENTITY)).any():
            raise AssertionError(f"{month:%Y-%m} {arm}: duplicate Meta identity")
        merged = base.loc[:, list(IDENTITY)].merge(meta, on=list(IDENTITY), how="left", validate="one_to_one")
        if len(merged) != len(base) or merged.meta_rank_ts.isna().any():
            raise AssertionError(f"{month:%Y-%m} {arm}: Meta receipt does not exactly cover Base identity")
        ranks.append(merged.meta_rank_ts.to_numpy(np.float32)); names.append(arm); weights.append(weight)
    if not np.isfinite(base.base_rank_ts.to_numpy(float)).all() or not all(np.isfinite(rank).all() for rank in ranks):
        raise AssertionError(f"{month:%Y-%m}: non-finite target-free rank")
    if ranks:
        matrix = np.column_stack(ranks)
        blend = np.average(matrix, axis=1, weights=np.asarray(weights, dtype=float)).astype(np.float32)
        # A blend of timestamp ranks is not itself necessarily a timestamp rank.
        # Rerank it only with frozen candidate identities and score values.
    else:
        # Matched Base-only control: preserve the same frozen candidate identity,
        # Base score, MC1 schedule, admission, and portfolio path.  It deliberately
        # supplies no Meta coordinate rather than a near-zero-weight pseudo head.
        matrix = np.empty((len(base), 0), dtype=np.float32)
        blend = base.base_rank_ts.to_numpy(np.float32)
    aggregate = base.loc[:, list(IDENTITY)].copy()
    aggregate["meta_weighted_rank"] = blend
    aggregate["meta_rank_ts"] = _rank_desc(aggregate, "meta_weighted_rank")
    agreement = (
        (1.0 - np.abs(matrix - base.base_rank_ts.to_numpy(float)[:, None])).clip(0.0, 1.0).mean(axis=1)
        if ranks else np.full(len(base), .5, dtype=np.float32)
    )
    current = aggregate.loc[:, list(IDENTITY)].copy()
    current["enhanced_base_routed"] = True
    current["base_rank42"] = base.base_rank_ts.to_numpy(np.float32)
    current["conditional_consensus_rank"] = aggregate.meta_rank_ts.to_numpy(np.float32)
    current["ordinary_shadow_consensus_rank"] = base.base_rank_ts.to_numpy(np.float32)
    current["correctness_rank"] = agreement.astype(np.float32)
    current["upstream"] = (.75 * current.base_rank42 + .25 * current.conditional_consensus_rank).astype(np.float32)
    current["final_score"] = current.upstream.to_numpy(np.float32)
    bcf = current.copy()
    bcf["conditional_consensus_rank"] = bcf.base_rank42.to_numpy(np.float32)
    bcf["ordinary_shadow_consensus_rank"] = bcf.base_rank42.to_numpy(np.float32)
    bcf["correctness_rank"] = np.float32(.5)
    bcf["upstream"] = bcf.base_rank42.to_numpy(np.float32)
    bcf["final_score"] = bcf.base_rank42.to_numpy(np.float32)
    return current, {
        "month": f"{month:%Y-%m}", "rows": len(current), "meta_arms": names,
        "meta_weights": weights, "target_free_identity_exact": True,
        "current_family": "0.75*Base_rank + 0.25*weighted_Meta_rank" if ranks else "Base_rank_only_control",
        "bcf_family": "Base_rank",
    }


def _target_free_panels(
    *, base_root: Path, metas: Sequence[tuple[Path, str, float]], months: Sequence[pd.Timestamp], out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    currents: list[pd.DataFrame] = []
    bcfs: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in months:
        current, audit = _load_month(base_root, metas, month)
        bcf = current.copy()
        bcf["conditional_consensus_rank"] = bcf.base_rank42.to_numpy(np.float32)
        bcf["ordinary_shadow_consensus_rank"] = bcf.base_rank42.to_numpy(np.float32)
        bcf["correctness_rank"] = np.float32(.5)
        bcf["upstream"] = bcf.base_rank42.to_numpy(np.float32)
        bcf["final_score"] = bcf.base_rank42.to_numpy(np.float32)
        for family, frame in (("current", current), ("bcf", bcf)):
            path = out / "target_free_scores" / family / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, compression="zstd")
        currents.append(current); bcfs.append(bcf); audits.append(audit)
    return pd.concat(currents, ignore_index=True), pd.concat(bcfs, ignore_index=True), audits


def _policy(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path)
    required = {
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    }
    missing = sorted(required.difference(policy.columns))
    if missing or policy.candidate_id.duplicated().any():
        raise AssertionError(f"invalid canonical policy source: missing={missing}")
    policy["policy_label_available_ts"] = pd.to_datetime(policy.policy_label_available_ts, utc=True, errors="raise")
    return policy


def _join_policy(scores: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    result = scores.merge(policy.drop(columns=["__decision_ts__", "side_name"], errors="ignore"), on="candidate_id", how="left", validate="one_to_one")
    if len(result) != len(scores) or not result.candidate_id.equals(scores.candidate_id):
        raise AssertionError("policy join changed persisted target-free score identities")
    return result


def run(
    *, base_root: Path, metas: Sequence[tuple[Path, str, float]], policy_path: Path, months: tuple[pd.Timestamp, ...],
    out: Path, threshold_bps: float,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    if threshold_bps <= 0.0:
        raise ValueError("threshold must be positive")
    out.mkdir(parents=True)
    current_scores, bcf_scores, score_audit = _target_free_panels(base_root=base_root, metas=metas, months=months, out=out)
    _once(out / "target_free_score_audit.json", {
        "schema": SCHEMA, "months": [f"{month:%Y-%m}" for month in months],
        "base_root": str(base_root), "metas": [{"root": str(root), "arm": arm, "weight": weight} for root, arm, weight in metas],
        "score_audit": score_audit, "prohibited_outcome_columns_absent": True,
        "policy_join_occurs_only_after_target_free_scores_persisted": True,
    })
    policy = _policy(policy_path)
    current, bcf = _join_policy(current_scores, policy), _join_policy(bcf_scores, policy)
    old_months, old_train_months, old_threshold = parent.SCORE_MONTHS, parent.MC1_TRAIN_MONTHS, parent.MC1_THRESHOLD_BPS
    try:
        parent.SCORE_MONTHS = months
        parent.MC1_TRAIN_MONTHS = 3
        parent.MC1_THRESHOLD_BPS = float(threshold_bps)
        current_pred, current_audit = parent._mc1_predictions(current, "current", out)
        bcf_pred, bcf_audit = parent._mc1_predictions(bcf, "bcf", out)
        combined = parent._combined_challenger(current_pred, bcf_pred)
        evaluation_start = months[3]
        combined = combined.loc[combined.__decision_ts__.ge(evaluation_start)].copy()
        metrics = parent._portfolio_metrics(combined, "p8u_meta_combo", f"{evaluation_start:%Y%m}_{months[-1]:%Y%m}", out)
    finally:
        parent.SCORE_MONTHS, parent.MC1_TRAIN_MONTHS, parent.MC1_THRESHOLD_BPS = old_months, old_train_months, old_threshold
    current_audit.to_parquet(out / "current_mc1_fit_audit.parquet", index=False, compression="zstd")
    bcf_audit.to_parquet(out / "bcf_mc1_fit_audit.parquet", index=False, compression="zstd")
    combined.to_parquet(out / "dual_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame([metrics]).to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    correctness = {
        "p8u_target_free_base_and_meta_identity_exact": True,
        "all_target_free_scores_persisted_before_policy_join": True,
        "mc1_maps_separate_and_strict_prequential": True,
        "mc1_labels_resolved_before_held_month": True,
        "dual_admission_and_shared_chronological_portfolio": True,
        "no_live_or_exchange_mutation": True,
    }
    if metas:
        correctness["current_bcf_coordinates_distinct"] = True
    else:
        correctness["base_only_control_current_bcf_identical_by_design"] = True
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u Meta combination / independent dual-MC1 / constrained portfolio research only",
        "months": [f"{month:%Y-%m}" for month in months], "evaluation_start": f"{evaluation_start:%Y-%m}",
        "base_root": str(base_root), "metas": [{"root": str(root), "arm": arm, "weight": weight} for root, arm, weight in metas],
        "policy": str(policy_path), "threshold_bps": threshold_bps,
        "score_families": {"current": "0.75*Base rank + 0.25*weighted Meta rank" if metas else "Base rank only control", "bcf": "Base rank only"},
        "mc1": {"train_months": 3, "features": list(parent.MC1_FEATURES), "portfolio_priority": "bcf_mc1_expected_bps"},
        "metrics": metrics,
        "source_sha256": _sha([base_root, *(root for root, _arm, _weight in metas), policy_path]),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--meta", action="append", default=[], help="ROOT::ARM[::WEIGHT]; repeat for a declared Meta blend")
    parser.add_argument("--base-only", action="store_true", help="Matched Base-only control; do not pass --meta")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--months", default="2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        base_root=args.base_root.resolve(), metas=_parse_meta(args.meta, base_only=args.base_only), policy_path=args.policy.resolve(),
        months=_months(args.months), threshold_bps=args.threshold_bps, out=args.out.resolve(),
    ))


if __name__ == "__main__":
    main()
