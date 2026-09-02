#!/usr/bin/env python3
"""Strict target-free Base-only versus Base-plus-Meta dual-MC1 diagnostic.

This is deliberately narrower than the historical production score families.
It exists to test one Base contract without pretending that a duplicated MC1
coordinate is a dual admission system:

    BCF family     = frozen Base rank alone
    Current family = 75% frozen Base rank + 25% strict-OOF Meta rank

Both families are persisted target-free before the canonical policy labels are
joined.  They then receive independent strict-prequential MC1 fits and pass a
common dual-MC1 threshold into one chronological portfolio replay.  The tool
is research-only and cannot mutate a live bundle or make exchange calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_p8u_singlebase_true_dual_mc1_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
POLICY_FORBIDDEN = {
    "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_exit_bar_15m", "policy_exit_price", "policy_entry_price",
    "policy_label_available_ts",
}


def _write_once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if path.is_dir():
            members = sorted(path.rglob("*.parquet"))
        else:
            members = [path]
        for member in members:
            digest.update(str(member).encode("utf-8"))
            with member.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(
        pd.Timestamp(f"{token.strip()}-01", tz="UTC")
        for token in raw.split(",")
        if token.strip()
    )
    if len(values) < 4 or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("--months must contain at least four unique chronological YYYY-MM values")
    return values


def _base_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}" / "scores_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing Base target-free source: {path}")
    return path


def _meta_path(root: Path, arm: str, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing Meta target-free source: {path}")
    return path


def _load_month(base_root: Path, meta_root: Path, arm: str, month: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(
        _base_path(base_root, month),
        columns=[*IDENTITY, "base_rank_ts", "enhanced_base_routed"],
    )
    meta = pd.read_parquet(
        _meta_path(meta_root, arm, month),
        columns=[*IDENTITY, "meta_rank_ts"],
    )
    if POLICY_FORBIDDEN.intersection(base.columns) or POLICY_FORBIDDEN.intersection(meta.columns):
        raise AssertionError(f"{month:%Y-%m}: policy/outcome column found in target-free score input")
    for frame, name in ((base, "Base"), (meta, "Meta")):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame.duplicated(list(IDENTITY)).any():
            raise AssertionError(f"{month:%Y-%m}: {name} target-free identity is not unique")
    merged = base.merge(meta, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(meta):
        raise AssertionError(f"{month:%Y-%m}: Base/Meta target-free identities do not match exactly")
    if not merged.enhanced_base_routed.fillna(False).astype(bool).all():
        raise AssertionError(f"{month:%Y-%m}: non-Router50 row reached single-Base dual diagnostic")
    rank_columns = ["base_rank_ts", "meta_rank_ts"]
    if not np.isfinite(merged.loc[:, rank_columns].to_numpy(float)).all():
        raise AssertionError(f"{month:%Y-%m}: non-finite target-free Base/Meta rank")
    if (merged.loc[:, rank_columns].to_numpy(float) < 0.0).any() or (merged.loc[:, rank_columns].to_numpy(float) > 1.0).any():
        raise AssertionError(f"{month:%Y-%m}: Base/Meta rank outside [0,1]")
    agreement = (1.0 - (merged.base_rank_ts - merged.meta_rank_ts).abs()).clip(0.0, 1.0)
    current = merged.loc[:, list(IDENTITY)].copy()
    current["enhanced_base_routed"] = True
    current["base_rank42"] = merged.base_rank_ts.to_numpy(np.float32)
    current["conditional_consensus_rank"] = merged.meta_rank_ts.to_numpy(np.float32)
    current["ordinary_shadow_consensus_rank"] = merged.base_rank_ts.to_numpy(np.float32)
    current["correctness_rank"] = agreement.to_numpy(np.float32)
    current["upstream"] = (
        .75 * merged.base_rank_ts.to_numpy(float) + .25 * merged.meta_rank_ts.to_numpy(float)
    ).astype(np.float32)
    current["final_score"] = current.upstream.to_numpy(np.float32)
    bcf = current.copy()
    bcf["conditional_consensus_rank"] = bcf.base_rank42.to_numpy(np.float32)
    bcf["ordinary_shadow_consensus_rank"] = bcf.base_rank42.to_numpy(np.float32)
    bcf["correctness_rank"] = np.float32(.5)
    bcf["upstream"] = bcf.base_rank42.to_numpy(np.float32)
    bcf["final_score"] = bcf.base_rank42.to_numpy(np.float32)
    if np.allclose(current.final_score.to_numpy(float), bcf.final_score.to_numpy(float), rtol=0.0, atol=1e-10):
        raise AssertionError(f"{month:%Y-%m}: current and BCF target-free coordinates unexpectedly match")
    return current, bcf


def _target_free_panels(
    *, base_root: Path, meta_root: Path, arm: str, months: tuple[pd.Timestamp, ...], out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    current_parts: list[pd.DataFrame] = []
    bcf_parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for month in months:
        current, bcf = _load_month(base_root, meta_root, arm, month)
        for family, frame in (("current", current), ("bcf", bcf)):
            target = out / "target_free_scores" / family / f"month={month:%Y-%m}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target, index=False, compression="zstd")
        current_parts.append(current)
        bcf_parts.append(bcf)
        audit.append({
            "month": f"{month:%Y-%m}", "rows": int(len(current)),
            "target_free_identity_exact": True, "router50_only": True,
            "families_have_distinct_scores": True,
            "current_score": "0.75*base_rank + 0.25*meta_rank",
            "bcf_score": "base_rank",
        })
    return pd.concat(current_parts, ignore_index=True), pd.concat(bcf_parts, ignore_index=True), audit


def _policy(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"canonical policy label source missing {missing}")
    if frame.duplicated("candidate_id").any():
        raise AssertionError("canonical policy label source has duplicate candidate IDs")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    return frame


def _join_policy(scores: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    frame = scores.merge(
        policy.drop(columns=["__decision_ts__", "side_name"], errors="ignore"),
        on="candidate_id", how="left", validate="one_to_one",
    )
    if len(frame) != len(scores) or not frame.candidate_id.equals(scores.candidate_id):
        raise AssertionError("policy join changed target-free score identity/order")
    return frame


def run(
    *, base_root: Path, meta_root: Path, meta_arm: str, policy_path: Path,
    months: tuple[pd.Timestamp, ...], out: Path, threshold_bps: float,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if threshold_bps <= 0.0:
        raise ValueError("threshold must be positive")
    out.mkdir(parents=True)
    current_scores, bcf_scores, score_audit = _target_free_panels(
        base_root=base_root, meta_root=meta_root, arm=meta_arm, months=months, out=out,
    )
    _write_once(out / "target_free_score_audit.json", {
        "schema": SCHEMA, "months": [f"{month:%Y-%m}" for month in months],
        "base_root": str(base_root), "meta_root": str(meta_root), "meta_arm": meta_arm,
        "score_audit": score_audit,
        "prohibited_outcome_columns_absent": True,
        "policy_join_occurs_only_after_target_free_scores_persisted": True,
    })
    policy = _policy(policy_path)
    current = _join_policy(current_scores, policy)
    bcf = _join_policy(bcf_scores, policy)
    original_months = parent.SCORE_MONTHS
    original_train_months = parent.MC1_TRAIN_MONTHS
    original_threshold = parent.MC1_THRESHOLD_BPS
    try:
        parent.SCORE_MONTHS = months
        parent.MC1_TRAIN_MONTHS = 3
        parent.MC1_THRESHOLD_BPS = float(threshold_bps)
        current_predictions, current_audit = parent._mc1_predictions(current, "current", out)
        bcf_predictions, bcf_audit = parent._mc1_predictions(bcf, "bcf", out)
        combined = parent._combined_challenger(current_predictions, bcf_predictions)
        evaluation_start = months[3]
        combined = combined.loc[combined["__decision_ts__"].ge(evaluation_start)].copy()
        metrics = parent._portfolio_metrics(combined, "singlebase_true_dual", f"{evaluation_start:%Y%m}_{months[-1]:%Y%m}", out)
    finally:
        parent.SCORE_MONTHS = original_months
        parent.MC1_TRAIN_MONTHS = original_train_months
        parent.MC1_THRESHOLD_BPS = original_threshold
    current_audit.to_parquet(out / "current_mc1_fit_audit.parquet", index=False, compression="zstd")
    bcf_audit.to_parquet(out / "bcf_mc1_fit_audit.parquet", index=False, compression="zstd")
    combined.to_parquet(out / "dual_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame([metrics]).to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    _write_once(out / "correctness_report.json", {
        "target_free_base_meta_scores_persisted_before_policy_join": True,
        "base_meta_target_free_identity_exact": True,
        "router50_only": True,
        "current_uses_meta_rank": True,
        "bcf_is_base_only": True,
        "current_bcf_score_coordinates_are_distinct": True,
        "independent_current_bcf_mc1_maps": True,
        "mc1_training_labels_resolved_before_held_month": True,
        "daily_mc1_shift_uses_prior_resolved_labels_only": True,
        "dual_admission_threshold_applied": True,
        "shared_chronological_portfolio_state": True,
        "no_live_or_exchange_mutation": True,
    })
    _write_once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict Base-only versus Base-plus-Meta independent dual-MC1 diagnostic; no live/exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months],
        "evaluation_start": f"{evaluation_start:%Y-%m}",
        "base_root": str(base_root), "meta_root": str(meta_root), "meta_arm": meta_arm,
        "policy_path": str(policy_path),
        "families": {
            "current": "0.75*Base timestamp rank + 0.25*strict-OOF unexpected-trailing Meta timestamp rank",
            "bcf": "Base timestamp rank only",
        },
        "mc1": {"train_months": 3, "dual_admission_threshold_bps": threshold_bps, "priority": "bcf_mc1_expected_bps"},
        "portfolio_metrics": metrics,
        "source_sha256": _sha256((base_root, meta_root, policy_path)),
        "next_stage": "diagnostic only; compare matched Base contracts before any canonical/live change",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--meta-arm", default="under_atr1__timestamp")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--months", default="2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        base_root=args.base_root.resolve(), meta_root=args.meta_root.resolve(), meta_arm=args.meta_arm,
        policy_path=args.policy.resolve(), months=_months(args.months), threshold_bps=args.threshold_bps,
        out=args.out.resolve(),
    ))


if __name__ == "__main__":
    main()
