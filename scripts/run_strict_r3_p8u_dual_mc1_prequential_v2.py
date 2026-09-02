#!/usr/bin/env python3
"""Rebuild the P8U dual MC1 maps as six-month inference packages.

This is the successor to ``run_strict_r3_p8u_meta_mc1_combination_v1.py``.
The predecessor temporarily changed the shared runner's six-month default to
three months and discarded each fitted HGB model.  This script makes the
six-month window explicit, persists the fitted BCF and Current maps for every
monthly vintage, and writes the exact prior-21-day shift state used for its
held prediction panel.

It is offline research/package construction only.  It has no exchange or live
trading imports or side effects.
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


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_strict_r3_enhanced_base_live_stack_challenger as portfolio_parent  # noqa: E402
import run_strict_r3_p8u_meta_mc1_combination_v1 as score_parent  # noqa: E402
from extreme_price_movements.inference.p8u_mc1_inference_package import (  # noqa: E402
    FEATURES,
    P8UMC1InferencePackage,
    apply_shift,
    build_shift_state,
    fit_package,
    load_package,
    save_package,
)


SCHEMA = "strict_r3_p8u_dual_mc1_prequential_v2"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
TRAIN_MONTHS = 6
MIN_FIT_ROWS = 5_000
# Calendar coverage belongs to the frozen upstream Base source, while a Meta
# candidate must cover *every* timestamp that Base actually produced.  Keeping
# those conditions separate prevents a universal observed Base-source gap from
# being misreported as candidate-specific score loss.  Missing Base hours are
# audited and never reconstructed.
MIN_BASE_SOURCE_HOURLY_COVERAGE = 0.975


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*")) if path.is_dir() else [path]
    for member in members:
        if not member.is_file():
            continue
        digest.update(str(member.relative_to(path) if path.is_dir() else member.name).encode("utf-8"))
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in raw.split(",") if value.strip())
    if len(values) <= TRAIN_MONTHS or tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError("need chronological unique months with at least one six-month held fold")
    expected = tuple(pd.date_range(values[0], values[-1], freq="MS", tz="UTC"))
    if values != expected:
        raise ValueError("months must be a complete consecutive calendar-month sequence")
    return values


def _policy_contract(policy: pd.DataFrame, policy_path: Path) -> dict[str, object]:
    costs = sorted(pd.to_numeric(policy.get("policy_cost_bps"), errors="coerce").dropna().unique().tolist())
    return {
        "source": str(policy_path),
        "source_sha256": _sha256(policy_path),
        "target": "canonical rich 15-minute policy net bps",
        "cost_application": "embedded exactly once in policy_net_bps",
        "observed_policy_cost_bps": costs,
        "label_availability": "policy_label_available_ts < held decision boundary",
    }


def _calendar_coverage(reference_decision_ts: pd.DatetimeIndex, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS", tz="UTC"):
        month_end = month + pd.offsets.MonthBegin(1)
        expected_hours = pd.date_range(month, month_end - pd.Timedelta(hours=1), freq="h", tz="UTC")
        observed = reference_decision_ts[(reference_decision_ts >= month) & (reference_decision_ts < month_end)]
        final_day = expected_hours[expected_hours.normalize() == (month_end - pd.Timedelta(days=1)).normalize()]
        rows.append({
            "month": f"{month:%Y-%m}",
            "expected_hours": int(len(expected_hours)),
            "base_source_hours": int(len(observed)),
            "base_source_calendar_coverage": float(len(expected_hours.intersection(observed)) / max(1, len(expected_hours))),
            "base_source_terminal_day_present": bool(len(final_day.intersection(observed))),
        })
    return pd.DataFrame(rows)


def _assert_full_train_coverage(
    frame: pd.DataFrame, *, reference_decision_ts: pd.DatetimeIndex, train_start: pd.Timestamp, held_start: pd.Timestamp,
) -> None:
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    months = tuple(pd.date_range(train_start, held_start - pd.offsets.MonthBegin(1), freq="MS", tz="UTC"))
    failures: list[str] = []
    for month in months:
        end = month + pd.offsets.MonthBegin(1)
        expected_hours = pd.date_range(month, end - pd.Timedelta(hours=1), freq="h", tz="UTC")
        base_hours = reference_decision_ts[(reference_decision_ts >= month) & (reference_decision_ts < end)]
        observed_hours = pd.DatetimeIndex(decision.loc[decision.ge(month) & decision.lt(end)].unique())
        base_coverage = len(expected_hours.intersection(base_hours)) / max(1, len(expected_hours))
        missing_from_candidate = base_hours.difference(observed_hours)
        unexpected_candidate_hours = observed_hours.difference(base_hours)
        # The last calendar day is a distinct guard against a source whose
        # date partition exists but ends several days early.  Do not require
        # every hour: isolated historical source gaps are preserved as missing
        # target-free rows and are not reconstructed from later information.
        final_day = expected_hours[expected_hours.normalize() == (end - pd.Timedelta(days=1)).normalize()]
        terminal_day_present = bool(len(final_day.intersection(base_hours)))
        if (
            base_coverage < MIN_BASE_SOURCE_HOURLY_COVERAGE
            or not terminal_day_present
            or len(missing_from_candidate)
            or len(unexpected_candidate_hours)
        ):
            failures.append(
                f"{month:%Y-%m}(base_coverage={base_coverage:.4%},terminal_day_present={terminal_day_present},"
                f"candidate_missing_base_hours={len(missing_from_candidate)},candidate_extra_hours={len(unexpected_candidate_hours)})"
            )
    if failures:
        raise AssertionError(
            "six-month P8U MC1 fit lacks sufficient frozen-Base source coverage or exact candidate/Base timestamp coverage: "
            + ", ".join(failures)
        )


def _strict_train(frame: pd.DataFrame, *, train_start: pd.Timestamp, held_start: pd.Timestamp) -> pd.DataFrame:
    valid = (
        frame["__decision_ts__"].ge(train_start)
        & frame["__decision_ts__"].lt(held_start)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & frame["policy_label_available_ts"].lt(held_start)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    )
    train = frame.loc[valid].copy()
    if train.empty or not train["policy_label_available_ts"].lt(held_start).all():
        raise AssertionError("P8U MC1 fit consumed a non-pre-resolved policy label")
    return train


def _fit_score_family(
    frame: pd.DataFrame,
    *,
    family: str,
    months: tuple[pd.Timestamp, ...],
    out: Path,
    source_hashes: dict[str, str],
    policy_contract: dict[str, object],
    base_decision_ts: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    frame = frame.copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    package_index: list[dict[str, object]] = []
    for held_start in months[TRAIN_MONTHS:]:
        held_end = held_start + pd.offsets.MonthBegin(1)
        train_start = held_start - pd.DateOffset(months=TRAIN_MONTHS)
        _assert_full_train_coverage(
            frame, reference_decision_ts=base_decision_ts, train_start=train_start, held_start=held_start,
        )
        train = _strict_train(frame, train_start=train_start, held_start=held_start)
        held = frame.loc[frame["__decision_ts__"].ge(held_start) & frame["__decision_ts__"].lt(held_end)].copy()
        if len(train) < MIN_FIT_ROWS or held.empty:
            raise AssertionError(
                f"{family} {held_start:%Y-%m}: strict six-month MC1 support insufficient "
                f"(train={len(train)}, held={len(held)})"
            )
        package = fit_package(
            train,
            family=family,
            train_start=train_start,
            train_end_exclusive=held_start,
            held_start=held_start,
            held_end_exclusive=held_end,
            train_months=TRAIN_MONTHS,
            source_hashes=source_hashes,
            policy_contract=policy_contract,
        )
        held["static_expected_bps"] = package.predict_static(held)
        held["score_band_curve_bps"] = package.curve_for(held)
        shift_state = build_shift_state(package, frame, held_start=held_start, held_end_exclusive=held_end)
        held["recent_shift_bps"] = apply_shift(held["static_expected_bps"], held["__decision_ts__"], shift_state) - held["static_expected_bps"]
        held["mc1_expected_bps"] = held["static_expected_bps"] + held["recent_shift_bps"]
        held["mc1_family"] = family
        package_path = out / "mc1_packages" / f"family={family}" / f"month={held_start:%Y-%m}"
        members = save_package(package, shift_state, package_path)
        reloaded = load_package(package_path)
        package_delta = float(np.max(np.abs(reloaded.predict_static(held) - held["static_expected_bps"].to_numpy(float))))
        if package_delta > 1e-12:
            raise AssertionError(f"{family} {held_start:%Y-%m}: serialized MC1 model score mismatch {package_delta}")
        package_hash = _sha256(package_path)
        outputs.append(held)
        audit.append({
            "family": family,
            "month": f"{held_start:%Y-%m}",
            "status": "scored",
            "train_start": train_start,
            "train_end_exclusive": held_start,
            "held_end_exclusive": held_end,
            "train_months": TRAIN_MONTHS,
            "train_rows": int(len(train)),
            "held_rows": int(len(held)),
            "clip_low": package.target_clip[0],
            "clip_high": package.target_clip[1],
            "package_path": str(package_path.relative_to(out)),
            "package_sha256": package_hash,
            "serialized_static_prediction_max_abs_delta": package_delta,
            "shift_state_rows": int(len(shift_state)),
            "shift_max_label_available_ts_lt_decision_day": bool(
                (
                    pd.to_datetime(shift_state["max_policy_label_available_ts"], utc=True, errors="coerce").lt(
                        pd.to_datetime(shift_state["decision_day"], utc=True)
                    )
                    | shift_state["max_policy_label_available_ts"].isna()
                ).all()
            ),
        })
        package_index.append({
            "family": family,
            "month": f"{held_start:%Y-%m}",
            "path": str(package_path.relative_to(out)),
            "sha256": package_hash,
            "members": members,
            "is_latest": held_start == months[-1],
        })
    if not outputs:
        raise AssertionError(f"{family}: no strict six-month MC1 packages were produced")
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit), package_index


def run(
    *,
    base_root: Path,
    metas: Sequence[tuple[Path, str, float]],
    policy_path: Path,
    months: tuple[pd.Timestamp, ...],
    out: Path,
    threshold_bps: float,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    if threshold_bps <= 0.0:
        raise ValueError("threshold must be positive")
    out.mkdir(parents=True, exist_ok=False)
    base_timestamp_parts: list[pd.Series] = []
    for month in months:
        panel = pd.read_parquet(base_root / f"month={month:%Y-%m}.parquet", columns=["__decision_ts__"])
        base_timestamp_parts.append(pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise"))
    base_decision_ts = pd.DatetimeIndex(pd.concat(base_timestamp_parts, ignore_index=True).unique())
    _calendar_coverage(
        base_decision_ts, start=months[0], end=months[-1] + pd.offsets.MonthBegin(1),
    ).to_parquet(out / "base_source_calendar_coverage.parquet", index=False, compression="zstd")
    current_scores, bcf_scores, score_audit = score_parent._target_free_panels(
        base_root=base_root, metas=metas, months=months, out=out,
    )
    _once(out / "target_free_score_audit.json", {
        "schema": SCHEMA,
        "months": [f"{month:%Y-%m}" for month in months],
        "base_root": str(base_root),
        "metas": [{"root": str(root), "arm": arm, "weight": weight} for root, arm, weight in metas],
        "score_audit": score_audit,
        "prohibited_outcome_columns_absent": True,
        "policy_join_occurs_only_after_target_free_scores_persisted": True,
        "strict_prequential_mc1_train_months": TRAIN_MONTHS,
        "candidate_score_timestamps_must_match_frozen_base_source_timestamps": True,
        "unavailable_base_hours_are_audited_and_never_imputed": True,
    })
    policy = score_parent._policy(policy_path)
    current = score_parent._join_policy(current_scores, policy)
    bcf = score_parent._join_policy(bcf_scores, policy)
    source_hashes = {
        "base_target_free_scores": _sha256(base_root),
        "policy_labels": _sha256(policy_path),
        "runner_source": _sha256(Path(__file__)),
        "mc1_package_source": _sha256(ROOT / "extreme_price_movements/inference/p8u_mc1_inference_package.py"),
    }
    for index, (meta_root, arm, _weight) in enumerate(metas):
        source_hashes[f"meta_{index}_{arm}_target_free_scores"] = _sha256(meta_root)
    policy_contract = _policy_contract(policy, policy_path)
    current_pred, current_audit, current_index = _fit_score_family(
        current, family="current", months=months, out=out,
        source_hashes=source_hashes, policy_contract=policy_contract, base_decision_ts=base_decision_ts,
    )
    bcf_pred, bcf_audit, bcf_index = _fit_score_family(
        bcf, family="bcf", months=months, out=out,
        source_hashes=source_hashes, policy_contract=policy_contract, base_decision_ts=base_decision_ts,
    )
    combined = portfolio_parent._combined_challenger(current_pred, bcf_pred)
    evaluation_start = months[TRAIN_MONTHS]
    combined = combined.loc[combined["__decision_ts__"].ge(evaluation_start)].copy()
    old_threshold = portfolio_parent.MC1_THRESHOLD_BPS
    try:
        portfolio_parent.MC1_THRESHOLD_BPS = float(threshold_bps)
        metrics = portfolio_parent._portfolio_metrics(
            combined, "p8u_dual_mc1_six_month", f"{evaluation_start:%Y%m}_{months[-1]:%Y%m}", out,
        )
    finally:
        portfolio_parent.MC1_THRESHOLD_BPS = old_threshold
    current_pred.to_parquet(out / "enhanced_current_mc1_predictions.parquet", index=False, compression="zstd")
    bcf_pred.to_parquet(out / "enhanced_bcf_mc1_predictions.parquet", index=False, compression="zstd")
    current_audit.to_parquet(out / "current_mc1_fit_audit.parquet", index=False, compression="zstd")
    bcf_audit.to_parquet(out / "bcf_mc1_fit_audit.parquet", index=False, compression="zstd")
    combined.to_parquet(out / "dual_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame([metrics]).to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    _once(out / "mc1_package_index.json", {
        "schema": SCHEMA,
        "feature_order": list(FEATURES),
        "train_months": TRAIN_MONTHS,
        "shift_state": "prior-21-day causal robust residual shift; package state is a replay snapshot and must be incrementally updated from resolved outcomes at inference",
        "families": current_index + bcf_index,
        "latest_package_by_family": {
            family: next(item for item in reversed(current_index + bcf_index) if item["family"] == family)
            for family in ("current", "bcf")
        },
    })
    correctness = {
        "all_target_free_scores_persisted_before_policy_join": True,
        "target_free_score_panels_are_outcome_free": True,
        "mc1_maps_are_separate_by_family": True,
        "mc1_training_window_is_exactly_six_complete_calendar_months": True,
        "all_mc1_labels_are_resolved_before_held_month": True,
        "all_mc1_models_and_maps_are_serialized": True,
        "all_serialized_static_scores_match_persisted_predictions": True,
        "prior21_shift_uses_only_prior_resolved_labels": True,
        "no_live_or_exchange_mutation": True,
    }
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8U dual-MC1 six-month inference-package construction; no live or exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months],
        "evaluation_start": f"{evaluation_start:%Y-%m}",
        "base_root": str(base_root),
        "metas": [{"root": str(root), "arm": arm, "weight": weight} for root, arm, weight in metas],
        "policy": str(policy_path),
        "policy_contract": policy_contract,
        "threshold_bps": threshold_bps,
        "score_families": {
            "current": "0.75*Base rank + 0.25*weighted Under F120 rank",
            "bcf": "Base rank only",
        },
        "mc1": {
            "train_months": TRAIN_MONTHS,
            "features": list(FEATURES),
            "model": "HistGradientBoostingRegressor depth=2, 80 iterations",
            "band_curve": "monotone score-band structural curve",
            "shift": "prior-21-day robust residual shift; resolved labels only",
            "portfolio_priority": "bcf_mc1_expected_bps",
        },
        "metrics": metrics,
        "source_hashes": source_hashes,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--meta", action="append", default=[], help="ROOT::ARM[::WEIGHT]; repeat for declared meta blend")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--months", default="2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07,2026-08")
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        base_root=args.base_root.resolve(),
        metas=score_parent._parse_meta(args.meta, base_only=False),
        policy_path=args.policy.resolve(),
        months=_months(args.months),
        threshold_bps=float(args.threshold_bps),
        out=args.out.resolve(),
    ))


if __name__ == "__main__":
    main()
