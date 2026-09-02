#!/usr/bin/env python3
"""Turn matched full Meta -> MC1 replays into learned-proxy ground-truth labels.

The learned proxy is trained on actual downstream behaviour, never on a
hand-weighted Meta-native score.  This utility consumes one frozen six-month
MC1 control and one or more candidate replacement-Under MC1 replays.  It
keeps Priority and Gate labels separate:

* Priority is evaluated at a matched dual-admission budget per timestamp,
  ranked by the frozen BCF MC1 priority coordinate.
* Gate uses the real dual Current/BCF >= +50 bps rule.
* The chronological constrained portfolio is stored as an independent later
  confirmation target, not the sole learned label.

All candidate score panels were already persisted target-free by their MC1
producer; this script only audits and aggregates the completed offline
receipts.  It has no score, model, admission, portfolio, live, or exchange
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_p8u_meta_proxy_downstream_labels_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
THRESHOLD_BPS = 50.0
UTILITY_LOW, UTILITY_HIGH = -400.0, 400.0
PRIORITY_COMPONENTS = (
    ("priority_top1_delta_bps", .20),
    ("priority_top2_delta_bps", .40),
    ("priority_captured_utility_delta_bps_per_timestamp", .25),
    ("priority_weekly_q10_delta_bps", .15),
)
GATE_COMPONENTS = (
    ("gate_admitted_ev_delta_bps", .35),
    ("gate_total_utility_delta_bps_per_timestamp", .25),
    ("gate_precision_gt50_delta", .15),
    ("gate_precision_gt100_delta", .10),
    ("gate_volume_delta_per_timestamp", .10),
    ("gate_weekly_q10_delta_bps", .05),
)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _robust_location_scale(values: pd.Series) -> tuple[float, float]:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = raw[np.isfinite(raw)]
    if not len(finite):
        return 0.0, 1.0
    location = float(np.median(finite))
    scale = float(np.median(np.abs(finite - location))) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(finite))
    return location, scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def _week(timestamp: pd.Series) -> pd.Series:
    # Calendar week is explicitly UTC; strip the already-normalised timezone
    # only to avoid pandas' benign PeriodArray warning.
    return pd.to_datetime(timestamp, utc=True, errors="raise").dt.tz_localize(None).dt.to_period("W-SUN").astype(str)


def _era(timestamp: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamp, utc=True, errors="raise").dt.strftime("%Y-%m")


def _root_receipt(root: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(root / "run_manifest.json")
    correctness = _read_json(root / "correctness_report.json")
    if manifest.get("schema") != "strict_r3_p8u_dual_mc1_prequential_v2":
        raise AssertionError(f"{root}: not the strict six-month MC1 schema")
    if int(manifest.get("mc1", {}).get("train_months", -1)) != 6 or float(manifest.get("threshold_bps", np.nan)) != THRESHOLD_BPS:
        raise AssertionError(f"{root}: does not have frozen six-month/+50 MC1 contract")
    required_flags = {
        "all_target_free_scores_persisted_before_policy_join",
        "target_free_score_panels_are_outcome_free",
        "mc1_maps_are_separate_by_family",
        "mc1_training_window_is_exactly_six_complete_calendar_months",
        "all_mc1_labels_are_resolved_before_held_month",
        "prior21_shift_uses_only_prior_resolved_labels",
    }
    if not all(correctness.get(flag) is True for flag in required_flags):
        missing = sorted(flag for flag in required_flags if correctness.get(flag) is not True)
        raise AssertionError(f"{root}: MC1 causality receipt failed {missing}")
    frame = pd.read_parquet(root / "dual_predictions.parquet")
    required = {
        *IDENTITY, "current_final_score", "current_mc1_expected_bps", "bcf_final_score", "bcf_mc1_expected_bps",
        "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"{root}: dual prediction fields missing {missing}")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    if frame.duplicated(list(IDENTITY)).any() or not frame.side_name.eq("long").all():
        raise AssertionError(f"{root}: invalid dual MC1 identities")
    return manifest, frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), correctness


def _assert_matched(control: pd.DataFrame, candidate: pd.DataFrame, *, name: str) -> pd.DataFrame:
    left = control.loc[:, list(IDENTITY)].reset_index(drop=True)
    right = candidate.loc[:, list(IDENTITY)].reset_index(drop=True)
    if len(left) != len(right) or not left.equals(right):
        raise AssertionError(f"{name}: candidate/control MC1 identities are not exact")
    shared_policy = ("policy_path_valid", "policy_net_bps", "policy_label_available_ts")
    for column in shared_policy:
        a, b = control[column], candidate[column]
        if pd.api.types.is_numeric_dtype(a):
            if not np.allclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), equal_nan=True, rtol=0.0, atol=1e-9):
                raise AssertionError(f"{name}: policy source mismatch in {column}")
        elif not a.equals(b):
            raise AssertionError(f"{name}: policy source mismatch in {column}")
    merged = control.loc[:, list(IDENTITY) + ["current_mc1_expected_bps", "bcf_mc1_expected_bps", "current_final_score", "bcf_final_score", "policy_path_valid", "policy_net_bps"]].copy()
    merged = merged.rename(columns={
        "current_mc1_expected_bps": "control_current_mc1", "bcf_mc1_expected_bps": "control_bcf_mc1",
        "current_final_score": "control_current_score", "bcf_final_score": "control_bcf_score",
    })
    for column in ("current_mc1_expected_bps", "bcf_mc1_expected_bps", "current_final_score", "bcf_final_score"):
        merged[f"candidate_{column}"] = pd.to_numeric(candidate[column], errors="coerce").to_numpy(float)
    # BCF is intentionally meta-independent.  An inequality would expose a
    # drift in source Base, map HPO, or the current/BCF family boundary.
    if not np.allclose(merged.control_bcf_score, merged.candidate_bcf_final_score, rtol=0.0, atol=1e-12):
        raise AssertionError(f"{name}: BCF score is not identical under matched Meta replacement")
    if not np.allclose(merged.control_bcf_mc1, merged.candidate_bcf_mc1_expected_bps, rtol=0.0, atol=1e-9):
        raise AssertionError(f"{name}: BCF MC1 mapping is not identical under matched Meta replacement")
    return merged


def _select_priority(group: pd.DataFrame, *, prefix: str, budget: int) -> pd.DataFrame:
    score = f"{prefix}_bcf_mc1" if prefix == "control" else f"candidate_bcf_mc1_expected_bps"
    return group.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(budget)


def _priority_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    work = frame.loc[valid].copy()
    work["control_gate"] = work.control_current_mc1.ge(THRESHOLD_BPS) & work.control_bcf_mc1.ge(THRESHOLD_BPS)
    work["candidate_gate"] = work.candidate_current_mc1_expected_bps.ge(THRESHOLD_BPS) & work.candidate_bcf_mc1_expected_bps.ge(THRESHOLD_BPS)
    rows: list[dict[str, Any]] = []
    for timestamp, group in work.groupby("__decision_ts__", sort=True):
        control_pool = group.loc[group.control_gate]
        candidate_pool = group.loc[group.candidate_gate]
        budget = min(len(control_pool), len(candidate_pool))
        if budget <= 0:
            continue
        control = _select_priority(control_pool, prefix="control", budget=budget)
        candidate = _select_priority(candidate_pool, prefix="candidate", budget=budget)
        ctop1 = float(candidate.iloc[0].policy_net_bps); btop1 = float(control.iloc[0].policy_net_bps)
        ctop2 = float(candidate.head(min(2, budget)).policy_net_bps.mean())
        btop2 = float(control.head(min(2, budget)).policy_net_bps.mean())
        cids, bids = set(candidate.candidate_id), set(control.candidate_id)
        conly = candidate.loc[~candidate.candidate_id.isin(bids), "policy_net_bps"]
        bonly = control.loc[~control.candidate_id.isin(cids), "policy_net_bps"]
        rows.append({
            "__decision_ts__": timestamp, "matched_budget": int(budget),
            "priority_top1_delta_bps": ctop1 - btop1, "priority_top2_delta_bps": ctop2 - btop2,
            "priority_captured_utility_delta_bps": float(candidate.policy_net_bps.clip(UTILITY_LOW, UTILITY_HIGH).sum() - control.policy_net_bps.clip(UTILITY_LOW, UTILITY_HIGH).sum()),
            "priority_candidate_only_minus_control_only_bps": float(conly.mean() - bonly.mean()) if len(conly) and len(bonly) else float("nan"),
            "priority_candidate_only_count": int(len(conly),), "priority_control_only_count": int(len(bonly)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["week"] = _week(result.__decision_ts__)
    result["era"] = _era(result.__decision_ts__)
    return result


def _gate_weekly(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    valid = frame.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    if prefix == "control":
        gate = frame.control_current_mc1.ge(THRESHOLD_BPS) & frame.control_bcf_mc1.ge(THRESHOLD_BPS)
    else:
        gate = frame.candidate_current_mc1_expected_bps.ge(THRESHOLD_BPS) & frame.candidate_bcf_mc1_expected_bps.ge(THRESHOLD_BPS)
    work = frame.loc[valid & gate].copy()
    if work.empty:
        return pd.DataFrame(columns=[
            "era", "week", "ev", "utility", "precision50", "precision100", "volume", "timestamps",
            "utility_per_timestamp", "volume_per_timestamp",
        ])
    work["week"] = _week(work.__decision_ts__)
    work["era"] = _era(work.__decision_ts__)
    grouped = work.groupby(["era", "week"], sort=True)
    result = grouped.agg(
        ev=("policy_net_bps", "mean"),
        utility=("policy_net_bps", lambda values: values.clip(UTILITY_LOW, UTILITY_HIGH).sum()),
        precision50=("policy_net_bps", lambda values: values.gt(50.0).mean()),
        precision100=("policy_net_bps", lambda values: values.gt(100.0).mean()),
        volume=("candidate_id", "size"),
        timestamps=("__decision_ts__", "nunique"),
    ).reset_index()
    # The aggregate Gate label uses utility and volume *per decision
    # timestamp*.  Bootstrap units must live on that same scale.  Leaving the
    # weekly totals here made high-activity weeks look like enormous label
    # uncertainty and collapsed every GateProxy reliability weight.
    denom = result.timestamps.clip(lower=1).to_numpy(float)
    result["utility_per_timestamp"] = result.utility.to_numpy(float) / denom
    result["volume_per_timestamp"] = result.volume.to_numpy(float) / denom
    return result


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean()) if len(frame) else float("nan")


def _quantile_or_nan(frame: pd.DataFrame, column: str, q: float) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def _trial_metrics(name: str, frame: pd.DataFrame, portfolio: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    priority = _priority_weekly(frame)
    if priority.empty:
        raise AssertionError(f"{name}: no matched priority budget after real dual gate")
    control_gate = _gate_weekly(frame, prefix="control")
    candidate_gate = _gate_weekly(frame, prefix="candidate")
    gate = candidate_gate.merge(control_gate, on=["era", "week"], how="outer", suffixes=("_candidate", "_control")).fillna(0.0)
    priority_week = priority.groupby(["era", "week"], sort=True).agg(
        priority_top1_delta_bps=("priority_top1_delta_bps", "mean"),
        priority_top2_delta_bps=("priority_top2_delta_bps", "mean"),
        priority_captured_utility_delta_bps=("priority_captured_utility_delta_bps", "mean"),
        priority_candidate_only_minus_control_only_bps=("priority_candidate_only_minus_control_only_bps", "mean"),
        matched_budget=("matched_budget", "sum"),
    ).reset_index()
    weekly = priority_week.merge(gate, on=["era", "week"], how="outer").fillna(0.0)
    weekly["gate_admitted_ev_delta_bps"] = weekly.ev_candidate - weekly.ev_control
    # Weekly quantities on the same per-timestamp scale as their aggregate
    # counterparts are required for a coherent block bootstrap.
    weekly["gate_total_utility_delta_bps"] = weekly.utility_per_timestamp_candidate - weekly.utility_per_timestamp_control
    weekly["gate_precision_gt50_delta"] = weekly.precision50_candidate - weekly.precision50_control
    weekly["gate_precision_gt100_delta"] = weekly.precision100_candidate - weekly.precision100_control
    weekly["gate_volume_delta"] = weekly.volume_per_timestamp_candidate - weekly.volume_per_timestamp_control
    candidate_gate_rows = int((frame.policy_path_valid.fillna(False).astype(bool) & frame.candidate_current_mc1_expected_bps.ge(THRESHOLD_BPS) & frame.candidate_bcf_mc1_expected_bps.ge(THRESHOLD_BPS)).sum())
    control_gate_rows = int((frame.policy_path_valid.fillna(False).astype(bool) & frame.control_current_mc1.ge(THRESHOLD_BPS) & frame.control_bcf_mc1.ge(THRESHOLD_BPS)).sum())
    timestamps = max(1, int(frame.__decision_ts__.nunique()))
    item = {
        "trial": name, "evaluation_rows": int(len(frame)), "timestamps": timestamps,
        "priority_matched_timestamps": int(priority.__decision_ts__.nunique()), "priority_matched_entries": int(priority.matched_budget.sum()),
        "priority_top1_delta_bps": _mean_or_nan(priority, "priority_top1_delta_bps"),
        "priority_top2_delta_bps": _mean_or_nan(priority, "priority_top2_delta_bps"),
        "priority_candidate_only_minus_control_only_bps": _mean_or_nan(priority, "priority_candidate_only_minus_control_only_bps"),
        "priority_captured_utility_delta_bps_per_timestamp": float(priority.priority_captured_utility_delta_bps.sum() / timestamps),
        "priority_weekly_q10_delta_bps": _quantile_or_nan(priority_week, "priority_top2_delta_bps", .10),
        "gate_admitted_rows": candidate_gate_rows, "control_gate_admitted_rows": control_gate_rows,
        "gate_admitted_ev_bps": _mean_or_nan(candidate_gate, "ev"), "control_gate_admitted_ev_bps": _mean_or_nan(control_gate, "ev"),
        "gate_admitted_ev_delta_bps": _mean_or_nan(candidate_gate, "ev") - _mean_or_nan(control_gate, "ev"),
        "gate_total_utility_delta_bps_per_timestamp": float((candidate_gate.utility.sum() - control_gate.utility.sum()) / timestamps),
        "gate_precision_gt50_delta": _mean_or_nan(candidate_gate, "precision50") - _mean_or_nan(control_gate, "precision50"),
        "gate_precision_gt100_delta": _mean_or_nan(candidate_gate, "precision100") - _mean_or_nan(control_gate, "precision100"),
        "gate_volume_delta_per_timestamp": float((candidate_gate.volume.sum() - control_gate.volume.sum()) / timestamps),
        "gate_weekly_q10_delta_bps": _quantile_or_nan(weekly, "gate_admitted_ev_delta_bps", .10),
        "portfolio_net_ev_bps_per_trade": portfolio.get("net_ev_bps_per_realised_trade"),
        "portfolio_net_sum_bps": portfolio.get("net_sum_bps_realised"),
        "portfolio_worst_month_bps": portfolio.get("worst_month_bps"),
        "portfolio_worst_week_bps": portfolio.get("worst_week_bps"),
        "portfolio_max_drawdown": portfolio.get("max_drawdown"),
        "portfolio_accepted_rows": portfolio.get("accepted_rows"),
    }
    weekly["trial"] = name
    return item, weekly


def _monthly_from_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (trial, era), group in weekly.groupby(["trial", "era"], sort=True):
        rows.append({
            "trial": trial, "era": era,
            "priority_top1_delta_bps": group.priority_top1_delta_bps.mean(),
            "priority_top2_delta_bps": group.priority_top2_delta_bps.mean(),
            "priority_captured_utility_delta_bps_per_timestamp": group.priority_captured_utility_delta_bps.mean(),
            "priority_weekly_q10_delta_bps": group.priority_top2_delta_bps.quantile(.10),
            "gate_admitted_ev_delta_bps": group.gate_admitted_ev_delta_bps.mean(),
            "gate_total_utility_delta_bps_per_timestamp": group.gate_total_utility_delta_bps.mean(),
            "gate_precision_gt50_delta": group.gate_precision_gt50_delta.mean(),
            "gate_precision_gt100_delta": group.gate_precision_gt100_delta.mean(),
            "gate_volume_delta_per_timestamp": group.gate_volume_delta.mean(),
            "gate_weekly_q10_delta_bps": group.gate_admitted_ev_delta_bps.quantile(.10),
        })
    return pd.DataFrame(rows)


def _bootstrap_labels(labels: pd.DataFrame, weekly: pd.DataFrame, *, iterations: int, seed: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    # Freeze robust cross-trial normalisation before estimating per-trial SEs.
    normalisation: dict[str, dict[str, float]] = {}
    for column, _weight in (*PRIORITY_COMPONENTS, *GATE_COMPONENTS):
        normalisation[column] = dict(zip(("location", "scale"), _robust_location_scale(labels[column])))
    output = labels.copy()
    for family, components in (("priority", PRIORITY_COMPONENTS), ("gate", GATE_COMPONENTS)):
        value = np.zeros(len(output), dtype=float)
        for column, weight in components:
            params = normalisation[column]
            value += weight * (pd.to_numeric(output[column], errors="coerce").fillna(params["location"]).to_numpy(float) - params["location"]) / params["scale"]
        output[f"d{family}_raw"] = value
    for index, row in output.iterrows():
        trial = str(row.trial)
        part = weekly.loc[weekly.trial.eq(trial)].copy()
        rng = np.random.default_rng(seed + int.from_bytes(hashlib.sha256(trial.encode()).digest()[:4], "little"))
        if len(part) < 3:
            output.loc[index, "dpriority_bootstrap_se"] = np.nan
            output.loc[index, "dgate_bootstrap_se"] = np.nan
            continue
        samples = np.empty((iterations, 2), dtype=float)
        for draw in range(iterations):
            boot = part.iloc[rng.integers(0, len(part), size=len(part))]
            priority_values = {
                "priority_top1_delta_bps": boot.priority_top1_delta_bps.mean(),
                "priority_top2_delta_bps": boot.priority_top2_delta_bps.mean(),
                "priority_captured_utility_delta_bps_per_timestamp": boot.priority_captured_utility_delta_bps.mean(),
                "priority_weekly_q10_delta_bps": boot.priority_top2_delta_bps.quantile(.10),
            }
            gate_values = {
                "gate_admitted_ev_delta_bps": boot.gate_admitted_ev_delta_bps.mean(),
                "gate_total_utility_delta_bps_per_timestamp": boot.gate_total_utility_delta_bps.mean(),
                "gate_precision_gt50_delta": boot.gate_precision_gt50_delta.mean(),
                "gate_precision_gt100_delta": boot.gate_precision_gt100_delta.mean(),
                "gate_volume_delta_per_timestamp": boot.gate_volume_delta.mean(),
                "gate_weekly_q10_delta_bps": boot.gate_admitted_ev_delta_bps.quantile(.10),
            }
            for pos, (family, components, values) in enumerate((("priority", PRIORITY_COMPONENTS, priority_values), ("gate", GATE_COMPONENTS, gate_values))):
                samples[draw, pos] = sum(
                    weight * (float(values[column]) - normalisation[column]["location"]) / normalisation[column]["scale"]
                    for column, weight in components
                )
        output.loc[index, "dpriority_bootstrap_se"] = float(np.std(samples[:, 0], ddof=1))
        output.loc[index, "dgate_bootstrap_se"] = float(np.std(samples[:, 1], ddof=1))
    for family in ("priority", "gate"):
        target = f"d{family}_raw"; se = f"d{family}_bootstrap_se"
        tau = _robust_location_scale(output[target])[1]
        reliability = tau**2 / (tau**2 + pd.to_numeric(output[se], errors="coerce").fillna(np.inf).to_numpy(float)**2)
        output[f"d{family}_reliability_weight"] = reliability
        output[f"d{family}_shrunk"] = reliability * output[target].to_numpy(float)
    return output, normalisation


def _parse_candidate(values: Iterable[str]) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for value in values:
        name, sep, raw = value.partition("::")
        if not sep or not name or not raw:
            raise ValueError("--candidate must be TRIAL::MC1_ROOT")
        results.append((name, Path(raw).resolve()))
    if len({name for name, _ in results}) != len(results):
        raise ValueError("duplicate candidate trial")
    return results


def _discover_candidate_roots(roots: Iterable[Path]) -> list[tuple[str, Path]]:
    """Discover completed candidate receipts without a hand-copied list.

    A selected-MC1 parent keeps each immutable trial under ``candidate_mc1``.
    We accept only immediate child directories with their own completion
    receipt; ``_root_receipt`` below remains the authoritative schema and
    causality check.  This is deliberately not a glob over arbitrary nested
    directories, which could silently mix partial package folders with trial
    outputs.
    """
    results: list[tuple[str, Path]] = []
    for root in (Path(value).resolve() for value in roots):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            if not (child / "run_manifest.json").is_file() or not (child / "correctness_report.json").is_file():
                raise AssertionError(f"{child}: incomplete candidate MC1 receipt")
            results.append((child.name, child))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="TRIAL::MC1_ROOT; repeat")
    parser.add_argument(
        "--candidate-root", type=Path, action="append", default=[],
        help="directory whose immediate children are completed immutable candidate MC1 roots; repeat",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    candidates = _parse_candidate(args.candidate) + _discover_candidate_roots(args.candidate_root)
    if not candidates:
        raise ValueError("at least one --candidate or --candidate-root is required")
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("duplicate candidate trial across explicit/discovered inputs")
    if args.out.exists():
        raise FileExistsError(args.out)
    control_manifest, control, _ = _root_receipt(args.control.resolve())
    labels: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for name, root in candidates:
        manifest, candidate, _ = _root_receipt(root)
        merged = _assert_matched(control, candidate, name=name)
        metrics, weekly = _trial_metrics(name, merged, _read_json(root / "run_manifest.json")["metrics"])
        labels.append(metrics); weekly_parts.append(weekly)
        audits.append({
            "trial": name, "candidate_root": str(root),
            "same_evaluation_identity": True, "same_policy_source": True,
            "bcf_score_identical": True, "bcf_mc1_identical": True,
            "candidate_six_month_mc1": int(manifest["mc1"]["train_months"]) == 6,
            "candidate_prior21_shift_causal": True,
        })
    weekly = pd.concat(weekly_parts, ignore_index=True)
    output, normalisation = _bootstrap_labels(pd.DataFrame(labels), weekly, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed)
    monthly = _monthly_from_weekly(weekly)
    args.out.mkdir(parents=True)
    output.to_parquet(args.out / "downstream_trial_labels.parquet", index=False, compression="zstd")
    weekly.to_parquet(args.out / "downstream_weekly_labels.parquet", index=False, compression="zstd")
    monthly.to_parquet(args.out / "downstream_monthly_labels.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out / "correctness_audit.parquet", index=False, compression="zstd")
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline learned-proxy MC1 labels only; no Meta HPO selection, no live score/admission/portfolio mutation",
        "control": str(args.control.resolve()), "candidates": [{"trial": name, "root": str(root)} for name, root in candidates],
        "frozen_contract": "six-complete-month separate Current/BCF MC1, prior21 resolved shift, real dual >=50 gate, BCF priority",
        "priority_label": {"matched_budget": "min(candidate, control real-dual-admitted candidates per timestamp)", "components": list(PRIORITY_COMPONENTS)},
        "gate_label": {"real_gate": "Current MC1 >=50 and BCF MC1 >=50", "components": list(GATE_COMPONENTS)},
        "portfolio_label": "candidate root's independent chronological constrained-portfolio metrics; confirmation only",
        "normalisation": normalisation, "bootstrap": {"unit": "week", "iterations": int(args.bootstrap_iterations), "seed": int(args.bootstrap_seed)},
        "gate_bootstrap_scale": "weekly utility and volume normalised per decision timestamp, matching aggregate Gate components",
        "selection_authority": "none; these labels train/falsify later learned proxy models",
    })
    _once(args.out / "correctness_report.json", {
        "all_candidate_mc1_roots_have_target_free_before_policy_join_receipts": True,
        "all_candidate_mc1_roots_have_six_complete_month_prequential_training": True,
        "all_candidate_mc1_roots_use_prior21_resolved_shift": True,
        "candidate_and_control_identity_and_policy_are_exact": True,
        "bcf_score_and_mc1_map_are_identical_for_matched_meta_replacements": True,
        "priority_uses_matched_real_dual_admission_budget": True,
        "gate_uses_real_dual_50bps_contract": True,
        "gate_bootstrap_uses_same_per_timestamp_scale_as_aggregate_label": True,
        "portfolio_is_confirmation_not_the_only_label": True,
        "no_live_or_exchange_mutation": True,
    })
    print(args.out)


if __name__ == "__main__":
    main()
