#!/usr/bin/env python3
"""Strict-prequential bounded Meta -> BCF-MC1 demotion research.

This is deliberately *not* a replacement ranker.  It reads frozen MC1
coordinates and four target-free Meta-head ranks, then fits a shallow prior-only
adverse-path model for each held month.  Its correction is bounded to be zero
or negative and is applied only to a declared BCF-MC1 expected-EV interval.
The independent Current MC1 map, the dual gate, and the chronological portfolio
adapter are unchanged.

The source MC1 parquet files also contain realised-policy columns for historical
reporting.  This runner selects and persists the target-free coordinates first;
only then does it join the canonical policy labels for fold fitting and metrics.
No live or exchange path is imported.
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
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_bounded_bcf_demotion_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
TARGET_FREE_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "__symbol__",
    "enhanced_base_routed", "final_score", "mc1_expected_bps",
)
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
)
HEADS = (
    ("under_f120", "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_xendcg_f120_20260828_v1", "xendcg_selected_under_bps100"),
    ("magnitude", "data_perp/artifacts/strict_r3_p8u_meta_target_query_magnitude_jan_jul2026_20260828_v1", "magnitude_bps__base_band_block28"),
    ("over", "data_perp/artifacts/strict_r3_p8u_meta_target_query_over_jan_jul2026_20260828_v1", "over_atr1__timestamp"),
    ("state", "data_perp/artifacts/strict_r3_p8u_meta_target_query_state_jan_jul2026_20260828_v1", "state_bps__base_band_block28"),
)
BCF_BAND_EDGES = np.asarray([30.0, 50.0, 75.0, 100.0, 150.0, np.inf])
FEATURES = ("bcf_mc1_expected_bps", "current_mc1_expected_bps", *[name for name, _, _ in HEADS])
SEED = 1729


def _once(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in raw.split(",") if value.strip())
    if len(values) < 5 or tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError("need at least five increasing monthly folds")
    return values


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _score_month_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _head_month_path(root: Path, arm: str, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_mc1_month(path: Path, family: str) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    needed = set(TARGET_FREE_COLUMNS)
    missing = needed.difference(available)
    if missing:
        raise AssertionError(f"{path}: missing target-free MC1 fields {sorted(missing)}")
    result = pd.read_parquet(path, columns=list(TARGET_FREE_COLUMNS))
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any() or not result["side_name"].eq("long").all():
        raise AssertionError(f"{path}: invalid MC1 target-free identity")
    result = result.rename(columns={"final_score": f"{family}_final_score", "mc1_expected_bps": f"{family}_mc1_expected_bps"})
    return result


def _load_target_free(
    current_path: Path, bcf_path: Path, month: pd.Timestamp,
) -> tuple[pd.DataFrame, list[Path]]:
    current = _read_mc1_month(current_path, "current")
    bcf = _read_mc1_month(bcf_path, "bcf")
    current = current.loc[current["__decision_ts__"].dt.to_period("M").eq(month.tz_localize(None).to_period("M"))].copy()
    bcf = bcf.loc[bcf["__decision_ts__"].dt.to_period("M").eq(month.tz_localize(None).to_period("M"))].copy()
    compact = current.merge(
        bcf.loc[:, ["candidate_id", "__decision_ts__", "bcf_final_score", "bcf_mc1_expected_bps"]],
        on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one",
    )
    paths = [current_path, bcf_path]
    for name, root_raw, arm in HEADS:
        path = _head_month_path(ROOT / root_raw, arm, month)
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_exit_bar_15m"}
        if forbidden.intersection(schema):
            raise AssertionError(f"{path}: target-free Meta score contains outcome field")
        meta = pd.read_parquet(path, columns=[*IDENTITY, "meta_rank_ts"])
        meta["__decision_ts__"] = pd.to_datetime(meta["__decision_ts__"], utc=True, errors="raise")
        if meta.duplicated(list(IDENTITY)).any():
            raise AssertionError(f"{path}: duplicate Meta identity")
        compact = compact.merge(
            meta.rename(columns={"meta_rank_ts": name}),
            on=list(IDENTITY), how="left", validate="one_to_one",
        )
        paths.append(path)
    if len(compact) != len(current) or compact.loc[:, list(FEATURES)].isna().any().any():
        raise AssertionError(f"{month:%Y-%m}: incomplete target-free join")
    return compact, paths


def _read_policy(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy label source has duplicate candidate IDs")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="raise")
    return policy


def _join_policy(frame: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    result = frame.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(result) != len(frame) or not result["candidate_id"].equals(frame["candidate_id"]):
        raise AssertionError("post-persistence policy join changed target-free identity")
    return result


def _bcf_band(values: pd.Series) -> np.ndarray:
    return np.searchsorted(BCF_BAND_EDGES, pd.to_numeric(values, errors="coerce").to_numpy(float), side="right").astype(np.int16)


def _fit_demotion(
    fit: pd.DataFrame, held: pd.DataFrame, *, limit_bps: float, target: str, authority: float,
) -> tuple[np.ndarray, dict[str, object]]:
    train = fit.loc[
        pd.to_numeric(fit["bcf_mc1_expected_bps"], errors="coerce").between(30.0, limit_bps)
        & fit["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(fit["policy_net_bps"], errors="coerce")),
        [*FEATURES, "policy_net_bps"],
    ].copy()
    if len(train) < 2_000:
        raise ValueError(f"insufficient bounded train support: {len(train)}")
    net = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    threshold = -100.0 if target == "severe100" else 0.0
    y = (net <= threshold).astype(np.int8)
    if y.min() == y.max():
        raise ValueError(f"{target}: degenerate training target")
    medians = train.loc[:, list(FEATURES)].median()
    x = train.loc[:, list(FEATURES)].fillna(medians)
    model = HistGradientBoostingClassifier(
        max_depth=2, max_iter=70, learning_rate=.04, l2_regularization=50.0,
        min_samples_leaf=500, random_state=SEED,
    ).fit(x, y)
    band = _bcf_band(train["bcf_mc1_expected_bps"])
    base_prob = {int(value): float(y[band == value].mean()) for value in np.unique(band)}
    severity = float(np.clip(np.mean(-net[y.astype(bool)]), 50.0, 400.0))
    support = {int(value): int((band == value).sum()) for value in np.unique(band)}
    held_x = held.loc[:, list(FEATURES)].fillna(medians)
    raw_probability = model.predict_proba(held_x)[:, 1]
    held_band = _bcf_band(held["bcf_mc1_expected_bps"])
    baseline = np.asarray([base_prob.get(int(value), float(y.mean())) for value in held_band])
    # A 1,500-row prior makes a low-support BCF band collapse toward zero
    # authority.  No feature-level or held outcome is used in this shrinkage.
    shrink = np.asarray([support.get(int(value), 0) / (support.get(int(value), 0) + 1500.0) for value in held_band])
    active = pd.to_numeric(held["bcf_mc1_expected_bps"], errors="coerce").between(30.0, limit_bps).to_numpy(bool)
    penalty = authority * shrink * np.maximum(0.0, raw_probability - baseline) * severity
    correction = np.where(active, -np.minimum(penalty, 100.0), 0.0).astype(np.float32)
    if (correction > 1e-7).any():
        raise AssertionError("bounded Meta correction promoted a candidate")
    audit = {
        "limit_bps": float(limit_bps), "target": target, "authority": float(authority),
        "train_rows": int(len(train)), "prevalence": float(y.mean()), "severity_bps": severity,
        "support_by_bcf_band": json.dumps({str(key): value for key, value in support.items()}, sort_keys=True),
        "mean_correction_bps_active": float(correction[active].mean()) if active.any() else 0.0,
        "fraction_active": float(active.mean()), "max_demotion_bps": float(-correction.min()),
    }
    return correction, audit


def _monthly_metrics(frame: pd.DataFrame, *, arm: str, out: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for month, group in frame.groupby(frame["__decision_ts__"].dt.to_period("M"), sort=True):
        metrics = parent._portfolio_metrics(group.copy(), arm, str(month).replace("-", ""), out)
        records.append(metrics)
    return pd.DataFrame(records)


def run(
    *, current_path: Path, bcf_path: Path, policy_path: Path, months: tuple[pd.Timestamp, ...],
    out: Path, train_months: int,
) -> None:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    target_free: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    for month in months:
        frame, paths = _load_target_free(current_path, bcf_path, month)
        destination = out / "target_free_inputs" / f"month={month:%Y-%m}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")
        target_free.append(frame); source_paths.extend(paths)
    _once(out / "target_free_input_audit.json", {
        "schema": SCHEMA, "months": [f"{month:%Y-%m}" for month in months],
        "target_free_columns": list(target_free[0].columns), "outcome_columns_absent": True,
        "source_sha256": _sha256(source_paths),
    })
    policy = _read_policy(policy_path)
    panel = _join_policy(pd.concat(target_free, ignore_index=True), policy)
    evaluation = months[train_months:]
    grid = [(limit, target, authority) for limit in (100.0, 150.0, 200.0) for target in ("severe100", "loss0") for authority in (.5, 1.0)]
    raw_records: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    old_threshold = parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS = 50.0
        for limit, target, authority in grid:
            name = f"bcf{int(limit)}_{target}_a{int(authority * 100):03d}"
            folds: list[pd.DataFrame] = []
            audits: list[dict[str, object]] = []
            for month in evaluation:
                fit_start = month - pd.DateOffset(months=train_months)
                fit = panel.loc[
                    panel["__decision_ts__"].ge(fit_start) & panel["__decision_ts__"].lt(month)
                    & panel["policy_label_available_ts"].lt(month)
                ].copy()
                held = panel.loc[panel["__decision_ts__"].ge(month) & panel["__decision_ts__"].lt(_month_end(month))].copy()
                correction, audit = _fit_demotion(fit, held, limit_bps=limit, target=target, authority=authority)
                audit["arm"] = name; audit["held_month"] = f"{month:%Y-%m}"
                held["meta_bounded_demotion_bps"] = correction
                held["bcf_mc1_expected_bps_raw"] = held["bcf_mc1_expected_bps"].to_numpy(float)
                held["bcf_mc1_expected_bps"] = held["bcf_mc1_expected_bps_raw"].to_numpy(float) + correction
                if (held["bcf_mc1_expected_bps"] > held["bcf_mc1_expected_bps_raw"] + 1e-6).any():
                    raise AssertionError("BCF correction promoted a held candidate")
                folds.append(held); audits.append(audit)
            candidate = pd.concat(folds, ignore_index=True)
            metrics = parent._portfolio_metrics(candidate, name, "mayjul2026", out)
            raw_records.append({"arm": name, "limit_bps": limit, "target": target, "authority": authority, **metrics})
            monthly = _monthly_metrics(candidate, arm=name, out=out)
            monthly["arm"] = name
            monthly.to_parquet(out / f"{name}_monthly_metrics.parquet", index=False, compression="zstd")
            pd.DataFrame(audits).to_parquet(out / f"{name}_fold_audit.parquet", index=False, compression="zstd")
            all_predictions.append(candidate.assign(arm=name))
        control = panel.loc[
            panel["__decision_ts__"].ge(evaluation[0]) & panel["__decision_ts__"].lt(_month_end(evaluation[-1]))
        ].copy()
        control["meta_bounded_demotion_bps"] = np.float32(0.0)
        control["bcf_mc1_expected_bps_raw"] = control["bcf_mc1_expected_bps"].to_numpy(float)
        metrics = parent._portfolio_metrics(control, "control", "mayjul2026", out)
        raw_records.insert(0, {"arm": "control", "limit_bps": np.nan, "target": "none", "authority": 0.0, **metrics})
        _monthly_metrics(control, arm="control", out=out).assign(arm="control").to_parquet(out / "control_monthly_metrics.parquet", index=False, compression="zstd")
    finally:
        parent.MC1_THRESHOLD_BPS = old_threshold
    summary = pd.DataFrame(raw_records)
    control_row = summary.loc[summary.arm.eq("control")].iloc[0]
    for column in (
        "accepted_rows", "candidate_admitted_rows", "net_ev_bps_per_realised_trade",
        "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown",
    ):
        if column in summary:
            summary[f"delta_{column}"] = pd.to_numeric(summary[column], errors="coerce") - float(control_row[column])
    summary.to_parquet(out / "summary.parquet", index=False, compression="zstd")
    pd.concat(all_predictions, ignore_index=True).to_parquet(out / "all_arm_predictions.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "schema": SCHEMA,
        "status": "passed",
        "target_free_persisted_before_policy_join": True,
        "meta_target_free_scores_have_no_outcome_columns": True,
        "all_training_labels_available_before_held_month": True,
        "correction_only_active_in_declared_bcf_range": True,
        "correction_never_promotes": True,
        "current_mc1_unchanged": True,
        "dual_admission_threshold_bps": 50.0,
        "auction_priority": "adjusted BCF-MC1 only inside declared interval; otherwise original BCF-MC1",
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "current_mc1": str(current_path), "bcf_mc1": str(bcf_path),
        "policy_labels": str(policy_path), "months": [f"{month:%Y-%m}" for month in months],
        "evaluation_months": [f"{month:%Y-%m}" for month in evaluation], "train_months": train_months,
        "heads": [{"name": name, "root": root, "arm": arm} for name, root, arm in HEADS],
        "grid": [{"limit_bps": a, "target": b, "authority": c} for a, b, c in grid],
        "research_only": True, "live_mutation": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--bcf", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--months", default="2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    months = _months(args.months)
    if args.train_months < 4 or args.train_months >= len(months):
        raise ValueError("--train-months must leave at least one held month")
    run(current_path=args.current.resolve(), bcf_path=args.bcf.resolve(), policy_path=args.policy.resolve(), months=months, out=args.out.resolve(), train_months=args.train_months)


if __name__ == "__main__":
    main()
