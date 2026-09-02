#!/usr/bin/env python3
"""Strict-prequential banded CatBoost Meta calibration mapper.

For each frozen BCF-MC1 EV band, a shallow CatBoost regressor consumes only
target-free current/BCF MC1 coordinates plus target-free Meta outputs.  It is
fit before a 28-day calibration reserve; an isotonic map is then fit on that
reserve's out-of-model predictions and fully resolved policy outcomes.  The
held month is predicted without its policy data.  This makes the mapper a
candidate expected-EV coordinate, not a broad Meta ranker.

Offline research only.  No live, execution, canonical, or exchange state is
loaded or modified.
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
import pyarrow.parquet as pq
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_banded_catboost_mapper_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
MC1_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "__symbol__", "enhanced_base_routed",
    "final_score", "mc1_expected_bps",
)
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    "policy_label_available_ts", "policy_cost_bps",
)
HEADS = (
    ("under_f120", "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_xendcg_f120_20260828_v1", "xendcg_selected_under_bps100"),
    ("magnitude", "data_perp/artifacts/strict_r3_p8u_meta_target_query_magnitude_jan_jul2026_20260828_v1", "magnitude_bps__base_band_block28"),
    ("over", "data_perp/artifacts/strict_r3_p8u_meta_target_query_over_jan_jul2026_20260828_v1", "over_atr1__timestamp"),
    ("state", "data_perp/artifacts/strict_r3_p8u_meta_target_query_state_jan_jul2026_20260828_v1", "state_bps__base_band_block28"),
)
FEATURES = ("bcf_mc1_expected_bps", "current_mc1_expected_bps", "bcf_final_score", "current_final_score", *[name for name, _, _ in HEADS])
LOWER_BAND_EDGES = np.asarray([30.0, 50.0, 75.0, 100.0, 150.0])
UPPER_BAND_EDGES = np.asarray([50.0, 75.0, 100.0, 150.0, np.inf])
SEED = 1729
TARGET_LOW_BPS = -300.0
TARGET_HIGH_BPS = 600.0
CALIBRATION_DAYS = 28


def _once(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _sha(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in raw.split(",") if value.strip())
    if len(result) < 5 or len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError("need at least five unique increasing monthly folds")
    return result


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _month_mask(frame: pd.DataFrame, month: pd.Timestamp) -> pd.Series:
    return frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(_month_end(month))


def _band(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(raw), -1, dtype=np.int8)
    for index, (low, high) in enumerate(zip(LOWER_BAND_EDGES, UPPER_BAND_EDGES, strict=True)):
        result[(raw >= low) & (raw < high)] = index
    return result


def _mc1(path: Path, family: str) -> pd.DataFrame:
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(MC1_COLUMNS).difference(schema)
    if missing:
        raise AssertionError(f"{path}: missing MC1 target-free columns {sorted(missing)}")
    result = pd.read_parquet(path, columns=list(MC1_COLUMNS))
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any() or not result["side_name"].eq("long").all():
        raise AssertionError(f"{path}: invalid target-free MC1 identity")
    return result.rename(columns={"final_score": f"{family}_final_score", "mc1_expected_bps": f"{family}_mc1_expected_bps"})


def _head_path(root: Path, arm: str, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _target_free_month(current_path: Path, bcf_path: Path, month: pd.Timestamp) -> tuple[pd.DataFrame, list[Path]]:
    current = _mc1(current_path, "current")
    bcf = _mc1(bcf_path, "bcf")
    current = current.loc[_month_mask(current, month)].copy()
    bcf = bcf.loc[_month_mask(bcf, month)].copy()
    frame = current.merge(
        bcf.loc[:, ["candidate_id", "__decision_ts__", "bcf_final_score", "bcf_mc1_expected_bps"]],
        on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one",
    )
    paths = [current_path, bcf_path]
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_exit_bar_15m"}
    for name, root_raw, arm in HEADS:
        path = _head_path(ROOT / root_raw, arm, month)
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        if forbidden.intersection(schema):
            raise AssertionError(f"{path}: Meta score is not target-free")
        meta = pd.read_parquet(path, columns=[*IDENTITY, "meta_rank_ts"])
        meta["__decision_ts__"] = pd.to_datetime(meta["__decision_ts__"], utc=True, errors="raise")
        if meta.duplicated(list(IDENTITY)).any():
            raise AssertionError(f"{path}: duplicate Meta identities")
        frame = frame.merge(meta.rename(columns={"meta_rank_ts": name}), on=list(IDENTITY), how="left", validate="one_to_one")
        paths.append(path)
    if len(frame) != len(current) or frame.loc[:, list(FEATURES)].isna().any().any():
        raise AssertionError(f"{month:%Y-%m}: incomplete target-free score merge")
    frame["bcf_score_band"] = _band(frame["bcf_mc1_expected_bps"])
    return frame, paths


def _policy(path: Path) -> pd.DataFrame:
    result = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    if result.candidate_id.duplicated().any():
        raise AssertionError("canonical policy labels have duplicate candidate IDs")
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="raise")
    return result


def _join_policy(scores: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    joined = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(joined) != len(scores) or not joined.candidate_id.equals(scores.candidate_id):
        raise AssertionError("policy join altered persisted target-free score identity")
    return joined


def _fit_band(
    model_fit: pd.DataFrame, calibration: pd.DataFrame, held: pd.DataFrame, *, band: int,
) -> tuple[np.ndarray, dict[str, object]]:
    subset = model_fit.loc[model_fit["bcf_score_band"].eq(band)].copy()
    reserve = calibration.loc[calibration["bcf_score_band"].eq(band)].copy()
    target = pd.to_numeric(subset["policy_net_bps"], errors="coerce").clip(TARGET_LOW_BPS, TARGET_HIGH_BPS)
    reserve_target = pd.to_numeric(reserve["policy_net_bps"], errors="coerce").clip(TARGET_LOW_BPS, TARGET_HIGH_BPS)
    # The immutable Jan--Apr warm-up has sparse middle-Band support.  We keep
    # bands separate (never pool their semantics), use depth one there, and
    # fail closed below a small but meaningful model/reserve floor.
    if len(subset) < 500 or len(reserve) < 150 or target.nunique() < 5 or reserve_target.nunique() < 5:
        raise ValueError(f"band={band}: insufficient strict prequential support model={len(subset)} reserve={len(reserve)}")
    medians = subset.loc[:, list(FEATURES)].median()
    depth = 1 if len(subset) < 1_500 or len(reserve) < 500 else 2
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=100, depth=depth, learning_rate=0.035,
        l2_leaf_reg=50.0, random_seed=SEED + int(band), verbose=False,
        allow_writing_files=False, thread_count=8,
    )
    model.fit(subset.loc[:, list(FEATURES)].fillna(medians), target)
    reserve_raw = model.predict(reserve.loc[:, list(FEATURES)].fillna(medians))
    calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(reserve_raw, reserve_target)
    hold_mask = held["bcf_score_band"].eq(band).to_numpy(bool)
    mapped = np.full(len(held), np.nan, dtype=np.float32)
    if hold_mask.any():
        held_raw = model.predict(held.loc[hold_mask, list(FEATURES)].fillna(medians))
        mapped[hold_mask] = calibrator.predict(held_raw).astype(np.float32)
    return mapped, {
        "band": int(band), "band_low_bps": float(LOWER_BAND_EDGES[band]), "band_high_bps": float(UPPER_BAND_EDGES[band]),
        "model_rows": int(len(subset)), "reserve_rows": int(len(reserve)),
        "model_target_mean_bps": float(target.mean()), "reserve_target_mean_bps": float(reserve_target.mean()),
        "isotonic_knots": int(len(calibrator.X_thresholds_)), "held_rows": int(hold_mask.sum()), "depth": depth,
    }


def _predict_month(panel: pd.DataFrame, month: pd.Timestamp, train_months: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    reserve_start = month - pd.Timedelta(days=CALIBRATION_DAYS)
    model_start = month - pd.DateOffset(months=train_months)
    valid = (
        panel["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))
    )
    model_fit = panel.loc[
        panel["__decision_ts__"].ge(model_start) & panel["__decision_ts__"].lt(reserve_start)
        & panel["policy_label_available_ts"].lt(reserve_start) & valid,
    ].copy()
    calibration = panel.loc[
        panel["__decision_ts__"].ge(reserve_start) & panel["__decision_ts__"].lt(month)
        & panel["policy_label_available_ts"].lt(month) & valid,
    ].copy()
    held = panel.loc[_month_mask(panel, month)].copy()
    held["catboost_expected_bps"] = np.nan
    audits: list[dict[str, object]] = []
    for band in range(len(LOWER_BAND_EDGES)):
        mapped, audit = _fit_band(model_fit, calibration, held, band=band)
        mask = np.isfinite(mapped)
        held.loc[mask, "catboost_expected_bps"] = mapped[mask]
        audit.update({"held_month": f"{month:%Y-%m}", "model_start": str(model_start), "reserve_start": str(reserve_start)})
        audits.append(audit)
    # Below +30 BCF mapping support is explicitly unavailable: CatBoost cannot
    # manufacture admissions from a region omitted by its own band contract.
    return held, audits


def _metrics(frame: pd.DataFrame, arm: str, period: str, out: Path, *, require_bcf: bool) -> dict[str, object]:
    work = frame.copy()
    work["bcf_mc1_expected_bps_raw"] = work["bcf_mc1_expected_bps"].to_numpy(float)
    work["bcf_mc1_expected_bps"] = work["catboost_expected_bps"].to_numpy(float)
    valid = (
        work["enhanced_base_routed"].fillna(False).astype(bool)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(work["catboost_expected_bps"])
        & pd.to_numeric(work["current_mc1_expected_bps"], errors="coerce").ge(50.0)
        & pd.to_numeric(work["catboost_expected_bps"], errors="coerce").ge(50.0)
    )
    if require_bcf:
        valid &= pd.to_numeric(work["bcf_mc1_expected_bps_raw"], errors="coerce").ge(50.0)
    candidate = work.loc[valid].copy()
    # Parent adapter retains the exact canonical global chronological auction.
    old_threshold = parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS = 50.0
        result = parent._portfolio_metrics(candidate, arm, period, out)
    finally:
        parent.MC1_THRESHOLD_BPS = old_threshold
    result["catboost_mapped_admitted_rows"] = int(len(candidate))
    result["bcf_confirmation_required"] = bool(require_bcf)
    return result


def _high_bcf_priority_only_metrics(frame: pd.DataFrame, arm: str, period: str, out: Path) -> dict[str, object]:
    """Retain exact dual admission; change auction priority only at BCF >=150.

    This is the single role justified by the held calibration diagnostic: the
    CatBoost score is not an absolute admission coordinate, but its within-band
    ordering is directionally useful in the 150+ BCF region.  Candidates below
    the boundary retain their byte-identical BCF priority.
    """
    from extreme_price_movements.portfolio_policy_replay import replay_candidates
    from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics as report_metrics, _params

    old_threshold = parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS = 50.0
        candidates = parent._portfolio_input(frame.copy(), "bcf_mc1_expected_bps")
    finally:
        parent.MC1_THRESHOLD_BPS = old_threshold
    priority_by_id = frame.set_index("candidate_id")["bcf_mc1_expected_bps"].astype(float).copy()
    high = frame["bcf_mc1_expected_bps"].ge(150.0) & frame["catboost_expected_bps"].notna()
    priority_by_id.loc[frame.loc[high, "candidate_id"]] = frame.loc[high, "catboost_expected_bps"].to_numpy(float)
    priority = candidates["candidate_id"].map(priority_by_id)
    if priority.isna().any():
        raise AssertionError("high-band priority could not be aligned to portfolio identities")
    candidates["calibrated_score"] = priority.to_numpy(float)
    candidates["mapped_expected_net_bps"] = priority.to_numpy(float)
    candidates["normalized_rank_score"] = candidates.groupby("timestamp", sort=False)["calibrated_score"].rank(pct=True, method="average")
    candidates["strategy_rank_pct"] = candidates["normalized_rank_score"].to_numpy(float)
    decisions, equity, _ = replay_candidates(candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0)
    # Match the parent controlled adapter: every candidate reaching this
    # offline replay was label-valid by construction, while the generic
    # normaliser deliberately omits the research-only coverage column.
    decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{arm}_{period}_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}_{period}_equity.parquet", index=False, compression="zstd")
    result = report_metrics(decisions, equity, arm, period)
    result["candidate_admitted_rows"] = int(len(candidates))
    result["high_bcf_priority_only"] = True
    result["high_bcf_priority_boundary_bps"] = 150.0
    return result


def run(*, current_path: Path, bcf_path: Path, policy_path: Path, months: tuple[pd.Timestamp, ...], train_months: int, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    for month in months:
        frame, paths = _target_free_month(current_path, bcf_path, month)
        destination = out / "target_free_inputs" / f"month={month:%Y-%m}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False, compression="zstd")
        frames.append(frame); sources.extend(paths)
    _once(out / "target_free_input_audit.json", {
        "schema": SCHEMA, "source_sha256": _sha(sources), "months": [f"{month:%Y-%m}" for month in months],
        "target_free_columns": list(frames[0].columns), "outcome_columns_absent": True,
        "policy_labels_joined_after_target_free_persistence": True,
    })
    panel = _join_policy(pd.concat(frames, ignore_index=True), _policy(policy_path))
    evaluation = months[train_months:]
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in evaluation:
        held, fold_audit = _predict_month(panel, month, train_months)
        predictions.append(held); audits.extend(fold_audit)
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_parquet(out / "policy_joined_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    old_threshold = parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS = 50.0
        control = parent._portfolio_metrics(prediction, "dual_mc1_control", "mayjul2026", out)
    finally:
        parent.MC1_THRESHOLD_BPS = old_threshold
    replace = _metrics(prediction, "catboost_replace_bcf", "mayjul2026", out, require_bcf=False)
    confirm = _metrics(prediction, "catboost_confirm_bcf", "mayjul2026", out, require_bcf=True)
    priority_only = _high_bcf_priority_only_metrics(prediction, "catboost_priority_bcf150plus", "mayjul2026", out)
    summary = pd.DataFrame([
        {"arm": "dual_mc1_control", **control},
        {"arm": "catboost_replace_bcf", **replace},
        {"arm": "catboost_confirm_bcf", **confirm},
        {"arm": "catboost_priority_bcf150plus", **priority_only},
    ])
    base = summary.loc[summary["arm"].eq("dual_mc1_control")].iloc[0]
    for column in ("accepted_rows", "candidate_admitted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"):
        summary[f"delta_{column}"] = pd.to_numeric(summary[column], errors="coerce") - float(base[column])
    summary.to_parquet(out / "summary.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "schema": SCHEMA, "status": "passed", "target_free_persisted_before_label_join": True,
        "model_fit_labels_available_before_reserve": True, "isotonic_uses_out_of_model_reserve_predictions": True,
        "held_policy_outcomes_not_model_inputs": True, "bands": list(zip(LOWER_BAND_EDGES.tolist(), UPPER_BAND_EDGES.tolist())),
        "catboost_depth": "2; depth-1 when the strict isolated band has <1,500 model or <500 reserve rows",
        "catboost_iterations": 100, "target_clip_bps": [TARGET_LOW_BPS, TARGET_HIGH_BPS],
        "threshold_bps": 50.0, "portfolio": "existing global chronological auction",
        "priority_only_arm": "retains exact dual admission and modifies CatBoost priority only where raw BCF MC1 EV >=150 bps",
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "current_mc1": str(current_path), "bcf_mc1": str(bcf_path), "policy": str(policy_path),
        "months": [f"{month:%Y-%m}" for month in months], "evaluation_months": [f"{month:%Y-%m}" for month in evaluation],
        "train_months": train_months, "calibration_days": CALIBRATION_DAYS,
        "heads": [{"name": name, "root": root, "arm": arm} for name, root, arm in HEADS],
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
        raise ValueError("--train-months must be at least 4 and leave a held month")
    run(current_path=args.current.resolve(), bcf_path=args.bcf.resolve(), policy_path=args.policy.resolve(), months=months, train_months=args.train_months, out=args.out.resolve())


if __name__ == "__main__":
    main()
