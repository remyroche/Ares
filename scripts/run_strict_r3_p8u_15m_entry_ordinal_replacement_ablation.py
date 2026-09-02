#!/usr/bin/env python3
"""Strict-OOS weighted ordinal entry-head replacement-promotion ablation.

The source population is fixed before policy labels are opened.  A five-grade
ordinal model trains on dual-MC1 >=20 bps candidates, with an explicit weight
on the 20--30 bps reserve band.  Reserve candidates may enter only as a
replacement for a demoted core candidate; they never bypass a model score,
causal label boundary, or the normal global portfolio auction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params
from scripts.replay_strict_r3_p8u_15m_continuation_portfolio import _attach_ids, _period_metrics


FEATURE_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v4_manifested_results/target_free_15m_features.parquet"
LABEL_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_rich_policy_labels_20260830_v1_control"
BASELINE_PREDICTIONS = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_retrain_20260830_v3_control/walkforward_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_ordinal_replacement_20260830_v1"
GRADE_BINS = [-np.inf, -100.0, 0.0, 50.0, 150.0, np.inf]
RESERVE_FLOOR = 20.0
CORE_FLOOR = 30.0
WEIGHTS = (1.0, 1.5, 2.0)


def _labels(root: Path) -> pd.DataFrame:
    parts = sorted(root.resolve().glob("policy_parts/symbol=*/policy_labels.parquet"))
    if not parts:
        raise FileNotFoundError(f"no labels under {root}")
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in parts], ignore_index=True)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    if frame.candidate_id.duplicated().any():
        raise AssertionError("policy label identities are not unique")
    valid = frame.policy_path_valid.fillna(False).astype(bool)
    if not np.isclose(
        pd.to_numeric(frame.loc[valid, "policy_gross_bps"], errors="coerce")
        - pd.to_numeric(frame.loc[valid, "policy_net_bps"], errors="coerce"),
        100.0, rtol=0.0, atol=1e-8,
    ).all():
        raise AssertionError("rich policy cost is not exactly 100 bps once")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    return frame


def _grade(values: pd.Series) -> pd.Series:
    return pd.cut(values, bins=GRADE_BINS, labels=False, include_lowest=True).astype(int)


def _fit(train: pd.DataFrame, reserve_weight: float) -> tuple[lgb.LGBMClassifier, IsotonicRegression]:
    y = _grade(pd.to_numeric(train.policy_net_bps, errors="raise"))
    weights = np.where(pd.to_numeric(train.dual_mc1_min_bps, errors="raise") < CORE_FLOOR, reserve_weight, 1.0)
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=400, learning_rate=0.025,
        max_depth=4, num_leaves=15, min_child_samples=max(4, int(np.ceil(len(train) * 0.02))),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=1729, n_jobs=2, verbosity=-1,
    )
    model.fit(train.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS], y, sample_weight=weights)
    probabilities = np.asarray(model.predict_proba(train.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS]), dtype=float)
    grade_score = probabilities @ np.arange(5, dtype=float)
    # This maps the ordinal evidence back to policy bps using only the prior
    # resolved fold.  Isotonic calibration preserves the ordinal direction.
    calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    calibrator.fit(grade_score, pd.to_numeric(train.policy_net_bps, errors="raise"))
    return model, calibrator


def _predict(model: lgb.LGBMClassifier, calibrator: IsotonicRegression, rows: pd.DataFrame) -> pd.DataFrame:
    probabilities = np.asarray(model.predict_proba(rows.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS]), dtype=float)
    score = probabilities @ np.arange(5, dtype=float)
    out = rows.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "dual_mc1_min_bps", "bcf_mc1_expected_bps"]].copy()
    out["ordinal_score"] = score
    out["ordinal_expected_bps"] = calibrator.predict(score)
    out["ordinal_p_good"] = probabilities[:, 3] + probabilities[:, 4]
    return out


def _candidate_table(rows: pd.DataFrame, labels: pd.DataFrame, *, priority: pd.Series, arm: str) -> pd.DataFrame:
    label_columns = [column for column in labels.columns if column != "candidate_id" and column in rows.columns]
    frame = rows.drop(columns=label_columns, errors="ignore").merge(labels, on="candidate_id", how="left", validate="one_to_one")
    if not frame.policy_path_valid.fillna(False).all():
        raise AssertionError(f"{arm}: selected invalid rich-policy path")
    exit_bar = pd.to_numeric(frame.policy_exit_bar_15m, errors="raise").astype(int)
    gross_bps = pd.to_numeric(frame.policy_gross_bps, errors="raise")
    candidates = pd.DataFrame({
        "timestamp": pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise"),
        "candidate_id": frame.candidate_id.astype(str),
        "symbol": frame["__symbol__"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_p8u_15m_ordinal_replacement_long",
        "policy_archetype": arm,
        "normalized_rank_score": 1.0,
        "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0,
        "portfolio_priority_adjustment": priority.to_numpy(float),
        "entry_price": pd.to_numeric(frame.policy_entry_price, errors="raise"),
        "exit_timestamp": pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise") + pd.to_timedelta((exit_bar + 1) * 15, unit="m"),
        "exit_price": pd.to_numeric(frame.policy_exit_price, errors="raise"),
        "net_return": pd.to_numeric(frame.policy_net_bps, errors="raise") / 10_000.0,
        "gross_return": gross_bps / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": frame.policy_exit_reason.astype(str),
        "fees_bps": 100.0, "expected_friction_bps": 0.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
    })
    return normalise_candidate_table(candidates)


def _run_portfolio(rows: pd.DataFrame, labels: pd.DataFrame, *, priority: pd.Series, arm: str, output: Path) -> dict[str, object]:
    candidates = _candidate_table(rows, labels, priority=priority, arm=arm)
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    decisions = _attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    monthly = _period_metrics(accepted, "month")
    monthly["arm"] = arm
    monthly.to_parquet(output / f"{arm}_monthly.parquet", index=False)
    net_bps = pd.to_numeric(candidates.iloc[pd.to_numeric(accepted.candidate_index, errors="raise").astype(int).to_numpy()].net_return, errors="raise") * 10_000.0
    return {
        "arm": arm, "entry_selected": len(candidates), "portfolio_accepted": len(accepted),
        "policy_net_bps_per_trade": float(net_bps.mean()) if len(net_bps) else np.nan,
        "total_policy_net_bps": float(net_bps.sum()), "policy_win_rate": float((net_bps > 0).mean()) if len(net_bps) else np.nan,
        **compute_replay_metrics(candidates, decisions, equity, params=params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-panel", type=Path, default=FEATURE_PANEL)
    parser.add_argument("--labels-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--baseline-predictions", type=Path, default=BASELINE_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    features = pd.read_parquet(args.feature_panel.resolve())
    labels = _labels(args.labels_root)
    panel = features.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    panel = panel.loc[
        pd.to_numeric(panel.dual_mc1_min_bps, errors="coerce").ge(RESERVE_FLOOR)
        & panel.policy_path_valid.fillna(False)
        & panel.policy_label_available_ts.notna()
        & pd.to_numeric(panel.finite_15m_feature_count, errors="coerce").ge(50)
    ].copy()
    for feature in FIFTEEN_MINUTE_FEATURE_KEYS:
        if feature not in panel:
            raise AssertionError(f"missing entry feature {feature}")
    predictions: list[pd.DataFrame] = []
    held_months = pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC")
    for held in held_months:
        start, end = held - pd.DateOffset(months=2), held + pd.offsets.MonthBegin(1)
        train = panel.loc[
            panel["__decision_ts__"].ge(start) & panel["__decision_ts__"].lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        observed = set(train["__decision_ts__"].dt.strftime("%Y-%m"))
        required = {(held - pd.DateOffset(months=1)).strftime("%Y-%m"), (held - pd.DateOffset(months=2)).strftime("%Y-%m")}
        test = panel.loc[panel["__decision_ts__"].ge(held) & panel["__decision_ts__"].lt(end)].copy()
        if not required.issubset(observed) or len(train) < 200 or test.empty:
            continue
        for weight in WEIGHTS:
            model, calibrator = _fit(train, weight)
            predicted = _predict(model, calibrator, test)
            predicted["reserve_weight"] = weight
            predicted["held_month"] = held.strftime("%Y-%m")
            predictions.append(predicted)
    all_predictions = pd.concat(predictions, ignore_index=True)
    if all_predictions.duplicated(["candidate_id", "reserve_weight"]).any():
        raise AssertionError("one OOS ordinal prediction per candidate/weight is required")
    output.mkdir(parents=True, exist_ok=False)
    all_predictions.to_parquet(output / "ordinal_predictions.parquet", index=False)
    # Current Huber-veto baseline is a frozen selected set, rather than a
    # newly fitted comparison model.  It retains the exact prior authority.
    baseline = pd.read_parquet(args.baseline_predictions.resolve())
    selected_col = "selected__veto_pred_ge_0"
    baseline = baseline.loc[
        baseline.floor_bps.eq(30.0) & baseline.model_spec.eq("lgb_huber_bps") & baseline[selected_col].fillna(False)
    , ["candidate_id"]].copy()
    baseline["candidate_id"] = baseline.candidate_id.astype(str)
    if baseline.candidate_id.duplicated().any():
        raise AssertionError("baseline entry selection is not unique")
    methods: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    baseline_rows = panel.merge(baseline, on="candidate_id", how="inner", validate="one_to_one")
    methods["H0_huber_core_veto"] = (baseline_rows, pd.to_numeric(baseline_rows.bcf_mc1_expected_bps, errors="raise"))
    for weight in WEIGHTS:
        pred = all_predictions.loc[all_predictions.reserve_weight.eq(weight)].copy()
        joined = panel.merge(pred, on=["candidate_id", "__decision_ts__", "__symbol__", "dual_mc1_min_bps", "bcf_mc1_expected_bps"], how="inner", validate="one_to_one")
        core = pd.to_numeric(joined.dual_mc1_min_bps, errors="raise").ge(CORE_FLOOR)
        core_ok = core & pd.to_numeric(joined.ordinal_expected_bps, errors="raise").ge(0.0)
        methods[f"O{weight:g}_core_veto"] = (joined.loc[core_ok].copy(), pd.to_numeric(joined.loc[core_ok, "bcf_mc1_expected_bps"], errors="raise"))
        if weight == 2.0:
            for promote_floor in (50.0, 75.0):
                reserve = (~core) & pd.to_numeric(joined.ordinal_expected_bps, errors="raise").ge(promote_floor) & pd.to_numeric(joined.ordinal_p_good, errors="raise").ge(0.50)
                chosen = joined.loc[core_ok | reserve].copy()
                bcf = pd.to_numeric(chosen.bcf_mc1_expected_bps, errors="raise")
                expected = pd.to_numeric(chosen.ordinal_expected_bps, errors="raise")
                methods[f"R2_p{int(promote_floor)}_bcf"] = (chosen, bcf)
                # Promotion carries only half of the ordinal excess over BCF;
                # it cannot fully replace the frozen score authority.
                methods[f"R2_p{int(promote_floor)}_blend50"] = (chosen, bcf + 0.50 * (expected - bcf))
    summary: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []
    for arm, (rows, priority) in methods.items():
        summary.append(_run_portfolio(rows, labels, priority=priority.reset_index(drop=True), arm=arm, output=output))
        monthly_rows.append(pd.read_parquet(output / f"{arm}_monthly.parquet"))
    summary_frame = pd.DataFrame(summary)
    control = summary_frame.loc[summary_frame.arm.eq("H0_huber_core_veto")].iloc[0]
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "compounded_return", "max_drawdown", "sortino", "worst_week"):
        summary_frame[f"delta_vs_huber_{metric}"] = summary_frame[metric] - control[metric]
    summary_frame.to_parquet(output / "portfolio_summary.parquet", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_parquet(output / "monthly_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_p8u_15m_weighted_ordinal_replacement_v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation",
        "parent_policy": "unchanged current frozen rich policy; labels materialised separately from target-free identities",
        "oos": "2026-04 through 2026-08; exact two complete prior calendar months, labels resolved before held boundary",
        "training_population": "dual MC1 >=20 bps with reserve 20--30 bps weights 1.0/1.5/2.0",
        "core": "dual MC1 >=30 bps and ordinal expected policy net >=0",
        "replacement": "20--30 bps reserve requires ordinal expected >=50 or >=75 bps and P(good grade)>=0.50",
        "priority": "BCF-only or bounded 50% blend of BCF MC1 and ordinal expected bps",
        "portfolio": asdict(portfolio_params()),
        "cost": "100 bps embedded once in the frozen policy labels",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
