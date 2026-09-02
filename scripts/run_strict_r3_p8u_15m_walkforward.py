#!/usr/bin/env python3
"""P8U 15m ordinal-entry challenger, with strict monthly walk-forward.

This is a research-only runner. It never changes P8U/F72/UnderF120 artifacts,
does not call exchange APIs, and writes the 15m panel before outcomes are
joined. For each Jan--Aug 2026 holdout it fits only the preceding two calendar
months whose H12 policy labels were already resolved at the held-month start.

The entry head is deliberately fitted inside a causal dual-MC1 eligibility
population rather than a post-hoc top-k tail.  It evaluates fixed 50/40/30/20
bps MC1 floors, ordinal vetoes, and a bounded MC1 x ordinal combination.  The
continuation-policy head is intentionally a separate runner: it needs
state-by-state counterfactual labels rather than one realised entry outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import (
    FIFTEEN_MINUTE_FEATURE_KEYS,
    VWAP_15M_FEATURE_KEYS,
    compute_15m_features,
    compute_15m_vwap_features,
)


DUAL = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4/dual_predictions.parquet"
# This append-only historical store contains the point-in-time 15m bars back
# through the walk-forward window. The live raw refresh directory starts only
# in July 2026 and is therefore not a valid Jan--Aug research source.
RAW_15M = ROOT / "15m_ohlcv_perp"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_walkforward_20260830_v2"
SEED = 1729
UTILITY_BINS = [-np.inf, -100.0, 0.0, 100.0, 250.0, np.inf]
MC1_THRESHOLDS = (50.0, 40.0, 30.0, 20.0)
ORDINAL_VETO_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0)


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _symbol_from_id(candidate_id: str) -> str:
    return str(candidate_id).split("|", 1)[0]


def _bar_path(symbol: str, root: Path) -> Path:
    stem = symbol.replace("/", "").lower()
    return root / f"{stem}_15m.parquet"


def load_target_free_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return score-only candidate source and a separate outcome table."""
    score_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "bcf_final_score",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps", "enhanced_base_routed", "side_name",
    ]
    label_columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_label_available_ts",
    ]
    scores = pd.read_parquet(DUAL, columns=score_columns)
    labels = pd.read_parquet(DUAL, columns=label_columns)
    scores = scores.loc[scores["side_name"].eq("long")].copy()
    scores["__decision_ts__"] = pd.to_datetime(scores["__decision_ts__"], utc=True)
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    return scores, labels


def select_target_free_rows(scores: pd.DataFrame, candidate_floor_bps: float) -> pd.DataFrame:
    """Select causal source rows by the dual-MC1 floor, never by outcomes."""
    work = scores.loc[scores["enhanced_base_routed"].fillna(False)].copy()
    work["dual_mc1_min_bps"] = work[["bcf_mc1_expected_bps", "current_mc1_expected_bps"]].min(axis=1)
    work = work.loc[work["dual_mc1_min_bps"] >= float(candidate_floor_bps)].copy()
    work = work.sort_values(["__decision_ts__", "bcf_final_score", "candidate_id"], ascending=[True, False, True])
    work["base_timestamp_rank"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    return work.reset_index(drop=True)


def make_target_free_feature_panel(
    candidates: pd.DataFrame,
    bars_root: Path,
    *,
    include_vwap_overlay: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise feature rows without accessing outcomes or labels."""
    rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for symbol, group in candidates.groupby("__symbol__", sort=True):
        path = _bar_path(str(symbol), bars_root)
        if not path.exists():
            coverage.append({"__symbol__": symbol, "candidate_count": len(group), "bar_file": str(path), "status": "missing_bar_file"})
            continue
        # Only the causal 30-hour context through the last decision for this
        # symbol can contribute to this feature family. Predicate push-down is
        # crucial for old instruments whose parquet file starts years before
        # the walk-forward window.
        source_start = _utc(group["__decision_ts__"].min()) - pd.Timedelta(hours=30)
        source_end = _utc(group["__decision_ts__"].max())
        requested_columns = ["open", "high", "low", "close", "volume", "exchange_observed"]
        try:
            bars = pd.read_parquet(
                path,
                columns=requested_columns,
                filters=[
                    ("__index_level_0__", ">=", source_start),
                    ("__index_level_0__", "<", source_end),
                ],
            )
        except (KeyError, ValueError, NotImplementedError, AttributeError):
            bars = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
            bars = bars.loc[(bars.index >= source_start) & (bars.index < source_end)]
        if "ts" in bars.columns:
            bars = bars.set_index("ts")
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars.sort_index()
        complete = 0
        for candidate_id, decision_value, rank, base_score, bcf_mc1, current_mc1, dual_mc1 in group[
            ["candidate_id", "__decision_ts__", "base_timestamp_rank", "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps"]
        ].itertuples(index=False, name=None):
            decision = _utc(decision_value)
            # The contract's longest rolling lookback is 24 hours. Passing a
            # bounded causal slice avoids copying an entire multi-month symbol
            # history for every hourly candidate while preserving exact values.
            feature_bars = bars.loc[
                (bars.index >= decision - pd.Timedelta(hours=30))
                & (bars.index < decision)
            ]
            observed_complete = True
            economic_window = True
            source_status = "complete"
            if "exchange_observed" in feature_bars.columns:
                observed_complete = bool(
                    len(feature_bars) >= 100
                    and feature_bars["exchange_observed"].tail(100).fillna(False).astype(bool).all()
                )
            # Historical carry-forward bars can be marked exchange-observed
            # even when every price is identical and volume is zero.  They
            # are source-complete but have no realised 15m microstructure.
            # Do not impute them or train on their apparent "features".
            recent = feature_bars.tail(100)
            if len(recent) < 100:
                economic_window = False
                source_status = "insufficient_25h_bars"
            else:
                ohlc = recent[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
                positive = bool(np.isfinite(ohlc.to_numpy(float)).all() and (ohlc.to_numpy(float) > 0.0).all())
                changed = int(ohlc["close"].nunique(dropna=True)) >= 3 or bool((ohlc["high"] > ohlc["low"]).sum() >= 4)
                traded = bool((pd.to_numeric(recent["volume"], errors="coerce") > 0.0).sum() >= 4)
                economic_window = bool(positive and changed and traded)
                if not positive:
                    source_status = "nonpositive_or_missing_ohlc"
                elif not changed:
                    source_status = "stale_flat_25h_window"
                elif not traded:
                    source_status = "no_traded_15m_evidence"
            if not observed_complete:
                source_status = "exchange_observation_incomplete"
            values = (
                compute_15m_features(feature_bars, decision, side="long")
                if observed_complete and economic_window
                else {name: np.nan for name in FIFTEEN_MINUTE_FEATURE_KEYS}
            )
            vwap_values = (
                compute_15m_vwap_features(feature_bars, decision, side="long")
                if include_vwap_overlay and observed_complete and economic_window
                else {name: np.nan for name in VWAP_15M_FEATURE_KEYS}
            )
            finite = int(np.isfinite(np.fromiter(values.values(), dtype=float)).sum())
            finite_vwap = int(np.isfinite(np.fromiter(vwap_values.values(), dtype=float)).sum())
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "__decision_ts__": decision,
                    "__symbol__": symbol,
                    "base_timestamp_rank": rank,
                    "bcf_final_score": base_score,
                    "bcf_mc1_expected_bps": bcf_mc1,
                    "current_mc1_expected_bps": current_mc1,
                    "dual_mc1_min_bps": dual_mc1,
                    "feature_source_end_ts": decision - pd.Timedelta(minutes=15),
                    "finite_15m_feature_count": finite,
                    "finite_vwap_feature_count": finite_vwap,
                    "exchange_observed_25h_complete": observed_complete,
                    "economic_15m_window_complete": economic_window,
                    "feature_source_status": source_status,
                    **values,
                    **vwap_values,
                }
            )
            complete += finite == len(FIFTEEN_MINUTE_FEATURE_KEYS)
        coverage.append({"__symbol__": symbol, "candidate_count": len(group), "bar_file": str(path), "status": "ok", "fully_finite_rows": complete})
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def utility_grade(net_bps: pd.Series) -> pd.Series:
    return pd.cut(net_bps, bins=UTILITY_BINS, labels=False, include_lowest=True).astype("float")


def exit_risk_grade(frame: pd.DataFrame) -> pd.Series:
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    bar = pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce")
    grade = pd.Series(np.select([net > 100, net > 0, net > -100, net > -250], [0, 1, 2, 3], default=4), index=frame.index, dtype=float)
    early_adverse = (net <= 0) & (bar >= 0) & (bar <= 16)
    return np.minimum(4.0, grade + early_adverse.astype(float))


def _model(train_rows: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=350,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=max(2, math.ceil(train_rows * 0.02)),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=4.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _monthly_metric(
    rows: pd.DataFrame,
    method: str,
    selected: pd.Series,
    held_month: pd.Timestamp,
    mc1_threshold_bps: float,
    ordinal_veto_threshold: float | None = None,
) -> dict[str, object]:
    chosen = rows.loc[selected.fillna(False)].copy()
    net = pd.to_numeric(chosen.get("policy_net_bps"), errors="coerce").dropna()
    return {
        "held_month": held_month.strftime("%Y-%m"),
        "mc1_threshold_bps": float(mc1_threshold_bps),
        "ordinal_veto_threshold": ordinal_veto_threshold,
        "method": method,
        "selected_rows": int(len(chosen)),
        "timestamps": int(chosen["__decision_ts__"].nunique()),
        "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
        "total_net_bps": float(net.sum()) if len(net) else 0.0,
        "positive_rate": float((net > 0).mean()) if len(net) else np.nan,
        "gt50_rate": float((net >= 50).mean()) if len(net) else np.nan,
    }


def run_walkforward(
    panel: pd.DataFrame,
    output: Path,
    mc1_thresholds: Iterable[float],
    *,
    min_finite_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = panel.copy()
    joined["__decision_ts__"] = pd.to_datetime(joined["__decision_ts__"], utc=True)
    joined["policy_label_available_ts"] = pd.to_datetime(joined["policy_label_available_ts"], utc=True)
    joined = joined.loc[
        joined["policy_path_valid"].fillna(False)
        & joined["policy_net_bps"].notna()
        & (joined["finite_15m_feature_count"] >= min_finite_features)
    ].copy()
    joined["utility_grade"] = utility_grade(joined["policy_net_bps"])
    months = pd.date_range("2026-01-01", "2026-08-01", freq="MS", tz="UTC")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    importance: list[dict[str, object]] = []
    # The feature panel is target-free and contains the broadest requested
    # source floor.  Each head is fitted only on the threshold-specific,
    # pre-decision dual-MC1 population, then evaluated on that exact floor.
    for mc1_threshold in sorted({float(value) for value in mc1_thresholds}, reverse=True):
        scoped = joined.loc[
            pd.to_numeric(joined["dual_mc1_min_bps"], errors="coerce") >= mc1_threshold
        ].copy()
        for held_start in months:
            train_start = held_start - pd.DateOffset(months=2)
            held_end = held_start + pd.offsets.MonthBegin(1)
            train = scoped.loc[
                (scoped["__decision_ts__"] >= train_start)
                & (scoped["__decision_ts__"] < held_start)
                & (scoped["policy_label_available_ts"] < held_start)
            ].copy()
            test = scoped.loc[(scoped["__decision_ts__"] >= held_start) & (scoped["__decision_ts__"] < held_end)].copy()
            if len(train) < 200 or len(test) == 0:
                continue
            x_train = train.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS]
            x_test = test.loc[:, FIFTEEN_MINUTE_FEATURE_KEYS]
            utility = _model(len(train))
            utility.fit(x_train, train["utility_grade"])
            test["ordinal_utility_prediction"] = utility.predict(x_test)
            grade_mean = train.groupby("utility_grade")["policy_net_bps"].mean().reindex(range(5))
            grade_mean = grade_mean.interpolate(limit_direction="both").fillna(float(train["policy_net_bps"].mean()))
            test["ordinal_expected_net_bps"] = np.interp(test["ordinal_utility_prediction"], np.arange(5), grade_mean.to_numpy())
            # A grade of 2 is neutral.  The bounded multiplier leaves neutral
            # MC1 unchanged, cuts grade 0 to half authority and caps grade 4
            # at 1.5x.  The "combined" arm is explicitly exploratory because
            # it can admit a row whose raw MC1 is below the named threshold.
            test["ordinal_mc1_multiplier"] = np.clip(
                0.50 + 0.25 * test["ordinal_utility_prediction"], 0.50, 1.50
            )
            test["combined_dual_mc1_bps"] = (
                pd.to_numeric(test["dual_mc1_min_bps"], errors="coerce")
                * test["ordinal_mc1_multiplier"]
            )
            base_admit = pd.to_numeric(test["dual_mc1_min_bps"], errors="coerce") >= mc1_threshold
            metrics.append(_monthly_metric(test, "dual_mc1_baseline", base_admit, held_start, mc1_threshold))
            for veto_threshold in ORDINAL_VETO_THRESHOLDS:
                selector = base_admit & (test["ordinal_utility_prediction"] >= veto_threshold)
                metrics.append(_monthly_metric(
                    test,
                    "ordinal_veto",
                    selector,
                    held_start,
                    mc1_threshold,
                    veto_threshold,
                ))
            selector = base_admit & (test["combined_dual_mc1_bps"] >= mc1_threshold)
            metrics.append(_monthly_metric(
                test,
                "ordinal_combined_veto",
                selector,
                held_start,
                mc1_threshold,
            ))
            # This is intentionally separate: it permits the ordinal signal
            # to promote a lower raw-MC1 row.  It must never be interpreted as
            # a conservative veto or a live-admission recommendation.
            selector = test["combined_dual_mc1_bps"] >= mc1_threshold
            metrics.append(_monthly_metric(test, "ordinal_combined_promote", selector, held_start, mc1_threshold))
            for feature, value in zip(FIFTEEN_MINUTE_FEATURE_KEYS, utility.feature_importances_, strict=True):
                importance.append({"held_month": held_start.strftime("%Y-%m"), "mc1_threshold_bps": mc1_threshold, "head": "utility", "feature": feature, "importance": int(value)})
            predictions.append(test.assign(held_month=held_start.strftime("%Y-%m"), mc1_threshold_bps=mc1_threshold))
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return prediction_frame, pd.DataFrame(metrics), pd.DataFrame(importance)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bars-root", type=Path, default=RAW_15M)
    parser.add_argument("--mc1-thresholds", default="50,40,30,20", help="comma-separated dual-MC1 floors in bps")
    parser.add_argument("--candidate-floor-bps", type=float, default=20.0, help="lowest target-free dual-MC1 source floor")
    parser.add_argument("--min-finite-features", type=int, default=50)
    parser.add_argument("--start", help="optional UTC decision-time lower bound")
    parser.add_argument("--end", help="optional UTC decision-time exclusive upper bound")
    parser.add_argument("--feature-panel", type=Path, help="reuse an immutable target-free 15m feature artifact")
    parser.add_argument("--candidate-ids-from", type=Path, help="optional target-free artifact whose candidate IDs bound materialisation; never reads outcomes")
    parser.add_argument("--include-vwap-overlay", action="store_true", help="materialise the opt-in causal eight-field VWAP research overlay")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    mc1_thresholds = tuple(sorted({float(value) for value in args.mc1_thresholds.split(",")}, reverse=True))
    if not mc1_thresholds or min(mc1_thresholds) < float(args.candidate_floor_bps):
        raise ValueError("--mc1-thresholds must be non-empty and no lower than --candidate-floor-bps")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.feature_panel:
        feature_panel = pd.read_parquet(args.feature_panel)
        labels = load_target_free_scores()[1]
        coverage = pd.DataFrame(
            [{"status": "reused_target_free_feature_panel", "source": str(args.feature_panel.resolve())}]
        )
    else:
        scores, labels = load_target_free_scores()
        candidates = select_target_free_rows(scores, args.candidate_floor_bps)
        if args.candidate_ids_from:
            source_ids = pd.read_parquet(args.candidate_ids_from.resolve(), columns=["candidate_id"])
            ids = set(source_ids["candidate_id"].dropna().astype(str))
            candidates = candidates.loc[candidates["candidate_id"].astype(str).isin(ids)].copy()
            if candidates.empty:
                raise RuntimeError("--candidate-ids-from has no target-free overlap with the selected score universe")
        if args.start:
            candidates = candidates.loc[candidates["__decision_ts__"] >= _utc(args.start)].copy()
        if args.end:
            candidates = candidates.loc[candidates["__decision_ts__"] < _utc(args.end)].copy()
        feature_panel, coverage = make_target_free_feature_panel(
            candidates,
            args.bars_root,
            include_vwap_overlay=args.include_vwap_overlay,
        )
    feature_panel.to_parquet(output / "target_free_15m_features.parquet", index=False)
    coverage.to_parquet(output / "feature_source_coverage.parquet", index=False)
    manifest = {
        "schema": "p8u-15m-ordinal-mc1-threshold-challenger-v2",
        "feature_count": len(FIFTEEN_MINUTE_FEATURE_KEYS) + (len(VWAP_15M_FEATURE_KEYS) if args.include_vwap_overlay else 0),
        "feature_keys": list(FIFTEEN_MINUTE_FEATURE_KEYS) + (list(VWAP_15M_FEATURE_KEYS) if args.include_vwap_overlay else []),
        "vwap_overlay": bool(args.include_vwap_overlay),
        "vwap_contract": "completed-bar 24h volume-weighted typical price; zero-volume bars have no weight and a no-volume window is missing" if args.include_vwap_overlay else "not materialised",
        "features_target_free_before_labels": True,
        "decision_bar_hidden": True,
        "fold": "previous two calendar months; label_available_ts < held-month start",
        "model": {"family": "LightGBM L1 ordinal regression", "max_depth": 4, "min_leaf_fraction": 0.02, "seed": SEED},
        "ordinal_target": {
            "source": "policy_net_bps",
            "bins_bps": list(UTILITY_BINS),
            "grade_definition": "0: <= -100; 1: (-100, 0]; 2: (0, 100]; 3: (100, 250]; 4: > 250",
            "label_contract": "resolved policy outcome; the policy cost is inherited exactly once from policy_net_bps",
        },
        "dual_mc1_thresholds_bps": list(mc1_thresholds),
        "candidate_dual_mc1_floor_bps": float(args.candidate_floor_bps),
        "candidate_ids_bound": str(args.candidate_ids_from.resolve()) if args.candidate_ids_from else None,
        "ordinal_veto_thresholds": list(ORDINAL_VETO_THRESHOLDS),
        "ordinal_mc1_multiplier": "clip(0.50 + 0.25 * ordinal_grade, 0.50, 1.50)",
        "source_hashes": {"dual_predictions": _sha256(DUAL)},
        "minimum_finite_features": args.min_finite_features,
        "economic_source_guard": "requires 25h exchange-observed bars plus price variation and nonzero-volume evidence; stale flat carry-forward windows are excluded",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.materialize_only:
        return
    # Outcomes are intentionally joined only after the target-free feature
    # artifact is durably materialised above.
    panel = feature_panel.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    predictions, metrics, importance = run_walkforward(
        panel, output, mc1_thresholds, min_finite_features=args.min_finite_features
    )
    predictions.to_parquet(output / "walkforward_predictions.parquet", index=False)
    metrics.to_parquet(output / "monthly_metrics.parquet", index=False)
    importance.to_parquet(output / "feature_importance.parquet", index=False)
    if len(metrics):
        aggregate = metrics.groupby(["mc1_threshold_bps", "method", "ordinal_veto_threshold"], dropna=False, as_index=False).agg(
            held_months=("held_month", "nunique"), selected_rows=("selected_rows", "sum"),
            timestamps=("timestamps", "sum"), total_net_bps=("total_net_bps", "sum"),
        )
        aggregate["net_bps_per_trade"] = np.where(
            aggregate["selected_rows"] > 0,
            aggregate["total_net_bps"] / aggregate["selected_rows"],
            np.nan,
        )
        rates = metrics.groupby(["mc1_threshold_bps", "method", "ordinal_veto_threshold"], dropna=False, as_index=False).agg(
            positive_rate=("positive_rate", "mean"), gt50_rate=("gt50_rate", "mean"),
        )
        aggregate = aggregate.merge(rates, on=["mc1_threshold_bps", "method", "ordinal_veto_threshold"], how="left")
    else:
        aggregate = pd.DataFrame(columns=["mc1_threshold_bps", "method", "ordinal_veto_threshold", "held_months", "selected_rows", "timestamps", "net_bps_per_trade", "total_net_bps", "positive_rate", "gt50_rate"])
    aggregate.to_parquet(output / "aggregate_metrics.parquet", index=False)


if __name__ == "__main__":
    main()
