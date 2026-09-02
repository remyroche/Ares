#!/usr/bin/env python3
"""Strict-OOS C1 continuation-policy multiplier ablation (research only).

The continuation ordinal model is updated after each completed 15-minute bar.
It may tighten, but never loosen, the parent policy's hard-stop, trailing
activation, or trailing giveback floor.  This is deliberately separate from
the predictive gate and never changes a live/replay policy contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from extreme_price_movements.p8u_continuation_state import (
    CONTINUATION_STATE_FEATURE_KEYS,
    replay_open_long_policy_with_continuation_modulator,
)
from scripts import run_strict_r3_p8u_15m_continuation_walkforward as base


STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v3"
PARITY_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v5_sourcealigned_results"
FEATURE_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v3/target_free_15m_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_c1_tightening_ablation_20260830_v1"
THRESHOLDS = (50.0, 40.0, 30.0, 20.0)
CONFIGS = {
    # C0 travels through the exact same causal state/model callback and must
    # reproduce the frozen parent.  It is a plumbing regression control.
    "C0_parent": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.0},
    "C1_mild": {"sl_tighten": 0.15, "giveback_tighten": 0.20, "activation_earlier": 0.15},
    "C1_medium": {"sl_tighten": 0.30, "giveback_tighten": 0.35, "activation_earlier": 0.30},
    "C1_strong": {"sl_tighten": 0.45, "giveback_tighten": 0.50, "activation_earlier": 0.45},
    # Decompose the medium joint arm before considering any exit-policy change.
    "C1_sl_only": {"sl_tighten": 0.30, "giveback_tighten": 0.0, "activation_earlier": 0.0},
    "C1_giveback_only": {"sl_tighten": 0.0, "giveback_tighten": 0.35, "activation_earlier": 0.0},
    "C1_activation_only": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.30},
    # Bounded authority curve for the HPO-selected continuation model. These
    # are all one-way tightening controls; the parent is C0.
    "C1_activation_10": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.10},
    "C1_activation_20": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.20},
    "C1_activation_40": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.40},
    "C1_activation_50": {"sl_tighten": 0.0, "giveback_tighten": 0.0, "activation_earlier": 0.50},
}
MODEL_FEATURES = (*FIFTEEN_MINUTE_FEATURE_KEYS, *CONTINUATION_STATE_FEATURE_KEYS)
MODEL_SPECS = ("lgb_l1_grade", "lgb_l2_grade", "cat_mae_grade", "cat_rmse_grade", "lgb_l1_bps")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_panel(state_root: Path, parity_root: Path) -> pd.DataFrame:
    states = base._read_state_panel(state_root / "target_free_continuation_state_parts")
    labels = base._labels()
    incompatible = pd.read_parquet(parity_root / "parent_policy_state_incompatible_candidates.parquet")
    ids = set(incompatible["candidate_id"].astype(str))
    panel = states.merge(labels, on="candidate_id", how="left", validate="many_to_one")
    del states, labels, incompatible
    panel = panel.loc[
        ~panel["candidate_id"].astype(str).isin(ids)
        & panel["policy_path_valid"].fillna(False)
        & panel["policy_net_bps"].notna()
        & panel["policy_label_available_ts"].notna()
        & panel["finite_15m_feature_count"].ge(50)
    ].copy()
    panel["entry_decision_ts"] = pd.to_datetime(panel["entry_decision_ts"], utc=True)
    panel["policy_label_available_ts"] = pd.to_datetime(panel["policy_label_available_ts"], utc=True)
    panel["continuation_delta_bps"] = pd.to_numeric(panel["policy_net_bps"], errors="coerce") - pd.to_numeric(panel["current_PnL"], errors="coerce")
    panel["continuation_grade"] = base._grade(panel["continuation_delta_bps"])
    return panel


@dataclass(frozen=True)
class CompactBars:
    timestamp_ns: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def _load_symbol_bars(symbol: str) -> CompactBars | None:
    path = base.BARS_ROOT / base._symbol_filename(str(symbol))
    if not path.is_file():
        return None
    bars = pd.read_parquet(path, columns=["high", "low", "close"])
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars.loc[
        ~bars.index.isna() & ~bars.index.duplicated(keep="last")
        & (bars.index >= pd.Timestamp("2026-04-01T00:00:00Z"))
        & (bars.index < pd.Timestamp("2026-09-01T00:00:00Z"))
    ].sort_index()
    return CompactBars(
        timestamp_ns=bars.index.asi8.copy(),
        high=pd.to_numeric(bars["high"], errors="coerce").to_numpy(float),
        low=pd.to_numeric(bars["low"], errors="coerce").to_numpy(float),
        close=pd.to_numeric(bars["close"], errors="coerce").to_numpy(float),
    )


def _bar_path(bars: CompactBars, decision: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    position = int(np.searchsorted(bars.timestamp_ns, decision.value))
    if position >= len(bars.timestamp_ns) or bars.timestamp_ns[position] != decision.value:
        return None
    locations = position + np.arange(base.HORIZON_BARS, dtype=np.int64)
    if locations[-1] >= len(bars.timestamp_ns):
        return None
    arrays = tuple(array[locations] for array in (bars.high, bars.low, bars.close))
    return arrays if all(np.isfinite(item).all() for item in arrays) else None


def _fit(train: pd.DataFrame, spec: str = "lgb_l1_grade"):
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size")
    if spec == "lgb_l1_grade":
        model = base._model(len(train))
        target = train["continuation_grade"]
    elif spec == "lgb_l2_grade":
        model = lgb.LGBMRegressor(
            objective="regression_l2", n_estimators=350, learning_rate=0.03,
            max_depth=4, num_leaves=15, min_child_samples=max(2, int(np.ceil(len(train) * 0.02))),
            subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0, random_state=1729, n_jobs=2, verbosity=-1,
        )
        target = train["continuation_grade"]
    elif spec in {"cat_mae_grade", "cat_rmse_grade"}:
        loss = "MAE" if spec == "cat_mae_grade" else "RMSE"
        model = CatBoostRegressor(
            loss_function=loss, iterations=300, depth=4, learning_rate=0.04,
            l2_leaf_reg=5.0, random_seed=1729, verbose=False, thread_count=2,
            allow_writing_files=False,
        )
        target = train["continuation_grade"]
    elif spec == "lgb_l1_bps":
        model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=350, learning_rate=0.03,
            max_depth=4, num_leaves=15, min_child_samples=max(2, int(np.ceil(len(train) * 0.02))),
            subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0, random_state=1729, n_jobs=2, verbosity=-1,
        )
        target = train["continuation_delta_bps"]
    else:
        raise ValueError(f"unknown continuation model spec: {spec}")
    model.fit(train.loc[:, MODEL_FEATURES], target, sample_weight=weights)
    return model, spec


def _grade_prediction(model, spec: str, values: np.ndarray) -> float:
    raw = float(model.booster_.predict(values.reshape(1, -1))[0]) if hasattr(model, "booster_") else float(model.predict(values.reshape(1, -1))[0])
    if spec == "lgb_l1_bps":
        raw = float(np.interp(raw, [-1.0e9, -100.0, -25.0, 25.0, 100.0, 1.0e9], [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]))
    return float(np.clip(raw, 0.0, 4.0))


def _simulate_entry(group: pd.DataFrame, bars: CompactBars, model, spec: str, params, median: float, config: dict[str, float]) -> dict[str, object] | None:
    first = group.iloc[0]
    decision = pd.Timestamp(first["entry_decision_ts"])
    path = _bar_path(bars, decision)
    if path is None:
        return None
    high, low, close = path
    static = {
        int(row.state_bar_15m): row.loc[list(FIFTEEN_MINUTE_FEATURE_KEYS)].to_numpy(float)
        for _, row in group.iterrows()
    }
    # A candidate receives at most one evaluation per completed bar.  Reuse
    # its dense feature vector instead of allocating a fresh 83-field array
    # for every state/model call.
    feature_buffer = np.empty(len(MODEL_FEATURES), dtype=float)

    def prediction(dynamic: dict[str, float]) -> float | None:
        bar = int(dynamic.pop("state_bar_15m"))
        values = static.get(bar)
        if values is None:
            return None
        state_values = [
            float(first["MC1_expected_bps"]) if key == "MC1_expected_bps" else dynamic[key]
            for key in CONTINUATION_STATE_FEATURE_KEYS
        ]
        feature_buffer[:len(FIFTEEN_MINUTE_FEATURE_KEYS)] = values
        feature_buffer[len(FIFTEEN_MINUTE_FEATURE_KEYS):] = state_values
        # The source contract requires at least 50 finite 15-minute fields;
        # its remaining static fields may be NaN and LightGBM was trained on
        # exactly that missing-value representation.  Dynamic open-trade
        # state itself must always be finite.
        if not np.isfinite(np.asarray(state_values, dtype=float)).all():
            return None
        # The callback is deliberately hot: it runs once per completed 15m bar.
        # Use the fitted booster directly because the preallocated ndarray already
        # follows MODEL_FEATURES exactly.  This avoids sklearn's repeated
        # feature-name validation warning without changing the model input order.
        return _grade_prediction(model, spec, feature_buffer)

    trace = replay_open_long_policy_with_continuation_modulator(
        entry=float(first["entry_price"]), signal_atr=float(first["signal_atr"]),
        highs=high, lows=low, closes=close, params=params, median_atr_fraction=median,
        prediction_for_completed_bar=prediction, **config,
    )
    return {
        "candidate_id": str(first["candidate_id"]), "__symbol__": str(first["__symbol__"]),
        "entry_decision_ts": decision, "baseline_net_bps": float(first["policy_net_bps"]),
        "baseline_gross_bps": float(first["policy_gross_bps"]), "baseline_exit_bar": int(first["policy_exit_bar_15m"]),
        "baseline_exit_reason": str(first["policy_exit_reason"]),
        "c1_gross_bps": trace.terminal_gross_bps, "c1_net_bps": trace.terminal_gross_bps - 100.0,
        "c1_exit_bar": trace.terminal_exit_bar, "c1_exit_reason": trace.terminal_reason,
        "model_calls": len(trace.predictions),
        "mean_ordinal_prediction": float(np.mean(trace.predictions)) if trace.predictions else np.nan,
    }


def run(
    panel: pd.DataFrame,
    params,
    median: float,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    held_months: tuple[pd.Timestamp, ...] | None = None,
    configs: dict[str, dict[str, float]] = CONFIGS,
    model_spec: str = "lgb_l1_grade",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    months = held_months or tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC"))
    bars_cache: dict[str, CompactBars | None] = {}
    for threshold in thresholds:
        scoped = panel.loc[pd.to_numeric(panel["MC1_expected_bps"], errors="coerce").ge(threshold)].copy()
        for held_start in months:
            held_end, train_start = held_start + pd.offsets.MonthBegin(1), held_start - pd.DateOffset(months=2)
            train = scoped.loc[
                scoped["entry_decision_ts"].ge(train_start) & scoped["entry_decision_ts"].lt(held_start)
                & scoped["policy_label_available_ts"].lt(held_start)
            ].copy()
            test = scoped.loc[scoped["entry_decision_ts"].ge(held_start) & scoped["entry_decision_ts"].lt(held_end)].copy()
            if train["candidate_id"].nunique() < 100 or test.empty:
                continue
            model, spec = _fit(train, model_spec)
            arm_rows: dict[str, list[dict[str, object]]] = {arm: [] for arm in configs}
            ordered = test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable")
            # Keep at most one complete symbol history resident.  The former
            # candidate loop re-read each parquet path and let the allocator
            # retain hundreds of histories, which was both slow and memory-heavy.
            for symbol, symbol_rows in ordered.groupby("__symbol__", sort=True):
                token = str(symbol)
                if token not in bars_cache:
                    bars_cache[token] = _load_symbol_bars(token)
                bars = bars_cache[token]
                if bars is None:
                    continue
                for _, group in symbol_rows.groupby("candidate_id", sort=True):
                    for arm, config in configs.items():
                        row = _simulate_entry(group, bars, model, spec, params, median, config)
                        if row is not None:
                            arm_rows[arm].append(row)
            for arm, config in configs.items():
                rows = arm_rows[arm]
                frame = pd.DataFrame([row for row in rows if row is not None])
                if frame.empty:
                    continue
                frame["arm"] = arm
                frame["mc1_threshold_bps"] = threshold
                frame["held_month"] = held_start.strftime("%Y-%m")
                frame["net_delta_bps"] = frame["c1_net_bps"] - frame["baseline_net_bps"]
                details.append(frame)
                metrics.append({
                    "held_month": held_start.strftime("%Y-%m"), "mc1_threshold_bps": threshold, "arm": arm,
                    "entries": len(frame), "baseline_net_bps_per_trade": frame["baseline_net_bps"].mean(),
                    "c1_net_bps_per_trade": frame["c1_net_bps"].mean(), "net_delta_bps_per_trade": frame["net_delta_bps"].mean(),
                    "baseline_total_net_bps": frame["baseline_net_bps"].sum(), "c1_total_net_bps": frame["c1_net_bps"].sum(),
                    "changed_exit_fraction": float((frame["c1_exit_bar"] != frame["baseline_exit_bar"]).mean()),
                    "mean_model_calls": frame["model_calls"].mean(),
                })
    return pd.concat(details, ignore_index=True) if details else pd.DataFrame(), pd.DataFrame(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--parity-root", type=Path, default=PARITY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mc1-threshold", type=float, action="append", choices=THRESHOLDS, help="repeatable; defaults to all thresholds")
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; defaults to Apr--Aug 2026")
    parser.add_argument("--arm", action="append", choices=tuple(CONFIGS), help="repeatable; defaults to all C0/C1 arms")
    parser.add_argument("--model-spec", choices=MODEL_SPECS, default="lgb_l1_grade")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    params, median, policy = base._load_policy()
    panel = _load_panel(args.state_root.resolve(), args.parity_root.resolve())
    thresholds = tuple(args.mc1_threshold) if args.mc1_threshold else THRESHOLDS
    held_months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.held_month) if args.held_month else None
    configs = {name: CONFIGS[name] for name in args.arm} if args.arm else CONFIGS
    details, monthly = run(panel, params, median, thresholds=thresholds, held_months=held_months, configs=configs, model_spec=args.model_spec)
    details.to_parquet(output / "entry_outcomes.parquet", index=False)
    monthly.to_parquet(output / "monthly_metrics.parquet", index=False)
    aggregate = monthly.groupby(["mc1_threshold_bps", "arm"], as_index=False).agg(
        held_months=("held_month", "nunique"), entries=("entries", "sum"),
        baseline_net_bps_per_trade=("baseline_net_bps_per_trade", "mean"),
        c1_net_bps_per_trade=("c1_net_bps_per_trade", "mean"),
        net_delta_bps_per_trade=("net_delta_bps_per_trade", "mean"),
        baseline_total_net_bps=("baseline_total_net_bps", "sum"), c1_total_net_bps=("c1_total_net_bps", "sum"),
        worst_month_c1_bps=("c1_net_bps_per_trade", "min"), worst_month_delta_bps=("net_delta_bps_per_trade", "min"),
        changed_exit_fraction=("changed_exit_fraction", "mean"), mean_model_calls=("mean_model_calls", "mean"),
    ) if len(monthly) else pd.DataFrame()
    aggregate.to_parquet(output / "aggregate_metrics.parquet", index=False)
    manifest = {
        "schema": "p8u-15m-continuation-c1-tightening-ablation-v1",
        "scope": "offline strict-OOS research only; no admission, live, or parent-policy mutation",
        "state_update": "one ordinal prediction after every completed 15-minute bar; action applies only to next bar",
        "entry_scope": "source-guarded dual-MC1 candidates; five source-incompatible parent labels excluded using prior parity receipt",
        "fold": "previous two calendar months with policy labels resolved before held boundary",
        "policy": str(base.POLICY), "policy_sha256": _sha256(base.POLICY), "policy_params": policy["params"],
        "thresholds": thresholds, "held_months": [token.strftime("%Y-%m") for token in (held_months or tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC")))], "arms": configs,
        "C1_contract": "tightening-only: no hard-stop widening and no trailing-floor loosening; Adaptive Exit/smooth protection are unmodified",
        "cost": "100 bps is embedded once in both baseline and C1 net outcomes",
        "model": args.model_spec,
        "stage": "exploratory C1 ablation; never promote without separate portfolio/execution replay and user approval",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
