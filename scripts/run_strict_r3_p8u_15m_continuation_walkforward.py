#!/usr/bin/env python3
"""Stateful P8U continuation-value head: predictive stage only.

The script is intentionally a two-stage research workflow.  This first stage
creates one causal state after every completed 15-minute bar of an open trade,
then tests whether an ordinal LGBM predicts the value of continuing the frozen
rich parent policy instead of exiting at that state.  It does *not* alter SL,
trailing activation, giveback, the frozen policy, or live execution.  Policy
multiplier backtests belong to a separately named second stage and may only be
run after these out-of-sample predictive diagnostics pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, compute_15m_features
from extreme_price_movements.p8u_continuation_state import CONTINUATION_STATE_FEATURE_KEYS, trace_open_long_policy_states
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams
from scripts.run_strict_r3_rich_policy_hpo import HORIZON_BARS, _hourly_signal_atr, _symbol_filename


FEATURE_PANEL_DEFAULT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v2/target_free_15m_features.parquet"
DUAL = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4/dual_predictions.parquet"
BARS_ROOT = ROOT / "15m_ohlcv_perp"
POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_walkforward_20260830_v1"
SEED = 1729
CONTINUATION_BINS = [-np.inf, -100.0, -25.0, 25.0, 100.0, np.inf]
THRESHOLDS = (50.0, 40.0, 30.0, 20.0)
MODEL_FEATURES = (*FIFTEEN_MINUTE_FEATURE_KEYS, *CONTINUATION_STATE_FEATURE_KEYS)


def _utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy() -> tuple[RichPolicyParams, float, dict[str, object]]:
    payload = json.loads(POLICY.read_text(encoding="utf-8"))
    params = RichPolicyParams.from_mapping(dict(payload["params"]))
    median = float(payload["median_atr_fraction_fitted_on_complete_2024_development"])
    if not (np.isfinite(median) and median > 0.0 and bool(params.smooth_capital_protection_enabled)):
        raise AssertionError("continuation parent policy is not the sealed rich smooth-policy contract")
    if not np.isclose(float(payload["cost_bps"]), 100.0):
        raise AssertionError("continuation parent policy must retain the exact 100-bps total cost")
    return params, median, payload


def _window_status(bars: pd.DataFrame, decision: pd.Timestamp) -> tuple[str, pd.DataFrame]:
    """Return the same observed/economic source guard as the entry panel."""
    context = bars.loc[(bars.index >= decision - pd.Timedelta(hours=30)) & (bars.index < decision)].copy()
    if len(context) < 100:
        return "insufficient_25h_bars", context
    recent = context.tail(100)
    if "exchange_observed" in recent and not recent["exchange_observed"].fillna(False).astype(bool).all():
        return "exchange_observation_incomplete", context
    ohlc = recent[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    array = ohlc.to_numpy(float)
    if not (np.isfinite(array).all() and (array > 0.0).all()):
        return "nonpositive_or_missing_ohlc", context
    changed = int(ohlc["close"].nunique(dropna=True)) >= 3 or bool((ohlc["high"] > ohlc["low"]).sum() >= 4)
    if not changed:
        return "stale_flat_25h_window", context
    if not bool((pd.to_numeric(recent["volume"], errors="coerce") > 0.0).sum() >= 4):
        return "no_traded_15m_evidence", context
    return "complete", context


def _entry_source(feature_panel: Path, source_floor: float, min_finite: int) -> pd.DataFrame:
    panel = pd.read_parquet(feature_panel)
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True)
    required = {"candidate_id", "__decision_ts__", "__symbol__", "dual_mc1_min_bps", "finite_15m_feature_count"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"feature panel lacks {missing}")
    source = panel.loc[
        pd.to_numeric(panel["dual_mc1_min_bps"], errors="coerce").ge(source_floor)
        & pd.to_numeric(panel["finite_15m_feature_count"], errors="coerce").ge(min_finite)
    ].copy()
    if source["candidate_id"].duplicated().any():
        raise AssertionError("entry feature panel has duplicate candidate identities")
    return source.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable")


def _read_state_panel(path: Path) -> pd.DataFrame:
    """Read an immutable single parquet or symbol-partitioned state panel."""
    if path.is_file():
        return pd.read_parquet(path)
    parts = sorted(path.glob("symbol=*/states.parquet"))
    if not parts:
        raise FileNotFoundError(f"no continuation-state parquet parts under {path}")
    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)


def materialize_target_free_states(
    source: pd.DataFrame,
    *,
    params: RichPolicyParams,
    median_atr_fraction: float,
    min_finite_features: int,
    state_parts_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Advance each parent-policy state before any outcome labels are joined."""
    states: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    # Cache compact numeric vectors, rather than 70-key dictionaries, and
    # retain only states that can actually satisfy the declared feature
    # contract.  Unavailable states remain fully counted in the audit below.
    state_unavailable = Counter()
    for symbol, group in source.groupby("__symbol__", sort=True):
        # A state timestamp cannot be shared across symbols.  Resetting this
        # compact cache at the symbol boundary preserves deterministic feature
        # values while bounding memory for a full multi-month materialisation.
        feature_cache: dict[pd.Timestamp, tuple[str, np.ndarray]] = {}
        symbol_states: list[dict[str, object]] = []
        path = BARS_ROOT / _symbol_filename(str(symbol))
        if not path.exists():
            audit.append({"__symbol__": symbol, "entries": len(group), "traced_entries": 0, "state_rows": 0, "reason": "missing_15m_symbol_source"})
            continue
        bars = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume", "exchange_observed"])
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars.loc[~bars.index.isna() & ~bars.index.duplicated(keep="last")].sort_index()
        bars = bars.apply(pd.to_numeric, errors="coerce")
        atr = _hourly_signal_atr(bars)
        traced = 0
        state_count = 0
        for _, row in group.iterrows():
            decision = _utc(row["__decision_ts__"])
            location = bars.index.get_indexer([decision])[0]
            if location < 0 or location + HORIZON_BARS > len(bars):
                continue
            locations = location + np.arange(HORIZON_BARS, dtype=np.int64)
            fallback_entry = float(bars["open"].iloc[location])
            fallback_atr = float(atr.reindex([decision]).iloc[0])
            # The continuation state must be generated from the same retained
            # 15-minute source that feeds its live/replay feature contract.
            # Do not mix in a separately materialised historical ATR panel:
            # its row meanings are not identical to this parent-label source.
            entry = fallback_entry
            signal_atr = fallback_atr
            high = bars["high"].iloc[locations].to_numpy(float)
            low = bars["low"].iloc[locations].to_numpy(float)
            close = bars["close"].iloc[locations].to_numpy(float)
            if not (np.isfinite([entry, signal_atr]).all() and entry > 0.0 and signal_atr > 0.0 and np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()):
                continue
            trace = trace_open_long_policy_states(
                candidate_id=str(row["candidate_id"]),
                entry_decision_ts=decision,
                entry=entry,
                signal_atr=signal_atr,
                highs=high,
                lows=low,
                closes=close,
                params=params,
                median_atr_fraction=median_atr_fraction,
                mc1_expected_bps=float(row["dual_mc1_min_bps"]),
            )
            traced += 1
            audit.append({
                "candidate_id": str(row["candidate_id"]),
                "__symbol__": symbol,
                "entry_decision_ts": decision,
                "traced_entries": 1,
                "state_rows": len(trace.states),
                "simulated_parent_exit_bar": trace.terminal_exit_bar,
                "simulated_parent_gross_bps": trace.terminal_gross_bps,
                "simulated_parent_exit_reason": trace.terminal_reason,
                "entry_atr_provenance": "causal_hourly_wilder14_from_retained_15m",
                "reason": "ok",
            })
            for state in trace.states:
                state_ts = _utc(state["state_decision_ts"])
                key = state_ts
                if key not in feature_cache:
                    status, context = _window_status(bars, state_ts)
                    values = compute_15m_features(context, state_ts, side="long") if status == "complete" else {key: np.nan for key in FIFTEEN_MINUTE_FEATURE_KEYS}
                    feature_cache[key] = (status, np.asarray([values[name] for name in FIFTEEN_MINUTE_FEATURE_KEYS], dtype=float))
                status, value_vector = feature_cache[key]
                finite = int(np.isfinite(value_vector).sum())
                if finite < min_finite_features:
                    reason = status if status != "complete" else "fewer_than_min_finite_features"
                    state_unavailable[(str(symbol), reason)] += 1
                    continue
                symbol_states.append({
                    **state,
                    "__symbol__": symbol,
                    "state_feature_source_status": status,
                    "finite_15m_feature_count": finite,
                    **dict(zip(FIFTEEN_MINUTE_FEATURE_KEYS, value_vector, strict=True)),
                })
                state_count += 1
        if state_parts_root is not None and symbol_states:
            destination = state_parts_root / f"symbol={str(symbol).replace('/', '_')}" / "states.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(symbol_states).to_parquet(destination, index=False, compression="zstd")
        elif state_parts_root is None:
            states.extend(symbol_states)
        if not any(item.get("__symbol__") == symbol and item.get("reason") == "ok" for item in audit):
            audit.append({"__symbol__": symbol, "entries": len(group), "traced_entries": traced, "state_rows": state_count, "reason": "no_complete_parent_path"})
    audit.extend(
        {
            "__symbol__": symbol,
            "entries": 0,
            "traced_entries": 0,
            "state_rows": int(count),
            "reason": reason,
            "audit_kind": "unavailable_state",
        }
        for (symbol, reason), count in sorted(state_unavailable.items())
    )
    return pd.DataFrame(states), pd.DataFrame(audit)


def _labels() -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_label_available_ts",
    ]
    labels = pd.read_parquet(DUAL, columns=columns)
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    if labels.candidate_id.duplicated().any():
        raise AssertionError("dual label source has duplicate candidate identities")
    return labels


def _model(train_rows: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=350, learning_rate=0.03,
        max_depth=4, num_leaves=15,
        min_child_samples=max(2, math.ceil(train_rows * 0.02)),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def _grade(values: pd.Series) -> pd.Series:
    return pd.cut(values, bins=CONTINUATION_BINS, labels=False, include_lowest=True).astype(float)


def run_predictive_walkforward(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = states.copy()
    states["entry_decision_ts"] = pd.to_datetime(states["entry_decision_ts"], utc=True)
    states["policy_label_available_ts"] = pd.to_datetime(states["policy_label_available_ts"], utc=True)
    states = states.loc[
        states["policy_path_valid"].fillna(False)
        & states["policy_net_bps"].notna()
        & states["finite_15m_feature_count"].ge(50)
    ].copy()
    # Both exit-now PnL and policy outcome include the same single 100-bps
    # round-trip cost, so the continuation target never double-counts it.
    states["continuation_delta_bps"] = pd.to_numeric(states["policy_net_bps"], errors="coerce") - pd.to_numeric(states["current_PnL"], errors="coerce")
    states["continuation_grade"] = _grade(states["continuation_delta_bps"])
    months = pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    importance: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        scoped = states.loc[pd.to_numeric(states["MC1_expected_bps"], errors="coerce").ge(threshold)].copy()
        for held_start in months:
            held_end = held_start + pd.offsets.MonthBegin(1)
            train_start = held_start - pd.DateOffset(months=2)
            train = scoped.loc[
                scoped["entry_decision_ts"].ge(train_start)
                & scoped["entry_decision_ts"].lt(held_start)
                & scoped["policy_label_available_ts"].lt(held_start)
            ].copy()
            test = scoped.loc[scoped["entry_decision_ts"].ge(held_start) & scoped["entry_decision_ts"].lt(held_end)].copy()
            if train["candidate_id"].nunique() < 100 or test.empty:
                continue
            # Equalise entry influence: a long-lived trade may furnish more
            # state observations, but may not dominate the fit merely because
            # it survived longer.
            weight = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size")
            model = _model(len(train))
            model.fit(train.loc[:, MODEL_FEATURES], train["continuation_grade"], sample_weight=weight)
            test["continuation_ordinal_prediction"] = model.predict(test.loc[:, MODEL_FEATURES])
            corr = test[["continuation_ordinal_prediction", "continuation_delta_bps"]].corr(method="spearman").iloc[0, 1]
            test["prediction_quintile"] = pd.qcut(test["continuation_ordinal_prediction"].rank(method="first"), 5, labels=False, duplicates="drop")
            bucket = test.groupby("prediction_quintile", observed=True)["continuation_delta_bps"].mean()
            metrics.append({
                "held_month": held_start.strftime("%Y-%m"), "mc1_threshold_bps": threshold,
                "train_entries": int(train.candidate_id.nunique()), "train_states": int(len(train)),
                "test_entries": int(test.candidate_id.nunique()), "test_states": int(len(test)),
                "spearman_prediction_delta": float(corr) if np.isfinite(corr) else np.nan,
                "bottom_quintile_delta_bps": float(bucket.iloc[0]) if len(bucket) else np.nan,
                "top_quintile_delta_bps": float(bucket.iloc[-1]) if len(bucket) else np.nan,
                "quintile_spread_bps": float(bucket.iloc[-1] - bucket.iloc[0]) if len(bucket) > 1 else np.nan,
            })
            for feature, value in zip(MODEL_FEATURES, model.feature_importances_, strict=True):
                importance.append({"held_month": held_start.strftime("%Y-%m"), "mc1_threshold_bps": threshold, "feature": feature, "importance": int(value)})
            predictions.append(test.assign(held_month=held_start.strftime("%Y-%m"), mc1_threshold_bps=threshold))
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(metrics), pd.DataFrame(importance),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-panel", type=Path, default=FEATURE_PANEL_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-floor-bps", type=float, default=20.0)
    parser.add_argument("--min-finite-features", type=int, default=50)
    parser.add_argument("--state-panel", type=Path, help="reuse an immutable target-free continuation-state panel")
    parser.add_argument("--parent-audit", type=Path, help="required parity audit when reusing --state-panel; defaults to its sibling audit")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    params, median, payload = _load_policy()
    if args.state_panel:
        state_panel_source = args.state_panel.resolve()
        parent_audit_path = args.parent_audit.resolve() if args.parent_audit else args.state_panel.resolve().with_name("parent_policy_state_audit.parquet")
        if not parent_audit_path.is_file():
            raise FileNotFoundError("reused continuation state panel requires its original parent-policy audit")
        parent_audit = pd.read_parquet(parent_audit_path)
    else:
        entries = _entry_source(args.feature_panel.resolve(), args.source_floor_bps, args.min_finite_features)
        state_panel_source = output / "target_free_continuation_state_parts"
        _, parent_audit = materialize_target_free_states(
            entries, params=params, median_atr_fraction=median, min_finite_features=args.min_finite_features,
            state_parts_root=state_panel_source,
        )
    parent_audit.to_parquet(output / "parent_policy_state_audit.parquet", index=False)
    provenance = (
        parent_audit.loc[parent_audit.get("candidate_id", pd.Series(index=parent_audit.index, dtype=object)).notna()]
        .groupby("entry_atr_provenance", dropna=False, as_index=False)
        .agg(entries=("candidate_id", "nunique"))
        if "entry_atr_provenance" in parent_audit
        else pd.DataFrame(columns=["entry_atr_provenance", "entries"])
    )
    provenance.to_parquet(output / "parent_entry_atr_provenance_coverage.parquet", index=False)
    manifest = {
        "schema": "p8u-15m-stateful-continuation-predictive-v1",
        "scope": "offline predictive research only; no exit-policy modulation or live mutation",
        "entry_source_floor_dual_mc1_bps": args.source_floor_bps,
        "continuation_state_update": "each completed 15m bar; state/thresholds apply only to following interval",
        "parent_policy": str(POLICY), "parent_policy_sha256": _sha256(POLICY),
        "parent_policy_params": payload["params"], "parent_policy_median_atr_fraction": median,
        "parent_entry_atr": {
            "source": "causal hourly Wilder-14 reconstruction from retained 15m bars",
            "contract": "all inputs are complete before the decision open; the decision hour itself is shifted out",
            "cross_substrate_metadata": "explicitly excluded because it is not source-identical to this parent-label substrate",
            "provenance_coverage": "parent_entry_atr_provenance_coverage.parquet",
        },
        "features": list(MODEL_FEATURES),
        "target": "policy_net_bps - current_PnL; both include the same exact 100-bps trade cost once",
        "target_bins_bps": CONTINUATION_BINS,
        "fold": "previous two calendar months; resolved policy label available before held-month boundary",
        "model": {"family": "LightGBM L1 ordinal regression", "max_depth": 4, "min_leaf_fraction": 0.02, "seed": SEED, "equal_entry_weighting": True},
        "source_hashes": {"dual_predictions": _sha256(DUAL), "entry_feature_panel": _sha256(args.feature_panel.resolve())},
        "target_free_state_panel": str(state_panel_source),
        "stage": "predictive_only; C1 policy multipliers prohibited until OOS predictive diagnostics are reviewed",
        "parent_label_serialization_tolerance_bps": 0.05,
        "parent_label_incompatibility": "any parent-policy trace mismatch is persisted and excluded from supervised fitting/evaluation; it never changes source state by realized outcome",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.materialize_only:
        return
    # Outcome labels join only after the complete state artifact above exists.
    state_panel = _read_state_panel(state_panel_source)
    labels = _labels()
    parity_source = parent_audit.loc[parent_audit.get("candidate_id", pd.Series(index=parent_audit.index, dtype=object)).notna()].copy()
    audited = parity_source.merge(labels, on="candidate_id", how="left", validate="one_to_one") if not parity_source.empty else parity_source
    incompatible_ids: set[str] = set()
    if "simulated_parent_exit_bar" in audited:
        valid = audited["policy_path_valid"].fillna(False)
        mismatch = valid & (
            pd.to_numeric(audited["simulated_parent_exit_bar"], errors="coerce").ne(pd.to_numeric(audited["policy_exit_bar_15m"], errors="coerce"))
            # Existing frozen labels serialise a small subset of gross outcomes
            # through float32.  Exit bar/reason remain exact; 0.05 bps is only
            # an artifact-serialization tolerance (observed maximum 0.0253).
            | ~np.isclose(pd.to_numeric(audited["simulated_parent_gross_bps"], errors="coerce"), pd.to_numeric(audited["policy_gross_bps"], errors="coerce"), rtol=0.0, atol=0.05, equal_nan=True)
        )
        audited["parent_policy_state_compatible"] = ~mismatch
        incompatible_ids = set(audited.loc[mismatch, "candidate_id"].astype(str))
    audited.to_parquet(output / "parent_policy_parity_audit.parquet", index=False)
    panel = state_panel.merge(labels, on="candidate_id", how="left", validate="many_to_one")
    # A mismatch is a label-substrate integrity failure, not a learnable
    # outcome.  Preserve it in the audit while excluding its open-trade states
    # from every fit and held-period metric.
    if incompatible_ids:
        panel = panel.loc[~panel["candidate_id"].astype(str).isin(incompatible_ids)].copy()
        pd.DataFrame({"candidate_id": sorted(incompatible_ids)}).to_parquet(
            output / "parent_policy_state_incompatible_candidates.parquet", index=False
        )
    predictions, monthly, importance = run_predictive_walkforward(panel)
    predictions.to_parquet(output / "walkforward_predictions.parquet", index=False)
    monthly.to_parquet(output / "monthly_predictive_metrics.parquet", index=False)
    importance.to_parquet(output / "feature_importance.parquet", index=False)
    aggregate = monthly.groupby("mc1_threshold_bps", as_index=False).agg(
        held_months=("held_month", "nunique"), test_entries=("test_entries", "sum"), test_states=("test_states", "sum"),
        mean_spearman=("spearman_prediction_delta", "mean"), worst_spearman=("spearman_prediction_delta", "min"),
        mean_quintile_spread_bps=("quintile_spread_bps", "mean"), worst_quintile_spread_bps=("quintile_spread_bps", "min"),
    ) if len(monthly) else pd.DataFrame()
    aggregate.to_parquet(output / "aggregate_predictive_metrics.parquet", index=False)


if __name__ == "__main__":
    main()
