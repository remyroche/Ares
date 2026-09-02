#!/usr/bin/env python3
"""Direct activation-50 continuation ablations with causal stable selection.

Every held month is trained using only the preceding two full months whose
rich-policy outcomes have resolved.  The action model observes completed
15-minute v2 states and may only activate the existing 50% earlier-trailing
overlay on the following interval.  This is a research-only pipeline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from extreme_price_movements.p8u_continuation_state import replay_open_long_policy_with_continuation_modulator
from extreme_price_movements.p8u_continuation_v2_features import EXTENDED_STATE_FEATURE_KEYS, MANDATORY_STATE_FEATURE_KEYS
from extreme_price_movements.portfolio_policy_replay import compute_replay_metrics, replay_candidates
from scripts import replay_strict_r3_p8u_15m_continuation_portfolio as port
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params as portfolio_params


TARGET_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"
FEATURE_COVERAGE = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_features_20260830_v1/feature_coverage.parquet"
ENTRY_PREDICTIONS = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_retrain_20260830_v3_control/walkforward_predictions.parquet"
STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v3/target_free_continuation_state_parts"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_advantage_ablation_20260830_v1"
SEED = 1729

# State is mandatory; selection must not rediscover the variables that define
# a valid open-trade decision.  These are all available at every C1 decision.
MANDATORY = (*MANDATORY_STATE_FEATURE_KEYS, "MC1_expected_bps")
# Add selected orthogonal legacy microstructure fields to the v2 proposal.
LEGACY_OPTIONAL = (
    "path_efficiency_15m_1h", "directional_consistency_15m_2h", "return_acceleration_15m_1h",
    "latest_impulse_size_atr_15m", "pullback_depth_atr_15m", "rv_15m_1h",
    "compression_percentile_15m_2h", "relative_volume_15m_1h", "volume_acceleration_15m_1h",
    "trend_alignment_15m_vs_1h", "adverse_efficiency_15m", "micro_regime_flip_score_15m",
)
ARM_SPECS = (
    ("A1_all_v2_mean0", "mean", 0.0, "all"),
    ("A2_stable_mean0", "mean", 0.0, "stable"),
    ("A3_stable_q20_0", "q20", 0.0, "stable"),
    ("A4_stable_q20_10", "q20", 10.0, "stable"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_ids(path: Path) -> set[str]:
    frame = pd.read_parquet(path)
    required = {"candidate_id", "floor_bps", "model_spec", "selected__veto_pred_ge_0"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"entry predictions lack {sorted(missing)}")
    selected = frame.loc[
        frame["floor_bps"].eq(30.0) & frame["model_spec"].eq("lgb_huber_bps")
        & frame["selected__veto_pred_ge_0"].fillna(False).astype(bool), "candidate_id"
    ].astype(str)
    if selected.duplicated().any():
        raise AssertionError("frozen entry selection has duplicate identities")
    return set(selected)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    return (1.0 / frame.groupby("candidate_id")["candidate_id"].transform("size")).to_numpy(float)


def _fit(frame: pd.DataFrame, features: tuple[str, ...], kind: str) -> lgb.LGBMRegressor:
    if kind == "mean":
        objective, alpha = "regression_l1", None
    elif kind == "q20":
        objective, alpha = "quantile", 0.20
    else:
        raise ValueError(f"unknown direct-action model {kind}")
    kwargs: dict[str, object] = {"objective": objective, "n_estimators": 350, "learning_rate": 0.03, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": max(8, int(np.ceil(len(frame) * 0.025))), "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 8.0, "random_state": SEED, "n_jobs": 2, "verbosity": -1}
    if alpha is not None:
        kwargs["alpha"] = alpha
    model = lgb.LGBMRegressor(**kwargs)
    model.fit(frame.loc[:, features], frame["activation50_advantage_bps"], sample_weight=_weights(frame))
    return model


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # Candidate-balanced sampling avoids letting 48-state H12 paths dominate.
    candidates = frame["candidate_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected: list[str] = []
    count = 0
    grouped = frame.groupby("candidate_id", sort=False)
    for candidate in candidates:
        size = len(grouped.get_group(candidate))
        if selected and count + size > maximum:
            continue
        selected.append(candidate)
        count += size
        if count >= maximum:
            break
    return frame.loc[frame["candidate_id"].isin(selected)].copy()


def _subspace_stability(train: pd.DataFrame, mandatory: tuple[str, ...], optional: tuple[str, ...], runs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train random subspace models only inside one fold's prior training set."""
    ordered = train.sort_values("state_decision_ts", kind="stable")
    cutoff = ordered["state_decision_ts"].quantile(0.80)
    fit_rows = ordered.loc[ordered["state_decision_ts"].lt(cutoff)].copy()
    valid = ordered.loc[ordered["state_decision_ts"].ge(cutoff)].copy()
    if len(fit_rows) < 500 or len(valid) < 200:
        raise RuntimeError("insufficient temporal split for subspace feature selection")
    fit_rows = _sample(fit_rows, 12_000, seed)
    valid = _sample(valid, 6_000, seed + 1)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for run in range(runs):
        fraction = float(rng.uniform(0.30, 0.60))
        count = max(1, int(round(len(optional) * fraction)))
        chosen = tuple(sorted(rng.choice(np.asarray(optional, dtype=object), size=count, replace=False).tolist()))
        features = (*mandatory, *chosen)
        model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=90, learning_rate=0.05, max_depth=3, num_leaves=7,
            min_child_samples=max(8, int(np.ceil(len(fit_rows) * 0.03))), subsample=0.75, colsample_bytree=0.75,
            reg_lambda=8.0, random_state=seed + run, n_jobs=2, verbosity=-1,
        )
        model.fit(fit_rows.loc[:, features], fit_rows["activation50_advantage_bps"], sample_weight=_weights(fit_rows))
        prediction = model.predict(valid.loc[:, features])
        top = valid.assign(_prediction=prediction).nlargest(max(1, int(np.ceil(len(valid) * 0.20))), "_prediction")
        utility = float(top["activation50_advantage_bps"].mean())
        month_utility = top.assign(_month=pd.to_datetime(top["entry_decision_ts"], utc=True).dt.strftime("%Y-%m")).groupby("_month")["activation50_advantage_bps"].mean()
        temporal = float((month_utility > 0.0).mean()) if len(month_utility) else 0.0
        gains = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(min(12, len(features))).index
        for feature in gains:
            if feature in optional:
                rows.append({"feature": feature, "run": run, "utility": utility, "temporal_positive_fraction": temporal, "selected_topk": True})
        for feature in chosen:
            if feature not in gains:
                rows.append({"feature": feature, "run": run, "utility": utility, "temporal_positive_fraction": temporal, "selected_topk": False})
    runs_frame = pd.DataFrame(rows)
    if runs_frame.empty:
        raise RuntimeError("random-subspace selector emitted no optional-feature records")
    utility_median = float(runs_frame.drop_duplicates("run")["utility"].median())
    aggregate = runs_frame.groupby("feature", as_index=False).agg(
        selection_frequency=("selected_topk", "mean"),
        mean_subspace_utility=("utility", "mean"),
        temporal_stability=("temporal_positive_fraction", "mean"),
        inclusion_runs=("run", "nunique"),
    )
    aggregate["mean_performance_of_strong_subspaces"] = aggregate["mean_subspace_utility"].map(lambda x: 1.0 / (1.0 + np.exp(-(x - utility_median) / 50.0)))
    aggregate["stability_score"] = aggregate["selection_frequency"] * aggregate["mean_performance_of_strong_subspaces"] * aggregate["temporal_stability"]
    return aggregate.sort_values(["stability_score", "selection_frequency"], ascending=False, kind="stable"), runs_frame


def _impute(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return values.fillna(values.median(numeric_only=True)).fillna(0.0).to_numpy(float)


def _residual_cmi(train: pd.DataFrame, selected: tuple[str, ...], candidate: str) -> float:
    sample = _sample(train, 10_000, SEED + len(selected))
    y = pd.to_numeric(sample["activation50_advantage_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    x = _impute(sample, (candidate,))[:, 0]
    if selected:
        context = _impute(sample, selected)
        y = y - Ridge(alpha=4.0).fit(context, y).predict(context)
        x = x - Ridge(alpha=4.0).fit(context, x).predict(context)
    if np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return 0.0
    return float(mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False, n_neighbors=5, random_state=SEED)[0])


EPS = 1.0e-12


def _validation_utility(train: pd.DataFrame, features: tuple[str, ...]) -> float:
    ordered = train.sort_values("state_decision_ts", kind="stable")
    cutoff = ordered["state_decision_ts"].quantile(0.80)
    fit_rows, valid = ordered.loc[ordered.state_decision_ts.lt(cutoff)], ordered.loc[ordered.state_decision_ts.ge(cutoff)]
    if len(fit_rows) < 300 or valid.empty:
        return float("-inf")
    model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=80, learning_rate=0.05, max_depth=3, num_leaves=7,
                              min_child_samples=max(8, int(np.ceil(len(fit_rows) * .03))), reg_lambda=8.0, random_state=SEED, n_jobs=2, verbosity=-1)
    model.fit(fit_rows.loc[:, features], fit_rows["activation50_advantage_bps"], sample_weight=_weights(fit_rows))
    top = valid.assign(_prediction=model.predict(valid.loc[:, features])).nlargest(max(1, int(np.ceil(len(valid) * .20))), "_prediction")
    return float(top["activation50_advantage_bps"].mean())


def _select_features(train: pd.DataFrame, optional: tuple[str, ...], runs: int) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    mandatory = tuple(name for name in MANDATORY if name in train.columns)
    if len(mandatory) != len(MANDATORY):
        raise AssertionError("mandatory continuation state feature is missing")
    ranking, run_rows = _subspace_stability(train, mandatory, optional, runs, SEED)
    selected = list(mandatory)
    trace: list[dict[str, object]] = []
    base_utility = _validation_utility(train, tuple(selected))
    stale = 0
    for _, row in ranking.iterrows():
        feature = str(row["feature"])
        cmi = _residual_cmi(train, tuple(selected), feature)
        candidate = tuple((*selected, feature))
        utility = _validation_utility(train, candidate)
        delta = utility - base_utility
        accept = bool(cmi >= 0.002 and delta >= -2.0)
        trace.append({**row.to_dict(), "conditional_mi": cmi, "incremental_oof_policy_utility": delta, "accepted": accept, "feature_count_after": len(candidate) if accept else len(selected)})
        if accept:
            selected.append(feature)
            base_utility = utility
            stale = 0
        else:
            stale += 1
        if len(selected) >= 45 or (len(selected) >= 30 and stale >= 6):
            break
    if len(selected) < 30:
        for feature in ranking["feature"]:
            if feature not in selected:
                selected.append(str(feature))
            if len(selected) >= 30:
                break
    return tuple(selected), pd.DataFrame(trace), ranking


def _simulate(group: pd.DataFrame, bars: c1.CompactBars, model, features: tuple[str, ...], threshold: float, params, median: float) -> dict[str, object] | None:
    first = group.iloc[0]
    path = c1._bar_path(bars, pd.Timestamp(first["entry_decision_ts"]))
    if path is None:
        return None
    high, low, close = path
    static = {int(row.state_bar_15m): row.loc[list(features)].to_numpy(float) for _, row in group.iterrows()}
    buffer = np.empty(len(features), dtype=float)
    calls = 0
    action_calls = 0

    def callback(dynamic: dict[str, float]) -> float | None:
        nonlocal calls, action_calls
        bar = int(dynamic.pop("state_bar_15m"))
        values = static.get(bar)
        if values is None:
            return None
        calls += 1
        buffer[:] = values
        prediction = float(model.booster_.predict(buffer.reshape(1, -1))[0])
        action = prediction >= threshold
        action_calls += int(action)
        return 0.0 if action else 2.0

    trace = replay_open_long_policy_with_continuation_modulator(
        entry=float(first["entry_price"]), signal_atr=float(first["signal_atr"]), highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median, prediction_for_completed_bar=callback,
        sl_tighten=0.0, giveback_tighten=0.0, activation_earlier=0.50,
    )
    return {
        "candidate_id": str(first.candidate_id), "__symbol__": str(first.__symbol__), "entry_decision_ts": pd.Timestamp(first.entry_decision_ts),
        "baseline_net_bps": float(first.policy_net_bps), "baseline_gross_bps": float(first.policy_gross_bps),
        "baseline_exit_bar": int(first.policy_exit_bar_15m), "baseline_exit_reason": str(first.policy_exit_reason),
        "c1_gross_bps": float(trace.terminal_gross_bps), "c1_net_bps": float(trace.terminal_gross_bps - 100.0),
        "c1_exit_bar": int(trace.terminal_exit_bar), "c1_exit_reason": str(trace.terminal_reason),
        "model_calls": calls, "action_calls": action_calls,
    }


def _parent_rows(frame: pd.DataFrame) -> pd.DataFrame:
    first = frame.sort_values(["candidate_id", "state_bar_15m"], kind="stable").groupby("candidate_id", as_index=False).first()
    return pd.DataFrame({
        "candidate_id": first.candidate_id.astype(str), "__symbol__": first.__symbol__.astype(str), "entry_decision_ts": pd.to_datetime(first.entry_decision_ts, utc=True),
        "baseline_net_bps": first.policy_net_bps, "baseline_gross_bps": first.policy_gross_bps,
        "baseline_exit_bar": first.policy_exit_bar_15m.astype(int), "baseline_exit_reason": first.policy_exit_reason.astype(str),
        "c1_gross_bps": first.policy_gross_bps, "c1_net_bps": first.policy_net_bps,
        "c1_exit_bar": first.policy_exit_bar_15m.astype(int), "c1_exit_reason": first.policy_exit_reason.astype(str),
        "model_calls": 0, "action_calls": 0,
    })


def _replay_portfolio(rows: pd.DataFrame, arm: str, output: Path) -> dict[str, object]:
    prices = port._entry_prices(STATE_ROOT)
    priorities = port._bcf_priority()
    tagged = rows.copy(); tagged["arm"] = arm; tagged["mc1_threshold_bps"] = 30.0
    candidates = port._candidate_table(tagged, prices, priorities)
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    decisions = port._attach_ids(decisions, candidates)
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    port._period_metrics(accepted, "month").assign(arm=arm).to_parquet(output / f"{arm}_monthly.parquet", index=False)
    returns = pd.to_numeric(candidates.iloc[pd.to_numeric(accepted.candidate_index, errors="raise").astype(int).to_numpy()].net_return, errors="raise") * 10_000.0
    return {"arm": arm, "routed_candidates": len(candidates), "portfolio_accepted": len(accepted),
            "policy_net_bps_per_trade": float(returns.mean()) if len(returns) else np.nan, "total_policy_net_bps": float(returns.sum()),
            **compute_replay_metrics(candidates, decisions, equity, params=params)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-panel", type=Path, default=TARGET_PANEL)
    parser.add_argument("--feature-coverage", type=Path, default=FEATURE_COVERAGE)
    parser.add_argument("--entry-predictions", type=Path, default=ENTRY_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subspace-runs", type=int, default=200)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Apr--Aug 2026")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    panel_path = args.target_panel.resolve()
    panel = pd.read_parquet(panel_path)
    panel["candidate_id"] = panel.candidate_id.astype(str)
    for time in ("entry_decision_ts", "state_decision_ts", "policy_label_available_ts"):
        panel[time] = pd.to_datetime(panel[time], utc=True, errors="raise")
    coverage = pd.read_parquet(args.feature_coverage.resolve())
    available_extended = tuple(coverage.loc[coverage.available.fillna(False), "feature"].astype(str))
    optional = tuple(dict.fromkeys([*(name for name in available_extended if name not in MANDATORY), *(name for name in LEGACY_OPTIONAL if name in panel.columns)]))
    candidate_features = (*MANDATORY, *optional)
    if not (70 <= len(candidate_features) <= 90):
        raise AssertionError(f"candidate feature pool must remain compact 70--90, got {len(candidate_features)}")
    if set(candidate_features).difference(panel.columns):
        raise AssertionError("candidate feature pool has absent column")
    selected_ids = _selection_ids(args.entry_predictions.resolve())
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-04-01", "2026-08-01", freq="MS", tz="UTC"))
    params, median, policy = c1.base._load_policy()
    outcomes: dict[str, list[pd.DataFrame]] = defaultdict(list)
    selection_rows: list[pd.DataFrame] = []
    stability_rows: list[pd.DataFrame] = []
    subspace_rows: list[pd.DataFrame] = []
    bars_cache: dict[str, c1.CompactBars | None] = {}
    for held in held_months:
        held_end, train_start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=2)
        train = panel.loc[
            pd.to_numeric(panel.MC1_expected_bps, errors="coerce").ge(30.0)
            & panel.entry_decision_ts.ge(train_start) & panel.entry_decision_ts.lt(held)
            & panel.policy_label_available_ts.lt(held)
        ].copy()
        required = {(held - pd.DateOffset(months=1)).strftime("%Y-%m"), (held - pd.DateOffset(months=2)).strftime("%Y-%m")}
        observed = set(train.entry_decision_ts.dt.strftime("%Y-%m"))
        test = panel.loc[
            panel.entry_decision_ts.ge(held) & panel.entry_decision_ts.lt(held_end)
            & panel.candidate_id.isin(selected_ids)
        ].copy()
        if not required.issubset(observed) or train.candidate_id.nunique() < 100 or test.empty:
            continue
        stable, trace, ranking = _select_features(train, optional, args.subspace_runs)
        if not (30 <= len(stable) <= 45):
            raise AssertionError(f"stable feature process produced {len(stable)}, outside 30--45")
        selection_rows.append(pd.DataFrame({"held_month": held.strftime("%Y-%m"), "feature": stable, "position": np.arange(len(stable)), "mandatory": [name in MANDATORY for name in stable]}))
        trace["held_month"] = held.strftime("%Y-%m"); stability_rows.append(trace)
        ranking["held_month"] = held.strftime("%Y-%m"); subspace_rows.append(ranking)
        all_features = tuple(candidate_features)
        models = {"A1_all_v2_mean0": _fit(train, all_features, "mean"), "A2_stable_mean0": _fit(train, stable, "mean"),
                  "A3_stable_q20_0": _fit(train, stable, "q20"), "A4_stable_q20_10": _fit(train, stable, "q20")}
        outcomes["C0_parent"].append(_parent_rows(test).assign(held_month=held.strftime("%Y-%m")))
        ordered = test.sort_values(["__symbol__", "candidate_id", "state_bar_15m"], kind="stable")
        for arm, kind, threshold, feature_mode in ARM_SPECS:
            feature_set = all_features if feature_mode == "all" else stable
            rows: list[dict[str, object]] = []
            for symbol, symbol_rows in ordered.groupby("__symbol__", sort=True):
                if str(symbol) not in bars_cache:
                    bars_cache[str(symbol)] = c1._load_symbol_bars(str(symbol))
                bars = bars_cache[str(symbol)]
                if bars is None:
                    continue
                for _, group in symbol_rows.groupby("candidate_id", sort=True):
                    row = _simulate(group, bars, models[arm], feature_set, threshold, params, median)
                    if row is not None:
                        rows.append(row)
            result = pd.DataFrame(rows)
            if result.empty:
                continue
            result["held_month"] = held.strftime("%Y-%m")
            outcomes[arm].append(result)
    if "C0_parent" not in outcomes or any(arm not in outcomes for arm, *_ in ARM_SPECS):
        raise RuntimeError("one or more direct-action arms lack strict-OOS outcomes")
    output.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, object]] = []
    for arm, frames in outcomes.items():
        frame = pd.concat(frames, ignore_index=True)
        if frame.candidate_id.duplicated().any():
            raise AssertionError(f"{arm}: duplicate candidate OOS outcome")
        frame.to_parquet(output / f"{arm}_entry_outcomes.parquet", index=False, compression="zstd")
        summaries.append(_replay_portfolio(frame, arm, output))
    summary = pd.DataFrame(summaries)
    control = summary.loc[summary.arm.eq("C0_parent")]
    if len(control) != 1:
        raise AssertionError("exactly one parent control is required")
    for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "compounded_return", "sortino", "max_drawdown", "worst_week"):
        summary[f"delta_vs_C0_{metric}"] = summary[metric] - control.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    pd.concat(selection_rows, ignore_index=True).to_parquet(output / "stable_selected_features.parquet", index=False)
    pd.concat(stability_rows, ignore_index=True).to_parquet(output / "selection_trace.parquet", index=False)
    pd.concat(subspace_rows, ignore_index=True).to_parquet(output / "subspace_stability.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-v2-direct-activation-advantage-ablation-v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation",
        "target": "direct activation-50 net advantage over parent rich policy; action begins only on next 15m interval",
        "fold": "two full preceding calendar months; all parent action labels resolved before held boundary",
        "state": "one model decision after each completed 15m state; static feature row is the exact state-bar row",
        "candidate_pool": len(candidate_features), "mandatory": list(MANDATORY), "optional": list(optional),
        "selection": {"random_subspace_runs": args.subspace_runs, "feature_fraction": "30--60% optional", "top_k_importance": 12,
                      "score": "selection_frequency * performance_of_containing_subspaces * temporal_stability", "conditional_mi": True,
                      "target_features": "30--45 total"},
        "arms": [arm for arm, *_ in ARM_SPECS], "policy": policy["params"],
        "target_panel": str(panel_path), "target_panel_sha256": _sha256(panel_path),
        "entry_predictions": str(args.entry_predictions.resolve()), "entry_predictions_sha256": _sha256(args.entry_predictions.resolve()),
        "cost": "100 bps embedded once in baseline and each action-policy outcome",
        "portfolio": asdict(portfolio_params()),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
