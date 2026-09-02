#!/usr/bin/env python3
"""Causal remaining-trade regime ablation for the rich H4 parent exit policy.

This replaces a one-interval actuator tweak with coherent *persistent* regimes
selected at a completed 15-minute state and held until exit:

* Parent — unchanged rich policy;
* Protect — earlier trailing activation and tighter giveback;
* TrendRide — later activation and wider giveback.

Every counterfactual retains the actual parent path up to the selected state.
From the following minute onward it applies the regime vector for the entire
remaining H12 path.  The study is offline-only and never imports live modules,
changes admission/portfolio logic, refits Geometry/K9, or calls an exchange.

It runs the requested hierarchy:

1. sampled-state perfect-regime oracle;
2. train-only coarse-state lookup;
3. strict-prior conditional ML selector using mean and quantile-LCB values;
4. a frozen 2026 confirmation of the 2025-selected feasible arm.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from scripts import run_causal_sr_h4_actuator_counterfactual_ablation as base
except ModuleNotFoundError:
    import run_causal_sr_h4_actuator_counterfactual_ablation as base
from extreme_price_movements.exact_1m_h4_overlay_research import replay_exact_1m_gradual_h4_overlay


H12 = pd.Timedelta(hours=12)
REGIMES: dict[str, dict[str, float]] = {
    # A coherent protection vector: activation happens earlier and any
    # trailing giveback is smaller.  The parent hard loss / smooth lock remain
    # intact, which preserves the parent safety floor.
    "protect": {"activation_multiplier": 0.65, "giveback_multiplier": 0.75, "sl_distance_multiplier": 1.0},
    # A coherent trend vector.  It does not loosen a standing trailing or
    # smooth floor, but postpones a *future* activation and allows a wider
    # subsequent giveback when the parent has not yet ratcheted it.
    "trend_ride": {"activation_multiplier": 1.25, "giveback_multiplier": 1.25, "sl_distance_multiplier": 1.0},
}
KEY = ("candidate_id", "state_decision_ts")
LABEL_CONTEXT: dict[str, object] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def _sample_states(states: pd.DataFrame, eligible: set[str], maximum: int = 3) -> pd.DataFrame:
    """First/middle/last target-free state per candidate, outcome-independent."""
    pieces: list[pd.DataFrame] = []
    for _, group in states.loc[states["candidate_id"].isin(eligible)].groupby("candidate_id", sort=True):
        ordered = group.sort_values("state_decision_ts", kind="stable")
        take = np.unique(np.linspace(0, len(ordered) - 1, min(maximum, len(ordered)), dtype=int))
        pieces.append(ordered.iloc[take])
    sampled = pd.concat(pieces, ignore_index=True) if pieces else states.iloc[:0].copy()
    if sampled.duplicated(list(KEY)).any():
        raise AssertionError("target-free regime state sample duplicates identity")
    return sampled


def _label_candidate(task: tuple[int, tuple[int, ...], float]) -> list[dict[str, object]]:
    position, state_ns, parent_net = task
    rows = LABEL_CONTEXT["rows"]
    arrays = LABEL_CONTEXT["arrays"]
    params, median = LABEL_CONTEXT["policy"]
    assert isinstance(rows, pd.DataFrame) and isinstance(arrays, dict)
    row = rows.iloc[position]
    output: list[dict[str, object]] = []
    for target in state_ns:
        for regime, vector in REGIMES.items():
            def modulator(state: dict[str, float], target_ns: int = target, values: dict[str, float] = vector):
                # The selected regime begins only after this completed state
                # and persists across every following state.  Pre-switch
                # parent thresholds and MFE state are reproduced exactly.
                now = int(pd.Timestamp(state["state_decision_ts"]).value)
                return dict(values) if now >= target_ns else None
            trace = replay_exact_1m_gradual_h4_overlay(
                entry_price=float(arrays["entry"][position]), signal_atr=float(arrays["atr"][position]),
                entry_ts=row["entry_ts"], highs=arrays["high"][position], lows=arrays["low"][position], closes=arrays["close"][position],
                params=params, median_atr_fraction=float(median), mc1_expected_bps=0.0,
                state_modulator=modulator, allow_stop_extension=False, max_stop_loss_fraction=.05, emit_states=False,
            )
            output.append({
                "candidate_id": str(row["candidate_id"]), "state_decision_ts": pd.Timestamp(target, tz="UTC"),
                "regime": regime, "parent_exact_net_bps": float(parent_net), "counterfactual_net_bps": float(trace["net_bps"]),
                "advantage_bps": float(trace["net_bps"] - parent_net),
                "counterfactual_gross_bps": float(trace["gross_bps"]),
                "counterfactual_exit_price": float(trace["exit_price"]), "counterfactual_exit_ts": _utc(trace["exit_timestamp"]),
                "counterfactual_exit_minute": int(trace["exit_minute"]), "counterfactual_exit_reason": str(trace["exit_reason"]),
                "policy_label_available_ts": _utc(row["entry_ts"]) + H12,
            })
    return output


def _labels(rows: pd.DataFrame, arrays: dict[str, np.ndarray], policy: tuple[object, float], route: pd.DataFrame, outcomes: pd.DataFrame, sampled: pd.DataFrame, workers: int) -> pd.DataFrame:
    parent = outcomes.set_index("candidate_id")["parent_exact_net_bps"].astype(float)
    positions = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    tasks: list[tuple[int, tuple[int, ...], float]] = []
    eligible = set(route["candidate_id"].astype(str))
    for candidate, group in sampled.loc[sampled["candidate_id"].isin(eligible)].groupby("candidate_id", sort=True):
        value = str(candidate)
        if value in positions.index and value in parent.index:
            tasks.append((int(positions[value]), tuple(pd.to_datetime(group["state_decision_ts"], utc=True).astype("int64").tolist()), float(parent.loc[value])))
    if not tasks:
        raise RuntimeError("no complete candidates for remaining-regime labels")
    LABEL_CONTEXT.clear(); LABEL_CONTEXT.update({"rows": rows, "arrays": arrays, "policy": policy})
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("fork")) as executor:
        for ordinal, block in enumerate(executor.map(_label_candidate, tasks), 1):
            records.extend(block)
            if ordinal == 1 or ordinal % 500 == 0 or ordinal == len(tasks):
                print(f"remaining-regime labels {ordinal}/{len(tasks)}", flush=True)
    result = pd.DataFrame(records)
    if result.duplicated([*KEY, "regime"]).any() or set(result["regime"]) != set(REGIMES):
        raise AssertionError("remaining-regime label grid is incomplete")
    return result


def _conditional_frame(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    result = frame.loc[:, [*fields, "regime"]].copy()
    for regime in REGIMES:
        result[f"regime__{regime}"] = result["regime"].eq(regime).astype(float)
    return result.drop(columns="regime"), (*fields, *(f"regime__{regime}" for regime in REGIMES))


def _fit_value(train: pd.DataFrame, fields: tuple[str, ...], *, objective: str, alpha: float = .2) -> tuple[lgb.LGBMRegressor, tuple[str, ...]]:
    values, conditional_fields = _conditional_frame(train, fields)
    model = lgb.LGBMRegressor(
        objective=objective, alpha=float(alpha), n_estimators=340, learning_rate=.03,
        max_depth=3, num_leaves=7, min_child_samples=max(64, int(np.ceil(len(train) * .05))),
        subsample=.8, colsample_bytree=.8, reg_lambda=80., random_state=1729, n_jobs=2, verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    if objective != "quantile":
        weights *= np.where(train["advantage_bps"].to_numpy(float) > 0.0, 3.0, 1.0)
    model.fit(values.loc[:, conditional_fields], train["advantage_bps"].to_numpy(float), sample_weight=weights)
    return model, conditional_fields


def _predict_values(model: lgb.LGBMRegressor, conditional_fields: tuple[str, ...], states: pd.DataFrame, fields: tuple[str, ...], regime: str) -> np.ndarray:
    probe = states.loc[:, list(fields)].copy()
    for name in REGIMES:
        probe[f"regime__{name}"] = float(name == regime)
    return model.predict(probe.loc[:, conditional_fields])


def _coarse_edges(train: pd.DataFrame) -> dict[str, np.ndarray]:
    edges: dict[str, np.ndarray] = {}
    for field in ("current_MFE_ATR", "giveback_from_MFE_ATR", "current_pnl_atr", "time_in_trade"):
        values = pd.to_numeric(train[field], errors="coerce").dropna().to_numpy(float)
        cut = np.unique(np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])) if len(values) else np.array([])
        edges[field] = cut if len(cut) >= 3 else np.array([])
    return edges


def _coarse_key(frame: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for field, cut in edges.items():
        if len(cut) >= 3:
            adjusted = cut.copy(); adjusted[0], adjusted[-1] = -np.inf, np.inf
            result[field] = pd.cut(pd.to_numeric(frame[field], errors="coerce"), bins=adjusted, labels=False, include_lowest=True).fillna(-1).astype(int)
        else:
            result[field] = 0
    result["is_trailing_active"] = pd.to_numeric(frame["is_trailing_active"], errors="coerce").fillna(-1).round().astype(int)
    return result


def _coarse_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    edges = _coarse_edges(train)
    train_keys, test_keys = _coarse_key(train, edges), _coarse_key(test, edges)
    key_fields = list(train_keys.columns)
    source = train.loc[:, ["regime", "advantage_bps"]].join(train_keys)
    global_stats = source.groupby("regime")["advantage_bps"].agg(["mean", "std", "count"])
    grouped = source.groupby(["regime", *key_fields], as_index=False)["advantage_bps"].agg(["mean", "std", "count"]).reset_index()
    records: list[pd.DataFrame] = []
    prior_weight = 30.0
    for regime in REGIMES:
        values = test.loc[:, ["candidate_id", "state_decision_ts"]].copy().join(test_keys)
        probe = values.assign(regime=regime).merge(grouped.loc[grouped["regime"].eq(regime)], on=["regime", *key_fields], how="left")
        glob = global_stats.loc[regime]
        count = pd.to_numeric(probe["count"], errors="coerce").fillna(0.0)
        mean = pd.to_numeric(probe["mean"], errors="coerce").fillna(float(glob["mean"]))
        std = pd.to_numeric(probe["std"], errors="coerce").fillna(float(glob["std"]) if np.isfinite(glob["std"]) else 100.0)
        shrink = (count * mean + prior_weight * float(glob["mean"])) / (count + prior_weight)
        lcb = shrink - 1.28 * std / np.sqrt(np.maximum(count + prior_weight, 1.0))
        records.append(probe.loc[:, ["candidate_id", "state_decision_ts"]].assign(regime=regime, pred_mean_bps=shrink, pred_lcb_bps=lcb, support=count))
    return pd.concat(records, ignore_index=True)


def _first_selector(predictions: pd.DataFrame, *, permitted: tuple[str, ...], gate: float, score_column: str, name: str) -> pd.DataFrame:
    """Select the first causal state with a conservative regime advantage.

    Once selected the regime persists to exit, giving a remainder-of-trade
    commitment (there is no later model-driven flip-flop).  This is stricter
    than a 30/60-minute hold requirement.
    """
    selected = predictions.loc[predictions["regime"].isin(permitted)].copy()
    selected = selected.loc[(selected[score_column] >= gate) & (selected["pred_mean_bps"] > 0.0)].copy()
    if selected.empty:
        return pd.DataFrame(columns=[*KEY, "regime", "selection_score_bps", "selection_mean_bps", "selector"])
    selected = selected.sort_values(["candidate_id", "state_decision_ts", score_column, "regime"], ascending=[True, True, False, True], kind="stable")
    first = selected.groupby("candidate_id", as_index=False, sort=False).head(1).copy()
    return first.loc[:, [*KEY, "regime"]].assign(selection_score_bps=first[score_column].to_numpy(float), selection_mean_bps=first["pred_mean_bps"].to_numpy(float), selector=name)


def _replay_regimes(rows: pd.DataFrame, arrays: dict[str, np.ndarray], route: pd.DataFrame, params: object, median: float, selected: pd.DataFrame | None) -> pd.DataFrame:
    positions = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    actions: dict[str, tuple[int, str]] = {}
    if selected is not None and not selected.empty:
        if selected.duplicated("candidate_id").any():
            raise AssertionError("a persistent selector may choose at most one state per trade")
        actions = {
            str(row.candidate_id): (int(pd.Timestamp(row.state_decision_ts).value), str(row.regime))
            for row in selected.itertuples(index=False)
        }
    records: list[dict[str, object]] = []
    for route_row in route.sort_values(["timestamp", "candidate_id"], kind="stable").itertuples(index=False):
        candidate = str(route_row.candidate_id)
        position = int(positions.loc[candidate])
        action = actions.get(candidate)
        def modulator(state: dict[str, float], action_value=action):
            if action_value is None:
                return None
            target, regime = action_value
            return dict(REGIMES[regime]) if int(pd.Timestamp(state["state_decision_ts"]).value) >= target else None
        trace = replay_exact_1m_gradual_h4_overlay(
            entry_price=float(arrays["entry"][position]), signal_atr=float(arrays["atr"][position]), entry_ts=rows.iloc[position]["entry_ts"],
            highs=arrays["high"][position], lows=arrays["low"][position], closes=arrays["close"][position],
            params=params, median_atr_fraction=float(median), mc1_expected_bps=float(route_row.bcf_mc1_expected_bps),
            state_modulator=modulator if action is not None else None, allow_stop_extension=False, max_stop_loss_fraction=.05, emit_states=False,
        )
        records.append({
            "candidate_id": candidate, "timestamp": route_row.timestamp, "entry_ts": rows.iloc[position]["entry_ts"], "symbol": route_row.symbol,
            "bcf_mc1_expected_bps": float(route_row.bcf_mc1_expected_bps), "current_mc1_expected_bps": float(route_row.current_mc1_expected_bps),
            "auction_priority_bps": float(route_row.auction_priority_bps), "exact_entry_price": float(arrays["entry"][position]),
            "exact_net_bps": float(trace["net_bps"]), "exact_gross_bps": float(trace["gross_bps"]), "exact_exit_price": float(trace["exit_price"]),
            "exact_exit_ts": _utc(trace["exit_timestamp"]), "exact_exit_minute": int(trace["exit_minute"]), "exact_exit_reason": str(trace["exit_reason"]),
            "selected_regime": action[1] if action is not None else "parent", "selected_state_ts": pd.Timestamp(action[0], tz="UTC") if action is not None else pd.NaT,
        })
    return pd.DataFrame(records)


def _oracle_outcomes(route: pd.DataFrame, parent: pd.DataFrame, labels: pd.DataFrame, permitted: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfect sampled-state oracle; explicitly outcome-aware, never deployable."""
    candidates = set(route["candidate_id"].astype(str))
    choices = labels.loc[labels["candidate_id"].isin(candidates) & labels["regime"].isin(permitted)].copy()
    choices = choices.loc[choices["advantage_bps"] > 0.0].sort_values(["candidate_id", "advantage_bps", "state_decision_ts", "regime"], ascending=[True, False, True, True], kind="stable")
    choices = choices.groupby("candidate_id", as_index=False, sort=False).head(1).copy()
    base_frame = parent.copy()
    # Use the already parity-checked parent replay for immutable candidate context, then replace
    # exact outcome fields by the precomputed state-specific counterfactual.
    if not choices.empty:
        lookup = choices.set_index("candidate_id")
        mask = base_frame["candidate_id"].isin(lookup.index)
        for field, target in (
            ("counterfactual_net_bps", "exact_net_bps"), ("counterfactual_gross_bps", "exact_gross_bps"),
            ("counterfactual_exit_price", "exact_exit_price"), ("counterfactual_exit_ts", "exact_exit_ts"),
            ("counterfactual_exit_minute", "exact_exit_minute"), ("counterfactual_exit_reason", "exact_exit_reason"),
        ):
            base_frame.loc[mask, target] = base_frame.loc[mask, "candidate_id"].map(lookup[field])
        base_frame.loc[mask, "selected_regime"] = base_frame.loc[mask, "candidate_id"].map(lookup["regime"])
        base_frame.loc[mask, "selected_state_ts"] = base_frame.loc[mask, "candidate_id"].map(lookup["state_decision_ts"])
    return base_frame, choices


def _portfolio_metrics(outcome: pd.DataFrame, arm: str, out: Path, prefix: str) -> dict[str, object]:
    candidates, decisions, accepted, equity, metrics = base._portfolio(outcome, arm)
    if not accepted.empty:
        accepted = accepted.copy()
        positions = pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int).to_numpy()
        accepted["holding_bars"] = candidates.iloc[positions]["holding_bars"].to_numpy()
    extra = base._extra_metrics(accepted)
    for name, data in (("exact1m_outcomes", outcome), ("portfolio_accepted", accepted), ("portfolio_decisions", decisions), ("portfolio_equity", equity)):
        data.to_parquet(out / f"{prefix}_{arm}_{name}.parquet", index=False, compression="zstd")
    return {"arm": arm, **metrics, **extra, "selected_candidates": int((outcome["selected_regime"] != "parent").sum())}


def _monthly(out: Path, prefix: str) -> None:
    records: list[pd.DataFrame] = []
    for path in sorted(out.glob(f"{prefix}_*_portfolio_accepted.parquet")):
        arm = path.name.removeprefix(f"{prefix}_").removesuffix("_portfolio_accepted.parquet")
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["month"] = pd.to_datetime(frame["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
        grouped = frame.groupby("month", as_index=False).agg(trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum"))
        grouped.insert(0, "arm", arm); records.append(grouped)
    (pd.concat(records, ignore_index=True) if records else pd.DataFrame()).to_parquet(out / f"{prefix}_monthly_metrics.parquet", index=False, compression="zstd")


def _oof_predictions(labels: pd.DataFrame, states: pd.DataFrame, fields: tuple[str, ...], *, start: pd.Timestamp, end: pd.Timestamp, validation: bool = False) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    # Validation is a single, frozen forward block: fit on every label resolved
    # before its first decision, then score the whole block.  It is deliberately
    # not the monthly OOF loop with `end` used as a zero-length holdout.
    months = [start] if validation else [pd.Timestamp(period.start_time, tz="UTC") for period in pd.period_range(start, end - pd.offsets.MonthBegin(1), freq="M")]
    for held_start in months:
        held_end = end if validation else held_start + pd.offsets.MonthBegin(1)
        if validation:
            # `labels` is itself the predeclared 2025 label substrate.  The
            # frozen 2026 model may use all of it, provided every label had
            # resolved before the first 2026 decision.  Do not reuse the
            # validation start as a lower training bound.
            train = labels.loc[labels["policy_label_available_ts"].lt(held_start)].copy()
        else:
            train = labels.loc[labels["entry_decision_ts"].ge(start) & labels["entry_decision_ts"].lt(held_start) & labels["policy_label_available_ts"].lt(held_start)].copy()
        test = states.loc[states["entry_decision_ts"].ge(held_start) & states["entry_decision_ts"].lt(held_end)].copy()
        if test.empty or train["candidate_id"].nunique() < 250:
            continue
        mean_model, conditional = _fit_value(train, fields, objective="regression_l1")
        lcb_model, _ = _fit_value(train, fields, objective="quantile", alpha=.20)
        coarse = _coarse_predict(train, test)
        for regime in REGIMES:
            base_rows = test.loc[:, ["candidate_id", "state_decision_ts"]].copy().assign(regime=regime)
            base_rows["pred_mean_bps"] = _predict_values(mean_model, conditional, test, fields, regime)
            base_rows["pred_lcb_bps"] = _predict_values(lcb_model, conditional, test, fields, regime)
            base_rows["selector_model"] = "ml"
            base_rows["held_start"] = held_start
            pieces.append(base_rows)
        coarse["selector_model"] = "coarse"; coarse["held_start"] = held_start; pieces.append(coarse)
    if not pieces:
        raise RuntimeError("no remaining-regime strict-prior predictions")
    return pd.concat(pieces, ignore_index=True)


def _run_period(rows: pd.DataFrame, arrays: dict[str, np.ndarray], route: pd.DataFrame, params: object, median: float, labels: pd.DataFrame, predictions: pd.DataFrame, *, prefix: str, out: Path, include_oracle: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    parent = _replay_regimes(rows, arrays, route, params, median, None)
    records.append(_portfolio_metrics(parent, "parent", out, prefix))
    if include_oracle:
        for name, permitted in (("oracle_r1_protect", ("protect",)), ("oracle_r2_protect_trend", tuple(REGIMES))):
            outcome, selected = _oracle_outcomes(route, parent, labels, permitted)
            selected.to_parquet(out / f"{prefix}_{name}_sampled_state_choices.parquet", index=False, compression="zstd")
            records.append(_portfolio_metrics(outcome, name, out, prefix))
    selection_defs = (
        ("coarse_r1_lcb20", "coarse", ("protect",), 20.0, "pred_lcb_bps"),
        ("coarse_r2_lcb20", "coarse", tuple(REGIMES), 20.0, "pred_lcb_bps"),
        ("ml_r1_mean20", "ml", ("protect",), 20.0, "pred_mean_bps"),
        ("ml_r2_mean20", "ml", tuple(REGIMES), 20.0, "pred_mean_bps"),
        ("ml_r1_lcb20", "ml", ("protect",), 20.0, "pred_lcb_bps"),
        ("ml_r2_lcb20", "ml", tuple(REGIMES), 20.0, "pred_lcb_bps"),
        ("ml_r2_lcb30", "ml", tuple(REGIMES), 30.0, "pred_lcb_bps"),
    )
    for arm, model, permitted, gate, score in selection_defs:
        available = predictions.loc[predictions["selector_model"].eq(model)].copy()
        schedule = _first_selector(available, permitted=permitted, gate=gate, score_column=score, name=arm)
        schedule.to_parquet(out / f"{prefix}_{arm}_persistent_schedule.parquet", index=False, compression="zstd")
        outcome = _replay_regimes(rows, arrays, route, params, median, schedule)
        value = _portfolio_metrics(outcome, arm, out, prefix)
        value.update({"selector_model": model, "permitted_regimes": ",".join(permitted), "gate_bps": gate, "score": score, "scheduled_state_actions": int(len(schedule))})
        records.append(value)
        del outcome; gc.collect()
    summary = pd.DataFrame(records)
    ref = summary.loc[summary["arm"].eq("parent")].iloc[0]
    for field in ("net_bps_per_trade", "total_net_bps", "sortino", "max_drawdown", "worst_week", "cvar10_bps", "worst_month_bps"):
        if field in summary:
            summary[f"delta_vs_parent_{field}"] = summary[field] - ref[field]
    summary["total_bps_per_abs_drawdown"] = summary["total_net_bps"] / summary["max_drawdown"].abs().clip(lower=1e-9)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, default=base.DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATES)
    parser.add_argument("--policy", type=Path, default=base.DEFAULT_POLICY)
    parser.add_argument("--label-workers", type=int, default=8)
    parser.add_argument("--resume-labels-from", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    parent_root, state_root, policy_path = args.parent_root.resolve(), args.state_root.resolve(), args.policy.resolve()
    route, rows, outcomes, arrays, states = base._load_parent(parent_root, state_root)
    params, median, _ = base._load_policy(policy_path)
    fields = base._state_fields(states)
    select_start, select_end = _month("2025-06"), _month("2026-01")
    validate_start, validate_end = _month("2026-06"), _month("2026-09")
    label_route = route.loc[(route["bcf_mc1_expected_bps"] >= 40.0) & (route["current_mc1_expected_bps"] >= 40.0) & route["timestamp"].ge(select_start) & route["timestamp"].lt(select_end)].copy()
    normal = route.loc[(route["bcf_mc1_expected_bps"] >= 50.0) & (route["current_mc1_expected_bps"] >= 50.0)].copy()
    normal_2025 = normal.loc[normal["timestamp"].ge(select_start) & normal["timestamp"].lt(select_end)].copy()
    normal_2026 = normal.loc[normal["timestamp"].ge(validate_start) & normal["timestamp"].lt(validate_end)].copy()
    if args.resume_labels_from is None:
        sampled = _sample_states(states, set(label_route["candidate_id"].astype(str)))
        labels = _labels(rows, arrays, (params, median), label_route, outcomes, sampled, args.label_workers)
        resumed = None
    else:
        resumed = args.resume_labels_from.resolve()
        sampled = pd.read_parquet(resumed / "training_state_sample_target_free.parquet")
        labels = pd.read_parquet(resumed / "remaining_regime_counterfactual_labels.parquet")
        if set(labels["regime"]) != set(REGIMES):
            raise AssertionError("resumed remaining-regime labels are incomplete")
    label_states = states.merge(labels.loc[:, [*KEY, "regime", "advantage_bps", "policy_label_available_ts"]], on=list(KEY), how="inner", validate="one_to_many")
    label_states["entry_decision_ts"] = pd.to_datetime(label_states["entry_decision_ts"], utc=True, errors="raise")
    normal_states_2025 = states.loc[states["candidate_id"].isin(set(normal_2025["candidate_id"].astype(str)))].copy()
    normal_states_2026 = states.loc[states["candidate_id"].isin(set(normal_2026["candidate_id"].astype(str)))].copy()
    # Parent parity must hold before any persistent counterfactual is accepted.
    probe = normal_2025.head(64)
    replay = _replay_regimes(rows, arrays, probe, params, median, None).set_index("candidate_id")["exact_net_bps"]
    archived = outcomes.set_index("candidate_id").loc[replay.index, "parent_exact_net_bps"]
    if not np.allclose(replay.to_numpy(float), archived.to_numpy(float), atol=1e-8, rtol=0.0):
        raise AssertionError("remaining-regime parent replay parity failed")
    out.mkdir(parents=True, exist_ok=False)
    sampled.to_parquet(out / "training_state_sample_target_free.parquet", index=False, compression="zstd")
    labels.to_parquet(out / "remaining_regime_counterfactual_labels.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(fields)), "feature": fields}).to_parquet(out / "target_free_h4_feature_contract.parquet", index=False, compression="zstd")
    label_diagnostics = labels.groupby("regime", as_index=False).agg(rows=("candidate_id", "size"), mean_advantage_bps=("advantage_bps", "mean"), positive_share=("advantage_bps", lambda value: float((value > 0.0).mean())), exit_change_share=("advantage_bps", lambda value: float((value != 0.0).mean())))
    label_diagnostics.to_parquet(out / "label_regime_diagnostics.parquet", index=False, compression="zstd")
    oof = _oof_predictions(label_states, normal_states_2025, fields, start=select_start, end=select_end)
    oof.to_parquet(out / "2025_strict_prior_regime_predictions.parquet", index=False, compression="zstd")
    summary_2025 = _run_period(rows, arrays, normal_2025, params, median, labels, oof, prefix="2025_oof", out=out, include_oracle=True)
    summary_2025.to_parquet(out / "2025_oof_constrained_regime_summary.parquet", index=False, compression="zstd")
    _monthly(out, "2025_oof")
    # Feasible-arm selection excludes both outcome-aware sampled-state oracles.
    feasible = summary_2025.loc[~summary_2025["arm"].str.startswith("oracle_") & ~summary_2025["arm"].eq("parent")].copy()
    winner = feasible.sort_values(["total_bps_per_abs_drawdown", "net_bps_per_trade", "worst_week"], ascending=False, kind="stable").iloc[0]
    frozen = _oof_predictions(label_states, normal_states_2026, fields, start=validate_start, end=validate_end, validation=True)
    frozen.to_parquet(out / "2026_frozen_regime_predictions.parquet", index=False, compression="zstd")
    arm = str(winner["arm"]); model = str(winner["selector_model"]); permitted = tuple(str(winner["permitted_regimes"]).split(",")); gate = float(winner["gate_bps"]); score = str(winner["score"])
    schedule = _first_selector(frozen.loc[frozen["selector_model"].eq(model)], permitted=permitted, gate=gate, score_column=score, name=arm)
    schedule.to_parquet(out / "2026_frozen_winner_persistent_schedule.parquet", index=False, compression="zstd")
    parent_outcome = _replay_regimes(rows, arrays, normal_2026, params, median, None)
    winner_outcome = _replay_regimes(rows, arrays, normal_2026, params, median, schedule)
    val = pd.DataFrame([_portfolio_metrics(parent_outcome, "parent", out, "2026_frozen"), _portfolio_metrics(winner_outcome, "winner", out, "2026_frozen")])
    reference = val.loc[val["arm"].eq("parent")].iloc[0]
    for field in ("net_bps_per_trade", "total_net_bps", "sortino", "max_drawdown", "worst_week", "cvar10_bps", "worst_month_bps"):
        val[f"delta_vs_parent_{field}"] = val[field] - reference[field]
    val.to_parquet(out / "2026_frozen_confirmation_summary.parquet", index=False, compression="zstd")
    _monthly(out, "2026_frozen")
    manifest = {
        "schema": "causal-sr-h4-remaining-regime-ablation-v1",
        "scope": "offline causal research only; no live policy, exchange, admission, portfolio, Geometry/K9, MC1, or C1 S/R mutation",
        "regimes": REGIMES,
        "target": "exact H12 net-bps difference when a coherent regime begins after one completed H4 state and persists until trade exit",
        "selection": "2025-06..2025-12 strict-prior monthly OOF; first qualifying conservative state selects one persistent remainder-of-trade regime",
        "oracle": "sampled-state outcome-aware upper bound only; it is not feasible or eligible for frozen winner selection",
        "commitment": "selected regime persists for the remaining trade, stronger than a 30/60-minute commitment and prohibits flip-flopping",
        "frozen_confirmation": "2026-06..2026-08; fit only on resolved 2025 labels, with one 2025-selected feasible arm",
        "features": "unchanged 91-field target-free H4 contract",
        "parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"),
        "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json"),
        "policy_sha256": _sha256(policy_path), "resumed_labels_from": str(resumed) if resumed is not None else None,
        "selected_2025_feasible_arm": winner.to_dict(), "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
