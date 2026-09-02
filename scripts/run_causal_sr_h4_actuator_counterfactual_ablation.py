#!/usr/bin/env python3
"""Strict-OOS actuator-specific H4 counterfactual exit study.

This is an offline research runner.  It starts with an immutable causal S/R
and paired-MC1 target-free route, materialised exact one-minute rich-parent
paths, and target-free H4 state snapshots.  It then:

* builds direct counterfactual labels for giveback, trailing activation, and
  stop distance separately;
* uses a broader dual-MC1 route only for *state-label training*;
* selects/evaluates controllers on the normal dual-50 route with the normal
  global portfolio auction; and
* keeps every 2026 model frozen from 2025 labels.

No live modules, exchange clients, order submission, score fit, admission fit,
or portfolio rule is changed by this script.

The implementation intentionally begins with the linked specification's
Stage-1 shallow-LGBM screen.  It compares direct dual-advantage and ordinal
counterfactual controllers, including widen-only, tighten-only and
asymmetric authority.  Later HPO must consume this receipt rather than tune
against the 2026 confirmation period.
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import (
    replay_exact_1m_gradual_h4_overlay,
)
from scripts.run_causal_sr_c1_h4_exact1m_refit_replay import H12
from scripts.run_causal_sr_c1_h4_exact1m_transfer_replay import _portfolio
from scripts.run_strict_r3_p8u_exact_1m_simple_policy_optimiser import _load_policy


DEFAULT_PARENT = ROOT / "data_perp/artifacts/causal_sr_c1_exact1m_parent_t40_expanded_20260831_v1"
DEFAULT_STATES = ROOT / "data_perp/artifacts/causal_sr_c1_h4_expanded_support_20260831_v1"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_h4_actuator_counterfactual_2025oof_2026confirm_20260831_v1"

ACTUATORS = ("giveback", "activation", "stop")
MULTIPLIERS = (0.65, 0.80, 1.00, 1.25, 1.50)
LABEL_CONTEXT: dict[str, object] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _load_parent(parent: Path, state_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], tuple[object, float]]:
    """Load the immutable exact path and target-free contextual substrates."""
    route = pd.read_parquet(parent / "union_target_free_dual_admitted.parquet").copy()
    route["candidate_id"] = route["candidate_id"].astype(str)
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="raise")
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    for field in ("bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"):
        route[field] = pd.to_numeric(route[field], errors="raise")
    rows = pd.read_parquet(parent / "valid_exact_paths_rows.parquet").copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="raise")
    rows["entry_ts"] = pd.to_datetime(rows["entry_ts"], utc=True, errors="raise")
    packed = np.load(parent / "exact_paths.npz", allow_pickle=True)
    if not np.array_equal(rows["candidate_id"].to_numpy(str), packed["candidate_id"].astype(str)):
        raise AssertionError("exact path archive lost candidate identity")
    arrays = {name: np.asarray(packed[name]) for name in ("entry", "atr", "high", "low", "close")}
    if any(len(values) != len(rows) for values in arrays.values()):
        raise AssertionError("exact path arrays are not row-aligned")
    # The broader target-free route can contain otherwise valid scored
    # candidates whose exact 12-hour minute path was intentionally excluded
    # from the parent materialisation.  They cannot receive an exact
    # counterfactual label *or* an exact rich-policy outcome.  Remove them at
    # the shared loader boundary so fitting and portfolio assessment use the
    # same complete-path universe; never treat their absence as a flat or
    # capacity-reserving pseudo-trade.
    exact_ids = set(rows["candidate_id"])
    route = route.loc[route["candidate_id"].isin(exact_ids)].copy()
    if route.empty:
        raise RuntimeError("no scored route candidates have a complete exact path")
    outcome = pd.read_parquet(parent / "exact_1m_rich_parent_outcomes.parquet").copy()
    outcome["candidate_id"] = outcome["candidate_id"].astype(str)
    outcome["parent_exact_net_bps"] = pd.to_numeric(outcome["exact_net_bps"], errors="raise")
    if outcome["candidate_id"].duplicated().any():
        raise AssertionError("parent outcome duplicates candidate identity")
    states = pd.read_parquet(state_root / "all_h4_context_states_target_free.parquet").copy()
    states["candidate_id"] = states["candidate_id"].astype(str)
    states["state_decision_ts"] = pd.to_datetime(states["state_decision_ts"], utc=True, errors="raise")
    states["entry_decision_ts"] = pd.to_datetime(states["entry_decision_ts"], utc=True, errors="raise")
    key = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    if states.duplicated(key).any():
        raise AssertionError("target-free H4 state panel duplicates an identity")
    known = set(rows["candidate_id"])
    states = states.loc[states["candidate_id"].isin(known)].copy()
    if states.empty:
        raise RuntimeError("no H4 states align to the exact path archive")
    return route, rows, outcome, arrays, states


def _state_fields(states: pd.DataFrame) -> tuple[str, ...]:
    """Existing H4 fields only; no post-2025 feature selection is imported.

    The C4 feature contract was selected in 2026, so using it to assess 2025
    would leak a later selection decision.  Stage 1 therefore uses the full
    pre-existing numeric H4 state namespace.  Tree regularisation, not a
    later feature selection receipt, controls capacity.
    """
    forbidden = {
        "candidate_id", "state_decision_ts", "state_source_end_ts", "entry_decision_ts",
        "__symbol__", "state_bar_15m", "entry_price", "signal_atr",
    }
    fields: list[str] = []
    for field in states.columns:
        if field in forbidden:
            continue
        values = states[field]
        if pd.api.types.is_numeric_dtype(values):
            fields.append(str(field))
    if len(fields) < 30:
        raise AssertionError("unexpectedly small existing H4 feature namespace")
    return tuple(fields)


def _sample_states(states: pd.DataFrame, *, maximum: int) -> pd.DataFrame:
    """First/middle/last observable states, independent of outcomes/labels."""
    if maximum < 1:
        raise ValueError("maximum state samples must be positive")
    pieces: list[pd.DataFrame] = []
    for _, group in states.groupby("candidate_id", sort=True):
        group = group.sort_values("state_decision_ts", kind="stable")
        count = min(int(maximum), len(group))
        take = np.unique(np.linspace(0, len(group) - 1, count, dtype=int))
        pieces.append(group.iloc[take])
    result = pd.concat(pieces, ignore_index=True) if pieces else states.iloc[:0].copy()
    key = ["candidate_id", "state_decision_ts"]
    if result.duplicated(key).any():
        raise AssertionError("target-free state sampling duplicated a state")
    return result


def _counterfactual_label_candidate(task: tuple[int, tuple[int, ...], float]) -> list[dict[str, object]]:
    """Exact H12 values for permanent single-actuator changes from each state."""
    position, state_ns, parent_net = task
    rows = LABEL_CONTEXT["rows"]
    arrays = LABEL_CONTEXT["arrays"]
    params, median = LABEL_CONTEXT["policy"]
    row = rows.iloc[position]
    records: list[dict[str, object]] = []
    for target_ns in state_ns:
        for actuator in ACTUATORS:
            for multiplier in MULTIPLIERS:
                if np.isclose(multiplier, 1.0):
                    value = float(parent_net)
                else:
                    def modulator(state: dict[str, float], target: int = target_ns, act: str = actuator, mult: float = multiplier):
                        if int(pd.Timestamp(state["state_decision_ts"]).value) < target:
                            return None
                        values = {"activation_multiplier": 1.0, "giveback_multiplier": 1.0, "sl_distance_multiplier": 1.0}
                        values[f"{act}_multiplier" if act != "activation" else "activation_multiplier"] = mult
                        return values
                    trace = replay_exact_1m_gradual_h4_overlay(
                        entry_price=float(arrays["entry"][position]), signal_atr=float(arrays["atr"][position]),
                        entry_ts=row["entry_ts"], highs=arrays["high"][position], lows=arrays["low"][position],
                        closes=arrays["close"][position], params=params, median_atr_fraction=float(median),
                        mc1_expected_bps=0.0, state_modulator=modulator, allow_stop_extension=True,
                        max_stop_loss_fraction=0.05, emit_states=False,
                    )
                    value = float(trace["net_bps"])
                records.append({
                    "candidate_id": str(row["candidate_id"]),
                    "state_decision_ts": pd.Timestamp(target_ns, tz="UTC"),
                    "actuator": actuator, "multiplier": float(multiplier), "counterfactual_net_bps": value,
                    "parent_exact_net_bps": float(parent_net), "advantage_bps": float(value - parent_net),
                    "policy_label_available_ts": pd.Timestamp(row["entry_ts"]) + H12,
                })
    return records


def _labels(
    rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    policy: tuple[object, float],
    route: pd.DataFrame,
    outcomes: pd.DataFrame,
    samples: pd.DataFrame,
    workers: int,
) -> pd.DataFrame:
    parent = outcomes.set_index("candidate_id")["parent_exact_net_bps"].astype(float)
    position = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    eligible_ids = set(route["candidate_id"].astype(str))
    tasks: list[tuple[int, tuple[int, ...], float]] = []
    for candidate, group in samples.loc[samples["candidate_id"].isin(eligible_ids)].groupby("candidate_id", sort=True):
        candidate = str(candidate)
        if candidate not in position.index or candidate not in parent.index:
            continue
        tasks.append((
            int(position[candidate]),
            tuple(pd.to_datetime(group["state_decision_ts"], utc=True).astype("int64").tolist()),
            float(parent.loc[candidate]),
        ))
    if not tasks:
        raise RuntimeError("no expanded target-free state samples have exact parent labels")
    LABEL_CONTEXT.clear()
    LABEL_CONTEXT.update({"rows": rows, "arrays": arrays, "policy": policy})
    result: list[dict[str, object]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=context) as executor:
        for ordinal, block in enumerate(executor.map(_counterfactual_label_candidate, tasks), start=1):
            result.extend(block)
            if ordinal == 1 or ordinal % 500 == 0 or ordinal == len(tasks):
                print(f"H4 actuator labels {ordinal}/{len(tasks)} candidates", flush=True)
    label = pd.DataFrame(result)
    key = ["candidate_id", "state_decision_ts", "actuator", "multiplier"]
    if label.duplicated(key).any():
        raise AssertionError("counterfactual label duplicate")
    if label["multiplier"].eq(1.0).any() and not np.allclose(label.loc[label["multiplier"].eq(1.0), "advantage_bps"], 0.0):
        raise AssertionError("baseline counterfactual is not exactly zero advantage")
    return label


def _label_panel(states: pd.DataFrame, labels: pd.DataFrame, *, epsilon: float) -> pd.DataFrame:
    """Attach sparse exact labels to every target-free state/action.

    A selected state obtains an outcome-derived label only after H12, while
    all completed states remain available for the later causal inference
    schedule.  This is essential: limiting replay authority to label-sampled
    states would make the controller's evaluated behavior depend on a future
    training-sampling convenience.
    """
    key = ["candidate_id", "state_decision_ts"]
    pivot = labels.pivot(index=[*key, "actuator"], columns="multiplier", values="advantage_bps").reset_index()
    lookup = {float(value): value for value in pivot.columns if isinstance(value, float)}
    required = set(MULTIPLIERS)
    if set(lookup) != required:
        raise AssertionError("counterfactual multiplier grid is incomplete")
    pivot["tight_advantage_bps"] = pivot[[lookup[0.65], lookup[0.80]]].max(axis=1)
    pivot["wide_advantage_bps"] = pivot[[lookup[1.25], lookup[1.50]]].max(axis=1)
    choices = np.asarray(MULTIPLIERS, dtype=float)
    value_matrix = pivot.loc[:, [lookup[value] for value in choices]].to_numpy(float)
    best_idx = value_matrix.argmax(axis=1)
    best_value = value_matrix[np.arange(len(pivot)), best_idx]
    selected = choices[best_idx]
    selected[np.maximum(pivot["tight_advantage_bps"], pivot["wide_advantage_bps"]).to_numpy(float) < float(epsilon)] = 1.0
    ordinal = {0.65: -2, 0.80: -1, 1.0: 0, 1.25: 1, 1.5: 2}
    pivot["ordinal_action"] = pd.Series(selected).map(ordinal).to_numpy(int)
    pivot["oracle_advantage_bps"] = np.maximum(best_value, 0.0)
    availability = labels.loc[:, [*key, "policy_label_available_ts"]].drop_duplicates(key)
    if availability.duplicated(key).any():
        raise AssertionError("counterfactual label availability is not unique per state")
    base = states.copy()
    base["_join"] = 1
    actions = pd.DataFrame({"actuator": ACTUATORS, "_join": 1})
    expanded = base.merge(actions, on="_join", how="inner", validate="many_to_many").drop(columns="_join")
    merged = expanded.merge(pivot, on=[*key, "actuator"], how="left", validate="one_to_one")
    merged = merged.merge(availability, on=key, how="left", validate="many_to_one")
    return merged


def _fit_regression(train: pd.DataFrame, fields: tuple[str, ...], target: str) -> lgb.LGBMRegressor:
    child = max(64, int(np.ceil(len(train) * .05)))
    model = lgb.LGBMRegressor(
        objective="regression_l2", n_estimators=280, learning_rate=.035,
        max_depth=3, num_leaves=7, min_child_samples=child,
        subsample=.80, colsample_bytree=.80, reg_lambda=40.0,
        random_state=1729, n_jobs=2, verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], train[target].to_numpy(float), sample_weight=weights)
    return model


def _fit_ordinal(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMClassifier:
    child = max(64, int(np.ceil(len(train) * .05)))
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=280, learning_rate=.035,
        max_depth=3, num_leaves=7, min_child_samples=child,
        subsample=.80, colsample_bytree=.80, reg_lambda=40.0,
        random_state=1729, n_jobs=2, verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], (train["ordinal_action"] + 2).to_numpy(int), sample_weight=weights)
    return model


def _schedule(
    panel: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    train_start: pd.Timestamp,
    held_start: pd.Timestamp,
    held_end: pd.Timestamp,
    frozen_2025: bool,
    authority_bps: float,
) -> pd.DataFrame:
    """Strict-prior monthly OOF schedule, or one fixed 2025 model for 2026."""
    test = panel.loc[panel["entry_decision_ts"].ge(held_start) & panel["entry_decision_ts"].lt(held_end)].copy()
    base = test.loc[:, ["candidate_id", "state_decision_ts", "actuator"]].copy()
    for name in ("wide_advantage_bps", "tight_advantage_bps"):
        base[f"pred_{name}"] = 0.0
    base["pred_ordinal_action"] = 0
    if test.empty:
        return base
    train = panel.loc[
        panel["entry_decision_ts"].ge(train_start) & panel["entry_decision_ts"].lt(held_start)
        & panel["policy_label_available_ts"].lt(held_start)
        & panel["wide_advantage_bps"].notna() & panel["tight_advantage_bps"].notna()
        & panel["ordinal_action"].notna()
    ].copy()
    if train["candidate_id"].nunique() < 250:
        return base
    parts: list[pd.DataFrame] = []
    for actuator, test_group in test.groupby("actuator", sort=True):
        train_group = train.loc[train["actuator"].eq(actuator)].copy()
        result = test_group.loc[:, ["candidate_id", "state_decision_ts", "actuator"]].copy()
        if train_group["candidate_id"].nunique() < 100:
            result["pred_wide_advantage_bps"] = 0.0
            result["pred_tight_advantage_bps"] = 0.0
            result["pred_ordinal_action"] = 0
        else:
            wide = _fit_regression(train_group, fields, "wide_advantage_bps")
            tight = _fit_regression(train_group, fields, "tight_advantage_bps")
            ordinal = _fit_ordinal(train_group, fields)
            result["pred_wide_advantage_bps"] = wide.predict(test_group.loc[:, fields])
            result["pred_tight_advantage_bps"] = tight.predict(test_group.loc[:, fields])
            probabilities = ordinal.predict_proba(test_group.loc[:, fields])
            # With an explicit ``num_class=5`` LightGBM emits the complete
            # ordered probability vector even when a sparse early fold has
            # observed only its neutral class.  ``classes_`` then contains
            # only observed labels, so use the fixed ordinal output geometry.
            classes = np.arange(probabilities.shape[1], dtype=float) - 2.0
            result["pred_ordinal_action"] = np.rint(probabilities @ classes).astype(int)
        parts.append(result)
    return pd.concat(parts, ignore_index=True) if parts else base


def _modifiers(predictions: pd.DataFrame, *, threshold: float) -> dict[str, pd.DataFrame]:
    """Stage-1 semantically compatible mappings from direct labels."""
    common = predictions.loc[:, ["candidate_id", "state_decision_ts", "actuator"]].copy()
    wide = predictions["pred_wide_advantage_bps"].to_numpy(float)
    tight = predictions["pred_tight_advantage_bps"].to_numpy(float)
    ordinal = predictions["pred_ordinal_action"].to_numpy(int)
    result: dict[str, pd.DataFrame] = {}
    for name in ("dual_widen", "dual_tighten", "dual_asymmetric", "ordinal_step"):
        frame = common.copy()
        frame["multiplier"] = 1.0
        if name == "dual_widen":
            frame.loc[wide >= threshold, "multiplier"] = 1.25
        elif name == "dual_tighten":
            frame.loc[tight >= threshold, "multiplier"] = .80
        elif name == "dual_asymmetric":
            frame.loc[(wide >= threshold) & (wide > tight), "multiplier"] = 1.25
            frame.loc[(tight >= threshold) & (tight > wide), "multiplier"] = .80
        else:
            frame["multiplier"] = np.select(
                [ordinal <= -2, ordinal == -1, ordinal == 1, ordinal >= 2],
                [.65, .80, 1.25, 1.50], default=1.0,
            )
        result[name] = frame
    return result


def _replay(
    rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    route: pd.DataFrame,
    params: object,
    median: float,
    schedule: pd.DataFrame | None,
    actuator: str | None,
) -> pd.DataFrame:
    pos = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    lookup: dict[str, dict[int, float]] = {}
    if schedule is not None and actuator is not None:
        selected = schedule.loc[schedule["actuator"].eq(actuator)]
        lookup = {
            str(candidate): dict(zip(pd.to_datetime(group["state_decision_ts"], utc=True).astype("int64"), group["multiplier"].astype(float), strict=True))
            for candidate, group in selected.groupby("candidate_id", sort=False)
        }
    records: list[dict[str, object]] = []
    for row in route.sort_values(["timestamp", "candidate_id"], kind="stable").itertuples(index=False):
        candidate = str(row.candidate_id)
        index = int(pos[candidate])
        action = lookup.get(candidate, {})
        def modulator(state: dict[str, float], actions=action, act=actuator):
            if act is None:
                return None
            value = actions.get(int(pd.Timestamp(state["state_decision_ts"]).value), 1.0)
            key = (
                "activation_multiplier" if act == "activation"
                else "sl_distance_multiplier" if act == "stop"
                else "giveback_multiplier"
            )
            return {key: float(value)}
        trace = replay_exact_1m_gradual_h4_overlay(
            entry_price=float(arrays["entry"][index]), signal_atr=float(arrays["atr"][index]), entry_ts=rows.iloc[index]["entry_ts"],
            highs=arrays["high"][index], lows=arrays["low"][index], closes=arrays["close"][index],
            params=params, median_atr_fraction=float(median), mc1_expected_bps=float(row.bcf_mc1_expected_bps),
            state_modulator=modulator if actuator is not None else None, allow_stop_extension=True,
            max_stop_loss_fraction=.05, emit_states=False,
        )
        records.append({
            "candidate_id": candidate, "timestamp": row.timestamp, "entry_ts": rows.iloc[index]["entry_ts"], "symbol": row.symbol,
            "bcf_mc1_expected_bps": float(row.bcf_mc1_expected_bps), "current_mc1_expected_bps": float(row.current_mc1_expected_bps),
            "auction_priority_bps": float(row.auction_priority_bps), "exact_entry_price": float(arrays["entry"][index]),
            "exact_net_bps": float(trace["net_bps"]), "exact_gross_bps": float(trace["gross_bps"]),
            "exact_exit_price": float(trace["exit_price"]), "exact_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            "exact_exit_minute": int(trace["exit_minute"]), "exact_exit_reason": str(trace["exit_reason"]),
        })
    return pd.DataFrame(records)


def _extra_metrics(accepted: pd.DataFrame) -> dict[str, float | int]:
    if accepted.empty:
        return {"hit_rate": float("nan"), "cvar10_bps": float("nan"), "worst_month_bps": float("nan"), "positive_months": 0, "average_hold_minutes": float("nan")}
    values = accepted.copy()
    net = pd.to_numeric(values["net_bps"], errors="raise")
    count = max(1, int(np.ceil(len(net) * .10)))
    month = pd.to_datetime(values["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    monthly = net.groupby(month).mean()
    return {
        "hit_rate": float(net.gt(0.0).mean()),
        "cvar10_bps": float(net.nsmallest(count).mean()),
        "worst_month_bps": float(monthly.min()),
        "positive_months": int(monthly.gt(0.0).sum()),
        "average_hold_minutes": float(pd.to_numeric(values["holding_bars"], errors="raise").mean()),
    }


def _evaluate_period(
    *, rows: pd.DataFrame, arrays: dict[str, np.ndarray], route: pd.DataFrame,
    params: object, median: float, schedules: dict[str, pd.DataFrame], period: str,
    out: Path,
) -> pd.DataFrame:
    arms: list[tuple[str, str | None, pd.DataFrame | None]] = [("parent", None, None)]
    for actuator in ACTUATORS:
        for mapping, schedule in schedules.items():
            arms.append((f"{actuator}__{mapping}", actuator, schedule))
    records: list[dict[str, object]] = []
    for arm, actuator, schedule in arms:
        outcome = _replay(rows, arrays, route, params, median, schedule, actuator)
        candidates, decisions, accepted, equity, metrics = _portfolio(outcome, arm)
        # ``replay_candidates`` emits the accepted-decision ledger without
        # carrying every candidate context column.  Restore the exact
        # one-minute holding duration by candidate_index for reporting only;
        # it cannot affect acceptance, sizing, or any controller action.
        if not accepted.empty:
            accepted = accepted.copy()
            positions = pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int).to_numpy()
            accepted["holding_bars"] = candidates.iloc[positions]["holding_bars"].to_numpy()
        accepted_extra = _extra_metrics(accepted)
        outcome.to_parquet(out / f"{period}_{arm}_exact1m_outcomes.parquet", index=False, compression="zstd")
        accepted.to_parquet(out / f"{period}_{arm}_portfolio_accepted.parquet", index=False, compression="zstd")
        decisions.to_parquet(out / f"{period}_{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
        equity.to_parquet(out / f"{period}_{arm}_portfolio_equity.parquet", index=False, compression="zstd")
        records.append({"period": period, "arm": arm, **metrics, **accepted_extra})
        # Exact paths are already materialised in ``arrays``.  Release each
        # per-arm outcome/auction frame before the next controller arm so the
        # research runner cannot retain an accidental 13-arm copy.
        del outcome, candidates, decisions, accepted, equity
        gc.collect()
    table = pd.DataFrame(records)
    ref = table.loc[table["arm"].eq("parent")].iloc[0]
    for field in ("net_bps_per_trade", "total_net_bps", "sortino", "max_drawdown", "worst_week", "cvar10_bps", "worst_month_bps"):
        if field in table:
            table[f"delta_vs_parent_{field}"] = table[field] - ref[field]
    return table


def _write_monthly(out: Path, period: str) -> None:
    records: list[pd.DataFrame] = []
    for path in sorted(out.glob(f"{period}_*_portfolio_accepted.parquet")):
        arm = path.name.removeprefix(f"{period}_").removesuffix("_portfolio_accepted.parquet")
        data = pd.read_parquet(path)
        if data.empty:
            continue
        data["month"] = pd.to_datetime(data["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
        frame = data.groupby("month", as_index=False).agg(
            trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum"),
        )
        frame.insert(0, "arm", arm)
        records.append(frame)
    (pd.concat(records, ignore_index=True) if records else pd.DataFrame()).to_parquet(out / f"{period}_monthly_metrics.parquet", index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--training-threshold-bps", type=float, default=40.0)
    parser.add_argument("--evaluation-threshold-bps", type=float, default=50.0)
    parser.add_argument("--train-start", default="2025-06")
    parser.add_argument("--train-end", default="2026-01", help="exclusive; all target selection remains inside this range")
    parser.add_argument("--validate-start", default="2026-06")
    parser.add_argument("--validate-end", default="2026-09", help="exclusive")
    parser.add_argument("--state-samples-per-candidate", type=int, default=3)
    parser.add_argument("--label-workers", type=int, default=8)
    parser.add_argument("--authority-bps", type=float, default=25.0)
    parser.add_argument(
        "--resume-labels-from", type=Path,
        help="Reuse an exact counterfactual label receipt after a downstream-only failure; validates no label recomputation occurs.",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if not (0.0 < args.evaluation_threshold_bps and args.training_threshold_bps <= args.evaluation_threshold_bps):
        raise ValueError("research threshold must be positive and no stricter than evaluation threshold")
    train_start, train_end = _month(args.train_start), _month(args.train_end)
    validate_start, validate_end = _month(args.validate_start), _month(args.validate_end)
    if not (train_start < train_end <= validate_start < validate_end):
        raise ValueError("require a non-overlapping 2025 selection period and later 2026 validation")
    parent, state_root, policy_path = args.parent_root.resolve(), args.state_root.resolve(), args.policy.resolve()
    route, rows, outcomes, arrays, states = _load_parent(parent, state_root)
    params, median, _ = _load_policy(policy_path)
    fields = _state_fields(states)
    training_route = route.loc[
        route["bcf_mc1_expected_bps"].ge(args.training_threshold_bps)
        & route["current_mc1_expected_bps"].ge(args.training_threshold_bps)
        & route["timestamp"].ge(train_start) & route["timestamp"].lt(train_end)
    ].copy()
    normal_route = route.loc[
        route["bcf_mc1_expected_bps"].ge(args.evaluation_threshold_bps)
        & route["current_mc1_expected_bps"].ge(args.evaluation_threshold_bps)
    ].copy()
    if args.resume_labels_from is None:
        samples = _sample_states(
            states.loc[states["candidate_id"].isin(set(training_route["candidate_id"]))].copy(),
            maximum=args.state_samples_per_candidate,
        )
        labels = _labels(rows, arrays, (params, median), training_route, outcomes, samples, args.label_workers)
        resume_source = None
    else:
        resume_source = args.resume_labels_from.resolve()
        samples = pd.read_parquet(resume_source / "training_state_sample_target_free.parquet")
        labels = pd.read_parquet(resume_source / "exact_counterfactual_labels_training_only.parquet")
        label_ids = set(labels["candidate_id"].astype(str))
        expected_ids = set(training_route["candidate_id"].astype(str))
        if not label_ids.issubset(expected_ids):
            raise AssertionError("resumed counterfactual labels escape the frozen training route")
        if set(labels["multiplier"].astype(float).unique()) != set(MULTIPLIERS) or set(labels["actuator"].astype(str).unique()) != set(ACTUATORS):
            raise AssertionError("resumed counterfactual label contract is incomplete")
    panel = _label_panel(states, labels, epsilon=float(args.authority_bps))
    # Preserve source constraints: a row can only train if its exact parent
    # outcome resolves before the model's held boundary.  The broader route is
    # not portfolio-auctioned during label/model fitting.
    normal_train = normal_route.loc[normal_route["timestamp"].ge(train_start) & normal_route["timestamp"].lt(train_end)].copy()
    normal_validate = normal_route.loc[normal_route["timestamp"].ge(validate_start) & normal_route["timestamp"].lt(validate_end)].copy()
    if normal_train.empty or normal_validate.empty:
        raise RuntimeError("normal dual-50 evaluation population is empty")
    # The generalized one-actuator adapter must be a bit-identical parent
    # replay when every multiplier remains one.  Check a deterministic spread
    # of candidates before any counterfactual result can be trusted.
    probe = pd.concat([training_route.head(32), normal_validate.head(32)], ignore_index=True).drop_duplicates("candidate_id")
    parity = _replay(rows, arrays, probe, params, median, None, None).set_index("candidate_id")["exact_net_bps"]
    archived = outcomes.set_index("candidate_id").loc[parity.index, "parent_exact_net_bps"]
    if not np.allclose(parity.to_numpy(float), archived.to_numpy(float), rtol=0.0, atol=1e-8):
        raise AssertionError("generalized actuator adapter is not exact-parent-policy equivalent")
    out.mkdir(parents=True, exist_ok=False)
    samples.to_parquet(out / "training_state_sample_target_free.parquet", index=False, compression="zstd")
    labels.to_parquet(out / "exact_counterfactual_labels_training_only.parquet", index=False, compression="zstd")
    panel.to_parquet(out / "all_target_free_state_action_panel.parquet", index=False, compression="zstd")
    panel.loc[
        panel["entry_decision_ts"].ge(train_start) & panel["entry_decision_ts"].lt(train_end)
        & panel["wide_advantage_bps"].notna()
    ].to_parquet(out / "strict_prior_counterfactual_training_panel.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(fields)), "feature": fields}).to_parquet(out / "stage1_existing_feature_contract.parquet", index=False, compression="zstd")
    # 2025 strict OOF: monthly models are fit only on earlier resolved 2025 labels.
    predictions: list[pd.DataFrame] = []
    for period in pd.period_range(train_start, train_end - pd.offsets.MonthBegin(1), freq="M"):
        held = pd.Timestamp(period.start_time, tz="UTC")
        pred = _schedule(panel, fields, train_start=train_start, held_start=held, held_end=held + pd.offsets.MonthBegin(1), frozen_2025=False, authority_bps=float(args.authority_bps))
        pred["held_month"] = held.strftime("%Y-%m")
        predictions.append(pred)
    oof_predictions = pd.concat(predictions, ignore_index=True)
    oof_schedules = _modifiers(oof_predictions, threshold=float(args.authority_bps))
    for name, schedule in oof_schedules.items():
        schedule.to_parquet(out / f"2025_oof_{name}_schedule.parquet", index=False, compression="zstd")
    # 2026: one model per actuator fitted only on all resolved 2025 labels.
    frozen_predictions = _schedule(panel, fields, train_start=train_start, held_start=train_end, held_end=validate_end, frozen_2025=True, authority_bps=float(args.authority_bps))
    frozen_predictions = frozen_predictions.loc[
        frozen_predictions["candidate_id"].isin(set(normal_validate["candidate_id"]))
    ].copy()
    frozen_schedules = _modifiers(frozen_predictions, threshold=float(args.authority_bps))
    for name, schedule in frozen_schedules.items():
        schedule.to_parquet(out / f"2026_frozen_{name}_schedule.parquet", index=False, compression="zstd")
    # Oracle ceiling uses label rows only and never participates in portfolio evaluation.
    oracle = panel.loc[
        panel["entry_decision_ts"].ge(train_start) & panel["entry_decision_ts"].lt(train_end)
        & panel["oracle_advantage_bps"].notna()
    ].groupby("actuator", as_index=False).agg(
        states=("candidate_id", "size"), candidates=("candidate_id", "nunique"),
        mean_tight_advantage_bps=("tight_advantage_bps", "mean"), mean_wide_advantage_bps=("wide_advantage_bps", "mean"),
        mean_oracle_advantage_bps=("oracle_advantage_bps", "mean"),
        materially_adjustable_share=("oracle_advantage_bps", lambda x: float(pd.Series(x).ge(args.authority_bps).mean())),
    )
    oracle.to_parquet(out / "2025_counterfactual_oracle_diagnostics.parquet", index=False, compression="zstd")
    # Both schedule families and all label-only diagnostics are now sealed.
    # The expanded state/action panel contains three copies of the observable
    # state matrix solely for label fitting, and is not needed for the exact
    # portfolio replays.  Releasing it here preserves the identical models
    # while keeping the evaluation bounded on a normal research workstation.
    del panel, labels, samples, states, oof_predictions, frozen_predictions
    gc.collect()
    train_summary = _evaluate_period(rows=rows, arrays=arrays, route=normal_train, params=params, median=median, schedules=oof_schedules, period="2025_oof", out=out)
    train_summary.to_parquet(out / "2025_oof_portfolio_summary.parquet", index=False, compression="zstd")
    _write_monthly(out, "2025_oof")
    validation_summary = _evaluate_period(rows=rows, arrays=arrays, route=normal_validate, params=params, median=median, schedules=frozen_schedules, period="2026_validation", out=out)
    validation_summary.to_parquet(out / "2026_validation_portfolio_summary.parquet", index=False, compression="zstd")
    _write_monthly(out, "2026_validation")
    manifest = {
        "schema": "causal-sr-h4-actuator-counterfactual-stage1-v1",
        "scope": "offline research only; no live, exchange, model-admission, or portfolio-contract mutation",
        "source_link": "https://chatgpt.com/s/t_6a95e8b6f264819198ea7ee5e2834ff6",
        "parent_root": str(parent), "parent_manifest_sha256": _sha256(parent / "run_manifest.json"),
        "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json"),
        "policy": str(policy_path), "policy_sha256": _sha256(policy_path),
        "source_contract": "causal S/R and paired BCF/current MC1 target-free route; exact one-minute rich parent; long-only",
        "training_population": f"dual MC1 >= {args.training_threshold_bps:g} bps, no portfolio constraints, first/middle/last target-free completed states, labels resolved after H12",
        "resumed_labels_from": str(resume_source) if resume_source is not None else None,
        "assessment_population": f"dual MC1 >= {args.evaluation_threshold_bps:g} bps, normal chronological portfolio auction only",
        "selection_window": [train_start.isoformat(), train_end.isoformat()],
        "validation_window": [validate_start.isoformat(), validate_end.isoformat()],
        "features": "all pre-existing numeric target-free H4 state fields; no 2026-selected feature subset is used to assess 2025",
        "counterfactual": "exact one-minute rich parent replay; one actuator varied at a time; multipliers .65/.80/1.00/1.25/1.50; stop extension hard-capped at 5% entry distance",
        "stage1_models": "per-actuator shallow LGBM d3/l7/min-child-5%/L2-40/LR-.035/280; dual L2 advantages and 5-class ordinal control",
        "mappings": "dual widen-only, tighten-only, asymmetric dead-zone, ordinal stepped; all other actuators remain parent baseline",
        "causality": "2025 schedules are monthly strict-prior OOF; 2026 schedules are frozen from 2025 labels only; all state actions begin next interval",
        "portfolio": "unchanged normal global chronological constrained auction; training is explicitly unauctioned",
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
