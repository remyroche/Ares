#!/usr/bin/env python3
"""Strict-prequential H4 refit on the updated C1 population.

The older H4 model was trained on the legacy E2/BCF reserve route.  This
research-only runner rebuilds its activation-50 labels from the corrected
exact-one-minute C1 paths themselves.  A July model sees only June labels;
an August model sees only June--July labels resolved before its month.  H4
actions remain tightening-only and are evaluated with the normal chronological
portfolio auction.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import replay_exact_1m_h4_giveback20
from scripts.run_causal_sr_c1_exact1m_parent_portfolio_replay import _common_target_free_routes
from scripts.run_causal_sr_c1_h4_exact1m_transfer_replay import (
    DEFAULT_AUG,
    DEFAULT_FEATURE_CONTRACT,
    DEFAULT_JUNJUL,
    DEFAULT_PARENT_ROOT,
    DEFAULT_POLICY,
    _fields,
    _portfolio,
    _raw_states,
    _state_features,
)

DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_c1_h4_exact1m_refit_20260831_v1"
ARM = "C1_refit_core_plus_causal_sr"
H12 = pd.Timedelta(hours=12)
_LABEL_CONTEXT: dict[str, object] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(train: pd.DataFrame, fields: tuple[str, ...], loss: str) -> lgb.LGBMRegressor:
    """Fit the frozen H4 geometry with a sparse-label-safe central tendency.

    The latching counterfactual retains a large structural point mass at zero.
    L1 therefore estimates zero almost everywhere and has no usable authority.
    L2 is deliberately the only alternative here: it estimates the conditional
    expected incremental economics without changing tree geometry, fields,
    weighting, or the tightening-only authority.
    """
    if loss not in {"l1", "l2"}:
        raise ValueError(f"unsupported H4 loss: {loss}")
    child = max(8, int(np.ceil(len(train) * 0.05)))
    model = lgb.LGBMRegressor(
        objective="regression_l2" if loss == "l2" else "regression_l1",
        n_estimators=420,
        learning_rate=0.025,
        max_depth=4,
        num_leaves=15,
        min_child_samples=child,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=20.0,
        random_state=1729,
        n_jobs=2,
        verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(
        train.loc[:, fields],
        train["activation50_advantage_bps"].to_numpy(float),
        sample_weight=weights,
    )
    return model


def _label_candidate(task: tuple[int, tuple[int, ...], float]) -> list[dict[str, object]]:
    """Exact action-only advantages for every parent-observable state of one trade."""
    position, action_ns, parent_net = task
    rows = _LABEL_CONTEXT["rows"]
    arrays = _LABEL_CONTEXT["arrays"]
    params, median = _LABEL_CONTEXT["policy"]
    mode = str(_LABEL_CONTEXT["target_mode"])
    target_giveback_tighten = float(_LABEL_CONTEXT["target_giveback_tighten"])
    row = rows.iloc[position]
    records: list[dict[str, object]] = []
    for target in action_ns:
        def decide(state: dict[str, float], target_ns: int = target) -> bool:
            state_ns = int(pd.Timestamp(state["state_decision_ts"]).value)
            if mode == "one_interval_activation50":
                return state_ns == target_ns
            if mode == "latched_activation50_giveback20":
                # A positive action is a permanent tightening for this position:
                # no future state can undo the earlier activation or giveback lock.
                return state_ns >= target_ns
            raise ValueError(f"unsupported target mode: {mode}")

        trace = replay_exact_1m_h4_giveback20(
            entry_price=float(arrays["entry"][position]),
            signal_atr=float(arrays["atr"][position]),
            entry_ts=row["entry_ts"],
            highs=arrays["high"][position],
            lows=arrays["low"][position],
            closes=arrays["close"][position],
            params=params,
            median_atr_fraction=float(median),
            mc1_expected_bps=0.0,
            state_decider=decide,
            giveback_tighten=target_giveback_tighten,
            emit_states=False,
        )
        records.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "state_decision_ts": pd.Timestamp(target, tz="UTC"),
                "activation50_net_bps": float(trace["net_bps"]),
                "parent_exact_net_bps": float(parent_net),
                "activation50_advantage_bps": float(trace["net_bps"] - parent_net),
                "policy_label_available_ts": pd.Timestamp(row["entry_ts"]) + H12,
            }
        )
    return records


def _labels(
    rows: pd.DataFrame,
    raw: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    policy: tuple[object, float],
    route: pd.DataFrame,
    parent_outcomes: pd.DataFrame,
    workers: int,
    target_mode: str,
    target_giveback_tighten: float,
) -> pd.DataFrame:
    route_ids = set(route["candidate_id"].astype(str))
    parent = parent_outcomes.set_index("candidate_id")["exact_net_bps"].astype(float)
    position = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    task_rows: list[tuple[int, tuple[int, ...], float]] = []
    for candidate_id, group in raw.loc[raw["candidate_id"].astype(str).isin(route_ids)].groupby("candidate_id", sort=True):
        candidate = str(candidate_id)
        if candidate not in parent.index:
            continue
        states = tuple(pd.to_datetime(group["state_decision_ts"], utc=True).view("int64").tolist())
        task_rows.append((int(position[candidate]), states, float(parent.loc[candidate])))
    if not task_rows:
        raise RuntimeError("no updated C1 states have complete exact parent labels")

    _LABEL_CONTEXT.clear()
    _LABEL_CONTEXT.update({
        "rows": rows,
        "arrays": arrays,
        "policy": policy,
        "target_mode": str(target_mode),
        "target_giveback_tighten": float(target_giveback_tighten),
    })
    labelled: list[dict[str, object]] = []
    # Fork safely shares the immutable path arrays without serialising 720-minute
    # vectors into every job.  This is research-only and has no external I/O.
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=context) as executor:
        for ordinal, result in enumerate(executor.map(_label_candidate, task_rows), start=1):
            labelled.extend(result)
            if ordinal % 100 == 0 or ordinal == len(task_rows):
                print(f"updated-H4 labels {ordinal}/{len(task_rows)} candidates", flush=True)
    target = pd.DataFrame(labelled)
    keys = ["candidate_id", "state_decision_ts"]
    if target.duplicated(keys).any():
        raise AssertionError("updated H4 target duplicated a state identity")
    return target


def _actions(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    route: pd.DataFrame,
    contract: Path,
    loss: str,
    state_gate: str,
    mfe_ready_atr: float,
) -> pd.DataFrame:
    keys = ["candidate_id", "state_decision_ts"]
    scored = states.merge(
        labels.loc[:, [*keys, "activation50_advantage_bps", "policy_label_available_ts"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    scored = scored.merge(
        route.loc[:, ["candidate_id", "bcf_mc1_expected_bps"]],
        on="candidate_id",
        how="inner",
        validate="many_to_one",
    )
    scored["MC1_expected_bps"] = pd.to_numeric(scored.pop("bcf_mc1_expected_bps"), errors="raise")
    scored["entry_decision_ts"] = pd.to_datetime(scored["entry_decision_ts"], utc=True, errors="raise")
    if state_gate == "all":
        scored["h4_actionable_state"] = True
    elif state_gate == "mfe_ready":
        # This is a decision-time mechanical condition: the running MFE has
        # reached the exact 50%-earlier trailing-activation level.  It does
        # not inspect the future path or the realised label.
        scored["h4_actionable_state"] = pd.to_numeric(
            scored["current_MFE_ATR"], errors="coerce"
        ).ge(float(mfe_ready_atr))
    else:
        raise ValueError(f"unsupported H4 state gate: {state_gate}")
    actions: list[pd.DataFrame] = []
    for period in sorted(scored["entry_decision_ts"].dt.to_period("M").unique()):
        held = pd.Timestamp(period.start_time, tz="UTC")
        end = held + pd.offsets.MonthBegin(1)
        test = scored.loc[
            scored["entry_decision_ts"].ge(held) & scored["entry_decision_ts"].lt(end)
        ].copy()
        test["h4_predicted_activation50_advantage_bps"] = float("-inf")
        eligible_test = test.loc[test["h4_actionable_state"]].copy()
        # June intentionally has no current-population predecessor labels.
        if held <= pd.Timestamp("2026-06-01", tz="UTC"):
            pass
        else:
            fields = _fields(contract, held)
            train = scored.loc[
                scored["entry_decision_ts"].lt(held)
                & scored["policy_label_available_ts"].lt(held)
                & scored["activation50_advantage_bps"].notna()
                & scored["h4_actionable_state"]
            ].copy()
            missing = set(fields).difference(train.columns) | set(fields).difference(eligible_test.columns)
            if missing:
                raise AssertionError(f"{held:%Y-%m}: missing C4 field(s) {sorted(missing)}")
            if train["candidate_id"].nunique() < 100 and not eligible_test.empty:
                raise RuntimeError(f"{held:%Y-%m}: inadequate current-population H4 support")
            if not eligible_test.empty:
                model = _fit(train, fields, loss)
                test.loc[eligible_test.index, "h4_predicted_activation50_advantage_bps"] = model.predict(
                    eligible_test.loc[:, fields]
                )
        action = test.loc[:, ["candidate_id", "state_decision_ts", "h4_predicted_activation50_advantage_bps"]].copy()
        # A zero prediction carries no evidence of incremental value.  This
        # strict gate prevents a zero-inflated target from silently enabling
        # the action everywhere.
        action["h4_enable"] = action["h4_predicted_activation50_advantage_bps"].gt(0.0)
        action["held_month"] = held.strftime("%Y-%m")
        actions.append(action)
    result = pd.concat(actions, ignore_index=True)
    if result.duplicated(keys).any():
        raise AssertionError("updated H4 schedule duplicated state identity")
    return result


def _outcomes(
    rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    route: pd.DataFrame,
    actions: pd.DataFrame,
    policy: tuple[object, float],
    giveback_tighten: float,
    latch: bool,
) -> pd.DataFrame:
    params, median = policy
    lookup = {
        str(candidate): set(pd.to_datetime(group["state_decision_ts"], utc=True).view("int64").tolist())
        for candidate, group in actions.loc[actions["h4_enable"]].groupby("candidate_id", sort=False)
    }
    position = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    records: list[dict[str, object]] = []
    for _, scored in route.sort_values(["timestamp", "candidate_id"], kind="stable").iterrows():
        candidate = str(scored["candidate_id"])
        if candidate not in position.index:
            continue
        index = int(position[candidate])
        active = lookup.get(candidate, set())
        latched = False
        def decide(state: dict[str, float], selected: set[int] = active) -> bool:
            nonlocal latched
            active_now = int(pd.Timestamp(state["state_decision_ts"]).value) in selected
            if latch:
                latched = latched or active_now
                return latched
            return active_now
        trace = replay_exact_1m_h4_giveback20(
            entry_price=float(arrays["entry"][index]),
            signal_atr=float(arrays["atr"][index]),
            entry_ts=rows.iloc[index]["entry_ts"],
            highs=arrays["high"][index],
            lows=arrays["low"][index],
            closes=arrays["close"][index],
            params=params,
            median_atr_fraction=float(median),
            mc1_expected_bps=float(scored["bcf_mc1_expected_bps"]),
            state_decider=decide,
            giveback_tighten=float(giveback_tighten),
            emit_states=False,
        )
        records.append(
            {
                "candidate_id": candidate,
                "timestamp": scored["timestamp"],
                "entry_ts": rows.iloc[index]["entry_ts"],
                "symbol": scored["symbol"],
                "bcf_mc1_expected_bps": float(scored["bcf_mc1_expected_bps"]),
                "auction_priority_bps": float(scored["auction_priority_bps"]),
                "exact_entry_price": float(arrays["entry"][index]),
                "exact_net_bps": float(trace["net_bps"]),
                "exact_gross_bps": float(trace["gross_bps"]),
                "exact_exit_price": float(trace["exit_price"]),
                "exact_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
                "exact_exit_minute": int(trace["exit_minute"]),
                "exact_exit_reason": str(trace["exit_reason"]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--junjul-root", type=Path, default=DEFAULT_JUNJUL)
    parser.add_argument("--august-root", type=Path, default=DEFAULT_AUG)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--label-workers", type=int, default=6)
    parser.add_argument(
        "--target-mode",
        choices=("one_interval_activation50", "latched_activation50_giveback20"),
        default="latched_activation50_giveback20",
    )
    parser.add_argument(
        "--state-gate",
        choices=("all", "mfe_ready"),
        default="mfe_ready",
        help="Causal state support gate; mfe_ready requires MFE >= the 50%%-earlier activation point.",
    )
    parser.add_argument(
        "--loss",
        choices=("l1", "l2"),
        default="l2",
        help="H4 refit loss; L2 is the sparse-latched-target default.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    parent_root = args.parent_root.resolve()
    rows, raw, arrays, policy = _raw_states(parent_root, args.policy.resolve())
    routes = _common_target_free_routes(args.junjul_root.resolve(), args.august_root.resolve())
    route = routes[ARM].copy()
    states, state_coverage = _state_features(raw.loc[raw["candidate_id"].astype(str).isin(set(route["candidate_id"].astype(str)))].copy())
    parent_outcomes = pd.read_parquet(parent_root / "exact_1m_rich_parent_outcomes.parquet")
    parent_outcomes["candidate_id"] = parent_outcomes["candidate_id"].astype(str)
    target_giveback_tighten = .20 if args.target_mode == "latched_activation50_giveback20" else 0.0
    label = _labels(
        rows, raw, arrays, policy, route, parent_outcomes, args.label_workers,
        args.target_mode, target_giveback_tighten,
    )
    params, _ = policy
    mfe_ready_atr = 0.5 * float(params.protection_activation_atr)
    action = _actions(
        states, label, route, args.feature_contract.resolve(), args.loss,
        args.state_gate, mfe_ready_atr,
    )
    out.mkdir(parents=True, exist_ok=False)
    label.to_parquet(out / "updated_population_activation50_advantage_labels.parquet", index=False, compression="zstd")
    states.to_parquet(out / "updated_population_h4_context_states_target_free.parquet", index=False, compression="zstd")
    state_coverage.to_parquet(out / "updated_population_h4_state_source_coverage.parquet", index=False, compression="zstd")
    action.to_parquet(out / "updated_population_h4_action_schedule_target_free.parquet", index=False, compression="zstd")

    parent_summary = pd.read_parquet(parent_root / "portfolio_summary.parquet").set_index("arm").loc[ARM].to_dict()
    summary_rows = [{"arm": "C1_parent_exact1m", **parent_summary}]
    for name, tighten in (("C1_h4_refit_activation50", 0.0), ("C1_h4_refit_activation50_giveback20", 0.20)):
        outcome = _outcomes(
            rows, arrays, route, action, policy, tighten,
            latch=args.target_mode == "latched_activation50_giveback20",
        )
        candidates, decisions, accepted, equity, metrics = _portfolio(outcome, name)
        outcome.to_parquet(out / f"{name}_exact1m_outcomes.parquet", index=False, compression="zstd")
        candidates.to_parquet(out / f"{name}_portfolio_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(out / f"{name}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(out / f"{name}_portfolio_accepted.parquet", index=False, compression="zstd")
        equity.to_parquet(out / f"{name}_portfolio_equity.parquet", index=False, compression="zstd")
        monthly = accepted.assign(month=pd.to_datetime(accepted["decision_timestamp"], utc=True).dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
            trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum")
        )
        monthly.to_parquet(out / f"{name}_monthly_portfolio_metrics.parquet", index=False, compression="zstd")
        summary_rows.append({"arm": name, **metrics})
    summary = pd.DataFrame(summary_rows)
    reference = summary.loc[summary["arm"].eq("C1_parent_exact1m")].iloc[0]
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
        if field in summary:
            summary[f"delta_vs_parent_{field}"] = summary[field] - reference[field]
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "causal-sr-c1-h4-exact1m-refit-v2",
        "scope": "offline research-only updated-population H4 refit; exact 1m exit paths and normal global portfolio auction",
        "parent_root": str(parent_root),
        "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"),
        "target": (
            "per completed 15m parent-observable state, exact continuation-policy net "
            "minus exact unchanged-parent net; action begins following interval"
        ),
        "target_mode": str(args.target_mode),
        "target_giveback_tighten": target_giveback_tighten,
        "state_gate": str(args.state_gate),
        "mfe_ready_atr": mfe_ready_atr,
        "prequential": "June receives no C1 H4 authority; July fits June labels resolved before July; August fits June--July labels resolved before August",
        "feature_contract": str(args.feature_contract.resolve()),
        "feature_contract_sha256": _sha256(args.feature_contract.resolve()),
        "policy": str(args.policy.resolve()),
        "policy_sha256": _sha256(args.policy.resolve()),
        "h4_model": (
            f"H4 {args.loss.upper()} d4/l15/min-child-5%/L2-20/LR-.025/420 "
            "retrained only on updated C1 labels"
        ),
        "authority": (
            "latched target actions persist to trade exit when selected; comparison "
            "separately evaluates 0% and 20% giveback tightening; never loosens "
            "stop or prior protection; action requires strictly positive predicted advantage"
        ),
        "portfolio": "global chronological controlled 7x/10%-margin slot, two new entries per timestamp, eight concurrent, 80%-wallet budget",
        "no_exchange_calls": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
