#!/usr/bin/env python3
"""Exact next-15m, bidirectional H4 actuator ablation (offline research only).

For one actuator (activation, giveback, or stop), labels are the exact H12
policy difference from a *temporary* action at a completed 15-minute state:
the multiplier is applied for that next interval and reset to the rich parent
setting at the following state if the trade remains open.  The same causal
state schedule is then assessed under the normal constrained portfolio route.

2025 is the only selection period.  The selected 2025 arm is fitted from
resolved 2025 labels and checked once on 2026.  This runner never imports or
mutates live/exchange, C1 S/R, MC1, admission, geometry, or portfolio code.
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


MULTIPLIERS = (.65, .80, 1.00, 1.25, 1.50)
H12 = pd.Timedelta(hours=12)
DEFAULT_LABEL_ROOT = Path("data_perp/artifacts/causal_sr_h4_actuator_counterfactual_2025oof_2026confirm_20260901_v1")

LABEL_CONTEXT: dict[str, object] = {}


def _label_column(multiplier: float) -> str:
    """Stable target name for an exact temporary-action multiplier."""
    value = f"{float(multiplier):.4f}".rstrip("0").rstrip(".")
    return f"adv_{value.replace('-', 'm').replace('.', 'p')}"


def _action_targets() -> tuple[str, ...]:
    return tuple(_label_column(value) for value in MULTIPLIERS if not np.isclose(value, 1.0))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _sample_states(states: pd.DataFrame, eligible: set[str]) -> pd.DataFrame:
    """First/middle/last target-free state per complete-path candidate."""
    pieces: list[pd.DataFrame] = []
    for _, g in states.loc[states["candidate_id"].isin(eligible)].groupby("candidate_id", sort=True):
        g = g.sort_values("state_decision_ts", kind="stable")
        take = np.unique(np.linspace(0, len(g) - 1, min(3, len(g)), dtype=int))
        pieces.append(g.iloc[take])
    out = pd.concat(pieces, ignore_index=True) if pieces else states.iloc[:0].copy()
    if out.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("state sample duplicated")
    return out


def _label_candidate(task: tuple[int, tuple[int, ...], float, str]) -> list[dict[str, object]]:
    position, state_ns, parent_net, actuator = task
    rows = LABEL_CONTEXT["rows"]
    arrays = LABEL_CONTEXT["arrays"]
    params, median = LABEL_CONTEXT["policy"]
    assert isinstance(rows, pd.DataFrame) and isinstance(arrays, dict)
    row = rows.iloc[position]
    parent_outcomes = LABEL_CONTEXT["parent_outcomes"]
    assert isinstance(parent_outcomes, pd.DataFrame)
    parent_trace = parent_outcomes.loc[str(row["candidate_id"])]
    parent_exit_price = float(parent_trace["exact_exit_price"])
    parent_exit_timestamp = _utc(parent_trace["exact_exit_ts"])
    parent_exit_minute = int(parent_trace["exact_exit_minute"])
    parent_exit_reason = str(parent_trace["exact_exit_reason"])
    key = (
        "activation_multiplier" if actuator == "activation"
        else "sl_distance_multiplier" if actuator == "stop"
        else "giveback_multiplier"
    )
    results: list[dict[str, object]] = []
    for target in state_ns:
        for multiplier in MULTIPLIERS:
            if np.isclose(multiplier, 1.0):
                value = parent_net
                counterfactual_exit_price = parent_exit_price
                counterfactual_exit_timestamp = parent_exit_timestamp
                counterfactual_exit_minute = parent_exit_minute
                counterfactual_exit_reason = parent_exit_reason
            else:
                def modulator(state: dict[str, float], t: int = target, m: float = multiplier, field: str = key):
                    now = int(pd.Timestamp(state["state_decision_ts"]).value)
                    if now < t:
                        return None
                    # The action starts immediately after the target state and
                    # is explicitly neutralised at the next completed state.
                    return {field: float(m) if now == t else 1.0}
                trace = replay_exact_1m_gradual_h4_overlay(
                    entry_price=float(arrays["entry"][position]), signal_atr=float(arrays["atr"][position]), entry_ts=row["entry_ts"],
                    highs=arrays["high"][position], lows=arrays["low"][position], closes=arrays["close"][position],
                    params=params, median_atr_fraction=float(median), mc1_expected_bps=0.0,
                    state_modulator=modulator, allow_stop_extension=True, max_stop_loss_fraction=.05, emit_states=False,
                )
                value = float(trace["net_bps"])
                counterfactual_exit_price = float(trace["exit_price"])
                counterfactual_exit_timestamp = _utc(trace["exit_timestamp"])
                counterfactual_exit_minute = int(trace["exit_minute"])
                counterfactual_exit_reason = str(trace["exit_reason"])
            exit_changed = bool(
                counterfactual_exit_minute != parent_exit_minute
                or counterfactual_exit_reason != parent_exit_reason
                or not np.isclose(counterfactual_exit_price, parent_exit_price, atol=1e-12, rtol=0.0)
            )
            results.append({
                "candidate_id": str(row["candidate_id"]), "state_decision_ts": pd.Timestamp(target, tz="UTC"),
                "actuator": actuator, "multiplier": float(multiplier), "counterfactual_net_bps": float(value),
                "parent_exact_net_bps": float(parent_net), "advantage_bps": float(value - parent_net),
                "counterfactual_exit_price": counterfactual_exit_price,
                "counterfactual_exit_ts": counterfactual_exit_timestamp,
                "counterfactual_exit_minute": counterfactual_exit_minute,
                "counterfactual_exit_reason": counterfactual_exit_reason,
                "parent_exit_price": parent_exit_price,
                "parent_exit_ts": parent_exit_timestamp,
                "parent_exit_minute": parent_exit_minute,
                "parent_exit_reason": parent_exit_reason,
                "exit_changed": exit_changed,
                "policy_label_available_ts": pd.Timestamp(row["entry_ts"]) + H12,
            })
    return results


def _labels(rows: pd.DataFrame, arrays: dict[str, np.ndarray], policy: tuple[object, float], route: pd.DataFrame, outcomes: pd.DataFrame, sample: pd.DataFrame, actuator: str, workers: int) -> pd.DataFrame:
    parent = outcomes.set_index("candidate_id")["parent_exact_net_bps"]
    pos = pd.Series(np.arange(len(rows)), index=rows["candidate_id"].astype(str))
    tasks: list[tuple[int, tuple[int, ...], float, str]] = []
    ids = set(route["candidate_id"].astype(str))
    for candidate, g in sample.loc[sample["candidate_id"].isin(ids)].groupby("candidate_id", sort=True):
        c = str(candidate)
        if c in pos.index and c in parent.index:
            tasks.append((int(pos[c]), tuple(pd.to_datetime(g["state_decision_ts"], utc=True).astype("int64").tolist()), float(parent.loc[c]), actuator))
    if not tasks:
        raise RuntimeError("no complete candidates for temporary-action labels")
    trace_fields = ["candidate_id", "exact_exit_price", "exact_exit_ts", "exact_exit_minute", "exact_exit_reason"]
    parent_outcomes = outcomes.loc[:, trace_fields].copy().set_index("candidate_id", verify_integrity=True)
    if parent_outcomes.loc[list(ids)].isna().any().any():
        raise AssertionError("parent exit trace is incomplete for a temporary-action label")
    LABEL_CONTEXT.clear(); LABEL_CONTEXT.update({"rows": rows, "arrays": arrays, "policy": policy, "parent_outcomes": parent_outcomes})
    result: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("fork")) as ex:
        for i, block in enumerate(ex.map(_label_candidate, tasks), 1):
            result.extend(block)
            if i == 1 or i % 500 == 0 or i == len(tasks):
                print(f"next15 {actuator} labels {i}/{len(tasks)}", flush=True)
    out = pd.DataFrame(result)
    if out.duplicated(["candidate_id", "state_decision_ts", "multiplier"]).any():
        raise AssertionError("temporary-action label duplicated")
    if not np.allclose(out.loc[out["multiplier"].eq(1.0), "advantage_bps"], 0.0):
        raise AssertionError("temporary neutral action is not exact parent")
    neutral = out.loc[out["multiplier"].eq(1.0), "exit_changed"]
    if neutral.any():
        raise AssertionError("temporary neutral action changed the parent exit")
    return out


def _label_states(states: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    key = ["candidate_id", "state_decision_ts"]
    pivot = labels.pivot(index=key, columns="multiplier", values="advantage_bps").reset_index()
    found = {float(c) for c in pivot.columns if isinstance(c, float)}
    if found != set(MULTIPLIERS):
        raise AssertionError("temporary label grid incomplete")
    availability = labels.loc[:, [*key, "policy_label_available_ts"]].drop_duplicates(key)
    result = states.merge(pivot, on=key, how="inner", validate="one_to_one")
    result = result.merge(availability, on=key, how="inner", validate="one_to_one")
    result.rename(
        columns={float(value): _label_column(float(value)) for value in MULTIPLIERS if not np.isclose(value, 1.0)},
        inplace=True,
    )
    return result


def _fit(train: pd.DataFrame, fields: tuple[str, ...], target: str, model_config: dict[str, float | int | str]) -> lgb.LGBMRegressor:
    child = max(64, int(np.ceil(len(train) * float(model_config["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective=str(model_config["objective"]), n_estimators=280, learning_rate=.035,
        max_depth=int(model_config["max_depth"]), num_leaves=int(model_config["num_leaves"]),
        min_child_samples=child, subsample=.80, colsample_bytree=.80, reg_lambda=float(model_config["reg_lambda"]),
        alpha=float(model_config["quantile_alpha"]),
        random_state=1729, n_jobs=2, verbosity=-1,
    )
    values = train[target].to_numpy(float)
    weight = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    weight *= np.where(values > 0.0, float(model_config["positive_weight"]), 1.0)
    model.fit(train.loc[:, fields], values, sample_weight=weight)
    return model


def _schedule(labels: pd.DataFrame, target_states: pd.DataFrame, fields: tuple[str, ...], model_config: dict[str, float | int | str], *, start: pd.Timestamp, held_start: pd.Timestamp, held_end: pd.Timestamp) -> pd.DataFrame:
    test = target_states.loc[target_states["entry_decision_ts"].ge(held_start) & target_states["entry_decision_ts"].lt(held_end)].copy()
    cols = ["candidate_id", "state_decision_ts"]
    result = test.loc[:, cols].copy()
    for target in _action_targets():
        result[f"pred_{target}"] = 0.0
    train = labels.loc[
        labels["entry_decision_ts"].ge(start) & labels["entry_decision_ts"].lt(held_start)
        & labels["policy_label_available_ts"].lt(held_start)
    ].copy()
    if test.empty or train["candidate_id"].nunique() < 250:
        return result
    for target in _action_targets():
        result[f"pred_{target}"] = _fit(train, fields, target, model_config).predict(test.loc[:, fields])
    return result


def _oof(labels: pd.DataFrame, target_states: pd.DataFrame, fields: tuple[str, ...], model_config: dict[str, float | int | str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in pd.period_range(start, end - pd.offsets.MonthBegin(1), freq="M"):
        held = pd.Timestamp(month.start_time, tz="UTC")
        one = _schedule(labels, target_states, fields, model_config, start=start, held_start=held, held_end=held + pd.offsets.MonthBegin(1))
        one["held_month"] = held.strftime("%Y-%m")
        pieces.append(one)
    return pd.concat(pieces, ignore_index=True)


def _mapping(pred: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    values = {value: pred[f"pred_{_label_column(value)}"].to_numpy(float) for value in MULTIPLIERS if not np.isclose(value, 1.0)}
    tight = tuple(value for value in MULTIPLIERS if value < 1.0)
    wide = tuple(value for value in MULTIPLIERS if value > 1.0)
    if mode == "tight_gradual":
        if not tight:
            raise AssertionError("gradual tightening requires at least one multiplier below one")
        # A continuous, bounded authority curve.  A non-positive predicted
        # advantage leaves the parent policy untouched; confidence reaches
        # full authority only at the predeclared scale.  The controller acts
        # for one next 15m interval and never loosens protection.
        strongest = min(tight)
        confidence = np.clip(values[strongest] / float(threshold), 0.0, 1.0)
        selected = 1.0 + (float(strongest) - 1.0) * confidence
        out = pred.loc[:, ["candidate_id", "state_decision_ts"]].copy()
        out["actuator"] = "placeholder"
        out["multiplier"] = selected
        out["confidence"] = confidence
        return out
    allowed = tight if mode == "tight" else wide if mode == "wide" else tuple(values)
    if not allowed:
        raise AssertionError(f"{mode} authority has no non-neutral multipliers")
    matrix = np.column_stack([values[m] for m in allowed])
    idx = matrix.argmax(axis=1); best = matrix[np.arange(len(matrix)), idx]
    selected = np.asarray(allowed, dtype=float)[idx]
    selected[best < float(threshold)] = 1.0
    out = pred.loc[:, ["candidate_id", "state_decision_ts"]].copy()
    out["actuator"] = "placeholder"  # overwritten by caller; makes receipt explicit
    out["multiplier"] = selected
    return out


def _evaluate(rows: pd.DataFrame, arrays: dict[str, np.ndarray], route: pd.DataFrame, params: object, median: float, schedule: pd.DataFrame | None, actuator: str, arm: str, out: Path) -> dict[str, object]:
    outcome = base._replay(rows, arrays, route, params, median, schedule, actuator if schedule is not None else None)
    candidates, decisions, accepted, equity, metrics = base._portfolio(outcome, arm)
    if not accepted.empty:
        accepted = accepted.copy(); ix = pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int).to_numpy()
        accepted["holding_bars"] = candidates.iloc[ix]["holding_bars"].to_numpy()
    extra = base._extra_metrics(accepted)
    for name, data in (("exact1m_outcomes", outcome), ("portfolio_accepted", accepted), ("portfolio_decisions", decisions), ("portfolio_equity", equity)):
        data.to_parquet(out / f"{arm}_{name}.parquet", index=False, compression="zstd")
    del outcome, candidates, decisions, accepted, equity; gc.collect()
    return {"arm": arm, **metrics, **extra}


def _monthly(out: Path, arms: list[str], name: str) -> None:
    frames=[]
    for arm in arms:
        p=out/f"{arm}_portfolio_accepted.parquet"
        if not p.exists(): continue
        d=pd.read_parquet(p)
        if d.empty: continue
        d["month"]=pd.to_datetime(d["decision_timestamp"],utc=True).dt.strftime("%Y-%m")
        x=d.groupby("month",as_index=False).agg(trades=("candidate_id","size"),net_bps_per_trade=("net_bps","mean"),total_net_bps=("net_bps","sum"))
        x.insert(0,"arm",arm); frames.append(x)
    (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()).to_parquet(out/name,index=False,compression="zstd")


def _threshold_tag(value: float) -> str:
    """Stable file-safe authority label; threshold is in predicted bps."""
    return (f"{value:g}".replace("-", "m").replace(".", "p"))


def _authority_defs(thresholds: tuple[float, ...], gradual_scales: tuple[float, ...]) -> tuple[tuple[str, str, float], ...]:
    """Test both directions independently and together at each fixed gate."""
    fixed = tuple(
        (f"{mode}_b{_threshold_tag(threshold)}", mode, threshold)
        for threshold in thresholds
        for mode in ("tight", "wide", "asym")
    )
    gradual = tuple(
        (f"tight_gradual_c{_threshold_tag(scale)}", "tight_gradual", scale)
        for scale in gradual_scales
    )
    return fixed + gradual


def main() -> None:
    global MULTIPLIERS
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actuator",choices=("activation","giveback","stop"),required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--label-root",type=Path,default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--parent-root",type=Path,default=base.DEFAULT_PARENT)
    ap.add_argument("--state-root",type=Path,default=base.DEFAULT_STATES)
    ap.add_argument("--extra-feature-panel",type=Path,
                    help="Optional exact-keyed target-free H4 feature panel. It is merged before fitting only; no live contract changes.")
    ap.add_argument("--policy",type=Path,default=base.DEFAULT_POLICY)
    ap.add_argument("--label-workers",type=int,default=8)
    ap.add_argument(
        "--thresholds", default="15,25,35",
        help="Comma-separated predicted temporary-advantage gates in bps; selected only in 2025 OOF.",
    )
    ap.add_argument(
        "--multipliers", default="0.65,0.8,1.0,1.25,1.5",
        help="Comma-separated temporary actuator multipliers. Must include neutral 1.0; label-only runs may use an extended envelope.",
    )
    ap.add_argument(
        "--gradual-confidence-scales", default="",
        help="Optional predicted-advantage bps scales for a monotone continuous tightening curve; selected only in 2025 OOF.",
    )
    ap.add_argument(
        "--resume-labels-from", type=Path,
        help="Reuse immutable exact next-15m labels after a downstream-only authority-grid change.",
    )
    ap.add_argument(
        "--label-only", action="store_true",
        help="Materialise and validate exact labels only; useful to predeclare a sensible 2025-only authority grid.",
    )
    ap.add_argument("--objective", default="regression_l2", choices=("regression_l2", "regression_l1", "huber", "quantile"))
    ap.add_argument("--quantile-alpha", type=float, default=.20,
                    help="Lower quantile for objective=quantile; used as a conservative action-value lower confidence bound.")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--num-leaves", type=int, default=7)
    ap.add_argument("--min-child-fraction", type=float, default=.05)
    ap.add_argument("--reg-lambda", type=float, default=40.)
    ap.add_argument("--positive-weight", type=float, default=1.)
    args=ap.parse_args(); out=args.out.resolve()
    if out.exists(): raise FileExistsError(f"immutable output exists: {out}")
    parent,state_root,policy=args.parent_root.resolve(),args.state_root.resolve(),args.policy.resolve()
    route,rows,outcomes,arrays,states=base._load_parent(parent,state_root)
    params,median,_=base._load_policy(policy); control_fields=base._state_fields(states)
    extra_fields: tuple[str,...]=()
    if args.extra_feature_panel is not None:
        extra_path=args.extra_feature_panel.resolve(); extra=pd.read_parquet(extra_path).copy()
        key=["candidate_id","state_decision_ts"]
        if set(key).difference(extra.columns): raise AssertionError("extra feature panel lacks state identity")
        extra["candidate_id"]=extra["candidate_id"].astype(str); extra["state_decision_ts"]=pd.to_datetime(extra["state_decision_ts"],utc=True,errors="raise")
        if extra.duplicated(key).any(): raise AssertionError("extra feature panel duplicates state identity")
        expected=states.loc[:,key].copy()
        if len(extra)!=len(expected) or not extra.loc[:,key].sort_values(key).reset_index(drop=True).equals(expected.sort_values(key).reset_index(drop=True)):
            raise AssertionError("extra feature panel is not exactly aligned to target-free H4 states")
        extra_fields=tuple(str(c) for c in extra.columns if c not in key and pd.api.types.is_numeric_dtype(extra[c]))
        if not extra_fields: raise AssertionError("extra feature panel has no numeric fields")
        overlap=set(extra_fields).intersection(control_fields)
        if overlap: raise AssertionError(f"extra feature panel overlaps control fields: {sorted(overlap)}")
        states=states.merge(extra.loc[:,[*key,*extra_fields]],on=key,how="left",validate="one_to_one")
    fields=(*control_fields,*extra_fields)
    start,end=_month("2025-06"),_month("2026-01"); vstart,vend=_month("2026-06"),_month("2026-09")
    train_route=route.loc[(route["bcf_mc1_expected_bps"]>=40)&(route["current_mc1_expected_bps"]>=40)&route["timestamp"].ge(start)&route["timestamp"].lt(end)].copy()
    normal=route.loc[(route["bcf_mc1_expected_bps"]>=50)&(route["current_mc1_expected_bps"]>=50)].copy()
    normal25=normal.loc[normal["timestamp"].ge(start)&normal["timestamp"].lt(end)].copy(); normal26=normal.loc[normal["timestamp"].ge(vstart)&normal["timestamp"].lt(vend)].copy()
    thresholds=tuple(float(value.strip()) for value in str(args.thresholds).split(",") if value.strip())
    MULTIPLIERS=tuple(float(value.strip()) for value in str(args.multipliers).split(",") if value.strip())
    gradual_scales=tuple(float(value.strip()) for value in str(args.gradual_confidence_scales).split(",") if value.strip())
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("thresholds must contain one or more positive bps values")
    if any(value <= 0 for value in gradual_scales):
        raise ValueError("gradual-confidence-scales must contain positive bps scales")
    if len(MULTIPLIERS) < 3 or len(set(MULTIPLIERS)) != len(MULTIPLIERS) or not any(np.isclose(value, 1.0) for value in MULTIPLIERS):
        raise ValueError("multipliers must be unique, include neutral 1.0, and include at least one action on each side or direction")
    if any(value < .20 or value > 3.0 for value in MULTIPLIERS):
        raise ValueError("multipliers must stay inside the replay engine's sealed [0.20, 3.00] safety envelope")
    if args.max_depth < 1 or args.num_leaves < 2 or not (0.0 < args.min_child_fraction <= 1.0) or args.reg_lambda < 0 or args.positive_weight < 1 or not (0.0 < args.quantile_alpha < 0.5):
        raise ValueError("invalid sealed model configuration")
    model_config = {"objective": args.objective, "quantile_alpha":args.quantile_alpha, "max_depth": args.max_depth, "num_leaves": args.num_leaves, "min_child_fraction": args.min_child_fraction, "reg_lambda": args.reg_lambda, "positive_weight": args.positive_weight}
    if args.resume_labels_from is None:
        sample=_sample_states(states,set(train_route["candidate_id"].astype(str)))
        labels=_labels(rows,arrays,(params,median),train_route,outcomes,sample,args.actuator,args.label_workers)
        label_source=None
    else:
        label_source=args.resume_labels_from.resolve()
        sample=pd.read_parquet(label_source/"training_state_sample_target_free.parquet")
        labels=pd.read_parquet(label_source/"next15_counterfactual_labels.parquet")
        expected=set(train_route["candidate_id"].astype(str))
        if not set(labels["candidate_id"].astype(str)).issubset(expected):
            raise AssertionError("resumed labels escape the frozen 2025 training route")
        if set(labels["actuator"].astype(str)) != {args.actuator}:
            raise AssertionError("resumed labels use the wrong actuator")
        if set(pd.to_numeric(labels["multiplier"], errors="raise").astype(float)) != set(MULTIPLIERS):
            raise AssertionError("resumed labels have an incomplete multiplier grid")
    lstate=_label_states(states,labels)
    normal_state25=states.loc[states["candidate_id"].isin(set(normal25["candidate_id"]))].copy(); normal_state26=states.loc[states["candidate_id"].isin(set(normal26["candidate_id"]))].copy()
    # One neutral replay identity check before fitting.
    probe=normal25.head(64); rep=base._replay(rows,arrays,probe,params,median,None,None).set_index("candidate_id")["exact_net_bps"]
    actual=outcomes.set_index("candidate_id").loc[rep.index,"parent_exact_net_bps"]
    if not np.allclose(rep.to_numpy(float),actual.to_numpy(float),atol=1e-8,rtol=0): raise AssertionError("parent replay parity failure")
    out.mkdir(parents=True,exist_ok=False)
    sample.to_parquet(out/"training_state_sample_target_free.parquet",index=False,compression="zstd"); labels.to_parquet(out/"next15_counterfactual_labels.parquet",index=False,compression="zstd")
    pd.DataFrame({"position":range(len(fields)),"feature":fields}).to_parquet(out/"existing_target_free_feature_contract.parquet",index=False,compression="zstd")
    if args.label_only:
        label_diag = labels.groupby("multiplier", as_index=False).agg(
            rows=("candidate_id", "size"),
            mean_advantage_bps=("advantage_bps", "mean"),
            positive_share=("advantage_bps", lambda s: float((s > 0).mean())),
            p90_advantage_bps=("advantage_bps", lambda s: float(s.quantile(.9))),
        )
        label_diag.to_parquet(out/"next15_label_direction_diagnostics.parquet",index=False,compression="zstd")
        manifest = {
            "schema": "causal-sr-h4-next15m-bidirectional-actuator-labels-v1",
            "scope": "offline research only; no live, exchange, admission, portfolio, geometry, C1 S/R, or MC1 mutation",
            "actuator": args.actuator,
            "target": "exact H12 net-bps difference when one actuator multiplier applies only from sampled completed state through the next completed 15m state, then resets to parent setting",
            "multipliers": MULTIPLIERS,
            "training_population": "paired MC1 >=40 bps unauctioned label route, complete exact paths only",
            "features": {"control_91":list(control_fields),"extra_target_free_fields":list(extra_fields)},
            "parent_root": str(parent),
            "parent_manifest_sha256": _sha256(parent/"run_manifest.json"),
            "state_root": str(state_root),
            "state_manifest_sha256": _sha256(state_root/"run_manifest.json"),
            "policy_sha256": _sha256(policy),
            "no_exchange_calls": True,
        }
        (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
        print(out)
        return
    # Strict 2025 schedule on normal portfolio candidates only.
    schedule25=_oof(lstate,normal_state25,fields,model_config,start,end)
    schedule25.to_parquet(out/"2025_oof_action_value_predictions.parquet",index=False,compression="zstd")
    defs=_authority_defs(thresholds, gradual_scales)
    arms=[]; records=[_evaluate(rows,arrays,normal25,params,median,None,args.actuator,"parent",out)]
    for name,mode,threshold in defs:
        sch=_mapping(schedule25,mode,threshold); sch["actuator"]=args.actuator; sch.to_parquet(out/f"2025_oof_schedule_{name}.parquet",index=False,compression="zstd")
        arm=f"{args.actuator}__{name}"; summary=_evaluate(rows,arrays,normal25,params,median,sch,args.actuator,arm,out)
        records.append({**summary,"mode":mode,"threshold_bps":threshold,"scheduled_state_actions":int(sch["multiplier"].ne(1).sum())}); arms.append(arm)
        del sch;gc.collect()
    summary=pd.DataFrame(records); ref=summary.loc[summary.arm.eq("parent")].iloc[0]
    for col in ("net_bps_per_trade","total_net_bps","sortino","max_drawdown","worst_week","cvar10_bps","worst_month_bps"): summary[f"delta_vs_parent_{col}"]=summary[col]-ref[col]
    summary["total_bps_per_abs_drawdown"]=summary["total_net_bps"]/summary["max_drawdown"].abs().clip(lower=1e-9)
    summary.to_parquet(out/"2025_oof_exact_portfolio_summary.parquet",index=False,compression="zstd"); _monthly(out,["parent",*arms],"2025_oof_monthly_metrics.parquet")
    winner=summary.loc[summary.arm.ne("parent")].sort_values(["total_bps_per_abs_drawdown","net_bps_per_trade","worst_week"],ascending=False,kind="stable").iloc[0]
    # One frozen 2026 fit using 2025 labels only; do not revisit the grid.
    frozen=_schedule(
        lstate, normal_state26, fields, model_config,
        start=start, held_start=end, held_end=vend,
    )
    mode=str(winner["mode"]); threshold=float(winner["threshold_bps"]); sch26=_mapping(frozen,mode,threshold);sch26["actuator"]=args.actuator;sch26.to_parquet(out/"2026_frozen_winner_schedule.parquet",index=False,compression="zstd")
    val=pd.DataFrame([_evaluate(rows,arrays,normal26,params,median,None,args.actuator,"2026_parent",out),_evaluate(rows,arrays,normal26,params,median,sch26,args.actuator,"2026_frozen_winner",out)])
    ref=val.loc[val.arm.eq("2026_parent")].iloc[0]
    for col in ("net_bps_per_trade","total_net_bps","sortino","max_drawdown","worst_week","cvar10_bps","worst_month_bps"): val[f"delta_vs_parent_{col}"]=val[col]-ref[col]
    val.to_parquet(out/"2026_frozen_confirmation_summary.parquet",index=False,compression="zstd");_monthly(out,["2026_parent","2026_frozen_winner"],"2026_frozen_monthly_metrics.parquet")
    label_diag = labels.groupby("multiplier", as_index=False).agg(
        rows=("candidate_id", "size"),
        mean_advantage_bps=("advantage_bps", "mean"),
        positive_share=("advantage_bps", lambda s: float((s > 0).mean())),
        p90_advantage_bps=("advantage_bps", lambda s: float(s.quantile(.9))),
    )
    label_diag.to_parquet(out/"next15_label_direction_diagnostics.parquet",index=False,compression="zstd")
    manifest={"schema":"causal-sr-h4-next15m-bidirectional-actuator-v1","scope":"offline research only; no live, exchange, admission, portfolio, geometry, C1 S/R, or MC1 mutation","actuator":args.actuator,"target":"exact H12 net-bps difference when one actuator multiplier applies only from sampled completed state through the next completed 15m state, then resets to parent setting","multipliers":MULTIPLIERS,"selection":"2025-06..2025-12 strict-prior OOF with 2025-only exact constrained portfolio selection","confirmation":"one frozen 2026-06..2026-08 check fitted on resolved 2025 labels only","training_population":"paired MC1 >=40 bps unauctioned label route, complete exact paths only","assessment_population":"paired MC1 >=50 bps normal global chronological constrained auction, complete exact paths only","features":{"control_91":list(control_fields),"extra_target_free_fields":list(extra_fields)},"extra_feature_panel":str(args.extra_feature_panel.resolve()) if args.extra_feature_panel is not None else None,"model_config":model_config,"authority_thresholds_bps":thresholds,"gradual_confidence_scales_bps":gradual_scales,"authority_arms":defs,"resumed_labels_from":str(label_source) if label_source else None,"selected_2025_winner":winner.to_dict(),"parent_root":str(parent),"parent_manifest_sha256":_sha256(parent/"run_manifest.json"),"state_root":str(state_root),"state_manifest_sha256":_sha256(state_root/"run_manifest.json"),"policy_sha256":_sha256(policy),"no_exchange_calls":True}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    print(out)

if __name__=="__main__": main()
