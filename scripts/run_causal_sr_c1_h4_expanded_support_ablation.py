#!/usr/bin/env python3
"""Expanded, strict-prequential H4 support and feature-pruning ablation.

This is deliberately a research-only successor to the sparse June--August H4
refit.  It starts from an immutable, target-free C1 route and its already
materialised exact-one-minute parent paths.  A state label is formed only
after the parent trade's 12-hour outcome has resolved.  Monthly H4 fits use
only earlier resolved labels; a 15-minute H4 decision is then applied to the
*following* interval and can only tighten the rich parent policy.

The three feature arms share one May-2026 frozen master contract.  F35 and F25
remove 10 and 20 non-mandatory fields using prior-window availability and
redundancy only--never a label, prediction, held-month outcome, or portfolio
result.  This makes the removal experiment causal even though the master
field namespace originates from the existing H4 research contract.
"""

from __future__ import annotations

import argparse
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

from scripts.run_causal_sr_c1_h4_exact1m_refit_replay import (
    ARM,
    H12,
    _LABEL_CONTEXT,
    _label_candidate,
    _outcomes,
)
from scripts.run_causal_sr_c1_h4_exact1m_transfer_replay import (
    DEFAULT_FEATURE_CONTRACT,
    DEFAULT_POLICY,
    _portfolio,
    _raw_states,
    _state_features,
)


DEFAULT_PARENT = ROOT / "data_perp/artifacts/causal_sr_c1_exact1m_parent_t40_expanded_20260831_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_c1_h4_expanded_support_20260831_v1"
MANDATORY = (
    "time_in_trade",
    "current_pnl_atr",
    "current_MFE_ATR",
    "current_MAE_ATR",
    "giveback_from_MFE_ATR",
    "distance_to_current_SL_ATR",
    "is_trailing_active",
    "current_protection_state",
    "MC1_expected_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_start(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _master_fields(contract: Path) -> tuple[str, ...]:
    values = pd.read_parquet(contract)
    chosen = values.loc[
        values["arm"].eq("C4_normalized_vwap_fs")
        & values["held_month"].eq("2026-05")
    ].sort_values("position", kind="stable")
    fields = tuple(chosen["feature"].astype(str))
    if len(fields) != 45 or len(set(fields)) != len(fields):
        raise AssertionError("2026-05 C4 master contract must contain exactly 45 unique fields")
    if tuple(fields[: len(MANDATORY)]) != MANDATORY:
        raise AssertionError("master contract lost mandatory H4 state order")
    return fields


def _route(parent: Path) -> pd.DataFrame:
    route = pd.read_parquet(parent / "union_target_free_dual_admitted.parquet").copy()
    rows = pd.read_parquet(parent / "valid_exact_paths_rows.parquet", columns=["candidate_id"]).copy()
    route["candidate_id"] = route["candidate_id"].astype(str)
    valid = set(rows["candidate_id"].astype(str))
    route = route.loc[route["candidate_id"].isin(valid)].copy()
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="raise")
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    route["bcf_mc1_expected_bps"] = pd.to_numeric(route["bcf_mc1_expected_bps"], errors="raise")
    route["auction_priority_bps"] = pd.to_numeric(route["auction_priority_bps"], errors="raise")
    if route["candidate_id"].duplicated().any():
        raise AssertionError("expanded route has duplicate identities")
    return route.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _label_state_sample(states: pd.DataFrame, *, mfe_ready_atr: float, maximum: int) -> pd.DataFrame:
    """Take a fixed causal state sample per trade without examining the target.

    The first ready state, equally spaced ready states, and the final observed
    ready state are all decision-time states.  The label remains a later exact
    counterfactual; its value never affects the sample membership.
    """
    eligible = states.loc[pd.to_numeric(states["current_MFE_ATR"], errors="coerce").ge(mfe_ready_atr)].copy()
    parts: list[pd.DataFrame] = []
    for _, group in eligible.groupby("candidate_id", sort=True):
        group = group.sort_values("state_decision_ts", kind="stable")
        count = min(int(maximum), len(group))
        take = np.unique(np.linspace(0, len(group) - 1, count, dtype=int))
        parts.append(group.iloc[take])
    result = pd.concat(parts, ignore_index=True) if parts else eligible.iloc[0:0].copy()
    keys = ["candidate_id", "state_decision_ts"]
    if result.duplicated(keys).any():
        raise AssertionError("deterministic state sample duplicated an identity")
    return result


def _labels(
    rows: pd.DataFrame,
    sampled: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    policy: tuple[object, float],
    parent_outcomes: pd.DataFrame,
    workers: int,
) -> pd.DataFrame:
    parent = parent_outcomes.set_index("candidate_id")["exact_net_bps"].astype(float)
    position = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    tasks: list[tuple[int, tuple[int, ...], float]] = []
    for candidate, group in sampled.groupby("candidate_id", sort=True):
        candidate = str(candidate)
        if candidate not in parent.index or candidate not in position.index:
            continue
        states = tuple(pd.to_datetime(group["state_decision_ts"], utc=True).view("int64").tolist())
        tasks.append((int(position[candidate]), states, float(parent.loc[candidate])))
    if not tasks:
        raise RuntimeError("no MFE-ready exact H4 states available for labels")
    _LABEL_CONTEXT.clear()
    _LABEL_CONTEXT.update({
        "rows": rows,
        "arrays": arrays,
        "policy": policy,
        "target_mode": "latched_activation50_giveback20",
        "target_giveback_tighten": 0.20,
    })
    result: list[dict[str, object]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=context) as executor:
        for ordinal, block in enumerate(executor.map(_label_candidate, tasks), start=1):
            result.extend(block)
            if ordinal == 1 or ordinal % 500 == 0 or ordinal == len(tasks):
                print(f"expanded-H4 labels {ordinal}/{len(tasks)} candidates", flush=True)
    label = pd.DataFrame(result)
    if label.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("expanded H4 labels duplicated a state identity")
    return label


def _rank_context_fields(
    master: tuple[str, ...],
    prior: pd.DataFrame,
    test: pd.DataFrame,
    total: int,
) -> tuple[str, ...]:
    """Retain a target-free non-redundant subset of a frozen 45-field master."""
    if total not in {25, 35, 45}:
        raise ValueError("feature totals must be 25, 35, or 45")
    missing = set(master).difference(prior.columns) | set(master).difference(test.columns)
    if missing:
        raise AssertionError(f"state feature panel missing master fields: {sorted(missing)}")
    if total == 45:
        return master
    contexts = list(master[len(MANDATORY):])
    work = prior.loc[:, contexts].apply(pd.to_numeric, errors="coerce")
    availability = work.notna().mean(axis=0)
    # A deterministic bounded sample prevents the correlation pass from
    # letting a large historical month dominate feature survival.
    if len(work) > 50_000:
        stride = max(1, len(work) // 50_000)
        work = work.iloc[::stride].head(50_000)
    work = work.replace([np.inf, -np.inf], np.nan)
    median = work.median(axis=0)
    work = work.fillna(median).fillna(0.0)
    scale = (work.quantile(.75) - work.quantile(.25)).abs()
    corr = work.corr(method="spearman").abs().fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    redundancy = corr.max(axis=0)
    # High coverage, non-degenerate, low-redundancy fields survive.  No label,
    # portfolio, score, or held-month outcome occurs in this calculation.
    frame = pd.DataFrame({"field": contexts, "coverage": availability, "scale": scale, "redundancy": redundancy})
    frame["usable"] = frame["coverage"].ge(.95) & frame["scale"].gt(1e-10)
    frame = frame.sort_values(["usable", "redundancy", "coverage", "field"], ascending=[False, True, False, True], kind="stable")
    needed = int(total) - len(MANDATORY)
    kept = set(frame.head(needed)["field"].astype(str))
    ordered = tuple([*MANDATORY, *(field for field in contexts if field in kept)])
    if len(ordered) != total:
        raise AssertionError("target-free feature pruning returned wrong contract length")
    return ordered


def _fit(train: pd.DataFrame, fields: tuple[str, ...], depth: int, *, loss: str = "l2") -> lgb.LGBMRegressor:
    if depth not in {2, 4}:
        raise ValueError("supported H4 depths are 2 and 4")
    if loss not in {"l1", "l2"}:
        raise ValueError("supported H4 losses are l1 and l2")
    leaves = 7 if depth == 2 else 15
    child = max(32, int(np.ceil(len(train) * .05)))
    model = lgb.LGBMRegressor(
        objective=f"regression_{loss}",
        n_estimators=420,
        learning_rate=.025,
        max_depth=depth,
        num_leaves=leaves,
        min_child_samples=child,
        subsample=.80,
        colsample_bytree=.80,
        reg_lambda=20.0,
        random_state=1729,
        n_jobs=2,
        verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], train["activation50_advantage_bps"].to_numpy(float), sample_weight=weights)
    return model


def _actions(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    route: pd.DataFrame,
    master: tuple[str, ...],
    *,
    train_months: int,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    mfe_ready_atr: float,
    loss: str = "l2",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    keys = ["candidate_id", "state_decision_ts"]
    panel = states.merge(labels.loc[:, [*keys, "activation50_advantage_bps", "policy_label_available_ts"]], on=keys, how="left", validate="one_to_one")
    panel = panel.merge(route.loc[:, ["candidate_id", "bcf_mc1_expected_bps"]], on="candidate_id", how="inner", validate="many_to_one")
    panel["MC1_expected_bps"] = pd.to_numeric(panel.pop("bcf_mc1_expected_bps"), errors="raise")
    panel["entry_decision_ts"] = pd.to_datetime(panel["entry_decision_ts"], utc=True, errors="raise")
    panel["h4_actionable_state"] = pd.to_numeric(panel["current_MFE_ATR"], errors="coerce").ge(float(mfe_ready_atr))
    names = (("F45_d4", 45, 4), ("F35_d4", 35, 4), ("F25_d4", 25, 4), ("F45_d2", 45, 2))
    schedules: dict[str, list[pd.DataFrame]] = {name: [] for name, _, _ in names}
    contracts: list[dict[str, object]] = []
    for period in sorted(panel["entry_decision_ts"].dt.to_period("M").unique()):
        held = pd.Timestamp(period.start_time, tz="UTC")
        end = held + pd.offsets.MonthBegin(1)
        if held < eval_start or held >= eval_end:
            continue
        train_start = held - pd.DateOffset(months=int(train_months))
        train = panel.loc[
            panel["entry_decision_ts"].ge(train_start)
            & panel["entry_decision_ts"].lt(held)
            & panel["policy_label_available_ts"].lt(held)
            & panel["activation50_advantage_bps"].notna()
            & panel["h4_actionable_state"]
        ].copy()
        test = panel.loc[panel["entry_decision_ts"].ge(held) & panel["entry_decision_ts"].lt(end)].copy()
        if train["candidate_id"].nunique() < 500:
            raise RuntimeError(f"{held:%Y-%m}: inadequate prior H4 candidate support ({train['candidate_id'].nunique()})")
        if test.empty:
            continue
        prior_features = panel.loc[
            panel["entry_decision_ts"].ge(train_start) & panel["entry_decision_ts"].lt(held)
        ].copy()
        for name, total, depth in names:
            fields = _rank_context_fields(master, prior_features, test, total)
            contracts.extend({"arm": name, "held_month": held.strftime("%Y-%m"), "feature": field, "position": pos} for pos, field in enumerate(fields))
            schedule = test.loc[:, ["candidate_id", "state_decision_ts"]].copy()
            schedule["h4_predicted_activation50_advantage_bps"] = float("-inf")
            eligible = test.loc[test["h4_actionable_state"]].copy()
            if not eligible.empty:
                model = _fit(train, fields, depth, loss=loss)
                schedule.loc[eligible.index, "h4_predicted_activation50_advantage_bps"] = model.predict(eligible.loc[:, fields])
            schedule["h4_enable"] = schedule["h4_predicted_activation50_advantage_bps"].gt(0.0)
            schedule["held_month"] = held.strftime("%Y-%m")
            schedules[name].append(schedule)
    out = {name: pd.concat(parts, ignore_index=True) for name, parts in schedules.items() if parts}
    contract = pd.DataFrame(contracts)
    for name, value in out.items():
        if value.duplicated(["candidate_id", "state_decision_ts"]).any():
            raise AssertionError(f"{name}: action schedule duplicated a state identity")
    return out, contract


def _parent_frame(parent: Path, route: pd.DataFrame) -> pd.DataFrame:
    outcome = pd.read_parquet(parent / "exact_1m_rich_parent_outcomes.parquet").copy()
    outcome["candidate_id"] = outcome["candidate_id"].astype(str)
    return route.merge(outcome, on=["candidate_id", "timestamp", "entry_ts", "symbol"], how="inner", validate="one_to_one")


def _monthly(accepted: pd.DataFrame) -> pd.DataFrame:
    values = accepted.copy()
    values["month"] = pd.to_datetime(values["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    return values.groupby("month", as_index=False).agg(
        trades=("candidate_id", "size"),
        net_bps_per_trade=("net_bps", "mean"),
        total_net_bps=("net_bps", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--eval-start", default="2026-06")
    parser.add_argument("--eval-end", default="2026-09", help="exclusive YYYY-MM")
    parser.add_argument("--train-months", type=int, default=9)
    parser.add_argument("--label-states-per-candidate", type=int, default=4)
    parser.add_argument("--label-workers", type=int, default=8)
    parser.add_argument("--loss", choices=("l1", "l2"), default="l2", help="fixed model loss; does not alter action authority")
    parser.add_argument(
        "--reuse-label-root",
        type=Path,
        default=None,
        help="reuse an immutable target-free state and exact label materialisation instead of recomputing labels",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if args.train_months < 4 or args.label_states_per_candidate < 1:
        raise ValueError("requires at least four train months and one sampled state per candidate")
    eval_start, eval_end = _month_start(args.eval_start), _month_start(args.eval_end)
    if eval_end <= eval_start:
        raise ValueError("eval-end must be after eval-start")
    parent = args.parent_root.resolve()
    route = _route(parent)
    rows, raw, arrays, policy = _raw_states(parent, args.policy.resolve())
    params, _ = policy
    mfe_ready_atr = .5 * float(params.protection_activation_atr)
    if args.reuse_label_root is None:
        states, state_coverage = _state_features(raw)
        sampled = _label_state_sample(raw, mfe_ready_atr=mfe_ready_atr, maximum=args.label_states_per_candidate)
        parent_outcomes = pd.read_parquet(parent / "exact_1m_rich_parent_outcomes.parquet")
        parent_outcomes["candidate_id"] = parent_outcomes["candidate_id"].astype(str)
        labels = _labels(rows, sampled, arrays, policy, parent_outcomes, args.label_workers)
        reused_label_root = None
    else:
        reused_label_root = args.reuse_label_root.resolve()
        states = pd.read_parquet(reused_label_root / "all_h4_context_states_target_free.parquet")
        labels = pd.read_parquet(reused_label_root / "latched_activation50_giveback20_labels.parquet")
        sampled = pd.read_parquet(reused_label_root / "deterministic_mfe_ready_state_label_sample_target_free.parquet")
        coverage_path = reused_label_root / "h4_state_feature_coverage.parquet"
        state_coverage = pd.read_parquet(coverage_path) if coverage_path.is_file() else pd.DataFrame()
        required = {"candidate_id", "state_decision_ts", "entry_decision_ts", "current_MFE_ATR"}
        missing = sorted(required.difference(states.columns))
        if missing:
            raise ValueError(f"reused target-free H4 states lack {missing}")
        if labels.duplicated(["candidate_id", "state_decision_ts"]).any():
            raise ValueError("reused exact H4 labels duplicate state identity")
    master = _master_fields(args.feature_contract.resolve())
    schedules, contracts = _actions(
        states, labels, route, master,
        train_months=args.train_months,
        eval_start=eval_start,
        eval_end=eval_end,
        mfe_ready_atr=mfe_ready_atr,
        loss=args.loss,
    )
    eval_route = route.loc[route["timestamp"].ge(eval_start) & route["timestamp"].lt(eval_end)].copy()
    parent_frame = _parent_frame(parent, eval_route)
    out.mkdir(parents=True, exist_ok=False)
    raw.to_parquet(out / "all_parent_observable_h4_states_target_free.parquet", index=False, compression="zstd")
    states.to_parquet(out / "all_h4_context_states_target_free.parquet", index=False, compression="zstd")
    sampled.to_parquet(out / "deterministic_mfe_ready_state_label_sample_target_free.parquet", index=False, compression="zstd")
    labels.to_parquet(out / "latched_activation50_giveback20_labels.parquet", index=False, compression="zstd")
    state_coverage.to_parquet(out / "h4_state_feature_coverage.parquet", index=False, compression="zstd")
    contracts.to_parquet(out / "prequential_feature_contracts.parquet", index=False, compression="zstd")
    parent_candidates, parent_decisions, parent_accepted, parent_equity, parent_metrics = _portfolio(parent_frame, "C1_parent_exact1m")
    parent_accepted.to_parquet(out / "C1_parent_exact1m_portfolio_accepted.parquet", index=False, compression="zstd")
    parent_equity.to_parquet(out / "C1_parent_exact1m_portfolio_equity.parquet", index=False, compression="zstd")
    summary = [{"arm": "C1_parent_exact1m", **parent_metrics}]
    for name, actions in schedules.items():
        outcome = _outcomes(rows, arrays, eval_route, actions, policy, .20, latch=True)
        candidates, decisions, accepted, equity, metrics = _portfolio(outcome, name)
        outcome.to_parquet(out / f"{name}_exact1m_outcomes.parquet", index=False, compression="zstd")
        candidates.to_parquet(out / f"{name}_portfolio_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(out / f"{name}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(out / f"{name}_portfolio_accepted.parquet", index=False, compression="zstd")
        equity.to_parquet(out / f"{name}_portfolio_equity.parquet", index=False, compression="zstd")
        _monthly(accepted).to_parquet(out / f"{name}_monthly_portfolio_metrics.parquet", index=False, compression="zstd")
        actions.to_parquet(out / f"{name}_action_schedule_target_free.parquet", index=False, compression="zstd")
        summary.append({"arm": name, **metrics})
    table = pd.DataFrame(summary)
    reference = table.loc[table["arm"].eq("C1_parent_exact1m")].iloc[0]
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
        if field in table:
            table[f"delta_vs_parent_{field}"] = table[field] - reference[field]
    table.to_parquet(out / "portfolio_summary.parquet", index=False)
    manifest = {
        "schema": "causal_sr_c1_h4_expanded_support_ablation_v1",
        "scope": "offline research only; no live model, feature contract, or exchange action changes",
        "parent_root": str(parent),
        "parent_manifest_sha256": _sha256(parent / "run_manifest.json"),
        "route": "frozen target-free C1 dual 40-bps route, valid exact paths only after routing",
        "state_label": "latched exact activation50+giveback20 continuation net minus unchanged exact parent net",
        "state_sampling": f"first/equally-spaced/last MFE-ready parent-observable states; at most {args.label_states_per_candidate} per candidate; independent of label",
        "prequential": f"monthly held model uses at most {args.train_months} prior calendar months and labels resolved before the held month",
        "evaluation": f"[{eval_start.isoformat()}, {eval_end.isoformat()})",
        "feature_master": "frozen May-2026 C4 45-field contract; F35/F25 prune only non-mandatory fields from prior target-free state geometry",
        "mandatory_fields": list(MANDATORY),
        "model_arms": {"F45_d4": f"{args.loss.upper()} d4/l15", "F35_d4": f"{args.loss.upper()} d4/l15", "F25_d4": f"{args.loss.upper()} d4/l15", "F45_d2": f"{args.loss.upper()} d2/l7"},
        "reused_exact_label_root": None if reused_label_root is None else str(reused_label_root),
        "authority": "strictly positive prediction on completed MFE-ready 15m state latches a 20% giveback tightening; no stop loosening, promotion, sizing, or same-bar action",
        "portfolio": "normal global chronological auction, unchanged from parent; rich exact-1m policy and 100-bps cost once",
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
