#!/usr/bin/env python3
"""Exact-1m C0/C1 + frozen-H4 transfer replay with a common portfolio auction.

This is deliberately a transfer diagnostic.  It uses the frozen C4 feature
contracts and H4 geometry trained on their historical, prior-resolved H4
labels, but runs the held candidates through an exact one-minute rich-policy
path.  C0 and C1 therefore share candidate identities, minute paths, policy,
and portfolio constraints; only their target-free MC1 values alter H4's
state input and the dual-MC1 route.

E2 is not included here: its historic BCF-top-two reserve population is not
the C0/C1 route.  Its transfer requires a separately materialised predecessor
pair ledger; silently splicing the legacy selection would be invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import replay_exact_1m_h4_giveback20
from extreme_price_movements.p8u_continuation_v2_features import (
    add_causal_age_expectations,
    materialize_extended_state_features,
)
from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as h4_study
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4_hpo
from scripts.materialize_strict_r3_p8u_v58_exact_h4_states import _read_bars
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _params as portfolio_params,
)
from scripts.run_causal_sr_c1_exact1m_parent_portfolio_replay import (
    ARMS,
    DEFAULT_OUT as DEFAULT_PARENT_ROOT,
    _common_target_free_routes,
)
from scripts.run_strict_r3_p8u_exact_1m_simple_policy_optimiser import _load_policy


DEFAULT_FEATURE_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2/stable_selected_features.parquet"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_c1_h4_exact1m_transfer_20260831_v1"
DEFAULT_JUNJUL = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v5"
DEFAULT_AUG = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_august_20260831_v3"
SEED = 1729


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(contract: Path, held: pd.Timestamp) -> tuple[str, ...]:
    selected = pd.read_parquet(contract)
    rows = selected.loc[
        selected["arm"].eq("C4_normalized_vwap_fs")
        & selected["held_month"].eq(held.strftime("%Y-%m"))
    ].sort_values("position", kind="stable")
    fields = tuple(rows["feature"].astype(str))
    if len(fields) != 45 or len(set(fields)) != len(fields):
        raise AssertionError(f"{held:%Y-%m}: missing frozen 45-field C4 contract")
    return fields


def _fit(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMRegressor:
    spec = h4_hpo.SPECS["H4_l1_d4_l15_leaf5_reg20"]
    child = max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=int(spec["n_estimators"]), learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]), num_leaves=int(spec["num_leaves"]), min_child_samples=child,
        subsample=.80, colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]),
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], train["activation50_advantage_bps"].to_numpy(float), sample_weight=weights)
    return model


def _raw_states(parent_root: Path, policy: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], tuple[object, float]]:
    rows = pd.read_parquet(parent_root / "valid_exact_paths_rows.parquet").copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="raise")
    rows["entry_ts"] = pd.to_datetime(rows["entry_ts"], utc=True, errors="raise")
    # Candidate IDs are persisted as an object array by the controlled parent
    # replay.  This archive is an immutable, locally generated input; allow its
    # identity array while retaining the explicit row-alignment assertion below.
    packed = np.load(parent_root / "exact_paths.npz", allow_pickle=True)
    ids = packed["candidate_id"].astype(str)
    if not np.array_equal(rows["candidate_id"].to_numpy(str), ids):
        raise AssertionError("exact path archive lost row identity")
    arrays = {name: np.asarray(packed[name]) for name in ("entry", "atr", "high", "low", "close")}
    if any(len(values) != len(rows) for values in arrays.values()):
        raise AssertionError("exact path arrays do not align to rows")
    params, median, _ = _load_policy(policy)
    parts: list[pd.DataFrame] = []
    outcomes: list[dict[str, object]] = []
    for position, row in rows.iterrows():
        trace = replay_exact_1m_h4_giveback20(
            entry_price=float(arrays["entry"][position]), signal_atr=float(arrays["atr"][position]), entry_ts=row["entry_ts"],
            highs=arrays["high"][position], lows=arrays["low"][position], closes=arrays["close"][position],
            params=params, median_atr_fraction=float(median), mc1_expected_bps=0.0,
            state_decider=None, emit_states=True,
        )
        outcomes.append({
            "candidate_id": str(row["candidate_id"]), "parent_exact_net_bps": float(trace["net_bps"]),
            "parent_exact_gross_bps": float(trace["gross_bps"]), "parent_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            "parent_exit_price": float(trace["exit_price"]), "parent_exit_minute": int(trace["exit_minute"]),
            "parent_exit_reason": str(trace["exit_reason"]),
        })
        if trace["states"]:
            state = pd.DataFrame(trace["states"])
            state["candidate_id"] = str(row["candidate_id"])
            state["__symbol__"] = str(row["symbol"])
            state["entry_decision_ts"] = pd.Timestamp(row["timestamp"])
            state["entry_price"] = float(arrays["entry"][position])
            state["signal_atr"] = float(arrays["atr"][position])
            state["state_bar_15m"] = np.arange(len(state), dtype=np.int16)
            state["current_PnL"] = state["current_pnl_atr"].to_numpy(float) * float(arrays["atr"][position]) / float(arrays["entry"][position]) * 10_000.0 - 100.0
            parts.append(state)
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw.empty:
        raise RuntimeError("complete exact paths produced no completed 15-minute H4 states")
    keys = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    if raw.duplicated(keys).any():
        raise AssertionError("exact H4 states duplicate identity")
    return rows, raw, arrays, (params, float(median))


def _state_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    parts: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for symbol, group in raw.groupby("__symbol__", sort=True):
        bars = _read_bars(str(symbol))
        if bars is None:
            coverage.append({"symbol": str(symbol), "state_rows": int(len(group)), "materialised": 0, "reason": "missing_or_unreadable_15m_source"})
            continue
        try:
            expanded = materialize_extended_state_features(group.drop(columns=["current_pnl_atr"]), bars, side="long")
        except Exception as exc:
            coverage.append({"symbol": str(symbol), "state_rows": int(len(group)), "materialised": 0, "reason": f"feature_error:{type(exc).__name__}"})
            continue
        parts.append(expanded)
        coverage.append({"symbol": str(symbol), "state_rows": int(len(group)), "materialised": int(len(expanded)), "reason": "ok"})
    if not parts:
        raise RuntimeError("no H4 context features available")
    panel = pd.concat(parts, ignore_index=True)
    dynamic = raw.loc[:, [*keys, "current_pnl_atr"]]
    panel = panel.merge(dynamic, on=keys, how="left", validate="one_to_one", suffixes=("", "__exact"))
    if "current_pnl_atr__exact" not in panel:
        raise AssertionError("exact H4 PnL state was lost in contextual materialisation")
    panel["current_pnl_atr"] = panel.pop("current_pnl_atr__exact")
    panel = add_causal_age_expectations(panel)
    if panel.duplicated(keys).any():
        raise AssertionError("H4 feature materialisation changed state identity")
    return panel, pd.DataFrame(coverage)


def _actions(
    states: pd.DataFrame,
    route: pd.DataFrame,
    *,
    contract: Path,
) -> pd.DataFrame:
    old = h4_study._load_panel(h4_study.TARGET_PANEL, h4_study.VWAP_PANEL)
    score = route.loc[:, ["candidate_id", "bcf_mc1_expected_bps"]].copy()
    panel = states.merge(score, on="candidate_id", how="inner", validate="many_to_one", suffixes=("", "__route"))
    panel["MC1_expected_bps"] = pd.to_numeric(panel.pop("bcf_mc1_expected_bps"), errors="raise")
    actions: list[pd.DataFrame] = []
    for held in sorted(pd.to_datetime(panel["entry_decision_ts"], utc=True).dt.to_period("M").unique()):
        held_ts = pd.Timestamp(held.start_time, tz="UTC")
        end = held_ts + pd.offsets.MonthBegin(1)
        fields = _fields(contract, held_ts)
        train = old.loc[
            old["entry_decision_ts"].ge(held_ts - pd.DateOffset(months=4))
            & old["entry_decision_ts"].lt(held_ts)
            & old["policy_label_available_ts"].lt(held_ts)
            & pd.to_numeric(old["MC1_expected_bps"], errors="coerce").ge(30.0)
        ].copy()
        test = panel.loc[panel["entry_decision_ts"].ge(held_ts) & panel["entry_decision_ts"].lt(end)].copy()
        missing = set(fields).difference(train.columns) | set(fields).difference(test.columns)
        if missing:
            raise AssertionError(f"{held_ts:%Y-%m}: missing frozen H4 field(s) {sorted(missing)}")
        if train.candidate_id.nunique() < 100:
            raise RuntimeError(f"{held_ts:%Y-%m}: insufficient strict-prior frozen H4 support")
        model = _fit(train, fields)
        value = model.predict(test.loc[:, fields]) if not test.empty else np.asarray([], dtype=float)
        action = test.loc[:, ["candidate_id", "state_decision_ts"]].copy()
        action["h4_predicted_activation50_advantage_bps"] = value
        action["h4_enable_giveback20"] = action["h4_predicted_activation50_advantage_bps"].ge(0.0)
        action["held_month"] = held_ts.strftime("%Y-%m")
        actions.append(action)
    result = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(columns=["candidate_id", "state_decision_ts", "h4_enable_giveback20"])
    if result.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("H4 action schedule duplicates state identity")
    return result


def _replay_h4(
    rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    route: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    params: object,
    median: float,
) -> pd.DataFrame:
    lookup = {candidate: dict(zip(group["state_decision_ts"], group["h4_enable_giveback20"], strict=True)) for candidate, group in actions.groupby("candidate_id", sort=False)}
    position = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    selected = route.loc[route["candidate_id"].isin(position.index)].copy()
    results: list[dict[str, object]] = []
    for _, row in selected.sort_values(["timestamp", "candidate_id"], kind="stable").iterrows():
        idx = int(position[str(row["candidate_id"])])
        schedule = lookup.get(str(row["candidate_id"]), {})
        def decide(state, schedule=schedule):
            return bool(schedule.get(pd.Timestamp(state["state_decision_ts"]), False))
        trace = replay_exact_1m_h4_giveback20(
            entry_price=float(arrays["entry"][idx]), signal_atr=float(arrays["atr"][idx]), entry_ts=rows.iloc[idx]["entry_ts"],
            highs=arrays["high"][idx], lows=arrays["low"][idx], closes=arrays["close"][idx],
            params=params, median_atr_fraction=median, mc1_expected_bps=float(row["bcf_mc1_expected_bps"]),
            state_decider=decide, giveback_tighten=.20, emit_states=False,
        )
        results.append({
            "candidate_id": str(row["candidate_id"]), "timestamp": row["timestamp"], "entry_ts": rows.iloc[idx]["entry_ts"], "symbol": row["symbol"],
            "bcf_mc1_expected_bps": float(row["bcf_mc1_expected_bps"]), "auction_priority_bps": float(row["auction_priority_bps"]),
            "exact_entry_price": float(arrays["entry"][idx]), "exact_net_bps": float(trace["net_bps"]), "exact_gross_bps": float(trace["gross_bps"]),
            "exact_exit_price": float(trace["exit_price"]), "exact_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            "exact_exit_minute": int(trace["exit_minute"]), "exact_exit_reason": str(trace["exit_reason"]),
        })
    return pd.DataFrame(results)


def _portfolio(frame: pd.DataFrame, arm: str):
    candidates = pd.DataFrame({
        "timestamp": frame["entry_ts"], "decision_timestamp": frame["timestamp"], "candidate_id": frame["candidate_id"], "symbol": frame["symbol"],
        "side": "long", "strategy_id": arm, "policy_archetype": "exact1m_h4_giveback20_transfer",
        "normalized_rank_score": 1.0, "strategy_rank_pct": 1.0, "base_strategy_threshold": 0.0, "calibrated_score": 1.0,
        "portfolio_priority_adjustment": frame["auction_priority_bps"], "entry_price": frame["exact_entry_price"],
        "exit_timestamp": frame["exact_exit_ts"], "exit_price": frame["exact_exit_price"], "net_return": frame["exact_net_bps"] / 10_000.0,
        "gross_return": frame["exact_gross_bps"] / 10_000.0, "holding_bars": np.maximum(frame["exact_exit_minute"].to_numpy(int) + 1, 1),
        "simple_policy_exit_reason": frame["exact_exit_reason"], "fees_bps": 100.0, "expected_friction_bps": 0.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "policy_outcome_available": True,
    })
    candidates = normalise_candidate_table(candidates)
    decisions, equity, _ = replay_candidates(candidates, portfolio_params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    positions = pd.to_numeric(decisions["candidate_index"], errors="raise").astype(int).to_numpy()
    decisions = decisions.copy()
    decisions["candidate_id"] = candidates.iloc[positions]["candidate_id"].to_numpy()
    decisions["decision_timestamp"] = candidates.iloc[positions]["decision_timestamp"].to_numpy()
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="raise") * 10_000.0
    metrics = compute_replay_metrics(candidates, decisions, equity, params=portfolio_params())
    metrics.update({"portfolio_accepted": int(len(accepted)), "net_bps_per_trade": float(accepted["net_bps"].mean()) if len(accepted) else np.nan, "total_net_bps": float(accepted["net_bps"].sum())})
    return candidates, decisions, accepted, equity, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--junjul-root", type=Path, default=DEFAULT_JUNJUL)
    parser.add_argument("--august-root", type=Path, default=DEFAULT_AUG)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    parent_root = args.parent_root.resolve()
    rows, raw, arrays, (params, median) = _raw_states(parent_root, args.policy.resolve())
    states, coverage = _state_features(raw)
    routes = _common_target_free_routes(args.junjul_root.resolve(), args.august_root.resolve())
    out.mkdir(parents=True, exist_ok=False)
    raw.to_parquet(out / "exact_h4_dynamic_states_target_free.parquet", index=False, compression="zstd")
    states.to_parquet(out / "exact_h4_context_states_target_free.parquet", index=False, compression="zstd")
    coverage.to_parquet(out / "exact_h4_state_source_coverage.parquet", index=False, compression="zstd")
    metrics_rows: list[dict[str, object]] = []
    for arm, route in routes.items():
        action = _actions(states, route, contract=args.feature_contract.resolve())
        outcome = _replay_h4(rows, arrays, route, action, params=params, median=median)
        candidates, decisions, accepted, equity, metrics = _portfolio(outcome, arm)
        action.insert(0, "arm", arm)
        action.to_parquet(out / f"{arm}_h4_action_schedule_target_free.parquet", index=False, compression="zstd")
        outcome.to_parquet(out / f"{arm}_h4_exact1m_outcomes.parquet", index=False, compression="zstd")
        candidates.to_parquet(out / f"{arm}_portfolio_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(out / f"{arm}_portfolio_accepted.parquet", index=False, compression="zstd")
        equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
        monthly = accepted.assign(month=pd.to_datetime(accepted["decision_timestamp"], utc=True).dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum"))
        monthly.insert(0, "arm", arm)
        monthly.to_parquet(out / f"{arm}_monthly_portfolio_metrics.parquet", index=False, compression="zstd")
        metrics_rows.append({"arm": arm, **metrics})
    summary = pd.DataFrame(metrics_rows)
    ref = summary.loc[summary["arm"].eq("C0_refit_core_postfeb")].iloc[0]
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
        summary[f"delta_vs_C0_{field}"] = summary[field] - ref[field]
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "causal-sr-c1-h4-exact1m-transfer-v1",
        "scope": "offline research-only C0/C1 H4 transfer; exact one-minute exits and normal global portfolio constraints",
        "parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"),
        "feature_contract": str(args.feature_contract.resolve()), "feature_contract_sha256": _sha256(args.feature_contract.resolve()),
        "policy": str(args.policy.resolve()), "policy_sha256": _sha256(args.policy.resolve()),
        "H4": "frozen C4 45-field contract; H4 L1 d4/l15/min-child-5%/L2-20/LR-.025/420; prior-resolved legacy H4 labels used as explicit transfer training source; H4 controls only following 15-minute interval while exact 1-minute path executes exits",
        "portfolio": "global chronological controlled 7x/10%-margin slot, two new entries per timestamp, eight concurrent, 80%-wallet budget",
        "E2": "not composable from the legacy BCF-top-two reserve receipt; no E2 authority is asserted here",
        "no_exchange_calls": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
