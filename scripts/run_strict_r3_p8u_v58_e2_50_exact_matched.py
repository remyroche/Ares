#!/usr/bin/env python3
"""Matched, research-only E2-50 replay on the sealed v58-style exact ledger.

The archived E2 contract used a 20--30 bps reserve, whereas the live control
admits only dual-MC1 >=50 bps candidates.  It therefore has no executable
reserve population under the live contract.  This runner is an explicitly new
``E2_50`` successor: it preserves E2's two independent q50 component models
and intersection-only replacement authority, but draws the reserve from the
already live-admissible dual-50 population.  It never adds capacity.

The runner is deliberately offline.  It reads a target-free score ledger and
causal 15-minute bars, joins exact one-minute outcomes only after target-free
E2 selections have been written, and never imports live execution code.
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

from extreme_price_movements.exact_1m_rich_policy_contract import (
    Exact1mRichExecutionContract,
    RichExitExtensions,
    replay_exact_1m_rich_policy,
)
from extreme_price_movements.p8u_15m_features import (
    FIFTEEN_MINUTE_FEATURE_KEYS,
    VWAP_15M_FEATURE_KEYS,
    compute_15m_features,
    compute_15m_vwap_features,
)
from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _params as portfolio_params,
)
from scripts.run_strict_r3_p8u_15m_continuation_walkforward import (
    BARS_ROOT,
    _symbol_filename,
)
from scripts.run_strict_r3_rich_policy_hpo import _hourly_signal_atr
from scripts.run_strict_r3_p8u_exact_1m_simple_policy_optimiser import _load_policy


DEFAULT_SCORE_LEDGER = ROOT / "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_20260828_v1/dual_predictions.parquet"
DEFAULT_PATH_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_exact1m_simple_policy_optimiser_mayjul2026_20260829_v1"
DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v2/stable_selected_features.parquet"
DEFAULT_CONTROL_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_exact1m_constrained_dual50_aug01_27_20260829_v1"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_e2_50_exact_matched_20260830_v3_causal_atr"

SEED = 1729
DUAL_FLOOR = 50.0
MAX_NEW = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(values: object) -> pd.DatetimeIndex | pd.Timestamp:
    value = pd.to_datetime(values, utc=True, errors="raise")
    return value


def _read_features(contract: Path, held_month: str) -> tuple[str, ...]:
    frame = pd.read_parquet(contract)
    rows = frame.loc[
        frame["variant"].eq("E3_vwap_fs") & frame["held_month"].eq(held_month)
    ].sort_values("position", kind="stable")
    fields = tuple(rows["feature"].astype(str))
    if len(fields) != 30 or len(set(fields)) != len(fields):
        raise AssertionError(f"{held_month}: expected ordered E3 30-field contract")
    return fields


def _load_route(score_ledger: Path, path_root: Path) -> pd.DataFrame:
    route = pd.read_parquet(path_root / "target_free_candidates.parquet")
    scores = pd.read_parquet(
        score_ledger,
        columns=[
            "candidate_id", "__decision_ts__", "__symbol__", "side_name",
            "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        ],
    )
    route["candidate_id"] = route["candidate_id"].astype(str)
    scores["candidate_id"] = scores["candidate_id"].astype(str)
    route["timestamp"] = _utc(route["timestamp"])
    route["entry_ts"] = _utc(route["entry_ts"])
    scores["__decision_ts__"] = _utc(scores["__decision_ts__"])
    merged = route.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol"]].merge(
        scores, on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(merged) != len(route):
        raise AssertionError("score ledger does not fully cover sealed target-free route")
    if not (merged["timestamp"] == merged["__decision_ts__"]).all():
        raise AssertionError("score ledger changed candidate decision identities")
    if not (merged["symbol"].astype(str) == merged["__symbol__"].astype(str)).all():
        raise AssertionError("score ledger changed candidate symbols")
    merged["dual_mc1_min_bps"] = merged[["bcf_mc1_expected_bps", "current_mc1_expected_bps"]].min(axis=1)
    if not merged["dual_mc1_min_bps"].ge(DUAL_FLOOR).all():
        raise AssertionError("sealed route contains a non-dual50 candidate")
    return merged.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _load_exact_outcomes(path_root: Path, policy: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path_root / "valid_exact_paths_rows.parquet")
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["timestamp"] = _utc(rows["timestamp"])
    rows["entry_ts"] = _utc(rows["entry_ts"])
    archive = np.load(path_root / "exact_paths.npz", allow_pickle=False)
    entry = np.asarray(archive["entry"], dtype=float)
    atr = np.asarray(archive["atr"], dtype=float)
    high = np.asarray(archive["high"], dtype=np.float32)
    low = np.asarray(archive["low"], dtype=np.float32)
    close = np.asarray(archive["close"], dtype=np.float32)
    if not (len(rows) == len(entry) == len(atr) == len(high)):
        raise AssertionError("exact numeric path arrays do not align to valid row panel")
    params, median, policy_receipt = _load_policy(policy)
    replay = replay_exact_1m_rich_policy(
        positions=pd.DataFrame({"entry_price": entry, "atr": atr, "entry_ts": rows["entry_ts"]}),
        highs=high, lows=low, closes=close, params=params,
        median_atr_fraction=median,
        contract=Exact1mRichExecutionContract(entry_delay_minutes=5),
        extensions=RichExitExtensions(),
    )
    if not np.asarray(replay["path_valid"], dtype=bool).all():
        raise AssertionError("a sealed complete exact path did not replay")
    out = rows.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol"]].copy()
    out["exact_entry_price"] = entry
    out["exact_signal_atr"] = atr
    out["exact_gross_bps"] = np.asarray(replay["gross_bps"], dtype=float)
    out["exact_net_bps"] = np.asarray(replay["net_bps"], dtype=float)
    out["exact_exit_price"] = np.asarray(replay["exit_price"], dtype=float)
    out["exact_exit_ts"] = pd.to_datetime(replay["exit_timestamp"], utc=True)
    out["exact_exit_minute"] = np.asarray(replay["exit_minute"], dtype=int)
    out["exact_exit_reason"] = np.asarray(replay["exit_reason"], dtype=object)
    out.attrs["policy_receipt"] = policy_receipt
    return out


def _materialize_target_free_features(route: pd.DataFrame) -> pd.DataFrame:
    """Compute all static E3 inputs before outcomes are joined to them.

    Signal ATR is reconstructed from causal completed 15-minute bars using
    the same hourly Wilder-14 transform as the rich parent policy.  No exact
    path, outcome, exit, or future-bar quantity enters this function.
    """
    source = route.copy()
    records: list[dict[str, object]] = []
    cache: dict[str, pd.DataFrame] = {}
    for symbol, group in source.groupby("symbol", sort=True):
        path = BARS_ROOT / _symbol_filename(str(symbol))
        if not path.is_file():
            for _, row in group.iterrows():
                records.append({"candidate_id": row["candidate_id"], "feature_source_status": "missing_15m_source"})
            continue
        try:
            bars = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
        except Exception as exc:
            # E2 is an optional, replacement-only authority.  A local 15m
            # source failure must never abort the live-style control or be
            # converted into an imputed feature.  Mark its rows unavailable
            # so the pair builder can fail closed for E2 only.
            status = f"unreadable_15m_source:{type(exc).__name__}"
            for _, row in group.iterrows():
                records.append({"candidate_id": row["candidate_id"], "feature_source_status": status})
            continue
        bars.index = _utc(bars.index)
        bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
        signal_atr = _hourly_signal_atr(bars)
        cache[str(symbol)] = bars
        values = bars.index.asi8
        for _, row in group.iterrows():
            decision = pd.Timestamp(row["timestamp"])
            loc = int(np.searchsorted(values, decision.value, side="left"))
            # The feature helpers copy only this bounded causal tail rather
            # than the complete symbol history on each candidate.
            tail = bars.iloc[max(0, loc - 104):loc].copy()
            base: dict[str, float]
            vwap: dict[str, float]
            status = "ok"
            try:
                atr_value = float(signal_atr.reindex(pd.DatetimeIndex([decision])).iloc[0])
                if not np.isfinite(atr_value) or atr_value <= 0.0:
                    raise ValueError("missing_signal_atr")
                base = compute_15m_features(tail, decision, signal_atr=atr_value, side="long")
                vwap = compute_15m_vwap_features(tail, decision, signal_atr=atr_value, side="long")
            except Exception as exc:  # candidate-local, fail closed for E2 only
                base = {name: np.nan for name in FIFTEEN_MINUTE_FEATURE_KEYS}
                vwap = {name: np.nan for name in VWAP_15M_FEATURE_KEYS}
                status = f"feature_error:{type(exc).__name__}"
            record: dict[str, object] = {"candidate_id": row["candidate_id"], "feature_source_status": status}
            record.update(base)
            record.update(vwap)
            records.append(record)
    features = pd.DataFrame(records)
    if features["candidate_id"].duplicated().any() or len(features) != len(route):
        raise AssertionError("target-free feature materialization lost identity")
    features["finite_feature_count"] = features.loc[:, [*FIFTEEN_MINUTE_FEATURE_KEYS, *VWAP_15M_FEATURE_KEYS]].notna().sum(axis=1)
    return features


def _pair_frame(frame: pd.DataFrame, feature_fields: tuple[str, ...], *, labelled: bool) -> pd.DataFrame:
    """Make bounded reserve-vs-marginal pairs from the live dual-50 route."""
    records: list[dict[str, object]] = []
    raw_fields = tuple(dict.fromkeys(
        name.removeprefix("margin__") for name in feature_fields
        if name != "incumbent_bcf_mc1_expected_bps"
    ))
    for timestamp, group in frame.groupby("timestamp", sort=True):
        ranked = group.sort_values(
            ["bcf_mc1_expected_bps", "bcf_final_score", "candidate_id"],
            ascending=[False, False, True], kind="stable",
        ).reset_index(drop=True)
        if len(ranked) <= MAX_NEW:
            continue
        incumbent = ranked.iloc[MAX_NEW - 1]
        # An E2 replacement is allowed only when both candidates have the
        # complete selected causal 15-minute contract.  The ordinary BCF
        # auction remains available for every sealed control row.
        if str(incumbent.get("feature_source_status", "ok")) != "ok":
            continue
        for _, reserve in ranked.iloc[MAX_NEW:].iterrows():
            if str(reserve.get("feature_source_status", "ok")) != "ok":
                continue
            row: dict[str, object] = {
                "timestamp": timestamp,
                "reserve_candidate_id": str(reserve["candidate_id"]),
                "incumbent_candidate_id": str(incumbent["candidate_id"]),
                "reserve_bcf_mc1_expected_bps": float(reserve["bcf_mc1_expected_bps"]),
                "reserve_dual_mc1_min_bps": float(reserve["dual_mc1_min_bps"]),
                "incumbent_bcf_mc1_expected_bps": float(incumbent["bcf_mc1_expected_bps"]),
            }
            for name in feature_fields:
                if name == "incumbent_bcf_mc1_expected_bps":
                    row[name] = float(incumbent["bcf_mc1_expected_bps"])
                elif name.startswith("margin__"):
                    key = name.removeprefix("margin__")
                    row[name] = float(reserve[key]) - float(incumbent[key])
                else:
                    row[name] = reserve[name]
            # LightGBM's native missing-value routing is part of the frozen
            # E2 model family: several path-state features are genuinely
            # undefined (for example, no completed pullback).  Fail closed
            # only on an unavailable source above; do not confuse a valid,
            # semantically undefined feature with a source failure.
            if labelled:
                if not (bool(reserve["outcome_available"]) and bool(incumbent["outcome_available"])):
                    continue
                row["pair_advantage_bps"] = float(reserve["exact_net_bps"]) - float(incumbent["exact_net_bps"])
            records.append(row)
    return pd.DataFrame(records)


def _fit_pairwise(train: pd.DataFrame, fields: tuple[str, ...], *, depth: int, leaves: int, child_fraction: float, l2: float):
    child = max(8, int(np.ceil(len(train) * child_fraction)))
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=.50, n_estimators=350, learning_rate=.03,
        max_depth=depth, num_leaves=leaves, min_child_samples=child,
        subsample=.80, colsample_bytree=.80, reg_lambda=l2,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    # The archived 20--30 weighting becomes a constant on a dual-50-only
    # population.  Unit weights are thus the only non-arbitrary successor.
    model.fit(train.loc[:, fields], train["pair_advantage_bps"].to_numpy(float), sample_weight=np.ones(len(train)))
    return model


def _apply_e2(route: pd.DataFrame, pairs: pd.DataFrame, fields: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bounded E2 intersection: swap only one marginal incumbent per timestamp."""
    h0 = _fit_pairwise(pairs, fields, depth=3, leaves=7, child_fraction=.03, l2=8.0)
    h3 = _fit_pairwise(pairs, fields, depth=2, leaves=3, child_fraction=.04, l2=12.0)
    test_pairs = _pair_frame(route, fields, labelled=False)
    if test_pairs.empty:
        return route.assign(
            e2_action="no_reserve",
            e2_priority_bps=route["bcf_mc1_expected_bps"].astype(float),
        ), test_pairs
    test_pairs["h0_q50_advantage_bps"] = h0.predict(test_pairs.loc[:, fields])
    test_pairs["h3_q50_advantage_bps"] = h3.predict(test_pairs.loc[:, fields])
    test_pairs["e2_qualified"] = test_pairs["h0_q50_advantage_bps"].ge(50.0) & test_pairs["h3_q50_advantage_bps"].ge(50.0)
    action = route.copy()
    action["e2_action"] = "unchanged"
    action["e2_priority_bps"] = action["bcf_mc1_expected_bps"].astype(float)
    logs: list[dict[str, object]] = []
    for timestamp, group in test_pairs.loc[test_pairs["e2_qualified"]].groupby("timestamp", sort=True):
        proposal = group.sort_values(
            ["h0_q50_advantage_bps", "h3_q50_advantage_bps", "reserve_bcf_mc1_expected_bps", "reserve_candidate_id"],
            ascending=[False, False, False, True], kind="stable",
        ).iloc[0]
        reserve_id, incumbent_id = str(proposal["reserve_candidate_id"]), str(proposal["incumbent_candidate_id"])
        at = action["timestamp"].eq(timestamp)
        reserve = at & action["candidate_id"].eq(reserve_id)
        incumbent = at & action["candidate_id"].eq(incumbent_id)
        if reserve.sum() != 1 or incumbent.sum() != 1:
            raise AssertionError("E2 target-free pair lost identity")
        marginal_priority = float(action.loc[incumbent, "e2_priority_bps"].iloc[0])
        # Exactly one local swap.  Other candidates retain the BCF priority
        # used by the live control, and portfolio capacity is unchanged.
        action.loc[reserve, "e2_priority_bps"] = np.nextafter(marginal_priority, np.inf)
        # Keep the candidate table finite for the portfolio normaliser while
        # making the demoted incumbent unambiguously lose every auction tie.
        action.loc[incumbent, "e2_priority_bps"] = float(action["e2_priority_bps"].min()) - 1_000_000.0
        action.loc[reserve, "e2_action"] = "e2_50_replacement"
        action.loc[incumbent, "e2_action"] = "e2_50_demoted_marginal"
        logs.append({
            "timestamp": timestamp, "reserve_candidate_id": reserve_id,
            "incumbent_candidate_id": incumbent_id,
            "h0_q50_advantage_bps": float(proposal["h0_q50_advantage_bps"]),
            "h3_q50_advantage_bps": float(proposal["h3_q50_advantage_bps"]),
        })
    return action, pd.concat([test_pairs, pd.DataFrame(logs)], ignore_index=True, sort=False)


def _candidate_table(outcomes: pd.DataFrame, priorities: pd.Series) -> pd.DataFrame:
    frame = outcomes.copy()
    table = pd.DataFrame({
        "timestamp": frame["entry_ts"], "decision_timestamp": frame["timestamp"],
        "candidate_id": frame["candidate_id"], "symbol": frame["symbol"], "side": "long",
        "strategy_id": "strict_r3_p8u_v58_e2_50_exact_research",
        "policy_archetype": "strict_r3_p8u_v58_e2_50_exact_research",
        "normalized_rank_score": 1.0, "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0, "calibrated_score": 1.0,
        "portfolio_priority_adjustment": priorities.to_numpy(float),
        "entry_price": frame["exact_entry_price"], "exit_timestamp": frame["exact_exit_ts"],
        "exit_price": frame["exact_exit_price"], "net_return": frame["exact_net_bps"] / 10_000.0,
        "gross_return": frame["exact_gross_bps"] / 10_000.0,
        "holding_bars": np.maximum(frame["exact_exit_minute"].to_numpy(int) + 1, 1),
        "simple_policy_exit_reason": frame["exact_exit_reason"], "fees_bps": 100.0,
        "expected_friction_bps": 0.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "policy_outcome_available": True,
        "slippage_bps": 0.0,
    })
    return normalise_candidate_table(table)


def _replay(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    params = portfolio_params()
    decisions, equity, _ = replay_candidates(table, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp")
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    indices = pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int).to_numpy()
    bps = pd.to_numeric(table.iloc[indices]["net_return"], errors="raise").to_numpy(float) * 10_000.0
    metrics = compute_replay_metrics(table, decisions, equity, params=params)
    metrics.update({"portfolio_accepted": int(len(accepted)), "net_bps_per_trade": float(np.mean(bps)) if len(bps) else np.nan, "total_net_bps": float(np.sum(bps))})
    return decisions, accepted, equity, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-ledger", type=Path, default=DEFAULT_SCORE_LEDGER)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-month", default="2026-07")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    held = pd.Timestamp(f"{args.held_month}-01", tz="UTC")
    held_end = held + pd.offsets.MonthBegin(1)
    route = _load_route(args.score_ledger.resolve(), args.path_root.resolve())
    route_features = _materialize_target_free_features(route)
    route_features.to_parquet(output.with_name(output.name + "_targetfree_features_staging.parquet"), index=False, compression="zstd")
    # Target-free static source is now immutable.  Only after persisting it do
    # we materialise and join exact one-minute outcome labels.
    outcomes = _load_exact_outcomes(args.path_root.resolve(), args.policy.resolve())
    panel = route.merge(route_features, on="candidate_id", how="inner", validate="one_to_one")
    panel = panel.merge(outcomes.drop(columns=["timestamp", "entry_ts", "symbol"]), on="candidate_id", how="left", validate="one_to_one")
    panel["outcome_available"] = panel["exact_net_bps"].notna()
    fields = _read_features(args.feature_contract.resolve(), args.held_month)
    train = panel.loc[panel["timestamp"].lt(held) & panel["outcome_available"]].copy()
    test = panel.loc[panel["timestamp"].ge(held) & panel["timestamp"].lt(held_end)].copy()
    train_pairs = _pair_frame(train, fields, labelled=True)
    if len(train_pairs) < 500:
        raise RuntimeError(f"E2-50 exact train support is too small: {len(train_pairs)} pairs")
    action, proposal_audit = _apply_e2(test, train_pairs, fields)
    output.mkdir(parents=True, exist_ok=False)
    route_features.rename(columns={"feature_source_status": "targetfree_feature_source_status"}).to_parquet(output / "target_free_e2_features.parquet", index=False, compression="zstd")
    test.loc[:, ["candidate_id", "timestamp", "symbol", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps"]].to_parquet(output / "control_selection_target_free.parquet", index=False, compression="zstd")
    action.loc[:, ["candidate_id", "timestamp", "symbol", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps", "e2_action", "e2_priority_bps"]].to_parquet(output / "e2_50_selection_target_free.parquet", index=False, compression="zstd")
    train_pairs.to_parquet(output / "e2_50_train_pairs_prior_resolved.parquet", index=False, compression="zstd")
    proposal_audit.to_parquet(output / "e2_50_proposal_audit_target_free.parquet", index=False, compression="zstd")
    valid = test.loc[test["outcome_available"]].copy()
    control_table = _candidate_table(valid, valid["bcf_mc1_expected_bps"])
    e2_priority = action.loc[:, ["candidate_id", "e2_priority_bps"]]
    e2_valid = valid.merge(e2_priority, on="candidate_id", how="inner", validate="one_to_one")
    e2_table = _candidate_table(e2_valid, e2_valid["e2_priority_bps"])
    results: list[dict[str, object]] = []
    for arm, table in (("C0_live_style_exact_control", control_table), ("E2_50_q50_agreement", e2_table)):
        decisions, accepted, equity, metrics = _replay(table)
        table.to_parquet(output / f"{arm}_portfolio_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(output / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(output / f"{arm}_accepted_trades.parquet", index=False, compression="zstd")
        equity.to_parquet(output / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
        results.append({"arm": arm, **metrics})
    summary = pd.DataFrame(results)
    ref = summary.loc[summary["arm"].eq("C0_live_style_exact_control")].iloc[0]
    for metric in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
        summary[f"delta_vs_C0_{metric}"] = summary[metric] - ref[metric]
    # Only the supplied external control covers August.  Other held months
    # remain matched, same-run controls and must not be falsely described as
    # bit-identical to that August receipt.
    actual = pd.read_parquet(output / "C0_live_style_exact_control_accepted_trades.parquet")
    if args.held_month == "2026-08":
        expected = pd.read_parquet(args.control_root.resolve() / "accepted_trades.parquet")
        expected_ids = expected["candidate_id"].astype(str).tolist()
        actual_ids = actual["candidate_id"].astype(str).tolist()
        parity = {"accepted_identity_equal": expected_ids == actual_ids, "expected_accepted": len(expected_ids), "actual_accepted": len(actual_ids)}
        if not parity["accepted_identity_equal"]:
            raise AssertionError("E2 exact control does not reproduce sealed live-style August auction")
    else:
        parity = {"accepted_identity_equal": None, "control": "same-run exact matched control; no external receipt for this held month", "actual_accepted": len(actual)}
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    coverage = pd.DataFrame({
        "stage": ["sealed_target_free_route", "exact_outcome_available", "direct_causal_e2_feature_ok", "E2_50_replacements"],
        "rows": [len(test), int(test["outcome_available"].sum()), int(route_features.loc[route_features.candidate_id.isin(test.candidate_id), "finite_feature_count"].ge(50).sum()), int((action["e2_action"] == "e2_50_replacement").sum())],
    })
    coverage.to_parquet(output / "coverage_audit.parquet", index=False)
    staging = output.with_name(output.name + "_targetfree_features_staging.parquet")
    staging.replace(output / "target_free_e2_features_staging_source.parquet")
    manifest = {
        "schema": "strict-r3-p8u-v58-e2-50-exact-matched-v1",
        "scope": "offline research only; no live config, execution code, exchange IO, or order submission",
        "control": "sealed dual-MC1>=50 route; BCF MC1 priority; exact +5m one-minute rich parent outcomes; normal portfolio auction",
        "E2_50": "two frozen q50 E2 geometries intersect; one already-dual50 reserve may replace only the second BCF incumbent; no new capacity",
        "support_change": "original 20--30 reserve is empty under dual>=50, so E2_50 uses live-admissible reserves and unit weights; it is not the archived E2 artifact",
        "target_free_before_outcomes": True,
        "feature_contract": str(args.feature_contract.resolve()), "feature_contract_sha256": _sha256(args.feature_contract.resolve()),
        "score_ledger": str(args.score_ledger.resolve()), "score_ledger_sha256": _sha256(args.score_ledger.resolve()),
        "path_root": str(args.path_root.resolve()),
        "path_files_sha256": {name: _sha256(args.path_root.resolve() / name) for name in ("target_free_candidates.parquet", "valid_exact_paths_rows.parquet", "exact_paths.npz")},
        "policy": str(args.policy.resolve()), "policy_sha256": _sha256(args.policy.resolve()),
        "held_month": args.held_month,
        "train_period": f"{train.timestamp.min().isoformat()} through {train.timestamp.max().isoformat()}",
        "control_parity": parity,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
