#!/usr/bin/env python3
"""Four-arm, full-population v58 replay using exact H4 state semantics.

This is a bounded *transfer* promotion diagnostic: H4's frozen C4 selection
and L1 geometry are fit only on prior-resolved historical H4 labels, then
scored on the newly materialised exact +5m states.  The exit path itself is
fully exact one-minute.  The manifest explicitly distinguishes this from a
future exact-H4-label refit, so the result cannot silently be promoted.
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
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as h4_study
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4_hpo
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import (
    DEFAULT_PATH_ROOT,
    DEFAULT_POLICY,
    DEFAULT_SCORE_LEDGER,
    _candidate_table,
    _load_policy,
    _replay,
)


STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_exact_h4_states_20260830_v1"
FEATURE_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2/stable_selected_features.parquet"
E2_JUNE = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_e2_50_exact_matched_20260830_v4_causal_atr_june/e2_50_selection_target_free.parquet"
E2_JULY = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_e2_50_exact_matched_20260830_v3_causal_atr/e2_50_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_five_arm_h4_transfer_20260830_v2"
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
    values = tuple(rows["feature"].astype(str))
    if len(values) != 45 or len(set(values)) != len(values):
        raise AssertionError(f"{held:%Y-%m}: missing frozen 45-field C4 contract")
    return values


def _fit(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMRegressor:
    spec = h4_hpo.SPECS["H4_l1_d4_l15_leaf5_reg20"]
    min_child = max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=int(spec["n_estimators"]), learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]), num_leaves=int(spec["num_leaves"]), min_child_samples=min_child,
        subsample=.80, colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]),
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    weights = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], train["activation50_advantage_bps"].to_numpy(float), sample_weight=weights)
    return model


def _outcome_frame(route: pd.DataFrame, outcome: pd.DataFrame, *, net: str, gross: str, exit_ts: str, exit_price: str, exit_minute: str, exit_reason: str) -> pd.DataFrame:
    frame = route.merge(outcome, on="candidate_id", how="inner", validate="one_to_one")
    result = frame.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol", "bcf_mc1_expected_bps"]].copy()
    result["exact_entry_price"] = frame["entry_price"].to_numpy(float)
    result["exact_net_bps"] = frame[net].to_numpy(float)
    result["exact_gross_bps"] = frame[gross].to_numpy(float)
    result["exact_exit_ts"] = pd.to_datetime(frame[exit_ts], utc=True)
    result["exact_exit_price"] = frame[exit_price].to_numpy(float)
    result["exact_exit_minute"] = frame[exit_minute].to_numpy(int)
    result["exact_exit_reason"] = frame[exit_reason].astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--feature-contract", type=Path, default=FEATURE_CONTRACT)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    state_root = args.state_root.resolve()
    route = pd.read_parquet(state_root / "target_free_route.parquet")
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="raise")
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    parent = pd.read_parquet(state_root / "exact_parent_outcomes.parquet")
    states = pd.read_parquet(state_root / "exact_parent_h4_state_features_target_free.parquet")
    states["entry_decision_ts"] = pd.to_datetime(states["entry_decision_ts"], utc=True, errors="raise")
    states["state_decision_ts"] = pd.to_datetime(states["state_decision_ts"], utc=True, errors="raise")
    if parent.candidate_id.nunique() != len(route):
        raise AssertionError("parent exact outcomes do not cover sealed route")

    # Strict-prior historical H4 labels.  They define H4's frozen target and
    # model geometry, while test states are the new full exact +5m panel.
    old = h4_study._load_panel(h4_study.TARGET_PANEL, h4_study.VWAP_PANEL)
    actions: list[pd.DataFrame] = []
    for held in (pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")):
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=4)
        fields = _fields(args.feature_contract.resolve(), held)
        train = old.loc[
            old["entry_decision_ts"].ge(start) & old["entry_decision_ts"].lt(held)
            & old["policy_label_available_ts"].lt(held)
            & pd.to_numeric(old["MC1_expected_bps"], errors="coerce").ge(30.0)
        ].copy()
        test = states.loc[states["entry_decision_ts"].ge(held) & states["entry_decision_ts"].lt(end)].copy()
        if train.candidate_id.nunique() < 100 or test.empty:
            raise RuntimeError(f"{held:%Y-%m}: incomplete strict-prior H4 fold")
        missing = set(fields).difference(train.columns) | set(fields).difference(test.columns)
        if missing:
            raise AssertionError(f"{held:%Y-%m}: missing H4 fields {sorted(missing)}")
        model = _fit(train, fields)
        prediction = model.predict(test.loc[:, fields])
        action = test.loc[:, ["candidate_id", "state_decision_ts", "state_minute"]].copy()
        action["h4_predicted_activation50_advantage_bps"] = prediction
        action["h4_enable_giveback20"] = action["h4_predicted_activation50_advantage_bps"].ge(0.0)
        action["held_month"] = held.strftime("%Y-%m")
        actions.append(action)
    state_actions = pd.concat(actions, ignore_index=True)
    keys = ["candidate_id", "state_decision_ts"]
    if state_actions.duplicated(keys).any():
        raise AssertionError("H4 action schedule is not unique")

    rows = pd.read_parquet(args.path_root.resolve() / "valid_exact_paths_rows.parquet")
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    index = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows.candidate_id)
    route["__path_index__"] = route.candidate_id.astype(str).map(index)
    if route["__path_index__"].isna().any():
        raise AssertionError("four-arm path identity mismatch")
    archive = np.load(args.path_root.resolve() / "exact_paths.npz", allow_pickle=False)
    entry = np.asarray(archive["entry"], dtype=float); atr = np.asarray(archive["atr"], dtype=float)
    high = np.asarray(archive["high"], dtype=np.float32); low = np.asarray(archive["low"], dtype=np.float32); close = np.asarray(archive["close"], dtype=np.float32)
    params, median, _ = _load_policy(args.policy.resolve())
    schedules = {
        candidate: {
            pd.Timestamp(ts): bool(active)
            for ts, active in zip(group.state_decision_ts, group.h4_enable_giveback20, strict=True)
        }
        for candidate, group in state_actions.groupby("candidate_id", sort=False)
    }
    # This extra decomposition arm measures full H4 activation authority
    # versus the untouched parent separately from Giveback-20.
    h4_rows: list[dict[str, object]] = []
    for _, row in route.sort_values(["timestamp", "candidate_id"], kind="stable").iterrows():
        candidate = str(row["candidate_id"]); path_i = int(row["__path_index__"])
        schedule = schedules.get(candidate, {})
        def decide(state: dict[str, float], schedule=schedule) -> bool:
            return bool(schedule.get(pd.Timestamp(state["state_decision_ts"]), False))
        for variant, tighten in (("H4_activation50_only", 0.0), ("H4_giveback20", 0.20)):
            trace = replay_exact_1m_h4_giveback20(
                entry_price=float(entry[path_i]), signal_atr=float(atr[path_i]), entry_ts=row["entry_ts"],
                highs=high[path_i], lows=low[path_i], closes=close[path_i], params=params,
                median_atr_fraction=float(median), mc1_expected_bps=float(row["bcf_mc1_expected_bps"]),
                state_decider=decide, giveback_tighten=tighten, emit_states=False,
            )
            h4_rows.append({
                "variant": variant, "candidate_id": candidate, "entry_price": float(entry[path_i]),
                "h4_exact_net_bps": float(trace["net_bps"]), "h4_exact_gross_bps": float(trace["gross_bps"]),
                "h4_exit_ts": pd.Timestamp(trace["exit_timestamp"]), "h4_exit_price": float(trace["exit_price"]),
                "h4_exit_minute": int(trace["exit_minute"]), "h4_exit_reason": str(trace["exit_reason"]),
            })
    h4_outcome = pd.DataFrame(h4_rows)
    if h4_outcome.duplicated(["variant", "candidate_id"]).any() or len(h4_outcome) != 2 * len(route):
        raise AssertionError("H4 exact outcomes lost route identities")

    # E2 is the already completed strict-prior successor.  It has no May
    # authority because no prior v58 E2 fit was materialised for that month.
    selections: list[pd.DataFrame] = []
    for path in (E2_JUNE, E2_JULY):
        selection = pd.read_parquet(path)
        selection["candidate_id"] = selection["candidate_id"].astype(str)
        selections.append(selection.loc[:, ["candidate_id", "e2_priority_bps"]])
    e2_selected = pd.concat(selections, ignore_index=True)
    if e2_selected.candidate_id.duplicated().any():
        raise AssertionError("strict-prior E2 selections overlap across held months")
    priority = route.loc[:, ["candidate_id", "bcf_mc1_expected_bps"]].merge(
        e2_selected, on="candidate_id", how="left", validate="one_to_one",
    )
    priority["e2_priority_bps"] = priority["e2_priority_bps"].fillna(priority["bcf_mc1_expected_bps"])

    parent_enriched = parent.merge(route.loc[:, ["candidate_id", "__path_index__"]], on="candidate_id", how="inner", validate="one_to_one")
    parent_enriched["entry_price"] = np.asarray([entry[int(i)] for i in parent_enriched["__path_index__"]], dtype=float)
    parent_frame = _outcome_frame(
        route, parent_enriched,
        net="parent_exact_net_bps", gross="parent_exact_gross_bps", exit_ts="parent_exit_ts", exit_price="entry_price", exit_minute="parent_exit_minute", exit_reason="parent_exit_reason",
    )
    # Parent outcomes do not persist exit price; reproduce it from its exact
    # net return for the portfolio's mark-to-exit ledger.
    parent_frame["exact_exit_price"] = parent_frame["exact_entry_price"] * (1.0 + parent_frame["exact_gross_bps"] / 10_000.0)
    activation_frame = _outcome_frame(route, h4_outcome.loc[h4_outcome.variant.eq("H4_activation50_only")].drop(columns="variant"), net="h4_exact_net_bps", gross="h4_exact_gross_bps", exit_ts="h4_exit_ts", exit_price="h4_exit_price", exit_minute="h4_exit_minute", exit_reason="h4_exit_reason")
    h4_frame = _outcome_frame(route, h4_outcome.loc[h4_outcome.variant.eq("H4_giveback20")].drop(columns="variant"), net="h4_exact_net_bps", gross="h4_exact_gross_bps", exit_ts="h4_exit_ts", exit_price="h4_exit_price", exit_minute="h4_exit_minute", exit_reason="h4_exit_reason")
    arms = {
        "C0_current_live_style": (parent_frame, parent_frame["bcf_mc1_expected_bps"]),
        "E2_only": (parent_frame, parent_frame.merge(priority[["candidate_id", "e2_priority_bps"]], on="candidate_id", how="left", validate="one_to_one")["e2_priority_bps"]),
        "H4_activation50_only": (activation_frame, activation_frame["bcf_mc1_expected_bps"]),
        "H4_only": (h4_frame, h4_frame["bcf_mc1_expected_bps"]),
        "E2_H4": (h4_frame, h4_frame.merge(priority[["candidate_id", "e2_priority_bps"]], on="candidate_id", how="left", validate="one_to_one")["e2_priority_bps"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    for name, (frame, priorities) in arms.items():
        table = _candidate_table(frame, priorities)
        decisions, accepted, equity, metrics = _replay(table)
        table.to_parquet(output / f"{name}_portfolio_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(output / f"{name}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(output / f"{name}_accepted.parquet", index=False, compression="zstd")
        equity.to_parquet(output / f"{name}_equity.parquet", index=False, compression="zstd")
        results.append({"arm": name, **metrics})
    summary = pd.DataFrame(results)
    ref = summary.loc[summary.arm.eq("C0_current_live_style")].iloc[0]
    for metric in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "worst_week", "sortino", "compounded_return"):
        summary[f"delta_vs_C0_{metric}"] = summary[metric] - ref[metric]
    state_actions.to_parquet(output / "h4_action_schedule_target_free.parquet", index=False, compression="zstd")
    h4_outcome.to_parquet(output / "h4_exact_outcomes.parquet", index=False, compression="zstd")
    priority.to_parquet(output / "e2_priority_target_free.parquet", index=False, compression="zstd")
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict-r3-p8u-v58-five-arm-h4-transfer-v2",
        "scope": "offline promotion diagnostic only; no live mutation or exchange IO",
        "arms": ["current live-style control", "E2 only", "H4 activation-only", "H4 Giveback-20", "E2+H4 Giveback-20"],
        "shared_route": "Router50, dual BCF/current MC1 >=50, BCF priority, exact +5m one-minute paths",
        "portfolio": "each arm replays the same full chronological candidate population with normal global auction state",
        "E2": "precomputed strict-prior E2-50 June/July authority; May has no E2 authority",
        "H4": "full exact-state application with activation-only and Giveback-20 decomposition; frozen C4/H4 geometry trained on prior-resolved historical H4 labels; H4 label source has not yet been regenerated from exact +5m states",
        "state_root": str(state_root), "state_root_manifest_sha256": _sha256(state_root / "run_manifest.json"),
        "feature_contract": str(args.feature_contract.resolve()), "feature_contract_sha256": _sha256(args.feature_contract.resolve()),
        "path_root": str(args.path_root.resolve()), "policy": str(args.policy.resolve()), "policy_sha256": _sha256(args.policy.resolve()),
        "h4_spec": h4_hpo.SPECS["H4_l1_d4_l15_leaf5_reg20"], "seed": SEED,
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
