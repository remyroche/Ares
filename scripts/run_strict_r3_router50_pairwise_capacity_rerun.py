#!/usr/bin/env python3
"""Matched Router50 pairwise ordinal capacity-replacement rerun.

This offline challenger answers a narrow comparability question.  It uses the
same Router50 score/MCl ledger, canonical rich-policy labels, and global
portfolio replay as ``docs/pipeline_metrics_with_router.md``.  The model is
strictly out of sample: at every held month it trains only on the two prior
calendar months whose paired labels resolved before that month.

Authority is intentionally limited.  A reserve whose two MC1 maps are in
[20, 30) bps may replace, at most, the second BCF-MC1-ranked incumbent at the
same timestamp.  It can never create a third timestamp-level slot.  The
replacement is given just enough ordinal priority to take that incumbent's
place; predicted advantage is an eligibility test, not an unconstrained new
auction score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table, replay_candidates
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params


SCORE_ROOT = ROOT / "data_perp/artifacts/strict_r3_router50_baseN_metaN_mc1R_routedonly_20260826_v1"
CANONICAL_LABELS = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet"
FROZEN_CONTROL = SCORE_ROOT / "routed_base_dual_30_2026_marjul_decisions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_router50_pairwise_capacity_rerun_20260830_v1"
CORE_FLOOR = 30.0
RESERVE_FLOOR = 20.0
CAPACITY = 2
HELD_MONTHS = tuple(pd.date_range("2026-04-01", "2026-07-01", freq="MS", tz="UTC"))
SCORE_FIELDS = (
    "final_score", "mc1_expected_bps", "base_rank42", "conditional_consensus_rank",
    "ordinary_shadow_consensus_rank", "correctness_rank", "router_primary_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_target_free_scores(root: Path) -> pd.DataFrame:
    keep = ["candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", *SCORE_FIELDS]
    current = pd.read_parquet(root / "enhanced_current_mc1_predictions.parquet", columns=keep)
    bcf = pd.read_parquet(root / "enhanced_bcf_mc1_predictions.parquet", columns=keep)
    current = current.rename(columns={field: f"current_{field}" for field in SCORE_FIELDS})
    bcf = bcf.drop(columns=["__decision_ts__", "side_name", "enhanced_base_routed"]).rename(
        columns={field: f"bcf_{field}" for field in SCORE_FIELDS}
    )
    frame = current.merge(bcf, on="candidate_id", how="inner", validate="one_to_one")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    # Candidate identity is ``symbol|side|signal_timestamp`` and is part of
    # the immutable target-free score contract; recover the symbol locally
    # rather than borrowing it from the future-policy label ledger.
    frame["__symbol__"] = frame["candidate_id"].str.rsplit("|", n=2).str[0]
    if frame.duplicated("candidate_id").any():
        raise AssertionError("target-free router score identity is not unique")
    if not frame["side_name"].eq("long").all():
        raise AssertionError("this long-only rerun received a non-long score row")
    frame["dual_mc1_min_bps"] = frame[["current_mc1_expected_bps", "bcf_mc1_expected_bps"]].min(axis=1)
    if not frame["enhanced_base_routed"].fillna(False).astype(bool).all():
        raise AssertionError("Router50 score ledger unexpectedly includes unrouted rows")
    return frame


def _load_labels(path: Path, ids: pd.Index) -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    labels = pd.read_parquet(path, columns=columns, filters=[("candidate_id", "in", list(ids))])
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    if labels.duplicated("candidate_id").any():
        raise AssertionError("canonical policy label identity is not unique")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="raise")
    return labels


def _eligible_core(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[_numeric(frame["dual_mc1_min_bps"]).ge(CORE_FLOOR)].copy()


def _ranked_core(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(
        ["bcf_mc1_expected_bps", "bcf_final_score", "candidate_id"],
        ascending=[False, False, True], kind="stable",
    )


def _pair_features(reserve: pd.Series, incumbent: pd.Series) -> dict[str, float]:
    # All coordinates are frozen score/map outputs at the decision timestamp.
    record: dict[str, float] = {}
    for family in ("current", "bcf"):
        for field in SCORE_FIELDS:
            name = f"{family}_{field}"
            reserve_value = float(reserve[name]) if pd.notna(reserve[name]) else np.nan
            incumbent_value = float(incumbent[name]) if pd.notna(incumbent[name]) else np.nan
            record[f"reserve__{name}"] = reserve_value
            record[f"delta__{name}"] = reserve_value - incumbent_value
    record["reserve__dual_mc1_min_bps"] = float(reserve["dual_mc1_min_bps"])
    record["delta__dual_mc1_min_bps"] = float(reserve["dual_mc1_min_bps"] - incumbent["dual_mc1_min_bps"])
    record["incumbent_bcf_mc1_expected_bps"] = float(incumbent["bcf_mc1_expected_bps"])
    record["incumbent_current_mc1_expected_bps"] = float(incumbent["current_mc1_expected_bps"])
    return record


def _pairs(scores: pd.DataFrame, labels: pd.DataFrame | None, *, require_labels: bool) -> pd.DataFrame:
    """Reserve versus actual second incumbent; no one-core expansion is allowed."""
    records: list[dict[str, object]] = []
    core = _eligible_core(scores)
    lookup = labels.set_index("candidate_id", drop=False) if require_labels and labels is not None else None
    reserve = scores.loc[
        _numeric(scores["dual_mc1_min_bps"]).ge(RESERVE_FLOOR)
        & _numeric(scores["dual_mc1_min_bps"]).lt(CORE_FLOOR)
    ].copy()
    for timestamp, reserve_group in reserve.groupby("__decision_ts__", sort=True):
        incumbent_group = core.loc[core["__decision_ts__"].eq(timestamp)]
        # Strict capacity replacement, not capacity expansion: a real second
        # incumbent must exist before a reserve can be considered.
        if len(incumbent_group) < CAPACITY:
            continue
        incumbent = _ranked_core(incumbent_group).iloc[CAPACITY - 1]
        for _, candidate in reserve_group.iterrows():
            row: dict[str, object] = {
                "reserve_candidate_id": str(candidate["candidate_id"]),
                "incumbent_candidate_id": str(incumbent["candidate_id"]),
                "__decision_ts__": timestamp,
                "reserve_symbol": str(candidate["__symbol__"]),
                **_pair_features(candidate, incumbent),
            }
            if require_labels:
                if lookup is None:
                    raise AssertionError("labels are required for a labelled pair")
                reserve_label = lookup.loc[str(candidate["candidate_id"])] if str(candidate["candidate_id"]) in lookup.index else None
                incumbent_label = lookup.loc[str(incumbent["candidate_id"])] if str(incumbent["candidate_id"]) in lookup.index else None
                if reserve_label is None or incumbent_label is None:
                    continue
                if not bool(reserve_label["policy_path_valid"]) or not bool(incumbent_label["policy_path_valid"]):
                    continue
                reserve_net = _numeric(pd.Series([reserve_label["policy_net_bps"]])).iloc[0]
                incumbent_net = _numeric(pd.Series([incumbent_label["policy_net_bps"]])).iloc[0]
                if not np.isfinite(reserve_net) or not np.isfinite(incumbent_net):
                    continue
                row["pair_advantage_bps"] = float(reserve_net - incumbent_net)
                row["pair_label_available_ts"] = max(
                    pd.Timestamp(reserve_label["policy_label_available_ts"]),
                    pd.Timestamp(incumbent_label["policy_label_available_ts"]),
                )
            records.append(row)
    return pd.DataFrame(records)


def _feature_columns(pairs: pd.DataFrame) -> list[str]:
    return [
        column for column in pairs.columns
        if column.startswith(("reserve__", "delta__", "incumbent_"))
        and not column.endswith("_id")
    ]


def _fit(train: pd.DataFrame, features: list[str], quantile: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile", alpha=float(quantile), n_estimators=300, learning_rate=.03,
        max_depth=3, num_leaves=7, min_child_samples=max(20, int(np.ceil(len(train) * .025)),),
        subsample=.8, colsample_bytree=.8, reg_lambda=10.0, random_state=1729, n_jobs=2, verbosity=-1,
    ).fit(train.loc[:, features], _numeric(train["pair_advantage_bps"]))


def _replacement_selection(scores: pd.DataFrame, proposals: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the full ordinary admission set; swap only actual rank-two rows."""
    base = _eligible_core(scores).copy()
    base["ordinal_priority_bps"] = _numeric(base["bcf_mc1_expected_bps"])
    log: list[dict[str, object]] = []
    allowed = proposals.loc[_numeric(proposals["pair_lcb_advantage_bps"]).ge(threshold)].copy()
    if allowed.empty:
        return base, pd.DataFrame(log)
    allowed = allowed.sort_values(
        ["__decision_ts__", "pair_lcb_advantage_bps", "reserve_candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    by_id = scores.set_index("candidate_id", drop=False)
    for timestamp, choices in allowed.groupby("__decision_ts__", sort=True):
        choice = choices.iloc[0]
        incumbent_id = str(choice["incumbent_candidate_id"])
        reserve_id = str(choice["reserve_candidate_id"])
        active = base.loc[base["__decision_ts__"].eq(timestamp)]
        ranked = _ranked_core(active)
        if len(ranked) < CAPACITY or str(ranked.iloc[CAPACITY - 1]["candidate_id"]) != incumbent_id:
            log.append({**choice.to_dict(), "promoted": False, "reason": "marginal_changed"})
            continue
        if reserve_id not in by_id.index:
            raise AssertionError("proposal reserve is absent from frozen target-free scores")
        reserve = by_id.loc[reserve_id].copy()
        # The score only crosses the incumbent's rank boundary by an epsilon.
        # This avoids turning an advantage estimate into an unconstrained
        # portfolio-size authority.
        reserve["ordinal_priority_bps"] = float(ranked.iloc[CAPACITY - 1]["bcf_mc1_expected_bps"]) + 1e-6
        base = base.loc[~base["candidate_id"].eq(incumbent_id)].copy()
        base = pd.concat([base, pd.DataFrame([reserve])], ignore_index=True)
        log.append({**choice.to_dict(), "promoted": True, "reason": "replaced_second_incumbent"})
    if base.groupby("__decision_ts__").size().lt(0).any():
        raise AssertionError("unreachable timestamp capacity failure")
    return base, pd.DataFrame(log)


def _portfolio_input(selection: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    joined = selection.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    valid = (
        joined["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(_numeric(joined["policy_net_bps"]))
        & np.isfinite(_numeric(joined["policy_gross_bps"]))
        & np.isfinite(_numeric(joined["policy_exit_bar_15m"]))
    )
    unavailable = int((~valid).sum())
    joined = joined.loc[valid].copy()
    priority = _numeric(joined["ordinal_priority_bps"])
    joined["auction_rank"] = joined.groupby("__decision_ts__", sort=False)["ordinal_priority_bps"].rank(pct=True, method="average")
    exit_bar = _numeric(joined["policy_exit_bar_15m"]).astype(int)
    decision = pd.to_datetime(joined["__decision_ts__"], utc=True, errors="raise")
    candidates = pd.DataFrame({
        "timestamp": decision, "symbol": joined["__symbol__"].astype(str), "side": "long",
        # Preserve the parent Router50 strategy/archetype identity exactly:
        # the portfolio engine can use that coordinate for its chronological
        # allocation bookkeeping even when this offline challenger changes
        # only timestamp-local candidate membership.
        "strategy_id": "strict_r3_enhanced_live_stack_long", "policy_archetype": "strict_r3_enhanced_live_stack_long",
        "normalized_rank_score": joined["auction_rank"].to_numpy(float), "strategy_rank_pct": joined["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0, "calibrated_score": priority.to_numpy(float),
        "entry_price": _numeric(joined["policy_entry_price"]),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": _numeric(joined["policy_exit_price"]),
        "net_return": _numeric(joined["policy_net_bps"]) / 10_000.0,
        "gross_return": _numeric(joined["policy_gross_bps"]) / 10_000.0,
        "holding_bars": exit_bar + 1, "simple_policy_exit_reason": joined["policy_exit_reason"].astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"), "candidate_id": joined["candidate_id"].astype(str),
        "mapped_expected_net_bps": priority.to_numpy(float), "policy_outcome_available": np.ones(len(joined), dtype=bool),
    })
    return normalise_candidate_table(candidates), unavailable


def _replay(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> dict[str, object]:
    candidates, unavailable = _portfolio_input(selection, labels)
    decisions, equity, _ = replay_candidates(
        candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    # The normaliser deliberately drops this research-only coverage flag.
    # This rerun evaluates label-valid rows only, exactly like the Router50
    # receipt, so restore the equivalent terminal audit field.
    if "policy_outcome_available" not in decisions.columns:
        decisions["policy_outcome_available"] = True
    decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
    candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    values = _numeric(accepted["position_net_return"]) * 10_000.0
    result = _metrics(decisions, equity, arm, "2026_aprjul")
    result.update({
        "selected_target_free_rows": int(len(selection)), "outcome_unavailable_after_selection": unavailable,
        "portfolio_accepted": int(len(accepted)), "policy_net_bps_per_trade": float(values.mean()) if len(values) else np.nan,
        "total_policy_net_bps": float(values.sum()),
    })
    return result


def _exact_control_assertion(current: dict[str, object], frozen_path: Path) -> dict[str, object]:
    frozen = pd.read_parquet(frozen_path)
    frozen_accepted = frozen.loc[frozen["accepted"].fillna(False).astype(bool)]
    checks = {
        "accepted_rows_expected": int(len(frozen_accepted)),
        "accepted_rows_actual": int(current["accepted_rows"]),
        "net_ev_expected": float(_numeric(frozen_accepted["position_net_return"]).mean() * 10_000.0),
        "net_ev_actual": float(current["net_ev_bps_per_realised_trade"]),
        "total_expected": float(_numeric(frozen_accepted["position_net_return"]).sum() * 10_000.0),
        "total_actual": float(current["net_sum_bps_realised"]),
    }
    exact = (
        checks["accepted_rows_expected"] == checks["accepted_rows_actual"]
        and np.isclose(checks["net_ev_expected"], checks["net_ev_actual"], rtol=0.0, atol=1e-9)
        and np.isclose(checks["total_expected"], checks["total_actual"], rtol=0.0, atol=1e-6)
    )
    checks["exact"] = bool(exact)
    if not exact:
        raise AssertionError(f"Router50 control reconstruction differs from frozen receipt: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, default=SCORE_ROOT)
    parser.add_argument("--labels", type=Path, default=CANONICAL_LABELS)
    parser.add_argument("--frozen-control", type=Path, default=FROZEN_CONTROL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")

    scores = _load_target_free_scores(args.score_root.resolve())
    labels = _load_labels(args.labels.resolve(), pd.Index(scores["candidate_id"].unique()))
    missing = scores.loc[~scores["candidate_id"].isin(labels["candidate_id"]), "candidate_id"]
    if len(missing):
        raise AssertionError(f"canonical labels missing {len(missing)} Router50 score identities")
    eval_start, eval_end = HELD_MONTHS[0], HELD_MONTHS[-1] + pd.offsets.MonthBegin(1)
    eval_scores = scores.loc[scores["__decision_ts__"].ge(eval_start) & scores["__decision_ts__"].lt(eval_end)].copy()
    output.mkdir(parents=True)
    baseline = _eligible_core(eval_scores)
    baseline["ordinal_priority_bps"] = _numeric(baseline["bcf_mc1_expected_bps"])
    results = [_replay(baseline, labels, "B0_router50_dual30", output)]
    control = _exact_control_assertion(results[0], args.frozen_control.resolve())

    all_predictions: list[pd.DataFrame] = []
    all_logs: list[pd.DataFrame] = []
    # Q20 is the conservative lower-bound authority.  Q50 is a labelled
    # sensitivity only: it asks whether a central pair estimate has usable
    # replacement information after the exact Router50 baseline is restored.
    variant_specs = ((.20, "Q20_LCB", (0.0, 10.0, 20.0)), (.50, "Q50_MED", (0.0, 25.0, 50.0)))
    selections: dict[str, list[pd.DataFrame]] = {
        f"{prefix}{int(threshold)}": []
        for _, prefix, thresholds in variant_specs for threshold in thresholds
    }
    train_receipts: list[dict[str, object]] = []
    for held in HELD_MONTHS:
        end = held + pd.offsets.MonthBegin(1)
        train_start = held - pd.DateOffset(months=2)
        train_scores = scores.loc[scores["__decision_ts__"].ge(train_start) & scores["__decision_ts__"].lt(held)].copy()
        train_pairs = _pairs(train_scores, labels, require_labels=True)
        train_pairs = train_pairs.loc[pd.to_datetime(train_pairs["pair_label_available_ts"], utc=True).lt(held)].copy()
        test_scores = scores.loc[scores["__decision_ts__"].ge(held) & scores["__decision_ts__"].lt(end)].copy()
        test_pairs = _pairs(test_scores, None, require_labels=False)
        observed = set(pd.to_datetime(train_pairs["__decision_ts__"], utc=True).dt.strftime("%Y-%m")) if len(train_pairs) else set()
        required = {(held - pd.DateOffset(months=2)).strftime("%Y-%m"), (held - pd.DateOffset(months=1)).strftime("%Y-%m")}
        if not required.issubset(observed) or len(train_pairs) < 100 or test_pairs.empty:
            raise AssertionError(f"insufficient strict-OOS pair support for {held:%Y-%m}: {len(train_pairs)} rows / {observed}")
        features = _feature_columns(train_pairs)
        train_receipts.append({
            "held_month": held.strftime("%Y-%m"), "train_start": str(train_start), "train_rows": int(len(train_pairs)),
            "test_pairs": int(len(test_pairs)), "features": features,
            "all_labels_resolved_before_hold": bool(pd.to_datetime(train_pairs["pair_label_available_ts"], utc=True).lt(held).all()),
        })
        for quantile, prefix, thresholds in variant_specs:
            model = _fit(train_pairs, features, quantile)
            predicted = test_pairs.loc[:, ["reserve_candidate_id", "incumbent_candidate_id", "__decision_ts__", "reserve_symbol"]].copy()
            predicted["pair_lcb_advantage_bps"] = model.predict(test_pairs.loc[:, features])
            predicted["quantile"] = quantile
            predicted["held_month"] = held.strftime("%Y-%m")
            all_predictions.append(predicted)
            for threshold in thresholds:
                arm = f"{prefix}{int(threshold)}"
                selection, log = _replacement_selection(test_scores, predicted, threshold)
                selection["held_month"] = held.strftime("%Y-%m")
                log["held_month"] = held.strftime("%Y-%m") if len(log) else held.strftime("%Y-%m")
                log["arm"] = arm
                selections[arm].append(selection)
                all_logs.append(log)

    for arm, pieces in selections.items():
        selection = pd.concat(pieces, ignore_index=True)
        selection.to_parquet(output / f"{arm}_selection_target_free.parquet", index=False, compression="zstd")
        results.append(_replay(selection, labels, arm, output))
    summary = pd.DataFrame(results)
    base = summary.loc[summary["arm"].eq("B0_router50_dual30")].iloc[0]
    for column in ("accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown", "final_wallet"):
        summary[f"delta_vs_B0_{column}"] = summary[column] - base[column]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False, compression="zstd")
    pd.concat(all_predictions, ignore_index=True).to_parquet(output / "pairwise_predictions.parquet", index=False, compression="zstd")
    logs = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    logs.to_parquet(output / "promotion_log.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_router50_pairwise_capacity_rerun_v1",
        "scope": "offline challenger only; no live/canonical mutation or exchange I/O",
        "comparison": "exact Router50 dual-MC1 >=30 + full global rich-policy portfolio control",
        "control_parity": control,
        "score_root": str(args.score_root.resolve()), "score_root_manifest_sha256": _sha256(args.score_root.resolve() / "run_manifest.json"),
        "labels": str(args.labels.resolve()), "labels_sha256": _sha256(args.labels.resolve()),
        "held_months": [month.strftime("%Y-%m") for month in HELD_MONTHS],
        "training": "two prior complete months; pair labels must resolve before held boundary",
        "pair_target": "reserve rich-policy net bps minus actual timestamp rank-two BCF-MC1 incumbent net bps",
        "reserve": "both MC1 maps in [20,30) bps", "incumbent": "both MC1 maps >=30 bps",
        "authority": "one reserve may replace only an existing second incumbent; exact two-slot capacity is never expanded",
        "model": {"kind": "LightGBM quantile", "alphas": [.20, .50], "n_estimators": 300, "max_depth": 3, "num_leaves": 7, "seed": 1729},
        "portfolio": asdict(_params()), "cost": "canonical rich policy net includes 100 bps once",
        "feature_contract": "target-free Router50 current/BCF map, score, consensus, correctness, and router coordinates only",
        "selection_precedes_labels": True, "train_receipts": train_receipts,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
