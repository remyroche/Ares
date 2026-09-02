#!/usr/bin/env python3
"""Monthly-prequential test of the sealed MC1_d2 -> LambdaRank auction arm.

This is deliberately an *offline* continuation of the two-stage ablation.  It
does not change the frozen MC1_d2 admission model, its +50-bps threshold, or
any live/inference artifact.  It asks one narrow deployment question:

    Does re-fitting the already-selected second-stage auction ranker with only
    labels resolved before each month preserve its benefit over frozen
    final-score ordering?

The model/hyperparameters were selected previously on chronological 2025
folds.  This runner neither runs HPO nor changes the feature contract.  Every
test row is first admitted by the immutable MC1 score; LambdaRank only orders
the already admitted timestamp-local candidates for the portfolio auction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _candidate_table,
    _metrics,
    _params as portfolio_params,
)
from scripts.run_strict_r3_mc1_d2_two_stage_auction_ranker import (  # noqa: E402
    _fit_predict,
)
from extreme_price_movements.portfolio_policy_replay import replay_candidates  # noqa: E402


DEFAULT_CONTROL = ROOT / (
    "data_perp/artifacts/strict_r3_mc1_d2_historical_strictlt_2025_2026_"
    "20260816_v1/predictions_mc1_d2_historical_parity.parquet"
)
DEFAULT_HUBER = ROOT / (
    "data_perp/artifacts/strict_r3_mc1_d2_target_loss_ablation_2025hpo_"
    "strictlt_20260816_v1/predictions_huber_asin.parquet"
)
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_"
    "strictfull_prior28_optimizedpolicy_20260813_v2/"
    "walkforward_scored_label_ledger.parquet"
)
DEFAULT_WINNER = ROOT / (
    "data_perp/artifacts/strict_r3_mc1_d2_two_stage_auction_ranker_"
    "20260816_v2/hpo_winner.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _attach_outcomes(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Restore explicit policy-outcome provenance required by shared metrics."""
    if decisions.empty:
        out = decisions.copy()
        out["policy_outcome_available"] = pd.Series(dtype=bool)
        return out
    if "candidate_index" not in decisions:
        raise ValueError("portfolio decision lacks candidate_index provenance")
    lookup = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
    lookup.index.name = "candidate_index"
    out = decisions.merge(lookup, on="candidate_index", how="left", validate="many_to_one")
    if out["policy_outcome_available"].isna().any():
        raise ValueError("portfolio decision is missing policy outcome provenance")
    return out


def _replay(
    prediction: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    candidates = _candidate_table(prediction, policy, 50.0)
    decisions, equity, _ = replay_candidates(
        candidates,
        portfolio_params(),
        mode="global_auction",
        ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps",
        initial_wallet=1000.0,
    )
    decisions = _attach_outcomes(decisions, candidates)
    return decisions, equity, _metrics(decisions, equity, arm, "2026-01..07")


def _month_metrics(decisions: pd.DataFrame, arm: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True)
    accepted = accepted.loc[accepted.policy_outcome_available.fillna(False).astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["arm", "month", "trades", "net_ev_bps", "net_sum_bps"])
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
    grouped = accepted.groupby(accepted.timestamp.dt.strftime("%Y-%m"), sort=True)["net_bps"]
    out = grouped.agg(trades="size", net_ev_bps="mean", net_sum_bps="sum").reset_index(names="month")
    out.insert(0, "arm", arm)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--huber", type=Path, default=DEFAULT_HUBER)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_mc1_d2_two_stage_monthly_refit_20260816_v1",
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    for path in (args.control, args.huber, args.ledger, args.winner):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True)

    control_columns = [
        "candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps",
        "base_rank42", "conditional_consensus_rank", "upstream",
        "ordinary_shadow_consensus_rank", "correctness_rank",
    ]
    control = pd.read_parquet(args.control, columns=control_columns).rename(
        columns={"final_score": "frozen_final_score", "mc1_expected_bps": "frozen_mc1_expected_bps"},
    )
    control["__decision_ts__"] = pd.to_datetime(control["__decision_ts__"], utc=True)
    huber = pd.read_parquet(args.huber, columns=["candidate_id", "mc1_expected_bps"]).rename(
        columns={"mc1_expected_bps": "huber_expected_bps"},
    )
    policy_columns = [
        "candidate_id", "__symbol__", "policy_label_available_ts", "policy_path_valid",
        "policy_net_bps", "policy_gross_bps", "policy_exit_bar_15m", "policy_entry_price",
        "policy_exit_price", "policy_exit_reason",
    ]
    policy = pd.read_parquet(args.ledger, columns=policy_columns)
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True)
    if policy.candidate_id.duplicated().any():
        raise ValueError("policy candidate identity must be unique")
    data = control.merge(huber, on="candidate_id", how="inner", validate="one_to_one").merge(
        policy, on="candidate_id", how="inner", validate="one_to_one",
    )
    # MC1 is the immutable admission authority.  This filter is target-free at
    # inference time: it uses the frozen mapper prediction, never policy data.
    data = data.loc[data.frozen_mc1_expected_bps.ge(50.0)].copy()
    if data.empty:
        raise ValueError("no rows pass frozen MC1 +50-bps admission")
    winner = json.loads(args.winner.read_text())

    all_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    starts = pd.date_range("2026-01-01", "2026-07-01", freq="MS", tz="UTC")
    for start in starts:
        stop = start + pd.offsets.MonthBegin(1)
        train = data.loc[
            data["__decision_ts__"].lt(start)
            & data.policy_label_available_ts.lt(start)
            & data.policy_path_valid.fillna(False).astype(bool)
            & data.policy_net_bps.notna()
        ].copy()
        held = data.loc[data["__decision_ts__"].between(start, stop, inclusive="left")].copy()
        if len(train) < 3_000 or held.empty:
            raise ValueError(f"insufficient strict train/held support at {start.isoformat()}: {len(train)}, {len(held)}")
        held["ranker_score"] = _fit_predict(train, held, winner, seed=args.seed)
        held["fold_start"] = start
        all_predictions.append(held)
        fold_rows.append({
            "fold_start": start.isoformat(), "fold_stop": stop.isoformat(),
            "strict_train_rows": int(len(train)), "held_mc1_admitted_rows": int(len(held)),
            "max_training_label_available_ts": train.policy_label_available_ts.max().isoformat(),
        })
        print(json.dumps({"event": "fold_complete", **fold_rows[-1]}), flush=True)

    predicted = pd.concat(all_predictions, ignore_index=True)
    monthly_prediction = predicted.rename(
        columns={"frozen_mc1_expected_bps": "mc1_expected_bps", "ranker_score": "final_score"},
    )
    baseline_prediction = predicted.rename(
        columns={"frozen_mc1_expected_bps": "mc1_expected_bps", "frozen_final_score": "final_score"},
    )
    monthly_decisions, monthly_equity, monthly_metric = _replay(
        monthly_prediction, policy, arm="monthly_prequential_two_stage",
    )
    baseline_decisions, baseline_equity, baseline_metric = _replay(
        baseline_prediction, policy, arm="frozen_final_score_same_mc1_population",
    )
    metrics = pd.DataFrame([baseline_metric, monthly_metric])
    deltas = {
        key: float(monthly_metric[key]) - float(baseline_metric[key])
        for key in (
            "accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised",
            "worst_month_bps", "worst_week_bps", "max_drawdown",
        )
    }
    predicted.to_parquet(args.out_dir / "monthly_prequential_predictions_2026.parquet", index=False, compression="zstd")
    monthly_decisions.to_parquet(args.out_dir / "monthly_prequential_decisions_2026.parquet", index=False, compression="zstd")
    monthly_equity.to_parquet(args.out_dir / "monthly_prequential_equity_2026.parquet", index=False, compression="zstd")
    baseline_decisions.to_parquet(args.out_dir / "baseline_decisions_2026.parquet", index=False, compression="zstd")
    baseline_equity.to_parquet(args.out_dir / "baseline_equity_2026.parquet", index=False, compression="zstd")
    pd.concat([
        _month_metrics(baseline_decisions, "frozen_final_score_same_mc1_population"),
        _month_metrics(monthly_decisions, "monthly_prequential_two_stage"),
    ], ignore_index=True).to_parquet(args.out_dir / "monthly_metrics_2026.parquet", index=False)
    metrics.to_parquet(args.out_dir / "portfolio_metrics_2026.parquet", index=False)
    (args.out_dir / "fold_ledger.json").write_text(json.dumps(fold_rows, indent=2) + "\n")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_two_stage_monthly_prequential_v1",
        "status": "complete",
        "purpose": "offline monthly refit of preselected LambdaRank auction only",
        "admission": "frozen MC1_d2 expected policy net >= +50 bps; unmodified",
        "auction_control": "frozen final score on identical MC1-admitted candidate population",
        "challenger": "monthly prequential LambdaRank score only on that same population",
        "target": "six-bin policy net ordinal: <-200, -200..-50, -50..50, 50..150, 150..250, >250 bps",
        "params": winner,
        "seed": args.seed,
        "training": "policy_label_available_ts < each held-month start",
        "evaluation": "January-July 2026; chronological portfolio replay after fixed predictions",
        "delta_monthly_ranker_minus_control": deltas,
        "sources": {str(path): _sha256(path) for path in (args.control, args.huber, args.ledger, args.winner)},
        "exclusions": ["R5", "live state", "exchange I/O", "MC1 admission retuning", "HPO"],
        "promotion": "research-only; requires untouched forward validation before any canonical/live change",
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "control": baseline_metric, "challenger": monthly_metric, "delta": deltas}), flush=True)


if __name__ == "__main__":
    main()
