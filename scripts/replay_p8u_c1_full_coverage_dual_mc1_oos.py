#!/usr/bin/env python3
"""Matched OOS portfolio comparison for full-coverage C1 S/R MC1 inputs.

This is a research-only replay.  It deliberately compares the retained frozen
dual-MC1 maps against the new, strict-prequential C1-augmented maps on the
*same* candidate identities and after attaching one canonical policy ledger.
It has no live, exchange, or portfolio authority.
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

from extreme_price_movements.portfolio_policy_replay import replay_candidates  # noqa: E402
from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import _to_candidates  # noqa: E402
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _metrics,
    _params,
)


ADMISSION_BPS = 50.0
POLICY_COLUMNS = (
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_outcome_source",
    "policy_cost_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_unique(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"duplicate candidate identity: {path}")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame


def _assert_same_numeric(left: pd.Series, right: pd.Series, name: str) -> None:
    if not np.isclose(
        pd.to_numeric(left, errors="raise").to_numpy(float),
        pd.to_numeric(right, errors="raise").to_numpy(float),
        equal_nan=True,
    ).all():
        raise AssertionError(f"source score mismatch for {name}")


def _attach_outcomes(target_free: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    if set(target_free["candidate_id"]) != set(outcome["candidate_id"]):
        raise AssertionError("outcome attachment candidate identities differ from target-free predictions")
    if any(column in target_free.columns for column in POLICY_COLUMNS):
        raise AssertionError("target-free panel must not already contain policy outcomes")
    attached = target_free.merge(
        outcome.loc[:, ["candidate_id", *POLICY_COLUMNS]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    label_ts = pd.to_datetime(attached["policy_label_available_ts"], utc=True, errors="raise")
    if (label_ts < attached["__decision_ts__"]).any():
        raise AssertionError("policy label resolved before its decision timestamp")
    return attached


def _valid_outcome(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_exit_bar_15m"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_entry_price"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_exit_price"], errors="coerce"))
    )


def _monthly(decisions: pd.DataFrame, arm: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions.get("accepted", False).fillna(False).astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["arm", "month", "accepted_trades", "net_ev_bps_per_trade", "net_sum_bps"])
    accepted["month"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.strftime("%Y-%m")
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="raise") * 10_000.0
    result = accepted.groupby("month", sort=True).agg(
        accepted_trades=("net_bps", "size"),
        net_ev_bps_per_trade=("net_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
    ).reset_index()
    result.insert(0, "arm", arm)
    return result


def _run_arm(panel: pd.DataFrame, *, arm: str, out: Path) -> tuple[dict[str, object], pd.DataFrame]:
    admitted = panel.loc[
        _valid_outcome(panel) & panel["dual_mc1_admitted"].fillna(False).astype(bool)
    ].copy()
    candidates = _to_candidates(
        panel,
        admission=_valid_outcome(panel) & panel["dual_mc1_admitted"].fillna(False).astype(bool),
        priority=pd.to_numeric(panel["auction_priority_bps"], errors="raise"),
    )
    decisions, equity, _ = replay_candidates(
        candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    if decisions.empty:
        decisions["policy_outcome_available"] = pd.Series(dtype=bool)
    else:
        provenance = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
        provenance.index.name = "candidate_index"
        decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
        if decisions["policy_outcome_available"].isna().any():
            raise AssertionError("portfolio decision lacks policy provenance")
    decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    metric = _metrics(decisions, equity, arm, "2026-05_to_2026-07")
    metric["dual_admitted_rows"] = int(len(admitted))
    metric["valid_outcome_rows"] = int(_valid_outcome(panel).sum())
    metric["candidate_rows"] = int(len(panel))
    metric["admission_threshold_bps"] = ADMISSION_BPS
    metric["auction_priority"] = "bcf_mc1_expected_bps"
    return metric, _monthly(decisions, arm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-root", required=True, type=Path)
    parser.add_argument("--frozen-bcf", required=True, type=Path)
    parser.add_argument("--frozen-current", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--challenger-name", default="c1_full_coverage_dual_mc1",
        help="immutable label for the re-fitted challenger arm",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    args.output.mkdir(parents=True)

    c1_target_free = _read_unique(args.c1_root / "dual_target_free_predictions.parquet")
    c1_outcome = _read_unique(args.c1_root / "dual_outcome_replay_panel.parquet")
    c1_target_free = c1_target_free.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    c1_outcome = c1_outcome.sort_values("candidate_id", kind="stable").reset_index(drop=True)

    frozen_bcf = _read_unique(args.frozen_bcf)
    frozen_current = _read_unique(args.frozen_current)
    wanted = c1_target_free.loc[:, ["candidate_id"]]
    frozen_bcf = wanted.merge(frozen_bcf, on="candidate_id", how="left", validate="one_to_one")
    frozen_current = wanted.merge(frozen_current, on="candidate_id", how="left", validate="one_to_one")
    if frozen_bcf.isna().any().any() or frozen_current.isna().any().any():
        raise AssertionError("frozen control does not cover every C1 candidate identity")
    for field in ("__decision_ts__", "__symbol__"):
        if not frozen_bcf[field].astype(str).equals(c1_target_free[field].astype(str)):
            raise AssertionError(f"BCF frozen identity mismatch: {field}")
    _assert_same_numeric(frozen_bcf["final_score"], c1_target_free["final_score"], "bcf_final_score")

    baseline_target_free = pd.DataFrame({
        "candidate_id": c1_target_free["candidate_id"].astype(str),
        "__decision_ts__": c1_target_free["__decision_ts__"],
        "__symbol__": c1_target_free["__symbol__"].astype(str),
        "final_score": pd.to_numeric(frozen_bcf["final_score"], errors="raise"),
        "bcf_mc1_expected_bps": pd.to_numeric(frozen_bcf["mc1_expected_bps"], errors="raise"),
        "current_mc1_expected_bps": pd.to_numeric(frozen_current["mc1_expected_bps"], errors="raise"),
    })
    baseline_target_free["dual_mc1_admitted"] = (
        baseline_target_free["bcf_mc1_expected_bps"].ge(ADMISSION_BPS)
        & baseline_target_free["current_mc1_expected_bps"].ge(ADMISSION_BPS)
    )
    baseline_target_free["auction_priority_bps"] = baseline_target_free["bcf_mc1_expected_bps"]
    baseline = _attach_outcomes(baseline_target_free, c1_outcome)
    c1 = _attach_outcomes(c1_target_free, c1_outcome)
    baseline.to_parquet(args.output / "baseline_target_free_plus_outcomes.parquet", index=False, compression="zstd")
    c1.to_parquet(args.output / "c1_target_free_plus_outcomes.parquet", index=False, compression="zstd")

    metrics: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    for arm, panel in (("frozen_dual_mc1_control", baseline), (str(args.challenger_name), c1)):
        row, by_month = _run_arm(panel, arm=arm, out=args.output)
        metrics.append(row)
        monthly.append(by_month)
    summary = pd.DataFrame(metrics)
    control = summary.loc[summary["arm"].eq("frozen_dual_mc1_control")].iloc[0]
    for field in (
        "dual_admitted_rows", "accepted_rows", "net_ev_bps_per_realised_trade",
        "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown", "final_wallet",
    ):
        summary[f"delta_vs_control_{field}"] = summary[field] - control[field]
    summary.to_parquet(args.output / "portfolio_summary.parquet", index=False, compression="zstd")
    pd.concat(monthly, ignore_index=True).to_parquet(args.output / "monthly_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "p8u_c1_full_coverage_matched_oos_portfolio_v1",
        "scope": "research-only; no exchange, live-entry, or activation authority",
        "period": "2026-05-01 through 2026-07-31",
        "policy": "one attached canonical 15-minute parent-policy outcome panel, valid outcomes only",
        "admission": "both BCF/current mapped expected EV >= +50 bps",
        "auction": "global chronological constrained portfolio; BCF mapped expected EV priority",
        "comparison": "frozen incumbent maps versus full-coverage strictly-prequential C1 augmented maps on identical candidate IDs",
        "challenger_name": str(args.challenger_name),
        "input_hashes": {
            "c1_target_free": _sha256(args.c1_root / "dual_target_free_predictions.parquet"),
            "c1_outcomes": _sha256(args.c1_root / "dual_outcome_replay_panel.parquet"),
            "frozen_bcf": _sha256(args.frozen_bcf),
            "frozen_current": _sha256(args.frozen_current),
        },
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
