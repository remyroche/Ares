#!/usr/bin/env python3
"""Matched top-50%-route base-layer diagnostics, research only.

Every score is ranked *after* the identical strict-OOF router has retained
the timestamp-local top 50%.  This distinguishes a full-universe base merely
evaluated on the route from a base genuinely trained on routed rows.

The report has two complementary views:
* score quality: timestamp-average top-k policy net and top-10% >50-bps
  precision; and
* a no-admission base-only top-two portfolio mirror using the same rich policy
  outcomes, exit bars and global auction constraints as the downstream stack.

It is deliberately not an MC1/admission result and never mutates live state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_router_routed_base_stack as routed  # noqa: E402


EVAL_START = pd.Timestamp("2026-04-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-08-01T00:00:00Z")
FRACTIONS = (.01, .02, .05, .10)
ROUTE = .50
POLICY_COLUMNS = (
    "candidate_id", "__decision_ts__", "__symbol__", "policy_path_valid",
    "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason",
)


def _month_tokens(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, ...]:
    return tuple(f"{value:%Y-%m}" for value in pd.date_range(start, end - pd.Timedelta(nanoseconds=1), freq="MS", tz="UTC"))


def _read_router(router_root: Path, months: Iterable[str]) -> pd.DataFrame:
    parts = []
    for token in months:
        path = router_root / "target_free_scores" / f"month={token}.parquet"
        part = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "side_name", "router_primary_rank"])
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    if out["candidate_id"].duplicated().any():
        raise AssertionError("router has duplicate candidate IDs")
    return out


def _read_policy(path: Path) -> pd.DataFrame:
    out = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    if out["candidate_id"].duplicated().any():
        raise AssertionError("policy has duplicate candidate IDs")
    return out


def _read_source(root: Path, months: Iterable[str], score: str) -> pd.DataFrame:
    parts = []
    for token in months:
        path = root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        names = set(pq.ParquetFile(path).schema_arrow.names)
        needed = ["candidate_id", "__decision_ts__", "side_name", score]
        if "enhanced_base_routed" in names:
            needed.append("enhanced_base_routed")
        parts.append(pd.read_parquet(path, columns=needed))
    out = pd.concat(parts, ignore_index=True)
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    if out["candidate_id"].duplicated().any():
        raise AssertionError(f"{score}: duplicate score identities")
    return out


def _route(frame: pd.DataFrame, router: pd.DataFrame, *, source_already_routed: bool) -> pd.DataFrame:
    keys = ["candidate_id", "__decision_ts__", "side_name"]
    router = router.copy()
    router["__expected_route__"] = parent._exact_timestamp_top_fraction(router, "router_primary_rank", ROUTE).to_numpy(bool)
    joined = frame.merge(router, on=keys, how="left", validate="one_to_one")
    if joined["router_primary_rank"].isna().any():
        raise AssertionError("score/router identity mismatch")
    if source_already_routed:
        expected_ids = set(router.loc[router["__expected_route__"], "candidate_id"].astype(str))
        actual_ids = set(joined["candidate_id"].astype(str))
        actual_flag = joined["enhanced_base_routed"].fillna(False).astype(bool)
        if not actual_flag.all() or actual_ids != expected_ids:
            raise AssertionError("routed base source does not match the exact top-50% router membership")
        return joined.copy()
    return joined.loc[joined["__expected_route__"].fillna(False).astype(bool)].copy()


def _score_metrics(frame: pd.DataFrame, score: str) -> dict[str, object]:
    rows: dict[str, object] = {"timestamps": int(frame["__decision_ts__"].nunique()), "routed_valid_rows": int(len(frame))}
    top2: list[pd.DataFrame] = []
    for fraction in FRACTIONS:
        samples: list[float] = []
        precision: list[float] = []
        selected = 0
        for _, group in frame.groupby("__decision_ts__", sort=False):
            count = max(1, int(np.ceil(len(group) * fraction)))
            chosen = group.nlargest(count, score, keep="first")
            samples.append(float(chosen["policy_net_bps"].mean()))
            precision.append(float(chosen["policy_net_bps"].gt(50.0).mean()))
            selected += len(chosen)
        key = f"top{int(fraction * 100):02d}"
        rows[f"{key}_timestamp_net_bps"] = float(np.mean(samples))
        rows[f"{key}_precision_gt50"] = float(np.mean(precision))
        rows[f"{key}_rows"] = int(selected)
    for _, group in frame.groupby("__decision_ts__", sort=False):
        top2.append(group.nlargest(min(2, len(group)), score, keep="first"))
    top = pd.concat(top2, ignore_index=True)
    rows["top2_timestamp_net_bps"] = float(top.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean().mean())
    rows["top2_rows"] = int(len(top))
    rows["top2_total_net_bps"] = float(top["policy_net_bps"].sum())
    return rows, top


def _portfolio_metrics(top: pd.DataFrame, score: str, label: str, out: Path) -> dict[str, object]:
    from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table, replay_candidates
    from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params

    work = top.copy()
    work["auction_rank"] = work.groupby("__decision_ts__", sort=False)[score].rank(pct=True, method="average")
    exit_bar = pd.to_numeric(work["policy_exit_bar_15m"], errors="coerce").astype(int)
    decision = pd.to_datetime(work["__decision_ts__"], utc=True)
    candidate = pd.DataFrame({
        "timestamp": decision, "symbol": work["__symbol__"].astype(str), "side": "long",
        "strategy_id": label, "policy_archetype": "strict_r3_rich_policy",
        "normalized_rank_score": work["auction_rank"].to_numpy(float),
        "strategy_rank_pct": work["auction_rank"].to_numpy(float), "base_strategy_threshold": 0.0,
        "calibrated_score": pd.to_numeric(work[score], errors="coerce").to_numpy(float),
        "entry_price": pd.to_numeric(work["policy_entry_price"], errors="coerce"),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(work["policy_exit_price"], errors="coerce"),
        "net_return": pd.to_numeric(work["policy_net_bps"], errors="coerce") / 10_000.0,
        "gross_return": pd.to_numeric(work["policy_gross_bps"], errors="coerce") / 10_000.0,
        "holding_bars": exit_bar + 1, "simple_policy_exit_reason": work["policy_exit_reason"].astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"), "candidate_id": work["candidate_id"].astype(str),
        "mapped_expected_net_bps": pd.to_numeric(work[score], errors="coerce"), "policy_outcome_available": True,
    })
    candidates = normalise_candidate_table(candidate)
    decisions, equity, _ = replay_candidates(candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0)
    # This diagnostic forms candidates only after ``policy_path_valid`` has
    # been required.  The generic portfolio normaliser omits this provenance
    # flag (and candidate IDs), so restore the explicit all-resolved contract
    # for the shared portfolio metric helper.
    decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{label}_top2_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{label}_top2_equity.parquet", index=False, compression="zstd")
    metrics = _metrics(decisions, equity, label, "2026_aprjul_top2")
    # A common rich-policy, top-two mirror makes a daily Sortino comparable
    # across base variants.  We use end-of-day wallet marks and no risk-free
    # return; it is a selection diagnostic, not a claim about live sizing.
    if {"timestamp", "wallet"}.issubset(equity.columns):
        marks = equity.loc[:, ["timestamp", "wallet"]].copy()
        marks["timestamp"] = pd.to_datetime(marks["timestamp"], utc=True, errors="coerce")
        marks["wallet"] = pd.to_numeric(marks["wallet"], errors="coerce")
        marks = marks.dropna().sort_values("timestamp", kind="stable")
        marks["day"] = marks["timestamp"].dt.floor("D")
        daily = marks.groupby("day", sort=True)["wallet"].last().pct_change().dropna()
        downside = float(np.sqrt(np.mean(np.minimum(daily.to_numpy(float), 0.0) ** 2))) if len(daily) else np.nan
        metrics["sortino_daily_annualized"] = (
            float(np.sqrt(365.0) * daily.mean() / downside) if np.isfinite(downside) and downside > 0 else np.nan
        )
        metrics["portfolio_days"] = int(len(daily))
    else:
        metrics["sortino_daily_annualized"] = np.nan
        metrics["portfolio_days"] = 0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--full-source-root", type=Path, required=True)
    parser.add_argument("--routed-no-router-root", type=Path, required=True)
    parser.add_argument("--routed-router-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--routed-only-selection", action="store_true",
        help="Keep full-trained arms diagnostic-only; select only routed-population base fits.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    months = _month_tokens(EVAL_START, EVAL_END)
    router = _read_router(args.router_root, months)
    policy = _read_policy(args.policy_path)
    policy = policy.loc[policy["__decision_ts__"].ge(EVAL_START) & policy["__decision_ts__"].lt(EVAL_END)].copy()
    variants = (
        ("B0_full_trained_evaluate_routed", args.full_source_root, "base_bps", False),
        ("Enhanced_threeway_full_trained_evaluate_routed", args.full_source_root, "enhanced_base_bps", False),
        ("Enhanced_threeway_routed_trained", args.routed_no_router_root, "enhanced_base_bps", True),
        ("Enhanced_threeway_routed_trained_router_inputs", args.routed_router_root, "enhanced_base_bps", True),
    )
    rows = []
    for label, root, score, routed_source in variants:
        scores = _read_source(root, months, score)
        selected = _route(scores, router, source_already_routed=routed_source)
        joined = selected.merge(policy, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
        valid = joined.loc[joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))].copy()
        metrics, top2 = _score_metrics(valid, score)
        portfolio = _portfolio_metrics(top2, score, label, args.out)
        rows.append({"variant": label, "score": score, **metrics, **{f"portfolio_{k}": v for k, v in portfolio.items()}})
    table = pd.DataFrame(rows)
    table.to_parquet(args.out / "base_variant_metrics.parquet", index=False, compression="zstd")
    # Predeclared selection: a candidate must retain top-10 >50-bps precision
    # within three percentage points of the B0 control.  Among survivors,
    # choose timestamp-local top-two EV, then portfolio EV, daily Sortino and
    # smaller drawdown.  This avoids selecting a precision-only model that
    # discards materially more economic opportunity.
    # Under the explicit routed-only architecture, the relevant control is
    # the same routed base *without* numerical router inputs.  B0/full-base
    # rows remain reported as diagnostics but cannot veto both permitted
    # routed fits merely because they were trained on a broader population.
    control_label = (
        "Enhanced_threeway_routed_trained"
        if args.routed_only_selection else
        "B0_full_trained_evaluate_routed"
    )
    control_precision = float(table.loc[
        table["variant"].eq(control_label), "top10_precision_gt50"
    ].iloc[0])
    eligible = table.loc[table["top10_precision_gt50"].ge(control_precision - .03)].copy()
    if args.routed_only_selection:
        eligible = eligible.loc[
            eligible["variant"].str.startswith("Enhanced_threeway_routed_trained")
        ].copy()
    if eligible.empty:
        raise AssertionError("no base variant retained the predeclared precision floor")
    winner = eligible.sort_values(
        ["top2_timestamp_net_bps", "portfolio_net_ev_bps_per_realised_trade", "portfolio_sortino_daily_annualized", "portfolio_max_drawdown"],
        ascending=[False, False, False, False], kind="stable",
    ).iloc[0]
    selection = {
        "selection_rule": (
            "routed-base-only; precision >= B0 minus 3pp; then top2 timestamp EV, "
            "portfolio EV, daily Sortino, lower drawdown"
            if args.routed_only_selection else
            "precision >= B0 minus 3pp; then top2 timestamp EV, portfolio EV, daily Sortino, lower drawdown"
        ),
        "control_top10_precision_gt50": control_precision,
        "selection_control": control_label,
        "winner": str(winner["variant"]),
        "winner_metrics": winner.to_dict(),
        "eligible_variants": eligible["variant"].tolist(),
        "period": "2026-04 through 2026-07; strict-OOF router top-50% at every timestamp",
        "policy": "canonical reconciled rich policy: trailing, smooth capital protection, 100-bps cost",
    }
    args.out.joinpath("base_selection.json").write_text(json.dumps(selection, indent=2, default=str) + "\n")
    args.out.joinpath("run_manifest.json").write_text(json.dumps({
        "scope": "research only; common strict-OOF top-50% timestamp-local router route",
        "period": [str(EVAL_START), str(EVAL_END)], "policy": str(args.policy_path),
        "router": str(args.router_root), "variants": [row["variant"] for row in rows],
        "routed_only_selection": bool(args.routed_only_selection),
        "selection": selection,
        "policy_contract": "canonical reconciled rich policy including trailing, smooth capital protection and 100-bps cost",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
