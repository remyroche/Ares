#!/usr/bin/env python3
"""Matched causal admission comparison on one frozen strict-R3 score ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from scripts.replay_strict_r3_forward_portfolio import _auction_candidates  # noqa: E402
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402


MODES = (
    "pooled_parent_side_shrinkage_v1",
    "hierarchical_tail_side_shrinkage_v2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _admission_summary(
    frame: pd.DataFrame,
    *,
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame]:
    evaluation = frame.loc[
        frame["__decision_ts__"].ge(start)
        & frame["__decision_ts__"].lt(end)
    ].copy()
    finite_score = np.isfinite(pd.to_numeric(evaluation["final_score"], errors="coerce"))
    population = evaluation.loc[finite_score].copy()
    top2_count = max(1, int(math.ceil(0.02 * len(population))))
    top2_ids = set(
        population.nlargest(top2_count, "final_score", keep="first")[
            "candidate_id"
        ].astype(str)
    )
    admitted = evaluation.loc[
        evaluation["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
    ].copy()
    outcome_valid = (
        admitted["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(admitted["policy_net_bps"], errors="coerce"))
    )
    valid = admitted.loc[outcome_valid].copy()
    summary = {
        "mode": mode,
        "population_rows": int(len(population)),
        "admitted_rows": int(len(admitted)),
        "admission_rate": float(len(admitted) / max(len(population), 1)),
        "admitted_outcome_valid_rows": int(len(valid)),
        "admitted_outcome_coverage": float(len(valid) / max(len(admitted), 1)),
        "admitted_missing_outcomes": int(len(admitted) - len(valid)),
        "top2_selected_rows": int(top2_count),
        "top2_admitted_rows": int(
            admitted["candidate_id"].astype(str).isin(top2_ids).sum()
        ),
        "top2_admission_recall": float(
            admitted["candidate_id"].astype(str).isin(top2_ids).sum()
            / max(top2_count, 1)
        ),
        "gross_bps_per_valid_admitted_trade": float(
            pd.to_numeric(valid["policy_gross_bps"], errors="coerce").mean()
        ),
        "net_bps_per_valid_admitted_trade": float(
            pd.to_numeric(valid["policy_net_bps"], errors="coerce").mean()
        ),
        "positive_rate_valid_admitted": float(
            pd.to_numeric(valid["policy_net_bps"], errors="coerce").gt(0).mean()
        ),
    }
    monthly = valid.assign(
        month=valid["__decision_ts__"].dt.strftime("%Y-%m"),
    ).groupby("month", as_index=False).agg(
        trades=("candidate_id", "size"),
        gross_bps_per_trade=("policy_gross_bps", "mean"),
        net_bps_per_trade=("policy_net_bps", "mean"),
        positive_rate=("policy_net_bps", lambda value: float((value > 0.0).mean())),
    )
    monthly.insert(0, "mode", mode)
    return summary, monthly


def _portfolio_summary(
    admitted: pd.DataFrame,
    *,
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    evaluation = admitted.loc[
        admitted["__decision_ts__"].ge(start)
        & admitted["__decision_ts__"].lt(end)
    ].copy()
    try:
        candidates = _auction_candidates(evaluation)
    except ValueError:
        return {
            "mode": mode, "accepted_trades": 0,
            "accepted_outcome_unavailable": 0,
            "trades_per_calendar_day": 0.0,
            "net_bps_per_trade": np.nan,
            "gross_bps_per_trade": np.nan,
            "positive_rate": np.nan,
            "max_drawdown": np.nan,
        }, pd.DataFrame(), pd.DataFrame()
    decisions, _, monthly, raw = _run(
        candidates, 0.0, mode,
        initial_wallet=1_000.0,
        perp_leverage=7.0,
        margin_slot_wallet_fraction=0.10,
    )
    lineage_columns = [
        "timestamp", "symbol", "side", "strategy_id",
        "policy_outcome_available", "policy_outcome_source",
    ]
    missing_lineage = [
        column for column in lineage_columns[4:] if column not in decisions
    ]
    if missing_lineage:
        lineage = candidates[lineage_columns].drop_duplicates(
            lineage_columns[:4],
        )
        decisions = decisions.merge(
            lineage,
            on=lineage_columns[:4],
            how="left",
            validate="many_to_one",
        )
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)]
    unavailable = (
        ~accepted["policy_outcome_available"].fillna(False).astype(bool)
        if "policy_outcome_available" in accepted else pd.Series(False, index=accepted.index)
    )
    replay = raw.get("replay_metric_summary", {})
    if isinstance(replay, str):
        replay = json.loads(replay)
    days = max((end - start).total_seconds() / 86_400.0, 1.0)
    summary = {
        "mode": mode,
        "accepted_trades": int(raw["accepted_trades"]),
        "accepted_outcome_unavailable": int(unavailable.sum()),
        "trades_per_calendar_day": float(raw["accepted_trades"] / days),
        "net_bps_per_trade": float(raw["net_bps_per_trade"]),
        "gross_bps_per_trade": float(raw["gross_bps_per_trade"]),
        "positive_rate": float(raw["positive_rate"]),
        "max_drawdown": float(replay.get("max_drawdown", np.nan)),
    }
    if "policy_outcome_source" not in accepted:
        accepted["policy_outcome_source"] = "unspecified"
    accepted_valid = accepted.loc[~unavailable].copy()
    source = accepted_valid.groupby(
        "policy_outcome_source", dropna=False, as_index=False,
    ).agg(
        trades=("position_net_return", "size"),
        net_bps_per_trade=(
            "position_net_return",
            lambda value: float(pd.to_numeric(value, errors="coerce").mean() * 10_000.0),
        ),
        gross_bps_per_trade=(
            "position_gross_return",
            lambda value: float(pd.to_numeric(value, errors="coerce").mean() * 10_000.0),
        ),
        positive_rate=(
            "position_net_return",
            lambda value: float((pd.to_numeric(value, errors="coerce") > 0.0).mean()),
        ),
    )
    source.insert(0, "mode", mode)
    return summary, monthly.assign(mode=mode), source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start = pd.Timestamp(args.evaluation_start, tz="UTC")
    end = pd.Timestamp(args.evaluation_end, tz="UTC")
    frame = pd.read_parquet(args.predictions)
    for column in ("__decision_ts__", "policy_label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("prediction ledger is empty or has duplicate identities")
    if frame["__decision_ts__"].min() >= start:
        raise ValueError(
            "prediction ledger lacks the required pre-evaluation admission warm-up",
        )

    summaries: list[dict[str, object]] = []
    portfolios: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []
    portfolio_monthly: list[pd.DataFrame] = []
    portfolio_sources: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for mode in args.modes:
        admitted, audit = apply_causal_21d_side_admission(
            frame,
            score_column="final_score",
            net_column="policy_net_bps",
            decision_column="__decision_ts__",
            label_available_column="policy_label_available_ts",
            spec=Causal21dAdmissionSpec(mode=mode),
        )
        summary, monthly = _admission_summary(
            admitted, mode=mode, start=start, end=end,
        )
        portfolio, portfolio_month, portfolio_source = _portfolio_summary(
            admitted, mode=mode, start=start, end=end,
        )
        summaries.append(summary)
        portfolios.append(portfolio)
        monthly_rows.append(monthly)
        portfolio_monthly.append(portfolio_month)
        portfolio_sources.append(portfolio_source)
        audits.append(audit.assign(mode=mode))

    args.out_dir.mkdir(parents=True)
    pd.DataFrame(summaries).to_parquet(
        args.out_dir / "admission_comparison.parquet", index=False,
    )
    pd.DataFrame(portfolios).to_parquet(
        args.out_dir / "portfolio_comparison.parquet", index=False,
    )
    pd.concat(monthly_rows, ignore_index=True).to_parquet(
        args.out_dir / "admission_monthly.parquet", index=False,
    )
    nonempty = [value for value in portfolio_monthly if not value.empty]
    if nonempty:
        pd.concat(nonempty, ignore_index=True).to_parquet(
            args.out_dir / "portfolio_monthly.parquet", index=False,
        )
    nonempty_sources = [value for value in portfolio_sources if not value.empty]
    if nonempty_sources:
        pd.concat(nonempty_sources, ignore_index=True).to_parquet(
            args.out_dir / "portfolio_by_outcome_source.parquet", index=False,
        )
    pd.concat(audits, ignore_index=True).to_parquet(
        args.out_dir / "mapping_audit.parquet", index=False,
    )
    manifest = {
        "schema": "strict_r3_admission_map_comparison_v1",
        "predictions": str(args.predictions),
        "predictions_sha256": _sha256(args.predictions),
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "modes": args.modes,
        "ranking": "global score tail; no per-timestamp top-k",
        "admission_floor_bps": 50.0,
        "causality": "prior-resolved labels only; score ledger includes prior warm-up",
        "future_path_selection": "prohibited; unavailable accepted rows reserve H12 slots",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
