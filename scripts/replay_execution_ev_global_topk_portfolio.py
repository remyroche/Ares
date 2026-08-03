#!/usr/bin/env python3
"""Replay mapped execution-EV global top-k scores through portfolio constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)

ID_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser


def build_candidates(
    oof: pd.DataFrame,
    handoff: pd.DataFrame,
    *,
    score_col: str,
    top_k_fraction: float,
    eligibility_flag_col: str | None = None,
) -> pd.DataFrame:
    if not 0.0 < float(top_k_fraction) <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    flag_col = eligibility_flag_col or f"{score_col}__is_oof"
    required_oof = {*ID_COLUMNS, score_col, flag_col}
    missing = sorted(required_oof.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF table missing columns: {missing}")
    label_columns = [
        *ID_COLUMNS,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_exit_reason",
        "execution_exit_hour",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
    ]
    missing = sorted(set(label_columns).difference(handoff.columns))
    if missing:
        raise ValueError(f"Handoff table missing columns: {missing}")
    labels = handoff.loc[:, label_columns]
    if labels.duplicated(list(ID_COLUMNS)).any():
        raise ValueError("Handoff identity is not one-to-one")
    work = oof.loc[oof[flag_col].astype(bool), [*ID_COLUMNS, score_col]].merge(
        labels,
        on=list(ID_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(work) != int(oof[flag_col].astype(bool).sum()):
        raise ValueError("Exact-policy labels do not cover all eligible OOF scores")
    score = pd.to_numeric(work[score_col], errors="coerce")
    if score.isna().any():
        raise ValueError("Mapped score contains nulls")
    # One pooled rank across every timestamp and both sides. This is the
    # production research contract; portfolio constraints only remove or defer
    # rows after this global auction and never create timestamp-local quotas.
    work["normalized_rank_score"] = score.rank(method="max", pct=True)
    top_k_rows = int(np.ceil(float(top_k_fraction) * len(work)))
    work = (
        work.assign(__mapped_score__=score)
        .sort_values(
            [
                "__mapped_score__",
                "__ts__",
                "__symbol__",
                "side_name",
                "candidate_id",
            ],
            ascending=[False, True, True, True, True],
            kind="stable",
        )
        .head(top_k_rows)
        .drop(columns="__mapped_score__")
        .reset_index(drop=True)
    )
    score = pd.to_numeric(work[score_col], errors="raise")
    threshold = 1.0 - float(top_k_fraction)
    gross = pd.to_numeric(work["execution_gross_ev_12h"], errors="raise")
    side = work["side_name"].astype(str).str.lower()
    exit_price = np.where(side.eq("short"), 1.0 - gross, 1.0 + gross)
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                work["execution_decision_utc"], utc=True, errors="raise"
            ),
            "symbol": work["__symbol__"].astype(str),
            "side": side,
            "strategy_id": "execution_ev_residual",
            "base_strategy_threshold": threshold,
            "calibrated_score": score,
            "normalized_rank_score": work["normalized_rank_score"],
            "entry_price": 1.0,
            "exit_timestamp": pd.to_datetime(
                work["execution_label_end_utc"], utc=True, errors="raise"
            ),
            "exit_price": np.maximum(exit_price, 1e-9),
            "net_return": pd.to_numeric(
                work["execution_net_ev_12h"], errors="raise"
            ),
            "gross_return": gross,
            "holding_bars": pd.to_numeric(
                work["execution_exit_hour"], errors="coerce"
            ).fillna(24.0),
            "simple_policy_exit_reason": work["execution_exit_reason"].astype(str),
            "fees_bps": 100.0,
            "price_gap_bps": 0.0,
            "expected_friction_bps": 0.0,
            "candidate_id": work["candidate_id"].astype(str),
        }
    )
    return normalise_candidate_table(candidates)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    oof = pd.read_parquet(args.oof)
    handoff = pd.read_parquet(args.handoff)
    candidates = build_candidates(
        oof,
        handoff,
        score_col=args.score_col,
        top_k_fraction=args.top_k_fraction,
    )
    params = load_portfolio_policy_params(args.portfolio_config)
    # Identity monotone curve avoids fitting any portfolio calibration on the
    # evaluation outcomes. With zero incremental price-gap penalty it preserves
    # the mapped global-score ordering within each executable decision bar.
    identity_ev_curve = {
        "schema": "monotone_ev_curve_v1",
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "ev_span": 1.0,
        "n_rows": 0,
    }
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=identity_ev_curve,
        initial_wallet=float(args.initial_wallet),
        market_mode="perps",
    )
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    accepted["month"] = pd.to_datetime(
        accepted["timestamp"], utc=True
    ).dt.strftime("%Y-%m")
    monthly = []
    for month, group in accepted.groupby("month", sort=True):
        returns = pd.to_numeric(group["position_net_return"], errors="coerce")
        monthly.append(
            {
                "month": str(month),
                "accepted_trades": int(len(group)),
                "mean_net_return": float(returns.mean()),
                "net_return_sum": float(returns.sum()),
                "positive_rate": float((returns > 0.0).mean()),
            }
        )
    candidates.to_parquet(args.output_dir / "candidates.parquet", index=False)
    decisions.to_parquet(args.output_dir / "portfolio_decisions.parquet", index=False)
    equity.to_parquet(args.output_dir / "portfolio_equity.parquet", index=False)
    payload = {
        "schema": "execution_ev_global_topk_portfolio_replay_v1",
        "ranking_contract": {
            "scope": "global_pooled_oof",
            "stage": "after_causal_21d_recent_ev_mapping",
            "top_k_fraction": float(args.top_k_fraction),
            "per_timestamp_quota": False,
        },
        "sources": {
            "oof": {"path": str(args.oof), "sha256": _sha256(args.oof)},
            "handoff": {
                "path": str(args.handoff),
                "sha256": _sha256(args.handoff),
            },
            "portfolio_config": {
                "path": str(args.portfolio_config),
                "sha256": _sha256(args.portfolio_config),
            },
            "score_col": args.score_col,
        },
        "coverage": {
            "candidate_rows": int(len(candidates)),
            "global_top_k_rows": int(len(candidates)),
            "accepted_rows": int(len(accepted)),
            "min_timestamp": candidates["timestamp"].min().isoformat(),
            "max_timestamp": candidates["timestamp"].max().isoformat(),
        },
        "monthly_accepted_metrics": monthly,
        "portfolio_metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = run(_parser().parse_args())
    print(json.dumps(payload["coverage"], indent=2))
    print(json.dumps(payload["monthly_accepted_metrics"], indent=2))


if __name__ == "__main__":
    main()
