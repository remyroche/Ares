#!/usr/bin/env python3
"""One-factor portfolio-constraint ablations for mapped execution-EV OOF scores.

The model score is selected exactly once: a single pooled global top-k book after
the supplied causal recent-EV mapping.  Every replay arm consumes that immutable
book; the arms only change the executable portfolio constraint under test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    load_portfolio_policy_params,
    replay_candidates,
)
from scripts.replay_execution_ev_global_topk_portfolio import (  # noqa: E402
    ID_COLUMNS,
    _sha256,
    build_candidates,
)


IDENTITY = [*ID_COLUMNS]
DEFAULT_FOLD_COL = "execution_ev_model_ablation_oof_fold"


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def _safe_arm_value(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _identity_ev_curve() -> dict[str, object]:
    """Avoid fitting a post-selection outcome curve during the OOF replay."""
    return {"schema": "monotone_ev_curve_v1", "x": [0.0, 1.0], "y": [0.0, 1.0], "ev_span": 1.0, "n_rows": 0}


def build_constraint_arms(
    baseline: PortfolioPolicyParams,
    *,
    concurrency_caps: Sequence[int] = (),
    wallet_caps: Sequence[float] = (),
    per_symbol_caps: Sequence[int] = (),
    per_side_caps: Sequence[int] = (),
    new_entry_caps: Sequence[int] = (),
) -> list[tuple[str, str, Any, PortfolioPolicyParams]]:
    """Return baseline plus one-factor constraint perturbations.

    A count-cap arm explicitly enables count-cap enforcement.  This is necessary
    because the fixed live policy may deliberately leave the count cap disabled;
    the resulting arm is clearly named and is never presented as a pure numeric
    change to an inactive field.
    """
    arms: list[tuple[str, str, Any, PortfolioPolicyParams]] = [
        ("baseline", "baseline", None, baseline)
    ]
    for cap in concurrency_caps:
        if cap < 1:
            raise ValueError("concurrency caps must be >= 1")
        arms.append((
            f"concurrency_total_{cap}",
            "max_concurrent_positions",
            cap,
            replace(baseline, enforce_position_count_cap=True, max_concurrent_positions=int(cap)),
        ))
    for cap in wallet_caps:
        if not 0.0 < cap <= 1.0:
            raise ValueError("wallet allocation caps must be in (0, 1]")
        arms.append((
            f"wallet_allocation_{_safe_arm_value(cap)}",
            "max_total_wallet_allocation_pct",
            cap,
            replace(baseline, max_total_wallet_allocation_pct=float(cap)),
        ))
    for cap in per_symbol_caps:
        if cap < 1:
            raise ValueError("per-symbol caps must be >= 1")
        arms.append((
            f"per_symbol_{cap}",
            "max_concurrent_per_symbol",
            cap,
            replace(baseline, max_concurrent_per_symbol=int(cap)),
        ))
    for cap in per_side_caps:
        if cap < 1:
            raise ValueError("per-side caps must be >= 1")
        arms.append((
            f"per_side_{cap}",
            "max_concurrent_per_side",
            cap,
            replace(baseline, enforce_position_count_cap=True, max_concurrent_per_side=int(cap)),
        ))
    for cap in new_entry_caps:
        if cap < 1:
            raise ValueError("new-entry caps must be >= 1")
        arms.append((
            f"new_entries_per_bar_{cap}",
            "max_new_entries_per_bar",
            cap,
            replace(baseline, max_new_entries_per_bar=int(cap)),
        ))
    return arms


def attach_oof_fold(
    candidates: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    score_col: str,
    fold_col: str,
    eligibility_flag_col: str | None = None,
) -> pd.DataFrame:
    """Attach diagnostic OOF-fold identity without changing the selected book."""
    if fold_col not in oof.columns:
        raise ValueError(f"OOF table has no fold column {fold_col!r}")
    flag_col = eligibility_flag_col or f"{score_col}__is_oof"
    required = {*IDENTITY, flag_col, fold_col}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF table missing columns: {missing}")
    lookup = oof.loc[oof[flag_col].astype(bool), [*IDENTITY, fold_col]].copy()
    if lookup.duplicated(IDENTITY).any():
        raise ValueError("OOF fold lookup identity is not one-to-one")
    if lookup["candidate_id"].duplicated().any() or candidates["candidate_id"].duplicated().any():
        raise ValueError("Execution-EV candidate_id must be globally unique for fold attribution")
    out = candidates.merge(
        lookup.rename(columns={"__ts__": "source_score_timestamp", fold_col: "oof_fold"}),
        left_on=["candidate_id"],
        right_on=["candidate_id"],
        how="left",
    )
    # Candidate ids are globally unique in the execution-EV handoff.  Preserve
    # the other identity columns in the lookup only for an exact contract check.
    for left, right in (("symbol", "__symbol__"), ("side", "side_name")):
        if right in out.columns and not out[left].astype(str).eq(out[right].astype(str)).all():
            raise ValueError(f"Candidate identity mismatch while attaching folds: {left}")
    if out["oof_fold"].isna().any():
        raise ValueError("Selected global-top-k candidates lack OOF fold coverage")
    return out.drop(columns=[column for column in ["__symbol__", "side_name"] if column in out.columns])


def _accepted_with_periods(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted.assign(month=pd.Series(dtype=str), oof_fold=pd.Series(dtype=float))
    keys = ["timestamp", "symbol", "side", "strategy_id"]
    metadata = candidates.loc[:, [*keys, "candidate_id", "oof_fold"]]
    if metadata.duplicated(keys).any():
        raise ValueError("Selected candidate decision key is not unique")
    accepted = accepted.merge(metadata, on=keys, how="left", validate="one_to_one")
    if accepted["oof_fold"].isna().any():
        raise ValueError("Accepted replay rows lack OOF-fold attribution")
    accepted["month"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.strftime("%Y-%m")
    accepted["oof_fold"] = pd.to_numeric(accepted["oof_fold"], errors="raise").astype(int)
    return accepted


def _period_metrics(accepted: pd.DataFrame, group_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, group in accepted.groupby(group_col, sort=True):
        net = pd.to_numeric(group["position_net_return"], errors="coerce")
        size = pd.to_numeric(group["position_size"], errors="coerce")
        weighted = float((net * size).sum() / max(float(size.sum()), 1e-12))
        rows.append({
            group_col: value,
            "accepted_trades": int(len(group)),
            "mean_net_return": float(net.mean()),
            "net_return_sum": float(net.sum()),
            "positive_rate": float((net > 0.0).mean()),
            "notional_weighted_net_return": weighted,
            "net_pnl": float((net * size).sum()),
        })
    return rows


def evaluate_arm(
    candidates: pd.DataFrame,
    params: PortfolioPolicyParams,
    *,
    initial_wallet: float,
    latest_fold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Replay one policy arm and return full, month, and fold evidence."""
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=_identity_ev_curve(),
        initial_wallet=float(initial_wallet),
        market_mode="perps",
    )
    accepted = _accepted_with_periods(decisions, candidates)
    monthly = pd.DataFrame(_period_metrics(accepted, "month"))
    folds = pd.DataFrame(_period_metrics(accepted, "oof_fold"))
    if not folds.empty:
        folds["is_latest_fold"] = folds["oof_fold"].eq(
            int(folds["oof_fold"].max() if latest_fold is None else latest_fold)
        )
    else:
        folds["is_latest_fold"] = pd.Series(dtype=bool)
    return decisions, equity, metrics, monthly, folds


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    oof = pd.read_parquet(args.oof)
    handoff = pd.read_parquet(args.handoff)
    candidates = build_candidates(
        oof,
        handoff,
        score_col=args.score_col,
        top_k_fraction=args.top_k_fraction,
        eligibility_flag_col=args.eligibility_flag_col,
    )
    candidates = attach_oof_fold(
        candidates,
        oof,
        score_col=args.score_col,
        fold_col=args.fold_col,
        eligibility_flag_col=args.eligibility_flag_col,
    )
    baseline = load_portfolio_policy_params(args.portfolio_config)
    arms = build_constraint_arms(
        baseline,
        concurrency_caps=_csv_ints(args.concurrency_caps),
        wallet_caps=_csv_floats(args.wallet_caps),
        per_symbol_caps=_csv_ints(args.per_symbol_caps),
        per_side_caps=_csv_ints(args.per_side_caps),
        new_entry_caps=_csv_ints(args.new_entry_caps),
    )
    candidates.to_parquet(args.output_dir / "global_topk_candidates.parquet", index=False)
    summary: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for arm, dimension, value, params in arms:
        decisions, equity, metrics, monthly, folds = evaluate_arm(
            candidates,
            params,
            initial_wallet=args.initial_wallet,
            latest_fold=args.latest_fold,
        )
        decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
        equity.to_parquet(args.output_dir / f"{arm}_equity.parquet", index=False)
        rejected = decisions.loc[~decisions["accepted"].astype(bool), "rejection_reason"]
        row = {
            "arm": arm,
            "dimension": dimension,
            "value": value,
            "selected_global_top_k_rows": int(len(candidates)),
            "accepted_trades": int(metrics.get("trade_count", 0)),
            "acceptance_rate_of_fixed_global_book": float(metrics.get("trade_count", 0) / max(len(candidates), 1)),
            "rejection_count": int(len(rejected)),
            "rejected_by_constraint_under_test": int(
                rejected.eq(
                    {
                        "max_concurrent_positions": "max_concurrent_positions_reached",
                        "max_total_wallet_allocation_pct": "max_capital_allocation_reached",
                        "max_concurrent_per_symbol": "symbol_already_open",
                        "max_concurrent_per_side": "max_concurrent_per_side_reached",
                        "max_new_entries_per_bar": "max_new_entries_per_bar_reached",
                    }.get(dimension, "")
                ).sum()
            ),
            **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))},
            "params_json": json.dumps(asdict(params), sort_keys=True, default=str),
        }
        summary.append(row)
        for record in monthly.to_dict("records"):
            month_rows.append({"arm": arm, "dimension": dimension, "value": value, **record})
        for record in folds.to_dict("records"):
            fold_rows.append({"arm": arm, "dimension": dimension, "value": value, **record})
    pd.DataFrame(summary).to_csv(args.output_dir / "constraint_ablation_summary.csv", index=False)
    pd.DataFrame(month_rows).to_csv(args.output_dir / "constraint_ablation_monthly.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "constraint_ablation_folds.csv", index=False)
    manifest = {
        "schema": "execution_ev_global_topk_portfolio_constraint_ablation_v1",
        "ranking_contract": {
            "scope": (
                "one_pooled_global_evaluation_book"
                if args.eligibility_flag_col
                else "one_pooled_global_oof_book"
            ),
            "stage": "after_supplied_causal_recent_ev_mapping",
            "top_k_fraction": float(args.top_k_fraction),
            "per_timestamp_quota": False,
            "selection_recomputed_per_arm": False,
        },
        "sources": {
            "oof": {"path": str(args.oof), "sha256": _sha256(args.oof)},
            "handoff": {"path": str(args.handoff), "sha256": _sha256(args.handoff)},
            "portfolio_config": {"path": str(args.portfolio_config), "sha256": _sha256(args.portfolio_config)},
            "score_col": args.score_col,
            "eligibility_flag_col": (
                args.eligibility_flag_col
                or f"{args.score_col}__is_oof"
            ),
            "fold_col": args.fold_col,
        },
        "coverage": {
            "selected_global_top_k_rows": int(len(candidates)),
            "folds": sorted(pd.to_numeric(candidates["oof_fold"], errors="raise").astype(int).unique().tolist()),
            "latest_fold": int(args.latest_fold) if args.latest_fold is not None else int(pd.to_numeric(candidates["oof_fold"], errors="raise").max()),
            "latest_fold_selected_rows": int(
                candidates["oof_fold"].eq(
                    int(args.latest_fold)
                    if args.latest_fold is not None
                    else int(
                        pd.to_numeric(
                            candidates["oof_fold"], errors="raise"
                        ).max()
                    )
                ).sum()
            ),
        },
        "arms": [{"arm": arm, "dimension": dimension, "value": value, "params": asdict(params)} for arm, dimension, value, params in arms],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument(
        "--eligibility-flag-col",
        default=None,
        help=(
            "Explicit evaluation eligibility flag. Defaults to "
            "<score-col>__is_oof; use a separately named flag for mixed "
            "OOF plus forward-OOS diagnostics."
        ),
    )
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-col", default=DEFAULT_FOLD_COL)
    parser.add_argument("--latest-fold", type=int, default=None)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    parser.add_argument("--concurrency-caps", default="8,16,32")
    parser.add_argument("--wallet-caps", default="0.40,0.55,0.70")
    parser.add_argument("--per-symbol-caps", default="1,2")
    parser.add_argument("--per-side-caps", default="")
    parser.add_argument("--new-entry-caps", default="1,2,3")
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    print(json.dumps(payload["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
