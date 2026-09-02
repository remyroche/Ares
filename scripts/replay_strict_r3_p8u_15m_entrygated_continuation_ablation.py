#!/usr/bin/env python3
"""Portfolio-constrained C1 ablation on a frozen strict-OOS entry cohort.

The continuation challenger must be evaluated on the same entry decisions
that the entry selector made.  This utility filters already materialised,
strict-OOS rich-policy continuation outcomes to that frozen cohort and then
replays each C1 action separately through the unchanged BCF-priority auction.
It neither trains a model nor communicates with an exchange.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_strict_r3_p8u_15m_continuation_portfolio as base
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)
from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    replay_candidates,
)


DEFAULT_OUTCOMES = (
    ROOT
    / "data_perp/artifacts/strict_r3_p8u_15m_continuation_c1_catmae_activation_hpo_20260830_v1"
    / "entry_outcomes.parquet"
)
DEFAULT_ENTRY = (
    ROOT
    / "data_perp/artifacts/strict_r3_p8u_15m_entry_policycap_retrain_20260830_v3_control"
    / "walkforward_predictions.parquet"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entrygated_continuation_ablation_20260830_v1"
DEFAULT_ARMS = (
    "C0_parent",
    "C1_activation_10",
    "C1_activation_20",
    "C1_activation_only",
    "C1_activation_40",
    "C1_activation_50",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_ids(path: Path) -> set[str]:
    rows = pd.read_parquet(path)
    required = {"candidate_id", "selected__veto_pred_ge_0", "floor_bps"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"entry predictions lack {sorted(missing)}")
    # This contract is deliberately literal: it evaluates the exact frozen
    # Huber selection at the +30-bps floor, not a recreated selection.
    floor = pd.to_numeric(rows["floor_bps"], errors="raise")
    selected = rows.loc[
        rows["selected__veto_pred_ge_0"].fillna(False).astype(bool) & floor.eq(30.0),
        "candidate_id",
    ].astype(str)
    if selected.duplicated().any():
        raise AssertionError("frozen entry selection has duplicate identities")
    if selected.empty:
        raise RuntimeError("frozen entry selection is empty")
    return set(selected)


def _period_metrics(accepted: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=["month", "trades", "net_bps_per_trade", "net_sum_bps", "win_rate"])
    work = accepted.copy()
    work["month"] = pd.to_datetime(work["timestamp"], utc=True).dt.strftime("%Y-%m")
    work["net_bps"] = pd.to_numeric(work["position_net_return"], errors="raise") * 10_000.0
    return work.groupby("month", as_index=False).agg(
        trades=("candidate_id", "size"),
        net_bps_per_trade=("net_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
        win_rate=("net_bps", lambda values: float((values > 0.0).mean())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--entry-predictions", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATE_ROOT)
    parser.add_argument("--arm", action="append", default=list(DEFAULT_ARMS), choices=DEFAULT_ARMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    outcomes_path = args.outcomes.resolve()
    entry_path = args.entry_predictions.resolve()
    selected = _selected_ids(entry_path)
    outcomes, _ = base._read_inputs([str(outcomes_path.relative_to(ROOT))])
    outcomes = outcomes.loc[
        outcomes["arm"].isin(tuple(dict.fromkeys(args.arm)))
        & pd.to_numeric(outcomes["mc1_threshold_bps"], errors="raise").eq(30.0)
        & outcomes["candidate_id"].astype(str).isin(selected)
    ].copy()
    if outcomes.empty:
        raise RuntimeError("no selected C1 outcomes")
    expected_per_arm = outcomes.groupby("arm")["candidate_id"].nunique()
    if expected_per_arm.nunique() != 1:
        raise AssertionError("selected candidate coverage differs by C1 arm")
    if outcomes.duplicated(["arm", "candidate_id"]).any():
        raise AssertionError("duplicate selected candidate within C1 arm")

    prices = base._entry_prices(args.state_root.resolve())
    priorities = base._bcf_priority()
    params = canonical_portfolio_params()
    output.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, object]] = []
    for arm in tuple(dict.fromkeys(args.arm)):
        subset = outcomes.loc[outcomes["arm"].eq(arm)].copy()
        candidates = base._candidate_table(subset, prices, priorities)
        decisions, equity, _ = replay_candidates(
            candidates, params, mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp"
        )
        decisions = base._attach_ids(decisions, candidates)
        accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
        candidates.to_parquet(output / f"{arm}_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(output / f"{arm}_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(output / f"{arm}_accepted.parquet", index=False, compression="zstd")
        equity.to_parquet(output / f"{arm}_equity.parquet", index=False, compression="zstd")
        _period_metrics(accepted).assign(arm=arm).to_parquet(output / f"{arm}_monthly.parquet", index=False)
        summaries.append({
            "arm": arm,
            "frozen_entry_selected": len(selected),
            "entry_state_covered": len(candidates),
            "entry_state_unavailable": len(selected) - len(candidates),
            "portfolio_accepted": len(accepted),
            **compute_replay_metrics(candidates, decisions, equity, params=params),
        })
    summary = pd.DataFrame(summaries)
    control = summary.loc[summary["arm"].eq("C0_parent")]
    if len(control) != 1:
        raise AssertionError("exactly one C0 parent control is required")
    for field in ("portfolio_accepted", "mean_net_return_per_trade", "net_pnl", "compounded_return", "sortino", "max_drawdown", "worst_week"):
        summary[f"delta_vs_C0_{field}"] = summary[field] - control.iloc[0][field]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-entrygated-continuation-ablation-v1",
        "scope": "offline strict-OOS research; continuation outcomes were materialised before this portfolio replay",
        "entry_cohort": "exact frozen Huber 30-bps selection, selected__veto_pred_ge_0=true",
        "entry_predictions": str(entry_path),
        "entry_predictions_sha256": _sha256(entry_path),
        "outcomes": str(outcomes_path),
        "outcomes_sha256": _sha256(outcomes_path),
        "state_root": str(args.state_root.resolve()),
        "arms": list(dict.fromkeys(args.arm)),
        "candidate_identity": "exact intersection of frozen entry cohort and each strict-OOS C1 outcome arm",
        "priority": "sealed BCF MC1 expected bps only; continuation outcomes never influence entry ordering",
        "portfolio": asdict(params),
        "cost": "100 bps embedded exactly once in pre-materialised policy outcomes",
        "live_or_canonical_mutation": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
