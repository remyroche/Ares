#!/usr/bin/env python3
"""Replay a frozen policy auction using a shadow tail-health admission arm.

The script consumes already strict-OOF exact-producer scores and policy
outcomes.  It changes only the candidate-keyed, causal admission flag and
expected-net ordering produced by ``ablate_strict_r3_tail_health_gate.py``.
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

from scripts.replay_strict_r3_forward_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _auction_candidates,
    _run,
    _wallet_periods,
    _weekly,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-predictions", type=Path, required=True)
    parser.add_argument(
        "--scored-predictions-extra",
        type=Path,
        action="append",
        default=[],
        help="Additional identity-disjoint scored-ledger partitions.",
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--selection-extra",
        type=Path,
        action="append",
        default=[],
        help="Additional identity-disjoint selection partitions with the same fields.",
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--admitted-column",
        help="Explicit selection admission field; defaults to <arm>__admitted.",
    )
    parser.add_argument(
        "--expected-column",
        help="Explicit expected-net/order field; defaults to <arm>__expected_net_bps.",
    )
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-wallet", type=float, default=1000.0)
    parser.add_argument("--perp-leverage", type=float, default=7.0)
    parser.add_argument("--margin-slot-wallet-fraction", type=float, default=0.10)
    parser.add_argument(
        "--same-producer-score-tiebreak", action="store_true",
        help=(
            "Use the existing final_score only to resolve exact equal-EV ties "
            "inside the portfolio auction; it cannot change admission or sizing."
        ),
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable tail-health portfolio replay exists: {args.out_dir}")
    admitted_column = args.admitted_column or f"{args.arm}__admitted"
    expected_column = args.expected_column or f"{args.arm}__expected_net_bps"
    start = pd.to_datetime(args.evaluation_start, utc=True)
    end = pd.to_datetime(args.evaluation_end, utc=True)
    score_parts = [pd.read_parquet(args.scored_predictions)]
    score_parts.extend(pd.read_parquet(path) for path in args.scored_predictions_extra)
    score = pd.concat(score_parts, ignore_index=True, sort=False)
    selection_parts = [pd.read_parquet(args.selection)]
    selection_parts.extend(pd.read_parquet(path) for path in args.selection_extra)
    selection = pd.concat(selection_parts, ignore_index=True, sort=False)
    required = {"candidate_id", admitted_column, expected_column}
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise ValueError(f"tail-health selection lacks: {missing}")
    if score["candidate_id"].duplicated().any() or selection["candidate_id"].duplicated().any():
        raise ValueError("tail-health portfolio replay requires unique candidate identities")
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True)
    if "__decision_ts__" in selection:
        selection["__decision_ts__"] = pd.to_datetime(selection["__decision_ts__"], utc=True)
        selection = selection.loc[
            selection["__decision_ts__"].ge(start)
            & selection["__decision_ts__"].lt(end)
        ].copy()
    score = score.loc[
        score["__decision_ts__"].ge(start) & score["__decision_ts__"].lt(end)
    ].copy()
    # The control ledger may itself supply the selected admission fields.
    # Drop those copies before the keyed join so the explicit selection
    # contract remains authoritative without pandas suffixing the names.
    score = score.drop(columns=[admitted_column, expected_column], errors="ignore")
    frame = score.merge(
        selection.loc[:, ["candidate_id", admitted_column, expected_column]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    if frame[admitted_column].isna().any():
        raise ValueError("tail-health selection does not cover the scored OOF ledger")
    # A strict fail-closed map legitimately has no expected-bps estimate for
    # rejected cold-start rows.  It is an error only if such a row is admitted;
    # rejected NaNs never reach the auction and must not be converted into a
    # synthetic economic value merely to satisfy a replay convenience check.
    admitted_raw = frame[admitted_column].fillna(False).astype(bool)
    expected_raw = pd.to_numeric(frame[expected_column], errors="coerce")
    if (admitted_raw & ~np.isfinite(expected_raw)).any():
        raise ValueError("an admitted candidate lacks a finite mapped expected net")
    frame["causal_21d_side_admitted_ge_50bps"] = frame[admitted_column].astype(bool)
    frame["causal_21d_side_expected_net_bps"] = expected_raw
    if args.same_producer_score_tiebreak:
        if "final_score" not in frame:
            raise ValueError("same-producer tie-break requires final_score")
        score = pd.to_numeric(frame["final_score"], errors="coerce")
        if not np.isfinite(score).all():
            raise ValueError("same-producer tie-break requires finite final_score")
        frame["auction_tie_break_score"] = score
    # Newer lockstep ledgers persist producer lineage as its three immutable
    # hashes rather than an opaque convenience ID.  Reconstruct that ID only
    # for audit output; it is not used to rank, admit, size, or select trades.
    if "producer_bundle_id" not in frame:
        lineage = ["conversion_bundle_sha256", "upstream_bundle_sha256", "geometry_bundle_sha256"]
        missing_lineage = sorted(set(lineage).difference(frame.columns))
        if missing_lineage:
            raise ValueError(
                "scored predictions lack producer_bundle_id and cannot reconstruct it "
                f"from lineage: {missing_lineage}"
            )
        if frame.loc[:, lineage].isna().any().any():
            raise ValueError("cannot reconstruct producer provenance with null lineage")
        frame["producer_bundle_id"] = frame.loc[:, lineage].astype(str).agg("|".join, axis=1)
    evaluation = frame.copy()
    candidates = _auction_candidates(
        evaluation, strategy_prefix=f"strict_r3_tail_health_{args.arm}",
    )
    decisions, equity, monthly, summary = _run(
        candidates,
        0.0,
        f"{start.isoformat()}_to_{end.isoformat()}_tail_health_{args.arm}",
        initial_wallet=float(args.initial_wallet),
        perp_leverage=float(args.perp_leverage),
        margin_slot_wallet_fraction=float(args.margin_slot_wallet_fraction),
        ev_curve=CAUSAL_AUCTION_CURVE,
    )
    evaluation_days = max((end - start).total_seconds() / 86_400.0, 1.0)
    summary["active_span_trades_per_day"] = float(summary.get("trades_per_day", float("nan")))
    summary["evaluation_calendar_days"] = float(evaluation_days)
    summary["trades_per_day"] = float(summary["accepted_trades"] / evaluation_days)
    monthly_wallet = _wallet_periods(
        equity, frequency="month", initial_wallet=float(args.initial_wallet),
        evaluation_start=start, evaluation_end=end,
    ).rename(columns={"period": "month"})
    monthly = monthly.merge(monthly_wallet, on="month", how="outer", validate="one_to_one")
    weekly = _weekly(decisions)
    weekly_wallet = _wallet_periods(
        equity, frequency="week", initial_wallet=float(args.initial_wallet),
        evaluation_start=start, evaluation_end=end,
    ).rename(columns={"period": "week"})
    weekly = weekly.merge(weekly_wallet, on="week", how="outer", validate="one_to_one")
    args.out_dir.mkdir(parents=True)
    frame.loc[:, [
        "candidate_id", "__decision_ts__", "producer_bundle_id",
        "causal_21d_side_expected_net_bps", "causal_21d_side_admitted_ge_50bps",
    ]].to_parquet(args.out_dir / "tail_health_admission_provenance.parquet", index=False)
    decisions.to_parquet(args.out_dir / "portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(args.out_dir / "portfolio_equity.parquet", index=False, compression="zstd")
    monthly.to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    weekly.to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    (args.out_dir / "portfolio_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema": "strict_r3_tail_health_shadow_portfolio_v1",
        "arm": args.arm,
        "scored_predictions": str(args.scored_predictions),
        "scored_predictions_sha256": _sha(args.scored_predictions),
        "scored_predictions_extra": [str(path) for path in args.scored_predictions_extra],
        "scored_predictions_extra_sha256": [
            _sha(path) for path in args.scored_predictions_extra
        ],
        "selection": str(args.selection),
        "selection_sha256": _sha(args.selection),
        "selection_extra": [str(path) for path in args.selection_extra],
        "selection_extra_sha256": [_sha(path) for path in args.selection_extra],
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "portfolio": "global auction; 8 concurrent; 2 new per 15m bar; 1 per asset; 80% margin cap",
        "auction_ev_curve": CAUSAL_AUCTION_CURVE,
        "policy": "already materialised frozen SimplePolicyOptimiser winner; no policy outcome recomputation",
        "contract": (
            "shadow-only candidate-keyed exact-producer tail-health admission; "
            "no model retraining, score mutation, or future outcome selection; "
            + (
                "existing same-producer final-score used only as a deterministic secondary order in exact mapped-EV ties"
                if args.same_producer_score_tiebreak
                else "no secondary score tie-break"
            )
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "summary": summary, **manifest}))


if __name__ == "__main__":
    main()
