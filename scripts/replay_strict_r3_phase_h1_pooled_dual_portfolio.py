#!/usr/bin/env python3
"""Run four phase-local prequential dual maps through one global portfolio.

The current-v5 and BCF score routes remain phase-local, but a live four-times
per-hour trader owns one portfolio.  This producer therefore joins each phase
on its target-free identity, freezes dual admission before looking at policy
outcomes, excludes invalid outcome paths before simulated capacity, and runs a
single chronological constrained auction across all supplied phases.
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

from extreme_price_movements.portfolio_policy_replay import replay_candidates
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _candidate_table,
    _metrics,
    _params,
)

POLICY = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, family: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score", "mc1_expected_bps", *POLICY}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{family} misses required fields: {missing}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {family}: {path}")
    return frame


def _join(phase: int, current_path: Path, bcf_path: Path, threshold: float) -> tuple[pd.DataFrame, dict[str, int]]:
    current = _load(current_path, f"phase{phase}_current")
    bcf = _load(bcf_path, f"phase{phase}_bcf")
    bcf_columns = list(dict.fromkeys(["candidate_id", "__decision_ts__", "mc1_expected_bps", *POLICY]))
    bcf_map = bcf.loc[:, bcf_columns].copy()
    joined = current.merge(bcf_map, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "__bcf"))
    if len(joined) != len(current):
        raise ValueError(f"phase {phase}: every current-routed identity must have BCF coordinates")
    if not joined["__decision_ts__"].eq(joined["__decision_ts____bcf"]).all():
        raise ValueError(f"phase {phase}: current/BCF decision identity mismatch")
    for col in POLICY:
        if col == "candidate_id":
            continue
        left, right = joined[col], joined[f"{col}__bcf"]
        if pd.api.types.is_numeric_dtype(left):
            same = np.isclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), equal_nan=True)
        else:
            same = left.fillna("<NA>").astype(str).eq(right.fillna("<NA>").astype(str)).to_numpy()
        if not bool(np.all(same)):
            raise ValueError(f"phase {phase}: post-score policy contract differs across score families ({col})")
    joined = joined.drop(columns=["__decision_ts____bcf", *[f"{col}__bcf" for col in POLICY if col != "candidate_id"]])
    joined = joined.rename(columns={"mc1_expected_bps": "current_mc1_expected_bps", "mc1_expected_bps__bcf": "bcf_mc1_expected_bps"})
    joined["phase_minutes"] = phase
    joined["dual_admitted"] = joined["current_mc1_expected_bps"].ge(threshold) & joined["bcf_mc1_expected_bps"].ge(threshold)
    # The minimum carries the exact "both >= threshold" rule into the
    # no-lookahead auction adapter.  Auction priority stays final_score.
    joined["mc1_expected_bps"] = np.minimum(joined["current_mc1_expected_bps"], joined["bcf_mc1_expected_bps"])
    if not joined["dual_admitted"].eq(joined["mc1_expected_bps"].ge(threshold)).all():
        raise AssertionError("dual admission/minimum EV equivalence failed")
    audit = {
        "phase_minutes": phase,
        "current_routed_rows": int(len(current)),
        "dual_admitted_target_free_rows": int(joined["dual_admitted"].sum()),
    }
    return joined, audit


def _aggregate(decisions: pd.DataFrame, key: str) -> pd.DataFrame:
    accepted_flag = decisions.get("accepted", pd.Series(False, index=decisions.index))
    accepted = decisions.loc[accepted_flag.fillna(False).astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=[key, "entries", "net_ev_bps", "net_sum_bps"])
    timestamp = pd.to_datetime(accepted["timestamp"], utc=True)
    accepted[key] = timestamp.dt.strftime("%Y-%m" if key == "month" else "%G-W%V")
    return accepted.groupby(key, sort=True).agg(
        entries=("candidate_id", "size"),
        net_ev_bps=("position_net_return", lambda x: float(pd.to_numeric(x, errors="coerce").mean() * 10_000.0)),
        net_sum_bps=("position_net_return", lambda x: float(pd.to_numeric(x, errors="coerce").sum() * 10_000.0)),
    ).reset_index()


def _phase_hourly_admissions(
    combined: pd.DataFrame,
    decisions: pd.DataFrame,
    threshold_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report executable admission provenance at each phase decision hour.

    The decision is frozen before outcome fields are consulted.  Outcome
    validity and auction acceptance are attached afterward solely for audit,
    so this table makes the distinction explicit instead of presenting a
    retrospective tail statistic as an admission result.
    """
    work = combined.loc[:, [
        "candidate_id", "__decision_ts__", "phase_minutes", "current_mc1_expected_bps",
        "bcf_mc1_expected_bps", "dual_admitted", "policy_path_valid",
    ]].copy()
    work["hour"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.floor("1h")
    work["current_mapper_pass"] = work["current_mc1_expected_bps"].ge(threshold_bps)
    work["bcf_mapper_pass"] = work["bcf_mc1_expected_bps"].ge(threshold_bps)
    work["dual_valid_outcome"] = (
        work["dual_admitted"].fillna(False).astype(bool)
        & work["policy_path_valid"].fillna(False).astype(bool)
    )
    accepted = decisions.loc[
        decisions.get("accepted", pd.Series(False, index=decisions.index)).fillna(False).astype(bool),
        ["candidate_id", "position_net_return"],
    ].copy() if len(decisions) else pd.DataFrame(columns=["candidate_id", "position_net_return"])
    accepted["candidate_id"] = accepted["candidate_id"].astype(str)
    accepted["portfolio_accepted"] = True
    accepted["accepted_net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
    work = work.merge(
        accepted.loc[:, ["candidate_id", "portfolio_accepted", "accepted_net_bps"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    # ``merge`` widens the sparse accepted flag to object dtype.  Equality is
    # both explicit about missing rows meaning "not accepted" and avoids a
    # pandas down-cast during the audit-only aggregation.
    work["portfolio_accepted"] = work["portfolio_accepted"].eq(True)
    hourly = work.groupby(["phase_minutes", "hour"], as_index=False, sort=True).agg(
        current_routed_rows=("candidate_id", "size"),
        current_mapper_pass_rows=("current_mapper_pass", "sum"),
        bcf_mapper_pass_rows=("bcf_mapper_pass", "sum"),
        dual_admitted_target_free_rows=("dual_admitted", "sum"),
        dual_admitted_valid_outcome_rows=("dual_valid_outcome", "sum"),
        portfolio_accepted_rows=("portfolio_accepted", "sum"),
        accepted_net_ev_bps=("accepted_net_bps", "mean"),
        accepted_net_sum_bps=("accepted_net_bps", "sum"),
    )
    summary = hourly.groupby("phase_minutes", as_index=False, sort=True).agg(
        decision_hours=("hour", "size"),
        current_routed_rows=("current_routed_rows", "sum"),
        current_mapper_pass_rows=("current_mapper_pass_rows", "sum"),
        bcf_mapper_pass_rows=("bcf_mapper_pass_rows", "sum"),
        dual_admitted_target_free_rows=("dual_admitted_target_free_rows", "sum"),
        dual_admitted_valid_outcome_rows=("dual_admitted_valid_outcome_rows", "sum"),
        portfolio_accepted_rows=("portfolio_accepted_rows", "sum"),
        accepted_net_sum_bps=("accepted_net_sum_bps", "sum"),
    )
    summary["accepted_net_ev_bps"] = (
        summary["accepted_net_sum_bps"]
        / summary["portfolio_accepted_rows"].replace(0, np.nan)
    )
    return hourly, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", action="append", nargs=3, metavar=("MINUTES", "CURRENT", "BCF"), required=True)
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    phases = []
    for raw_phase, current, bcf in args.phase:
        phase = int(raw_phase)
        if phase not in (0, 15, 30, 45):
            raise ValueError(f"unsupported phase {phase}")
        phases.append((phase, Path(current), Path(bcf)))
    if len({phase for phase, _, _ in phases}) != len(phases):
        raise ValueError("phase supplied more than once")
    pieces, audits = [], []
    for phase, current, bcf in sorted(phases):
        frame, audit = _join(phase, current, bcf, args.threshold_bps)
        pieces.append(frame)
        audits.append(audit)
    combined = pd.concat(pieces, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    if combined["candidate_id"].duplicated().any():
        raise AssertionError("candidate IDs collide across phases")
    policy = combined.loc[:, list(POLICY)].copy()
    candidates = _candidate_table(combined, policy, args.threshold_bps, invalid_outcome_mode="exclude")
    # The generic portfolio adapter drops research-only provenance fields.
    # Reattach the immutable phase tag after admission by candidate identity;
    # it is reporting metadata only and cannot affect ranking or capacity.
    phase_lookup = combined.loc[:, ["candidate_id", "phase_minutes"]]
    candidates = candidates.merge(phase_lookup, on="candidate_id", how="left", validate="one_to_one")
    if candidates["phase_minutes"].isna().any():
        raise AssertionError("admitted candidate lost its phase provenance")
    decisions, equity, _ = replay_candidates(
        candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    if len(decisions):
        lookup = candidates.loc[:, ["candidate_id", "phase_minutes", "policy_outcome_available", "mapped_expected_net_bps"]].reset_index(drop=True)
        lookup.index.name = "candidate_index"
        decisions = decisions.merge(lookup, on="candidate_index", how="left", validate="many_to_one")
    else:
        decisions["policy_outcome_available"] = pd.Series(dtype=bool)
    accepted = decisions.get("accepted", pd.Series(False, index=decisions.index)).fillna(False).astype(bool)
    phase_valid = candidates.groupby("phase_minutes", sort=True).size().to_dict()
    for audit in audits:
        audit["valid_outcome_admitted_rows"] = int(phase_valid.get(audit["phase_minutes"], 0))
        audit["portfolio_accepted_rows"] = int(decisions.loc[accepted, "phase_minutes"].eq(audit["phase_minutes"]).sum())
    metrics = _metrics(decisions, equity, "pooled_four_phase_dual", "all")
    metrics.update({
        "threshold_bps": float(args.threshold_bps),
        "phases": sorted(phase for phase, _, _ in phases),
        "dual_scored_rows": int(len(combined)),
        "dual_admitted_target_free_rows": int(combined["dual_admitted"].sum()),
        "valid_outcome_rows_after_admission": int(len(candidates)),
        "auction_priority": "current_v5_final_score",
    })
    args.out_dir.mkdir(parents=True)
    combined.to_parquet(args.out_dir / "pooled_dual_target_free_then_outcome.parquet", index=False, compression="zstd")
    candidates.to_parquet(args.out_dir / "pooled_admitted_candidates_after_outcome_join.parquet", index=False, compression="zstd")
    decisions.to_parquet(args.out_dir / "portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(args.out_dir / "portfolio_equity.parquet", index=False, compression="zstd")
    _aggregate(decisions, "month").to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    _aggregate(decisions, "week").to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    hourly_admissions, phase_hourly_summary = _phase_hourly_admissions(
        combined, decisions, args.threshold_bps,
    )
    hourly_admissions.to_parquet(args.out_dir / "phase_hourly_admissions.parquet", index=False)
    phase_hourly_summary.to_parquet(args.out_dir / "phase_hourly_admission_summary.parquet", index=False)
    pd.DataFrame(audits).to_parquet(args.out_dir / "phase_admission_audit.parquet", index=False)
    pd.DataFrame([metrics]).to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_phase_h1_pooled_dual_portfolio_v1",
        "target_free_until_policy_join": True,
        "phase_score_contract": "current-v5 base route/top-30 + BCF; phase-local strict-prequential maps",
        "admission": "current MC1 >= floor AND BCF MC1 >= floor",
        "threshold_bps": float(args.threshold_bps),
        "auction": "one global chronological stateful auction; current-v5 final_score priority",
        "outcome_policy": "invalid paths excluded after admission and before capacity",
        "inputs": [{"phase": phase, "current": str(current), "current_sha256": _sha(current), "bcf": str(bcf), "bcf_sha256": _sha(bcf)} for phase, current, bcf in phases],
        "phase_admission_audit": audits,
        "per_phase_hourly_admission_metrics": "phase_hourly_admissions.parquet",
        "metrics": metrics,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "complete", **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
