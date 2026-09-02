#!/usr/bin/env python3
"""Diagnose how direct-base information survives the executable stack.

This is an offline, strict-OOS analysis companion to
``run_strict_r3_long_supportive_consensus_walkforward.py``.  It refits only
the already-declared monthly models for the B0 control and the selected
B0/direct stack, then records aggregate cohorts:

    B0 route -> direct-only route additions -> dual-MC1 admission ->
    rich-policy portfolio candidates -> constrained auction acceptances.

Outcomes are joined only after each monthly held score is generated.  No
output is eligible for model selection, live scoring, or execution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_long_supportive_consensus_walkforward as stack


SCHEMA = "strict_r3_long_supportive_consensus_waterfall_v1"
CONTROL = ("I0_b0_single", "L0_policy_residual")
CHALLENGER = ("I3_b0_direct_stack", "L1_direct_residual")
PERIODS = {
    "selection_2025_juldec": tuple(pd.date_range("2025-07-01", "2025-12-01", freq="MS", tz="UTC")),
    "portability_2026_aprjul": tuple(pd.date_range("2026-04-01", "2026-07-01", freq="MS", tz="UTC")),
}
COMPACT_HELD_COLUMNS = (
    "candidate_id", "__decision_ts__", "__symbol__", "base_routed",
    "consensus_final_score", "policy_path_valid", "policy_gross_bps",
    "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price",
    "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
    "current_mc1_expected_bps", "bcf_mc1_expected_bps",
)


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    expanded: list[Path] = []
    for path in paths:
        expanded.extend(sorted(path.rglob("*.parquet")) if path.is_dir() else [path])
    for path in sorted(expanded):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _sources_from_args(args: argparse.Namespace) -> stack.Sources:
    return stack.Sources(
        stage1=args.stage1.resolve(), direct=args.direct.resolve(), stage2=args.stage2.resolve(),
        causal_joint=args.causal_joint.resolve(), current_mc1=args.current_mc1.resolve(),
        bcf_mc1=args.bcf_mc1.resolve(), policy=args.policy.resolve(),
    )


def _load_population(source: stack.Sources) -> pd.DataFrame:
    population = stack._load_population(source)
    policy, mc1 = stack._load_policy_and_mc1(source)
    population = population.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    population = population.merge(mc1, on="candidate_id", how="left", validate="one_to_one")
    if population["policy_path_valid"].isna().mean() > 0.80:
        raise AssertionError("rich-policy sidecar coverage unexpectedly absent")
    return population


def _score_period(
    population: pd.DataFrame,
    months: tuple[pd.Timestamp, ...],
    architecture: str,
    target: str,
    period: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prepared = stack._prepare(population, architecture)
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for month in months:
        held, audit = stack._score_month(prepared, month, architecture=architecture, target=target)
        missing = sorted(set(COMPACT_HELD_COLUMNS).difference(held.columns))
        if missing:
            raise AssertionError(f"scored held panel misses compact waterfall fields: {missing}")
        # Do not retain the large model-feature frame.  These are the only
        # post-score fields used by downstream routing, admission, auction,
        # and retrospective cohort metrics.
        held = held.loc[:, list(COMPACT_HELD_COLUMNS)].copy()
        held["period"] = period
        pieces.append(held)
        audits.append({"period": period, "month": month, **audit})
    return pd.concat(pieces, ignore_index=True), audits


def _valid_outcome(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_exit_bar_15m"], errors="coerce"))
    )


def _cohort_row(frame: pd.DataFrame, *, period: str, threshold: float, cohort: str, mask: pd.Series) -> dict[str, Any]:
    selected = frame.loc[mask].copy()
    net = pd.to_numeric(selected.get("policy_net_bps", pd.Series(dtype=float)), errors="coerce")
    return {
        "period": period,
        "admission_threshold_bps": float(threshold),
        "cohort": cohort,
        "rows": int(len(selected)),
        "net_ev_bps_per_trade": float(net.mean()) if len(net) else float("nan"),
        "net_sum_bps": float(net.sum()) if len(net) else 0.0,
        "positive_fraction": float(net.gt(0).mean()) if len(net) else float("nan"),
        "policy_ge50_fraction": float(net.ge(50.0).mean()) if len(net) else float("nan"),
    }


def _candidate_decisions(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidate = stack._candidate_table(frame, threshold_bps=float(threshold))
    decisions, _ = stack._replay_research_contract(candidate)
    return decisions


def _waterfall_for_period(
    control: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    period: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = ["candidate_id", "__decision_ts__"]
    left = control.loc[:, [*keys, "base_routed", "consensus_final_score"]].rename(columns={
        "base_routed": "b0_routed", "consensus_final_score": "b0_consensus_final_score",
    })
    work = challenger.merge(left, on=keys, how="inner", validate="one_to_one")
    if len(work) != len(challenger):
        raise AssertionError("control/challenger held identities differ")
    valid = _valid_outcome(work)
    direct_union = work["base_routed"].fillna(False).astype(bool)
    b0_route = work["b0_routed"].fillna(False).astype(bool)
    direct_only = direct_union & ~b0_route
    dual = (
        pd.to_numeric(work["current_mc1_expected_bps"], errors="coerce").ge(float(threshold))
        & pd.to_numeric(work["bcf_mc1_expected_bps"], errors="coerce").ge(float(threshold))
    )
    rows = [
        _cohort_row(work, period=period, threshold=threshold, cohort="all valid scored", mask=valid),
        _cohort_row(work, period=period, threshold=threshold, cohort="B0 routed valid", mask=valid & b0_route),
        _cohort_row(work, period=period, threshold=threshold, cohort="direct-union routed valid", mask=valid & direct_union),
        _cohort_row(work, period=period, threshold=threshold, cohort="direct-only routed addition", mask=valid & direct_only),
        _cohort_row(work, period=period, threshold=threshold, cohort="direct-only addition passing dual MC1", mask=valid & direct_only & dual),
        _cohort_row(work, period=period, threshold=threshold, cohort="B0 routed passing dual MC1", mask=valid & b0_route & dual),
    ]

    c_decisions = _candidate_decisions(control, threshold)
    d_decisions = _candidate_decisions(challenger, threshold)
    c_accepted = set(c_decisions.loc[c_decisions["accepted"].fillna(False).astype(bool), "candidate_id"].astype(str))
    d_accepted = set(d_decisions.loc[d_decisions["accepted"].fillna(False).astype(bool), "candidate_id"].astype(str))
    work_id = work["candidate_id"].astype(str)
    candidate_set = set(d_decisions["candidate_id"].astype(str))
    rows.extend([
        _cohort_row(work, period=period, threshold=threshold, cohort="direct-only addition entering auction", mask=valid & direct_only & work_id.isin(candidate_set)),
        _cohort_row(work, period=period, threshold=threshold, cohort="direct-only addition auction accepted", mask=valid & direct_only & work_id.isin(d_accepted)),
        _cohort_row(work, period=period, threshold=threshold, cohort="accepted by both architectures", mask=valid & work_id.isin(c_accepted & d_accepted)),
        _cohort_row(work, period=period, threshold=threshold, cohort="I3 direct-stack-only accepted", mask=valid & work_id.isin(d_accepted - c_accepted)),
        _cohort_row(work, period=period, threshold=threshold, cohort="I0 B0-control-only accepted", mask=valid & work_id.isin(c_accepted - d_accepted)),
    ])

    reject = d_decisions.loc[
        d_decisions["candidate_id"].astype(str).isin(set(work.loc[valid & direct_only & dual, "candidate_id"].astype(str)))
        & ~d_decisions["accepted"].fillna(False).astype(bool)
    ]
    rejection_rows = (
        reject.groupby("reject_reason", dropna=False, sort=True).size().reset_index(name="rows")
        if not reject.empty else pd.DataFrame(columns=["reject_reason", "rows"])
    )
    rejections = [
        {"period": period, "admission_threshold_bps": float(threshold), "cohort": "direct-only dual-MC1", "reject_reason": str(row.reject_reason), "rows": int(row.rows)}
        for row in rejection_rows.itertuples(index=False)
    ]
    return rows, rejections


def run(sources: stack.Sources, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    audit_rows: list[dict[str, Any]] = []
    waterfall_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for period, months in PERIODS.items():
        # Each period needs only its three-month rolling-fit history plus held
        # months.  Loading/scoping separately keeps the diagnostic bounded
        # without changing the source population or monthly fit contract.
        all_population = _load_population(sources)
        start = min(months) - pd.DateOffset(months=3)
        end = max(months) + pd.offsets.MonthBegin(1)
        population = all_population.loc[
            all_population["__decision_ts__"].ge(start)
            & all_population["__decision_ts__"].lt(end)
        ].copy()
        del all_population
        gc.collect()
        control, control_audit = _score_period(population, months, *CONTROL, period)
        challenger, challenger_audit = _score_period(population, months, *CHALLENGER, period)
        audit_rows.extend(control_audit); audit_rows.extend(challenger_audit)
        for threshold in stack.MC1_THRESHOLDS_BPS:
            rows, rejections = _waterfall_for_period(control, challenger, period=period, threshold=float(threshold))
            waterfall_rows.extend(rows); rejection_rows.extend(rejections)
        del control, challenger, population
        gc.collect()
    pd.DataFrame(audit_rows).to_parquet(out / "walkforward_fit_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(waterfall_rows).to_parquet(out / "waterfall_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(rejection_rows).to_parquet(out / "direct_only_auction_rejections.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "scope": "offline diagnostic only; no model selection, live mutation, exchange I/O, or execution",
        "control": {"architecture": CONTROL[0], "target": CONTROL[1]},
        "challenger": {"architecture": CHALLENGER[0], "target": CHALLENGER[1]},
        "periods": {name: [str(month) for month in months] for name, months in PERIODS.items()},
        "dual_admission": {"thresholds_bps": list(stack.MC1_THRESHOLDS_BPS), "rule": "current MC1 >= threshold AND BCF MC1 >= threshold"},
        "portfolio": "same narrow offline constraint mirror as source consensus evaluation",
        "sources": {key: str(value.resolve()) for key, value in vars(sources).items()},
        "source_sha256": {key: _sha256([value]) for key, value in vars(sources).items()},
        "waterfall": "direct-only means challenger top-30 union route minus B0 top-30 route; outcomes appear only in aggregate retrospective metrics",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--causal-joint", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    print(run(_sources_from_args(args), args.out.resolve()))


if __name__ == "__main__":
    main()
