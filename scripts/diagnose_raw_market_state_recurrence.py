#!/usr/bin/env python3
"""Backward causal raw-market-state recurrence and specialist-gate diagnosis.

This is deliberately a *diagnostic*, not a specialist training run.  It joins
the frozen direct execution-EV score with the raw market-state handoff on the
candidate identity, validates the completed-hour/source-staleness contract,
and re-fits an outcome-free raw-state basis before each weekly OOS block.

For each block, realised residuals from earlier, fully resolved blocks are
re-expressed in that block's frozen state basis.  They are then passed to the
predeclared ``specialist_eligibility`` gate from the adjacent-July adapter.
This lets us ask whether a state mapping has actually recurred before a
specialist is considered.  Week boundaries are evaluation partitions only:
they are never state features, labels, or sample weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_execution_regimes import CausalRegimeStateModel  # noqa: E402
from scripts.run_adjacent_july_state_adapter_ablation import (  # noqa: E402
    BASE_SCORE,
    DECISION,
    IDENTITY,
    RESOLUTION,
    SIDE,
    TARGET,
    specialist_eligibility,
)


SCHEMA = "raw_market_state_backward_recurrence_v1"
RAW_ROWS = ROOT / (
    "data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v1/"
    "raw_market_state_transition_rows.parquet"
)
SCORES = ROOT / (
    "data_perp/artifacts/"
    "execution_ev_context_clean_exact_recent_correction_forward_july19_20260726_v2/"
    "mapped_oof_and_forward.parquet"
)
TRANSITION_V2_SUMMARY = ROOT / (
    "data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v2/summary.json"
)
OUTPUT = ROOT / "data_perp/artifacts/raw_market_state_backward_recurrence_20260726_v1"
SOURCE_TIME = "raw_state_source_utc_h0"
RAW_PREFIX = "mkt_state__"
RAW_SUFFIX = "__h0"


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def raw_state_columns(frame: pd.DataFrame, *, coverage: float = 0.95) -> list[str]:
    """Return full-history raw fields which meet the stated coverage contract."""

    fields = [
        column
        for column in frame.columns
        if column.startswith(RAW_PREFIX) and column.endswith(RAW_SUFFIX)
    ]
    selected = [
        column
        for column in fields
        if float(pd.to_numeric(frame[column], errors="coerce").notna().mean())
        >= float(coverage)
    ]
    if not selected:
        raise ValueError("no raw state fields satisfy the coverage threshold")
    return sorted(selected)


def attach_frozen_scores(
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    stale_limit: pd.Timedelta = pd.Timedelta(minutes=90),
) -> pd.DataFrame:
    """Attach only the frozen direct score, with exact identity/time audits."""

    needed_raw = {*IDENTITY, DECISION, RESOLUTION, TARGET, SOURCE_TIME}
    needed_score = {*IDENTITY, DECISION, RESOLUTION, TARGET, BASE_SCORE}
    if missing := needed_raw.difference(raw.columns):
        raise ValueError(f"raw rows missing: {sorted(missing)}")
    if missing := needed_score.difference(scores.columns):
        raise ValueError(f"score rows missing: {sorted(missing)}")
    if raw["candidate_id"].duplicated().any() or scores["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique in both handoffs")
    raw = raw.copy()
    scores = scores.copy()
    for frame in (raw, scores):
        for column in ("__ts__", DECISION, RESOLUTION):
            frame[column] = _utc(frame[column])
    raw[SOURCE_TIME] = _utc(raw[SOURCE_TIME])
    score_keep = ["candidate_id", BASE_SCORE]
    if "evaluation_origin" in scores:
        score_keep.append("evaluation_origin")
    joined = raw.merge(
        scores.loc[:, score_keep], on="candidate_id", how="inner", validate="one_to_one"
    )
    # The raw-state builder's source timestamp is the completed hourly bar
    # open.  A row can only use a source at least one hour old and no more than
    # 90 minutes stale at the execution decision.
    age = joined[DECISION] - joined[SOURCE_TIME]
    valid = (
        joined[SOURCE_TIME].le(joined[DECISION] - pd.Timedelta(hours=1))
        & age.le(stale_limit)
        & age.ge(pd.Timedelta(hours=1))
    )
    if not bool(valid.all()):
        bad = joined.loc[~valid, ["candidate_id", SOURCE_TIME, DECISION]].head(5)
        raise ValueError(f"raw-state source timing violation: {bad.to_dict('records')}")
    numeric = joined.loc[:, [TARGET, BASE_SCORE]].apply(pd.to_numeric, errors="coerce")
    joined = joined.loc[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)].copy()
    joined.loc[:, numeric.columns] = numeric.loc[joined.index]
    return joined.sort_values([DECISION, "__symbol__", SIDE, "candidate_id"], kind="stable").reset_index(drop=True)


def completed_week_blocks(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Weekly evaluation partitions, independent of any state/regime meaning."""

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("block boundaries must be timezone-aware UTC")
    blocks: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current = start
    observed = _utc(frame[DECISION]).max() + pd.Timedelta(nanoseconds=1)
    final = min(end, observed)
    while current < final:
        stop = min(current + pd.Timedelta(days=7), final)
        if bool(_utc(frame[DECISION]).ge(current).mul(_utc(frame[DECISION]).lt(stop)).any()):
            blocks.append((current, stop, f"block_{current:%Y%m%d}"))
        current = stop
    return blocks


def resolved_prior_blocks(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    block_start: pd.Timestamp,
) -> pd.DataFrame:
    """Return prior block rows whose execution labels resolved before cutoff."""

    prior = frame.loc[
        _utc(frame[DECISION]).lt(cutoff) & _utc(frame[RESOLUTION]).lt(cutoff)
    ].copy()
    # The current partial prefix is intentionally excluded.  This turns every
    # eligibility decision into a completed-block decision, with the 12h label
    # purge applying before the candidate week begins.
    prior = prior.loc[_utc(prior[DECISION]).lt(block_start)].copy()
    ordering = [column for column in (DECISION, "__symbol__", SIDE, "candidate_id") if column in prior]
    return prior.sort_values(ordering, kind="stable")


def gate_context(
    prior: pd.DataFrame,
    state_model: CausalRegimeStateModel,
    raw_features: list[str],
) -> pd.DataFrame:
    """Transform resolved prior blocks into the evaluation's frozen basis."""

    if prior.empty:
        return prior.copy()
    transformed = state_model.transform(prior.loc[:, raw_features]).reset_index(drop=True)
    out = pd.concat([prior.reset_index(drop=True), transformed], axis=1)
    out["july_block"] = _utc(out[DECISION]).map(
        lambda ts: f"block_{(ts.normalize() - pd.Timedelta(days=ts.weekday())):%Y%m%d}"
    )
    # ``specialist_eligibility`` obtains the final two blocks in row order;
    # retain an explicit chronological ordering under its legacy field name.
    return out.sort_values([DECISION, "__symbol__", SIDE, "candidate_id"], kind="stable")


def _recurrence_rows(context: pd.DataFrame, eligibility: dict[str, Any], cutoff: pd.Timestamp) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if context.empty:
        return pd.DataFrame()
    for side in ("long", "short"):
        local = context.loc[context[SIDE].astype(str).eq(side)].copy()
        if local.empty:
            continue
        local["economic_residual"] = local[TARGET].to_numpy(dtype=float) - local[BASE_SCORE].to_numpy(dtype=float)
        grouped = local.groupby(["july_block", "causal_regime_state"], observed=True).agg(
            rows=("candidate_id", "size"),
            mean_economic_residual=("economic_residual", "mean"),
            mean_net_ev=(TARGET, "mean"),
        ).reset_index()
        grouped["side_name"] = side
        grouped["evaluation_cutoff_utc"] = cutoff
        report = eligibility.get("sides", {}).get(side, {})
        grouped["gate_side_eligible"] = bool(report.get("eligible", False))
        grouped["gate_recurring_states"] = json.dumps(report.get("recurring_states", []))
        grouped["gate_rank_correlation"] = report.get("rank_correlation")
        grouped["gate_sign_consistency"] = report.get("sign_consistency")
        grouped["gate_minimum_effect_range"] = report.get("minimum_within_block_effect_range")
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    raw = pd.read_parquet(args.raw_rows)
    scores = pd.read_parquet(args.scores)
    frame = attach_frozen_scores(raw, scores, stale_limit=pd.Timedelta(minutes=args.stale_minutes))
    features = raw_state_columns(frame, coverage=args.coverage_threshold)
    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    start = pd.Timestamp(args.first_eval, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    blocks = completed_week_blocks(frame, start=start, end=end)
    reports: list[dict[str, Any]] = []
    recurrence: list[pd.DataFrame] = []
    emitted: list[pd.DataFrame] = []
    decision = _utc(frame[DECISION])
    for block_start, block_end, block_name in blocks:
        current = frame.loc[decision.ge(block_start) & decision.lt(block_end)].copy()
        prior_state_fit = frame.loc[decision.lt(block_start)].copy()
        side_payload: dict[str, Any] = {}
        for side in ("long", "short"):
            state_train = prior_state_fit.loc[prior_state_fit[SIDE].astype(str).eq(side)].copy()
            evaluation = current.loc[current[SIDE].astype(str).eq(side)].copy()
            if len(state_train) < int(args.min_state_fit_rows) or evaluation.empty:
                side_payload[side] = {"status": "insufficient_state_fit_or_evaluation", "state_fit_rows": int(len(state_train)), "evaluation_rows": int(len(evaluation))}
                continue
            model = CausalRegimeStateModel.fit(state_train, features)
            prior = resolved_prior_blocks(
                frame.loc[frame[SIDE].astype(str).eq(side)],
                cutoff=block_start,
                block_start=block_start,
            )
            context = gate_context(prior, model, features)
            eligibility = specialist_eligibility(
                context,
                min_state_rows=args.min_state_rows,
                min_recurring_states=args.min_recurring_states,
                min_effect_range=args.min_effect_range,
                min_week_rank_correlation=args.min_week_rank_correlation,
            )
            recurrent = _recurrence_rows(context, eligibility, block_start)
            if not recurrent.empty:
                recurrence.append(recurrent)
            transformed = model.transform(evaluation.loc[:, features]).reset_index(drop=True)
            diagnostic = pd.concat([evaluation.reset_index(drop=True), transformed], axis=1)
            diagnostic["evaluation_block"] = block_name
            diagnostic["state_fit_cutoff_utc"] = block_start
            diagnostic["specialist_eligible_before_block"] = bool(
                eligibility.get("sides", {}).get(side, {}).get("eligible", False)
            )
            emitted.append(diagnostic)
            side_report = eligibility["sides"].get(side, {})
            side_payload[side] = {
                "status": "evaluated",
                "state_fit_rows": int(len(state_train)),
                "evaluation_rows": int(len(evaluation)),
                "selected_k": int(model.selected_k),
                "resolved_prior_rows": int(len(context)),
                "resolved_prior_max_label_end": _utc(context[RESOLUTION]).max().isoformat() if len(context) else None,
                "specialist_eligibility": side_report,
                "gate_decision": eligibility["decision"],
                "raw_state_feature_count": len(features),
            }
        reports.append({"block": block_name, "start": block_start.isoformat(), "end_exclusive": block_end.isoformat(), "sides": side_payload})
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "summary.json"
    recurrence_path = args.output_dir / "recurrence_by_prior_block_state.csv"
    rows_path = args.output_dir / "weekly_raw_state_diagnostic_rows.parquet"
    pd.concat(recurrence, ignore_index=True).to_csv(recurrence_path, index=False) if recurrence else pd.DataFrame().to_csv(recurrence_path, index=False)
    pd.concat(emitted, ignore_index=True).to_parquet(rows_path, index=False) if emitted else pd.DataFrame().to_parquet(rows_path, index=False)
    transition_summary: dict[str, Any] | None = None
    if args.transition_v2_summary.is_file():
        transition_summary = json.loads(args.transition_v2_summary.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed_diagnostic_not_specialist_promotion",
        "contract": {
            "score": "frozen direct execution-EV score only; no fitted score, calendar label, or sample weighting",
            "raw_state": "raw market-state fields attached only from completed hourly sources: source <= decision-1h and source age <= 90m",
            "state_fit": "per side, K=3..5 outcome-free state geometry refit only on raw states preceding each evaluation week",
            "gate": "existing specialist_eligibility gate applied only to fully resolved earlier weekly blocks re-expressed in the current frozen state basis",
            "purge": "a prior economic residual is eligible only when execution_label_end_utc < evaluation week start",
            "calendar": "weekly partitions support temporal OOS evaluation only; no calendar feature, state label, or calendar/regime weight exists",
        },
        "sources": {"raw_rows": str(args.raw_rows.resolve()), "scores": str(args.scores.resolve()), "transition_v2_summary": str(args.transition_v2_summary.resolve()) if args.transition_v2_summary.is_file() else None},
        "transition_v2_coordinate": {"schema": transition_summary.get("schema"), "strict_oof": transition_summary.get("strict_oof"), "rows": transition_summary.get("rows")} if transition_summary else None,
        "joined_rows": int(len(frame)),
        "date_range": {"decision_min": _utc(frame[DECISION]).min().isoformat(), "decision_max": _utc(frame[DECISION]).max().isoformat()},
        "raw_features": features,
        "feature_coverage_threshold": float(args.coverage_threshold),
        "blocks": reports,
        "outputs": {"recurrence": str(recurrence_path.resolve()), "diagnostic_rows": str(rows_path.resolve())},
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {"report": report_path, "recurrence": recurrence_path, "diagnostic_rows": rows_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rows", type=Path, default=RAW_ROWS)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--transition-v2-summary", type=Path, default=TRANSITION_V2_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--first-eval", default="2026-06-08")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--stale-minutes", type=int, default=90)
    parser.add_argument("--min-state-fit-rows", type=int, default=500)
    parser.add_argument("--min-state-rows", type=int, default=100)
    parser.add_argument("--min-recurring-states", type=int, default=2)
    parser.add_argument("--min-effect-range", type=float, default=0.002)
    parser.add_argument("--min-week-rank-correlation", type=float, default=0.50)
    return parser


if __name__ == "__main__":
    print(json.dumps({key: str(value) for key, value in run(_parser().parse_args()).items()}, indent=2))
