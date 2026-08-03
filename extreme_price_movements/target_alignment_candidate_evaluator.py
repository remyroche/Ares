"""Strict candidate-level evaluator for the target--feature--execution matrix.

The evaluator deliberately contains no model fitting and no portfolio logic.
It accepts only prequential OOF candidate scores, ranks each arm once across
the complete candidate population, then attributes that frozen global book by
side/month/context.  A threshold entry table is separate from the global-tail
diagnostics and has no quotas, replacement, or cross-candidate constraints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.candidate_evaluation import FRACTIONS, EvaluationError, paired_day_block_bootstrap, stable_global_top_k


REQUIRED_ARMS = ("T0_native24_control", "T1_clean_opportunity", "T2_direct_net", "T3_competing_risk_expected_net", "T4_hurdle_decomposition")


class AlignmentEvaluationError(EvaluationError):
    """Raised when candidate-level target evidence is not strict and paired."""


@dataclass(frozen=True)
class Columns:
    score: str = "score"
    net: str = "execution_net_ev_12h"
    gross: str = "execution_gross_ev_12h"
    cost: str = "execution_cost_return"
    decision: str = "__decision_ts__"
    label_available: str = "__label_available_at__"
    fit_end: str = "prediction_fit_end_ts"
    generated: str = "prediction_generated_ts"
    strict_oof: str = "strict_prequential_oof"
    diagnostic_oof: str = "diagnostic_noncausal_oof"


def _bps(frame: pd.DataFrame, name: str, unit: str) -> pd.Series:
    value = pd.to_numeric(frame[name], errors="coerce")
    if not np.isfinite(value).all():
        raise AlignmentEvaluationError(f"{name} must be finite")
    if unit == "return":
        return value * 10_000.0
    if unit == "bps":
        return value
    raise AlignmentEvaluationError("units must be return or bps")


def _id_hash(values: Iterable[object]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _normalise(frame: pd.DataFrame, columns: Columns) -> pd.DataFrame:
    required = {
        "candidate_id", "target_arm", columns.score, columns.net, columns.gross, columns.cost,
        columns.decision, columns.label_available, columns.fit_end, columns.generated, columns.strict_oof,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AlignmentEvaluationError(f"candidate evaluator lacks required fields: {missing}")
    work = frame.copy()
    for name in (columns.decision, columns.label_available, columns.fit_end, columns.generated):
        work[name] = pd.to_datetime(work[name], utc=True, errors="coerce")
    if "__ts__" not in work:
        # The generic bootstrap helper consumes a protocol timestamp.  Test
        # callers may provide only the declared decision timestamp; alias it
        # without changing the immutable candidate identity.
        work["__ts__"] = work[columns.decision]
    if work[[columns.decision, columns.label_available, columns.fit_end, columns.generated]].isna().any().any():
        raise AlignmentEvaluationError("candidate OOF timestamp fields must be valid UTC")
    if not work[columns.strict_oof].astype(bool).all():
        raise AlignmentEvaluationError("diagnostic/non-strict OOF predictions cannot enter alignment evaluation")
    if columns.diagnostic_oof in work and work[columns.diagnostic_oof].astype(bool).any():
        raise AlignmentEvaluationError("diagnostic_noncausal_oof rows cannot enter alignment evaluation")
    if not work[columns.fit_end].lt(work[columns.decision]).all():
        raise AlignmentEvaluationError("prediction_fit_end_ts must precede candidate decision_ts")
    if not work[columns.generated].le(work[columns.decision]).all():
        raise AlignmentEvaluationError("prediction_generated_ts cannot follow candidate decision_ts")
    if not work[columns.label_available].ge(work[columns.decision] + pd.Timedelta(hours=12)).all():
        raise AlignmentEvaluationError("exact H12 labels must resolve at or after the horizon")
    cell_keys = ["candidate_id", "target_arm"]
    if "support_stage" in work:
        cell_keys.append("support_stage")
    if work.duplicated(cell_keys).any():
        raise AlignmentEvaluationError(
            "one score is required per candidate/target/support cell"
        )
    gross, cost, net = (_bps(work, getattr(columns, name), "return") for name in ("gross", "cost", "net"))
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise AlignmentEvaluationError("exact-net evaluation requires gross minus one row cost equal net")
    return work


def validate_pairing(frame: pd.DataFrame, columns: Columns = Columns()) -> list[dict[str, Any]]:
    """Return machine-readable causal/paired-invariant results without fitting."""
    checks: list[dict[str, Any]] = []
    try:
        work = _normalise(frame, columns)
        checks.append({"check": "strict_prequential_candidate_oof", "passed": True, "value": str(int(len(work)))})
    except AlignmentEvaluationError as error:
        return [{"check": "strict_prequential_candidate_oof", "passed": False, "value": str(error)}]
    cell_columns = ["target_arm"] + (["support_stage"] if "support_stage" in work else [])
    identities = {
        tuple(str(value) for value in (key if isinstance(key, tuple) else (key,))): tuple(sorted(group.candidate_id.astype(str)))
        for key, group in work.groupby(cell_columns, sort=True, observed=True)
    }
    first = next(iter(identities.values()), ())
    checks.append({
        "check": "identical_evaluation_rows_across_target_arms",
        "passed": bool(all(ids == first for ids in identities.values())),
        "value": json.dumps({"|".join(arm): _id_hash(ids) for arm, ids in identities.items()}, sort_keys=True),
    })
    present_arms = {key[0] for key in identities}
    checks.append({"check": "all_required_t0_t4_arms_present", "passed": set(REQUIRED_ARMS).issubset(present_arms), "value": json.dumps(sorted(present_arms))})
    checks.append({"check": "cost_applied_exactly_once", "passed": True, "value": "gross_bps - row_cost_bps == exact_net_bps"})
    return checks


def _summary(selected: pd.DataFrame, columns: Columns, *, net_unit: str, gross_unit: str, cost_unit: str) -> dict[str, float | int]:
    net, gross, cost = (_bps(selected, getattr(columns, name), unit) for name, unit in (("net", net_unit), ("gross", gross_unit), ("cost", cost_unit)))
    positive, negative = net.gt(0.0), net.lt(0.0)
    result: dict[str, float | int] = {
        "rows": int(len(selected)), "gross_bps_per_trade": float(gross.mean()), "cost_bps_per_trade": float(cost.mean()),
        "net_bps_per_trade": float(net.mean()), "win_rate": float(positive.mean()),
        "payoff_ratio": float(net[positive].mean() / abs(net[negative].mean())) if positive.any() and negative.any() and abs(net[negative].mean()) > 0 else np.nan,
        "profit_factor": float(net[positive].sum() / abs(net[negative].sum())) if negative.any() and abs(net[negative].sum()) > 0 else (np.inf if positive.any() else 0.0),
    }
    for output, choices in (("mean_mfe", ("mfe_atr_h12", "peak_mfe_atr_h12", "__peak_mfe_atr_12h__")), ("mean_mae", ("mae_atr_h12", "worst_mae_atr_h12", "__mae_before_meaningful_mfe_atr_12h__"))):
        source = next((name for name in choices if name in selected), None)
        result[output] = float(pd.to_numeric(selected[source], errors="coerce").mean()) if source else np.nan
    for output, choices in (("clean_share", ("clean", "favorable_first")), ("adverse_share", ("adverse", "adverse_first")), ("timeout_share", ("timeout",))):
        source = next((name for name in choices if name in selected), None)
        result[output] = float(pd.to_numeric(selected[source], errors="coerce").mean()) if source else np.nan
    return result


def _attribution(book: pd.DataFrame, *, dimension: str, columns: Columns, units: tuple[str, str, str]) -> pd.DataFrame:
    if dimension not in book:
        return pd.DataFrame()
    rows = []
    for value, group in book.groupby(dimension, observed=True, dropna=False, sort=True):
        rows.append({"dimension": dimension, "dimension_value": "<missing>" if pd.isna(value) else str(value), "selected_share": len(group) / len(book), **_summary(group, columns, net_unit=units[0], gross_unit=units[1], cost_unit=units[2])})
    return pd.DataFrame(rows)


def evaluate_target_arms(
    frame: pd.DataFrame,
    *,
    columns: Columns = Columns(),
    score_unit: str = "return",
    net_unit: str = "return",
    gross_unit: str = "return",
    cost_unit: str = "return",
    fractions: Sequence[float] = FRACTIONS,
    entry_threshold: float | None = 0.0,
    minimum_clean_probability: float | None = None,
    clean_probability_column: str = "predicted_clean_probability",
    context_columns: Sequence[str] = ("side_name", "regime", "liquidity_decile", "predicted_cost_decile", "opportunity_hurdle_decile"),
    bootstrap_baseline_arm: str | None = "T2_direct_net",
) -> Mapping[str, pd.DataFrame]:
    """Evaluate T0--T4/supportive arms with frozen, pooled-global masks."""
    checks = validate_pairing(frame, columns)
    if not all(row["passed"] for row in checks):
        raise AlignmentEvaluationError(f"alignment invariants fail: {[row for row in checks if not row['passed']]}")
    work = _normalise(frame, columns)
    work["__month__"] = work[columns.decision].dt.strftime("%Y-%m")
    units = (net_unit, gross_unit, cost_unit)
    tails: list[dict[str, Any]] = []
    membership: list[pd.DataFrame] = []
    attribution: list[pd.DataFrame] = []
    calibration: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    books: dict[tuple[str, str, float], pd.DataFrame] = {}
    group_columns = ["target_arm"] + (["support_stage"] if "support_stage" in work else [])
    for key, group in work.groupby(group_columns, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        prefix = dict(zip(group_columns, key, strict=True))
        if entry_threshold is not None:
            entered = group.loc[pd.to_numeric(group[columns.score], errors="coerce").gt(float(entry_threshold))].copy()
            if minimum_clean_probability is not None:
                if clean_probability_column not in group:
                    raise AlignmentEvaluationError("minimum clean-probability gate requested without a clean-probability field")
                entered = entered.loc[pd.to_numeric(entered[clean_probability_column], errors="coerce").ge(float(minimum_clean_probability))]
            if len(entered):
                threshold_rows.append({**prefix, "selection_scope": "candidate_score_threshold_no_portfolio_constraints", "entry_threshold": float(entry_threshold), "minimum_clean_probability": minimum_clean_probability, "population_rows": len(group), **_summary(entered, columns, net_unit=net_unit, gross_unit=gross_unit, cost_unit=cost_unit)})
            else:
                threshold_rows.append({**prefix, "selection_scope": "candidate_score_threshold_no_portfolio_constraints", "entry_threshold": float(entry_threshold), "minimum_clean_probability": minimum_clean_probability, "population_rows": len(group), "rows": 0})
        for fraction in fractions:
            book = stable_global_top_k(group, columns.score, float(fraction)).copy()
            score = pd.to_numeric(book[columns.score], errors="coerce")
            book = book.assign(global_rank=score.rank(method="first", ascending=False).astype(int), global_tail_fraction=float(fraction), global_tail_member=True)
            selected_ids = tuple(book.sort_values("candidate_id", kind="stable").candidate_id.astype(str))
            books[(*key, float(fraction))] = book
            tails.append({**prefix, "selection_scope": "one_pooled_global_post_score_top_k", "top_fraction": float(fraction), "population_rows": len(group), "selected_candidate_id_sha256": _id_hash(selected_ids), **_summary(book, columns, net_unit=net_unit, gross_unit=gross_unit, cost_unit=cost_unit)})
            membership.append(book.loc[:, ["candidate_id", *group_columns, "global_rank", "global_tail_fraction", "global_tail_member"]])
            for dimension in ("side_name", "__month__", *context_columns):
                table = _attribution(book, dimension=dimension, columns=columns, units=units)
                if not table.empty:
                    attribution.append(table.assign(**prefix, top_fraction=float(fraction)))
        # Score-to-exact-net calibration is diagnostic only, but using ten
        # global score bins avoids a side-local calibration substitution.
        rank = pd.to_numeric(group[columns.score], errors="coerce").rank(method="first", pct=True)
        decile = np.ceil(rank * 10.0).clip(1, 10).astype(int)
        for bucket, local in group.assign(score_decile=decile).groupby("score_decile", sort=True):
            calibration.append({**prefix, "score_decile": int(bucket), "rows": len(local), "mean_predicted_score_bps": float(_bps(local, columns.score, score_unit).mean()), "mean_realised_exact_net_bps": float(_bps(local, columns.net, net_unit).mean())})
    bootstrap: list[dict[str, Any]] = []
    if bootstrap_baseline_arm is not None:
        for key, book in books.items():
            arm, *rest = key
            if arm == bootstrap_baseline_arm or float(rest[-1]) != .10:
                continue
            baseline_key = (bootstrap_baseline_arm, *rest)
            baseline = books.get(baseline_key)
            if baseline is not None:
                # The controlled runner already carries the protocol timestamp
                # as ``__ts__``.  Do not rename the decision column into a
                # duplicate key when passing the frozen books to the generic
                # bootstrap helper.
                bootstrap.append({"target_arm": arm, "baseline_arm": bootstrap_baseline_arm, "top_fraction": .10, **paired_day_block_bootstrap(baseline, book, net_column=columns.net, net_unit=net_unit)})
    tails_frame = pd.DataFrame(tails)
    attribution_frame = pd.concat(attribution, ignore_index=True) if attribution else pd.DataFrame()
    acceptance: list[dict[str, Any]] = []
    for key, group in tails_frame.groupby(group_columns, observed=True, sort=True):
        top = group.set_index("top_fraction")
        top10 = top.loc[.10] if .10 in top.index else None
        values = key if isinstance(key, tuple) else (key,)
        latest_side = attribution_frame.loc[
            attribution_frame.dimension.eq("__month__")
            & attribution_frame.top_fraction.eq(.10)
            & np.logical_and.reduce([attribution_frame[name].eq(value) for name, value in zip(group_columns, values, strict=True)])
        ] if not attribution_frame.empty else pd.DataFrame()
        latest_ok = bool(not latest_side.empty and float(latest_side.loc[latest_side.dimension_value.eq(latest_side.dimension_value.max()), "net_bps_per_trade"].iloc[0]) >= 0.0)
        acceptance.append({**dict(zip(group_columns, values, strict=True)), "pooled_top10_positive": bool(top10 is not None and float(top10.net_bps_per_trade) > 0), "top1_top5_severe_reversal": bool((.01 in top.index and .05 in top.index) and min(float(top.loc[.01, "net_bps_per_trade"]), float(top.loc[.05, "net_bps_per_trade"])) < 0), "latest_selected_month_nonnegative": latest_ok, "promotion_eligible": False})
    return {"global_tail_metrics": tails_frame, "global_tail_membership": pd.concat(membership, ignore_index=True), "global_tail_attribution": attribution_frame, "score_calibration": pd.DataFrame(calibration), "threshold_entry_metrics": pd.DataFrame(threshold_rows), "paired_day_block_bootstrap": pd.DataFrame(bootstrap), "acceptance_checks": pd.DataFrame(acceptance), "correctness_checks": pd.DataFrame(checks)}
