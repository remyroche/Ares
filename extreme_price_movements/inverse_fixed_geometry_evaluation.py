"""Read-only global-book evaluation for the 2022 inverse-PI research lineage.

This module deliberately consumes *already produced* scored ablation rows.  It
does not fit a model, tune a threshold, remap a score, or replay a policy.  Its
job is narrower: make all fixed-geometry arms answer the same economic
question with one deterministic pooled-global book per calendar month.

The Jan--Jul 2022 inverse-PI lineage is not OOF execution evidence.  The
returned contract says so explicitly and callers must not convert this output
into a promotion decision.  Mapping is accepted only when it is either an
explicit identity/no-fit arm or is attested as train-only with every mapped row
using outcomes resolved strictly before its decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


INVERSE_FIXED_GEOMETRY_EVALUATION_SCHEMA = "inverse_fixed_geometry_evaluation_v1"
DEFAULT_TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
IDENTITY_COLUMNS = ("candidate_id",)


@dataclass(frozen=True)
class InverseFixedGeometryEvaluation:
    """Tables and an explicit non-promotable evidence contract."""

    monthly: pd.DataFrame
    summary: pd.DataFrame
    monotonicity: pd.DataFrame
    selections: pd.DataFrame
    comparisons: pd.DataFrame
    contract: dict[str, Any]


def _as_utc(values: pd.Series, *, name: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{name!r} contains invalid timestamps")
    return converted


def _finite_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column!r} must be finite")
    return values.astype(float)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    lhs = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=float)
    rhs = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=float)
    if np.std(lhs) <= 1e-12 or np.std(rhs) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _calibration(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(target) & np.isfinite(score)
    if int(valid.sum()) < 2 or np.std(score[valid]) <= 1e-12:
        return {
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_mae": float("nan"),
        }
    slope, intercept = np.polyfit(score[valid], target[valid], deg=1)
    return {
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_mae": float(np.mean(np.abs(target[valid] - score[valid]))),
    }


def _sign_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target > 0.0
    return (
        float(roc_auc_score(positive.astype(np.int8), score))
        if len(np.unique(positive)) == 2
        else float("nan")
    )


def _validate_mapping(
    work: pd.DataFrame,
    *,
    decision_col: str,
    mapping_status_col: str,
    mapping_resolution_col: str,
) -> None:
    """Fail closed if a mapped score could have used its row's outcome.

    Identity arms need no fitted mapping and therefore use the explicit
    ``identity_no_fit`` status with a null resolution timestamp.  Any other
    accepted status is a train-only mapping and needs a row-level maximum
    training-label resolution.  Causal mappings must resolve strictly before
    the decision; explicitly non-causal leave-block-out research mappings may
    use complementary future blocks but remain non-promotable.
    """

    status = work[mapping_status_col].astype(str).str.strip().str.lower()
    allowed = {
        "identity_no_fit",
        "causal_train_only",
        "out_of_block_train_only_noncausal",
    }
    unsupported = sorted(set(status).difference(allowed))
    if unsupported:
        raise ValueError(
            "mapping status must be identity_no_fit, causal_train_only, or "
            "out_of_block_train_only_noncausal; got "
            + ", ".join(unsupported)
        )
    causal = status.eq("causal_train_only")
    fitted = status.ne("identity_no_fit")
    resolution = pd.to_datetime(work[mapping_resolution_col], utc=True, errors="coerce")
    if resolution.loc[fitted].isna().any():
        raise ValueError("fitted mapping rows require mapping max-resolution timestamps")
    if (resolution.loc[causal] >= work.loc[causal, decision_col]).any():
        raise ValueError(
            "causal mapping must use labels resolved strictly before each decision"
        )
    if resolution.loc[~fitted].notna().any():
        raise ValueError("identity_no_fit rows must not claim a fitted mapping timestamp")


def validate_inverse_fixed_geometry_inputs(
    frame: pd.DataFrame,
    *,
    arm_col: str = "arm",
    decision_col: str = "execution_decision_utc",
    score_col: str = "mapped_score",
    net_col: str = "execution_net_ev_12h",
    gross_col: str = "execution_gross_ev_12h",
    side_col: str = "side_name",
    eligibility_col: str | None = "eligible",
    mapping_status_col: str = "mapping_status",
    mapping_resolution_col: str = "mapping_max_label_resolution_utc",
    require_matched_arms: bool = True,
) -> pd.DataFrame:
    """Validate and normalize scored fixed-geometry ablation rows.

    A matched identity population is the default because a feature/architecture
    comparison that changes the candidate pool is not an ablation of ranking.
    """

    required = [
        arm_col, *IDENTITY_COLUMNS, decision_col, score_col, net_col, gross_col,
        side_col, mapping_status_col, mapping_resolution_col,
    ]
    if eligibility_col is not None:
        required.append(eligibility_col)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("inverse fixed-geometry evaluation missing columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("inverse fixed-geometry evaluation requires at least one row")

    work = frame.copy()
    work[decision_col] = _as_utc(work[decision_col], name=decision_col)
    for column in (score_col, net_col, gross_col):
        work[column] = _finite_column(work, column)
    work[arm_col] = work[arm_col].astype(str)
    if work[arm_col].str.strip().eq("").any():
        raise ValueError("arm names must be non-empty")
    work[side_col] = work[side_col].astype(str).str.strip().str.lower()
    if set(work[side_col]).difference({"long", "short"}):
        raise ValueError("side values must be long or short")
    if work.duplicated([arm_col, *IDENTITY_COLUMNS]).any():
        raise ValueError("each arm must have one row per candidate identity")
    if eligibility_col is None:
        work["__eligible__"] = True
    else:
        eligible = work[eligibility_col]
        if eligible.isna().any():
            raise ValueError("eligibility must be explicit for every row")
        work["__eligible__"] = eligible.astype(bool)
    _validate_mapping(
        work,
        decision_col=decision_col,
        mapping_status_col=mapping_status_col,
        mapping_resolution_col=mapping_resolution_col,
    )
    work["evaluation_month"] = work[decision_col].dt.tz_localize(None).dt.to_period("M").astype(str)

    if require_matched_arms:
        reference: pd.DataFrame | None = None
        comparison_columns = [decision_col, net_col, gross_col, side_col, "__eligible__"]
        for arm, local in work.groupby(arm_col, sort=True):
            comparable = local.set_index(list(IDENTITY_COLUMNS)).loc[:, comparison_columns].sort_index()
            if reference is None:
                reference = comparable
            elif not comparable.equals(reference):
                raise ValueError(
                    f"arm {arm!r} does not have the same candidate identities, outcomes, "
                    "sides, decisions, and eligibility as the reference arm"
                )
    return work.sort_values([arm_col, decision_col, *IDENTITY_COLUMNS], kind="mergesort").reset_index(drop=True)


def _select_global_book(
    local: pd.DataFrame,
    *,
    score_col: str,
    fraction: float,
) -> pd.DataFrame:
    eligible = local.loc[local["__eligible__"]].copy()
    if eligible.empty:
        return eligible.assign(rank_in_month=pd.Series(dtype="int64"))
    size = max(1, int(np.ceil(len(eligible) * float(fraction))))
    ordered = eligible.sort_values(
        [score_col, *IDENTITY_COLUMNS],
        ascending=[False, *([True] * len(IDENTITY_COLUMNS))],
        kind="mergesort",
    ).head(size).copy()
    ordered["rank_in_month"] = np.arange(1, len(ordered) + 1, dtype=np.int64)
    return ordered


def _monotonicity_rows(
    local: pd.DataFrame,
    *,
    arm: str,
    month: str,
    score_col: str,
    net_col: str,
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = local.sort_values([score_col, *IDENTITY_COLUMNS], kind="mergesort").copy()
    ranks = ranked[score_col].rank(method="first", pct=True)
    bins = np.minimum(n_bins - 1, np.floor(ranks.to_numpy(dtype=float) * n_bins).astype(int))
    ranked["__score_bin__"] = bins
    rows: list[dict[str, Any]] = []
    means: list[float] = []
    for bin_id, group in ranked.groupby("__score_bin__", sort=True):
        mean_net = float(group[net_col].mean())
        means.append(mean_net)
        rows.append({
            "arm": arm, "evaluation_month": month, "score_bin": int(bin_id) + 1,
            "bins_requested": int(n_bins), "rows": int(len(group)),
            "mean_score": float(group[score_col].mean()),
            "mean_net_bps": mean_net * 10_000.0,
            "positive_net_rate": float(group[net_col].gt(0.0).mean()),
        })
    bin_axis = np.arange(1, len(means) + 1, dtype=float)
    summary = {
        "monotonicity_bins": int(len(means)),
        "monotonicity_rank_ic": _spearman(bin_axis, np.asarray(means, dtype=float)),
        "monotonicity_top_minus_bottom_bps": (
            (means[-1] - means[0]) * 10_000.0 if len(means) >= 2 else float("nan")
        ),
        "monotonicity_non_decreasing": bool(
            len(means) >= 2 and np.all(np.diff(np.asarray(means, dtype=float)) >= -1e-12)
        ),
    }
    return rows, summary


def _month_metrics(
    candidate_local: pd.DataFrame,
    eligible_local: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    arm: str,
    month: str,
    fraction: float,
    score_col: str,
    net_col: str,
    gross_col: str,
    monotonicity: dict[str, Any],
) -> dict[str, Any]:
    score = candidate_local[score_col].to_numpy(dtype=float)
    net = candidate_local[net_col].to_numpy(dtype=float)
    gross = candidate_local[gross_col].to_numpy(dtype=float)
    selected_net = selected[net_col].to_numpy(dtype=float)
    selected_gross = selected[gross_col].to_numpy(dtype=float)
    cutoff = float(selected[score_col].iloc[-1]) if len(selected) else float("nan")
    cutoff_ties = int(np.isclose(score, cutoff, rtol=0.0, atol=0.0).sum()) if len(selected) else 0
    return {
        "arm": arm,
        "evaluation_month": month,
        "top_fraction": float(fraction),
        "selection_scope": "one_pooled_global_top_k_per_evaluation_month_after_declared_train_only_mapping",
        "candidate_rows": int(len(candidate_local)),
        "eligible_rows": int(len(eligible_local)),
        "book_depth": int(len(selected)),
        "selection_rate": float(len(selected) / len(eligible_local)) if len(eligible_local) else float("nan"),
        "cutoff_score": cutoff,
        "cutoff_tie_rows": cutoff_ties,
        "mean_net_bps": float(selected_net.mean() * 10_000.0) if len(selected) else float("nan"),
        "sum_net": float(selected_net.sum()) if len(selected) else 0.0,
        "mean_gross_bps": float(selected_gross.mean() * 10_000.0) if len(selected) else float("nan"),
        "sum_gross": float(selected_gross.sum()) if len(selected) else 0.0,
        "positive_net_rate": float((selected_net > 0.0).mean()) if len(selected) else float("nan"),
        "long_share": float(selected["__side__"].eq("long").mean()) if len(selected) else float("nan"),
        "short_share": float(selected["__side__"].eq("short").mean()) if len(selected) else float("nan"),
        "rank_ic_net": _spearman(score, net),
        "rank_ic_gross": _spearman(score, gross),
        "sign_auc_net": _sign_auc(net, score),
        **_calibration(net, score),
        **monotonicity,
    }


def evaluate_inverse_fixed_geometry_arms(
    frame: pd.DataFrame,
    *,
    arm_col: str = "arm",
    decision_col: str = "execution_decision_utc",
    score_col: str = "mapped_score",
    net_col: str = "execution_net_ev_12h",
    gross_col: str = "execution_gross_ev_12h",
    side_col: str = "side_name",
    eligibility_col: str | None = "eligible",
    mapping_status_col: str = "mapping_status",
    mapping_resolution_col: str = "mapping_max_label_resolution_utc",
    evaluation_month_col: str | None = None,
    top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS,
    expected_months: Iterable[str] | None = None,
    baseline_arm: str | None = None,
    arm_metadata_cols: Sequence[str] = (),
    n_monotonicity_bins: int = 10,
    require_matched_arms: bool = True,
) -> InverseFixedGeometryEvaluation:
    """Evaluate all fixed-geometry arms on monthly pooled-global books.

    The input scores must already incorporate their causal/train-only map.  A
    month is selected as one cross-side book; there is no timestamp quota,
    side quota, replacement or implicit backfill.  The returned summary is a
    trade-weighted aggregation of those monthly books, not a second global
    re-ranking across months.
    """

    fractions = tuple(float(value) for value in top_fractions)
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("top_fractions must be non-empty values in (0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("top_fractions must be unique")
    if n_monotonicity_bins < 2:
        raise ValueError("n_monotonicity_bins must be at least two")
    if len(set(arm_metadata_cols)) != len(tuple(arm_metadata_cols)):
        raise ValueError("arm_metadata_cols must be unique")
    work = validate_inverse_fixed_geometry_inputs(
        frame, arm_col=arm_col, decision_col=decision_col, score_col=score_col,
        net_col=net_col, gross_col=gross_col, side_col=side_col,
        eligibility_col=eligibility_col, mapping_status_col=mapping_status_col,
        mapping_resolution_col=mapping_resolution_col,
        require_matched_arms=require_matched_arms,
    )
    if evaluation_month_col is not None:
        if evaluation_month_col not in work:
            raise ValueError(
                f"evaluation month column {evaluation_month_col!r} is missing"
            )
        parsed_month = pd.to_datetime(
            work[evaluation_month_col], utc=True, errors="coerce"
        )
        if parsed_month.isna().any():
            textual = work[evaluation_month_col].astype(str)
            if not textual.str.fullmatch(r"\d{4}-\d{2}").all():
                raise ValueError("evaluation month column is not parseable")
            work["evaluation_month"] = textual
        else:
            work["evaluation_month"] = parsed_month.dt.strftime("%Y-%m")
    work["__side__"] = work[side_col]
    for column in arm_metadata_cols:
        if column not in work:
            raise ValueError(f"arm metadata column {column!r} is missing")
        cardinality = work.groupby(arm_col, sort=False)[column].nunique(dropna=False)
        if cardinality.gt(1).any():
            bad = ", ".join(cardinality.loc[cardinality.gt(1)].index.astype(str))
            raise ValueError(f"arm metadata {column!r} must be constant within an arm; bad arms: {bad}")
    observed_months = sorted(work["evaluation_month"].unique())
    required_months = sorted(set(expected_months)) if expected_months is not None else observed_months
    missing_months = sorted(set(required_months).difference(observed_months))
    monthly_rows: list[dict[str, Any]] = []
    monotonicity_rows: list[dict[str, Any]] = []
    selection_parts: list[pd.DataFrame] = []
    for arm, arm_frame in work.groupby(arm_col, sort=True):
        for month, local_all in arm_frame.groupby("evaluation_month", sort=True):
            local = local_all.loc[local_all["__eligible__"]].copy()
            mono_rows, mono = _monotonicity_rows(
                local, arm=str(arm), month=str(month), score_col=score_col,
                net_col=net_col, n_bins=n_monotonicity_bins,
            )
            monotonicity_rows.extend(mono_rows)
            for fraction in fractions:
                selected = _select_global_book(local, score_col=score_col, fraction=fraction)
                selection = selected.loc[:, ["candidate_id", decision_col, arm_col, "evaluation_month", score_col, net_col, gross_col, "__side__", "rank_in_month"]].copy()
                selection["top_fraction"] = fraction
                selection_parts.append(selection)
                monthly_rows.append(_month_metrics(
                    local_all, local, selected, arm=str(arm), month=str(month), fraction=fraction,
                    score_col=score_col, net_col=net_col, gross_col=gross_col,
                    monotonicity=mono,
                ))
    monthly = pd.DataFrame(monthly_rows)
    monotonicity = pd.DataFrame(monotonicity_rows)
    selections = pd.concat(selection_parts, ignore_index=True) if selection_parts else pd.DataFrame()
    summaries: list[dict[str, Any]] = []
    for (arm, fraction), local in monthly.groupby(["arm", "top_fraction"], sort=True):
        selected = selections.loc[(selections[arm_col].eq(arm)) & (selections["top_fraction"].eq(fraction))]
        selected_net = selected[net_col].to_numpy(dtype=float)
        selected_gross = selected[gross_col].to_numpy(dtype=float)
        arm_rows = work.loc[work[arm_col].eq(arm)]
        summaries.append({
            "arm": arm,
            "top_fraction": float(fraction),
            "selection_scope": "one_pooled_global_top_k_per_evaluation_month_after_declared_train_only_mapping",
            "evaluation_months_expected": int(len(required_months)),
            "evaluation_months_observed": int(local["evaluation_month"].nunique()),
            "evaluation_months_missing": ",".join(missing_months),
            "month_coverage_complete": bool(not missing_months),
            "month_coverage": float(local["evaluation_month"].nunique() / len(required_months)) if required_months else float("nan"),
            "candidate_rows": int(len(arm_rows)),
            "eligible_rows": int(arm_rows["__eligible__"].sum()),
            "book_depth": int(len(selected)),
            "mean_net_bps": float(selected_net.mean() * 10_000.0) if len(selected) else float("nan"),
            "mean_gross_bps": float(selected_gross.mean() * 10_000.0) if len(selected) else float("nan"),
            "positive_net_rate": float((selected_net > 0.0).mean()) if len(selected) else float("nan"),
            "long_share": float(selected["__side__"].eq("long").mean()) if len(selected) else float("nan"),
            "short_share": float(selected["__side__"].eq("short").mean()) if len(selected) else float("nan"),
            "worst_month_mean_net_bps": float(local["mean_net_bps"].min()),
            "worst_month": str(local.loc[local["mean_net_bps"].idxmin(), "evaluation_month"]),
            "mean_month_rank_ic_net": float(local["rank_ic_net"].mean()),
            "mean_month_sign_auc_net": float(local["sign_auc_net"].mean()),
            "mean_month_calibration_mae": float(local["calibration_mae"].mean()),
            "mean_month_monotonicity_rank_ic": float(local["monotonicity_rank_ic"].mean()),
            "global_rank_ic_net": _spearman(
                arm_rows[score_col].to_numpy(dtype=float), arm_rows[net_col].to_numpy(dtype=float)
            ),
            "global_sign_auc_net": _sign_auc(
                arm_rows[net_col].to_numpy(dtype=float), arm_rows[score_col].to_numpy(dtype=float)
            ),
            **{f"global_{key}": value for key, value in _calibration(
                arm_rows[net_col].to_numpy(dtype=float), arm_rows[score_col].to_numpy(dtype=float)
            ).items()},
            **{
                column: arm_rows[column].iloc[0]
                for column in arm_metadata_cols
            },
        })
    summary = pd.DataFrame(summaries)
    if baseline_arm is None:
        baseline_arm = str(sorted(work[arm_col].unique())[0])
    if baseline_arm not in set(summary["arm"]):
        raise ValueError(f"baseline arm {baseline_arm!r} is absent")
    comparisons: list[dict[str, Any]] = []
    for fraction in fractions:
        baseline = summary.loc[(summary["arm"].eq(baseline_arm)) & (summary["top_fraction"].eq(fraction))]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, arm in summary.loc[summary["top_fraction"].eq(fraction)].iterrows():
            comparisons.append({
                "baseline_arm": baseline_arm, "arm": arm["arm"], "top_fraction": float(fraction),
                "delta_mean_net_bps_vs_baseline": float(arm["mean_net_bps"] - base["mean_net_bps"]),
                "delta_mean_gross_bps_vs_baseline": float(arm["mean_gross_bps"] - base["mean_gross_bps"]),
                "delta_worst_month_net_bps_vs_baseline": float(arm["worst_month_mean_net_bps"] - base["worst_month_mean_net_bps"]),
                "delta_rank_ic_vs_baseline": float(arm["global_rank_ic_net"] - base["global_rank_ic_net"]),
                "matched_candidate_population": bool(require_matched_arms),
                "research_only_non_oof": True,
                "promotion_eligible": False,
                **{column: arm[column] for column in arm_metadata_cols},
            })
    contract = {
        "schema": INVERSE_FIXED_GEOMETRY_EVALUATION_SCHEMA,
        "population": "jan_jul_2022_inverse_pi_market_grid_causal_features_v1",
        "evidence_status": "research_only_non_oof_not_promotable",
        "promotion_eligible": False,
        "mapping": (
            "identity_no_fit, causal_train_only, or explicitly non-causal "
            "out_of_block_train_only_noncausal; all fitted rows attest their "
            "maximum training-label resolution and only causal rows require it "
            "strictly before the decision"
        ),
        "selection": "one pooled global top-k per evaluation month after mapping; no timestamp or side quota, replacement, or backfill",
        "fractions": list(fractions),
        "month_coverage_expected": required_months,
        "month_coverage_missing": missing_months,
        "evaluation_month_source": (
            evaluation_month_col or f"UTC calendar month of {decision_col}"
        ),
        "fixed_geometry": "scores and outcomes are supplied; evaluator fits no model, changes no geometry, and performs no portfolio replay",
    }
    return InverseFixedGeometryEvaluation(
        monthly=monthly, summary=summary, monotonicity=monotonicity,
        selections=selections, comparisons=pd.DataFrame(comparisons), contract=contract,
    )
