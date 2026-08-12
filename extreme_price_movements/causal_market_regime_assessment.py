"""Label-free selection diagnostics for causal soft market-regime candidates.

This module deliberately sits *beside* the regime generator.  It never builds
features, fits a state model, or consumes realised returns.  Instead it scores
already causal candidate timelines on properties that should transfer across
eras: input coverage, support, posterior confidence, state persistence,
transition behaviour, and invariant distribution drift between chronological
evaluation folds.  State coordinates are not compared across folds; soft
occupancies are sorted first, so the portability test remains valid after a
fresh GMM fit permutes its components.

The companion runner uses a small deterministic grid of K/stickiness fits on
bounded proxy rows.  This module is also usable on a materialised OOF timeline
from a more expensive run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.causal_market_regime_systems import FORBIDDEN_INPUT_TOKENS, PHASE_NAMES


SCHEMA = "causal_market_regime_parameter_assessment_v1"
_EPS = 1e-12
# Do not ban generic ``score``/``return`` substrings: the causal generator
# itself exposes a market-direction score and a regime feature may legitimately
# be based on a signed return.  The assessment receives only generated state
# outputs, and must reject realised execution outcomes specifically.
_ASSESSMENT_FORBIDDEN_TOKENS = tuple(
    dict.fromkeys((*FORBIDDEN_INPUT_TOKENS, "exact_net", "gross", "residual", "profit"))
)


class RegimeAssessmentError(ValueError):
    """Raised when a candidate timeline cannot support a causal assessment."""


@dataclass(frozen=True)
class RegimeAssessmentColumns:
    timestamp: str = "source_utc"
    candidate: str = "candidate_id"
    fold: str = "assessment_fold_id"
    train_end: str = "regime_train_end_utc"
    input_coverage: str = "input_coverage"
    state_age_hours: str = "state_age_hours"
    switch_probability: str = "state_switch_probability"
    window: str = "assessment_window_id"
    seed: str = "assessment_seed"


@dataclass(frozen=True)
class RegimeAssessmentConfig:
    """Predeclared structural gates and a compact, label-free score.

    The targets reflect a 6--12 hour trading horizon.  They are not fitted to
    any outcome: a candidate with an extremely low switch rate is not rewarded
    simply for being inert, and a candidate with a very high switch rate is
    not rewarded merely for producing many transition labels.
    """

    minimum_coverage: float = 0.80
    minimum_soft_occupancy: float = 0.02
    minimum_median_dwell_hours: float = 4.0
    maximum_switch_rate: float = 0.25
    target_switch_rate: float = 0.08
    minimum_mean_confidence: float = 0.55
    portability_weight: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise RegimeAssessmentError("minimum_coverage must lie in (0, 1]")
        if not 0.0 < self.minimum_soft_occupancy < 1.0:
            raise RegimeAssessmentError("minimum_soft_occupancy must lie in (0, 1)")
        if self.minimum_median_dwell_hours <= 0.0:
            raise RegimeAssessmentError("minimum_median_dwell_hours must be positive")
        if not 0.0 < self.maximum_switch_rate <= 1.0:
            raise RegimeAssessmentError("maximum_switch_rate must lie in (0, 1]")
        if not 0.0 <= self.target_switch_rate <= 1.0:
            raise RegimeAssessmentError("target_switch_rate must lie in [0, 1]")
        if not 0.0 < self.minimum_mean_confidence <= 1.0:
            raise RegimeAssessmentError("minimum_mean_confidence must lie in (0, 1]")
        if not 0.0 <= self.portability_weight <= 0.75:
            raise RegimeAssessmentError("portability_weight must lie in [0, .75]")


@dataclass(frozen=True)
class RegimeAssessmentResult:
    """Tables intentionally suitable for parquet/CSV reporting."""

    fold_diagnostics: pd.DataFrame
    portability_diagnostics: pd.DataFrame
    candidate_summary: pd.DataFrame


def candidate_grid(
    k_values: Sequence[int] = (3, 4, 5, 6),
    stickiness_values: Sequence[float] = (0.0, 0.35, 0.60, 0.80),
) -> tuple[tuple[int, float], ...]:
    """Return a bounded deterministic K/stickiness grid, rejecting unsafe values."""

    ks = tuple(sorted({int(value) for value in k_values}))
    rhos = tuple(sorted({float(value) for value in stickiness_values}))
    if not ks or any(value < 2 or value > 8 for value in ks):
        raise RegimeAssessmentError("K candidates must be unique integers in [2, 8]")
    if not rhos or any(not 0.0 <= value < 1.0 for value in rhos):
        raise RegimeAssessmentError("stickiness candidates must lie in [0, 1)")
    return tuple((k, rho) for k in ks for rho in rhos)


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise RegimeAssessmentError(f"timeline lacks required column {column!r}")
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise RegimeAssessmentError(f"{column!r} must contain valid UTC timestamps")
    return values


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise RegimeAssessmentError(f"timeline lacks required column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RegimeAssessmentError(f"{column!r} must be finite")
    return values


def _column(prefix: str, suffix: str) -> str:
    return f"{prefix}__{suffix}"


def regime_output_columns(frame: pd.DataFrame, *, prefix: str) -> tuple[list[str], dict[str, str]]:
    """Resolve one generator output prefix without relying on state IDs semantically."""

    probability = [
        column for column in frame.columns
        if column.startswith(_column(prefix, "state_p_"))
        # The sequential K screen concatenates candidates with different K.
        # A wider candidate therefore contributes an all-null padded column
        # to a narrower candidate's local assessment frame; it is not a
        # state posterior and must not invalidate that candidate's simplex.
        and frame[column].notna().any()
    ]
    probability.sort(key=lambda column: int(column.rsplit("_", 1)[-1]))
    if len(probability) < 2:
        raise RegimeAssessmentError(f"{prefix!r} has fewer than two state posterior fields")
    required = {
        "input_coverage": _column(prefix, "input_coverage"),
        "state_age_hours": _column(prefix, "state_age_hours"),
        "switch_probability": _column(prefix, "state_switch_probability"),
    }
    optional = {
        "entropy": _column(prefix, "entropy"),
        "top2_margin": _column(prefix, "top2_margin"),
        "ood_distance_percentile": _column(prefix, "ood_distance_percentile"),
    }
    missing = [column for column in required.values() if column not in frame]
    if missing:
        raise RegimeAssessmentError(f"{prefix!r} lacks state diagnostics: {missing}")
    return probability, {**required, **{key: value for key, value in optional.items() if value in frame}}


def regime_feature_bundle(frame: pd.DataFrame, *, prefix: str) -> tuple[str, ...]:
    """Return the complete per-view bundle for a later *joint* ablation.

    This preserves all posterior coordinates as a bundle.  The selector never
    promotes one coordinate in isolation, because frozen component numbering
    is only meaningful within a fitted fold.
    """

    probability, state = regime_output_columns(frame, prefix=prefix)
    ordered = [*probability]
    for name in ("entropy", "top2_margin", "state_age_hours", "switch_probability"):
        if name in state:
            ordered.append(state[name])
    return tuple(ordered)


def _phase_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [_column(prefix, f"phase_p_{phase}") for phase in PHASE_NAMES]
    return columns if all(column in frame for column in columns) else []


def _stable_top_indices(score: np.ndarray, identity: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(float(fraction) * len(score))))
    order = np.lexsort((identity.astype(str), -np.asarray(score, dtype=np.float64)))
    return order[:count]


def _run_metrics(
    posterior: np.ndarray,
    timestamps: pd.Series,
) -> tuple[float, float]:
    """Return median dwell hours and hard-label switch rate for one ordered fold."""

    if len(posterior) < 2:
        return 0.0, 0.0
    labels = np.argmax(posterior, axis=1)
    ts = timestamps.astype("int64").to_numpy()
    change = labels[1:] != labels[:-1]
    contiguous = np.diff(ts) <= int(pd.Timedelta(hours=2).value)
    valid = contiguous
    switch_rate = float(np.mean(change[valid])) if valid.any() else 0.0
    lengths: list[float] = []
    start = 0
    for position in range(1, len(labels)):
        if not contiguous[position - 1] or labels[position] != labels[position - 1]:
            elapsed = max(1.0, (ts[position - 1] - ts[start]) / float(pd.Timedelta(hours=1).value) + 1.0)
            lengths.append(elapsed)
            start = position
    elapsed = max(1.0, (ts[-1] - ts[start]) / float(pd.Timedelta(hours=1).value) + 1.0)
    lengths.append(elapsed)
    return float(np.median(lengths)), switch_rate


def _phase_summary(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {f"phase_share_{name}": np.nan for name in PHASE_NAMES} | {
            "phase_simplex_passed": np.nan, "transition_active_share": np.nan,
        }
    simplex = bool(
        np.isfinite(values).all()
        and (values >= -1e-6).all()
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-4)
    )
    output = {f"phase_share_{name}": float(values[:, position].mean()) for position, name in enumerate(PHASE_NAMES)}
    output["phase_simplex_passed"] = bool(simplex)
    output["transition_active_share"] = float(np.mean(values[:, 1] + values[:, 2] >= 0.50))
    return output


def _fold_diagnostic(
    frame: pd.DataFrame,
    *,
    prefix: str,
    columns: RegimeAssessmentColumns,
    config: RegimeAssessmentConfig,
) -> dict[str, Any]:
    probability_columns, state = regime_output_columns(frame, prefix=prefix)
    timestamp = _utc(frame, columns.timestamp)
    train_end = _utc(frame, columns.train_end)
    if not (train_end < timestamp).all():
        raise RegimeAssessmentError("candidate timeline contains state outputs without strictly prior train end")
    posterior = frame.loc[:, probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(posterior).all() or (posterior < -1e-7).any() or not np.allclose(posterior.sum(axis=1), 1.0, atol=1e-4):
        raise RegimeAssessmentError("state posteriors must be finite non-negative simplexes")
    order = np.argsort(timestamp.astype("int64").to_numpy(), kind="stable")
    posterior = posterior[order]
    timestamp = timestamp.iloc[order].reset_index(drop=True)
    coverage = _numeric(frame, state["input_coverage"])[order]
    age = _numeric(frame, state["state_age_hours"])[order]
    switch = _numeric(frame, state["switch_probability"])[order]
    if not np.all((coverage >= -1e-6) & (coverage <= 1.0 + 1e-6)):
        raise RegimeAssessmentError("input coverage must lie in [0, 1]")
    if not np.all((switch >= -1e-6) & (switch <= 1.0 + 1e-6)):
        raise RegimeAssessmentError("switch probability must lie in [0, 1]")
    soft_occupancy = posterior.mean(axis=0)
    median_dwell, hard_switch = _run_metrics(posterior, timestamp)
    sorted_probability = np.sort(posterior, axis=1)
    phase_columns = _phase_columns(frame, prefix)
    phase = frame.loc[:, phase_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)[order] if phase_columns else np.empty((0, 4))
    phase_metrics = _phase_summary(phase)
    confidence = posterior.max(axis=1)
    safe_posterior = np.clip(posterior, _EPS, 1.0)
    entropy = -np.sum(np.where(posterior > _EPS, posterior * np.log(safe_posterior), 0.0), axis=1)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": str(frame[columns.candidate].iloc[0]),
        "assessment_fold_id": str(frame[columns.fold].iloc[0]),
        "prefix": prefix,
        "rows": int(len(frame)),
        "evaluation_start_utc": timestamp.min(),
        "evaluation_end_utc": timestamp.max(),
        "train_end_max_utc": train_end.max(),
        "causality_passed": True,
        "state_count": int(posterior.shape[1]),
        "mean_input_coverage": float(coverage.mean()),
        "low_coverage_share": float(np.mean(coverage < config.minimum_coverage)),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_top2_margin": float((sorted_probability[:, -1] - sorted_probability[:, -2]).mean()),
        "minimum_soft_occupancy": float(soft_occupancy.min()),
        "maximum_soft_occupancy": float(soft_occupancy.max()),
        "soft_occupancy_sorted_json": ",".join(f"{value:.8f}" for value in np.sort(soft_occupancy)),
        "median_dwell_hours": median_dwell,
        "hard_switch_rate": hard_switch,
        "mean_switch_probability": float(switch.mean()),
        "median_state_age_hours": float(np.median(age)),
        "coverage_gate_passed": bool(coverage.mean() >= config.minimum_coverage),
        "support_gate_passed": bool(soft_occupancy.min() >= config.minimum_soft_occupancy),
        "confidence_gate_passed": bool(confidence.mean() >= config.minimum_mean_confidence),
        "persistence_gate_passed": bool(median_dwell >= config.minimum_median_dwell_hours and hard_switch <= config.maximum_switch_rate),
        **phase_metrics,
    }
    for field in ("system", "candidate_k", "candidate_stickiness", columns.window, columns.seed):
        if field in frame:
            unique = frame[field].dropna().unique()
            if len(unique) != 1:
                raise RegimeAssessmentError(f"{field!r} must be constant within an assessment fold")
            record[field] = unique[0]
    for field in ("centroid_min_separation", "centroid_mean_separation"):
        if field in frame:
            values = _numeric(frame, field)
            if not np.allclose(values, values[0], atol=1e-7):
                raise RegimeAssessmentError(f"{field!r} must be frozen within an assessment fold")
            record[field] = float(values[0])
    if "ood_distance_percentile" in state:
        ood = _numeric(frame, state["ood_distance_percentile"])[order]
        record["ood_p95"] = float(np.quantile(ood, 0.95))
    return record


def _decode_occupancy(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(",") if item], dtype=np.float64)


def _portability_rows(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, local in folds.groupby("candidate_id", observed=True, sort=True):
        # Average two seed replicas first.  This avoids treating seed changes
        # as an era shift and leaves the cross-window comparison coordinate
        # invariant.
        window = "assessment_window_id" if "assessment_window_id" in local else "assessment_fold_id"
        numeric = [column for column in local if pd.api.types.is_numeric_dtype(local[column])]
        grouped = local.groupby(window, observed=True, sort=True)
        records: list[dict[str, Any]] = []
        for window_id, part in grouped:
            row = {column: (float(part[column].mean()) if column in numeric else part[column].iloc[0]) for column in part.columns}
            row[window] = window_id
            row["soft_occupancy_sorted_json"] = part["soft_occupancy_sorted_json"].iloc[0]
            records.append(row)
        local = pd.DataFrame(records).sort_values("evaluation_start_utc", kind="stable").reset_index(drop=True)
        if len(local) < 2:
            rows.append({"schema": SCHEMA, "candidate_id": candidate, "comparisons": 0, "portability_score": np.nan})
            continue
        drifts: list[dict[str, float]] = []
        for position in range(1, len(local)):
            left, right = local.iloc[position - 1], local.iloc[position]
            left_occupancy, right_occupancy = _decode_occupancy(left.soft_occupancy_sorted_json), _decode_occupancy(right.soft_occupancy_sorted_json)
            if left_occupancy.shape != right_occupancy.shape:
                occupancy_drift = 1.0
            else:
                occupancy_drift = float(0.5 * np.abs(left_occupancy - right_occupancy).sum())
            entropy_scale = max(float(np.log(max(int(left.state_count), int(right.state_count)))), 1.0)
            drifts.append({
                "occupancy_drift": occupancy_drift,
                "confidence_drift": abs(float(left.mean_confidence) - float(right.mean_confidence)),
                "entropy_drift": abs(float(left.mean_entropy) - float(right.mean_entropy)) / entropy_scale,
                "switch_drift": abs(float(left.hard_switch_rate) - float(right.hard_switch_rate)),
                "coverage_drift": abs(float(left.mean_input_coverage) - float(right.mean_input_coverage)),
                "transition_drift": abs(float(left.transition_active_share) - float(right.transition_active_share)) if np.isfinite(left.transition_active_share) and np.isfinite(right.transition_active_share) else 0.0,
            })
        aggregate = {key: float(np.mean([item[key] for item in drifts])) for key in drifts[0]}
        penalty = float(np.mean(list(aggregate.values())))
        rows.append({"schema": SCHEMA, "candidate_id": candidate, "comparisons": len(drifts), **aggregate, "portability_score": float(np.clip(1.0 - penalty, 0.0, 1.0))})
    return pd.DataFrame(rows)


def _posterior_seed_stability(
    timeline: pd.DataFrame,
    *,
    prefix: str,
    columns: RegimeAssessmentColumns,
) -> pd.DataFrame:
    """Compare seed replicas using sorted posterior vectors at equal times.

    Sorting makes this a shape-stability metric rather than an invalid claim
    that GMM component ``0`` has the same semantic identity across refits.
    """

    if columns.seed not in timeline or columns.window not in timeline:
        return pd.DataFrame(columns=["candidate_id", "posterior_seed_stability", "seed_pair_count"])
    probability, _state = regime_output_columns(timeline, prefix=prefix)
    work = timeline.loc[:, [columns.candidate, columns.window, columns.seed, columns.timestamp, *probability]].copy()
    work[columns.timestamp] = _utc(work, columns.timestamp)
    ordered = np.sort(work.loc[:, probability].to_numpy(np.float64), axis=1)
    for number in range(ordered.shape[1]):
        work[f"__sorted_p_{number}"] = ordered[:, number]
    values: list[dict[str, Any]] = []
    for (candidate, window), local in work.groupby([columns.candidate, columns.window], observed=True, sort=True):
        seeds = sorted(local[columns.seed].drop_duplicates().tolist())
        distance: list[float] = []
        for left_index in range(len(seeds)):
            for right_index in range(left_index + 1, len(seeds)):
                left = local.loc[local[columns.seed].eq(seeds[left_index])].set_index(columns.timestamp)
                right = local.loc[local[columns.seed].eq(seeds[right_index])].set_index(columns.timestamp)
                joined = left.join(right, how="inner", lsuffix="_l", rsuffix="_r")
                fields = [f"__sorted_p_{number}" for number in range(ordered.shape[1])]
                if not joined.empty:
                    diff = np.abs(joined[[f"{field}_l" for field in fields]].to_numpy(float) - joined[[f"{field}_r" for field in fields]].to_numpy(float))
                    distance.append(float(0.5 * diff.sum(axis=1).mean()))
        values.append({"candidate_id": candidate, "assessment_window_id": window, "posterior_seed_stability": float(1.0 - np.mean(distance)) if distance else np.nan, "seed_pair_count": len(distance)})
    result = pd.DataFrame(values)
    if result.empty:
        return pd.DataFrame(columns=["candidate_id", "posterior_seed_stability", "seed_pair_count"])
    return result.groupby("candidate_id", observed=True).agg(posterior_seed_stability=("posterior_seed_stability", "mean"), seed_pair_count=("seed_pair_count", "sum")).reset_index()


def _pareto_mask(summary: pd.DataFrame) -> np.ndarray:
    """Maximise all structural dimensions except hard switch rate."""

    metrics = summary.loc[:, ["mean_input_coverage", "minimum_soft_occupancy", "mean_confidence", "median_dwell_hours", "portability_score"]].to_numpy(float)
    switch = -summary.hard_switch_rate.to_numpy(float)[:, None]
    value = np.column_stack((metrics, switch))
    output = np.ones(len(value), dtype=bool)
    for left in range(len(value)):
        for right in range(len(value)):
            if left != right and np.all(value[right] >= value[left]) and np.any(value[right] > value[left]):
                output[left] = False
                break
    return output


def select_regime_parameter_recommendation(summary: pd.DataFrame) -> dict[str, Any]:
    """Apply a Pareto gate then a one-SE, lower-complexity recommendation.

    A selected configuration must already pass all structural gates.  Among
    Pareto candidates within one standard error of the top score, prefer lower
    K and then lower stickiness.  This deliberately prevents a full-grid
    maximum from winning by an immaterial proxy difference.
    """

    required = {"candidate_id", "parameter_gate_passed", "structural_score", "structural_score_se", "pareto_efficient"}
    if not required.issubset(summary):
        raise RegimeAssessmentError("summary lacks the fields needed for one-SE recommendation")
    eligible = summary.loc[summary.parameter_gate_passed & summary.pareto_efficient].copy()
    if eligible.empty:
        return {"recommended_candidate_id": None, "status": "NO_CANDIDATE_CLEARS_STRUCTURAL_GATES"}
    best = eligible.sort_values(["structural_score", "candidate_id"], ascending=[False, True], kind="stable").iloc[0]
    threshold = float(best.structural_score - best.structural_score_se)
    near = eligible.loc[eligible.structural_score.ge(threshold)].copy()
    for field, fallback in (("candidate_k", np.inf), ("candidate_stickiness", np.inf)):
        near[field] = pd.to_numeric(near.get(field, fallback), errors="coerce").fillna(fallback)
    chosen = near.sort_values(["candidate_k", "candidate_stickiness", "structural_score", "candidate_id"], ascending=[True, True, False, True], kind="stable").iloc[0]
    return {
        "recommended_candidate_id": str(chosen.candidate_id),
        "status": "PARETO_ONE_SE_LABEL_FREE_RECOMMENDATION",
        "best_structural_score": float(best.structural_score),
        "one_se_threshold": threshold,
        "candidate_k": int(chosen.candidate_k) if np.isfinite(chosen.candidate_k) else None,
        "candidate_stickiness": float(chosen.candidate_stickiness) if np.isfinite(chosen.candidate_stickiness) else None,
    }


def assess_regime_candidate_timeline(
    timeline: pd.DataFrame,
    *,
    prefix: str,
    columns: RegimeAssessmentColumns = RegimeAssessmentColumns(),
    config: RegimeAssessmentConfig = RegimeAssessmentConfig(),
) -> RegimeAssessmentResult:
    """Assess a label-free causal candidate timeline.

    ``timeline`` may contain several candidate configurations and chronological
    folds.  It is rejected if it contains outcome-like fields, if any train end
    is current/future relative to its state output, or if posterior/phase
    simplexes are invalid.  The summary score is structural only and is meant
    to decide which small candidate set deserves a later supervised ablation;
    it is not an execution-performance score.
    """

    required = (columns.timestamp, columns.candidate, columns.fold, columns.train_end)
    missing = [column for column in required if column not in timeline]
    if missing:
        raise RegimeAssessmentError(f"timeline lacks identity/provenance columns: {missing}")
    forbidden = [column for column in timeline.columns if any(token in str(column).lower() for token in _ASSESSMENT_FORBIDDEN_TOKENS)]
    if forbidden:
        raise RegimeAssessmentError(f"assessment timeline must be label-free; forbidden fields include {forbidden[:8]}")
    if timeline.empty:
        raise RegimeAssessmentError("candidate timeline cannot be empty")
    if timeline.loc[:, [columns.candidate, columns.fold]].isna().any().any():
        raise RegimeAssessmentError("candidate and fold identifiers must be non-null")
    group_columns = [columns.candidate, columns.fold]
    if columns.seed in timeline:
        group_columns.append(columns.seed)
    rows = [
        _fold_diagnostic(local, prefix=prefix, columns=columns, config=config)
        for _key, local in timeline.groupby(group_columns, observed=True, sort=True)
    ]
    folds = pd.DataFrame(rows)
    portability = _portability_rows(folds)
    aggregate = folds.groupby("candidate_id", observed=True).agg(
        folds=("assessment_fold_id", "nunique"),
        rows=("rows", "sum"),
        causality_passed=("causality_passed", "all"),
        coverage_passed=("coverage_gate_passed", "all"),
        support_passed=("support_gate_passed", "all"),
        confidence_passed=("confidence_gate_passed", "all"),
        persistence_passed=("persistence_gate_passed", "all"),
        mean_input_coverage=("mean_input_coverage", "mean"),
        minimum_soft_occupancy=("minimum_soft_occupancy", "min"),
        mean_confidence=("mean_confidence", "mean"),
        median_dwell_hours=("median_dwell_hours", "median"),
        hard_switch_rate=("hard_switch_rate", "mean"),
        mean_entropy=("mean_entropy", "mean"),
        transition_active_share=("transition_active_share", "mean"),
    ).reset_index()
    summary = aggregate.merge(portability, on="candidate_id", how="left", validate="one_to_one")
    portability_value = summary.portability_score.fillna(0.0).to_numpy(float)
    switch_quality = np.clip(1.0 - np.abs(summary.hard_switch_rate.to_numpy(float) - config.target_switch_rate) / max(config.maximum_switch_rate, _EPS), 0.0, 1.0)
    base_score = (
        0.23 * np.clip(summary.mean_input_coverage.to_numpy(float) / config.minimum_coverage, 0.0, 1.0)
        + 0.20 * np.clip(summary.minimum_soft_occupancy.to_numpy(float) / config.minimum_soft_occupancy, 0.0, 1.0)
        + 0.17 * np.clip(summary.mean_confidence.to_numpy(float) / config.minimum_mean_confidence, 0.0, 1.0)
        + 0.17 * np.clip(summary.median_dwell_hours.to_numpy(float) / config.minimum_median_dwell_hours, 0.0, 1.0)
        + 0.08 * switch_quality
        + 0.15 * np.clip(summary.transition_active_share.fillna(0.0).to_numpy(float) / 0.10, 0.0, 1.0)
    )
    # Preserve score scale when portability is unavailable (a one-fold proxy),
    # but it cannot pass the portability gate needed for a multi-fold choice.
    summary["structural_score"] = (1.0 - config.portability_weight) * base_score + config.portability_weight * portability_value
    summary["portability_passed"] = summary.comparisons.ge(1) & summary.portability_score.ge(0.70)
    summary["parameter_gate_passed"] = (
        summary.causality_passed & summary.coverage_passed & summary.support_passed
        & summary.confidence_passed & summary.persistence_passed & summary.portability_passed
    )
    summary["selection_basis"] = "label_free_structural_and_cross_fold_invariant_portability"
    summary["schema"] = SCHEMA
    # Fold-level standard errors feed a conservative one-SE choice.  The
    # portability term stays candidate-level so an isolated weak fold cannot
    # be hidden by a good average.
    fold_score = (
        0.23 * np.clip(folds.mean_input_coverage.to_numpy(float) / config.minimum_coverage, 0.0, 1.0)
        + 0.20 * np.clip(folds.minimum_soft_occupancy.to_numpy(float) / config.minimum_soft_occupancy, 0.0, 1.0)
        + 0.17 * np.clip(folds.mean_confidence.to_numpy(float) / config.minimum_mean_confidence, 0.0, 1.0)
        + 0.17 * np.clip(folds.median_dwell_hours.to_numpy(float) / config.minimum_median_dwell_hours, 0.0, 1.0)
        + 0.08 * np.clip(1.0 - np.abs(folds.hard_switch_rate.to_numpy(float) - config.target_switch_rate) / max(config.maximum_switch_rate, _EPS), 0.0, 1.0)
        + 0.15 * np.clip(folds.transition_active_share.fillna(0.0).to_numpy(float) / 0.10, 0.0, 1.0)
    )
    folds["fold_structural_score"] = fold_score
    fold_se = folds.groupby("candidate_id", observed=True).fold_structural_score.agg(lambda values: float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0).rename("structural_score_se").reset_index()
    summary = summary.merge(fold_se, on="candidate_id", how="left", validate="one_to_one")
    seed_stability = _posterior_seed_stability(timeline, prefix=prefix, columns=columns)
    summary = summary.merge(seed_stability, on="candidate_id", how="left", validate="one_to_one")
    for field in ("system", "candidate_k", "candidate_stickiness"):
        if field in folds:
            values = folds.groupby("candidate_id", observed=True)[field].first().reset_index()
            summary = summary.merge(values, on="candidate_id", how="left", validate="one_to_one")
    summary["pareto_efficient"] = _pareto_mask(summary)
    summary = summary.sort_values(["parameter_gate_passed", "structural_score", "candidate_id"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    return RegimeAssessmentResult(folds, portability, summary)


__all__ = [
    "SCHEMA", "RegimeAssessmentError", "RegimeAssessmentColumns", "RegimeAssessmentConfig",
    "RegimeAssessmentResult", "candidate_grid", "regime_output_columns",
    "regime_feature_bundle", "assess_regime_candidate_timeline",
    "select_regime_parameter_recommendation",
]
