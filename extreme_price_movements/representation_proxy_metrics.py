"""Outcome-free diagnostics for cached latent representations and diagonal GMMs.

The functions in this module intentionally accept only representations, GMM
parameters, and observable strata.  They do not accept labels, scores, or
outcomes, so they can be used while selecting a frozen AE/GMM representation
without turning an unsupervised screen into an economic evaluation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture

try:  # Keep two concurrent GMM fits from oversubscribing BLAS threads.
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional runtime guard.
    threadpool_limits = None

try:  # scipy is a sklearn dependency, but retain a small deterministic fallback.
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


DEFAULT_GMM_PANEL: tuple[tuple[int, str, float], ...] = (
    (4, "diag", 0.003),
    (6, "diag", 0.003),
    (8, "diag", 0.003),
)
EPSILON = 1.0e-12


@dataclass(frozen=True)
class GmmPanelSpec:
    n_components: int
    covariance_type: str = "diag"
    reg_covar: float = 0.003


@dataclass(frozen=True)
class ComponentAlignment:
    metric: str
    reference_to_candidate: tuple[int, ...]
    candidate_to_reference: tuple[int, ...]
    matched_cost: float
    mean_cost: float
    max_cost: float
    cost_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class OodCalibration:
    q95: float
    q99: float
    reference_count: int
    tier_names: tuple[str, str, str] = ("in_distribution", "elevated", "extreme")


@dataclass(frozen=True)
class GmmPanelFit:
    spec: GmmPanelSpec
    seed: int
    state: dict[str, Any]


@dataclass(frozen=True)
class ProxyMetricsResult:
    """A JSON-ready container for a representation-only proxy screen."""

    entropy: dict[str, Any]
    occupancy: list[dict[str, Any]]
    perturbation: dict[str, Any]
    ood: dict[str, Any]
    component_alignment: dict[str, Any] | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def json_safe(value: Any) -> Any:
    """Convert numpy and dataclass values to deterministic JSON-safe primitives."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_matrix(values: Any, *, name: str, dtype: np.dtype = np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and component")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _as_vector(values: Any, *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def normalized_posteriors(posteriors: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Return finite row-normalized posterior probabilities in the requested dtype."""
    raw = _as_matrix(posteriors, name="posteriors", dtype=np.float64)
    if np.any(raw < 0.0):
        raise ValueError("posteriors cannot contain negative values")
    totals = raw.sum(axis=1, keepdims=True)
    if np.any(totals <= EPSILON):
        raise ValueError("every posterior row must have positive mass")
    return (raw / totals).astype(dtype, copy=False)


def normalized_entropy(posteriors: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Shannon entropy normalized to [0, 1] for a posterior matrix."""
    probs = normalized_posteriors(posteriors, dtype=np.float64)
    if probs.shape[1] == 1:
        return np.zeros(len(probs), dtype=dtype)
    entropy = -np.sum(np.where(probs > 0.0, probs * np.log(probs), 0.0), axis=1)
    return (entropy / np.log(float(probs.shape[1]))).astype(dtype, copy=False)


def entropy_distribution_diagnostics(
    posteriors: Any,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Distribution diagnostics for normalized posterior entropy, without outcomes."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    entropy = normalized_entropy(posteriors, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    counts, _ = np.histogram(np.clip(entropy, 0.0, 1.0), bins=edges)
    q = np.quantile(entropy, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return json_safe(
        {
            "n_rows": int(len(entropy)),
            "n_components": int(np.asarray(posteriors).shape[1]),
            "mean": float(np.mean(entropy)),
            "std": float(np.std(entropy)),
            "min": float(np.min(entropy)),
            "max": float(np.max(entropy)),
            "quantiles": {
                key: float(value)
                for key, value in zip(("q01", "q05", "q25", "q50", "q75", "q95", "q99"), q)
            },
            "histogram_edges": edges,
            "histogram_counts": counts.astype(int),
            "histogram_density": counts / max(int(len(entropy)), 1),
        }
    )


def stratified_entropy_distribution_diagnostics(
    posteriors: Any,
    *,
    strata: Mapping[str, Any] | None = None,
    bins: int = 10,
) -> dict[str, Any]:
    """Entropy distributions overall and by observable symbol/calendar/regime strata."""
    probs = normalized_posteriors(posteriors)
    reports: list[dict[str, Any]] = []
    for stratum, level, mask in _strata_groups(strata, len(probs)):
        report = entropy_distribution_diagnostics(probs[mask], bins=bins)
        report.update({"stratum": stratum, "level": level})
        reports.append(report)
    return {"overall": reports[0], "strata": reports[1:]}


def _mean_pairwise_l1(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    pairs = [np.mean(np.abs(left - right)) for left, right in combinations(values, 2)]
    return float(np.mean(pairs)) if pairs else 0.0


def _strata_groups(strata: Mapping[str, Any] | None, n_rows: int) -> list[tuple[str, str, np.ndarray]]:
    groups: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(n_rows, dtype=bool))]
    if not strata:
        return groups
    for name, labels in strata.items():
        values = np.asarray(labels, dtype=object).reshape(-1)
        if len(values) != n_rows:
            raise ValueError(f"stratum {name!r} must have {n_rows} rows")
        canonical = np.asarray(["<missing>" if value is None else str(value) for value in values], dtype=object)
        for level in sorted(np.unique(canonical).tolist()):
            groups.append((str(name), str(level), canonical == level))
    return groups


def occupancy_excess_instability(
    posterior_runs: Mapping[str, Any],
    *,
    expected_weights: Any | Mapping[str, Any] | None = None,
    strata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Measure occupancy and occupancy-minus-weight instability across cached runs.

    ``posterior_runs`` can represent seeds, bootstrap/resample fits, or input
    perturbations.  Strata are observable labels such as symbol, calendar bin,
    or regime; their values are never fitted or ranked using outcomes.
    """
    if not posterior_runs:
        raise ValueError("posterior_runs cannot be empty")
    keyed_runs = {str(name): values for name, values in posterior_runs.items()}
    if len(keyed_runs) != len(posterior_runs):
        raise ValueError("posterior run names must remain unique after string conversion")
    names = sorted(keyed_runs)
    matrices = [normalized_posteriors(keyed_runs[name], dtype=np.float64) for name in names]
    n_rows, n_components = matrices[0].shape
    if any(matrix.shape != (n_rows, n_components) for matrix in matrices[1:]):
        raise ValueError("all posterior runs must share shape (rows, components)")
    expected_by_run: list[np.ndarray] = []
    for name in names:
        raw = expected_weights.get(name) if isinstance(expected_weights, Mapping) else expected_weights
        if raw is None:
            raw = np.full(n_components, 1.0 / n_components, dtype=np.float64)
        weights = _as_vector(raw, name="expected_weights", length=n_components)
        if np.any(weights < 0.0) or weights.sum() <= EPSILON:
            raise ValueError("expected_weights must have positive total non-negative mass")
        expected_by_run.append(weights / weights.sum())
    rows: list[dict[str, Any]] = []
    for stratum, level, mask in _strata_groups(strata, n_rows):
        occupancy = np.asarray([matrix[mask].mean(axis=0) for matrix in matrices], dtype=np.float64)
        expected = np.asarray(expected_by_run, dtype=np.float64)
        excess = occupancy - expected
        rows.append(
            json_safe(
                {
                    "stratum": stratum,
                    "level": level,
                    "n_rows": int(mask.sum()),
                    "n_runs": int(len(names)),
                    "run_names": names,
                    "mean_occupancy": occupancy.mean(axis=0),
                    "occupancy_std": occupancy.std(axis=0),
                    "occupancy_range": occupancy.max(axis=0) - occupancy.min(axis=0),
                    "mean_excess": excess.mean(axis=0),
                    "excess_std": excess.std(axis=0),
                    "mean_pairwise_occupancy_l1": _mean_pairwise_l1(occupancy),
                    "mean_pairwise_excess_l1": _mean_pairwise_l1(excess),
                    "max_component_occupancy_std": float(occupancy.std(axis=0).max()),
                    "max_component_abs_mean_excess": float(np.abs(excess.mean(axis=0)).max()),
                }
            )
        )
    return rows


def _rank01(values: Any) -> np.ndarray:
    values = _as_vector(values, name="values")
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # Average tied ranks so consistency is unaffected by arbitrary tie order.
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0.0) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks / float(len(values) - 1)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def perturbation_consistency(
    latent_reference: Any,
    latent_perturbed: Any,
    posterior_reference: Any,
    posterior_perturbed: Any,
    ood_reference: Any,
    ood_perturbed: Any,
) -> dict[str, Any]:
    """Compare two cached transforms on identical rows using no outcome signal."""
    ref_latent = _as_matrix(latent_reference, name="latent_reference")
    alt_latent = _as_matrix(latent_perturbed, name="latent_perturbed")
    ref_post = normalized_posteriors(posterior_reference, dtype=np.float64)
    alt_post = normalized_posteriors(posterior_perturbed, dtype=np.float64)
    ref_ood = _as_vector(ood_reference, name="ood_reference")
    alt_ood = _as_vector(ood_perturbed, name="ood_perturbed")
    if ref_latent.shape != alt_latent.shape or ref_post.shape != alt_post.shape:
        raise ValueError("reference and perturbed latent/posterior shapes must match")
    if len(ref_latent) != len(ref_post) or len(ref_ood) != len(ref_post) or len(alt_ood) != len(ref_post):
        raise ValueError("all perturbation inputs must have equal row counts")
    norms = np.linalg.norm(ref_latent, axis=1) * np.linalg.norm(alt_latent, axis=1)
    cosine = np.divide(np.sum(ref_latent * alt_latent, axis=1), norms, out=np.ones(len(norms)), where=norms > EPSILON)
    posterior_tv = 0.5 * np.abs(ref_post - alt_post).sum(axis=1)
    entropy_ref = normalized_entropy(ref_post, dtype=np.float64)
    entropy_alt = normalized_entropy(alt_post, dtype=np.float64)
    return json_safe(
        {
            "n_rows": int(len(ref_post)),
            "latent_cosine_mean": float(np.mean(cosine)),
            "latent_cosine_p05": float(np.quantile(cosine, 0.05)),
            "latent_rmse": float(np.sqrt(np.mean((ref_latent - alt_latent) ** 2))),
            "posterior_tv_mean": float(np.mean(posterior_tv)),
            "posterior_tv_p95": float(np.quantile(posterior_tv, 0.95)),
            "hard_assignment_agreement": float(np.mean(np.argmax(ref_post, axis=1) == np.argmax(alt_post, axis=1))),
            "entropy_mae": float(np.mean(np.abs(entropy_ref - entropy_alt))),
            "entropy_correlation": _correlation(entropy_ref, entropy_alt),
            "ood_mae": float(np.mean(np.abs(ref_ood - alt_ood))),
            "ood_rank_correlation": _correlation(_rank01(ref_ood), _rank01(alt_ood)),
        }
    )


def fit_ood_calibration(reference_ood: Any) -> OodCalibration:
    """Fit the fixed 95/99 percentile thresholds for three observable OOD tiers."""
    values = _as_vector(reference_ood, name="reference_ood")
    q95, q99 = np.quantile(values, [0.95, 0.99])
    return OodCalibration(q95=float(q95), q99=float(max(q99, q95)), reference_count=int(len(values)))


def apply_ood_calibration(ood_scores: Any, calibration: OodCalibration) -> dict[str, Any]:
    """Apply a frozen three-tier OOD calibration and return tier frequencies."""
    values = _as_vector(ood_scores, name="ood_scores")
    tiers = np.where(values > calibration.q99, 2, np.where(values > calibration.q95, 1, 0)).astype(np.int8)
    counts = np.bincount(tiers, minlength=3)
    return json_safe(
        {
            "thresholds": {"q95": calibration.q95, "q99": calibration.q99},
            "tier_names": calibration.tier_names,
            "tiers": tiers,
            "counts": counts,
            "fractions": counts / max(len(values), 1),
        }
    )


def _probability_greater(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a corrupted score exceeds a clean score, with tied half-credit."""
    ordered = np.sort(negative)
    below = np.searchsorted(ordered, positive, side="left")
    through = np.searchsorted(ordered, positive, side="right")
    return float(np.mean((below + 0.5 * (through - below)) / len(ordered)))


def _aligned_rank_consistency(
    candidate: np.ndarray,
    clean: np.ndarray | None,
) -> dict[str, Any]:
    if clean is None:
        return {"aligned": False, "n_rows": 0, "rank_correlation": None}
    if len(candidate) != len(clean):
        return {
            "aligned": False,
            "n_rows": 0,
            "rank_correlation": None,
            "reason": "row_count_mismatch",
        }
    return {
        "aligned": True,
        "n_rows": int(len(candidate)),
        "rank_correlation": _correlation(_rank01(clean), _rank01(candidate)),
        "rank_mae": float(np.mean(np.abs(_rank01(clean) - _rank01(candidate)))),
    }


def evaluate_ood_proxy(
    clean_untouched_later_scores: Any,
    mild_synthetic_scores: Any,
    structural_synthetic_scores: Any,
    natural_temporal_scores: Any,
    *,
    calibration_scores: Any | None = None,
    aligned_clean_scores: Mapping[str, Any | None] | None = None,
) -> dict[str, Any]:
    """Evaluate an OOD score proxy without labels or model outcomes.

    ``calibration_scores`` should normally be an earlier frozen clean reference
    set.  The untouched-later clean rows are then used only to estimate false
    positives.  Synthetic sets can share rows with clean inputs; rank
    consistency is reported only when alignment is explicit or row counts
    match.  Natural temporal rows may be unaligned and are never forced into a
    rank comparison.
    """
    scores = {
        "clean_untouched_later": _as_vector(
            clean_untouched_later_scores, name="clean_untouched_later_scores"
        ),
        "mild_synthetic": _as_vector(mild_synthetic_scores, name="mild_synthetic_scores"),
        "structural_synthetic": _as_vector(
            structural_synthetic_scores, name="structural_synthetic_scores"
        ),
        "natural_temporal": _as_vector(natural_temporal_scores, name="natural_temporal_scores"),
    }
    calibration_source = (
        scores["clean_untouched_later"]
        if calibration_scores is None
        else _as_vector(calibration_scores, name="calibration_scores")
    )
    calibration = fit_ood_calibration(calibration_source)
    clean = scores["clean_untouched_later"]
    applied = {name: apply_ood_calibration(values, calibration) for name, values in scores.items()}
    severity_names = tuple(scores)
    means = np.asarray([np.mean(scores[name]) for name in severity_names], dtype=np.float64)
    medians = np.asarray([np.median(scores[name]) for name in severity_names], dtype=np.float64)
    deltas = np.diff(means)
    median_deltas = np.diff(medians)
    rank_consistency: dict[str, Any] = {}
    for name, values in scores.items():
        if name == "clean_untouched_later":
            rank_consistency[name] = {
                "aligned": True,
                "n_rows": int(len(clean)),
                "rank_correlation": 1.0,
                "rank_mae": 0.0,
            }
            continue
        explicit_clean = None
        if aligned_clean_scores is not None and name in aligned_clean_scores:
            raw_clean = aligned_clean_scores[name]
            explicit_clean = (
                None
                if raw_clean is None
                else _as_vector(raw_clean, name=f"aligned_clean_scores[{name!r}]")
            )
        comparison_clean = explicit_clean if aligned_clean_scores is not None else clean
        rank_consistency[name] = _aligned_rank_consistency(values, comparison_clean)
    separation: dict[str, Any] = {}
    clean_q95 = float(np.quantile(clean, 0.95))
    for name in severity_names[1:]:
        values = scores[name]
        separation[name] = {
            "mean_gap_vs_clean": float(np.mean(values) - np.mean(clean)),
            "median_gap_vs_clean": float(np.median(values) - np.median(clean)),
            "probability_corrupted_gt_clean": _probability_greater(values, clean),
            "fraction_above_clean_q95": float(np.mean(values > clean_q95)),
            "elevated_or_extreme_rate": float(np.mean(np.asarray(applied[name]["tiers"]) >= 1)),
        }
    clean_tiers = np.asarray(applied["clean_untouched_later"]["tiers"])
    return json_safe(
        {
            "outcome_free": True,
            "calibration": json_safe(calibration),
            "calibration_source": (
                "clean_untouched_later" if calibration_scores is None else "external_frozen_clean_reference"
            ),
            "score_sets": {
                name: {
                    "n_rows": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "tier_fractions": applied[name]["fractions"],
                }
                for name, values in scores.items()
            },
            "severity_order": list(severity_names),
            "monotonicity": {
                "mean_non_decreasing": bool(np.all(deltas >= 0.0)),
                "median_non_decreasing": bool(np.all(median_deltas >= 0.0)),
                "mean_deltas": deltas,
                "median_deltas": median_deltas,
                "mean_violation_count": int(np.sum(deltas < 0.0)),
                "median_violation_count": int(np.sum(median_deltas < 0.0)),
            },
            "rank_consistency": rank_consistency,
            "clean_corrupted_separation": separation,
            "untouched_later_false_positive_rate": {
                "elevated_or_extreme": float(np.mean(clean_tiers >= 1)),
                "extreme": float(np.mean(clean_tiers == 2)),
                "n_rows": int(len(clean_tiers)),
            },
        }
    )


def _pairwise_component_cost(
    reference_means: np.ndarray,
    reference_covariances: np.ndarray,
    candidate_means: np.ndarray,
    candidate_covariances: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    if reference_means.shape != candidate_means.shape:
        raise ValueError("reference and candidate GMM means must share shape")
    if reference_covariances.shape != reference_means.shape or candidate_covariances.shape != candidate_means.shape:
        raise ValueError("diagonal covariance arrays must match the means")
    ref_var = np.maximum(reference_covariances, EPSILON)
    cand_var = np.maximum(candidate_covariances, EPSILON)
    delta2 = (reference_means[:, None, :] - candidate_means[None, :, :]) ** 2
    if metric == "bhattacharyya":
        average_var = 0.5 * (ref_var[:, None, :] + cand_var[None, :, :])
        quadratic = 0.125 * np.sum(delta2 / average_var, axis=2)
        determinant = 0.5 * np.sum(
            np.log(average_var) - 0.5 * (np.log(ref_var[:, None, :]) + np.log(cand_var[None, :, :])), axis=2
        )
        return quadratic + determinant
    if metric in {"wasserstein", "wasserstein2", "w2"}:
        return np.sum(delta2 + (np.sqrt(ref_var[:, None, :]) - np.sqrt(cand_var[None, :, :])) ** 2, axis=2)
    raise ValueError("metric must be 'bhattacharyya' or 'wasserstein2'")


def align_diagonal_gmm_components(
    reference: Mapping[str, Any] | GaussianMixture,
    candidate: Mapping[str, Any] | GaussianMixture,
    *,
    metric: str = "bhattacharyya",
) -> ComponentAlignment:
    """Find a one-to-one component mapping using a diagonal Gaussian distance."""
    ref = diagonal_gmm_state(reference)
    alt = diagonal_gmm_state(candidate)
    cost = _pairwise_component_cost(ref["means"], ref["covariances"], alt["means"], alt["covariances"], metric=metric)
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
    else:  # pragma: no cover - scipy is normally installed with sklearn.
        remaining = set(range(cost.shape[1]))
        rows = np.arange(cost.shape[0])
        chosen: list[int] = []
        for row in rows:
            col = min(remaining, key=lambda candidate_col: (cost[row, candidate_col], candidate_col))
            chosen.append(int(col))
            remaining.remove(int(col))
        cols = np.asarray(chosen, dtype=int)
    reference_to_candidate = np.empty(len(rows), dtype=int)
    reference_to_candidate[rows] = cols
    candidate_to_reference = np.empty(len(cols), dtype=int)
    candidate_to_reference[cols] = rows
    matched = cost[rows, cols]
    return ComponentAlignment(
        metric="wasserstein2" if metric in {"wasserstein", "wasserstein2", "w2"} else "bhattacharyya",
        reference_to_candidate=tuple(int(item) for item in reference_to_candidate),
        candidate_to_reference=tuple(int(item) for item in candidate_to_reference),
        matched_cost=float(np.sum(matched)),
        mean_cost=float(np.mean(matched)),
        max_cost=float(np.max(matched)),
        cost_matrix=tuple(tuple(float(value) for value in row) for row in cost),
    )


def reorder_posteriors_to_reference(posteriors: Any, alignment: ComponentAlignment, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Place candidate posterior columns in reference component order."""
    probs = normalized_posteriors(posteriors, dtype=dtype)
    order = np.asarray(alignment.reference_to_candidate, dtype=int)
    if probs.shape[1] != len(order):
        raise ValueError("alignment component count does not match posterior columns")
    return probs[:, order]


def diagonal_gmm_state(model_or_state: Mapping[str, Any] | GaussianMixture) -> dict[str, np.ndarray]:
    """Extract compact diagonal GMM arrays from sklearn or a serialization mapping."""
    if isinstance(model_or_state, GaussianMixture):
        if model_or_state.covariance_type != "diag":
            raise ValueError("only sklearn diagonal GaussianMixture models are supported")
        means = model_or_state.means_
        covariances = model_or_state.covariances_
        weights = model_or_state.weights_
    else:
        means = model_or_state.get("means", model_or_state.get("gmm_means"))
        covariances = model_or_state.get("covariances", model_or_state.get("gmm_covariances"))
        weights = model_or_state.get("weights", model_or_state.get("gmm_weights"))
    means_array = _as_matrix(means, name="gmm means")
    covariance_array = _as_matrix(covariances, name="gmm covariances")
    weights_array = _as_vector(weights, name="gmm weights", length=len(means_array))
    if covariance_array.shape != means_array.shape or np.any(covariance_array <= 0.0):
        raise ValueError("diagonal GMM covariances must be positive and match means")
    if np.any(weights_array < 0.0) or weights_array.sum() <= EPSILON:
        raise ValueError("GMM weights must have positive total non-negative mass")
    return {"means": means_array, "covariances": covariance_array, "weights": weights_array / weights_array.sum()}


def diagonal_gmm_statistics(
    latent: Any,
    model_or_state: Mapping[str, Any] | GaussianMixture,
    *,
    dtype: np.dtype = np.float32,
    batch_rows: int = 50_000,
) -> dict[str, np.ndarray]:
    """Vectorized posterior, density, and Mahalanobis computations for cached latents."""
    z = _as_matrix(latent, name="latent", dtype=np.float64)
    state = diagonal_gmm_state(model_or_state)
    means, variances, weights = state["means"], state["covariances"], state["weights"]
    if z.shape[1] != means.shape[1]:
        raise ValueError("latent dimension does not match GMM means")
    log_det = np.sum(np.log(variances), axis=1)
    rows, components = len(z), len(weights)
    posteriors = np.empty((rows, components), dtype=dtype)
    mahalanobis = np.empty((rows, components), dtype=dtype)
    log_density = np.empty(rows, dtype=dtype)
    # Keeping the chunk below 50k avoids a 100k x K x latent_dim float64
    # temporary while preserving the exact row-wise density calculation.
    for start in range(0, rows, max(1, int(batch_rows))):
        stop = min(rows, start + max(1, int(batch_rows)))
        block = z[start:stop]
        delta = block[:, None, :] - means[None, :, :]
        mahal_sq = np.sum(delta * delta / variances[None, :, :], axis=2)
        log_joint = np.log(np.maximum(weights, EPSILON))[None, :] - 0.5 * (
            mahal_sq + log_det[None, :] + block.shape[1] * np.log(2.0 * np.pi)
        )
        log_norm = np.max(log_joint, axis=1, keepdims=True)
        block_density = log_norm[:, 0] + np.log(
            np.exp(log_joint - log_norm).sum(axis=1)
        )
        posteriors[start:stop] = np.exp(log_joint - block_density[:, None]).astype(
            dtype, copy=False
        )
        mahalanobis[start:stop] = np.sqrt(np.maximum(mahal_sq, 0.0)).astype(
            dtype, copy=False
        )
        log_density[start:stop] = block_density.astype(dtype, copy=False)
    return {
        "posteriors": posteriors,
        "mahalanobis": mahalanobis,
        "log_density": log_density,
        "ood_score": (-log_density).astype(dtype, copy=False),
    }


def bounded_diagonal_gmm_overlap(
    model_or_state: Mapping[str, Any] | GaussianMixture,
    *,
    metric: str = "bhattacharyya",
    scale: float = 1.0,
) -> dict[str, Any]:
    """Return a weighted, bounded pairwise diagonal-GMM overlap diagnostic."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    state = diagonal_gmm_state(model_or_state)
    means, variances, weights = (
        state["means"],
        state["covariances"],
        state["weights"],
    )
    if len(means) < 2:
        return {
            "metric": "wasserstein2" if metric in {"wasserstein", "wasserstein2", "w2"} else "bhattacharyya",
            "scale": float(scale),
            "n_pairs": 0,
            "bounded_overlap": 0.0,
            "unweighted_mean_overlap": 0.0,
        }
    left, right = np.triu_indices(len(means), k=1)
    delta2 = (means[left] - means[right]) ** 2
    if metric == "bhattacharyya":
        average = 0.5 * (variances[left] + variances[right])
        distance = 0.125 * np.sum(delta2 / average, axis=1) + 0.5 * np.sum(
            np.log(average)
            - 0.5 * (np.log(variances[left]) + np.log(variances[right])),
            axis=1,
        )
        overlap = np.exp(-np.clip(distance, 0.0, 80.0))
        canonical_metric = "bhattacharyya"
    elif metric in {"wasserstein", "wasserstein2", "w2"}:
        distance2 = np.sum(
            delta2 + (np.sqrt(variances[left]) - np.sqrt(variances[right])) ** 2,
            axis=1,
        )
        overlap = float(scale) / (float(scale) + np.maximum(distance2, 0.0))
        canonical_metric = "wasserstein2"
    else:
        raise ValueError("metric must be 'bhattacharyya' or 'wasserstein2'")
    pair_weights = weights[left] * weights[right]
    return json_safe(
        {
            "metric": canonical_metric,
            "scale": float(scale),
            "n_pairs": int(len(overlap)),
            "bounded_overlap": float(np.sum(pair_weights * overlap) / np.sum(pair_weights)),
            "unweighted_mean_overlap": float(np.mean(overlap)),
        }
    )


def heldout_nll_degradation(
    heldout_latent: Any,
    baseline: Mapping[str, Any] | GaussianMixture,
    candidate: Mapping[str, Any] | GaussianMixture,
    *,
    max_degradation: float | None = None,
) -> dict[str, Any]:
    """Compare candidate versus baseline mean NLL on outcome-free held-out rows."""
    if max_degradation is not None and max_degradation < 0.0:
        raise ValueError("max_degradation must be non-negative when supplied")
    baseline_nll = -np.mean(
        diagonal_gmm_statistics(heldout_latent, baseline, dtype=np.float64)["log_density"]
    )
    candidate_nll = -np.mean(
        diagonal_gmm_statistics(heldout_latent, candidate, dtype=np.float64)["log_density"]
    )
    degradation = float(candidate_nll - baseline_nll)
    return json_safe(
        {
            "n_rows": int(len(_as_matrix(heldout_latent, name="heldout_latent"))),
            "baseline_mean_nll": float(baseline_nll),
            "candidate_mean_nll": float(candidate_nll),
            "mean_nll_degradation": degradation,
            "max_degradation": max_degradation,
            "passes_max_degradation": (
                None if max_degradation is None else bool(degradation <= max_degradation)
            ),
        }
    )


def refinement_promotion_diagnostics(
    heldout_latent: Any,
    baseline: Mapping[str, Any] | GaussianMixture,
    candidate: Mapping[str, Any] | GaussianMixture,
    *,
    overlap_metric: str = "bhattacharyya",
    overlap_scale: float = 1.0,
    max_heldout_nll_degradation: float | None = None,
) -> dict[str, Any]:
    """Combine overlap and held-out NLL checks for refinement-promotion policy."""
    overlap_before = bounded_diagonal_gmm_overlap(
        baseline, metric=overlap_metric, scale=overlap_scale
    )
    overlap_after = bounded_diagonal_gmm_overlap(
        candidate, metric=overlap_metric, scale=overlap_scale
    )
    nll = heldout_nll_degradation(
        heldout_latent,
        baseline,
        candidate,
        max_degradation=max_heldout_nll_degradation,
    )
    overlap_not_increased = bool(
        overlap_after["bounded_overlap"] <= overlap_before["bounded_overlap"] + 1.0e-12
    )
    nll_passes = nll["passes_max_degradation"]
    return {
        "overlap_before": overlap_before,
        "overlap_after": overlap_after,
        "bounded_overlap_delta": float(
            overlap_after["bounded_overlap"] - overlap_before["bounded_overlap"]
        ),
        "overlap_not_increased": overlap_not_increased,
        "heldout_nll": nll,
        "promotion_eligible": bool(
            overlap_not_increased and (nll_passes is None or nll_passes)
        ),
    }


def fit_common_gmm_panel(
    latent: Any,
    *,
    seeds: Sequence[int] = (0,),
    specs: Sequence[GmmPanelSpec | tuple[int, str, float]] = DEFAULT_GMM_PANEL,
    max_iter: int = 200,
    n_init: int = 1,
    retry_reg_covars: Sequence[float] = (0.01, 0.03, 0.1),
    failure_records: list[dict[str, Any]] | None = None,
    max_workers: int = 2,
) -> list[GmmPanelFit]:
    """Fit the standard K=4/6/8 diagonal, reg=.003 outcome-free panel."""
    z = _as_matrix(latent, name="latent", dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("latent contains non-finite values")
    if not np.any(np.ptp(z, axis=0) > 1.0e-10):
        raise ValueError("latent is constant and cannot support density clustering")
    normalized_specs = [spec if isinstance(spec, GmmPanelSpec) else GmmPanelSpec(*spec) for spec in specs]
    tasks: list[tuple[GmmPanelSpec, int]] = []
    for spec in normalized_specs:
        if spec.covariance_type != "diag" or spec.n_components < 1:
            raise ValueError("GMM panel only supports positive-component diagonal configurations")
        if len(z) < spec.n_components:
            raise ValueError("latent rows must be at least n_components for every panel member")
        for seed in seeds:
            tasks.append((spec, int(seed)))

    def fit_task(spec: GmmPanelSpec, seed: int) -> tuple[GmmPanelFit | None, dict[str, Any] | None]:
        retry_ladder = list(dict.fromkeys([
            float(spec.reg_covar),
            *[float(value) for value in retry_reg_covars if float(value) > float(spec.reg_covar)],
        ]))
        errors: list[str] = []
        limiter = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
        with limiter:
            for reg_covar in retry_ladder:
                try:
                    model = GaussianMixture(
                        n_components=spec.n_components,
                        covariance_type="diag",
                        reg_covar=float(reg_covar),
                        random_state=int(seed),
                        max_iter=int(max_iter),
                        n_init=int(n_init),
                    ).fit(z)
                    event = None
                    if float(reg_covar) != float(spec.reg_covar):
                        event = {
                            "status": "recovered_with_higher_reg_covar",
                            "n_components": int(spec.n_components),
                            "seed": int(seed),
                            "requested_reg_covar": float(spec.reg_covar),
                            "effective_reg_covar": float(reg_covar),
                            "errors": errors,
                        }
                    return (
                        GmmPanelFit(
                            spec=replace(spec, reg_covar=float(reg_covar)),
                            seed=int(seed),
                            state=json_safe(diagonal_gmm_state(model)),
                        ),
                        event,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    errors.append(f"reg={reg_covar:g}: {exc}")
        return None, {
            "status": "panel_member_rejected",
            "n_components": int(spec.n_components),
            "seed": int(seed),
            "requested_reg_covar": float(spec.reg_covar),
            "errors": errors,
        }

    workers = max(1, min(2, int(max_workers), len(tasks)))
    if workers == 1:
        results = [fit_task(spec, seed) for spec, seed in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gmm-panel") as pool:
            results = [future.result() for future in [pool.submit(fit_task, spec, seed) for spec, seed in tasks]]
    fits = [fit for fit, _event in results if fit is not None]
    if failure_records is not None:
        failure_records.extend(event for _fit, event in results if event is not None)
    if not fits:
        raise ValueError("No common-panel GMM member fit successfully")
    return fits


def _inverse_softplus(values: Any) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), EPSILON)
    return values + np.log(-np.expm1(-values))


def _bounded_overlap_penalty_torch(means: Any, variances: Any, weights: Any, *, metric: str, scale: float) -> Any:
    """Weighted pairwise overlap in [0, 1], implemented lazily with torch tensors."""
    import torch

    k = int(means.shape[0])
    if k < 2:
        return means.new_zeros(())
    left, right = torch.triu_indices(k, k, offset=1, device=means.device)
    delta2 = (means[left] - means[right]).pow(2)
    if metric == "bhattacharyya":
        average = 0.5 * (variances[left] + variances[right])
        distance = 0.125 * (delta2 / average).sum(dim=1) + 0.5 * (torch.log(average) - 0.5 * (torch.log(variances[left]) + torch.log(variances[right]))).sum(dim=1)
        overlap = torch.exp(-torch.clamp(distance, min=0.0, max=80.0))
    elif metric in {"wasserstein", "wasserstein2", "w2"}:
        distance2 = delta2.sum(dim=1) + (torch.sqrt(variances[left]) - torch.sqrt(variances[right])).pow(2).sum(dim=1)
        overlap = float(scale) / (float(scale) + torch.clamp(distance2, min=0.0))
    else:
        raise ValueError("overlap_metric must be 'bhattacharyya' or 'wasserstein2'")
    pair_weights = weights[left] * weights[right]
    return (pair_weights * overlap).sum() / torch.clamp(pair_weights.sum(), min=EPSILON)


def refine_diagonal_gmm(
    latent: Any,
    sklearn_model: GaussianMixture | Mapping[str, Any],
    *,
    overlap_lambda: float = 0.0,
    overlap_metric: str = "bhattacharyya",
    overlap_scale: float = 1.0,
    steps: int = 100,
    learning_rate: float = 0.01,
    min_variance: float = 1.0e-6,
    device: str = "cpu",
    tensor_cache: dict[tuple[int, tuple[int, ...], str], Any] | None = None,
) -> dict[str, Any]:
    """Optionally refine a sklearn-initialized diagonal GMM with bounded overlap.

    Variances are ``min_variance + softplus(raw_variance)`` and weights are a
    softmax.  Critically, ``overlap_lambda=0`` returns the unmodified sklearn
    state before importing torch, preserving ordinary sklearn semantics exactly.
    """
    initial = diagonal_gmm_state(sklearn_model)
    overlap_before = bounded_diagonal_gmm_overlap(
        initial, metric=overlap_metric, scale=overlap_scale
    )
    if overlap_lambda < 0.0:
        raise ValueError("overlap_lambda must be non-negative")
    if overlap_lambda == 0.0:
        return json_safe(
            {
                "state": initial,
                "refined": False,
                "overlap_lambda": 0.0,
                "steps": 0,
                "overlap_before": overlap_before,
                "overlap_after": overlap_before,
            }
        )
    if steps < 1 or learning_rate <= 0.0 or min_variance <= 0.0 or overlap_scale <= 0.0:
        raise ValueError("steps, learning_rate, min_variance, and overlap_scale must be positive")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError("torch is required only for overlap_lambda > 0 refinement") from exc
    z = _as_matrix(latent, name="latent", dtype=np.float64)
    if z.shape[1] != initial["means"].shape[1]:
        raise ValueError("latent dimension does not match GMM means")
    resolved_device = str(device)
    if resolved_device == "auto":
        resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Apple MPS does not implement float64 tensors.  CPU retains float64 for
    # mixture-NLL precision; MPS uses float32 consistently for the cached
    # latent matrix and all trainable mixture parameters.
    torch_dtype = torch.float32 if resolved_device == "mps" else torch.float64
    cache_key = (int(z.__array_interface__["data"][0]), tuple(z.shape), resolved_device, str(torch_dtype))
    tensor_z = None if tensor_cache is None else tensor_cache.get(cache_key)
    if tensor_z is None:
        tensor_z = torch.as_tensor(z, dtype=torch_dtype, device=resolved_device)
        if tensor_cache is not None:
            tensor_cache[cache_key] = tensor_z
    means = torch.nn.Parameter(torch.as_tensor(initial["means"], dtype=torch_dtype, device=resolved_device))
    raw_variance = torch.nn.Parameter(torch.as_tensor(_inverse_softplus(np.maximum(initial["covariances"] - min_variance, EPSILON)), dtype=torch_dtype, device=resolved_device))
    logits = torch.nn.Parameter(torch.as_tensor(np.log(np.maximum(initial["weights"], EPSILON)), dtype=torch_dtype, device=resolved_device))
    optimizer = torch.optim.Adam((means, raw_variance, logits), lr=float(learning_rate))
    last_nll, last_overlap = 0.0, 0.0
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        variances = float(min_variance) + torch.nn.functional.softplus(raw_variance)
        weights = torch.softmax(logits, dim=0)
        delta = tensor_z[:, None, :] - means[None, :, :]
        mahal_sq = (delta.pow(2) / variances[None, :, :]).sum(dim=2)
        log_joint = torch.log(weights)[None, :] - 0.5 * (mahal_sq + torch.log(variances).sum(dim=1)[None, :] + tensor_z.shape[1] * np.log(2.0 * np.pi))
        nll = -torch.logsumexp(log_joint, dim=1).mean()
        overlap = _bounded_overlap_penalty_torch(means, variances, weights, metric=overlap_metric, scale=float(overlap_scale))
        loss = nll + float(overlap_lambda) * overlap
        loss.backward()
        optimizer.step()
        last_nll, last_overlap = float(nll.detach().cpu()), float(overlap.detach().cpu())
    refined = {
        "means": means.detach().cpu().numpy(),
        "covariances": (float(min_variance) + torch.nn.functional.softplus(raw_variance)).detach().cpu().numpy(),
        "weights": torch.softmax(logits, dim=0).detach().cpu().numpy(),
    }
    overlap_after = bounded_diagonal_gmm_overlap(
        refined, metric=overlap_metric, scale=overlap_scale
    )
    return json_safe(
        {
            "state": diagonal_gmm_state(refined),
            "refined": True,
            "overlap_lambda": float(overlap_lambda),
            "overlap_metric": overlap_metric,
            "overlap_scale": float(overlap_scale),
            "steps": int(steps),
            "final_nll": last_nll,
            "final_overlap": last_overlap,
            "overlap_before": overlap_before,
            "overlap_after": overlap_after,
        }
    )


def evaluate_representation_proxies(
    posteriors: Any,
    *,
    posterior_runs: Mapping[str, Any] | None = None,
    expected_weights: Any | Mapping[str, Any] | None = None,
    strata: Mapping[str, Any] | None = None,
    latent_reference: Any | None = None,
    latent_perturbed: Any | None = None,
    posterior_perturbed: Any | None = None,
    ood_reference: Any | None = None,
    ood_perturbed: Any | None = None,
    component_alignment: ComponentAlignment | None = None,
) -> ProxyMetricsResult:
    """Convenience aggregator for a serialization-friendly proxy report."""
    probs = normalized_posteriors(posteriors)
    runs = dict(posterior_runs or {"reference": probs})
    runs.setdefault("reference", probs)
    entropy = stratified_entropy_distribution_diagnostics(probs, strata=strata)
    occupancy = occupancy_excess_instability(runs, expected_weights=expected_weights, strata=strata)
    if ood_reference is None:
        ood_reference = 1.0 - np.max(probs, axis=1)
    calibration = fit_ood_calibration(ood_reference)
    ood = {"reference": apply_ood_calibration(ood_reference, calibration), "calibration": json_safe(calibration)}
    perturbation: dict[str, Any] = {}
    if all(item is not None for item in (latent_reference, latent_perturbed, posterior_perturbed, ood_perturbed)):
        perturbation = perturbation_consistency(latent_reference, latent_perturbed, probs, posterior_perturbed, ood_reference, ood_perturbed)
        ood["perturbed"] = apply_ood_calibration(ood_perturbed, calibration)
    return ProxyMetricsResult(
        entropy=entropy,
        occupancy=occupancy,
        perturbation=perturbation,
        ood=ood,
        component_alignment=json_safe(component_alignment) if component_alignment is not None else None,
        manifest={"outcome_free": True, "n_rows": int(len(probs)), "n_components": int(probs.shape[1]), "strata": sorted(str(name) for name in (strata or {}))},
    )


__all__ = [
    "DEFAULT_GMM_PANEL",
    "ComponentAlignment",
    "GmmPanelFit",
    "GmmPanelSpec",
    "OodCalibration",
    "ProxyMetricsResult",
    "align_diagonal_gmm_components",
    "apply_ood_calibration",
    "bounded_diagonal_gmm_overlap",
    "diagonal_gmm_state",
    "diagonal_gmm_statistics",
    "entropy_distribution_diagnostics",
    "evaluate_ood_proxy",
    "evaluate_representation_proxies",
    "fit_common_gmm_panel",
    "fit_ood_calibration",
    "heldout_nll_degradation",
    "json_safe",
    "normalized_entropy",
    "normalized_posteriors",
    "occupancy_excess_instability",
    "perturbation_consistency",
    "refine_diagonal_gmm",
    "refinement_promotion_diagnostics",
    "reorder_posteriors_to_reference",
    "stratified_entropy_distribution_diagnostics",
]
