"""Stage-A chronological feature-portability audit.

The module is intentionally model-free.  It audits a declared feature
contract before feature selection or model fitting by comparing each era with
strictly earlier rows from the same optional stratum (for example, side).
It fails closed on latent AE/GMM/cluster/archetype regime outputs: Stage A is
about portable raw/context feature contracts, not a regime model.

The main entry point, :func:`run_chronological_feature_portability_audit`,
returns data frames suitable for an experiment runner and can be persisted by
the paired :func:`write_feature_portability_artifacts` helper or invoked with
``python -m extreme_price_movements.feature_portability_audit``.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_portability import (
    EPS,
    FeaturePortabilityError,
    FeatureRole,
    FeatureSemanticRole,
    PortabilityPolicy,
    assign_feature_dispositions,
    classify_feature_roles,
)


SCHEMA = "stage_a_feature_portability_audit_v1"
_LATENT_ROLES = {FeatureRole.LATENT_REGIME_OUTPUT, FeatureRole.FOLD_LOCAL_STATE}


@dataclass(frozen=True)
class ChronologicalAuditPolicy:
    """Stage-A gates applied after every strictly-prior era comparison."""

    portability: PortabilityPolicy = field(default_factory=PortabilityPolicy)
    min_reference_rows: int = 100
    distribution_bins: int = 10
    max_distribution_rows: int = 50_000
    max_era_shortcut_auc: float = 0.65
    min_semantic_stability: float = 0.50

    def __post_init__(self) -> None:
        if self.min_reference_rows < 2:
            raise FeaturePortabilityError("min_reference_rows must be at least two")
        if self.distribution_bins < 2 or self.max_distribution_rows < 2:
            raise FeaturePortabilityError("distribution bins and sample cap must be at least two")
        if not 0.5 <= self.max_era_shortcut_auc <= 1.0:
            raise FeaturePortabilityError("max_era_shortcut_auc must be in [0.5, 1]")
        if not 0.0 <= self.min_semantic_stability <= 1.0:
            raise FeaturePortabilityError("min_semantic_stability must be in [0, 1]")


@dataclass(frozen=True)
class ChronologicalFeaturePortabilityAudit:
    """The two Stage-A tables and a JSON-serializable immutable manifest."""

    era_audit: pd.DataFrame
    dispositions: pd.DataFrame
    manifest: Mapping[str, Any]


def _finite(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array[~np.isfinite(array)] = np.nan
    return array


def _bounded_distribution(values: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic quantile subsample so wide audits remain memory-bounded."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) <= cap:
        return finite
    ordered = np.sort(finite)
    positions = np.linspace(0, len(ordered) - 1, cap, dtype=np.int64)
    return ordered[positions]


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Reference-quantile PSI, with duplicate quantiles handled explicitly."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 2 or not len(current):
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference, current[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(reference, bins=edges)[0].astype(float)
    cur_hist = np.histogram(current, bins=edges)[0].astype(float)
    p = np.clip(ref_hist / ref_hist.sum(), EPS, None)
    q = np.clip(cur_hist / cur_hist.sum(), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def wasserstein_distance_1d(reference: np.ndarray, current: np.ndarray) -> float:
    """Exact 1-D Wasserstein distance without a SciPy dependency."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return float("nan")
    values = np.unique(np.concatenate([reference, current]))
    if len(values) < 2:
        return 0.0
    left = values[:-1]
    widths = np.diff(values)
    cdf_ref = np.searchsorted(reference, left, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, left, side="right") / len(current)
    return float(np.sum(np.abs(cdf_ref - cdf_cur) * widths))


def _binary_auc(reference: np.ndarray, current: np.ndarray) -> float:
    """Mann-Whitney AUC for recognising current-era rows from one feature."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return float("nan")
    combined = np.concatenate([reference, current])
    ranks = pd.Series(combined).rank(method="average").to_numpy(float)
    n_ref, n_current = len(reference), len(current)
    u_current = ranks[n_ref:].sum() - n_current * (n_current + 1.0) / 2.0
    return float(u_current / (n_ref * n_current))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rx = pd.Series(x[valid]).rank(method="average").to_numpy(float)
    ry = pd.Series(y[valid]).rank(method="average").to_numpy(float)
    sx, sy = rx.std(), ry.std()
    if sx <= EPS or sy <= EPS:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _top_bottom_delta(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return float("nan")
    xv, yv = x[valid], y[valid]
    lower, upper = np.quantile(xv, (0.20, 0.80))
    bottom, top = yv[xv <= lower], yv[xv >= upper]
    if not len(bottom) or not len(top):
        return float("nan")
    return float(top.mean() - bottom.mean())


def _semantic_stability(reference_effect: float, current_effect: float, *, floor: float = 0.02) -> tuple[float, bool | None]:
    """Continuous outcome-association stability proxy and a sign check."""
    if not (np.isfinite(reference_effect) and np.isfinite(current_effect)):
        return float("nan"), None
    scale = max(abs(reference_effect), abs(current_effect), floor)
    score = float(1.0 / (1.0 + abs(current_effect - reference_effect) / scale))
    if abs(reference_effect) < floor or abs(current_effect) < floor:
        return score, True
    return score, bool(np.sign(reference_effect) == np.sign(current_effect))


def _reference_edges(reference: np.ndarray, *, bins: int) -> np.ndarray:
    quantiles = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 2:
        return np.empty(0, dtype=float)
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    return quantiles


def _bin_support(current: np.ndarray, reference: np.ndarray, *, bins: int) -> tuple[int, int, int]:
    edges = _reference_edges(reference, bins=bins)
    if len(edges) < 2 or not len(current):
        return 0, 0, 0
    counts = np.histogram(current, bins=edges)[0]
    return int(len(counts)), int(counts.min()), int((counts > 0).sum())


def _era_labels(
    timestamps: pd.Series,
    frame: pd.DataFrame,
    *,
    era_column: str | None,
    era_frequency: str,
) -> pd.Series:
    if era_column is not None:
        if era_column not in frame:
            raise FeaturePortabilityError(f"missing era column {era_column!r}")
        labels = frame[era_column].copy()
        if labels.isna().any():
            raise FeaturePortabilityError("chronological audit era labels must be non-missing")
        return labels.astype(str)
    # PeriodIndex requires naive timestamps; the upstream conversion already
    # made these explicitly UTC, so removing timezone here is a representation
    # operation rather than a local-time inference.
    return pd.Series(
        pd.PeriodIndex(timestamps.dt.tz_localize(None), freq=era_frequency).astype(str),
        index=frame.index,
    )


def _prepare_chronological_panel(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    era_column: str | None,
    era_frequency: str,
    strata_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[tuple[int, str, np.ndarray, np.ndarray]], dict[int, dict[str, Any]]]:
    required = [timestamp_column, *strata_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise FeaturePortabilityError(f"missing chronological audit columns: {missing}")
    timestamp = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise FeaturePortabilityError("chronological audit requires finite UTC timestamps")
    if strata_columns and frame.loc[:, list(strata_columns)].isna().any().any():
        raise FeaturePortabilityError("chronological audit strata must be non-missing")
    work = frame.copy(deep=False).assign(
        __portability_timestamp__=timestamp.to_numpy(),
        __portability_position__=np.arange(len(frame), dtype=np.int64),
    )
    work["__portability_era__"] = _era_labels(
        timestamp, frame, era_column=era_column, era_frequency=era_frequency
    ).to_numpy()
    if strata_columns:
        work["__portability_scope__"] = pd.factorize(
            pd.MultiIndex.from_frame(work.loc[:, list(strata_columns)]), sort=False
        )[0]
    else:
        work["__portability_scope__"] = 0
    work = work.sort_values(
        ["__portability_scope__", "__portability_timestamp__", "__portability_position__"],
        kind="stable",
    ).reset_index(drop=True)

    period_rows: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    scope_metadata: dict[int, dict[str, Any]] = {}
    for scope, scoped in work.groupby("__portability_scope__", sort=False, observed=True):
        scope_id = int(scope)
        first = scoped.iloc[0]
        scope_metadata[scope_id] = {column: first[column] for column in strata_columns}
        timestamps_scope = scoped["__portability_timestamp__"].to_numpy(dtype="datetime64[ns]")
        local_positions = scoped.index.to_numpy(dtype=np.int64)
        periods = scoped.groupby("__portability_era__", sort=False, observed=True)["__portability_timestamp__"].agg(["min", "max"])
        periods = periods.sort_values("min", kind="stable")
        previous_end: pd.Timestamp | None = None
        for era, bounds in periods.iterrows():
            start = pd.Timestamp(bounds["min"])
            end = pd.Timestamp(bounds["max"])
            if previous_end is not None and start <= previous_end:
                raise FeaturePortabilityError(
                    "era labels must form non-overlapping chronological intervals within every stratum"
                )
            current_local = scoped.index[scoped["__portability_era__"].eq(era)].to_numpy(dtype=np.int64)
            reference_count = int(np.searchsorted(timestamps_scope, start.to_datetime64(), side="left"))
            period_rows.append((scope_id, str(era), current_local, local_positions[:reference_count]))
            previous_end = end
    return work, period_rows, scope_metadata


def _valid_support_metrics(reference: np.ndarray, current: np.ndarray, *, bins: int, cap: int) -> dict[str, float | int]:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    output: dict[str, float | int] = {
        "reference_finite_rows": int(len(reference)),
        "current_finite_rows": int(len(current)),
        "reference_p005": float("nan"),
        "reference_p995": float("nan"),
        "extrapolation_rate": float("nan"),
        "below_reference_p005_rate": float("nan"),
        "above_reference_p995_rate": float("nan"),
        "reference_bin_count": 0,
        "current_bin_min_support": 0,
        "current_bin_nonempty_count": 0,
        "psi": float("nan"),
        "wasserstein": float("nan"),
        "wasserstein_robust": float("nan"),
        "robust_median_shift": float("nan"),
        "mean_shift_std": float("nan"),
        "era_shortcut_auc_signed": float("nan"),
        "era_shortcut_auc": float("nan"),
    }
    if len(reference) < 2 or not len(current):
        return output
    p005, q25, q75, p995 = np.quantile(reference, (0.005, 0.25, 0.75, 0.995))
    below = current < p005
    above = current > p995
    output.update(
        {
            "reference_p005": float(p005),
            "reference_p995": float(p995),
            "below_reference_p005_rate": float(below.mean()),
            "above_reference_p995_rate": float(above.mean()),
            "extrapolation_rate": float((below | above).mean()),
        }
    )
    bin_count, min_support, nonempty = _bin_support(current, reference, bins=bins)
    output.update(
        {
            "reference_bin_count": bin_count,
            "current_bin_min_support": min_support,
            "current_bin_nonempty_count": nonempty,
        }
    )
    ref_sample = _bounded_distribution(reference, cap)
    cur_sample = _bounded_distribution(current, cap)
    iqr = float(q75 - q25)
    output["psi"] = population_stability_index(ref_sample, cur_sample, bins=bins)
    wasserstein = wasserstein_distance_1d(ref_sample, cur_sample)
    output["wasserstein"] = wasserstein
    output["wasserstein_robust"] = wasserstein / (iqr / 1.349) if abs(iqr) > EPS else (0.0 if wasserstein == 0.0 else float("inf"))
    median = float(np.median(current))
    output["robust_median_shift"] = (median - float(np.median(reference))) / (iqr / 1.349) if abs(iqr) > EPS else (0.0 if median == float(np.median(reference)) else float("inf"))
    reference_std = float(np.std(reference, ddof=0))
    output["mean_shift_std"] = (float(current.mean()) - float(reference.mean())) / reference_std if reference_std > EPS else (0.0 if float(current.mean()) == float(reference.mean()) else float("inf"))
    signed_auc = _binary_auc(ref_sample, cur_sample)
    output["era_shortcut_auc_signed"] = signed_auc
    output["era_shortcut_auc"] = max(signed_auc, 1.0 - signed_auc) if np.isfinite(signed_auc) else float("nan")
    return output


def _effect_metrics(feature: np.ndarray, outcome: np.ndarray | None) -> dict[str, float]:
    if outcome is None:
        return {"effect_spearman": float("nan"), "effect_top_bottom_delta": float("nan"), "effect_support": 0.0}
    valid = np.isfinite(feature) & np.isfinite(outcome)
    return {
        "effect_spearman": _spearman(feature, outcome),
        "effect_top_bottom_delta": _top_bottom_delta(feature, outcome),
        "effect_support": float(valid.sum()),
    }


def _stage_a_dispositions(
    era_audit: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    policy: ChronologicalAuditPolicy,
) -> pd.DataFrame:
    # Burn-in eras have no strictly-prior reference by construction.  They are
    # persisted in ``era_audit`` but never turn a later supported feature into
    # a false failure in its feature-level promotion disposition.
    eligible = era_audit.loc[era_audit["reference_ready"]].copy()
    if eligible.empty:
        base = pd.DataFrame({"feature": era_audit["feature"].drop_duplicates()})
        base = base.merge(roles, on="feature", how="left", validate="one_to_one")
        base["base_disposition"] = "REVIEW_REFERENCE_SUPPORT"
        base["base_disposition_reason"] = "no era has sufficient strictly-prior support"
    else:
        base = assign_feature_dispositions(eligible, roles=roles, policy=policy.portability)
        base = base.rename(columns={
            "disposition": "base_disposition",
            "disposition_reason": "base_disposition_reason",
        })
    summary = eligible.groupby("feature", observed=True, sort=True).agg(
        stage_a_min_coverage=("coverage", "min"),
        stage_a_max_extrapolation_rate=("extrapolation_rate", "max"),
        stage_a_min_bin_support=("current_bin_min_support", "min"),
        stage_a_min_nonempty_bins=("current_bin_nonempty_count", "min"),
        stage_a_max_era_shortcut_auc=("era_shortcut_auc", "max"),
        stage_a_min_semantic_stability=("semantic_stability_proxy", "min"),
        stage_a_evaluation_era_count=("era", "size"),
    ).reset_index()
    burnin = era_audit.groupby("feature", observed=True, sort=True).agg(
        stage_a_total_era_count=("era", "size"),
        stage_a_reference_ready_fraction=("reference_ready", "mean"),
    ).reset_index()
    result = base.merge(summary, on="feature", how="left", validate="one_to_one")
    result = result.merge(burnin, on="feature", how="left", validate="one_to_one")
    for index, row in result.iterrows():
        semantic = str(row["role"])
        lineage = str(row["portability_role"])
        disposition: str
        reason: str
        if lineage in {
            FeatureRole.OUTCOME_DERIVED, FeatureRole.IDENTITY,
            FeatureRole.CONTROL, FeatureRole.FOLD_LOCAL_STATE,
            FeatureRole.LATENT_REGIME_OUTPUT,
        }:
            disposition, reason = "REJECTED_LINEAGE", "non-causal, identity, control, or latent-regime lineage"
        elif lineage == FeatureRole.UNKNOWN:
            disposition, reason = "REJECTED_LINEAGE", "unapproved feature lineage/semantics"
        elif not np.isfinite(row.get("stage_a_evaluation_era_count", np.nan)):
            disposition, reason = "UNSTABLE", "no era has sufficient strictly-prior reference support"
        elif float(row["stage_a_min_coverage"]) < policy.portability.min_coverage:
            disposition, reason = "UNSTABLE", "99% per-era finite-coverage gate fails"
        elif float(row["stage_a_max_extrapolation_rate"]) > policy.portability.max_extrapolation_rate:
            disposition, reason = "UNSTABLE", "reference p0.5/p99.5 extrapolation gate fails"
        elif int(row["stage_a_min_nonempty_bins"]) < policy.portability.min_bins_represented:
            disposition, reason = "UNSTABLE", "bin-support gate: fewer than 8/10 reference-distribution bins are represented"
        elif np.isfinite(row["stage_a_max_era_shortcut_auc"]) and float(row["stage_a_max_era_shortcut_auc"]) > policy.max_era_shortcut_auc:
            disposition, reason = "ERA_SHORTCUT", "one field predicts the chronological era too well"
        elif np.isfinite(row["stage_a_min_semantic_stability"]) and float(row["stage_a_min_semantic_stability"]) < policy.min_semantic_stability:
            disposition = "INTERACTION_ONLY" if semantic in {FeatureSemanticRole.RELATIONSHIP_BREAK, FeatureSemanticRole.SETUP_ALIGNMENT} else "UNSTABLE"
            reason = "feature-outcome association is not stable enough as a stand-alone input"
        elif str(row.get("base_disposition", "")).startswith("REVIEW_EFFECT"):
            disposition = "INTERACTION_ONLY" if semantic in {FeatureSemanticRole.RELATIONSHIP_BREAK, FeatureSemanticRole.SETUP_ALIGNMENT} else "UNSTABLE"
            reason = "cross-era target-effect diagnostics are unstable"
        elif str(row.get("base_disposition", "")).startswith("REVIEW_INSUFFICIENT"):
            disposition, reason = "UNSTABLE", "finite support or uniqueness is insufficient"
        elif semantic == FeatureSemanticRole.MODEL_OUTPUT:
            disposition, reason = "CALIBRATION_ONLY", "model output may calibrate/admit but is not portable raw context"
        elif semantic == FeatureSemanticRole.SUPPORT_OR_TRUST:
            disposition, reason = "SUPPORT_ONLY", "support/trust context is not a directional alpha feature"
        elif semantic == FeatureSemanticRole.RELATIONSHIP_BREAK:
            disposition, reason = "CONDITIONABLE", "portable only as a declared condition/context"
        elif semantic == FeatureSemanticRole.SETUP_ALIGNMENT:
            disposition, reason = "INTERACTION_ONLY", "setup alignment requires a declared primary feature interaction"
        elif lineage == FeatureRole.CAUSAL_TRANSFORM_REQUIRED:
            disposition, reason = "INVARIANT_NORMALIZED", "raw level requires the causal transform bundle"
        elif semantic in {FeatureSemanticRole.RELATIVE_LEVEL, FeatureSemanticRole.CHANGE, FeatureSemanticRole.ACCELERATION}:
            disposition, reason = "INVARIANT_RELATIVE", "relative/change representation passes Stage-A gates"
        else:
            disposition, reason = "INVARIANT_RAW", "raw portable representation passes Stage-A gates"
        result.loc[index, ["disposition", "disposition_reason"]] = (disposition, reason)
    return result.sort_values("feature", kind="stable").reset_index(drop=True)


def run_chronological_feature_portability_audit(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    timestamp_column: str,
    era_column: str | None = None,
    era_frequency: str = "M",
    strata_columns: Sequence[str] = (),
    target_column: str | None = None,
    economic_residual_column: str | None = None,
    role_overrides: Mapping[str, str] | None = None,
    policy: ChronologicalAuditPolicy = ChronologicalAuditPolicy(),
) -> ChronologicalFeaturePortabilityAudit:
    """Run the reusable Stage-A audit on a raw causal feature contract.

    The reference for each ``(stratum, era)`` is all earlier observations in
    that stratum.  The first era intentionally fails the reference-support
    gate; callers should start the audited evaluation after enough history or
    retain the row as an explicit burn-in record rather than treating it as
    evidence.  Targets/residuals are diagnostics only and are not generated
    features.
    """
    features = tuple(dict.fromkeys(map(str, feature_names)))
    if not features:
        raise FeaturePortabilityError("Stage-A audit requires at least one feature")
    missing = [name for name in features if name not in frame]
    if missing:
        raise FeaturePortabilityError(f"Stage-A feature columns are missing: {missing[:12]}")
    for optional in (target_column, economic_residual_column):
        if optional is not None and optional not in frame:
            raise FeaturePortabilityError(f"Stage-A diagnostic column is missing: {optional!r}")
    roles = classify_feature_roles(features, overrides=role_overrides)
    latent = roles.loc[roles["portability_role"].isin(_LATENT_ROLES), "feature"].tolist()
    if latent:
        raise FeaturePortabilityError(
            "Stage-A audit forbids latent/fold-local regime outputs as inputs: "
            f"{latent[:12]}"
        )
    work, periods, scope_metadata = _prepare_chronological_panel(
        frame,
        timestamp_column=timestamp_column,
        era_column=era_column,
        era_frequency=era_frequency,
        strata_columns=strata_columns,
    )
    target = _finite(work[target_column]) if target_column else None
    residual = _finite(work[economic_residual_column]) if economic_residual_column else None
    feature_arrays = {name: _finite(work[name]) for name in features}
    rows: list[dict[str, Any]] = []
    for scope, era, current_positions, reference_positions in periods:
        reference_ready = len(reference_positions) >= policy.min_reference_rows
        for name in features:
            values = feature_arrays[name]
            current_values = values[current_positions]
            reference_values = values[reference_positions]
            metric = _valid_support_metrics(
                reference_values, current_values,
                bins=policy.distribution_bins, cap=policy.max_distribution_rows,
            )
            current_rows = len(current_positions)
            reference_rows = len(reference_positions)
            metric["coverage"] = float(np.isfinite(current_values).sum() / current_rows) if current_rows else float("nan")
            metric["reference_coverage"] = float(np.isfinite(reference_values).sum() / reference_rows) if reference_rows else float("nan")
            metric["rows"] = int(current_rows)
            metric["reference_rows"] = int(reference_rows)
            metric["finite_rows"] = int(np.isfinite(current_values).sum())
            metric["unique_values"] = int(len(np.unique(current_values[np.isfinite(current_values)])))
            metric["reference_ready"] = bool(reference_ready and metric["reference_finite_rows"] >= policy.min_reference_rows)
            current_effect = _effect_metrics(current_values, target[current_positions] if target is not None else None)
            reference_effect = _effect_metrics(reference_values, target[reference_positions] if target is not None else None)
            semantic, sign_consistent = _semantic_stability(
                reference_effect["effect_spearman"], current_effect["effect_spearman"]
            )
            metric.update(current_effect)
            metric["reference_effect_spearman"] = reference_effect["effect_spearman"]
            metric["reference_effect_top_bottom_delta"] = reference_effect["effect_top_bottom_delta"]
            metric["semantic_stability_proxy"] = semantic
            metric["semantic_sign_consistent"] = sign_consistent
            current_residual = _effect_metrics(current_values, residual[current_positions] if residual is not None else None)
            reference_residual = _effect_metrics(reference_values, residual[reference_positions] if residual is not None else None)
            residual_semantic, residual_sign = _semantic_stability(
                reference_residual["effect_spearman"], current_residual["effect_spearman"]
            )
            metric["economic_residual_spearman"] = current_residual["effect_spearman"]
            metric["reference_economic_residual_spearman"] = reference_residual["effect_spearman"]
            metric["economic_residual_semantic_stability"] = residual_semantic
            metric["economic_residual_sign_consistent"] = residual_sign
            if residual is not None:
                current_residual_values = residual[current_positions]
                reference_residual_values = residual[reference_positions]
                metric["economic_residual_mean"] = float(np.nanmean(current_residual_values)) if np.isfinite(current_residual_values).any() else float("nan")
                metric["reference_economic_residual_mean"] = float(np.nanmean(reference_residual_values)) if np.isfinite(reference_residual_values).any() else float("nan")
                metric["economic_residual_median"] = float(np.nanmedian(current_residual_values)) if np.isfinite(current_residual_values).any() else float("nan")
                metric["economic_residual_positive_rate"] = float((current_residual_values[np.isfinite(current_residual_values)] > 0.0).mean()) if np.isfinite(current_residual_values).any() else float("nan")
            else:
                metric.update({
                    "economic_residual_mean": float("nan"),
                    "reference_economic_residual_mean": float("nan"),
                    "economic_residual_median": float("nan"),
                    "economic_residual_positive_rate": float("nan"),
                })
            metric.update({"feature": name, "era": era, **scope_metadata[scope]})
            rows.append(metric)
    era_audit = pd.DataFrame(rows)
    if era_audit.empty:
        raise FeaturePortabilityError("Stage-A audit has no chronological era rows")
    era_audit = era_audit.merge(roles, on="feature", how="left", validate="many_to_one")
    sort_columns = ["feature", *strata_columns, "era"]
    era_audit = era_audit.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    dispositions = _stage_a_dispositions(era_audit, roles, policy=policy)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "rows": int(len(frame)),
        "feature_count": int(len(features)),
        "feature_names": list(features),
        "timestamp_column": timestamp_column,
        "era_column": era_column,
        "era_frequency": era_frequency if era_column is None else None,
        "strata_columns": list(strata_columns),
        "target_column": target_column,
        "economic_residual_column": economic_residual_column,
        "reference_semantics": "strictly earlier timestamp rows within the same stratum",
        "latent_regime_outputs_allowed": False,
        "policy": {
            "min_coverage": policy.portability.min_coverage,
            "max_extrapolation_rate": policy.portability.max_extrapolation_rate,
            "min_bin_support": policy.portability.min_bin_support,
            "min_bins_represented": policy.portability.min_bins_represented,
            "min_reference_rows": policy.min_reference_rows,
            "distribution_bins": policy.distribution_bins,
            "max_era_shortcut_auc": policy.max_era_shortcut_auc,
            "min_semantic_stability": policy.min_semantic_stability,
        },
        "disposition_counts": dispositions["disposition"].value_counts().sort_index().to_dict(),
        "features": dispositions.to_dict(orient="records"),
    }
    return ChronologicalFeaturePortabilityAudit(
        era_audit=era_audit, dispositions=dispositions, manifest=manifest
    )


def write_feature_portability_artifacts(
    result: ChronologicalFeaturePortabilityAudit,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Persist the three Stage-A artifacts without changing any model runner."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "era_audit": directory / "feature_portability_era_audit.parquet",
        "dispositions": directory / "feature_portability_dispositions.parquet",
        "manifest": directory / "feature_portability_role_disposition_manifest.json",
    }
    result.era_audit.to_parquet(paths["era_audit"], index=False)
    result.dispositions.to_parquet(paths["dispositions"], index=False)
    paths["manifest"].write_text(
        json.dumps(_json_safe(result.manifest), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas scalars and diagnostic NaNs to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _feature_names_from_args(args: Any) -> list[str]:
    names: list[str] = []
    if args.features:
        names.extend(item.strip() for item in args.features.split(",") if item.strip())
    if args.feature_list:
        names.extend(
            item.strip()
            for item in Path(args.feature_list).read_text(encoding="utf-8").splitlines()
            if item.strip() and not item.lstrip().startswith("#")
        )
    names = list(dict.fromkeys(names))
    if not names:
        raise FeaturePortabilityError("supply --features or --feature-list")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    """Small standalone CLI; it audits a parquet panel and writes Stage-A artifacts."""
    parser = ArgumentParser(description="Run the model-free Stage-A feature portability audit")
    parser.add_argument("--input", required=True, help="Causal input parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp-column", required=True)
    parser.add_argument("--features", help="Comma-separated input feature names")
    parser.add_argument("--feature-list", help="Newline-delimited feature contract")
    parser.add_argument("--era-column")
    parser.add_argument("--era-frequency", default="M")
    parser.add_argument("--strata-column", action="append", default=[])
    parser.add_argument("--target-column")
    parser.add_argument("--economic-residual-column")
    parser.add_argument("--min-reference-rows", type=int, default=100)
    args = parser.parse_args(argv)
    feature_names = _feature_names_from_args(args)
    frame = pd.read_parquet(args.input)
    policy = ChronologicalAuditPolicy(min_reference_rows=args.min_reference_rows)
    result = run_chronological_feature_portability_audit(
        frame,
        feature_names=feature_names,
        timestamp_column=args.timestamp_column,
        era_column=args.era_column,
        era_frequency=args.era_frequency,
        strata_columns=args.strata_column,
        target_column=args.target_column,
        economic_residual_column=args.economic_residual_column,
        policy=policy,
    )
    paths = write_feature_portability_artifacts(result, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "ChronologicalAuditPolicy",
    "ChronologicalFeaturePortabilityAudit",
    "population_stability_index",
    "run_chronological_feature_portability_audit",
    "wasserstein_distance_1d",
    "write_feature_portability_artifacts",
]
