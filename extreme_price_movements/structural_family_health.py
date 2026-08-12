"""Strict-historical health features for token-free structural tree families.

Rule alignment belongs to the outcome-free structural stage.  This separate
sidecar attaches only information from rows whose labels resolved strictly
before the candidate's decision timestamp.  It works on a bounded wide matrix
of selected soft family memberships rather than expanding candidates by leaves
or trees.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


POSTERIOR_PREFIX = "base_structural_family__"
FORBIDDEN_TOKENS = ("leaf", "target", "label", "outcome", "realised", "realized", "pnl", "gross")


class StructuralFamilyHealthError(ValueError):
    """Raised when the structural historical-health contract is unsafe."""


@dataclass(frozen=True)
class StructuralFamilyHealthConfig:
    """Fixed bounded controls; none are outcome/HPO selection settings."""

    max_context_fields: int = 10
    prior_correctness_strength: float = 16.0
    context_variance_floor: float = 1e-4
    context_z_clip: float = 6.0

    def validate(self) -> None:
        if not 1 <= int(self.max_context_fields) <= 20:
            raise StructuralFamilyHealthError("max_context_fields must lie in [1, 20]")
        if self.prior_correctness_strength <= 0 or self.context_variance_floor <= 0 or self.context_z_clip <= 0:
            raise StructuralFamilyHealthError("health prior and context controls must be positive")


def _posterior_columns(frame: pd.DataFrame) -> list[str]:
    fields = [
        str(name) for name in frame
        if str(name).startswith(POSTERIOR_PREFIX)
        and str(name) != f"{POSTERIOR_PREFIX}unassigned_mass"
    ]
    if not fields:
        raise StructuralFamilyHealthError("selected token-free structural posterior features are required")
    bad = [name for name in fields if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if bad:
        raise StructuralFamilyHealthError(f"raw/outcome structural fields are forbidden: {bad}")
    return sorted(fields)


def _validate(
    candidates: pd.DataFrame, context_columns: Sequence[str], config: StructuralFamilyHealthConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    config.validate()
    required = {"candidate_id", "decision_ts", "label_available_ts", "net_bps", "base_expected_bps"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise StructuralFamilyHealthError(f"candidate frame misses {missing}")
    posterior = _posterior_columns(candidates)
    context = list(map(str, context_columns))[: int(config.max_context_fields)]
    missing = sorted(set(context).difference(candidates.columns))
    if missing:
        raise StructuralFamilyHealthError(f"candidate frame misses context fields {missing}")
    bad_context = [name for name in context if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if bad_context:
        raise StructuralFamilyHealthError(f"outcome-like context fields are forbidden: {bad_context}")
    fields = ["candidate_id", "decision_ts", "label_available_ts", "net_bps", "base_expected_bps", *posterior, *context]
    work = candidates.loc[:, fields].copy()
    for name in ("decision_ts", "label_available_ts"):
        work[name] = pd.to_datetime(work[name], utc=True, errors="coerce")
    if work[["decision_ts", "label_available_ts"]].isna().any().any():
        raise StructuralFamilyHealthError("decision and label timestamps must be finite UTC values")
    if not work["label_available_ts"].ge(work["decision_ts"]).all():
        raise StructuralFamilyHealthError("a label cannot resolve before its decision")
    if work["candidate_id"].isna().any() or work["candidate_id"].astype(str).str.strip().eq("").any() or work["candidate_id"].duplicated().any():
        raise StructuralFamilyHealthError("candidate_id must be unique and nonblank")
    numeric = ["net_bps", "base_expected_bps", *posterior, *context]
    for name in numeric:
        work[name] = pd.to_numeric(work[name], errors="coerce")
    if not np.isfinite(work[numeric].to_numpy(float)).all():
        raise StructuralFamilyHealthError("impute causal context and posterior values before historical health")
    weights = work[posterior].to_numpy(float)
    if (weights < -1e-7).any() or (weights > 1.0 + 1e-7).any() or (weights.sum(axis=1) > 1.0 + 1e-5).any():
        raise StructuralFamilyHealthError("selected structural posterior mass must be finite and lie in [0, 1]")
    return work.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True), posterior, context


def _completed_period_metrics(period_mass: dict[str, np.ndarray], period_residual: dict[str, np.ndarray], before_month: str, families: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return count/std/sign-reversal/worst from months fully before current."""
    keys = [key for key in sorted(period_mass) if key < before_month]
    result = tuple(np.zeros(families, dtype=np.float64) for _ in range(4))
    if not keys:
        return result
    mass = np.stack([period_mass[key] for key in keys], axis=1)
    resid = np.stack([period_residual[key] for key in keys], axis=1)
    count, deviation, reversal, worst = result
    for family in range(families):
        valid = mass[family] > 0.0
        values = resid[family, valid] / mass[family, valid]
        if not len(values):
            continue
        count[family] = len(values)
        deviation[family] = float(np.std(values))
        reversal[family] = 2.0 * min(float(np.mean(values > 0.0)), float(np.mean(values < 0.0)))
        worst[family] = float(np.min(values))
    return result


def build_structural_family_historical_health(
    candidates: pd.DataFrame,
    *, context_columns: Sequence[str],
    config: StructuralFamilyHealthConfig = StructuralFamilyHealthConfig(),
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Materialise candidate-level H1/H2/F5-style structural health.

    Metrics are contribution-posterior-weighted aggregates across selected
    structural families.  The update loop is vectorised by decision timestamp;
    its state is O(selected families × context fields × completed months), not
    O(candidates × trees).  A label at exactly ``t`` is deliberately excluded
    while scoring candidates at ``t``.
    """
    work, posterior, context = _validate(candidates, context_columns, config)
    rows, families, dimensions = len(work), len(posterior), len(context)
    weight = work[posterior].to_numpy(np.float64, copy=False)
    net = work["net_bps"].to_numpy(np.float64)
    residual = net - work["base_expected_bps"].to_numpy(np.float64)
    success = (net > 0.0).astype(np.float64)
    context_values = work[context].to_numpy(np.float64, copy=False) if dimensions else np.empty((rows, 0), dtype=np.float64)
    decision = work["decision_ts"].to_numpy()
    available = work["label_available_ts"].to_numpy()

    mass = np.zeros(families, dtype=np.float64)
    successes = np.zeros(families, dtype=np.float64)
    residual_sum = np.zeros(families, dtype=np.float64)
    residual_square_sum = np.zeros(families, dtype=np.float64)
    context_sum = np.zeros((families, dimensions), dtype=np.float64)
    context_square_sum = np.zeros((families, dimensions), dtype=np.float64)
    period_mass: dict[str, np.ndarray] = {}
    period_residual: dict[str, np.ndarray] = {}
    pending: dict[pd.Timestamp, list[np.ndarray]] = {}
    output = np.zeros((rows, 11), dtype=np.float32)
    snapshot = tuple(np.zeros(families, dtype=np.float64) for _ in range(4))
    last_month: str | None = None

    def resolve(indices: np.ndarray, label_time: pd.Timestamp) -> None:
        if not len(indices):
            return
        local = weight[indices]
        nonlocal mass, successes, residual_sum, residual_square_sum, context_sum, context_square_sum
        mass += local.sum(axis=0)
        successes += local.T @ success[indices]
        residual_sum += local.T @ residual[indices]
        residual_square_sum += local.T @ np.square(residual[indices])
        if dimensions:
            context_sum += local.T @ context_values[indices]
            context_square_sum += local.T @ np.square(context_values[indices])
        month = str(label_time.strftime("%Y-%m"))
        period_mass.setdefault(month, np.zeros(families, dtype=np.float64))[:] += local.sum(axis=0)
        period_residual.setdefault(month, np.zeros(families, dtype=np.float64))[:] += local.T @ residual[indices]

    cursor = 0
    while cursor < rows:
        current = pd.Timestamp(decision[cursor])
        end = cursor + 1
        while end < rows and pd.Timestamp(decision[end]) == current:
            end += 1
        # Strict availability: equality is not causal availability.
        for due in sorted(key for key in pending if key < current):
            for indices in pending.pop(due):
                resolve(indices, due)
        month = str(current.strftime("%Y-%m"))
        if month != last_month:
            snapshot = _completed_period_metrics(period_mass, period_residual, month, families)
            last_month = month

        local = weight[cursor:end]
        active = local.sum(axis=1)
        denominator = np.maximum(active, 1e-12)
        correctness = (successes + .5 * config.prior_correctness_strength) / (mass + config.prior_correctness_strength)
        average_residual = np.divide(residual_sum, mass, out=np.zeros(families), where=mass > 0.0)
        residual_variance = np.divide(residual_square_sum, mass, out=np.zeros(families), where=mass > 0.0) - np.square(average_residual)
        output[cursor:end, 0] = active
        output[cursor:end, 1] = local @ np.log1p(mass) / denominator
        output[cursor:end, 2] = local @ correctness / denominator
        output[cursor:end, 3] = local @ average_residual / denominator
        output[cursor:end, 4] = local @ np.sqrt(np.maximum(residual_variance, 0.0)) / denominator
        for offset, values in enumerate(snapshot, start=5):
            output[cursor:end, offset] = local @ values / denominator
        if dimensions:
            means = np.divide(context_sum, mass[:, None], out=np.zeros_like(context_sum), where=mass[:, None] > 0.0)
            variance = np.divide(context_square_sum, mass[:, None], out=np.zeros_like(context_square_sum), where=mass[:, None] > 0.0) - np.square(means)
            compatibility = np.zeros((end - cursor, families), dtype=np.float64)
            for family in range(families):
                if mass[family] <= 0.0:
                    continue
                z = (context_values[cursor:end] - means[family]) / np.sqrt(np.maximum(variance[family], config.context_variance_floor))
                compatibility[:, family] = np.exp(-np.mean(np.square(np.clip(z, -config.context_z_clip, config.context_z_clip)), axis=1))
            output[cursor:end, 9] = (local * compatibility).sum(axis=1) / denominator
        output[cursor:end, 10] = local @ average_residual / denominator

        batch_indices = np.arange(cursor, end, dtype=np.int64)
        schedule = pd.DataFrame({
            "__index__": batch_indices,
            "__label_time__": pd.to_datetime(available[cursor:end], utc=True),
        })
        for label_time, group in schedule.groupby("__label_time__", sort=False, observed=True):
            pending.setdefault(pd.Timestamp(label_time), []).append(
                group["__index__"].to_numpy(dtype=np.int64)
            )
        cursor = end

    names = (
        "structural_health__active_posterior_mass",
        "structural_health__historical_log_support",
        "structural_health__historical_correctness",
        "structural_health__historical_residual_bps",
        "structural_health__historical_residual_std_bps",
        "structural_health__completed_period_count",
        "structural_health__period_residual_std_bps",
        "structural_health__period_sign_reversal_rate",
        "structural_health__period_worst_residual_bps",
        "structural_health__context_compatibility",
        "structural_health__contextual_residual_bps",
    )
    result = work.loc[:, ["candidate_id", "decision_ts"]].copy()
    for index, name in enumerate(names):
        result[name] = np.nan_to_num(output[:, index], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite structural health sidecar: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False, compression="zstd")
    return result


__all__ = ["StructuralFamilyHealthConfig", "StructuralFamilyHealthError", "build_structural_family_historical_health"]
