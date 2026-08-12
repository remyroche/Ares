"""Train-only input-population diagnostics for monthly base-model drift.

This module deliberately has no artifact, label, or model-runner dependency.
It answers only whether the population presented to a frozen base feature
contract has changed.  In particular, the helpers never use realised outcomes
and require an explicit *earlier* reference frame when timestamps are given.

The runner is responsible for resolving a side/fold's actual selected feature
list and for supplying only rows that were available at its fit cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


class BasePopulationDriftError(ValueError):
    """Raised when an input-population diagnostic violates its safety contract."""


@dataclass(frozen=True)
class AdversarialSeparability:
    """Held-out separability of earlier and current input populations."""

    held_out_auc: float
    train_rows_sampled: int
    current_rows_sampled: int
    held_out_rows: int
    feature_contributions: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, kind: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise BasePopulationDriftError(f"{kind} frame lacks columns: {missing}")


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _psi(reference: np.ndarray, current: np.ndarray, *, bins: int) -> float:
    reference, current = _finite(reference), _finite(current)
    if len(reference) < 2 or len(current) < 1:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference, current[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, bins=edges)[0].astype(float) / len(reference)
    q = np.histogram(current, bins=edges)[0].astype(float) / len(current)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _quantile_or_nan(values: np.ndarray, quantile: float) -> float:
    values = _finite(values)
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def _validate_train_before_current(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    timestamp_column: str | None,
) -> None:
    if timestamp_column is None:
        return
    _require_columns(reference, [timestamp_column], kind="reference")
    _require_columns(current, [timestamp_column], kind="current")
    reference_ts = pd.to_datetime(reference[timestamp_column], utc=True, errors="coerce")
    current_ts = pd.to_datetime(current[timestamp_column], utc=True, errors="coerce")
    if reference_ts.isna().any() or current_ts.isna().any():
        raise BasePopulationDriftError("timestamps must be non-missing and parseable")
    if len(reference_ts) and len(current_ts) and reference_ts.max() >= current_ts.min():
        raise BasePopulationDriftError(
            "reference rows must strictly precede the current population; "
            "use the base model's fit cutoff rather than a contemporaneous window"
        )


def feature_distribution_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    timestamp_column: str | None = None,
    psi_bins: int = 10,
    support_quantiles: tuple[float, float] = (0.005, 0.995),
) -> pd.DataFrame:
    """Return train-referenced per-feature coverage and distribution drift.

    ``reference`` must be the population actually eligible for the base
    model's fit.  If ``timestamp_column`` is supplied, the function enforces
    the strict temporal boundary directly.  Missing and constant fields are
    reported rather than zero-imputed, because either condition is itself
    candidate-population drift.
    """
    names = tuple(dict.fromkeys(map(str, feature_names)))
    if not names:
        raise BasePopulationDriftError("at least one feature is required")
    if psi_bins < 2:
        raise BasePopulationDriftError("psi_bins must be at least two")
    low_q, high_q = support_quantiles
    if not 0.0 <= low_q < high_q <= 1.0:
        raise BasePopulationDriftError("support quantiles must satisfy 0 <= low < high <= 1")
    _require_columns(reference, names, kind="reference")
    _require_columns(current, names, kind="current")
    _validate_train_before_current(reference, current, timestamp_column=timestamp_column)

    rows: list[dict[str, float | int | str]] = []
    for name in names:
        train = _numeric(reference[name])
        test = _numeric(current[name])
        train_finite, test_finite = _finite(train), _finite(test)
        q25, q50, q75 = (_quantile_or_nan(train, q) for q in (0.25, 0.5, 0.75))
        support_low, support_high = (_quantile_or_nan(train, q) for q in (low_q, high_q))
        robust_scale = (q75 - q25) / 1.349 if np.isfinite(q25) and np.isfinite(q75) else np.nan
        test_median = _quantile_or_nan(test, 0.5)
        if np.isfinite(robust_scale) and abs(robust_scale) > EPS:
            robust_median_shift = (test_median - q50) / robust_scale
        elif np.isfinite(test_median) and np.isfinite(q50) and np.isclose(test_median, q50):
            robust_median_shift = 0.0
        else:
            robust_median_shift = np.nan
        if len(test_finite) and np.isfinite(support_low) and np.isfinite(support_high):
            lower = float(np.mean(test_finite < support_low))
            upper = float(np.mean(test_finite > support_high))
        else:
            lower = upper = np.nan
        rows.append({
            "feature": name,
            "reference_rows": int(len(train)), "current_rows": int(len(test)),
            "reference_finite_rate": float(np.isfinite(train).mean()),
            "current_finite_rate": float(np.isfinite(test).mean()),
            "coverage_delta": float(np.isfinite(test).mean() - np.isfinite(train).mean()),
            "reference_unique_values": int(pd.Series(train_finite).nunique()),
            "current_unique_values": int(pd.Series(test_finite).nunique()),
            "reference_mean": float(np.mean(train_finite)) if len(train_finite) else np.nan,
            "current_mean": float(np.mean(test_finite)) if len(test_finite) else np.nan,
            "reference_std": float(np.std(train_finite)) if len(train_finite) else np.nan,
            "current_std": float(np.std(test_finite)) if len(test_finite) else np.nan,
            "reference_q005": support_low, "reference_q25": q25, "reference_median": q50,
            "reference_q75": q75, "reference_q995": support_high,
            "current_median": test_median,
            "robust_median_shift": robust_median_shift,
            "psi": _psi(train, test, bins=psi_bins),
            "wasserstein": float(wasserstein_distance(train_finite, test_finite))
            if len(train_finite) and len(test_finite) else np.nan,
            "lower_extrapolation_rate": lower, "upper_extrapolation_rate": upper,
            "extrapolation_rate": lower + upper if np.isfinite(lower) and np.isfinite(upper) else np.nan,
            "is_reference_constant": bool(len(train_finite) > 0 and np.ptp(train_finite) <= EPS),
            "is_current_constant": bool(len(test_finite) > 0 and np.ptp(test_finite) <= EPS),
        })
    return pd.DataFrame(rows)


def population_composition(
    frame: pd.DataFrame,
    *,
    month_column: str,
    side_column: str,
    asset_column: str,
    label_valid_column: str | None = None,
    class_column: str | None = None,
    numeric_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarise candidate volume, asset concentration and causal composition.

    This accepts no outcome column.  ``label_valid_column`` is included only
    as a substrate-coverage rate; it must not be treated as a model feature.
    Numeric composition fields are intended for decision-time values such as
    ATR/cost ratios or causal regime memberships.
    """
    required = [month_column, side_column, asset_column, *numeric_columns]
    if label_valid_column:
        required.append(label_valid_column)
    if class_column:
        required.append(class_column)
    _require_columns(frame, required, kind="population")
    if frame[[month_column, side_column, asset_column]].isna().any().any():
        raise BasePopulationDriftError("month, side and asset composition keys must be non-missing")
    rows: list[dict[str, object]] = []
    for (month, side), local in frame.groupby([month_column, side_column], sort=True, observed=True):
        shares = local[asset_column].value_counts(normalize=True, dropna=False)
        row: dict[str, object] = {
            month_column: month, side_column: side, "candidate_rows": int(len(local)),
            "active_assets": int(shares.size), "asset_hhi": float((shares * shares).sum()),
            "largest_asset_share": float(shares.iloc[0]),
        }
        if label_valid_column:
            values = local[label_valid_column]
            if not values.dropna().isin([True, False, 0, 1]).all():
                raise BasePopulationDriftError("label_valid_column must be boolean-like")
            row["label_valid_rate"] = float(values.astype("boolean").mean())
        if class_column:
            counts = local[class_column].value_counts(normalize=True, dropna=False)
            for label, share in counts.items():
                row[f"class_share__{label}"] = float(share)
        for name in numeric_columns:
            values = _finite(_numeric(local[name]))
            row[f"{name}__finite_rate"] = float(np.isfinite(_numeric(local[name])).mean())
            row[f"{name}__mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{name}__median"] = _quantile_or_nan(values, 0.5)
            row[f"{name}__q05"] = _quantile_or_nan(values, 0.05)
            row[f"{name}__q95"] = _quantile_or_nan(values, 0.95)
        rows.append(row)
    return pd.DataFrame(rows).sort_values([month_column, side_column], kind="stable").reset_index(drop=True)


def held_out_adversarial_separability(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    timestamp_column: str | None = None,
    max_rows_per_population: int = 75_000,
    held_out_fraction: float = 0.25,
    random_state: int = 20260810,
) -> AdversarialSeparability:
    """Measure population separability with a balanced, held-out classifier.

    Imputation medians and scaling are fit on the classifier's training split
    only.  The resulting AUC is therefore a diagnostic of available input
    shift, never an in-sample artefact or a realised-outcome metric.
    """
    names = tuple(dict.fromkeys(map(str, feature_names)))
    if not names:
        raise BasePopulationDriftError("at least one feature is required")
    if max_rows_per_population < 4:
        raise BasePopulationDriftError("max_rows_per_population must be at least four")
    if not 0.0 < held_out_fraction < 0.5:
        raise BasePopulationDriftError("held_out_fraction must be in (0, 0.5)")
    _require_columns(reference, names, kind="reference")
    _require_columns(current, names, kind="current")
    _validate_train_before_current(reference, current, timestamp_column=timestamp_column)
    n = min(len(reference), len(current), int(max_rows_per_population))
    if n < 4:
        raise BasePopulationDriftError("each population needs at least four rows")
    rng = np.random.default_rng(random_state)
    reference_index = rng.choice(len(reference), n, replace=False)
    current_index = rng.choice(len(current), n, replace=False)
    raw = np.vstack([
        reference.iloc[reference_index].loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        current.iloc[current_index].loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(float),
    ])
    labels = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=held_out_fraction, random_state=random_state)
    train_index, test_index = next(splitter.split(raw, labels))
    train_raw, test_raw = raw[train_index], raw[test_index]
    medians = np.nanmedian(train_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_clean = np.where(np.isfinite(train_raw), train_raw, medians)
    test_clean = np.where(np.isfinite(test_raw), test_raw, medians)
    scaler = StandardScaler().fit(train_clean)
    classifier = LogisticRegression(C=0.2, max_iter=500, random_state=random_state, n_jobs=1)
    classifier.fit(scaler.transform(train_clean), labels[train_index])
    auc = float(roc_auc_score(labels[test_index], classifier.predict_proba(scaler.transform(test_clean))[:, 1]))
    contribution = pd.DataFrame({
        "feature": names,
        "standardized_coefficient": classifier.coef_[0].astype(float),
    })
    contribution["absolute_standardized_coefficient"] = contribution["standardized_coefficient"].abs()
    contribution = contribution.sort_values("absolute_standardized_coefficient", ascending=False, kind="stable").reset_index(drop=True)
    return AdversarialSeparability(
        held_out_auc=auc, train_rows_sampled=n, current_rows_sampled=n,
        held_out_rows=int(len(test_index)), feature_contributions=contribution,
    )
