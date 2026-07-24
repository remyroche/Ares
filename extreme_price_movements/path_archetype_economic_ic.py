"""Leakage-safe prediction-to-outcome IC diagnostics for path archetypes.

The CatBoost path-archetype classifier emits a probability vector, while the
taxonomy itself is constructed from realised path outcomes.  This module
measures whether those probabilities continuously rank *out-of-fold* rows by
economically relevant realised outcomes.  It deliberately fits class outcome
priors, robust scales, and compact class centroids on a permitted training
fold only.  OOS outcomes are accepted only by :meth:`evaluate` and are never
used to update fitted state.

The IC sign convention is explicit:

* net EV and MFE: higher is better;
* MAE, time to realization, and stop probability: lower is better.

All reported ICs therefore have the same interpretation: a positive value
means the probability-weighted train-only class mapping orders OOS rows in the
economically useful direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_EPS = 1e-8


@dataclass(frozen=True)
class EconomicOutcomeSpec:
    """Column and economic-orientation contract for one realised outcome."""

    name: str
    column: str
    direction: float
    pooled_weight: float

    @property
    def higher_is_better(self) -> bool:
        return self.direction > 0.0


DEFAULT_OUTCOME_SPECS: tuple[EconomicOutcomeSpec, ...] = (
    EconomicOutcomeSpec("net_ev_after_cost", "net_ev_after_1pct_return", 1.0, 0.35),
    EconomicOutcomeSpec("mfe", "peak_mfe_12h_atr", 1.0, 0.20),
    EconomicOutcomeSpec("mae", "mae_12h_atr", -1.0, 0.20),
    EconomicOutcomeSpec("time_to_realization", "time_to_first_meaningful_mfe", -1.0, 0.10),
    EconomicOutcomeSpec("stop_probability", "stop_probability", -1.0, 0.15),
)


@dataclass(frozen=True)
class EconomicICColumns:
    """Non-outcome OOS grouping fields used exclusively for reporting."""

    class_label: str = "path_geometry_label"
    timestamp: str = "__ts__"
    side: str = "side"
    symbol: str = "symbol"


@dataclass(frozen=True)
class EconomicICConfig:
    """Evaluation contract.  Defaults are conservative for cross-sections."""

    outcomes: tuple[EconomicOutcomeSpec, ...] = DEFAULT_OUTCOME_SPECS
    columns: EconomicICColumns = EconomicICColumns()
    min_group_rows: int = 8
    min_symbol_rows: int = 8

    def validate(self) -> None:
        if len(self.outcomes) < 2:
            raise ValueError("At least two outcomes are required for pooled IC")
        names = [spec.name for spec in self.outcomes]
        if len(set(names)) != len(names):
            raise ValueError("Outcome names must be unique")
        columns = [spec.column for spec in self.outcomes]
        if len(set(columns)) != len(columns):
            raise ValueError("Outcome columns must be unique")
        if any(spec.direction not in {-1.0, 1.0} for spec in self.outcomes):
            raise ValueError("Outcome directions must be either -1 or 1")
        if any(spec.pooled_weight < 0.0 for spec in self.outcomes):
            raise ValueError("Pooled outcome weights cannot be negative")
        if not np.isclose(sum(spec.pooled_weight for spec in self.outcomes), 1.0):
            raise ValueError("Pooled outcome weights must sum to one")
        if self.min_group_rows < 2 or self.min_symbol_rows < 2:
            raise ValueError("Minimum IC group support must be at least two rows")


@dataclass
class EconomicICDiagnostics:
    """OOS-only IC output plus frozen train-fold state required for auditing."""

    global_ic: pd.DataFrame
    probability_class_ic: pd.DataFrame
    true_archetype_ic: pd.DataFrame
    timestamp_ic: pd.DataFrame
    timestamp_summary: pd.DataFrame
    monthly_ic: pd.DataFrame
    side_ic: pd.DataFrame
    symbol_ic: pd.DataFrame
    symbol_neutral_ic: pd.DataFrame
    class_priors: pd.DataFrame
    class_centroids: pd.DataFrame
    provenance: dict[str, Any]
    quality: pd.DataFrame

    def persist(self, output_dir: str | Path, *, prefix: str = "path_archetype_economic_ic") -> dict[str, Path]:
        """Persist tables and a manifest without serializing OOS outcomes as fit state."""

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        tables: Mapping[str, pd.DataFrame] = {
            "global_ic": self.global_ic,
            "probability_class_ic": self.probability_class_ic,
            "true_archetype_ic": self.true_archetype_ic,
            "timestamp_ic": self.timestamp_ic,
            "timestamp_summary": self.timestamp_summary,
            "monthly_ic": self.monthly_ic,
            "side_ic": self.side_ic,
            "symbol_ic": self.symbol_ic,
            "symbol_neutral_ic": self.symbol_neutral_ic,
            "class_priors": self.class_priors,
            "class_centroids": self.class_centroids,
            "quality": self.quality,
        }
        paths: dict[str, Path] = {}
        for name, table in tables.items():
            path = directory / f"{prefix}_{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path
        manifest = {
            "state_scope": "train_only_class_outcome_priors_and_centroids",
            "oos_outcomes_used_only_for_evaluation": True,
            "provenance": self.provenance,
            "tables": {name: str(path) for name, path in paths.items()},
        }
        manifest_path = directory / f"{prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        paths["manifest"] = manifest_path
        return paths


def _as_probability_matrix(values: Any, expected_rows: int, expected_columns: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (expected_rows, expected_columns):
        raise ValueError(
            "probabilities must have shape "
            f"({expected_rows}, {expected_columns}); got {matrix.shape}"
        )
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    total = matrix.sum(axis=1, keepdims=True)
    if (total <= 0.0).any():
        raise ValueError("every probability row must contain positive mass")
    return matrix / total


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


def _finite_mean(values: np.ndarray, fallback: float) -> float:
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else float(fallback)


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    scale = np.maximum(q75 - q25, _EPS)
    return median.astype(np.float64, copy=False), scale.astype(np.float64, copy=False)


def _spearman(expected: np.ndarray, actual: np.ndarray) -> tuple[float, int, str]:
    valid = np.isfinite(expected) & np.isfinite(actual)
    rows = int(valid.sum())
    if rows < 2:
        return np.nan, rows, "insufficient_finite_rows"
    left = pd.Series(expected[valid]).rank(method="average").to_numpy(dtype=np.float64)
    right = pd.Series(actual[valid]).rank(method="average").to_numpy(dtype=np.float64)
    left_std = float(left.std(ddof=0))
    right_std = float(right.std(ddof=0))
    if left_std <= _EPS or right_std <= _EPS:
        return np.nan, rows, "constant_expected_or_outcome"
    return float(np.corrcoef(left, right)[0, 1]), rows, "ok"


def _month_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.strftime("%Y-%m").astype("string")


class PathArchetypeEconomicIC:
    """Fit train-only outcome priors and evaluate OOS probability ordering.

    Class labels are required by :meth:`fit`.  If an OOS realised archetype
    label is supplied, :meth:`evaluate` uses it only to break down already
    computed ICs; its expected-outcome mapping always comes entirely from
    class probabilities and frozen train-only priors.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        classes: Sequence[str],
        *,
        config: EconomicICConfig = EconomicICConfig(),
    ) -> None:
        self.classes = tuple(str(value) for value in classes)
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must contain at least two unique values")
        config.validate()
        self.config = config
        self._fitted = False

    def _require_outcomes(self, frame: pd.DataFrame, *, scope: str) -> None:
        required = [spec.column for spec in self.config.outcomes]
        missing = [column for column in required if column not in frame]
        if missing:
            raise KeyError(f"{scope} is missing outcome columns: {missing}")

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        class_labels: Sequence[str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "PathArchetypeEconomicIC":
        """Fit class priors/centroids only on an authorized training fold."""

        self._require_outcomes(train_frame, scope="train_frame")
        label_values = (
            train_frame[self.config.columns.class_label]
            if class_labels is None
            else pd.Series(class_labels, index=train_frame.index)
        )
        if len(label_values) != len(train_frame):
            raise ValueError("class_labels must align one-for-one with train_frame")
        labels = label_values.astype("string")
        unknown = sorted(set(labels.dropna().astype(str)).difference(self.classes))
        if unknown:
            raise ValueError(f"train labels contain unknown classes: {unknown}")
        if labels.isna().any():
            raise ValueError("train labels cannot be missing")

        raw = np.column_stack([_numeric_column(train_frame, spec.column) for spec in self.config.outcomes])
        global_prior = np.asarray([_finite_mean(raw[:, index], 0.0) for index in range(raw.shape[1])])
        medians, iqr = _robust_location_scale(raw)
        filled = np.where(np.isfinite(raw), raw, medians[None, :])
        standardized = np.clip((filled - medians) / iqr, -12.0, 12.0)

        priors = np.empty((len(self.classes), len(self.config.outcomes)), dtype=np.float64)
        centroids = np.empty_like(priors)
        support = np.zeros(len(self.classes), dtype=np.int64)
        labels_array = labels.astype(str).to_numpy()
        for class_index, class_name in enumerate(self.classes):
            mask = labels_array == class_name
            support[class_index] = int(mask.sum())
            if not support[class_index]:
                priors[class_index] = global_prior
                centroids[class_index] = np.zeros(raw.shape[1], dtype=np.float64)
                continue
            class_raw = raw[mask]
            priors[class_index] = np.asarray(
                [_finite_mean(class_raw[:, index], global_prior[index]) for index in range(raw.shape[1])]
            )
            centroids[class_index] = np.median(standardized[mask], axis=0)

        self.class_priors_ = priors
        self.class_centroids_ = centroids
        self.class_support_ = support
        self.global_prior_ = global_prior
        self.outcome_median_ = medians
        self.outcome_iqr_ = iqr
        self.provenance_ = {
            "state_version": self.STATE_VERSION,
            "fit_scope": "authorized_train_rows_only",
            "oos_outcomes_used_for_fit": False,
            "train_rows": int(len(train_frame)),
            "classes": list(self.classes),
            "outcomes": [
                {
                    "name": spec.name,
                    "column": spec.column,
                    "direction": spec.direction,
                    "higher_is_better": spec.higher_is_better,
                    "pooled_weight": spec.pooled_weight,
                }
                for spec in self.config.outcomes
            ],
            **dict(provenance or {}),
        }
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PathArchetypeEconomicIC must be fitted before evaluation")

    def _expected_and_actual(self, oos_frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_actual = np.column_stack([_numeric_column(oos_frame, spec.column) for spec in self.config.outcomes])
        expected_raw = probabilities @ self.class_priors_
        directions = np.asarray([spec.direction for spec in self.config.outcomes], dtype=np.float64)
        expected_utility = directions[None, :] * expected_raw
        actual_utility = directions[None, :] * raw_actual
        expected_scaled = (expected_utility - directions[None, :] * self.outcome_median_) / self.outcome_iqr_
        actual_scaled = (actual_utility - directions[None, :] * self.outcome_median_) / self.outcome_iqr_
        weights = np.asarray([spec.pooled_weight for spec in self.config.outcomes], dtype=np.float64)
        pooled_expected = expected_scaled @ weights
        pooled_actual = actual_scaled @ weights
        return expected_raw, raw_actual, np.column_stack((expected_utility, pooled_expected)), np.column_stack((actual_utility, pooled_actual))

    def _ic_rows(
        self,
        expected_raw: np.ndarray,
        actual_raw: np.ndarray,
        expected_extended: np.ndarray,
        actual_extended: np.ndarray,
        *,
        probability_variant: str,
        scope: str,
        group_values: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        base = {"probability_variant": probability_variant, "scope": scope, **dict(group_values or {})}
        for index, spec in enumerate(self.config.outcomes):
            ic, support, status = _spearman(expected_raw[:, index], actual_raw[:, index])
            rows.append(
                {
                    **base,
                    "component": spec.name,
                    "outcome_column": spec.column,
                    "direction": spec.direction,
                    "higher_is_better": spec.higher_is_better,
                    "ic": ic,
                    "rows": int(len(expected_raw)),
                    "finite_rows": support,
                    "status": status,
                }
            )
        pooled_ic, pooled_support, pooled_status = _spearman(expected_extended[:, -1], actual_extended[:, -1])
        rows.append(
            {
                **base,
                "component": "pooled_economic_utility",
                "outcome_column": "train_robust_scaled_weighted_outcome_vector",
                "direction": 1.0,
                "higher_is_better": True,
                "ic": pooled_ic,
                "rows": int(len(expected_raw)),
                "finite_rows": pooled_support,
                "status": pooled_status,
            }
        )
        return rows

    def _group_ic(
        self,
        work: pd.DataFrame,
        expected_raw: np.ndarray,
        actual_raw: np.ndarray,
        expected_extended: np.ndarray,
        actual_extended: np.ndarray,
        *,
        probability_variant: str,
        group_column: str,
        scope: str,
        minimum_rows: int,
    ) -> pd.DataFrame:
        result: list[dict[str, Any]] = []
        values = work[group_column]
        for group_value, positions in values.groupby(values, dropna=False, observed=True).groups.items():
            index = np.asarray(list(positions), dtype=np.int64)
            metadata = {group_column: group_value}
            if len(index) < minimum_rows:
                for spec in (*self.config.outcomes, EconomicOutcomeSpec("pooled_economic_utility", "", 1.0, 1.0)):
                    result.append(
                        {
                            "probability_variant": probability_variant,
                            "scope": scope,
                            **metadata,
                            "component": spec.name,
                            "rows": int(len(index)),
                            "finite_rows": 0,
                            "ic": np.nan,
                            "status": "below_minimum_group_rows",
                        }
                    )
                continue
            result.extend(
                self._ic_rows(
                    expected_raw[index],
                    actual_raw[index],
                    expected_extended[index],
                    actual_extended[index],
                    probability_variant=probability_variant,
                    scope=scope,
                    group_values=metadata,
                )
            )
        return pd.DataFrame(result)

    def _probability_class_ic(
        self,
        probabilities: np.ndarray,
        actual_raw: np.ndarray,
        actual_extended: np.ndarray,
        *,
        probability_variant: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for class_index, class_name in enumerate(self.classes):
            probability = probabilities[:, class_index]
            for outcome_index, spec in enumerate(self.config.outcomes):
                # Multiply realised adverse components by -1 so probability
                # ranking retains the same positive-is-good IC semantics.
                ic, support, status = _spearman(probability, spec.direction * actual_raw[:, outcome_index])
                rows.append(
                    {
                        "probability_variant": probability_variant,
                        "probability_class": class_name,
                        "component": spec.name,
                        "outcome_column": spec.column,
                        "direction": spec.direction,
                        "higher_is_better": spec.higher_is_better,
                        "ic": ic,
                        "rows": int(len(probability)),
                        "finite_rows": support,
                        "status": status,
                    }
                )
            pooled_ic, pooled_support, pooled_status = _spearman(probability, actual_extended[:, -1])
            rows.append(
                {
                    "probability_variant": probability_variant,
                    "probability_class": class_name,
                    "component": "pooled_economic_utility",
                    "outcome_column": "train_robust_scaled_weighted_outcome_vector",
                    "direction": 1.0,
                    "higher_is_better": True,
                    "ic": pooled_ic,
                    "rows": int(len(probability)),
                    "finite_rows": pooled_support,
                    "status": pooled_status,
                }
            )
        return pd.DataFrame(rows)

    def evaluate(
        self,
        oos_frame: pd.DataFrame,
        probability_variants: Mapping[str, Any],
    ) -> EconomicICDiagnostics:
        """Evaluate raw/calibrated OOS probability vectors against outcomes.

        ``oos_frame`` must provide realised outcomes only for diagnostics.
        An optional OOS realised class label is used only for a reporting
        breakdown.  It is never read by the expected-outcome mapper and never
        mutates fitted state.
        """

        self._require_fitted()
        self._require_outcomes(oos_frame, scope="oos_frame")
        required_groups = (self.config.columns.timestamp, self.config.columns.side, self.config.columns.symbol)
        missing = [column for column in required_groups if column not in oos_frame]
        if missing:
            raise KeyError(f"oos_frame is missing diagnostic grouping columns: {missing}")
        if not probability_variants:
            raise ValueError("at least one probability variant is required")
        if "raw" not in probability_variants:
            raise ValueError("probability_variants must include a raw probability matrix")

        work = oos_frame.reset_index(drop=True).copy()
        work["__month__"] = _month_utc(work[self.config.columns.timestamp])
        timestamp = pd.to_datetime(work[self.config.columns.timestamp], utc=True, errors="coerce")
        work["__timestamp__"] = timestamp
        quality_rows: list[dict[str, Any]] = []
        global_frames: list[pd.DataFrame] = []
        probability_frames: list[pd.DataFrame] = []
        true_archetype_frames: list[pd.DataFrame] = []
        timestamp_frames: list[pd.DataFrame] = []
        timestamp_summary_rows: list[dict[str, Any]] = []
        month_frames: list[pd.DataFrame] = []
        side_frames: list[pd.DataFrame] = []
        symbol_frames: list[pd.DataFrame] = []
        symbol_summary_rows: list[dict[str, Any]] = []

        for variant, raw_probabilities in probability_variants.items():
            probabilities = _as_probability_matrix(raw_probabilities, len(work), len(self.classes))
            expected_raw, actual_raw, expected_extended, actual_extended = self._expected_and_actual(work, probabilities)
            global_frames.append(
                pd.DataFrame(
                    self._ic_rows(
                        expected_raw,
                        actual_raw,
                        expected_extended,
                        actual_extended,
                        probability_variant=str(variant),
                        scope="all_oos_rows",
                    )
                )
            )
            probability_frames.append(
                self._probability_class_ic(
                    probabilities,
                    actual_raw,
                    actual_extended,
                    probability_variant=str(variant),
                )
            )
            if self.config.columns.class_label in work:
                true_archetype_frames.append(
                    self._group_ic(
                        work,
                        expected_raw,
                        actual_raw,
                        expected_extended,
                        actual_extended,
                        probability_variant=str(variant),
                        group_column=self.config.columns.class_label,
                        scope="within_realized_archetype_diagnostic_only",
                        minimum_rows=self.config.min_group_rows,
                    )
                )
            timestamp_table = self._group_ic(
                work,
                expected_raw,
                actual_raw,
                expected_extended,
                actual_extended,
                probability_variant=str(variant),
                group_column="__timestamp__",
                scope="per_timestamp_cross_section",
                minimum_rows=self.config.min_group_rows,
            )
            timestamp_frames.append(timestamp_table)
            for component, group in timestamp_table.groupby("component", observed=True):
                valid = group.loc[group["status"].eq("ok") & group["ic"].notna()]
                timestamp_summary_rows.append(
                    {
                        "probability_variant": str(variant),
                        "component": component,
                        "timestamps_total": int(len(group)),
                        "timestamps_with_valid_ic": int(len(valid)),
                        "mean_cross_sectional_ic": float(valid["ic"].mean()) if len(valid) else np.nan,
                        "median_cross_sectional_ic": float(valid["ic"].median()) if len(valid) else np.nan,
                        "mean_cross_sectional_support": float(valid["finite_rows"].mean()) if len(valid) else np.nan,
                        "minimum_group_rows": self.config.min_group_rows,
                    }
                )
            month_frames.append(
                self._group_ic(
                    work,
                    expected_raw,
                    actual_raw,
                    expected_extended,
                    actual_extended,
                    probability_variant=str(variant),
                    group_column="__month__",
                    scope="monthly",
                    minimum_rows=self.config.min_group_rows,
                )
            )
            side_frames.append(
                self._group_ic(
                    work,
                    expected_raw,
                    actual_raw,
                    expected_extended,
                    actual_extended,
                    probability_variant=str(variant),
                    group_column=self.config.columns.side,
                    scope="side",
                    minimum_rows=self.config.min_group_rows,
                )
            )
            symbol_table = self._group_ic(
                work,
                expected_raw,
                actual_raw,
                expected_extended,
                actual_extended,
                probability_variant=str(variant),
                group_column=self.config.columns.symbol,
                scope="within_symbol",
                minimum_rows=self.config.min_symbol_rows,
            )
            symbol_frames.append(symbol_table)
            for component, group in symbol_table.groupby("component", observed=True):
                valid = group.loc[group["status"].eq("ok") & group["ic"].notna()]
                weights = valid["finite_rows"].to_numpy(dtype=np.float64)
                symbol_summary_rows.append(
                    {
                        "probability_variant": str(variant),
                        "component": component,
                        "symbols_total": int(len(group)),
                        "symbols_with_valid_ic": int(len(valid)),
                        "symbol_neutral_weighted_ic": float(np.average(valid["ic"], weights=weights)) if len(valid) and weights.sum() > 0.0 else np.nan,
                        "symbol_neutral_equal_weight_ic": float(valid["ic"].mean()) if len(valid) else np.nan,
                        "total_symbol_ic_weight": float(weights.sum()),
                        "minimum_symbol_rows": self.config.min_symbol_rows,
                        "interpretation": "diagnostic_only_weighted_aggregation_of_within_symbol_oos_ics",
                    }
                )
            quality_rows.append(
                {
                    "probability_variant": str(variant),
                    "rows": int(len(work)),
                    "rows_with_valid_timestamp": int(work["__timestamp__"].notna().sum()),
                    "probability_rows_normalized": True,
                    "oos_outcomes_fit_priors": False,
                    "oos_class_labels_used_only_for_reporting": self.config.columns.class_label in work,
                    "class_labels_consumed_from_oos_for_fit": False,
                }
            )

        prior_rows: list[dict[str, Any]] = []
        centroid_rows: list[dict[str, Any]] = []
        for class_index, class_name in enumerate(self.classes):
            prior_row: dict[str, Any] = {
                "class_name": class_name,
                "train_rows": int(self.class_support_[class_index]),
                "prior_source": "class_train_mean" if self.class_support_[class_index] else "global_train_mean_fallback_no_class_support",
            }
            centroid_row: dict[str, Any] = {"class_name": class_name, "train_rows": int(self.class_support_[class_index])}
            for outcome_index, spec in enumerate(self.config.outcomes):
                prior_row[f"prior_{spec.name}"] = float(self.class_priors_[class_index, outcome_index])
                centroid_row[f"robust_centroid_{spec.name}"] = float(self.class_centroids_[class_index, outcome_index])
                centroid_row[f"train_median_{spec.name}"] = float(self.outcome_median_[outcome_index])
                centroid_row[f"train_iqr_{spec.name}"] = float(self.outcome_iqr_[outcome_index])
            prior_rows.append(prior_row)
            centroid_rows.append(centroid_row)

        return EconomicICDiagnostics(
            global_ic=pd.concat(global_frames, ignore_index=True),
            probability_class_ic=pd.concat(probability_frames, ignore_index=True),
            true_archetype_ic=(
                pd.concat(true_archetype_frames, ignore_index=True)
                if true_archetype_frames
                else pd.DataFrame(
                    columns=[
                        "probability_variant",
                        "scope",
                        self.config.columns.class_label,
                        "component",
                        "ic",
                        "rows",
                        "finite_rows",
                        "status",
                    ]
                )
            ),
            timestamp_ic=pd.concat(timestamp_frames, ignore_index=True),
            timestamp_summary=pd.DataFrame(timestamp_summary_rows),
            monthly_ic=pd.concat(month_frames, ignore_index=True),
            side_ic=pd.concat(side_frames, ignore_index=True),
            symbol_ic=pd.concat(symbol_frames, ignore_index=True),
            symbol_neutral_ic=pd.DataFrame(symbol_summary_rows),
            class_priors=pd.DataFrame(prior_rows),
            class_centroids=pd.DataFrame(centroid_rows),
            provenance=dict(self.provenance_),
            quality=pd.DataFrame(quality_rows),
        )


__all__ = [
    "DEFAULT_OUTCOME_SPECS",
    "EconomicICColumns",
    "EconomicICConfig",
    "EconomicICDiagnostics",
    "EconomicOutcomeSpec",
    "PathArchetypeEconomicIC",
]
