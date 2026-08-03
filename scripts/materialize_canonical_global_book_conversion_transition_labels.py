#!/usr/bin/env python3
"""Materialize before/after conversion labels for causal mapped global books.

Each window forms exactly one pooled global top-k book from the authoritative
21-day causal mapped EV, with candidate-ID tie breaking.  No timestamp, side,
asset or regime quota participates.  All emitted metrics are outcome labels or
audit metadata and are unavailable until the selected candidates resolve.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scripts.materialize_canonical_economic_conversion_transition_labels import (
        EXIT_CLASSES,
        POST_WINDOW_PUBLICATION_LAG,
        robust_mean,
        sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from materialize_canonical_economic_conversion_transition_labels import (
        EXIT_CLASSES,
        POST_WINDOW_PUBLICATION_LAG,
        robust_mean,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
MAPPING_SOURCE = (
    ROOT
    / "data_perp/artifacts/historical_causal_score_economics_mapping_20260729_v1/"
    "canonical_base__score_base_alpha"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_conversion_transition_labels_20260729_v1"
)
SCHEMA = "canonical_global_book_conversion_transition_labels_v1"
HORIZONS = ((12, "primary"), (3, "auxiliary"))
BOOK_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 1.00)
GLOBAL_EV_BAND_EDGES = (0.00, 0.50, 0.80, 0.90, 0.95, 1.00)
GLOBAL_EV_BANDS = ("B0", "B1", "B2", "B3", "B4")
GLOBAL_EV_REFERENCE_DAYS = 21
MINIMUM_GLOBAL_EV_REFERENCE_ROWS = 1_000
# Historical exact-1m ledgers are stored as float32.  Gross-cost-net identity
# can therefore differ by one float32 ULP after Parquet round-tripping.
ACCOUNTING_ATOL = 1e-7
REQUIRED_COLUMNS = (
    "candidate_id",
    "__symbol__",
    "side_name",
    "execution_decision_utc",
    "execution_label_end_utc",
    "candidate_month",
    "mapped_eligible",
    "mapped_direct_net",
    "map_reference_rows",
    "map_side_reference_rows",
    "map_cell_reference_rows",
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
    "execution_exit_class",
    "opportunity_gross_above_cost_0bps",
    "opportunity_gross_above_cost_25bps",
)


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _mapping_hashes(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    paths = (
        root / "causal_mapped_candidates.parquet",
        root / "causal_snapshot_audit.parquet",
        root / "manifest.json",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"canonical mapping artifact is incomplete: {missing}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "causal_score_economics_conversion_mapping_v1":
        raise ValueError(f"unexpected mapping schema: {manifest.get('schema')}")
    contract = manifest.get("causal_contract", {})
    if (
        contract.get("window_days") != 21
        or contract.get("reference_rule")
        != "execution_label_end_utc < snapshot"
    ):
        raise ValueError("canonical 21-day resolved-before-snapshot mapping required")
    selection = manifest.get("selection_contract", {})
    if (
        selection.get("primary") != "one pooled global top-k"
        or not selection.get("not_per_timestamp")
        or selection.get("tie_break") != "candidate_id ascending"
    ):
        raise ValueError("canonical pooled-global selection contract changed")
    embedded = manifest.get("outputs", {})
    expected_mapped = embedded.get("mapped", {}).get("sha256")
    expected_audit = embedded.get("audit", {}).get("sha256")
    if expected_mapped != sha256(root / "causal_mapped_candidates.parquet"):
        raise ValueError("canonical mapping parquet differs from embedded checksum")
    if expected_audit != sha256(root / "causal_snapshot_audit.parquet"):
        raise ValueError("canonical mapping audit differs from embedded checksum")
    audit = pd.read_parquet(
        root / "causal_snapshot_audit.parquet",
        columns=["snapshot_utc", "reference_label_end_max_utc"],
    )
    snapshot = pd.to_datetime(audit["snapshot_utc"], utc=True, errors="raise")
    reference_max = pd.to_datetime(
        audit["reference_label_end_max_utc"], utc=True, errors="coerce"
    )
    if not reference_max.loc[reference_max.notna()].lt(
        snapshot.loc[reference_max.notna()]
    ).all():
        raise ValueError("mapping audit contains a noncausal reference outcome")
    return manifest, {str(path): sha256(path) for path in paths}


def _normalise_mapping(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"mapping source lacks global-book fields: {missing}")
    work = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["__symbol__"] = work["__symbol__"].astype(str)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise ValueError("mapping source contains noncanonical sides")
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
        if not work[column].dt.floor("h").eq(work[column]).all():
            raise ValueError(f"{column} is not UTC-hour aligned")
    for column in (
        "mapped_direct_net",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "opportunity_gross_above_cost_0bps",
        "opportunity_gross_above_cost_25bps",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work["candidate_id"].duplicated().any():
        raise ValueError("mapping candidate identity is not unique")
    warmup_rows = int((~work["mapped_eligible"].astype(bool)).sum())
    eligible = work.loc[work["mapped_eligible"].astype(bool)].copy()
    if not np.isfinite(eligible["mapped_direct_net"]).all():
        raise ValueError("eligible mapping rows contain non-finite common-unit EV")
    if not np.isfinite(
        eligible[
            [
                "execution_gross_ev_12h",
                "execution_cost_return",
                "execution_net_ev_12h",
            ]
        ].to_numpy(float)
    ).all():
        raise ValueError("eligible mapping rows contain non-finite exact economics")
    if not np.allclose(
        eligible["execution_gross_ev_12h"]
        - eligible["execution_cost_return"],
        eligible["execution_net_ev_12h"],
        rtol=0.0,
        atol=ACCOUNTING_ATOL,
    ):
        raise ValueError("exact gross-cost-net accounting changed")
    eligible = eligible.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    return eligible, {
        "input_rows": int(len(work)),
        "mapped_eligible_rows": int(len(eligible)),
        "warmup_unmapped_rows": warmup_rows,
    }


def add_causal_global_mapped_ev_coordinates(
    eligible: pd.DataFrame,
    *,
    window_days: int = GLOBAL_EV_REFERENCE_DAYS,
    minimum_reference_rows: int = MINIMUM_GLOBAL_EV_REFERENCE_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach global prior-candidate mapped-EV percentiles without outcomes."""

    if int(window_days) < 1 or int(minimum_reference_rows) < 1:
        raise ValueError("global mapped-EV coordinate contract must be positive")
    work = eligible.copy()
    work["causal_global_mapped_ev_percentile"] = np.nan
    work["causal_global_mapped_ev_band"] = "UNAVAILABLE"
    work["causal_global_mapped_ev_reference_rows"] = 0
    work["causal_global_mapped_ev_cutoff_p90"] = np.nan
    work["causal_global_mapped_ev_margin_to_p90"] = np.nan
    decision_day = work["execution_decision_utc"].dt.floor("D")
    audits: list[dict[str, Any]] = []
    for day, indices in work.groupby(decision_day, sort=True).groups.items():
        day = pd.Timestamp(day)
        lower = day - pd.Timedelta(days=int(window_days))
        reference = work.loc[
            work["execution_decision_utc"].ge(lower)
            & work["execution_decision_utc"].lt(day)
        ]
        current = work.loc[indices]
        reference_rows = int(len(reference))
        available = reference_rows >= int(minimum_reference_rows)
        cutoff = float("nan")
        if available:
            sorted_reference = np.sort(
                reference["mapped_direct_net"].to_numpy(float)
            )
            values = current["mapped_direct_net"].to_numpy(float)
            left = np.searchsorted(sorted_reference, values, side="left")
            right = np.searchsorted(sorted_reference, values, side="right")
            percentile = (left + right) / (2.0 * reference_rows)
            percentile = np.clip(percentile, 0.0, 1.0)
            band_index = np.searchsorted(
                np.asarray(GLOBAL_EV_BAND_EDGES[1:-1], dtype=float),
                percentile,
                side="right",
            )
            bands = np.asarray(GLOBAL_EV_BANDS, dtype=object)[band_index]
            cutoff = float(np.quantile(sorted_reference, 0.90))
            work.loc[indices, "causal_global_mapped_ev_percentile"] = percentile
            work.loc[indices, "causal_global_mapped_ev_band"] = bands
            work.loc[
                indices, "causal_global_mapped_ev_reference_rows"
            ] = reference_rows
            work.loc[indices, "causal_global_mapped_ev_cutoff_p90"] = cutoff
            work.loc[indices, "causal_global_mapped_ev_margin_to_p90"] = (
                values - cutoff
            )
        audits.append(
            {
                "snapshot_utc": day,
                "reference_window_start_utc": lower,
                "reference_window_end_utc": day,
                "reference_rows": reference_rows,
                "coordinate_available": available,
                "current_rows": int(len(current)),
                "mapped_ev_cutoff_p90": cutoff,
            }
        )
    available = work["causal_global_mapped_ev_band"].ne("UNAVAILABLE")
    if available.any():
        if not work.loc[
            available, "causal_global_mapped_ev_percentile"
        ].between(0.0, 1.0).all():
            raise ValueError("causal global mapped-EV percentile is invalid")
        if not work.loc[
            available, "causal_global_mapped_ev_band"
        ].isin(GLOBAL_EV_BANDS).all():
            raise ValueError("causal global mapped-EV band is invalid")
    return work, pd.DataFrame.from_records(audits)


def _stable_select(
    frame: pd.DataFrame, *, score_column: str, fraction: float
) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("book fraction must be in (0,1]")
    count = max(1, int(math.ceil(float(fraction) * len(frame))))
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -score))
    return frame.iloc[order[:count]].copy()


def _window_book_metrics(
    population: pd.DataFrame,
    *,
    fraction: float,
    window_end: pd.Timestamp,
    preordered: pd.DataFrame | None = None,
) -> dict[str, Any]:
    population_support = int(len(population))
    result: dict[str, Any] = {
        "population_candidate_support": population_support,
        "selected_candidate_support": 0,
        "window_missing_support_flag": population_support == 0,
        "mapped_score_cutoff": float("nan"),
        "mapped_score_mean": float("nan"),
        "mapped_score_std": float("nan"),
        "mean_conversion_residual": float("nan"),
        "cutoff_plateau_population_rows": 0,
        "cutoff_plateau_selected_rows": 0,
        "map_reference_rows_min": 0,
        "map_side_reference_rows_min": 0,
        "map_cell_reference_rows_min": 0,
        "direct_mean_net": float("nan"),
        "mean_gross": float("nan"),
        "mean_cost": float("nan"),
        "opportunity_probability_0bps": float("nan"),
        "opportunity_probability_25bps": float("nan"),
        "positive_net_contribution": float("nan"),
        "positive_net_contribution_robust_mean": float("nan"),
        "loss_net_contribution": float("nan"),
        "loss_net_contribution_robust_mean": float("nan"),
        "long_share": float("nan"),
        "unique_assets": 0,
        "largest_asset_share": float("nan"),
        "raw_components_reconcile_direct_mean_flag": False,
        "mapped_plus_residual_reconciles_direct_mean_flag": False,
        "band_contribution_complete_flag": False,
        "selection_available_utc": window_end,
        "target_available_utc": pd.NaT,
    }
    for exit_class in EXIT_CLASSES:
        result[f"p_exit_{exit_class}"] = float("nan")
        result[f"conditional_net_{exit_class}"] = float("nan")
        result[f"exit_{exit_class}_support"] = 0
    for band in GLOBAL_EV_BANDS:
        result[f"band_{band}_selected_support"] = 0
        result[f"band_{band}_mapped_ev_contribution"] = float("nan")
        result[f"band_{band}_conversion_residual_contribution"] = float("nan")
        result[f"band_{band}_net_contribution"] = float("nan")
    if not population_support:
        return result
    if preordered is None:
        selected = _stable_select(
            population, score_column="mapped_direct_net", fraction=fraction
        )
    else:
        if len(preordered) != len(population):
            raise ValueError("preordered global-book population length changed")
        count = max(1, int(math.ceil(float(fraction) * len(preordered))))
        selected = preordered.iloc[:count].copy()
    score = selected["mapped_direct_net"].to_numpy(float)
    cutoff = float(score.min())
    population_at_cutoff = population["mapped_direct_net"].eq(cutoff)
    selected_at_cutoff = selected["mapped_direct_net"].eq(cutoff)
    net = selected["execution_net_ev_12h"].to_numpy(float)
    residual = net - score
    positive = np.maximum(net, 0.0)
    loss = np.maximum(-net, 0.0)
    positive_mean = float(positive.mean())
    loss_mean = float(loss.mean())
    direct = float(net.mean())
    result.update(
        {
            "selected_candidate_support": int(len(selected)),
            "window_missing_support_flag": False,
            "mapped_score_cutoff": cutoff,
            "mapped_score_mean": float(score.mean()),
            "mapped_score_std": float(score.std(ddof=0)),
            "mean_conversion_residual": float(residual.mean()),
            "cutoff_plateau_population_rows": int(population_at_cutoff.sum()),
            "cutoff_plateau_selected_rows": int(selected_at_cutoff.sum()),
            "map_reference_rows_min": int(selected["map_reference_rows"].min()),
            "map_side_reference_rows_min": int(
                selected["map_side_reference_rows"].min()
            ),
            "map_cell_reference_rows_min": int(
                selected["map_cell_reference_rows"].min()
            ),
            "direct_mean_net": direct,
            "mean_gross": float(selected["execution_gross_ev_12h"].mean()),
            "mean_cost": float(selected["execution_cost_return"].mean()),
            "opportunity_probability_0bps": float(
                selected["opportunity_gross_above_cost_0bps"].mean()
            ),
            "opportunity_probability_25bps": float(
                selected["opportunity_gross_above_cost_25bps"].mean()
            ),
            "positive_net_contribution": positive_mean,
            "positive_net_contribution_robust_mean": robust_mean(positive),
            "loss_net_contribution": loss_mean,
            "loss_net_contribution_robust_mean": robust_mean(loss),
            "long_share": float(selected["side_name"].eq("long").mean()),
            "unique_assets": int(selected["__symbol__"].nunique()),
            "largest_asset_share": float(
                selected["__symbol__"].value_counts(normalize=True).max()
            ),
            "raw_components_reconcile_direct_mean_flag": bool(
                np.isclose(
                    positive_mean - loss_mean,
                    direct,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "mapped_plus_residual_reconciles_direct_mean_flag": bool(
                np.isclose(
                    float(score.mean()) + float(residual.mean()),
                    direct,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "target_available_utc": max(
                window_end,
                selected["execution_label_end_utc"].max()
                + POST_WINDOW_PUBLICATION_LAG,
            ),
        }
    )
    band_complete = (
        "causal_global_mapped_ev_band" in selected
        and selected["causal_global_mapped_ev_band"].isin(GLOBAL_EV_BANDS).all()
    )
    result["band_contribution_complete_flag"] = bool(band_complete)
    if band_complete:
        denominator = float(len(selected))
        for band in GLOBAL_EV_BANDS:
            mask = selected["causal_global_mapped_ev_band"].eq(band).to_numpy()
            result[f"band_{band}_selected_support"] = int(mask.sum())
            result[f"band_{band}_mapped_ev_contribution"] = float(
                score[mask].sum() / denominator
            )
            result[f"band_{band}_conversion_residual_contribution"] = float(
                residual[mask].sum() / denominator
            )
            result[f"band_{band}_net_contribution"] = float(
                net[mask].sum() / denominator
            )
    for exit_class in EXIT_CLASSES:
        mask = selected["execution_exit_class"].astype(str).eq(exit_class)
        support = int(mask.sum())
        result[f"exit_{exit_class}_support"] = support
        result[f"p_exit_{exit_class}"] = float(mask.mean())
        result[f"conditional_net_{exit_class}"] = (
            float(selected.loc[mask, "execution_net_ev_12h"].mean())
            if support
            else float("nan")
        )
    return result


def _global_hour_completeness(
    observed_hours: set[pd.Timestamp],
    anchor: pd.Timestamp,
    horizon_hours: int,
) -> tuple[int, bool, int, bool]:
    before = pd.date_range(
        anchor - pd.Timedelta(hours=horizon_hours),
        periods=horizon_hours,
        freq="h",
        tz="UTC",
    )
    after = pd.date_range(
        anchor, periods=horizon_hours, freq="h", tz="UTC"
    )
    before_count = sum(stamp in observed_hours for stamp in before)
    after_count = sum(stamp in observed_hours for stamp in after)
    return (
        before_count,
        before_count == horizon_hours,
        after_count,
        after_count == horizon_hours,
    )


def _stable_order(frame: pd.DataFrame) -> pd.DataFrame:
    score = frame["mapped_direct_net"].to_numpy(float)
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -score))
    return frame.iloc[order].copy()


def materialize_global_book_labels(
    mapping: pd.DataFrame, *, prepared: bool = False
) -> pd.DataFrame:
    if prepared:
        rows = mapping.copy()
    else:
        rows, _ = _normalise_mapping(mapping)
        rows, _ = add_causal_global_mapped_ev_coordinates(rows)
    hours = pd.DatetimeIndex(sorted(rows["execution_decision_utc"].unique()))
    if not len(hours):
        raise ValueError("eligible causal mapping population is empty")
    observed_hours = set(hours)
    stamps = rows["execution_decision_utc"].to_numpy(dtype="datetime64[ns]")
    records: list[dict[str, Any]] = []
    for horizon_hours, horizon_role in HORIZONS:
        horizon = pd.Timedelta(hours=horizon_hours)
        for anchor in hours:
            before_left = np.searchsorted(
                stamps, (anchor - horizon).to_datetime64(), side="left"
            )
            before_right = np.searchsorted(
                stamps, anchor.to_datetime64(), side="left"
            )
            after_left = before_right
            after_right = np.searchsorted(
                stamps, (anchor + horizon).to_datetime64(), side="left"
            )
            before_population = rows.iloc[before_left:before_right]
            after_population = rows.iloc[after_left:after_right]
            before_ordered = (
                _stable_order(before_population)
                if len(before_population)
                else before_population
            )
            after_ordered = (
                _stable_order(after_population)
                if len(after_population)
                else after_population
            )
            before_hours, before_complete, after_hours, after_complete = (
                _global_hour_completeness(
                    observed_hours, anchor, horizon_hours
                )
            )
            for fraction in BOOK_FRACTIONS:
                before = _window_book_metrics(
                    before_population,
                    fraction=fraction,
                    window_end=anchor,
                    preordered=before_ordered,
                )
                after = _window_book_metrics(
                    after_population,
                    fraction=fraction,
                    window_end=anchor + horizon,
                    preordered=after_ordered,
                )
                record: dict[str, Any] = {
                    "cohort_anchor_utc": anchor,
                    "horizon_hours": int(horizon_hours),
                    "horizon_role": horizon_role,
                    "book_fraction": float(fraction),
                    "before_window_start_utc": anchor - horizon,
                    "before_window_end_utc": anchor,
                    "after_window_start_utc": anchor,
                    "after_window_end_utc": anchor + horizon,
                    "before_global_hour_support": before_hours,
                    "after_global_hour_support": after_hours,
                    "before_global_hour_complete_flag": before_complete,
                    "after_global_hour_complete_flag": after_complete,
                    "outcome_only_not_model_feature": True,
                    "selection_contract": "one_pooled_global_mapped_direct_net",
                }
                record.update(
                    {f"before_{name}": value for name, value in before.items()}
                )
                record.update(
                    {f"after_{name}": value for name, value in after.items()}
                )
                for metric in (
                    "mapped_score_cutoff",
                    "mapped_score_mean",
                    "mean_conversion_residual",
                    "direct_mean_net",
                    "mean_gross",
                    "mean_cost",
                    "opportunity_probability_0bps",
                    "opportunity_probability_25bps",
                    "positive_net_contribution",
                    "positive_net_contribution_robust_mean",
                    "loss_net_contribution",
                    "loss_net_contribution_robust_mean",
                    "long_share",
                    *[f"p_exit_{exit_class}" for exit_class in EXIT_CLASSES],
                    *[
                        f"conditional_net_{exit_class}"
                        for exit_class in EXIT_CLASSES
                    ],
                ):
                    record[f"delta_{metric}"] = after[metric] - before[metric]
                records.append(record)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["horizon_hours", "cohort_anchor_utc", "book_fraction"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _band_window_metrics(
    population: pd.DataFrame, *, window_end: pd.Timestamp
) -> dict[str, Any]:
    support = int(len(population))
    result: dict[str, Any] = {
        "candidate_support": support,
        "window_missing_support_flag": support == 0,
        "mean_mapped_ev": float("nan"),
        "mean_realized_net": float("nan"),
        "mean_conversion_residual": float("nan"),
        "opportunity_probability_0bps": float("nan"),
        "positive_net_contribution": float("nan"),
        "loss_net_contribution": float("nan"),
        "long_share": float("nan"),
        "mapped_plus_residual_reconciles_net_flag": False,
        "positive_minus_loss_reconciles_net_flag": False,
        "target_available_utc": pd.NaT,
    }
    for exit_class in EXIT_CLASSES:
        result[f"p_exit_{exit_class}"] = float("nan")
    if not support:
        return result
    mapped = population["mapped_direct_net"].to_numpy(float)
    net = population["execution_net_ev_12h"].to_numpy(float)
    residual = net - mapped
    positive = np.maximum(net, 0.0)
    loss = np.maximum(-net, 0.0)
    result.update(
        {
            "mean_mapped_ev": float(mapped.mean()),
            "mean_realized_net": float(net.mean()),
            "mean_conversion_residual": float(residual.mean()),
            "opportunity_probability_0bps": float(
                population["opportunity_gross_above_cost_0bps"].mean()
            ),
            "positive_net_contribution": float(positive.mean()),
            "loss_net_contribution": float(loss.mean()),
            "long_share": float(population["side_name"].eq("long").mean()),
            "mapped_plus_residual_reconciles_net_flag": bool(
                np.isclose(
                    mapped.mean() + residual.mean(),
                    net.mean(),
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "positive_minus_loss_reconciles_net_flag": bool(
                np.isclose(
                    positive.mean() - loss.mean(),
                    net.mean(),
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "target_available_utc": max(
                window_end,
                population["execution_label_end_utc"].max()
                + POST_WINDOW_PUBLICATION_LAG,
            ),
        }
    )
    for exit_class in EXIT_CLASSES:
        result[f"p_exit_{exit_class}"] = float(
            population["execution_exit_class"].astype(str).eq(exit_class).mean()
        )
    return result


def materialize_global_ev_band_labels(
    mapping: pd.DataFrame, *, prepared: bool = False
) -> pd.DataFrame:
    if prepared:
        rows = mapping.copy()
    else:
        rows, _ = _normalise_mapping(mapping)
        rows, _ = add_causal_global_mapped_ev_coordinates(rows)
    rows = rows.loc[
        rows["causal_global_mapped_ev_band"].isin(GLOBAL_EV_BANDS)
    ].copy()
    hours = pd.DatetimeIndex(sorted(rows["execution_decision_utc"].unique()))
    if not len(hours):
        raise ValueError("causal global mapped-EV bands have no available rows")
    observed_hours = set(hours)
    records: list[dict[str, Any]] = []
    for horizon_hours, horizon_role in HORIZONS:
        horizon = pd.Timedelta(hours=horizon_hours)
        for band in GLOBAL_EV_BANDS:
            cohort = rows.loc[
                rows["causal_global_mapped_ev_band"].eq(band)
            ].sort_values(["execution_decision_utc", "candidate_id"], kind="stable")
            stamps = cohort["execution_decision_utc"].to_numpy(
                dtype="datetime64[ns]"
            )
            for anchor in hours:
                before_left = np.searchsorted(
                    stamps, (anchor - horizon).to_datetime64(), side="left"
                )
                before_right = np.searchsorted(
                    stamps, anchor.to_datetime64(), side="left"
                )
                after_left = before_right
                after_right = np.searchsorted(
                    stamps, (anchor + horizon).to_datetime64(), side="left"
                )
                before = _band_window_metrics(
                    cohort.iloc[before_left:before_right], window_end=anchor
                )
                after = _band_window_metrics(
                    cohort.iloc[after_left:after_right],
                    window_end=anchor + horizon,
                )
                before_hours, before_complete, after_hours, after_complete = (
                    _global_hour_completeness(
                        observed_hours, anchor, horizon_hours
                    )
                )
                record: dict[str, Any] = {
                    "cohort_anchor_utc": anchor,
                    "horizon_hours": int(horizon_hours),
                    "horizon_role": horizon_role,
                    "global_common_ev_band": band,
                    "before_global_hour_support": before_hours,
                    "after_global_hour_support": after_hours,
                    "before_global_hour_complete_flag": before_complete,
                    "after_global_hour_complete_flag": after_complete,
                    "outcome_only_not_model_feature": True,
                }
                record.update(
                    {f"before_{name}": value for name, value in before.items()}
                )
                record.update(
                    {f"after_{name}": value for name, value in after.items()}
                )
                for metric in (
                    "mean_mapped_ev",
                    "mean_realized_net",
                    "mean_conversion_residual",
                    "opportunity_probability_0bps",
                    "positive_net_contribution",
                    "loss_net_contribution",
                    "long_share",
                    *[f"p_exit_{exit_class}" for exit_class in EXIT_CLASSES],
                ):
                    record[f"delta_{metric}"] = after[metric] - before[metric]
                records.append(record)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["horizon_hours", "cohort_anchor_utc", "global_common_ev_band"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _support_summary(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby(
            ["horizon_hours", "horizon_role", "book_fraction"],
            observed=True,
            sort=True,
        )
        .agg(
            rows=("cohort_anchor_utc", "size"),
            complete_before=("before_global_hour_complete_flag", "sum"),
            complete_after=("after_global_hour_complete_flag", "sum"),
            median_before_population=(
                "before_population_candidate_support",
                "median",
            ),
            median_after_population=(
                "after_population_candidate_support",
                "median",
            ),
            median_before_selected=(
                "before_selected_candidate_support",
                "median",
            ),
            median_after_selected=(
                "after_selected_candidate_support",
                "median",
            ),
            before_reconciliation_failures=(
                "before_raw_components_reconcile_direct_mean_flag",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            after_reconciliation_failures=(
                "after_raw_components_reconcile_direct_mean_flag",
                lambda values: int((~values.astype(bool)).sum()),
            ),
        )
        .reset_index()
    )


def _verify_book_reconciliation(labels: pd.DataFrame) -> dict[str, int]:
    checked = 0
    for phase in ("before", "after"):
        nonempty = labels[f"{phase}_selected_candidate_support"].gt(0)
        if not labels.loc[
            nonempty, f"{phase}_raw_components_reconcile_direct_mean_flag"
        ].astype(bool).all():
            raise ValueError(f"{phase} positive/loss book reconciliation fails")
        if not labels.loc[
            nonempty,
            f"{phase}_mapped_plus_residual_reconciles_direct_mean_flag",
        ].astype(bool).all():
            raise ValueError(f"{phase} mapped/residual book reconciliation fails")
        band_complete = nonempty & labels[
            f"{phase}_band_contribution_complete_flag"
        ].astype(bool)
        mapped_sum = sum(
            labels.loc[
                band_complete, f"{phase}_band_{band}_mapped_ev_contribution"
            ]
            for band in GLOBAL_EV_BANDS
        )
        residual_sum = sum(
            labels.loc[
                band_complete,
                f"{phase}_band_{band}_conversion_residual_contribution",
            ]
            for band in GLOBAL_EV_BANDS
        )
        net_sum = sum(
            labels.loc[
                band_complete, f"{phase}_band_{band}_net_contribution"
            ]
            for band in GLOBAL_EV_BANDS
        )
        if not np.allclose(
            mapped_sum,
            labels.loc[band_complete, f"{phase}_mapped_score_mean"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{phase} band mapped-EV contributions do not add")
        if not np.allclose(
            residual_sum,
            labels.loc[band_complete, f"{phase}_mean_conversion_residual"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{phase} band residual contributions do not add")
        if not np.allclose(
            net_sum,
            labels.loc[band_complete, f"{phase}_direct_mean_net"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{phase} band net contributions do not add")
        checked += int(band_complete.sum())
    return {"book_phase_rows_checked": checked}


def _verify_band_reconciliation(labels: pd.DataFrame) -> dict[str, int]:
    checked = 0
    for phase in ("before", "after"):
        nonempty = labels[f"{phase}_candidate_support"].gt(0)
        if not labels.loc[
            nonempty, f"{phase}_mapped_plus_residual_reconciles_net_flag"
        ].astype(bool).all():
            raise ValueError(f"{phase} mapped-band residual reconciliation fails")
        if not labels.loc[
            nonempty, f"{phase}_positive_minus_loss_reconciles_net_flag"
        ].astype(bool).all():
            raise ValueError(f"{phase} mapped-band contribution reconciliation fails")
        checked += int(nonempty.sum())
    return {"band_phase_rows_checked": checked}


def plan(mapping_source: Path, output: Path) -> dict[str, Any]:
    manifest, hashes = _mapping_hashes(mapping_source)
    return {
        "action": "PLAN_ONLY_NO_MATERIALIZATION",
        "schema": SCHEMA,
        "mapping_source": str(mapping_source),
        "output": str(output),
        "source_sha256": hashes,
        "causal_mapping_contract": manifest["causal_contract"],
        "book_fractions": BOOK_FRACTIONS,
        "horizons": HORIZONS,
        "selection": "one pooled global mapped_direct_net top-k per complete before/after window; candidate_id ascending tie-break",
        "warmup": "mapped_eligible=false rows are explicitly excluded, never imputed",
        "feature_surface": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mapping_source = Path(args.mapping_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(mapping_source, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    mapping_manifest, source_hashes = _mapping_hashes(mapping_source)
    source = pd.read_parquet(
        mapping_source / "causal_mapped_candidates.parquet",
        columns=list(REQUIRED_COLUMNS),
    )
    eligible, population_audit = _normalise_mapping(source)
    coordinates, coordinate_audit = add_causal_global_mapped_ev_coordinates(
        eligible
    )
    population_audit["coordinate_available_rows"] = int(
        coordinates["causal_global_mapped_ev_band"].isin(GLOBAL_EV_BANDS).sum()
    )
    population_audit["coordinate_unavailable_rows"] = int(
        coordinates["causal_global_mapped_ev_band"].eq("UNAVAILABLE").sum()
    )
    labels = materialize_global_book_labels(coordinates, prepared=True)
    band_labels = materialize_global_ev_band_labels(coordinates, prepared=True)
    reconciliation = {
        **_verify_book_reconciliation(labels),
        **_verify_band_reconciliation(band_labels),
    }
    summary = _support_summary(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    labels.to_parquet(
        temporary / "global_book_transition_labels.parquet",
        index=False,
        compression="zstd",
    )
    band_labels.to_parquet(
        temporary / "global_ev_band_transition_labels.parquet",
        index=False,
        compression="zstd",
    )
    coordinate_columns = [
        "candidate_id",
        "execution_decision_utc",
        "mapped_direct_net",
        "map_reference_rows",
        "map_side_reference_rows",
        "map_cell_reference_rows",
        "causal_global_mapped_ev_percentile",
        "causal_global_mapped_ev_band",
        "causal_global_mapped_ev_reference_rows",
        "causal_global_mapped_ev_cutoff_p90",
        "causal_global_mapped_ev_margin_to_p90",
    ]
    coordinates.loc[:, coordinate_columns].to_parquet(
        temporary / "candidate_global_mapped_ev_coordinates.parquet",
        index=False,
        compression="zstd",
    )
    coordinate_audit.to_parquet(
        temporary / "global_mapped_ev_coordinate_audit.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_parquet(
        temporary / "support_and_reconciliation_summary.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_OUTCOME_ONLY_GLOBAL_BOOK_LABEL_ARTIFACT",
        "source_artifacts_sha256": source_hashes,
        "source_mapping_contract": mapping_manifest["causal_contract"],
        "source_selection_contract": mapping_manifest["selection_contract"],
        "population_audit": population_audit,
        "reconciliation_audit": reconciliation,
        "rows": int(len(labels)),
        "band_rows": int(len(band_labels)),
        "book_fractions": BOOK_FRACTIONS,
        "horizons": HORIZONS,
        "global_ev_bands": {
            "edges": GLOBAL_EV_BAND_EDGES,
            "labels": GLOBAL_EV_BANDS,
            "reference_days": GLOBAL_EV_REFERENCE_DAYS,
            "minimum_reference_rows": MINIMUM_GLOBAL_EV_REFERENCE_ROWS,
        },
        "contracts": {
            "selection": "one pooled global top-k within each complete before/after window, ranked on causal mapped_direct_net with candidate_id ascending tie-break",
            "no_quotas": "no timestamp, side, asset, regime or calendar quota/backfill",
            "warmup": "mapped_eligible=false rows excluded and counted; no score imputation",
            "windows": "before [s-H,s), after [s,s+H) on execution-decision UTC",
            "availability": "max(window end, selected candidates' actual execution-label end + 1h)",
            "raw_accounting": "positive contribution minus loss contribution equals direct mean net",
            "conversion_residual": "realized net = causal mapped EV + conversion residual",
            "band_coordinates": "global percentile of mapped_direct_net against all prior mapped candidates in [UTC day-21d, UTC day); ties retain a common midpoint percentile and band",
            "band_contributions": "mapped EV, conversion residual and realized-net band contributions divide by total selected book slots and add to the global book values",
            "coordinate_feature_usage": "candidate_global_mapped_ev_coordinates.parquet is decision-time context only; every transition-label metric remains prohibited as a model feature",
        },
        "label_columns_not_model_features": [
            column
            for column in labels.columns
            if column
            not in {
                "cohort_anchor_utc",
                "horizon_hours",
                "horizon_role",
                "book_fraction",
            }
        ],
        "support_summary": summary.to_dict(orient="records"),
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "rows": int(len(labels)),
        "band_rows": int(len(band_labels)),
        **reconciliation,
        **population_audit,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mapping-source", type=Path, default=MAPPING_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
