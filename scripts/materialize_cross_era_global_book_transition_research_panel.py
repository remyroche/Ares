#!/usr/bin/env python3
"""Materialize a source-separated cross-era regime-transition research panel.

The panel binds three independently valid global-book label families to one
outcome-free market-state geometry:

* January--April 2025 exact-1m, fee-only reconstructed economics;
* February--April 2025 exact spread-aware canonical economics; and
* May--July 2026 exact spread-aware current-lineage economics.

Economic values are never pooled as if the policy/cost contracts were equal.
They share feature geometry and transition-label semantics only.  Labels use
the exact before ``[s-H,s)`` and after ``[s,s+H)`` windows emitted upstream.
The recommended non-walk-forward evaluation rows are spaced by ``2H`` so
their full before+after windows do not overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = ROOT / (
    "data_perp/artifacts/exact_history_state_recurrence_20260727_v1/"
    "timestamp_state_geometry.parquet"
)
DEFAULT_CANONICAL_LABELS = ROOT / (
    "data_perp/artifacts/canonical_global_book_conversion_transition_labels_"
    "20260729_v1"
)
DEFAULT_RECONSTRUCTED_LABELS = ROOT / (
    "data_perp/artifacts/reconstructed_exact1m_global_book_conversion_"
    "transition_labels_20260729_v1"
)
DEFAULT_CURRENT_LABELS = ROOT / (
    "data_perp/artifacts/current_exact_policy_global_book_conversion_"
    "transition_labels_20260729_v1"
)
DEFAULT_CURRENT_MAPPING_SOURCE = ROOT / (
    "data_perp/artifacts/current_exact_policy_global_book_mapping_source_"
    "20260729_v1/causal_mapped_candidates.parquet"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/cross_era_global_book_transition_research_panel_"
    "20260730_v4"
)
SCHEMA = "cross_era_global_book_transition_research_panel_v4"
LABEL_SCHEMA = "canonical_global_book_conversion_transition_labels_v1"
BOOK_FRACTION = 0.10
HORIZONS = (3, 12)
ADVERSE_SENSITIVITY_BPS = (50, 75, 100)
PRIMARY_ADVERSE_SENSITIVITY_BPS = 75
MECHANISM_THRESHOLD = 0.0050
ECONOMIC_DELTA_COLUMNS = (
    "delta_direct_mean_net",
    "delta_mean_gross",
    "delta_mean_cost",
    "delta_mean_conversion_residual",
    "delta_opportunity_probability_0bps",
    "delta_opportunity_probability_25bps",
    "delta_positive_net_contribution",
    "delta_positive_net_contribution_robust_mean",
    "delta_loss_net_contribution",
    "delta_loss_net_contribution_robust_mean",
    "delta_p_exit_trailing",
    "delta_p_exit_timeout",
    "delta_p_exit_full_stop",
    "delta_p_exit_adverse_exit",
)


SOURCE_CONTRACTS: dict[str, dict[str, str]] = {
    "reconstructed_exact1m_janapr2025": {
        "economics_tier": "exact_1m_fee_only_reconstructed",
        "policy_cost_contract": "fee_only_side_parent_not_spread_comparable",
        "path_frequency": "1m",
        "promotion_use": "research_only",
    },
    "canonical_spread_febapr2025": {
        "economics_tier": "exact_1m_spread_aware_canonical",
        "policy_cost_contract": "canonical_historical_spread_aware",
        "path_frequency": "1m",
        "promotion_use": "historical_diagnostic",
    },
    "current_exact_spread_mayjul2026": {
        "economics_tier": "exact_1m_spread_aware_current",
        "policy_cost_contract": "corrected_current_exact_policy",
        "path_frequency": "1m",
        "promotion_use": "source_separated_research",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_label_family(root: Path, source_family: str) -> pd.DataFrame:
    manifest_path = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    label_path = root / "global_book_transition_labels.parquet"
    if not all(path.is_file() for path in (manifest_path, sidecar, label_path)):
        raise FileNotFoundError(f"incomplete transition-label artifact: {root}")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError(f"transition-label manifest checksum fails: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != LABEL_SCHEMA:
        raise ValueError(f"unexpected transition-label schema: {root}")
    if (
        manifest.get("outputs_sha256", {}).get(label_path.name)
        != sha256(label_path)
    ):
        raise ValueError(f"transition-label parquet checksum fails: {root}")
    labels = pd.read_parquet(label_path)
    labels["source_family"] = source_family
    for key, value in SOURCE_CONTRACTS[source_family].items():
        labels[key] = value
    return labels


def _state_geometry_wide(geometry: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = {"side_name", "__ts__"}
    missing = sorted(required.difference(geometry.columns))
    if missing:
        raise ValueError(f"state geometry lacks {missing}")
    work = geometry.copy()
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise ValueError("state geometry contains noncanonical sides")
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    if work.duplicated(["__ts__", "side_name"], keep=False).any():
        raise ValueError("state geometry must have one row per timestamp and side")
    geometry_columns = [
        column
        for column in work.columns
        if column.startswith("median__") or column.startswith("iqr__")
    ]
    if not geometry_columns:
        raise ValueError("state geometry has no common observable fields")
    by_side: dict[str, pd.DataFrame] = {}
    for side in ("long", "short"):
        local = work.loc[
            work["side_name"].eq(side),
            ["__ts__", *geometry_columns],
        ].copy()
        local = local.rename(
            columns={
                column: f"{column}__{side}"
                for column in local.columns
                if column != "__ts__"
            }
        )
        by_side[side] = local
    wide = by_side["long"].merge(
        by_side["short"], on="__ts__", how="outer", validate="one_to_one"
    )
    feature_columns: list[str] = []
    for column in geometry_columns:
        left = pd.to_numeric(wide[f"{column}__long"], errors="coerce")
        right = pd.to_numeric(wide[f"{column}__short"], errors="coerce")
        mean_name = f"context__state_mean__{column}"
        gap_name = f"context__state_long_short_gap__{column}"
        wide[mean_name] = pd.concat([left, right], axis=1).mean(axis=1)
        wide[gap_name] = left - right
        feature_columns.extend((mean_name, gap_name))
    # Exact timestamp joins prevent a row shift from crossing the 2025--2026
    # gap or silently treating missing hours as consecutive observations.
    mean_features = [
        column for column in feature_columns if column.startswith("context__state_mean__")
    ]
    indexed = wide.set_index("__ts__", drop=False)
    for lag in (1, 3, 12):
        prior = indexed.reindex(
            indexed.index - pd.Timedelta(hours=lag)
        ).set_axis(indexed.index)
        normalized_changes: list[pd.Series] = []
        for column in mean_features:
            delta_name = f"context__past_delta_{lag}h__{column.removeprefix('context__state_mean__')}"
            indexed[delta_name] = indexed[column] - prior[column]
            feature_columns.append(delta_name)
            scale = indexed[column].abs() + prior[column].abs() + 1e-9
            normalized_changes.append(
                (indexed[column] - prior[column]).abs() / scale
            )
        summary = f"context__past_geometry_shift_{lag}h"
        indexed[summary] = pd.concat(normalized_changes, axis=1).mean(axis=1)
        feature_columns.append(summary)
    return indexed.reset_index(drop=True), feature_columns


def _hourly_coordinate_context(
    coordinates: pd.DataFrame, source_family: str
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate only causal mapped-score coordinates available at decision."""

    required = {
        "execution_decision_utc",
        "mapped_direct_net",
        "map_reference_rows",
        "causal_global_mapped_ev_percentile",
        "causal_global_mapped_ev_band",
        "causal_global_mapped_ev_margin_to_p90",
    }
    missing = sorted(required.difference(coordinates.columns))
    if missing:
        raise ValueError(f"{source_family} mapping coordinates lack {missing}")
    work = coordinates.copy()
    work["execution_decision_utc"] = pd.to_datetime(
        work["execution_decision_utc"], utc=True, errors="raise"
    )
    work["mapped_direct_net"] = pd.to_numeric(
        work["mapped_direct_net"], errors="coerce"
    )
    work = work.loc[
        work["causal_global_mapped_ev_band"].ne("UNAVAILABLE")
        & work["mapped_direct_net"].notna()
    ].copy()
    if work.empty:
        raise ValueError(f"{source_family} has no causal mapping coordinates")
    work["_score_sq"] = work["mapped_direct_net"] ** 2
    work["_percentile"] = pd.to_numeric(
        work["causal_global_mapped_ev_percentile"], errors="coerce"
    )
    work["_margin"] = pd.to_numeric(
        work["causal_global_mapped_ev_margin_to_p90"], errors="coerce"
    )
    work["_log_reference"] = np.log1p(
        pd.to_numeric(work["map_reference_rows"], errors="coerce")
    )
    work["_above_p90"] = work["_margin"].gt(0.0).astype(float)
    for band in ("B0", "B1", "B2", "B3", "B4"):
        work[f"_band_{band}"] = work[
            "causal_global_mapped_ev_band"
        ].eq(band).astype(float)
    grouped = work.groupby("execution_decision_utc", sort=True)
    hourly = grouped.agg(
        _count=("mapped_direct_net", "size"),
        _score_sum=("mapped_direct_net", "sum"),
        _score_sq_sum=("_score_sq", "sum"),
        _score_p10=("mapped_direct_net", lambda values: values.quantile(0.10)),
        _score_p50=("mapped_direct_net", "median"),
        _score_p90=("mapped_direct_net", lambda values: values.quantile(0.90)),
        _percentile_sum=("_percentile", "sum"),
        _margin_sum=("_margin", "sum"),
        _log_reference_sum=("_log_reference", "sum"),
        _log_reference_min=("_log_reference", "min"),
        _above_p90_sum=("_above_p90", "sum"),
        **{
            f"_band_{band}_sum": (f"_band_{band}", "sum")
            for band in ("B0", "B1", "B2", "B3", "B4")
        },
    )
    context = pd.DataFrame(index=hourly.index)
    features: list[str] = []

    def add_scope(scope: str, summary: pd.DataFrame) -> None:
        count = summary["_count"].replace(0, np.nan)
        mean = summary["_score_sum"] / count
        variance = np.maximum(summary["_score_sq_sum"] / count - mean**2, 0.0)
        fields = {
            "support": summary["_count"],
            "mapped_mean": mean,
            "mapped_std": np.sqrt(variance),
            "mapped_p10": summary["_score_p10"],
            "mapped_p50": summary["_score_p50"],
            "mapped_p90": summary["_score_p90"],
            "percentile_mean": summary["_percentile_sum"] / count,
            "margin_to_p90_mean": summary["_margin_sum"] / count,
            "log_reference_mean": summary["_log_reference_sum"] / count,
            "log_reference_min": summary["_log_reference_min"],
            "above_p90_share": summary["_above_p90_sum"] / count,
            **{
                f"band_{band}_share": summary[f"_band_{band}_sum"] / count
                for band in ("B0", "B1", "B2", "B3", "B4")
            },
        }
        for name, values in fields.items():
            column = f"context__mapping_{scope}__{name}"
            context[column] = values
            features.append(column)

    add_scope("current", hourly)
    for hours in (3, 12):
        rolling = pd.DataFrame(index=hourly.index)
        for column in (
            "_count",
            "_score_sum",
            "_score_sq_sum",
            "_percentile_sum",
            "_margin_sum",
            "_log_reference_sum",
            "_above_p90_sum",
            *(f"_band_{band}_sum" for band in ("B0", "B1", "B2", "B3", "B4")),
        ):
            rolling[column] = hourly[column].rolling(
                f"{hours}h", closed="left", min_periods=1
            ).sum()
        for column in ("_score_p10", "_score_p50", "_score_p90"):
            rolling[column] = hourly[column].rolling(
                f"{hours}h", closed="left", min_periods=1
            ).mean()
        rolling["_log_reference_min"] = hourly["_log_reference_min"].rolling(
            f"{hours}h", closed="left", min_periods=1
        ).min()
        add_scope(f"trailing_{hours}h", rolling)
    context = context.reset_index().rename(
        columns={"execution_decision_utc": "cohort_anchor_utc"}
    )
    context["source_family"] = source_family
    return context, features


def _exact_shift(
    frame: pd.DataFrame, column: str, offsets: Sequence[int]
) -> list[pd.Series]:
    indexed = frame.set_index("cohort_anchor_utc")[column]
    return [
        indexed.reindex(
            frame["cohort_anchor_utc"] + pd.Timedelta(hours=offset)
        ).reset_index(drop=True)
        for offset in offsets
    ]


def _add_active_onset_label_family(
    work: pd.DataFrame,
    *,
    raw_target: str,
    raw_available: str,
    suffix: str,
) -> None:
    """Derive persistence/onset targets and every dependent availability time."""

    active_target = f"target__active_adverse{suffix}"
    active_available = f"{active_target}_available_utc"
    onset_target = f"target__adverse_onset{suffix}"
    onset_available = f"{onset_target}_available_utc"
    lead_target = f"target__adverse_onset_within_3h{suffix}"
    lead_available = f"{lead_target}_available_utc"

    future_raw = _exact_shift(work, raw_target, (0, 1, 2))
    future_matrix = pd.concat(future_raw, axis=1)
    future_complete = future_matrix.notna().all(axis=1)
    work[active_target] = (
        future_matrix.sum(axis=1).ge(2).where(future_complete).astype(float)
    )
    future_available = _exact_shift(work, raw_available, (0, 1, 2))
    future_available_matrix = pd.concat(future_available, axis=1)
    work[active_available] = future_available_matrix.max(axis=1).where(
        future_complete & future_available_matrix.notna().all(axis=1)
    )

    prior_active = _exact_shift(work, active_target, (-6, -5, -4, -3, -2, -1))
    prior_matrix = pd.concat(prior_active, axis=1)
    prior_complete = prior_matrix.notna().all(axis=1)
    onset_complete = prior_complete & work[active_target].notna()
    work[onset_target] = (
        work[active_target].eq(1.0) & prior_matrix.max(axis=1).eq(0.0)
    ).astype(float).where(onset_complete)
    onset_dependencies = _exact_shift(
        work, active_available, (-6, -5, -4, -3, -2, -1, 0)
    )
    onset_dependency_matrix = pd.concat(onset_dependencies, axis=1)
    work[onset_available] = onset_dependency_matrix.max(axis=1).where(
        onset_complete & onset_dependency_matrix.notna().all(axis=1)
    )

    future_onset = _exact_shift(work, onset_target, (0, 1, 2))
    future_onset_matrix = pd.concat(future_onset, axis=1)
    future_onset_complete = future_onset_matrix.notna().all(axis=1)
    work[lead_target] = future_onset_matrix.max(axis=1).where(
        future_onset_complete
    )
    future_onset_available = _exact_shift(work, onset_available, (0, 1, 2))
    future_onset_available_matrix = pd.concat(future_onset_available, axis=1)
    work[lead_available] = future_onset_available_matrix.max(axis=1).where(
        future_onset_complete & future_onset_available_matrix.notna().all(axis=1)
    )


def _add_persistent_adverse_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add adverse sensitivity and conditional mechanisms with exact availability."""

    pieces: list[pd.DataFrame] = []
    for _, source in frame.groupby(
        ["source_family", "horizon_hours", "book_fraction"], sort=False
    ):
        work = source.sort_values("cohort_anchor_utc").reset_index(drop=True)
        work["target__raw_adverse_available_utc"] = pd.concat(
            [
                pd.to_datetime(work["before_target_available_utc"], utc=True),
                pd.to_datetime(work["after_target_available_utc"], utc=True),
            ],
            axis=1,
        ).max(axis=1)
        for threshold_bps in ADVERSE_SENSITIVITY_BPS:
            threshold = threshold_bps / 10_000.0
            raw_target = f"target__raw_adverse_sensitivity_{threshold_bps}bps"
            raw_available = f"{raw_target}_available_utc"
            work[raw_target] = (
                work["after_mean_conversion_residual"].le(-0.0050)
                & work["delta_mean_conversion_residual"].le(-threshold)
                & work["delta_direct_mean_net"].le(-threshold)
            ).astype(float)
            work[raw_available] = work["target__raw_adverse_available_utc"]
            _add_active_onset_label_family(
                work,
                raw_target=raw_target,
                raw_available=raw_available,
                suffix=f"_sensitivity_{threshold_bps}bps",
            )

        # Preserve the original primary target names as 75-bps aliases for
        # downstream readers, while making the sensitivity lineage explicit.
        primary_suffix = f"_sensitivity_{PRIMARY_ADVERSE_SENSITIVITY_BPS}bps"
        for stem in (
            "raw_adverse",
            "active_adverse",
            "adverse_onset",
            "adverse_onset_within_3h",
        ):
            sensitivity_name = f"target__{stem}{primary_suffix}"
            work[f"target__{stem}"] = work[sensitivity_name]
            work[f"target__{stem}_available_utc"] = work[
                f"{sensitivity_name}_available_utc"
            ]

        # These labels are intentionally undefined outside a confirmed active
        # adverse state.  A mechanism classifier answers a conditional
        # question after the adverse-state head, not "is a normal row a loss".
        active = work["target__active_adverse"].eq(1.0)
        work["target__mechanism_upside_collapse"] = (
            work["delta_positive_net_contribution"].le(-MECHANISM_THRESHOLD)
        ).astype(float).where(active)
        work["target__mechanism_loss_expansion"] = (
            work["delta_loss_net_contribution"].ge(MECHANISM_THRESHOLD)
        ).astype(float).where(active)
        work["target__mechanism_upside_collapse_available_utc"] = work[
            "target__active_adverse_available_utc"
        ].where(active)
        work["target__mechanism_loss_expansion_available_utc"] = work[
            "target__active_adverse_available_utc"
        ].where(active)
        pieces.append(work)
    return pd.concat(pieces, ignore_index=True)


def build_transition_panel(
    label_families: Mapping[str, pd.DataFrame],
    geometry: pd.DataFrame,
    coordinate_families: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    for source_family, source in label_families.items():
        if source_family not in SOURCE_CONTRACTS:
            raise ValueError(f"unknown transition source family: {source_family}")
        work = source.copy()
        work["source_family"] = source_family
        for key, value in SOURCE_CONTRACTS[source_family].items():
            work[key] = value
        frames.append(work)
    labels = pd.concat(frames, ignore_index=True, sort=False)
    labels["cohort_anchor_utc"] = pd.to_datetime(
        labels["cohort_anchor_utc"], utc=True, errors="raise"
    )
    labels = labels.loc[
        labels["horizon_hours"].isin(HORIZONS)
        & np.isclose(labels["book_fraction"], BOOK_FRACTION)
        & labels["before_global_hour_complete_flag"].astype(bool)
        & labels["after_global_hour_complete_flag"].astype(bool)
        & labels["before_selected_candidate_support"].gt(0)
        & labels["after_selected_candidate_support"].gt(0)
    ].copy()
    if labels.empty:
        raise ValueError("no complete global-book transition labels remain")
    if not labels["outcome_only_not_model_feature"].astype(bool).all():
        raise ValueError("upstream economic labels lost outcome-only guard")
    labels["signal_context_utc"] = (
        labels["cohort_anchor_utc"] - pd.Timedelta(hours=1)
    )

    state, feature_columns = _state_geometry_wide(geometry)
    context = state.rename(columns={"__ts__": "signal_context_utc"})
    panel = labels.merge(
        context,
        on="signal_context_utc",
        how="left",
        validate="many_to_one",
    )
    panel["context_available"] = panel[feature_columns].notna().any(axis=1)
    if not panel["context_available"].any():
        raise ValueError("no exact state context joins transition anchors")
    if coordinate_families:
        coordinate_frames: list[pd.DataFrame] = []
        coordinate_features: list[str] = []
        for source_family, coordinates in coordinate_families.items():
            local, columns = _hourly_coordinate_context(
                coordinates, source_family
            )
            coordinate_frames.append(local)
            coordinate_features.extend(columns)
        mapping_context = pd.concat(coordinate_frames, ignore_index=True)
        panel = panel.merge(
            mapping_context,
            on=["cohort_anchor_utc", "source_family"],
            how="left",
            validate="many_to_one",
        )
        feature_columns.extend(list(dict.fromkeys(coordinate_features)))

    future = state.loc[
        :,
        [
            "__ts__",
            *[
                column
                for column in feature_columns
                if column.startswith("context__state_mean__")
            ],
        ],
    ].copy()
    future_columns = {
        column: f"future__{column}"
        for column in future.columns
        if column != "__ts__"
    }
    future = future.rename(columns=future_columns)
    panel["future_context_utc"] = panel["signal_context_utc"] + pd.to_timedelta(
        panel["horizon_hours"], unit="h"
    )
    panel = panel.merge(
        future.rename(columns={"__ts__": "future_context_utc"}),
        on="future_context_utc",
        how="left",
        validate="many_to_one",
    )
    target_columns: list[str] = list(ECONOMIC_DELTA_COLUMNS)
    shifts: list[pd.Series] = []
    for column in [
        name
        for name in feature_columns
        if name.startswith("context__state_mean__")
    ]:
        future_column = f"future__{column}"
        scale = panel[column].abs() + panel[future_column].abs() + 1e-9
        shifts.append((panel[future_column] - panel[column]).abs() / scale)
    panel["target__future_market_geometry_shift"] = pd.concat(
        shifts, axis=1
    ).mean(axis=1)
    target_columns.append("target__future_market_geometry_shift")

    panel["target__net_crosses_below_zero"] = (
        panel["before_direct_mean_net"].gt(0.0)
        & panel["after_direct_mean_net"].le(0.0)
    ).astype(float)
    panel["target__opportunity_collapse_10pp"] = panel[
        "delta_opportunity_probability_0bps"
    ].le(-0.10).astype(float)
    panel["target__loss_expansion_25bps"] = panel[
        "delta_loss_net_contribution"
    ].ge(0.0025).astype(float)
    panel["target__soft_net_deterioration_25bps"] = np.clip(
        -panel["delta_direct_mean_net"] / 0.0025, 0.0, 1.0
    )
    panel["target__soft_opportunity_collapse_10pp"] = np.clip(
        -panel["delta_opportunity_probability_0bps"] / 0.10, 0.0, 1.0
    )
    panel["target__soft_loss_expansion_25bps"] = np.clip(
        panel["delta_loss_net_contribution"] / 0.0025, 0.0, 1.0
    )
    panel["target__soft_conversion_deterioration_25bps"] = np.clip(
        -panel["delta_mean_conversion_residual"] / 0.0025, 0.0, 1.0
    )
    panel["target__adverse_transition_any"] = (
        panel[
            [
                "target__net_crosses_below_zero",
                "target__opportunity_collapse_10pp",
                "target__loss_expansion_25bps",
            ]
        ]
        .max(axis=1)
        .astype(float)
    )
    panel = _add_persistent_adverse_labels(panel)
    sensitivity_target_columns = [
        name
        for threshold_bps in ADVERSE_SENSITIVITY_BPS
        for name in (
            f"target__raw_adverse_sensitivity_{threshold_bps}bps",
            f"target__active_adverse_sensitivity_{threshold_bps}bps",
            f"target__adverse_onset_sensitivity_{threshold_bps}bps",
            f"target__adverse_onset_within_3h_sensitivity_{threshold_bps}bps",
        )
    ]
    target_columns.extend(
        [
            "target__net_crosses_below_zero",
            "target__opportunity_collapse_10pp",
            "target__loss_expansion_25bps",
            "target__soft_net_deterioration_25bps",
            "target__soft_opportunity_collapse_10pp",
            "target__soft_loss_expansion_25bps",
            "target__soft_conversion_deterioration_25bps",
            "target__adverse_transition_any",
            "target__raw_adverse_primary",
            "target__active_adverse",
            "target__adverse_onset",
            "target__adverse_onset_within_3h",
            "target__mechanism_upside_collapse",
            "target__mechanism_loss_expansion",
            "target__raw_adverse_sensitivity_50bps",
            "target__raw_adverse_sensitivity_75bps",
            "target__raw_adverse_sensitivity_100bps",
            *sensitivity_target_columns,
        ]
    )

    epoch_hours = (
        panel["cohort_anchor_utc"].astype("int64") // 3_600_000_000_000
    )
    panel["nonoverlap_anchor_flag"] = (
        epoch_hours % (2 * panel["horizon_hours"].astype(int))
    ).eq(0)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    block_number = (
        (panel["cohort_anchor_utc"] - epoch) // pd.Timedelta(days=7)
    ).astype(int)
    panel["cv_block_start_utc"] = epoch + pd.to_timedelta(
        block_number * 7, unit="D"
    )
    panel["cv_group_id"] = (
        "utc7d_" + block_number.astype(str)
    )
    # The default audit field follows the primary actionable target.  Every
    # sensitivity and conditional mechanism also retains its own availability
    # field; callers must use that field when fitting a different target.
    panel["target_available_utc"] = pd.to_datetime(
        panel["target__adverse_onset_within_3h_available_utc"], utc=True
    )

    prohibited = [
        column
        for column in feature_columns
        if any(
            token in column.lower()
            for token in (
                "target",
                "outcome",
                "execution",
                "mfe",
                "mae",
                "exit",
                "future",
                "realized",
            )
        )
    ]
    if prohibited:
        raise ValueError(f"outcome/future fields entered feature surface: {prohibited}")
    panel = panel.sort_values(
        ["cohort_anchor_utc", "source_family", "horizon_hours"],
        kind="stable",
    ).reset_index(drop=True)
    return panel, feature_columns, target_columns


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    label_roots = {
        "reconstructed_exact1m_janapr2025": Path(args.reconstructed_labels),
        "canonical_spread_febapr2025": Path(args.canonical_labels),
        "current_exact_spread_mayjul2026": Path(args.current_labels),
    }
    label_frames = {
        name: _load_label_family(path, name)
        for name, path in label_roots.items()
    }
    coordinate_frames = {
        name: pd.read_parquet(
            path / "candidate_global_mapped_ev_coordinates.parquet"
        )
        for name, path in label_roots.items()
    }
    geometry_path = Path(args.geometry)
    panel, feature_columns, target_columns = build_transition_panel(
        label_frames,
        pd.read_parquet(geometry_path),
        coordinate_frames,
    )
    panel["mapping_provenance_role"] = "strict_oof"
    current_mapping_path = Path(args.current_mapping_source)
    current_mapping = pd.read_parquet(
        current_mapping_path,
        columns=[
            "execution_decision_utc",
            "mapped_eligible",
            "causal_recent_side_isotonic_ev__is_oof",
            "causal_recent_side_isotonic_ev__is_forward_oos",
        ],
    )
    current_mapping["execution_decision_utc"] = pd.to_datetime(
        current_mapping["execution_decision_utc"], utc=True, errors="raise"
    )
    current_mapping = current_mapping.loc[
        current_mapping["mapped_eligible"].astype(bool)
    ].copy()
    provenance = (
        current_mapping.groupby("execution_decision_utc", sort=True)
        .agg(
            provenance_oof_share=(
                "causal_recent_side_isotonic_ev__is_oof",
                "mean",
            ),
            provenance_forward_oos_share=(
                "causal_recent_side_isotonic_ev__is_forward_oos",
                "mean",
            ),
        )
        .reset_index()
        .rename(columns={"execution_decision_utc": "cohort_anchor_utc"})
    )
    current_mask = panel["source_family"].eq(
        "current_exact_spread_mayjul2026"
    )
    current_rows = panel.loc[current_mask].merge(
        provenance,
        on="cohort_anchor_utc",
        how="left",
        validate="many_to_one",
    )
    if current_rows["provenance_oof_share"].isna().any():
        raise ValueError("current transition rows lack mapping provenance")
    current_rows["mapping_provenance_role"] = np.select(
        [
            current_rows["provenance_oof_share"].eq(1.0),
            current_rows["provenance_forward_oos_share"].eq(1.0),
        ],
        ["strict_oof", "frozen_forward_oos"],
        default="mixed",
    )
    panel = pd.concat(
        [panel.loc[~current_mask], current_rows],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["cohort_anchor_utc", "source_family", "horizon_hours"],
        kind="stable",
    ).reset_index(drop=True)
    panel["provenance_oof_share"] = panel[
        "provenance_oof_share"
    ].fillna(1.0)
    panel["provenance_forward_oos_share"] = panel[
        "provenance_forward_oos_share"
    ].fillna(0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    panel_path = temporary / "transition_research_panel.parquet"
    catalog_path = temporary / "field_catalog.csv"
    panel.to_parquet(panel_path, index=False, compression="zstd")
    catalog_rows = []
    for column in panel.columns:
        role = (
            "decision_time_feature"
            if column in feature_columns
            else "target_availability"
            if column.startswith("target__") and column.endswith("_available_utc")
            else "target"
            if column in target_columns or column.startswith(("before_", "after_", "delta_", "future__"))
            else "metadata"
        )
        catalog_rows.append({"column": column, "role": role})
    pd.DataFrame(catalog_rows).to_csv(catalog_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "SOURCE_SEPARATED_TRANSITION_RESEARCH_PANEL_COMPLETE",
        "rows": int(len(panel)),
        "nonoverlap_rows": int(panel["nonoverlap_anchor_flag"].sum()),
        "context_available_rows": int(panel["context_available"].sum()),
        "source_rows": {
            str(key): int(value)
            for key, value in panel["source_family"].value_counts().items()
        },
        "horizon_rows": {
            str(int(key)): int(value)
            for key, value in panel["horizon_hours"].value_counts().items()
        },
        "feature_count": len(feature_columns),
        "target_count": len(target_columns),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "source_contracts": SOURCE_CONTRACTS,
        "contracts": {
            "selection": "one pooled global top 10% within each source family and exact before/after window; never per timestamp or side",
            "windows": "before [s-H,s), after [s,s+H), H in {3h,12h}",
            "context_time": "only raw median/IQR state geometry at source signal s-1h, exact prior lags, and causal mapped-EV coordinates available at s",
            "future_geometry": "target-only raw geometry at source signal s+H-1h; historical fitted state IDs/distances/OOD are excluded",
            "nonoverlap_evaluation": "retain anchors on a fixed UTC 2H stride; their complete before+after windows touch but do not overlap",
            "cross_validation": "shuffle 7-day UTC groups across folds, keeping duplicate calendar blocks from every source family together; no walk-forward requirement",
            "source_separation": "report every metric per source family/economics tier; never pool fee-only and spread-aware PnL",
            "calendar_features": "month, year, source family and cv group are metadata only and prohibited as model features",
            "label_hpo": "soft net/opportunity/loss/conversion components are separate; weights must be selected inside grouped training folds",
            "primary_adverse_label": "the 75-bps sensitivity label is primary: after conversion residual <= -50bps AND delta conversion residual <= -75bps AND delta direct net <= -75bps; active requires 2 of current/next two anchors; onset requires no active state in prior 6h; onset-within-3h uses current/next two onsets",
            "adverse_label_sensitivity": "predeclared 50/75/100-bps delta thresholds each have independent raw, active, onset and onset-within-3h targets; 75bps is exposed through the legacy primary aliases",
            "mechanism_labels": "upside-collapse and loss-expansion labels are defined only conditional on the primary 75-bps active adverse state; inactive rows are null, not negative examples",
            "derived_label_availability": "each raw, active, onset, onset-within-3h and conditional mechanism target has its own exact max of every dependent upstream label availability timestamp",
            "current_provenance": "current exact rows retain strict mapped OOF versus frozen nonpromotable forward-OOS role; provenance is metadata, never a model feature",
        },
        "sources_sha256": {
            "geometry": {
                "path": str(geometry_path),
                "sha256": sha256(geometry_path),
            },
            "current_mapping_provenance": {
                "path": str(current_mapping_path),
                "sha256": sha256(current_mapping_path),
            },
            **{
                name: {
                    "path": str(path),
                    "manifest_sha256": sha256(path / "manifest.json"),
                    "labels_sha256": sha256(
                        path / "global_book_transition_labels.parquet"
                    ),
                    "coordinates_sha256": sha256(
                        path / "candidate_global_mapped_ev_coordinates.parquet"
                    ),
                }
                for name, path in label_roots.items()
            },
        },
        "outputs": {
            "panel": {
                "path": panel_path.name,
                "sha256": sha256(panel_path),
            },
            "catalog": {
                "path": catalog_path.name,
                "sha256": sha256(catalog_path),
            },
        },
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
        "rows": int(len(panel)),
        "nonoverlap_rows": int(panel["nonoverlap_anchor_flag"].sum()),
        "context_available_rows": int(panel["context_available"].sum()),
        "features": len(feature_columns),
        "targets": len(target_columns),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    result.add_argument(
        "--canonical-labels", type=Path, default=DEFAULT_CANONICAL_LABELS
    )
    result.add_argument(
        "--reconstructed-labels",
        type=Path,
        default=DEFAULT_RECONSTRUCTED_LABELS,
    )
    result.add_argument(
        "--current-labels", type=Path, default=DEFAULT_CURRENT_LABELS
    )
    result.add_argument(
        "--current-mapping-source",
        type=Path,
        default=DEFAULT_CURRENT_MAPPING_SOURCE,
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
