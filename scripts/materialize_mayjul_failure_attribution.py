#!/usr/bin/env python3
"""Attribute May--July opportunity/capture failure inside frozen global books.

The runner is diagnostic only.  It selects each score's pooled monthly global
top decile once, then describes that frozen book by exact path outcomes,
strict-OOF adverse-risk predictions, and decision-time global transition
context.  It never reranks by a slice, fits a model, or reads future transition
targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.materialize_meaningful_mfe_tail_recall import (
    DEFAULT_ALLSCORE,
    DEFAULT_GRID,
    PRIMARY_GRID,
    bind_opportunity_labels,
)
from scripts.materialize_source_separated_ic_ev_waterfall import (
    IDENTITY_COLUMNS,
    safe,
    sha256,
    stable_top,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT = ROOT / (
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/"
    "joined.parquet"
)
DEFAULT_CONTEXT_MANIFEST = DEFAULT_CONTEXT.with_name("manifest.json")
DEFAULT_RISK = ROOT / (
    "data_perp/artifacts/path_auxiliary_mae_competing_risk_20260725_v1/"
    "mae_competing_risk.parquet"
)
DEFAULT_RISK_MANIFEST = DEFAULT_RISK.with_name("mae_competing_risk.manifest.json")
DEFAULT_TRANSITION = ROOT / (
    "data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4/"
    "transition_research_panel.parquet"
)
DEFAULT_TRANSITION_CATALOG = DEFAULT_TRANSITION.with_name("field_catalog.csv")
DEFAULT_TRANSITION_MANIFEST = DEFAULT_TRANSITION.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/mayjul_failure_attribution_20260730_v1"
)
SCORES = (
    "score_base_alpha",
    "score_residual_expected_ev",
    "score_direct_q25_challenger_bps",
    "score_transfer_parent_bps",
)
TRANSITION_FEATURES = (
    "context__state_mean__median__atr_compression_ratio",
    "context__state_mean__median__breakout_24h",
    "context__state_mean__median__dir_path_risk_skew_2h",
    "context__state_mean__median__jump_intensity",
    "context__state_mean__median__leverage_build_score",
    "context__state_mean__median__memory_asymmetry_1ATR",
    "context__state_mean__median__spread_proxy_abs_return_bps_robust_z",
    "context__past_geometry_shift_3h",
    "context__mapping_current__above_p90_share",
)
RISK_COLUMNS = (
    "prediction_p_adverse_0_5r_before_mfe",
    "prediction_p_stop_1r_before_mfe",
)
MIN_FINDING_ROWS = 100
MIN_FINDING_DAYS = 5
BOOTSTRAP_REPS = 500
SEED = 20260730


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    out = frame.copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["side_name"] = out["side_name"].astype(str)
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    if out.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{source} has duplicate identities")
    return out


def _verify_sources(
    *,
    context_path: Path,
    context_manifest_path: Path,
    risk_path: Path,
    risk_manifest_path: Path,
    transition_path: Path,
    transition_catalog_path: Path,
    transition_manifest_path: Path,
) -> None:
    context_manifest = _read_json(context_manifest_path)
    if context_manifest.get("schema") != "canonical_exact_policy_regime_input_v1":
        raise ValueError("unexpected canonical context manifest")
    if context_manifest.get("output", {}).get("sha256") != sha256(context_path):
        raise ValueError("canonical context hash mismatch")
    risk_manifest = _read_json(risk_manifest_path)
    if (
        risk_manifest.get("schema")
        != "path_auxiliary_mae_competing_risk_side_local_v1_strict_oof"
    ):
        raise ValueError("unexpected adverse-risk manifest")
    if risk_manifest.get("source_artifact_sha256") != sha256(risk_path):
        raise ValueError("adverse-risk hash mismatch")
    transition_manifest = _read_json(transition_manifest_path)
    if (
        transition_manifest.get("schema")
        != "cross_era_global_book_transition_research_panel_v4"
    ):
        raise ValueError("unexpected transition manifest")
    outputs = transition_manifest.get("outputs", {})
    if outputs.get("panel", {}).get("sha256") != sha256(transition_path):
        raise ValueError("transition panel hash mismatch")
    if outputs.get("catalog", {}).get("sha256") != sha256(
        transition_catalog_path
    ):
        raise ValueError("transition catalog hash mismatch")


def _left_exact(
    anchor: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    selected = source.loc[:, [*IDENTITY_COLUMNS, *columns]]
    out = anchor.merge(
        selected,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not out["_merge"].eq("both").all():
        raise ValueError(f"{source_name} does not cover every canonical identity")
    return out.drop(columns="_merge")


def _within_decision_percentile(frame: pd.DataFrame, score: str) -> pd.Series:
    result = pd.Series(index=frame.index, dtype=float)
    for _, local in frame.groupby("execution_decision_utc", sort=False):
        values = pd.to_numeric(local[score], errors="raise").to_numpy(float)
        ids = local["candidate_id"].astype(str).to_numpy()
        order = np.lexsort((ids, values))
        ranks = np.empty(len(local), dtype=float)
        ranks[order] = (np.arange(len(local), dtype=float) + 1.0) / len(local)
        result.loc[local.index] = ranks
    return result


def prepare_attribution_frame(
    allscore: pd.DataFrame,
    grid: pd.DataFrame,
    context: pd.DataFrame,
    risk: pd.DataFrame,
    transition: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bind exact outcomes, strict-OOF risk, and causal transition context."""

    frame = bind_opportunity_labels(
        allscore,
        grid,
        grid_names=[PRIMARY_GRID],
        expected_rows=len(allscore),
    )
    context = _identity(context, "canonical context")
    context_columns = [
        "execution_decision_utc",
        "execution_net_ev_12h",
        "catboost_archetype",
    ]
    frame = _left_exact(
        frame, context, source_name="canonical context", columns=context_columns
    )
    if not pd.to_datetime(
        frame["execution_decision_utc_y"], utc=True, errors="raise"
    ).eq(
        pd.to_datetime(frame["execution_decision_utc_x"], utc=True, errors="raise")
    ).all():
        raise ValueError("canonical context decision timestamps disagree")
    if (
        pd.to_numeric(frame["execution_net_ev_12h_x"], errors="raise")
        - pd.to_numeric(frame["execution_net_ev_12h_y"], errors="raise")
    ).abs().max() > 1e-12:
        raise ValueError("canonical context exact-policy net disagrees")
    frame = frame.rename(
        columns={
            "execution_decision_utc_x": "execution_decision_utc",
            "execution_net_ev_12h_x": "execution_net_ev_12h",
        }
    ).drop(columns=["execution_decision_utc_y", "execution_net_ev_12h_y"])

    risk = _identity(risk, "strict-OOF adverse risk")
    risk_columns = [
        *RISK_COLUMNS,
        "available_at",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "oof_fold",
    ]
    frame = _left_exact(frame, risk, source_name="adverse risk", columns=risk_columns)
    available = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    cutoff = pd.to_datetime(
        frame["train_decision_cutoff"], utc=True, errors="raise"
    )
    decision = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    if not available.le(decision).all():
        raise ValueError("adverse-risk prediction is unavailable at decision")
    if not cutoff.lt(decision).all():
        raise ValueError("adverse-risk training cutoff reaches decision")
    for column in RISK_COLUMNS:
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.between(0.0, 1.0).all():
            raise ValueError(f"{column} falls outside [0,1]")

    if catalog.columns.tolist() != ["column", "role"]:
        raise ValueError("unexpected transition field catalog schema")
    roles = dict(zip(catalog["column"], catalog["role"]))
    illegal = [
        field
        for field in TRANSITION_FEATURES
        if roles.get(field) != "decision_time_feature"
    ]
    if illegal:
        raise ValueError(f"transition fields are not decision-time: {illegal}")
    if any(
        field.startswith("target") or field.startswith("future")
        for field in TRANSITION_FEATURES
    ):
        raise ValueError("future/target transition field requested")

    current = transition.loc[
        transition["source_family"].eq("current_exact_spread_mayjul2026")
        & transition["horizon_hours"].eq(12)
        & transition["book_fraction"].eq(0.10)
        & transition["mapping_provenance_role"].eq("strict_oof")
        & transition["context_available"].astype(bool)
    ].copy()
    current["cohort_anchor_utc"] = pd.to_datetime(
        current["cohort_anchor_utc"], utc=True, errors="raise"
    )
    current["signal_context_utc"] = pd.to_datetime(
        current["signal_context_utc"], utc=True, errors="raise"
    )
    if current["cohort_anchor_utc"].duplicated().any():
        raise ValueError("transition context has duplicate hourly anchors")
    if not current["signal_context_utc"].eq(
        current["cohort_anchor_utc"] - pd.Timedelta(hours=1)
    ).all():
        raise ValueError("transition context is not available one hour before anchor")
    transition_columns = ["cohort_anchor_utc", *TRANSITION_FEATURES]
    frame = frame.merge(
        current.loc[:, transition_columns],
        left_on="execution_decision_utc",
        right_on="cohort_anchor_utc",
        how="left",
        validate="many_to_one",
    )
    frame["transition_context_available"] = frame[
        "cohort_anchor_utc"
    ].notna()

    for score in SCORES:
        if score not in frame or not np.isfinite(
            pd.to_numeric(frame[score], errors="raise")
        ).all():
            raise ValueError(f"score is missing/nonfinite: {score}")
        frame[f"{score}__decision_percentile"] = _within_decision_percentile(
            frame, score
        )
    frame["direct_minus_base_decision_percentile"] = (
        frame[
            "score_direct_q25_challenger_bps__decision_percentile"
        ]
        - frame["score_base_alpha__decision_percentile"]
    )
    frame["candidate_day"] = decision.dt.floor("D")

    historical_reference = transition.loc[
        pd.to_datetime(transition["cohort_anchor_utc"], utc=True, errors="raise")
        < pd.Timestamp("2026-05-01", tz="UTC")
    ].copy()
    return frame, historical_reference


def _fixed_band(values: pd.Series, cuts: Sequence[float]) -> pd.Series:
    boundaries = [-np.inf, *[float(value) for value in cuts], np.inf]
    labels = [f"B{index}" for index in range(len(boundaries) - 1)]
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=boundaries,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    ).astype("string").fillna("UNAVAILABLE")


def _cvar05(values: pd.Series) -> float:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if not len(ordered):
        return np.nan
    return float(ordered.iloc[: max(1, int(np.ceil(0.05 * len(ordered))))].mean())


def _conditional_rate(frame: pd.DataFrame, condition: str) -> float:
    mask = frame[condition].astype(bool)
    return float(frame.loc[mask, "exact_net_positive"].mean()) if mask.any() else np.nan


def _seed_for(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") ^ SEED


def _day_bootstrap(
    frame: pd.DataFrame, *, reps: int, seed: int
) -> tuple[float, float]:
    groups = [
        local["execution_net_ev_12h"].to_numpy(float)
        for _, local in frame.groupby("candidate_day", sort=True)
    ]
    if not groups:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for index in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        values = np.concatenate([groups[item] for item in chosen])
        means[index] = values.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low * 1e4), float(high * 1e4)


def _book_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    asset_share = frame["__symbol__"].value_counts(normalize=True)
    return {
        "selected_rows": int(len(frame)),
        "selected_days": int(frame["candidate_day"].nunique()),
        "long_share": float(frame["side_name"].eq("long").mean()),
        "unique_assets": int(frame["__symbol__"].nunique()),
        "largest_asset_share": (
            float(asset_share.iloc[0]) if len(asset_share) else np.nan
        ),
        "asset_hhi": (
            float(np.square(asset_share.to_numpy(float)).sum())
            if len(asset_share)
            else np.nan
        ),
        "mean_mfe_bps": float(frame["execution_mfe_return_12h"].mean() * 1e4),
        "mean_gross_bps": float(frame["execution_gross_ev_12h"].mean() * 1e4),
        "mean_cost_bps": float(frame["execution_cost_return"].mean() * 1e4),
        "mean_net_bps": float(frame["execution_net_ev_12h"].mean() * 1e4),
        "cvar05_net_bps": float(_cvar05(frame["execution_net_ev_12h"]) * 1e4),
        "any_touch_rate": float(frame["meaningful_mfe_any_touch"].mean()),
        "clean_first_rate": float(frame["meaningful_mfe_clean_first"].mean()),
        "adverse_first_rate": float(frame["adverse_first"].mean()),
        "exact_cost_opportunity_rate": float(
            frame["path_opportunity_above_exact_cost"].mean()
        ),
        "positive_net_rate": float(frame["exact_net_positive"].mean()),
        "positive_net_given_any_touch": _conditional_rate(
            frame, "meaningful_mfe_any_touch"
        ),
        "positive_net_given_clean_first": _conditional_rate(
            frame, "meaningful_mfe_clean_first"
        ),
        "mean_predicted_adverse_risk": float(
            frame["prediction_p_adverse_0_5r_before_mfe"].mean()
        ),
    }


def _axis_values(
    frame: pd.DataFrame, historical_reference: pd.DataFrame
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    axes: dict[str, pd.Series] = {
        "side": frame["side_name"].astype("string"),
        "asset": frame["__symbol__"].astype("string"),
        "archetype": frame["catboost_archetype"].astype("string"),
        "predicted_adverse_risk": _fixed_band(
            frame["prediction_p_adverse_0_5r_before_mfe"], [0.2, 0.4, 0.6, 0.8]
        ),
        "predicted_stop_risk": _fixed_band(
            frame["prediction_p_stop_1r_before_mfe"], [0.2, 0.4, 0.6, 0.8]
        ),
        "base_direct_disagreement": _fixed_band(
            frame["direct_minus_base_decision_percentile"],
            [-0.40, -0.15, 0.15, 0.40],
        ),
        "entry_atr_fraction": _fixed_band(
            frame["oof_entry_atr_fraction"], [0.005, 0.01, 0.02]
        ),
        "transition_context_coverage": frame[
            "transition_context_available"
        ].map({True: "AVAILABLE", False: "UNAVAILABLE"}),
    }
    threshold_rows: list[dict[str, Any]] = []
    for feature in TRANSITION_FEATURES:
        reference = pd.to_numeric(
            historical_reference.get(feature), errors="coerce"
        ).dropna()
        if len(reference) < 100:
            raise ValueError(f"insufficient pre-May transition reference: {feature}")
        if reference.nunique() < 2:
            raise ValueError(f"degenerate transition reference bands: {feature}")
        cuts = np.unique(reference.quantile([0.25, 0.50, 0.75]).to_numpy(float))
        axes[feature] = _fixed_band(frame[feature], cuts)
        for index, value in enumerate(cuts, start=1):
            threshold_rows.append(
                {
                    "axis": feature,
                    "threshold_index": index,
                    "threshold": float(value),
                    "reference": "all pre-2026-05-01 transition-panel anchors",
                    "outcome_free": True,
                }
            )
    return axes, pd.DataFrame(threshold_rows)


def build_attribution(
    frame: pd.DataFrame, historical_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    axes, thresholds = _axis_values(frame, historical_reference)
    frame = frame.copy()
    for axis, values in axes.items():
        frame[f"slice__{axis}"] = values
    summaries: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []

    for month, month_rows in frame.groupby("candidate_month", sort=True):
        selected_by_score = {
            score: stable_top(month_rows, score, 0.10) for score in SCORES
        }
        base_ids = set(selected_by_score["score_base_alpha"]["candidate_id"])
        direct_ids = set(
            selected_by_score["score_direct_q25_challenger_bps"]["candidate_id"]
        )
        for score, selected in selected_by_score.items():
            low, high = _day_bootstrap(
                selected,
                reps=BOOTSTRAP_REPS,
                seed=_seed_for(str(month), score, "book"),
            )
            summaries.append(
                {
                    "candidate_month": str(month),
                    "score": score,
                    "candidate_rows": int(len(month_rows)),
                    "selected_fraction": float(len(selected) / len(month_rows)),
                    "net_bootstrap_low_bps": low,
                    "net_bootstrap_high_bps": high,
                    "transition_context_coverage": float(
                        selected["transition_context_available"].mean()
                    ),
                    **_book_metrics(selected),
                }
            )
            book_net = float(selected["execution_net_ev_12h"].mean() * 1e4)
            for axis in axes:
                column = f"slice__{axis}"
                for slice_name, local in selected.groupby(
                    column, sort=True, observed=True
                ):
                    low, high = _day_bootstrap(
                        local,
                        reps=BOOTSTRAP_REPS,
                        seed=_seed_for(str(month), score, axis, str(slice_name)),
                    )
                    share = len(local) / len(selected)
                    metrics = _book_metrics(local)
                    slices.append(
                        {
                            "candidate_month": str(month),
                            "score": score,
                            "selection_scope": "pooled_month_global_top10",
                            "axis": axis,
                            "slice": str(slice_name),
                            "book_rows": int(len(selected)),
                            "slice_share": float(share),
                            "slice_minus_book_net_bps": float(
                                metrics["mean_net_bps"] - book_net
                            ),
                            "net_shortfall_contribution_bps": float(
                                share * (metrics["mean_net_bps"] - book_net)
                            ),
                            "net_bootstrap_low_bps": low,
                            "net_bootstrap_high_bps": high,
                            "support_status": (
                                "finding_support"
                                if len(local) >= MIN_FINDING_ROWS
                                and local["candidate_day"].nunique()
                                >= MIN_FINDING_DAYS
                                else "descriptive_underpowered"
                            ),
                            **metrics,
                        }
                    )

        common = month_rows["candidate_id"].isin(base_ids & direct_ids)
        base_only = month_rows["candidate_id"].isin(base_ids - direct_ids)
        direct_only = month_rows["candidate_id"].isin(direct_ids - base_ids)
        for role, mask in (
            ("common", common),
            ("base_only", base_only),
            ("direct_only", direct_only),
        ):
            local = month_rows.loc[mask]
            replacements.append(
                {
                    "candidate_month": str(month),
                    "replacement_role": role,
                    "base_selected_rows": len(base_ids),
                    "direct_selected_rows": len(direct_ids),
                    "overlap_rows": len(base_ids & direct_ids),
                    "jaccard": len(base_ids & direct_ids) / len(base_ids | direct_ids),
                    **_book_metrics(local),
                }
            )

    slice_table = pd.DataFrame(slices)
    # Every axis is exhaustive, including explicit UNAVAILABLE context bands.
    audit = (
        slice_table.groupby(
            ["candidate_month", "score", "axis"], sort=True, observed=True
        )
        .agg(
            slice_rows=("selected_rows", "sum"),
            book_rows=("book_rows", "first"),
            share_sum=("slice_share", "sum"),
            shortfall_sum_bps=("net_shortfall_contribution_bps", "sum"),
        )
        .reset_index()
    )
    if not (audit["slice_rows"] == audit["book_rows"]).all():
        raise ValueError("slice rows do not reconcile to frozen global book")
    if float((audit["share_sum"] - 1.0).abs().max()) > 1e-12:
        raise ValueError("slice shares do not reconcile")
    if float(audit["shortfall_sum_bps"].abs().max()) > 1e-8:
        raise ValueError("slice shortfall contributions do not sum to zero")
    return (
        pd.DataFrame(summaries),
        slice_table,
        pd.DataFrame(replacements),
        thresholds,
    )


def june_july_slice_decomposition(
    summaries: pd.DataFrame, slices: pd.DataFrame
) -> pd.DataFrame:
    """Exact June->July first-period fixed-slice composition decomposition."""

    rows: list[dict[str, Any]] = []
    for score in SCORES:
        book = summaries.loc[summaries["score"].eq(score)].set_index(
            "candidate_month"
        )
        if not {"2026-06", "2026-07"}.issubset(book.index):
            continue
        june_book = float(book.loc["2026-06", "mean_net_bps"])
        july_book = float(book.loc["2026-07", "mean_net_bps"])
        score_rows = slices.loc[slices["score"].eq(score)]
        for axis, axis_rows in score_rows.groupby("axis", sort=True):
            june = axis_rows.loc[
                axis_rows["candidate_month"].eq("2026-06")
            ].set_index("slice")
            july = axis_rows.loc[
                axis_rows["candidate_month"].eq("2026-07")
            ].set_index("slice")
            local_rows: list[dict[str, Any]] = []
            for slice_name in sorted(set(june.index) | set(july.index)):
                share_june = (
                    float(june.loc[slice_name, "slice_share"])
                    if slice_name in june.index
                    else 0.0
                )
                share_july = (
                    float(july.loc[slice_name, "slice_share"])
                    if slice_name in july.index
                    else 0.0
                )
                net_june_reference = (
                    float(june.loc[slice_name, "mean_net_bps"])
                    if slice_name in june.index
                    else june_book
                )
                net_july = (
                    float(july.loc[slice_name, "mean_net_bps"])
                    if slice_name in july.index
                    else net_june_reference
                )
                local_rows.append(
                    {
                        "score": score,
                        "axis": axis,
                        "slice": str(slice_name),
                        "june_share": share_june,
                        "july_share": share_july,
                        "june_net_reference_bps": net_june_reference,
                        "july_net_bps": net_july,
                        "composition_effect_bps": (
                            (share_july - share_june) * net_june_reference
                        ),
                        "within_slice_effect_bps": (
                            share_july * (net_july - net_june_reference)
                        ),
                    }
                )
            composition = sum(
                row["composition_effect_bps"] for row in local_rows
            )
            within = sum(row["within_slice_effect_bps"] for row in local_rows)
            delta = july_book - june_book
            if abs(delta - composition - within) > 1e-8:
                raise ValueError(
                    f"June-July slice decomposition does not reconcile: {score}/{axis}"
                )
            for row in local_rows:
                row.update(
                    {
                        "book_june_net_bps": june_book,
                        "book_july_net_bps": july_book,
                        "book_delta_bps": delta,
                        "axis_composition_effect_bps": composition,
                        "axis_within_slice_effect_bps": within,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def materialize_attribution_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Persist the exact row-level handoff for the next bounded ablations."""

    out = frame.copy()
    for score in SCORES:
        selected = []
        for _, month_rows in out.groupby("candidate_month", sort=True):
            selected.extend(stable_top(month_rows, score, 0.10).index.tolist())
        out[f"selected_global_top10__{score}"] = False
        out.loc[selected, f"selected_global_top10__{score}"] = True
    keep = [
        *IDENTITY_COLUMNS,
        "execution_decision_utc",
        "execution_label_end_utc",
        "label_resolution_utc",
        "candidate_month",
        "candidate_day",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "meaningful_mfe_any_touch",
        "meaningful_mfe_clean_first",
        "path_opportunity_above_exact_cost",
        "exact_net_positive",
        "soft_label",
        "adverse_first",
        "timeout",
        "catboost_archetype",
        "oof_entry_atr_fraction",
        *RISK_COLUMNS,
        "available_at",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "oof_fold",
        "transition_context_available",
        "cohort_anchor_utc",
        *TRANSITION_FEATURES,
        *SCORES,
        *[f"{score}__decision_percentile" for score in SCORES],
        "direct_minus_base_decision_percentile",
        *[f"selected_global_top10__{score}" for score in SCORES],
    ]
    missing = sorted(set(keep).difference(out.columns))
    if missing:
        raise ValueError(f"attribution row handoff missing columns: {missing}")
    result = out.loc[:, keep].copy()
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("attribution row handoff has duplicate identities")
    for score in SCORES:
        expected = (
            result.groupby("candidate_month", sort=True)
            .size()
            .map(lambda rows: int(np.ceil(0.10 * rows)))
        )
        actual = result.groupby("candidate_month", sort=True)[
            f"selected_global_top10__{score}"
        ].sum()
        if not actual.astype(int).eq(expected.astype(int)).all():
            raise ValueError(f"row handoff top10 flag does not reproduce {score}")
    return result


def run(
    *,
    allscore_path: Path = DEFAULT_ALLSCORE,
    grid_path: Path = DEFAULT_GRID,
    context_path: Path = DEFAULT_CONTEXT,
    context_manifest_path: Path = DEFAULT_CONTEXT_MANIFEST,
    risk_path: Path = DEFAULT_RISK,
    risk_manifest_path: Path = DEFAULT_RISK_MANIFEST,
    transition_path: Path = DEFAULT_TRANSITION,
    transition_catalog_path: Path = DEFAULT_TRANSITION_CATALOG,
    transition_manifest_path: Path = DEFAULT_TRANSITION_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    _verify_sources(
        context_path=context_path,
        context_manifest_path=context_manifest_path,
        risk_path=risk_path,
        risk_manifest_path=risk_manifest_path,
        transition_path=transition_path,
        transition_catalog_path=transition_catalog_path,
        transition_manifest_path=transition_manifest_path,
    )
    frame, historical = prepare_attribution_frame(
        pd.read_parquet(allscore_path),
        pd.read_parquet(grid_path),
        pd.read_parquet(context_path),
        pd.read_parquet(risk_path),
        pd.read_parquet(transition_path),
        pd.read_csv(transition_catalog_path),
    )
    summaries, slices, replacements, thresholds = build_attribution(
        frame, historical
    )
    decomposition = june_july_slice_decomposition(summaries, slices)
    row_handoff = materialize_attribution_rows(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, table in (
        ("book_summary", summaries),
        ("slice_metrics", slices),
        ("base_direct_replacements", replacements),
        ("transition_band_thresholds", thresholds),
        ("june_july_slice_decomposition", decomposition),
        ("attribution_rows", row_handoff),
    ):
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        outputs[name] = {
            "path": str(path),
            "rows": int(len(table)),
            "sha256": sha256(path),
        }
    manifest = {
        "schema": "mayjul_failure_attribution_v1",
        "status": "DIAGNOSTIC_ONLY_FROZEN_GLOBAL_BOOKS_NO_LOCAL_RERANK",
        "promotion_eligible": False,
        "scores": list(SCORES),
        "selection": (
            "one pooled monthly global top10 per raw score, selected before "
            "side/asset/risk/transition slicing"
        ),
        "contracts": {
            "opportunity": (
                "h12_u1p5atr any-touch and clean-first remain distinct from "
                "row-cost path opportunity and exact-policy positive-net capture"
            ),
            "risk": (
                "strict side-local OOF; train cutoff < decision and prediction "
                "available no later than decision asserted on every row"
            ),
            "transition": (
                "only v4 catalog decision_time_feature fields; strict current "
                "OOF source; signal context is anchor-1h; unavailable hours are "
                "an explicit slice and never backfilled"
            ),
            "transition_bands": (
                "outcome-free quartiles fixed from pre-May transition-panel anchors"
            ),
            "liquidity": (
                "decision-time global spread-proxy context only; static current "
                "per-asset spread baseline excluded as non-PIT"
            ),
            "causality": (
                "slice associations localize failure but do not prove causal effect"
            ),
            "row_handoff": (
                "targets/outcomes and decision-time candidate context coexist in "
                "one diagnostic table; any training consumer must select features "
                "explicitly and enforce each target's recorded resolution timestamp"
            ),
        },
        "inputs": {
            "allscore": {"path": str(allscore_path), "sha256": sha256(allscore_path)},
            "grid": {"path": str(grid_path), "sha256": sha256(grid_path)},
            "context": {"path": str(context_path), "sha256": sha256(context_path)},
            "risk": {"path": str(risk_path), "sha256": sha256(risk_path)},
            "transition": {
                "path": str(transition_path),
                "sha256": sha256(transition_path),
            },
            "transition_catalog": {
                "path": str(transition_catalog_path),
                "sha256": sha256(transition_catalog_path),
            },
        },
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    (output_dir / "manifest.sha256").write_text(
        f"{sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    return safe(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
