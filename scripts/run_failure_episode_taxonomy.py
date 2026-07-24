#!/usr/bin/env python3
"""Build local and parent failure episodes from a resolved model ledger.

This is the descriptive first stage of failure-state discovery. It never
materializes episode outcomes as inference features.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from extreme_price_movements.residual_event_block_taxonomy import (
    BlockTaxonomyConfig,
    annotate_onset_mechanism_profiles,
    attach_event_blocks,
    block_family_profiles,
    build_block_taxonomy,
    daily_observable_state,
    matched_benign_block_controls,
)
from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    FailureEpisodeConfig,
    build_failure_episodes,
)
from extreme_price_movements.unsupervised_regime_learning.failure_taxonomy_models import (
    FailureTaxonomyModelConfig,
    failure_taxonomy_nonredundancy,
    failure_taxonomy_temporal_stability,
    fit_failure_taxonomy_models,
    fit_frozen_consensus_taxonomy,
)

DEFAULT_LEDGER = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/"
    "frozen_champion_single_source_ledger.parquet"
)
DEFAULT_OUTPUT = Path("data_perp/reports/failure_episode_taxonomy_20260719_v1")
DEFAULT_CANDIDATE_ROOT = DEFAULT_LEDGER.parent / "candidate_shards"
SOURCE_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
    "hit_probability",
    "clean_exec",
    "ev_after_1pct",
    "exec_margin",
    "dirty_positive",
    "base_score",
    "score_meta_base_soft_label",
    "historical_rank",
    "full_path_bad_mae_1r",
    "first_touch_bad_mae_1r",
    "timeout",
    "stop_or_adverse",
    "selected_for_monitor",
    "outcomes_available",
    "evidence_scope",
)
OUTCOME_OR_ID_COLUMNS = {
    "row_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
    "source_tag",
    "ev_after_1pct",
    "exec_margin",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "stop_or_adverse",
    "outcomes_available",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(args.ledger)
    paths = sorted(path.glob("candidates_*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"No source parquet files found under {path}")
    available = set(pq.ParquetFile(paths[0]).schema_arrow.names)
    for shard in paths[1:]:
        available.intersection_update(pq.ParquetFile(shard).schema_arrow.names)
    columns = [name for name in SOURCE_COLUMNS if name in available]
    missing_required = sorted(
        {
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "hit_probability",
            "clean_exec",
            "ev_after_1pct",
        }.difference(columns)
    )
    if missing_required:
        raise KeyError(f"Ledger missing required columns: {missing_required}")
    table = ds.dataset([str(shard) for shard in paths], format="parquet").to_table(
        columns=columns
    )
    frame = table.to_pandas()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].notna()]
    if args.start:
        frame = frame.loc[frame["__ts__"].ge(pd.Timestamp(args.start, tz="UTC"))]
    if args.end:
        frame = frame.loc[frame["__ts__"].lt(pd.Timestamp(args.end, tz="UTC"))]
    if not args.include_all_rows and "selected_for_monitor" in frame:
        frame = frame.loc[frame["selected_for_monitor"].fillna(False).astype(bool)]
    frame = frame.loc[pd.to_numeric(frame["ev_after_1pct"], errors="coerce").notna()]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No resolved rows remain after scope filters")
    start = frame["__ts__"].min()
    end = frame["__ts__"].max()
    span_days = int((end.floor("D") - start.floor("D")).days + 1)
    required_days = int(round(float(args.required_years) * 365.25))
    return frame, {
        "ledger": str(path.resolve()),
        "source_shards": int(len(paths)),
        "parquet_rows": int(
            sum(pq.ParquetFile(shard).metadata.num_rows for shard in paths)
        ),
        "loaded_rows": int(len(frame)),
        "start": start,
        "end": end,
        "span_days": span_days,
        "required_years": float(args.required_years),
        "required_days": required_days,
        "three_year_coverage_pass": bool(span_days >= required_days),
        "coverage_shortfall_days": int(max(0, required_days - span_days)),
        "provenance": args.provenance,
    }


def _observable_candidate_columns(paths: list[Path]) -> list[str]:
    if not paths:
        return []
    common = set(pq.ParquetFile(paths[0]).schema_arrow.names)
    for path in paths[1:]:
        common.intersection_update(pq.ParquetFile(path).schema_arrow.names)
    excluded_tokens = (
        "target",
        "outcome",
        "future",
        "exit_",
        "mfe",
        "mae",
        "timeout",
        "realized",
    )
    return sorted(
        name
        for name in common
        if name not in OUTCOME_OR_ID_COLUMNS
        and name != "selected_top30"
        and not any(token in name.lower() for token in excluded_tokens)
    )


def _stream_daily_observable_state(
    candidate_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted(candidate_root.glob("candidates_*.parquet"))
    features = _observable_candidate_columns(paths)
    daily_parts: list[pd.DataFrame] = []
    rows_read = 0
    for path in paths:
        columns = [
            "__ts__",
            "side_name",
            "archetype_policy_key",
            "selected_top30",
            *features,
        ]
        part = pd.read_parquet(path, columns=columns)
        rows_read += len(part)
        daily_parts.append(
            daily_observable_state(part, features=features, selected_only=True)
        )
    if not daily_parts:
        return pd.DataFrame(), {
            "candidate_root": str(candidate_root),
            "candidate_shards": 0,
            "candidate_rows": 0,
            "observable_features": 0,
        }
    daily = pd.concat(daily_parts, ignore_index=True)
    daily = (
        daily.groupby(
            ["day", "side_name", "archetype_policy_key"],
            observed=True,
            as_index=False,
        )[features]
        .median()
        .sort_values(["day", "side_name", "archetype_policy_key"], kind="stable")
    )
    return daily, {
        "candidate_root": str(candidate_root.resolve()),
        "candidate_shards": int(len(paths)),
        "candidate_rows": int(rows_read),
        "observable_features": int(len(features)),
        "observable_feature_names": features,
        "snapshot_contract": "first timestamp of each UTC day, then top30 and cross-sectional median",
    }


def _local_model_error_shape(source: pd.DataFrame) -> pd.DataFrame:
    """Materialize local ex-post base/meta residual vectors for taxonomy only."""
    work = source.copy(deep=False)
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    clean = pd.to_numeric(work["clean_exec"], errors="coerce")
    base = pd.to_numeric(work.get("base_score"), errors="coerce")
    meta = pd.to_numeric(work.get("score_meta_base_soft_label"), errors="coerce")
    ev = pd.to_numeric(work["ev_after_1pct"], errors="coerce")
    rank = pd.to_numeric(work.get("historical_rank"), errors="coerce")
    work["_base_residual"] = clean - base
    work["_meta_residual"] = clean - meta
    work["_base_brier"] = work["_base_residual"].pow(2)
    work["_meta_brier"] = work["_meta_residual"].pow(2)
    base_clipped = base.clip(1e-6, 1.0 - 1e-6)
    meta_clipped = meta.clip(1e-6, 1.0 - 1e-6)
    work["_base_log_loss"] = -(
        clean * np.log(base_clipped) + (1.0 - clean) * np.log1p(-base_clipped)
    )
    work["_meta_log_loss"] = -(
        clean * np.log(meta_clipped) + (1.0 - clean) * np.log1p(-meta_clipped)
    )
    base_meta_valid = base.notna() & meta.notna()
    work["_base_meta_disagreement"] = (base - meta).abs().where(base_meta_valid)
    work["_base_meta_sign_disagreement"] = (
        base.ge(0.5).ne(meta.ge(0.5)).astype(np.float32).where(base_meta_valid)
    )
    work["_meta_abs_error_delta"] = (
        work["_meta_residual"].abs() - work["_base_residual"].abs()
    )
    work["_meta_corrects_base"] = (
        work["_meta_abs_error_delta"]
        .lt(0.0)
        .astype(np.float32)
        .where(base_meta_valid & clean.notna())
    )
    work["_base_false_positive"] = (
        (base.ge(0.5) & clean.lt(0.5)).astype(np.float32).where(
            base.notna() & clean.notna()
        )
    )
    work["_meta_false_positive"] = (
        (meta.ge(0.5) & clean.lt(0.5)).astype(np.float32).where(
            meta.notna() & clean.notna()
        )
    )
    work["_dirty_positive"] = pd.to_numeric(work.get("dirty_positive"), errors="coerce")
    work["_first_bad_mae"] = pd.to_numeric(
        work.get("first_touch_bad_mae_1r"), errors="coerce"
    )
    work["_full_bad_mae"] = pd.to_numeric(
        work.get("full_path_bad_mae_1r"), errors="coerce"
    )
    work["_rank"] = rank
    work["_ev"] = ev
    keys = ["day", "side_name", "archetype_policy_key"]
    rows: list[dict[str, Any]] = []
    for values, group in work.groupby(keys, observed=True, sort=True):
        rank_pair = group.loc[:, ["_rank", "_ev"]].dropna()
        ranking = (
            float(rank_pair["_rank"].corr(rank_pair["_ev"], method="spearman"))
            if len(rank_pair) >= 4
            and rank_pair["_rank"].nunique() >= 2
            and rank_pair["_ev"].nunique() >= 2
            else np.nan
        )

        def calibration_fit(score_name: str) -> tuple[float, float]:
            pair = group.loc[:, [score_name]].copy()
            pair["clean"] = clean.reindex(group.index)
            pair = pair.dropna()
            if len(pair) < 4 or float(pair[score_name].var(ddof=0)) <= 1e-12:
                return np.nan, np.nan
            slope = float(
                np.cov(
                    pair[score_name].to_numpy(np.float64),
                    pair["clean"].to_numpy(np.float64),
                    ddof=0,
                )[0, 1]
                / pair[score_name].var(ddof=0)
            )
            intercept = float(pair["clean"].mean() - slope * pair[score_name].mean())
            return slope, intercept

        base_slope, base_intercept = calibration_fit("base_score")
        meta_slope, meta_intercept = calibration_fit("score_meta_base_soft_label")
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "expost__base_signed_residual": float(group["_base_residual"].mean()),
                "expost__base_absolute_residual": float(
                    group["_base_residual"].abs().mean()
                ),
                "expost__base_brier": float(group["_base_brier"].mean()),
                "expost__base_log_loss": float(group["_base_log_loss"].mean()),
                "expost__base_calibration_slope": base_slope,
                "expost__base_calibration_intercept": base_intercept,
                "expost__meta_signed_residual": float(group["_meta_residual"].mean()),
                "expost__meta_absolute_residual": float(
                    group["_meta_residual"].abs().mean()
                ),
                "expost__meta_brier": float(group["_meta_brier"].mean()),
                "expost__meta_log_loss": float(group["_meta_log_loss"].mean()),
                "expost__meta_calibration_slope": meta_slope,
                "expost__meta_calibration_intercept": meta_intercept,
                "expost__base_meta_disagreement": float(
                    group["_base_meta_disagreement"].mean()
                ),
                "expost__base_meta_sign_disagreement_rate": float(
                    group["_base_meta_sign_disagreement"].mean()
                ),
                "expost__meta_abs_error_delta": float(
                    group["_meta_abs_error_delta"].mean()
                ),
                "expost__meta_correction_rate": float(
                    group["_meta_corrects_base"].mean()
                ),
                "expost__base_false_positive_rate": float(
                    group["_base_false_positive"].mean()
                ),
                "expost__meta_false_positive_rate": float(
                    group["_meta_false_positive"].mean()
                ),
                "expost__ranking_spearman": ranking,
                "expost__dirty_positive_rate": float(group["_dirty_positive"].mean()),
                "expost__first_touch_bad_mae_rate": float(
                    group["_first_bad_mae"].mean()
                ),
                "expost__full_path_bad_mae_rate": float(group["_full_bad_mae"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _local_adverse_calendar(result: Any, source: pd.DataFrame) -> pd.DataFrame:
    cells = result.local.daily_cells.copy()
    error_shape_columns = (
        "signed_hit_surprise",
        "residual_variance",
        "expected_clean_rate",
        "clean_rate",
        "mean_ev_after_cost",
        "worst_ev",
        "loss_rate",
        "bad_mae_rate",
        "timeout_rate",
        "loss_mean",
        "mean_positive_ev",
        "mean_negative_ev",
        "acute_adverse_rate",
        "slow_timeout_loss_rate",
        "clean_negative_ev_rate",
        "dirty_negative_ev_rate",
        "durable_clean_positive_rate",
        "payoff_asymmetry",
        "surprise_ac_rolling",
        "loss_ac_rolling",
    )
    for name in error_shape_columns:
        if name in cells:
            cells[f"expost__{name}"] = pd.to_numeric(cells[name], errors="coerce")
    cells["day"] = pd.to_datetime(cells["day"], utc=True).dt.floor("D")
    adverse_ids = set(
        result.local.events.loc[
            result.local.events.get("event_class", "").isin(
                ["adverse", "payoff_disagreement"]
            ),
            "event_id",
        ].astype(str)
    )
    membership = result.local.event_membership.copy()
    membership["day"] = pd.to_datetime(membership["day"], utc=True).dt.floor("D")
    membership["adverse_calendar_cell"] = (
        membership["event_id"].astype(str).isin(adverse_ids)
    )
    flags = membership.groupby(
        ["day", "side_name", "archetype_policy_key"],
        observed=True,
        as_index=False,
    )["adverse_calendar_cell"].max()
    calendar = cells.merge(
        flags,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
    )
    model_error = _local_model_error_shape(source)
    calendar = calendar.merge(
        model_error,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_one",
    )
    calendar["adverse_calendar_cell"] = (
        calendar["adverse_calendar_cell"].astype("boolean").fillna(False).astype(bool)
    )
    return calendar.rename(
        columns={
            "mean_ev_after_cost": "mean_ev_after_1pct",
            "clean_rate": "clean_exec_rate",
            "signed_hit_surprise": "signed_surprise",
        }
    )


def _parent_market_calendar(result: Any) -> pd.DataFrame:
    """Materialize a global event calendar without leaking outcomes into state."""

    daily = result.daily_global.copy()
    daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.floor("D")
    active_days = set(pd.to_datetime(result.parent_membership["day"], utc=True))
    calendar = pd.DataFrame(
        {
            "day": daily["day"],
            "side_name": "global",
            "archetype_policy_key": "global_market",
            "adverse_calendar_cell": daily["day"].isin(active_days),
            "negative_pnl_day": daily["negative_pnl_day"],
            "selected_rows": daily["selected_rows"],
            "mean_ev_after_1pct": daily["mean_ev"],
            "signed_surprise": daily.get("expost__signed_residual", np.nan),
        }
    )
    for name in daily.columns:
        if name.startswith("expost__"):
            calendar[name] = pd.to_numeric(daily[name], errors="coerce")
    return calendar


def _parent_market_state(daily_state: pd.DataFrame) -> pd.DataFrame:
    """Collapse local observable states to one broad market state per day."""

    features = [
        name
        for name in daily_state.columns
        if name not in {"day", "side_name", "archetype_policy_key"}
    ]
    state = (
        daily_state.groupby("day", observed=True, as_index=False)[features]
        .median(numeric_only=True)
        .sort_values("day", kind="stable")
    )
    state["side_name"] = "global"
    state["archetype_policy_key"] = "global_market"
    return state.loc[:, ["day", "side_name", "archetype_policy_key", *features]]


PARENT_MAX_EVENT_DAYS = 14


def _calendar_with_event_blocks(
    calendar: pd.DataFrame,
    *,
    max_event_days: int | None = None,
) -> pd.DataFrame:
    """Expose stable event-block IDs for chronological detector training."""

    return attach_event_blocks(calendar, max_event_days=max_event_days)


def _mixture_profiles(
    taxonomy: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    if taxonomy.empty or assignments.empty:
        return pd.DataFrame()
    joined = assignments.merge(
        taxonomy.reset_index().rename(columns={"index": "source_index"}),
        on=[
            "source_index",
            "side_name",
            "archetype_policy_key",
            "event_block",
            "event_start",
            "event_end",
        ],
        how="left",
        validate="one_to_one",
    )
    family_columns = [name for name in joined if name.startswith("family__")]
    rows: list[dict[str, Any]] = []
    group_columns = [
        "side_name",
        "archetype_policy_key",
        "method",
        "latent_dim",
        "clusters",
        "cluster_id",
    ]
    for keys, group in joined.groupby(group_columns, observed=True, sort=True):
        means = group[family_columns].mean(numeric_only=True)
        dominant = means.abs().sort_values(ascending=False).head(8)
        dominant_text = "|".join(
            f"{name}={means[name]:+.3f}" for name in dominant.index
        )
        mechanism_scores: dict[str, float] = {}
        for name, value in means.items():
            if not name.startswith("family__"):
                continue
            mechanism = name.split("__", 2)[1]
            if mechanism in {"error_shape", "error_vector"}:
                continue
            mechanism_scores[mechanism] = max(
                mechanism_scores.get(mechanism, 0.0), abs(float(value))
            )
        mechanism = (
            max(mechanism_scores, key=mechanism_scores.get)
            if mechanism_scores
            else "unresolved_market_state"
        )
        mean_ev = float(group["calendar_mean_ev"].mean())
        mean_surprise = float(group["calendar_mean_signed_surprise"].mean())

        def error_level(suffix: str) -> float:
            name = f"calendar_error__{suffix}"
            if name not in group:
                return np.nan
            values = pd.to_numeric(group[name], errors="coerce")
            return float(values.mean()) if values.notna().any() else np.nan

        ranking = error_level("ranking_spearman")
        meta_residual = error_level("meta_signed_residual")
        meta_abs_error_delta = error_level("meta_abs_error_delta")
        meta_correction = error_level("meta_correction_rate")
        meta_false_positive = error_level("meta_false_positive_rate")
        disagreement = error_level("base_meta_disagreement")
        timeout_rate = error_level("timeout_rate")
        dirty_rate = error_level("dirty_positive_rate")
        bad_mae_rate = error_level("full_path_bad_mae_rate")
        if (
            (np.isfinite(ranking) and ranking < -0.05)
            or (
                np.isfinite(meta_false_positive)
                and meta_false_positive >= 0.65
                and mean_surprise <= -0.08
            )
        ):
            error_shape = "directional_inversion"
        elif (
            np.isfinite(meta_abs_error_delta)
            and meta_abs_error_delta >= 0.02
            and (
                not np.isfinite(meta_correction)
                or meta_correction < 0.40
            )
            and (not np.isfinite(disagreement) or disagreement >= 0.05)
        ):
            error_shape = "meta_amplification"
        elif np.isfinite(timeout_rate) and timeout_rate >= 0.35:
            error_shape = "slow_timeout_failure"
        elif (
            (np.isfinite(bad_mae_rate) and bad_mae_rate >= 0.65)
            or (np.isfinite(dirty_rate) and dirty_rate >= 0.50)
        ):
            error_shape = "adverse_path_failure"
        elif np.isfinite(ranking) and ranking <= 0.05:
            error_shape = "ranking_collapse"
        elif (
            (np.isfinite(meta_residual) and meta_residual <= -0.05)
            or mean_surprise <= -0.05
        ):
            error_shape = "overconfident_false_positive"
        elif np.isfinite(meta_residual) and meta_residual >= 0.05:
            error_shape = "underconfident_missed_opportunity"
        elif mean_ev <= -0.005 and mean_surprise >= -0.03:
            error_shape = "payoff_asymmetry"
        elif mean_ev < 0.0:
            error_shape = "negative_ev_path_failure"
        else:
            error_shape = "adverse_label_event_positive_ev"
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "semantic_label": f"{error_shape}__{mechanism}",
                "blocks": int(len(group)),
                "first_event_start": group["event_start"].min(),
                "last_event_end": group["event_end"].max(),
                "active_months": int(
                    pd.to_datetime(group["event_start"], utc=True)
                    .dt.strftime("%Y-%m")
                    .nunique()
                ),
                "mean_calendar_ev": mean_ev,
                "worst_calendar_ev": float(group["calendar_mean_ev"].min()),
                "mean_signed_surprise": mean_surprise,
                "mean_ranking_spearman": ranking,
                "mean_meta_signed_residual": meta_residual,
                "mean_meta_abs_error_delta": meta_abs_error_delta,
                "mean_meta_correction_rate": meta_correction,
                "mean_meta_false_positive_rate": meta_false_positive,
                "mean_base_meta_disagreement": disagreement,
                "mean_timeout_rate": timeout_rate,
                "mean_dirty_positive_rate": dirty_rate,
                "mean_full_path_bad_mae_rate": bad_mae_rate,
                "mean_posterior_max": float(group["cluster_posterior_max"].mean()),
                "mean_cluster_entropy": float(group["cluster_entropy"].mean()),
                "dominant_trajectory_features": dominant_text,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_calendar_ev", "blocks"], ascending=[True, False], kind="stable"
    )


def _semantic_failure_assignments(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Attach stable semantic labels to the selected mixture assignments."""

    if assignments.empty:
        return pd.DataFrame()
    keys = [
        "side_name",
        "archetype_policy_key",
        "method",
        "latent_dim",
        "clusters",
        "cluster_id",
    ]
    result = assignments.copy()
    if not profiles.empty and set(keys).issubset(profiles.columns):
        profile_columns = [
            *keys,
            *[
                name
                for name in (
                    "semantic_label",
                    "blocks",
                    "active_months",
                    "mean_calendar_ev",
                    "worst_calendar_ev",
                    "mean_signed_surprise",
                )
                if name in profiles.columns
            ],
        ]
        result = result.merge(
            profiles.loc[:, profile_columns].drop_duplicates(keys),
            on=keys,
            how="left",
            validate="many_to_one",
        )
    result["failure_mode_id"] = (
        result["method"].astype(str)
        + "__d"
        + result["latent_dim"].astype(str)
        + "__k"
        + result["clusters"].astype(str)
        + "__c"
        + result["cluster_id"].astype(str)
    )
    if "semantic_label" not in result:
        result["semantic_label"] = pd.NA
    result["semantic_label"] = result["semantic_label"].fillna(
        result["failure_mode_id"]
    )
    return result


def _negative_day_mode_catalog(
    daily_global: pd.DataFrame,
    parent_calendar: pd.DataFrame,
    parent_assignments: pd.DataFrame,
    parent_profiles: pd.DataFrame,
    local_calendar: pd.DataFrame,
    local_assignments: pd.DataFrame,
    local_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Map every negative-PnL day to parent and contributing local modes.

    This artifact is descriptive and deliberately contains realized PnL and
    ex-post mode labels. It must never be joined to an inference feature frame.
    """

    daily = daily_global.copy()
    daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.floor("D")
    daily = daily.loc[daily["negative_pnl_day"].fillna(False).astype(bool)].copy()
    if daily.empty:
        return pd.DataFrame()
    base_columns = [
        name
        for name in ("day", "net_ev", "mean_ev", "selected_rows", "distinct_assets")
        if name in daily.columns
    ]
    output = daily.loc[:, base_columns].copy()

    parent = parent_calendar.copy()
    parent["day"] = pd.to_datetime(parent["day"], utc=True).dt.floor("D")
    parent_semantic = _semantic_failure_assignments(
        parent_assignments, parent_profiles
    )
    parent_columns = [
        "side_name",
        "archetype_policy_key",
        "event_block",
        "failure_mode_id",
        "semantic_label",
        "cluster_posterior_max",
        "cluster_entropy",
    ]
    if not parent_semantic.empty:
        parent = parent.merge(
            parent_semantic.loc[
                :, [name for name in parent_columns if name in parent_semantic.columns]
            ].drop_duplicates(
                ["side_name", "archetype_policy_key", "event_block"]
            ),
            on=["side_name", "archetype_policy_key", "event_block"],
            how="left",
            validate="many_to_one",
        )
    output = output.merge(
        parent.loc[
            :,
            [
                name
                for name in (
                    "day",
                    "event_block",
                    "failure_mode_id",
                    "semantic_label",
                    "cluster_posterior_max",
                    "cluster_entropy",
                )
                if name in parent.columns
            ],
        ].rename(
            columns={
                "event_block": "parent_event_block",
                "failure_mode_id": "parent_failure_mode_id",
                "semantic_label": "parent_semantic_label",
                "cluster_posterior_max": "parent_mode_posterior_max",
                "cluster_entropy": "parent_mode_entropy",
            }
        ),
        on="day",
        how="left",
        validate="one_to_one",
    )

    local = local_calendar.copy()
    local["day"] = pd.to_datetime(local["day"], utc=True).dt.floor("D")
    if "adverse_event" in local:
        local = local.loc[local["adverse_event"].fillna(False).astype(bool)].copy()
    elif "adverse_calendar_cell" in local:
        local = local.loc[
            local["adverse_calendar_cell"].fillna(False).astype(bool)
        ].copy()
    local_semantic = _semantic_failure_assignments(local_assignments, local_profiles)
    if not local.empty and not local_semantic.empty:
        semantic_columns = [
            "side_name",
            "archetype_policy_key",
            "event_block",
            "failure_mode_id",
            "semantic_label",
        ]
        local = local.merge(
            local_semantic.loc[:, semantic_columns].drop_duplicates(
                ["side_name", "archetype_policy_key", "event_block"]
            ),
            on=["side_name", "archetype_policy_key", "event_block"],
            how="left",
            validate="many_to_one",
        )
    if not local.empty:
        if "failure_mode_id" not in local:
            local["failure_mode_id"] = pd.NA
        if "semantic_label" not in local:
            local["semantic_label"] = pd.NA
        local["local_mode_descriptor"] = (
            local["side_name"].astype(str)
            + "::"
            + local["archetype_policy_key"].astype(str)
            + "::"
            + local.get("semantic_label", pd.Series(pd.NA, index=local.index))
            .fillna("unresolved")
            .astype(str)
        )
        local_summary = (
            local.groupby("day", observed=True, as_index=False)
            .agg(
                active_local_failure_cells=("event_block", "size"),
                assigned_local_failure_cells=(
                    "failure_mode_id",
                    lambda values: int(pd.Series(values).notna().sum()),
                ),
                local_failure_modes=(
                    "local_mode_descriptor",
                    lambda values: "|".join(sorted(set(map(str, values)))),
                ),
            )
        )
        output = output.merge(local_summary, on="day", how="left", validate="one_to_one")
    for name in ("active_local_failure_cells", "assigned_local_failure_cells"):
        if name not in output:
            output[name] = 0
        output[name] = pd.to_numeric(output[name], errors="coerce").fillna(0).astype(int)
    if "local_failure_modes" not in output:
        output["local_failure_modes"] = ""
    output["local_failure_modes"] = output["local_failure_modes"].fillna("")
    output["parent_mode_assigned"] = output.get(
        "parent_failure_mode_id", pd.Series(pd.NA, index=output.index)
    ).notna()
    output["all_active_local_modes_assigned"] = output[
        "assigned_local_failure_cells"
    ].ge(output["active_local_failure_cells"])
    return output.sort_values("day", kind="stable").reset_index(drop=True)


def _failure_mode_composition_audit(
    source: pd.DataFrame,
    calendar: pd.DataFrame,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    parent_scope: bool,
) -> pd.DataFrame:
    """Report whether a mode is only one symbol, side, or archetype."""

    semantic = _semantic_failure_assignments(assignments, profiles)
    if source.empty or calendar.empty or semantic.empty:
        return pd.DataFrame()
    rows = source.loc[
        :,
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
    ].copy()
    rows["day"] = pd.to_datetime(rows["__ts__"], utc=True).dt.floor("D")
    event_calendar = calendar.copy()
    event_calendar["day"] = pd.to_datetime(
        event_calendar["day"], utc=True
    ).dt.floor("D")
    if "adverse_event" in event_calendar:
        event_calendar = event_calendar.loc[
            event_calendar["adverse_event"].fillna(False).astype(bool)
        ]
    join_keys = ["day"] if parent_scope else [
        "day",
        "side_name",
        "archetype_policy_key",
    ]
    rows = rows.merge(
        event_calendar.loc[:, [*join_keys, "event_block"]],
        on=join_keys,
        how="inner",
        validate="many_to_one",
    )
    assignment_keys = ["event_block"] if parent_scope else [
        "side_name",
        "archetype_policy_key",
        "event_block",
    ]
    assignment_columns = [
        *assignment_keys,
        "failure_mode_id",
        "semantic_label",
        "method",
        "latent_dim",
        "clusters",
        "cluster_id",
    ]
    rows = rows.merge(
        semantic.loc[:, assignment_columns].drop_duplicates(assignment_keys),
        on=assignment_keys,
        how="inner",
        validate="many_to_one",
    )
    reports: list[dict[str, Any]] = []
    group_columns = [
        "failure_mode_id",
        "semantic_label",
        "method",
        "latent_dim",
        "clusters",
        "cluster_id",
    ]
    for values, group in rows.groupby(group_columns, observed=True, sort=True):
        symbol_share = group["__symbol__"].astype(str).value_counts(normalize=True)
        side_share = group["side_name"].astype(str).value_counts(normalize=True)
        archetype_share = (
            group["archetype_policy_key"].astype(str).value_counts(normalize=True)
        )
        max_symbol = float(symbol_share.max()) if len(symbol_share) else np.nan
        max_side = float(side_share.max()) if len(side_share) else np.nan
        max_archetype = (
            float(archetype_share.max()) if len(archetype_share) else np.nan
        )
        reports.append(
            {
                **dict(zip(group_columns, values, strict=True)),
                "rows": int(len(group)),
                "days": int(group["day"].nunique()),
                "distinct_symbols": int(group["__symbol__"].nunique()),
                "dominant_symbol": str(symbol_share.index[0]) if len(symbol_share) else "",
                "dominant_symbol_fraction": max_symbol,
                "distinct_sides": int(group["side_name"].nunique()),
                "dominant_side": str(side_share.index[0]) if len(side_share) else "",
                "dominant_side_fraction": max_side,
                "distinct_archetypes": int(group["archetype_policy_key"].nunique()),
                "dominant_archetype": (
                    str(archetype_share.index[0]) if len(archetype_share) else ""
                ),
                "dominant_archetype_fraction": max_archetype,
                "composition_redundancy_warning": bool(
                    (np.isfinite(max_symbol) and max_symbol >= 0.80)
                    or (
                        parent_scope
                        and (
                            (np.isfinite(max_side) and max_side >= 0.80)
                            or (
                                np.isfinite(max_archetype)
                                and max_archetype >= 0.80
                            )
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(reports).sort_values(
        ["composition_redundancy_warning", "rows"],
        ascending=[True, False],
        kind="stable",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source, source_manifest = _load_source(args)
    config = FailureEpisodeConfig(
        significance_z=float(args.significance_z),
        causal_min_days=int(args.causal_min_days),
        join_gap_days=int(args.join_gap_days),
        min_selected_rows=int(args.min_selected_rows),
        min_parent_cells=int(args.min_parent_cells),
        bootstrap_draws=int(args.bootstrap_draws),
    )
    result = build_failure_episodes(source, config)
    result.daily_global.to_parquet(output / "daily_global_health.parquet", index=False)
    result.local.daily_cells.to_parquet(
        output / "daily_side_archetype_health.parquet", index=False
    )
    result.local.events.to_parquet(
        output / "local_failure_episodes.parquet", index=False
    )
    result.local.event_membership.to_parquet(
        output / "local_failure_membership.parquet", index=False
    )
    result.parent_episodes.to_parquet(
        output / "parent_failure_episodes.parquet", index=False
    )
    result.parent_membership.to_parquet(
        output / "parent_failure_membership.parquet", index=False
    )
    result.coverage.to_csv(output / "negative_pnl_day_coverage.csv", index=False)
    result.local.summary.to_csv(
        output / "local_summary_side_archetype.csv", index=False
    )
    result.local.sensitivity.to_csv(
        output / "event_boundary_sensitivity.csv", index=False
    )
    daily_state, state_manifest = _stream_daily_observable_state(
        Path(args.candidate_root)
    )
    taxonomy_rows = profiles = controls = mixture_assignments = mixture_diagnostics = (
        mixture_profiles
    ) = pd.DataFrame()
    mixture_nonredundancy = pd.DataFrame()
    mixture_temporal_stability = pd.DataFrame()
    parent_taxonomy = parent_profiles = parent_assignments = parent_diagnostics = (
        parent_mixture_profiles
    ) = pd.DataFrame()
    parent_nonredundancy = pd.DataFrame()
    parent_temporal_stability = pd.DataFrame()
    frozen_local_assignments = pd.DataFrame()
    frozen_local_diagnostics = pd.DataFrame()
    frozen_parent_assignments = pd.DataFrame()
    frozen_parent_diagnostics = pd.DataFrame()
    frozen_local_state: dict[str, Any] = {}
    frozen_parent_state: dict[str, Any] = {}
    frozen_local_profiles = pd.DataFrame()
    frozen_parent_profiles = pd.DataFrame()
    frozen_local_semantic_assignments = pd.DataFrame()
    frozen_parent_semantic_assignments = pd.DataFrame()
    negative_day_modes = pd.DataFrame()
    descriptive_negative_day_modes = pd.DataFrame()
    local_semantic_assignments = pd.DataFrame()
    parent_semantic_assignments = pd.DataFrame()
    local_mode_composition = pd.DataFrame()
    parent_mode_composition = pd.DataFrame()
    if not daily_state.empty:
        calendar = _local_adverse_calendar(result, source)
        # Episode construction needs the wide candidate ledger, but the
        # remaining taxonomy stages only need row identity for composition
        # audits. Releasing hundreds of observable columns here keeps the
        # representation grid within a bounded memory footprint.
        source = source.loc[
            :,
            ["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
        ].copy()
        gc.collect()
        reference_end = pd.Timestamp(args.prospective_taxonomy_reference_end)
        if reference_end.tzinfo is None:
            raise ValueError("--prospective-taxonomy-reference-end must be timezone-aware")
        reference_end = reference_end.tz_convert("UTC")
        reuse_root = args.reuse_local_taxonomy_root
        if reuse_root is not None:
            reuse_root = Path(reuse_root)
            taxonomy_rows = pd.read_parquet(
                reuse_root / "local_failure_block_taxonomy.parquet"
            )
            profiles = pd.read_csv(reuse_root / "local_failure_family_profiles.csv")
            controls = pd.read_parquet(reuse_root / "matched_benign_controls.parquet")
            mixture_assignments = pd.read_parquet(
                reuse_root / "local_failure_mixture_assignments.parquet"
            )
            mixture_diagnostics = pd.read_csv(
                reuse_root / "local_failure_mixture_diagnostics.csv"
            )
            mixture_profiles = pd.read_csv(
                reuse_root / "local_failure_mixture_profiles.csv"
            )
            mixture_nonredundancy = pd.read_csv(
                reuse_root / "local_failure_mixture_nonredundancy.csv"
            )
            mixture_temporal_stability = pd.read_csv(
                reuse_root / "local_failure_mode_temporal_stability.csv"
            )
            frozen_local_assignments = pd.read_parquet(
                reuse_root / "local_frozen_failure_mode_assignments.parquet"
            )
            frozen_local_diagnostics = pd.read_csv(
                reuse_root / "local_frozen_failure_mode_diagnostics.csv"
            )
            frozen_local_profiles = pd.read_csv(
                reuse_root / "local_frozen_failure_mode_profiles.csv"
            )
            frozen_local_semantic_assignments = pd.read_parquet(
                reuse_root / "local_frozen_failure_mode_semantic_assignments.parquet"
            )
            frozen_local_state = json.loads(
                (reuse_root / "local_frozen_failure_taxonomy_state.json").read_text(
                    encoding="utf-8"
                )
            )
            reused_reference = pd.Timestamp(frozen_local_state["reference_end"])
            if reused_reference.tzinfo is None:
                raise ValueError("Reused local taxonomy reference_end must be aware")
            if reused_reference.tz_convert("UTC") != reference_end:
                raise ValueError(
                    "Reused local taxonomy reference cutoff does not match this run"
                )
        else:
            taxonomy_rows, trajectories = build_block_taxonomy(calendar, daily_state)
            taxonomy_rows = annotate_onset_mechanism_profiles(taxonomy_rows)
            profiles = block_family_profiles(taxonomy_rows)
            controls = matched_benign_block_controls(
                calendar,
                daily_state,
                taxonomy_rows,
            )
            mixture_assignments, mixture_diagnostics = fit_failure_taxonomy_models(
                taxonomy_rows,
                config=FailureTaxonomyModelConfig(
                    min_cluster_episodes=int(args.min_cluster_episodes),
                ),
            )
            mixture_nonredundancy = failure_taxonomy_nonredundancy(
                taxonomy_rows, mixture_assignments
            )
            mixture_temporal_stability = failure_taxonomy_temporal_stability(
                taxonomy_rows, mixture_assignments
            )
            mixture_profiles = _mixture_profiles(taxonomy_rows, mixture_assignments)
            (
                frozen_local_assignments,
                frozen_local_diagnostics,
                frozen_local_state,
            ) = fit_frozen_consensus_taxonomy(
                taxonomy_rows,
                reference_end=reference_end,
                config=FailureTaxonomyModelConfig(
                    min_cluster_episodes=int(args.min_cluster_episodes),
                ),
            )
            frozen_local_profiles = _mixture_profiles(
                taxonomy_rows, frozen_local_assignments
            )
            frozen_local_semantic_assignments = _semantic_failure_assignments(
                frozen_local_assignments, frozen_local_profiles
            )
        parent_calendar = _parent_market_calendar(result)
        parent_state = _parent_market_state(daily_state)
        parent_taxonomy, _ = build_block_taxonomy(
            parent_calendar,
            parent_state,
            config=BlockTaxonomyConfig(max_event_days=PARENT_MAX_EVENT_DAYS),
        )
        parent_taxonomy = annotate_onset_mechanism_profiles(parent_taxonomy)
        parent_profiles = block_family_profiles(parent_taxonomy)
        parent_assignments, parent_diagnostics = fit_failure_taxonomy_models(
            parent_taxonomy,
            config=FailureTaxonomyModelConfig(
                min_cluster_episodes=int(args.min_cluster_episodes),
            ),
        )
        parent_nonredundancy = failure_taxonomy_nonredundancy(
            parent_taxonomy, parent_assignments
        )
        parent_temporal_stability = failure_taxonomy_temporal_stability(
            parent_taxonomy, parent_assignments
        )
        parent_mixture_profiles = _mixture_profiles(parent_taxonomy, parent_assignments)
        (
            frozen_parent_assignments,
            frozen_parent_diagnostics,
            frozen_parent_state,
        ) = fit_frozen_consensus_taxonomy(
            parent_taxonomy,
            reference_end=reference_end,
            config=FailureTaxonomyModelConfig(
                min_cluster_episodes=int(args.min_cluster_episodes),
            ),
        )
        frozen_parent_profiles = _mixture_profiles(
            parent_taxonomy, frozen_parent_assignments
        )
        frozen_parent_semantic_assignments = _semantic_failure_assignments(
            frozen_parent_assignments, frozen_parent_profiles
        )
        local_semantic_assignments = _semantic_failure_assignments(
            mixture_assignments, mixture_profiles
        )
        parent_semantic_assignments = _semantic_failure_assignments(
            parent_assignments, parent_mixture_profiles
        )
        descriptive_negative_day_modes = _negative_day_mode_catalog(
            result.daily_global,
            _calendar_with_event_blocks(
                parent_calendar,
                max_event_days=PARENT_MAX_EVENT_DAYS,
            ),
            parent_assignments,
            parent_mixture_profiles,
            _calendar_with_event_blocks(calendar),
            mixture_assignments,
            mixture_profiles,
        )
        negative_day_modes = _negative_day_mode_catalog(
            result.daily_global,
            _calendar_with_event_blocks(
                parent_calendar,
                max_event_days=PARENT_MAX_EVENT_DAYS,
            ),
            frozen_parent_assignments,
            frozen_parent_profiles,
            _calendar_with_event_blocks(calendar),
            frozen_local_assignments,
            frozen_local_profiles,
        )
        local_mode_composition = _failure_mode_composition_audit(
            source,
            _calendar_with_event_blocks(calendar),
            mixture_assignments,
            mixture_profiles,
            parent_scope=False,
        )
        parent_mode_composition = _failure_mode_composition_audit(
            source,
            _calendar_with_event_blocks(
                parent_calendar,
                max_event_days=PARENT_MAX_EVENT_DAYS,
            ),
            parent_assignments,
            parent_mixture_profiles,
            parent_scope=True,
        )
        daily_state.to_parquet(output / "daily_observable_state.parquet", index=False)
        _calendar_with_event_blocks(calendar).to_parquet(
            output / "local_adverse_calendar.parquet", index=False
        )
        taxonomy_rows.to_parquet(
            output / "local_failure_block_taxonomy.parquet", index=False
        )
        profiles.to_csv(output / "local_failure_family_profiles.csv", index=False)
        controls.to_parquet(output / "matched_benign_controls.parquet", index=False)
        mixture_assignments.to_parquet(
            output / "local_failure_mixture_assignments.parquet", index=False
        )
        mixture_diagnostics.to_csv(
            output / "local_failure_mixture_diagnostics.csv", index=False
        )
        mixture_profiles.to_csv(
            output / "local_failure_mixture_profiles.csv", index=False
        )
        mixture_nonredundancy.to_csv(
            output / "local_failure_mixture_nonredundancy.csv", index=False
        )
        mixture_temporal_stability.to_csv(
            output / "local_failure_mode_temporal_stability.csv", index=False
        )
        parent_state.to_parquet(
            output / "daily_parent_market_state.parquet", index=False
        )
        _calendar_with_event_blocks(
            parent_calendar,
            max_event_days=PARENT_MAX_EVENT_DAYS,
        ).to_parquet(
            output / "parent_adverse_calendar.parquet", index=False
        )
        parent_taxonomy.to_parquet(
            output / "parent_failure_block_taxonomy.parquet", index=False
        )
        parent_profiles.to_csv(
            output / "parent_failure_family_profiles.csv", index=False
        )
        parent_assignments.to_parquet(
            output / "parent_failure_mixture_assignments.parquet", index=False
        )
        parent_diagnostics.to_csv(
            output / "parent_failure_mixture_diagnostics.csv", index=False
        )
        parent_mixture_profiles.to_csv(
            output / "parent_failure_mixture_profiles.csv", index=False
        )
        parent_nonredundancy.to_csv(
            output / "parent_failure_mixture_nonredundancy.csv", index=False
        )
        parent_temporal_stability.to_csv(
            output / "parent_failure_mode_temporal_stability.csv", index=False
        )
        frozen_local_assignments.to_parquet(
            output / "local_frozen_failure_mode_assignments.parquet", index=False
        )
        frozen_local_diagnostics.to_csv(
            output / "local_frozen_failure_mode_diagnostics.csv", index=False
        )
        frozen_local_profiles.to_csv(
            output / "local_frozen_failure_mode_profiles.csv", index=False
        )
        frozen_local_semantic_assignments.to_parquet(
            output / "local_frozen_failure_mode_semantic_assignments.parquet",
            index=False,
        )
        (output / "local_frozen_failure_taxonomy_state.json").write_text(
            json.dumps(frozen_local_state, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        frozen_parent_assignments.to_parquet(
            output / "parent_frozen_failure_mode_assignments.parquet", index=False
        )
        frozen_parent_diagnostics.to_csv(
            output / "parent_frozen_failure_mode_diagnostics.csv", index=False
        )
        frozen_parent_profiles.to_csv(
            output / "parent_frozen_failure_mode_profiles.csv", index=False
        )
        frozen_parent_semantic_assignments.to_parquet(
            output / "parent_frozen_failure_mode_semantic_assignments.parquet",
            index=False,
        )
        (output / "parent_frozen_failure_taxonomy_state.json").write_text(
            json.dumps(frozen_parent_state, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        local_semantic_assignments.to_parquet(
            output / "local_failure_semantic_assignments.parquet", index=False
        )
        parent_semantic_assignments.to_parquet(
            output / "parent_failure_semantic_assignments.parquet", index=False
        )
        negative_day_modes.to_csv(
            output / "negative_pnl_day_failure_modes.csv", index=False
        )
        descriptive_negative_day_modes.to_csv(
            output / "negative_pnl_day_descriptive_failure_modes.csv", index=False
        )
        local_mode_composition.to_csv(
            output / "local_failure_mode_composition.csv", index=False
        )
        parent_mode_composition.to_csv(
            output / "parent_failure_mode_composition.csv", index=False
        )
    manifest = {
        **result.manifest,
        "source": source_manifest,
        "observable_state": state_manifest,
        "reused_local_taxonomy_root": (
            str(args.reuse_local_taxonomy_root).strip()
            if args.reuse_local_taxonomy_root is not None
            else ""
        ),
        "local_taxonomy_blocks": int(len(taxonomy_rows)),
        "local_taxonomy_profiles": int(len(profiles)),
        "matched_benign_controls": int(len(controls)),
        "mixture_assignment_rows": int(len(mixture_assignments)),
        "mixture_diagnostic_rows": int(len(mixture_diagnostics)),
        "mixture_profile_rows": int(len(mixture_profiles)),
        "mixture_nonredundancy_rows": int(len(mixture_nonredundancy)),
        "mixture_redundancy_warnings": int(
            mixture_nonredundancy.get(
                "calendar_redundancy_warning", pd.Series(dtype=bool)
            ).sum()
        ),
        "mixture_temporal_stability_rows": int(len(mixture_temporal_stability)),
        "mixture_temporal_stability_warnings": int(
            mixture_temporal_stability.get(
                "temporal_stability_warning", pd.Series(dtype=bool)
            ).sum()
        ),
        "parent_taxonomy_blocks": int(len(parent_taxonomy)),
        "parent_mixture_assignment_rows": int(len(parent_assignments)),
        "parent_mixture_diagnostic_rows": int(len(parent_diagnostics)),
        "parent_mixture_nonredundancy_rows": int(len(parent_nonredundancy)),
        "parent_mixture_redundancy_warnings": int(
            parent_nonredundancy.get(
                "calendar_redundancy_warning", pd.Series(dtype=bool)
            ).sum()
        ),
        "parent_temporal_stability_rows": int(len(parent_temporal_stability)),
        "parent_temporal_stability_warnings": int(
            parent_temporal_stability.get(
                "temporal_stability_warning", pd.Series(dtype=bool)
            ).sum()
        ),
        "prospective_taxonomy_reference_end": str(
            args.prospective_taxonomy_reference_end
        ),
        "frozen_local_mode_assignment_rows": int(len(frozen_local_assignments)),
        "frozen_local_mode_groups": int(len(frozen_local_diagnostics)),
        "frozen_parent_mode_assignment_rows": int(len(frozen_parent_assignments)),
        "frozen_parent_mode_groups": int(len(frozen_parent_diagnostics)),
        "negative_day_mode_assignment_contract": "frozen_reference_prototypes",
        "descriptive_negative_day_mode_rows": int(
            len(descriptive_negative_day_modes)
        ),
        "frozen_local_semantic_modes": int(
            frozen_local_profiles.get("semantic_label", pd.Series(dtype=str)).nunique()
        ),
        "frozen_parent_semantic_modes": int(
            frozen_parent_profiles.get("semantic_label", pd.Series(dtype=str)).nunique()
        ),
        "negative_day_mode_rows": int(len(negative_day_modes)),
        "negative_day_parent_mode_coverage": (
            float(negative_day_modes["parent_mode_assigned"].mean())
            if len(negative_day_modes)
            else np.nan
        ),
        "negative_day_local_mode_coverage": (
            float(negative_day_modes["all_active_local_modes_assigned"].mean())
            if len(negative_day_modes)
            else np.nan
        ),
        "local_mode_composition_rows": int(len(local_mode_composition)),
        "local_mode_composition_warnings": int(
            local_mode_composition.get(
                "composition_redundancy_warning", pd.Series(dtype=bool)
            ).sum()
        ),
        "parent_mode_composition_rows": int(len(parent_mode_composition)),
        "parent_mode_composition_warnings": int(
            parent_mode_composition.get(
                "composition_redundancy_warning", pd.Series(dtype=bool)
            ).sum()
        ),
        "failure_population": (
            "all_resolved_candidate_rows"
            if args.include_all_rows
            else "selected_for_monitor_top10_equivalent"
        ),
        "status": (
            "complete"
            if source_manifest["three_year_coverage_pass"]
            and (
                not len(negative_day_modes)
                or bool(negative_day_modes["parent_mode_assigned"].all())
            )
            else (
                "incomplete_negative_day_taxonomy_coverage"
                if source_manifest["three_year_coverage_pass"]
                else "complete_available_history_three_year_source_pending"
            )
        ),
        "outputs": {
            "daily_global": str(output / "daily_global_health.parquet"),
            "daily_local": str(output / "daily_side_archetype_health.parquet"),
            "local_episodes": str(output / "local_failure_episodes.parquet"),
            "parent_episodes": str(output / "parent_failure_episodes.parquet"),
            "negative_day_coverage": str(output / "negative_pnl_day_coverage.csv"),
            "negative_day_failure_modes": str(
                output / "negative_pnl_day_failure_modes.csv"
            ),
            "local_mode_composition": str(
                output / "local_failure_mode_composition.csv"
            ),
            "parent_mode_composition": str(
                output / "parent_failure_mode_composition.csv"
            ),
            "local_mode_temporal_stability": str(
                output / "local_failure_mode_temporal_stability.csv"
            ),
            "parent_mode_temporal_stability": str(
                output / "parent_failure_mode_temporal_stability.csv"
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(manifest, default=_json_default), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--required-years", type=float, default=3.0)
    parser.add_argument(
        "--provenance",
        choices=("oof_oos", "frozen_backcast_diagnostic", "mixed"),
        default="oof_oos",
    )
    parser.add_argument("--include-all-rows", action="store_true")
    parser.add_argument("--significance-z", type=float, default=1.96)
    parser.add_argument("--causal-min-days", type=int, default=20)
    parser.add_argument("--join-gap-days", type=int, default=0)
    parser.add_argument("--min-selected-rows", type=int, default=8)
    parser.add_argument("--min-parent-cells", type=int, default=7)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--min-cluster-episodes", type=int, default=3)
    parser.add_argument("--reuse-local-taxonomy-root", type=Path, default=None)
    parser.add_argument(
        "--prospective-taxonomy-reference-end",
        default="2025-01-01T00:00:00Z",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
