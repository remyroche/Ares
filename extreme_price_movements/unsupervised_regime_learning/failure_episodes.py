"""Failure-episode construction for model-error regime discovery.

This module is deliberately descriptive. Realized outcomes may define and
describe historical failure episodes, but are kept in an explicit ex-post
column block that must never be passed to an inference transform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.residual_state_discovery import (
    ReliabilityEventConfig,
    ReliabilityEventResult,
    discover_reliability_events,
)

EX_POST_PREFIX = "expost__"
OBSERVABLE_PREFIX = "state__"


@dataclass(frozen=True)
class FailureEpisodeConfig:
    """Configuration for local and parent failure episodes."""

    timestamp_col: str = "__ts__"
    symbol_col: str = "__symbol__"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    probability_col: str = "hit_probability"
    hit_col: str = "clean_exec"
    ev_col: str = "ev_after_1pct"
    exec_margin_col: str = "exec_margin"
    dirty_positive_col: str = "dirty_positive"
    base_score_col: str = "base_score"
    meta_score_col: str = "score_meta_base_soft_label"
    rank_col: str = "historical_rank"
    join_gap_days: int = 0
    min_selected_rows: int = 8
    min_parent_cells: int = 7
    significance_z: float = 1.96
    causal_min_days: int = 20
    bootstrap_draws: int = 500
    pre_window_days: int = 3
    recovery_horizon_days: int = 14
    confidence_band_edges: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


@dataclass(frozen=True)
class FailureEpisodeResult:
    daily_global: pd.DataFrame
    local: ReliabilityEventResult
    parent_episodes: pd.DataFrame
    parent_membership: pd.DataFrame
    coverage: pd.DataFrame
    manifest: dict[str, Any]


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _binary_log_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(pair) < 4 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def _confidence_band_names(edges: Sequence[float]) -> tuple[str, ...]:
    """Return stable, human-readable confidence band names for configured edges."""

    intervals = len(edges) - 1
    if intervals == 3:
        return ("low", "medium", "high")
    return tuple(f"band_{index}" for index in range(intervals))


def _prediction_sources(
    work: pd.DataFrame,
    cfg: FailureEpisodeConfig,
) -> dict[str, pd.Series]:
    """Expose probability-like model outputs that are present in this source.

    The primary prediction is required by the episode contract. Base and meta
    residual vectors are optional because older historical ledgers do not
    always retain both scores.
    """

    candidates = {
        "primary": cfg.probability_col,
        "base": cfg.base_score_col,
        "meta": cfg.meta_score_col,
    }
    sources: dict[str, pd.Series] = {}
    for name, column in candidates.items():
        if column not in work:
            continue
        values = _numeric(work, column)
        # Scores outside the probability domain are not calibrated residuals.
        values = values.where(values.between(0.0, 1.0))
        if values.notna().any():
            sources[name] = values
    return sources


def _confidence_band_metrics(
    score: pd.Series,
    hit: pd.Series,
    *,
    source: str,
    cfg: FailureEpisodeConfig,
) -> dict[str, float]:
    """Summarize false-positive/negative composition by prediction confidence."""

    edges = tuple(float(edge) for edge in cfg.confidence_band_edges)
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("confidence_band_edges must start at 0.0 and end at 1.0")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("confidence_band_edges must be strictly increasing")

    valid = pd.DataFrame({"score": score, "hit": hit}).dropna()
    valid = valid.loc[valid["score"].between(0.0, 1.0)]
    false_positive = valid["score"].ge(0.5) & valid["hit"].lt(0.5)
    false_negative = valid["score"].lt(0.5) & valid["hit"].ge(0.5)
    total_false_positive = int(false_positive.sum())
    total_false_negative = int(false_negative.sum())
    output: dict[str, float] = {}
    for index, band in enumerate(_confidence_band_names(edges)):
        lower, upper = edges[index], edges[index + 1]
        in_band = valid["score"].ge(lower) & (
            valid["score"].le(upper)
            if index == len(edges) - 2
            else valid["score"].lt(upper)
        )
        band_rows = int(in_band.sum())
        fp_count = int((false_positive & in_band).sum())
        fn_count = int((false_negative & in_band).sum())
        prefix = f"{EX_POST_PREFIX}{source}_confidence_{band}"
        output[f"{prefix}_row_share"] = (
            float(band_rows / len(valid)) if len(valid) else np.nan
        )
        output[f"{prefix}_false_positive_rate"] = (
            float(fp_count / band_rows) if band_rows else np.nan
        )
        output[f"{prefix}_false_negative_rate"] = (
            float(fn_count / band_rows) if band_rows else np.nan
        )
        output[f"{prefix}_false_positive_composition"] = (
            float(fp_count / total_false_positive) if total_false_positive else np.nan
        )
        output[f"{prefix}_false_negative_composition"] = (
            float(fn_count / total_false_negative) if total_false_negative else np.nan
        )
    return output


def _daily_global_health(
    frame: pd.DataFrame, cfg: FailureEpisodeConfig
) -> pd.DataFrame:
    work = frame.copy(deep=False)
    work[cfg.timestamp_col] = pd.to_datetime(
        work[cfg.timestamp_col], utc=True, errors="coerce"
    )
    work = work.loc[work[cfg.timestamp_col].notna()].copy()
    work["day"] = work[cfg.timestamp_col].dt.floor("D")
    hit = _numeric(work, cfg.hit_col)
    prediction_sources = _prediction_sources(work, cfg)
    probability = prediction_sources["primary"].clip(1e-6, 1.0 - 1e-6)
    ev = _numeric(work, cfg.ev_col)
    work["_residual"] = hit - probability
    work["_abs_residual"] = work["_residual"].abs()
    work["_brier"] = work["_residual"].pow(2)
    work["_log_loss"] = _binary_log_loss(
        hit.to_numpy(np.float64), probability.to_numpy(np.float64)
    )
    work["_negative_ev"] = ev.lt(0.0)
    dirty_positive = _numeric(work, cfg.dirty_positive_col)
    work["_dirty_positive"] = dirty_positive.where(dirty_positive.notna())
    work["_false_positive"] = probability.ge(0.5) & hit.lt(0.5)
    work["_false_negative"] = probability.lt(0.5) & hit.ge(0.5)
    work["_base_meta_disagreement"] = (
        _numeric(work, cfg.meta_score_col) - _numeric(work, cfg.base_score_col)
    ).abs()
    for source, score in prediction_sources.items():
        work[f"_{source}_signed_residual"] = hit - score
        work[f"_{source}_absolute_residual"] = work[f"_{source}_signed_residual"].abs()

    rows: list[dict[str, Any]] = []
    for day, group in work.groupby("day", observed=True, sort=True):
        weights = np.ones(len(group), dtype=np.float64)
        ev_values = _numeric(group, cfg.ev_col).to_numpy(np.float64)
        side_counts = group[cfg.side_col].astype(str).value_counts(normalize=True)
        archetype_counts = (
            group[cfg.archetype_col].astype(str).value_counts(normalize=True)
        )
        row = {
            "day": day,
            "selected_rows": int(len(group)),
            "distinct_assets": int(group[cfg.symbol_col].nunique()),
            "net_ev": float(np.nansum(ev_values)),
            "mean_ev": float(np.nanmean(ev_values)),
            "negative_pnl_day": bool(np.nansum(ev_values) < 0.0),
            f"{EX_POST_PREFIX}signed_residual": float(group["_residual"].mean()),
            f"{EX_POST_PREFIX}absolute_residual": float(group["_abs_residual"].mean()),
            f"{EX_POST_PREFIX}brier": float(group["_brier"].mean()),
            f"{EX_POST_PREFIX}log_loss": float(group["_log_loss"].mean()),
            f"{EX_POST_PREFIX}false_positive_rate": float(
                group["_false_positive"].mean()
            ),
            f"{EX_POST_PREFIX}false_negative_rate": float(
                group["_false_negative"].mean()
            ),
            f"{EX_POST_PREFIX}negative_trade_rate": float(group["_negative_ev"].mean()),
            f"{EX_POST_PREFIX}dirty_positive_rate": float(
                group["_dirty_positive"].mean()
            ),
            f"{EX_POST_PREFIX}mean_exec_margin": float(
                _numeric(group, cfg.exec_margin_col).mean()
            ),
            f"{EX_POST_PREFIX}base_meta_disagreement": float(
                group["_base_meta_disagreement"].mean()
            ),
            f"{EX_POST_PREFIX}ranking_spearman": _safe_corr(
                _numeric(group, cfg.rank_col), _numeric(group, cfg.ev_col)
            ),
            f"{EX_POST_PREFIX}side_concentration": float(side_counts.max())
            if len(side_counts)
            else np.nan,
            f"{EX_POST_PREFIX}archetype_concentration": float(archetype_counts.max())
            if len(archetype_counts)
            else np.nan,
        }
        for source in prediction_sources:
            residual = _numeric(group, f"_{source}_signed_residual")
            absolute_residual = _numeric(group, f"_{source}_absolute_residual")
            row[f"{EX_POST_PREFIX}{source}_signed_residual"] = float(residual.mean())
            row[f"{EX_POST_PREFIX}{source}_absolute_residual"] = float(
                absolute_residual.mean()
            )
            row[f"{EX_POST_PREFIX}{source}_residual_std"] = float(residual.std(ddof=0))
            row[f"{EX_POST_PREFIX}{source}_overconfidence_rate"] = float(
                residual.lt(0.0).mean()
            )
            row[f"{EX_POST_PREFIX}{source}_underconfidence_rate"] = float(
                residual.gt(0.0).mean()
            )
            row.update(
                _confidence_band_metrics(
                    _numeric(group, cfg.probability_col)
                    if source == "primary"
                    else _numeric(
                        group,
                        cfg.base_score_col if source == "base" else cfg.meta_score_col,
                    ),
                    _numeric(group, cfg.hit_col),
                    source=source,
                    cfg=cfg,
                )
            )
        for source, destination in (
            ("full_path_bad_mae_1r", "bad_mae_rate"),
            ("first_touch_bad_mae_1r", "first_touch_bad_mae_rate"),
            ("timeout", "timeout_rate"),
            ("stop_or_adverse", "stop_or_adverse_rate"),
        ):
            row[f"{EX_POST_PREFIX}{destination}"] = float(
                _numeric(group, source).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("day", kind="stable").reset_index(drop=True)


def _parent_active_days(
    daily_global: pd.DataFrame,
    local_membership: pd.DataFrame,
    cfg: FailureEpisodeConfig,
) -> pd.DataFrame:
    local = local_membership.copy()
    if local.empty:
        local_daily = pd.DataFrame(
            columns=["day", "active_local_cells", "local_event_ids"]
        )
    else:
        local["day"] = pd.to_datetime(local["day"], utc=True).dt.floor("D")
        local_daily = (
            local.groupby("day", observed=True, sort=True)
            .agg(
                active_local_cells=("event_id", "size"),
                local_event_ids=(
                    "event_id",
                    lambda values: "|".join(sorted(set(map(str, values)))),
                ),
            )
            .reset_index()
        )
    merged = daily_global.merge(local_daily, on="day", how="left")
    merged["active_local_cells"] = (
        pd.to_numeric(merged["active_local_cells"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    merged["local_event_ids"] = merged["local_event_ids"].fillna("")
    merged["parent_active"] = merged["negative_pnl_day"] | merged[
        "active_local_cells"
    ].ge(int(cfg.min_parent_cells))
    return merged


def _assign_parent_ids(days: pd.DataFrame, cfg: FailureEpisodeConfig) -> pd.DataFrame:
    active = days.loc[days["parent_active"]].sort_values("day", kind="stable").copy()
    if active.empty:
        active["parent_episode_id"] = pd.Series(dtype=str)
        return active
    identifiers: list[str] = []
    current = 0
    previous: pd.Timestamp | None = None
    for day in pd.to_datetime(active["day"], utc=True):
        if previous is None or (day - previous).days > int(cfg.join_gap_days) + 1:
            current += 1
        identifiers.append(f"PFE-{current:04d}")
        previous = day
    active["parent_episode_id"] = identifiers
    return active


def _phase_numeric_fields(
    source: pd.DataFrame,
    *,
    phase: str,
    weights: pd.Series | None = None,
) -> dict[str, Any]:
    """Encode a phase snapshot without weakening the ex-post namespace."""

    if source.empty:
        return {}
    result: dict[str, Any] = {}
    for name in source:
        if not name.startswith(EX_POST_PREFIX):
            continue
        values = pd.to_numeric(source[name], errors="coerce")
        if not values.notna().any():
            continue
        output = f"{EX_POST_PREFIX}{phase}_{name[len(EX_POST_PREFIX) :]}"
        if weights is None:
            result[output] = float(values.mean())
        else:
            valid = values.notna() & weights.notna()
            result[output] = (
                float(np.average(values.loc[valid], weights=weights.loc[valid]))
                if valid.any()
                else np.nan
            )
    return result


def _episode_phase_summary(
    daily_global: pd.DataFrame,
    episode_days: pd.DataFrame,
    cfg: FailureEpisodeConfig,
) -> dict[str, Any]:
    """Describe pre-event, onset, worst point, and recovery from realized health."""

    start = pd.Timestamp(episode_days["day"].min())
    end = pd.Timestamp(episode_days["day"].max())
    all_days = daily_global.sort_values("day", kind="stable")
    pre_start = start - pd.Timedelta(days=max(int(cfg.pre_window_days), 0))
    pre = all_days.loc[(all_days["day"] < start) & (all_days["day"] >= pre_start)]
    transition = episode_days.iloc[[0]]
    severity = pd.to_numeric(episode_days["mean_ev"], errors="coerce")
    if severity.notna().any():
        peak = episode_days.loc[[severity.idxmin()]]
    else:
        peak = transition
    recovery_candidates = all_days.loc[
        (all_days["day"] > end)
        & (all_days["day"] <= end + pd.Timedelta(days=int(cfg.recovery_horizon_days)))
        & pd.to_numeric(all_days["mean_ev"], errors="coerce").ge(0.0)
    ]
    recovery = (
        recovery_candidates.iloc[[0]]
        if not recovery_candidates.empty
        else recovery_candidates
    )
    recovery_day = recovery["day"].iloc[0] if not recovery.empty else pd.NaT
    output: dict[str, Any] = {
        f"{EX_POST_PREFIX}duration_days": int((end - start).days + 1),
        f"{EX_POST_PREFIX}pre_start": pre["day"].min() if not pre.empty else pd.NaT,
        f"{EX_POST_PREFIX}pre_end": pre["day"].max() if not pre.empty else pd.NaT,
        f"{EX_POST_PREFIX}transition_day": transition["day"].iloc[0],
        f"{EX_POST_PREFIX}peak_day": peak["day"].iloc[0],
        f"{EX_POST_PREFIX}peak_offset_days": int(
            (pd.Timestamp(peak["day"].iloc[0]) - start).days
        ),
        f"{EX_POST_PREFIX}recovery_day": recovery_day,
        f"{EX_POST_PREFIX}recovered_within_horizon": bool(not recovery.empty),
        f"{EX_POST_PREFIX}recovery_lag_days": (
            int((pd.Timestamp(recovery_day) - end).days)
            if not recovery.empty
            else np.nan
        ),
    }
    output.update(_phase_numeric_fields(pre, phase="pre"))
    output.update(_phase_numeric_fields(transition, phase="transition"))
    output.update(_phase_numeric_fields(peak, phase="peak"))
    output.update(_phase_numeric_fields(recovery, phase="recovery"))
    return output


def _summarize_parent_episodes(
    membership: pd.DataFrame,
    local_cells: pd.DataFrame,
    local_membership: pd.DataFrame,
    daily_global: pd.DataFrame,
    cfg: FailureEpisodeConfig,
) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame()
    local = local_membership.copy()
    if not local.empty:
        local["day"] = pd.to_datetime(local["day"], utc=True).dt.floor("D")
    cells = local_cells.copy()
    if not cells.empty:
        cells["day"] = pd.to_datetime(cells["day"], utc=True).dt.floor("D")
    rows: list[dict[str, Any]] = []
    ex_post = [name for name in membership if name.startswith(EX_POST_PREFIX)]
    for episode_id, group in membership.groupby("parent_episode_id", sort=True):
        start = pd.Timestamp(group["day"].min())
        end = pd.Timestamp(group["day"].max())
        local_slice = (
            local.loc[local["day"].between(start, end)] if not local.empty else local
        )
        cell_slice = (
            cells.loc[cells["day"].between(start, end)] if not cells.empty else cells
        )
        row: dict[str, Any] = {
            "parent_episode_id": episode_id,
            "event_start": start,
            "event_end": end,
            "duration_days": int((end - start).days + 1),
            "negative_pnl_days": int(group["negative_pnl_day"].sum()),
            "selected_rows": int(group["selected_rows"].sum()),
            "distinct_assets_peak": int(group["distinct_assets"].max()),
            "net_ev": float(group["net_ev"].sum()),
            "mean_ev": float(
                np.average(
                    group["mean_ev"], weights=group["selected_rows"].clip(lower=1)
                )
            ),
            "worst_day_ev": float(group["mean_ev"].min()),
            "active_local_cells": int(group["active_local_cells"].sum()),
            "affected_sides": "|".join(
                sorted(
                    set(cell_slice.get("side_name", pd.Series(dtype=str)).astype(str))
                )
            ),
            "affected_archetypes": "|".join(
                sorted(
                    set(
                        cell_slice.get(
                            "archetype_policy_key", pd.Series(dtype=str)
                        ).astype(str)
                    )
                )
            ),
            "local_event_ids": "|".join(
                sorted(
                    set(local_slice.get("event_id", pd.Series(dtype=str)).astype(str))
                )
            ),
        }
        for name in ex_post:
            values = pd.to_numeric(group[name], errors="coerce")
            row[name] = float(
                np.average(
                    values.fillna(0.0), weights=group["selected_rows"].clip(lower=1)
                )
            )
            row[f"{name}__peak_abs"] = float(values.abs().max())
        row.update(_episode_phase_summary(daily_global, group, cfg))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("event_start", kind="stable")
        .reset_index(drop=True)
    )


def negative_day_coverage(
    daily_global: pd.DataFrame,
    parent_membership: pd.DataFrame,
) -> pd.DataFrame:
    negative = daily_global.loc[
        daily_global["negative_pnl_day"], ["day", "net_ev"]
    ].copy()
    covered = set(pd.to_datetime(parent_membership.get("day"), utc=True))
    negative["covered_by_parent_episode"] = pd.to_datetime(
        negative["day"], utc=True
    ).isin(covered)
    return negative.sort_values("day", kind="stable").reset_index(drop=True)


def validate_inference_feature_columns(columns: Iterable[str]) -> None:
    forbidden_prefixes = (
        EX_POST_PREFIX,
        "target__",
        "label__",
        "outcome__",
        "future__",
        "realized__",
        "availability__",
    )
    forbidden_exact = {
        "adverse_event",
        "event_block",
        "failure_mode",
        "failure_mode_available_day",
        "negative_pnl_day",
        "mean_ev_after_1pct",
        "net_ev",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "stop_or_adverse",
        "selected_rows",
        "signed_surprise",
        "persistence_strength",
        "large_event_strength",
    }
    leaked = sorted(
        name
        for name in map(str, columns)
        if name in forbidden_exact or name.startswith(forbidden_prefixes)
    )
    if leaked:
        raise ValueError(
            "Ex-post failure features or outcome columns cannot enter an inference transform: "
            f"{leaked[:12]}"
        )


def build_failure_episodes(
    selected_rows: pd.DataFrame,
    config: FailureEpisodeConfig = FailureEpisodeConfig(),
) -> FailureEpisodeResult:
    """Build local side/archetype events and parent market failure episodes."""

    required = {
        config.timestamp_col,
        config.symbol_col,
        config.side_col,
        config.archetype_col,
        config.probability_col,
        config.hit_col,
        config.ev_col,
    }
    missing = required.difference(selected_rows.columns)
    if missing:
        raise KeyError(f"Failure episode source is missing columns: {sorted(missing)}")
    local_cfg = ReliabilityEventConfig(
        timestamp_col=config.timestamp_col,
        symbol_col=config.symbol_col,
        side_col=config.side_col,
        archetype_col=config.archetype_col,
        probability_col=config.probability_col,
        hit_col=config.hit_col,
        ev_col=config.ev_col,
        significance_z=config.significance_z,
        causal_min_days=config.causal_min_days,
        join_gap_days=config.join_gap_days,
        min_event_selected_rows=config.min_selected_rows,
        bootstrap_draws=config.bootstrap_draws,
    )
    local = discover_reliability_events(selected_rows, local_cfg)
    daily = _daily_global_health(selected_rows, config)
    parent_days = _parent_active_days(daily, local.event_membership, config)
    membership = _assign_parent_ids(parent_days, config)
    episodes = _summarize_parent_episodes(
        membership,
        local.daily_cells,
        local.event_membership,
        daily,
        config,
    )
    coverage = negative_day_coverage(daily, membership)
    uncovered = (
        int((~coverage["covered_by_parent_episode"]).sum()) if len(coverage) else 0
    )
    manifest = {
        "schema": "failure_episode_taxonomy_source_v1",
        "config": asdict(config),
        "source_rows": int(len(selected_rows)),
        "source_start": pd.to_datetime(
            selected_rows[config.timestamp_col], utc=True
        ).min(),
        "source_end": pd.to_datetime(
            selected_rows[config.timestamp_col], utc=True
        ).max(),
        "local_events": int(len(local.events)),
        "parent_episodes": int(len(episodes)),
        "negative_pnl_days": int(len(coverage)),
        "uncovered_negative_pnl_days": uncovered,
        "negative_day_coverage_pass": bool(uncovered == 0),
        "ex_post_feature_prefix": EX_POST_PREFIX,
        "observable_feature_prefix": OBSERVABLE_PREFIX,
        "leakage_contract": (
            "expost__ columns and episode memberships are descriptive outcomes only; "
            "prospective detectors may consume only causal state__ features and already-resolved history"
        ),
    }
    return FailureEpisodeResult(daily, local, episodes, membership, coverage, manifest)


def episode_feature_columns(
    episodes: pd.DataFrame,
    *,
    include_ex_post: bool,
) -> Sequence[str]:
    """Return a feature block with an explicit leakage boundary."""

    columns = [name for name in episodes if name.startswith(OBSERVABLE_PREFIX)]
    if include_ex_post:
        columns.extend(name for name in episodes if name.startswith(EX_POST_PREFIX))
    return tuple(dict.fromkeys(columns))
