"""Causal health bins and failure episodes for a globally ranked model.

The production auction ranks one pooled stream across timestamps and sides.
Historical failure discovery must nevertheless avoid a retrospective full-panel
quantile.  This module uses a trailing, decision-time-only score distribution
to approximate the frozen global admission frontier, then waits for exact
outcomes before updating residual-severity references.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FailureHealthConfig:
    timestamp_col: str = "execution_decision_utc"
    label_end_col: str = "execution_label_end_utc"
    score_col: str = "causal_recent_side_isotonic_ev"
    score_oof_col: str = "causal_recent_side_isotonic_ev__is_oof"
    gross_col: str = "execution_gross_ev_12h"
    net_col: str = "execution_net_ev_12h"
    candidate_id_col: str = "candidate_id"
    side_col: str = "side_name"
    symbol_col: str = "__symbol__"
    evaluation_origin_col: str = "evaluation_origin"
    admission_lookback_days: int = 21
    admission_quantile: float = 0.90
    minimum_cutoff_rows: int = 4_000
    health_bin_hours: int = 6
    minimum_admitted_rows: int = 20
    residual_lookback_days: int = 21
    minimum_resolved_bins: int = 20
    join_gap_hours: int = 12


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _stable_id(prefix: str, *values: object) -> str:
    payload = "|".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _causal_score_cutoffs(
    frame: pd.DataFrame,
    *,
    config: FailureHealthConfig,
) -> pd.DataFrame:
    """Return one trailing-score cutoff per decision timestamp and origin."""

    timestamp = config.timestamp_col
    score = config.score_col
    records: list[dict[str, Any]] = []
    for origin, local in frame.groupby(
        config.evaluation_origin_col, sort=True, observed=True
    ):
        grouped = (
            local.loc[:, [timestamp, score]]
            .dropna()
            .groupby(timestamp, sort=True, observed=True)[score]
            .apply(lambda values: values.to_numpy(np.float64))
        )
        lookback = pd.Timedelta(days=int(config.admission_lookback_days))
        history: deque[tuple[pd.Timestamp, np.ndarray]] = deque()
        rows = 0
        for decision, current_scores in grouped.items():
            decision = pd.Timestamp(decision)
            while history and history[0][0] < decision - lookback:
                _, removed = history.popleft()
                rows -= int(len(removed))
            if rows >= int(config.minimum_cutoff_rows):
                values = np.concatenate([values for _, values in history])
                cutoff = float(
                    np.quantile(values, float(config.admission_quantile))
                )
                reference_start = history[0][0]
                reference_end = history[-1][0]
            else:
                cutoff = np.nan
                reference_start = pd.NaT
                reference_end = pd.NaT
            records.append(
                {
                    timestamp: decision,
                    config.evaluation_origin_col: str(origin),
                    "cutoff_reference_start_utc": reference_start,
                    "cutoff_reference_end_utc": reference_end,
                    "cutoff_reference_rows": int(rows),
                    "admission_cutoff": cutoff,
                }
            )
            history.append((decision, current_scores))
            rows += int(len(current_scores))
    return pd.DataFrame.from_records(records)


def build_causal_decision_health(
    frame: pd.DataFrame,
    config: FailureHealthConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal six-hour model-health bins and row membership.

    Returns ``(decision_health, membership)``.  Realized fields remain explicit
    ex-post diagnostics and are never returned as inference features.
    """

    cfg = config or FailureHealthConfig()
    required = {
        cfg.timestamp_col,
        cfg.label_end_col,
        cfg.score_col,
        cfg.gross_col,
        cfg.net_col,
        cfg.candidate_id_col,
        cfg.side_col,
        cfg.symbol_col,
        cfg.score_oof_col,
        cfg.evaluation_origin_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError("failure-health source missing: " + ", ".join(missing))
    work = frame.copy()
    work[cfg.timestamp_col] = _utc(work[cfg.timestamp_col])
    work[cfg.label_end_col] = _utc(work[cfg.label_end_col])
    work[cfg.score_col] = pd.to_numeric(work[cfg.score_col], errors="coerce")
    work[cfg.gross_col] = pd.to_numeric(work[cfg.gross_col], errors="coerce")
    work[cfg.net_col] = pd.to_numeric(work[cfg.net_col], errors="coerce")
    work[cfg.evaluation_origin_col] = work[cfg.evaluation_origin_col].astype(str)
    work = work.loc[work[cfg.score_oof_col].fillna(False).astype(bool)].copy()
    work = work.loc[
        work[cfg.timestamp_col].notna()
        & work[cfg.label_end_col].notna()
        & work[cfg.score_col].notna()
        & work[cfg.gross_col].notna()
        & work[cfg.net_col].notna()
    ].copy()
    work = work.sort_values(
        [cfg.timestamp_col, cfg.candidate_id_col], kind="stable"
    ).reset_index(drop=True)
    if work.empty:
        raise ValueError("no strict OOF rows remain for failure-health construction")
    cutoffs = _causal_score_cutoffs(work, config=cfg)
    work = work.merge(
        cutoffs,
        on=[cfg.timestamp_col, cfg.evaluation_origin_col],
        how="left",
        validate="many_to_one",
    )
    work["admitted"] = (
        work["admission_cutoff"].notna()
        & work[cfg.score_col].ge(work["admission_cutoff"])
    )
    work["decision_bin_start_utc"] = work[cfg.timestamp_col].dt.floor(
        f"{int(cfg.health_bin_hours)}h"
    )
    work["expost__gross_ev"] = work[cfg.gross_col]
    work["expost__net_ev"] = work[cfg.net_col]
    work["expost__cost"] = work[cfg.gross_col] - work[cfg.net_col]

    health_rows: list[dict[str, Any]] = []
    group_columns = ["decision_bin_start_utc", cfg.evaluation_origin_col]
    for (bin_start, origin), group in work.groupby(
        group_columns, sort=True, observed=True
    ):
        admitted = group.loc[group["admitted"]].copy()
        predicted = admitted[cfg.score_col]
        realized_gross = admitted[cfg.gross_col]
        realized_net = admitted[cfg.net_col]
        residual = realized_net - predicted
        cutoff_rows = group["cutoff_reference_rows"].dropna()
        row = {
            "decision_bin_start_utc": bin_start,
            "decision_bin_end_utc": bin_start
            + pd.Timedelta(hours=int(cfg.health_bin_hours)),
            "bin_available_utc": admitted[cfg.label_end_col].max()
            if len(admitted)
            else group[cfg.label_end_col].max(),
            "score_spec_id": cfg.score_col,
            "evaluation_origin": str(origin),
            "cutoff_reference_start_utc": group[
                "cutoff_reference_start_utc"
            ].min(),
            "cutoff_reference_end_utc": group["cutoff_reference_end_utc"].max(),
            "cutoff_reference_rows": int(cutoff_rows.min())
            if len(cutoff_rows)
            else 0,
            "admission_cutoff": float(group["admission_cutoff"].mean()),
            "candidate_rows": int(len(group)),
            "score_valid_rows": int(group[cfg.score_col].notna().sum()),
            "admitted_rows": int(len(admitted)),
            "admission_rate": float(len(admitted) / len(group)),
            "predicted_net_ev_sum": float(predicted.sum()),
            "predicted_net_ev_mean": float(predicted.mean()),
            "realized_gross_ev_sum": float(realized_gross.sum()),
            "realized_cost_sum": float((realized_gross - realized_net).sum()),
            "realized_net_ev_sum": float(realized_net.sum()),
            "realized_net_ev_mean": float(realized_net.mean()),
            "economic_residual_sum": float(residual.sum()),
            "economic_residual_mean": float(residual.mean()),
            "positive_net_rate": float(realized_net.gt(0.0).mean()),
            "negative_net_rate": float(realized_net.lt(0.0).mean()),
            "false_positive_rate": float(
                (predicted.gt(0.0) & realized_net.lt(0.0)).mean()
            ),
            "support_pass": bool(len(admitted) >= int(cfg.minimum_admitted_rows)),
            "negative_economics": bool(
                len(admitted) and float(realized_net.mean()) < 0.0
            ),
            "outcome_complete": bool(
                len(admitted)
                and admitted[cfg.label_end_col].notna().all()
                and admitted[cfg.gross_col].notna().all()
                and admitted[cfg.net_col].notna().all()
            ),
            "provenance_status": "strict_oof_causal_shadow_admission",
        }
        health_rows.append(row)
    health = pd.DataFrame.from_records(health_rows).sort_values(
        ["decision_bin_start_utc", "evaluation_origin"], kind="stable"
    )
    for name in (
        "prior_residual_q05",
        "prior_residual_q10",
        "prior_residual_q20",
        "prior_residual_mean",
        "prior_residual_std",
        "residual_z",
    ):
        health[name] = np.nan
    health["severe_residual"] = False
    health["severity_tier"] = "not_failure"
    health["model_failure_bin"] = False
    lookback = pd.Timedelta(days=int(cfg.residual_lookback_days))
    for index, row in health.iterrows():
        earlier = health.loc[
            health["bin_available_utc"].lt(row["decision_bin_start_utc"])
            & health["evaluation_origin"].astype(str).eq(
                str(row["evaluation_origin"])
            )
            & health["decision_bin_start_utc"].ge(
                row["decision_bin_start_utc"] - lookback
            )
            & health["support_pass"]
            & health["outcome_complete"]
            & health["economic_residual_mean"].notna()
        ]
        if len(earlier) < int(cfg.minimum_resolved_bins):
            continue
        values = earlier["economic_residual_mean"].to_numpy(np.float64)
        q05, q10, q20 = np.quantile(values, [0.05, 0.10, 0.20])
        mean, std = float(np.mean(values)), float(np.std(values))
        residual = float(row["economic_residual_mean"])
        health.loc[index, "prior_residual_q05"] = q05
        health.loc[index, "prior_residual_q10"] = q10
        health.loc[index, "prior_residual_q20"] = q20
        health.loc[index, "prior_residual_mean"] = mean
        health.loc[index, "prior_residual_std"] = std
        health.loc[index, "residual_z"] = (
            (residual - mean) / std if std > 1e-12 else np.nan
        )
        support = bool(row["support_pass"] and row["outcome_complete"])
        negative = bool(row["negative_economics"])
        if support and negative and residual <= q05:
            tier = "catastrophic"
        elif support and negative and residual <= q10:
            tier = "primary_failure"
        elif support and negative and residual <= q20:
            tier = "warning"
        else:
            tier = "not_failure"
        health.loc[index, "severity_tier"] = tier
        health.loc[index, "severe_residual"] = tier in {
            "catastrophic",
            "primary_failure",
            "warning",
        }
        health.loc[index, "model_failure_bin"] = tier in {
            "catastrophic",
            "primary_failure",
        }
    membership_columns = [
        cfg.candidate_id_col,
        cfg.timestamp_col,
        cfg.label_end_col,
        cfg.symbol_col,
        cfg.side_col,
        cfg.evaluation_origin_col,
        "decision_bin_start_utc",
        cfg.score_col,
        "admission_cutoff",
        "admitted",
        "expost__gross_ev",
        "expost__cost",
        "expost__net_ev",
    ]
    membership = work.loc[:, membership_columns].rename(
        columns={cfg.score_col: "mapped_score"}
    )
    return health.reset_index(drop=True), membership.reset_index(drop=True)


def group_failure_bins_into_episodes(
    health: pd.DataFrame,
    membership: pd.DataFrame,
    config: FailureHealthConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge causally defined primary bins without crossing model generations."""

    cfg = config or FailureHealthConfig()
    failed = health.loc[health["model_failure_bin"]].copy()
    failed = failed.sort_values(
        ["evaluation_origin", "decision_bin_start_utc"], kind="stable"
    )
    if failed.empty:
        empty = pd.DataFrame(
            columns=[
                "episode_id",
                "evaluation_origin",
                "episode_onset_decision_utc",
                "episode_onset_available_utc",
                "episode_end_decision_utc",
                "episode_end_available_utc",
                "trigger_bin_count",
            ]
        )
        return empty, membership.iloc[0:0].assign(episode_id=pd.Series(dtype=str))
    failed["episode_number"] = 0
    for origin, positions in failed.groupby(
        "evaluation_origin", observed=True, sort=False
    ).indices.items():
        indices = np.asarray(positions, dtype=np.int64)
        episode = 0
        previous: pd.Timestamp | None = None
        for index in indices:
            current = pd.Timestamp(failed.iloc[index]["decision_bin_start_utc"])
            if (
                previous is not None
                and current - previous
                > pd.Timedelta(hours=int(cfg.join_gap_hours))
            ):
                episode += 1
            failed.iloc[
                index, failed.columns.get_loc("episode_number")
            ] = episode
            previous = current
    episodes: list[dict[str, Any]] = []
    membership_parts: list[pd.DataFrame] = []
    for (origin, number), bins in failed.groupby(
        ["evaluation_origin", "episode_number"], sort=True, observed=True
    ):
        onset = bins["decision_bin_start_utc"].min()
        end = bins["decision_bin_start_utc"].max()
        episode_id = _stable_id("failure", origin, onset, end, number)
        relevant = membership.loc[
            membership["evaluation_origin"].astype(str).eq(str(origin))
            & membership["decision_bin_start_utc"].isin(
                bins["decision_bin_start_utc"]
            )
        ].copy()
        relevant["episode_id"] = episode_id
        membership_parts.append(relevant)
        admitted = relevant.loc[relevant["admitted"]]
        peak_index = bins["economic_residual_mean"].idxmin()
        peak = health.loc[peak_index]
        tiers = bins["severity_tier"].astype(str)
        severity = (
            "catastrophic" if tiers.eq("catastrophic").any() else "primary_failure"
        )
        sides = sorted(admitted[cfg.side_col].astype(str).unique().tolist())
        assets = sorted(admitted[cfg.symbol_col].astype(str).unique().tolist())
        episodes.append(
            {
                "episode_id": episode_id,
                "score_spec_id": cfg.score_col,
                "evaluation_origin": str(origin),
                "episode_onset_decision_utc": onset,
                "episode_onset_available_utc": bins.loc[
                    bins["decision_bin_start_utc"].eq(onset), "bin_available_utc"
                ].max(),
                "episode_end_decision_utc": end,
                "episode_end_available_utc": bins["bin_available_utc"].max(),
                "peak_decision_utc": peak["decision_bin_start_utc"],
                "peak_available_utc": peak["bin_available_utc"],
                "trigger_bin_count": int(len(bins)),
                "duration_hours": float(
                    (end - onset) / pd.Timedelta(hours=1)
                    + int(cfg.health_bin_hours)
                ),
                "admitted_rows": int(len(admitted)),
                "predicted_net_ev_sum": float(admitted["mapped_score"].sum()),
                "realized_gross_ev_sum": float(admitted["expost__gross_ev"].sum()),
                "realized_cost_sum": float(admitted["expost__cost"].sum()),
                "realized_net_ev_sum": float(admitted["expost__net_ev"].sum()),
                "economic_residual_sum": float(
                    (admitted["expost__net_ev"] - admitted["mapped_score"]).sum()
                ),
                "worst_bin_realized_net_ev_mean": float(
                    bins["realized_net_ev_mean"].min()
                ),
                "worst_bin_residual_mean": float(
                    bins["economic_residual_mean"].min()
                ),
                "severity_tier": severity,
                "failure_reason": "negative_net_and_causal_residual_tail",
                "affected_sides": ",".join(sides),
                "affected_assets": ",".join(assets),
                "complete_window_coverage": False,
                "descriptive_only": True,
            }
        )
    episode_membership = pd.concat(membership_parts, ignore_index=True)
    return pd.DataFrame.from_records(episodes), episode_membership


__all__ = [
    "FailureHealthConfig",
    "build_causal_decision_health",
    "group_failure_bins_into_episodes",
]
