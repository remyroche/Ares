#!/usr/bin/env python3
"""Materialize descriptive economic-opportunity states and frozen event packets.

This runner deliberately does not train a failure classifier or create an
inference-time regime label.  It preserves the historical and current
execution lineages, describes every strict economic-failure episode with a
broad multilabel taxonomy, and freezes the underlying hourly trajectory for
later recurrence analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_HISTORICAL = ROOT / (
    "data_perp/artifacts/historical_exact_model_health_failure_20260729_v3"
)
DEFAULT_CURRENT = ROOT / (
    "data_perp/artifacts/current_lineage_exact_failure_labels_resolved_july19_"
    "20260729_v1"
)
DEFAULT_CURRENT_HANDOFF = ROOT / (
    "data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_"
    "20260726_v7/joined.parquet"
)
DEFAULT_FORWARD_HANDOFF = ROOT / (
    "data_perp/artifacts/execution_ev_context_head_clean_forward_july19_"
    "20260726_v2/strict_forward_winner_inputs_and_raw_scores.parquet"
)
DEFAULT_ACTIVE = ROOT / (
    "data_perp/artifacts/regime_transition_active_head_chronological_oos_"
    "20260729_v2/chronological_oos.parquet"
)
DEFAULT_DESTINATION = ROOT / (
    "data_perp/artifacts/regime_transition_destination_chronological_oos_"
    "20260729_v1/destination_chronological_oos.parquet"
)
DEFAULT_BOCPD = ROOT / (
    "data_perp/artifacts/regime_transition_changepoint_ablation_20260727_v2/"
    "grouped_oof_predictions_and_changepoint_context.parquet"
)
DEFAULT_ORIGIN_STATE = ROOT / (
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)

IDENTITY = ("candidate_id", "__ts__", "side_name")
ROBUST_SCALE_FLOORS = {
    "opportunity_rate": 0.03,
    "positive_net_contribution": 0.0003,
    "favorable_payoff_mean": 0.0005,
    "loss_net_contribution": 0.0003,
    "adverse_payoff_magnitude_mean": 0.0005,
    "timeout_rate": 0.03,
    "timeout_conditional_net": 0.0005,
    "exit_conversion_loss_mean": 0.0005,
    "cost_mean": 0.0001,
    "net_mean": 0.0005,
    "selected_asset_hhi": 0.02,
    "distinct_assets": 1.0,
}

PRIMARY_STATE_NAMES = (
    "sparse_opportunity",
    "favorable_payoff_compression",
    "high_opportunity_poor_conversion",
    "adverse_payoff_expansion",
    "timeout_degradation",
    "exit_conversion_failure",
    "execution_liquidity_impairment",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna(default).astype(bool)


def _validate_candidate_identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    _require(
        frame,
        (
            *IDENTITY,
            "__symbol__",
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_net_ev_12h",
            "execution_cost_return",
        ),
        source,
    )
    result = frame.copy()
    for column in ("__ts__", "execution_decision_utc", "execution_label_end_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{source} contains invalid {column}")
    if result["candidate_id"].astype(str).duplicated().any():
        raise ValueError(f"{source} candidate IDs must be unique")
    if not result["execution_decision_utc"].eq(
        result["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError(f"{source} violates signal+1h decision contract")
    if not result["execution_label_end_utc"].eq(
        result["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError(f"{source} violates exact 12h label contract")
    return result


def assemble_current_selected_details(
    selected: pd.DataFrame,
    current_handoff: pd.DataFrame,
    forward_handoff: pd.DataFrame,
) -> pd.DataFrame:
    """Join exact path and score details to the frozen current global-top10 book."""

    selected = _validate_candidate_identity(selected, "current selected candidates")
    detail_columns = (
        "candidate_id",
        "__ts__",
        "side_name",
        "base_oof_score",
        "existing_alpha_ev",
        "execution_exit_reason",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
    )
    for name, frame in (
        ("current handoff", current_handoff),
        ("forward handoff", forward_handoff),
    ):
        _require(frame, detail_columns, name)
    details = pd.concat(
        [
            current_handoff.loc[:, list(detail_columns)],
            forward_handoff.loc[:, list(detail_columns)],
        ],
        ignore_index=True,
    )
    details["__ts__"] = pd.to_datetime(details["__ts__"], utc=True, errors="coerce")
    details = details.drop_duplicates("candidate_id", keep="last")
    joined = selected.merge(
        details,
        on="candidate_id",
        how="left",
        suffixes=("", "__detail"),
        validate="one_to_one",
    )
    if joined["execution_exit_reason"].isna().any():
        raise ValueError("current frozen book lacks exact rich path details")
    for column in ("__ts__", "side_name"):
        detail = f"{column}__detail"
        if not joined[column].astype(str).eq(joined[detail].astype(str)).all():
            raise ValueError(f"current detail identity mismatch: {column}")
        joined = joined.drop(columns=detail)
    joined["score_base"] = joined["base_oof_score"]
    joined["score_alpha_residual"] = joined["existing_alpha_ev"]
    joined["score_direct_execution_ev"] = joined[
        "catboost__residual__without_hpo__all_features"
    ]
    joined["score_mapped_execution_ev"] = joined[
        "causal_recent_side_isotonic_ev"
    ]
    return joined


def prepare_historical_selected(frame: pd.DataFrame) -> pd.DataFrame:
    result = _validate_candidate_identity(frame, "historical selected candidates")
    result["score_base"] = _numeric(result, "score_base_alpha")
    result["score_alpha_residual"] = np.nan
    result["score_direct_execution_ev"] = np.nan
    result["score_mapped_execution_ev"] = _numeric(result, "mapped_direct_net")
    return result


def _asset_hhi(symbols: pd.Series) -> float:
    share = symbols.astype(str).value_counts(normalize=True)
    return float(np.square(share).sum()) if len(share) else np.nan


def build_hourly_components(candidates: pd.DataFrame, lineage: str) -> pd.DataFrame:
    """Build candidate-weighted hourly economic components for one frozen book."""

    work = candidates.copy()
    work["source_utc"] = pd.to_datetime(work["__ts__"], utc=True)
    gross = _numeric(work, "execution_gross_ev_12h")
    net = _numeric(work, "execution_net_ev_12h")
    cost = _numeric(work, "execution_cost_return")
    mfe = _numeric(work, "execution_mfe_return_12h")
    exit_reason = work.get(
        "execution_exit_reason", pd.Series("", index=work.index)
    ).astype(str).str.lower()
    opportunity = (
        _bool(work, "opportunity_gross_above_cost_0bps")
        if "opportunity_gross_above_cost_0bps" in work
        else gross.gt(cost)
    )
    timeout = (
        _bool(work, "exit_is_timeout")
        if "exit_is_timeout" in work
        else exit_reason.str.contains("timeout", regex=False)
    )
    work["_gross"] = gross
    work["_net"] = net
    work["_cost"] = cost
    work["_opportunity"] = opportunity.astype(float)
    work["_positive_contribution"] = net.clip(lower=0.0)
    work["_favorable"] = gross.where(opportunity)
    work["_adverse"] = (-net).where(net.le(0.0))
    work["_loss_contribution"] = (-net).clip(lower=0.0)
    work["_timeout"] = timeout.astype(float)
    work["_timeout_loss"] = (-net).clip(lower=0.0).where(timeout, 0.0)
    work["_timeout_net"] = net.where(timeout)
    work["_conversion_loss"] = (mfe - gross).clip(lower=0.0)
    work["_long"] = work["side_name"].astype(str).str.lower().eq("long").astype(float)
    for column in (
        "score_base",
        "score_alpha_residual",
        "score_direct_execution_ev",
        "score_mapped_execution_ev",
    ):
        work[f"_{column}"] = _numeric(work, column)
    grouped = work.groupby("source_utc", observed=True, sort=True)
    hourly = grouped.agg(
        selected_rows=("candidate_id", "size"),
        distinct_assets=("__symbol__", "nunique"),
        long_share=("_long", "mean"),
        gross_mean=("_gross", "mean"),
        net_mean=("_net", "mean"),
        cost_mean=("_cost", "mean"),
        opportunity_rate=("_opportunity", "mean"),
        positive_net_contribution=("_positive_contribution", "mean"),
        favorable_payoff_mean=("_favorable", "mean"),
        adverse_payoff_magnitude_mean=("_adverse", "mean"),
        loss_net_contribution=("_loss_contribution", "mean"),
        timeout_rate=("_timeout", "mean"),
        timeout_loss_contribution=("_timeout_loss", "mean"),
        timeout_conditional_net=("_timeout_net", "mean"),
        exit_conversion_loss_mean=("_conversion_loss", "mean"),
        score_base_mean=("_score_base", "mean"),
        score_base_std=("_score_base", "std"),
        score_alpha_residual_mean=("_score_alpha_residual", "mean"),
        score_alpha_residual_std=("_score_alpha_residual", "std"),
        score_direct_execution_ev_mean=("_score_direct_execution_ev", "mean"),
        score_direct_execution_ev_std=("_score_direct_execution_ev", "std"),
        score_mapped_execution_ev_mean=("_score_mapped_execution_ev", "mean"),
        score_mapped_execution_ev_std=("_score_mapped_execution_ev", "std"),
        outcome_available_utc=("execution_label_end_utc", "max"),
    ).reset_index()
    hhi = grouped["__symbol__"].apply(_asset_hhi).rename("selected_asset_hhi")
    hourly = hourly.merge(
        hhi.reset_index(), on="source_utc", how="left", validate="one_to_one"
    )
    hourly.insert(0, "lineage", lineage)
    return hourly


def attach_context(
    hourly: pd.DataFrame,
    health: pd.DataFrame,
    active: pd.DataFrame,
    destination: pd.DataFrame,
    bocpd: pd.DataFrame | None = None,
    origin_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = hourly.copy()
    local_health = health.copy()
    local_health["source_utc"] = pd.to_datetime(
        local_health["source_utc"], utc=True, errors="coerce"
    )
    keep_health = [
        name
        for name in local_health
        if name == "source_utc" or name.startswith("health__")
    ]
    result = result.merge(
        local_health.loc[:, keep_health],
        on="source_utc",
        how="left",
        validate="one_to_one",
    )
    local_active = active.loc[:, ["source_utc", "prediction"]].copy()
    local_active["source_utc"] = pd.to_datetime(
        local_active["source_utc"], utc=True, errors="coerce"
    )
    local_active = local_active.drop_duplicates("source_utc", keep="last").rename(
        columns={"prediction": "active_transition_probability"}
    )
    result = result.merge(
        local_active, on="source_utc", how="left", validate="one_to_one"
    )
    destination_columns = [
        name
        for name in destination
        if name == "source_utc"
        or name.startswith("p_destination__")
        or name in {"destination_confidence", "destination_entropy"}
    ]
    local_destination = destination.loc[:, destination_columns].copy()
    local_destination["source_utc"] = pd.to_datetime(
        local_destination["source_utc"], utc=True, errors="coerce"
    )
    local_destination = local_destination.drop_duplicates("source_utc", keep="last")
    result = result.merge(
        local_destination, on="source_utc", how="left", validate="one_to_one"
    )
    result["destination_available"] = result[
        "destination_confidence"
    ].notna()
    if bocpd is not None:
        columns = [
            "source_utc",
            *[
                name
                for name in bocpd
                if name.startswith("bocpd_context__")
            ],
        ]
        local = bocpd.loc[:, columns].copy()
        local["source_utc"] = pd.to_datetime(local["source_utc"], utc=True)
        local = local.drop_duplicates("source_utc", keep="last")
        result = result.merge(
            local, on="source_utc", how="left", validate="one_to_one"
        )
    if origin_state is not None:
        columns = [
            "source_utc",
            *[
                name
                for name in origin_state
                if name.startswith("state_context__")
            ],
        ]
        local = origin_state.loc[:, columns].copy()
        local["source_utc"] = pd.to_datetime(local["source_utc"], utc=True)
        local = local.drop_duplicates("source_utc", keep="last")
        result = result.merge(
            local, on="source_utc", how="left", validate="one_to_one"
        )
    return result


def _robust_reference(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return np.nan, np.nan
    median = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - median)) * 1.4826)
    return median, mad


def classify_opportunity_state(
    event: pd.Series,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    """Apply predeclared broad multilabel rules to one resolved event summary.

    Thresholds are descriptive constants, not fitted cutoffs.  Every z-score
    uses only the caller-supplied prior, resolved same-lineage reference.
    """

    result: dict[str, Any] = {}
    z: dict[str, float] = {}
    for column, floor in ROBUST_SCALE_FLOORS.items():
        median, mad = _robust_reference(reference[column])
        scale = max(mad, float(floor))
        value = float(event[column]) if pd.notna(event[column]) else np.nan
        score = (
            float((value - median) / scale)
            if np.isfinite(value) and np.isfinite(median) and scale > 0
            else np.nan
        )
        z[column] = score
        result[f"state_reference__{column}_median"] = median
        result[f"state_reference__{column}_robust_scale"] = scale
        result[f"state_z__{column}"] = score

    result["state__sparse_opportunity"] = bool(
        z["opportunity_rate"] <= -1.0
        and z["positive_net_contribution"] <= -1.0
    )
    result["state__favorable_payoff_compression"] = bool(
        z["favorable_payoff_mean"] <= -1.0 and z["net_mean"] <= -1.0
    )
    result["state__high_opportunity_poor_conversion"] = bool(
        z["opportunity_rate"] >= -0.5
        and z["exit_conversion_loss_mean"] >= 1.0
        and z["net_mean"] <= -1.0
    )
    result["state__adverse_payoff_expansion"] = bool(
        z["loss_net_contribution"] >= 1.0 and z["net_mean"] <= -1.0
    )
    result["state__timeout_degradation"] = bool(
        z["timeout_rate"] >= 1.0 and z["timeout_conditional_net"] <= -1.0
    )
    result["state__exit_conversion_failure"] = bool(
        z["exit_conversion_loss_mean"] >= 1.0 and z["net_mean"] <= -1.0
    )
    result["state__execution_liquidity_impairment"] = bool(
        z["cost_mean"] >= 1.0
        and (z["selected_asset_hhi"] >= 1.0 or z["distinct_assets"] <= -1.0)
    )
    active = sum(bool(result[f"state__{name}"]) for name in PRIMARY_STATE_NAMES)
    result["state__normal_opportunity"] = bool(
        active == 0 and z["net_mean"] >= -0.5 and z["opportunity_rate"] >= -0.5
    )
    result["state__mixed"] = bool(active >= 2)
    result["state__unclassified"] = bool(
        active == 0 and not result["state__normal_opportunity"]
    )
    result["state_label_count"] = int(active)
    result["state_labels"] = "|".join(
        state.removeprefix("state__")
        for state, flag in result.items()
        if state.startswith("state__")
        and state
        not in {
            "state__mixed",
            "state__unclassified",
            "state__normal_opportunity",
        }
        and bool(flag)
    )
    return result


def _window_summary(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    weights = pd.to_numeric(frame["selected_rows"], errors="coerce").fillna(0.0)
    for column in (
        "opportunity_rate",
        "positive_net_contribution",
        "favorable_payoff_mean",
        "adverse_payoff_magnitude_mean",
        "loss_net_contribution",
        "timeout_rate",
        "timeout_loss_contribution",
        "timeout_conditional_net",
        "exit_conversion_loss_mean",
        "cost_mean",
        "gross_mean",
        "net_mean",
        "long_share",
        "selected_asset_hhi",
        "distinct_assets",
        "score_base_mean",
        "score_alpha_residual_mean",
        "score_direct_execution_ev_mean",
        "score_mapped_execution_ev_mean",
        "active_transition_probability",
        "destination_confidence",
        "destination_entropy",
    ):
        values = pd.to_numeric(frame.get(column), errors="coerce")
        valid = values.notna() & weights.gt(0.0)
        result[column] = (
            float(np.average(values.loc[valid], weights=weights.loc[valid]))
            if valid.any()
            else np.nan
        )
    result["selected_rows"] = float(weights.sum())
    result["source_hours"] = int(frame["source_utc"].nunique())
    return result


def _recovery_time(
    hourly: pd.DataFrame,
    incident_end: pd.Timestamp,
    reference_net: float,
    horizon_hours: int = 72,
) -> tuple[pd.Timestamp | pd.NaT, float]:
    post = hourly.loc[
        hourly["source_utc"].ge(incident_end)
        & hourly["source_utc"].le(
            incident_end + pd.Timedelta(hours=horizon_hours)
        )
    ].sort_values("source_utc")
    if post.empty or not np.isfinite(reference_net):
        return pd.NaT, np.nan
    recovered = pd.to_numeric(post["net_mean"], errors="coerce").rolling(
        6, min_periods=6
    ).mean().ge(reference_net)
    if not recovered.any():
        return pd.NaT, np.nan
    stamp = pd.Timestamp(post.loc[recovered, "source_utc"].iloc[0])
    return stamp, float((stamp - incident_end) / pd.Timedelta(hours=1))


def consolidate_failure_incidents(
    events: pd.DataFrame,
    lineage: str,
    merge_gap_hours: int = 6,
) -> pd.DataFrame:
    """Merge overlapping broad/strict anchors into lineage-local incidents."""

    _require(
        events,
        (
            "economic_event_id",
            "failure_label",
            "anchor_source_utc",
            "target_available_utc",
        ),
        f"{lineage} events",
    )
    work = events.copy()
    work["anchor_source_utc"] = pd.to_datetime(
        work["anchor_source_utc"], utc=True, errors="coerce"
    )
    work["target_available_utc"] = pd.to_datetime(
        work["target_available_utc"], utc=True, errors="coerce"
    )
    work = work.sort_values(
        ["anchor_source_utc", "failure_label", "economic_event_id"], kind="stable"
    )
    incidents: list[dict[str, Any]] = []
    current: list[pd.Series] = []
    current_end: pd.Timestamp | None = None
    for _, row in work.iterrows():
        anchor = pd.Timestamp(row["anchor_source_utc"])
        active_end = anchor + pd.Timedelta(hours=12)
        if (
            current
            and current_end is not None
            and anchor > current_end + pd.Timedelta(hours=merge_gap_hours)
        ):
            incidents.append(_finish_incident(current, lineage, len(incidents) + 1))
            current = []
            current_end = None
        current.append(row)
        current_end = active_end if current_end is None else max(current_end, active_end)
    if current:
        incidents.append(_finish_incident(current, lineage, len(incidents) + 1))
    result = pd.DataFrame(incidents)
    if result["opportunity_incident_id"].duplicated().any():
        raise ValueError("opportunity incident IDs must be unique")
    return result


def _finish_incident(
    rows: list[pd.Series],
    lineage: str,
    ordinal: int,
) -> dict[str, Any]:
    anchors = [pd.Timestamp(row["anchor_source_utc"]) for row in rows]
    labels = {str(row["failure_label"]) for row in rows}
    start = min(anchors)
    end = max(anchor + pd.Timedelta(hours=12) for anchor in anchors)
    return {
        "lineage": lineage,
        "opportunity_incident_id": (
            f"{lineage}__opportunity_incident_{start:%Y%m%d%H}_{ordinal:04d}"
        ),
        "incident_start_utc": start,
        "incident_end_utc": end,
        "anchor_source_utc": start,
        "incident_has_broad_failure": "broad" in labels,
        "incident_has_strict_failure": "strict" in labels,
        "source_event_ids": "|".join(str(row["economic_event_id"]) for row in rows),
        "source_anchor_count": len(rows),
        "source_target_available_utc": max(
            pd.Timestamp(row["target_available_utc"]) for row in rows
        ),
    }


def build_event_packets(
    hourly: pd.DataFrame,
    events: pd.DataFrame,
    lineage: str,
    reference_days: int = 30,
    minimum_reference_hours: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one packet and frozen trajectory per strict-containing incident."""

    incidents = consolidate_failure_incidents(events, lineage)
    incidents = incidents.loc[incidents["incident_has_strict_failure"]].copy()
    rows: list[dict[str, Any]] = []
    trajectories: list[pd.DataFrame] = []
    for event in incidents.itertuples(index=False):
        anchor = pd.Timestamp(event.incident_start_utc)
        incident_end = pd.Timestamp(event.incident_end_utc)
        reference = hourly.loc[
            hourly["source_utc"].ge(anchor - pd.Timedelta(days=reference_days))
            & hourly["source_utc"].lt(anchor)
            & hourly["outcome_available_utc"].lt(anchor)
        ].copy()
        event_window = hourly.loc[
            hourly["source_utc"].ge(anchor)
            & hourly["source_utc"].lt(incident_end)
        ].copy()
        if len(reference) < minimum_reference_hours:
            reference_status = "INSUFFICIENT_CAUSAL_REFERENCE"
        else:
            reference_status = "AVAILABLE"
        summary = _window_summary(event_window)
        row: dict[str, Any] = {
            "lineage": lineage,
            "opportunity_incident_id": event.opportunity_incident_id,
            "source_event_ids": event.source_event_ids,
            "source_anchor_count": int(event.source_anchor_count),
            "incident_has_broad_failure": bool(event.incident_has_broad_failure),
            "incident_has_strict_failure": bool(event.incident_has_strict_failure),
            "incident_start_utc": anchor,
            "incident_end_utc": incident_end,
            "anchor_source_utc": anchor,
            "reference_start_utc": (
                reference["source_utc"].min() if len(reference) else pd.NaT
            ),
            "reference_end_utc": (
                reference["source_utc"].max() if len(reference) else pd.NaT
            ),
            "reference_hours": int(len(reference)),
            "reference_status": reference_status,
            **{f"event__{key}": value for key, value in summary.items()},
        }
        if reference_status == "AVAILABLE":
            row.update(classify_opportunity_state(pd.Series(summary), reference))
        else:
            for state in PRIMARY_STATE_NAMES:
                row[f"state__{state}"] = False
            row["state__normal_opportunity"] = False
            row["state__mixed"] = False
            row["state__unclassified"] = True
            row["state_label_count"] = 0
            row["state_labels"] = ""
        reference_net, _ = _robust_reference(reference["net_mean"])
        recovery, recovery_hours = _recovery_time(hourly, incident_end, reference_net)
        row["recovery_source_utc"] = recovery
        row["recovery_hours"] = recovery_hours
        row["recovered_within_72h"] = bool(pd.notna(recovery))
        row["recovery_censored_72h"] = bool(pd.isna(recovery))
        recovery_end = (
            recovery if pd.notna(recovery) else incident_end + pd.Timedelta(hours=72)
        )
        resolution_rows = hourly.loc[
            hourly["source_utc"].ge(anchor - pd.Timedelta(hours=24))
            & hourly["source_utc"].le(recovery_end)
        ]
        outcome_available = pd.to_datetime(
            resolution_rows["outcome_available_utc"], utc=True, errors="coerce"
        ).max()
        row["packet_available_utc"] = max(
            pd.Timestamp(event.source_target_available_utc),
            outcome_available + pd.Timedelta(hours=1),
        )
        row["packet_frozen"] = bool(
            pd.notna(outcome_available)
            and outcome_available <= hourly["outcome_available_utc"].max()
        )
        rows.append(row)

        trajectory = hourly.loc[
            hourly["source_utc"].ge(anchor - pd.Timedelta(hours=24))
            & hourly["source_utc"].le(recovery_end)
        ].copy()
        trajectory.insert(1, "opportunity_incident_id", event.opportunity_incident_id)
        trajectory.insert(
            2,
            "relative_hour",
            ((trajectory["source_utc"] - anchor) / pd.Timedelta(hours=1)).astype(int),
        )
        trajectory.insert(
            3,
            "packet_phase",
            np.select(
                [
                    trajectory["source_utc"].lt(anchor),
                    trajectory["source_utc"].lt(incident_end),
                ],
                ["origin", "event"],
                default="recovery",
            ),
        )
        trajectories.append(trajectory)
    packet = pd.DataFrame(rows)
    trajectory = (
        pd.concat(trajectories, ignore_index=True)
        if trajectories
        else pd.DataFrame()
    )
    return packet, trajectory


def _load_lineage(
    root: Path,
    lineage: str,
    active: pd.DataFrame,
    destination: pd.DataFrame,
    bocpd: pd.DataFrame,
    origin_state: pd.DataFrame,
    current_handoff: Path | None = None,
    forward_handoff: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    if lineage == "historical_2025_raw_alpha":
        candidate_path = root / "frozen_global_top10_candidates.parquet"
        hourly_path = root / "hourly_exact_model_health_and_failure_labels.parquet"
        candidates = prepare_historical_selected(pd.read_parquet(candidate_path))
    else:
        candidate_path = root / "frozen_global_top10_mapped_candidates.parquet"
        hourly_path = root / "hourly_current_health_and_failure_labels.parquet"
        if current_handoff is None or forward_handoff is None:
            raise ValueError("current lineage requires both rich handoffs")
        candidates = assemble_current_selected_details(
            pd.read_parquet(candidate_path),
            pd.read_parquet(current_handoff),
            pd.read_parquet(forward_handoff),
        )
    health = pd.read_parquet(hourly_path)
    components = build_hourly_components(candidates, lineage)
    components = attach_context(
        components, health, active, destination, bocpd, origin_state
    )
    event_path = root / "economic_failure_events.parquet"
    events = pd.read_parquet(event_path)
    sources = [candidate_path, hourly_path, event_path]
    if current_handoff is not None:
        sources.append(current_handoff)
    if forward_handoff is not None:
        sources.append(forward_handoff)
    return components, events, candidates, sources


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    active_path = Path(args.active)
    destination_path = Path(args.destination)
    bocpd_path = Path(args.bocpd)
    origin_state_path = Path(args.origin_state)
    active = pd.read_parquet(active_path)
    destination = pd.read_parquet(destination_path)
    bocpd = pd.read_parquet(bocpd_path)
    origin_state = pd.read_parquet(origin_state_path)
    components: list[pd.DataFrame] = []
    packets: list[pd.DataFrame] = []
    trajectories: list[pd.DataFrame] = []
    selected_books: list[pd.DataFrame] = []
    sources: list[Path] = [
        active_path,
        destination_path,
        bocpd_path,
        origin_state_path,
    ]
    specifications = (
        (
            Path(args.historical),
            "historical_2025_raw_alpha",
            None,
            None,
        ),
        (
            Path(args.current),
            "current_2026_execution_ev",
            Path(args.current_handoff),
            Path(args.forward_handoff),
        ),
    )
    for root, lineage, current_handoff, forward_handoff in specifications:
        hourly, events, candidates, local_sources = _load_lineage(
            root,
            lineage,
            active,
            destination,
            bocpd,
            origin_state,
            current_handoff,
            forward_handoff,
        )
        packet, trajectory = build_event_packets(
            hourly,
            events,
            lineage,
            reference_days=int(args.reference_days),
            minimum_reference_hours=int(args.minimum_reference_hours),
        )
        components.append(hourly)
        packets.append(packet)
        trajectories.append(trajectory)
        frozen_candidates = candidates.copy()
        frozen_candidates.insert(0, "lineage", lineage)
        selected_books.append(frozen_candidates)
        sources.extend(local_sources)
    hourly_all = pd.concat(components, ignore_index=True)
    packet_all = pd.concat(packets, ignore_index=True)
    trajectory_all = pd.concat(trajectories, ignore_index=True)
    selected_all = pd.concat(selected_books, ignore_index=True)
    summary_columns = [
        "lineage",
        *[f"state__{state}" for state in PRIMARY_STATE_NAMES],
        "state__normal_opportunity",
        "state__mixed",
        "state__unclassified",
    ]
    summary = (
        packet_all.loc[:, summary_columns]
        .groupby("lineage", observed=True)
        .agg(["sum", "mean"])
    )
    summary.columns = [f"{left}__{right}" for left, right in summary.columns]
    summary = summary.reset_index()
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "hourly_components": output / "hourly_opportunity_state_components.parquet",
        "event_packets": output / "strict_event_packets.parquet",
        "event_trajectories": output / "strict_event_trajectories.parquet",
        "taxonomy_summary": output / "taxonomy_summary.parquet",
        "selected_candidates": output / "frozen_selected_candidates.parquet",
    }
    hourly_all.to_parquet(paths["hourly_components"], index=False, compression="zstd")
    packet_all.to_parquet(paths["event_packets"], index=False, compression="zstd")
    trajectory_all.to_parquet(
        paths["event_trajectories"], index=False, compression="zstd"
    )
    summary.to_parquet(paths["taxonomy_summary"], index=False, compression="zstd")
    selected_all.to_parquet(
        paths["selected_candidates"], index=False, compression="zstd"
    )
    source_manifest = [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in dict.fromkeys(sources)
    ]
    report = {
        "schema": "economic_opportunity_state_packets_v1",
        "status": "DESCRIPTIVE_MULTILABEL_PACKETS_COMPLETE_NO_ROUTER_AUTHORIZED",
        "purpose": (
            "separate economic opportunity state from generic market regime; "
            "freeze resolved strict-event evidence without training a hard router"
        ),
        "selection_contracts": {
            "historical": "frozen pooled-global raw-alpha top10; candidate-ID tie break",
            "current": "frozen pooled-global top10 after causal recent side-EV mapping; candidate-ID tie break",
            "never": "per timestamp, side quota, transition veto, or outcome-selected admission",
        },
        "taxonomy_rules": {
            "robust_scale_floors": ROBUST_SCALE_FLOORS,
            "primary_states": PRIMARY_STATE_NAMES,
            "fixed_z_thresholds": {
                "adverse_activation": 1.0,
                "normal_floor": -0.5,
                "mixed_minimum_labels": 2,
            },
        },
        "reference_contract": {
            "lookback_days": int(args.reference_days),
            "minimum_hours": int(args.minimum_reference_hours),
            "causal_resolution": "reference outcome_available_utc < event anchor",
            "threshold": "prior median +/- max(1.4826*MAD, fixed economic floor)",
            "multilabel": True,
            "mixed": "two or more primary labels",
        },
        "rows": {
            "hourly_components": int(len(hourly_all)),
            "strict_event_packets": int(len(packet_all)),
            "strict_event_trajectories": int(len(trajectory_all)),
            "frozen_selected_candidates": int(len(selected_all)),
        },
        "strict_event_support": {
            str(lineage): int(rows)
            for lineage, rows in packet_all.groupby("lineage", observed=True).size().items()
        },
        "promotion_gate": {
            "required_independent_events_per_lineage": "60-100",
            "supervised_router_authorized": False,
            "reason": (
                "support remains below the promotion range and historical/current "
                "lineages are not pooled as one exact model lineage"
            ),
        },
        "explicitly_not_done": [
            "no failure classifier",
            "no transition exposure rule",
            "no opportunity-state admission veto",
            "no pooling of reconstructed and current lineages",
            "no use of resolved taxonomy labels as decision-time features",
        ],
        "sources": source_manifest,
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, report)
    (output / "manifest.sha256").write_text(
        _sha256(manifest_path) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--current-handoff", type=Path, default=DEFAULT_CURRENT_HANDOFF)
    parser.add_argument("--forward-handoff", type=Path, default=DEFAULT_FORWARD_HANDOFF)
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--bocpd", type=Path, default=DEFAULT_BOCPD)
    parser.add_argument("--origin-state", type=Path, default=DEFAULT_ORIGIN_STATE)
    parser.add_argument("--reference-days", type=int, default=30)
    parser.add_argument("--minimum-reference-hours", type=int, default=168)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
