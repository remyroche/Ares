#!/usr/bin/env python3
"""Build matched-control dossiers for high-priority residual-state events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_state_discovery import (  # noqa: E402
    feature_quality_metrics,
    matched_control_feature_diagnostics,
)

DEFAULT_ROOT = ROOT / "data_perp/reports/global_residual_state_discovery_20260711_v1"
DEFAULT_STATES = (
    DEFAULT_ROOT / "global_side_latent_states/side_timestamp_market_states.parquet"
)
LOCAL_KEYS = ("__ts__", "side_name", "archetype_policy_key")
EVENT_KEYS = ("day", "side_name", "archetype_policy_key")
OUTCOME_COLUMNS = {
    "clean_exec",
    "hit_probability",
    "ev_after_1pct",
    "full_path_bad_mae_1r",
    "timeout",
    "dirty_positive",
    "target_signed_surprise",
    "target_mean_ev",
    "target_negative_ev",
    "target_bad_mae_rate",
    "target_timeout_rate",
}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _state_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "universe_rows",
        "universe_assets",
        "selected_rows",
        "selected_assets",
    }
    features = [
        name
        for name in frame.select_dtypes(include=[np.number, "bool"]).columns
        if not name.startswith("target_")
        and name not in excluded
        and "coverage__" not in name
    ]

    # The phase block is the current hypothesis. Put it first so a bounded
    # matched-control diagnostic actually tests the causal lifecycle features
    # rather than spending its budget on broad aggregate variants.
    def priority(name: str) -> tuple[int, str]:
        lowered = str(name).lower()
        if name.startswith("state_phase__"):
            return (0, name)
        if any(
            token in lowered
            for token in (
                "liquidation",
                "flush",
                "oi_",
                "price_oi",
                "breadth",
                "short_cover",
                "systemic",
                "synchron",
            )
        ):
            return (1, name)
        return (2, name)

    return sorted(features, key=priority)


def _matching_columns(frame: pd.DataFrame) -> list[str]:
    groups = (
        ("return", ("ret", "return")),
        ("volatility", ("rv_", "volatility", "atr_")),
        ("rank", ("score_mean", "score_std")),
        ("archetype", ("selected_archetype_share__",)),
    )
    selected: list[str] = []
    numeric = set(frame.select_dtypes(include=[np.number, "bool"]).columns)
    for _, hints in groups:
        matches = [
            name
            for name in frame.columns
            if name in numeric and any(hint in name.lower() for hint in hints)
        ]
        selected.extend(matches[:4])
    selected.extend(
        [name for name in ("selected_rows", "universe_rows") if name in frame]
    )
    return [
        name
        for name in dict.fromkeys(selected)
        if name not in OUTCOME_COLUMNS and not name.startswith("target_")
    ]


def _event_ledger_path(root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return Path(supplied)
    manifest_path = root / "event_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Event dossier needs --ledger or an event_manifest.json with a ledger path"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = manifest.get("ledger")
    if not ledger:
        raise KeyError("event_manifest.json does not contain a ledger path")
    return Path(str(ledger))


def _local_state_observations(states: pd.DataFrame, ledger_path: Path) -> pd.DataFrame:
    """Expand side-level state coordinates into observed side/archetype streams.

    The state itself is pre-entry and shared by a side at a timestamp.  Candidate
    activity and score context are aggregated only from the policy-selected
    stream so event controls represent the exact population that produced the
    retrospective event. Outcome fields are retained for dossier reporting but
    are explicitly excluded from matching columns and never become state inputs.
    """
    available = set(pq.ParquetFile(ledger_path).schema_arrow.names)
    requested = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "selected_for_monitor",
        "score",
        "score_meta_base_soft_label",
        "base_score",
        "historical_rank",
        "production_adjusted_rank",
        "clean_exec",
        "hit_probability",
        "ev_after_1pct",
        "full_path_bad_mae_1r",
        "timeout",
        "dirty_positive",
    ]
    ledger = pd.read_parquet(
        ledger_path, columns=[name for name in requested if name in available]
    )
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce")
    ledger = ledger.loc[ledger["__ts__"].notna()].copy()
    if "selected_for_monitor" in ledger:
        ledger = ledger.loc[
            ledger["selected_for_monitor"].fillna(False).astype(bool)
        ].copy()
    required = {"side_name", "archetype_policy_key"}
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise KeyError(f"Ledger does not contain local population keys: {missing}")
    for name in (
        "score",
        "score_meta_base_soft_label",
        "base_score",
        "historical_rank",
        "production_adjusted_rank",
        "clean_exec",
        "hit_probability",
        "ev_after_1pct",
        "full_path_bad_mae_1r",
        "timeout",
        "dirty_positive",
    ):
        if name in ledger:
            ledger[name] = pd.to_numeric(ledger[name], errors="coerce").astype(
                np.float32
            )
    if "clean_exec" in ledger and "hit_probability" in ledger:
        ledger["_signed_surprise"] = ledger["clean_exec"] - ledger["hit_probability"]
    else:
        ledger["_signed_surprise"] = np.nan
    if "ev_after_1pct" in ledger:
        ledger["_negative_ev"] = (-ledger["ev_after_1pct"]).clip(lower=0.0)
    else:
        ledger["_negative_ev"] = np.nan
    aggregations: dict[str, tuple[str, str]] = {
        "local_selected_rows": ("__ts__", "size"),
        "local_selected_assets": ("__symbol__", "nunique"),
        "target_signed_surprise": ("_signed_surprise", "mean"),
        "target_mean_ev": ("ev_after_1pct", "mean"),
        "target_negative_ev": ("_negative_ev", "mean"),
        "target_bad_mae_rate": ("full_path_bad_mae_1r", "mean"),
        "target_timeout_rate": ("timeout", "mean"),
    }
    for source, target in (
        ("score", "local_score_mean"),
        ("score_meta_base_soft_label", "local_meta_score_mean"),
        ("base_score", "local_base_score_mean"),
        ("historical_rank", "local_rank_mean"),
        ("production_adjusted_rank", "local_production_rank_mean"),
    ):
        if source in ledger:
            aggregations[target] = (source, "mean")
    local = (
        ledger.groupby(list(LOCAL_KEYS), observed=True, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    base = states.copy()
    base["__ts__"] = pd.to_datetime(base["__ts__"], utc=True, errors="coerce")
    base = base.loc[base["__ts__"].notna()].copy()
    output = local.merge(
        base,
        on=["__ts__", "side_name"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_state"),
    )
    output["day"] = output["__ts__"].dt.floor("D")
    return output


def _attach_event_membership(
    states: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    states = states.copy()
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True, errors="coerce")
    states["day"] = states["__ts__"].dt.floor("D")
    membership = membership.copy()
    membership["day"] = pd.to_datetime(membership["day"], utc=True, errors="coerce")
    event_local_day = (
        membership.groupby(list(EVENT_KEYS), observed=True)
        .agg(
            event_ids=(
                "event_id",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            event_evidence=(
                "evidence_type",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    states = states.merge(
        event_local_day, on=list(EVENT_KEYS), how="left", validate="many_to_one"
    )
    states["is_event"] = states["event_ids"].notna()
    return states


def _period_metrics(frame: pd.DataFrame, event: pd.Series) -> list[dict[str, Any]]:
    start = pd.Timestamp(event["event_start"])
    end = pd.Timestamp(event["event_end"])
    periods = (
        ("pre_24h", start - pd.Timedelta(hours=24), start),
        ("event", start, end + pd.Timedelta(days=1)),
        ("post_24h", end + pd.Timedelta(days=1), end + pd.Timedelta(days=2)),
        ("post_48h", end + pd.Timedelta(days=1), end + pd.Timedelta(days=3)),
    )
    rows: list[dict[str, Any]] = []
    for label, left, right in periods:
        local = frame[frame["__ts__"].ge(left) & frame["__ts__"].lt(right)]
        rows.append(
            {
                "event_id": event["event_id"],
                "side_name": event["side_name"],
                "archetype_policy_key": event["archetype_policy_key"],
                "period": label,
                "hours": int(len(local)),
                "selected_rows": int(
                    pd.to_numeric(
                        local.get("local_selected_rows", local.get("selected_rows")),
                        errors="coerce",
                    ).sum()
                ),
                "mean_signed_surprise": float(
                    pd.to_numeric(
                        local.get("target_signed_surprise"), errors="coerce"
                    ).mean()
                ),
                "mean_ev": float(
                    pd.to_numeric(local.get("target_mean_ev"), errors="coerce").mean()
                ),
                "negative_ev_rate": float(
                    pd.to_numeric(local.get("target_negative_ev"), errors="coerce")
                    .gt(0)
                    .mean()
                ),
                "bad_mae_rate": float(
                    pd.to_numeric(
                        local.get("target_bad_mae_rate"), errors="coerce"
                    ).mean()
                ),
                "timeout_rate": float(
                    pd.to_numeric(
                        local.get("target_timeout_rate"), errors="coerce"
                    ).mean()
                ),
            }
        )
    return rows


def _state_refined_event_membership(
    states: pd.DataFrame,
    membership: pd.DataFrame,
    feature_columns: list[str],
    distance_threshold: float = 5.0,
) -> pd.DataFrame:
    daily = (
        states.groupby(list(EVENT_KEYS), observed=True)[feature_columns]
        .median()
        .reset_index()
    )
    values = daily[feature_columns].apply(pd.to_numeric, errors="coerce")
    center = values.median()
    scale = (
        (values.quantile(0.75) - values.quantile(0.25)).replace(0.0, 1.0).fillna(1.0)
    )
    daily[feature_columns] = ((values - center) / scale).fillna(0.0).clip(-8.0, 8.0)
    event_days = (
        membership[["event_id", *EVENT_KEYS]]
        .drop_duplicates()
        .merge(daily, on=list(EVENT_KEYS), how="left")
    )
    rows: list[dict[str, Any]] = []
    for event_id, local in event_days.groupby("event_id", observed=True, sort=True):
        local = local.sort_values("day", kind="stable")
        split = 0
        previous_day: pd.Timestamp | None = None
        previous_vector: np.ndarray | None = None
        for row in local.itertuples(index=False):
            day = pd.Timestamp(getattr(row, "day"))
            vector = np.asarray(
                [getattr(row, name) for name in feature_columns], dtype=float
            )
            distance = (
                float(
                    np.linalg.norm(vector - previous_vector)
                    / np.sqrt(max(len(vector), 1))
                )
                if previous_vector is not None
                else 0.0
            )
            gap = (day - previous_day).days if previous_day is not None else 0
            if previous_day is not None and (
                gap > 2 or distance > float(distance_threshold)
            ):
                split += 1
            rows.append(
                {
                    "event_id": event_id,
                    "state_refined_event_id": f"{event_id}-S{split:02d}",
                    "day": day,
                    "side_name": getattr(row, "side_name"),
                    "archetype_policy_key": getattr(row, "archetype_policy_key"),
                    "state_distance_from_previous_day": distance,
                }
            )
            previous_day = day
            previous_vector = vector
    return pd.DataFrame(rows)


def _recurring_mechanism_feature_summary(
    per_event_diagnostics: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate repeated same-side/archetype event signatures.

    A large single-event AUC is useful for a dossier but insufficient for a
    model feature. This table retains direction consistency and repeated-event
    support so state research can distinguish reusable mechanisms from
    calendar-specific coincidences.
    """
    if per_event_diagnostics.empty:
        return pd.DataFrame()
    event_columns = [
        name
        for name in (
            "event_id",
            "event_start",
            "event_end",
            "side_name",
            "archetype_policy_key",
            "state_failure_mechanism",
            "event_class",
            "event_priority",
            "selected_rows",
        )
        if name in events.columns
    ]
    joined = per_event_diagnostics.merge(
        events[event_columns], on="event_id", how="left", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    groups = (
        "side_name",
        "archetype_policy_key",
        "state_failure_mechanism",
        "feature",
    )
    for keys, local in joined.groupby(list(groups), observed=True, sort=True):
        local = local.dropna(
            subset=["univariate_event_auc", "standardized_mean_difference"]
        )
        if local.empty:
            continue
        smd = pd.to_numeric(local["standardized_mean_difference"], errors="coerce")
        sign = np.sign(smd.to_numpy(dtype=np.float64))
        sign = sign[sign != 0.0]
        sign_consistency = (
            float(max(np.mean(sign > 0.0), np.mean(sign < 0.0))) if len(sign) else 0.0
        )
        episodes = int(local["event_id"].nunique())
        median_auc = float(
            pd.to_numeric(local["univariate_event_auc"], errors="coerce").median()
        )
        median_smd = float(smd.median())
        rows.append(
            {
                "side_name": keys[0],
                "archetype_policy_key": keys[1],
                "state_failure_mechanism": keys[2],
                "feature": keys[3],
                "episodes": episodes,
                "total_event_rows": int(
                    pd.to_numeric(local.get("selected_rows"), errors="coerce").sum()
                ),
                "median_event_auc": median_auc,
                "median_incremental_matched_control_auc": float(
                    pd.to_numeric(
                        local["incremental_matched_control_auc"], errors="coerce"
                    ).median()
                ),
                "median_standardized_mean_difference": median_smd,
                "sign_consistency": sign_consistency,
                "first_event_start": pd.to_datetime(
                    local["event_start"], utc=True, errors="coerce"
                ).min(),
                "last_event_end": pd.to_datetime(
                    local["event_end"], utc=True, errors="coerce"
                ).max(),
                "stable_observable_candidate": bool(
                    episodes >= 2 and median_auc >= 0.60 and sign_consistency >= 0.67
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "stable_observable_candidate",
            "episodes",
            "median_event_auc",
            "sign_consistency",
        ],
        ascending=[False, False, False, False],
        kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory; defaults to <root>/event_dossiers.",
    )
    parser.add_argument("--top-events", type=int, default=25)
    parser.add_argument("--max-diagnostic-features", type=int, default=180)
    args = parser.parse_args()
    output = (
        Path(args.output)
        if args.output is not None
        else Path(args.root) / "event_dossiers"
    )
    output.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(args.states)
    events = pd.read_csv(Path(args.root) / "unreliability_event_catalog.csv")
    membership = pd.read_parquet(
        Path(args.root) / "unreliability_event_membership.parquet"
    )
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True)
    events["event_end"] = pd.to_datetime(events["event_end"], utc=True)
    ledger_path = _event_ledger_path(Path(args.root), args.ledger)
    states = _local_state_observations(states, ledger_path)
    states = _attach_event_membership(states, membership)
    features = _state_feature_columns(states)
    match_columns = _matching_columns(states)
    refined = _state_refined_event_membership(
        states, membership, match_columns, distance_threshold=5.0
    )
    refined.to_csv(output / "state_refined_event_membership.csv", index=False)
    diagnostics_parts: list[pd.DataFrame] = []
    matched_parts: list[pd.DataFrame] = []
    for (side, archetype), local in states.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        diagnostic, matched = matched_control_feature_diagnostics(
            local,
            features[: int(args.max_diagnostic_features)],
            match_columns,
            event_col="is_event",
            neighbors=3,
        )
        if not diagnostic.empty:
            diagnostic["side_name"] = side
            diagnostic["archetype_policy_key"] = archetype
            diagnostics_parts.append(diagnostic)
        if not matched.empty:
            matched["side_name"] = side
            matched["archetype_policy_key"] = archetype
            matched_parts.append(matched)
    diagnostics = (
        pd.concat(diagnostics_parts, ignore_index=True)
        if diagnostics_parts
        else pd.DataFrame()
    )
    matched = (
        pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    )
    diagnostics.to_csv(
        output / "all_events_matched_control_feature_diagnostics.csv", index=False
    )
    matched.to_parquet(output / "all_events_matched_pairs.parquet", index=False)

    quality = feature_quality_metrics(
        states,
        features,
        timestamp_col="__ts__",
        july_start="2026-07-01",
    )
    quality.to_csv(output / "state_feature_quality.csv", index=False)
    period_rows: list[dict[str, Any]] = []
    dossier_rows: list[pd.DataFrame] = []
    per_event_diagnostics: list[pd.DataFrame] = []
    priority_column = (
        "adverse_priority" if "adverse_priority" in events else "event_priority"
    )
    focus_events = events.sort_values(
        priority_column, ascending=False, kind="stable"
    ).head(int(args.top_events))
    for event in focus_events.itertuples(index=False):
        event_series = pd.Series(event._asdict())
        local_partition = states.loc[
            states["side_name"].eq(event_series["side_name"])
            & states["archetype_policy_key"].eq(event_series["archetype_policy_key"])
        ]
        period_rows.extend(_period_metrics(local_partition, event_series))
        left = event_series["event_start"] - pd.Timedelta(hours=24)
        right = event_series["event_end"] + pd.Timedelta(hours=48)
        dossier = local_partition.loc[
            local_partition["__ts__"].ge(left) & local_partition["__ts__"].le(right)
        ].copy()
        dossier["focus_event_id"] = event_series["event_id"]
        dossier["focus_event_start"] = event_series["event_start"]
        dossier["focus_event_end"] = event_series["event_end"]
        dossier_rows.append(dossier)

        local = local_partition.copy()
        local["is_focus_event"] = local["__ts__"].ge(
            event_series["event_start"]
        ) & local["__ts__"].lt(event_series["event_end"] + pd.Timedelta(days=1))
        event_diag, _ = matched_control_feature_diagnostics(
            local,
            features[: int(args.max_diagnostic_features)],
            match_columns,
            event_col="is_focus_event",
            neighbors=3,
        )
        if not event_diag.empty:
            event_diag["event_id"] = event_series["event_id"]
            per_event_diagnostics.append(event_diag)
    pd.DataFrame(period_rows).to_csv(
        output / "event_pre_during_post_metrics.csv", index=False
    )
    if dossier_rows:
        pd.concat(dossier_rows, ignore_index=True).to_parquet(
            output / "event_windows_24h_pre_48h_post.parquet",
            index=False,
            compression="zstd",
        )
    if per_event_diagnostics:
        per_event = pd.concat(per_event_diagnostics, ignore_index=True)
        per_event.to_csv(
            output / "per_event_matched_control_feature_diagnostics.csv", index=False
        )
        recurrence = _recurring_mechanism_feature_summary(per_event, events)
        recurrence.to_csv(
            output / "recurring_mechanism_feature_summary.csv", index=False
        )
    _write_json(
        output / "manifest.json",
        {
            "schema": "global_residual_event_dossiers_v1",
            "states": str(Path(args.states).resolve()),
            "ledger": str(ledger_path.resolve()),
            "events": int(len(events)),
            "dossier_events": int(min(len(events), args.top_events)),
            "state_features_assessed": len(features),
            "matched_control_features": match_columns,
            "state_refined_events": int(refined["state_refined_event_id"].nunique()),
            "matching_contract": (
                "Controls are non-event rows from the same side x archetype stream, matched on "
                "observable market return/volatility, score distribution, and selected activity."
            ),
            "outcome_columns_excluded_from_matching": sorted(OUTCOME_COLUMNS),
        },
    )


if __name__ == "__main__":
    main()
