#!/usr/bin/env python3
"""Materialize pooled historical regime-transition research data.

The market spine is deliberately broader than the exact model-health ledger.
It creates symmetric market transition targets over the complete compact
hourly history, then adds a separate native-hour economic failure overlay
where exact admitted-candidate outcomes are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_research import (  # noqa: E402
    TransitionResearchConfig,
    add_causal_transition_features,
    attach_pooled_states,
    discover_stabilized_transition_events,
    fit_pooled_state_geometry,
    load_compact_market_panel,
    materialize_event_snapshots,
    materialize_transition_labels,
)


DEFAULT_MARKET_SOURCE = Path(
    "data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"
)
DEFAULT_HISTORICAL_MEMBERSHIP = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_historical_20260726_v12/"
    "candidate_membership_expost.parquet"
)
DEFAULT_CURRENT_MEMBERSHIP = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_20260726_v6/"
    "candidate_membership_expost.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v3"
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _forward_rolling(
    values: pd.Series,
    window: int,
    *,
    operation: str,
) -> pd.Series:
    reversed_values = values.iloc[::-1]
    rolling = reversed_values.rolling(window, min_periods=window)
    if operation == "mean":
        result = rolling.mean()
    elif operation == "sum":
        result = rolling.sum()
    elif operation == "max":
        result = rolling.max()
    else:
        raise ValueError(operation)
    return result.iloc[::-1]


def materialize_native_hourly_economics(
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create true hourly pre/post economics and failure anchors.

    There is no six-hour-bin expansion.  The origin cohort is ``[-12h,0)``
    and the adverse/destination cohort is ``[0,+12h)``.  Quantiles are pooled
    within each model generation because this is explicitly non-walk-forward
    research.
    """

    parts: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        part = pd.read_parquet(path)
        part["source_artifact"] = str(path)
        parts.append(part)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    rows["execution_decision_utc"] = pd.to_datetime(
        rows["execution_decision_utc"], utc=True, errors="coerce"
    ).dt.floor("h")
    rows["execution_label_end_utc"] = pd.to_datetime(
        rows["execution_label_end_utc"], utc=True, errors="coerce"
    )
    for name in ("mapped_score", "expost__net_ev", "expost__gross_ev"):
        rows[name] = pd.to_numeric(rows[name], errors="coerce")
    rows = rows.loc[
        rows["execution_decision_utc"].notna()
        & rows["execution_label_end_utc"].notna()
    ].copy()
    rows["evaluation_origin"] = rows["evaluation_origin"].astype(str)
    rows["admitted"] = rows["admitted"].fillna(False).astype(bool)
    records: list[dict[str, object]] = []
    for (origin, decision), local in rows.groupby(
        ["evaluation_origin", "execution_decision_utc"],
        sort=True,
        observed=True,
    ):
        admitted = local.loc[local["admitted"]]
        records.append(
            {
                "evaluation_origin": origin,
                "execution_decision_utc": decision,
                "source_utc": decision - pd.Timedelta(hours=1),
                "candidate_rows": int(len(local)),
                "admitted_rows": int(len(admitted)),
                "mapped_score_mean": float(admitted["mapped_score"].mean()),
                "net_ev_mean": float(admitted["expost__net_ev"].mean()),
                "gross_ev_mean": float(admitted["expost__gross_ev"].mean()),
                "economic_residual_mean": float(
                    (admitted["expost__net_ev"] - admitted["mapped_score"]).mean()
                ),
                "positive_net_rate": float(admitted["expost__net_ev"].gt(0).mean()),
                "outcome_available_utc": admitted[
                    "execution_label_end_utc"
                ].max()
                if len(admitted)
                else local["execution_label_end_utc"].max(),
            }
        )
    hourly = pd.DataFrame.from_records(records)
    enriched: list[pd.DataFrame] = []
    failure_events: list[dict[str, object]] = []
    for origin, local in hourly.groupby(
        "evaluation_origin", sort=True, observed=True
    ):
        local = local.sort_values("execution_decision_utc", kind="stable").copy()
        complete = pd.date_range(
            local["execution_decision_utc"].min(),
            local["execution_decision_utc"].max(),
            freq="h",
            tz="UTC",
        )
        local = local.set_index("execution_decision_utc").reindex(complete)
        local.index.name = "execution_decision_utc"
        local["evaluation_origin"] = origin
        local["source_utc"] = local.index - pd.Timedelta(hours=1)
        gap = local["candidate_rows"].isna()
        local["economic_segment_id"] = gap.cumsum()
        for _, segment in local.loc[~gap].groupby(
            "economic_segment_id", sort=False
        ):
            index = segment.index
            pre_net = segment["net_ev_mean"].rolling(
                12, min_periods=12
            ).mean().shift(1)
            post_net = _forward_rolling(segment["net_ev_mean"], 12, operation="mean")
            pre_residual = segment["economic_residual_mean"].rolling(
                12, min_periods=12
            ).mean().shift(1)
            post_residual = _forward_rolling(
                segment["economic_residual_mean"], 12, operation="mean"
            )
            post_support = _forward_rolling(
                segment["admitted_rows"], 12, operation="sum"
            )
            post_available = _forward_rolling(
                segment["outcome_available_utc"].astype("int64").astype(float),
                12,
                operation="max",
            )
            local.loc[index, "pre_12h_net_ev_mean"] = pre_net
            local.loc[index, "post_12h_net_ev_mean"] = post_net
            local.loc[index, "pre_12h_residual_mean"] = pre_residual
            local.loc[index, "post_12h_residual_mean"] = post_residual
            local.loc[index, "post_12h_admitted_rows"] = post_support
            local.loc[index, "target_available_utc"] = pd.to_datetime(
                post_available, utc=True, errors="coerce"
            )
        local["post_minus_pre_net_ev"] = (
            local["post_12h_net_ev_mean"] - local["pre_12h_net_ev_mean"]
        )
        local["post_minus_pre_residual"] = (
            local["post_12h_residual_mean"]
            - local["pre_12h_residual_mean"]
        )
        valid_shift = local["post_minus_pre_residual"].dropna()
        median = float(valid_shift.median()) if len(valid_shift) else np.nan
        mad = (
            float((valid_shift - median).abs().median()) * 1.4826
            if len(valid_shift)
            else np.nan
        )
        local["post_minus_pre_residual_rz"] = (
            (local["post_minus_pre_residual"] - median) / (mad + 1e-8)
        ).clip(-12, 12)
        raw_failure = (
            local["post_12h_net_ev_mean"].lt(0)
            & local["post_minus_pre_residual_rz"].le(-1.0)
            & local["post_12h_admitted_rows"].ge(20)
        )
        # A failure must persist in at least two of the next three hourly
        # anchor evaluations.  This suppresses single-hour cohort noise.
        persistent = (
            raw_failure.astype(int)
            .iloc[::-1]
            .rolling(3, min_periods=3)
            .sum()
            .iloc[::-1]
            .ge(2)
        )
        local["target__economic_failure_active"] = persistent.astype(np.int8)
        onset = persistent & ~persistent.shift(1, fill_value=False)
        anchors = np.flatnonzero(onset.to_numpy())
        last = -100
        for position in anchors:
            if position - last < 12:
                continue
            stamp = local.index[position]
            failure_events.append(
                {
                    "economic_event_id": (
                        f"economic_failure_{origin}_{stamp.strftime('%Y%m%d%H')}"
                    ),
                    "evaluation_origin": origin,
                    "anchor_decision_utc": stamp,
                    "anchor_source_utc": stamp - pd.Timedelta(hours=1),
                    "target_available_utc": local.iloc[position][
                        "target_available_utc"
                    ],
                    "pre_12h_net_ev_mean": local.iloc[position][
                        "pre_12h_net_ev_mean"
                    ],
                    "post_12h_net_ev_mean": local.iloc[position][
                        "post_12h_net_ev_mean"
                    ],
                    "post_minus_pre_residual_rz": local.iloc[position][
                        "post_minus_pre_residual_rz"
                    ],
                    "post_12h_admitted_rows": local.iloc[position][
                        "post_12h_admitted_rows"
                    ],
                    "label_contract": (
                        "native hourly; pre[-12h,0); post[0,+12h); "
                        "negative post EV; generation-pooled robust residual shift"
                    ),
                }
            )
            last = position
        enriched.append(local.reset_index())
    return (
        pd.concat(enriched, ignore_index=True),
        pd.DataFrame.from_records(failure_events),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-source", type=Path, default=DEFAULT_MARKET_SOURCE)
    parser.add_argument(
        "--historical-membership",
        type=Path,
        default=DEFAULT_HISTORICAL_MEMBERSHIP,
    )
    parser.add_argument(
        "--current-membership", type=Path, default=DEFAULT_CURRENT_MEMBERSHIP
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2023-01-01T00:00:00Z")
    parser.add_argument("--end", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = TransitionResearchConfig()
    panel = load_compact_market_panel(
        args.market_source,
        start=args.start,
        end=args.end,
        minimum_feature_coverage=config.minimum_feature_coverage,
    )
    panel, transition_columns = add_causal_transition_features(panel)
    level_columns = [
        name
        for name in panel.columns
        if name
        not in {"source_utc", "execution_decision_utc", "segment_id"}
        and not name.startswith(("mkt_regime_change__", "transition_new__"))
        and pd.api.types.is_numeric_dtype(panel[name])
    ]
    geometry = fit_pooled_state_geometry(
        panel, feature_columns=level_columns, config=config
    )
    panel = attach_pooled_states(panel, geometry)
    events = discover_stabilized_transition_events(panel, config=config)
    labels = materialize_transition_labels(panel, events)
    snapshots = materialize_event_snapshots(panel, events)
    hourly_economics, economic_events = materialize_native_hourly_economics(
        [args.historical_membership, args.current_membership]
    )
    if len(economic_events) and len(events):
        economic_stamps = pd.DatetimeIndex(economic_events["anchor_source_utc"])
        nearest_ids: list[str | None] = []
        nearest_distance: list[float] = []
        for stamp in pd.DatetimeIndex(events["anchor_source_utc"]):
            distance = np.abs(
                (economic_stamps - stamp) / pd.Timedelta(hours=1)
            )
            position = int(np.argmin(distance))
            if float(distance[position]) <= 6:
                nearest_ids.append(
                    str(economic_events.iloc[position]["economic_event_id"])
                )
                nearest_distance.append(float(distance[position]))
            else:
                nearest_ids.append(None)
                nearest_distance.append(np.nan)
        events["economic_failure_event_within_6h"] = nearest_ids
        events["economic_failure_distance_hours"] = nearest_distance
    labels.to_parquet(output / "hourly_transition_dataset.parquet", index=False)
    events.to_parquet(output / "transition_events.parquet", index=False)
    snapshots.to_parquet(output / "transition_event_snapshots.parquet", index=False)
    geometry.selection.to_csv(output / "state_cluster_selection.csv", index=False)
    joblib.dump(geometry, output / "pooled_state_geometry.joblib")
    hourly_economics.to_parquet(
        output / "native_hourly_economics.parquet", index=False
    )
    economic_events.to_parquet(
        output / "economic_failure_events.parquet", index=False
    )
    report = {
        "schema": "pooled_symmetric_regime_transition_research_v1",
        "research_only": True,
        "walk_forward_required": False,
        "promotion_evidence": False,
        "market_source": str(args.market_source),
        "market_start": labels["source_utc"].min(),
        "market_end": labels["source_utc"].max(),
        "hourly_rows": len(labels),
        "segments": int(labels["segment_id"].nunique()),
        "retained_market_features": len(level_columns),
        "existing_and_new_transition_features": len(transition_columns),
        "selected_states": int(geometry.cluster.n_clusters),
        "transition_events": len(events),
        "transition_event_rate_per_30d": (
            float(len(events) / len(labels) * 24 * 30) if len(labels) else np.nan
        ),
        "economic_hourly_rows": len(hourly_economics),
        "economic_failure_events": len(economic_events),
        "economic_linked_transition_events": int(
            events.get(
                "economic_failure_event_within_6h",
                pd.Series(dtype=object),
            )
            .notna()
            .sum()
        ),
        "label_contract": {
            "origin": "[-12h,-3h)",
            "approach": "[-12h,-6h)",
            "acceleration": "[-6h,-3h)",
            "immediate_lead": "[-3h,0h)",
            "active": "[0h, first 3h-persistent destination), capped at +6h",
            "early_destination": "[transition_end,+6h)",
            "settled_destination": "[+6h,+12h)",
            "exact_snapshots": [-48, -24, -12, -6, -3, 0, 3, 6, 12],
        },
        "validation_contract": (
            "pooled research; use event/control-block grouped validation; "
            "never random-row validation"
        ),
    }
    _write_json(output / "manifest.json", report)
    return report


def main() -> None:
    report = run(_parser().parse_args())
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
