#!/usr/bin/env python3
"""Materialize compact exact-policy health and economic-failure labels.

This is an older canonical raw-alpha lineage, not a backfill of the current
execution-EV lineage.  Selection is one pooled global top 10% over the declared
research interval.  Health fields are decision-time score/context summaries
plus strictly matured outcomes.  Failure labels use exact-policy economics with
pre[-12h,0) and post[0,+12h) windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


HEALTH_COLUMNS = (
    "health__candidate_rows",
    "health__distinct_assets",
    "health__long_share",
    "health__candidate_rows_delta_24h",
    "health__raw_score_mean",
    "health__raw_score_std",
    "health__raw_score_p90",
    "health__raw_score_long_minus_short",
    "health__mapped_net_mean",
    "health__mapped_net_std",
    "health__mapped_net_p90",
    "health__mapped_net_long_minus_short",
    "health__causal_percentile_mean",
    "health__causal_percentile_std",
    "health__causal_percentile_entropy",
    "health__raw_mapped_rank_spearman",
    "health__raw_mapped_rank_abs_gap",
    "health__map_reference_log1p_mean",
    "health__low_map_support_share",
    "health__selected_rows",
    "health__selected_symbol_hhi",
    "health__selected_long_share",
    "health__recent_resolved_net_ev_hl3d",
    "health__recent_resolved_hit_rate_hl3d",
    "health__recent_resolved_mapping_error_hl3d",
    "health__recent_resolved_cost_bps_hl3d",
    "health__recent_resolved_full_stop_rate_hl3d",
    "health__recent_resolved_effective_rows_hl3d",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_global_top_k(
    frame: pd.DataFrame, *, score_column: str, fraction: float
) -> pd.DataFrame:
    count = max(1, int(math.ceil(float(fraction) * len(frame))))
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    order = np.lexsort(
        (frame["candidate_id"].astype(str).to_numpy(), -score)
    )
    return frame.iloc[order[:count]].copy()


def _percentile_entropy(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return np.nan
    count, _ = np.histogram(
        numeric, bins=np.linspace(0.0, 1.0, 11)
    )
    probability = count[count > 0] / count.sum()
    return float(-(probability * np.log(probability)).sum())


def _rank_relationship(group: pd.DataFrame) -> pd.Series:
    local = group[["score_raw", "mapped_direct_net"]].dropna()
    if (
        len(local) < 3
        or local["score_raw"].nunique() < 2
        or local["mapped_direct_net"].nunique() < 2
    ):
        return pd.Series(
            {
                "health__raw_mapped_rank_spearman": np.nan,
                "health__raw_mapped_rank_abs_gap": np.nan,
            }
        )
    rank = local.rank(method="average", pct=True)
    return pd.Series(
        {
            "health__raw_mapped_rank_spearman": float(
                rank["score_raw"].corr(rank["mapped_direct_net"])
            ),
            "health__raw_mapped_rank_abs_gap": float(
                (rank["score_raw"] - rank["mapped_direct_net"]).abs().mean()
            ),
        }
    )


def _selected_symbol_hhi(symbols: pd.Series) -> float:
    share = symbols.astype(str).value_counts(normalize=True)
    return float(np.square(share).sum()) if len(share) else np.nan


def causal_resolved_health(
    selected: pd.DataFrame, decisions: pd.Series
) -> pd.DataFrame:
    outcome = pd.DataFrame(
        {
            "resolved_at": pd.to_datetime(
                selected["effective_label_resolution_utc"],
                utc=True,
                errors="raise",
            ),
            "net": pd.to_numeric(
                selected["execution_net_ev_12h"], errors="coerce"
            ),
            "hit": pd.to_numeric(
                selected["execution_net_ev_12h"], errors="coerce"
            ).gt(0.0).astype(float),
            "mapping_error": (
                pd.to_numeric(
                    selected["execution_net_ev_12h"], errors="coerce"
                )
                - pd.to_numeric(selected["mapped_direct_net"], errors="coerce")
            ),
            "cost_bps": (
                pd.to_numeric(
                    selected["execution_cost_return"], errors="coerce"
                )
                * 10_000.0
            ),
            "full_stop": selected["execution_exit_class"]
            .astype(str)
            .eq("full_stop")
            .astype(float),
        }
    ).dropna(subset=["resolved_at", "net"])
    grouped = outcome.groupby("resolved_at", sort=True).agg(
        net_sum=("net", "sum"),
        hit_sum=("hit", "sum"),
        mapping_error_sum=("mapping_error", "sum"),
        cost_sum=("cost_bps", "sum"),
        stop_sum=("full_stop", "sum"),
        rows=("net", "size"),
    )
    decision_index = pd.DatetimeIndex(
        pd.to_datetime(decisions, utc=True, errors="raise").sort_values().unique()
    )
    start = min(decision_index.min(), grouped.index.min()).floor("h")
    end = max(decision_index.max(), grouped.index.max()).ceil("h")
    rate = float(np.exp(-np.log(2.0) / 72.0))
    sums = np.zeros(5, dtype=float)
    weight = 0.0
    records: dict[pd.Timestamp, tuple[float, ...]] = {}
    for stamp in pd.date_range(start, end, freq="h", tz="UTC"):
        sums *= rate
        weight *= rate
        if stamp in decision_index:
            values = (
                tuple(sums / weight)
                if weight > 0.0
                else (np.nan,) * 5
            )
            records[stamp] = (*values, weight)
        # Incorporate after recording: labels resolving exactly at the
        # decision timestamp are not yet available.
        if stamp in grouped.index:
            local = grouped.loc[stamp]
            sums += local[
                [
                    "net_sum",
                    "hit_sum",
                    "mapping_error_sum",
                    "cost_sum",
                    "stop_sum",
                ]
            ].to_numpy(float)
            weight += float(local["rows"])
    return pd.DataFrame.from_dict(
        records,
        orient="index",
        columns=HEALTH_COLUMNS[-6:],
    ).rename_axis("execution_decision_utc").reset_index()


def _forward_rolling(values: pd.Series, window: int, operation: str) -> pd.Series:
    rolling = values.iloc[::-1].rolling(window, min_periods=window)
    if operation == "mean":
        result = rolling.mean()
    elif operation == "sum":
        result = rolling.sum()
    elif operation == "max":
        result = rolling.max()
    else:
        raise ValueError(operation)
    return result.iloc[::-1]


def _assign_episode_ids(
    active: pd.Series,
    timestamps: pd.DatetimeIndex,
    *,
    label: str,
    merge_gap_hours: int = 6,
) -> tuple[pd.Series, pd.DataFrame]:
    output = pd.Series(index=active.index, dtype=object)
    event_rows: list[dict[str, Any]] = []
    last_active_position: int | None = None
    event_id: str | None = None
    for position, flag in enumerate(active.fillna(False).to_numpy(bool)):
        if not flag:
            continue
        if (
            last_active_position is None
            or position - last_active_position > int(merge_gap_hours)
        ):
            event_id = (
                f"exact_failure_{label}_{timestamps[position].strftime('%Y%m%d%H')}"
            )
            event_rows.append(
                {
                    "failure_label": label,
                    "economic_event_id": event_id,
                    "anchor_source_utc": timestamps[position],
                    "anchor_decision_utc": (
                        timestamps[position] + pd.Timedelta(hours=1)
                    ),
                }
            )
        output.iloc[position] = event_id
        last_active_position = position
    return output, pd.DataFrame(event_rows)


def add_failure_labels(
    hourly_economics: pd.DataFrame,
    *,
    thresholds: Mapping[str, float],
    selection_contract: str = "one pooled global top10 score book",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = hourly_economics.copy()
    local["source_utc"] = pd.to_datetime(
        local["source_utc"], utc=True, errors="raise"
    )
    if local["source_utc"].duplicated().any():
        raise ValueError("failure labels require one row per source hour")
    if not local["source_utc"].dt.floor("h").eq(local["source_utc"]).all():
        raise ValueError("failure labels require hour-aligned source timestamps")
    local = local.sort_values(
        "source_utc", kind="stable"
    ).set_index("source_utc")
    observed_hours = pd.DatetimeIndex(local.index)
    full_hours = pd.date_range(
        observed_hours.min(), observed_hours.max(), freq="h", tz="UTC"
    )
    # Window sizes are elapsed calendar hours, not a count of observed rows.
    # Missing source hours stay missing and conservatively invalidate any
    # pre/post window that crosses them.
    local = local.reindex(full_hours)
    local.index.name = "source_utc"
    local["__observed_source_hour"] = local.index.isin(observed_hours)
    local["health__selected_rows"] = pd.to_numeric(
        local["health__selected_rows"], errors="coerce"
    ).fillna(0.0)
    for sum_column, mean_column in (
        ("realized_net_sum", "realized_net_mean"),
        ("mapping_residual_sum", "mapping_residual_mean"),
    ):
        if sum_column not in local:
            local[sum_column] = (
                pd.to_numeric(local[mean_column], errors="coerce")
                * local["health__selected_rows"]
            )
        local[sum_column] = pd.to_numeric(
            local[sum_column], errors="coerce"
        ).fillna(0.0)
    local["pre_12h_source_hours"] = (
        local["__observed_source_hour"]
        .astype(float)
        .rolling(12, min_periods=12)
        .sum()
        .shift(1)
    )
    local["post_12h_source_hours"] = _forward_rolling(
        local["__observed_source_hour"].astype(float), 12, "sum"
    )
    local["pre_12h_selected_rows"] = (
        local["health__selected_rows"]
        .rolling(12, min_periods=12)
        .sum()
        .shift(1)
    )
    local["post_12h_selected_rows"] = _forward_rolling(
        local["health__selected_rows"], 12, "sum"
    )
    for prefix, sum_column in (
        ("net_ev", "realized_net_sum"),
        ("mapping_residual", "mapping_residual_sum"),
    ):
        pre_sum = (
            local[sum_column]
            .rolling(12, min_periods=12)
            .sum()
            .shift(1)
        )
        post_sum = _forward_rolling(local[sum_column], 12, "sum")
        local[f"pre_12h_{prefix}_mean"] = (
            pre_sum / local["pre_12h_selected_rows"].replace(0.0, np.nan)
        )
        local[f"post_12h_{prefix}_mean"] = (
            post_sum / local["post_12h_selected_rows"].replace(0.0, np.nan)
        )
        local.loc[
            local["pre_12h_source_hours"].ne(12.0),
            f"pre_12h_{prefix}_mean",
        ] = np.nan
        local.loc[
            local["post_12h_source_hours"].ne(12.0),
            f"post_12h_{prefix}_mean",
        ] = np.nan
    outcome_available = pd.to_datetime(
        local["outcome_available_utc"], utc=True, errors="coerce"
    )
    no_selected = local["health__selected_rows"].eq(0.0)
    outcome_available.loc[no_selected] = (
        local.index[no_selected] + pd.Timedelta(hours=1)
    )
    outcome_available_ns = pd.Series(
        outcome_available.astype("int64").astype(float),
        index=local.index,
    )
    outcome_available_ns.loc[outcome_available.isna()] = np.nan
    local["target_available_utc"] = pd.to_datetime(
        _forward_rolling(outcome_available_ns, 12, "max"),
        utc=True,
        errors="coerce",
    )
    local["post_minus_pre_mapping_residual"] = (
        local["post_12h_mapping_residual_mean"]
        - local["pre_12h_mapping_residual_mean"]
    )
    shift = local["post_minus_pre_mapping_residual"].dropna()
    if shift.empty:
        raise ValueError("no complete exact 12-hour failure-label windows")
    median = float(shift.median())
    mad = float((shift - median).abs().median()) * 1.4826
    local["post_minus_pre_mapping_residual_rz"] = (
        (local["post_minus_pre_mapping_residual"] - median) / (mad + 1e-8)
    ).clip(-12, 12)
    event_frames: list[pd.DataFrame] = []
    local["label_window_complete"] = (
        local["pre_12h_source_hours"].eq(12.0)
        & local["post_12h_source_hours"].eq(12.0)
        & local["pre_12h_selected_rows"].gt(0.0)
        & local["post_12h_selected_rows"].gt(0.0)
        & local["pre_12h_net_ev_mean"].notna()
        & local["post_12h_net_ev_mean"].notna()
        & local["pre_12h_mapping_residual_mean"].notna()
        & local["post_12h_mapping_residual_mean"].notna()
        & local["target_available_utc"].notna()
    )
    for label, threshold in thresholds.items():
        raw = (
            local["label_window_complete"]
            & local["post_12h_net_ev_mean"].lt(0.0)
            & local["post_minus_pre_mapping_residual_rz"].le(float(threshold))
            & local["post_12h_selected_rows"].ge(20)
        )
        persistent = (
            raw.astype(int)
            .iloc[::-1]
            .rolling(3, min_periods=3)
            .sum()
            .iloc[::-1]
            .ge(2)
        )
        target_column = f"target__economic_failure_{label}_active"
        event_column = f"target__economic_failure_{label}_event_id"
        local[target_column] = persistent.astype(np.int8)
        event_ids, events = _assign_episode_ids(
            persistent, local.index, label=label
        )
        local[event_column] = event_ids
        if len(events):
            indexed = local.reset_index().set_index(event_column)
            events = events.set_index("economic_event_id")
            for column in (
                "pre_12h_net_ev_mean",
                "post_12h_net_ev_mean",
                "post_minus_pre_mapping_residual_rz",
                "post_12h_selected_rows",
                "target_available_utc",
            ):
                events[column] = indexed.groupby(level=0)[column].first()
            events["threshold_rz"] = float(threshold)
            events["label_contract"] = (
                f"{selection_contract}; exact-policy economics; "
                "candidate-row-weighted exact-hour pre[-12h,0); "
                "post[0,+12h); negative post net; "
                "2-of-next-3 persistence"
            )
            event_frames.append(events.reset_index())
    events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )
    local = local.loc[local["__observed_source_hour"]].drop(
        columns="__observed_source_hour"
    )
    return local.reset_index(), events


def build_health_and_labels(
    candidates: pd.DataFrame,
    *,
    top_k_fraction: float,
    low_support_rows: int,
    selection_score_column: str = "score_raw",
    selection_contract: str = "one pooled global top10 raw-alpha book",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = candidates.loc[candidates["mapped_eligible"].astype(bool)].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["execution_decision_utc"] = pd.to_datetime(
        work["execution_decision_utc"], utc=True, errors="raise"
    )
    work["effective_label_resolution_utc"] = pd.to_datetime(
        work["effective_label_resolution_utc"], utc=True, errors="raise"
    )
    selected = stable_global_top_k(
        work,
        score_column=selection_score_column,
        fraction=top_k_fraction,
    )
    grouped = work.groupby("__ts__", observed=True, sort=True)
    hourly = grouped.agg(
        health__candidate_rows=("candidate_id", "size"),
        health__distinct_assets=("__symbol__", "nunique"),
        health__long_share=(
            "side_name",
            lambda value: float(value.astype(str).eq("long").mean()),
        ),
        health__raw_score_mean=("score_raw", "mean"),
        health__raw_score_std=("score_raw", "std"),
        health__raw_score_p90=("score_raw", lambda value: value.quantile(0.90)),
        health__mapped_net_mean=("mapped_direct_net", "mean"),
        health__mapped_net_std=("mapped_direct_net", "std"),
        health__mapped_net_p90=(
            "mapped_direct_net",
            lambda value: value.quantile(0.90),
        ),
        health__causal_percentile_mean=("causal_score_percentile", "mean"),
        health__causal_percentile_std=("causal_score_percentile", "std"),
        health__causal_percentile_entropy=(
            "causal_score_percentile",
            _percentile_entropy,
        ),
        health__map_reference_log1p_mean=(
            "map_reference_rows",
            lambda value: float(np.log1p(value).mean()),
        ),
        health__low_map_support_share=(
            "map_reference_rows",
            lambda value: float(
                pd.to_numeric(value, errors="coerce")
                .lt(int(low_support_rows))
                .mean()
            ),
        ),
    )
    side = work.groupby(["__ts__", "side_name"], observed=True).agg(
        raw=("score_raw", "mean"),
        mapped=("mapped_direct_net", "mean"),
    ).unstack("side_name")
    for metric, name in (
        ("raw", "health__raw_score_long_minus_short"),
        ("mapped", "health__mapped_net_long_minus_short"),
    ):
        long = side.get((metric, "long"), pd.Series(index=hourly.index, dtype=float))
        short = side.get((metric, "short"), pd.Series(index=hourly.index, dtype=float))
        hourly[name] = long - short
    hourly = hourly.join(grouped.apply(_rank_relationship, include_groups=False))
    selected_hourly = selected.groupby("__ts__", observed=True, sort=True).agg(
        health__selected_rows=("candidate_id", "size"),
        health__selected_symbol_hhi=("__symbol__", _selected_symbol_hhi),
        health__selected_long_share=(
            "side_name",
            lambda value: float(value.astype(str).eq("long").mean()),
        ),
        realized_net_mean=("execution_net_ev_12h", "mean"),
        realized_net_sum=("execution_net_ev_12h", "sum"),
        realized_gross_mean=("execution_gross_ev_12h", "mean"),
        expected_mapped_net_mean=("mapped_direct_net", "mean"),
        expected_mapped_net_sum=("mapped_direct_net", "sum"),
        selected_positive_rate=(
            "execution_net_ev_12h",
            lambda value: float(value.gt(0.0).mean()),
        ),
        selected_full_stop_rate=(
            "execution_exit_class",
            lambda value: float(value.astype(str).eq("full_stop").mean()),
        ),
        outcome_available_utc=(
            "effective_label_resolution_utc",
            "max",
        ),
    )
    selected_hourly["mapping_residual_mean"] = (
        selected_hourly["realized_net_mean"]
        - selected_hourly["expected_mapped_net_mean"]
    )
    selected_hourly["mapping_residual_sum"] = (
        selected_hourly["realized_net_sum"]
        - selected_hourly["expected_mapped_net_sum"]
    )
    hourly = hourly.join(selected_hourly)
    hourly = hourly.reset_index().rename(columns={"__ts__": "source_utc"})
    hourly["execution_decision_utc"] = (
        hourly["source_utc"] + pd.Timedelta(hours=1)
    )
    hourly["health__candidate_rows_delta_24h"] = hourly[
        "health__candidate_rows"
    ].diff(24)
    resolved = causal_resolved_health(
        selected, hourly["execution_decision_utc"]
    )
    hourly = hourly.merge(
        resolved,
        on="execution_decision_utc",
        how="left",
        validate="one_to_one",
    )
    hourly, events = add_failure_labels(
        hourly,
        thresholds={"broad": -0.5, "strict": -1.0},
        selection_contract=selection_contract,
    )
    missing = sorted(set(HEALTH_COLUMNS).difference(hourly.columns))
    if missing:
        raise AssertionError(f"missing health fields: {missing}")
    return hourly, events, selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.candidates)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    hourly, events, selected = build_health_and_labels(
        pd.read_parquet(source),
        top_k_fraction=float(args.top_k_fraction),
        low_support_rows=int(args.low_support_rows),
    )
    output.mkdir(parents=True, exist_ok=False)
    hourly_path = output / "hourly_exact_model_health_and_failure_labels.parquet"
    event_path = output / "economic_failure_events.parquet"
    selected_path = output / "frozen_global_top10_candidates.parquet"
    feature_path = output / "health_feature_catalog.csv"
    hourly.to_parquet(hourly_path, index=False, compression="zstd")
    events.to_parquet(event_path, index=False, compression="zstd")
    selected.to_parquet(selected_path, index=False, compression="zstd")
    pd.DataFrame({"feature": HEALTH_COLUMNS}).to_csv(feature_path, index=False)
    manifest = {
        "schema": "historical_exact_model_health_failure_v1",
        "status": "RESEARCH_ONLY_CANONICAL_RAW_ALPHA_LINEAGE_COMPLETE",
        "current_lineage": False,
        "lineage_disclosure": (
            "canonical February-April 2025 raw-alpha/exact-policy lineage; "
            "never substitute for May-July 2026 current execution-EV health"
        ),
        "selection_contract": (
            "one pooled global top 10% by raw score with candidate-ID tie break; "
            "never per timestamp or side"
        ),
        "label_contract": (
            "exact hourly calendar pre[-12h,0) versus post[0,+12h); windows "
            "crossing a missing source hour are ineligible; candidate-row-"
            "weighted exact-policy net and causal mapped residual; "
            "2-of-next-3 persistence; active episodes merged across gaps no "
            "longer than six hours; target available after all post-window "
            "outcomes"
        ),
        "rows": int(len(hourly)),
        "selected_candidates": int(len(selected)),
        "health_feature_count": len(HEALTH_COLUMNS),
        "failure_events": {
            label: int(
                events.loc[events["failure_label"].eq(label), "economic_event_id"]
                .nunique()
            )
            for label in ("broad", "strict")
        },
        "sources": {
            "candidates": {"path": str(source), "sha256": _sha256(source)}
        },
        "outputs": {},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    outputs = {
        "hourly": hourly_path,
        "events": event_path,
        "selected_candidates": selected_path,
        "health_features": feature_path,
    }
    manifest["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in outputs.items()
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path("/Users/remyroche/Documents/Ares")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/historical_causal_score_economics_mapping_20260729_v1/"
            "canonical_base__score_base_alpha/causal_mapped_candidates.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--low-support-rows", type=int, default=100)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
