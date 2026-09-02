#!/usr/bin/env python3
"""Materialize a hash-bound cross-era exact 12h payoff research dataset.

The artifact combines the frozen February--April 2025 top-40 candidate
context with the May--July 2026 capture universe.  Only the common frozen
256-column pre-entry feature contract is retained.  Exact policy economics
and path-event labels are joined after candidate identities are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SCHEMA = "cross_era_tail_payoff_dataset_v3"
GRID = "h12_u1p5atr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_many(paths: Iterable[Path], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in sorted(paths)]
    if not frames:
        raise ValueError("no parquet shards found")
    return pd.concat(frames, ignore_index=True)


def normalize_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the documented slash/underscore symbol storage difference."""
    result = frame.copy()
    result["__symbol__"] = (
        result["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    )
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    return result


def validate_feature_contract(contract: dict[str, object]) -> list[str]:
    features = list(contract["feature_columns"])
    if len(features) != 256 or len(features) != len(set(features)):
        raise ValueError("frozen feature contract must contain 256 unique fields")
    return features


def derive_event_columns(
    frame: pd.DataFrame, event_column: str
) -> pd.DataFrame:
    result = frame.copy()
    event = result[event_column].astype(str)
    allowed = {"favorable_first", "adverse_first_or_conflict", "timeout"}
    unexpected = set(event.unique()) - allowed
    if unexpected:
        raise ValueError(f"unexpected event labels: {sorted(unexpected)}")
    result["clean_first"] = event.eq("favorable_first").astype(np.int8)
    result["adverse_first"] = event.eq("adverse_first_or_conflict").astype(np.int8)
    result["timeout_event"] = event.eq("timeout").astype(np.int8)
    if not (
        result[["clean_first", "adverse_first", "timeout_event"]]
        .sum(axis=1)
        .eq(1)
        .all()
    ):
        raise AssertionError("competing-risk events must be mutually exclusive")
    net = pd.to_numeric(result["execution_net_ev_12h"], errors="raise")
    result["positive_net"] = net.gt(0).astype(np.int8)
    result["negative_net"] = net.le(0).astype(np.int8)
    result["event_class"] = event
    return result.drop(columns=[event_column])


def add_candidate_relative_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    group = result.groupby(["__ts__", "side_name"], observed=True)
    score = pd.to_numeric(result["base_oof_score"], errors="coerce")
    result["candidate_group_size"] = group["candidate_id"].transform("size").astype(
        np.int16
    )
    result["base_rank_timestamp_side"] = group["base_oof_score"].rank(
        method="average", ascending=False
    )
    result["base_rank_pct_timestamp_side"] = (
        result["base_rank_timestamp_side"] - 1.0
    ) / np.maximum(result["candidate_group_size"] - 1, 1)
    mean = group["base_oof_score"].transform("mean")
    std = group["base_oof_score"].transform("std").replace(0, np.nan)
    result["base_score_z_timestamp_side"] = ((score - mean) / std).fillna(0.0)
    cutoff = group["base_oof_score"].transform("min")
    result["base_margin_to_candidate_cutoff"] = score - cutoff
    return result


def materialize_old(
    context_dir: Path,
    path_label_dir: Path,
    exact_ev_path: Path,
    features: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    context_paths = list((context_dir / "shards").glob("*.parquet"))
    context_columns = [*IDENTITY, "base_oof_score", *features]
    context = normalize_identity(_read_many(context_paths, context_columns))
    label_paths = list((path_label_dir / "shards").glob("part-*/labels.parquet"))
    label_columns = [
        *IDENTITY,
        "__label_available_at__",
        "__soft_tb_first_event__",
        "__soft_tb_upper_hit_12h__",
        "__soft_tb_lower_hit_12h__",
        "__meaningful_mfe_reached_12h__",
        "__peak_mfe_atr_12h__",
        "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_meaningful_mfe_atr_12h__",
        "__future_slope_atr_per_hour_12h__",
    ]
    labels = normalize_identity(_read_many(label_paths, label_columns))
    economics = normalize_identity(
        pd.read_parquet(
            exact_ev_path,
            columns=[
                *IDENTITY,
                "execution_gross_ev_12h",
                "execution_cost_return",
                "execution_net_ev_12h",
            ],
        )
    )
    joined = (
        context.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
        .merge(economics, on=list(IDENTITY), how="inner", validate="one_to_one")
    )
    if len(joined) != len(context):
        raise AssertionError(
            f"older exact join lost rows: {len(context)} -> {len(joined)}"
        )
    joined = joined.rename(
        columns={
            "__label_available_at__": "label_resolution_utc",
            "__soft_tb_first_event__": "_event",
            "__soft_tb_upper_hit_12h__": "soft_upper_hit",
            "__soft_tb_lower_hit_12h__": "soft_lower_hit",
            "__meaningful_mfe_reached_12h__": "meaningful_mfe_reached",
            "__peak_mfe_atr_12h__": "peak_mfe_atr",
            "__time_to_first_meaningful_mfe_hours_12h__":
                "time_to_meaningful_mfe_hours",
            "__mae_before_meaningful_mfe_atr_12h__":
                "mae_before_meaningful_mfe_atr",
            "__future_slope_atr_per_hour_12h__": "future_slope_atr_per_hour",
        }
    )
    joined = derive_event_columns(joined, "_event")
    joined["era"] = "2025_feb_apr"
    bindings = [_binding(context_dir / "manifest.json"),
                _binding(path_label_dir / "index.json"), _binding(exact_ev_path)]
    return joined, bindings


def materialize_recent(
    feature_path: Path,
    grid_path: Path,
    exact_event_path: Path,
    features: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    available = normalize_identity(pd.read_parquet(feature_path, columns=None))
    source_columns = {
        feature: (
            f"capture_candidate__{feature}"
            if f"capture_candidate__{feature}" in available.columns
            else feature
        )
        for feature in features
    }
    missing = [feature for feature, source in source_columns.items()
               if source not in available.columns]
    if missing:
        raise ValueError(f"recent feature surface misses {len(missing)} fields")
    recent = available.loc[
        :, [*IDENTITY, "base_oof_score", *source_columns.values()]
    ].copy()
    recent = recent.rename(columns={source: feature
                                    for feature, source in source_columns.items()})
    label_columns = [
        *IDENTITY,
        "label_resolution_utc",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
        "soft_label",
        "peak_mfe_atr",
        "time_to_80pct_mfe_hours",
        "future_close_slope_atr_per_hour",
    ]
    labels = normalize_identity(pd.read_parquet(
        grid_path, filters=[("grid_name", "==", GRID)], columns=label_columns
    )).rename(
        columns={
            "time_to_80pct_mfe_hours": "time_to_meaningful_mfe_hours",
            "future_close_slope_atr_per_hour": "future_slope_atr_per_hour",
        }
    )
    exact = normalize_identity(
        pd.read_parquet(
            exact_event_path,
            columns=[
                *IDENTITY,
                "__soft_tb_label_available_at__",
                "__soft_tb_first_event__",
                "__soft_tb_upper_hit_12h__",
                "__soft_tb_lower_hit_12h__",
            ],
        )
    ).rename(
        columns={
            "__soft_tb_label_available_at__": "exact_event_resolution_utc",
            "__soft_tb_first_event__": "_event",
            "__soft_tb_upper_hit_12h__": "soft_upper_hit",
            "__soft_tb_lower_hit_12h__": "soft_lower_hit",
        }
    )
    labels = labels.merge(
        exact, on=list(IDENTITY), how="inner", validate="one_to_one"
    )
    if len(labels) != len(exact):
        raise AssertionError("recent exact-1m event/economic label join is incomplete")
    grid_resolution = pd.to_datetime(labels["label_resolution_utc"], utc=True)
    exact_resolution = pd.to_datetime(labels["exact_event_resolution_utc"], utc=True)
    if not grid_resolution.eq(exact_resolution).all():
        raise AssertionError("recent exact event and economic labels resolve differently")
    labels = labels.drop(columns="exact_event_resolution_utc")
    labels["meaningful_mfe_reached"] = labels["soft_upper_hit"].astype(np.int8)
    # The recent grid does not contain the older path-specific MAE-before-MFE
    # target.  Preserve the semantic distinction rather than substituting the
    # available early-three-bar adverse excursion.
    labels["mae_before_meaningful_mfe_atr"] = np.nan
    labels["execution_cost_return"] = (
        pd.to_numeric(labels["execution_gross_ev_12h"], errors="raise")
        - pd.to_numeric(labels["execution_net_ev_12h"], errors="raise")
    )
    joined = recent.merge(
        labels, on=list(IDENTITY), how="inner", validate="one_to_one"
    )
    if len(joined) != len(recent):
        raise AssertionError(
            f"recent exact join lost rows: {len(recent)} -> {len(joined)}"
        )
    joined = derive_event_columns(joined, "_event")
    joined["era"] = "2026_may_jul19"
    return joined, [
        _binding(feature_path), _binding(grid_path), _binding(exact_event_path)
    ]


def validate_output(frame: pd.DataFrame, features: list[str]) -> None:
    if frame.duplicated(list(IDENTITY)).any():
        raise AssertionError("duplicate candidate identity")
    if not set(frame["side_name"].astype(str).unique()) <= {"long", "short"}:
        raise AssertionError("unexpected side")
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(frame["execution_cost_return"], errors="raise")
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise")
    if float(np.nanmax(np.abs(gross - cost - net))) > 1e-10:
        raise AssertionError("gross - cost != net")
    if frame[features].shape[1] != 256:
        raise AssertionError("output feature matrix is not 256 columns")
    if frame[features].isna().all(axis=0).any():
        raise AssertionError("an entire feature column is missing")
    resolution = pd.to_datetime(frame["label_resolution_utc"], utc=True)
    decision = pd.to_datetime(frame["__ts__"], utc=True) + pd.Timedelta(hours=1)
    if (resolution < decision + pd.Timedelta(hours=12)).any():
        raise AssertionError("a label resolves before decision + 12h")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = json.loads(args.feature_contract.read_text())
    features = validate_feature_contract(contract)
    old, old_bindings = materialize_old(
        args.old_context, args.old_path_labels, args.old_exact_ev, features
    )
    recent, recent_bindings = materialize_recent(
        args.recent_features, args.recent_grid, args.recent_exact_events, features
    )
    columns = [
        *IDENTITY,
        "era",
        "label_resolution_utc",
        "base_oof_score",
        *features,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "positive_net",
        "negative_net",
        "event_class",
        "clean_first",
        "adverse_first",
        "timeout_event",
        "soft_upper_hit",
        "soft_lower_hit",
        "meaningful_mfe_reached",
        "peak_mfe_atr",
        "time_to_meaningful_mfe_hours",
        "mae_before_meaningful_mfe_atr",
        "future_slope_atr_per_hour",
    ]
    combined = pd.concat(
        [old.loc[:, columns], recent.loc[:, columns]], ignore_index=True
    ).sort_values(["__ts__", "side_name", "candidate_id"], kind="stable")
    combined = add_candidate_relative_context(combined)
    validate_output(combined, features)
    args.output_dir.mkdir(parents=True)
    data_path = args.output_dir / "cross_era_tail_payoff_dataset.parquet"
    combined.to_parquet(data_path, index=False)
    feature_path = args.output_dir / "feature_contract.json"
    _write_json(
        feature_path,
        {
            "schema": SCHEMA,
            "feature_columns": features,
            "feature_count": len(features),
            "candidate_context_columns": [
                "base_oof_score",
                "candidate_group_size",
                "base_rank_timestamp_side",
                "base_rank_pct_timestamp_side",
                "base_score_z_timestamp_side",
                "base_margin_to_candidate_cutoff",
            ],
            "source_contract": _binding(args.feature_contract),
        },
    )
    summary = (
        combined.assign(
            month=pd.to_datetime(combined["__ts__"], utc=True).dt.strftime("%Y-%m")
        )
        .groupby(["era", "month", "side_name"], observed=True)
        .agg(
            rows=("candidate_id", "size"),
            clean_first_rate=("clean_first", "mean"),
            adverse_first_rate=("adverse_first", "mean"),
            timeout_rate=("timeout_event", "mean"),
            positive_net_rate=("positive_net", "mean"),
            mean_net_bps=("execution_net_ev_12h", lambda x: x.mean() * 1e4),
        )
        .reset_index()
    )
    summary_path = args.output_dir / "support_by_era_month_side.csv"
    summary.to_csv(summary_path, index=False)
    report = {
        "schema": SCHEMA,
        "status": "materialized_research_only",
        "rows": len(combined),
        "rows_by_era": combined["era"].value_counts().sort_index().to_dict(),
        "rows_by_side": combined["side_name"].value_counts().sort_index().to_dict(),
        "feature_count": len(features),
        "event_contract": {
            "grid": GRID,
            "favorable": "1.5 ATR/cost-aware meaningful barrier first",
            "adverse": "1 ATR adverse first or same-minute conflict",
            "timeout": "neither barrier first within 12h",
            "resolution": (
                "spread-adjusted executable exact 1m paths for both eras; "
                "same-minute conflict is adverse"
            ),
        },
        "inputs": {
            "old": old_bindings,
            "recent": recent_bindings,
            "feature_contract": _binding(args.feature_contract),
        },
        "outputs": {
            "dataset": {**_binding(data_path), "rows": len(combined)},
            "feature_contract": _binding(feature_path),
            "support": {**_binding(summary_path), "rows": len(summary)},
        },
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "status": report["status"],
            "report": _binding(report_path),
            "outputs": report["outputs"],
        },
    )
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feature-contract", type=Path,
        default=Path(
            "data_perp/artifacts/packb_side_local_ae_20260724_v1/long/"
            "loader_evidence/frozen_feature_contract.json"
        ),
    )
    p.add_argument(
        "--old-context", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "febapr2025_historical_path_head_context_20260727_v1"
        ),
    )
    p.add_argument(
        "--old-path-labels", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "febapr2025_top40_exact1m_path_head_labels_20260727_v1"
        ),
    )
    p.add_argument(
        "--old-exact-ev", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "febapr2025_native12h_execution_ev_divergence_20260729_v1/"
            "joined_scores_execution_ev.parquet"
        ),
    )
    p.add_argument(
        "--recent-features", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "exact_policy_capture_feature_universe_20260727_v2/"
            "capture_feature_universe.parquet"
        ),
    )
    p.add_argument(
        "--recent-grid", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "meaningful_mfe_label_grid.parquet"
        ),
    )
    p.add_argument(
        "--recent-exact-events", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "harmonized_mayjul19_exact1m_clean_first_labels_20260730_v1/"
            "exact_clean_first_labels.parquet"
        ),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "data_perp/artifacts/"
            "cross_era_tail_payoff_dataset_20260730_v3"
        ),
    )
    return p


if __name__ == "__main__":
    run(parser().parse_args())
