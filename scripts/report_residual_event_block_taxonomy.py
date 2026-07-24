#!/usr/bin/env python3
"""Build a descriptive taxonomy of difficult residual periods, not trade rows."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.residual_event_block_taxonomy import (
    BlockTaxonomyConfig,
    MECHANISM_FAMILIES,
    annotate_onset_mechanism_profiles,
    attach_detector_block_coverage,
    block_family_profiles,
    build_block_taxonomy,
    daily_observable_state,
    detector_recognized_missed_contrasts,
    matched_benign_block_controls,
)


def _load_calendar(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one calendar is required")
    result = pd.concat(frames, ignore_index=True, copy=False)
    result["day"] = pd.to_datetime(result["day"], utc=True).dt.floor("D")
    return result.drop_duplicates(
        ["day", "side_name", "archetype_policy_key"], keep="last"
    )


def _overlay_event_calendar(daily: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    """Use a sparse adverse-event calendar to mark an otherwise full history."""

    if not paths:
        return daily
    events = _load_calendar(paths)
    event_column = (
        "adverse_calendar_cell"
        if "adverse_calendar_cell" in events.columns
        else "adverse_event_rows"
    )
    if event_column not in events.columns:
        raise KeyError("Event calendar needs adverse_calendar_cell or adverse_event_rows")
    keys = ["day", "side_name", "archetype_policy_key"]
    event_columns = [*keys, event_column, *[
        name for name in (
            "persistence_strength", "large_event_strength", "selected_rows",
            "mean_ev_after_1pct", "clean_exec_rate", "clean_exec_precision",
            "signed_surprise",
        )
        if name in events.columns
    ]]
    event_values = events.loc[:, event_columns].copy()
    event_values["__event_flag"] = pd.to_numeric(
        event_values[event_column], errors="coerce"
    ).fillna(0).gt(0).astype(np.int8)
    result = daily.merge(
        event_values.drop(columns=[event_column]), on=keys, how="left", suffixes=("", "__event"), validate="one_to_one"
    )
    result["adverse_calendar_cell"] = result["__event_flag"].fillna(0).astype(np.int8)
    result = result.drop(columns=["__event_flag"])
    for name in (
        "persistence_strength", "large_event_strength", "selected_rows",
        "mean_ev_after_1pct", "clean_exec_rate", "clean_exec_precision",
        "signed_surprise",
    ):
        event_name = f"{name}__event"
        if event_name in result.columns:
            if name in result.columns:
                result[name] = result[event_name].combine_first(result[name])
            else:
                result[name] = result[event_name]
            result = result.drop(columns=[event_name])
    return result


def _state_columns(path: Path, requested: list[str]) -> list[str]:
    # Read parquet metadata, not an entire 1M+ row historical state store.
    available = set(pq.ParquetFile(path).schema.names)
    required = ["__ts__", "side_name", "archetype_policy_key", "selected_top30"]
    return [name for name in [*required, *requested] if name in available]


def _daily_state_part_streaming(path: Path, features: list[str]) -> pd.DataFrame:
    """Reduce one large state Parquet without materializing its full matrix.

    The block taxonomy only needs one cross-sectional snapshot per day x side
    x archetype.  First find each group's earliest available timestamp, then
    retain only rows at that timestamp for the median reduction.  This is
    exactly the existing daily-open contract, but caps peak memory near a
    single record batch instead of the full multi-million-row state artifact.
    """

    columns = _state_columns(path, features)
    required = {"__ts__", "side_name", "archetype_policy_key"}
    missing = required.difference(columns)
    if missing:
        raise KeyError(f"State artifact {path} missing keys: {sorted(missing)}")
    available = [name for name in features if name in columns]
    source = pq.ParquetFile(path)
    key_columns = ["day", "side_name", "archetype_policy_key"]
    first_rows: list[pd.DataFrame] = []
    for batch in source.iter_batches(columns=["__ts__", "side_name", "archetype_policy_key"], batch_size=100_000):
        frame = batch.to_pandas()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame = frame.loc[frame["__ts__"].notna()].copy()
        if frame.empty:
            continue
        frame["day"] = frame["__ts__"].dt.floor("D")
        first_rows.append(
            frame.groupby(key_columns, observed=True, as_index=False)["__ts__"].min()
        )
    if not first_rows:
        return pd.DataFrame(columns=[*key_columns, *available])
    first = pd.concat(first_rows, ignore_index=True, copy=False)
    first = first.groupby(key_columns, observed=True, as_index=False)["__ts__"].min()
    selected_rows: list[pd.DataFrame] = []
    projection = ["__ts__", "side_name", "archetype_policy_key", *available]
    for batch in source.iter_batches(columns=projection, batch_size=100_000):
        frame = batch.to_pandas()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame = frame.loc[frame["__ts__"].notna()].copy()
        if frame.empty:
            continue
        frame["day"] = frame["__ts__"].dt.floor("D")
        frame = frame.merge(first, on=[*key_columns, "__ts__"], how="inner")
        if not frame.empty:
            selected_rows.append(frame.loc[:, [*key_columns, *available]])
    if not selected_rows:
        return pd.DataFrame(columns=[*key_columns, *available])
    selected = pd.concat(selected_rows, ignore_index=True, copy=False)
    for name in available:
        selected[name] = pd.to_numeric(selected[name], errors="coerce").astype(np.float32)
    return selected.groupby(key_columns, observed=True, as_index=False)[available].median()


def _load_daily_state(paths: list[Path], features: list[str]) -> pd.DataFrame:
    """Collapse each parquet independently with a bounded daily-open reducer."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(_daily_state_part_streaming(path, features))
        gc.collect()
    result = pd.concat(frames, ignore_index=True, copy=False)
    value_columns = [name for name in result.columns if name not in {"day", "side_name", "archetype_policy_key"}]
    # Artifacts can overlap at a month boundary.  Taking a daily median again
    # prevents a duplicate source from changing the cross-sectional estimate.
    return (
        result.groupby(["day", "side_name", "archetype_policy_key"], observed=True, as_index=False)[value_columns]
        .median()
        .sort_values(["day", "side_name", "archetype_policy_key"], kind="stable")
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(feature for group in MECHANISM_FAMILIES.values() for feature in group))
    # The primary contract is the observable top-30 base candidate state.  The
    # all-rows flag only changes that within each parquet, never the later
    # block-level outcome handling.
    if args.all_rows:
        raise ValueError("--all-rows is not supported by the memory-safe taxonomy path")
    daily = _load_daily_state(args.state_artifact, requested)
    if args.calendar_from_state:
        calendar = daily.loc[:, ["day", "side_name", "archetype_policy_key"]].copy()
    else:
        calendar = _load_calendar(args.calendar)
    calendar = _overlay_event_calendar(calendar, args.event_calendar)
    config = BlockTaxonomyConfig(
        pre_days=args.pre_days,
        post_days=args.post_days,
        min_reference_days=args.min_reference_days,
        controls_per_block=args.controls_per_block,
        max_clusters=args.max_clusters,
        min_cluster_blocks=args.min_cluster_blocks,
    )
    blocks, trajectories = build_block_taxonomy(calendar, daily, config=config)
    blocks = annotate_onset_mechanism_profiles(blocks)
    if args.detector_oof:
        detector = pd.read_parquet(args.detector_oof)
        required = {"model_arm", "model_target", "day", "side_name", "archetype_policy_key", args.detector_risk_column}
        missing = required.difference(detector.columns)
        if missing:
            raise KeyError(f"Detector artifact missing: {sorted(missing)}")
        detector = detector.loc[
            detector["model_arm"].astype(str).eq(args.detector_arm)
            & detector["model_target"].astype(str).eq(args.detector_target)
        ]
        blocks = attach_detector_block_coverage(
            blocks,
            detector,
            risk_column=args.detector_risk_column,
            threshold=args.detector_threshold,
        )
    # Coverage is attached after taxonomy construction; derive profiles from
    # that enriched block table so train-unavailable blocks are not mistaken
    # for detector misses.
    profiles = block_family_profiles(blocks)
    calendar_columns = [
        name
        for name in (
            "event_start", "event_end", "event_days", "side_name",
            "archetype_policy_key", "event_block", "block_family",
            "block_family_id", "cluster_silhouette", "onset_primary_mechanism",
            "onset_primary_mechanism_score", "onset_mechanism_margin",
            "onset_mechanism_confident", "calendar_mean_ev",
            "calendar_mean_signed_surprise", "calendar_persistence_strength",
            "calendar_large_event_strength", "calendar_selected_rows",
        )
        if name in blocks.columns
    ]
    mechanism_calendar = blocks.loc[:, calendar_columns].sort_values(
        ["event_start", "side_name", "archetype_policy_key"], kind="stable"
    ) if not blocks.empty else pd.DataFrame(columns=calendar_columns)
    onset_inventory = (
        blocks.groupby(
            ["side_name", "archetype_policy_key", "onset_primary_mechanism"],
            observed=True,
            as_index=False,
        )
        .agg(
            blocks=("event_block", "size"),
            confident_blocks=("onset_mechanism_confident", "sum"),
            mean_mechanism_score=("onset_primary_mechanism_score", "mean"),
            mean_mechanism_margin=("onset_mechanism_margin", "mean"),
            mechanism_families_available=("onset_mechanism_available_count", "max"),
            mean_event_days=("event_days", "mean"),
        )
        .sort_values(
            ["side_name", "archetype_policy_key", "blocks"],
            ascending=[True, True, False],
            kind="stable",
        )
    ) if not blocks.empty else pd.DataFrame()
    detector_contrasts = detector_recognized_missed_contrasts(blocks)
    controls = matched_benign_block_controls(calendar, daily, blocks, config=config)
    blocks.to_csv(args.output / "event_block_taxonomy.csv", index=False)
    mechanism_calendar.to_csv(
        args.output / "event_block_mechanism_calendar.csv", index=False
    )
    trajectories.to_parquet(args.output / "event_block_trajectories.parquet", index=False, compression="zstd")
    profiles.to_csv(args.output / "block_family_profiles.csv", index=False)
    onset_inventory.to_csv(args.output / "onset_mechanism_inventory.csv", index=False)
    detector_contrasts.to_csv(
        args.output / "detector_recognized_vs_missed_block_contrasts.csv", index=False
    )
    controls.to_csv(args.output / "matched_benign_block_controls.csv", index=False)
    manifest = {
        "purpose": "descriptive block-level adverse-regime taxonomy; not an inference policy",
        "calendar_paths": [str(path) for path in args.calendar],
        "calendar_from_state": bool(args.calendar_from_state),
        "event_calendar_paths": [str(path) for path in args.event_calendar],
        "state_artifacts": [str(path) for path in args.state_artifact],
        "daily_snapshot_contract": "first available timestamp per day, then cross-sectional median; no same-day future values",
        "observable_features_requested": requested,
        "observable_features_available": [name for name in requested if name in daily.columns],
        # A primitive can exist in a sparse historical artifact without enough
        # finite trajectory data to form an onset score.  Record the latter:
        # it is the actual contract used by the catalogue.
        "onset_mechanism_families_available": [
            name.removeprefix("onset_mechanism_score__")
            for name in blocks.columns
            if name.startswith("onset_mechanism_score__")
        ],
        "block_count": int(len(blocks)),
        "local_family_count": int(blocks.loc[blocks["block_family_id"].ge(0), ["side_name", "archetype_policy_key", "block_family_id"]].drop_duplicates().shape[0]) if not blocks.empty else 0,
        "config": vars(args),
        "causal_contract": (
            "Outcome calendar identifies adverse blocks only. All trajectory values and control matching use "
            "observable state features, and each block normalizes against preceding local history."
        ),
        "detector_coverage": {
            "artifact": str(args.detector_oof) if args.detector_oof else None,
            "model_arm": args.detector_arm if args.detector_oof else None,
            "model_target": args.detector_target if args.detector_oof else None,
            "risk_column": args.detector_risk_column if args.detector_oof else None,
            "threshold": args.detector_threshold if args.detector_oof else None,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, action="append", default=[])
    parser.add_argument("--calendar-from-state", action="store_true", help="Use all available state days as the daily calendar; pair with --event-calendar for adverse labels.")
    parser.add_argument("--event-calendar", type=Path, action="append", default=[], help="Optional sparse adverse-event calendar overlaid on the full daily calendar.")
    parser.add_argument("--state-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-days", type=int, default=2)
    parser.add_argument("--post-days", type=int, default=1)
    parser.add_argument("--min-reference-days", type=int, default=30)
    parser.add_argument("--controls-per-block", type=int, default=3)
    parser.add_argument("--skip-controls", action="store_true", help="Skip matched benign controls for broad taxonomy inventory runs.")
    parser.add_argument("--max-clusters", type=int, default=5)
    parser.add_argument("--min-cluster-blocks", type=int, default=3)
    parser.add_argument("--all-rows", action="store_true", help="Use all base rows instead of observable top-30 candidate state.")
    parser.add_argument("--detector-oof", type=Path, help="Optional frozen detector OOF predictions for read-only block-coverage reporting.")
    parser.add_argument("--detector-arm", default="episode_lgbm__episode_onset_top10_adverse_period")
    parser.add_argument("--detector-target", default="episode_onset_top10_adverse_period_target")
    parser.add_argument("--detector-risk-column", default="residual_error_risk_percentile")
    parser.add_argument("--detector-threshold", type=float, default=0.925)
    args = parser.parse_args()
    if not args.calendar and not args.calendar_from_state:
        parser.error("one or more --calendar paths or --calendar-from-state is required")
    if args.skip_controls:
        args.controls_per_block = 0
    return args


if __name__ == "__main__":
    manifest = run(parse_args())
    # The complete manifest is written to disk.  Keep CLI output deliberately
    # small so a large path/config object cannot obscure a successful run.
    print(
        f"completed blocks={manifest['block_count']} "
        f"local_families={manifest['local_family_count']}"
    )
