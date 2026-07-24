#!/usr/bin/env python3
"""Compose validated local overlay revisions and report adverse-event coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
RANK = "parent_rank_v9_residual_error_overlay"
PARENT_RANK = "parent_rank_v9"
EVENT = "adverse_calendar_cell"
SHORT_BREAKOUT = ("short", "short_breakout_precision")


def _metrics(frame: pd.DataFrame, rank: np.ndarray) -> dict[str, float]:
    selected = rank >= 0.90
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(np.float32)
    clean = pd.to_numeric(frame["clean_exec"], errors="coerce").to_numpy(np.float32)
    event = frame[EVENT].to_numpy(bool)
    return {
        "selected_rows": int(selected.sum()),
        "mean_ev_after_1pct": float(np.nanmean(ev[selected])),
        "positive_ev_rate": float(np.mean(ev[selected] > 0.0)),
        "clean_exec_precision": float(np.nanmean(clean[selected])),
        "event_mean_ev_after_1pct": float(np.nanmean(ev[selected & event])),
        "normal_mean_ev_after_1pct": float(np.nanmean(ev[selected & ~event])),
    }


def _accepted(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame = pd.read_csv(path / "accepted_local_overlays.csv")
    return {
        (str(row.side_name), str(row.archetype_policy_key)): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _risk_by_cell(
    predictions: pd.DataFrame,
    accepted: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (side, archetype), params in accepted.items():
        risk_column = str(params["risk_variant"])
        local = predictions.loc[
            predictions["side_name"].astype(str).eq(side)
            & predictions["archetype_policy_key"].astype(str).eq(archetype)
            & predictions[PARENT_RANK].ge(0.90),
            ["day", "side_name", "archetype_policy_key", risk_column],
        ]
        if local.empty:
            continue
        grouped = (
            local.groupby(
                ["day", "side_name", "archetype_policy_key"],
                observed=True,
                sort=False,
            )[risk_column]
            .max()
            .rename("max_risk")
            .reset_index()
        )
        grouped["risk_threshold"] = float(params["threshold"])
        grouped["recognized"] = grouped["max_risk"].ge(grouped["risk_threshold"])
        grouped["recognition_margin"] = (
            grouped["max_risk"] - grouped["risk_threshold"]
        )
        parts.append(grouped)
    if not parts:
        return pd.DataFrame(
            columns=[
                "day",
                "side_name",
                "archetype_policy_key",
                "max_risk",
                "risk_threshold",
                "recognized",
                "recognition_margin",
            ]
        )
    return pd.concat(parts, ignore_index=True, copy=False)


def _add_blocks(calendar: pd.DataFrame) -> pd.DataFrame:
    result = calendar.sort_values(
        ["side_name", "archetype_policy_key", "day"], kind="stable"
    ).copy()
    gap = (
        result.groupby(["side_name", "archetype_policy_key"], observed=True)["day"]
        .diff()
        .dt.days
    )
    new_block = gap.isna() | gap.gt(1)
    result["event_block"] = (
        new_block.groupby(
            [result["side_name"], result["archetype_policy_key"]],
            observed=True,
        )
        .cumsum()
        .astype(np.int16)
    )
    return result


def _coverage_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["side_name", "archetype_policy_key"]
    for keys, local in events.groupby(groups, observed=True, sort=True):
        assessable = local["hybrid_max_risk"].notna()
        recognized = local["hybrid_recognized"].fillna(False)
        baseline_recognized = local["baseline_recognized"].fillna(False)
        mechanism_recognized = local.get(
            "mechanism_recognized", pd.Series(False, index=local.index)
        ).astype("boolean").fillna(False)
        block = local.loc[assessable].groupby("event_block", observed=True)
        hybrid_blocks = block["hybrid_recognized"].max() if assessable.any() else pd.Series(dtype=bool)
        baseline_blocks = block["baseline_recognized"].max() if assessable.any() else pd.Series(dtype=bool)
        rows.append(
            {
                "side_name": keys[0],
                "archetype_policy_key": keys[1],
                "adverse_cells": len(local),
                "assessable_cells": int(assessable.sum()),
                "hybrid_recognized_cells": int((assessable & recognized).sum()),
                "hybrid_explained_cell_pct": float(recognized[assessable].mean())
                if assessable.any()
                else np.nan,
                "hybrid_unexplained_cell_pct": float((~recognized[assessable]).mean())
                if assessable.any()
                else np.nan,
                "baseline_recognized_cells": int((assessable & baseline_recognized).sum()),
                "delta_recognized_cells": int(
                    (assessable & recognized).sum()
                    - (assessable & baseline_recognized).sum()
                ),
                "temporal_mechanism_detected_cells": int(
                    (assessable & mechanism_recognized).sum()
                ),
                "temporal_mechanism_detected_cell_pct": float(
                    mechanism_recognized[assessable].mean()
                )
                if assessable.any()
                else np.nan,
                "assessable_event_blocks": int(len(hybrid_blocks)),
                "hybrid_recognized_event_blocks": int(hybrid_blocks.sum()),
                "hybrid_explained_event_block_pct": float(hybrid_blocks.mean())
                if len(hybrid_blocks)
                else np.nan,
                "baseline_recognized_event_blocks": int(baseline_blocks.sum()),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(args.baseline / "oos_predictions.parquet")
    temporal = pd.read_parquet(args.temporal / "oos_predictions.parquet", columns=KEYS + [RANK])
    merged = baseline.merge(
        temporal,
        on=KEYS,
        how="left",
        suffixes=("_baseline", "_temporal"),
        validate="one_to_one",
    )
    use_temporal = (
        merged["side_name"].astype(str).eq(SHORT_BREAKOUT[0])
        & merged["archetype_policy_key"].astype(str).eq(SHORT_BREAKOUT[1])
    ).to_numpy()
    hybrid_rank = merged[f"{RANK}_baseline"].to_numpy(np.float32).copy()
    temporal_rank = pd.to_numeric(
        merged[f"{RANK}_temporal"], errors="coerce"
    ).to_numpy(np.float32)
    replace = use_temporal & np.isfinite(temporal_rank)
    hybrid_rank[replace] = temporal_rank[replace]
    merged["parent_rank_v9_temporal_greedy_revision"] = hybrid_rank
    merged["temporal_revision_source"] = np.where(replace, "temporal_short_breakout", "v11")
    merged.to_parquet(args.output / "oos_predictions.parquet", index=False, compression="zstd")

    metric_rows = []
    for name, rank in (
        ("v9_parent", merged[PARENT_RANK].to_numpy(np.float32)),
        ("v11", merged[f"{RANK}_baseline"].to_numpy(np.float32)),
        ("temporal_full", temporal_rank),
        ("temporal_greedy_revision", hybrid_rank),
    ):
        metric_rows.append({"selector": name, **_metrics(merged, rank)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "summary.csv", index=False)

    baseline_accepted = _accepted(args.baseline)
    temporal_accepted = _accepted(args.temporal)
    hybrid_accepted = dict(baseline_accepted)
    hybrid_accepted[SHORT_BREAKOUT] = temporal_accepted[SHORT_BREAKOUT]

    prediction_parts: list[pd.DataFrame] = []
    for root, source in ((args.baseline, "baseline"), (args.temporal, "temporal")):
        for filename, stage in (
            ("train_oof_predictions.parquet", "train_oof"),
            ("oos_predictions.parquet", "eval_oos"),
        ):
            frame = pd.read_parquet(root / filename)
            frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
            frame["source"] = source
            frame["stage"] = stage
            prediction_parts.append(frame)
    all_predictions = pd.concat(prediction_parts, ignore_index=True, copy=False)

    baseline_predictions = all_predictions.loc[all_predictions["source"].eq("baseline")]
    baseline_risk = _risk_by_cell(baseline_predictions, baseline_accepted).add_prefix("baseline_")
    baseline_risk = baseline_risk.rename(
        columns={
            "baseline_day": "day",
            "baseline_side_name": "side_name",
            "baseline_archetype_policy_key": "archetype_policy_key",
        }
    )

    hybrid_parts: list[pd.DataFrame] = []
    for key, params in hybrid_accepted.items():
        source = "temporal" if key == SHORT_BREAKOUT else "baseline"
        local_predictions = all_predictions.loc[all_predictions["source"].eq(source)]
        part = _risk_by_cell(local_predictions, {key: params})
        hybrid_parts.append(part)
    hybrid_risk = pd.concat(hybrid_parts, ignore_index=True, copy=False).add_prefix("hybrid_")
    hybrid_risk = hybrid_risk.rename(
        columns={
            "hybrid_day": "day",
            "hybrid_side_name": "side_name",
            "hybrid_archetype_policy_key": "archetype_policy_key",
        }
    )

    mechanism_risk = None
    if args.temporal_only_diagnostic is not None:
        mechanism_accepted = _accepted(args.temporal_only_diagnostic)
        diagnostic_parts: list[pd.DataFrame] = []
        for filename in ("train_oof_predictions.parquet", "oos_predictions.parquet"):
            frame = pd.read_parquet(args.temporal_only_diagnostic / filename)
            frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
            diagnostic_parts.append(frame)
        mechanism_risk = _risk_by_cell(
            pd.concat(diagnostic_parts, ignore_index=True, copy=False),
            mechanism_accepted,
        ).add_prefix("mechanism_")
        mechanism_risk = mechanism_risk.rename(
            columns={
                "mechanism_day": "day",
                "mechanism_side_name": "side_name",
                "mechanism_archetype_policy_key": "archetype_policy_key",
            }
        )

    calendars = []
    for root, filename, stage in (
        (args.baseline, "v9_train_residual_calendar.csv", "train"),
        (args.baseline, "v9_eval_residual_calendar.csv", "eval"),
    ):
        frame = pd.read_csv(root / filename)
        frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
        frame["calendar_stage"] = stage
        calendars.append(frame.loc[frame[EVENT].gt(0)])
    calendar = _add_blocks(pd.concat(calendars, ignore_index=True, copy=False))
    join_keys = ["day", "side_name", "archetype_policy_key"]
    calendar = calendar.merge(baseline_risk, on=join_keys, how="left", validate="one_to_one")
    calendar = calendar.merge(hybrid_risk, on=join_keys, how="left", validate="one_to_one")
    if mechanism_risk is not None:
        calendar = calendar.merge(
            mechanism_risk, on=join_keys, how="left", validate="one_to_one"
        )
    for prefix in ("baseline", "hybrid"):
        calendar[f"{prefix}_recognized"] = calendar[f"{prefix}_recognized"].astype("boolean")
    calendar["status"] = np.select(
        [
            calendar["hybrid_max_risk"].isna(),
            calendar["hybrid_recognized"].fillna(False).to_numpy(bool),
            calendar.get(
                "mechanism_recognized", pd.Series(False, index=calendar.index)
            ).astype("boolean").fillna(False).to_numpy(bool),
        ],
        ["not_oof_assessable", "recognized", "diagnostic_only_not_promoted"],
        default="still_unresolved",
    )
    calendar["recognition_change_vs_v11"] = np.select(
        [
            calendar["baseline_recognized"].fillna(False).to_numpy(bool)
            == calendar["hybrid_recognized"].fillna(False).to_numpy(bool),
            calendar["hybrid_recognized"].fillna(False).to_numpy(bool),
        ],
        ["unchanged", "newly_recognized"],
        default="lost_recognition",
    )
    calendar.to_csv(args.output / "adverse_event_status.csv", index=False)
    coverage = _coverage_summary(calendar)
    coverage.to_csv(args.output / "archetype_explained_unexplained.csv", index=False)
    unexplained = calendar.loc[calendar["status"].eq("still_unresolved")].sort_values(
        ["mean_ev_after_1pct", "signed_surprise", "day"], kind="stable"
    )
    unexplained.to_csv(args.output / "most_unexplained.csv", index=False)

    manifest = {
        "schema": "temporal_mechanism_greedy_revision_v1",
        "baseline": str(args.baseline),
        "temporal": str(args.temporal),
        "temporal_only_diagnostic": str(args.temporal_only_diagnostic)
        if args.temporal_only_diagnostic is not None
        else None,
        "revision": {
            "long_volcompression_wideslow_candidate": "v11",
            "short_default_clean_path": "v11",
            "short_breakout_precision": "temporal_v12",
            "short_mixed_clean_path": "no_validated_overlay",
        },
        "selection_contract": (
            "Every local component was selected on chronological OOF predictions. "
            "April-June OOS metrics were not used to choose the component source."
        ),
        "assessable_adverse_cells": int(calendar["hybrid_max_risk"].notna().sum()),
        "recognized_adverse_cells": int(calendar["hybrid_recognized"].fillna(False).sum()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--temporal-only-diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
