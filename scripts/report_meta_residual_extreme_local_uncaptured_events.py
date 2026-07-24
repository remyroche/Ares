#!/usr/bin/env python3
"""Audit residual extreme-event cells missed by the strict local revision."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_residual_extreme_local_champion_overlay import (  # noqa: E402
    FEATURES,
    KEYS,
    _adjust_rank,
    _composite,
    _fit_references,
)


def _load_meta_oof_rank(
    directory: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["__ts__"] = pd.to_datetime(cached["__ts__"], utc=True)
        cached = cached.loc[cached["__ts__"].ge(start) & cached["__ts__"].lt(end)]
        cached["parent_rank"] = pd.to_numeric(
            cached["historical_rank"], errors="coerce"
        ).astype(np.float32)
        cached["selection_contract"] = "chronological_meta_oof_global_train_rank"
        return cached.loc[:, [*KEYS, "parent_rank", "selection_contract"]]
    parts: list[pd.DataFrame] = []
    columns = [*KEYS, "score_meta_base_soft_label"]
    for raw_path in sorted(glob.glob(str(directory / "*.parquet"))):
        part = pd.read_parquet(raw_path, columns=columns)
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True)
        part = part.loc[part["__ts__"].ge(start) & part["__ts__"].lt(end)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError("No chronological meta OOF rows in audit period")
    oof = pd.concat(parts, ignore_index=True, copy=False).drop_duplicates(
        KEYS, keep="last"
    )
    score = pd.to_numeric(oof["score_meta_base_soft_label"], errors="coerce")
    oof = oof.loc[score.notna()].copy()
    oof["parent_rank"] = score.loc[score.notna()].rank(
        method="average", pct=True
    ).astype(np.float32)
    oof["selection_contract"] = "chronological_meta_oof_global_train_rank"
    return oof.loc[:, [*KEYS, "parent_rank", "selection_contract"]]


def _daily_capture(rows: pd.DataFrame, threshold: float, alpha_down: float) -> pd.DataFrame:
    parent_rank = pd.to_numeric(rows["parent_rank"], errors="coerce").fillna(0.0)
    parent_selected = parent_rank.ge(0.90).to_numpy()
    adverse = pd.to_numeric(rows["adverse_composite"], errors="coerce").fillna(0.5)
    positive = pd.to_numeric(rows["positive_composite"], errors="coerce").fillna(0.5)
    adjusted = _adjust_rank(
        parent_rank.to_numpy(dtype=np.float32),
        adverse.to_numpy(dtype=np.float32),
        positive.to_numpy(dtype=np.float32),
        threshold=threshold,
        alpha_down=alpha_down,
        alpha_up=0.0,
    )
    tail = adverse.ge(threshold).to_numpy()
    rows = rows.copy()
    rows["day"] = rows["__ts__"].dt.floor("D")
    rows["parent_selected"] = parent_selected
    rows["adverse_tail_detected"] = tail
    rows["selected_adverse_tail"] = parent_selected & tail
    rows["demoted"] = parent_selected & (adjusted < 0.90)
    return (
        rows.groupby(
            ["day", "side_name", "archetype_policy_key"],
            observed=True,
            dropna=False,
        )
        .agg(
            candidate_rows=("parent_selected", "size"),
            parent_selected_rows=("parent_selected", "sum"),
            adverse_tail_rows=("adverse_tail_detected", "sum"),
            selected_adverse_tail_rows=("selected_adverse_tail", "sum"),
            demoted_rows=("demoted", "sum"),
            selection_contract=("selection_contract", "first"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tail9596_20260712_v7"
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3"
        ),
    )
    parser.add_argument(
        "--meta-oof-dir",
        type=Path,
        default=Path(
            "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_"
            "mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_"
            "payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_"
            "lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/"
            "best_full_oos_fixedfs_streamed_v1/prediction_shards"
        ),
    )
    parser.add_argument(
        "--meta-oof-rank-cache",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3/meta_oof_global_rank_"
            "202504_202603.parquet"
        ),
    )
    parser.add_argument(
        "--july-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/prediction_shards/"
            "predictions_2026-07.parquet"
        ),
    )
    parser.add_argument(
        "--precoverage-calendar",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/"
            "significant_surprise_calendar.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_uncaptured_events_"
            "202501_20260708_v1"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2025-04-01", tz="UTC")
    train_end = pd.Timestamp("2026-04-01", tz="UTC")
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    audit_end = pd.Timestamp("2026-07-09", tz="UTC")
    manifest = json.loads((args.overlay_dir / "manifest.json").read_text())
    params = dict(manifest["strict_best"])
    threshold = float(params["threshold"])
    alpha_down = float(params["alpha_down"])
    catalog = pd.read_csv(args.overlay_dir / "selected_local_features_strict.csv")

    state_path = args.state_dir / "oos_residual_event_states.parquet"
    state = pd.read_parquet(state_path, columns=[*KEYS, *FEATURES])
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    state = state.loc[state["__ts__"].ge(start) & state["__ts__"].lt(audit_end)]
    state = state.drop_duplicates(KEYS, keep="last")
    references = _fit_references(
        state.loc[state["__ts__"].lt(train_end)],
        catalog,
        int(params["top_feature_count"]),
    )
    state["adverse_composite"] = _composite(state, references, "adverse")
    state["positive_composite"] = _composite(state, references, "positive")

    pre_eval_rank = _load_meta_oof_rank(
        args.meta_oof_dir, start, train_end, args.meta_oof_rank_cache
    )
    pre_eval = state.loc[state["__ts__"].lt(train_end)].merge(
        pre_eval_rank, on=KEYS, how="inner", validate="one_to_one"
    )

    eval_predictions = pd.read_parquet(
        args.overlay_dir / "oos_predictions.parquet",
        columns=[*KEYS, "historical_rank", "selected_parent"],
    )
    eval_predictions["__ts__"] = pd.to_datetime(eval_predictions["__ts__"], utc=True)
    eval_rows = state.loc[
        state["__ts__"].ge(train_end) & state["__ts__"].lt(july_start)
    ].merge(eval_predictions, on=KEYS, how="inner", validate="one_to_one")
    eval_rows = eval_rows.rename(columns={"historical_rank": "parent_rank"})
    eval_rows["selection_contract"] = "exact_champion_causal_historical_rank"

    july = pd.read_parquet(
        args.july_predictions,
        columns=[
            *KEYS,
            "threshold_alternative_rank",
            "threshold_alternative_selected",
        ],
    )
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True)
    july = july.loc[july["__ts__"].ge(july_start) & july["__ts__"].lt(audit_end)]
    july_rows = state.loc[state["__ts__"].ge(july_start)].merge(
        july, on=KEYS, how="inner", validate="one_to_one"
    )
    july_rows = july_rows.rename(columns={"threshold_alternative_rank": "parent_rank"})
    july_rows["selection_contract"] = "exact_july_8d_reachable_ev_policy_rank"

    all_rows = pd.concat([pre_eval, eval_rows, july_rows], ignore_index=True, copy=False)
    capture = _daily_capture(all_rows, threshold, alpha_down)

    events = pd.read_csv(args.state_dir / "residual_event_calendar.csv")
    events["day"] = pd.to_datetime(events["day"], utc=True)
    events = events.loc[events["day"].ge(start) & events["day"].lt(audit_end)]
    event_cells = events.loc[
        events["adverse_event_rows"].gt(0) | events["favorable_event_rows"].gt(0)
    ].merge(
        capture,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "candidate_rows",
        "parent_selected_rows",
        "adverse_tail_rows",
        "selected_adverse_tail_rows",
        "demoted_rows",
    ):
        event_cells[column] = event_cells[column].fillna(0).astype(int)
    event_cells["adverse_state_detected"] = event_cells["adverse_tail_rows"].gt(0)
    event_cells["adverse_selected_state_detected"] = event_cells[
        "selected_adverse_tail_rows"
    ].gt(0)
    event_cells["selection_changed"] = event_cells["demoted_rows"].gt(0)
    event_cells["material_extreme"] = event_cells["large_event_strength"].ge(1.0) | event_cells[
        "persistence_strength"
    ].ge(1.0)
    event_cells["uncaptured_reason"] = np.select(
        [
            event_cells["adverse_event_rows"].le(0),
            ~event_cells["adverse_state_detected"],
            ~event_cells["adverse_selected_state_detected"],
            ~event_cells["selection_changed"],
        ],
        [
            "favorable_event_not_targeted",
            "local_tail_not_active",
            "local_tail_active_outside_parent_top10",
            "tail_detected_but_no_threshold_crossing",
        ],
        default="captured_and_changed",
    )

    precoverage = pd.read_csv(args.precoverage_calendar)
    precoverage["day"] = pd.to_datetime(precoverage["day"], utc=True)
    precoverage = precoverage.loc[
        precoverage["day"].lt(start)
        & precoverage["day"].lt(audit_end)
        & precoverage["scope"].eq("side_archetype")
        & precoverage["significant"].eq(True)
    ].copy()
    precoverage["uncaptured_reason"] = np.where(
        precoverage["surprise_sign"].eq("positive"),
        "no_state_coverage_positive_event_not_targeted",
        "no_residual_state_oos_coverage_before_2025_04",
    )
    precoverage["material_extreme"] = True
    precoverage["selection_changed"] = False

    event_cells.to_csv(args.output_dir / "all_extreme_event_cells.csv", index=False)
    uncaptured = event_cells.loc[
        event_cells["material_extreme"] & ~event_cells["selection_changed"]
    ].copy()
    uncaptured.to_csv(args.output_dir / "material_uncaptured_event_cells.csv", index=False)
    precoverage.to_csv(args.output_dir / "precoverage_uncaptured_event_cells.csv", index=False)
    dates = pd.concat(
        [
            uncaptured.assign(evidence="oos_residual_event_calendar").loc[
                :, ["day", "side_name", "archetype_policy_key", "uncaptured_reason", "evidence"]
            ],
            precoverage.assign(evidence="retrospective_champion_calendar").loc[
                :, ["day", "side_name", "archetype_policy_key", "uncaptured_reason", "evidence"]
            ],
        ],
        ignore_index=True,
    ).sort_values(["day", "side_name", "archetype_policy_key"], kind="stable")
    dates.to_csv(args.output_dir / "uncaptured_event_dates.csv", index=False)
    date_summary = (
        dates.groupby("day", observed=True, dropna=False)
        .agg(
            cells=("uncaptured_reason", "size"),
            sides=("side_name", lambda values: "|".join(sorted(set(map(str, values))))),
            archetypes=(
                "archetype_policy_key",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            reasons=(
                "uncaptured_reason",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            evidence=("evidence", lambda values: "|".join(sorted(set(map(str, values))))),
        )
        .reset_index()
        .sort_values("day", kind="stable")
    )
    date_summary.to_csv(
        args.output_dir / "uncaptured_event_date_summary.csv", index=False
    )
    adverse_dates = (
        uncaptured.loc[uncaptured["adverse_event_rows"].gt(0)]
        .groupby("day", observed=True, dropna=False)
        .agg(
            cells=("uncaptured_reason", "size"),
            adverse_event_rows=("adverse_event_rows", "sum"),
            sides=("side_name", lambda values: "|".join(sorted(set(map(str, values))))),
            archetypes=(
                "archetype_policy_key",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            reasons=(
                "uncaptured_reason",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
        .sort_values(
            ["adverse_event_rows", "day"], ascending=[False, True], kind="stable"
        )
    )
    adverse_dates.to_csv(
        args.output_dir / "material_uncaptured_adverse_dates.csv", index=False
    )

    summary = {
        "schema": "meta_residual_extreme_local_uncaptured_events_v1",
        "audit_start": "2025-01-01",
        "audit_end_exclusive": str(audit_end),
        "state_oos_coverage_start": str(start),
        "tail_scope": "side_x_archetype",
        "tail_quantile": threshold,
        "alpha_down": alpha_down,
        "event_cells": int(len(event_cells)),
        "material_event_cells": int(event_cells["material_extreme"].sum()),
        "material_uncaptured_cells": int(len(uncaptured)),
        "material_uncaptured_dates": int(uncaptured["day"].nunique()),
        "material_uncaptured_adverse_dates": int(len(adverse_dates)),
        "precoverage_uncaptured_cells": int(len(precoverage)),
        "precoverage_uncaptured_dates": int(precoverage["day"].nunique()),
        "reason_counts": dates["uncaptured_reason"].value_counts().to_dict(),
        "evidence_warning": (
            "January-March 2025 dates come from the retrospective frozen-champion "
            "calendar because leakage-safe residual-state OOS outputs begin in April."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
