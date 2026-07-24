#!/usr/bin/env python3
"""Sequentially refine the v9 local adverse-state tail threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_residual_extreme_local_champion_overlay import (  # noqa: E402
    FEATURES,
    KEYS,
    PARENT,
    _adjust_rank,
    _autocorrelation_report,
    _breakdown,
    _composite,
    _fit_references,
    _metric_row,
)

ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
INITIAL_THRESHOLDS = (0.925, 0.90, 0.875)


def _train_objective(
    frame: pd.DataFrame,
    adjusted: np.ndarray,
    parent_count: int,
) -> dict[str, float]:
    selected = adjusted >= 0.90
    work = frame.loc[selected].copy()
    work["month"] = work["__ts__"].dt.strftime("%Y-%m")
    monthly_ev = work.groupby("month", observed=True)["ev_after_1pct"].mean()
    monthly_clean = work.groupby("month", observed=True)["clean_exec"].mean()
    activity = float(selected.sum()) / max(float(parent_count), 1.0)
    objective = (
        float(monthly_ev.mean())
        - 0.5 * float(monthly_ev.std(ddof=0))
        + 0.25 * float(monthly_ev.min())
        + 0.002 * float(monthly_clean.mean())
        - 0.01 * abs(np.log(max(activity, 1e-8)))
    )
    return {
        "objective": objective,
        "activity_ratio": activity,
        "selected_rows": int(selected.sum()),
        "mean_month_ev": float(monthly_ev.mean()),
        "worst_month_ev": float(monthly_ev.min()),
    }


def _select_alpha(
    train: pd.DataFrame,
    adverse: np.ndarray,
    positive: np.ndarray,
    threshold: float,
) -> tuple[float, pd.DataFrame]:
    parent_rank = pd.to_numeric(train["historical_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    parent_count = int(np.sum(parent_rank >= 0.90))
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        adjusted = _adjust_rank(
            parent_rank,
            adverse,
            positive,
            threshold=threshold,
            alpha_down=alpha,
            alpha_up=0.0,
        )
        row = _train_objective(train, adjusted, parent_count)
        row.update({"threshold": threshold, "alpha_down": alpha})
        rows.append(row)
    search = pd.DataFrame(rows)
    eligible = search.loc[search["activity_ratio"].ge(0.90)]
    if eligible.empty:
        eligible = search
    best = eligible.sort_values(
        ["objective", "activity_ratio", "alpha_down"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    return float(best["alpha_down"]), search


def _load_frames(
    state_path: Path,
    rank_cache: Path,
    parent_eval_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = pd.read_parquet(
        state_path,
        columns=[
            *KEYS,
            "ev_after_1pct",
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            *FEATURES,
        ],
    )
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    state = state.drop_duplicates(KEYS, keep="last")
    rank = pd.read_parquet(rank_cache)
    rank["__ts__"] = pd.to_datetime(rank["__ts__"], utc=True)
    train = state.loc[
        state["__ts__"].ge("2025-04-01") & state["__ts__"].lt("2026-04-01")
    ].merge(rank, on=KEYS, how="inner", validate="one_to_one")

    eval_columns = [
        *KEYS,
        "historical_rank_adjusted",
        "hit_prob_adjusted",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    valid = pd.read_parquet(parent_eval_path, columns=eval_columns).rename(
        columns={
            "historical_rank_adjusted": "historical_rank",
            "hit_prob_adjusted": "hit_probability",
        }
    )
    valid["__ts__"] = pd.to_datetime(valid["__ts__"], utc=True)
    valid = valid.loc[valid["__ts__"].ge("2026-04-01") & valid["__ts__"].lt("2026-07-01")]
    valid = valid.merge(
        state.loc[:, [*KEYS, *FEATURES]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v9-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9"
        ),
    )
    parser.add_argument(
        "--state-artifact",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3/oos_residual_event_states.parquet"
        ),
    )
    parser.add_argument(
        "--rank-cache",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"
        ),
    )
    parser.add_argument(
        "--parent-eval",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_"
            "globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_tail_refinement_20260712_v1"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, valid = _load_frames(args.state_artifact, args.rank_cache, args.parent_eval)
    catalog = pd.read_csv(args.v9_dir / "selected_local_features_strict.csv")
    references = _fit_references(train, catalog, 1)
    train_adverse = _composite(train, references, "adverse")
    train_positive = _composite(train, references, "positive")
    valid_adverse = _composite(valid, references, "adverse")
    valid_positive = _composite(valid, references, "positive")
    parent_rank = pd.to_numeric(valid["historical_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    parent_mask = parent_rank >= 0.90
    parent_selector = PARENT
    parent_metric = _metric_row(valid, parent_mask, parent_selector)
    _, parent_ac_report = _autocorrelation_report(
        valid, {parent_selector: parent_mask}
    )
    parent_ac = float(parent_ac_report["signed_surprise_autocorr_lag1"].abs().mean())
    parent_metric["mean_abs_signed_surprise_autocorr_lag1"] = parent_ac

    evaluated: dict[float, dict[str, Any]] = {}
    search_parts: list[pd.DataFrame] = []

    def evaluate(threshold: float) -> dict[str, Any]:
        if threshold in evaluated:
            return evaluated[threshold]
        alpha, search = _select_alpha(
            train, train_adverse, train_positive, threshold
        )
        search_parts.append(search)
        adjusted = _adjust_rank(
            parent_rank,
            valid_adverse,
            valid_positive,
            threshold=threshold,
            alpha_down=alpha,
            alpha_up=0.0,
        )
        mask = adjusted >= 0.90
        selector = f"{PARENT}_local_tail_{threshold:.5f}"
        metric = _metric_row(valid, mask, selector)
        _, ac_report = _autocorrelation_report(valid, {selector: mask})
        metric["mean_abs_signed_surprise_autocorr_lag1"] = float(
            ac_report["signed_surprise_autocorr_lag1"].abs().mean()
        )
        metric.update(
            {
                "threshold": threshold,
                "alpha_down": alpha,
                "dropped_rows": int(np.sum(parent_mask & ~mask)),
                "added_rows": int(np.sum(~parent_mask & mask)),
                "mask": mask,
            }
        )
        evaluated[threshold] = metric
        return metric

    previous = evaluate(0.95)
    stop_pair: tuple[float, float] | None = None
    sequence = [0.95]
    for threshold in INITIAL_THRESHOLDS:
        current = evaluate(threshold)
        sequence.append(threshold)
        ev_improved = current["mean_ev_after_1pct"] > previous["mean_ev_after_1pct"]
        ac_improved = (
            current["mean_abs_signed_surprise_autocorr_lag1"]
            < previous["mean_abs_signed_surprise_autocorr_lag1"]
        )
        if not (ev_improved and ac_improved):
            stop_pair = (float(previous["threshold"]), threshold)
            break
        previous = current
    midpoint = None
    if stop_pair is not None:
        midpoint = float(np.mean(stop_pair))
        evaluate(midpoint)
        sequence.append(midpoint)

    rows = []
    breakdowns = []
    for threshold in sequence:
        metric = dict(evaluated[threshold])
        mask = metric.pop("mask")
        rows.append(metric)
        breakdowns.append(_breakdown(valid, mask, metric["selector"]))
    rows.insert(0, parent_metric)
    pd.DataFrame(rows).to_csv(args.output_dir / "tail_refinement_summary.csv", index=False)
    pd.concat(search_parts, ignore_index=True).to_csv(
        args.output_dir / "train_alpha_search.csv", index=False
    )
    pd.concat(breakdowns, ignore_index=True).to_csv(
        args.output_dir / "tail_refinement_breakdowns.csv", index=False
    )
    manifest = {
        "schema": "meta_residual_extreme_local_tail_refinement_v1",
        "parent": PARENT,
        "sequence": sequence,
        "stop_pair": stop_pair,
        "midpoint": midpoint,
        "stopping_rule": (
            "continue only while both OOS EV/trade rises and mean absolute signed "
            "surprise autocorrelation falls; then evaluate the midpoint"
        ),
        "training_contract": (
            "feature catalog frozen from v9; alpha selected on chronological meta OOF "
            "rows through March 2026; threshold stopping uses April-June validation"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
