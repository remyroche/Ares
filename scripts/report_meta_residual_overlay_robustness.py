#!/usr/bin/env python3
"""Robustness, placebo, and positive-preservation report for residual overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _selection_mask,
)

DIRTY = "meta_resid_arch_expected_dirty_positive"
HIT = "meta_resid_arch_expected_hit_surprise"


def _score(frame: pd.DataFrame, hit_alpha: float, dirty_lambda: float) -> np.ndarray:
    base = (
        pd.to_numeric(frame["score_lifecycle_only"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
    )
    hit = (
        pd.to_numeric(frame[HIT], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    dirty = (
        pd.to_numeric(frame[DIRTY], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return np.clip(
        base + np.float32(hit_alpha) * hit - np.float32(dirty_lambda) * dirty, 0.0, 1.0
    )


def _top10(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return _selection_mask(frame, score_col, 0.10, ["calendar_month", "side_name"])


def _metrics(frame: pd.DataFrame, score_col: str, arm: str) -> dict[str, Any]:
    mask = _top10(frame, score_col)
    selected = frame.loc[mask]
    weekly = []
    for _, group in frame.groupby("week_start", sort=True):
        local = group.loc[_top10(group, score_col)]
        weekly.append(
            float(pd.to_numeric(local["ev_after_1pct"], errors="coerce").mean())
        )
    return {
        "arm": arm,
        "selected_rows": int(mask.sum()),
        "mean_ev_after_1pct": float(
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
        ),
        "clean_exec_precision": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
        "dirty_positive_rate": float(
            pd.to_numeric(selected["dirty_positive"], errors="coerce").mean()
        ),
        "first_touch_bad_mae_rate": float(
            pd.to_numeric(selected["first_touch_bad_mae_1r"], errors="coerce").mean()
        ),
        "full_path_bad_mae_rate": float(
            pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(selected["timeout"], errors="coerce").mean()
        ),
        "worst_week_ev": float(np.nanmin(weekly)),
        "positive_weeks": int(np.sum(np.asarray(weekly) > 0.0)),
        "weeks": int(len(weekly)),
    }


def _shuffle_within_blocks(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, idx in out.groupby(
        ["calendar_month", "side_name", "archetype_policy_key"],
        sort=False,
        dropna=False,
    ).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        shuffled = rng.permutation(pos)
        out.loc[pos, [HIT, DIRTY]] = out.loc[shuffled, [HIT, DIRTY]].to_numpy()
    return out


def _time_shift(frame: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    source = frame[
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key", HIT, DIRTY]
    ].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True) + pd.Timedelta(
        days=days
    )
    source = source.drop_duplicates(
        ["__ts__", "__symbol__", "side_name", "archetype_policy_key"], keep="last"
    )
    out = frame.drop(columns=[HIT, DIRTY]).merge(
        source,
        on=["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_one",
    )
    out[HIT] = pd.to_numeric(out[HIT], errors="coerce").fillna(0.0)
    out[DIRTY] = pd.to_numeric(out[DIRTY], errors="coerce").fillna(0.0)
    return out


def _weekly_bootstrap(
    frame: pd.DataFrame, seed: int = 52, draws: int = 10_000
) -> dict[str, float]:
    values = []
    for _, group in frame.groupby("week_start", sort=True):
        life = group.loc[_top10(group, "score_lifecycle_only")]
        over = group.loc[_top10(group, "score_overlay")]
        values.append(
            float(pd.to_numeric(over["ev_after_1pct"], errors="coerce").mean())
            - float(pd.to_numeric(life["ev_after_1pct"], errors="coerce").mean())
        )
    values_arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values_arr[
        rng.integers(0, len(values_arr), size=(draws, len(values_arr)))
    ].mean(axis=1)
    return {
        "weekly_delta_mean": float(values_arr.mean()),
        "weekly_delta_ci025": float(np.quantile(sampled, 0.025)),
        "weekly_delta_ci975": float(np.quantile(sampled, 0.975)),
        "weekly_delta_positive_probability": float(np.mean(sampled > 0.0)),
        "weeks": int(len(values_arr)),
    }


def _positive_preservation(frame: pd.DataFrame) -> dict[str, float]:
    work = frame.copy()
    life_mask = _top10(work, "score_lifecycle_only")
    overlay_mask = _top10(work, "score_overlay")
    work["life_selected"] = life_mask
    work["overlay_selected"] = overlay_mask
    work["life_surprise"] = pd.to_numeric(
        work["clean_exec"], errors="coerce"
    ) - pd.to_numeric(work["hit_prob_current_reference"], errors="coerce")
    day = (
        work.loc[life_mask]
        .groupby(pd.to_datetime(work.loc[life_mask, "__ts__"], utc=True).dt.floor("D"))[
            "life_surprise"
        ]
        .mean()
    )
    threshold = float(day.quantile(0.90))
    positive_days = set(day[day.ge(max(threshold, 0.0))].index)
    event_rows = work[
        pd.to_datetime(work["__ts__"], utc=True).dt.floor("D").isin(positive_days)
    ]
    life_event = event_rows["life_selected"]
    retained = event_rows["life_selected"] & event_rows["overlay_selected"]
    suppressed = event_rows["life_selected"] & ~event_rows["overlay_selected"]
    return {
        "positive_event_days": int(len(positive_days)),
        "lifecycle_top10_positive_event_rows": int(life_event.sum()),
        "overlay_retained_rows": int(retained.sum()),
        "top10_positive_event_trade_retention": float(
            retained.sum() / max(life_event.sum(), 1)
        ),
        "positive_opportunity_false_suppression_rate": float(
            suppressed.sum() / max(life_event.sum(), 1)
        ),
        "suppressed_positive_event_mean_ev": float(
            pd.to_numeric(
                event_rows.loc[suppressed, "ev_after_1pct"], errors="coerce"
            ).mean()
        ),
    }


def main() -> None:
    root = DEFAULT_OUT_DIR
    overlay_dir = root / "lifecycle_residual_overlay"
    frame = pd.read_parquet(overlay_dir / "oos_predictions.parquet")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["week_start"] = pd.to_datetime(frame["week_start"], utc=True)
    manifest = json.loads((overlay_dir / "manifest.json").read_text())
    hit_alpha = float(manifest["hit_alpha"])
    dirty_lambda = float(manifest["dirty_lambda"])
    frame["score_overlay"] = _score(frame, hit_alpha, dirty_lambda)
    frame["score_positive_only"] = _score(frame, hit_alpha, 0.0)
    frame["score_negative_only"] = _score(frame, 0.0, dirty_lambda)
    arms = [
        _metrics(frame, "score_lifecycle_only", "lifecycle_only"),
        _metrics(frame, "score_positive_only", "positive_residual_only"),
        _metrics(frame, "score_negative_only", "negative_residual_only"),
        _metrics(frame, "score_overlay", "positive_and_negative_residual"),
    ]
    for seed in range(10):
        shuffled = _shuffle_within_blocks(frame, 10_000 + seed)
        shuffled["score_placebo"] = _score(shuffled, hit_alpha, dirty_lambda)
        arms.append(_metrics(shuffled, "score_placebo", f"shuffle_placebo_{seed:02d}"))
    shifted = _time_shift(frame, days=7)
    shifted["score_placebo"] = _score(shifted, hit_alpha, dirty_lambda)
    arms.append(_metrics(shifted, "score_placebo", "temporal_shift_7d_placebo"))
    metrics = pd.DataFrame(arms)
    metrics.to_csv(overlay_dir / "robustness_placebo_metrics.csv", index=False)

    life = _top10(frame, "score_lifecycle_only")
    overlay = _top10(frame, "score_overlay")
    overlap = {
        "lifecycle_selected": int(life.sum()),
        "overlay_selected": int(overlay.sum()),
        "intersection": int((life & overlay).sum()),
        "union": int((life | overlay).sum()),
        "top10_overlap_rate": float((life & overlay).sum() / max(life.sum(), 1)),
        "top10_jaccard": float((life & overlay).sum() / max((life | overlay).sum(), 1)),
        "rank_turnover": float(1.0 - (life & overlay).sum() / max(life.sum(), 1)),
    }
    report = {
        "schema": "meta_residual_overlay_robustness_v1",
        "overlay": {"hit_alpha": hit_alpha, "dirty_lambda": dirty_lambda},
        "weekly_block_bootstrap": _weekly_bootstrap(frame),
        "positive_preservation": _positive_preservation(frame),
        "rank_overlap": overlap,
        "placebo_mean_ev": float(
            metrics[metrics["arm"].str.startswith("shuffle_placebo")][
                "mean_ev_after_1pct"
            ].mean()
        ),
        "real_overlay_ev": float(
            metrics.loc[
                metrics["arm"].eq("positive_and_negative_residual"),
                "mean_ev_after_1pct",
            ].iloc[0]
        ),
    }
    (overlay_dir / "robustness_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
