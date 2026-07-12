#!/usr/bin/env python3
"""Robustness and placebo suite for the frozen local residual overlay."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
)
from extreme_price_movements.meta_residual_overlay import (  # noqa: E402
    DIRTY_FEATURE,
    HIT_FEATURE,
    ResidualOverlayState,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _calibrate,  # noqa: E402
    _selection_mask,
)

ARM = "lifecycle_residual_local_overlay"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in frame.columns
        ],
        errors="ignore",
    )


def _apply(
    frame: pd.DataFrame, state: ResidualOverlayState, score_col: str = "score_placebo"
) -> pd.DataFrame:
    out = frame.copy()
    out[score_col] = state.transform(
        _safe_frame(out),
        pd.to_numeric(out["score_lifecycle_only"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32),
    )
    return out


def _top(frame: pd.DataFrame, score_col: str, fraction: float = 0.10) -> pd.Series:
    return _selection_mask(frame, score_col, fraction, ["calendar_month", "side_name"])


def _metrics(
    frame: pd.DataFrame, score_col: str, name: str, fraction: float = 0.10
) -> dict[str, Any]:
    mask = _top(frame, score_col, fraction)
    selected = frame.loc[mask]
    weekly: list[float] = []
    for _, group in frame.groupby("week_start", sort=True):
        local = group.loc[_top(group, score_col, fraction)]
        weekly.append(
            float(pd.to_numeric(local["ev_after_1pct"], errors="coerce").mean())
        )
    return {
        "arm": name,
        "fraction": float(fraction),
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


def _shuffle_features(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, idx in out.groupby(
        ["calendar_month", "side_name", "archetype_policy_key"], sort=False
    ).groups.items():
        pos = np.asarray(list(idx), dtype=np.int64)
        out.loc[pos, [HIT_FEATURE, DIRTY_FEATURE]] = out.loc[
            rng.permutation(pos), [HIT_FEATURE, DIRTY_FEATURE]
        ].to_numpy()
    return out


def _shift_features(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    source = frame[KEYS + [HIT_FEATURE, DIRTY_FEATURE]].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True) + pd.Timedelta(
        hours=hours
    )
    source = source.drop_duplicates(KEYS, keep="last")
    out = frame.drop(columns=[HIT_FEATURE, DIRTY_FEATURE]).merge(
        source, on=KEYS, how="left", validate="one_to_one"
    )
    for name in (HIT_FEATURE, DIRTY_FEATURE):
        out[name] = (
            pd.to_numeric(out[name], errors="coerce").fillna(0.0).astype(np.float32)
        )
    return out


def _randomize_archetypes(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, idx in out.groupby(
        ["calendar_month", "side_name"], sort=False
    ).groups.items():
        pos = np.asarray(list(idx), dtype=np.int64)
        out.loc[pos, "archetype_policy_key"] = rng.permutation(
            out.loc[pos, "archetype_policy_key"].astype(str).to_numpy()
        )
    return out


def _noise_features(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for name in (HIT_FEATURE, DIRTY_FEATURE):
        values = pd.to_numeric(out[name], errors="coerce")
        out[name] = rng.normal(
            float(values.mean()), max(float(values.std(ddof=0)), 1e-6), len(out)
        ).astype(np.float32)
    return out


def _weekly_bootstrap(
    frame: pd.DataFrame, draws: int = 10_000, seed: int = 20260711
) -> dict[str, Any]:
    deltas: list[float] = []
    for _, group in frame.groupby("week_start", sort=True):
        life = group.loc[_top(group, "score_lifecycle_only")]
        local = group.loc[_top(group, "score_alternative")]
        deltas.append(
            float(pd.to_numeric(local["ev_after_1pct"], errors="coerce").mean())
            - float(pd.to_numeric(life["ev_after_1pct"], errors="coerce").mean())
        )
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(
        axis=1
    )
    return {
        "weeks": int(len(values)),
        "mean_delta": float(values.mean()),
        "ci025": float(np.quantile(samples, 0.025)),
        "ci975": float(np.quantile(samples, 0.975)),
        "positive_probability": float(np.mean(samples > 0.0)),
    }


def _positive_preservation(frame: pd.DataFrame) -> dict[str, Any]:
    life = _top(frame, "score_lifecycle_only")
    local = _top(frame, "score_alternative")
    day = frame.loc[life].copy()
    day["date"] = pd.to_datetime(day["__ts__"], utc=True).dt.floor("D")
    by_day = day.groupby("date")["clean_exec"].mean()
    positive_days = set(by_day[by_day.ge(by_day.quantile(0.90))].index)
    event = frame[
        pd.to_datetime(frame["__ts__"], utc=True).dt.floor("D").isin(positive_days)
    ]
    life_event = life.loc[event.index]
    retained = life.loc[event.index] & local.loc[event.index]
    suppressed = life.loc[event.index] & ~local.loc[event.index]
    return {
        "positive_event_days": int(len(positive_days)),
        "lifecycle_rows": int(life_event.sum()),
        "retained_rows": int(retained.sum()),
        "retention_rate": float(retained.sum() / max(life_event.sum(), 1)),
        "suppressed_rows_mean_ev": float(
            pd.to_numeric(
                event.loc[suppressed, "ev_after_1pct"], errors="coerce"
            ).mean()
        ),
    }


def _score_with_archetype_assignment(
    frame: pd.DataFrame,
    state: ResidualOverlayState,
    assigned_archetype: np.ndarray,
) -> np.ndarray:
    side = frame[state.side_col].astype(str).str.lower().to_numpy()
    keys = np.char.add(
        np.char.add(side.astype(str), "||"), assigned_archetype.astype(str)
    )
    hit = pd.to_numeric(frame[state.hit_feature], errors="coerce").to_numpy(
        dtype=np.float32
    )
    dirty = pd.to_numeric(frame[state.dirty_feature], errors="coerce").to_numpy(
        dtype=np.float32
    )
    base = (
        pd.to_numeric(frame["score_lifecycle_only"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
    )
    hit_mean_map = {key: value.hit_mean for key, value in state.group_stats.items()}
    hit_std_map = {
        key: max(value.hit_std, state.min_std)
        for key, value in state.group_stats.items()
    }
    dirty_mean_map = {key: value.dirty_mean for key, value in state.group_stats.items()}
    dirty_std_map = {
        key: max(value.dirty_std, state.min_std)
        for key, value in state.group_stats.items()
    }
    key_series = pd.Series(keys, copy=False)
    hit_mean = (
        key_series.map(hit_mean_map)
        .fillna(state.global_stats.hit_mean)
        .to_numpy(dtype=np.float32)
    )
    hit_std = (
        key_series.map(hit_std_map)
        .fillna(state.global_stats.hit_std)
        .to_numpy(dtype=np.float32)
    )
    dirty_mean = (
        key_series.map(dirty_mean_map)
        .fillna(state.global_stats.dirty_mean)
        .to_numpy(dtype=np.float32)
    )
    dirty_std = (
        key_series.map(dirty_std_map)
        .fillna(state.global_stats.dirty_std)
        .to_numpy(dtype=np.float32)
    )
    hit = np.where(np.isfinite(hit), hit, hit_mean)
    dirty = np.where(np.isfinite(dirty), dirty, dirty_mean)
    hit_z = np.clip(
        (hit - hit_mean) / np.maximum(hit_std, state.min_std),
        -state.z_clip,
        state.z_clip,
    )
    dirty_z = np.clip(
        (dirty - dirty_mean) / np.maximum(dirty_std, state.min_std),
        -state.z_clip,
        state.z_clip,
    )
    return np.clip(
        base
        + np.float32(state.hit_alpha) * hit
        - np.float32(state.dirty_lambda) * dirty
        + np.float32(state.local_hit_alpha) * hit_z
        - np.float32(state.local_dirty_lambda) * dirty_z,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)


def _fast_top10_mask(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    labels = frame["calendar_month"].astype(str) + "||" + frame["side_name"].astype(str)
    mask = np.zeros(len(frame), dtype=bool)
    for positions in labels.groupby(labels, sort=False).groups.values():
        idx = np.asarray(positions, dtype=np.int64)
        keep = len(idx) - int(np.ceil(0.90 * len(idx))) + 1
        order = np.argsort(scores[idx], kind="stable")
        mask[idx[order[-keep:]]] = True
    return mask


def _calendar_placebo_metric(
    frame: pd.DataFrame,
    scores: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    mask = _fast_top10_mask(frame, scores)
    selected = pd.DataFrame(
        {
            "date": pd.to_datetime(frame.loc[mask, "__ts__"], utc=True)
            .dt.floor("D")
            .to_numpy(),
            "side": frame.loc[mask, "side_name"].astype(str).to_numpy(),
            "archetype": frame.loc[mask, "archetype_policy_key"].astype(str).to_numpy(),
            "surprise": (
                pd.to_numeric(frame.loc[mask, "clean_exec"], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                - probabilities[mask]
            ),
        }
    )
    values: list[float] = []
    for _, group in selected.groupby(["side", "archetype"], sort=False):
        daily = group.groupby("date", sort=True)["surprise"].mean()
        if len(daily) >= 3:
            value = float(daily.autocorr(1))
            if np.isfinite(value):
                values.append(abs(value))
    return {
        "mean_ev_after_1pct": float(
            pd.to_numeric(frame.loc[mask, "ev_after_1pct"], errors="coerce").mean()
        ),
        "clean_exec_precision": float(
            pd.to_numeric(frame.loc[mask, "clean_exec"], errors="coerce").mean()
        ),
        "mean_abs_signed_surprise_autocorr_lag1": float(np.mean(values)),
        "selected_rows": int(mask.sum()),
    }


def _archetype_identity_placebo(
    frame: pd.DataFrame,
    state: ResidualOverlayState,
    calibrator: Any,
    *,
    seeds: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual_archetype = frame[state.archetype_col].astype(str).to_numpy()
    actual_scores = _score_with_archetype_assignment(frame, state, actual_archetype)
    actual_probabilities = _calibrate(calibrator, pd.Series(actual_scores)).astype(
        np.float32
    )
    actual = _calendar_placebo_metric(frame, actual_scores, actual_probabilities)
    rows: list[dict[str, Any]] = [
        {"arm": "actual_archetype_mapping", "seed": -1, **actual}
    ]
    block_positions = [
        np.asarray(value, dtype=np.int64)
        for value in frame.groupby(
            ["calendar_month", "side_name"], sort=False
        ).groups.values()
    ]
    for seed in range(int(seeds)):
        rng = np.random.default_rng(20260711 + seed)
        assigned = actual_archetype.copy()
        for idx in block_positions:
            assigned[idx] = rng.permutation(assigned[idx])
        scores = _score_with_archetype_assignment(frame, state, assigned)
        probabilities = _calibrate(calibrator, pd.Series(scores)).astype(np.float32)
        rows.append(
            {
                "arm": "randomized_archetype_mapping",
                "seed": seed,
                **_calendar_placebo_metric(frame, scores, probabilities),
            }
        )
    results = pd.DataFrame(rows)
    random = results[results["arm"].eq("randomized_archetype_mapping")]
    summary = {
        "seeds": int(seeds),
        "actual": actual,
        "randomized_median_ev": float(random["mean_ev_after_1pct"].median()),
        "randomized_median_clean": float(random["clean_exec_precision"].median()),
        "randomized_median_mean_abs_autocorr": float(
            random["mean_abs_signed_surprise_autocorr_lag1"].median()
        ),
        "probability_actual_ev_exceeds_randomized": float(
            (actual["mean_ev_after_1pct"] > random["mean_ev_after_1pct"]).mean()
        ),
        "probability_actual_autocorr_below_randomized": float(
            (
                actual["mean_abs_signed_surprise_autocorr_lag1"]
                < random["mean_abs_signed_surprise_autocorr_lag1"]
            ).mean()
        ),
        "interpretation": (
            "The local mapping is retained for calendar decorrelation only when actual "
            "side-by-archetype identity beats block-randomized identity on autocorrelation. "
            "Economic lift is attributed separately to the causal residual prior."
        ),
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=ARM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = str(args.arm)
    root = DEFAULT_OUT_DIR
    arm_dir = root / arm
    frame = pd.read_parquet(arm_dir / "oos_predictions.parquet")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["week_start"] = frame["__ts__"].dt.floor("D") - pd.to_timedelta(
        frame["__ts__"].dt.weekday,
        unit="D",
    )
    state: ResidualOverlayState = joblib.load(arm_dir / "residual_overlay_state.joblib")
    calibrator = joblib.load(arm_dir / "hit_calibrator.joblib")
    rows = [
        _metrics(frame, "score_lifecycle_only", "lifecycle_only", fraction)
        for fraction in (0.05, 0.10, 0.15, 0.20)
    ]
    rows.extend(
        _metrics(frame, "score_alternative", arm, fraction)
        for fraction in (0.05, 0.10, 0.15, 0.20)
    )
    positive_state = replace(state, dirty_lambda=0.0, local_dirty_lambda=0.0)
    negative_state = replace(state, hit_alpha=0.0, local_hit_alpha=0.0)
    rows.append(
        _metrics(_apply(frame, positive_state), "score_placebo", "positive_only")
    )
    rows.append(
        _metrics(_apply(frame, negative_state), "score_placebo", "negative_only")
    )
    for seed in range(10):
        rows.append(
            _metrics(
                _apply(_shuffle_features(frame, 10_000 + seed), state),
                "score_placebo",
                f"shuffle_{seed:02d}",
            )
        )
    for hours in (-168, -48, -24, 24, 48, 168):
        rows.append(
            _metrics(
                _apply(_shift_features(frame, hours), state),
                "score_placebo",
                f"shift_{hours:+d}h",
            )
        )
    rows.append(
        _metrics(
            _apply(_randomize_archetypes(frame, 52), state),
            "score_placebo",
            "randomized_archetypes",
        )
    )
    rows.append(
        _metrics(
            _apply(_noise_features(frame, 53), state),
            "score_placebo",
            "matched_noise_features",
        )
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(arm_dir / "robustness_extended_metrics.csv", index=False)
    archetype_placebos, archetype_placebo_summary = _archetype_identity_placebo(
        frame,
        state,
        calibrator,
    )
    identity_applicable = bool(
        abs(float(state.local_hit_alpha)) > 0.0
        or abs(float(state.local_dirty_lambda)) > 0.0
    )
    archetype_placebo_summary["applicable"] = identity_applicable
    if not identity_applicable:
        archetype_placebo_summary["interpretation"] = (
            "Not applicable: local side-by-archetype normalization coefficients are zero, "
            "so relabeling archetypes cannot change the overlay score."
        )
    archetype_placebos.to_csv(arm_dir / "archetype_identity_placebo.csv", index=False)
    actual = metrics[(metrics["arm"].eq(arm)) & (metrics["fraction"].eq(0.10))].iloc[0]
    placebo_names = ["matched_noise_features"]
    if identity_applicable:
        placebo_names.append("randomized_archetypes")
    placebo = metrics[
        metrics["arm"].str.startswith(("shuffle_", "shift_"))
        | metrics["arm"].isin(placebo_names)
    ]
    report = {
        "schema": "meta_residual_local_overlay_robustness_v1",
        "weekly_block_bootstrap": _weekly_bootstrap(frame),
        "positive_preservation": _positive_preservation(frame),
        "actual_top10_ev": float(actual["mean_ev_after_1pct"]),
        "max_placebo_top10_ev": float(placebo["mean_ev_after_1pct"].max()),
        "actual_beats_all_placebos": bool(
            float(actual["mean_ev_after_1pct"])
            > float(placebo["mean_ev_after_1pct"].max())
        ),
        "archetype_identity_placebo": archetype_placebo_summary,
        "rank_thresholds": metrics[
            metrics["arm"].isin(["lifecycle_only", arm])
        ].to_dict(orient="records"),
    }
    (arm_dir / "robustness_extended_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
