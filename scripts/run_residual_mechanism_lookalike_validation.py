#!/usr/bin/env python3
"""Chronologically validate train-derived historical mechanism lookalikes.

Each arm is local to a side x archetype and mechanism family.  At each fold it
uses only adverse blocks that began before the fold, derives the family from
their causal onset state, matches benign block starts, fits a small regularized
LGBM, freezes a top-five-percent threshold from *training* scores, and scores
the next chronological interval.  The output is research-only.

The focal residual calendar is used solely to report coverage of the specified
hard blocks.  It never contributes to lookalike selection, feature screening,
model training, or threshold fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from extreme_price_movements.residual_event_block_taxonomy import (
    BlockTaxonomyConfig,
    MECHANISM_FAMILIES,
    annotate_onset_mechanism_profiles,
    attach_event_blocks,
    build_block_taxonomy,
    matched_benign_block_controls,
)
from scripts.report_residual_event_block_taxonomy import (
    _load_calendar,
    _load_daily_state,
    _overlay_event_calendar,
)
from scripts.run_residual_block_family_detector import (
    _daily_start_features,
    _family_train_samples,
    _phase_event_days,
    _screen_features,
)


KEYS = ["day", "side_name", "archetype_policy_key"]
DEFAULT_FOLDS = (
    ("2025-05-15", "2025-08-01"),
    ("2025-08-01", "2025-11-01"),
    ("2025-11-01", "2026-02-01"),
    ("2026-02-01", "2026-05-01"),
    ("2026-05-01", "2026-07-01"),
)


def _parse_group(value: str) -> tuple[str, str]:
    side, separator, archetype = value.partition("::")
    if not separator or not side or not archetype:
        raise ValueError(f"Expected side::archetype, got {value!r}")
    return side, archetype


def _parse_fold(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, separator, end = value.partition("::")
    if not separator:
        raise ValueError(f"Expected START::END fold, got {value!r}")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if start_ts >= end_ts:
        raise ValueError(f"Invalid chronological fold {value!r}")
    return start_ts, end_ts


def _fill_scale(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    q25 = np.nanquantile(train, 0.25, axis=0)
    q75 = np.nanquantile(train, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    for values in (train, score):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        values -= median
        values /= scale
        np.clip(values, -8.0, 8.0, out=values)
    return train.astype(np.float32, copy=False), score.astype(np.float32, copy=False)


def _fit_predict(
    train: pd.DataFrame, score: pd.DataFrame, features: list[str], *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    x_train, x_score = _fill_scale(x_train, x_score)
    y = train["target"].to_numpy(np.int8)
    positives = max(int(y.sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.035,
            "max_depth": 2,
            "num_leaves": 4,
            "min_data_in_leaf": max(4, min(12, len(train) // 6)),
            "min_gain_to_split": 0.05,
            "lambda_l1": 3.0,
            "lambda_l2": 16.0,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.90,
            "bagging_freq": 1,
            "seed": seed,
            "num_threads": 1,
            "verbosity": -1,
            "force_col_wise": True,
        },
        lgb.Dataset(
            x_train,
            label=y,
            weight=np.where(y > 0, negatives / positives, 1.0).astype(np.float32),
            feature_name=features,
        ),
        num_boost_round=100,
    )
    return (
        np.asarray(model.predict(x_train), dtype=np.float32),
        np.asarray(model.predict(x_score), dtype=np.float32),
    )


def _frozen_threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    selected = frame["risk"].ge(threshold).to_numpy(bool)
    event = frame["event_start"].to_numpy(bool)
    precision = float(event[selected].mean()) if selected.any() else np.nan
    prevalence = float(event.mean()) if len(event) else np.nan
    return {
        "oos_days": int(len(frame)),
        "oos_event_starts": int(event.sum()),
        "selected_days": int(selected.sum()),
        "selected_rate": float(selected.mean()) if len(selected) else np.nan,
        "precision": precision,
        "fpr": float(selected[~event].mean()) if (~event).any() else np.nan,
        "lift": precision / max(prevalence, 1e-9) if np.isfinite(precision) else np.nan,
        "event_recall": float((selected & event).sum() / max(event.sum(), 1)),
    }


def _focus_events(audit: Path) -> pd.DataFrame:
    frame = pd.read_csv(audit)
    frame = frame.loc[~frame["legacy_calendar_status"].eq("fully_recognized")].copy()
    frame["event_start"] = pd.to_datetime(frame["event_start"], utc=True)
    frame["event_end"] = pd.to_datetime(frame["event_end"], utc=True)
    return frame.loc[:, [
        "event_start", "event_end", "side_name", "archetype_policy_key",
        "event_block", "legacy_calendar_status", "onset_primary_mechanism",
    ]]


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    def progress(message: str) -> None:
        if args.verbose:
            print(f"[lookalike-validation] {message}", flush=True)

    requested = list(dict.fromkeys(name for family in MECHANISM_FAMILIES.values() for name in family))
    progress("loading daily observable state")
    daily = _load_daily_state(args.state_artifact, requested)
    progress(f"daily state loaded rows={len(daily)}")
    calendar = _overlay_event_calendar(daily.loc[:, KEYS].copy(), args.event_calendar)
    start_features, feature_columns = _daily_start_features(daily)
    groups = [_parse_group(value) for value in args.group]
    group_frame = pd.DataFrame(groups, columns=["side_name", "archetype_policy_key"]).drop_duplicates()
    focus = _focus_events(args.focus_audit)
    folds = [_parse_fold(value) for value in args.fold] if args.fold else [
        _parse_fold(f"{start}::{end}") for start, end in DEFAULT_FOLDS
    ]
    reports: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    lookalikes: list[pd.DataFrame] = []
    config = BlockTaxonomyConfig(
        pre_days=2,
        post_days=1,
        min_reference_days=30,
        controls_per_block=args.controls_per_block,
        max_clusters=4,
        min_cluster_blocks=args.min_family_blocks,
    )

    for fold_index, (train_end, eval_end) in enumerate(folds):
        progress(f"fold={fold_index} train_end={train_end.date()} eval_end={eval_end.date()}")
        # The detector is local.  Taxonomy/control matching for unrelated
        # populations contributes no positive examples and becomes expensive
        # as the historical calendar expands.
        train_calendar = calendar.loc[calendar["day"].lt(train_end)].merge(
            group_frame, on=["side_name", "archetype_policy_key"], how="inner"
        )
        train_daily = daily.loc[daily["day"].lt(train_end)].merge(
            group_frame, on=["side_name", "archetype_policy_key"], how="inner"
        )
        taxonomy, _ = build_block_taxonomy(train_calendar, train_daily, config=config)
        taxonomy = annotate_onset_mechanism_profiles(taxonomy)
        progress(f"fold={fold_index} train_blocks={len(taxonomy)}")
        controls = matched_benign_block_controls(train_calendar, train_daily, taxonomy, config=config)
        train_events = attach_event_blocks(train_calendar)
        eval_events = attach_event_blocks(calendar.loc[
            calendar["day"].ge(train_end) & calendar["day"].lt(eval_end)
        ].merge(group_frame, on=["side_name", "archetype_policy_key"], how="inner"))
        for side, archetype in groups:
            progress(f"fold={fold_index} group={side}::{archetype}")
            local_taxonomy = taxonomy.loc[
                taxonomy["side_name"].eq(side)
                & taxonomy["archetype_policy_key"].eq(archetype)
                & taxonomy["onset_primary_mechanism"].ne("unavailable")
            ].copy()
            local_eval = start_features.loc[
                start_features["side_name"].eq(side)
                & start_features["archetype_policy_key"].eq(archetype)
                & start_features["day"].ge(train_end)
                & start_features["day"].lt(eval_end)
            ].copy()
            local_eval_events = eval_events.loc[
                eval_events["side_name"].eq(side)
                & eval_events["archetype_policy_key"].eq(archetype)
            ]
            event_days = _phase_event_days(local_eval_events, event_phase="onset")["day"]
            local_eval["event_start"] = local_eval["day"].isin(event_days)
            for family, family_blocks in local_taxonomy.groupby("onset_primary_mechanism", observed=True):
                samples = _family_train_samples(
                    taxonomy,
                    controls,
                    start_features,
                    train_events,
                    side=side,
                    archetype=archetype,
                    family=str(family),
                    family_column="onset_primary_mechanism",
                    event_phase="onset",
                )
                selected = _screen_features(samples, feature_columns, maximum=args.max_features) if not samples.empty else []
                support_ok = (
                    not local_eval.empty
                    and len(family_blocks) >= args.min_family_blocks
                    and int((samples.get("target", pd.Series(dtype=np.int8)) == 0).sum()) >= args.min_family_blocks
                    and len(selected) >= 2
                )
                base = {
                    "fold_start": train_end,
                    "fold_end": eval_end,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "mechanism_family": family,
                    "train_lookalike_blocks": int(len(family_blocks)),
                    "train_positive_days": int(samples.get("target", pd.Series(dtype=np.int8)).sum()),
                    "train_control_days": int((samples.get("target", pd.Series(dtype=np.int8)) == 0).sum()),
                    "features": "|".join(selected),
                }
                lookalikes.append(family_blocks.assign(
                    fold_start=train_end,
                    fold_end=eval_end,
                    mechanism_family=family,
                    lookalike_role="train_adverse_same_mechanism",
                ))
                if not support_ok:
                    reports.append({**base, "status": "insufficient_prior_lookalike_support"})
                    continue
                progress(
                    f"fold={fold_index} group={side}::{archetype} family={family} "
                    f"lookalikes={len(family_blocks)}"
                )
                train_score, eval_score = _fit_predict(
                    samples, local_eval, selected, seed=args.seed + fold_index
                )
                threshold = float(np.nanquantile(train_score, 0.95))
                local_score = local_eval.loc[:, ["day", "event_start"]].copy()
                local_score["risk"] = eval_score
                local_score["frozen_top05_threshold"] = threshold
                local_score["admit"] = local_score["risk"].ge(threshold)
                train_selected_rate = float(np.mean(train_score >= threshold))
                score_std = float(np.nanstd(train_score))
                unique_scores = int(len(np.unique(train_score[np.isfinite(train_score)])))
                oos_metrics = _frozen_threshold_metrics(local_score, threshold)
                if unique_scores < 2 or score_std <= 1e-7:
                    status = "degenerate_train_score"
                elif train_selected_rate > 0.10:
                    status = "coarse_train_tail"
                elif float(oos_metrics["selected_rate"]) > 0.15:
                    # This does not assert that the OOS period was benign; it
                    # says the arm cannot be assessed as a sparse, precise
                    # alert at its stated operating point.
                    status = "coarse_oos_activation"
                else:
                    status = "ok"
                local_score["arm_status"] = status
                reports.append({
                    **base,
                    "status": status,
                    "frozen_top05_threshold": threshold,
                    "train_score_std": score_std,
                    "train_unique_score_count": unique_scores,
                    "train_selected_rate": train_selected_rate,
                    **oos_metrics,
                })
                predictions.append(local_score.assign(
                    fold_start=train_end,
                    fold_end=eval_end,
                    side_name=side,
                    archetype_policy_key=archetype,
                    mechanism_family=family,
                ))

    report = pd.DataFrame(reports)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    lookalike_frame = pd.concat(lookalikes, ignore_index=True) if lookalikes else pd.DataFrame()
    report.to_csv(args.output / "chronological_family_detector_metrics.csv", index=False)
    prediction_frame.to_parquet(
        args.output / "chronological_family_detector_daily_predictions.parquet",
        index=False,
        compression="zstd",
    )
    lookalike_frame.to_csv(args.output / "historical_lookalike_blocks.csv", index=False)

    coverage_rows: list[dict[str, object]] = []
    for event in focus.itertuples(index=False):
        matching = prediction_frame.loc[
            prediction_frame.get("side_name", pd.Series(dtype=str)).eq(event.side_name)
            & prediction_frame.get("archetype_policy_key", pd.Series(dtype=str)).eq(event.archetype_policy_key)
            & prediction_frame.get("day", pd.Series(dtype="datetime64[ns, UTC]")).between(
                event.event_start - pd.Timedelta(days=args.warning_days),
                event.event_start - pd.Timedelta(days=1),
            )
        ] if not prediction_frame.empty else pd.DataFrame()
        if matching.empty:
            coverage_rows.append({
                **event._asdict(),
                "status": "no_eligible_prior_chronological_detector",
                "arms_available": 0,
                "arms_alerted": 0,
                "max_risk": np.nan,
            })
            continue
        coverage_rows.append({
            **event._asdict(),
            "status": "scored",
            "arms_available": int(matching["mechanism_family"].nunique()),
            "valid_arms_available": int(matching.loc[matching["arm_status"].eq("ok"), "mechanism_family"].nunique()),
            "arms_alerted": int(matching.loc[matching["admit"] & matching["arm_status"].eq("ok"), "mechanism_family"].nunique()),
            "max_valid_risk": float(matching.loc[matching["arm_status"].eq("ok"), "risk"].max())
            if matching["arm_status"].eq("ok").any() else np.nan,
            "alerting_families": "|".join(sorted(matching.loc[
                matching["admit"] & matching["arm_status"].eq("ok"), "mechanism_family"
            ].unique())),
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "focus_event_chronological_coverage.csv", index=False)
    summary = {
        "purpose": "research-only chronological historical-lookalike mechanism validation; inactive",
        "state_contract": "daily-open price/OI/funding/cross-sectional state only",
        "training_contract": "same side x archetype adverse onset blocks with train-derived causal onset mechanism labels versus matched benign controls",
        "threshold_contract": "95th percentile of each arm's training score; no OOS top-k thresholding",
        "focus_calendar_contract": "reporting only; no focus event is used to train or choose an arm",
        "folds": [(str(start), str(end)) for start, end in folds],
        "groups": args.group,
        "focus_events": int(len(focus)),
        "valid_arms": int((report.get("status", pd.Series(dtype=str)) == "ok").sum()),
        "lookalike_rows": int(len(lookalike_frame)),
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-calendar", type=Path, action="append", required=True)
    parser.add_argument("--state-artifact", type=Path, action="append", required=True)
    parser.add_argument("--focus-audit", type=Path, required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", action="append", default=[], help="START::END; may be repeated.")
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--min-family-blocks", type=int, default=3)
    parser.add_argument("--controls-per-block", type=int, default=3)
    parser.add_argument("--warning-days", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
