#!/usr/bin/env python3
"""Evaluate train-derived adverse-block family detectors at daily-open only.

This is deliberately not a row-level error model.  Positive examples are
starts of adverse calendar blocks; negatives are matched benign block starts.
Each family is discovered using train blocks only, then a shallow detector is
frozen and scored on later daily-open state.  Its output is research context,
not an active overlay or policy gate.
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
    _load_daily_state,
    _load_calendar,
    _overlay_event_calendar,
)


FOLDS = (
    (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
    (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
    (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")),
)


def _parse_group(value: str) -> tuple[str, str]:
    side, separator, archetype = value.partition("::")
    if not separator or not side or not archetype:
        raise ValueError(f"Expected side::archetype, got {value!r}")
    return side, archetype


def _daily_start_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create only current and previous-two-day observable state features."""

    keys = ["day", "side_name", "archetype_policy_key"]
    raw = [name for name in daily.columns if name not in keys]
    pieces: list[pd.DataFrame] = []
    for _, local in daily.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False):
        local = local.sort_values("day", kind="stable").copy()
        values = local[raw].apply(pd.to_numeric, errors="coerce")
        previous = values.shift(1).rolling(2, min_periods=1).mean()
        # Construct each transformation block at once.  Repeated ``insert``
        # calls create a fragmented dataframe with a large cost on the wide
        # observable state contract used by the full taxonomy run.
        current = values.astype(np.float32)
        prior = previous.astype(np.float32)
        onset = (current - prior).astype(np.float32)
        current.columns = [f"state__{name}" for name in raw]
        prior.columns = [f"prior2__{name}" for name in raw]
        onset.columns = [f"onset__{name}" for name in raw]
        pieces.append(
            pd.concat(
                [local[keys].reset_index(drop=True), current.reset_index(drop=True),
                 prior.reset_index(drop=True), onset.reset_index(drop=True)],
                axis=1,
                copy=False,
            )
        )
    result = pd.concat(pieces, ignore_index=True, copy=False)
    return result, [name for name in result.columns if name not in keys]


def _screen_features(samples: pd.DataFrame, features: list[str], *, maximum: int) -> list[str]:
    """Robust train-only univariate screen suitable for a few dozen blocks."""

    y = samples["target"].to_numpy(bool)
    rows: list[tuple[str, float]] = []
    for name in features:
        values = pd.to_numeric(samples[name], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.mean() < 0.80 or not finite[y].any() or not finite[~y].any():
            continue
        q25, q75 = np.nanquantile(values[finite], [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-4)
        difference = abs(float(np.nanmedian(values[y]) - np.nanmedian(values[~y]))) / scale
        rows.append((name, difference))
    return [name for name, _ in sorted(rows, key=lambda item: item[1], reverse=True)[:maximum]]


def _matrix(train: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = train[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    median = np.nanmedian(x_train, axis=0)
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)
    for values in (x_train, x_score):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        np.clip(values, -8.0, 8.0, out=values)
    return x_train, x_score, median


def _top_k_metrics(frame: pd.DataFrame, *, fraction: float) -> dict[str, float]:
    """Evaluate exact daily top-k admission for a rare block-start target."""

    suffix = f"top{int(round(fraction * 100)):02d}"
    if frame.empty:
        return {
            f"{suffix}_selected_days": 0,
            f"{suffix}_precision": np.nan,
            f"{suffix}_fpr": np.nan,
            f"{suffix}_lift": np.nan,
            f"{suffix}_block_recall": np.nan,
        }
    top_count = max(1, int(np.ceil(len(frame) * float(fraction))))
    rank = frame["risk"].rank(method="first", ascending=False)
    selected = rank.le(top_count).to_numpy(bool)
    event = frame["event_start"].to_numpy(bool)
    precision = float(event[selected].mean()) if selected.any() else np.nan
    prevalence = float(event.mean())
    fpr = float(selected[~event].mean()) if (~event).any() else np.nan
    return {
        f"{suffix}_selected_days": int(selected.sum()),
        f"{suffix}_precision": precision,
        f"{suffix}_fpr": fpr,
        f"{suffix}_lift": precision / max(prevalence, 1e-9) if np.isfinite(precision) else np.nan,
        f"{suffix}_block_recall": float((selected & event).sum() / max(event.sum(), 1)),
    }


def _operating_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Report conservative and diagnostic operating points for a detector."""

    result: dict[str, float] = {
        "days": int(len(frame)),
        "event_starts": int(frame["event_start"].sum()) if not frame.empty else 0,
    }
    for fraction in (0.01, 0.03, 0.05, 0.10):
        result.update(_top_k_metrics(frame, fraction=fraction))
    # Keep legacy top-10 names for comparison with the first detector smoke.
    result.update(
        {
            "selected_days": result["top10_selected_days"],
            "precision": result["top10_precision"],
            "fpr": result["top10_fpr"],
            "lift": result["top10_lift"],
            "block_recall": result["top10_block_recall"],
        }
    )
    return result


def _family_train_samples(
    taxonomy: pd.DataFrame,
    controls: pd.DataFrame,
    start_features: pd.DataFrame,
    events: pd.DataFrame,
    *,
    side: str,
    archetype: str,
    family: str,
    family_column: str,
    event_phase: str,
) -> pd.DataFrame:
    # ``event_001`` and similar labels are only unique within a side x
    # archetype calendar.  Restrict before joining: merging bare block IDs
    # silently turns a local mechanism family into unrelated positives from
    # every other group that happened to reuse the same ordinal.
    local_events = events.loc[
        events["side_name"].eq(side)
        & events["archetype_policy_key"].eq(archetype)
    ].copy()
    selected_blocks = taxonomy.loc[
        taxonomy["side_name"].eq(side)
        & taxonomy["archetype_policy_key"].eq(archetype)
        & taxonomy[family_column].eq(family),
        ["event_block"],
    ]
    positive = _phase_event_days(
        local_events,
        event_phase=event_phase,
    ).merge(selected_blocks, on="event_block", how="inner", validate="many_to_one")
    positive = positive.loc[:, ["event_block", "day", "event_start"]]
    positive["target"] = 1
    control = controls.loc[
        controls["side_name"].eq(side)
        & controls["archetype_policy_key"].eq(archetype)
        & controls["event_block"].isin(positive["event_block"]),
        ["event_block", "event_start", "control_start"],
    ]
    negative = positive.loc[:, ["event_block", "day", "event_start"]].merge(
        control, on=["event_block", "event_start"], how="inner", validate="many_to_many"
    )
    negative["day"] = (
        pd.to_datetime(negative["control_start"], utc=True)
        + (pd.to_datetime(negative["day"], utc=True) - pd.to_datetime(negative["event_start"], utc=True))
    )
    negative = negative.loc[:, ["day"]]
    negative["target"] = 0
    samples = pd.concat([positive[["day", "target"]], negative], ignore_index=True)
    local = start_features.loc[
        start_features["side_name"].eq(side)
        & start_features["archetype_policy_key"].eq(archetype)
    ]
    return samples.merge(local, on="day", how="inner", validate="many_to_one")


def _phase_event_days(events: pd.DataFrame, *, event_phase: str) -> pd.DataFrame:
    """Return causal daily labels for one phase of an adverse event block."""

    event = events.loc[events["event_block"].ne("normal")].copy()
    if event.empty:
        return pd.DataFrame(columns=["event_block", "day", "event_start"])
    keys = ["side_name", "archetype_policy_key", "event_block"]
    event = event.sort_values([*keys, "day"], kind="stable")
    event["event_start"] = event.groupby(keys, observed=True)["day"].transform("min")
    if event_phase == "onset":
        result = event.loc[event["day"].eq(event["event_start"])]
    elif event_phase == "active":
        result = event
    elif event_phase == "late":
        length = event.groupby(keys, observed=True)["day"].transform("size")
        end = event.groupby(keys, observed=True)["day"].transform("max")
        result = event.loc[length.gt(1) & event["day"].eq(end)]
    else:
        raise ValueError(f"Unsupported event phase: {event_phase!r}")
    return result.loc[:, ["event_block", "day", "event_start"]].copy()


def _fit_score(train: pd.DataFrame, score: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x_train, x_score, _ = _matrix(train, score, features)
    y = train["target"].to_numpy(np.int8)
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    weights = np.where(y > 0, negatives / positives, 1.0).astype(np.float32)
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.04,
            "max_depth": 2,
            "num_leaves": 4,
            "min_data_in_leaf": max(3, min(8, len(train) // 5)),
            "min_gain_to_split": 0.05,
            "lambda_l1": 2.0,
            "lambda_l2": 12.0,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.90,
            "bagging_freq": 1,
            "seed": seed,
            "verbosity": -1,
            "force_col_wise": True,
        },
        lgb.Dataset(x_train, label=y, weight=weights, feature_name=features),
        num_boost_round=80,
    )
    return np.asarray(model.predict(x_score), dtype=np.float32)


def _family_summary(report: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics without silently promoting one-off wins."""

    if report.empty:
        return pd.DataFrame()
    valid = report.loc[report["status"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()
    keys = ["family_source", "side_name", "archetype_policy_key", "family"]
    aggregations: dict[str, tuple[str, object]] = {"folds": ("fold_start", "size")}
    for suffix in ("top01", "top03", "top05", "top10"):
        aggregations[f"{suffix}_mean_lift"] = (f"{suffix}_lift", "mean")
        aggregations[f"{suffix}_mean_fpr"] = (f"{suffix}_fpr", "mean")
        aggregations[f"{suffix}_mean_recall"] = (f"{suffix}_block_recall", "mean")
        aggregations[f"{suffix}_hit_folds"] = (
            f"{suffix}_block_recall", lambda value: int((value > 0).sum())
        )
    summary = valid.groupby(keys, observed=True, as_index=False).agg(**aggregations)
    # This is a research gate, not a policy decision.  The requirement is
    # deliberately strict because each detector sees a rare transition class.
    summary["passes_top05_repetition_gate"] = (
        summary["folds"].ge(3)
        & summary["top05_hit_folds"].ge(3)
        & summary["top05_mean_lift"].ge(1.5)
        & summary["top05_mean_fpr"].le(0.15)
    )
    return summary.sort_values(
        ["passes_top05_repetition_gate", "top05_mean_lift", "top05_mean_fpr"],
        ascending=[False, False, True],
        kind="stable",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    raw_features = list(dict.fromkeys(feature for group in MECHANISM_FAMILIES.values() for feature in group))
    daily = _load_daily_state(args.state_artifact, raw_features)
    calendar = daily.loc[:, ["day", "side_name", "archetype_policy_key"]].copy()
    calendar = _overlay_event_calendar(calendar, args.event_calendar)
    start_features, feature_columns = _daily_start_features(daily)
    groups = [_parse_group(value) for value in args.group]
    group_frame = pd.DataFrame(
        groups, columns=["side_name", "archetype_policy_key"]
    ).drop_duplicates(ignore_index=True)
    reports: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    config = BlockTaxonomyConfig(
        pre_days=2, post_days=1, min_reference_days=30,
        controls_per_block=args.controls_per_block, max_clusters=4,
        min_cluster_blocks=args.min_family_blocks,
    )
    for fold_index, (train_end, eval_end) in enumerate(FOLDS):
        train_calendar = calendar.loc[calendar["day"].lt(train_end)].merge(
            group_frame, on=["side_name", "archetype_policy_key"], how="inner"
        )
        eval_calendar = calendar.loc[
            calendar["day"].ge(train_end) & calendar["day"].lt(eval_end)
        ].merge(group_frame, on=["side_name", "archetype_policy_key"], how="inner")
        train_daily = daily.loc[daily["day"].lt(train_end)].merge(
            group_frame, on=["side_name", "archetype_policy_key"], how="inner"
        )
        taxonomy, _ = build_block_taxonomy(train_calendar, train_daily, config=config)
        taxonomy = annotate_onset_mechanism_profiles(taxonomy)
        controls = matched_benign_block_controls(train_calendar, train_daily, taxonomy, config=config)
        starts = attach_event_blocks(eval_calendar)
        for side, archetype in groups:
            family_column = (
                "onset_primary_mechanism"
                if args.family_source == "onset_mechanism"
                else "block_family"
            )
            local_taxonomy = taxonomy.loc[
                taxonomy["side_name"].eq(side)
                & taxonomy["archetype_policy_key"].eq(archetype)
            ]
            if args.family_source == "cluster":
                local_taxonomy = local_taxonomy.loc[
                    local_taxonomy["block_family_id"].ge(0)
                ]
            else:
                local_taxonomy = local_taxonomy.loc[
                    local_taxonomy["onset_primary_mechanism"].ne("unavailable")
                ]
            for family, family_taxonomy in local_taxonomy.groupby(family_column, observed=True):
                if len(family_taxonomy) < args.min_family_blocks:
                    continue
                samples = _family_train_samples(
                    taxonomy, controls, start_features, attach_event_blocks(train_calendar), side=side,
                    archetype=archetype, family=str(family),
                    family_column=family_column,
                    event_phase=args.event_phase,
                )
                local_eval = start_features.loc[
                    start_features["side_name"].eq(side)
                    & start_features["archetype_policy_key"].eq(archetype)
                    & start_features["day"].ge(train_end)
                    & start_features["day"].lt(eval_end)
                ].copy()
                # Event-block identifiers are local to side x archetype, so
                # retain only the requested group before the day membership.
                local_group_events = starts.loc[
                    starts["side_name"].eq(side)
                    & starts["archetype_policy_key"].eq(archetype)
                ]
                event_days = _phase_event_days(
                    local_group_events, event_phase=args.event_phase
                )["day"]
                local_eval["event_start"] = local_eval["day"].isin(event_days)
                selected = _screen_features(samples, feature_columns, maximum=args.max_features)
                if samples["target"].sum() < args.min_family_blocks or (samples["target"] == 0).sum() < args.min_family_blocks or len(selected) < 2 or local_eval.empty:
                    reports.append({"fold_start": train_end, "fold_end": eval_end, "side_name": side, "archetype_policy_key": archetype, "family_source": args.family_source, "family": family, "status": "insufficient_train_support", "train_positive_blocks": int(samples["target"].sum()), "train_controls": int((samples["target"] == 0).sum()), "features": "|".join(selected)})
                    continue
                local_eval["risk"] = _fit_score(samples, local_eval, selected, seed=args.seed + fold_index)
                metrics = _operating_metrics(local_eval)
                reports.append({"fold_start": train_end, "fold_end": eval_end, "side_name": side, "archetype_policy_key": archetype, "family_source": args.family_source, "family": family, "status": "ok", "train_positive_blocks": int(samples["target"].sum()), "train_controls": int((samples["target"] == 0).sum()), "features": "|".join(selected), **metrics})
                predictions.append(local_eval.assign(fold_start=train_end, family_source=args.family_source, family=family))
    report = pd.DataFrame(reports)
    oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    report.to_csv(args.output / "family_detector_oof_metrics.csv", index=False)
    _family_summary(report).to_csv(
        args.output / "family_detector_oof_summary.csv", index=False
    )
    oof.to_parquet(args.output / "family_detector_oof_daily_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "purpose": "block-level, family-specific daily-open detector research; inactive and not a policy overlay",
        "event_calendar": [str(path) for path in args.event_calendar],
        "state_artifacts": [str(path) for path in args.state_artifact],
        "groups": args.group,
        "folds": [(str(start), str(end)) for start, end in FOLDS],
        "daily_snapshot_contract": "first daily timestamp only; current, trailing-prior, and onset-change observables only",
        "train_label_contract": "train-derived adverse block family versus matched benign block starts",
        "family_source": args.family_source,
        "event_phase": args.event_phase,
        "oos_label_contract": f"all adverse block {args.event_phase} days, evaluation only",
        "rows": int(len(oof)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-calendar", type=Path, action="append", required=True)
    parser.add_argument("--state-artifact", type=Path, action="append", required=True)
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument(
        "--family-source", choices=("cluster", "onset_mechanism"), default="onset_mechanism",
        help="Use train-local unsupervised clusters or causal onset-mechanism strata as positives.",
    )
    parser.add_argument(
        "--event-phase", choices=("onset", "active", "late"), default="onset",
        help="Train/evaluate event-block onset, all adverse days, or late-event days.",
    )
    parser.add_argument("--min-family-blocks", type=int, default=3)
    parser.add_argument("--controls-per-block", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if not args.group:
        args.group = [
            "long::long_breakout_diagnostic_candidate",
            "long::long_mixed_wideslow_tentative",
        ]
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(f"completed oof_daily_rows={result['rows']}")
