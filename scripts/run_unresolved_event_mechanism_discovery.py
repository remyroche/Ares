#!/usr/bin/env python3
"""Diagnose unresolved residual episodes against matched non-event periods.

The unit of analysis is a **calendar episode**, never an individual trade.
For each unresolved/partial side x archetype block this runner builds a
two-day causal warning target, compares it with matched benign daily states,
and runs compact interpretable and sequence challengers:

* Spearman/robust feature contrasts;
* RuleFit, Bayesian Rule Lists, contrastive subgroups, and recursive partition;
* shallow episode LightGBM/MLP;
* short causal CNN and a 31-day receptive-field TCN.

Because some groups contain one or two unresolved episodes, leave-one-event
evaluation is explicitly labelled *non-chronological discovery evidence*.
Only a later chronological forward test can promote a finding.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from extreme_price_movements.residual_event_block_taxonomy import MECHANISM_FAMILIES
from extreme_price_movements.residual_rule_models import build_rule_arm, matched_benign_controls
from scripts.report_residual_event_block_taxonomy import _load_daily_state
from scripts.run_residual_hard_period_cnn import (
    _cnn_fit_predict,
    _fill_scale,
    _sequence_bundle,
)


KEYS = ["day", "side_name", "archetype_policy_key"]
ARMS = (
    "rulefit",
    "brl",
    "contrastive_subgroup",
    "model_based_recursive_partition",
    "episode_lgbm",
    "episode_mlp",
    "causal_cnn",
    "causal_tcn",
)


def _daily_transitions(daily: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Current, trailing-two-day, and change features from observable state."""

    pieces: list[pd.DataFrame] = []
    for _, local in daily.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False):
        local = local.sort_values("day", kind="stable").reset_index(drop=True)
        values = local[features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        prior = values.shift(1).rolling(2, min_periods=1).mean().astype(np.float32)
        state = values.rename(columns=lambda name: f"state__{name}")
        change = (values - prior).rename(columns=lambda name: f"change__{name}").astype(np.float32)
        pieces.append(pd.concat([local[KEYS], state, change], axis=1, copy=False))
    return pd.concat(pieces, ignore_index=True, copy=False)


def _focus_warning_labels(local: pd.DataFrame, events: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Label only days before a selected unresolved event, never event outcomes."""

    result = local.loc[:, KEYS].copy()
    result["focus_target"] = False
    result["focus_event_id"] = ""
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_start, tz="UTC") if pd.Timestamp(event.event_start).tzinfo is None else pd.Timestamp(event.event_start).tz_convert("UTC")
        mask = result["day"].ge(start - pd.Timedelta(days=horizon_days)) & result["day"].lt(start)
        result.loc[mask, "focus_target"] = True
        result.loc[mask, "focus_event_id"] = str(event.event_block)
    return result


def _screen(x: np.ndarray, y: np.ndarray, names: list[str], maximum: int) -> tuple[list[int], list[dict[str, object]]]:
    """Rank observable fields by robust separation and matched-sample Spearman."""

    rows: list[dict[str, object]] = []
    y_bool = y.astype(bool)
    for index, name in enumerate(names):
        values = x[:, index].astype(np.float64, copy=False)
        finite = np.isfinite(values)
        if (
            finite.mean() < 0.75
            or finite.sum() < 4
            or np.nanstd(values[finite]) <= 1e-7
            or not finite[y_bool].any()
            or not finite[~y_bool].any()
        ):
            continue
        q25, q75 = np.nanquantile(values[finite], [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-4)
        delta = float(np.nanmedian(values[y_bool]) - np.nanmedian(values[~y_bool]))
        rho = float(spearmanr(values[finite], y[finite]).statistic)
        rows.append({
            "feature": name,
            "spearman": rho,
            "median_positive": float(np.nanmedian(values[y_bool])),
            "median_control": float(np.nanmedian(values[~y_bool])),
            "robust_delta": delta / scale,
            "absolute_score": abs(delta / scale) + 0.50 * abs(rho if np.isfinite(rho) else 0.0),
        })
    rows.sort(key=lambda row: float(row["absolute_score"]), reverse=True)
    selected = [names.index(str(row["feature"])) for row in rows[:maximum]]
    return selected, rows


def _controls(x: np.ndarray, y: np.ndarray, event_ids: np.ndarray, *, seed: int) -> np.ndarray:
    blocks = pd.factorize(event_ids)[0].astype(np.int32)
    blocks[y == 0] = -1
    selected, _ = matched_benign_controls(x, y, blocks, controls_per_event=6)
    # The deterministic matcher can be too small for RuleFit/BRL on a one-day
    # episode. Add a bounded random benign supplement, never outcome rows.
    positives = np.flatnonzero(y > 0)
    benign = np.flatnonzero(y == 0)
    need = max(12, 5 * len(positives)) - int(selected.sum())
    if need > 0 and len(benign):
        rng = np.random.default_rng(seed)
        remaining = benign[~selected[benign]]
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        selected[extra] = True
    return selected | (y > 0)


def _metrics(score: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
    """Evaluate a real top-tail threshold, never an arbitrary tie break.

    A number of sparse-rule arms output a constant or two-valued score.  Ranking
    such a score with ``method='first'`` can accidentally select an event only
    because the event happens to occur early in the daily frame.  The evaluator
    therefore selects all rows strictly above the nominal fifth-percentile
    boundary, or all genuinely tied top-score rows when that is the natural
    threshold.  Constant scores produce an explicit degenerate result.
    """

    finite = np.isfinite(score)
    score = score[finite]
    y = y[finite].astype(bool)
    if len(score) < 5 or not y.any():
        return {
            "ranking_status": "insufficient_evaluation_support",
            "score_std": np.nan,
            "unique_score_count": 0,
            "top05_selected_rate": np.nan,
            "top05_lift": np.nan,
            "top05_fpr": np.nan,
            "top05_recall": np.nan,
            "top05_precision": np.nan,
        }
    unique = np.unique(score)
    score_std = float(np.std(score))
    if len(unique) < 2 or score_std <= 1e-7:
        return {
            "ranking_status": "degenerate_score",
            "score_std": score_std,
            "unique_score_count": int(len(unique)),
            "top05_selected_rate": 0.0,
            "top05_lift": np.nan,
            "top05_fpr": np.nan,
            "top05_recall": np.nan,
            "top05_precision": np.nan,
        }
    count = max(1, int(np.ceil(0.05 * len(score))))
    boundary = float(np.partition(score, len(score) - count)[len(score) - count])
    selected = score >= boundary
    precision = float(y[selected].mean())
    prevalence = float(y.mean())
    # A discretized rule can have a non-constant score while still making the
    # nominal top-5% operating point impossible: every high-score row must be
    # admitted.  Keep that as a useful descriptive result, but do not label it
    # a precision detector when it activates on more than twice the budget.
    ranking_status = "ok" if selected.mean() <= 0.10 else "coarse_score_exceeds_tail_budget"
    return {
        "ranking_status": ranking_status,
        "score_std": score_std,
        "unique_score_count": int(len(unique)),
        "top05_selected_rate": float(selected.mean()),
        "top05_lift": precision / max(prevalence, 1e-9),
        "top05_fpr": float(selected[~y].mean()) if (~y).any() else np.nan,
        "top05_recall": float((selected & y).sum() / y.sum()),
        "top05_precision": precision,
    }


def _fit_arm(arm: str, x: np.ndarray, y: np.ndarray, names: list[str], *, seed: int) -> tuple[object | None, str | None]:
    try:
        if arm in {"causal_cnn", "causal_tcn"}:
            return None, None
        model = build_rule_arm(arm, seed=seed)
        weights = np.where(y > 0, max(1.0, (y == 0).sum() / max(int(y.sum()), 1)), 1.0).astype(np.float32)
        # Third-party rule implementations use older sklearn internals and can
        # emit deprecation warnings.  They remain research-only; suppressing
        # the warning noise does not hide fitting errors, which are recorded.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            model.fit(x, y, weights, names)
        return model, None
    except Exception as exc:  # Discovery arms must not hide dependency/support failure.
        return None, f"{type(exc).__name__}: {exc}"


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit)
    focus = audit.loc[~audit["legacy_calendar_status"].eq("fully_recognized")].copy()
    focus["event_start"] = pd.to_datetime(focus["event_start"], utc=True)
    raw_features = list(dict.fromkeys(name for family in MECHANISM_FAMILIES.values() for name in family))
    daily = _load_daily_state(args.state_artifact, raw_features)
    features = [name for name in raw_features if name in daily]
    transformed = _daily_transitions(daily, features)
    metrics: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    feature_by_event_rows: list[dict[str, object]] = []
    rules: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for (side, archetype), events in focus.groupby(["side_name", "archetype_policy_key"], observed=True):
        local = transformed.loc[
            transformed["side_name"].eq(side) & transformed["archetype_policy_key"].eq(archetype)
        ].sort_values("day", kind="stable").reset_index(drop=True)
        labels = _focus_warning_labels(local, events, args.horizon_days)
        local = local.merge(labels, on=KEYS, how="left", validate="one_to_one")
        local["focus_target"] = local["focus_target"].fillna(False).astype(np.int8)
        local["focus_event_id"] = local["focus_event_id"].fillna("").astype(str)
        candidates = [name for name in local.columns if name.startswith(("state__", "change__"))]
        values = local[candidates].to_numpy(np.float32, copy=True)
        y_all = local["focus_target"].to_numpy(np.int8)
        event_ids = local["focus_event_id"].to_numpy(str)

        # Descriptive contrast over the selected focus periods vs matched
        # non-event days. This remains useful for singleton historical events.
        sample_mask = _controls(values, y_all, event_ids, seed=args.seed)
        _, contrasts = _screen(values[sample_mask], y_all[sample_mask], candidates, args.max_features)
        for row in contrasts:
            feature_rows.append({
                "side_name": side, "archetype_policy_key": archetype,
                "focus_events": int(len(events)), "evidence_type": "matched_event_vs_non_event_descriptive", **row,
            })

        event_ids_unique = [value for value in events["event_block"].astype(str).unique() if value]
        # Event-by-event contrasts answer a different question from pooled
        # relevance: does an observable field separate *several* episodes of
        # the same archetype from their matched benign lookalikes with a stable
        # direction?  This is descriptive discovery evidence, not OOS model
        # validation, because the calendar supplied the event labels.
        for event_id in event_ids_unique:
            event_y = (event_ids == event_id).astype(np.int8)
            event_mask = _controls(values, event_y, event_ids, seed=args.seed + len(feature_by_event_rows))
            _, event_contrasts = _screen(
                values[event_mask], event_y[event_mask], candidates, len(candidates)
            )
            for row in event_contrasts:
                feature_by_event_rows.append({
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "event_block": event_id,
                    "event_days": int(event_y.sum()),
                    "evidence_type": "matched_single_event_vs_non_event_descriptive",
                    **row,
                })

        for held in event_ids_unique:
            train_mask = event_ids != held
            test_mask = np.ones(len(local), dtype=bool)
            # Exclude held warning days from fitting and score the full local
            # history. This is leave-one-event-out, not chronological OOS.
            train_y = y_all[train_mask]
            if train_y.sum() < args.min_train_positive_days:
                metrics.append({
                    "side_name": side, "archetype_policy_key": archetype, "held_event": held,
                    "status": "insufficient_other_events", "focus_events": int(len(events)),
                    "train_positive_days": int(train_y.sum()),
                })
                continue
            train_x = values[train_mask].copy()
            score_x = values[test_mask].copy()
            train_x, score_x = _fill_scale(train_x, score_x)
            control_mask = _controls(train_x, train_y, event_ids[train_mask], seed=args.seed + len(metrics))
            selected_idx, selected_contrasts = _screen(
                train_x[control_mask], train_y[control_mask], candidates, args.max_features
            )
            if len(selected_idx) < 3:
                continue
            x_fit = train_x[control_mask][:, selected_idx]
            y_fit = train_y[control_mask]
            x_score = score_x[:, selected_idx]
            names = [candidates[index] for index in selected_idx]
            # Other focus events are training positives for this leave-one-event
            # fit. They are neither negatives nor valid test outcomes for the
            # held episode, so exclude them from its false-alert denominator.
            evaluation_mask = (event_ids == held) | (event_ids == "")
            y_test = (event_ids[evaluation_mask] == held).astype(np.int8)
            for arm in ARMS:
                if arm in {"causal_cnn", "causal_tcn"}:
                    # Build sequences on the full daily clock, but train only
                    # on other selected events plus their matched controls.
                    sequence_frame = pd.DataFrame(x_score, columns=names)
                    sequence_frame["day"] = local["day"].to_numpy()
                    sequence_frame["event_start"] = y_all.astype(bool)
                    bundle = _sequence_bundle(sequence_frame, names, args.window_days, horizon=1)
                    fit_idx = np.flatnonzero(train_mask)
                    risk_all = _cnn_fit_predict(
                        bundle.x[fit_idx], y_all[fit_idx], bundle.x,
                        seed=args.seed + len(metrics), epochs=args.epochs,
                        architecture="cnn" if arm == "causal_cnn" else "tcn",
                    )
                    risk = risk_all[evaluation_mask]
                    error = None
                    description: list[dict[str, object]] = []
                else:
                    model, error = _fit_arm(arm, x_fit, y_fit, names, seed=args.seed + len(metrics))
                    risk_all = np.full(len(local), np.nan, dtype=np.float32) if model is None else model.predict_proba(x_score)
                    risk = risk_all[evaluation_mask]
                    description = [] if model is None else model.describe()
                result = _metrics(risk, y_test)
                status = "ok" if error is None and result["ranking_status"] == "ok" else (
                    str(result["ranking_status"]) if error is None else "fit_failed"
                )
                metrics.append({
                    "side_name": side, "archetype_policy_key": archetype, "held_event": held,
                    "model": arm, "status": status,
                    "error": error, "focus_events": int(len(events)),
                    "train_positive_days": int(train_y.sum()), "train_samples": int(len(y_fit)),
                    "features": "|".join(names), "evaluation_contract": "leave_one_event_out_nonchronological_discovery",
                    **result,
                })
                if args.write_scores and error is None:
                    for day, target, value in zip(local.loc[evaluation_mask, "day"], y_test, risk):
                        score_rows.append({
                            "side_name": side,
                            "archetype_policy_key": archetype,
                            "held_event": held,
                            "model": arm,
                            "day": pd.Timestamp(day).isoformat(),
                            "is_held_warning_day": int(target),
                            "score": float(value) if np.isfinite(value) else np.nan,
                        })
                for item in description[:12]:
                    rules.append({
                        "side_name": side, "archetype_policy_key": archetype,
                        "held_event": held, "model": arm, **item,
                    })
            del train_x, score_x, x_fit, x_score
            gc.collect()

    metrics_frame = pd.DataFrame(metrics)
    features_frame = pd.DataFrame(feature_rows)
    feature_by_event_frame = pd.DataFrame(feature_by_event_rows)
    rules_frame = pd.DataFrame(rules)
    scores_frame = pd.DataFrame(score_rows)
    metrics_frame.to_csv(args.output / "focused_event_model_metrics.csv", index=False)
    features_frame.to_csv(args.output / "focused_event_feature_spearman.csv", index=False)
    feature_by_event_frame.to_csv(args.output / "focused_event_feature_by_event.csv", index=False)
    if not feature_by_event_frame.empty:
        consistency = (
            feature_by_event_frame.assign(
                direction=np.sign(feature_by_event_frame["robust_delta"].to_numpy(np.float64, copy=False))
            )
            .groupby(["side_name", "archetype_policy_key", "feature"], observed=True)
            .agg(
                events_observed=("event_block", "nunique"),
                positive_direction_events=("direction", lambda values: int((values > 0).sum())),
                negative_direction_events=("direction", lambda values: int((values < 0).sum())),
                median_robust_delta=("robust_delta", "median"),
                median_abs_robust_delta=("robust_delta", lambda values: float(np.median(np.abs(values)))),
                median_spearman=("spearman", "median"),
            )
            .reset_index()
        )
        consistency["same_direction_events"] = consistency[["positive_direction_events", "negative_direction_events"]].max(axis=1)
        consistency["direction_consistency"] = (
            consistency["same_direction_events"] / consistency["events_observed"].clip(lower=1)
        )
        consistency["common_pattern_candidate"] = (
            (consistency["events_observed"] >= 3)
            & (consistency["same_direction_events"] >= 3)
            & (consistency["direction_consistency"] >= 0.75)
            & (consistency["median_abs_robust_delta"] >= 0.50)
        )
        consistency.sort_values(
            ["common_pattern_candidate", "same_direction_events", "median_abs_robust_delta"],
            ascending=[False, False, False], inplace=True,
        )
    else:
        consistency = pd.DataFrame()
    consistency.to_csv(args.output / "focused_event_feature_consistency.csv", index=False)
    rules_frame.to_csv(args.output / "focused_event_extracted_rules.csv", index=False)
    if args.write_scores:
        scores_frame.to_csv(args.output / "focused_event_model_scores.csv", index=False)
    summary = {
        "purpose": "unresolved/partial residual event mechanism discovery; inactive and non-chronological where singleton support requires it",
        "focus_blocks": int(len(focus)),
        "groups": int(focus.groupby(["side_name", "archetype_policy_key"]).ngroups),
        "observable_feature_count": int(len(features)),
        "arms": list(ARMS),
        "warning_horizon_days": int(args.horizon_days),
        "models_with_valid_scores": int(len(metrics_frame.loc[metrics_frame.get("status", pd.Series(dtype=str)).eq("ok")])),
        "models_with_degenerate_scores": int(len(metrics_frame.loc[metrics_frame.get("status", pd.Series(dtype=str)).eq("degenerate_score")])),
        "models_with_coarse_scores": int(len(metrics_frame.loc[metrics_frame.get("status", pd.Series(dtype=str)).eq("coarse_score_exceeds_tail_budget")])),
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--state-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-days", type=int, default=2)
    parser.add_argument("--window-days", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--min-train-positive-days", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--write-scores", action="store_true", help="Export held-event daily score paths for audit.")
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
