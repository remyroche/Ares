#!/usr/bin/env python3
"""One bounded online-BOCPD ablation for three-hour transition onset.

The experiment has exactly three supervised arms using the same fixed
LightGBM specification and the canonical v3 grouped research split:

* ``transition_baseline``: existing causal transition feature matrix;
* ``changepoint_only``: four compact causal BOCPD summaries;
* ``transition_plus_changepoint``: baseline plus those four summaries.

BOCPD inputs are causal and the 30-day warm-up fit is strictly in each score's
past.  This is a compact context test, not a new market-state taxonomy.  Model
OOF is grouped research validation rather than walk-forward promotion
evidence.  For operational metrics, thresholds are frozen from the earliest
chronological OOF calibration period using *scores only* (an alert-frequency
budget); subsequent rows are held out from threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_changepoint import (  # noqa: E402
    CHANGEPOINT_FEATURE_COLUMNS,
    CHANGEPOINT_INPUT_COLUMNS,
    CausalChangePointConfig,
    materialize_causal_changepoint_context,
)
from scripts.run_regime_transition_classifier_ablation import _groups, _safe  # noqa: E402


DEFAULT_DATASET = ROOT / "data_perp/artifacts/regime_transition_research_20260726_v3/hourly_transition_dataset.parquet"
DEFAULT_EVENTS = ROOT / "data_perp/artifacts/regime_transition_research_20260726_v3/transition_events.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/regime_transition_changepoint_ablation_20260727_v1"
SCHEMA = "pooled_grouped_online_bocpd_onset_ablation_v1"


def _model(seed: int) -> LGBMClassifier:
    """Frozen prior expressive onset-HPO family; no new BOCPD HPO sweep."""

    return LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=30,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=8.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def causal_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return every numeric decision-time v3 field, excluding labels/IDs."""

    blocked = {"source_utc", "execution_decision_utc", "segment_id"}
    return [
        name
        for name in frame.columns
        if name not in blocked
        and not name.startswith(("target__", "expost__"))
        and pd.api.types.is_numeric_dtype(frame[name])
    ]


def grouped_oof(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate aligned grouped research OOF probabilities and fold IDs."""

    x = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    prediction = np.full(len(frame), np.nan, dtype=np.float32)
    fold_id = np.full(len(frame), -1, dtype=np.int16)
    for fold, (train, evaluation) in enumerate(splitter.split(x, y, groups=groups)):
        model = _model(int(seed) + fold)
        model.fit(x.iloc[train], y[train])
        prediction[evaluation] = model.predict_proba(x.iloc[evaluation])[:, 1]
        fold_id[evaluation] = fold
    if not np.isfinite(prediction).all() or (fold_id < 0).any():
        raise ValueError("grouped OOF did not score every row")
    return prediction, fold_id


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, np.clip(score, 1e-7, 1 - 1e-7))),
        "prevalence": float(np.mean(y)),
    }


def _alert_episodes(
    timestamp: pd.Series,
    segment: pd.Series,
    score: np.ndarray,
    threshold: float,
    *,
    refractory_hours: int,
) -> int:
    """Count score-only alerts with a fixed refractory period per segment."""

    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamp, utc=True),
            "segment": segment.to_numpy(),
            "score": np.asarray(score, dtype=float),
        }
    ).sort_values("timestamp", kind="stable")
    alerts = 0
    for _, local in work.loc[work["score"].ge(float(threshold))].groupby("segment", observed=True, sort=False):
        last: pd.Timestamp | None = None
        for stamp in local["timestamp"]:
            if last is None or stamp - last >= pd.Timedelta(hours=int(refractory_hours)):
                alerts += 1
                last = stamp
    return alerts


def freeze_score_only_threshold(
    timestamp: pd.Series,
    segment: pd.Series,
    score: np.ndarray,
    *,
    budget_per_30d: float,
    refractory_hours: int = 6,
) -> tuple[float, float]:
    """Freeze a threshold from score frequency only, without event labels.

    The selected cutoff is the least restrictive one that remains within the
    requested alert budget on the calibration period.  It cannot see any
    calibration or evaluation onset labels.
    """

    score = np.asarray(score, dtype=float)
    finite = score[np.isfinite(score)]
    if not len(finite):
        return float("inf"), 0.0
    stamp = pd.to_datetime(timestamp, utc=True)
    days = max((stamp.max() - stamp.min()).total_seconds() / 86_400.0, 1.0)
    # At most 120 six-hour alerts are possible per 30 days.  The fixed
    # 1/2/4-alert budgets necessarily live in the high-score tail; a compact
    # predetermined 0.80--1.00 grid is more than adequate and avoids turning
    # a threshold freeze into an expensive implicit search.
    candidates = np.unique(np.quantile(finite, np.linspace(0.80, 1.0, 81)))
    candidates = np.r_[candidates, np.nextafter(float(np.max(finite)), np.inf)]
    rows: list[tuple[float, float]] = []
    for threshold in candidates:
        rate = _alert_episodes(stamp, segment, score, float(threshold), refractory_hours=refractory_hours) / days * 30.0
        if rate <= float(budget_per_30d):
            rows.append((float(threshold), float(rate)))
    if not rows:
        return float(np.nextafter(float(np.max(finite)), np.inf)), 0.0
    # Lowest allowable threshold maximises alerts/detection while honoring the
    # budget; tie-breaking on rate is deterministic.
    return min(rows, key=lambda item: (item[0], -item[1]))


def _event_kind(events: pd.DataFrame) -> pd.Series:
    required = {"event_id", "anchor_source_utc", "transition_end_utc"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"events missing {sorted(missing)}")
    duration = (
        pd.to_datetime(events["transition_end_utc"], utc=True)
        - pd.to_datetime(events["anchor_source_utc"], utc=True)
    ) / pd.Timedelta(hours=1)
    return pd.Series(np.where(duration.le(3.0), "abrupt", "gradual"), index=events["event_id"].astype(str))


def frozen_operational_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    calibration_end: pd.Timestamp,
    budget_per_30d: float,
    event_kinds: pd.Series,
    refractory_hours: int = 6,
) -> dict[str, float | int | str | None]:
    """Freeze score-only cutoff in the early era, then score later events."""

    timestamp = pd.to_datetime(frame["source_utc"], utc=True)
    calibration = timestamp.lt(calibration_end).to_numpy()
    evaluation = ~calibration
    threshold, calibration_rate = freeze_score_only_threshold(
        timestamp.loc[calibration],
        frame.loc[calibration, "segment_id"],
        score[calibration],
        budget_per_30d=budget_per_30d,
        refractory_hours=refractory_hours,
    )
    eval_frame = frame.loc[evaluation, ["source_utc", "segment_id", "target__event_id", "target__onset_within_3h"]].copy()
    eval_score = score[evaluation]
    negative = eval_frame["target__onset_within_3h"].eq(0).to_numpy()
    eval_days = max(
        (pd.to_datetime(eval_frame["source_utc"], utc=True).max() - pd.to_datetime(eval_frame["source_utc"], utc=True).min()).total_seconds() / 86_400.0,
        1.0,
    )
    false_rate = _alert_episodes(
        eval_frame.loc[negative, "source_utc"],
        eval_frame.loc[negative, "segment_id"],
        eval_score[negative],
        threshold,
        refractory_hours=refractory_hours,
    ) / eval_days * 30.0
    lead = eval_frame[eval_frame["target__onset_within_3h"].eq(1) & eval_frame["target__event_id"].notna()].copy()
    # ``lead`` retains original row indexes; attach from an aligned full array
    # to avoid any accidental positional shift after chronological slicing.
    full_score = np.asarray(score, dtype=float)
    lead["score"] = full_score[lead.index.to_numpy()]
    per_event = lead.groupby("target__event_id", observed=True)["score"].max()
    detected = per_event.ge(threshold)
    kinds = event_kinds.reindex(per_event.index.astype(str)).fillna("unknown")
    output: dict[str, float | int | str | None] = {
        "budget_per_30d": float(budget_per_30d),
        "threshold": float(threshold),
        "calibration_alerts_per_30d_score_only": float(calibration_rate),
        "evaluation_false_alerts_per_30d": float(false_rate),
        "evaluation_event_count": int(len(per_event)),
        "evaluation_event_recall": float(detected.mean()) if len(detected) else None,
    }
    for kind in ("abrupt", "gradual"):
        mask = kinds.eq(kind)
        output[f"evaluation_{kind}_event_count"] = int(mask.sum())
        output[f"evaluation_{kind}_event_recall"] = float(detected.loc[mask].mean()) if mask.any() else None
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    frame = pd.read_parquet(args.dataset).reset_index(drop=True)
    frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True)
    events = pd.read_parquet(args.events)
    context, context_features = materialize_causal_changepoint_context(
        frame,
        config=CausalChangePointConfig(
            warmup_hours=int(args.warmup_hours),
            expected_run_hours=int(args.expected_run_hours),
            max_run_hours=int(args.max_run_hours),
        ),
    )
    frame = pd.concat([frame, context.reset_index(drop=True)], axis=1)
    baseline = causal_feature_columns(frame)
    baseline = [name for name in baseline if name not in set(context_features)]
    if len(context_features) != len(CHANGEPOINT_FEATURE_COLUMNS):
        raise AssertionError("unexpected changepoint feature contract")
    all_arms = {
        "transition_baseline": baseline,
        "changepoint_only": context_features,
        "transition_plus_changepoint": [*baseline, *context_features],
    }
    arms = all_arms if args.arm == "all" else {str(args.arm): all_arms[str(args.arm)]}
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    calibration_end = frame["source_utc"].min() + (frame["source_utc"].max() - frame["source_utc"].min()) * float(args.calibration_fraction)
    calibration_end = pd.Timestamp(calibration_end)
    if calibration_end.tzinfo is None:
        calibration_end = calibration_end.tz_localize("UTC")
    kinds = _event_kind(events)
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    oof = frame[["source_utc", "execution_decision_utc", "segment_id", "target__event_id", "target__time_to_onset_hours", "target__onset_within_3h"]].copy()
    final_models: dict[str, Any] = {}
    for offset, (arm, features) in enumerate(arms.items()):
        prediction, fold_ids = grouped_oof(frame, features=features, folds=int(args.folds), seed=int(args.seed) + 100 * offset)
        oof[f"prediction__{arm}"] = prediction
        oof[f"fold__{arm}"] = fold_ids
        metric_rows.append({"arm": arm, "feature_count": len(features), **binary_metrics(y, prediction)})
        for budget in (1.0, 2.0, 4.0):
            event_rows.append({"arm": arm, **frozen_operational_metrics(frame, prediction, calibration_end=calibration_end, budget_per_30d=budget, event_kinds=kinds)})
        model = _model(int(args.seed) + 1000 + offset)
        model.fit(frame.loc[:, features].apply(pd.to_numeric, errors="coerce"), y)
        final_models[arm] = {"features": list(features), "model": model}
    metrics = pd.DataFrame(metric_rows).sort_values(["average_precision", "roc_auc"], ascending=False)
    operations = pd.DataFrame(event_rows).sort_values(["budget_per_30d", "arm"])
    oof = pd.concat([oof, context.reset_index(drop=True)], axis=1)
    metrics.to_csv(output / "grouped_oof_metrics.csv", index=False)
    operations.to_csv(output / "frozen_operational_event_metrics.csv", index=False)
    oof.to_parquet(output / "grouped_oof_predictions_and_changepoint_context.parquet", index=False)
    joblib.dump(final_models, output / "final_models.joblib")
    manifest = {
        "schema": SCHEMA,
        "research_only": True,
        "dataset": str(args.dataset),
        "events": str(args.events),
        "rows": int(len(frame)),
        "event_count": int(events["event_id"].nunique()),
        "positive_onset_rows": int(y.sum()),
        "arms": {name: len(features) for name, features in arms.items()},
        "intended_arms": {name: len(features) for name, features in all_arms.items()},
        "changepoint": {
            "family": "univariate Normal-Inverse-Gamma BOCPD, then fixed multi-signal summaries",
            "inputs": list(CHANGEPOINT_INPUT_COLUMNS),
            "features": list(context_features),
            "warmup_hours": int(args.warmup_hours),
            "expected_run_hours": int(args.expected_run_hours),
            "max_run_hours": int(args.max_run_hours),
            "online_contract": "each signal is robust-scaled from its initial preceding warm-up only; the BOCPD posterior at t sees values through t only; segment/gap boundaries reset state",
        },
        "validation": "5-fold stratified grouped research OOF; each event lead window remains together and stable controls are grouped by calendar week",
        "threshold_contract": {
            "calibration_end": str(calibration_end),
            "calibration_fraction": float(args.calibration_fraction),
            "selection": "thresholds use only early OOF score frequency and a six-hour refractory alert counter; no event labels are read while selecting thresholds",
            "evaluation": "later chronological OOF era only; false alerts use known negatives only for retrospective reporting",
            "budgets_per_30d": [1.0, 2.0, 4.0],
        },
        "caveats": [
            "The supervised grouped OOF folds are shuffled research folds, not a chronological walk-forward fit; they cannot establish prospective promotion eligibility.",
            "The initial 30-day BOCPD warm-up is deliberately unscored. Its robust scaling data are historical by the first usable score, but the earliest panel rows have no changepoint signal.",
            "Frozen alert cutoffs are score-only, but a future production cutoff must be frozen on an earlier completed deployment-era calibration period before use.",
            "This ablation tests a compact context block only; it does not create states, route models, or authorize a hard trading veto.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return {"winner": metrics.iloc[0].to_dict(), "operational": operations.to_dict("records"), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2727)
    parser.add_argument("--warmup-hours", type=int, default=720)
    parser.add_argument("--expected-run-hours", type=int, default=48)
    parser.add_argument("--max-run-hours", type=int, default=96)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument(
        "--arm",
        choices=("all", "transition_baseline", "changepoint_only", "transition_plus_changepoint"),
        default="all",
        help="Run one predeclared arm when environment limits require checkpoints.",
    )
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.50:
        parser.error("--calibration-fraction must be between 0.05 and 0.50")
    print(json.dumps(_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
