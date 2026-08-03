#!/usr/bin/env python3
"""Test whether an economic regime is observable from decision-time inputs.

Two products are kept deliberately separate:

1. frozen-reference robust shift/OOD summaries, fitted only on rows before each
   evaluation week; and
2. a week-outcome classifier whose target is whether the frozen score's single
   pooled-global top-k book was profitable in that week.

The classifier is fit only on *completed* earlier weeks.  A week becomes
eligible for training after its final candidate label has resolved.  Rows are
weighted inversely by week support so a dense week cannot masquerade as
independent regime evidence.  This is an observability diagnostic, not a trade
gate and not promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCHEMA = "chronological_regime_observability_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
TARGET = "execution_net_ev_12h"
WEEK = "__week_start__"
WEEK_LABEL = "__week_profitable__"
WEEK_EV = "__week_topk_net_ev__"

HEAD_FEATURES = (
    "existing_alpha_ev",
    "pred_peak_MFE_12h_ATR",
    "catboost_entropy",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
    "base_oof_score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "oof_clean_favorable_probability",
    *(f"catboost_p_{index}" for index in range(7)),
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _week_start(values: pd.Series) -> pd.Series:
    values = pd.to_datetime(values, utc=True, errors="raise")
    return values.dt.floor("D") - pd.to_timedelta(values.dt.dayofweek, unit="D")


def feature_families(columns: Iterable[str]) -> dict[str, list[str]]:
    """Route only contemporaneous h0 fields into interpretable fixed families."""

    available = set(map(str, columns))
    families: dict[str, list[str]] = {
        "head_context": [column for column in HEAD_FEATURES if column in available],
        "volatility": [],
        "trend_range": [],
        "breadth": [],
        "correlation": [],
        "funding": [],
        "open_interest": [],
        "liquidation_proxy": [],
    }
    for column in sorted(available):
        if not column.startswith("mkt_state__") or not column.endswith("__h0"):
            continue
        lowered = column.lower()
        if any(token in lowered for token in ("volatility", "atr_")):
            families["volatility"].append(column)
        elif any(token in lowered for token in ("efficiency", "trend", "range_", "breakout")):
            families["trend_range"].append(column)
        elif "breadth" in lowered:
            families["breadth"].append(column)
        elif "corr" in lowered:
            families["correlation"].append(column)
        elif "funding" in lowered:
            families["funding"].append(column)
        elif any(token in lowered for token in ("_oi_", "oi_")):
            families["open_interest"].append(column)
        elif "liquidation" in lowered:
            families["liquidation_proxy"].append(column)
    market = [
        column
        for name, columns_ in families.items()
        if name != "head_context"
        for column in columns_
    ]
    families["market_h0"] = list(dict.fromkeys(market))
    families["head_plus_market_h0"] = list(
        dict.fromkeys([*families["head_context"], *market])
    )
    return {name: values for name, values in families.items() if values}


def join_score_ledger(
    input_path: Path,
    score_ledger_path: Path,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_parquet(input_path)
    score = pd.read_parquet(score_ledger_path)
    required_input = [*IDENTITY, DECISION, RESOLUTION, TARGET]
    missing = [column for column in required_input if column not in frame]
    if missing:
        raise ValueError("input is missing columns: " + ", ".join(missing))
    missing = [column for column in (*IDENTITY, score_column) if column not in score]
    if missing:
        raise ValueError("score ledger is missing columns: " + ", ".join(missing))
    if frame.duplicated(list(IDENTITY)).any() or score.duplicated(list(IDENTITY)).any():
        raise ValueError("input and score ledger identities must be unique")
    work = frame.merge(
        score.loc[:, [*IDENTITY, score_column]],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    audit = {
        "input_rows": int(len(frame)),
        "score_rows": int(len(score)),
        "identity_intersection_rows": int(len(work)),
    }
    return work, audit


def add_week_economic_labels(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_k_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError("top-k fraction must be in (0, 1]")
    work = frame.copy()
    work[DECISION] = pd.to_datetime(work[DECISION], utc=True, errors="raise")
    work[RESOLUTION] = pd.to_datetime(work[RESOLUTION], utc=True, errors="raise")
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work = work.loc[
        work[TARGET].notna()
        & work[score_column].notna()
        & work[DECISION].notna()
        & work[RESOLUTION].notna()
    ].copy()
    if (work[RESOLUTION] < work[DECISION]).any():
        raise ValueError("label resolution cannot precede decision time")
    work[WEEK] = _week_start(work[DECISION])
    summaries: list[dict[str, Any]] = []
    for week, positions in work.groupby(WEEK, sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        sample = work.iloc[index]
        count = max(1, int(np.ceil(len(sample) * top_k_fraction)))
        selected = sample.nlargest(count, score_column, keep="first")
        topk_ev = float(selected[TARGET].mean())
        summaries.append(
            {
                WEEK: week,
                "week_end_exclusive": week + pd.Timedelta(days=7),
                "week_label_available_at": max(
                    week + pd.Timedelta(days=7),
                    work.iloc[index][RESOLUTION].max(),
                ),
                "population_rows": int(len(sample)),
                "topk_rows": int(len(selected)),
                WEEK_EV: topk_ev,
                WEEK_LABEL: int(topk_ev > 0.0),
            }
        )
    weeks = pd.DataFrame(summaries).sort_values(WEEK).reset_index(drop=True)
    work = work.merge(weeks[[WEEK, WEEK_EV, WEEK_LABEL, "week_label_available_at"]], on=WEEK)
    return work.sort_values(DECISION).reset_index(drop=True), weeks


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    return (
        float(roc_auc_score(target, score))
        if np.unique(target).size >= 2
        else float("nan")
    )


def _safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    return (
        float(average_precision_score(target, score))
        if np.unique(target).size >= 2
        else float("nan")
    )


def _train_feature_columns(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    train: np.ndarray,
    *,
    min_coverage: float,
) -> list[str]:
    numeric = frame.loc[train, list(candidates)].apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean()
    variance = numeric.var(skipna=True)
    return [
        column
        for column in candidates
        if coverage.get(column, 0.0) >= min_coverage
        and np.isfinite(variance.get(column, np.nan))
        and variance[column] > 0.0
    ]


def _robust_shift(
    train_values: pd.DataFrame,
    evaluation_values: pd.DataFrame,
) -> tuple[float, str | None, float]:
    train_median = train_values.median()
    train_q25 = train_values.quantile(0.25)
    train_q75 = train_values.quantile(0.75)
    scale = (train_q75 - train_q25).replace(0.0, np.nan)
    shift = ((evaluation_values.median() - train_median) / scale).abs()
    finite = shift.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float("nan"), None, float("nan")
    top = str(finite.idxmax())
    return float(finite.mean()), top, float(finite.loc[top])


def chronological_observability(
    frame: pd.DataFrame,
    weeks: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    *,
    first_evaluation: pd.Timestamp,
    min_train_weeks: int,
    min_feature_coverage: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit fixed logistic models on earlier completed week labels only."""

    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for evaluation_week in weeks.loc[weeks[WEEK] >= first_evaluation, WEEK]:
        evaluation_start = pd.Timestamp(evaluation_week)
        completed_weeks = weeks.loc[
            weeks["week_label_available_at"] < evaluation_start, WEEK
        ]
        if len(completed_weeks) < min_train_weeks:
            continue
        train = frame[WEEK].isin(completed_weeks).to_numpy()
        valid = frame[WEEK].eq(evaluation_week).to_numpy()
        if not valid.any():
            continue
        train_week_labels = (
            frame.loc[train, [WEEK, WEEK_LABEL]].drop_duplicates()[WEEK_LABEL]
        )
        for family, candidates in families.items():
            selected = _train_feature_columns(
                frame,
                candidates,
                train,
                min_coverage=min_feature_coverage,
            )
            if not selected:
                continue
            train_x = frame.loc[train, selected].apply(pd.to_numeric, errors="coerce")
            valid_x = frame.loc[valid, selected].apply(pd.to_numeric, errors="coerce")
            target = frame.loc[train, WEEK_LABEL].to_numpy(dtype=np.int8)
            week_counts = frame.loc[train, WEEK].value_counts()
            sample_weight = (
                frame.loc[train, WEEK].map(lambda value: 1.0 / week_counts.loc[value]).to_numpy()
            )
            sample_weight *= len(sample_weight) / sample_weight.sum()
            if np.unique(target).size < 2:
                probability = np.full(valid.sum(), float(target.mean()))
                model_status = "train_only_constant_prior"
                coefficients = np.zeros(len(selected), dtype=float)
            else:
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        C=0.25,
                        max_iter=2_000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                )
                model.fit(train_x, target, logisticregression__sample_weight=sample_weight)
                probability = model.predict_proba(valid_x)[:, 1]
                model_status = "fitted_fixed_logistic"
                coefficients = model.named_steps["logisticregression"].coef_[0]
            mean_shift, top_shift_feature, top_shift = _robust_shift(train_x, valid_x)
            actual = int(frame.loc[valid, WEEK_LABEL].iloc[0])
            predicted_week_probability = float(np.mean(probability))
            fold_rows.append(
                {
                    "feature_family": family,
                    "evaluation_week": evaluation_week,
                    "train_rows": int(train.sum()),
                    "train_weeks": int(len(completed_weeks)),
                    "train_positive_weeks": int(train_week_labels.sum()),
                    "train_negative_weeks": int(len(train_week_labels) - train_week_labels.sum()),
                    "evaluation_rows": int(valid.sum()),
                    "selected_features": int(len(selected)),
                    "model_status": model_status,
                    "actual_profitable_week": actual,
                    "actual_week_topk_net_ev_bps": float(frame.loc[valid, WEEK_EV].iloc[0] * 10_000.0),
                    "predicted_profitable_probability": predicted_week_probability,
                    "prediction_correct_at_050": int((predicted_week_probability >= 0.5) == bool(actual)),
                    "mean_abs_robust_median_shift": mean_shift,
                    "top_shift_feature": top_shift_feature,
                    "top_abs_robust_median_shift": top_shift,
                    "train_label_cutoff_utc": evaluation_start,
                }
            )
            pred = frame.loc[valid, [*IDENTITY, DECISION, WEEK, WEEK_LABEL, WEEK_EV]].copy()
            pred["feature_family"] = family
            pred["profitable_week_probability"] = probability
            pred["observability_oos"] = True
            pred["train_label_cutoff_utc"] = evaluation_start
            predictions.append(pred)
            for feature, coefficient in zip(selected, coefficients, strict=True):
                importance_rows.append(
                    {
                        "feature_family": family,
                        "evaluation_week": evaluation_week,
                        "feature": feature,
                        "coefficient": float(coefficient),
                        "abs_coefficient": float(abs(coefficient)),
                    }
                )
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    importance = pd.DataFrame(importance_rows)
    return prediction_frame, folds, importance


def aggregate_observability_metrics(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for family, sample in predictions.groupby("feature_family", sort=True):
        target = sample[WEEK_LABEL].to_numpy(dtype=np.int8)
        score = sample["profitable_week_probability"].to_numpy(dtype=float)
        week_sample = folds.loc[folds["feature_family"].eq(family)]
        week_target = week_sample["actual_profitable_week"].to_numpy(dtype=np.int8)
        week_score = week_sample["predicted_profitable_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_family": family,
                "oos_rows": int(len(sample)),
                "oos_weeks": int(len(week_sample)),
                "positive_weeks": int(week_target.sum()),
                "negative_weeks": int(len(week_target) - week_target.sum()),
                "row_auc": _safe_auc(target, score),
                "row_average_precision": _safe_ap(target, score),
                "row_brier": float(brier_score_loss(target, score)),
                "week_auc": _safe_auc(week_target, week_score),
                "week_average_precision": _safe_ap(week_target, week_score),
                "week_brier": float(brier_score_loss(week_target, week_score)),
                "week_accuracy_at_050": float(
                    ((week_score >= 0.5) == week_target.astype(bool)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, join_audit = join_score_ledger(args.input, args.score_ledger, args.score_column)
    frame, weeks = add_week_economic_labels(
        frame,
        score_column=args.score_column,
        top_k_fraction=float(args.top_k_fraction),
    )
    families = feature_families(frame.columns)
    requested = (
        list(families)
        if not args.feature_family
        else list(dict.fromkeys(args.feature_family))
    )
    missing = [family for family in requested if family not in families]
    if missing:
        raise ValueError("unknown or unavailable feature families: " + ", ".join(missing))
    chosen = {family: families[family] for family in requested}
    first_evaluation = pd.Timestamp(args.first_evaluation)
    if first_evaluation.tzinfo is None:
        first_evaluation = first_evaluation.tz_localize("UTC")
    else:
        first_evaluation = first_evaluation.tz_convert("UTC")
    predictions, folds, importance = chronological_observability(
        frame,
        weeks,
        chosen,
        first_evaluation=first_evaluation,
        min_train_weeks=int(args.min_train_weeks),
        min_feature_coverage=float(args.min_feature_coverage),
        seed=int(args.seed),
    )
    metrics = aggregate_observability_metrics(predictions, folds)
    args.output_dir.mkdir(parents=True)
    outputs = {
        "week_economics": args.output_dir / "week_economics.csv",
        "fold_metrics": args.output_dir / "observability_fold_metrics.csv",
        "aggregate_metrics": args.output_dir / "observability_aggregate_metrics.csv",
        "feature_coefficients": args.output_dir / "observability_feature_coefficients.csv",
        "oos_predictions": args.output_dir / "observability_oos_predictions.parquet",
    }
    weeks.to_csv(outputs["week_economics"], index=False)
    folds.to_csv(outputs["fold_metrics"], index=False)
    metrics.to_csv(outputs["aggregate_metrics"], index=False)
    importance.to_csv(outputs["feature_coefficients"], index=False)
    predictions.to_parquet(outputs["oos_predictions"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "diagnostic_only_not_a_trade_gate_or_promotion_input",
        "contract": {
            "score": args.score_column,
            "week_label": "positive iff the week's one pooled-global top-k mean execution_net_ev_12h is above zero",
            "ranking_scope": "within each week, pooled globally across timestamps and sides; never per timestamp or side",
            "causality": "each fold trains only on weeks whose final label resolved strictly before evaluation-week start",
            "row_weighting": "inverse rows per training week; each completed week has equal total loss weight",
            "model": "fixed median-imputed standardized logistic regression C=0.25; no HPO",
            "shift": "evaluation median relative to earlier-train IQR; no evaluation-fitted transform",
            "effective_sample_warning": "week metrics, not row count, determine regime evidence strength",
        },
        "inputs": {
            "input": str(args.input),
            "input_sha256": _sha256(args.input),
            "score_ledger": str(args.score_ledger),
            "score_ledger_sha256": _sha256(args.score_ledger),
            "join_audit": join_audit,
        },
        "feature_families": chosen,
        "first_evaluation": first_evaluation,
        "min_train_weeks": int(args.min_train_weeks),
        "min_feature_coverage": float(args.min_feature_coverage),
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--score-ledger", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--first-evaluation", default="2026-06-15")
    parser.add_argument("--min-train-weeks", type=int, default=4)
    parser.add_argument("--min-feature-coverage", type=float, default=0.95)
    parser.add_argument("--feature-family", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
