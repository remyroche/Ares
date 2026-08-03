#!/usr/bin/env python3
"""Nested side-local feature screen for exact-policy capture probability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    fit_train_only_isotonic_ev_mapping,
)
from scripts.run_execution_ev_mixed_period_remedies import (  # noqa: E402
    ARCHETYPE_COLUMN,
    DECISION_COLUMN,
    DEFAULT_WINDOWS,
    IDENTITY_COLUMNS,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    _temporal_oof_blocks,
    build_forward_split,
)
from scripts.run_exact_policy_capture_hurdle_ablation import (  # noqa: E402
    _fit_or_constant_classifier,
    _predict_classifier,
)
from scripts.run_exact_policy_capture_support_ablation import (  # noqa: E402
    apply_recent_mapping_scope,
)

SCHEMA = "capture_specific_nested_feature_screen_v1"
SIDES = ("long", "short")
ARMS = ("core_capture", "capture_selected_raw")
PREFIX = "capture_candidate__"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def feature_family(column: str) -> str:
    name = column.removeprefix(PREFIX).lower()
    rules = (
        ("leverage", ("oi", "funding", "liquid", "leverage", "flush")),
        ("volatility", ("atr", "vol", "range", "variance", "std", "entropy")),
        ("market", ("btc", "eth", "market", "mkt", "breadth", "corr", "universe")),
        ("liquidity", ("spread", "amihud", "volume", "impact", "liquidity")),
        ("trend", ("ret", "mom", "trend", "breakout", "ema", "vwap", "slope")),
        ("microstructure", ("clv", "body", "wick", "delta", "order", "rejection")),
        ("time", ("hour", "dow", "calendar", "session")),
    )
    for family, tokens in rules:
        if any(token in name for token in tokens):
            return family
    return "other"


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _segment_statistics(
    values: np.ndarray,
    capture: np.ndarray,
    net: np.ndarray,
) -> tuple[float, float]:
    finite = np.isfinite(values)
    if finite.sum() < 200 or np.unique(values[finite]).size < 2:
        return np.nan, np.nan
    ic = pd.Series(values[finite]).corr(
        pd.Series(capture[finite]), method="spearman"
    )
    if not np.isfinite(ic) or abs(ic) < 1e-12:
        return float(ic), np.nan
    count = max(1, int(np.ceil(0.10 * finite.sum())))
    positions = np.flatnonzero(finite)
    order = np.argsort(
        -np.sign(ic) * values[positions], kind="mergesort"
    )[:count]
    selected = positions[order]
    lift_bps = float((net[selected].mean() - net[finite].mean()) * 10_000.0)
    return float(ic), lift_bps


def select_capture_features(
    fit: pd.DataFrame,
    candidate_columns: Sequence[str],
    *,
    max_features: int,
    minimum_coverage: float,
    maximum_per_family: int,
    correlation_cap: float,
) -> tuple[list[str], dict[str, Any]]:
    decision = pd.to_datetime(fit[DECISION_COLUMN], utc=True, errors="raise")
    unique_days = pd.Index(decision.dt.floor("D").unique()).sort_values()
    if len(unique_days) < 4:
        raise ValueError("capture feature selection requires at least four days")
    boundary = unique_days[len(unique_days) // 2]
    segment_masks = (
        decision.dt.floor("D").lt(boundary).to_numpy(),
        decision.dt.floor("D").ge(boundary).to_numpy(),
    )
    capture = (fit[TARGET_COLUMN].to_numpy(dtype=float) > 0.0).astype(float)
    net = fit[TARGET_COLUMN].to_numpy(dtype=float)
    matrix = _numeric(fit, candidate_columns)
    rows = []
    for column in candidate_columns:
        values = matrix[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        coverage = float(finite.mean())
        variance = float(np.nanvar(values))
        segment = [
            _segment_statistics(values[mask], capture[mask], net[mask])
            for mask in segment_masks
        ]
        ics = [value[0] for value in segment]
        lifts = [value[1] for value in segment]
        stable_sign = (
            all(np.isfinite(value) for value in ics)
            and ics[0] * ics[1] > 0.0
        )
        robust_ic = min(abs(value) for value in ics) if stable_sign else 0.0
        robust_lift = (
            min(lifts) if all(np.isfinite(value) for value in lifts) else -np.inf
        )
        eligible = (
            coverage >= float(minimum_coverage)
            and np.isfinite(variance)
            and variance > 1e-10
            and stable_sign
            and robust_lift > 0.0
        )
        rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "coverage": coverage,
                "variance": variance,
                "early_ic": ics[0],
                "late_ic": ics[1],
                "early_net_lift_bps": lifts[0],
                "late_net_lift_bps": lifts[1],
                "robust_ic": robust_ic,
                "robust_net_lift_bps": robust_lift,
                "selection_score": (
                    robust_lift + 50.0 * robust_ic if eligible else -np.inf
                ),
                "eligible": bool(eligible),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["eligible", "selection_score", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    filled = matrix.copy()
    for column in candidate_columns:
        median = float(filled[column].median())
        filled[column] = filled[column].fillna(median)
    for row in ranking.loc[ranking["eligible"]].itertuples(index=False):
        if len(selected) >= int(max_features):
            break
        family = str(row.family)
        if family_counts.get(family, 0) >= int(maximum_per_family):
            continue
        redundant = False
        for existing in selected:
            correlation = filled[column := str(row.feature)].corr(filled[existing])
            if np.isfinite(correlation) and abs(correlation) >= float(correlation_cap):
                redundant = True
                break
        if redundant:
            continue
        selected.append(str(row.feature))
        family_counts[family] = family_counts.get(family, 0) + 1
    return selected, {
        "status": "selected" if selected else "no_stable_positive_additions",
        "fit_rows": int(len(fit)),
        "fit_start_utc": decision.min(),
        "fit_end_utc": decision.max(),
        "segment_boundary_utc": pd.Timestamp(boundary),
        "candidate_features": int(len(candidate_columns)),
        "eligible_features": int(ranking["eligible"].sum()),
        "selected_features": selected,
        "selected_count": int(len(selected)),
        "family_counts": family_counts,
        "ranking": ranking.to_dict(orient="records"),
    }


def _fit_capture(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    columns: Sequence[str],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    fit_x = _numeric(fit, columns)
    score_x = _numeric(score, columns)
    target = (fit[TARGET_COLUMN].to_numpy(dtype=float) > 0.0).astype(np.int8)
    model, constant = _fit_or_constant_classifier(
        fit_x,
        target,
        iterations=iterations,
        seed=seed,
        n_jobs=n_jobs,
    )
    return _predict_classifier(model, constant, score_x)


def _metric_row(
    evaluation: pd.DataFrame,
    score: np.ndarray,
    *,
    window: str,
    arm: str,
    stage: str,
    scope: str,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    mask = (
        np.ones(len(evaluation), dtype=bool)
        if scope == "pooled_global"
        else evaluation[SIDE_COLUMN]
        .astype(str)
        .eq(scope.removeprefix("side_"))
        .to_numpy()
    )
    sample = evaluation.loc[mask].reset_index(drop=True)
    prediction = np.asarray(score, dtype=float)[mask]
    target = sample[TARGET_COLUMN].to_numpy(dtype=float)
    capture = (target > 0.0).astype(np.int8)
    count = max(1, int(np.ceil(0.10 * len(sample))))
    selected_pos = np.argsort(-prediction, kind="mergesort")[:count]
    selected = sample.iloc[selected_pos]
    selected_decision = pd.to_datetime(
        selected[DECISION_COLUMN], utc=True, errors="raise"
    )
    latest_start = pd.to_datetime(
        sample[DECISION_COLUMN], utc=True, errors="raise"
    ).max() - pd.Timedelta(days=7)
    latest = selected.loc[selected_decision.ge(latest_start)]
    return {
        "window": window,
        "arm": arm,
        "stage": stage,
        "scope": scope,
        "rows": int(len(sample)),
        "auc": (
            float(roc_auc_score(capture, prediction))
            if np.unique(capture).size == 2
            else np.nan
        ),
        "spearman": float(
            pd.Series(prediction).corr(pd.Series(target), method="spearman")
        ),
        "top10_rows": int(count),
        "top10_net_bps": float(selected[TARGET_COLUMN].mean() * 10_000.0),
        "top10_positive_rate": float((selected[TARGET_COLUMN] > 0.0).mean()),
        "latest_7d_selected_rows": int(len(latest)),
        "latest_7d_net_bps": (
            float(latest[TARGET_COLUMN].mean() * 10_000.0)
            if len(latest)
            else np.nan
        ),
        "first_selected_utc": selected_decision.min(),
        "last_selected_utc": selected_decision.max(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = pd.read_parquet(args.input)
    feature_universe = json.loads(args.feature_universe_manifest.read_text())
    candidate_columns = list(
        feature_universe["eligible_full_period_feature_columns"]
    )
    core_manifest = json.loads(args.core_feature_manifest.read_text())
    core_columns = list(core_manifest["feature_columns"])
    for column in core_columns:
        prefix = "catboost_archetype__"
        if column.startswith(prefix) and column not in frame:
            level = column[len(prefix) :]
            frame[column] = (
                frame[ARCHETYPE_COLUMN].astype(str).eq(level).astype(np.float32)
            )
    required = [
        *IDENTITY_COLUMNS,
        DECISION_COLUMN,
        RESOLUTION_COLUMN,
        TARGET_COLUMN,
        SIDE_COLUMN,
        *core_columns,
        *candidate_columns,
    ]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError("capture feature screen missing columns: " + ", ".join(missing))
    frame = frame.sort_values(
        [DECISION_COLUMN, "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    metric_rows = []
    prediction_parts = []
    reports: dict[str, Any] = {}
    selection_rows = []
    for window_index, window in enumerate(DEFAULT_WINDOWS):
        train_pos, evaluation_pos, split = build_forward_split(
            frame, window, purge_hours=args.purge_hours
        )
        train = frame.iloc[train_pos].copy().reset_index(drop=True)
        evaluation = frame.iloc[evaluation_pos].copy().reset_index(drop=True)
        train_scores = {arm: np.full(len(train), np.nan) for arm in ARMS}
        evaluation_scores = {arm: np.full(len(evaluation), np.nan) for arm in ARMS}
        raw_evaluation_scores = {
            arm: np.full(len(evaluation), np.nan) for arm in ARMS
        }
        window_report: dict[str, Any] = {"split": split, "sides": {}}
        for side_index, side in enumerate(SIDES):
            train_side = train.loc[train[SIDE_COLUMN].astype(str).eq(side)].copy()
            train_side["__global_position__"] = train_side.index
            train_side = train_side.reset_index(drop=True)
            eval_side = evaluation.loc[
                evaluation[SIDE_COLUMN].astype(str).eq(side)
            ].copy()
            eval_side["__global_position__"] = eval_side.index
            eval_side = eval_side.reset_index(drop=True)
            raw_oof = {arm: np.full(len(train_side), np.nan) for arm in ARMS}
            fold_reports = []
            for fold, (fit_pos, valid_pos) in enumerate(
                _temporal_oof_blocks(train_side, min_train_rows=2_000), start=1
            ):
                fit = train_side.iloc[fit_pos]
                valid = train_side.iloc[valid_pos]
                selected, selection = select_capture_features(
                    fit,
                    candidate_columns,
                    max_features=args.max_selected_features,
                    minimum_coverage=args.minimum_train_coverage,
                    maximum_per_family=args.maximum_per_family,
                    correlation_cap=args.correlation_cap,
                )
                raw_oof["core_capture"][valid_pos] = _fit_capture(
                    fit,
                    valid,
                    core_columns,
                    iterations=args.n_estimators,
                    seed=args.random_state
                    + 100_000 * window_index
                    + 10_000 * side_index
                    + 100 * fold,
                    n_jobs=args.n_jobs,
                )
                raw_oof["capture_selected_raw"][valid_pos] = _fit_capture(
                    fit,
                    valid,
                    [*core_columns, *selected],
                    iterations=args.n_estimators,
                    seed=args.random_state
                    + 100_000 * window_index
                    + 10_000 * side_index
                    + 100 * fold
                    + 1,
                    n_jobs=args.n_jobs,
                )
                for row in selection["ranking"]:
                    selection_rows.append(
                        {
                            "window": window.name,
                            "side_name": side,
                            "selection_stage": f"oof_fold_{fold}",
                            **row,
                        }
                    )
                fold_reports.append(
                    {
                        "fold": fold,
                        "fit_rows": int(len(fit)),
                        "validation_rows": int(len(valid)),
                        "selection": {
                            key: value
                            for key, value in selection.items()
                            if key != "ranking"
                        },
                    }
                )
            selected_final, final_selection = select_capture_features(
                train_side,
                candidate_columns,
                max_features=args.max_selected_features,
                minimum_coverage=args.minimum_train_coverage,
                maximum_per_family=args.maximum_per_family,
                correlation_cap=args.correlation_cap,
            )
            raw_eval = {
                "core_capture": _fit_capture(
                    train_side,
                    eval_side,
                    core_columns,
                    iterations=args.n_estimators,
                    seed=args.random_state
                    + 100_000 * window_index
                    + 10_000 * side_index
                    + 9_000,
                    n_jobs=args.n_jobs,
                ),
                "capture_selected_raw": _fit_capture(
                    train_side,
                    eval_side,
                    [*core_columns, *selected_final],
                    iterations=args.n_estimators,
                    seed=args.random_state
                    + 100_000 * window_index
                    + 10_000 * side_index
                    + 9_001,
                    n_jobs=args.n_jobs,
                ),
            }
            for row in final_selection["ranking"]:
                selection_rows.append(
                    {
                        "window": window.name,
                        "side_name": side,
                        "selection_stage": "final_train",
                        **row,
                    }
                )
            global_train = train_side["__global_position__"].to_numpy(dtype=int)
            global_eval = eval_side["__global_position__"].to_numpy(dtype=int)
            for arm in ARMS:
                raw_evaluation_scores[arm][global_eval] = raw_eval[arm]
                mapper = fit_train_only_isotonic_ev_mapping(
                    raw_oof[arm],
                    train_side[TARGET_COLUMN].to_numpy(dtype=float),
                    min_rows=24,
                )
                finite = np.isfinite(raw_oof[arm])
                mapped_oof = np.full(len(train_side), np.nan)
                mapped_oof[finite] = mapper.predict(raw_oof[arm][finite])
                train_scores[arm][global_train] = mapped_oof
                evaluation_scores[arm][global_eval] = mapper.predict(raw_eval[arm])
            window_report["sides"][side] = {
                "train_rows": int(len(train_side)),
                "evaluation_rows": int(len(eval_side)),
                "folds": fold_reports,
                "final_selection": {
                    key: value
                    for key, value in final_selection.items()
                    if key != "ranking"
                },
            }
        for arm in ARMS:
            for scope in ("pooled_global", "side_long", "side_short"):
                metric_rows.append(
                    _metric_row(
                        evaluation,
                        raw_evaluation_scores[arm],
                        window=window.name,
                        arm=arm,
                        stage="raw_head_probability",
                        scope=scope,
                    )
                )
            for scope in ("pooled_global", "side_long", "side_short"):
                metric_rows.append(
                    _metric_row(
                        evaluation,
                        evaluation_scores[arm],
                        window=window.name,
                        arm=arm,
                        stage="pre_recent_mapping",
                        scope=scope,
                    )
                )
            mapped, mapping_report = apply_recent_mapping_scope(
                train,
                evaluation,
                train_scores[arm],
                evaluation_scores[arm],
                scope="global",
            )
            for scope in ("pooled_global", "side_long", "side_short"):
                metric_rows.append(
                    _metric_row(
                        evaluation,
                        mapped,
                        window=window.name,
                        arm=arm,
                        stage="causal_global_recent_mapping",
                        scope=scope,
                    )
                )
            part = evaluation.loc[:, list(IDENTITY_COLUMNS)].copy()
            part["window"] = window.name
            part["arm"] = arm
            part["raw_head_probability"] = raw_evaluation_scores[arm]
            part["pre_recent_ev_score"] = evaluation_scores[arm]
            part["causal_global_recent_ev_score"] = mapped
            prediction_parts.append(part)
            window_report.setdefault("mapping", {})[arm] = mapping_report
        reports[window.name] = window_report
    args.output_dir.mkdir(parents=True)
    paths = {
        "metrics": args.output_dir / "capture_feature_screen_metrics.csv",
        "predictions": args.output_dir / "capture_feature_screen_predictions.parquet",
        "selection": args.output_dir / "capture_feature_selection.csv",
    }
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        paths["predictions"], index=False
    )
    pd.DataFrame(selection_rows).to_csv(paths["selection"], index=False)
    output = {
        "schema": SCHEMA,
        "status": "completed_research_oos_not_promotion_evidence",
        "contract": {
            "feature_selection": "inside each side-local temporal fit only",
            "selector": "early/late stable signed IC and positive exact-net top-decile lift; family cap; correlation pruning",
            "models": "fixed CatBoost capture classifiers; no HPO",
            "mapping": "side-local train-OOF isotonic EV plus causal global 21-day recent correction",
            "ranking": "one pooled global top 10%; no timestamp or side quotas",
            "retention": "selected add-one must improve June and later July with adequate latest coverage",
        },
        "inputs": {
            "data": {"path": str(args.input), "sha256": _sha256(args.input)},
            "feature_universe_manifest": {
                "path": str(args.feature_universe_manifest),
                "sha256": _sha256(args.feature_universe_manifest),
            },
            "core_feature_manifest": {
                "path": str(args.core_feature_manifest),
                "sha256": _sha256(args.core_feature_manifest),
            },
        },
        "selector_parameters": {
            "candidate_features": int(len(candidate_columns)),
            "max_selected_features": int(args.max_selected_features),
            "minimum_train_coverage": float(args.minimum_train_coverage),
            "maximum_per_family": int(args.maximum_per_family),
            "correlation_cap": float(args.correlation_cap),
        },
        "model": {
            "iterations": int(args.n_estimators),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
        },
        "windows": reports,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet"
        ),
    )
    parser.add_argument(
        "--feature-universe-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/feature_universe_manifest.json"
        ),
    )
    parser.add_argument(
        "--core-feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/regime_diagnosis_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260727)
    parser.add_argument("--max-selected-features", type=int, default=24)
    parser.add_argument("--minimum-train-coverage", type=float, default=0.99)
    parser.add_argument("--maximum-per-family", type=int, default=4)
    parser.add_argument("--correlation-cap", type=float, default=0.95)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
