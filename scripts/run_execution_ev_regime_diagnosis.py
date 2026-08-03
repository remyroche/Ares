#!/usr/bin/env python3
"""Plan or run bounded forward/reversed regime diagnostics for a fixed EV model.

The default is a dry run that writes only the temporal split ledger.  Passing
``--execute`` fits the declared fixed ExtraTrees control; production users may
instead call :func:`evaluate_regime_diagnosis` with their frozen winning model's
fit/predict hook.  Reverse-time results are explicitly diagnostic and non-OOS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_regime_diagnosis import (  # noqa: E402
    RegimeDiagnosisConfig,
    build_regime_diagnosis_splits,
    evaluate_regime_diagnosis,
    feature_regime_diagnostics,
    regime_diagnosis_manifest,
    split_audit_frame,
    validate_regime_diagnosis_input,
)


def _parse_columns(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns or len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError("columns must be a non-empty unique list")
    return columns


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _fixed_extra_trees_hook(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
    n_jobs: int,
):
    """Return a fixed-parameter model hook; it performs no HPO or calibration."""

    def fit_predict(
        train_x: pd.DataFrame,
        train_y: np.ndarray,
        evaluation_x: pd.DataFrame,
        sample_weight: np.ndarray | None,
    ) -> np.ndarray:
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            max_features=1.0,
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
        model.fit(train_x, train_y, sample_weight=sample_weight)
        return np.asarray(model.predict(evaluation_x), dtype=float)

    return fit_predict


def _fixed_catboost_residual_hook(
    *,
    baseline_column: str,
    n_estimators: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_strength: float,
    bagging_temperature: float,
    random_state: int,
    n_jobs: int,
):
    """Return the frozen winning residual-CatBoost architecture.

    The model learns net EV minus the frozen alpha EV and is translated back
    to absolute net EV before global top-k evaluation.  No tuning, evaluation
    labels, or evaluation-derived calibration enters the fit.
    """

    def fit_predict(
        train_x: pd.DataFrame,
        train_y: np.ndarray,
        evaluation_x: pd.DataFrame,
        sample_weight: np.ndarray | None,
    ) -> np.ndarray:
        from catboost import CatBoostRegressor

        if baseline_column not in train_x or baseline_column not in evaluation_x:
            raise ValueError(
                f"residual baseline column {baseline_column!r} is missing"
            )
        train_baseline = train_x[baseline_column].to_numpy(dtype=float)
        evaluation_baseline = evaluation_x[baseline_column].to_numpy(dtype=float)
        model = CatBoostRegressor(
            loss_function="MAE",
            iterations=int(n_estimators),
            learning_rate=float(learning_rate),
            depth=int(depth),
            l2_leaf_reg=float(l2_leaf_reg),
            random_strength=float(random_strength),
            bagging_temperature=float(bagging_temperature),
            bootstrap_type="Bayesian",
            random_seed=int(random_state),
            thread_count=int(n_jobs),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_x,
            np.asarray(train_y, dtype=float) - train_baseline,
            sample_weight=sample_weight,
        )
        return (
            evaluation_baseline
            + np.asarray(model.predict(evaluation_x), dtype=float)
        )

    return fit_predict


def _sample_weight_hook(column: str):
    def hook(train: pd.DataFrame) -> np.ndarray:
        if column not in train.columns:
            raise ValueError(f"sample-weight column {column!r} is missing")
        return pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)

    return hook


def _recency_sample_weight_hook(decision_time_col: str, half_life_days: float):
    if not np.isfinite(half_life_days) or half_life_days <= 0.0:
        raise ValueError("recency half-life must be finite and positive")

    def hook(train: pd.DataFrame) -> np.ndarray:
        decision = pd.to_datetime(train[decision_time_col], utc=True, errors="raise")
        age_days = (decision.max() - decision).dt.total_seconds().to_numpy() / 86400.0
        return np.exp2(-age_days / float(half_life_days))

    return hook


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cols", type=_parse_columns, required=True)
    parser.add_argument("--catboost-archetype-col", default=None)
    parser.add_argument("--archetype-levels", type=_parse_columns, default=None)
    parser.add_argument(
        "--configuration-name", default="fixed_execution_ev_configuration"
    )
    parser.add_argument("--decision-time-col", default="execution_decision_utc")
    parser.add_argument("--label-resolution-col", default="execution_label_end_utc")
    parser.add_argument("--target-col", default="execution_net_ev_12h")
    parser.add_argument(
        "--side",
        choices=("long", "short"),
        default=None,
        help="Optional side-local diagnosis; required for side-specific production research.",
    )
    parser.add_argument("--train-window-months", type=int, default=3)
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--max-periods", type=int, default=6)
    parser.add_argument("--start-month", default=None)
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--sample-weight-col", default=None)
    parser.add_argument("--recency-half-life-days", type=float, default=None)
    parser.add_argument(
        "--model",
        choices=("extra_trees_control", "catboost_residual_winner"),
        default="extra_trees_control",
    )
    parser.add_argument("--baseline-col", default="existing_alpha_ev")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--random-strength", type=float, default=0.5)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Fit the fixed bounded control. Without this, only write the split "
            "ledger."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    frame = pd.read_parquet(args.input)
    if args.side is not None:
        if "side_name" not in frame:
            raise ValueError("--side requires a side_name column")
        frame = frame.loc[
            frame["side_name"].astype(str).str.lower().eq(str(args.side))
        ].copy()
        if frame.empty:
            raise ValueError(f"no rows for requested side {args.side!r}")
    if args.sample_weight_col and args.recency_half_life_days is not None:
        raise ValueError(
            "sample-weight column and recency half-life are mutually exclusive"
        )
    feature_columns = list(args.feature_cols)
    if args.catboost_archetype_col or args.archetype_levels:
        if not args.catboost_archetype_col or not args.archetype_levels:
            raise ValueError(
                "catboost archetype column and fixed levels must be supplied together"
            )
        archetype_col = str(args.catboost_archetype_col)
        if archetype_col not in frame:
            raise ValueError(f"categorical archetype column {archetype_col!r} is missing")
        archetype = frame[archetype_col].astype(str)
        unknown = sorted(set(archetype.unique()).difference(args.archetype_levels))
        if unknown:
            raise ValueError(f"unrecognized fixed archetype levels: {unknown}")
        for level in args.archetype_levels:
            name = f"catboost_archetype__{level}"
            frame[name] = archetype.eq(level).astype("float32")
            if name not in feature_columns:
                feature_columns.append(name)
    config = RegimeDiagnosisConfig(
        decision_time_col=str(args.decision_time_col),
        label_resolution_col=str(args.label_resolution_col),
        target_col=str(args.target_col),
        train_window_months=int(args.train_window_months),
        purge_hours=float(args.purge_hours),
        min_train_rows=int(args.min_train_rows),
        top_k_fraction=float(args.top_k_fraction),
        huber_delta=float(args.huber_delta),
        max_periods=int(args.max_periods) if args.max_periods is not None else None,
        random_state=int(args.random_state),
    )
    work = validate_regime_diagnosis_input(frame, feature_columns, config=config)
    splits = build_regime_diagnosis_splits(
        work,
        config=config,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    if not splits:
        raise ValueError("no bounded regime-diagnosis split satisfies the contract")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"splits": root / "regime_diagnosis_splits.csv"}
    split_audit_frame(splits).to_csv(paths["splits"], index=False)
    paths["feature_diagnostics"] = root / "regime_feature_diagnostics.csv"
    feature_regime_diagnostics(
        frame, feature_columns, config=config
    ).to_csv(paths["feature_diagnostics"], index=False)

    result = None
    if bool(args.execute):
        if args.model == "catboost_residual_winner":
            fit_predict = _fixed_catboost_residual_hook(
                baseline_column=str(args.baseline_col),
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                depth=int(args.max_depth),
                l2_leaf_reg=float(args.l2_leaf_reg),
                random_strength=float(args.random_strength),
                bagging_temperature=float(args.bagging_temperature),
                random_state=int(args.random_state),
                n_jobs=int(args.n_jobs),
            )
        else:
            fit_predict = _fixed_extra_trees_hook(
                n_estimators=int(args.n_estimators),
                max_depth=(int(args.max_depth) if args.max_depth is not None else None),
                min_samples_leaf=int(args.min_samples_leaf),
                random_state=int(args.random_state),
                n_jobs=int(args.n_jobs),
            )
        result = evaluate_regime_diagnosis(
            frame,
            feature_columns,
            fit_predict,
            config=config,
            sample_weight_hook=(
                _sample_weight_hook(str(args.sample_weight_col))
                if args.sample_weight_col
                else (
                    _recency_sample_weight_hook(
                        str(args.decision_time_col),
                        float(args.recency_half_life_days),
                    )
                    if args.recency_half_life_days is not None
                    else None
                )
            ),
            start_month=args.start_month,
            end_month=args.end_month,
        )
        paths["metrics"] = root / "regime_diagnosis_metrics.csv"
        paths["predictions"] = root / "regime_diagnosis_predictions.parquet"
        result.metrics.to_csv(paths["metrics"], index=False)
        try:
            result.predictions.to_parquet(paths["predictions"], index=False)
        except (ImportError, ValueError):
            paths["predictions"] = root / "regime_diagnosis_predictions.pkl"
            result.predictions.to_pickle(paths["predictions"])

    paths["manifest"] = root / "regime_diagnosis_manifest.json"
    _write_json(
        paths["manifest"],
        {
            **regime_diagnosis_manifest(
                result,
                config=config,
                feature_columns=feature_columns,
                configuration_name=str(args.configuration_name),
                split_count=len(splits),
            ),
            "input": {"path": str(args.input), "rows": int(len(work))},
            "side": args.side,
            "execution": {
                "requested": bool(args.execute),
                "model": (
                    str(args.model)
                    if args.execute
                    else None
                ),
                "target_mode": (
                    "residual_plus_frozen_alpha"
                    if args.model == "catboost_residual_winner"
                    else "direct"
                ),
                "baseline_column": (
                    str(args.baseline_col)
                    if args.model == "catboost_residual_winner"
                    else None
                ),
                "sample_weight_column": args.sample_weight_col,
                "recency_half_life_days": args.recency_half_life_days,
            },
        },
    )
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = run(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
