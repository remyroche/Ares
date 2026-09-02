#!/usr/bin/env python3
"""Fit a market-state short_boll rank-scope assessor.

The assessor learns a timestamp-level preference between the current T1
within-timestamp short_boll rank contract and the causal global-over-time rank
contract.  It emits the same formal router schedule consumed by
run_market_state_short_boll_rank_scope_switch.py:

* weight 1.0 -> use timestamp rank for short_boll;
* weight 0.0 -> use global-over-time rank for short_boll;
* intermediate values -> blend rank columns in shadow replay.

The model is shadow-only.  It does not change thresholds, q-fail, HeadHealth,
position sizing, score columns, or the active production stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STATE_PANEL = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260626_t1_lgbm_maturity_contract_v1"
    "/market_state_timestamp_panel.parquet"
)
DEFAULT_TIMESTAMP_UTILITY = Path(
    "data_perp/reports/t1_rank_contract_walkforward_20260626_prejune_timestamp_vs_global_v3"
    "/rank_contract_walkforward_timestamp_utility.csv"
)
DEFAULT_EVAL_STATE_PANEL = Path(
    "data_perp/reports/market_state_controller_bundle_t1_lgbm_maturity_noop_20260626"
    "/market_state_timestamp_panel.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_rank_scope_assessor_20260626_v1")
MARKET_STATE_PREFIXES = ("state_", "forecast_", "latent_")
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "coverage",
    "feature_count",
    "candidate",
    "strategy",
    "head",
    "rank",
    "score",
    "target",
    "label",
    "outcome",
    "pnl",
    "trade",
    "accepted",
    "rejected",
    "timestamp_count",
    "row_count",
    "count_",
    "_count",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _timestamp_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def dedupe_state_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one timestamp-level market-state row per timestamp."""

    if "timestamp" not in frame.columns:
        raise ValueError("state panel missing timestamp")
    work = frame.copy()
    work["timestamp"] = _timestamp_utc(work["timestamp"])
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        raise ValueError("state panel has no parseable timestamps")
    numeric_cols = [
        col
        for col in work.columns
        if col != "timestamp" and pd.api.types.is_numeric_dtype(work[col])
    ]
    if not numeric_cols:
        raise ValueError("state panel has no numeric market-state features")
    if work["timestamp"].duplicated().any():
        work = (
            work[["timestamp", *numeric_cols]]
            .groupby("timestamp", as_index=False, observed=True)
            .mean(numeric_only=True)
        )
    else:
        work = work[["timestamp", *numeric_cols]].copy()
    return work.sort_values("timestamp").reset_index(drop=True)


def select_market_state_features(frame: pd.DataFrame) -> list[str]:
    """Select causal timestamp-level state features and drop nuisance columns."""

    features: list[str] = []
    for col in frame.columns:
        if col == "timestamp":
            continue
        text = str(col).lower()
        if not str(col).startswith(MARKET_STATE_PREFIXES):
            continue
        if any(token in text for token in FORBIDDEN_FEATURE_SUBSTRINGS):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 3:
            continue
        if float(values.nunique(dropna=True)) <= 1.0:
            continue
        features.append(str(col))
    return sorted(features)


def build_training_table(
    state_panel: pd.DataFrame,
    timestamp_utility: pd.DataFrame,
    *,
    target_col: str,
    target_scale: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Join market-state features to timestamp rank-contract utility targets."""

    if target_col not in timestamp_utility.columns:
        raise ValueError(f"timestamp utility missing target column: {target_col}")
    util = timestamp_utility.copy()
    if "timestamp" not in util.columns:
        raise ValueError("timestamp utility missing timestamp")
    util["timestamp"] = _timestamp_utc(util["timestamp"])
    util = util.dropna(subset=["timestamp"])
    state = dedupe_state_panel(state_panel)
    train = state.merge(util, on="timestamp", how="inner")
    if "fold_y" in train.columns:
        train["fold"] = pd.to_numeric(train["fold_y"], errors="coerce")
        train = train.drop(columns=["fold_x", "fold_y"], errors="ignore")
    elif "fold" not in train.columns and "fold_x" in train.columns:
        train["fold"] = pd.to_numeric(train["fold_x"], errors="coerce")
        train = train.drop(columns=["fold_x"], errors="ignore")
    if train.empty:
        raise ValueError("no shared timestamps between market-state panel and rank utility")
    raw = pd.to_numeric(train[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = raw.notna()
    train = train.loc[finite].copy()
    raw = raw.loc[finite]
    if train.empty:
        raise ValueError("rank utility target has no finite values")
    if target_scale is None or not np.isfinite(target_scale) or target_scale <= 0.0:
        q75 = float(raw.quantile(0.75))
        q25 = float(raw.quantile(0.25))
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(raw.abs().median())
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(raw.abs().mean())
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
    else:
        scale = float(target_scale)
    train["rank_scope_target_raw"] = raw.to_numpy(dtype=float)
    train["rank_scope_target"] = np.tanh(raw.to_numpy(dtype=float) / scale)
    train["rank_scope_sample_weight"] = 1.0 + np.minimum(np.abs(raw.to_numpy(dtype=float)) / scale, 5.0)
    return train, float(scale)


def _feature_matrix(
    frame: pd.DataFrame,
    features: list[str],
    *,
    medians: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if not features:
        raise ValueError("no market-state features selected")
    work = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = {}
        for col in features:
            value = float(work[col].median())
            if not np.isfinite(value):
                value = 0.0
            medians[col] = value
    for col in features:
        work[col] = work[col].fillna(float(medians.get(col, 0.0)))
    return work.to_numpy(dtype=np.float32, copy=False), medians


def _make_model(backend: str, *, seed: int, n_rows: int) -> Any:
    backend = str(backend).lower()
    min_leaf = max(5, int(np.ceil(max(n_rows, 1) * 0.05)))
    if backend == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=120,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=7,
            min_child_samples=min_leaf,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=1.0,
            random_state=int(seed),
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
    if backend == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=120,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=min_leaf,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.10,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=1,
            tree_method="hist",
        )
    if backend == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=160,
            max_depth=4,
            min_samples_leaf=min_leaf,
            random_state=int(seed),
            n_jobs=1,
        )
    raise ValueError("backend must be one of: lgbm, xgb, rf")


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(model.predict(x), dtype=float)
    return np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)


def expanding_fold_diagnostics(
    train: pd.DataFrame,
    features: list[str],
    *,
    backend: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "fold" not in train.columns:
        return pd.DataFrame(), pd.DataFrame()
    folds = sorted(pd.to_numeric(train["fold"], errors="coerce").dropna().astype(int).unique())
    preds: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = pd.to_numeric(train["fold"], errors="coerce") < int(fold)
        valid_mask = pd.to_numeric(train["fold"], errors="coerce") == int(fold)
        if int(train_mask.sum()) < 20 or int(valid_mask.sum()) < 5:
            continue
        fit_rows = train.loc[train_mask].copy()
        valid_rows = train.loc[valid_mask].copy()
        x_train, medians = _feature_matrix(fit_rows, features)
        y_train = pd.to_numeric(fit_rows["rank_scope_target"], errors="coerce").to_numpy(dtype=float)
        w_train = pd.to_numeric(fit_rows["rank_scope_sample_weight"], errors="coerce").to_numpy(dtype=float)
        model = _make_model(backend, seed=seed + int(fold), n_rows=len(fit_rows))
        model.fit(x_train, y_train, sample_weight=w_train)
        x_valid, _ = _feature_matrix(valid_rows, features, medians=medians)
        pred = _predict(model, x_valid)
        y_valid = pd.to_numeric(valid_rows["rank_scope_target"], errors="coerce").to_numpy(dtype=float)
        raw_valid = pd.to_numeric(valid_rows["rank_scope_target_raw"], errors="coerce").to_numpy(dtype=float)
        pred_s = pd.Series(pred)
        y_s = pd.Series(y_valid)
        spearman = float(pred_s.corr(y_s, method="spearman")) if len(pred_s) > 2 else np.nan
        rows.append(
            {
                "fold": int(fold),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "spearman": spearman,
                "directional_accuracy": float((np.sign(pred) == np.sign(y_valid)).mean()),
                "mse": float(np.mean((pred - y_valid) ** 2)),
                "positive_target_share": float((raw_valid > 0.0).mean()),
                "positive_prediction_share": float((pred > 0.0).mean()),
            }
        )
        pred_frame = valid_rows[["timestamp", "fold", "rank_scope_target_raw", "rank_scope_target"]].copy()
        pred_frame["rank_scope_prediction"] = pred
        preds.append(pred_frame)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def fit_final_model(
    train: pd.DataFrame,
    features: list[str],
    *,
    backend: str,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    x_train, medians = _feature_matrix(train, features)
    y_train = pd.to_numeric(train["rank_scope_target"], errors="coerce").to_numpy(dtype=float)
    w_train = pd.to_numeric(train["rank_scope_sample_weight"], errors="coerce").to_numpy(dtype=float)
    model = _make_model(backend, seed=seed, n_rows=len(train))
    model.fit(x_train, y_train, sample_weight=w_train)
    return model, medians


def build_router_schedule(
    eval_state_panel: pd.DataFrame,
    model: Any,
    features: list[str],
    medians: dict[str, float],
    *,
    prediction_temperature: float,
) -> pd.DataFrame:
    state = dedupe_state_panel(eval_state_panel)
    x_eval, _ = _feature_matrix(state, features, medians=medians)
    pred = _predict(model, x_eval)
    temp = float(prediction_temperature)
    if not np.isfinite(temp) or temp <= 1e-12:
        temp = 0.25
    z = np.clip(pred / temp, -30.0, 30.0)
    weight = 1.0 / (1.0 + np.exp(-z))
    scope = np.where(weight >= 0.5, "timestamp_rank", "global_rank")
    return pd.DataFrame(
        {
            "timestamp": state["timestamp"].to_numpy(),
            "short_boll_rank_scope_preference": pred,
            "short_boll_timestamp_weight_raw": weight,
            "short_boll_timestamp_weight": np.clip(weight, 0.0, 1.0),
            "short_boll_rank_scope": scope,
            "router_valid": np.isfinite(pred),
            "router_fallback_reference": "",
            "router_mode": "direct_market_state_rank_scope_assessor",
            "router_layer": "rank_reference_before_threshold",
            "target_head": "short_boll",
            "reference_head": "global_rank_contract",
            "changes_thresholds": False,
            "changes_scores": False,
            "changes_active_stack": False,
            "promotion_status": "shadow_only",
        }
    ).sort_values("timestamp").reset_index(drop=True)


def _summary(train: pd.DataFrame, schedule: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    target = pd.to_numeric(train["rank_scope_target_raw"], errors="coerce")
    pred = pd.to_numeric(schedule["short_boll_rank_scope_preference"], errors="coerce")
    weight = pd.to_numeric(schedule["short_boll_timestamp_weight"], errors="coerce")
    return {
        "train_rows": int(len(train)),
        "train_timestamp_min": train["timestamp"].min(),
        "train_timestamp_max": train["timestamp"].max(),
        "selected_feature_count": int(len(features)),
        "target_mean": float(target.mean()),
        "target_p25": float(target.quantile(0.25)),
        "target_p50": float(target.quantile(0.50)),
        "target_p75": float(target.quantile(0.75)),
        "timestamp_target_positive_share": float((target > 0.0).mean()),
        "eval_rows": int(len(schedule)),
        "eval_timestamp_min": schedule["timestamp"].min(),
        "eval_timestamp_max": schedule["timestamp"].max(),
        "prediction_mean": float(pred.mean()),
        "prediction_p25": float(pred.quantile(0.25)),
        "prediction_p50": float(pred.quantile(0.50)),
        "prediction_p75": float(pred.quantile(0.75)),
        "timestamp_weight_mean": float(weight.mean()),
        "timestamp_weight_p25": float(weight.quantile(0.25)),
        "timestamp_weight_p50": float(weight.quantile(0.50)),
        "timestamp_weight_p75": float(weight.quantile(0.75)),
        "timestamp_rank_share": float(schedule["short_boll_rank_scope"].eq("timestamp_rank").mean()),
        "global_rank_share": float(schedule["short_boll_rank_scope"].eq("global_rank").mean()),
    }


def render_report(manifest: dict[str, Any], cv_metrics: pd.DataFrame) -> str:
    summary = dict(manifest["summary"])
    lines = [
        "# Market-State Rank-Scope Assessor",
        "",
        "This is a shadow-only LGBM/XGB market-state assessor for choosing the short_boll rank-reference layer before thresholding.",
        "",
        "## Contract",
        "",
        "- Active stack changed: `false`",
        "- q-fail active: `false`",
        "- HeadHealth active: `false`",
        "- Thresholds changed: `false`",
        "- Scores changed: `false`",
        "- Features: timestamp-level market-state columns only",
        "",
        "## Summary",
        "",
        f"- Backend: `{manifest['params']['backend']}`",
        f"- Training rows: `{summary['train_rows']}`",
        f"- Training range: `{summary['train_timestamp_min']}` to `{summary['train_timestamp_max']}`",
        f"- Eval rows: `{summary['eval_rows']}`",
        f"- Eval range: `{summary['eval_timestamp_min']}` to `{summary['eval_timestamp_max']}`",
        f"- Selected features: `{summary['selected_feature_count']}`",
        f"- Training timestamp-preference positive share: `{summary['timestamp_target_positive_share']:.6f}`",
        f"- Eval timestamp-rank share: `{summary['timestamp_rank_share']:.6f}`",
        f"- Eval mean timestamp-rank weight: `{summary['timestamp_weight_mean']:.6f}`",
        "",
        "## Expanding-Fold Diagnostics",
        "",
        cv_metrics.to_markdown(index=False) if not cv_metrics.empty else "_Not enough fold history for expanding-fold diagnostics._",
        "",
        "## Interpretation",
        "",
        "A positive prediction means timestamp ranking is preferred for short_boll; a negative prediction means global-over-time ranking is preferred. This schedule still needs paired rank-scope replay and later-window gates before any production use.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_panel = _read_frame(args.state_panel)
    timestamp_utility = _read_frame(args.timestamp_utility)
    train, target_scale = build_training_table(
        state_panel,
        timestamp_utility,
        target_col=str(args.target_col),
        target_scale=args.target_scale,
    )
    features = select_market_state_features(train)
    if args.max_features and int(args.max_features) > 0:
        variability = (
            train[features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .std(numeric_only=True)
            .sort_values(ascending=False)
        )
        features = [str(col) for col in variability.head(int(args.max_features)).index]
    cv_metrics, cv_predictions = expanding_fold_diagnostics(
        train,
        features,
        backend=str(args.backend),
        seed=int(args.seed),
    )
    model, medians = fit_final_model(
        train,
        features,
        backend=str(args.backend),
        seed=int(args.seed),
    )
    eval_state = _read_frame(args.eval_state_panel)
    schedule = build_router_schedule(
        eval_state,
        model,
        features,
        medians,
        prediction_temperature=float(args.prediction_temperature),
    )
    summary = _summary(train, schedule, features)
    manifest = {
        "generated_by": "build_market_state_rank_scope_assessor",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "shadow_direct_market_state_rank_scope_assessor",
        "contract": {
            "changes_active_stack": False,
            "changes_scores": False,
            "changes_thresholds": False,
            "changes_rank_reference_in_production": False,
            "router_layer": "rank_reference_before_threshold",
            "qfail_active": False,
            "head_health_active": False,
            "production_eligible": False,
            "promotion_status": "shadow_only",
            "feature_scope": "timestamp_level_market_state_only",
            "target_uses_outcomes": True,
            "target_uses_outcomes_only_in_training_period": True,
        },
        "params": {
            "backend": str(args.backend),
            "target_col": str(args.target_col),
            "target_scale": float(target_scale),
            "prediction_temperature": float(args.prediction_temperature),
            "max_features": int(args.max_features or 0),
            "seed": int(args.seed),
        },
        "inputs": {
            "state_panel": str(args.state_panel),
            "state_panel_sha256": _sha256(args.state_panel),
            "timestamp_utility": str(args.timestamp_utility),
            "timestamp_utility_sha256": _sha256(args.timestamp_utility),
            "eval_state_panel": str(args.eval_state_panel),
            "eval_state_panel_sha256": _sha256(args.eval_state_panel),
        },
        "selected_features": features,
        "feature_medians": medians,
        "summary": summary,
        "outputs": {
            "manifest": str(args.output_dir / "rank_scope_assessor_manifest.json"),
            "report": str(args.output_dir / "rank_scope_assessor_report.md"),
            "schedule": str(args.output_dir / "rank_reference_router_schedule.parquet"),
            "schedule_csv": str(args.output_dir / "rank_reference_router_schedule.csv"),
            "cv_metrics": str(args.output_dir / "rank_scope_assessor_cv_metrics.csv"),
            "cv_predictions": str(args.output_dir / "rank_scope_assessor_cv_predictions.csv"),
            "training_table": str(args.output_dir / "rank_scope_assessor_training_table.parquet"),
        },
    }
    schedule.to_parquet(args.output_dir / "rank_reference_router_schedule.parquet", index=False)
    schedule.to_csv(args.output_dir / "rank_reference_router_schedule.csv", index=False)
    cv_metrics.to_csv(args.output_dir / "rank_scope_assessor_cv_metrics.csv", index=False)
    cv_predictions.to_csv(args.output_dir / "rank_scope_assessor_cv_predictions.csv", index=False)
    train.to_parquet(args.output_dir / "rank_scope_assessor_training_table.parquet", index=False)
    (args.output_dir / "rank_scope_assessor_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "rank_scope_assessor_report.md").write_text(
        render_report(manifest, cv_metrics),
        encoding="utf-8",
    )
    return {key: Path(value) for key, value in manifest["outputs"].items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-panel", type=Path, default=DEFAULT_STATE_PANEL)
    parser.add_argument("--timestamp-utility", type=Path, default=DEFAULT_TIMESTAMP_UTILITY)
    parser.add_argument("--eval-state-panel", type=Path, default=DEFAULT_EVAL_STATE_PANEL)
    parser.add_argument(
        "--target-col",
        default="timestamp_minus_global_net_pnl",
        help="Positive target values favor timestamp-rank short_boll routing.",
    )
    parser.add_argument("--target-scale", type=float)
    parser.add_argument("--prediction-temperature", type=float, default=0.25)
    parser.add_argument("--backend", choices=["lgbm", "xgb", "rf"], default="xgb")
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(
        json.dumps(
            {
                "manifest": str(paths["manifest"]),
                "schedule": str(paths["schedule"]),
                "report": str(paths["report"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
