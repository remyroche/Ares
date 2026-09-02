#!/usr/bin/env python3
"""Train shallow, date-split execution-deterioration research controls.

This script deliberately starts with a shrinkage baseline and shallow models.
It produces prediction evidence for a future bounded pre-emption overlay but
does not import, retune, or modify the canonical close-price exit policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


Task = Literal["regression", "quantile", "classification"]
Arm = Literal["empirical", "ohlcv_only", "l2_aware"]
TargetKind = Literal["book_cost_delta", "max_book_cost_delta", "spread_delta", "max_spread_delta"]


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _read_surfaces(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("surface.parquet"))
    if not paths:
        raise FileNotFoundError(f"no execution surface files under {root}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True, copy=False)


def _parse_cost_grid(surface: pd.DataFrame, *, side: str) -> list[tuple[float, str]]:
    prefix = "sell_book_cost_bps_n" if side == "sell" else "buy_book_cost_bps_n"
    columns: list[tuple[float, str]] = []
    for column in surface.columns:
        if not column.startswith(prefix):
            continue
        token = column.removeprefix(prefix).replace("m", "-").replace("p", ".")
        try:
            columns.append((float(token), column))
        except ValueError:
            continue
    if not columns:
        raise ValueError(f"surface has no {side} execution-cost columns")
    return sorted(columns)


def reshape_surface_target(
    surface: pd.DataFrame,
    *,
    horizon_minutes: int,
    sides: Sequence[str],
    target_kind: TargetKind = "book_cost_delta",
) -> pd.DataFrame:
    """Create one supervised row per causal state and declared risk target."""
    required = {"symbol", "state_minute", "book_valid"}
    missing = required.difference(surface.columns)
    if missing:
        raise ValueError(f"surface lacks columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    if target_kind in {"spread_delta", "max_spread_delta"}:
        target_column = (
            f"spread_widening_{horizon_minutes}m"
            if target_kind == "spread_delta"
            # Over a one-minute path, terminal and maximum deterioration are
            # mathematically identical.  The compact surface intentionally
            # persists path maxima only for horizons with multiple intervening
            # minutes, so use the already-declared terminal label here rather
            # than inventing or backfilling a duplicate future field.
            else (
                "spread_widening_1m"
                if horizon_minutes == 1
                else f"max_spread_widening_next_{horizon_minutes}m"
            )
        )
        if target_column not in surface.columns:
            raise ValueError(f"missing future label {target_column}; rebuild surface without --features-only")
        work = surface.copy()
        work["liquidation_side"] = "neutral"
        work["position_notional"] = np.nan
        work["cost_now_bps"] = pd.to_numeric(work["spread_bps"], errors="coerce")
        work["deterioration_target_bps"] = pd.to_numeric(work[target_column], errors="coerce")
        work["target_horizon_minutes"] = int(horizon_minutes)
        frames.append(work)
    else:
        if target_kind == "max_book_cost_delta" and horizon_minutes not in {1, 3, 5, 10, 15, 30}:
            raise ValueError("max_book_cost_delta requires a declared complete-path horizon")
    for liquidation in sides:
        if target_kind in {"spread_delta", "max_spread_delta"}:
            break
        prefix = "sell" if liquidation == "sell" else "buy"
        for notional, cost_column in _parse_cost_grid(surface, side=prefix):
            token = cost_column.removeprefix(f"{prefix}_book_cost_bps_")
            target_column = (
                f"deterioration_{prefix}_{horizon_minutes}m_{token}"
                if target_kind == "book_cost_delta"
                else (
                    f"deterioration_{prefix}_1m_{token}"
                    if horizon_minutes == 1
                    else f"max_deterioration_{prefix}_{horizon_minutes}m_{token}"
                )
            )
            if target_column not in surface.columns:
                raise ValueError(f"missing future label {target_column}; rebuild surface without --features-only")
            work = surface.copy()
            work["liquidation_side"] = liquidation
            work["position_notional"] = float(notional)
            work["cost_now_bps"] = pd.to_numeric(work[cost_column], errors="coerce")
            work["deterioration_target_bps"] = pd.to_numeric(work[target_column], errors="coerce")
            work["target_horizon_minutes"] = int(horizon_minutes)
            frames.append(work)
    result = pd.concat(frames, ignore_index=True, copy=False)
    result["state_minute"] = _utc(result["state_minute"])
    result["date"] = result["state_minute"].dt.floor("D")
    result["month"] = result["state_minute"].dt.to_period("M").astype(str)
    valid = result["book_valid"].fillna(False).astype(bool)
    return result.loc[valid & result["deterioration_target_bps"].notna()].reset_index(drop=True)


def add_ohlcv_features(panel: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Causally as-of join optional canonical OHLCV values onto the surface."""
    required = {"symbol", "timestamp", "close"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV input lacks columns: {sorted(missing)}")
    right = ohlcv.copy()
    right["timestamp"] = _utc(right["timestamp"])
    right = right.sort_values(["symbol", "timestamp"], kind="stable")
    left = panel.sort_values(["symbol", "state_minute"], kind="stable").copy()
    left = pd.merge_asof(
        left, right, left_on="state_minute", right_on="timestamp", by="symbol",
        direction="backward", allow_exact_matches=True,
    )
    grouped = left.groupby("symbol", sort=False)
    close = pd.to_numeric(left["close"], errors="coerce")
    for lookback in (1, 3, 5):
        left[f"return_{lookback}m"] = grouped["close"].pct_change(lookback)
    if {"high", "low"}.issubset(left.columns):
        high = pd.to_numeric(left["high"], errors="coerce")
        low = pd.to_numeric(left["low"], errors="coerce")
        left["range_fraction"] = (high - low) / close.replace(0.0, np.nan)
        left["close_location"] = (close - low) / (high - low).replace(0.0, np.nan)
    if "volume" in left.columns:
        volume = pd.to_numeric(left["volume"], errors="coerce")
        median = grouped["volume"].shift(1).groupby(left["symbol"], sort=False).transform(
            lambda values: values.rolling(30, min_periods=10).median()
        )
        left["volume_shock"] = volume / pd.to_numeric(median, errors="coerce").replace(0.0, np.nan)
    return left


def feature_columns(frame: pd.DataFrame, arm: Arm) -> list[str]:
    ohlcv = [
        "return_1m", "return_3m", "return_5m", "range_fraction", "close_location", "volume_shock",
        "position_notional",
    ]
    l2 = [
        "cost_now_bps", "spread_bps", "bid_depth_10bps", "bid_depth_25bps", "bid_depth_50bps",
        "bid_depth_100bps", "bid_depth_200bps", "ask_depth_10bps", "ask_depth_25bps", "ask_depth_50bps",
        "ask_depth_100bps", "ask_depth_200bps", "book_imbalance_25bps", "book_imbalance_50bps",
        "book_imbalance_100bps", "spread_bps_change_1m", "spread_bps_change_3m", "spread_bps_change_5m",
        "bid_depth_50bps_change_1m", "bid_depth_50bps_change_3m", "bid_depth_50bps_change_5m",
        "ask_depth_50bps_change_1m", "ask_depth_50bps_change_3m", "ask_depth_50bps_change_5m",
        "spread_vs_recent_median", "depth_vs_recent_median", "spread_acceleration", "depth_collapse_rate",
    ]
    selected = [] if arm == "empirical" else list(ohlcv)
    if arm == "l2_aware":
        selected += l2
    return [column for column in selected if column in frame.columns]


@dataclass(frozen=True)
class EmpiricalShrinkage:
    global_mean: float
    group_means: pd.DataFrame
    strength: float
    bucket_edges: dict[str, tuple[float, ...]]

    _BUCKET_COLUMNS = ("bid_depth_50bps", "return_3m", "spread_bps", "position_notional")

    @classmethod
    def _bucket_frame(
        cls,
        frame: pd.DataFrame,
        *,
        edges_by_column: dict[str, tuple[float, ...]],
    ) -> pd.DataFrame:
        """Apply training-derived bins, never bins derived from held rows."""
        work = frame.copy()
        for column in cls._BUCKET_COLUMNS:
            edges = np.asarray(edges_by_column.get(column, ()), dtype=float)
            values = (
                pd.to_numeric(work[column], errors="coerce")
                if column in work.columns
                else pd.Series(np.nan, index=work.index, dtype=float)
            )
            if len(edges) >= 2:
                work[f"{column}_bucket"] = pd.cut(
                    values,
                    bins=edges,
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str)
            else:
                work[f"{column}_bucket"] = "all"
        return work

    @classmethod
    def fit(cls, frame: pd.DataFrame, *, target: str, strength: float = 100.0) -> "EmpiricalShrinkage":
        edges_by_column: dict[str, tuple[float, ...]] = {}
        # Fit bins only from the training period.  They represent a small
        # interpretable baseline rather than an unconstrained learner.
        for column in cls._BUCKET_COLUMNS:
            values = (
                pd.to_numeric(frame[column], errors="coerce")
                if column in frame.columns
                else pd.Series(np.nan, index=frame.index, dtype=float)
            )
            if values.notna().sum() >= 20:
                edges = np.unique(values.dropna().quantile([0, .25, .5, .75, 1]).to_numpy(float))
                if len(edges) >= 2:
                    # Extend the outer bins so valid held observations outside
                    # the train extrema map to edge bins rather than being
                    # silently converted into an unknown category.
                    edges = edges.astype(float, copy=True)
                    edges[0], edges[-1] = -np.inf, np.inf
                    edges_by_column[column] = tuple(float(value) for value in edges)
        work = cls._bucket_frame(frame, edges_by_column=edges_by_column)
        keys = ["liquidation_side", "bid_depth_50bps_bucket", "return_3m_bucket", "spread_bps_bucket", "position_notional_bucket"]
        grouped = work.groupby(keys, dropna=False)[target].agg(["mean", "count"]).reset_index()
        global_mean = float(pd.to_numeric(work[target], errors="coerce").mean())
        grouped["prediction"] = (grouped["mean"] * grouped["count"] + global_mean * strength) / (grouped["count"] + strength)
        return cls(
            global_mean=global_mean,
            group_means=grouped[keys + ["prediction"]],
            strength=float(strength),
            bucket_edges=edges_by_column,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        # New rows are binned using edges fitted on training data.  A missing
        # group receives the explicit global prior, never a held-period bin.
        keys = [column for column in self.group_means.columns if column.endswith("_bucket") or column == "liquidation_side"]
        work = self._bucket_frame(frame, edges_by_column=self.bucket_edges)
        merged = work.merge(self.group_means, on=keys, how="left", sort=False)
        return pd.to_numeric(merged["prediction"], errors="coerce").fillna(self.global_mean).to_numpy(float)


def chronological_split(frame: pd.DataFrame, *, validation_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(frame["month"].dropna().unique())
    if len(months) <= validation_months:
        raise ValueError("not enough independent months for chronological validation")
    held = set(months[-int(validation_months):])
    return frame.loc[~frame["month"].isin(held)].copy(), frame.loc[frame["month"].isin(held)].copy()


def _metric_summary(y: np.ndarray, prediction: np.ndarray, *, task: Task, threshold: float, quantile: float) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[mask], prediction[mask]
    if not len(y):
        return {"rows": 0.0}
    result: dict[str, float] = {"rows": float(len(y))}
    if task == "classification":
        target = y > threshold
        result["positive_rate"] = float(target.mean())
        try:
            from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
            result["pr_auc"] = float(average_precision_score(target, prediction)) if target.any() else float("nan")
            result["roc_auc"] = float(roc_auc_score(target, prediction)) if target.any() and (~target).any() else float("nan")
            result["brier"] = float(brier_score_loss(target, prediction))
        except ImportError:  # pragma: no cover - optional reporting dependency
            pass
    else:
        error = y - prediction
        result["mae_bps"] = float(np.abs(error).mean())
        result["rmse_bps"] = float(np.sqrt(np.mean(error ** 2)))
        if task == "quantile":
            result["pinball_loss"] = float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())
            result["quantile_coverage"] = float((y <= prediction).mean())
    return result


def _fit_shallow_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    columns: list[str],
    target: str,
    task: Task,
    threshold: float,
    quantile: float,
    max_depth: int,
) -> tuple[np.ndarray, Any]:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install lightgbm explicitly to run this model arm; empirical baseline remains available") from exc
    x_train = train.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_test = test.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    params: dict[str, Any] = {
        "n_estimators": 200, "learning_rate": 0.04, "max_depth": int(max_depth),
        "num_leaves": min(2 ** int(max_depth), 15), "min_child_samples": 200,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0,
        "random_state": 1729, "n_jobs": -1,
    }
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    if task == "classification":
        model = LGBMClassifier(objective="binary", **params)
        model.fit(x_train, y > threshold)
        return model.predict_proba(x_test)[:, 1], model
    objective = "quantile" if task == "quantile" else "huber"
    model = LGBMRegressor(objective=objective, alpha=float(quantile) if task == "quantile" else 0.9, **params)
    model.fit(x_train, y)
    return model.predict(x_test), model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ohlcv", type=Path, help="Optional canonical OHLCV parquet")
    parser.add_argument("--arm", choices=("empirical", "ohlcv_only", "l2_aware"), default="l2_aware")
    parser.add_argument("--task", choices=("regression", "quantile", "classification"), default="quantile")
    parser.add_argument("--quantile", type=float, default=.75)
    parser.add_argument("--threshold-bps", type=float, default=100.0)
    parser.add_argument(
        "--target-kind",
        choices=("book_cost_delta", "max_book_cost_delta", "spread_delta", "max_spread_delta"),
        default="book_cost_delta",
        help="Future-only target; cost targets are size/side-specific while spread targets are common-book diagnostics.",
    )
    parser.add_argument("--horizon-minutes", type=int, choices=(1, 2, 3, 5, 10, 15, 30), default=3)
    parser.add_argument("--liquidation-side", choices=("sell", "buy", "both"), default="sell")
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--max-depth", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--oracle", type=Path, help="Optional oracle parquet for A0-A3 descriptive summary")
    args = parser.parse_args()

    surface = _read_surfaces(args.surface_root)
    sides = ("sell", "buy") if args.liquidation_side == "both" else (args.liquidation_side,)
    target_kind: TargetKind = args.target_kind
    panel = reshape_surface_target(
        surface,
        horizon_minutes=int(args.horizon_minutes),
        sides=sides,
        target_kind=target_kind,
    )
    if args.ohlcv:
        panel = add_ohlcv_features(panel, pd.read_parquet(args.ohlcv))
    train, test = chronological_split(panel, validation_months=int(args.validation_months))
    target = "deterioration_target_bps"
    task: Task = args.task
    arm: Arm = args.arm
    if arm == "empirical":
        baseline = EmpiricalShrinkage.fit(train, target=target)
        prediction = baseline.predict(test)
        columns: list[str] = []
        fitted_model: Any = baseline
    else:
        columns = feature_columns(panel, arm)
        if not columns:
            raise ValueError(f"{arm} needs its matching feature fields; provide --ohlcv for OHLCV-only")
        finite_train = train.loc[:, columns].apply(pd.to_numeric, errors="coerce").notna().mean()
        columns = finite_train.loc[finite_train.ge(.80)].index.tolist()
        if not columns:
            raise ValueError("no model features have >=80% training coverage")
        prediction, fitted_model = _fit_shallow_lgbm(
            train, test, columns=columns, target=target, task=task,
            threshold=float(args.threshold_bps), quantile=float(args.quantile), max_depth=int(args.max_depth),
        )
    test = test.copy()
    test["execution_risk_prediction"] = prediction
    metric = _metric_summary(
        pd.to_numeric(test[target], errors="coerce").to_numpy(float), np.asarray(prediction, dtype=float),
        task=task, threshold=float(args.threshold_bps), quantile=float(args.quantile),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.out_dir / "execution_risk_training.parquet"
    test.to_parquet(training_path, index=False)
    model_path = args.out_dir / "execution_risk_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(fitted_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report: dict[str, Any] = {
        "schema": "ares.execution_risk_training.v1", "arm": arm, "task": task,
        "horizon_minutes": int(args.horizon_minutes), "liquidation_sides": list(sides),
        "target_kind": target_kind,
        "feature_columns": columns, "training_months": sorted(train["month"].unique().tolist()),
        "validation_months": sorted(test["month"].unique().tolist()), "metrics": metric,
        "model_path": str(model_path),
        "validation": "chronological held months; no row-level random split; symbol-days remain in one date/month partition",
        "inference_contract": "future deterioration columns are excluded; only causal surface/OHLCV inputs listed in feature_columns are eligible",
        "canonical_policy_modified": False,
    }
    if args.oracle:
        oracle = pd.read_parquet(args.oracle)
        report["oracle_preemption_summary"] = {
            str(minutes): {
                "rows": int(len(group)),
                "mean_gain_bps": float(pd.to_numeric(group["preemption_gain_bps"], errors="coerce").mean()),
                "positive_fraction": float(pd.to_numeric(group["preemption_gain_bps"], errors="coerce").gt(0.0).mean()),
            }
            for minutes, group in oracle.groupby("preempt_minutes", sort=True)
        }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
