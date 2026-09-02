#!/usr/bin/env python3
"""Chronologically evaluate causal L2 spread/slippage deterioration models.

The purpose is research evidence for an *entry friction* predictor, not a
replacement for the current alpha, admission, or exit stack.  Every fold is a
contiguous held set of UTC dates; all fits and preprocessing use only earlier
dates.  Grouped MDA cyclically permutes whole feature families within each
symbol-date trajectory, never individual rows across a temporal fold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.execution.train_execution_risk import EmpiricalShrinkage, reshape_surface_target  # noqa: E402


Task = Literal["regression", "classification", "ranking"]
ModelName = Literal["bayesian_shrinkage", "ridge", "lgbm"]


def _read_panels(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("surface.parquet"))
    if not paths:
        raise FileNotFoundError(f"no materialised transition panels below {root}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True, copy=False)


def _causal_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Fixed semantic groups; labels and provenance never enter a model."""
    excluded_fragments = (
        "future", "deterioration", "spread_widening", "max_spread_next", "max_cost_next",
        "execution_label", "target_horizon", "label_valid", "rank_target", "rank_grade",
        "rank_query", "asset_spread_baseline", "asset_spread_scale",
    )
    exclusions = {
        "state_minute", "decision_ts", "available_ts", "exchange_timestamp", "date", "month", "symbol",
        # These are provenance/validity receipts, not execution predictors.
        "raw_trade_data_retained", "source_rows", "book_has_snapshot", "book_crossed_or_empty",
        "book_valid", "source_available_by_decision",
    }
    numeric = [
        column for column in frame.columns
        if column not in exclusions
        and not any(fragment in column for fragment in excluded_fragments)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    families: dict[str, list[str]] = {
        "current_book": [], "book_transition": [], "book_flow_rates": [],
        "trade_flow": [], "asset_state": [], "btc_benchmark": [],
        "market_cross_asset": [],
        "position": [], "external_context": [],
    }
    for column in numeric:
        if column == "position_notional" or column.startswith("position_to_"):
            family = "position"
        elif any(token in column for token in ("trade_", "order_flow", "quote_volume", "volume_ratio")):
            # Completed-minute executed-trade aggregates are distinct from
            # L2 quote-message transitions, even when they also expose a
            # one/three/five-minute change column.
            family = "trade_flow"
        elif any(token in column for token in (
            "_cancel_rate", "_replenishment_rate", "_replenishment_failure",
            "book_flow_", "book_update_",
        )):
            # Retained L2 quote-event state. This is deliberately separate
            # from static book geometry and ordinary depth/spread transitions
            # so grouped MDA can assess whether compact event flow is
            # incremental.
            family = "book_flow_rates"
        elif any(token in column for token in ("cancel", "replenish", "book_flow", "book_update", "_change_", "acceleration", "collapse", "recent_median")):
            family = "book_transition"
        elif column.startswith(("btc_", "btc_context_")):
            family = "btc_benchmark"
        elif column.startswith(("market_", "asset_minus_market", "liquidity_rank")):
            family = "market_cross_asset"
        elif column.startswith(("ret_", "drawdown", "realized_vol", "volume_ratio")):
            family = "asset_state"
        elif any(token in column for token in ("fund", "open_interest", "oi_", "context_")):
            family = "external_context"
        elif any(token in column for token in ("spread", "depth", "imbalance", "microprice", "book_cost", "vwap", "insufficient")):
            family = "current_book"
        else:
            # Keep unknown numeric context only in a clearly auditable family.
            family = "external_context"
        families[family].append(column)
    return {name: columns for name, columns in families.items() if columns}


def _sample_dates(frame: pd.DataFrame, *, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or cap <= 0:
        return frame
    pieces = []
    for _, group in frame.groupby("date", sort=True):
        pieces.append(group if len(group) <= cap else group.sample(n=cap, random_state=seed))
    return pd.concat(pieces, ignore_index=True)


def chronological_date_folds(frame: pd.DataFrame, *, folds: int) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    dates = sorted(pd.to_datetime(frame["date"], utc=True, errors="coerce").dropna().unique())
    if len(dates) < 3:
        raise ValueError("need at least three independent UTC dates for chronological evaluation")
    chunks = [list(chunk) for chunk in np.array_split(np.asarray(dates), min(int(folds) + 1, len(dates))) if len(chunk)]
    result: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    for index in range(1, len(chunks)):
        train = [pd.Timestamp(value) for chunk in chunks[:index] for value in chunk]
        test = [pd.Timestamp(value) for value in chunks[index]]
        if train and test:
            result.append((train, test))
    if not result:
        raise ValueError("no non-empty chronological date fold")
    return result


def _prepare_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rolling_asset_spread_baseline(
    frame: pd.DataFrame,
    *,
    observations: int,
) -> tuple[pd.Series, pd.Series]:
    """Return strictly-prior per-symbol median/MAD spread baselines.

    The robust scale is part of the *offline target construction*, never an
    inference feature.  It ensures the ranking labels compare a BTC-like
    narrow book and a structurally wider low-volume book in their own causal
    liquidity units rather than allowing the latter to dominate every query.
    """
    n = int(observations)
    if n < 4:
        raise ValueError("rank asset baseline needs at least four prior observations")
    current = pd.to_numeric(frame["spread_bps"], errors="coerce")
    prior = current.groupby(frame["symbol"], sort=False).shift(1)
    minimum = max(4, n // 3)
    median = prior.groupby(frame["symbol"], sort=False).transform(
        lambda values, window=n, min_periods=minimum: values.rolling(window, min_periods=min_periods).median()
    )
    mad = prior.groupby(frame["symbol"], sort=False).transform(
        lambda values, window=n, min_periods=minimum: values.rolling(window, min_periods=min_periods).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True,
        )
    )
    # One basis point is a conservative absolute lower scale: a virtually
    # tick-constant book remains comparable without exploding a tiny MAD.
    scale = mad.mul(1.4826).where(mad.notna(), np.nan).clip(lower=1.0)
    return median, scale


def _strict_prior_asset_median(
    frame: pd.DataFrame,
    column: str,
    *,
    observations: int,
) -> pd.Series:
    """Return a strictly-prior per-symbol rolling median for a live field.

    This is causal *feature context*, not a label transform.  It lets the
    ranker recognize a spread or displayed depth relative to the normal
    liquidity profile of the same asset without mixing in future rows or
    treating a high-volume BTC-like book as interchangeable with a thin alt.
    """
    current = pd.to_numeric(frame[column], errors="coerce")
    prior = current.groupby(frame["symbol"], sort=False).shift(1)
    return prior.groupby(frame["symbol"], sort=False).transform(
        lambda values, window=int(observations), min_periods=max(4, int(observations) // 3): values.rolling(
            window, min_periods=min_periods
        ).median()
    )


def build_lambdarank_spread_target(
    surface: pd.DataFrame,
    *,
    horizon_minutes: int,
    target_kind: str,
    asset_baseline_observations: int,
) -> pd.DataFrame:
    """Make timestamp queries and leakage-safe grades for three rank arms.

    ``absolute_future_spread`` uses literal future spread in bps for its
    timestamp-local target/grade.  It is therefore a genuine absolute-spread
    control, rather than a renamed relative-spread target.  Causal
    per-asset-relative spread and displayed-depth fields are added as model
    features so this arm still transfers across liquidity profiles.
    ``spread_delta`` measures widening from the live spread, and
    ``asset_relative_deviation`` measures future spread versus the asset's
    strictly-prior baseline.  All three use five within-timestamp ordinal
    grades for native LambdaRank and keep their raw bps target for reporting.
    """
    future_column = f"spread_bps_future_{int(horizon_minutes)}m"
    if future_column not in surface.columns:
        raise ValueError(f"missing future spread label {future_column}; source cadence/horizon is incompatible")
    if target_kind not in {"absolute_future_spread", "spread_delta", "asset_relative_deviation"}:
        raise ValueError(f"unsupported rank target {target_kind}")
    panel = surface.copy()
    panel["state_minute"] = pd.to_datetime(panel["state_minute"], utc=True, errors="coerce")
    panel = panel.sort_values(["symbol", "state_minute"], kind="stable").reset_index(drop=True)
    current = pd.to_numeric(panel["spread_bps"], errors="coerce")
    future = pd.to_numeric(panel[future_column], errors="coerce")
    baseline, scale = _rolling_asset_spread_baseline(panel, observations=int(asset_baseline_observations))
    panel["asset_spread_baseline_bps"] = baseline
    panel["asset_spread_scale_bps"] = scale
    panel["spread_bps_to_asset_prior_median"] = current / baseline.replace(0.0, np.nan)
    # There is deliberately no trade/volume input in this compact contract.
    # Displayed L2 depth is its point-in-time liquidity proxy.  Each ratio is
    # normalized by a strictly-prior asset-local baseline, never an in-query
    # or future cross-section.
    for column in (
        "bid_depth_10bps", "ask_depth_10bps", "bid_depth_50bps", "ask_depth_50bps",
        "bid_depth_100bps", "ask_depth_100bps",
    ):
        if column not in panel.columns:
            continue
        depth_baseline = _strict_prior_asset_median(
            panel, column, observations=int(asset_baseline_observations),
        )
        panel[f"{column}_to_asset_prior_median"] = (
            pd.to_numeric(panel[column], errors="coerce") / depth_baseline.replace(0.0, np.nan)
        )
    if target_kind == "absolute_future_spread":
        raw = future
        normalized = future
    elif target_kind == "spread_delta":
        raw = future - current
        normalized = raw / scale
    else:
        raw = future - baseline
        normalized = raw / scale
    panel["rank_target_raw_bps"] = raw
    panel["rank_target_normalized"] = normalized.replace([np.inf, -np.inf], np.nan)
    panel["rank_query"] = panel["state_minute"].astype("int64").astype(str)
    valid = panel["book_valid"].fillna(False).astype(bool) & panel["rank_target_normalized"].notna()
    panel = panel.loc[valid].copy()
    query_size = panel.groupby("rank_query", sort=False)["symbol"].transform("size")
    panel = panel.loc[query_size.ge(5)].copy()
    # Deterministic average rank handles exchange tick ties without arbitrary
    # file-order influence.  Native LambdaRank receives ordered 0..4 labels.
    percentile = panel.groupby("rank_query", sort=False)["rank_target_normalized"].rank(method="average", pct=True)
    panel["rank_grade"] = np.minimum(4, np.floor(percentile.to_numpy(float) * 5.0).astype(int))
    panel["deterioration_target_bps"] = panel["rank_target_raw_bps"]
    panel["date"] = panel["state_minute"].dt.floor("D")
    panel["month"] = panel["state_minute"].dt.to_period("M").astype(str)
    return panel.reset_index(drop=True)


@dataclass
class FittedModel:
    name: ModelName
    model: Any
    columns: list[str]
    task: Task
    threshold: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.name == "bayesian_shrinkage":
            return self.model.predict(frame)
        matrix = _prepare_matrix(frame, self.columns)
        if self.task == "classification":
            return self.model.predict_proba(matrix)[:, 1]
        return self.model.predict(matrix)


def _fit_model(
    train: pd.DataFrame,
    *,
    name: ModelName,
    columns: list[str],
    target: str,
    task: Task,
    threshold: float,
    depth: int,
    seed: int,
) -> FittedModel:
    if task == "ranking" and name != "lgbm":
        raise ValueError("native LambdaRank evaluation supports only the lgbm model")
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    if name == "bayesian_shrinkage":
        if task == "classification":
            raise ValueError("bayesian_shrinkage is a regression baseline; use regression task")
        return FittedModel(name=name, model=EmpiricalShrinkage.fit(train, target=target, strength=150.0), columns=[], task=task, threshold=threshold)
    x = _prepare_matrix(train, columns)
    if name == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        model: Any = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            ("model", LogisticRegression(C=.15, max_iter=1_000, random_state=seed, class_weight="balanced") if task == "classification" else Ridge(alpha=30.0)),
        ])
        model.fit(x, y > threshold if task == "classification" else y)
    else:
        try:
            from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
        except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("lightgbm is required for the lgbm arm") from exc
        common = {
            "n_estimators": 400, "learning_rate": .035, "max_depth": int(depth),
            "num_leaves": min(2 ** int(depth), 15), "min_child_samples": 500,
            "subsample": .80, "colsample_bytree": .80, "reg_lambda": 15.0,
            "random_state": int(seed), "n_jobs": -1, "verbosity": -1,
        }
        if task == "ranking":
            if "rank_query" not in train.columns:
                raise ValueError("LambdaRank fitting requires rank_query")
            ranked = train.loc[train["rank_query"].notna() & np.isfinite(y)].copy()
            ranked = ranked.sort_values(["rank_query", "symbol", "state_minute"], kind="stable")
            groups = ranked.groupby("rank_query", sort=False).size().to_numpy(int)
            if len(groups) < 2 or groups.min() < 2:
                raise ValueError("LambdaRank fitting needs at least two nontrivial timestamp queries")
            # The 500-row floor is appropriate for the broad one-minute L2
            # panel.  A compact 15-minute chronology has materially fewer
            # rows; retain a conservative 5%-of-fold floor, bounded at 100,
            # instead of suppressing every split in early chronological folds.
            common["min_child_samples"] = max(100, min(500, len(ranked) // 20))
            model = LGBMRanker(
                objective="lambdarank", metric="ndcg",
                lambdarank_truncation_level=5, label_gain=(0, 1, 3, 7, 15), **common,
            )
            model.fit(_prepare_matrix(ranked, columns), pd.to_numeric(ranked[target], errors="coerce").astype(int), group=groups)
        else:
            model = LGBMClassifier(objective="binary", **common) if task == "classification" else LGBMRegressor(objective="huber", alpha=.9, **common)
            model.fit(x, y > threshold if task == "classification" else y)
    return FittedModel(name=name, model=model, columns=list(columns), task=task, threshold=threshold)


def _loss(y: np.ndarray, prediction: np.ndarray, *, task: Task, threshold: float) -> float:
    finite = np.isfinite(y) & np.isfinite(prediction)
    if not finite.any():
        return float("nan")
    y, prediction = y[finite], prediction[finite]
    if task == "regression":
        return float(np.abs(y - prediction).mean())
    if task == "ranking":
        raise ValueError("ranking loss requires query identities and is not used by grouped MDA")
    from sklearn.metrics import roc_auc_score
    target = y > threshold
    return float(-roc_auc_score(target, prediction)) if target.any() and (~target).any() else float("nan")


def _metrics(y: np.ndarray, prediction: np.ndarray, *, task: Task, threshold: float) -> dict[str, float]:
    finite = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[finite], prediction[finite]
    output: dict[str, float] = {"rows": float(len(y))}
    if not len(y):
        return output
    if task == "regression":
        error = y - prediction
        output.update({"mae_bps": float(np.abs(error).mean()), "rmse_bps": float(np.sqrt(np.mean(error ** 2))), "bias_bps": float(error.mean())})
    elif task == "classification":
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        target = y > threshold
        output.update({
            "positive_rate": float(target.mean()),
            "roc_auc": float(roc_auc_score(target, prediction)) if target.any() and (~target).any() else float("nan"),
            "pr_auc": float(average_precision_score(target, prediction)) if target.any() else float("nan"),
            "brier": float(brier_score_loss(target, prediction)),
        })
    return output


def _ranking_metrics(held: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    """Assess native-ranker ordering in both normalized and raw economics."""
    from sklearn.metrics import ndcg_score

    scored = held.loc[:, ["rank_query", "rank_grade", "rank_target_normalized", "rank_target_raw_bps"]].copy()
    scored["prediction"] = np.asarray(prediction, dtype=float)
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna()
    output: dict[str, float] = {"rows": float(len(scored)), "queries": 0.0}
    if scored.empty:
        return output
    ndcgs: dict[int, list[float]] = {1: [], 3: [], 5: []}
    normalized_rhos: list[float] = []
    raw_rhos: list[float] = []
    raw_top: dict[int, list[float]] = {1: [], 3: [], 5: []}
    raw_pool: list[float] = []
    for _, group in scored.groupby("rank_query", sort=False):
        if len(group) < 2:
            continue
        truth = group["rank_grade"].to_numpy(float)[None, :]
        scores = group["prediction"].to_numpy(float)[None, :]
        for k, values in ndcgs.items():
            values.append(float(ndcg_score(truth, scores, k=min(k, len(group)))))
        normalized_rho = (
            float(group["rank_target_normalized"].corr(group["prediction"], method="spearman"))
            if group["rank_target_normalized"].nunique(dropna=True) > 1 and group["prediction"].nunique(dropna=True) > 1
            else float("nan")
        )
        raw_rho = (
            float(group["rank_target_raw_bps"].corr(group["prediction"], method="spearman"))
            if group["rank_target_raw_bps"].nunique(dropna=True) > 1 and group["prediction"].nunique(dropna=True) > 1
            else float("nan")
        )
        if np.isfinite(normalized_rho):
            normalized_rhos.append(normalized_rho)
        if np.isfinite(raw_rho):
            raw_rhos.append(raw_rho)
        ordered = group.sort_values("prediction", ascending=False, kind="stable")
        raw_pool.append(float(group["rank_target_raw_bps"].mean()))
        for k, values in raw_top.items():
            values.append(float(ordered.head(k)["rank_target_raw_bps"].mean()))
    output["queries"] = float(len(normalized_rhos))
    for k, values in ndcgs.items():
        output[f"ndcg_at_{k}"] = float(np.nanmean(values)) if values else float("nan")
    output["spearman_normalized"] = float(np.nanmean(normalized_rhos)) if normalized_rhos else float("nan")
    output["spearman_raw_bps"] = float(np.nanmean(raw_rhos)) if raw_rhos else float("nan")
    output["mean_raw_bps_pool"] = float(np.nanmean(raw_pool)) if raw_pool else float("nan")
    for k, values in raw_top.items():
        output[f"mean_raw_bps_top_{k}"] = float(np.nanmean(values)) if values else float("nan")
        output[f"raw_bps_uplift_top_{k}"] = (
            float(np.nanmean(values) - np.nanmean(raw_pool)) if values and raw_pool else float("nan")
        )
    return output


def _block_permute(frame: pd.DataFrame, columns: Sequence[str], *, seed: int) -> pd.DataFrame:
    """Circularly shift a whole feature family within each symbol-date block.

    This preserves the within-family relationships and avoids an invalid
    row-random shuffle.  Its unit is a complete chronological intraday block;
    date-fold metrics are then aggregated across independent held dates.
    """
    output = frame.copy()
    rng = np.random.default_rng(seed)
    for _, index in output.groupby(["date", "symbol"], sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        if len(positions) < 3:
            continue
        shift = int(rng.integers(1, len(positions)))
        output.loc[positions, list(columns)] = output.loc[np.roll(positions, shift), list(columns)].to_numpy()
    return output


def _mda_rows(
    fitted: FittedModel,
    held: pd.DataFrame,
    *,
    target: str,
    groups: dict[str, list[str]],
    seed: int,
) -> list[dict[str, object]]:
    actual = pd.to_numeric(held[target], errors="coerce").to_numpy(float)
    baseline = _loss(actual, fitted.predict(held), task=fitted.task, threshold=fitted.threshold)
    rows: list[dict[str, object]] = []
    for offset, (family, columns) in enumerate(groups.items()):
        usable = [column for column in columns if column in fitted.columns]
        if not usable:
            continue
        shuffled = _block_permute(held, usable, seed=seed + offset + 1)
        permuted = _loss(actual, fitted.predict(shuffled), task=fitted.task, threshold=fitted.threshold)
        rows.append({
            "feature_group": family,
            "features": json.dumps(usable),
            "baseline_loss": baseline,
            "permuted_loss": permuted,
            "importance_loss_increase": permuted - baseline,
            "permutation_unit": "cyclic_whole_feature_family_within_symbol_date",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-kind", choices=("book_cost_delta", "max_book_cost_delta", "spread_delta", "max_spread_delta"), default="book_cost_delta")
    parser.add_argument("--horizon-minutes", type=int, choices=(1, 2, 3, 5, 10, 15, 30), default=5)
    parser.add_argument("--liquidation-side", choices=("sell", "buy"), default="sell")
    parser.add_argument("--task", choices=("regression", "classification", "ranking"), default="regression")
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--models", choices=("bayesian_shrinkage", "ridge", "lgbm"), nargs="+")
    parser.add_argument(
        "--rank-target",
        choices=("absolute_future_spread", "spread_delta", "asset_relative_deviation"),
        help="Required for --task ranking; uses timestamp-local LambdaRank queries. Delta/deviation targets are profile-normalized; absolute is literal bps.",
    )
    parser.add_argument(
        "--rank-asset-baseline-observations", type=int, default=12,
        help="Strictly-prior source observations for per-symbol target normalization and causal liquidity-context features.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--lgbm-depth", choices=(2, 3, 4), type=int, default=3)
    parser.add_argument("--max-rows-per-date", type=int, help="Deterministic bounded research sample per UTC date")
    parser.add_argument(
        "--exclude-feature-groups",
        nargs="*",
        default=(),
        help="Predeclared semantic feature groups to exclude for a matched ablation.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    surface = _read_panels(args.panel_root)
    task: Task = args.task
    if task == "ranking":
        if not args.rank_target:
            raise ValueError("--task ranking requires --rank-target")
        panel = build_lambdarank_spread_target(
            surface, horizon_minutes=int(args.horizon_minutes), target_kind=str(args.rank_target),
            asset_baseline_observations=int(args.rank_asset_baseline_observations),
        )
    else:
        panel = reshape_surface_target(surface, horizon_minutes=int(args.horizon_minutes), sides=(args.liquidation_side,), target_kind=args.target_kind)
    # Legacy shrinkage baseline calls the short return bucket ``return_3m``.
    # Preserve the causal surface's preferred ``ret_3m`` naming while exposing
    # this explicit compatibility alias only to the baseline/model panel.
    if "return_3m" not in panel.columns and "ret_3m" in panel.columns:
        panel["return_3m"] = pd.to_numeric(panel["ret_3m"], errors="coerce")
    panel = _sample_dates(panel, cap=args.max_rows_per_date, seed=int(args.seed))
    groups = _causal_feature_groups(panel)
    unknown_groups = sorted(set(args.exclude_feature_groups).difference(groups))
    if unknown_groups:
        raise ValueError(f"unknown feature groups: {unknown_groups}")
    groups = {name: fields for name, fields in groups.items() if name not in set(args.exclude_feature_groups)}
    columns = [column for fields in groups.values() for column in fields]
    coverage = panel.loc[:, columns].notna().mean() if columns else pd.Series(dtype=float)
    columns = coverage.loc[coverage.ge(.80)].index.tolist()
    if not columns:
        raise ValueError("no causal feature has >=80% coverage on the training panel")
    groups = {name: [column for column in fields if column in columns] for name, fields in groups.items()}
    groups = {name: fields for name, fields in groups.items() if fields}
    folds = chronological_date_folds(panel, folds=int(args.folds))
    models = tuple(args.models or (("lgbm",) if task == "ranking" else ("bayesian_shrinkage", "ridge", "lgbm")))
    if task == "ranking" and set(models) != {"lgbm"}:
        raise ValueError("ranking runs require --models lgbm (or omit --models)")
    metrics_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    mda: list[dict[str, object]] = []
    target = "rank_grade" if task == "ranking" else "deterioration_target_bps"
    for fold_number, (train_dates, test_dates) in enumerate(folds, start=1):
        train = panel.loc[panel["date"].isin(train_dates)].copy()
        held = panel.loc[panel["date"].isin(test_dates)].copy()
        for name in models:
            model = _fit_model(
                train, name=name, columns=columns, target=target, task=task,
                threshold=float(args.threshold_bps), depth=int(args.lgbm_depth), seed=int(args.seed) + fold_number,
            )
            prediction = model.predict(held)
            metric = (
                _ranking_metrics(held, prediction)
                if task == "ranking"
                else _metrics(pd.to_numeric(held[target], errors="coerce").to_numpy(float), prediction, task=task, threshold=float(args.threshold_bps))
            )
            metrics_rows.append({
                "fold": fold_number, "model": name, "train_dates": len(train_dates), "test_dates": len(test_dates),
                "train_last_date": str(max(train_dates).date()), "test_first_date": str(min(test_dates).date()),
                "test_last_date": str(max(test_dates).date()), **metric,
            })
            score_columns = ["symbol", "state_minute", "date", target]
            if task == "ranking":
                score_columns += ["rank_query", "rank_target_raw_bps", "rank_target_normalized"]
            scored = held.loc[:, score_columns].copy()
            scored["fold"] = fold_number
            scored["model"] = name
            scored["prediction"] = prediction
            predictions.append(scored)
            if name != "bayesian_shrinkage" and task != "ranking":
                for row in _mda_rows(model, held, target=target, groups=groups, seed=int(args.seed) + fold_number * 100):
                    mda.append({"fold": fold_number, "model": name, **row})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(metrics_rows).to_parquet(args.out_dir / "date_fold_metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(args.out_dir / "oof_predictions.parquet", index=False)
    mda_frame = pd.DataFrame.from_records(mda)
    mda_frame.to_parquet(args.out_dir / "grouped_mda_by_date_fold.parquet", index=False)
    stability = (
        mda_frame.groupby(["model", "feature_group"], as_index=False)["importance_loss_increase"]
        .agg(["mean", "median", "min", "max", "std", "count"]).reset_index()
        if not mda_frame.empty else pd.DataFrame()
    )
    stability.to_parquet(args.out_dir / "grouped_mda_stability.parquet", index=False)
    manifest = {
        "schema": "ares.liquidity_transition_evaluation.v1",
        "panel_root": str(args.panel_root), "target_kind": args.target_kind, "horizon_minutes": int(args.horizon_minutes),
        "task": task, "threshold_bps": float(args.threshold_bps), "models": list(models),
        "rank_target": args.rank_target,
        "rank_target_contract": (
            "timestamp-local queries; absolute future-spread arm ranks literal bps; delta/deviation labels use strictly-prior "
            "per-symbol robust normalization; causal per-symbol spread/depth-relative context is available to all arms; "
            "raw bps target retained for economics" if task == "ranking" else None
        ),
        "rank_asset_baseline_observations": int(args.rank_asset_baseline_observations) if task == "ranking" else None,
        "feature_columns": columns, "feature_groups": groups,
        "excluded_feature_groups": list(args.exclude_feature_groups),
        "validation": "expanding chronological UTC-date folds; no row-level random split",
        "mda": "one whole-family cyclic block permutation within each symbol-date trajectory, summarized across held date folds",
        "sample_rows": int(len(panel)), "unique_dates": int(panel["date"].nunique()), "folds": len(folds),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**manifest, "metrics": metrics_rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
