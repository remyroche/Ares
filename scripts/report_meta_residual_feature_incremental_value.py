#!/usr/bin/env python3
"""Measure held-out residual information from lifecycle feature families."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_residual_ae_representation_ablation import (
    _candidate_features,  # noqa: E402
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

FOLDS = (
    (("2026-04",), "2026-05"),
    (("2026-04", "2026-05"), "2026-06"),
)
BASE_NUMERIC = (
    "score_current_reference",
    "historical_rank_current_reference",
)


def _families(features: list[str]) -> dict[str, list[str]]:
    lower = {name: name.lower() for name in features}
    groups = {
        "oi_lifecycle": [
            name
            for name in features
            if "oi_" in lower[name] and "funding" not in lower[name]
        ],
        "funding_lifecycle": [name for name in features if "funding" in lower[name]],
        "market_wide": [
            name
            for name in features
            if lower[name].startswith(("mkt_", "market_"))
            or lower[name].startswith(("eth_", "btc_"))
        ],
        "asset_relative": [
            name
            for name in features
            if any(
                token in lower[name]
                for token in (
                    "asset_minus",
                    "divergence",
                    "bench_resid",
                    "rv_rel_universe",
                    "cs_rank",
                )
            )
        ],
        "ohlcv_lifecycle": [
            name
            for name in features
            if any(
                token in lower[name]
                for token in (
                    "price_",
                    "ret",
                    "volume",
                    "wick",
                    "breadth",
                    "rv_",
                    "atr",
                    "shock",
                    "breakout",
                    "pullback",
                    "body_ratio",
                    "bb_pos",
                )
            )
        ],
    }
    return {name: sorted(set(values)) for name, values in groups.items() if values}


def _group_dummies(frame: pd.DataFrame, categories: list[str]) -> np.ndarray:
    key = (
        frame["side_name"].astype(str)
        + "||"
        + frame["archetype_policy_key"].astype(str)
    )
    mapping = {name: idx for idx, name in enumerate(categories)}
    positions = key.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
    output = np.zeros((len(frame), len(categories)), dtype=np.float32)
    valid = positions >= 0
    output[np.flatnonzero(valid), positions[valid]] = 1.0
    return output


def _matrix(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_columns: list[str],
    categories: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = (
        train.reindex(columns=numeric_columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    valid_x = (
        valid.reindex(columns=numeric_columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    medians = np.nanmedian(train_x, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0)
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    valid_x = np.where(np.isfinite(valid_x), valid_x, medians)
    train_x = np.concatenate([train_x, _group_dummies(train, categories)], axis=1)
    valid_x = np.concatenate([valid_x, _group_dummies(valid, categories)], axis=1)
    return train_x.astype(np.float32, copy=False), valid_x.astype(
        np.float32, copy=False
    )


def _native_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    *,
    objective: str,
    seed: int,
) -> np.ndarray:
    params = {
        "objective": objective,
        "learning_rate": 0.035,
        "max_depth": 3,
        "num_leaves": 7,
        "min_data_in_leaf": 180,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "feature_fraction": 0.80,
        "lambda_l1": 0.10,
        "lambda_l2": 8.0,
        "seed": int(seed),
        "num_threads": 2,
        "verbosity": -1,
        "force_col_wise": True,
    }
    dataset = lgb.Dataset(train_x, label=train_y, free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=120)
    return np.asarray(booster.predict(valid_x), dtype=np.float32)


def _tail_thresholds(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    key = (
        train["side_name"].astype(str)
        + "||"
        + train["archetype_policy_key"].astype(str)
    )
    thresholds: dict[str, tuple[float, float]] = {}
    for name, positions in key.groupby(key, sort=True).groups.items():
        values = pd.to_numeric(train.loc[positions, "hit_surprise"], errors="coerce")
        thresholds[str(name)] = (
            float(values.quantile(0.10)),
            float(values.quantile(0.90)),
        )
    return thresholds


def _tail_labels(
    frame: pd.DataFrame, thresholds: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    key = (
        frame["side_name"].astype(str)
        + "||"
        + frame["archetype_policy_key"].astype(str)
    )
    surprise = pd.to_numeric(frame["hit_surprise"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    negative = np.zeros(len(frame), dtype=np.int8)
    positive = np.zeros(len(frame), dtype=np.int8)
    fallback_low = float(np.nanquantile(surprise, 0.10))
    fallback_high = float(np.nanquantile(surprise, 0.90))
    for name, positions in key.groupby(key, sort=False).groups.items():
        idx = np.asarray(list(positions), dtype=np.int64)
        low, high = thresholds.get(str(name), (fallback_low, fallback_high))
        negative[idx] = surprise[idx] <= low
        positive[idx] = surprise[idx] >= high
    return negative, positive


def _classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    base_rate = float(np.mean(y))
    cutoff = float(np.quantile(probability, 0.90))
    selected = probability >= cutoff
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "top_decile_lift": float(np.mean(y[selected]) / max(base_rate, 1e-8)),
        "base_rate": base_rate,
    }


def _evaluate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_columns: list[str],
    categories: list[str],
    seed: int,
    *,
    shuffled_targets: bool = False,
) -> dict[str, float]:
    train_x, valid_x = _matrix(train, valid, numeric_columns, categories)
    y_train = pd.to_numeric(train["hit_surprise"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    y_valid = pd.to_numeric(valid["hit_surprise"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    thresholds = _tail_thresholds(train)
    negative_train, positive_train = _tail_labels(train, thresholds)
    negative_valid, positive_valid = _tail_labels(valid, thresholds)
    if shuffled_targets:
        rng = np.random.default_rng(seed)
        timestamp = pd.to_datetime(train["__ts__"], utc=True, errors="coerce")
        week = timestamp.dt.floor("D") - pd.to_timedelta(
            timestamp.dt.weekday.to_numpy(), unit="D"
        )
        keys = pd.DataFrame(
            {
                "week": week,
                "side": train["side_name"].astype(str),
                "archetype": train["archetype_policy_key"].astype(str),
            }
        )
        order = np.arange(len(train), dtype=np.int64)
        for positions in keys.groupby(
            ["week", "side", "archetype"], sort=False
        ).groups.values():
            idx = np.asarray(list(positions), dtype=np.int64)
            order[idx] = rng.permutation(idx)
        y_train = y_train[order]
        negative_train = negative_train[order]
        positive_train = positive_train[order]
    regression = _native_predict(
        train_x,
        y_train,
        valid_x,
        objective="huber",
        seed=seed,
    )
    negative_probability = _native_predict(
        train_x,
        negative_train,
        valid_x,
        objective="binary",
        seed=seed + 1,
    )
    positive_probability = _native_predict(
        train_x,
        positive_train,
        valid_x,
        objective="binary",
        seed=seed + 2,
    )
    error = regression - y_valid
    denominator = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
    rho = spearmanr(y_valid, regression, nan_policy="omit").statistic
    result = {
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error * error) / denominator)
        if denominator > 0.0
        else np.nan,
        "spearman_ic": float(rho) if np.isfinite(rho) else np.nan,
    }
    result.update(
        {
            f"negative_{name}": value
            for name, value in _classification_metrics(
                negative_valid, negative_probability
            ).items()
        }
    )
    result.update(
        {
            f"positive_{name}": value
            for name, value in _classification_metrics(
                positive_valid, positive_probability
            ).items()
        }
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    prediction = pd.read_parquet(
        root / "historical_rank_oos" / "oos_predictions_historical_rank.parquet"
    )
    prediction = prediction[
        pd.to_numeric(
            prediction["historical_rank_current_reference"], errors="coerce"
        ).ge(0.80)
    ].copy()
    prediction["hit_surprise"] = (
        pd.to_numeric(prediction["clean_exec"], errors="coerce")
        - pd.to_numeric(prediction["hit_prob_current_reference"], errors="coerce")
    ).astype(np.float32)
    compact_path = root / "cache" / "compact_reference_with_lifecycle.parquet"
    schema_columns = pq.read_schema(compact_path).names
    candidate_features = _candidate_features(pd.DataFrame(columns=schema_columns), root)
    keys = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    compact = pd.read_parquet(
        compact_path,
        columns=keys + candidate_features,
        filters=[
            ("__ts__", ">=", pd.Timestamp("2026-04-01", tz="UTC")),
            ("__ts__", "<", pd.Timestamp("2026-07-01", tz="UTC")),
        ],
    ).drop_duplicates(keys, keep="last")
    frame = prediction.merge(
        compact, on=keys, how="left", validate="one_to_one", suffixes=("", "__ctx")
    )
    families = _families(candidate_features)
    categories = sorted(
        (
            frame["side_name"].astype(str)
            + "||"
            + frame["archetype_policy_key"].astype(str)
        )
        .unique()
        .tolist()
    )
    feature_sets: dict[str, list[str]] = {"baseline": list(BASE_NUMERIC)}
    for name, columns in families.items():
        feature_sets[f"plus_{name}"] = list(dict.fromkeys([*BASE_NUMERIC, *columns]))
    feature_sets["all_families"] = list(
        dict.fromkeys([*BASE_NUMERIC, *candidate_features])
    )
    for name, columns in families.items():
        feature_sets[f"all_minus_{name}"] = [
            feature
            for feature in feature_sets["all_families"]
            if feature not in set(columns)
        ]

    rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    for fold_idx, (train_months, valid_month) in enumerate(FOLDS):
        train = frame[
            frame["calendar_month"].astype(str).isin(train_months)
        ].reset_index(drop=True)
        valid = frame[frame["calendar_month"].astype(str).eq(valid_month)].reset_index(
            drop=True
        )
        for set_name, columns in feature_sets.items():
            metrics = _evaluate(
                train, valid, columns, categories, 20260711 + fold_idx * 100
            )
            rows.append(
                {
                    "feature_set": set_name,
                    "train_months": ",".join(train_months),
                    "valid_month": valid_month,
                    "feature_count": len(columns),
                    **metrics,
                }
            )
        for draw in range(10):
            metrics = _evaluate(
                train,
                valid,
                feature_sets["all_families"],
                categories,
                20260711 + fold_idx * 1000 + draw * 7,
                shuffled_targets=True,
            )
            shuffle_rows.append({"valid_month": valid_month, "draw": draw, **metrics})
    results = pd.DataFrame(rows)
    shuffled = pd.DataFrame(shuffle_rows)
    results.to_csv(
        report_dir / "stage5_feature_family_incremental_value.csv", index=False
    )
    shuffled.to_csv(report_dir / "stage5_feature_label_shuffle.csv", index=False)
    summary = (
        results.groupby("feature_set", sort=True)
        .agg(
            folds=("valid_month", "size"),
            mean_mse=("mse", "mean"),
            mean_spearman_ic=("spearman_ic", "mean"),
            mean_negative_ap=("negative_average_precision", "mean"),
            mean_negative_top_decile_lift=("negative_top_decile_lift", "mean"),
            mean_positive_ap=("positive_average_precision", "mean"),
            mean_positive_top_decile_lift=("positive_top_decile_lift", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(report_dir / "stage5_feature_family_summary.csv", index=False)
    baseline = summary[summary["feature_set"].eq("baseline")].iloc[0]
    all_features = summary[summary["feature_set"].eq("all_families")].iloc[0]
    shuffle_negative_ap = float(shuffled["negative_average_precision"].mean())
    shuffle_positive_ap = float(shuffled["positive_average_precision"].mean())
    manifest = {
        "schema": "meta_residual_feature_incremental_value_v1",
        "population": "causal_historical_top20",
        "candidate_feature_count": len(candidate_features),
        "family_counts": {name: len(columns) for name, columns in families.items()},
        "baseline_mean_mse": float(baseline["mean_mse"]),
        "all_features_mean_mse": float(all_features["mean_mse"]),
        "baseline_negative_ap": float(baseline["mean_negative_ap"]),
        "all_features_negative_ap": float(all_features["mean_negative_ap"]),
        "baseline_positive_ap": float(baseline["mean_positive_ap"]),
        "all_features_positive_ap": float(all_features["mean_positive_ap"]),
        "shuffled_negative_ap": shuffle_negative_ap,
        "shuffled_positive_ap": shuffle_positive_ap,
        "incremental_value_pass": bool(
            float(all_features["mean_mse"]) < float(baseline["mean_mse"])
            and float(all_features["mean_negative_ap"])
            > float(baseline["mean_negative_ap"])
            and float(all_features["mean_positive_ap"])
            > float(baseline["mean_positive_ap"])
            and float(all_features["mean_negative_ap"]) > shuffle_negative_ap
            and float(all_features["mean_positive_ap"]) > shuffle_positive_ap
        ),
        "leakage_contract": "Each model trains only on earlier OOS-generated rows and evaluates the next calendar month.",
    }
    (report_dir / "stage5_feature_incremental_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
