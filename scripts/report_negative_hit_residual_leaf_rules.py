#!/usr/bin/env python3
"""Explain causal negative-hit-residual states with forward-OOS tree leaves.

This is an explanatory companion to ``run_executable_quality_transition_ablation``.
For every forward month it rebuilds the causal negative-hit-residual target from
prior rows only, fits shallow side x archetype recognizers, and evaluates every
tree leaf only on that month's OOS rows.  It emits both exact threshold paths
and threshold-free feature combinations, ranked by recurring OOS failure lift.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/candidate_shards"
)
DEFAULT_FEATURE_STORE_ID = "20260711_070000"
KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
OUTCOMES = (
    "ev_after_1pct",
    "exec_margin",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
)


def _parse_months(value: str) -> list[pd.Period]:
    months = [pd.Period(item.strip(), freq="M") for item in value.split(",") if item.strip()]
    if not months:
        raise ValueError("at least one OOS month is required")
    return months


def _local_key(frame: pd.DataFrame) -> pd.Series:
    return frame["side_name"].astype(str).str.lower() + "|" + frame["archetype_policy_key"].astype(str)


def _fit_score_value_map(reference: pd.DataFrame, *, value_col: str, bins: int = 8) -> dict[str, Any]:
    score = pd.to_numeric(reference["score"], errors="coerce").to_numpy(dtype=np.float32)
    value = pd.to_numeric(reference[value_col], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(score) & np.isfinite(value)
    if int(valid.sum()) < 500:
        return {"edges": np.array([0.0, 1.0], dtype=np.float32), "global": np.zeros(1, dtype=np.float32), "local": {}}
    edges = np.unique(np.nanquantile(score[valid], np.linspace(0.0, 1.0, bins + 1))).astype(np.float32)
    if len(edges) < 3:
        edges = np.array([float(np.nanmin(score[valid])) - 1e-6, float(np.nanmax(score[valid])) + 1e-6], dtype=np.float32)
    bucket = np.clip(np.searchsorted(edges[1:-1], score, side="right"), 0, len(edges) - 2)
    n_bins = len(edges) - 1
    counts = np.bincount(bucket[valid], minlength=n_bins).astype(np.float32)
    sums = np.bincount(bucket[valid], weights=value[valid], minlength=n_bins).astype(np.float64)
    global_mean = float(np.nanmean(value[valid]))
    global_rates = ((sums + 120.0 * global_mean) / np.maximum(counts + 120.0, 1.0)).astype(np.float32)
    keys = _local_key(reference).to_numpy(dtype=object)
    local: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in np.unique(keys[valid]):
        mask = valid & (keys == key)
        if int(mask.sum()) < 80:
            continue
        local[str(key)] = (
            np.bincount(bucket[mask], weights=value[mask], minlength=n_bins).astype(np.float32),
            np.bincount(bucket[mask], minlength=n_bins).astype(np.float32),
        )
    return {"edges": edges, "global": global_rates, "local": local}


def _expected_ev(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    score = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    edges = np.asarray(state["edges"], dtype=np.float32)
    global_rates = np.asarray(state["global"], dtype=np.float32)
    bucket = np.clip(np.searchsorted(edges[1:-1], score, side="right"), 0, len(global_rates) - 1)
    result = global_rates[bucket].astype(np.float32, copy=True)
    keys = _local_key(frame).to_numpy(dtype=object)
    for key in pd.unique(keys):
        values = state["local"].get(str(key))
        if values is None:
            continue
        sums, counts = values
        pos = np.flatnonzero(keys == key).astype(np.int64, copy=False)
        local_rate = sums[bucket[pos]] / np.maximum(counts[bucket[pos]], 1.0)
        weight = counts[bucket[pos]] / np.maximum(counts[bucket[pos]] + 120.0, 1.0)
        result[pos] = weight * local_rate + (1.0 - weight) * global_rates[bucket[pos]]
    return result


def _residual_thresholds(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    values = pd.to_numeric(frame["__residual__"], errors="coerce")
    finite = values[np.isfinite(values)]
    global_cut = float(np.quantile(finite, 0.20)) if len(finite) >= 100 else -0.005
    local: dict[str, float] = {}
    for key, part in frame.groupby("__local_key__", observed=True, sort=False):
        raw = pd.to_numeric(part["__residual__"], errors="coerce").dropna()
        if len(raw) < 80:
            continue
        weight = len(raw) / (len(raw) + 120.0)
        local[str(key)] = float(weight * np.quantile(raw, 0.20) + (1.0 - weight) * global_cut)
    return global_cut, local


def _causal_residual_target(train: pd.DataFrame, *, value_col: str, label_col: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable").reset_index(drop=True)
    history: list[pd.DataFrame] = []
    labelled: list[pd.DataFrame] = []
    residual_history: list[pd.DataFrame] = []
    for block_index, positions in enumerate(np.array_split(np.arange(len(ordered), dtype=np.int64), 4)):
        block = ordered.iloc[positions].reset_index(drop=True)
        if block_index == 0:
            history.append(block)
            continue
        reference = pd.concat(history, ignore_index=True, copy=False)
        expectation = _fit_score_value_map(reference, value_col=value_col)
        residual = pd.DataFrame({"__residual__": pd.to_numeric(block[value_col], errors="coerce").to_numpy(dtype=np.float32) - _expected_ev(block, expectation), "__local_key__": _local_key(block).to_numpy(dtype=object)})
        if residual_history:
            global_cut, local_cut = _residual_thresholds(pd.concat(residual_history, ignore_index=True, copy=False))
            thresholds = np.asarray([local_cut.get(str(key), global_cut) for key in residual["__local_key__"]], dtype=np.float32)
            candidate = block.copy(deep=False)
            candidate[label_col] = (residual["__residual__"].to_numpy(dtype=np.float32) <= thresholds).astype(np.float32)
            labelled.append(candidate)
        residual_history.append(residual)
        history.append(block)
    if not labelled or not residual_history:
        return ordered.iloc[0:0].copy(), np.empty(0, dtype=np.float32), {"global_cut": -0.005, "local_cuts": {}, "state": _fit_score_value_map(ordered, value_col=value_col)}
    global_cut, local_cut = _residual_thresholds(pd.concat(residual_history, ignore_index=True, copy=False))
    labelled_frame = pd.concat(labelled, ignore_index=True, copy=False)
    return labelled_frame, labelled_frame[label_col].to_numpy(dtype=np.float32), {"global_cut": global_cut, "local_cuts": local_cut, "state": _fit_score_value_map(ordered, value_col=value_col)}


def _negative_residual_event(frame: pd.DataFrame, state: dict[str, Any], *, value_col: str) -> np.ndarray:
    residual = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=np.float32) - _expected_ev(frame, state["state"])
    thresholds = np.asarray([state["local_cuts"].get(str(key), state["global_cut"]) for key in _local_key(frame)], dtype=np.float32)
    return (residual <= thresholds).astype(np.float32)

DEFAULT_OUTPUT = ROOT / "data_perp/reports/negative_hit_residual_leaf_audit_20260719_v1"

# This is deliberately a compact, economically complete state basket.  The
# leaf audit is an interpretability exercise, not another broad feature search;
# a 100+ column basket produces unstable alternative split paths and consumes
# memory without making the resulting failure states easier to understand.
LEAF_AUDIT_FEATURES = (
    "score",
    "mark_perp_dislocation",
    "ob_top_liquidity_to_qv_24h",
    "shock_12h",
    "volatility_ratio_short_long",
    "volume_zscore_48h",
    "oi_drawdown_from_peak_24h",
    "oi_drawdown_from_peak_72h",
    "oi_recovery_fraction_24h",
    "bars_since_max_oi_drop_24h_norm",
    "oi_drop_acceleration_4h_rz",
    "oi_drop_deceleration_4h_rz",
    "price_down_oi_down_1h_rz",
    "price_down_oi_up_1h_rz",
    "price_up_oi_down_1h_rz",
    "price_up_oi_up_1h_rz",
    "price_down_oi_down_4h_rz",
    "price_up_oi_down_4h_rz",
    "price_recovery_fraction_24h",
    "price_minus_oi_recovery_24h",
    "price_recovery_oi_still_falling_1h",
    "funding_sign_persistence_24h",
    "funding_crowding_release_4h",
    "mkt_median_oi_chg_4h_rz",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_median_long_flush_intensity_4h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
    "market_breadth_recovery_from_24h_min",
    "market_downside_pairwise_corr_24h",
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_chg_4h",
    "gmm_prob_0",
    "gmm_prob_3",
    "gmm_mahal_2",
    "gmm_mahal_3",
    "dae_reconstruction_error_delta_1",
    "min_mahalanobis_delta_1",
    "gmm_posterior_delta_1",
    "cluster_speed",
    "state_spectral_top3_reconstruction_error",
)


def _projection_columns(root: Path) -> list[str]:
    """Select the explanatory pre-entry block before loading candidate shards.

    The transition ablation hydrates a few lifecycle fields from the static
    store.  That is useful when testing new features, but it materializes a
    wide timestamp x symbol panel and is unnecessary here: the frozen candidate
    ledger already contains the explanatory market-state block.  Projecting
    only this block keeps the leaf audit bounded while retaining exactly the
    inputs the recognizer is allowed to use.
    """

    first = next(iter(sorted(root.glob("candidates_*.parquet"))), None)
    if first is None:
        raise FileNotFoundError(f"No candidate shards under {root}")
    columns = pq.ParquetFile(first).schema_arrow.names
    keep = set(KEYS) | set(OUTCOMES) | {"score"}
    keep.update(name for name in LEAF_AUDIT_FEATURES if name in columns)
    return [name for name in columns if name in keep]


def _shards_before(root: Path, end: pd.Timestamp) -> list[Path]:
    return [
        path
        for path in sorted(root.glob("candidates_*.parquet"))
        if pd.Timestamp(pd.Period(path.stem.removeprefix("candidates_"), freq="M").start_time, tz="UTC") < end
    ]


def _score_cutoff(paths: list[Path], quantile: float = 0.80) -> float:
    """Calculate the candidate cutoff without retaining any feature columns."""

    scores = [pd.read_parquet(path, columns=["score"])["score"] for path in paths]
    values = pd.concat(scores, ignore_index=True, copy=False)
    return float(np.nanquantile(pd.to_numeric(values, errors="coerce"), quantile))


def _load_tail_projected(
    paths: list[Path],
    cutoff: float,
    requested: list[str],
    *,
    max_rows_per_shard: int | None = None,
) -> pd.DataFrame:
    """Load only the fixed train-tail population for one fold.

    The Parquet predicate is applied before pandas materialization.  A second
    in-memory mask protects against row-group statistics that cannot prune a
    mixed score row group.
    """

    frames: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        part = pd.read_parquet(
            path,
            columns=[name for name in requested if name in available],
            filters=[("score", ">=", float(cutoff))],
        )
        if part.empty:
            continue
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        score = pd.to_numeric(part["score"], errors="coerce")
        part = part.loc[part["__ts__"].notna() & score.ge(cutoff)]
        if max_rows_per_shard is not None and len(part) > max_rows_per_shard:
            # Evenly spaced positions retain beginning/middle/end behaviour and
            # avoid allowing any high-activity month to dominate the local fit.
            part = part.iloc[np.linspace(0, len(part) - 1, max_rows_per_shard, dtype=np.int64)]
        frames.append(part)
    if not frames:
        raise RuntimeError("No candidate-tail rows were loaded")
    data = pd.concat(frames, ignore_index=True, copy=False)
    data = data.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
    data = data.drop_duplicates(list(KEYS), keep="last").reset_index(drop=True)
    for name in data.select_dtypes(include=["float64"]).columns:
        data[name] = pd.to_numeric(data[name], downcast="float")
    return data


def _safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _leaf_paths(tree: dict[str, Any], feature_names: list[str]) -> dict[int, list[tuple[str, str]]]:
    """Return human-readable root-to-leaf conditions for a dumped tree."""

    paths: dict[int, list[tuple[str, str]]] = {}

    def visit(node: dict[str, Any], conditions: list[tuple[str, str]]) -> None:
        if "leaf_index" in node:
            paths[int(node["leaf_index"])] = conditions
            return
        feature_idx = int(node["split_feature"])
        feature = feature_names[feature_idx]
        threshold = node.get("threshold")
        decision = str(node.get("decision_type", "<="))
        if decision == "<=" and isinstance(threshold, (float, int)):
            threshold_text = f"{float(threshold):.5g}"
        else:
            threshold_text = str(threshold)
        visit(node["left_child"], [*conditions, (feature, f"{feature} {decision} {threshold_text}")])
        inverse = ">" if decision == "<=" else f"NOT {decision}"
        visit(node["right_child"], [*conditions, (feature, f"{feature} {inverse} {threshold_text}")])

    visit(tree["tree_structure"], [])
    return paths


def _fit_leaf_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    target: np.ndarray,
    residual_state: dict[str, Any],
    seed: int,
    month: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit side/archetype shallow recognizers and collect OOS leaf evidence."""

    medians = train[features].median(numeric_only=True).reindex(features).fillna(0.0)

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        return (
            frame.reindex(columns=features)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(medians)
            .to_numpy(dtype=np.float32)
        )

    train_groups = train.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False).groups
    side_groups = train.groupby("side_name", observed=True, sort=False).groups
    leaf_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for (side, archetype), test_idx in test.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).groups.items():
        local_idx = train_groups.get((side, archetype))
        source_key = f"{side}|{archetype}"
        fallback = "local"
        if local_idx is None or len(local_idx) < 800 or np.unique(target[local_idx]).size < 2:
            local_idx = side_groups.get(side)
            source_key = f"{side}|__side_fallback__"
            fallback = "side"
        if local_idx is None or len(local_idx) < 1_500 or np.unique(target[local_idx]).size < 2:
            continue
        local_idx = np.asarray(local_idx, dtype=np.int64)
        target_local = target[local_idx]
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.035,
                "num_leaves": 8,
                "max_depth": 3,
                "min_data_in_leaf": 120,
                "min_gain_to_split": 0.02,
                "lambda_l1": 1.0,
                "lambda_l2": 5.0,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "seed": int(seed),
                "num_threads": 2,
                "verbosity": -1,
            },
            lgb.Dataset(matrix(train.iloc[local_idx]), label=target_local, free_raw_data=True),
            num_boost_round=140,
        )
        test_pos = np.asarray(test_idx, dtype=np.int64)
        test_part = test.iloc[test_pos]
        observed = _negative_residual_event(
            test_part,
            # ``target`` is causally constructed on train; this observed event
            # is only the OOS evaluation label, never an input to the model.
            state=residual_state,
            value_col="clean_exec",
        )
        base_rate = float(np.mean(observed)) if len(observed) else 0.0
        x_test = matrix(test_part)
        prediction = np.asarray(model.predict(x_test), dtype=np.float32)
        leaves = np.asarray(model.predict(x_test, pred_leaf=True), dtype=np.int32)
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        dump = model.dump_model()
        trees = dump.get("tree_info", [])
        model_rows.append(
            {
                "month": month,
                "side_name": side,
                "archetype_policy_key": archetype,
                "training_source": source_key,
                "fallback": fallback,
                "train_rows": int(len(local_idx)),
                "oos_rows": int(len(test_part)),
                "oos_negative_hit_residual_rate": base_rate,
                "oos_auc": (
                    float(roc_auc_score(observed, prediction))
                    if np.unique(observed).size > 1
                    else np.nan
                ),
                "oos_average_precision": (
                    float(average_precision_score(observed, prediction))
                    if np.unique(observed).size > 1
                    else np.nan
                ),
            }
        )
        for feature, gain, split in zip(
            features,
            model.feature_importance(importance_type="gain"),
            model.feature_importance(importance_type="split"),
        ):
            if gain <= 0 and split <= 0:
                continue
            importance_rows.append(
                {
                    "month": month,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "training_source": source_key,
                    "feature": feature,
                    "gain": float(gain),
                    "splits": int(split),
                    "oos_rows": int(len(test_part)),
                }
            )
        for tree_index, tree in enumerate(trees):
            paths = _leaf_paths(tree, features)
            for leaf_index, conditions in paths.items():
                mask = leaves[:, tree_index] == leaf_index
                support = int(mask.sum())
                if support < 40:
                    continue
                event_rate = float(np.mean(observed[mask]))
                features_path = tuple(dict.fromkeys(feature for feature, _rule in conditions))
                leaf_rows.append(
                    {
                        "month": month,
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "training_source": source_key,
                        "fallback": fallback,
                        "tree_index": int(tree_index),
                        "leaf_index": int(leaf_index),
                        "path_depth": int(len(conditions)),
                        "path_rules": " AND ".join(rule for _feature, rule in conditions),
                        "path_features": " | ".join(features_path),
                        "oos_rows": support,
                        "oos_negative_hit_residual_rate": event_rate,
                        "oos_baseline_rate": base_rate,
                        "oos_event_lift": event_rate / max(base_rate, 1e-6),
                        "oos_mean_ev_after_1pct": float(pd.to_numeric(test_part.loc[mask, "ev_after_1pct"], errors="coerce").mean()),
                        "oos_clean_exec_rate": float(pd.to_numeric(test_part.loc[mask, "clean_exec"], errors="coerce").mean()),
                    }
                )
    return leaf_rows, importance_rows, model_rows


def _aggregate_combinations(leaf: pd.DataFrame) -> pd.DataFrame:
    if leaf.empty:
        return leaf
    work = leaf.copy(deep=False)
    work["expected_events"] = work["oos_rows"] * work["oos_negative_hit_residual_rate"]
    work["baseline_events"] = work["oos_rows"] * work["oos_baseline_rate"]
    rows: list[dict[str, Any]] = []
    for combo, part in work.groupby("path_features", observed=True, sort=False):
        support = float(part["oos_rows"].sum())
        event_rate = float(part["expected_events"].sum() / max(support, 1.0))
        baseline = float(part["baseline_events"].sum() / max(support, 1.0))
        rows.append(
            {
                "path_features": combo,
                "path_depth": int(part["path_depth"].median()),
                "oos_rows": int(support),
                "months": int(part["month"].nunique()),
                "side_archetype_cells": int(part[["side_name", "archetype_policy_key"]].drop_duplicates().shape[0]),
                "leaf_occurrences": int(len(part)),
                "oos_negative_hit_residual_rate": event_rate,
                "oos_baseline_rate": baseline,
                "oos_event_lift": event_rate / max(baseline, 1e-6),
                "oos_mean_ev_after_1pct": float(np.average(part["oos_mean_ev_after_1pct"], weights=part["oos_rows"])),
                "oos_clean_exec_rate": float(np.average(part["oos_clean_exec_rate"], weights=part["oos_rows"])),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["months", "oos_event_lift", "oos_rows"], ascending=[False, False, False], kind="stable"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--feature-store-id", default=DEFAULT_FEATURE_STORE_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "start", "output": str(args.output)}), flush=True)
    months = _parse_months(args.months)
    all_end = pd.Timestamp((max(months) + 1).start_time, tz="UTC")
    requested = _projection_columns(args.candidate_root)
    features = [name for name in LEAF_AUDIT_FEATURES if name in requested]
    print(json.dumps({"event": "schema", "feature_count": len(features)}), flush=True)
    leaves: list[dict[str, Any]] = []
    importance: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for fold, month in enumerate(months):
        start = pd.Timestamp(month.start_time, tz="UTC")
        end = pd.Timestamp((month + 1).start_time, tz="UTC")
        train_paths = _shards_before(args.candidate_root, start)
        test_paths = _shards_before(args.candidate_root, end)
        test_paths = [path for path in test_paths if path not in train_paths]
        cutoff = _score_cutoff(train_paths)
        print(json.dumps({"event": "fold_start", "month": str(month), "cutoff": cutoff}), flush=True)
        train = _load_tail_projected(train_paths, cutoff, requested, max_rows_per_shard=6_000)
        test = _load_tail_projected(test_paths, cutoff, requested)
        print(json.dumps({"event": "fold_loaded", "month": str(month), "train_rows": len(train), "test_rows": len(test)}), flush=True)
        if len(train) < 25_000 or len(test) < 1_000:
            del train, test
            gc.collect()
            continue
        residual_train, residual_target, residual_state = _causal_residual_target(
            train,
            value_col="clean_exec",
            label_col="__negative_hit_residual_event__",
        )
        part_leaf, part_importance, part_models = _fit_leaf_models(
            residual_train,
            test,
            features=features,
            target=residual_target,
            residual_state=residual_state,
            seed=int(args.seed + fold),
            month=str(month),
        )
        leaves.extend(part_leaf)
        importance.extend(part_importance)
        models.extend(part_models)
        del train, test, residual_train, residual_target
        gc.collect()
    args.output.mkdir(parents=True, exist_ok=True)
    leaf_frame = pd.DataFrame(leaves)
    leaf_frame.to_csv(args.output / "oos_leaf_rules.csv", index=False)
    combinations = _aggregate_combinations(leaf_frame)
    combinations.to_csv(args.output / "oos_feature_combinations.csv", index=False)
    importance_frame = pd.DataFrame(importance)
    if not importance_frame.empty:
        importance_frame["weighted_gain"] = importance_frame["gain"] * importance_frame["oos_rows"]
        summary = (
            importance_frame.groupby("feature", observed=True)
            .agg(
                folds=("month", "nunique"),
                side_archetype_cells=("archetype_policy_key", "nunique"),
                oos_rows=("oos_rows", "sum"),
                weighted_gain=("weighted_gain", "sum"),
                splits=("splits", "sum"),
            )
            .reset_index()
            .sort_values(["folds", "weighted_gain"], ascending=[False, False], kind="stable")
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(args.output / "oos_feature_importance.csv", index=False)
    pd.DataFrame(models).to_csv(args.output / "oos_model_coverage.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            _safe(
                {
                    "schema": "negative_hit_residual_leaf_audit_v1",
                    "months": [str(month) for month in months],
                    "features": features,
                    "feature_count": len(features),
                    "input_source": "frozen candidate ledger; no static-store hydration",
                    "target": "causal lower-tail clean_exec residual, score x side x archetype expectation",
                    "evaluation": "all leaf lifts and EV metrics are calculated on each OOS month only",
                    "leaf_model": "side x archetype shallow LGBM; depth=3; min_data_in_leaf=120",
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "output": str(args.output), "leaf_rows": len(leaf_frame)}), flush=True)


if __name__ == "__main__":
    main()
