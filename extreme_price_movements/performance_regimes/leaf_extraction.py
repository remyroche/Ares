"""Extract timestamp-level LightGBM leaves and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.first_stage_models import (
    FirstStageModelBundle,
)
from extreme_price_movements.performance_regimes.labels import (
    StrategyPerformanceLabelBundle,
)
from extreme_price_movements.performance_regimes.leaf_scoring import (
    estimate_leaf_stability,
    prune_leaves,
    score_directional_edges,
    score_leaf_oof_contribution,
)
from extreme_price_movements.performance_regimes.membership import (
    active_positions_from_membership,
)


@dataclass(frozen=True)
class ExtractedLeaf:
    leaf_uid: str
    strategy: str
    direction: Literal["bad", "good"]
    fold_id: int
    tree_id: int
    leaf_id: int
    leaf_value: float
    parent_value: float | None
    coverage: float
    weighted_coverage: float
    n_active: int
    weighted_n_active: float
    leaf_label_mean: float
    global_label_mean: float
    leaf_strategy_perf_mean: float
    global_strategy_perf_mean: float
    directional_label_edge: float
    positive_label_edge: float
    label_edge_mass: float
    directional_perf_edge: float
    positive_perf_edge: float
    perf_edge_mass: float
    oof_contribution: float
    contribution_share: float
    stability: float
    split_path_features: tuple[str, ...]
    split_path_thresholds: tuple[float, ...]
    split_path_operators: tuple[str, ...]
    timestamp_membership: np.ndarray
    active_positions: np.ndarray | None = None


@dataclass(frozen=True)
class LeafTable:
    leaves: tuple[ExtractedLeaf, ...]
    frame: pd.DataFrame
    diagnostics: pd.DataFrame


def _tree_paths(tree: dict, feature_names: list[str]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}

    def rec(node: dict, features: list[str], thresholds: list[float], operators: list[str], parent_value):
        if "leaf_index" in node:
            out[int(node["leaf_index"])] = {
                "leaf_value": float(node.get("leaf_value", np.nan)),
                "parent_value": float(parent_value) if parent_value is not None else None,
                "features": tuple(features),
                "thresholds": tuple(thresholds),
                "operators": tuple(operators),
            }
            return
        split_feature = int(node.get("split_feature", -1))
        feature = feature_names[split_feature] if 0 <= split_feature < len(feature_names) else f"f{split_feature}"
        threshold = float(node.get("threshold", np.nan))
        decision_type = str(node.get("decision_type", "<="))
        internal_value = node.get("internal_value", parent_value)
        rec(
            node.get("left_child", {}),
            features + [feature],
            thresholds + [threshold],
            operators + [decision_type],
            internal_value,
        )
        rec(
            node.get("right_child", {}),
            features + [feature],
            thresholds + [threshold],
            operators + [f"not({decision_type})"],
            internal_value,
        )

    rec(tree, [], [], [], None)
    return out


def _model_leaf_paths(model, feature_names: list[str]) -> dict[tuple[int, int], dict[str, object]]:
    try:
        dump = model.booster_.dump_model()
    except Exception:
        return {}
    paths: dict[tuple[int, int], dict[str, object]] = {}
    for tree_info in dump.get("tree_info", []):
        tree_id = int(tree_info.get("tree_index", len(paths)))
        for leaf_id, meta in _tree_paths(tree_info.get("tree_structure", {}), feature_names).items():
            paths[(tree_id, int(leaf_id))] = meta
    return paths


def _time_blocks(index: pd.Index, block_count: int = 8) -> pd.Series:
    ts = pd.to_datetime(index, utc=True, errors="coerce")
    if pd.notna(ts).all() and len(ts) > 0:
        bins = np.array_split(np.arange(len(ts)), min(block_count, len(ts)))
        labels = np.zeros(len(ts), dtype=int)
        for i, block in enumerate(bins):
            labels[block] = i
        return pd.Series(labels, index=index)
    return pd.Series(np.arange(len(index)) // max(len(index) // max(block_count, 1), 1), index=index)


def _brier_improvement_for_positions(
    active_pos: np.ndarray,
    *,
    y: np.ndarray,
    sample_weight: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    denominator: float,
) -> float:
    """Return weighted Brier degradation from ablating active positions to baseline."""

    if denominator <= 0.0:
        return 0.0
    pos = np.asarray(active_pos, dtype=np.int64)
    pos = pos[(pos >= 0) & (pos < y.size)]
    if pos.size == 0:
        return 0.0
    yy = y[pos]
    ww = sample_weight[pos]
    model = model_pred[pos]
    base = baseline_pred[pos]
    ok = np.isfinite(yy) & np.isfinite(ww) & np.isfinite(model) & np.isfinite(base) & (ww >= 0.0)
    if not ok.any():
        return 0.0
    yy = np.clip(yy[ok].astype(np.float64, copy=False), 0.0, 1.0)
    ww = np.maximum(ww[ok].astype(np.float64, copy=False), 1e-12)
    model = np.clip(model[ok].astype(np.float64, copy=False), 0.0, 1.0)
    base = np.clip(base[ok].astype(np.float64, copy=False), 0.0, 1.0)
    diff = ww * ((yy - base) ** 2 - (yy - model) ** 2)
    return float(np.sum(diff) / max(float(denominator), 1e-12))


def _brier_denominator(
    *,
    y: np.ndarray,
    sample_weight: np.ndarray,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> float:
    ok = (
        np.isfinite(y)
        & np.isfinite(sample_weight)
        & np.isfinite(model_pred)
        & np.isfinite(baseline_pred)
        & (sample_weight >= 0.0)
    )
    if not ok.any():
        return 0.0
    return float(np.sum(np.maximum(sample_weight[ok].astype(np.float64, copy=False), 1e-12)))


def extract_model_leaves(
    model_bundle: FirstStageModelBundle,
    X_t: pd.DataFrame,
    labels: StrategyPerformanceLabelBundle,
    *,
    timestamp_col: str = "timestamp",
) -> LeafTable:
    """Extract validation-fold timestamp memberships for every model leaf."""

    X = (
        X_t.sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32, copy=False)
    )
    all_leaves: list[ExtractedLeaf] = []
    rows: list[dict[str, object]] = []
    stability_time_blocks = _time_blocks(X.index)
    for (strategy, direction), result in model_bundle.by_strategy_direction.items():
        label_set = labels.by_strategy[strategy]
        y = (label_set.bad_label if direction == "bad" else label_set.good_label).reindex(X.index).fillna(0.5)
        sw = (
            label_set.bad_sample_weight if direction == "bad" else label_set.good_sample_weight
        ).reindex(X.index).fillna(1.0)
        perf = label_set.ewma_performance.reindex(X.index).fillna(0.0)
        valid_pred = result.oof_predictions
        valid_base = result.baseline_oof_predictions
        y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float32, copy=False)
        sw_arr = pd.to_numeric(sw, errors="coerce").to_numpy(dtype=np.float32, copy=False)
        pred_arr = (
            pd.to_numeric(valid_pred.reindex(X.index), errors="coerce")
            .to_numpy(dtype=np.float32, copy=False)
        )
        base_arr = (
            pd.to_numeric(valid_base.reindex(X.index), errors="coerce")
            .to_numpy(dtype=np.float32, copy=False)
        )
        brier_denominator = _brier_denominator(
            y=y_arr,
            sample_weight=sw_arr,
            model_pred=pred_arr,
            baseline_pred=base_arr,
        )
        total_improvement = _brier_improvement_for_positions(
            np.flatnonzero(np.isfinite(pred_arr)).astype(np.int32, copy=False),
            y=y_arr,
            sample_weight=sw_arr,
            model_pred=pred_arr,
            baseline_pred=base_arr,
            denominator=brier_denominator,
        )
        for fold in result.fold_models:
            valid_idx = np.asarray(fold.valid_idx, dtype=int)
            if valid_idx.size == 0:
                continue
            X_valid = (
                X.iloc[valid_idx]
                .reindex(columns=fold.feature_columns)
                .fillna(fold.fill_values)
                .astype(np.float32, copy=False)
            )
            try:
                leaf_matrix = np.asarray(fold.model.predict(X_valid, pred_leaf=True))
            except Exception:
                continue
            if leaf_matrix.ndim == 1:
                leaf_matrix = leaf_matrix.reshape(-1, 1)
            path_lookup = _model_leaf_paths(fold.model, list(fold.feature_columns))
            y_valid_arr = y.iloc[valid_idx].to_numpy(dtype=np.float32, copy=False)
            sw_valid_arr = sw.iloc[valid_idx].to_numpy(dtype=np.float32, copy=False)
            perf_valid_arr = perf.iloc[valid_idx].to_numpy(dtype=np.float32, copy=False)
            finite_weight = np.where(np.isfinite(sw_valid_arr), sw_valid_arr, 0.0)
            finite_weight = np.maximum(finite_weight, 0.0).astype(np.float32, copy=False)
            valid_weight_sum = float(np.sum(finite_weight))
            safe_weight = np.maximum(finite_weight, 1e-12)
            global_label_mean = float(np.average(y_valid_arr, weights=safe_weight))
            global_perf_mean = float(np.average(perf_valid_arr, weights=safe_weight))
            for tree_id in range(leaf_matrix.shape[1]):
                leaf_values = np.asarray(leaf_matrix[:, tree_id], dtype=np.int64)
                unique_leaf_ids, inverse = np.unique(leaf_values, return_inverse=True)
                counts = np.bincount(inverse).astype(np.int64, copy=False)
                weighted_counts = np.bincount(inverse, weights=finite_weight).astype(np.float64, copy=False)
                label_sums = np.bincount(
                    inverse,
                    weights=np.asarray(y_valid_arr * finite_weight, dtype=np.float64),
                )
                perf_sums = np.bincount(
                    inverse,
                    weights=np.asarray(perf_valid_arr * finite_weight, dtype=np.float64),
                )
                for local_leaf_pos, leaf_id in enumerate(unique_leaf_ids):
                    full_active = np.zeros(len(X), dtype=bool)
                    local_active = inverse == int(local_leaf_pos)
                    active_pos = valid_idx[local_active].astype(np.int32, copy=False)
                    full_active[active_pos] = True
                    n_active = int(counts[local_leaf_pos])
                    if n_active <= 0:
                        continue
                    weighted_n_active = float(weighted_counts[local_leaf_pos])
                    coverage = float(n_active / max(len(valid_idx), 1))
                    weighted_coverage = float(weighted_n_active / max(valid_weight_sum, 1e-12))
                    if weighted_n_active > 1e-12:
                        leaf_label_mean = float(label_sums[local_leaf_pos] / weighted_n_active)
                        leaf_perf_mean = float(perf_sums[local_leaf_pos] / weighted_n_active)
                    else:
                        leaf_label_mean = float(np.nanmean(y_valid_arr[local_active]))
                        leaf_perf_mean = float(np.nanmean(perf_valid_arr[local_active]))
                    edge = score_directional_edges(
                        direction=direction,
                        leaf_label_mean=leaf_label_mean,
                        global_label_mean=global_label_mean,
                        leaf_strategy_perf_mean=leaf_perf_mean,
                        global_strategy_perf_mean=global_perf_mean,
                        weighted_coverage=weighted_coverage,
                    )
                    path = path_lookup.get((int(tree_id), int(leaf_id)), {})
                    uid = f"{strategy}__{direction}__fold{fold.fold_id}__tree{tree_id}__leaf{int(leaf_id)}"
                    proto = ExtractedLeaf(
                        leaf_uid=uid,
                        strategy=strategy,
                        direction=direction,
                        fold_id=int(fold.fold_id),
                        tree_id=int(tree_id),
                        leaf_id=int(leaf_id),
                        leaf_value=float(path.get("leaf_value", np.nan)),
                        parent_value=path.get("parent_value"),  # type: ignore[arg-type]
                        coverage=coverage,
                        weighted_coverage=weighted_coverage,
                        n_active=n_active,
                        weighted_n_active=weighted_n_active,
                        leaf_label_mean=leaf_label_mean,
                        global_label_mean=global_label_mean,
                        leaf_strategy_perf_mean=leaf_perf_mean,
                        global_strategy_perf_mean=global_perf_mean,
                        directional_label_edge=edge["directional_label_edge"],
                        positive_label_edge=edge["positive_label_edge"],
                        label_edge_mass=edge["label_edge_mass"],
                        directional_perf_edge=edge["directional_perf_edge"],
                        positive_perf_edge=edge["positive_perf_edge"],
                        perf_edge_mass=edge["perf_edge_mass"],
                        oof_contribution=0.0,
                        contribution_share=0.0,
                        stability=0.0,
                        split_path_features=tuple(path.get("features", ())),
                        split_path_thresholds=tuple(path.get("thresholds", ())),
                        split_path_operators=tuple(path.get("operators", ())),
                        timestamp_membership=full_active,
                        active_positions=active_pos,
                    )
                    contribution = _brier_improvement_for_positions(
                        active_pos,
                        y=y_arr,
                        sample_weight=sw_arr,
                        model_pred=pred_arr,
                        baseline_pred=base_arr,
                        denominator=brier_denominator,
                    )
                    contribution_share = contribution / max(total_improvement, 1e-12)
                    scored_proto = ExtractedLeaf(
                        **{
                            **proto.__dict__,
                            "oof_contribution": float(contribution),
                            "contribution_share": float(contribution_share),
                        }
                    )
                    stability = estimate_leaf_stability(
                        scored_proto,
                        time_blocks=stability_time_blocks,
                    )
                    leaf = ExtractedLeaf(
                        **{
                            **scored_proto.__dict__,
                            "stability": float(stability),
                        }
                    )
                    all_leaves.append(leaf)
                    rows.append(
                        {
                            key: value
                            for key, value in leaf.__dict__.items()
                            if key != "timestamp_membership"
                        }
                    )
    frame = pd.DataFrame(rows)
    diagnostics = (
        frame.groupby(["strategy", "direction"], dropna=False)
        .agg(leaf_count=("leaf_uid", "count"), positive_contribution_count=("oof_contribution", lambda s: int((s > 0).sum())))
        .reset_index()
        if not frame.empty
        else pd.DataFrame(columns=["strategy", "direction", "leaf_count", "positive_contribution_count"])
    )
    return LeafTable(leaves=tuple(all_leaves), frame=frame, diagnostics=diagnostics)


def extract_score_prune_leaves(
    model_bundle: FirstStageModelBundle,
    X_t: pd.DataFrame,
    labels: StrategyPerformanceLabelBundle,
    *,
    timestamp_col: str = "timestamp",
    **prune_kwargs,
) -> tuple[LeafTable, list[ExtractedLeaf]]:
    """Convenience wrapper for the spec's extract/score/prune step."""

    table = extract_model_leaves(
        model_bundle,
        X_t,
        labels,
        timestamp_col=timestamp_col,
    )
    return table, prune_leaves(list(table.leaves), **prune_kwargs)
