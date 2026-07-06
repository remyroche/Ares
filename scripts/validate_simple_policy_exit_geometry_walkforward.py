#!/usr/bin/env python3
"""Walk-forward holdout validation for simple-policy exit geometry.

For each validation week, parameters are optimised on prior replayable rows only
and then evaluated on the withheld week using train-fitted EV curves. This keeps
the validation period out of both Optuna search and portfolio priority
calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    DEFAULT_CANDIDATES,
    DEFAULT_PATH_LEN,
    StrategyBundle,
    _candidate_table_for_overrides,
    _load_bundles,
    _metrics_row,
    _prepare_rows,
    _suggest_capital_lock,
    _suggest_geometry,
    _suggest_time_decay,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    _json_safe,
    _path_take,
)


DEFAULT_OUT_DIR = Path(
    "data_perp/reports/simple_policy_exit_geometry_walkforward_holdout"
)
DEFAULT_REGIME_COLUMN = "oof_regime_centroid_similarity_train"
DEFAULT_ARCHETYPE_COLUMN = "policy_archetype"
_GROUP_CANDIDATE_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_GROUP_CANDIDATE_CACHE_MAX = 8192


def _week_start(ts: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(ts, utc=True, errors="coerce")
    return timestamps.dt.floor("D") - pd.to_timedelta(timestamps.dt.weekday, unit="D")


def _timestamp_range(bundles: Iterable[StrategyBundle]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    values: List[pd.Timestamp] = []
    for bundle in bundles:
        if not bundle.rows.empty:
            values.extend(
                [
                    pd.to_datetime(bundle.rows["timestamp"].min(), utc=True),
                    pd.to_datetime(bundle.rows["timestamp"].max(), utc=True),
                ]
            )
    if not values:
        return pd.NaT, pd.NaT
    return min(values), max(values)


def _subset_bundles(
    bundles: Iterable[StrategyBundle],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> List[StrategyBundle]:
    out: List[StrategyBundle] = []
    for bundle in bundles:
        ts = pd.to_datetime(bundle.rows["timestamp"], utc=True, errors="coerce")
        mask = pd.Series(True, index=bundle.rows.index)
        if start is not None:
            mask &= ts >= pd.Timestamp(start)
        if end is not None:
            mask &= ts < pd.Timestamp(end)
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            continue
        out.append(
            replace(
                bundle,
                rows=bundle.rows.iloc[idx].reset_index(drop=True),
                paths=_path_take(bundle.paths, idx),
            )
        )
    return out


def _bundle_side(bundle: StrategyBundle) -> str:
    if "side" in bundle.rows.columns and not bundle.rows.empty:
        raw = bundle.rows["side"].iloc[0]
        try:
            if float(raw) < 0:
                return "short"
            if float(raw) > 0:
                return "long"
        except (TypeError, ValueError):
            text = str(raw).strip().lower()
            if text.startswith("short") or text in {"sell", "-1"}:
                return "short"
    strategy_id = str(bundle.strategy_id).lower()
    return "short" if strategy_id.startswith("short") else "long"


def _row_side_tokens(rows: pd.DataFrame) -> np.ndarray:
    if "side" not in rows.columns:
        return np.full(len(rows), "long", dtype=object)
    values = rows["side"].to_numpy()
    out: List[str] = []
    for raw in values:
        try:
            value = float(raw)
            out.append("short" if np.isfinite(value) and value < 0.0 else "long")
            continue
        except (TypeError, ValueError):
            text = str(raw).strip().lower()
            out.append("short" if text.startswith("short") or text in {"sell", "-1"} else "long")
    return np.asarray(out, dtype=object)


def _uses_state_labels(group_by: str) -> bool:
    text = str(group_by)
    return "regime" in text or "archetype" in text


def _label_prefix(group_by: str) -> str:
    return "archetype" if "archetype" in str(group_by) else "regime"


def _active_state_column(*, group_by: str, regime_column: str, archetype_column: str) -> str:
    if "archetype" in str(group_by):
        return str(archetype_column)
    return str(regime_column)


def _resolve_state_column(
    rows: pd.DataFrame,
    *,
    group_by: str,
    regime_column: str,
    archetype_column: str,
) -> str:
    requested = _active_state_column(
        group_by=group_by,
        regime_column=regime_column,
        archetype_column=archetype_column,
    )
    if "archetype" not in str(group_by) or requested in rows.columns:
        return requested
    for candidate in (
        "policy_archetype",
        "archetype",
        "__archetype_policy_key__",
        "__archetype_label_family__",
        "local_archetype",
        "local_side_archetype",
    ):
        if candidate in rows.columns:
            return candidate
    return requested


def _safe_label_token(value: Any) -> str:
    text = str(value).strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    token = "_".join(part for part in "".join(chars).split("_") if part)
    return token or "missing"


def _numeric_like(values: pd.Series) -> bool:
    raw = values.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return False
    numeric = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return bool(float(numeric.notna().mean()) >= 0.90)


def _fit_regime_edges(
    bundles: Iterable[StrategyBundle],
    *,
    regime_column: str,
    regime_bins: int,
) -> List[float]:
    if int(regime_bins) <= 1 or not str(regime_column).strip():
        return []
    values: List[np.ndarray] = []
    for bundle in bundles:
        if regime_column not in bundle.rows.columns:
            continue
        series = pd.to_numeric(bundle.rows[regime_column], errors="coerce")
        arr = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if arr.size:
            values.append(arr)
    if not values:
        return []
    all_values = np.concatenate(values)
    quantiles = [i / float(regime_bins) for i in range(1, int(regime_bins))]
    edges = np.nanquantile(all_values, quantiles)
    clean_edges = sorted({float(v) for v in edges if np.isfinite(v)})
    return clean_edges


def _regime_labels(
    rows: pd.DataFrame,
    *,
    regime_column: str,
    regime_edges: Sequence[float],
    label_prefix: str = "regime",
) -> pd.Series:
    if not str(regime_column).strip() or regime_column not in rows.columns:
        return pd.Series(f"{label_prefix}_all", index=rows.index)
    raw_values = rows[regime_column]
    if not _numeric_like(raw_values):
        labels = []
        for value in raw_values:
            if pd.isna(value) or not str(value).strip():
                labels.append(f"{label_prefix}_missing")
            else:
                labels.append(f"{label_prefix}_{_safe_label_token(value)}")
        return pd.Series(labels, index=rows.index)
    values = pd.to_numeric(raw_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    edges = np.asarray(list(regime_edges), dtype=float)
    labels: List[str] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append(f"{label_prefix}_missing")
        elif edges.size == 0:
            labels.append(f"{label_prefix}_all")
        else:
            labels.append(
                f"{label_prefix}_q{int(np.searchsorted(edges, value, side='right'))}"
            )
    return pd.Series(labels, index=rows.index)


def _bundle_group_key(
    bundle: StrategyBundle,
    *,
    group_by: str,
    regime_label: str = "regime_all",
    side: str | None = None,
) -> str:
    side = str(side or _bundle_side(bundle))
    if group_by == "side":
        return side
    if group_by == "head":
        return str(bundle.strategy_id)
    if group_by == "side_head":
        return f"{side}|{bundle.strategy_id}"
    if group_by in {"regime", "archetype"}:
        return str(regime_label)
    if group_by in {"side_regime", "side_archetype"}:
        return f"{side}|{regime_label}"
    if group_by in {"side_regime_head", "side_archetype_head"}:
        return f"{side}|{regime_label}|{bundle.strategy_id}"
    return f"{regime_label}|{bundle.strategy_id}"


def _bundle_group_keys(
    bundle: StrategyBundle,
    *,
    group_by: str,
    regime_column: str,
    regime_edges: Sequence[float],
) -> List[str]:
    if not _uses_state_labels(group_by):
        return [_bundle_group_key(bundle, group_by=group_by)]
    labels = _regime_labels(
        bundle.rows,
        regime_column=regime_column,
        regime_edges=regime_edges,
        label_prefix=_label_prefix(group_by),
    )
    if str(group_by) in {"side_regime", "side_archetype", "side_regime_head", "side_archetype_head"}:
        sides = _row_side_tokens(bundle.rows)
        label_text = labels.astype(str).to_numpy()
        return sorted(
            {
                _bundle_group_key(
                    bundle,
                    group_by=group_by,
                    regime_label=str(label),
                    side=str(side),
                )
                for side, label in zip(sides, label_text)
            }
        )
    return sorted(
        {
            _bundle_group_key(bundle, group_by=group_by, regime_label=str(label))
            for label in labels.dropna().unique()
        }
    )


def _group_keys(
    bundles: Iterable[StrategyBundle],
    *,
    group_by: str,
    regime_column: str,
    regime_edges: Sequence[float],
) -> List[str]:
    keys: set[str] = set()
    for bundle in bundles:
        keys.update(
            _bundle_group_keys(
                bundle,
                group_by=group_by,
                regime_column=regime_column,
                regime_edges=regime_edges,
            )
        )
    return sorted(keys)


def _copy_group_overrides(
    group_overrides: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(group): dict(params)
        for group, params in dict(group_overrides or {}).items()
        if isinstance(params, Mapping) and params
    }


def _override_cache_key(
    params: Mapping[str, Any],
) -> str:
    return json.dumps(_json_safe(dict(params or {})), sort_keys=True, separators=(",", ":"))


def _cached_group_candidate_frame(
    *,
    bundle: StrategyBundle,
    idx: np.ndarray,
    group_key: str,
    regime_label: str,
    overrides: Mapping[str, Any],
    cost_pct: float,
    market_mode: str,
    group_by: str,
    regime_column: str,
) -> pd.DataFrame:
    params = dict(overrides or {})
    key = (
        id(bundle.rows),
        id(bundle.paths[0]),
        int(len(bundle.rows)),
        str(bundle.strategy_id),
        str(group_key),
        str(regime_label),
        str(group_by),
        str(regime_column),
        round(float(cost_pct), 10),
        str(market_mode),
        _override_cache_key(params),
    )
    cached = _GROUP_CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached.copy(deep=False)

    grouped_bundle = replace(
        bundle,
        rows=bundle.rows.iloc[idx].reset_index(drop=True),
        paths=_path_take(bundle.paths, idx),
    )
    frame = _candidate_table_for_overrides(
        [grouped_bundle],
        overrides=params,
        cost_pct=cost_pct,
        market_mode=market_mode,
        arm="__cached_group_candidate__",
    )
    if frame.empty:
        return frame
    if len(_GROUP_CANDIDATE_CACHE) >= _GROUP_CANDIDATE_CACHE_MAX:
        _GROUP_CANDIDATE_CACHE.clear()
    _GROUP_CANDIDATE_CACHE[key] = frame
    return frame.copy(deep=False)


def _candidate_table_for_group_overrides(
    bundles: Iterable[StrategyBundle],
    *,
    group_overrides: Mapping[str, Mapping[str, Any]],
    group_by: str,
    regime_column: str,
    regime_edges: Sequence[float],
    cost_pct: float,
    market_mode: str,
    arm: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    overrides_by_group = _copy_group_overrides(group_overrides)
    for bundle in bundles:
        if _uses_state_labels(group_by):
            labels = _regime_labels(
                bundle.rows,
                regime_column=regime_column,
                regime_edges=regime_edges,
                label_prefix=_label_prefix(group_by),
            )
            label_text = labels.astype(str).to_numpy()
            if str(group_by) in {
                "side_regime",
                "side_archetype",
                "side_regime_head",
                "side_archetype_head",
            }:
                side_text = _row_side_tokens(bundle.rows)
                group_units = sorted(
                    {
                        (str(side), str(label))
                        for side, label in zip(side_text, label_text)
                        if str(label) and str(label).lower() != "nan"
                    }
                )
            else:
                side_text = np.full(len(bundle.rows), _bundle_side(bundle), dtype=object)
                group_units = [
                    (_bundle_side(bundle), str(label))
                    for label in labels.dropna().unique()
                ]
        else:
            labels = pd.Series(f"{_label_prefix(group_by)}_all", index=bundle.rows.index)
            label_text = labels.astype(str).to_numpy()
            side_text = _row_side_tokens(bundle.rows)
            group_units = [(_bundle_side(bundle), f"{_label_prefix(group_by)}_all")]

        for group_side, regime_label in group_units:
            if str(group_by) in {
                "side_regime",
                "side_archetype",
                "side_regime_head",
                "side_archetype_head",
            }:
                idx = np.flatnonzero(
                    (label_text == str(regime_label)) & (side_text == str(group_side))
                )
            else:
                idx = np.flatnonzero(label_text == str(regime_label))
            if len(idx) == 0:
                continue
            group_key = _bundle_group_key(
                bundle,
                group_by=group_by,
                regime_label=str(regime_label),
                side=str(group_side),
            )
            frame = _cached_group_candidate_frame(
                bundle=bundle,
                idx=idx,
                group_key=group_key,
                regime_label=str(regime_label),
                overrides=overrides_by_group.get(group_key, {}),
                cost_pct=cost_pct,
                market_mode=market_mode,
                group_by=group_by,
                regime_column=regime_column,
            )
            if frame.empty:
                continue
            frame = frame.copy(deep=False)
            frame["exit_geometry_ablation_arm"] = str(arm)
            frame["exit_geometry_group_key"] = group_key
            frame["exit_geometry_group_by"] = str(group_by)
            frame["exit_geometry_regime_label"] = str(regime_label)
            frame["exit_geometry_regime_column"] = str(regime_column)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return (
        pd.concat(frames, ignore_index=True, copy=False)
        .sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _inner_fold_defs(
    bundles: Iterable[StrategyBundle],
    *,
    min_train_weeks: int,
    embargo_days: float,
    max_folds: int = 0,
) -> List[Dict[str, Any]]:
    rows = pd.concat([bundle.rows for bundle in bundles], ignore_index=True)
    rows["week_start"] = _week_start(rows["timestamp"])
    weeks = sorted(pd.Timestamp(v) for v in rows["week_start"].dropna().unique())
    folds: List[Dict[str, Any]] = []
    for fold_idx, validation_start in enumerate(weeks[int(min_train_weeks) :]):
        validation_start = pd.Timestamp(validation_start)
        validation_end = validation_start + pd.Timedelta(days=7)
        train_end = validation_start - pd.Timedelta(days=float(embargo_days))
        train_bundles = _subset_bundles(bundles, end=train_end)
        validation_bundles = _subset_bundles(
            bundles,
            start=validation_start,
            end=validation_end,
        )
        if not train_bundles or not validation_bundles:
            continue
        folds.append(
            {
                "inner_fold": f"inner_{fold_idx:02d}_{validation_start.date()}",
                "train_bundles": train_bundles,
                "validation_bundles": validation_bundles,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "train_end_exclusive": train_end,
            }
        )
        if int(max_folds) > 0 and len(folds) >= int(max_folds):
            break
    return folds


def _score_with_train_curve(
    *,
    train_candidates: pd.DataFrame,
    eval_candidates: pd.DataFrame,
    market_mode: str,
    global_threshold_floor: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if train_candidates.empty or eval_candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "objective": float("-inf"),
            "trade_count": 0,
            "net_pnl": 0.0,
        }
    ev_curve = fit_hierarchical_ev_curves(train_candidates)
    decisions, equity, metrics = replay_candidates(
        eval_candidates,
        PortfolioPolicyParams(global_threshold_floor=float(global_threshold_floor)),
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    return decisions, equity, dict(metrics)


def _evaluate_inner_folds(
    inner_folds: Iterable[Mapping[str, Any]],
    *,
    group_overrides: Mapping[str, Mapping[str, Any]],
    group_by: str,
    regime_column: str,
    regime_edges: Sequence[float],
    cost_pct: float,
    market_mode: str,
    global_threshold_floor: float,
    arm: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in inner_folds:
        train_candidates = _candidate_table_for_group_overrides(
            fold["train_bundles"],
            group_overrides=group_overrides,
            group_by=group_by,
            regime_column=regime_column,
            regime_edges=regime_edges,
            cost_pct=cost_pct,
            market_mode=market_mode,
            arm=f"{arm}_{fold['inner_fold']}_train",
        )
        validation_candidates = _candidate_table_for_group_overrides(
            fold["validation_bundles"],
            group_overrides=group_overrides,
            group_by=group_by,
            regime_column=regime_column,
            regime_edges=regime_edges,
            cost_pct=cost_pct,
            market_mode=market_mode,
            arm=f"{arm}_{fold['inner_fold']}_validation",
        )
        decisions, _equity, metrics = _score_with_train_curve(
            train_candidates=train_candidates,
            eval_candidates=validation_candidates,
            market_mode=market_mode,
            global_threshold_floor=global_threshold_floor,
        )
        row = _metrics_row(
            arm=arm,
            stage="inner_walkforward",
            overrides={},
            candidates=validation_candidates,
            decisions=decisions,
            metrics=metrics,
        )
        row.update(
            {
                "inner_fold": str(fold["inner_fold"]),
                "validation_start": pd.Timestamp(
                    fold["validation_start"]
                ).isoformat(),
                "validation_end": pd.Timestamp(fold["validation_end"]).isoformat(),
                "train_end_exclusive": pd.Timestamp(
                    fold["train_end_exclusive"]
                ).isoformat(),
            }
        )
        rows.append(row)
    return rows


def _delta_score_from_rows(
    *,
    candidate_rows: List[Mapping[str, Any]],
    baseline_rows: List[Mapping[str, Any]],
    worst_fold_penalty: float,
    std_penalty: float,
    pnl_weight: float,
    pnl_scale: float,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    baseline_by_fold = {str(row["inner_fold"]): row for row in baseline_rows}
    deltas: List[Dict[str, Any]] = []
    for row in candidate_rows:
        inner_fold = str(row["inner_fold"])
        baseline = baseline_by_fold.get(inner_fold)
        if baseline is None:
            continue
        candidate_objective = float(row.get("portfolio_objective", -np.inf))
        baseline_objective = float(baseline.get("portfolio_objective", -np.inf))
        candidate_pnl = float(row.get("portfolio_net_pnl", 0.0))
        baseline_pnl = float(baseline.get("portfolio_net_pnl", 0.0))
        if not np.isfinite(candidate_objective) or not np.isfinite(baseline_objective):
            continue
        deltas.append(
            {
                "inner_fold": inner_fold,
                "validation_start": row.get("validation_start"),
                "validation_end": row.get("validation_end"),
                "candidate_objective": candidate_objective,
                "baseline_objective": baseline_objective,
                "delta_objective": candidate_objective - baseline_objective,
                "candidate_net_pnl": candidate_pnl,
                "baseline_net_pnl": baseline_pnl,
                "delta_net_pnl": candidate_pnl - baseline_pnl,
                "candidate_accepted_trades": int(row.get("accepted_trades", 0) or 0),
                "baseline_accepted_trades": int(
                    baseline.get("accepted_trades", 0) or 0
                ),
            }
        )
    if not deltas:
        return -1.0e12, {"inner_fold_count": 0}, []

    delta_objectives = np.asarray(
        [row["delta_objective"] for row in deltas], dtype=float
    )
    delta_pnls = np.asarray([row["delta_net_pnl"] for row in deltas], dtype=float)
    mean_delta = float(np.nanmean(delta_objectives))
    worst_delta = float(np.nanmin(delta_objectives))
    std_delta = float(np.nanstd(delta_objectives))
    mean_delta_pnl = float(np.nanmean(delta_pnls))
    score = (
        mean_delta
        - max(0.0, -worst_delta) * max(0.0, float(worst_fold_penalty))
        - std_delta * max(0.0, float(std_penalty))
        + max(0.0, float(pnl_weight))
        * mean_delta_pnl
        / max(float(pnl_scale), 1e-12)
    )
    summary = {
        "inner_fold_count": int(len(deltas)),
        "mean_delta_objective": mean_delta,
        "worst_delta_objective": worst_delta,
        "std_delta_objective": std_delta,
        "mean_delta_net_pnl": mean_delta_pnl,
        "sum_delta_net_pnl": float(np.nansum(delta_pnls)),
        "passing_objective_folds": int(np.sum(delta_objectives > 0.0)),
        "passing_net_pnl_folds": int(np.sum(delta_pnls > 0.0)),
        "walkforward_delta_score": float(score),
    }
    return float(score), summary, deltas


def _walkforward_delta_score(
    inner_folds: Iterable[Mapping[str, Any]],
    *,
    baseline_rows: List[Mapping[str, Any]],
    group_overrides: Mapping[str, Mapping[str, Any]],
    group_by: str,
    regime_column: str,
    regime_edges: Sequence[float],
    cost_pct: float,
    market_mode: str,
    global_threshold_floor: float,
    arm: str,
    worst_fold_penalty: float,
    std_penalty: float,
    pnl_weight: float,
    pnl_scale: float,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    candidate_rows = _evaluate_inner_folds(
        inner_folds,
        group_overrides=group_overrides,
        group_by=group_by,
        regime_column=regime_column,
        regime_edges=regime_edges,
        cost_pct=cost_pct,
        market_mode=market_mode,
        global_threshold_floor=global_threshold_floor,
        arm=arm,
    )
    return _delta_score_from_rows(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        worst_fold_penalty=worst_fold_penalty,
        std_penalty=std_penalty,
        pnl_weight=pnl_weight,
        pnl_scale=pnl_scale,
    )


def _optimise_group_stage_delta(
    *,
    stage: str,
    target_group: str,
    current_group_overrides: Mapping[str, Mapping[str, Any]],
    suggest: Callable[[optuna.Trial], Dict[str, Any]],
    inner_folds: List[Mapping[str, Any]],
    baseline_rows: List[Mapping[str, Any]],
    n_trials: int,
    seed: int,
    group_by: str,
    regime_column: str,
    regime_edges: Sequence[float],
    cost_pct: float,
    market_mode: str,
    global_threshold_floor: float,
    min_accept_delta: float,
    worst_fold_penalty: float,
    std_penalty: float,
    pnl_weight: float,
    pnl_scale: float,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    current_overrides = _copy_group_overrides(current_group_overrides)
    current_score, current_summary, _current_deltas = _walkforward_delta_score(
        inner_folds,
        baseline_rows=baseline_rows,
        group_overrides=current_overrides,
        group_by=group_by,
        regime_column=regime_column,
        regime_edges=regime_edges,
        cost_pct=cost_pct,
        market_mode=market_mode,
        global_threshold_floor=global_threshold_floor,
        arm=f"{stage}_{target_group}_current",
        worst_fold_penalty=worst_fold_penalty,
        std_penalty=std_penalty,
        pnl_weight=pnl_weight,
        pnl_scale=pnl_scale,
    )
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_rows: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest(trial)
        trial_overrides = _copy_group_overrides(current_overrides)
        group_params = dict(trial_overrides.get(target_group, {}))
        group_params.update(trial_params)
        trial_overrides[target_group] = group_params
        score, summary, deltas = _walkforward_delta_score(
            inner_folds,
            baseline_rows=baseline_rows,
            group_overrides=trial_overrides,
            group_by=group_by,
            regime_column=regime_column,
            regime_edges=regime_edges,
            cost_pct=cost_pct,
            market_mode=market_mode,
            global_threshold_floor=global_threshold_floor,
            arm=f"{stage}_{target_group}_trial_{trial.number}",
            worst_fold_penalty=worst_fold_penalty,
            std_penalty=std_penalty,
            pnl_weight=pnl_weight,
            pnl_scale=pnl_scale,
        )
        row = {
            "stage": stage,
            "target_group": target_group,
            "trial_number": int(trial.number),
            "objective": float(score),
            "current_objective": float(current_score),
            **summary,
            "trial_params_json": json.dumps(_json_safe(trial_params), sort_keys=True),
            "group_params_json": json.dumps(_json_safe(group_params), sort_keys=True),
            "inner_deltas_json": json.dumps(_json_safe(deltas), sort_keys=True),
        }
        for key, value in trial_params.items():
            row[f"param_{key}"] = value
        trial_rows.append(row)
        trial.set_user_attr("metrics", row)
        return float(score) if np.isfinite(score) else -1.0e12

    if inner_folds and int(n_trials) > 0:
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        best_score = float(study.best_value)
        best_params = dict(study.best_trial.params)
    else:
        best_score = -1.0e12
        best_params = {}

    accepted = bool(best_score > current_score + float(min_accept_delta))
    next_overrides = _copy_group_overrides(current_overrides)
    if accepted:
        group_params = dict(next_overrides.get(target_group, {}))
        group_params.update(best_params)
        next_overrides[target_group] = group_params

    decision = {
        "stage": stage,
        "target_group": target_group,
        "accepted": accepted,
        "current_walkforward_delta_score": float(current_score),
        "best_walkforward_delta_score": float(best_score),
        "score_improvement": float(best_score - current_score),
        "min_accept_delta": float(min_accept_delta),
        **{f"current_{k}": v for k, v in current_summary.items()},
        "selected_params_json": json.dumps(
            _json_safe(next_overrides.get(target_group, {})),
            sort_keys=True,
        ),
        "best_trial_params_json": json.dumps(_json_safe(best_params), sort_keys=True),
    }
    return next_overrides, trial_rows, decision


def _suggest_threshold_policy(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "base_strategy_threshold": trial.suggest_float(
            "base_strategy_threshold", 0.70, 0.97, step=0.01
        ),
        "portfolio_rank_adjustment": trial.suggest_float(
            "portfolio_rank_adjustment", -0.05, 0.05, step=0.01
        ),
    }


def _suggest_sizing_policy(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "best_size_power": trial.suggest_float("best_size_power", 0.50, 2.50, step=0.25),
        "portfolio_size_multiplier": trial.suggest_categorical(
            "portfolio_size_multiplier", [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        ),
        "portfolio_wallet_cap_multiplier": trial.suggest_categorical(
            "portfolio_wallet_cap_multiplier", [0.25, 0.50, 0.75, 1.00]
        ),
    }


def _suggest_portfolio_policy(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "portfolio_priority_multiplier": trial.suggest_categorical(
            "portfolio_priority_multiplier", [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
        ),
        "portfolio_priority_adjustment": trial.suggest_float(
            "portfolio_priority_adjustment", -0.20, 0.20, step=0.025
        ),
        "portfolio_max_concurrent_per_strategy": trial.suggest_categorical(
            "portfolio_max_concurrent_per_strategy", [1, 2, 3, 4]
        ),
        "portfolio_max_new_entries_per_strategy_per_bar": trial.suggest_categorical(
            "portfolio_max_new_entries_per_strategy_per_bar", [1, 2, None]
        ),
    }


def _accepted_with_pnl(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted
    accepted["net_pnl"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    )
    start = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    end = pd.to_datetime(
        accepted["position_exit_timestamp"], utc=True, errors="coerce"
    )
    accepted["holding_hours"] = (end - start).dt.total_seconds() / 3600.0
    return accepted


def _write_walkforward_progress(
    *,
    out_dir: Path,
    stage_rows: List[Dict[str, Any]],
    improvement_rows: List[Dict[str, Any]],
    head_rows: List[Dict[str, Any]],
    trial_rows: List[Dict[str, Any]],
    selected_rows: List[Dict[str, Any]],
) -> None:
    """Persist fold-complete progress so long optimisation runs are recoverable."""
    pd.DataFrame(stage_rows).to_csv(
        out_dir / "walkforward_stage_summary.partial.csv",
        index=False,
    )
    pd.DataFrame(improvement_rows).to_csv(
        out_dir / "walkforward_improvement_summary.partial.csv",
        index=False,
    )
    pd.DataFrame(head_rows).to_csv(
        out_dir / "walkforward_head_summary.partial.csv",
        index=False,
    )
    pd.DataFrame(trial_rows).to_csv(
        out_dir / "walkforward_trials.partial.csv",
        index=False,
    )
    pd.DataFrame(selected_rows).to_csv(
        out_dir / "walkforward_selected_overrides.partial.csv",
        index=False,
    )


def _head_rows(
    *,
    fold: str,
    stage: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    decisions: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if decisions.empty:
        return []
    rows: List[Dict[str, Any]] = []
    work = decisions.copy()
    work["accepted_bool"] = work["accepted"].astype(bool)
    accepted = _accepted_with_pnl(work)
    for strategy_id, group in work.groupby("strategy_id", sort=True):
        acc = accepted.loc[accepted["strategy_id"].astype(str) == str(strategy_id)]
        rows.append(
            {
                "fold": fold,
                "stage": stage,
                "validation_start": validation_start.isoformat(),
                "validation_end": validation_end.isoformat(),
                "strategy_id": str(strategy_id),
                "candidate_rows": int(len(group)),
                "accepted_trades": int(group["accepted_bool"].sum()),
                "net_pnl": float(acc["net_pnl"].sum()) if not acc.empty else 0.0,
                "mean_net_return": (
                    float(
                        pd.to_numeric(
                            acc["position_net_return"], errors="coerce"
                        ).mean()
                    )
                    if not acc.empty
                    else 0.0
                ),
                "win_rate": (
                    float(
                        (
                            pd.to_numeric(
                                acc["position_net_return"], errors="coerce"
                            )
                            > 0.0
                        ).mean()
                    )
                    if not acc.empty
                    else 0.0
                ),
                "avg_holding_hours": (
                    float(acc["holding_hours"].mean()) if not acc.empty else 0.0
                ),
            }
        )
    return rows


def _improvement_row(
    *,
    fold: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
) -> Dict[str, Any]:
    base_obj = float(baseline.get("portfolio_objective", -np.inf))
    final_obj = float(final.get("portfolio_objective", -np.inf))
    base_pnl = float(baseline.get("portfolio_net_pnl", 0.0))
    final_pnl = float(final.get("portfolio_net_pnl", 0.0))
    return {
        "fold": fold,
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
        "baseline_objective": base_obj,
        "final_objective": final_obj,
        "delta_objective": final_obj - base_obj,
        "baseline_net_pnl": base_pnl,
        "final_net_pnl": final_pnl,
        "delta_net_pnl": final_pnl - base_pnl,
        "baseline_accepted_trades": int(baseline.get("accepted_trades", 0) or 0),
        "final_accepted_trades": int(final.get("accepted_trades", 0) or 0),
        "passes_objective": bool(final_obj > base_obj),
        "passes_net_pnl": bool(final_pnl > base_pnl),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-rank", type=float, default=0.70)
    parser.add_argument("--path-len", type=int, default=DEFAULT_PATH_LEN)
    parser.add_argument("--min-rows-per-strategy", type=int, default=5)
    parser.add_argument("--min-train-weeks", type=int, default=2)
    parser.add_argument("--embargo-days", type=float, default=1.0)
    parser.add_argument(
        "--max-validation-folds",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs; 0 uses all outer validation folds.",
    )
    parser.add_argument("--inner-min-train-weeks", type=int, default=1)
    parser.add_argument("--inner-embargo-days", type=float, default=1.0)
    parser.add_argument(
        "--max-inner-folds",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs; 0 uses all inner folds.",
    )
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--cost-pct", type=float, default=DEFAULT_POLICY_PER_SIDE_COST_PCT)
    parser.add_argument(
        "--round-trip-cost-pct",
        type=float,
        default=None,
        help=(
            "Optional round-trip cost override. When provided, the simulator "
            "uses half of this value as the per-side cost_pct."
        ),
    )
    parser.add_argument(
        "--group-by",
        default="regime_head",
        choices=[
            "regime_head",
            "side_regime_head",
            "side_regime",
            "regime",
            "archetype_head",
            "side_archetype_head",
            "side_archetype",
            "archetype",
            "side_head",
            "head",
            "side",
        ],
        help=(
            "Scope parameter overrides. Use side_archetype for execution policy "
            "studies without per-strategy masks."
        ),
    )
    parser.add_argument("--regime-column", default=DEFAULT_REGIME_COLUMN)
    parser.add_argument("--archetype-column", default=DEFAULT_ARCHETYPE_COLUMN)
    parser.add_argument("--regime-bins", type=int, default=3)
    parser.add_argument("--min-accept-delta", type=float, default=0.0)
    parser.add_argument("--worst-fold-penalty", type=float, default=1.0)
    parser.add_argument("--std-penalty", type=float, default=0.25)
    parser.add_argument("--pnl-weight", type=float, default=0.0)
    parser.add_argument("--pnl-scale", type=float, default=10_000.0)
    parser.add_argument(
        "--enable-geometry-overrides",
        action="store_true",
        help=(
            "Opt in to the experimental regime/head exit-geometry override stages. "
            "By default this validator reports baseline-only walk-forward metrics."
        ),
    )
    parser.add_argument(
        "--enable-execution-policy-overrides",
        action="store_true",
        help=(
            "Opt in to side/archetype policy stages for thresholds, sizing, "
            "portfolio controls, and exit geometry."
        ),
    )
    parser.add_argument(
        "--download-missing-1m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Opt in to downloading/materializing missing 1m delayed-entry execution "
            "candles. Also enables missing 15m chart fallback for true 15m replay."
        ),
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero unless final policy beats baseline in aggregate and every fold.",
    )
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EPM_SIMPLE_POLICY_1M_DOWNLOAD"] = (
        "1" if bool(args.download_missing_1m) else "0"
    )
    os.environ["EPM_SIMPLE_POLICY_15M_DOWNLOAD"] = (
        "1" if bool(args.download_missing_1m) else "0"
    )
    cost_pct = (
        float(args.round_trip_cost_pct) / 2.0
        if args.round_trip_cost_pct is not None
        else float(args.cost_pct)
    )

    rows = _prepare_rows(args.candidates, min_rank=float(args.min_rank))
    raw_timestamps = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    state_column = _resolve_state_column(
        rows,
        group_by=str(args.group_by),
        regime_column=str(args.regime_column),
        archetype_column=str(args.archetype_column),
    )
    bundles = _load_bundles(
        rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=int(args.min_rows_per_strategy),
    )
    replay_rows = pd.concat([bundle.rows for bundle in bundles], ignore_index=True)
    replay_rows["week_start"] = _week_start(replay_rows["timestamp"])
    weeks = sorted(pd.Timestamp(v) for v in replay_rows["week_start"].dropna().unique())
    validation_weeks = weeks[int(args.min_train_weeks) :]
    if int(args.max_validation_folds) > 0:
        validation_weeks = validation_weeks[: int(args.max_validation_folds)]

    if bool(args.enable_execution_policy_overrides):
        stage_plan = [
            ("A1_threshold_policy", "threshold_policy", _suggest_threshold_policy),
            ("A2_sizing_policy", "sizing_policy", _suggest_sizing_policy),
            ("A3_portfolio_policy", "portfolio_policy", _suggest_portfolio_policy),
            ("A4_geometry_envelope", "geometry_envelope", _suggest_geometry),
            ("A5_capital_lock", "capital_lock", _suggest_capital_lock),
            ("A6_time_decay", "time_decay", _suggest_time_decay),
        ]
    elif bool(args.enable_geometry_overrides):
        stage_plan = [
            ("A1_geometry_envelope", "geometry_envelope", _suggest_geometry),
            ("A2_capital_lock", "capital_lock", _suggest_capital_lock),
            ("A3_time_decay", "time_decay", _suggest_time_decay),
        ]
    else:
        stage_plan = []
    final_stage_arm = stage_plan[-1][0] if stage_plan else "A0_baseline"

    stage_rows: List[Dict[str, Any]] = []
    improvement_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []

    for fold_idx, validation_start in enumerate(validation_weeks):
        validation_start = pd.Timestamp(validation_start)
        validation_end = validation_start + pd.Timedelta(days=7)
        train_end = validation_start - pd.Timedelta(days=float(args.embargo_days))
        train_bundles = _subset_bundles(bundles, end=train_end)
        validation_bundles = _subset_bundles(
            bundles,
            start=validation_start,
            end=validation_end,
        )
        if not train_bundles or not validation_bundles:
            continue

        fold = f"fold_{fold_idx:02d}_{validation_start.date()}"
        train_raw_rows = int((raw_timestamps < train_end).sum())
        validation_raw_rows = int(
            ((raw_timestamps >= validation_start) & (raw_timestamps < validation_end)).sum()
        )
        train_replay_rows = int(sum(len(bundle.rows) for bundle in train_bundles))
        validation_replay_rows = int(sum(len(bundle.rows) for bundle in validation_bundles))
        train_start, train_last = _timestamp_range(train_bundles)
        validation_first, validation_last = _timestamp_range(validation_bundles)
        regime_edges = _fit_regime_edges(
            train_bundles,
            regime_column=state_column,
            regime_bins=int(args.regime_bins),
        )

        group_overrides: Dict[str, Dict[str, Any]] = {}
        fold_stage_rows: Dict[str, Dict[str, Any]] = {}
        inner_folds = _inner_fold_defs(
            train_bundles,
            min_train_weeks=int(args.inner_min_train_weeks),
            embargo_days=float(args.inner_embargo_days),
            max_folds=int(args.max_inner_folds),
        )
        inner_baseline_rows = _evaluate_inner_folds(
            inner_folds,
            group_overrides={},
            group_by=str(args.group_by),
            regime_column=state_column,
            regime_edges=regime_edges,
            cost_pct=cost_pct,
            market_mode=str(args.market_mode),
            global_threshold_floor=float(args.global_threshold_floor),
            arm=f"{fold}_inner_baseline",
        )
        target_groups = _group_keys(
            train_bundles,
            group_by=str(args.group_by),
            regime_column=state_column,
            regime_edges=regime_edges,
        )
        print(
            "[walkforward] "
            f"fold={fold} train_rows={train_replay_rows} "
            f"validation_rows={validation_replay_rows} "
            f"inner_folds={len(inner_folds)} target_groups={len(target_groups)}",
            flush=True,
        )

        for stage_arm, stage, suggest in [
            ("A0_baseline", "baseline", None),
            *stage_plan,
        ]:
            print(
                f"[walkforward] fold={fold} stage={stage_arm} start",
                flush=True,
            )
            if suggest is not None:
                for group_idx, target_group in enumerate(target_groups):
                    print(
                        "[walkforward] "
                        f"fold={fold} stage={stage_arm} "
                        f"group={group_idx + 1}/{len(target_groups)} "
                        f"target={target_group}",
                        flush=True,
                    )
                    group_overrides, stage_trials, selected = _optimise_group_stage_delta(
                        stage=stage,
                        target_group=target_group,
                        current_group_overrides=group_overrides,
                        suggest=suggest,
                        inner_folds=inner_folds,
                        baseline_rows=inner_baseline_rows,
                        n_trials=int(args.n_trials),
                        seed=(
                            int(args.seed)
                            + fold_idx * 100_000
                            + len(trial_rows) * 17
                            + group_idx * 101
                        ),
                        group_by=str(args.group_by),
                        regime_column=state_column,
                        regime_edges=regime_edges,
                        cost_pct=cost_pct,
                        market_mode=str(args.market_mode),
                        global_threshold_floor=float(args.global_threshold_floor),
                        min_accept_delta=float(args.min_accept_delta),
                        worst_fold_penalty=float(args.worst_fold_penalty),
                        std_penalty=float(args.std_penalty),
                        pnl_weight=float(args.pnl_weight),
                        pnl_scale=float(args.pnl_scale),
                    )
                    selected.update(
                        {
                            "fold": fold,
                            "train_start": train_start.isoformat(),
                            "train_end_exclusive": train_end.isoformat(),
                            "validation_start": validation_start.isoformat(),
                            "validation_end": validation_end.isoformat(),
                            "group_by": str(args.group_by),
                            "regime_column": state_column,
                            "archetype_column": str(args.archetype_column),
                            "regime_bins": int(args.regime_bins),
                            "regime_edges_json": json.dumps(_json_safe(regime_edges)),
                            "inner_fold_count": int(len(inner_folds)),
                        }
                    )
                    selected_rows.append(selected)
                    for row in stage_trials:
                        row = dict(row)
                        row["fold"] = fold
                        row["train_start"] = train_start.isoformat()
                        row["train_end_exclusive"] = train_end.isoformat()
                        row["validation_start"] = validation_start.isoformat()
                        row["validation_end"] = validation_end.isoformat()
                        row["group_by"] = str(args.group_by)
                        row["regime_column"] = state_column
                        row["archetype_column"] = str(args.archetype_column)
                        row["regime_bins"] = int(args.regime_bins)
                        row["regime_edges_json"] = json.dumps(_json_safe(regime_edges))
                        trial_rows.append(row)

            train_candidates = _candidate_table_for_group_overrides(
                train_bundles,
                group_overrides=group_overrides if suggest is not None else {},
                group_by=str(args.group_by),
                regime_column=state_column,
                regime_edges=regime_edges,
                cost_pct=cost_pct,
                market_mode=str(args.market_mode),
                arm=f"{fold}_{stage_arm}_train",
            )
            validation_candidates = _candidate_table_for_group_overrides(
                validation_bundles,
                group_overrides=group_overrides if suggest is not None else {},
                group_by=str(args.group_by),
                regime_column=state_column,
                regime_edges=regime_edges,
                cost_pct=cost_pct,
                market_mode=str(args.market_mode),
                arm=f"{fold}_{stage_arm}_validation",
            )
            decisions, _equity, metrics = _score_with_train_curve(
                train_candidates=train_candidates,
                eval_candidates=validation_candidates,
                market_mode=str(args.market_mode),
                global_threshold_floor=float(args.global_threshold_floor),
            )
            row = _metrics_row(
                arm=stage_arm,
                stage=stage,
                overrides={},
                candidates=validation_candidates,
                decisions=decisions,
                metrics=metrics,
            )
            row.update(
                {
                    "fold": fold,
                    "train_start": train_start.isoformat(),
                    "train_last_timestamp": train_last.isoformat(),
                    "train_end_exclusive": train_end.isoformat(),
                    "validation_start": validation_start.isoformat(),
                    "validation_end": validation_end.isoformat(),
                    "validation_first_timestamp": validation_first.isoformat(),
                    "validation_last_timestamp": validation_last.isoformat(),
                    "train_raw_rows": train_raw_rows,
                    "validation_raw_rows": validation_raw_rows,
                    "train_replay_rows": train_replay_rows,
                    "validation_replay_rows": validation_replay_rows,
                    "train_candidate_rows": int(len(train_candidates)),
                    "validation_candidate_rows": int(len(validation_candidates)),
                    "group_by": str(args.group_by),
                    "regime_column": state_column,
                    "archetype_column": str(args.archetype_column),
                    "regime_bins": int(args.regime_bins),
                    "regime_edges_json": json.dumps(_json_safe(regime_edges)),
                    "inner_fold_count": int(len(inner_folds)),
                    "group_overrides_json": json.dumps(
                        _json_safe(group_overrides if suggest is not None else {}),
                        sort_keys=True,
                    ),
                }
            )
            stage_rows.append(row)
            fold_stage_rows[stage_arm] = row
            head_rows.extend(
                _head_rows(
                    fold=fold,
                    stage=stage,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    decisions=decisions,
                )
            )
            print(
                "[walkforward] "
                f"fold={fold} stage={stage_arm} "
                f"validation_candidates={len(validation_candidates)} "
                f"accepted={row.get('accepted_trades', 0)} "
                f"net_pnl={row.get('portfolio_net_pnl', 0.0)}",
                flush=True,
            )

        if "A0_baseline" in fold_stage_rows and final_stage_arm in fold_stage_rows:
            improvement_rows.append(
                _improvement_row(
                    fold=fold,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    baseline=fold_stage_rows["A0_baseline"],
                    final=fold_stage_rows[final_stage_arm],
                )
            )
        _write_walkforward_progress(
            out_dir=args.out_dir,
            stage_rows=stage_rows,
            improvement_rows=improvement_rows,
            head_rows=head_rows,
            trial_rows=trial_rows,
            selected_rows=selected_rows,
        )
        print(f"[walkforward] fold={fold} complete", flush=True)

    stage_summary = pd.DataFrame(stage_rows)
    improvements = pd.DataFrame(improvement_rows)
    head_summary = pd.DataFrame(head_rows)
    trials = pd.DataFrame(trial_rows)
    selected = pd.DataFrame(selected_rows)
    if trials.empty:
        trials = pd.DataFrame(
            columns=[
                "stage",
                "target_group",
                "trial_number",
                "objective",
                "current_objective",
                "inner_fold_count",
                "mean_delta_objective",
                "mean_delta_net_pnl",
                "walkforward_delta_score",
                "trial_params_json",
                "group_params_json",
                "inner_deltas_json",
            ]
        )
    if selected.empty:
        selected = pd.DataFrame(
            columns=[
                "stage",
                "target_group",
                "accepted",
                "current_walkforward_delta_score",
                "best_walkforward_delta_score",
                "score_improvement",
                "min_accept_delta",
                "selected_params_json",
                "best_trial_params_json",
                "fold",
                "validation_start",
                "validation_end",
            ]
        )

    if improvements.empty:
        verdict = {
            "status": "fail",
            "reason": "no_walkforward_folds",
            "fold_count": 0,
        }
    else:
        baseline_net_pnl = float(improvements["baseline_net_pnl"].sum())
        final_net_pnl = float(improvements["final_net_pnl"].sum())
        baseline_objective = float(improvements["baseline_objective"].mean())
        final_objective = float(improvements["final_objective"].mean())
        all_folds_pass_net_pnl = bool(improvements["passes_net_pnl"].all())
        all_folds_pass_objective = bool(improvements["passes_objective"].all())
        aggregate_pass_net_pnl = bool(final_net_pnl > baseline_net_pnl)
        aggregate_pass_objective = bool(final_objective > baseline_objective)
        comparison_enabled = bool(
            args.enable_geometry_overrides or args.enable_execution_policy_overrides
        )
        verdict = {
            "status": (
                (
                    "pass"
                    if aggregate_pass_net_pnl
                    and aggregate_pass_objective
                    and all_folds_pass_net_pnl
                    and all_folds_pass_objective
                    else "fail"
                )
                if comparison_enabled
                else "baseline_only"
            ),
            "comparison_enabled": comparison_enabled,
            "final_stage_arm": str(final_stage_arm),
            "fold_count": int(len(improvements)),
            "passing_net_pnl_folds": int(improvements["passes_net_pnl"].sum()),
            "passing_objective_folds": int(improvements["passes_objective"].sum()),
            "all_folds_pass_net_pnl": all_folds_pass_net_pnl,
            "all_folds_pass_objective": all_folds_pass_objective,
            "aggregate_pass_net_pnl": aggregate_pass_net_pnl,
            "aggregate_pass_objective": aggregate_pass_objective,
            "baseline_net_pnl": baseline_net_pnl,
            "final_net_pnl": final_net_pnl,
            "delta_net_pnl": final_net_pnl - baseline_net_pnl,
            "baseline_mean_objective": baseline_objective,
            "final_mean_objective": final_objective,
            "delta_mean_objective": final_objective - baseline_objective,
        }

    stage_summary.to_csv(args.out_dir / "walkforward_stage_summary.csv", index=False)
    improvements.to_csv(args.out_dir / "walkforward_improvement_summary.csv", index=False)
    head_summary.to_csv(args.out_dir / "walkforward_head_summary.csv", index=False)
    trials.to_csv(args.out_dir / "walkforward_trials.csv", index=False)
    selected.to_csv(args.out_dir / "walkforward_selected_overrides.csv", index=False)

    manifest = {
        "generated_by": "validate_simple_policy_exit_geometry_walkforward",
        "candidate_path": str(args.candidates),
        "raw_rows": int(len(rows)),
        "replay_rows": int(len(replay_rows)),
        "path_survival_fraction": (
            float(len(replay_rows) / len(rows)) if len(rows) else 0.0
        ),
        "data_root": str(args.data_root),
        "market_mode": str(args.market_mode),
        "min_rank": float(args.min_rank),
        "path_len": int(args.path_len),
        "min_rows_per_strategy": int(args.min_rows_per_strategy),
        "min_train_weeks": int(args.min_train_weeks),
        "embargo_days": float(args.embargo_days),
        "max_validation_folds": int(args.max_validation_folds),
        "inner_min_train_weeks": int(args.inner_min_train_weeks),
        "inner_embargo_days": float(args.inner_embargo_days),
        "max_inner_folds": int(args.max_inner_folds),
        "n_trials_per_stage": int(args.n_trials),
        "geometry_overrides_enabled": bool(args.enable_geometry_overrides),
        "execution_policy_overrides_enabled": bool(
            args.enable_execution_policy_overrides
        ),
        "download_missing_1m": bool(args.download_missing_1m),
        "stage_plan": [
            {"arm": arm, "stage": stage}
            for arm, stage, _suggest in stage_plan
        ],
        "global_threshold_floor": float(args.global_threshold_floor),
        "cost_pct": float(cost_pct),
        "per_side_cost_pct": float(cost_pct),
        "round_trip_cost_pct": float(cost_pct * 2.0),
        "group_by": str(args.group_by),
        "regime_column": state_column,
        "requested_regime_column": str(args.regime_column),
        "archetype_column": str(args.archetype_column),
        "regime_bins": int(args.regime_bins),
        "objective": {
            "name": "inner_walkforward_delta_vs_baseline",
            "min_accept_delta": float(args.min_accept_delta),
            "worst_fold_penalty": float(args.worst_fold_penalty),
            "std_penalty": float(args.std_penalty),
            "pnl_weight": float(args.pnl_weight),
            "pnl_scale": float(args.pnl_scale),
        },
        "strategy_count": int(len(bundles)),
        "source_replay_start": replay_rows["timestamp"].min().isoformat(),
        "source_replay_end": replay_rows["timestamp"].max().isoformat(),
        "verdict": verdict,
        "folds": _json_safe(improvement_rows),
    }
    (args.out_dir / "walkforward_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2)
    )
    (args.out_dir / "walkforward_verdict.json").write_text(
        json.dumps(_json_safe(verdict), indent=2)
    )
    if (
        args.require_pass
        and bool(args.enable_geometry_overrides or args.enable_execution_policy_overrides)
        and verdict.get("status") != "pass"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
