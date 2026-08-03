#!/usr/bin/env python3
"""Run fixed-geometry direct-utility ablations on the Jan--Jul 2022 PI grid.

This is deliberately non-walk-forward, non-promotable research.  Five fixed
calendar blocks are held out in turn; each side-local model trains on every
other block.  The direct deployed-policy net-return head is the only ranking
score.  Auxiliary heads share the representation but never enter an algebraic
EV formula.  Evaluation forms one pooled-global book across sides and
timestamps per month after a train-only side-local EV mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inverse_fixed_geometry_evaluation import (
    evaluate_inverse_fixed_geometry_arms,
)

SCHEMA = "inverse_pi_direct_utility_multitask_ablation_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
TARGET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
FOLD_INTERVALS = (
    ("2022-01-01T00:00:00Z", "2022-02-12T09:00:00Z"),
    ("2022-02-12T10:00:00Z", "2022-03-26T19:00:00Z"),
    ("2022-03-26T20:00:00Z", "2022-05-08T05:00:00Z"),
    ("2022-05-08T06:00:00Z", "2022-06-19T14:00:00Z"),
    ("2022-06-19T15:00:00Z", "2022-07-31T23:00:00Z"),
)
TRANSITION_PREFIX = "transition_raw__"
SCORE_FEATURES = ("base_score",)
INTERACTION_TRANSITION_SUFFIXES = (
    "__z_72h",
)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "future",
    "execution_net",
    "execution_gross",
    "execution_cost",
    "label",
    "mfe",
    "mae",
    "timeout",
    "adverse",
    "favorable",
    "opportunity",
    "conversion",
    "exit_",
    "realized",
    "realised",
)


@dataclass(frozen=True)
class Task:
    name: str
    column: str
    kind: str
    weight: float


DIRECT = Task("direct_net", TARGET, "regression", 4.0)
ECONOMIC_TASKS = (
    Task("opportunity", "__opportunity_occurred_12h__", "binary", 0.25),
    Task("favorable", "__favorable_payoff_return_12h__", "regression", 0.20),
    Task("adverse_first", "__adverse_competing_risk_12h__", "binary", 0.15),
    Task("adverse_magnitude", "execution_mae_return_12h", "regression", 0.15),
    Task("conversion", "__exit_conversion_loss_return_12h__", "regression", 0.15),
    Task("timeout", "__timeout_outcome_12h__", "binary", 0.10),
)
PATH_TASKS = (
    Task("peak_mfe", "__log1p_peak_mfe_atr_12h__", "regression", 0.05),
    Task(
        "time_to_mfe",
        "__log1p_time_to_first_meaningful_mfe_hours_12h__",
        "regression",
        0.05,
    ),
    Task(
        "mae_before_mfe",
        "__log1p_mae_before_meaningful_mfe_atr_12h__",
        "regression",
        0.05,
    ),
    Task(
        "bars_decreasing",
        "__log1p_bars_before_price_stops_decreasing_12h__",
        "regression",
        0.05,
    ),
    Task(
        "future_slope",
        "__log1p_future_slope_atr_per_hour_12h__",
        "regression",
        0.05,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_feature_names(features: Sequence[str]) -> None:
    for feature in features:
        lower = feature.lower()
        if any(token in lower for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"forbidden outcome/action feature: {feature}")


def feature_arms(
    feature_columns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    ordered = tuple(dict.fromkeys(feature_columns))
    validate_feature_names(ordered)
    transition = tuple(name for name in ordered if name.startswith(TRANSITION_PREFIX))
    levels = tuple(
        name
        for name in ordered
        if not name.startswith(TRANSITION_PREFIX)
    )
    asset = tuple(
        name
        for name in levels
        if not name.startswith("market_")
        and not name.startswith("btc_minus_alt_")
    )
    market = tuple(name for name in levels if name not in asset)
    bounded_transition = tuple(
        name
        for name in transition
        if name.endswith(INTERACTION_TRANSITION_SUFFIXES)
    )
    if len(levels) != 44 or len(transition) != 25 or len(bounded_transition) != 5:
        raise ValueError(
            "expected exactly 44 level fields, 25 transition fields and "
            "5 predeclared interaction fields"
        )
    return {
        "market": (*SCORE_FEATURES, *levels),
        "transition_context": (*SCORE_FEATURES, *transition),
        "market_transition": (*SCORE_FEATURES, *levels, *transition),
        "market_transition_interactions": (
            *SCORE_FEATURES,
            *levels,
            *transition,
            *(f"interaction__base_score__x__{name}" for name in bounded_transition),
        ),
    }


def add_bounded_interactions(
    frame: pd.DataFrame, features: Sequence[str],
) -> pd.DataFrame:
    output = frame.copy()
    for feature in features:
        prefix = "interaction__base_score__x__"
        if feature.startswith(prefix):
            source = feature.removeprefix(prefix)
            output[feature] = (
                pd.to_numeric(output["base_score"], errors="raise")
                * pd.to_numeric(output[source], errors="raise")
            )
    return output


def task_arms() -> dict[str, tuple[Task, ...]]:
    return {
        "direct_only": (DIRECT,),
        "economic_multitask": (DIRECT, *ECONOMIC_TASKS),
        "economic_path_multitask": (DIRECT, *ECONOMIC_TASKS, *PATH_TASKS),
    }


def experiment_arms(
    features: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    required = {
        "market",
        "transition_context",
        "market_transition",
        "market_transition_interactions",
    }
    if set(features) != required:
        raise ValueError("feature-arm contract changed")
    return (
        ("market", "direct_only"),
        ("market", "economic_multitask"),
        ("transition_context", "economic_multitask"),
        ("market_transition", "economic_multitask"),
        ("market_transition_interactions", "economic_multitask"),
        ("market_transition_interactions", "economic_path_multitask"),
    )


def fold_ids(frame: pd.DataFrame) -> np.ndarray:
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    result = np.full(len(frame), -1, dtype=np.int8)
    for fold, (start, end) in enumerate(FOLD_INTERVALS):
        mask = signal.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        result[mask.to_numpy()] = fold
    if (result < 0).any():
        unexpected = signal.loc[result < 0].head().astype(str).tolist()
        raise ValueError(f"rows outside the fixed January--July blocks: {unexpected}")
    return result


def purged_training_mask(frame: pd.DataFrame, fold: int) -> np.ndarray:
    if fold not in range(len(FOLD_INTERVALS)):
        raise ValueError("invalid fold")
    evaluation = frame.loc[frame["fold"].eq(fold)]
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="raise")
    resolution = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    first_decision = pd.to_datetime(
        evaluation["decision_timestamp"], utc=True, errors="raise"
    ).min()
    last_resolution = pd.to_datetime(
        evaluation["execution_label_end_utc"], utc=True, errors="raise"
    ).max()
    return (
        frame["fold"].ne(fold).to_numpy()
        & (
            resolution.lt(first_decision).to_numpy()
            | decision.gt(last_resolution).to_numpy()
        )
    )


def stable_global_top_mask(
    frame: pd.DataFrame, score: Sequence[float], fraction: float
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    values = np.asarray(score, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("ranking score is incomplete")
    order = pd.DataFrame(
        {
            "position": np.arange(len(frame)),
            "candidate_id": frame["candidate_id"].astype(str),
            "score": values,
        }
    ).sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    count = max(1, int(math.ceil(len(frame) * fraction)))
    mask = np.zeros(len(frame), dtype=bool)
    mask[order["position"].to_numpy()[:count]] = True
    return mask


def _rank_ic(actual: Sequence[float], predicted: Sequence[float]) -> float:
    left = pd.Series(np.asarray(actual, dtype=float)).rank(method="average")
    right = pd.Series(np.asarray(predicted, dtype=float)).rank(method="average")
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))


def _auc(actual: Sequence[float], predicted: Sequence[float]) -> float:
    target = np.asarray(actual, dtype=int)
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, np.asarray(predicted, dtype=float)))


def _ev_calibration_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    rank = pd.Series(predicted).rank(method="first", pct=True)
    bins = np.minimum((rank.to_numpy() * 10).astype(int), 9)
    errors = []
    weights = []
    for value in range(10):
        mask = bins == value
        if mask.any():
            errors.append(abs(float(actual[mask].mean() - predicted[mask].mean())))
            weights.append(int(mask.sum()))
    return float(np.average(errors, weights=weights))


def _decile_monotonicity(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rank = pd.Series(predicted).rank(method="first", pct=True)
    bins = np.minimum((rank.to_numpy() * 10).astype(int), 9)
    means = np.array(
        [actual[bins == value].mean() for value in range(10)], dtype=float
    )
    correlation = _rank_ic(np.arange(10, dtype=float), means)
    adjacent = float(np.mean(np.diff(means) >= 0.0))
    return correlation, adjacent


def fit_mapping(
    raw_score: Sequence[float], target: Sequence[float]
) -> IsotonicRegression:
    score = np.asarray(raw_score, dtype=float)
    actual = np.asarray(target, dtype=float)
    if len(score) < 500 or not np.isfinite(score).all() or not np.isfinite(actual).all():
        raise ValueError("mapping reference is incomplete")
    mapper = IsotonicRegression(out_of_bounds="clip", y_min=-0.25, y_max=0.25)
    mapper.fit(score, actual)
    return mapper


def _prepare_features(
    train: pd.DataFrame, evaluate: pd.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_x = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    eval_x = evaluate.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    train_x[~np.isfinite(train_x)] = np.nan
    eval_x[~np.isfinite(eval_x)] = np.nan
    median = np.nanmedian(train_x, axis=0)
    median[~np.isfinite(median)] = 0.0
    train_x = np.where(np.isnan(train_x), median, train_x)
    eval_x = np.where(np.isnan(eval_x), median, eval_x)
    low = np.quantile(train_x, 0.005, axis=0)
    high = np.quantile(train_x, 0.995, axis=0)
    train_x = np.clip(train_x, low, high)
    eval_x = np.clip(eval_x, low, high)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (
        ((train_x - mean) / scale).astype(np.float32),
        ((eval_x - mean) / scale).astype(np.float32),
        {
            "median": median.tolist(),
            "winsor_low": low.tolist(),
            "winsor_high": high.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
    )


def _prepare_targets(
    train: pd.DataFrame, tasks: Sequence[Task]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    values = []
    weights = []
    audit = []
    for task in tasks:
        raw = pd.to_numeric(train[task.column], errors="coerce").to_numpy(float)
        mask = np.isfinite(raw)
        if mask.mean() < 0.95:
            raise ValueError(f"task {task.name} lacks required support")
        if task.kind == "binary":
            unique = set(np.unique(raw[mask]))
            if not unique.issubset({0.0, 1.0}) or len(unique) < 2:
                raise ValueError(f"binary task {task.name} is degenerate")
            location, scale = 0.0, 1.0
            transformed = raw
        else:
            location = float(np.mean(raw[mask]))
            scale = max(float(np.std(raw[mask])), 1e-6)
            transformed = (raw - location) / scale
        transformed[~mask] = 0.0
        values.append(transformed.astype(np.float32))
        weights.append(np.float32(task.weight))
        audit.append(
            {
                "name": task.name,
                "column": task.column,
                "kind": task.kind,
                "weight": task.weight,
                "location": location,
                "scale": scale,
                "valid_rows": int(mask.sum()),
            }
        )
    return np.column_stack(values), np.asarray(weights), audit


def fit_shared_model(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    features: Sequence[str],
    tasks: Sequence[Task],
    seed: int,
    epochs: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class Network(nn.Module):
        def __init__(self, inputs: int, outputs: int):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(inputs, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(outputs)])

        def forward(self, value):
            shared = self.trunk(value)
            return torch.cat([head(shared) for head in self.heads], dim=1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(6, max(1, os.cpu_count() or 1)))
    train_x, eval_x, preprocessing = _prepare_features(train, evaluate, features)
    train_y, weights, target_audit = _prepare_targets(train, tasks)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=1024, shuffle=True, generator=generator
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Network(train_x.shape[1], len(tasks)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1.0e-2
    )
    weight_tensor = torch.from_numpy(weights).to(device)
    for _ in range(int(epochs)):
        model.train()
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch)
            losses = []
            for position, task in enumerate(tasks):
                if task.kind == "binary":
                    loss = functional.binary_cross_entropy_with_logits(
                        output[:, position], y_batch[:, position]
                    )
                else:
                    loss = functional.smooth_l1_loss(
                        output[:, position], y_batch[:, position], beta=1.0
                    )
                losses.append(loss)
            total = (torch.stack(losses) * weight_tensor).sum()
            total.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_raw = model(torch.from_numpy(train_x).to(device)).cpu().numpy()
        eval_raw = model(torch.from_numpy(eval_x).to(device)).cpu().numpy()
    train_predictions: dict[str, np.ndarray] = {}
    eval_predictions: dict[str, np.ndarray] = {}
    for position, task in enumerate(tasks):
        if task.kind == "binary":
            train_value = 1.0 / (1.0 + np.exp(-np.clip(train_raw[:, position], -40, 40)))
            eval_value = 1.0 / (1.0 + np.exp(-np.clip(eval_raw[:, position], -40, 40)))
        else:
            location = target_audit[position]["location"]
            scale = target_audit[position]["scale"]
            train_value = location + scale * train_raw[:, position]
            eval_value = location + scale * eval_raw[:, position]
        train_predictions[task.name] = train_value.astype(float)
        eval_predictions[task.name] = eval_value.astype(float)
    return train_predictions, eval_predictions, {
        "features": list(features),
        "tasks": target_audit,
        "preprocessing": preprocessing,
        "geometry": {
            "hidden": [64, 32],
            "dropout": 0.10,
            "optimizer": "AdamW",
            "learning_rate": 1.5e-3,
            "weight_decay": 1.0e-2,
            "batch_size": 1024,
            "epochs": int(epochs),
            "device": str(device),
        },
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(evaluate)),
    }


def evaluate_month(
    frame: pd.DataFrame,
    *,
    arm: str,
    feature_arm: str,
    task_arm: str,
    fold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    net = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    gross = pd.to_numeric(frame[GROSS], errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame[COST], errors="raise").to_numpy(float)
    score = pd.to_numeric(frame["mapped_direct_score"], errors="raise").to_numpy(float)
    raw = pd.to_numeric(frame["raw_direct_score"], errors="raise").to_numpy(float)
    monotonic_ic, adjacent = _decile_monotonicity(net, score)
    summary = {
        "arm": arm,
        "feature_arm": feature_arm,
        "task_arm": task_arm,
        "fold": fold,
        "month": str(frame["month"].iloc[0]),
        "rows": int(len(frame)),
        "rank_ic_mapped": _rank_ic(net, score),
        "rank_ic_raw": _rank_ic(net, raw),
        "net_positive_auc": _auc(net > 0.0, score),
        "mapped_mae_bps": float(np.mean(np.abs(net - score)) * 10_000.0),
        "mapped_bias_bps": float(np.mean(score - net) * 10_000.0),
        "ev_calibration_error_bps": float(
            _ev_calibration_error(net, score) * 10_000.0
        ),
        "decile_monotonicity_ic": monotonic_ic,
        "decile_adjacent_non_decreasing": adjacent,
    }
    if (
        "pred__opportunity" in frame
        and frame["pred__opportunity"].notna().all()
    ):
        summary["opportunity_auc"] = _auc(
            frame["__opportunity_occurred_12h__"],
            frame["pred__opportunity"],
        )
    tails = []
    for fraction in FRACTIONS:
        mask = stable_global_top_mask(frame, score, fraction)
        selected = frame.loc[mask]
        counts = selected.groupby("__ts__", observed=True).size()
        tails.append(
            {
                "arm": arm,
                "feature_arm": feature_arm,
                "task_arm": task_arm,
                "fold": fold,
                "month": str(frame["month"].iloc[0]),
                "fraction": fraction,
                "selected_rows": int(mask.sum()),
                "mean_net_bps": float(net[mask].mean() * 10_000.0),
                "mean_gross_bps": float(gross[mask].mean() * 10_000.0),
                "mean_cost_bps": float(cost[mask].mean() * 10_000.0),
                "positive_rate": float((net[mask] > 0.0).mean()),
                "long_share": float(selected["side_name"].eq("long").mean()),
                "distinct_assets": int(selected["__symbol__"].nunique()),
                "distinct_timestamps": int(selected["__ts__"].nunique()),
                "mean_candidates_per_selected_timestamp": float(counts.mean()),
                "ranking_scope": "one_pooled_global_cross_side_cross_timestamp",
            }
        )
    return tails, summary


def _validate_panel(
    panel: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, Any], tuple[str, ...]]:
    manifest = _json(manifest_path)
    if manifest.get("schema") != "inverse_exact_id_research_panel_v1":
        raise ValueError("panel manifest schema is invalid")
    output = manifest.get("outputs", {}).get("inverse_exact_id_research_panel", {})
    if output.get("sha256") != _sha256(panel):
        raise ValueError("panel manifest does not bind panel bytes")
    if (
        manifest.get("candidate_population_lineage")
        != "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
        or manifest.get("economics_label_contract")
        != "inverse_quote_notional_current_spread_counterfactual"
        or bool(manifest.get("promotion_eligible"))
        or manifest.get("oof_status") != "not_oof"
    ):
        raise ValueError("panel lineage/economics contract is invalid")
    features = tuple(manifest.get("feature_columns", ()))
    validate_feature_names(features)
    frame = pd.read_parquet(panel)
    frame["__ts__"] = pd.to_datetime(
        frame["signal_timestamp"], utc=True, errors="raise"
    )
    frame["__symbol__"] = frame["symbol"].astype(str)
    required = {
        *IDENTITY,
        TARGET,
        GROSS,
        COST,
        "execution_label_end_utc",
        *(task.column for task in (*ECONOMIC_TASKS, *PATH_TASKS)),
        *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"panel lacks required columns: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("panel candidate IDs are not unique")
    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"], utc=True, errors="raise"
    )
    frame["execution_label_end_utc"] = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not frame["execution_label_end_utc"].eq(
        frame["__ts__"] + pd.Timedelta(hours=13)
    ).all():
        raise ValueError("panel label timing is not signal+13h")
    net = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    gross = pd.to_numeric(frame[GROSS], errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame[COST], errors="raise").to_numpy(float)
    if not np.allclose(gross - cost, net, atol=1e-10, rtol=0.0):
        raise ValueError("panel violates net = gross - cost")
    return frame, manifest, features


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, panel_manifest, base_features = _validate_panel(
        args.panel, args.panel_manifest
    )
    features = feature_arms(base_features)
    all_features = tuple(
        dict.fromkeys(item for values in features.values() for item in values)
    )
    frame = add_bounded_interactions(frame, all_features)
    frame["fold"] = fold_ids(frame)
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    tasks = task_arms()
    predictions = []
    fit_audit = []
    for arm_index, (feature_arm, task_arm) in enumerate(experiment_arms(features)):
        arm = f"{feature_arm}__{task_arm}"
        active_features = features[feature_arm]
        active_tasks = tasks[task_arm]
        for fold, interval in enumerate(FOLD_INTERVALS):
            fold_outputs: list[pd.DataFrame] = []
            mapping_references: list[pd.DataFrame] = []
            fold_audits: list[dict[str, Any]] = []
            for side_index, side in enumerate(("long", "short")):
                train_mask = purged_training_mask(frame, fold)
                train = frame.loc[
                    train_mask & frame["side_name"].eq(side)
                ].reset_index(drop=True)
                evaluate = frame.loc[
                    frame["fold"].eq(fold) & frame["side_name"].eq(side)
                ].reset_index(drop=True)
                train_predictions, eval_predictions, audit = fit_shared_model(
                    train,
                    evaluate,
                    features=active_features,
                    tasks=active_tasks,
                    seed=int(args.seed + 1000 * arm_index + 100 * fold + side_index),
                    epochs=int(args.epochs),
                )
                mapping_references.append(
                    pd.DataFrame(
                        {
                            "side_name": side,
                            "raw_direct_score": train_predictions["direct_net"],
                            TARGET: pd.to_numeric(
                                train[TARGET], errors="raise"
                            ).to_numpy(float),
                        }
                    )
                )
                output = evaluate.loc[:, list(IDENTITY)].copy()
                output["decision_timestamp"] = evaluate[
                    "decision_timestamp"
                ].to_numpy()
                output["execution_label_end_utc"] = evaluate[
                    "execution_label_end_utc"
                ].to_numpy()
                for column in (
                    TARGET,
                    GROSS,
                    COST,
                    "__opportunity_occurred_12h__",
                ):
                    output[column] = evaluate[column].to_numpy()
                output["raw_direct_score"] = eval_predictions["direct_net"]
                for name, values in eval_predictions.items():
                    if name != "direct_net":
                        output[f"pred__{name}"] = values
                output["arm"] = arm
                output["feature_arm"] = feature_arm
                output["task_arm"] = task_arm
                output["fold"] = fold
                output["evaluation_block_start"] = interval[0]
                output["evaluation_block_end"] = interval[1]
                fold_outputs.append(output)
                audit.update(
                    {
                        "arm": arm,
                        "feature_arm": feature_arm,
                        "task_arm": task_arm,
                        "fold": fold,
                        "evaluation_interval": list(interval),
                        "side": side,
                    }
                )
                fold_audits.append(audit)
            mapping_reference = pd.concat(mapping_references, ignore_index=True)
            pooled_mapper = fit_mapping(
                mapping_reference["raw_direct_score"],
                mapping_reference[TARGET],
            )
            max_resolution = frame.loc[
                purged_training_mask(frame, fold), "execution_label_end_utc"
            ].max()
            for output, audit in zip(fold_outputs, fold_audits):
                side = str(output["side_name"].iloc[0])
                side_reference = mapping_reference.loc[
                    mapping_reference["side_name"].eq(side)
                ]
                side_mapper = fit_mapping(
                    side_reference["raw_direct_score"],
                    side_reference[TARGET],
                )
                raw = output["raw_direct_score"].to_numpy(float)
                shrink_weight = float(len(side_reference) / (len(side_reference) + 5000.0))
                output["mapped_direct_score"] = (
                    shrink_weight * side_mapper.predict(raw)
                    + (1.0 - shrink_weight) * pooled_mapper.predict(raw)
                )
                output["mapped_score"] = output["mapped_direct_score"]
                output["execution_decision_utc"] = output["decision_timestamp"]
                output["eligible"] = True
                output["mapping_status"] = "out_of_block_train_only_noncausal"
                output["mapping_max_label_resolution_utc"] = max_resolution
                output["month"] = output["__ts__"].dt.strftime("%Y-%m")
                output["selection_month"] = output["month"]
                predictions.append(output)
                audit["mapping"] = {
                    "family": "out_of_block_pooled_plus_side_shrunk_isotonic",
                    "pooled_reference_rows": int(len(mapping_reference)),
                    "side_reference_rows": int(len(side_reference)),
                    "side_shrink_weight": shrink_weight,
                    "side_shrink_prior_rows": 5000,
                    "maximum_label_resolution_utc": str(max_resolution),
                    "evaluation_rows": int(len(output)),
                    "common_output_unit": "decimal_policy_net_return",
                    "causal": False,
                }
                fit_audit.append(audit)
    scored = pd.concat(predictions, ignore_index=True)
    tail_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    for (arm, month), group in scored.groupby(["arm", "month"], sort=True):
        tails, summary = evaluate_month(
            group.reset_index(drop=True),
            arm=arm,
            feature_arm=str(group["feature_arm"].iloc[0]),
            task_arm=str(group["task_arm"].iloc[0]),
            fold=int(group["fold"].iloc[0]),
        )
        tail_rows.extend(tails)
        month_rows.append(summary)
    tails = pd.DataFrame(tail_rows)
    monthly = pd.DataFrame(month_rows)
    top10 = tails.loc[tails["fraction"].eq(0.10)]
    arm_summary = (
        top10.groupby(["arm", "feature_arm", "task_arm"], sort=True)
        .agg(
            months=("month", "nunique"),
            mean_top10_net_bps=("mean_net_bps", "mean"),
            worst_month_top10_net_bps=("mean_net_bps", "min"),
            mean_top10_gross_bps=("mean_gross_bps", "mean"),
            mean_top10_positive_rate=("positive_rate", "mean"),
            min_long_share=("long_share", "min"),
            max_long_share=("long_share", "max"),
        )
        .reset_index()
    )
    arm_summary["side_balance_gate"] = (
        arm_summary["min_long_share"].ge(0.05)
        & arm_summary["max_long_share"].le(0.95)
    )
    arm_summary["coverage_gate"] = arm_summary["months"].eq(7)
    core = (
        tails.loc[tails["fraction"].isin((0.05, 0.10, 0.20))]
        .groupby("arm", sort=True)["mean_net_bps"]
        .mean()
    )
    arm_summary["selection_score"] = (
        arm_summary["arm"].map(core)
        + 0.5 * arm_summary["worst_month_top10_net_bps"]
    )
    arm_summary.loc[
        ~(arm_summary["side_balance_gate"] & arm_summary["coverage_gate"]),
        "selection_score",
    ] = -1.0e9
    arm_summary = arm_summary.sort_values(
        ["selection_score", "arm"], ascending=[False, True]
    ).reset_index(drop=True)
    reusable = evaluate_inverse_fixed_geometry_arms(
        scored,
        top_fractions=FRACTIONS,
        expected_months=tuple(f"2022-{month:02d}" for month in range(1, 8)),
        baseline_arm="market__direct_only",
        arm_metadata_cols=("feature_arm", "task_arm"),
        evaluation_month_col="selection_month",
    )

    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "oof_scores.parquet"
    tails_path = args.output_dir / "pooled_global_tail_metrics.csv"
    monthly_path = args.output_dir / "monthly_model_metrics.csv"
    summary_path = args.output_dir / "arm_summary.csv"
    fit_path = args.output_dir / "fit_audit.json"
    reusable_monthly_path = args.output_dir / "matched_monthly_global_topk.csv"
    reusable_summary_path = args.output_dir / "matched_arm_summary.csv"
    comparison_path = args.output_dir / "matched_arm_comparisons.csv"
    monotonicity_path = args.output_dir / "matched_decile_monotonicity.csv"
    selection_path = args.output_dir / "matched_global_selections.parquet"
    scored.to_parquet(score_path, index=False, compression="zstd")
    tails.to_csv(tails_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    arm_summary.to_csv(summary_path, index=False)
    reusable.monthly.to_csv(reusable_monthly_path, index=False)
    reusable.summary.to_csv(reusable_summary_path, index=False)
    reusable.comparisons.to_csv(comparison_path, index=False)
    reusable.monotonicity.to_csv(monotonicity_path, index=False)
    reusable.selections.to_parquet(selection_path, index=False, compression="zstd")
    _write_json(fit_path, {"fits": _json_safe(fit_audit)})
    manifest = {
        "schema": SCHEMA,
        "status": "fixed_geometry_non_walk_forward_diagnostic_complete",
        "evidence_scope": "inverse_pi_market_grid_causal_features_research_not_oof",
        "oof_status": "calendar_block_out_of_block_diagnostic_not_strict_walk_forward",
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "population_lineage": panel_manifest["candidate_population_lineage"],
        "economics": panel_manifest["economics_label_contract"],
        "contracts": {
            "primary_target": TARGET,
            "ranking_score": "mapped direct policy-net head only",
            "auxiliary_role": "shared-representation regularizers only",
            "no_algebraic_auxiliary_ev": True,
            "transition_role": "conditional context only; never direct veto/control",
            "folds": [list(interval) for interval in FOLD_INTERVALS],
            "validation": "non_walk_forward_leave_calendar_block_out",
            "purge": "two-sided 12h path-overlap purge around each held-out block",
            "mapping": (
                "out-of-block pooled isotonic plus side-local isotonic shrunk "
                "toward pooled with 5,000-row prior; explicitly non-causal"
            ),
            "selection": "one pooled-global top-k per month after mapping; never per timestamp",
            "hpo": "none; one predeclared geometry and seed per fit",
            "portfolio_constraints": "not_applied_research_diagnostic",
        },
        "feature_arms": {name: list(values) for name, values in features.items()},
        "task_arms": {
            name: [task.name for task in values] for name, values in tasks.items()
        },
        "experiment_arms": [
            f"{feature_arm}__{task_arm}"
            for feature_arm, task_arm in experiment_arms(features)
        ],
        "rows": int(len(frame)),
        "scored_rows": int(len(scored)),
        "months": sorted(frame["month"].unique()),
        "fits": int(len(fit_audit)),
        "diagnostic_leader": (
            arm_summary.iloc[0].to_dict() if len(arm_summary) else None
        ),
        "sources": {
            "panel": {
                "path": str(args.panel.resolve()),
                "sha256": _sha256(args.panel),
            },
            "panel_manifest": {
                "path": str(args.panel_manifest.resolve()),
                "sha256": _sha256(args.panel_manifest),
            },
        },
        "outputs": {
            "oof_scores": {"path": str(score_path.resolve()), "sha256": _sha256(score_path)},
            "tail_metrics": {"path": str(tails_path.resolve()), "sha256": _sha256(tails_path)},
            "monthly_metrics": {"path": str(monthly_path.resolve()), "sha256": _sha256(monthly_path)},
            "arm_summary": {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)},
            "fit_audit": {"path": str(fit_path.resolve()), "sha256": _sha256(fit_path)},
            "matched_monthly_global_topk": {
                "path": str(reusable_monthly_path.resolve()),
                "sha256": _sha256(reusable_monthly_path),
            },
            "matched_arm_summary": {
                "path": str(reusable_summary_path.resolve()),
                "sha256": _sha256(reusable_summary_path),
            },
            "matched_arm_comparisons": {
                "path": str(comparison_path.resolve()),
                "sha256": _sha256(comparison_path),
            },
            "matched_decile_monotonicity": {
                "path": str(monotonicity_path.resolve()),
                "sha256": _sha256(monotonicity_path),
            },
            "matched_global_selections": {
                "path": str(selection_path.resolve()),
                "sha256": _sha256(selection_path),
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(args.output_dir / "manifest.json", _json_safe(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
