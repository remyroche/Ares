#!/usr/bin/env python3
"""Benchmark a small causal CNN for early warning of adverse event blocks.

This is a period-level research model.  It consumes only the causal daily-open
observable state for one side x archetype and predicts whether a difficult
calendar block begins in the next ``--horizon-days``.  It is deliberately not
fed realised residuals, recent hit rate, model scores, or policy outcomes.

The benchmark fits a shallow multi-scale 1-D CNN and a compact LightGBM lag
summary baseline on the exact same samples and chronological folds.  Models
are local to each side x archetype; no global average is allowed to hide a
local failure mechanism.  Output is research-only and never changes live
policy or meta scores.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from extreme_price_movements.residual_event_block_taxonomy import (
    MECHANISM_FAMILIES,
    attach_event_blocks,
)
from scripts.report_residual_event_block_taxonomy import _load_daily_state, _load_calendar, _overlay_event_calendar


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


FOLDS = (
    (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
    (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
    (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")),
)
KEYS = ["day", "side_name", "archetype_policy_key"]


@dataclass(frozen=True)
class SequenceBundle:
    days: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    event_start: np.ndarray


class SmallCausalCNN(nn.Module):
    """Two shallow causal branches; receptive field stays below 16 days."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.short = nn.Sequential(
            nn.Conv1d(feature_count, 12, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(12, 8, kernel_size=3, dilation=2, padding=4),
            nn.ReLU(),
        )
        self.medium = nn.Sequential(
            nn.Conv1d(feature_count, 8, kernel_size=7, padding=6),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Dropout(0.10), nn.Linear(12, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Right-most position is the current daily-open state.  Left padding
        # ensures no branch accesses a later day.
        short = self.short(x)[..., -1]
        medium = self.medium(x)[..., -1]
        return self.head(torch.cat([short, medium], dim=1)).squeeze(1)


class SmallCausalTCN(nn.Module):
    """Bounded 31-day receptive-field TCN for gradual state transitions."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(feature_count, 8, kernel_size=3, dilation=1),
                nn.Conv1d(8, 8, kernel_size=3, dilation=2),
                nn.Conv1d(8, 8, kernel_size=3, dilation=4),
                nn.Conv1d(8, 8, kernel_size=3, dilation=8),
            ]
        )
        self.head = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Dropout(0.10), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # Causal left-only padding.  The final representation cannot use
            # a later daily state even though the model has a long context.
            x = torch.relu(layer(torch.nn.functional.pad(x, (2 * layer.dilation[0], 0))))
        return self.head(x[..., -1]).squeeze(1)


def _parse_group(value: str) -> tuple[str, str]:
    side, separator, archetype = value.partition("::")
    if not separator or not side or not archetype:
        raise ValueError(f"Expected side::archetype, got {value!r}")
    return side, archetype


def _candidate_features(daily: pd.DataFrame) -> list[str]:
    requested = list(dict.fromkeys(name for values in MECHANISM_FAMILIES.values() for name in values))
    return [name for name in requested if name in daily.columns]


def _event_starts(calendar: pd.DataFrame) -> pd.DataFrame:
    blocks = attach_event_blocks(calendar)
    blocks["event_start"] = False
    event = blocks["event_block"].ne("normal")
    earliest = blocks.loc[event].groupby(
        ["side_name", "archetype_policy_key", "event_block"], observed=True
    )["day"].transform("min")
    blocks.loc[event, "event_start"] = blocks.loc[event, "day"].eq(earliest)
    return blocks.loc[:, [*KEYS, "adverse_event", "event_start"]]


def _screen(train: pd.DataFrame, features: list[str], target: np.ndarray, maximum: int) -> list[str]:
    """Cheap train-only robust screen; it replaces a costly row-level MDA."""

    selected: list[tuple[str, float]] = []
    positive = target.astype(bool)
    if positive.sum() < 2 or (~positive).sum() < 2:
        return []
    for name in features:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(np.float32, copy=False)
        finite = np.isfinite(values)
        if finite.mean() < 0.80 or not finite[positive].any() or not finite[~positive].any():
            continue
        q25, q75 = np.nanquantile(values[finite], [0.25, 0.75])
        separation = abs(float(np.nanmedian(values[positive]) - np.nanmedian(values[~positive]))) / max(float(q75 - q25), 1e-4)
        selected.append((name, separation))
    return [name for name, _ in sorted(selected, key=lambda item: item[1], reverse=True)[:maximum]]


def _fill_scale(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    q25 = np.nanquantile(train, 0.25, axis=0)
    q75 = np.nanquantile(train, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    median = np.nan_to_num(median, nan=0.0).astype(np.float32)
    for values in (train, score):
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(median, np.nonzero(missing)[1])
        values -= median
        values /= scale
        np.clip(values, -6.0, 6.0, out=values)
    return train.astype(np.float32, copy=False), score.astype(np.float32, copy=False)


def _sequence_bundle(local: pd.DataFrame, features: list[str], window: int, horizon: int) -> SequenceBundle:
    local = local.sort_values("day", kind="stable").reset_index(drop=True)
    values = local[features].to_numpy(np.float32, copy=True)
    starts = local["event_start"].to_numpy(bool)
    target = np.zeros(len(local), dtype=np.int8)
    for offset in range(1, horizon + 1):
        target[:-offset] |= starts[offset:]
    x = np.zeros((len(local), len(features), window), dtype=np.float32)
    for index in range(len(local)):
        left = max(0, index - window + 1)
        segment = values[left:index + 1]
        x[index, :, -len(segment):] = segment.T
    return SequenceBundle(
        days=pd.DatetimeIndex(local["day"]), x=x, y=target, event_start=starts,
    )


def _subsample_indices(y: np.ndarray, *, maximum_negative_ratio: int, seed: int) -> np.ndarray:
    positive = np.flatnonzero(y > 0)
    negative = np.flatnonzero(y == 0)
    if not len(positive):
        return np.array([], dtype=np.int64)
    cap = min(len(negative), maximum_negative_ratio * len(positive))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(negative, size=cap, replace=False) if cap < len(negative) else negative
    return np.sort(np.concatenate([positive, chosen]))


def _cnn_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_score: np.ndarray,
    *,
    seed: int,
    epochs: int,
    architecture: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    indices = _subsample_indices(y_train, maximum_negative_ratio=4, seed=seed)
    if len(indices) < 8 or y_train[indices].sum() < 2:
        return np.full(len(x_score), np.nan, dtype=np.float32)
    x = torch.from_numpy(x_train[indices])
    y = torch.from_numpy(y_train[indices].astype(np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=min(32, len(x)), shuffle=True, num_workers=0)
    model = SmallCausalCNN(x_train.shape[1]) if architecture == "cnn" else SmallCausalTCN(x_train.shape[1])
    positives = max(int(y_train[indices].sum()), 1)
    negatives = max(len(indices) - positives, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        result = torch.sigmoid(model(torch.from_numpy(x_score))).cpu().numpy().astype(np.float32)
    # The full seven-archetype sweep creates many small local models.  Release
    # each immediately instead of retaining allocator state across folds.
    del model, optimizer, loader, x, y
    gc.collect()
    return result


def _summary_matrix(x: np.ndarray) -> np.ndarray:
    """Current level, eight-day mean, and 8d-minus-24d trend baseline."""

    current = x[:, :, -1]
    short = x[:, :, -8:].mean(axis=2)
    long = x.mean(axis=2)
    return np.concatenate([current, short, short - long], axis=1).astype(np.float32, copy=False)


def _lgbm_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_score: np.ndarray, *, seed: int) -> np.ndarray:
    indices = _subsample_indices(y_train, maximum_negative_ratio=4, seed=seed)
    if len(indices) < 8 or y_train[indices].sum() < 2:
        return np.full(len(x_score), np.nan, dtype=np.float32)
    train = _summary_matrix(x_train)[indices]
    score = _summary_matrix(x_score)
    positives = max(int(y_train[indices].sum()), 1)
    negatives = max(len(indices) - positives, 1)
    # Use the native interface.  The installed sklearn wrapper is not
    # compatible with this environment's newer sklearn validation signature.
    # Native LightGBM also avoids an unnecessary dataframe/scikit conversion.
    model = lgb.train(
        {
            "objective": "binary", "metric": "None", "learning_rate": 0.035,
            "max_depth": 2, "num_leaves": 4,
            "min_data_in_leaf": max(4, min(12, len(indices) // 8)),
            "lambda_l1": 2.0, "lambda_l2": 12.0, "feature_fraction": 0.85,
            "bagging_fraction": 0.90, "bagging_freq": 1, "seed": seed,
            "num_threads": 1, "verbosity": -1, "force_col_wise": True,
        },
        lgb.Dataset(
            train,
            label=y_train[indices],
            weight=np.where(y_train[indices] > 0, negatives / positives, 1.0),
        ),
        num_boost_round=80,
    )
    return np.asarray(model.predict(score), dtype=np.float32)


def _metrics(frame: pd.DataFrame, fraction: float) -> dict[str, float]:
    count = max(1, int(np.ceil(len(frame) * fraction)))
    chosen = frame["risk"].rank(method="first", ascending=False).le(count).to_numpy(bool)
    y = frame["target"].to_numpy(bool)
    precision = float(y[chosen].mean()) if chosen.any() else np.nan
    prevalence = float(y.mean())
    return {
        "selected_days": int(chosen.sum()),
        "precision": precision,
        "fpr": float(chosen[~y].mean()) if (~y).any() else np.nan,
        "lift": precision / max(prevalence, 1e-9) if np.isfinite(precision) else np.nan,
        "event_recall": float((chosen & y).sum() / max(y.sum(), 1)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(name for values in MECHANISM_FAMILIES.values() for name in values))
    daily = _load_daily_state(args.state_artifact, requested)
    calendar = daily.loc[:, KEYS].copy()
    calendar = _overlay_event_calendar(calendar, args.event_calendar)
    events = _event_starts(calendar)
    daily = daily.merge(events, on=KEYS, how="left", validate="one_to_one")
    daily["event_start"] = daily["event_start"].fillna(False).astype(bool)
    groups = [_parse_group(value) for value in args.group]
    feature_candidates = _candidate_features(daily)
    reports: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold_index, (train_end, eval_end) in enumerate(FOLDS):
        for side, archetype in groups:
            local = daily.loc[
                daily["side_name"].eq(side) & daily["archetype_policy_key"].eq(archetype)
            ].sort_values("day", kind="stable")
            train = local.loc[local["day"].lt(train_end)].copy()
            score = local.loc[local["day"].ge(train_end) & local["day"].lt(eval_end)].copy()
            if len(train) < args.window_days + 12 or len(score) < 8:
                continue
            # Target is event onset in the *next* horizon.  It never uses a
            # current or historical model outcome as an input feature.
            starts = train["event_start"].to_numpy(bool)
            train_target = np.zeros(len(train), dtype=np.int8)
            for offset in range(1, args.horizon_days + 1):
                train_target[:-offset] |= starts[offset:]
            features = _screen(train, feature_candidates, train_target, args.max_features)
            if len(features) < 3 or train_target.sum() < args.min_positive_days:
                reports.append({
                    "fold_start": train_end, "fold_end": eval_end, "side_name": side,
                    "archetype_policy_key": archetype, "status": "insufficient_train_support",
                    "train_positive_days": int(train_target.sum()), "features": "|".join(features),
                })
                continue
            combined = pd.concat([train, score], ignore_index=True, copy=False)
            train_values = combined.loc[: len(train) - 1, features].to_numpy(np.float32, copy=True)
            score_values = combined.loc[len(train):, features].to_numpy(np.float32, copy=True)
            train_values, score_values = _fill_scale(train_values, score_values)
            combined.loc[:, features] = np.vstack([train_values, score_values])
            bundle = _sequence_bundle(combined, features, args.window_days, args.horizon_days)
            split = len(train)
            eval_days = bundle.days[split:]
            target = bundle.y[split:]
            valid = np.arange(len(target) - args.horizon_days)
            if not len(valid):
                continue
            # Labels for the last ``horizon`` train days would look into the
            # evaluation period.  Purge them from model fitting entirely.
            train_fit_end = split - args.horizon_days
            if train_fit_end < args.window_days + 4:
                continue
            for model_name, fit in (
                ("causal_cnn", _cnn_fit_predict),
                ("causal_tcn", _cnn_fit_predict),
                ("lag_summary_lgbm", _lgbm_fit_predict),
            ):
                if model_name in {"causal_cnn", "causal_tcn"}:
                    risk = fit(
                        bundle.x[:train_fit_end], bundle.y[:train_fit_end], bundle.x[split:],
                        seed=args.seed + fold_index, epochs=args.epochs,
                        architecture="cnn" if model_name == "causal_cnn" else "tcn",
                    )
                else:
                    risk = fit(bundle.x[:train_fit_end], bundle.y[:train_fit_end], bundle.x[split:], seed=args.seed + fold_index)
                frame = pd.DataFrame({"day": eval_days[valid], "risk": risk[valid], "target": target[valid].astype(bool)})
                frame = frame.loc[np.isfinite(frame["risk"])].copy()
                if frame.empty:
                    continue
                values: dict[str, object] = {
                    "fold_start": train_end, "fold_end": eval_end, "side_name": side,
                    "archetype_policy_key": archetype, "model": model_name,
                    "status": "ok", "train_positive_days": int(train_target.sum()),
                    "features": "|".join(features), "window_days": args.window_days,
                    "horizon_days": args.horizon_days,
                }
                for fraction in (0.01, 0.03, 0.05, 0.10):
                    suffix = f"top{int(fraction * 100):02d}"
                    values.update({f"{suffix}_{key}": value for key, value in _metrics(frame, fraction).items()})
                reports.append(values)
                predictions.append(frame.assign(
                    fold_start=train_end, fold_end=eval_end, side_name=side,
                    archetype_policy_key=archetype, model=model_name,
                ))
    report = pd.DataFrame(reports)
    report.to_csv(args.output / "hard_period_cnn_oof_metrics.csv", index=False)
    valid = report.loc[report["status"].eq("ok")].copy() if not report.empty else report
    if not valid.empty:
        aggregations: dict[str, tuple[str, object]] = {"folds": ("fold_start", "size")}
        for suffix in ("top01", "top03", "top05", "top10"):
            for metric in ("lift", "fpr", "event_recall", "precision"):
                aggregations[f"{suffix}_mean_{metric}"] = (f"{suffix}_{metric}", "mean")
            aggregations[f"{suffix}_hit_folds"] = (f"{suffix}_event_recall", lambda values: int((values > 0).sum()))
        summary = valid.groupby(["model", "side_name", "archetype_policy_key"], observed=True, as_index=False).agg(**aggregations)
        summary["passes_top05_repetition_gate"] = (
            summary["folds"].ge(3) & summary["top05_hit_folds"].ge(3)
            & summary["top05_mean_lift"].ge(1.5) & summary["top05_mean_fpr"].le(0.15)
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(args.output / "hard_period_cnn_oof_summary.csv", index=False)
    oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    oof.to_parquet(args.output / "hard_period_cnn_oof_daily_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "purpose": "research-only causal hard-period early-warning benchmark; not an overlay or policy gate",
        "state_contract": "daily-open observable price/OI/funding/cross-sectional state only; no realised residuals, scores, or recent performance features",
        "target_contract": f"adverse event onset in next {args.horizon_days} daily snapshots",
        "folds": [(str(start), str(end)) for start, end in FOLDS],
        "window_days": args.window_days,
        "max_features": args.max_features,
        "negative_sampling": "at most 4 negative windows per positive during fitting",
        "models": ["causal_cnn", "causal_tcn", "lag_summary_lgbm"],
        "rows": int(len(oof)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-calendar", type=Path, action="append", required=True)
    parser.add_argument("--state-artifact", type=Path, action="append", required=True)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=32)
    parser.add_argument("--horizon-days", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--min-positive-days", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
