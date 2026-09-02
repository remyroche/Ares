#!/usr/bin/env python3
"""Search auditable gated residual composites across the joined event calendar."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
)


EPS = 1e-8


@dataclass(frozen=True)
class Fold:
    name: str
    train: np.ndarray
    valid: np.ndarray


def _daily_features(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    composites = set(NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS)
    parts = []
    for feature in NEGATIVE_RESIDUAL_META_FEATURE_KEYS:
        grouped = frame[feature].groupby(frame.index.floor("D"))
        parts.append((grouped.max() if feature in composites else grouped.mean()).rename(feature))
    return pd.concat(parts, axis=1).astype(np.float32)


def _robust_scale(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = train[np.isfinite(train)]
    if finite.size < 20:
        return np.full_like(values, np.nan, dtype=np.float32)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, float(np.std(finite)) * 0.10, EPS)
    return np.clip((values - median) / scale, -5.0, 5.0).astype(np.float32)


def _composite(a: np.ndarray, b: np.ndarray, form: str, gate_q: float) -> np.ndarray:
    ap = np.maximum(a, 0.0)
    if form == "positive":
        raw = ap * np.maximum(b, 0.0)
    elif form == "negative":
        raw = ap * np.maximum(-b, 0.0)
    elif form == "threshold":
        raw = ap * (b > gate_q)
    elif form == "contrast":
        raw = ap * np.maximum(b - a, 0.0)
    else:
        raise ValueError(form)
    return np.arcsinh(raw).astype(np.float32)


def _metrics(score: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    valid = np.isfinite(score)
    score = score[valid]
    target = target[valid].astype(bool)
    if not len(score):
        return {key: np.nan for key in ("precision", "lift", "recall", "fpr", "auc")}
    selected = score >= threshold
    prevalence = float(target.mean())
    precision = float(target[selected].mean()) if selected.any() else 0.0
    positives = score[target]
    negatives = score[~target]
    if len(positives) and len(negatives):
        ranks = pd.Series(np.r_[positives, negatives]).rank(method="average").to_numpy()
        auc = float((ranks[: len(positives)].sum() - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))
    else:
        auc = np.nan
    return {
        "precision": precision,
        "lift": precision / max(prevalence, EPS),
        "recall": float((selected & target).sum() / max(target.sum(), 1)),
        "fpr": float((selected & ~target).sum() / max((~target).sum(), 1)),
        "auc": auc,
    }


def _folds(index: pd.DatetimeIndex) -> list[Fold]:
    # Expanding quarterly validation; every threshold and transform is train-only.
    periods = pd.period_range("2025Q2", "2026Q2", freq="Q")
    output = []
    for period in periods:
        start = pd.Timestamp(period.start_time, tz="UTC")
        stop = pd.Timestamp(period.end_time + pd.Timedelta(days=1), tz="UTC")
        train = np.asarray(index < start)
        valid = np.asarray((index >= start) & (index < stop))
        if train.sum() >= 60 and valid.sum() >= 20:
            output.append(Fold(str(period), train, valid))
    return output


def _screen_gate(a: np.ndarray, b: np.ndarray, y: np.ndarray, broad: np.ndarray) -> tuple[float, float]:
    mask = broad & np.isfinite(b)
    adverse = b[mask & y]
    benign = b[mask & ~y]
    if len(adverse) < 2 or len(benign) < 10:
        return np.nan, np.nan
    pooled = b[mask]
    scale = max(float(np.median(np.abs(pooled - np.median(pooled)))), EPS)
    effect = float((np.median(adverse) - np.median(benign)) / scale)
    return effect, float(np.quantile(b[mask], 0.75))


def _deduplicate_local_frontier(
    local: pd.DataFrame,
    values: np.ndarray,
    feature_names: list[str],
    target: np.ndarray,
    max_features: int,
) -> pd.DataFrame:
    """Greedily retain candidates whose realized score correlation is <= 0.90."""
    if local.empty:
        return local
    train = np.arange(len(values)) < max(int(len(values) * 0.60), 60)
    retained_rows: list[pd.Series] = []
    retained_scores: list[np.ndarray] = []
    for _, definition in local.sort_values("search_score", ascending=False).iterrows():
        ai = feature_names.index(definition["base_feature"])
        bi = feature_names.index(definition["gate_feature"])
        az = _robust_scale(values[train, ai], values[:, ai])
        bz = _robust_scale(values[train, bi], values[:, bi])
        broad = az[train] >= np.nanquantile(az[train], 0.80)
        effect, gate_q = _screen_gate(az[train], bz[train], target[train], broad)
        if not np.isfinite(effect):
            continue
        if effect < 0:
            bz, gate_q = -bz, -gate_q
        score = _composite(az, bz, definition["form"], gate_q)
        if any(
            abs(pd.Series(score).corr(pd.Series(existing), method="spearman")) > 0.90
            for existing in retained_scores
        ):
            continue
        retained_rows.append(definition)
        retained_scores.append(score)
        if len(retained_rows) >= max_features:
            break
    return pd.DataFrame(retained_rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    daily = _daily_features(args.feature_file, args.start, args.end)
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.drop_duplicates(["day", "side_name", "archetype_policy_key"])
    folds = _folds(daily.index)
    feature_names = list(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    values = daily.to_numpy(dtype=np.float32, copy=False)
    candidate_rows: list[dict[str, object]] = []

    for (side, archetype), events in calendar.groupby(["side_name", "archetype_policy_key"], observed=True):
        y = np.asarray(daily.index.isin(set(events["day"])), dtype=bool)
        for ai, a_name in enumerate(feature_names):
            for bi, b_name in enumerate(feature_names):
                if ai == bi:
                    continue
                fold_rows = []
                for fold in folds:
                    if y[fold.train].sum() < 3 or y[fold.valid].sum() < 1:
                        continue
                    az = _robust_scale(values[fold.train, ai], values[:, ai])
                    bz = _robust_scale(values[fold.train, bi], values[:, bi])
                    broad_q = float(np.nanquantile(az[fold.train], 0.80))
                    broad = az >= broad_q
                    effect, gate_q = _screen_gate(az[fold.train], bz[fold.train], y[fold.train], broad[fold.train])
                    if not np.isfinite(effect) or abs(effect) < args.min_effect:
                        continue
                    oriented_b = bz if effect >= 0 else -bz
                    oriented_gate = gate_q if effect >= 0 else -gate_q
                    base_threshold = float(np.nanquantile(az[fold.train], 0.90))
                    base_valid = _metrics(az[fold.valid], y[fold.valid], base_threshold)
                    for form in ("positive", "negative", "threshold", "contrast"):
                        score = _composite(az, oriented_b, form, oriented_gate)
                        threshold = float(np.nanquantile(score[fold.train], 0.90))
                        valid_metrics = _metrics(score[fold.valid], y[fold.valid], threshold)
                        fold_rows.append(
                            {
                                "fold": fold.name,
                                "form": form,
                                "effect": effect,
                                **valid_metrics,
                                "precision_gain": valid_metrics["precision"] - base_valid["precision"],
                                "auc_gain": valid_metrics["auc"] - base_valid["auc"],
                                "event_hits": int(((score[fold.valid] >= threshold) & y[fold.valid]).sum()),
                            }
                        )
                if not fold_rows:
                    continue
                fold_frame = pd.DataFrame(fold_rows)
                for form, local in fold_frame.groupby("form", observed=True):
                    positive_folds = int(((local["lift"] >= 1.0) & (local["precision_gain"] > 0)).sum())
                    lift = float(local["lift"].mean())
                    fpr = float(local["fpr"].mean())
                    auc_gain = float(local["auc_gain"].mean())
                    precision_gain = float(local["precision_gain"].mean())
                    instability = float(local["lift"].std(ddof=0))
                    score = float(np.log(max(lift, EPS)) - 2.0 * fpr + 0.5 * auc_gain - 0.5 * instability)
                    candidate_rows.append(
                        {
                            "side_name": side,
                            "archetype_policy_key": archetype,
                            "base_feature": a_name,
                            "gate_feature": b_name,
                            "form": form,
                            "mean_lift": lift,
                            "mean_fpr": fpr,
                            "mean_precision_gain": precision_gain,
                            "mean_auc_gain": auc_gain,
                            "fold_lift_std": instability,
                            "positive_folds": positive_folds,
                            "evaluated_folds": int(len(local)),
                            "adverse_support": int(local["event_hits"].sum()),
                            "search_score": score,
                        }
                    )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise RuntimeError("No gated candidates had sufficient train-fold support")
    candidates["promoted"] = (
        (candidates["mean_lift"] >= args.min_lift)
        & (candidates["mean_fpr"] <= args.max_fpr)
        & (candidates["mean_precision_gain"] > 0)
        & (candidates["positive_folds"] >= args.min_positive_folds)
        & (candidates["adverse_support"] >= args.min_support)
    )
    candidates = candidates.sort_values("search_score", ascending=False, kind="stable")
    candidates.to_csv(args.output / "gated_composite_candidates.csv", index=False)

    # Joined accounting: retain a small non-redundant local frontier, then ask
    # whether any promoted composite recognizes each of the 393 calendar cells.
    promoted_parts = []
    for (side, archetype), local in candidates.loc[candidates["promoted"]].groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        target = np.asarray(
            daily.index.isin(
                set(
                    calendar.loc[
                        (calendar["side_name"] == side)
                        & (calendar["archetype_policy_key"] == archetype),
                        "day",
                    ]
                )
            ),
            dtype=bool,
        )
        promoted_parts.append(
            _deduplicate_local_frontier(
                local,
                values,
                feature_names,
                target,
                args.max_per_archetype,
            )
        )
    promoted = pd.concat(promoted_parts, ignore_index=True) if promoted_parts else candidates.iloc[0:0]
    promoted.to_csv(args.output / "promoted_gated_composites.csv", index=False)
    recognition_rows = []
    for _, cell in calendar.iterrows():
        local = promoted.loc[
            (promoted["side_name"] == cell["side_name"])
            & (promoted["archetype_policy_key"] == cell["archetype_policy_key"])
        ]
        hits = []
        day = cell["day"]
        if day in daily.index:
            pos = int(daily.index.get_loc(day))
            history = np.arange(len(daily)) < pos
            for _, definition in local.iterrows():
                ai = feature_names.index(definition["base_feature"])
                bi = feature_names.index(definition["gate_feature"])
                az = _robust_scale(values[history, ai], values[:, ai])
                bz = _robust_scale(values[history, bi], values[:, bi])
                broad = az[history] >= np.nanquantile(az[history], 0.80)
                event_days = set(calendar.loc[
                    (calendar["side_name"] == cell["side_name"])
                    & (calendar["archetype_policy_key"] == cell["archetype_policy_key"]), "day"
                ])
                y_hist = np.asarray(daily.index[history].isin(event_days), dtype=bool)
                effect, gate_q = _screen_gate(az[history], bz[history], y_hist, broad)
                if not np.isfinite(effect):
                    continue
                if effect < 0:
                    bz, gate_q = -bz, -gate_q
                score = _composite(az, bz, definition["form"], gate_q)
                threshold = float(np.nanquantile(score[history], 0.90))
                if np.isfinite(score[pos]) and score[pos] >= threshold:
                    hits.append(f"{definition['base_feature']}::{definition['form']}::{definition['gate_feature']}")
        recognition_rows.append({**cell.to_dict(), "joined_recognized": bool(hits), "joined_hits": "|".join(hits)})
    recognition = pd.DataFrame(recognition_rows)
    recognition.to_csv(args.output / "calendar_joined_recognition.csv", index=False)
    manifest = {
        "schema": "joined_gated_residual_composite_search_v1",
        "calendar_cells": int(len(recognition)),
        "calendar_days": int(recognition["day"].nunique()),
        "candidate_definitions": int(len(candidates)),
        "promoted_definitions": int(len(promoted)),
        "joined_recognized_cells": int(recognition["joined_recognized"].sum()),
        "joined_coverage": float(recognition["joined_recognized"].mean()),
        "folds": [fold.name for fold in folds],
        "promotion": {
            "min_lift": args.min_lift,
            "max_fpr": args.max_fpr,
            "min_positive_folds": args.min_positive_folds,
            "min_support": args.min_support,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/residual_calendar_feature_matches_20260712_v1/calendar_cells_with_feature_matches.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/joined_gated_residual_composites_20260712_v1"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--min-effect", type=float, default=0.50)
    parser.add_argument("--min-lift", type=float, default=1.50)
    parser.add_argument("--max-fpr", type=float, default=0.15)
    parser.add_argument("--min-positive-folds", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--max-per-archetype", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
