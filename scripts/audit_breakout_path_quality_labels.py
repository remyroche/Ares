#!/usr/bin/env python3
"""Audit breakout path-quality labels before fitting any predictive model.

All labels are constructed with cutoffs fitted on rows strictly before the
scored fold.  This is a diagnostic artifact: realized path and economic
outcomes are never exported as inference features.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mutual_info_score

from extreme_price_movements.breakout_path_quality_labels import (
    fit_breakout_path_quality_thresholds,
    materialize_breakout_path_quality_labels,
)
from scripts.validate_breakout_path_quality_labels import (
    OUTCOME_NAMES,
    REQUIRED as LABEL_REQUIRED,
    _derive_outcomes,
    _quarter_starts,
)


ECONOMIC_COLUMNS = [
    "__u_policy_net__",
    "__path_full_bad_mae_1r__",
    "__first_touch_net_positive__",
    "__first_touch_timeout__",
]
RAW_REQUIRED = [*LABEL_REQUIRED, *ECONOMIC_COLUMNS]
LABEL_COLUMNS = [
    "breakout_retention_failure",
    "breakout_low_efficiency",
    "breakout_participation_failure",
    "breakout_rapid_reversal",
]


def _load_values(labels_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(labels_dir.glob("train_global_*.parquet")):
        available = set(pq.read_schema(path).names)
        if not set(RAW_REQUIRED).issubset(available):
            continue
        raw = pd.read_parquet(path, columns=RAW_REQUIRED)
        outcome = _derive_outcomes(raw)
        output = outcome.copy()
        output["u_policy_net"] = pd.to_numeric(raw["__u_policy_net__"], errors="coerce")
        output["full_path_bad_mae"] = pd.to_numeric(
            raw["__path_full_bad_mae_1r__"], errors="coerce"
        )
        net_positive = pd.to_numeric(raw["__first_touch_net_positive__"], errors="coerce")
        timeout = pd.to_numeric(raw["__first_touch_timeout__"], errors="coerce")
        output["clean_exec"] = (
            net_positive.eq(1.0)
            & output["full_path_bad_mae"].eq(0.0)
            & timeout.eq(0.0)
        ).astype(np.int8)
        parts.append(output)
    if not parts:
        raise FileNotFoundError("No label partitions contain the required path-quality audit fields")
    return pd.concat(parts, ignore_index=True, copy=False)


def _phi(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.astype(np.float64), right.astype(np.float64)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _redundancy_rows(
    labels: pd.DataFrame,
    *,
    base: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = labels["breakout_quality_label_valid"].eq(1).to_numpy(bool)
    for left, right in combinations(LABEL_COLUMNS, 2):
        a = labels.loc[valid, left].to_numpy(np.int8)
        b = labels.loc[valid, right].to_numpy(np.int8)
        union = int(np.logical_or(a, b).sum())
        intersection = int(np.logical_and(a, b).sum())
        rows.append(
            {
                **base,
                "left_label": left,
                "right_label": right,
                "valid_rows": int(len(a)),
                "phi": _phi(a, b),
                "jaccard": float(intersection / union) if union else np.nan,
                "p_right_given_left": float(intersection / a.sum()) if a.sum() else np.nan,
                "p_left_given_right": float(intersection / b.sum()) if b.sum() else np.nan,
                "mutual_information": float(mutual_info_score(a, b)) if len(a) else np.nan,
            }
        )
    return rows


def _economic_rows(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    base: dict[str, object],
) -> list[dict[str, object]]:
    joined = pd.concat([values.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    joined = joined.loc[joined["breakout_quality_label_valid"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    metrics = {
        "mean_ev": "u_policy_net",
        "clean_precision": "clean_exec",
        "bad_mae_rate": "full_path_bad_mae",
        "retention_rate": "breakout_retention_outcome",
        "mean_reversal": "breakout_reversal_magnitude_outcome",
    }
    for label in LABEL_COLUMNS:
        grouped = joined.groupby(label, observed=True)
        negative = grouped.get_group(0) if 0 in grouped.groups else joined.iloc[:0]
        positive = grouped.get_group(1) if 1 in grouped.groups else joined.iloc[:0]
        record: dict[str, object] = {
            **base,
            "label": label,
            "valid_rows": int(len(joined)),
            "positive_rows": int(len(positive)),
            "positive_rate": float(len(positive) / len(joined)) if len(joined) else np.nan,
        }
        for name, column in metrics.items():
            positive_value = float(pd.to_numeric(positive[column], errors="coerce").mean())
            negative_value = float(pd.to_numeric(negative[column], errors="coerce").mean())
            record[f"positive_{name}"] = positive_value
            record[f"negative_{name}"] = negative_value
            record[f"delta_positive_minus_negative_{name}"] = positive_value - negative_value
        rate = record["positive_rate"]
        record["base_rate_status"] = (
            "candidate_binary" if 0.10 <= rate <= 0.50 else
            "too_prevalent_use_continuous_or_tail" if rate > 0.75 else
            "sparse_or_borderline"
        )
        rows.append(record)
    return rows


def _stability(economic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, local in economic.groupby(["side_name", "archetype_policy_key", "label"], observed=True):
        prevalence = pd.to_numeric(local["positive_rate"], errors="coerce")
        ev_delta = pd.to_numeric(local["delta_positive_minus_negative_mean_ev"], errors="coerce")
        rows.append(
            {
                "side_name": keys[0],
                "archetype_policy_key": keys[1],
                "label": keys[2],
                "folds": int(len(local)),
                "positive_rate_min": float(prevalence.min()),
                "positive_rate_max": float(prevalence.max()),
                "positive_rate_range": float(prevalence.max() - prevalence.min()),
                "ev_delta_min": float(ev_delta.min()),
                "ev_delta_max": float(ev_delta.max()),
                "ev_delta_range": float(ev_delta.max() - ev_delta.min()),
                "same_ev_direction_all_folds": bool((ev_delta.ge(0.0)).all() or (ev_delta.le(0.0)).all()),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    values = _load_values(args.labels_dir)
    start, end = pd.Timestamp(args.eval_start, tz="UTC"), pd.Timestamp(args.eval_end, tz="UTC")
    values = values.loc[values["__ts__"].lt(end)].copy()
    redundancy: list[dict[str, object]] = []
    economic: list[dict[str, object]] = []
    for side, archetype in values.loc[:, ["side_name", "archetype_policy_key"]].drop_duplicates().itertuples(index=False, name=None):
        local = values.loc[
            values["side_name"].eq(side) & values["archetype_policy_key"].eq(archetype)
        ].sort_values("__ts__", kind="stable")
        for fold_start in _quarter_starts(start, end):
            fold_end = min(fold_start + pd.DateOffset(months=3), end)
            train = local.loc[local["__ts__"].lt(fold_start)]
            scored = local.loc[local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)]
            if len(train) < args.minimum_train_rows or len(scored) < args.minimum_eval_rows:
                continue
            thresholds = fit_breakout_path_quality_thresholds(train.loc[:, OUTCOME_NAMES])
            labels = materialize_breakout_path_quality_labels(scored.loc[:, OUTCOME_NAMES], thresholds)
            base = {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "side_name": side,
                "archetype_policy_key": archetype,
                "train_rows": int(len(train)),
                "eval_rows": int(len(scored)),
            }
            redundancy.extend(_redundancy_rows(labels, base=base))
            economic.extend(_economic_rows(scored, labels, base=base))
    redundancy_frame = pd.DataFrame(redundancy)
    economic_frame = pd.DataFrame(economic)
    stability = _stability(economic_frame)
    redundancy_frame.to_csv(args.output / "label_redundancy_by_fold.csv", index=False)
    economic_frame.to_csv(args.output / "label_economic_ordering_by_fold.csv", index=False)
    stability.to_csv(args.output / "label_temporal_stability.csv", index=False)
    focus = economic_frame.loc[economic_frame["archetype_policy_key"].isin([
        "short_breakout_precision", "long_breakout_diagnostic_candidate"
    ])]
    manifest = {
        "schema": "breakout_path_quality_label_audit_v1",
        "status": "pre_model_label_quality_audit_complete",
        "rows": int(len(values)),
        "fold_label_rows": int(len(economic_frame)),
        "focus_label_rows": int(len(focus)),
        "target_suitability": {
            "candidate_binary_rate_range": [0.10, 0.50],
            "too_prevalent_rate": 0.75,
            "redundancy_warning": "Inspect phi/Jaccard/conditional overlap before retaining both labels.",
        },
        "leakage_contract": "All cutoffs are fit on pre-fold rows. Economic/path columns are diagnostics and never inference features.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-start", default="2025-07-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--minimum-train-rows", type=int, default=500)
    parser.add_argument("--minimum-eval-rows", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
