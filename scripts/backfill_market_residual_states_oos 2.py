#!/usr/bin/env python3
"""Append leakage-safe market residual states to existing local OOS predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.market_residual_archetypes import (  # noqa: E402
    MarketResidualConfig,
    MarketResidualStateRecognizer,
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    inference_feature_columns,
    strip_outcomes_for_oos,
)
from scripts.run_meta_residual_archetype_discovery import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_LEDGER,
    _folds_for_data,
    _resolve_score,
)

DEFAULT_LOCAL = Path(
    "data_perp/reports/meta_residual_archetype_discovery_early2025_july_20260712_v1/"
    "oos_residual_state_predictions.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/meta_residual_archetype_discovery_early2025_july_20260712_v1/"
    "oos_residual_state_predictions_with_market_v2.parquet"
)
KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_source(data_path: Path, ledger_path: Path) -> pd.DataFrame:
    data = pd.read_parquet(data_path)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    if ledger_path.exists() and "row_id" in data.columns:
        available = set(pq.ParquetFile(ledger_path).schema_arrow.names)
        columns = [
            name
            for name in (
                "row_id",
                "score",
                "score_regime_calibrated",
                "hit_probability",
            )
            if name in available
        ]
        ledger = pd.read_parquet(ledger_path, columns=columns)
        ledger = ledger.rename(columns={"score": "score_meta_uncalibrated"})
        ledger = ledger.drop_duplicates("row_id", keep="last")
        replacement = [
            name for name in ledger.columns if name != "row_id" and name in data.columns
        ]
        data = data.drop(columns=replacement).merge(
            ledger, on="row_id", how="left", validate="one_to_one", sort=False
        )
    return data


def _align_generated(
    local_fold: pd.DataFrame, valid: pd.DataFrame, generated: pd.DataFrame
) -> pd.DataFrame:
    left = local_fold.reset_index(drop=True)
    right_keys = valid.loc[:, list(KEYS)].reset_index(drop=True)
    right = pd.concat([right_keys, generated.reset_index(drop=True)], axis=1)
    if len(left) == len(right):
        same = True
        for key in KEYS:
            lhs = left[key]
            rhs = right[key]
            if key == "__ts__":
                lhs = pd.to_datetime(lhs, utc=True, errors="coerce")
                rhs = pd.to_datetime(rhs, utc=True, errors="coerce")
            else:
                lhs = lhs.astype(str)
                rhs = rhs.astype(str)
            same = same and bool(lhs.reset_index(drop=True).equals(rhs.reset_index(drop=True)))
        if same:
            additions = generated.reset_index(drop=True)
            return pd.concat([left, additions], axis=1, copy=False)

    left = left.copy()
    right = right.copy()
    left["__occurrence"] = left.groupby(list(KEYS), observed=True).cumcount()
    right["__occurrence"] = right.groupby(list(KEYS), observed=True).cumcount()
    output = left.merge(
        right,
        on=[*KEYS, "__occurrence"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__market_source"),
        sort=False,
    ).drop(columns="__occurrence")
    market_columns = [name for name in generated.columns if name.startswith("meta_resid_market_")]
    if output[market_columns].notna().sum().sum() == 0:
        raise RuntimeError("Market-state OOS alignment produced no matched features")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--local-predictions", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_source(args.data, args.ledger)
    score_col = _resolve_score(data, args.score_column)
    data = (
        data.loc[pd.to_numeric(data[score_col], errors="coerce").notna()]
        .sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
        .reset_index(drop=True)
    )
    candidates = inference_feature_columns(data, data.columns)
    local = pd.read_parquet(args.local_predictions)
    local["__ts__"] = pd.to_datetime(local["__ts__"], utc=True, errors="coerce")
    output_folds: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for fold_index, (fold, valid_start, valid_end) in enumerate(_folds_for_data(data)):
        train = data.loc[data["__ts__"].lt(valid_start)]
        valid = data.loc[
            data["__ts__"].ge(valid_start) & data["__ts__"].lt(valid_end)
        ]
        local_fold = local.loc[local["fold"].astype(str).eq(str(fold))]
        if len(valid) < 100 or local_fold.empty:
            continue
        recognizer = MarketResidualStateRecognizer(
            MarketResidualConfig(
                score_col=score_col,
                random_state=20260712 + fold_index * 101,
            ),
            candidates,
        ).fit(train)
        generated = recognizer.transform_oos(strip_outcomes_for_oos(valid))
        aligned = _align_generated(local_fold, valid, generated)
        output_folds.append(aligned)
        manifests.append(
            {
                "fold": fold,
                "train_end": str(train["__ts__"].max()),
                "valid_start": str(valid_start),
                "valid_end_exclusive": str(valid_end),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "market": recognizer.manifest(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "market_residual_fold_complete",
                    "fold": fold,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                }
            ),
            flush=True,
        )
    if not output_folds:
        raise RuntimeError("No market residual folds were materialized")
    output = pd.concat(output_folds, ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": "market_residual_oos_backfill_v2",
        "source_data": str(args.data.resolve()),
        "source_local_predictions": str(args.local_predictions.resolve()),
        "output": str(args.output.resolve()),
        "rows": int(len(output)),
        "market_features": [
            name for name in output.columns if name.startswith("meta_resid_market_")
        ],
        "folds": manifests,
        "leakage_contract": {
            "fit": "strictly prior rows for each named OOS fold",
            "transform": "frozen market recognizer and pre-entry features only",
            "local_predictions": "reused unchanged from the original OOS artifact",
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
