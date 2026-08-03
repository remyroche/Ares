#!/usr/bin/env python3
"""Paired day-block bootstrap for global-book component hurdle arms."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.bootstrap_canonical_global_book_component_attribution import (
        _rank_ic,
        _tail_spread,
    )
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        sha256,
    )
except ModuleNotFoundError:
    from bootstrap_canonical_global_book_component_attribution import (
        _rank_ic,
        _tail_spread,
    )
    from run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_component_hurdle_ablation_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_hurdle_bootstrap_20260729_v1"
)
SCHEMA = "canonical_global_book_hurdle_bootstrap_v1"
SOURCE_SCHEMA = "canonical_global_book_component_hurdle_ablation_v1"
MODELS = (
    "combined__raw_regression__B1_B4_sum",
    "band_only__hurdle_signed_mean__B1_B4_sum",
    "combined__hurdle_sign_magnitude__B1_B4_sum",
)


def _wide(source: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        source / "reconciled_sum_oof_predictions.parquet"
    )
    frame = frame.loc[
        frame["model_name"].isin(MODELS)
        & frame["target_valid"].astype(bool)
    ].copy()
    key = [
        "cohort_anchor_utc",
        "horizon_hours",
        "book_fraction",
        "fold_id",
        "validation_start_utc",
        "validation_end_utc",
    ]
    reference = frame.loc[
        frame["model_name"].eq(MODELS[0]),
        [
            *key,
            "target_delta",
            "delta_direct_mean_net",
        ],
    ].copy()
    if reference.duplicated(key).any():
        raise ValueError("raw component comparator identity is not unique")
    for model in MODELS:
        prediction = frame.loc[
            frame["model_name"].eq(model), [*key, "delta_prediction"]
        ].rename(columns={"delta_prediction": model})
        reference = reference.merge(
            prediction, on=key, how="inner", validate="one_to_one"
        )
    if len(reference) != len(
        frame.loc[frame["model_name"].eq(MODELS[0])]
    ):
        raise ValueError("hurdle model rows are not exactly paired")
    for column in (
        "cohort_anchor_utc",
        "validation_start_utc",
        "validation_end_utc",
    ):
        reference[column] = pd.to_datetime(
            reference[column], utc=True, errors="raise"
        )
    reference["calendar_day"] = reference["cohort_anchor_utc"].dt.floor("D")
    reference["month"] = reference["cohort_anchor_utc"].dt.strftime("%Y-%m")
    reference["full_fold"] = (
        reference["validation_end_utc"]
        - reference["validation_start_utc"]
    ).ge(pd.Timedelta(days=14))
    return reference


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["target_delta"].to_numpy(dtype=float)
    zero_mae = float(np.abs(y).mean())
    raw = MODELS[0]
    raw_prediction = frame[raw].to_numpy(dtype=float)
    raw_mae = float(np.abs(y - raw_prediction).mean())
    result: dict[str, float] = {
        "zero_mae": zero_mae,
        "raw_mae": raw_mae,
        "raw_minus_zero_mae": raw_mae - zero_mae,
    }
    for model in MODELS[1:]:
        prediction = frame[model].to_numpy(dtype=float)
        mae = float(np.abs(y - prediction).mean())
        prefix = (
            "band_hurdle"
            if model.startswith("band_only")
            else "combined_hurdle"
        )
        result[f"{prefix}_mae"] = mae
        result[f"{prefix}_minus_zero_mae"] = mae - zero_mae
        result[f"{prefix}_minus_raw_mae"] = mae - raw_mae
        result[f"{prefix}_rank_ic"] = _rank_ic(y, prediction)
        result[f"{prefix}_target_quintile_spread"] = _tail_spread(
            frame, model, "target_delta"
        )
        result[f"{prefix}_direct_net_quintile_spread"] = _tail_spread(
            frame, model, "delta_direct_mean_net"
        )
    result["raw_rank_ic"] = _rank_ic(y, raw_prediction)
    result["raw_target_quintile_spread"] = _tail_spread(
        frame, raw, "target_delta"
    )
    result["raw_direct_net_quintile_spread"] = _tail_spread(
        frame, raw, "delta_direct_mean_net"
    )
    return result


def _bootstrap(
    frame: pd.DataFrame, *, scope: str, draws: int, seed: int
) -> pd.DataFrame:
    blocks = {
        day: group.index.to_numpy(dtype=np.int64)
        for day, group in frame.groupby("calendar_day", sort=True)
    }
    days = np.array(list(blocks), dtype=object)
    random = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    for draw in range(int(draws)):
        sampled_days = random.choice(days, size=len(days), replace=True)
        indices = np.concatenate([blocks[day] for day in sampled_days])
        records.append(
            {
                "scope": scope,
                "draw": draw,
                **_metrics(frame.loc[indices]),
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifest, source_hashes = _artifact_manifest(source, SOURCE_SCHEMA)
    predictions = source / "reconciled_sum_oof_predictions.parquet"
    if manifest.get("outputs_sha256", {}).get(predictions.name) != sha256(
        predictions
    ):
        raise ValueError("hurdle prediction hash mismatch")
    frame = _wide(source)
    latest_full = int(frame.loc[frame["full_fold"], "fold_id"].max())
    scopes = {
        "development_full_folds_0_3": frame.loc[frame["full_fold"]].copy(),
        "latest_full_fold_3": frame.loc[
            frame["fold_id"].eq(latest_full)
        ].copy(),
        "march_2025": frame.loc[frame["month"].eq("2025-03")].copy(),
        "april_2025": frame.loc[frame["month"].eq("2025-04")].copy(),
    }
    draw_parts: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    for index, (scope, values) in enumerate(scopes.items()):
        draws = _bootstrap(
            values,
            scope=scope,
            draws=int(args.draws),
            seed=int(args.seed) + index,
        )
        draw_parts.append(draws)
        point = _metrics(values)
        for metric, estimate in point.items():
            distribution = pd.to_numeric(
                draws[metric], errors="coerce"
            ).dropna()
            summary_records.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "point_estimate": estimate,
                    "ci_2_5": float(distribution.quantile(0.025)),
                    "ci_50": float(distribution.quantile(0.50)),
                    "ci_97_5": float(distribution.quantile(0.975)),
                    "probability_below_zero": float(
                        (distribution < 0).mean()
                    ),
                    "probability_above_zero": float(
                        (distribution > 0).mean()
                    ),
                    "rows": int(len(values)),
                    "day_blocks": int(
                        values["calendar_day"].nunique()
                    ),
                }
            )
    draw_table = pd.concat(draw_parts, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    draw_table.to_parquet(
        temporary / "bootstrap_draws.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_parquet(
        temporary / "bootstrap_summary.parquet",
        index=False,
        compression="zstd",
    )
    artifact_manifest = {
        "schema": SCHEMA,
        "status": "PAIRED_DAY_BLOCK_BOOTSTRAP_DIAGNOSTIC",
        "source_artifacts_sha256": {
            **source_hashes,
            str(predictions): sha256(predictions),
        },
        "models": list(MODELS),
        "draws_per_scope": int(args.draws),
        "seed": int(args.seed),
        "contracts": {
            "paired": "raw and both hurdle arms use identical H12/global-10% OOF anchors and targets",
            "resampling": "UTC calendar-day blocks sampled with replacement; hourly H12 rows are not treated as independent",
            "scope": "uncertainty only; no promotion, admission or policy claim",
        },
        "outputs_sha256": {
            path.name: sha256(path)
            for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(artifact_manifest), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "draw_rows": int(len(draw_table)),
        "summary_rows": int(len(summary)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--draws", type=int, default=2_000)
    result.add_argument("--seed", type=int, default=20260729)
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
