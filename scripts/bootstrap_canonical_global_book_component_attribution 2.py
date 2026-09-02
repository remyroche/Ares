#!/usr/bin/env python3
"""Block-bootstrap exact-book direct versus reconciled component predictions."""

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
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        sha256,
    )
except ModuleNotFoundError:
    from run_canonical_economic_conversion_transition_head_ablation import (
        _artifact_manifest,
        _safe,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_reconciled_component_ablation_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_component_bootstrap_20260729_v1"
)
SCHEMA = "canonical_global_book_component_bootstrap_v1"
SOURCE_SCHEMA = "canonical_global_book_reconciled_component_ablation_v1"
MODELS = (
    "compact_global_direct_residual",
    "reconciled_component_sum_B1_B4",
)


def _paired(source: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(source / "oof_predictions.parquet")
    predictions = predictions.loc[
        predictions["horizon_hours"].eq(12)
        & predictions["model_name"].isin(MODELS)
        & predictions["target_valid"].astype(bool)
    ].copy()
    key = [
        "cohort_anchor_utc",
        "horizon_hours",
        "book_fraction",
        "fold_id",
        "validation_start_utc",
        "validation_end_utc",
    ]
    values = [
        "target_delta",
        "delta_prediction",
        "delta_direct_mean_net",
    ]
    direct = predictions.loc[
        predictions["model_name"].eq(MODELS[0]), [*key, *values]
    ].rename(
        columns={
            "target_delta": "target_delta_direct",
            "delta_prediction": "direct_prediction",
            "delta_direct_mean_net": "direct_net_direct",
        }
    )
    component = predictions.loc[
        predictions["model_name"].eq(MODELS[1]), [*key, *values]
    ].rename(
        columns={
            "target_delta": "target_delta_component",
            "delta_prediction": "component_prediction",
            "delta_direct_mean_net": "direct_net_component",
        }
    )
    paired = direct.merge(
        component, on=key, how="inner", validate="one_to_one"
    )
    if not np.allclose(
        paired["target_delta_direct"],
        paired["target_delta_component"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("direct and component targets are not identical")
    if not np.allclose(
        paired["direct_net_direct"],
        paired["direct_net_component"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("direct and component economic rows are not identical")
    paired = paired.rename(
        columns={
            "target_delta_direct": "target_delta",
            "direct_net_direct": "delta_direct_mean_net",
        }
    ).drop(
        columns=["target_delta_component", "direct_net_component"]
    )
    paired["cohort_anchor_utc"] = pd.to_datetime(
        paired["cohort_anchor_utc"], utc=True, errors="raise"
    )
    paired["validation_start_utc"] = pd.to_datetime(
        paired["validation_start_utc"], utc=True, errors="raise"
    )
    paired["validation_end_utc"] = pd.to_datetime(
        paired["validation_end_utc"], utc=True, errors="raise"
    )
    paired["full_fold"] = (
        paired["validation_end_utc"] - paired["validation_start_utc"]
    ).ge(pd.Timedelta(days=14))
    paired["calendar_day"] = paired["cohort_anchor_utc"].dt.floor("D")
    paired["month"] = paired["cohort_anchor_utc"].dt.strftime("%Y-%m")
    return paired


def _rank_ic(y: np.ndarray, prediction: np.ndarray) -> float:
    if (
        len(y) < 2
        or np.unique(y).size < 2
        or np.unique(prediction).size < 2
    ):
        return np.nan
    return float(
        pd.Series(y).corr(pd.Series(prediction), method="spearman")
    )


def _tail_spread(
    frame: pd.DataFrame, prediction_column: str, outcome_column: str
) -> float:
    if len(frame) < 5:
        return np.nan
    ranks = pd.qcut(
        frame[prediction_column].rank(method="first"),
        5,
        labels=False,
        duplicates="drop",
    )
    return float(
        frame.loc[ranks.eq(ranks.max()), outcome_column].mean()
        - frame.loc[ranks.eq(ranks.min()), outcome_column].mean()
    )


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["target_delta"].to_numpy(dtype=float)
    direct = frame["direct_prediction"].to_numpy(dtype=float)
    component = frame["component_prediction"].to_numpy(dtype=float)
    zero = np.zeros(len(frame), dtype=float)
    direct_mae = float(np.abs(y - direct).mean())
    component_mae = float(np.abs(y - component).mean())
    zero_mae = float(np.abs(y - zero).mean())
    return {
        "direct_mae": direct_mae,
        "component_mae": component_mae,
        "zero_mae": zero_mae,
        "direct_minus_zero_mae": direct_mae - zero_mae,
        "component_minus_zero_mae": component_mae - zero_mae,
        "component_minus_direct_mae": component_mae - direct_mae,
        "direct_rank_ic": _rank_ic(y, direct),
        "component_rank_ic": _rank_ic(y, component),
        "component_minus_direct_rank_ic": _rank_ic(y, component)
        - _rank_ic(y, direct),
        "direct_target_quintile_spread": _tail_spread(
            frame, "direct_prediction", "target_delta"
        ),
        "component_target_quintile_spread": _tail_spread(
            frame, "component_prediction", "target_delta"
        ),
        "direct_net_quintile_spread": _tail_spread(
            frame, "direct_prediction", "delta_direct_mean_net"
        ),
        "component_direct_net_quintile_spread": _tail_spread(
            frame, "component_prediction", "delta_direct_mean_net"
        ),
    }


def _draws(
    frame: pd.DataFrame,
    *,
    scope: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    blocks = {
        day: group.index.to_numpy(dtype=np.int64)
        for day, group in frame.groupby("calendar_day", sort=True)
    }
    days = np.array(list(blocks), dtype=object)
    if not len(days):
        return pd.DataFrame()
    random = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    for draw in range(int(draws)):
        sampled_days = random.choice(days, size=len(days), replace=True)
        indices = np.concatenate([blocks[day] for day in sampled_days])
        sampled = frame.loc[indices]
        records.append(
            {
                "scope": scope,
                "draw": draw,
                "sampled_day_blocks": int(len(sampled_days)),
                "sampled_rows": int(len(sampled)),
                **_metrics(sampled),
            }
        )
    return pd.DataFrame(records)


def _summary(
    frame: pd.DataFrame, draws: pd.DataFrame, scope: str
) -> pd.DataFrame:
    point = _metrics(frame)
    records: list[dict[str, Any]] = []
    for metric, value in point.items():
        samples = pd.to_numeric(draws[metric], errors="coerce").dropna()
        records.append(
            {
                "scope": scope,
                "metric": metric,
                "point_estimate": value,
                "ci_2_5": float(samples.quantile(0.025)),
                "ci_50": float(samples.quantile(0.50)),
                "ci_97_5": float(samples.quantile(0.975)),
                "probability_below_zero": float((samples < 0).mean()),
                "probability_above_zero": float((samples > 0).mean()),
                "rows": int(len(frame)),
                "day_blocks": int(frame["calendar_day"].nunique()),
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifest, source_hashes = _artifact_manifest(source, SOURCE_SCHEMA)
    predictions = source / "oof_predictions.parquet"
    if not predictions.is_file():
        raise FileNotFoundError("component artifact lacks OOF predictions")
    expected = manifest.get("outputs_sha256", {}).get(predictions.name)
    if expected != sha256(predictions):
        raise ValueError("component OOF prediction hash mismatch")
    paired = _paired(source)
    scopes = {
        "development_full_folds_0_3": paired.loc[
            paired["full_fold"]
        ].copy(),
        "latest_full_fold_3": paired.loc[
            paired["fold_id"].eq(
                paired.loc[paired["full_fold"], "fold_id"].max()
            )
        ].copy(),
        "march_2025": paired.loc[paired["month"].eq("2025-03")].copy(),
        "april_2025": paired.loc[paired["month"].eq("2025-04")].copy(),
    }
    draw_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for index, (scope, frame) in enumerate(scopes.items()):
        draws = _draws(
            frame,
            scope=scope,
            draws=int(args.draws),
            seed=int(args.seed) + index,
        )
        draw_parts.append(draws)
        summary_parts.append(_summary(frame, draws, scope))
    draw_table = pd.concat(draw_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
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
        "draws_per_scope": int(args.draws),
        "seed": int(args.seed),
        "scopes": {
            scope: {
                "rows": int(len(frame)),
                "day_blocks": int(frame["calendar_day"].nunique()),
            }
            for scope, frame in scopes.items()
        },
        "contracts": {
            "paired_rows": "same exact H12 10% global-book anchors and target for direct and reconciled component predictions",
            "resampling": "whole UTC calendar-day blocks sampled with replacement; hourly overlapping windows are never treated as independent",
            "selection": "folds 0-3 are full development folds; truncated fold 4 is excluded from selection scopes",
            "scope": "uncertainty and frozen-tail attribution only; no admission, policy or PnL claim",
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
