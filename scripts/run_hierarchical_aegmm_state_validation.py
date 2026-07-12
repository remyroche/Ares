#!/usr/bin/env python3
"""Validate shared AE/GMM states with local side x archetype responses.

This is a state-discovery gate, not a meta-model or policy experiment.  Every
fold fits AE/GMM geometry and outcome response maps only on rows ending before
the OOS month (with an execution-horizon embargo), then evaluates the next
month.  The final April--June 2026 folds remain separately identifiable from
the 2025 development research folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.hierarchical_state_validation import (  # noqa: E402
    state_validation_metrics,
)
from extreme_price_movements.local_economic_aegmm import (  # noqa: E402
    META_CROSS_SECTIONAL_GEOMETRY_FEATURES,
    OUTCOME_OR_DERIVED_COLUMNS,
    EconomicAEGMMBlock,
    HierarchicalEconomicAEGMM,
    HierarchicalEconomicAEGMMConfig,
    default_meta_economic_aegmm_blocks,
)
from extreme_price_movements.meta_cross_sectional_geometry import (  # noqa: E402
    materialize_cross_sectional_geometry,
)

DEFAULT_ARCHIVE = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260712_v3_"
    "basegeometry_fullstate/cache/local_aegmm_training_archive_state_features.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/hierarchical_aegmm_state_validation_20260712_v1"
)
DEFAULT_FOLDS = (
    "2025-06",
    "2025-09",
    "2025-12",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
ARM_BLOCKS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "hierarchical_aegmm_market": ("market_state",),
    "hierarchical_aegmm_geometry": ("cross_sectional_geometry",),
    "hierarchical_aegmm_joint": ("joint_market_geometry",),
    "hierarchical_aegmm_all_three": (
        "market_state",
        "cross_sectional_geometry",
        "joint_market_geometry",
    ),
}
OUTCOME_COLUMNS = set(OUTCOME_OR_DERIVED_COLUMNS) | {
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "dirty_positive",
    "clean_exec",
    "timeout",
    "ev_after_1pct",
    "exec_margin",
}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    for name in frame.select_dtypes(include=["float64"]).columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce", downcast="float")
    for name in frame.select_dtypes(include=["int64"]).columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce", downcast="integer")
    for name in ("side_name", "archetype_policy_key", "__symbol__"):
        if name in frame.columns:
            frame[name] = frame[name].astype("category")
    return frame


def _required_columns(path: Path) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    blocks = default_meta_economic_aegmm_blocks()
    requested = {
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score_base",
        "score",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "exec_margin",
    }
    for block in blocks:
        requested.update(block.features)
    return [name for name in sorted(requested) if name in available]


def _load_archive(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=_required_columns(path))
    if "score_base" not in frame.columns:
        if "score" not in frame.columns:
            raise ValueError("State archive needs frozen base score_base or score")
        frame["score_base"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].notna()].copy()
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["archetype_policy_key"] = (
        frame["archetype_policy_key"].astype(str).fillna("missing")
    )
    frame = frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    # The archive contains the raw relative inputs, not the derived candidate
    # book geometry. Materialize it once from the frozen base score, exactly as
    # inference does, before fitting any state model.
    missing_geometry = [
        name
        for name in META_CROSS_SECTIONAL_GEOMETRY_FEATURES
        if name.startswith("meta_xsgeom_")
    ]
    if missing_geometry:
        generated = materialize_cross_sectional_geometry(frame, score_col="score_base")
        frame = pd.concat([frame, generated], axis=1, copy=False)
    return _downcast(frame)


def _fold_months(raw: str) -> list[str]:
    months = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = [value for value in months if not pd.Period(value, freq="M")]
    if invalid:
        raise ValueError(f"Invalid fold months: {invalid}")
    return list(dict.fromkeys(months))


def _safe_oos_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[name for name in OUTCOME_COLUMNS if name in frame.columns],
        errors="ignore",
    )


def _warmup_context(
    archive: pd.DataFrame,
    *,
    before: pd.Timestamp,
    bars: int,
) -> pd.DataFrame:
    if int(bars) <= 0:
        return archive.iloc[:0]
    prior = archive.loc[archive["__ts__"].lt(before)]
    if prior.empty:
        return prior
    timestamps = prior["__ts__"].drop_duplicates(keep="last").tail(int(bars))
    return prior.loc[prior["__ts__"].isin(timestamps)].copy(deep=False)


def _state_blocks() -> dict[str, EconomicAEGMMBlock]:
    return {block.name: block for block in default_meta_economic_aegmm_blocks()}


def _fold_tag(month: str) -> str:
    return str(month).replace("-", "")


def _write_fold_outputs(
    *,
    fold_dir: Path,
    oos: pd.DataFrame,
    generated: pd.DataFrame,
    model: HierarchicalEconomicAEGMM,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keep = [
        name
        for name in (
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "score_base",
            "ev_after_1pct",
            "clean_exec",
            "dirty_positive",
            "first_touch_bad_mae_1r",
            "full_path_bad_mae_1r",
            "timeout",
            "exec_margin",
        )
        if name in oos.columns
    ]
    prediction = pd.concat(
        [oos.loc[:, keep].reset_index(drop=True), generated.reset_index(drop=True)],
        axis=1,
        copy=False,
    )
    prediction = _downcast(prediction)
    prediction.to_parquet(
        fold_dir / "oos_state_predictions.parquet", index=False, compression="zstd"
    )
    joblib.dump(model, fold_dir / "hierarchical_economic_aegmm.joblib", compress=3)
    model.catalog_.to_csv(fold_dir / "state_response_catalog.csv", index=False)
    manifest = model.manifest()
    manifest.update(
        {
            "fit_start": str(fit_start),
            "fit_end_exclusive_after_embargo": str(fit_end),
            "oos_start": str(oos_start),
            "oos_end_exclusive": str(oos_end),
            "oos_rows": int(len(oos)),
            "outcome_columns_removed_before_oos_transform": sorted(
                name for name in OUTCOME_COLUMNS if name in oos.columns
            ),
            "evaluation_contract": (
                "All reported state metrics are computed only from this fold's OOS rows. "
                "The state geometry and response mappings were fitted before the OOS start "
                "with an execution-horizon embargo."
            ),
        }
    )
    _write_json(fold_dir / "manifest.json", manifest)
    return prediction, manifest


def _baseline_metric_frame(prediction: pd.DataFrame) -> pd.DataFrame:
    """Provide the identical candidate-book outcome comparator without a state."""

    prefix = "local_econ_aegmm_baseline_"
    baseline = prediction.copy(deep=False)
    baseline[f"{prefix}gmm_cluster_id"] = np.int8(0)
    baseline[f"{prefix}expected_top10_ev"] = np.float32(np.nan)
    baseline[f"{prefix}expected_ev"] = np.float32(np.nan)
    baseline[f"{prefix}expected_top10_bad_mae"] = np.float32(np.nan)
    baseline[f"{prefix}expected_bad_mae"] = np.float32(np.nan)
    baseline[f"{prefix}support_log1p"] = np.float32(np.nan)
    return baseline


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive = _load_archive(Path(args.archive))
    available_start = archive["__ts__"].min()
    available_end = archive["__ts__"].max()
    blocks_by_name = _state_blocks()
    requested_arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = sorted(set(requested_arms) - set(ARM_BLOCKS))
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; available={sorted(ARM_BLOCKS)}")
    requested_blocks = tuple(
        dict.fromkeys(block for arm in requested_arms for block in ARM_BLOCKS[arm])
    )
    folds = _fold_months(args.fold_months)
    all_summary: list[pd.DataFrame] = []
    all_state: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    all_autocorr: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    embargo = pd.Timedelta(hours=max(int(args.embargo_hours), 0))
    fit_start = pd.Timestamp(args.fit_start, tz="UTC")
    for fold_index, month in enumerate(folds):
        period = pd.Period(month, freq="M")
        oos_start = pd.Timestamp(period.start_time, tz="UTC")
        oos_end = pd.Timestamp((period + 1).start_time, tz="UTC")
        fit_end = oos_start - embargo
        train = archive.loc[
            archive["__ts__"].ge(fit_start) & archive["__ts__"].lt(fit_end)
        ]
        oos = archive.loc[
            archive["__ts__"].ge(oos_start) & archive["__ts__"].lt(oos_end)
        ]
        if len(train) < int(args.min_train_rows) or len(oos) < int(args.min_oos_rows):
            print(
                json.dumps(
                    {
                        "event": "state_fold_skipped",
                        "fold": month,
                        "train_rows": len(train),
                        "oos_rows": len(oos),
                    }
                ),
                flush=True,
            )
            continue
        fold_dir = output / "folds" / _fold_tag(month)
        fold_dir.mkdir(parents=True, exist_ok=True)
        state = HierarchicalEconomicAEGMM(
            config=HierarchicalEconomicAEGMMConfig(
                min_fit_rows=int(args.min_state_fit_rows),
                ae_max_train_rows=int(args.ae_max_train_rows),
                gmm_max_train_rows=int(args.gmm_max_train_rows),
                ae_max_iter=int(args.ae_max_iter),
                full_train_fit=bool(args.full_train_fit),
                random_state=int(args.seed + fold_index * 10_003),
            ),
            blocks=tuple(blocks_by_name[name] for name in requested_blocks),
        )
        print(
            json.dumps(
                {
                    "event": "state_fold_fit_start",
                    "fold": month,
                    "train_rows": len(train),
                    "oos_rows": len(oos),
                    "blocks": list(requested_blocks),
                }
            ),
            flush=True,
        )
        state.fit(train)
        context = _warmup_context(
            archive, before=oos_start, bars=args.state_warmup_bars
        )
        generated = state.transform_oos_with_history(
            _safe_oos_features(context), _safe_oos_features(oos)
        )
        prediction, fold_manifest = _write_fold_outputs(
            fold_dir=fold_dir,
            oos=oos,
            generated=generated,
            model=state,
            fit_start=fit_start,
            fit_end=fit_end,
            oos_start=oos_start,
            oos_end=oos_end,
        )
        fold_manifest["fold"] = month
        fold_manifest["state_warmup_bars"] = int(args.state_warmup_bars)
        fold_manifest["state_warmup_rows"] = int(len(context))
        fold_manifests.append(fold_manifest)
        for arm in requested_arms:
            if arm == "baseline":
                summary, by_state, daily, autocorr = state_validation_metrics(
                    _baseline_metric_frame(prediction),
                    fold=month,
                    state_block="baseline",
                )
                for table in (summary, by_state, daily, autocorr):
                    table["arm"] = arm
                all_summary.append(summary)
                all_state.append(by_state)
                all_daily.append(daily)
                all_autocorr.append(autocorr)
                continue
            for block_name in ARM_BLOCKS[arm]:
                summary, by_state, daily, autocorr = state_validation_metrics(
                    prediction,
                    fold=month,
                    state_block=block_name,
                )
                for table in (summary, by_state, daily, autocorr):
                    table["arm"] = arm
                all_summary.append(summary)
                all_state.append(by_state)
                all_daily.append(daily)
                all_autocorr.append(autocorr)
        print(
            json.dumps(
                {
                    "event": "state_fold_complete",
                    "fold": month,
                    "models": len(state.shared_models),
                    "response_groups": sum(
                        len(item.local_matrices) for item in state.responses.values()
                    ),
                }
            ),
            flush=True,
        )
        del train, oos, context, state, generated, prediction
        gc.collect()

    summary = (
        pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    )
    by_state = pd.concat(all_state, ignore_index=True) if all_state else pd.DataFrame()
    daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    autocorr = (
        pd.concat(all_autocorr, ignore_index=True) if all_autocorr else pd.DataFrame()
    )
    summary.to_csv(output / "oos_zone_metrics_by_side_archetype.csv", index=False)
    by_state.to_csv(output / "oos_state_metrics_by_side_archetype.csv", index=False)
    daily.to_csv(output / "oos_daily_hit_surprise_by_side_archetype.csv", index=False)
    autocorr.to_csv(
        output / "oos_hit_surprise_autocorrelation_by_side_archetype.csv", index=False
    )
    # No baseline state feature exists. Its rows deliberately remain absent
    # from the state-separation table; it is the identical candidate-book
    # comparator used by later fixed-meta ablations.
    final_folds = {"2026-04", "2026-05", "2026-06"}
    report = {
        "schema": "hierarchical_aegmm_state_validation_v1",
        "archive": str(args.archive),
        "archive_available_start": str(available_start),
        "archive_available_end": str(available_end),
        "folds_requested": folds,
        "folds_completed": [item["fold"] for item in fold_manifests],
        "development_folds": [month for month in folds if month not in final_folds],
        "final_holdout_folds": [month for month in folds if month in final_folds],
        "arms": requested_arms,
        "state_blocks": list(requested_blocks),
        "embargo_hours": int(args.embargo_hours),
        "state_warmup_bars": int(args.state_warmup_bars),
        "metrics": {
            "zone_rows": int(len(summary)),
            "state_rows": int(len(by_state)),
            "daily_rows": int(len(daily)),
            "autocorrelation_rows": int(len(autocorr)),
        },
        "fold_manifests": fold_manifests,
        "leakage_contract": (
            "Each fold fits shared state geometry and side/archetype response maps on rows "
            "strictly before the OOS month minus embargo. OOS transforms receive no outcome "
            "fields. Economic metrics, state transfer, and signed-surprise autocorrelation "
            "are calculated only from OOS rows."
        ),
        "research_contract": (
            "2025 folds are development research. April-June 2026 are retained as distinct "
            "final holdout folds; no meta MDA/HPO or policy optimization is executed here."
        ),
    }
    _write_json(output / "manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-start", default="2025-02-01")
    parser.add_argument("--fold-months", default=",".join(DEFAULT_FOLDS))
    parser.add_argument("--arms", default=",".join(ARM_BLOCKS))
    parser.add_argument("--embargo-hours", type=int, default=12)
    parser.add_argument(
        "--state-warmup-bars",
        type=int,
        default=96,
        help="Pre-OOS 15m timestamps used only to seed frozen state dynamics.",
    )
    parser.add_argument("--min-train-rows", type=int, default=40_000)
    parser.add_argument("--min-oos-rows", type=int, default=1_000)
    parser.add_argument("--min-state-fit-rows", type=int, default=400)
    parser.add_argument("--ae-max-train-rows", type=int, default=15_000)
    parser.add_argument("--gmm-max-train-rows", type=int, default=100_000)
    parser.add_argument("--ae-max-iter", type=int, default=80)
    parser.add_argument("--full-train-fit", action="store_true")
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            _safe(
                {
                    "status": "complete",
                    "output_dir": args.output_dir,
                    "folds": result["folds_completed"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
