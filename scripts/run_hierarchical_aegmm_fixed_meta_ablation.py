#!/usr/bin/env python3
"""Run fixed-meta ablations for validated hierarchical AE/GMM state blocks.

This runner is deliberately gated behind ``--state-validation-passed``.  It
uses the same base candidate stream and frozen meta hyperparameters as the
reference model.  It does not rerun MDA or HPO: the only varying inputs are
the validated shared-state outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.local_economic_aegmm import (  # noqa: E402
    OUTCOME_OR_DERIVED_COLUMNS,
    HierarchicalEconomicAEGMM,
    HierarchicalEconomicAEGMMConfig,
    default_meta_economic_aegmm_blocks,
    local_economic_aegmm_feature_names,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    META_POST_SELECTION_OOD_FEATURE_NAMES,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_FEATURE_ROOT,
    DEFAULT_HANDOFF,
    DEFAULT_REFERENCE_DIR,
    EVAL_MONTHS,
    _append_cross_sectional_geometry,
    _downcast,
    _ensure_base_score,
    _prepare_local_aegmm_training_archive,
    _write_json,
    metrics_by_scope,
    prepare_dataset,
    surprise_calendar,
    train_arm_oos,
)

DEFAULT_ARCHIVE = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260712_v3_"
    "basegeometry_fullstate/cache/local_aegmm_training_archive_state_features.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/hierarchical_aegmm_fixed_meta_ablation_20260712_v1"
)
ARMS: dict[str, tuple[str, ...]] = {
    "baseline_retrained": (),
    "hierarchical_aegmm_market": ("market_state",),
    # This arm exposes only side x archetype-specific response outputs. It
    # prevents a global LightGBM split on raw posterior/latent coordinates from
    # helping one archetype while damaging another.
    "hierarchical_aegmm_market_response_only": ("market_state",),
    "hierarchical_aegmm_geometry": ("cross_sectional_geometry",),
    "hierarchical_aegmm_joint": ("joint_market_geometry",),
    "hierarchical_aegmm_all_three": (
        "market_state",
        "cross_sectional_geometry",
        "joint_market_geometry",
    ),
}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _blocks_for_arm(arm: str):
    known = {block.name: block for block in default_meta_economic_aegmm_blocks()}
    return tuple(known[name] for name in ARMS[arm])


def _fit_hierarchical_state(
    *,
    archive: pd.DataFrame,
    blocks: Sequence[Any],
    output_dir: Path,
    arm: str,
    fit_start: str,
    fit_end: str,
    seed: int,
    full_train_fit: bool,
) -> HierarchicalEconomicAEGMM:
    start = pd.Timestamp(fit_start, tz="UTC")
    end = pd.Timestamp(fit_end, tz="UTC")
    block_tag = "__".join(str(block.name) for block in blocks)
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}_{'full' if full_train_fit else 'sampled'}_{block_tag}_hierarchical_v1"
    state_dir = output_dir / arm / "state" / tag
    state_path = state_dir / "hierarchical_economic_aegmm.joblib"
    if state_path.exists():
        return joblib.load(state_path)
    timestamp = pd.to_datetime(archive["__ts__"], utc=True, errors="coerce")
    train = archive.loc[timestamp.ge(start) & timestamp.lt(end)]
    if len(train) < 10_000:
        raise ValueError(f"Hierarchical state archive is too small: {len(train)} rows")
    state = HierarchicalEconomicAEGMM(
        config=HierarchicalEconomicAEGMMConfig(
            random_state=int(seed),
            full_train_fit=bool(full_train_fit),
        ),
        blocks=tuple(blocks),
    ).fit(train)
    state_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, state_path, compress=3)
    _write_json(
        state_dir / "hierarchical_economic_aegmm.manifest.json", state.manifest()
    )
    state.catalog_.to_csv(state_dir / "state_response_catalog.csv", index=False)
    return state


def _append_state(
    data: pd.DataFrame,
    state: HierarchicalEconomicAEGMM,
    *,
    history: pd.DataFrame,
    warmup_bars: int,
) -> pd.DataFrame:
    timestamp = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    start = timestamp.min()
    prior = history.loc[
        pd.to_datetime(history["__ts__"], utc=True, errors="coerce").lt(start)
    ]
    warmup_timestamps = (
        prior["__ts__"].drop_duplicates(keep="last").tail(max(int(warmup_bars), 0))
    )
    warmup = prior.loc[prior["__ts__"].isin(warmup_timestamps)]
    generated = state.transform_oos_with_history(
        warmup.drop(
            columns=[
                name for name in OUTCOME_OR_DERIVED_COLUMNS if name in warmup.columns
            ],
            errors="ignore",
        ),
        data.drop(
            columns=[
                name for name in OUTCOME_OR_DERIVED_COLUMNS if name in data.columns
            ],
            errors="ignore",
        ),
    )
    return pd.concat(
        [
            data.drop(columns=[name for name in generated if name in data.columns]),
            generated,
        ],
        axis=1,
        copy=False,
    )


def _fixed_features(
    reference_features: Sequence[str],
    data: pd.DataFrame,
    blocks: Sequence[Any],
    *,
    response_only: bool,
) -> list[str]:
    # The baseline keeps the previous meta feature universe untouched. Each
    # challenger differs only by its continuous frozen state features.
    selected = [
        name
        for name in reference_features
        if (
            name in data.columns
            or name in META_POST_SELECTION_OOD_FEATURE_NAMES
            or name.startswith(("rel_rankband_", "rel_marginband_"))
        )
    ]
    if blocks:
        state_features = local_economic_aegmm_feature_names(
            [block.name for block in blocks]
        )
        if response_only:
            state_features = [
                name
                for name in state_features
                if any(
                    token in name
                    for token in (
                        "_expected_",
                        "_prob__",
                        "_support_log1p",
                        "_local_model",
                        "_enabled",
                    )
                )
            ]
        selected.extend(state_features)
    return list(dict.fromkeys(selected))


def _delta_vs_baseline(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metrics = metrics.copy()
    baseline = metrics.loc[
        metrics["arm"].eq("baseline_retrained")
        & metrics["selector"].eq("baseline_retrained")
    ].copy()
    challengers = metrics.loc[
        ~metrics["arm"].eq("baseline_retrained")
        & (metrics["selector"] == metrics["arm"])
    ].copy()
    dimensions = [
        name
        for name in (
            "scope",
            "fraction",
            "selection_basis",
            "calendar_month",
            "week_start",
            "side_name",
            "archetype_policy_key",
        )
        if name in metrics.columns
    ]
    values = [
        "mean_ev_after_1pct",
        "sum_ev_after_1pct",
        "clean_exec_precision",
        "dirty_positive_rate",
        "first_touch_bad_mae_rate",
        "timeout_rate",
        "mean_hit_surprise",
    ]
    right = baseline.reindex(columns=dimensions + values).rename(
        columns={name: f"baseline_{name}" for name in values}
    )
    result = challengers.merge(
        right, on=dimensions, how="inner", validate="many_to_one"
    )
    for name in values:
        result[f"delta_{name}"] = result[name] - result[f"baseline_{name}"]
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.state_validation_passed:
        raise RuntimeError(
            "State validation has not been approved. Run the state-only validation, "
            "review OOS transfer, then pass --state-validation-passed explicitly."
        )
    requested = [value.strip() for value in args.arms.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(ARMS))
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; available={sorted(ARMS)}")
    output = Path(args.output_dir)
    dataset_path, dataset_manifest, reference_features, params = prepare_dataset(
        handoff=args.handoff,
        reference_dir=args.reference_dir,
        feature_root=args.feature_root,
        output_dir=output,
        force=False,
    )
    data = pd.read_parquet(dataset_path)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = _append_cross_sectional_geometry(_ensure_base_score(_downcast(data)))
    supplied_archive = args.aegmm_training_archive
    if supplied_archive is not None and supplied_archive.exists():
        archive_path = supplied_archive
        archive_manifest = {
            "hydrated_archive": str(archive_path),
            "source": "direct_existing_hydrated_archive",
            "columns": len(pq.ParquetFile(archive_path).schema_arrow.names),
        }
    else:
        archive_path, archive_manifest = _prepare_local_aegmm_training_archive(
            archive_path=args.handoff,
            feature_root=args.feature_root,
            output_dir=output,
            state_feature_keys=[
                *default_meta_economic_aegmm_blocks()[0].features,
                *default_meta_economic_aegmm_blocks()[1].features,
            ],
            force=False,
        )
    archive = pd.read_parquet(archive_path)
    archive["__ts__"] = pd.to_datetime(archive["__ts__"], utc=True, errors="coerce")
    archive = _append_cross_sectional_geometry(_ensure_base_score(_downcast(archive)))
    requested_blocks = tuple(
        dict.fromkeys(block for arm in requested for block in ARMS[arm])
    )
    shared_state = None
    state_data = None
    if requested_blocks:
        all_blocks = {
            block.name: block for block in default_meta_economic_aegmm_blocks()
        }
        shared_state = _fit_hierarchical_state(
            archive=archive,
            blocks=tuple(all_blocks[name] for name in requested_blocks),
            output_dir=output,
            arm="shared_frozen_state",
            fit_start=args.aegmm_fit_start,
            fit_end=args.aegmm_fit_end,
            seed=int(args.seed),
            full_train_fit=bool(args.aegmm_full_fit),
        )
        state_data = _append_state(
            data,
            shared_state,
            history=archive,
            warmup_bars=int(args.state_warmup_bars),
        )
    metrics_all: list[pd.DataFrame] = []
    autocorr_all: list[pd.DataFrame] = []
    calendar_all: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for index, arm in enumerate(requested):
        blocks = _blocks_for_arm(arm)
        arm_data = data
        state = None
        if blocks:
            state = shared_state
            if state is None or state_data is None:
                raise RuntimeError("Shared frozen state was not materialized")
            arm_data = state_data
        response_only = arm.endswith("_response_only")
        features = _fixed_features(
            reference_features,
            arm_data,
            blocks,
            response_only=response_only,
        )
        arm_dir = output / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature": features}).to_csv(
            arm_dir / "fixed_features.csv", index=False
        )
        predictions, train_manifest = train_arm_oos(
            arm=arm,
            data=arm_data,
            selected_features=features,
            params=params,
            output_dir=output,
            seed=int(args.seed + index * 1009),
            local_aegmm_state=state,
        )
        metrics = metrics_by_scope(predictions, arm)
        calendar, autocorr, _period = surprise_calendar(predictions, arm)
        metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
        calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
        autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
        metrics_all.append(metrics)
        calendar_all.append(calendar)
        autocorr_all.append(autocorr)
        manifests.append(
            {
                "arm": arm,
                "blocks": [block.name for block in blocks],
                "response_only": bool(response_only),
                "feature_count": int(len(features)),
                "frozen_meta_params": params,
                "state_manifest": state.manifest() if state is not None else None,
                "train_manifest": train_manifest,
            }
        )
        del arm_data, state, predictions, metrics, calendar, autocorr
        gc.collect()
    merged_metrics = (
        pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    )
    merged_calendar = (
        pd.concat(calendar_all, ignore_index=True) if calendar_all else pd.DataFrame()
    )
    merged_autocorr = (
        pd.concat(autocorr_all, ignore_index=True) if autocorr_all else pd.DataFrame()
    )
    merged_metrics.to_csv(output / "all_metrics_by_scope.csv", index=False)
    merged_calendar.to_csv(output / "all_hit_surprise_calendar.csv", index=False)
    merged_autocorr.to_csv(output / "all_hit_surprise_autocorrelation.csv", index=False)
    _delta_vs_baseline(merged_metrics).to_csv(
        output / "delta_vs_baseline.csv", index=False
    )
    manifest = {
        "schema": "hierarchical_aegmm_fixed_meta_ablation_v1",
        "dataset": dataset_manifest,
        "state_archive": archive_manifest,
        "arms": manifests,
        "eval_months": list(EVAL_MONTHS),
        "base_model": "fixed; never retrained",
        "meta_hpo": "fixed reference parameters; not rerun",
        "feature_selection": "disabled for this state-ablation gate; fixed reference features plus all state outputs",
        "leakage_contract": (
            "Each state bundle is fitted only through the fixed pre-April cutoff and frozen "
            "for all April-June meta folds. Meta fold training uses only earlier rows; OOS "
            "rows receive frozen state outputs and no outcome fields."
        ),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--aegmm-training-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aegmm-fit-start", default="2025-02-01")
    parser.add_argument("--aegmm-fit-end", default="2026-03-01")
    parser.add_argument("--aegmm-full-fit", action="store_true")
    parser.add_argument("--state-warmup-bars", type=int, default=96)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--state-validation-passed", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            _safe(
                {"status": "complete", "arms": [item["arm"] for item in result["arms"]]}
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
