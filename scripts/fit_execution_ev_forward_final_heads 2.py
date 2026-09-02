#!/usr/bin/env python3
"""Fit frozen final direct-net and capture heads for future-only inference.

The source data, labels, feature manifest and base-margin screen are pinned to
the already completed v8 temporal-OOF experiment.  This script performs no HPO,
feature selection, threshold selection, mapping, or evaluation.  It refits the
two raw heads on every row whose exact-policy label resolved strictly before a
declared future-block cutoff and serializes deployable, side-local CatBoost
models plus their complete feature and lineage contract.

The resulting models are inference machinery, not new OOS evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_mixed_period_remedies import (  # noqa: E402
    ARCHETYPE_COLUMN,
    BASELINE_COLUMN,
    DECISION_COLUMN,
    IDENTITY_COLUMNS,
    RESOLUTION_COLUMN,
    SIDE_COLUMN,
    TARGET_COLUMN,
    _model_features,
)
from scripts.run_exact_policy_capture_hurdle_ablation import (  # noqa: E402
    _classifier,
    _regressor,
)
from scripts.run_exact_policy_capture_support_ablation import (  # noqa: E402
    SIDES,
    add_support_targets,
    assert_frozen_interaction_sources,
    load_frozen_base_margin_interaction,
)


SCHEMA = "execution_ev_forward_final_heads_v1"
DEFAULT_INPUT = Path(
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/"
    "joined.parquet"
)
DEFAULT_CAPTURE_LABELS = Path(
    "data_perp/artifacts/exact_policy_capture_labels_20260727_v1/"
    "exact_policy_capture_labels.parquet"
)
DEFAULT_LABEL_GRID = Path(
    "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
    "meaningful_mfe_label_grid.parquet"
)
DEFAULT_FEATURE_MANIFEST = Path(
    "data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_"
    "20260726_v1/regime_diagnosis_manifest.json"
)
DEFAULT_SCREEN = Path(
    "data_perp/artifacts/execution_ev_false_positive_feature_diagnosis_20260727_v2/"
    "frozen_screens.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def prepare_training_frame(
    frame: pd.DataFrame,
    capture: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    grid_name: str,
    training_label_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Build the exact v8 head-training population and enforce its cutoff."""

    training_cutoff = _utc(
        training_label_end_exclusive,
        name="training_label_end_exclusive",
    )
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("canonical input contains duplicate identities")
    if capture.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("capture labels contain duplicate identities")
    selected_grid = grid.loc[
        grid["grid_name"].eq(grid_name) & grid["label_valid"],
        [*IDENTITY_COLUMNS, "favorable_first", "adverse_first", "timeout"],
    ].copy()
    if selected_grid.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("comparison label grid contains duplicate identities")
    work = frame.merge(
        selected_grid,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    renamed_capture = capture.rename(
        columns={
            "execution_gross_ev_12h": "capture_label_exact_gross",
            "execution_cost_return": "capture_label_exact_cost",
        }
    )
    keep_capture = [
        *IDENTITY_COLUMNS,
        "capture_label_exact_gross",
        "capture_label_exact_cost",
        "pre_exit_mfe_return",
        "pre_exit_mae_return",
        "pre_exit_mfe_to_gross_gap",
        "pre_exit_gross_capture_ratio",
        "post_peak_close_giveback_ratio",
        "giveback_after_80pct_mfe_ratio",
        "favorable_before_adverse_at_cost",
        "adverse_before_favorable_at_cost",
        "exact_net_positive",
        "exact_net_loss_worse_two_costs",
    ]
    work = work.merge(
        renamed_capture.loc[:, keep_capture],
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    work = add_support_targets(work)
    resolution = pd.to_datetime(work[RESOLUTION_COLUMN], utc=True, errors="raise")
    work = work.loc[resolution.lt(training_cutoff)].copy()
    if work.empty:
        raise ValueError("no fully resolved rows precede the final-refit cutoff")
    if pd.to_datetime(work[RESOLUTION_COLUMN], utc=True).max() >= training_cutoff:
        raise AssertionError("training-label cutoff was not enforced")
    if (
        pd.to_datetime(work[RESOLUTION_COLUMN], utc=True)
        <= pd.to_datetime(work[DECISION_COLUMN], utc=True)
    ).any():
        raise ValueError("execution label must resolve after its decision")
    return work.sort_values([DECISION_COLUMN, "candidate_id"], kind="stable").reset_index(
        drop=True
    )


def _feature_columns(
    manifest: Mapping[str, Any],
    frame: pd.DataFrame,
) -> dict[str, list[str]]:
    if "feature_columns_by_side" in manifest:
        by_side = {
            side: list(manifest["feature_columns_by_side"][side]) for side in SIDES
        }
    else:
        by_side = {side: list(manifest["feature_columns"]) for side in SIDES}
    for side, columns in by_side.items():
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"{side} feature contract is empty or duplicated")
        for column in columns:
            prefix = "catboost_archetype__"
            if column.startswith(prefix) and column not in frame:
                level = column[len(prefix) :]
                frame[column] = (
                    frame[ARCHETYPE_COLUMN].astype(str).eq(level).astype("float32")
                )
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{side} final-head features missing: {missing}")
    return by_side


def fit_final_heads(
    frame: pd.DataFrame,
    feature_columns_by_side: Mapping[str, Sequence[str]],
    *,
    iterations: int,
    seed: int,
    n_jobs: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit the exact raw direct-residual and capture classifiers per side."""

    output_dir.mkdir(parents=True, exist_ok=False)
    side_reports: dict[str, Any] = {}
    for side_index, side in enumerate(SIDES):
        local = frame.loc[frame[SIDE_COLUMN].astype(str).eq(side)].copy()
        if len(local) < 2_000:
            raise ValueError(f"{side} has insufficient final-refit support")
        features = list(feature_columns_by_side[side])
        x = _model_features(local, local, features, trust_composites=False)
        direct_target = (
            local[TARGET_COLUMN].to_numpy(dtype=float)
            - local[BASELINE_COLUMN].to_numpy(dtype=float)
        )
        capture_target = local["exact_net_positive"].to_numpy(dtype=np.int8)
        if np.unique(capture_target).size != 2:
            raise ValueError(f"{side} capture target has only one class")
        direct = _regressor(
            iterations=iterations,
            seed=seed + 10_000 * side_index,
            n_jobs=n_jobs,
        )
        capture = _classifier(
            iterations=iterations,
            seed=seed + 10_000 * side_index + 11,
            n_jobs=n_jobs,
        )
        direct.fit(x, direct_target)
        capture.fit(x, capture_target)
        direct_path = output_dir / f"direct_exact_net_residual_{side}.cbm"
        capture_path = output_dir / f"capture_probability_{side}.cbm"
        direct.save_model(direct_path)
        capture.save_model(capture_path)
        side_reports[side] = {
            "rows": int(len(local)),
            "decision_min_utc": pd.to_datetime(
                local[DECISION_COLUMN], utc=True
            ).min(),
            "decision_max_utc": pd.to_datetime(
                local[DECISION_COLUMN], utc=True
            ).max(),
            "training_label_end_max_utc": pd.to_datetime(
                local[RESOLUTION_COLUMN], utc=True
            ).max(),
            "exact_net_positive_rate": float(capture_target.mean()),
            "direct_residual_target_mean": float(direct_target.mean()),
            "feature_count": len(features),
            "feature_columns": features,
            "models": {
                "direct_exact_net_residual": {
                    "path": direct_path,
                    "sha256": _sha256(direct_path),
                },
                "capture_probability": {
                    "path": capture_path,
                    "sha256": _sha256(capture_path),
                },
            },
        }
    return side_reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # This assertion deliberately binds the refit to the already-consumed v8
    # research sources.  New outcomes must never enter this final fit.
    source_hashes = assert_frozen_interaction_sources(args)
    interaction = load_frozen_base_margin_interaction(args.base_margin_screen)
    frame = prepare_training_frame(
        pd.read_parquet(args.input),
        pd.read_parquet(args.capture_labels),
        pd.read_parquet(args.label_grid),
        grid_name=args.grid_name,
        training_label_end_exclusive=_utc(
            args.training_label_end_exclusive,
            name="training_label_end_exclusive",
        ),
    )
    feature_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    features = _feature_columns(feature_manifest, frame)
    side_reports = fit_final_heads(
        frame,
        features,
        iterations=args.n_estimators,
        seed=args.random_state,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
    )
    contract_path = args.output_dir / "feature_contract.json"
    contract = {
        "schema": "execution_ev_forward_final_head_feature_contract_v1",
        "feature_columns_by_side": features,
        "preprocessing": (
            "numeric float32 in frozen order; no imputation; all values finite; "
            "catboost archetype one-hot fields derived from the pre-entry predicted archetype"
        ),
        "direct_score": "existing_alpha_ev + predicted exact-net residual",
        "capture_score": "predict_proba(exact_net_positive)[:, 1]",
        "base_margin_interaction": interaction,
    }
    _write_json(contract_path, contract)
    manifest = {
        "schema": SCHEMA,
        "status": "frozen_final_refit_for_future_inference_not_oos_evidence",
        "training_label_end_exclusive_utc": _utc(
            args.training_label_end_exclusive,
            name="training_label_end_exclusive",
        ),
        "contract": {
            "feature_selection": "reused byte-for-byte from pinned v8 feature manifest",
            "hpo": "none; fixed v8 CatBoost geometry",
            "side_local": True,
            "targets": {
                "direct": "execution_net_ev_12h - existing_alpha_ev",
                "capture": "exact_net_positive",
            },
            "future_evaluation": (
                "score only identities after the separately frozen forward cutoff; "
                "no refit, HPO, mapping selection, or formula changes"
            ),
        },
        "inputs": {
            "data": {"path": args.input, "sha256": _sha256(args.input)},
            "capture_labels": {
                "path": args.capture_labels,
                "sha256": _sha256(args.capture_labels),
            },
            "comparison_label_grid": {
                "path": args.label_grid,
                "sha256": _sha256(args.label_grid),
                "grid_name": args.grid_name,
            },
            "feature_manifest": {
                "path": args.feature_manifest,
                "sha256": _sha256(args.feature_manifest),
            },
            "base_margin_screen": {
                "path": args.base_margin_screen,
                "sha256": _sha256(args.base_margin_screen),
            },
            "pinned_v8_source_hashes_verified": source_hashes,
        },
        "model": {
            "iterations": args.n_estimators,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
        },
        "feature_contract": {
            "path": contract_path,
            "sha256": _sha256(contract_path),
        },
        "sides": side_reports,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--capture-labels", type=Path, default=DEFAULT_CAPTURE_LABELS)
    parser.add_argument("--label-grid", type=Path, default=DEFAULT_LABEL_GRID)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--base-margin-screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument(
        "--training-label-end-exclusive",
        default="2026-07-28T00:00:00Z",
    )
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260727)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_forward_final_heads_20260728_v1"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(_parser())
    print(
        json.dumps(
            {
                "status": result["status"],
                "training_label_end_exclusive_utc": str(
                    result["training_label_end_exclusive_utc"]
                ),
                "rows": {
                    side: report["rows"]
                    for side, report in result["sides"].items()
                },
            },
            indent=2,
        )
    )
