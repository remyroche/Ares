#!/usr/bin/env python3
"""Retrain an alternative meta model with causal residual-surprise features.

The current base predictions, current meta model, label geometry, and current
meta LightGBM parameters remain frozen.  Only the meta feature universe and
the canonical lgbm_pipeline feature selection are changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.meta_residual_archetypes import (
    OUTCOME_COLUMNS,  # noqa: E402
)
from scripts.report_meta_residual_historical_rank import (  # noqa: E402
    _calendar as historical_calendar,
)
from scripts.report_meta_residual_historical_rank import (
    _metrics as historical_metrics,
)
from scripts.report_meta_residual_historical_rank import (
    _true_monday_week_start,
    _walkforward_ranks,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    AE_GMM_HINTS,
    DEFAULT_OUT_DIR,
    _arm_candidate_features,
    _downcast,
    _merge_residual_features,
    _select_arm_features,
    _write_json,
    metrics_by_scope,
    surprise_calendar,
    train_arm_oos,
)

ARM = "lifecycle_residual_surprise_head_retrained"
HEAD_CACHE = "residual_walkforward_surprise_head_pca8_clip8.parquet"
EVAL_MONTHS_WITH_BURNIN = ("2026-03", "2026-04", "2026-05", "2026-06")
OOS_MONTHS = ("2026-04", "2026-05", "2026-06")


def _safe_json(value: Any) -> Any:
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_dataset(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data["archetype_policy_key"] = (
        data.get("archetype_policy_key", "missing").astype(str).fillna("missing")
    )
    data = (
        _downcast(data)
        .sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
        .reset_index(drop=True)
    )
    head = pd.read_parquet(root / "cache" / HEAD_CACHE)
    head["__ts__"] = pd.to_datetime(head["__ts__"], utc=True, errors="coerce")
    data = _merge_residual_features(data, head)
    del head
    gc.collect()
    return data, manifest


def _candidate_features(
    data: pd.DataFrame,
    reference_selected: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    lifecycle = [
        str(name)
        for name in CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", [])
        if str(name) in data.columns
    ]
    base = _arm_candidate_features(
        "lifecycle_only", data, reference_selected, lifecycle
    )
    existing_aegmm = [
        str(name)
        for name in data.columns
        if any(token.lower() in str(name).lower() for token in AE_GMM_HINTS)
        and str(name) not in OUTCOME_COLUMNS
    ]
    surprise = [
        str(name) for name in data.columns if str(name).startswith("meta_resid_")
    ]
    candidates = list(dict.fromkeys([*base, *existing_aegmm, *surprise]))
    families = {
        "reference_and_lifecycle": list(base),
        "existing_base_ae_gmm": existing_aegmm,
        "walkforward_surprise_heads": surprise,
    }
    return candidates, families


def _overall_top10(metrics: pd.DataFrame, arm: str) -> dict[str, Any]:
    rows = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(arm)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _mean_abs_autocorr(autocorr: pd.DataFrame, selector: str) -> float:
    values = pd.to_numeric(
        autocorr.loc[autocorr["selector"].eq(selector), "surprise_autocorr_lag1"],
        errors="coerce",
    )
    return float(values.abs().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--reuse-selection-from",
        default="",
        help="Reuse another arm's selected_features.csv instead of rerunning selection.",
    )
    parser.add_argument(
        "--force-surprise-heads",
        action="store_true",
        help="Append all five causal surprise-head outputs after the reused/selected contract.",
    )
    parser.add_argument(
        "--force-feature-selection",
        action="store_true",
        help="Remove only this arm's cached feature-selection report before fitting.",
    )
    args = parser.parse_args()

    root = args.root
    arm = str(args.arm)
    arm_dir = root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    selected_path = arm_dir / "selected_features.csv"
    if args.force_feature_selection and selected_path.exists():
        selected_path.unlink()

    data, dataset_manifest = _load_dataset(root)
    reference_selected = [
        str(name) for name in dataset_manifest["reference_selected_features"]
    ]
    params = dict(dataset_manifest["reference_model_params"])
    candidates, candidate_families = _candidate_features(data, reference_selected)
    reused_from = str(args.reuse_selection_from).strip()
    if reused_from:
        source_path = root / reused_from / "selected_features.csv"
        selection_rows = pd.read_csv(source_path)
        selected_mask = (
            selection_rows.get("selected", pd.Series(True, index=selection_rows.index))
            .fillna(False)
            .astype(bool)
        )
        selected = selection_rows.loc[selected_mask, "feature"].astype(str).tolist()
    else:
        selected, selection_rows = _select_arm_features(
            arm=arm,
            data=data,
            candidates=candidates,
            output_dir=root,
            seed=int(args.seed),
        )
    forced_heads = []
    if args.force_surprise_heads:
        forced_heads = list(candidate_families["walkforward_surprise_heads"])
        newly_forced = [name for name in forced_heads if name not in selected]
        selected = list(dict.fromkeys([*selected, *forced_heads]))
        if newly_forced:
            forced_rows = pd.DataFrame(
                {
                    "feature": newly_forced,
                    "selected": True,
                    "rank": np.arange(
                        len(selection_rows) + 1,
                        len(selection_rows) + 1 + len(newly_forced),
                    ),
                    "score": np.nan,
                    "feature_selection_method": "forced_residual_support_ablation",
                    "feature_selection_status": "support_ablation_only",
                }
            )
            selection_rows = pd.concat(
                [selection_rows, forced_rows], ignore_index=True, sort=False
            )
    selection_rows.to_csv(selected_path, index=False)

    all_predictions, fold_manifest = train_arm_oos(
        arm=arm,
        data=data,
        selected_features=selected,
        params=params,
        output_dir=root,
        seed=int(args.seed) + 1009,
        eval_months=EVAL_MONTHS_WITH_BURNIN,
        artifact_tag="with_march_burnin",
    )
    burnin = all_predictions[
        all_predictions["calendar_month"].astype(str).eq("2026-03")
    ].copy()
    oos = all_predictions[
        all_predictions["calendar_month"].astype(str).isin(OOS_MONTHS)
    ].copy()
    if burnin.empty or oos.empty:
        raise RuntimeError("Expected March burn-in and April-June OOS predictions.")
    burnin.to_parquet(
        arm_dir / "burnin_predictions_march.parquet", index=False, compression="zstd"
    )
    oos.to_parquet(arm_dir / "oos_predictions.parquet", index=False, compression="zstd")

    # Batch rank is retained as a diagnostic. The deployment-aligned result uses
    # a score CDF fitted only on prior months and carried forward month by month.
    batch_metrics = metrics_by_scope(oos, arm)
    batch_calendar, batch_autocorr, batch_events = surprise_calendar(oos, arm)
    batch_metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
    batch_calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
    batch_autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
    batch_events.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)

    ranked, rank_folds = _walkforward_ranks(burnin, oos)
    ranked["week_start"] = _true_monday_week_start(ranked["__ts__"])
    hist_metrics = historical_metrics(ranked, arm)
    hist_calendar, hist_autocorr, hist_events = historical_calendar(ranked, arm)
    hist_dir = root / f"historical_rank_oos_{arm}"
    hist_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(
        hist_dir / "oos_predictions_historical_rank.parquet",
        index=False,
        compression="zstd",
    )
    hist_metrics.to_csv(hist_dir / "metrics_by_scope.csv", index=False)
    hist_calendar.to_csv(hist_dir / "hit_surprise_calendar.csv", index=False)
    hist_autocorr.to_csv(hist_dir / "hit_surprise_autocorrelation.csv", index=False)
    hist_events.to_csv(hist_dir / "high_surprise_period_comparison.csv", index=False)

    selected_set = set(selected)
    family_selection = {
        family: {
            "candidate_count": len(names),
            "selected": [name for name in names if name in selected_set],
            "selected_count": sum(name in selected_set for name in names),
        }
        for family, names in candidate_families.items()
    }
    manifest = {
        "schema": "train_meta_residual_surprise_head_retrained_v1",
        "arm": arm,
        "current_base_model_retrained": False,
        "current_meta_model_overwritten": False,
        "target": "same base economic soft-binary label as current meta",
        "model_params": params,
        "model_params_source": dataset_manifest.get("reference_manifest"),
        "candidate_feature_count": len(candidates),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "feature_families": family_selection,
        "feature_selection": {
            "pipeline": "canonical lgbm_pipeline staged univariate + ReliefF + correlation pruning + iterative MDA",
            "requested_top_n": 0,
            "automatic_feature_count": True,
            "fit_end": "2026-02-28",
            "validation_month": "2026-03",
            "fit_once": True,
            "reused_from_arm": reused_from or None,
            "forced_surprise_heads": bool(args.force_surprise_heads),
            "forced_surprise_head_features": forced_heads,
        },
        "folds": fold_manifest.get("folds", []),
        "historical_rank_folds": rank_folds,
        "batch_top10": _overall_top10(batch_metrics, arm),
        "historical_top10": _overall_top10(hist_metrics, arm),
        "historical_current_top10": _overall_top10(hist_metrics, "current_reference"),
        "batch_mean_abs_surprise_autocorr_lag1": _mean_abs_autocorr(
            batch_autocorr, arm
        ),
        "historical_mean_abs_surprise_autocorr_lag1": _mean_abs_autocorr(
            hist_autocorr, arm
        ),
        "historical_current_mean_abs_surprise_autocorr_lag1": _mean_abs_autocorr(
            hist_autocorr, "current_reference"
        ),
        "leakage_contract": {
            "surprise_features": "monthly walk-forward models fitted only on rows before each feature month",
            "feature_selection": "through February with March validation; April-June excluded",
            "model_folds": "expanding train before each scored month",
            "historical_rank": "prior score CDF only",
            "outcomes_at_inference": False,
        },
    }
    _write_json(arm_dir / "manifest.json", _safe_json(manifest))
    _write_json(
        hist_dir / "manifest.json",
        _safe_json(
            {
                "schema": "meta_residual_retrained_historical_rank_v1",
                "arm": arm,
                "rank_contract": "expanding_prior_score_cdf_by_side",
                "folds": rank_folds,
                "top10": manifest["historical_top10"],
                "current_top10": manifest["historical_current_top10"],
                "current_meta_model_overwritten": False,
            }
        ),
    )
    print(json.dumps(_safe_json(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
