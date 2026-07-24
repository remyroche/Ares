#!/usr/bin/env python3
"""Compare recovered meta models with the raw and adjusted V9 references.

The report keeps artifact-native metrics separate from exact-row comparisons.
That matters because the historical V9 candidate ledger and the current top-30
base handoff do not contain the same rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["__ts__", "__symbol__", "side_name"]
OUTCOMES = [
    "ev_after_1pct",
    "clean_exec",
    "full_path_bad_mae_1r",
    "timeout",
]

DEFAULT_CURRENT = ROOT / "data_perp/artifacts/20260713_meta_reset_sharedchampion_oodp95_iqr_v1"
DEFAULT_EXACT_V9 = ROOT / "data_perp/reports/meta_v9_recovery_20260713/exact_v9_global_current_rows"
DEFAULT_V9_PARENT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_"
    "baseline_globaloverlay_sparse_shock_composite/"
    "oos_predictions_historical_rank.parquet"
)
DEFAULT_V9_FEATURE_MANIFEST = ROOT / (
    "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_"
    "mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/"
    "train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_"
    "oos15_top30_hpo45k_20260706_v5/best_full_oos_fixedfs_streamed_v1/manifest.json"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_path(run_dir: Path) -> Path:
    return run_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet"


def _resolve_hpo_deployment(root: Path) -> tuple[Path, dict[str, Any]]:
    root_manifest = _read_manifest(root / "manifest.json")
    deployment = (
        root_manifest.get("best_full_oos_manifest")
        or root_manifest.get("best_trial_manifest")
    )
    if not deployment:
        raise ValueError(f"No deployment manifest recorded in {root / 'manifest.json'}")
    run_dir = Path(str(deployment))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return run_dir, _read_manifest(run_dir / "manifest.json")


def _load_score_frame(path: Path, score_col: str) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    wanted = list(dict.fromkeys(KEYS + ["calendar_month", "week_start", score_col] + OUTCOMES))
    frame = pd.read_parquet(path, columns=[col for col in wanted if col in available])
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if "calendar_month" not in frame:
        frame["calendar_month"] = frame["__ts__"].dt.strftime("%Y-%m")
    if "week_start" not in frame:
        frame["week_start"] = (
            frame["__ts__"].dt.normalize()
            - pd.to_timedelta(frame["__ts__"].dt.weekday, unit="D")
        )
    frame = frame.loc[frame["calendar_month"].isin(["2026-04", "2026-05", "2026-06"])].copy()
    frame["model_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    return frame


def _top10_mask(frame: pd.DataFrame) -> pd.Series:
    rank = frame.groupby("calendar_month", sort=False)["model_score"].rank(
        pct=True, method="average"
    )
    return rank.ge(0.90) & frame["model_score"].notna()


def _metric_rows(frame: pd.DataFrame, *, model: str, scope: str) -> list[dict[str, Any]]:
    selected = _top10_mask(frame)
    work = frame.loc[selected].copy()
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", work)]
    groups.extend((str(month), part) for month, part in work.groupby("calendar_month", sort=True))
    for month, part in groups:
        weekly = part.groupby("week_start", sort=True)["ev_after_1pct"].mean()
        rows.append(
            {
                "scope": scope,
                "model": model,
                "month": month,
                "candidate_rows": int(len(frame) if month == "all" else frame["calendar_month"].eq(month).sum()),
                "selected_rows": int(len(part)),
                "mean_ev_after_1pct": float(pd.to_numeric(part.get("ev_after_1pct"), errors="coerce").mean()),
                "clean_exec_precision": float(pd.to_numeric(part.get("clean_exec"), errors="coerce").mean()),
                "full_path_bad_mae_rate": float(pd.to_numeric(part.get("full_path_bad_mae_1r"), errors="coerce").mean()),
                "timeout_rate": float(pd.to_numeric(part.get("timeout"), errors="coerce").mean()),
                "worst_week_ev_after_1pct": float(weekly.min()) if len(weekly) else float("nan"),
            }
        )
    return rows


def _join_scores(
    base: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    *,
    base_name: str,
) -> pd.DataFrame:
    keep = KEYS + ["calendar_month", "week_start"] + OUTCOMES
    merged = base[keep + ["model_score"]].rename(
        columns={"model_score": f"score__{base_name}"}
    )
    for name, frame in sources.items():
        score = frame[KEYS + ["model_score"]].rename(columns={"model_score": f"score__{name}"})
        merged = merged.merge(score, on=KEYS, how="inner", validate="one_to_one")
    return merged


def _feature_rows(name: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    union = [str(v) for v in manifest.get("selected_feature_union", []) or []]
    by_side = manifest.get("selected_features_by_side", {}) or {}
    rows: list[dict[str, Any]] = []
    for feature in union:
        rows.append(
            {
                "model": name,
                "feature": feature,
                "selected_long": feature in set(by_side.get("long", union)),
                "selected_short": feature in set(by_side.get("short", union)),
            }
        )
    return rows


def _parameter_rows(name: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    params = manifest.get("regressor_params") or manifest.get("classifier_params") or {}
    by_side = manifest.get("model_params_by_side", {}) or {}
    rows = [{"model": name, "side": "shared", "parameter": k, "value": v} for k, v in sorted(params.items())]
    for side, side_params in sorted(by_side.items()):
        rows.extend(
            {"model": name, "side": side, "parameter": k, "value": v}
            for k, v in sorted((side_params or {}).items())
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-hpo-root", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--exact-v9-run", type=Path, default=DEFAULT_EXACT_V9)
    parser.add_argument("--v9-parent", type=Path, default=DEFAULT_V9_PARENT)
    parser.add_argument("--v9-feature-manifest", type=Path, default=DEFAULT_V9_FEATURE_MANIFEST)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    new_dir, new_manifest = _resolve_hpo_deployment(args.new_hpo_root)
    current_manifest = _read_manifest(args.current_run / "manifest.json")
    exact_manifest = _read_manifest(args.exact_v9_run / "manifest.json")
    v9_feature_manifest = _read_manifest(args.v9_feature_manifest)

    frames = {
        "current_meta": _load_score_frame(_prediction_path(args.current_run), "score_meta_base_soft_label"),
        "exact_v9_contract_refit": _load_score_frame(_prediction_path(args.exact_v9_run), "score_meta_base_soft_label"),
        "recovered_meta": _load_score_frame(_prediction_path(new_dir), "score_meta_base_soft_label"),
        "v9_parent_adjusted": _load_score_frame(args.v9_parent, "score_adjusted"),
    }
    base_frame = _load_score_frame(_prediction_path(new_dir), "score_base")

    native_rows: list[dict[str, Any]] = []
    native_rows.extend(_metric_rows(base_frame, model="base_score", scope="current_native"))
    for name in ("current_meta", "exact_v9_contract_refit", "recovered_meta"):
        native_rows.extend(_metric_rows(frames[name], model=name, scope="current_native"))
    native_rows.extend(
        _metric_rows(frames["v9_parent_adjusted"], model="v9_parent_adjusted", scope="v9_native")
    )
    pd.DataFrame(native_rows).to_csv(args.out_dir / "metrics_native.csv", index=False)

    current_common = _join_scores(
        base_frame,
        {name: frames[name] for name in ("current_meta", "exact_v9_contract_refit", "recovered_meta")},
        base_name="base",
    )
    common_rows: list[dict[str, Any]] = []
    for name in ("base", "current_meta", "exact_v9_contract_refit", "recovered_meta"):
        local = current_common.rename(columns={f"score__{name}": "model_score"})
        common_rows.extend(_metric_rows(local, model=name, scope="exact_current_rows"))
    pd.DataFrame(common_rows).to_csv(args.out_dir / "metrics_exact_current_rows.csv", index=False)

    v9_overlap = _join_scores(
        frames["v9_parent_adjusted"],
        {
            "base": base_frame,
            "current_meta": frames["current_meta"],
            "exact_v9_contract_refit": frames["exact_v9_contract_refit"],
            "recovered_meta": frames["recovered_meta"],
        },
        base_name="v9_parent_adjusted",
    )
    overlap_rows: list[dict[str, Any]] = []
    for name in ("v9_parent_adjusted", "base", "current_meta", "exact_v9_contract_refit", "recovered_meta"):
        score_col = f"score__{name}"
        if score_col not in v9_overlap:
            continue
        local = v9_overlap.rename(columns={score_col: "model_score"})
        overlap_rows.extend(_metric_rows(local, model=name, scope="exact_v9_overlap_rows"))
    pd.DataFrame(overlap_rows).to_csv(args.out_dir / "metrics_exact_v9_overlap_rows.csv", index=False)

    feature_rows = []
    feature_rows.extend(_feature_rows("v9_reference_55", v9_feature_manifest))
    feature_rows.extend(_feature_rows("current_meta", current_manifest))
    feature_rows.extend(_feature_rows("recovered_meta", new_manifest))
    pd.DataFrame(feature_rows).to_csv(args.out_dir / "feature_comparison_long.csv", index=False)

    parameter_rows = []
    parameter_rows.extend(_parameter_rows("v9_reference_55", v9_feature_manifest))
    parameter_rows.extend(_parameter_rows("current_meta", current_manifest))
    parameter_rows.extend(_parameter_rows("recovered_meta", new_manifest))
    pd.DataFrame(parameter_rows).to_csv(args.out_dir / "parameter_comparison_long.csv", index=False)

    manifest = {
        "schema": "meta_v9_recovery_comparison_v1",
        "new_hpo_root": args.new_hpo_root,
        "new_deployment_dir": new_dir,
        "current_run": args.current_run,
        "exact_v9_run": args.exact_v9_run,
        "v9_parent": args.v9_parent,
        "topk_contract": "top 10% independently within each calendar month; exact-row reports use inner joins",
        "validation_classification": (
            "policy-OOS with accepted feature-selection/HPO calibration overlap; "
            "monthly model fits remain expanding-window OOS"
        ),
        "outputs": sorted(str(path) for path in args.out_dir.glob("*.csv")),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
