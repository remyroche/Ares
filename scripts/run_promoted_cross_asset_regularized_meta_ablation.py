#!/usr/bin/env python3
"""Run regularized train-meta smoke profiles on promoted cross-asset features.

The promoted cross-asset risk features work best when consumed directly by the
meta learner.  Rule overlays underperformed, so this script tests whether a
more regularized meta learner can keep the top-k lift while reducing damaged
cell/path-quality behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_s52_train_meta_regime_handoff_smoke as smoke  # noqa: E402


DEFAULT_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_xmarket_v1"
)
DEFAULT_BASELINE_SMOKE_DIR = DEFAULT_ROOT / "train_meta_smoke_baseline_for_promoted_compare_v2"
DEFAULT_HANDOFF_DIR = DEFAULT_ROOT / "train_meta_handoff_promoted_cross_asset_v1"
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "regularized_meta_ablation_v1"

PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "default": {},
    "reg_mid": {
        "classifier": {
            "n_estimators": 160,
            "learning_rate": 0.030,
            "num_leaves": 11,
            "min_child_samples": 70,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.50,
            "reg_lambda": 18.0,
        },
        "regressor": {
            "n_estimators": 180,
            "learning_rate": 0.030,
            "num_leaves": 11,
            "min_child_samples": 70,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.50,
            "reg_lambda": 20.0,
        },
    },
    "reg_high": {
        "classifier": {
            "n_estimators": 130,
            "learning_rate": 0.025,
            "num_leaves": 7,
            "min_child_samples": 110,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.65,
            "reg_alpha": 1.25,
            "reg_lambda": 35.0,
        },
        "regressor": {
            "n_estimators": 150,
            "learning_rate": 0.025,
            "num_leaves": 7,
            "min_child_samples": 110,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "reg_alpha": 1.25,
            "reg_lambda": 40.0,
        },
    },
    "reg_high_subsample": {
        "classifier": {
            "n_estimators": 120,
            "learning_rate": 0.025,
            "num_leaves": 7,
            "min_child_samples": 140,
            "subsample": 0.65,
            "subsample_freq": 1,
            "colsample_bytree": 0.55,
            "reg_alpha": 2.0,
            "reg_lambda": 50.0,
        },
        "regressor": {
            "n_estimators": 140,
            "learning_rate": 0.025,
            "num_leaves": 7,
            "min_child_samples": 140,
            "subsample": 0.65,
            "colsample_bytree": 0.55,
            "reg_alpha": 2.0,
            "reg_lambda": 55.0,
        },
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _fit_classifier_factory(params: dict[str, Any]):
    def _fit_classifier(x: pd.DataFrame, y: pd.Series, train: pd.DataFrame, seed: int) -> Any:
        target = smoke._num(y).fillna(0.0).astype(int)
        if int(target.nunique(dropna=True)) < 2:
            return None
        weights = smoke._classification_weights(target, train)
        if smoke._LIGHTGBM_AVAILABLE and smoke.LGBMClassifier is not None:
            model = smoke.LGBMClassifier(
                objective="binary",
                random_state=int(seed),
                n_jobs=2,
                verbosity=-1,
                **params,
            )
            model.fit(x, target, sample_weight=weights)
            return model
        model = smoke.ExtraTreesClassifier(
            n_estimators=180,
            max_depth=5,
            min_samples_leaf=max(25, int(params.get("min_child_samples", 80) // 3)),
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=2,
        )
        model.fit(x, target, sample_weight=weights)
        return model

    return _fit_classifier


def _fit_regressor_factory(params: dict[str, Any]):
    def _fit_regressor(x: pd.DataFrame, y: pd.Series, train: pd.DataFrame, seed: int) -> Any:
        target = smoke._num(y).replace([np.inf, -np.inf], np.nan)
        valid = target.notna()
        if int(valid.sum()) < 50 or float(target.loc[valid].std()) <= 1e-12:
            return None
        weights = smoke._regression_weights(target.loc[valid], train.loc[valid])
        if smoke._LIGHTGBM_AVAILABLE and smoke.LGBMRegressor is not None:
            model = smoke.LGBMRegressor(
                objective="regression",
                random_state=int(seed),
                n_jobs=2,
                verbosity=-1,
                **params,
            )
            model.fit(x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
            return model
        model = smoke.ExtraTreesRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=max(25, int(params.get("min_child_samples", 80) // 3)),
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=2,
        )
        model.fit(x.loc[valid], target.loc[valid].astype(np.float32), sample_weight=weights)
        return model

    return _fit_regressor


def _run_profile(
    *,
    profile_name: str,
    profile: dict[str, dict[str, Any]],
    handoff_dir: Path,
    out_dir: Path,
    frontier: str,
    train_scope: str,
    seed: int,
) -> dict[str, Any]:
    profile_out = out_dir / profile_name
    if profile_out.exists():
        shutil.rmtree(profile_out)
    if profile_name == "default":
        return smoke.run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=None,
            out_dir=profile_out,
            frontier=frontier,
            seed=seed,
            train_scope=train_scope,
        )
    old_classifier = smoke._fit_classifier
    old_regressor = smoke._fit_regressor
    try:
        smoke._fit_classifier = _fit_classifier_factory(profile["classifier"])  # type: ignore[assignment]
        smoke._fit_regressor = _fit_regressor_factory(profile["regressor"])  # type: ignore[assignment]
        return smoke.run_smoke(
            handoff_dir=handoff_dir,
            ledger_path=None,
            out_dir=profile_out,
            frontier=frontier,
            seed=seed,
            train_scope=train_scope,
        )
    finally:
        smoke._fit_classifier = old_classifier  # type: ignore[assignment]
        smoke._fit_regressor = old_regressor  # type: ignore[assignment]


def _comparison_rows(profile_manifests: dict[str, dict[str, Any]], baseline_manifest: dict[str, Any]) -> pd.DataFrame:
    baseline_best = baseline_manifest.get("best_selector") or {}
    rows: list[dict[str, Any]] = []

    def metric(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key))
        except Exception:
            return float("nan")

    for profile, manifest in profile_manifests.items():
        best = manifest.get("best_selector") or {}
        row = {
            "profile": profile,
            "selector": best.get("selector"),
            "status": best.get("meta_smoke_status"),
            "mean_keep010_exec_margin": metric(best, "mean_keep010_exec_margin"),
            "mean_keep010_clean_exec_precision": metric(best, "mean_keep010_clean_exec_precision"),
            "mean_keep010_full_path_bad_mae": metric(best, "mean_keep010_full_path_bad_mae"),
            "mean_keep010_timeout": metric(best, "mean_keep010_timeout"),
            "mean_keep010_oracle_recall": metric(best, "mean_keep010_oracle_recall"),
            "mean_keep030_exec_margin": metric(best, "mean_keep030_exec_margin"),
            "mean_keep030_clean_exec_precision": metric(best, "mean_keep030_clean_exec_precision"),
            "mean_keep030_full_path_bad_mae": metric(best, "mean_keep030_full_path_bad_mae"),
            "mean_keep030_timeout": metric(best, "mean_keep030_timeout"),
            "mean_keep030_oracle_recall": metric(best, "mean_keep030_oracle_recall"),
            "mean_ap_clean_exec": metric(best, "mean_ap_clean_exec"),
            "mean_auc_clean_exec": metric(best, "mean_auc_clean_exec"),
        }
        for key in list(row):
            if key.startswith("mean_"):
                row[f"delta_vs_baseline__{key}"] = float(row[key]) - metric(baseline_best, key)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["regularized_gate_status"] = np.where(
        (out["delta_vs_baseline__mean_keep010_exec_margin"] > 0.0)
        & (out["delta_vs_baseline__mean_keep010_clean_exec_precision"] >= 0.0)
        & (out["delta_vs_baseline__mean_keep010_full_path_bad_mae"] <= 0.0)
        & (out["delta_vs_baseline__mean_keep030_full_path_bad_mae"] <= 0.010)
        & (out["delta_vs_baseline__mean_ap_clean_exec"] >= 0.0),
        "candidate_for_deeper_meta_eval",
        "diagnostic_or_fail",
    )
    return out.sort_values(
        [
            "regularized_gate_status",
            "mean_keep010_exec_margin",
            "mean_keep010_full_path_bad_mae",
        ],
        ascending=[True, False, True],
    )


def _write_markdown(out_dir: Path, manifest: dict[str, Any], comparison: pd.DataFrame) -> Path:
    lines = [
        "# Promoted Cross-Asset Regularized Meta Ablation",
        "",
        "## Verdict",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- best profile: `{manifest.get('best_profile')}`",
        f"- baseline smoke: `{manifest.get('baseline_smoke_dir')}`",
        "",
        "## Comparison",
        "",
    ]
    if comparison.empty:
        lines.append("_No profile rows._")
    else:
        display_cols = [
            "profile",
            "selector",
            "regularized_gate_status",
            "mean_keep010_exec_margin",
            "mean_keep010_clean_exec_precision",
            "mean_keep010_full_path_bad_mae",
            "mean_keep010_oracle_recall",
            "mean_keep030_exec_margin",
            "mean_keep030_full_path_bad_mae",
            "mean_ap_clean_exec",
        ]
        lines.append(comparison[[col for col in display_cols if col in comparison.columns]].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a train-meta smoke profile search over model regularization only. It does not select production thresholds and is not frozen replay evidence.",
        ]
    )
    path = out_dir / "regularized_meta_ablation.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_ablation(
    *,
    handoff_dir: Path,
    baseline_smoke_dir: Path,
    out_dir: Path,
    frontier: str,
    train_scope: str,
    seed: int,
    profiles: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_manifests: dict[str, dict[str, Any]] = {}
    for profile_name in profiles:
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile {profile_name!r}; available={sorted(PROFILES)}")
        profile_manifests[profile_name] = _run_profile(
            profile_name=profile_name,
            profile=PROFILES[profile_name],
            handoff_dir=handoff_dir,
            out_dir=out_dir,
            frontier=frontier,
            train_scope=train_scope,
            seed=seed,
        )
    baseline_manifest = _read_json(baseline_smoke_dir / "manifest.json")
    comparison = _comparison_rows(profile_manifests, baseline_manifest)
    comparison_path = out_dir / "regularized_meta_ablation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    best = comparison.iloc[0].to_dict() if not comparison.empty else {}
    status = str(best.get("regularized_gate_status") or "diagnostic_or_fail")
    manifest = {
        "generated_by": "run_promoted_cross_asset_regularized_meta_ablation",
        "handoff_dir": str(handoff_dir),
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "out_dir": str(out_dir),
        "frontier": frontier,
        "train_scope": train_scope,
        "seed": int(seed),
        "profiles": profiles,
        "profile_params": {name: PROFILES[name] for name in profiles},
        "status": status,
        "best_profile": best.get("profile"),
        "best_row": _json_safe(best),
        "outputs": {
            "comparison": str(comparison_path),
            "markdown": str(out_dir / "regularized_meta_ablation.md"),
            "json": str(out_dir / "manifest.json"),
        },
        "leakage_contract": {
            "split": "delegates to month-forward train_meta smoke",
            "profile_search_scope": "same baseline/promoted train_meta rows; compares regularization profiles only",
            "labels_used_for": "training labels and validation metrics inside month-forward smoke only",
        },
    }
    markdown = _write_markdown(out_dir, manifest, comparison)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frontier", choices=["top10", "top20", "top30"], default="top10")
    parser.add_argument("--train-scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--profiles", default="default,reg_mid,reg_high,reg_high_subsample")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profiles = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    manifest = run_ablation(
        handoff_dir=args.handoff_dir,
        baseline_smoke_dir=args.baseline_smoke_dir,
        out_dir=args.out_dir,
        frontier=args.frontier,
        train_scope=args.train_scope,
        seed=args.seed,
        profiles=profiles,
    )
    print(json.dumps(_json_safe({"event": "regularized_meta_ablation_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
