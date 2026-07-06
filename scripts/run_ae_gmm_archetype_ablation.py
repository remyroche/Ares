#!/usr/bin/env python3
"""Run no/old/path-aware AE-GMM archetype ablations for downstream smoke checks.

The script is intentionally a harness around the existing train-base smoke and
train-meta smoke scripts. It records exact commands and environment switches so
G0-G5 comparisons are reproducible and leakage/audit reviews can tell which
representation policy was used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_side_archetype_effect_matrix import run_report as run_side_effect_report  # noqa: E402
from scripts.report_ae_gmm_crossfit_contract import build_report as build_crossfit_contract_report  # noqa: E402
from scripts.report_ae_gmm_archetype_ablation_comparison import (  # noqa: E402
    build_comparison as build_ablation_comparison,
    write_report as write_ablation_comparison,
)
from scripts.report_ae_gmm_schema_audit import (  # noqa: E402
    build_schema_audit,
    write_outputs as write_schema_audit_outputs,
)
from scripts.report_ae_gmm_promotion_decision_pack import build_pack as build_promotion_decision_pack  # noqa: E402

DEFAULT_SELECTION_DIR = Path(
    "data_perp/reports/conditional_gmm_feature_selection_20260702_lowcost_strict_econ_target_wide_sidebalanced_hpo"
)
DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260702_211500_single_head_monthly_walkforward_bidirectional_sideaware_"
    "lowcost_strict_economic_target_labels/labels"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260629_050000")


ARMS: dict[str, dict[str, Any]] = {
    "G0": {
        "description": "no AE/GMM features",
        "disable_ae_gmm": True,
        "path_aware_hpo": False,
        "temporal_concentration_hpo": False,
        "include_cluster_id": False,
        "feature_policy": "none",
        "smoke_feature_policy": "all",
    },
    "G1": {
        "description": "old AE/GMM objective: economic/signature/stability/side/occupancy, no path/time HPO",
        "disable_ae_gmm": False,
        "path_aware_hpo": False,
        "temporal_concentration_hpo": False,
        "include_cluster_id": True,
        "feature_policy": "legacy_all_generated_features",
        "smoke_feature_policy": "all",
    },
    "G2": {
        "description": "path-aware AE/GMM objective",
        "disable_ae_gmm": False,
        "path_aware_hpo": True,
        "temporal_concentration_hpo": True,
        "include_cluster_id": True,
        "feature_policy": "path_time_aware_all_generated_features",
        "smoke_feature_policy": "all",
    },
    "G3": {
        "description": "long/short path-aware AE/GMM representation including generated cluster context",
        "disable_ae_gmm": False,
        "path_aware_hpo": True,
        "temporal_concentration_hpo": True,
        "include_cluster_id": True,
        "feature_policy": "path_time_aware_long_short_all_generated_features",
        "smoke_feature_policy": "all",
        "side_context_mode": "long_short",
    },
    "G4": {
        "description": "path-aware global plus long/short continuous features only, hard cluster IDs excluded",
        "disable_ae_gmm": False,
        "path_aware_hpo": True,
        "temporal_concentration_hpo": True,
        "include_cluster_id": False,
        "feature_policy": "long_short_continuous_only_no_cluster_id",
        "smoke_feature_policy": "continuous_no_cluster_id",
        "side_context_mode": "long_short",
    },
    "G5": {
        "description": "path-aware global plus long/short soft probabilities/distances/transition features, hard cluster IDs excluded",
        "disable_ae_gmm": False,
        "path_aware_hpo": True,
        "temporal_concentration_hpo": True,
        "include_cluster_id": False,
        "feature_policy": "long_short_soft_distance_transition_no_cluster_id",
        "smoke_feature_policy": "soft_distance_transition_no_cluster_id",
        "side_context_mode": "long_short",
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_selected_features(selection_dir: Path, limit: int) -> list[str]:
    candidates = [
        selection_dir / "conditional_gmm_training_feature_list.csv",
        selection_dir / "conditional_selected_features.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "used_by_model" in frame.columns:
            used = frame["used_by_model"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
            frame = frame[used].copy()
        if "selected_feature_position" in frame.columns:
            frame = frame.sort_values("selected_feature_position")
        if "feature" in frame.columns:
            features = [str(v) for v in frame["feature"].dropna().drop_duplicates().tolist()]
            return features[: int(limit)] if int(limit) > 0 else features
    raise FileNotFoundError(f"No selected feature list found under {selection_dir}")


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _write_feature_list(path: Path, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": list(features)}).to_csv(path, index=False)


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"cmd": cmd, "dry_run": bool(dry_run), "returncode": None}
    if dry_run:
        return record
    print(f"[run_ae_gmm_archetype_ablation] START {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, check=False)
    print(
        f"[run_ae_gmm_archetype_ablation] END returncode={int(proc.returncode)} {' '.join(cmd)}",
        flush=True,
    )
    record.update({"returncode": int(proc.returncode)})
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return record


def _arm_env(base_env: dict[str, str], arm: dict[str, Any]) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    env["EPM_AE_GMM_PATH_AWARE_HPO"] = "1" if bool(arm["path_aware_hpo"]) else "0"
    env["EPM_AE_GMM_TEMPORAL_CONCENTRATION_HPO"] = "1" if bool(arm["temporal_concentration_hpo"]) else "0"
    env["EPM_LGBM_AE_GMM_INCLUDE_CLUSTER_ID_MODEL_FEATURES"] = "1" if bool(arm["include_cluster_id"]) else "0"
    env["EPM_AE_GMM_SMOKE_FEATURE_POLICY"] = str(arm.get("smoke_feature_policy", "all"))
    env["EPM_AE_GMM_SIDE_CONTEXT_MODE"] = str(arm.get("side_context_mode", "off"))
    env["EPM_AE_GMM_CROSSFIT_TRAIN_FEATURES"] = "0"
    return env


def _smoke_cmd(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list: Path,
    output_dir: Path,
    arm: dict[str, Any],
    label_arms: str,
    weight_arms: str,
    seeds: str,
    top_fracs: str,
    train_lookback_months: int,
    target_symbol_count: int | None,
    spread_baseline_path: Path | None,
    max_feature_store_features: int,
    candidate_selectors: str,
    candidate_ledger_only: bool,
    candidate_ledger_fast_mode: bool,
    include_risk_selector_variants: bool,
    ae_max_iter: int,
    ae_max_train_rows: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_label_feature_store_model_smoke.py",
        "--labels-path",
        str(labels_path),
        "--feature-dir",
        str(feature_dir),
        "--feature-list-csv",
        str(feature_list),
        "--output-dir",
        str(output_dir),
        "--label-arms",
        str(label_arms),
        "--weight-arms",
        str(weight_arms),
        "--seeds",
        str(seeds),
        "--top-fracs",
        str(top_fracs),
        "--train-lookback-months",
        str(int(train_lookback_months)),
        "--max-feature-store-features",
        str(int(max_feature_store_features)),
        "--ae-gmm-state-feature-max-iter",
        str(int(ae_max_iter)),
        "--ae-gmm-state-feature-max-train-rows",
        str(int(ae_max_train_rows)),
    ]
    if spread_baseline_path is not None:
        cmd.extend(["--spread-baseline-path", str(spread_baseline_path)])
    if target_symbol_count is not None and spread_baseline_path is not None:
        cmd.extend(["--target-symbol-count", str(int(target_symbol_count))])
    if str(candidate_selectors).strip():
        cmd.extend(["--candidate-ledger-selector-names", str(candidate_selectors)])
    if include_risk_selector_variants:
        cmd.append("--include-risk-selector-variants")
    if candidate_ledger_only:
        cmd.append("--candidate-ledger-only")
    if candidate_ledger_fast_mode:
        cmd.append("--candidate-ledger-fast-mode")
    if bool(arm["disable_ae_gmm"]):
        cmd.append("--disable-ae-gmm-state-features")
    return cmd


def _best_utility_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "mean_u" not in frame.columns:
        return {}
    work = frame.copy()
    for col in (
        "mean_u",
        "worst_month_mean_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "final_oracle_recall",
        "positive_months",
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    sort_cols = [
        col
        for col in (
            "mean_u",
            "worst_month_mean_u",
            "final_oracle_recall",
            "bad_mae_1r_rate",
            "timeout_rate",
        )
        if col in work.columns
    ]
    if not sort_cols:
        return work.iloc[0].to_dict()
    ascending = [
        False if col in {"mean_u", "worst_month_mean_u", "final_oracle_recall"} else True
        for col in sort_cols
    ]
    return work.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].to_dict()


def _copy_best_row_metrics(prefix: str, row: dict[str, Any], out: dict[str, Any]) -> None:
    for col in (
        "selector_variant",
        "meta_variant",
        "keep_frac",
        "mean_u",
        "worst_month_mean_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "final_oracle_recall",
        "positive_months",
    ):
        if col not in row:
            continue
        value = row.get(col)
        if isinstance(value, (np.integer,)):
            out[f"{prefix}{col}"] = int(value)
        elif isinstance(value, (np.floating, float)):
            out[f"{prefix}{col}"] = float(value)
        else:
            out[f"{prefix}{col}"] = value


def _summarize_smoke(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = output_dir / "label_feature_store_model_smoke_manifest.json"
    aggregate_path = output_dir / "label_feature_store_model_smoke_aggregate.csv"
    diagnostics_path = output_dir / "label_feature_store_model_smoke_diagnostics.csv"
    candidate_ledger_path = output_dir / "label_feature_store_model_smoke_candidate_ledger.csv"
    out: dict[str, Any] = {
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "aggregate_path": str(aggregate_path) if aggregate_path.exists() else None,
        "diagnostics_path": str(diagnostics_path) if diagnostics_path.exists() else None,
        "candidate_ledger_path": str(candidate_ledger_path) if candidate_ledger_path.exists() else None,
    }
    if aggregate_path.exists():
        agg = pd.read_csv(aggregate_path)
        out["aggregate_rows"] = int(len(agg))
        _copy_best_row_metrics("best_row_", _best_utility_row(agg), out)
        for col in ("mean_u", "worst_month_mean_u", "final_oracle_recall"):
            if col in agg.columns:
                values = pd.to_numeric(agg[col], errors="coerce")
                if values.notna().any():
                    out[f"max_{col}"] = float(values.max())
    if diagnostics_path.exists():
        diag = pd.read_csv(diagnostics_path)
        out["diagnostic_rows"] = int(len(diag))
        for col in (
            "ae_gmm_state_features_enabled",
            "ae_gmm_state_train_feature_scope",
            "ae_gmm_state_validation_feature_scope",
            "ae_gmm_state_crossfit_coverage",
            "ae_gmm_state_path_cleanliness_score",
            "ae_gmm_state_temporal_concentration_score",
            "ae_gmm_state_feature_count",
            "ae_gmm_side_context_mode",
            "ae_gmm_side_context_feature_count",
        ):
            if col in diag.columns:
                vals = diag[col].dropna().astype(str if diag[col].dtype == object else float)
                out[f"diagnostic_{col}_sample"] = vals.head(5).tolist()
    return out


def _run_meta_smoke_for_arm(
    *,
    arm_dir: Path,
    candidate_ledger_path: Path,
    candidate_streams: list[str],
    keep_fracs: list[float],
    seeds: str,
    train_lookback_months: int,
    max_feature_store_features: int,
    max_side_share: float,
    min_train_rows: int,
) -> dict[str, Any]:
    if not candidate_ledger_path.exists() or candidate_ledger_path.stat().st_size <= 0:
        return {
            "enabled": True,
            "status": "skipped_empty_candidate_ledger",
            "candidate_ledger_path": str(candidate_ledger_path),
        }
    try:
        preview = pd.read_csv(candidate_ledger_path, nrows=5)
    except Exception as exc:
        return {
            "enabled": True,
            "status": "skipped_unreadable_candidate_ledger",
            "candidate_ledger_path": str(candidate_ledger_path),
            "error": str(exc),
        }
    if preview.empty:
        return {
            "enabled": True,
            "status": "skipped_empty_candidate_ledger",
            "candidate_ledger_path": str(candidate_ledger_path),
        }
    # Meta is meant to filter the base candidate stream using the same
    # inference-available AE/GMM context exported in the ledger. The meta smoke
    # module reads this policy at import time, so set defaults before import.
    os.environ.setdefault("EPM_META_CONTEXT_FEATURE_POLICY", "ae_gmm_only")
    os.environ.setdefault("EPM_META_CONTEXT_FEATURE_BLOCKS", "all")
    from scripts.run_gmm_train_meta_path_filter_smoke import run_meta_smoke  # noqa: WPS433

    manifest = run_meta_smoke(
        report_dir=ROOT,
        output_dir=arm_dir / "meta_path_filter_smoke",
        candidate_streams=list(candidate_streams),
        keep_fracs=list(keep_fracs),
        candidate_ledger_path=candidate_ledger_path,
        seeds=[int(part.strip()) for part in str(seeds).split(",") if part.strip()],
        train_lookback_months=int(train_lookback_months),
        max_feature_store_features=int(max_feature_store_features),
        max_side_share=float(max_side_share),
        min_train_rows=int(min_train_rows),
    )
    aggregate_path = Path(str(manifest.get("outputs", {}).get("aggregate", "")))
    summary: dict[str, Any] = {
        "enabled": True,
        "status": str(manifest.get("status", "unknown")),
        "manifest_path": str(manifest.get("outputs", {}).get("manifest")),
        "aggregate_path": str(aggregate_path) if aggregate_path else None,
        "selected_rows": int(manifest.get("selected_rows", 0) or 0),
        "simple_policy_handoff_rows": int(manifest.get("simple_policy_handoff_rows", 0) or 0),
        "best_candidate": manifest.get("best_candidate"),
    }
    if aggregate_path.exists():
        aggregate = pd.read_csv(aggregate_path)
        summary["aggregate_rows"] = int(len(aggregate))
        _copy_best_row_metrics("best_row_", _best_utility_row(aggregate), summary)
        for col in ("mean_u", "worst_month_mean_u", "final_oracle_recall"):
            if col in aggregate.columns:
                values = pd.to_numeric(aggregate[col], errors="coerce")
                if values.notna().any():
                    summary[f"max_{col}"] = float(values.max())
    return summary


def run_ablation(
    *,
    output_dir: Path,
    selection_dir: Path,
    labels_path: Path,
    feature_dir: Path,
    arms: list[str],
    feature_limit: int,
    execute: bool,
    label_arms: str,
    weight_arms: str,
    seeds: str,
    top_fracs: str,
    train_lookback_months: int,
    target_symbol_count: int | None,
    spread_baseline_path: Path | None,
    max_feature_store_features: int,
    candidate_selectors: str,
    candidate_ledger_only: bool,
    candidate_ledger_fast_mode: bool,
    include_risk_selector_variants: bool,
    ae_max_iter: int,
    ae_max_train_rows: int,
    build_effect_matrix: bool,
    run_meta_smoke: bool,
    meta_keep_fracs: list[float],
    meta_min_train_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = _read_selected_features(selection_dir, int(feature_limit))
    feature_list = output_dir / "ablation_feature_list.csv"
    _write_feature_list(feature_list, features)
    base_env = dict(os.environ)
    effective_candidate_selectors = str(candidate_selectors).strip()
    if bool(build_effect_matrix) and not effective_candidate_selectors:
        effective_candidate_selectors = "raw_utility"
    if bool(run_meta_smoke) and bool(include_risk_selector_variants) and not effective_candidate_selectors:
        effective_candidate_selectors = "s8_lgbm_utility_ranker_stageA_rerank_side_cap_70"
    arm_records: list[dict[str, Any]] = []
    for arm_name in arms:
        if arm_name not in ARMS:
            raise ValueError(f"Unknown arm {arm_name}; expected one of {sorted(ARMS)}")
        arm = dict(ARMS[arm_name])
        arm_dir = output_dir / arm_name.lower()
        arm_dir.mkdir(parents=True, exist_ok=True)
        env = _arm_env(base_env, arm)
        cmd = _smoke_cmd(
            labels_path=labels_path,
            feature_dir=feature_dir,
            feature_list=feature_list,
            output_dir=arm_dir,
            arm=arm,
            label_arms=label_arms,
            weight_arms=weight_arms,
            seeds=seeds,
            top_fracs=top_fracs,
            train_lookback_months=int(train_lookback_months),
            target_symbol_count=target_symbol_count,
            spread_baseline_path=spread_baseline_path,
            max_feature_store_features=int(max_feature_store_features),
            candidate_selectors=effective_candidate_selectors,
            candidate_ledger_only=bool(candidate_ledger_only),
            candidate_ledger_fast_mode=bool(candidate_ledger_fast_mode),
            include_risk_selector_variants=bool(include_risk_selector_variants),
            ae_max_iter=int(ae_max_iter),
            ae_max_train_rows=int(ae_max_train_rows),
        )
        command_record = _run(cmd, env=env, cwd=ROOT, dry_run=not bool(execute))
        summary = _summarize_smoke(arm_dir) if execute else {}
        crossfit_contract_manifest: dict[str, Any] | None = None
        effect_manifest: dict[str, Any] | None = None
        meta_manifest: dict[str, Any] | None = None
        candidate_path = summary.get("candidate_ledger_path")
        manifest_path = summary.get("manifest_path")
        if execute and manifest_path:
            try:
                crossfit_contract_manifest = build_crossfit_contract_report(
                    manifest_path=Path(str(manifest_path)),
                    diagnostics_path=Path(str(summary["diagnostics_path"]))
                    if summary.get("diagnostics_path")
                    else None,
                    candidate_ledger_path=Path(str(candidate_path)) if candidate_path else None,
                    output_dir=arm_dir / "crossfit_contract_audit",
                    min_global_coverage=0.60,
                    min_side_coverage=0.50,
                )
            except Exception as exc:
                crossfit_contract_manifest = {"enabled": False, "error": str(exc)}
        if execute and build_effect_matrix and candidate_path:
            candidate_file = Path(str(candidate_path))
            if candidate_file.exists() and candidate_file.stat().st_size > 0:
                try:
                    preview = pd.read_csv(candidate_file, nrows=5)
                    if not preview.empty:
                        effect_manifest = run_side_effect_report(
                            input_path=candidate_file,
                            output_dir=arm_dir / "side_archetype_effect_matrix",
                            rank_bands=5,
                            max_features=80,
                            min_support=25,
                            quantile=0.75,
                        )
                except Exception as exc:
                    effect_manifest = {"enabled": False, "error": str(exc)}
        if execute and run_meta_smoke and candidate_path:
            meta_manifest = _run_meta_smoke_for_arm(
                arm_dir=arm_dir,
                candidate_ledger_path=Path(str(candidate_path)),
                candidate_streams=[
                    item.strip()
                    for item in str(effective_candidate_selectors).split(",")
                    if item.strip()
                ],
                keep_fracs=list(meta_keep_fracs),
                seeds=str(seeds),
                train_lookback_months=int(train_lookback_months),
                max_feature_store_features=int(max_feature_store_features),
                max_side_share=0.70,
                min_train_rows=int(meta_min_train_rows),
            )
        arm_records.append(
            {
                "arm": arm_name,
                **arm,
                "output_dir": str(arm_dir),
                "env_overrides": {
                    key: env[key]
                    for key in (
                        "EPM_AE_GMM_PATH_AWARE_HPO",
                        "EPM_AE_GMM_TEMPORAL_CONCENTRATION_HPO",
                        "EPM_LGBM_AE_GMM_INCLUDE_CLUSTER_ID_MODEL_FEATURES",
                        "EPM_AE_GMM_SMOKE_FEATURE_POLICY",
                        "EPM_AE_GMM_SIDE_CONTEXT_MODE",
                        "EPM_AE_GMM_CROSSFIT_TRAIN_FEATURES",
                    )
                },
                "command": command_record,
                "summary": summary,
                "crossfit_contract_audit": crossfit_contract_manifest,
                "side_effect_matrix": effect_manifest,
                "meta_path_filter_smoke": meta_manifest,
            }
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "ae_gmm_archetype_ablation_v1",
        "execute": bool(execute),
        "selection_dir": str(selection_dir),
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list": str(feature_list),
        "feature_count": int(len(features)),
        "candidate_selectors": effective_candidate_selectors,
        "run_meta_smoke": bool(run_meta_smoke),
        "meta_keep_fracs": [float(v) for v in meta_keep_fracs],
        "meta_min_train_rows": int(meta_min_train_rows),
        "arms": arm_records,
        "next_validation": {
            "primary_delta": "G2_minus_G1 on downstream OOF/meta residual utility and paired portfolio replay",
            "side_effect_matrix": "run scripts/report_side_archetype_effect_matrix.py on each candidate ledger",
            "side_contract": "global side scope only: use side in {long, short}; no per-strategy/head grouping",
            "hard_gate_policy": "forbidden; use continuous features for threshold/size shadow diagnostics only",
        },
    }
    manifest_path = output_dir / "ae_gmm_archetype_ablation_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    if execute:
        post_reports: dict[str, Any] = {}
        try:
            comparison, comparison_payload = build_ablation_comparison(manifest_path)
            comparison_manifest = write_ablation_comparison(
                comparison,
                comparison_payload,
                output_dir / "comparison",
            )
            post_reports["comparison"] = comparison_manifest
        except Exception as exc:
            post_reports["comparison"] = {"error": str(exc)}
        try:
            schema_frame, schema_payload = build_schema_audit(manifest_path)
            schema_outputs = write_schema_audit_outputs(
                schema_frame,
                schema_payload,
                output_dir / "schema_audit",
            )
            post_reports["schema_audit"] = {**schema_payload, "outputs": schema_outputs}
        except Exception as exc:
            post_reports["schema_audit"] = {"error": str(exc)}
        try:
            comparison_path = output_dir / "comparison" / "ae_gmm_archetype_ablation_comparison.csv"
            schema_path = output_dir / "schema_audit" / "ae_gmm_schema_audit.csv"
            post_reports["decision_pack"] = build_promotion_decision_pack(
                manifest_path=manifest_path,
                comparison_path=comparison_path if comparison_path.exists() else None,
                schema_path=schema_path if schema_path.exists() else None,
                output_dir=output_dir / "decision_pack",
            )
        except Exception as exc:
            post_reports["decision_pack"] = {"error": str(exc)}
        manifest["post_reports"] = post_reports
        manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--arms", default="G0,G1,G2,G3,G4,G5")
    parser.add_argument("--feature-limit", type=int, default=80)
    parser.add_argument("--execute", action="store_true", help="Actually run child smoke commands. Default is dry-run manifest only.")
    parser.add_argument("--label-arms", default="S34_exec_guard_broad_policy")
    parser.add_argument("--weight-arms", default="W0_base")
    parser.add_argument("--seeds", default="913")
    parser.add_argument("--top-fracs", default="0.02")
    parser.add_argument("--train-lookback-months", type=int, default=3)
    parser.add_argument("--target-symbol-count", type=int, default=None)
    parser.add_argument("--spread-baseline-path", type=Path, default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=80)
    parser.add_argument("--candidate-selectors", default="")
    parser.add_argument("--candidate-ledger-only", action="store_true")
    parser.add_argument("--candidate-ledger-fast-mode", action="store_true")
    parser.add_argument(
        "--include-risk-selector-variants",
        action="store_true",
        help="Enable risk/ranker selector variants in the underlying base smoke.",
    )
    parser.add_argument("--ae-max-iter", type=int, default=8)
    parser.add_argument("--ae-max-train-rows", type=int, default=3000)
    parser.add_argument("--build-effect-matrix", action="store_true")
    parser.add_argument(
        "--run-meta-smoke",
        action="store_true",
        help="Run downstream train_meta-style path-filter smoke on each arm candidate ledger.",
    )
    parser.add_argument("--meta-keep-fracs", default="0.50,0.60,0.70,0.80")
    parser.add_argument("--meta-min-train-rows", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = [item.strip() for item in str(args.arms).split(",") if item.strip()]
    manifest = run_ablation(
        output_dir=args.output_dir,
        selection_dir=args.selection_dir,
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        arms=arms,
        feature_limit=int(args.feature_limit),
        execute=bool(args.execute),
        label_arms=str(args.label_arms),
        weight_arms=str(args.weight_arms),
        seeds=str(args.seeds),
        top_fracs=str(args.top_fracs),
        train_lookback_months=int(args.train_lookback_months),
        target_symbol_count=args.target_symbol_count,
        spread_baseline_path=args.spread_baseline_path,
        max_feature_store_features=int(args.max_feature_store_features),
        candidate_selectors=str(args.candidate_selectors),
        candidate_ledger_only=bool(args.candidate_ledger_only),
        candidate_ledger_fast_mode=bool(args.candidate_ledger_fast_mode),
        include_risk_selector_variants=bool(args.include_risk_selector_variants),
        ae_max_iter=int(args.ae_max_iter),
        ae_max_train_rows=int(args.ae_max_train_rows),
        build_effect_matrix=bool(args.build_effect_matrix),
        run_meta_smoke=bool(args.run_meta_smoke),
        meta_keep_fracs=_parse_float_csv(str(args.meta_keep_fracs), (0.50, 0.60, 0.70, 0.80)),
        meta_min_train_rows=int(args.meta_min_train_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
