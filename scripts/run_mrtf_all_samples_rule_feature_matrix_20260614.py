#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.run_top2_recency_pipeline_20260611 as pipeline
from scripts.run_mrtf_stage1_only_20260613 import (
    BASELINE_RUN_ID,
    _assert_same_strategies_as_baseline,
    _enable_full_lgbm_search,
)


ROOT = Path(__file__).resolve().parents[1]
RULE_CSV_DEFAULT = (
    ROOT
    / "data_perp"
    / "artifacts"
    / "20260523_015947"
    / "lgbm_based_mask_generation_v2_regen_20260529_1345_top30_perps"
    / "run_20260529_114456_069338"
    / "diversified_final_selection.csv"
)


def _variant_run_id(prefix: str, suffix: str) -> str:
    base = f"{prefix}_{suffix}"
    if os.environ.get("EPM_MRTF_FORCE_NEW_RUN_ID", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return base
    root = pipeline.DATA_ROOT / "artifacts"
    if not (root / base).exists():
        return base
    for i in range(2, 100):
        candidate = f"{base}_v{i}"
        if not (root / candidate).exists():
            return candidate
    raise RuntimeError(f"could not allocate run id for {base}")


def _matrix_prefix() -> str:
    return os.environ.get(
        "EPM_MRTF_ALLSAMPLES_MATRIX_PREFIX",
        "20260614_000000_top2_mrtf_allsamples",
    ).strip()


def _rule_csv() -> Path:
    return Path(
        os.environ.get("EPM_LGBM_RULE_MASK_FEATURES_CSV", str(RULE_CSV_DEFAULT))
    ).expanduser()


def _all_samples_env(
    *,
    run_id: str,
    base_winner: Path,
    meta_winner: Path,
    enable_rule_masks: bool,
) -> dict[str, str]:
    slice_plan_path = pipeline._build_recent_tail_slice_plan(
        pipeline.POLICY_SLICE_SOURCE_RUN_ID,
        run_id,
    )
    env = pipeline._train_env(
        run_id=run_id,
        label_source_run_id=pipeline.SOURCE_RUN_ID,
        preset_source_run_id=pipeline.SOURCE_RUN_ID,
        slice_plan_path=slice_plan_path,
        params_only=False,
        full_scope=False,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    env = _enable_full_lgbm_search(env)
    env.update(
        {
            "EPM_TRAIN_EXTEND_TO_LATEST": "1",
            "EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER": "1",
            "EPM_MR_TF_MASKS_ENABLED": "1",
            "EPM_MR_TF_OPTUNA_ENABLED": "1",
            "EPM_MR_TF_OPTUNA_TRIALS": "300",
            "EPM_MR_TF_OPTUNA_PATIENCE": "40",
            "EPM_MR_TF_OPTUNA_USE_NUMBA": "1",
            "EPM_LGBM_RULE_MASK_FEATURES_ENABLED": "1" if enable_rule_masks else "0",
            "EPM_LGBM_RULE_MASK_FEATURES_CSV": str(_rule_csv()),
            "EPM_LGBM_RULE_MASK_FEATURES_SIDE_FILTER": "0",
            # Keep the matrices manageable while still using the full label rows.
            "EPM_TOP2_LGBM_RACE_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LGBM_RACE_MAX_ROWS", "80000"
            ),
            "EPM_TOP2_LGBM_HPO_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LGBM_HPO_MAX_ROWS", "8000"
            ),
            "EPM_TOP2_LABEL_HPO_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LABEL_HPO_MAX_ROWS", "8000"
            ),
            "EPM_TOP2_LABEL_HPO_ELECTION_MAX_ROWS": os.environ.get(
                "EPM_TOP2_LABEL_HPO_ELECTION_MAX_ROWS", "25000"
            ),
        }
    )
    env.pop("EPM_TRAIN_RECENT_DAYS", None)
    return env


def _policy_cmd(run_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "extreme_price_movements/simple_policy_optimiser.py",
        "--data_root",
        "data_perp",
        "--run_id",
        run_id,
        "--market-mode",
        "perps",
        "--strategy-ids",
        ",".join(f"{row['side']}_{row['strategy_id']}" for row in pipeline.TOP2),
    ]


def _diagnostic_policy_env(env: dict[str, str]) -> dict[str, str]:
    """Use final-fit policy-slice predictions for all-samples diagnostics.

    These variants intentionally train on the full available label history, so
    the strict train-meta-frozen policy-OOS handoff is not valid. Keep that
    guard intact and make this launcher opt into simple_policy_optimiser's
    explicit diagnostic final-fit path instead.
    """
    out = dict(env)
    out["EPM_SIMPLE_POLICY_USE_POLICY_OOS_PREDICTIONS"] = "0"
    out["EPM_SIMPLE_POLICY_USE_PRECOMPUTED_META_OOF"] = "0"
    out["EPM_SIMPLE_POLICY_ALLOW_FINAL_FIT_POLICY_GENERATION"] = "1"
    out["EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY"] = "1"
    out["EPM_SIMPLE_POLICY_SOURCE_NOTE"] = (
        "diagnostic_final_fit_policy_slice_predictions_all_samples_training"
    )
    return out


def _run_variant(
    *,
    run_id: str,
    suffix: str,
    enable_rule_masks: bool,
    base_winner: Path,
    meta_winner: Path,
) -> None:
    marker = pipeline.DATA_ROOT / "artifacts" / run_id / "mrtf_all_samples_complete.json"
    if marker.exists():
        pipeline._append(f"Variant already complete: {run_id}")
        return
    env = _all_samples_env(
        run_id=run_id,
        base_winner=base_winner,
        meta_winner=meta_winner,
        enable_rule_masks=enable_rule_masks,
    )
    pipeline._append(
        f"Variant {suffix} starting run_id={run_id} "
        f"rule_masks={enable_rule_masks} all_samples=disable_exact_plan_filter "
        f"mrtf_trials={env.get('EPM_MR_TF_OPTUNA_TRIALS')} "
        f"lgbm_trials={env.get('EPM_LGBM_HPO_TRIALS')} "
        f"label_hpo={env.get('EPM_LGBM_BASE_LABEL_WEIGHT_HPO')}"
    )

    if pipeline._base_artifacts_ready(run_id):
        pipeline._append(f"{run_id}: base artifacts ready; skipping train_base.")
    else:
        pipeline._run_step(
            f"{suffix}_train_base_mrtf_all_samples_fs_hpo_labelhpo",
            pipeline._pipeline_cmd("train_base", run_id),
            env,
        )
    pipeline._require_file(
        pipeline.DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl",
        f"{run_id} base models",
    )

    meta_env = _enable_full_lgbm_search(pipeline._meta_fallback_env(env))
    meta_env.update(
        {
            "EPM_TRAIN_EXTEND_TO_LATEST": "1",
            "EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER": "1",
            "EPM_MR_TF_MASKS_ENABLED": "1",
            "EPM_MR_TF_OPTUNA_ENABLED": "1",
            "EPM_MR_TF_OPTUNA_TRIALS": "300",
            "EPM_MR_TF_OPTUNA_PATIENCE": "40",
            "EPM_MR_TF_OPTUNA_USE_NUMBA": "1",
            "EPM_LGBM_RULE_MASK_FEATURES_ENABLED": "1" if enable_rule_masks else "0",
            "EPM_LGBM_RULE_MASK_FEATURES_CSV": str(_rule_csv()),
            "EPM_LGBM_RULE_MASK_FEATURES_SIDE_FILTER": "0",
        }
    )
    meta_env.pop("EPM_TRAIN_RECENT_DAYS", None)
    meta_env["EPM_META_HPO_TRIALS"] = os.environ.get("EPM_TOP2_META_HPO_TRIALS", "150")
    if pipeline._meta_artifacts_ready(run_id):
        pipeline._append(f"{run_id}: meta artifacts ready; skipping train_meta.")
    else:
        pipeline._run_step(
            f"{suffix}_train_meta_mrtf_all_samples_fs_hpo_labelhpo",
            pipeline._pipeline_cmd("train_meta", run_id),
            meta_env,
        )
    pipeline._require_file(
        pipeline.DATA_ROOT / "artifacts" / run_id / "models" / "model_state_meta.pkl",
        f"{run_id} meta state",
    )

    pipeline._append(
        f"{run_id}: skipping strict policy-OOS handoff for all-samples matrix; "
        "simple_policy_optimiser will use explicit diagnostic final-fit "
        "policy-slice predictions."
    )

    with_env = _diagnostic_policy_env(env)
    with_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "1"
    pipeline._run_step(
        f"{suffix}_simple_policy_with_regime_adaptor",
        _policy_cmd(run_id),
        with_env,
    )
    pipeline._copy_policy_variant(run_id, "with_regime_adaptor")

    no_env = _diagnostic_policy_env(env)
    no_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "0"
    pipeline._run_step(
        f"{suffix}_simple_policy_without_regime_adaptor",
        _policy_cmd(run_id) + ["--no-regime-adaptor"],
        no_env,
    )
    pipeline._copy_policy_variant(run_id, "without_regime_adaptor")
    pipeline._restore_policy_variant(run_id, "with_regime_adaptor")
    pipeline._verify_stage_logs(run_id)
    marker.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suffix": suffix,
                "enable_rule_masks": bool(enable_rule_masks),
                "rule_csv": str(_rule_csv()),
                "complete": True,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_policy_metrics(run_id: str, variant: str) -> dict[str, Any]:
    root = (
        pipeline.DATA_ROOT
        / "artifacts"
        / run_id
        / f"simple_policy_optimiser_{variant}"
    )
    metrics_path = root / "policy_optimisation_oos_metrics_perps.json"
    if not metrics_path.exists():
        metrics_path = root / "policy_optimisation_oos_metrics.json"
    out: dict[str, Any] = {
        "policy_variant": variant,
        "metrics_path": str(metrics_path),
        "exists": metrics_path.exists(),
    }
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        out["strategies"] = {}
        for sid, payload in (data.get("strategies") or {}).items():
            val = ((payload.get("validation_metrics") or {}).get("top_30") or {})
            out["strategies"][sid] = {
                "n_trades": val.get("n_trades"),
                "avg_pnl_sized": val.get("avg_pnl_sized"),
                "avg_pnl_bankroll": val.get("avg_pnl_bankroll"),
                "pnl_positive_rate": val.get("pnl_positive_rate"),
                "sortino_proxy": val.get("sortino_proxy"),
            }
        out["prediction_source"] = (data.get("prediction_source") or {}).get("source")
    replay_path = root / "portfolio_policy_replay" / "portfolio_policy_replay_report.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        out["portfolio_replay"] = {
            key: replay.get(key)
            for key in (
                "objective",
                "accepted",
                "n_trades",
                "mean_net_pnl_per_trade",
                "pnl_positive_rate",
                "sortino",
                "max_drawdown",
            )
            if key in replay
        }
    return out


def _label_hpo_summary_from_model(model: Any) -> dict[str, Any]:
    report = dict(getattr(model, "label_weight_hpo_report_", {}) or {})
    if not report:
        metrics = getattr(model, "metrics", {}) or {}
        report = dict(metrics.get("label_weight_hpo_report") or {})
    return {
        "enabled": report.get("enabled"),
        "selected": report.get("selected"),
        "winner": report.get("winner"),
        "baseline_objective": (report.get("baseline") or {}).get("objective"),
        "best_objective": (report.get("best_optimized") or {}).get("objective"),
        "delta_vs_baseline": report.get("objective_delta_vs_baseline"),
    }


def _load_base_route_and_label_summary(run_id: str) -> dict[str, Any]:
    path = pipeline.DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with path.open("rb") as f:
        state = pickle.load(f)
    out: dict[str, Any] = {"exists": True, "path": str(path), "strategies": {}}
    alpha = (state or {}).get("alpha_models") or {}
    for side, by_strategy in alpha.items():
        if not isinstance(by_strategy, dict):
            continue
        for sid, info in by_strategy.items():
            h_payloads = (info or {}).get("models_by_h") or {}
            for h, payload in h_payloads.items():
                model = (payload or {}).get("model")
                mrtf = (payload or {}).get("mr_tf_specialists") or {}
                routes = {}
                for route, route_payload in (mrtf.get("routes") or {}).items():
                    cmp_payload = (route_payload or {}).get("baseline_comparison") or {}
                    support = (route_payload or {}).get("support") or {}
                    routes[route] = {
                        "enabled": (route_payload or {}).get("enabled"),
                        "promoted": (route_payload or {}).get("promoted"),
                        "support_n": support.get("n"),
                        "support_ok": support.get("ok"),
                        "uplift": cmp_payload.get("uplift"),
                        "promotion_metric": cmp_payload.get("promotion_metric"),
                        "prune_reason": (route_payload or {}).get("prune_reason"),
                    }
                out["strategies"][f"{side}_{sid}_H{h}"] = {
                    "selected_features": len((payload or {}).get("feat_cols") or []),
                    "rule_mask_features_selected": len(
                        [
                            c
                            for c in ((payload or {}).get("feat_cols") or [])
                            if str(c).startswith("lgbm_rule_mask_")
                        ]
                    ),
                    "mr_tf_routes": routes,
                    "mask_counts": ((mrtf.get("mask_diagnostics") or {}).get("counts")),
                    "label_hpo": _label_hpo_summary_from_model(model),
                }
    return out


def _write_comparison(run_ids: dict[str, str]) -> Path:
    rows = []
    summary: dict[str, Any] = {
        "baseline_run_id": BASELINE_RUN_ID,
        "run_ids": run_ids,
        "runs": {},
    }
    for label, run_id in run_ids.items():
        run_summary = {
            "base": _load_base_route_and_label_summary(run_id),
            "policy_with_regime_adaptor": _load_policy_metrics(
                run_id, "with_regime_adaptor"
            ),
            "policy_without_regime_adaptor": _load_policy_metrics(
                run_id, "without_regime_adaptor"
            ),
        }
        summary["runs"][label] = run_summary
        for variant_key in ("policy_with_regime_adaptor", "policy_without_regime_adaptor"):
            policy = run_summary[variant_key]
            for sid, metrics in (policy.get("strategies") or {}).items():
                rows.append(
                    {
                        "run_label": label,
                        "run_id": run_id,
                        "policy_variant": policy.get("policy_variant"),
                        "strategy_id": sid,
                        **metrics,
                    }
                )
    out_dir = pipeline.DATA_ROOT / "artifacts" / "mrtf_all_samples_rule_matrix_20260614"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_summary.json"
    csv_path = out_dir / "policy_strategy_comparison.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    pipeline._append(f"Wrote MR/TF matrix comparison: {json_path} and {csv_path}")
    return json_path


def main() -> int:
    prefix = _matrix_prefix()
    pipeline.LOG_PATH = pipeline.LOG_DIR / f"mrtf_all_samples_rule_matrix_{prefix}.log"
    pipeline._append("MR/TF all-samples rule-feature matrix starting")
    _assert_same_strategies_as_baseline()
    rule_csv = _rule_csv()
    if not rule_csv.exists():
        raise SystemExit(f"rule-mask CSV missing: {rule_csv}")
    with rule_csv.open("r", encoding="utf-8", newline="") as f:
        rule_count = max(0, sum(1 for _ in csv.DictReader(f)))
    pipeline._append(f"Rule-mask CSV: {rule_csv} rows={rule_count}")

    base_winner = pipeline.ensure_base_recency_winner()
    pipeline.ensure_loc_ema_meta()
    meta_winner = pipeline.ensure_meta_recency_winner()
    run_ids = {
        "plain": _variant_run_id(prefix, "plain_fullhpo_labelhpo"),
        "rules": _variant_run_id(prefix, "rules_fullhpo_labelhpo"),
    }
    manifest = {
        "prefix": prefix,
        "run_ids": run_ids,
        "rule_csv": str(rule_csv),
        "base_winner": str(base_winner),
        "meta_winner": str(meta_winner),
        "baseline_run_id": BASELINE_RUN_ID,
    }
    manifest_path = (
        pipeline.DATA_ROOT
        / "artifacts"
        / "mrtf_all_samples_rule_matrix_20260614"
        / "matrix_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pipeline._append(f"Matrix manifest: {manifest_path}")

    _run_variant(
        run_id=run_ids["plain"],
        suffix="plain",
        enable_rule_masks=False,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    _run_variant(
        run_id=run_ids["rules"],
        suffix="rules",
        enable_rule_masks=True,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    comparison = _write_comparison(run_ids)
    pipeline._append(f"MR/TF all-samples rule-feature matrix complete: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
