#!/usr/bin/env python3
"""Train four historical market-regime heads with mkt_ret_eq_24h removed from masks."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

FEATURE_SOURCE_RUN_ID = "20260523_015947"
DEFAULT_RUN_ID = "20260610_094500_mkt_eq_stripped_hpo_v7_sliced_full_labels_exact"

STRATEGIES = [
    {
        "side": "short",
        "strategy_id": (
            "loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_"
            "up_down_semivol_ratio_tanh_-0_39156261"
        ),
        "old_strategy_id": (
            "loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_"
            "mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_"
            "up_down_semivol_ratio_tanh_-0_39156261"
        ),
        "canonical_key": (
            "(*)|(loc_ema_stack_pos_24<=0.43357179)|"
            "(compression_ratio>-0.33411601&up_down_semivol_ratio_tanh>-0.39156261)"
        ),
        "ranking_score": 4.0,
    },
    {
        "side": "short",
        "strategy_id": "dist_rolling_7d_high_0_13977644_rolling_range_20_-0_40672407",
        "old_strategy_id": (
            "dist_rolling_7d_high_0_13977644_mkt_ret_eq_24h_-0_56630391_"
            "rolling_range_20_-0_40672407"
        ),
        "canonical_key": (
            "(*)|(dist_rolling_7d_high>0.13977644)|(rolling_range_20>-0.40672407)"
        ),
        "ranking_score": 3.0,
    },
    {
        "side": "long",
        "strategy_id": (
            "loc_prev_week_range_pos_48_0_42586401_loc_vwap_dev_z_24_0_10701825_"
            "zscore_price_50_1_0128103_up_down_return_mass_ratio_tanh_1_1231147"
        ),
        "old_strategy_id": (
            "loc_prev_week_range_pos_48_0_42586401_loc_vwap_dev_z_24_0_10701825_"
            "zscore_price_50_1_0128103_mkt_ret_eq_24h_-0_78752208_"
            "up_down_return_mass_ratio_tanh_1_1231147"
        ),
        "canonical_key": (
            "(*)|(loc_prev_week_range_pos_48>0.42586401&loc_vwap_dev_z_24>0.10701825"
            "&zscore_price_50>1.0128103)|(up_down_return_mass_ratio_tanh<=1.1231147)"
        ),
        "ranking_score": 4.0,
    },
    {
        "side": "long",
        "strategy_id": (
            "dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_"
            "volume_autocorr_48_-0_38378653"
        ),
        "old_strategy_id": (
            "dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_"
            "mkt_ret_eq_24h_-0_43956268_volume_autocorr_48_-0_38378653"
        ),
        "canonical_key": (
            "(*)|(dist_weekly_vwap<=0.074823022&loc_prev_week_range_pos_48>0.48354843)"
            "|(volume_autocorr_48<=-0.38378653)"
        ),
        "ranking_score": 3.0,
    },
]


def _write_registry(run_id: str) -> Path:
    out_dir = ROOT / "data_perp" / "artifacts" / run_id / "strategy_registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mkt_eq_stripped_final_rule_registry.csv"
    fieldnames = [
        "market_mode",
        "side",
        "trade_side",
        "source_horizon",
        "source_target",
        "strategy_id",
        "old_strategy_id",
        "canonical_key",
        "base_event_trigger",
        "move_bucket",
        "candidate_bucket",
        "ranking_score",
        "score_for_best_params",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in STRATEGIES:
            writer.writerow(
                {
                    "market_mode": "perps",
                    "side": row["side"],
                    "trade_side": row["side"],
                    "source_horizon": 10,
                    "source_target": "manual_new_score_meta_metrics",
                    "strategy_id": row["strategy_id"],
                    "old_strategy_id": row["old_strategy_id"],
                    "canonical_key": row["canonical_key"],
                    "base_event_trigger": row["canonical_key"],
                    "move_bucket": "",
                    "candidate_bucket": "worst",
                    "ranking_score": row["ranking_score"],
                    "score_for_best_params": row["ranking_score"],
                }
            )
    return path


def _base_env(run_id: str, registry_path: Path) -> dict[str, str]:
    ids_csv = ",".join(row["strategy_id"] for row in STRATEGIES)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ".",
            "EPM_MASK_STRATEGY_SOURCE_CSV": str(registry_path),
            "EPM_MASK_STRATEGY_TOP_N": "2",
            "EPM_MASK_STRATEGY_RANKING_METRIC": "score_for_best_params",
            "EPM_LABEL_STRATEGY_IDS": ids_csv,
            "EPM_BASE_STRATEGY_IDS": ids_csv,
            "EPM_META_STRATEGY_IDS": ids_csv,
            "EPM_POLICY_STRATEGY_IDS": ids_csv,
            "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
            "EPM_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_NO_PENALTY": "1",
            "EPM_LGBM_USE_NATIVE_PRESET": "0",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "0",
            "EPM_LGBM_CV_SPLITS": "3",
            "EPM_LGBM_CV_MODE": "interleaved_spread",
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_LGBM_RECENCY_WEIGHTING": "1",
            "EPM_LGBM_BASE_RECENCY_HALF_LIFE_DAYS": "365",
            "EPM_LGBM_META_RECENCY_HALF_LIFE_DAYS": "182.5",
            "EPM_LGBM_TRUE_SOFT_LABELS": "1",
            "EPM_LGBM_REBALANCE_EFFECTIVE_CLASSES": "1",
            "EPM_LGBM_REBALANCE_POS_MASS_MIN": "0.25",
            "EPM_LGBM_REBALANCE_POS_MASS_MAX": "0.55",
            "EPM_LGBM_REBALANCE_MAX_MULTIPLIER": "2.0",
            "EPM_LGBM_UNIVARIATE_MAX_ROWS": "20000",
            "EPM_LGBM_HPO_TRIALS": "200",
            "EPM_LGBM_HPO_EARLY_STOP_PATIENCE": "40",
            "EPM_BASE_HPO_TRIALS": "200",
            "EPM_BASE_HPO_EARLY_STOP_PATIENCE": "40",
            "EPM_META_HPO_TRIALS": "200",
            "EPM_META_HPO_EARLY_STOP_PATIENCE": "40",
            "EPM_TRAIN_EXTEND_TO_LATEST": "1",
            "EPM_OUTPUT_RUN_ID": run_id,
            "EPM_LABEL_ARTIFACT_RUN_ID": run_id,
            "EPM_LABEL_PERSIST_INCREMENTAL": "0",
            "EPM_LABEL_INCREMENTAL_ONLY_MISSING": "0",
            "EPM_TRAIN_SLICE_PLAN_RUN_ID": run_id,
            "EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID": run_id,
            "EPM_POLICY_LABEL_SOURCE_RUN_ID": run_id,
            "EPM_POLICY_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
        }
    )
    return env


def _run_step(name: str, cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(f"\n=== START {name} ===\n".encode())
        log_fp.write(("CMD " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = proc.wait()
        log_fp.write(f"\n=== END {name} ret={ret} ===\n".encode())
    if ret != 0:
        raise SystemExit(ret)


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"{label} missing or empty: {path}")


def _require_labels(run_id: str) -> None:
    manifest = ROOT / "data_perp" / "artifacts" / run_id / "labels" / "labels_manifest.json"
    _require_file(manifest, "labels manifest")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") or {}
    expected = {f"train_{row['strategy_id']}_10" for row in STRATEGIES}
    missing = sorted(expected - set(datasets))
    if missing:
        raise SystemExit(f"labels manifest missing {len(missing)} expected datasets: {missing}")
    too_small = {
        key: int((datasets.get(key) or {}).get("rows", 0))
        for key in sorted(expected)
        if int((datasets.get(key) or {}).get("rows", 0)) < 5000
    }
    if too_small:
        raise SystemExit(f"label datasets too small for HPO: {too_small}")


def main() -> int:
    run_id = os.environ.get("EPM_STRIPPED_MKT_EQ_RUN_ID", DEFAULT_RUN_ID).strip()
    registry_path = _write_registry(run_id)
    env = _base_env(run_id, registry_path)
    log_path = LOG_DIR / f"train_{run_id}.log"

    labels_cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        "labels",
        "--market-mode",
        "perps",
        "--exchange",
        "kraken",
        "--ts",
        FEATURE_SOURCE_RUN_ID,
        "--run-id",
        run_id,
    ]
    _run_step("labels", labels_cmd, env, log_path)
    _require_labels(run_id)

    train_env = dict(env)
    train_env.update(
        {
            "EPM_ARTIFACT_SOURCE_RUN_ID": run_id,
            "EPM_LABEL_SOURCE_RUN_ID": run_id,
            "EPM_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
        }
    )
    common_train_args = [
        "--market-mode",
        "perps",
        "--exchange",
        "kraken",
        "--model-backend",
        "lgbm_pipeline",
        "--ts",
        FEATURE_SOURCE_RUN_ID,
        "--run-id",
        run_id,
    ]
    _run_step(
        "train_base",
        [sys.executable, "-u", "extreme_price_movements/run_pipeline.py", "train_base", *common_train_args],
        train_env,
        log_path,
    )
    _require_file(
        ROOT / "data_perp" / "artifacts" / run_id / "base_models_intermediate.pkl",
        "base models intermediate",
    )
    _run_step(
        "train_meta",
        [sys.executable, "-u", "extreme_price_movements/run_pipeline.py", "train_meta", *common_train_args],
        train_env,
        log_path,
    )
    _require_file(
        ROOT / "data_perp" / "artifacts" / run_id / "models" / "model_state_meta.pkl",
        "meta model state",
    )

    for row in STRATEGIES:
        _run_step(
            f"policy_oos:{row['strategy_id']}",
            [
                sys.executable,
                "-u",
                "scripts/generate_policy_oos_predictions.py",
                "--data-root",
                "data_perp",
                "--run-id",
                run_id,
                "--market-mode",
                "perps",
                "--strategy-id",
                row["strategy_id"],
            ],
            train_env,
            log_path,
        )

    _run_step(
        "simple_policy_optimiser",
        [
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
            ",".join(row["strategy_id"] for row in STRATEGIES),
        ],
        train_env,
        log_path,
    )
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
