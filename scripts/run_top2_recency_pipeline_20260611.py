#!/usr/bin/env python3
"""Continue loc_ema recency-HPO into top-two retrain/policy/full-scope rounds."""
from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / "logs"
DATA_ROOT = ROOT / "data_perp"
LOG_DIR.mkdir(exist_ok=True)

SOURCE_RUN_ID = "20260610_094500_mkt_eq_stripped_hpo_v7_sliced_full_labels_exact"
FEATURE_SOURCE_RUN_ID = "20260523_015947"
POLICY_SLICE_SOURCE_RUN_ID = os.environ.get(
    "EPM_TOP2_POLICY_SLICE_SOURCE_RUN_ID",
    SOURCE_RUN_ID,
).strip()
LOC_EMA = (
    "loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_"
    "up_down_semivol_ratio_tanh_-0_39156261"
)
DIST_ROLLING = "dist_rolling_7d_high_0_13977644_rolling_range_20_-0_40672407"
TOP2 = [
    {
        "side": "short",
        "strategy_id": LOC_EMA,
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
        "strategy_id": DIST_ROLLING,
        "old_strategy_id": (
            "dist_rolling_7d_high_0_13977644_mkt_ret_eq_24h_-0_56630391_"
            "rolling_range_20_-0_40672407"
        ),
        "canonical_key": (
            "(*)|(dist_rolling_7d_high>0.13977644)|(rolling_range_20>-0.40672407)"
        ),
        "ranking_score": 3.0,
    },
]

STAGE1_RUN_ID = os.environ.get(
    "EPM_TOP2_RESELECT_RUN_ID",
    "20260611_003000_top2_reselect_recency_params_policy",
).strip()
STAGE3_RUN_ID = os.environ.get(
    "EPM_TOP2_FULLSCOPE_RUN_ID",
    "20260611_013000_top2_fullscope_recency_final",
).strip()
LOG_PATH = LOG_DIR / f"top2_recency_pipeline_{STAGE1_RUN_ID}.log"


def _append(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    _append(f"START {name}: {' '.join(cmd)}")
    with LOG_PATH.open("ab", buffering=0) as log_fp:
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
    _append(f"END {name}: ret={ret}")
    if ret != 0:
        raise SystemExit(ret)


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"{label} missing or empty: {path}")


def _winner_paths() -> tuple[Path, Path]:
    root = DATA_ROOT / "artifacts" / SOURCE_RUN_ID / "recency_hpo" / LOC_EMA
    return (
        root / "top3_confirmed" / "base_winner.json",
        root / "meta_top3_confirmed" / "meta_winner.json",
    )


def _wait_for_file(path: Path, *, label: str, timeout_sec: int = 0) -> bool:
    started = time.monotonic()
    while True:
        if path.exists() and path.stat().st_size > 0:
            _append(f"{label} available: {path}")
            return True
        if timeout_sec and time.monotonic() - started > timeout_sec:
            return False
        time.sleep(60)


def _current_base_recency_alive() -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", "run_recency_hpo_loc_ema_stack_20260610.py"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def ensure_base_recency_winner() -> Path:
    base_winner, _ = _winner_paths()
    if base_winner.exists() and base_winner.stat().st_size > 0:
        return base_winner
    while _current_base_recency_alive():
        _append("Waiting for active base recency-HPO job to produce base_winner.json")
        if _wait_for_file(base_winner, label="base recency winner", timeout_sec=300):
            return base_winner
    _append("No active base recency-HPO job found; relaunching fixed-contract top3 run")
    _run_step(
        "base_recency_hpo_loc_ema_top3",
        [sys.executable, "-u", "scripts/run_recency_hpo_loc_ema_stack_20260610.py"],
        _common_env(SOURCE_RUN_ID),
    )
    _require_file(base_winner, "base recency winner")
    return base_winner


def _top2_registry(run_id: str) -> Path:
    out_dir = DATA_ROOT / "artifacts" / run_id / "strategy_registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "top2_mkt_eq_stripped_rule_registry.csv"
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
        for row in TOP2:
            writer.writerow(
                {
                    "market_mode": "perps",
                    "side": row["side"],
                    "trade_side": row["side"],
                    "source_horizon": 10,
                    "source_target": "top2_recency_reselect",
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


def _build_recent_tail_slice_plan(source_run_id: str, target_run_id: str) -> Path:
    from extreme_price_movements.slice_plan_store import (
        _load_label_events_for_slice_plan,
        load_or_build_slice_plan,
    )
    from extreme_price_movements.simple_policy_optimiser import (
        _load_policy_stage_view,
        _load_slice_plan_source_validation,
    )

    target = DATA_ROOT / "artifacts" / target_run_id / "slices" / "slice_plan.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    events = _load_label_events_for_slice_plan(str(DATA_ROOT), source_run_id)
    if events is None or events.empty:
        raise SystemExit(f"Could not load source label events for recent slice plan: {source_run_id}")
    old_env = os.environ.get("EPM_TRAIN_SLICE_PLAN_RUN_ID")
    os.environ["EPM_TRAIN_SLICE_PLAN_RUN_ID"] = target_run_id
    try:
        load_or_build_slice_plan(
            {
                "data_root": "data_perp",
                "output_run_id": target_run_id,
                "run_id": target_run_id,
                "market_mode": "perps",
                "exchange": "krakenfutures",
                "slice_planner_preset": "fast",
                "policy_optimiser_recent_weeks_enable": True,
                "policy_optimiser_optimise_start_weeks_ago": 8,
                "policy_optimiser_optimise_end_weeks_ago": 0,
                "policy_optimiser_validation_start_weeks_ago": 13,
                "policy_optimiser_validation_end_weeks_ago": 9,
                "policy_optimiser_max_sample_fraction": 0.30,
            },
            pd.to_datetime(events["t0"].max(), utc=True),
            events_df=events,
            force_refresh=True,
        )
    finally:
        if old_env is None:
            os.environ.pop("EPM_TRAIN_SLICE_PLAN_RUN_ID", None)
        else:
            os.environ["EPM_TRAIN_SLICE_PLAN_RUN_ID"] = old_env

    if not target.exists():
        raise SystemExit(f"recent slice plan was not written: {target}")
    validation = _load_slice_plan_source_validation(target)
    stage_view, stage_name = _load_policy_stage_view(target)
    if stage_name != "policy_optimiser" or len(stage_view.get("allowed_periods") or []) < 2:
        raise SystemExit(f"missing recent policy_optimiser stage view in {target}: {stage_name}")
    if not validation.get("oos_policy_slice_verified"):
        raise SystemExit(f"recent policy-OOS slice is not verified in {target}: {validation}")
    roles = set(validation.get("policy_holdout_predict_roles") or [])
    required_roles = {
        "policy_holdout_recent_validation",
        "policy_holdout_recent_optimise",
    }
    if not required_roles.issubset(roles) or "policy_holdout_middle" in roles:
        raise SystemExit(f"policy-OOS holdout is not recent-tail only in {target}: {validation}")
    if validation.get("policy_optimiser_optimise_start_weeks_ago") != [8]:
        raise SystemExit(f"unexpected optimise start weeks in {target}: {validation}")
    if validation.get("policy_optimiser_optimise_end_weeks_ago") != [0]:
        raise SystemExit(f"unexpected optimise end weeks in {target}: {validation}")
    if validation.get("policy_optimiser_validation_start_weeks_ago") != [13]:
        raise SystemExit(f"unexpected validation start weeks in {target}: {validation}")
    if validation.get("policy_optimiser_validation_end_weeks_ago") != [9]:
        raise SystemExit(f"unexpected validation end weeks in {target}: {validation}")
    max_event = pd.to_datetime(events["t0"].max(), utc=True)
    predict_end = pd.to_datetime(
        validation.get("policy_optimiser_predict_end"), utc=True, errors="coerce"
    )
    if pd.isna(predict_end) or (max_event - predict_end) > pd.Timedelta(days=2):
        raise SystemExit(
            f"recent policy-OOS holdout does not reach latest event tail in {target}: "
            f"predict_end={predict_end}, max_event={max_event}, validation={validation}"
        )
    _append(
        "Built verified recent-tail policy-OOS slice plan at "
        f"{target} roles={sorted(roles)} predict={validation.get('policy_optimiser_predict_start')}.."
        f"{validation.get('policy_optimiser_predict_end')}"
    )
    return target


def _common_env(run_id: str) -> dict[str, str]:
    ids = ",".join(row["strategy_id"] for row in TOP2)
    policy_ids = ",".join(f"{row['side']}_{row['strategy_id']}" for row in TOP2)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ".",
            "MPLCONFIGDIR": "/private/tmp/mplconfig",
            "EPM_OUTPUT_RUN_ID": run_id,
            "EPM_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_META_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_NO_PENALTY": "1",
            "EPM_BASE_STRATEGY_IDS": ids,
            "EPM_META_STRATEGY_IDS": ids,
            "EPM_LABEL_STRATEGY_IDS": ids,
            "EPM_POLICY_STRATEGY_IDS": policy_ids,
            "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
            "EPM_EXCHANGE": "kraken",
            "EPM_LGBM_ARCHETYPE_FEATURES": "1",
            "EPM_LGBM_RAW_CONTRIB_OOF_EXPORT": "0",
            "EPM_LGBM_META_LEAF_DIAGNOSTICS": "0",
            "EPM_LGBM_META_LEAF_LITE_DIAGNOSTICS": "1",
            "EPM_LGBM_META_LEAF_SUPPORT_DIAGNOSTICS": "1",
            "EPM_LGBM_META_LEAF_TARGET_DIAGNOSTICS": "1",
            "EPM_LGBM_META_LEAF_CENTROID_DIAGNOSTICS": "0",
            "EPM_LGBM_META_LEAF_MAX_TREES": "64",
            "EPM_LGBM_META_CONTRIB_DIAGNOSTICS": "1",
            "EPM_LGBM_META_CONTRIB_METHOD": "path",
            "EPM_LGBM_META_SCORE_PATH_DIAGNOSTICS": "1",
            "EPM_LGBM_META_SCORE_PATH_MAX_TREES": "64",
            "EPM_LGBM_META_DRIFT_FEATURES": "1",
            "EPM_LGBM_META_DRIFT_MAX_ROWS": "100000",
            "EPM_LGBM_META_DRIFT_MAX_FEATURES": "32",
            "EPM_LGBM_FINAL_OOF_CONTEXT_FEATURES": "0",
            "EPM_LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES": "1",
            "EPM_LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES": "0",
            "EPM_LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES": "0",
            "EPM_LGBM_LABEL_WEIGHT_HPO_NUMBA": "1",
            "EPM_LGBM_LABEL_WEIGHT_HPO_MAX_ROWS": "20000",
            "EPM_LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS": "50000",
            "EPM_LGBM_CV_SPLITS": "3",
            "EPM_LGBM_CV_MODE": "interleaved_spread",
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_BASE_HPO_TRIALS": "0",
            "EPM_META_HPO_TRIALS": "0",
            "EPM_LGBM_RECENCY_WEIGHTING": "1",
            "EPM_LGBM_TRUE_SOFT_LABELS": "1",
            "EPM_LGBM_REBALANCE_EFFECTIVE_CLASSES": "1",
            "EPM_LGBM_REBALANCE_POS_MASS_MIN": "0.25",
            "EPM_LGBM_REBALANCE_POS_MASS_MAX": "0.55",
            "EPM_LGBM_REBALANCE_MAX_MULTIPLIER": "2.0",
            "EPM_LGBM_UNIVARIATE_MAX_ROWS": "20000",
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _train_env(
    *,
    run_id: str,
    label_source_run_id: str,
    preset_source_run_id: str,
    slice_plan_path: Path,
    params_only: bool,
    full_scope: bool,
    base_winner: Path,
    meta_winner: Path,
) -> dict[str, str]:
    registry = _top2_registry(run_id)
    env = _common_env(run_id)
    env.update(
        {
            "EPM_MASK_STRATEGY_SOURCE_CSV": str(registry),
            "EPM_MASK_STRATEGY_TOP_N": "10",
            "EPM_MASK_STRATEGY_RANKING_METRIC": "score_for_best_params",
            "EPM_ARTIFACT_SOURCE_RUN_ID": label_source_run_id,
            "EPM_LABEL_SOURCE_RUN_ID": label_source_run_id,
            "EPM_LABEL_ARTIFACT_RUN_ID": label_source_run_id,
            "EPM_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
            "EPM_TRAIN_SLICE_PLAN_PATH": str(slice_plan_path),
            "EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID": label_source_run_id,
            "EPM_LGBM_USE_NATIVE_PRESET": "1",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "1",
            "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": preset_source_run_id,
            "EPM_LGBM_NATIVE_PRESET_PARAMS_ONLY": "1" if params_only else "0",
            "EPM_RECENCY_HPO_USE_WINNER": "1",
            "EPM_RECENCY_HPO_BASE_WINNER_PATH": str(base_winner),
            "EPM_RECENCY_HPO_META_WINNER_PATH": str(meta_winner),
            "EPM_TRAIN_EXTEND_TO_LATEST": "1",
        }
    )
    if full_scope:
        env.update(
            {
                "EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER": "1",
                "EPM_TRAIN_RECENT_DAYS": "1095",
            }
        )
    else:
        env.pop("EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER", None)
        env.pop("EPM_TRAIN_RECENT_DAYS", None)
    return env


def _meta_fallback_env(env: dict[str, str]) -> dict[str, str]:
    meta_env = dict(env)
    meta_env.update(
        {
            # Meta features now include newly generated base diagnostics, so stale
            # source-run meta selected-features/best-params must not short-circuit
            # feature selection or HPO.
            "EPM_LGBM_USE_NATIVE_PRESET": "0",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "0",
            "EPM_META_HPO_TRIALS": os.environ.get("EPM_TOP2_META_HPO_TRIALS", "150"),
            "EPM_META_PRESERVE_EXISTING_OOF": "0",
        }
    )
    return meta_env


def _pipeline_cmd(stage: str, run_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        stage,
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


def _meta_state_has_loc_ema(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        state = joblib.load(path)
    except Exception:
        try:
            with path.open("rb") as f:
                state = pickle.load(f)
        except Exception:
            return False
    keys: set[str] = set()

    def collect(obj: Any, *, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.add(str(key))
                collect(value, depth=depth + 1)

    if isinstance(state, dict):
        collect(state)
    return any(LOC_EMA in key for key in keys)


def _base_artifacts_ready(run_id: str) -> bool:
    root = DATA_ROOT / "artifacts" / run_id
    required = [
        root / "base_models_intermediate.pkl",
        root / "models" / "trained_state.pkl",
        root / "oof" / "base_oof_all.parquet",
    ]
    for row in TOP2:
        strategy_id = row["strategy_id"]
        required.append(root / "oof" / f"oof_{strategy_id}_H10.parquet")
        required.append(root / "lgbm_reference" / "base" / strategy_id)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        _append(f"Stage1 base resume check: missing {missing[:5]}")
        return False
    empty = [
        str(path)
        for path in required
        if path.is_file() and path.stat().st_size <= 0
    ]
    if empty:
        _append(f"Stage1 base resume check: empty {empty[:5]}")
        return False
    return True


def _meta_artifacts_ready(run_id: str) -> bool:
    root = DATA_ROOT / "artifacts" / run_id
    required = [
        root / "models" / "model_state_meta.pkl",
        root / "models" / "model_state_meta.manifest.json",
        root / "meta_oof" / "meta_feature_contract.json",
    ]
    for row in TOP2:
        strategy_id = row["strategy_id"]
        head = f"short_{strategy_id}_tbm_clf"
        required.append(root / "meta_oof" / f"meta_oof_{head}.parquet")
        required.append(root / "lgbm_reference" / "meta" / f"meta_{head}")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        _append(f"Stage1 meta resume check: missing {missing[:5]}")
        return False
    empty = [
        str(path)
        for path in required
        if path.is_file() and path.stat().st_size <= 0
    ]
    if empty:
        _append(f"Stage1 meta resume check: empty {empty[:5]}")
        return False
    return True


def ensure_loc_ema_meta() -> Path:
    meta_path = DATA_ROOT / "artifacts" / SOURCE_RUN_ID / "models" / "model_state_meta.pkl"
    if _meta_state_has_loc_ema(meta_path):
        _append(f"loc_ema meta model already present: {meta_path}")
        return meta_path
    _run_step(
        "train_meta_loc_ema",
        [sys.executable, "-u", "scripts/run_mkt_eq_stripped_loc_ema_meta_20260610.py"],
        _common_env(SOURCE_RUN_ID),
    )
    if not _meta_state_has_loc_ema(meta_path):
        raise SystemExit(f"loc_ema meta model missing after train_meta: {meta_path}")
    return meta_path


def ensure_meta_recency_winner() -> Path:
    _, meta_winner = _winner_paths()
    if meta_winner.exists() and meta_winner.stat().st_size > 0:
        return meta_winner
    _run_step(
        "meta_recency_hpo_loc_ema_top3",
        [sys.executable, "-u", "scripts/run_recency_hpo_loc_ema_stack_meta_20260610.py"],
        _common_env(SOURCE_RUN_ID),
    )
    _require_file(meta_winner, "meta recency winner")
    return meta_winner


def _copy_policy_variant(run_id: str, suffix: str) -> None:
    src = DATA_ROOT / "artifacts" / run_id / "simple_policy_optimiser"
    dst = DATA_ROOT / "artifacts" / run_id / f"simple_policy_optimiser_{suffix}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    _append(f"Copied simple_policy_optimiser variant: {dst}")


def _restore_policy_variant(run_id: str, suffix: str) -> None:
    src = DATA_ROOT / "artifacts" / run_id / f"simple_policy_optimiser_{suffix}"
    dst = DATA_ROOT / "artifacts" / run_id / "simple_policy_optimiser"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _verify_stage_logs(run_id: str) -> None:
    log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    required = [
        "native LGBM preset params-only mode",
        "LGBM candidate recency-HPO sample weighting enabled",
        "feature_drift_ks_core",
        "feature_drift_psi",
        "raw contribution",
        "Regime adaptor trained from simple_policy_optimiser",
    ]
    missing = [needle for needle in required if needle not in log_text]
    report = {
        "run_id": run_id,
        "checked_log": str(LOG_PATH),
        "required_markers": required,
        "missing_markers": missing,
    }
    out = DATA_ROOT / "artifacts" / run_id / "top2_stage_verification.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if missing:
        _append(f"Stage verification warning: missing markers {missing}; report={out}")
    else:
        _append(f"Stage verification passed: {out}")


def run_reselect_and_policy(base_winner: Path, meta_winner: Path) -> str:
    run_id = STAGE1_RUN_ID
    marker = DATA_ROOT / "artifacts" / run_id / "top2_reselect_policy_complete.json"
    if marker.exists():
        _append(f"Stage1 already complete: {marker}")
        return run_id
    slice_plan_path = _build_recent_tail_slice_plan(POLICY_SLICE_SOURCE_RUN_ID, run_id)
    env = _train_env(
        run_id=run_id,
        label_source_run_id=SOURCE_RUN_ID,
        preset_source_run_id=SOURCE_RUN_ID,
        slice_plan_path=slice_plan_path,
        params_only=False,
        full_scope=False,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    if _base_artifacts_ready(run_id):
        _append(
            "Stage1 base artifacts already present; skipping train_base and "
            "resuming at train_meta."
        )
    else:
        _run_step("stage1_train_base_native_preset_reuse", _pipeline_cmd("train_base", run_id), env)
    _require_file(DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl", "stage1 base models")
    meta_env = _meta_fallback_env(env)
    if _meta_artifacts_ready(run_id):
        _append(
            "Stage1 meta artifacts already present; skipping train_meta and "
            "resuming at policy-OOS."
        )
    else:
        _run_step("stage1_train_meta_native_or_fresh_hpo", _pipeline_cmd("train_meta", run_id), meta_env)
    _require_file(DATA_ROOT / "artifacts" / run_id / "models" / "model_state_meta.pkl", "stage1 meta state")
    for row in TOP2:
        _run_step(
            f"stage1_policy_oos_{row['strategy_id']}",
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
            env,
        )
    policy_cmd = [
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
        ",".join(f"{row['side']}_{row['strategy_id']}" for row in TOP2),
    ]
    with_env = dict(env)
    with_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "1"
    _run_step("stage1_simple_policy_with_regime_adaptor", policy_cmd, with_env)
    _copy_policy_variant(run_id, "with_regime_adaptor")
    for row in TOP2:
        adaptor = (
            DATA_ROOT
            / "artifacts"
            / run_id
            / "simple_policy_optimiser"
            / "regime_adaptors"
            / row["strategy_id"]
            / "regime_adaptor.json"
        )
        if adaptor.exists():
            _append(f"Regime adaptor artifact present for {row['strategy_id']}: {adaptor}")
    no_env = dict(env)
    no_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "0"
    _run_step("stage1_simple_policy_without_regime_adaptor", policy_cmd + ["--no-regime-adaptor"], no_env)
    _copy_policy_variant(run_id, "without_regime_adaptor")
    _restore_policy_variant(run_id, "with_regime_adaptor")
    _verify_stage_logs(run_id)
    marker.write_text(json.dumps({"run_id": run_id, "complete": True}, indent=2) + "\n")
    return run_id


def run_fullscope_final(base_winner: Path, meta_winner: Path, preset_run_id: str) -> str:
    run_id = STAGE3_RUN_ID
    marker = DATA_ROOT / "artifacts" / run_id / "top2_fullscope_complete.json"
    if marker.exists():
        _append(f"Stage3 already complete: {marker}")
        return run_id
    slice_plan_path = _build_recent_tail_slice_plan(POLICY_SLICE_SOURCE_RUN_ID, run_id)
    env = _train_env(
        run_id=run_id,
        label_source_run_id=SOURCE_RUN_ID,
        preset_source_run_id=preset_run_id,
        slice_plan_path=slice_plan_path,
        params_only=False,
        full_scope=True,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    _run_step("stage3_train_base_fullscope", _pipeline_cmd("train_base", run_id), env)
    _require_file(DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl", "stage3 base models")
    _run_step("stage3_train_meta_fullscope", _pipeline_cmd("train_meta", run_id), env)
    _require_file(DATA_ROOT / "artifacts" / run_id / "models" / "model_state_meta.pkl", "stage3 meta state")
    marker.write_text(json.dumps({"run_id": run_id, "complete": True}, indent=2) + "\n")
    return run_id


def main() -> int:
    _append("Top-two recency pipeline starting")
    base_winner = ensure_base_recency_winner()
    _build_recent_tail_slice_plan(POLICY_SLICE_SOURCE_RUN_ID, SOURCE_RUN_ID)
    ensure_loc_ema_meta()
    meta_winner = ensure_meta_recency_winner()
    _append(f"Base winner: {base_winner}")
    _append(f"Meta winner: {meta_winner}")
    stage1_run = run_reselect_and_policy(base_winner, meta_winner)
    stage3_run = run_fullscope_final(base_winner, meta_winner, stage1_run)
    _append(f"Pipeline complete: stage1={stage1_run} stage3={stage3_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
