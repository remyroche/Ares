#!/usr/bin/env python3
"""Rerun the 20260609_210500 four-head perps pipeline with fresh HPO/drift features.

The historical 20260609_210500 run reused native LGBM presets and disabled
label/weight HPO.  This launcher keeps the same four heads and the same
middle-holdout slice, but runs fresh feature selection, model HPO, and an
otherwise identical baseline-label variant for comparison.
"""
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
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_perp"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OLD_RUN_ID = "20260609_210500"
SOURCE_RUN_ID = os.environ.get("EPM_NO_MKT4_SOURCE_RUN_ID", "20260523_015947").strip()
SLICE_PLAN_SOURCE_RUN_ID = os.environ.get(
    "EPM_NO_MKT4_SLICE_PLAN_SOURCE_RUN_ID",
    "20260612_203000_top2_fullscope_labelhpo_drift_leaflite_native",
).strip()
FEATURE_SOURCE_RUN_ID = os.environ.get(
    "EPM_NO_MKT4_FEATURE_SOURCE_RUN_ID",
    "20260605_070000",
).strip()
MATRIX_ID = os.environ.get(
    "EPM_NO_MKT4_FULLHPO_MATRIX_ID",
    "20260614_210500_no_mkt4_fullhpo_drift",
).strip()
LOG_PATH = LOG_DIR / f"no_mkt4_fullhpo_labelhpo_drift_{MATRIX_ID}.log"

STRATEGIES: list[dict[str, str]] = [
    {
        "side": "long",
        "strategy_id": (
            "dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_"
            "leverage_build_score_0_45107844_return_autocorr_48_1_18643_"
            "rolling_range_20_-0_25967735"
        ),
    },
    {
        "side": "long",
        "strategy_id": (
            "bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115_"
            "loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_"
            "range_24h_pct_0_13988039_variance_ratio_10_48_0_92117828"
        ),
    },
    {
        "side": "short",
        "strategy_id": (
            "bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385_"
            "price_rv_15d_robust_z_0_060036644"
        ),
    },
    {
        "side": "short",
        "strategy_id": (
            "asset_minus_mkt_oi_1d_peer_resid_0_34164831_"
            "oi_expansion_compression_balance_24h_0_42287597"
        ),
    },
]


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
    _fail_if_log_failed(name)


def _fail_if_log_failed(step_name: str) -> None:
    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    marker = f"=== START {step_name} ==="
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    else:
        text = text[-250_000:]
    markers = (
        "PIPELINE FAILED",
        "ERROR: No alpha label datasets found",
        "ERROR: Base models intermediate not found",
        "Traceback (most recent call last):",
    )
    hit = [m for m in markers if m in text]
    if hit:
        raise SystemExit(f"{step_name} wrote failure marker(s) to {LOG_PATH}: {hit}")


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"{label} missing or empty: {path}")


def _copy_old_slice_plan(run_id: str) -> Path:
    candidates = [
        DATA_ROOT / "artifacts" / OLD_RUN_ID / "slices" / "slice_plan.json",
        DATA_ROOT / "artifacts" / SLICE_PLAN_SOURCE_RUN_ID / "slices" / "slice_plan.json",
    ]
    src = next((path for path in candidates if path.exists() and path.stat().st_size > 0), candidates[0])
    dst = DATA_ROOT / "artifacts" / run_id / "slices" / "slice_plan.json"
    _require_file(src, "source slice plan")
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["run_id"] = run_id
    payload["copied_from_run_id"] = src.parents[1].name if len(src.parents) > 1 else ""
    dst.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    try:
        from extreme_price_movements.simple_policy_optimiser import (
            _load_policy_stage_view,
            _load_slice_plan_source_validation,
        )

        validation = _load_slice_plan_source_validation(dst)
        stage_view, stage_name = _load_policy_stage_view(dst)
        if stage_name != "policy_optimiser" or not stage_view.get("allowed_periods"):
            raise SystemExit(f"missing policy_optimiser stage view in copied plan: {dst}")
        if not validation.get("oos_policy_slice_verified"):
            raise SystemExit(f"policy-OOS slice is not verified in copied plan: {validation}")
        roles = set(validation.get("policy_holdout_predict_roles") or [])
        has_middle = "policy_holdout_middle" in roles
        has_recent = {
            "policy_holdout_recent_optimise",
            "policy_holdout_recent_validation",
        }.issubset(roles)
        if not (has_middle or has_recent):
            raise SystemExit(
                f"copied slice is neither middle nor recent policy holdout: {validation}"
            )
    except ImportError:
        _append("WARNING: could not import slice-plan validators; copied plan without validator import")
    return dst


def _source_registry_rows() -> dict[str, dict[str, str]]:
    source = DATA_ROOT / "artifacts" / SOURCE_RUN_ID / "policy_oos_retrain_strategy_source_perps.csv"
    if not source.exists() or source.stat().st_size <= 0:
        _append(
            "Source strategy registry unavailable; using explicit STRATEGIES "
            f"fallback only: {source}"
        )
        return {}
    out: dict[str, dict[str, str]] = {}
    with source.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = str(row.get("strategy_id", "")).strip()
            side = str(row.get("side", "") or row.get("trade_side", "")).strip()
            core = raw
            for prefix in ("long_", "short_"):
                if core.startswith(prefix):
                    core = core[len(prefix) :]
                    break
            out[core] = row
            if side and raw:
                out[f"{side}_{core}"] = row
    return out


def _write_registry(run_id: str) -> Path:
    source_rows = _source_registry_rows()
    out_dir = DATA_ROOT / "artifacts" / run_id / "strategy_registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "no_mkt4_fullhpo_strategy_registry.csv"
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
        "stage_e_rank_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(STRATEGIES, start=1):
            core = item["strategy_id"]
            side = item["side"]
            sid = f"{side}_{core}"
            src = source_rows.get(f"{side}_{core}") or source_rows.get(core) or {}
            score = src.get("stage_e_rank_score") or src.get("ranking_score") or str(10 - rank)
            writer.writerow(
                {
                    "market_mode": "perps",
                    "side": side,
                    "trade_side": side,
                    "source_horizon": src.get("source_horizon", "5"),
                    "source_target": src.get("source_target", "returns_target"),
                    "strategy_id": sid,
                    "old_strategy_id": src.get("strategy_id", ""),
                    "canonical_key": src.get("canonical_key", core),
                    "base_event_trigger": src.get("base_event_trigger", src.get("canonical_key", core)),
                    "move_bucket": src.get("move_bucket", ""),
                    "candidate_bucket": "worst",
                    "ranking_score": score,
                    "score_for_best_params": score,
                    "stage_e_rank_score": score,
                }
            )
    return path


def _core_ids() -> str:
    return ",".join(f"{row['side']}_{row['strategy_id']}" for row in STRATEGIES)


def _policy_ids() -> str:
    return ",".join(f"{row['side']}_{row['strategy_id']}" for row in STRATEGIES)


def _base_env(run_id: str, *, label_hpo: bool) -> dict[str, str]:
    slice_plan = _copy_old_slice_plan(run_id)
    registry = _write_registry(run_id)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ".",
            "MPLCONFIGDIR": "/private/tmp/mplconfig",
            "MPLBACKEND": "Agg",
            "EPM_OUTPUT_RUN_ID": run_id,
            "EPM_ARTIFACT_SOURCE_RUN_ID": SOURCE_RUN_ID,
            "EPM_LABEL_SOURCE_RUN_ID": SOURCE_RUN_ID,
            "EPM_LABEL_ARTIFACT_RUN_ID": SOURCE_RUN_ID,
            "EPM_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
            "EPM_TRAIN_SLICE_PLAN_PATH": str(slice_plan),
            "EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID": SLICE_PLAN_SOURCE_RUN_ID,
            "EPM_POLICY_LABEL_SOURCE_RUN_ID": SOURCE_RUN_ID,
            "EPM_POLICY_FEATURE_SOURCE_RUN_ID": FEATURE_SOURCE_RUN_ID,
            "EPM_MASK_STRATEGY_SOURCE_CSV": str(registry),
            "EPM_MASK_STRATEGY_TOP_N": "10",
            "EPM_MASK_STRATEGY_RANKING_METRIC": "stage_e_rank_score",
            "EPM_BASE_STRATEGY_IDS": _core_ids(),
            "EPM_META_STRATEGY_IDS": _core_ids(),
            "EPM_LABEL_STRATEGY_IDS": _core_ids(),
            "EPM_POLICY_STRATEGY_IDS": _policy_ids(),
            "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
            "EPM_EXCHANGE": "kraken",
            "EPM_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_META_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_NO_PENALTY": "1",
            "EPM_META_TRAIN_TBM_CLF_HEAD": "1",
            "EPM_META_TRAIN_CORRECTNESS_CLF_HEAD": "0",
            "EPM_META_CLF_ENABLED": "1",
            "EPM_LGBM_USE_NATIVE_PRESET": "0",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "0",
            "EPM_LGBM_NATIVE_PRESET_PARAMS_ONLY": "0",
            "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": "",
            "EPM_LGBM_HPO_TRIALS": os.environ.get("EPM_NO_MKT4_LGBM_HPO_TRIALS", "200"),
            "EPM_LGBM_HPO_EARLY_STOP_PATIENCE": os.environ.get("EPM_NO_MKT4_LGBM_HPO_PATIENCE", "40"),
            "EPM_BASE_HPO_TRIALS": os.environ.get("EPM_NO_MKT4_BASE_HPO_TRIALS", "200"),
            "EPM_META_HPO_TRIALS": os.environ.get("EPM_NO_MKT4_META_HPO_TRIALS", "200"),
            "EPM_LGBM_HPO_MAX_ROWS": os.environ.get("EPM_NO_MKT4_LGBM_HPO_MAX_ROWS", "8000"),
            "EPM_LGBM_RACE_MAX_ROWS": os.environ.get("EPM_NO_MKT4_LGBM_RACE_MAX_ROWS", "80000"),
            "EPM_LGBM_UNIVARIATE_MAX_ROWS": os.environ.get("EPM_NO_MKT4_UNIVARIATE_MAX_ROWS", "12000"),
            "EPM_LGBM_RELIEF_ENABLED": os.environ.get("EPM_NO_MKT4_RELIEF_ENABLED", "0"),
            "EPM_LGBM_CV_SPLITS": "3",
            "EPM_LGBM_CV_MODE": "interleaved_spread",
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_LGBM_RECENCY_WEIGHTING": "1",
            "EPM_LGBM_TRUE_SOFT_LABELS": "1",
            "EPM_LGBM_REBALANCE_EFFECTIVE_CLASSES": "1",
            "EPM_LGBM_REBALANCE_POS_MASS_MIN": "0.25",
            "EPM_LGBM_REBALANCE_POS_MASS_MAX": "0.55",
            "EPM_LGBM_REBALANCE_MAX_MULTIPLIER": "2.0",
            "EPM_LGBM_BASE_LABEL_WEIGHT_HPO": "1" if label_hpo else "0",
            "EPM_LGBM_LABEL_WEIGHT_HPO_NUMBA": "1",
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS": os.environ.get("EPM_NO_MKT4_LABEL_HPO_LAYER1_TRIALS", "300"),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_PATIENCE": os.environ.get("EPM_NO_MKT4_LABEL_HPO_LAYER1_PATIENCE", "40"),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS": os.environ.get("EPM_NO_MKT4_LABEL_HPO_LAYER2_TRIALS", "150"),
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_PATIENCE": os.environ.get("EPM_NO_MKT4_LABEL_HPO_LAYER2_PATIENCE", "30"),
            "EPM_LGBM_LABEL_WEIGHT_HPO_MAX_ROWS": os.environ.get("EPM_NO_MKT4_LABEL_HPO_MAX_ROWS", "8000"),
            "EPM_LGBM_LABEL_WEIGHT_HPO_ELECTION_MAX_ROWS": os.environ.get("EPM_NO_MKT4_LABEL_HPO_ELECTION_MAX_ROWS", "25000"),
            "EPM_LABEL_WEIGHT_DISABLE": "1",
            "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT": "0",
            "EPM_TRAIN_EXTEND_TO_LATEST": "0",
            "EPM_LABEL_PERSIST_INCREMENTAL": "0",
            "EPM_LABEL_INCREMENTAL_ONLY_MISSING": "0",
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
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "3"),
            "OMP_THREAD_LIMIT": os.environ.get("OMP_THREAD_LIMIT", "3"),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _pipeline_cmd(stage: str, run_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        stage,
        "--perps",
        "--exchange",
        "krakenfutures",
        "--model-backend",
        "lgbm_pipeline",
        "--run-id",
        run_id,
    ]


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
        _policy_ids(),
    ]


def _base_ready(run_id: str) -> bool:
    root = DATA_ROOT / "artifacts" / run_id
    required = [
        root / "base_models_intermediate.pkl",
        root / "models" / "trained_state.pkl",
        root / "oof" / "base_oof_all.parquet",
    ]
    return all(path.exists() and (path.is_dir() or path.stat().st_size > 0) for path in required)


def _meta_ready(run_id: str) -> bool:
    root = DATA_ROOT / "artifacts" / run_id
    required = [
        root / "models" / "model_state_meta.pkl",
        root / "models" / "model_state_meta.manifest.json",
        root / "meta_oof" / "meta_feature_contract.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _base_error_backfill_ready(run_id: str) -> bool:
    root = DATA_ROOT / "artifacts" / run_id
    manifest = root / "oof" / "base_error_archetypes" / "manifest.json"
    states = root / "oof" / "base_error_archetypes" / "states.pkl"
    if not (manifest.exists() and states.exists()):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("state_count", 0) or 0) >= len(STRATEGIES)


def _policy_oos_ready(run_id: str) -> bool:
    manifest = DATA_ROOT / "artifacts" / run_id / "policy_oos_predictions" / "manifest.json"
    return manifest.exists() and manifest.stat().st_size > 0


def _policy_variant_ready(run_id: str, suffix: str) -> bool:
    root = DATA_ROOT / "artifacts" / run_id
    manifest = root / f"simple_policy_optimiser_{suffix}" / "manifest.json"
    params = root / f"simple_policy_optimiser_{suffix}" / "deployment" / "best_policy_params.json"
    return manifest.exists() and params.exists() and manifest.stat().st_size > 0 and params.stat().st_size > 0


def _copy_policy_variant(run_id: str, suffix: str) -> None:
    root = DATA_ROOT / "artifacts" / run_id
    copies = [
        (root / "simple_policy_optimiser", root / f"simple_policy_optimiser_{suffix}"),
        (root / "portfolio_policy_replay", root / f"portfolio_policy_replay_{suffix}"),
        (root / "policy_params", root / f"policy_params_{suffix}"),
    ]
    for src, dst in copies:
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for name in (
        "policy_optimisation_oos_metrics_perps.json",
        "policy_optimisation_oos_metrics.json",
        "policy_optimisation_perps.json",
        "policy_optimisation.json",
        "best_policy_params_perps.json",
        "best_policy_params.json",
    ):
        src = root / name
        if src.exists():
            shutil.copy2(src, root / f"{Path(name).stem}_{suffix}{Path(name).suffix}")
    _append(f"Copied policy variant {suffix} for {run_id}")


def _restore_policy_variant(run_id: str, suffix: str) -> None:
    root = DATA_ROOT / "artifacts" / run_id
    src = root / f"simple_policy_optimiser_{suffix}"
    dst = root / "simple_policy_optimiser"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _check_required_labels() -> None:
    root = DATA_ROOT / "artifacts" / SOURCE_RUN_ID / "labels"
    missing: list[str] = []
    for row in STRATEGIES:
        path = root / f"train_{row['side']}_{row['strategy_id']}_5.parquet"
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(str(path))
    if missing:
        raise SystemExit("missing required H5 label parquet(s): " + ", ".join(missing))
    _append("Required H5 label parquets present for all four strategies")


def _destination_label_manifest_ready(run_id: str) -> bool:
    manifest = DATA_ROOT / "artifacts" / run_id / "labels" / "labels_manifest.json"
    if not manifest.exists() or manifest.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    datasets = payload.get("datasets") or {}
    required = {
        f"train_{row['side']}_{row['strategy_id']}_5"
        for row in STRATEGIES
    }
    return required.issubset(set(map(str, datasets.keys())))


def _source_labels_available() -> bool:
    root = DATA_ROOT / "artifacts" / SOURCE_RUN_ID / "labels"
    return all(
        (root / f"train_{row['side']}_{row['strategy_id']}_5.parquet").exists()
        and (root / f"train_{row['side']}_{row['strategy_id']}_5.parquet").stat().st_size > 0
        for row in STRATEGIES
    )


def _materialize_label_manifest(run_id: str, env: dict[str, str]) -> Path:
    """Copy the four historical label parquets into a manifest-backed run root."""
    if _destination_label_manifest_ready(run_id):
        out = DATA_ROOT / "artifacts" / run_id / "labels" / "labels_manifest.json"
        _append(f"Label manifest already present for {run_id}: {out}")
        return out
    if not _source_labels_available():
        _append(
            "Historical four-strategy label parquets unavailable; generating fresh "
            f"labels for {run_id} from feature_source_run_id={FEATURE_SOURCE_RUN_ID}."
        )
        _run_step(f"{run_id}_labels", _pipeline_cmd("labels", run_id), env)
        out = DATA_ROOT / "artifacts" / run_id / "labels" / "labels_manifest.json"
        _require_file(out, "generated labels manifest")
        if not _destination_label_manifest_ready(run_id):
            raise SystemExit(
                f"generated labels manifest does not contain all four required datasets: {out}"
            )
        return out
    _check_required_labels()
    src_root = DATA_ROOT / "artifacts" / SOURCE_RUN_ID / "labels"
    dst_root = DATA_ROOT / "artifacts" / run_id / "labels"
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "source_run_id": SOURCE_RUN_ID,
        "generated_by": Path(__file__).name,
        "datasets": {},
    }
    for row in STRATEGIES:
        name = f"train_{row['side']}_{row['strategy_id']}_5"
        src = src_root / f"{name}.parquet"
        dst = dst_root / f"{name}.parquet"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        pf = pq.ParquetFile(dst)
        manifest["datasets"][name] = {
            "file": f"{name}.parquet",
            "rows": int(pf.metadata.num_rows),
            "columns": list(pf.schema.names),
            "source_file": str(src),
        }
    out = dst_root / "labels_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append(f"Materialized label manifest for {run_id}: {out}")
    return out


def _run_policy_variants(run_id: str, env: dict[str, str]) -> None:
    policy_cmd = _policy_cmd(run_id)
    if _policy_variant_ready(run_id, "with_regime_adaptor"):
        _append(f"{run_id}: with-regime policy variant already present; skipping")
    else:
        with_env = dict(env)
        with_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "1"
        _run_step(f"{run_id}_simple_policy_with_regime_adaptor", policy_cmd, with_env)
        _copy_policy_variant(run_id, "with_regime_adaptor")

    if _policy_variant_ready(run_id, "without_regime_adaptor"):
        _append(f"{run_id}: no-regime policy variant already present; skipping")
    else:
        no_env = dict(env)
        no_env["EPM_SIMPLE_POLICY_REGIME_ADAPTOR"] = "0"
        _run_step(
            f"{run_id}_simple_policy_without_regime_adaptor",
            policy_cmd + ["--no-regime-adaptor"],
            no_env,
        )
        _copy_policy_variant(run_id, "without_regime_adaptor")
    _restore_policy_variant(run_id, "with_regime_adaptor")


def _run_variant(run_id: str, *, label_hpo: bool) -> None:
    env = _base_env(run_id, label_hpo=label_hpo)
    marker = DATA_ROOT / "artifacts" / run_id / "no_mkt4_fullhpo_variant_complete.json"
    if marker.exists() and marker.stat().st_size > 0:
        _append(f"Variant already complete: {run_id}")
        return
    _append(
        f"Variant start run_id={run_id} label_hpo={label_hpo} "
        f"model_hpo={env.get('EPM_LGBM_HPO_TRIALS')} "
        f"base_hpo={env.get('EPM_BASE_HPO_TRIALS')} meta_hpo={env.get('EPM_META_HPO_TRIALS')}"
    )
    _materialize_label_manifest(run_id, env)
    env = dict(env)
    env["EPM_LABEL_SOURCE_RUN_ID"] = run_id
    env["EPM_LABEL_ARTIFACT_RUN_ID"] = run_id
    env["EPM_ARTIFACT_SOURCE_RUN_ID"] = run_id
    env["EPM_FEATURE_SOURCE_RUN_ID"] = FEATURE_SOURCE_RUN_ID
    env["EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID"] = SLICE_PLAN_SOURCE_RUN_ID
    if _base_ready(run_id):
        _append(f"{run_id}: base artifacts ready; skipping train_base")
    else:
        _run_step(f"{run_id}_train_base", _pipeline_cmd("train_base", run_id), env)
    _require_file(DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl", "base models")
    if _base_error_backfill_ready(run_id):
        _append(f"{run_id}: base-error archetype backfill ready; skipping")
    else:
        _run_step(
            f"{run_id}_base_error_archetype_backfill",
            [
                sys.executable,
                "-m",
                "extreme_price_movements.base_error_archetype_backfill",
                "--artifact-dir",
                str(DATA_ROOT / "artifacts" / run_id),
                "--force",
            ],
            env,
        )
    if _meta_ready(run_id):
        _append(f"{run_id}: meta artifacts ready; skipping train_meta")
    else:
        meta_env = dict(env)
        meta_env["EPM_META_PRESERVE_EXISTING_OOF"] = "0"
        _run_step(f"{run_id}_train_meta", _pipeline_cmd("train_meta", run_id), meta_env)
    _require_file(DATA_ROOT / "artifacts" / run_id / "models" / "model_state_meta.pkl", "meta state")
    if _policy_oos_ready(run_id):
        _append(f"{run_id}: policy-OOS predictions ready; skipping generation")
    else:
        _run_step(
            f"{run_id}_policy_oos_predictions",
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
            ],
            env,
        )
    _require_file(
        DATA_ROOT / "artifacts" / run_id / "policy_oos_predictions" / "manifest.json",
        "policy-OOS manifest",
    )
    _run_policy_variants(run_id, env)
    marker.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "label_hpo": bool(label_hpo),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "old_run_id": OLD_RUN_ID,
                "strategy_ids": _policy_ids().split(","),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _policy_metrics(run_id: str, suffix: str | None) -> dict[str, Any]:
    root = DATA_ROOT / "artifacts" / run_id
    if suffix:
        simple_root = root / f"simple_policy_optimiser_{suffix}"
        replay = root / f"portfolio_policy_replay_{suffix}" / "portfolio_policy_replay_report.json"
        metrics = root / f"policy_optimisation_oos_metrics_perps_{suffix}.json"
        if not metrics.exists():
            metrics = simple_root / "policy_optimisation_oos_metrics_perps.json"
    else:
        simple_root = root / "simple_policy_optimiser"
        replay = root / "portfolio_policy_replay" / "portfolio_policy_replay_report.json"
        metrics = root / "policy_optimisation_oos_metrics_perps.json"
    out: dict[str, Any] = {
        "run_id": run_id,
        "variant": suffix or "default",
        "simple_policy_root": str(simple_root),
        "metrics_path": str(metrics),
        "replay_path": str(replay),
    }
    m = _read_json(metrics)
    out["prediction_source"] = (m.get("prediction_source") or {}).get("source")
    out["reporting_rank_threshold"] = m.get("reporting_rank_threshold")
    out["strategies"] = {}
    for sid, payload in (m.get("strategies") or {}).items():
        validation = (payload.get("validation_metrics") or {})
        top = validation.get("top_30") or validation.get("top_20") or validation.get("all") or {}
        out["strategies"][sid] = {
            "n_trades": top.get("n_trades"),
            "avg_pnl_sized": top.get("avg_pnl_sized"),
            "avg_pnl_bankroll": top.get("avg_pnl_bankroll"),
            "pnl_positive_rate": top.get("pnl_positive_rate"),
            "sortino_proxy": top.get("sortino_proxy"),
            "mean_net_bps": top.get("mean_net_bps"),
        }
    r = _read_json(replay)
    gm = r.get("global_auction_metrics") or r
    out["portfolio_replay"] = {
        key: gm.get(key)
        for key in (
            "objective",
            "accepted",
            "n_trades",
            "trades_per_day",
            "mean_net_pnl_per_trade",
            "mean_net_return_per_trade",
            "pnl_positive_rate",
            "sortino",
            "max_drawdown",
            "net_pnl",
        )
        if key in gm
    }
    return out


def _label_hpo_summary(run_id: str) -> dict[str, Any]:
    path = DATA_ROOT / "artifacts" / run_id / "base_models_intermediate.pkl"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with path.open("rb") as f:
        state = pickle.load(f)
    out: dict[str, Any] = {"exists": True, "path": str(path), "heads": {}}
    for side, by_strategy in ((state or {}).get("alpha_models") or {}).items():
        if not isinstance(by_strategy, dict):
            continue
        for sid, info in by_strategy.items():
            h_payloads = (info or {}).get("models_by_h") or {}
            for h, payload in h_payloads.items():
                model = (payload or {}).get("model")
                report = dict(getattr(model, "label_weight_hpo_report_", {}) or {})
                if not report:
                    report = dict((getattr(model, "metrics", {}) or {}).get("label_weight_hpo_report") or {})
                out["heads"][f"{side}_{sid}_H{h}"] = {
                    "enabled": report.get("enabled"),
                    "selected": report.get("selected"),
                    "winner": report.get("winner"),
                    "baseline_objective": (report.get("baseline") or {}).get("objective"),
                    "election_baseline_objective": (report.get("election_baseline") or {}).get("objective"),
                    "best_objective": (report.get("best_optimized") or {}).get("objective"),
                    "delta_vs_baseline": report.get("objective_delta_vs_baseline"),
                    "selected_features": len((payload or {}).get("feat_cols") or []),
                }
    return out


def _meta_head_summary(run_id: str) -> dict[str, Any]:
    path = DATA_ROOT / "artifacts" / run_id / "meta_oof" / "meta_head_metrics.json"
    metrics = _read_json(path)
    out: dict[str, Any] = {"path": str(path), "heads": {}}
    for key, payload in metrics.items():
        if not str(key).endswith("_tbm_clf"):
            continue
        out["heads"][key] = {
            "n_samples": payload.get("n_samples"),
            "oof_mean": payload.get("oof_mean"),
            "oof_std": payload.get("oof_std"),
            "ic_target": payload.get("ic_target"),
            "ic_y_bin": payload.get("ic_y_bin"),
            "ic_y_ret": payload.get("ic_y_ret"),
        }
    return out


def _write_comparison(run_ids: dict[str, str]) -> Path:
    out_dir = DATA_ROOT / "artifacts" / MATRIX_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "matrix_id": MATRIX_ID,
        "old_run_id": OLD_RUN_ID,
        "new_run_ids": run_ids,
        "runs": {
            "old_default": {
                "policy": _policy_metrics(OLD_RUN_ID, None),
                "label_hpo": _label_hpo_summary(OLD_RUN_ID),
                "meta": _meta_head_summary(OLD_RUN_ID),
            }
        },
    }
    rows: list[dict[str, Any]] = []
    for label, run_id in run_ids.items():
        summary["runs"][label] = {
            "label_hpo": _label_hpo_summary(run_id),
            "meta": _meta_head_summary(run_id),
            "policy_with_regime_adaptor": _policy_metrics(run_id, "with_regime_adaptor"),
            "policy_without_regime_adaptor": _policy_metrics(run_id, "without_regime_adaptor"),
        }
    for run_label, payload in summary["runs"].items():
        for policy_key, policy in payload.items():
            if not str(policy_key).startswith("policy"):
                continue
            replay = policy.get("portfolio_replay") or {}
            rows.append(
                {
                    "run_label": run_label,
                    "policy_variant": policy.get("variant"),
                    "run_id": policy.get("run_id"),
                    "objective": replay.get("objective"),
                    "accepted": replay.get("accepted"),
                    "n_trades": replay.get("n_trades"),
                    "trades_per_day": replay.get("trades_per_day"),
                    "mean_net_pnl_per_trade": replay.get("mean_net_pnl_per_trade"),
                    "pnl_positive_rate": replay.get("pnl_positive_rate"),
                    "sortino": replay.get("sortino"),
                    "max_drawdown": replay.get("max_drawdown"),
                    "net_pnl": replay.get("net_pnl"),
                    "metrics_path": policy.get("metrics_path"),
                    "replay_path": policy.get("replay_path"),
                }
            )
    json_path = out_dir / "comparison_summary.json"
    csv_path = out_dir / "policy_replay_comparison.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    _append(f"Wrote comparison summary: {json_path}")
    return json_path


def main() -> int:
    _append("No-mkt4 full-HPO label/drift matrix starting")
    run_baseline_variant = os.environ.get(
        "EPM_NO_MKT4_RUN_BASELINE_LABEL_VARIANT",
        "1",
    ).strip().lower() not in {"0", "false", "no", "n", "off"}
    run_ids = {
        "new_labelhpo": f"{MATRIX_ID}_labelhpo",
    }
    if run_baseline_variant:
        run_ids["new_baseline_labels"] = f"{MATRIX_ID}_baseline_labels"
    manifest = {
        "matrix_id": MATRIX_ID,
        "old_run_id": OLD_RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "feature_source_run_id": FEATURE_SOURCE_RUN_ID,
        "run_ids": run_ids,
        "strategy_ids": _policy_ids().split(","),
    }
    manifest_dir = DATA_ROOT / "artifacts" / MATRIX_ID
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append(f"Matrix manifest: {manifest_dir / 'matrix_manifest.json'}")
    _run_variant(run_ids["new_labelhpo"], label_hpo=True)
    if run_baseline_variant:
        _run_variant(run_ids["new_baseline_labels"], label_hpo=False)
    else:
        _append(
            "Baseline-label variant skipped by "
            "EPM_NO_MKT4_RUN_BASELINE_LABEL_VARIANT=0; the label-HPO variant "
            "still elects baseline labels/weights per head when baseline wins."
        )
    comparison = _write_comparison(run_ids)
    _append(f"No-mkt4 full-HPO label/drift matrix complete: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
