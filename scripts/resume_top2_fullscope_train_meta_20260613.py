#!/usr/bin/env python3
"""Resume the top-two full-scope run at train_meta.

The full-scope base stage is expected to be complete for both top-two heads.
Run only train_meta with fresh meta feature selection/HPO so the newly generated
base diagnostics participate in selection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "EPM_TOP2_RESELECT_RUN_ID",
    "20260612_183500_top2_reselect_labelhpo_drift_leaflite_native",
)
os.environ.setdefault(
    "EPM_TOP2_FULLSCOPE_RUN_ID",
    "20260612_203000_top2_fullscope_labelhpo_drift_leaflite_native",
)

from scripts.run_top2_recency_pipeline_20260611 import (  # noqa: E402
    DATA_ROOT,
    POLICY_SLICE_SOURCE_RUN_ID,
    SOURCE_RUN_ID,
    STAGE1_RUN_ID,
    STAGE3_RUN_ID,
    _append,
    _build_recent_tail_slice_plan,
    _meta_fallback_env,
    _pipeline_cmd,
    _require_file,
    _run_step,
    _train_env,
    _winner_paths,
)


def run_fullscope_train_meta_resume() -> None:
    base_winner, meta_winner = _winner_paths()
    _require_file(base_winner, "base recency winner")
    _require_file(meta_winner, "meta recency winner")
    run_root = DATA_ROOT / "artifacts" / STAGE3_RUN_ID
    _require_file(run_root / "base_models_intermediate.pkl", "full-scope base models")
    _require_file(run_root / "models" / "trained_state.pkl", "full-scope trained state")
    _require_file(run_root / "oof" / "base_oof_all.parquet", "full-scope consolidated base OOF")

    slice_plan_path = _build_recent_tail_slice_plan(POLICY_SLICE_SOURCE_RUN_ID, STAGE3_RUN_ID)
    env = _train_env(
        run_id=STAGE3_RUN_ID,
        label_source_run_id=SOURCE_RUN_ID,
        preset_source_run_id=STAGE1_RUN_ID,
        slice_plan_path=slice_plan_path,
        params_only=False,
        full_scope=True,
        base_winner=base_winner,
        meta_winner=meta_winner,
    )
    meta_env = _meta_fallback_env(env)
    meta_env.update(
        {
            "EPM_META_HPO_TRIALS": os.environ.get("EPM_TOP2_META_HPO_TRIALS", "150"),
            "EPM_META_PRESERVE_EXISTING_OOF": "0",
            "EPM_LGBM_USE_NATIVE_PRESET": "0",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "0",
            # The resumed loc_ema base bundle is valid, but its gate metrics were
            # seeded from native artifacts and can be incomplete.  Disable this
            # pre-meta gate so both top-two heads get fresh full-scope meta HPO.
            "EPM_META_BASE_QUALITY_GATE_ENABLE": "0",
            # Keep the full-scope meta resume OOM-conservative.
            "EPM_LGBM_N_JOBS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    _append(
        "Full-scope train_meta resume settings: fresh meta feature selection/HPO, "
        "native meta presets disabled, meta_hpo_trials="
        f"{meta_env.get('EPM_META_HPO_TRIALS')}, base_quality_gate=0, lgbm_n_jobs=1."
    )
    _run_step(
        "resume_stage3_train_meta_fresh_hpo",
        _pipeline_cmd("train_meta", STAGE3_RUN_ID),
        meta_env,
    )
    _require_file(run_root / "models" / "model_state_meta.pkl", "full-scope meta state")
    _append("Resume top2 full-scope train_meta completed.")


def main() -> int:
    run_fullscope_train_meta_resume()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
