#!/usr/bin/env python3
"""Resume the top-two full-scope base run after the dist_rolling OOM.

The full-scope run already saved the loc_ema native/reference/OOF artifacts but
died before dist_rolling artifacts were written.  Seed the intermediate bundle
from the saved loc_ema native artifact, then rerun only dist_rolling with the
stage1 native preset and merge the result back into the full-scope run.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

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

from extreme_price_movements.model_loader import load_alpha_models
from scripts.run_top2_recency_pipeline_20260611 import (
    DATA_ROOT,
    DIST_ROLLING,
    FEATURE_SOURCE_RUN_ID,
    LOC_EMA,
    POLICY_SLICE_SOURCE_RUN_ID,
    SOURCE_RUN_ID,
    STAGE1_RUN_ID,
    STAGE3_RUN_ID,
    _append,
    _build_recent_tail_slice_plan,
    _pipeline_cmd,
    _require_file,
    _run_step,
    _train_env,
    _winner_paths,
)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with tmp.open("rb") as f:
        pickle.load(f)
    os.replace(tmp, path)


def _model_metrics(conf: dict[str, Any]) -> dict[str, Any]:
    race = conf.get("model")
    metrics: dict[str, Any] = {}
    for obj in (race, getattr(race, "best_model", None)):
        try:
            metrics.update(dict(getattr(obj, "metrics", {}) or {}))
        except Exception:
            pass
    metrics.setdefault("fit_status", "trained")
    for src, dst in (
        ("lift30", "en_lift30"),
        ("Lift@30", "en_lift30"),
        ("precision10", "prec10"),
        ("auc", "auc"),
        ("AUC", "auc"),
    ):
        if dst not in metrics and src in metrics:
            metrics[dst] = metrics[src]
    return metrics


def _merge_seed_bundle(path: Path, seed_bundle: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            existing = loaded

    bundle = dict(existing)
    alpha = dict(bundle.get("alpha_models") or {})
    short_models = dict(alpha.get("short") or {})
    short_models.update((seed_bundle.get("alpha_models") or {}).get("short") or {})
    alpha["short"] = short_models
    bundle["alpha_models"] = alpha

    for key in ("base_variant_models", "spike_models", "specialist_models"):
        bundle.setdefault(key, seed_bundle.get(key, {}))
    blocked = set(str(s) for s in (bundle.get("blocked_strategy_ids") or []))
    blocked.update(str(s) for s in (seed_bundle.get("blocked_strategy_ids") or []))
    bundle["blocked_strategy_ids"] = sorted(blocked)

    diag = dict(bundle.get("alpha_fit_diagnostics") or {})
    diag.update(seed_bundle.get("alpha_fit_diagnostics") or {})
    bundle["alpha_fit_diagnostics"] = diag
    _atomic_pickle(path, bundle)


def seed_fullscope_loc_ema_intermediate() -> None:
    native_dir = DATA_ROOT / "artifacts" / STAGE3_RUN_ID / "models" / "native"
    loc_model = native_dir / f"short_{LOC_EMA}_H10" / "model.joblib"
    _require_file(loc_model, "full-scope loc_ema native model")

    alpha_flat = load_alpha_models(str(native_dir))
    loc_conf = alpha_flat.get(LOC_EMA) or alpha_flat.get(f"short_{LOC_EMA}")
    if not isinstance(loc_conf, dict):
        raise SystemExit(f"loc_ema not loadable from native model store: {native_dir}")

    diag = _model_metrics(loc_conf)
    loc_conf["alpha_diag"] = diag
    for h_info in (loc_conf.get("models_by_h") or {}).values():
        if isinstance(h_info, dict):
            h_info.setdefault("alpha_diag", diag)

    seed_bundle = {
        "alpha_models": {"short": {LOC_EMA: loc_conf}},
        "base_variant_models": {},
        "spike_models": {},
        "specialist_models": {},
        "blocked_strategy_ids": [],
        "alpha_fit_diagnostics": {f"short_{LOC_EMA}_H10": diag},
    }

    run_root = DATA_ROOT / "artifacts" / STAGE3_RUN_ID
    _merge_seed_bundle(run_root / "base_models_intermediate.pkl", seed_bundle)
    trained_state_path = run_root / "models" / "trained_state.pkl"
    existing_state: dict[str, Any] = {}
    if trained_state_path.exists() and trained_state_path.stat().st_size > 0:
        with trained_state_path.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            existing_state = loaded
    state = dict(existing_state)
    state["bundle"] = dict(state.get("bundle") or {})
    state["bundle"]["alpha_models"] = seed_bundle["alpha_models"]
    state.setdefault("ts_trained", STAGE3_RUN_ID)
    _atomic_pickle(trained_state_path, state)
    _append(
        "Seeded full-scope intermediate/trained_state from saved loc_ema native "
        f"artifact with features={len(loc_conf.get('feat_cols') or [])}."
    )


def run_dist_rolling_fullscope_resume() -> None:
    base_winner, meta_winner = _winner_paths()
    _require_file(base_winner, "base recency winner")
    _require_file(meta_winner, "meta recency winner")
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
    env.update(
        {
            "EPM_BASE_STRATEGY_IDS": DIST_ROLLING,
            "EPM_LABEL_STRATEGY_IDS": DIST_ROLLING,
            "EPM_MERGE_EXISTING_BASE_MODELS": "1",
            "EPM_MERGE_EXISTING_TRAINED_STATE": "1",
            "EPM_LGBM_NATIVE_PRESET_PARAMS_ONLY": "0",
            "EPM_RECENCY_HPO_ONLY": "0",
            # The previous full-scope run died from memory pressure.  One
            # strategy plus single-threaded BLAS/LGBM is slower but avoids
            # competing native thread pools while preserving the full row set.
            "EPM_LGBM_N_JOBS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    _append(
        "Full-scope resume settings: dist_rolling only, native preset "
        "features+best_params, merge_existing_base_models=1, lgbm_n_jobs=1."
    )
    _run_step(
        "resume_stage3_train_base_dist_rolling_native_preset",
        _pipeline_cmd("train_base", STAGE3_RUN_ID),
        env,
    )
    run_root = DATA_ROOT / "artifacts" / STAGE3_RUN_ID
    _require_file(
        run_root / "models" / "native" / f"short_{DIST_ROLLING}_H10" / "model.joblib",
        "full-scope dist_rolling native model",
    )
    _require_file(run_root / "base_models_intermediate.pkl", "full-scope base models")


def main() -> int:
    _append(
        "Resume top2 full-scope missing base artifact: preserving loc_ema and "
        "rerunning dist_rolling after OOM with single-threaded worker caps."
    )
    seed_fullscope_loc_ema_intermediate()
    run_dist_rolling_fullscope_resume()
    _append("Resume top2 full-scope dist_rolling base completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
