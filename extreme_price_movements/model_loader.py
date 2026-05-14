"""
Model Bundle Loader for Live Inference.

This module provides functions to load trained models from persisted artifacts
for use in the live execution pipeline.

File Structure Reference:
    data/artifacts/{run_id}/
    ├── models/
    │   ├── native/
    │   │   ├── long_mr_H2/
    │   │   │   ├── model.lgb (or .cbm, .ubj)
    │   │   │   └── sidecar.pkl
    │   │   └── ... (other models)
    │   └── trained_state.pkl
    ├── ridge_sizer/
    │   └── sizer_weights.json
    └── ...
"""

import glob
import json
import os
import pickle
import re
from typing import Any, Optional

import joblib

from extreme_price_movements.entry_policy import flatten_bucket_policy
from extreme_price_movements.path_utils import resolve_mode_file
from extreme_price_movements.utils import tprint


def _normalize_market_mode(market_mode: str | None = None) -> str:
    mode = str(market_mode or os.environ.get("EPM_MARKET_MODE", "")).strip().lower()
    if mode in {"perp", "perps", "future", "futures"}:
        return "perps"
    if mode == "spot":
        return "spot"
    return "perps" if str(market_mode or "").endswith(("_perp", "_perps")) else "spot"


def _market_file_name(filename: str, market_mode: str | None = None) -> str:
    mode = _normalize_market_mode(market_mode)
    stem, ext = os.path.splitext(filename)
    for suffix in ("_spot", "_perps", "_perp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}_{mode}{ext}"


def _allow_legacy_market_fallback() -> bool:
    return str(
        os.environ.get("EPM_ALLOW_LEGACY_MARKET_FALLBACK", "")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def find_latest_run_id(data_root: str) -> Optional[str]:
    """Find the latest run_id from the artifacts directory.

    Args:
        data_root: Root data directory (e.g., "data")

    Returns:
        Latest run_id string (e.g., "20260213_080000") or None if no runs found
    """
    artifacts_dir = os.path.join(data_root, "artifacts")
    if not os.path.exists(artifacts_dir):
        tprint(f"WARNING: Artifacts directory not found: {artifacts_dir}")
        return None

    # List all directories that match the run_id pattern (YYYYMMDD_HHMMSS)
    run_pattern = re.compile(r"^\d{8}_\d{6}$")
    run_ids = []

    for name in os.listdir(artifacts_dir):
        if run_pattern.match(name):
            run_ids.append(name)

    if not run_ids:
        tprint(f"WARNING: No run directories found in {artifacts_dir}")
        return None

    # Sort by name (which sorts chronologically for YYYYMMDD_HHMMSS format)
    run_ids.sort(reverse=True)
    latest = run_ids[0]
    tprint(f"Found latest run_id: {latest}")
    return latest


def load_model_bundle(run_id: str, data_root: str) -> dict:
    """Load the complete model bundle for live inference.

    Assembles all model components needed for inference:
    - Alpha models (from native format)
    - Meta models (from pickle)
    - Spike models (from pickle)
    - Specialist models (from pickle)
    - Alpha OOF metrics (from pickle)
    - Quality gate report (from pickle)
    - EV decomposition (from pickle)
    - Ridge position sizer weights (from JSON)

    Args:
        run_id: Training run identifier (e.g., "20260213_080000")
        data_root: Root data directory

    Returns:
        Complete model bundle dict ready for inference:
        {
            "alpha_models": {
                "long": {"mr": {...}, "tf": {...}},
                "short": {"mr": {...}, "tf": {...}}
            },
            "meta_models": {...},
            "spike_models": {...},
            "specialist_models": {...},
            "alpha_oof_metrics": {...},
            "quality_gate_report": {...},
            "ev_decomposition": {...},
            "ridge_weights": {...}
        }
    """
    tprint(f"Loading model bundle for run_id={run_id}")

    bundle = {
        "alpha_models": {},
        "meta_models": {},
        "spike_models": {},
        "specialist_models": {},
        "ridge_weights": {},
        "ridge_offset_model": {},
        "alpha_oof_metrics": {},
        "quality_gate_report": {},
        "ev_decomposition": {},
    }

    # Paths
    run_dir = os.path.join(data_root, "artifacts", run_id)
    native_dir = os.path.join(run_dir, "models", "native")
    trained_state_path = os.path.join(run_dir, "models", "trained_state.pkl")
    ridge_path = os.path.join(run_dir, "ridge_sizer", "sizer_weights.json")

    # 1. Load alpha models from native format
    if os.path.exists(native_dir):
        bundle["alpha_models"] = load_alpha_models(native_dir)
    else:
        tprint(f"WARNING: Native models directory not found: {native_dir}")

    # 2. Load meta models, spike models, specialist models from pickle
    if os.path.exists(trained_state_path):
        # Load meta models
        meta_models = load_meta_models_from_pickle(trained_state_path)
        if meta_models:
            bundle["meta_models"] = meta_models
        else:
            tprint("WARNING: No meta models found in trained_state.pkl")

        # Load spike models
        spike_models = load_spike_models(trained_state_path)
        if spike_models:
            bundle["spike_models"] = spike_models
            # Also set spike_model for backward compatibility
            bundle["spike_model"] = spike_models.get("best") or spike_models.get(
                "worst"
            )
        else:
            tprint("WARNING: No spike models found in trained_state.pkl")

        # Load specialist models
        specialist_models = load_specialist_models(trained_state_path)
        if specialist_models:
            bundle["specialist_models"] = specialist_models
        else:
            tprint("WARNING: No specialist models found in trained_state.pkl")

        # Load alpha OOF metrics
        alpha_oof_metrics = load_alpha_oof_metrics(trained_state_path)
        if alpha_oof_metrics:
            bundle["alpha_oof_metrics"] = alpha_oof_metrics
        else:
            tprint("WARNING: No alpha OOF metrics found in trained_state.pkl")

        # Load quality gate report
        quality_gate_report = load_quality_gate_report(trained_state_path)
        if quality_gate_report:
            bundle["quality_gate_report"] = quality_gate_report
        else:
            tprint("WARNING: No quality gate report found in trained_state.pkl")

        # Load EV decomposition (full bundle)
        ev_decomposition = load_ev_decomposition(trained_state_path)
        if ev_decomposition:
            bundle["ev_decomposition"] = ev_decomposition
        else:
            tprint("WARNING: No EV decomposition found in trained_state.pkl")
    else:
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")

    # 3. Load ridge position sizer weights / offset optimizer metadata
    if os.path.exists(ridge_path):
        bundle["ridge_weights"] = load_ridge_weights(ridge_path)
    else:
        tprint(f"WARNING: Ridge sizer weights not found: {ridge_path}")

    ridge_model_path = _find_ridge_sizer_model_path(run_id, data_root)
    if ridge_model_path is not None:
        try:
            from extreme_price_movements.ridge_position_sizer import RidgePositionSizer

            ridge_sizer = RidgePositionSizer.load(ridge_model_path)
            bundle["ridge_offset_model"] = _ridge_offset_metadata_from_sizer(
                ridge_sizer
            )
        except Exception as e:
            tprint(f"WARNING: Failed to load ridge offset metadata: {e}")

    # Log summary
    alpha = bundle["alpha_models"]
    tprint("Model bundle loaded:")
    if isinstance(alpha, dict):
        alpha_summary = sorted(list(alpha.keys()))
    else:
        alpha_summary = []
    tprint(f"  Alpha models: {alpha_summary}")
    tprint(f"  Meta models: {list(bundle['meta_models'].keys())}")
    tprint(f"  Spike models: {list(bundle['spike_models'].keys())}")
    tprint(f"  Specialist models: {list(bundle['specialist_models'].keys())}")
    tprint(f"  Alpha OOF metrics: {list(bundle['alpha_oof_metrics'].keys())}")
    tprint(
        f"  Quality gate report: {'present' if bundle['quality_gate_report'] else 'missing'}"
    )
    tprint(
        f"  EV decomposition: {'present' if bundle['ev_decomposition'] else 'missing'}"
    )
    tprint(f"  Ridge weights: {list(bundle['ridge_weights'].keys())}")
    tprint(
        f"  Ridge offset model: {'present' if bundle['ridge_offset_model'] else 'missing'}"
    )

    return bundle


def load_alpha_models(native_dir: str) -> dict:
    """Load alpha models from native format directories.

    Scans directories matching pattern {side}_{kind}_H{h}/ and loads:
    - Model files (.lgb → LightGBM, .cbm → CatBoost, .ubj → XGBoost)
    - sidecar.pkl for feature columns and metadata

    Args:
        native_dir: Path to native models directory

    Returns:
        Dict of alpha models grouped by side and kind:
        {
            "long": {
                "mr": {"model": ModelRace, "feat_cols": [...], "H": 4, "models_by_h": {...}},
                "tf": {...}
            },
            "short": {...}
        }
    """
    alpha_models = {}

    if not os.path.exists(native_dir):
        tprint(f"WARNING: Native directory does not exist: {native_dir}")
        return alpha_models

    # Pattern: {strategy_id}_H{h}
    # We no longer use long/short or mr/tf, we strictly use strategy_id.
    pattern = re.compile(r"^(.+)_H(\d+)$")

    # Group models by (strategy_id) to handle multi-horizon
    models_by_strategy = {}  # {strategy_id: {H: model_info}}

    for dirname in os.listdir(native_dir):
        match = pattern.match(dirname)
        if not match:
            continue

        strategy_id, H = match.groups()
        H = int(H)
        model_dir = os.path.join(native_dir, dirname)

        if not os.path.isdir(model_dir):
            continue

        # Load model using ModelRace.load_native
        try:
            from extreme_price_movements.model_race import ModelRace

            model = ModelRace.load_native(model_dir)

            # Load feature columns from sidecar or columns.json.
            # Newer native exports persist feature lists in columns.json.
            sidecar_path = os.path.join(model_dir, "sidecar.pkl")
            columns_path = os.path.join(model_dir, "columns.json")
            feat_cols = []
            if os.path.exists(sidecar_path):
                with open(sidecar_path, "rb") as f:
                    sidecar = pickle.load(f)
                    # Try to get feat_cols from various sources
                    if "feat_cols" in sidecar:
                        feat_cols = sidecar["feat_cols"]
                    elif "columns" in sidecar:
                        feat_cols = sidecar["columns"]
                    elif "selected_features" in sidecar:
                        feat_cols = sidecar["selected_features"]

            if (not feat_cols) and os.path.exists(columns_path):
                with open(columns_path, "r") as f:
                    columns_info = json.load(f)
                if isinstance(columns_info, dict):
                    feat_cols = (
                        columns_info.get("feat_cols")
                        or columns_info.get("selected_features")
                        or columns_info.get("columns")
                        or []
                    )
                elif isinstance(columns_info, list):
                    feat_cols = columns_info

            model_info = {
                "model": model,
                "feat_cols": feat_cols,
                "H": H,
            }

            if strategy_id not in models_by_strategy:
                models_by_strategy[strategy_id] = {}
            models_by_strategy[strategy_id][H] = model_info

            tprint(
                f"  Loaded alpha model: {strategy_id}_H{H} ({len(feat_cols)} features)"
            )

        except Exception as e:
            tprint(f"  WARNING: Failed to load {dirname}: {e}")
            continue

    # Build final structure with best horizon and multi-horizon support
    for strategy_id, h_models in models_by_strategy.items():
        # Select best horizon (highest H for now, could use metrics)
        best_H = max(h_models.keys())
        best_info = h_models[best_H]

        # Build models_by_h for multi-horizon averaging
        models_by_h = {}
        for H, info in h_models.items():
            models_by_h[H] = {
                "model": info["model"],
                "feat_cols": info["feat_cols"],
            }

        # Use flat dict structure: alpha_models[strategy_id]
        alpha_models[strategy_id] = {
            "model": best_info["model"],
            "feat_cols": best_info["feat_cols"],
            "H": best_H,
            "models_by_h": models_by_h,
        }

    return alpha_models


def load_meta_models_from_pickle(trained_state_path: str) -> dict:
    """Load meta models from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of MetaModel objects keyed by {side}_{kind}:
        {
            "long_mr": MetaModel,
            "long_tf": MetaModel,
            "short_mr": MetaModel,
            "short_tf": MetaModel,
            # Plus classifier variants if available
        }
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    def _ensure_meta_aliases(meta_models: dict) -> dict:
        if not isinstance(meta_models, dict):
            return {}
        meta_models = dict(meta_models)
        # Extract base names by removing suffixes like _reg, _clf, _early_inval
        base_names = set()
        for k in meta_models.keys():
            if k.endswith("_reg"):
                base_names.add(k[:-4])
            elif k.endswith("_clf"):
                base_names.add(k[:-4])
            elif k.endswith("_early_inval"):
                base_names.add(k[:-12])
            else:
                base_names.add(k)

        for base in base_names:
            reg = meta_models.get(f"{base}_reg")
            clf = meta_models.get(f"{base}_clf") or meta_models.get(
                f"{base}_early_inval"
            )
            if base not in meta_models and reg is not None:
                meta_models[base] = reg
            if f"{base}_clf" not in meta_models and clf is not None:
                meta_models[f"{base}_clf"] = clf
        return meta_models

    def _extract_meta_models(state_obj) -> dict:
        bundle = (
            state_obj.get("bundle", state_obj) if isinstance(state_obj, dict) else {}
        )
        meta_models = bundle.get("meta_models", {}) if isinstance(bundle, dict) else {}
        return _ensure_meta_aliases(meta_models)

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)
        meta_models = _extract_meta_models(state)
        if meta_models:
            tprint(f"  Loaded {len(meta_models)} meta models")
            return meta_models
        tprint("  No meta_models found in trained state")
    except Exception as e:
        tprint(f"  WARNING: Failed to load meta models from trained_state.pkl: {e}")

    meta_state_path = os.path.join(
        os.path.dirname(trained_state_path), "model_state_meta.pkl"
    )
    if not os.path.exists(meta_state_path):
        return {}

    try:
        meta_state = joblib.load(meta_state_path)
        meta_models = _extract_meta_models(meta_state)
        if meta_models:
            tprint(f"  Loaded {len(meta_models)} meta models from model_state_meta.pkl")
            return meta_models
        tprint("  No meta_models found in model_state_meta.pkl")
    except Exception as e:
        tprint(f"  WARNING: Failed to load meta models from model_state_meta.pkl: {e}")
    return {}


def load_ridge_weights(ridge_path: str) -> dict:
    """Load ridge position sizer weights from JSON file.

    Args:
        ridge_path: Path to sizer_weights.json file

    Returns:
        Dict of ridge weights per bucket:
        {
            "long_mr": {"coefs": [...], "intercept": ..., "feature_names": [...]},
            "long_tf": {...},
            ...
        }
    """
    if not os.path.exists(ridge_path):
        tprint(f"WARNING: Ridge weights file not found: {ridge_path}")
        return {}

    try:
        with open(ridge_path, "r") as f:
            weights = json.load(f)

        tprint(f"  Loaded ridge weights for {len(weights)} buckets")
        return weights

    except Exception as e:
        tprint(f"  WARNING: Failed to load ridge weights: {e}")
        return {}


def _find_ridge_sizer_model_path(run_id: str, data_root: str) -> Optional[str]:
    artifact_models_dir = os.path.join(data_root, "artifacts", run_id, "models")
    shared_models_dir = os.path.join(data_root, "models")
    legacy_shared_models_dir = os.path.join(
        "extreme_price_movements", data_root, "models"
    )
    preferred = [
        os.path.join(artifact_models_dir, f"ridge_position_sizer_{run_id}.json"),
        os.path.join(shared_models_dir, f"ridge_position_sizer_{run_id}.json"),
        os.path.join(legacy_shared_models_dir, f"ridge_position_sizer_{run_id}.json"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    candidates = sorted(
        glob.glob(os.path.join(artifact_models_dir, "ridge_position_sizer_*.json"))
        + glob.glob(os.path.join(shared_models_dir, "ridge_position_sizer_*.json"))
        + glob.glob(
            os.path.join(legacy_shared_models_dir, "ridge_position_sizer_*.json")
        )
    )
    return candidates[-1] if candidates else None


def _ridge_offset_metadata_from_sizer(ridge_sizer: Any) -> dict:
    if ridge_sizer is None:
        return {}
    offset_bundle = getattr(ridge_sizer, "limit_offset_model_bundle_", None) or {}
    return {
        "base_name": offset_bundle.get("base_name"),
        "smoother_name": offset_bundle.get("smoother_name"),
        "features": list(getattr(ridge_sizer, "limit_offset_features_", None) or []),
        "diag": dict(getattr(ridge_sizer, "limit_offset_diag_", None) or {}),
    }


def load_spike_models(trained_state_path: str) -> dict:
    """Load spike models from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of spike models:
        {
            "best": {"gmm": GMM, "scaler": StandardScaler, "columns": [...]},
            "worst": {...}
        }
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        bundle = state.get("bundle", state)
        spike_models = bundle.get("spike_models", {})

        if not spike_models:
            tprint("  No spike_models found in trained state")
            return {}

        tprint(f"  Loaded {len(spike_models)} spike models")
        return spike_models

    except Exception as e:
        tprint(f"  WARNING: Failed to load spike models: {e}")
        return {}


def load_specialist_models(trained_state_path: str) -> dict:
    """Load specialist models from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of specialist models:
        {
            "trap_model": {"gmm": GMM, "scaler": StandardScaler, "columns": [...], ...},
            "gamma_model": GammaSpecialist
        }
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        bundle = state.get("bundle", state)
        specialist_models = bundle.get("specialist_models", {})

        if not specialist_models:
            tprint("  No specialist_models found in trained state")
            return {}

        tprint(f"  Loaded {len(specialist_models)} specialist models")
        return specialist_models

    except Exception as e:
        tprint(f"  WARNING: Failed to load specialist models: {e}")
        return {}


def load_alpha_oof_metrics(trained_state_path: str) -> dict:
    """Load alpha out-of-fold metrics from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of OOF metrics per model:
        {
            "long_mr": {"oof_preds": [...], "oof_targets": [...], "metrics": {...}},
            "long_tf": {...},
            "short_mr": {...},
            "short_tf": {...}
        }
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        bundle = state.get("bundle", state)
        alpha_oof_metrics = bundle.get("alpha_oof_metrics", {})

        if not alpha_oof_metrics:
            tprint("  No alpha_oof_metrics found in trained state")
            return {}

        tprint(f"  Loaded alpha OOF metrics for {len(alpha_oof_metrics)} models")
        return alpha_oof_metrics

    except Exception as e:
        tprint(f"  WARNING: Failed to load alpha OOF metrics: {e}")
        return {}


def load_quality_gate_report(trained_state_path: str) -> dict:
    """Load quality gate report from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of quality gate report:
        {
            "overall_pass": bool,
            "checks": [...],
            "warnings": [...],
            "failures": [...]
        }
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        bundle = state.get("bundle", state)
        quality_gate_report = bundle.get("quality_gate_report", {})

        if not quality_gate_report:
            tprint("  No quality_gate_report found in trained state")
            return {}

        tprint("  Loaded quality gate report")
        return quality_gate_report

    except Exception as e:
        tprint(f"  WARNING: Failed to load quality gate report: {e}")
        return {}


def load_ev_decomposition(trained_state_path: str) -> dict:
    """Load expected value decomposition from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of EV decomposition components:
        {
            "components": {...},
            "weights": {...},
            "metadata": {...}
        }
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        ev_decomposition = state.get("ev_decomposition", {})

        if not ev_decomposition:
            tprint("  No ev_decomposition found in trained state")
            return {}

        tprint("  Loaded EV decomposition")
        return ev_decomposition

    except Exception as e:
        tprint(f"  WARNING: Failed to load EV decomposition: {e}")
        return {}


def load_risk_params(trained_state_path: str) -> dict:
    """Load risk parameters from trained_state.pkl.

    Args:
        trained_state_path: Path to trained_state.pkl file

    Returns:
        Dict of risk parameters for position management
    """
    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {}

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        risk_params = state.get("risk_params", {})

        if not risk_params:
            tprint("  No risk_params found in trained state")
            return {}

        tprint("  Loaded risk params")
        return risk_params

    except Exception as e:
        tprint(f"  WARNING: Failed to load risk params: {e}")
        return {}


def _sizer_artifact_dirs(data_root: str, run_id: str) -> list[str]:
    """Return sizer artifact directories in lookup priority order."""
    root = os.path.join(data_root, "artifacts", run_id)
    return [
        os.path.join(root, "simple_position_sizer"),
        os.path.join(root, "ridge_sizer"),
    ]


def _first_existing_path(paths: list[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_bucket_params(run_id: str, data_root: str) -> dict:
    """Load optimized bucket parameters (TP/SL, exit policy) from optimise step.

    These parameters are generated by `simple_position_sizer.py` and define the
    exit policy (TP, SL, early exit, trailing profit thresholds) for each bucket.

    Args:
        run_id: Training run identifier
        data_root: Root data directory

    Returns:
        Dict of strategy/bucket parameters:
        {
            "strategy_id_or_legacy_bucket": {"tp_mult": 3.0, "sl_mult": 1.0, ...},
        }
    """
    model_dirs = _sizer_artifact_dirs(data_root, run_id)
    bucket_params_path = _first_existing_path(
        [os.path.join(model_dir, "strategy_params.json") for model_dir in model_dirs]
    )
    if bucket_params_path is None:
        tprint(
            "WARNING: Policy params file not found in any sizer artifact dir: "
            f"{model_dirs}"
        )
        return {}

    try:
        with open(bucket_params_path, "r") as f:
            params = json.load(f)

        buckets_raw = params.get("buckets", {})
        buckets = {k: flatten_bucket_policy(v) for k, v in buckets_raw.items()}
        tprint(
            f"  Loaded policy params from {bucket_params_path} "
            f"for {len(buckets)} buckets/strategies"
        )
        return buckets

    except Exception as e:
        tprint(f"  WARNING: Failed to load policy params: {e}")
        return {}


def load_booster_bundles(run_id: str, data_root: str) -> dict:
    model_dirs = _sizer_artifact_dirs(data_root, run_id)
    bundle_dir = _first_existing_path(
        [os.path.join(model_dir, "booster_bundles") for model_dir in model_dirs]
    )
    if bundle_dir is None or not os.path.isdir(bundle_dir):
        return {}
    bundles = {}
    for fname in os.listdir(bundle_dir):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(bundle_dir, fname)
        try:
            with open(path, "rb") as f:
                b = pickle.load(f)
            sid = b.get("strategy_id", fname[:-4])
            bundles[sid] = b
        except Exception as e:
            tprint(f"WARNING: Failed to load booster bundle {fname}: {e}")
    if bundles:
        tprint(f"  Loaded {len(bundles)} booster bundles from {bundle_dir}")
    return bundles


def load_regime_adaptors(
    run_id: str, data_root: str, market_mode: str | None = None
) -> dict:
    model_dirs = _sizer_artifact_dirs(data_root, run_id)
    adaptor_dir = _first_existing_path(
        [os.path.join(model_dir, "regime_adaptors") for model_dir in model_dirs]
    )
    if adaptor_dir is None or not os.path.isdir(adaptor_dir):
        return {}
    adaptors = {}
    market_mode = _normalize_market_mode(market_mode or data_root)
    preferred_name = _market_file_name("regime_adaptor.json", market_mode)
    for root, _, files in os.walk(adaptor_dir):
        if preferred_name in files:
            path = os.path.join(root, preferred_name)
        elif _allow_legacy_market_fallback() and "regime_adaptor.json" in files:
            path = os.path.join(root, "regime_adaptor.json")
        else:
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            sid = str(payload.get("strategy_id", os.path.basename(root)))
            adaptors[sid] = payload
        except Exception as e:
            tprint(f"WARNING: Failed to load regime adaptor {path}: {e}")
    if adaptors:
        tprint(f"  Loaded {len(adaptors)} regime adaptors from {adaptor_dir}")
    return adaptors


# Convenience function for full state loading
def load_full_state(run_id: str, data_root: str) -> dict:
    """Load complete training state including bundle and risk params.

    Args:
        run_id: Training run identifier
        data_root: Root data directory

    Returns:
        Dict with:
        {
            "ts_trained": Timestamp,
            "bundle": model_bundle,
            "risk_params": dict,
            "bucket_params": dict  # Optimized exit policy per bucket
        }
    """
    run_dir = os.path.join(data_root, "artifacts", run_id)
    trained_state_path = os.path.join(run_dir, "models", "trained_state.pkl")

    if not os.path.exists(trained_state_path):
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
        return {
            "ts_trained": None,
            "bundle": load_model_bundle(run_id, data_root),
            "risk_params": {},
            "bucket_params": {},
        }

    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)

        if isinstance(state, dict):
            bundle = state.get("bundle", {})
            if isinstance(bundle, dict):
                meta_models = bundle.get("meta_models", {})
                if not isinstance(meta_models, dict) or len(meta_models) == 0:
                    bundle["meta_models"] = load_meta_models_from_pickle(
                        trained_state_path
                    )
                state["bundle"] = bundle

        # Also load ridge weights separately if not in state
        if "ridge_weights" not in state.get("bundle", {}):
            ridge_path = os.path.join(run_dir, "ridge_sizer", "sizer_weights.json")
            if os.path.exists(ridge_path):
                state["bundle"]["ridge_weights"] = load_ridge_weights(ridge_path)

        # Load full RidgePositionSizer bundle (for full inference with calibration).
        # Historical runs may store this either under:
        #   data/artifacts/{run_id}/models/
        # or:
        #   data/models/
        ridge_model_path = _find_ridge_sizer_model_path(run_id, data_root)
        if ridge_model_path is not None:
            try:
                from extreme_price_movements.ridge_position_sizer import (
                    RidgePositionSizer,
                )

                state["ridge_sizer"] = RidgePositionSizer.load(ridge_model_path)
                tprint(f"  Loaded RidgePositionSizer from {ridge_model_path}")
                state.setdefault("bundle", {})
                state["bundle"][
                    "ridge_offset_model"
                ] = _ridge_offset_metadata_from_sizer(state["ridge_sizer"])
            except Exception as e:
                tprint(f"WARNING: Failed to load RidgePositionSizer: {e}")
        else:
            tprint(
                "  RidgePositionSizer model not found in artifact or shared model directories"
            )

        # Load bucket params (optimized exit policy)
        bucket_params = load_bucket_params(run_id, data_root)
        state["bucket_params"] = bucket_params

        # Load booster bundles (ET/LGBM/LGBM_clf fold models)
        state["booster_bundles"] = load_booster_bundles(run_id, data_root)
        state["regime_adaptors"] = load_regime_adaptors(run_id, data_root)

        tprint(f"Loaded full state for run_id={run_id}")
        return state

    except Exception as e:
        tprint(f"WARNING: Failed to load full state: {e}")

        # Try to load RidgePositionSizer even if state loading failed.
        ridge_model_path = _find_ridge_sizer_model_path(run_id, data_root)
        ridge_sizer = None
        if ridge_model_path is not None:
            try:
                from extreme_price_movements.ridge_position_sizer import (
                    RidgePositionSizer,
                )

                ridge_sizer = RidgePositionSizer.load(ridge_model_path)
                tprint(f"  Loaded RidgePositionSizer from fallback: {ridge_model_path}")
            except Exception as ridge_e:
                tprint(
                    f"WARNING: Failed to load RidgePositionSizer in fallback: {ridge_e}"
                )

        return {
            "ts_trained": None,
            "bundle": load_model_bundle(run_id, data_root),
            "risk_params": {},
            "bucket_params": {},
            "ridge_sizer": ridge_sizer,
        }
