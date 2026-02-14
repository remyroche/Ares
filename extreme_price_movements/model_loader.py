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

import os
import json
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from extreme_price_movements.utils import tprint


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
            bundle["spike_model"] = spike_models.get("best") or spike_models.get("worst")
        else:
            tprint("WARNING: No spike models found in trained_state.pkl")
        
        # Load specialist models
        specialist_models = load_specialist_models(trained_state_path)
        if specialist_models:
            bundle["specialist_models"] = specialist_models
        else:
            tprint("WARNING: No specialist models found in trained_state.pkl")
    else:
        tprint(f"WARNING: Trained state file not found: {trained_state_path}")
    
    # 3. Load ridge position sizer weights
    if os.path.exists(ridge_path):
        bundle["ridge_weights"] = load_ridge_weights(ridge_path)
    else:
        tprint(f"WARNING: Ridge sizer weights not found: {ridge_path}")
    
    # Log summary
    alpha = bundle["alpha_models"]
    tprint(f"Model bundle loaded:")
    tprint(f"  Alpha models: long_mr={bool(alpha.get('long', {}).get('mr'))}, "
           f"long_tf={bool(alpha.get('long', {}).get('tf'))}, "
           f"short_mr={bool(alpha.get('short', {}).get('mr'))}, "
           f"short_tf={bool(alpha.get('short', {}).get('tf'))}")
    tprint(f"  Meta models: {list(bundle['meta_models'].keys())}")
    tprint(f"  Spike models: {list(bundle['spike_models'].keys())}")
    tprint(f"  Specialist models: {list(bundle['specialist_models'].keys())}")
    tprint(f"  Ridge weights: {list(bundle['ridge_weights'].keys())}")
    
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
    
    # Pattern: {side}_{kind}_H{h}
    pattern = re.compile(r"^(long|short)_(mr|tf)_H(\d+)$")
    
    # Group models by (side, kind) to handle multi-horizon
    models_by_side_kind = {}  # {(side, kind): {H: model_info}}
    
    for dirname in os.listdir(native_dir):
        match = pattern.match(dirname)
        if not match:
            continue
        
        side, kind, H = match.groups()
        H = int(H)
        model_dir = os.path.join(native_dir, dirname)
        
        if not os.path.isdir(model_dir):
            continue
        
        # Load model using ModelRace.load_native
        try:
            from extreme_price_movements.model_race import ModelRace
            model = ModelRace.load_native(model_dir)
            
            # Load sidecar for feat_cols
            sidecar_path = os.path.join(model_dir, "sidecar.pkl")
            feat_cols = []
            if os.path.exists(sidecar_path):
                with open(sidecar_path, "rb") as f:
                    sidecar = pickle.load(f)
                    # Try to get feat_cols from various sources
                    if "feat_cols" in sidecar:
                        feat_cols = sidecar["feat_cols"]
                    elif "columns" in sidecar:
                        feat_cols = sidecar["columns"]
            
            model_info = {
                "model": model,
                "feat_cols": feat_cols,
                "H": H,
            }
            
            key = (side, kind)
            if key not in models_by_side_kind:
                models_by_side_kind[key] = {}
            models_by_side_kind[key][H] = model_info
            
            tprint(f"  Loaded alpha model: {side}_{kind}_H{H} ({len(feat_cols)} features)")
            
        except Exception as e:
            tprint(f"  WARNING: Failed to load {dirname}: {e}")
            continue
    
    # Build final structure with best horizon and multi-horizon support
    for (side, kind), h_models in models_by_side_kind.items():
        if side not in alpha_models:
            alpha_models[side] = {}
        
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
        
        alpha_models[side][kind] = {
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
    
    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)
        
        # Extract bundle and meta_models
        bundle = state.get("bundle", state)
        meta_models = bundle.get("meta_models", {})
        
        if not meta_models:
            tprint("  No meta_models found in trained state")
            return {}
        
        tprint(f"  Loaded {len(meta_models)} meta models")
        return meta_models
        
    except Exception as e:
        tprint(f"  WARNING: Failed to load meta models: {e}")
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
        
        tprint(f"  Loaded risk params")
        return risk_params
        
    except Exception as e:
        tprint(f"  WARNING: Failed to load risk params: {e}")
        return {}


def load_bucket_params(run_id: str, data_root: str) -> dict:
    """Load optimized bucket parameters (TP/SL, exit policy) from optimise step.
    
    These parameters are generated by the tpsl_optimiser and define the
    exit policy (TP, SL, early exit, trailing profit thresholds) for each bucket.
    
    Args:
        run_id: Training run identifier
        data_root: Root data directory
        
    Returns:
        Dict of bucket parameters:
        {
            "LONG_MR": {"tp_mult": 3.0, "sl_mult": 1.0, ...},
            "LONG_TF": {...},
            "SHORT_MR": {...},
            "SHORT_TF": {...}
        }
    """
    bucket_params_path = os.path.join(data_root, "artifacts", run_id, "models", "bucket_params.json")
    
    if not os.path.exists(bucket_params_path):
        tprint(f"WARNING: Bucket params file not found: {bucket_params_path}")
        return {}
    
    try:
        with open(bucket_params_path, "r") as f:
            params = json.load(f)
        
        buckets = params.get("buckets", {})
        tprint(f"  Loaded bucket params for {len(buckets)} buckets")
        return buckets
        
    except Exception as e:
        tprint(f"  WARNING: Failed to load bucket params: {e}")
        return {}


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
            "bucket_params": {}
        }
    
    try:
        with open(trained_state_path, "rb") as f:
            state = pickle.load(f)
        
        # Also load ridge weights separately if not in state
        if "ridge_weights" not in state.get("bundle", {}):
            ridge_path = os.path.join(run_dir, "ridge_sizer", "sizer_weights.json")
            if os.path.exists(ridge_path):
                state["bundle"]["ridge_weights"] = load_ridge_weights(ridge_path)
        
        # Load bucket params (optimized exit policy)
        bucket_params = load_bucket_params(run_id, data_root)
        state["bucket_params"] = bucket_params
        
        tprint(f"Loaded full state for run_id={run_id}")
        return state
        
    except Exception as e:
        tprint(f"WARNING: Failed to load full state: {e}")
        return {
            "ts_trained": None,
            "bundle": load_model_bundle(run_id, data_root),
            "risk_params": {},
            "bucket_params": {}
        }
