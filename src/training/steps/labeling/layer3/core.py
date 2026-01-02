
"""
Layer 3 Core Orchestration

Main orchestration function for Layer 3 meta-modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
from scipy.special import expit

from .geometry_system import generate_geometries_adaptive, select_best_geometries_adaptive
from .feature_engineering import enhance_layer3_features_optimized, hierarchical_feature_filtering
from .model_training import train_dual_head_models
from .weighting_system import create_enhanced_weighting_schemes
from .reporting import generate_layer3_reports
from .utils import finalize_sample_weights, calculate_alpha_target, validate_feature_matrix, calculate_sample_weights_efficient

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

# Import causal framework modules for mechanism break detection
try:
    from src.training.steps.labeling.causal_surprise_events import CausalSurpriseDetector
    from src.training.steps.labeling.causal_specialists import CausalSpecialistManager
    CAUSAL_MODULES_AVAILABLE_LAYER3 = True
except ImportError:
    CAUSAL_MODULES_AVAILABLE_LAYER3 = False

logger = logging.getLogger(__name__)

CAUSAL_CONTEXT_KEYS = {
    "specialist_predictions",
    "causal_graph",
    "causal_targets",
    "causal_events",
    "causal_sample_weight",
    "surprise_summary",
    "geometry_metrics",
    "geometry_trials",
    "label_batch_metadata",
    "model_race_seed_models",
    "discovery_bootstrap_samples",
    "specialist_train_workers",
    "specialist_registration_workers",
    "treatment_max_features",
    "treatment_min_coverage",
    "model_race_target_precision",
    "model_race_plateau_patience",
    "dataset_fingerprint",
}

CAUSAL_HYPERPARAM_KEYS = (
    "discovery_bootstrap_samples",
    "specialist_train_workers",
    "specialist_registration_workers",
    "treatment_max_features",
    "treatment_min_coverage",
    "model_race_target_precision",
    "model_race_plateau_patience",
)


def _normalize_causal_context(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Merge causal context sources and determine if causal mode should be enforced."""
    context: Dict[str, Any] = {}
    for key in ("causal_context", "layer2_artifacts"):
        value = cfg.get(key)
        if isinstance(value, dict):
            context.update(value)
    for key in CAUSAL_CONTEXT_KEYS:
        if key in cfg and key not in context:
            context[key] = cfg[key]

    framework_type = cfg.get("framework_type")
    if not framework_type:
        framework_type = "modern_de_prado_causal" if cfg.get("enable_causal_framework") else "afml_legacy"
        cfg["framework_type"] = framework_type

    specialist_predictions = context.get("specialist_predictions")
    causal_targets = context.get("causal_targets")
    causal_mode = (
        framework_type == "modern_de_prado_causal"
        or bool(specialist_predictions)
        or bool(causal_targets)
    )

    return context, causal_mode


def _propagate_causal_hyperparams(cfg: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Bubble up Layer 2 causal hyperparameters so Layer 3 can introspect/report them."""
    if not context:
        return
    inherited = {}
    for key in CAUSAL_HYPERPARAM_KEYS:
        if key in context and key not in cfg:
            cfg[key] = context[key]
            inherited[key] = context[key]
    if inherited:
        cfg.setdefault("causal_hyperparams", {}).update(inherited)


def _inject_causal_features(
    df: pd.DataFrame,
    causal_events: Optional[pd.DataFrame],
    context: Dict[str, Any]
) -> List[str]:
    """Augment dataframe with causal diagnostics to make them available for geometry + models."""
    added_cols: List[str] = []
    if isinstance(causal_events, pd.DataFrame) and not causal_events.empty:
        event_flag = pd.Series(0, index=df.index)
        overlapping_idx = df.index.intersection(causal_events.index)
        event_flag.loc[overlapping_idx] = 1
        df["causal_event_flag"] = event_flag
        added_cols.append("causal_event_flag")

        if "surprise_strength" in causal_events.columns:
            df["causal_event_strength"] = (
                causal_events["surprise_strength"]
                .reindex(df.index)
                .fillna(0.0)
            )
            added_cols.append("causal_event_strength")
        if "surprise_consensus" in causal_events.columns:
            df["causal_event_consensus"] = (
                causal_events["surprise_consensus"]
                .reindex(df.index)
                .fillna(0.0)
            )
            added_cols.append("causal_event_consensus")

    surprise_summary = context.get("surprise_summary")
    if isinstance(surprise_summary, dict):
        density = surprise_summary.get("density")
        if density is not None:
            df["causal_surprise_density"] = density
            added_cols.append("causal_surprise_density")

    return added_cols


def _apply_causal_targets(
    df: pd.DataFrame,
    target_col: str,
    y_alpha: np.ndarray,
    y_prob: np.ndarray,
    w_alpha: np.ndarray,
    w_prob: np.ndarray,
    causal_targets: Optional[pd.DataFrame],
    causal_sample_weight: Optional[pd.Series]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[pd.Series], Dict[str, Any]]:
    """Blend causal targets + weights into existing alpha/probability definitions."""
    if causal_targets is None or causal_targets.empty:
        return y_alpha, y_prob, w_alpha, w_prob, None, {}

    aligned_targets = causal_targets.reindex(df.index)
    effect_cols = [c for c in aligned_targets.columns if c.endswith("_effect")]
    residual_cols = [c for c in aligned_targets.columns if c.endswith("_residual")]
    direction_cols = [c for c in aligned_targets.columns if c.endswith("_direction")]

    alpha_series = pd.Series(y_alpha, index=df.index, dtype=float)
    prob_series = pd.Series(y_prob, index=df.index, dtype=float)
    w_alpha_series = pd.Series(w_alpha, index=df.index, dtype=float)
    w_prob_series = pd.Series(w_prob, index=df.index, dtype=float)

    metadata: Dict[str, Any] = {}

    if effect_cols:
        alpha_from_effect = aligned_targets[effect_cols].mean(axis=1)
        alpha_series = alpha_from_effect.fillna(alpha_series)
        metadata["effect_columns"] = effect_cols
    elif residual_cols:
        alpha_from_residual = aligned_targets[residual_cols].mean(axis=1)
        alpha_series = alpha_from_residual.fillna(alpha_series)
        metadata["effect_columns"] = residual_cols

    if direction_cols:
        prob_from_direction = aligned_targets[direction_cols].mean(axis=1)
        prob_series = prob_from_direction.fillna(prob_series)
        metadata["direction_columns"] = direction_cols
    elif effect_cols:
        # Map alpha effect to probability via logistic squashing
        prob_series = pd.Series(expit(alpha_series.values), index=df.index)
        metadata["direction_columns"] = effect_cols

    weight_source = None
    if causal_sample_weight is not None:
        causal_weights = pd.Series(causal_sample_weight, index=df.index).fillna(1.0)
        w_alpha_series *= causal_weights
        w_prob_series *= causal_weights
        weight_source = "causal_sample_weight"
    elif "sample_weight" in aligned_targets.columns:
        sample_weights = aligned_targets["sample_weight"].fillna(1.0)
        w_alpha_series *= sample_weights
        w_prob_series *= sample_weights
        weight_source = "causal_targets.sample_weight"

    metadata["weight_source"] = weight_source
    metadata["target_rows"] = int(aligned_targets.dropna(how="all").shape[0])

    return (
        alpha_series.to_numpy(),
        prob_series.to_numpy(),
        w_alpha_series.to_numpy(),
        w_prob_series.to_numpy(),
        prob_series,
        metadata,
    )


def _build_causal_summary(
    context: Dict[str, Any],
    specialist_predictions: Dict[str, Any],
    causal_targets: Optional[pd.DataFrame],
    causal_events: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["framework_type"] = context.get("framework_type") or "modern_de_prado_causal"
    summary["specialist_count"] = len(specialist_predictions or {})
    summary["causal_event_count"] = int(len(causal_events)) if isinstance(causal_events, pd.DataFrame) else 0
    summary["surprise_density"] = None
    surprise_summary = context.get("surprise_summary")
    if isinstance(surprise_summary, dict):
        summary["surprise_density"] = surprise_summary.get("density")
    if isinstance(causal_targets, pd.DataFrame):
        summary["causal_target_columns"] = len(causal_targets.columns)
    summary["dataset_fingerprint"] = context.get("dataset_fingerprint")
    return summary

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Initialize config first
    cfg = config if isinstance(config, dict) else {}
    context, causal_mode = _normalize_causal_context(cfg)
    _propagate_causal_hyperparams(cfg, context)
    cfg['causal_context'] = context

    # Derive framework toggles
    cfg['enable_causal_framework'] = cfg.get('enable_causal_framework', True) or causal_mode
    enable_causal_framework = cfg['enable_causal_framework']
    enable_causal_mechanism_breaks = cfg.get('enable_causal_mechanism_breaks', True) or causal_mode

    if enable_causal_framework and not CAUSAL_MODULES_AVAILABLE_LAYER3:
        raise RuntimeError("Layer 3 causal mode requested but causal modules are unavailable.")

    # Merge Layer 2 artifacts / caches
    specialist_predictions = cfg.get('specialist_predictions') or context.get('specialist_predictions') or {}
    cfg['specialist_predictions'] = specialist_predictions
    causal_graph = cfg.get('causal_graph') or context.get('causal_graph') or {}
    cfg['causal_graph'] = causal_graph
    causal_events = context.get('causal_events')
    causal_targets = context.get('causal_targets')
    causal_sample_weight = context.get('causal_sample_weight')
    causal_geometry_trials = context.get('geometry_trials')
    causal_geometry_metrics = context.get('geometry_metrics')
    label_batch_metadata = context.get('label_batch_metadata')

    # Promote model race seeds if provided
    if context.get('model_race_seed_models'):
        cfg.setdefault('model_race_seed_models', context['model_race_seed_models'])

    tprint_info("🚀 Layer 3: Starting Multi-Geometry Meta-Models Pipeline")
    tprint_info(f"   📊 Input data: {len(oof_df)} rows, {len(base_model_cols)} base features")
    tprint_info(f"   🎯 Target column: {target_col}")
    tprint_info(f"   🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Memory optimization: avoid full copy, use view when possible
    memory_mode = "view" if cfg.get('fast_mode', False) else "copy"
    tprint_info(f"   💾 Memory mode: {memory_mode}")
    df = oof_df.copy(deep=False) if cfg.get('fast_mode', False) else oof_df.copy()
    tprint_info(f"   📊 Working data shape: {df.shape}")

    # Inject causal diagnostic features when available
    causal_feature_cols: List[str] = []
    if enable_causal_framework:
        causal_feature_cols = _inject_causal_features(df, causal_events, context)
        if causal_feature_cols:
            tprint_info(f"   🧬 Added {len(causal_feature_cols)} causal diagnostic features")
        else:
            tprint_info("   🧬 No causal diagnostic features were added")

    # Ensure Layer 2 causal sample weights propagate to probability weighting
    if causal_sample_weight is not None:
        if isinstance(causal_sample_weight, pd.Series):
            layer2_weight = causal_sample_weight.reindex(df.index)
        else:
            layer2_weight = pd.Series(causal_sample_weight, index=df.index)
    causal_summary = _build_causal_summary(context, specialist_predictions, causal_targets, causal_events)
    cfg['causal_summary'] = causal_summary

    # Initialize timestamps and paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcomes_dir = Path(cfg.get('outcomes_dir', 'outcomes'))
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    tprint_info(f"   📂 Outcomes directory: {outcomes_dir}")

    # Check for causal framework configuration

    tprint_info("   🔍 Layer 3: Checking causal framework configuration...")
    tprint_info(f"      - Causal mechanism breaks: {enable_causal_mechanism_breaks}")
    tprint_info(f"      - Specialist predictions available: {len(specialist_predictions) > 0}")
    tprint_info(f"      - Causal graph available: {len(causal_graph) > 0}")
    tprint_info(f"      - Causal modules available: {CAUSAL_MODULES_AVAILABLE_LAYER3}")

    if enable_causal_mechanism_breaks and CAUSAL_MODULES_AVAILABLE_LAYER3 and specialist_predictions:
        tprint_info("   🔬 Layer 3: Activating Causal Framework - Mechanism Break Detection")
        tprint_info("      - Computing mechanism break features...")
        mechanism_break_features = _compute_mechanism_break_features(
            df, specialist_predictions, causal_graph
        )

        if mechanism_break_features:
            tprint_info("      - Adding mechanism break features to dataframe...")
            for col, values in mechanism_break_features.items():
                df[col] = values
                tprint_info(f"         • Added: {col}")
            tprint_success(f"   ✅ Layer 3: Added {len(mechanism_break_features)} mechanism break features")
            tprint_info(f"      - Updated dataframe shape: {df.shape}")
        else:
            tprint_warning("   ⚠️ Layer 3: No mechanism break features computed")
    elif enable_causal_mechanism_breaks and not CAUSAL_MODULES_AVAILABLE_LAYER3:
        tprint_warning("   ⚠️ Layer 3: Causal mechanism breaks requested but modules not available")
    elif enable_causal_mechanism_breaks and not specialist_predictions:
        tprint_warning("   ⚠️ Layer 3: Causal mechanism breaks requested but no specialist predictions provided")
    else:
        tprint_info("   📊 Layer 3: Using traditional meta-modeling approach")

    # ---------------------------------------------------------
    # 1. Feature Engineering with Layer 0 Integration
    # ---------------------------------------------------------
    tprint_info("🔧 PHASE 1: Enhanced Feature Engineering")
    
    # Base feature preparation
    safe_base_cols = [c for c in base_model_cols if c in df.columns]
    if safe_base_cols:
        df[safe_base_cols] = df[safe_base_cols].fillna(0.5)
        tprint_success(f"✅ Prepared {len(safe_base_cols)} base features")

    # Market data alignment
    if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
        market_cols_added = []
        for c in ['volume', 'high', 'low', 'close']:
            if c in market_data.columns:
                df[c] = market_data[c].reindex(df.index)
                market_cols_added.append(c)
        tprint_success(f"✅ Aligned {len(market_cols_added)} market data columns: {market_cols_added}")

    # Generate standard Layer 3 features
    initial_feature_count = len(df.columns)

    # Fast mode: skip expensive feature generation
    if cfg.get('fast_mode', False):
        tprint_info("⚡ Fast mode: skipping standard Layer 3 feature generation")
        standard_features_added = 0
    else:
        try:
            from src.training.steps.labeling.label_based_layer_3 import generate_layer3_features
            df = generate_layer3_features(df, safe_base_cols)
            standard_features_added = len(df.columns) - initial_feature_count
            tprint_success(f"✅ Generated {standard_features_added} standard Layer 3 features")
        except Exception as e:
            tprint_warning(f"⚠️ Standard Layer 3 feature generation failed: {e}")
            standard_features_added = 0

    # Enhanced Layer 0 feature integration
    pre_layer0_count = len(df.columns)

    # Fast mode: skip expensive Layer 0 integration
    if cfg.get('fast_mode', False) or cfg.get('skip_layer0_integration', False):
        tprint_info("⚡ Skipping Layer 0 feature integration for speed")
        layer0_features_added = 0
    else:
        try:
            df = enhance_layer3_features_optimized(df, market_data, layer1_weight, fast_mode=cfg.get('fast_mode', False))
            layer0_features_added = len(df.columns) - pre_layer0_count
            tprint_success(f"✅ Enhanced with {layer0_features_added} Layer 0 optimized features")
            tprint_info(f"📈 Total features after Layer 0 integration: {len(df.columns)}")
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced feature generation failed: {e}")
            layer0_features_added = 0

    # ---------------------------------------------------------
    # 2. Data Preparation and Validation
    # ---------------------------------------------------------
    tprint_info("📊 PHASE 2: Data Preparation and Validation")
    
    # Prepare returns and volatility - vectorized
    if net_returns is None:
        if 'close' in df.columns:
            # Vectorized log returns calculation
            close_values = df['close'].values
            log_returns = np.zeros_like(close_values)
            log_returns[1:] = np.log(close_values[1:] / close_values[:-1])
            net_returns = pd.Series(log_returns, index=df.index)
            tprint_info("📈 Computed log returns from close prices")
        else:
            net_returns = pd.Series(0, index=df.index)
            tprint_warning("⚠️ No close column found, using zero returns")

    ret_series = net_returns.reindex(df.index)
    
    if 'volatility_1d' in df.columns:
        # Vectorized volatility processing
        vol_values = df['volatility_1d'].values
        vol_values = np.where(vol_values == 0, np.nan, vol_values)
        # Forward fill NaNs efficiently
        mask = np.isnan(vol_values)
        if np.any(mask):
            idx = np.where(~mask, np.arange(len(vol_values)), 0)
            np.maximum.accumulate(idx, out=idx)
            vol_values = vol_values[idx]
        vol_values = np.nan_to_num(vol_values, nan=0.001)
        vol_series = pd.Series(vol_values, index=df.index)
        tprint_success("✅ Using existing volatility_1d series")
    else:
        # Vectorized rolling volatility
        ret_values = ret_series.values
        vol_values = np.full_like(ret_values, 0.001, dtype=float)
        window = 24
        for i in range(window, len(ret_values)):
            vol_values[i] = np.std(ret_values[i-window:i])
        vol_series = pd.Series(vol_values, index=df.index)
        tprint_info("📊 Computed 24-bar rolling volatility")

    tprint_info(f"📈 Returns statistics: mean={ret_series.mean():.6f}, std={ret_series.std():.6f}")
    tprint_info(f"📊 Volatility statistics: mean={vol_series.mean():.6f}, std={vol_series.std():.6f}")

    # Define candidate features
    candidate_features = []
    candidate_features.extend(safe_base_cols)
    
    # Import centralized feature patterns
    from .feature_registry import get_layer3_feature_patterns, get_core_layer3_features
    
    # Add enhanced features from centralized registry
    enhanced_feature_patterns = get_layer3_feature_patterns()
    core_features = get_core_layer3_features()
    
    # Process feature patterns - highly vectorized and memory efficient
    pattern_counts = {}
    df_columns_array = np.array(df.columns.tolist(), dtype=str)  # Convert to numpy array for vectorized operations

    for pattern in enhanced_feature_patterns:
        # Vectorized string matching using numpy
        pattern_matches = np.char.find(df_columns_array, pattern) >= 0
        matching_cols = df_columns_array[pattern_matches].tolist()

        if matching_cols:
            candidate_features.extend(matching_cols)
            pattern_counts[pattern] = len(matching_cols)
    
    # Add core features
    for core_feat in core_features:
        if core_feat in df.columns:
            candidate_features.append(core_feat)
    
    meta_features = [c for c in list(dict.fromkeys(candidate_features)) if c in df.columns]
    
    # Clean features - vectorized operation for speed
    other_cols = [c for c in meta_features if c not in set(safe_base_cols)]
    if other_cols:
        # Vectorized cleaning
        X_other = df[other_cols].values
        inf_mask = np.isinf(X_other)
        inf_count = inf_mask.sum()
        nan_mask = np.isnan(X_other)
        nan_count = nan_mask.sum()

        # Replace inf and nan with 0.0
        X_other[inf_mask | nan_mask] = 0.0
        df[other_cols] = X_other

        tprint_info(f"🧹 Cleaned {inf_count} infinite and {nan_count} NaN values in features")

    tprint_success(f"📈 Final feature set: {len(meta_features)} total features")
        
    # Add Minimal Layer 3 Causal Features (Additive - Keep Existing)
    layer3_minimal_causal_enabled = cfg.get("layer3_minimal_causal_enabled", True)
    if layer3_minimal_causal_enabled and len(meta_features) > 5:
        tprint_info(">>> Layer 3: Adding Minimal Causal Meta-Features (Additive)...")
        try:
            from ..minimal_causal_features import generate_minimal_layer3_features
            
            # Generate confounder features
            from ..ohlcv_regime_features import OHLCVRegimeFeatures
            regime_generator = OHLCVRegimeFeatures(
                volatility_window=20,
                trend_window=20,
                enable_volatility_regimes=True,
                enable_trend_features=True,
                enable_microstructure_proxies=True,
                verbose=False
            )
            
            df_subset = df.loc[df.index.intersection(meta_features)]
            if len(df_subset) > 0:
                custom_features = regime_generator.generate_features(df_subset)
                
                # Create meta_features DataFrame
                meta_features_df = df[meta_features].copy()
                
                # Generate minimal causal meta-features (3-4 features only)
                minimal_causal_features = generate_minimal_layer3_features(
                    df=meta_features_df,
                    base_model_cols=safe_base_cols,
                    target_col=target_col,
                    custom_features=custom_features,
                    surprise_threshold=2.0,
                    rolling_window=20,
                    verbose=True
                )
                
                # Add minimal causal features to existing meta-features (ADDITIVE)
                causal_cols_added = []
                for col in minimal_causal_features.columns:
                    if col not in meta_features:
                        df[col] = minimal_causal_features[col]
                        meta_features.append(col)
                        causal_cols_added.append(col)
                
                tprint_success(f"   ✅ Added {len(causal_cols_added)} minimal causal meta-features to existing {len(meta_features)-len(causal_cols_added)} features")
                tprint_info(f"   - New total: {len(meta_features)} features (+{len(causal_cols_added)} causal)")
                
                # Store for downstream use
                layer3_minimal_causal_features = minimal_causal_features
            else:
                tprint_warning("   ⚠️ Insufficient data for minimal causal features")
                layer3_minimal_causal_features = pd.DataFrame(index=df.index)
                
        except Exception as e:
            tprint_warning(f"   ⚠️ Minimal Layer 3 causal features failed: {e}")
            layer3_minimal_causal_features = pd.DataFrame(index=df.index)
    else:
        tprint_info("⏭️ Skipping minimal Layer 3 causal features (disabled or insufficient features)")
        layer3_minimal_causal_features = pd.DataFrame(index=df.index)
        
    for pattern, count in pattern_counts.items():
        if count > 0:
            tprint_info(f"   - {pattern}: {count} features")

    # ---------------------------------------------------------
    # 3. Geometry Generation and Selection (4 Geometries)
    # ---------------------------------------------------------
    tprint_info("🔮 PHASE 3: Geometry Generation and Selection")
    
    # Get CUSUM signals for geometry generation
    cusum_cols = []
    cusum_df = None
    
    if market_data is not None:
        cusum_cols = [c for c in market_data.columns if 'trend_signal' in c or 'reversal_signal' in c]
        if cusum_cols:
            cusum_df = market_data[cusum_cols].reindex(df.index).fillna(0.0)
            tprint_success(f"✅ Found {len(cusum_cols)} CUSUM signals in market_data")
    
    if not cusum_cols:
        cusum_cols = [c for c in df.columns if 'trend_signal' in c or 'reversal_signal' in c]
        if cusum_cols:
            cusum_df = df[cusum_cols]
            tprint_info(f"📊 Found {len(cusum_cols)} CUSUM signals in main dataframe")
    
    if not cusum_cols or cusum_df is None:
        tprint_warning("⚠️ No CUSUM signals found, using fallback signals")
        cusum_df = pd.DataFrame(index=df.index)
        cusum_df['trend_signal_24'] = np.zeros(len(df))
        cusum_df['reversal_signal_24'] = np.zeros(len(df))
        cusum_cols = ['trend_signal_24', 'reversal_signal_24']

    # Generate geometries (4 by default, or fewer in fast mode)
    geometry_start_time = datetime.now()
    n_geoms = cfg.get('n_geometries', 2) if cfg.get('fast_mode', False) else cfg.get('n_geometries', 4)
    geometries_dict = generate_geometries_adaptive(
        base_signals=cusum_df,
        volatility=vol_series,
        mfe=df.get('mfe', pd.Series(0.02, index=df.index)),
        mae=df.get('mae', pd.Series(0.01, index=df.index)),
        n_geometries=n_geoms,
        fast_mode=cfg.get('fast_mode', False)
    )
    
    geometry_time = (datetime.now() - geometry_start_time).total_seconds()
    tprint_success(f"✅ Generated {len(geometries_dict)} geometries in {geometry_time:.2f}s")
    
    # Display geometry objectives
    for geom_id, geom_data in geometries_dict.items():
        tprint_info(f"   - {geom_id}: alpha={geom_data.get('alpha', 'N/A'):.2f}, activation={geom_data.get('activation', 'N/A')}")

    # Select best 4 geometries with correlation-based selection
    selection_start_time = datetime.now()
    y_target = df[target_col].fillna(0)
    
    top_k = cfg.get('top_k_geometries', 2) if cfg.get('fast_mode', False) else cfg.get('top_k_geometries', 4)
    selected_geoms_df = select_best_geometries_adaptive(
        geometries_dict, y_target, X=df[meta_features],
        model_type='classifier', objective_func='binary_logloss',
        top_k=top_k, correlation_weight=0.3
    )
    
    selection_time = (datetime.now() - selection_start_time).total_seconds()
    
    if selected_geoms_df.empty:
        tprint_warning("⚠️ No geometries selected, using fallback")
        selected_ids = ['fallback']
        geometries_dict['fallback'] = {
            'composite_signal': np.zeros(len(df)),
            'sigma_eff': vol_series.values,
            'alpha': 0.5,
            'activation': 'linear'
        }
    else:
        selected_ids = selected_geoms_df['id'].values.tolist()
        tprint_success(f"✅ Selected {len(selected_ids)} geometries in {selection_time:.2f}s: {selected_ids}")
        
        # Show selection scores
        if 'score' in selected_geoms_df.columns:
            for _, row in selected_geoms_df.iterrows():
                tprint_info(f"   - {row['id']}: score={row.get('score', 'N/A'):.4f}")

    # ---------------------------------------------------------
    # 4. Enhanced Weighting System
    # ---------------------------------------------------------
    tprint_info("⚖️  PHASE 4: Enhanced Weighting System")
    
    weighting_start_time = datetime.now()
    weighting_schemes = create_enhanced_weighting_schemes(
        layer1_weight,
        layer2_weight,
        layer2_weight_quality,
        market_data,
        df,
        causal_events=causal_events,
        causal_summary=causal_summary,
    )
    
    weighting_time = (datetime.now() - weighting_start_time).total_seconds()
    
    tprint_success(f"✅ Created {len(weighting_schemes)} weighting schemes in {weighting_time:.2f}s:")
    for scheme_name, weights in weighting_schemes.items():
        weight_stats = f"mean={weights.mean():.3f}, std={weights.std():.3f}"
        tprint_info(f"   - {scheme_name}: {weight_stats}")

    # ---------------------------------------------------------
    # 5. Target and Weight Preparation
    # ---------------------------------------------------------
    tprint_info("🎯 PHASE 5: Target and Weight Preparation")
    
    # Alpha target: Volatility-standardized returns (JIT compiled)
    y_alpha = calculate_alpha_target(ret_series.values, vol_series.values)
    alpha_stats = f"mean={y_alpha.mean():.4f}, std={y_alpha.std():.4f}"
    tprint_info(f"📈 Alpha target: {alpha_stats}")
    
    # Probability target: Binary (highly vectorized)
    if target_col in df.columns:
        # Vectorized numeric conversion and thresholding
        target_values = df[target_col].values
        # Use numpy operations for better performance
        numeric_values = np.asarray(pd.to_numeric(target_values, errors='coerce'))
        # Fill NaN with 0.5 and convert to binary
        valid_mask = np.isfinite(numeric_values)
        numeric_values[~valid_mask] = 0.5
        y_prob = (numeric_values >= 0.5).astype(np.int32)
        positive_rate = np.mean(y_prob)
        tprint_info(f"🎯 Probability target: {positive_rate:.1%} positive rate")
    else:
        # Vectorized return-based target
        y_prob = (ret_series.values > 0).astype(np.int32)
        positive_rate = np.mean(y_prob)
        tprint_info(f"🎯 Probability target (from returns): {positive_rate:.1%} positive rate")

    # Optimized weight calculation using efficient JIT-compiled function
    layer1_weights = layer1_weight.reindex(df.index).fillna(1.0).values if layer1_weight is not None else None
    layer2_weights = pd.Series(layer2_weight).reindex(df.index).fillna(1.0).values if layer2_weight is not None else None

    # Calculate alpha weights (volatility-based)
    w_alpha = calculate_sample_weights_efficient(
        ret_series.values, vol_series.values,
        layer1_weights=layer1_weights
    )
    alpha_weight_stats = f"mean={w_alpha.mean():.3f}, std={w_alpha.std():.3f}"
    tprint_info(f"⚖️  Alpha weights: {alpha_weight_stats}")

    # Calculate probability weights (Layer 2 composite)
    if layer2_weights is not None:
        w_prob = calculate_sample_weights_efficient(
            ret_series.values, vol_series.values,
            layer1_weights=layer1_weights,
            layer2_weights=layer2_weights
        )
        tprint_success("✅ Using provided Layer 2 weights")
    else:
        w_prob = calculate_sample_weights_efficient(
            ret_series.values, vol_series.values,
            layer1_weights=layer1_weights
        )
        tprint_info("📊 Using uniform probability weights")
    
    prob_weight_stats = f"mean={w_prob.mean():.3f}, std={w_prob.std():.3f}"
    tprint_info(f"⚖️  Probability weights: {prob_weight_stats}")

    causal_target_metadata: Dict[str, Any] = {}
    causal_prob_series: Optional[pd.Series] = None
    if enable_causal_framework:
        (
            y_alpha,
            y_prob,
            w_alpha,
            w_prob,
            causal_prob_series,
            causal_target_metadata,
        ) = _apply_causal_targets(
            df,
            target_col,
            y_alpha,
            y_prob,
            w_alpha,
            w_prob,
            causal_targets,
            causal_sample_weight,
        )
        if causal_prob_series is not None:
            df["causal_meta_prob"] = causal_prob_series
        if causal_target_metadata:
            cfg.setdefault("causal_target_metadata", {}).update(causal_target_metadata)

    # ---------------------------------------------------------
    # 6. Cross-Validation Setup
    # ---------------------------------------------------------
    tprint_info("🔄 PHASE 6: Cross-Validation Setup")
    
    from src.utils.purged_kfold import PurgedKFoldTime
    
    n_splits = cfg.get('cv_folds', 3) if cfg.get('fast_mode', False) else cfg.get('cv_folds', 5)
    if len(df) < n_splits * 2:
        tprint_error(f"❌ Insufficient data for CV ({len(df)} rows < {n_splits*2} minimum)")
        fallback_df = df.copy()
        fallback_df['meta_alpha'] = 0.0
        fallback_df['meta_prob'] = 0.5
        return fallback_df, {'error': 'Insufficient data'}

    cv = PurgedKFoldTime(n_splits=n_splits, purge=100, embargo=50)
    splits = list(cv.split(df))
    
    tprint_success(f"✅ Created {n_splits}-fold purged CV with {len(splits)} splits")
    tprint_info(f"   - Purge: 100 bars, Embargo: 50 bars")
    
    # Show split sizes
    split_sizes = [len(val_idx) for _, val_idx in splits]
    tprint_info(f"   - Validation split sizes: {split_sizes}")

    # ---------------------------------------------------------
    # 7. Feature Selection (CMI + De Prado)
    # ---------------------------------------------------------
    
    enable_advanced_selection = cfg.get("enable_advanced_feature_selection", True)
    
    # Fast mode: skip expensive feature selection
    if cfg.get('fast_mode', False) or cfg.get('skip_feature_selection', False):
        tprint_info("⚡ Fast mode: skipping advanced feature selection")
    elif enable_advanced_selection and len(meta_features) > 20:
        tprint_info("🔍 PHASE 7: Advanced Feature Selection")
        
        selection_start_time = datetime.now()
        
        try:
            # Hierarchical filtering for performance
            pre_filter_count = len(meta_features)
            # Use first base model column instead of ensemble_prob for CMI filtering
            base_col_for_cmi = safe_base_cols[0] if safe_base_cols else None
            base_predictions_series = df[base_col_for_cmi] if base_col_for_cmi and base_col_for_cmi in df.columns else pd.Series(0.5, index=df.index)
            
            X_filtered = hierarchical_feature_filtering(
                df[meta_features],
                pd.Series(y_prob, index=df.index),
                base_predictions_series,
                fast_mode=cfg.get('fast_mode', False)
            )

            meta_features = X_filtered.columns.tolist()
            selection_time = (datetime.now() - selection_start_time).total_seconds()

            reduction_pct = (1 - len(meta_features) / pre_filter_count) * 100
            tprint_success(f"✅ Feature selection completed in {selection_time:.2f}s")
            tprint_info(f"📉 Features reduced: {pre_filter_count} → {len(meta_features)} ({reduction_pct:.1f}% reduction)")
        except Exception as e:
            tprint_warning(f"⚠️ Advanced feature selection failed: {e}")
            tprint_info(f"📊 Using all {len(meta_features)} features")
    else:
        if not enable_advanced_selection:
            tprint_info("📊 Advanced feature selection disabled in config")
        else:
            tprint_info(f"📊 Skipping advanced selection (only {len(meta_features)} features)")

    # ---------------------------------------------------------
    # 8. Dual-Head Model Training
    # ---------------------------------------------------------
    
    tprint_info("🤖 PHASE 8: Dual-Head Model Training")
    
    # Prepare feature matrix - avoid copy if possible for memory efficiency
    X = df[meta_features]  # Use view instead of copy to save memory
    tprint_info(f"📊 Feature matrix shape: {X.shape}")
    
    # Validate and clean
    validation_start_time = datetime.now()
    X, y_alpha_valid = validate_feature_matrix(X, pd.Series(y_alpha, index=df.index))
    X, y_prob_valid = validate_feature_matrix(X, pd.Series(y_prob, index=df.index))
    
    # Align weights
    w_alpha_aligned = w_alpha[:len(X)]
    w_prob_aligned = w_prob[:len(X)]
    
    validation_time = (datetime.now() - validation_start_time).total_seconds()
    tprint_success(f"✅ Data validation completed in {validation_time:.2f}s")
    tprint_info(f"📊 Final training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train models
    training_start_time = datetime.now()
    model_results = train_dual_head_models(
        X, y_alpha_valid, y_prob_valid, w_alpha_aligned, w_prob_aligned, splits, cfg, cfg.get('fast_mode', False)
    )
    training_time = (datetime.now() - training_start_time).total_seconds()
    
    tprint_success(f"✅ Dual-Head training completed in {training_time:.2f}s")
    
    # Add predictions to dataframe
    df['meta_alpha'] = model_results['alpha_oof']
    df['meta_prob'] = model_results['prob_oof']
    
    # Show model performance
    if 'alpha_metrics' in model_results:
        alpha_metrics = model_results['alpha_metrics']
        tprint_info(f"📈 Alpha Head Performance:")
        tprint_info(f"   - Final IC: {alpha_metrics.get('final_ic', 'N/A'):.4f}")
        tprint_info(f"   - Selected Models: {alpha_metrics.get('selected_models', [])}")
    
    if 'prob_metrics' in model_results:
        prob_metrics = model_results['prob_metrics']
        tprint_info(f"🎯 Probability Head Performance:")
        tprint_info(f"   - Final AUC: {prob_metrics.get('final_auc', 'N/A'):.4f}")
        tprint_info(f"   - Final LogLoss: {prob_metrics.get('final_logloss', 'N/A'):.4f}")
        tprint_info(f"   - Selected Models: {prob_metrics.get('selected_models', [])}")

    # ---------------------------------------------------------
    # 9. Reporting and Outputs
    # ---------------------------------------------------------
    
    tprint_info("📊 PHASE 9: Reporting and Outputs")
    
    # Prepare models dictionary
    models_dict = {
        'alpha_models': model_results['alpha_models'],
        'prob_models': model_results['prob_models'],
        'calibrated_models': model_results['calibrated_models'],
        'alpha_metrics': model_results['alpha_metrics'],
        'prob_metrics': model_results['prob_metrics'],
        'geometry_models': selected_ids,
        'meta_features': meta_features
    }
    
    # Generate comprehensive reports
    reporting_start_time = datetime.now()
    try:
        generate_layer3_reports(
            df, models_dict, [], meta_features, target_col, outcomes_dir, ts, cfg
        )
        reporting_time = (datetime.now() - reporting_start_time).total_seconds()
        tprint_success(f"✅ Reports generated in {reporting_time:.2f}s")
        tprint_success(f"📂 Reports saved to: {outcomes_dir}")
    except Exception as e:
        tprint_warning(f"⚠️ Report generation failed: {e}")

    # Join causal targets to main DataFrame for Layer 4 consumption
    if causal_targets is not None and not causal_targets.empty:
        try:
            # Align causal targets with main DataFrame index
            aligned_causal_targets = causal_targets.reindex(df.index)
            
            # Add causal target columns to main DataFrame
            for col in aligned_causal_targets.columns:
                if col not in df.columns:  # Avoid overwriting existing columns
                    df[col] = aligned_causal_targets[col].fillna(0.0)
            
            tprint_success(f"✅ Joined {len(aligned_causal_targets.columns)} causal target columns to DataFrame")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to join causal targets to DataFrame: {e}")

    # Final summary
    total_time = (datetime.now() - datetime.strptime(ts, "%Y%m%d_%H%M%S")).total_seconds()
    tprint_success(f"🎉 Layer 3 Complete! Total time: {total_time:.2f}s")
    tprint_success(f"📈 Generated {len(df)} rows with meta_alpha and meta_prob")
    tprint_success(f"🤖 Final models: {len(model_results['alpha_models'])} alpha, {len(model_results['prob_models'])} probability")
    tprint_success(f"🔮 Selected geometries: {len(selected_ids)} ({selected_ids})")
    tprint_success(f"📊 Final feature count: {len(meta_features)}")

    return df, models_dict

def _compute_mechanism_break_features(
    df: pd.DataFrame,
    specialist_predictions: Dict[str, pd.Series],
    causal_graph: Dict[str, List[str]]
) -> Dict[str, pd.Series]:
    """
    Compute mechanism break features for Layer 3 meta-models.

    In the causal framework, Layer 3 detects when the "Physics" of the market
    is broken - when specialist predictions deviate from expected causal relationships.
    """
    try:
        tprint_info("🔍 Layer 3: Computing Mechanism Break Features...")
        tprint_info(f"   📊 Input data shape: {df.shape}")
        tprint_info(f"   🧠 Specialists available: {len(specialist_predictions)}")
        tprint_info(f"   🔗 Causal relationships: {len(causal_graph)}")

        mechanism_features = {}

        if not specialist_predictions:
            tprint_warning("   ⚠️ Layer 3: No specialist predictions available for mechanism break detection")
            return mechanism_features

        # Create surprise detector for mechanism break detection
        tprint_info("   🚨 Layer 3: Initializing causal surprise detector...")
        surprise_detector = CausalSurpriseDetector(verbose=False)

        # Register specialists and compute surprise scores
        tprint_info("   📝 Layer 3: Registering specialists and computing surprise scores...")
        surprise_scores = {}
        registered_count = 0

        for spec_name, predictions in specialist_predictions.items():
            tprint_info(f"      - Processing specialist: {spec_name}")

            # Create target (assume specialist predicts related market variable)
            if spec_name in df.columns:
                targets = df[spec_name]
                tprint_info("         • Target: column from dataframe")
            elif 'close' in df.columns:
                # Fallback: use close price as target
                targets = df['close']
                tprint_info("         • Target: close price (fallback)")
            else:
                tprint_warning("         • No suitable target available, skipping")
                continue

            # Align data
            common_idx = predictions.index.intersection(targets.index).intersection(df.index)
            tprint_info(f"         • Common samples: {len(common_idx)}")

            if len(common_idx) < 10:
                tprint_warning(f"         • Insufficient samples ({len(common_idx)}), skipping")
                continue

            pred_aligned = predictions.loc[common_idx]
            target_aligned = targets.loc[common_idx]

            # Register specialist
            surprise_detector.register_specialist(spec_name, pred_aligned, target_aligned)
            registered_count += 1

            # Compute surprise scores
            surprise_score = surprise_detector.compute_specialist_surprise(spec_name)
            surprise_scores[spec_name] = surprise_score
            tprint_info(f"         • Surprise score computed: {len(surprise_score)} samples")

        tprint_info(f"      - Specialists successfully processed: {registered_count}/{len(specialist_predictions)}")

        if not surprise_scores:
            tprint_warning("   ⚠️ Layer 3: No valid surprise scores computed")
            return mechanism_features

        # Aggregate surprise across specialists (consensus mechanism breaks)
        tprint_info("   🔄 Layer 3: Aggregating surprise scores across specialists...")
        surprise_df = pd.DataFrame(surprise_scores)
        tprint_info(f"      - Surprise matrix shape: {surprise_df.shape}")

        # Mechanism break features
        tprint_info("   🏗️ Layer 3: Computing mechanism break indicators...")

        mechanism_features['mechanism_break_max'] = surprise_df.max(axis=1)
        mechanism_features['mechanism_break_mean'] = surprise_df.mean(axis=1)
        mechanism_features['mechanism_break_consensus'] = (surprise_df > surprise_detector.surprise_threshold).sum(axis=1)
        mechanism_features['mechanism_break_any'] = (surprise_df > surprise_detector.surprise_threshold).any(axis=1).astype(int)

        tprint_info("      - Basic mechanism break features computed")
        tprint_info(f"         • Max surprise: {mechanism_features['mechanism_break_max'].max():.4f}")
        tprint_info(f"         • Mean surprise: {mechanism_features['mechanism_break_mean'].mean():.4f}")
        tprint_info(f"         • Consensus breaks: {mechanism_features['mechanism_break_any'].sum()}")

        # Structural breaks (longer-term mechanism failures)
        if len(surprise_df) > 20:
            tprint_info("      - Computing structural break indicators...")
            rolling_break_threshold = surprise_df.rolling(window=20, min_periods=10).quantile(0.95)
            mechanism_features['structural_break'] = (surprise_df > rolling_break_threshold).any(axis=1).astype(int)
            structural_breaks = mechanism_features['structural_break'].sum()
            tprint_info(f"         • Structural breaks detected: {structural_breaks}")
        else:
            tprint_info("      - Skipping structural breaks (insufficient data)")

        # Causal relationship strength indicators
        if causal_graph:
            tprint_info("      - Computing causal relationship indicators...")
            active_relationships = len([parents for parents in causal_graph.values() if parents])
            mechanism_features['causal_relationships_active'] = pd.Series(active_relationships, index=df.index)
            tprint_info(f"         • Active causal relationships: {active_relationships}")

        # Reindex to match main dataframe
        tprint_info("      - Reindexing features to match main dataframe...")
        original_total_samples = len(df)
        for feature_name in mechanism_features:
            original_length = len(mechanism_features[feature_name])
            mechanism_features[feature_name] = mechanism_features[feature_name].reindex(df.index, fill_value=0)
            final_length = len(mechanism_features[feature_name])
            coverage = final_length / original_total_samples if original_total_samples > 0 else 0
            tprint_info(f"         • {feature_name}: {original_length} → {final_length} samples ({coverage:.1%} coverage)")

        # Summary statistics
        total_features = len(mechanism_features)
        total_samples = len(df)
        avg_coverage = sum(len(feat.dropna()) / total_samples for feat in mechanism_features.values()) / total_features if total_features > 0 else 0

        tprint_success(f"✅ Layer 3: Computed {total_features} mechanism break features")
        tprint_info(f"   📊 Average feature coverage: {avg_coverage:.2%}")
        tprint_info(f"   🎯 Features created: {list(mechanism_features.keys())}")

        return mechanism_features

    except Exception as e:
        tprint_error(f"❌ Layer 3: Mechanism break feature computation failed: {e}")
        import traceback
        tprint_error(f"❌ Layer 3: Traceback: {traceback.format_exc()}")
        return {}
    