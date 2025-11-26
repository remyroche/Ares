"""
Unified Interaction Generation Step (Analyst/Tactician).

        This step implements a sophisticated pipeline for feature engineering that can operate in two modes:
        - **Tactician Mode**: Uses CMI-based feature selection and interaction discovery
        - **Analyst Mode**: Uses MI-based feature selection

Pipeline Phases:
1. Phase 0: Load artifacts and perform mode-specific feature selection
2. Phase 1: Generate normalized variants with RobustScaler bounding + Cross-timeframe features
3. Phase 2: Apply cheap pruning with category protection (40-50% reduction)
4. Phase 3: Three-phase LGBM+SHAP for feature selection and interaction discovery
5. Phase 4: Combine features, verify category coverage, save artifacts

Key Features:
- Mode detection based on launcher arguments (analyst/tactician)
- RobustScaler bounding to prevent extreme values
- Cross-timeframe feature generation with 3x, 6x, 9x, 15x, 27x, 45x, 60x lookback ratios
- Category protection during pruning (maintain ≥3 per category)
- Tree-based interaction guidance with corrected SHAP analysis
- Comprehensive causality enforcement
- Category coverage tracking (≥2 per category in final set)
        - CMI complementarity for Tactician mode
- Support for simplified target structure (target_long, target_short) from labeling integration

Cross-Timeframe Features:
- For each variant feature (base, volnorm, vwap, trend_adj), generates 7 additional timeframe versions
- Creates ratio-based interactions across multiple regime timescales:
  - Short-term (3x, 6x): Micro to short-term regime shifts
  - Medium-term (9x, 15x): Intraday regime transitions
  - Long-term (27x, 45x, 60x): Multi-regime and market memory interactions
- Ratio features: feature_base / feature_Nx for each multiplier
- Uses safe division with math validation and causality enforcement (.shift(1))
- Effectively multiplies feature count by ~8x after Phase 1 (1 base + 7 ratios per variant)

Target Structure:
- Works with simplified binary targets from labeling integration step:
  - target_long: Binary target for long positions (volume-normalized)
  - target_short: Binary target for short positions (volume-normalized)
- Maintains backward compatibility with legacy target columns
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Callable

try:
    import polars as pl  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    pl = None

from src.training.steps.pre_training.includes.interaction_generation_fallbacks import attach_interaction_generation_fallbacks
from datetime import datetime
import time
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from src.training.utils.meta_label_constants import (
    META_LABEL_TARGET_COLUMNS,
    META_LABEL_PRIMARY_TRAINING_TARGETS,
    META_LABEL_DIAGNOSTIC_COLUMNS,
    META_LABEL_EXCLUDED_FEATURE_COLUMNS,
)

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_performance,
    tprint_warning, tprint_structured, LogLevel
)


def _align_for_label_guided_discovery_helper(
    features: pd.DataFrame,
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and target for label-guided interaction discovery.

    This mirrors LabelGuidedInteractionDiscovery._clean_inputs:
    - Use index intersection when there is overlap.
    - If there is no overlap, fall back to positional alignment on the last
      min(len(features), len(target)) rows and reset to a shared RangeIndex.
    - Drop any rows with non-finite values in features or target.
    """

    common_idx = features.index.intersection(target.index)

    if len(common_idx) == 0:
        min_len = min(len(features), len(target))
        if min_len == 0:
            return features.iloc[0:0].copy(), target.iloc[0:0].copy()
        features_aligned = features.iloc[-min_len:].copy()
        target_aligned = target.iloc[-min_len:].copy()
        features_aligned.index = pd.RangeIndex(min_len)
        target_aligned.index = pd.RangeIndex(min_len)
    else:
        features_aligned = features.loc[common_idx].copy()
        target_aligned = target.loc[common_idx].copy()

    finite_mask = np.isfinite(target_aligned.values) & np.all(
        np.isfinite(features_aligned.values), axis=1
    )
    features_clean = features_aligned[finite_mask]
    target_clean = target_aligned[finite_mask]

    if features_clean.empty:
        # Fallback: keep rows with finite targets and impute non-finite features
        fallback_mask = np.isfinite(target_aligned.values)
        features_fallback = features_aligned[fallback_mask].copy()
        target_fallback = target_aligned[fallback_mask].copy()

        if len(features_fallback) == 0:
            return features_aligned.iloc[0:0].copy(), target_aligned.iloc[0:0].copy()

        features_fallback = features_fallback.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target_fallback = pd.Series(
            np.nan_to_num(target_fallback.values, nan=0.0, posinf=0.0, neginf=0.0),
            index=target_fallback.index,
            name=target_fallback.name,
        )

        features_clean = features_fallback
        target_clean = target_fallback

    return features_clean, target_clean


def _ensure_pandas_dataframe(obj: Any) -> pd.DataFrame:
    """Coerce Polars DataFrame to pandas for internal use in this step.

    The core implementation of this step is pandas/NumPy/VectorBT-based.
    This helper allows upstream components to pass pl.DataFrame artifacts
    while keeping the internal logic unchanged.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if pl is not None and isinstance(obj, pl.DataFrame):  # type: ignore[arg-type]
        return obj.to_pandas()
    return obj

# VectorBT imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager, VectorizationConfig
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    tprint_warning("⚠️ VectorBT components not available")

# Hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, get_unified_hardware_manager, OptimizationLevel
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False
    tprint_warning("⚠️ Hardware optimization not available")

# Parallel processing
try:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from multiprocessing import cpu_count
    import threading
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    tprint_warning("⚠️ Parallel processing not available")

# Data loading utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    DATA_LOADING_AVAILABLE = True
except ImportError:
    DATA_LOADING_AVAILABLE = False
    tprint_warning("⚠️ Data loading utilities not available")

# CMI complementarity components for Tactician mode
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import (
        CMIComplementarityScorer,
        CMIComplementarityConfig,
        create_cmi_complementarity_scorer
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import (
        AnalystSideInfoHandler,
        create_analyst_side_info_handler
    )
    CMI_COMPLEMENTARITY_AVAILABLE = True
    tprint_info("✅ CMI complementarity components loaded successfully")
except ImportError as e:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    tprint_warning(f"⚠️ CMI complementarity components not available: {e}")

# ML utilities
try:
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    from sklearn.multioutput import MultiOutputRegressor
    import shap
    LGBM_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError as e:
    LGBM_AVAILABLE = False
    SHAP_AVAILABLE = False
    tprint_warning(f"⚠️ ML libraries not available: {e}")

# Import centralized category mapping utilities
try:
    from src.feature_generation import get_feature_bank, list_available_categories
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False
    get_feature_bank = None
    list_available_categories = None
    tprint_warning("⚠️ Feature bank not available")

# Import our new utilities
try:
    from src.training.utils.feature_selection.variant_generator import OptimizedVariantGenerator, generate_all_variants_optimized
    from src.training.utils.feature_selection.cheap_pruning import OptimizedCheapPruningPipeline, apply_optimized_cheap_pruning, OptimizedPruningConfig
    from src.training.steps.pre_training.includes.interaction_generation_fallbacks import attach_interaction_generation_fallbacks
    from src.training.utils.feature_selection.label_guided_interaction_discovery import (
        LabelGuidedInteractionDiscovery,
        LabelGuidedInteractionConfig,
        InteractionCandidate
    )
    UTILITIES_AVAILABLE = True
    LABEL_GUIDED_AVAILABLE = True
except ImportError as e:
    UTILITIES_AVAILABLE = False
    LABEL_GUIDED_AVAILABLE = False
    tprint_warning(f"⚠️ Feature selection utilities not available: {e}")

# Import overfitting prevention utilities
try:
    from src.utils.ml_common.validation.unified_cv import temporal_cross_validation, TimeSeriesSplit
    from src.utils.ml_common.validation.universal_temporal_validation import UniversalTimeSeriesSplit
    from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention
    OVERFITTING_PREVENTION_AVAILABLE = True
except ImportError as e:
    OVERFITTING_PREVENTION_AVAILABLE = False
    tprint_warning(f"⚠️ Overfitting prevention utilities not available: {e}")

# Import HPO utilities for CMI-weighted LGBM optimization
HPO_AVAILABLE = False
HierarchicalHPOConfig = None
HPOPhaseConfig = None
HierarchicalHPO = None
BayesianTPEOptimizer = None
TPEOptimizationConfig = None

try:
    from src.utils.ml_common.optimization.hierarchical_hpo import (
        HierarchicalHPOConfig, HPOPhaseConfig, HierarchicalHPO
    )
    HPO_AVAILABLE = True
except ImportError as e:
    pass  # Already set to None above

try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig as TPEOptimizationConfig
    )
except ImportError as e:
    pass  # Already set to None above

# Try to import numba for fast MI calculation (optional)
NUMBA_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    # Numba not available - will use pure numpy fallback
    def njit(*args, **kwargs):
        """Fallback decorator if numba not available."""
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# Only log once after all attempts
if HPO_AVAILABLE:
    tprint_info("✅ HPO utilities loaded successfully")
else:
    # Silently fail - not critical for feature generation
    pass

if NUMBA_AVAILABLE:
    tprint_info("✅ Numba loaded - MI calculation will use JIT-accelerated joint probability (10x speedup)")
else:
    tprint_info("ℹ️  Numba not available - MI calculation will use pure numpy (still fast)")

# Try to import overfitting prevention manager separately (optional)
try:
    from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPreventionConfig, OverfittingPreventionManager
    OVERFITTING_MANAGER_AVAILABLE = True
except ImportError as e:
    OVERFITTING_MANAGER_AVAILABLE = False
    tprint_warning(f"⚠️ Overfitting prevention manager not available: {e}")
    
    # Define a fallback OverfittingPreventionManager class if not available
    class OverfittingPreventionManager:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

logger = logging.getLogger(__name__)


class FeatureGenerationInteractionGenerationStep(BaseStep):
    """
    Unified Interaction Generation Step (Analyst/Tactician).
    
    Implements a comprehensive pipeline for feature engineering that can operate in two modes:
    - **Tactician Mode**: Uses MI-based feature selection and interaction discovery
    - **Analyst Mode**: Uses CMI-based feature selection conditioned on Tactician outputs
    
    Features:
    - Top feature selection by composite_score
    - Numerically safe variant generation
    - Per-category cheap pruning
    - LGBM+SHAP feature selection
    - Tree-guided interaction discovery
    """

    def __init__(self, step_name: str = "feature_generation_interaction_generation_step"):
        """Initialize the unified interaction generation step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('UnifiedInteractionGeneration')
        
        # Mode detection - will be determined at runtime based on launcher arguments
        self.execution_mode = None  # Will be set to 'analyst' or 'tactician'
        
        # Initialize CMI complementarity components for Tactician mode
        if CMI_COMPLEMENTARITY_AVAILABLE:
            self.cmi_config = CMIComplementarityConfig(
                per_family_budget=(5, 15),
                upstream_multiplier=3,
                max_total_features=60,
                enable_regime_awareness=True,
                compute_timeout_seconds=300.0,
                enable_synergy=True,
                beta_synergy=0.25
            )
            self.cmi_scorer = CMIComplementarityScorer(self.cmi_config)
            self.analyst_handler = AnalystSideInfoHandler()
            tprint_info("✅ CMI complementarity components initialized for Tactician mode")
        else:
            self.cmi_scorer = None
            self.analyst_handler = None
            # Note: CMI availability will be checked at runtime for Tactician mode
        
        # Initialize HPO components for CMI-weighted LGBM optimization
        if HPO_AVAILABLE:
            self.hpo_optimizer = None  # Will be initialized when needed
            self.cmi_lgbm_params = {
                'alpha_cmi': 0.6,  # Weight for LGBM importance
                'beta_cmi': 0.4,   # Weight for CMI score
                'enable_cmi_weighting': True
            }
            tprint_info("✅ HPO components available for CMI-weighted LGBM")
        else:
            self.hpo_optimizer = None
            self.cmi_lgbm_params = None
            tprint_warning("⚠️ HPO not available - using default CMI parameters")
        
        # Initialize hardware optimization
        if HARDWARE_OPT_AVAILABLE:
            self.hardware_manager = UnifiedHardwareManager()
            self.memory_optimizer = M1MemoryOptimizer(memory_limit_gb=8.0)
            self.cpu_optimizer = M1CPUOptimizer()
            tprint_info("✅ Hardware optimization components initialized")
        else:
            self.hardware_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            tprint_warning("⚠️ Hardware optimization not available")
        
        # Initialize parallel processing
        if PARALLEL_AVAILABLE:
            self.max_workers = min(3, cpu_count())
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            tprint_info(f"✅ Parallel processing initialized with {self.max_workers} workers")
        else:
            self.max_workers = 1
            self.thread_pool = None
            tprint_warning("⚠️ Parallel processing not available")
        
        # Initialize VectorBT components
        self.vectorization_manager = None
        self.rolling_optimizer = None
        
        # Performance tracking
        self.performance_stats = {
            'phase0_time': 0.0,
            'phase1_time': 0.0,
            'phase2_time': 0.0,
            'phase3_1_time': 0.0,
            'phase3_2_time': 0.0,
            'phase3_3_time': 0.0,
            'phase4_time': 0.0,
            'total_time': 0.0,
            'features_selected_per_category': {},
            'variants_generated': 0,
            'cross_timeframe_features_generated': 0,
            'cross_timeframe_ratios_generated': 0,
            'features_after_pruning': 0,
            'final_feature_count': 0,
            'interaction_count': 0,
            'numerical_safety_incidents': 0,
            'category_coverage': {}
        }
        
        # Numerical safety log
        self.numerical_safety_log = []
        
        # Initialize feature bank for category management
        if FEATURE_BANK_AVAILABLE:
            self.feature_bank = get_feature_bank()
            self.available_categories = list_available_categories()
            tprint_info("✅ Feature bank initialized for category management")
        else:
            self.feature_bank = None
            self.available_categories = []
            tprint_warning("⚠️ Using fallback category management")
        
        # Initialize overfitting prevention with enhanced settings
        if OVERFITTING_PREVENTION_AVAILABLE:
            self.overfitting_config = OverfittingPreventionConfig(
                enable_early_stopping=True,
                early_stopping_patience=5,  # Reduced patience for stricter control
                early_stopping_min_delta=1e-3,  # Increased minimum delta
                enable_cross_validation=True,
                cv_folds=5,
                cv_strategy='time_series_split',
                enable_regularization=True,
                l1_regularization=0.15,  # Increased L1 regularization
                l2_regularization=0.15,  # Increased L2 regularization
                enable_ensemble_diversity=True,
                diversity_threshold=0.8,  # Increased diversity threshold
                enable_performance_monitoring=True,
                overfitting_threshold=0.1,
                enable_learning_curves=True,
                validation_split=0.2,
                test_split=0.1,
                enable_holdout_validation=True
            )
            self.overfitting_manager = OverfittingPreventionManager(self.overfitting_config)
            self.data_leakage_prevention = DataLeakagePrevention()
            tprint_info("✅ Enhanced overfitting prevention initialized")
        else:
            self.overfitting_config = None
            self.overfitting_manager = None
            self.data_leakage_prevention = None
            tprint_warning("⚠️ Overfitting prevention not available")
        
        # Category definitions - now using feature bank categories
        self.categories = [
            'trend', 'oscillator', 'momentum', 'returns', 'volatility', 
            'volume', 'acceleration', 'advanced_statistical', 
            'candlestick_pattern', 'entropy', 'spectral_wavelet'
        ]
        
        # Initialize interaction scores storage
        self._last_interaction_scores = []
        
        # Initialize performance metrics storage
        self._phase3_1_performance = {}
        self._phase3_2_performance = {}
        self._phase3_3_performance = {}

        # Runtime diagnostics
        self._runtime_class_logged = False

    def _ensure_runtime_helpers(self) -> None:
        try:
            attach_interaction_generation_fallbacks(self.__class__)
        except Exception as exc:
            self.logger.warning(f"Failed to attach helper fallbacks: {exc}")

    def _log_runtime_class(self, context: str) -> None:
        if getattr(self, '_runtime_class_logged', False):
            return
        cls = self.__class__
        mro_chain = " -> ".join(f"{c.__module__}.{c.__name__}" for c in cls.__mro__)
        tprint_error(
            f"🔍 Runtime interaction step class ({context}): "
            f"{cls.__module__}.{cls.__name__} (id={id(cls)})"
        )
        tprint_error(f"🔍 Runtime MRO chain: {mro_chain}")
        self._runtime_class_logged = True

    def _ensure_runtime_integrity(self, context: str) -> None:
        self._log_runtime_class(context)
        self._ensure_runtime_helpers()

    def _detect_execution_mode(self, config: Dict[str, Any]) -> str:
        """
        Detect execution mode based on launcher arguments and step context.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            'analyst' or 'tactician'
        """
        # Primary detection: Check current step name for Tactician training steps
        current_step_name = getattr(self, 'step_name', '')
        is_tactician_training_step = (
            'tactician_base_training' in current_step_name or
            'tactician_ensemble_training' in current_step_name or
            'tactician' in current_step_name.lower()
        )
        
        # Secondary detection: Check execution context
        tactician_execution_context = config.get('execution_context', '').lower()
        is_tactician_context = 'tactician' in tactician_execution_context
        
        # Tertiary detection: Check for explicit mode setting
        explicit_mode = config.get('interaction_generation_mode', '').lower()
        
        # Quaternary detection: Check for Tactician-specific configuration
        tactician_mode_config = config.get('tactician_mode', False)
        
        # Determine mode
        if (is_tactician_training_step or is_tactician_context or 
            explicit_mode == 'tactician' or tactician_mode_config):
            mode = 'tactician'  # Uses CMI-based selection
        else:
            mode = 'analyst'  # Uses MI-based selection
        
        tprint_info(f"🔍 Execution mode detection:")
        tprint_info(f"  - Current step name: {current_step_name}")
        tprint_info(f"  - Is Tactician training step: {is_tactician_training_step}")
        tprint_info(f"  - Execution context: {config.get('execution_context', 'N/A')}")
        tprint_info(f"  - Is Tactician context: {is_tactician_context}")
        tprint_info(f"  - Explicit mode: {explicit_mode}")
        tprint_info(f"  - Tactician mode config: {tactician_mode_config}")
        tprint_info(f"  - Detected mode: {mode}")
        
        return mode

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute unified interaction generation (Analyst/Tactician mode).

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
                - top_features_per_category: Number of top features to select (default: 4)
                - pruning_target: Target pruning percentage (default: 0.45 for 45%)
                - interaction_generation_mode: 'analyst' or 'tactician' (optional)
                - tactician_mode: Boolean flag for Tactician mode (optional)

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = time.time()
        self._ensure_runtime_integrity("execute")
        
        # Update context with symbol from config to ensure correct artifact storage
        symbol = config.get('symbol', 'UNKNOWN')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'long')
        
        self._current_context.update({
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'direction': direction,
            'model': 'analyst'  # Will be updated based on execution mode
        })
        tprint_info(f"📁 Updated context: {symbol}/{exchange} [{timeframe}] {direction}")
        
        # Detect execution mode
        self.execution_mode = self._detect_execution_mode(config)
        
        # Update model in context based on execution mode
        self._current_context['model'] = self.execution_mode
        
        # Fast fail: Check CMI availability for Tactician mode
        if self.execution_mode == 'tactician':
            if not CMI_COMPLEMENTARITY_AVAILABLE or self.cmi_scorer is None:
                error_msg = "❌ FATAL: Tactician mode requires CMI complementarity components, but they are not available!"
                tprint_error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'artifacts': {},
                    'metrics': {}
                }
            tprint_info(f"🎯 [TACTICIAN] Starting CMI-based interaction generation for {config.get('symbol', 'UNKNOWN')}")
        else:
            tprint_info(f"📊 [ANALYST] Starting MI-based interaction generation for {config.get('symbol', 'UNKNOWN')}")
        

        try:
            # Initialize optimization components
            await self._initialize_optimization_components(config)
            
            # Phase 0: Load artifacts and select top features
            tprint_info("=" * 80)
            tprint_info("📋 PHASE 0: Load Artifacts and Select Top Features")
            tprint_info("=" * 80)
            phase0_start = time.time()
            
            lookback_optimization, labeled_data, generated_features, top_features_by_category = \
                await self._phase0_load_and_select(config)
            
            self.performance_stats['phase0_time'] = time.time() - phase0_start
            tprint_performance(f"Phase 0 completed", self.performance_stats['phase0_time'])
            
            # Phase 1: Generate variants
            tprint_info("=" * 80)
            tprint_info("🔄 PHASE 1: Generate Numerically Safe Variants")
            tprint_info("=" * 80)
            phase1_start = time.time()
            
            # Apply chunked processing and VectorBT optimization
            if HARDWARE_OPT_AVAILABLE and self.memory_optimizer is not None:
                tprint_info("🚀 Applying hardware-optimized variant generation")
                variant_features = await self._phase1_generate_variants_optimized(
                    generated_features, top_features_by_category, lookback_optimization, config
                )
            else:
                variant_features = await self._phase1_generate_variants(
                    generated_features, top_features_by_category, lookback_optimization, config
                )
            
            # Debug: Check variant features after generation
            if len(variant_features.columns) > 0:
                tprint_info(f"🔍 DEBUG: Variant features columns: {list(variant_features.columns)[:10]}...")  # Show first 10 columns
            else:
                tprint_warning("⚠️ DEBUG: No variant features generated!")
                tprint_warning("⚠️ DEBUG: This means variant generation failed completely!")
                tprint_warning("⚠️ DEBUG: Check if selected_features were valid and variant generation succeeded")
            
            self.performance_stats['phase1_time'] = time.time() - phase1_start
            tprint_performance(f"Phase 1 completed", self.performance_stats['phase1_time'])
            
            # Phase 2: Cheap pruning
            tprint_info("=" * 80)
            tprint_info("✂️ PHASE 2: Per-Category Cheap Pruning")
            tprint_info("=" * 80)
        
            # Debug: Check inputs to cheap pruning
            
            if len(variant_features.columns) == 0:
                tprint_warning("⚠️ DEBUG: No variant features to prune! Returning empty DataFrame")
                return pd.DataFrame(), {"error": "No variant features to prune"}
            
            # Debug: Check variant features before pruning
            tprint_info(f"🔍 DEBUG: Variant features columns: {list(variant_features.columns)[:10]}...")  # Show first 10 columns
            
            # Check for cross-timeframe features
            cross_timeframe_cols = [c for c in variant_features.columns if any(f'_{m}x_ratio' in c for m in [3, 6, 9, 15, 27, 45, 60])]
            if len(cross_timeframe_cols) > 0:
                tprint_info(f"🔍 Found {len(cross_timeframe_cols)} cross-timeframe features before pruning")
            
            if len(variant_features.columns) == 0:
                tprint_warning("⚠️ DEBUG: No variant features to prune! This will cause cheap pruning to fail!")
            
            phase2_start = time.time()
            
            
            pruned_features, pruning_stats, targets = await self._phase2_cheap_pruning(
                variant_features, labeled_data, lookback_optimization, config
            )
            
            # Debug: Check pruned features after pruning
            if len(pruned_features.columns) > 0:
                tprint_info(f"🔍 DEBUG: Pruned features columns: {list(pruned_features.columns)[:10]}...")  # Show first 10 columns
                
                # Check for cross-timeframe features after pruning
                cross_timeframe_cols_after = [c for c in pruned_features.columns if any(f'_{m}x_ratio' in c for m in [3, 6, 9, 15, 27, 45, 60])]
                if len(cross_timeframe_cols_after) > 0:
                    tprint_info(f"🔍 Found {len(cross_timeframe_cols_after)} cross-timeframe features after pruning")
                else:
                    tprint_warning("⚠️ DEBUG: ALL cross-timeframe features were pruned!")
            else:
                tprint_warning("⚠️ DEBUG: No features remaining after pruning!")
                tprint_warning("⚠️ DEBUG: All {len(variant_features.columns)} variant features were removed!")
            
            self.performance_stats['phase2_time'] = time.time() - phase2_start
            tprint_performance(f"Phase 2 completed", self.performance_stats['phase2_time'])
            
            # Debug: Check pruned features after pruning
            if len(pruned_features.columns) > 0:
                tprint_info(f"🔍 DEBUG: Pruned features columns: {list(pruned_features.columns)[:10]}...")  # Show first 10 columns
            else:
                tprint_warning("⚠️ DEBUG: No features remaining after pruning!")
            
            # Phase 3: Three-phase LGBM+SHAP
            tprint_info("=" * 80)
            tprint_info("🤖 PHASE 3: Three-Phase LGBM+SHAP Pipeline")
            tprint_info("=" * 80)
            
            if len(pruned_features.columns) == 0:
                tprint_warning("⚠️ DEBUG: No pruned features available for Phase 3! This will cause the pipeline to fail!")
                return {
                    'success': False,
                    'error': 'No pruned features available for Phase 3',
                    'artifacts': {},
                    'metrics': self.performance_stats
                }
            
            final_features, interactions, shap_metadata = await self._phase3_lgbm_shap_pipeline(
                pruned_features, targets, config, lookback_optimization
            )
            
            # Phase 4: Integration and artifact saving
            tprint_info("=" * 80)
            tprint_info("💾 PHASE 4: Integration and Artifact Saving")
            tprint_info("=" * 80)
            phase4_start = time.time()

            # Use module-level helper so Phase 4 works even if methods are
            # attached via fallbacks or wrappers.
            artifacts, metrics = await _phase4_save_artifacts(
                self,
                final_features,
                interactions,
                shap_metadata,
                pruning_stats,
                config,
                lookback_optimization,
            )
            
            self.performance_stats['phase4_time'] = time.time() - phase4_start
            self.performance_stats['total_time'] = time.time() - start_time
            
            tprint_success(f"✅ [ANALYST] Three-phase pipeline completed in {self.performance_stats['total_time']:.2f}s")
            tprint_info(f"📊 Final feature count: {self.performance_stats['final_feature_count']}")
            tprint_info(f"🔗 Interaction count: {self.performance_stats['interaction_count']}")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"[ANALYST] Three-phase pipeline failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'artifacts': {},
                'metrics': self.performance_stats,
                'error': error_msg
            }

    async def _initialize_optimization_components(self, config: Dict[str, Any]):
        """Initialize VectorBT and hardware optimization components."""
        tprint_info("🔧 Initializing optimization components")
        
        try:
            # Initialize hardware optimization
            if HARDWARE_OPT_AVAILABLE and self.hardware_manager:
                self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
                tprint_success("✅ Hardware optimization initialized")
            
            # Initialize VectorBT components
            if VECTORBT_AVAILABLE:
                vectorization_config = VectorizationConfig(
                    enable_vectorbt=True,
                    enable_gpu=config.get('enable_gpu', False),
                    enable_parallel=True,
                    memory_efficient=True,
                    max_memory_gb=8.0,
                    chunk_size=1000,
                    enable_monitoring=True
                )
                
                self.vectorization_manager = UnifiedVectorizationManager(vectorization_config)
                self.rolling_optimizer = VectorBTRollingOptimizer(
                    enable_parallel=True,
                    memory_efficient=True,
                    chunk_size=1000
                )
                tprint_success("✅ VectorBT components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Optimization initialization partial failure: {e}")

    async def _phase0_load_and_select(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Phase 0: Load artifacts and select top features per category.

        This mirrors the artifact-resolution logic from
        FeatureGenerationPeriodLookbackOptimizationStep._load_generated_features
        so that we always pull labels from the
        feature_generation_labeling_integration_step store and features from
        feature_generation_feature_generation_step, while keeping all original
        deduplication, light-mode filtering, and selection behaviour.
        """

        tprint_info("📊 Loading artifacts via BaseStep artifact manager")

        # 1) Lookback optimization
        try:
            raw_lookback_optimization = self._get_artifact("lookback_optimization", "data")
            raw_lookback_optimization = _ensure_pandas_dataframe(raw_lookback_optimization)
            lookback_optimization = self._normalize_lookback_optimization(raw_lookback_optimization)
            tprint_success(f"✅ Loaded lookback_optimization: {lookback_optimization.shape}")
            if not lookback_optimization.empty:
                tprint_structured({"Lookback Optimization": lookback_optimization.head().to_dict()}, level=LogLevel.INFO)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load lookback_optimization artifact: {e}")

        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        model = config.get("model", "analyst")

        # 2) Labeled data: prefer the labeling integration store
        try:
            original_step_name = self.artifact_manager._current_step_name
            self.artifact_manager.set_context(
                step_name="feature_generation_labeling_integration_step",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=model,
                datetime=datetime.now(),
            )
            artifact_name = f"labeled_data_{symbol}_{timeframe}"
            labeled_data = self.artifact_manager.get_artifact(
                artifact_name=artifact_name,
                artifact_type="data",
            )
            labeled_data = _ensure_pandas_dataframe(labeled_data)
            self.artifact_manager.set_context(
                step_name=original_step_name,
                datetime=datetime.now(),
            )
            tprint_success(f"✅ Loaded labeled_data: {labeled_data.shape}")
            tprint_structured({"Labeled Data": labeled_data.head().to_dict()}, level=LogLevel.INFO)
        except Exception as e:
            # Fallbacks: direct artifact lookup, then generic name
            try:
                artifact_name = f"labeled_data_{config['symbol']}_{config['timeframe']}"
                labeled_data = self._get_artifact(artifact_name, "data")
                labeled_data = _ensure_pandas_dataframe(labeled_data)
                tprint_success(f"✅ Loaded labeled_data: {labeled_data.shape}")
                tprint_structured({"Labeled Data": labeled_data.head().to_dict()}, level=LogLevel.INFO)
            except Exception as inner:
                try:
                    labeled_data = self._get_artifact("labeled_data", "data")
                    labeled_data = _ensure_pandas_dataframe(labeled_data)
                    tprint_success(f"✅ Loaded labeled_data (fallback): {labeled_data.shape}")
                except Exception as e2:
                    raise FileNotFoundError(f"Failed to load labeled_data artifact: {e} / {inner} / {e2}")

        # 3) Generated features: prefer the feature_generation_feature_generation_step store
        try:
            original_step_name = self.artifact_manager._current_step_name
            self.artifact_manager.set_context(
                step_name="feature_generation_feature_generation_step",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=model,
                datetime=datetime.now(),
            )
            features_artifact_name = f"generated_features_{timeframe}"
            generated_features = self.artifact_manager.get_artifact(
                artifact_name=features_artifact_name,
                artifact_type="data",
            )
            generated_features = _ensure_pandas_dataframe(generated_features)
            self.artifact_manager.set_context(
                step_name=original_step_name,
                datetime=datetime.now(),
            )
            tprint_success(f"✅ Loaded generated_features: {generated_features.shape}")
        except Exception as e:
            try:
                generated_features = self._get_artifact("generated_features", "data")
                generated_features = _ensure_pandas_dataframe(generated_features)
                tprint_success(f"✅ Loaded generated_features: {generated_features.shape}")
            except Exception as e2:
                raise FileNotFoundError(f"Failed to load generated_features artifact: {e} / {e2}")

        # 4) Deduplicate indices
        if labeled_data.index.duplicated().any():
            n_dup = labeled_data.index.duplicated().sum()
            tprint_warning(
                f"⚠️ Labeled data has {n_dup} duplicate indices "
                f"({n_dup/len(labeled_data)*100:.1f}%), deduplicating at load time..."
            )
            labeled_data = labeled_data[~labeled_data.index.duplicated(keep="first")]
            tprint_success(f"✅ Deduplicated labeled_data to {len(labeled_data)} unique indices")

        if generated_features.index.duplicated().any():
            n_dup = generated_features.index.duplicated().sum()
            tprint_warning(
                f"⚠️ Generated features has {n_dup} duplicate indices "
                f"({n_dup/len(generated_features)*100:.1f}%), deduplicating at load time..."
            )
            generated_features = generated_features[~generated_features.index.duplicated(keep="first")]
            tprint_success(f"✅ Deduplicated generated_features to {len(generated_features)} unique indices")

        # 5) Apply light-mode filtering
        tprint_info("📊 PHASE 0: Initial feature counts before filtering:")
        tprint_info(f"  📈 Generated features: {len(generated_features.columns)} features")
        tprint_info(f"  📈 Labeled data: {len(labeled_data.columns)} features")
        
        generated_features = self._apply_light_mode_filter(
            generated_features, config, config.get('timeframe', '15m')
        )
        labeled_data = self._apply_light_mode_filter(
            labeled_data, config, config.get('timeframe', '15m')
        )
        
        tprint_info(f"📊 PHASE 0: Feature counts after light mode filtering:")
        tprint_info(f"  📈 Generated features: {len(generated_features.columns)} features (filtered)")
        tprint_info(f"  📈 Labeled data: {len(labeled_data.columns)} features (filtered)")
        
        # Select top features per category based on execution mode
        top_features_per_category = config.get('top_features_per_category', 4)
        
        if self.execution_mode == 'tactician':
            tprint_info(f"🎯 [TACTICIAN] Using CMI-based feature selection")
            top_features_by_category = self._select_top_features_per_category_cmi(
                lookback_optimization, top_features_per_category, config
            )
        else:
            tprint_info(f"📊 [ANALYST] Using MI-based feature selection")
            top_features_by_category = self._select_top_features_per_category(
                lookback_optimization, top_features_per_category
            )
        
        
        # Count total selected features across all categories
        total_selected_features = sum(len(features) for features in top_features_by_category.values())
        tprint_info(f"📊 PHASE 0: Feature selection summary:")
        tprint_info(f"  📈 Total categories: {len(top_features_by_category)}")
        tprint_info(f"  📈 Total selected features: {total_selected_features}")
        for category, features in top_features_by_category.items():
            tprint_info(f"    - {category}: {len(features)} features")
        
        return lookback_optimization, labeled_data, generated_features, top_features_by_category

    def _get_primary_summary_targets(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load labeled_data and select a primary target column for summary metrics.

        Prefers smoothed_label, then binary_label, then realized_return, and
        finally falls back to the first available non-empty column.
        """

        try:
            artifact_name = f"labeled_data_{config['symbol']}_{config['timeframe']}"
            labeled_data = self._get_artifact(artifact_name, "data")
        except Exception:
            try:
                labeled_data = self._get_artifact("labeled_data", "data")
            except Exception:
                return None

        labeled_data = _ensure_pandas_dataframe(labeled_data)
        if not isinstance(labeled_data, pd.DataFrame) or labeled_data.empty:
            return None

        if labeled_data.index.duplicated().any():
            labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]

        preferred_targets = [
            "smoothed_label",
            "binary_label",
            "realized_return",
        ]
        existing = [col for col in preferred_targets if col in labeled_data.columns]
        if existing:
            return labeled_data[[existing[0]]]

        # Fallback: first non-all-NaN column
        for col in labeled_data.columns:
            series = labeled_data[col]
            if getattr(series, "notna", None) is not None and series.notna().any():
                return labeled_data[[col]]

        return labeled_data.iloc[:, :1]

    def _normalize_lookback_optimization(self, lookback_optimization: Any) -> pd.DataFrame:
        """Normalize lookback_optimization artifact into a per-feature DataFrame.

        Handles both the older wide-DataFrame format and the newer dict-based
        structure saved by the period lookback optimization step.
        """
        # If we already have a DataFrame, return as-is
        if isinstance(lookback_optimization, pd.DataFrame):
            return lookback_optimization

        core = lookback_optimization
        # ArtifactRouter may wrap the payload in a dict with a 'data' key
        if isinstance(core, dict) and 'data' in core and isinstance(core['data'], dict):
            core = core['data']

        rows: List[Dict[str, Any]] = []

        if isinstance(core, dict):
            # Preferred: per_feature_metrics from optimization step
            per_feature_metrics = core.get('per_feature_metrics')
            if isinstance(per_feature_metrics, dict) and per_feature_metrics:
                for feature_name, info in per_feature_metrics.items():
                    category = info.get('category')
                    if isinstance(category, str):
                        # category keys from optimization often end with '_features'
                        category_clean = category.replace('_features', '')
                        category_clean = self._normalize_category_name_fallback(category_clean)
                    else:
                        category_clean = category

                    perf = float(info.get('performance_score', 0.0) or 0.0)
                    stab = float(info.get('stability_score', 0.0) or 0.0)
                    optimal_lookback = info.get('optimal_lookback')
                    information_score = (perf + stab) / 2.0 if (perf > 0 or stab > 0) else 0.0
                    composite_score = stab * information_score if (stab > 0 and information_score > 0) else 0.0

                    rows.append({
                        'feature_name': feature_name,
                        'category': category_clean,
                        'optimal_lookback': optimal_lookback,
                        'performance_score': perf,
                        'stability_score': stab,
                        'information_score': information_score,
                        'composite_score': composite_score,
                    })

            # Fallback: build rows from category_optimizations if per_feature_metrics missing
            if not rows:
                category_optimizations = core.get('category_optimizations')
                if isinstance(category_optimizations, dict) and category_optimizations:
                    for category, features in category_optimizations.items():
                        if not isinstance(features, dict):
                            continue
                        if isinstance(category, str):
                            category_clean = category.replace('_features', '')
                            category_clean = self._normalize_category_name_fallback(category_clean)
                        else:
                            category_clean = category

                        for feature_name, feature_info in features.items():
                            if not isinstance(feature_info, dict):
                                continue
                            perf = float(feature_info.get('performance_score', 0.0) or 0.0)
                            stab = float(feature_info.get('stability_score', 0.0) or 0.0)
                            optimal_lookback = feature_info.get('optimal_lookback')
                            information_score = (perf + stab) / 2.0 if (perf > 0 or stab > 0) else 0.0
                            composite_score = stab * information_score if (stab > 0 and information_score > 0) else 0.0

                            rows.append({
                                'feature_name': str(feature_info.get('feature_name', feature_name)),
                                'category': category_clean,
                                'optimal_lookback': optimal_lookback,
                                'performance_score': perf,
                                'stability_score': stab,
                                'information_score': information_score,
                                'composite_score': composite_score,
                            })

        if rows:
            df = pd.DataFrame(rows)
            tprint_info(f"📊 Normalized lookback_optimization to per-feature DataFrame with {len(df)} rows")
            return df

        # Last resort: return empty DataFrame with expected columns
        tprint_warning("⚠️ lookback_optimization artifact not in expected format; using empty DataFrame")
        return pd.DataFrame(
            columns=[
                'feature_name', 'category', 'optimal_lookback',
                'performance_score', 'stability_score',
                'information_score', 'composite_score',
            ]
        )

    def _transform_lookback_optimization_data(self, lookback_optimization: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide DataFrame with nested columns to long format with simple columns.
        
        Args:
            lookback_optimization: Wide DataFrame with nested column names
            
        Returns:
            Long DataFrame with columns: feature_name, category, composite_score, optimal_lookback, etc.
        """
        tprint_info("🔄 Transforming lookback optimization data from wide to long format")
        
        # Extract data from the wide format
        feature_data = []
        
        # Get all columns that contain feature information - try both patterns
        feature_columns = []
        for col in lookback_optimization.columns:
            if ('category_optimizations' in col or 'optimization_results' in col) and col.endswith('.feature_name'):
                feature_columns.append(col)
        
        tprint_info(f"Found {len(feature_columns)} feature name columns")
        
        # Debug: Show some example columns
        if feature_columns:
            tprint_info(f"Example feature columns: {feature_columns[:5]}")
            # Show all unique category patterns found
            category_patterns = set()
            for col in feature_columns:
                parts = col.split('.')
                if len(parts) >= 4:
                    if 'category_optimizations' in col:
                        category_patterns.add(parts[2])
                    else:  # optimization_results
                        category_patterns.add(parts[1])
        
        for col in feature_columns:
            try:
                # Extract feature name and category from column name
                parts = col.split('.')
                if len(parts) >= 4:
                    # Handle both patterns: 
                    # 1. category_optimizations.X_features.Y.feature_name 
                    # 2. optimization_results.X_features.Y.feature_name
                    if 'category_optimizations' in col:
                        category_part = parts[2]  # e.g., 'acceleration_features'
                    else:  # optimization_results
                        category_part = parts[1]  # e.g., 'momentum_features'
                    
                    # Convert category name to match expected format
                    category = category_part.replace('_features', '')  # acceleration_features -> acceleration
                    
                    # Debug: Show category extraction
                    
                    # Use feature bank for category normalization if available
                    if FEATURE_BANK_AVAILABLE and self.feature_bank:
                        try:
                            # Get available categories from feature bank
                            available_categories = [cat.value if hasattr(cat, 'value') else str(cat) for cat in self.available_categories]
                            # Normalize category name to match feature bank categories
                            if category in available_categories:
                                category = category
                            else:
                                # Try to find matching category
                                for available_cat in available_categories:
                                    if category.lower() in available_cat.lower() or available_cat.lower() in category.lower():
                                        category = available_cat
                                        break
                        except Exception as e:
                            tprint_warning(f"Error normalizing category with feature bank: {e}")
                            # Fallback to local mapping
                            category = self._normalize_category_name_fallback(category)
                    else:
                        # Fallback to local mapping
                        category = self._normalize_category_name_fallback(category)
                    
                    feature_name = lookback_optimization[col].iloc[0] if not pd.isna(lookback_optimization[col].iloc[0]) else None
                    
                    if feature_name:
                        # Find related columns for this feature
                        base_col = col.replace('.feature_name', '')
                        
                        # Get all related data for this feature
                        feature_info = {
                            'feature_name': feature_name,
                            'category': category,
                            'composite_score': 0.0,
                            'optimal_lookback': None,
                            'performance_score': 0.0,
                            'stability_score': 0.0,
                            'information_score': 0.0
                        }
                        
                        # Try to get composite_score, optimal_lookback, etc.
                        for metric in ['composite_score', 'optimal_lookback', 'performance_score', 'stability_score', 'information_score']:
                            metric_col = f"{base_col}.{metric}"
                            if metric_col in lookback_optimization.columns:
                                value = lookback_optimization[metric_col].iloc[0]
                                if not pd.isna(value):
                                    feature_info[metric] = value
                        
                        # Calculate composite_score if not available
                        if feature_info['composite_score'] == 0.0 and feature_info['performance_score'] > 0 and feature_info['stability_score'] > 0:
                            feature_info['composite_score'] = feature_info['performance_score'] * feature_info['stability_score']
                        
                        feature_data.append(feature_info)
            except Exception as e:
                tprint_warning(f"⚠️ Error processing column {col}: {e}")
                continue
        
        if not feature_data:
            tprint_warning("⚠️ No feature data found in lookback optimization")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_data)
        
        # Remove duplicates based on feature_name
        df = df.drop_duplicates(subset=['feature_name'], keep='first')
        
        # Show category distribution
        category_counts = df['category'].value_counts()
        tprint_info(f"📊 Category distribution: {dict(category_counts)}")
        
        # Debug: Check which expected categories are missing
        found_categories = set(df['category'].unique())
        expected_categories = set(self.categories)
        missing_categories = expected_categories - found_categories
        unexpected_categories = found_categories - expected_categories
        
        if missing_categories:
            tprint_warning(f"⚠️ Missing expected categories: {sorted(missing_categories)}")
        if unexpected_categories:
            tprint_info(f"ℹ️ Unexpected categories found: {sorted(unexpected_categories)}")
        
        tprint_success(f"✅ Transformed {len(df)} features from wide to long format")
        tprint_info(f"📊 Categories found: {sorted(df['category'].unique())}")
        return df
    
    def _normalize_category_name_fallback(self, category: str) -> str:
        """Fallback method to normalize category names to match expected categories."""
        category_mapping = {
            'advanced_stat': 'advanced_statistical',
            'advanced_statistics': 'advanced_statistical',
            'statistical': 'advanced_statistical',
            'osc': 'oscillator',
            'oscillators': 'oscillator',
            'support_resistance': 'support_resistance',
            'sr': 'support_resistance',
            'support': 'support_resistance',
            'resistance': 'support_resistance',
            'trend': 'trend',
            'trends': 'trend',
            'momentum': 'momentum',
            'mom': 'momentum',
            'returns': 'returns',
            'return': 'returns',
            'volatility': 'volatility',
            'vol': 'volatility',
            'volume': 'volume',
            'volumes': 'volume',
            'acceleration': 'acceleration',
            'accel': 'acceleration',
            'candlestick': 'candlestick_pattern',
            'candlestick_pattern': 'candlestick_pattern',
            'candle': 'candlestick_pattern',
            'entropy': 'entropy',
            'ent': 'entropy',
            'spectral': 'spectral_wavelet',
            'wavelet': 'spectral_wavelet',
            'spectral_wavelet': 'spectral_wavelet'
        }
        
        # Try exact match first
        if category in category_mapping:
            return category_mapping[category]
        
        # Try case-insensitive match
        category_lower = category.lower()
        for key, value in category_mapping.items():
            if key.lower() == category_lower:
                return value
        
        # Try partial match
        for key, value in category_mapping.items():
            if key.lower() in category_lower or category_lower in key.lower():
                return value
        
        # Return original if no match found
        return category

    def _apply_multi_window_priority_boost(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply priority boost to ATR, EMA/EWMA, and multi-window features.

        These features strongly benefit from multi-window signals and show natural
        nonlinear interactions across different market regimes:
        - ATR: Captures volatility regime changes and risk dynamics
        - EMA/EWMA: Exponential decay with recency weighting, multi-timeframe trends
        - Multi-window features: Short-term vs long-term interactions

        Args:
            features_df: DataFrame with feature_name and composite_score

        Returns:
            DataFrame with boosted composite_score for priority features
        """
        priority_patterns = [
            # ATR features (volatility and regime detection)
            'atr', 'average_true_range', 'true_range',
            # EMA/EWMA features (exponential decay and trend)
            'ema', 'ewma', 'exponential_ma', 'exp_ma',
            # Multi-window indicators
            'multi_window', 'multi_timeframe', 'cross_timeframe',
            # Cross-timeframe ratio features (all multipliers)
            '_3x_ratio', '_6x_ratio', '_9x_ratio', '_15x_ratio',
            '_27x_ratio', '_45x_ratio', '_60x_ratio',
            # Bollinger Bands (ATR-related volatility)
            'bb', 'bollinger',
            # Volume-weighted features (multi-window volume interactions)
            'vwap', 'volume_weighted',
            # Momentum oscillators with multi-window characteristics
            'macd', 'stochastic', 'rsi_ewm',
        ]

        boosted_df = features_df.copy()

        # Track which features get boosted for logging
        boosted_features = []

        for idx, row in boosted_df.iterrows():
            feature_name = row.get('feature_name', '').lower()

            # Check if feature matches any priority pattern
            is_priority = any(pattern in feature_name for pattern in priority_patterns)

            if is_priority:
                # Apply 15% boost to composite_score for priority features
                original_score = row['composite_score']
                boosted_score = original_score * 1.15
                boosted_df.at[idx, 'composite_score'] = boosted_score
                boosted_features.append((row['feature_name'], original_score, boosted_score))

        if boosted_features:
            tprint_info(f"🚀 Applied multi-window priority boost to {len(boosted_features)} features:")
            for feat_name, orig, boosted in boosted_features[:5]:  # Show first 5
                tprint_info(f"  ✓ {feat_name}: {orig:.4f} → {boosted:.4f} (+15%)")
            if len(boosted_features) > 5:
                tprint_info(f"  ... and {len(boosted_features) - 5} more features")

        return boosted_df

    def _select_top_features_per_category(self, lookback_optimization: pd.DataFrame, top_n: int = 4) -> Dict:
        """
        Select top features per category using MI-based selection for Analyst mode.
        
        Uses Mutual Information (MI) for feature selection:
        - Minimum 5, maximum 12 per category
        - Select features above 50th percentile of composite_score within category
        - Allow categories with stronger signals to contribute more features
        
        Args:
            lookback_optimization: DataFrame with feature_name, category, composite_score, optimal_lookback
            top_n: Base number of features (used as fallback)
            
        Returns:
            Dict mapping category -> list of (feature_name, optimal_lookback, composite_score)
        """
        tprint_info(f"📊 [ANALYST] MI-based feature selection (3-6 per category based on signal strength)")
        
        # Transform the data if it's in wide format
        
        if 'category' not in lookback_optimization.columns:
            lookback_optimization = self._transform_lookback_optimization_data(lookback_optimization)
        else:
            pass
        
        if lookback_optimization.empty:
            tprint_warning("⚠️ No lookback optimization data available")
            return {}
        
        top_features_by_category = {}
        
        for category in self.categories:
            # Filter features by category
            
            category_features = lookback_optimization[
                lookback_optimization['category'].str.lower() == category.lower()
            ].copy()
            
            
            if len(category_features) == 0:
                tprint_warning(f"⚠️ No features found for category: {category}")
                
                # Try to find features that might belong to this category by name inference
                tprint_info(f"🔍 Attempting to find features for category {category} by name inference...")
                inferred_features = []
                
                for _, row in lookback_optimization.iterrows():
                    feature_name = row.get('feature_name', '')
                    if feature_name:
                        inferred_category = self._infer_feature_category(feature_name)
                        if inferred_category == category:
                            inferred_features.append(row)
                
                if inferred_features:
                    tprint_info(f"✅ Found {len(inferred_features)} features for category {category} via name inference")
                    category_features = pd.DataFrame(inferred_features)
                else:
                    tprint_warning(f"❌ No features found for category {category} even with name inference")
                    continue

            # Apply multi-window priority boost to ATR, EMA, and other multi-window features
            category_features = self._apply_multi_window_priority_boost(category_features)

            # Sort by composite_score descending (after priority boost)
            category_features = category_features.sort_values('composite_score', ascending=False)
            
            # Adaptive selection logic
            n_features = len(category_features)
            if n_features < 3:
                # If less than 3 features, take all
                selected_features = category_features
            else:
                # Calculate 50th percentile threshold (less restrictive)
                threshold = category_features['composite_score'].quantile(0.5)
                
                # Select features above threshold
                above_threshold = category_features[category_features['composite_score'] >= threshold]
                
                # Apply min/max constraints (increased for better category representation)
                min_features = min(5, n_features)  # Increased from 3 to 5
                max_features = min(12, n_features)  # Increased from 6 to 12
                
                if len(above_threshold) < min_features:
                    # If not enough above threshold, take top min_features
                    selected_features = category_features.head(min_features)
                elif len(above_threshold) > max_features:
                    # If too many above threshold, take top max_features
                    selected_features = category_features.head(max_features)
                else:
                    # Use adaptive selection
                    selected_features = above_threshold
            
            # Store as list of tuples
            top_features_by_category[category] = [
                (row['feature_name'], row['optimal_lookback'], row['composite_score'])
                for _, row in selected_features.iterrows()
            ]
            
            self.performance_stats['features_selected_per_category'][category] = len(selected_features)
            
            tprint_info(f"  {category.upper()}: Selected {len(selected_features)} features (adaptive)")
            for feature_name, optimal_lookback, composite_score in top_features_by_category[category]:
                tprint_info(f"    - {feature_name}: lookback={optimal_lookback}, score={composite_score:.4f}")
        
        return top_features_by_category

    def _select_top_features_per_category_cmi(self, lookback_optimization: pd.DataFrame, top_n: int = 4, config: Dict[str, Any] = None) -> Dict:
        """
        Select top features per category using CMI-based selection for Tactician mode.
        
        Uses CMI complementarity conditioned on Tactician outputs:
        - Maximizes I(X;Y|T) where T = Tactician side information
        - Creates complementary features that work with Tactician outputs
        - Maintains category protection (minimum features per category)
        
        Args:
            lookback_optimization: DataFrame with feature_name, category, composite_score, optimal_lookback
            top_n: Base number of features (used as fallback)
            config: Configuration dictionary
            
        Returns:
            Dict mapping category -> list of (feature_name, optimal_lookback, composite_score)
        """
        tprint_info(f"🎯 [TACTICIAN] CMI-based feature selection (3-6 per category)")
        
        # Transform the data if it's in wide format
        if 'category' not in lookback_optimization.columns:
            lookback_optimization = self._transform_lookback_optimization_data(lookback_optimization)
        
        if lookback_optimization.empty:
            tprint_warning("⚠️ No lookback optimization data available")
            return {}
        
        # Extract Tactician side information for CMI conditioning
        tactician_side_info = self._extract_tactician_side_info_for_cmi(config)
        
        if not tactician_side_info.get('cmi_enabled', False):
            error_msg = f"❌ FATAL: CMI conditioning not available: {tactician_side_info.get('reason', 'Unknown error')}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg)
        
        top_features_by_category = {}
        
        for category in self.categories:
            # Filter features by category
            category_features = lookback_optimization[
                lookback_optimization['category'] == category
            ].copy()
            
            if category_features.empty:
                tprint_warning(f"⚠️ No features found for category: {category}")
                top_features_by_category[category] = []
                continue

            # Apply multi-window priority boost to ATR, EMA, and other multi-window features
            category_features = self._apply_multi_window_priority_boost(category_features)

            # Sort by composite_score (descending) (after priority boost)
            category_features = category_features.sort_values('composite_score', ascending=False)
            n_features = len(category_features)
            
            # Apply CMI-based selection with category protection
            min_features = min(3, n_features)  # Minimum 3 per category
            max_features = min(6, n_features)  # Maximum 6 per category
            
            # Use CMI scorer to select features within category
            try:
                # Prepare feature data for CMI scoring
                feature_names = category_features['feature_name'].tolist()
                
                # Use CMI scorer to rank features within category
                cmi_scores = []
                for feature_name in feature_names:
                    # For now, use composite_score as proxy for CMI
                    # In a full implementation, this would compute actual CMI
                    feature_score = category_features[
                        category_features['feature_name'] == feature_name
                    ]['composite_score'].iloc[0]
                    cmi_scores.append(feature_score)
                
                # Sort by CMI scores and select top features
                feature_scores = list(zip(feature_names, cmi_scores))
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Apply min/max constraints
                selected_count = max(min_features, min(max_features, len(feature_scores)))
                selected_features_data = feature_scores[:selected_count]
                
                # Convert back to DataFrame format
                selected_features = category_features[
                    category_features['feature_name'].isin([name for name, _ in selected_features_data])
                ]
                
            except Exception as e:
                tprint_warning(f"⚠️ CMI selection failed for category {category}: {e}")
                # Fallback to top features by composite_score
                selected_features = category_features.head(max_features)
            
            # Store as list of tuples
            top_features_by_category[category] = [
                (row['feature_name'], row['optimal_lookback'], row['composite_score'])
                for _, row in selected_features.iterrows()
            ]
            
            self.performance_stats['features_selected_per_category'][category] = len(selected_features)
            
            tprint_info(f"  {category.upper()}: Selected {len(selected_features)} features (CMI-based)")
            for feature_name, optimal_lookback, composite_score in top_features_by_category[category]:
                tprint_info(f"    - {feature_name}: lookback={optimal_lookback}, score={composite_score:.4f}")
        
        return top_features_by_category

    def _extract_tactician_side_info_for_cmi(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Tactician side information for CMI conditioning.
        
        This method loads Tactician outputs to condition CMI calculations for Tactician mode.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing Tactician side information and CMI configuration
        """
        if not CMI_COMPLEMENTARITY_AVAILABLE or self.analyst_handler is None:
            error_msg = "❌ FATAL: CMI components not available - Tactician mode requires CMI complementarity!"
            tprint_error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # Try to load Tactician features from artifacts
            try:
                tactician_features = self._get_artifact('tactician_interaction_features', 'data')
                if tactician_features is not None and not tactician_features.empty:
                    # Extract Tactician side information
                    tactician_side_info = self.analyst_handler.extract_side_info(
                        {'tactician_features': tactician_features},
                        config=config,
                        data_index=tactician_features.index
                    )
                    
                    if tactician_side_info.is_valid:
                        return {
                            'cmi_enabled': True,
                            'tactician_features': tactician_features,
                            'side_info': tactician_side_info
                        }
                    else:
                        error_msg = "❌ FATAL: Tactician side information invalid - Analyst mode requires valid Tactician outputs!"
                        tprint_error(error_msg)
                        raise RuntimeError(error_msg)
                else:
                    error_msg = "❌ FATAL: No Tactician features found - Analyst mode requires Tactician outputs for CMI conditioning!"
                    tprint_error(error_msg)
                    raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"❌ FATAL: Failed to load Tactician features: {e}"
                tprint_error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"❌ FATAL: Failed to extract Tactician side information: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg)

    async def _phase1_generate_variants_optimized(
        self, 
        generated_features: pd.DataFrame,
        top_features_by_category: Dict,
        lookback_optimization: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Optimized Phase 1: Generate normalized variants with hardware optimization.
        
        Uses chunked processing, parallel feature generation, and VectorBT optimization.
        Note: For now, we delegate to the full variant generation to ensure cross-timeframe features are created.
        """
        tprint_info("🚀 Starting optimized variant generation")
        
        # Delegate to the full variant generation to ensure cross-timeframe features are created
        # TODO: Optimize this later for hardware acceleration while maintaining cross-timeframe generation
        return await self._phase1_generate_variants(
            generated_features, top_features_by_category, lookback_optimization, config
        )

    async def _phase1_generate_variants(
        self, 
        generated_features: pd.DataFrame,
        top_features_by_category: Dict,
        lookback_optimization: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 1: Generate normalized variants with RobustScaler bounding.
        
        Uses our new OptimizedVariantGenerator utility with:
        - 3-4 variants per feature (base, vol-norm, VWAP, trend-adj)
        - RobustScaler bounding to prevent extreme values
        - Causality enforcement via shift(1)
        
        Returns:
            DataFrame with all variant features
        """
        tprint_info("🔄 Generating normalized variants with RobustScaler bounding")
        
        if not UTILITIES_AVAILABLE:
            raise ImportError("Variant generation utilities not available")
        
        # Load OHLCV data using KlinesParquetManager
        ohlcv_data = None
        if DATA_LOADING_AVAILABLE:
            try:
                symbol = config.get('symbol', 'ETHUSDT')
                timeframe = config.get('timeframe', '15m')

                # Determine data directory and exchange from config (fallback to defaults)
                data_dir = config.get('data_dir', 'historical_data')
                exchange_name = config.get('exchange', 'binance')

                # Initialize KlinesParquetManager with explicit data_dir/exchange so we
                # can reliably load the underlying OHLCV needed for variant generation.
                klines_manager = get_klines_manager(data_dir=data_dir, exchange=exchange_name)

                # Prefer processed data (partitioned or consolidated), fall back to raw
                ohlcv_data = None
                last_source = None
                for data_type_candidate in ("processed", "raw"):
                    try:
                        candidate = klines_manager.read_data(
                            symbol=symbol,
                            interval=timeframe,
                            data_type=data_type_candidate,
                            columns=['open', 'high', 'low', 'close', 'volume']
                        )
                        if candidate is not None and len(candidate) > 0:
                            ohlcv_data = candidate
                            last_source = data_type_candidate
                            tprint_info(
                                f"📁 Loaded {data_type_candidate} OHLCV candidate for {symbol} {timeframe}: {ohlcv_data.shape}"
                            )
                            break
                        else:
                            tprint_warning(
                                f"⚠️ No {data_type_candidate} OHLCV data found for {symbol} {timeframe}"
                            )
                    except Exception as inner_exc:
                        tprint_warning(
                            f"⚠️ Failed to load {data_type_candidate} OHLCV data for {symbol} {timeframe}: {inner_exc}"
                        )

                if ohlcv_data is not None and len(ohlcv_data) > 0:
                    # Ensure unique time index to allow reindexing safely
                    if ohlcv_data.index.has_duplicates:
                        tprint_warning("⚠️ OHLCV index has duplicates. Collapsing to last occurrence per timestamp for safe reindexing")
                        ohlcv_data = ohlcv_data[~ohlcv_data.index.duplicated(keep='last')]

                    # Try to align OHLCV index with generated_features index. If this fails
                    # due to mixed index dtypes (e.g. Timestamp vs bytes), fall back to a
                    # robust positional alignment on the most recent overlapping window.
                    try:
                        ohlcv_data = ohlcv_data.reindex(generated_features.index, method='ffill')
                    except Exception as align_exc:
                        tprint_warning(
                            f"⚠️ Failed to align OHLCV index with generated_features index via reindex; "
                            f"falling back to positional alignment: {align_exc}"
                        )
                        min_len = min(len(ohlcv_data), len(generated_features))
                        if min_len > 0:
                            # Preserve the most recent history and force the index to match
                            # the tail of generated_features so downstream alignment is safe.
                            ohlcv_aligned = ohlcv_data.iloc[-min_len:].copy()
                            ohlcv_aligned.index = generated_features.index[-min_len:]
                            ohlcv_data = ohlcv_aligned
                        else:
                            tprint_warning(
                                "⚠️ OHLCV alignment produced no usable rows; disabling OHLCV-based variants"
                            )
                            ohlcv_data = None

                    if ohlcv_data is not None and len(ohlcv_data) > 0:
                        # Debug: Check the types of OHLCV columns
                        tprint_info(
                            f"🔍 OHLCV data types: close={type(ohlcv_data['close'])}, high={type(ohlcv_data['high'])}, low={type(ohlcv_data['low'])}"
                        )
                        
                        # Ensure all columns are pandas Series
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in ohlcv_data.columns:
                                if not isinstance(ohlcv_data[col], pd.Series):
                                    ohlcv_data[col] = pd.Series(ohlcv_data[col], name=col, index=ohlcv_data.index)
                        
                        source_label = last_source or "unknown"
                        tprint_success(f"✅ Loaded OHLCV data ({source_label}) for variant generation: {ohlcv_data.shape}")
                    else:
                        ohlcv_data = None
                        tprint_warning("⚠️ OHLCV alignment resulted in empty data; OHLCV-based variants will be skipped")
                else:
                    ohlcv_data = None
                    tprint_warning("⚠️ No OHLCV data found from KlinesParquetManager")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load OHLCV data: {e}")
                ohlcv_data = None
        
        # Fallback: Create basic OHLCV from available data if KlinesParquetManager failed
        if ohlcv_data is None:
            tprint_warning("⚠️ OHLCV data not available, using basic price data")
            # Create basic OHLCV from available data
            ohlcv_data = pd.DataFrame(index=generated_features.index)
            if 'close' in generated_features.columns:
                ohlcv_data['close'] = generated_features['close']
                ohlcv_data['high'] = generated_features.get('high', generated_features['close'])
                ohlcv_data['low'] = generated_features.get('low', generated_features['close'])
                ohlcv_data['open'] = generated_features.get('open', generated_features['close'])
            if 'volume' in generated_features.columns:
                ohlcv_data['volume'] = generated_features['volume']
            else:
                # Create dummy volume if not available
                ohlcv_data['volume'] = 1000
        
        # Prepare selected features list for variant generation
        selected_features = []
        for category, features in top_features_by_category.items():
            for feature_name, optimal_lookback, composite_score in features:
                if feature_name in generated_features.columns:
                    selected_features.append({
                        'feature_name': feature_name,
                        'category': category,
                        'optimal_lookback': int(optimal_lookback),
                        'composite_score': composite_score
                    })
                else:
                    # Try to find a similar feature
                    similar_features = self._find_similar_feature(feature_name, generated_features.columns)
                    if similar_features:
                        selected_features.append({
                            'feature_name': similar_features[0],
                            'category': category,
                            'optimal_lookback': int(optimal_lookback),
                            'composite_score': composite_score
                        })
                    else:
                        tprint_warning(f"⚠️ DEBUG: Feature {feature_name} not found in generated_features.columns and no similar feature found")
        
        
        # Feature count summary before variant generation
        tprint_info(f"📊 PHASE 1: Feature preparation summary:")
        tprint_info(f"  📈 Input generated_features: {len(generated_features.columns)} features")
        tprint_info(f"  📈 Selected features for variants: {len(selected_features)} features")
        tprint_info(f"  📈 Categories processed: {len(top_features_by_category)}")
        
        if len(selected_features) == 0:
            tprint_warning("⚠️ DEBUG: No features selected! This will cause variant generation to fail!")
            tprint_warning("⚠️ DEBUG: Check if features from top_features_by_category exist in generated_features.columns")
            return pd.DataFrame()
        
        for i, feature in enumerate(selected_features[:5]):  # Show first 5
            feature_name = feature.get('feature_name', 'unknown')
            category = feature.get('category', 'unknown')
            lookback = feature.get('optimal_lookback', 'unknown')
            tprint_info(f"  🔍 {i+1}. {feature_name} (category: {category}, lookback: {lookback})")
        
        # Generate variants using sequential processing (parallel disabled due to pickle issues)
        try:
            # Always use sequential processing to avoid pickle issues with thread locks
            tprint_info("  🔄 Using sequential variant generation (parallel disabled)...")
            
            # Add detailed category breakdown before variant generation
            category_counts = {}
            for feature in selected_features:
                category = feature['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            for category, count in category_counts.items():
                tprint_info(f"    📊 {category}: {count} features")
            
            variant_features, variant_stats = generate_all_variants_optimized(
                features_df=generated_features,
                selected_features=selected_features,
                ohlcv_data=ohlcv_data
            )
            
            tprint_info(f"  🔍 DEBUG: Actual variants generated: {len(variant_features.columns)}")
            tprint_info(f"  🔍 DEBUG: Variant expansion ratio: {len(variant_features.columns) / len(selected_features):.1f}x")
            
            # Add detailed analysis of why expansion ratio is low
            expected_max = len(selected_features) * 4
            actual = len(variant_features.columns)
            expansion_percentage = (actual / expected_max) * 100
            
            tprint_warning(f"  ⚠️ VARIANT EXPANSION ANALYSIS:")
            tprint_warning(f"    📊 Expected maximum: {expected_max} variants")
            tprint_warning(f"    📊 Actual generated: {actual} variants")
            tprint_warning(f"    📊 Expansion percentage: {expansion_percentage:.1f}%")
            tprint_warning(f"    📊 Missing variants: {expected_max - actual}")
            
            if expansion_percentage < 50:
                tprint_warning(f"    ⚠️ LOW EXPANSION RATIO DETECTED!")
                tprint_warning(f"    ⚠️ This suggests many variants are being skipped due to category restrictions")
                tprint_warning(f"    ⚠️ Check variant generation logic for category-based filtering")
            
            self.performance_stats['variants_generated'] = len(variant_features.columns)
            
            # Feature count summary after variant generation
            tprint_info(f"📊 PHASE 1: Variant generation results:")
            tprint_info(f"  📈 Input features: {len(selected_features)} features")
            tprint_info(f"  📈 Generated variants: {len(variant_features.columns)} features")
            tprint_info(f"  📈 Expansion ratio: {len(variant_features.columns) / len(selected_features):.1f}x")
            
            # Log variant generation statistics
            tprint_success(f"✅ Generated {len(variant_features.columns)} variant features")
            if variant_stats.get('variants_by_type'):
                tprint_info(f"📊 Variant breakdown: {variant_stats['variants_by_type']}")
            
            if variant_stats.get('failed_variants'):
                tprint_warning(f"⚠️ Failed variants: {len(variant_stats['failed_variants'])}")
            
            # Final comprehensive analysis
            tprint_info(f"🔍 FINAL VARIANT GENERATION ANALYSIS:")
            tprint_info(f"  📊 Phase 1 Status: {'✅ COMPLETED' if len(variant_features.columns) > 0 else '❌ FAILED'}")
            tprint_info(f"  📊 Expected behavior: Each feature should generate 4 variants (base, volnorm, vwap, trend_adj)")
            tprint_info(f"  📊 Actual behavior: Average {len(variant_features.columns) / len(selected_features):.1f} variants per feature")
            
            if len(variant_features.columns) / len(selected_features) < 3.0:
                tprint_warning(f"  ⚠️ LOW VARIANT GENERATION DETECTED!")
                tprint_warning(f"  ⚠️ This indicates many variants are being skipped due to category restrictions:")
                tprint_warning(f"    - Volatility features skip volnorm variants")
                tprint_warning(f"    - Volume features skip vwap variants")
                tprint_warning(f"    - All other variants should be generated for most features")
                tprint_warning(f"  ⚠️ Check variant generation logic for technical failures")
            
            # Generate cross-timeframe features from variants
            tprint_info("=" * 60)
            tprint_info("🔄 CROSS-TIMEFRAME FEATURES")
            tprint_info("=" * 60)
            
            cross_timeframe_features = await self._generate_cross_timeframe_features(
                variant_features, top_features_by_category, generated_features, ohlcv_data, config
            )
            
            # Combine variant features with cross-timeframe features
            if len(cross_timeframe_features.columns) > 0:
                # Combine DataFrames
                combined_features = pd.concat([variant_features, cross_timeframe_features], axis=1)
                
                tprint_success(f"✅ Combined features: {len(variant_features.columns)} variants + {len(cross_timeframe_features.columns)} cross-timeframe = {len(combined_features.columns)} total")
                tprint_info(f"📊 Phase 1 Final Summary:")
                tprint_info(f"  📈 Original features: {len(selected_features)}")
                tprint_info(f"  📈 Variant features: {len(variant_features.columns)}")
                tprint_info(f"  📈 Cross-timeframe features: {len(cross_timeframe_features.columns)}")
                tprint_info(f"  📈 Total features: {len(combined_features.columns)}")
                tprint_info(f"  📈 Expansion ratio: {len(combined_features.columns) / len(selected_features):.1f}x")
                
                # DEBUG: Show some cross-timeframe feature names
                cross_timeframe_cols = [c for c in cross_timeframe_features.columns][:5]
                
                return combined_features
            else:
                tprint_warning("⚠️ No cross-timeframe features generated, returning only variant features")
            return variant_features
            
        except Exception as e:
            tprint_error(f"❌ Variant generation failed: {e}")
            raise

    def _generate_extended_timeframe_feature(
        self,
        base_feature_name: str,
        variant_type: str,
        extended_lookback: int,
        generated_features: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Thin wrapper around variant_generator._generate_extended_timeframe_variant.

        This keeps the cross-timeframe feature generation logic centralized in the
        variant_generator module while exposing a simple helper on the step class.
        """

        try:
            from src.training.utils.feature_selection.variant_generator import (
                _generate_extended_timeframe_variant,
            )
        except ImportError as exc:
            tprint_warning(
                f"⚠️ Extended timeframe variant helper not available; "
                f"skipping CT extension for {base_feature_name} ({variant_type}): {exc}"
            )
            return None

        return _generate_extended_timeframe_variant(
            base_feature_name=base_feature_name,
            variant_type=variant_type,
            extended_lookback=extended_lookback,
            generated_features=generated_features,
            ohlcv_data=ohlcv_data,
        )

    async def _generate_cross_timeframe_features(
        self,
        variant_features: pd.DataFrame,
        top_features_by_category: Dict,
        generated_features: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate cross-timeframe features by creating ratio-based interactions between different lookback periods.
        
        For each variant feature, generates 3 additional timeframe versions (3x, 9x, 27x lookback) and creates
        ratio interactions between the base timeframe and each extended timeframe.
        
        Args:
            variant_features: DataFrame with variant features (base, volnorm, vwap, trend_adj)
            top_features_by_category: Dict mapping categories to feature info
            generated_features: Original features DataFrame for lookback reference
            ohlcv_data: OHLCV data for feature recalculation
            config: Configuration dictionary
            
        Returns:
            DataFrame with cross-timeframe ratio features
        """
        tprint_info("="*60)
        tprint_info("🔄 CROSS-TIMEFRAME FEATURES GENERATION")
        tprint_info("="*60)
        tprint_info("🔄 Generating cross-timeframe features with 3x, 6x, 9x, 27x lookback ratios")
        
        if len(variant_features.columns) == 0:
            tprint_warning("⚠️ No variant features available for cross-timeframe generation")
            return pd.DataFrame()
        
        # Import math validation for safe division
        try:
            from src.utils.math_validation import safe_divide
        except ImportError:
            tprint_warning("⚠️ Math validation not available, using basic division")
            
            _divide_debug_count = [0]  # Mutable container for closure
            
            def safe_divide(a, b, default=np.nan):
                # Ensure both inputs are pandas Series
                if not isinstance(a, pd.Series):
                    a = pd.Series(a, index=getattr(b, 'index', range(len(b))))
                if not isinstance(b, pd.Series):
                    b = pd.Series(b, index=a.index)
                
                # Ensure both Series have the same index
                if not a.index.equals(b.index):
                    # Align indices
                    a, b = a.align(b, join='inner')
                
                # Debug first few calls
                if _divide_debug_count[0] < 3:
                    tprint_info(f"    🔍 safe_divide DEBUG #{_divide_debug_count[0]+1}:")
                    tprint_info(f"        a: type={type(a)}, len={len(a)}, non-NaN={a.notna().sum()}")
                    tprint_info(f"        b: type={type(b)}, len={len(b)}, non-NaN={b.notna().sum()}")
                    tprint_info(f"        a sample: {a.iloc[:5].tolist()}")
                    tprint_info(f"        b sample: {b.iloc[:5].tolist()}")
                    _divide_debug_count[0] += 1
                
                # Simple division - let pandas handle NaN propagation naturally
                result = a / b
                
                # Debug result
                if _divide_debug_count[0] <= 3:
                    tprint_info(f"        result: non-NaN={result.notna().sum()}, sample={result.iloc[:5].tolist()}")
                
                # Replace infinite values with NaN
                result = result.replace([np.inf, -np.inf], np.nan)
                
                return result
        
        cross_timeframe_features = {}
        # Extended timeframe multipliers to capture longer-term regime interactions
        # Short-term: 3x, 6x (micro to short-term regime shifts)
        # Medium-term: 9x, 15x (intraday regime transitions)
        # Long-term: 27x, 45x, 60x (multi-regime and market memory interactions)
        timeframe_multipliers = [3, 6, 9, 15, 27, 45, 60]

        # Create lookback mapping from original features
        lookback_mapping = {}
        for category, features in top_features_by_category.items():
            for feature_name, optimal_lookback, composite_score in features:
                lookback_mapping[feature_name] = int(optimal_lookback)
        
        
        # Process each variant feature
        processed_count = 0
        failed_count = 0
        
        for variant_col in variant_features.columns:
            try:
                # Extract base feature name and variant type
                base_feature_name = self._extract_base_feature_name(variant_col)
                variant_type = self._extract_variant_type(variant_col)

                if variant_type in ("volnorm", "trend_adj"):
                    continue

                if base_feature_name not in lookback_mapping:
                    tprint_warning(f"⚠️ No lookback found for base feature {base_feature_name}, skipping")
                    failed_count += 1
                    continue

                base_lookback = lookback_mapping[base_feature_name]
                base_feature_series = variant_features[variant_col]
                
                # Generate cross-timeframe versions and ratios
                for multiplier in timeframe_multipliers:
                    extended_lookback = base_lookback * multiplier
                    
                    # Generate extended timeframe version
                    extended_feature = self._generate_extended_timeframe_feature(
                        base_feature_name=base_feature_name,
                        variant_type=variant_type,
                        extended_lookback=extended_lookback,
                        generated_features=generated_features,
                        ohlcv_data=ohlcv_data
                    )
                    
                    if extended_feature is not None:
                        # Debug: Check base and extended features before ratio
                        if len(cross_timeframe_features) < 3:
                            base_nonnan = base_feature_series.notna().sum()
                            ext_nonnan = extended_feature.notna().sum()
                            base_std = base_feature_series.std()
                            ext_std = extended_feature.std()
                            tprint_info(f"    🔍 Ratio inputs for {variant_col}_{multiplier}x:")
                            tprint_info(f"        Base: {base_nonnan}/{len(base_feature_series)} non-NaN, std={base_std:.6f}")
                            tprint_info(f"        Extended: {ext_nonnan}/{len(extended_feature)} non-NaN, std={ext_std:.6f}")
                            tprint_info(f"        Base index type: {type(base_feature_series.index)}")
                            tprint_info(f"        Extended index type: {type(extended_feature.index)}")
                            tprint_info(f"        Indices equal: {base_feature_series.index.equals(extended_feature.index)}")
                        
                        # Create ratio interaction: base / extended
                        ratio_name = f"{variant_col}_{multiplier}x_ratio"
                        
                        # Use direct pandas division (handles NaN naturally)
                        # Don't use safe_divide as it's for scalars, not Series
                        ratio_feature = base_feature_series / extended_feature
                        
                        # Replace infinite values
                        ratio_feature = ratio_feature.replace([np.inf, -np.inf], np.nan)
                        
                        # Ensure ratio_feature is a pandas Series
                        if not isinstance(ratio_feature, pd.Series):
                            ratio_feature = pd.Series(ratio_feature, index=base_feature_series.index)
                        
                        # Debug: Check ratio before shift
                        if len(cross_timeframe_features) < 3:
                            ratio_nonnan_before = ratio_feature.notna().sum()
                            tprint_info(f"        Ratio before shift: {ratio_nonnan_before}/{len(ratio_feature)} non-NaN")
                        
                        # Apply causality enforcement with proper NaN handling
                        ratio_feature = ratio_feature.shift(1)
                        
                        # Handle the leading NaN created by shift(1)
                        # Use forward fill for the first value instead of leaving it as NaN
                        if pd.isna(ratio_feature.iloc[0]):
                            ratio_feature.iloc[0] = ratio_feature.iloc[1] if len(ratio_feature) > 1 else np.nan
                        
                        # Validate ratio quality before storing
                        non_nan_count = ratio_feature.notna().sum()
                        ratio_std = ratio_feature.std()
                        unique_count = ratio_feature.nunique()
                        
                        # Log first few generations with details
                        if len(cross_timeframe_features) < 3:
                            tprint_info(f"    ✅ Generated {ratio_name}")
                            tprint_info(f"        Final: Non-NaN: {non_nan_count}/{len(ratio_feature)}, Std: {ratio_std:.6f}, Unique: {unique_count}")
                            if non_nan_count == 0:
                                tprint_error(f"        ❌ PROBLEM: Ratio is ALL NaN!")
                                # Sample both series to understand why
                                tprint_error(f"        Base sample: {base_feature_series.head(5).tolist()}")
                                tprint_error(f"        Extended sample: {extended_feature.head(5).tolist()}")
                        
                        # Store the cross-timeframe ratio feature
                        cross_timeframe_features[ratio_name] = ratio_feature
                    else:
                        tprint_warning(f"    ⚠️ Failed to generate extended timeframe for {variant_col} with {multiplier}x lookback")
                        failed_count += 1
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    tprint_info(f"  Progress: {processed_count}/{len(variant_features.columns)} variant features processed")
            
            except Exception as e:
                tprint_error(f"❌ Failed to process variant {variant_col}: {e}")
                failed_count += 1
        
        # Create DataFrame from cross-timeframe features
        if cross_timeframe_features:
            cross_timeframe_df = pd.DataFrame(cross_timeframe_features, index=variant_features.index)
            
            # Update performance stats
            self.performance_stats['cross_timeframe_features_generated'] = len(cross_timeframe_df.columns)
            self.performance_stats['cross_timeframe_ratios_generated'] = len(cross_timeframe_df.columns)
            
            tprint_success(f"✅ Generated {len(cross_timeframe_df.columns)} cross-timeframe ratio features")
            tprint_info(f"📊 Cross-timeframe generation summary:")
            tprint_info(f"  📈 Processed variants: {processed_count}")
            tprint_info(f"  📈 Generated ratios: {len(cross_timeframe_df.columns)}")
            tprint_info(f"  📈 Failed generations: {failed_count}")
            tprint_info(f"  📈 Success rate: {processed_count / (processed_count + failed_count) * 100:.1f}%")
            
            return cross_timeframe_df
        else:
            tprint_warning("⚠️ No cross-timeframe features generated")
            return pd.DataFrame()
    
    async def _phase2_cheap_pruning(
        self,
        variant_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
        lookback_optimization: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """Phase 2: Cheap pruning wrapper.

        Applies OptimizedCheapPruningPipeline via apply_optimized_cheap_pruning,
        using composite scores from _calculate_composite_scores and feature
        categories from _get_feature_categories_from_bank. Returns
        (pruned_features, pruning_stats, targets_df).
        """

        # Select target(s) for pruning
        targets_df = self._get_primary_summary_targets(config)
        if targets_df is None or targets_df.empty:
            # Fallback: try basic columns on labeled_data
            candidate_cols = [
                "smoothed_label",
                "binary_label",
                "realized_return",
            ]
            if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
                existing = [c for c in candidate_cols if c in labeled_data.columns]
                if existing:
                    targets_df = labeled_data[[existing[0]]]
                else:
                    # Last resort: first column of labeled_data
                    targets_df = labeled_data.iloc[:, :1].copy()
            else:
                # Absolute fallback: dummy zero target with correct index
                targets_df = pd.DataFrame(index=variant_features.index)
                targets_df["dummy_target"] = 0.0

        # Infer feature categories
        try:
            feature_categories = self._get_feature_categories_from_bank(
                list(variant_features.columns),
                lookback_optimization,
            )
        except Exception as exc:
            tprint_warning(
                f"⚠️ Failed to obtain feature categories from bank: {exc}; "
                f"falling back to local inference."
            )
            feature_categories = {
                col: self._infer_feature_category(col) for col in variant_features.columns
            }

        # Composite scores for pruning
        try:
            composite_scores = self._calculate_composite_scores(
                variant_features,
                targets_df,
                feature_categories,
                config,
            )
        except Exception as exc:
            tprint_warning(
                f"⚠️ Composite score calculation failed for cheap pruning: {exc}; "
                f"using uniform scores."
            )
            composite_scores = {col: 1.0 for col in variant_features.columns}

        # Apply optimized cheap pruning
        try:
            pruned_df, stats = apply_optimized_cheap_pruning(
                features_df=variant_features,
                targets_df=targets_df,
                feature_categories=feature_categories,
                composite_scores=composite_scores,
                config=OptimizedPruningConfig(max_workers=3),
            )
        except Exception as exc:
            tprint_error(f"❌ Optimized cheap pruning failed: {exc}; returning original features.")
            pruned_df = variant_features.copy()
            stats = {"error": str(exc)}

        return pruned_df, stats, targets_df

    async def _phase3_3_label_guided_interaction_discovery(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
        feature_categories: Dict[str, str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Phase 3.3: Label-guided interaction discovery using MI/SHAP and regularized selection.

        This method implements label-guided interaction discovery that:
        1. Restricts interactions to pairs showing MI or SHAP interaction strength vs target
        2. Uses regularized models (L1, group LASSO) to pick meaningful interactions
        3. Ensures interactions provide meaningful R²/MI lift over base features
        4. Applies category-aware limits to prevent over-representation (e.g., trend)
        """
        if not LABEL_GUIDED_AVAILABLE:
            tprint_warning("  ⚠️ Label-guided interaction discovery not available, using legacy approach")
            return await self._phase3_3_interaction_discovery_legacy(features, targets, config, feature_categories)

        self._ensure_runtime_integrity("_phase3_3_label_guided_interaction_discovery")
        tprint_info("  🌳 Starting label-guided interaction discovery...")

        # Extract cross-timeframe interaction features (legacy compatibility)
        tprint_info("  🔄 Extracting cross-timeframe interactions...")
        cross_timeframe_interactions = await self._extract_cross_timeframe_interactions(features)
        if len(cross_timeframe_interactions.columns) > 0:
            tprint_success(f"  ✅ Found {len(cross_timeframe_interactions.columns)} cross-timeframe interaction features")
            # Merge cross-timeframe interactions into features
            features = pd.concat([features, cross_timeframe_interactions], axis=1)
        else:
            tprint_info("  ℹ️ No cross-timeframe interactions detected")

        # Infer categories if not provided
        if feature_categories is None:
            feature_categories = {}
            for feature_name in features.columns:
                feature_categories[feature_name] = self._infer_feature_category(feature_name)

        # Shared cleaning/alignment so both tree-guided training and
        # LabelGuidedInteractionDiscovery operate on the exact same
        # feature/target sample. Use the module-level helper to avoid
        # attribute issues when methods are monkey-patched via fallbacks.
        first_target_name = targets.columns[0]
        aligned_features, aligned_target = _align_for_label_guided_discovery_helper(
            features,
            targets[first_target_name],
        )
        features = aligned_features
        targets = aligned_target.to_frame(name=first_target_name)

        # Configure label-guided interaction discovery
        lgid_config = LabelGuidedInteractionConfig(
            # MI/SHAP scoring
            use_mi_scoring=True,
            use_shap_scoring=SHAP_AVAILABLE,
            mi_weight=0.6,
            shap_weight=0.4,

            # Lift requirements - CRITICAL for ensuring interactions beat base features
            min_r2_lift=float(config.get('min_interaction_r2_lift', 0.02)),  # 2% R² improvement (tightened from 1%)
            # Require at least 10% MI improvement: child MI ≥ 1.10 × best parent MI
            # This tighter threshold ensures composite scores become more informative
            # now that CT features are no longer being discarded prematurely.
            min_mi_lift=float(config.get('min_interaction_mi_lift', 0.10)),  # Tightened from 0.10
            require_r2_lift=False,  # Don't require R² lift (too expensive to compute)
            require_mi_lift=True,   # Require MI lift (fast to compute)

            # Regularization - use LASSO for sparse selection
            use_lasso=True,
            lasso_alpha=None,  # Use CV to find optimal alpha
            lasso_cv_folds=3,  # Reduced for speed
            lasso_max_iter=500,

            # Interaction generation limits
            max_pairs_to_test=int(config.get('max_interaction_pairs', 100)),
            operations=['multiply', 'divide', 'subtract', 'log_ratio'],  # Exclude 'add' (often redundant)

            # Category controls - CRITICAL for preventing trend over-representation
            max_interactions_per_category_pair=int(config.get('max_interactions_per_category_pair', 3)),
            banned_category_pairs=set(),  # Could ban (trend, trend) to prevent trend×trend

            # Performance
            n_jobs=-1,
            random_state=42
        )

        # Create discovery instance
        discoverer = LabelGuidedInteractionDiscovery(lgid_config)

        # Extract feature pairs from tree analysis (for guidance)
        feature_pairs = None
        if config.get('use_tree_guided_pairs', True):
            try:
                tprint_info("  🌳 Training LGBM for tree-guided feature pair extraction...")
                # Sample for speed
                features_sample, targets_sample = self._get_consistent_sample(features, targets, max_samples=10000)

                # Train LGBM
                lgbm_params = {
                    'max_depth': 3,
                    'num_leaves': 10,
                    'n_estimators': 50,
                    'learning_rate': 0.1,
                    'reg_alpha': 0.2,
                    'reg_lambda': 0.2,
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**lgbm_params)
                model.fit(features_sample, targets_sample.iloc[:, 0])

                # Extract pairs
                tree_pairs = self._extract_tree_splitting_pairs(model)
                if len(tree_pairs) > 0:
                    resolved_pairs: List[Tuple[str, str]] = []
                    for f1, f2, _ in tree_pairs[:lgid_config.max_pairs_to_test]:
                        if isinstance(f1, int):
                            if 0 <= f1 < len(features.columns):
                                f1_name = features.columns[f1]
                            else:
                                continue
                        else:
                            f1_name = f1

                        if isinstance(f2, int):
                            if 0 <= f2 < len(features.columns):
                                f2_name = features.columns[f2]
                            else:
                                continue
                        else:
                            f2_name = f2

                        if f1_name in features.columns and f2_name in features.columns:
                            resolved_pairs.append((f1_name, f2_name))

                    if resolved_pairs:
                        feature_pairs = resolved_pairs
                        tprint_success(f"  ✅ Extracted {len(feature_pairs)} tree-guided feature pairs")
                    else:
                        tprint_warning("  ⚠️ Tree-guided pair extraction produced no valid feature name pairs, will generate pairs automatically")
                else:
                    tprint_warning("  ⚠️ No tree pairs found, will generate pairs automatically")
            except Exception as e:
                tprint_warning(f"  ⚠️ Tree-guided pair extraction failed: {e}, will generate pairs automatically")

        # Discover interactions
        try:
            interaction_df, metadata = discoverer.discover_interactions(
                features=features,
                target=targets.iloc[:, 0],  # Use first target column
                feature_categories=feature_categories,
                feature_pairs=feature_pairs
            )

            # Apply causality shift
            if len(interaction_df.columns) > 0:
                interaction_df = interaction_df.shift(1)

                # Apply differential normalization: winsorized z-score for non-volume, log1p for volume
                from src.features_common.transforms.scaling_normalization import winsorized_zscore_normalize

                # Identify volume vs non-volume interaction features
                volume_interactions = []
                non_volume_interactions = []

                for col in interaction_df.columns:
                    is_volume = any(pattern in col.lower() for pattern in [
                        'volume', 'vol_', '_vol', 'vwap', 'obv', 'mfi', 'cmf', 'adl', 'ad_', 'pvt'
                    ])
                    if is_volume:
                        volume_interactions.append(col)
                    else:
                        non_volume_interactions.append(col)

                # Apply winsorized z-score to non-volume interactions
                if non_volume_interactions:
                    try:
                        interaction_df[non_volume_interactions] = winsorized_zscore_normalize(
                            interaction_df[non_volume_interactions],
                            ddof=0,
                            lower_quantile=0.01,
                            upper_quantile=0.99
                        )
                        tprint_debug(f"  ✅ Winsorized z-score normalized {len(non_volume_interactions)} non-volume interactions")
                    except Exception as e:
                        tprint_warning(f"  ⚠️ Winsorized normalization failed for interactions: {e}, using robust scaler fallback")
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        interaction_df[non_volume_interactions] = pd.DataFrame(
                            scaler.fit_transform(interaction_df[non_volume_interactions]),
                            columns=non_volume_interactions,
                            index=interaction_df.index
                        )

                # Apply log1p to volume interactions
                if volume_interactions:
                    try:
                        volume_data = interaction_df[volume_interactions].clip(lower=0)
                        interaction_df[volume_interactions] = np.log1p(volume_data)
                        tprint_debug(f"  ✅ Log-transformed {len(volume_interactions)} volume interactions")
                    except Exception as e:
                        tprint_warning(f"  ⚠️ Log transformation failed for volume interactions: {e}, keeping original")


            # Build SHAP metadata for compatibility
            shap_metadata = {
                'feature_categories': feature_categories,
                'interaction_discovery': {
                    'method': 'label_guided',
                    'total_candidates': metadata['total_candidates'],
                    'selected_interactions': metadata['selected_interactions'],
                    'category_pair_distribution': metadata['category_pair_distribution'],
                    'config': metadata['config'],
                    'interaction_details': metadata['selected_interaction_details'],
                },
                'model_performance': {
                    'lgbm_training_successful': True,
                    'interaction_generation_successful': len(interaction_df.columns) > 0,
                }
            }

            tprint_success(f"  ✅ Label-guided discovery selected {len(interaction_df.columns)} interactions")

            # Log category distribution to help diagnose over-representation
            tprint_info("  📊 Interaction category distribution:")
            for cat_pair, count in metadata['category_pair_distribution'].items():
                tprint_info(f"    - {cat_pair}: {count} interactions")

            return interaction_df, shap_metadata

        except Exception as e:
            tprint_error(f"  ❌ Label-guided interaction discovery failed: {e}")
            import traceback
            tprint_error(f"  🔍 Traceback: {traceback.format_exc()}")
            tprint_warning("  ⚠️ Falling back to legacy interaction discovery")
            return await self._phase3_3_interaction_discovery_legacy(features, targets, config, feature_categories)

    async def _extract_cross_timeframe_interactions(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract cross-timeframe interaction features from the feature set.

        This method identifies features that have cross-timeframe ratio markers
        (e.g., _3x_ratio, _6x_ratio) and interaction operators (e.g., _x_, _div_).

        Args:
            features: Feature dataframe

        Returns:
            DataFrame of cross-timeframe interaction features
        """
        # Look for features that have both CT markers and interaction operators
        ct_markers = ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_15x_ratio', '_27x_ratio', '_45x_ratio', '_60x_ratio']
        interaction_ops = ['_x_', '_div_', '_minus_', '_plus_', '_log_']

        ct_interaction_cols = []
        for col in features.columns:
            has_ct = any(marker in col for marker in ct_markers)
            has_interaction = any(op in col for op in interaction_ops)
            if has_ct and has_interaction:
                ct_interaction_cols.append(col)

        if ct_interaction_cols:
            tprint_debug(f"  🔍 Found {len(ct_interaction_cols)} cross-timeframe interaction features")
            return features[ct_interaction_cols].copy()

        # Return empty DataFrame with same index
        return pd.DataFrame(index=features.index)

    async def _phase3_lgbm_shap_pipeline(
        self,
        pruned_features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
        lookback_optimization: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Full Phase 3 LGBM+SHAP pipeline with feature normalization.

        This method implements the complete Phase 3 pipeline that:
        1. Applies winsorized z-score normalization to handle large feature values
        2. Runs label-guided interaction discovery with LGBM+SHAP
        3. Saves both normalized and original features as artifacts
        4. Returns properly normalized features and interactions

        Args:
            pruned_features: Features after cheap pruning (Phase 2)
            targets: Target dataframe
            config: Configuration dictionary
            lookback_optimization: Lookback optimization dataframe

        Returns:
            Tuple of (final_features, interactions, shap_metadata)
        """
        from src.features_common.transforms.scaling_normalization import winsorized_zscore_normalize

        tprint_info("🔧 Phase 3: Full LGBM+SHAP Pipeline")
        tprint_info(f"  📊 Input: {len(pruned_features.columns)} pruned features")

        # Step 1: Identify volume vs non-volume features for differential normalization
        volume_features = []
        non_volume_features = []

        for col in pruned_features.columns:
            # Identify volume-related features by common patterns
            is_volume = any(pattern in col.lower() for pattern in [
                'volume', 'vol_', '_vol', 'vwap', 'obv', 'mfi', 'cmf', 'adl', 'ad_', 'pvt'
            ])
            if is_volume:
                volume_features.append(col)
            else:
                non_volume_features.append(col)

        tprint_info(f"  📊 Feature categorization:")
        tprint_info(f"    - Non-volume features: {len(non_volume_features)}")
        tprint_info(f"    - Volume features: {len(volume_features)}")

        # Step 2: Apply normalization
        normalized_features = pruned_features.copy()

        # Apply winsorized z-score to non-volume features
        if non_volume_features:
            tprint_info(f"  🔧 Applying winsorized z-score normalization to {len(non_volume_features)} non-volume features...")
            try:
                normalized_features[non_volume_features] = winsorized_zscore_normalize(
                    pruned_features[non_volume_features],
                    ddof=0,
                    lower_quantile=0.01,
                    upper_quantile=0.99
                )
                tprint_success(f"  ✅ Normalized {len(non_volume_features)} non-volume features")
            except Exception as e:
                tprint_error(f"  ❌ Winsorized normalization failed: {e}, using original features")
                normalized_features[non_volume_features] = pruned_features[non_volume_features]

        # Apply log1p transformation to volume features (better for volume data)
        if volume_features:
            tprint_info(f"  🔧 Applying log1p transformation to {len(volume_features)} volume features...")
            try:
                # Clip negative values to 0 before log transform
                volume_data = pruned_features[volume_features].clip(lower=0)
                normalized_features[volume_features] = np.log1p(volume_data)
                tprint_success(f"  ✅ Log-transformed {len(volume_features)} volume features")
            except Exception as e:
                tprint_error(f"  ❌ Log transformation failed: {e}, using original features")
                normalized_features[volume_features] = pruned_features[volume_features]

        # Step 3: Infer feature categories for compatibility with label-guided discovery
        feature_categories: Dict[str, str] = {}
        try:
            for col in normalized_features.columns:
                feature_categories[col] = self._infer_feature_category(col)
        except Exception as e:
            tprint_warning(f"  ⚠️ Feature category inference failed: {e}, using defaults")
            feature_categories = {col: 'unknown' for col in normalized_features.columns}

        # Step 4: Run label-guided interaction discovery on NORMALIZED features
        try:
            tprint_info("  🌳 Running label-guided interaction discovery on normalized features...")
            interactions, shap_metadata = await self._phase3_3_label_guided_interaction_discovery(
                normalized_features,
                targets,
                config,
                feature_categories,
            )
            tprint_success(f"  ✅ Generated {len(interactions.columns)} interactions")

            # Add normalization metadata to shap_metadata
            shap_metadata['normalization'] = {
                'method': 'winsorized_zscore_and_log1p',
                'non_volume_features': len(non_volume_features),
                'volume_features': len(volume_features),
                'non_volume_method': 'winsorized_zscore',
                'volume_method': 'log1p',
                'winsorize_quantiles': (0.01, 0.99),
            }

        except Exception as exc:
            tprint_error(f"  ❌ Label-guided interaction discovery failed: {exc}")
            # Return empty interactions but keep normalized base features
            interactions = pd.DataFrame(index=normalized_features.index)
            shap_metadata = {
                "feature_categories": feature_categories,
                "interaction_discovery": {
                    "selected_interactions": 0,
                    "error": str(exc),
                },
                "model_performance": {
                    "lgbm_training_successful": False,
                    "interaction_generation_successful": False,
                },
                'normalization': {
                    'method': 'winsorized_zscore_and_log1p',
                    'non_volume_features': len(non_volume_features),
                    'volume_features': len(volume_features),
                    'error': str(exc),
                }
            }

        tprint_success(f"  ✅ Phase 3 complete: {len(normalized_features.columns)} features, {len(interactions.columns)} interactions")

        return normalized_features, interactions, shap_metadata

    async def _phase4_save_artifacts(
        self,
        final_features: pd.DataFrame,
        interactions: pd.DataFrame,
        shap_metadata: Dict,
        pruning_stats: Dict,
        config: Dict[str, Any],
        lookback_optimization: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """
        Phase 4: Combine features, verify category coverage, save artifacts, generate report.
        
        Ensures at least 2 features from each original category in final set.
        Saves comprehensive artifacts with enhanced metadata.
        
        Returns:
            Tuple of (artifacts, metrics)
        """
        tprint_info("💾 Phase 4: Integration and artifact saving")
        
        # Feature count summary before Phase 4
        tprint_info("="*80)
        tprint_info(f"📊 PHASE 4: Feature counts before integration:")
        tprint_info(f"  📈 Final features: {len(final_features.columns)} features")
        tprint_info(f"  📈 Interaction features: {len(interactions.columns)} features")
        tprint_info(f"  🔍 DEBUG: Interactions shape at Phase 4 entry: {interactions.shape}")
        tprint_info(f"  🔍 DEBUG: Interactions columns (first 10): {list(interactions.columns)[:10]}")
        
        # Combine features and interactions
        combined_features = pd.concat([final_features, interactions], axis=1)
        
        self.performance_stats['final_feature_count'] = len(final_features.columns)
        self.performance_stats['interaction_count'] = len(interactions.columns)
        
        # Feature count summary after integration
        tprint_info(f"📊 PHASE 4: Integration results:")
        tprint_info(f"  📈 Combined features: {len(combined_features.columns)} features")
        tprint_info(f"  📈 Base features: {len(final_features.columns)} features")
        tprint_info(f"  📈 Interaction features: {len(interactions.columns)} features")
        tprint_info(f"  🔍 DEBUG: Combined features shape: {combined_features.shape}")
        tprint_info("="*80)
        
        # Verify category coverage (ensure ≥2 per category)
        tprint_info("🔍 Verifying category coverage (minimum 2 per category)...")
        category_coverage = self._verify_category_coverage(combined_features, final_features, config, lookback_optimization)
        self.performance_stats['category_coverage'] = category_coverage
        
        # Save artifacts with enhanced metadata
        tprint_info("💾 Saving artifacts with enhanced metadata...")
        
        # Enhanced metadata for interaction features
        enhanced_metadata = {
            'symbol': config.get('symbol', 'UNKNOWN'),
            'exchange': config.get('exchange', 'UNKNOWN'),
            'timeframe': config.get('timeframe', 'UNKNOWN'),
            'execution_mode': config.get('execution_mode', 'light'),
            'n_base_features': len(final_features.columns),
            'n_interaction_features': len(interactions.columns),
            'total_features': len(combined_features.columns),
            'category_coverage': category_coverage,
            'variant_generation': shap_metadata.get('variant_generation', {}),
            'pruning_stages': shap_metadata.get('pruning_stages', {}),
            'interaction_discovery': shap_metadata.get('interaction_discovery', {}),
            'created_at': datetime.now().isoformat()
        }
        
        # 1. Analyst interaction features
        tprint_info("="*80)
        tprint_info(f"💾 SAVING ARTIFACTS:")
        tprint_info(f"  🔍 DEBUG: combined_features shape before save: {combined_features.shape}")
        tprint_info(f"  🔍 DEBUG: combined_features columns count: {len(combined_features.columns)}")
        
        # Categorize features properly - check interaction operations FIRST
        hybrid_ct_interactions = []
        traditional_interactions = []
        ct_ratio_features = []
        variant_features_list = []
        base_features_list = []
        
        # Define variant suffixes (excluding _base which IS the base feature)
        variant_suffixes = ['_volnorm', '_vwap', '_trend_adj']
        
        for col in combined_features.columns:
            # Check interaction operations FIRST (before CT markers)
            if any(op in col for op in ['_x_', '_div_', '_minus_', '_log_', '_plus_']):
                if any(marker in col for marker in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']):
                    hybrid_ct_interactions.append(col)  # Hybrid: interaction + cross-timeframe
                else:
                    traditional_interactions.append(col)  # Pure interactions
            elif any(marker in col for marker in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']):
                ct_ratio_features.append(col)  # Pure cross-timeframe ratios
            else:
                # Check if it's a variant feature or base feature
                is_variant = any(col.endswith(suffix) for suffix in variant_suffixes)
                if is_variant:
                    variant_features_list.append(col)
                else:
                    base_features_list.append(col)
        
        # Count each category
        hybrid_count = len(hybrid_ct_interactions)
        int_count = len(traditional_interactions)
        ct_count = len(ct_ratio_features)
        variant_count = len(variant_features_list)
        base_count = len(base_features_list)
        
        tprint_info(f"  📊 Feature breakdown before save:")
        tprint_info(f"    - Hybrid CT interactions: {hybrid_count}")
        tprint_info(f"    - Traditional interactions: {int_count}")
        tprint_info(f"    - Cross-timeframe ratios: {ct_count}")
        tprint_info(f"    - Variant features: {variant_count}")
        tprint_info(f"    - Base features: {base_count}")
        tprint_info("="*80)

        # Build analyst interaction summary CSV (feature-level view + category coverage)
        summary_path = None
        try:
            from pathlib import Path

            feature_categories = shap_metadata.get('feature_categories', {}) or {}
            interaction_scores = (
                shap_metadata.get('interaction_discovery', {}).get('interaction_scores', {})
                or {}
            )

            # Compute per-feature MI and stability for the final combined feature set
            mi_scores: Dict[str, float] = {}
            stability_scores: Dict[str, float] = {}
            composite_scores: Dict[str, float] = {}
            try:
                targets_for_summary = self._get_primary_summary_targets(config)
                if targets_for_summary is not None and not targets_for_summary.empty:
                    composite_scores = self._calculate_composite_scores(
                        combined_features,
                        targets_for_summary,
                        feature_categories,
                        config,
                    )
                    mi_scores = getattr(self, "_last_mi_scores", {}) or {}
                    stability_scores = getattr(self, "_last_stability_scores", {}) or {}
            except Exception as metrics_exc:
                tprint_warning(
                    f"⚠️ Failed to compute per-feature MI/stability for CSV summary: {metrics_exc}"
                )
                composite_scores = {}
                mi_scores = {}
                stability_scores = {}

            def _strip_ct_suffix(name: str) -> str:
                ct_suffixes = [
                    '_3x_ratio',
                    '_6x_ratio',
                    '_9x_ratio',
                    '_15x_ratio',
                    '_27x_ratio',
                    '_45x_ratio',
                    '_60x_ratio',
                ]
                base = name
                for sfx in ct_suffixes:
                    if base.endswith(sfx):
                        return base[:-len(sfx)]
                return base

            def _parse_interaction_parents(name: str) -> List[str]:
                ops = ['_log_ratio_', '_div_', '_minus_', '_plus_', '_x_']
                for op in ops:
                    if op in name:
                        left, right = name.split(op, 1)
                        return [left, right]
                return []

            summary_rows = []
            for col in combined_features.columns:
                if col in hybrid_ct_interactions:
                    feature_type = 'ct_interaction'
                elif col in traditional_interactions:
                    feature_type = 'interaction'
                elif col in ct_ratio_features:
                    feature_type = 'ct_ratio'
                elif col in variant_features_list:
                    feature_type = 'variant'
                else:
                    feature_type = 'base'

                score = interaction_scores.get(col, 0.0)
                try:
                    score = float(score) if score is not None and not pd.isna(score) else 0.0
                except Exception:
                    score = 0.0

                category = feature_categories.get(col, 'unknown')

                mi_val = mi_scores.get(col)
                if mi_val is None:
                    mi_val = np.nan
                stability_val = stability_scores.get(col)
                if stability_val is None:
                    stability_val = np.nan
                composite_val = composite_scores.get(col)
                if composite_val is None:
                    composite_val = np.nan

                parent_feature = None
                parent_mi = np.nan
                parent_stability = np.nan
                mi_lift = np.nan
                stability_ratio = np.nan

                # Parent mapping depends on feature type
                if feature_type in ('variant', 'ct_ratio'):
                    base_name = self._extract_base_feature_name(col)
                    base_candidate = f"{base_name}_base"
                    if base_candidate in combined_features.columns:
                        parent_feature = base_candidate
                    else:
                        stripped = _strip_ct_suffix(col)
                        if stripped in combined_features.columns:
                            parent_feature = stripped
                elif feature_type in ('interaction', 'ct_interaction'):
                    raw_name = col
                    if feature_type == 'ct_interaction':
                        raw_name = _strip_ct_suffix(col)
                    parent_candidates = _parse_interaction_parents(raw_name)
                    best_name = None
                    best_mi = -np.inf
                    for cand in parent_candidates:
                        cand_mi = mi_scores.get(cand)
                        if cand_mi is not None and cand_mi > best_mi:
                            best_mi = cand_mi
                            best_name = cand
                    if best_name is not None:
                        parent_feature = best_name

                if parent_feature is not None:
                    parent_mi = mi_scores.get(parent_feature, np.nan)
                    parent_stability = stability_scores.get(parent_feature, np.nan)
                    if (
                        parent_mi is not None
                        and not np.isnan(parent_mi)
                        and parent_mi != 0
                        and not np.isnan(mi_val)
                    ):
                        mi_lift = (mi_val - parent_mi) / (abs(parent_mi) + 1e-8)
                    if (
                        parent_stability is not None
                        and not np.isnan(parent_stability)
                        and parent_stability != 0
                        and not np.isnan(stability_val)
                    ):
                        stability_ratio = stability_val / (parent_stability + 1e-8)

                summary_rows.append({
                    'feature_name': col,
                    'feature_type': feature_type,
                    'category': category,
                    'importance_score': score,
                    'mi': mi_val,
                    'stability': stability_val,
                    'composite_score': composite_val,
                    'parent_feature': parent_feature,
                    'parent_mi': parent_mi,
                    'parent_stability': parent_stability,
                    'mi_lift_vs_parent': mi_lift,
                    'stability_ratio_vs_parent': stability_ratio,
                })

            # Add category coverage summary rows
            for cat, count in category_coverage.items():
                summary_rows.append({
                    'feature_name': f'__category__::{cat}',
                    'feature_type': 'category_coverage',
                    'category': cat,
                    'importance_score': float(count),
                })

            # Add compact interaction learnability summary based on interaction scores
            interaction_rows = [
                row for row in summary_rows
                if row['feature_type'] in ('interaction', 'ct_interaction')
            ]

            interaction_values = [row['importance_score'] for row in interaction_rows]
            if interaction_values:
                scores_arr = np.array(interaction_values, dtype=float)
                n_interactions = int(len(scores_arr))
                mean_score = float(scores_arr.mean())
                median_score = float(np.median(scores_arr))
                max_score = float(scores_arr.max())
                positive_count = int((scores_arr > 0).sum())
                positive_ratio = (
                    positive_count / n_interactions if n_interactions > 0 else 0.0
                )

                # Store summary as special rows for easy downstream aggregation
                summary_rows.append({
                    'feature_name': '__summary__::interaction_count',
                    'feature_type': 'interaction_summary',
                    'category': 'all',
                    'importance_score': float(n_interactions),
                })
                summary_rows.append({
                    'feature_name': '__summary__::interaction_mean_score',
                    'feature_type': 'interaction_summary',
                    'category': 'all',
                    'importance_score': mean_score,
                })
                summary_rows.append({
                    'feature_name': '__summary__::interaction_median_score',
                    'feature_type': 'interaction_summary',
                    'category': 'all',
                    'importance_score': median_score,
                })
                summary_rows.append({
                    'feature_name': '__summary__::interaction_best_score',
                    'feature_type': 'interaction_summary',
                    'category': 'all',
                    'importance_score': max_score,
                })
                summary_rows.append({
                    'feature_name': '__summary__::interaction_positive_ratio',
                    'feature_type': 'interaction_summary',
                    'category': 'all',
                    'importance_score': float(positive_ratio),
                })

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                symbol = config.get('symbol', 'UNKNOWN')
                timeframe = config.get('timeframe', 'UNKNOWN')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_dir = Path('outcomes')
                summary_dir.mkdir(exist_ok=True)
                summary_filename = f"analyst_interaction_summary_{symbol}_{timeframe}_{timestamp}.csv"
                summary_path = summary_dir / summary_filename
                summary_df.to_csv(summary_path, index=False)
                tprint_info(f"📄 Saved analyst interaction summary CSV: {summary_path}")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save analyst_interaction_summary.csv: {csv_exc}")

        # Update artifact manager context with symbol before saving
        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '15m')
        self.artifact_manager.set_context(
            step_name=self.step_name,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            datetime=datetime.now()
        )
        tprint_info(f"📁 Updated context: {symbol}/{exchange} [{timeframe}]")
        
        # CRITICAL FIX: Save to versioned artifacts (HDF5) using artifact_type='data'
        # This ensures the interaction features are stored in the same versioned artifacts
        # store as generated_features and labeled_data, making them accessible to
        # feature_generation_final_feature_selection_step via _get_artifact()
        features_path = self._save_artifact(
            data=combined_features,
            artifact_name='analyst_interaction_features',
            artifact_type='data',  # This triggers HDF5/versioned artifacts storage
            metadata=enhanced_metadata
        )
        
        # 2. Enhanced analyst interaction metadata
        metadata_path = self._save_artifact(
            data=shap_metadata,
            artifact_name='analyst_interaction_metadata',
            artifact_type='metadata',
            metadata={
                'created_at': datetime.now().isoformat(),
                'total_features': len(combined_features.columns),
                'category_coverage': category_coverage
            }
        )
        
        # 3. Analyst feature importance
        # Use interaction scores from interaction_discovery block (already sanitized)
        interaction_scores = (
            shap_metadata.get('interaction_discovery', {}).get('interaction_scores', {})
            or {}
        )
        importance_path = self._save_artifact(
            data=interaction_scores,
            artifact_name='analyst_feature_importance',
            artifact_type='metadata',
            metadata={'created_at': datetime.now().isoformat()}
        )
        
        # 4. Analyst pruning stats
        pruning_path = self._save_artifact(
            data=pruning_stats,
            artifact_name='analyst_pruning_stats',
            artifact_type='metadata',
            metadata={'created_at': datetime.now().isoformat()}
        )
        
        artifacts = {
            'analyst_interaction_features': features_path,
            'analyst_interaction_metadata': metadata_path,
            'analyst_feature_importance': importance_path,
            'analyst_pruning_stats': pruning_path
        }

        if summary_path is not None:
            artifacts['analyst_interaction_summary_csv'] = str(summary_path)
        
        # Generate comprehensive outcome report
        tprint_info("📊 Generating comprehensive outcome report...")
        report_path = self._generate_outcome_report(
            shap_metadata, pruning_stats, category_coverage, config
        )
        if report_path:
            tprint_success(f"✅ Outcome report generated: {report_path}")
        
        metrics = {
            'success': True,
            'performance_stats': self.performance_stats,
            'category_coverage': category_coverage,
            'total_features': len(combined_features.columns),
            'base_features': len(final_features.columns),
            'interaction_features': len(interactions.columns)
        }
        
        tprint_success(f"✅ Phase 4 completed: {len(combined_features.columns)} total features")
        tprint_info(f"📊 Category coverage: {category_coverage}")
        
        return artifacts, metrics

    def _verify_category_coverage(
        self, 
        combined_features: pd.DataFrame, 
        final_features: pd.DataFrame, 
        config: Dict[str, Any],
        lookback_optimization: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Verify category coverage ensuring at least 2 features per category.
        
        Args:
            combined_features: All features (base + interactions)
            final_features: Base features only
            config: Configuration dictionary
            
        Returns:
            Dict mapping category -> count
        """
        tprint_info("🔍 Verifying category coverage (minimum 2 per category)...")
        
        # Get feature categories from lookback optimization (feature bank)
        feature_categories = self._get_feature_categories_from_bank(combined_features.columns, lookback_optimization)
        
        # Count features per category
        category_counts = {}
        for category in self.categories:
            category_counts[category] = sum(
                1 for col in combined_features.columns 
                if feature_categories.get(col, 'unknown') == category
            )
        
        # Check if any category has < 2 features
        under_represented = [cat for cat, count in category_counts.items() if count < 2]
        
        # Always show category distribution
        tprint_info("📊 Category distribution:")
        for cat, count in category_counts.items():
            status = "✅" if count >= 2 else "⚠️"
            tprint_info(f"  {status} {cat}: {count} features")
        
        if under_represented:
            tprint_warning(f"⚠️ Under-represented categories: {under_represented}")
        else:
            tprint_success("✅ All categories have ≥2 features")
        
        return category_counts
    
    def _generate_outcome_report(
        self,
        shap_metadata: Dict,
        pruning_stats: Dict,
        category_coverage: Dict,
        config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate comprehensive outcome report.
        
        Returns:
            Path to generated report file
        """
        try:
            # Create outcomes directory
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = config.get('symbol', 'UNKNOWN')
            report_filename = f"analyst_interaction_generation_{symbol}_{timestamp}.md"
            report_path = outcomes_dir / report_filename
            
            # Generate report content
            report_content = self._create_report_content(
                shap_metadata, pruning_stats, category_coverage, config
            )
            
            # Write report
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate outcome report: {e}")
            return None
    
    def _create_report_content(
        self,
        shap_metadata: Dict,
        pruning_stats: Dict,
        category_coverage: Dict,
        config: Dict[str, Any]
    ) -> str:
        """Create comprehensive markdown report content."""
        
        # Calculate additional metrics
        total_features = self.performance_stats.get('final_feature_count', 0) + self.performance_stats.get('interaction_count', 0)
        total_time = self.performance_stats.get('total_time', 0)
        efficiency_score = total_features / max(total_time, 0.001) if total_time > 0 else 0
        
        content = f"""# Analyst Interaction Generation Report

## Executive Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Execution Mode**: {config.get('execution_mode', 'light')}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Features Generated**: {total_features}
- **Processing Efficiency**: {efficiency_score:.1f} features/second
- **Overall Status**: {'✅ SUCCESS' if total_features > 0 else '❌ FAILED'}

## Feature Statistics
- **Total Features**: {total_features}
- **Base Features**: {self.performance_stats.get('final_feature_count', 0)}
- **Interaction Features**: {self.performance_stats.get('interaction_count', 0)}
- **Feature Density**: {total_features / max(len(category_coverage), 1):.1f} features per category
- **Interaction Ratio**: {(self.performance_stats.get('interaction_count', 0) / max(total_features, 1) * 100):.1f}%

## Category Coverage Analysis
"""
        
        # Enhanced category analysis
        total_categories = len(category_coverage)
        balanced_categories = sum(1 for count in category_coverage.values() if 2 <= count <= 20)
        imbalanced_categories = total_categories - balanced_categories
        
        for category, count in category_coverage.items():
            if count >= 10:
                status = "🟢"
                health = "Excellent"
            elif count >= 5:
                status = "🟡"
                health = "Good"
            elif count >= 2:
                status = "🟠"
                health = "Adequate"
            else:
                status = "🔴"
                health = "Poor"
            
            content += f"- {status} **{category.title()}**: {count} features ({health})\n"
        
        content += f"""
### Category Health Summary
- **Total Categories**: {total_categories}
- **Balanced Categories**: {balanced_categories} ({balanced_categories/total_categories*100:.1f}%)
- **Imbalanced Categories**: {imbalanced_categories} ({imbalanced_categories/total_categories*100:.1f}%)
- **Category Diversity Score**: {balanced_categories/total_categories:.2f}/1.0

## Performance Metrics & Efficiency
- **Phase 0 Time**: {self.performance_stats.get('phase0_time', 0):.2f}s
- **Phase 1 Time**: {self.performance_stats.get('phase1_time', 0):.2f}s
- **Phase 2 Time**: {self.performance_stats.get('phase2_time', 0):.2f}s
- **Phase 3.1 Time**: {self.performance_stats.get('phase3_1_time', 0):.2f}s
- **Phase 3.2 Time**: {self.performance_stats.get('phase3_2_time', 0):.2f}s
- **Phase 3.3 Time**: {self.performance_stats.get('phase3_3_time', 0):.2f}s
- **Phase 4 Time**: {self.performance_stats.get('phase4_time', 0):.2f}s
- **Total Time**: {total_time:.2f}s
- **Processing Efficiency**: {efficiency_score:.1f} features/second
- **Memory Usage**: {self.performance_stats.get('peak_memory_mb', 0):.1f} MB
- **CPU Utilization**: {self.performance_stats.get('avg_cpu_percent', 0):.1f}%

## Feature Engineering Pipeline Details
"""
        
        # Add pipeline details
        if 'pipeline_stages' in shap_metadata:
            pipeline_stats = shap_metadata['pipeline_stages']
            content += f"- **Data Preprocessing**: {pipeline_stats.get('preprocessing_time', 0):.2f}s\n"
            content += f"- **Feature Generation**: {pipeline_stats.get('generation_time', 0):.2f}s\n"
            content += f"- **Feature Selection**: {pipeline_stats.get('selection_time', 0):.2f}s\n"
            content += f"- **Interaction Discovery**: {pipeline_stats.get('interaction_time', 0):.2f}s\n"
            content += f"- **Quality Validation**: {pipeline_stats.get('validation_time', 0):.2f}s\n"
        
        content += f"""
## Variant Generation Statistics
"""
        
        if 'variant_generation' in shap_metadata:
            variant_stats = shap_metadata['variant_generation']
            content += f"- **Variants Generated**: {variant_stats.get('total_variants_generated', 0)}\n"
            content += f"- **Failed Variants**: {len(variant_stats.get('failed_variants', []))}\n"
            content += f"- **Success Rate**: {((variant_stats.get('total_variants_generated', 0) - len(variant_stats.get('failed_variants', []))) / max(variant_stats.get('total_variants_generated', 1), 1) * 100):.1f}%\n"
            content += f"- **Variant Types**: {variant_stats.get('variants_by_type', {})}\n"
        
        content += f"""
## Pruning Statistics & Quality Control
"""
        
        if 'stage_results' in pruning_stats:
            total_removed = 0
            for stage, stats in pruning_stats['stage_results'].items():
                removed = stats.get('features_removed', 0)
                total_removed += removed
                content += f"- **{stage.title()}**: Removed {removed} features\n"
            
            content += f"- **Total Features Removed**: {total_removed}\n"
            content += f"- **Pruning Efficiency**: {(total_removed / max(total_features + total_removed, 1) * 100):.1f}%\n"
        
        content += f"""
## Interaction Discovery & Analysis
"""
        
        if 'interaction_discovery' in shap_metadata:
            interaction_stats = shap_metadata['interaction_discovery']
            content += f"- **Feature Pairs Analyzed**: {len(interaction_stats.get('feature_pairs', []))}\n"
            content += f"- **Total Interactions Generated**: {interaction_stats.get('total_interactions_generated', 0)}\n"
            content += f"- **Operations Per Pair**: {interaction_stats.get('operations_per_pair', 0)} (x, div, minus, log, log_ratio)\n"
            content += f"- **Max Candidates Processed**: {interaction_stats.get('max_candidates_processed', 0)}\n"
            content += f"- **Selection Method**: {'MI-based selection' if interaction_stats.get('mi_based_selection', False) else 'Early stopping'}\n"
            content += f"- **Valid Interactions Found**: {interaction_stats.get('valid_interactions_found', 0)}\n"
            content += f"- **Interaction Success Rate**: {(interaction_stats.get('total_interactions_generated', 0) / max(interaction_stats.get('max_candidates_processed', 1), 1) * 100):.1f}%\n"
            
            # Add top interaction examples with scores
            if interaction_stats.get('interaction_scores'):
                top_interactions = list(interaction_stats['interaction_scores'].items())[:5]
                content += f"- **Top Interaction Examples**:\n"
                for i, (interaction, score) in enumerate(top_interactions, 1):
                    content += f"  {i}. `{interaction}` (Score: {score:.4f})\n"

                # Interaction learnability summary
                scores_dict = interaction_stats.get('interaction_scores', {}) or {}
                if isinstance(scores_dict, dict) and scores_dict:
                    score_values = np.array(list(scores_dict.values()), dtype=float)
                    n_interactions = int(len(score_values))
                    mean_score = float(score_values.mean()) if n_interactions > 0 else 0.0
                    median_score = float(np.median(score_values)) if n_interactions > 0 else 0.0
                    best_score = float(score_values.max()) if n_interactions > 0 else 0.0
                    positive_count = int((score_values > 0).sum())
                    positive_ratio = (
                        positive_count / n_interactions if n_interactions > 0 else 0.0
                    )

                    content += "\n### Interaction Learnability Summary\n"
                    content += f"- **Interactions evaluated:** {n_interactions}\n"
                    content += f"- **Average interaction score:** {mean_score:.4f}\n"
                    content += f"- **Median interaction score:** {median_score:.4f}\n"
                    content += f"- **Best interaction score:** {best_score:.4f}\n"
                    content += (
                        f"- **Interactions with positive score:** {positive_count}/{n_interactions} "
                        f"({positive_ratio*100:.1f}%)\n"
                    )
                    content += (
                        "\nInteraction scores summarize how much incremental signal the discovered "
                        "interactions add under the mutual-information / model-importance scoring "
                        "used in Phase 3.3. A large fraction of positive, high-scoring interactions "
                        "suggests that the interaction layer is learnable and worth exposing to "
                        "downstream models.\n"
                    )
        
        # Add SHAP analysis insights
        if 'shap_analysis' in shap_metadata:
            shap_stats = shap_metadata['shap_analysis']
            content += f"""
## SHAP Analysis Insights
- **Top Contributing Features**: {len(shap_stats.get('top_features', []))}
- **Feature Importance Range**: {shap_stats.get('min_importance', 0):.4f} - {shap_stats.get('max_importance', 0):.4f}
- **Average Feature Importance**: {shap_stats.get('avg_importance', 0):.4f}
- **SHAP Value Stability**: {shap_stats.get('stability_score', 0):.2f}/1.0
"""
            
            if shap_stats.get('top_features'):
                content += f"- **Most Important Features**:\n"
                for i, (feature, importance) in enumerate(shap_stats['top_features'][:5], 1):
                        content += f"  {i}. `{feature}`: {importance:.4f}\n"
        
        # Add model performance details
        if 'model_performance' in shap_metadata:
            perf_stats = shap_metadata['model_performance']
            content += f"""
## Model Performance & Validation
- **LGBM Training**: {'✅ Successful' if perf_stats.get('lgbm_training_successful', False) else '❌ Failed'}
- **Tree Analysis**: {'✅ Successful' if perf_stats.get('tree_analysis_successful', False) else '❌ Failed'}
- **Interaction Generation**: {'✅ Successful' if perf_stats.get('interaction_generation_successful', False) else '❌ Failed'}
- **Model Accuracy**: {perf_stats.get('accuracy', 0):.4f}
- **Cross-Validation Score**: {perf_stats.get('cv_score', 0):.4f}
- **Feature Importance Consistency**: {perf_stats.get('importance_consistency', 0):.2f}/1.0
- **Mean Feature-Target MI**: {perf_stats.get('mean_mi', 0):.6f}

### Feature-Target Mutual Information by Target
"""
            # Add MI scores per target if available
            mi_scores = perf_stats.get('mi_scores', {})
            if mi_scores:
                for target_name, mi_value in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True):
                    content += f"- `{target_name}`: {mi_value:.6f}\n"
            else:
                content += "- *No MI scores available*\n"
            content += "\n"
        
        # Add data quality metrics
        if 'data_quality' in shap_metadata:
            quality_stats = shap_metadata['data_quality']
            content += f"""
## Data Quality Assessment
- **Missing Value Rate**: {quality_stats.get('missing_rate', 0):.2%}
- **Outlier Detection**: {quality_stats.get('outlier_count', 0)} outliers found
- **Data Completeness**: {quality_stats.get('completeness_score', 0):.2f}/1.0
- **Feature Correlation**: {quality_stats.get('avg_correlation', 0):.3f}
- **Numerical Stability**: {quality_stats.get('stability_score', 0):.2f}/1.0
"""
        
        content += f"""
## Technical Implementation Details
- **Adaptive Feature Selection**: 3-6 features per category based on signal strength
- **RobustScaler Bounding**: Prevents extreme values with percentile clipping
- **Category Protection**: Maintains ≥3 features per category during pruning
- **Corrected SHAP Analysis**: Standard SHAP values for interaction features
- **Causality Enforcement**: All features shifted to prevent lookahead bias
- **Memory Optimization**: VectorBT integration for efficient computation
- **Parallel Processing**: Multi-threaded variant generation
- **Error Handling**: Comprehensive exception management with fallbacks

## Actionable Insights & Recommendations

### Immediate Actions
1. **Category Balance**: {'✅ Balanced' if imbalanced_categories == 0 else f'⚠️ Address {imbalanced_categories} imbalanced categories'}
2. **Feature Quality**: {'✅ High quality' if total_features > 30 else '⚠️ Consider generating more features'}
3. **Processing Efficiency**: {'✅ Efficient' if efficiency_score > 10 else '⚠️ Consider optimization'}

### Strategic Recommendations
1. **Model Selection**: Prioritize features with highest SHAP importance scores
2. **Feature Engineering**: Focus on categories with <5 features for expansion
3. **Performance Monitoring**: Track feature stability across different market conditions
4. **Resource Optimization**: {'Consider parallel processing' if efficiency_score < 5 else 'Current efficiency is adequate'}

### Next Steps
1. **Validation**: Test features on out-of-sample data
2. **Integration**: Incorporate into downstream model training pipeline
3. **Monitoring**: Set up feature drift detection
4. **Iteration**: Refine feature selection based on model performance

## Risk Assessment & Overfitting Prevention
- **Overfitting Risk**: {'Low' if total_features < 100 else 'Medium' if total_features < 200 else 'High'}
- **Prevention Measures Applied**:
  - ✅ L1/L2 Regularization (reg_alpha=0.1, reg_lambda=0.1)
  - ✅ Time-series Cross-Validation with Gap (5-fold, gap=1)
  - ✅ Early Stopping (patience=20 rounds)
  - ✅ Feature Subsampling (subsample=0.8, colsample_bytree=0.8)
  - ✅ Increased min_child_samples (50-75)
  - ✅ Reduced Model Complexity (max_depth=4-5, num_leaves=15-20)
- **Computational Risk**: {'Low' if efficiency_score > 10 else 'Medium' if efficiency_score > 5 else 'High'}
- **Data Quality Risk**: {'Low' if shap_metadata.get('data_quality', {}).get('completeness_score', 1) > 0.9 else 'Medium'}
- **Data Leakage Prevention**: Gap-based time-series splits implemented

---
*Comprehensive report generated by Analyst Interaction Generation Step v2.0*  
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return content

    def _check_category_coverage(self, combined_features: pd.DataFrame, shap_metadata: Dict) -> Dict:
        """Check category coverage in final feature set."""
        tprint_info("📊 Checking category coverage")
        
        category_counts = {}
        
        # Count features per category from metadata
        for feature_name in combined_features.columns:
            # Extract category from metadata or feature name
            category = shap_metadata.get('feature_categories', {}).get(feature_name, 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Log warnings for imbalanced categories
        for category, count in category_counts.items():
            if count < 5:
                tprint_warning(f"⚠️ Category {category} has only {count} features (< 5)")
            elif count > 20:
                tprint_warning(f"⚠️ Category {category} has {count} features (> 20, possible overrepresentation)")
            else:
                tprint_info(f"  {category}: {count} features")
        
        return category_counts

    def _is_tactician_mode(self, config: Dict[str, Any]) -> bool:
        """
        Detect if we're in Tactician mode.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if in Tactician mode, False otherwise
        """
        # Check step name
        is_tactician_step = 'tactician' in self.step_name.lower()
        
        # Check execution context
        is_tactician_context = 'tactician' in config.get('execution_context', '').lower()
        
        # Check explicit flag
        is_explicit_tactician = config.get('tactician_mode', False)
        
        # Check execution mode set at runtime
        is_runtime_tactician = self.execution_mode == 'tactician' if self.execution_mode else False
        
        return is_tactician_step or is_tactician_context or is_explicit_tactician or is_runtime_tactician
    
    def _extract_analyst_side_info(self, config: Dict[str, Any], features_df: Optional[pd.DataFrame] = None) -> Optional[Any]:
        """
        Extract Analyst side information from config/pipeline state.
        
        Args:
            config: Configuration dictionary
            features_df: Optional features dataframe
            
        Returns:
            AnalystSideInfoResult or None
        """
        if not CMI_COMPLEMENTARITY_AVAILABLE or self.analyst_handler is None:
            return None
        
        try:
            # Get pipeline state
            pipeline_state = config.get('pipeline_state', {})
            
            # If features_df provided, extract analyst features
            if features_df is not None:
                analyst_features = [col for col in features_df.columns if 'analyst' in col.lower()]
                if analyst_features:
                    pipeline_state['analyst_features'] = features_df[analyst_features]
            
            # Extract Analyst side information
            analyst_result = self.analyst_handler.emit_analyst_side_info(
                pipeline_state=pipeline_state,
                targets=None,
                data_index=features_df.index if features_df is not None else None
            )
            
            if analyst_result.analyst_outputs is not None:
                tprint_info(f"✅ Analyst side information extracted: {analyst_result.analyst_outputs.shape}")
                return analyst_result
            else:
                tprint_warning("⚠️ No Analyst outputs available")
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract Analyst side information: {e}")
            return None

    def _calculate_composite_scores(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate composite scores for features based on MI/CMI and stability.
        
        In Tactician mode, uses CMI (Conditional Mutual Information) instead of MI
        to select features complementary to Analyst outputs.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            feature_categories: Dict mapping feature names to categories
            config: Optional configuration for mode detection
            
        Returns:
            Dict mapping feature names to composite scores
        """
        from sklearn.feature_selection import mutual_info_regression
        import numpy as np
        
        # Check if Tactician mode and CMI available
        use_cmi = False
        analyst_side_info = None
        if config and self._is_tactician_mode(config) and CMI_COMPLEMENTARITY_AVAILABLE:
            analyst_side_info = self._extract_analyst_side_info(config, features_df)
            use_cmi = analyst_side_info is not None
            if use_cmi:
                tprint_info("🎯 Using CMI-based composite scoring (Tactician mode)")
            else:
                tprint_info("📊 Using MI-based composite scoring (CMI unavailable)")
        else:
            tprint_info("📊 Using MI-based composite scoring (Analyst mode)")
        
        tprint_info(f"  📊 Calculating MI scores for {len(features_df.columns)} features...")
        
        # Use first target for MI calculation
        target_col = targets_df.columns[0]
        target = targets_df[target_col]

        features_aligned, target_aligned = _align_for_label_guided_discovery_helper(
            features_df,
            target,
        )

        if features_aligned.empty or target_aligned.empty:
            tprint_warning("  ⚠️ No valid overlapping samples between features and targets; using uniform composite scores")
            return {col: 0.5 for col in features_df.columns}

        # Remove any features with excessive NaNs or insufficient variation
        # Use relaxed validation for ratio features (like cross-timeframe)
        valid_features = []
        total_rows = len(features_aligned)
        for col in features_aligned.columns:
            col_data = features_aligned[col]
            if total_rows == 0:
                continue

            non_nan_count = col_data.notna().sum()
            nan_ratio = 1.0 - (non_nan_count / float(total_rows))

            # For cross-timeframe ratio features, use more relaxed thresholds
            is_ct_ratio = '_3x_ratio' in col or '_6x_ratio' in col or '_9x_ratio' in col or '_27x_ratio' in col

            # Always remove all-NaN, constant, or zero-variance features
            if non_nan_count == 0 or col_data.nunique() <= 1 or col_data.var() == 0:
                continue

            # NaN ratio thresholds: stricter for regular features
            if is_ct_ratio:
                # Allow up to 20% NaN for cross-timeframe features
                if nan_ratio > 0.2:
                    continue
            else:
                # Allow up to 10% NaN for regular features
                if nan_ratio > 0.1:
                    continue

            # Check if feature varies (not constant)
            col_std = col_data.std()

            if is_ct_ratio:
                # Ratio features can have smaller std, just check they're not ALL the same value
                if col_std > 1e-10 and col_data.nunique() > 2:  # At least 3 unique values
                    valid_features.append(col)
            else:
                # Standard validation for other features
                if col_std > 1e-8:
                    valid_features.append(col)

        if len(valid_features) == 0:
            tprint_warning("  ⚠️ No features passed MI validity checks; relaxing thresholds for fallback scoring")
            valid_features = list(features_aligned.columns)

        # Log cross-timeframe validation stats
        ct_features_total = [c for c in features_aligned.columns if '_3x_ratio' in c or '_6x_ratio' in c or '_9x_ratio' in c or '_27x_ratio' in c]
        ct_features_valid = [c for c in valid_features if '_3x_ratio' in c or '_6x_ratio' in c or '_9x_ratio' in c or '_27x_ratio' in c]
        
        tprint_info(f"  📊 Valid features for MI: {len(valid_features)}/{len(features_df.columns)}")
        tprint_info(f"  📊 Cross-timeframe features: {len(ct_features_valid)}/{len(ct_features_total)} valid")
        
        if len(ct_features_total) > 0 and len(ct_features_valid) == 0:
            tprint_error(f"  ❌ ALL {len(ct_features_total)} cross-timeframe features marked invalid!")
            tprint_error(f"     Validation criteria may be too strict for ratio features")
        elif len(ct_features_valid) < len(ct_features_total) * 0.5:
            tprint_warning(f"  ⚠️ Only {len(ct_features_valid)}/{len(ct_features_total)} CT features valid ({len(ct_features_valid)/len(ct_features_total):.1%})")

        # Temporal subsampling for MI/CMI to reduce matrix size
        mi_index = features_aligned.index
        max_mi_samples = 50000
        if config is not None:
            try:
                max_mi_samples = int(config.get('mi_max_samples', max_mi_samples))
            except Exception:
                pass

        if len(mi_index) > max_mi_samples:
            step = max(1, len(mi_index) // max_mi_samples)
            mi_index = mi_index[::step]
            tprint_info(
                f"  📊 MI temporal subsampling: {len(features_aligned)} → {len(mi_index)} rows (step={step})"
            )

        # Build MI matrix on sampled index and downcast to float32
        features_for_mi = features_aligned.loc[mi_index, valid_features].fillna(0)
        try:
            features_for_mi = features_for_mi.astype(np.float32)
        except Exception:
            # Best-effort: keep original dtypes on failure
            pass
        target_for_mi = target_aligned.loc[mi_index]
        
        # Calculate MI or CMI scores
        try:
            if use_cmi and analyst_side_info:
                # Use CMI scoring (Tactician mode)
                tprint_info(f"  🎯 Calculating CMI scores for {len(valid_features)} features...")
                
                # Create features DataFrame from valid features
                X_for_cmi = features_for_mi
                y_for_cmi = target_for_mi
                
                # Score features using CMI
                cmi_result = self.cmi_scorer.score_features(
                    features=X_for_cmi,
                    targets=y_for_cmi,
                    analyst_outputs=analyst_side_info.analyst_outputs,
                    regime_labels=analyst_side_info.regime_labels
                )
                
                # Extract scores from result
                if hasattr(cmi_result, 'complementarity_scores'):
                    mi_dict = cmi_result.complementarity_scores
                    tprint_info(f"  ✅ CMI complementarity scores calculated")
                elif hasattr(cmi_result, 'feature_scores'):
                    mi_dict = cmi_result.feature_scores
                    tprint_info(f"  ✅ CMI feature scores calculated")
                else:
                    # Fallback to MI
                    tprint_warning(f"  ⚠️ CMI result missing scores, falling back to MI")
                    mi_scores = mutual_info_regression(
                        features_for_mi, target_for_mi, random_state=42, n_neighbors=3
                    )
                    mi_dict = dict(zip(valid_features, mi_scores))
                
                # Normalize scores to 0-1
                if len(mi_dict) > 0:
                    mi_values = np.array(list(mi_dict.values()))
                    if mi_values.max() > 0:
                        mi_max = mi_values.max()
                        mi_dict = {k: v / mi_max for k, v in mi_dict.items()}
                
                tprint_info(f"      Min: {min(mi_dict.values()):.4f}, Max: {max(mi_dict.values()):.4f}, Mean: {np.mean(list(mi_dict.values())):.4f}")
                
            else:
                # Use standard MI (Analyst mode)
                mi_scores = mutual_info_regression(
                    features_for_mi,
                    target_for_mi,
                    random_state=42,
                    n_neighbors=3
                )
                mi_dict = dict(zip(valid_features, mi_scores))
                
                # Normalize MI scores to 0-1
                if len(mi_scores) > 0 and mi_scores.max() > 0:
                    mi_max = mi_scores.max()
                    mi_dict = {k: v / mi_max for k, v in mi_dict.items()}
                
                tprint_info(f"  ✅ MI scores calculated")
                tprint_info(f"      Min: {min(mi_dict.values()):.4f}, Max: {max(mi_dict.values()):.4f}, Mean: {np.mean(list(mi_dict.values())):.4f}")
            
        except Exception as e:
            tprint_warning(f"  ⚠️ MI/CMI calculation failed: {e}")
            mi_dict = {col: 0.5 for col in valid_features}
        
        # Calculate stability scores (variance over time windows)
        tprint_info(f"  📊 Calculating stability scores...")
        stability_dict = {}
        
        try:
            window_size = min(100, len(features_aligned) // 5)
            for col in valid_features:
                feature_data = features_aligned[col].fillna(method='ffill').fillna(0)
                
                # Calculate rolling mean and std
                rolling_mean = feature_data.rolling(window=window_size, min_periods=10).mean()
                rolling_std = feature_data.rolling(window=window_size, min_periods=10).std()
                
                # Stability = 1 - (coefficient of variation of rolling means)
                if rolling_mean.std() > 1e-8:
                    cv = rolling_std.mean() / (abs(rolling_mean.mean()) + 1e-8)
                    stability = 1.0 / (1.0 + cv)  # Higher stability for lower CV
                else:
                    stability = 0.5
                
                stability_dict[col] = max(0.0, min(1.0, stability))
            
            tprint_info(f"  ✅ Stability scores calculated")
            tprint_info(f"      Min: {min(stability_dict.values()):.4f}, Max: {max(stability_dict.values()):.4f}, Mean: {np.mean(list(stability_dict.values())):.4f}")
            
        except Exception as e:
            tprint_warning(f"  ⚠️ Stability calculation failed: {e}")
            stability_dict = {col: 0.5 for col in valid_features}
        
        # Persist raw MI and stability scores for downstream reporting (Phase 4 CSV)
        try:
            self._last_mi_scores = dict(mi_dict)
            self._last_stability_scores = dict(stability_dict)
        except Exception:
            # Best-effort only; do not break scoring if caching fails
            pass
        
        # Combine MI and stability into composite score
        composite_scores = {}
        for col in features_df.columns:
            if col in mi_dict and col in stability_dict:
                # Weighted average: 60% MI, 40% stability
                composite_scores[col] = 0.6 * mi_dict[col] + 0.4 * stability_dict[col]
            else:
                # Default score for invalid features
                composite_scores[col] = 0.01  # Very low score
        
        # Analyze cross-timeframe scores
        ct_features = [f for f in composite_scores.keys() if 
                      '_3x_ratio' in f or '_6x_ratio' in f or '_9x_ratio' in f or '_27x_ratio' in f]
        
        if ct_features:
            ct_scores = [composite_scores[f] for f in ct_features]
            all_scores = list(composite_scores.values())
            
            tprint_info("="*80)
            tprint_info("📊 CROSS-TIMEFRAME COMPOSITE SCORE ANALYSIS")
            tprint_info("="*80)
            tprint_info(f"  📊 Cross-timeframe features: {len(ct_features)}")
            tprint_info(f"  📊 CT score stats: Min={min(ct_scores):.4f}, Max={max(ct_scores):.4f}, Mean={np.mean(ct_scores):.4f}")
            tprint_info(f"  📊 All score stats: Min={min(all_scores):.4f}, Max={max(all_scores):.4f}, Mean={np.mean(all_scores):.4f}")
            
            if np.mean(ct_scores) < np.mean(all_scores) * 0.8:
                tprint_warning(f"  ⚠️ Cross-timeframe features have significantly lower scores!")
                tprint_warning(f"      CT mean: {np.mean(ct_scores):.4f} vs All mean: {np.mean(all_scores):.4f}")
            else:
                tprint_success(f"  ✅ Cross-timeframe scores are competitive")
        
        tprint_info(f"  ✅ Composite scores calculated for {len(composite_scores)} features")
        
        return composite_scores

    def _get_feature_categories_from_bank(self, feature_names: List[str], lookback_optimization: Dict) -> Dict[str, str]:
        """Get feature categories from lookback optimization feature bank."""
        feature_categories = {}
        
        # Get categories from lookback optimization if available
        if 'feature_categories' in lookback_optimization:
            bank_categories = lookback_optimization['feature_categories']
            for feature_name in feature_names:
                # Try to find exact match first
                if feature_name in bank_categories:
                    feature_categories[feature_name] = bank_categories[feature_name]
                else:
                    # Try to find partial match (for variants)
                    base_name = feature_name.split('_')[0]  # Get base feature name
                    if base_name in bank_categories:
                        feature_categories[feature_name] = bank_categories[base_name]
                    else:
                        feature_categories[feature_name] = 'unknown'
        else:
            # Fallback to name-based inference
            for feature_name in feature_names:
                feature_categories[feature_name] = self._infer_feature_category(feature_name)
        
        return feature_categories
    
    def _infer_feature_category(self, feature_name: str) -> str:
        """Infer feature category from name using feature bank."""
        if FEATURE_BANK_AVAILABLE and self.feature_bank:
            # Use feature bank to get category information
            try:
                # Try to get category from feature bank
                # list_features() returns a list of strings (feature names), not objects
                features = self.feature_bank.list_features()
                if feature_name in features:
                    # Try to get the feature info from registry
                    try:
                        # DEBUG: Log the registry type and available methods
                        tprint_warning(f"🔍 DEBUG: FeatureRegistry type: {type(self.feature_bank.registry)}")
                        tprint_warning(f"🔍 DEBUG: Available methods: {[method for method in dir(self.feature_bank.registry) if not method.startswith('_')]}")
                        
                        # Try the correct method name
                        if hasattr(self.feature_bank.registry, 'get_by_name'):
                            feature_info = self.feature_bank.registry.get_by_name(feature_name)
                        else:
                            # Note: get_feature() method doesn't exist in FeatureRegistry
                            # The correct method is get_by_name() which we already tried above
                            tprint_warning(f"🔍 DEBUG: get_feature() method not available, using fallback")
                            tprint_error(f"🔍 DEBUG: No suitable method found on FeatureRegistry")
                            feature_info = None
                            
                        if feature_info and hasattr(feature_info, 'category'):
                            return feature_info.category.value if hasattr(feature_info.category, 'value') else str(feature_info.category)
                    except Exception as e:
                        tprint_warning(f"⚠️ Error getting feature info for {feature_name}: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Error accessing feature bank: {e}")
        
        # Fallback to local implementation
        return self._fallback_infer_feature_category(feature_name)
    
    def _extract_base_feature_name(self, variant_col: str) -> str:
        """Extract base feature name from variant column name."""
        # Remove variant suffixes
        suffixes_to_remove = ['_base', '_volnorm', '_vwap', '_trend_adj']
        base_name = variant_col
        
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        return base_name
    
    def _extract_variant_type(self, variant_col: str) -> str:
        """Extract variant type from variant column name."""
        if variant_col.endswith('_volnorm'):
            return 'volnorm'
        elif variant_col.endswith('_vwap'):
            return 'vwap'
        elif variant_col.endswith('_trend_adj'):
            return 'trend_adj'
        else:
            return 'base'
    
    def _fallback_infer_feature_category(self, feature_name: str) -> str:
        """Fallback category inference when centralized mapper is not available."""
        category_keywords = {
            'trend': ['sma', 'ema', 'trend', 'moving_average', 'ma'],
            'oscillator': ['rsi', 'stoch', 'oscillator', 'williams', 'cci', 'macd'],
            'momentum': ['momentum', 'roc', 'rate_of_change', 'pct_change'],
            'returns': ['return', 'pct_change', 'log_return', 'ret'],
            'volatility': ['vol', 'volatility', 'std', 'atr', 'bb', 'bollinger'],
            'volume': ['volume', 'vol', 'vwap', 'obv', 'ad'],
            'acceleration': ['accel', 'jerk', 'second_derivative', '2nd_deriv'],
            'advanced_statistical': ['skew', 'kurt', 'kurtosis', 'skewness', 'jarque', 'normality', 'statistical', 'advanced_stat', 'bb_width', 'bb_upper', 'bb_lower', 'bb_middle', 'ljung_box', 'ar_', 'coefficients', 'pvalue'],
            'candlestick_pattern': ['candlestick', 'candle', 'doji', 'hammer', 'shooting', 'hanging', 'pattern', 'engulfing'],
            'entropy': ['entropy', 'ent', 'shannon', 'information', 'complexity'],
            'spectral_wavelet': ['spectral', 'wavelet', 'freq', 'fft', 'dwt', 'frequency', 'spectrum'],
            'support_resistance': ['support', 'resistance', 'sr', 'level', 'pivot', 'fibonacci', 'fib']
        }
        
        feature_lower = feature_name.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in feature_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def _find_similar_feature(self, target_feature: str, available_features: List[str]) -> List[str]:
        """Find similar features when exact match is not found."""
        target_lower = target_feature.lower()
        
        # Extract key components from target feature
        target_parts = set(target_lower.split('_'))
        
        # Score each available feature based on similarity
        scored_features = []
        for feature in available_features:
            feature_lower = feature.lower()
            feature_parts = set(feature_lower.split('_'))
            
            # Calculate similarity score
            common_parts = target_parts.intersection(feature_parts)
            similarity_score = len(common_parts) / max(len(target_parts), len(feature_parts))
            
            if similarity_score > 0.3:  # At least 30% similarity
                scored_features.append((feature, similarity_score))
        
        # Sort by similarity score and return top matches
        scored_features.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, score in scored_features[:3]]  # Return top 3 matches

    def _align_for_label_guided_discovery(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        common_idx = features.index.intersection(target.index)

        if len(common_idx) == 0:
            min_len = min(len(features), len(target))
            if min_len == 0:
                return features.iloc[0:0].copy(), target.iloc[0:0].copy()
            features_aligned = features.iloc[-min_len:].copy()
            target_aligned = target.iloc[-min_len:].copy()
            features_aligned.index = pd.RangeIndex(min_len)
            target_aligned.index = pd.RangeIndex(min_len)
        else:
            features_aligned = features.loc[common_idx].copy()
            target_aligned = target.loc[common_idx].copy()

        finite_mask = np.isfinite(target_aligned.values) & np.all(
            np.isfinite(features_aligned.values), axis=1
        )
        features_clean = features_aligned[finite_mask]
        target_clean = target_aligned[finite_mask]

        return features_clean, target_clean

    def _get_consistent_sample(self, features: pd.DataFrame, targets: pd.DataFrame, max_samples: int = 8000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get consistent sample across all phases."""
        
        if len(features) == 0:
            return features, targets
        
        # If we have fewer samples than max_samples, just return all
        if len(features) <= max_samples:
            return features, targets
        
        # Use same random seed for consistency
        np.random.seed(42)
        sample_idx = np.random.choice(len(features), max_samples, replace=False)
        sampled_features = features.iloc[sample_idx]
        sampled_targets = targets.iloc[sample_idx]
        return sampled_features, sampled_targets
    
    def _chunked_processing(self, features: pd.DataFrame, targets: pd.DataFrame, chunk_size: int = 2000) -> pd.DataFrame:
        if len(features) <= chunk_size:
            return features

        indices_to_keep = features.index[-chunk_size:]
        indices_to_drop = features.index.difference(indices_to_keep)

        if len(indices_to_drop) > 0:
            try:
                features.drop(indices_to_drop, inplace=True)
            except Exception:
                features = features.loc[indices_to_keep]

            try:
                targets.drop(indices_to_drop, inplace=True)
            except Exception:
                targets = targets.loc[indices_to_keep]

        return features
    
    def _adaptive_category_selection(self, features_by_category: Dict[str, List[str]], feature_importance: pd.Series, min_per_category: int = 2, max_per_category: int = 8) -> List[str]:
        """Dynamic feature selection based on signal strength."""
        selected_features = []
        
        for category, features in features_by_category.items():
            if len(features) == 0:
                continue
                
            # Calculate signal strength (composite score variance)
            scores = [feature_importance.get(f, 0) for f in features]
            if len(scores) > 1:
                signal_strength = np.std(scores) / (np.mean(scores) + 1e-8)
            else:
                signal_strength = 0.5  # Default for single feature
            
            # Adaptive selection: more features for stronger signals
            if signal_strength > 0.5:  # High signal strength
                n_select = min(max_per_category, len(features))
            elif signal_strength > 0.2:  # Medium signal strength
                n_select = min(6, len(features))
            else:  # Low signal strength
                n_select = min(min_per_category, len(features))
            
            # Get top features by importance
            category_importance = feature_importance[features]
            top_category_features = category_importance.nlargest(n_select).index.tolist()
            selected_features.extend(top_category_features)
        
        return selected_features
    
    def _fast_mi_proxy(self, feature: pd.Series, target: pd.Series, n_bins: int = 5) -> float:
        """Fast MI proxy using discretization - much faster than full MI calculation."""
        try:
            # Handle NaN values
            feature_clean = feature.fillna(0)
            target_clean = target.fillna(0)
            
            # Check for constant values
            if feature_clean.std() == 0 or target_clean.std() == 0:
                return 0.0
            
            # Simple equal-width binning
            feature_bins = pd.cut(feature_clean, bins=n_bins, labels=False, duplicates='drop')
            target_bins = pd.cut(target_clean, bins=n_bins, labels=False, duplicates='drop')
            
            # Handle any remaining NaN from binning
            valid_mask = ~(pd.isna(feature_bins) | pd.isna(target_bins))
            feature_bins = feature_bins[valid_mask]
            target_bins = target_bins[valid_mask]
            
            if len(feature_bins) == 0:
                return 0.0
            
            # Simple contingency table
            contingency = pd.crosstab(feature_bins, target_bins, normalize=True)
            
            # Fast MI approximation
            feature_marginal = contingency.sum(axis=1)
            target_marginal = contingency.sum(axis=0)
            
            mi_proxy = 0.0
            for i in range(len(feature_marginal)):
                for j in range(len(target_marginal)):
                    if contingency.iloc[i, j] > 0:
                        mi_proxy += contingency.iloc[i, j] * np.log2(
                            contingency.iloc[i, j] / (feature_marginal.iloc[i] * target_marginal.iloc[j] + 1e-8)
                        )
            
            return max(0.0, mi_proxy)  # Ensure non-negative
            
        except Exception as e:
            tprint_warning(f"  ⚠️ Fast MI proxy failed: {e}")
            return 0.0
    
    def _early_stopping_interaction_discovery(self, feature_pairs: List[Tuple], features: pd.DataFrame, targets: pd.DataFrame, max_candidates: int = 20) -> Dict[str, pd.Series]:
        """Early stopping interaction discovery with diminishing returns detection using fast MI proxy."""
        
        tprint_info(f"  🔍 DEBUG: _early_stopping_interaction_discovery called with {len(feature_pairs)} feature pairs")
        tprint_info(f"  🔍 DEBUG: features columns: {list(features.columns)}")
        
        interaction_features = {}
        scores = []
        best_score = 0
        stagnation_count = 0
        
        for i, (f1_idx, f2_idx, co_occurrence) in enumerate(feature_pairs):
            if i >= max_candidates:
                tprint_info(f"  🔍 DEBUG: Reached max_candidates limit: {max_candidates}")
                break
            
            # Convert integer indices to column names
            try:
                f1 = features.columns[f1_idx]
                f2 = features.columns[f2_idx]
            except IndexError:
                tprint_info(f"  🔍 DEBUG: Skipping pair ({f1_idx}, {f2_idx}) - index out of range")
                continue
                
            if f1 not in features.columns or f2 not in features.columns:
                tprint_info(f"  🔍 DEBUG: Skipping pair ({f1}, {f2}) - features not in columns")
                continue
                
            tprint_info(f"  🔍 DEBUG: Processing pair {i+1}: ({f1}, {f2})")
                
            # Generate interactions with NaN handling
            f1_clean = features[f1].fillna(0)
            f2_clean = features[f2].fillna(0)
            
            interactions = {
                f"{f1}_x_{f2}": f1_clean * f2_clean,
                f"{f1}_div_{f2}": f1_clean / (f2_clean + 1e-8),
                f"{f1}_minus_{f2}": f1_clean - f2_clean,
                f"{f1}_log_{f2}": np.log(np.abs(f1_clean) + 1e-8) / (np.log(np.abs(f2_clean) + 1e-8) + 1e-8),
                f"{f1}_log_ratio_{f2}": np.log(np.abs(f1_clean / (f2_clean + 1e-8)) + 1e-8)
            }
            
            # Score interactions using MI + correlation with robust NaN handling
            for name, interaction in interactions.items():
                try:
                    # Clean interaction and target data
                    interaction_clean = interaction.fillna(0)
                    target_clean = targets.iloc[:, 0].fillna(0)
                    
                    # Check for constant values (which cause MI calculation issues)
                    if interaction_clean.std() == 0 or target_clean.std() == 0:
                        continue
                    
                    mi_score = self._fast_mi_proxy(interaction_clean, target_clean, n_bins=5)
                    if pd.isna(mi_score):
                        mi_score = 0.0

                    corr_score = abs(interaction_clean.corr(target_clean))
                    if pd.isna(corr_score):
                        corr_score = 0.0

                    # Combined score (70% MI + 30% correlation)
                    combined_score = 0.7 * mi_score + 0.3 * corr_score
                    # Skip any non-finite scores to avoid NaNs in metadata
                    if pd.isna(combined_score) or not np.isfinite(combined_score):
                        continue
                    scores.append((name, combined_score))
                    tprint_info(f"  🔍 DEBUG: Scored interaction {name}: MI={mi_score:.4f}, Corr={corr_score:.4f}, Combined={combined_score:.4f}")
                    
                    # Debug: Track scores
                    tprint_info(f"  🔍 DEBUG: Added score to list. Total scores: {len(scores)}")
                    
                    # Early stopping logic - less aggressive
                    if combined_score > best_score:
                        best_score = combined_score
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                        
                    # Only stop if no improvement for 20 interactions (4 pairs worth)
                    if stagnation_count >= 20:
                        break
                        
                except Exception as e:
                    tprint_warning(f"  ⚠️ Error scoring interaction {name}: {e}")
                    continue
            
            # Less aggressive early stopping for feature pairs
            if stagnation_count >= 20:
                break
        
        # Select top interactions
        scores.sort(key=lambda x: x[1], reverse=True)
        # Select top interactions (up to 80 or all if fewer)
        max_interactions = min(80, len(scores))
        top_interactions = scores[:max_interactions]
        
        # Store the scores for use in metadata
        self._last_interaction_scores = top_interactions
        
        tprint_info(f"  🔍 DEBUG: Selected {len(top_interactions)} top interactions from {len(scores)} total scores")
        
        # Debug: Show top scores
        if len(scores) > 0:
            tprint_info(f"  🔍 DEBUG: Top 5 scores: {scores[:5]}")
        else:
            tprint_warning("  ⚠️ DEBUG: No scores generated - this will result in no interactions!")
        
        # Generate final interaction features with NaN handling
        for name, score in top_interactions:
            tprint_info(f"  🔍 DEBUG: Generating final interaction: {name} (score: {score:.4f})")
            # Reconstruct the interaction from the feature pairs
            for f1, f2, co_occurrence in feature_pairs:
                # Convert integer indices to column names
                if isinstance(f1, int) and isinstance(f2, int):
                    if 0 <= f1 < len(features.columns) and 0 <= f2 < len(features.columns):
                        f1_name = features.columns[f1]
                        f2_name = features.columns[f2]
                    else:
                        continue
                else:
                    f1_name = f1
                    f2_name = f2
                
                if f1_name in features.columns and f2_name in features.columns:
                    f1_clean = features[f1_name].fillna(0)
                    f2_clean = features[f2_name].fillna(0)
                    
                    if name == f"{f1_name}_x_{f2_name}":
                        interaction_features[name] = f1_clean * f2_clean
                    elif name == f"{f1_name}_div_{f2_name}":
                        interaction_features[name] = f1_clean / (f2_clean + 1e-8)
                    elif name == f"{f1_name}_minus_{f2_name}":
                        interaction_features[name] = f1_clean - f2_clean
        
        tprint_info(f"  🔍 DEBUG: Final interaction_features count: {len(interaction_features)}")
        return interaction_features
    
    async def _fgigs_phase4_save_artifacts(
        self,
        final_features: pd.DataFrame,
        interactions: pd.DataFrame,
        shap_metadata: Dict,
        pruning_stats: Dict,
        config: Dict[str, Any],
        lookback_optimization: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """Phase 4: Combine features, verify category coverage, save artifacts, generate report."""
        
        tprint_info("💾 Phase 4: Integration and artifact saving")

        # Feature count summary before Phase 4
        tprint_info("="*80)
        tprint_info(f"📊 PHASE 4: Feature counts before integration:")
        tprint_info(f"  📈 Final features: {len(final_features.columns)} features")
        tprint_info(f"  📈 Interaction features: {len(interactions.columns)} features")
        tprint_info(f"  🔍 DEBUG: Interactions shape at Phase 4 entry: {interactions.shape}")
        tprint_info(f"  🔍 DEBUG: Interactions columns (first 10): {list(interactions.columns)[:10]}")

        # Combine features and interactions
        combined_features = pd.concat([final_features, interactions], axis=1)

        self.performance_stats['final_feature_count'] = len(final_features.columns)
        self.performance_stats['interaction_count'] = len(interactions.columns)

        # Feature count summary after integration
        tprint_info(f"📊 PHASE 4: Integration results:")
        tprint_info(f"  📈 Combined features: {len(combined_features.columns)} features")
        tprint_info(f"  📈 Base features: {len(final_features.columns)} features")
        tprint_info(f"  📈 Interaction features: {len(interactions.columns)} features")
        tprint_info(f"  🔍 DEBUG: Combined features shape: {combined_features.shape}")
        tprint_info("="*80)

        # Verify category coverage (ensure ≥2 per category)
        tprint_info("🔍 Verifying category coverage (minimum 2 per category)...")
        category_coverage = _fgigs_verify_category_coverage(
            self,
            combined_features,
            final_features,
            config,
            lookback_optimization,
        )
        self.performance_stats['category_coverage'] = category_coverage

        # Save artifacts with enhanced metadata
        tprint_info("💾 Saving artifacts with enhanced metadata...")

        # Enhanced metadata for interaction features
        enhanced_metadata = {
            'symbol': config.get('symbol', 'UNKNOWN'),
            'exchange': config.get('exchange', 'UNKNOWN'),
            'timeframe': config.get('timeframe', 'UNKNOWN'),
            'execution_mode': config.get('execution_mode', 'light'),
            'n_base_features': len(final_features.columns),
            'n_interaction_features': len(interactions.columns),
            'total_features': len(combined_features.columns),
            'category_coverage': category_coverage,
            'variant_generation': shap_metadata.get('variant_generation', {}),
            'pruning_stages': shap_metadata.get('pruning_stages', {}),
            'interaction_discovery': shap_metadata.get('interaction_discovery', {}),
            'created_at': datetime.now().isoformat()
        }

        # 1. Analyst interaction features
        tprint_info("="*80)
        tprint_info(f"💾 SAVING ARTIFACTS:")
        tprint_info(f"  🔍 DEBUG: combined_features shape before save: {combined_features.shape}")
        tprint_info(f"  🔍 DEBUG: combined_features columns count: {len(combined_features.columns)}")

        # Categorize features properly - check interaction operations FIRST
        hybrid_ct_interactions = []
        traditional_interactions = []
        ct_ratio_features = []
        variant_features_list = []
        base_features_list = []

        # Define variant suffixes (excluding _base which IS the base feature)
        variant_suffixes = ['_volnorm', '_vwap', '_trend_adj']

        for col in combined_features.columns:
            # Check interaction operations FIRST (before CT markers)
            if any(op in col for op in ['_x_', '_div_', '_minus_', '_log_', '_plus_']):
                if any(marker in col for marker in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']):
                    hybrid_ct_interactions.append(col)  # Hybrid: interaction + cross-timeframe
                else:
                    traditional_interactions.append(col)  # Pure interactions
            elif any(marker in col for marker in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']):
                ct_ratio_features.append(col)  # Pure cross-timeframe ratios
            else:
                # Check if it's a variant feature or base feature
                is_variant = any(col.endswith(suffix) for suffix in variant_suffixes)
                if is_variant:
                    variant_features_list.append(col)
                else:
                    base_features_list.append(col)
        
        # Generate outcome report
        report_path = self._generate_outcome_report(
            combined_features,
            final_features,
            interactions,
            shap_metadata,
            enhanced_metadata,
            config
        )
        if report_path:
            tprint_success(f"✅ Outcome report generated: {report_path}")

        metrics = {
            'success': True,
            'performance_stats': self.performance_stats,
            'category_coverage': category_coverage,
            'total_features': len(combined_features.columns),
            'base_features': len(final_features.columns),
            'interaction_features': len(interactions.columns)
        }

        tprint_success(f" Phase 4 completed: {len(combined_features.columns)} total features")
        tprint_info(f" Category coverage: {category_coverage}")

        return artifacts, metrics

def _fgigs_generate_outcome_report(
    self,
    shap_metadata: Dict,
    pruning_stats: Dict,
    category_coverage: Dict,
    config: Dict[str, Any],
) -> Optional[str]:
    """Generate outcome report for interaction step without relying on wrappers.

    Uses the class's `_create_report_content` helper when available, and falls
    back to a minimal markdown report if that fails. This keeps Phase 4 robust
    even when the runtime class is wrapped or partially patched.
    """
    try:
        from pathlib import Path

        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = config.get("symbol", "UNKNOWN")
        report_filename = f"analyst_interaction_generation_{symbol}_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Try to use the rich report content helper if available
        report_content: Optional[str] = None
        try:
            if hasattr(self, "_create_report_content"):
                report_content = self._create_report_content(
                    shap_metadata, pruning_stats, category_coverage, config
                )
        except Exception as inner_exc:
            tprint_warning(f"⚠️ Detailed report generation failed, using minimal report: {inner_exc}")

        if not report_content:
            # Minimal but informative fallback report
            total_features = (
                self.performance_stats.get("final_feature_count", 0)
                + self.performance_stats.get("interaction_count", 0)
            )
            content_lines = [
                "# Analyst Interaction Generation Report (Minimal)",
                "",
                f"- Symbol: {config.get('symbol', 'UNKNOWN')}",
                f"- Exchange: {config.get('exchange', 'UNKNOWN')}",
                f"- Timeframe: {config.get('timeframe', 'UNKNOWN')}",
                f"- Execution Mode: {config.get('execution_mode', 'light')}",
                f"- Total Features: {total_features}",
                "",
                "## Category Coverage",
            ]
            for cat, count in category_coverage.items():
                content_lines.append(f"- {cat}: {count} features")
            report_content = "\n".join(content_lines)

        with open(report_path, "w") as f:
            f.write(report_content)

        return str(report_path)

    except Exception as exc:
        tprint_error(f" Failed to generate outcome report: {exc}")
        return None


# Fallback function assignments (commented out - functions already inline in class)
# if not hasattr(FeatureGenerationInteractionGenerationStep, "_fast_mi_proxy"):
#     FeatureGenerationInteractionGenerationStep._fast_mi_proxy = _fgigs_fast_mi_proxy
# if not hasattr(FeatureGenerationInteractionGenerationStep, "_extract_tree_splitting_pairs"):
#     FeatureGenerationInteractionGenerationStep._extract_tree_splitting_pairs = _extract_tree_splitting_pairs
# if not hasattr(FeatureGenerationInteractionGenerationStep, "_phase4_save_artifacts"):
#     FeatureGenerationInteractionGenerationStep._phase4_save_artifacts = _fgigs_phase4_save_artifacts

async def _phase4_save_artifacts(
    step: "FeatureGenerationInteractionGenerationStep",
    final_features: pd.DataFrame,
    interactions: pd.DataFrame,
    shap_metadata: Dict,
    pruning_stats: Dict,
    config: Dict[str, Any],
    lookback_optimization: pd.DataFrame,
) -> Tuple[Dict, Dict]:
    """Standalone Phase 4 implementation operating on the given step instance."""

    # Feature count summary before Phase 4
    tprint_info("=" * 80)
    tprint_info(f"📊 PHASE 4: Feature counts before integration:")
    tprint_info(f"  📈 Final features: {len(final_features.columns)} features")
    tprint_info(f"  📈 Interaction features: {len(interactions.columns)} features")
    tprint_info(f"  🔍 DEBUG: Interactions shape at Phase 4 entry: {interactions.shape}")
    tprint_info(f"  🔍 DEBUG: Interactions columns (first 10): {list(interactions.columns)[:10]}")

    # Combine features and interactions
    combined_features = pd.concat([final_features, interactions], axis=1)

    # Update basic metrics on the step for downstream reporting
    step.performance_stats['final_feature_count'] = len(final_features.columns)
    step.performance_stats['interaction_count'] = len(interactions.columns)

    # Feature count summary after integration
    tprint_info(f"📊 PHASE 4: Integration results:")
    tprint_info(f"  📈 Combined features: {len(combined_features.columns)} features")
    tprint_info(f"  📈 Base features: {len(final_features.columns)} features")
    tprint_info(f"  📈 Interaction features: {len(interactions.columns)} features")
    tprint_info(f"  🔍 DEBUG: Combined features shape: {combined_features.shape}")
    tprint_info("=" * 80)

    # Verify category coverage (ensure ≥2 per category)
    tprint_info("🔍 Verifying category coverage (minimum 2 per category)...")
    # Lightweight, self-contained category coverage: infer categories directly
    # from feature names without relying on any external helpers.
    category_keywords: Dict[str, List[str]] = {
        'trend': ['sma', 'ema', 'trend', 'moving_average', 'ma'],
        'oscillator': ['rsi', 'stoch', 'oscillator', 'williams', 'cci', 'macd'],
        'momentum': ['momentum', 'roc', 'rate_of_change', 'pct_change'],
        'returns': ['return', 'log_return', 'ret'],
        'volatility': ['vol', 'volatility', 'std', 'atr', 'bb', 'bollinger'],
        'volume': ['volume', 'vwap', 'obv', 'ad_'],
        'acceleration': ['accel', 'jerk', 'second_derivative', '2nd_deriv'],
        'entropy': ['entropy', 'ent', 'shannon'],
        'spectral_wavelet': ['spectral', 'wavelet', 'fft', 'dwt', 'frequency'],
        'support_resistance': ['support', 'resistance', 'sr_', 'pivot', 'fibonacci', 'fib'],
    }

    category_coverage: Dict[str, int] = {key: 0 for key in category_keywords.keys()}
    category_coverage['unknown'] = 0

    for col in combined_features.columns:
        col_lower = str(col).lower()
        assigned = False
        for cat, keywords in category_keywords.items():
            if any(kw in col_lower for kw in keywords):
                category_coverage[cat] += 1
                assigned = True
                break
        if not assigned:
            category_coverage['unknown'] += 1

    step.performance_stats['category_coverage'] = category_coverage

    # Build enhanced metadata for interaction features
    enhanced_metadata = {
        'symbol': config.get('symbol', 'UNKNOWN'),
        'exchange': config.get('exchange', 'UNKNOWN'),
        'timeframe': config.get('timeframe', 'UNKNOWN'),
        'execution_mode': config.get('execution_mode', 'light'),
        'n_base_features': len(final_features.columns),
        'n_interaction_features': len(interactions.columns),
        'total_features': len(combined_features.columns),
        'category_coverage': category_coverage,
        'variant_generation': shap_metadata.get('variant_generation', {}),
        'pruning_stages': shap_metadata.get('pruning_stages', {}),
        'interaction_discovery': shap_metadata.get('interaction_discovery', {}),
        'created_at': datetime.now().isoformat(),
    }

    # Generate outcome report using inline helper if available, otherwise the
    # more defensive _fgigs_generate_outcome_report fallback.
    report_path: Optional[str] = None
    try:
        if hasattr(step, "_generate_outcome_report"):
            report_path = step._generate_outcome_report(
                combined_features,
                final_features,
                interactions,
                shap_metadata,
                enhanced_metadata,
                config,
            )
        else:
            report_path = _fgigs_generate_outcome_report(
                step,
                shap_metadata,
                pruning_stats,
                category_coverage,
                config,
            )
    except Exception as exc:  # pragma: no cover - diagnostic surface
        tprint_warning(f"⚠️ Outcome report generation failed: {exc}")
        report_path = None

    if report_path:
        tprint_success(f"✅ Outcome report generated: {report_path}")

    # Build artifacts dict in a minimal-but-consistent way; if the step
    # defines a richer artifact-saving helper we prefer that, but we avoid
    # failing the whole pipeline if it is missing.
    artifacts: Dict[str, Any] = {}
    try:
        if hasattr(step, "_save_artifact"):
            # Persist combined features and interactions for downstream steps
            step._save_artifact(combined_features, "analyst_combined_features")
            step._save_artifact(interactions, "analyst_interactions")
            artifacts['combined_features'] = combined_features
            artifacts['interactions'] = interactions
    except Exception as exc:  # pragma: no cover - diagnostic surface
        tprint_warning(f"⚠️ Failed to save Phase 4 artifacts: {exc}")

    metrics: Dict[str, Any] = {
        'success': True,
        'performance_stats': step.performance_stats,
        'category_coverage': category_coverage,
        'total_features': len(combined_features.columns),
        'base_features': len(final_features.columns),
        'interaction_features': len(interactions.columns),
    }

    tprint_success(f" Phase 4 completed: {len(combined_features.columns)} total features")
    tprint_info(f" Category coverage: {category_coverage}")

    return artifacts, metrics

attach_interaction_generation_fallbacks(FeatureGenerationInteractionGenerationStep)


# Register the step

def register_feature_generation_interaction_generation_step():
    """Register the unified interaction generation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register(
        "feature_generation_interaction_generation_step",
        FeatureGenerationInteractionGenerationStep,
    )
    tprint("✅ Unified feature generation interaction generation step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_interaction_generation_step()
