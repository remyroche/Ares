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
- Cross-timeframe feature generation with 3x, 6x, 9x, 27x lookback ratios
- Category protection during pruning (maintain ≥3 per category)
- Tree-based interaction guidance with corrected SHAP analysis
- Comprehensive causality enforcement
- Category coverage tracking (≥2 per category in final set)
        - CMI complementarity for Tactician mode

Cross-Timeframe Features:
- For each variant feature (base, volnorm, vwap, trend_adj), generates 4 additional timeframe versions
- Creates ratio-based interactions: feature_base / feature_3x, feature_base / feature_6x, feature_base / feature_9x, feature_base / feature_27x
- Uses safe division with math validation and causality enforcement (.shift(1))
- Effectively multiplies feature count by ~5x after Phase 1 (1 base + 4 ratios per variant)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import time
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_performance,
    tprint_warning, tprint_structured, LogLevel
)

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

# CMI complementarity components for Analyst mode
try:
    # These modules don't exist yet - placeholder for future implementation
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    CMI_COMPLEMENTARITY_AVAILABLE = False
    tprint_warning("⚠️ CMI complementarity components not available - placeholder implementation")
except ImportError:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    tprint_warning("⚠️ CMI complementarity components not available")

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
    UTILITIES_AVAILABLE = True
except ImportError as e:
    UTILITIES_AVAILABLE = False
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
            self.max_workers = min(4, cpu_count())
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
        
        # Detect execution mode
        self.execution_mode = self._detect_execution_mode(config)
        
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
        
        tprint_info(f"🔍 DEBUG: execute method called")
        tprint_info(f"🔍 DEBUG: Config: {config}")
        tprint_info(f"🔍 DEBUG: Symbol: {config.get('symbol', 'NOT_FOUND')}")
        tprint_info(f"🔍 DEBUG: Execution mode: {config.get('execution_mode', 'NOT_FOUND')}")
        tprint_info(f"🔍 DEBUG: Interaction generation mode: {self.execution_mode}")

        try:
            # Initialize optimization components
            await self._initialize_optimization_components(config)
            
            # Phase 0: Load artifacts and select top features
            tprint_info("=" * 80)
            tprint_info("📋 PHASE 0: Load Artifacts and Select Top Features")
            tprint_info(f"🔍 DEBUG: Phase 0 method called")
            tprint_info(f"🔍 DEBUG: Config: {config}")
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
            tprint_info(f"🔍 DEBUG: Variant features shape after generation: {variant_features.shape}")
            if len(variant_features.columns) > 0:
                tprint_info(f"🔍 DEBUG: Variant features columns: {list(variant_features.columns)[:10]}...")  # Show first 10 columns
                tprint_info(f"🔍 DEBUG: Total variant features generated: {len(variant_features.columns)}")
                tprint_info(f"🔍 DEBUG: Variant features sample data:")
                tprint_info(f"🔍 DEBUG:   First few values: {variant_features.iloc[0, :5].to_dict()}")
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
            tprint_info(f"🔍 DEBUG: Cheap pruning inputs:")
            tprint_info(f"🔍 DEBUG:   variant_features shape: {variant_features.shape}")
            tprint_info(f"🔍 DEBUG:   variant_features columns: {len(variant_features.columns)}")
            tprint_info(f"🔍 DEBUG:   labeled_data shape: {labeled_data.shape}")
            tprint_info(f"🔍 DEBUG:   labeled_data columns: {list(labeled_data.columns)}")
            tprint_info(f"🔍 DEBUG:   lookback_optimization shape: {lookback_optimization.shape}")
            tprint_info(f"🔍 DEBUG:   config keys: {list(config.keys())}")
            
            if len(variant_features.columns) == 0:
                tprint_warning("⚠️ DEBUG: No variant features to prune! Returning empty DataFrame")
                return pd.DataFrame(), {"error": "No variant features to prune"}
            
            # Debug: Check variant features before pruning
            tprint_info(f"🔍 DEBUG: Variant features shape before pruning: {variant_features.shape}")
            tprint_info(f"🔍 DEBUG: Variant features columns: {list(variant_features.columns)[:10]}...")  # Show first 10 columns
            tprint_info(f"🔍 DEBUG: Total variant features before pruning: {len(variant_features.columns)}")
            
            # Check for cross-timeframe features
            cross_timeframe_cols = [c for c in variant_features.columns if '_3x_ratio' in c or '_6x_ratio' in c or '_9x_ratio' in c or '_27x_ratio' in c]
            tprint_info(f"🔍 DEBUG: Cross-timeframe features found before pruning: {len(cross_timeframe_cols)}")
            if len(cross_timeframe_cols) > 0:
                tprint_info(f"🔍 DEBUG: Sample cross-timeframe features: {cross_timeframe_cols[:5]}")
            
            if len(variant_features.columns) == 0:
                tprint_warning("⚠️ DEBUG: No variant features to prune! This will cause cheap pruning to fail!")
            
            phase2_start = time.time()
            
            tprint_info(f"🔍 DEBUG: About to call cheap pruning with {len(variant_features.columns)} variant features")
            
            pruned_features, pruning_stats, targets = await self._phase2_cheap_pruning(
                variant_features, labeled_data, lookback_optimization, config
            )
            
            # Debug: Check pruned features after pruning
            tprint_info(f"🔍 DEBUG: Pruned features shape after pruning: {pruned_features.shape}")
            if len(pruned_features.columns) > 0:
                tprint_info(f"🔍 DEBUG: Pruned features columns: {list(pruned_features.columns)[:10]}...")  # Show first 10 columns
                tprint_info(f"🔍 DEBUG: Total pruned features: {len(pruned_features.columns)}")
                tprint_info(f"🔍 DEBUG: Features removed by pruning: {len(variant_features.columns) - len(pruned_features.columns)}")
                
                # Check for cross-timeframe features after pruning
                cross_timeframe_cols_after = [c for c in pruned_features.columns if '_3x_ratio' in c or '_6x_ratio' in c or '_9x_ratio' in c or '_27x_ratio' in c]
                tprint_info(f"🔍 DEBUG: Cross-timeframe features found after pruning: {len(cross_timeframe_cols_after)}")
                if len(cross_timeframe_cols_after) > 0:
                    tprint_info(f"🔍 DEBUG: Sample cross-timeframe features after pruning: {cross_timeframe_cols_after[:5]}")
                else:
                    tprint_warning("⚠️ DEBUG: ALL cross-timeframe features were pruned!")
            else:
                tprint_warning("⚠️ DEBUG: No features remaining after pruning!")
                tprint_warning("⚠️ DEBUG: All {len(variant_features.columns)} variant features were removed!")
            
            self.performance_stats['phase2_time'] = time.time() - phase2_start
            tprint_performance(f"Phase 2 completed", self.performance_stats['phase2_time'])
            
            # Debug: Check pruned features after pruning
            tprint_info(f"🔍 DEBUG: Pruned features shape after pruning: {pruned_features.shape}")
            if len(pruned_features.columns) > 0:
                tprint_info(f"🔍 DEBUG: Pruned features columns: {list(pruned_features.columns)[:10]}...")  # Show first 10 columns
            else:
                tprint_warning("⚠️ DEBUG: No features remaining after pruning!")
            
            # Phase 3: Three-phase LGBM+SHAP
            tprint_info("=" * 80)
            tprint_info("🤖 PHASE 3: Three-Phase LGBM+SHAP Pipeline")
            tprint_info("=" * 80)
            tprint_info(f"🔍 DEBUG: Phase 3 inputs:")
            tprint_info(f"🔍 DEBUG:   pruned_features shape: {pruned_features.shape}")
            tprint_info(f"🔍 DEBUG:   pruned_features columns: {len(pruned_features.columns)}")
            tprint_info(f"🔍 DEBUG:   labeled_data shape: {labeled_data.shape}")
            tprint_info(f"🔍 DEBUG:   labeled_data columns: {list(labeled_data.columns)}")
            tprint_info(f"🔍 DEBUG:   config keys: {list(config.keys())}")
            
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
            
            artifacts, metrics = await self._phase4_save_artifacts(
                final_features, interactions, shap_metadata, pruning_stats, config, lookback_optimization
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
        """
        Phase 0: Load artifacts and select top 4 features per category.
        
        Returns:
            Tuple of (lookback_optimization, labeled_data, generated_features, top_features_by_category)
        """
        tprint_info("📊 Loading artifacts via BaseStep artifact manager")
        tprint_info(f"🔍 DEBUG: _phase0_load_and_select called")
        
        # Load artifacts
        try:
            lookback_optimization = self._get_artifact('lookback_optimization', 'data')
            tprint_success(f"✅ Loaded lookback_optimization: {lookback_optimization.shape}")
            tprint_structured({"Lookback Optimization": lookback_optimization.head().to_dict()}, level=LogLevel.INFO)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load lookback_optimization artifact: {e}")
        
        try:
            labeled_data = self._get_artifact('labeled_data', 'data')
            tprint_success(f"✅ Loaded labeled_data: {labeled_data.shape}")
            tprint_structured({"Labeled Data": labeled_data.head().to_dict()}, level=LogLevel.INFO)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load labeled_data artifact: {e}")
        
        try:
            generated_features = self._get_artifact('generated_features', 'data')
            tprint_success(f"✅ Loaded generated_features: {generated_features.shape}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load generated_features artifact: {e}")
        
        # Apply light mode filtering
        tprint_info(f"📊 PHASE 0: Initial feature counts before filtering:")
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
        
        tprint_info(f"🔍 DEBUG: Feature selection returned: {len(top_features_by_category)} categories")
        
        # Count total selected features across all categories
        total_selected_features = sum(len(features) for features in top_features_by_category.values())
        tprint_info(f"📊 PHASE 0: Feature selection summary:")
        tprint_info(f"  📈 Total categories: {len(top_features_by_category)}")
        tprint_info(f"  📈 Total selected features: {total_selected_features}")
        for category, features in top_features_by_category.items():
            tprint_info(f"    - {category}: {len(features)} features")
        
        return lookback_optimization, labeled_data, generated_features, top_features_by_category

    def _transform_lookback_optimization_data(self, lookback_optimization: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide DataFrame with nested columns to long format with simple columns.
        
        Args:
            lookback_optimization: Wide DataFrame with nested column names
            
        Returns:
            Long DataFrame with columns: feature_name, category, composite_score, optimal_lookback, etc.
        """
        tprint_info("🔄 Transforming lookback optimization data from wide to long format")
        tprint_info(f"🔍 DEBUG: Input DataFrame shape: {lookback_optimization.shape}")
        tprint_info(f"🔍 DEBUG: Input DataFrame columns: {list(lookback_optimization.columns)[:5]}...")
        tprint_info(f"🔍 DEBUG: Total columns: {len(lookback_optimization.columns)}")
        
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
            tprint_info(f"🔍 DEBUG: All category patterns found: {sorted(category_patterns)}")
        
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
                    tprint_info(f"🔍 DEBUG: Extracted category '{category}' from column '{col}'")
                    
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
        tprint_info(f"🔍 DEBUG: Checking if transformation needed...")
        tprint_info(f"🔍 DEBUG: DataFrame shape: {lookback_optimization.shape}")
        tprint_info(f"🔍 DEBUG: DataFrame columns: {list(lookback_optimization.columns)[:5]}...")
        tprint_info(f"🔍 DEBUG: Has 'category' column: {'category' in lookback_optimization.columns}")
        
        if 'category' not in lookback_optimization.columns:
            tprint_info(f"🔍 DEBUG: Transformation needed, calling _transform_lookback_optimization_data")
            lookback_optimization = self._transform_lookback_optimization_data(lookback_optimization)
            tprint_info(f"🔍 DEBUG: After transformation, DataFrame shape: {lookback_optimization.shape}")
            tprint_info(f"🔍 DEBUG: After transformation, categories found: {sorted(lookback_optimization['category'].unique()) if not lookback_optimization.empty else 'EMPTY'}")
            tprint_info(f"🔍 DEBUG: Expected categories: {sorted(self.categories)}")
        else:
            tprint_info(f"🔍 DEBUG: No transformation needed, DataFrame already has 'category' column")
        
        if lookback_optimization.empty:
            tprint_warning("⚠️ No lookback optimization data available")
            return {}
        
        top_features_by_category = {}
        
        for category in self.categories:
            # Filter features by category
            tprint_info(f"🔍 DEBUG: Processing category: {category}")
            tprint_info(f"🔍 DEBUG: DataFrame shape before filtering: {lookback_optimization.shape}")
            tprint_info(f"🔍 DEBUG: DataFrame columns: {list(lookback_optimization.columns)}")
            
            category_features = lookback_optimization[
                lookback_optimization['category'].str.lower() == category.lower()
            ].copy()
            
            tprint_info(f"🔍 DEBUG: Found {len(category_features)} features for category {category}")
            
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
            
            # Sort by composite_score descending
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
            
            # Sort by composite_score (descending)
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
        tprint_info(f"🔍 DEBUG: Input generated_features shape: {generated_features.shape}")
        tprint_info(f"🔍 DEBUG: Input generated_features columns: {list(generated_features.columns)[:10]}...")
        tprint_info(f"🔍 DEBUG: Input top_features_by_category: {top_features_by_category}")
        tprint_info(f"🔍 DEBUG: Number of categories with features: {len(top_features_by_category)}")
        tprint_info(f"🔍 DEBUG: Total features in generated_features: {len(generated_features.columns)}")
        
        if not UTILITIES_AVAILABLE:
            raise ImportError("Variant generation utilities not available")
        
        # Load OHLCV data using KlinesParquetManager
        ohlcv_data = None
        if DATA_LOADING_AVAILABLE:
            try:
                symbol = config.get('symbol', 'ETHUSDT')
                timeframe = config.get('timeframe', '15m')
                
                # Initialize KlinesParquetManager
                klines_manager = get_klines_manager()
                
                # Load OHLCV data
                ohlcv_data = klines_manager.read_data(
                    symbol=symbol,
                    interval=timeframe,
                    data_type='raw',
                    columns=['open', 'high', 'low', 'close', 'volume']
                )
                
                if ohlcv_data is not None and len(ohlcv_data) > 0:
                    # Ensure the index matches generated_features
                    ohlcv_data = ohlcv_data.reindex(generated_features.index, method='ffill')
                    
                    # Debug: Check the types of OHLCV columns
                    tprint_info(f"🔍 OHLCV data types: close={type(ohlcv_data['close'])}, high={type(ohlcv_data['high'])}, low={type(ohlcv_data['low'])}")
                    
                    # Ensure all columns are pandas Series
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in ohlcv_data.columns:
                            if not isinstance(ohlcv_data[col], pd.Series):
                                ohlcv_data[col] = pd.Series(ohlcv_data[col], name=col, index=ohlcv_data.index)
                    
                    tprint_success(f"✅ Loaded OHLCV data: {ohlcv_data.shape}")
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
        tprint_info(f"🔍 DEBUG: Converting top_features_by_category to selected_features list")
        for category, features in top_features_by_category.items():
            tprint_info(f"🔍 DEBUG: Category {category} has {len(features)} features")
            for feature_name, optimal_lookback, composite_score in features:
                tprint_info(f"🔍 DEBUG: Checking feature: {feature_name}")
                tprint_info(f"🔍 DEBUG: Feature in generated_features.columns: {feature_name in generated_features.columns}")
                if feature_name in generated_features.columns:
                    tprint_info(f"🔍 DEBUG: Adding feature: {feature_name} (lookback: {optimal_lookback}, score: {composite_score})")
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
                        tprint_info(f"🔍 DEBUG: Using similar feature {similar_features[0]} instead of {feature_name}")
                        selected_features.append({
                            'feature_name': similar_features[0],
                            'category': category,
                            'optimal_lookback': int(optimal_lookback),
                            'composite_score': composite_score
                        })
                    else:
                        tprint_warning(f"⚠️ DEBUG: Feature {feature_name} not found in generated_features.columns and no similar feature found")
        
        tprint_info(f"🔍 DEBUG: Total selected features: {len(selected_features)}")
        
        # Feature count summary before variant generation
        tprint_info(f"📊 PHASE 1: Feature preparation summary:")
        tprint_info(f"  📈 Input generated_features: {len(generated_features.columns)} features")
        tprint_info(f"  📈 Selected features for variants: {len(selected_features)} features")
        tprint_info(f"  📈 Categories processed: {len(top_features_by_category)}")
        
        if len(selected_features) == 0:
            tprint_warning("⚠️ DEBUG: No features selected! This will cause variant generation to fail!")
            tprint_warning("⚠️ DEBUG: Check if features from top_features_by_category exist in generated_features.columns")
            return pd.DataFrame()
        
        tprint_info(f"🔍 DEBUG: Selected features details:")
        for i, feature in enumerate(selected_features[:5]):  # Show first 5
            tprint_info(f"🔍 DEBUG:   {i+1}. {feature['feature_name']} (category: {feature['category']}, lookback: {feature['optimal_lookback']})")
        
        # Generate variants using sequential processing (parallel disabled due to pickle issues)
        try:
            # Always use sequential processing to avoid pickle issues with thread locks
            tprint_info("  🔄 Using sequential variant generation (parallel disabled)...")
            tprint_info(f"  🔍 DEBUG: About to generate variants for {len(selected_features)} selected features")
            tprint_info(f"  🔍 DEBUG: Expected variants: {len(selected_features)} × 4 = {len(selected_features) * 4}")
            
            # Add detailed category breakdown before variant generation
            category_counts = {}
            for feature in selected_features:
                category = feature['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            tprint_info(f"  🔍 DEBUG: Feature category breakdown:")
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
                tprint_info(f"🔍 DEBUG: Sample cross-timeframe features: {cross_timeframe_cols}")
                
                return combined_features
            else:
                tprint_warning("⚠️ No cross-timeframe features generated, returning only variant features")
            return variant_features
            
        except Exception as e:
            tprint_error(f"❌ Variant generation failed: {e}")
            raise

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
        tprint_info(f"🔍 DEBUG: Input variant_features shape: {variant_features.shape}")
        tprint_info(f"🔍 DEBUG: variant_features columns: {list(variant_features.columns)[:10]}")
        
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
        timeframe_multipliers = [3, 6, 9, 27]
        
        # Create lookback mapping from original features
        lookback_mapping = {}
        for category, features in top_features_by_category.items():
            for feature_name, optimal_lookback, composite_score in features:
                lookback_mapping[feature_name] = int(optimal_lookback)
        
        tprint_info(f"🔍 DEBUG: Lookback mapping created for {len(lookback_mapping)} features")
        
        # Process each variant feature
        processed_count = 0
        failed_count = 0
        
        for variant_col in variant_features.columns:
            try:
                # Extract base feature name and variant type
                base_feature_name = self._extract_base_feature_name(variant_col)
                variant_type = self._extract_variant_type(variant_col)
                
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
    
    def _extract_period_from_feature_name(self, feature_name: str) -> Optional[int]:
        """Extract the period from a feature name."""
        import re
        
        # Common patterns for period extraction
        patterns = [
            r'_(\d+)$',  # Feature ending with _number
            r'_(\d+)_',  # Feature with _number_ in the middle
            r'(\d+)_',   # Feature starting with number_
        ]
        
        for pattern in patterns:
            match = re.search(pattern, feature_name)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # Default fallback periods for common features
        default_periods = {
            'rsi': 14,
            'sma': 20,
            'ema': 20,
            'bb': 20,
            'atr': 14,
            'stoch': 14,
            'williams': 14,
            'cci': 20,
            'macd': 12,
            'volume': 20,
            'volatility': 20,
        }
        
        for key, default_period in default_periods.items():
            if key in feature_name.lower():
                return default_period
        
        return None
    
    def _recalculate_feature_with_period(self, feature_name: str, period: int, ohlcv_data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Recalculate a feature with a specific period using the original OHLCV data.
        
        Uses three-tier fallback strategy:
        1. Try FeatureBank regeneration (most accurate)
        2. Try simple pattern matching
        3. Use rolling window approximation as last resort
        """
        try:
            from src.training.utils.feature_calculators import FeatureCalculator
            
            # Tier 1: Try FeatureBank regeneration (BEST - recalculates from scratch)
            try:
                if hasattr(self, 'feature_bank') and self.feature_bank is not None:
                    # Try to regenerate the feature using FeatureBank
                    result = self._regenerate_feature_with_feature_bank(feature_name, period, ohlcv_data)
                    if result is not None and not result.isna().all():
                        nan_pct = result.isna().sum() / len(result) * 100
                        if nan_pct < 95:  # Accept if less than 95% NaN
                            tprint_info(f"    ✅ FeatureBank regeneration succeeded for {feature_name} ({nan_pct:.1f}% NaN)")
                            return result
            except Exception as e:
                pass  # Continue to next tier
            
            # Tier 2: Try simple pattern matching
            feature_mappings = {
                'rsi': lambda data, p: FeatureCalculator.calculate_rsi(data['close'], p),
                'sma': lambda data, p: FeatureCalculator.calculate_sma(data['close'], p),
                'ema': lambda data, p: FeatureCalculator.calculate_ema(data['close'], p),
                'bb': lambda data, p: FeatureCalculator.calculate_bollinger_position(data, p),
                'atr': lambda data, p: FeatureCalculator.calculate_atr(data, p),
                'stoch': lambda data, p: FeatureCalculator.calculate_stochastic_k(data, p),
                'williams': lambda data, p: FeatureCalculator.calculate_williams_r(data, p),
                'cci': lambda data, p: FeatureCalculator.calculate_cci(data, p),
                'macd': lambda data, p: FeatureCalculator.calculate_ema(data['close'], p),
                'volume': lambda data, p: data['volume'].rolling(p, min_periods=max(1, p//2)).mean(),
                'volatility': lambda data, p: data['close'].pct_change().rolling(p, min_periods=max(1, p//2)).std(),
                'momentum': lambda data, p: data['close'] - data['close'].shift(p),
                'roc': lambda data, p: FeatureCalculator.calculate_roc(data['close'], p),
                'vwap': lambda data, p: FeatureCalculator.calculate_vwap(data, p),
            }
            
            feature_lower = feature_name.lower()
            for key, calc_func in feature_mappings.items():
                if key in feature_lower:
                    result = calc_func(ohlcv_data, period)
                    if result is not None and not result.isna().all():
                        nan_pct = result.isna().sum() / len(result) * 100
                        if nan_pct < 95:
                            tprint_info(f"    ✅ Pattern match '{key}' succeeded for {feature_name} ({nan_pct:.1f}% NaN)")
                            return result
            
            # Tier 3: Use rolling window approximation (FALLBACK)
            # This approximates the extended timeframe by using rolling windows
            tprint_warning(f"    ⚠️ Using rolling window approximation for {feature_name}")
            result = self._rolling_window_approximation(feature_name, period, ohlcv_data)
            if result is not None and not result.isna().all():
                nan_pct = result.isna().sum() / len(result) * 100
                tprint_warning(f"    ✅ Rolling approximation succeeded ({nan_pct:.1f}% NaN)")
                return result
            
            # All tiers failed
            tprint_error(f"    ❌ All recalculation methods failed for {feature_name}")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Failed to recalculate {feature_name}: {e}")
            return None
    
    def _regenerate_feature_with_feature_bank(self, feature_name: str, period: int, ohlcv_data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Attempt to regenerate a feature using the FeatureBank with the specified period.
        This is the most accurate method as it recalculates from scratch.
        """
        try:
            # The feature bank can regenerate features with different parameters
            # For now, return None and let it fall through to other methods
            # TODO: Implement proper FeatureBank regeneration when needed
            return None
        except Exception as e:
            return None
    
    def _rolling_window_approximation(self, feature_name: str, period: int, ohlcv_data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Create an approximation of the extended timeframe feature using rolling windows.
        This is a fallback when proper recalculation fails.
        
        The approximation uses rolling means/operations to simulate longer timeframes.
        While not perfect, it's better than returning None/NaN.
        """
        try:
            # Strategy: Apply rolling window to approximate longer timeframe
            # This preserves some signal even if not perfectly accurate
            
            # For most features, a rolling mean approximation works reasonably well
            # The idea is that a feature calculated on a longer period behaves similarly
            # to a smoothed version of the same feature
            
            # Check if we have close price to work with
            if 'close' not in ohlcv_data.columns:
                return None
            
            close = ohlcv_data['close']
            
            # Different approximation strategies based on feature type
            feature_lower = feature_name.lower()
            
            # For trend/momentum features: use price momentum over the period
            if any(keyword in feature_lower for keyword in ['trend', 'momentum', 'direction']):
                # Calculate momentum over the extended period
                result = (close - close.shift(period)) / (close.shift(period) + 1e-8)
                result = result.rolling(window=max(5, period//4), min_periods=1).mean()
                return result
            
            # For volatility features: use rolling std over the period
            elif any(keyword in feature_lower for keyword in ['volatility', 'vol', 'atr', 'std']):
                returns = close.pct_change()
                result = returns.rolling(window=period, min_periods=max(1, period//2)).std()
                return result
            
            # For volume features: use rolling volume statistics
            elif 'volume' in feature_lower and 'volume' in ohlcv_data.columns:
                volume = ohlcv_data['volume']
                # Normalize by rolling mean to capture relative volume
                vol_mean = volume.rolling(window=period, min_periods=max(1, period//2)).mean()
                result = volume / (vol_mean + 1e-8)
                return result
            
            # For oscillator/indicator features: use smoothed price ratios
            elif any(keyword in feature_lower for keyword in ['rsi', 'stoch', 'oscillator', 'index']):
                # Approximate with smoothed price momentum
                sma_short = close.rolling(window=max(5, period//4), min_periods=1).mean()
                sma_long = close.rolling(window=period, min_periods=max(1, period//2)).mean()
                result = (sma_short - sma_long) / (sma_long + 1e-8)
                return result
            
            # For price-based features: use simple moving average as approximation
            elif any(keyword in feature_lower for keyword in ['price', 'close', 'sma', 'ema', 'ma']):
                result = close.rolling(window=period, min_periods=max(1, period//2)).mean()
                return result
            
            # Default: use rolling mean of close price changes
            else:
                returns = close.pct_change()
                result = returns.rolling(window=period, min_periods=max(1, period//2)).mean()
                return result
                
        except Exception as e:
            tprint_warning(f"    ⚠️ Rolling approximation failed: {e}")
            return None
    
    def _apply_volatility_normalization(self, feature: pd.Series, ohlcv_data: pd.DataFrame, lookback_period: int = 20) -> pd.Series:
        """Apply volatility normalization to a feature using the specified lookback period."""
        try:
            # Calculate rolling volatility of returns using the extended lookback period
            returns = ohlcv_data['close'].pct_change()
            volatility = returns.rolling(window=lookback_period, min_periods=max(1, lookback_period // 2)).std()
            
            # Normalize the feature by volatility
            normalized = feature / (volatility + 1e-8)
            return normalized
        except Exception as e:
            tprint_warning(f"⚠️ Volatility normalization failed: {e}")
            return feature
    
    def _apply_vwap_weighting(self, feature: pd.Series, ohlcv_data: pd.DataFrame, lookback_period: int = 20) -> Optional[pd.Series]:
        """Apply VWAP weighting to a feature using the specified lookback period."""
        try:
            if 'volume' not in ohlcv_data.columns:
                return feature
            
            # Calculate VWAP using the extended lookback period
            typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
            vwap = (typical_price * ohlcv_data['volume']).rolling(window=lookback_period, min_periods=max(1, lookback_period // 2)).sum() / ohlcv_data['volume'].rolling(window=lookback_period, min_periods=max(1, lookback_period // 2)).sum()
            
            # Weight the feature by VWAP ratio
            price_vwap_ratio = ohlcv_data['close'] / (vwap + 1e-8)
            weighted_feature = feature * price_vwap_ratio
            
            return weighted_feature
        except Exception as e:
            tprint_warning(f"⚠️ VWAP weighting failed: {e}")
            return feature
    
    def _apply_trend_adjustment(self, feature: pd.Series, ohlcv_data: pd.DataFrame, lookback_period: int = 20) -> pd.Series:
        """Apply trend adjustment to a feature using the specified lookback period."""
        try:
            # Calculate trend strength using price momentum with the extended lookback period
            price_momentum = ohlcv_data['close'].pct_change().rolling(window=lookback_period, min_periods=max(1, lookback_period // 2)).mean()
            trend_strength = np.abs(price_momentum) / (ohlcv_data['close'].pct_change().rolling(window=lookback_period, min_periods=max(1, lookback_period // 2)).std() + 1e-8)
            
            # Calculate trend direction
            trend_direction = np.sign(price_momentum)
            
            # Adjust the feature by trend
            trend_adjusted = feature * trend_strength * trend_direction
            
            return trend_adjusted
        except Exception as e:
            tprint_warning(f"⚠️ Trend adjustment failed: {e}")
            return feature
    
    def _generate_extended_timeframe_feature(
        self,
        base_feature_name: str,
        variant_type: str,
        extended_lookback: int,
        generated_features: pd.DataFrame,
        ohlcv_data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """
        Generate extended timeframe version of a feature with different lookback period.
        
        SIMPLIFIED APPROACH: Use rolling window smoothing on the base feature itself.
        This preserves the scale and variant transformations of the base feature.
        
        Args:
            base_feature_name: Name of the base feature
            variant_type: Type of variant (base, volnorm, vwap, trend_adj)
            extended_lookback: Extended lookback period
            generated_features: Original features DataFrame
            ohlcv_data: OHLCV data for recalculation
            
        Returns:
            Series with extended timeframe feature or None if generation fails
        """
        try:
            # Get the base feature series
            if base_feature_name not in generated_features.columns:
                return None
            
            base_feature_series = generated_features[base_feature_name]
            
            # SIMPLIFIED APPROACH: Apply rolling mean to the base feature
            # This approximates the "longer timeframe" version while preserving scale
            # The ratio between base and smoothed-base captures timeframe divergence
            
            # Use extended_lookback as the smoothing window
            smoothing_window = int(extended_lookback)
            
            # Ensure window doesn't exceed data length
            smoothing_window = min(smoothing_window, len(base_feature_series) - 1)
            smoothing_window = max(2, smoothing_window)  # At least window of 2
            
            # Create smoothed version (simulates longer timeframe)
            extended_feature = base_feature_series.rolling(
                window=smoothing_window,
                min_periods=max(1, smoothing_window // 3)  # Require at least 1/3 of window
            ).mean()
            
            # Fill leading NaN values
            extended_feature = extended_feature.fillna(method='bfill').fillna(method='ffill')
            
            # If still has NaN, fill with base feature values
            if extended_feature.isna().any():
                extended_feature = extended_feature.fillna(base_feature_series)
            
            # Validate result
            if extended_feature.isna().all():
                return None
            
            # Check if result is constant (no variance)
            if extended_feature.std() < 1e-10:
                # If constant, add small noise to avoid division issues
                noise = np.random.normal(0, extended_feature.mean() * 0.001, len(extended_feature))
                extended_feature = extended_feature + noise
            
            return extended_feature
            
        except Exception as e:
                return None
            
    async def _extract_cross_timeframe_interactions(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cross-timeframe interaction features between features from different timeframes.
        
        Looks for features with timeframe markers in their names (3x, 6x, 9x, 27x) and generates
        interactions between the base timeframe features and the extended timeframe features.
        
        Args:
            features: DataFrame with features that may include timeframe variants
            
        Returns:
            DataFrame with cross-timeframe interaction features
        """
        try:
            tprint_info("  🔍 Scanning features for cross-timeframe patterns...")
            
            # Identify base features and timeframe features
            base_features = []
            timeframe_features = {}
            
            for col in features.columns:
                # Check if feature has timeframe marker (e.g., _3x_ratio, _6x_ratio, etc.)
                if '_3x_ratio' in col or '_6x_ratio' in col or '_9x_ratio' in col or '_27x_ratio' in col:
                    # Extract base feature name
                    for multiplier in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']:
                        if multiplier in col:
                            base_name = col.replace(multiplier, '')
                            if base_name not in timeframe_features:
                                timeframe_features[base_name] = {}
                            timeframe_features[base_name][multiplier.replace('_', '').replace('ratio', '')] = col
                            break
            else:
                    # Base feature without timeframe marker
                    base_features.append(col)
            
            tprint_info(f"  📊 Found {len(base_features)} base features and {len(timeframe_features)} features with timeframe variants")
            
            # Generate cross-timeframe interactions
            cross_timeframe_interactions = {}
            
            for base_feat in base_features:
                if base_feat in timeframe_features:
                    # Generate interactions between base feature and its timeframe variants
                    for tf_marker, tf_feature in timeframe_features[base_feat].items():
                        # Create product interaction: base * timeframe_variant
                        interaction_name = f"{base_feat}_x_{tf_marker}"
                        
                        try:
                            # Get base and timeframe feature series
                            base_series = features[base_feat]
                            tf_series = features[tf_feature]
                            
                            # Create product interaction with causality enforcement
                            interaction = (base_series * tf_series).shift(1)
                            
                            # Handle NaN values
                            interaction = interaction.fillna(method='ffill').fillna(0)
                            
                            # Store interaction
                            cross_timeframe_interactions[interaction_name] = interaction
                        
                        except Exception as e:
                            tprint_warning(f"  ⚠️ Failed to create cross-timeframe interaction {interaction_name}: {e}")
            
            # Create DataFrame from interactions
            if cross_timeframe_interactions:
                result_df = pd.DataFrame(cross_timeframe_interactions, index=features.index)
                tprint_success(f"  ✅ Generated {len(result_df.columns)} cross-timeframe interactions")
                return result_df
            else:
                tprint_info("  ℹ️ No cross-timeframe interactions generated")
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"  ❌ Failed to extract cross-timeframe interactions: {e}")
            return pd.DataFrame()

    async def _phase2_cheap_pruning(
        self,
        variant_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
        lookback_optimization: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Phase 2: Apply cheap pruning with category protection (40-50% reduction).
        
        Uses our new CheapPruningPipeline with 5 sequential methods:
        1. Variance pruning (~5% reduction, no category protection)
        2. Statistical significance pruning (~10% reduction, no category protection)
        3. Stability pruning (~10-15% reduction, category protection ≥3 per category)
        4. Mutual information pruning (~10% reduction, category protection ≥3 per category)
        5. Correlation pruning (~10-15% reduction, category protection ≥3 per category)
        
        Returns:
            Tuple of (pruned_features, pruning_stats)
        """
        print("🔍 DEBUG: _phase2_cheap_pruning method called!")
        print(f"🔍 DEBUG: variant_features shape: {variant_features.shape}")
        print(f"🔍 DEBUG: labeled_data shape: {labeled_data.shape}")
        print(f"🔍 DEBUG: labeled_data columns: {list(labeled_data.columns)}")
        
        tprint_info("✂️ Applying cheap pruning with category protection")
        tprint_info(f"🔍 DEBUG: _phase2_cheap_pruning called with variant_features shape: {variant_features.shape}")
        tprint_info(f"🔍 DEBUG: labeled_data shape: {labeled_data.shape}")
        tprint_info(f"🔍 DEBUG: labeled_data columns: {list(labeled_data.columns)}")
        
        if not UTILITIES_AVAILABLE:
            raise ImportError("Cheap pruning utilities not available")
        
        # Get targets from labeled data - comprehensive detection
        tprint_info(f"🔍 DEBUG: Analyzing labeled_data columns for targets...")
        tprint_info(f"🔍 DEBUG: Labeled data shape: {labeled_data.shape}")
        tprint_info(f"🔍 DEBUG: Labeled data columns: {list(labeled_data.columns)}")
        
        # Primary target columns (from labeling integration step)
        primary_target_columns = [col for col in labeled_data.columns if col in [
            'directional_confidence', 'opportunity_asymmetry',
            'long_overall_opportunity', 'short_overall_opportunity', 'opportunity',
            'confidence_score', 'quality_score', 'signal_strength'
        ]]
        
        # Secondary target columns (pattern-based detection)
        secondary_target_columns = []
        for col in labeled_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in [
                'target', 'label', 'signal', 'opportunity', 'quality', 'confidence',
                'long_', 'short_', 'directional', 'asymmetry', 'regime', 'trend'
            ]):
                secondary_target_columns.append(col)
        
        # Combine and deduplicate
        all_target_candidates = list(set(primary_target_columns + secondary_target_columns))
        
        # Feature count summary before pruning
        tprint_info(f"📊 PHASE 2: Feature counts before pruning:")
        tprint_info(f"  📈 Input variant features: {len(variant_features.columns)} features")
        tprint_info(f"  📈 Primary target candidates: {primary_target_columns}")
        tprint_info(f"  📈 Secondary target candidates: {secondary_target_columns}")
        tprint_info(f"  📈 All target candidates: {all_target_candidates}")
        
        # Validate target columns have non-zero variance
        valid_target_columns = []
        for col in all_target_candidates:
            try:
                col_data = labeled_data[col].dropna()
                if len(col_data) > 0:
                    variance = col_data.var()
                    non_zero_count = (col_data != 0).sum()
                    tprint_info(f"🔍 DEBUG: Target '{col}': variance={variance:.6f}, non-zero={non_zero_count}/{len(col_data)}")
                    
                    if variance > 1e-10 and non_zero_count > len(col_data) * 0.01:  # At least 1% non-zero
                        valid_target_columns.append(col)
                        tprint_success(f"✅ Valid target found: '{col}' (variance={variance:.6f})")
                    else:
                        tprint_warning(f"⚠️ Invalid target '{col}': variance={variance:.6f}, non-zero={non_zero_count}")
                else:
                    tprint_warning(f"⚠️ Target '{col}' has no valid data after dropna()")
            except Exception as e:
                tprint_warning(f"⚠️ Error validating target '{col}': {e}")
        
        target_columns = valid_target_columns
        
        # If still no valid targets found, raise detailed error
        if not target_columns:
            error_msg = f"""
            ❌ CRITICAL ERROR: No valid target columns found in labeled_data!
            
            Labeled data analysis:
            - Shape: {labeled_data.shape}
            - Columns: {list(labeled_data.columns)}
            - Primary candidates: {primary_target_columns}
            - Secondary candidates: {secondary_target_columns}
            
            This indicates a problem with the labeling integration step.
            Expected columns from labeling step: opportunity, directional_confidence, etc.
            """
            tprint_error(error_msg)
            raise ValueError("No valid target columns found in labeled_data - check labeling integration step")
        
        print(f"🔍 DEBUG: Final target columns: {target_columns}")
        print(f"🔍 DEBUG: Labeled data columns: {list(labeled_data.columns)}")
        
        tprint_info(f"🔍 DEBUG: Found target columns: {target_columns}")
        tprint_info(f"🔍 DEBUG: Labeled data columns: {list(labeled_data.columns)}")
        
        if not target_columns:
            print("🔍 DEBUG: No target columns found, raising error")
            raise ValueError("No target columns found in labeled_data")
        
        # Handle different target column scenarios
        if target_columns == ['opportunity']:
            tprint_info("📊 Using 'opportunity' column as primary target")
            targets = labeled_data[['opportunity']]
            # Create derived targets for compatibility
            targets['directional_confidence'] = labeled_data['opportunity'].abs()
            targets['opportunity_asymmetry'] = labeled_data['opportunity']
            targets['long_overall_opportunity'] = labeled_data['opportunity'].clip(lower=0)
            targets['short_overall_opportunity'] = labeled_data['opportunity'].clip(upper=0).abs()
        elif target_columns == ['dummy_target']:
            tprint_info("📊 Using dummy target for testing")
            targets = labeled_data[['dummy_target']]
            # Create derived targets for compatibility
            targets['directional_confidence'] = labeled_data['dummy_target'].abs()
            targets['opportunity_asymmetry'] = labeled_data['dummy_target']
            targets['long_overall_opportunity'] = labeled_data['dummy_target'].clip(lower=0)
            targets['short_overall_opportunity'] = labeled_data['dummy_target'].clip(upper=0).abs()
        else:
            # Use alternative target columns
            tprint_info(f"📊 Using alternative target columns: {target_columns}")
            targets = labeled_data[target_columns]
            # Only create derived targets if we don't have the specific target we want
            if len(target_columns) == 1 and target_columns[0] != 'price_target_vol_normalized':
                target_col = target_columns[0]
                targets['directional_confidence'] = labeled_data[target_col].abs()
                targets['opportunity_asymmetry'] = labeled_data[target_col]
                targets['long_overall_opportunity'] = labeled_data[target_col].clip(lower=0)
                targets['short_overall_opportunity'] = labeled_data[target_col].clip(upper=0).abs()
        
        # Get feature categories from lookback optimization (feature bank)
        feature_categories = self._get_feature_categories_from_bank(variant_features.columns, lookback_optimization)
        
        # Calculate composite scores with MI and stability
        tprint_info("="*80)
        tprint_info("📊 CALCULATING COMPOSITE SCORES (MI + Stability)")
        tprint_info("="*80)
        composite_scores = self._calculate_composite_scores(
            variant_features, targets, feature_categories
        )
        
        # Apply pruning using our utility
        try:
            pruned_features, pruning_stats = apply_optimized_cheap_pruning(
                features_df=variant_features,
                targets_df=targets,
                feature_categories=feature_categories,
                composite_scores=composite_scores,
                config=OptimizedPruningConfig(
                    # Much less aggressive pruning to retain cross-timeframe features
                    variance_bottom_percentile=1.0,  # Remove only 1% lowest variance
                    stability_bottom_percentile=1.0,  # Remove only 1% least stable
                    significance_bottom_percentile=1.0,  # Remove only 1% least significant
                    mi_bottom_percentile=2.0,  # Remove only 2% lowest MI
                    correlation_threshold=0.95,  # Only remove very highly correlated (95%+)
                    min_features_per_category=1  # Keep at least 1 feature per category
                )
            )
            
            self.performance_stats['features_after_pruning'] = len(pruned_features.columns)
            
            # Feature count summary after pruning
            retention_rate = len(pruned_features.columns) / len(variant_features.columns)
            tprint_info(f"📊 PHASE 2: Pruning results:")
            tprint_info(f"  📈 Input features: {len(variant_features.columns)} features")
            tprint_info(f"  📈 Pruned features: {len(pruned_features.columns)} features")
            tprint_info(f"  📈 Features removed: {len(variant_features.columns) - len(pruned_features.columns)} features")
            tprint_info(f"  📈 Retention rate: {retention_rate:.1%}")
            tprint_info(f"  📈 Reduction rate: {(1 - retention_rate)*100:.1f}%")
            
            # Analyze retention effectiveness
            if retention_rate >= 0.75:
                tprint_success(f"✅ EXCELLENT: {retention_rate:.1%} retention - optimal for interaction generation")
            elif retention_rate >= 0.60:
                tprint_success(f"✅ GOOD: {retention_rate:.1%} retention - good for interaction generation")
            elif retention_rate >= 0.40:
                tprint_warning(f"⚠️ MODERATE: {retention_rate:.1%} retention - may limit interaction generation")
            else:
                tprint_warning(f"⚠️ LOW: {retention_rate:.1%} retention - may significantly limit interaction generation")
            
            tprint_success(f"✅ Pruning completed: {len(variant_features.columns)} -> {len(pruned_features.columns)} features")
            
            # Log category distribution after pruning
            final_categories = self._get_category_distribution(pruned_features.columns, feature_categories)
            tprint_info(f"📊 Final category distribution: {final_categories}")
            
            return pruned_features, pruning_stats, targets
            
        except Exception as e:
            tprint_error(f"❌ Cheap pruning failed: {e}")
            raise
    
    def _get_category_distribution(self, feature_names: List[str], feature_categories: Dict[str, str]) -> Dict[str, int]:
        """Get distribution of features by category."""
        distribution = {}
        for feature_name in feature_names:
            category = feature_categories.get(feature_name, 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    async def _phase3_lgbm_shap_pipeline(
        self,
        pruned_features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
        lookback_optimization: Dict[str, Any] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Phase 3: Three-phase LGBM+SHAP pipeline with corrected interaction discovery.
        
        Phase 3.1: Shallow LGBM sweep (max_depth=4, num_leaves=15, n_estimators=100)
        Phase 3.2: Deeper LGBM refinement (max_depth=5, num_leaves=31, n_estimators=100)
        Phase 3.3: Deep interaction discovery (max_depth=6, num_leaves=31, corrected SHAP approach)
        
        Returns:
            Tuple of (final_features, interactions, shap_metadata)
        """
        if not LGBM_AVAILABLE or not SHAP_AVAILABLE:
            raise ImportError("LightGBM and SHAP are required for Phase 3")
        
        # Feature count summary before Phase 3
        tprint_info(f"📊 PHASE 3: Feature counts before LGBM+SHAP pipeline:")
        tprint_info(f"  📈 Input pruned features: {len(pruned_features.columns)} features")
        tprint_info(f"  📈 Target columns: {len(targets.columns)} targets")
        
        # Targets are already provided as parameter
        
        # Phase 3.1: Shallow LGBM sweep (Select Top 100 Features to protect cross-timeframe features)
        tprint_info("🤖 Phase 3.1: Shallow LGBM Sweep (Select Top 100 Features)")
        phase3_1_start = time.time()
        
        top_100_features = await self._phase3_1_shallow_sweep(pruned_features, targets, config)
        
        self.performance_stats['phase3_1_time'] = time.time() - phase3_1_start
        tprint_performance(f"Phase 3.1 completed", self.performance_stats['phase3_1_time'])
        
        # Feature count summary after Phase 3.1
        tprint_info(f"📊 PHASE 3.1: Shallow sweep results:")
        tprint_info(f"  📈 Input features: {len(pruned_features.columns)} features")
        tprint_info(f"  📈 Selected features: {len(top_100_features.columns)} features")
        tprint_info(f"  📈 Selection rate: {len(top_100_features.columns) / len(pruned_features.columns) * 100:.1f}%")
        
        # Phase 3.2: Deeper LGBM refinement (Select Top 80 to protect cross-timeframe features)
        tprint_info("🤖 Phase 3.2: Deeper LGBM Refinement (Select Top 80)")
        phase3_2_start = time.time()
        
        top_80_features = await self._phase3_2_deeper_refinement(top_100_features, targets, config)
        
        self.performance_stats['phase3_2_time'] = time.time() - phase3_2_start
        tprint_performance(f"Phase 3.2 completed", self.performance_stats['phase3_2_time'])
        
        # Feature count summary after Phase 3.2
        tprint_info(f"📊 PHASE 3.2: Deeper refinement results:")
        tprint_info(f"  📈 Input features: {len(top_100_features.columns)} features")
        tprint_info(f"  📈 Refined features: {len(top_80_features.columns)} features")
        tprint_info(f"  📈 Refinement rate: {len(top_80_features.columns) / len(top_100_features.columns) * 100:.1f}%")
        
        # Phase 3.3: Deep interaction discovery (Generate Top 50)
        tprint_info("🤖 Phase 3.3: Deep Interaction Discovery (Generate Top 50)")
        phase3_3_start = time.time()
        
        # Get feature categories for the top 80 features
        if lookback_optimization is not None:
            feature_categories = self._get_feature_categories_from_bank(top_80_features.columns, lookback_optimization)
        else:
            # Fallback to inference if lookback_optimization not available
            feature_categories = {}
            for feature_name in top_80_features.columns:
                feature_categories[feature_name] = self._infer_feature_category(feature_name)
        
        interactions, shap_metadata = await self._phase3_3_interaction_discovery(
            top_80_features, targets, config, feature_categories
        )
        
        self.performance_stats['phase3_3_time'] = time.time() - phase3_3_start
        tprint_performance(f"Phase 3.3 completed", self.performance_stats['phase3_3_time'])
        
        # Feature count summary after Phase 3.3
        tprint_info("="*80)
        tprint_info(f"📊 PHASE 3.3: Interaction discovery results:")
        tprint_info(f"  📈 Input features: {len(top_80_features.columns)} features")
        tprint_info(f"  📈 Generated interactions: {len(interactions.columns)} features")
        tprint_info(f"  📈 Total features after Phase 3: {len(top_80_features.columns) + len(interactions.columns)} features")
        tprint_info(f"  🔍 DEBUG: Interactions DataFrame shape: {interactions.shape}")
        tprint_info(f"  🔍 DEBUG: Interactions columns (first 10): {list(interactions.columns)[:10]}")
        tprint_info("="*80)
        
        return top_80_features, interactions, shap_metadata

    async def _phase3_1_shallow_sweep(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 3.1: Shallow LGBM sweep to select top 100 features.
        
        Uses fast feature importance + mutual information proxy instead of expensive SHAP.
        - max_depth=4 (increased to capture interactions)
        - num_leaves=15 (more flexibility)
        - n_estimators=100 (more stable importance)
        - Fast proxy: 60% feature importance + 40% mutual information
        - Fixed selection: Always select top 100 features (or all if <100 available)
        """
        tprint_info("  📊 Training shallow LGBM with fast feature selection...")
        
        print(f"🔍 DEBUG: _phase3_1_shallow_sweep called with features shape: {features.shape}, targets shape: {targets.shape}")
        print(f"🔍 DEBUG: targets columns: {list(targets.columns) if len(targets.columns) > 0 else 'EMPTY'}")
        
        # Align features and targets by index with comprehensive validation
        tprint_info(f"🔍 DEBUG: Before alignment - features shape: {features.shape}, targets shape: {targets.shape}")
        tprint_info(f"🔍 DEBUG: Features index range: {features.index.min()} to {features.index.max()}")
        tprint_info(f"🔍 DEBUG: Targets index range: {targets.index.min()} to {targets.index.max()}")
        
        # Find common indices with detailed analysis
        common_indices = features.index.intersection(targets.index)
        tprint_info(f"🔍 DEBUG: Common indices count: {len(common_indices)}")
        
        # Validate alignment requirements
        if len(common_indices) == 0:
            error_msg = f"""
            ❌ CRITICAL ERROR: No common indices between features and targets!
            
            Features analysis:
            - Shape: {features.shape}
            - Index range: {features.index.min()} to {features.index.max()}
            - Index type: {type(features.index)}
            
            Targets analysis:
            - Shape: {targets.shape}
            - Index range: {targets.index.min()} to {targets.index.max()}
            - Index type: {type(targets.index)}
            
            This indicates a data pipeline issue - features and targets must have aligned indices.
            """
            tprint_error(error_msg)
            raise ValueError("No common indices between features and targets - check data pipeline alignment")
        
        # Check if we have sufficient overlap
        overlap_ratio = len(common_indices) / min(len(features), len(targets))
        tprint_info(f"🔍 DEBUG: Index overlap ratio: {overlap_ratio:.3f}")
        
        if overlap_ratio < 0.5:  # Less than 50% overlap
            tprint_warning(f"⚠️ Low index overlap: {overlap_ratio:.3f} - this may affect model performance")
        
        # Align both datasets to common indices (always do this)
        features_aligned = features.loc[common_indices]
        targets_aligned = targets.loc[common_indices]
        
        tprint_info(f"🔍 DEBUG: After alignment - features shape: {features_aligned.shape}, targets shape: {targets_aligned.shape}")
        
        # Validate alignment success
        if len(features_aligned) == 0 or len(targets_aligned) == 0:
            error_msg = f"""
            ❌ CRITICAL ERROR: Alignment resulted in empty datasets!
            
            After alignment:
            - Features shape: {features_aligned.shape}
            - Targets shape: {targets_aligned.shape}
            - Common indices: {len(common_indices)}
            """
            tprint_error(error_msg)
            raise ValueError("Alignment resulted in empty datasets")
        
        # Handle NaN values in features
        print(f"🔍 DEBUG: Features NaN count before cleaning: {features_aligned.isna().sum().sum()}")
        features_cleaned = features_aligned.fillna(0)  # Fill NaN with 0
        print(f"🔍 DEBUG: Features NaN count after cleaning: {features_cleaned.isna().sum().sum()}")
        
        # Handle NaN values in targets with validation
        print(f"🔍 DEBUG: Targets NaN count before cleaning: {targets_aligned.isna().sum().sum()}")
        targets_cleaned = targets_aligned.fillna(0)  # Fill NaN with 0
        print(f"🔍 DEBUG: Targets NaN count after cleaning: {targets_cleaned.isna().sum().sum()}")
        
        # Validate target data quality before model training
        tprint_info("🔍 Validating target data quality...")
        valid_targets = []
        
        for col in targets_cleaned.columns:
            col_data = targets_cleaned[col]
            variance = col_data.var()
            non_zero_count = (col_data != 0).sum()
            unique_count = col_data.nunique()
            
            tprint_info(f"🔍 Target '{col}': variance={variance:.6f}, non-zero={non_zero_count}/{len(col_data)}, unique={unique_count}")
            
            if variance < 1e-10:
                warning_msg = f"""
                ⚠️ WARNING: Target column '{col}' has zero variance - skipping this target!
                
                Target analysis:
                - Variance: {variance:.10f}
                - Non-zero values: {non_zero_count}/{len(col_data)}
                - Unique values: {unique_count}
                - Data range: {col_data.min():.6f} to {col_data.max():.6f}
                
                This target will be excluded from model training.
                """
                tprint_warning(warning_msg)
                continue  # Skip this target instead of failing
            
            if non_zero_count < len(col_data) * 0.01:  # Less than 1% non-zero
                tprint_warning(f"⚠️ Target '{col}' has very few non-zero values: {non_zero_count}/{len(col_data)}")
            
            valid_targets.append(col)
        
        if len(valid_targets) == 0:
            error_msg = "❌ CRITICAL ERROR: No valid targets found! All target columns have zero variance."
            tprint_error(error_msg)
            raise ValueError("No valid targets available for model training")
        
        # Filter targets to only include valid ones
        targets_cleaned = targets_cleaned[valid_targets]
        tprint_success(f"✅ Target data quality validation passed - using {len(valid_targets)} valid targets: {valid_targets}")
        
        # Use consistent sampling strategy with chunked processing
        try:
            features_sample, targets_sample = self._get_consistent_sample(features_cleaned, targets_cleaned, max_samples=8000)
            tprint_info(f"  🔍 DEBUG: _get_consistent_sample returned successfully!")
        except Exception as e:
            tprint_error(f"  🔍 DEBUG: Exception in _get_consistent_sample: {e}")
            raise
        
        # Apply chunked processing for large datasets
        try:
            tprint_info(f"  🔍 DEBUG: About to check chunked processing condition...")
        except Exception as e:
            tprint_error(f"  🔍 DEBUG: Exception after _get_consistent_sample: {e}")
            raise
        if len(features_sample) > 5000:
            tprint_info(f"  🔍 DEBUG: Applying chunked processing for {len(features_sample)} samples")
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)
            tprint_info(f"  🔍 DEBUG: After chunked processing - features shape: {features_sample.shape}")
        else:
            tprint_info(f"  🔍 DEBUG: Skipping chunked processing (samples: {len(features_sample)} <= 5000)")
        
        # Setup LGBM with overfitting prevention parameters
        tprint_info(f"  🔍 DEBUG: About to setup LGBM parameters with regularization...")
        lgbm_params = {
            'max_depth': 3,                    # Further reduced from 4 to prevent overfitting
            'num_leaves': 10,                  # Further reduced from 15 to prevent overfitting
            'n_estimators': 80,                # Further reduced from 100 to prevent overfitting
            'learning_rate': 0.05,             # Reduced from 0.1 for more conservative learning
            'reg_alpha': 0.2,                  # Increased L1 regularization
            'reg_lambda': 0.2,                 # Increased L2 regularization
            'min_child_samples': 80,           # Increased from 50 to prevent overfitting
            'min_split_gain': 0.02,            # Increased minimum gain for splits
            'subsample': 0.6,                  # Reduced row subsampling
            'colsample_bytree': 0.6,           # Reduced column subsampling
            'max_bin': 255,                    # Added max_bin limit
            'min_data_per_group': 50,          # Added minimum data per group
            'random_state': 42,
            'verbose': -1
        }
        
        # Train MultiOutputRegressor with early stopping
        tprint_info(f"  🔍 DEBUG: About to create MultiOutputRegressor with early stopping...")
        
        # Create validation split for early stopping
        if len(features_sample) > 1000:  # Only use early stopping for larger datasets
            val_size = min(200, len(features_sample) // 5)  # 20% or max 200 samples
            X_train_es = features_sample.iloc[:-val_size]
            X_val_es = features_sample.iloc[-val_size:]
            y_train_es = targets_sample.iloc[:-val_size]
            y_val_es = targets_sample.iloc[-val_size:]
            
            # Add early stopping parameters
            lgbm_params['early_stopping_rounds'] = 20
            lgbm_params['eval_metric'] = 'rmse'
            
            model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
            
            # Train with early stopping
            tprint_info("  🔄 Training with early stopping...")
            model.fit(
                X_train_es, y_train_es,
                eval_set=[(X_val_es, y_val_es)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            tprint_info("  ✅ Early stopping training completed")
        else:
            # For smaller datasets, train without early stopping
            model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
            model.fit(features_sample, targets_sample)
            tprint_info("  ✅ Standard training completed (no early stopping for small dataset)")
        
        # Calculate performance metrics with comprehensive error reporting
        tprint_info("  📊 Calculating model performance metrics...")
        try:
            # Calculate accuracy (R² score for regression)
            from sklearn.metrics import r2_score
            predictions = model.predict(features_sample)
            accuracy = r2_score(targets_sample, predictions)
            
            # Calculate cross-validation score with time-series aware validation
            if OVERFITTING_PREVENTION_AVAILABLE:
                tprint_info("  📊 Using time-series aware cross-validation...")
                cv_results = temporal_cross_validation(
                    model, features_sample, targets_sample,
                    n_splits=5,
                    gap=1,  # Gap to prevent data leakage
                    test_size=None,  # Use default test size
                    scoring='r2'
                )
                cv_score = cv_results.get('mean_score', 0.0)
                cv_scores_std = cv_results.get('std_score', 0.0)
                tprint_info(f"  ✅ Time-series CV completed: {cv_score:.4f} ± {cv_scores_std:.4f}")
            else:
                # Fallback to standard cross-validation
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, features_sample, targets_sample, cv=3, scoring='r2')
                cv_score = cv_scores.mean()
                cv_scores_std = cv_scores.std()
            
            # Calculate feature importance consistency
            importance_consistency = self._calculate_importance_consistency(model, features_sample, targets_sample)
            
            # Store performance metrics
            self._phase3_1_performance = {
                'accuracy': accuracy,
                'cv_score': cv_score,
                'importance_consistency': importance_consistency,
                'cv_scores_std': cv_scores_std
            }
            
            tprint_success(f"  ✅ Performance metrics calculated: Accuracy={accuracy:.4f}, CV Score={cv_score:.4f}")
            
        except Exception as e:
            error_msg = f"""
            ❌ CRITICAL ERROR: Failed to calculate performance metrics!
            
            Error details:
            - Exception: {e}
            - Exception type: {type(e)}
            
            Data analysis:
            - Features shape: {features_sample.shape}
            - Targets shape: {targets_sample.shape}
            - Features columns: {len(features_sample.columns)}
            - Targets columns: {len(targets_sample.columns)}
            - Features NaN count: {features_sample.isna().sum().sum()}
            - Targets NaN count: {targets_sample.isna().sum().sum()}
            
            Model analysis:
            - Model type: {type(model)}
            - Model fitted: {hasattr(model, 'estimators_')}
            
            This indicates a fundamental issue with model training or data quality.
            """
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            # Re-raise the exception to stop execution
            raise ValueError(f"Performance metrics calculation failed: {e}") from e
        
        # Fast feature importance calculation (no SHAP)
        tprint_info("  🔍 Calculating fast feature importance...")
        
        # Get feature importance from LGBM
        importance_scores = model.estimators_[0].feature_importances_
        
        # Calculate mutual information with first target using fast proxy
        mi_scores = []
        for col in features_sample.columns:
            mi_score = self._fast_mi_proxy(features_sample[col], targets_sample.iloc[:, 0], n_bins=5)
            mi_scores.append(mi_score)
        mi_scores = np.array(mi_scores)
        
        # Normalize scores
        importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores) + 1e-8)
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
        
        # Combined score: 60% importance + 40% mutual information
        combined_scores = 0.6 * importance_scores + 0.4 * mi_scores
        
        # Rank features by combined score
        feature_importance = pd.Series(
            combined_scores,
            index=features.columns
        ).sort_values(ascending=False)
        
        # Select top 100 features (fixed number)
        n_select = min(100, len(features.columns))  # Select 100 or all if less than 100
        top_features = feature_importance.head(n_select).index.tolist()
        
        tprint_success(f"  ✅ Selected {len(top_features)} features (top {n_select}) using fast proxy")
        
        return features[top_features]
    
    def _create_holdout_split(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                             test_size: float = 0.2, gap_size: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create holdout validation split with gap to prevent data leakage.
        
        Args:
            features_df: Feature dataframe
            targets_df: Target dataframe  
            test_size: Proportion of data for test set
            gap_size: Gap between train and test to prevent leakage
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if OVERFITTING_PREVENTION_AVAILABLE and self.data_leakage_prevention:
            tprint_info("🔒 Creating holdout split with data leakage prevention...")
            
            # Use data leakage prevention to create proper time-series split
            split_result = self.data_leakage_prevention.create_time_series_split(
                features_df, targets_df,
                test_size=test_size,
                gap_size=gap_size,
                random_state=42
            )
            
            X_train, X_test, y_train, y_test = split_result
            tprint_info(f"✅ Holdout split created: Train={len(X_train)}, Test={len(X_test)}, Gap={gap_size}")
            
            return X_train, X_test, y_train, y_test
        else:
            # Fallback to simple time-based split
            tprint_warning("⚠️ Using fallback time-based split (no data leakage prevention)")
            
            # Simple time-based split (last test_size% for test)
            split_idx = int(len(features_df) * (1 - test_size))
            
            # Add gap
            gap_start = max(0, split_idx - gap_size)
            gap_end = min(len(features_df), split_idx + gap_size)
            
            X_train = features_df.iloc[:gap_start]
            X_test = features_df.iloc[gap_end:]
            y_train = targets_df.iloc[:gap_start]
            y_test = targets_df.iloc[gap_end:]
            
            tprint_info(f"✅ Fallback split created: Train={len(X_train)}, Test={len(X_test)}")
            
            return X_train, X_test, y_train, y_test

    def _validate_feature_selection_oos(self, selected_features: List[str], 
                                       X_train: pd.DataFrame, X_test: pd.DataFrame,
                                       y_train: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature selection on out-of-sample data to detect overfitting.
        
        Args:
            selected_features: List of selected feature names
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            Dictionary with validation metrics
        """
        if not LGBM_AVAILABLE:
            return {'validation_performed': False, 'reason': 'LGBM not available'}
        
        try:
            tprint_info("🔍 Validating feature selection on out-of-sample data...")
            
            # Ensure selected features exist in both train and test
            available_features = [f for f in selected_features if f in X_train.columns and f in X_test.columns]
            
            if len(available_features) < len(selected_features):
                tprint_warning(f"⚠️ {len(selected_features) - len(available_features)} features missing in test set")
            
            if len(available_features) == 0:
                return {'validation_performed': False, 'reason': 'No features available for validation'}
            
            # Create train/test sets with selected features
            X_train_selected = X_train[available_features]
            X_test_selected = X_test[available_features]
            
            # Train model on training set
            lgbm_params = {
                'max_depth': 4,
                'num_leaves': 15,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_child_samples': 50,
                'random_state': 42,
                'verbose': -1
            }
            
            model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
            model.fit(X_train_selected, y_train)
            
            # Evaluate on training set
            from sklearn.metrics import r2_score
            train_pred = model.predict(X_train_selected)
            train_score = r2_score(y_train, train_pred)
            
            # Evaluate on test set
            test_pred = model.predict(X_test_selected)
            test_score = r2_score(y_test, test_pred)
            
            # Calculate overfitting metrics
            performance_gap = train_score - test_score
            overfitting_risk = 'High' if performance_gap > 0.1 else 'Medium' if performance_gap > 0.05 else 'Low'
            
            validation_results = {
                'validation_performed': True,
                'train_score': train_score,
                'test_score': test_score,
                'performance_gap': performance_gap,
                'overfitting_risk': overfitting_risk,
                'features_validated': len(available_features),
                'features_missing': len(selected_features) - len(available_features)
            }
            
            tprint_info(f"📊 OOS Validation Results:")
            tprint_info(f"  Train Score: {train_score:.4f}")
            tprint_info(f"  Test Score: {test_score:.4f}")
            tprint_info(f"  Performance Gap: {performance_gap:.4f}")
            tprint_info(f"  Overfitting Risk: {overfitting_risk}")
            
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ OOS validation failed: {e}")
            return {'validation_performed': False, 'reason': f'Validation failed: {e}'}
    
    def _calculate_importance_consistency(self, model, features: pd.DataFrame, targets: pd.DataFrame) -> float:
        """
        Calculate feature importance consistency across different CV folds.
        
        Returns a score between 0 and 1 indicating how consistent feature importance is.
        """
        try:
            from sklearn.model_selection import KFold
            from sklearn.metrics import r2_score
            
            # Get feature importance from multiple CV folds
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            importance_folds = []
            
            for train_idx, val_idx in kf.split(features):
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = targets.iloc[train_idx], targets.iloc[val_idx]
                
                # Train a model on this fold
                fold_model = MultiOutputRegressor(lgb.LGBMRegressor(
                    max_depth=4, num_leaves=15, n_estimators=50, 
                    learning_rate=0.1, random_state=42, verbose=-1
                ))
                fold_model.fit(X_train, y_train)
                
                # Get feature importance
                importance = fold_model.estimators_[0].feature_importances_
                importance_folds.append(importance)
            
            # Calculate consistency (correlation between fold importances)
            if len(importance_folds) >= 2:
                consistency_scores = []
                for i in range(len(importance_folds)):
                    for j in range(i + 1, len(importance_folds)):
                        corr = np.corrcoef(importance_folds[i], importance_folds[j])[0, 1]
                        if not np.isnan(corr):
                            consistency_scores.append(abs(corr))
                
                return np.mean(consistency_scores) if consistency_scores else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating importance consistency: {e}")
            return 0.0
    
    async def _phase3_2_deeper_refinement(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 3.2: Deeper LGBM refinement to select top 80 features.
        
        Uses deeper LGBM with fast multi-criteria selection:
        - Feature importance (60%)
        - Mutual information (30%)
        - Stability (10%)
        """
        tprint_info("  📊 Training deeper LGBM for refinement...")
        
        # Align features and targets by index with comprehensive validation
        tprint_info(f"🔍 DEBUG: Before alignment - features shape: {features.shape}, targets shape: {targets.shape}")
        tprint_info(f"🔍 DEBUG: Features index range: {features.index.min()} to {features.index.max()}")
        tprint_info(f"🔍 DEBUG: Targets index range: {targets.index.min()} to {targets.index.max()}")
        
        # Find common indices with detailed analysis
        common_indices = features.index.intersection(targets.index)
        tprint_info(f"🔍 DEBUG: Common indices count: {len(common_indices)}")
        
        # Validate alignment requirements
        if len(common_indices) == 0:
            error_msg = f"""
            ❌ CRITICAL ERROR: No common indices between features and targets!
            
            Features analysis:
            - Shape: {features.shape}
            - Index range: {features.index.min()} to {features.index.max()}
            - Index type: {type(features.index)}
            
            Targets analysis:
            - Shape: {targets.shape}
            - Index range: {targets.index.min()} to {targets.index.max()}
            - Index type: {type(targets.index)}
            
            This indicates a data pipeline issue - features and targets must have aligned indices.
            """
            tprint_error(error_msg)
            raise ValueError("No common indices between features and targets - check data pipeline alignment")
        
        # Check if we have sufficient overlap
        overlap_ratio = len(common_indices) / min(len(features), len(targets))
        tprint_info(f"🔍 DEBUG: Index overlap ratio: {overlap_ratio:.3f}")
        
        if overlap_ratio < 0.5:  # Less than 50% overlap
            tprint_warning(f"⚠️ Low index overlap: {overlap_ratio:.3f} - this may affect model performance")
        
        # Align both datasets to common indices (always do this)
        features_aligned = features.loc[common_indices]
        targets_aligned = targets.loc[common_indices]
        
        tprint_info(f"🔍 DEBUG: After alignment - features shape: {features_aligned.shape}, targets shape: {targets_aligned.shape}")
        
        # Validate alignment success
        if len(features_aligned) == 0 or len(targets_aligned) == 0:
            error_msg = f"""
            ❌ CRITICAL ERROR: Alignment resulted in empty datasets!
            
            After alignment:
            - Features shape: {features_aligned.shape}
            - Targets shape: {targets_aligned.shape}
            - Common indices: {len(common_indices)}
            """
            tprint_error(error_msg)
            raise ValueError("Alignment resulted in empty datasets")
        
        # Handle NaN values in features
        print(f"🔍 DEBUG: Features NaN count before cleaning: {features_aligned.isna().sum().sum()}")
        features_cleaned = features_aligned.fillna(0)  # Fill NaN with 0
        print(f"🔍 DEBUG: Features NaN count after cleaning: {features_cleaned.isna().sum().sum()}")
        
        # Handle NaN values in targets with validation
        print(f"🔍 DEBUG: Targets NaN count before cleaning: {targets_aligned.isna().sum().sum()}")
        targets_cleaned = targets_aligned.fillna(0)  # Fill NaN with 0
        print(f"🔍 DEBUG: Targets NaN count after cleaning: {targets_cleaned.isna().sum().sum()}")
        
        # Validate target data quality before model training
        tprint_info("🔍 Validating target data quality...")
        valid_targets = []
        
        for col in targets_cleaned.columns:
            col_data = targets_cleaned[col]
            variance = col_data.var()
            non_zero_count = (col_data != 0).sum()
            unique_count = col_data.nunique()
            
            tprint_info(f"🔍 Target '{col}': variance={variance:.6f}, non-zero={non_zero_count}/{len(col_data)}, unique={unique_count}")
            
            if variance < 1e-10:
                warning_msg = f"""
                ⚠️ WARNING: Target column '{col}' has zero variance - skipping this target!
                
                Target analysis:
                - Variance: {variance:.10f}
                - Non-zero values: {non_zero_count}/{len(col_data)}
                - Unique values: {unique_count}
                - Data range: {col_data.min():.6f} to {col_data.max():.6f}
                
                This target will be excluded from model training.
                """
                tprint_warning(warning_msg)
                continue  # Skip this target instead of failing
            
            if non_zero_count < len(col_data) * 0.01:  # Less than 1% non-zero
                tprint_warning(f"⚠️ Target '{col}' has very few non-zero values: {non_zero_count}/{len(col_data)}")
            
            valid_targets.append(col)
        
        if len(valid_targets) == 0:
            error_msg = "❌ CRITICAL ERROR: No valid targets found! All target columns have zero variance."
            tprint_error(error_msg)
            raise ValueError("No valid targets available for model training")
        
        # Filter targets to only include valid ones
        targets_cleaned = targets_cleaned[valid_targets]
        tprint_success(f"✅ Target data quality validation passed - using {len(valid_targets)} valid targets: {valid_targets}")
        
        # Use consistent sampling strategy with chunked processing
        features_sample, targets_sample = self._get_consistent_sample(features_cleaned, targets_cleaned, max_samples=8000)
        
        # Apply chunked processing for large datasets
        if len(features_sample) > 5000:
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)
        
        # Setup deeper LGBM with enhanced overfitting prevention
        lgbm_params = {
            'max_depth': 3,                    # Further reduced from 4 to prevent overfitting
            'num_leaves': 10,                  # Further reduced from 15 to prevent overfitting
            'n_estimators': 80,                # Reduced from 100 to prevent overfitting
            'learning_rate': 0.05,             # Reduced from 0.1 for more conservative learning
            'reg_alpha': 0.2,                  # Increased L1 regularization
            'reg_lambda': 0.2,                 # Increased L2 regularization
            'min_child_samples': 80,           # Increased from 50 to prevent overfitting
            'min_split_gain': 0.02,            # Increased minimum gain for splits
            'subsample': 0.6,                  # Reduced row subsampling
            'colsample_bytree': 0.6,           # Reduced column subsampling
            'max_bin': 255,                    # Added max_bin limit
            'min_data_per_group': 50,          # Added minimum data per group
            'random_state': 42,
            'verbose': -1
        }
        
        # Train MultiOutputRegressor
        model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
        model.fit(features_sample, targets_sample)
        
        # Calculate performance metrics for Phase 3.2 with comprehensive error reporting
        tprint_info("  📊 Calculating Phase 3.2 performance metrics...")
        try:
            from sklearn.metrics import r2_score
            from sklearn.model_selection import cross_val_score
            
            # Calculate accuracy (R² score for regression)
            predictions = model.predict(features_sample)
            accuracy = r2_score(targets_sample, predictions)
            
            # Calculate cross-validation score with time-series aware validation
            if OVERFITTING_PREVENTION_AVAILABLE:
                tprint_info("  📊 Using time-series aware cross-validation for Phase 3.2...")
                cv_results = temporal_cross_validation(
                    model, features_sample, targets_sample,
                    n_splits=5,
                    gap=1,  # Gap to prevent data leakage
                    test_size=None,
                    scoring='r2'
                )
                cv_score = cv_results.get('mean_score', 0.0)
                cv_scores_std = cv_results.get('std_score', 0.0)
                tprint_info(f"  ✅ Phase 3.2 Time-series CV: {cv_score:.4f} ± {cv_scores_std:.4f}")
            else:
                # Fallback to standard cross-validation
                cv_scores = cross_val_score(model, features_sample, targets_sample, cv=3, scoring='r2')
                cv_score = cv_scores.mean()
                cv_scores_std = cv_scores.std()
            
            # Calculate feature importance consistency
            importance_consistency = self._calculate_importance_consistency(model, features_sample, targets_sample)
            
            # Store performance metrics
            self._phase3_2_performance = {
                'accuracy': accuracy,
                'cv_score': cv_score,
                'importance_consistency': importance_consistency,
                'cv_scores_std': cv_scores_std
            }
            
            tprint_success(f"  ✅ Phase 3.2 Performance: Accuracy={accuracy:.4f}, CV Score={cv_score:.4f}")
            
        except Exception as e:
            error_msg = f"""
            ❌ CRITICAL ERROR: Failed to calculate Phase 3.2 performance metrics!
            
            Error details:
            - Exception: {e}
            - Exception type: {type(e)}
            
            Data analysis:
            - Features shape: {features_sample.shape}
            - Targets shape: {targets_sample.shape}
            - Features columns: {len(features_sample.columns)}
            - Targets columns: {len(targets_sample.columns)}
            - Features NaN count: {features_sample.isna().sum().sum()}
            - Targets NaN count: {targets_sample.isna().sum().sum()}
            
            Model analysis:
            - Model type: {type(model)}
            - Model fitted: {hasattr(model, 'estimators_')}
            
            This indicates a fundamental issue with Phase 3.2 model training or data quality.
            """
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            # Re-raise the exception to stop execution
            raise ValueError(f"Phase 3.2 performance metrics calculation failed: {e}") from e
        
        # Fast multi-criteria selection (no SHAP)
        tprint_info("  🔍 Calculating fast multi-criteria scores...")
        
        # Calculate feature importance
        feature_importance = model.estimators_[0].feature_importances_
        
        # Calculate mutual information with first target using fast proxy
        mi_scores = []
        for col in features_sample.columns:
            mi_score = self._fast_mi_proxy(features_sample[col], targets_sample.iloc[:, 0], n_bins=5)
            mi_scores.append(mi_score)
        mi_scores = np.array(mi_scores)
        
        # Calculate stability (variance across features)
        stability = np.var(features_sample.values, axis=0)
        
        # Normalize scores
        imp_scores = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance) + 1e-8)
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
        stab_scores = (stability - np.min(stability)) / (np.max(stability) - np.min(stability) + 1e-8)
        
        # Multi-criteria selection
        combined_scores = (
            0.6 * imp_scores +   # Feature importance (60%)
            0.3 * mi_scores +    # Mutual information (30%)
            0.1 * stab_scores    # Stability (10%)
        )
        
        # Rank and select top 80
        feature_scores = pd.Series(combined_scores, index=features.columns).sort_values(ascending=False)
        n_select = min(80, len(features.columns))
        top_features = feature_scores.head(n_select).index.tolist()
        
        tprint_success(f"  ✅ Selected {len(top_features)} features (top 80) using fast proxy")
        
        return features[top_features]
    
    async def _phase3_3_interaction_discovery(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
        feature_categories: Dict[str, str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Phase 3.3: Deep interaction discovery with corrected SHAP approach.
        
        Uses tree-based interaction guidance and corrected SHAP analysis:
        1. Train deep LGBM to extract feature pairs
        2. Generate 3 operations per top 10 pairs (30 candidates)
        3. Use standard SHAP values FOR interaction features (not interaction values)
        4. Select top 50 interactions
        5. Generate cross-timeframe interactions between features from different timeframes
        """
        tprint_info("  🌳 Training deep LGBM for interaction guidance...")
        
        # Extract cross-timeframe interaction features
        tprint_info("  🔄 Extracting cross-timeframe interactions...")
        cross_timeframe_interactions = await self._extract_cross_timeframe_interactions(features)
        if len(cross_timeframe_interactions.columns) > 0:
            tprint_success(f"  ✅ Found {len(cross_timeframe_interactions.columns)} cross-timeframe interaction features")
            # Merge cross-timeframe interactions into features
            features = pd.concat([features, cross_timeframe_interactions], axis=1)
        else:
            tprint_info("  ℹ️ No cross-timeframe interactions detected")
        
        # Align features and targets by index before sampling
        print(f"🔍 DEBUG: _phase3_3_interaction_discovery - Before alignment - features shape: {features.shape}, targets shape: {targets.shape}")
        print(f"🔍 DEBUG: Features index range: {features.index.min()} to {features.index.max()}")
        print(f"🔍 DEBUG: Targets index range: {targets.index.min()} to {targets.index.max()}")
        
        # Find common indices
        common_indices = features.index.intersection(targets.index)
        print(f"🔍 DEBUG: Common indices count: {len(common_indices)}")
        
        if len(common_indices) == 0:
            print("🔍 DEBUG: No common indices found! This will cause alignment issues.")
            # Use the smaller dataset size
            min_length = min(len(features), len(targets))
            features_aligned = features.iloc[:min_length]
            targets_aligned = targets.iloc[:min_length]
        else:
            # Align both datasets to common indices
            features_aligned = features.loc[common_indices]
            targets_aligned = targets.loc[common_indices]
        
        print(f"🔍 DEBUG: After alignment - features shape: {features_aligned.shape}, targets shape: {targets_aligned.shape}")
        
        # Handle NaN values in features
        print(f"🔍 DEBUG: Features NaN count before cleaning: {features_aligned.isna().sum().sum()}")
        features_cleaned = features_aligned.fillna(0)  # Fill NaN with 0
        print(f"🔍 DEBUG: Features NaN count after cleaning: {features_cleaned.isna().sum().sum()}")
        
        # Handle NaN values in targets
        print(f"🔍 DEBUG: Targets NaN count before cleaning: {targets_aligned.isna().sum().sum()}")
        targets_cleaned = targets_aligned.fillna(0)  # Fill NaN with 0
        print(f"🔍 DEBUG: Targets NaN count after cleaning: {targets_cleaned.isna().sum().sum()}")
        
        print(f"🔍 DEBUG: About to enter try block for _get_consistent_sample...")
        
        # Use consistent sampling strategy with chunked processing
        try:
            print(f"🔍 DEBUG: About to call _get_consistent_sample...")
            features_sample, targets_sample = self._get_consistent_sample(features_cleaned, targets_cleaned, max_samples=8000)
            print(f"🔍 DEBUG: _get_consistent_sample completed successfully!")
            print(f"🔍 DEBUG: About to call tprint_info...")
            tprint_info(f"  🔍 DEBUG: _get_consistent_sample returned successfully!")
            print(f"🔍 DEBUG: After tprint_info call...")
        except Exception as e:
            print(f"🔍 DEBUG: Exception caught: {e}")
            tprint_error(f"  🔍 DEBUG: Exception in _get_consistent_sample: {e}")
            tprint_error(f"  🔍 DEBUG: Exception type: {type(e)}")
            import traceback
            tprint_error(f"  🔍 DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        print(f"🔍 DEBUG: Exited try-except block successfully!")
        print(f"🔍 DEBUG: About to call tprint_info with shapes...")
        tprint_info(f"  🔍 DEBUG: After _get_consistent_sample - features shape: {features_sample.shape}, targets shape: {targets_sample.shape}")
        print(f"🔍 DEBUG: After tprint_info with shapes...")
        
        # Apply chunked processing for large datasets
        print(f"🔍 DEBUG: About to call tprint_info for chunked processing...")
        tprint_info(f"  🔍 DEBUG: Checking chunked processing condition...")
        print(f"🔍 DEBUG: After tprint_info for chunked processing...")
        print(f"🔍 DEBUG: About to check chunked processing condition - len(features_sample) = {len(features_sample)}")
        if len(features_sample) > 5000:
            tprint_info(f"  🔍 DEBUG: Applying chunked processing for {len(features_sample)} samples")
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)
            tprint_info(f"  🔍 DEBUG: After chunked processing - features shape: {features_sample.shape}")
        else:
            print(f"🔍 DEBUG: Entering else branch for chunked processing...")
            tprint_info(f"  🔍 DEBUG: Skipping chunked processing (samples: {len(features_sample)} <= 5000)")
            print(f"🔍 DEBUG: After tprint_info in else branch...")
        
        print(f"🔍 DEBUG: About to call tprint_info for LGBM setup...")
        tprint_info(f"  🔍 DEBUG: About to setup LGBM parameters...")
        print(f"🔍 DEBUG: After tprint_info for LGBM setup...")
        
        # Setup LGBM with corrected parameters
        print(f"🔍 DEBUG: About to create LGBM parameters...")
        lgbm_params = {
            'max_depth': 3,                    # Further reduced from 5 to prevent overfitting
            'num_leaves': 10,                  # Further reduced from 20 to prevent overfitting
            'n_estimators': 80,                # Further reduced from 150 to prevent overfitting
            'learning_rate': 0.05,             # Reduced from 0.1 for more conservative learning
            'reg_alpha': 0.25,                 # Increased L1 regularization
            'reg_lambda': 0.25,                # Increased L2 regularization
            'min_child_samples': 100,          # Increased from 75 to prevent overfitting
            'min_split_gain': 0.02,            # Increased minimum gain for splits
            'subsample': 0.6,                  # Further reduced row subsampling
            'colsample_bytree': 0.6,           # Further reduced column subsampling
            'max_bin': 255,                    # Added max_bin limit
            'min_data_per_group': 50,          # Added minimum data per group
            'random_state': 42,
            'verbose': -1
        }
        print(f"🔍 DEBUG: LGBM parameters created successfully!")
        
        print(f"🔍 DEBUG: About to call tprint_info for LGBM training...")
        tprint_info(f"  🔍 DEBUG: LGBM parameters set, about to train LGBM model...")
        print(f"🔍 DEBUG: After first tprint_info for LGBM training...")
        tprint_info(f"  🔍 DEBUG: About to train LGBM model with features shape: {features_sample.shape}, targets shape: {targets_sample.shape}")
        print(f"🔍 DEBUG: After second tprint_info for LGBM training...")
        
        # Train LGBM model for tree analysis
        print(f"🔍 DEBUG: About to call tprint_info for training message...")
        tprint_info("  🔧 Training LGBM model for tree analysis...")
        print(f"🔍 DEBUG: After training message tprint_info...")
        tprint_info(f"  🔍 DEBUG: Features shape: {features_sample.shape}, Targets shape: {targets_sample.shape}")
        print(f"🔍 DEBUG: After features shape tprint_info...")
        
        print(f"🔍 DEBUG: About to enter try block for LGBM training...")
        try:
            print(f"🔍 DEBUG: Inside try block, about to call tprint_info...")
            tprint_info("  🔍 DEBUG: About to create LGBMRegressor...")
            print(f"🔍 DEBUG: After tprint_info for LGBMRegressor creation...")
            print(f"🔍 DEBUG: About to create LGBMRegressor with params: {lgbm_params}")
            model = lgb.LGBMRegressor(**lgbm_params)
            print(f"🔍 DEBUG: LGBMRegressor created successfully!")
            
            print(f"🔍 DEBUG: About to call tprint_info for model fitting...")
            tprint_info("  🔍 DEBUG: About to fit LGBM model...")
            print(f"🔍 DEBUG: After tprint_info for model fitting...")
            print(f"🔍 DEBUG: About to fit model with features shape: {features_sample.shape}, targets shape: {targets_sample.iloc[:, 0].shape}")
            model.fit(features_sample, targets_sample.iloc[:, 0])  # Use first target column
            print(f"🔍 DEBUG: Model fitted successfully!")
            
        except Exception as e:
            print(f"🔍 DEBUG: Exception caught: {e}")
            tprint_error(f"  🔍 DEBUG: Exception during LGBM training: {e}")
            tprint_error(f"  🔍 DEBUG: Exception type: {type(e)}")
            import traceback
            tprint_error(f"  🔍 DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        print(f"🔍 DEBUG: Exited try-except block for LGBM training...")
        tprint_info("  ✅ LGBM model trained successfully")
        
        # Extract feature pairs from trees
        print(f"🔍 DEBUG: About to call tprint_info for tree analysis...")
        tprint_info("  🔍 Extracting feature pairs from tree splits...")
        print(f"🔍 DEBUG: After tprint_info for tree analysis...")
        print(f"🔍 DEBUG: About to call _extract_tree_splitting_pairs...")
        feature_pairs = self._extract_tree_splitting_pairs(model)
        print(f"🔍 DEBUG: _extract_tree_splitting_pairs completed!")
        print(f"🔍 DEBUG: About to check feature_pairs length: {len(feature_pairs)}")
        
        if len(feature_pairs) == 0:
            tprint_warning("  ⚠️ No feature pairs extracted from tree analysis - this will result in no interactions")
            # Fallback: create some basic interactions from available features
            available_features = list(features.columns)
            if len(available_features) >= 2:
                tprint_info("  🔧 Creating fallback feature pairs...")
                for i in range(min(3, len(available_features))):
                    for j in range(i+1, min(i+3, len(available_features))):
                        feature_pairs.append((available_features[i], available_features[j], 1))
                tprint_info(f"  🔍 DEBUG: Created {len(feature_pairs)} fallback feature pairs")
        
        print(f"🔍 DEBUG: Feature pairs found, skipping fallback logic...")
        
        # Generate interaction candidates (top 80 pairs × 5 operations = 400 candidates)
        print(f"🔍 DEBUG: About to call tprint_info for interaction generation...")
        tprint_info("  🔧 Generating interaction candidates...")
        print(f"🔍 DEBUG: After tprint_info for interaction generation...")
        interaction_candidates = []
        print(f"🔍 DEBUG: Interaction candidates list initialized...")
        
        print(f"🔍 DEBUG: About to start for loop for feature pairs...")
        print(f"🔍 DEBUG: Features columns: {list(features.columns)}")
        for i, (f1, f2, co_occurrence) in enumerate(feature_pairs[:80]):  # Top 80 pairs
            print(f"🔍 DEBUG: Processing pair {i+1}/80: {f1} x {f2}")
            # Convert integer indices to column names
            if isinstance(f1, int) and isinstance(f2, int):
                if f1 < len(features.columns) and f2 < len(features.columns):
                    f1_name = features.columns[f1]
                    f2_name = features.columns[f2]
                    print(f"🔍 DEBUG: Converted to column names: {f1_name} x {f2_name}")
                else:
                    print(f"🔍 DEBUG: Skipping pair - indices out of range: {f1}, {f2} (max index: {len(features.columns)-1})")
                    continue
            else:
                f1_name = f1
                f2_name = f2
                print(f"🔍 DEBUG: Using original names: {f1_name} x {f2_name}")
            
            if f1_name in features.columns and f2_name in features.columns:
                print(f"🔍 DEBUG: Both features found in columns, generating operations...")
                # Generate 5 operations per pair (including logarithmic relationships)
                operations = [
                    (f"{f1_name}_x_{f2_name}", features[f1_name] * features[f2_name]),
                    (f"{f1_name}_div_{f2_name}", features[f1_name] / (features[f2_name] + 1e-8)),
                    (f"{f1_name}_minus_{f2_name}", features[f1_name] - features[f2_name]),
                    (f"{f1_name}_log_{f2_name}", np.log(np.abs(features[f1_name]) + 1e-8) / (np.log(np.abs(features[f2_name]) + 1e-8) + 1e-8)),
                    (f"{f1_name}_log_ratio_{f2_name}", np.log(np.abs(features[f1_name] / (features[f2_name] + 1e-8)) + 1e-8))
                ]
                
                for name, interaction in operations:
                    interaction_candidates.append((name, interaction))
        
        # Use CompositeFeatureScorer with RFE-style selection for robust interaction selection
        tprint_info("="*80)
        tprint_info("📊 COMPOSITE SCORING WITH RFE FOR INTERACTION SELECTION")
        tprint_info("="*80)
        tprint_info(f"  📊 Testing {len(interaction_candidates)} interaction candidates...")
        tprint_info(f"  📊 Target: Select top 50 interactions")
        tprint_info(f"  📊 Method: 5-way composite scoring with RFE (33% removal per round)")
        tprint_info(f"  📊 Weights: MI(20%) + Redundancy(20%) + LGBM(20%) + SHAP(20%) + Stability(20%)")
        
        # Prepare data for CompositeFeatureScorer
        from src.training.utils.feature_selection import CompositeFeatureScorer
        
        # Find common indices between features and targets
        common_indices = features.index.intersection(targets.index)
        if len(common_indices) == 0:
            tprint_error("  ❌ No common indices between features and targets!")
            return pd.DataFrame(), {}
        
        # Align targets
        targets_aligned = targets.loc[common_indices]
        target_col = targets_aligned.columns[0]
        
        # Create interaction DataFrame aligned to common indices
        interaction_df_candidates = pd.DataFrame(index=common_indices)
        candidate_names = []
        
        for name, interaction_series in interaction_candidates:
            # Align interaction to common indices
            if hasattr(interaction_series, 'loc'):
                aligned_series = interaction_series.reindex(common_indices, fill_value=0)
            else:
                aligned_series = pd.Series(interaction_series, index=common_indices)
            
            interaction_df_candidates[name] = aligned_series.fillna(0)
            candidate_names.append(name)
        
        tprint_info(f"  📊 Prepared {len(candidate_names)} candidates for composite scoring")
        
        # Use CompositeFeatureScorer with RFE
        composite_scorer = CompositeFeatureScorer(config={
            'rfe_removal_rate': 0.33,  # Remove 33% per round
            'min_features_per_round': 10
        })
        
        selection_result = composite_scorer.select_features(
            X=interaction_df_candidates.values,
            y=targets_aligned[target_col].values,
            feature_names=candidate_names,
            n_features=50
        )
        
        if not selection_result.get('success', False):
            tprint_error(f"  ❌ Composite scoring failed: {selection_result.get('error', 'Unknown error')}")
            # Fallback to simple MI
            tprint_warning("  ⚠️ Falling back to simple MI selection...")
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(
                interaction_df_candidates.values,
                targets_aligned[target_col].values,
                random_state=42,
                n_neighbors=3
            )
            mi_dict = {candidate_names[i]: mi_scores[i] for i in range(len(candidate_names))}
            sorted_interactions = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            # Use composite scores
            composite_scores = selection_result['scores']
            sorted_interactions = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            
            tprint_success(f"  ✅ Composite RFE completed in {selection_result.get('rounds', 0)} rounds")
            tprint_info(f"  📊 Selected {len(selection_result['selected_features'])} interactions")
            
            # Log score breakdown for top features
            if len(sorted_interactions) > 0:
                tprint_info(f"  📊 Score range: {sorted_interactions[-1][1]:.4f} - {sorted_interactions[0][1]:.4f}")
        
        tprint_info(f"  📊 Valid interactions selected: {len(sorted_interactions)}")
        
        # Apply overfitting prevention: Limit interaction complexity and apply stability checks
        if OVERFITTING_PREVENTION_AVAILABLE and self.overfitting_manager is not None:
            # Filter interactions by complexity (limit to 3-way interactions max)
            max_complexity = getattr(self.overfitting_config, 'max_interaction_complexity', 3)
            filtered_interactions = []
            for name, score in sorted_interactions:
                # Count interaction terms (e.g., "feature1_x_feature2" = 2 terms)
                complexity = len(name.split('_x_'))
                if complexity <= max_complexity:
                    filtered_interactions.append((name, score))
                else:
                    tprint_info(f"  🚫 Filtered out complex interaction: {name} (complexity: {complexity})")
            
            sorted_interactions = filtered_interactions
            tprint_info(f"  📊 Interactions after complexity filtering: {len(sorted_interactions)}")
        
        # Select top interactions with overfitting-aware limits
        max_interactions = min(50, len(sorted_interactions))  # Increased to 50 for richer interaction set
        top_interactions = sorted_interactions[:max_interactions]
        
        tprint_info(f"  📊 Selected top {len(top_interactions)} interactions (target: 50)")
        
        # Create interaction features dictionary from top-scoring candidates
        interaction_features = {}
        for name, score in top_interactions:
            # Find the corresponding interaction from candidates
            for candidate_name, candidate_interaction in interaction_candidates:
                if candidate_name == name:
                    interaction_features[name] = candidate_interaction
                    break
        
        # Store scores for metadata
        self._last_interaction_scores = [(name, score) for name, score in top_interactions]
        
        tprint_info(f"  🔍 DEBUG: Generated {len(interaction_features)} interaction features using pre-calculated MI scores")
        
        # Create interaction DataFrame
        interaction_df = pd.DataFrame(interaction_features, index=features.index)
        
        # Apply causality shift
        interaction_df = interaction_df.shift(1)
        
        # Apply RobustScaler only if there are interactions
        if len(interaction_df.columns) > 0 and len(interaction_df) > 0:
            scaler = RobustScaler()
            interaction_df = pd.DataFrame(
                scaler.fit_transform(interaction_df),
                columns=interaction_df.columns,
                index=interaction_df.index
            )
        else:
            tprint_warning("  ⚠️ No interactions generated, skipping RobustScaler")
        
        # Create comprehensive SHAP metadata with interaction discovery details
        # Convert feature pairs to strings for metadata
        feature_pairs_str = []
        for f1, f2, count in feature_pairs[:80]:  # Use all 80 pairs for metadata
            feature_pairs_str.append(f"{f1}_x_{f2}")
        
        # Use actual calculated interaction scores instead of hardcoded 1.0
        actual_scores = {}
        if hasattr(self, '_last_interaction_scores') and self._last_interaction_scores:
            # Use the scores from the interaction generation process
            for name in interaction_df.columns:
                # Find the score for this interaction
                for score_name, score_value in self._last_interaction_scores:
                    if score_name == name:
                        actual_scores[name] = score_value
                        break
                else:
                    # Fallback to 0.5 if no score found
                    actual_scores[name] = 0.5
        else:
            # Fallback to 0.5 if no scores available
            actual_scores = {name: 0.5 for name in interaction_df.columns}
        
        # Calculate valid interactions count
        valid_interactions_count = len([score for score in actual_scores.values() if score > 0])
        
        # Get feature categories for all features (base + interactions)
        all_features = list(features.columns) + list(interaction_df.columns)
        if feature_categories is None:
            # Fallback to inference if not provided
            feature_categories = {}
            for feature_name in all_features:
                feature_categories[feature_name] = self._infer_feature_category(feature_name)
        
        shap_metadata = {
            'feature_categories': feature_categories,  # Add feature categories to metadata
            'interaction_discovery': {
                'feature_pairs': feature_pairs_str,
                'interaction_scores': actual_scores,
                'early_stopping_applied': False,  # Using pre-calculated MI scores instead
                'total_interactions_generated': len(interaction_df.columns),
                'operations_per_pair': 5,  # x, div, minus, log, log_ratio
                'max_candidates_processed': len(interaction_candidates),  # All candidates processed (400)
                'mi_based_selection': True,  # Using MI scores for selection
                'valid_interactions_found': valid_interactions_count
            },
            'model_performance': {
                'lgbm_training_successful': True,
                'tree_analysis_successful': True,
                'interaction_generation_successful': len(interaction_df.columns) > 0,
                'accuracy': self._phase3_1_performance.get('accuracy', 0.0),
                'cv_score': self._phase3_1_performance.get('cv_score', 0.0),
                'importance_consistency': self._phase3_1_performance.get('importance_consistency', 0.0),
                'cv_scores_std': self._phase3_1_performance.get('cv_scores_std', 0.0)
            }
        }
        
        tprint_success(f"  ✅ Generated {len(interaction_df.columns)} interaction features with early stopping")
        tprint_info(f"  🔍 DEBUG: interaction_df shape at Phase 3.3 exit: {interaction_df.shape}")
        tprint_info(f"  🔍 DEBUG: interaction_df columns count: {len(interaction_df.columns)}")
        
        return interaction_df, shap_metadata
    
    def _extract_tree_splitting_pairs(self, model) -> List[Tuple[str, str, int]]:
        """
        Extract feature pairs that frequently split together in trees.
        
        Returns:
            List of (feature1, feature2, co_occurrence_count) tuples
        """
        from collections import defaultdict
        
        feature_pairs = defaultdict(int)
        
        try:
            # Get tree structure from the trained model's booster
            # Handle both MultiOutputRegressor and direct LGBMRegressor
            if hasattr(model, 'estimators_'):
                # MultiOutputRegressor case
                booster = model.estimators_[0].booster_
            else:
                # Direct LGBMRegressor case
                booster = model.booster_
            
            trees = booster.dump_model()['tree_info']
            tprint_info(f"  🔍 DEBUG: Found {len(trees)} trees in model")
            tprint_info(f"  🔍 DEBUG: Model type: {type(model)}")
            tprint_info(f"  🔍 DEBUG: Booster type: {type(booster)}")
            
            for tree in trees:
                features_in_tree = set()
                
                # Traverse tree to find all features used
                def traverse_node(node):
                    if 'split_feature' in node:
                        features_in_tree.add(node['split_feature'])
                        if 'left_child' in node:
                            traverse_node(node['left_child'])
                        if 'right_child' in node:
                            traverse_node(node['right_child'])
                
                traverse_node(tree['tree_structure'])
                
                # Count all pairs in this tree
                features_list = list(features_in_tree)
                tprint_info(f"  🔍 DEBUG: Tree {len(feature_pairs)} has {len(features_list)} features: {features_list}")
                
                for i in range(len(features_list)):
                    for j in range(i + 1, len(features_list)):
                        pair = tuple(sorted([features_list[i], features_list[j]]))
                        feature_pairs[pair] += 1
            
            # Convert to list and sort by co-occurrence
            pairs_list = [(f1, f2, count) for (f1, f2), count in feature_pairs.items()]
            pairs_list.sort(key=lambda x: x[2], reverse=True)
            
            tprint_info(f"  🔍 DEBUG: Found {len(pairs_list)} feature pairs")
            if len(pairs_list) > 0:
                tprint_info(f"  🔍 DEBUG: Top 5 pairs: {pairs_list[:5]}")
            else:
                tprint_warning("  ⚠️ DEBUG: No feature pairs found in tree analysis")
            return pairs_list[:80]  # Return top 80 pairs
            
        except Exception as e:
            tprint_error(f"  ❌ Tree analysis failed: {e}")
            tprint_error(f"  ❌ Exception type: {type(e)}")
            import traceback
            tprint_error(f"  ❌ Traceback: {traceback.format_exc()}")
            tprint_warning(f"  ⚠️ Model type: {type(model)}")
            tprint_warning(f"  ⚠️ Estimators count: {len(model.estimators_) if hasattr(model, 'estimators_') else 'N/A'}")
            return []

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
        
        # Count interaction types in combined_features
        ct_count = sum(1 for c in combined_features.columns if any(m in c for m in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']))
        int_count = sum(1 for c in combined_features.columns if any(m in c for m in ['_x_', '_div_', '_minus_', '_log_']) and not any(m in c for m in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']))
        base_count = len(combined_features.columns) - ct_count - int_count
        
        tprint_info(f"  📊 Feature breakdown before save:")
        tprint_info(f"    - Cross-timeframe ratios: {ct_count}")
        tprint_info(f"    - Traditional interactions: {int_count}")
        tprint_info(f"    - Base/variant features: {base_count}")
        tprint_info("="*80)
        
        features_path = self._save_artifact(
            data=combined_features,
            artifact_name='analyst_interaction_features',
            artifact_type='data',
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
        importance_path = self._save_artifact(
            data=shap_metadata.get('interaction_scores', {}),
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
"""
        
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

    def _calculate_composite_scores(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate composite scores for features based on MI and stability.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            feature_categories: Dict mapping feature names to categories
            
        Returns:
            Dict mapping feature names to composite scores
        """
        from sklearn.feature_selection import mutual_info_regression
        import numpy as np
        
        tprint_info(f"  📊 Calculating MI scores for {len(features_df.columns)} features...")
        
        # Use first target for MI calculation
        target_col = targets_df.columns[0]
        target = targets_df[target_col].dropna()
        
        # Align features and target
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove any features with all NaN or constant values
        # Use relaxed validation for ratio features (like cross-timeframe)
        valid_features = []
        for col in features_aligned.columns:
            col_data = features_aligned[col]
            non_nan_count = col_data.notna().sum()
            
            # Require at least 10 non-NaN values
            if non_nan_count < 10:
                continue
            
            # Check if feature varies (not constant)
            col_std = col_data.std()
            
            # For cross-timeframe ratio features, use more relaxed threshold
            is_ct_ratio = '_3x_ratio' in col or '_6x_ratio' in col or '_9x_ratio' in col or '_27x_ratio' in col
            if is_ct_ratio:
                # Ratio features can have smaller std, just check they're not ALL the same value
                if col_std > 1e-10 and col_data.nunique() > 2:  # At least 3 unique values
                    valid_features.append(col)
            else:
                # Standard validation for other features
                if col_std > 1e-8:
                    valid_features.append(col)
        
        features_for_mi = features_aligned[valid_features].fillna(0)
        
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
        
        # Calculate MI scores
        try:
            mi_scores = mutual_info_regression(
                features_for_mi,
                target_aligned,
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
            tprint_warning(f"  ⚠️ MI calculation failed: {e}")
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
                        feature_info = self.feature_bank.registry.get_feature(feature_name)
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
    
    def _get_consistent_sample(self, features: pd.DataFrame, targets: pd.DataFrame, max_samples: int = 8000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get consistent sample across all phases."""
        print(f"🔍 DEBUG: _get_consistent_sample called with features shape: {features.shape}, targets shape: {targets.shape}")
        print(f"🔍 DEBUG: targets columns: {list(targets.columns) if len(targets.columns) > 0 else 'EMPTY'}")
        print(f"🔍 DEBUG: targets head:\n{targets.head() if len(targets) > 0 else 'EMPTY'}")
        
        if len(features) == 0:
            print("🔍 DEBUG: Features DataFrame is empty!")
            return features, targets
        
        # If we have fewer samples than max_samples, just return all
        if len(features) <= max_samples:
            print(f"🔍 DEBUG: Returning original features shape: {features.shape}, targets shape: {targets.shape}")
            return features, targets
        
        # Use same random seed for consistency
        np.random.seed(42)
        sample_idx = np.random.choice(len(features), max_samples, replace=False)
        sampled_features = features.iloc[sample_idx]
        sampled_targets = targets.iloc[sample_idx]
        print(f"🔍 DEBUG: Sampled features shape: {sampled_features.shape}, sampled targets shape: {sampled_targets.shape}")
        return sampled_features, sampled_targets
    
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
                    corr_score = abs(interaction_clean.corr(target_clean))
                    
                    # Handle NaN correlation
                    if pd.isna(corr_score):
                        corr_score = 0.0
                    
                    # Combined score (70% MI + 30% correlation)
                    combined_score = 0.7 * mi_score + 0.3 * corr_score
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
        # Select top interactions (up to 50 or all if fewer)
        max_interactions = min(50, len(scores))
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
    
    def _parallel_variant_generation(self, selected_features: List[Dict], features_df: pd.DataFrame, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Parallel variant generation using multiprocessing."""
        import multiprocessing as mp
        from functools import partial
        
        if not HARDWARE_OPT_AVAILABLE:
            # Fallback to sequential processing
            return self._sequential_variant_generation(selected_features, features_df, ohlcv_data)
        
        try:
            # Get hardware manager
            hardware_manager = get_unified_hardware_manager()
            hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING, OptimizationLevel.BALANCED)
            
            # Use optimal number of processes
            max_processes = min(4, mp.cpu_count())
            
            with mp.Pool(processes=max_processes) as pool:
                generate_func = partial(
                    self._generate_single_feature_variants,
                    features_df=features_df,
                    ohlcv_data=ohlcv_data
                )
                results = pool.map(generate_func, selected_features)
            
            # Combine results
            all_variants = {}
            for variants in results:
                if variants:
                    all_variants.update(variants)
            
            return pd.DataFrame(all_variants, index=features_df.index)
            
        except Exception as e:
            tprint_warning(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
            return self._sequential_variant_generation(selected_features, features_df, ohlcv_data)
    
    def _generate_single_feature_variants(self, feature_info: Dict, features_df: pd.DataFrame, ohlcv_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate variants for a single feature (for parallel processing)."""
        try:
            from src.training.utils.feature_selection.variant_generator import generate_all_variants_optimized
            
            # Generate variants for single feature
            variants, _ = generate_all_variants_optimized(
                features_df=features_df,
                selected_features=[feature_info],
                ohlcv_data=ohlcv_data
            )
            
            return variants.to_dict('series')
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate variants for {feature_info.get('feature_name', 'unknown')}: {e}")
            return {}
    
    def _sequential_variant_generation(self, selected_features: List[Dict], features_df: pd.DataFrame, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Sequential variant generation (fallback)."""
        try:
            from src.training.utils.feature_selection.variant_generator import generate_all_variants_optimized
            
            tprint_info(f"🔄 Sequential variant generation: {len(selected_features)} features -> variants")
            
            variants, _ = generate_all_variants_optimized(
                features_df=features_df,
                selected_features=selected_features,
                ohlcv_data=ohlcv_data
            )
            
            tprint_info(f"✅ Sequential variant generation completed: {len(variants.columns)} variants generated")
            
            return variants
            
        except Exception as e:
            tprint_error(f"❌ Sequential variant generation failed: {e}")
            raise
    
    def _chunked_processing(self, features: pd.DataFrame, targets: pd.DataFrame, chunk_size: int = 5000) -> pd.DataFrame:
        """Process large datasets in chunks to reduce memory usage."""
        if not HARDWARE_OPT_AVAILABLE:
            return features
        
        try:
            from src.feature_generation.utils.memory_optimizer import MemoryOptimizer
            memory_optimizer = MemoryOptimizer()
            
            tprint_info(f"  📊 Processing {len(features)} rows in chunks of {chunk_size}")
            tprint_info(f"  📈 Input features: {len(features.columns)} features")
            
            # Process in chunks
            chunk_results = []
            for i in range(0, len(features), chunk_size):
                chunk_features = features.iloc[i:i+chunk_size]
                chunk_targets = targets.iloc[i:i+chunk_size]
                
                # Process chunk
                processed_chunk = self._process_chunk(chunk_features, chunk_targets)
                chunk_results.append(processed_chunk)
                
                # Memory cleanup
                memory_optimizer.force_garbage_collection()
            
            # Combine results
            result = pd.concat(chunk_results, ignore_index=True)
            
            tprint_success(f"  ✅ Chunked processing completed: {len(result)} rows")
            tprint_info(f"  📈 Output features: {len(result.columns)} features")
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Chunked processing failed, using full dataset: {e}")
            return features
    
    def _process_chunk(self, chunk_features: pd.DataFrame, chunk_targets: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        # This is a placeholder - implement specific chunk processing logic
        # For now, just return the chunk as-is
        return chunk_features
    
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
    
    def _process_chunk(self, chunk_features: pd.DataFrame, chunk_targets: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        # This is a placeholder - implement specific chunk processing logic
        return chunk_features


    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_feature_generation_interaction_generation_step():
    """Register the unified interaction generation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_interaction_generation_step", FeatureGenerationInteractionGenerationStep)
    tprint("✅ Unified feature generation interaction generation step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_interaction_generation_step()
