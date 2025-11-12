"""
Feature Generation Final Feature Selection Step

This step performs final feature selection from all previously generated and selected features.
It creates multiple optimized feature sets (60, 50, 40 features) for different model configurations
and generates comprehensive SHAP values and selection metadata.

Features:
- Combines features from interaction generation steps
- Performs final feature ranking and selection
- Creates multiple feature sets (60, 50, 40 features)
- Generates SHAP values for interpretability
- Comprehensive selection metadata and reporting
"""

import asyncio
import logging
import warnings
import pandas as pd
import numpy as np

from src.utils.tprint import tprint

# Fix NumPy compatibility for older libraries
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path
import json

# Import BaseStep and step registry
from src.training.steps.base_step import BaseStep

# Import feature selection component
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent
)

# Import VectorBT optimization tools
from src.feature_generation.utils.vectorbt_rolling_optimizer import (
    VectorBTRollingOptimizer,
    get_vectorbt_rolling_optimizer
)

# Import unified vectorization manager
from src.feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager,
    VectorizationConfig,
    get_unified_vectorization_manager
)

# Note: Hardware optimization components are optional for feature selection

# Import hardware optimization tools
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager,
    HardwareConfig,
    WorkloadType,
    OptimizationLevel
)

# Import additional hardware optimization components
from src.utils.hardware.adaptive_optimization_engine import (
    AdaptiveOptimizationEngine,
    LearningAlgorithm
)

# Import CMI complementarity components for Tactician mode
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
    tprint("✅ CMI complementarity components loaded successfully")
except ImportError as e:
    CMI_COMPLEMENTARITY_AVAILABLE = False
    CMIComplementarityScorer = None
    CMIComplementarityConfig = None
    AnalystSideInfoHandler = None
    tprint_warning(f"⚠️ CMI complementarity components not available: {e}")

# Import OptimizationStrategy from the correct location
from src.feature_generation.core.optimization_strategies import (
    OptimizationStrategy,
    ConservativeOptimizationStrategy,
    BalancedOptimizationStrategy,
    AggressiveOptimizationStrategy
)

from src.utils.hardware.advanced_cpu_optimizer import (
    AdvancedM1CPUOptimizer,
    WorkloadProfile,
    CoreType
)

from src.utils.hardware.enhanced_gpu_manager import (
    EnhancedM1GPUManager,
    GPUOperationType
)

from src.utils.hardware.advanced_memory_optimizer import (
    AdvancedM1MemoryOptimizer,
    MemoryStrategy
)

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, configure_tprint, TPrintConfig, LogLevel
from src.utils.artifact_manager import ArtifactManager

# Configure tprint for minimal mode to reduce overhead
configure_tprint(TPrintConfig(
    use_colors=False,
    output_to_file=False,
    log_to_python_logger=False,
    integrate_with_logging=False,
    min_log_level=LogLevel.INFO,
    enable_lazy_evaluation=True,
    cache_timestamps=True
))

logger = logging.getLogger(__name__)

# Define target column names once to avoid hardcoding throughout the codebase
# Updated to include new simplified target structure (target_long, target_short)
TARGET_COLUMN_NAMES = ['target', 'label', 'return', 'price_target_vol_normalized', 'target_long', 'target_short']


class FeatureGenerationFinalFeatureSelectionStep(BaseStep):
    """
    Final feature selection step for the feature generation pipeline.

    This step combines all previously selected features and performs final selection
    to create optimized feature sets for model training.
    """

    def __init__(self, step_name: str = "feature_generation_final_feature_selection_step"):
        """Initialize the final feature selection step."""
        super().__init__(step_name)
        self.selection_component: Optional[FinalFeatureSelectionComponent] = None
        
        # Initialize VectorBT optimization components
        self.vectorization_manager: Optional[UnifiedVectorizationManager] = None
        self.rolling_optimizer: Optional[VectorBTRollingOptimizer] = None
        self.optimization_enabled: bool = True
        
        # Initialize hardware optimization components
        self.hardware_manager: Optional[UnifiedHardwareManager] = None
        self.adaptive_engine: Optional[AdaptiveOptimizationEngine] = None
        self.cpu_optimizer: Optional[AdvancedM1CPUOptimizer] = None
        self.gpu_manager: Optional[EnhancedM1GPUManager] = None
        self.memory_optimizer: Optional[AdvancedM1MemoryOptimizer] = None
        self.hardware_optimization_enabled: bool = True
        
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
            tprint_info("✅ CMI complementarity components initialized for final feature selection")
        else:
            self.cmi_scorer = None
            self.analyst_handler = None
            tprint_warning("⚠️ CMI complementarity components not available")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final feature selection.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Timeframe
                - execution_mode: Execution mode (light, full, etc.)
                - feature_set_sizes: List of feature set sizes to create [60, 50, 40]
                - selection_config: Optional selection configuration overrides

        Returns:
            Dict containing execution results and artifacts
        """
        import time
        step_start = time.time()
        
        try:
            # LOW PRIORITY FIX: Add execution mode validation
            execution_mode = config.get('execution_mode', 'blank')
            valid_modes = ['blank', 'light', 'full']
            
            if execution_mode not in valid_modes:
                self.logger.error(f"❌ Invalid execution mode: '{execution_mode}'. Valid modes: {valid_modes}")
                raise ValueError(f"Invalid execution mode: '{execution_mode}'. Valid modes: {valid_modes}")
            else:
                self.logger.info(f"✅ Execution mode validation passed: {execution_mode}")
            
            # Set context for artifact storage with proper symbol/exchange/timeframe
            self.set_context(
                symbol=config.get('symbol', 'UNKNOWN'),
                exchange=config.get('exchange', 'binance'),
                timeframe=config.get('timeframe', '15m'),
                direction=config.get('direction', 'long'),
                model=config.get('execution_mode', 'analyst')
            )
            
            tprint_info(f"🎯 Starting {self.step_name} execution...")
            tprint_info(f"📊 Context: {config.get('symbol', 'UNKNOWN')}/{config.get('exchange', 'binance')} [{config.get('timeframe', '15m')}] {config.get('direction', 'long')}/{config.get('execution_mode', 'analyst')}")

            # Get required data from previous steps
            # Look for artifacts created by labeling integration step
            t0 = time.time()
            tprint_info("⏱️ [1/10] Loading labeled data...")
            
            # CRITICAL FIX: Load the FULL labeled_data from versioned artifacts
            # The generic _get_artifact() was loading a 300-row subset
            # We need the full dataset with execution mode filtering applied
            from src.utils.versioned_artifacts.store import VersionedArtifactStore
            
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            execution_mode = config.get('execution_mode', 'blank')
            
            # HIGH PRIORITY FIX: Add explicit debug logging for data source
            tprint_info(f"🔍 DEBUG: Loading labeled data with execution_mode={execution_mode}")
            tprint_info(f"🔍 DEBUG: Data source priority: 1) Versioned store 2) Generic artifact")
            tprint_info(f"🔍 DEBUG: Expected full dataset size for blank mode: ~180 days of data")
            
            # CRITICAL FIX: Load BOTH labeled_data (targets) AND generated_features_15m (large feature set)
            # Use execution_mode to determine the correct store path
            store_path = f'versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_analyst'
            tprint_info(f"🔍 DEBUG: Store path: {store_path}")
            tprint_info(f"🔍 DEBUG: Execution mode: {execution_mode}")
            labeled_df = None
            large_features_df = None
            data_source = "unknown"
            
            try:
                tprint_info(f"🔍 DEBUG: Loading BOTH labeled_data AND generated_features from: {store_path}")
                store = VersionedArtifactStore(store_path)
                
                # STEP 1: Load labeled_data (targets) - needed for feature selection
                labeled_versions = [v for v in store.list_versions() if 'labeled_data' in v.lower() or 'labeled_dataframe' in v.lower()]
                tprint_error(f"🔍 CRITICAL DEBUG: Found labeled_data versions: {labeled_versions}")
                tprint_error(f"🔍 CRITICAL DEBUG: Total versions found: {len(labeled_versions)}")
                
                if labeled_versions:
                    latest_labeled = sorted(labeled_versions)[-1]
                    tprint_error(f"📂 CRITICAL: Loading labeled_data from: {latest_labeled}")
                    labeled_df = store.get_view(latest_labeled).materialize()
                    tprint_error(f"✅ CRITICAL: Loaded labeled_data shape: {labeled_df.shape}")
                    tprint_error(f"✅ CRITICAL: Loaded labeled_data time range: {labeled_df.index.min()} to {labeled_df.index.max()}")
                    
                    if len(labeled_df) < 1000:
                        tprint_error(f"❌ CRITICAL ISSUE: Loaded dataset is SMALL ({len(labeled_df)} rows)!")
                        tprint_error(f"❌ This is the 300-row bottleneck source!")
                    else:
                        tprint_success(f"✅ CRITICAL SUCCESS: Loaded LARGE dataset ({len(labeled_df)} rows)!")
                else:
                    # FALLBACK: Try UNKNOWN store (common issue with labeling integration)
                    unknown_store_path = f'versioned_artifacts/UNKNOWN_{exchange}_{timeframe}_{direction}_analyst'
                    tprint_warning(f"⚠️ No labeled_data in {store_path}, trying UNKNOWN store: {unknown_store_path}")
                    try:
                        unknown_store = VersionedArtifactStore(unknown_store_path)
                        unknown_labeled_versions = [v for v in unknown_store.list_versions() if 'labeled_data' in v.lower() or 'labeled_dataframe' in v.lower()]
                        tprint_error(f"🔍 CRITICAL DEBUG: UNKNOWN store versions: {unknown_labeled_versions}")
                        tprint_error(f"🔍 CRITICAL DEBUG: UNKNOWN store total versions: {len(unknown_labeled_versions)}")
                        
                        if unknown_labeled_versions:
                            latest_labeled = sorted(unknown_labeled_versions)[-1]
                            tprint_error(f"📂 CRITICAL: Loading from UNKNOWN store: {latest_labeled}")
                            labeled_df = unknown_store.get_view(latest_labeled).materialize()
                            tprint_error(f"✅ CRITICAL: UNKNOWN store data shape: {labeled_df.shape}")
                            tprint_error(f"✅ CRITICAL: UNKNOWN store time range: {labeled_df.index.min()} to {labeled_df.index.max()}")
                            
                            if len(labeled_df) < 1000:
                                tprint_error(f"❌ CRITICAL: UNKNOWN store data is SMALL ({len(labeled_df)} rows)!")
                                tprint_error(f"❌ This confirms the 300-row bottleneck!")
                            else:
                                tprint_success(f"✅ CRITICAL SUCCESS: UNKNOWN store has LARGE dataset ({len(labeled_df)} rows)!")
                        else:
                            tprint_error(f"❌ CRITICAL: No labeled_data found in UNKNOWN store either!")
                    except Exception as e:
                        tprint_error(f"❌ CRITICAL: Failed to access UNKNOWN store: {e}")
                
                # STEP 2: Load generated_features_15m (large feature set) - the 170K+ rows we need
                feature_versions = [v for v in store.list_versions() if 'generated_features_15m' in v.lower()]
                tprint_info(f"🔍 DEBUG: Found generated_features versions: {feature_versions}")
                
                if feature_versions:
                    # Use the largest dataset (should be 173,434 rows)
                    latest_features = sorted(feature_versions)[-1]
                    tprint_info(f"📂 Loading LARGE generated_features from: {latest_features}")
                    large_features_df = store.get_view(latest_features).materialize()
                    tprint_success(f"✅ Loaded LARGE feature dataset: {large_features_df.shape}")
                    data_source = f"versioned_store:labeled={latest_labeled if labeled_df is not None else 'none'},features={latest_features}"
                
                # STEP 3: If we have both, use the large features as base and align with labeled targets
                if labeled_df is not None and large_features_df is not None:
                    tprint_info(f"🔗 Combining labeled targets with large feature set...")
                    
                    # DETAILED DEBUGGING: Show time periods and index details
                    tprint_info(f"📊 DETAILED INDEX ANALYSIS:")
                    tprint_info(f"   Labeled data: {labeled_df.shape}")
                    tprint_info(f"   - Index type: {type(labeled_df.index)}")
                    tprint_info(f"   - Index range: {labeled_df.index.min()} to {labeled_df.index.max()}")
                    tprint_info(f"   - Sample indices: {labeled_df.index[:5].tolist()} ... {labeled_df.index[-5:].tolist()}")
                    
                    tprint_info(f"   Generated features: {large_features_df.shape}")
                    tprint_info(f"   - Index type: {type(large_features_df.index)}")
                    tprint_info(f"   - Index range: {large_features_df.index.min()} to {large_features_df.index.max()}")
                    tprint_info(f"   - Sample indices: {large_features_df.index[:5].tolist()} ... {large_features_df.index[-5:].tolist()}")
                    
                    # Check for time period overlap
                    labeled_start, labeled_end = labeled_df.index.min(), labeled_df.index.max()
                    features_start, features_end = large_features_df.index.min(), large_features_df.index.max()
                    
                    overlap_start = max(labeled_start, features_start)
                    overlap_end = min(labeled_end, features_end)
                    
                    tprint_info(f"📅 TIME PERIOD ANALYSIS:")
                    tprint_info(f"   Labeled data period: {labeled_start} to {labeled_end}")
                    tprint_info(f"   Features period: {features_start} to {features_end}")
                    tprint_info(f"   Theoretical overlap: {overlap_start} to {overlap_end}")
                    
                    if overlap_start <= overlap_end:
                        overlap_days = (overlap_end - overlap_start).days
                        tprint_info(f"   Overlap duration: {overlap_days} days")
                        expected_samples = overlap_days * 96  # 96 samples per day for 15m
                        tprint_info(f"   Expected samples in overlap: ~{expected_samples}")
                    else:
                        tprint_error(f"   ❌ NO TIME OVERLAP! Labeled ends before features start or vice versa")
                    
                    # Find common index between labeled_data and large_features
                    common_index = labeled_df.index.intersection(large_features_df.index)
                    tprint_error(f"📊 CRITICAL: ACTUAL COMMON INDEX: {len(common_index)} rows")
                    tprint_error(f"📊 CRITICAL: Labeled data has {len(labeled_df)} rows")
                    tprint_error(f"📊 CRITICAL: Features data has {len(large_features_df)} rows")
                    
                    if len(common_index) > 0:
                        common_start, common_end = common_index.min(), common_index.max()
                        tprint_error(f"   CRITICAL: Common index range: {common_start} to {common_end}")
                        tprint_error(f"   CRITICAL: Common index sample: {common_index[:10].tolist()}")
                    
                    # Check for potential issues
                    if len(common_index) < 1000:
                        tprint_error(f"🔍 CRITICAL: DEBUGGING LOW INTERSECTION ({len(common_index)} rows):")
                        tprint_error(f"   This is the ROOT CAUSE of the 300-row bottleneck!")
                        
                        # Check if indices are different types or formats
                        labeled_sample = labeled_df.index[:100]
                        features_sample = large_features_df.index[:100]
                        
                        # Check for exact matches in samples
                        exact_matches = labeled_sample.intersection(features_sample)
                        tprint_info(f"   Exact matches in first 100: {len(exact_matches)}")
                        
                        # Check if there are timezone or format differences
                        if len(labeled_sample) > 0 and len(features_sample) > 0:
                            tprint_info(f"   Labeled sample[0]: {labeled_sample[0]} (type: {type(labeled_sample[0])})")
                            tprint_info(f"   Features sample[0]: {features_sample[0]} (type: {type(features_sample[0])})")
                            
                            # Check if they're close in time but not exact matches
                            if hasattr(labeled_sample[0], 'tz') and hasattr(features_sample[0], 'tz'):
                                tprint_info(f"   Labeled timezone: {labeled_sample[0].tz}")
                                tprint_info(f"   Features timezone: {features_sample[0].tz}")
                    
                    if len(common_index) > 1000:  # Reasonable threshold
                        # CRITICAL FIX: Use FULL large features dataset, not just common_index!
                        tprint_error(f"🔧 FIXING TEMPORAL FILTERING BOTTLENECK:")
                        tprint_error(f"   OLD APPROACH: Filter to common_index ({len(common_index)} rows)")
                        tprint_error(f"   NEW APPROACH: Use FULL large_features_df ({len(large_features_df)} rows)")
                        
                        # Use the FULL large features dataset as base
                        combined_df = large_features_df.copy()
                        
                        # Add target columns from labeled_data where available (reindex to match)
                        original_labeled_df = labeled_df  # Save reference to original labeled data
                        target_cols = [col for col in original_labeled_df.columns if 'target' in col.lower() or col == 'price_target_vol_normalized']
                        
                        if target_cols:
                            tprint_info(f"🎯 Adding target columns to FULL dataset: {target_cols}")
                            for col in target_cols:
                                # Reindex targets to match the full features dataset
                                aligned_targets = original_labeled_df[col].reindex(large_features_df.index)
                                combined_df[col] = aligned_targets
                                non_null_count = aligned_targets.notna().sum()
                                tprint_info(f"   {col}: {non_null_count}/{len(aligned_targets)} non-null values ({non_null_count/len(aligned_targets)*100:.1f}%)")
                        else:
                            # Add target from labeled_data (assume last column is target)
                            tprint_info("🎯 Adding target from last column of labeled_data to FULL dataset")
                            aligned_targets = original_labeled_df.iloc[:, -1].reindex(large_features_df.index)
                            combined_df['price_target_vol_normalized'] = aligned_targets
                            non_null_count = aligned_targets.notna().sum()
                            tprint_info(f"   price_target_vol_normalized: {non_null_count}/{len(aligned_targets)} non-null values ({non_null_count/len(aligned_targets)*100:.1f}%)")
                        
                        labeled_df = combined_df  # Replace with FULL combined dataset
                        tprint_success(f"✅ FIXED: Now using FULL dataset: {labeled_df.shape} (was {len(common_index)} rows)")
                        tprint_success(f"✅ Temporal filtering bottleneck RESOLVED!")
                    else:
                        tprint_error(f"❌ Too few common indices ({len(common_index)}) between targets and features")
                        tprint_error(f"   This suggests a fundamental mismatch in time periods or index formats")
                        # Fall back to labeled_df only
                        tprint_warning("⚠️ Falling back to labeled_df only due to index mismatch")
                        
            except Exception as e:
                tprint_error(f"❌ Failed to load from versioned store: {e}")
                data_source = "fallback_to_generic_artifact_after_error"
            
            # REMOVED: Generic artifact fallback (was causing 300-row bottleneck)
            # The generic artifact contains pre-filtered small datasets
            # We must use versioned stores only for full datasets
            
            # FINAL VALIDATION: Log the data source and size
            tprint_info(f"🔍 DEBUG: Final data source: {data_source}")
            tprint_info(f"🔍 DEBUG: Final dataset size: {labeled_df.shape if labeled_df is not None else 'None'}")
            tprint_info(f"🔍 DEBUG: Execution mode: {execution_mode}")
            
            if labeled_df is None:
                raise ValueError("Failed to load labeled_data from any source")
            
            # CRITICAL DEBUG: Explicit row count and time range analysis
            tprint_error("=" * 80)
            tprint_error("🔍 CRITICAL: LOADED LABELED DATA ANALYSIS")
            tprint_error("=" * 80)
            tprint_error(f"📊 LABELED_DF LOADED:")
            tprint_error(f"   Rows: {len(labeled_df)}")
            tprint_error(f"   Columns: {labeled_df.shape[1]}")
            tprint_error(f"   Time range: {labeled_df.index.min()} to {labeled_df.index.max()}")
            
            # Calculate time span
            time_span_days = (labeled_df.index.max() - labeled_df.index.min()).days
            expected_samples_at_15m = time_span_days * 96  # 96 samples per day for 15m timeframe
            tprint_error(f"   Time span: {time_span_days} days")
            tprint_error(f"   Expected samples at 15m: ~{expected_samples_at_15m}")
            tprint_error(f"   Actual samples: {len(labeled_df)}")
            tprint_error(f"   Sample density: {len(labeled_df) / expected_samples_at_15m * 100:.1f}%")
            
            if len(labeled_df) < 1000:
                tprint_error(f"❌ CRITICAL: Only {len(labeled_df)} rows loaded!")
                tprint_error(f"❌ At 96 samples/day, this is only {len(labeled_df) / 96:.1f} days of data")
                tprint_error(f"❌ This is the SOURCE of the 300-row bottleneck!")
            else:
                tprint_success(f"✅ GOOD: {len(labeled_df)} rows loaded ({time_span_days} days)")
            tprint_error("=" * 80)
            
            targets = self._get_artifact('labeling_metadata')
            tprint_info(f"✅ Loaded in {time.time()-t0:.2f}s")
            
            # Debug: Check what columns are in labeled_df
            if labeled_df is not None:
                tprint_info(f"🔍 DEBUG: labeled_df shape: {labeled_df.shape}")
                tprint_info(f"🔍 DEBUG: labeled_df columns: {list(labeled_df.columns)}")
                target_cols_present = [col for col in labeled_df.columns if 'target' in col.lower()]
                tprint_info(f"🔍 DEBUG: Target columns in labeled_df: {target_cols_present}")

            if labeled_df is None or targets is None:
                raise ValueError("Required artifacts 'labeled_data' and 'labeling_metadata' not found")

            # Collect features from previous steps
            t0 = time.time()
            tprint_info("⏱️ [2/10] Collecting features from previous steps...")
            features_data = self._collect_features_from_previous_steps()
            tprint_info(f"✅ Collected in {time.time()-t0:.2f}s")
            
            # CRITICAL DEBUG: Log what we collected with explicit row counts
            tprint_error("=" * 80)
            tprint_error("🔍 CRITICAL: FEATURES COLLECTED FROM PREVIOUS STEPS")
            tprint_error("=" * 80)
            for key, data in features_data.items():
                if data is not None and hasattr(data, 'shape'):
                    tprint_error(f"📊 {key.upper()}:")
                    tprint_error(f"   Rows: {len(data)}")
                    tprint_error(f"   Columns: {data.shape[1]}")
                    if hasattr(data, 'index'):
                        try:
                            tprint_error(f"   Time range: {data.index.min()} to {data.index.max()}")
                            time_span_days = (data.index.max() - data.index.min()).days
                            tprint_error(f"   Time span: {time_span_days} days")
                            tprint_error(f"   Expected samples at 15m: ~{time_span_days * 96}")
                            
                            if len(data) < 1000:
                                tprint_error(f"   ❌ SMALL: Only {len(data)} rows ({len(data) / 96:.1f} days)")
                            else:
                                tprint_success(f"   ✅ LARGE: {len(data)} rows ({time_span_days} days)")
                        except (AttributeError, TypeError):
                            # Index is not datetime, skip time range analysis
                            if len(data) < 1000:
                                tprint_error(f"   ❌ SMALL: Only {len(data)} rows")
                            else:
                                tprint_success(f"   ✅ LARGE: {len(data)} rows")
                else:
                    tprint_error(f"📊 {key.upper()}: {type(data)} (None or no shape)")
            tprint_error("=" * 80)

            # Combine all features
            t0 = time.time()
            tprint_info("⏱️ [3/10] Combining features...")
            combined_features_df = self._combine_features(features_data, labeled_df)
            tprint_info(f"✅ Combined {combined_features_df.shape} in {time.time()-t0:.2f}s")
            
            # CRITICAL DEBUGGING: Check if we got the full dataset
            tprint_error(f"🔍 CRITICAL CHECK: Combined features dataset size")
            tprint_error(f"   Shape: {combined_features_df.shape}")
            tprint_error(f"   Time range: {combined_features_df.index.min()} to {combined_features_df.index.max()}")
            tprint_error(f"   Expected: ~16,000 rows (full dataset)")
            tprint_error(f"   Actual: {len(combined_features_df)} rows")
            
            if len(combined_features_df) < 10000:
                tprint_error(f"❌ STILL GETTING SMALL DATASET! Need to investigate further.")
                tprint_error(f"   This means the _combine_features fix didn't work as expected.")
            else:
                tprint_success(f"✅ SUCCESS: Got large dataset as expected!")

            if combined_features_df.empty:
                raise ValueError("No features available for final selection")

            # Setup selection configuration
            t0 = time.time()
            tprint_info("⏱️ [4/10] Setting up selection config...")
            selection_config = self._setup_selection_config(config)
            tprint_info(f"✅ Setup in {time.time()-t0:.2f}s")

            # Initialize optimization components
            t0 = time.time()
            tprint_info("⏱️ [5/10] Initializing optimization components...")
            await self._initialize_optimization_components(config)
            await self._initialize_hardware_optimization_components(config)
            tprint_info(f"✅ Initialized in {time.time()-t0:.2f}s")

            # Initialize selection component
            t0 = time.time()
            tprint_info("⏱️ [6/10] Initializing selection component...")
            self.selection_component = FinalFeatureSelectionComponent(selection_config)
            tprint_info(f"✅ Initialized in {time.time()-t0:.2f}s")

            # Perform feature selection for different set sizes
            t0 = time.time()
            tprint_info("⏱️ [7/10] Performing multi-size selection...")
            feature_sets = self._perform_multi_size_selection(combined_features_df, targets, config)
            tprint_info(f"✅ Selection completed in {time.time()-t0:.2f}s")

            # Perform enhanced analysis on the largest feature set
            if feature_sets:
                t0 = time.time()
                tprint_info("⏱️ [8/10] Performing enhanced analysis...")
                largest_set_key = max([k for k in feature_sets.keys() if k.startswith('selected_features_')], 
                                     key=lambda x: int(x.split('_')[-1]))
                largest_features = feature_sets[largest_set_key]
                
                # Separate features from targets for analysis
                feature_cols = [col for col in combined_features_df.columns 
                               if col not in TARGET_COLUMN_NAMES + ['timestamp']]
                X = combined_features_df[feature_cols]
                y = combined_features_df[targets.name] if hasattr(targets, 'name') else combined_features_df.iloc[:, -1]
                
                # Perform comprehensive analysis
                enhanced_analysis = self._perform_enhanced_analysis(X, y, largest_features)
                
                # Store analysis results in feature_sets for report generation
                feature_sets['enhanced_analysis'] = enhanced_analysis
                tprint_info(f"✅ Enhanced analysis completed in {time.time()-t0:.2f}s")

            # Generate SHAP values for interpretability
            t0 = time.time()
            tprint_info("⏱️ [9/10] Generating SHAP values...")
            shap_values = self._generate_shap_values(feature_sets, combined_features_df, targets, config)
            tprint_info(f"✅ SHAP values generated in {time.time()-t0:.2f}s")

            # Generate artifacts
            t0 = time.time()
            tprint_info("⏱️ [10/10] Generating and saving artifacts...")
            artifacts = self._generate_artifacts(feature_sets, shap_values, config, combined_features_df)

            # Create comprehensive outcome report
            outcome_report = self._create_outcome_report(feature_sets, shap_values, config)

            # Save artifacts
            saved_artifacts = []
            for artifact_name, artifact_data in artifacts.items():
                # Determine if this should go to versioned artifacts (HDF5)
                is_feature_artifact = (
                    artifact_name.startswith('selected_features_') or 
                    artifact_name.startswith('selected_feature_dataframe_')
                )
                
                # For feature artifacts, use data_category='features' to trigger HDF5 storage
                # The artifact name will be clean (e.g., "selected_feature_dataframe_60")
                # and the versioned artifacts system will add timestamp and context
                data_category = 'features' if is_feature_artifact else None
                
                artifact_path = self._save_artifact(
                    artifact_data,
                    artifact_name,
                    artifact_type="data",
                    data_category=data_category
                )
                saved_artifacts.append({
                    'name': artifact_name,
                    'path': artifact_path,
                    'type': 'data',
                    'versioned': is_feature_artifact
                })

            # Save outcome report (pickle format)
            report_path = self._save_artifact(
                outcome_report,
                "final_feature_selection_outcome_report",
                artifact_type="report"
            )
            
            # Generate and save markdown report
            markdown_report = self._generate_markdown_report(outcome_report, feature_sets, shap_values, config)
            markdown_path = self._save_markdown_report(markdown_report, "final_feature_selection_outcome_report")
            tprint_info(f"✅ Artifacts saved in {time.time()-t0:.2f}s")

            # Calculate metrics
            tprint_info("📊 Calculating final metrics...")
            metrics = self._calculate_metrics(feature_sets, shap_values, config)
            
            # Add optimization performance metrics
            optimization_metrics = self._get_optimization_metrics()
            metrics.update(optimization_metrics)

            total_time = time.time() - step_start
            tprint_info(f"🎉 Step completed successfully in {total_time:.2f}s")
            
            execution_result = {
                'success': True,
                'artifacts': saved_artifacts,
                'metrics': metrics,
                'feature_sets': {k: len(v) for k, v in feature_sets.items()},
                'shap_summary': self._summarize_shap_values(shap_values),
                'outcome_report_path': report_path,
                'markdown_report_path': markdown_path,
                'execution_time': total_time,
                'optimization_enabled': self.optimization_enabled,
                'vectorization_stats': self._get_vectorization_stats()
            }

            tprint_success(f"✅ {self.step_name} completed successfully")
            tprint_info(f"📊 Created feature sets: {metrics.get('total_features_selected', 0)} total features across {len(feature_sets)} sets")
            tprint_info("✅ Using permutation importance - captures feature interactions for better predictions")
            tprint_info("📊 Permutation importance is more reliable than Gini for complex trading strategies")

            return execution_result

        except Exception as e:
            error_msg = f"Final feature selection step failed: {str(e)}"
            tprint_error(error_msg)

            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {},
                'execution_time': 0.0
            }

    def _collect_features_from_previous_steps(self) -> Dict[str, Any]:
        """Collect features from previous steps in the pipeline.
        
        CRITICAL FIX: Load the correct features:
        1. Load generated_features (from feature_generation_step) - base features
        2. Load analyst_interaction_features (from interaction_generation_step) - interaction features
        3. Merge them together for comprehensive feature selection
        
        DO NOT load:
        - feature_dataframe (generic, may be small 300 rows)
        - lookback_optimization (may be small)
        - selected_feature_dataframe_* (cached, we want fresh selection)
        """
        features_data = {}

        tprint_info("🔍 Loading features for feature selection...")
        tprint_info("🔍 FORCING fresh feature generation load (skipping cached selected_feature_dataframe_*)")
        tprint_info("🔍 This ensures we always perform new feature selection with all available features")
        
        # DISABLED PRIORITY 1: Skip loading previously selected features
        # This was causing the step to use buggy cached artifacts with only 2 columns
        # We ALWAYS want to perform fresh feature selection from the full generated features
        selected_loaded = False
        tprint_info("⚠️ Skipping cached selected_feature_dataframe_* artifacts to force fresh selection")
        
        # PRIORITY 2: Load fresh generated features (ALWAYS)
        if not selected_loaded:
            tprint_info("🔍 No selected features found, loading generated features...")
            try:
                generated_features = None
                for artifact_name in ['generated_features_15m', 'generated_features', 'generated_features_1h', 'generated_features_long']:
                    try:
                        generated_features = self._get_artifact(artifact_name)
                        if generated_features is not None:
                            # CRITICAL: Verify this is the large dataset (180 days, 16K+ rows)
                            if hasattr(generated_features, 'shape') and len(generated_features) >= 10000:
                                features_data['generated_features'] = generated_features
                                tprint_success(f"✅ Retrieved LARGE generated features ({artifact_name}): {generated_features.shape}")
                                tprint_success(f"✅ Time range: {generated_features.index.min()} to {generated_features.index.max()}")
                                time_span = (generated_features.index.max() - generated_features.index.min()).days
                                tprint_success(f"✅ Time span: {time_span} days (~{len(generated_features)} rows)")
                                break
                            else:
                                tprint_warning(f"⚠️ Skipping {artifact_name} - too small ({len(generated_features) if hasattr(generated_features, 'shape') else 'N/A'} rows)")
                    except Exception as e:
                        tprint_warning(f"⚠️ Could not load {artifact_name}: {e}")
                        continue
                
                if generated_features is None:
                    tprint_error("❌ CRITICAL: Could not get large generated features from any artifact name!")
                    tprint_error("❌ This will cause the feature selection to fail!")
            except Exception as e:
                tprint_error(f"❌ CRITICAL: Could not get main generated features: {e}")

        # PRIORITY 3: Load interaction features from interaction generation step
        tprint_info("🔍 Loading interaction features from interaction generation step...")
        tprint_info("🔍 Trying multiple artifact names for interaction features...")
        
        interaction_artifact_names = [
            'analyst_interaction_features',
            'interaction_features',
            'analyst_interactions',
            'generated_interaction_features'
        ]
        
        interaction_loaded = False
        for artifact_name in interaction_artifact_names:
            try:
                tprint_info(f"🔍 Trying artifact name: {artifact_name}")
                interaction_features = self._get_artifact(artifact_name)
                if interaction_features is not None and hasattr(interaction_features, 'shape'):
                    features_data['analyst_interactions'] = interaction_features
                    tprint_success(f"✅ Retrieved interaction features from '{artifact_name}': {interaction_features.shape}")
                    tprint_success(f"✅ Time range: {interaction_features.index.min()} to {interaction_features.index.max()}")
                    time_span = (interaction_features.index.max() - interaction_features.index.min()).days
                    tprint_success(f"✅ Time span: {time_span} days (~{len(interaction_features)} rows)")
                    
                    # Show sample of interaction feature names
                    interaction_cols = [col for col in interaction_features.columns if 'interaction' in col.lower() or 'x_' in col.lower()]
                    tprint_success(f"✅ Found {len(interaction_cols)} interaction columns")
                    if interaction_cols:
                        tprint_success(f"✅ Sample interaction features: {interaction_cols[:5]}")
                    
                    tprint_success(f"✅ Interaction features will be merged with generated features for selection")
                    interaction_loaded = True
                    break
                else:
                    tprint_info(f"   Artifact '{artifact_name}' not found or empty")
            except Exception as e:
                tprint_info(f"   Could not load '{artifact_name}': {e}")
        
        if not interaction_loaded:
            tprint_warning("⚠️ No interaction features found from any artifact name")
            tprint_warning("⚠️ Tried: " + ", ".join(interaction_artifact_names))
            tprint_warning("⚠️ This means interaction_generation_step may not have run yet")
            tprint_warning("⚠️ Continuing with only generated features")

        # FINAL VALIDATION
        if 'generated_features' in features_data:
            tprint_success(f"✅ FINAL: Using generated features: {features_data['generated_features'].shape}")
            tprint_success(f"✅ This ensures we use the full 180-day dataset for feature selection")
        else:
            tprint_error("❌ CRITICAL: No large features found! Feature selection will fail!")
        
        if 'analyst_interactions' in features_data:
            tprint_success(f"✅ FINAL: Using interaction features: {features_data['analyst_interactions'].shape}")
            tprint_success(f"✅ Total feature sources: {len(features_data)}")
        else:
            tprint_warning("⚠️ No interaction features loaded - using only generated features")

        return features_data

    async def _initialize_optimization_components(self, config: Dict[str, Any]) -> None:
        """Initialize VectorBT optimization components."""
        try:
            tprint_info("🚀 Initializing VectorBT optimization components...")
            
            # Initialize unified vectorization manager
            vectorization_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=config.get('enable_parallel', True),
                memory_efficient=config.get('memory_efficient', True),
                max_memory_gb=config.get('max_memory_gb', 8.0),
                chunk_size=config.get('chunk_size', 1000),
                enable_monitoring=config.get('enable_monitoring', True),
                batch_size=config.get('batch_size', 10000),
                enable_batch_processing=True,
                rolling_optimization_threshold=config.get('rolling_optimization_threshold', 1000),
                enable_rolling_optimization=True
            )
            
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            tprint_success("✅ Unified vectorization manager initialized")
            
            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=config.get('enable_parallel', True),
                memory_efficient=config.get('memory_efficient', True),
                chunk_size=config.get('chunk_size', 1000),
                fast_fail=config.get('fast_fail', True),
                enable_logging=config.get('enable_logging', True)
            )
            tprint_success("✅ VectorBT rolling optimizer initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize optimization components: {e}")
            self.optimization_enabled = False
            tprint_warning("⚠️ Continuing without VectorBT optimizations")

    async def _initialize_hardware_optimization_components(self, config: Dict[str, Any]) -> None:
        """Initialize hardware optimization components."""
        try:
            tprint_info("🚀 Initializing hardware optimization components...")
            
            # Initialize unified hardware manager
            hardware_config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.BALANCED,
                gpu_optimization_level=OptimizationLevel.BALANCED,
                memory_optimization_level=OptimizationLevel.BALANCED,
                enable_adaptive_optimization=True,
                enable_learning=True,
                auto_tuning_enabled=True,
                performance_monitoring_enabled=True,
                memory_limit_gb=config.get('memory_limit_gb', 8.0),
                enable_memory_pooling=True,
                enable_predictive_allocation=True,
                enable_compression=True
            )
            
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            init_result = self.hardware_manager.initialize()
            if init_result:
                tprint_success("✅ Unified hardware manager initialized")
            else:
                tprint_warning("⚠️ Unified hardware manager initialization failed")
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = AdaptiveOptimizationEngine(
                database_path="optimization_performance.db"
            )
            # Initialize hardware managers for the adaptive engine
            self.adaptive_engine.initialize_hardware_managers()
            tprint_success("✅ Adaptive optimization engine initialized")
            
            # Initialize CPU optimizer with warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*CoreAffinityManager.*")
                warnings.filterwarnings("ignore", message=".*core affinity.*")
                self.cpu_optimizer = AdvancedM1CPUOptimizer()
                # Add custom workload profile for feature engineering
                feature_engineering_profile = WorkloadProfile(
                    name='feature_engineering',
                    cpu_intensity=0.7,
                    memory_intensity=0.8,
                    thermal_sensitivity=0.4,
                    power_sensitivity=0.5,
                    preferred_cores=CoreType.PERFORMANCE,
                    max_threads=6
                )
                self.cpu_optimizer.add_workload_profile(feature_engineering_profile)
                # Optimize for feature engineering workload
                self.cpu_optimizer.optimize_for_workload_profile('feature_engineering')
            tprint_success("✅ Advanced CPU optimizer initialized")
            
            # Initialize GPU manager
            self.gpu_manager = EnhancedM1GPUManager()
            # EnhancedM1GPUManager doesn't have an initialize method
            tprint_success("✅ Enhanced GPU manager initialized")
            
            # Initialize memory optimizer
            self.memory_optimizer = AdvancedM1MemoryOptimizer(
                memory_limit_gb=config.get('memory_limit_gb', 8.0),
                strategy=MemoryStrategy.ADAPTIVE
            )
            # AdvancedM1MemoryOptimizer doesn't have an initialize method
            tprint_success("✅ Advanced memory optimizer initialized")
            
            tprint_success("✅ All hardware optimization components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize hardware optimization components: {e}")
            self.hardware_optimization_enabled = False
            tprint_warning("⚠️ Continuing without hardware optimizations")

    def _combine_features(self, features_data: Dict[str, Any], labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Combine features from different sources into a single DataFrame with VectorBT optimizations."""
        tprint_error("🔄 CRITICAL DEBUG: Starting _combine_features method")
        
        # STEP 0: Log input data sizes
        tprint_error(f"📊 INPUT DATA ANALYSIS:")
        tprint_error(f"   labeled_df shape: {labeled_df.shape}")
        tprint_error(f"   labeled_df time range: {labeled_df.index.min()} to {labeled_df.index.max()}")
        tprint_error(f"   features_data keys: {list(features_data.keys())}")
        
        for key, data in features_data.items():
            if data is not None and hasattr(data, 'shape'):
                tprint_error(f"   {key} shape: {data.shape}")
                if hasattr(data, 'index'):
                    tprint_error(f"   {key} time range: {data.index.min()} to {data.index.max()}")
            else:
                tprint_error(f"   {key}: {type(data)} (None or no shape)")
        
        tprint_info("🔄 Combining features with VectorBT optimizations...")
        
        # PRIORITY 1: Start with labeled dataframe to preserve target column
        base_features = labeled_df.copy()
        tprint_error(f"📊 STEP 1: Using labeled_df as base: {base_features.shape}")
        
        # STEP 0: Remove exact duplicates before any processing
        tprint_info("🔍 Removing exact duplicate features...")
        base_features = self._remove_exact_duplicates(base_features)
        tprint_info(f"📊 Using labeled dataframe as base: {base_features.shape}")
        tprint_info(f"📊 Target columns in base: {[col for col in base_features.columns if 'target' in col.lower()]}")
        
        # Collect all feature dataframes to concatenate once (memory optimization)
        feature_chunks = []
        
        # PRIORITY 2: Add main generated features if available
        if 'generated_features' in features_data and features_data['generated_features'] is not None:
            generated_features = features_data['generated_features']
            tprint_info(f"📊 Adding main generated features: {generated_features.shape}")
            
            # Check data alignment
            if generated_features.shape[0] != base_features.shape[0]:
                tprint_warning(f"⚠️ Shape mismatch: base_features {base_features.shape} vs generated_features {generated_features.shape}")
                
                # DETAILED DEBUGGING: Show time ranges
                tprint_error(f"🔍 CRITICAL ALIGNMENT ISSUE DETECTED:")
                tprint_error(f"   Base features (labeled_df): {base_features.shape}")
                tprint_error(f"   - Time range: {base_features.index.min()} to {base_features.index.max()}")
                tprint_error(f"   Generated features: {generated_features.shape}")
                tprint_error(f"   - Time range: {generated_features.index.min()} to {generated_features.index.max()}")
                
                # Try to align by index if possible
                if hasattr(generated_features.index, 'intersection') and hasattr(base_features.index, 'intersection'):
                    common_index = base_features.index.intersection(generated_features.index)
                    tprint_error(f"   Common index: {len(common_index)} rows")
                    
                    if len(common_index) > 0:
                        tprint_error(f"   Common range: {common_index.min()} to {common_index.max()}")
                        
                        # CRITICAL FIX: Instead of reducing both to common index,
                        # use the dataset with MORE FEATURES (generated_features) as base and add targets from labeled_df
                        # Compare column count, not row count!
                        if len(generated_features.columns) > len(base_features.columns):
                            tprint_info(f"🔧 FIXING: Using larger generated_features ({len(generated_features)} rows) as base")
                            tprint_info(f"🔧 Adding target columns from labeled_df to generated_features")
                            
                            # Start with the larger generated_features dataset
                            new_base = generated_features.copy()
                            
                            # Add target columns from labeled_df where available
                            target_cols = [col for col in base_features.columns if 'target' in col.lower()]
                            for target_col in target_cols:
                                if target_col in base_features.columns:
                                    # Check for duplicate indices before reindexing
                                    if base_features.index.duplicated().any():
                                        tprint_warning(f"⚠️ Removing {base_features.index.duplicated().sum()} duplicate indices from labeled_df")
                                        base_features = base_features[~base_features.index.duplicated(keep='first')]
                                    
                                    if generated_features.index.duplicated().any():
                                        tprint_warning(f"⚠️ Removing {generated_features.index.duplicated().sum()} duplicate indices from generated_features")
                                        generated_features = generated_features[~generated_features.index.duplicated(keep='first')]
                                        new_base = generated_features.copy()  # Update new_base after deduplication
                                    
                                    # Align target data to generated_features index
                                    try:
                                        aligned_targets = base_features[target_col].reindex(generated_features.index)
                                        new_base[target_col] = aligned_targets
                                        tprint_info(f"   Added target column '{target_col}': {aligned_targets.notna().sum()} non-null values")
                                    except ValueError as e:
                                        tprint_error(f"❌ Failed to align target column '{target_col}': {e}")
                                        # Fallback: use intersection approach
                                        common_idx = base_features.index.intersection(generated_features.index)
                                        if len(common_idx) > 0:
                                            new_base.loc[common_idx, target_col] = base_features.loc[common_idx, target_col]
                                            tprint_info(f"   Fallback: Added target column '{target_col}' for {len(common_idx)} common indices")
                            
                            # Replace base_features with the larger aligned dataset
                            base_features = new_base
                            generated_features = None  # Already included in base_features
                            
                            tprint_success(f"✅ FIXED: Now using {base_features.shape} dataset (was {len(common_index)} common rows)")
                            tprint_success(f"✅ base_features now contains {len([c for c in base_features.columns if 'target' not in c.lower()])} feature columns + targets")
                            
                            # CRITICAL FIX: Don't skip interaction features even on early return path!
                            # We need to merge interaction features before returning
                            tprint_error("=" * 80)
                            tprint_error("🔧 FIX: Early return path - but loading interaction features first!")
                            tprint_error("=" * 80)
                            
                            # Load and merge interaction features
                            for interaction_type in ['analyst_interactions', 'tactician_interactions']:
                                tprint_info(f"🔍 Checking for '{interaction_type}' in features_data...")
                                if interaction_type in features_data and features_data[interaction_type] is not None:
                                    interaction_df = features_data[interaction_type]
                                    if isinstance(interaction_df, pd.DataFrame):
                                        tprint_success(f"✅ Found {interaction_type}: {interaction_df.shape}")
                                        
                                        # Align to base_features index
                                        interaction_df_aligned = interaction_df.reindex(base_features.index)
                                        
                                        # Add columns that don't already exist
                                        new_cols = [col for col in interaction_df_aligned.columns if col not in base_features.columns]
                                        if new_cols:
                                            base_features = pd.concat([base_features, interaction_df_aligned[new_cols]], axis=1)
                                            tprint_success(f"✅ Merged {len(new_cols)} interaction features from {interaction_type}")
                                            
                                            # Check for interaction feature names
                                            interaction_cols = [col for col in new_cols if 'interaction' in col.lower() or '_x_' in col.lower()]
                                            tprint_success(f"   Including {len(interaction_cols)} columns with 'interaction' or '_x_' in name")
                                        else:
                                            tprint_warning(f"⚠️ All columns from {interaction_type} already exist in base_features")
                                    else:
                                        tprint_warning(f"⚠️ {interaction_type} is not a DataFrame")
                                else:
                                    tprint_info(f"   {interaction_type} not found in features_data")
                            
                            tprint_success(f"✅ Final base_features shape after interaction merge: {base_features.shape}")
                            
                            # Just clean up and return
                            numeric_cols = []
                            for col in base_features.columns:
                                if col == 'timestamp' or col in TARGET_COLUMN_NAMES or pd.api.types.is_numeric_dtype(base_features[col]):
                                    numeric_cols.append(col)
                            
                            result_df = base_features[numeric_cols].copy()
                            
                            tprint_error("=" * 80)
                            tprint_error("🔍 CRITICAL: _combine_features FINAL RESULT:")
                            tprint_error(f"   Final shape: {result_df.shape}")
                            tprint_error(f"   Final time range: {result_df.index.min()} to {result_df.index.max()}")
                            tprint_error(f"   Feature columns: {len([c for c in result_df.columns if 'target' not in c.lower()])}")
                            tprint_error(f"   Target columns: {[c for c in result_df.columns if 'target' in c.lower()]}")
                            tprint_error("=" * 80)
                            
                            if len(result_df) < 10000:
                                tprint_error(f"❌ CRITICAL: Still getting small dataset ({len(result_df)} rows)!")
                            else:
                                tprint_success(f"✅ CRITICAL: _combine_features returning LARGE dataset ({len(result_df)} rows)!")
                            
                            return result_df
                        else:
                            tprint_info(f"📊 Using standard alignment with {len(common_index)} common indices")
                            generated_features = generated_features.loc[common_index]
                            base_features = base_features.loc[common_index]
                            tprint_info(f"📊 After alignment - base_features shape: {base_features.shape}")
                            tprint_info(f"📊 After alignment - target columns: {[col for col in base_features.columns if 'target' in col.lower()]}")
                    else:
                        tprint_warning("⚠️ No common indices found, skipping generated features")
                        generated_features = None
                else:
                    tprint_warning("⚠️ Cannot align dataframes, skipping generated features")
                    generated_features = None
            
            if generated_features is not None:
                # Add generated features (excluding any duplicate columns and target columns)
                generated_cols = [col for col in generated_features.columns
                               if col not in base_features.columns and col not in TARGET_COLUMN_NAMES]
                if generated_cols:
                    feature_chunks.append(generated_features[generated_cols])
                    tprint_info(f"📊 Queued {len(generated_cols)} generated features for concatenation")

        # Use vectorization manager for optimized operations if available
        if self.vectorization_manager and self.optimization_enabled:
            try:
                # Optimize the base dataframe for memory efficiency
                base_features = self.vectorization_manager.optimize_dataframe(base_features)
                tprint_info("✅ Base features optimized for memory efficiency")
            except Exception as e:
                tprint_warning(f"⚠️ Memory optimization failed: {e}")

        # PRIORITY 2: Add lookback optimization features (most sophisticated engineered features)
        if 'lookback_optimization' in features_data and features_data['lookback_optimization'] is not None:
            lookback_data = features_data['lookback_optimization']
            if isinstance(lookback_data, pd.DataFrame):
                # Check if lookback data has the correct shape (samples, features)
                if lookback_data.shape[0] > 1:  # Multiple samples
                    # Optimize lookback dataframe if available
                    if self.vectorization_manager and self.optimization_enabled:
                        try:
                            lookback_data = self.vectorization_manager.optimize_dataframe(lookback_data)
                        except Exception as e:
                            tprint_warning(f"⚠️ Lookback dataframe optimization failed: {e}")
                    
                    # Add lookback features (excluding any duplicate columns)
                    lookback_cols = [col for col in lookback_data.columns
                                   if col not in base_features.columns]
                    if lookback_cols:
                        feature_chunks.append(lookback_data[lookback_cols])
                        tprint_info(f"📊 Queued {len(lookback_cols)} lookback optimization features (PRIORITY 2)")
                else:
                    tprint_warning(f"⚠️ Lookback optimization data has wrong shape {lookback_data.shape}, skipping")
            elif isinstance(lookback_data, dict):
                # Lookback optimization produces metadata, not feature data
                tprint_info(f"📊 Lookback optimization metadata available: {len(lookback_data)} categories")
                tprint_info(f"📊 Lookback optimization categories: {list(lookback_data.keys())}")
                # TODO: Use this metadata to generate features with optimized lookback periods
                tprint_info("ℹ️ Note: Lookback optimization metadata should be used to generate features with optimized lookback periods")

        # CRITICAL FIX: Collect all dataframes first, find common index, then align once
        # This prevents repeatedly reducing base_features with each chunk
        all_dataframes = [base_features]
        dataframe_info = [{'name': 'base_features', 'df': base_features}]
        
        # PRIORITY 3: Add interaction features (complex feature interactions)
        tprint_error("=" * 80)
        tprint_error("🔍 HYPOTHESIS TEST: Checking interaction features collection")
        tprint_error("=" * 80)
        for interaction_type in ['analyst_interactions', 'tactician_interactions']:
            tprint_error(f"🔍 Checking for '{interaction_type}' in features_data...")
            if interaction_type in features_data and features_data[interaction_type] is not None:
                interaction_df = features_data[interaction_type]
                tprint_error(f"✅ Found {interaction_type}: type={type(interaction_df)}, shape={interaction_df.shape if hasattr(interaction_df, 'shape') else 'N/A'}")
                
                if isinstance(interaction_df, pd.DataFrame):
                    # Check for interaction feature names
                    interaction_cols = [col for col in interaction_df.columns if 'interaction' in col.lower() or '_x_' in col.lower()]
                    tprint_error(f"🔍 {interaction_type} has {len(interaction_cols)} columns with 'interaction' or '_x_' in name")
                    if interaction_cols:
                        tprint_error(f"   Sample: {interaction_cols[:5]}")
                    
                    # Optimize interaction dataframe if available
                    if self.vectorization_manager and self.optimization_enabled:
                        try:
                            interaction_df = self.vectorization_manager.optimize_dataframe(interaction_df)
                        except Exception as e:
                            tprint_warning(f"⚠️ Interaction dataframe optimization failed: {e}")
                    
                    # Add to collection for later alignment
                    dataframe_info.append({'name': interaction_type, 'df': interaction_df})
                    tprint_success(f"✅ Collected {interaction_type}: {interaction_df.shape}")
                else:
                    tprint_error(f"❌ {interaction_type} is not a DataFrame: {type(interaction_df)}")
            else:
                tprint_error(f"❌ {interaction_type} not found in features_data")
        tprint_error("=" * 80)

        # PRIORITY 4: Add features from feature dataframe if available
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        basic_time_cols = ['hour', 'day_of_week', 'base_threshold']
        
        if 'feature_dataframe' in features_data and features_data['feature_dataframe'] is not None:
            feature_df = features_data['feature_dataframe']
            if isinstance(feature_df, pd.DataFrame):
                # Optimize if available
                if self.vectorization_manager and self.optimization_enabled:
                    try:
                        feature_df = self.vectorization_manager.optimize_dataframe(feature_df)
                    except Exception as e:
                        tprint_warning(f"⚠️ Feature dataframe optimization failed: {e}")
                
                # Add to collection
                dataframe_info.append({'name': 'feature_dataframe', 'df': feature_df})
                tprint_info(f"📊 Collected feature_dataframe: {feature_df.shape}")
        
        # CRITICAL: Find common index across ALL dataframes
        tprint_info(f"📊 Finding common index across {len(dataframe_info)} dataframes...")
        common_index = dataframe_info[0]['df'].index
        for info in dataframe_info[1:]:
            common_index = common_index.intersection(info['df'].index)
            tprint_info(f"📊 After {info['name']}: {len(common_index)} common indices")
        
        if len(common_index) < 100:
            tprint_error(f"❌ Too few common indices ({len(common_index)}). Minimum required: 100")
            tprint_warning("⚠️ This will cause feature selection to fail. Check data alignment issues.")
        
        tprint_success(f"✅ Found {len(common_index)} common indices across all dataframes")
        
        # Align ALL dataframes to common index
        tprint_info("📊 Aligning all dataframes to common index...")
        for info in dataframe_info:
            info['df'] = info['df'].loc[common_index]
            tprint_info(f"📊 Aligned {info['name']}: {info['df'].shape}")
        
        # Now extract base_features and build feature_chunks
        base_features = dataframe_info[0]['df']
        feature_chunks = []
        
        tprint_error("=" * 80)
        tprint_error("🔍 HYPOTHESIS TEST: Processing dataframes for concatenation")
        tprint_error(f"   Total dataframes to process: {len(dataframe_info) - 1}")
        tprint_error("=" * 80)
        
        for info in dataframe_info[1:]:
            df = info['df']
            name = info['name']
            
            tprint_error(f"🔍 Processing '{name}': {df.shape}")
            tprint_error(f"   Columns in {name}: {len(df.columns)}")
            
            # Check for interaction columns in this dataframe
            interaction_cols_in_df = [col for col in df.columns if 'interaction' in col.lower() or '_x_' in col.lower()]
            tprint_error(f"   Interaction columns in {name}: {len(interaction_cols_in_df)}")
            if interaction_cols_in_df:
                tprint_error(f"   Sample interaction cols: {interaction_cols_in_df[:3]}")
            
            if name == 'feature_dataframe':
                # Exclude OHLCV, basic time, and target columns
                feature_cols = [col for col in df.columns
                              if col not in ohlcv_cols and col not in basic_time_cols and col not in TARGET_COLUMN_NAMES]
                tprint_error(f"   feature_dataframe: Filtered to {len(feature_cols)} columns (excluded OHLCV/time/targets)")
            else:
                # For interactions, exclude columns already in base_features
                before_filter = len(df.columns)
                feature_cols = [col for col in df.columns if col not in base_features.columns]
                after_filter = len(feature_cols)
                tprint_error(f"   {name}: {before_filter} columns -> {after_filter} after removing duplicates with base_features")
                
                # HYPOTHESIS 1 TEST: Are interaction features being filtered as duplicates?
                if before_filter != after_filter:
                    removed = before_filter - after_filter
                    tprint_error(f"   ⚠️ REMOVED {removed} duplicate columns from {name}")
                    duplicate_cols = [col for col in df.columns if col in base_features.columns]
                    tprint_error(f"   Duplicate columns: {duplicate_cols[:5]}...")
            
            if feature_cols:
                # Check how many interaction features survive
                interaction_cols_surviving = [col for col in feature_cols if 'interaction' in col.lower() or '_x_' in col.lower()]
                tprint_error(f"   Interaction columns surviving filter: {len(interaction_cols_surviving)}")
                
                feature_chunks.append(df[feature_cols])
                tprint_success(f"✅ Queued {len(feature_cols)} features from {name}")
            else:
                tprint_error(f"❌ No features to queue from {name} after filtering")
        
        # Concatenate all feature chunks at once
        if feature_chunks:
            tprint_info(f"📊 Concatenating {len(feature_chunks)} feature chunks...")
            base_features = pd.concat([base_features] + feature_chunks, axis=1)
            tprint_success(f"✅ Successfully concatenated all feature chunks: {base_features.shape}")

        # Remove any non-numeric columns except timestamp and target columns
        tprint_error("=" * 80)
        tprint_error("🔍 HYPOTHESIS TEST: Numeric column filtering")
        tprint_error(f"   Total columns before numeric filter: {len(base_features.columns)}")
        
        # Check interaction columns before numeric filter
        interaction_cols_before = [col for col in base_features.columns if 'interaction' in col.lower() or '_x_' in col.lower()]
        tprint_error(f"   Interaction columns before numeric filter: {len(interaction_cols_before)}")
        if interaction_cols_before:
            tprint_error(f"   Sample: {interaction_cols_before[:3]}")
        
        numeric_cols = []
        non_numeric_cols = []
        
        for col in base_features.columns:
            if col == 'timestamp' or col in TARGET_COLUMN_NAMES or pd.api.types.is_numeric_dtype(base_features[col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
        
        # HYPOTHESIS 3 TEST: Are interaction features non-numeric?
        interaction_cols_numeric = [col for col in numeric_cols if 'interaction' in col.lower() or '_x_' in col.lower()]
        interaction_cols_non_numeric = [col for col in non_numeric_cols if 'interaction' in col.lower() or '_x_' in col.lower()]
        
        tprint_error(f"   Numeric columns: {len(numeric_cols)}")
        tprint_error(f"   Non-numeric columns (will be removed): {len(non_numeric_cols)}")
        tprint_error(f"   Interaction columns that are numeric: {len(interaction_cols_numeric)}")
        tprint_error(f"   Interaction columns that are NON-numeric (LOST): {len(interaction_cols_non_numeric)}")
        
        if interaction_cols_non_numeric:
            tprint_error(f"   ⚠️ LOSING {len(interaction_cols_non_numeric)} non-numeric interaction columns!")
            tprint_error(f"   Sample non-numeric interactions: {interaction_cols_non_numeric[:5]}")
            # Check dtypes
            for col in interaction_cols_non_numeric[:3]:
                tprint_error(f"      {col}: dtype={base_features[col].dtype}")
        
        tprint_error("=" * 80)

        result_df = base_features[numeric_cols].copy()
        
        # Final check
        interaction_cols_final = [col for col in result_df.columns if 'interaction' in col.lower() or '_x_' in col.lower()]
        tprint_error(f"🔍 FINAL CHECK: Interaction columns in result_df: {len(interaction_cols_final)}")
        if interaction_cols_final:
            tprint_success(f"✅ Interaction features survived! Sample: {interaction_cols_final[:5]}")
        else:
            tprint_error(f"❌ NO INTERACTION FEATURES in final result!")
        
        # Debug: Check if target column is present with priority for new simplified target structure
        # First check for new simplified target structure (highest priority)
        if 'target_long' in result_df.columns and 'target_short' in result_df.columns:
            available_targets = ['target_long', 'target_short']
            tprint_info("📊 Using new simplified target structure: target_long, target_short")
            tprint_info(f"📊 Target columns found: target_long ({result_df['target_long'].notna().sum()} non-NaN), target_short ({result_df['target_short'].notna().sum()} non-NaN)")
        else:
            # Fall back to legacy target detection
            available_targets = [col for col in TARGET_COLUMN_NAMES if col in result_df.columns]
            tprint_info(f"📊 Using legacy target detection: {available_targets}")
            # Check if we have the old price_target_vol_normalized column
            if 'price_target_vol_normalized' in result_df.columns:
                tprint_warning("⚠️ Legacy target 'price_target_vol_normalized' found - consider migrating to new simplified target structure")
        
        tprint_info(f"📊 Combined feature matrix: {len(numeric_cols)} features, {len(result_df)} samples")
        tprint_info(f"📊 Available target columns: {available_targets}")

        if not available_targets:
            tprint_warning("⚠️ No target columns found in combined features!")
            tprint_info(f"📊 All columns in result_df: {list(result_df.columns)[:20]}...")

        # Handle NaN values with optimized operations
        if self.vectorization_manager and self.optimization_enabled:
            try:
                # Use vectorized operations for NaN handling
                tprint_info("🔄 Optimizing NaN handling...")
                
                # Drop columns with too many NaN values (more lenient for sophisticated features)
                nan_threshold = int(0.5 * len(result_df))  # More lenient threshold
                valid_cols = []
                for col in result_df.columns:
                    # ALWAYS keep target columns regardless of NaN count
                    if col in TARGET_COLUMN_NAMES or 'target' in col.lower():
                        valid_cols.append(col)
                        tprint_info(f"📊 Keeping target column: {col}")
                    elif result_df[col].count() >= nan_threshold:
                        valid_cols.append(col)
                    else:
                        # Check if it's a sophisticated feature and be more lenient
                        if any(keyword in col.lower() for keyword in ['vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced', 'statistical', 'wavelet', 'entropy', 'ad_line', 'obv', 'volatility', 'order_flow']):
                            if result_df[col].count() >= int(0.3 * len(result_df)):  # Even more lenient for sophisticated features
                                valid_cols.append(col)
                                tprint_info(f"📊 Keeping sophisticated feature with low data coverage: {col}")
                
                result_df = result_df[valid_cols]
                tprint_info(f"📊 After NaN filtering - columns remaining: {len(valid_cols)}, target columns: {[col for col in valid_cols if 'target' in col.lower()]}")
                
                # Fill remaining NaN with median using vectorized operations
                numeric_cols_only = result_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols_only) > 0:
                    medians = result_df[numeric_cols_only].median()
                    result_df[numeric_cols_only] = result_df[numeric_cols_only].fillna(medians)
                
                tprint_success("✅ NaN handling optimized with sophisticated feature protection")
            except Exception as e:
                tprint_warning(f"⚠️ Optimized NaN handling failed, using standard method: {e}")
                result_df = result_df.dropna(axis=1, thresh=int(0.7 * len(result_df)))
                result_df = result_df.fillna(result_df.median())
        else:
            # Standard NaN handling - but preserve target columns
            tprint_info("🔄 Standard NaN handling...")
            target_cols_to_preserve = [col for col in result_df.columns if col in TARGET_COLUMN_NAMES or 'target' in col.lower()]
            tprint_info(f"📊 Preserving target columns: {target_cols_to_preserve}")
            
            # Separate target columns
            target_data = result_df[target_cols_to_preserve].copy() if target_cols_to_preserve else None
            feature_data = result_df.drop(columns=target_cols_to_preserve, errors='ignore')
            
            # Drop feature columns with too many NaNs
            feature_data = feature_data.dropna(axis=1, thresh=int(0.7 * len(feature_data)))
            feature_data = feature_data.fillna(feature_data.median())
            
            # Recombine with target columns
            if target_data is not None:
                result_df = pd.concat([feature_data, target_data], axis=1)
                tprint_info(f"📊 After NaN handling - features: {feature_data.shape[1]}, targets: {len(target_cols_to_preserve)}")
            else:
                result_df = feature_data
                tprint_warning("⚠️ No target columns found to preserve!")

        # Final optimization if vectorization manager is available
        if self.vectorization_manager and self.optimization_enabled:
            try:
                result_df = self.vectorization_manager.optimize_dataframe(result_df)
                tprint_success("✅ Final feature matrix optimized")
            except Exception as e:
                tprint_warning(f"⚠️ Final optimization failed: {e}")

        tprint_info(f"📊 Combined feature matrix: {result_df.shape[1]} features, {result_df.shape[0]} samples")
        
        # CRITICAL DEBUG: Final result analysis
        tprint_error(f"🔍 CRITICAL: _combine_features FINAL RESULT:")
        tprint_error(f"   Final shape: {result_df.shape}")
        tprint_error(f"   Final time range: {result_df.index.min()} to {result_df.index.max()}")
        tprint_error(f"   Target columns: {[col for col in result_df.columns if 'target' in col.lower()]}")
        
        if len(result_df) < 10000:
            tprint_error(f"❌ CRITICAL: _combine_features returning SMALL dataset ({len(result_df)} rows)!")
            tprint_error(f"❌ This is where the 300-row bottleneck is happening!")
        else:
            tprint_success(f"✅ CRITICAL: _combine_features returning LARGE dataset ({len(result_df)} rows)!")
        
        return result_df
    
    def _remove_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with identical values to prevent perfect correlation (1.0).
        
        Args:
            df: DataFrame to deduplicate
            
        Returns:
            DataFrame with exact duplicate columns removed
        """
        try:
            if df.empty:
                return df
                
            original_cols = len(df.columns)
            duplicate_cols = []
            
            # Check for exact duplicates
            for i in range(len(df.columns)):
                for j in range(i+1, len(df.columns)):
                    col_i = df.iloc[:, i]
                    col_j = df.iloc[:, j]
                    
                    # Check if columns are identical (including NaN patterns)
                    if col_i.equals(col_j):
                        duplicate_cols.append(df.columns[j])
                        tprint_info(f"🔍 Found exact duplicate: '{df.columns[j]}' == '{df.columns[i]}'")
            
            # Remove duplicate columns
            if duplicate_cols:
                df = df.drop(columns=duplicate_cols)
                tprint_info(f"🗑️ Removed {len(duplicate_cols)} exact duplicate columns")
                tprint_info(f"📊 Column count: {original_cols} -> {len(df.columns)}")
            else:
                tprint_info("✅ No exact duplicate columns found")
                
            return df
            
        except Exception as e:
            tprint_warning(f"⚠️ Error removing exact duplicates: {e}")
            return df

    def _setup_selection_config(self, config: Dict[str, Any]) -> FinalFeatureSelectionConfig:
        """Setup feature selection configuration."""
        # Add hardware optimization configuration to the config
        selection_config = config.copy()

        # Add default hardware optimization parameters if not present
        selection_config.setdefault('enable_hardware_optimization', True)
        selection_config.setdefault('memory_limit_gb', 8.0)
        selection_config.setdefault('max_memory_mb', 2048.0)
        selection_config.setdefault('streaming_chunk_size', 10000)
        selection_config.setdefault('memory_pressure_threshold', 0.8)
        selection_config.setdefault('enable_caching', True)
        selection_config.setdefault('cache_memory_mb', 1024)
        selection_config.setdefault('cache_memory_limit_gb', 4.0)
        selection_config.setdefault('enable_compression', True)

        return FinalFeatureSelectionConfig(
            max_features=config.get('max_features', 100),
            min_features=config.get('min_features', 10),
            selection_method=config.get('selection_method', 'permutation'),
            scoring_threshold=config.get('scoring_threshold', 0.01),
            use_tree_based=config.get('use_tree_based', True),
            use_permutation_importance=config.get('use_permutation_importance', True)
        )

    def _perform_multi_size_selection(self, features_df: pd.DataFrame, targets: pd.Series, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Perform feature selection for multiple feature set sizes with CMI-aware Tactician mode support."""
        # Define feature set sizes
        feature_set_sizes = config.get('feature_set_sizes', [60, 50, 40])

        feature_sets = {}

        # Detect Tactician mode and check for CMI availability
        is_tactician_mode = self._detect_tactician_mode(features_df, config)
        cmi_available = CMI_COMPLEMENTARITY_AVAILABLE and self.cmi_scorer is not None
        
        if is_tactician_mode and cmi_available:
            tprint_info("🎯 Tactician mode detected with CMI support - using CMI-based feature selection")
            return self._perform_cmi_aware_selection(features_df, targets, config, feature_set_sizes)
        elif is_tactician_mode and not cmi_available:
            tprint_warning("⚠️ Tactician mode detected but CMI not available - using standard selection")
        else:
            tprint_info("📊 Standard mode - using permutation-based feature selection (captures interactions)")

        # Separate features from targets and exclude raw data columns
        raw_data_columns = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day_of_week', 'base_threshold']
        basic_features = ['open_time', 'close_time', 'body_size', 'close_return', 'price_range_pct', 
                         'volume_return', 'close_log_return', 'volume_log_return', 'price_range', 
                         'body_size_pct', 'trades', 'quote_volume', 'day', 'lookahead_periods', 'is_weekend']
        
        # CRITICAL: Exclude performance metrics and forward-looking columns that are NOT predictive features
        # These are calculated from future data or are outcome metrics, not input features
        performance_metrics = [
            'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'recovery_factor',
            'win_rate', 'profit_factor', 'total_return', 'annualized_return', 'volatility',
            'var_95', 'cvar_95', 'downside_deviation', 'upside_capture', 'downside_capture',
            'information_ratio', 'treynor_ratio', 'jensen_alpha', 'max_consecutive_wins',
            'max_consecutive_losses', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
            'equity_curve', 'cumulative_returns', 'drawdown', 'underwater_curve'
        ]
        
        # Debug: Show all available columns
        tprint_info(f"🔍 DEBUG: All columns in features_df: {list(features_df.columns)}")
        
        # Combine all columns to exclude (including performance metrics)
        excluded_columns = TARGET_COLUMN_NAMES + ['timestamp'] + raw_data_columns + performance_metrics
        tprint_info(f"🔍 Excluding {len(excluded_columns)} columns: targets, timestamp, raw data, and performance metrics")
        
        # Prioritize sophisticated engineered features over basic ones
        sophisticated_features = [col for col in features_df.columns
                                if col not in excluded_columns
                                and any(keyword in col.lower() for keyword in ['vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced', 'statistical', 'wavelet', 'entropy', 'ad_line', 'obv', 'volatility', 'order_flow'])]
        
        basic_engineered_features = [col for col in features_df.columns
                                   if col not in excluded_columns
                                   and col not in sophisticated_features]
        
        # Prioritize sophisticated features first
        feature_cols = sophisticated_features + basic_engineered_features
        
        # Check for new simplified target structure first (highest priority)
        if 'target_long' in features_df.columns and 'target_short' in features_df.columns:
            target_cols = ['target_long', 'target_short']
            tprint_info("📊 Using new simplified target structure: target_long, target_short")
            # Log target statistics for new simplified structure
            long_signals = (features_df['target_long'] > 0).sum()
            short_signals = (features_df['target_short'] > 0).sum()
            tprint_info(f"📊 Target statistics: Long signals={long_signals}, Short signals={short_signals}")
        else:
            # Fall back to legacy target detection
            target_cols = [col for col in TARGET_COLUMN_NAMES
                          if col in features_df.columns]
            tprint_info(f"📊 Using legacy target detection: {target_cols}")
            # Check if we have old price_target_vol_normalized column
            if 'price_target_vol_normalized' in features_df.columns:
                tprint_warning("⚠️ Legacy target 'price_target_vol_normalized' found - consider migrating to new simplified target structure")

        tprint_info(f"🔍 Sophisticated features: {len(sophisticated_features)}")
        tprint_info(f"🔍 Basic engineered features: {len(basic_engineered_features)}")
        tprint_info(f"🔍 Total available features: {len(feature_cols)}")
        tprint_info(f"🔍 Available targets: {len(target_cols)}")
        tprint_info(f"🔍 Sophisticated features: {sophisticated_features[:5]}...")  # Show first 5 sophisticated features
        tprint_info(f"🔍 Basic engineered features: {basic_engineered_features[:5]}...")  # Show first 5 basic features
        tprint_info(f"🔍 Target columns: {target_cols}")

        if not target_cols:
            raise ValueError("No target column found in features dataframe")

        if not feature_cols:
            raise ValueError("No feature columns found in features dataframe")

        X = features_df[feature_cols]
        y = features_df[target_cols[0]]

        # CRITICAL: Clean NaN values from target variable and align X accordingly
        tprint_info(f"🔍 Checking for NaN values in target variable...")
        nan_count_before = y.isna().sum()
        total_samples = len(y)
        
        if nan_count_before > 0:
            tprint_warning(f"⚠️ Found {nan_count_before} NaN values in target variable ({100*nan_count_before/total_samples:.2f}%)")
            
            # Get valid indices (where target is not NaN)
            valid_indices = y.notna()
            
            # Filter both X and y to remove NaN rows
            X = X[valid_indices]
            y = y[valid_indices]
            
            tprint_success(f"✅ Removed {nan_count_before} rows with NaN targets. Remaining samples: {len(y)}")
            
            # Verify no NaN values remain
            if y.isna().sum() > 0:
                tprint_error(f"❌ ERROR: Target still contains {y.isna().sum()} NaN values after cleaning!")
                raise ValueError(f"Target variable still contains NaN values after cleaning")
        else:
            tprint_success(f"✅ Target variable is clean (no NaN values)")
        
        # Also check for NaN values in features and clean if necessary
        feature_nan_count = X.isna().sum().sum()
        if feature_nan_count > 0:
            tprint_warning(f"⚠️ Found {feature_nan_count} NaN values in features")
            # Fill feature NaNs with median (safer than dropping more rows)
            X = X.fillna(X.median())
            tprint_success(f"✅ Filled feature NaN values with median")

        tprint_info(f"🔍 Performing feature selection on {len(feature_cols)} features using permutation importance...")
        tprint_info(f"📊 Final dataset: {len(X)} samples, {len(X.columns)} features")
        tprint_info("📊 Using permutation importance to capture feature interactions (not just Gini splits)")

        # Use batch processing if vectorization manager is available
        if self.vectorization_manager and self.optimization_enabled and len(feature_cols) > 1000:
            try:
                tprint_info("🚀 Using VectorBT batch processing for feature selection...")
                
                # Create feature configurations for batch processing
                feature_configs = []
                for size in feature_set_sizes:
                    feature_configs.append({
                        'name': f'selected_features_{size}',
                        'type': 'selection',
                        'params': {
                            'max_features': size,
                            'min_features': max(5, size // 2),
                            'selection_method': config.get('selection_method', 'permutation'),
                            'scoring_threshold': config.get('scoring_threshold', 0.01),
                            'use_tree_based': config.get('use_tree_based', True),
                            'use_permutation_importance': config.get('use_permutation_importance', True),
                            'X': X,
                            'y': y,
                            'feature_names': feature_cols
                        }
                    })
                
                # Process features in batch
                batch_results = self.vectorization_manager.batch_process_features(
                    features_df, feature_configs
                )
                
                # Extract results
                for size in feature_set_sizes:
                    result_key = f'selected_features_{size}'
                    if result_key in batch_results.columns:
                        selected_features = batch_results[result_key].dropna().tolist()
                        feature_sets[result_key] = selected_features
                        feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + target_cols].copy()
                
                tprint_success("✅ Batch feature selection completed")
                return feature_sets
                
            except Exception as e:
                tprint_warning(f"⚠️ Batch processing failed, falling back to sequential: {e}")

        # OPTIMIZED: Select top 60 once, then slice for 50 and 40 (avoid redundant computations)
        tprint_info("🔄 Using optimized feature selection with permutation importance...")
        tprint_info("⚡ OPTIMIZATION: Selecting top 60 once, then slicing for 50 and 40 (no redundant computations)")
        
        # Get the maximum size we need to select
        max_size = max(feature_set_sizes)
        
        # Create config for maximum size
        max_size_config = FinalFeatureSelectionConfig(
            max_features=max_size,
            min_features=max(5, max_size // 2),
            selection_method=config.get('selection_method', 'permutation'),
            scoring_threshold=config.get('scoring_threshold', 0.01),
            use_tree_based=config.get('use_tree_based', True),
            use_permutation_importance=config.get('use_permutation_importance', True),
            stability_weight=config.get('stability_weight', 0.3)  # Default 0.3 = balanced (30% stability, 70% importance)
        )

        # Perform selection ONCE for the maximum size
        tprint_info(f"🎯 Selecting top {max_size} features using permutation importance (captures interactions)...")
        temp_component = FinalFeatureSelectionComponent(max_size_config)
        all_selected_features = temp_component.select_features(X, y, feature_cols)

        # CRITICAL DEBUG: Check what was selected
        tprint_error(f"🔍 CRITICAL DEBUG for max size {max_size}:")
        tprint_error(f"   Selected features count: {len(all_selected_features)}")
        tprint_error(f"   Selected features sample: {all_selected_features[:5] if len(all_selected_features) > 0 else 'EMPTY'}")
        
        if not all_selected_features:
            tprint_error(f"❌ CRITICAL: No features selected!")
            tprint_error(f"   Input X shape: {X.shape}")
            tprint_error(f"   Input y shape: {y.shape}")
            tprint_error(f"   Input feature_cols count: {len(feature_cols)}")
            return feature_sets
        
        # Now create feature sets by slicing the ranked list (no redundant computation!)
        for size in sorted(feature_set_sizes, reverse=True):  # Process from largest to smallest
            tprint_info(f"📊 Creating feature set for size {size} (slicing from top {max_size})...")
            
            # Slice the top N features from the already-ranked list
            selected_features = all_selected_features[:size]
            
            tprint_error(f"🔍 DEBUG for size {size}:")
            tprint_error(f"   Selected features count: {len(selected_features)}")
            tprint_error(f"   Selected features sample: {selected_features[:5]}")
            
            if not selected_features:
                tprint_error(f"❌ CRITICAL: No features for size {size}!")
                continue
            
            feature_sets[f'selected_features_{size}'] = selected_features

            # CRITICAL FIX: Validate features exist in features_df before creating dataframe
            available_features = [f for f in selected_features if f in features_df.columns]
            missing_features = [f for f in selected_features if f not in features_df.columns]
            
            tprint_error(f"   Features available in features_df: {len(available_features)}/{len(selected_features)}")
            if missing_features:
                tprint_error(f"   ❌ Missing features: {missing_features[:10]}..." if len(missing_features) > 10 else f"   ❌ Missing features: {missing_features}")
            
            if not available_features:
                tprint_error(f"❌ CRITICAL: NO features from selected list exist in features_df!")
                tprint_error(f"   Selected features: {selected_features[:5]}")
                tprint_error(f"   features_df columns: {list(features_df.columns)[:10]}")
                continue
            
            # Create dataframe with available features + targets
            all_cols_to_include = available_features + target_cols
            selected_dataframe = features_df[all_cols_to_include].copy()
            
            tprint_success(f"✅ Created selected_feature_dataframe_{size}:")
            tprint_success(f"   Shape: {selected_dataframe.shape}")
            tprint_success(f"   Features: {len(available_features)}")
            tprint_success(f"   Rows: {len(selected_dataframe)}")
            tprint_success(f"   Time range: {selected_dataframe.index.min()} to {selected_dataframe.index.max()}")
            
            feature_sets[f'selected_feature_dataframe_{size}'] = selected_dataframe

        tprint_success(f"✅ Created {len(feature_sets)} feature sets using optimized selection (1 computation instead of {len(feature_set_sizes)})")
        return feature_sets

    def _detect_tactician_mode(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> bool:
        """
        Detect if we're in Tactician mode based on launcher commands and available features.
        
        Args:
            features_df: Combined features dataframe
            config: Configuration dictionary
            
        Returns:
            True if in Tactician mode, False otherwise
        """
        # Primary detection: Check current step name for Tactician training steps
        # This is the most reliable method since it comes directly from ares_launcher.py
        current_step_name = getattr(self, 'step_name', '')
        is_tactician_training_step = (
            'tactician_base_training' in current_step_name or
            'tactician_ensemble_training' in current_step_name or
            'tactician' in current_step_name.lower()
        )
        
        # Also check if we're in a Tactician execution context
        # This could be set by upstream steps or the launcher
        tactician_execution_context = config.get('execution_context', '').lower()
        is_tactician_context = 'tactician' in tactician_execution_context
        
        # Secondary detection: Check for Tactician-specific features
        tactician_features = [col for col in features_df.columns if 'tactician' in col.lower()]
        
        # Tertiary detection: Check for CMI-based Tactician features
        cmi_tactician_features = [col for col in features_df.columns if 'cmi' in col.lower()]
        
        # Quaternary detection: Check configuration for explicit Tactician mode
        explicit_tactician_mode = config.get('tactician_mode', False)
        
        # Quinary detection: Check for Analyst features (if present, we might be in complementarity mode)
        analyst_features = [col for col in features_df.columns if 'analyst' in col.lower()]
        
        # Determine mode based on step name (primary) or feature analysis (secondary)
        is_tactician_mode = (
            is_tactician_training_step or
            is_tactician_context or
            len(tactician_features) > 0 or 
            len(cmi_tactician_features) > 0 or 
            explicit_tactician_mode or
            (len(analyst_features) > 0 and config.get('enable_cmi_complementarity', False))
        )
        
        tprint_info(f"🔍 Tactician mode detection:")
        tprint_info(f"  - Current step name: {current_step_name}")
        tprint_info(f"  - Is Tactician training step: {is_tactician_training_step}")
        tprint_info(f"  - Execution context: {config.get('execution_context', 'N/A')}")
        tprint_info(f"  - Is Tactician context: {is_tactician_context}")
        tprint_info(f"  - Tactician features: {len(tactician_features)}")
        tprint_info(f"  - CMI Tactician features: {len(cmi_tactician_features)}")
        tprint_info(f"  - Analyst features: {len(analyst_features)}")
        tprint_info(f"  - Explicit Tactician mode: {explicit_tactician_mode}")
        tprint_info(f"  - CMI complementarity enabled: {config.get('enable_cmi_complementarity', False)}")
        tprint_info(f"  - Detected Tactician mode: {is_tactician_mode}")
        
        return is_tactician_mode

    def _perform_enhanced_analysis(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive enhanced analysis on selected features.
        
        Args:
            X: Feature matrix
            y: Target variable
            selected_features: List of selected features
            
        Returns:
            Dictionary containing all enhanced analysis results
        """
        try:
            tprint_info("🔍 Starting enhanced feature analysis...")
            
            # Create a temporary component for analysis
            temp_config = FinalFeatureSelectionConfig(
                max_features=len(selected_features),
                min_features=5,
                selection_method='permutation',
                scoring_threshold=0.01,
                use_tree_based=True,
                use_permutation_importance=True
            )
            temp_component = FinalFeatureSelectionComponent(temp_config)
            
            # Perform all enhanced analyses
            analysis_results = {}
            
            # Get method results if available
            method_results = getattr(temp_component, 'method_results', None)
            
            # 1. Correlation Analysis
            tprint_info("📊 Performing correlation analysis...")
            correlation_analysis = temp_component.analyze_feature_correlations(X, selected_features)
            analysis_results['correlation_analysis'] = correlation_analysis
            
            # 2. Redundancy Detection - SKIPPED (too slow, correlation analysis is sufficient)
            tprint_info("🔍 Skipping redundancy detection (using correlation analysis instead)...")
            redundancy_analysis = {'skipped': True, 'reason': 'Performance optimization - correlation analysis provides sufficient information'}
            analysis_results['redundancy_analysis'] = redundancy_analysis
            
            # 3. Stability Analysis
            tprint_info("⏰ Analyzing feature stability across time windows...")
            stability_analysis = temp_component.analyze_feature_stability(X, y, selected_features, n_windows=5)
            analysis_results['stability_analysis'] = stability_analysis
            
            # 4. Cross-validation Analysis
            tprint_info("🔄 Performing cross-validation analysis...")
            # FIX: Increase CV folds to 10 for maximum coverage and stability
            cv_analysis = temp_component.cross_validate_feature_selection(X, y, selected_features, cv_folds=10)
            analysis_results['cv_analysis'] = cv_analysis
            
            # 5. Baseline Comparison
            tprint_info("📈 Comparing with baseline random selection...")
            baseline_analysis = temp_component.compare_with_baseline(X, y, selected_features)
            analysis_results['baseline_analysis'] = baseline_analysis
            
            # 6. Method Results Analysis (if available)
            if method_results:
                tprint_info("🔍 Analyzing method-specific results...")
                analysis_results['method_analysis'] = {
                    'methods_used': list(method_results.keys()),
                    'method_results': method_results,
                    'lgbm_shap_available': 'lgbm_shap' in method_results and 'error' not in method_results.get('lgbm_shap', {}),
                    'shap_scores': method_results.get('lgbm_shap', {}).get('scores', []) if 'lgbm_shap' in method_results else []
                }
            
            tprint_success("✅ Enhanced analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            tprint_error(f"❌ Error in enhanced analysis: {e}")
            return {"error": str(e)}

    def _perform_cmi_aware_selection(self, features_df: pd.DataFrame, targets: pd.Series, 
                                   config: Dict[str, Any], feature_set_sizes: List[int]) -> Dict[str, List[str]]:
        """
        Perform CMI-aware feature selection for Tactician mode.
        
        Args:
            features_df: Combined features dataframe
            targets: Target variables
            config: Configuration dictionary
            feature_set_sizes: List of feature set sizes to create
            
        Returns:
            Dictionary of feature sets
        """
        tprint_info("🎯 Performing CMI-aware feature selection for Tactician mode (with permutation importance)...")
        
        try:
            # Extract Analyst side information for CMI conditioning
            analyst_side_info = self._extract_analyst_side_info_for_cmi(features_df, config)
            
            if not analyst_side_info.get('cmi_enabled', False):
                tprint_warning("⚠️ CMI not available, falling back to standard selection")
                return self._perform_standard_selection(features_df, targets, config, feature_set_sizes)
            
            # Separate Tactician and Analyst features
            tactician_features = [col for col in features_df.columns 
                                if 'tactician' in col.lower() or 'cmi' in col.lower()]
            analyst_features = [col for col in features_df.columns 
                              if 'analyst' in col.lower()]
            other_features = [col for col in features_df.columns 
                            if col not in tactician_features + analyst_features 
                            and col not in TARGET_COLUMN_NAMES + ['timestamp']]
            
            tprint_info(f"🔍 Feature separation:")
            tprint_info(f"  - Tactician features: {len(tactician_features)}")
            tprint_info(f"  - Analyst features: {len(analyst_features)}")
            tprint_info(f"  - Other features: {len(other_features)}")
            
            # Prepare features for CMI selection
            all_features = tactician_features + other_features
            if not all_features:
                tprint_warning("⚠️ No features available for CMI selection")
                return self._perform_standard_selection(features_df, targets, config, feature_set_sizes)
            
            X = features_df[all_features]
            y = features_df[targets.name] if hasattr(targets, 'name') else targets
            
            # Perform CMI-based selection for each size
            feature_sets = {}
            for size in feature_set_sizes:
                tprint_info(f"🎯 CMI-based selection for {size} features...")
                
                # Use CMI scorer for feature selection
                cmi_result = self.cmi_scorer.score_features(
                    features=X,
                    targets=y,
                    analyst_outputs=analyst_side_info['side_info'].analyst_outputs,
                    regime_labels=analyst_side_info['side_info'].regime_labels
                )
                
                # Get selected features from result
                selected_features = cmi_result.selected_features[:size] if hasattr(cmi_result, 'selected_features') else []
                
                # If no selected features, fall back to top features by score
                if not selected_features and hasattr(cmi_result, 'complementarity_scores'):
                    sorted_features = sorted(
                        cmi_result.complementarity_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    selected_features = [f[0] for f in sorted_features[:size]]
                
                feature_sets[f'selected_features_{size}'] = selected_features
                
                # Get target column name
                target_col = targets.name if hasattr(targets, 'name') else 'target'
                target_cols = [target_col] if target_col in features_df.columns else []
                
                feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + target_cols].copy()
                
                tprint_success(f"✅ CMI-based selection completed: {len(selected_features)} features selected")
            
            return feature_sets
            
        except Exception as e:
            tprint_error(f"❌ CMI-aware selection failed: {e}")
            tprint_warning("⚠️ Falling back to standard selection")
            return self._perform_standard_selection(features_df, targets, config, feature_set_sizes)

    def _extract_analyst_side_info_for_cmi(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Analyst side information for CMI conditioning.
        
        Args:
            features_df: Combined features dataframe
            config: Configuration dictionary
            
        Returns:
            Dictionary containing Analyst side information and CMI configuration
        """
        if not CMI_COMPLEMENTARITY_AVAILABLE or self.analyst_handler is None:
            return {
                'cmi_enabled': False,
                'reason': 'CMI complementarity not available'
            }
        
        try:
            # Extract Analyst features
            analyst_features = [col for col in features_df.columns if 'analyst' in col.lower()]
            
            if not analyst_features:
                return {
                    'cmi_enabled': False,
                    'reason': 'No Analyst features found'
                }
            
            # Create Analyst features dataframe
            analyst_df = features_df[analyst_features]
            
            # Extract Analyst side information using emit_analyst_side_info
            pipeline_state = config.get('pipeline_state', {})
            pipeline_state['analyst_features'] = analyst_df
            
            analyst_side_info_result = self.analyst_handler.emit_analyst_side_info(
                pipeline_state=pipeline_state,
                targets=None,  # Will be provided later
                data_index=analyst_df.index
            )
            
            # Check if we got valid analyst outputs
            if analyst_side_info_result.analyst_outputs is not None:
                return {
                    'cmi_enabled': True,
                    'analyst_features': analyst_df,
                    'side_info': analyst_side_info_result
                }
            else:
                return {
                    'cmi_enabled': False,
                    'reason': 'No Analyst outputs available'
                }
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract Analyst side information: {e}")
            return {
                'cmi_enabled': False,
                'reason': f'Extraction failed: {e}'
            }

    def _perform_standard_selection(self, features_df: pd.DataFrame, targets: pd.Series, 
                                  config: Dict[str, Any], feature_set_sizes: List[int]) -> Dict[str, List[str]]:
        """
        Perform standard feature selection (fallback method).
        
        Args:
            features_df: Combined features dataframe
            targets: Target variables
            config: Configuration dictionary
            feature_set_sizes: List of feature set sizes to create
            
        Returns:
            Dictionary of feature sets
        """
        tprint_info("📊 Performing standard feature selection with permutation importance...")
        
        # Use the original selection logic
        feature_sets = {}
        
        # Separate features from targets and exclude raw data columns
        raw_data_columns = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day_of_week', 'base_threshold']
        
        # CRITICAL: Exclude performance metrics and forward-looking columns
        performance_metrics = [
            'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'recovery_factor',
            'win_rate', 'profit_factor', 'total_return', 'annualized_return', 'volatility',
            'var_95', 'cvar_95', 'downside_deviation', 'upside_capture', 'downside_capture',
            'information_ratio', 'treynor_ratio', 'jensen_alpha', 'max_consecutive_wins',
            'max_consecutive_losses', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
            'equity_curve', 'cumulative_returns', 'drawdown', 'underwater_curve'
        ]
        
        # Combine all columns to exclude
        excluded_columns = TARGET_COLUMN_NAMES + ['timestamp'] + raw_data_columns + performance_metrics
        
        # Prioritize sophisticated engineered features over basic ones
        sophisticated_features = [col for col in features_df.columns
                                if col not in excluded_columns
                                and any(keyword in col.lower() for keyword in ['vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced', 'statistical', 'wavelet', 'entropy', 'ad_line', 'obv', 'volatility', 'order_flow'])]
        
        basic_engineered_features = [col for col in features_df.columns
                                   if col not in excluded_columns
                                   and col not in sophisticated_features]
        
        # Prioritize sophisticated features first
        feature_cols = sophisticated_features + basic_engineered_features
        
        # Check for new simplified target structure first (highest priority)
        if 'target_long' in features_df.columns and 'target_short' in features_df.columns:
            target_cols = ['target_long', 'target_short']
            tprint_info("📊 Using new simplified target structure: target_long, target_short")
        else:
            # Fall back to legacy target detection
            target_cols = [col for col in TARGET_COLUMN_NAMES
                          if col in features_df.columns]
            tprint_info(f"📊 Using legacy target detection: {target_cols}")

        if not target_cols:
            raise ValueError("No target column found in features dataframe")

        if not feature_cols:
            raise ValueError("No feature columns found in features dataframe")

        X = features_df[feature_cols]
        y = features_df[target_cols[0]]

        # Create selection configs for different sizes
        for size in feature_set_sizes:
            tprint_info(f"🎯 Selecting top {size} features using permutation importance (captures interactions)...")

            # Create config for this size
            size_config = FinalFeatureSelectionConfig(
                max_features=size,
                min_features=max(5, size // 2),  # Minimum is half the size or 5, whichever is larger
                selection_method=config.get('selection_method', 'permutation'),
                scoring_threshold=config.get('scoring_threshold', 0.01),
                use_tree_based=config.get('use_tree_based', True),
                use_permutation_importance=config.get('use_permutation_importance', True)
            )

            # Create temporary component for this selection
            temp_component = FinalFeatureSelectionComponent(size_config)
            selected_features = temp_component.select_features(X, y, feature_cols)

            feature_sets[f'selected_features_{size}'] = selected_features

            # Also create the corresponding dataframes
            feature_sets[f'selected_feature_dataframe_{size}'] = features_df[selected_features + target_cols].copy()

        tprint_success(f"✅ Created {len(feature_sets)} feature sets using permutation-based importance")
        return feature_sets

    def _generate_shap_values(self, feature_sets: Dict[str, List[str]], features_df: pd.DataFrame, targets: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP values for interpretability."""
        shap_values = {}

        try:
            # Import SHAP (optional import)
            import shap
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            import warnings
            
            # Suppress NumPy deprecation warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*np\.bool.*")
                warnings.filterwarnings("ignore", message=".*np\.int.*")
                warnings.filterwarnings("ignore", message=".*np\.float.*")
                warnings.filterwarnings("ignore", message=".*np\.complex.*")

            # Get target column with priority for new simplified target structure
            if 'target_long' in features_df.columns and 'target_short' in features_df.columns:
                target_cols = ['target_long', 'target_short']
            else:
                # Fall back to legacy target detection
                target_cols = [col for col in TARGET_COLUMN_NAMES
                              if col in features_df.columns]
            if not target_cols:
                tprint_warning("⚠️ No target column found for SHAP analysis")
                return shap_values

            target_col = target_cols[0]
            feature_cols = [col for col in features_df.columns if col != target_col]

            X = features_df[feature_cols]
            y = features_df[target_col]

            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a simple model for SHAP analysis
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

                # Create SHAP explainer with additivity check disabled
            explainer = shap.TreeExplainer(rf_model)

            # Calculate SHAP values for each feature set
            for set_name, feature_list in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    size = set_name.split('_')[-1]
                    tprint_info(f"🔄 Processing SHAP for {size} features...")

                    if len(feature_list) > 0:
                        # Get SHAP values for this feature set with additivity check disabled
                        # Skip SHAP for now - it's causing hangs
                        tprint_warning(f"  ⚠️ Skipping SHAP computation (known to hang) - using feature importance instead")
                        continue
                        # tprint_info(f"  Computing SHAP values...")
                        # shap_test = explainer.shap_values(X_test[feature_list], check_additivity=False)
                        # tprint_info(f"  ✅ SHAP computation done, shape: {shap_test.shape if hasattr(shap_test, 'shape') else 'N/A'}")

                        # Store SHAP summary (avoid large tolist() conversions)
                        tprint_info(f"  Storing SHAP summary...")
                        mean_abs = np.mean(np.abs(shap_test), axis=0)
                        shap_values[f'shap_values_{size}'] = {
                            'shap_values': None,  # Don't store full values to avoid memory issues
                            'feature_names': feature_list,
                            'mean_abs_shap': mean_abs.tolist(),
                            'feature_importance': dict(zip(feature_list, mean_abs))
                        }
                        tprint_info(f"  ✅ Stored summary")

                        tprint_info(f"📊 Generated SHAP values for {size} features")

            tprint_success("✅ SHAP value generation completed")

        except ImportError:
            tprint_warning("⚠️ SHAP not available, skipping SHAP value generation")
        except Exception as e:
            tprint_warning(f"⚠️ Error generating SHAP values: {e}")

        return shap_values

    def _generate_artifacts(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any], combined_features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate artifacts from feature selection results.
        
        Feature lists and dataframes are saved to versioned artifacts (HDF5) with proper naming:
        Format: {artifact_name}_{date}_{mode}_{direction}_{symbol}_{exchange}
        
        Other artifacts (metadata, scores) are saved to regular artifacts directory.
        """
        artifacts = {}
        
        # Get context for versioned artifact naming
        symbol = config.get('symbol', 'UNKNOWN')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'long')
        mode = config.get('execution_mode', 'analyst')
        date_str = datetime.now().strftime('%Y%m%d')

        # Feature sets - these will be saved to versioned artifacts
        # CRITICAL FIX: Only save selected_feature_dataframe_* (with actual data values)
        # Do NOT save selected_features_* (just feature name lists) to HDF5
        for set_name, feature_list in feature_sets.items():
            if set_name.startswith('selected_features_'):
                # CRITICAL: Create the actual feature dataframe with VALUES for all 180 days
                # Extract the selected features from the full dataset
                size = set_name.split('_')[-1]  # e.g., "60" from "selected_features_60"
                dataframe_name = f'selected_feature_dataframe_{size}'
                
                # CRITICAL DEBUG: Check what we're trying to match
                tprint_error(f"🔍 DEBUG {set_name}:")
                tprint_error(f"   Feature list length: {len(feature_list)}")
                tprint_error(f"   Feature list sample: {feature_list[:5] if len(feature_list) > 0 else 'EMPTY'}")
                tprint_error(f"   Combined features columns: {len(combined_features_df.columns)}")
                tprint_error(f"   Combined features sample: {list(combined_features_df.columns)[:5]}")
                
                # Get the actual feature data for these selected features + target columns
                available_features = [f for f in feature_list if f in combined_features_df.columns]
                
                # Add target columns
                target_cols = [col for col in combined_features_df.columns if 'target' in col.lower()]
                all_cols = available_features + target_cols
                
                tprint_error(f"   Available features found: {len(available_features)}")
                tprint_error(f"   Target columns found: {target_cols}")
                
                if available_features:
                    # Create dataframe with selected features + targets for ALL rows (180 days in blank mode)
                    selected_data = combined_features_df[all_cols].copy()
                    artifacts[dataframe_name] = selected_data
                    tprint_success(f"✅ Created {dataframe_name}:")
                    tprint_success(f"   Features: {len(available_features)}")
                    tprint_success(f"   Rows: {len(selected_data)} (full 180-day dataset)")
                    tprint_success(f"   Time range: {selected_data.index.min()} to {selected_data.index.max()}")
                    tprint_success(f"   Targets: {target_cols}")
                else:
                    tprint_error(f"❌ CRITICAL: No features from {set_name} found in combined_features_df!")
                    tprint_error(f"❌ This means feature_list is empty or feature names don't match!")
                    tprint_error(f"❌ Saving ONLY targets for now...")
                
                # DO NOT save the feature name list to HDF5 - it causes errors
                # Feature names are already in the dataframe columns
                
            elif set_name.startswith('selected_feature_dataframe_'):
                # Feature dataframes - already in correct format for HDF5
                # Artifact name: selected_feature_dataframe_60 (versioning system adds timestamp)
                data = feature_sets[set_name]
                if isinstance(data, pd.DataFrame) and len(data) > 0:
                    artifacts[set_name] = data
                    tprint_success(f"✅ Saving {set_name}: {data.shape}")
                else:
                    tprint_warning(f"⚠️ Skipping {set_name} - not a valid dataframe or empty")

        # Feature scores from selection component (regular artifact)
        if self.selection_component:
            artifacts['feature_scores'] = self.selection_component.get_feature_scores()

        # SHAP values (regular artifact)
        for shap_name, shap_data in shap_values.items():
            artifacts[shap_name] = shap_data

        # Selection metadata (regular artifact)
        selection_metadata = {
            'total_features_available': len([col for col in combined_features_df.columns
                                           if col not in TARGET_COLUMN_NAMES + ['timestamp']]),
            'feature_set_sizes': config.get('feature_set_sizes', [60, 50, 40]),
            'selection_method': config.get('selection_method', 'permutation'),
            'scoring_threshold': config.get('scoring_threshold', 0.01),
            'use_tree_based': config.get('use_tree_based', True),
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'direction': direction,
            'mode': mode,
            'execution_mode': config.get('execution_mode', 'light')
        }
        artifacts['selection_metadata'] = selection_metadata

        return artifacts

    def _calculate_metrics(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for the feature selection."""
        total_features_selected = sum(len(features) for name, features in feature_sets.items()
                                    if name.startswith('selected_features_'))

        metrics = {
            'total_features_selected': total_features_selected,
            'feature_sets_created': len([name for name in feature_sets.keys() if name.startswith('selected_features_')]),
            'shap_values_generated': len(shap_values),
            'execution_timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'execution_mode': config.get('execution_mode', 'light')
        }

        # Feature set details
        for set_name, feature_list in feature_sets.items():
            if set_name.startswith('selected_features_'):
                size = set_name.split('_')[-1]
                metrics[f'features_{size}'] = len(feature_list)

        return metrics

    def _summarize_shap_values(self, shap_values: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of SHAP values."""
        summary = {}

        for shap_name, shap_data in shap_values.items():
            if shap_name.startswith('shap_values_'):
                size = shap_name.split('_')[-1]
                if isinstance(shap_data, dict) and 'feature_importance' in shap_data:
                    top_features = sorted(shap_data['feature_importance'].items(),
                                        key=lambda x: x[1], reverse=True)[:10]
                    summary[f'top_10_features_{size}'] = top_features

        return summary

    def _get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        metrics = {
            'optimization_enabled': self.optimization_enabled,
            'vectorization_manager_available': self.vectorization_manager is not None,
            'rolling_optimizer_available': self.rolling_optimizer is not None,
            'hardware_optimization_enabled': self.hardware_optimization_enabled,
            'hardware_manager_available': self.hardware_manager is not None,
            'adaptive_engine_available': self.adaptive_engine is not None,
            'cpu_optimizer_available': self.cpu_optimizer is not None,
            'gpu_manager_available': self.gpu_manager is not None,
            'memory_optimizer_available': self.memory_optimizer is not None
        }
        
        if self.vectorization_manager and self.optimization_enabled:
            try:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
                metrics.update({
                    'vectorization_operations': vectorization_stats.get('total_operations', 0),
                    'vectorbt_operations': vectorization_stats.get('vectorbt_operations', 0),
                    'memory_optimizations': vectorization_stats.get('memory_optimizations', 0),
                    'cache_hit_rate': vectorization_stats.get('cache_hit_rate', 0),
                    'batch_operations': vectorization_stats.get('batch_operations', 0)
                })
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get vectorization stats: {e}")
        
        if self.rolling_optimizer and self.optimization_enabled:
            try:
                rolling_stats = self.rolling_optimizer.get_performance_stats()
                metrics.update({
                    'rolling_operations': rolling_stats.get('total_operations', 0),
                    'vectorbt_rolling_operations': rolling_stats.get('vectorbt_operations', 0),
                    'rolling_optimization_rate': rolling_stats.get('vectorbt_usage_rate', 0)
                })
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get rolling optimizer stats: {e}")
        
        # Add hardware optimization metrics
        if self.hardware_manager and self.hardware_optimization_enabled:
            try:
                # Check if the method exists before calling it
                if hasattr(self.hardware_manager, 'get_performance_metrics'):
                    hardware_stats = self.hardware_manager.get_performance_metrics()
                    metrics.update({
                        'hardware_optimization_operations': hardware_stats.get('total_operations', 0),
                        'cpu_optimization_operations': hardware_stats.get('cpu_optimizations', 0),
                        'gpu_optimization_operations': hardware_stats.get('gpu_optimizations', 0),
                        'memory_optimization_operations': hardware_stats.get('memory_optimizations', 0),
                        'adaptive_optimization_operations': hardware_stats.get('adaptive_optimizations', 0)
                })
                else:
                    # Use default values if method doesn't exist
                    metrics.update({
                        'hardware_optimization_operations': 0,
                        'cpu_optimization_operations': 0,
                        'gpu_optimization_operations': 0,
                        'memory_optimization_operations': 0,
                        'adaptive_optimization_operations': 0
                })
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get hardware optimization stats: {e}")
        
        return metrics

    def _get_vectorization_stats(self) -> Dict[str, Any]:
        """Get detailed vectorization statistics."""
        if not self.vectorization_manager or not self.optimization_enabled:
            return {'enabled': False}
        
        try:
            stats = self.vectorization_manager.get_performance_stats()
            analytics = self.vectorization_manager.get_performance_analytics()
            
            return {
                'enabled': True,
                'performance_stats': stats,
                'analytics': analytics,
                'memory_profiling': self.vectorization_manager.get_memory_profiling(),
                'cache_statistics': self.vectorization_manager.get_cache_statistics()
            }
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get detailed vectorization stats: {e}")
            return {'enabled': True, 'error': str(e)}

    def _create_outcome_report(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create comprehensive outcome report."""
        try:
            report = f"""# Final Feature Selection Outcome Report

**Execution Details:**
- **Symbol:** {config.get('symbol', 'unknown')}
- **Exchange:** {config.get('exchange', 'binance')}
- **Timeframe:** {config.get('timeframe', '15m')}
- **Execution Mode:** {config.get('execution_mode', 'light')}
- **Timestamp:** {datetime.now().isoformat()}

## Feature Selection Summary

**Feature Set Sizes:** {config.get('feature_set_sizes', [60, 50, 40])}

**Selection Methodology:**
- ✅ **Using Permutation Importance** (captures feature interactions)
- 🔬 Unlike standard Gini importance, permutation importance measures how features work together
- 📊 More reliable for complex trading strategies with feature dependencies

**Results:**
"""

            # Feature set details
            for set_name, feature_list in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    size = set_name.split('_')[-1]
                    report += f"\n### Top {size} Features\n"
                    report += f"- **Count:** {len(feature_list)}\n"
                    if feature_list:
                        report += f"- **Top 5 Features:** {', '.join(feature_list[:5])}\n"

            # Feature scores if available
            if self.selection_component and self.selection_component.get_feature_scores():
                scores = self.selection_component.get_feature_scores()
                if scores:
                    report += "\n## Feature Importance Scores (Permutation-Based)\n"
                    report += "These scores measure true predictive impact including feature interactions.\n\n"
                    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
                    for feature, score in top_scores:
                        report += f"- **{feature}:** {score:.6f}\n"

            # SHAP summary
            if shap_values:
                report += "\n## SHAP Analysis\n"
                report += f"- **SHAP sets generated:** {len(shap_values)}\n"

                for shap_name, shap_data in shap_values.items():
                    if shap_name.startswith('shap_values_'):
                        size = shap_name.split('_')[-1]
                        if isinstance(shap_data, dict) and 'feature_importance' in shap_data:
                            top_shap = sorted(shap_data['feature_importance'].items(),
                                            key=lambda x: x[1], reverse=True)[:5]
                            report += f"\n**Top SHAP Features ({size}):**\n"
                            for feature, importance in top_shap:
                                report += f"- {feature}: {importance:.6f}\n"

            # Configuration
            report += "\n## Configuration\n"
            report += f"- **Selection Method:** {config.get('selection_method', 'permutation')} ✅\n"
            report += f"- **Use Permutation Importance:** {config.get('use_permutation_importance', True)} (captures feature interactions) ✅\n"
            report += f"- **Importance Type:** Permutation (not Gini) - More reliable for interaction-heavy models 📊\n"
            report += f"- **Scoring Threshold:** {config.get('scoring_threshold', 0.01)}\n"
            report += f"- **Tree-based Selection:** {config.get('use_tree_based', True)}\n"
            report += f"- **Why Permutation?** Measures true impact on predictions, not just split quality\n"
            
            # Optimization Information
            report += "\n## Optimization Status\n"
            report += f"- **VectorBT Optimization:** {'Enabled' if self.optimization_enabled else 'Disabled'}\n"
            report += f"- **Hardware Optimization:** {'Enabled' if self.hardware_optimization_enabled else 'Disabled'}\n"
            report += f"- **Vectorization Manager:** {'Available' if self.vectorization_manager else 'Not Available'}\n"
            report += f"- **Rolling Optimizer:** {'Available' if self.rolling_optimizer else 'Not Available'}\n"
            report += f"- **Hardware Manager:** {'Available' if self.hardware_manager else 'Not Available'}\n"
            report += f"- **Adaptive Engine:** {'Available' if self.adaptive_engine else 'Not Available'}\n"
            report += f"- **CPU Optimizer:** {'Available' if self.cpu_optimizer else 'Not Available'}\n"
            report += f"- **GPU Manager:** {'Available' if self.gpu_manager else 'Not Available'}\n"
            report += f"- **Memory Optimizer:** {'Available' if self.memory_optimizer else 'Not Available'}\n"

            # Generated artifacts
            report += "\n## Generated Artifacts\n"
            artifact_count = len([name for name in feature_sets.keys() if name.startswith('selected_features_')]) * 2  # features + dataframes
            artifact_count += len(shap_values)
            artifact_count += 2  # feature_scores + selection_metadata

            report += f"- Feature sets: {len([name for name in feature_sets.keys() if name.startswith('selected_features_')])}\n"
            report += f"- Feature dataframes: {len([name for name in feature_sets.keys() if name.startswith('selected_feature_dataframe_')])}\n"
            report += f"- SHAP analyses: {len(shap_values)}\n"
            report += f"- Metadata and scores: 2\n"
            report += f"- **Total artifacts:** {artifact_count + 1}\n"  # +1 for the report

            report += f"""

---
*Generated by Feature Generation Final Feature Selection Step at {datetime.now().isoformat()}*
"""

            return report

        except Exception as e:
            tprint_error(f"⚠️ Failed to create outcome report: {e}")
            return f"# Final Feature Selection Outcome Report\n\nError creating report: {str(e)}"

    def _generate_markdown_report(self, outcome_report: Dict[str, Any], 
                                 feature_sets: Dict[str, List[str]], 
                                 shap_values: Dict[str, Any], 
                                 config: Dict[str, Any]) -> str:
        """
        Generate a comprehensive markdown report for the final feature selection step.
        
        Args:
            outcome_report: The outcome report dictionary
            feature_sets: Dictionary of feature sets
            shap_values: SHAP values dictionary
            config: Configuration object
            
        Returns:
            Markdown formatted report string
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""# Final Feature Selection Report

**Generated:** {timestamp}
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** {config.get('symbol', 'N/A')}
- **Exchange:** {config.get('exchange', 'N/A')}
- **Timeframe:** {config.get('timeframe', 'N/A')}
- **Execution Mode:** {config.get('execution_mode', 'N/A')}
- **Feature Count Targets:** {config.get('feature_set_sizes', [60, 50, 40])}
- **Selection Method:** {config.get('selection_method', 'permutation')} ✅
- **Importance Type:** Permutation (captures feature interactions, not just Gini splits) 📊
- **Optimization Enabled:** {self.optimization_enabled}

## Feature Selection Methodology

✅ **Using Permutation Importance**
- Captures how features work together (feature interactions)
- More reliable than standard Gini importance for complex trading strategies
- Measures true impact on model predictions
- Better for identifying genuinely predictive features

## Feature Selection Results

"""
            
            # Add feature set summaries
            for set_name, features in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    count = set_name.split('_')[-1]
                    report += f"- **{count} Features Set:** {len(features)} features selected\n"
            
            report += f"\n- **Total Feature Sets:** {len([k for k in feature_sets.keys() if k.startswith('selected_features_')])}\n"
            
            # Add SHAP analysis summary
            if shap_values:
                report += f"\n## SHAP Analysis Summary\n\n"
                report += f"- **SHAP Analyses Generated:** {len(shap_values)}\n"
                for shap_name, shap_data in shap_values.items():
                    if isinstance(shap_data, dict) and 'top_features' in shap_data:
                        report += f"- **{shap_name}:** {len(shap_data['top_features'])} top features analyzed\n"
            
            # Add detailed feature lists with SHAP metrics
            report += f"\n## Selected Features by Set\n\n"
            
            for set_name, features in feature_sets.items():
                if set_name.startswith('selected_features_'):
                    count = set_name.split('_')[-1]
                    report += f"### {count} Features Set ({len(features)} features)\n\n"
                    
                    # Get SHAP values for this feature set if available
                    shap_key = f'shap_values_{count}'
                    feature_importance = {}
                    if shap_key in shap_values and isinstance(shap_values[shap_key], dict):
                        feature_importance = shap_values[shap_key].get('feature_importance', {})
                    
                    for i, feature in enumerate(features[:20], 1):  # Show first 20 features
                        shap_score = feature_importance.get(feature, 0.0)
                        report += f"{i}. {feature}"
                        if shap_score > 0:
                            report += f" (SHAP: {shap_score:.4f})"
                        report += "\n"
                    
                    if len(features) > 20:
                        report += f"... and {len(features) - 20} more features\n"
                    
                    # Add SHAP summary for this set
                    if shap_key in shap_values and isinstance(shap_values[shap_key], dict):
                        mean_abs_shap = shap_values[shap_key].get('mean_abs_shap', [])
                        if mean_abs_shap:
                            avg_shap = sum(mean_abs_shap) / len(mean_abs_shap)
                            report += f"\n**Average SHAP Importance:** {avg_shap:.4f}\n"
                    
                    report += "\n"
            
            # Add enhanced analysis results
            if 'enhanced_analysis' in feature_sets:
                enhanced_analysis = feature_sets['enhanced_analysis']
                report += f"\n## Enhanced Feature Analysis\n\n"
                
                # Correlation Analysis
                if 'correlation_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['correlation_analysis']:
                    corr_analysis = enhanced_analysis['correlation_analysis']
                    report += f"### Correlation Analysis\n\n"
                    report += f"- **Average Correlation:** {corr_analysis.get('average_correlation', 'N/A'):.4f}\n"
                    report += f"- **Max Correlation:** {corr_analysis.get('max_correlation', 'N/A'):.4f}\n"
                    report += f"- **Min Correlation:** {corr_analysis.get('min_correlation', 'N/A'):.4f}\n"
                    report += f"- **High Correlation Pairs:** {len(corr_analysis.get('high_correlation_pairs', []))}\n"
                    report += f"- **Correlation Threshold:** {corr_analysis.get('correlation_threshold', 'N/A')}\n\n"
                
                # Redundancy Analysis
                if 'redundancy_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['redundancy_analysis']:
                    red_analysis = enhanced_analysis['redundancy_analysis']
                    report += f"### Redundancy Detection\n\n"
                    # Handle skipped redundancy detection
                    if red_analysis.get('skipped'):
                        report += f"- **Status:** Skipped ({red_analysis.get('reason', 'Performance optimization')})\n\n"
                    else:
                        redundancy_score = red_analysis.get('redundancy_score', 'N/A')
                        if isinstance(redundancy_score, (int, float)):
                            report += f"- **Redundancy Score:** {redundancy_score:.4f}\n"
                        else:
                            report += f"- **Redundancy Score:** {redundancy_score}\n"
                        report += f"- **Redundant Features:** {red_analysis.get('redundant_features', 'N/A')}\n"
                        report += f"- **Total Features:** {red_analysis.get('total_features', 'N/A')}\n"
                        report += f"- **Correlation Redundant Pairs:** {len(red_analysis.get('redundancy_results', {}).get('correlation_redundant', []))}\n"
                        report += f"- **Mutual Info Redundant Pairs:** {len(red_analysis.get('redundancy_results', {}).get('mutual_info_redundant', []))}\n"
                        report += f"- **Low Variance Features:** {len(red_analysis.get('redundancy_results', {}).get('variance_redundant', []))}\n\n"
                
                # Stability Analysis
                if 'stability_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['stability_analysis']:
                    stab_analysis = enhanced_analysis['stability_analysis']
                    report += f"### Stability Analysis\n\n"
                    report += f"- **Average Stability:** {stab_analysis.get('average_stability', 'N/A'):.4f}\n"
                    report += f"- **Stable Features:** {len(stab_analysis.get('stable_features', []))}\n"
                    report += f"- **Stability Threshold:** {stab_analysis.get('stability_threshold', 'N/A')}\n"
                    report += f"- **Time Windows:** {stab_analysis.get('n_windows', 'N/A')}\n\n"
                
                # Cross-validation Analysis
                if 'cv_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['cv_analysis']:
                    cv_analysis = enhanced_analysis['cv_analysis']
                    report += f"### Cross-Validation Analysis\n\n"
                    report += f"- **Average Consistency:** {cv_analysis.get('average_consistency', 'N/A'):.4f}\n"
                    report += f"- **Consistent Features:** {len(cv_analysis.get('consistent_features', []))}\n"
                    report += f"- **Consistency Threshold:** {cv_analysis.get('consistency_threshold', 'N/A')}\n"
                    report += f"- **CV Folds:** {cv_analysis.get('cv_folds', 'N/A')}\n\n"
                
                # Baseline Comparison
                if 'baseline_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['baseline_analysis']:
                    base_analysis = enhanced_analysis['baseline_analysis']
                    report += f"### Baseline Comparison\n\n"
                    report += f"- **Improvement Ratio:** {base_analysis.get('improvement_ratio', 'N/A'):.2f}x\n"
                    report += f"- **Selected Features Avg Score:** {base_analysis.get('average_selected_score', 'N/A'):.6f}\n"
                    report += f"- **Baseline Avg Score:** {base_analysis.get('average_baseline_score', 'N/A'):.6f}\n"
                    report += f"- **Baseline Trials:** {base_analysis.get('n_baseline_trials', 'N/A')}\n"
                    report += f"- **Features Compared:** {base_analysis.get('n_features', 'N/A')}\n\n"
                
                # Method Analysis
                if 'method_analysis' in enhanced_analysis:
                    method_analysis = enhanced_analysis['method_analysis']
                    report += f"### Multi-Method Selection Analysis\n\n"
                    report += f"- **Methods Used:** {', '.join(method_analysis.get('methods_used', []))}\n"
                    report += f"- **LGBM-SHAP Available:** {'Yes' if method_analysis.get('lgbm_shap_available', False) else 'No'}\n"
                    
                    if method_analysis.get('lgbm_shap_available', False):
                        shap_scores = method_analysis.get('shap_scores', [])
                        if shap_scores:
                            report += f"- **SHAP Scores Range:** {min(shap_scores):.6f} - {max(shap_scores):.6f}\n"
                            report += f"- **Average SHAP Score:** {np.mean(shap_scores):.6f}\n"
                            report += f"- **Top SHAP Features:** {len([s for s in shap_scores if s > np.mean(shap_scores)])}\n"
                    
                    # Method-specific results
                    method_results = method_analysis.get('method_results', {})
                    for method_name, method_data in method_results.items():
                        if 'error' not in method_data:
                            report += f"- **{method_name.title()} Features:** {len(method_data.get('features', []))}\n"
                    report += f"\n"
            
            # Add performance metrics (removed inappropriate model metrics)
            report += f"## Performance Metrics\n\n"
            if isinstance(outcome_report, dict):
                report += f"- **Execution Time:** {outcome_report.get('execution_time', 'N/A')} seconds\n"
            else:
                report += f"- **Execution Time:** N/A seconds\n"
            report += f"- **Optimization Enabled:** {'Yes' if self.optimization_enabled else 'No'}\n"
            report += f"- **Hardware Optimization:** {'Yes' if self.hardware_optimization_enabled else 'No'}\n"
            
            # Add optimization details
            if self.optimization_enabled:
                report += f"\n## Optimization Details\n\n"
                report += f"- **VectorBT Optimization:** {'Enabled' if self.vectorization_manager else 'Disabled'}\n"
                report += f"- **Rolling Optimizer:** {'Available' if self.rolling_optimizer else 'Not Available'}\n"
                report += f"- **Hardware Manager:** {'Available' if self.hardware_manager else 'Not Available'}\n"
            
            # Add artifacts summary
            report += f"\n## Generated Artifacts\n\n"
            artifact_count = len([name for name in feature_sets.keys() if name.startswith('selected_features_')]) * 2
            artifact_count += len(shap_values) if shap_values else 0
            artifact_count += 2  # feature_scores + selection_metadata
            
            report += f"- **Feature Sets:** {len([name for name in feature_sets.keys() if name.startswith('selected_features_')])}\n"
            report += f"- **Feature DataFrames:** {len([name for name in feature_sets.keys() if name.startswith('selected_feature_dataframe_')])}\n"
            report += f"- **SHAP Analyses:** {len(shap_values) if shap_values else 0}\n"
            report += f"- **Metadata Files:** 2\n"
            report += f"- **Total Artifacts:** {artifact_count + 2}\n"  # +2 for pickle and markdown reports
            
            report += f"\n## Summary\n\n"
            report += f"Final feature selection completed successfully. Generated {len([k for k in feature_sets.keys() if k.startswith('selected_features_')])} optimized feature sets "
            report += f"with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.\n"
            
            report += f"\n---\n"
            report += f"*Generated by Feature Generation Final Feature Selection Step at {timestamp}*\n"
            
            return report
            
        except Exception as e:
            tprint_error(f"⚠️ Failed to generate markdown report: {e}")
            return f"# Final Feature Selection Report\n\nError generating report: {str(e)}"

    def _save_markdown_report(self, markdown_content: str, base_name: str) -> str:
        """
        Save a markdown report to the outcomes directory.
        
        Args:
            markdown_content: The markdown content to save
            base_name: Base name for the file
            
        Returns:
            Path where the markdown file was saved
        """
        try:
            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_report_{timestamp}.md"
            file_path = outcomes_dir / filename
            
            # Write markdown content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            tprint_success(f"✅ Markdown report saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            tprint_error(f"⚠️ Failed to save markdown report: {e}")
            raise


# Register the step
def register_feature_generation_final_feature_selection_step():
    """Register the feature generation final feature selection step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_final_feature_selection_step", FeatureGenerationFinalFeatureSelectionStep)
    tprint("✅ Feature generation final feature selection step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_final_feature_selection_step()
