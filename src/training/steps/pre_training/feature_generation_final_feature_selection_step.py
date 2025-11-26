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
import re
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

# Optional Polars support for upstream preprocessing compatibility
try:
    import polars as pl  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    pl = None

# Import BaseStep and step registry
from src.training.steps.base_step import BaseStep

# Import feature selection component
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent
)

from src.features_common.transforms.scaling_normalization import robust_normalize
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

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
from src.training.utils.meta_label_constants import (
    META_LABEL_TARGET_COLUMNS,
    META_LABEL_PRIMARY_TRAINING_TARGETS,
    META_LABEL_DIAGNOSTIC_COLUMNS,
    META_LABEL_EXCLUDED_FEATURE_COLUMNS,
)

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

# Keep backward-compatibility aliases within this module
TARGET_COLUMN_NAMES = META_LABEL_TARGET_COLUMNS + META_LABEL_DIAGNOSTIC_COLUMNS
PRIMARY_TARGET_COLUMN_NAMES = META_LABEL_PRIMARY_TRAINING_TARGETS


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
                    
                    # Check for time period overlap (best-effort; guard against
                    # mixed index dtypes such as integer vs bytes/datetime).
                    try:
                        labeled_start, labeled_end = labeled_df.index.min(), labeled_df.index.max()
                        features_start, features_end = large_features_df.index.min(), large_features_df.index.max()

                        tprint_info(f"📅 TIME PERIOD ANALYSIS:")
                        tprint_info(f"   Labeled data period: {labeled_start} to {labeled_end}")
                        tprint_info(f"   Features period: {features_start} to {features_end}")

                        # Only attempt overlap math when both endpoints look
                        # datetime-like; otherwise skip to avoid type errors.
                        from datetime import datetime as _dt_cls
                        def _is_dt_like(v):
                            return isinstance(v, (pd.Timestamp, _dt_cls))

                        if _is_dt_like(labeled_start) and _is_dt_like(labeled_end) and _is_dt_like(features_start) and _is_dt_like(features_end):
                            overlap_start = max(labeled_start, features_start)
                            overlap_end = min(labeled_end, features_end)
                            tprint_info(f"   Theoretical overlap: {overlap_start} to {overlap_end}")

                            if overlap_start <= overlap_end:
                                overlap_days = (overlap_end - overlap_start).days
                                tprint_info(f"   Overlap duration: {overlap_days} days")
                                expected_samples = overlap_days * 96  # 96 samples per day for 15m
                                tprint_info(f"   Expected samples in overlap: ~{expected_samples}")
                            else:
                                tprint_error("   ❌ NO TIME OVERLAP! Labeled ends before features start or vice versa")
                        else:
                            tprint_info("   Skipping detailed overlap analysis (non-datetime indices)")
                    except Exception as overlap_exc:
                        tprint_warning(f"   Skipping time period overlap analysis due to error: {overlap_exc}")
                    
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
            
            # Calculate time span (robust to non-datetime indices). When the
            # index is not datetime-like (e.g. RangeIndex), approximate the
            # span from the number of rows instead of relying on .days.
            try:
                idx_min = labeled_df.index.min()
                idx_max = labeled_df.index.max()
                delta = idx_max - idx_min
                if hasattr(delta, "days"):
                    time_span_days = float(delta.days)
                else:
                    # Fallback: approximate span from sample count
                    time_span_days = float(len(labeled_df)) / 96.0
            except Exception:
                time_span_days = float(len(labeled_df)) / 96.0

            expected_samples_at_15m = max(time_span_days, 0.0) * 96  # 96 samples per day for 15m timeframe
            tprint_error(f"   Time span (approx): {time_span_days:.2f} days")
            tprint_error(f"   Expected samples at 15m: ~{expected_samples_at_15m:.0f}")
            tprint_error(f"   Actual samples: {len(labeled_df)}")
            if expected_samples_at_15m > 0:
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
            
            # Blank-mode specific shaping: compress extremes and normalize key feature blocks
            combined_features_df = self._apply_blank_mode_shaping(combined_features_df, config)
            
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
                feature_cols = [
                    col for col in combined_features_df.columns
                    if col not in META_LABEL_EXCLUDED_FEATURE_COLUMNS + ['timestamp']
                ]
                X = combined_features_df[feature_cols]
                y = combined_features_df[targets.name] if hasattr(targets, 'name') else combined_features_df.iloc[:, -1]
                
                # Perform comprehensive analysis
                enhanced_analysis = self._perform_enhanced_analysis(X, y, largest_features)

                # Add explicit meta-label diagnostics (IC/AUC vs binary_label and realized_return)
                try:
                    meta_diag = self._compute_meta_label_feature_diagnostics(
                        combined_features_df, largest_features
                    )
                    enhanced_analysis['meta_label_diagnostics'] = meta_diag
                except Exception as diag_exc:
                    tprint_warning(f"⚠️ Meta-label diagnostics failed: {diag_exc}")
                
                # Store analysis results in feature_sets for report generation
                feature_sets['enhanced_analysis'] = enhanced_analysis
                tprint_info(f"✅ Enhanced analysis completed in {time.time()-t0:.2f}s")

            # Generate SHAP values for interpretability
            t0 = time.time()
            tprint_info("⏱️ [9/10] Generating SHAP values...")
            shap_values = self._generate_shap_values(feature_sets, combined_features_df, targets, config)
            tprint_info(f"✅ SHAP values generated in {time.time()-t0:.2f}s")

            # Run baseline predictive check if enabled
            baseline_check_results = None
            baseline_check_enabled = config.get('run_baseline_check', True)  # Default: enabled
            if baseline_check_enabled:
                t0 = time.time()
                tprint_info("🔍 Running baseline predictive check on selected features...")
                try:
                    # Run check on the largest feature set (60 features)
                    largest_set_name = None
                    for name in ['selected_feature_dataframe_60', 'selected_feature_dataframe_50', 'selected_feature_dataframe_40']:
                        if name in feature_sets:
                            largest_set_name = name
                            break

                    if largest_set_name and isinstance(feature_sets[largest_set_name], pd.DataFrame):
                        baseline_check_results = self._run_baseline_predictive_check(
                            feature_sets[largest_set_name],
                            targets,
                            config
                        )
                        if baseline_check_results and baseline_check_results.get('success', False):
                            tprint_success(f"✅ Baseline predictive check completed in {time.time()-t0:.2f}s")
                            # Add to feature_sets so it's included in the report
                            feature_sets['baseline_check'] = baseline_check_results
                        else:
                            tprint_warning(f"⚠️ Baseline predictive check failed or returned no results")
                    else:
                        tprint_warning(f"⚠️ No suitable feature set found for baseline check")
                except Exception as e:
                    tprint_warning(f"⚠️ Baseline predictive check failed: {e}")
                    logger.warning(f"Baseline predictive check failed: {e}")
            else:
                tprint_info(f"ℹ️ Baseline predictive check disabled (run_baseline_check=False)")

            # Generate artifacts
            t0 = time.time()
            tprint_info("⏱️ [10/10] Generating and saving artifacts...")
            artifacts = self._generate_artifacts(feature_sets, shap_values, config, combined_features_df)

            # Create comprehensive outcome report
            outcome_report = self._create_outcome_report(feature_sets, shap_values, config, baseline_check_results)

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
            error_msg = (
                "❌ CRITICAL: No interaction features found from any artifact name. "
                "Tried: " + ", ".join(interaction_artifact_names) + ". "
                "The interaction generation step must run successfully and emit 'analyst_interaction_features' (or an accepted alias) before final selection."
            )
            tprint_error(error_msg)
            raise ValueError(error_msg)

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

    def _apply_blank_mode_shaping(self, combined_features_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply blank-mode specific outlier compression and normalization to key feature blocks.

        This is a light, post-feature-generation shaping pass that:
        - only runs when execution_mode == 'blank'
        - targets heavy-tailed blocks (wavelet / volatility / AD-line features)
        - applies modest quantile winsorization followed by robust normalization
        """
        try:
            execution_mode = config.get("execution_mode", "blank")
            if execution_mode != "blank":
                return combined_features_df

            if combined_features_df is None or combined_features_df.empty:
                return combined_features_df

            df = combined_features_df.copy()

            # Identify numeric feature columns (exclude targets and timestamp)
            feature_cols: List[str] = [
                col
                for col in df.columns
                if col not in TARGET_COLUMN_NAMES + ["timestamp"]
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if not feature_cols:
                return df

            # Focus on the heavy-tailed families highlighted in validation reports
            wavelet_cols = [col for col in feature_cols if "wavelet" in col.lower()]
            volatility_cols = [
                col
                for col in feature_cols
                if "volatility" in col.lower()
                or "band_limited_volatility" in col.lower()
                or "enhanced_volatility" in col.lower()
            ]
            ad_line_cols = [
                col
                for col in feature_cols
                if "ad_line" in col.lower()
                or "accumulation_distribution" in col.lower()
                or "adline" in col.lower()
            ]

            blocks: List[Tuple[str, List[str]]] = [
                ("wavelet", wavelet_cols),
                ("volatility", volatility_cols),
                ("ad_line", ad_line_cols),
            ]

            # Default to conservative tails, but allow config override if needed
            lower_q = float(config.get("blank_mode_lower_quantile", 0.0025))
            upper_q = float(config.get("blank_mode_upper_quantile", 0.9975))

            tprint_info(
                f"🔧 [BLANK_SHAPING] Applying quantile winsorization/robust normalization "
                f"to key blocks (lower_q={lower_q}, upper_q={upper_q})"
            )

            for block_name, cols in blocks:
                if not cols:
                    continue

                try:
                    block_df = df[cols]

                    # Quantile-based clipping per column
                    lower_bounds = block_df.quantile(lower_q)
                    upper_bounds = block_df.quantile(upper_q)
                    block_clipped = block_df.clip(lower=lower_bounds, upper=upper_bounds, axis="columns")

                    # Robust normalization (median/IQR-style) on the clipped block
                    block_normalized = robust_normalize(block_clipped)

                    df[cols] = block_normalized
                    tprint_info(
                        f"📊 [BLANK_SHAPING] Block '{block_name}' shaped: {len(cols)} features"
                    )
                except Exception as e:
                    tprint_warning(
                        f"⚠️ [BLANK_SHAPING] Skipping block '{block_name}' due to error: {e}"
                    )

            return df
        except Exception as e:
            # Fail safe: never break the pipeline because of shaping
            tprint_warning(f"⚠️ [BLANK_SHAPING] Failed to apply shaping, returning original data: {e}")
            return combined_features_df

    def _combine_features(self, features_data: Dict[str, Any], labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Combine features from different sources into a single DataFrame with VectorBT optimizations."""
        tprint_error("🔄 CRITICAL DEBUG: Starting _combine_features method")
        
        # Polars compatibility: normalize any Polars DataFrames to pandas before combining.
        # This allows upstream steps to operate in Polars while keeping the existing
        # pandas/VectorBT-based feature selection logic unchanged.
        if pl is not None:
            # Normalize labeled_df if it is a Polars DataFrame
            if isinstance(labeled_df, pl.DataFrame):
                labeled_df = labeled_df.to_pandas()

            # Normalize DataFrames inside features_data if they are Polars
            normalized_features_data: Dict[str, Any] = {}
            for key, data in features_data.items():
                if isinstance(data, pl.DataFrame):
                    normalized_features_data[key] = data.to_pandas()
                else:
                    normalized_features_data[key] = data
            features_data = normalized_features_data
        
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
        
        # CRITICAL: Find common index across ALL dataframes, with a robust
        # fallback when indices do not overlap (e.g. RangeIndex vs bytes
        # datetime index). When the intersection is empty, fall back to
        # positional alignment on the last min_len rows and reset to a shared
        # RangeIndex so we never end up with a 0-row combined_features_df.
        tprint_info(f"📊 Finding common index across {len(dataframe_info)} dataframes...")
        common_index = dataframe_info[0]['df'].index
        for info in dataframe_info[1:]:
            common_index = common_index.intersection(info['df'].index)
            tprint_info(f"📊 After {info['name']}: {len(common_index)} common indices")

        if len(common_index) == 0:
            tprint_error("❌ No common indices across dataframes; falling back to positional alignment")

            # Positional fallback: align all dataframes on the last min_len
            # rows and reset indices to a shared RangeIndex.
            try:
                min_len = min(len(info['df']) for info in dataframe_info)
            except Exception as e:
                tprint_error(f"❌ Failed to compute min_len for positional alignment: {e}")
                min_len = 0

            if min_len <= 0:
                tprint_error("❌ Positional alignment failed: no non-empty dataframes")
                common_index = dataframe_info[0]['df'].index[:0]
            else:
                tprint_info(f"📊 Positional alignment using last {min_len} rows from each dataframe")
                for info in dataframe_info:
                    df_local = info['df']
                    if len(df_local) > min_len:
                        df_local = df_local.iloc[-min_len:].copy()
                    else:
                        df_local = df_local.copy()
                    df_local.index = pd.RangeIndex(min_len)
                    info['df'] = df_local
                    tprint_info(f"📊 Positional-aligned {info['name']}: {info['df'].shape}")

                # After positional alignment, indices are already a shared
                # RangeIndex, so we can define common_index accordingly.
                common_index = dataframe_info[0]['df'].index
        else:
            if len(common_index) < 100:
                tprint_error(f"❌ Too few common indices ({len(common_index)}). Minimum required: 100")
                tprint_warning("⚠️ This may cause feature selection to be unstable. Check data alignment issues.")

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
        
        # Debug: Check if target column is present with priority for binary_label, then new simplified target structure
        if 'binary_label' in result_df.columns:
            available_targets = ['binary_label']
            tprint_info("📊 Using primary binary target: binary_label")
            tprint_info(f"📊 Target column 'binary_label' non-NaN count: {result_df['binary_label'].notna().sum()}")
        # Next check for new simplified target structure (highest priority among price-based targets)
        elif 'target_long_fused' in result_df.columns and 'target_short_fused' in result_df.columns:
            available_targets = ['target_long_fused', 'target_short_fused']
            tprint_info("📊 Using fused target structure: target_long_fused, target_short_fused")
            tprint_info(f"📊 Target columns found: target_long_fused ({result_df['target_long_fused'].notna().sum()} non-NaN), target_short_fused ({result_df['target_short_fused'].notna().sum()} non-NaN)")
        elif 'target_long' in result_df.columns and 'target_short' in result_df.columns:
            available_targets = ['target_long', 'target_short']
            tprint_info("📊 Using new simplified target structure: target_long, target_short")
            tprint_info(f"📊 Target columns found: target_long ({result_df['target_long'].notna().sum()} non-NaN), target_short ({result_df['target_short'].notna().sum()} non-NaN)")
        else:
            # Fall back to legacy target detection
            available_targets = [col for col in PRIMARY_TARGET_COLUMN_NAMES if col in result_df.columns]
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
        nan_handling_succeeded = False

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
                nan_handling_succeeded = True
            except Exception as e:
                tprint_warning(f"⚠️ Optimized NaN handling failed, falling back: {e}")

        # Polars-optimized NaN handling when vectorization manager is not used
        if (not nan_handling_succeeded) and pl is not None and self.optimization_enabled:
            try:
                tprint_info("🔄 Polars-optimized NaN handling...")

                # Convert pandas result_df to Polars for efficient column-wise operations
                pl_df = pl.from_pandas(result_df)

                n_rows = len(pl_df)
                nan_threshold = int(0.5 * n_rows)

                sophisticated_keywords = [
                    'vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced',
                    'statistical', 'wavelet', 'entropy', 'ad_line', 'obv',
                    'volatility', 'order_flow'
                ]

                valid_cols = []
                for col in pl_df.columns:
                    if col in TARGET_COLUMN_NAMES or 'target' in col.lower():
                        valid_cols.append(col)
                        tprint_info(f"📊 Keeping target column (Polars path): {col}")
                        continue

                    # Count non-null values using Polars
                    non_null_count = pl_df.select(pl.col(col).is_not_null().sum().alias('cnt'))['cnt'][0]

                    if non_null_count >= nan_threshold:
                        valid_cols.append(col)
                    else:
                        # Check if it's a sophisticated feature and be more lenient
                        if any(keyword in col.lower() for keyword in sophisticated_keywords):
                            if non_null_count >= int(0.3 * n_rows):
                                valid_cols.append(col)
                                tprint_info(f"📊 Keeping sophisticated feature with low data coverage (Polars path): {col}")

                # Filter to valid columns only
                pl_df = pl_df.select(valid_cols)

                # Fill remaining NaN/null with median for numeric columns only
                numeric_cols_only = [
                    c for c, dt in zip(pl_df.columns, pl_df.dtypes)
                    if dt in (
                        pl.Float64, pl.Float32,
                        pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
                    )
                ]

                if numeric_cols_only:
                    medians_row = pl_df.select([
                        pl.col(c).median().alias(c) for c in numeric_cols_only
                    ]).to_dicts()[0]

                    pl_df = pl_df.with_columns([
                        pl.col(c)
                        .fill_nan(medians_row[c])
                        .fill_null(medians_row[c])
                        .alias(c)
                        for c in numeric_cols_only
                    ])

                # Convert back to pandas for downstream vectorbt/sklearn compatibility
                result_df = pl_df.to_pandas()
                nan_handling_succeeded = True
                tprint_success("✅ NaN handling optimized via Polars")
            except Exception as e:
                tprint_warning(f"⚠️ Polars-optimized NaN handling failed, falling back to standard method: {e}")

        if not nan_handling_succeeded:
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

    def _apply_feature_caps(
        self,
        ranked_features: List[str],
        max_interaction_features: int = 10,
        max_cross_timeframe_per_base: int = 10,
        max_variant_features: int = 10,
        min_interaction_features: int = 4,
        min_cross_timeframe_features: int = 4,
        min_variant_features: int = 4,
        max_cross_timeframe_total: Optional[int] = None,
    ) -> List[str]:
        interaction_count = 0
        variant_count = 0
        cross_timeframe_counts: Dict[str, int] = {}
        capped_features: List[str] = []
        cross_timeframe_total = 0

        for feature in ranked_features:
            name = str(feature)
            name_lower = name.lower()

            is_interaction = ("interaction" in name_lower) or ("_x_" in name_lower)
            is_variant = (
                name_lower.endswith("_volnorm")
                or name_lower.endswith("_vwap")
                or name_lower.endswith("_trend_adj")
            )
            is_cross_timeframe = (
                "ctf_" in name_lower
                or "cross_timeframe" in name_lower
                or re.search(r"\d+[mhd]", name_lower) is not None
            )

            if is_interaction and interaction_count >= max_interaction_features:
                continue

            cross_base: Optional[str] = None
            if is_cross_timeframe:
                parts = name.split("_")
                filtered_parts = []
                for part in parts:
                    if re.fullmatch(r"\d+[mhd]", part.lower()):
                        continue
                    filtered_parts.append(part)
                cross_base = "_".join(filtered_parts) if filtered_parts else name
                if max_cross_timeframe_per_base is not None and max_cross_timeframe_per_base < 0:
                    max_cross_timeframe_per_base = 0
                # Enforce global cap across all bases (if provided)
                if max_cross_timeframe_total is not None and cross_timeframe_total >= max_cross_timeframe_total:
                    continue
                # Enforce per-base cap
                if cross_timeframe_counts.get(cross_base, 0) >= max_cross_timeframe_per_base:
                    continue

            if is_variant and variant_count >= max_variant_features:
                continue

            capped_features.append(name)

            if is_interaction:
                interaction_count += 1
            if is_cross_timeframe and cross_base is not None:
                cross_timeframe_counts[cross_base] = cross_timeframe_counts.get(cross_base, 0) + 1
                cross_timeframe_total += 1
            if is_variant:
                variant_count += 1

        if interaction_count < min_interaction_features:
            tprint_warning(
                f"⚠️ Soft minimum for interaction features not met: "
                f"{interaction_count} < {min_interaction_features}"
            )
        if cross_timeframe_total < min_cross_timeframe_features:
            tprint_warning(
                f"⚠️ Soft minimum for cross-timeframe features not met: "
                f"{cross_timeframe_total} < {min_cross_timeframe_features}"
            )
        if variant_count < min_variant_features:
            tprint_warning(
                f"⚠️ Soft minimum for variant features not met: "
                f"{variant_count} < {min_variant_features}"
            )

        return capped_features

    def _perform_multi_size_selection(
        self,
        features_df,
        targets,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform feature selection for multiple feature set sizes with CMI-aware Tactician mode support."""
        # Define feature set sizes
        feature_set_sizes = config.get('feature_set_sizes', [60, 50, 40, 30])

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

        # Prefer meta-label outputs from feature_generation_meta_labeling_step.
        # We no longer fall back to fused/simplified price-based targets here;
        # if meta-label outputs are missing or empty, this step should fail
        # explicitly so the meta-labeling pipeline can be fixed.
        target_cols: List[str] = []
        meta_preferred_targets = ['binary_label', 'smoothed_label', 'realized_return']
        for col in meta_preferred_targets:
            if col in features_df.columns:
                non_null = features_df[col].notna().sum()
                tprint_info(f"📊 Meta-label column '{col}' non-NaN count: {non_null}")
                if non_null > 0:
                    target_cols = [col]
                    tprint_info(f"📊 Using meta-label target: {col} (non-NaN={non_null})")
                    break

        if not target_cols:
            msg = (
                "No usable meta-label target found for final feature selection. "
                "Expected at least one of ['binary_label', 'smoothed_label', 'realized_return'] "
                "with non-NaN values. Ensure feature_generation_meta_labeling_step has populated these."
            )
            tprint_error(f"❌ {msg}")
            raise ValueError(msg)

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
        
        # PRIORITY 2: Add coverage-aware feature filtering before selection
        # Remove features with less than 95% coverage (more than 5% NaN)
        tprint_info("=" * 80)
        tprint_info("🔍 PRIORITY 2: Coverage-Aware Feature Filtering")
        tprint_info("=" * 80)

        MIN_COVERAGE_PCT = 95.0  # Require at least 95% coverage (max 5% NaN)
        MAX_NAN_PCT = 100.0 - MIN_COVERAGE_PCT

        tprint_info(f"📊 Analyzing coverage for {len(X.columns)} candidate features...")
        tprint_info(f"📋 Coverage threshold: {MIN_COVERAGE_PCT}% (max {MAX_NAN_PCT}% NaN allowed)")

        features_to_keep = []
        features_to_remove = []
        coverage_stats = []

        for col in X.columns:
            nan_count = X[col].isna().sum()
            nan_pct = 100 * nan_count / len(X)
            coverage_pct = 100 - nan_pct

            coverage_stats.append({
                'feature': col,
                'coverage_pct': coverage_pct,
                'nan_pct': nan_pct,
                'nan_count': nan_count
            })

            if nan_pct <= MAX_NAN_PCT:
                features_to_keep.append(col)
                if nan_pct > 0:
                    tprint_info(f"  ✅ KEEP: '{col}' - coverage: {coverage_pct:.1f}% ({nan_count} NaN)")
            else:
                features_to_remove.append(col)
                tprint_warning(f"  ❌ REMOVE: '{col}' - coverage: {coverage_pct:.1f}% (BELOW {MIN_COVERAGE_PCT}% threshold)")

        # Sort coverage stats by coverage percentage (worst first)
        coverage_stats.sort(key=lambda x: x['coverage_pct'])

        tprint_info("=" * 80)
        tprint_info("📊 Coverage Filtering Results:")
        tprint_info(f"  Total features analyzed: {len(X.columns)}")
        tprint_info(f"  Features kept (≥{MIN_COVERAGE_PCT}% coverage): {len(features_to_keep)}")
        tprint_info(f"  Features removed (<{MIN_COVERAGE_PCT}% coverage): {len(features_to_remove)}")

        if features_to_remove:
            tprint_warning(f"⚠️ Removed {len(features_to_remove)} sparse features:")
            for feat in features_to_remove[:10]:  # Show first 10
                stat = next(s for s in coverage_stats if s['feature'] == feat)
                tprint_warning(f"    - '{feat}': {stat['coverage_pct']:.1f}% coverage")
            if len(features_to_remove) > 10:
                tprint_warning(f"    ... and {len(features_to_remove) - 10} more")

        # Show features with worst coverage that were kept
        tprint_info("📊 Features with lowest coverage (but still kept):")
        kept_stats = [s for s in coverage_stats if s['feature'] in features_to_keep]
        for stat in kept_stats[:5]:  # Show 5 worst
            tprint_info(f"  - '{stat['feature']}': {stat['coverage_pct']:.1f}% coverage")

        tprint_info("=" * 80)

        # Filter X to only keep dense features
        if features_to_remove:
            X = X[features_to_keep]
            feature_cols = features_to_keep
            tprint_success(f"✅ Filtered features: {len(X.columns)} features remain after coverage filtering")
        else:
            tprint_success(f"✅ All features meet coverage threshold (≥{MIN_COVERAGE_PCT}%)")

        # Now handle remaining NaN values in the kept features (should be minimal)
        feature_nan_count = X.isna().sum().sum()
        if feature_nan_count > 0:
            tprint_info(f"🔧 Imputing {feature_nan_count} remaining NaN values in kept features")

            # Intelligent imputation based on feature type
            for col in X.columns:
                nan_count = X[col].isna().sum()
                if nan_count > 0:
                    # For candlestick patterns and binary indicators: absence = 0
                    if any(keyword in col.lower() for keyword in ['candlestick', 'pattern', 'signal', 'flag']):
                        X[col] = X[col].fillna(0)
                        tprint_info(f"  📍 '{col}': {nan_count} NaNs → 0 (pattern not present)")
                    # For ratio/cross-timeframe features: forward fill then median
                    elif any(keyword in col.lower() for keyword in ['ratio', '_x_', 'cross']):
                        median_val = X[col].median()
                        X[col] = X[col].fillna(method='ffill').fillna(median_val)
                        tprint_info(f"  📍 '{col}': {nan_count} NaNs → forward fill + median")
                    # For continuous features: forward fill then median
                    else:
                        median_val = X[col].median()
                        X[col] = X[col].fillna(method='ffill').fillna(median_val)
                        tprint_info(f"  📍 '{col}': {nan_count} NaNs → forward fill + median")

            tprint_success(f"✅ Imputed all remaining NaN values")
        else:
            tprint_success(f"✅ No NaN values in kept features")

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
            stability_weight=config.get('stability_weight', 0.2)  # Default 0.3 = balanced (30% stability, 70% importance)
        )

        # Perform selection ONCE for the maximum size
        tprint_info(f"🎯 Selecting top {max_size} features using permutation importance (captures interactions)...")
        temp_component = FinalFeatureSelectionComponent(max_size_config)
        all_selected_features = temp_component.select_features(X, y, feature_cols)

        # CRITICAL DEBUG: Check what was selected
        tprint_error(f"🔍 CRITICAL DEBUG for max size {max_size}:")
        tprint_error(f"   Selected features count before caps: {len(all_selected_features)}")
        tprint_error(f"   Selected features sample before caps: {all_selected_features[:5] if len(all_selected_features) > 0 else 'EMPTY'}")
        
        if not all_selected_features:
            tprint_error(f"❌ CRITICAL: No features selected!")
            tprint_error(f"   Input X shape: {X.shape}")
            tprint_error(f"   Input y shape: {y.shape}")
            tprint_error(f"   Input feature_cols count: {len(feature_cols)}")
            return feature_sets

        # Apply caps to control the mix of interaction, cross-timeframe, and
        # variant features in the final sets. We want:
        #   - base features and their VWAP/vol-normal variants to be well
        #     represented,
        #   - interaction and cross-timeframe features to be allowed but
        #     bounded so they don't dominate the pool.

        max_interaction_features = int(config.get('max_interaction_features', 10))
        max_cross_timeframe_per_base = int(config.get('max_cross_timeframe_per_base', 10))
        # Global cap on cross-timeframe features across all bases (user
        # requirement: up to ~20 cross-timeframe features in total).
        max_cross_timeframe_total = int(config.get('max_cross_timeframe_total', 20))
        # Variants (VWAP/vol-normal/trend-adjusted) should be represented but
        # not explode combinatorially; cap them at a moderate number.
        max_variant_features = int(config.get('max_variant_features', 40))

        capped_features = self._apply_feature_caps(
            all_selected_features,
            max_interaction_features=max_interaction_features,
            max_cross_timeframe_per_base=max_cross_timeframe_per_base,
            max_variant_features=max_variant_features,
            max_cross_timeframe_total=max_cross_timeframe_total,
            # Soft minimums can be overridden via config but default to
            # encouraging variants while not forcing interactions/CTF.
            min_interaction_features=int(config.get('min_interaction_features', 0)),
            min_cross_timeframe_features=int(config.get('min_cross_timeframe_features', 0)),
            min_variant_features=int(config.get('min_variant_features', 4)),
        )

        if not capped_features:
            tprint_error("❌ CRITICAL: No features remain after applying interaction/cross-timeframe/variant caps")
            return feature_sets

        tprint_error(f"   Selected features count after caps: {len(capped_features)}")
        tprint_error(f"   Selected features sample after caps: {capped_features[:5] if len(capped_features) > 0 else 'EMPTY'}")
        
        # Now create feature sets by slicing the ranked list (no redundant computation!)
        for size in sorted(feature_set_sizes, reverse=True):  # Process from largest to smallest
            tprint_info(f"📊 Creating feature set for size {size} (slicing from capped list of length {len(capped_features)})...")
            
            # Slice the top N features from the already-ranked list after applying caps
            selected_features = capped_features[:size]
            
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

            # Event-aware feature scoring (reward, penalty, combined score)
            try:
                X_sel = X[selected_features]
                event_scores = temp_component._event_aware_feature_scores(X_sel, y)
                event_rewards = getattr(temp_component, 'event_reward_scores', {}) or {}
                event_penalties = getattr(temp_component, 'event_penalty_scores', {}) or {}
                analysis_results['event_aware_metrics'] = {
                    'scores': event_scores.to_dict(),
                    'reward': event_rewards,
                    'penalty': event_penalties,
                }
            except Exception as e:
                analysis_results['event_aware_metrics'] = {'error': str(e)}
            
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

            # 6. NEW: Selection Frequency Distribution Analysis
            tprint_info("📊 Analyzing selection frequency distribution...")
            freq_dist_analysis = temp_component.analyze_selection_frequency_distribution()
            analysis_results['frequency_distribution'] = freq_dist_analysis

            # 7. NEW: Null Importance Analysis (statistical significance)
            tprint_info("🎲 Calculating null importance baseline...")
            null_importance = temp_component.calculate_null_importance_baseline(X, y, selected_features, n_permutations=50)
            analysis_results['null_importance'] = null_importance

            # 8. NEW: Walk-Forward Validation
            tprint_info("🚶 Performing walk-forward validation...")
            walk_forward = temp_component.walk_forward_feature_validation(X, y, selected_features, n_splits=5)
            analysis_results['walk_forward_validation'] = walk_forward

            # 9. NEW: Feature Redundancy Clustering
            tprint_info("🔗 Clustering redundant features...")
            redundancy_clustering = temp_component.cluster_redundant_features(X, selected_features, corr_threshold=0.85)
            analysis_results['redundancy_clustering'] = redundancy_clustering

            # 10. NEW: Mutual Information Stability (vectorized proxy)
            tprint_info("📊 Calculating MI stability...")
            mi_stability = temp_component.calculate_mi_stability(X, y, selected_features, cv_folds=5)
            analysis_results['mi_stability'] = mi_stability

            # 11. PHASE 3: Data Leakage Detection (CRITICAL)
            tprint_info("🔍 Detecting potential data leakage...")
            leakage_detection = temp_component.detect_potential_leakage(X, y, selected_features)
            analysis_results['leakage_detection'] = leakage_detection

            # 12. PHASE 3: Feature Information Content
            tprint_info("📊 Checking feature information content...")
            information_content = temp_component.check_feature_information_content(X, selected_features)
            analysis_results['information_content'] = information_content

            # 13. Method Results Analysis (if available)
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

    def _compute_meta_label_feature_diagnostics(
        self,
        full_df: pd.DataFrame,
        selected_features: List[str],
    ) -> Dict[str, Any]:
        """Compute IC/AUC diagnostics vs binary_label and realized_return.

        Uses the full labeled dataset (including regime and TTO diagnostics) to
        evaluate selected features. Results are organized into:
        - overall: metrics on full sample
        - by_volatility_regime: per volatility_regime bucket, if available
        - by_tto_bucket: per event_tto_mean_last_50 tercile, if available
        """
        results: Dict[str, Any] = {
            'overall': {},
            'by_volatility_regime': {},
            'by_tto_bucket': {},
        }

        has_binary = 'binary_label' in full_df.columns
        has_rr = 'realized_return' in full_df.columns

        if not (has_binary or has_rr):
            return {'error': 'binary_label and realized_return not found in dataset'}

        bin_series = full_df['binary_label'] if has_binary else None
        rr_series = full_df['realized_return'] if has_rr else None

        def _safe_corr(x: pd.Series, y: Optional[pd.Series]) -> Optional[float]:
            if x is None or y is None:
                return None
            valid = x.notna() & y.notna()
            if valid.sum() < 100:
                return None
            try:
                val = float(x[valid].corr(y[valid]))
                return None if not np.isfinite(val) else val
            except Exception:
                return None

        def _compute_for_mask(mask: pd.Series) -> Dict[str, Any]:
            slice_result: Dict[str, Any] = {}
            if mask is None or mask.sum() < 100:
                return slice_result

            for feat in selected_features:
                if feat not in full_df.columns:
                    continue

                s_feat = pd.to_numeric(full_df.loc[mask, feat], errors='coerce')
                feat_metrics: Dict[str, Any] = {}

                if has_binary:
                    yb = bin_series.loc[mask]
                    valid = s_feat.notna() & yb.notna()
                    auc_val: Optional[float] = None
                    if valid.sum() >= 100 and yb[valid].nunique() == 2:
                        try:
                            auc_val = float(roc_auc_score(yb[valid], s_feat[valid]))
                        except Exception:
                            auc_val = None
                    ic_clf = _safe_corr(s_feat, yb)
                    feat_metrics['binary_label'] = {
                        'ic': ic_clf,
                        'auc': auc_val,
                        'n': int(valid.sum()),
                    }

                if has_rr:
                    yr = rr_series.loc[mask]
                    ic_ret = _safe_corr(s_feat, yr)
                    n_ret = int((s_feat.notna() & yr.notna()).sum())
                    feat_metrics['realized_return'] = {
                        'ic': ic_ret,
                        'n': n_ret,
                    }

                if feat_metrics:
                    slice_result[feat] = feat_metrics

            return slice_result

        # Overall diagnostics
        base_mask = full_df.index.notnull()
        results['overall'] = _compute_for_mask(base_mask)

        # Volatility regime slices (if available)
        if 'volatility_regime' in full_df.columns:
            try:
                regimes = pd.unique(full_df['volatility_regime'].dropna())
                for reg_val in regimes:
                    reg_mask = base_mask & (full_df['volatility_regime'] == reg_val)
                    slice_res = _compute_for_mask(reg_mask)
                    if slice_res:
                        results['by_volatility_regime'][str(reg_val)] = slice_res
            except Exception as e:
                results['by_volatility_regime'] = {'error': str(e)}

        # TTO slices using terciles of event_tto_mean_last_50 (if available)
        if 'event_tto_mean_last_50' in full_df.columns:
            try:
                tto = pd.to_numeric(full_df['event_tto_mean_last_50'], errors='coerce')
                tto_non_null = tto.dropna()
                if len(tto_non_null) >= 150:
                    q1, q2 = tto_non_null.quantile([0.33, 0.66])
                    buckets = {
                        'short': (tto <= q1),
                        'medium': (tto > q1) & (tto < q2),
                        'long': (tto >= q2),
                    }
                    for bucket_name, bucket_mask in buckets.items():
                        mask = base_mask & bucket_mask & tto.notna()
                        slice_res = _compute_for_mask(mask)
                        if slice_res:
                            results['by_tto_bucket'][bucket_name] = slice_res
            except Exception as e:
                results['by_tto_bucket'] = {'error': str(e)}

        return results

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
        if 'target_long_fused' in features_df.columns and 'target_short_fused' in features_df.columns:
            target_cols = ['target_long_fused', 'target_short_fused']
            tprint_info("📊 Using fused target structure: target_long_fused, target_short_fused")
        elif 'target_long' in features_df.columns and 'target_short' in features_df.columns:
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
            if 'target_long_fused' in features_df.columns and 'target_short_fused' in features_df.columns:
                target_cols = ['target_long_fused', 'target_short_fused']
            elif 'target_long' in features_df.columns and 'target_short' in features_df.columns:
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
            # Pass sample weights if present alongside features
            sw = None
            if 'target_sample_weight' in features_df.columns:
                try:
                    sw = features_df.loc[X_train.index, 'target_sample_weight'].values
                except Exception:
                    sw = None
            rf_model.fit(X_train, y_train, sample_weight=sw)

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
            'feature_set_sizes': config.get('feature_set_sizes', [60, 50, 40, 30]),
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

        # Generate CSV quality report in outcomes/ with datetime
        try:
            self._generate_feature_quality_csv_report(combined_features_df, feature_sets, config)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate feature quality CSV report: {e}")

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

    def _generate_feature_quality_csv_report(self, combined_features_df: pd.DataFrame, feature_sets: Dict[str, List[str]], config: Dict[str, Any]) -> None:
        """Generate a CSV with predictive power, stability, robustness, and collinearity metrics per feature.
        Saved to outcomes/feature_quality_<timestamp>.csv.
        """
        import os
        import numpy as np
        import pandas as pd
        from datetime import datetime
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, r2_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.inspection import permutation_importance

        # Choose target: only use meta-label outputs. If these are not
        # available we skip the CSV report rather than falling back to fused
        # or legacy price-based targets.
        meta_target_pref = ['binary_label', 'smoothed_label', 'realized_return']
        target_cols = [c for c in meta_target_pref if c in combined_features_df.columns]
        if not target_cols:
            tprint_warning(
                "⚠️ No meta-label target column (binary_label/smoothed_label/realized_return) "
                "found for quality CSV report; skipping"
            )
            return
        target_col = target_cols[0]

        # Get selected features for 60-set
        sel_key = 'selected_feature_dataframe_60'
        selected_cols = []
        if sel_key in feature_sets:
            # Some code paths store DataFrame under this key
            df60 = feature_sets[sel_key]
            if isinstance(df60, pd.DataFrame):
                selected_cols = [c for c in df60.columns if c != target_col and c in combined_features_df.columns]
        if not selected_cols:
            # Try list key
            list_key = 'selected_features_60'
            if list_key in feature_sets:
                selected_cols = [c for c in feature_sets[list_key] if c in combined_features_df.columns]
        if not selected_cols:
            # Fallback: use all non-target numeric columns (cap to 60)
            candidate = combined_features_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = [c for c in candidate if c != target_col][:60]

        if not selected_cols:
            tprint_warning("⚠️ No selected features found for quality CSV report; skipping")
            return

        # PRIORITY 1 FIX: Handle sparse features properly instead of dropping all rows
        # Original code: df = combined_features_df[selected_cols + [target_col]].dropna()
        # This caused 99% data loss when sparse features (like candlestick patterns) were selected

        tprint_info("=" * 80)
        tprint_info("🔍 PRIORITY 1: Analyzing feature sparsity before filtering")
        tprint_info("=" * 80)

        df = combined_features_df[selected_cols + [target_col]].copy()
        initial_rows = len(df)
        tprint_info(f"📊 Initial dataset: {initial_rows} rows")

        # Analyze NaN patterns for each feature
        tprint_info(f"📊 Analyzing NaN patterns for {len(selected_cols)} selected features:")
        sparse_features = []
        dense_features = []

        for col in selected_cols:
            nan_count = df[col].isna().sum()
            nan_pct = 100 * nan_count / len(df)
            coverage = 100 - nan_pct

            if nan_pct > 50:  # More than 50% NaN
                sparse_features.append(col)
                tprint_warning(f"  ⚠️ SPARSE: '{col}' - {nan_pct:.1f}% NaN (coverage: {coverage:.1f}%)")
            elif nan_pct > 10:  # 10-50% NaN
                tprint_info(f"  📊 MODERATE: '{col}' - {nan_pct:.1f}% NaN (coverage: {coverage:.1f}%)")
            else:
                dense_features.append(col)
                tprint_success(f"  ✅ DENSE: '{col}' - {nan_pct:.1f}% NaN (coverage: {coverage:.1f}%)")

        tprint_info(f"📊 Sparsity summary:")
        tprint_info(f"  - Dense features (>90% coverage): {len(dense_features)}")
        tprint_info(f"  - Moderate features (50-90% coverage): {len(selected_cols) - len(sparse_features) - len(dense_features)}")
        tprint_info(f"  - Sparse features (<50% coverage): {len(sparse_features)}")

        # Option A: Only require valid target (most permissive)
        tprint_info("🔧 Applying Option A: Require only valid target")
        df = df[df[target_col].notna()]
        rows_after_target_filter = len(df)
        tprint_info(f"  After target filter: {rows_after_target_filter} rows ({100*rows_after_target_filter/initial_rows:.1f}% retained)")

        # Option C: Impute sparse features intelligently
        tprint_info("🔧 Applying Option C: Intelligent imputation for sparse features")

        for col in selected_cols:
            nan_count_before = df[col].isna().sum()
            if nan_count_before > 0:
                # For candlestick patterns and binary indicators: absence = 0
                if any(keyword in col.lower() for keyword in ['candlestick', 'pattern', 'signal', 'flag']):
                    df[col] = df[col].fillna(0)
                    tprint_info(f"  📍 Imputed '{col}': {nan_count_before} NaNs → 0 (pattern not present)")
                # For ratio/cross-timeframe features: forward fill then 0
                elif any(keyword in col.lower() for keyword in ['ratio', '_x_', 'cross']):
                    df[col] = df[col].fillna(method='ffill').fillna(0)
                    tprint_info(f"  📍 Imputed '{col}': {nan_count_before} NaNs → forward fill + 0")
                # For continuous features: forward fill then median
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(method='ffill').fillna(median_val)
                    tprint_info(f"  📍 Imputed '{col}': {nan_count_before} NaNs → forward fill + median ({median_val:.4f})")

        # Final check: verify no NaNs remain
        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            tprint_warning(f"⚠️ {remaining_nans} NaN values remain after imputation, filling with 0")
            df = df.fillna(0)
        else:
            tprint_success("✅ All NaN values successfully imputed")

        final_rows = len(df)
        retention_rate = 100 * final_rows / initial_rows

        tprint_info("=" * 80)
        tprint_info(f"📊 PRIORITY 1 RESULTS:")
        tprint_info(f"  Initial rows: {initial_rows}")
        tprint_info(f"  Final rows: {final_rows}")
        tprint_info(f"  Retention rate: {retention_rate:.1f}%")
        tprint_info(f"  Rows saved from deletion: {final_rows - (initial_rows - rows_after_target_filter)}")

        if retention_rate < 50:
            tprint_error(f"❌ CRITICAL: Low retention rate ({retention_rate:.1f}%)! Investigate data issues.")
        elif retention_rate < 90:
            tprint_warning(f"⚠️ Moderate retention rate ({retention_rate:.1f}%). Check target quality.")
        else:
            tprint_success(f"✅ Excellent retention rate ({retention_rate:.1f}%)!")
        tprint_info("=" * 80)

        if len(df) < 200:
            tprint_warning(f"⚠️ Too few samples for quality report ({len(df)} < 200); skipping")
            return

        X = df[selected_cols].astype(float)
        y = df[target_col]
        is_classification = set(np.unique(y)).issubset({0, 1}) and y.nunique() <= 2

        # Baseline predictions
        if is_classification:
            baseline_pred = np.full(len(y), y.mean())
        else:
            baseline_pred = np.full(len(y), y.mean())

        # Mutual information per feature so we can compute relative MI vs the
        # strongest base feature. This lets us see whether interaction,
        # cross-timeframe, or variant features add MI beyond the best base.
        mi_scores = {}
        try:
            if is_classification:
                mi_raw = mutual_info_classif(X.values, y.values, random_state=42)
            else:
                mi_raw = mutual_info_regression(X.values, y.values, random_state=42)
            mi_scores = {feat: float(mi_raw[i]) for i, feat in enumerate(selected_cols)}
        except Exception as e:
            tprint_warning(f"⚠️ MI calculation for feature quality CSV failed: {e}")
            mi_scores = {}

        # Model
        if is_classification:
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            score_fn = lambda yt, pr: roc_auc_score(yt, pr)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            # Information Coefficient proxy: Spearman is costly; use Pearson IC
            score_fn = lambda yt, pr: np.corrcoef(yt, pr)[0,1] if np.std(pr) > 0 else 0.0

        # TimeSeries CV uplift
        tscv = TimeSeriesSplit(n_splits=5)
        scores, base_scores = [], []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(Xtr, ytr)
            pr = model.predict_proba(Xte)[:,1] if is_classification else model.predict(Xte)
            scores.append(score_fn(yte, pr))
            base_scores.append(score_fn(yte, baseline_pred[test_idx]))
        cv_score = float(np.nanmean(scores))
        cv_base = float(np.nanmean(base_scores))
        cv_uplift = float(cv_score - cv_base)

        # Permutation importance on full data
        pi = permutation_importance(model.fit(X, y), X, y, n_repeats=5, random_state=42, n_jobs=-1)
        pi_scores = dict(zip(selected_cols, pi.importances_mean))
        topk = set(sorted(pi_scores, key=pi_scores.get, reverse=True)[:20])

        # SHAP placeholder (skipped earlier); fill NaN
        shap_importance = {c: np.nan for c in selected_cols}

        # Rolling IC monthly/quarterly based on model predictions on full data
        preds_full = model.predict_proba(X)[:,1] if is_classification else model.predict(X)
        df_preds = pd.DataFrame({'pred': preds_full, 'target': y.values}, index=df.index)
        monthly_ic = df_preds.resample('M').apply(lambda s: s['pred'].corr(s['target'])).dropna()
        quarterly_ic = df_preds.resample('Q').apply(lambda s: s['pred'].corr(s['target'])).dropna()
        rolling_ic_mean = float(np.nanmean(monthly_ic.values)) if len(monthly_ic) else np.nan
        rolling_ic_std = float(np.nanstd(monthly_ic.values)) if len(monthly_ic) else np.nan

        # PSI across time splits: first half vs last half
        mid = len(df) // 2
        psi_values = {}
        n_bins = 10
        for c in selected_cols:
            a = df[c].values[:mid]
            b = df[c].values[mid:]
            try:
                # Bin by combined quantiles
                qs = np.quantile(np.concatenate([a, b]), np.linspace(0,1,n_bins+1))
                qs[0], qs[-1] = -np.inf, np.inf
                def dist(v):
                    hist, _ = np.histogram(v, bins=qs)
                    p = hist / max(len(v),1)
                    p = np.where(p==0, 1e-6, p)
                    return p
                p, q = dist(a), dist(b)
                psi = np.sum((p - q) * np.log(p / q))
            except Exception:
                psi = np.nan
            psi_values[c] = float(psi)

        # Robustness: Outlier clipping sensitivity (p99, p95)
        def clip_and_score(p):
            Xc = X.copy()
            lo = Xc.quantile(1-p)
            hi = Xc.quantile(p)
            Xc = Xc.clip(lower=lo, upper=hi, axis=1)
            m = model.__class__(**model.get_params())
            m.fit(Xc, y)
            pr = m.predict_proba(Xc)[:,1] if is_classification else m.predict(Xc)
            return float(score_fn(y, pr))
        try:
            full_score = float(score_fn(y, preds_full))
        except Exception:
            full_score = np.nan
        try:
            clip_p99 = clip_and_score(0.99)
            clip_p95 = clip_and_score(0.95)
        except Exception:
            clip_p99 = np.nan
            clip_p95 = np.nan
        delta_p99 = float(clip_p99 - full_score) if not np.isnan(clip_p99) and not np.isnan(full_score) else np.nan
        delta_p95 = float(clip_p95 - full_score) if not np.isnan(clip_p95) and not np.isnan(full_score) else np.nan

        # Temporal robustness: train early, test late
        split = int(len(X)*0.6)
        try:
            model.fit(X.iloc[:split], y.iloc[:split])
            pr_early = model.predict_proba(X.iloc[split:])[:,1] if is_classification else model.predict(X.iloc[split:])
            temporal_score = float(score_fn(y.iloc[split:], pr_early))
            early_score = float(score_fn(y.iloc[:split], model.predict_proba(X.iloc[:split])[:,1] if is_classification else model.predict(X.iloc[:split])))
            temporal_degradation = float(early_score - temporal_score)
        except Exception:
            temporal_degradation = np.nan

        # Collinearity clusters (|rho|>0.9) and stable representative (highest pi)
        corr = X.corr().abs()
        visited = set()
        cluster_id = {}
        representative = {}
        max_abs_corr = {c: float(np.nanmax(corr[c].drop(c))) if c in corr.columns else np.nan for c in selected_cols}
        cid = 0
        for c in selected_cols:
            if c in visited:
                continue
            cluster = set([c])
            # Expand cluster by threshold
            added = True
            while added:
                added = False
                for d in selected_cols:
                    if d in cluster:
                        continue
                    if any(corr.loc[d, m] > 0.9 for m in cluster if d in corr.index and m in corr.columns):
                        cluster.add(d)
                        added = True
            for m in cluster:
                cluster_id[m] = cid
            # Representative: highest permutation importance
            rep = max(list(cluster), key=lambda z: pi_scores.get(z, 0.0))
            representative[cid] = rep
            visited |= cluster
            cid += 1

        # Helper to classify feature types for downstream slicing
        def _classify_feature_for_csv(name: str):
            nl = str(name).lower()
            is_interaction = ("interaction" in nl) or ("_x_" in nl)
            is_variant = (
                nl.endswith("_volnorm")
                or nl.endswith("_vwap")
                or nl.endswith("_trend_adj")
            )
            is_cross_timeframe = (
                "ctf_" in nl
                or "cross_timeframe" in nl
                or re.search(r"\d+[mhd]", nl) is not None
            )
            return is_interaction, is_variant, is_cross_timeframe

        # Identify base features (neither interaction, variant, nor cross-timeframe)
        base_features = []
        for f in selected_cols:
            is_interaction, is_variant, is_cross = _classify_feature_for_csv(f)
            if not (is_interaction or is_variant or is_cross):
                base_features.append(f)

        # Best base MI (for relative MI computation)
        best_base_mi = None
        if mi_scores:
            base_mi_vals = [mi_scores.get(f) for f in base_features if f in mi_scores]
            base_mi_vals = [v for v in base_mi_vals if v is not None and not np.isnan(v)]
            if base_mi_vals:
                best_base_mi = float(max(base_mi_vals))

        # Stability scores (if available) to measure stability delta vs base
        stability_scores = {}
        base_stability_mean = None
        enhanced_analysis_local = feature_sets.get('enhanced_analysis') if isinstance(feature_sets, dict) else None
        if enhanced_analysis_local and isinstance(enhanced_analysis_local, dict):
            stab = enhanced_analysis_local.get('stability_analysis')
            if stab and isinstance(stab, dict):
                stability_scores = (
                    (stab.get('stability_results', {}) or {}).get('stability_scores', {}) or {}
                )
        if stability_scores:
            base_stab_vals = [
                float(stability_scores[f])
                for f in base_features
                if f in stability_scores and stability_scores[f] is not None
            ]
            base_stab_vals = [v for v in base_stab_vals if not np.isnan(v)]
            if base_stab_vals:
                base_stability_mean = float(np.mean(base_stab_vals))

        # Build CSV rows
        rows = []
        for f in selected_cols:
            is_interaction, is_variant, is_cross = _classify_feature_for_csv(f)

            mi_val = mi_scores.get(f, np.nan) if mi_scores else np.nan
            if best_base_mi is not None and best_base_mi > 0 and not np.isnan(mi_val):
                mi_lift_pct = float(100.0 * (mi_val - best_base_mi) / best_base_mi)
            else:
                mi_lift_pct = np.nan

            stab_val = stability_scores.get(f, np.nan) if stability_scores else np.nan
            if base_stability_mean is not None and not np.isnan(stab_val):
                stab_delta = float(stab_val - base_stability_mean)
            else:
                stab_delta = np.nan

            row = {
                'feature': f,
                'perm_importance': float(pi_scores.get(f, np.nan)),
                'shap_importance': float(shap_importance.get(f, np.nan)),
                'in_top20_perm': f in topk,
                'rolling_ic_mean_monthly': rolling_ic_mean,
                'rolling_ic_std_monthly': rolling_ic_std,
                'psi_first_last': psi_values.get(f, np.nan),
                'robust_clip_p99_delta': delta_p99,
                'robust_clip_p95_delta': delta_p95,
                'temporal_degradation': temporal_degradation,
                'cluster_id': cluster_id.get(f, -1),
                'is_cluster_rep': representative.get(cluster_id.get(f, -1), None) == f,
                'max_abs_corr': max_abs_corr.get(f, np.nan),
                'cv_score': cv_score,
                'cv_baseline': cv_base,
                'cv_uplift': cv_uplift,
                # Feature type flags for easy slicing of interaction/CTF/variants
                'is_interaction': is_interaction,
                'is_cross_timeframe': is_cross,
                'is_variant': is_variant,
                # MI and stability diagnostics relative to base features
                'mi_score': float(mi_val) if not np.isnan(mi_val) else np.nan,
                'mi_lift_vs_best_base_pct': mi_lift_pct,
                'stability_score': float(stab_val) if not np.isnan(stab_val) else np.nan,
                'stability_delta_vs_base_mean': stab_delta,
            }
            rows.append(row)

        # Augment rows with enhanced analysis metrics when available
        enhanced_analysis = feature_sets.get('enhanced_analysis')
        if enhanced_analysis and isinstance(enhanced_analysis, dict):
            # Map feature name -> row dict for easy enrichment
            row_map = {row.get('feature'): row for row in rows if 'feature' in row}

            # 1) Distributional sanity checks (information content)
            info_content = enhanced_analysis.get('information_content')
            if info_content and isinstance(info_content, dict):
                feature_stats = info_content.get('feature_stats', {}) or {}
                low_var_list = info_content.get('low_variance_features', []) or []
                quasi_const_list = info_content.get('quasi_constant_features', []) or []

                low_var_set = {name for name, _ in low_var_list}
                quasi_const_set = {name for name, _ in quasi_const_list}

                for name, stats in feature_stats.items():
                    row = row_map.get(name)
                    if row is None or not isinstance(stats, dict):
                        continue
                    # Variance
                    try:
                        row['dist_variance'] = float(stats.get('variance', np.nan))
                    except Exception:
                        row['dist_variance'] = np.nan
                    # Max value proportion
                    try:
                        row['dist_max_value_proportion'] = float(stats.get('max_value_proportion', np.nan))
                    except Exception:
                        row['dist_max_value_proportion'] = np.nan
                    # Number of unique values
                    try:
                        n_unique_val = stats.get('n_unique', np.nan)
                        row['dist_n_unique'] = int(n_unique_val) if not pd.isna(n_unique_val) else np.nan
                    except Exception:
                        row['dist_n_unique'] = stats.get('n_unique', np.nan)

                    # Flags
                    row['dist_is_low_variance'] = name in low_var_set
                    row['dist_is_quasi_constant'] = name in quasi_const_set

            # 2) Predictive robustness / ablation (walk-forward feature contributions)
            wf_validation = enhanced_analysis.get('walk_forward_validation')
            if wf_validation and isinstance(wf_validation, dict):
                contributions = wf_validation.get('feature_contributions', {}) or {}
                for name, contrib in contributions.items():
                    row = row_map.get(name)
                    if row is None:
                        continue
                    try:
                        row['oos_marginal_r2'] = float(contrib)
                    except Exception:
                        row['oos_marginal_r2'] = np.nan

            # 3) Complementarity (redundancy clustering)
            redundancy_cluster = enhanced_analysis.get('redundancy_clustering')
            if redundancy_cluster and isinstance(redundancy_cluster, dict):
                feature_clusters = redundancy_cluster.get('feature_clusters', {}) or {}
                redundant_features = redundancy_cluster.get('redundant_features', {}) or {}
                representative_features = set(redundancy_cluster.get('representative_features', []) or [])

                # Build per-feature cluster size map
                cluster_sizes = {}
                for _, feats in feature_clusters.items():
                    try:
                        size = len(feats)
                    except Exception:
                        size = 0
                    for name in feats:
                        cluster_sizes[name] = size

                for name, row in row_map.items():
                    size = cluster_sizes.get(name)
                    row['complementarity_cluster_size'] = int(size) if isinstance(size, int) and size > 0 else 1
                    row['complementarity_is_representative'] = name in representative_features
                    row['complementarity_is_redundant'] = name in redundant_features

            # 4) Event-aware metrics (reward, penalty frequency, combined score)
            event_metrics = enhanced_analysis.get('event_aware_metrics')
            if event_metrics and isinstance(event_metrics, dict):
                scores_map = event_metrics.get('scores', {}) or {}
                rewards_map = event_metrics.get('reward', {}) or {}
                penalties_map = event_metrics.get('penalty', {}) or {}

                for name, row in row_map.items():
                    if name not in scores_map and name not in rewards_map and name not in penalties_map:
                        continue
                    try:
                        row['event_score'] = float(scores_map.get(name, np.nan))
                    except Exception:
                        row['event_score'] = np.nan
                    try:
                        row['event_reward'] = float(rewards_map.get(name, np.nan))
                    except Exception:
                        row['event_reward'] = np.nan
                    try:
                        row['event_penalty_freq'] = float(penalties_map.get(name, np.nan))
                    except Exception:
                        row['event_penalty_freq'] = np.nan

        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = outcomes_dir / f"final_selection_feature_quality_{config.get('symbol','UNKNOWN')}_{ts}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        tprint_success(f"✅ Feature quality CSV report saved: {csv_path}")

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

    def _run_baseline_predictive_check(
        self,
        feature_dataframe: pd.DataFrame,
        targets: pd.Series,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run baseline predictive check on selected features."""
        try:
            from src.training.steps.pre_training.baseline_predictive_check import BaselinePredictiveCheck
            from pathlib import Path

            # Identify target column
            target = None

            # First try to infer target from feature_dataframe with priority for fused targets
            direction = str(config.get('direction', 'long')).lower()
            candidate_cols: List[str] = []

            if 'target_long_fused' in feature_dataframe.columns and 'target_short_fused' in feature_dataframe.columns:
                if 'short' in direction:
                    candidate_cols.extend(['target_short_fused', 'target_long_fused'])
                else:
                    candidate_cols.extend(['target_long_fused', 'target_short_fused'])
            elif 'target_long' in feature_dataframe.columns and 'target_short' in feature_dataframe.columns:
                if 'short' in direction:
                    candidate_cols.extend(['target_short', 'target_long'])
                else:
                    candidate_cols.extend(['target_long', 'target_short'])

            # If no directional structure detected, fall back to generic target columns in features
            if not candidate_cols:
                for col in TARGET_COLUMN_NAMES:
                    if col in feature_dataframe.columns:
                        candidate_cols.append(col)

            for col in candidate_cols:
                if col in feature_dataframe.columns:
                    tprint_info(f"🎯 Baseline check using target column from features: {col}")
                    target = feature_dataframe[col]
                    break

            # Fallback: use targets argument if we still haven't found a target
            if target is None:
                if isinstance(targets, pd.Series):
                    target = targets
                elif isinstance(targets, pd.DataFrame):
                    target_col = None
                    for col in ['target_long_fused', 'target_short_fused',
                                'target_long', 'target_short',
                                'target', 'price_target_vol_normalized', 'label', 'return']:
                        if col in targets.columns:
                            target_col = col
                            break
                    if target_col:
                        tprint_info(f"🎯 Baseline check using target column from metadata: {target_col}")
                        target = targets[target_col]
                    else:
                        # Use last column
                        target = targets.iloc[:, -1]
                else:
                    tprint_warning("⚠️ Invalid target type, skipping baseline check")
                    return None

            # Get feature columns (exclude target columns)
            target_column_names = TARGET_COLUMN_NAMES
            feature_cols = [col for col in feature_dataframe.columns if col not in target_column_names]

            if not feature_cols:
                tprint_warning("⚠️ No feature columns found, skipping baseline check")
                return None

            X = feature_dataframe[feature_cols]

            # Run the check (no feature limit - use all selected features)
            tprint_info(f"🔍 Running baseline check on all {len(feature_cols)} selected features...")
            checker = BaselinePredictiveCheck(max_features=None, random_state=42)
            results = checker.run_check(X, target)

            # Save CSV to outcomes directory
            if results.get('success', False):
                outcomes_dir = Path('outcomes')
                outcomes_dir.mkdir(exist_ok=True)
                csv_path = checker.save_results_to_csv(
                    outcomes_dir,
                    filename_prefix="baseline_check_final_feature_selection"
                )
                if csv_path:
                    tprint_info(f"📊 Baseline check CSV saved: {csv_path}")
                    results['csv_path'] = csv_path

            return results

        except Exception as e:
            logger.error(f"Baseline predictive check failed: {e}", exc_info=True)
            return None

    def _create_outcome_report(self, feature_sets: Dict[str, List[str]], shap_values: Dict[str, Any], config: Dict[str, Any], baseline_check_results: Optional[Dict[str, Any]] = None) -> str:
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

            # Add baseline predictive check results if available
            if baseline_check_results and baseline_check_results.get('success', False):
                from src.training.steps.pre_training.baseline_predictive_check import BaselinePredictiveCheck

                # Create a temporary checker to format results
                temp_checker = BaselinePredictiveCheck()
                temp_checker.results = baseline_check_results

                # Add formatted markdown section
                report += "\n" + temp_checker.format_for_markdown()

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

## Top IC Features (Meta-Label Overview)

"""
            # High-level IC summary from meta-label diagnostics (if available)
            try:
                enhanced_analysis = feature_sets.get('enhanced_analysis', {})
                meta_diag = enhanced_analysis.get('meta_label_diagnostics') if isinstance(enhanced_analysis, dict) else None
                if isinstance(meta_diag, dict):
                    overall = meta_diag.get('overall', {}) or {}

                    def _collect_top_ic(target_key: str, top_n: int = 5) -> List[Tuple[str, float]]:
                        rows: List[Tuple[str, float]] = []
                        for feat_name, metrics in overall.items():
                            if not isinstance(metrics, dict):
                                continue
                            tmetrics = metrics.get(target_key) or {}
                            ic_val = tmetrics.get('ic')
                            if isinstance(ic_val, (int, float)) and np.isfinite(ic_val):
                                rows.append((feat_name, float(ic_val)))
                        rows_sorted = sorted(rows, key=lambda x: abs(x[1]), reverse=True)
                        return rows_sorted[:top_n]

                    top_bin = _collect_top_ic('binary_label')
                    top_rr = _collect_top_ic('realized_return')

                    if top_bin or top_rr:
                        report += "**Top 5 features by IC vs binary_label (overall):**\n\n"
                        if top_bin:
                            for rank, (feat, ic_val) in enumerate(top_bin, 1):
                                report += f"{rank}. {feat} (IC = {ic_val:.4f})\\n"
                        else:
                            report += "No valid IC scores for binary_label available.\\n"

                        report += "\n**Top 5 features by IC vs realized_return (overall):**\n\n"
                        if top_rr:
                            for rank, (feat, ic_val) in enumerate(top_rr, 1):
                                report += f"{rank}. {feat} (IC = {ic_val:.4f})\\n"
                        else:
                            report += "No valid IC scores for realized_return available.\\n"
            except Exception:
                # Do not fail report generation if diagnostics are missing
                pass

            report += "\n\n## Feature Selection Methodology\n\n"

            report += (
                "✅ **Using Permutation Importance**\n"
                "- Captures how features work together (feature interactions)\n"
                "- More reliable than standard Gini importance for complex trading strategies\n"
                "- Measures true impact on model predictions\n"
                "- Better for identifying genuinely predictive features\n\n"
                "## Feature Selection Results\n\n"
            )
            
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
                    report += f"- **Average Correlation:** {corr_analysis.get('average_correlation', 'N/A'):.4f}  — average pairwise |ρ| between features; lower is better and values <0.2 indicate low redundancy.\n"
                    report += f"- **Max Correlation:** {corr_analysis.get('max_correlation', 'N/A'):.4f}  — highest |ρ| observed; very high values may indicate near-duplicate signals.\n"
                    report += f"- **Min Correlation:** {corr_analysis.get('min_correlation', 'N/A'):.4f}  — lowest |ρ|; values near 0 show some features are nearly independent.\n"
                    report += f"- **High Correlation Pairs:** {len(corr_analysis.get('high_correlation_pairs', []))}  — number of feature pairs above the threshold; 0 is ideal.\n"
                    report += f"- **Correlation Threshold:** {corr_analysis.get('correlation_threshold', 'N/A')}  — pairs above this are considered redundant for clustering.\n\n"
                
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
                    report += f"- **Average Stability:** {stab_analysis.get('average_stability', 'N/A'):.4f}  — 0–1 score of importance consistency across time windows; higher is better and >0.5 is strong.\n"
                    report += f"- **Stable Features:** {len(stab_analysis.get('stable_features', []))}  — features above the stability threshold; more indicates a more robust set.\n"
                    report += f"- **Stability Threshold:** {stab_analysis.get('stability_threshold', 'N/A')}  — adaptive cutoff used to classify features as stable.\n"
                    report += f"- **Time Windows:** {stab_analysis.get('n_windows', 'N/A')}  — number of rolling windows used for stability estimation.\n\n"
                
                # Cross-validation Analysis
                if 'cv_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['cv_analysis']:
                    cv_analysis = enhanced_analysis['cv_analysis']
                    report += f"### Cross-Validation Analysis\n\n"
                    report += f"- **Average Consistency:** {cv_analysis.get('average_consistency', 'N/A'):.4f}  — average selection frequency across folds (0–1); higher means features reappear more often.\n"
                    report += f"- **Consistent Features:** {len(cv_analysis.get('consistent_features', []))}  — features with consistency above the threshold; more is better.\n"
                    report += f"- **Consistency Threshold:** {cv_analysis.get('consistency_threshold', 'N/A')}  — minimum fold frequency to be considered consistent.\n"
                    report += f"- **CV Folds:** {cv_analysis.get('cv_folds', 'N/A')}  — number of time-series splits used; more folds give a stricter stability test.\n\n"
                
                # Baseline Comparison
                if 'baseline_analysis' in enhanced_analysis and 'error' not in enhanced_analysis['baseline_analysis']:
                    base_analysis = enhanced_analysis['baseline_analysis']
                    report += f"### Baseline Comparison\n\n"
                    report += f"- **Improvement Ratio:** {base_analysis.get('improvement_ratio', 'N/A'):.2f}x  — selected set score / baseline score; values <1.0 mean the selection outperforms baseline.\n"
                    report += f"- **Selected Features Avg Score:** {base_analysis.get('average_selected_score', 'N/A'):.6f}  — mean importance of selected features; higher is better.\n"
                    report += f"- **Baseline Avg Score:** {base_analysis.get('average_baseline_score', 'N/A'):.6f}  — mean importance over all features; acts as a reference level.\n"
                    report += f"- **Baseline Trials:** {base_analysis.get('n_baseline_trials', 'N/A')}  — number of random baseline draws; more gives a more stable baseline estimate.\n"
                    report += f"- **Features Compared:** {base_analysis.get('n_features', 'N/A')}  — size of the selected feature set used for the comparison.\n\n"

                # NEW: Selection Frequency Distribution
                if 'frequency_distribution' in enhanced_analysis and enhanced_analysis['frequency_distribution'] and 'error' not in enhanced_analysis['frequency_distribution']:
                    freq_dist = enhanced_analysis['frequency_distribution']
                    report += f"### Selection Frequency Distribution\n\n"
                    report += f"- **Distribution Mode:** {freq_dist.get('selection_mode', 'N/A')}\n"
                    report += f"- **Interpretation:** {freq_dist.get('interpretation', 'N/A')}\n"
                    report += f"- **Highly Stable Features (>80%):** {freq_dist.get('highly_stable_count', 'N/A')}\n"
                    report += f"- **Highly Unstable Features (<20%):** {freq_dist.get('highly_unstable_count', 'N/A')}\n"
                    report += f"- **Unstable Features Ratio:** {freq_dist.get('unstable_features_ratio', 0):.1%}\n"

                    # Add frequency histogram
                    histogram = freq_dist.get('frequency_histogram', {})
                    if histogram:
                        report += f"\n**Frequency Breakdown:**\n"
                        for bin_name, data in sorted(histogram.items()):
                            if isinstance(data, dict):
                                report += f"- {bin_name}: {data.get('count', 0)} features ({data.get('percentage', 0):.1f}%)\n"
                            else:
                                report += f"- {bin_name}: {data} features\n"

                    # Add warnings
                    warnings = freq_dist.get('warnings', [])
                    if warnings:
                        report += f"\n**⚠️ Warnings:**\n"
                        for warning in warnings:
                            report += f"- {warning}\n"
                    report += f"\n"

                # NEW: Null Importance Analysis
                if 'null_importance' in enhanced_analysis and enhanced_analysis['null_importance'] and 'error' not in enhanced_analysis['null_importance']:
                    null_analysis = enhanced_analysis['null_importance']
                    report += f"### Null Importance Analysis (Statistical Significance)\n\n"
                    report += f"- **Significant Features (p < 0.05):** {null_analysis.get('n_significant', 'N/A')}\n"
                    report += f"- **FDR-Adjusted Significant:** {null_analysis.get('n_fdr_significant', 'N/A')}\n"
                    report += f"- **Mean P-Value:** {null_analysis.get('mean_p_value', 'N/A'):.4f}\n"
                    report += f"- **Permutations:** {null_analysis.get('n_permutations', 'N/A')}\n"
                    report += f"- **Execution Time:** {null_analysis.get('execution_time', 'N/A'):.1f}s\n"

                    # Add significance interpretation
                    total_features = len(selected_features_60) if selected_features_60 else 1
                    sig_ratio = null_analysis.get('n_significant', 0) / total_features if total_features > 0 else 0
                    if sig_ratio >= 0.8:
                        report += f"\n✅ **{sig_ratio:.0%}** of features are statistically significant\n"
                    elif sig_ratio >= 0.6:
                        report += f"\n⚠️ Only **{sig_ratio:.0%}** of features are statistically significant\n"
                    else:
                        report += f"\n🚨 **WARNING:** Only **{sig_ratio:.0%}** of features are statistically significant!\n"
                    report += f"\n"

                # NEW: Walk-Forward Validation
                if 'walk_forward_validation' in enhanced_analysis and enhanced_analysis['walk_forward_validation'] and 'error' not in enhanced_analysis['walk_forward_validation']:
                    wf_analysis = enhanced_analysis['walk_forward_validation']
                    report += f"### Walk-Forward Validation (OOS Performance)\n\n"
                    report += f"- **Optimal Feature Count:** {wf_analysis.get('optimal_feature_count', 'N/A')}\n"
                    report += f"- **Maximum OOS R²:** {wf_analysis.get('max_r2', 'N/A'):.4f}\n"
                    report += f"- **Positive Contribution Features:** {wf_analysis.get('n_positive_features', 'N/A')}\n"
                    report += f"- **Execution Time:** {wf_analysis.get('execution_time', 'N/A'):.1f}s\n"

                    # Add performance interpretation
                    max_r2 = wf_analysis.get('max_r2', 0)
                    if max_r2 >= 0.1:
                        report += f"\n✅ Good OOS performance (R² = {max_r2:.3f})\n"
                    elif max_r2 >= 0.05:
                        report += f"\n⚠️ Moderate OOS performance (R² = {max_r2:.3f})\n"
                    else:
                        report += f"\n🚨 **WARNING:** Low OOS performance (R² = {max_r2:.3f})\n"
                    report += f"\n"

                # NEW: Redundancy Clustering
                if 'redundancy_clustering' in enhanced_analysis and enhanced_analysis['redundancy_clustering'] and 'error' not in enhanced_analysis['redundancy_clustering']:
                    redun_cluster = enhanced_analysis['redundancy_clustering']
                    report += f"### Feature Redundancy Clustering\n\n"
                    report += f"- **Clusters Found:** {redun_cluster.get('n_clusters', 'N/A')}\n"
                    report += f"- **Representative Features:** {redun_cluster.get('n_representatives', 'N/A')}\n"
                    report += f"- **Redundant Features:** {redun_cluster.get('n_redundant', 'N/A')}\n"
                    report += f"- **Redundancy Ratio:** {redun_cluster.get('redundancy_ratio', 0):.1%}\n"
                    report += f"- **Execution Time:** {redun_cluster.get('execution_time', 'N/A'):.1f}s\n"

                    # Add redundancy interpretation
                    redun_ratio = redun_cluster.get('redundancy_ratio', 0)
                    if redun_ratio < 0.2:
                        report += f"\n✅ Low redundancy ({redun_ratio:.0%})\n"
                    elif redun_ratio < 0.4:
                        report += f"\n⚠️ Moderate redundancy ({redun_ratio:.0%}) - consider using representatives only\n"
                    else:
                        report += f"\n🚨 High redundancy ({redun_ratio:.0%}) - recommend filtering to representatives\n"
                    report += f"\n"

                # NEW: MI Stability Analysis
                if 'mi_stability' in enhanced_analysis and enhanced_analysis['mi_stability'] and 'error' not in enhanced_analysis['mi_stability']:
                    mi_stab = enhanced_analysis['mi_stability']
                    report += f"### Mutual Information Stability (Correlation Proxy)\n\n"
                    report += f"- **Stable Features (CV < 0.3):** {mi_stab.get('n_stable', 'N/A')}\n"
                    report += f"- **High MI Features (>0.1):** {mi_stab.get('n_high_mi', 'N/A')}\n"
                    report += f"- **Mean MI Stability:** {mi_stab.get('mean_mi_stability', 'N/A'):.3f}\n"
                    report += f"- **Method:** {mi_stab.get('method', 'N/A')}\n"
                    report += f"- **Execution Time:** {mi_stab.get('execution_time', 'N/A'):.1f}s\n"

                    # Add MI interpretation
                    mean_stability = mi_stab.get('mean_mi_stability', 0)
                    if mean_stability >= 0.7:
                        report += f"\n✅ High MI stability across folds\n"
                    elif mean_stability >= 0.5:
                        report += f"\n⚠️ Moderate MI stability\n"
                    else:
                        report += f"\n🚨 Low MI stability - features may not generalize well\n"
                    report += f"\n"

                # PHASE 3: Data Leakage Detection
                if 'leakage_detection' in enhanced_analysis and enhanced_analysis['leakage_detection'] and 'error' not in enhanced_analysis['leakage_detection']:
                    leakage = enhanced_analysis['leakage_detection']
                    report += f"### Data Leakage Detection (Phase 3)\n\n"
                    report += f"- **Perfect Correlations (>0.99):** {leakage.get('n_perfect', 0)}\n"
                    report += f"- **Suspicious Correlations (>0.95):** {leakage.get('n_suspicious', 0)}\n"
                    report += f"- **Execution Time:** {leakage.get('execution_time', 'N/A'):.1f}s\n"

                    # Show perfect features (critical)
                    perfect_features = leakage.get('perfect_features', [])
                    if perfect_features:
                        report += f"\n🚨 **CRITICAL - Potential Data Leakage:**\n"
                        for feature, corr in perfect_features[:10]:
                            report += f"- {feature}: r = {corr:.4f}\n"
                        report += f"\n**ACTION REQUIRED:** Investigate these features for data leakage!\n"

                    # Show suspicious features (warning)
                    suspicious_features = leakage.get('suspicious_features', [])
                    if suspicious_features and not perfect_features:
                        report += f"\n⚠️ **Suspicious Features:**\n"
                        for feature, corr in suspicious_features[:5]:
                            report += f"- {feature}: r = {corr:.4f}\n"
                        report += f"\n**RECOMMENDED:** Review these features to ensure no leakage\n"

                    # All clear
                    if not perfect_features and not suspicious_features:
                        report += f"\n✅ No data leakage detected\n"

                    report += f"\n"

                # PHASE 3: Feature Information Content
                if 'information_content' in enhanced_analysis and enhanced_analysis['information_content'] and 'error' not in enhanced_analysis['information_content']:
                    info_content = enhanced_analysis['information_content']
                    report += f"### Feature Information Content (Phase 3)\n\n"
                    report += f"- **Low Variance Features (<0.01):** {info_content.get('n_low_variance', 0)}\n"
                    report += f"- **Quasi-Constant Features (>99%):** {info_content.get('n_quasi_constant', 0)}\n"
                    report += f"- **Execution Time:** {info_content.get('execution_time', 'N/A'):.1f}s\n"

                    # Show low variance features
                    low_variance = info_content.get('low_variance_features', [])
                    if low_variance:
                        report += f"\n⚠️ **Low Variance Features:**\n"
                        for feature, var in low_variance[:5]:
                            report += f"- {feature}: variance = {var:.6f}\n"
                        if len(low_variance) > 5:
                            report += f"- ... and {len(low_variance) - 5} more\n"
                        report += f"\n**RECOMMENDED:** Remove low variance features\n"

                    # Show quasi-constant features
                    quasi_constant = info_content.get('quasi_constant_features', [])
                    if quasi_constant:
                        report += f"\n⚠️ **Quasi-Constant Features:**\n"
                        for feature, prop in quasi_constant[:5]:
                            report += f"- {feature}: {prop*100:.1f}% same value\n"
                        if len(quasi_constant) > 5:
                            report += f"- ... and {len(quasi_constant) - 5} more\n"
                        report += f"\n**RECOMMENDED:** Remove quasi-constant features\n"

                    # All clear
                    if not low_variance and not quasi_constant:
                        report += f"\n✅ All features have sufficient information content\n"

                    report += f"\n"

                # NEW: Meta-Label Diagnostics (IC/AUC vs binary_label and realized_return)
                meta_diag = enhanced_analysis.get('meta_label_diagnostics')
                if isinstance(meta_diag, dict):
                    report += f"### Meta-Label Diagnostics (IC/AUC vs Targets)\n\n"
                    report += (
                        "These diagnostics summarize how the final selected features relate to the "
                        "meta-label targets: binary_label (classification) and realized_return (economic P&L). "
                        "Scores are reported as Information Coefficient (Pearson correlation) and AUC where applicable.\n\n"
                    )

                    def _format_slice(title: str, slice_data: Dict[str, Any]) -> None:
                        nonlocal report
                        if not isinstance(slice_data, dict) or not slice_data:
                            return

                        rows: List[Dict[str, Any]] = []
                        for feat_name, metrics in slice_data.items():
                            if not isinstance(metrics, dict):
                                continue

                            bin_m = metrics.get('binary_label') or {}
                            ret_m = metrics.get('realized_return') or {}

                            ic_bin = bin_m.get('ic')
                            auc_bin = bin_m.get('auc')
                            n_bin = bin_m.get('n')
                            ic_ret = ret_m.get('ic')
                            n_ret = ret_m.get('n')

                            # Determine best IC magnitude for ranking
                            best_ic_vals = [v for v in [ic_bin, ic_ret] if isinstance(v, (int, float)) and np.isfinite(v)]
                            best_ic = max(best_ic_vals, key=lambda x: abs(x)) if best_ic_vals else None

                            rows.append({
                                'feature': feat_name,
                                'ic_bin': ic_bin,
                                'auc_bin': auc_bin,
                                'n_bin': n_bin,
                                'ic_ret': ic_ret,
                                'n_ret': n_ret,
                                'rank_score': abs(best_ic) if best_ic is not None else 0.0,
                            })

                        if not rows:
                            return

                        # Sort by absolute best IC
                        rows_sorted = sorted(rows, key=lambda r: r['rank_score'], reverse=True)[:10]

                        report += f"#### {title}\n\n"
                        report += (
                            "| Rank | Feature | IC (binary_label) | AUC (binary_label) | N (binary) | IC (realized_return) | N (ret) |\n" \
                            "| --- | --- | --- | --- | --- | --- | --- |\n"
                        )
                        for idx, row in enumerate(rows_sorted, 1):
                            def _fmt(v: Any) -> str:
                                if isinstance(v, (int, float)) and np.isfinite(v):
                                    return f"{v:.4f}"
                                return "N/A"

                            report += (
                                f"| {idx} | {row['feature']} | "
                                f"{_fmt(row['ic_bin'])} | {_fmt(row['auc_bin'])} | "
                                f"{row['n_bin'] if row.get('n_bin') is not None else 'N/A'} | "
                                f"{_fmt(row['ic_ret'])} | {row['n_ret'] if row.get('n_ret') is not None else 'N/A'} |\n"
                            )
                        report += "\n"

                    # Overall diagnostics
                    overall_slice = meta_diag.get('overall', {})
                    _format_slice("Overall (Full Sample)", overall_slice)

                    # Volatility regime slices
                    by_regime = meta_diag.get('by_volatility_regime', {})
                    if isinstance(by_regime, dict):
                        for reg_name, reg_slice in by_regime.items():
                            if isinstance(reg_slice, dict):
                                _format_slice(f"Volatility Regime = {reg_name}", reg_slice)

                    # TTO buckets
                    by_tto = meta_diag.get('by_tto_bucket', {})
                    if isinstance(by_tto, dict):
                        for bucket, bucket_slice in by_tto.items():
                            if isinstance(bucket_slice, dict):
                                label = {
                                    'short': 'Short TTO (fast exits)',
                                    'medium': 'Medium TTO',
                                    'long': 'Long TTO (slow exits)',
                                }.get(bucket, bucket)
                                _format_slice(f"TTO Bucket = {label}", bucket_slice)

            # Baseline learnability diagnostics for the final selected feature set (if available)
            baseline_check_results = feature_sets.get('baseline_check')
            if isinstance(baseline_check_results, dict) and baseline_check_results.get('success', False):
                from src.training.steps.pre_training.baseline_predictive_check import BaselinePredictiveCheck

                report += "\n## Baseline Learnability of Selected Features\n\n"
                report += (
                    "This baseline fits simple models (linear regression and small LightGBM baselines) "
                    "using only the final selected features. It provides an upper bound on how much of "
                    "the target variance is explainable by this feature set alone, before any complex "
                    "downstream modeling.\n\n"
                )

                temp_checker = BaselinePredictiveCheck()
                temp_checker.results = baseline_check_results
                report += temp_checker.format_for_markdown()

                csv_path = baseline_check_results.get('csv_path')
                if csv_path:
                    report += f"\n**Baseline learnability CSV:** `{csv_path}`\n"

                report += "\n### How to Read These Learnability Metrics\n\n"
                report += (
                    "- **Test R²** rows show, for each selected feature, how much of the target variance it "
                    "explains out-of-sample in a simple regression. Values near 0 mean weak signal; values "
                    "above roughly 0.3–0.4 indicate strong linear signal; negative values indicate that even "
                    "a simple model fails to generalize.\n"
                )
                report += (
                    "- The **quality score** aggregates how many features achieve positive Test R², how strong "
                    "the best feature(s) are, and how consistent performance is across evaluated features. "
                    "Scores close to 1.0 mean that many features contain robust, learnable signal; scores near "
                    "0 indicate that this feature set behaves mostly like noise.\n"
                )
                report += (
                    "If the selected-feature quality score is low, or if most Test R² values are negative, "
                    "it suggests that the final selection may be too aggressive or misaligned with the target. "
                    "In that case, consider revisiting labeling, feature generation, or selection thresholds "
                    "before relying on this set in production models.\n\n"
                )

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
