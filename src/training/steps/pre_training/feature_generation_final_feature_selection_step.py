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
import lightgbm as lgb

from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_success,
    tprint_warning,
    tprint_error,
    configure_tprint,
    TPrintConfig,
    LogLevel,
)

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

# Import FeatureSelector
try:
    from feature_selection.feature_selection_with_lgbm import FeatureSelector
    FEATURE_SELECTOR_AVAILABLE = True
    tprint_info("✅ FeatureSelector loaded from feature_selection.feature_selection_with_lgbm")
except ImportError as e:
    FEATURE_SELECTOR_AVAILABLE = False
    FeatureSelector = None
    tprint_warning(f"⚠️ FeatureSelector not available: {e}")

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

    def _create_selection_subsample(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a subsampled dataset for feature selection/discovery phases.

        Logic:
        1. Identify the selection window (default: last 6 months).
        2. Divide this window into N segments (default: 4).
        3. From each segment, take the last M days (default: 30 days).
        4. Concatenate these chunks.

        If the total dataset is smaller than the selection window, the entire
        dataset is used (with potential further subsampling if it's still huge).

        Args:
            features: Full feature DataFrame.
            targets: Full target DataFrame (aligned with features).
            config: Step configuration.

        Returns:
            Tuple of (subsampled_features, subsampled_targets).
        """
        # Determine whether targets are row-aligned with features. If not,
        # we will subsample only the feature matrix and keep targets
        # unchanged (e.g. when targets is labeling_metadata with 1 row).
        align_targets = False
        try:
            if isinstance(targets, (pd.DataFrame, pd.Series)):
                if len(targets) == len(features) and targets.index.equals(features.index):
                    align_targets = True
        except Exception:
            align_targets = False
        # Default config: Selection window = 180 days (6 months)
        selection_window_days = int(config.get("selection_window_days", 180))
        # Default config: 4 segments
        subsample_count = int(config.get("subsample_count", 4))
        # Default config: 30 days per segment
        subsample_days = int(config.get("subsample_period_days", 30))
        # Minimum total rows required to trigger subsampling (e.g. < 6 months data -> no subsampling)
        min_rows_threshold = 20000

        if len(features) < min_rows_threshold:
            tprint_info(
                f"📊 Dataset size ({len(features)}) < threshold ({min_rows_threshold}); "
                "skipping subsampling for final selection"
            )
            return features, targets

        try:
            # Prefer time-based subsampling when index is a DatetimeIndex
            if not isinstance(features.index, pd.DatetimeIndex):
                # Fallback to row-based slicing if no datetime index
                tprint_warning(
                    "⚠️ Features index is not DatetimeIndex; "
                    "falling back to row-based subsampling for final selection"
                )
                total_rows = len(features)
                rows_per_day = 96  # approx 15m bars
                selection_window_rows = selection_window_days * rows_per_day
                start_idx = max(0, total_rows - selection_window_rows)

                # Slice to selection window
                features_window = features.iloc[start_idx:]
                if align_targets:
                    targets_window = targets.iloc[start_idx:]
                else:
                    targets_window = targets

                # Split into segments
                segment_size = max(1, len(features_window) // subsample_count)
                subsample_size = subsample_days * rows_per_day

                indices_to_keep: List[int] = []
                for i in range(subsample_count):
                    seg_start = i * segment_size
                    seg_end = min(len(features_window), (i + 1) * segment_size)
                    if seg_start >= seg_end:
                        continue
                    # Take last chunk of segment
                    chunk_start = max(seg_start, seg_end - subsample_size)
                    indices_to_keep.extend(
                        range(start_idx + chunk_start, start_idx + seg_end)
                    )

                # Ensure unique and sorted
                indices_to_keep = sorted(set(indices_to_keep))
                if not indices_to_keep:
                    return features, targets

                sub_feats = features.iloc[indices_to_keep]
                if align_targets:
                    sub_targs = targets.iloc[indices_to_keep]
                else:
                    sub_targs = targets

                tprint_info(
                    "📊 Row-based subsampling (final FS): "
                    f"{len(features)} -> {len(sub_feats)} rows "
                    f"({len(indices_to_keep)/len(features):.1%})"
                )
                return sub_feats, sub_targs

            # Time-based slicing
            end_ts = features.index.max()
            start_ts = end_ts - pd.Timedelta(days=selection_window_days)

            # Slice to selection window
            mask_window = (features.index >= start_ts) & (features.index <= end_ts)
            features_window = features.loc[mask_window]
            if align_targets:
                targets_window = targets.loc[mask_window]
            else:
                targets_window = targets

            if features_window.empty:
                tprint_warning(
                    "⚠️ Selection window for final FS empty; using full dataset"
                )
                return features, targets

            window_duration = (
                features_window.index.max() - features_window.index.min()
            )
            if subsample_count <= 0 or window_duration <= pd.Timedelta(0):
                return features_window, targets_window

            segment_duration = window_duration / subsample_count
            subsample_duration = pd.Timedelta(days=subsample_days)

            chunks_features: List[pd.DataFrame] = []
            chunks_targets: List[pd.DataFrame] = []

            tprint_info(
                "📊 Subsampling for final FS from last "
                f"{selection_window_days} days (window: {start_ts} to {end_ts})"
            )

            for i in range(subsample_count):
                seg_start_ts = features_window.index.min() + i * segment_duration
                seg_end_ts = seg_start_ts + segment_duration

                # Define subsample range: [segment_end - subsample_days, segment_end]
                sub_end_ts = seg_end_ts
                sub_start_ts = max(seg_start_ts, sub_end_ts - subsample_duration)

                mask_sub = (features.index >= sub_start_ts) & (
                    features.index < sub_end_ts
                )
                chunk_f = features.loc[mask_sub]

                if not chunk_f.empty:
                    chunks_features.append(chunk_f)
                    if align_targets:
                        chunk_t = targets.loc[mask_sub]
                        if not chunk_t.empty:
                            chunks_targets.append(chunk_t)

            if not chunks_features:
                tprint_warning(
                    "⚠️ No chunks generated for final FS subsampling; using full dataset"
                )
                return features, targets

            sub_feats = pd.concat(chunks_features).sort_index()
            if align_targets and chunks_targets:
                sub_targs = pd.concat(chunks_targets).sort_index()
            else:
                # Targets are metadata or otherwise not row-aligned; keep
                # them unchanged while using a subsampled feature matrix.
                sub_targs = targets

            tprint_info(
                "📊 Time-based subsampling (final FS): "
                f"{len(features)} -> {len(sub_feats)} rows "
                f"({len(sub_feats)/len(features):.1%})"
            )
            tprint_info(
                f"   Using {subsample_count} chunks of ~{subsample_days} days "
                f"from the last {selection_window_days} days"
            )

            return sub_feats, sub_targs

        except Exception as e:
            tprint_error(
                f"Error in final FS subsampling: {e}; using full dataset"
            )
            return features, targets



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
                
                # STEP 2: Load generated_features_15m (large feature set) - use the latest version
                feature_versions = [v for v in store.list_versions() if 'generated_features_15m' in v.lower()]
                tprint_info(f"🔍 DEBUG: Found generated_features versions: {feature_versions}")

                if feature_versions:
                    # Heuristic: version names contain timestamps, so lexical sort gives latest
                    latest_features = sorted(feature_versions)[-1]
                    tprint_info(f"📂 Loading latest generated_features from: {latest_features}")
                    try:
                        large_features_df = store.get_view(latest_features).materialize()
                        tprint_success(f"✅ Loaded feature dataset: {large_features_df.shape}")
                        data_source = (
                            f"versioned_store:labeled="
                            f"{latest_labeled if labeled_df is not None else 'none'},"
                            f"features={latest_features}"
                        )
                    except Exception as e:
                        tprint_error(
                            f"❌ Failed to load generated_features_15m version '{latest_features}': {e}"
                        )
                        large_features_df = None
                
                # STEP 3: CRITICAL FIX - Align features and targets with proper time period handling
                # Priority: Use the intersection of time periods to ensure valid targets for all rows
                if large_features_df is not None and len(large_features_df) > 0:
                    tprint_success(f"🎯 PRIORITY: Using generated_features_15m ({len(large_features_df)} rows) as base dataset")
                    
                    # DETAILED DEBUGGING: Show time periods and index details
                    tprint_info(f"📊 DETAILED INDEX ANALYSIS:")
                    tprint_info(f"   Generated features (BASE): {large_features_df.shape}")
                    tprint_info(f"   - Index type: {type(large_features_df.index)}")
                    tprint_info(f"   - Index range: {large_features_df.index.min()} to {large_features_df.index.max()}")
                    
                    if labeled_df is not None:
                        tprint_info(f"   Labeled data (targets source): {labeled_df.shape}")
                        tprint_info(f"   - Index type: {type(labeled_df.index)}")
                        tprint_info(f"   - Index range: {labeled_df.index.min()} to {labeled_df.index.max()}")
                        
                        # Check if labeled_data covers a meaningful time range of features
                        common_index = labeled_df.index.intersection(large_features_df.index)
                        overlap_pct = len(common_index) / len(large_features_df) * 100 if len(large_features_df) > 0 else 0
                        
                        tprint_info(f"📊 Index overlap: {len(common_index)} rows ({overlap_pct:.1f}%)")
                        
                        # STRATEGY SELECTION based on overlap quality
                        if overlap_pct < 10:
                            # Very poor overlap - filter features to labeled_data's time range
                            tprint_warning(f"⚠️ Poor overlap ({overlap_pct:.1f}%) - using labeled_data's time range")
                            
                            if isinstance(labeled_df.index, pd.DatetimeIndex) and isinstance(large_features_df.index, pd.DatetimeIndex):
                                labeled_start = labeled_df.index.min()
                                labeled_end = labeled_df.index.max()
                                
                                # Filter features to the labeled_data's time range
                                time_mask = (large_features_df.index >= labeled_start) & (large_features_df.index <= labeled_end)
                                combined_df = large_features_df[time_mask].copy()
                                tprint_info(f"   Filtered features to labeled period: {len(combined_df)} rows")
                                
                                # Align targets using EXACT index matching only (no nearest-neighbor)
                                target_cols = [col for col in labeled_df.columns if 'target' in col.lower() or col == 'price_target_vol_normalized']
                                for col in target_cols:
                                    aligned_targets = labeled_df[col].reindex(combined_df.index)
                                    combined_df[col] = aligned_targets
                                    non_null = combined_df[col].notna().sum()
                                    tprint_info(f"   ✅ {col}: {non_null}/{len(combined_df)} non-null ({non_null/len(combined_df)*100:.1f}%)")
                            else:
                                # Non-datetime index - use common_index if any
                                if len(common_index) > 100:
                                    combined_df = large_features_df.loc[common_index].copy()
                                    for col in [c for c in labeled_df.columns if 'target' in c.lower()]:
                                        combined_df[col] = labeled_df.loc[common_index, col]
                                else:
                                    combined_df = large_features_df.copy()
                        else:
                            # Good overlap - use full features with target alignment
                            combined_df = large_features_df.copy()
                        
                        # Try multiple alignment strategies to get targets onto the features
                        target_cols = [col for col in labeled_df.columns if 'target' in col.lower() or col == 'price_target_vol_normalized']
                        tprint_info(f"🎯 Target columns to align: {target_cols}")
                        
                        if target_cols:
                            # Strategy 1: Direct index alignment (if indices match)
                            common_index = labeled_df.index.intersection(large_features_df.index)
                            tprint_info(f"   Direct index intersection: {len(common_index)} rows")
                            
                            for col in target_cols:
                                if len(common_index) > len(large_features_df) * 0.5:
                                    # Good overlap - use direct reindex
                                    aligned_targets = labeled_df[col].reindex(large_features_df.index)
                                    combined_df[col] = aligned_targets
                                    non_null_count = aligned_targets.notna().sum()
                                    tprint_info(f"   ✅ {col}: {non_null_count}/{len(aligned_targets)} non-null ({non_null_count/len(aligned_targets)*100:.1f}%)")
                                else:
                                    # Strategy 2: Try positional alignment if lengths are similar
                                    tprint_warning(f"   ⚠️ Low index overlap ({len(common_index)}/{len(large_features_df)}), trying alternative alignment...")
                                    
                                    # Check if indices are datetime but with timezone mismatch
                                    try:
                                        if hasattr(labeled_df.index, 'tz') and hasattr(large_features_df.index, 'tz'):
                                            # Try to localize/convert timezones
                                            labeled_tz = getattr(labeled_df.index, 'tz', None)
                                            features_tz = getattr(large_features_df.index, 'tz', None)
                                            
                                            if labeled_tz != features_tz:
                                                tprint_info(f"   Timezone mismatch: labeled={labeled_tz}, features={features_tz}")
                                                # Try converting labeled to match features
                                                if features_tz is None:
                                                    converted_idx = labeled_df.index.tz_localize(None)
                                                else:
                                                    converted_idx = labeled_df.index.tz_convert(features_tz) if labeled_tz else labeled_df.index.tz_localize(features_tz)
                                                
                                                temp_labeled = labeled_df.copy()
                                                temp_labeled.index = converted_idx
                                                aligned_targets = temp_labeled[col].reindex(large_features_df.index)
                                                combined_df[col] = aligned_targets
                                                non_null_count = aligned_targets.notna().sum()
                                                tprint_info(f"   ✅ {col} (tz-fixed): {non_null_count}/{len(aligned_targets)} non-null ({non_null_count/len(aligned_targets)*100:.1f}%)")
                                                continue
                                    except Exception as tz_err:
                                        tprint_warning(f"   Timezone alignment failed: {tz_err}")
                                    
                                    # Strategy 3: Fallback - use direct reindex with exact matching (may have many NaNs)
                                    aligned_targets = labeled_df[col].reindex(large_features_df.index)
                                    combined_df[col] = aligned_targets
                                    non_null_count = aligned_targets.notna().sum()
                                    tprint_warning(f"   ⚠️ {col} (fallback): {non_null_count}/{len(aligned_targets)} non-null ({non_null_count/len(aligned_targets)*100:.1f}%)")
                        else:
                            tprint_warning("⚠️ No target columns found in labeled_data, looking for alternatives...")
                            # Try to find any column that could be a target
                            potential_targets = [c for c in labeled_df.columns if any(t in c.lower() for t in ['label', 'return', 'signal'])]
                            if potential_targets:
                                tprint_info(f"   Found potential target columns: {potential_targets}")
                                for col in potential_targets[:1]:  # Take first one
                                    aligned_targets = labeled_df[col].reindex(large_features_df.index)
                                    combined_df[col] = aligned_targets
                    else:
                        tprint_warning("⚠️ No labeled_data available for target alignment - features only mode")
                    
                    # Use the combined dataset as labeled_df
                    labeled_df = combined_df
                    data_source = f"versioned_store:generated_features_15m_base,targets_from_labeled_data"
                    tprint_success(f"✅ FINAL: Using FULL dataset: {labeled_df.shape}")
                    tprint_success(f"✅ This is the FIX for the data size bottleneck!")
                    
                elif labeled_df is not None and large_features_df is None:
                    # Fallback: Only labeled_data available, no generated_features
                    tprint_warning(f"⚠️ No generated_features_15m found, using labeled_data as base ({len(labeled_df)} rows)")
                    tprint_warning(f"⚠️ This may result in a smaller dataset!")
                else:
                    tprint_error("❌ Neither generated_features_15m nor labeled_data available!")
                        
            except Exception as e:
                tprint_error(f"❌ Failed to load from versioned store: {e}")
                data_source = "fallback_to_generic_artifact_after_error"

            # FINAL FALLBACK: if versioned stores did not provide labeled_data,
            # fall back to the generic 'labeled_data' artifact. This preserves
            # the old working path for targets while the large feature matrix
            # still comes from generated_features_15m, avoiding the 300-row
            # bottleneck.
            if labeled_df is None:
                try:
                    tprint_warning("⚠️ Versioned stores yielded no labeled_data; falling back to generic 'labeled_data' artifact")
                    generic_labeled = self._get_artifact('labeled_data')
                    if generic_labeled is not None and hasattr(generic_labeled, 'shape'):
                        labeled_df = generic_labeled
                        data_source = f"generic_artifact:labeled_data"
                        tprint_success(f"✅ Fallback: loaded labeled_data via generic artifact: {labeled_df.shape}")
                    else:
                        tprint_warning("⚠️ Generic 'labeled_data' artifact missing or invalid; labeled_df remains None")
                except Exception as generic_exc:
                    tprint_warning(f"⚠️ Fallback generic labeled_data load failed: {generic_exc}")
            
            # REMOVED: Generic artifact fallback (was causing 300-row bottleneck)
            # The generic artifact contains pre-filtered small datasets
            # We must use versioned stores only for full datasets
            
            # FINAL VALIDATION: Log the data source and size
            tprint_info(f"🔍 DEBUG: Final data source: {data_source}")
            tprint_info(f"🔍 DEBUG: Final dataset size: {labeled_df.shape if labeled_df is not None else 'None'}")
            tprint_info(f"🔍 DEBUG: Execution mode: {execution_mode}")
            
            if labeled_df is None:
                raise ValueError("Failed to load labeled_data from any source")
            
            # CRITICAL FIX: If labeled_df is truncated (e.g. 97 rows) but we have a large feature set,
            # we MUST recover by using the large feature set and trying to salvage targets.
            # This happens when a previous step (labeling) runs in a truncated debug mode but we want full features.
            if large_features_df is not None and len(labeled_df) < 1000 and len(large_features_df) > 1000:
                tprint_error(f"🚨 DETECTED TRUNCATED LABELED DATA ({len(labeled_df)} rows) vs LARGE FEATURES ({len(large_features_df)} rows)")
                tprint_error("   Forcing use of LARGE feature set index and attempting target recovery...")

                # Use large features as the base dataframe
                full_df = large_features_df.copy()

                # Try to map targets from the small labeled_df onto the large df
                # This only works if the small df is a subset of the large one (by index)
                # If targets are missing for the rest, they will be NaN -> effectively unsupervised for those rows
                # BUT this preserves the feature data for calculating stats/distributions
                target_cols = [c for c in labeled_df.columns if 'target' in c.lower() or c == 'price_target_vol_normalized']

                for col in target_cols:
                    if col in labeled_df.columns:
                        # Reindex will introduce NaNs where index doesn't match
                        full_df[col] = labeled_df[col].reindex(full_df.index)

                # Swap the active dataframe
                tprint_success(f"   ✅ Swapped to full dataset: {len(full_df)} rows")
                labeled_df = full_df

                # Note: If targets are all NaN outside the small window, feature selection will only use the small window
                # BUT the artifacts saved will have the full history, which is critical for backtesting/inference.

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
            
            # CRITICAL FIX: If we loaded a larger generated_features_15m from the versioned store
            # in step 1, use that instead of the potentially smaller artifact from _collect_features
            if large_features_df is not None and hasattr(large_features_df, 'shape'):
                artifact_gf = features_data.get('generated_features')
                artifact_gf_rows = len(artifact_gf) if artifact_gf is not None and hasattr(artifact_gf, '__len__') else 0
                
                if len(large_features_df) > artifact_gf_rows:
                    tprint_success(f"🔧 CRITICAL FIX: Replacing artifact generated_features ({artifact_gf_rows} rows) "
                                 f"with versioned store generated_features_15m ({len(large_features_df)} rows)")
                    features_data['generated_features'] = large_features_df
                else:
                    tprint_info(f"📊 Artifact generated_features ({artifact_gf_rows} rows) >= "
                              f"versioned store ({len(large_features_df)} rows), keeping artifact version")
            
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
            combined_features_df = self._combine_features(features_data, labeled_df, config)
            tprint_info(f"✅ Combined {combined_features_df.shape} in {time.time()-t0:.2f}s")

            combined_features_df = self._apply_blank_mode_shaping(combined_features_df, config)

            # Sanitize leakage and near-constant features before final selection
            combined_features_df = self._sanitize_features_for_final_selection(combined_features_df, config)
            try:
                # Track total available candidate features (excluding targets/timestamp)
                pre_fs_counts = config.get("pre_feature_selector_stage_counts", {}) or {}
                combined_candidate_cols = [
                    c
                    for c in combined_features_df.columns
                    if c not in TARGET_COLUMN_NAMES + ["timestamp"]
                ]
                pre_fs_counts["combined_available_features"] = len(combined_candidate_cols)
                config["pre_feature_selector_stage_counts"] = pre_fs_counts
                tprint_info(
                    f"📊 Pre-FS: {len(combined_candidate_cols)} candidate feature columns after "
                    f"combine/shaping/sanitization (excluding targets/timestamp)"
                )
            except Exception:
                # Never break selection due to diagnostics bookkeeping
                pass
            
            # Log final combined feature matrix characteristics for all execution modes
            execution_mode_local = str(config.get('execution_mode', 'blank')).lower()
            tprint_info("📊 Combined features dataset after shaping/sanitization:")
            tprint_info(f"   Execution mode: {execution_mode_local}")
            tprint_info(f"   Shape: {combined_features_df.shape}")
            if isinstance(combined_features_df.index, pd.DatetimeIndex) and not combined_features_df.empty:
                tprint_info(
                    f"   Time range: {combined_features_df.index.min()} "
                    f"to {combined_features_df.index.max()}"
                )

            if combined_features_df.empty:
                raise ValueError("No features available for final selection")

            # Setup selection configuration
            selection_config = self._setup_selection_config(config)

            if self.selection_component is None:
                self.selection_component = FinalFeatureSelectionComponent(selection_config)

            # Apply subsampling for feature selection phases
            # Use subsample for selection to ensure regime diversity over recent history
            # But use full dataset for final artifact generation
            use_subsampling = config.get('subsample_selection', True)
            if use_subsampling:
                tprint_info("🔍 Creating subsampled dataset for final feature selection...")
                # Note: We need a helper method for subsampling in this class too
                # We'll define it or use a utility if available. Since it's private in the other step,
                # we'll implement a local version here to avoid dependencies.
                features_for_selection, targets_for_selection = self._create_selection_subsample(
                    combined_features_df, targets, config
                )
                tprint_info(f"📊 Selection subsample size: {len(features_for_selection)} rows")
            else:
                features_for_selection = combined_features_df
                targets_for_selection = targets

            feature_sets = self._perform_multi_size_selection(
                features_for_selection,
                targets_for_selection,
                config,
                full_features_df=combined_features_df # Pass full DF for artifact creation
            )

            shap_values = self._generate_shap_values(
                feature_sets,
                features_for_selection, # Use subsample for SHAP speed
                targets_for_selection,
                config,
            )

            baseline_check_results = None
            try:
                largest_key = max(
                    [k for k in feature_sets.keys() if k.startswith('selected_feature_dataframe_')],
                    default=None,
                    key=lambda name: int(str(name).split('_')[-1]) if str(name).split('_')[-1].isdigit() else 0,
                )
                if largest_key is not None:
                    largest_df = feature_sets[largest_key]
                    if isinstance(largest_df, pd.DataFrame) and not largest_df.empty:
                        baseline_check_results = self._run_baseline_predictive_check(
                            largest_df,
                            largest_df,
                            config,
                        )
                        if isinstance(baseline_check_results, dict):
                            feature_sets['baseline_check'] = baseline_check_results
            except Exception as e:
                tprint_warning(f"⚠️ Baseline predictive check skipped due to error: {e}")

            artifacts = self._generate_artifacts(
                feature_sets,
                shap_values,
                config,
                combined_features_df,
            )

            saved_artifacts: List[Dict[str, Any]] = []
            for artifact_name, artifact_data in artifacts.items():
                if isinstance(artifact_data, pd.DataFrame) and artifact_name.startswith('selected_feature_dataframe_'):
                    try:
                        artifact_path = self._save_artifact(
                            data=artifact_data,
                            artifact_name=artifact_name,
                            artifact_type='data',
                            data_category='features',
                        )
                        saved_artifacts.append(
                            {
                                'name': artifact_name,
                                'path': artifact_path,
                                'type': 'data',
                            }
                        )
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to save artifact '{artifact_name}': {e}")

            if saved_artifacts:
                artifacts['saved_feature_dataframes'] = saved_artifacts

            # CRITICAL: Save feature metadata for live trading
            # We save the top feature set (default 60) for live trading reconstruction
            try:
                feature_set_sizes = config.get('feature_set_sizes', [60, 50, 40, 30])
                if not feature_set_sizes:
                    feature_set_sizes = [60, 50, 40, 30]
                top_size = max(feature_set_sizes)
                selected_features_key = f'selected_features_{top_size}'

                if selected_features_key in feature_sets:
                    selected_features_list = feature_sets[selected_features_key]

                    # Create metadata dict matching FeatureMetadataStore expectation
                    # (though simple list is often enough for FeatureCalculator)
                    feature_metadata = {
                        'selected_features': selected_features_list,
                        'timestamp': datetime.now().isoformat(),
                        'feature_count': len(selected_features_list),
                        'model_type': config.get('execution_mode', 'analyst')
                    }

                    # Save as JSON artifact
                    # Note: We save it as 'feature_metadata' which is the standard key
                    # expected by the live pipeline loader
                    artifacts['feature_metadata'] = feature_metadata

                    tprint_success(f"✅ Saved feature_metadata with {len(selected_features_list)} features for live trading")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save feature_metadata for live trading: {e}")

            metrics = self._calculate_metrics(feature_sets, shap_values, config)
            try:
                optimization_metrics = self._get_optimization_metrics()
                if isinstance(optimization_metrics, dict):
                    metrics['optimization'] = optimization_metrics
            except Exception as e:
                tprint_warning(f"⚠️ Failed to collect optimization metrics: {e}")

            outcome_report = self._create_outcome_report(
                feature_sets,
                shap_values,
                config,
                baseline_check_results=baseline_check_results if isinstance(baseline_check_results, dict) else None,
            )
            markdown_report = self._generate_markdown_report(
                outcome_report,
                feature_sets,
                shap_values,
                config,
            )
            report_path = self._save_markdown_report(
                markdown_report,
                base_name='feature_generation_final_feature_selection',
            )

            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'feature_sets': feature_sets,
                'shap_values': shap_values,
                'outcome_report_path': report_path,
                'execution_time': 0.0,
            }

        except Exception as e:
            tprint_error(f"❌ Error executing {self.step_name}: {e}")
            self.logger.exception("Failed to execute final feature selection step")
            return {
                "success": False,
                "error": str(e),
                "artifacts": {},
                "metrics": {},
            }

    def _apply_blank_mode_shaping(self, combined_features_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply outlier compression and normalization to key feature blocks.

        This is a light, post-feature-generation shaping pass that:
        - runs in all execution modes (blank, light, full)
        - targets heavy-tailed blocks (wavelet / volatility / AD-line features)
        - applies modest quantile winsorization followed by robust normalization
        """
        try:
            execution_mode = str(config.get("execution_mode", "blank")).lower()

            if combined_features_df is None or combined_features_df.empty:
                return combined_features_df

            df = combined_features_df.copy()

            if execution_mode == "blank":
                raw_ohlcv_cols = [
                    "open", "high", "low", "close", "volume",
                    "symbol", "exchange", "interval", "timeframe",
                ]
                meta_time_cols = [
                    "open_time", "close_time", "day", "day_of_week", "hour",
                    "is_weekend", "price_range", "quote_volume", "trades",
                    "volatility_1d", "base_threshold",
                ]
                leakage_cols = [
                    "log_ret", "primary_signal", "volume_return",
                    "adaptive_profit_threshold", "wavelet_energy_vwap_9x_ratio",
                ]

                drop_candidates = [
                    c
                    for c in raw_ohlcv_cols + meta_time_cols + leakage_cols
                    if c in df.columns
                ]
                if drop_candidates:
                    df = df.drop(columns=drop_candidates, errors="ignore")

                max_nan_pct = float(config.get("blank_mode_max_nan_pct", 90.0))
                candidate_numeric = [
                    c
                    for c in df.columns
                    if c not in TARGET_COLUMN_NAMES + ["timestamp"]
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
                high_nan_cols: List[str] = []
                n_rows = len(df)
                if n_rows > 0:
                    for c in candidate_numeric:
                        nan_pct = 100.0 * float(df[c].isna().sum()) / float(n_rows)
                        if nan_pct > max_nan_pct:
                            high_nan_cols.append(c)
                if high_nan_cols:
                    df = df.drop(columns=high_nan_cols, errors="ignore")

                # Record detailed diagnostics for blank-mode shaping
                try:
                    pre_fs_detailed = config.get("pre_feature_selector_stage_counts_detailed", {}) or {}
                    shaping_detail = pre_fs_detailed.get("blank_mode_shaping", {})
                    shaping_detail["raw_meta_leakage_dropped"] = len(drop_candidates)
                    shaping_detail["candidate_numeric_before_high_nan"] = len(candidate_numeric)
                    shaping_detail["high_nan_dropped"] = len(high_nan_cols)
                    shaping_detail["candidate_numeric_after_high_nan"] = max(0, len(candidate_numeric) - len(high_nan_cols))
                    pre_fs_detailed["blank_mode_shaping"] = shaping_detail
                    config["pre_feature_selector_stage_counts_detailed"] = pre_fs_detailed
                except Exception:
                    pass

            feature_cols: List[str] = [
                col
                for col in df.columns
                if col not in TARGET_COLUMN_NAMES + ["timestamp", "log_ret", "primary_signal"]
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if not feature_cols:
                return df

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

            lower_q = float(config.get("blank_mode_lower_quantile", 0.0025))
            upper_q = float(config.get("blank_mode_upper_quantile", 0.9975))

            tprint_info(
                f"🔧 [SHAPING] execution_mode={execution_mode}: applying quantile "
                f"winsorization/robust normalization to key blocks "
                f"(lower_q={lower_q}, upper_q={upper_q})"
            )

            for block_name, cols in blocks:
                if not cols:
                    continue

                try:
                    block_df = df[cols]
                    lower_bounds = block_df.quantile(lower_q)
                    upper_bounds = block_df.quantile(upper_q)
                    block_clipped = block_df.clip(lower=lower_bounds, upper=upper_bounds, axis="columns")
                    block_normalized = robust_normalize(block_clipped)
                    df[cols] = block_normalized
                    tprint_info(
                        f"📊 [SHAPING] Block '{block_name}' shaped: {len(cols)} "
                        f"features (mode={execution_mode})"
                    )
                except Exception as e:
                    tprint_warning(
                        f"⚠️ [SHAPING] Skipping block '{block_name}' due to error "
                        f"(mode={execution_mode}): {e}"
                    )

            return df
        except Exception as e:
            tprint_warning(
                f"⚠️ [SHAPING] Failed to apply shaping (mode={execution_mode}), "
                f"returning original data: {e}"
            )
            return combined_features_df

    def _sanitize_features_for_final_selection(self, combined_features_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Programmatically clean leakage and near-constant features before final selection.

        This step is applied after blank-mode shaping and before final feature selection
        so that downstream training never sees known leakage columns or effectively
        constant features (which harm stability and diagnostics).
        """
        try:
            if combined_features_df is None or combined_features_df.empty:
                return combined_features_df

            df = combined_features_df.copy()

            default_leakage = {
                "wavelet_energy_vwap_9x_ratio",
                "volume_return",
                "log_ret",
                "primary_signal",
                "adaptive_profit_threshold",
            }
            cfg_leakage = set(config.get("leakage_feature_blacklist", []))
            leakage_cols = sorted((default_leakage | cfg_leakage) & set(df.columns))
            if leakage_cols:
                tprint_warning(
                    f"⚠️ Dropping {len(leakage_cols)} known leakage feature(s) before selection: {leakage_cols}"
                )
                df = df.drop(columns=leakage_cols, errors="ignore")

            near_const_std_threshold = float(config.get("near_constant_std_threshold", 1e-6))
            near_const_max_freq = float(config.get("near_constant_max_frequency", 0.999))
            candidate_cols: List[str] = [
                c
                for c in df.columns
                if c not in TARGET_COLUMN_NAMES + ["timestamp"]
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            near_constant_cols: List[str] = []
            n_rows = len(df)
            for col in candidate_cols:
                s = df[col]
                non_null = s.notna().sum()
                if non_null == 0:
                    near_constant_cols.append(col)
                    continue
                try:
                    std_val = float(s.std())
                except Exception:
                    std_val = np.nan
                if np.isfinite(std_val) and std_val <= near_const_std_threshold:
                    near_constant_cols.append(col)
                    continue
                try:
                    vc = s.value_counts(normalize=True, dropna=True)
                    if not vc.empty and float(vc.iloc[0]) >= near_const_max_freq:
                        near_constant_cols.append(col)
                except Exception:
                    continue

            if "adaptive_stop_threshold" in df.columns and "adaptive_stop_threshold" not in near_constant_cols:
                near_constant_cols.append("adaptive_stop_threshold")

            near_constant_cols = sorted(set(near_constant_cols))
            if near_constant_cols:
                preview = near_constant_cols[:10]
                tprint_warning(
                    f"⚠️ Dropping {len(near_constant_cols)} near-constant feature(s) before selection; "
                    f"sample: {preview}{' ...' if len(near_constant_cols) > 10 else ''}"
                )
                df = df.drop(columns=near_constant_cols, errors="ignore")

            # Record detailed diagnostics for sanitization
            try:
                pre_fs_detailed = config.get("pre_feature_selector_stage_counts_detailed", {}) or {}
                sanitize_detail = pre_fs_detailed.get("sanitize", {})
                sanitize_detail["leakage_dropped"] = len(leakage_cols)
                sanitize_detail["near_constant_dropped"] = len(near_constant_cols)
                sanitize_detail["candidate_before"] = len(candidate_cols)
                candidate_after = len([
                    c
                    for c in df.columns
                    if c not in TARGET_COLUMN_NAMES + ["timestamp"]
                    and pd.api.types.is_numeric_dtype(df[c])
                ])
                sanitize_detail["candidate_after"] = candidate_after
                pre_fs_detailed["sanitize"] = sanitize_detail
                config["pre_feature_selector_stage_counts_detailed"] = pre_fs_detailed
            except Exception:
                pass

            tprint_info(
                f"📊 Feature sanitization complete: {combined_features_df.shape[1]} → {df.shape[1]} columns "
                f"({n_rows} rows)"
            )
            return df
        except Exception as e:
            tprint_warning(f"⚠️ Feature sanitization failed, returning original data: {e}")
            return combined_features_df

    def _collect_features_from_previous_steps(self) -> Dict[str, Any]:
        features_data: Dict[str, Any] = {}

        timeframe = self._current_context.get("timeframe", "15m")

        generated_df = None
        try:
            main_name = f"generated_features_{timeframe}"
            generated_df = self._get_artifact(main_name, "data")
        except Exception:
            generated_df = None

        if generated_df is None:
            try:
                generated_df = self._get_artifact("generated_features", "data")
            except Exception:
                generated_df = None

        if generated_df is not None and hasattr(generated_df, "shape"):
            features_data["generated_features"] = generated_df
            tprint_info(f"📊 Loaded generated features: {generated_df.shape}")

        lookback_df = None
        try:
            lookback_df = self._get_artifact("lookback_optimization", "data")
        except Exception:
            lookback_df = None

        if lookback_df is not None:
            features_data["lookback_optimization"] = lookback_df

        for name in ["analyst_interactions", "tactician_interactions"]:
            inter_df = None
            try:
                inter_df = self._get_artifact(name, "data")
            except Exception:
                inter_df = None

            # Cross-load analyst_interactions from the analyst model store when
            # running in a different model context (e.g. blank) and the local
            # store does not contain the artifact. This allows blank-mode final
            # selection to reuse rich interaction features generated by the
            # analyst interaction step.
            if inter_df is None and name == "analyst_interactions":
                try:
                    ctx = self._current_context.copy()
                    symbol = ctx.get("symbol", "UNKNOWN")
                    exchange = ctx.get("exchange", "binance")
                    timeframe = ctx.get("timeframe", "15m")
                    direction = ctx.get("direction", "long")
                    cross_context = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "direction": direction,
                        "model": "analyst",
                        "step_name": "feature_generation_interaction_generation_step",
                    }
                    tprint_info(
                        "📂 Cross-loading 'analyst_interactions' from analyst model store "
                        "for use in final feature selection"
                    )
                    inter_df = self.artifact_router.load(
                        artifact_name="analyst_interactions",
                        artifact_type="data",
                        data_category="features",
                        context=cross_context,
                    )
                except Exception as e:
                    tprint_warning(
                        f"⚠️ Failed to cross-load analyst_interactions from analyst store: {e}"
                    )

            if inter_df is not None and hasattr(inter_df, "shape"):
                features_data[name] = inter_df
                tprint_info(f"📊 Loaded {name}: {inter_df.shape}")

        return features_data

    def _combine_features(self, features_data: Dict[str, Any], labeled_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Combine features from different sources into a single DataFrame with VectorBT optimizations.

        The config argument is used for mode-aware target selection (e.g. direction/model_type).
        """
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

        # Detailed pre-FeatureSelector diagnostics: per-source feature counts
        try:
            pre_fs_detailed = config.get("pre_feature_selector_stage_counts_detailed", {}) or {}
            sources_detail = pre_fs_detailed.get("sources", {})

            def _count_numeric_features(df: pd.DataFrame) -> int:
                return len([
                    c
                    for c in df.columns
                    if c not in TARGET_COLUMN_NAMES + ["timestamp"]
                    and pd.api.types.is_numeric_dtype(df[c])
                ])

            if isinstance(labeled_df, pd.DataFrame):
                sources_detail["labeled_df_non_target"] = _count_numeric_features(labeled_df)

            for key, data in features_data.items():
                if isinstance(data, pd.DataFrame):
                    # Count numeric, non-target features contributed by each source
                    sources_detail[key] = _count_numeric_features(data)

            pre_fs_detailed["sources"] = sources_detail
            config["pre_feature_selector_stage_counts_detailed"] = pre_fs_detailed
        except Exception:
            # Never break feature combination due to diagnostics bookkeeping
            pass
        
        tprint_info("🔄 Combining features with VectorBT optimizations...")
        
        # CRITICAL FIX: Detect if labeled_df contains only target columns (no actual features)
        # If so, use generated_features as the base and merge targets onto it
        target_like_cols = [c for c in labeled_df.columns 
                          if any(t in c.lower() for t in ['target', 'label', 'weight', 'fused'])]
        non_target_cols = [c for c in labeled_df.columns if c not in target_like_cols]
        
        labeled_df_has_features = len(non_target_cols) > 5  # More than just timestamp/index cols
        
        if not labeled_df_has_features and 'generated_features' in features_data:
            gf = features_data.get('generated_features')
            if gf is not None and isinstance(gf, pd.DataFrame) and not gf.empty:
                tprint_warning(f"⚠️ labeled_df has only {len(non_target_cols)} non-target columns: {non_target_cols}")
                tprint_info(f"🔧 FIX: Using generated_features ({gf.shape}) as base instead of labeled_df")
                
                # Start with generated_features as the base
                base_features = gf.copy()
                
                # Merge target columns from labeled_df onto generated_features
                # Handle index alignment
                if len(labeled_df) == len(gf):
                    # Same length: align by position
                    labeled_df_aligned = labeled_df.copy()
                    labeled_df_aligned.index = gf.index
                    for tc in target_like_cols:
                        if tc in labeled_df_aligned.columns:
                            base_features[tc] = labeled_df_aligned[tc].values
                            tprint_info(f"   ✅ Added target '{tc}' by position alignment")
                else:
                    # Different lengths: try index-based merge
                    try:
                        for tc in target_like_cols:
                            if tc in labeled_df.columns:
                                base_features[tc] = labeled_df[tc].reindex(gf.index)
                                non_null = base_features[tc].notna().sum()
                                tprint_info(f"   ✅ Added target '{tc}' by index merge ({non_null} non-null)")
                    except Exception as e:
                        tprint_warning(f"   ⚠️ Failed to merge targets by index: {e}")
                
                tprint_success(f"✅ base_features now has {len(base_features.columns)} columns "
                             f"({len([c for c in base_features.columns if c not in target_like_cols])} features + targets)")
                
                # Remove generated_features from features_data to avoid duplicate merging
                features_data = {k: v for k, v in features_data.items() if k != 'generated_features'}
            else:
                tprint_warning(f"⚠️ labeled_df has no features and generated_features is not available/valid")
                base_features = labeled_df.copy()
        else:
            # PRIORITY 1: Start with labeled dataframe to preserve target column
            base_features = labeled_df.copy()

        # If generated_features has the same number of rows but a different
        # index (typically a DatetimeIndex), align base_features to use that
        # index so that the engineered feature timestamps become canonical.
        try:
            if 'generated_features' in features_data and hasattr(features_data['generated_features'], 'shape'):
                gf = features_data['generated_features']
                if isinstance(gf, pd.DataFrame) and len(gf) == len(base_features):
                    if not base_features.index.equals(gf.index):
                        tprint_info("🔧 Aligning base_features index to generated_features index (shape matched)")
                        base_features = base_features.copy()
                        base_features.index = gf.index
        except Exception as e:
            tprint_warning(f"⚠️ Failed to align base_features index to generated_features index: {e}")
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

        # Also include generated_features as a primary engineered feature
        # dataframe when available so we do not lose the main feature block.
        if 'generated_features' in features_data and hasattr(features_data['generated_features'], 'shape'):
            gf = features_data['generated_features']
            if isinstance(gf, pd.DataFrame):
                dataframe_info.append({'name': 'generated_features', 'df': gf})
        
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

        # Now extract base_features and build feature_chunks
        base_features = dataframe_info[0]['df']
        feature_chunks = []
        
        tprint_error("=" * 80)
        tprint_error("🔍 HYPOTHESIS TEST: Processing dataframes for concatenation")
        tprint_error(f"   Total dataframes to process: {len(dataframe_info) - 1}")
        tprint_error("=" * 80)
        
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        basic_time_cols = ['open_time', 'close_time', 'day', 'day_of_week', 'hour', 'is_weekend', 'price_range', 'quote_volume', 'trades']
        
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
            
            if name in ('feature_dataframe', 'generated_features'):
                # Exclude OHLCV, basic time, target columns, AND columns already in base_features
                # This prevents double-counting when labeled_df already contains generated_features
                feature_cols = [col for col in df.columns
                              if col not in ohlcv_cols 
                              and col not in basic_time_cols 
                              and col not in TARGET_COLUMN_NAMES
                              and col not in base_features.columns]
                tprint_error(f"   {name}: Filtered to {len(feature_cols)} columns (excluded OHLCV/time/targets/duplicates)")
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

            # Ensure indices are unique before concatenation to avoid
            # pandas.errors.InvalidIndexError: "Reindexing only valid with
            # uniquely valued Index objects".
            concat_dfs: List[pd.DataFrame] = []
            for i, df in enumerate([base_features] + feature_chunks):
                if not df.index.is_unique:
                    dup_count = df.index.duplicated().sum()
                    tprint_warning(
                        f"⚠️ DataFrame #{i} has {dup_count} duplicate index entries; "
                        "dropping duplicates (keep='first') before concat"
                    )
                    df = df[~df.index.duplicated(keep="first")]
                concat_dfs.append(df)

            base_features = pd.concat(concat_dfs, axis=1)
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

        # Record detailed diagnostics for numeric filtering
        try:
            pre_fs_detailed = config.get("pre_feature_selector_stage_counts_detailed", {}) or {}
            combine_detail = pre_fs_detailed.get("combine", {})
            combine_detail["total_before_numeric_filter"] = len(base_features.columns)
            combine_detail["non_numeric_dropped"] = len(non_numeric_cols)
            combine_detail["total_after_numeric_filter"] = len(numeric_cols)
            pre_fs_detailed["combine"] = combine_detail
            config["pre_feature_selector_stage_counts_detailed"] = pre_fs_detailed
        except Exception:
            pass

        result_df = base_features[numeric_cols].copy()
        
        # Final check
        interaction_cols_final = [col for col in result_df.columns if 'interaction' in col.lower() or '_x_' in col.lower()]
        tprint_error(f"🔍 FINAL CHECK: Interaction columns in result_df: {len(interaction_cols_final)}")
        if interaction_cols_final:
            tprint_success(f"✅ Interaction features survived! Sample: {interaction_cols_final[:5]}")
        else:
            tprint_error(f"❌ NO INTERACTION FEATURES in final result!")
        
        # Debug: Check if target column is present
        # Priority: regression targets (target_long/short) > classifier targets (binary_label_long/short)
        direction = str(config.get('direction', 'long')).lower()
        model_type = str(config.get('model_type', 'regressor')).lower()
        
        # Log available targets for debugging
        available_targets = []
        if 'target_long' in result_df.columns or 'target_short' in result_df.columns:
            if direction == 'long' and 'target_long' in result_df.columns:
                available_targets = ['target_long']
                tprint_info(f"📊 Primary regression target: target_long ({result_df['target_long'].notna().sum()} non-NaN)")
            elif direction == 'short' and 'target_short' in result_df.columns:
                available_targets = ['target_short']
                tprint_info(f"📊 Primary regression target: target_short ({result_df['target_short'].notna().sum()} non-NaN)")
            else:
                available_targets = ['target_long', 'target_short']
                tprint_info(f"📊 Regression targets found: target_long ({result_df.get('target_long', pd.Series()).notna().sum()} non-NaN), target_short ({result_df.get('target_short', pd.Series()).notna().sum()} non-NaN)")
        elif 'target_long_fused' in result_df.columns or 'target_short_fused' in result_df.columns:
            available_targets = ['target_long_fused', 'target_short_fused']
            tprint_info("📊 Using fused target structure: target_long_fused, target_short_fused")
        elif 'binary_label_long' in result_df.columns or 'binary_label_short' in result_df.columns:
            if direction == 'long' and 'binary_label_long' in result_df.columns:
                available_targets = ['binary_label_long']
                tprint_info(f"📊 Classifier target: binary_label_long ({result_df['binary_label_long'].notna().sum()} non-NaN)")
            elif direction == 'short' and 'binary_label_short' in result_df.columns:
                available_targets = ['binary_label_short']
                tprint_info(f"📊 Classifier target: binary_label_short ({result_df['binary_label_short'].notna().sum()} non-NaN)")
            else:
                available_targets = ['binary_label_long', 'binary_label_short']
        # NOTE: legacy 'binary_label' fallback removed - use directional labels only
        else:
            # Fall back to legacy target detection
            available_targets = [col for col in PRIMARY_TARGET_COLUMN_NAMES if col in result_df.columns]
            tprint_info(f"📊 Using legacy target detection: {available_targets}")
        
        tprint_info(f"📊 Model type: {model_type}, Direction: {direction}")
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

        if 'timestamp' in result_df.columns:
            try:
                ts = pd.to_datetime(result_df['timestamp'], errors='coerce')
                valid_mask = ~ts.isna()
                if valid_mask.any():
                    if not valid_mask.all():
                        result_df = result_df.loc[valid_mask].copy()
                        ts = ts[valid_mask]
                    result_df.index = ts
                    result_df.index.name = 'timestamp'
                    result_df = result_df.sort_index()
            except Exception as e:
                tprint_warning(f"⚠️ Failed to standardize index from timestamp: {e}")
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
            # FIXED: More precise cross-timeframe detection to avoid misclassifying
            # features like "momentum_10" as cross-timeframe. Only match explicit
            # timeframe patterns like "_3x_ratio", "_15m_", or "ctf_" prefixes.
            is_cross_timeframe = (
                "ctf_" in name_lower
                or "cross_timeframe" in name_lower
                or "_ratio" in name_lower  # Cross-timeframe ratio features like "_3x_ratio"
                or re.search(r"_\d+x_", name_lower) is not None  # Explicit multiplier patterns like _3x_, _6x_
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
        full_features_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Perform feature selection for multiple feature set sizes with CMI-aware Tactician mode support.

        Args:
            features_df: The feature dataframe to use for selection (may be subsampled).
            targets: The target dataframe/series to use for selection (may be subsampled).
            config: Configuration dictionary.
            full_features_df: The full feature dataframe to use for creating final artifact dataframes.
                              If None, features_df is used (legacy behavior).
        """
        if full_features_df is None:
            full_features_df = features_df

        min_samples = int(config.get("lgbm_fs_min_target_samples", 200))
        # Master ranking from FeatureSelector (used to define the 60-feature set)
        fs_master_rank: List[str] = []

        def _is_usable_target(series: pd.Series, label: str) -> bool:
            try:
                non_null = int(series.notna().sum())
                total = int(series.shape[0]) if hasattr(series, "shape") else non_null
                effective_min = min(min_samples, max(10, max(1, total // 20)))
                if non_null < effective_min:
                    tprint_warning(
                        f"📊 Skipping primary target candidate {label} due to insufficient non-null samples: "
                        f"n_non_null={non_null}, min_required={effective_min} (total={total})"
                    )
                    return False
                unique = int(series.nunique(dropna=True))
                if unique < 2:
                    tprint_warning(
                        f"📊 Skipping primary target candidate {label} due to insufficient variation: "
                        f"n_unique={unique}"
                    )
                    return False
                return True
            except Exception as stats_exc:
                tprint_warning(
                    f"📊 Skipping primary target candidate {label} due to stats error: {stats_exc}"
                )
                return False

        def _select_primary_target_column(df: pd.DataFrame) -> Optional[str]:
            direction_local = str(config.get("direction", "long")).lower()
            # NEW: Support model_type config to choose between classifier and regressor
            # - classifier: Uses binary_label_long/short (directional classification)
            # - regressor: Uses target_long/short (expected returns, current default)
            model_type = str(config.get("model_type", "regressor")).lower()
            
            # Build direction-specific candidate lists based on model type
            directional_candidates_local: List[str] = []
            
            if model_type == "classifier":
                # For classifiers, prefer directional binary labels
                if direction_local == "long":
                    directional_candidates_local = [
                        "binary_label_long",  # NEW: Direction-specific binary label
                        "target_long",        # Fallback to regressor target
                        "target_long_fused",
                    ]
                elif direction_local == "short":
                    directional_candidates_local = [
                        "binary_label_short",  # NEW: Direction-specific binary label
                        "target_short",        # Fallback to regressor target
                        "target_short_fused",
                    ]
                else:
                    # For 'both' direction with classifier, use both directional labels
                    directional_candidates_local = [
                        "binary_label_long",
                        "binary_label_short",
                        "target_long",
                        "target_short",
                        "target_long_fused",
                        "target_short_fused",
                    ]
                tprint_info(f"🔧 Model type: classifier (using binary classification targets for {direction_local} direction)")
            else:
                # For regressors, prefer continuous target values (current behavior)
                if direction_local == "long":
                    directional_candidates_local = ["target_long", "target_long_fused"]
                elif direction_local == "short":
                    directional_candidates_local = ["target_short", "target_short_fused"]
                else:
                    directional_candidates_local = [
                        "target_long",
                        "target_short",
                        "target_long_fused",
                        "target_short_fused",
                    ]
                tprint_info(f"🔧 Model type: regressor (using continuous regression targets for {direction_local} direction)")

            for col in directional_candidates_local:
                if col in df.columns:
                    series = df[col]
                    if _is_usable_target(series, col):
                        tprint_info(f"📊 Using directional target for final selection: {col}")
                        return col

            for col in PRIMARY_TARGET_COLUMN_NAMES:
                if col in df.columns:
                    series = df[col]
                    if _is_usable_target(series, col):
                        tprint_info(f"📊 Using fallback primary target for final selection: {col}")
                        return col

            # Directional binary labels as fallback diagnostic candidates
            # NOTE: legacy 'binary_label' removed - use directional labels only
            diagnostic_candidates_local = [
                "binary_label_long",   # Direction-specific
                "binary_label_short",  # Direction-specific
                "realized_return",
            ]
            for col in diagnostic_candidates_local:
                if col in df.columns:
                    series = df[col]
                    if _is_usable_target(series, col):
                        tprint_info(
                            f"📊 Using diagnostic primary target for final feature selection: {col}"
                        )
                        return col

            return None
        def _apply_correlation_pruning(feature_names: List[str]) -> List[str]:
            try:
                if not feature_names:
                    return feature_names

                max_pool = int(config.get("final_fs_max_redundancy_pool", len(feature_names)))
                if max_pool <= 0:
                    return feature_names

                pool = feature_names[:max_pool]
                data = features_df[pool].copy()
                for col in data.columns:
                    data[col] = data[col].fillna(data[col].median())

                corr_matrix = data.corr(method="pearson").abs().fillna(0.0)
                threshold = float(config.get("final_fs_redundancy_corr_threshold", 0.7))

                selected_local: List[str] = []
                for fname in pool:
                    if not selected_local:
                        selected_local.append(fname)
                        continue

                    if fname not in corr_matrix.index:
                        selected_local.append(fname)
                        continue

                    max_corr = 0.0
                    for kept in selected_local:
                        if kept in corr_matrix.columns:
                            val = float(corr_matrix.loc[fname, kept])
                            if not np.isnan(val) and val > max_corr:
                                max_corr = val

                    if max_corr < threshold:
                        selected_local.append(fname)

                return selected_local
            except Exception as e:
                tprint_warning(f"⚠️ Correlation pruning in final selection failed, using original ranking: {e}")
                return feature_names
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
            # Standard Analyst mode now uses a dedicated 3-stage LightGBM pipeline:
            #   1) Gain importance with GOSS + very light params to drop the bottom 33%
            #   2) Permutation importance with TimeSeriesSplit CV=3 (GOSS, light params),
            #      preserving the top 60 and skimming 50% of features beyond rank 60
            #   3) Permutation importance with TimeSeriesSplit CV=5 (GOSS, normal params)
            #      for the final ranking that drives the 40/50/60 subsets.
            tprint_info("📊 Standard mode - using 3-stage LGBM selection (Gain → PI CV3 → PI CV5, captures interactions)")

        # Separate features from targets and exclude raw data columns
        raw_data_columns = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day_of_week', 'base_threshold']
        basic_features = [
            'open_time',
            'close_time',
            'body_size',
            'close_return',
            'price_range_pct',
            'volume_return',
            'close_log_return',
            'volume_log_return',
            'price_range',
            'body_size_pct',
            'trades',
            'quote_volume',
            'day',
            'lookahead_periods',
            'is_weekend',
            # Simple microstructure/meta columns from labeled_dataframe that we
            # still allow into the pool; they should not be hard-excluded.
            'volatility_1d',
            'adaptive_profit_threshold',
        ]
        
        # CRITICAL: Exclude performance metrics and forward-looking columns that are NOT predictive features
        # These are calculated from future data or are outcome/diagnostic metrics, not input features
        performance_metrics = [
            'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'recovery_factor',
            'win_rate', 'profit_factor', 'total_return', 'annualized_return', 'volatility',
            'var_95', 'cvar_95', 'downside_deviation', 'upside_capture', 'downside_capture',
            'information_ratio', 'treynor_ratio', 'jensen_alpha', 'max_consecutive_wins',
            'max_consecutive_losses', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
            'equity_curve', 'cumulative_returns', 'drawdown', 'underwater_curve',
            # Treat these as non-features for final selection; they are base diagnostics/parameters
            'volume_return', 'log_ret', 'primary_signal', 'adaptive_profit_threshold',
        ]
        
        # Debug: Show all available columns
        tprint_info(f"🔍 DEBUG: All columns in features_df: {list(features_df.columns)}")
        
        # Columns that we treat as meta/utility rather than final engineered
        # features for selection. These should not dominate the selected
        # feature sets in blank mode.
        meta_utility_columns = [
            'volatility_1d',
            'day',
            'quote_volume',
            'is_weekend',
            'price_range',
            'trades',
            'open_time',
        ]

        # Combine all columns to exclude (including performance metrics) but
        # NOT basic engineered microstructure features. We only remove raw
        # OHLCV/time columns, explicit target/label columns, outcome
        # performance metrics, and simple meta/utility columns.
        excluded_columns = (
            TARGET_COLUMN_NAMES
            + ['timestamp']
            + raw_data_columns
            + performance_metrics
            + meta_utility_columns
        )
        tprint_info(f"🔍 Excluding {len(excluded_columns)} columns: targets, timestamp, raw OHLCV/time, and performance metrics")
        
        # Candidate features are all non-excluded columns. Within this set we
        # soft-prioritize more advanced engineered features by name pattern
        # (vectorbt, wavelet, interactions, etc.), but basic engineered
        # features remain eligible.
        sophisticated_features = [
            col
            for col in features_df.columns
            if col not in excluded_columns
            and any(
                keyword in str(col).lower()
                for keyword in [
                    'vectorbt', 'interaction', 'enhanced', 'optimized', 'advanced',
                    'statistical', 'wavelet', 'entropy', 'ad_line', 'obv',
                    'volatility', 'order_flow', 'sr_', 'gmm_', 'hmm_', 'cluster',
                ]
            )
        ]
        basic_engineered_features = [
            col
            for col in features_df.columns
            if col not in excluded_columns and col not in sophisticated_features
        ]

        # Prioritize sophisticated features first, then remaining engineered
        # features so that we preserve a large candidate pool (>> 2 features).
        feature_cols = sophisticated_features + basic_engineered_features

        # Track pre-FeatureSelector stage counts for exclusions
        try:
            pre_fs_counts = config.get("pre_feature_selector_stage_counts", {}) or {}
            combined_available = pre_fs_counts.get("combined_available_features")
            pre_fs_counts["candidate_after_exclusions"] = len(feature_cols)
            if isinstance(combined_available, int):
                pre_fs_counts["excluded_dropped"] = max(0, combined_available - len(feature_cols))
            config["pre_feature_selector_stage_counts"] = pre_fs_counts

            if isinstance(combined_available, int):
                dropped_excl = pre_fs_counts.get("excluded_dropped", 0)
                tprint_info(
                    "📊 Pre-FS: "
                    f"{len(feature_cols)} candidate features after exclusions "
                    f"(dropped {dropped_excl} from {combined_available} combined features)"
                )
            else:
                tprint_info(
                    "📊 Pre-FS: "
                    f"{len(feature_cols)} candidate features after exclusions "
                    "(targets/raw/time/perf/meta removed)"
                )
        except Exception:
            # Diagnostics should not interfere with selection
            pass

        # Prefer meta-label outputs from feature_generation_meta_labeling_step.
        # We no longer fall back to fused/simplified price-based targets here;
        # if meta-label outputs are missing or empty, this step should fail
        target_cols: List[str] = []

        primary_col = _select_primary_target_column(features_df)
        if primary_col is not None:
            target_cols = [primary_col]

        if not target_cols:
            msg = (
                "No usable directional or meta-label target found for final feature selection. "
                "Expected one of ['target_long', 'target_short', 'binary_label', 'smoothed_label', 'realized_return'] "
                "with non-NaN values. Ensure labeling/meta-labeling steps have populated these."
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
        tprint_info(f"🎯 Final 3-stage LGBM selection will use target: '{target_cols[0]}'")

        # Expose the chosen primary target to downstream reporting via config
        # so that Markdown outcome reports can clearly state which label was
        # used for the final LGBM-based feature selection.
        try:
            config["final_selection_target_column"] = target_cols[0]
        except Exception:
            # Never break selection if config is not mutable for some reason.
            pass

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
        
        # NOTE: full_features_df already represents the full combined feature
        # history passed into this function (e.g. multi-year grid). Do NOT
        # overwrite it with the (possibly subsampled) features_df here – we
        # only subselect rows for selection, while artifacts are always built
        # on the full history.
        
        if nan_count_before > 0:
            pct_nan = 100 * nan_count_before / total_samples
            tprint_warning(f"⚠️ Found {nan_count_before} NaN values in target variable ({pct_nan:.2f}%)")
            
            # CRITICAL CHECK: If too many targets are NaN, we have an alignment problem
            if pct_nan > 90:
                tprint_error("=" * 80)
                tprint_error("🚨 CRITICAL: >90% of target values are NaN!")
                tprint_error("   This indicates a severe index alignment issue between features and targets.")
                tprint_error(f"   Features index range: {features_df.index.min()} to {features_df.index.max()}")
                tprint_error(f"   Non-NaN targets count: {(~y.isna()).sum()} out of {len(y)}")
                tprint_error("   The labeling integration step may have produced targets for a different time period.")
                tprint_error("=" * 80)
            
            # Get valid indices (where target is not NaN)
            valid_indices = y.notna()
            
            # Filter both X and y to remove NaN rows FOR SELECTION ONLY
            X = X[valid_indices]
            y = y[valid_indices]
            
            tprint_success(f"✅ For feature selection: {len(y)} rows with valid targets")
            tprint_info(f"   (Full dataset has {len(full_features_df)} rows - will be used for output)")
            
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

        # Coverage threshold: allow configuration and be more tolerant in
        # blank-model diagnostics so we don't collapse to a tiny core set of
        # always-present base features.
        model_name_local = str(config.get('model', '') or config.get('model_name', '')).lower()
        execution_mode_local = str(config.get('execution_mode', 'full')).lower()

        # Default for normal analyst/tactician runs
        default_min_cov = 95.0
        # In blank execution mode, drastically relax the default coverage
        # requirement unless the user explicitly overrides it. This preserves
        # sparse engineered features for diagnostics instead of collapsing to
        # a handful of always-present base/meta columns.
        if execution_mode_local == 'blank':
            default_min_cov = float(config.get('blank_mode_min_feature_coverage_pct', 5.0))

        MIN_COVERAGE_PCT = float(config.get('min_feature_coverage_pct', default_min_cov))
        MAX_NAN_PCT = 100.0 - MIN_COVERAGE_PCT

        # In blank mode, many engineered "event" features (signals/flags/patterns/
        # interactions) are defined only on sparse event timestamps and NaN on
        # other bars. For diagnostics we want "no event" on a bar to be treated
        # as 0 instead of NaN so coverage reflects actual information rather
        # than sparsity.
        if model_name_local == 'blank':
            event_keywords = ['signal', 'flag', 'pattern', 'event', 'interaction']
            for col in list(X.columns):
                name_lower = str(col).lower()
                if any(kw in name_lower for kw in event_keywords):
                    non_null = X[col].dropna()
                    if non_null.empty:
                        continue
                    # Only treat low-cardinality discrete features this way.
                    if non_null.nunique() <= 3:
                        nan_pct_raw = 100.0 * X[col].isna().sum() / len(X)
                        if nan_pct_raw > 0.0:
                            tprint_info(
                                f"🔧 Blank-mode: treating NaNs as 'no event' for sparse feature '{col}' "
                                f"(raw NaN={nan_pct_raw:.1f}%) before coverage filtering"
                            )
                            X[col] = X[col].fillna(0.0)

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

        # Track coverage-based reduction for pre-FeatureSelector diagnostics
        try:
            pre_fs_counts = config.get("pre_feature_selector_stage_counts", {}) or {}
            pre_fs_counts["coverage_kept"] = len(features_to_keep)
            pre_fs_counts["coverage_dropped"] = len(features_to_remove)
            config["pre_feature_selector_stage_counts"] = pre_fs_counts

            tprint_info(
                "📊 Pre-FS coverage filter: "
                f"kept {pre_fs_counts['coverage_kept']} features, "
                f"dropped {pre_fs_counts['coverage_dropped']} "
                f"(from {len(features_to_keep) + len(features_to_remove)})"
            )
        except Exception:
            pass

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

        # Use a correlation-based clustering preselector to build an orthogonal
        # master pool of features before running the main 3-stage LGBM pipeline.
        # Grid-search has been disabled; we fix the configuration to the
        # empirically best-performing setup: target_n ≈ 120, rho_max ≈ 0.3.
        if FEATURE_SELECTOR_AVAILABLE:
            # Determine the maximum requested set size (e.g. 60) and expand to a
            # larger, more diverse pool (default 120) before permutation-based RFE.
            fs_target_n = int(max(config.get('feature_set_sizes', [60, 50, 40, 30])))
            preselector_target_n = int(
                config.get('preselector_target_n_features', max(fs_target_n * 2, fs_target_n))
            )
            # Allow a light override of rho_max via config; default to 0.3.
            preselector_rho_max = float(config.get('preselector_rho_max', 0.3))

            tprint_info(
                "🚀 Running fixed correlation-based preselector (orthogonality-first) "
                f"with target_n={preselector_target_n}, rho_max={preselector_rho_max:.3f}..."
            )

            try:
                # Ensure y is a Series
                if isinstance(y, pd.DataFrame):
                    y_series = y.iloc[:, 0]
                else:
                    y_series = y

                # Rank features by simple Spearman correlation strength vs target
                # (absolute value).
                spearman_scores: Dict[str, float] = {}
                for col in X.columns:
                    try:
                        corr_val = X[col].corr(y_series, method="spearman")
                    except Exception:
                        corr_val = np.nan
                    spearman_scores[col] = corr_val

                spearman_series = pd.Series(spearman_scores).fillna(0.0)
                ranked_features = list(
                    spearman_series.abs().sort_values(ascending=False).index
                )

                # Orthogonality-first cluster cap: reuse FeatureSelector's
                # _cluster_cap_by_correlation with fixed rho_max and
                # max_per_cluster=1 to form an orthogonal pool up to the
                # requested preselector_target_n.
                fs = FeatureSelector(target_n_features=preselector_target_n, verbose=False)
                fs_master_rank = fs._cluster_cap_by_correlation(
                    X, ranked_features, max_per_cluster=1, rho_max=preselector_rho_max
                )

                if fs_master_rank:
                    if len(fs_master_rank) > preselector_target_n:
                        fs_master_rank = fs_master_rank[:preselector_target_n]

                    X = X[fs_master_rank]
                    feature_cols = fs_master_rank
                    tprint_success(
                        f"✅ Correlation-based preselector produced master pool with "
                        f"{len(fs_master_rank)} features (target {preselector_target_n}, "
                        f"rho_max={preselector_rho_max:.3f})"
                    )
                else:
                    tprint_warning(
                        "⚠️ Correlation-based preselector returned no features, proceeding with all features"
                    )
            except Exception as e:
                tprint_error(f"❌ Correlation-based preselector failed: {e}")
                tprint_warning("⚠️ Proceeding with all features")
        else:
            tprint_warning("⚠️ FeatureSelector not available, skipping pre-selection")

        tprint_info(f"🔍 Performing final feature selection on {len(feature_cols)} features using 3-stage LGBM pipeline...")
        tprint_info(f"📊 Final dataset: {len(X)} samples, {len(X.columns)} features")
        tprint_info("📊 Pipeline: Stage 1 Gain (GOSS, very light) → Stage 2 PI CV3 (GOSS, light) → Stage 3 PI CV5 (GOSS, normal)")

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

        # OPTIMIZED: Run the 3-stage LGBM pipeline ONCE, but:
        #   - the 60-feature set is defined directly by FeatureSelector's master
        #     ranking (fs_master_rank),
        #   - FinalFeatureSelectionComponent is only used to refine 50/40/30
        #     subsets **within** that 60-feature pool (no new features).
        tprint_info(
            "🔄 Using 3-stage LGBM selection only to refine subsets below the "
            "master 60-feature set from FeatureSelector..."
        )

        # Get the maximum size we need to select (typically 60)
        max_size = max(feature_set_sizes)

        if self.selection_component is None:
            raise RuntimeError("selection_component is not initialized before multi-size selection")

        # Ensure the main selection component is configured for the maximum size
        # so that its Stage 3 ranking can be reused for all subsets.
        try:
            # Update max_features/min_features on the existing config without re-instantiating
            self.selection_component.config.max_features = max_size
            if hasattr(self.selection_component.config, 'min_features'):
                self.selection_component.config.min_features = max(5, max_size // 2)
        except Exception:
            # If anything goes wrong, we still proceed with selection; the component
            # will fall back to its existing configuration.
            tprint_warning("⚠️ Failed to update selection component config; using existing max_features")

        tprint_info(
            f"🎯 Running 3-stage LGBM selection on FeatureSelector pool to refine subsets "
            f"(max_size={max_size}) using target '{target_cols[0]}'..."
        )

        leaky_zigzag_features = {
            "vectorbt_zigzag_10.0_5",
            "vectorbt_zigzag_5.0_2",
            "vectorbt_zigzag_7.0_3",
        }
        feature_cols = [f for f in feature_cols if f not in leaky_zigzag_features]
        if isinstance(X, pd.DataFrame):
            cols_in_X = [f for f in feature_cols if f in X.columns]
            X = X[cols_in_X]
            feature_cols = cols_in_X

        all_selected_features = self.selection_component.select_features(
            X, y, feature_cols, target_name=target_cols[0]
        )

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
        #     represented (increased caps for variants/base),
        #   - interaction and cross-timeframe features to be allowed but
        #     bounded so they don't dominate the pool.
        # 
        # IMPROVED BALANCE: Default caps favor more base/variant features:
        # - max_interaction_features: 15 (was 10) - allow some interactions
        # - max_cross_timeframe_total: 20 (unchanged) - reasonable CT limit
        # - max_variant_features: 50 (was 40) - allow more normalized variants
        # - min_variant_features: 10 (was 4) - ensure variants are well-represented

        max_interaction_features = int(config.get('max_interaction_features', 15))
        max_cross_timeframe_per_base = int(config.get('max_cross_timeframe_per_base', 10))
        # Global cap on cross-timeframe features across all bases (user
        # requirement: up to ~20 cross-timeframe features in total).
        max_cross_timeframe_total = int(config.get('max_cross_timeframe_total', 20))
        # Variants (VWAP/vol-normal/trend-adjusted) should be well represented
        # to capture different market dynamics.
        max_variant_features = int(config.get('max_variant_features', 50))

        # Log feature type distribution BEFORE capping
        pre_cap_interaction = sum(1 for f in all_selected_features if 'interaction' in f.lower() or '_x_' in f.lower())
        pre_cap_variant = sum(1 for f in all_selected_features if f.lower().endswith(('_volnorm', '_vwap', '_trend_adj', '_base')))
        pre_cap_ct = sum(1 for f in all_selected_features if 'ctf_' in f.lower() or 'cross_timeframe' in f.lower() or '_ratio' in f.lower())
        pre_cap_base = len(all_selected_features) - pre_cap_interaction - pre_cap_variant - pre_cap_ct
        
        tprint_info(f"📊 Feature type distribution BEFORE caps:")
        tprint_info(f"   - Base features: {pre_cap_base}")
        tprint_info(f"   - Variant features (volnorm/vwap/trend_adj/base): {pre_cap_variant}")
        tprint_info(f"   - Cross-timeframe features: {pre_cap_ct}")
        tprint_info(f"   - Interaction features: {pre_cap_interaction}")

        capped_features = self._apply_feature_caps(
            all_selected_features,
            max_interaction_features=max_interaction_features,
            max_cross_timeframe_per_base=max_cross_timeframe_per_base,
            max_variant_features=max_variant_features,
            max_cross_timeframe_total=max_cross_timeframe_total,
            # Soft minimums to ensure balanced representation
            min_interaction_features=int(config.get('min_interaction_features', 5)),
            min_cross_timeframe_features=int(config.get('min_cross_timeframe_features', 5)),
            min_variant_features=int(config.get('min_variant_features', 10)),
        )
        
        # Log feature type distribution AFTER capping
        post_cap_interaction = sum(1 for f in capped_features if 'interaction' in f.lower() or '_x_' in f.lower())
        post_cap_variant = sum(1 for f in capped_features if f.lower().endswith(('_volnorm', '_vwap', '_trend_adj', '_base')))
        post_cap_ct = sum(1 for f in capped_features if 'ctf_' in f.lower() or 'cross_timeframe' in f.lower() or '_ratio' in f.lower())
        post_cap_base = len(capped_features) - post_cap_interaction - post_cap_variant - post_cap_ct
        
        tprint_info(f"📊 Feature type distribution AFTER caps:")
        tprint_info(f"   - Base features: {post_cap_base}")
        tprint_info(f"   - Variant features: {post_cap_variant}")
        tprint_info(f"   - Cross-timeframe features: {post_cap_ct}")
        tprint_info(f"   - Interaction features: {post_cap_interaction}")

        # Guardrail: if caps are too aggressive and leave us with fewer
        # features than the largest requested set size, but the underlying
        # Stage 3 ranking produced enough candidates, fall back to the
        # uncapped top-N. This prevents all 60/50/40/30 sets from collapsing
        # down to a tiny core purely due to cap settings.
        if len(capped_features) < max_size and len(all_selected_features) >= max_size:
            tprint_warning(
                f"⚠️ Feature caps reduced candidate pool to {len(capped_features)} "
                f"(< max_size={max_size}); falling back to uncapped top-{max_size} "
                f"features from Stage 3 ranking. Consider relaxing cap settings "
                f"if you want stronger interaction/cross-timeframe limits."
            )
            capped_features = all_selected_features[:max_size]

        if not capped_features and not all_selected_features and not feature_cols:
            tprint_error("❌ CRITICAL: No features remain after selection and caps")
            return feature_sets

        tprint_error(f"   Selected features count after caps: {len(capped_features)}")
        tprint_error(f"   Selected features sample after caps: {capped_features[:5] if len(capped_features) > 0 else 'EMPTY'}")
        
        # Econ-aware second-stage preference: re-rank features so that those with
        # strong classification AUC vs label and positive economic alignment
        # (rank IC / return uplift vs realized_return) are preferred first, while
        # preserving the original LGBM ranking within each preference bucket.
        try:
            econ_diag = self._compute_meta_label_feature_diagnostics(features_df, capped_features)
            overall = econ_diag.get('overall') or {}

            econ_min_auc = float(config.get('econ_min_auc', 0.55))
            high_conf: List[str] = []
            mid_conf: List[str] = []
            low_conf: List[str] = []

            for name in capped_features:
                metrics = overall.get(name) or {}
                bin_m = metrics.get('binary_label') or {}
                rr_m = metrics.get('realized_return') or {}
                auc_val = bin_m.get('auc')
                rank_ic = rr_m.get('rank_ic')
                uplift = rr_m.get('return_uplift_top_vs_bottom')

                is_good_auc = isinstance(auc_val, (int, float)) and np.isfinite(auc_val) and auc_val >= econ_min_auc
                is_pos_rank = isinstance(rank_ic, (int, float)) and np.isfinite(rank_ic) and rank_ic > 0.0
                is_pos_uplift = isinstance(uplift, (int, float)) and np.isfinite(uplift) and uplift > 0.0

                if is_good_auc and (is_pos_rank or is_pos_uplift):
                    high_conf.append(name)
                elif is_good_auc or is_pos_rank or is_pos_uplift:
                    mid_conf.append(name)
                else:
                    low_conf.append(name)

            # Preserve original LGBM ordering within each bucket
            bucketed_order = [
                f for f in capped_features
                if f in high_conf
            ] + [
                f for f in capped_features
                if f in mid_conf
            ] + [
                f for f in capped_features
                if f in low_conf
            ]

            if bucketed_order and len(bucketed_order) == len(capped_features):
                tprint_info(
                    f"📊 Econ-aware re-ranking applied: high_conf={len(high_conf)}, "
                    f"mid_conf={len(mid_conf)}, low_conf={len(low_conf)}"
                )
                capped_features = bucketed_order
            else:
                tprint_warning("⚠️ Econ-aware re-ranking skipped due to inconsistent bucket sizes")
        except Exception as econ_exc:
            tprint_warning(f"⚠️ Econ-aware feature re-ranking failed, using original ranking: {econ_exc}")
        
        # Build a full ranked feature list that always uses as many candidate
        # features as possible for the requested set sizes. We start from the
        # capped/econ-ranked features, then append any remaining candidate
        # features from feature_cols that were not selected, preserving their
        # original order.
        primary_rank = capped_features if capped_features else all_selected_features
        full_ranked_features: List[str] = list(primary_rank) if primary_rank else []

        # Append any remaining candidate features that did not make it into
        # the primary ranking so that we can still fill 60/50/40/30 sets up
        # to the total number of available features.
        remaining_candidates = [
            f for f in feature_cols
            if f not in full_ranked_features
        ]
        if remaining_candidates:
            tprint_info(
                f"📊 Extending ranked feature list with {len(remaining_candidates)} "
                f"additional candidate features to satisfy multi-size set sizes"
            )
            full_ranked_features.extend(remaining_candidates)

        if not full_ranked_features:
            tprint_error("❌ No features available after building full ranked feature list")
            return feature_sets

        # Now create feature sets by slicing the ranked list:
        # - size == max_size (e.g. 60): use FeatureSelector master ranking directly
        # - smaller sizes: use LGBM/Econ-ranked list within the same pool, with
        #   lightweight correlation pruning that never introduces new features.
        for size in sorted(feature_set_sizes, reverse=True):  # Process from largest to smallest
            target_size = min(size, len(full_ranked_features))
            tprint_info(
                f"📊 Creating feature set for requested size {size} "
                f"(using {target_size} features from pool of {len(full_ranked_features)})..."
            )

            if size == max_size and fs_master_rank:
                # 60-feature set = pure FeatureSelector master ranking (no extra pruning)
                selected_features = fs_master_rank[:target_size]
                tprint_info(
                    f"📊 Using FeatureSelector master ranking for size {size}: "
                    f"{len(selected_features)} features"
                )
            else:
                # Slice the top N features from the full ranked list
                selected_features = full_ranked_features[:target_size]
                pruned_features = _apply_correlation_pruning(selected_features)

                # If correlation pruning removed too many features, refill from remaining pool
                if len(pruned_features) < target_size:
                    remaining_pool = [
                        f for f in full_ranked_features
                        if f not in pruned_features
                    ]
                    for f in remaining_pool:
                        pruned_features.append(f)
                        if len(pruned_features) >= target_size:
                            break

                selected_features = pruned_features
            
            tprint_error(f"🔍 DEBUG for size {size}:")
            tprint_error(f"   Selected features count: {len(selected_features)}")
            tprint_error(f"   Selected features sample: {selected_features[:5]}")
            
            if not selected_features:
                tprint_error(f"❌ CRITICAL: No features for size {size}!")
                continue
            
            feature_sets[f'selected_features_{size}'] = selected_features

            # CRITICAL FIX: Validate features exist in full_features_df before creating dataframe
            # Use full_features_df to ensure output artifacts cover the full history
            available_features = [f for f in selected_features if f in full_features_df.columns]
            missing_features = [f for f in selected_features if f not in full_features_df.columns]
            
            tprint_error(f"   Features available in full_features_df: {len(available_features)}/{len(selected_features)}")
            if missing_features:
                tprint_error(f"   ❌ Missing features: {missing_features[:10]}..." if len(missing_features) > 10 else f"   ❌ Missing features: {missing_features}")
            
            if not available_features:
                tprint_error(f"❌ CRITICAL: NO features from selected list exist in full_features_df!")
                tprint_error(f"   Selected features: {selected_features[:5]}")
                tprint_error(f"   full_features_df columns: {list(full_features_df.columns)[:10]}")
                continue
            
            # Create dataframe with available features + targets using full history
            all_cols_to_include = available_features + target_cols
            selected_dataframe = full_features_df[all_cols_to_include].copy()
            
            tprint_success(f"✅ Created selected_feature_dataframe_{size} (from FULL history):")
            tprint_success(f"   Shape: {selected_dataframe.shape}")
            tprint_success(f"   Features: {len(available_features)}")
            tprint_success(f"   Rows: {len(selected_dataframe)}")
            tprint_success(f"   Time range: {selected_dataframe.index.min()} to {selected_dataframe.index.max()}")
            
            feature_sets[f'selected_feature_dataframe_{size}'] = selected_dataframe

        tprint_success(f"✅ Created {len(feature_sets)} feature sets using optimized selection (1 computation instead of {len(feature_set_sizes)})")
        return feature_sets

    def _detect_tactician_mode(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> bool:
        """Detect whether we are in Tactician mode.

        This uses the current step name, execution context, and presence of
        Tactician/Analyst-specific features to infer whether the final feature
        selection step is running for a Tactician model.

        Args:
            features_df: Combined features dataframe.
            config: Configuration dictionary.

        Returns:
            True if in Tactician mode, False otherwise.
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
                    valid_ret = s_feat.notna() & yr.notna()
                    ic_ret = _safe_corr(s_feat, yr)

                    rank_ic_ret: Optional[float] = None
                    mean_ret_top: Optional[float] = None
                    mean_ret_bottom: Optional[float] = None
                    hit_rate_top: Optional[float] = None
                    hit_rate_bottom: Optional[float] = None
                    uplift_ret: Optional[float] = None
                    uplift_hit: Optional[float] = None
                    bucket_stats: Optional[Dict[str, Any]] = None

                    if valid_ret.sum() >= 100:
                        s_valid = s_feat[valid_ret]
                        r_valid = yr[valid_ret]

                        # Rank IC (Spearman-style)
                        try:
                            rank_ic_val = s_valid.rank().corr(r_valid.rank())
                            if isinstance(rank_ic_val, (int, float)) and np.isfinite(rank_ic_val):
                                rank_ic_ret = float(rank_ic_val)
                        except Exception:
                            rank_ic_ret = None

                        # Quantile-based economic metrics (returns and hit-rate by feature quantile)
                        try:
                            # Use up to 5 quantiles; duplicates='drop' handles low-variance features gracefully
                            q_labels = pd.qcut(s_valid, 5, labels=False, duplicates='drop')
                            df_q = pd.DataFrame({'q': q_labels, 'ret': r_valid})
                            grouped = df_q.groupby('q')

                            mean_ret_by_q = grouped['ret'].mean()
                            hit_rate_by_q = grouped['ret'].apply(lambda x: float((x > 0).mean()) if len(x) > 0 else np.nan)

                            bucket_stats = {
                                'mean_return_by_quantile': {int(k): float(v) for k, v in mean_ret_by_q.dropna().items()},
                                'hit_rate_by_quantile': {int(k): float(v) for k, v in hit_rate_by_q.dropna().items()},
                            }

                            if len(mean_ret_by_q) >= 2 and len(hit_rate_by_q) >= 2:
                                top_q = int(mean_ret_by_q.idxmax())
                                bottom_q = int(mean_ret_by_q.idxmin())

                                try:
                                    mean_ret_top = float(mean_ret_by_q.loc[top_q])
                                    mean_ret_bottom = float(mean_ret_by_q.loc[bottom_q])
                                    uplift_ret = mean_ret_top - mean_ret_bottom
                                except Exception:
                                    mean_ret_top = None
                                    mean_ret_bottom = None
                                    uplift_ret = None

                                try:
                                    hit_rate_top = float(hit_rate_by_q.loc[top_q])
                                    hit_rate_bottom = float(hit_rate_by_q.loc[bottom_q])
                                    uplift_hit = hit_rate_top - hit_rate_bottom
                                except Exception:
                                    hit_rate_top = None
                                    hit_rate_bottom = None
                                    uplift_hit = None
                        except Exception:
                            bucket_stats = None

                    n_ret = int(valid_ret.sum())
                    feat_metrics['realized_return'] = {
                        'ic': ic_ret,
                        'rank_ic': rank_ic_ret,
                        'n': n_ret,
                        'mean_return_top_quantile': mean_ret_top,
                        'mean_return_bottom_quantile': mean_ret_bottom,
                        'hit_rate_top_quantile': hit_rate_top,
                        'hit_rate_bottom_quantile': hit_rate_bottom,
                        'return_uplift_top_vs_bottom': uplift_ret,
                        'hit_rate_uplift_top_vs_bottom': uplift_hit,
                        'bucket_stats': bucket_stats,
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

        try:
            overall = results.get('overall') or {}
            econ_rank = []
            econ_uplift = []
            for name, metrics in overall.items():
                if not isinstance(metrics, dict):
                    continue
                rr_metrics = metrics.get('realized_return') or {}
                v_rank = rr_metrics.get('rank_ic')
                v_uplift = rr_metrics.get('return_uplift_top_vs_bottom')
                if isinstance(v_rank, (int, float)) and np.isfinite(v_rank):
                    econ_rank.append((name, float(v_rank)))
                if isinstance(v_uplift, (int, float)) and np.isfinite(v_uplift):
                    econ_uplift.append((name, float(v_uplift)))
            econ_rank = sorted(econ_rank, key=lambda x: abs(x[1]), reverse=True)[:5]
            econ_uplift = sorted(econ_uplift, key=lambda x: abs(x[1]), reverse=True)[:5]
            if econ_rank:
                msg = ', '.join(f"{n}={v:.4f}" for n, v in econ_rank)
                tprint_info(f"📊 Econ rank-IC (realized_return) top: {msg}")
            if econ_uplift:
                msg = ', '.join(f"{n}={v:.6f}" for n, v in econ_uplift)
                tprint_info(f"📊 Econ return uplift (top-bottom) top: {msg}")
        except Exception:
            pass

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
                
                # Get the actual feature data for these selected features + target columns.
                # Treat log_ret/primary_signal/smoothed_label as pseudo-targets and
                # never include them as training features.
                pseudo_targets = {'log_ret', 'primary_signal', 'smoothed_label'}
                available_features = [
                    f
                    for f in feature_list
                    if f in combined_features_df.columns and f not in pseudo_targets
                ]
                
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

        # Feature scores and LGBM stage summary from selection component (regular artifacts)
        if self.selection_component:
            artifacts['feature_scores'] = self.selection_component.get_feature_scores()
            # Persist the per-stage LGBM pipeline summary so downstream tools and
            # reports can see how many features were kept at each stage and
            # which CV configuration was used.
            try:
                stage_summary = getattr(self.selection_component, 'stage_summary', None)
            except Exception:
                stage_summary = None
            if stage_summary:
                artifacts['lgbm_stage_summary'] = stage_summary

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
        # Attach LGBM multi-stage pipeline summary if available
        if self.selection_component is not None:
            try:
                selection_metadata['lgbm_stage_summary'] = getattr(self.selection_component, 'stage_summary', None)
            except Exception:
                selection_metadata['lgbm_stage_summary'] = None
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

        # FeatureSelector stage-wise counts and cluster diagnostics (if available)
        fs_report = config.get('feature_selector_selection_report')
        if isinstance(fs_report, dict):
            stage_keys = [
                'prefilter_input',
                'stage1_kept', 'stage1_dropped',
                'stage2_kept', 'stage2_dropped',
                'stage3_kept', 'stage3_dropped',
            ]
            metrics['feature_selector_stage_counts'] = {
                k: fs_report.get(k) for k in stage_keys
            }

            cluster_hist = fs_report.get('cluster_histogram')
            if isinstance(cluster_hist, dict):
                metrics['feature_selector_cluster_histogram'] = cluster_hist

            cluster_csv = fs_report.get('cluster_assignments_csv_path')
            if isinstance(cluster_csv, str) and cluster_csv:
                metrics['feature_selector_cluster_csv'] = cluster_csv

        # Pre-FeatureSelector pipeline stage counts (if available)
        pre_fs_report = config.get('pre_feature_selector_stage_counts')
        if isinstance(pre_fs_report, dict):
            metrics['pre_feature_selector_stage_counts'] = pre_fs_report

        # Detailed Pre-FeatureSelector diagnostics (if available)
        pre_fs_detailed = config.get('pre_feature_selector_stage_counts_detailed')
        if isinstance(pre_fs_detailed, dict):
            metrics['pre_feature_selector_stage_counts_detailed'] = pre_fs_detailed

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

        # Prefer LightGBM for model-based diagnostics; fall back to RandomForest
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
            lgbm_available = True
        except Exception as e:
            lgbm_available = False
            tprint_warning(
                f"⚠️ LightGBM not available for feature quality CSV report; "
                f"falling back to RandomForest: {e}"
            )

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

        # Never treat target_* columns as features in quality diagnostics
        selected_cols = [c for c in selected_cols if not str(c).startswith("target_")]
        if not selected_cols:
            tprint_warning("⚠️ Only target_* columns were available for quality CSV report; skipping")
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

        # Step 1: Require valid target (always)
        tprint_info("🔧 Step 1: Require only valid target")
        df = df[df[target_col].notna()]
        rows_after_target_filter = len(df)
        tprint_info(f"  After target filter: {rows_after_target_filter} rows ({100*rows_after_target_filter/initial_rows:.1f}% retained)")

        # Step 2: Drop first 200 rows if they have ANY NaN (indicator warm-up period)
        WARMUP_ROWS = 200
        tprint_info(f"🔧 Step 2: Drop first {WARMUP_ROWS} rows with NaN (indicator warm-up period)")

        if len(df) > WARMUP_ROWS:
            warmup_df = df.iloc[:WARMUP_ROWS]
            post_warmup_df = df.iloc[WARMUP_ROWS:]

            # Drop rows with ANY NaN in warm-up period
            warmup_clean = warmup_df.dropna()
            rows_dropped_warmup = len(warmup_df) - len(warmup_clean)

            tprint_info(f"  Warm-up period ({WARMUP_ROWS} rows):")
            tprint_info(f"    - Rows with NaN: {rows_dropped_warmup}")
            tprint_info(f"    - Clean rows kept: {len(warmup_clean)}")
            tprint_info(f"    - Retention: {100*len(warmup_clean)/len(warmup_df):.1f}%")

            # Recombine (clean warm-up + rest of data)
            df = pd.concat([warmup_clean, post_warmup_df])
            rows_after_warmup_filter = len(df)
            tprint_info(f"  After warm-up filter: {rows_after_warmup_filter} rows ({100*rows_after_warmup_filter/initial_rows:.1f}% retained)")
        else:
            tprint_warning(f"  ⚠️ Dataset too small ({len(df)} rows) for warm-up filtering, skipping")
            rows_after_warmup_filter = len(df)

        # Step 3: Impute remaining NaN values (post warm-up period only)
        tprint_info("🔧 Step 3: Intelligent imputation for remaining sparse features (post warm-up)")

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
        warmup_dropped = rows_after_target_filter - rows_after_warmup_filter

        tprint_info("=" * 80)
        tprint_info(f"📊 PRIORITY 1 RESULTS:")
        tprint_info(f"  Initial rows: {initial_rows}")
        tprint_info(f"  After target filter: {rows_after_target_filter}")
        tprint_info(f"  After warm-up filter: {rows_after_warmup_filter} (dropped {warmup_dropped} warm-up rows)")
        tprint_info(f"  Final rows: {final_rows}")
        tprint_info(f"  Overall retention rate: {retention_rate:.1f}%")
        tprint_info(f"  Breakdown:")
        tprint_info(f"    - Target filter: {initial_rows - rows_after_target_filter} rows removed")
        tprint_info(f"    - Warm-up filter: {warmup_dropped} rows removed (first {WARMUP_ROWS} with NaN)")
        tprint_info(f"    - Imputation: {rows_after_warmup_filter - final_rows} additional rows handled")

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

        # Model (use LGBM with Stage-3 style params when available)
        if is_classification:
            if lgbm_available:
                model = LGBMClassifier(
                    objective="binary",
                    boosting_type="gbdt",
                    data_sample_strategy="goss",
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=128,
                    max_depth=7,
                    subsample=1.0,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    top_rate=0.2,
                    other_rate=0.1,
                )
            else:
                model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            score_fn = lambda yt, pr: roc_auc_score(yt, pr)
        else:
            if lgbm_available:
                model = LGBMRegressor(
                    objective="regression",
                    boosting_type="gbdt",
                    data_sample_strategy="goss",
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=128,
                    max_depth=7,
                    subsample=1.0,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    top_rate=0.2,
                    other_rate=0.1,
                )
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
            # Variant features end with specific suffixes
            is_variant = (
                nl.endswith("_volnorm")
                or nl.endswith("_vwap")
                or nl.endswith("_trend_adj")
                or nl.endswith("_base")  # Include _base variants too
            )
            # FIXED: More precise cross-timeframe detection to avoid misclassifying
            # features like "momentum_10" as cross-timeframe. Only match explicit
            # timeframe patterns like "_3x_ratio", "_15m_", or "ctf_" prefixes.
            is_cross_timeframe = (
                "ctf_" in nl
                or "cross_timeframe" in nl
                or "_ratio" in nl  # Cross-timeframe ratio features
                or re.search(r"_\d+x_", nl) is not None  # Explicit multiplier patterns like _3x_, _6x_
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

            meta_diag = enhanced_analysis.get('meta_label_diagnostics')
            if meta_diag and isinstance(meta_diag, dict):
                overall = meta_diag.get('overall') or {}
                for name, metrics in overall.items():
                    if not isinstance(metrics, dict):
                        continue
                    row = row_map.get(name)
                    if row is None:
                        continue
                    rr_metrics = metrics.get('realized_return') or {}
                    v_ic = rr_metrics.get('ic')
                    v_rank_ic = rr_metrics.get('rank_ic')
                    v_ret_uplift = rr_metrics.get('return_uplift_top_vs_bottom')
                    v_hit_uplift = rr_metrics.get('hit_rate_uplift_top_vs_bottom')
                    if isinstance(v_ic, (int, float)) and np.isfinite(v_ic):
                        row['econ_rr_ic'] = float(v_ic)
                    if isinstance(v_rank_ic, (int, float)) and np.isfinite(v_rank_ic):
                        row['econ_rr_rank_ic'] = float(v_rank_ic)
                    if isinstance(v_ret_uplift, (int, float)) and np.isfinite(v_ret_uplift):
                        row['econ_rr_return_uplift_top_vs_bottom'] = float(v_ret_uplift)
                    if isinstance(v_hit_uplift, (int, float)) and np.isfinite(v_hit_uplift):
                        row['econ_rr_hit_rate_uplift_top_vs_bottom'] = float(v_hit_uplift)

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
                        'hardwadaptive_optimization_operationsare_optimization_operations': 0,
                        'cpu_optimization_operations': 0,
                        'gpu_optimization_operations': 0,
                        'memory_optimization_operations': 0,
                        '': 0
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

            if results.get('success', False):
                summary = results.get('summary', {}) or {}
                best_auc = summary.get('clf_best_test_auc')
                best_auc_feat = summary.get('clf_best_feature')
                if isinstance(best_auc, (int, float)) and np.isfinite(best_auc):
                    tprint_info(f"📊 Baseline best AUC: {best_auc:.3f} (feature={best_auc_feat})")

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
- **Final Selection Target Column:** {config.get('final_selection_target_column', 'unknown')}
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

                # Add Expected Sharpe Ratio Calculation
                summary_local = baseline_check_results.get('summary', {})
                best_test_r2 = summary_local.get('best_test_r2', 0.0)
                if best_test_r2 > 0:
                    # Expected Sharpe ≈ sqrt(R²) * sqrt(N_bets)
                    # Assumption: 1 trade per day annualized (252 days)
                    n_bets_annual = 252
                    expected_sharpe = np.sqrt(best_test_r2) * np.sqrt(n_bets_annual)

                    report += f"\n### Backtest Implications (Expected Sharpe)\n\n"
                    report += f"- **Best Test R²:** {best_test_r2:.4f}\n"
                    report += f"- **Assumed Trading Frequency:** 1 trade/day (252/year)\n"
                    report += f"- **Expected Annualized Sharpe Ratio:** **{expected_sharpe:.2f}**\n"
                    report += f"  *(Derived from Information Coefficient approximation: Sharpe ≈ IC * sqrt(N))*\n"

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

                    def _collect_top_rr_metric(metric_key: str, top_n: int = 5) -> List[Tuple[str, float]]:
                        rows: List[Tuple[str, float]] = []
                        for feat_name, metrics in overall.items():
                            if not isinstance(metrics, dict):
                                continue
                            rr_metrics = metrics.get('realized_return') or {}
                            val = rr_metrics.get(metric_key)
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                rows.append((feat_name, float(val)))
                        rows_sorted = sorted(rows, key=lambda x: abs(x[1]), reverse=True)
                        return rows_sorted[:top_n]

                    top_bin = _collect_top_ic('binary_label')
                    top_rr = _collect_top_ic('realized_return')
                    top_rr_rank_ic = _collect_top_rr_metric('rank_ic')
                    top_rr_uplift = _collect_top_rr_metric('return_uplift_top_vs_bottom')

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

                    if top_rr_rank_ic or top_rr_uplift:
                        report += "\n\n### Econ-Aligned Diagnostics vs realized_return\n\n"
                        if top_rr_rank_ic:
                            report += "**Top 5 features by rank IC vs realized_return:**\n\n"
                            for rank, (feat, val) in enumerate(top_rr_rank_ic, 1):
                                report += f"{rank}. {feat} (Rank IC = {val:.4f})\\n"
                        if top_rr_uplift:
                            report += "\n**Top 5 features by top-vs-bottom return uplift (by feature quantile):**\n\n"
                            for rank, (feat, val) in enumerate(top_rr_uplift, 1):
                                report += f"{rank}. {feat} (Return uplift = {val:.6f})\\n"
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
            )

            # Pre-FeatureSelector pipeline summary (if available)
            pre_fs = config.get('pre_feature_selector_stage_counts')
            if isinstance(pre_fs, dict) and pre_fs:
                report += "## Pre-FeatureSelector Pipeline Stage Summary\n\n"
                report += "| Stage | Kept | Dropped |\n"
                report += "|-------|------|---------|\n"

                combined = pre_fs.get('combined_available_features')
                candidate = pre_fs.get('candidate_after_exclusions')
                cov_kept = pre_fs.get('coverage_kept')
                cov_dropped = pre_fs.get('coverage_dropped')

                if isinstance(combined, int):
                    report += f"| Combined (after shaping/sanitization) | {combined} | - |\n"

                if isinstance(candidate, int):
                    if isinstance(combined, int):
                        excl_drop = max(0, combined - candidate)
                        report += (
                            "| After exclusions (no targets/raw/time/perf/meta) | "
                            f"{candidate} | {excl_drop} |\n"
                        )
                    else:
                        report += (
                            "| After exclusions (no targets/raw/time/perf/meta) | "
                            f"{candidate} | - |\n"
                        )

                if isinstance(cov_kept, int):
                    if isinstance(candidate, int):
                        cov_drop = max(0, candidate - cov_kept)
                    elif isinstance(cov_dropped, int):
                        cov_drop = cov_dropped
                    else:
                        cov_drop = "-"
                    report += (
                        "| After coverage filter (FS input) | "
                        f"{cov_kept} | {cov_drop} |\n\n"
                    )

            # Detailed Pre-FeatureSelector diagnostics (if available)
            pre_fs_detailed = config.get('pre_feature_selector_stage_counts_detailed')
            if isinstance(pre_fs_detailed, dict) and pre_fs_detailed:
                report += "### Pre-FeatureSelector Detailed Diagnostics\n\n"

                # Per-source feature counts
                sources = pre_fs_detailed.get('sources') or {}
                if isinstance(sources, dict) and sources:
                    report += "**Per-Source Numeric Feature Counts (pre-combine):**\n\n"
                    report += "| Source | Numeric Features (non-target) |\n"
                    report += "|--------|-------------------------------|\n"
                    for src_name, cnt in sorted(sources.items()):
                        report += f"| {src_name} | {cnt} |\n"
                    report += "\n"

                # Blank-mode shaping diagnostics
                shaping = pre_fs_detailed.get('blank_mode_shaping') or {}
                if isinstance(shaping, dict) and shaping:
                    report += "**Blank-Mode Shaping (raw/meta/leakage + high-NaN) :**\n\n"
                    report += "| Metric | Value |\n"
                    report += "|--------|-------|\n"
                    report += f"| Raw/meta/leakage dropped | {shaping.get('raw_meta_leakage_dropped', 'N/A')} |\n"
                    report += f"| Candidate numeric before high-NaN | {shaping.get('candidate_numeric_before_high_nan', 'N/A')} |\n"
                    report += f"| High-NaN features dropped | {shaping.get('high_nan_dropped', 'N/A')} |\n"
                    report += f"| Candidate numeric after high-NaN | {shaping.get('candidate_numeric_after_high_nan', 'N/A')} |\n\n"

                # Sanitization diagnostics
                sanitize = pre_fs_detailed.get('sanitize') or {}
                if isinstance(sanitize, dict) and sanitize:
                    report += "**Sanitization (leakage + near-constant):**\n\n"
                    report += "| Metric | Value |\n"
                    report += "|--------|-------|\n"
                    report += f"| Known leakage dropped | {sanitize.get('leakage_dropped', 'N/A')} |\n"
                    report += f"| Near-constant features dropped | {sanitize.get('near_constant_dropped', 'N/A')} |\n"
                    report += f"| Candidate features before sanitization | {sanitize.get('candidate_before', 'N/A')} |\n"
                    report += f"| Candidate features after sanitization | {sanitize.get('candidate_after', 'N/A')} |\n\n"

                # Combine-level numeric filtering diagnostics
                combine_detail = pre_fs_detailed.get('combine') or {}
                if isinstance(combine_detail, dict) and combine_detail:
                    report += "**Combine Stage (numeric filtering after concatenation):**\n\n"
                    report += "| Metric | Value |\n"
                    report += "|--------|-------|\n"
                    report += f"| Columns before numeric filter | {combine_detail.get('total_before_numeric_filter', 'N/A')} |\n"
                    report += f"| Non-numeric columns dropped | {combine_detail.get('non_numeric_dropped', 'N/A')} |\n"
                    report += f"| Columns after numeric filter | {combine_detail.get('total_after_numeric_filter', 'N/A')} |\n\n"

            # FeatureSelector stage-wise summary and cluster diagnostics (if available)
            fs_report = config.get('feature_selector_selection_report')
            if isinstance(fs_report, dict) and fs_report:
                report += "## FeatureSelector Stage Summary (Master 60-Set)\n\n"
                report += "| Stage | Kept | Dropped |\n"
                report += "|-------|------|---------|\n"
                pre_input = fs_report.get('prefilter_input', 'N/A')
                report += f"| Input | {pre_input} | - |\n"
                s1k = fs_report.get('stage1_kept', 'N/A')
                s1d = fs_report.get('stage1_dropped', 'N/A')
                s2k = fs_report.get('stage2_kept', 'N/A')
                s2d = fs_report.get('stage2_dropped', 'N/A')
                s3k = fs_report.get('stage3_kept', 'N/A')
                s3d = fs_report.get('stage3_dropped', 'N/A')
                report += f"| Stage 1 (Pre-filters) | {s1k} | {s1d} |\n"
                report += f"| Stage 2 (Clustering) | {s2k} | {s2d} |\n"
                report += f"| Stage 3 (LGBM RFE + cluster cap) | {s3k} | {s3d} |\n\n"

                cluster_hist = fs_report.get('cluster_histogram')
                if isinstance(cluster_hist, dict) and cluster_hist:
                    report += "### Correlation-Cluster Histogram (Final 60-Set)\n\n"
                    report += "| Cluster ID | Feature Count |\n"
                    report += "|------------|---------------|\n"
                    for cid in sorted(cluster_hist.keys()):
                        report += f"| {cid} | {cluster_hist.get(cid, 0)} |\n"
                    csv_path = fs_report.get('cluster_assignments_csv_path')
                    if isinstance(csv_path, str) and csv_path:
                        report += f"\nCluster assignments CSV: `{csv_path}`\n\n"

            report += "## Feature Selection Results\n\n"
            
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

                # Add Expected Sharpe Ratio Calculation
                summary_local = baseline_check_results.get('summary', {})
                best_test_r2 = summary_local.get('best_test_r2', 0.0)
                if best_test_r2 > 0:
                    # Expected Sharpe ≈ sqrt(R²) * sqrt(N_bets)
                    # Assumption: 1 trade per day annualized (252 days)
                    n_bets_annual = 252
                    expected_sharpe = np.sqrt(best_test_r2) * np.sqrt(n_bets_annual)

                    report += f"\n### Backtest Implications (Expected Sharpe)\n\n"
                    report += f"- **Best Test R²:** {best_test_r2:.4f}\n"
                    report += f"- **Assumed Trading Frequency:** 1 trade/day (252/year)\n"
                    report += f"- **Expected Annualized Sharpe Ratio:** **{expected_sharpe:.2f}**\n"
                    report += f"  *(Derived from Information Coefficient approximation: Sharpe ≈ IC * sqrt(N))*\n"

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
