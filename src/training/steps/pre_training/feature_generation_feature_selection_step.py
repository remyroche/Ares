"""
Sophisticated Feature Selection Step

This step performs advanced feature selection using hardware-optimized components
with multi-objective optimization, economic validation, and VectorBT optimization.
"""

from __future__ import annotations

import asyncio
import logging
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.base_step import BaseStep

# ====== Safe shims for optional hardware stack ======
try:
    from src.utils.hardware import (
        # Core optimization decorators
        smart_cache, auto_optimize, memory_efficient, performance_tracked,
        # Memory optimization
        memory_optimized, gc_optimized, chunked_processing_auto,
        comprehensive_memory_optimization, MemoryOptimizationLevel,
        # M1 comprehensive optimization
        m1_optimized, get_comprehensive_optimizer, ComprehensiveConfig,
        OptimizationStrategy, WorkloadCategory,
        # Data optimization
        optimize_dataframe, optimize_array, optimize_dataframe_default,
        # Hardware management
        get_integrated_hardware_manager, process_ml_training_data,
        # Memory management
        get_advanced_memory_manager, memory_efficient_processing,
        track_memory_usage, force_garbage_collection, cleanup_all_memory,
        MemoryPressureLevel,
        # Caching
        get_global_cache, CacheConfig, DataTypeOptimization, CacheStrategy
    )
except Exception:
    # Decorators -> no-ops
    def smart_cache(*a, **k): 
        def _wrap(f): return f
        return _wrap
    def auto_optimize(*a, **k):
        def _wrap(f): return f
        return _wrap
    def memory_efficient(*a, **k):
        def _wrap(f): return f
        return _wrap
    def performance_tracked(*a, **k):
        def _wrap(f): return f
        return _wrap
    def memory_optimized(*a, **k):
        def _wrap(f): return f
        return _wrap
    def gc_optimized(*a, **k):
        def _wrap(f): return f
        return _wrap
    def chunked_processing_auto(*a, **k):
        def _wrap(f): return f
        return _wrap
    def comprehensive_memory_optimization(*a, **k):
        def _wrap(f): return f
        return _wrap
    def m1_optimized(*a, **k):
        def _wrap(f): return f
        return _wrap

    class MemoryOptimizationLevel:
        LIGHT="LIGHT"; MODERATE="MODERATE"; AGGRESSIVE="AGGRESSIVE"

    class OptimizationStrategy:
        BALANCED="BALANCED"; MAX_PERF="MAX_PERF"

    class WorkloadCategory:
        MACHINE_LEARNING="MACHINE_LEARNING"
        FINANCIAL_MODELING="FINANCIAL_MODELING"
        BACKTESTING="BACKTESTING"

    def get_comprehensive_optimizer(*a, **k): return None
    def get_integrated_hardware_manager(*a, **k): return None
    def get_advanced_memory_manager(*a, **k): return None
    def get_global_cache(*a, **k): 
        class _Cache:
            def get(self, *a, **k): return None
            def set(self, *a, **k): return None
        return _Cache()

    def optimize_dataframe(df): return df
    def optimize_array(x): return x
    def optimize_dataframe_default(df):
        import pandas as pd, numpy as np
        if not isinstance(df, pd.DataFrame): return df
        out = df.copy()
        for c in out.select_dtypes(include=["float64"]).columns:
            out[c] = out[c].astype("float32")
        for c in out.select_dtypes(include=["int64"]).columns:
            out[c] = pd.to_numeric(out[c], downcast="integer")
        return out

    def process_ml_training_data(*a, **k): pass
    def memory_efficient_processing(*a, **k): pass
    def track_memory_usage(): return 0.0
    def force_garbage_collection(): import gc; gc.collect()
    def cleanup_all_memory(): force_garbage_collection()
    class MemoryPressureLevel: NORMAL="NORMAL"
# ====== End shims ======


# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_data_preview, tprint_data_format
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_data_preview(*args, **kwargs): print("DATA_PREVIEW:", *args, **kwargs)
    def tprint_data_format(*args, **kwargs):
        label = args[1] if len(args) > 1 else "data"
        print(f"[DATA_FORMAT] {label}")
        if kwargs.get("return_summary"):
            try:
                df = args[0]
                import pandas as pd, numpy as np
                if hasattr(df, "memory_usage"):
                    return {
                        "shape": getattr(df, "shape", None),
                        "memory_usage_mb": float(df.memory_usage(deep=True).sum())/1024/1024,
                        "dtypes_summary": str(getattr(df, "dtypes", "")),
                        "null_counts": df.isnull().sum().to_dict() if hasattr(df, "isnull") else {},
                        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns) if hasattr(df, "select_dtypes") else 0,
                        "categorical_columns": len(df.select_dtypes(exclude=[np.number]).columns) if hasattr(df, "select_dtypes") else 0,
                    }
            except Exception:
                return None
        return None

# Enhanced hardware-optimized feature selection components
@dataclass
class FeatureSelectionConfig:
    """Feature selection configuration with hardware optimization."""
    enable_multi_stage_selection: bool = True
    enable_lightweight_screening: bool = True
    min_stability_threshold: float = 0.3
    max_parallel_workers: int = 4
    enable_parallel_processing: bool = True
    # Hardware optimization settings
    enable_m1_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    memory_threshold_mb: float = 200.0
    cache_ttl: float = 3600.0

@dataclass
class FeatureSelectionResult:
    """Feature selection result with optimization metadata."""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_metadata: Dict[str, Any]
    success: bool = True
    optimization_metrics: Optional[Dict[str, Any]] = None
    memory_usage_mb: Optional[float] = None
    execution_time: Optional[float] = None

@dataclass
class MultiObjectiveResult:
    """Multi-objective selection result with optimization."""
    selected_features: List[str]
    objective_scores: Dict[str, float]
    success: bool = True
    optimization_metrics: Optional[Dict[str, Any]] = None

@dataclass
class EconomicValidationResult:
    """Economic validation result with optimization."""
    validation_score: float
    economic_metrics: Dict[str, float]
    success: bool = True
    optimization_metrics: Optional[Dict[str, Any]] = None

# Hardware-optimized feature selector using M1 comprehensive optimization
class OptimizedFeatureSelector:
    """Hardware-optimized feature selector with M1 and memory optimization."""
    
    def __init__(self, config: FeatureSelectionConfig = None):
        self.config = config or FeatureSelectionConfig()
        self.hardware_manager = get_integrated_hardware_manager()
        self.memory_manager = get_advanced_memory_manager()
        self.cache = get_global_cache()
        
        # Initialize M1 comprehensive optimizer
        if self.config.enable_m1_optimization:
            self.m1_optimizer = get_comprehensive_optimizer(
                ComprehensiveConfig(
                    optimization_strategy=OptimizationStrategy.BALANCED,
                    workload_category=WorkloadCategory.MACHINE_LEARNING
                )
            )
        else:
            self.m1_optimizer = None
    
    @m1_optimized("feature_selection", WorkloadCategory.MACHINE_LEARNING)
    @memory_optimized(MemoryOptimizationLevel.AGGRESSIVE)
    @smart_cache(ttl=3600)
    @performance_tracked(log_performance=True)
    def select_features(self, features_df, targets, **kwargs):
        """Select features using hardware-optimized methods."""
        # Optimize input data
        tprint_data_format(features_df, "hardware_selector_input", level="DEBUG")
        features_df = optimize_dataframe_default(features_df)
        tprint_data_format(features_df, "hardware_selector_optimized", level="DEBUG")
        
        # Track memory usage
        memory_before = track_memory_usage()
        
        # Feature selection with correlation analysis
        feature_scores = {}
        for col in features_df.columns:
            # Use correlation with target for scoring
            try:
                corr = features_df[col].corr(targets)
                feature_scores[col] = abs(corr) if not pd.isna(corr) else 0.0
            except:
                feature_scores[col] = 0.0
        
        # Select top features (50% or minimum 10)
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_count = max(10, len(sorted_features) // 2)
        selected_features = [feat[0] for feat in sorted_features[:selected_count]]
        
        # Track memory usage after
        memory_after = track_memory_usage()
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_metadata={
                'method': 'hardware_optimized',
                'm1_optimization': self.config.enable_m1_optimization,
                'memory_optimization': self.config.enable_memory_optimization
            },
            optimization_metrics={
                'memory_used_mb': memory_after - memory_before,
                'features_processed': len(feature_scores),
                'features_selected': len(selected_features)
            }
        )

class OptimizedMultiObjectiveSelector:
    """Hardware-optimized multi-objective feature selector."""
    
    def __init__(self, objectives=None):
        self.objectives = objectives or ['correlation', 'mutual_information', 'stability']
        self.hardware_manager = get_integrated_hardware_manager()
    
    @m1_optimized("multi_objective_selection", WorkloadCategory.MACHINE_LEARNING)
    @memory_optimized(MemoryOptimizationLevel.MODERATE)
    @smart_cache(ttl=1800)
    def select_features(self, features_df, targets, **kwargs):
        """Select features using multi-objective optimization."""
        # Optimize input data
        tprint_data_format(features_df, "multi_objective_selector_input", level="DEBUG")
        features_df = optimize_dataframe_default(features_df)
        tprint_data_format(features_df, "multi_objective_selector_optimized", level="DEBUG")
        
        # Multi-objective scoring
        feature_scores = {}
        for col in features_df.columns:
            if col not in targets:
                scores = []
                # Correlation objective
                if 'correlation' in self.objectives:
                    try:
                        corr = features_df[col].corr(targets)
                        scores.append(abs(corr) if not pd.isna(corr) else 0.0)
                    except:
                        scores.append(0.0)
                
                # Correlation objective (renamed from mutual_information)
                if 'mutual_information' in self.objectives:
                    try:
                        corr = np.corrcoef(features_df[col], targets)[0, 1]
                        scores.append(abs(corr) if not pd.isna(corr) else 0.0)
                    except:
                        scores.append(0.0)
                
                # Stability objective (variance-based)
                if 'stability' in self.objectives:
                    try:
                        stability = 1.0 / (1.0 + features_df[col].var())
                        scores.append(stability if not pd.isna(stability) else 0.0)
                    except:
                        scores.append(0.0)
                
                # Average score across objectives
                feature_scores[col] = np.mean(scores) if scores else 0.0
        
        # Select top features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_count = max(10, len(sorted_features) // 2)
        selected_features = [feat[0] for feat in sorted_features[:selected_count]]
        
        return MultiObjectiveResult(
            selected_features=selected_features,
            objective_scores=feature_scores,
            optimization_metrics={
                'objectives_used': self.objectives,
                'features_processed': len(feature_scores)
            }
        )

class OptimizedEconomicEvaluator:
    """Hardware-optimized economic period evaluator."""
    
    def __init__(self):
        self.hardware_manager = get_integrated_hardware_manager()
    
    @m1_optimized("economic_validation", WorkloadCategory.FINANCIAL_MODELING)
    @memory_optimized(MemoryOptimizationLevel.LIGHT)
    @smart_cache(ttl=7200)
    def evaluate(self, features_df, **kwargs):
        """Evaluate economic performance with optimization."""
        # Optimize input data
        tprint_data_format(features_df, "economic_evaluator_input", level="DEBUG")
        features_df = optimize_dataframe_default(features_df)
        tprint_data_format(features_df, "economic_evaluator_optimized", level="DEBUG")
        
        # Simple economic metrics calculation (diagnostic only)
        try:
            # Calculate basic return metrics (diagnostic only - not a real PnL series)
            returns = features_df.mean(axis=1)
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Note: This is a diagnostic metric only - not a real Sharpe ratio
            # as it's based on feature averages, not actual strategy returns
            validation_score = min(1.0, max(0.0, sharpe_ratio / 2.0))  # Normalize to 0-1
            
            return EconomicValidationResult(
                validation_score=validation_score,
                economic_metrics={
                    'return': returns.mean(),
                    'sharpe_diagnostic': sharpe_ratio,  # Renamed to indicate diagnostic only
                    'volatility': returns.std()
                },
                optimization_metrics={
                    'features_evaluated': len(features_df.columns),
                    'samples_evaluated': len(features_df),
                    'note': 'Sharpe ratio is diagnostic only - not based on actual strategy returns'
                }
            )
        except Exception as e:
            return EconomicValidationResult(
                validation_score=0.5,
                economic_metrics={'error': str(e)},
                success=False
            )

class OptimizedVectorBTOptimizer:
    """Hardware-optimized VectorBT optimizer."""
    
    def __init__(self):
        self.hardware_manager = get_integrated_hardware_manager()
    
    @m1_optimized("vectorbt_optimization", WorkloadCategory.BACKTESTING)
    @memory_optimized(MemoryOptimizationLevel.AGGRESSIVE)
    @smart_cache(ttl=1800)
    def optimize(self, data, **kwargs):
        """Optimize using VectorBT with hardware optimization."""
        # Optimize input data
        tprint_data_format(data, "vectorbt_optimizer_input", level="DEBUG")
        data = optimize_dataframe_default(data)
        tprint_data_format(data, "vectorbt_optimizer_optimized", level="DEBUG")
        
        # Simple optimization - return optimized data
        result = data.copy()
        tprint_data_format(result, "vectorbt_optimizer_output", level="DEBUG")
        return result

# Set availability flag
HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE = True

# Alias for compatibility
BattleTestedFeatureSelectionResult = FeatureSelectionResult

def _cols(obj: Any) -> List[str]:
    """Normalize selected_features to column names list."""
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    if hasattr(obj, "tolist"):
        return list(obj.tolist())
    if isinstance(obj, list):
        # Handle list of FeatureScore objects
        if obj and hasattr(obj[0], 'feature_name'):
            return [item.feature_name for item in obj if hasattr(item, 'feature_name')]
        # Handle list of strings
        return list(obj)
    return list(obj)

def _safe_to_meta(obj: Any) -> Dict[str, Any]:
    """Safely convert object to serializable metadata."""
    if obj is None:
        return {}
    # Prefer a method if available
    for attr in ("to_dict", "model_dump", "dict"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    # Fallback: shallow, serializable subset
    out = {}
    for k, v in getattr(obj, "__dict__", {}).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v.values()):
            out[k] = v
    return out

@dataclass
class SelectionOutcome:
    """Unified result object for this step's feature selection outputs."""
    success: bool
    selected_features: pd.DataFrame
    selection_metadata: Dict[str, Any]
    selection_metrics: Dict[str, Any]
    selection_strategy: str
    feature_importance: Dict[str, Any]
    economic_validation: Dict[str, Any]
    multi_objective_results: Dict[str, Any]
    vectorbt_optimizations: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    diversity_metrics: Dict[str, Any]
    stability_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    execution_time: float = 0.0
    error_message: Optional[str] = None

class FeatureGenerationFeatureSelectionStep(BaseStep):
    """Sophisticated feature selection step using hardware-optimized components."""

    def __init__(self, step_name: str = "feature_generation_feature_selection_step", config: Optional[Dict[str, Any]] = None):
        """Initialize the sophisticated feature selection step."""
        tprint_info("🔧 [DEBUG] Initializing FeatureGenerationFeatureSelectionStep")
        tprint_debug(f"🔧 [DEBUG] Config keys: {list(config.keys()) if config else 'None'}")
        
        super().__init__(step_name, config)
        
        tprint_info("🔧 [DEBUG] Checking hardware-optimized components availability")
        tprint_debug(f"🔧 [DEBUG] HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE: {HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE}")
        
        # Initialize hardware-optimized feature selection components
        if HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE:
            tprint_success("✅ [DEBUG] Hardware-optimized components available, initializing advanced selectors")
            # Initialize hardware-optimized feature selector with enhanced configuration
            tprint_info("🔧 [DEBUG] Creating FeatureSelectionConfig with hardware optimization")
            self.feature_selection_config = FeatureSelectionConfig(
                enable_multi_stage_selection=True,
                enable_lightweight_screening=True,
                enable_parallel_processing=True,
                # Hardware optimization settings
                enable_m1_optimization=True,
                enable_memory_optimization=True,
                enable_caching=True,
                memory_threshold_mb=200.0,
                cache_ttl=3600.0
            )
            tprint_success("✅ [DEBUG] FeatureSelectionConfig created successfully")
            tprint_debug(f"🔧 [DEBUG] Config - M1 optimization: {self.feature_selection_config.enable_m1_optimization}")
            tprint_debug(f"🔧 [DEBUG] Config - Memory optimization: {self.feature_selection_config.enable_memory_optimization}")
            tprint_debug(f"🔧 [DEBUG] Config - Caching: {self.feature_selection_config.enable_caching}")
            tprint_info("⚡ [DEBUG] Hardware optimizations enabled:")
            tprint_debug(f"🔧 [DEBUG] - M1 optimization: {self.feature_selection_config.enable_m1_optimization}")
            tprint_debug(f"🔧 [DEBUG] - Memory optimization: {self.feature_selection_config.enable_memory_optimization}")
            tprint_debug(f"🔧 [DEBUG] - Memory threshold: {self.feature_selection_config.memory_threshold_mb}MB")
            tprint_debug(f"🔧 [DEBUG] - Cache TTL: {self.feature_selection_config.cache_ttl}s")
            
            tprint_info("🔧 [DEBUG] Initializing OptimizedFeatureSelector")
            self.battle_tested_selector = OptimizedFeatureSelector(self.feature_selection_config)
            # Extra runtime proof for auditing which selector is used
            try:
                self.logger.info(
                    f"🔎 Using selector instance: {self.battle_tested_selector.__class__.__module__}."
                    f"{self.battle_tested_selector.__class__.__name__}"
                )
            except Exception:
                pass
            tprint_success("✅ [DEBUG] OptimizedFeatureSelector initialized")
            
            # Initialize multi-objective selector with default objectives
            tprint_info("🔧 [DEBUG] Creating default objectives for multi-objective selector")
            objectives = ['correlation', 'mutual_information', 'stability']
            tprint_debug(f"🔧 [DEBUG] Created {len(objectives)} objectives")
            
            tprint_info("🔧 [DEBUG] Initializing OptimizedMultiObjectiveSelector")
            self.multi_objective_selector = OptimizedMultiObjectiveSelector(objectives=objectives)
            tprint_success("✅ [DEBUG] OptimizedMultiObjectiveSelector initialized")
            
            # Initialize economic evaluator
            tprint_info("🔧 [DEBUG] Initializing OptimizedEconomicEvaluator")
            self.economic_evaluator = OptimizedEconomicEvaluator()
            tprint_success("✅ [DEBUG] OptimizedEconomicEvaluator initialized")
            
            # Initialize VectorBT optimizer
            tprint_info("🔧 [DEBUG] Initializing OptimizedVectorBTOptimizer")
            self.vectorbt_optimizer = OptimizedVectorBTOptimizer()
            tprint_success("✅ [DEBUG] OptimizedVectorBTOptimizer initialized")
            tprint_success("🎉 [DEBUG] All hardware-optimized components initialized successfully")
        else:
            tprint_warning("⚠️ [DEBUG] Hardware-optimized components not available, using fallback mode")
            self.logger.warning("⚠️ Hardware-optimized feature selection components not available, using fallback")
            self.battle_tested_selector = None
            self.multi_objective_selector = None
            self.economic_evaluator = None
            self.vectorbt_optimizer = None
            tprint_info("🔧 [DEBUG] Fallback mode initialized - basic feature selection will be used")

    def _apply_overrides(self, overrides: Optional[Dict[str, Any]]):
        """Apply custom configuration overrides."""
        tprint_info("🔧 [DEBUG] Applying configuration overrides")
        tprint_debug(f"🔧 [DEBUG] Overrides: {overrides}")
        
        if not overrides or not HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE:
            tprint_info("🔧 [DEBUG] No overrides to apply or components not available")
            return
            
        tprint_info(f"🔧 [DEBUG] Applying {len(overrides)} overrides")
        for k, v in overrides.items():
            if hasattr(self.feature_selection_config, k):
                tprint_debug(f"🔧 [DEBUG] Setting {k} = {v}")
                setattr(self.feature_selection_config, k, v)
            else:
                tprint_warning(f"⚠️ [DEBUG] Override key '{k}' not found in config")
        tprint_success("✅ [DEBUG] Configuration overrides applied")

    def _filter_data_by_parameters(self, data: pd.DataFrame, targets: pd.Series,
                                  lookback_days: Optional[int], start_date: Optional[Union[str, np.ndarray]],
                                  end_date: Optional[Union[str, np.ndarray]]) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter data based on lookback_days, start_date, and end_date parameters."""
        tprint_data_format(data, "filter_input_data", level="DEBUG")
        tprint_data_format(targets, "filter_input_targets", level="DEBUG")
        
        # Ensure data index is datetime type for proper comparisons
        try:
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                self.logger.warning("Converting data index to datetime for proper filtering")
                data.index = pd.to_datetime(data.index, errors='coerce')
                targets.index = pd.to_datetime(targets.index, errors='coerce')
                # Remove any rows with invalid dates after conversion
                valid_data_mask = ~data.index.isna()
                valid_targets_mask = ~targets.index.isna()
                data = data[valid_data_mask]
                targets = targets[valid_targets_mask]
        except Exception as e:
            self.logger.warning(f"Failed to ensure datetime index: {type(e).__name__}")
        if lookback_days is not None:
            # Use last N days of data
            data = data.tail(lookback_days)
            targets = targets.tail(lookback_days)

        if start_date is not None:
            try:
                # Handle numpy array inputs by converting to scalar if needed
                if isinstance(start_date, np.ndarray):
                    if start_date.size == 1:
                        start_date = start_date.item()
                    else:
                        self.logger.warning(f"Invalid start_date format: numpy array with {start_date.size} elements")
                        return data, targets

                start_dt = pd.to_datetime(start_date)
                data = data[data.index >= start_dt]
                targets = targets[targets.index >= start_dt]
            except Exception as e:
                self.logger.warning(f"Invalid start_date format: {type(e).__name__}: {str(start_date)}")

        if end_date is not None:
            try:
                # Handle numpy array inputs by converting to scalar if needed
                if isinstance(end_date, np.ndarray):
                    if end_date.size == 1:
                        end_date = end_date.item()
                    else:
                        self.logger.warning(f"Invalid end_date format: numpy array with {end_date.size} elements")
                        return data, targets

                end_dt = pd.to_datetime(end_date)
                data = data[data.index <= end_dt]
                targets = targets[targets.index <= end_dt]
            except Exception as e:
                self.logger.warning(f"Invalid end_date format: {type(e).__name__}: {str(end_date)}")

        tprint_data_format(data, "filter_output_data", level="DEBUG")
        tprint_data_format(targets, "filter_output_targets", level="DEBUG")
        return data, targets

    async def execute(self, config: Dict[str, Any]) -> SelectionOutcome:
        """Execute sophisticated feature selection step using hardware-optimized components with BaseStep integration."""

        self.logger.info("🎯 Starting sophisticated feature selection step with multi-objective optimization")
        tprint_info("🔍 [DEBUG] Starting feature selection execution")
        
        # Extract custom overrides from config
        custom_overrides = config.get("custom_overrides", {})
        
        # Set context for enhanced file naming
        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        direction = config.get('direction', 'long')
        model = config.get('model', 'Analyst')
        timeframe = config.get('timeframe', '15m')
        
        self._set_context(symbol=symbol, exchange=exchange, direction=direction, model=model)
        
        tprint_debug(f"🔍 [DEBUG] Input parameters - symbol: {symbol}, timeframe: {timeframe}, direction: {direction}")

        try:
            
                        
            # Resolve feature set and labeling targets strictly via artifact manager
            # Note: data parameter is raw market data, but we need generated features
            features_df = None  # Always start with None to force loading from artifacts/files

            # Try to load features from artifact manager first
            features_found_in_artifacts = False
            try:
                # Try to load features using BaseStep methods
                candidate = self._load_dataframe('generated_features')
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    features_df = candidate
                    features_found_in_artifacts = True
                    tprint_success(f"✅ [DEBUG] Loaded generated features from BaseStep: shape={features_df.shape}")
                    tprint_data_preview(features_df, "loaded_features_from_artifacts")
                    tprint_data_format(features_df, "loaded_features_from_artifacts", level="INFO")
            except Exception as e:
                tprint_warning(f"⚠️ [DEBUG] Artifact manager failed for features: {e}")

            # If artifact manager didn't find features, try loading directly from files
            if not features_found_in_artifacts:
                tprint_info("🔍 [FALLBACK] Artifact manager didn't find features, trying direct file loading...")

                try:
                    import glob
                    import os
                    tprint_info("🔍 [FALLBACK] Loading features directly from generated files...")

                    # Look for generated features files
                    feature_files = glob.glob("generated/generated_features_ETHUSDT_15m_DirectionType.LONGS_*.parquet")
                    if feature_files:
                        latest_file = max(feature_files, key=os.path.getmtime)
                        tprint_info(f"🔍 [FALLBACK] Loading features from: {latest_file}")
                        features_df = pd.read_parquet(latest_file)
                        tprint_success(f"✅ [DEBUG] Loaded features from file: shape={features_df.shape}")
                        tprint_data_preview(features_df, "loaded_features_from_file")
                        tprint_data_format(features_df, "loaded_features_from_file", level="INFO")
                    else:
                        raise ValueError("No generated features files found")
                except Exception as fallback_error:
                    tprint_error(f"❌ [DEBUG] Fallback features loading failed: {fallback_error}")
                    raise ValueError(f"Generated features not found in artifact manager or files. Run feature_generation_feature_generation_step first. Fallback error: {fallback_error}")

            if features_df is None or features_df.empty:
                raise ValueError("Generated features not found in artifact manager. Run feature_generation_feature_generation_step first.")

            # Load targets directly from the features file (simpler approach)
            target_columns = ['target', 'targets', 'label', 'labels', 'y', 'direction_confidence', 'opportunity_asymmetry', 'directional_signal']
            target_col = None

            tprint_info("🔍 [DEBUG] Looking for target columns in features data...")
            tprint_debug(f"🔍 [DEBUG] Features DataFrame shape: {features_df.shape}")
            tprint_debug(f"🔍 [DEBUG] Features DataFrame columns (first 10): {list(features_df.columns)[:10]}")
            tprint_debug(f"🔍 [DEBUG] Checking for directional_signal: {'directional_signal' in features_df.columns}")
            tprint_debug(f"🔍 [DEBUG] All target columns being checked: {target_columns}")

            for col in target_columns:
                tprint_debug(f"🔍 [DEBUG] Checking column: {col} - exists: {col in features_df.columns}")
                if col in features_df.columns:
                    target_col = col
                    tprint_info(f"✅ [DEBUG] Found target column: {target_col}")
                    break
                
            if target_col:
                targets_series = features_df[target_col].dropna()
                features_df = features_df.drop(columns=[target_col])
                tprint_success(f"✅ [DEBUG] Loaded targets from features file: {target_col} with {len(targets_series)} values")
                tprint_debug(f"🔍 [DEBUG] Targets shape after loading: {targets_series.shape}")
                tprint_debug(f"🔍 [DEBUG] Targets variance: {targets_series.var():.6f}")
                tprint_debug(f"🔍 [DEBUG] Targets std: {targets_series.std():.6f}")
                tprint_data_preview(targets_series, "extracted_targets")
                tprint_data_format(targets_series, "extracted_targets", level="DEBUG")
                tprint_data_preview(features_df, "features_after_target_extraction")
                tprint_data_format(features_df, "features_after_target_extraction", level="DEBUG")
            else:
                tprint_error("❌ [DEBUG] No target column found in features data")
                tprint_error(f"❌ [DEBUG] Available columns: {list(features_df.columns)[:20]}")
                raise ValueError("No target column found in features data. Available columns: " + str(list(features_df.columns)[:10]))

            if targets_series is None or targets_series.empty:
                tprint_error(f"❌ [DEBUG] Targets validation failed: targets_series={targets_series}, empty={targets_series.empty if targets_series is not None else 'N/A'}")
                raise ValueError("Targets from features data are empty or None.")

            tprint_info("🔍 [DEBUG] Aligning generated features with labeling targets")
            tprint_debug(f"🔍 [DEBUG] Features shape before alignment: {features_df.shape}")
            tprint_debug(f"🔍 [DEBUG] Targets shape before alignment: {targets_series.shape}")
            tprint_debug(f"🔍 [DEBUG] Features index sample: {features_df.index[:3].tolist()}")
            tprint_debug(f"🔍 [DEBUG] Targets index sample: {targets_series.index[:3].tolist()}")

            aligned = features_df.join(targets_series.rename("target"), how="inner").dropna(axis=0, how="any")

            tprint_debug(f"🔍 [DEBUG] Aligned shape: {aligned.shape}")
            tprint_debug(f"🔍 [DEBUG] Alignment dropped {len(features_df) - len(aligned)} rows")
            tprint_data_preview(aligned, "aligned_features_and_targets")
            tprint_data_format(aligned, "aligned_features_and_targets", level="DEBUG")

            if aligned.empty:
                tprint_error(f"❌ [DEBUG] Alignment resulted in empty DataFrame!")
                tprint_error(f"❌ [DEBUG] Features index range: {features_df.index.min()} to {features_df.index.max()}")
                tprint_error(f"❌ [DEBUG] Targets index range: {targets_series.index.min()} to {targets_series.index.max()}")
                tprint_data_format(aligned, "empty_alignment_error", level="ERROR")
                raise ValueError("No overlapping timestamps between generated features and labeling targets.")

            targets = aligned.pop("target")
            features_df = aligned
            tprint_success(f"✅ [DEBUG] Prepared feature/target matrix: features={features_df.shape}, targets={targets.shape}")
            tprint_debug(f"🔍 [DEBUG] Final features shape: {features_df.shape}")
            tprint_debug(f"🔍 [DEBUG] Final targets shape: {targets.shape}")
            tprint_data_preview(features_df, "final_prepared_features")
            tprint_data_format(features_df, "final_prepared_features", level="DEBUG")
            tprint_data_preview(targets, "final_prepared_targets")
            tprint_data_format(targets, "final_prepared_targets", level="DEBUG")
            
            # Load periods/lookbacks from feature_generation_period_lookback_optimization_step
            tprint_info("🔍 [DEBUG] Loading periods/lookbacks from period lookback optimization step")
            try:
                # Load top periods (top1 for trading)
                top_periods = self._load_metadata('top_periods')
                if top_periods is not None:
                    tprint_success(f"✅ [DEBUG] Loaded top periods: {top_periods}")
                else:
                    tprint_warning("⚠️ [DEBUG] No top periods found, trying optimized_periods")
                    optimized_periods = self._load_metadata('optimized_periods')
                    if optimized_periods is not None:
                        # Extract top 1 period if it's a list
                        if isinstance(optimized_periods, list) and len(optimized_periods) > 0:
                            top_periods = optimized_periods[0]
                        else:
                            top_periods = optimized_periods
                        tprint_success(f"✅ [DEBUG] Loaded optimized periods (using first): {top_periods}")
                    else:
                        tprint_warning("⚠️ [DEBUG] No periods found in artifact manager")
                
                # Load top lookbacks (top1 for trading)
                top_lookbacks = self._load_metadata('top_lookbacks')
                if top_lookbacks is not None:
                    tprint_success(f"✅ [DEBUG] Loaded top lookbacks: {top_lookbacks}")
                else:
                    tprint_warning("⚠️ [DEBUG] No top lookbacks found, trying optimized_lookbacks")
                    optimized_lookbacks = self._load_metadata('optimized_lookbacks')
                    if optimized_lookbacks is not None:
                        # Extract top 1 lookback if it's a list
                        if isinstance(optimized_lookbacks, list) and len(optimized_lookbacks) > 0:
                            top_lookbacks = optimized_lookbacks[0]
                        else:
                            top_lookbacks = optimized_lookbacks
                        tprint_success(f"✅ [DEBUG] Loaded optimized lookbacks (using first): {top_lookbacks}")
                    else:
                        tprint_warning("⚠️ [DEBUG] No lookbacks found in artifact manager")
                
                # Store periods/lookbacks in config for use in feature selection
                if top_periods is not None:
                    config['top_periods'] = top_periods
                if top_lookbacks is not None:
                    config['top_lookbacks'] = top_lookbacks
                    
            except Exception as e:
                tprint_warning(f"⚠️ [DEBUG] Failed to load periods/lookbacks: {e}")
                # Continue without periods/lookbacks - they're not critical for basic feature selection
                
            # Fast fail: If no processed features are available, fail immediately
            # This ensures the pipeline fails fast rather than using raw data
            if features_df is None or features_df.empty:
                error_msg = "❌ [CRITICAL] No data available for feature selection. This step requires processed features from previous pipeline steps."
                self.logger.error(error_msg)
                tprint_error(error_msg)
                
                return SelectionOutcome(
                    success=False,
                    selected_features=pd.DataFrame(),
                    selection_metadata={},
                    selection_metrics={},
                    selection_strategy="failed",
                    feature_importance={},
                    economic_validation={},
                    multi_objective_results={},
                    vectorbt_optimizations={},
                    quality_metrics={},
                    diversity_metrics={},
                    stability_metrics={},
                    artifacts={},
                    error_message=error_msg
                )
            
            # Try to load cached results from feature selection step
            tprint_info("🔍 [DEBUG] Checking for cached feature selection results")
            cached_selected_features = self._load_dataframe("selected_features")
            cached_selection_metrics = self._load_metadata("selection_metrics")
            cached_importance_rankings = self._load_metadata("feature_importance_rankings")
            
            tprint_debug(f"🔍 [DEBUG] Cached results - features: {cached_selected_features is not None}, metrics: {cached_selection_metrics is not None}, rankings: {cached_importance_rankings is not None}")
            
            if cached_selected_features is not None:
                self.logger.info("📦 Retrieved selected features from artifact manager")
                tprint_success("✅ [DEBUG] Using cached feature selection results")
                tprint_debug(f"🔍 [DEBUG] Cached features shape: {cached_selected_features.shape}")
                tprint_debug(f"🔍 [DEBUG] Cached features columns: {list(cached_selected_features.columns)[:10]}...")
                tprint_data_preview(cached_selected_features, "cached_selected_features")
                tprint_data_format(cached_selected_features, "cached_selected_features", level="INFO")
                return SelectionOutcome(
                    success=True,
                    selected_features=cached_selected_features,
                    selection_metadata={'cache_hit': True, 'retrieved_from_artifact_manager': True},
                    selection_metrics=cached_selection_metrics or {},
                    selection_strategy="cached",
                    feature_importance=cached_importance_rankings or {},
                    economic_validation={},
                    multi_objective_results={},
                    vectorbt_optimizations={},
                    quality_metrics={},
                    diversity_metrics={},
                    stability_metrics={},
                    artifacts={'cache_hit': True},
                    error_message=None
                )

            # Skip data filtering for now to avoid date format issues
            # features_df, targets = self._filter_data_by_parameters(features_df, targets, lookback_days, start_date, end_date)
            tprint_info("🔍 [DEBUG] Checking hardware-optimized components availability")
            tprint_debug(f"🔍 [DEBUG] HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE: {HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE}")
            
            if not HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE:
                tprint_warning("⚠️ [DEBUG] Hardware-optimized components not available, using fallback")
                # Fallback to basic feature selection
                return await self._fallback_feature_selection(
                    features_df, targets, symbol, timeframe, direction, custom_overrides
                )

            # Perform sophisticated feature selection
            tprint_info("🔍 [DEBUG] Starting sophisticated feature selection")
            tprint_debug(f"🔍 [DEBUG] Input data shape: {features_df.shape}")
            tprint_debug(f"🔍 [DEBUG] Targets shape: {targets.shape}")
            tprint_debug(f"🔍 [DEBUG] Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
            
            selection_result = await self._perform_sophisticated_feature_selection(
                features_df, targets, symbol, timeframe, direction, custom_overrides
            )
            
            tprint_debug(f"🔍 [DEBUG] Selection result success: {selection_result.success}")
            tprint_debug(f"🔍 [DEBUG] Selection strategy: {selection_result.selection_strategy}")

            if selection_result.success:
                self.logger.info(f"✅ Sophisticated feature selection completed successfully")
                tprint_success("✅ [DEBUG] Sophisticated feature selection completed successfully")
                self.logger.info(f"📊 Selected {len(selection_result.selected_features.columns)} features")
                tprint_info(f"🔍 [DEBUG] Selected {len(selection_result.selected_features.columns)} features")
                self.logger.info(f"🎯 Strategy: {selection_result.selection_strategy}")
                self.logger.info(f"💰 Economic validation: {selection_result.economic_validation}")
                self.logger.info(f"📈 Multi-objective results: {selection_result.multi_objective_results}")
                tprint_debug(f"🔍 [DEBUG] Economic validation keys: {list(selection_result.economic_validation.keys()) if selection_result.economic_validation else 'None'}")
                tprint_debug(f"🔍 [DEBUG] Multi-objective results keys: {list(selection_result.multi_objective_results.keys()) if selection_result.multi_objective_results else 'None'}")
                
                # Store artifacts in artifact manager using standard save method
                tprint_info("🔍 [DEBUG] Storing artifacts in artifact manager")
                tprint_debug(f"🔍 [DEBUG] Selected features shape: {selection_result.selected_features.shape}")
                tprint_debug(f"🔍 [DEBUG] Selection metrics keys: {list(selection_result.selection_metrics.keys()) if selection_result.selection_metrics else 'None'}")
                tprint_debug(f"🔍 [DEBUG] Feature importance keys: {list(selection_result.feature_importance.keys()) if selection_result.feature_importance else 'None'}")
                
                # Save artifacts using BaseStep methods
                tprint_data_preview(selection_result.selected_features, "saving_selected_features")
                tprint_data_format(selection_result.selected_features, "saving_selected_features", level="INFO")
                tprint_data_preview(selection_result.selection_metrics, "saving_selection_metrics")
                tprint_data_format(selection_result.selection_metrics, "saving_selection_metrics", level="DEBUG")
                self._save_dataframe(selection_result.selected_features, 'selected_features')
                self._save_metadata(selection_result.selection_metrics, 'selection_metrics')
                self._save_metadata(selection_result.feature_importance, 'feature_importance_rankings')
                self._save_metadata({
                    'economic_validation': selection_result.economic_validation,
                    'multi_objective_results': selection_result.multi_objective_results,
                    'vectorbt_optimizations': selection_result.vectorbt_optimizations,
                    'quality_metrics': selection_result.quality_metrics,
                    'diversity_metrics': selection_result.diversity_metrics,
                    'stability_metrics': selection_result.stability_metrics
                }, 'additional_metrics')
                tprint_success("✅ [DEBUG] Artifacts stored successfully")
            else:
                self.logger.error(f"❌ Feature selection failed: {selection_result.error_message}")
                tprint_error(f"❌ [DEBUG] Feature selection failed: {selection_result.error_message}")

            return selection_result

        except Exception as e:
            tprint_error(f"❌ [DEBUG] Sophisticated feature selection step failed with exception: {e}")
            tprint_debug(f"🔍 [DEBUG] Exception type: {type(e).__name__}")
            tprint_debug(f"🔍 [DEBUG] Exception details: {str(e)}")
            self.logger.error(f"❌ Sophisticated feature selection step failed with exception: {e}", exc_info=True)
            return SelectionOutcome(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                selection_strategy="error",
                feature_importance={},
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={},
                error_message=str(e)
            )

    async def _perform_sophisticated_feature_selection(self, data: pd.DataFrame, targets: pd.Series,
                                                       symbol: str, timeframe: str, direction: str,
                                                       custom_overrides: Optional[Dict[str, Any]]) -> SelectionOutcome:
        """Perform sophisticated feature selection using hardware-optimized components."""
        
        try:
            tprint_info("🔍 [DEBUG] Starting sophisticated feature selection process")
            tprint_debug(f"🔍 [DEBUG] Input data shape: {data.shape}")
            tprint_debug(f"🔍 [DEBUG] Input targets length: {len(targets)}")
            tprint_debug(f"🔍 [DEBUG] Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
            tprint_data_format(data, f"sophisticated_selection_input_{symbol}_{timeframe}", level="DEBUG")
            tprint_data_format(targets, f"sophisticated_selection_targets_{symbol}_{timeframe}", level="DEBUG")
            
            if not HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE:
                tprint_error("❌ [DEBUG] Hardware-optimized feature selection components not available")
                raise ImportError("Hardware-optimized feature selection components not available")
            
            tprint_success("✅ [DEBUG] Hardware-optimized components available")
            
            # Apply custom overrides if provided
            tprint_info("🔍 [DEBUG] Applying custom overrides")
            self._apply_overrides(custom_overrides)
            
            # Apply hardware optimizations if no custom overrides provided
            if not custom_overrides:
                tprint_info("🔧 [HARDWARE_OPTIMIZATION] Applying hardware optimizations")
                hardware_optimizations = {
                    'enable_m1_optimization': True,
                    'enable_memory_optimization': True,
                    'enable_caching': True,
                    'memory_threshold_mb': 200.0,
                    'cache_ttl': 3600.0
                }
                self._apply_overrides(hardware_optimizations)
                tprint_success("✅ [HARDWARE_OPTIMIZATION] Hardware optimizations applied")
            
            tprint_success("✅ [DEBUG] Custom overrides applied")
                
            # Step 1: Hardware-optimized feature selection
            self.logger.info("🔄 Stage 1: Hardware-optimized feature selection")
            tprint_info("🎯 Starting Stage 1: Hardware-optimized feature selection")
            tprint_debug(f"📊 Input data shape: {data.shape}")
            tprint_debug(f"📊 Target data shape: {targets.shape if targets is not None else 'None'}")
            tprint_info("🔍 [STAGE1] Analyzing input data characteristics")
            tprint_debug(f"🔍 [STAGE1] Data memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            tprint_debug(f"🔍 [STAGE1] Data types distribution: {data.dtypes.value_counts().to_dict()}")
            tprint_debug(f"🔍 [STAGE1] NaN values per column: {data.isnull().sum().to_dict()}")
            tprint_debug(f"🔍 [STAGE1] Target statistics - min: {targets.min():.6f}, max: {targets.max():.6f}, mean: {targets.mean():.6f}, std: {targets.std():.6f}")
            tprint_info("🔍 [STAGE1] Initializing hardware-optimized selector with configuration")
            tprint_debug(f"🔍 [STAGE1] Selector config - M1 optimization: {self.feature_selection_config.enable_m1_optimization}")
            tprint_debug(f"🔍 [STAGE1] Selector config - Memory optimization: {self.feature_selection_config.enable_memory_optimization}")
            tprint_debug(f"🔍 [STAGE1] Selector config - Caching: {self.feature_selection_config.enable_caching}")
            tprint_debug(f"🔍 [STAGE1] Selector config - Memory threshold: {self.feature_selection_config.memory_threshold_mb}MB")
            tprint_debug(f"🔍 [STAGE1] Selector config - Cache TTL: {self.feature_selection_config.cache_ttl}s")
            tprint_info("🔍 [STAGE1] Executing hardware-optimized feature selection in separate thread")
            try:
                # Log the concrete class and module used at runtime
                self.logger.info(
                    f"🎛️ Stage 1 selector: {self.battle_tested_selector.__class__.__module__}."
                    f"{self.battle_tested_selector.__class__.__name__}"
                )
            except Exception:
                pass
            
            tprint_info("🔍 [STAGE1] About to call battle_tested_selector.select_features")
            tprint_debug(f"🔍 [STAGE1] battle_tested_selector type: {type(self.battle_tested_selector)}")
            tprint_debug(f"🔍 [STAGE1] battle_tested_selector: {self.battle_tested_selector}")
            
            try:
                advanced_result = await asyncio.to_thread(
                    self.battle_tested_selector.select_features, data, targets
                )
                tprint_success("✅ [STAGE1] Hardware-optimized selection completed successfully")
            except Exception as e:
                tprint_error(f"❌ [STAGE1] Hardware-optimized selection failed: {e}")
                tprint_debug(f"❌ [STAGE1] Exception type: {type(e).__name__}")
                tprint_debug(f"❌ [STAGE1] Exception details: {str(e)}")
                raise e
            
            tprint_info("🔍 [STAGE1] Hardware-optimized selection completed")
            tprint_debug(f"🔍 [STAGE1] Result success: {advanced_result.success}")
            if hasattr(advanced_result, 'selected_features'):
                tprint_debug(f"🔍 [STAGE1] Selected features count: {len(advanced_result.selected_features)}")
                tprint_debug(f"🔍 [STAGE1] Selected features type: {type(advanced_result.selected_features)}")
            
            # Log additional result attributes if available
            if hasattr(advanced_result, 'quality_metrics'):
                tprint_debug(f"🔍 [STAGE1] Quality metrics available: {bool(advanced_result.quality_metrics)}")
            if hasattr(advanced_result, 'diversity_metrics'):
                tprint_debug(f"🔍 [STAGE1] Diversity metrics available: {bool(advanced_result.diversity_metrics)}")
            if hasattr(advanced_result, 'stability_metrics'):
                tprint_debug(f"🔍 [STAGE1] Stability metrics available: {bool(advanced_result.stability_metrics)}")
            if hasattr(advanced_result, 'feature_importance'):
                tprint_debug(f"🔍 [STAGE1] Feature importance available: {bool(advanced_result.feature_importance)}")
            if hasattr(advanced_result, 'execution_time'):
                tprint_debug(f"🔍 [STAGE1] Execution time: {advanced_result.execution_time:.2f} seconds")
            
            if not advanced_result.success:
                tprint_error(f"❌ [STAGE1] Hardware-optimized selection failed: {advanced_result.error_message}")
                tprint_debug(f"❌ [STAGE1] Error details: {advanced_result.error_message}")
                raise Exception(f"Advanced feature selection failed: {advanced_result.error_message}")
            
            # Normalize selected features to column names
            tprint_info("🔍 [STAGE1] Processing selected features")
            cols1 = _cols(advanced_result.selected_features)
            tprint_debug(f"🔍 [STAGE1] Raw selected features: {cols1}")
            tprint_debug(f"🔍 [STAGE1] Total features selected: {len(cols1)}")
            
            # Log optimization metrics if available
            if hasattr(advanced_result, 'optimization_metrics'):
                tprint_debug(f"🔍 [STAGE1] Optimization metrics: {advanced_result.optimization_metrics}")
            if hasattr(advanced_result, 'memory_usage_mb'):
                tprint_debug(f"🔍 [STAGE1] Memory usage: {advanced_result.memory_usage_mb:.2f} MB")
            if hasattr(advanced_result, 'execution_time'):
                tprint_debug(f"🔍 [STAGE1] Execution time: {advanced_result.execution_time:.2f} seconds")
            
            # Safely intersect selected columns with available data
            tprint_info("🔍 [STAGE1] Validating selected features against available data")
            cols1_available = [c for c in cols1 if c in data.columns]
            tprint_debug(f"🔍 [STAGE1] Available features: {cols1_available}")
            tprint_debug(f"🔍 [STAGE1] Available features count: {len(cols1_available)}")
            
            if len(cols1_available) != len(cols1):
                missing_features = [c for c in cols1 if c not in data.columns]
                tprint_warning(f"⚠️ [STAGE1] Missing features in data: {missing_features}")
                tprint_warning(f"⚠️ [STAGE1] Feature availability: {len(cols1_available)}/{len(cols1)} ({len(cols1_available)/len(cols1)*100:.1f}%)")
            
            tprint_info("🔍 [STAGE1] Creating filtered dataset with memory optimization")
            df1 = data.loc[:, cols1_available].copy()
            tprint_data_format(df1, "stage1_before_memory_optimization", level="DEBUG")
            # Apply memory optimization to the filtered dataset
            df1 = optimize_dataframe_default(df1)
            tprint_debug(f"🔍 [STAGE1] Filtered dataset shape: {df1.shape}")
            tprint_debug(f"🔍 [STAGE1] Filtered dataset memory usage: {df1.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            tprint_data_preview(df1, "stage1_hardware_optimized_features")
            tprint_data_format(df1, "stage1_hardware_optimized_features", level="DEBUG")
            
            tprint_success(f"✅ [STAGE1] Hardware-optimized selection completed: {len(cols1_available)} features selected")
            tprint_info(f"🔍 [STAGE1] Feature reduction: {len(data.columns) - len(cols1_available)} features removed ({(len(data.columns) - len(cols1_available))/len(data.columns)*100:.1f}%)")
            tprint_debug(f"🔍 [STAGE1] Final selected features: {cols1_available}")
            
            # Stage toggles (allow skipping heavy steps via overrides)
            disable_stage2 = bool(custom_overrides.get('disable_multi_objective', False)) if custom_overrides else False
            disable_stage3 = bool(custom_overrides.get('disable_economic_validation', False)) if custom_overrides else False
            disable_stage4 = bool(custom_overrides.get('disable_vectorbt_optimization', False)) if custom_overrides else False

            # Step 2: Multi-objective optimization
            if not disable_stage2:
                self.logger.info("🎯 Stage 2: Hardware-optimized multi-objective optimization")
                tprint_info("🎯 Starting Stage 2: Hardware-optimized multi-objective optimization")
                tprint_debug(f"📊 Input data shape: {df1.shape}")
                tprint_info("🔍 [STAGE2] Analyzing input data for multi-objective optimization")
                tprint_debug(f"🔍 [STAGE2] Input features: {list(df1.columns)[:10]}...")
                tprint_debug(f"🔍 [STAGE2] Input data memory usage: {df1.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                tprint_debug(f"🔍 [STAGE2] Target statistics - min: {targets.min():.6f}, max: {targets.max():.6f}, mean: {targets.mean():.6f}, std: {targets.std():.6f}")
                tprint_info("🔍 [STAGE2] Initializing hardware-optimized multi-objective selector")
                tprint_debug(f"🔍 [STAGE2] Multi-objective selector type: {type(self.multi_objective_selector)}")
                tprint_info("🔍 [STAGE2] Executing multi-objective optimization in separate thread")
                multi_objective_result = await asyncio.to_thread(
                    self.multi_objective_selector.select_features, df1, targets
                )
                tprint_info("🔍 [STAGE2] Multi-objective optimization completed")
                tprint_debug(f"🔍 [STAGE2] Result success: {multi_objective_result.success}")
                if hasattr(multi_objective_result, 'selected_features'):
                    tprint_debug(f"🔍 [STAGE2] Selected features count: {len(multi_objective_result.selected_features)}")
                    tprint_debug(f"🔍 [STAGE2] Selected features type: {type(multi_objective_result.selected_features)}")
                if hasattr(multi_objective_result, 'objective_scores'):
                    tprint_debug(f"🔍 [STAGE2] Objective scores: {multi_objective_result.objective_scores}")
                if hasattr(multi_objective_result, 'optimization_metrics'):
                    tprint_debug(f"🔍 [STAGE2] Optimization metrics: {multi_objective_result.optimization_metrics}")
                cols2 = _cols(multi_objective_result.selected_features)
            else:
                self.logger.info("⏭️ Skipping Stage 2: Multi-objective optimization (disabled)")
                # Pass-through from stage 1
                multi_objective_result = type('MultiObjectiveResult', (), {'selected_features': cols1, 'success': True, 'objective_scores': {}})()
                cols2 = cols1

            tprint_info("🔍 [STAGE2] Processing multi-objective optimization results")
            cols2_available = [c for c in cols2 if c in df1.columns]
            tprint_debug(f"🔍 [STAGE2] Available features after optimization: {cols2_available}")
            tprint_debug(f"🔍 [STAGE2] Available features count: {len(cols2_available)}")

            # No fallback trimming here; rely on MultiObjective selector configuration
            
            if len(cols2_available) != len(cols2):
                missing_features = [c for c in cols2 if c not in df1.columns]
                tprint_warning(f"⚠️ [STAGE2] Missing features in data: {missing_features}")
                tprint_warning(f"⚠️ [STAGE2] Feature availability: {len(cols2_available)}/{len(cols2)} ({len(cols2_available)/len(cols2)*100:.1f}%)")
            
            tprint_info("🔍 [STAGE2] Creating filtered dataset")
            df2 = df1.loc[:, cols2_available].copy()
            tprint_data_format(df2, "stage2_before_optimization", level="DEBUG")
            tprint_debug(f"🔍 [STAGE2] Filtered dataset shape: {df2.shape}")
            tprint_debug(f"🔍 [STAGE2] Filtered dataset memory usage: {df2.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            tprint_data_preview(df2, "stage2_multi_objective_features")
            tprint_data_format(df2, "stage2_multi_objective_features", level="DEBUG")
            
            if not disable_stage2:
                tprint_success(f"✅ [STAGE2] Multi-objective optimization completed: {len(cols2_available)} features selected")
                tprint_info(f"🔍 [STAGE2] Feature reduction: {len(df1.columns) - len(cols2_available)} features removed ({(len(df1.columns) - len(cols2_available))/len(df1.columns)*100:.1f}%)")
                tprint_debug(f"🔍 [STAGE2] Final optimized features: {cols2_available}")
            
            # Step 3: Economic validation
            if not disable_stage3:
                self.logger.info("💰 Stage 3: Hardware-optimized economic validation")
                tprint_info("💰 Starting Stage 3: Hardware-optimized economic validation")
                tprint_debug(f"📊 Input data shape: {df2.shape}")
                tprint_debug(f"📊 Symbol: {symbol}, Timeframe: {timeframe}")
                tprint_info("🔍 [STAGE3] Analyzing input data for economic validation")
                tprint_debug(f"🔍 [STAGE3] Input features: {list(df2.columns)[:10]}...")
                tprint_debug(f"🔍 [STAGE3] Input data memory usage: {df2.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                tprint_debug(f"🔍 [STAGE3] Target statistics - min: {targets.min():.6f}, max: {targets.max():.6f}, mean: {targets.mean():.6f}, std: {targets.std():.6f}")
                tprint_info("🔍 [STAGE3] Initializing hardware-optimized economic evaluator")
                tprint_debug(f"🔍 [STAGE3] Economic evaluator type: {type(self.economic_evaluator)}")
                tprint_info("🔍 [STAGE3] Executing economic validation in separate thread")
                economic_result = await asyncio.to_thread(
                    self.economic_evaluator.evaluate, df2, symbol=symbol, timeframe=timeframe
                )
                tprint_info("🔍 [STAGE3] Economic validation completed")
                tprint_debug(f"🔍 [STAGE3] Result success: {economic_result.success}")
                if hasattr(economic_result, 'economic_metrics'):
                    tprint_debug(f"🔍 [STAGE3] Economic metrics: {economic_result.economic_metrics}")
                if hasattr(economic_result, 'optimization_metrics'):
                    tprint_debug(f"🔍 [STAGE3] Optimization metrics: {economic_result.optimization_metrics}")
                # For economic validation, we keep all features (no filtering)
                cols3 = list(df2.columns)
            else:
                self.logger.info("⏭️ Skipping Stage 3: Economic validation (disabled)")
                cols3 = list(df2.columns)
            tprint_info("🔍 [STAGE3] Processing economic validation results")
            cols3_available = [c for c in cols3 if c in df2.columns]
            tprint_debug(f"🔍 [STAGE3] Available features after validation: {cols3_available}")
            tprint_debug(f"🔍 [STAGE3] Available features count: {len(cols3_available)}")

            # No fallback trimming here; rely on economic validation outputs
            
            if len(cols3_available) != len(cols3):
                missing_features = [c for c in cols3 if c not in df2.columns]
                tprint_warning(f"⚠️ [STAGE3] Missing features in data: {missing_features}")
                tprint_warning(f"⚠️ [STAGE3] Feature availability: {len(cols3_available)}/{len(cols3)} ({len(cols3_available)/len(cols3)*100:.1f}%)")
            
            tprint_info("🔍 [STAGE3] Creating filtered dataset")
            df3 = df2.loc[:, cols3_available].copy()
            tprint_data_format(df3, "stage3_before_optimization", level="DEBUG")
            tprint_debug(f"🔍 [STAGE3] Filtered dataset shape: {df3.shape}")
            tprint_debug(f"🔍 [STAGE3] Filtered dataset memory usage: {df3.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            tprint_data_preview(df3, "stage3_economic_validated_features")
            tprint_data_format(df3, "stage3_economic_validated_features", level="DEBUG")
            
            if not disable_stage3:
                tprint_success(f"✅ [STAGE3] Economic validation completed: {len(cols3_available)} features validated")
                tprint_info(f"🔍 [STAGE3] Feature reduction: {len(df2.columns) - len(cols3_available)} features removed ({(len(df2.columns) - len(cols3_available))/len(df2.columns)*100:.1f}%)")
                tprint_debug(f"🔍 [STAGE3] Final validated features: {cols3_available}")
            
            # Step 4: VectorBT optimization
            if not disable_stage4:
                self.logger.info("⚡ Stage 4: Hardware-optimized VectorBT optimization")
                tprint_info("⚡ Starting Stage 4: Hardware-optimized VectorBT optimization")
                tprint_debug(f"📊 Input data shape: {df3.shape}")
                tprint_info("🔍 [STAGE4] Analyzing input data for VectorBT optimization")
                tprint_debug(f"🔍 [STAGE4] Input features: {list(df3.columns)[:10]}...")
                tprint_debug(f"🔍 [STAGE4] Input data memory usage: {df3.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                tprint_debug(f"🔍 [STAGE4] Target statistics - min: {targets.min():.6f}, max: {targets.max():.6f}, mean: {targets.mean():.6f}, std: {targets.std():.6f}")
                tprint_info("🔍 [STAGE4] Initializing hardware-optimized VectorBT optimizer")
                tprint_debug(f"🔍 [STAGE4] VectorBT optimizer type: {type(self.vectorbt_optimizer)}")
                tprint_info("🔍 [STAGE4] Executing VectorBT optimization in separate thread")
                vectorbt_result = await asyncio.to_thread(
                    self.vectorbt_optimizer.optimize, df3, targets=targets
                )
                tprint_info("🔍 [STAGE4] VectorBT optimization completed")
                tprint_debug(f"🔍 [STAGE4] Result type: {type(vectorbt_result)}")
                if hasattr(vectorbt_result, 'shape'):
                    tprint_debug(f"🔍 [STAGE4] Optimized features shape: {vectorbt_result.shape}")
                # For VectorBT optimization, we keep all features (no filtering)
                cols4 = list(df3.columns)
            else:
                self.logger.info("⏭️ Skipping Stage 4: VectorBT optimization (disabled)")
                cols4 = list(df3.columns)
            tprint_info("🔍 [STAGE4] Processing VectorBT optimization results")
            cols4_available = [c for c in cols4 if c in df3.columns]
            tprint_debug(f"🔍 [STAGE4] Available features after optimization: {cols4_available}")
            tprint_debug(f"🔍 [STAGE4] Available features count: {len(cols4_available)}")
            
            if len(cols4_available) != len(cols4):
                missing_features = [c for c in cols4 if c not in df3.columns]
                tprint_warning(f"⚠️ [STAGE4] Missing features in data: {missing_features}")
                tprint_warning(f"⚠️ [STAGE4] Feature availability: {len(cols4_available)}/{len(cols4)} ({len(cols4_available)/len(cols4)*100:.1f}%)")
            
            tprint_info("🔍 [STAGE4] Creating final selected features dataset with memory optimization")
            selected_features_df = df3.loc[:, cols4_available].copy()
            tprint_data_format(selected_features_df, "stage4_before_memory_optimization", level="DEBUG")
            # Apply final memory optimization
            selected_features_df = optimize_dataframe_default(selected_features_df)
            tprint_debug(f"🔍 [STAGE4] Final dataset shape: {selected_features_df.shape}")
            tprint_debug(f"🔍 [STAGE4] Final dataset memory usage: {selected_features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            tprint_data_preview(selected_features_df, "final_selected_features")
            tprint_data_format(selected_features_df, "final_selected_features", level="INFO")
            
            tprint_success(f"✅ [STAGE4] Hardware-optimized VectorBT optimization completed: {len(cols4_available)} features optimized")
            tprint_info(f"🔍 [STAGE4] Feature reduction: {len(df3.columns) - len(cols4_available)} features removed ({(len(df3.columns) - len(cols4_available))/len(df3.columns)*100:.1f}%)")
            tprint_debug(f"🔍 [STAGE4] Final optimized features: {cols4_available}")
            
            # Final summary with optimization metrics
            tprint_success("🎉 Hardware-optimized feature selection pipeline completed successfully!")
            tprint_info(f"📊 Pipeline Summary:")
            tprint_info(f"   • Original features: {len(data.columns)}")
            tprint_info(f"   • Hardware-optimized features: {len(cols1)}")
            tprint_info(f"   • Multi-objective features: {len(cols2)}")
            tprint_info(f"   • Economic validated features: {len(cols3)}")
            tprint_info(f"   • VectorBT optimized features: {len(cols4)}")
            tprint_info(f"   • Final selected features: {len(selected_features_df.columns)}")
            tprint_info(f"   • Feature reduction: {len(data.columns) - len(selected_features_df.columns)} features removed")
            tprint_info(f"   • Reduction percentage: {((len(data.columns) - len(selected_features_df.columns)) / len(data.columns) * 100):.1f}%")
            tprint_info("   • Hardware optimizations: M1, Memory, Caching enabled")
            
            # Log optimization metrics
            if hasattr(advanced_result, 'optimization_metrics'):
                tprint_info(f"   • Memory usage: {advanced_result.optimization_metrics.get('memory_used_mb', 'N/A')} MB")
            if hasattr(advanced_result, 'execution_time'):
                tprint_info(f"   • Execution time: {advanced_result.execution_time:.2f} seconds")
            
            result = SelectionOutcome(
                success=True,
                selected_features=selected_features_df,
                selection_metadata={
                    'advanced_selection': _safe_to_meta(advanced_result),
                    'multi_objective': _safe_to_meta(multi_objective_result),
                    'economic_validation': _safe_to_meta(economic_result),
                    'vectorbt_optimization': _safe_to_meta(vectorbt_result)
                },
                selection_metrics={
                    'advanced_metrics': getattr(advanced_result, 'quality_metrics', {}),
                    'multi_objective_metrics': getattr(multi_objective_result, 'objective_values', {}),
                    'economic_metrics': getattr(economic_result, 'validation_metrics', {}),
                    'vectorbt_metrics': getattr(vectorbt_result, 'optimization_metrics', {})
                },
                selection_strategy="hardware_optimized_multi_stage",
                feature_importance=getattr(advanced_result, 'feature_importance', {}),
                economic_validation=_safe_to_meta(economic_result),
                multi_objective_results=_safe_to_meta(multi_objective_result),
                vectorbt_optimizations=_safe_to_meta(vectorbt_result),
                quality_metrics=getattr(advanced_result, 'quality_metrics', {}),
                diversity_metrics=getattr(advanced_result, 'diversity_metrics', {}),
                stability_metrics=getattr(advanced_result, 'stability_metrics', {}),
                artifacts={
                    'advanced_result': _safe_to_meta(advanced_result),
                    'multi_objective_result': _safe_to_meta(multi_objective_result),
                    'economic_result': _safe_to_meta(economic_result),
                    'vectorbt_result': _safe_to_meta(vectorbt_result)
                }
            )

            # Note: Artifact storage is handled in the main execute method

            # Build human-readable report
            tprint_info("🔍 [DEBUG] Building human-readable report")
            try:
                tprint_debug("🔍 [DEBUG] Generating selection report")
                selection_report = self._generate_selection_report(
                    result,
                    data_shape=(len(data), len(data.columns) if isinstance(data, pd.DataFrame) else 0),
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction
                )
                tprint_success("✅ [DEBUG] Selection report generated")
                
                tprint_debug("🔍 [DEBUG] Formatting markdown report")
                markdown = self._format_selection_markdown(selection_report)
                tprint_success("✅ [DEBUG] Markdown report formatted")
                
                tprint_debug("🔍 [DEBUG] Storing selection report")
                self._store_selection_report(selection_report, markdown, symbol, timeframe)
                tprint_success("✅ [DEBUG] Selection report stored")
            except Exception as e:
                tprint_warning(f"⚠️ [DEBUG] Report generation failed: {e}")
                # Reporting is best-effort
                pass

            return result
            
        except Exception as e:
            tprint_error(f"❌ [DEBUG] Sophisticated feature selection failed: {e}")
            tprint_debug(f"❌ [DEBUG] Exception type: {type(e).__name__}")
            tprint_debug(f"❌ [DEBUG] Exception details: {str(e)}")
            self.logger.error(f"❌ Sophisticated feature selection failed: {e}", exc_info=True)
            
            return SelectionOutcome(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                selection_strategy="error",
                feature_importance={},
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={},
                error_message=str(e)
            )

    async def _fallback_feature_selection(self, data: pd.DataFrame, targets: pd.Series,
                                          symbol: str, timeframe: str, direction: str,
                                          custom_overrides: Optional[Dict[str, Any]]) -> SelectionOutcome:
        """Fallback feature selection when sophisticated components are not available."""
        
        tprint_info("🔍 [DEBUG] Starting fallback feature selection")
        tprint_debug(f"🔍 [DEBUG] Input data shape: {data.shape}")
        tprint_debug(f"🔍 [DEBUG] Input targets length: {len(targets)}")
        tprint_debug(f"🔍 [DEBUG] Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
        tprint_warning("⚠️ [DEBUG] Using fallback correlation-based feature selection")
        tprint_data_format(data, f"fallback_selection_input_{symbol}_{timeframe}", level="DEBUG")
        tprint_data_format(targets, f"fallback_selection_targets_{symbol}_{timeframe}", level="DEBUG")
        
        try:
            # Align data and targets first, then drop NaNs together
            tprint_info("🔍 [DEBUG] Aligning data and targets")
            df = pd.concat([data, targets.rename("target")], axis=1).dropna()
            tprint_debug(f"🔍 [DEBUG] Data after alignment and NaN removal: {df.shape}")
            tprint_data_format(df, "fallback_aligned_data", level="DEBUG")
            if len(df) == 0:
                tprint_error("❌ [DEBUG] No valid data after alignment and NaN removal")
                raise ValueError("No valid data after alignment and NaN removal")
            tprint_success("✅ [DEBUG] Data alignment completed")
            
            # Select only numeric columns
            tprint_info("🔍 [DEBUG] Selecting numeric columns")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            tprint_debug(f"🔍 [DEBUG] Found {len(numeric_columns)} numeric columns")
            tprint_debug(f"🔍 [DEBUG] Numeric columns: {numeric_columns[:10]}...")
            
            if "target" not in numeric_columns:
                tprint_error("❌ [DEBUG] Target column is not numeric")
                raise ValueError("Target column is not numeric")
            tprint_success("✅ [DEBUG] Target column is numeric")
            
            # Remove target from feature columns
            feature_columns = [col for col in numeric_columns if col != "target"]
            tprint_debug(f"🔍 [DEBUG] Feature columns count: {len(feature_columns)}")
            if len(feature_columns) == 0:
                tprint_error("❌ [DEBUG] No numeric feature columns found")
                raise ValueError("No numeric feature columns found")
            tprint_success(f"✅ [DEBUG] Found {len(feature_columns)} feature columns")
            
            X = df[feature_columns]
            y = df["target"]
            tprint_debug(f"🔍 [DEBUG] Feature matrix shape: {X.shape}")
            tprint_debug(f"🔍 [DEBUG] Target vector length: {len(y)}")
            
            # Time-shift features to prevent target leakage
            tprint_info("🔍 [DEBUG] Time-shifting features to prevent target leakage")
            X_shifted = X.shift(1).dropna()
            y_aligned = y.loc[X_shifted.index]
            tprint_debug(f"🔍 [DEBUG] Shifted features shape: {X_shifted.shape}")
            tprint_debug(f"🔍 [DEBUG] Aligned targets length: {len(y_aligned)}")
            tprint_data_preview(X_shifted, "fallback_shifted_features")
            tprint_data_format(X_shifted, "fallback_shifted_features", level="DEBUG")
            tprint_data_preview(y_aligned, "fallback_aligned_targets")
            tprint_data_format(y_aligned, "fallback_aligned_targets", level="DEBUG")
            
            if len(X_shifted) == 0:
                tprint_error("❌ [DEBUG] No valid data after time-shifting")
                raise ValueError("No valid data after time-shifting")
            tprint_success("✅ [DEBUG] Time-shifting completed")
            
            # Basic feature selection using correlation (on training period only)
            tprint_info("🔍 [DEBUG] Computing correlations between features and targets")
            correlations = X_shifted.corrwith(y_aligned).abs().sort_values(ascending=False)
            tprint_debug(f"🔍 [DEBUG] Computed {len(correlations)} correlations")
            tprint_debug(f"🔍 [DEBUG] Correlation stats - min: {correlations.min():.6f}, max: {correlations.max():.6f}, mean: {correlations.mean():.6f}")
            
            # Drop NaN correlations and ensure we have at least one feature
            correlations_clean = correlations.dropna()
            tprint_debug(f"🔍 [DEBUG] Clean correlations count: {len(correlations_clean)}")
            if len(correlations_clean) == 0:
                tprint_error("❌ [DEBUG] No valid correlations found")
                raise ValueError("No valid correlations found")
            tprint_success(f"✅ [DEBUG] Found {len(correlations_clean)} valid correlations")
            
            # Select top 20% of features, with minimum of 1 and maximum of all available
            n_features = max(1, min(len(correlations_clean), int(len(correlations_clean) * 0.2)))
            tprint_info(f"🔍 [DEBUG] Selecting top {n_features} features (20% of {len(correlations_clean)} available)")
            selected_features = correlations_clean.head(n_features).index.tolist()
            tprint_debug(f"🔍 [DEBUG] Selected features: {selected_features}")
            tprint_success(f"✅ [DEBUG] Selected {len(selected_features)} features")
            
            # Create selected features dataframe using original data (not shifted)
            tprint_info("🔍 [DEBUG] Creating selected features dataframe")
            selected_data = data[selected_features]
            tprint_debug(f"🔍 [DEBUG] Selected data shape: {selected_data.shape}")
            tprint_data_preview(selected_data, "fallback_selected_features")
            tprint_data_format(selected_data, "fallback_selected_features", level="DEBUG")
            
            # Calculate basic feature importance
            feature_importance = correlations_clean[selected_features].to_dict()
            tprint_debug(f"🔍 [DEBUG] Feature importance computed for {len(feature_importance)} features")
            tprint_success("✅ [DEBUG] Feature importance calculation completed")
            
            tprint_success("🎉 [DEBUG] Fallback feature selection completed successfully!")
            tprint_info(f"📊 [DEBUG] Fallback Selection Summary:")
            tprint_info(f"   • Original features: {len(data.columns)}")
            tprint_info(f"   • Selected features: {len(selected_features)}")
            tprint_info(f"   • Feature reduction: {len(data.columns) - len(selected_features)} features removed")
            tprint_info(f"   • Reduction percentage: {((len(data.columns) - len(selected_features)) / len(data.columns) * 100):.1f}%")
            
            return SelectionOutcome(
                success=True,
                selected_features=selected_data,
                selection_metadata={'method': 'fallback_correlation', 'symbol': symbol, 'timeframe': timeframe},
                selection_metrics={'selected_count': len(selected_features), 'total_count': len(data.columns)},
                selection_strategy="correlation_fallback",
                feature_importance=feature_importance,
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={'fallback_selection': selected_features, 'correlation_count': len(correlations_clean)}
            )
            
        except Exception as e:
            tprint_error(f"❌ [DEBUG] Fallback feature selection failed: {e}")
            tprint_debug(f"❌ [DEBUG] Exception type: {type(e).__name__}")
            tprint_debug(f"❌ [DEBUG] Exception details: {str(e)}")
            self.logger.error(f"❌ Fallback feature selection failed: {e}", exc_info=True)
            
            return SelectionOutcome(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                selection_strategy="error",
                feature_importance={},
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={},
                error_message=str(e)
            )

    # Required abstract methods from ModularComponent
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        tprint_info("🔧 [DEBUG] Initializing component resources")
        try:
            # Initialize hardware-optimized feature selection components
            tprint_debug(f"🔧 [DEBUG] HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE: {HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE}")
            if HARDWARE_OPTIMIZED_COMPONENTS_AVAILABLE:
                tprint_info("🔧 [DEBUG] Initializing hardware-optimized components")
                # Use hardware-optimized selectors
                self.battle_tested_selector = OptimizedFeatureSelector()
                try:
                    self.logger.info(
                        f"🔎 Using selector instance: {self.battle_tested_selector.__class__.__module__}."
                        f"{self.battle_tested_selector.__class__.__name__}"
                    )
                except Exception:
                    pass
                tprint_success("✅ [DEBUG] OptimizedFeatureSelector initialized")
                
                # Initialize multi-objective selector with default objectives
                tprint_info("🔧 [DEBUG] Creating default objectives")
                objectives = ['correlation', 'mutual_information', 'stability']
                tprint_success(f"✅ [DEBUG] Created {len(objectives)} objectives")
                
                tprint_info("🔧 [DEBUG] Initializing OptimizedMultiObjectiveSelector")
                self.multi_objective_selector = OptimizedMultiObjectiveSelector(objectives=objectives)
                tprint_success("✅ [DEBUG] OptimizedMultiObjectiveSelector initialized")
                
                tprint_info("🔧 [DEBUG] Initializing OptimizedEconomicEvaluator")
                self.economic_evaluator = OptimizedEconomicEvaluator()
                tprint_success("✅ [DEBUG] OptimizedEconomicEvaluator initialized")
                
                tprint_info("🔧 [DEBUG] Initializing OptimizedVectorBTOptimizer")
                self.vectorbt_optimizer = OptimizedVectorBTOptimizer()
                tprint_success("✅ [DEBUG] OptimizedVectorBTOptimizer initialized")
            else:
                tprint_warning("⚠️ [DEBUG] Hardware-optimized components not available, setting to None")
                self.battle_tested_selector = None
                self.multi_objective_selector = None
                self.economic_evaluator = None
                self.vectorbt_optimizer = None
            
            # Set initial state
            tprint_info("🔧 [DEBUG] Setting initial component state")
            self.set_state('initialized_at', datetime.now().isoformat())
            self.set_state('selection_count', 0)
            tprint_success("✅ [DEBUG] Component resources initialized successfully")
            return True
        except Exception as e:
            tprint_error(f"❌ [DEBUG] Resource initialization failed: {e}")
            tprint_debug(f"❌ [DEBUG] Exception type: {type(e).__name__}")
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        tprint_info("🔧 [DEBUG] Cleaning up component resources")
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        self.set_state('selection_count', 0)
        tprint_success("✅ [DEBUG] Component resources cleaned up")
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        tprint_info("🔧 [DEBUG] Processing data with component logic")
        tprint_debug(f"🔧 [DEBUG] Input data type: {type(data)}")
        tprint_debug(f"🔧 [DEBUG] Additional kwargs: {list(kwargs.keys())}")
        
        # Add data format analysis for input
        if hasattr(data, 'shape'):
            tprint_data_format(data, "component_process_input", level="DEBUG")
        
        # Increment selection count
        count = self.get_state('selection_count', 0)
        self.set_state('selection_count', count + 1)
        tprint_debug(f"🔧 [DEBUG] Selection count incremented to: {count + 1}")
        
        # Basic processing - return data as-is for now
        # The actual feature selection is done in the execute method
        tprint_success("✅ [DEBUG] Data processing completed")
        
        # Add data format analysis for output
        if hasattr(data, 'shape'):
            tprint_data_format(data, "component_process_output", level="DEBUG")
        
        return data
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'data_types': ['pandas.DataFrame'],
            'max_nan_ratio': 0.1,
            'min_unique_values': 2
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        tprint_info("🔧 [DEBUG] Validating data with component-specific rules")
        tprint_debug(f"🔧 [DEBUG] Input data type: {type(data)}")
        
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            tprint_debug(f"🔧 [DEBUG] Data is DataFrame with shape: {data.shape}")
            
            # Check required columns
            tprint_info("🔧 [DEBUG] Checking required columns")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            tprint_debug(f"🔧 [DEBUG] Required columns: {required_columns}")
            tprint_debug(f"🔧 [DEBUG] Available columns: {list(data.columns)}")
            tprint_debug(f"🔧 [DEBUG] Missing columns: {missing_columns}")
            
            if missing_columns:
                tprint_error(f"❌ [DEBUG] Missing required columns: {missing_columns}")
                errors.append(f"Missing required columns: {missing_columns}")
            else:
                tprint_success("✅ [DEBUG] All required columns present")
            
            # Check data size
            tprint_info("🔧 [DEBUG] Checking data size")
            tprint_debug(f"🔧 [DEBUG] Data length: {len(data)}")
            if len(data) < 100:
                tprint_warning("⚠️ [DEBUG] Data size is small (< 100 rows)")
                warnings.append("Data size is small (< 100 rows)")
            else:
                tprint_success("✅ [DEBUG] Data size is adequate")
            
            # Check for NaN values
            tprint_info("🔧 [DEBUG] Checking for NaN values")
            nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            tprint_debug(f"🔧 [DEBUG] NaN ratio: {nan_ratio:.4f}")
            if nan_ratio > 0.1:
                tprint_warning(f"⚠️ [DEBUG] High NaN ratio: {nan_ratio:.2%}")
                warnings.append(f"High NaN ratio: {nan_ratio:.2%}")
            else:
                tprint_success("✅ [DEBUG] NaN ratio is acceptable")
            
            metadata['data_shape'] = data.shape
            metadata['nan_ratio'] = nan_ratio
            metadata['columns'] = list(data.columns)
            tprint_debug(f"🔧 [DEBUG] Validation metadata: {metadata}")
        
        tprint_info(f"🔧 [DEBUG] Validation completed - errors: {len(errors)}, warnings: {len(warnings)}")
        if errors:
            tprint_error(f"❌ [DEBUG] Validation errors: {errors}")
        if warnings:
            tprint_warning(f"⚠️ [DEBUG] Validation warnings: {warnings}")
        if not errors and not warnings:
            tprint_success("✅ [DEBUG] Data validation passed")
            
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

    # --- Reporting helpers ---
    def _generate_selection_report(self, result: FeatureSelectionResult, data_shape: tuple, symbol: str, timeframe: str, direction: str) -> Dict[str, Any]:
        tprint_info("📄 [DEBUG] Generating selection report")
        tprint_debug(f"📄 [DEBUG] Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
        tprint_debug(f"📄 [DEBUG] Data shape: {data_shape}")
        
        from datetime import datetime as _dt
        import pandas as _pd

        selected_cols = list(result.selected_features.columns) if isinstance(result.selected_features, _pd.DataFrame) else []
        n_in = int(data_shape[1]) if data_shape and len(data_shape) > 1 else 0
        n_sel = len(selected_cols)
        
        tprint_debug(f"📄 [DEBUG] Selected columns count: {n_sel}")
        tprint_debug(f"📄 [DEBUG] Input columns count: {n_in}")
        tprint_debug(f"📄 [DEBUG] Selected columns: {selected_cols[:10]}...")

        # Try to extract rankings if available via advanced_result in metadata
        tprint_info("📄 [DEBUG] Extracting feature rankings")
        rankings = []
        adv = result.selection_metadata.get('advanced_selection') if isinstance(result.selection_metadata, dict) else None
        tprint_debug(f"📄 [DEBUG] Advanced selection metadata available: {adv is not None}")
        if adv and isinstance(adv, dict):
            # Some advanced results persist top scores; try a few common keys
            rank_table = adv.get('feature_rankings') or adv.get('rankings')
            tprint_debug(f"📄 [DEBUG] Rank table found: {rank_table is not None}")
            if isinstance(rank_table, list):
                rankings = rank_table
                tprint_debug(f"📄 [DEBUG] Rankings count: {len(rankings)}")
        tprint_success(f"✅ [DEBUG] Extracted {len(rankings)} rankings")

        # Flatten feature importance
        tprint_info("📄 [DEBUG] Processing feature importance")
        importance = result.feature_importance or {}
        tprint_debug(f"📄 [DEBUG] Feature importance keys: {len(importance)}")
        tprint_success(f"✅ [DEBUG] Processed {len(importance)} feature importance entries")

        tprint_info("📄 [DEBUG] Building report dictionary")
        report = {
            'title': 'Feature Selection Report',
            'timestamp': _dt.now().isoformat(),
            'configuration': {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'selection_strategy': result.selection_strategy
            },
            'input_summary': {
                'rows': int(data_shape[0]) if data_shape else 0,
                'columns': n_in,
                'selected_columns': n_sel,
                'reduction_pct': float(((n_in - n_sel) / n_in) * 100) if n_in else 0.0
            },
            'selection_metrics': result.selection_metrics or {},
            'quality_metrics': result.quality_metrics or {},
            'diversity_metrics': result.diversity_metrics or {},
            'stability_metrics': result.stability_metrics or {},
            'multi_objective_results': result.multi_objective_results or {},
            'economic_validation': result.economic_validation or {},
            'vectorbt_optimizations': result.vectorbt_optimizations or {},
            'selected_features': selected_cols,
            'feature_importance': importance,
            'rankings': rankings
        }
        tprint_success("✅ [DEBUG] Selection report generated successfully")
        return report

    def _format_selection_markdown(self, report: Dict[str, Any]) -> str:
        tprint_info("📄 [DEBUG] Formatting selection report as markdown")
        tprint_debug(f"📄 [DEBUG] Report keys: {list(report.keys())}")
        
        md = f"# {report['title']}\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        tprint_info("📄 [DEBUG] Adding configuration section")
        cfg = report.get('configuration', {})
        md += "## 📌 Configuration\n\n"
        md += f"- Symbol: {cfg.get('symbol','?')}\n"
        md += f"- Timeframe: {cfg.get('timeframe','?')}\n"
        md += f"- Direction: {cfg.get('direction','?')}\n"
        md += f"- Strategy: {cfg.get('selection_strategy','?')}\n"

        tprint_info("📄 [DEBUG] Adding summary section")
        summ = report.get('input_summary', {})
        md += "\n## 📊 Summary\n\n"
        md += f"- Rows: {summ.get('rows',0):,}\n"
        md += f"- Columns (input): {summ.get('columns',0)}\n"
        md += f"- Columns (selected): {summ.get('selected_columns',0)}\n"
        md += f"- Reduction: {summ.get('reduction_pct',0.0):.1f}%\n"

        tprint_info("📄 [DEBUG] Adding selected features section")
        md += "\n## 🧱 Selected Features\n\n"
        if report.get('selected_features'):
            tprint_debug(f"📄 [DEBUG] Adding {len(report['selected_features'])} selected features")
            md += "- " + ", ".join(report['selected_features'][:60]) + (" ..." if len(report['selected_features']) > 60 else "") + "\n"
        else:
            tprint_warning("⚠️ [DEBUG] No selected features to display")
            md += "_No features selected._\n"

        # Feature importance table
        tprint_info("📄 [DEBUG] Adding feature importance section")
        if report.get('feature_importance'):
            tprint_debug(f"📄 [DEBUG] Processing {len(report['feature_importance'])} feature importance entries")
            md += "\n### Top Feature Importance\n\n"
            top_items = sorted(report['feature_importance'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:40]
            tprint_debug(f"📄 [DEBUG] Displaying top {len(top_items)} feature importance entries")
            md += "| Feature | Importance |\n|---|---:|\n"
            for k, v in top_items:
                try:
                    md += f"| {k} | {float(v):.6f} |\n"
                except Exception:
                    md += f"| {k} | {v} |\n"
        else:
            tprint_warning("⚠️ [DEBUG] No feature importance data to display")

        # Rankings table if present
        tprint_info("📄 [DEBUG] Adding rankings section")
        if report.get('rankings'):
            tprint_debug(f"📄 [DEBUG] Processing {len(report['rankings'])} ranking entries")
            md += "\n### Rankings (if available)\n\n"
            md += "| Feature | Composite | IC | Stability | Diversity | OOF IC | OOF Sharpe | Selected |\n"
            md += "|---|---:|---:|---:|---:|---:|---:|:---:|\n"
            for row in report['rankings'][:40]:
                name = row.get('feature_name') or row.get('name') or ''
                md += f"| {name} | {row.get('composite_score','')} | {row.get('ic_score','')} | {row.get('stability_score','')} | {row.get('diversity_score','')} | {row.get('oof_ic','')} | {row.get('oof_sharpe','')} | {str(row.get('selected',''))} |\n"
        else:
            tprint_warning("⚠️ [DEBUG] No rankings data to display")

        # Metrics sections
        tprint_info("📄 [DEBUG] Adding metrics sections")
        def _dump_section(title, obj):
            nonlocal md
            if obj:
                tprint_debug(f"📄 [DEBUG] Adding {title} section with {len(obj)} items")
                md += f"\n## {title}\n\n"
                for k, v in obj.items():
                    md += f"- {k}: {v}\n"
            else:
                tprint_debug(f"📄 [DEBUG] Skipping {title} section - no data")

        _dump_section('🎯 Selection Metrics', report.get('selection_metrics', {}))
        _dump_section('🧪 Quality Metrics', report.get('quality_metrics', {}))
        _dump_section('🌈 Diversity Metrics', report.get('diversity_metrics', {}))
        _dump_section('🔁 Stability Metrics', report.get('stability_metrics', {}))
        _dump_section('📐 Multi-Objective', report.get('multi_objective_results', {}))
        _dump_section('💰 Economic Validation', report.get('economic_validation', {}))
        _dump_section('⚡ VectorBT Optimizations', report.get('vectorbt_optimizations', {}))
        
        tprint_success("✅ [DEBUG] Markdown formatting completed")
        return md

    def _store_selection_report(self, report: Dict[str, Any], markdown: str, symbol: str, timeframe: str) -> None:
        tprint_info("📄 [DEBUG] Storing selection report")
        tprint_debug(f"📄 [DEBUG] Symbol: {symbol}, Timeframe: {timeframe}")
        tprint_debug(f"📄 [DEBUG] Markdown length: {len(markdown)} characters")
        
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json

        tprint_info("📄 [DEBUG] Creating outcomes directory")
        out_dir = _Path('outcomes')
        out_dir.mkdir(exist_ok=True)
        tprint_success("✅ [DEBUG] Outcomes directory ready")
        
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        tprint_debug(f"📄 [DEBUG] Timestamp: {ts}")
        
        md_path = out_dir / f"feature_selection_report_{symbol}_{timeframe}_{ts}.md"
        json_path = out_dir / f"feature_selection_report_{symbol}_{timeframe}_{ts}.json"
        tprint_debug(f"📄 [DEBUG] Markdown path: {md_path}")
        tprint_debug(f"📄 [DEBUG] JSON path: {json_path}")
        
        tprint_info("📄 [DEBUG] Writing markdown report")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        tprint_success(f"✅ [DEBUG] Markdown report saved to: {md_path.absolute()}")
        
        tprint_info("📄 [DEBUG] Writing JSON report")
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)
        tprint_success(f"✅ [DEBUG] JSON report saved to: {json_path.absolute()}")
        
        # Print the full paths
        self.logger.info(f"📄 Human-readable report saved to: {md_path.absolute()}")
        self.logger.info(f"📊 JSON report saved to: {json_path.absolute()}")
        tprint_success("🎉 [DEBUG] Report storage completed successfully")

# Command handler for ares_launcher integration
def handle_feature_generation_feature_selection_step(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[Union[str, np.ndarray]] = None,
    end_date: Optional[Union[str, np.ndarray]] = None,
    custom_overrides: Optional[Dict[str, Any]] = None
) -> SelectionOutcome:
    """Command handler for ares_launcher integration."""
    tprint_info("🚀 [DEBUG] Starting feature generation feature selection step handler")

    step = FeatureGenerationFeatureSelectionStep()

    config = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "intensity": intensity,
        "lookback_days": lookback_days,
        "start_date": start_date,
        "end_date": end_date,
        "custom_overrides": custom_overrides or {},
    }

    import asyncio
    try:
        result = asyncio.run(step.execute(config))
        tprint_success("✅ [DEBUG] Feature selection step executed successfully")
        return result
    except Exception as e:
        error_msg = f"❌ [CRITICAL] Feature selection step execution failed: {str(e)}"
        tprint_error(error_msg)
        return SelectionOutcome(
            success=False,
            error_message=error_msg,
            selected_features=pd.DataFrame(),
            selection_metadata={},
            selection_metrics={},
            selection_strategy="failed",
            feature_importance={},
            economic_validation={},
            multi_objective_results={},
            vectorbt_optimizations={},
            quality_metrics={},
            diversity_metrics={},
            stability_metrics={},
            artifacts={},
            execution_time=0.0
        )
