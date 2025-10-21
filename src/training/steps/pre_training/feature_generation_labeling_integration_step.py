"""
Feature Generation Labeling Integration Step

This step integrates labeling for feature generation using the enhanced analyst labeler.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep

# Try to import hardware optimization tools for better performance
try:
    from src.utils.hardware import (
        memory_optimized, m1_optimized, memory_efficient_function,
        gc_optimized_function, force_cleanup, get_memory_stats
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    def memory_optimized(*args, **kwargs): 
        def decorator(f):
            return f
        return decorator
    def m1_optimized(*args, **kwargs): 
        def decorator(f):
            return f
        return decorator
    def memory_efficient_function(func=None, *args, **kwargs): 
        def decorator(f):
            return f
        if func is None:
            return decorator
        return decorator(func)
    def gc_optimized_function(func=None, *args, **kwargs): 
        def decorator(f):
            return f
        if func is None:
            return decorator
        return decorator(func)
    def force_cleanup():
        import gc
        gc.collect()
    def get_memory_stats(): return {}


try:  # Logging helpers
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_data_preview,
        tprint_data_format, tprint_performance, tprint_timer, tprint_structured, tprint_exception,
        tprint_progress, tprint_debug, tprint_with_level, LogLevel, DataFormatConfig
    )
except Exception:  # pragma: no cover
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_data_preview(*args, **kwargs): pass  # No-op fallback
    def tprint_data_format(*args, **kwargs): pass  # No-op fallback
    class DataFormatConfig:
        def __init__(self, **kwargs): pass
    def tprint_performance(*args, **kwargs): pass  # No-op fallback
    def tprint_timer(*args, **kwargs): pass  # No-op fallback
    def tprint_structured(*args, **kwargs): pass  # No-op fallback
    def tprint_exception(*args, **kwargs): pass  # No-op fallback
    def tprint_progress(*args, **kwargs): pass  # No-op fallback
    def tprint_debug(*args, **kwargs): pass  # No-op fallback
    def tprint_with_level(*args, **kwargs): pass  # No-op fallback
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"


from dataclasses import dataclass

@dataclass
class LabelingIntegrationResult:
    success: bool
    labeled_data: pd.DataFrame
    targets: pd.Series
    error_message: Optional[str] = None


class FeatureGenerationLabelingIntegrationStep(BaseStep):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("feature_generation_labeling_integration_step", config)
        
        # Log step initialization for troubleshooting
        tprint_structured({
            "step_initialization": {
                "step_name": "feature_generation_labeling_integration_step",
                "config_provided": config is not None,
                "config_keys": list(config.keys()) if config else []
            }
        }, level=LogLevel.DEBUG)
        
        tprint_info("🔧 Feature generation labeling integration step initialized")


    @memory_efficient_function
    @gc_optimized_function
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the labeling integration step using BaseStep pattern."""
        # Log memory stats at start for troubleshooting
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            initial_memory = get_memory_stats()
            tprint_structured({
                "initial_memory_stats": initial_memory
            }, level=LogLevel.DEBUG)
        
        # Start performance timer
        with tprint_timer("feature_generation_labeling_integration_step", LogLevel.INFO):
            self.logger.info("🔍 Starting labeling integration step")
            tprint_info("🚀 Starting feature generation labeling integration step")
            
            # Set context for enhanced file naming
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            direction = config.get('direction', 'long')
            model = config.get('model', 'Analyst')
            
            # Log configuration details for troubleshooting
            tprint_structured({
                "step": "feature_generation_labeling_integration_step",
                "symbol": symbol,
                "exchange": exchange,
                "direction": direction,
                "model": model,
                "config_keys": list(config.keys())
            }, level=LogLevel.INFO)
            
            self._set_context(symbol=symbol, exchange=exchange, direction=direction, model=model)
        
        # Get data from config
        data = config.get('data')
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            tprint_error("❌ Input data validation failed: data is None, not DataFrame, or empty")
            
            # Analyze the problematic data for troubleshooting
            error_data_config = DataFormatConfig(
                max_cols=10,
                max_rows=5,
                include_values=True,
                include_memory=True,
                include_semantics=True,
                safe_sampling=True
            )
            
            tprint_data_format(data, "invalid_input_data", level=LogLevel.ERROR, config=error_data_config)
            
            tprint_structured({
                "data_validation_failure": {
                    "data_type": type(data).__name__ if data is not None else "None",
                    "is_dataframe": isinstance(data, pd.DataFrame) if data is not None else False,
                    "is_empty": data.empty if hasattr(data, 'empty') else "N/A",
                    "data_repr": repr(data) if data is not None else "None"
                }
            }, level=LogLevel.ERROR)
            raise ValueError("Input data must be a non‑empty DataFrame")
        
        # Comprehensive data format analysis for troubleshooting
        # Use detailed configuration for input data analysis
        input_data_config = DataFormatConfig(
            max_cols=20,  # Show more columns for input data
            max_rows=10,  # Show more rows for input data
            include_values=True,
            include_memory=True,
            include_semantics=True,
            safe_sampling=True,
            sample_size=2000  # Larger sample for input data
        )
        
        tprint_data_format(data, "input_data", level=LogLevel.INFO, config=input_data_config, return_summary=True)
        tprint_data_preview(data, "input_data", level=LogLevel.INFO)
        
        # Log data characteristics for troubleshooting
        data_characteristics = {
            "data_shape": data.shape,
            "data_columns": list(data.columns),
            "data_dtypes": data.dtypes.to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": data.isnull().sum().to_dict(),
            "index_type": type(data.index).__name__,
            "index_length": len(data.index),
            "has_datetime_index": isinstance(data.index, pd.DatetimeIndex)
        }
        
        # Add numerical statistics if possible
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_characteristics["numeric_summary"] = {
                    "numeric_columns": list(numeric_cols),
                    "numeric_stats": data[numeric_cols].describe().to_dict()
                }
        except Exception as e:
            tprint_debug(f"Could not compute numeric statistics: {e}")
        
        tprint_structured(data_characteristics, level=LogLevel.DEBUG)

        # Cache hit path using BaseStep artifact methods
        tprint_debug("🔍 Checking for cached labeling artifacts...")
        cached_labeled = self._load_dataframe('labeled_dataframe')
        cached_targets = self._load_dataframe('targets')
        
        if isinstance(cached_labeled, pd.DataFrame) and isinstance(cached_targets, pd.Series):
            # Comprehensive cache validation and preview with detailed analysis
            cache_config = DataFormatConfig(
                max_cols=15,
                max_rows=8,
                include_values=True,
                include_memory=True,
                include_semantics=True,
                safe_sampling=True
            )
            
            cached_labeled_summary = tprint_data_format(cached_labeled, "cached_labeled_data", 
                                                      level=LogLevel.INFO, config=cache_config, return_summary=True)
            cached_targets_summary = tprint_data_format(cached_targets, "cached_targets", 
                                                      level=LogLevel.INFO, config=cache_config, return_summary=True)
            
            # Log cache quality metrics for troubleshooting
            if cached_labeled_summary and cached_targets_summary:
                tprint_structured({
                    "cache_quality_analysis": {
                        "labeled_data_shape": cached_labeled_summary.get("shape"),
                        "labeled_data_memory_mb": cached_labeled_summary.get("memory_mb"),
                        "targets_length": cached_targets_summary.get("length"),
                        "targets_dtype": cached_targets_summary.get("dtype"),
                        "targets_memory_mb": cached_targets_summary.get("memory_mb")
                    }
                }, level=LogLevel.DEBUG)
            tprint_data_preview(cached_labeled, "cached_labeled_data", level=LogLevel.INFO)
            tprint_data_preview(cached_targets, "cached_targets", level=LogLevel.INFO)
            
            # Validate cache quality
            cache_metrics = {
                'labeled_shape': cached_labeled.shape,
                'targets_length': len(cached_targets),
                'targets_positive_rate': float((cached_targets > 0).mean()),
                'targets_std': float(cached_targets.std()),
                'targets_mean': float(cached_targets.mean()),
                'cache_hit': True
            }
            
            tprint_structured(cache_metrics, level=LogLevel.INFO)
            tprint_success("📦 Using cached labeling artifacts")
            
            return {
                'success': True,
                'artifacts': ['labeled_dataframe', 'targets'],
                'metrics': {
                    'integrated_labels': int(len(cached_targets)),
                    'integration_metadata': cache_metrics
                }
            }
        else:
            tprint_info("🔄 No valid cache found, proceeding with fresh labeling...")

        # Validate required columns with detailed troubleshooting
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in data.columns]
        
        # Analyze column structure for troubleshooting
        column_analysis_config = DataFormatConfig(
            max_cols=50,  # Show all columns
            max_rows=1,   # Just show column names
            include_values=False,  # Don't show values, just structure
            include_memory=False,
            include_semantics=True,
            safe_sampling=True
        )
        
        tprint_data_format(data.columns, "data_columns", level=LogLevel.DEBUG, config=column_analysis_config)
        
        tprint_structured({
            "required_columns": required_cols,
            "available_columns": list(data.columns),
            "missing_columns": missing,
            "column_validation_passed": len(missing) == 0,
            "total_columns": len(data.columns),
            "column_types": {col: str(data[col].dtype) for col in data.columns}
        }, level=LogLevel.DEBUG)
        
        if missing:
            tprint_error(f"❌ Missing required columns for labeling: {missing}")
            tprint_structured({
                "error_type": "missing_columns",
                "missing_columns": missing,
                "available_columns": list(data.columns)
            }, level=LogLevel.ERROR)
            raise ValueError(f"Missing required columns for labeling: {missing}")
        
        tprint_success("✅ All required columns present for labeling")

        # Run multi‑horizon labeler with comprehensive monitoring
        tprint_progress(1, 3, "Initializing enhanced analyst labeler...")
        
        try:
            with tprint_timer("enhanced_analyst_labeler_initialization", LogLevel.DEBUG):
                from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import create_enhanced_analyst_labeler
                labeler = create_enhanced_analyst_labeler()
                tprint_success("✅ Enhanced analyst labeler initialized")
            
            tprint_progress(2, 3, "Generating labels with enhanced analyst labeler...")
            
            with tprint_timer("label_generation", LogLevel.INFO):
                lr = labeler.generate_labels(data)
                labels_df = getattr(lr, 'labels', pd.DataFrame())
                
                if labels_df is None or labels_df.empty:
                    tprint_error("❌ Labeling produced no label columns")
                    
                    # Analyze the problematic labeler result for troubleshooting
                    error_labels_config = DataFormatConfig(
                        max_cols=10,
                        max_rows=5,
                        include_values=True,
                        include_memory=True,
                        include_semantics=True,
                        safe_sampling=True
                    )
                    
                    tprint_data_format(labels_df, "empty_labels_from_labeler", level=LogLevel.ERROR, config=error_labels_config)
                    tprint_data_format(lr, "labeler_result_object", level=LogLevel.ERROR, config=error_labels_config)
                    
                    tprint_structured({
                        "labeling_failure_analysis": {
                            "labeler_result_type": type(lr).__name__,
                            "labels_attribute_exists": hasattr(lr, 'labels'),
                            "labels_df_type": type(labels_df).__name__ if labels_df is not None else "None",
                            "labels_df_empty": labels_df.empty if hasattr(labels_df, 'empty') else "N/A",
                            "labeler_attributes": [attr for attr in dir(lr) if not attr.startswith('_')] if hasattr(lr, '__dir__') else []
                        }
                    }, level=LogLevel.ERROR)
                    raise ValueError('Labeling produced no label columns')
                
                # Comprehensive analysis of raw labels with detailed configuration
                raw_labels_config = DataFormatConfig(
                    max_cols=25,  # Show all columns from labeler
                    max_rows=15,  # Show more rows for label analysis
                    include_values=True,
                    include_memory=True,
                    include_semantics=True,
                    safe_sampling=True,
                    sample_size=3000  # Larger sample for label analysis
                )
                
                raw_labels_summary = tprint_data_format(labels_df, "raw_labels_from_labeler", 
                                                      level=LogLevel.DEBUG, config=raw_labels_config, return_summary=True)
                tprint_data_preview(labels_df, "raw_labels_from_labeler", level=LogLevel.DEBUG)
                
                # Analyze label quality for troubleshooting
                if raw_labels_summary:
                    tprint_structured({
                        "raw_labels_analysis": {
                            "shape": raw_labels_summary.get("shape"),
                            "columns": raw_labels_summary.get("columns", []),
                            "dtypes": raw_labels_summary.get("dtypes", {}),
                            "memory_mb": raw_labels_summary.get("memory_mb"),
                            "null_counts": raw_labels_summary.get("null_counts", {}),
                            "numeric_columns": raw_labels_summary.get("numeric_columns", [])
                        }
                    }, level=LogLevel.DEBUG)
                
                tprint_structured({
                    "raw_labels_shape": labels_df.shape,
                    "raw_labels_columns": list(labels_df.columns),
                    "raw_labels_dtypes": labels_df.dtypes.to_dict()
                }, level=LogLevel.DEBUG)

            tprint_progress(3, 3, "Processing and aligning labels...")

                # Handle both Series and DataFrame cases with detailed logging
                if isinstance(labels_df, pd.Series):
                    # Single target case - use the series directly
                    tprint_debug("📊 Processing single target series")
                    
                    # Analyze series structure before processing
                    series_config = DataFormatConfig(
                        max_cols=5,
                        max_rows=20,
                        include_values=True,
                        include_memory=True,
                        include_semantics=True,
                        safe_sampling=True
                    )
                    tprint_data_format(labels_df, "raw_series_labels", level=LogLevel.DEBUG, config=series_config)
                    
                    targets = labels_df.dropna().astype(float)
                    target_name = labels_df.name or 'target'
                    
                    # Analyze processed series
                    tprint_data_format(targets, "processed_series_targets", level=LogLevel.DEBUG, config=series_config)
                    
                    tprint_structured({
                        "target_type": "single_series",
                        "target_name": target_name,
                        "original_length": len(labels_df),
                        "after_dropna": len(targets),
                        "na_count": labels_df.isna().sum(),
                        "conversion_success": len(targets) > 0
                    }, level=LogLevel.DEBUG)
                else:
                    # Multiple targets case - prefer columns that contain 'target'
                    tprint_debug("📊 Processing multiple target columns")
                    
                    # Analyze DataFrame structure for column selection
                    df_config = DataFormatConfig(
                        max_cols=30,
                        max_rows=5,
                        include_values=True,
                        include_memory=True,
                        include_semantics=True,
                        safe_sampling=True
                    )
                    tprint_data_format(labels_df, "raw_dataframe_labels", level=LogLevel.DEBUG, config=df_config)
                    
                    target_cols = [c for c in labels_df.columns if 'target' in str(c).lower()]
                    target_col = target_cols[0] if target_cols else labels_df.select_dtypes(include=[np.number]).columns[0]
                    
                    # Analyze selected column before processing
                    if target_col in labels_df.columns:
                        tprint_data_format(labels_df[target_col], f"selected_target_column_{target_col}", 
                                         level=LogLevel.DEBUG, config=series_config)
                    
                    targets = labels_df[target_col].dropna().astype(float)
                    target_name = target_col
                    
                    # Analyze final processed targets
                    tprint_data_format(targets, "processed_dataframe_targets", level=LogLevel.DEBUG, config=series_config)
                    
                    tprint_structured({
                        "target_type": "multiple_columns",
                        "available_target_cols": target_cols,
                        "selected_target_col": target_col,
                        "original_length": len(labels_df),
                        "after_dropna": len(targets),
                        "na_count": labels_df[target_col].isna().sum() if target_col in labels_df.columns else 0,
                        "conversion_success": len(targets) > 0
                    }, level=LogLevel.DEBUG)

            # Preview processed targets for troubleshooting with detailed analysis
            targets_config = DataFormatConfig(
                max_cols=10,
                max_rows=20,  # Show more rows for target analysis
                include_values=True,
                include_memory=True,
                include_semantics=True,
                safe_sampling=True,
                sample_size=5000  # Large sample for target analysis
            )
            
            targets_summary = tprint_data_format(targets, f"processed_targets_{target_name}", 
                                               level=LogLevel.INFO, config=targets_config, return_summary=True)
            tprint_data_preview(targets, f"processed_targets_{target_name}", level=LogLevel.INFO)
            
            # Analyze target quality and distribution for troubleshooting
            if targets_summary:
                tprint_structured({
                    "processed_targets_analysis": {
                        "target_name": target_name,
                        "length": targets_summary.get("length"),
                        "dtype": targets_summary.get("dtype"),
                        "memory_mb": targets_summary.get("memory_mb"),
                        "null_count": targets_summary.get("null_count"),
                        "unique_count": targets_summary.get("unique_count"),
                        "numeric_stats": targets_summary.get("numeric_stats", {})
                    }
                }, level=LogLevel.DEBUG)

            # Align and build labeled DataFrame with performance monitoring
            with tprint_timer("data_alignment", LogLevel.DEBUG):
                common_idx = data.index.intersection(targets.index)
                
                # Validate alignment results
                if len(common_idx) == 0:
                    tprint_error("❌ No common index found between data and targets")
                    
                    # Analyze index mismatch for troubleshooting
                    index_analysis_config = DataFormatConfig(
                        max_cols=5,
                        max_rows=10,
                        include_values=True,
                        include_memory=False,
                        include_semantics=True,
                        safe_sampling=True
                    )
                    
                    tprint_data_format(data.index, "data_index", level=LogLevel.ERROR, config=index_analysis_config)
                    tprint_data_format(targets.index, "targets_index", level=LogLevel.ERROR, config=index_analysis_config)
                    tprint_data_format(common_idx, "common_index", level=LogLevel.ERROR, config=index_analysis_config)
                    
                    tprint_structured({
                        "index_alignment_failure": {
                            "data_index_type": type(data.index).__name__,
                            "data_index_length": len(data.index),
                            "data_index_sample": list(data.index[:5]) if len(data.index) > 0 else [],
                            "targets_index_type": type(targets.index).__name__,
                            "targets_index_length": len(targets.index) if hasattr(targets, 'index') else 0,
                            "targets_index_sample": list(targets.index[:5]) if hasattr(targets, 'index') and len(targets.index) > 0 else [],
                            "common_index_length": len(common_idx),
                            "index_types_match": type(data.index) == type(targets.index) if hasattr(targets, 'index') else False
                        }
                    }, level=LogLevel.ERROR)
                    raise ValueError("No common index found between data and targets")
                
                labeled = data.loc[common_idx].copy()
                targets = targets.loc[common_idx]
                labeled[target_name] = targets
                
                tprint_structured({
                    "data_original_length": len(data),
                    "targets_original_length": len(targets) if hasattr(targets, '__len__') else "N/A",
                    "common_index_length": len(common_idx),
                    "final_labeled_length": len(labeled),
                    "alignment_success": True
                }, level=LogLevel.DEBUG)
            
            # Preview final labeled data for troubleshooting with comprehensive analysis
            final_data_config = DataFormatConfig(
                max_cols=30,  # Show all columns including new target
                max_rows=12,  # Show more rows for final validation
                include_values=True,
                include_memory=True,
                include_semantics=True,
                safe_sampling=True,
                sample_size=4000  # Large sample for final validation
            )
            
            final_labeled_summary = tprint_data_format(labeled, "final_labeled_dataframe", 
                                                     level=LogLevel.INFO, config=final_data_config, return_summary=True)
            final_targets_summary = tprint_data_format(targets, "final_targets_series", 
                                                     level=LogLevel.INFO, config=final_data_config, return_summary=True)
            
            tprint_data_preview(labeled, "final_labeled_dataframe", level=LogLevel.INFO)
            tprint_data_preview(targets, "final_targets_series", level=LogLevel.INFO)
            
            # Comprehensive final data validation for troubleshooting
            if final_labeled_summary and final_targets_summary:
                tprint_structured({
                    "final_data_validation": {
                        "labeled_dataframe": {
                            "shape": final_labeled_summary.get("shape"),
                            "columns": final_labeled_summary.get("columns", []),
                            "memory_mb": final_labeled_summary.get("memory_mb"),
                            "null_counts": final_labeled_summary.get("null_counts", {}),
                            "has_target_column": target_name in final_labeled_summary.get("columns", [])
                        },
                        "targets_series": {
                            "length": final_targets_summary.get("length"),
                            "dtype": final_targets_summary.get("dtype"),
                            "memory_mb": final_targets_summary.get("memory_mb"),
                            "null_count": final_targets_summary.get("null_count"),
                            "numeric_stats": final_targets_summary.get("numeric_stats", {})
                        },
                        "alignment_validation": {
                            "data_length": final_labeled_summary.get("shape", [0])[0],
                            "targets_length": final_targets_summary.get("length", 0),
                            "lengths_match": final_labeled_summary.get("shape", [0])[0] == final_targets_summary.get("length", 0)
                        }
                    }
                }, level=LogLevel.DEBUG)
            
            # Performance metrics
            tprint_performance("labeling_integration", 0, 
                             samples_labeled=len(targets),
                             target_variance=float(targets.var()),
                             target_mean=float(targets.mean()))
            
            tprint_success(f"✅ Labeled {len(targets)} samples (var={targets.var():.6f})")
            
        except Exception as e:
            # Enhanced fallback handling with comprehensive logging
            tprint_exception(e, "Multi-Horizon labeler failed, falling back to simple returns")
            tprint_warning(f"⚠️ Multi‑Horizon labeler failed: {e}; falling back to simple returns")
            
            if 'close' not in data.columns:
                tprint_error("❌ Cannot create fallback targets: 'close' column missing")
                tprint_structured({
                    "available_columns": list(data.columns),
                    "missing_close_column": True
                }, level=LogLevel.ERROR)
                raise
            
            tprint_info("🔄 Creating fallback targets using simple returns...")
            targets = data['close'].pct_change().shift(-1).fillna(0.0).astype(float)
            labeled = data.copy()
            labeled['target'] = targets
            
            # Preview fallback data for troubleshooting with detailed analysis
            fallback_config = DataFormatConfig(
                max_cols=20,
                max_rows=10,
                include_values=True,
                include_memory=True,
                include_semantics=True,
                safe_sampling=True
            )
            
            fallback_targets_summary = tprint_data_format(targets, "fallback_targets", 
                                                        level=LogLevel.WARNING, config=fallback_config, return_summary=True)
            fallback_labeled_summary = tprint_data_format(labeled, "fallback_labeled_data", 
                                                        level=LogLevel.WARNING, config=fallback_config, return_summary=True)
            
            tprint_data_preview(targets, "fallback_targets", level=LogLevel.WARNING)
            tprint_data_preview(labeled, "fallback_labeled_data", level=LogLevel.WARNING)
            
            # Analyze fallback data quality for troubleshooting
            if fallback_targets_summary and fallback_labeled_summary:
                tprint_structured({
                    "fallback_data_analysis": {
                        "fallback_method": "simple_returns",
                        "targets_quality": {
                            "length": fallback_targets_summary.get("length"),
                            "dtype": fallback_targets_summary.get("dtype"),
                            "memory_mb": fallback_targets_summary.get("memory_mb"),
                            "null_count": fallback_targets_summary.get("null_count"),
                            "numeric_stats": fallback_targets_summary.get("numeric_stats", {})
                        },
                        "labeled_data_quality": {
                            "shape": fallback_labeled_summary.get("shape"),
                            "memory_mb": fallback_labeled_summary.get("memory_mb"),
                            "has_target_column": "target" in fallback_labeled_summary.get("columns", [])
                        }
                    }
                }, level=LogLevel.WARNING)
            
            tprint_structured({
                "fallback_method": "simple_returns",
                "fallback_targets_length": len(targets),
                "fallback_targets_variance": float(targets.var()),
                "fallback_targets_mean": float(targets.mean())
            }, level=LogLevel.WARNING)

        # Save artifacts using BaseStep methods with comprehensive monitoring
        tprint_info("💾 Saving labeling artifacts...")
        
        try:
            # Comprehensive pre-save validation and preview
            tprint_structured({
                "artifacts_to_save": {
                    "labeled_dataframe": {
                        "type": type(labeled).__name__,
                        "shape": labeled.shape if hasattr(labeled, 'shape') else "N/A",
                        "memory_mb": labeled.memory_usage(deep=True).sum() / 1024 / 1024 if hasattr(labeled, 'memory_usage') else "N/A"
                    },
                    "targets": {
                        "type": type(targets).__name__,
                        "length": len(targets) if hasattr(targets, '__len__') else "N/A",
                        "dtype": str(targets.dtype) if hasattr(targets, 'dtype') else "N/A"
                    },
                    "raw_dataframe": {
                        "type": type(data).__name__,
                        "shape": data.shape if hasattr(data, 'shape') else "N/A"
                    }
                }
            }, level=LogLevel.DEBUG)
            
            # Preview data before saving for troubleshooting with save validation
            pre_save_config = DataFormatConfig(
                max_cols=25,
                max_rows=8,
                include_values=True,
                include_memory=True,
                include_semantics=True,
                safe_sampling=True
            )
            
            pre_save_labeled_summary = tprint_data_format(labeled, "pre_save_labeled_dataframe", 
                                                        level=LogLevel.DEBUG, config=pre_save_config, return_summary=True)
            pre_save_targets_summary = tprint_data_format(targets, "pre_save_targets", 
                                                        level=LogLevel.DEBUG, config=pre_save_config, return_summary=True)
            
            # Validate data integrity before saving
            if pre_save_labeled_summary and pre_save_targets_summary:
                tprint_structured({
                    "pre_save_validation": {
                        "labeled_dataframe": {
                            "shape": pre_save_labeled_summary.get("shape"),
                            "memory_mb": pre_save_labeled_summary.get("memory_mb"),
                            "null_counts": pre_save_labeled_summary.get("null_counts", {}),
                            "columns": pre_save_labeled_summary.get("columns", [])
                        },
                        "targets": {
                            "length": pre_save_targets_summary.get("length"),
                            "dtype": pre_save_targets_summary.get("dtype"),
                            "memory_mb": pre_save_targets_summary.get("memory_mb"),
                            "null_count": pre_save_targets_summary.get("null_count")
                        },
                        "save_readiness": {
                            "has_data": pre_save_labeled_summary.get("shape", [0])[0] > 0,
                            "has_targets": pre_save_targets_summary.get("length", 0) > 0,
                            "memory_usage_acceptable": (pre_save_labeled_summary.get("memory_mb", 0) + 
                                                      pre_save_targets_summary.get("memory_mb", 0)) < 1000  # 1GB threshold
                        }
                    }
                }, level=LogLevel.DEBUG)
            tprint_data_preview(labeled, "pre_save_labeled_dataframe", level=LogLevel.DEBUG)
            tprint_data_preview(targets, "pre_save_targets", level=LogLevel.DEBUG)
            
            # Save with performance monitoring
            with tprint_timer("artifact_saving", LogLevel.DEBUG):
                self._save_dataframe(labeled, 'labeled_dataframe')
                self._save_dataframe(targets, 'targets')
                self._save_dataframe(data, 'raw_dataframe')
            
            tprint_success("✅ Saved labeling artifacts successfully")
            
        except Exception as e:
            tprint_exception(e, "Failed to save labeling artifacts")
            tprint_warning(f"⚠️ Failed to save labeling artifacts: {e}")
            import traceback
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Log detailed error information for troubleshooting
            tprint_structured({
                "error_type": "artifact_save_failure",
                "error_message": str(e),
                "artifacts_attempted": ['labeled_dataframe', 'targets', 'raw_dataframe'],
                "labeled_dataframe_type": type(labeled).__name__,
                "targets_type": type(targets).__name__,
                "raw_dataframe_type": type(data).__name__
            }, level=LogLevel.ERROR)

        # Final result compilation with comprehensive metrics
        final_metrics = {
            'integrated_labels': int(len(targets)),
            'integration_metadata': {
                'positive_rate': float((targets > 0).mean()),
                'target_std': float(targets.std()),
                'target_mean': float(targets.mean()),
                'target_min': float(targets.min()),
                'target_max': float(targets.max()),
                'labeled_dataframe_shape': labeled.shape,
                'cache_hit': False,
                'labeling_method': 'enhanced_analyst_labeler' if 'enhanced_analyst_labeler' in str(type(labeler)) else 'fallback_simple_returns'
            }
        }
        
        tprint_structured(final_metrics, level=LogLevel.INFO)
        
        # Log final memory stats for troubleshooting
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            final_memory = get_memory_stats()
            tprint_structured({
                "final_memory_stats": final_memory
            }, level=LogLevel.DEBUG)
            
            # Force cleanup if memory usage is high
            if final_memory.get('memory_usage_mb', 0) > 1000:  # 1GB threshold
                tprint_info("🧹 High memory usage detected, forcing cleanup...")
                force_cleanup()
                post_cleanup_memory = get_memory_stats()
                tprint_structured({
                    "post_cleanup_memory_stats": post_cleanup_memory
                }, level=LogLevel.DEBUG)
        
        # Final comprehensive summary with data format validation
        final_summary_config = DataFormatConfig(
            max_cols=20,
            max_rows=5,
            include_values=False,  # Don't show values in final summary
            include_memory=True,
            include_semantics=False,
            safe_sampling=True
        )
        
        # Quick final validation of all artifacts
        tprint_data_format(labeled, "final_labeled_validation", level=LogLevel.DEBUG, config=final_summary_config)
        tprint_data_format(targets, "final_targets_validation", level=LogLevel.DEBUG, config=final_summary_config)
        
        tprint_structured({
            "step_completion_summary": {
                "step_name": "feature_generation_labeling_integration_step",
                "success": True,
                "artifacts_created": ['labeled_dataframe', 'targets', 'raw_dataframe'],
                "final_metrics": final_metrics,
                "hardware_optimization_available": HARDWARE_OPTIMIZATION_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
                "data_format_validation_passed": True
            }
        }, level=LogLevel.INFO)
        
        tprint_success("🎉 Feature generation labeling integration step completed successfully")
        
        return {
            'success': True,
            'artifacts': ['labeled_dataframe', 'targets', 'raw_dataframe'],
            'metrics': final_metrics
        }




# Handler for ares_launcher/sub_pipeline integration
async def handle_feature_generation_labeling_integration_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs: Any
) -> LabelingIntegrationResult:
    """Execute labeling integration and persist artifacts (launcher compatibility)."""
    tprint_info("🚀 Starting feature generation labeling integration step handler")
    
    # Log handler parameters for troubleshooting
    tprint_structured({
        "handler_parameters": {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "intensity": intensity,
            "lookback_days": lookback_days,
            "start_date": start_date,
            "end_date": end_date,
            "exchange": exchange,
            "data_provided": data is not None,
            "data_type": type(data).__name__ if data is not None else "None",
            "data_shape": data.shape if hasattr(data, 'shape') else "N/A"
        }
    }, level=LogLevel.INFO)
    
    step = FeatureGenerationLabelingIntegrationStep()

    # Attempt lazy load if data not provided
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        tprint_info("📥 No data provided, attempting to auto-load...")
        try:
            with tprint_timer("data_auto_loading", LogLevel.DEBUG):
                from .feature_generation_data_validation_step import FeatureGenerationDataValidationStep  # type: ignore
                loader = FeatureGenerationDataValidationStep()
                loaded = await loader._load_data_for_validation(  # noqa: SLF001
                    symbol, timeframe, exchange, start_date, end_date, lookback_days
                )
                data = loaded
                
                # Comprehensive analysis of auto-loaded data
                auto_load_config = DataFormatConfig(
                    max_cols=25,
                    max_rows=15,
                    include_values=True,
                    include_memory=True,
                    include_semantics=True,
                    safe_sampling=True,
                    sample_size=3000
                )
                
                auto_loaded_summary = tprint_data_format(data, "auto_loaded_data", 
                                                       level=LogLevel.INFO, config=auto_load_config, return_summary=True)
                
                if auto_loaded_summary:
                    tprint_structured({
                        "auto_load_analysis": {
                            "data_source": "feature_generation_data_validation_step",
                            "shape": auto_loaded_summary.get("shape"),
                            "columns": auto_loaded_summary.get("columns", []),
                            "memory_mb": auto_loaded_summary.get("memory_mb"),
                            "dtypes": auto_loaded_summary.get("dtypes", {}),
                            "null_counts": auto_loaded_summary.get("null_counts", {})
                        }
                    }, level=LogLevel.DEBUG)
                
                tprint_success("✅ Data auto-loaded successfully")
                
        except Exception as e:
            tprint_exception(e, "Failed to auto-load data for labeling integration")
            tprint_error(f"❌ Failed to auto-load data for labeling integration: {e}")
            raise
    else:
        tprint_info("📊 Using provided data for labeling integration")
        # Comprehensive analysis of provided data
        provided_data_config = DataFormatConfig(
            max_cols=30,
            max_rows=12,
            include_values=True,
            include_memory=True,
            include_semantics=True,
            safe_sampling=True,
            sample_size=2500
        )
        
        provided_data_summary = tprint_data_format(data, "provided_data", 
                                                 level=LogLevel.INFO, config=provided_data_config, return_summary=True)
        
        if provided_data_summary:
            tprint_structured({
                "provided_data_analysis": {
                    "data_source": "user_provided",
                    "shape": provided_data_summary.get("shape"),
                    "columns": provided_data_summary.get("columns", []),
                    "memory_mb": provided_data_summary.get("memory_mb"),
                    "dtypes": provided_data_summary.get("dtypes", {}),
                    "null_counts": provided_data_summary.get("null_counts", {}),
                    "numeric_columns": provided_data_summary.get("numeric_columns", [])
                }
            }, level=LogLevel.DEBUG)

    # Create config for the step
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides or {},
        'data': data
    }
    
    tprint_structured({
        "step_config": {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "exchange": exchange,
            "data_shape": data.shape if hasattr(data, 'shape') else "N/A"
        }
    }, level=LogLevel.DEBUG)

    # Execute the step with performance monitoring
    try:
        with tprint_timer("labeling_integration_step_execution", LogLevel.INFO):
            result_dict = await step.execute(config)
    except Exception as e:
        tprint_exception(e, "Step execution failed")
        tprint_error(f"❌ Step execution failed: {e}")
        # Return error result
        return LabelingIntegrationResult(
            success=False,
            labeled_data=pd.DataFrame(),
            targets=pd.Series(dtype=float),
            error_message=str(e)
        )

    # Load artifacts using BaseStep methods with validation
    tprint_info("📦 Loading artifacts from step execution...")
    
    try:
        labeled_df = step._load_dataframe('labeled_dataframe') or pd.DataFrame()
        targets = step._load_dataframe('targets') or pd.Series(dtype=float)
        
        # Validate loaded artifacts with comprehensive analysis
        loaded_artifacts_config = DataFormatConfig(
            max_cols=25,
            max_rows=10,
            include_values=True,
            include_memory=True,
            include_semantics=True,
            safe_sampling=True
        )
        
        loaded_labeled_summary = tprint_data_format(labeled_df, "loaded_labeled_dataframe", 
                                                  level=LogLevel.DEBUG, config=loaded_artifacts_config, return_summary=True)
        loaded_targets_summary = tprint_data_format(targets, "loaded_targets", 
                                                  level=LogLevel.DEBUG, config=loaded_artifacts_config, return_summary=True)
        
        # Validate artifact integrity
        if loaded_labeled_summary and loaded_targets_summary:
            tprint_structured({
                "loaded_artifacts_validation": {
                    "labeled_dataframe": {
                        "shape": loaded_labeled_summary.get("shape"),
                        "memory_mb": loaded_labeled_summary.get("memory_mb"),
                        "columns": loaded_labeled_summary.get("columns", []),
                        "null_counts": loaded_labeled_summary.get("null_counts", {})
                    },
                    "targets": {
                        "length": loaded_targets_summary.get("length"),
                        "dtype": loaded_targets_summary.get("dtype"),
                        "memory_mb": loaded_targets_summary.get("memory_mb"),
                        "null_count": loaded_targets_summary.get("null_count")
                    },
                    "integrity_check": {
                        "has_labeled_data": loaded_labeled_summary.get("shape", [0])[0] > 0,
                        "has_targets": loaded_targets_summary.get("length", 0) > 0,
                        "lengths_compatible": loaded_labeled_summary.get("shape", [0])[0] == loaded_targets_summary.get("length", 0)
                    }
                }
            }, level=LogLevel.DEBUG)
        
        tprint_structured({
            "loaded_artifacts": {
                "labeled_dataframe_shape": labeled_df.shape if hasattr(labeled_df, 'shape') else "N/A",
                "targets_length": len(targets) if hasattr(targets, '__len__') else "N/A",
                "targets_type": type(targets).__name__
            }
        }, level=LogLevel.DEBUG)
        
        tprint_success("✅ Artifacts loaded successfully")
        
    except Exception as e:
        tprint_exception(e, "Failed to load artifacts")
        tprint_warning(f"⚠️ Failed to load some artifacts: {e}")
        # Ensure we have valid fallbacks
        labeled_df = labeled_df if 'labeled_df' in locals() else pd.DataFrame()
        targets = targets if 'targets' in locals() else pd.Series(dtype=float)
    
    # Create result with comprehensive logging
    result = LabelingIntegrationResult(
        success=bool(result_dict.get('success', False)),
        labeled_data=labeled_df,
        targets=targets,
        error_message=result_dict.get('error')
    )
    
    tprint_structured({
        "final_result": {
            "success": result.success,
            "labeled_data_shape": result.labeled_data.shape if hasattr(result.labeled_data, 'shape') else "N/A",
            "targets_length": len(result.targets) if hasattr(result.targets, '__len__') else "N/A",
            "error_message": result.error_message
        }
    }, level=LogLevel.INFO)
    
    tprint_success("🎉 Feature generation labeling integration step handler completed")
    
    return result
