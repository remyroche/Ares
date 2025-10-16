"""High-level regime analysis orchestration with enhanced monitoring, error handling, and performance optimization."""
from __future__ import annotations

import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import traceback

# Import common operations for data quality and validation
from src.utils.common_operations import (
    validate_dataframe_columns,
    calculate_data_quality_metrics,
    create_data_quality_report,
    safe_convert_dtypes,
    optimize_dataframe_dtypes,
    get_dataframe_info,
    create_summary_statistics,
    safe_fillna,
    safe_merge_dataframes,
    safe_drop_columns,
    safe_rename_columns,
    validate_timestamp_column,
    safe_timestamp_conversion,
    safe_resample,
    align_dataframes,
    validate_dataframe_schema,
    guard_dataframe_nulls,
    get_memory_usage,
    optimize_memory,
    memory_checkpoint,
    gpu_context,
    safe_json_dump,
    safe_json_load,
    safe_copy,
    safe_deepcopy,
    validate_file_path,
    get_file_size,
    check_disk_space
)

# Import math validation for safe operations
from src.utils.math_validation import (
    safe_mean,
    safe_std,
    safe_correlation,
    safe_covariance,
    validate_finite,
    validate_positive,
    validate_range,
    safe_percentage_change,
    safe_weighted_average,
    safe_kelly_calculation,
    safe_percentile,
    safe_matrix_inverse,
    validate_correlation_matrix,
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_power
)

# Import tprint for enhanced logging
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_performance,
    tprint_timer,
    tprint_structured,
    tprint_debug
)

# Import M1 hardware optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    get_m1_memory_optimizer = lambda: None
    get_m1_cpu_optimizer = lambda: None
    get_m1_gpu_manager = lambda: None

# Import VectorBT optimization utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    get_vectorbt_rolling_optimizer = lambda **kwargs: None
    get_unified_vectorization_manager = lambda: None

try:  # pragma: no cover - fallback retained for runtime parity
    from src.utils.logging_utils import get_logger, log_warning
except ImportError:  # pragma: no cover - ensures CLI still works without dependency
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)

    def log_warning(message: str) -> None:
        logging.getLogger("RegimeAnalyzer").warning(message)

from .data_access import load_regime_datasets
from .metrics import calculate_regime_distribution, calculate_clustering_metrics
from .reporting import print_detailed_metrics, print_analysis_summary

class RegimeAnalysisService:
    """Coordinates loading, computation, and reporting of regime metrics with enhanced monitoring and error handling."""

    def __init__(self, data_cache_path: Path | str = "data_cache", enable_vectorbt: bool = True):
        self.data_cache_path = Path(data_cache_path)
        if not self.data_cache_path.exists():
            raise FileNotFoundError(f"Data cache directory not found: {self.data_cache_path}")

        # Initialize logging
        self.logger = get_logger("RegimeAnalyzer")

        # Initialize hardware optimizers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.gpu_manager = get_m1_gpu_manager()

        # Initialize VectorBT optimization
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE
        if self.enable_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.gpu_manager is not None,
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000
            )
            self.vectorization_manager = get_unified_vectorization_manager()
            tprint_success("✅ VectorBT optimization enabled for regime analysis")
        else:
            self.rolling_optimizer = None
            self.vectorization_manager = None
            if enable_vectorbt and not VECTORBT_AVAILABLE:
                tprint_warning("⚠️ VectorBT not available, using standard operations")

        # Performance monitoring
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "memory_usage": [],
            "processing_times": {},
            "error_count": 0,
            "success_count": 0
        }

        # Start memory monitoring if available
        if self.memory_optimizer:
            self.memory_optimizer.start_monitoring()

        tprint_structured({
            "service": "RegimeAnalysisService",
            "data_cache_path": str(self.data_cache_path),
            "m1_hardware_available": M1_HARDWARE_AVAILABLE,
            "vectorbt_available": VECTORBT_AVAILABLE,
            "vectorbt_enabled": self.enable_vectorbt,
            "memory_optimizer": self.memory_optimizer is not None,
            "cpu_optimizer": self.cpu_optimizer is not None,
            "gpu_manager": self.gpu_manager is not None,
            "rolling_optimizer": self.rolling_optimizer is not None,
            "vectorization_manager": self.vectorization_manager is not None
        })
        tprint_success("🔍 Regime Analysis service initialized with enhanced monitoring")

    def analyze(self, symbol: str = "ETHUSDT") -> Dict[str, Any]:
        """Execute the full regime analysis workflow for a symbol with comprehensive monitoring and error handling."""
        # Input validation
        self._validate_analysis_inputs(symbol)

        # Initialize performance monitoring
        self.performance_metrics["start_time"] = time.time()
        self.performance_metrics["error_count"] = 0
        self.performance_metrics["success_count"] = 0

        with tprint_timer(f"Comprehensive regime analysis for {symbol}"):
            try:
                # Log analysis start
                tprint_structured({
                    "analysis_start": datetime.now().isoformat(),
                    "symbol": symbol,
                    "data_cache_path": str(self.data_cache_path),
                    "memory_usage_start": get_memory_usage() / (1024**2)
                })

                # Load datasets with monitoring
                with tprint_timer("Loading datasets"):
                    nas_features, nas_labels, tas_features, tas_labels = self._load_datasets(symbol)
                    self.performance_metrics["success_count"] += 1

                # Print initial overview with enhanced logging
                self._print_initial_overview(nas_labels, tas_labels)

                # Calculate distributions with monitoring
                with tprint_timer("Calculating regime distributions"):
                    nas_distribution = calculate_regime_distribution(nas_labels, "NAS")
                    tas_distribution = calculate_regime_distribution(tas_labels, "TAS")
                    self.performance_metrics["success_count"] += 1

                # Calculate clustering metrics with monitoring
                with tprint_timer("Calculating clustering metrics"):
                    nas_metrics = calculate_clustering_metrics(nas_features, nas_labels, "NAS")
                    tas_metrics = calculate_clustering_metrics(tas_features, tas_labels, "TAS")

                    self.performance_metrics["success_count"] += 1

                # Print detailed metrics
                print_detailed_metrics(nas_distribution, nas_metrics, "NAS")
                print_detailed_metrics(tas_distribution, tas_metrics, "TAS")

                # Compile analysis with enhanced metadata
                with tprint_timer("Compiling analysis"):
                    analysis = self._compile_analysis(
                        symbol,
                        nas_distribution,
                        tas_distribution,
                        nas_metrics,
                        tas_metrics,
                        nas_labels,
                        tas_labels,
                    )
                    self.performance_metrics["success_count"] += 1

                # Save analysis with monitoring
                with tprint_timer("Saving analysis"):
                    output_path = self._save_analysis(analysis, symbol)
                    self.performance_metrics["success_count"] += 1

                # Final performance metrics
                self.performance_metrics["end_time"] = time.time()
                total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]

                # Log VectorBT performance statistics if enabled
                vectorbt_stats = {}
                if self.enable_vectorbt and self.vectorization_manager:
                    stats = self.vectorization_manager.get_performance_stats()
                    vectorbt_stats = {
                        "vectorbt_operations": stats.get('total_operations', 0),
                        "vectorbt_usage_rate": stats.get('vectorbt_usage_rate', 0),
                        "average_operation_time": stats.get('average_operation_time', 0),
                        "memory_optimizations": stats.get('memory_optimizations', 0)
                    }
                    tprint_performance(f"📊 VectorBT Performance: {vectorbt_stats['vectorbt_operations']} operations, "
                                     f"{vectorbt_stats['vectorbt_usage_rate']:.1%} usage rate")

                # Log final performance
                tprint_structured({
                    "analysis_complete": datetime.now().isoformat(),
                    "total_time_seconds": round(total_time, 2),
                    "success_count": self.performance_metrics["success_count"],
                    "error_count": self.performance_metrics["error_count"],
                    "memory_usage_end": get_memory_usage() / (1024**2),
                    "vectorbt_enabled": self.enable_vectorbt,
                    "vectorbt_stats": vectorbt_stats,
                    "output_path": str(output_path)
                })

                tprint_success(f"Regime analysis completed and saved to {output_path}")
                print_analysis_summary(analysis)
                return analysis

            except ValueError as exc:
                # Handle missing features error specifically
                self.performance_metrics["error_count"] += 1

                # Log specific error for missing features
                tprint_error(f"🚨 Regime analysis failed for {symbol}: Missing features in regime_assignments file")
                tprint_error(f"   Error: {exc}")

                # Create detailed error result for missing features
                return {
                    "success": False,
                    "error": {
                        "type": "MissingFeaturesError",
                        "message": str(exc),
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "solution": "Re-run clustering step to generate features: python3 src/launcher/ares_launcher.py step05 nas_tas_clustering"
                    },
                    "performance_metrics": self.performance_metrics
                }

            except Exception as exc:
                self.performance_metrics["error_count"] += 1
                error_traceback = traceback.format_exc()

                # Log comprehensive error details
                tprint_error(f"Regime analysis failed for {symbol}: {exc}")
                tprint_debug(f"Error traceback: {error_traceback}")

                # Log structured error information
                error_info = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "symbol": symbol,
                    "error_count": self.performance_metrics["error_count"],
                    "success_count": self.performance_metrics["success_count"],
                    "traceback": error_traceback
                }
                tprint_structured(error_info)

                # Create detailed error result
                return {
                    "success": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": error_traceback,
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    },
                    "performance_metrics": self.performance_metrics
                }
            finally:
                # Cleanup and final monitoring
                if self.memory_optimizer:
                    self.memory_optimizer.stop_monitoring()

                # Log final performance summary
                self._log_performance_summary()

    def _load_datasets(self, symbol: str) -> Tuple[Any, ...]:
        try:
            return load_regime_datasets(self.data_cache_path, symbol)
        except Exception as exc:  # pragma: no cover - error surface for CLI
            log_warning(f"Failed to load regime datasets: {exc}")
            raise

    def _print_initial_overview(self, nas_labels, tas_labels) -> None:
        tprint("\n" + "=" * 80, "INFO")
        tprint("📊 REGIME ANALYSIS - INITIAL OVERVIEW", "INFO")
        tprint("=" * 80, "INFO")
        tprint(
            f"🔬 NAS regimes: {len(set(nas_labels))} {sorted(set(int(label) for label in nas_labels))}",
            "INFO",
        )
        tprint(
            f"🎯 TAS regimes: {len(set(tas_labels))} {sorted(set(int(label) for label in tas_labels))}",
            "INFO",
        )
        tprint("=" * 80, "INFO")

    def _compile_analysis(
        self,
        symbol: str,
        nas_distribution: Dict[str, Any],
        tas_distribution: Dict[str, Any],
        nas_metrics: Dict[str, Any],
        tas_metrics: Dict[str, Any],
        nas_labels,
        tas_labels,
    ) -> Dict[str, Any]:
        analysis_timestamp = datetime.now().isoformat()
        return {
            "symbol": symbol,
            "analysis_timestamp": analysis_timestamp,
            "nas_analysis": {
                "distribution": nas_distribution,
                "clustering_metrics": nas_metrics,
            },
            "tas_analysis": {
                "distribution": tas_distribution,
                "clustering_metrics": tas_metrics,
            },
            "summary": {
                "nas_regimes": len(set(nas_labels)),
                "tas_regimes": len(set(tas_labels)),
                "nas_samples": len(nas_labels),
                "tas_samples": len(tas_labels),
            },
        }

    def _save_analysis(self, analysis: Dict[str, Any], symbol: str) -> Path:
        """Save analysis with enhanced error handling and monitoring."""
        try:
            output_dir = Path("regime_analysis_results")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{symbol}_regime_analysis_{timestamp}.json"

            # Add performance metrics to analysis
            analysis["performance_metrics"] = {
                "total_time_seconds": self.performance_metrics.get("end_time", 0) - self.performance_metrics.get("start_time", 0),
                "success_count": self.performance_metrics.get("success_count", 0),
                "error_count": self.performance_metrics.get("error_count", 0),
                "memory_usage_mb": get_memory_usage() / (1024**2),
                "m1_hardware_available": M1_HARDWARE_AVAILABLE,
                "vectorbt_available": VECTORBT_AVAILABLE,
                "vectorbt_enabled": self.enable_vectorbt,
                "vectorbt_rolling_optimizer": self.rolling_optimizer is not None,
                "unified_vectorization_manager": self.vectorization_manager is not None
            }

            # Use safe JSON dump
            if safe_json_dump(analysis, output_path, indent=2):
                tprint_success(f"Analysis saved successfully to {output_path}")
                return output_path
            else:
                raise Exception("Failed to save analysis JSON")

        except Exception as exc:
            tprint_error(f"Failed to save analysis: {exc}")
            raise

    def _log_performance_summary(self) -> None:
        """Log comprehensive performance summary."""
        try:
            total_time = 0
            if self.performance_metrics["start_time"] and self.performance_metrics["end_time"]:
                total_time = self.performance_metrics["end_time"] - self.performance_metrics["start_time"]

            # Validate performance metrics using math_validation
            success_count = validate_positive(self.performance_metrics["success_count"], "success_count")
            error_count = validate_positive(self.performance_metrics["error_count"], "error_count")
            total_time = validate_finite(total_time, "total_time")

            performance_summary = {
                "total_time_seconds": round(validate_finite(total_time, "total_time_rounded"), 2),
                "success_count": int(success_count),
                "error_count": int(error_count),
                "success_rate": safe_divide(
                    success_count,
                    success_count + error_count,
                    default=0.0
                ),
                "memory_usage_mb": validate_finite(get_memory_usage() / (1024**2), "memory_usage_mb"),
                "m1_optimizations_used": M1_HARDWARE_AVAILABLE
            }

            tprint_structured({
                "performance_summary": performance_summary,
                "service": "RegimeAnalysisService"
            })

        except Exception as exc:
            tprint_warning(f"Failed to log performance summary: {exc}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "memory_usage": [],
            "processing_times": {},
            "error_count": 0,
            "success_count": 0
        }
        tprint_info("Performance metrics reset")

    def _validate_analysis_inputs(self, symbol: str) -> None:
        """Validate inputs for analysis method."""
        try:
            # Validate symbol
            if not isinstance(symbol, str):
                raise ValueError(f"Symbol must be a string, got {type(symbol)}")

            if not symbol or symbol.strip() == "":
                raise ValueError("Symbol cannot be empty")

            # Validate symbol format (basic check)
            if len(symbol) < 3 or len(symbol) > 20:
                raise ValueError(f"Symbol length invalid: {len(symbol)} (expected 3-20)")

            # Check for valid characters (alphanumeric only)
            # Remove common trading pair suffixes and check if remaining part is alphanumeric
            cleaned_symbol = symbol.replace("USDT", "").replace("USD", "").replace("BTC", "").replace("ETH", "")
            if cleaned_symbol and not cleaned_symbol.isalnum():
                raise ValueError(f"Symbol contains invalid characters: {symbol}")

            # Validate data cache path
            if not self.data_cache_path.exists():
                raise FileNotFoundError(f"Data cache directory not found: {self.data_cache_path}")

            if not self.data_cache_path.is_dir():
                raise ValueError(f"Data cache path is not a directory: {self.data_cache_path}")

            # Check if we have read permissions
            if not os.access(self.data_cache_path, os.R_OK):
                raise PermissionError(f"No read access to data cache directory: {self.data_cache_path}")

            tprint("Analysis input validation passed", "SUCCESS")

        except Exception as e:
            tprint_error(f"Analysis input validation failed: {e}")
            raise ValueError(f"Analysis input validation failed: {e}") from e

    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
        except Exception:
            pass  # Silently handle cleanup errors
