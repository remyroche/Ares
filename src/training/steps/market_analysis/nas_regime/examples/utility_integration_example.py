"""
Enhanced NAS Regime Detection with Complete Utility Integration Example

This example demonstrates the comprehensive integration of all utility tools:
- common_operations.py: DataFrame operations, validation, logging, hardware optimization
- common_utilities.py: Shared utility functions and common operations
- math_validation.py: Safe mathematical operations and validation
- kline_parquet.py: Data management and serialization utilities
- matrix_operations: Optimized computations
- ML common utilities: Enhanced validation, CV, optimization, and monitoring
- Hardware optimization: M1 GPU, CPU, and memory optimization

This example shows how to use all these tools together for production-ready NAS regime detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from typing import Dict, Any

# Import all enhanced utility tools
from src.utils.common_operations import (
    # DataFrame utilities
    create_empty_dataframe, validate_dataframe, validate_dataframe_columns,
    safe_dataframe_operation, safe_fillna, safe_convert_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    validate_timestamp_column, safe_timestamp_conversion, optimize_dataframe_dtypes,
    # Data quality utilities
    calculate_data_quality_metrics, get_dataframe_info, create_data_quality_report,
    # Math utilities
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    # String utilities
    safe_lower, safe_upper, safe_join,
    # Collection utilities
    safe_append, safe_extend, safe_dict_get, safe_dict_items,
    # Async utilities
    safe_sleep, safe_gather, create_async_task,
    # Performance utilities
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    # Matrix utilities
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    # File utilities
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    get_latest_outcome_file, load_latest_optimal_regime_clustering_outcome,
    safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls,
    # Memory optimization utilities
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space
)

from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    calculate_data_quality_metrics as calc_data_quality, create_summary_statistics,
    safe_convert_dtypes as safe_convert_dtypes_util, safe_merge_dataframes as safe_merge_dfs,
    safe_groupby_operation, safe_apply_function, safe_drop_columns as safe_drop_cols,
    safe_rename_columns as safe_rename_cols, validate_timestamp_column as validate_ts_col,
    safe_timestamp_conversion as safe_ts_conversion, get_dataframe_info as get_df_info,
    safe_filter_dataframe, create_data_quality_report as create_dq_report,
    CommonUtilities
)

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log, safe_sqrt as math_safe_sqrt,
    safe_power as math_safe_power, validate_finite as math_validate_finite,
    validate_positive as math_validate_positive, validate_range as math_validate_range,
    validate_numeric_array, safe_kelly_calculation as math_safe_kelly,
    safe_weighted_average as math_safe_weighted_avg, safe_percentage_change as math_safe_pct_change,
    safe_correlation, safe_covariance, safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile, validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inverse, math_safe, MathValidation
)

from src.utils.data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Import NAS Regime components from centralized utilities
from src.utils.nas_tas.core.nas_engine import NASEngine
from src.utils.nas_tas.optimization.architecture_search import ArchitectureSearchOptimizer, ArchitectureSearchConfig

from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    EnhancedPerfectNASRegimeDetector, PerfectNASConfig, NeuralArchitectureType
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilityIntegrationExample:
    """Example demonstrating complete utility integration for NAS regime detection."""

    def __init__(self):
        """Initialize the example with all utility tools."""
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.universal_serializer = UniversalSerializer()
        self.klines_manager = get_klines_manager()
        self.matrix_ops = UnifiedMatrixOperations(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True
        )

        logger.info("✅ All utility tools initialized for example")

    @timed_operation
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample market data using enhanced utilities."""
        logger.info(f"📊 Generating sample market data with {n_samples} samples...")

        # Create timestamp range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_samples)

        # Generate realistic market data
        np.random.seed(42)

        # Base price movements
        price_changes = np.random.randn(n_samples) * 0.02
        prices = 100.0 * np.exp(np.cumsum(price_changes))

        # Create OHLC data with realistic spreads
        spreads = np.random.uniform(0.001, 0.005, n_samples)
        highs = prices * (1 + spreads)
        lows = prices * (1 - spreads)
        opens = prices * (1 + np.random.randn(n_samples) * 0.002)
        closes = prices * (1 + np.random.randn(n_samples) * 0.002)

        # Volume data
        volumes = np.random.uniform(1000, 10000, n_samples)

        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        # Validate data quality
        data_quality = calculate_data_quality_metrics(data)
        logger.info(f"📊 Generated data quality: {data_quality}")

        # Optimize data types
        data = optimize_dataframe_dtypes(data)

        return data

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality using enhanced utilities."""
        logger.info("🔍 Validating data quality...")

        # Generate comprehensive data quality report
        report = create_data_quality_report(data)

        # Log key metrics
        quality = report['quality_metrics']
        logger.info(f"📊 Data Quality Metrics:")
        logger.info(f"   Total rows: {quality['total_rows']}")
        logger.info(f"   Total columns: {quality['total_columns']}")
        logger.info(f"   Missing values: {quality['missing_values']} ({quality['missing_percentage']:.2f}%)")
        logger.info(f"   Duplicate rows: {quality['duplicate_rows']} ({quality['duplicate_percentage']:.2f}%)")

        if report['issues']:
            logger.warning(f"⚠️ Data quality issues found: {report['issues']}")
        else:
            logger.info("✅ No data quality issues found")

        return report

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data using enhanced utilities."""
        logger.info("🔧 Preprocessing data with enhanced utilities...")

        # Guard against nulls
        data = guard_dataframe_nulls(data, threshold=0.1)

        # Optimize data types
        data = optimize_dataframe_dtypes(data)

        # Add technical indicators using safe math operations
        def calculate_sma(series: pd.Series, window: int) -> pd.Series:
            return safe_dataframe_operation(series, lambda x: self._vectorbt_rolling_operation(x, "mean", window))

        def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = safe_divide(gain, loss, default=0.0)
            return 100 - (100 / (1 + rs))

        # Add indicators with validation
        data['sma_20'] = calculate_sma(data['close'], 20)
        data['sma_50'] = calculate_sma(data['close'], 50)
        data['rsi_14'] = calculate_rsi(data['close'], 14)

        # Validate calculations
        for col in ['sma_20', 'sma_50', 'rsi_14']:
            if col in data.columns:
                math_validate_finite(data[col], f"indicator_{col}")

        logger.info(f"✅ Data preprocessing completed. Shape: {data.shape}")
        return data

    def save_and_load_data(self, data: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Demonstrate data serialization utilities."""
        logger.info(f"💾 Saving data with enhanced serialization utilities to {filename}...")

        # Ensure directory exists
        ensure_directory("data_cache")

        # Use universal serializer
        filepath = f"data_cache/{filename}"
        success = self.universal_serializer.save(data, filepath)

        if success:
            logger.info("✅ Data saved with universal serializer")
        else:
            # Fallback to parquet
            safe_to_parquet(data, filepath, compression='snappy')
            logger.info("⚠️ Universal serializer failed, used parquet fallback")

        # Load data back
        loaded_data = self.universal_serializer.load(filepath)
        if loaded_data is None:
            # Fallback to parquet reading
            loaded_data = safe_read_parquet(filepath)

        logger.info(f"✅ Data loaded successfully. Shape: {loaded_data.shape}")
        return loaded_data

    def demonstrate_memory_optimization(self):
        """Demonstrate memory optimization utilities."""
        logger.info("🧠 Demonstrating memory optimization utilities...")

        # Check initial memory usage
        initial_memory = get_memory_usage()
        logger.info(f"📈 Initial memory usage: {format_bytes(initial_memory)}")

        # Create memory checkpoint
        with memory_checkpoint("Example_Operation"):
            # Simulate memory-intensive operation
            large_array = np.random.randn(100000, 100)
            large_array = validate_numeric_array(large_array, "large_array")

            # Use matrix operations for memory-efficient computation
            if self.matrix_ops:
                result = self.matrix_ops.safe_matrix_multiply(large_array, large_array.T)
                logger.info("✅ Matrix multiplication completed with optimization")

        # Check memory after operation
        final_memory = get_memory_usage()
        logger.info(f"📈 Memory after operation: {format_bytes(final_memory)}")

        # Optimize memory
        memory_status = optimize_memory()
        if memory_status['success']:
            logger.info(f"✅ Memory optimization completed: {memory_status.get('method', 'unknown')}")
        else:
            logger.warning(f"⚠️ Memory optimization failed: {memory_status.get('error', 'unknown')}")

    def run_enhanced_nas_search(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run enhanced NAS search with all utility integrations."""
        logger.info("🚀 Running enhanced NAS search with full utility integration...")

        # Prepare data for NAS
        X = data[['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'rsi_14']].values
        y = (data['close'].shift(-1) > data['close']).astype(int).values[:-1]  # Next period up/down

        # Ensure we have enough data
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]

        if len(X) < 100:
            logger.warning("⚠️ Not enough data for NAS search")
            return {'error': 'Insufficient data'}

        # Create NAS engine configuration
        nas_config = ArchitectureSearchConfig(
            max_iterations=100,
            population_size=20,
            enable_parallel_processing=True,
            max_workers=4
        )

        # Create and run NAS engine
        nas_engine = NASEngine()
        architecture_optimizer = ArchitectureSearchOptimizer(nas_config)

        # Split data for training/validation
        split_idx = int(0.8 * len(X))
        train_data = (X[:split_idx], y[:split_idx])
        val_data = (X[split_idx:], y[split_idx:])

        # Run NAS search
        result = nas_engine.search(train_data, val_data)

        logger.info(f"✅ NAS search completed - Best score: {result.best_score:.4f}")
        logger.info(f"   Architecture evaluations: {result.n_evaluations}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")

        return {
            'best_score': result.best_score,
            'n_evaluations': result.n_evaluations,
            'execution_time': result.execution_time,
            'search_strategy': result.strategy_used,
            'metadata': result.metadata
        }

    def run_perfect_nas_regime_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run enhanced perfect NAS regime detection."""
        logger.info("🚀 Running enhanced perfect NAS regime detection...")

        # Create detector configuration
        detector_config = PerfectNASConfig(
            n_regimes=3,
            architecture_type=NeuralArchitectureType.HYBRID,
            enable_meta_learning=True,
            enable_optimization=True,
            max_training_time=600,
            memory_limit_mb=4096
        )

        # Create detector
        detector = EnhancedPerfectNASRegimeDetector(detector_config)

        # Run regime detection
        result = detector.detect_regimes(data)

        logger.info("✅ Perfect NAS regime detection completed")
        logger.info(f"   Detected regimes: {len(result.regime_labels) if result.regime_labels else 0}")
        logger.info(f"   Confidence: {result.confidence_score:.4f}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")

        return {
            'n_regimes': len(result.regime_labels) if result.regime_labels else 0,
            'confidence': result.confidence_score,
            'execution_time': result.execution_time,
            'regime_labels': result.regime_labels,
            'regime_probabilities': result.regime_probabilities,
            'hardware_metrics': result.hardware_metrics,
            'ml_common_metrics': result.ml_common_metrics
        }

    def demonstrate_kline_data_management(self):
        """Demonstrate kline data management utilities."""
        logger.info("📈 Demonstrating kline data management utilities...")

        # Generate sample kline data
        sample_data = self.generate_sample_data(500)

        # Validate kline data
        validation_result = validate_klines_data(sample_data)
        if validation_result['valid']:
            logger.info("✅ Klines data validation passed")
        else:
            logger.warning(f"⚠️ Klines validation issues: {validation_result['errors']}")

        # Save using kline manager
        success = save_klines_to_parquet(
            sample_data,
            symbol="EXAMPLE",
            interval="1m",
            data_type="raw",
            overwrite=True
        )

        if success:
            logger.info("✅ Sample data saved using kline manager")
        else:
            logger.error("❌ Failed to save data with kline manager")

        # Load data back
        loaded_data = load_klines_from_parquet("EXAMPLE", "1m", data_type="raw")

        if loaded_data is not None:
            logger.info(f"✅ Data loaded successfully: {loaded_data.shape}")
        else:
            logger.warning("⚠️ Failed to load data")

        return loaded_data

    def run_complete_example(self):
        """Run the complete utility integration example."""
        logger.info("🚀 Starting complete utility integration example...")

        try:
            # Step 1: Generate and validate data
            logger.info("\n=== Step 1: Data Generation and Quality Validation ===")
            data = self.generate_sample_data(1000)
            quality_report = self.validate_data_quality(data)

            # Step 2: Preprocess data
            logger.info("\n=== Step 2: Data Preprocessing ===")
            processed_data = self.preprocess_data(data)

            # Step 3: Save and load data
            logger.info("\n=== Step 3: Data Serialization ===")
            reloaded_data = self.save_and_load_data(processed_data, "example_market_data.parquet")

            # Step 4: Memory optimization demonstration
            logger.info("\n=== Step 4: Memory Optimization ===")
            self.demonstrate_memory_optimization()

            # Step 5: Enhanced NAS search
            logger.info("\n=== Step 5: Enhanced NAS Search ===")
            nas_results = self.run_enhanced_nas_search(reloaded_data)

            # Step 6: Perfect NAS regime detection
            logger.info("\n=== Step 6: Perfect NAS Regime Detection ===")
            regime_results = self.run_perfect_nas_regime_detection(reloaded_data)

            # Step 7: Klines data management
            logger.info("\n=== Step 7: Klines Data Management ===")
            kline_data = self.demonstrate_kline_data_management()

            # Step 8: Final validation and summary
            logger.info("\n=== Step 8: Final Summary ===")

            # Create comprehensive summary
            summary = {
                'data_quality': quality_report,
                'nas_results': nas_results,
                'regime_detection': regime_results,
                'kline_validation': validate_klines_data(kline_data) if kline_data is not None else None,
                'memory_usage': get_memory_usage(),
                'disk_status': check_disk_space(".", required_gb=0.1)
            }

            # Save summary using universal serializer
            self.universal_serializer.save(summary, "data_cache/example_summary.json")
            logger.info("✅ Complete example summary saved")

            # Log final metrics
            logger.info("📊 Final Integration Summary:")
            logger.info(f"   Memory usage: {format_bytes(summary['memory_usage'])}")
            if summary['disk_status']:
                logger.info(f"   Disk available: {summary['disk_status']['available_percentage']:.1f}%")
            logger.info(f"   NAS best score: {summary['nas_results'].get('best_score', 'N/A')}")
            logger.info(f"   Regime detection confidence: {summary['regime_detection'].get('confidence', 'N/A')}")

            logger.info("🎉 Complete utility integration example finished successfully!")

            return summary

        except Exception as e:
            logger.error(f"❌ Example failed: {e}")
            return {'error': str(e)}


def main():
    """Run the complete utility integration example."""
    example = UtilityIntegrationExample()
    results = example.run_complete_example()
    return results


if __name__ == "__main__":
    results = main()
    print(f"\nExample completed with results: {results}")