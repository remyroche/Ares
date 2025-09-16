"""
Market Analysis Utilities

This module provides utility functions and classes for market analysis operations,
integrating with the existing utility infrastructure and providing specialized
functions for triple barrier labeling and market data processing.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path
import warnings
from functools import partial
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import common utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_dataframe_operation, validate_dataframe_columns,
    create_summary_statistics, safe_convert_dtypes,
    optimize_dataframe_dtypes, safe_timestamp_conversion,
    safe_to_parquet, safe_read_parquet, list_parquet_files
)
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import MathValidation
from src.utils.serialization_utils import UniversalSerializer
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Import ML common utilities
from src.utils.ml_common.data_processing.data_labeling import EnhancedDataLabeler
from src.utils.ml_common.validation.cv_utils import TemporalCrossValidator, PurgedKFold

# Import hardware optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class MarketAnalysisUtils:
    """Utility class for market analysis operations."""
    
    def __init__(self):
        """Initialize market analysis utilities."""
        self.logger = logging.getLogger(f"{__name__}.MarketAnalysisUtils")
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ MarketAnalysisUtils initialized successfully")

    def _initialize_components(self):
        """Initialize utility components."""
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
            self.klines_manager = KlinesParquetManager()
            self.matrix_ops = UnifiedMatrixOperations()
            
            # Initialize enhanced data labeler
            self.enhanced_labeler = EnhancedDataLabeler()
            
            # Initialize hardware optimizations if available
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                try:
                    self.gpu_manager = get_m1_gpu_manager()
                    self.memory_optimizer = get_m1_memory_optimizer()
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    self.logger.info("✅ Hardware optimization utilities initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Hardware optimization failed: {e}")
                    self.gpu_manager = None
                    self.memory_optimizer = None
                    self.cpu_optimizer = None
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.logger.info("ℹ️ Hardware optimization not available")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def load_market_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        data_type: str = "raw"
    ) -> Optional[pd.DataFrame]:
        """Load market data for analysis.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            interval: Data interval (e.g., '1m', '5m', '1h')
            start_date: Start date for filtering
            end_date: End date for filtering
            data_type: 'raw' or 'processed'
            
        Returns:
            DataFrame with market data or None if not found
        """
        try:
            self.logger.info(f"📊 Loading market data: {symbol} {interval}")
            
            data = self.klines_manager.read_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type
            )
            
            if data is not None:
                self.logger.info(f"✅ Loaded {len(data)} records for {symbol} {interval}")
                return data
            else:
                self.logger.warning(f"⚠️ No data found for {symbol} {interval}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load market data: {e}")
            return None

    def validate_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data for analysis.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'data_quality': {}
            }
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check data size
            if len(data) < 10:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Insufficient data: {len(data)} rows (minimum 10 required)")
            
            # Check for null values
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                validation_result['warnings'].append(f"Null values found: {null_counts.to_dict()}")
            
            # Check price ranges
            for col in required_columns:
                if col in data.columns:
                    col_min = data[col].min()
                    col_max = data[col].max()
                    if col_min <= 0:
                        validation_result['warnings'].append(f"Non-positive values in {col}: min={col_min}")
                    if col_max / col_min > 1000:
                        validation_result['warnings'].append(f"Large price range in {col}: {col_min:.4f} - {col_max:.4f}")
            
            # Calculate data quality metrics
            validation_result['data_quality'] = self.common_utils.calculate_data_quality_metrics(data)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate market data: {e}")
            return {
                'is_valid': False,
                'warnings': [],
                'errors': [str(e)],
                'data_quality': {}
            }

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using feature_engineering module.
        
        Args:
            data: Market data with OHLC columns
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            # Import feature engineering generators
            from src.feature_engineering.feature_generators import (
                rsi_generator, macd_generator, bollinger_bands_generator,
                sma_generator, ema_generator
            )
            
            result = data.copy()
            
            # Use feature_engineering generators for technical indicators
            # RSI
            result['rsi_14'] = rsi_generator(data, lookback=14, price_column='close')
            
            # MACD
            result['macd'] = macd_generator(data, lookback=26, price_column='close')
            
            # Bollinger Bands
            result['bb_position'] = bollinger_bands_generator(data, lookback=20, price_column='close')
            
            # Moving averages
            result['sma_10'] = sma_generator(data, lookback=10, price_column='close')
            result['sma_20'] = sma_generator(data, lookback=20, price_column='close')
            result['sma_50'] = sma_generator(data, lookback=50, price_column='close')
            
            # EMAs
            result['ema_12'] = ema_generator(data, lookback=12, price_column='close')
            result['ema_26'] = ema_generator(data, lookback=26, price_column='close')
            
            # Basic price indicators
            result['returns'] = data['close'].pct_change()
            result['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            result['volatility'] = result['returns'].rolling(window=20).std()
            
            # Volume indicators (if available)
            if 'volume' in data.columns:
                result['volume_ma'] = data['volume'].rolling(window=20).mean()
                result['volume_ratio'] = data['volume'] / result['volume_ma']
                result['price_volume'] = data['close'] * data['volume']
            
            # Fill NaN values
            result = result.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"✅ Calculated technical indicators using feature_engineering for {len(result)} records")
            return result
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Feature engineering not available: {e}")
            self.logger.info("💡 Falling back to basic price indicators only")
            
            # Fallback to basic indicators
            result = data.copy()
            result['returns'] = data['close'].pct_change()
            result['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            result['volatility'] = result['returns'].rolling(window=20).std()
            result = result.fillna(method='ffill').fillna(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate technical indicators: {e}")
            return data

    # Note: RSI calculation now uses feature_engineering module

    def detect_market_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using HMM state only.
        
        Args:
            data: Market data with technical indicators
            
        Returns:
            DataFrame with regime information
        """
        try:
            # Check if HMM regime data already exists in the data
            hmm_columns = ['hmm_regime', 'composite_cluster_id', 'regime']
            existing_regime_col = None
            
            for col in hmm_columns:
                if col in data.columns:
                    existing_regime_col = col
                    break
            
            if existing_regime_col:
                self.logger.info(f"✅ Using existing HMM regime column: {existing_regime_col}")
                return pd.DataFrame({
                    'regime': data[existing_regime_col]
                }, index=data.index)
            
            # If no HMM regime data exists, warn user
            self.logger.warning("⚠️ No HMM regime data found in input data")
            self.logger.info("💡 HMM regime data should be provided by the pipeline (step03_hmm_regime_discovery)")
            
            # Create default regime assignment
            return self._create_default_regimes(data)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to detect market regimes: {e}")
            return pd.DataFrame({'regime': ['unknown'] * len(data)}, index=data.index)

    # Note: Only HMM-based regime detection is used
    # Other methods (volatility, trend) have been removed as per requirements

    def _create_default_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create default regime assignment when HMM data is not available."""
        self.logger.warning("⚠️ Using default regime assignment")
        
        # Simple alternating regime assignment
        regimes = []
        for i in range(len(data)):
            if i % 100 < 50:  # Alternate every 100 periods
                regimes.append('bull_market')
            else:
                regimes.append('bear_market')
        
        return pd.DataFrame({'regime': regimes}, index=data.index)

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        try:
            return optimize_dataframe_dtypes(df)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to optimize DataFrame memory: {e}")
            return df

    def save_analysis_results(
        self, 
        data: pd.DataFrame, 
        filepath: str, 
        format: str = "parquet"
    ) -> bool:
        """Save analysis results to file.
        
        Args:
            data: DataFrame to save
            filepath: Path to save file
            format: File format ('parquet', 'csv', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == "parquet":
                return safe_to_parquet(data, filepath)
            elif format == "csv":
                data.to_csv(filepath, index=True)
                return True
            elif format == "json":
                data.to_json(filepath, orient='records', date_format='iso')
                return True
            else:
                self.logger.error(f"❌ Unsupported format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis results: {e}")
            return False

    def load_analysis_results(self, filepath: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """Load analysis results from file.
        
        Args:
            filepath: Path to file
            format: File format ('parquet', 'csv', 'json')
            
        Returns:
            DataFrame or None if failed
        """
        try:
            if format == "parquet":
                return safe_read_parquet(filepath)
            elif format == "csv":
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format == "json":
                return pd.read_json(filepath, orient='records', date_format='iso')
            else:
                self.logger.error(f"❌ Unsupported format: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load analysis results: {e}")
            return None

    def create_analysis_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive analysis summary.
        
        Args:
            data: Analysis data
            
        Returns:
            Dictionary with analysis summary
        """
        try:
            summary = {
                'basic_info': {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                'data_quality': self.common_utils.calculate_data_quality_metrics(data),
                'statistical_summary': create_summary_statistics(data),
                'timestamp_range': {
                    'start': data.index.min() if hasattr(data.index, 'min') else None,
                    'end': data.index.max() if hasattr(data.index, 'max') else None
                }
            }
            
            # Add label-specific analysis if labels are present
            if 'label' in data.columns:
                label_counts = data['label'].value_counts()
                summary['label_analysis'] = {
                    'label_counts': label_counts.to_dict(),
                    'label_distribution': {
                        'positive': safe_divide(label_counts.get(1, 0), len(data)),
                        'negative': safe_divide(label_counts.get(-1, 0), len(data)),
                        'neutral': safe_divide(label_counts.get(0, 0), len(data))
                    }
                }
            
            # Add regime analysis if regimes are present
            if 'regime' in data.columns:
                regime_counts = data['regime'].value_counts()
                summary['regime_analysis'] = {
                    'regime_counts': regime_counts.to_dict(),
                    'regime_distribution': {
                        regime: safe_divide(count, len(data)) 
                        for regime, count in regime_counts.items()
                    }
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create analysis summary: {e}")
            return {'error': str(e)}

    def get_available_data(self) -> Dict[str, List[str]]:
        """Get information about available market data.
        
        Returns:
            Dictionary mapping symbols to available intervals
        """
        try:
            return self.klines_manager.list_available_data()
        except Exception as e:
            self.logger.error(f"❌ Failed to get available data: {e}")
            return {}

    def cleanup_resources(self):
        """Clean up resources and optimize memory."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("✅ Resources cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup resources: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            stats = {
                'hardware_optimization_available': HARDWARE_OPTIMIZATION_AVAILABLE,
                'gpu_manager_available': self.gpu_manager is not None,
                'memory_optimizer_available': self.memory_optimizer is not None,
                'cpu_optimizer_available': self.cpu_optimizer is not None
            }
            
            if self.memory_optimizer:
                try:
                    memory_stats = self.memory_optimizer.get_memory_stats()
                    stats['memory_stats'] = memory_stats
                except Exception:
                    pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance stats: {e}")
            return {'error': str(e)}

# Global instance for convenience
market_analysis_utils = MarketAnalysisUtils()

# Convenience functions
def load_market_data(symbol: str, interval: str, **kwargs) -> Optional[pd.DataFrame]:
    """Load market data (convenience function)."""
    return market_analysis_utils.load_market_data(symbol, interval, **kwargs)

def validate_market_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate market data (convenience function)."""
    return market_analysis_utils.validate_market_data(data)

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators (convenience function)."""
    return market_analysis_utils.calculate_technical_indicators(data)

def detect_market_regimes(data: pd.DataFrame, method: str = "volatility") -> pd.DataFrame:
    """Detect market regimes (convenience function)."""
    return market_analysis_utils.detect_market_regimes(data, method)