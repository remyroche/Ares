"""Validation Strategies Component

Contains validation strategy classes for different types of data validation.
Extracted from raw_data_quality_checker.py
"""

from datetime import timedelta
from typing import Any, Optional, List
import pandas as pd
import logging
import numpy as np

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

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

class ValidationStrategy:
    """Base class for validation strategies."""
    
    @log_important_calls
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild(self.__class__.__name__)
        
    def validate(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate data according to this strategy.
        
        Args:
            data: DataFrame to validate
            results: Results dictionary to update
            
        Returns:
            True if validation passed, False otherwise
        """
        # Base implementation - can be overridden by subclasses
        self.logger.info(f'Running base validation strategy: {self.__class__.__name__}')
        
        # Basic validation that all strategies should perform
        if data is None:
            results['critical_issues'].append('Data is None')
            return False
            
        if data.empty:
            results['critical_issues'].append('Data is empty')
            return False
            
        # Check if results dictionary has required structure
        if 'critical_issues' not in results:
            results['critical_issues'] = []
        if 'warnings' not in results:
            results['warnings'] = []
        if 'detailed_analysis' not in results:
            results['detailed_analysis'] = {}
            
        return True

class StructureValidationStrategy(ValidationStrategy):
    """Validates data structure and basic requirements."""
    
    def validate(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate basic data structure and required columns.
        
        Args:
            data: DataFrame to validate
            results: Results dictionary to update
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info('Validating data structure...')
        
        if data.empty:
            results['critical_issues'].append('Empty dataset provided')
            return False
            
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            results['critical_issues'].append(f'Missing required columns: {missing_columns}')
            return False
            
        # Check minimum records
        min_records = self.config.get('min_records', 1000)
        if len(data) < min_records:
            results['critical_issues'].append(f'Insufficient data: {len(data)} records (minimum: {min_records})')
            return False
            
        # Check datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning('⚠️ Data does not have datetime index, attempting to fix...')
            fixed = self._fix_datetime_index(data, results)
            if fixed is None:
                results['critical_issues'].append('Failed to create datetime index from data')
                return False
            self.logger.info('✅ Successfully created datetime index')
            results['warnings'].append('Created datetime index from data')
            data = fixed
            
        # Check for duplicate timestamps
        duplicate_ratio = data.index.duplicated().sum() / len(data)
        max_duplicates = self.config.get('max_duplicate_timestamps', 0.0005)
        if duplicate_ratio > max_duplicates:
            results['warnings'].append(f'High duplicate timestamps: {duplicate_ratio:.3f} (threshold: {max_duplicates})')
            
        # Store structure analysis
        results['detailed_analysis']['structure'] = {
            'total_records': len(data),
            'date_range': f'{data.index.min()} to {data.index.max()}',
            'duplicate_ratio': duplicate_ratio,
            'columns_present': list(data.columns)
        }
        
        return True
    @log_all_calls
        
    def _fix_datetime_index(self, data: pd.DataFrame, results: dict[str, Any]) -> pd.DataFrame | None:
        """Attempt to fix missing datetime index by creating one from available data."""
        try:
            self.logger.info('🔧 Attempting to create datetime index...')
            
            # Check for timestamp columns
            timestamp_columns = ['timestamp', 'time', 'date', 'datetime', 'index']
            for col in timestamp_columns:
                if col in data.columns:
                    self.logger.info(f'🔧 Found timestamp column: {col}')
                    try:
                        if data[col].dtype == 'object':
                            # Try different datetime formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f', 
                                      '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']:
                                try:
                                    timestamps = pd.to_datetime(data[col], format = fmt)
                                    if not timestamps.isna().all():
                                        fixed_data = data.copy()
                                        fixed_data.index = timestamps
                                        fixed_data = fixed_data.drop(columns=[col])
                                        self.logger.info(f'✅ Created datetime index from {col} using format {fmt}')
                                        return fixed_data
                                except Exception:
                                    continue
                        else:
                            timestamps = pd.to_datetime(data[col])
                            if not timestamps.isna().all():
                                fixed_data = data.copy()
                                fixed_data.index = timestamps
                                fixed_data = fixed_data.drop(columns=[col])
                                self.logger.info(f'✅ Created datetime index from {col}')
                                return fixed_data
                    except Exception as e:
                        self.logger.debug(f'⚠️ Failed to parse {col}: {e}')
                        continue
                        
            # Try to parse existing index
            try:
                if data.index.dtype == 'object':
                    timestamps = pd.to_datetime(data.index)
                    if not timestamps.isna().all():
                        fixed_data = data.copy()
                        fixed_data.index = timestamps
                        self.logger.info('✅ Created datetime index from existing index')
                        return fixed_data
            except Exception as e:
                self.logger.debug(f'⚠️ Failed to parse existing index: {e}')
                
            # Create synthetic datetime index as last resort
            self.logger.info('🔧 Creating synthetic datetime index...')
            timeframe = self._estimate_timeframe_from_data(data)
            self.logger.info(f'🔧 Estimated timeframe: {timeframe}')
            
            interval_map = {
                '1m': pd.Timedelta(minutes = 1),
                '5m': pd.Timedelta(minutes = 5),
                '15m': pd.Timedelta(minutes = 15),
                '30m': pd.Timedelta(minutes = 30),
                '1h': pd.Timedelta(hours = 1),
                '4h': pd.Timedelta(hours = 4),
                '1d': pd.Timedelta(days = 1)
            }
            interval = interval_map.get(timeframe, pd.Timedelta(minutes = 1))
            
            start_time = pd.Timestamp('2024-01-01 00:00:00')
            timestamps = [start_time + i * interval for i in range(len(data))]
            fixed_data = data.copy()
            fixed_data.index = timestamps
            self.logger.info(f'✅ Created synthetic datetime index with {timeframe} intervals')
            results['warnings'].append(f'Created synthetic datetime index with {timeframe} intervals - verify data alignment')
            return fixed_data
            
        except Exception as e:
            self.logger.exception(f'❌ Failed to create datetime index: {e}')
            return None
    @log_all_calls
            
    def _estimate_timeframe_from_data(self, data: pd.DataFrame) -> str:
        """Estimate the timeframe from data characteristics."""
        try:
            column_names = ' '.join(data.columns).lower()
            if any(tf in column_names for tf in ['1m', '1min', 'minute']):
                return '1m'
            elif any(tf in column_names for tf in ['5m', '5min']):
                return '5m'
            elif any(tf in column_names for tf in ['15m', '15min']):
                return '15m'
            elif any(tf in column_names for tf in ['30m', '30min']):
                return '30m'
            elif any(tf in column_names for tf in ['1h', 'hour']):
                return '1h'
            elif any(tf in column_names for tf in ['4h', '4hour']):
                return '4h'
            elif any(tf in column_names for tf in ['1d', 'day', 'daily']):
                return '1d'
            elif len(data) > 10000:
                return '1m'
            elif len(data) > 1000:
                return '5m'
            elif len(data) > 100:
                return '15m'
            else:
                return '1h'
        except Exception as e:
            self.logger.debug(f'⚠️ Error estimating timeframe: {e}')
            return '1m'

class CompletenessValidationStrategy(ValidationStrategy):
    """Validates data completeness and missing values."""
    
    def validate(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate data completeness and missing values.
        
        Args:
            data: DataFrame to validate
            results: Results dictionary to update
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info('Validating data completeness...')
        
        if data.empty:
            results['critical_issues'].append('Empty dataset provided')
            return False
            
        # Check OHLC missing values
        ohlc_columns = ['open', 'high', 'low', 'close']
        missing_ohlc = data[ohlc_columns].isnull().sum()
        missing_ohlc_ratio = missing_ohlc.sum() / (len(data) * len(ohlc_columns))
        max_missing_ohlc = self.config.get('max_missing_ohlc', 0.005)
        
        if missing_ohlc_ratio > max_missing_ohlc:
            results['critical_issues'].append(f'Too many missing OHLC values: {missing_ohlc_ratio:.3f} (threshold: {max_missing_ohlc})')
            return False
            
        # Check data span
        try:
            if len(data) == 0:
                data_span_days = 0
            elif data.index.min() == data.index.max():
                data_span_days = 0
            else:
                data_span_days = (data.index.max() - data.index.min()).days
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating data span: {e}')
            data_span_days = 0
            
        min_span_days = self.config.get('min_data_span_days', 7)
        if data_span_days < min_span_days:
            if data_span_days == 0:
                if len(data) == 0:
                    results['critical_issues'].append('Empty dataset provided')
                else:
                    results['critical_issues'].append(f'All data has the same timestamp: {data.index.min()}')
            else:
                results['critical_issues'].append(f'Insufficient data span: {data_span_days} days (minimum: {min_span_days})')
            return False
            
        # Check timestamp continuity
        if self.config.get('check_timestamp_continuity', True):
            time_diffs = data.index.to_series().diff().dropna()
            max_gap_hours = self.config.get('max_gap_hours', 1)
            large_gaps = time_diffs[time_diffs > timedelta(hours = max_gap_hours)]
            
            if len(large_gaps) > 0:
                results['warnings'].append(f'Found {len(large_gaps)} gaps larger than {max_gap_hours} hours')
                
        # Store completeness analysis
        results['detailed_analysis']['completeness'] = {
            'missing_ohlc_ratio': missing_ohlc_ratio,
            'data_span_days': data_span_days,
            'missing_by_column': missing_ohlc.to_dict(),
            'large_gaps_count': len(large_gaps) if 'large_gaps' in locals() else 0
        }
        
        return True

class IntegrityValidationStrategy(ValidationStrategy):
    """Validates data integrity and logical consistency."""
    
    def validate(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate data integrity and logical consistency.
        
        Args:
            data: DataFrame to validate
            results: Results dictionary to update
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info('Validating data integrity...')
        
        # Check OHLC consistency
        if self.config.get('check_ohlc_consistency', True):
            ohlc_inconsistent = (
                (data['high'] < data['low']) |
                (data['open'] > data['high']) |
                (data['close'] > data['high']) |
                (data['open'] < data['low']) |
                (data['close'] < data['low'])
            )
            ohlc_inconsistent_ratio = ohlc_inconsistent.sum() / len(data)
            
            if ohlc_inconsistent_ratio > 0:
                results['critical_issues'].append(f'OHLC inconsistency found: {ohlc_inconsistent_ratio:.3f} of records')
                return False
                
        # Check for negative prices
        negative_prices = (data[['open', 'high', 'low', 'close']] < 0).any(axis = 1)
        negative_price_ratio = negative_prices.sum() / len(data)
        max_negative = self.config.get('max_negative_prices', 0.0)
        
        if negative_price_ratio > max_negative:
            results['critical_issues'].append(f'Negative prices found: {negative_price_ratio:.3f} of records')
            return False
            
        # Check zero/negative volume
        zero_volume_ratio = (data['volume'] <= 0).sum() / len(data)
        max_zero_volume = self.config.get('max_zero_volume_ratio', 0.05)
        
        if zero_volume_ratio > max_zero_volume:
            results['warnings'].append(f'High zero/negative volume: {zero_volume_ratio:.3f} (threshold: {max_zero_volume})')
            
        # Check for extreme price movements
        price_changes = data['close'].pct_change().abs()
        extreme_moves = price_changes > 0.5
        extreme_move_ratio = extreme_moves.sum() / len(price_changes.dropna())
        
        if extreme_move_ratio > 0.001:
            results['warnings'].append(f'Extreme price movements detected: {extreme_move_ratio:.3f} of records')
            
        # Store integrity analysis
        results['detailed_analysis']['integrity'] = {
            'ohlc_inconsistent_ratio': ohlc_inconsistent_ratio if 'ohlc_inconsistent_ratio' in locals() else 0,
            'negative_price_ratio': negative_price_ratio,
            'zero_volume_ratio': zero_volume_ratio,
            'extreme_move_ratio': extreme_move_ratio
        }
        
        return True

class MarketSpecificValidationStrategy(ValidationStrategy):
    """Validates market-specific issues and anomalies."""
    
    def validate(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate market-specific issues and anomalies.
        
        Args:
            data: DataFrame to validate
            results: Results dictionary to update
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info('Validating market-specific issues...')
        
        # Check for market gaps (weekends/holidays)
        if self.config.get('check_for_market_gaps', True):
            time_diffs = data.index.to_series().diff().dropna()
            weekend_gaps = time_diffs[time_diffs > timedelta(hours = 48)]
            
            if len(weekend_gaps) > 0:
                results['warnings'].append(f'Detected {len(weekend_gaps)} potential market gaps (weekends/holidays)')
                
        # Check volume anomalies
        volume_mean = data['volume'].mean()
        volume_std = data['volume'].std()
        high_volume = data['volume'] > volume_mean + 3 * volume_std
        low_volume = data['volume'] < volume_mean - 3 * volume_std
        
        high_volume_ratio = high_volume.sum() / len(data)
        low_volume_ratio = low_volume.sum() / len(data)
        
        if high_volume_ratio > 0.02:
            results['warnings'].append(f'Unusual high volume periods: {high_volume_ratio:.3f} of records')
            
        if low_volume_ratio > 0.1:
            results['warnings'].append(f'Unusual low volume periods: {low_volume_ratio:.3f} of records')
            
        # Store market-specific analysis
        results['detailed_analysis']['market_specific'] = {
            'weekend_gaps_count': len(weekend_gaps) if 'weekend_gaps' in locals() else 0,
            'high_volume_ratio': high_volume_ratio,
            'low_volume_ratio': low_volume_ratio,
            'volume_statistics': {
                'mean': float(volume_mean),
                'std': float(volume_std),
                'min': float(data['volume'].min()),
                'max': float(data['volume'].max())
            }
        }
        
        return True

class FeatureEngineeringValidationStrategy(ValidationStrategy):
    """Validates data quality specifically for feature engineering requirements."""
    
    def validate(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate data quality specifically for feature engineering requirements.
        
        Args:
            data: DataFrame to validate
            results: Results dictionary to update
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info('Validating feature engineering requirements...')
        
        feature_eng_checks = self.config.get('feature_engineering_checks', {})
        
        # Check rolling window compatibility
        if feature_eng_checks.get('check_rolling_window_compatibility', True):
            min_rolling_periods = 50
            if len(data) < min_rolling_periods:
                results['warnings'].append('Insufficient data for rolling windows - consider longer lookback')
                
        # Check wavelet data requirements
        if feature_eng_checks.get('check_wavelet_data_requirements', True):
            time_diffs = data.index.to_series().diff().dropna()
            max_wavelet_gap = timedelta(hours = 6)
            large_gaps = time_diffs[time_diffs > max_wavelet_gap]
            
            if len(large_gaps) > 0:
                results['warnings'].append(f'Large gaps detected that may affect wavelet features: {len(large_gaps)} gaps > {max_wavelet_gap}')
                
            min_continuous_hours = self.config.get('min_continuous_data_hours', 48)
            continuous_periods = int(time_diffs[time_diffs <= timedelta(hours = 1)].count())
            
            if continuous_periods < min_continuous_hours:
                results['critical_issues'].append(f'Insufficient continuous data for wavelet analysis: {continuous_periods} hours (minimum: {min_continuous_hours})')
                return False
                
        # Check microstructure feature requirements
        if feature_eng_checks.get('check_microstructure_feature_requirements', True):
            if 'volume' not in data.columns:
                results['critical_issues'].append('Volume data required for microstructure features')
                return False
                
            volume = data['volume']
            close = data['close']
            volume_price_corr = volume.corr(close)
            
            if abs(volume_price_corr) > 0.95:
                results['warnings'].append(f'Unusually high volume-price correlation: {volume_price_corr:.3f} (may indicate data quality issues)')
                
            volume_mean = volume.mean()
            volume_std = volume.std()
            volume_spikes = volume > volume_mean + 5 * volume_std
            spike_ratio = volume_spikes.sum() / len(volume)
            max_spikes = self.config.get('max_volume_spikes', 0.01)
            
            if spike_ratio > max_spikes:
                results['warnings'].append(f'High volume spikes detected: {spike_ratio:.3f} (threshold: {max_spikes})')
                
        # Check multi-timeframe alignment
        if feature_eng_checks.get('check_multi_timeframe_alignment', True):
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                time_diffs_seconds = time_diffs.dt.total_seconds()
                interval_variance = time_diffs_seconds.var()
                expected_interval_seconds = expected_interval.total_seconds()
                variance_threshold = expected_interval_seconds * 0.15
                
                if interval_variance > variance_threshold:
                    mean_interval = time_diffs_seconds.mean()
                    cv = time_diffs_seconds.std() / mean_interval if mean_interval > 0 else 0
                    irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds = 30)]
                    irregular_ratio = len(irregular_intervals) / len(time_diffs)
                    
                    if cv > 0.3:
                        results['warnings'].append(f'High time interval variability (CV: {cv:.3f}, irregular: {irregular_ratio:.1%}) may affect multi-timeframe feature generation - consider data preprocessing')
                    elif cv > 0.2:
                        results['warnings'].append(f'Moderate time interval variability (CV: {cv:.3f}, irregular: {irregular_ratio:.1%}) may affect multi-timeframe feature generation')
                    else:
                        results['warnings'].append(f'Time interval variance ({interval_variance:.1f}s², irregular: {irregular_ratio:.1%}) may affect multi-timeframe feature generation')
                        
        # Check timestamp regularity
        if feature_eng_checks.get('check_timestamp_regularity', True):
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                tolerance_percentage = 0.15
                tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
                irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > timedelta(seconds = tolerance_seconds)]
                irregular_ratio = len(irregular_intervals) / len(time_diffs)
                max_irregular = self.config.get('max_timestamp_discontinuity', 0.02)
                
                if irregular_ratio > max_irregular:
                    if len(irregular_intervals) > 0:
                        irregular_positions = irregular_intervals.index
                        if len(irregular_positions) > 1:
                            irregular_gaps = irregular_positions.to_series().diff().dropna()
                            clustered_irregular = (irregular_gaps < timedelta(minutes = 5)).sum() > len(irregular_gaps) * 0.5
                            
                            if clustered_irregular:
                                results['warnings'].append(f'Clustered irregular timestamp intervals detected: {irregular_ratio:.1%} (threshold: {max_irregular:.1%}) - may indicate data collection issues')
                            else:
                                results['warnings'].append(f'Scattered irregular timestamp intervals: {irregular_ratio:.1%} (threshold: {max_irregular:.1%}) - may affect multi-timeframe feature generation')
                        else:
                            results['warnings'].append(f'Irregular timestamp intervals: {irregular_ratio:.1%} (threshold: {max_irregular:.1%}) - may affect multi-timeframe feature generation')
                            
        # Check data stationarity preconditions
        if feature_eng_checks.get('check_data_stationarity_preconditions', True):
            close = data['close']
            price_trend = close.pct_change().rolling(20).mean().abs().mean()
            
            if price_trend > 0.01:
                results['warnings'].append(f'Strong price trend detected: {price_trend:.3f} (may affect stationarity-based features)')
                
        # Store feature engineering analysis
        results['detailed_analysis']['feature_engineering'] = {
            'rolling_window_compatible': len(data) >= 50,
            'wavelet_gaps_count': len(large_gaps) if 'large_gaps' in locals() else 0,
            'continuous_data_hours': continuous_periods if 'continuous_periods' in locals() else 0,
            'volume_price_correlation': float(volume_price_corr) if 'volume_price_corr' in locals() else None,
            'volume_spike_ratio': float(spike_ratio) if 'spike_ratio' in locals() else 0.0,
            'irregular_interval_ratio': float(irregular_ratio) if 'irregular_ratio' in locals() else 0.0,
            'price_trend_strength': float(price_trend) if 'price_trend' in locals() else 0.0
        }
        
        return True

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
