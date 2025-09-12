from ...core.decorators import handles_errors, traced
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

"""Fractional Differentiation for enhanced feature engineering.
Implements fractional-order differentiation to preserve memory and maintain
stationarity while avoiding over-differencing.
"""
from typing import Any, Optional, Dict, List, Tuple
from statsmodels.tsa.stattools import adfuller
from src.utils.logger import get_logger
import numpy as np
import pandas as pd
import logging

class FractionalDifferentiation:
    """Fractional differentiation for enhanced feature engineering.

    Replaces integer-order differentiation with fractional-order differentiation
    to preserve memory and maintain stationarity while avoiding over-differencing.

    Key benefits:
    - Preserves long-term memory better than integer differentiation
    - Maintains stationarity without over-differencing
    - Captures persistent trends more effectively
    - Reduces feature multicollinearity
    """
    @log_important_calls

    def __init__(self, d: float = 0.5, threshold: float = 1e-05, window: int = 100, optimize_order: bool = True) -> None:
        """Initialize fractional differentiation.

        Args:
            d: Fractional order (0 < d < 1)
            threshold: Minimum value threshold for stationarity
            window: Memory window for computation
            optimize_order: Whether to automatically optimize fractional order
        """
        self.d = d
        self.threshold = threshold
        self.window = window
        self.optimize_order = optimize_order
        self.weights = self._get_fractional_weights(window)
        self.logger = get_logger('FractionalDifferentiation')
    @log_all_calls

    def _get_fractional_weights(self, window: int) -> np.ndarray:
        """Generate fractional differentiation weights using binomial expansion.

        The weights follow the expansion of (1-L)^d where L is the lag operator.
        """
        weights = np.zeros(window)
        weights[0] = -self.d
        for k in range(1, window):
            weights[k] = weights[k - 1] * (k - 1 - self.d) / k
        return weights

    @handles_errors(fallback=pd.Series(dtype=float))
    @traced(span_name='FractionalDifferentiation.fractional_diff')
    def fractional_diff(self, series: pd.Series, preserve_original: bool = True) -> pd.Series:
        """
        Apply fractional differentiation to time series.

        Args:
            series: Input time series
            preserve_original: Whether to preserve original series name

        Returns:
            Fractionally differentiated series
        """
        try:
            if series.empty:
                self.logger.warning('Empty series provided for fractional differentiation')
                return pd.Series(dtype=float, name=f'{series.name}_frac_diff_{self.d}')
            
            if len(series) < self.window:
                self.logger.warning(f'Series too short for fractional diff, using simple diff: {len(series)} < {self.window}')
                diff_series = series.diff().fillna(0)
                return pd.Series(diff_series, index=series.index, name=f'{series.name}_frac_diff_simple')
            
            result = np.zeros(len(series))
            series_array = series.values
            
            # Apply fractional differentiation
            for i in range(self.window, len(series)):
                result[i] = np.sum(self.weights * series_array[i - self.window:i])
            
            # Check for stationarity
            result_std = np.std(result[self.window:])
            if result_std < self.threshold:
                self.logger.info(f'Series {series.name} already stationary after fractional diff (std={result_std:.6f})')
            
            # Create result series with proper naming
            result_name = f'{series.name}_frac_diff_{self.d:.3f}' if series.name else f'frac_diff_{self.d:.3f}'
            return pd.Series(result, index=series.index, name=result_name)
            
        except Exception as e:
            self.logger.error(f'Error in fractional differentiation for series {series.name}: {e}')
            # Return simple difference as fallback
            try:
                fallback_diff = series.diff().fillna(0)
                return pd.Series(fallback_diff, index=series.index, name=f'{series.name}_frac_diff_fallback')
            except Exception as fallback_error:
                self.logger.error(f'Fallback differentiation also failed: {fallback_error}')
                return pd.Series(dtype=float, name=f'{series.name}_frac_diff_error')

    @handles_errors(fallback=0.5)
    @traced(span_name='FractionalDifferentiation.optimize_fractional_order')
    def optimize_fractional_order(self, series: pd.Series, max_d: float = 0.9, min_d: float = 0.1, steps: int = 10) -> float:
        """
        Optimize fractional order for stationarity using ADF test.

        Args:
            series: Input time series
            max_d: Maximum fractional order to test
            min_d: Minimum fractional order to test
            steps: Number of steps to test

        Returns:
            Optimal fractional order
        """
        try:
            if series.empty or len(series) < 50:
                self.logger.warning(f'Series too short for optimization: {len(series)} < 50')
                return 0.5
            
            best_d = min_d
            best_pvalue = 1.0
            best_adf_stat = 0
            successful_tests = 0
            
            self.logger.info(f'Optimizing fractional order for series {series.name} (length: {len(series)})')
            
            for d in np.linspace(min_d, max_d, steps):
                try:
                    temp_diff = FractionalDifferentiation(d=d, window=self.window, optimize_order=False)
                    diff_series = temp_diff.fractional_diff(series)
                    clean_series = diff_series.dropna()
                    
                    if len(clean_series) < 10:
                        self.logger.debug(f'Insufficient data for d={d:.3f}: {len(clean_series)} < 10')
                        continue
                    
                    # Perform ADF test
                    adf_result = adfuller(clean_series, autolag='AIC')
                    pvalue = adf_result[1]
                    adf_stat = adf_result[0]
                    
                    # Check for better stationarity (lower p-value and more negative ADF statistic)
                    if pvalue < best_pvalue and adf_stat < best_adf_stat:
                        best_pvalue = pvalue
                        best_adf_stat = adf_stat
                        best_d = d
                    
                    successful_tests += 1
                    
                except Exception as e:
                    self.logger.debug(f'ADF test failed for d={d:.3f}: {e}')
                    continue
            
            if successful_tests == 0:
                self.logger.warning('No successful ADF tests, using default d=0.5')
                return 0.5
            
            self.logger.info(f'Optimal fractional order for {series.name}: d={best_d:.3f} (p-value={best_pvalue:.4f}, ADF={best_adf_stat:.4f})')
            return best_d
            
        except Exception as e:
            self.logger.error(f'Error in fractional order optimization: {e}')
            return 0.5

    def apply_with_optimization(self, series: pd.Series) -> tuple[pd.Series, float]:
        """
        Apply fractional differentiation with automatic order optimization.

        Args:
            series: Input time series

        Returns:
            Tuple of (differentiated_series, optimal_order)
        """
        if self.optimize_order:
            optimal_d = self.optimize_fractional_order(series)
            self.d = optimal_d
            self.weights = self._get_fractional_weights(self.window)
        result = self.fractional_diff(series)
        return (result, self.d)

    @handles_errors(fallback=(pd.DataFrame(), {}))
    @traced(span_name='FractionalDifferentiation.batch_fractional_diff')
    def batch_fractional_diff(self, data: pd.DataFrame, columns: Optional[List[str]] = None, exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Apply fractional differentiation to multiple columns.

        Args:
            data: Input DataFrame
            columns: Columns to differentiate (if None, use all numeric columns)
            exclude_columns: Columns to exclude from differentiation

        Returns:
            Tuple of (DataFrame with additional fractional differentiation features, optimization results)
        """
        try:
            if data.empty:
                self.logger.warning('Empty DataFrame provided for batch fractional differentiation')
                return data.copy(), {}
            
            # Determine columns to process
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if exclude_columns:
                columns = [col for col in columns if col not in exclude_columns]
            
            # Filter to existing columns only
            columns = [col for col in columns if col in data.columns]
            
            if not columns:
                self.logger.warning('No valid columns found for fractional differentiation')
                return data.copy(), {}
            
            result_data = data.copy()
            optimization_results = {}
            successful_columns = 0
            
            self.logger.info(f'Processing {len(columns)} columns for fractional differentiation')
            
            for col in columns:
                try:
                    if data[col].isnull().all():
                        self.logger.warning(f'Column {col} is all NaN, skipping')
                        continue
                    
                    diff_series, optimal_d = self.apply_with_optimization(data[col])
                    
                    # Only add if we got a valid result
                    if not diff_series.empty and not diff_series.isnull().all():
                        result_data[f'{col}_frac_diff_{optimal_d:.3f}'] = diff_series
                        optimization_results[col] = optimal_d
                        successful_columns += 1
                    else:
                        self.logger.warning(f'Invalid result for column {col}, skipping')
                        
                except Exception as e:
                    self.logger.error(f'Failed to apply fractional diff to {col}: {e}')
                    continue
            
            self.logger.info(f'Successfully applied fractional differentiation to {successful_columns}/{len(columns)} columns')
            return result_data, optimization_results
            
        except Exception as e:
            self.logger.error(f'Error in batch fractional differentiation: {e}')
            return data.copy(), {}

class FractionalFeatureGenerator:
    """High-level interface for generating fractional differentiation features."""
    @log_important_calls

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize fractional feature generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {'enable_fractional_diff': True, 'default_d': 0.5, 'optimize_order': True, 'window': 100, 'threshold': 1e-05, 'price_columns': ['close', 'high', 'low', 'open'], 'volume_columns': ['volume'], 'exclude_columns': ['timestamp', 'datetime', 'date']}
        self.fractional_diff = FractionalDifferentiation(d = self.config['default_d'], threshold = self.config['threshold'], window = self.config['window'], optimize_order = self.config['optimize_order'])
        self.logger = get_logger('FractionalFeatureGenerator')

    @handles_errors(fallback=pd.DataFrame())
    @traced(span_name='FractionalFeatureGenerator.generate_features')
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fractional differentiation features.

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with additional fractional differentiation features
        """
        try:
            if not self.config.get('enable_fractional_diff', True):
                self.logger.info('Fractional differentiation disabled in config')
                return data
            
            if data.empty:
                self.logger.warning('Empty DataFrame provided for feature generation')
                return data
            
            self.logger.info(f'Generating fractional differentiation features for {len(data)} rows')
            
            # Process price columns
            price_columns = [col for col in self.config.get('price_columns', ['close', 'high', 'low', 'open']) if col in data.columns]
            if price_columns:
                self.logger.info(f'Processing price columns: {price_columns}')
                result_data, price_results = self.fractional_diff.batch_fractional_diff(data, columns=price_columns)
            else:
                self.logger.warning('No price columns found in data')
                result_data = data.copy()
                price_results = {}
            
            # Process volume columns
            volume_columns = [col for col in self.config.get('volume_columns', ['volume']) if col in data.columns]
            if volume_columns:
                self.logger.info(f'Processing volume columns: {volume_columns}')
                result_data, volume_results = self.fractional_diff.batch_fractional_diff(result_data, columns=volume_columns)
            else:
                self.logger.warning('No volume columns found in data')
                volume_results = {}
            
            # Calculate statistics
            total_features = len(price_results) + len(volume_results)
            original_columns = len(data.columns)
            new_columns = len(result_data.columns)
            
            self.logger.info(f'Generated {total_features} fractional differentiation features')
            self.logger.info(f'Columns: {original_columns} -> {new_columns} (+{new_columns - original_columns})')
            
            # Log optimization results
            if price_results:
                self.logger.info(f'Price optimization results: {price_results}')
            if volume_results:
                self.logger.info(f'Volume optimization results: {volume_results}')
            
            return result_data
            
        except Exception as e:
            self.logger.error(f'Error in fractional feature generation: {e}')
            return data

    @traced(span_name='FractionalFeatureGenerator.generate_features_with_volume')
    def generate_features_with_volume(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate fractional differentiation features with separate volume data handling.

        Args:
            price_data: Input DataFrame with OHLCV price data
            volume_data: Optional separate volume data DataFrame

        Returns:
            DataFrame with additional fractional differentiation features
        """
        try:
            if not self.config.get('enable_fractional_diff', True):
                return price_data
            
            # Start with price data
            result_data = self.generate_features(price_data)
            
            # Process volume data if provided
            if volume_data is not None and not volume_data.empty:
                self.logger.info('Processing separate volume data')
                volume_columns = [col for col in self.config.get('volume_columns', ['volume']) if col in volume_data.columns]
                if volume_columns:
                    _, volume_results = self.fractional_diff.batch_fractional_diff(volume_data, columns=volume_columns)
                    for col, optimal_d in volume_results.items():
                        diff_series = self.fractional_diff.fractional_diff(volume_data[col])
                        result_data[f'{col}_frac_diff_{optimal_d:.3f}'] = diff_series
                    self.logger.info(f'Added {len(volume_results)} volume fractional features')
            
            return result_data
            
        except Exception as e:
            self.logger.error(f'Error in volume-aware feature generation: {e}')
            return price_data

    @traced(span_name='FractionalFeatureGenerator.get_feature_statistics')
    def get_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about fractional differentiation features."""
        try:
            frac_diff_columns = [col for col in data.columns if 'frac_diff' in col]
            stats = {
                'total_frac_diff_features': len(frac_diff_columns),
                'frac_diff_columns': frac_diff_columns,
                'feature_statistics': {}
            }
            
            for col in frac_diff_columns:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if not col_data.empty:
                        stats['feature_statistics'][col] = {
                            'mean': col_data.mean(),
                            'std': col_data.std(),
                            'min': col_data.min(),
                            'max': col_data.max(),
                            'null_count': data[col].isnull().sum(),
                            'non_null_count': len(col_data),
                            'zero_count': (col_data == 0).sum()
                        }
            
            return stats
            
        except Exception as e:
            self.logger.error(f'Error calculating feature statistics: {e}')
            return {'error': str(e), 'total_frac_diff_features': 0, 'frac_diff_columns': [], 'feature_statistics': {}}