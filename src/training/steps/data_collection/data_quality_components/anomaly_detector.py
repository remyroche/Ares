"""Anomaly Detector Component

Detects various types of anomalies in market data.
Extracted from raw_data_quality_checker.py
"""

from typing import Any, Optional, List
import pandas as pd
import logging
import numpy as np
import warnings

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
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

except ImportError:

    cp = None

class AnomalyDetector:
    """Detects anomalies in market data using multiple detection methods.

    This class provides functionality for:
    - Statistical anomaly detection
    - Pattern-based anomaly detection
    - Time-based anomaly detection
    - Volume anomaly detection
    - Multi-dimensional anomaly detection
    """
    @log_important_calls
    def __init__(self, config: Optional[dict[str, Any]]=None) -> None:
        self.logger = system_logger.getChild('AnomalyDetector')
        self.config = config or self._get_default_config()
    @log_all_calls
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for anomaly detection."""
        return {'detection_methods': {'statistical': True, 'isolation_forest': False, 'local_outlier_factor': False, 'pattern_based': True, 'time_based': True}, 'statistical_params': {'zscore_threshold': 3.0, 'iqr_multiplier': 1.5, 'mad_threshold': 3.0, 'rolling_window': 20}, 'pattern_params': {'min_pattern_length': 3, 'similarity_threshold': 0.95}, 'volume_params': {'spike_threshold': 5.0, 'drop_threshold': 0.1, 'rolling_window': 20}}

    def detect_anomalies(self, data: pd.DataFrame, columns: Optional[List[str]]=None, methods: Optional[List[str]]=None) -> dict[str, Any]:
        """
        Detect anomalies using multiple methods.

        Args:
            data: DataFrame with market data
            columns: Columns to check for anomalies (None = all numeric)
            methods: Detection methods to use (None = use config defaults)

        Returns:
            Dictionary with anomaly detection results
        """
        self.logger.info('🔍 Starting comprehensive anomaly detection...')
        self.logger.info(f'📊 Input data shape: {data.shape}')

        if data.empty:
            self.logger.warning('⚠️ Empty data provided, returning empty results')
            return {'anomalies': {}, 'summary': {}, 'detailed_analysis': {}, 'recommendations': []}

        # Analyze data characteristics
        self.logger.info('📊 Data characteristics:')
        self.logger.info(f'   📅 Time range: {data.index.min()} to {data.index.max()}')
        self.logger.info(f'   ⏱️ Duration: {(data.index.max() - data.index.min()).total_seconds():.1f}s')
        self.logger.info(f'   📊 Columns: {list(data.columns)}')
        self.logger.info(f'   📈 Data types: {dict(data.dtypes)}')

        results = {'anomalies': {}, 'summary': {}, 'detailed_analysis': {}, 'recommendations': []}

        # Determine columns to analyze
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in columns if col not in ['timestamp', 'year', 'month', 'day']]
            self.logger.info(f'📊 Auto-selected numeric columns: {columns}')
        else:
            self.logger.info(f'📊 Using specified columns: {columns}')

        # Determine methods to use
        if methods is None:
            methods = [method for method, enabled in self.config['detection_methods'].items() if enabled]
            self.logger.info(f'🔧 Auto-selected methods: {methods}')
        else:
            self.logger.info(f'🔧 Using specified methods: {methods}')

        self.logger.info('🔍 Running anomaly detection algorithms...')

        # Run detection methods
        for i, method in enumerate(methods, 1):
            self.logger.info(f'🔍 {i}. Running {method} detection...')
            if method == 'statistical':
                self._detect_statistical_anomalies(data, columns, results)
            elif method == 'pattern_based':
                self._detect_pattern_anomalies(data, columns, results)
            elif method == 'time_based':
                self._detect_time_based_anomalies(data, results)
            else:
                self.logger.warning(f'⚠️ Unknown method: {method}')

        self.logger.info('📊 Generating anomaly summary...')
        self._generate_anomaly_summary(results)

        self.logger.info('💡 Generating recommendations...')
        results['recommendations'] = self._generate_anomaly_recommendations(results)

        # Log final summary
        total_anomalies = sum(len(anomalies) for anomalies in results['anomalies'].values() if isinstance(anomalies, list))
        self.logger.info('✅ Anomaly detection complete:')
        self.logger.info(f'   📊 Total anomalies found: {total_anomalies:,}')
        self.logger.info(f'   📈 Anomaly rate: {(total_anomalies / len(data) * 100):.2f}% of data points')
        self.logger.info(f'   📋 Recommendations: {len(results["recommendations"])}')

        return results
    @log_all_calls

    def _detect_statistical_anomalies(self, data: pd.DataFrame, columns: List[str], results: dict[str, Any]) -> None:
        """Detect statistical anomalies using z-score, IQR, and MAD methods."""
        self.logger.info('🔍 Detecting statistical anomalies...')
        self.logger.info(f'📊 Analyzing {len(columns)} columns: {columns}')

        total_anomalies = 0
        for i, col in enumerate(columns, 1):
            if col not in data.columns:
                self.logger.warning(f'⚠️ Column {col} not found in data, skipping')
                continue

            self.logger.info(f'📊 {i}. Analyzing column: {col}')
            col_data = data[col].dropna()

            if len(col_data) < 10:
                self.logger.warning(f'⚠️ Column {col} has only {len(col_data)} non-null values, skipping')
                continue

            self.logger.info(f'   📈 Non-null values: {len(col_data):,}')
            self.logger.info(f'   📊 Data range: {col_data.min():.6f} to {col_data.max():.6f}')
            self.logger.info(f'   📈 Mean: {col_data.mean():.6f}, Std: {col_data.std():.6f}')

            anomalies = {'zscore': [], 'iqr': [], 'mad': [], 'indices': []}

            # Z-score detection
            if self.config['statistical_params']['zscore_threshold'] > 0:
                self.logger.info(f'   🔍 Running z-score detection (threshold: {self.config["statistical_params"]["zscore_threshold"]})')
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    zscore_anomalies = z_scores > self.config['statistical_params']['zscore_threshold']
                    anomalies['zscore'] = col_data.index[zscore_anomalies].tolist()
                    self.logger.info(f'   📊 Z-score anomalies: {len(anomalies["zscore"]):,}')
                else:
                    self.logger.warning(f'   ⚠️ Zero standard deviation for {col}, skipping z-score detection')

            # IQR detection
            if self.config['statistical_params']['iqr_multiplier'] > 0:
                self.logger.info(f'   🔍 Running IQR detection (multiplier: {self.config["statistical_params"]["iqr_multiplier"]})')
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                multiplier = self.config['statistical_params']['iqr_multiplier']
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                iqr_anomalies = (col_data < lower_bound) | (col_data > upper_bound)
                anomalies['iqr'] = col_data.index[iqr_anomalies].tolist()
                self.logger.info(f'   📊 IQR bounds: [{lower_bound:.6f}, {upper_bound:.6f}]')
                self.logger.info(f'   📊 IQR anomalies: {len(anomalies["iqr"]):,}')

            # MAD detection
            if self.config['statistical_params']['mad_threshold'] > 0:
                self.logger.info(f'   🔍 Running MAD detection (threshold: {self.config["statistical_params"]["mad_threshold"]})')
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                if mad > 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    mad_anomalies = np.abs(modified_z_scores) > self.config['statistical_params']['mad_threshold']
                    anomalies['mad'] = col_data.index[mad_anomalies].tolist()
                    self.logger.info(f'   📊 MAD anomalies: {len(anomalies["mad"]):,}')
                else:
                    self.logger.warning(f'   ⚠️ Zero MAD for {col}, skipping MAD detection')

            all_indices = set()
            for method_indices in anomalies.values():
                if isinstance(method_indices, list):
                    all_indices.update(method_indices)
            anomalies['indices'] = sorted(list(all_indices))

            if anomalies['indices']:
                results['anomalies'][col] = anomalies
                total_anomalies += len(anomalies['indices'])
                self.logger.info(f'   ✅ Total anomalies for {col}: {len(anomalies["indices"]):,}')
            else:
                self.logger.info(f'   ✅ No anomalies found for {col}')

        self.logger.info(f'📊 Statistical anomaly detection complete: {total_anomalies:,} total anomalies found')
    @log_all_calls

    def _detect_pattern_anomalies(self, data: pd.DataFrame, columns: List[str], results: dict[str, Any]) -> None:
        """Detect anomalies based on unusual patterns."""
        self.logger.info('🔍 Detecting pattern-based anomalies...')
        self.logger.info(f'📊 Analyzing patterns in {len(columns)} columns')

        pattern_anomalies = {}
        total_pattern_anomalies = 0

        for i, col in enumerate(columns, 1):
            if col not in data.columns:
                self.logger.warning(f'⚠️ Column {col} not found in data, skipping')
                continue

            self.logger.info(f'📊 {i}. Analyzing patterns in column: {col}')

            if col in ['volume', 'close']:
                self.logger.info(f'   🔍 Running rolling window analysis (window: {self.config["statistical_params"]["rolling_window"]})')

                rolling_mean = data[col].rolling(window = self.config['statistical_params']['rolling_window']).mean()
                rolling_std = data[col].rolling(window = self.config['statistical_params']['rolling_window']).std()

                if rolling_mean is not None and rolling_std is not None:
                    self.logger.info(f'   📊 Rolling statistics calculated')
                    self.logger.info(f'   📈 Rolling mean range: {rolling_mean.min():.6f} to {rolling_mean.max():.6f}')
                    self.logger.info(f'   📊 Rolling std range: {rolling_std.min():.6f} to {rolling_std.max():.6f}')

                    deviations = np.abs(data[col] - rolling_mean) / rolling_std
                    pattern_anomaly_mask = deviations > 3
                    pattern_anomaly_indices = data.index[pattern_anomaly_mask.fillna(False)].tolist()

                    if pattern_anomaly_indices:
                        pattern_anomalies[col] = {'type': 'sudden_change', 'indices': pattern_anomaly_indices, 'count': len(pattern_anomaly_indices)}
                        total_pattern_anomalies += len(pattern_anomaly_indices)
                        self.logger.info(f'   ✅ Pattern anomalies found: {len(pattern_anomaly_indices):,}')

                        # Show some examples
                        if len(pattern_anomaly_indices) > 0:
                            self.logger.info('   📋 Sample pattern anomalies:')
                            for j, idx in enumerate(pattern_anomaly_indices[:3]):
                                deviation = deviations.loc[idx]
                                self.logger.info(f'      {j+1}. {idx}: deviation={deviation:.2f}')
                            if len(pattern_anomaly_indices) > 3:
                                self.logger.info(f'      ... and {len(pattern_anomaly_indices) - 3} more')
                    else:
                        self.logger.info(f'   ✅ No pattern anomalies found for {col}')
                else:
                    self.logger.warning(f'   ⚠️ Could not calculate rolling statistics for {col}')
            else:
                self.logger.info(f'   ⏭️ Skipping {col} (not in volume/close columns)')

        if pattern_anomalies:
            results['detailed_analysis']['pattern_anomalies'] = pattern_anomalies
            self.logger.info(f'📊 Pattern anomaly detection complete: {total_pattern_anomalies:,} total pattern anomalies found')
        else:
            self.logger.info('📊 Pattern anomaly detection complete: No pattern anomalies found')
    @log_all_calls

    def _detect_time_based_anomalies(self, data: pd.DataFrame, results: dict[str, Any]) -> None:
        """Detect time-based anomalies like unusual trading hours or gaps."""
        self.logger.info('🔍 Detecting time-based anomalies...')

        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning('⚠️ Data index is not datetime, skipping time-based anomaly detection')
            return

        self.logger.info('📊 Analyzing time-based patterns...')
        time_anomalies = {}
        total_time_anomalies = 0

        # Check for unusual trading hours
        self.logger.info('🔍 1. Checking for unusual trading hours...')
        trading_hours = data.index.hour
        unusual_hours = (trading_hours < 6) | (trading_hours > 22)

        if unusual_hours.any():
            unusual_count = unusual_hours.sum()
            unusual_percentage = float(unusual_hours.sum() / len(data) * 100)
            time_anomalies['unusual_hours'] = {
                'count': unusual_count,
                'percentage': unusual_percentage,
                'sample_times': data.index[unusual_hours][:10].tolist()
            }
            total_time_anomalies += unusual_count
            self.logger.info(f'   ⚠️ Found {unusual_count:,} unusual trading hours ({unusual_percentage:.2f}%)')
            self.logger.info(f'   📋 Sample unusual hours: {data.index[unusual_hours][:5].tolist()}')
        else:
            self.logger.info('   ✅ No unusual trading hours found')

        # Check for time gaps
        self.logger.info('🔍 2. Checking for significant time gaps...')
        time_diffs = data.index.to_series().diff()
        expected_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()

        if pd.notna(expected_interval):
            self.logger.info(f'   📊 Expected interval: {expected_interval}')
            significant_gaps = time_diffs > expected_interval * 10
            gap_count = significant_gaps.sum()

            if significant_gaps.any():
                gap_indices = data.index[significant_gaps]
                gap_times = []

                for i in range(len(data)):
                    if significant_gaps.iloc[i] and i > 0:
                        start_time = data.index[i - 1]
                        end_time = data.index[i]
                        duration_hours = float(time_diffs.iloc[i].total_seconds() / 3600)
                        gap_times.append({
                            'start': str(start_time),
                            'end': str(end_time),
                            'duration_hours': duration_hours
                        })

                time_anomalies['time_gaps'] = {
                    'count': gap_count,
                    'gap_times': gap_times[:10]
                }
                total_time_anomalies += gap_count
                self.logger.info(f'   ⚠️ Found {gap_count:,} significant time gaps')

                # Show some examples
                if gap_times:
                    self.logger.info('   📋 Sample time gaps:')
                    for j, gap in enumerate(gap_times[:3]):
                        self.logger.info(f'      {j+1}. {gap["start"]} to {gap["end"]}: {gap["duration_hours"]:.1f}h')
                    if len(gap_times) > 3:
                        self.logger.info(f'      ... and {len(gap_times) - 3} more gaps')
            else:
                self.logger.info('   ✅ No significant time gaps found')
        else:
            self.logger.warning('   ⚠️ Could not determine expected interval')

        if time_anomalies:
            results['detailed_analysis']['time_anomalies'] = time_anomalies
            self.logger.info(f'📊 Time-based anomaly detection complete: {total_time_anomalies:,} total time anomalies found')
        else:
            self.logger.info('📊 Time-based anomaly detection complete: No time anomalies found')

    def detect_volume_anomalies(self, data: pd.DataFrame, volume_col: str='volume') -> dict[str, Any]:
        """
        Detect volume-specific anomalies.

        Args:
            data: DataFrame with volume data
            volume_col: Name of volume column

        Returns:
            Volume anomaly detection results
        """
        self.logger.info('🔍 Starting volume anomaly detection...')
        self.logger.info(f'📊 Analyzing volume column: {volume_col}')

        results = {'volume_spikes': [], 'volume_drops': [], 'zero_volume_periods': [], 'statistics': {}}

        if volume_col not in data.columns:
            self.logger.warning(f'⚠️ Volume column {volume_col} not found in data')
            return results

        volume = data[volume_col]
        self.logger.info(f'📊 Volume data analysis:')
        self.logger.info(f'   📈 Total data points: {len(volume):,}')
        self.logger.info(f'   📊 Non-null values: {volume.notna().sum():,}')
        self.logger.info(f'   📈 Volume range: {volume.min():.2f} to {volume.max():.2f}')
        self.logger.info(f'   📊 Mean volume: {volume.mean():.2f}')
        self.logger.info(f'   📊 Std volume: {volume.std():.2f}')

        # Calculate rolling statistics
        rolling_window = self.config['volume_params']['rolling_window']
        self.logger.info(f'🔧 Calculating rolling statistics (window: {rolling_window})')
        rolling_mean = self._vectorbt_rolling_operation(volume, "mean", rolling_window)
        rolling_std = self._vectorbt_rolling_operation(volume, "std", rolling_window)

        # Detect volume spikes
        spike_threshold = self.config['volume_params']['spike_threshold']
        self.logger.info(f'🔍 Detecting volume spikes (threshold: {spike_threshold}x rolling mean)')
        volume_spikes = volume > rolling_mean * spike_threshold
        results['volume_spikes'] = data.index[volume_spikes.fillna(False)].tolist()
        self.logger.info(f'   📊 Volume spikes found: {len(results["volume_spikes"]):,}')

        # Detect volume drops
        drop_threshold = self.config['volume_params']['drop_threshold']
        self.logger.info(f'🔍 Detecting volume drops (threshold: {drop_threshold}x rolling mean)')
        volume_drops = (volume < rolling_mean * drop_threshold) & (volume > 0)
        results['volume_drops'] = data.index[volume_drops.fillna(False)].tolist()
        self.logger.info(f'   📊 Volume drops found: {len(results["volume_drops"]):,}')

        # Detect zero volume periods
        self.logger.info('🔍 Detecting zero volume periods...')
        zero_volume = volume == 0
        results['zero_volume_periods'] = data.index[zero_volume].tolist()
        self.logger.info(f'   📊 Zero volume periods found: {len(results["zero_volume_periods"]):,}')

        # Calculate statistics
        total_anomalies = len(results['volume_spikes']) + len(results['volume_drops']) + len(results['zero_volume_periods'])
        anomaly_rate = float(total_anomalies / len(data))

        results['statistics'] = {
            'mean_volume': float(volume.mean()),
            'std_volume': float(volume.std()),
            'min_volume': float(volume.min()),
            'max_volume': float(volume.max()),
            'spike_count': len(results['volume_spikes']),
            'drop_count': len(results['volume_drops']),
            'zero_count': len(results['zero_volume_periods']),
            'anomaly_rate': anomaly_rate
        }

        self.logger.info('✅ Volume anomaly detection complete:')
        self.logger.info(f'   📊 Total volume anomalies: {total_anomalies:,}')
        self.logger.info(f'   📈 Anomaly rate: {(anomaly_rate * 100):.2f}%')
        self.logger.info(f'   📊 Spikes: {len(results["volume_spikes"]):,}')
        self.logger.info(f'   📊 Drops: {len(results["volume_drops"]):,}')
        self.logger.info(f'   📊 Zero periods: {len(results["zero_volume_periods"]):,}')

        return results

    def detect_price_anomalies(self, data: pd.DataFrame, price_cols: Optional[List[str]]=None) -> dict[str, Any]:
        """
        Detect price-specific anomalies.

        Args:
            data: DataFrame with price data
            price_cols: Price columns to analyze

        Returns:
            Price anomaly detection results
        """
        if price_cols is None:
            price_cols = ['open', 'high', 'low', 'close']
        results = {'price_spikes': {}, 'price_drops': {}, 'price_reversals': {}, 'statistics': {}}
        for col in price_cols:
            if col not in data.columns:
                continue
            prices = data[col]
            returns = prices.pct_change()
            extreme_threshold = 0.1
            price_spikes = returns > extreme_threshold
            price_drops = returns < -extreme_threshold
            results['price_spikes'][col] = data.index[price_spikes.fillna(False)].tolist()
            results['price_drops'][col] = data.index[price_drops.fillna(False)].tolist()
            reversals = []
            for i in range(1, len(returns) - 1):
                if returns.iloc[i] > extreme_threshold and returns.iloc[i + 1] < -extreme_threshold * 0.5:
                    reversals.append(data.index[i])
                elif returns.iloc[i] < -extreme_threshold and returns.iloc[i + 1] > extreme_threshold * 0.5:
                    reversals.append(data.index[i])
            results['price_reversals'][col] = reversals
            results['statistics'][col] = {'mean_return': float(returns.mean()), 'std_return': float(returns.std()), 'max_return': float(returns.max()), 'min_return': float(returns.min()), 'spike_count': len(results['price_spikes'][col]), 'drop_count': len(results['price_drops'][col]), 'reversal_count': len(results['price_reversals'][col])}
        return results
    @log_all_calls

    def _generate_anomaly_summary(self, results: dict[str, Any]) -> None:
        """Generate summary of all detected anomalies."""
        total_anomalies = 0
        anomaly_columns = []
        if 'anomalies' in results:
            for col, anomaly_data in results['anomalies'].items():
                if 'indices' in anomaly_data:
                    count = len(anomaly_data['indices'])
                    total_anomalies += count
                    if count > 0:
                        anomaly_columns.append(col)
        if 'pattern_anomalies' in results.get('detailed_analysis', {}):
            for col, pattern_data in results['detailed_analysis']['pattern_anomalies'].items():
                total_anomalies += pattern_data.get('count', 0)
        if 'time_anomalies' in results.get('detailed_analysis', {}):
            for anomaly_type, time_data in results['detailed_analysis']['time_anomalies'].items():
                total_anomalies += time_data.get('count', 0)
        results['summary'] = {'total_anomalies': total_anomalies, 'columns_with_anomalies': anomaly_columns, 'anomaly_types_detected': list(results.get('detailed_analysis', {}).keys())}
    @log_all_calls

    def _generate_anomaly_recommendations(self, results: dict[str, Any]) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []
        if results['summary'].get('total_anomalies', 0) > 100:
            recommendations.append('High number of anomalies detected. Consider reviewing data source and collection process.')
        if 'time_anomalies' in results.get('detailed_analysis', {}):
            if 'time_gaps' in results['detailed_analysis']['time_anomalies']:
                recommendations.append('Significant time gaps detected. Verify data completeness and consider gap-filling strategies.')
        for col in results.get('anomalies', {}):
            if 'volume' in col.lower():
                recommendations.append('Volume anomalies detected. Review for potential data errors or unusual market conditions.')
                break
        price_cols = ['open', 'high', 'low', 'close']
        if any((col in results.get('anomalies', {}) for col in price_cols)):
            recommendations.append('Price anomalies detected. Validate against external sources and check for data feed issues.')
        return recommendations
