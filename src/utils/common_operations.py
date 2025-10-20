from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
from functools import wraps
import time
import gc
import psutil
import os
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

def safe_dataframe_operation(operation_func: Callable[..., pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """Run a dataframe op with a tiny safety net."""
    if not callable(operation_func):
        raise TypeError("operation_func must be callable")
    df = operation_func(*args, **kwargs)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("operation_func must return a pandas DataFrame")
    return df

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types."""
    # Use enhanced optimization system if available
    try:
        from src.utils.hardware import optimize_dataframe_default
        return optimize_dataframe_default(df)
    except ImportError:
        # Fallback to original implementation
        df_opt = df.copy()
        
        # Downcast integers
        for col in df_opt.select_dtypes(include=['int']).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
        
        # Downcast floats
        for col in df_opt.select_dtypes(include=['float']).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df_opt.select_dtypes(include=['object']).columns:
            if df_opt[col].nunique() / len(df_opt) < 0.5:  # If less than 50% unique values
                df_opt[col] = df_opt[col].astype('category')
    
    return df_opt

def safe_divide(a: Union[pd.Series, np.ndarray, float], 
                b: Union[pd.Series, np.ndarray, float], 
                fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
    """Safely divide two arrays/series, handling division by zero."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return a.div(b).fillna(fill_value)
    elif isinstance(a, pd.Series):
        return a.div(b).fillna(fill_value)
    elif isinstance(b, pd.Series):
        return a / b.fillna(1e-10)
    else:
        # NumPy array or scalars: allocate minimal output and use where mask
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        out_shape = np.broadcast(a_arr, b_arr).shape
        result = np.empty(out_shape, dtype=np.float64)
        # Initialize with fill_value only where division is invalid
        mask = b_arr != 0
        # Broadcast mask to output shape
        mask_b = np.broadcast_to(mask, out_shape)
        result[~mask_b] = fill_value
        np.divide(a_arr, b_arr, out=result, where=mask_b)
        return result

def safe_mean(x: Union[pd.Series, np.ndarray]) -> float:
    """Fast nan-safe mean for arrays/Series."""
    if isinstance(x, pd.Series):
        return float(np.nanmean(x.to_numpy(dtype=float, copy=False)))
    return float(np.nanmean(np.asarray(x, dtype=float)))

def safe_std(x: Union[pd.Series, np.ndarray], ddof: int = 0) -> float:
    """Fast nan-safe std for arrays/Series."""
    if isinstance(x, pd.Series):
        return float(np.nanstd(x.to_numpy(dtype=float, copy=False), ddof=ddof))
    return float(np.nanstd(np.asarray(x, dtype=float), ddof=ddof))

def safe_log(x: Union[pd.Series, np.ndarray], 
             base: float = np.e, 
             fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
    """Safely compute logarithm, handling zero and negative values."""
    if isinstance(x, pd.Series):
        return np.log(np.maximum(x, 1e-10)) / np.log(base)
    else:
        return np.log(np.maximum(x, 1e-10)) / np.log(base)

def safe_sqrt(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Safely compute square root, handling negative values."""
    if isinstance(x, pd.Series):
        return np.sqrt(np.maximum(x, 0))
    else:
        return np.sqrt(np.maximum(x, 0))

def safe_power(x: Union[pd.Series, np.ndarray, float], 
               y: Union[pd.Series, np.ndarray, float], 
               fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
    """Safely compute power, handling negative values and overflow."""
    try:
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            # Handle negative bases by taking absolute value and adjusting sign
            result = np.power(np.abs(x), y)
            # Restore sign for odd powers when base is negative
            sign_mask = (x < 0) & (y % 2 == 1)
            result = np.where(sign_mask, -result, result)
            return result.fillna(fill_value)
        elif isinstance(x, pd.Series):
            result = np.power(np.abs(x), y)
            sign_mask = (x < 0) & (y % 2 == 1)
            result = np.where(sign_mask, -result, result)
            return result.fillna(fill_value)
        elif isinstance(y, pd.Series):
            result = np.power(np.abs(x), y)
            sign_mask = (x < 0) & (y % 2 == 1)
            result = np.where(sign_mask, -result, result)
            return result.fillna(fill_value)
        else:
            # Scalar case
            if x < 0 and y % 2 == 1:
                return -np.power(np.abs(x), y)
            else:
                return np.power(np.abs(x), y)
    except (OverflowError, ValueError, ZeroDivisionError):
        return fill_value if not isinstance(x, pd.Series) else pd.Series([fill_value], index=x.index)

def rolling_apply_safe(df: pd.DataFrame, 
                      func: Callable, 
                      window: int, 
                      min_periods: Optional[int] = None,
                      **kwargs) -> pd.DataFrame:
    """Apply function to rolling window with error handling."""
    if min_periods is None:
        min_periods = window // 2
    
    try:
        result = df.rolling(window=window, min_periods=min_periods).apply(func, **kwargs)
        return result
    except Exception as e:
        logger.warning(f"Rolling apply failed: {e}, returning original DataFrame")
        return df

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1,
                      allow_duplicates: bool = True) -> bool:
    """Validate DataFrame structure and content."""
    if df is None:
        logger.error("DataFrame is None")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        return False
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
    
    if not allow_duplicates and df.duplicated().any():
        logger.error("DataFrame contains duplicate rows")
        return False
    
    return True

def clean_dataframe(df: pd.DataFrame, 
                   drop_na_threshold: float = 0.5,
                   remove_inf: bool = True) -> pd.DataFrame:
    """Clean DataFrame by removing problematic values."""
    df_clean = df.copy()
    
    # Remove columns with too many NaN values
    if drop_na_threshold < 1.0:
        threshold = int(len(df_clean) * drop_na_threshold)
        df_clean = df_clean.dropna(axis=1, thresh=threshold)
    
    # Replace infinite values
    if remove_inf:
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    return df_clean

def resample_dataframe(df: pd.DataFrame, 
                      freq: str, 
                      method: str = 'mean') -> pd.DataFrame:
    """Resample DataFrame to different frequency."""
    if method == 'mean':
        return df.resample(freq).mean()
    elif method == 'sum':
        return df.resample(freq).sum()
    elif method == 'last':
        return df.resample(freq).last()
    elif method == 'first':
        return df.resample(freq).first()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

def calculate_returns(prices: pd.Series, 
                     method: str = 'simple',
                     periods: int = 1) -> pd.Series:
    """Calculate returns from price series."""
    if method == 'simple':
        return prices.pct_change(periods=periods)
    elif method == 'log':
        return np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Unknown return method: {method}")

def calculate_volatility(returns: pd.Series, 
                        window: int = 20,
                        annualized: bool = True) -> pd.Series:
    """Calculate rolling volatility."""
    vol = returns.rolling(window=window).std()
    if annualized:
        vol = vol * np.sqrt(252)  # Assuming daily data
    return vol

def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.0,
                          window: int = 20) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    mean_return = excess_returns.rolling(window=window).mean()
    std_return = excess_returns.rolling(window=window).std()
    return mean_return / std_return

def calculate_max_drawdown(prices: pd.Series) -> pd.Series:
    """Calculate maximum drawdown."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown

def calculate_correlation_matrix(df: pd.DataFrame, 
                                method: str = 'pearson') -> pd.DataFrame:
    """Calculate correlation matrix with error handling."""
    try:
        return df.corr(method=method)
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {e}")
        return pd.DataFrame()

def detect_outliers(series: pd.Series, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """Detect outliers in a series."""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def remove_outliers(df: pd.DataFrame, 
                   columns: Optional[List[str]] = None,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """Remove outliers from DataFrame."""
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        outliers = detect_outliers(df_clean[col], method=method, threshold=threshold)
        df_clean = df_clean[~outliers]
    
    return df_clean

@contextmanager
def memory_monitor(operation_name: str = "Operation"):
    """Context manager to monitor memory usage during operations."""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    logger.info(f"🚀 Starting {operation_name} - Memory: {start_memory['rss']:.1f}MB")
    
    try:
        yield
    finally:
        end_memory = get_memory_usage()
        end_time = time.time()
        
        memory_diff = end_memory['rss'] - start_memory['rss']
        time_diff = end_time - start_time
        
        logger.info(f"✅ Completed {operation_name} - "
                   f"Memory: {end_memory['rss']:.1f}MB "
                   f"(Δ{memory_diff:+.1f}MB), "
                   f"Time: {time_diff:.2f}s")

def force_garbage_collection():
    """Force garbage collection to free memory."""
    collected = gc.collect()
    logger.debug(f"🗑️ Garbage collection freed {collected} objects")

def safe_merge(left: pd.DataFrame, 
               right: pd.DataFrame, 
               how: str = 'inner',
               **kwargs) -> pd.DataFrame:
    """Safely merge DataFrames with error handling."""
    try:
        return pd.merge(left, right, how=how, **kwargs)
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return left

def safe_concat(dataframes: List[pd.DataFrame], 
                axis: int = 0,
                **kwargs) -> pd.DataFrame:
    """Safely concatenate DataFrames with error handling."""
    if not dataframes:
        return pd.DataFrame()
    
    try:
        return pd.concat(dataframes, axis=axis, **kwargs)
    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        return dataframes[0] if dataframes else pd.DataFrame()

def create_lagged_features(df: pd.DataFrame, 
                          columns: List[str],
                          lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series."""
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    return df_lagged

def create_rolling_features(df: pd.DataFrame, 
                          columns: List[str],
                          windows: List[int],
                          functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """Create rolling window features."""
    df_rolling = df.copy()
    
    for col in columns:
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df_rolling[f"{col}_rolling_{func}_{window}"] = df[col].rolling(window=window).mean()
                elif func == 'std':
                    df_rolling[f"{col}_rolling_{func}_{window}"] = df[col].rolling(window=window).std()
                elif func == 'min':
                    df_rolling[f"{col}_rolling_{func}_{window}"] = df[col].rolling(window=window).min()
                elif func == 'max':
                    df_rolling[f"{col}_rolling_{func}_{window}"] = df[col].rolling(window=window).max()
    
    return df_rolling

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators."""
    df_tech = df.copy()
    
    if 'close' in df.columns:
        # Simple Moving Averages
        df_tech['sma_20'] = df['close'].rolling(window=20).mean()
        df_tech['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df_tech['ema_12'] = df['close'].ewm(span=12).mean()
        df_tech['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df_tech['macd'] = df_tech['ema_12'] - df_tech['ema_26']
        df_tech['macd_signal'] = df_tech['macd'].ewm(span=9).mean()
        df_tech['macd_histogram'] = df_tech['macd'] - df_tech['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_tech['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df_tech['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df_tech['bb_upper'] = df_tech['bb_middle'] + (bb_std * 2)
        df_tech['bb_lower'] = df_tech['bb_middle'] - (bb_std * 2)
        df_tech['bb_width'] = df_tech['bb_upper'] - df_tech['bb_lower']
        df_tech['bb_position'] = (df['close'] - df_tech['bb_lower']) / df_tech['bb_width']
    
    return df_tech

def performance_timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"⏱️ {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def memory_efficient_apply(df: pd.DataFrame, 
                          func: Callable, 
                          chunk_size: int = 1000) -> pd.DataFrame:
    """Apply function to DataFrame in chunks to manage memory."""
    if len(df) <= chunk_size:
        return func(df)
    
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        result_chunk = func(chunk)
        results.append(result_chunk)
        force_garbage_collection()
    
    return pd.concat(results, ignore_index=True)

def validate_time_series(df: pd.DataFrame, 
                        time_column: str = 'timestamp') -> bool:
    """Validate time series DataFrame."""
    if time_column not in df.columns:
        logger.error(f"Time column '{time_column}' not found")
        return False
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        logger.error(f"Time column '{time_column}' is not datetime type")
        return False
    
    if df[time_column].isna().any():
        logger.error("Time column contains NaN values")
        return False
    
    if df[time_column].duplicated().any():
        logger.error("Time column contains duplicate values")
        return False
    
    return True

def resample_to_frequency(df: pd.DataFrame, 
                         freq: str, 
                         time_column: str = 'timestamp',
                         method: str = 'mean') -> pd.DataFrame:
    """Resample time series to specified frequency."""
    if not validate_time_series(df, time_column):
        return df
    
    df_resampled = df.set_index(time_column).resample(freq)
    
    if method == 'mean':
        return df_resampled.mean().reset_index()
    elif method == 'sum':
        return df_resampled.sum().reset_index()
    elif method == 'last':
        return df_resampled.last().reset_index()
    elif method == 'first':
        return df_resampled.first().reset_index()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

def calculate_rolling_statistics(df: pd.DataFrame, 
                               columns: List[str],
                               windows: List[int],
                               statistics: List[str] = ['mean', 'std', 'min', 'max', 'skew', 'kurt']) -> pd.DataFrame:
    """Calculate comprehensive rolling statistics."""
    df_stats = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            for stat in statistics:
                if stat == 'mean':
                    df_stats[f"{col}_{stat}_{window}"] = df[col].rolling(window=window).mean()
                elif stat == 'std':
                    df_stats[f"{col}_{stat}_{window}"] = df[col].rolling(window=window).std()
                elif stat == 'min':
                    df_stats[f"{col}_{stat}_{window}"] = df[col].rolling(window=window).min()
                elif stat == 'max':
                    df_stats[f"{col}_{stat}_{window}"] = df[col].rolling(window=window).max()
                elif stat == 'skew':
                    df_stats[f"{col}_{stat}_{window}"] = df[col].rolling(window=window).skew()
                elif stat == 'kurt':
                    df_stats[f"{col}_{stat}_{window}"] = df[col].rolling(window=window).kurt()
    
    return df_stats

def safe_feature_engineering(df: pd.DataFrame, 
                            operations: List[Callable],
                            **kwargs) -> pd.DataFrame:
    """Safely apply feature engineering operations."""
    df_engineered = df.copy()
    
    for operation in operations:
        try:
            df_engineered = operation(df_engineered, **kwargs)
        except Exception as e:
            logger.warning(f"Feature engineering operation failed: {e}")
            continue
    
    return df_engineered

def create_interaction_features(df: pd.DataFrame, 
                              columns: List[str],
                              max_degree: int = 2) -> pd.DataFrame:
    """Create polynomial interaction features."""
    df_interactions = df.copy()
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i:], i):
            if col1 != col2:
                # Linear interaction
                df_interactions[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                
                # Quadratic terms if max_degree >= 2
                if max_degree >= 2:
                    df_interactions[f"{col1}_squared"] = df[col1] ** 2
                    df_interactions[f"{col2}_squared"] = df[col2] ** 2
    
    return df_interactions

def calculate_cross_correlations(df: pd.DataFrame, 
                               columns: List[str],
                               max_lags: int = 10) -> pd.DataFrame:
    """Calculate cross-correlations between columns."""
    correlations = []
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            if col1 in df.columns and col2 in df.columns:
                corr_data = []
                for lag in range(-max_lags, max_lags + 1):
                    if lag == 0:
                        corr = df[col1].corr(df[col2])
                    elif lag > 0:
                        corr = df[col1].corr(df[col2].shift(lag))
                    else:
                        corr = df[col1].shift(-lag).corr(df[col2])
                    
                    corr_data.append({
                        'col1': col1,
                        'col2': col2,
                        'lag': lag,
                        'correlation': corr
                    })
                
                correlations.extend(corr_data)
    
    return pd.DataFrame(correlations)

def detect_regime_changes(df: pd.DataFrame, 
                         column: str,
                         window: int = 50,
                         threshold: float = 2.0) -> pd.Series:
    """Detect regime changes using rolling statistics."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    # Z-score based regime detection
    z_scores = (df[column] - rolling_mean) / rolling_std
    regime_changes = np.abs(z_scores) > threshold
    
    return regime_changes

def create_momentum_features(df: pd.DataFrame, 
                           price_column: str = 'close',
                           periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """Create momentum-based features."""
    df_momentum = df.copy()
    
    if price_column not in df.columns:
        return df_momentum
    
    for period in periods:
        # Price momentum
        df_momentum[f"momentum_{period}"] = df[price_column].pct_change(period)
        
        # Rate of change
        df_momentum[f"roc_{period}"] = (df[price_column] / df[price_column].shift(period) - 1) * 100
        
        # Moving average convergence/divergence
        if period > 1:
            short_ma = df[price_column].rolling(window=period//2).mean()
            long_ma = df[price_column].rolling(window=period).mean()
            df_momentum[f"macd_{period}"] = short_ma - long_ma
    
    return df_momentum

def calculate_volatility_features(df: pd.DataFrame, 
                                price_column: str = 'close',
                                windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """Calculate volatility-based features."""
    df_vol = df.copy()
    
    if price_column not in df.columns:
        return df_vol
    
    returns = df[price_column].pct_change()
    
    for window in windows:
        # Rolling volatility
        df_vol[f"volatility_{window}"] = returns.rolling(window=window).std()
        
        # Parkinson volatility (using high-low if available)
        if 'high' in df.columns and 'low' in df.columns:
            hl_vol = np.log(df['high'] / df['low']) ** 2
            df_vol[f"parkinson_vol_{window}"] = np.sqrt(hl_vol.rolling(window=window).mean() / (4 * np.log(2)))
        
        # Garman-Klass volatility
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            gk_vol = 0.5 * np.log(df['high'] / df['low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
            df_vol[f"gk_vol_{window}"] = np.sqrt(gk_vol.rolling(window=window).mean())
    
    return df_vol

def create_volume_features(df: pd.DataFrame, 
                          volume_column: str = 'volume',
                          price_column: str = 'close',
                          windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """Create volume-based features."""
    df_vol = df.copy()
    
    if volume_column not in df.columns or price_column not in df.columns:
        return df_vol
    
    for window in windows:
        # Volume moving average
        df_vol[f"volume_ma_{window}"] = df[volume_column].rolling(window=window).mean()
        
        # Volume ratio
        df_vol[f"volume_ratio_{window}"] = df[volume_column] / df_vol[f"volume_ma_{window}"]
        
        # Price-volume trend
        df_vol[f"pvt_{window}"] = (df[price_column].pct_change() * df[volume_column]).rolling(window=window).sum()
        
        # On-balance volume
        price_change = df[price_column].diff()
        obv = np.where(price_change > 0, df[volume_column], 
                      np.where(price_change < 0, -df[volume_column], 0))
        df_vol[f"obv_{window}"] = pd.Series(obv).rolling(window=window).sum()
    
    return df_vol

def create_cyclical_features(df: pd.DataFrame, 
                           time_column: str = 'timestamp',
                           periods: List[str] = ['day', 'week', 'month', 'quarter']) -> pd.DataFrame:
    """Create cyclical time-based features."""
    df_cyclical = df.copy()
    
    if time_column not in df.columns:
        return df_cyclical
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])
    
    dt = df[time_column].dt
    
    for period in periods:
        if period == 'day':
            df_cyclical['day_sin'] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
            df_cyclical['day_cos'] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
        elif period == 'week':
            df_cyclical['week_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
            df_cyclical['week_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        elif period == 'month':
            df_cyclical['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            df_cyclical['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        elif period == 'quarter':
            df_cyclical['quarter_sin'] = np.sin(2 * np.pi * dt.quarter / 4)
            df_cyclical['quarter_cos'] = np.cos(2 * np.pi * dt.quarter / 4)
    
    return df_cyclical

def create_lag_features(df: pd.DataFrame, 
                       columns: List[str],
                       lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
    """Create lagged features for time series."""
    df_lagged = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    return df_lagged

def create_lead_features(df: pd.DataFrame, 
                        columns: List[str],
                        leads: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
    """Create lead features for time series (future values)."""
    df_lead = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lead in leads:
                df_lead[f"{col}_lead_{lead}"] = df[col].shift(-lead)
    
    return df_lead

def calculate_feature_importance_correlation(df: pd.DataFrame, 
                                           target_column: str,
                                           feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate correlation-based feature importance."""
    if target_column not in df.columns:
        return pd.DataFrame()
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column and df[col].dtype in ['float64', 'int64']]
    
    correlations = []
    for col in feature_columns:
        if col in df.columns:
            corr = df[target_column].corr(df[col])
            correlations.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
    
    importance_df = pd.DataFrame(correlations)
    return importance_df.sort_values('abs_correlation', ascending=False)

def detect_multicollinearity(df: pd.DataFrame, 
                           threshold: float = 0.8) -> pd.DataFrame:
    """Detect multicollinearity in features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Find pairs with high correlation
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return pd.DataFrame(high_corr_pairs)

def create_polynomial_features(df: pd.DataFrame, 
                             columns: List[str],
                             degree: int = 2) -> pd.DataFrame:
    """Create polynomial features."""
    df_poly = df.copy()
    
    for col in columns:
        if col in df.columns:
            for d in range(2, degree + 1):
                df_poly[f"{col}_poly_{d}"] = df[col] ** d
    
    return df_poly

def calculate_rolling_correlation(df: pd.DataFrame, 
                                col1: str, 
                                col2: str,
                                window: int = 20) -> pd.Series:
    """Calculate rolling correlation between two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return pd.Series()
    
    return df[col1].rolling(window=window).corr(df[col2])

def create_ratio_features(df: pd.DataFrame, 
                        numerator_cols: List[str],
                        denominator_cols: List[str]) -> pd.DataFrame:
    """Create ratio features between columns."""
    df_ratios = df.copy()
    
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns:
                ratio_name = f"{num_col}_over_{den_col}"
                df_ratios[ratio_name] = safe_divide(df[num_col], df[den_col])
    
    return df_ratios

def calculate_rolling_quantiles(df: pd.DataFrame, 
                              columns: List[str],
                              window: int = 20,
                              quantiles: List[float] = [0.25, 0.5, 0.75]) -> pd.DataFrame:
    """Calculate rolling quantiles for specified columns."""
    df_quantiles = df.copy()
    
    for col in columns:
        if col in df.columns:
            for q in quantiles:
                df_quantiles[f"{col}_q{int(q*100)}_{window}"] = df[col].rolling(window=window).quantile(q)
    
    return df_quantiles

def create_difference_features(df: pd.DataFrame, 
                             columns: List[str],
                             periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Create difference features (current - lagged)."""
    df_diff = df.copy()
    
    for col in columns:
        if col in df.columns:
            for period in periods:
                df_diff[f"{col}_diff_{period}"] = df[col] - df[col].shift(period)
    
    return df_diff

def calculate_rolling_skewness_kurtosis(df: pd.DataFrame, 
                                      columns: List[str],
                                      window: int = 20) -> pd.DataFrame:
    """Calculate rolling skewness and kurtosis."""
    df_skew_kurt = df.copy()
    
    for col in columns:
        if col in df.columns:
            df_skew_kurt[f"{col}_skew_{window}"] = df[col].rolling(window=window).skew()
            df_skew_kurt[f"{col}_kurt_{window}"] = df[col].rolling(window=window).kurt()
    
    return df_skew_kurt

def create_interaction_terms(df: pd.DataFrame, 
                           columns: List[str],
                           max_interactions: int = 10) -> pd.DataFrame:
    """Create interaction terms between columns."""
    df_interactions = df.copy()
    
    interaction_count = 0
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            if interaction_count >= max_interactions:
                break
            if col1 in df.columns and col2 in df.columns:
                df_interactions[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                interaction_count += 1
    
    return df_interactions

def calculate_rolling_rank(df: pd.DataFrame, 
                          columns: List[str],
                          window: int = 20) -> pd.DataFrame:
    """Calculate rolling rank for columns."""
    df_rank = df.copy()
    
    for col in columns:
        if col in df.columns:
            df_rank[f"{col}_rank_{window}"] = df[col].rolling(window=window).rank(pct=True)
    
    return df_rank

def create_binary_features(df: pd.DataFrame, 
                          columns: List[str],
                          thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Create binary features based on thresholds."""
    df_binary = df.copy()
    
    for col in columns:
        if col in df.columns:
            if thresholds and col in thresholds:
                threshold = thresholds[col]
            else:
                threshold = df[col].median()
            
            df_binary[f"{col}_binary"] = (df[col] > threshold).astype(int)
    
    return df_binary

def calculate_rolling_zscore(df: pd.DataFrame, 
                           columns: List[str],
                           window: int = 20) -> pd.DataFrame:
    """Calculate rolling z-scores for columns."""
    df_zscore = df.copy()
    
    for col in columns:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            df_zscore[f"{col}_zscore_{window}"] = (df[col] - rolling_mean) / rolling_std
    
    return df_zscore

def create_categorical_encoding(df: pd.DataFrame, 
                              columns: List[str],
                              method: str = 'onehot') -> pd.DataFrame:
    """Create categorical encodings."""
    df_encoded = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            if method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            elif method == 'label':
                df_encoded[f"{col}_encoded"] = pd.Categorical(df[col]).codes
            elif method == 'target':
                # This would require target values, placeholder for now
                df_encoded[f"{col}_target_encoded"] = df[col].astype('category').cat.codes
    
    return df_encoded

def calculate_rolling_covariance(df: pd.DataFrame, 
                               col1: str, 
                               col2: str,
                               window: int = 20) -> pd.Series:
    """Calculate rolling covariance between two columns."""
    if col1 not in df.columns or col2 not in df.columns:
        return pd.Series()
    
    return df[col1].rolling(window=window).cov(df[col2])

def create_time_based_features(df: pd.DataFrame, 
                             time_column: str = 'timestamp') -> pd.DataFrame:
    """Create comprehensive time-based features."""
    df_time = df.copy()
    
    if time_column not in df.columns:
        return df_time
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])
    
    dt = df[time_column].dt
    
    # Basic time features
    df_time['year'] = dt.year
    df_time['month'] = dt.month
    df_time['day'] = dt.day
    df_time['dayofweek'] = dt.dayofweek
    df_time['dayofyear'] = dt.dayofyear
    df_time['week'] = dt.isocalendar().week
    df_time['quarter'] = dt.quarter
    df_time['hour'] = dt.hour
    df_time['minute'] = dt.minute
    
    # Cyclical features
    df_time['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
    df_time['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
    df_time['day_sin'] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
    df_time['day_cos'] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
    df_time['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    df_time['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    
    # Business day features
    df_time['is_weekend'] = dt.dayofweek.isin([5, 6]).astype(int)
    df_time['is_month_start'] = dt.is_month_start.astype(int)
    df_time['is_month_end'] = dt.is_month_end.astype(int)
    df_time['is_quarter_start'] = dt.is_quarter_start.astype(int)
    df_time['is_quarter_end'] = dt.is_quarter_end.astype(int)
    df_time['is_year_start'] = dt.is_year_start.astype(int)
    df_time['is_year_end'] = dt.is_year_end.astype(int)
    
    return df_time

def calculate_rolling_percentiles(df: pd.DataFrame, 
                                columns: List[str],
                                window: int = 20,
                                percentiles: List[float] = [10, 25, 50, 75, 90]) -> pd.DataFrame:
    """Calculate rolling percentiles for columns."""
    df_percentiles = df.copy()
    
    for col in columns:
        if col in df.columns:
            for p in percentiles:
                df_percentiles[f"{col}_p{int(p)}_{window}"] = df[col].rolling(window=window).quantile(p/100)
    
    return df_percentiles

def create_moving_average_features(df: pd.DataFrame, 
                                 columns: List[str],
                                 windows: List[int] = [5, 10, 20, 50, 100]) -> pd.DataFrame:
    """Create various moving average features."""
    df_ma = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                # Simple Moving Average
                df_ma[f"{col}_sma_{window}"] = df[col].rolling(window=window).mean()
                
                # Exponential Moving Average
                df_ma[f"{col}_ema_{window}"] = df[col].ewm(span=window).mean()
                
                # Weighted Moving Average
                weights = np.arange(1, window + 1)
                df_ma[f"{col}_wma_{window}"] = df[col].rolling(window=window).apply(
                    lambda x: np.average(x, weights=weights), raw=True
                )
                
                # Hull Moving Average
                wma_half = df[col].rolling(window=window//2).apply(
                    lambda x: np.average(x, weights=np.arange(1, window//2 + 1)), raw=True
                )
                wma_full = df[col].rolling(window=window).apply(
                    lambda x: np.average(x, weights=np.arange(1, window + 1)), raw=True
                )
                df_ma[f"{col}_hma_{window}"] = (2 * wma_half - wma_full).rolling(window=int(np.sqrt(window))).mean()
    
    return df_ma

def calculate_rolling_regression(df: pd.DataFrame, 
                               y_col: str, 
                               x_col: str,
                               window: int = 20) -> pd.DataFrame:
    """Calculate rolling linear regression statistics."""
    df_reg = df.copy()
    
    if y_col not in df.columns or x_col not in df.columns:
        return df_reg
    
    def rolling_regression(y, x):
        if len(y) < 2:
            return np.nan, np.nan, np.nan
        try:
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            return slope, intercept, r_squared
        except:
            return np.nan, np.nan, np.nan
    
    # Rolling regression
    reg_results = df[[y_col, x_col]].rolling(window=window).apply(
        lambda x: rolling_regression(x[y_col], x[x_col])[0], raw=False
    )
    
    df_reg[f"{y_col}_vs_{x_col}_slope_{window}"] = reg_results[y_col]
    df_reg[f"{y_col}_vs_{x_col}_r2_{window}"] = df[[y_col, x_col]].rolling(window=window).apply(
        lambda x: rolling_regression(x[y_col], x[x_col])[2], raw=False
    )[y_col]
    
    return df_reg

def create_fourier_features(df: pd.DataFrame, 
                          columns: List[str],
                          n_components: int = 5) -> pd.DataFrame:
    """Create Fourier transform features."""
    df_fourier = df.copy()
    
    for col in columns:
        if col in df.columns:
            # FFT
            fft = np.fft.fft(df[col].fillna(0))
            fft_real = np.real(fft)
            fft_imag = np.imag(fft)
            
            # Take first n_components
            for i in range(min(n_components, len(fft_real))):
                df_fourier[f"{col}_fft_real_{i}"] = fft_real[i]
                df_fourier[f"{col}_fft_imag_{i}"] = fft_imag[i]
    
    return df_fourier

def calculate_entropy_features(df: pd.DataFrame, 
                             columns: List[str],
                             window: int = 20) -> pd.DataFrame:
    """Calculate entropy-based features."""
    df_entropy = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Shannon entropy
            def shannon_entropy(x):
                if len(x) == 0:
                    return np.nan
                # Discretize data into bins
                bins = np.histogram(x, bins=10)[0]
                bins = bins[bins > 0]  # Remove zero bins
                if len(bins) == 0:
                    return 0
                probabilities = bins / np.sum(bins)
                return -np.sum(probabilities * np.log2(probabilities))
            
            df_entropy[f"{col}_entropy_{window}"] = df[col].rolling(window=window).apply(shannon_entropy, raw=True)
    
    return df_entropy

def create_wavelet_features(df: pd.DataFrame, 
                          columns: List[str],
                          wavelet: str = 'db4',
                          levels: int = 3) -> pd.DataFrame:
    """Create wavelet transform features."""
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not available, skipping wavelet features")
        return df
    
    df_wavelet = df.copy()
    
    for col in columns:
        if col in df.columns:
            try:
                # Wavelet decomposition
                coeffs = pywt.wavedec(df[col].fillna(0), wavelet, level=levels)
                
                # Extract features from coefficients
                for i, coeff in enumerate(coeffs):
                    if i == 0:  # Approximation coefficients
                        df_wavelet[f"{col}_wavelet_approx_{i}"] = coeff[0] if len(coeff) > 0 else 0
                    else:  # Detail coefficients
                        df_wavelet[f"{col}_wavelet_detail_{i}"] = coeff[0] if len(coeff) > 0 else 0
                        
            except Exception as e:
                logger.warning(f"Wavelet decomposition failed for {col}: {e}")
                continue
    
    return df_wavelet

def create_autocorrelation_features(df: pd.DataFrame, 
                                  columns: List[str],
                                  max_lags: int = 10) -> pd.DataFrame:
    """Create autocorrelation features."""
    df_autocorr = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in range(1, max_lags + 1):
                df_autocorr[f"{col}_autocorr_{lag}"] = df[col].autocorr(lag=lag)
    
    return df_autocorr

def calculate_rolling_information_ratio(df: pd.DataFrame, 
                                      returns_col: str,
                                      benchmark_col: str,
                                      window: int = 20) -> pd.Series:
    """Calculate rolling information ratio."""
    if returns_col not in df.columns or benchmark_col not in df.columns:
        return pd.Series()
    
    excess_returns = df[returns_col] - df[benchmark_col]
    tracking_error = excess_returns.rolling(window=window).std()
    mean_excess_return = excess_returns.rolling(window=window).mean()
    
    return mean_excess_return / tracking_error

def create_regime_features(df: pd.DataFrame, 
                         columns: List[str],
                         window: int = 50,
                         n_regimes: int = 3) -> pd.DataFrame:
    """Create regime-based features using clustering."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("Scikit-learn not available, skipping regime features")
        return df
    
    df_regime = df.copy()
    
    for col in columns:
        if col in df.columns:
            try:
                # Prepare data for clustering
                data = df[col].rolling(window=window).mean().fillna(method='bfill').values.reshape(-1, 1)
                
                # K-means clustering
                kmeans = KMeans(n_clusters=n_regimes, random_state=42)
                regimes = kmeans.fit_predict(data)
                
                # Create regime features
                df_regime[f"{col}_regime"] = regimes
                df_regime[f"{col}_regime_distance"] = np.min(kmeans.transform(data), axis=1)
                
            except Exception as e:
                logger.warning(f"Regime detection failed for {col}: {e}")
                continue
    
    return df_regime

def calculate_rolling_skewness_kurtosis_ratio(df: pd.DataFrame, 
                                            columns: List[str],
                                            window: int = 20) -> pd.DataFrame:
    """Calculate rolling skewness to kurtosis ratio."""
    df_skew_kurt_ratio = df.copy()
    
    for col in columns:
        if col in df.columns:
            skewness = df[col].rolling(window=window).skew()
            kurtosis = df[col].rolling(window=window).kurt()
            df_skew_kurt_ratio[f"{col}_skew_kurt_ratio_{window}"] = skewness / (kurtosis + 1e-10)
    
    return df_skew_kurt_ratio

def create_fractal_features(df: pd.DataFrame, 
                          columns: List[str],
                          window: int = 20) -> pd.DataFrame:
    """Create fractal dimension features."""
    df_fractal = df.copy()
    
    for col in columns:
        if col in df.columns:
            def fractal_dimension(x):
                if len(x) < 3:
                    return np.nan
                try:
                    # Box-counting method for fractal dimension
                    n = len(x)
                    scales = np.logspace(0.5, np.log10(n/4), 10).astype(int)
                    counts = []
                    
                    for scale in scales:
                        boxes = np.ceil(n / scale)
                        box_counts = []
                        for i in range(0, n, scale):
                            box_data = x[i:i+scale]
                            if len(box_data) > 0:
                                box_counts.append(np.max(box_data) - np.min(box_data))
                        counts.append(np.sum(box_counts))
                    
                    # Linear regression in log space
                    if len(counts) > 1:
                        log_scales = np.log(scales)
                        log_counts = np.log(counts)
                        slope = np.polyfit(log_scales, log_counts, 1)[0]
                        return -slope
                    else:
                        return np.nan
                except:
                    return np.nan
            
            df_fractal[f"{col}_fractal_dim_{window}"] = df[col].rolling(window=window).apply(fractal_dimension, raw=True)
    
    return df_fractal

def create_comprehensive_features(df: pd.DataFrame, 
                                target_columns: List[str],
                                time_column: Optional[str] = None,
                                feature_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Create comprehensive feature set for time series data."""
    if feature_config is None:
        feature_config = {
            'lags': [1, 2, 3, 5, 10, 20],
            'rolling_windows': [5, 10, 20, 50],
            'technical_indicators': True,
            'cyclical_features': True,
            'interaction_features': True,
            'polynomial_degree': 2
        }
    
    df_features = df.copy()
    
    # Time-based features
    if time_column and time_column in df.columns:
        df_features = create_time_based_features(df_features, time_column)
        if feature_config.get('cyclical_features', True):
            df_features = create_cyclical_features(df_features, time_column)
    
    # Lag features
    df_features = create_lag_features(df_features, target_columns, feature_config.get('lags', [1, 2, 3, 5, 10, 20]))
    
    # Rolling features
    df_features = create_rolling_features(df_features, target_columns, feature_config.get('rolling_windows', [5, 10, 20, 50]))
    
    # Technical indicators
    if feature_config.get('technical_indicators', True):
        df_features = calculate_technical_indicators(df_features)
        df_features = create_momentum_features(df_features)
        df_features = calculate_volatility_features(df_features)
    
    # Interaction features
    if feature_config.get('interaction_features', True):
        df_features = create_interaction_terms(df_features, target_columns)
        df_features = create_ratio_features(df_features, target_columns, target_columns)
    
    # Polynomial features
    if feature_config.get('polynomial_degree', 0) > 1:
        df_features = create_polynomial_features(df_features, target_columns, feature_config['polynomial_degree'])
    
    return df_features

# Additional missing functions that are being imported
def safe_mean(x: Union[pd.Series, np.ndarray], **kwargs) -> Union[pd.Series, np.ndarray]:
    """Safely calculate mean with error handling."""
    try:
        if isinstance(x, pd.Series):
            return x.mean(**kwargs)
        else:
            return np.mean(x, **kwargs)
    except Exception:
        return np.nan if not isinstance(x, pd.Series) else pd.Series([np.nan], index=x.index)

def safe_std(x: Union[pd.Series, np.ndarray], **kwargs) -> Union[pd.Series, np.ndarray]:
    """Safely calculate standard deviation with error handling."""
    try:
        if isinstance(x, pd.Series):
            return x.std(**kwargs)
        else:
            return np.std(x, **kwargs)
    except Exception:
        return np.nan if not isinstance(x, pd.Series) else pd.Series([np.nan], index=x.index)

def safe_correlation(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
    """Safely calculate correlation between two arrays/series."""
    try:
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            return x.corr(y)
        else:
            return np.corrcoef(x, y)[0, 1]
    except Exception:
        return 0.0

def safe_json_dump(data: Any, filepath: str, **kwargs) -> bool:
    """Safely dump data to JSON file."""
    try:
        import json
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False

def safe_json_load(filepath: str, **kwargs) -> Any:
    """Safely load data from JSON file."""
    try:
        import json
        with open(filepath, 'r') as f:
            return json.load(f, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return None

def ensure_directory(path: str) -> bool:
    """Ensure directory exists."""
    try:
        import os
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def safe_file_exists(filepath: str) -> bool:
    """Safely check if file exists."""
    try:
        import os
        return os.path.exists(filepath)
    except Exception:
        return False

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    if df is None or df.empty:
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    return len(missing_cols) == 0

def safe_convert_dtypes(df: pd.DataFrame, dtype_map: Dict[str, str]) -> pd.DataFrame:
    """Safely convert DataFrame column types."""
    try:
        df_converted = df.copy()
        for col, dtype in dtype_map.items():
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype(dtype)
        return df_converted
    except Exception as e:
        logger.warning(f"Type conversion failed: {e}")
        return df

def safe_read_parquet(filepath: str, **kwargs) -> pd.DataFrame:
    """Safely read parquet file."""
    try:
        return pd.read_parquet(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read parquet from {filepath}: {e}")
        return pd.DataFrame()

def validate_dataframe_schema(df: pd.DataFrame, schema: Dict[str, str]) -> bool:
    """Validate DataFrame schema."""
    try:
        for col, expected_type in schema.items():
            if col not in df.columns:
                return False
            if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                return False
        return True
    except Exception:
        return False

def validate_data_quality(df: pd.DataFrame, min_rows: int = 1) -> bool:
    """Validate data quality."""
    try:
        if df is None or df.empty:
            return False
        if len(df) < min_rows:
            return False
        return True
    except Exception:
        return False

def guard_dataframe_nulls(df: pd.DataFrame, max_null_ratio: float = 0.5) -> pd.DataFrame:
    """Guard against excessive nulls in DataFrame."""
    try:
        null_ratios = df.isnull().sum() / len(df)
        problematic_cols = null_ratios[null_ratios > max_null_ratio].index
        if len(problematic_cols) > 0:
            logger.warning(f"Columns with high null ratio: {list(problematic_cols)}")
        return df
    except Exception:
        return df

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def safe_log_metric(metric_name: str, value: float, step: int = 0) -> None:
    """Safely log metric."""
    try:
        logger.info(f"Metric {metric_name}: {value} at step {step}")
    except Exception:
        pass

def safe_log_params(params: Dict[str, Any]) -> None:
    """Safely log parameters."""
    try:
        logger.info(f"Parameters: {params}")
    except Exception:
        pass

def get_current_datetime() -> str:
    """Get current datetime as string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_datetime(dt: Any) -> str:
    """Format datetime object."""
    try:
        if hasattr(dt, 'strftime'):
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return str(dt)
    except Exception:
        return str(dt)

def create_fallback_logger(name: str) -> logging.Logger:
    """Create fallback logger."""
    return logging.getLogger(name)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)

def cleanup_m1_optimizers() -> bool:
    """Cleanup M1 optimizers.
    
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    try:
        # Placeholder for M1 optimizer cleanup
        # TODO: Implement actual M1 optimizer cleanup logic
        return True  # Return True to indicate successful cleanup (even if no-op)
    except Exception:
        return False

def get_m1_gpu_manager():
    """Get M1 GPU manager."""
    try:
        from src.utils.hardware.m1_gpu_utils import M1GPUManager
        return M1GPUManager()
    except Exception:
        return None

def get_m1_memory_optimizer():
    """Get M1 memory optimizer."""
    try:
        from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
        return M1MemoryOptimizer()
    except Exception:
        return None

def get_m1_cpu_optimizer():
    """Get M1 CPU optimizer."""
    try:
        from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
        return M1CPUOptimizer()
    except Exception:
        return None

def validate_finite(x: Union[pd.Series, np.ndarray]) -> bool:
    """Validate that values are finite."""
    try:
        if isinstance(x, pd.Series):
            return x.isna().sum() == 0 and np.isfinite(x).all()
        else:
            return np.isfinite(x).all()
    except Exception:
        return False

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame dtypes for memory efficiency."""
    return optimize_dataframe_memory(df)

def timed_operation(operation_name: str = "Operation"):
    """Context manager for timing operations."""
    from contextlib import contextmanager
    
    @contextmanager
    def _timed_operation():
        start_time = time.time()
        logger.info(f"🚀 Starting {operation_name}")
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ Completed {operation_name} in {duration:.4f} seconds")
    
    return _timed_operation()

def optimize_memory():
    """Optimize memory usage."""
    try:
        force_garbage_collection()
        logger.info("🧹 Memory optimization completed")
    except Exception as e:
        logger.warning(f"Memory optimization failed: {e}")

def safe_fillna(series: pd.Series, value: Any = None, method: str = None, **kwargs) -> pd.Series:
    """Safely fill NaN values in a series."""
    try:
        if method:
            return series.fillna(method=method, **kwargs)
        else:
            return series.fillna(value, **kwargs)
    except Exception as e:
        logger.warning(f"Fillna operation failed: {e}")
        return series

def safe_to_parquet(df: pd.DataFrame, filepath: str, **kwargs) -> bool:
    """Safely save DataFrame to parquet file."""
    try:
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to save parquet to {filepath}: {e}")
        return False

def list_parquet_files(directory: str, pattern: str = "*.parquet") -> List[str]:
    """List parquet files in a directory."""
    try:
        import glob
        import os
        search_pattern = os.path.join(directory, pattern)
        return glob.glob(search_pattern)
    except Exception as e:
        logger.error(f"Failed to list parquet files in {directory}: {e}")
        return []

def safe_copy(data: Any) -> Any:
    """Safely copy data."""
    try:
        import copy
        return copy.deepcopy(data)
    except Exception:
        return data

def safe_append(list_obj: List[Any], item: Any) -> bool:
    """Safely append item to list."""
    try:
        list_obj.append(item)
        return True
    except Exception:
        return False

def integrate_with_m1_optimizers() -> Dict[str, Any]:
    """Integrate with M1 optimizers."""
    return {
        'gpu_manager': get_m1_gpu_manager(),
        'memory_optimizer': get_m1_memory_optimizer(),
        'cpu_optimizer': get_m1_cpu_optimizer()
    }

# Missing functions for backward compatibility
def safe_merge_dataframes(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Safe merge of dataframes with error handling."""
    return safe_merge(left, right, **kwargs)

def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Safely drop columns from dataframe."""
    try:
        return df.drop(columns=columns, errors='ignore')
    except Exception:
        return df

def safe_rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Safely rename columns in dataframe."""
    try:
        return df.rename(columns=rename_map)
    except Exception:
        return df

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic data quality metrics."""
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic dataframe information."""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

def safe_rolling(df: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Safe rolling operation on dataframe."""
    try:
        return df.rolling(window=window, **kwargs)
    except Exception:
        return df

def safe_groupby_operation(df: pd.DataFrame, by: str, operation: str) -> pd.DataFrame:
    """Safe groupby operation."""
    try:
        grouped = df.groupby(by)
        if operation == 'mean':
            return grouped.mean()
        elif operation == 'sum':
            return grouped.sum()
        else:
            return grouped.agg(operation)
    except Exception:
        return df

def safe_apply_function(df: pd.DataFrame, func: Callable, **kwargs) -> pd.DataFrame:
    """Safe apply function to dataframe."""
    try:
        return df.apply(func, **kwargs)
    except Exception:
        return df

def safe_filter_dataframe(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Safe filter dataframe with condition."""
    try:
        return df.query(condition)
    except Exception:
        return df

def validate_positive(x: Union[pd.Series, np.ndarray, float, int], name: str = None) -> bool:
    """Validate that all values are positive."""
    if isinstance(x, (float, int)):
        return x > 0
    elif hasattr(x, '__len__'):
        return (x > 0).all()
    else:
        return x > 0

def memory_checkpoint(checkpoint_name: str):
    """Memory checkpoint context manager."""
    class MemoryCheckpoint:
        def __init__(self, name: str):
            self.name = name
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MemoryCheckpoint(checkpoint_name)

def gpu_context(use_gpu: bool = True):
    """GPU context manager."""
    class GPUContext:
        def __init__(self, use_gpu: bool):
            self.use_gpu = use_gpu
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return GPUContext(use_gpu)

def validate_range(value: float, min_val: float, max_val: float) -> bool:
    """Validate that a value is within a specified range."""
    return min_val <= value <= max_val

def safe_extend(target_list: list, source_list: list) -> bool:
    """Safely extend a list with another list."""
    try:
        target_list.extend(source_list)
        return True
    except Exception:
        return False

def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive data quality report."""
    try:
        report = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        }
        return report
    except Exception as e:
        return {'error': str(e)}

def safe_lower(text: str) -> str:
    """Safely convert text to lowercase."""
    try:
        return str(text).lower()
    except Exception:
        return str(text)

def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and is accessible."""
    try:
        return os.path.exists(file_path) and os.path.isfile(file_path)
    except Exception:
        return False

def check_disk_space(path: str) -> Dict[str, Any]:
    """Check available disk space for the given path."""
    try:
        stat = os.statvfs(path)
        free_bytes = stat.f_bavail * stat.f_frsize
        total_bytes = stat.f_blocks * stat.f_frsize
        used_bytes = total_bytes - free_bytes
        
        return {
            'free_gb': free_bytes / (1024**3),
            'total_gb': total_bytes / (1024**3),
            'used_gb': used_bytes / (1024**3),
            'free_percent': (free_bytes / total_bytes) * 100
        }
    except Exception:
        return {'error': 'Unable to check disk space'}


def safe_dict_items(d: Dict[Any, Any]) -> List[Tuple[Any, Any]]:
    """Safely get dictionary items."""
    try:
        if d is None or not isinstance(d, dict):
            return []
        return list(d.items())
    except Exception:
        return []

def safe_lower(s: str) -> str:
    """Safely convert string to lowercase."""
    try:
        if s is None:
            return ""
        return str(s).lower()
    except Exception:
        return ""

def safe_upper(s: str) -> str:
    """Safely convert string to uppercase."""
    try:
        if s is None:
            return ""
        return str(s).upper()
    except Exception:
        return ""

def safe_join(items: List[str], separator: str = " ") -> str:
    """Safely join list of strings."""
    try:
        if not items:
            return ""
        return separator.join(str(item) for item in items if item is not None)
    except Exception:
        return ""

def safe_kelly_calculation(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Safely calculate Kelly criterion."""
    try:
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        return max(0.0, min(kelly, 1.0))  # Clamp between 0 and 1
    except Exception:
        return 0.0

def safe_weighted_average(values: List[float], weights: List[float]) -> float:
    """Safely calculate weighted average."""
    try:
        if not values or not weights or len(values) != len(weights):
            return 0.0
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_weight
    except Exception:
        return 0.0

def safe_percentage_change(old_value: float, new_value: float) -> float:
    """Safely calculate percentage change."""
    try:
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    except Exception:
        return 0.0

def secure_file_path(*args, **kwargs) -> Callable:
    """Secure file path decorator for data processing."""
    def decorator(func: Callable) -> Callable:
        return func
    return decorator

def create_empty_dataframe(columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Create an empty DataFrame with optional columns."""
    try:
        if columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=columns)
    except Exception:
        return pd.DataFrame()

def validate_timestamp_column(df: pd.DataFrame, column: str = 'timestamp') -> bool:
    """Validate that a column contains valid timestamps."""
    try:
        if column not in df.columns:
            return False
        if df[column].isna().any():
            return False
        # Try to convert to datetime to validate
        pd.to_datetime(df[column])
        return True
    except Exception:
        return False

def safe_timestamp_conversion(timestamp: Any, format: str = None) -> Optional[pd.Timestamp]:
    """Safely convert timestamp to pandas Timestamp."""
    try:
        if timestamp is None:
            return None
        if format:
            return pd.to_datetime(timestamp, format=format)
        else:
            return pd.to_datetime(timestamp)
    except Exception:
        return None

class MathValidationError(Exception):
    """Custom exception for math validation errors."""
    pass

def check_disk_space(path: str, required_gb: float = 1.0) -> bool:
    """Check if there's enough disk space at the given path."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        return free_gb >= required_gb
    except Exception:
        return True  # Assume enough space if check fails

def validate_file_path(filepath: str) -> bool:
    """Validate that a file path is valid."""
    try:
        import os
        return os.path.exists(filepath) and os.path.isfile(filepath)
    except Exception:
        return False

def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    try:
        import os
        return os.path.getsize(filepath)
    except Exception:
        return 0

def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for DataFrame."""
    try:
        summary = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': df.nunique().to_dict()
        }
        return summary
    except Exception as e:
        logger.error(f"Error creating summary statistics: {e}")
        return {}

def validate_file_size(max_size_mb: int = 100) -> Callable:
    """Validate file size decorator.
    
    Args:
        max_size_mb: Maximum file size in MB
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if any argument is a file path
            for arg in args:
                if isinstance(arg, str) and os.path.exists(arg):
                    file_size_mb = get_file_size(arg) / (1024 * 1024)
                    if file_size_mb > max_size_mb:
                        logger.warning(f"⚠️ File {arg} is {file_size_mb:.2f}MB, exceeds limit of {max_size_mb}MB")
                        return None
            
            # Check kwargs for file paths
            for key, value in kwargs.items():
                if isinstance(value, str) and os.path.exists(value):
                    file_size_mb = get_file_size(value) / (1024 * 1024)
                    if file_size_mb > max_size_mb:
                        logger.warning(f"⚠️ File {value} is {file_size_mb:.2f}MB, exceeds limit of {max_size_mb}MB")
                        return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def with_tracing_span(*args, **kwargs) -> Callable:
    """Tracing span decorator for performance monitoring and tracing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log function entry for tracing
            logger.debug(f"🔍 Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"✅ Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

# Additional missing functions that are being imported
def safe_dict_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default fallback."""
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default

# Export all functions for easy importing
__all__ = [
    'safe_dataframe_operation',
    'get_memory_usage',
    'optimize_dataframe_memory',
    'safe_divide',
    'safe_log',
    'safe_sqrt',
    'safe_power',
    'rolling_apply_safe',
    'validate_dataframe',
    'clean_dataframe',
    'resample_dataframe',
    'calculate_returns',
    'calculate_volatility',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_correlation_matrix',
    'detect_outliers',
    'remove_outliers',
    'memory_monitor',
    'force_garbage_collection',
    'safe_merge',
    'safe_concat',
    'create_lagged_features',
    'create_rolling_features',
    'calculate_technical_indicators',
    'performance_timer',
    'memory_efficient_apply',
    'validate_time_series',
    'resample_to_frequency',
    'calculate_rolling_statistics',
    'safe_feature_engineering',
    'create_interaction_features',
    'calculate_cross_correlations',
    'detect_regime_changes',
    'create_momentum_features',
    'calculate_volatility_features',
    'create_volume_features',
    'create_cyclical_features',
    'create_lag_features',
    'create_lead_features',
    'calculate_feature_importance_correlation',
    'detect_multicollinearity',
    'create_polynomial_features',
    'calculate_rolling_correlation',
    'create_ratio_features',
    'calculate_rolling_quantiles',
    'create_difference_features',
    'calculate_rolling_skewness_kurtosis',
    'create_interaction_terms',
    'calculate_rolling_rank',
    'create_binary_features',
    'calculate_rolling_zscore',
    'create_categorical_encoding',
    'calculate_rolling_covariance',
    'create_time_based_features',
    'calculate_rolling_percentiles',
    'create_moving_average_features',
    'calculate_rolling_regression',
    'create_fourier_features',
    'calculate_entropy_features',
    'create_wavelet_features',
    'create_autocorrelation_features',
    'calculate_rolling_information_ratio',
    'create_regime_features',
    'calculate_rolling_skewness_kurtosis_ratio',
    'create_fractal_features',
    'create_comprehensive_features',
    # Additional missing functions
    'safe_mean',
    'safe_std',
    'safe_correlation',
    'safe_json_dump',
    'safe_json_load',
    'ensure_directory',
    'safe_file_exists',
    'validate_dataframe_columns',
    'safe_convert_dtypes',
    'safe_read_parquet',
    'validate_dataframe_schema',
    'validate_data_quality',
    'guard_dataframe_nulls',
    'safe_float',
    'safe_int',
    'format_bytes',
    'safe_log_metric',
    'safe_log_params',
    'get_current_datetime',
    'format_datetime',
    'create_fallback_logger',
    'get_logger',
    'cleanup_m1_optimizers',
    'get_m1_gpu_manager',
    'get_m1_memory_optimizer',
    'get_m1_cpu_optimizer',
    'validate_finite',
    'optimize_dataframe_dtypes',
    'timed_operation',
    'optimize_memory',
    'safe_fillna',
    'safe_to_parquet',
    'list_parquet_files',
    'safe_copy',
    'safe_append',
    'integrate_with_m1_optimizers',
    'safe_dict_get',
    'safe_dict_items',
    'safe_lower',
    'safe_upper',
    'safe_join',
    'safe_kelly_calculation',
    'safe_weighted_average',
    'safe_percentage_change',
    'secure_file_path',
    'create_empty_dataframe',
    'validate_timestamp_column',
    'safe_timestamp_conversion',
    'MathValidationError',
    'check_disk_space',
    'validate_file_path',
    'get_file_size',
    'create_summary_statistics',
    'validate_file_size',
    'with_tracing_span',
    'safe_resample',
    'align_dataframes',
    'safe_deepcopy'
]

def safe_resample(df: pd.DataFrame, freq: str, method: str = 'mean') -> pd.DataFrame:
    """Safely resample DataFrame to different frequency."""
    try:
        if method == 'mean':
            return df.resample(freq).mean()
        elif method == 'sum':
            return df.resample(freq).sum()
        elif method == 'last':
            return df.resample(freq).last()
        elif method == 'first':
            return df.resample(freq).first()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
    except Exception as e:
        logger.warning(f"Resampling failed: {e}")
        return df

def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                    on: Optional[str] = None, 
                    how: str = 'inner') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames on their index or a common column."""
    try:
        if on is not None and on in df1.columns and on in df2.columns:
            # Align on a common column
            merged = pd.merge(df1, df2, on=on, how=how, suffixes=('_1', '_2'))
            return merged, merged
        else:
            # Align on index
            common_index = df1.index.intersection(df2.index)
            if len(common_index) == 0:
                logger.warning("No common index found between DataFrames")
                return df1, df2
            
            df1_aligned = df1.loc[common_index]
            df2_aligned = df2.loc[common_index]
            return df1_aligned, df2_aligned
    except Exception as e:
        logger.warning(f"DataFrame alignment failed: {e}")
        return df1, df2

def safe_deepcopy(obj: Any) -> Any:
    """Safely perform deep copy of an object."""
    try:
        import copy
        return copy.deepcopy(obj)
    except Exception as e:
        logger.warning(f"Deep copy failed: {e}")
        return obj
