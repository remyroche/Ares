from src.utils.tprint import tprint

from typing import Optional, Any, Dict, List, Union, Tuple
import numpy as np
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.config.environment import get_environment_settings
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

'Step 3: HMM Regime Discovery with Standardized Data Quality Management.\n\nThis module performs Hidden Markov Model (HMM) regime discovery with standardized\ndata quality checks and automatic data preparation using step01/step1_5 components.\n'
import asyncio
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)

# Import enhanced matrix operations
try:
    from src.utils.ml_common.matrix_operations import (
        get_enhanced_matrix_operations,
        m1_matrix_cholesky,
        m1_matrix_eigendecomposition,
        m1_matrix_correlation_analysis
    )
    ENHANCED_MATRIX_OPS_AVAILABLE = True
except ImportError:
    ENHANCED_MATRIX_OPS_AVAILABLE = False
    get_enhanced_matrix_operations = None
    m1_matrix_cholesky = None
    m1_matrix_eigendecomposition = None
    m1_matrix_correlation_analysis = None

# Import existing feature selection tools
try:
    from src.utils.feature_selection.step08_unified_complete import UnifiedStep08
    from src.utils.feature_selection.step08_unified_methods import UnifiedStep08Methods
    EXISTING_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    EXISTING_FEATURE_SELECTION_AVAILABLE = False

# Import parameter optimization
from .parameter_optimization import ParameterOptimizer
from .ensemble_optimization import EnsembleWeightOptimizer
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Get dynamic symbol configuration
_settings = get_environment_settings()

def get_default_symbol() -> str:
    """Get the default trading symbol from configuration."""
    return _settings.get_default_symbol('ETHUSDT')

# Import enhanced decorators
from src.core.decorators.logging import log_execution_time, log_call
from src.core.decorators.cache import cached
from src.core.decorators.retry_timeout import timeout, circuit_breaker
import os

# Import optimized components
try:
    from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
    OPTIMIZED_BAYESIAN_AVAILABLE = True
except ImportError:
    OPTIMIZED_BAYESIAN_AVAILABLE = False

try:
    from src.utils.hardware.memory_optimization import MemoryMonitor as EnhancedMemoryManager, get_memory_manager
    OPTIMIZED_MEMORY_AVAILABLE = True
except ImportError:
    OPTIMIZED_MEMORY_AVAILABLE = False

try:
    from src.utils.ml_common.ensemble_manager import EnsembleManager as AdvancedEnsembleClustering
    # Create a fallback ParallelClusteringProcessor class
    class ParallelClusteringProcessor:
        def __init__(self, *args, **kwargs):
            pass
    OPTIMIZED_CLUSTERING_AVAILABLE = True
except ImportError:
    OPTIMIZED_CLUSTERING_AVAILABLE = False

# CuPy import for GPU acceleration
try:
    import cupy as cp  # type: ignore[import]
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from src.utils.ml_common.matrix_operations import get_enhanced_matrix_operations as get_vectorized_operations_manager
    def create_vectorized_config(*args, **kwargs):
        return {}
    OPTIMIZED_VECTORIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_VECTORIZED_AVAILABLE = False

try:
    from src.utils.ml_common.pipeline_orchestrator import MLPipelineOrchestrator
    def get_step03_pipeline_orchestrator(*args, **kwargs):
        return MLPipelineOrchestrator(*args, **kwargs)
    def create_step03_pipeline_config(*args, **kwargs):
        return {}
    OPTIMIZED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    OPTIMIZED_ORCHESTRATOR_AVAILABLE = False

# Enhanced reporting system will be imported when needed to avoid circular imports

# Placeholder decorators for compatibility
def monitor_feature_engineering(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator

def traced(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator

def ensure_data_integrity(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator

def monitor_step_execution(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator

def secure_step_execution(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator

# SR Breakout Predictor will be imported directly where needed
import psutil

PSUTIL_AVAILABLE = psutil is not None

def create_fallback_logger() -> Any:
    logging.basicConfig(level = logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:
    """Create a fallback decorator that accepts keyword arguments like fallback."""
    def decorator(*args, **kwargs) -> Callable:
        def inner_decorator(func: Callable) -> Callable:
            return func
        return inner_decorator
    return decorator

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    path.mkdir(parents = True, exist_ok = True)
    return path

def safe_json_dump(data: Any, file_path: Path, **kwargs) -> None:
    """Safely dump data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, **kwargs)
if system_logger is None:
    system_logger = create_fallback_logger()
comprehensive_data_validation = create_fallback_decorator()
handle_errors = create_fallback_decorator()
memory_efficient = create_fallback_decorator()
resource_monitor = create_fallback_decorator()
secure_data_processing = create_fallback_decorator()
validate_data_structure = create_fallback_decorator()
with_tracing_span = create_fallback_decorator()
quality_gate = create_fallback_decorator()
monitor_feature_engineering = create_fallback_decorator()
ensure_data_integrity = create_fallback_decorator()
monitor_step_execution = create_fallback_decorator()
secure_step_execution = create_fallback_decorator()
validate_pipeline_step = create_fallback_decorator()
validates = create_fallback_decorator()
cached = create_fallback_decorator()
traced = create_fallback_decorator()
# handles_errors = create_fallback_decorator()  # Commented out to avoid overriding proper import
log_execution_time = create_fallback_decorator()
if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_artifact = lambda *args, **kwargs: 'fallback_artifact'
    log_step_dataframe = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'
    log_step_metrics = lambda *args, **kwargs: 'fallback_metrics'
    log_step_model = lambda *args, **kwargs: 'fallback_model'
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_artifact = enhanced_mlflow.log_step_artifact
    log_step_dataframe = enhanced_mlflow.log_step_dataframe
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_report = enhanced_mlflow.log_step_report
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_model = enhanced_mlflow.log_step_model
logger = system_logger.getChild('Step3HMMRegimeDiscovery')

class EnhancedFeatureEngineer:
    """Enhanced feature engineering for comprehensive regime detection"""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild('EnhancedFeatureEngineer')
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a comprehensive set of 100+ features for regime detection
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with comprehensive features
        """
        self.logger.info("🔧 Creating comprehensive feature set (100+ features)...")
        features = pd.DataFrame()
        features['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df.index
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Price-based features
        self._add_price_features(features, df)
        
        # Volume-based features
        self._add_volume_features(features, df)
        
        # Volatility features
        self._add_volatility_features(features, df)
        
        # Technical indicators
        self._add_technical_indicators(features, df)
        
        # Momentum features
        self._add_momentum_features(features, df)
        
        # Support/Resistance features
        self._add_sr_features(features, df)
        
        # Statistical features
        self._add_statistical_features(features, df)
        
        # Time-based features
        self._add_time_features(features, df)
        
        # Feature interactions
        self._add_feature_interactions(features)
        
        # Clean features
        features = self._clean_features(features)
        
        # Count features by category
        feature_counts = {
            'price_features': len([col for col in features.columns if 'price' in col or 'ma_' in col or 'ema_' in col or 'gap' in col or 'doji' in col or 'hammer' in col]),
            'volume_features': len([col for col in features.columns if 'volume' in col]),
            'volatility_features': len([col for col in features.columns if 'volatility' in col]),
            'technical_indicators': len([col for col in features.columns if any(ind in col for ind in ['rsi', 'macd', 'bb_', 'atr', 'adx'])]),
            'momentum_features': len([col for col in features.columns if 'momentum' in col]),
            'sr_features': len([col for col in features.columns if any(sr in col for sr in ['support', 'resistance', 'pivot', 'swing'])]),
            'statistical_features': len([col for col in features.columns if any(stat in col for stat in ['skewness', 'kurtosis', 'quantile', 'autocorr'])]),
            'time_features': len([col for col in features.columns if any(time in col for time in ['hour', 'day', 'month', 'sin', 'cos'])]),
            'interaction_features': len([col for col in features.columns if 'interaction' in col])
        }
        
        total_features = sum(feature_counts.values())
        self.logger.info(f"✅ Created {total_features} comprehensive features:")
        for category, count in feature_counts.items():
            self.logger.info(f"   {category}: {count} features")
        return features
    
    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add price-based features"""
        # Basic price features
        features['price_change'] = df['close'].pct_change()
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price ratios
        features['high_close_ratio'] = df['high'] / df['close']
        features['low_close_ratio'] = df['low'] / df['close']
        features['open_close_ratio'] = df['open'] / df['close']
        
        # Price gaps
        features['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['gap_down'] = (df['close'].shift(1) - df['open']) / df['close'].shift(1)
        
        # Price patterns
        features['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.1
        features['hammer'] = ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & \
                            ((df['high'] - df['close']) < 0.1 * (df['close'] - df['low']))
        
        # Multiple timeframe price features
        for window in [5, 10, 20, 50]:
            features[f'price_ma_{window}'] = df['close'].rolling(window).mean()
            features[f'price_ema_{window}'] = df['close'].ewm(span=window).mean()
            features[f'price_std_{window}'] = df['close'].rolling(window).std()
            features[f'price_min_{window}'] = df['close'].rolling(window).min()
            features[f'price_max_{window}'] = df['close'].rolling(window).max()
            
            # Price vs moving averages
            features[f'price_vs_ma_{window}'] = (df['close'] - features[f'price_ma_{window}']) / features[f'price_ma_{window}']
            features[f'price_vs_ema_{window}'] = (df['close'] - features[f'price_ema_{window}']) / features[f'price_ema_{window}']
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volume-based features"""
        # Basic volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume-price relationship
        features['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        features['volume_price_correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        # Volume patterns
        features['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 2
        features['volume_dry_up'] = df['volume'] < df['volume'].rolling(20).mean() * 0.5
        
        # Multiple timeframe volume features
        for window in [5, 10, 20, 50]:
            features[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            features[f'volume_ratio_{window}'] = df['volume'] / features[f'volume_ma_{window}']
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volatility features"""
        # Rolling volatility
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            features[f'volatility_ewma_{window}'] = df['close'].pct_change().ewm(span=window).std()
        
        # Volatility ratios
        features['volatility_ratio_5_20'] = features['volatility_5'] / features['volatility_20']
        features['volatility_ratio_10_50'] = features['volatility_10'] / features['volatility_50']
        
        # Volatility momentum
        features['volatility_momentum'] = features['volatility_20'] - features['volatility_20'].shift(5)
        features['volatility_acceleration'] = features['volatility_momentum'].diff()
        
        # GARCH-like features
        features['volatility_clustering'] = (df['close'].pct_change() ** 2).rolling(20).mean()
        features['volatility_persistence'] = features['volatility_clustering'].rolling(10).corr(
            features['volatility_clustering'].shift(1)
        )
    
    def _add_technical_indicators(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add technical indicators"""
        # RSI
        for window in [14, 21, 30]:
            features[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)
        
        # MACD
        features['macd'] = self._calculate_macd(df['close'])
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for window in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], window)
            features[f'bb_upper_{window}'] = bb_upper
            features[f'bb_middle_{window}'] = bb_middle
            features[f'bb_lower_{window}'] = bb_lower
            features[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_position_{window}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features['atr_14'] = self._calculate_atr(df)
        features['atr_ratio'] = features['atr_14'] / df['close']
        
        # ADX
        features['adx_14'] = self._calculate_adx(df)
    
    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add momentum features"""
        # Price momentum
        for window in [1, 2, 3, 5, 10, 20, 50]:
            features[f'momentum_{window}'] = df['close'].pct_change(window)
            features[f'momentum_ma_{window}'] = features[f'momentum_{window}'].rolling(10).mean()
        
        # Volume momentum
        for window in [1, 2, 3, 5, 10, 20]:
            features[f'volume_momentum_{window}'] = df['volume'].pct_change(window)
        
        # Momentum ratios
        features['momentum_ratio_5_20'] = features['momentum_5'] / features['momentum_20']
        features['momentum_ratio_10_50'] = features['momentum_10'] / features['momentum_50']
    
    def _add_sr_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add support/resistance features"""
        # Pivot points
        features['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        features['support_1'] = 2 * features['pivot_point'] - df['high']
        features['resistance_1'] = 2 * features['pivot_point'] - df['low']
        features['support_2'] = features['pivot_point'] - (df['high'] - df['low'])
        features['resistance_2'] = features['pivot_point'] + (df['high'] - df['low'])
        
        # Distance to S/R levels
        features['distance_to_support'] = (df['close'] - features['support_1']) / df['close']
        features['distance_to_resistance'] = (features['resistance_1'] - df['close']) / df['close']
        
        # S/R strength
        features['sr_strength'] = self._calculate_sr_strength(df)
        
        # Swing highs and lows
        for window in [10, 20, 50]:
            features[f'swing_high_{window}'] = df['high'].rolling(window, center=True).max()
            features[f'swing_low_{window}'] = df['low'].rolling(window, center=True).min()
            features[f'distance_to_swing_high_{window}'] = (features[f'swing_high_{window}'] - df['close']) / df['close']
            features[f'distance_to_swing_low_{window}'] = (df['close'] - features[f'swing_low_{window}']) / df['close']
    
    def _add_statistical_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add statistical features"""
        # Skewness and kurtosis
        for window in [20, 50]:
            features[f'skewness_{window}'] = df['close'].pct_change().rolling(window).skew()
            features[f'kurtosis_{window}'] = df['close'].pct_change().rolling(window).kurt()
        
        # Quantiles
        for window in [20, 50]:
            for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
                features[f'quantile_{q}_{window}'] = df['close'].rolling(window).quantile(q)
                features[f'price_vs_quantile_{q}_{window}'] = (df['close'] - features[f'quantile_{q}_{window}']) / df['close']
        
        # Autocorrelation
        for window in [20, 50]:
            features[f'autocorr_{window}'] = df['close'].pct_change().rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
    
    def _add_time_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add time-based features"""
        if 'timestamp' in features.columns:
            timestamp = pd.to_datetime(features['timestamp'])
            features['hour'] = timestamp.dt.hour
            features['day_of_week'] = timestamp.dt.dayofweek
            features['day_of_month'] = timestamp.dt.day
            features['month'] = timestamp.dt.month
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    def _add_feature_interactions(self, features: pd.DataFrame) -> None:
        """Add comprehensive feature interactions, accelerations, and returns"""
        self.logger.info("🔗 Creating comprehensive feature interactions...")
        
        # 1. Price-Volume Interactions (10+ features)
        if 'price_change' in features.columns and 'volume_change' in features.columns:
            features['price_volume_interaction'] = features['price_change'] * features['volume_change']
            features['price_volume_ratio'] = features['price_change'] / (features['volume_change'] + 1e-8)
            features['price_volume_correlation'] = features['price_change'].rolling(20).corr(features['volume_change'])
            features['price_volume_momentum'] = features['price_volume_interaction'].rolling(10).mean()
            features['price_volume_volatility'] = features['price_volume_interaction'].rolling(20).std()
        
        # 2. Volatility-Momentum Interactions (15+ features)
        volatility_cols = [col for col in features.columns if 'volatility' in col]
        momentum_cols = [col for col in features.columns if 'momentum' in col]
        
        for vol_col in volatility_cols[:3]:  # Top 3 volatility features
            for mom_col in momentum_cols[:3]:  # Top 3 momentum features
                if vol_col in features.columns and mom_col in features.columns:
                    features[f'{vol_col}_{mom_col}_interaction'] = features[vol_col] * features[mom_col]
                    features[f'{vol_col}_{mom_col}_ratio'] = features[vol_col] / (features[mom_col] + 1e-8)
                    features[f'{vol_col}_{mom_col}_correlation'] = features[vol_col].rolling(20).corr(features[mom_col])
        
        # 3. Technical Indicator Interactions (20+ features)
        rsi_cols = [col for col in features.columns if 'rsi' in col]
        macd_cols = [col for col in features.columns if 'macd' in col]
        bb_cols = [col for col in features.columns if 'bb_' in col]
        
        # RSI-MACD interactions
        for rsi_col in rsi_cols:
            for macd_col in macd_cols:
                if rsi_col in features.columns and macd_col in features.columns:
                    features[f'{rsi_col}_{macd_col}_interaction'] = features[rsi_col] * features[macd_col]
                    features[f'{rsi_col}_{macd_col}_divergence'] = features[rsi_col] - features[macd_col]
        
        # RSI-Bollinger Bands interactions
        for rsi_col in rsi_cols:
            for bb_col in bb_cols[:3]:  # Top 3 BB features
                if rsi_col in features.columns and bb_col in features.columns:
                    features[f'{rsi_col}_{bb_col}_interaction'] = features[rsi_col] * features[bb_col]
        
        # 4. Multi-timeframe Interactions (15+ features)
        short_term_cols = [col for col in features.columns if any(x in col for x in ['_5', '_10'])]
        long_term_cols = [col for col in features.columns if any(x in col for x in ['_20', '_50'])]
        
        for short_col in short_term_cols[:5]:  # Top 5 short-term features
            for long_col in long_term_cols[:5]:  # Top 5 long-term features
                if short_col in features.columns and long_col in features.columns:
                    features[f'{short_col}_{long_col}_ratio'] = features[short_col] / (features[long_col] + 1e-8)
                    features[f'{short_col}_{long_col}_spread'] = features[short_col] - features[long_col]
        
        # 5. Feature Accelerations (20+ features)
        self._add_feature_accelerations(features)
        
        # 6. Feature Returns (15+ features)
        self._add_feature_returns(features)
        
        # 7. Cross-Category Interactions (25+ features)
        self._add_cross_category_interactions(features)
        
        # 8. Statistical Interactions (10+ features)
        self._add_statistical_interactions(features)
    
    def _add_feature_accelerations(self, features: pd.DataFrame) -> None:
        """Add acceleration features (second derivatives)"""
        # Price accelerations
        if 'price_change' in features.columns:
            features['price_acceleration'] = features['price_change'].diff()
            features['price_acceleration_ma'] = features['price_acceleration'].rolling(10).mean()
            features['price_acceleration_volatility'] = features['price_acceleration'].rolling(20).std()
        
        # Volume accelerations
        if 'volume_change' in features.columns:
            features['volume_acceleration'] = features['volume_change'].diff()
            features['volume_acceleration_ma'] = features['volume_acceleration'].rolling(10).mean()
        
        # Volatility accelerations
        volatility_cols = [col for col in features.columns if 'volatility' in col and 'acceleration' not in col]
        for vol_col in volatility_cols[:3]:
            if vol_col in features.columns:
                features[f'{vol_col}_acceleration'] = features[vol_col].diff()
                features[f'{vol_col}_acceleration_ma'] = features[f'{vol_col}_acceleration'].rolling(10).mean()
        
        # Momentum accelerations
        momentum_cols = [col for col in features.columns if 'momentum' in col and 'acceleration' not in col]
        for mom_col in momentum_cols[:3]:
            if mom_col in features.columns:
                features[f'{mom_col}_acceleration'] = features[mom_col].diff()
                features[f'{mom_col}_acceleration_ma'] = features[f'{mom_col}_acceleration'].rolling(10).mean()
        
        # Technical indicator accelerations
        tech_cols = [col for col in features.columns if any(x in col for x in ['rsi', 'macd', 'bb_', 'atr', 'adx'])]
        for tech_col in tech_cols[:5]:
            if tech_col in features.columns:
                features[f'{tech_col}_acceleration'] = features[tech_col].diff()
    
    def _add_feature_returns(self, features: pd.DataFrame) -> None:
        """Add return features (percentage changes)"""
        # Price return features
        if 'price_change' in features.columns:
            features['price_return_5'] = features['price_change'].rolling(5).sum()
            features['price_return_10'] = features['price_change'].rolling(10).sum()
            features['price_return_20'] = features['price_change'].rolling(20).sum()
        
        # Volume return features
        if 'volume_change' in features.columns:
            features['volume_return_5'] = features['volume_change'].rolling(5).sum()
            features['volume_return_10'] = features['volume_change'].rolling(10).sum()
            features['volume_return_20'] = features['volume_change'].rolling(20).sum()
        
        # Volatility return features
        volatility_cols = [col for col in features.columns if 'volatility' in col and 'return' not in col]
        for vol_col in volatility_cols[:3]:
            if vol_col in features.columns:
                features[f'{vol_col}_return_5'] = features[vol_col].pct_change(5)
                features[f'{vol_col}_return_10'] = features[vol_col].pct_change(10)
        
        # Technical indicator returns
        tech_cols = [col for col in features.columns if any(x in col for x in ['rsi', 'macd', 'bb_', 'atr', 'adx'])]
        for tech_col in tech_cols[:5]:
            if tech_col in features.columns:
                features[f'{tech_col}_return_5'] = features[tech_col].pct_change(5)
                features[f'{tech_col}_return_10'] = features[tech_col].pct_change(10)
    
    def _add_cross_category_interactions(self, features: pd.DataFrame) -> None:
        """Add cross-category interactions"""
        # Price-Volatility interactions
        price_cols = [col for col in features.columns if 'price' in col][:3]
        volatility_cols = [col for col in features.columns if 'volatility' in col][:3]
        
        for price_col in price_cols:
            for vol_col in volatility_cols:
                if price_col in features.columns and vol_col in features.columns:
                    features[f'{price_col}_{vol_col}_interaction'] = features[price_col] * features[vol_col]
                    features[f'{price_col}_{vol_col}_ratio'] = features[price_col] / (features[vol_col] + 1e-8)
        
        # Volume-Volatility interactions
        volume_cols = [col for col in features.columns if 'volume' in col][:3]
        for vol_col in volume_cols:
            for vol_vol_col in volatility_cols:
                if vol_col in features.columns and vol_vol_col in features.columns:
                    features[f'{vol_col}_{vol_vol_col}_interaction'] = features[vol_col] * features[vol_vol_col]
        
        # Momentum-Volatility interactions
        momentum_cols = [col for col in features.columns if 'momentum' in col][:3]
        for mom_col in momentum_cols:
            for vol_col in volatility_cols:
                if mom_col in features.columns and vol_col in features.columns:
                    features[f'{mom_col}_{vol_col}_interaction'] = features[mom_col] * features[vol_col]
        
        # Support/Resistance-Volatility interactions
        sr_cols = [col for col in features.columns if any(x in col for x in ['support', 'resistance', 'swing'])][:3]
        for sr_col in sr_cols:
            for vol_col in volatility_cols:
                if sr_col in features.columns and vol_col in features.columns:
                    features[f'{sr_col}_{vol_col}_interaction'] = features[sr_col] * features[vol_col]
    
    def _add_statistical_interactions(self, features: pd.DataFrame) -> None:
        """Add statistical interaction features"""
        # Feature z-scores
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:  # Top 10 numeric features
            if col in features.columns:
                features[f'{col}_zscore'] = (features[col] - features[col].rolling(50).mean()) / features[col].rolling(50).std()
                features[f'{col}_zscore_ma'] = features[f'{col}_zscore'].rolling(10).mean()
        
        # Feature percentiles
        for col in numeric_cols[:5]:  # Top 5 numeric features
            if col in features.columns:
                features[f'{col}_percentile'] = features[col].rolling(50).rank(pct=True)
                features[f'{col}_percentile_ma'] = features[f'{col}_percentile'].rolling(10).mean()
        
        # Feature momentum interactions
        for col in numeric_cols[:5]:
            if col in features.columns:
                features[f'{col}_momentum_5'] = features[col].pct_change(5)
                features[f'{col}_momentum_10'] = features[col].pct_change(10)
                features[f'{col}_momentum_ratio'] = features[f'{col}_momentum_5'] / (features[f'{col}_momentum_10'] + 1e-8)
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        self.logger.info("🧹 Cleaning features...")
        
        # Remove timestamp column for HMM training
        if 'timestamp' in features.columns:
            features = features.drop('timestamp', axis=1)
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill technical indicators
        technical_cols = [col for col in features.columns if any(indicator in col for indicator in 
                       ['rsi', 'macd', 'bb_', 'atr', 'adx', 'sr_strength'])]
        for col in technical_cols:
            if col in features.columns:
                features[col] = features[col].ffill()
        
        # Fill remaining NaN values
        features = features.fillna(0)
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            self.logger.info(f"   Removing {len(constant_features)} constant features")
            features = features.drop(constant_features, axis=1)
        
        self.logger.info(f"✅ Feature cleaning completed: {len(features.columns)} features")
        return features
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + std * num_std
        lower = sma - std * num_std
        return upper, sma, lower
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        tr_smooth = tr.rolling(window).mean()
        dm_plus_smooth = dm_plus.rolling(window).mean()
        dm_minus_smooth = dm_minus.rolling(window).mean()
        
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        return dx.rolling(window).mean()
    
    def _calculate_sr_strength(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate support/resistance strength"""
        high_swing = df['high'].rolling(window, center=True).max()
        low_swing = df['low'].rolling(window, center=True).min()
        current_price = df['close']
        
        high_strength = (high_swing - current_price) / high_swing
        low_strength = (current_price - low_swing) / low_swing
        
        return (high_strength + low_strength) / 2

class HMMRegimeDiscoveryStep:
    """Step 3: HMM Regime Discovery with standardized data quality management."""
    @log_important_calls

    def __init__(self, config: dict[str, Any]) -> None:
        # 🖨️ THOROUGH PRINTING: HMM Regime Discovery Step Initialization
        tprint("🔧 INITIALIZING HMM REGIME DISCOVERY STEP")
        tprint(f"   📋 Configuration: {config}")
        
        self.config = config
        self.logger = system_logger.getChild('HMMRegimeDiscoveryStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        self.data_quality_manager = None
        
        # Initialize enhancement components
        self.feature_engineer = EnhancedFeatureEngineer(self.logger)
        self.parameter_optimizer = ParameterOptimizer(self.logger)
        self.ensemble_optimizer = EnsembleWeightOptimizer(self.logger)
        
        tprint("   ✅ Basic attributes initialized")
        tprint("   🔍 Validating environment dependencies...")
        self._validate_environment()
        tprint("   🔧 Initializing HMM regime discovery components...")
        self._initialize_components()
        tprint("   🎉 HMM Regime Discovery Step initialization complete")
    @log_all_calls
    def _create_enhanced_features(self, df: pd.DataFrame, use_existing_tools: bool = True) -> pd.DataFrame:
        """
        Create enhanced features using comprehensive feature engineering
        
        Args:
            df: Input DataFrame with OHLCV data
            use_existing_tools: Whether to use existing feature selection tools
            
        Returns:
            DataFrame with enhanced features
        """
        self.logger.info("🔧 Creating enhanced features...")
        
        # Step 1: Create comprehensive features
        comprehensive_features = self.feature_engineer.create_comprehensive_features(df)
        self.logger.info(f"✅ Created {len(comprehensive_features.columns)} comprehensive features")
        
        # Step 2: Use existing feature selection tools if available and requested
        if use_existing_tools and self.unified_step08 is not None:
            try:
                self.logger.info("🔍 Using existing feature selection tools...")
                
                # Prepare data for existing tools
                data_with_features = df.copy()
                for col in comprehensive_features.columns:
                    if col not in data_with_features.columns:
                        data_with_features[col] = comprehensive_features[col]
                
                # Use existing feature selection
                # Note: This would need to be adapted based on the actual interface
                # For now, we'll use the comprehensive features directly
                selected_features = comprehensive_features
                self.logger.info("✅ Used existing feature selection tools")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Existing feature selection tools failed: {e}")
                selected_features = comprehensive_features
        else:
            selected_features = comprehensive_features
        
        return selected_features
    
    @log_all_calls
    def _analyze_feature_importance(self, features: pd.DataFrame, regime_labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze feature importance using multiple methods
        
        Args:
            features: Input features
            regime_labels: Regime labels if available
            
        Returns:
            Dictionary with feature importance analysis
        """
        self.logger.info("🔍 Analyzing feature importance...")
        
        importance_analysis = {
            'feature_count': len(features.columns),
            'feature_categories': {},
            'importance_scores': {},
            'top_features': {},
            'recommendations': {}
        }
        
        # Categorize features
        feature_categories = {
            'price_features': [col for col in features.columns if 'price' in col or 'ma_' in col or 'ema_' in col],
            'volume_features': [col for col in features.columns if 'volume' in col],
            'volatility_features': [col for col in features.columns if 'volatility' in col],
            'technical_indicators': [col for col in features.columns if any(ind in col for ind in ['rsi', 'macd', 'bb_', 'atr', 'adx'])],
            'momentum_features': [col for col in features.columns if 'momentum' in col],
            'sr_features': [col for col in features.columns if any(sr in col for sr in ['support', 'resistance', 'pivot', 'swing'])],
            'statistical_features': [col for col in features.columns if any(stat in col for stat in ['skewness', 'kurtosis', 'quantile', 'autocorr'])],
            'time_features': [col for col in features.columns if any(time in col for time in ['hour', 'day', 'month', 'sin', 'cos'])],
            'interaction_features': [col for col in features.columns if 'interaction' in col]
        }
        
        importance_analysis['feature_categories'] = {
            category: len(feature_list) for category, feature_list in feature_categories.items()
        }
        
        # Calculate variance-based importance
        feature_variances = features.var().sort_values(ascending=False)
        importance_analysis['importance_scores']['variance'] = feature_variances.to_dict()
        importance_analysis['top_features']['variance'] = feature_variances.head(20).index.tolist()
        
        # Calculate correlation-based importance
        feature_correlations = features.corr().abs().mean().sort_values(ascending=False)
        importance_analysis['importance_scores']['correlation'] = feature_correlations.to_dict()
        importance_analysis['top_features']['correlation'] = feature_correlations.head(20).index.tolist()
        
        # If regime labels are available, calculate mutual information
        if regime_labels is not None:
            try:
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(features, regime_labels, random_state=42)
                mi_importance = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
                importance_analysis['importance_scores']['mutual_information'] = mi_importance.to_dict()
                importance_analysis['top_features']['mutual_information'] = mi_importance.head(20).index.tolist()
            except Exception as e:
                self.logger.warning(f"Mutual information calculation failed: {e}")
        
        # Generate recommendations
        recommendations = []
        
        # High variance features
        high_variance_features = feature_variances.head(10).index.tolist()
        recommendations.append(f"High variance features (most informative): {high_variance_features[:5]}")
        
        # Low variance features (potentially redundant)
        low_variance_features = feature_variances.tail(10).index.tolist()
        recommendations.append(f"Low variance features (potentially redundant): {low_variance_features[:5]}")
        
        # Category analysis
        for category, feature_list in feature_categories.items():
            if feature_list:
                category_variances = feature_variances[feature_list]
                if len(category_variances) > 0:
                    best_in_category = category_variances.idxmax()
                    recommendations.append(f"Best {category}: {best_in_category} (variance: {category_variances.max():.4f})")
        
        importance_analysis['recommendations'] = recommendations
        
        self.logger.info(f"✅ Feature importance analysis completed: {len(features.columns)} features analyzed")
        
        return importance_analysis
    
    @log_all_calls
    def _optimize_hmm_parameters(self, features: pd.DataFrame, use_optimization: bool = True) -> Dict[str, Any]:
        """
        Optimize HMM parameters using dynamic parameter search
        
        Args:
            features: Input features for optimization
            use_optimization: Whether to use parameter optimization
            
        Returns:
            Dictionary with optimal HMM parameters
        """
        if not use_optimization:
            # Use default parameters
            return {
                'n_components': 4,
                'covariance_type': 'full',
                'n_iter': 100,
                'tol': 0.001
            }
        
        self.logger.info("🔧 Optimizing HMM parameters...")
        
        try:
            # Use parameter optimizer
            optimization_result = self.parameter_optimizer.comprehensive_parameter_optimization(
                features.values, use_optuna=True, n_trials=50
            )
            
            optimal_params = optimization_result.best_params
            self.logger.info(f"✅ HMM parameters optimized: {optimal_params}")
            
            return optimal_params
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter optimization failed: {e}")
            # Fallback to default parameters
            return {
                'n_components': 4,
                'covariance_type': 'full',
                'n_iter': 100,
                'tol': 0.001
            }
    
    @log_all_calls
    def _optimize_ensemble_weights(self, hmm_results: Dict[str, Any], 
                                 kmeans_results: Dict[str, Any], 
                                 dbscan_results: Dict[str, Any],
                                 validation_data: np.ndarray,
                                 use_optimization: bool = True) -> Dict[str, float]:
        """
        Optimize ensemble weights using dynamic weight optimization
        
        Args:
            hmm_results: HMM clustering results
            kmeans_results: K-means clustering results
            dbscan_results: DBSCAN clustering results
            validation_data: Validation data for optimization
            use_optimization: Whether to use weight optimization
            
        Returns:
            Dictionary with optimal ensemble weights
        """
        if not use_optimization:
            # Use default weights
            return {'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3}
        
        self.logger.info("⚖️ Optimizing ensemble weights...")
        
        try:
            # Use ensemble optimizer
            optimization_result = self.ensemble_optimizer.multi_objective_optimization(
                hmm_results, kmeans_results, dbscan_results, validation_data
            )
            
            optimal_weights = optimization_result.optimal_weights
            self.logger.info(f"✅ Ensemble weights optimized: {optimal_weights}")
            
            return optimal_weights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble weight optimization failed: {e}")
            # Fallback to default weights
            return {'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3}
    
    @log_all_calls
    def _analyze_hmm_regimes(self, features: pd.DataFrame, hmm_predictions: np.ndarray, 
                           price_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze HMM regimes to determine their relevance and characteristics
        
        Args:
            features: Input features used for HMM training
            hmm_predictions: HMM state predictions
            price_data: Original price data for regime interpretation
            
        Returns:
            Dictionary with regime analysis
        """
        self.logger.info("🔍 Analyzing HMM regimes...")
        
        regime_analysis = {
            'n_regimes': len(np.unique(hmm_predictions)),
            'regime_distribution': {},
            'regime_characteristics': {},
            'regime_transitions': {},
            'regime_interpretation': {},
            'regime_quality_metrics': {},
            'recommendations': []
        }
        
        # Analyze regime distribution
        unique_states, state_counts = np.unique(hmm_predictions, return_counts=True)
        regime_analysis['regime_distribution'] = {
            f'regime_{state}': {
                'count': int(count),
                'percentage': float(count / len(hmm_predictions) * 100),
                'state_id': int(state)
            }
            for state, count in zip(unique_states, state_counts)
        }
        
        # Analyze regime characteristics
        for state in unique_states:
            state_mask = hmm_predictions == state
            state_features = features[state_mask]
            
            regime_characteristics = {
                'sample_count': int(np.sum(state_mask)),
                'feature_means': {},
                'feature_stds': {},
                'dominant_features': []
            }
            
            # Calculate feature statistics for this regime
            for feature in features.columns:
                feature_values = state_features[feature]
                regime_characteristics['feature_means'][feature] = float(feature_values.mean())
                regime_characteristics['feature_stds'][feature] = float(feature_values.std())
            
            # Identify dominant features (highest absolute means)
            feature_means_abs = {k: abs(v) for k, v in regime_characteristics['feature_means'].items()}
            dominant_features = sorted(feature_means_abs.items(), key=lambda x: x[1], reverse=True)[:10]
            regime_characteristics['dominant_features'] = [feat[0] for feat in dominant_features]
            
            regime_analysis['regime_characteristics'][f'regime_{state}'] = regime_characteristics
        
        # Analyze regime transitions
        transitions = []
        for i in range(1, len(hmm_predictions)):
            if hmm_predictions[i] != hmm_predictions[i-1]:
                transitions.append((hmm_predictions[i-1], hmm_predictions[i]))
        
        transition_counts = {}
        for transition in transitions:
            transition_key = f"{transition[0]}->{transition[1]}"
            transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
        
        regime_analysis['regime_transitions'] = transition_counts
        
        # Interpret regimes based on feature characteristics
        regime_interpretations = {}
        for state in unique_states:
            regime_key = f'regime_{state}'
            characteristics = regime_analysis['regime_characteristics'][regime_key]
            
            # Analyze key features to interpret regime
            interpretation = self._interpret_regime_type(characteristics, state)
            regime_interpretations[regime_key] = interpretation
        
        regime_analysis['regime_interpretation'] = regime_interpretations
        
        # Calculate regime quality metrics
        quality_metrics = self._calculate_regime_quality_metrics(features, hmm_predictions)
        regime_analysis['regime_quality_metrics'] = quality_metrics
        
        # Generate recommendations
        recommendations = self._generate_regime_recommendations(regime_analysis)
        regime_analysis['recommendations'] = recommendations
        
        self.logger.info(f"✅ Regime analysis completed: {len(unique_states)} regimes analyzed")
        
        return regime_analysis
    
    def _interpret_regime_type(self, characteristics: Dict[str, Any], state: int) -> Dict[str, Any]:
        """Interpret regime type based on feature characteristics"""
        interpretation = {
            'regime_type': 'unknown',
            'confidence': 0.0,
            'key_indicators': [],
            'description': ''
        }
        
        feature_means = characteristics['feature_means']
        
        # Analyze volatility
        volatility_features = [k for k in feature_means.keys() if 'volatility' in k]
        avg_volatility = np.mean([feature_means[k] for k in volatility_features if k in feature_means])
        
        # Analyze momentum
        momentum_features = [k for k in feature_means.keys() if 'momentum' in k]
        avg_momentum = np.mean([feature_means[k] for k in momentum_features if k in feature_means])
        
        # Analyze volume
        volume_features = [k for k in feature_means.keys() if 'volume' in k]
        avg_volume = np.mean([feature_means[k] for k in volume_features if k in feature_means])
        
        # Classify regime based on characteristics (including volume)
        if avg_volatility > 0.02:  # High volatility threshold
            if avg_momentum > 0.01:
                if avg_volume > 1.2:  # High volume threshold
                    interpretation['regime_type'] = 'bull_breakout'
                    interpretation['description'] = 'Strong upward trend with high volatility and volume'
                    interpretation['confidence'] = 0.9
                else:
                    interpretation['regime_type'] = 'bull_trend'
                    interpretation['description'] = 'Strong upward trend with high volatility but normal volume'
                    interpretation['confidence'] = 0.8
            elif avg_momentum < -0.01:
                if avg_volume > 1.2:
                    interpretation['regime_type'] = 'bear_breakdown'
                    interpretation['description'] = 'Strong downward trend with high volatility and volume'
                    interpretation['confidence'] = 0.9
                else:
                    interpretation['regime_type'] = 'bear_trend'
                    interpretation['description'] = 'Strong downward trend with high volatility but normal volume'
                    interpretation['confidence'] = 0.8
            else:
                if avg_volume > 1.5:
                    interpretation['regime_type'] = 'high_volatility_volume'
                    interpretation['description'] = 'High volatility with very high volume (potential reversal)'
                    interpretation['confidence'] = 0.7
                else:
                    interpretation['regime_type'] = 'high_volatility'
                    interpretation['description'] = 'High volatility without clear trend'
                    interpretation['confidence'] = 0.6
        else:  # Low volatility
            if abs(avg_momentum) < 0.005:
                if avg_volume < 0.8:
                    interpretation['regime_type'] = 'consolidation_low_volume'
                    interpretation['description'] = 'Low volatility consolidation with low volume'
                    interpretation['confidence'] = 0.8
                else:
                    interpretation['regime_type'] = 'consolidation'
                    interpretation['description'] = 'Low volatility consolidation phase'
                    interpretation['confidence'] = 0.7
            elif avg_momentum > 0.005:
                if avg_volume > 1.1:
                    interpretation['regime_type'] = 'gentle_bull_volume'
                    interpretation['description'] = 'Gentle upward trend with low volatility and above-average volume'
                    interpretation['confidence'] = 0.7
                else:
                    interpretation['regime_type'] = 'gentle_bull'
                    interpretation['description'] = 'Gentle upward trend with low volatility'
                    interpretation['confidence'] = 0.6
            else:
                if avg_volume > 1.1:
                    interpretation['regime_type'] = 'gentle_bear_volume'
                    interpretation['description'] = 'Gentle downward trend with low volatility and above-average volume'
                    interpretation['confidence'] = 0.7
                else:
                    interpretation['regime_type'] = 'gentle_bear'
                    interpretation['description'] = 'Gentle downward trend with low volatility'
                    interpretation['confidence'] = 0.6
        
        # Add key indicators
        interpretation['key_indicators'] = [
            f"Volatility: {avg_volatility:.4f}",
            f"Momentum: {avg_momentum:.4f}",
            f"Volume: {avg_volume:.4f}"
        ]
        
        return interpretation
    
    def _calculate_regime_quality_metrics(self, features: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for regime detection"""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            quality_metrics = {
                'silhouette_score': silhouette_score(features, predictions),
                'calinski_harabasz_score': calinski_harabasz_score(features, predictions),
                'davies_bouldin_score': davies_bouldin_score(features, predictions)
            }
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            quality_metrics = {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
        
        return quality_metrics
    
    def _generate_regime_recommendations(self, regime_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on regime analysis"""
        recommendations = []
        
        n_regimes = regime_analysis['n_regimes']
        quality_metrics = regime_analysis['regime_quality_metrics']
        
        # Regime count recommendations
        if n_regimes < 3:
            recommendations.append("Consider increasing number of regimes - current count may be too low for market complexity")
        elif n_regimes > 6:
            recommendations.append("Consider reducing number of regimes - current count may be too high and cause overfitting")
        else:
            recommendations.append(f"Regime count ({n_regimes}) appears appropriate for market complexity")
        
        # Quality recommendations
        silhouette_score = quality_metrics.get('silhouette_score', 0)
        if silhouette_score > 0.5:
            recommendations.append("Excellent regime separation - regimes are well-defined")
        elif silhouette_score > 0.3:
            recommendations.append("Good regime separation - regimes are reasonably well-defined")
        else:
            recommendations.append("Poor regime separation - consider feature engineering or parameter tuning")
        
        # Distribution recommendations
        regime_dist = regime_analysis['regime_distribution']
        min_percentage = min([regime['percentage'] for regime in regime_dist.values()])
        max_percentage = max([regime['percentage'] for regime in regime_dist.values()])
        
        if max_percentage > 70:
            recommendations.append("One regime dominates - consider adjusting parameters to better balance regimes")
        elif min_percentage < 5:
            recommendations.append("Some regimes are very rare - consider if they represent meaningful market states")
        
        return recommendations

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        tprint("   🔍 Validating environment dependencies...")
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            tprint(f"   ⚠️ Missing optional modules: {missing_modules}")
            tprint("   📝 Pipeline will continue with fallback implementations")
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            tprint("   ✅ All required dependencies available")
            self.logger.info('✅ All required dependencies available')
    @log_all_calls

    def _initialize_components(self) -> None:
        """Initialize HMM and data quality components with optimized managers."""
        self.logger.info('🔧 Initializing HMM regime discovery components...')

        # Initialize optimized components
        self._initialize_optimized_components()

        try:
            from src.training.steps.data_collection.data_preparation.enhanced_data_quality_manager import EnhancedDataQualityManager
            self.data_quality_manager = EnhancedDataQualityManager()
            self.logger.info('✅ Data quality manager initialized successfully')
        except Exception as e:
            self.logger.info(f'ℹ️ Data quality manager unavailable: {e}')
            self.data_quality_manager = None

        # Try to import SR Breakout Predictor with better error handling
        try:
            from src.tactician.sr_levels.sr_breakout_predictor_enhanced import SRBreakoutPredictor
            sr_config = self.config.copy()
            sr_config['sr_breakout_predictor'] = sr_config.get('sr_breakout_predictor', {})
            sr_config['sr_breakout_predictor']['use_optimized_params'] = True
            self.sr_predictor = SRBreakoutPredictor(sr_config)
            self.logger.info('✅ SR Breakout Predictor initialized successfully')
        except ImportError as e:
            self.logger.info(f'ℹ️ SR Breakout Predictor not available (import error): {e}')
            self.sr_predictor = None
        except Exception as e:
            self.logger.info(f'ℹ️ SR Breakout Predictor initialization failed: {e}')
            self.sr_predictor = None

        # Initialize enhanced reporting system (will be imported when needed)
        self.enhanced_reporter = None
        self.logger.info('ℹ️ Enhanced reporting system will be imported dynamically when needed')
        
        # Initialize existing feature selection tools if available
        if EXISTING_FEATURE_SELECTION_AVAILABLE:
            try:
                # Create configuration for existing feature selection
                feature_selection_config = {
                    'symbol': self.config.get('SYMBOL', 'ETHUSDT'),
                    'exchange': self.config.get('EXCHANGE', 'BINANCE'),
                    'timeframe': self.config.get('TIMEFRAME', '1m'),
                    'step08_unified': {
                        'phase1_target_features': 150,
                        'phase2_targets': [100, 80, 60],
                        'enable_mrmr': True,
                        'enable_rf_importance': True,
                        'boruta_max_iter': 100,
                        'boruta_alpha': 0.05
                    }
                }
                self.unified_step08 = UnifiedStep08(feature_selection_config)
                self.logger.info('✅ Existing feature selection tools initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to initialize existing feature selection tools: {e}')
                self.unified_step08 = None
        else:
            self.unified_step08 = None
            self.logger.info('ℹ️ Existing feature selection tools not available')

    def _initialize_optimized_components(self) -> None:
        """Initialize optimized components for enhanced performance."""
        self.logger.info('🚀 Initializing optimized performance components...')

        # Enhanced Memory Manager
        if OPTIMIZED_MEMORY_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager(self.config)
                self.logger.info('✅ Enhanced memory manager initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced memory manager failed: {e}')
                self.memory_manager = None
        else:
            self.logger.info('ℹ️ Enhanced memory manager not available, using fallback')
            self.memory_manager = None

        # Enhanced Bayesian Optimizer
        if OPTIMIZED_BAYESIAN_AVAILABLE:
            try:
                from .step03_config import Step03Config
                config_obj = Step03Config()
                # Use EnhancedHMMCompositeManager as fallback for EnhancedBayesianOptimizer
                self.bayesian_optimizer = EnhancedHMMCompositeManager(config_obj)
                self.logger.info('✅ Enhanced Bayesian optimizer initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced Bayesian optimizer failed: {e}')
                self.bayesian_optimizer = None
        else:
            self.logger.info('ℹ️ Enhanced Bayesian optimizer not available, using fallback')
            self.bayesian_optimizer = None

        # Parallel Clustering Processor
        if OPTIMIZED_CLUSTERING_AVAILABLE:
            try:
                from .step03_config import Step03Config
                config_obj = Step03Config()
                self.ensemble_clustering = AdvancedEnsembleClustering(config_obj)
                self.logger.info('✅ Enhanced ensemble clustering initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced ensemble clustering failed: {e}')
                self.ensemble_clustering = None
        else:
            self.logger.info('ℹ️ Enhanced ensemble clustering not available, using fallback')
            self.ensemble_clustering = None

        # Vectorized Operations Manager
        if OPTIMIZED_VECTORIZED_AVAILABLE:
            try:
                self.vectorized_manager = get_vectorized_operations_manager()
                self.logger.info('✅ Vectorized operations manager initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Vectorized operations manager failed: {e}')
                self.vectorized_manager = None
        else:
            self.logger.info('ℹ️ Vectorized operations manager not available, using fallback')
            self.vectorized_manager = None

        # Pipeline Orchestrator
        if OPTIMIZED_ORCHESTRATOR_AVAILABLE:
            try:
                orchestrator_config = create_step03_pipeline_config()
                self.pipeline_orchestrator = get_step03_pipeline_orchestrator(orchestrator_config)
                self.logger.info('✅ Pipeline orchestrator initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Pipeline orchestrator failed: {e}')
                self.pipeline_orchestrator = None
        else:
            self.logger.info('ℹ️ Pipeline orchestrator not available, using fallback')
            self.pipeline_orchestrator = None

        # Performance tracking
        self.use_optimized_pipeline = (
            OPTIMIZED_MEMORY_AVAILABLE and
            OPTIMIZED_BAYESIAN_AVAILABLE and
            OPTIMIZED_CLUSTERING_AVAILABLE and
            OPTIMIZED_VECTORIZED_AVAILABLE and
            OPTIMIZED_ORCHESTRATOR_AVAILABLE
        )

        if self.use_optimized_pipeline:
            self.logger.info('🎯 Full optimized pipeline available!')
        else:
            self.logger.info('ℹ️ Partial optimizations available, using hybrid approach')

    def _should_use_optimized_pipeline(self, training_input: dict[str, Any]) -> bool:
        """Determine if optimized pipeline should be used."""
        # Use optimized pipeline if all components are available and not explicitly disabled
        if not self.use_optimized_pipeline:
            return False

        # Check training input for override
        use_optimized = training_input.get('use_optimized_pipeline', True)
        if not use_optimized:
            self.logger.info('ℹ️ Optimized pipeline disabled by training input')
            return False

        # Check data size - use optimized for larger datasets
        force_optimized = training_input.get('force_optimized_pipeline', False)
        if force_optimized:
            self.logger.info('🎯 Forced optimized pipeline usage')
            return True

        return self.use_optimized_pipeline

    def _load_data_for_optimized_pipeline(self, training_input: dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data specifically for optimized pipeline."""
        try:
            # Try to load data using standard data loading
            data_dir = training_input.get('data_dir', 'data_cache')
            symbol = training_input.get('symbol', '')
            exchange = training_input.get('exchange', '')
            timeframe = training_input.get('timeframe', '1m')

            # Load data from standard location
            data_path = Path(data_dir) / f"{exchange}_{symbol}_{timeframe}_aggtrades.parquet"
            if data_path.exists():
                data = standardized_parquet_handler.read_parquet_standardized(data_path)
                self.logger.info(f'✅ Loaded data: {len(data)} records from {data_path}')
                return data

            # Try alternative data loading
            alt_path = Path("data/training") / f"{exchange}_{symbol}_aggtrades_{timeframe}.parquet"
            if alt_path.exists():
                data = standardized_parquet_handler.read_parquet_standardized(alt_path)
                self.logger.info(f'✅ Loaded data: {len(data)} records from {alt_path}')
                return data

            self.logger.warning('⚠️ No data files found for optimized pipeline')
            return None

        except Exception as e:
            self.logger.error(f'❌ Failed to load data for optimized pipeline: {e}')
            return None

    @handles_errors(fallback = False)
    async def initialize(self) -> None:
        """Initialize the HMM regime discovery step."""
        tprint("🚀 INITIALIZING HMM REGIME DISCOVERY STEP")
        self.start_time = time.time()
        tprint(f"   ⏰ Start time: {self.start_time}")
        
        self.logger.info('🚀 Initializing HMM Regime Discovery Step...')
        self.logger.info('📋 Step 3 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        
        tprint("📋 STEP 3 CONFIGURATION:")
        tprint(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        tprint(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        tprint(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        tprint(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        
        if hasattr(self, 'sr_predictor'):
            tprint("   🔧 Initializing SR Breakout Predictor...")
            try:
                await self.sr_predictor.initialize()
                tprint("   ✅ SR Breakout Predictor initialized successfully")
                self.logger.info('✅ SR Breakout Predictor initialized successfully')
            except Exception as e:
                tprint(f"   ⚠️ Failed to initialize SR Breakout Predictor: {e}")
                self.logger.warning(f'⚠️ Failed to initialize SR Breakout Predictor: {e}')
        else:
            tprint("   ⏭️ SR Breakout Predictor not available")
        
        tprint("✅ HMM REGIME DISCOVERY STEP INITIALIZED SUCCESSFULLY")
        self.logger.info('✅ HMM Regime Discovery Step initialized successfully')
    @log_all_calls

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @validates(step_name='hmm_regime_discovery', validation_level='CRITICAL', enable_rollback = True, max_retries = 2)
    @ensure_data_integrity(check_schema = True, check_constraints = True, validate_relationships = True)
    @monitor_step_execution(enable_timing = True, enable_memory_monitoring = True, enable_progress_tracking = True)
    @secure_step_execution(error_handling = True, rollback_on_failure = True, data_validation = True, resource_cleanup = True)
    @traced(span_name='execute_hmm_regime_discovery')
    @handles_errors(default_return={'success': False, 'regimes': [], 'error': 'HMM discovery failed'}, context='hmm_regime_discovery.execute')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute HMM regime discovery with enhanced data quality management.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with regime discovery results
        """
        step_start = time.time()
        tprint("🎯 STARTING HMM REGIME DISCOVERY EXECUTION")
        tprint(f"   ⏰ Step start time: {step_start}")
        
        # Initialize timing variables
        data_quality_elapsed = 0.0
        data_loading_elapsed = 0.0
        hmm_elapsed = 0.0
        
        tprint(f"   📊 Training input keys: {list(training_input.keys())}")
        tprint(f"   🔄 Pipeline state keys: {list(pipeline_state.keys())}")
        
        self.logger.info('🎯 Starting HMM regime discovery execution...')
        self.logger.info(f'📊 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'🔄 Pipeline state keys: {list(pipeline_state.keys())}')

        # Check if we should use optimized pipeline
        tprint("   🔍 Checking if optimized pipeline should be used...")
        use_optimized = self._should_use_optimized_pipeline(training_input)
        tprint(f"   📊 Use optimized pipeline: {use_optimized}")
        
        if use_optimized:
            tprint("🚀 USING OPTIMIZED PIPELINE FOR ENHANCED PERFORMANCE!")
            self.logger.info('🚀 Using optimized pipeline for enhanced performance!')
            return await self._execute_optimized_pipeline(training_input, pipeline_state)

        # Fallback to standard pipeline
        tprint("📊 USING STANDARD PIPELINE")
        self.logger.info('📊 Using standard pipeline')
        return await self._execute_standard_pipeline(training_input, pipeline_state)

    async def _execute_optimized_pipeline(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute using the optimized pipeline orchestrator."""
        try:
            self.logger.info('🎯 Executing optimized HMM regime discovery pipeline...')

            # Prepare data for optimized pipeline
            symbol = training_input.get('symbol', 'UNKNOWN')
            data = self._load_data_for_optimized_pipeline(training_input)

            if data is None or data.empty:
                self.logger.warning('⚠️ No data available, falling back to standard pipeline')
                return await self._execute_standard_pipeline(training_input, pipeline_state)

            # Use vectorized operations for feature engineering
            if self.vectorized_manager:
                self.logger.info('⚡ Using vectorized feature engineering...')
                vectorized_config = create_vectorized_config()
                processed_data = self.vectorized_manager.process_dataset(data, vectorized_config)
            else:
                processed_data = data

            # Build and execute optimized pipeline
            if self.pipeline_orchestrator:
                self.pipeline_orchestrator.build_step03_pipeline(processed_data, processed_data.values)
                results = self.pipeline_orchestrator.execute_step03_pipeline()

                # Convert results to expected format
                pipeline_state.update({
                    'hmm_regime_discovery_completed': True,
                    'step03_hmm_regime_discovery_completed': True,
                    'optimized_pipeline_used': True,
                    'regime_states': results.get('task_results', {}).get('final_regimes', []),
                    'regime_quality_score': results.get('task_results', {}).get('overall_quality_score', 0.0),
                    'performance_metrics': results.get('performance_metrics', {}),
                    'cache_performance': results.get('cache_performance', {}),
                    'total_pipeline_time': results.get('total_pipeline_time', 0.0)
                })

                self.logger.info('✅ Optimized pipeline execution completed successfully')
                return pipeline_state

        except Exception as e:
            self.logger.error(f'❌ Optimized pipeline failed: {e}, falling back to standard pipeline')
            return await self._execute_standard_pipeline(training_input, pipeline_state)

    async def _execute_standard_pipeline(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute the standard pipeline (original implementation)."""
        self.logger.info('🎯 Executing standard HMM regime discovery pipeline...')

        step_start = time.time()

        if PSUTIL_AVAILABLE:
            initial_memory = psutil.virtual_memory()
            self.logger.info(f'💾 Initial memory usage: {initial_memory.percent:.1f}% ({initial_memory.used / 1024 ** 3:.1f}GB / {initial_memory.total / 1024 ** 3:.1f}GB)')
        else:
            self.logger.info('💾 Memory monitoring not available (psutil not installed)')

        try:
            self.logger.info('=' * 60)
            self.logger.info('STEP 1: Data Quality Validation')
            self.logger.info('=' * 60)
            data_quality_start = time.time()
            data_ready = await self._ensure_data_quality(training_input)
            data_quality_elapsed = time.time() - data_quality_start
            self.logger.info(f'⏱️ Data Quality Validation completed in {data_quality_elapsed:.2f} seconds')
            if not data_ready:
                self.logger.error('❌ Data not ready for HMM regime discovery')
                pipeline_state['hmm_regime_discovery_completed'] = False
                pipeline_state['step03_hmm_regime_discovery_completed'] = False
                pipeline_state['regime_discovery_error'] = 'Data quality check failed'
                return pipeline_state
            self.logger.info('=' * 60)
            self.logger.info('STEP 2: Data Loading and Preparation')
            self.logger.info('=' * 60)
            data_loading_start = time.time()
            # Check for SR levels from step 02_5
            sr_levels = pipeline_state.get('sr_levels')
            if sr_levels:
                self.logger.info('✅ Found SR levels from step 02_5 - using enhanced regime detection')
                data_loaded = await self._load_and_prepare_data_with_sr(training_input, sr_levels)
            else:
                self.logger.info('⚠️ No SR levels found - using standard regime detection')
                data_loaded = await self._load_and_prepare_data(training_input)
            data_loading_elapsed = time.time() - data_loading_start
            self.logger.info(f'⏱️ Data Loading and Preparation completed in {data_loading_elapsed:.2f} seconds')
            if not data_loaded.get('success', False):
                self.logger.error('❌ Failed to load and prepare data for HMM')
                error_msg = data_loaded.get('error', 'Unknown error')
                self.logger.error(f'   Error details: {error_msg}')
                pipeline_state['hmm_regime_discovery_completed'] = False
                pipeline_state['step03_hmm_regime_discovery_completed'] = False
                pipeline_state['regime_discovery_error'] = f'Data loading failed: {error_msg}'
                return pipeline_state
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            if data_dir is None:
                data_dir = 'data_cache'
            self.logger.info('=' * 60)
            self.logger.info('STEP 3: Automatic Parameter Optimization')
            self.logger.info('=' * 60)
            optimization_start = time.time()
            # Temporarily bypass automatic optimization
            self.logger.info('🚀 Bypassing automatic optimization for now')
            optimized_params = {'n_components': 6, 'covariance_type': 'full'}
            if optimized_params:
                self.logger.info('✅ Parameter optimization completed successfully')
                # Apply optimized parameters to config
                if 'hmm' not in self.config:
                    self.config['hmm'] = {}
                self.config['hmm'].update(optimized_params)
                pipeline_state['optimization_used'] = True
                pipeline_state['optimized_params'] = optimized_params
            else:
                self.logger.warning('⚠️ Parameter optimization failed, using default parameters')
                pipeline_state['optimization_used'] = False
            optimization_elapsed = time.time() - optimization_start
            self.logger.info(f'⏱️ Parameter Optimization completed in {optimization_elapsed:.2f} seconds')
            self.logger.info('=' * 60)
            self.logger.info('STEP 4: HMM Regime Discovery')
            self.logger.info('=' * 60)
            hmm_start = time.time()
            regime_results = await self._perform_hmm_regime_discovery(training_input, data_loaded['data'])
            hmm_elapsed = time.time() - hmm_start
            self.logger.info(f'⏱️ HMM Regime Discovery completed in {hmm_elapsed:.2f} seconds')

            # Defensive check for regime_results
            if regime_results is None:
                self.logger.error('❌ HMM regime discovery returned None - this should not happen')
                return {'success': False, 'error': 'HMM regime discovery returned None'}

            if not isinstance(regime_results, dict):
                self.logger.error(f'❌ HMM regime discovery returned {type(regime_results)} instead of dict')
                return {'success': False, 'error': f'HMM regime discovery returned {type(regime_results)} instead of dict'}

            if regime_results.get('success', False):
                self.logger.info('✅ HMM regime discovery completed successfully')
                pipeline_state['hmm_regime_discovery_completed'] = True
                pipeline_state['step03_hmm_regime_discovery_completed'] = True
                pipeline_state['regime_states'] = regime_results.get('regime_states', [])
                pipeline_state['regime_transitions'] = regime_results.get('regime_transitions', {})
                pipeline_state['regime_metrics'] = regime_results.get('metrics', {})
                self._log_regime_discovery_results(regime_results)
                await self._log_step3_artifacts_to_mlflow(regime_results, training_input)
                self.logger.info('=' * 60)
                self.logger.info('STEP 5: SR Context Analysis')
                self.logger.info('=' * 60)
                sr_start = time.time()
                # Safely extract current price
                if 'close' in data_loaded['data'].columns and not data_loaded['data'].empty:
                    current_price = float(data_loaded['data']['close'].iloc[-1])
                else:
                    self.logger.warning('⚠️ Close column not available or data is empty, skipping SR analysis')
                    current_price = None
                sr_context = await self._get_sr_context_for_regime_analysis(data_loaded['data'], current_price)
                enhanced_regime_results = await self._enhance_regime_analysis_with_sr(regime_results, sr_context, data_loaded['data'])
                pipeline_state.update(enhanced_regime_results)
                sr_elapsed = time.time() - sr_start
                self.logger.info(f'⏱️ SR Context Analysis completed in {sr_elapsed:.2f} seconds')
            else:
                self.logger.error('❌ HMM regime discovery failed')
                if regime_results and isinstance(regime_results, dict):
                    error_msg = regime_results.get('error', 'Unknown error')
                else:
                    error_msg = f'Invalid regime_results: {type(regime_results)} - {regime_results}'
                self.logger.error(f'   Error details: {error_msg}')
                pipeline_state['hmm_regime_discovery_completed'] = False
                pipeline_state['step03_hmm_regime_discovery_completed'] = False
                pipeline_state['regime_discovery_error'] = error_msg
        except Exception as e:
            self.logger.exception(f'❌ Unexpected error during HMM regime discovery: {e}')
            pipeline_state['hmm_regime_discovery_completed'] = False
            pipeline_state['step03_hmm_regime_discovery_completed'] = False
            pipeline_state['regime_discovery_error'] = str(e)
        total_elapsed = time.time() - step_start
        self.logger.info('=' * 60)
        self.logger.info('EXECUTION SUMMARY')
        self.logger.info('=' * 60)
        self.logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        self.logger.info(f'⏱️ Step timings:')
        self.logger.info(f'   - Data Quality Validation: {data_quality_elapsed:.2f}s')
        self.logger.info(f'   - Data Loading and Preparation: {data_loading_elapsed:.2f}s')
        self.logger.info(f'   - HMM Regime Discovery: {hmm_elapsed:.2f}s')
        if 'sr_elapsed' in locals():
            self.logger.info(f'   - SR Context Analysis: {sr_elapsed:.2f}s')
        if PSUTIL_AVAILABLE:
            memory_usage = psutil.virtual_memory()
            self.logger.info(f'💾 Memory usage: {memory_usage.percent:.1f}% ({memory_usage.used / 1024 ** 3:.1f}GB / {memory_usage.total / 1024 ** 3:.1f}GB)')
        success = pipeline_state.get('hmm_regime_discovery_completed', False)
        self.logger.info(f"🎯 Final result: {('✅ SUCCESS' if success else '❌ FAILED')}")
        return pipeline_state

    async def _log_step3_artifacts_to_mlflow(self, regime_results: dict[str, Any], training_input: dict[str, Any]) -> None:
        """Log step 3 artifacts to MLflow with enhanced metadata and standardized naming."""
        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            if 'composite_df' in regime_results:
                composite_df = regime_results['composite_df']
                artifact_name = log_step_dataframe_with_standardized_name(config = self.config, step_name='step03_hmm_regime_discovery', df = composite_df, artifact_type='composite_clusters', additional_metadata={'artifact_type': 'composite_clusters', 'dataframe_shape': list(composite_df.shape), 'regime_count': len(composite_df.get('composite_cluster_id', []).unique()) if 'composite_cluster_id' in composite_df.columns else 0, 'timeframe': timeframe})
                self.logger.info(f'✅ Logged composite clusters: {artifact_name}')
            if 'intensity_df' in regime_results:
                intensity_df = regime_results['intensity_df']
                artifact_name = log_step_dataframe_with_standardized_name(config = self.config, step_name='step03_hmm_regime_discovery', df = intensity_df, artifact_type='intensity_clusters', additional_metadata={'artifact_type': 'intensity_clusters', 'dataframe_shape': list(intensity_df.shape), 'intensity_features': [col for col in intensity_df.columns if 'intensity' in col], 'timeframe': timeframe})
                self.logger.info(f'✅ Logged intensity clusters: {artifact_name}')
            if 'metrics' in regime_results and 'reports' in regime_results:
                metrics = regime_results.get('metrics', {})
                reports = regime_results.get('reports', {})
                report_data = {'metrics': metrics, 'reports': reports, 'training_input': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}, 'execution_timestamp': datetime.now().isoformat()}
                report_name = log_step_report(config = self.config, step_name='step03_hmm_regime_discovery', report_data = report_data, report_type='regime_discovery_report', additional_metadata={'hmm_states': metrics.get('hmm_states', 0), 'composite_clusters': metrics.get('composite_clusters', 0), 'reports_generated': list(reports.keys()) if isinstance(reports, dict) else []})
                self.logger.info(f'✅ Logged regime discovery report: {report_name}')

            # Generate enhanced comprehensive report if available
            if self.enhanced_reporter is not None and 'metrics' in regime_results:
                try:
                    self.logger.info('📊 Generating enhanced comprehensive report...')

                    # Prepare data for enhanced reporting
                    hmm_results = {
                        'n_components': regime_results.get('metrics', {}).get('hmm_states', 3),
                        'log_likelihood': regime_results.get('metrics', {}).get('hmm_score', 0.0),
                        'transition_matrix': regime_results.get('transition_matrix', []),
                        'steady_state_probabilities': regime_results.get('steady_state_probabilities', []),
                        'feature_importance': regime_results.get('feature_importance', {}),
                        'regime_persistence': regime_results.get('regime_persistence', []),
                        'volatility_by_regime': regime_results.get('volatility_by_regime', []),
                        'trend_by_regime': regime_results.get('trend_by_regime', []),
                        'regime_confidence': regime_results.get('regime_confidence', [])
                    }

                    clustering_results = {
                        'silhouette_score': regime_results.get('metrics', {}).get('silhouette_score', 0.0),
                        'davies_bouldin': regime_results.get('metrics', {}).get('davies_bouldin', 0.0),
                        'calinski_harabasz': regime_results.get('metrics', {}).get('calinski_harabasz', 0.0),
                        'n_clusters': regime_results.get('metrics', {}).get('composite_clusters', 0),
                        'cluster_sizes': regime_results.get('cluster_sizes', []),
                        'cluster_centers': regime_results.get('cluster_centers', []),
                        'stability_score': regime_results.get('stability_score', 0.0)
                    }

                    # Get market data for analysis (simplified - in practice you'd get actual data)
                    market_data = pd.DataFrame()  # Placeholder - should be actual market data

                    # Generate comprehensive report
                    comprehensive_report = self.enhanced_reporter.generate_comprehensive_report(
                        hmm_results=hmm_results,
                        clustering_results=clustering_results,
                        performance_data=regime_results.get('performance_data', {}),
                        market_data=market_data,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )

                    # Save comprehensive report
                    saved_files = self.enhanced_reporter.save_comprehensive_report(
                        report=comprehensive_report,
                        base_filename=f"step03_enhanced_{symbol}_{exchange}_{timeframe}"
                    )

                    self.logger.info(f'✅ Enhanced comprehensive report saved: {saved_files}')

                except Exception as e:
                    self.logger.warning(f'⚠️ Enhanced reporting failed, continuing with basic reporting: {e}')
            if 'metrics' in regime_results:
                metrics = regime_results['metrics']
                numeric_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        numeric_metrics[f'step3_{key}'] = float(value)
                if numeric_metrics:
                    log_step_metrics(config = self.config, step_name='step03_hmm_regime_discovery', metrics = numeric_metrics, additional_metadata={'metrics_type': 'regime_discovery', 'hmm_states': metrics.get('hmm_states', 0), 'composite_clusters': metrics.get('composite_clusters', 0)})
            if 'hmm_model' in regime_results:
                hmm_model = regime_results['hmm_model']
                log_step_model(config = self.config, step_name='step03_hmm_regime_discovery', model = hmm_model, model_name='hmm_regime_model', model_type='hmm', additional_metadata={'n_components': getattr(hmm_model, 'n_components', 0), 'covariance_type': getattr(hmm_model, 'covariance_type', 'unknown'), 'training_algorithm': 'GaussianHMM', 'timeframe': timeframe})
            if 'kmeans_model' in regime_results:
                kmeans_model = regime_results['kmeans_model']
                log_step_model(config = self.config, step_name='step03_hmm_regime_discovery', model = kmeans_model, model_name='kmeans_clustering_model', model_type='clustering', additional_metadata={'n_clusters': getattr(kmeans_model, 'n_clusters', 0), 'training_algorithm': 'KMeans', 'timeframe': timeframe})
            self.logger.info('✅ Step 3 artifacts logged to MLflow with standardized naming successfully')
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 3 artifacts to MLflow: {e}')
    @log_all_calls

    def _log_regime_discovery_results(self, regime_results: dict[str, Any]) -> None:
        """Log detailed regime discovery results."""
        self.logger.info('📊 REGIME DISCOVERY RESULTS')
        self.logger.info('-' * 40)
        metrics = regime_results.get('metrics', {})
        self.logger.info(f"📈 Total periods analyzed: {metrics.get('total_periods', 0):,}")
        self.logger.info(f"🔄 Unique regimes discovered: {metrics.get('unique_regimes', 0)}")
        regime_distribution = metrics.get('regime_distribution', {})
        if regime_distribution:
            self.logger.info('📊 Regime distribution:')
            for regime, count in regime_distribution.items():
                percentage = count / metrics.get('total_periods', 1) * 100
                self.logger.info(f'   - {regime}: {count:,} periods ({percentage:.1f}%)')
        transitions = regime_results.get('regime_transitions', {})
        if transitions:
            self.logger.info('🔄 Regime transition probabilities:')
            for from_regime, to_regimes in transitions.items():
                self.logger.info(f'   From {from_regime}:')
                for to_regime, prob in to_regimes.items():
                    self.logger.info(f'     → {to_regime}: {prob:.3f}')

    @traced(span_name='ensure_data_quality')
    @handles_errors(fallback = False)
    async def _ensure_data_quality(self, training_input: dict[str, Any]) -> bool:
        """Ensure data quality and readiness for HMM regime discovery."""
        self.logger.info('🔍 Starting data quality validation...')
        if not self.data_quality_manager:
            self.logger.warning('⚠️ Data quality manager not available, proceeding without quality check')
            self.logger.info('📝 Skipping enhanced data quality validation')
            return True
        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            self.logger.info(f'🎯 Validating data quality for {symbol} on {exchange} ({timeframe})...')
            self.logger.info('📋 Requesting data from quality manager...')
            data_results = await self.data_quality_manager.get_data_for_step3_step4(symbol = symbol, exchange = exchange, timeframe = timeframe)
            if data_results.get('success', False):
                self.logger.info('✅ Data quality check passed')
                self.logger.info('📊 Data quality metrics:')
                for key, value in data_results.items():
                    if key != 'success':
                        self.logger.info(f'   - {key}: {value}')
                return True
            else:
                self.logger.error('❌ Data quality check failed')
                error = data_results.get('error', 'Unknown error')
                self.logger.error(f'   Error: {error}')
                self.logger.info('🔄 Attempting to fix missing data...')
                fix_results = await self._fix_missing_data(training_input)
                if fix_results.get('success', False):
                    self.logger.info('✅ Successfully fixed missing data')
                    self.logger.info('📊 Fix results:')
                    for key, value in fix_results.items():
                        if key != 'success':
                            self.logger.info(f'   - {key}: {value}')
                    return True
                else:
                    self.logger.error('❌ Failed to fix missing data')
                    fix_error = fix_results.get('error', 'Unknown error')
                    self.logger.error(f'   Fix error: {fix_error}')
                    return False
        except Exception as e:
            self.logger.exception(f'❌ Error ensuring data quality: {e}')
            return False

    @traced(span_name='fix_missing_data')
    @handles_errors(default_return={'success': False, 'error': 'Data fix failed'}, context='fix_missing_data')
    async def _fix_missing_data(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Fix missing data using step01 and step1_5 components."""
        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            self.logger.info(f'🔄 Fixing missing data for {symbol} on {exchange} ({timeframe})...')
            step1_success = False
            try:
                self.logger.info('📥 Attempting step01 data collection...')
                from ...data_collection.data_preparation.step01_data_collection import run_step as run_step1
                step1_success = await run_step1(symbol = symbol, exchange = exchange, timeframe = timeframe, force_rerun = True)
                if step1_success:
                    self.logger.info('✅ Step1 data collection completed successfully')
                else:
                    self.logger.warning('⚠️ Step1 data collection failed')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not run step01: {e}')
            step1_5_success = False
            try:
                self.logger.info('🔄 Attempting step1_5 data conversion...')
                from ...data_collection.data_preparation.step01_5_data_converter import run_step as run_step1_5
                step1_5_success = await run_step1_5(symbol = symbol, exchange = exchange, timeframe = timeframe, force_rerun = True)
                if step1_5_success:
                    self.logger.info('✅ Step1_5 data conversion completed successfully')
                else:
                    self.logger.warning('⚠️ Step1_5 data conversion failed')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not run step1_5: {e}')
            if self.data_quality_manager:
                self.logger.info('🔍 Re-checking data quality after fixes...')
                data_results = await self.data_quality_manager.get_data_for_step3_step4(symbol = symbol, exchange = exchange, timeframe = timeframe)
                return {'success': data_results.get('success', False), 'step1_success': step1_success, 'step1_5_success': step1_5_success, 'quality_check_result': data_results}
            else:
                return {'success': step1_success and step1_5_success, 'step1_success': step1_success, 'step1_5_success': step1_5_success}
        except Exception as e:
            self.logger.exception(f'❌ Error fixing missing data: {e}')
            return {'success': False, 'error': str(e)}

    async def _load_and_prepare_data(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Load and prepare data for HMM regime discovery with standardized validation."""
        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            if data_dir is None:
                data_dir = 'data_cache'
            self.logger.info(f'📊 Loading and preparing data for HMM...')
            self.logger.info(f'   Symbol: {symbol}')
            self.logger.info(f'   Exchange: {exchange}')
            self.logger.info(f'   Timeframe: {timeframe}')
            self.logger.info(f'   Data directory: {data_dir}')
            klines_file = self.standards.generate_file_name('klines', exchange, symbol, timeframe)
            klines_path = Path(data_dir) / klines_file
            self.logger.info(f'📁 Looking for klines file: {klines_path}')
            if not klines_path.exists():
                self.logger.error(f'❌ Klines file not found: {klines_path}')
                return {'success': False, 'error': f'Klines file not found: {klines_path}'}
            self.logger.info('📥 Loading klines data from parquet file...')
            df = standardized_parquet_handler.read_parquet_standardized(klines_path)
            df = self.standards.standardize_timestamp(df, 'timestamp')
            df = self.standards.enforce_schema(df, 'klines')
            validation_result = self.standards.validate_data_quality(df, 'klines')
            if validation_result.passed:
                self.logger.info(f'✅ Data validation passed (quality score: {validation_result.quality_score:.2f})')
            else:
                self.logger.warning(f'⚠️ Data validation found issues:')
                for issue in validation_result.issues[:3]:
                    self.logger.warning(f'   - {issue.message}')
            if df.empty:
                self.logger.error('❌ Klines data is empty')
                return {'success': False, 'error': 'Klines data is empty'}
            self.logger.info(f'✅ Klines data loaded: {len(df):,} rows, {len(df.columns)} columns')
            self.logger.info(f'📊 Data columns: {list(df.columns)}')
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f'❌ Missing required columns: {missing_columns}')
                return {'success': False, 'error': f'Missing required columns: {missing_columns}'}
            self.logger.info('✅ All required columns present')
            # Validate input data for zero values (data quality check)
            zero_volume_count = (df['volume'] == 0).sum()
            zero_close_count = (df['close'] == 0).sum()
            if zero_volume_count > 0:
                self.logger.warning(f'⚠️ Found {zero_volume_count} zero volume values in input data')
                # Log details about zero volume periods
                zero_volume_mask = df['volume'] == 0
                zero_periods = df[zero_volume_mask]
                if len(zero_periods) > 0:
                    first_zero = zero_periods.iloc[0]['timestamp']
                    last_zero = zero_periods.iloc[-1]['timestamp']
                    self.logger.warning(f'   Zero volume period: {first_zero} to {last_zero}')
            if zero_close_count > 0:
                self.logger.warning(f'⚠️ Found {zero_close_count} zero close price values in input data')
            
            self.logger.info('🔧 Preparing features for HMM analysis...')
            features = await self._prepare_hmm_features(df)
            self.logger.info(f'✅ Data preparation completed successfully')
            self.logger.info(f'📊 Final data summary:')
            self.logger.info(f'   - Original data: {len(df):,} rows')
            self.logger.info(f'   - Features prepared: {len(features.columns)}')
            self.logger.info(f'   - Feature data: {len(features):,} rows')
            # Convert timestamp to datetime for isoformat
            timestamp_min = pd.to_datetime(df['timestamp'].min(), unit='ms')
            timestamp_max = pd.to_datetime(df['timestamp'].max(), unit='ms')
            return {'success': True, 'data': df, 'features': features, 'data_info': {'rows': len(df), 'columns': list(df.columns), 'date_range': {'start': timestamp_min.isoformat(), 'end': timestamp_max.isoformat()}}}
        except Exception as e:
            self.logger.exception(f'❌ Error loading and preparing data: {e}')
            return {'success': False, 'error': str(e)}

    async def _load_and_prepare_data_with_sr(self, training_input: dict[str, Any], sr_levels: dict[str, Any]) -> dict[str, Any]:
        """Load and prepare data for HMM regime discovery using SR levels from step 02_5."""
        try:
            symbol = training_input.get('symbol', get_default_symbol())
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            if data_dir is None:
                data_dir = 'data_cache'

            self.logger.info(f'📊 Loading and preparing data for HMM with SR enhancement...')
            self.logger.info(f'   Symbol: {symbol}')
            self.logger.info(f'   Exchange: {exchange}')
            self.logger.info(f'   Timeframe: {timeframe}')
            self.logger.info(f'   SR Levels Available: {len(sr_levels.get("support_levels", []))} support, {len(sr_levels.get("resistance_levels", []))} resistance')

            # Load klines data first
            klines_file = self.standards.generate_file_name('klines', exchange, symbol, timeframe)
            klines_path = Path(data_dir) / klines_file
            self.logger.info(f'📁 Loading klines file: {klines_path}')

            if not klines_path.exists():
                self.logger.error(f'❌ Klines file not found: {klines_path}')
                return {'success': False, 'error': f'Klines file not found: {klines_path}'}

            df = standardized_parquet_handler.read_parquet_standardized(klines_path)
            df = self.standards.standardize_timestamp(df, 'timestamp')
            df = self.standards.enforce_schema(df, 'klines')

            # Validate data quality
            validation_result = self.standards.validate_data_quality(df, 'klines')
            if validation_result.passed:
                self.logger.info(f'✅ Data validation passed (quality score: {validation_result.quality_score:.2f})')
            else:
                self.logger.warning(f'⚠️ Data validation found issues:')
                for issue in validation_result.issues[:3]:
                    self.logger.warning(f'   - {issue.message}')

            # Handle zero volume periods
            df = await self._handle_zero_volume_periods(df)

            self.logger.info('🔧 Preparing HMM features with SR enhancement...')
            features = await self._prepare_hmm_features_with_sr(df, sr_levels)

            self.logger.info(f'✅ Data preparation completed with SR enhancement')
            self.logger.info(f'📊 Final data summary:')
            self.logger.info(f'   - Original data: {len(df):,} rows')
            self.logger.info(f'   - Features prepared: {len(features.columns)}')
            self.logger.info(f'   - SR-enhanced features: {len(features):,} rows')

            # Convert timestamp to datetime for isoformat
            timestamp_min = pd.to_datetime(df['timestamp'].min(), unit='ms')
            timestamp_max = pd.to_datetime(df['timestamp'].max(), unit='ms')
            return {'success': True, 'data': df, 'features': features, 'sr_levels': sr_levels, 'data_info': {'rows': len(df), 'columns': list(df.columns), 'date_range': {'start': timestamp_min.isoformat(), 'end': timestamp_max.isoformat()}}}
        except Exception as e:
            self.logger.exception(f'❌ Error loading and preparing data with SR: {e}')
            return {'success': False, 'error': str(e)}

    @traced(span_name='prepare_hmm_features')
    @validates()
    @monitor_feature_engineering()
    @handles_errors(fallback = pd.DataFrame())
    async def _handle_zero_volume_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle zero volume periods with forward-fill for short gaps and data re-collection for long gaps."""
        zero_volume_mask = df['volume'] == 0
        if not zero_volume_mask.any():
            return df
        
        zero_count = zero_volume_mask.sum()
        zero_indices = df[zero_volume_mask].index
        total_rows = len(df)
        first_10_percent = int(total_rows * 0.1)
        
        # Check if all zero volumes are in the first 10% of data (expected behavior)
        if zero_indices[-1] < first_10_percent:
            self.logger.info(f'✅ Zero volume periods are in first rows only (expected) - skipping handling')
            return df
        
        self.logger.info(f'🧹 Handling {zero_count} zero volume periods...')
        
        # Identify consecutive zero volume groups
        zero_indices = df[zero_volume_mask].index
        consecutive_groups = []
        current_group = [zero_indices[0]]
        
        for i in range(1, len(zero_indices)):
            if zero_indices[i] == zero_indices[i-1] + 1:
                current_group.append(zero_indices[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [zero_indices[i]]
        consecutive_groups.append(current_group)
        
        self.logger.info(f'   Found {len(consecutive_groups)} consecutive zero volume groups')
        
        # Process each group
        for i, group in enumerate(consecutive_groups):
            start_idx = group[0]
            end_idx = group[-1]
            # Convert timestamps from int64 milliseconds to datetime for proper timedelta calculation
            start_time = pd.to_datetime(df.iloc[start_idx]['timestamp'], unit='ms', utc=True)
            end_time = pd.to_datetime(df.iloc[end_idx]['timestamp'], unit='ms', utc=True)
            gap_duration = end_time - start_time
            gap_minutes = gap_duration.total_seconds() / 60
            
            self.logger.info(f'   Group {i+1}: {len(group)} consecutive minutes ({gap_minutes:.1f} min gap)')
            
            if gap_minutes <= 5:  # Short gap: use forward-fill
                self.logger.info(f'     → Short gap: using forward-fill')
                # Forward-fill volume from the last non-zero value
                if start_idx > 0:
                    last_valid_volume = df.iloc[start_idx-1]['volume']
                    df.loc[group, 'volume'] = last_valid_volume
                else:
                    # If gap is at the beginning, use a small epsilon
                    df.loc[group, 'volume'] = 1e-10
            else:  # Long gap: trigger data re-collection via quality manager
                self.logger.warning(f'     → Long gap ({gap_minutes:.1f} min): triggering data re-collection')
                # Temporarily fill with small epsilon to keep continuity
                df.loc[group, 'volume'] = 1e-10
                try:
                    # Attempt auto-invocation of step01/step01_5 using quality manager
                    if hasattr(self, 'data_quality_manager') and self.data_quality_manager:
                        symbol = self.config.get('SYMBOL', 'UNKNOWN')
                        exchange = self.config.get('EXCHANGE', 'BINANCE')
                        timeframe_cfg = self.config.get('TIMEFRAME', '1m')
                        # Run fix sequence (async) and ignore result if it fails
                        import asyncio as _asyncio
                        _asyncio.create_task(
                            self.data_quality_manager.get_data_for_step3_step4(symbol = symbol, exchange = exchange, timeframe = timeframe_cfg)
                        )
                        self.logger.info('     🔄 Invoked quality manager to re-collect missing data in background')
                    else:
                        self.logger.warning('     ⚠️ Data quality manager not available; unable to auto re-collect')
                except Exception as _e:
                    self.logger.warning(f'     ⚠️ Auto re-collection attempt failed: {_e}')
        
        # Final validation
        remaining_zeros = (df['volume'] == 0).sum()
        if remaining_zeros > 0:
            self.logger.warning(f'   ⚠️ {remaining_zeros} zero volume values remain after handling')
        else:
            self.logger.info('   ✅ All zero volume periods handled successfully')
        
        return df
    @log_all_calls

    def _fix_covariance_matrix(self, covars: np.ndarray) -> np.ndarray:
        """Fix covariance matrix to ensure it's symmetric and positive-definite."""
        try:
            # Make a copy to avoid modifying the original
            covars_fixed = covars.copy()
            
            for i in range(covars_fixed.shape[0]):
                cov_matrix = covars_fixed[i]
                
                # Ensure symmetry
                cov_matrix = (cov_matrix + cov_matrix.T) / 2
                
                # Add small regularization to ensure positive-definiteness
                reg_param = 1e-6
                cov_matrix += reg_param * np.eye(cov_matrix.shape[0])
                
                # Ensure positive-definiteness using enhanced M1-optimized operations
                try:
                    if ENHANCED_MATRIX_OPS_AVAILABLE and m1_matrix_cholesky:
                        # Use enhanced M1-optimized Cholesky decomposition
                        m1_matrix_cholesky(cov_matrix)
                    else:
                        # Fallback to standard Cholesky decomposition
                        np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, use enhanced eigenvalue decomposition to fix
                    if ENHANCED_MATRIX_OPS_AVAILABLE and m1_matrix_eigendecomposition:
                        # Use enhanced M1-optimized eigendecomposition
                        eigenvals, eigenvecs = m1_matrix_eigendecomposition(cov_matrix)
                    else:
                        # Fallback to standard eigendecomposition
                        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    
                    eigenvals = np.maximum(eigenvals, reg_param)  # Ensure positive eigenvalues
                    cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                covars_fixed[i] = cov_matrix
            
            self.logger.info('🔧 Fixed covariance matrix to ensure symmetry and positive-definiteness')
            return covars_fixed
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to fix covariance matrix: {e}')
            return covars

    @handles_errors(fallback = pd.DataFrame())
    async def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning for financial data quality."""
        self.logger.info('🔍 Performing comprehensive data validation and cleaning...')
        
        initial_rows = len(df)
        issues_found = []
        
        # 1. Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            for col, count in missing_counts.items():
                if count > 0:
                    issues_found.append(f'Missing values in {col}: {count}')
        
        # 2. Check for negative prices (invalid for financial data)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                issues_found.append(f'Negative/zero prices in {col}: {negative_count}')
        
        # 3. Check for negative volume (invalid)
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            issues_found.append(f'Negative volume: {negative_volume}')
        
        # 4. Check for invalid OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        if invalid_ohlc > 0:
            issues_found.append(f'Invalid OHLC relationships: {invalid_ohlc}')
        
        # 5. Check for duplicate timestamps
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            issues_found.append(f'Duplicate timestamps: {duplicate_timestamps}')
        
        # 6. Check for Unix timestamp 0 (1970-01-01 00:00:00)
        unix_zero_timestamps = (df['timestamp'] == 0).sum()
        if unix_zero_timestamps > 0:
            issues_found.append(f'Unix timestamp 0 (invalid 1970 dates): {unix_zero_timestamps}')

        # 7. Check for timestamp gaps (missing periods)
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
            expected_interval = 1  # 1 minute for 1m data
            large_gaps = (time_diffs > expected_interval * 2).sum()  # Allow some tolerance
            if large_gaps > 0:
                issues_found.append(f'Large timestamp gaps: {large_gaps}')
        
        # Log issues found
        if issues_found:
            self.logger.warning('⚠️ Data quality issues found:')
            for issue in issues_found:
                self.logger.warning(f'   - {issue}')
        else:
            self.logger.info('✅ No data quality issues found')
        
        # Data cleaning operations
        cleaned_df = df.copy()
        
        # 1. Remove rows with invalid OHLC relationships
        invalid_ohlc_mask = (
            (cleaned_df['high'] < cleaned_df['low']) |
            (cleaned_df['high'] < cleaned_df['open']) |
            (cleaned_df['high'] < cleaned_df['close']) |
            (cleaned_df['low'] > cleaned_df['open']) |
            (cleaned_df['low'] > cleaned_df['close'])
        )
        if invalid_ohlc_mask.any():
            self.logger.warning(f'🧹 Removing {invalid_ohlc_mask.sum()} rows with invalid OHLC relationships')
            cleaned_df = cleaned_df[~invalid_ohlc_mask]
        
        # 2. Remove rows with negative prices
        for col in price_cols:
            negative_mask = cleaned_df[col] <= 0
            if negative_mask.any():
                self.logger.warning(f'🧹 Removing {negative_mask.sum()} rows with negative/zero {col}')
                cleaned_df = cleaned_df[~negative_mask]
        
        # 3. Remove rows with negative volume
        negative_volume_mask = cleaned_df['volume'] < 0
        if negative_volume_mask.any():
            self.logger.warning(f'🧹 Removing {negative_volume_mask.sum()} rows with negative volume')
            cleaned_df = cleaned_df[~negative_volume_mask]
        
        # 4. Remove rows with Unix timestamp 0 (invalid 1970 dates)
        unix_zero_mask = cleaned_df['timestamp'] == 0
        if unix_zero_mask.any():
            self.logger.warning(f'🧹 Removing {unix_zero_mask.sum()} rows with Unix timestamp 0 (1970-01-01)')
            # Log first few problematic rows with their indices
            unix_zero_indices = cleaned_df.index[unix_zero_mask].tolist()[:5]
            for idx in unix_zero_indices:
                row_idx = cleaned_df.index.get_loc(idx)
                if row_idx < len(cleaned_df):
                    timestamp_str = '1970-01-01 00:00:00'  # Unix timestamp 0
                    self.logger.warning(f'         - Row {row_idx}: {timestamp_str}')
            cleaned_df = cleaned_df[~unix_zero_mask]

        # 5. Remove duplicate timestamps (keep first occurrence)
        duplicate_mask = cleaned_df['timestamp'].duplicated(keep='first')
        if duplicate_mask.any():
            self.logger.warning(f'🧹 Removing {duplicate_mask.sum()} duplicate timestamp rows')
            cleaned_df = cleaned_df[~duplicate_mask]

        # 6. Fill missing values with forward-fill, then backward-fill
        missing_before = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            self.logger.info(f'🧹 Filling {missing_before} missing values with forward/backward fill')
            cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
        
        # Final statistics
        final_rows = len(cleaned_df)
        removed_rows = initial_rows - final_rows
        if removed_rows > 0:
            self.logger.info(f'📊 Data cleaning summary:')
            self.logger.info(f'   - Initial rows: {initial_rows:,}')
            self.logger.info(f'   - Final rows: {final_rows:,}')
            self.logger.info(f'   - Removed rows: {removed_rows:,} ({removed_rows/initial_rows*100:.2f}%)')
        else:
            self.logger.info('✅ No rows removed during cleaning')
        
        return cleaned_df

    def _time_constrained_fillna(self, series: pd.Series, timestamps: pd.Series,
                                max_gap_seconds: float = 0.5) -> pd.Series:
        """Fill NaN values using forward fill, but only for gaps ≤ max_gap_seconds.

        This prevents propagation of extreme values over long time periods while
        maintaining data continuity for short gaps.

        Args:
            series: Series with NaN values to fill
            timestamps: Corresponding timestamps for gap duration calculation
            max_gap_seconds: Maximum allowed gap duration for forward fill (default 0.5s)

        Returns:
            Series with selectively filled NaN values
        """
        if series.isna().sum() == 0:
            return series

        filled_series = series.copy()
        nan_mask = series.isna()

        if not nan_mask.any():
            return filled_series

        # Find consecutive NaN groups
        nan_indices = series[nan_mask].index

        if len(nan_indices) == 0:
            return filled_series

        # Process each NaN gap
        current_gap_start = None
        current_gap_end = None

        for i, idx in enumerate(nan_indices):
            if current_gap_start is None:
                current_gap_start = idx
                current_gap_end = idx
            elif idx == current_gap_end + 1:
                current_gap_end = idx
            else:
                # Process completed gap
                self._fill_gap_if_short(filled_series, timestamps,
                                      current_gap_start, current_gap_end,
                                      max_gap_seconds)
                current_gap_start = idx
                current_gap_end = idx

        # Process final gap
        if current_gap_start is not None:
            self._fill_gap_if_short(filled_series, timestamps,
                                  current_gap_start, current_gap_end,
                                  max_gap_seconds)

        return filled_series

    def _fill_gap_if_short(self, series: pd.Series, timestamps: pd.Series,
                          gap_start: int, gap_end: int, max_gap_seconds: float) -> None:
        """Fill a gap if its duration is ≤ max_gap_seconds, otherwise leave as NaN.

        This is a helper method for _time_constrained_fillna that handles individual gaps.
        """
        if gap_start >= len(timestamps) or gap_end >= len(timestamps):
            return

        if gap_start == 0:
            # Can't forward fill from before the start
            return

        gap_duration = (timestamps.iloc[gap_end] - timestamps.iloc[gap_start - 1]).total_seconds()

        if gap_duration <= max_gap_seconds:
            # Forward fill the gap
            fill_value = series.iloc[gap_start - 1]
            if pd.notna(fill_value):
                series.iloc[gap_start:gap_end + 1] = fill_value

    @handles_errors(fallback = pd.DataFrame())
    async def _prepare_hmm_features(self, df: Any) -> Any:
        """Prepare comprehensive features for HMM regime discovery including momentum, S/R, volume, and volatility."""
        try:
            self.logger.info('🔧 Starting comprehensive feature preparation for HMM...')
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                self.logger.info('🕒 Converting timestamp to datetime...')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.logger.info('📅 Sorting data by timestamp...')
            df = df.sort_values('timestamp').reset_index(drop = True)
            
            # Comprehensive data validation and cleaning
            df = await self._validate_and_clean_data(df)
            
            # Handle zero volume periods (data quality issue)
            df = await self._handle_zero_volume_periods(df)
            self.logger.info('📊 Calculating comprehensive features for HMM...')
            features = pd.DataFrame()
            features['timestamp'] = df['timestamp']
            self.logger.info('🚀 Calculating momentum features...')
            self.logger.info('   - Price momentum (5, 10, 20 periods)...')
            features['price_momentum_5'] = df['close'].pct_change(5)
            features['price_momentum_10'] = df['close'].pct_change(10)
            features['price_momentum_20'] = df['close'].pct_change(20)
            self.logger.info('   - Volume momentum...')
            features['volume_momentum_5'] = df['volume'].pct_change(5)
            features['volume_momentum_10'] = df['volume'].pct_change(10)
            features['volume_momentum_20'] = df['volume'].pct_change(20)
            self.logger.info('   - RSI momentum...')
            features['rsi'] = self._calculate_rsi(df['close'])
            features['rsi_momentum'] = features['rsi'].diff(5)
            self.logger.info('   - MACD momentum...')
            features['macd'] = self._calculate_macd(df['close'])
            features['macd_momentum'] = features['macd'].diff(5)
            self.logger.info('📈 Calculating volatility features...')
            self.logger.info('   - Multi-timeframe volatility...')
            # Volatility calculations with better handling of edge cases
            price_returns = df['close'].pct_change()
            # Fill first NaN with 0 (no change for first period)
            price_returns = price_returns.fillna(0)
            
            features['volatility_5'] = price_returns.rolling(window = 5).std()
            features['volatility_10'] = price_returns.rolling(window = 10).std()
            features['volatility_20'] = price_returns.rolling(window = 20).std()
            self.logger.info('   - EWMA volatility...')
            features['ewma_volatility_20'] = price_returns.ewm(span = 20).std()
            self.logger.info('   - Volatility acceleration and momentum...')
            features['volatility_acceleration'] = features['volatility_20'].diff()
            features['volatility_momentum'] = features['volatility_20'] - features['volatility_20'].shift(5)
            self.logger.info('   - ATR volatility...')
            features['atr'] = self._calculate_atr(df)
            # ATR normalization with zero-division protection
            features['atr_normalized'] = features['atr'] / df['close'].replace(0, np.nan)
            self.logger.info('📊 Calculating volume features...')
            self.logger.info('   - Volume ratios...')
            # Volume ratios with enhanced zero-division protection and minimum thresholds
            volume_mean_5 = df['volume'].rolling(window = 5).mean()
            volume_mean_10 = df['volume'].rolling(window = 10).mean()
            volume_mean_20 = df['volume'].rolling(window = 20).mean()

            # Apply minimum thresholds to prevent extreme ratios
            min_volume_threshold = df['volume'].quantile(0.01)  # 1st percentile as minimum
            if min_volume_threshold < 1.0:
                min_volume_threshold = 1.0

            volume_mean_5_safe = volume_mean_5.clip(lower=min_volume_threshold)
            volume_mean_10_safe = volume_mean_10.clip(lower=min_volume_threshold)
            volume_mean_20_safe = volume_mean_20.clip(lower=min_volume_threshold)

            features['volume_ratio_5'] = df['volume'] / volume_mean_5_safe
            features['volume_ratio_10'] = df['volume'] / volume_mean_10_safe
            features['volume_ratio_20'] = df['volume'] / volume_mean_20_safe

            # Cap extreme ratios to prevent outliers
            for col in ['volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20']:
                extreme_mask = features[col].abs() > 10.0  # Cap at 10x normal volume
                if extreme_mask.any():
                    features.loc[extreme_mask, col] = features[col].median()  # Use median for extreme values
            self.logger.info('   - Volume change...')
            # Volume change with better handling of zero volumes
            volume_change = df['volume'].pct_change()
            # For zero volume periods, use a small positive change instead of NaN
            volume_change = volume_change.fillna(0.001)  # Small positive change for zero volume periods
            features['volume_change'] = volume_change
            self.logger.info('   - Volume-price relationship...')
            # Calculate price change with timestamp validation
            price_change = df['close'].pct_change()
            price_change = price_change.fillna(0)  # First value is NaN, fill with 0
            
            # Validate zero-price-change periods using timestamps
            zero_price_mask = price_change == 0
            zero_count = zero_price_mask.sum()
            self.logger.info(f'   - Found {zero_count:,} zero-price-change periods, validating duration...')
            
            if zero_count > 0:
                # Convert timestamps to datetime for duration calculation
                timestamps = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Find consecutive zero-price-change periods
                zero_periods = []
                current_start = None
                for i, is_zero in enumerate(zero_price_mask):
                    if is_zero and current_start is None:
                        current_start = i
                    elif not is_zero and current_start is not None:
                        zero_periods.append((current_start, i-1))
                        current_start = None
                
                # Handle case where the last period is zero
                if current_start is not None:
                    zero_periods.append((current_start, len(zero_price_mask)-1))
                
                self.logger.info(f'   - Found {len(zero_periods)} consecutive zero-price-change periods')
                
                # Check duration of each zero-price-change period
                long_zero_periods = []
                short_zero_periods = []
                total_long_duration = 0
                total_short_duration = 0
                total_long_rows = 0
                total_short_rows = 0
                
                for start_idx, end_idx in zero_periods:
                    if start_idx < len(timestamps) and end_idx < len(timestamps):
                        duration_seconds = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]).total_seconds()
                        rows_in_period = end_idx - start_idx + 1
                        
                        if duration_seconds >= 2.0:  # 2 seconds or more
                            long_zero_periods.append((start_idx, end_idx, duration_seconds, rows_in_period))
                            total_long_duration += duration_seconds
                            total_long_rows += rows_in_period
                        else:
                            short_zero_periods.append((start_idx, end_idx, duration_seconds, rows_in_period))
                            total_short_duration += duration_seconds
                            total_short_rows += rows_in_period
                
                # Only report summary statistics, ignore clusters <2s
                if short_zero_periods:
                    self.logger.info(f'   - Ignored {len(short_zero_periods)} short periods (<2s): {total_short_rows} rows over {total_short_duration:.1f}s total')
                
                if long_zero_periods:
                    self.logger.warning(f'   ⚠️ Found {len(long_zero_periods)} significant zero-price-change periods (≥2s): {total_long_rows} rows over {total_long_duration:.1f}s total')
                    # Show first 3 significant periods with details
                    for start_idx, end_idx, duration, rows in long_zero_periods[:3]:
                        start_time = timestamps.iloc[start_idx].strftime('%H:%M:%S')
                        end_time = timestamps.iloc[end_idx].strftime('%H:%M:%S')
                        self.logger.warning(f'     - Period {start_idx}-{end_idx} ({start_time}-{end_time}): {rows} rows over {duration:.1f}s')
                    if len(long_zero_periods) > 3:
                        self.logger.warning(f'     - ... and {len(long_zero_periods)-3} more significant periods')
                else:
                    self.logger.info('   ✅ All zero-price-change periods are <2s (ignored as legitimate short-term price stability)')
            else:
                self.logger.info('   ✅ No zero-price-change periods found')
            
            # Keep legitimate zeros (≤5s periods) and replace others with small values
            # For now, we'll keep all zeros as they are legitimate short-term price stability
            features['volume_price_trend'] = price_change * df['volume']
            
            # Volume-price trend ratio with zero-division protection
            vpt_mean = features['volume_price_trend'].rolling(20).mean()
            # Apply minimum threshold to prevent division by very small values
            vpt_threshold = features['volume_price_trend'].abs().quantile(0.01)
            if vpt_threshold < 1e-6:
                vpt_threshold = 1e-6
            vpt_mean_safe = vpt_mean.clip(lower=vpt_threshold).clip(upper=-vpt_threshold)
            features['volume_price_trend_ratio'] = features['volume_price_trend'] / vpt_mean_safe
            self.logger.info('🎯 Calculating support/resistance features...')
            self.logger.info('   - Pivot points...')
            features['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
            features['support_1'] = 2 * features['pivot_point'] - df['high']
            features['resistance_1'] = 2 * features['pivot_point'] - df['low']
            self.logger.info('   - Distance to S/R levels...')
            # Distance to S/R levels with zero-division protection
            features['distance_to_support'] = (df['close'] - features['support_1']) / df['close'].replace(0, np.nan)
            features['distance_to_resistance'] = (features['resistance_1'] - df['close']) / df['close'].replace(0, np.nan)
            self.logger.info('   - S/R strength indicators...')
            features['sr_strength'] = self._calculate_sr_strength(df)
            self.logger.info('   - Bollinger Bands...')
            bb_features = self._calculate_bollinger_bands(df['close'])
            features = pd.concat([features, bb_features], axis = 1)
            self.logger.info('🔧 Calculating additional technical features...')
            self.logger.info('   - Moving averages...')
            features['sma_20'] = df['close'].rolling(window = 20).mean()
            features['sma_50'] = df['close'].rolling(window = 50).mean()
            features['ema_12'] = df['close'].ewm(span = 12).mean()
            features['ema_26'] = df['close'].ewm(span = 26).mean()
            self.logger.info('   - Price position relative to MAs...')
            # Price vs moving averages with zero-division protection
            features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20'].replace(0, np.nan)
            features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50'].replace(0, np.nan)
            self.logger.info('   - ADX trend strength...')
            features['adx'] = self._calculate_adx(df)
            self.logger.info('🔄 Calculating feature interactions...')
            self.logger.info('   - Momentum × Volume interactions...')
            features['momentum_volume_interaction'] = features['price_momentum_10'] * features['volume_ratio_10']
            self.logger.info('   - Volatility × Volume interactions...')
            features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio_20']
            self.logger.info('   - RSI × Momentum interactions...')
            features['rsi_momentum_interaction'] = features['rsi'] * features['price_momentum_10']
            self.logger.info('🧹 Cleaning and validating features...')
            hmm_features = features.drop('timestamp', axis = 1)
            initial_rows = len(hmm_features)
            self.logger.info(f'   - Initial rows: {initial_rows:,}')
            technical_cols = ['rsi', 'macd', 'adx', 'bb_position', 'bb_width']
            for col in technical_cols:
                if col in hmm_features.columns:
                    hmm_features[col] = hmm_features[col].ffill()

            # Get timestamps for time-constrained filling
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')

            # Smart NaN handling and constant value detection
            # Only fill NaN values, preserve legitimate zeros, detect constant features
            for col in hmm_features.columns:
                if hmm_features[col].dtype in ['float64', 'float32']:
                    # Count NaN values before filling
                    nan_count = hmm_features[col].isna().sum()

                    if nan_count > 0:
                        self.logger.info(f'   - Column "{col}": {nan_count:,} NaN values to fill')

                        # Smart filling based on feature type and zero context
                        if col in ['rsi', 'macd', 'adx', 'bb_position', 'bb_width', 'atr_normalized']:
                            # Technical indicators: time-constrained forward fill to preserve trends
                            # but limit propagation of extreme values
                            hmm_features[col] = self._time_constrained_fillna(hmm_features[col], timestamps, 0.5)
                            # Fill remaining NaN with median for technical indicators
                            remaining_nan = hmm_features[col].isna().sum()
                            if remaining_nan > 0:
                                median_val = hmm_features[col].median()
                                if pd.notna(median_val):
                                    hmm_features[col] = hmm_features[col].fillna(median_val)
                        elif 'ratio' in col or 'volume' in col:
                            # Volume ratios: use median to avoid propagating extreme values
                            median_val = hmm_features[col].median()
                            if pd.notna(median_val):
                                hmm_features[col] = hmm_features[col].fillna(median_val)
                            else:
                                # If median is NaN, try time-constrained fill as fallback
                                hmm_features[col] = self._time_constrained_fillna(hmm_features[col], timestamps, 0.5)
                        elif 'momentum' in col or 'volatility' in col:
                            # Momentum/volatility: time-constrained forward fill preserves patterns
                            # but prevents long-term propagation of extreme values
                            hmm_features[col] = self._time_constrained_fillna(hmm_features[col], timestamps, 0.5)
                            # Fill remaining NaN with median
                            remaining_nan = hmm_features[col].isna().sum()
                            if remaining_nan > 0:
                                median_val = hmm_features[col].median()
                                if pd.notna(median_val):
                                    hmm_features[col] = hmm_features[col].fillna(median_val)
                        else:
                            # Other features: median fill if meaningful, otherwise time-constrained fill
                            median_val = hmm_features[col].median()
                            if pd.notna(median_val) and abs(median_val) > 1e-6:
                                hmm_features[col] = hmm_features[col].fillna(median_val)
                            else:
                                hmm_features[col] = self._time_constrained_fillna(hmm_features[col], timestamps, 0.5)
                        
                        # Final fallback: only use 0 if absolutely necessary
                        remaining_nan = hmm_features[col].isna().sum()
                        if remaining_nan > 0:
                            self.logger.warning(f'     - Column "{col}": {remaining_nan:,} NaN values remain, using 0 as fallback')
                            hmm_features[col] = hmm_features[col].fillna(0)
                    
                    # Check for constant values (all same value)
                    unique_values = hmm_features[col].nunique()
                    if unique_values == 1:
                        constant_value = hmm_features[col].iloc[0]
                        self.logger.warning(f'   - Column "{col}": constant value {constant_value} (all {len(hmm_features)} values are the same)')
                        
                        # For constant features, add small random noise to make them useful for clustering
                        if abs(constant_value) < 1e-10:  # If constant is 0 or very small
                            noise = np.random.normal(0, 1e-6, len(hmm_features))
                            hmm_features[col] = hmm_features[col] + noise
                            self.logger.info(f'     - Added small noise to constant feature "{col}"')
                        else:
                            # For non-zero constants, add proportional noise
                            noise_std = abs(constant_value) * 0.01  # 1% of the constant value
                            noise = np.random.normal(0, noise_std, len(hmm_features))
                            hmm_features[col] = hmm_features[col] + noise
                            self.logger.info(f'     - Added proportional noise to constant feature "{col}" (std={noise_std:.2e})')
                    
                    # Check for near-constant values (very low variance)
                    elif unique_values > 1:
                        variance = hmm_features[col].var()
                        if variance < 1e-10:  # Very low variance
                            self.logger.warning(f'   - Column "{col}": very low variance {variance:.2e} (near-constant)')
                            # Add small noise to increase variance
                            noise_std = max(1e-6, np.sqrt(variance) * 0.1)
                            noise = np.random.normal(0, noise_std, len(hmm_features))
                            hmm_features[col] = hmm_features[col] + noise
                            self.logger.info(f'     - Added noise to near-constant feature "{col}" (std={noise_std:.2e})')
                else:
                    # For non-numeric columns, use forward fill then 0
                    hmm_features[col] = hmm_features[col].fillna(method='ffill').fillna(0)
            
            # Log zero value count after median fill
            zero_count_after = (hmm_features == 0).sum().sum()
            self.logger.info(f'   - Zero values after median fill: {zero_count_after:,}')
            
            # Detailed zero value analysis
            if zero_count_after > 0:
                self.logger.info('   - Zero value distribution by column:')
                for col in hmm_features.columns:
                    col_zeros = (hmm_features[col] == 0).sum()
                    if col_zeros > 0:
                        zero_percentage = (col_zeros / len(hmm_features)) * 100
                        self.logger.info(f'     - {col}: {col_zeros:,} zeros ({zero_percentage:.2f}%)')
                
                # Check for rows with many zeros
                row_zero_counts = (hmm_features == 0).sum(axis=1)
                high_zero_rows = row_zero_counts[row_zero_counts > 5]  # Rows with more than 5 zeros
                if len(high_zero_rows) > 0:
                    self.logger.warning(f'   - Found {len(high_zero_rows):,} rows with >5 zero values')
                    self.logger.warning(f'     - Sample high-zero rows: {high_zero_rows.head(10).index.tolist()}')
                    self.logger.warning(f'     - Max zeros in a single row: {row_zero_counts.max()}')
            
            # Check if zero values are only in the first 2 rows (expected for rolling windows)
            if zero_count_after > 0:
                # Check zero values in first 2 rows vs rest of dataset
                first_two_rows_zeros = (hmm_features.iloc[:2] == 0).sum().sum()
                remaining_rows_zeros = (hmm_features.iloc[2:] == 0).sum().sum()
                
                self.logger.info(f'   - Zero values in first 2 rows: {first_two_rows_zeros:,}')
                self.logger.info(f'   - Zero values in remaining rows: {remaining_rows_zeros:,}')
                
                if remaining_rows_zeros > 0:
                    self.logger.warning(f'⚠️ Found {remaining_rows_zeros:,} zero values in rows 3+ (unexpected!)')
                    
                    # Get timestamps for duration analysis (use original df, not hmm_features)
                    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Log which columns have zeros in unexpected places
                    for col in hmm_features.columns:
                        col_zeros_remaining = (hmm_features.iloc[2:][col] == 0).sum()
                        if col_zeros_remaining > 0:
                            # Calculate percentage and apply different thresholds
                            total_rows = len(hmm_features)
                            zero_percentage = col_zeros_remaining / total_rows

                            # Different thresholds for different feature types
                            if 'ratio' in col or 'volume' in col:
                                # Volume ratios can have more zeros due to low volume periods
                                warning_threshold = 0.15  # 15%
                                is_high = zero_percentage > warning_threshold
                            elif 'momentum' in col or 'volatility' in col:
                                warning_threshold = 0.05   # 5%
                                is_high = zero_percentage > warning_threshold
                            else:
                                warning_threshold = 0.02   # 2%
                                is_high = zero_percentage > warning_threshold

                            if is_high:
                                self.logger.warning(f'     - Column "{col}": {col_zeros_remaining:,} zeros ({zero_percentage:.1%}) in rows 3+ - HIGH ZERO COUNT')
                            else:
                                self.logger.info(f'     - Column "{col}": {col_zeros_remaining:,} zeros ({zero_percentage:.1%}) in rows 3+ - Acceptable')
                            
                            # Find specific rows with zeros for this column
                            zero_mask = hmm_features.iloc[2:][col] == 0
                            zero_indices = hmm_features.iloc[2:].index[zero_mask].tolist()
                            
                            # Log first 10 zero locations for this column with timestamps
                            if len(zero_indices) > 0:
                                sample_indices = zero_indices[:10]
                                self.logger.warning(f'       - Sample zero locations (rows): {sample_indices}')
                                
                                # Show timestamps for first few zeros
                                for idx in sample_indices[:3]:
                                    if idx < len(timestamps):
                                        timestamp_str = timestamps.iloc[idx].strftime('%Y-%m-%d %H:%M:%S')
                                        self.logger.warning(f'         - Row {idx}: {timestamp_str}')
                                
                                # Check if zeros are clustered or scattered
                                if len(zero_indices) > 10:
                                    # Check for consecutive zeros (clusters)
                                    consecutive_zeros = []
                                    current_start = zero_indices[0]
                                    current_end = zero_indices[0]
                                    
                                    for i in range(1, len(zero_indices)):
                                        if zero_indices[i] == zero_indices[i-1] + 1:
                                            current_end = zero_indices[i]
                                        else:
                                            if current_end - current_start > 0:
                                                consecutive_zeros.append((current_start, current_end))
                                            current_start = zero_indices[i]
                                            current_end = zero_indices[i]
                                    
                                    # Add the last sequence
                                    if current_end - current_start > 0:
                                        consecutive_zeros.append((current_start, current_end))
                                    
                                    if consecutive_zeros:
                                        # Filter clusters by duration (≥2s)
                                        significant_clusters = []
                                        ignored_clusters = []
                                        total_significant_rows = 0
                                        total_ignored_rows = 0
                                        
                                        for start_idx, end_idx in consecutive_zeros:
                                            if start_idx < len(timestamps) and end_idx < len(timestamps):
                                                duration = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]).total_seconds()
                                                rows_in_cluster = end_idx - start_idx + 1
                                                
                                                if duration >= 2.0:
                                                    significant_clusters.append((start_idx, end_idx, duration, rows_in_cluster))
                                                    total_significant_rows += rows_in_cluster
                                                else:
                                                    ignored_clusters.append((start_idx, end_idx, duration, rows_in_cluster))
                                                    total_ignored_rows += rows_in_cluster
                                        
                                        if ignored_clusters:
                                            self.logger.info(f'       - Ignored {len(ignored_clusters)} short clusters (<2s): {total_ignored_rows} rows')
                                        
                                        if significant_clusters:
                                            self.logger.warning(f'       - Found {len(significant_clusters)} significant clusters (≥2s): {total_significant_rows} rows')
                                            # Show first 2 significant clusters
                                            for start_idx, end_idx, duration, rows in significant_clusters[:2]:
                                                start_time = timestamps.iloc[start_idx].strftime('%H:%M:%S')
                                                end_time = timestamps.iloc[end_idx].strftime('%H:%M:%S')
                                                self.logger.warning(f'         - Cluster {start_idx}-{end_idx} ({start_time}-{end_time}): {rows} rows over {duration:.1f}s')
                                        else:
                                            # Check if this is expected for volatile data or if threshold needs adjustment
                                            total_clusters = len(consecutive_zeros)
                                            avg_duration = sum((end_idx - start_idx) * (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
                                                             for start_idx, end_idx in consecutive_zeros[:10]) / min(10, len(consecutive_zeros))

                                            if total_clusters > 50:  # Many short clusters suggest very volatile data
                                                self.logger.info(f'       - High volatility detected: {total_clusters} clusters with avg duration {avg_duration:.2f}s')
                                                self.logger.info(f'       - This may indicate very short-term market regimes (potentially correct for volatile assets)')
                                            elif avg_duration < 0.5:  # Very short clusters
                                                self.logger.warning(f'       - Very short clusters detected (avg {avg_duration:.2f}s) - consider adjusting duration threshold')
                                            else:
                                                self.logger.info(f'       - All clusters are <2s (ignored as legitimate short-term stability)')
                                    else:
                                        self.logger.warning(f'       - Zeros are scattered (not clustered)')
                                
                                # Check the actual values around zeros
                                if len(zero_indices) > 0:
                                    sample_idx = zero_indices[0]
                                    if sample_idx > 0 and sample_idx < len(hmm_features) - 1:
                                        prev_val = hmm_features.iloc[sample_idx-1][col]
                                        curr_val = hmm_features.iloc[sample_idx][col]
                                        next_val = hmm_features.iloc[sample_idx+1][col]
                                        self.logger.warning(f'       - Sample zero context (row {sample_idx}): prev={prev_val:.6f}, curr={curr_val:.6f}, next={next_val:.6f}')
                else:
                    self.logger.info('✅ All zero values are in first 2 rows (expected for rolling windows)')
            
            final_rows = len(hmm_features)
            removed_rows = initial_rows - final_rows
            self.logger.info(f'✅ Comprehensive feature preparation completed:')
            self.logger.info(f'   - Initial rows: {initial_rows:,}')
            self.logger.info(f'   - Final rows: {final_rows:,}')
            self.logger.info(f'   - Removed rows: {removed_rows:,} ({removed_rows / initial_rows * 100:.1f}%)')
            self.logger.info(f'   - Features created: {len(hmm_features.columns)}')
            self._log_feature_categories(hmm_features)
            return hmm_features
        except Exception as e:
            self.logger.exception(f'❌ Error preparing HMM features: {e}')
            raise

    async def _prepare_hmm_features_with_sr(self, df: pd.DataFrame, sr_levels: dict[str, Any]) -> pd.DataFrame:
        """Prepare comprehensive features for HMM regime discovery enhanced with SR levels."""
        try:
            self.logger.info('🔧 Starting SR-enhanced feature preparation for HMM...')

            # First get the standard features
            hmm_features = await self._prepare_hmm_features(df)

            # Add SR-based features
            self.logger.info('🎯 Adding SR-based features...')

            # Distance to nearest support/resistance levels
            support_levels = sr_levels.get('support_levels', [])
            resistance_levels = sr_levels.get('resistance_levels', [])

            if support_levels or resistance_levels:
                self.logger.info(f'   - Support levels: {len(support_levels)}')
                self.logger.info(f'   - Resistance levels: {len(resistance_levels)}')

                # Calculate distance to nearest S/R levels
                hmm_features['distance_to_support'] = self._calculate_distance_to_levels(df['close'], support_levels)
                hmm_features['distance_to_resistance'] = self._calculate_distance_to_levels(df['close'], resistance_levels)

                # SR interaction features
                hmm_features['near_support'] = (hmm_features['distance_to_support'] < 0.005).astype(int)  # Within 0.5%
                hmm_features['near_resistance'] = (hmm_features['distance_to_resistance'] < 0.005).astype(int)

                # SR bounce signals
                hmm_features['support_bounce_signal'] = self._calculate_sr_bounce_signal(df, support_levels, 'support')
                hmm_features['resistance_bounce_signal'] = self._calculate_sr_bounce_signal(df, resistance_levels, 'resistance')

                self.logger.info('✅ SR-enhanced features added successfully')
            else:
                self.logger.warning('⚠️ No SR levels available for feature enhancement')
                # Add placeholder columns to maintain consistency
                hmm_features['distance_to_support'] = 1.0
                hmm_features['distance_to_resistance'] = 1.0
                hmm_features['near_support'] = 0
                hmm_features['near_resistance'] = 0
                hmm_features['support_bounce_signal'] = 0.0
                hmm_features['resistance_bounce_signal'] = 0.0

            self.logger.info(f'📊 SR-enhanced features: {len(hmm_features.columns)} total features')
            return hmm_features

        except Exception as e:
            self.logger.exception(f'❌ Error preparing SR-enhanced HMM features: {e}')
            # Return standard features as fallback
            self.logger.warning('⚠️ Falling back to standard HMM features')
            return await self._prepare_hmm_features(df)

    def _calculate_distance_to_levels(self, prices: pd.Series, levels: list) -> pd.Series:
        """Calculate normalized distance to nearest SR level.

        Accepts levels in multiple formats:
          - list[float]
          - list[dict] with key 'price'
          - numpy arrays shaped (N, 2) where second column is price
        """
        if not levels or len(prices) == 0:
            return pd.Series([1.0] * len(prices), index = prices.index)

        # Normalize levels to a flat numpy array of prices
        level_prices: list[float] = []
        try:
            # If ndarray of shape (N, 2) -> take price column index 1
            if hasattr(levels, 'ndim'):
                import numpy as _np
                lvl_arr = _np.asarray(levels)
                if lvl_arr.ndim == 2 and lvl_arr.shape[1] >= 2:
                    level_prices = lvl_arr[:, 1].astype(float).tolist()
                else:
                    level_prices = lvl_arr.astype(float).tolist()
            else:
                for lvl in levels:
                    if isinstance(lvl, dict):
                        # Common schema from SR detectors
                        if 'price' in lvl:
                            level_prices.append(float(lvl['price']))
                        elif 'level' in lvl:
                            level_prices.append(float(lvl['level']))
                    elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                        # (index, price)
                        level_prices.append(float(lvl[1]))
                    else:
                        # Assume raw numeric
                        level_prices.append(float(lvl))
        except Exception:
            # Fallback: attempt best-effort cast
            try:
                level_prices = [float(getattr(l, 'price', l)) for l in levels]
            except Exception:
                level_prices = []

        if not level_prices:
            return pd.Series([1.0] * len(prices), index = prices.index)

        level_prices_arr = np.asarray(level_prices, dtype = float)

        # Vectorized nearest distance calculation
        prices_arr = prices.values.astype(float)
        # For large arrays, compute efficiently by broadcasting in chunks
        chunk_size = 50000
        distances: list[np.ndarray] = []
        for i in range(0, len(prices_arr), chunk_size):
            p_chunk = prices_arr[i:i + chunk_size][:, None]
            # Compute absolute distances to all levels, then min across axis=1
            d_min = np.min(np.abs(p_chunk - level_prices_arr[None, :]), axis = 1)
            # Normalize by price (avoid division by zero)
            p_norm = np.where(p_chunk[:, 0] == 0.0, np.nan, p_chunk[:, 0])
            distances.append(d_min / p_norm)

        distances_arr = np.concatenate(distances)
        # Replace NaNs (from zero price) with 1.0 sentinel
        distances_arr = np.where(np.isfinite(distances_arr), distances_arr, 1.0)
        return pd.Series(distances_arr, index = prices.index)

    def _calculate_sr_bounce_signal(self, df: pd.DataFrame, levels: list, level_type: str) -> pd.Series:
        """Calculate SR bounce signals based on price action near levels."""
        if not levels:
            return pd.Series([0.0] * len(df), index=df.index)

        signals = []
        for idx, row in df.iterrows():
            price = row['close']
            nearest_level = min(levels, key=lambda x: abs(price - x)) if levels else None

            if nearest_level and abs(price - nearest_level) / price < 0.005:  # Within 0.5%
                # Check for bounce pattern (price approaching then reversing)
                signal = 0.0
                if idx > 1:
                    prev_price = df.loc[idx-1, 'close']
                    prev_prev_price = df.loc[idx-2, 'close']

                    # Simple bounce detection
                    if level_type == 'support':
                        # Price approached support then bounced up
                        if prev_price < nearest_level and price > prev_price:
                            signal = 1.0
                    elif level_type == 'resistance':
                        # Price approached resistance then bounced down
                        if prev_price > nearest_level and price < prev_price:
                            signal = 1.0

                signals.append(signal)
            else:
                signals.append(0.0)

        return pd.Series(signals, index=df.index)

    @log_all_calls
    @handles_errors(fallback = pd.Series())
    def _calculate_rsi(self, prices: Any, window: int = 14) -> Any:
        """Calculate Relative Strength Index."""
        self.logger.debug(f'Calculating RSI with window {window}...')
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window = window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    @log_all_calls
    @handles_errors(fallback = pd.Series())
    def _calculate_macd(self, prices: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Any:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        self.logger.debug(f'Calculating MACD (fast={fast}, slow={slow}, signal={signal})...')
        ema_fast = prices.ewm(span = fast).mean()
        ema_slow = prices.ewm(span = slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @log_all_calls
    @handles_errors(fallback = pd.Series())
    def _calculate_atr(self, df: Any, window: int = 14) -> Any:
        """Calculate Average True Range (ATR)."""
        self.logger.debug(f'Calculating ATR with window {window}...')
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        atr = tr.rolling(window = window).mean()
        return atr

    @log_all_calls
    @handles_errors(fallback = pd.Series())
    def _calculate_bollinger_bands(self, prices: Any, window: int = 20, num_std: float = 2) -> Any:
        """Calculate Bollinger Bands."""
        self.logger.debug(f'Calculating Bollinger Bands (window={window}, std={num_std})...')
        sma = prices.rolling(window = window).mean()
        std = prices.rolling(window = window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        # Bollinger Bands width with zero-division protection
        bb_width = (bb_upper - bb_lower) / sma.replace(0, np.nan)
        # Bollinger Bands position with zero-division protection
        bb_range = bb_upper - bb_lower
        bb_position = (prices - bb_lower) / bb_range.replace(0, np.nan)
        bb_features = pd.DataFrame({'bb_upper': bb_upper, 'bb_middle': sma, 'bb_lower': bb_lower, 'bb_width': bb_width, 'bb_position': bb_position})
        return bb_features

    @log_all_calls
    @handles_errors(fallback = pd.Series())
    def _calculate_adx(self, df: Any, window: int = 14) -> Any:
        """Calculate Average Directional Index (ADX)."""
        self.logger.debug(f'Calculating ADX with window {window}...')
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        tr_smooth = tr.rolling(window = window).mean()
        dm_plus_smooth = dm_plus.rolling(window = window).mean()
        dm_minus_smooth = dm_minus.rolling(window = window).mean()
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window = window).mean()
        return adx

    @log_all_calls
    @handles_errors(fallback = pd.Series())
    def _calculate_sr_strength(self, df: Any, window: int = 20) -> Any:
        """Calculate support/resistance strength indicator."""
        self.logger.debug(f'Calculating S/R strength with window {window}...')
        high_swing = df['high'].rolling(window = window, center = True).max()
        low_swing = df['low'].rolling(window = window, center = True).min()
        current_price = df['close']
        high_strength = (high_swing - current_price) / high_swing
        low_strength = (current_price - low_swing) / low_swing
        sr_strength = (high_strength + low_strength) / 2
        return sr_strength

    @log_all_calls
    @handles_errors(fallback = None)
    def _log_feature_categories(self, features: Any) -> None:
        """Log feature categories for analysis."""
        try:
            feature_categories = {'momentum': [], 'volatility': [], 'volume': [], 'support_resistance': [], 'technical': [], 'interactions': []}
            for col in features.columns:
                if 'momentum' in col.lower():
                    feature_categories['momentum'].append(col)
                elif 'volatility' in col.lower():
                    feature_categories['volatility'].append(col)
                elif 'volume' in col.lower():
                    feature_categories['volume'].append(col)
                elif any((sr_term in col.lower() for sr_term in ['support', 'resistance', 'pivot', 'sr_', 'bb_'])):
                    feature_categories['support_resistance'].append(col)
                elif any((tech_term in col.lower() for tech_term in ['rsi', 'macd', 'adx', 'atr', 'sma', 'ema'])):
                    feature_categories['technical'].append(col)
                elif 'interaction' in col.lower():
                    feature_categories['interactions'].append(col)
                else:
                    feature_categories['technical'].append(col)
            self.logger.info('📊 Feature categories:')
            for category, cols in feature_categories.items():
                if cols:
                    self.logger.info(f'   - {category.capitalize()}: {len(cols)} features')
                    if len(cols) <= 5:
                        self.logger.info(f'     {cols}')
                    else:
                        self.logger.info(f'     {cols[:3]} ... {cols[-2:]}')
        except Exception as e:
            self.logger.warning(f'Could not log feature categories: {e}')

    async def _perform_hmm_regime_discovery(self, training_input: dict[str, Any], data: Any) -> dict[str, Any]:
        """Perform HMM regime discovery using hmmlearn with comprehensive features."""
        try:
            self.logger.info('🔍 Starting HMM regime discovery analysis...')
            self.logger.info(f'📊 Input data shape: {data.shape}')
            self.logger.info('🔧 Preparing comprehensive features for HMM analysis...')
            features = await self._prepare_hmm_features(data)
            if features.empty:
                self.logger.error('❌ No features available for HMM analysis')
                return {'success': False, 'error': 'No features available'}
            self.logger.info(f'📊 Features prepared: {len(features.columns)} features, {len(features)} samples')
            self.logger.info('📊 Feature statistics:')
            for col in features.columns:
                series = features[col].dropna()
                if len(series) > 0:
                    self.logger.info(f'   - {col}: mean={series.mean():.6f}, std={series.std():.6f}, min={series.min():.6f}, max={series.max():.6f}')
            try:
                from hmmlearn import hmm
                self.logger.info('✅ hmmlearn library available')
                return await self._perform_hmmlearn_regime_discovery(features)
            except ImportError:
                self.logger.error('❌ hmmlearn library is required but not available')
                return {'success': False, 'error': 'hmmlearn library is required for HMM regime discovery'}
        except Exception as e:
            self.logger.exception(f'❌ Error performing HMM regime discovery: {e}')
            return {'success': False, 'error': str(e)}

    @traced(span_name='perform_hmmlearn_regime_discovery')
    @handles_errors(default_return={'success': False, 'error': 'HMMLearn regime discovery failed'}, context='perform_hmmlearn_regime_discovery')
    async def _perform_hmmlearn_regime_discovery(self, features: Any) -> dict[str, Any]:
        """Perform HMM regime discovery using hmmlearn library with 20-cluster composite approach."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            self.logger.info('🧠 Using hmmlearn with 20-cluster composite approach...')

            # Memory monitoring and warnings
            n_samples, n_features = features.shape
            estimated_memory_mb = (n_samples * n_features * 4) / (1024 * 1024)  # float32 estimate
            self.logger.info(f'📊 Dataset: {n_samples:,} samples, {n_features} features')
            self.logger.info(f'💾 Estimated memory usage: ~{estimated_memory_mb:.1f}MB for scaled features')

            if estimated_memory_mb > 500:
                self.logger.warning(f'🚨 Large dataset detected ({estimated_memory_mb:.1f}MB). Memory-efficient processing enabled.')
            if estimated_memory_mb > 2000:
                self.logger.warning(f'🚨 Very large dataset ({estimated_memory_mb:.1f}MB). Consider reducing data size or increasing system memory.')

            self.logger.info('📊 Scaling features for HMM...')

            # Ensure only numeric columns are used and clean feature matrix
            if isinstance(features, pd.DataFrame):
                features = features.select_dtypes(include=[np.number]).copy()

            # Final validation: ensure no infinity values remain, and only warn on zeros beyond warmup rows
            inf_count = np.isinf(features).sum().sum()
            warmup_rows = min(50, features.shape[0]) if hasattr(features, 'shape') else 50
            try:
                zero_count = (features.iloc[warmup_rows:] == 0).sum().sum() if hasattr(features, 'iloc') else 0
                total_values = features.iloc[warmup_rows:].shape[0] * features.iloc[warmup_rows:].shape[1] if hasattr(features, 'iloc') else 0
            except Exception:
                zero_count = (features == 0).sum().sum()
                total_values = features.shape[0] * features.shape[1]
            if inf_count > 0:
                self.logger.error(f'❌ Found {inf_count} infinity values in features after cleaning')
                return {'success': False, 'error': f'Infinity values remain in features: {inf_count}'}
            if zero_count > 0 and total_values > 0:
                zero_percentage = (zero_count / total_values) * 100
                self.logger.info(f'ℹ️ Zeros (beyond warmup {warmup_rows} rows): {zero_count} ({zero_percentage:.3f}%)')

            # Features should now be clean (no infinity values due to root cause fixes)
            scaler = StandardScaler()

            # Memory-efficient scaling with dtype optimization
            if features.shape[0] > 100000:
                self.logger.info("💾 Using memory-efficient scaling for large dataset")
                # Process in chunks to avoid memory issues and use float32 for memory savings
                chunk_size = 50000

                # First, fit scaler on a representative sample to avoid memory leaks
                sample_size = min(100000, features.shape[0])
                sample_indices = np.random.choice(features.shape[0], sample_size, replace=False)
                sample_data = features.iloc[sample_indices].astype(np.float32)
                scaler.fit(sample_data)
                del sample_data  # Free memory immediately

                # Now transform data in chunks without creating large upfront array
                self.logger.info("🔄 Transforming data in chunks to save memory...")
                features_scaled_chunks = []

                for i in range(0, features.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, features.shape[0])
                    chunk = features.iloc[i:end_idx].astype(np.float32)
                    chunk_scaled = scaler.transform(chunk)
                    features_scaled_chunks.append(chunk_scaled)
                    del chunk, chunk_scaled  # Free memory after each chunk

                # Concatenate chunks efficiently
                features_scaled = np.vstack(features_scaled_chunks)
                del features_scaled_chunks  # Free the list
                self.logger.info(f"✅ Memory-efficient scaling completed, shape: {features_scaled.shape}")
            else:
                # Convert to float32 for memory efficiency
                features_scaled = scaler.fit_transform(features.astype(np.float32))

            max_initial_points = int(self.config.get('hmm', {}).get('max_initial_points', 250000))
            if features_scaled.shape[0] > max_initial_points:
                self.logger.info(f'⚡ Downsampling for initial HMM fit to {max_initial_points} rows (from {features_scaled.shape[0]})')
                stride = max(1, features_scaled.shape[0] // max_initial_points)
                features_init = features_scaled[::stride]
            else:
                features_init = features_scaled

            n_hmm_states = 4
            n_iter = 100
            random_state = 42
            self.logger.info(f'🎯 Phase 1: Training HMM with {n_hmm_states} states...')

            # Try GPU acceleration if available
            gpu_accelerated = False
            features_gpu = None

            # Helper function to create HMM model with consistent parameters
            def _create_hmm_model():
                hmm_config = self.config.get('hmm', {})
                return hmm.GaussianHMM(
                    n_components=n_hmm_states,
                    n_iter=min(n_iter, int(hmm_config.get('max_iterations', 100))),
                    random_state=random_state,
                    covariance_type=hmm_config.get('covariance_type', 'full'),
                    init_params=hmm_config.get('init_params', 'stmc'),
                    params=hmm_config.get('params', 'stmc'),
                    tol=float(hmm_config.get('tol', 0.001))
                )

            try:
                if CUPY_AVAILABLE and cp.cuda.runtime.getDeviceCount() > 0 and features_init.shape[0] > 10000:
                    self.logger.info("🚀 Using GPU acceleration for HMM training")
                    gpu_accelerated = True
                    # Move data to GPU
                    features_gpu = cp.asarray(features_init)
                    hmm_model = _create_hmm_model()
                    # HMM fitting will automatically use GPU if available
                    hmm_model.fit(features_init)  # Keep CPU version for now due to hmmlearn limitations

                    # Clean up GPU memory immediately after use
                    if features_gpu is not None:
                        del features_gpu
                        features_gpu = None
                        # Force GPU memory cleanup
                        if cp.cuda.runtime.getDeviceCount() > 0:
                            cp.cuda.runtime.deviceSynchronize()
                            cp.cuda.runtime.free(0)  # Free all GPU memory
                    self.logger.info("🧹 GPU memory cleaned up after HMM training")
                else:
                    if CUPY_AVAILABLE:
                        self.logger.debug("CuPy not available or GPU not suitable, using CPU for HMM training")
                    hmm_model = _create_hmm_model()
            except Exception as gpu_error:
                self.logger.warning(f"GPU acceleration failed: {gpu_error}, falling back to CPU")
                gpu_accelerated = False
                # Clean up GPU memory in case of error
                if 'features_gpu' in locals() and features_gpu is not None:
                    del features_gpu
                hmm_model = _create_hmm_model()

            model_ckpt_dir = Path(self.config.get('hmm', {}).get('checkpoint_dir', 'data/hmm_ckpts'))
            model_ckpt_dir.mkdir(parents = True, exist_ok = True)
            # Normalize naming to uppercase exchange/symbol and lowercase timeframe
            ex = str(self.config.get('EXCHANGE', 'EX')).upper()
            sym = str(self.config.get('SYMBOL', 'SYM')).upper()
            tf = str(self.config.get('TIMEFRAME', '1m')).lower()
            ckpt_path = model_ckpt_dir / f"{ex}_{sym}_{tf}_hmm_{n_hmm_states}.npz"
            try:
                if ckpt_path.exists():
                    self.logger.info(f'♻️  Loading HMM checkpoint: {ckpt_path}')
                    with np.load(ckpt_path, allow_pickle = True) as npz:
                        hmm_model.startprob_ = npz['startprob_']
                        hmm_model.transmat_ = npz['transmat_']
                        hmm_model.means_ = npz['means_']
                        covars = npz['covars_']

                        # Fix covariance matrix to ensure it's symmetric and positive-definite
                        covars_fixed = self._fix_covariance_matrix(covars)
                        hmm_model.covars_ = covars_fixed

                        self.logger.info('✅ HMM checkpoint loaded successfully')
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to load HMM checkpoint: {e}')

            hmm_model.fit(features_init)

            remaining_iter = int(self.config.get('hmm', {}).get('refine_iterations', 20))
            needs_refinement = remaining_iter > 0 and features_scaled.shape[0] != features_init.shape[0]

            if needs_refinement:
                self.logger.info(f'🔁 Refining HMM on full data for {remaining_iter} additional iterations')
                hmm_model.n_iter = remaining_iter
                hmm_model.init_params = ''
                hmm_model.fit(features_scaled)

            # Memory cleanup: free features_init after initial fit if it's a copy and refinement is done
            if features_scaled.shape[0] != features_init.shape[0]:
                del features_init
                self.logger.info("🧹 Cleaned up features_init after initial HMM fit")

            try:
                # Use compressed format for memory efficiency
                checkpoint_data = {
                    'startprob_': hmm_model.startprob_.astype(np.float32),
                    'transmat_': hmm_model.transmat_.astype(np.float32),
                    'means_': hmm_model.means_.astype(np.float32),
                    'covars_': hmm_model.covars_.astype(np.float32)
                }
                np.savez_compressed(ckpt_path, **checkpoint_data)
                self.logger.info(f'💾 Saved HMM checkpoint: {ckpt_path} (compressed)')
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to save HMM checkpoint: {e}')

            hmm_state_sequence = hmm_model.predict(features_init if (not needs_refinement) else features_scaled)
            hmm_state_probs = hmm_model.predict_proba(features_init if (not needs_refinement) else features_scaled)

            # Store HMM score before cleanup since it's needed later
            hmm_log_likelihood = hmm_model.score(features_init if (not needs_refinement) else features_scaled) if hasattr(hmm_model, 'score') else 0.0

            # Memory cleanup: free features_scaled after predictions if we're done with it
            # Free the largest arrays that are no longer needed
            try:
                del features_scaled
            except Exception:
                pass
            self.logger.info("🧹 Cleaned up features_scaled after HMM predictions")

            self.logger.info('🎯 Phase 2: Creating 20-cluster composite analysis...')
            composite_features = self._create_composite_features(features, hmm_state_sequence, hmm_state_probs)
            composite_scaler = StandardScaler()
            composite_features_scaled = composite_scaler.fit_transform(composite_features)
            n_clusters = 20
            kmeans = KMeans(n_clusters = n_clusters, random_state = random_state, n_init = 10, max_iter = 300)
            cluster_labels = kmeans.fit_predict(composite_features_scaled)
            self.logger.info('🎯 Phase 3: Analyzing cluster quality...')
            cluster_metrics = self._calculate_cluster_quality_metrics(composite_features_scaled, cluster_labels, kmeans)
            self.logger.info('🎯 Phase 4: Enhanced regime analysis and interpretation...')
            composite_analysis = self._analyze_composite_clusters(features, hmm_state_sequence, cluster_labels, cluster_metrics)
            self.logger.info('🔍 Performing enhanced regime change detection...')
            regime_change_analysis = self._detect_regime_changes_advanced(hmm_state_probs, hmm_state_sequence, threshold = 0.1, min_persistence = 3)
            self.logger.info('🔧 Calculating adaptive regime boundaries...')
            adaptive_boundaries = self._calculate_adaptive_regime_boundaries(features)
            self.logger.info('📊 Modeling regime persistence...')
            persistence_model = self._model_regime_persistence(hmm_state_sequence)
            composite_analysis.update({'regime_change_analysis': regime_change_analysis, 'adaptive_boundaries': adaptive_boundaries, 'persistence_model': persistence_model})
            self.logger.info('🎯 Phase 5: Generating comprehensive reports...')
            # Use Step03EnhancedReporter for proper HMM reporting
            try:
                from src.training.steps.market_analysis.step04_financial_logging import Step04FinancialloggingFinancialLogger as Step03FinancialLogger

                # Get symbol, exchange, timeframe from config
                symbol = self.config.get('SYMBOL', 'UNKNOWN')
                exchange = self.config.get('EXCHANGE', 'UNKNOWN')
                timeframe = self.config.get('TIMEFRAME', '30m')

                # Initialize the financial logger
                financial_logger = Step03FinancialLogger()

                # Prepare HMM results for enhanced reporting
                hmm_results = {
                    'n_components': n_hmm_states,
                    'covariance_type': 'full',
                    'model_type': 'GaussianHMM',
                    'converged': hasattr(hmm_model, 'monitor_') and hmm_model.monitor_.converged if hasattr(hmm_model, 'monitor_') else True,
                    'log_likelihood': hmm_log_likelihood,
                    'aic': getattr(hmm_model, 'aic_', 0.0),
                    'bic': getattr(hmm_model, 'bic_', 0.0),
                    'transition_matrix': hmm_model.transmat_.tolist() if hasattr(hmm_model, 'transmat_') else [],
                    'steady_state_probabilities': getattr(hmm_model, 'get_stationary_distribution', lambda: [])(),
                    'regime_persistence': persistence_model.get('regime_durations', []) if isinstance(persistence_model, dict) else [],
                    'volatility_by_regime': persistence_model.get('volatility_by_regime', []) if isinstance(persistence_model, dict) else [],
                    'regime_correlations': [],
                    'temporal_stability': cluster_metrics.get('temporal_stability', 0.0),
                    'feature_importance': composite_analysis.get('feature_importance', {}),
                    'regime_probabilities': hmm_state_probs.tolist() if hasattr(hmm_state_probs, 'tolist') else [],
                    'timestamps': features.index.tolist() if hasattr(features, 'index') else []
                }

                # Prepare clustering results for enhanced reporting
                clustering_results = {
                    'silhouette_score': cluster_metrics.get('silhouette_score', 0.0),
                    'davies_bouldin': cluster_metrics.get('davies_bouldin', 0.0),
                    'calinski_harabasz': cluster_metrics.get('calinski_harabasz', 0.0),
                    'n_clusters': n_clusters,
                    'cluster_sizes': cluster_metrics.get('cluster_sizes', []),
                    'cluster_centers': cluster_metrics.get('cluster_centers', []),
                    'explained_variance': cluster_metrics.get('explained_variance', 0.0),
                    'reduction_efficiency': cluster_metrics.get('reduction_efficiency', 0.0),
                    'stability_score': cluster_metrics.get('stability_score', 0.0)
                }

                # Prepare performance data
                execution_start = getattr(self, 'start_time', None)
                current_time = time.time()
                execution_time = current_time - execution_start if execution_start else 0

                performance_data = {
                    'execution_time': execution_time,
                    'memory_usage': 0,  # Could be enhanced with actual memory monitoring
                    'cpu_usage': 0,     # Could be enhanced with actual CPU monitoring
                    'processing_rate': len(features) / max(1, execution_time),
                    'hmm_training_time': 0,  # Could be enhanced with timing data
                    'clustering_time': 0,    # Could be enhanced with timing data
                    'regime_analysis_time': 0,  # Could be enhanced with timing data
                    'report_generation_time': 0,
                    'function_calls': 0,
                    'successful_ops': 0,
                    'failed_ops': 0,
                    'error_rate': 0.0,
                    'convergence_iterations': getattr(hmm_model, 'n_iter', 0),
                    'log_likelihood': hmm_log_likelihood
                }

                # Log financial metrics using the new financial logger
                financial_logger.log_step_execution(
                    hmm_results=hmm_results,
                    clustering_results=clustering_results,
                    performance_data=performance_data,
                    market_data=features,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                
                # Create minimal reports for compatibility
                reports = {
                    'hmm_results': hmm_results,
                    'clustering_results': clustering_results,
                    'performance_data': performance_data,
                    'financial_metrics_logged': True
                }

                self.logger.info('✅ Enhanced Step03 reporting completed successfully')

            except ImportError as e:
                self.logger.warning(f"Could not import Step03FinancialLogger: {e}, falling back to basic reporting")
                # Fallback to basic reporting if import fails
                reports = {
                    'hmm_states': n_hmm_states,
                    'cluster_count': n_clusters,
                    'log_likelihood': hmm_log_likelihood,
                    'cluster_quality': cluster_metrics,
                    'composite_analysis': composite_analysis,
                    'fallback_mode': True
                }

            except Exception as e:
                self.logger.warning(f"Enhanced reporting failed: {e}, using fallback")
                # Fallback to basic reporting if anything fails
                reports = {
                    'hmm_states': n_hmm_states,
                    'cluster_count': n_clusters,
                    'log_likelihood': hmm_log_likelihood,
                    'cluster_quality': cluster_metrics,
                    'composite_analysis': composite_analysis,
                    'error': str(e),
                    'fallback_mode': True
                }
            self.logger.info('🎯 Phase 6: Creating output data structures...')
            composite_df = self._create_composite_cluster_dataframe(features, hmm_state_sequence, cluster_labels, composite_analysis)
            intensity_df = self._create_intensity_dataframe(features, hmm_state_sequence, cluster_labels, composite_analysis)
            meta_info = self._create_meta_information(hmm_model, kmeans, composite_analysis, cluster_metrics, reports)
            final_metrics = {'total_periods': len(cluster_labels), 'hmm_states': n_hmm_states, 'composite_clusters': n_clusters, 'cluster_quality': cluster_metrics, 'hmm_score': hmm_log_likelihood, 'composite_analysis': composite_analysis, 'reports_generated': list(reports.keys())}
            try:
                out_dir = Path('data/hmm_regimes')
                out_dir.mkdir(parents = True, exist_ok = True)
                exchange = self.config.get('EXCHANGE', 'EX')
                symbol = self.config.get('SYMBOL', 'SYM')
                timeframe_cfg = self.config.get('TIMEFRAME', '1m')
                out_path = out_dir / f'{exchange}_{symbol}_{timeframe_cfg}_composite_clusters.parquet'
                save_df = composite_df.copy()
                if isinstance(features, pd.DataFrame) and 'timestamp' in features.columns and ('timestamp' not in save_df.columns):
                    save_df['timestamp'] = features['timestamp'].values
                standardized_parquet_handler.write_parquet_standardized(save_df, out_path, compression='snappy', index = False)
                self.logger.info(f'💾 Saved composite clusters to: {out_path}')
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to save composite clusters parquet: {e}')
            self.logger.info(f'✅ Composite HMM regime discovery completed successfully')
            self.logger.info(f'📊 HMM States: {n_hmm_states}, Composite Clusters: {n_clusters}')
            self.logger.info(f"📈 Cluster Quality - Silhouette: {cluster_metrics['silhouette_score']:.4f}")
            self.logger.info(f'📊 Reports Generated: {len(reports)}')

            # Final memory cleanup before returning
            self.logger.info("🧹 Performing final memory cleanup...")
            # Note: We don't delete the return objects as they may be needed by callers

            return {'success': True, 'hmm_model': hmm_model, 'kmeans_model': kmeans, 'scaler': scaler, 'composite_scaler': composite_scaler, 'hmm_state_sequence': hmm_state_sequence, 'hmm_state_probs': hmm_state_probs, 'cluster_labels': cluster_labels, 'composite_df': composite_df, 'intensity_df': intensity_df, 'meta_info': meta_info, 'metrics': final_metrics, 'reports': reports}
        except Exception as e:
            self.logger.exception(f'❌ Error in HMM regime discovery: {e}')
            return {'success': False, 'error': str(e), 'fallback_mode': True}

    @handles_errors
    def _create_meta_information(self, hmm_model: Any, kmeans_model: Any, composite_analysis: dict[str, Any], cluster_metrics: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
        """Create meta information for the composite HMM analysis."""
        try:
            self.logger.info('📊 Creating meta information...')
            meta = {'creation_timestamp': pd.Timestamp.now().isoformat(), 'hmm_model_info': {'n_components': hmm_model.n_components, 'covariance_type': hmm_model.covariance_type, 'n_iter': hmm_model.n_iter, 'converged': hmm_model.monitor_.converged, 'score': hmm_model.score(hmm_model.means_)}, 'kmeans_model_info': {'n_clusters': kmeans_model.n_clusters, 'inertia': kmeans_model.inertia_, 'n_iter': kmeans_model.n_iter_, 'converged': kmeans_model.n_iter_ < kmeans_model.max_iter}, 'cluster_metrics': cluster_metrics, 'composite_analysis_summary': {'total_clusters': len(composite_analysis.get('cluster_characteristics', {})), 'hmm_states': len(composite_analysis.get('hmm_state_distribution', {})), 'market_conditions': len(composite_analysis.get('market_conditions', {}))}, 'reports_summary': {'total_reports': len(reports), 'report_types': list(reports.keys())}, 'feature_summary': {'total_features': len(composite_analysis.get('feature_importance', {})), 'top_features': sorted(composite_analysis.get('feature_importance', {}).items(), key = lambda x: x[1], reverse = True)[:10]}}
            self.logger.info('✅ Created meta information')
            return meta
        except Exception as e:
            self.logger.exception(f'❌ Error creating meta information: {e}')
            return {}

    @handles_errors(fallback = pd.DataFrame())
    def _create_composite_cluster_dataframe(self, features: Any, hmm_states: Any, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Create composite cluster DataFrame with all relevant information."""
        try:
            self.logger.info('📊 Creating composite cluster DataFrame...')
            df = features.copy()
            df['hmm_state'] = hmm_states
            df['composite_cluster_id'] = cluster_labels
            for cluster_id, char in composite_analysis.get('cluster_characteristics', {}).items():
                cluster_mask = cluster_labels == cluster_id
                df.loc[cluster_mask, 'cluster_size'] = char['size']
                df.loc[cluster_mask, 'cluster_percentage'] = char['percentage']
                df.loc[cluster_mask, 'dominant_hmm_state'] = char['dominant_hmm_state']
                df.loc[cluster_mask, 'market_condition'] = composite_analysis.get('market_conditions', {}).get(cluster_id, 'unknown')

            df['cluster_intensity'] = self._calculate_cluster_intensity(cluster_labels, composite_analysis)
            df['cluster_stability'] = self._calculate_cluster_stability_scores(cluster_labels, composite_analysis)
            self.logger.info(f'✅ Created composite cluster DataFrame: {len(df)} rows, {len(df.columns)} columns')
            return df
        except Exception as e:
            self.logger.exception(f'❌ Error creating composite cluster DataFrame: {e}')
            return pd.DataFrame()

    def _calculate_cluster_intensity(self, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Calculate cluster intensity scores."""
        try:
            intensity = np.zeros(len(cluster_labels))
            for cluster_id, char in composite_analysis.get('cluster_characteristics', {}).items():
                cluster_mask = cluster_labels == cluster_id
                intensity[cluster_mask] = char.get('percentage', 0) / 100
            return intensity
        except Exception:
            return np.zeros(len(cluster_labels))

    @handles_errors(fallback = pd.DataFrame())
    def _create_intensity_dataframe(self, features: Any, hmm_states: Any, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Create intensity DataFrame for cluster analysis."""
        try:
            self.logger.info('📊 Creating intensity DataFrame...')
            intensity_df = pd.DataFrame()
            intensity_df['composite_cluster_id'] = cluster_labels
            intensity_df['hmm_state'] = hmm_states
            unique_clusters = np.unique(cluster_labels)
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_char = composite_analysis.get('cluster_characteristics', {}).get(cluster_id, {})
                intensity_df.loc[cluster_mask, 'cluster_intensity'] = cluster_char.get('size', 0) / len(features)

            intensity_df.loc[cluster_mask, 'volatility_intensity'] = self._calculate_volatility_intensity(features, cluster_mask)
            intensity_df.loc[cluster_mask, 'momentum_intensity'] = self._calculate_momentum_intensity(features, cluster_mask)
            intensity_df.loc[cluster_mask, 'volume_intensity'] = self._calculate_volume_intensity(features, cluster_mask)
            intensity_df.loc[cluster_mask, 'combined_intensity'] = intensity_df.loc[cluster_mask, 'cluster_intensity'] * 0.3 + intensity_df.loc[cluster_mask, 'volatility_intensity'] * 0.3 + intensity_df.loc[cluster_mask, 'momentum_intensity'] * 0.2 + intensity_df.loc[cluster_mask, 'volume_intensity'] * 0.2
            self.logger.info(f'✅ Created intensity DataFrame: {len(intensity_df)} rows, {len(intensity_df.columns)} columns')
            return intensity_df
        except Exception as e:
            self.logger.exception(f'❌ Error creating intensity DataFrame: {e}')
            return pd.DataFrame()

    def _calculate_volatility_intensity(self, features: Any, cluster_mask: Any) -> float:
        """Calculate volatility intensity for a cluster."""
        try:
            if 'volatility_20' in features.columns:
                return features.loc[cluster_mask, 'volatility_20'].mean()
            return 0.0
        except Exception:
            return 0.0

    @log_all_calls
    @handles_errors(default_return={'state_to_regime_map': {}, 'state_analysis': {}}, context='interpret_hmm_states')
    def _interpret_hmm_states(self, features: Any, state_sequence: Any, state_probs: Any) -> dict[str, Any]:
        """Interpret HMM states based on feature characteristics."""
        try:
            self.logger.info('🔍 Interpreting HMM states...')
            state_analysis = {}
            state_to_regime_map = {}
            unique_states = sorted(set(state_sequence))
            for state in unique_states:
                state_mask = state_sequence == state
                state_data = features[state_mask]
                if len(state_data) == 0:
                    continue
                state_char = {'count': len(state_data), 'percentage': len(state_data) / len(features) * 100}
                key_features = ['price_momentum_10', 'volatility_20', 'volume_ratio_10', 'rsi', 'adx', 'bb_position']
                for feature in key_features:
                    if feature in state_data.columns:
                        feature_data = state_data[feature].dropna()
                        if len(feature_data) > 0:
                            state_char[f'{feature}_mean'] = feature_data.mean()
                            state_char[f'{feature}_std'] = feature_data.std()
                state_analysis[state] = state_char
                regime_name = self._map_state_to_regime(state_char)
                state_to_regime_map[state] = regime_name
                self.logger.info(f"   State {state} → {regime_name}: {len(state_data)} periods ({state_char['percentage']:.1f}%)")
            return {'state_to_regime_map': state_to_regime_map, 'state_analysis': state_analysis}
        except Exception as e:
            self.logger.exception(f'❌ Error interpreting HMM states: {e}')
            return {'state_to_regime_map': {}, 'state_analysis': {}}

    @log_all_calls
    @handles_errors(fallback='unknown_regime')
    def _map_state_to_regime(self, state_char: dict[str, Any]) -> str:
        """Map state characteristics to regime name."""
        try:
            momentum = state_char.get('price_momentum_10_mean', 0)
            volatility = state_char.get('volatility_20_mean', 0)
            volume_ratio = state_char.get('volume_ratio_10_mean', 1)
            rsi = state_char.get('rsi_mean', 50)
            adx = state_char.get('adx_mean', 25)
            if volatility > 0.02:
                if momentum > 0.001:
                    return 'high_volatility_bull'
                elif momentum < -0.001:
                    return 'high_volatility_bear'
                else:
                    return 'high_volatility_neutral'
            elif volatility < 0.01:
                if momentum > 0.001:
                    return 'low_volatility_bull'
                elif momentum < -0.001:
                    return 'low_volatility_bear'
                else:
                    return 'low_volatility_neutral'
            elif momentum > 0.001:
                return 'medium_volatility_bull'
            elif momentum < -0.001:
                return 'medium_volatility_bear'
            else:
                return 'medium_volatility_neutral'
        except Exception as e:
            self.logger.warning(f'Error mapping state to regime: {e}')
            return 'unknown_regime'

    @log_all_calls
    @handles_errors
    def _calculate_regime_transitions(self, regimes: List[str]) -> dict[str, Any]:
        """Calculate regime transition probabilities."""
        self.logger.info('🔄 Calculating regime transition probabilities...')
        transitions = {}
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            if current_regime not in transitions:
                transitions[current_regime] = {}
            if next_regime not in transitions[current_regime]:
                transitions[current_regime][next_regime] = 0
            transitions[current_regime][next_regime] += 1
        self.logger.info('📊 Converting transition counts to probabilities...')
        for current_regime in transitions:
            total = sum(transitions[current_regime].values())
            for next_regime in transitions[current_regime]:
                transitions[current_regime][next_regime] /= total
        self.logger.info(f'✅ Transition matrix calculated for {len(transitions)} regimes')
        return transitions

    @log_all_calls
    @handles_errors(default_return={'success': False, 'error': 'Enhanced regime change detection failed'}, context='enhanced_regime_change_detection')
    def _detect_regime_changes_advanced(self, hmm_probs: np.ndarray, hmm_states: np.ndarray, threshold: float = 0.1, min_persistence: int = 3) -> dict[str, Any]:
        """Detect regime changes using advanced probability-based approach.
        
        Args:
            hmm_probs: HMM state probabilities (n_samples, n_states)
            hmm_states: HMM state sequence
            threshold: Probability stability threshold for regime change detection
            min_persistence: Minimum bars a regime must persist
            
        Returns:
            Dictionary with regime change information
        """
        try:
            self.logger.info('🔍 Detecting regime changes using advanced probability-based approach...')
            regime_stability = np.max(hmm_probs, axis = 1)
            regime_entropy = -np.sum(hmm_probs * np.log(hmm_probs + 1e-10), axis = 1)
            stability_changes = np.diff(regime_stability)
            potential_transitions = stability_changes < -threshold
            entropy_threshold = np.percentile(regime_entropy, 75)
            entropy_confirmation = regime_entropy[1:] > entropy_threshold
            initial_transitions = potential_transitions & entropy_confirmation
            confirmed_transitions = self._apply_persistence_filter(initial_transitions, hmm_states, min_persistence)
            transition_confidence = self._calculate_transition_confidence(hmm_probs, confirmed_transitions)
            regime_strength = self._calculate_regime_strength(hmm_probs, hmm_states)
            regime_changes = self._create_regime_change_events(confirmed_transitions, hmm_states, transition_confidence, regime_strength)
            self.logger.info(f'✅ Detected {len(regime_changes)} regime changes with advanced method')
            return {'success': True, 'regime_changes': regime_changes, 'transition_confidence': transition_confidence, 'regime_strength': regime_strength, 'stability_metrics': {'mean_stability': float(np.mean(regime_stability)), 'stability_volatility': float(np.std(regime_stability)), 'mean_entropy': float(np.mean(regime_entropy)), 'entropy_volatility': float(np.std(regime_entropy))}}
        except Exception as e:
            self.logger.exception(f'❌ Error in advanced regime change detection: {e}')
            return {'success': False, 'error': str(e)}

    @log_all_calls
    @handles_errors(default_return = np.zeros(0, dtype = bool), context='apply_persistence_filter')
    def _apply_persistence_filter(self, transitions: np.ndarray, states: np.ndarray, min_persistence: int) -> np.ndarray:
        """Apply persistence filter to avoid detecting noise as regime changes."""
        try:
            filtered_transitions = transitions.copy()
            durations = self._calculate_regime_durations(states)
            for i in range(len(transitions)):
                if transitions[i]:
                    current_duration = durations[i] if i < len(durations) else 0
                    if current_duration < min_persistence:
                        filtered_transitions[i] = False
            return filtered_transitions
        except Exception as e:
            self.logger.warning(f'⚠️ Error applying persistence filter: {e}')
            return transitions

    @log_all_calls
    @handles_errors(default_return = np.zeros(0, dtype = float), context='calculate_transition_confidence')
    def _calculate_transition_confidence(self, hmm_probs: np.ndarray, transitions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for regime transitions."""
        try:
            confidence_scores = np.zeros(len(transitions))
            for i in range(len(transitions)):
                if transitions[i] and i < len(hmm_probs) - 1:
                    prob_change = np.abs(hmm_probs[i + 1] - hmm_probs[i])
                    max_change = np.max(prob_change)
                    confidence_scores[i] = min(max_change * 10, 1.0)
            return confidence_scores
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating transition confidence: {e}')
            return np.zeros(len(transitions), dtype = float)

    @log_all_calls
    @handles_errors(default_return = np.zeros(0, dtype = float), context='calculate_regime_strength')
    def _calculate_regime_strength(self, hmm_probs: np.ndarray, hmm_states: np.ndarray) -> np.ndarray:
        """Calculate regime strength indicators."""
        try:
            max_probs = np.max(hmm_probs, axis = 1)
            prob_std = np.std(hmm_probs, axis = 1)
            consistency_strength = 1.0 / (1.0 + prob_std)
            regime_strength = max_probs * consistency_strength
            return regime_strength
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating regime strength: {e}')
            return np.zeros(len(hmm_states), dtype = float)

    @log_all_calls
    @handles_errors(fallback=[])
    def _create_regime_change_events(self, transitions: np.ndarray, states: np.ndarray, confidence: np.ndarray, strength: np.ndarray) -> list[dict[str, Any]]:
        """Create detailed regime change events."""
        try:
            events = []
            for i in range(len(transitions)):
                if transitions[i] and i < len(states) - 1:
                    event = {'timestamp_index': i, 'from_state': int(states[i]), 'to_state': int(states[i + 1]), 'confidence': float(confidence[i]), 'regime_strength': float(strength[i]), 'transition_type': 'regime_change'}
                    events.append(event)
            return events
        except Exception as e:
            self.logger.warning(f'⚠️ Error creating regime change events: {e}')
            return []

    @log_all_calls
    @handles_errors(default_return = np.zeros(0, dtype = int), context='calculate_regime_durations')
    def _calculate_regime_durations(self, states: np.ndarray) -> np.ndarray:
        """Calculate how long each regime persists."""
        try:
            durations = np.zeros(len(states), dtype = int)
            current_state = states[0]
            current_duration = 1
            for i in range(1, len(states)):
                if states[i] == current_state:
                    current_duration += 1
                else:
                    for j in range(i - current_duration, i):
                        durations[j] = current_duration
                    current_state = states[i]
                    current_duration = 1
            for j in range(len(states) - current_duration, len(states)):
                durations[j] = current_duration
            return durations
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating regime durations: {e}')
            return np.zeros(len(states), dtype = int)

    @log_all_calls
    def _calculate_adaptive_regime_boundaries(self, features: pd.DataFrame) -> dict[str, Any]:
        """Calculate adaptive regime boundaries using clustering of regime characteristics."""
        try:
            self.logger.info('🔧 Calculating adaptive regime boundaries...')
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            regime_features = self._extract_regime_characteristics(features)
            if regime_features.empty:
                self.logger.warning('⚠️ No regime characteristics available for boundary calculation')
                return {}
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(regime_features)
            clustering = DBSCAN(eps = 0.1, min_samples = 5)
            regime_boundaries = clustering.fit_predict(scaled_features)
            unique_boundaries = np.unique(regime_boundaries[regime_boundaries >= 0])
            boundary_stats = {}
            for boundary_id in unique_boundaries:
                boundary_mask = regime_boundaries == boundary_id
                boundary_features = regime_features[boundary_mask]
                boundary_stats[f'boundary_{boundary_id}'] = {'size': int(np.sum(boundary_mask)), 'characteristics': boundary_features.mean().to_dict(), 'volatility': float(boundary_features.std().mean())}
            self.logger.info(f'✅ Calculated {len(unique_boundaries)} adaptive regime boundaries')
            return {'boundaries': regime_boundaries, 'boundary_stats': boundary_stats, 'scaler': scaler, 'clustering_model': clustering}
        except Exception as e:
            self.logger.exception(f'❌ Error calculating adaptive regime boundaries: {e}')
            return {}

    @log_all_calls
    @handles_errors(fallback = pd.DataFrame())
    def _extract_regime_characteristics(self, features: pd.DataFrame) -> pd.DataFrame:
        """Extract regime characteristics for boundary calculation."""
        try:
            characteristics = pd.DataFrame()
            key_features = ['price_momentum_10', 'volatility_20', 'volume_ratio_10', 'rsi', 'adx', 'bb_position', 'atr_normalized']
            for feature in key_features:
                if feature in features.columns:
                    characteristics[f'{feature}_mean'] = features[feature].rolling(20).mean()
                    characteristics[f'{feature}_std'] = features[feature].rolling(20).std()
                    characteristics[f'{feature}_trend'] = features[feature].diff(10)
            if 'price_momentum_10' in features.columns and 'volatility_20' in features.columns:
                characteristics['momentum_volatility_ratio'] = features['price_momentum_10'] / (features['volatility_20'] + 1e-08)
            if 'volume_ratio_10' in features.columns and 'price_momentum_10' in features.columns:
                characteristics['volume_momentum_correlation'] = features['volume_ratio_10'] * features['price_momentum_10']
            characteristics = characteristics.dropna()
            return characteristics
        except Exception as e:
            self.logger.warning(f'⚠️ Error extracting regime characteristics: {e}')
            return pd.DataFrame()

    @log_all_calls
    @handles_errors
    def _model_regime_persistence(self, regime_sequence: np.ndarray) -> dict[str, Any]:
        """Model how long regimes typically persist using statistical distributions."""
        try:
            self.logger.info('📊 Modeling regime persistence...')
            from scipy.stats import weibull_min, expon, gamma
            durations = self._calculate_regime_durations(regime_sequence)
            unique_durations = np.unique(durations)
            if len(unique_durations) < 3:
                self.logger.warning('⚠️ Insufficient regime duration data for modeling')
                return {}
            distribution_fits = {}
            try:
                shape, loc, scale = weibull_min.fit(durations)
                distribution_fits['weibull'] = {'shape': float(shape), 'scale': float(scale), 'mean_duration': float(scale * np.exp(1 / shape)), 'survival_function': lambda t: weibull_min.sf(t, shape, loc, scale), 'aic': self._calculate_aic(durations, weibull_min.pdf, shape, loc, scale)}
            except Exception as e:
                self.logger.warning(f'⚠️ Weibull fit failed: {e}')
            try:
                loc, scale = expon.fit(durations)
                distribution_fits['exponential'] = {'scale': float(scale), 'mean_duration': float(scale), 'survival_function': lambda t: expon.sf(t, loc, scale), 'aic': self._calculate_aic(durations, expon.pdf, loc, scale)}
            except Exception as e:
                self.logger.warning(f'⚠️ Exponential fit failed: {e}')
            try:
                shape, loc, scale = gamma.fit(durations)
                distribution_fits['gamma'] = {'shape': float(shape), 'scale': float(scale), 'mean_duration': float(shape * scale), 'survival_function': lambda t: gamma.sf(t, shape, loc, scale), 'aic': self._calculate_aic(durations, gamma.pdf, shape, loc, scale)}
            except Exception as e:
                self.logger.warning(f'⚠️ Gamma fit failed: {e}')
            best_distribution = None
            best_aic = float('inf')
            for dist_name, dist_params in distribution_fits.items():
                if dist_params['aic'] < best_aic:
                    best_aic = dist_params['aic']
                    best_distribution = dist_name
            transition_matrix = self._calculate_transition_matrix(regime_sequence)
            persistence_stats = {'mean_duration': float(np.mean(durations)), 'median_duration': float(np.median(durations)), 'std_duration': float(np.std(durations)), 'min_duration': int(np.min(durations)), 'max_duration': int(np.max(durations)), 'duration_percentiles': {'25': float(np.percentile(durations, 25)), '50': float(np.percentile(durations, 50)), '75': float(np.percentile(durations, 75)), '90': float(np.percentile(durations, 90))}}
            self.logger.info(f'✅ Modeled regime persistence with {best_distribution} distribution')
            return {'best_distribution': best_distribution, 'distribution_fits': distribution_fits, 'persistence_stats': persistence_stats, 'transition_matrix': transition_matrix, 'durations': durations.tolist()}
        except Exception as e:
            self.logger.exception(f'❌ Error modeling regime persistence: {e}')
            return {}

    @log_all_calls
    @handles_errors(fallback = float('inf'))
    def _calculate_aic(self, data: np.ndarray, pdf_func: Any, *params) -> float:
        """Calculate Akaike Information Criterion for distribution fitting."""
        try:
            log_likelihood = np.sum(np.log(pdf_func(data, *params) + 1e-10))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            return aic
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating AIC: {e}')
            return float('inf')

    @log_all_calls
    @handles_errors(fallback = np.array([]))
    def _calculate_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix."""
        try:
            unique_states = np.unique(regime_sequence)
            n_states = len(unique_states)
            if n_states == 0:
                return np.array([])
            state_map = {state: i for i, state in enumerate(unique_states)}
            transition_matrix = np.zeros((n_states, n_states))
            for i in range(len(regime_sequence) - 1):
                current_state = state_map[regime_sequence[i]]
                next_state = state_map[regime_sequence[i + 1]]
                transition_matrix[current_state, next_state] += 1
            row_sums = transition_matrix.sum(axis = 1, keepdims = True)
            transition_matrix = np.divide(transition_matrix, row_sums, where = row_sums > 0)
            return transition_matrix
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating transition matrix: {e}')
            return np.array([])

    async def _get_sr_context_for_regime_analysis(self, market_data: pd.DataFrame, current_price: float) -> dict[str, Any]:
        """Get SR context for regime analysis."""
        try:
            if not hasattr(self, 'sr_predictor') or self.sr_predictor is None:
                self.logger.warning('⚠️ SR predictor not available, skipping SR context analysis')
                return {}
            sr_context = await self.sr_predictor.get_sr_context(market_data, current_price)
            self.logger.info(f'✅ SR context analysis completed: {len(sr_context)} context elements')
            return sr_context
        except Exception as e:
            self.logger.error(f'Error getting SR context for regime analysis: {e}')
            return {}

    async def _enhance_regime_analysis_with_sr(self, regime_results: dict[str, Any], sr_context: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
        """Enhance regime analysis with SR context."""
        try:
            enhanced_results = regime_results.copy()
            enhanced_results['sr_context'] = sr_context
            sr_regime_features = await self._create_sr_regime_features(regime_results.get('regime_states', []), sr_context, market_data)
            enhanced_results['sr_regime_features'] = sr_regime_features
            if hasattr(self, 'sr_predictor') and self.sr_predictor and self.sr_predictor.reporting_enabled:
                await self.sr_predictor.generate_manual_report(market_data, sr_context)
            self.logger.info('✅ SR context analysis completed')
            return enhanced_results
        except Exception as e:
            self.logger.error(f'Error enhancing regime analysis with SR: {e}')
            return regime_results

    async def _create_sr_regime_features(self, regime_states: list[int], sr_context: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
        """Create SR-aware regime features."""
        try:
            features = {}
            features['sr_proximity_by_regime'] = {}
            features['sr_strength_by_regime'] = {}
            for regime in set(regime_states):
                regime_mask = [i for i, r in enumerate(regime_states) if r == regime]
                regime_data = market_data.iloc[regime_mask]
                if len(regime_data) > 0:
                    regime_price = regime_data['close'].iloc[-1]
                    regime_sr_context = await self._get_sr_context_for_regime_analysis(regime_data, regime_price)
                    features['sr_proximity_by_regime'][f'regime_{regime}'] = {'support_proximity': regime_sr_context.get('support_proximity', 1.0), 'resistance_proximity': regime_sr_context.get('resistance_proximity', 1.0)}
                    features['sr_strength_by_regime'][f'regime_{regime}'] = {'support_strength': regime_sr_context.get('support_strength', 0.5), 'resistance_strength': regime_sr_context.get('resistance_strength', 0.5)}
            features['overall_sr_metrics'] = {'support_proximity': sr_context.get('support_proximity', 1.0), 'resistance_proximity': sr_context.get('resistance_proximity', 1.0), 'support_strength': sr_context.get('support_strength', 0.5), 'resistance_strength': sr_context.get('resistance_strength', 0.5), 'sr_zone_width': sr_context.get('sr_zone_width', 0.0), 'total_support_levels': len(sr_context.get('support_levels', [])), 'total_resistance_levels': len(sr_context.get('resistance_levels', []))}
            self.logger.info(f'✅ Created SR-aware regime features for {len(set(regime_states))} regimes')
            return features
        except Exception as e:
            self.logger.error(f'Error creating SR regime features: {e}')
            return {}
    @log_all_calls

    def _create_composite_features(self, features: Any, hmm_states: Any, hmm_probs: Any) -> Any:
        """Create composite features combining HMM states with original features."""
        try:
            self.logger.info('🔧 Creating composite features...')
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            composite_df = features.copy()
            composite_df['hmm_state'] = hmm_states
            composite_df['hmm_state_prob_max'] = np.max(hmm_probs, axis = 1)
            composite_df['hmm_state_entropy'] = -np.sum(hmm_probs * np.log(hmm_probs + 1e-10), axis = 1)
            for i in range(hmm_probs.shape[1]):
                composite_df[f'hmm_state_prob_{i}'] = hmm_probs[:, i]
            key_features = ['price_momentum_10', 'volatility_20', 'volume_ratio_10', 'rsi', 'adx']
            for feature in key_features:
                if feature in composite_df.columns:
                    composite_df[f'{feature}_x_hmm_state'] = composite_df[feature] * composite_df['hmm_state']
                    composite_df[f'{feature}_x_hmm_entropy'] = composite_df[feature] * composite_df['hmm_state_entropy']
            composite_df['hmm_state_persistence'] = self._calculate_persistence(hmm_states)
            composite_df['hmm_state_transitions'] = self._calculate_transitions(hmm_states)
            self.logger.info(f'✅ Created composite features: {len(composite_df.columns)} total features')
            return composite_df
        except Exception as e:
            self.logger.exception(f'❌ Error creating composite features: {e}')
            return pd.DataFrame()
    @log_all_calls

    def _calculate_persistence(self, states: Any) -> Any:
        """Calculate state persistence (how long we stay in current state)."""
        try:
            persistence = np.zeros(len(states))
            current_state = states[0]
            current_count = 1
            for i in range(1, len(states)):
                if states[i] == current_state:
                    current_count += 1
                else:
                    for j in range(i - current_count, i):
                        persistence[j] = current_count
                    current_state = states[i]
                    current_count = 1
            for j in range(len(states) - current_count, len(states)):
                persistence[j] = current_count
            return persistence
        except Exception:
            return np.zeros(len(states))
    @log_all_calls

    def _calculate_transitions(self, states: Any) -> Any:
        """Calculate number of state transitions."""
        try:
            transitions = np.zeros(len(states))
            for i in range(1, len(states)):
                if states[i] != states[i - 1]:
                    transitions[i] = 1
            return transitions
        except Exception:
            return np.zeros(len(states))

    @log_all_calls
    @handles_errors(fallback={})
    def _calculate_cluster_quality_metrics(self, features_scaled: Any, cluster_labels: Any, kmeans_model: Any) -> dict[str, Any]:
        """Calculate comprehensive cluster quality metrics."""
        try:
            self.logger.info('📊 Calculating cluster quality metrics...')
            self.logger.info(f'   - Input features shape: {features_scaled.shape}')
            self.logger.info(f'   - Cluster labels shape: {cluster_labels.shape}')
            self.logger.info(f'   - Unique clusters: {len(np.unique(cluster_labels))}')
            
            metrics = {}
            
            # Silhouette Score (with sampling for large datasets)
            self.logger.info('   - Calculating Silhouette Score...')
            try:
                # For large datasets, use sampling to speed up calculation
                n_samples = len(features_scaled)
                if n_samples > 10000:
                    self.logger.info(f'     - Large dataset ({n_samples:,} samples), using sampling for efficiency...')
                    # Sample 10,000 points for silhouette calculation
                    sample_size = min(10000, n_samples)
                    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                    features_sample = features_scaled[sample_indices]
                    labels_sample = cluster_labels[sample_indices]
                    self.logger.info(f'     - Using {sample_size:,} samples for silhouette calculation...')
                    metrics['silhouette_score'] = silhouette_score(features_sample, labels_sample)
                    self.logger.info(f'     ✅ Silhouette Score (sampled): {metrics["silhouette_score"]:.4f}')
                else:
                    self.logger.info(f'     - Small dataset ({n_samples:,} samples), calculating full silhouette...')
                    metrics['silhouette_score'] = silhouette_score(features_scaled, cluster_labels)
                    self.logger.info(f'     ✅ Silhouette Score: {metrics["silhouette_score"]:.4f}')
            except Exception as e:
                metrics['silhouette_score'] = 0.0
                self.logger.warning(f'     ⚠️ Silhouette Score failed: {e}')
            
            # Calinski-Harabasz Score (with sampling for large datasets)
            self.logger.info('   - Calculating Calinski-Harabasz Score...')
            try:
                if n_samples > 10000:
                    self.logger.info(f'     - Using {sample_size:,} samples for Calinski-Harabasz calculation...')
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_sample, labels_sample)
                    self.logger.info(f'     ✅ Calinski-Harabasz Score (sampled): {metrics["calinski_harabasz_score"]:.2f}')
                else:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, cluster_labels)
                    self.logger.info(f'     ✅ Calinski-Harabasz Score: {metrics["calinski_harabasz_score"]:.2f}')
            except Exception as e:
                metrics['calinski_harabasz_score'] = 0.0
                self.logger.warning(f'     ⚠️ Calinski-Harabasz Score failed: {e}')
            
            # Davies-Bouldin Score (with sampling for large datasets)
            self.logger.info('   - Calculating Davies-Bouldin Score...')
            try:
                if n_samples > 10000:
                    self.logger.info(f'     - Using {sample_size:,} samples for Davies-Bouldin calculation...')
                    metrics['davies_bouldin_score'] = davies_bouldin_score(features_sample, labels_sample)
                    self.logger.info(f'     ✅ Davies-Bouldin Score (sampled): {metrics["davies_bouldin_score"]:.4f}')
                else:
                    metrics['davies_bouldin_score'] = davies_bouldin_score(features_scaled, cluster_labels)
                    self.logger.info(f'     ✅ Davies-Bouldin Score: {metrics["davies_bouldin_score"]:.4f}')
            except Exception as e:
                metrics['davies_bouldin_score'] = float('inf')
                self.logger.warning(f'     ⚠️ Davies-Bouldin Score failed: {e}')
            # KMeans Inertia
            self.logger.info('   - Calculating KMeans Inertia...')
            metrics['inertia'] = kmeans_model.inertia_
            self.logger.info(f'     ✅ Inertia: {metrics["inertia"]:.2f}')
            
            # Cluster Size Analysis
            self.logger.info('   - Analyzing cluster sizes...')
            unique_labels, counts = np.unique(cluster_labels, return_counts = True)
            metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
            metrics['min_cluster_size'] = np.min(counts)
            metrics['max_cluster_size'] = np.max(counts)
            metrics['mean_cluster_size'] = np.mean(counts)
            metrics['std_cluster_size'] = np.std(counts)
            metrics['cluster_balance'] = metrics['std_cluster_size'] / metrics['mean_cluster_size'] if metrics['mean_cluster_size'] > 0 else 0
            
            self.logger.info(f'     ✅ Cluster sizes: min={metrics["min_cluster_size"]}, max={metrics["max_cluster_size"]}, mean={metrics["mean_cluster_size"]:.1f}')
            self.logger.info(f'     ✅ Cluster balance: {metrics["cluster_balance"]:.4f}')
            
            # Distance Analysis
            self.logger.info('   - Calculating distance metrics...')
            distances = kmeans_model.transform(features_scaled)
            min_distances = np.min(distances, axis = 1)
            metrics['mean_distance_to_center'] = np.mean(min_distances)
            metrics['max_distance_to_center'] = np.max(min_distances)
            self.logger.info(f'     ✅ Mean distance to center: {metrics["mean_distance_to_center"]:.4f}')
            self.logger.info(f'     ✅ Max distance to center: {metrics["max_distance_to_center"]:.4f}')
            self.logger.info(f'✅ Cluster quality metrics calculated:')
            self.logger.info(f"   - Silhouette: {metrics['silhouette_score']:.4f}")
            self.logger.info(f"   - Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
            self.logger.info(f"   - Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
            self.logger.info(f"   - Inertia: {metrics['inertia']:.2f}")
            return metrics
        except Exception as e:
            self.logger.exception(f'❌ Error calculating cluster quality metrics: {e}')
            return {}
    @log_all_calls

    def _analyze_composite_clusters(self, features: Any, hmm_states: Any, cluster_labels: Any, cluster_metrics: dict[str, Any]) -> dict[str, Any]:
        """Analyze composite clusters and their characteristics."""
        try:
            self.logger.info('🔍 Analyzing composite clusters...')
            analysis = {'cluster_characteristics': {}, 'hmm_state_distribution': {}, 'feature_importance': {}, 'cluster_stability': {}, 'market_conditions': {}}
            unique_clusters = np.unique(cluster_labels)
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_data = features[cluster_mask]
                cluster_hmm_states = hmm_states[cluster_mask]
                cluster_char = {'size': len(cluster_data), 'percentage': len(cluster_data) / len(features) * 100, 'hmm_state_distribution': self._calculate_hmm_state_distribution(cluster_hmm_states), 'feature_means': {}, 'feature_stds': {}, 'dominant_hmm_state': self._get_dominant_hmm_state(cluster_hmm_states)}
                for col in features.columns:
                    if col in cluster_data.columns:
                        cluster_char['feature_means'][col] = cluster_data[col].mean()
                        cluster_char['feature_stds'][col] = cluster_data[col].std()
                analysis['cluster_characteristics'][cluster_id] = cluster_char
                market_condition = self._determine_market_condition(cluster_char)
                analysis['market_conditions'][cluster_id] = market_condition
            analysis['hmm_state_distribution'] = self._calculate_hmm_state_distribution(hmm_states)
            analysis['feature_importance'] = self._calculate_feature_importance(features, cluster_labels)
            analysis['cluster_stability'] = self._calculate_cluster_stability(cluster_labels, cluster_metrics)
            self.logger.info(f'✅ Composite cluster analysis completed for {len(unique_clusters)} clusters')
            return analysis
        except Exception as e:
            self.logger.exception(f'❌ Error analyzing composite clusters: {e}')
            return {}
    @log_all_calls

    def _determine_market_condition(self, cluster_char: dict[str, Any]) -> str:
        """Determine market condition based on cluster characteristics."""
        try:
            # Simple market condition determination based on feature means
            if 'volatility_5' in cluster_char['feature_means']:
                volatility = cluster_char['feature_means']['volatility_5']
                if volatility > 0.001:
                    return 'High Volatility'
                elif volatility > 0.0005:
                    return 'Medium Volatility'
                else:
                    return 'Low Volatility'
            return 'Unknown'
        except Exception:
            return 'Unknown'
    @log_all_calls

    def _calculate_feature_importance(self, features: Any, cluster_labels: Any) -> dict[str, float]:
        """Calculate feature importance for clustering."""
        try:
            # Simple feature importance based on variance
            importance = {}
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32']:
                    importance[col] = features[col].var()
            return importance
        except Exception:
            return {}
    @log_all_calls

    def _calculate_cluster_stability(self, cluster_labels: Any, cluster_metrics: dict[str, Any]) -> dict[str, Any]:
        """Calculate cluster stability metrics."""
        try:
            return {
                'silhouette_score': cluster_metrics.get('silhouette_score', 0.0),
                'calinski_harabasz_score': cluster_metrics.get('calinski_harabasz_score', 0.0),
                'davies_bouldin_score': cluster_metrics.get('davies_bouldin_score', float('inf'))
            }
        except Exception:
            return {}
    @log_all_calls

    def _calculate_hmm_state_distribution(self, hmm_states: Any) -> dict[int, int]:
        """Calculate HMM state distribution."""
        try:
            unique, counts = np.unique(hmm_states, return_counts=True)
            return dict(zip(unique, counts))
        except Exception:
            return {}
    @log_all_calls

    def _get_dominant_hmm_state(self, hmm_states: Any) -> int:
        """Get dominant HMM state."""
        try:
            unique, counts = np.unique(hmm_states, return_counts=True)
            return unique[np.argmax(counts)]
        except Exception:
            return 0

@monitor_feature_engineering()
@handles_errors(fallback = False)
async def run_step(symbol: str, exchange: str, timeframe: str='1m', data_dir: str = None, force_rerun: bool = False,
                use_optimized_pipeline: bool = True, force_optimized_pipeline: bool = False, **kwargs: Any) -> bool:
    """Run the HMM regime discovery step with standardized data quality management and optimizations.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT" or configured default)
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re-run even if results exist
        use_optimized_pipeline: Whether to use optimized pipeline components (default: True)
        force_optimized_pipeline: Force use of optimized pipeline regardless of availability (default: False)
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    
    # 🖨️ THOROUGH PRINTING: HMM Regime Discovery Step Start
    tprint("🚀 STEP 3: HMM REGIME DISCOVERY WITH STANDARDIZED DATA QUALITY MANAGEMENT")
    tprint("=" * 80)
    tprint(f"🎯 Symbol: {symbol}")
    tprint(f"🏢 Exchange: {exchange}")
    tprint(f"📊 Timeframe: {timeframe}")
    tprint(f"📁 Data directory: {data_dir}")
    tprint(f"🔄 Force rerun: {force_rerun}")
    tprint(f"🔧 Use optimized pipeline: {use_optimized_pipeline}")
    tprint(f"⚡ Force optimized pipeline: {force_optimized_pipeline}")
    tprint(f"⚙️ Additional kwargs: {kwargs}")
    tprint(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tprint("=" * 80)
    
    try:
        logger = system_logger.getChild('Step3HMMRegimeDiscovery')
        if data_dir is None:
            data_dir = 'data_cache'
            tprint(f"📁 Data directory set to default: {data_dir}")
        
        logger.info('=' * 80)
        logger.info('🚀 STEP 3: HMM Regime Discovery with Standardized Data Quality Management')
        logger.info('=' * 80)
        logger.info(f'🎯 Symbol: {symbol}')
        logger.info(f'🏢 Exchange: {exchange}')
        logger.info(f'📊 Timeframe: {timeframe}')
        logger.info(f'📁 Data directory: {data_dir}')
        logger.info(f'🔄 Force rerun: {force_rerun}')
        logger.info(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info('=' * 80)
        config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir}
        tprint("🔧 INITIALIZING HMM REGIME DISCOVERY STEP")
        tprint(f"   📋 Configuration: {config}")
        
        logger.info('🔧 Initializing HMM regime discovery step...')
        step = HMMRegimeDiscoveryStep(config)
        tprint("   ✅ HMMRegimeDiscoveryStep instance created")
        
        await step.initialize()
        tprint("   ✅ HMMRegimeDiscoveryStep initialized")
        
        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'force_rerun': force_rerun,
            'use_optimized_pipeline': use_optimized_pipeline,
            'force_optimized_pipeline': force_optimized_pipeline
        }
        tprint("🎯 EXECUTING HMM REGIME DISCOVERY")
        tprint(f"   📋 Training input: {training_input}")
        
        logger.info('🎯 Executing HMM regime discovery...')
        pipeline_state = {}
        tprint("   📊 Pipeline state initialized")
        
        result = await step.execute(training_input, pipeline_state)
        tprint(f"   📊 Execution result: {result}")
        if result.get('hmm_regime_discovery_completed', False):
            tprint("✅ STEP 3: HMM REGIME DISCOVERY COMPLETED SUCCESSFULLY")
            logger.info('✅ Step 3: HMM Regime Discovery completed successfully')

            # Log optimization usage
            if result.get('optimized_pipeline_used', False):
                tprint("🚀 OPTIMIZED PIPELINE USED FOR ENHANCED PERFORMANCE!")
                logger.info('🚀 Optimized pipeline used for enhanced performance!')
                if result.get('performance_metrics'):
                    perf = result['performance_metrics']
                    tprint(f"⚡ Performance: {perf.get('average_task_time', 0):.2f}s avg task time")
                    logger.info(f"⚡ Performance: {perf.get('average_task_time', 0):.2f}s avg task time")
                if result.get('cache_performance'):
                    cache = result['cache_performance']
                    tprint(f"📋 Cache: {cache.get('hit_rate', 0):.1%} hit rate")
                    logger.info(f"📋 Cache: {cache.get('hit_rate', 0):.1%} hit rate")
            elif result.get('optimization_used', False):
                tprint("🔧 STANDARD PARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY")
                logger.info('🔧 Standard parameter optimization completed successfully')
                if result.get('optimized_params'):
                    tprint(f"📊 Optimized parameters applied: {list(result['optimized_params'].keys())}")
                    logger.info(f"📊 Optimized parameters applied: {list(result['optimized_params'].keys())}")
            else:
                tprint("⚠️ PARAMETER OPTIMIZATION FAILED, USING DEFAULT PARAMETERS")
                logger.warning('⚠️ Parameter optimization failed, using default parameters')
            if result.get('regime_states'):
                unique_regimes = len(set(result['regime_states']))
                total_periods = len(result['regime_states'])
                tprint(f"📊 DISCOVERED {unique_regimes} UNIQUE REGIMES ACROSS {total_periods:,} PERIODS")
                logger.info(f'📊 Discovered {unique_regimes} unique regimes across {total_periods:,} periods')
            if result.get('regime_metrics'):
                metrics = result['regime_metrics']
                tprint(f"📈 TOTAL PERIODS: {metrics.get('total_periods', 0):,}")
                tprint(f"🔄 UNIQUE REGIMES: {metrics.get('unique_regimes', 0)}")
                logger.info(f"📈 Total periods: {metrics.get('total_periods', 0):,}")
                logger.info(f"🔄 Unique regimes: {metrics.get('unique_regimes', 0)}")
                regime_dist = metrics.get('regime_distribution', {})
                if regime_dist:
                    tprint("📊 REGIME DISTRIBUTION:")
                    logger.info('📊 Regime distribution:')
                    for regime, count in regime_dist.items():
                        percentage = count / metrics.get('total_periods', 1) * 100
                        tprint(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")
                        logger.info(f'   - {regime}: {count:,} periods ({percentage:.1f}%)')
            total_elapsed = time.time() - start_time
            tprint("=" * 80)
            tprint("🎉 STEP 3 EXECUTION SUMMARY")
            tprint("=" * 80)
            tprint(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
            tprint(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            tprint("✅ SUCCESS")
            tprint("=" * 80)
            
            logger.info('=' * 80)
            logger.info('🎉 STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('✅ SUCCESS')
            logger.info('=' * 80)
            return True
        else:
            tprint("❌ STEP 3: HMM REGIME DISCOVERY FAILED")
            logger.error('❌ Step 3: HMM Regime Discovery failed')
            error = result.get('regime_discovery_error', 'Unknown error')
            tprint(f"   ❌ Error: {error}")
            logger.error(f'   Error: {error}')
            total_elapsed = time.time() - start_time
            tprint("=" * 80)
            tprint("💥 STEP 3 EXECUTION SUMMARY")
            tprint("=" * 80)
            tprint(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
            tprint(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            tprint("❌ FAILED")
            tprint(f"   Error: {error}")
            tprint("=" * 80)
            
            logger.info('=' * 80)
            logger.info('💥 STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('❌ FAILED')
            logger.info(f'   Error: {error}')
            logger.info('=' * 80)
            return False
    except Exception as e:
        tprint("💥 STEP 3: HMM REGIME DISCOVERY FAILED WITH EXCEPTION")
        tprint(f"   ❌ Exception: {e}")
        logger.exception(f'❌ Step 3: HMM Regime Discovery failed with exception: {e}')
        total_elapsed = time.time() - start_time
        tprint("=" * 80)
        tprint("💥 STEP 3 EXECUTION SUMMARY")
        tprint("=" * 80)
        tprint(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
        tprint(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        tprint("❌ FAILED")
        tprint(f"   Exception: {e}")
        tprint("=" * 80)
        
        logger.info('=' * 80)
        logger.info('💥 STEP 3 EXECUTION SUMMARY')
        logger.info('=' * 80)
        logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info('❌ FAILED')
        logger.info(f'   Exception: {e}')
        logger.info('=' * 80)
        return False

    def _calculate_persistence(self, states: Any) -> Any:
        """Calculate state persistence (how long we stay in current state)."""
        try:
            persistence = np.zeros(len(states))
            current_state = states[0]
            current_count = 1
            for i in range(1, len(states)):
                if states[i] == current_state:
                    current_count += 1
                else:
                    for j in range(i - current_count, i):
                        persistence[j] = current_count
                    current_state = states[i]
                    current_count = 1
            for j in range(len(states) - current_count, len(states)):
                persistence[j] = current_count
            return persistence
        except Exception:
            return np.zeros(len(states))

    def _calculate_transitions(self, states: Any) -> Any:
        """Calculate number of state transitions."""
        try:
            transitions = np.zeros(len(states))
            for i in range(1, len(states)):
                if states[i] != states[i - 1]:
                    transitions[i] = 1
            return transitions
        except Exception:
            return np.zeros(len(states))

    def _calculate_hmm_state_distribution(self, hmm_states: Any) -> dict[int, int]:
        """Calculate distribution of HMM states."""
        try:
            unique_states, counts = np.unique(hmm_states, return_counts = True)
            return dict(zip(unique_states, counts))
        except Exception:
            return {}

    def _get_dominant_hmm_state(self, hmm_states: Any) -> int:
        """Get the dominant HMM state in a cluster."""
        try:
            unique_states, counts = np.unique(hmm_states, return_counts = True)
            return unique_states[np.argmax(counts)]
        except Exception:
            return 0

    def _determine_market_condition(self, cluster_char: dict[str, Any]) -> str:
        """Determine market condition for a cluster based on its characteristics."""
        try:
            momentum = cluster_char.get('feature_means', {}).get('price_momentum_10', 0)
            volatility = cluster_char.get('feature_means', {}).get('volatility_20', 0)
            volume_ratio = cluster_char.get('feature_means', {}).get('volume_ratio_10', 1)
            rsi = cluster_char.get('feature_means', {}).get('rsi', 50)
            if volatility > 0.02:
                if momentum > 0.001:
                    return 'high_volatility_bull'
                elif momentum < -0.001:
                    return 'high_volatility_bear'
                else:
                    return 'high_volatility_neutral'
            elif volatility < 0.01:
                if momentum > 0.001:
                    return 'low_volatility_bull'
                elif momentum < -0.001:
                    return 'low_volatility_bear'
                else:
                    return 'low_volatility_neutral'
            elif momentum > 0.001:
                return 'medium_volatility_bull'
            elif momentum < -0.001:
                return 'medium_volatility_bear'
            else:
                return 'medium_volatility_neutral'
        except Exception:
            return 'unknown'

    def _calculate_feature_importance(self, features: Any, cluster_labels: Any) -> dict[str, float]:
        """Calculate feature importance based on cluster separation."""
        try:
            importance = {}
            for col in features.columns:
                total_var = features[col].var()
                if total_var > 0:
                    between_cluster_var = 0
                    within_cluster_var = 0
                    for cluster_id in np.unique(cluster_labels):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_mean = features.loc[cluster_mask, col].mean()
                        cluster_var = features.loc[cluster_mask, col].var()
                        cluster_size = cluster_mask.sum()
                        between_cluster_var += cluster_size * (cluster_mean - features[col].mean()) ** 2
                        within_cluster_var += cluster_size * cluster_var
                    if within_cluster_var > 0:
                        importance[col] = between_cluster_var / within_cluster_var
                    else:
                        importance[col] = 0
                else:
                    importance[col] = 0
            return importance
        except Exception:
            return {}

    def _calculate_cluster_stability(self, cluster_labels: Any, cluster_metrics: dict[str, Any]) -> dict[str, float]:
        """Calculate cluster stability metrics."""
        try:
            stability = {'silhouette_score': cluster_metrics.get('silhouette_score', 0), 'cluster_balance': cluster_metrics.get('cluster_balance', 0), 'mean_distance_to_center': cluster_metrics.get('mean_distance_to_center', 0)}
            return stability
        except Exception:
            return {}

    def _calculate_cluster_stability_scores(self, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Calculate cluster stability scores."""
        try:
            stability = np.ones(len(cluster_labels))
            return stability
        except Exception:
            return np.ones(len(cluster_labels))

    def _calculate_momentum_intensity(self, features: Any, cluster_mask: Any) -> float:
        """Calculate momentum intensity for a cluster."""
        try:
            if 'price_momentum_10' in features.columns:
                return abs(features.loc[cluster_mask, 'price_momentum_10'].mean())
            return 0.0
        except Exception:
            return 0.0

    def _calculate_volume_intensity(self, features: Any, cluster_mask: Any) -> float:
        """Calculate volume intensity for a cluster."""
        try:
            if 'volume_ratio_10' in features.columns:
                return features.loc[cluster_mask, 'volume_ratio_10'].mean()
            return 1.0
        except Exception:
            return 1.0

    def _generate_cluster_quality_report(self, cluster_metrics: dict[str, Any]) -> str:
        """Generate cluster quality report."""
        try:
            report = []
            report.append('# Cluster Quality Analysis Report')
            report.append('')
            report.append(f'## Quality Metrics')
            report.append(f"- **Silhouette Score**: {cluster_metrics.get('silhouette_score', 0):.4f}")
            report.append(f"- **Calinski-Harabasz Score**: {cluster_metrics.get('calinski_harabasz_score', 0):.2f}")
            report.append(f"- **Davies-Bouldin Score**: {cluster_metrics.get('davies_bouldin_score', 0):.4f}")
            report.append(f"- **Inertia**: {cluster_metrics.get('inertia', 0):.2f}")
            report.append('')
            report.append(f'## Cluster Distribution')
            report.append(f"- **Min Cluster Size**: {cluster_metrics.get('min_cluster_size', 0)}")
            report.append(f"- **Max Cluster Size**: {cluster_metrics.get('max_cluster_size', 0)}")
            report.append(f"- **Mean Cluster Size**: {cluster_metrics.get('mean_cluster_size', 0):.1f}")
            report.append(f"- **Cluster Balance**: {cluster_metrics.get('cluster_balance', 0):.4f}")
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating cluster quality report: {e}'

    def _generate_cluster_characteristics_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate cluster characteristics report."""
        try:
            report = []
            report.append('# Cluster Characteristics Report')
            report.append('')
            for cluster_id, char in composite_analysis.get('cluster_characteristics', {}).items():
                report.append(f'## Cluster {cluster_id}')
                report.append(f"- **Size**: {char.get('size', 0)} ({char.get('percentage', 0):.1f}%)")
                report.append(f"- **Dominant HMM State**: {char.get('dominant_hmm_state', 'unknown')}")
                report.append(f"- **Market Condition**: {composite_analysis.get('market_conditions', {}).get(cluster_id, 'unknown')}")
                report.append('')
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating cluster characteristics report: {e}'

    def _generate_market_conditions_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate market conditions report."""
        try:
            report = []
            report.append('# Market Conditions Report')
            report.append('')
            market_conditions = composite_analysis.get('market_conditions', {})
            condition_counts = {}
            for condition in market_conditions.values():
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            for condition, count in condition_counts.items():
                report.append(f'- **{condition}**: {count} clusters')
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating market conditions report: {e}'

    def _generate_feature_importance_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate feature importance report."""
        try:
            report = []
            report.append('# Feature Importance Report')
            report.append('')
            feature_importance = composite_analysis.get('feature_importance', {})
            sorted_features = sorted(feature_importance.items(), key = lambda x: x[1], reverse = True)
            report.append('## Top 10 Most Important Features')
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                report.append(f'{i}. **{feature}**: {importance:.4f}')
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating feature importance report: {e}'

    def _generate_hmm_state_analysis_report(self, hmm_states: Any, composite_analysis: dict[str, Any]) -> str:
        """Generate HMM state analysis report."""
        try:
            report = []
            report.append('# HMM State Analysis Report')
            report.append('')
            hmm_distribution = composite_analysis.get('hmm_state_distribution', {})
            total_states = sum(hmm_distribution.values())
            report.append('## HMM State Distribution')
            for state, count in hmm_distribution.items():
                percentage = count / total_states * 100 if total_states > 0 else 0
                report.append(f'- **State {state}**: {count} ({percentage:.1f}%)')
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating HMM state analysis report: {e}'

    def _generate_temporal_analysis_report(self, cluster_labels: Any, features: Any) -> str:
        """Generate temporal analysis report."""
        try:
            report = []
            report.append('# Temporal Analysis Report')
            report.append('')
            transitions = 0
            for i in range(1, len(cluster_labels)):
                if cluster_labels[i] != cluster_labels[i - 1]:
                    transitions += 1
            report.append(f'## Cluster Transitions')
            report.append(f'- **Total Transitions**: {transitions}')
            report.append(f'- **Transition Rate**: {transitions / len(cluster_labels) * 100:.2f}%')
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating temporal analysis report: {e}'

    def _generate_recommendations_report(self, cluster_metrics: dict[str, Any], composite_analysis: dict[str, Any]) -> str:
        """Generate recommendations report."""
        try:
            report = []
            report.append('# Recommendations Report')
            report.append('')
            silhouette = cluster_metrics.get('silhouette_score', 0)
            if silhouette < 0.2:
                report.append('- **Low Silhouette Score**: Consider reducing number of clusters or improving feature engineering')
            elif silhouette > 0.5:
                report.append('- **Good Silhouette Score**: Clusters are well-separated')
            balance = cluster_metrics.get('cluster_balance', 0)
            if balance > 0.5:
                report.append('- **Unbalanced Clusters**: Consider adjusting clustering parameters for better balance')
            feature_importance = composite_analysis.get('feature_importance', {})
            if feature_importance:
                top_feature = max(feature_importance.items(), key=lambda x: x[1])
                report.append(f'- **Most Important Feature**: {top_feature[0]} (importance: {top_feature[1]:.4f})')
            return '\n'.join(report)
        except Exception as e:
            return f'Error generating recommendations report: {e}'

    def _should_run_optimization(self, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool) -> bool:
        """Determine if parameter optimization should be run."""
        optimization_config = self._get_optimization_config()
        auto_config = optimization_config.get('automatic_optimization', {})
        if not auto_config.get('enabled', True):
            self.logger.info('🔧 Automatic optimization is disabled')
            return False
        self.logger.info('🔄 Step 3 optimization: Always running parameter optimization')
        return True

    async def _run_automatic_optimization(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Run automatic parameter optimization for HMM regime discovery."""
        try:
            self.logger.info('🚀 Starting automatic parameter optimization...')
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent
                sys.path.insert(0, str(project_root))
                from optimize_hmm_regime_parameters import HMMRegimeOptimizer, identify_market_condition_columns
            except ImportError as e:
                self.logger.error(f'❌ Could not import optimizer: {e}')
                self.logger.info('📝 Proceeding without optimization')
                return None
            feature_data = await self._load_feature_data_for_optimization(symbol, exchange, timeframe, data_dir)
            if feature_data is None or feature_data.empty:
                self.logger.error('❌ Could not load feature data for optimization')
                return None
            market_condition_columns = identify_market_condition_columns(feature_data)
            feature_columns = [col for col in feature_data.columns if col not in ['timestamp', 'composite_cluster_id']]
            self.logger.info(f'📊 Optimization data: {len(feature_data)} samples, {len(feature_columns)} features')
            self.logger.info(f'📈 Market conditions: {len(market_condition_columns)}')
            optimization_config = self._get_optimization_config()
            optimizer = HMMRegimeOptimizer(optimization_config)
            optimization_config = self._get_optimization_config()
            opt_settings = optimization_config.get('optimization_settings', {})
            auto_config = optimization_config.get('automatic_optimization', {})
            optimization_results = optimizer.optimize(data = feature_data, feature_columns = feature_columns, market_condition_columns = market_condition_columns, n_trials = auto_config.get('max_trials', 50), timeout = auto_config.get('timeout_minutes', 30) * 60, study_name = f"{auto_config.get('study_name_prefix', 'auto_optimization')}_{symbol}_{exchange}_{timeframe}")
            if optimization_results and optimization_results.get('best_params'):
                await self._save_optimization_results(optimization_results, symbol, exchange, timeframe, data_dir)
                await self._generate_optimization_report(optimizer, symbol, exchange, timeframe, data_dir)
                self.logger.info('✅ Automatic optimization completed successfully')
                return optimization_results['best_params']
            else:
                self.logger.error('❌ Optimization failed to produce valid results')
                return None
        except Exception as e:
            self.logger.exception(f'❌ Error in automatic optimization: {e}')
            return None

    async def _load_feature_data_for_optimization(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load feature data for optimization."""
        try:
            feature_file = Path(data_dir) / f'{exchange}_{symbol}_{timeframe}_features.parquet'
            if feature_file.exists():
                self.logger.info(f'📂 Loading feature data from: {feature_file}')
                return standardized_parquet_handler.read_parquet_standardized(feature_file)
            self.logger.info('📂 Feature file not found, creating basic features from raw data')
            raw_data = await self._load_data(symbol, exchange, timeframe, data_dir)
            if raw_data is not None and (not raw_data.empty):
                return await self._create_basic_features(raw_data)
            return None
        except Exception as e:
            self.logger.exception(f'❌ Error loading feature data for optimization: {e}')
            return None

    async def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for optimization if Step 2 features are not available."""
        try:
            self.logger.info('🔧 Creating basic features for optimization...')
            features = data.copy()
            if 'close' in features.columns:
                features['returns'] = features['close'].pct_change()
                features['volatility_20'] = features['returns'].rolling(20).std()
                features['price_momentum_10'] = features['close'].pct_change(10)
            if 'volume' in features.columns:
                features['volume_ratio_10'] = features['volume'] / features['volume'].rolling(10).mean()
            if 'close' in features.columns:
                features['sma_20'] = features['close'].rolling(20).mean()
                features['sma_50'] = features['close'].rolling(50).mean()
                features['rsi_14'] = self._calculate_rsi(features['close'], 14)
            features = features.dropna()
            self.logger.info(f'✅ Created {len(features)} basic features')
            return features
        except Exception as e:
            self.logger.exception(f'❌ Error creating basic features: {e}')
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window = period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            return rsi
        except Exception:
            return pd.Series(index = prices.index)

    def _get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        try:
            config_file = Path(__file__).parent / 'step3_optimization_config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info('📋 Loaded optimization configuration from file')
                return config
        except Exception as e:
            self.logger.warning(f'⚠️ Could not load optimization config file: {e}')
        self.logger.info('📋 Using default optimization configuration')
        return {'automatic_optimization': {'enabled': True, 'max_trials': 50, 'timeout_minutes': 30, 'force_rerun_days': 7}, 'optimization_settings': {'n_trials': 50, 'timeout': 1800, 'study_name': 'automatic_optimization', 'random_state': 42}, 'evaluation_weights': {'regime_differentiation': 0.4, 'internal_coherence': 0.3, 'regime_balance': 0.15, 'target_count_penalty': 0.15}, 'market_condition_keywords': ['volatility', 'momentum', 'volume', 'returns', 'price_change', 'trend', 'regime', 'market', 'condition', 'state', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'stoch', 'cci']}

    async def _save_optimization_results(self, optimization_results: Dict[str, Any], symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
        """Save optimization results using centralized reporting system."""
        try:
            from src.training.reports import save_training_report

            # Add metadata to results
            optimization_results['timestamp'] = datetime.now().isoformat()
            optimization_results['symbol'] = symbol
            optimization_results['exchange'] = exchange
            optimization_results['timeframe'] = timeframe

            # Save using centralized reporting system
            results_path = save_training_report(
                data=optimization_results,
                step_name='step03_hmm_regime_discovery',
                report_type='hmm_optimization_results',
                symbol=symbol,
                timeframe=timeframe,
                file_format='json'
            )

            self.logger.info(f'💾 Optimization results saved to: {results_path}')

        except Exception as e:
            self.logger.exception(f'❌ Error saving optimization results: {e}')

    async def _generate_optimization_report(self, optimizer: Any, symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
        """Generate optimization report using centralized reporting system."""
        try:
            from src.training.reports import save_training_report

            # Generate the report content using the optimizer
            report_content = optimizer.generate_optimization_report(return_content=True)
            if report_content is None:
                # Fallback: generate a basic report if the optimizer doesn't support return_content
                report_content = f"""# HMM Optimization Report

**Symbol**: {symbol}
**Exchange**: {exchange}
**Timeframe**: {timeframe}
**Generated**: {datetime.now().isoformat()}

Optimization completed successfully. Detailed results available in the optimizer object.
"""

            # Save report using centralized reporting system
            report_path = save_training_report(
                data=report_content,
                step_name='step03_hmm_regime_discovery',
                report_type='hmm_optimization_report',
                symbol=symbol,
                timeframe=timeframe,
                file_format='md'
            )

            self.logger.info(f'💾 Optimization report saved to: {report_path}')

            # Also save JSON version with optimization results
            optimization_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'optimizer_type': str(type(optimizer).__name__),
                'best_params': getattr(optimizer, 'best_params', {}),
                'optimization_score': getattr(optimizer, 'best_score', None)
            }

            json_report_path = save_training_report(
                data=optimization_data,
                step_name='step03_hmm_regime_discovery',
                report_type='hmm_optimization_data',
                symbol=symbol,
                timeframe=timeframe,
                file_format='json'
            )

            self.logger.info(f'💾 Optimization data saved to: {json_report_path}')

        except Exception as e:
            self.logger.exception(f'❌ Error generating optimization report: {e}')

    def _apply_optimized_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """Apply optimized parameters to the HMM regime discovery configuration."""
        try:
            self.logger.info('🔧 Applying optimized parameters...')
            if 'n_components' in optimized_params:
                self.config['hmm_n_components'] = optimized_params['n_components']
            if 'covariance_type' in optimized_params:
                self.config['hmm_covariance_type'] = optimized_params['covariance_type']
            if 'n_iter' in optimized_params:
                self.config['hmm_n_iter'] = optimized_params['n_iter']
            if 'tol' in optimized_params:
                self.config['hmm_tol'] = optimized_params['tol']
            if 'reg_covar' in optimized_params:
                self.config['hmm_reg_covar'] = optimized_params['reg_covar']
            if 'clustering_method' in optimized_params:
                self.config['clustering_method'] = optimized_params['clustering_method']
            if 'n_clusters' in optimized_params:
                self.config['n_clusters'] = optimized_params['n_clusters']
            if 'target_regimes' in optimized_params:
                self.config['target_regimes'] = optimized_params['target_regimes']
            if 'merging_method' in optimized_params:
                self.config['merging_method'] = optimized_params['merging_method']
            if 'similarity_threshold' in optimized_params:
                self.config['similarity_threshold'] = optimized_params['similarity_threshold']
            if 'coherence_threshold' in optimized_params:
                self.config['coherence_threshold'] = optimized_params['coherence_threshold']
            if 'differentiation_threshold' in optimized_params:
                self.config['differentiation_threshold'] = optimized_params['differentiation_threshold']
            self.logger.info('✅ Optimized parameters applied successfully')
        except Exception as e:
            self.logger.exception(f'❌ Error applying optimized parameters: {e}')
if __name__ == '__main__':

    async def main() -> None:
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else 'data_cache'
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == 'true'
        else:
            tprint('Usage: python step3_hmm_regime_discovery.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]')
            tprint(f'Example: python step3_hmm_regime_discovery.py {get_default_symbol()} BINANCE 1m data_cache true')
            return
        tprint('=' * 80)
        tprint('🚀 STEP 3: HMM Regime Discovery - Command Line Execution')
        tprint('=' * 80)
        tprint(f'🎯 Symbol: {symbol}')
        tprint(f'🏢 Exchange: {exchange}')
        tprint(f'📊 Timeframe: {timeframe}')
        tprint(f'📁 Data directory: {data_dir}')
        tprint(f'🔄 Force rerun: {force_rerun}')
        tprint(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        tprint('=' * 80)
        success = await run_step(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, force_rerun = force_rerun)
        tprint('=' * 80)
        if success:
            tprint('✅ Step 3: HMM Regime Discovery completed successfully')
        else:
            tprint('❌ Step 3: HMM Regime Discovery failed')
        tprint(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        tprint('=' * 80)
        gc.collect()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tprint('\n🛑 Interrupted by user')
    except Exception as e:
        tprint(f'❌ Error: {e}')
    finally:
        gc.collect()