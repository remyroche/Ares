#!/usr/bin/env python3
"""
Enhanced HMM Clustering for Market Analysis

This module provides comprehensive HMM clustering capabilities for market regime detection,
leveraging all common utilities for optimal performance and reliability.

Key Features:
- M1 hardware optimization (GPU, CPU, Memory)
- Matrix operations integration
- ML common utilities (CV, HPO, feature selection)
- Data processing utilities (klines, parquet)
- Math validation and error handling
- Comprehensive logging and monitoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import json

# Import common utilities
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, optimize_memory_usage
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_nan_to_num
)
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.serialization_utils import UniversalSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.ml_common.hmm_regime_detection import EnhancedHMMRegimeDetector
from src.utils.ml_common.matrix_cross_validation import TemporalCrossValidator
from src.utils.ml_common.feature_selection import FeatureSelector
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer

# Setup logging
logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """Enumeration of market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

@dataclass
class HMMClusteringConfig:
    """Configuration for HMM clustering."""
    # HMM Parameters
    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    
    # Feature Engineering
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "atr", "stochastic"
    ])
    
    # Optimization
    use_gpu: bool = True
    use_memory_optimization: bool = True
    use_cpu_optimization: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    purged_cv: bool = True
    
    # Feature Selection
    feature_selection_method: str = "mrmr"
    max_features: int = 50
    
    # Data Processing
    min_data_points: int = 1000
    max_missing_ratio: float = 0.1
    
    # Regime Analysis
    min_regime_duration: int = 10
    regime_stability_threshold: float = 0.7

@dataclass
class HMMClusteringResult:
    """Result container for HMM clustering."""
    model: Any
    regime_labels: np.ndarray
    regime_probabilities: np.ndarray
    regime_characteristics: Dict[str, Any]
    feature_importance: Dict[str, float]
    performance_metrics: Dict[str, float]
    config: HMMClusteringConfig
    processing_time: float
    memory_usage: Dict[str, float]

class EnhancedHMMClustering:
    """
    Enhanced HMM Clustering for Market Regime Detection.
    
    This class provides comprehensive HMM clustering capabilities with full integration
    of common utilities for optimal performance and reliability.
    """
    
    def __init__(self, config: Optional[HMMClusteringConfig] = None):
        """Initialize the enhanced HMM clustering system."""
        self.config = config or HMMClusteringConfig()
        self.logger = logger.getChild("EnhancedHMMClustering")
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager() if self.config.use_gpu else None
        self.memory_optimizer = get_m1_memory_optimizer() if self.config.use_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if self.config.use_cpu_optimization else None
        
        # Initialize utilities
        self.klines_manager = KlinesParquetManager()
        self.serializer = UniversalSerializer()
        self.matrix_ops = UnifiedMatrixOperations()
        self.hmm_detector = EnhancedHMMRegimeDetector()
        self.cv_validator = TemporalCrossValidator()
        self.feature_selector = FeatureSelector()
        
        # State tracking
        self.is_fitted = False
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self.logger.info("Enhanced HMM Clustering initialized successfully")
    
    def load_market_data(
        self, 
        symbol: str, 
        interval: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load market data using the klines parquet manager.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Data interval (e.g., '1h', '4h', '1d')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with market data
        """
        try:
            self.logger.info(f"Loading market data for {symbol} {interval}")
            
            # Get data info
            data_info = self.klines_manager.get_data_info(symbol, interval)
            if not data_info['available']:
                raise ValueError(f"No data available for {symbol} {interval}")
            
            # Load data
            data = self.klines_manager.load_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None or data.empty:
                raise ValueError("No data loaded")
            
            # Validate data quality
            quality_metrics = calculate_data_quality_metrics(data)
            self.logger.info(f"Data quality metrics: {quality_metrics}")
            
            # Check minimum data points
            if len(data) < self.config.min_data_points:
                raise ValueError(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
            
            # Check missing data ratio
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > self.config.max_missing_ratio:
                raise ValueError(f"Too much missing data: {missing_ratio:.2%} > {self.config.max_missing_ratio:.2%}")
            
            self.logger.info(f"Successfully loaded {len(data)} data points")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            raise
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for HMM clustering using common utilities.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info("Engineering features for HMM clustering")
            
            features = data.copy()
            
            # Calculate technical indicators
            for window in self.config.lookback_windows:
                # RSI
                if "rsi" in self.config.technical_indicators:
                    features[f"rsi_{window}"] = self._calculate_rsi(features['close'], window)
                
                # MACD
                if "macd" in self.config.technical_indicators:
                    macd_line, macd_signal, macd_hist = self._calculate_macd(features['close'], window, window*2, window*3)
                    features[f"macd_{window}"] = macd_line
                    features[f"macd_signal_{window}"] = macd_signal
                    features[f"macd_hist_{window}"] = macd_hist
                
                # Bollinger Bands
                if "bollinger_bands" in self.config.technical_indicators:
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(features['close'], window)
                    features[f"bb_upper_{window}"] = bb_upper
                    features[f"bb_middle_{window}"] = bb_middle
                    features[f"bb_lower_{window}"] = bb_lower
                    features[f"bb_width_{window}"] = (bb_upper - bb_lower) / bb_middle
                
                # ATR
                if "atr" in self.config.technical_indicators:
                    features[f"atr_{window}"] = self._calculate_atr(features, window)
                
                # Stochastic
                if "stochastic" in self.config.technical_indicators:
                    stoch_k, stoch_d = self._calculate_stochastic(features, window)
                    features[f"stoch_k_{window}"] = stoch_k
                    features[f"stoch_d_{window}"] = stoch_d
            
            # Price-based features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['price_momentum'] = features['close'] / features['close'].shift(20) - 1
            
            # Volume features
            if 'volume' in features.columns:
                features['volume_ma'] = features['volume'].rolling(window=20).mean()
                features['volume_ratio'] = features['volume'] / features['volume_ma']
                features['price_volume'] = features['close'] * features['volume']
            
            # High-Low features
            if all(col in features.columns for col in ['high', 'low']):
                features['hl_ratio'] = features['high'] / features['low']
                features['body_size'] = abs(features['close'] - features['open']) / features['high']
                features['upper_shadow'] = (features['high'] - features[['open', 'close']].max(axis=1)) / features['high']
                features['lower_shadow'] = (features[['open', 'close']].min(axis=1) - features['low']) / features['high']
            
            # Remove rows with NaN values
            features = features.dropna()
            
            # Validate features
            if features.empty:
                raise ValueError("No valid features after engineering")
            
            self.logger.info(f"Engineered {len(features.columns)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to engineer features: {e}")
            raise
    
    def select_features(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Select optimal features using ML common utilities.
        
        Args:
            features: Feature DataFrame
            target: Target variable (optional)
            
        Returns:
            DataFrame with selected features
        """
        try:
            self.logger.info("Selecting optimal features")
            
            # Prepare features for selection
            feature_data = features.select_dtypes(include=[np.number])
            
            # Remove constant features
            feature_data = feature_data.loc[:, feature_data.std() > 0]
            
            # Use feature selector
            selected_features = self.feature_selector.select_features(
                X=feature_data,
                y=target,
                method=self.config.feature_selection_method,
                max_features=self.config.max_features
            )
            
            # Get selected feature names
            selected_feature_names = feature_data.columns[selected_features].tolist()
            
            # Filter original features
            result = features[selected_feature_names]
            
            self.logger.info(f"Selected {len(selected_feature_names)} features out of {len(feature_data.columns)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to select features: {e}")
            raise
    
    def fit_hmm_model(self, features: pd.DataFrame) -> HMMClusteringResult:
        """
        Fit HMM model with hardware optimization.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            HMMClusteringResult with model and results
        """
        try:
            start_time = time.time()
            self.logger.info("Fitting HMM model")
            
            # Prepare features
            feature_array = features.values.astype(np.float32)
            self.feature_names = features.columns.tolist()
            
            # Memory optimization
            if self.memory_optimizer:
                feature_array = self.memory_optimizer.create_memory_efficient_array(
                    feature_array, dtype=np.float32
                )
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(feature_array)
            
            # GPU optimization
            if self.gpu_manager and self.config.use_gpu:
                features_scaled = self.gpu_manager.to_device(features_scaled, "matrix_mult")
                
                with self.gpu_manager.gpu_context("hmm_training"):
                    model = self._train_hmm_gpu(features_scaled)
            else:
                # CPU optimization
                if self.cpu_optimizer:
                    features_scaled = self.cpu_optimizer.optimize_array(features_scaled)
                
                model = self._train_hmm_cpu(features_scaled)
            
            # Get regime labels and probabilities
            regime_labels = model.predict(features_scaled)
            regime_probabilities = model.predict_proba(features_scaled)
            
            # Analyze regime characteristics
            regime_characteristics = self._analyze_regime_characteristics(
                features, regime_labels, regime_probabilities
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                features, regime_labels, regime_probabilities
            )
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                features, regime_labels
            )
            
            # Memory usage
            memory_usage = {}
            if self.memory_optimizer:
                memory_usage = self.memory_optimizer.get_memory_usage()
            
            processing_time = time.time() - start_time
            
            # Create result
            result = HMMClusteringResult(
                model=model,
                regime_labels=regime_labels,
                regime_probabilities=regime_probabilities,
                regime_characteristics=regime_characteristics,
                feature_importance=feature_importance,
                performance_metrics=performance_metrics,
                config=self.config,
                processing_time=processing_time,
                memory_usage=memory_usage
            )
            
            self.model = model
            self.is_fitted = True
            
            self.logger.info(f"HMM model fitted successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fit HMM model: {e}")
            raise
    
    def _train_hmm_gpu(self, features_scaled: np.ndarray) -> Any:
        """Train HMM model with GPU acceleration."""
        from hmmlearn import hmm
        
        model = hmm.GaussianHMM(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state
        )
        
        # Convert back to CPU for hmmlearn
        if hasattr(features_scaled, 'cpu'):
            features_cpu = features_scaled.cpu().numpy()
        else:
            features_cpu = features_scaled
        
        model.fit(features_cpu)
        return model
    
    def _train_hmm_cpu(self, features_scaled: np.ndarray) -> Any:
        """Train HMM model on CPU."""
        from hmmlearn import hmm
        
        model = hmm.GaussianHMM(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state
        )
        
        model.fit(features_scaled)
        return model
    
    def _analyze_regime_characteristics(
        self, 
        features: pd.DataFrame, 
        regime_labels: np.ndarray, 
        regime_probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        try:
            characteristics = {}
            
            for regime in range(self.config.n_components):
                regime_mask = regime_labels == regime
                regime_data = features[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                regime_char = {
                    'count': len(regime_data),
                    'percentage': len(regime_data) / len(features) * 100,
                    'mean_returns': regime_data['returns'].mean() if 'returns' in regime_data.columns else 0,
                    'volatility': regime_data['volatility'].mean() if 'volatility' in regime_data.columns else 0,
                    'mean_price': regime_data['close'].mean() if 'close' in regime_data.columns else 0,
                }
                
                # Add technical indicator characteristics
                for col in regime_data.columns:
                    if col.startswith(('rsi_', 'macd_', 'bb_', 'atr_', 'stoch_')):
                        regime_char[f'{col}_mean'] = regime_data[col].mean()
                        regime_char[f'{col}_std'] = regime_data[col].std()
                
                characteristics[f'regime_{regime}'] = regime_char
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze regime characteristics: {e}")
            return {}
    
    def _calculate_performance_metrics(
        self, 
        features: pd.DataFrame, 
        regime_labels: np.ndarray, 
        regime_probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics for the HMM model."""
        try:
            metrics = {}
            
            # Regime stability
            regime_changes = np.sum(np.diff(regime_labels) != 0)
            metrics['regime_stability'] = 1 - (regime_changes / len(regime_labels))
            
            # Regime balance
            unique_regimes, counts = np.unique(regime_labels, return_counts=True)
            regime_balance = 1 - np.std(counts) / np.mean(counts)
            metrics['regime_balance'] = regime_balance
            
            # Probability confidence
            max_probs = np.max(regime_probabilities, axis=1)
            metrics['avg_confidence'] = np.mean(max_probs)
            metrics['min_confidence'] = np.min(max_probs)
            
            # Regime duration statistics
            regime_durations = []
            current_regime = regime_labels[0]
            current_duration = 1
            
            for i in range(1, len(regime_labels)):
                if regime_labels[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = regime_labels[i]
                    current_duration = 1
            
            regime_durations.append(current_duration)
            
            if regime_durations:
                metrics['avg_regime_duration'] = np.mean(regime_durations)
                metrics['min_regime_duration'] = np.min(regime_durations)
                metrics['max_regime_duration'] = np.max(regime_durations)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def _calculate_feature_importance(
        self, 
        features: pd.DataFrame, 
        regime_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance for regime detection."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Train a random forest to predict regimes
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, regime_labels)
            
            # Get feature importance
            importance = rf.feature_importances_
            feature_names = features.columns.tolist()
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def predict_regimes(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes for new data.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (regime_labels, regime_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare features
            feature_array = features.values.astype(np.float32)
            
            # Scale features
            features_scaled = self.scaler.transform(feature_array)
            
            # Predict
            regime_labels = self.model.predict(features_scaled)
            regime_probabilities = self.model.predict_proba(features_scaled)
            
            return regime_labels, regime_probabilities
            
        except Exception as e:
            self.logger.error(f"Failed to predict regimes: {e}")
            raise
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model and configuration."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            
            return self.serializer.save(model_data, filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model and configuration."""
        try:
            model_data = self.serializer.load(filepath)
            
            if model_data is None:
                return False
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.is_fitted = model_data['is_fitted']
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _calculate_stochastic(self, data: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent


def run_hmm_clustering_analysis(
    symbol: str,
    interval: str = "1h",
    config: Optional[HMMClusteringConfig] = None,
    save_results: bool = True,
    output_dir: str = "market_analysis/hmm_clustering/results"
) -> HMMClusteringResult:
    """
    Run complete HMM clustering analysis for market regime detection.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Data interval (e.g., '1h', '4h', '1d')
        config: HMM clustering configuration
        save_results: Whether to save results to disk
        output_dir: Directory to save results
        
    Returns:
        HMMClusteringResult with complete analysis
    """
    try:
        # Initialize clustering system
        clustering = EnhancedHMMClustering(config)
        
        # Load market data
        data = clustering.load_market_data(symbol, interval)
        
        # Engineer features
        features = clustering.engineer_features(data)
        
        # Select optimal features
        selected_features = clustering.select_features(features)
        
        # Fit HMM model
        result = clustering.fit_hmm_model(selected_features)
        
        # Save results if requested
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = output_path / f"hmm_model_{symbol}_{interval}.pkl"
            clustering.save_model(str(model_path))
            
            # Save results
            results_path = output_path / f"hmm_results_{symbol}_{interval}.json"
            results_data = {
                'regime_labels': result.regime_labels.tolist(),
                'regime_probabilities': result.regime_probabilities.tolist(),
                'regime_characteristics': result.regime_characteristics,
                'feature_importance': result.feature_importance,
                'performance_metrics': result.performance_metrics,
                'config': result.config.__dict__,
                'processing_time': result.processing_time,
                'memory_usage': result.memory_usage
            }
            
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to run HMM clustering analysis: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    config = HMMClusteringConfig(
        n_components=4,
        lookback_windows=[5, 10, 20, 50],
        technical_indicators=["rsi", "macd", "bollinger_bands", "atr"],
        use_gpu=True,
        use_memory_optimization=True
    )
    
    result = run_hmm_clustering_analysis(
        symbol="BTCUSDT",
        interval="1h",
        config=config,
        save_results=True
    )
    
    print(f"HMM Clustering completed in {result.processing_time:.2f}s")
    print(f"Regime characteristics: {result.regime_characteristics}")
    print(f"Performance metrics: {result.performance_metrics}")