"""
Feature Engine Integration

This module provides integration with the existing feature_generation/ system,
including feature preprocessing, regime-aware feature engineering, and
financial-specific feature transformations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import warnings
import sys
import os

# Add the feature_generation path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../feature_generation'))

try:
    from feature_engineer import FeatureEngineer
    from feature_engineer.technical_indicators import TechnicalIndicators
    from feature_engineer.regime_detection import RegimeDetector as BaseRegimeDetector
    from feature_engineer.feature_selection import FeatureSelector
    from feature_engineer.feature_transformation import FeatureTransformer
except ImportError as e:
    logging.warning(f"Could not import feature_generation modules: {e}")
    # Create dummy classes for fallback
    class FeatureEngineer:
        def __init__(self, *args, **kwargs):
            pass
        def fit_transform(self, data):
            return data
        def transform(self, data):
            return data
    
    class TechnicalIndicators:
        def __init__(self, *args, **kwargs):
            pass
        def calculate_indicators(self, data):
            return data
    
    class BaseRegimeDetector:
        def __init__(self, *args, **kwargs):
            pass
        def detect_regimes(self, data):
            return np.zeros(len(data))
    
    class FeatureSelector:
        def __init__(self, *args, **kwargs):
            pass
        def select_features(self, data, target):
            return data
    
    class FeatureTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def transform(self, data):
            return data

from .financial_architecture_primitives import RegimeType, FinancialActivationType

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

except ImportError:
    
    cp = None

logger = logging.getLogger(__name__)

class FeatureIntegrationMode(Enum):
    """Modes for feature engine integration."""
    FULL_INTEGRATION = "full_integration"
    PARTIAL_INTEGRATION = "partial_integration"
    FALLBACK_MODE = "fallback_mode"
    CUSTOM_FEATURES = "custom_features"

class RegimeAwareFeatureMode(Enum):
    """Modes for regime-aware feature engineering."""
    REGIME_SPECIFIC = "regime_specific"
    REGIME_ADAPTIVE = "regime_adaptive"
    REGIME_ENSEMBLE = "regime_ensemble"
    REGIME_TRANSITION = "regime_transition"

@dataclass
class FeatureEngineIntegrationConfig:
    """Configuration for feature engine integration."""
    # Integration mode
    integration_mode: FeatureIntegrationMode = FeatureIntegrationMode.FULL_INTEGRATION
    enable_regime_awareness: bool = True
    regime_aware_mode: RegimeAwareFeatureMode = RegimeAwareFeatureMode.REGIME_ADAPTIVE
    
    # Feature engineering settings
    enable_technical_indicators: bool = True
    enable_regime_detection: bool = True
    enable_feature_selection: bool = True
    enable_feature_transformation: bool = True
    
    # Technical indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic', 'williams_r', 'cci', 'atr', 'adx'
    ])
    indicator_periods: Dict[str, int] = field(default_factory=lambda: {
        'sma': 20, 'ema': 20, 'rsi': 14, 'macd': (12, 26, 9), 'bollinger_bands': 20,
        'stochastic': 14, 'williams_r': 14, 'cci': 20, 'atr': 14, 'adx': 14
    })
    
    # Regime detection
    regime_detection_method: str = "gaussian_mixture"  # gaussian_mixture, kmeans, hidden_markov
    n_regimes: int = 4
    regime_window: int = 50
    regime_stability_threshold: float = 0.7
    
    # Feature selection
    feature_selection_method: str = "mutual_information"  # mutual_information, chi2, f_score, lasso
    n_features: int = 50
    feature_selection_threshold: float = 0.01
    
    # Feature transformation
    enable_scaling: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust, quantile
    enable_pca: bool = False
    pca_components: int = 10
    enable_polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Regime-aware features
    enable_regime_specific_features: bool = True
    enable_regime_transition_features: bool = True
    enable_regime_stability_features: bool = True
    enable_regime_volatility_features: bool = True
    
    # Financial-specific features
    enable_financial_features: bool = True
    enable_volatility_features: bool = True
    enable_momentum_features: bool = True
    enable_mean_reversion_features: bool = True
    enable_risk_features: bool = True
    
    # Feature validation
    enable_feature_validation: bool = True
    validation_threshold: float = 0.01
    max_correlation: float = 0.95
    
    # Performance optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    chunk_size: int = 1000

@dataclass
class FeatureIntegrationResult:
    """Result from feature engine integration."""
    processed_features: np.ndarray
    feature_names: List[str]
    regime_labels: np.ndarray
    regime_probabilities: np.ndarray
    feature_importance: Dict[str, float]
    feature_correlations: Dict[str, Dict[str, float]]
    regime_analysis: Dict[str, Any]
    feature_validation: Dict[str, Any]
    processing_time: float
    n_features: int
    n_regimes: int

class FeatureEngineIntegrator:
    """Integrates with the existing feature_generation system."""
    
    def __init__(self, config: FeatureEngineIntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize feature engineering components
        self.feature_engineer = None
        self.technical_indicators = None
        self.regime_detector = None
        self.feature_selector = None
        self.feature_transformer = None
        
        # Feature storage
        self.processed_features = None
        self.feature_names = []
        self.regime_labels = None
        self.regime_probabilities = None
        
        # Performance tracking
        self.feature_importance = {}
        self.feature_correlations = {}
        self.regime_analysis = {}
        
        self._initialize_components()
        
        self.logger.info("✅ Feature Engine Integration initialized")
        self.logger.info(f"   Integration Mode: {config.integration_mode.value}")
        self.logger.info(f"   Regime Awareness: {config.enable_regime_awareness}")
        self.logger.info(f"   Technical Indicators: {config.enable_technical_indicators}")
    
    def _initialize_components(self):
        """Initialize feature engineering components."""
        try:
            if self.config.integration_mode == FeatureIntegrationMode.FULL_INTEGRATION:
                self._initialize_full_integration()
            elif self.config.integration_mode == FeatureIntegrationMode.PARTIAL_INTEGRATION:
                self._initialize_partial_integration()
            else:  # FALLBACK_MODE
                self._initialize_fallback_mode()
        except Exception as e:
            self.logger.warning(f"Failed to initialize components: {e}")
            self._initialize_fallback_mode()
    
    def _initialize_full_integration(self):
        """Initialize full integration with feature_generation system."""
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Initialize technical indicators
        if self.config.enable_technical_indicators:
            self.technical_indicators = TechnicalIndicators(
                indicators=self.config.technical_indicators,
                periods=self.config.indicator_periods
            )
        
        # Initialize regime detector
        if self.config.enable_regime_detection:
            self.regime_detector = BaseRegimeDetector(
                method=self.config.regime_detection_method,
                n_regimes=self.config.n_regimes,
                window=self.config.regime_window
            )
        
        # Initialize feature selector
        if self.config.enable_feature_selection:
            self.feature_selector = FeatureSelector(
                method=self.config.feature_selection_method,
                n_features=self.config.n_features,
                threshold=self.config.feature_selection_threshold
            )
        
        # Initialize feature transformer
        if self.config.enable_feature_transformation:
            self.feature_transformer = FeatureTransformer(
                scaling_method=self.config.scaling_method,
                enable_pca=self.config.enable_pca,
                pca_components=self.config.pca_components
            )
    
    def _initialize_partial_integration(self):
        """Initialize partial integration with feature_generation system."""
        # Initialize only essential components
        self.feature_engineer = FeatureEngineer()
        
        if self.config.enable_technical_indicators:
            self.technical_indicators = TechnicalIndicators(
                indicators=self.config.technical_indicators[:5],  # Use only first 5 indicators
                periods=self.config.indicator_periods
            )
    
    def _initialize_fallback_mode(self):
        """Initialize fallback mode without feature_generation system."""
        # Create dummy components
        self.feature_engineer = FeatureEngineer()
        self.technical_indicators = TechnicalIndicators()
        self.regime_detector = BaseRegimeDetector()
        self.feature_selector = FeatureSelector()
        self.feature_transformer = FeatureTransformer()
    
    def process_features(self, market_data: pd.DataFrame, 
                        target_data: Optional[np.ndarray] = None) -> FeatureIntegrationResult:
        """Process features using the integrated feature engine."""
        start_time = time.time()
        self.logger.info("🔧 Processing features with integrated feature engine...")
        
        try:
            # Step 1: Basic feature engineering
            basic_features = self._process_basic_features(market_data)
            
            # Step 2: Technical indicators
            if self.config.enable_technical_indicators:
                technical_features = self._process_technical_indicators(market_data)
                basic_features = np.hstack([basic_features, technical_features])
            
            # Step 3: Regime detection
            if self.config.enable_regime_detection:
                regime_labels, regime_probabilities = self._detect_regimes(market_data)
            else:
                regime_labels = np.zeros(len(market_data))
                regime_probabilities = np.ones((len(market_data), 1))
            
            # Step 4: Regime-aware features
            if self.config.enable_regime_awareness:
                regime_features = self._process_regime_aware_features(market_data, regime_labels)
                basic_features = np.hstack([basic_features, regime_features])
            
            # Step 5: Financial-specific features
            if self.config.enable_financial_features:
                financial_features = self._process_financial_features(market_data)
                basic_features = np.hstack([basic_features, financial_features])
            
            # Step 6: Feature selection
            if self.config.enable_feature_selection and target_data is not None:
                selected_features, feature_importance = self._select_features(basic_features, target_data)
                basic_features = selected_features
                self.feature_importance = feature_importance
            
            # Step 7: Feature transformation
            if self.config.enable_feature_transformation:
                transformed_features = self._transform_features(basic_features)
                basic_features = transformed_features
            
            # Step 8: Feature validation
            if self.config.enable_feature_validation:
                validation_results = self._validate_features(basic_features)
            else:
                validation_results = {}
            
            # Step 9: Regime analysis
            regime_analysis = self._analyze_regimes(regime_labels, regime_probabilities)
            
            # Step 10: Feature correlations
            feature_correlations = self._calculate_feature_correlations(basic_features)
            
            processing_time = time.time() - start_time
            
            # Create feature names
            feature_names = self._create_feature_names(basic_features.shape[1])
            
            return FeatureIntegrationResult(
                processed_features=basic_features,
                feature_names=feature_names,
                regime_labels=regime_labels,
                regime_probabilities=regime_probabilities,
                feature_importance=self.feature_importance,
                feature_correlations=feature_correlations,
                regime_analysis=regime_analysis,
                feature_validation=validation_results,
                processing_time=processing_time,
                n_features=basic_features.shape[1],
                n_regimes=len(np.unique(regime_labels))
            )
            
        except Exception as e:
            self.logger.error(f"Feature processing failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _process_basic_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Process basic features from market data."""
        features = []
        
        # Price-based features
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            
            # Returns
            returns = np.diff(prices) / prices[:-1]
            features.append(returns)
            
            # Log returns
            log_returns = np.diff(np.log(prices))
            features.append(log_returns)
            
            # Price ratios
            price_ratios = prices[1:] / prices[:-1]
            features.append(price_ratios)
            
            # Volatility
            volatility = pd.Series(returns).rolling(window=20).std().values
            features.append(volatility)
        
        # Volume features
        if 'volume' in market_data.columns:
            volume = market_data['volume'].values
            
            # Volume ratios
            volume_ratios = volume[1:] / (volume[:-1] + 1e-8)
            features.append(volume_ratios)
            
            # Volume moving average
            volume_ma = pd.Series(volume).rolling(window=20).mean().values
            volume_ratio = volume / (volume_ma + 1e-8)
            features.append(volume_ratio)
        
        # Combine features
        if features:
            # Pad features to same length
            max_length = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_length:
                    padded_f = np.pad(f, (0, max_length - len(f)), mode='edge')
                else:
                    padded_f = f[:max_length]
                padded_features.append(padded_f)
            
            return np.column_stack(padded_features)
        else:
            # Return dummy features if no data
            return np.random.randn(len(market_data), 5)
    
    def _process_technical_indicators(self, market_data: pd.DataFrame) -> np.ndarray:
        """Process technical indicators."""
        try:
            if self.technical_indicators is None:
                return np.zeros((len(market_data), 10))  # Dummy features
            
            # Calculate technical indicators
            indicators = self.technical_indicators.calculate_indicators(market_data)
            
            # Convert to numpy array
            if isinstance(indicators, pd.DataFrame):
                indicators = indicators.values
            elif isinstance(indicators, dict):
                # Convert dictionary to array
                indicator_values = []
                for indicator_name in self.config.technical_indicators:
                    if indicator_name in indicators:
                        indicator_values.append(indicators[indicator_name])
                indicators = np.column_stack(indicator_values) if indicator_values else np.zeros((len(market_data), 10))
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f"Technical indicators processing failed: {e}")
            return np.zeros((len(market_data), 10))  # Dummy features
    
    def _detect_regimes(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Detect market regimes."""
        try:
            if self.regime_detector is None:
                # Return dummy regimes
                regime_labels = np.random.randint(0, self.config.n_regimes, len(market_data))
                regime_probabilities = np.random.rand(len(market_data), self.config.n_regimes)
                regime_probabilities = regime_probabilities / regime_probabilities.sum(axis=1, keepdims=True)
                return regime_labels, regime_probabilities
            
            # Detect regimes
            regime_labels = self.regime_detector.detect_regimes(market_data)
            
            # Calculate regime probabilities (simplified)
            regime_probabilities = np.zeros((len(market_data), self.config.n_regimes))
            for i, regime in enumerate(regime_labels):
                regime_probabilities[i, regime] = 1.0
            
            return regime_labels, regime_probabilities
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            # Return dummy regimes
            regime_labels = np.random.randint(0, self.config.n_regimes, len(market_data))
            regime_probabilities = np.random.rand(len(market_data), self.config.n_regimes)
            regime_probabilities = regime_probabilities / regime_probabilities.sum(axis=1, keepdims=True)
            return regime_labels, regime_probabilities
    
    def _process_regime_aware_features(self, market_data: pd.DataFrame, 
                                     regime_labels: np.ndarray) -> np.ndarray:
        """Process regime-aware features."""
        features = []
        
        # Regime-specific features
        if self.config.enable_regime_specific_features:
            regime_features = self._calculate_regime_specific_features(market_data, regime_labels)
            features.append(regime_features)
        
        # Regime transition features
        if self.config.enable_regime_transition_features:
            transition_features = self._calculate_regime_transition_features(regime_labels)
            features.append(transition_features)
        
        # Regime stability features
        if self.config.enable_regime_stability_features:
            stability_features = self._calculate_regime_stability_features(regime_labels)
            features.append(stability_features)
        
        # Regime volatility features
        if self.config.enable_regime_volatility_features:
            volatility_features = self._calculate_regime_volatility_features(market_data, regime_labels)
            features.append(volatility_features)
        
        # Combine features
        if features:
            return np.column_stack(features)
        else:
            return np.zeros((len(market_data), 5))  # Dummy features
    
    def _calculate_regime_specific_features(self, market_data: pd.DataFrame, 
                                          regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime-specific features."""
        features = []
        
        # Regime duration
        regime_duration = self._calculate_regime_duration(regime_labels)
        features.append(regime_duration)
        
        # Regime frequency
        regime_frequency = self._calculate_regime_frequency(regime_labels)
        features.append(regime_frequency)
        
        # Regime consistency
        regime_consistency = self._calculate_regime_consistency(regime_labels)
        features.append(regime_consistency)
        
        return np.column_stack(features) if features else np.zeros((len(market_data), 3))
    
    def _calculate_regime_duration(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime duration."""
        duration = np.zeros(len(regime_labels))
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == regime_labels[i-1]:
                current_duration += 1
            else:
                current_duration = 1
            duration[i] = current_duration
        
        return duration
    
    def _calculate_regime_frequency(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime frequency."""
        frequency = np.zeros(len(regime_labels))
        unique_regimes = np.unique(regime_labels)
        
        for i, regime in enumerate(regime_labels):
            count = np.sum(regime_labels[:i+1] == regime)
            frequency[i] = count / (i + 1)
        
        return frequency
    
    def _calculate_regime_consistency(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime consistency."""
        consistency = np.zeros(len(regime_labels))
        window = 10
        
        for i in range(window, len(regime_labels)):
            window_labels = regime_labels[i-window:i]
            unique_labels = len(np.unique(window_labels))
            consistency[i] = 1.0 - (unique_labels - 1) / window
        
        return consistency
    
    def _calculate_regime_transition_features(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition features."""
        features = []
        
        # Transition indicators
        transitions = np.diff(regime_labels) != 0
        transition_indicators = np.concatenate([[False], transitions])
        features.append(transition_indicators.astype(float))
        
        # Transition frequency
        transition_frequency = np.zeros(len(regime_labels))
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                transition_frequency[i] = 1.0
        
        features.append(transition_frequency)
        
        return np.column_stack(features) if features else np.zeros((len(regime_labels), 2))
    
    def _calculate_regime_stability_features(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime stability features."""
        features = []
        
        # Regime stability
        stability = np.zeros(len(regime_labels))
        window = 20
        
        for i in range(window, len(regime_labels)):
            window_labels = regime_labels[i-window:i]
            unique_labels = len(np.unique(window_labels))
            stability[i] = 1.0 - (unique_labels - 1) / window
        
        features.append(stability)
        
        return np.column_stack(features) if features else np.zeros((len(regime_labels), 1))
    
    def _calculate_regime_volatility_features(self, market_data: pd.DataFrame, 
                                            regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime volatility features."""
        features = []
        
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Regime-specific volatility
            regime_volatility = np.zeros(len(regime_labels))
            for regime in np.unique(regime_labels):
                regime_mask = regime_labels == regime
                if np.sum(regime_mask) > 1:
                    regime_returns = returns[regime_mask[:-1]]  # Adjust for returns length
                    regime_vol = np.std(regime_returns)
                    regime_volatility[regime_mask] = regime_vol
            
            features.append(regime_volatility)
        
        return np.column_stack(features) if features else np.zeros((len(regime_labels), 1))
    
    def _process_financial_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Process financial-specific features."""
        features = []
        
        # Volatility features
        if self.config.enable_volatility_features:
            volatility_features = self._calculate_volatility_features(market_data)
            features.append(volatility_features)
        
        # Momentum features
        if self.config.enable_momentum_features:
            momentum_features = self._calculate_momentum_features(market_data)
            features.append(momentum_features)
        
        # Mean reversion features
        if self.config.enable_mean_reversion_features:
            mean_reversion_features = self._calculate_mean_reversion_features(market_data)
            features.append(mean_reversion_features)
        
        # Risk features
        if self.config.enable_risk_features:
            risk_features = self._calculate_risk_features(market_data)
            features.append(risk_features)
        
        # Combine features
        if features:
            return np.column_stack(features)
        else:
            return np.zeros((len(market_data), 5))  # Dummy features
    
    def _calculate_volatility_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate volatility features."""
        features = []
        
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Rolling volatility
            volatility = pd.Series(returns).rolling(window=20).std().values
            features.append(volatility)
            
            # Volatility of volatility
            vol_of_vol = pd.Series(volatility).rolling(window=20).std().values
            features.append(vol_of_vol)
            
            # Volatility ratio
            vol_ratio = volatility / (pd.Series(volatility).rolling(window=50).mean().values + 1e-8)
            features.append(vol_ratio)
        
        return np.column_stack(features) if features else np.zeros((len(market_data), 3))
    
    def _calculate_momentum_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate momentum features."""
        features = []
        
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            
            # Price momentum
            momentum_5 = prices[5:] / prices[:-5] - 1
            momentum_10 = prices[10:] / prices[:-10] - 1
            momentum_20 = prices[20:] / prices[:-20] - 1
            
            # Pad to original length
            momentum_5 = np.pad(momentum_5, (5, 0), mode='edge')
            momentum_10 = np.pad(momentum_10, (10, 0), mode='edge')
            momentum_20 = np.pad(momentum_20, (20, 0), mode='edge')
            
            features.extend([momentum_5, momentum_10, momentum_20])
        
        return np.column_stack(features) if features else np.zeros((len(market_data), 3))
    
    def _calculate_mean_reversion_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate mean reversion features."""
        features = []
        
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            
            # Z-score
            zscore = (prices - pd.Series(prices).rolling(window=20).mean()) / (pd.Series(prices).rolling(window=20).std() + 1e-8)
            features.append(zscore.values)
            
            # Mean reversion ratio
            mean_reversion_ratio = (prices - pd.Series(prices).rolling(window=20).mean()) / prices
            features.append(mean_reversion_ratio.values)
        
        return np.column_stack(features) if features else np.zeros((len(market_data), 2))
    
    def _calculate_risk_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate risk features."""
        features = []
        
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Value at Risk (VaR)
            var_95 = pd.Series(returns).rolling(window=20).quantile(0.05).values
            features.append(var_95)
            
            # Expected Shortfall (CVaR)
            cvar_95 = pd.Series(returns).rolling(window=20).apply(
                lambda x: x[x <= x.quantile(0.05)].mean()
            ).values
            features.append(cvar_95)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (running_max - cumulative_returns) / running_max
            features.append(drawdown)
        
        return np.column_stack(features) if features else np.zeros((len(market_data), 3))
    
    def _select_features(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Select features using the integrated feature selector."""
        try:
            if self.feature_selector is None:
                return features, {}
            
            # Select features
            selected_features = self.feature_selector.select_features(features, target)
            
            # Calculate feature importance
            feature_importance = {}
            if hasattr(self.feature_selector, 'feature_importance_'):
                for i, importance in enumerate(self.feature_selector.feature_importance_):
                    feature_importance[f'feature_{i}'] = importance
            
            return selected_features, feature_importance
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return features, {}
    
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using the integrated feature transformer."""
        try:
            if self.feature_transformer is None:
                return features
            
            # Transform features
            transformed_features = self.feature_transformer.transform(features)
            
            return transformed_features
            
        except Exception as e:
            self.logger.warning(f"Feature transformation failed: {e}")
            return features
    
    def _validate_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Validate features."""
        validation_results = {}
        
        # Check for NaN values
        nan_count = np.isnan(features).sum()
        validation_results['nan_count'] = nan_count
        
        # Check for infinite values
        inf_count = np.isinf(features).sum()
        validation_results['inf_count'] = inf_count
        
        # Check feature variance
        feature_variance = np.var(features, axis=0)
        low_variance_features = np.sum(feature_variance < self.config.validation_threshold)
        validation_results['low_variance_features'] = low_variance_features
        
        # Check feature correlations
        if features.shape[1] > 1:
            correlation_matrix = np.corrcoef(features.T)
            high_correlation_pairs = np.sum(np.abs(correlation_matrix) > self.config.max_correlation) - features.shape[1]
            validation_results['high_correlation_pairs'] = high_correlation_pairs
        
        return validation_results
    
    def _analyze_regimes(self, regime_labels: np.ndarray, regime_probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze regime characteristics."""
        analysis = {}
        
        # Regime distribution
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        regime_distribution = dict(zip(unique_regimes, counts))
        analysis['regime_distribution'] = regime_distribution
        
        # Regime stability
        regime_stability = self._calculate_regime_stability(regime_labels)
        analysis['regime_stability'] = regime_stability
        
        # Regime transitions
        transitions = np.sum(np.diff(regime_labels) != 0)
        analysis['n_transitions'] = transitions
        
        # Regime probabilities
        if regime_probabilities is not None:
            mean_probabilities = np.mean(regime_probabilities, axis=0)
            analysis['mean_regime_probabilities'] = mean_probabilities.tolist()
        
        return analysis
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray) -> float:
        """Calculate regime stability."""
        if len(regime_labels) < 2:
            return 0.0
        
        # Calculate regime consistency
        unique_regimes = np.unique(regime_labels)
        regime_counts = {}
        
        for regime in unique_regimes:
            regime_counts[regime] = np.sum(regime_labels == regime)
        
        # Stability is the ratio of the most frequent regime to total length
        max_count = max(regime_counts.values())
        stability = max_count / len(regime_labels)
        
        return stability
    
    def _calculate_feature_correlations(self, features: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate feature correlations."""
        correlations = {}
        
        if features.shape[1] > 1:
            correlation_matrix = np.corrcoef(features.T)
            
            # Store correlations for each feature pair
            for i in range(features.shape[1]):
                correlations[f'feature_{i}'] = {}
                for j in range(features.shape[1]):
                    if i != j:
                        correlations[f'feature_{i}'][f'feature_{j}'] = correlation_matrix[i, j]
        
        return correlations
    
    def _create_feature_names(self, n_features: int) -> List[str]:
        """Create feature names."""
        feature_names = []
        
        # Basic features
        feature_names.extend(['returns', 'log_returns', 'price_ratios', 'volatility'])
        
        # Technical indicators
        if self.config.enable_technical_indicators:
            feature_names.extend(self.config.technical_indicators)
        
        # Regime-aware features
        if self.config.enable_regime_awareness:
            feature_names.extend(['regime_duration', 'regime_frequency', 'regime_consistency'])
        
        # Financial features
        if self.config.enable_financial_features:
            feature_names.extend(['volatility_features', 'momentum_features', 'mean_reversion_features', 'risk_features'])
        
        # Pad with generic names if needed
        while len(feature_names) < n_features:
            feature_names.append(f'feature_{len(feature_names)}')
        
        return feature_names[:n_features]
    
    def _create_error_result(self, error_message: str, processing_time: float) -> FeatureIntegrationResult:
        """Create error result."""
        return FeatureIntegrationResult(
            processed_features=np.array([]),
            feature_names=[],
            regime_labels=np.array([]),
            regime_probabilities=np.array([]),
            feature_importance={},
            feature_correlations={},
            regime_analysis={'error': error_message},
            feature_validation={},
            processing_time=processing_time,
            n_features=0,
            n_regimes=0
        )

def create_feature_engine_integrator(config: FeatureEngineIntegrationConfig) -> FeatureEngineIntegrator:
    """Create feature engine integrator instance."""
    return FeatureEngineIntegrator(config)

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
