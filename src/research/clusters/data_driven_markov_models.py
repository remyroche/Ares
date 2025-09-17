"""
Data-Driven Advanced Markov Models

This module implements sophisticated Markov modeling techniques that are
entirely data-driven, without imposing economic constraints or assumptions.
The models learn regime characteristics and durations directly from the data.

Key Features:
1. Markov-Switching Models with automatic structural break detection
2. Hidden Semi-Markov Models with self-determined duration distributions
3. Data-driven parameter estimation without economic priors
4. Enhanced forecasting capabilities during regime transitions
5. Adaptive regime boundary detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from src.utils.logger import system_logger

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn not available - some functionality limited")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures not available - structural break detection limited")


class DataDrivenRegimeType(Enum):
    """Data-determined regime types (no economic assumptions)."""
    HIGH_RETURN_LOW_VOL = "high_return_low_vol"
    HIGH_RETURN_HIGH_VOL = "high_return_high_vol"
    LOW_RETURN_LOW_VOL = "low_return_low_vol"
    LOW_RETURN_HIGH_VOL = "low_return_high_vol"
    EXTREME_VOLATILITY = "extreme_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"


class DurationLearningMethod(Enum):
    """Methods for learning duration distributions from data."""
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    BAYESIAN_INFERENCE = "bayesian_inference"
    KERNEL_DENSITY = "kernel_density"
    EMPIRICAL_DISTRIBUTION = "empirical_distribution"
    ADAPTIVE_MIXTURE = "adaptive_mixture"


@dataclass
class DataDrivenMSMConfig:
    """Configuration for data-driven Markov-Switching Models."""
    n_regimes: int = 3
    
    # Structural break detection
    enable_break_detection: bool = True
    break_detection_method: str = "pelt"  # "pelt", "binseg", "window"
    min_segment_length: int = 50  # Minimum observations per regime
    break_penalty: str = "bic"  # "bic", "aic", "hannan_quinn"
    
    # Data-driven regime identification
    regime_features: List[str] = field(default_factory=lambda: [
        'returns', 'volatility', 'momentum', 'volume_activity'
    ])
    
    # Model flexibility
    allow_regime_merging: bool = True
    allow_regime_splitting: bool = True
    adaptive_n_regimes: bool = True
    max_regimes: int = 8
    
    # Forecasting parameters
    forecast_horizon: int = 20
    transition_prediction: bool = True
    uncertainty_quantification: bool = True


@dataclass
class DataDrivenHSMMConfig:
    """Configuration for data-driven Hidden Semi-Markov Models."""
    n_states: int = 4
    
    # Duration learning
    duration_learning_method: DurationLearningMethod = DurationLearningMethod.MAXIMUM_LIKELIHOOD
    learn_duration_from_data: bool = True
    duration_distribution_candidates: List[str] = field(default_factory=lambda: [
        'gamma', 'weibull', 'lognormal', 'negative_binomial', 'poisson'
    ])
    
    # Adaptive duration modeling
    adaptive_durations: bool = True
    duration_model_selection: str = "aic"  # "aic", "bic", "cross_validation"
    
    # State identification
    state_clustering_method: str = "gmm"  # "kmeans", "gmm", "spectral"
    automatic_state_number: bool = True
    max_states: int = 10
    
    # Transition detection
    enhanced_transition_detection: bool = True
    transition_sensitivity: float = 0.8  # 0-1, higher = more sensitive


class StructuralBreakDetector:
    """Automatic structural break detection for Markov-Switching models."""
    
    def __init__(self, method: str = "pelt", penalty: str = "bic"):
        self.method = method
        self.penalty = penalty
        self.logger = system_logger.getChild('StructuralBreakDetector')
        
    def detect_breaks(self, data: np.ndarray, min_size: int = 50) -> List[int]:
        """
        Detect structural breaks in time series data.
        
        Args:
            data: Time series data
            min_size: Minimum segment size
            
        Returns:
            List of break points (indices)
        """
        if not RUPTURES_AVAILABLE:
            return self._fallback_break_detection(data, min_size)
        
        try:
            # Use ruptures library for sophisticated break detection
            if self.method == "pelt":
                algo = rpt.Pelt(model="rbf", min_size=min_size).fit(data.reshape(-1, 1))
            elif self.method == "binseg":
                algo = rpt.Binseg(model="rbf", min_size=min_size).fit(data.reshape(-1, 1))
            else:  # window
                algo = rpt.Window(width=min_size*2, model="rbf").fit(data.reshape(-1, 1))
            
            # Detect breaks with automatic penalty selection
            if self.penalty == "bic":
                penalty_value = np.log(len(data)) * data.shape[0] if len(data.shape) > 1 else np.log(len(data))
            elif self.penalty == "aic":
                penalty_value = 2 * data.shape[0] if len(data.shape) > 1 else 2
            else:  # hannan_quinn
                penalty_value = 2 * np.log(np.log(len(data))) * (data.shape[0] if len(data.shape) > 1 else 1)
            
            breaks = algo.predict(pen=penalty_value)
            
            # Remove the last break (end of series)
            if breaks and breaks[-1] == len(data):
                breaks = breaks[:-1]
            
            self.logger.info(f"🔍 Detected {len(breaks)} structural breaks using {self.method}")
            return breaks
            
        except Exception as e:
            self.logger.warning(f"Ruptures break detection failed: {e}, using fallback")
            return self._fallback_break_detection(data, min_size)
    
    def _fallback_break_detection(self, data: np.ndarray, min_size: int) -> List[int]:
        """Fallback break detection using variance change detection."""
        breaks = []
        
        # Simple variance-based break detection
        window_size = max(min_size, len(data) // 20)  # Adaptive window size
        
        for i in range(window_size, len(data) - window_size, window_size // 2):
            # Calculate variance before and after potential break point
            var_before = np.var(data[max(0, i - window_size):i])
            var_after = np.var(data[i:min(len(data), i + window_size)])
            
            # Detect significant variance change
            if var_after > 0 and var_before > 0:
                variance_ratio = max(var_after, var_before) / min(var_after, var_before)
                if variance_ratio > 2.0:  # Significant change threshold
                    breaks.append(i)
        
        # Remove breaks that are too close to each other
        if breaks:
            filtered_breaks = [breaks[0]]
            for break_point in breaks[1:]:
                if break_point - filtered_breaks[-1] > min_size:
                    filtered_breaks.append(break_point)
            breaks = filtered_breaks
        
        self.logger.info(f"🔍 Fallback method detected {len(breaks)} structural breaks")
        return breaks


class DataDrivenMarkovSwitchingModel:
    """
    Data-driven Markov-Switching Model with automatic structural break detection.
    
    This model learns regime characteristics entirely from data without
    imposing economic assumptions or constraints.
    """
    
    def __init__(self, config: DataDrivenMSMConfig):
        self.config = config
        self.logger = system_logger.getChild('DataDrivenMSM')
        
        # Model components
        self.break_detector = StructuralBreakDetector(
            method=config.break_detection_method,
            penalty=config.break_penalty
        )
        self.regime_models: Dict[int, Dict[str, Any]] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.regime_assignments: Optional[np.ndarray] = None
        
        # Forecasting components
        self.transition_predictor: Optional[Any] = None
        self.regime_forecasts: Dict[int, Dict[str, Any]] = {}
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit data-driven Markov-Switching model.
        
        Args:
            data: Market data
            
        Returns:
            Comprehensive fitting results
        """
        self.logger.info("🔬 Fitting Data-Driven Markov-Switching Model")
        
        try:
            # Step 1: Prepare features for regime detection
            features = self._prepare_regime_features(data)
            
            # Step 2: Detect structural breaks automatically
            if self.config.enable_break_detection:
                break_points = self._detect_structural_breaks(features)
            else:
                break_points = []
            
            # Step 3: Initial regime detection
            initial_regimes = self._detect_initial_regimes(features, break_points)
            
            # Step 4: Adaptive regime optimization
            if self.config.adaptive_n_regimes:
                optimized_regimes = self._optimize_regime_number(features, initial_regimes)
            else:
                optimized_regimes = initial_regimes
            
            # Step 5: Learn regime-specific models
            regime_models = self._learn_regime_models(data, features, optimized_regimes)
            
            # Step 6: Estimate transition dynamics
            transition_matrix = self._estimate_transition_matrix(optimized_regimes)
            
            # Step 7: Build forecasting models
            if self.config.transition_prediction:
                forecasting_models = self._build_forecasting_models(data, optimized_regimes)
            else:
                forecasting_models = {}
            
            # Store results
            self.regime_models = regime_models
            self.transition_matrix = transition_matrix
            self.regime_assignments = optimized_regimes
            self.regime_forecasts = forecasting_models
            
            # Generate comprehensive results
            results = {
                'regime_assignments': optimized_regimes,
                'n_regimes': len(np.unique(optimized_regimes)),
                'structural_breaks': break_points,
                'regime_models': regime_models,
                'transition_matrix': transition_matrix,
                'regime_characteristics': self._analyze_regime_characteristics(data, optimized_regimes),
                'forecasting_models': forecasting_models,
                'model_selection_info': self._get_model_selection_info(features, optimized_regimes)
            }
            
            self.logger.info(f"✅ MSM fitted: {results['n_regimes']} regimes, {len(break_points)} breaks detected")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Data-driven MSM fitting failed: {e}")
            raise
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection."""
        feature_data = pd.DataFrame()
        
        # Returns
        feature_data['returns'] = data['close'].pct_change()
        
        # Volatility (multiple windows)
        for window in [5, 10, 20]:
            feature_data[f'volatility_{window}'] = feature_data['returns'].rolling(window).std()
        
        # Momentum indicators
        for window in [10, 20, 50]:
            feature_data[f'momentum_{window}'] = data['close'].pct_change(window)
        
        # Volume activity (if available)
        if 'volume' in data.columns:
            feature_data['volume_activity'] = (
                data['volume'] / data['volume'].rolling(20).mean()
            ).fillna(1.0)
        else:
            feature_data['volume_activity'] = np.ones(len(data))
        
        # Price-based features
        feature_data['price_acceleration'] = feature_data['returns'].diff()
        feature_data['volatility_clustering'] = (
            feature_data['volatility_20'] / feature_data['volatility_20'].rolling(50).mean()
        )
        
        # Technical indicators
        feature_data['rsi_like'] = self._calculate_rsi_like(feature_data['returns'])
        feature_data['trend_strength'] = self._calculate_trend_strength(data['close'])
        
        # Clean and standardize
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Select features based on config
        if self.config.regime_features:
            available_features = [f for f in self.config.regime_features if f in feature_data.columns]
            if available_features:
                feature_data = feature_data[available_features]
        
        # Standardize
        scaler = StandardScaler()
        return scaler.fit_transform(feature_data)
    
    def _calculate_rsi_like(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI-like momentum indicator."""
        gains = returns.where(returns > 0, 0).rolling(window).mean()
        losses = (-returns).where(returns < 0, 0).rolling(window).mean()
        rs = gains / (losses + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength indicator."""
        ma = prices.rolling(window).mean()
        trend = (prices - ma) / ma
        return trend.rolling(window).mean()
    
    def _detect_structural_breaks(self, features: np.ndarray) -> List[int]:
        """Detect structural breaks in feature space."""
        # Use first principal component for break detection
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(features).flatten()
        
        # Detect breaks in the principal component
        breaks = self.break_detector.detect_breaks(
            principal_component, 
            min_size=self.config.min_segment_length
        )
        
        return breaks
    
    def _detect_initial_regimes(self, features: np.ndarray, break_points: List[int]) -> np.ndarray:
        """Detect initial regimes using structural breaks and clustering."""
        n_obs = len(features)
        
        if break_points:
            # Use break points to define initial regimes
            regimes = np.zeros(n_obs, dtype=int)
            
            # Create segments from break points
            segments = [0] + break_points + [n_obs]
            
            for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
                regimes[start:end] = i % self.config.n_regimes
            
        else:
            # Use clustering for initial regime detection
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42, n_init=10)
            regimes = kmeans.fit_predict(features)
        
        return regimes
    
    def _optimize_regime_number(self, features: np.ndarray, initial_regimes: np.ndarray) -> np.ndarray:
        """Optimize the number of regimes using model selection criteria."""
        if not self.config.adaptive_n_regimes:
            return initial_regimes
        
        best_regimes = initial_regimes
        best_score = float('-inf')
        
        # Try different numbers of regimes
        for n_regimes in range(2, self.config.max_regimes + 1):
            try:
                # Fit GMM with n_regimes components
                gmm = GaussianMixture(n_components=n_regimes, random_state=42)
                regime_probs = gmm.fit_predict(features)
                
                # Calculate model selection criterion
                if self.config.break_penalty == "bic":
                    score = gmm.bic(features)
                elif self.config.break_penalty == "aic":
                    score = gmm.aic(features)
                else:
                    score = gmm.bic(features)  # Default to BIC
                
                # Lower is better for BIC/AIC
                score = -score
                
                if score > best_score:
                    best_score = score
                    best_regimes = regime_probs
                    
            except Exception as e:
                self.logger.warning(f"Failed to fit {n_regimes} regimes: {e}")
                continue
        
        self.logger.info(f"🎯 Optimal number of regimes: {len(np.unique(best_regimes))}")
        return best_regimes
    
    def _learn_regime_models(self, 
                           data: pd.DataFrame, 
                           features: np.ndarray, 
                           regimes: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Learn regime-specific models from data."""
        regime_models = {}
        
        for regime_id in np.unique(regimes):
            regime_mask = regimes == regime_id
            regime_data = data[regime_mask]
            regime_features = features[regime_mask]
            
            if len(regime_data) < 10:  # Skip regimes with too little data
                continue
            
            returns = regime_data['close'].pct_change().dropna()
            
            # Learn regime characteristics
            model = {
                'regime_id': int(regime_id),
                'n_observations': len(regime_data),
                'frequency': float(np.sum(regime_mask) / len(regimes)),
                
                # Return characteristics
                'mean_return': float(returns.mean()),
                'return_std': float(returns.std()),
                'return_skewness': float(returns.skew()),
                'return_kurtosis': float(returns.kurtosis()),
                
                # Volatility characteristics
                'volatility_mean': float(returns.rolling(20).std().mean()),
                'volatility_std': float(returns.rolling(20).std().std()),
                
                # Feature characteristics
                'feature_centroid': np.mean(regime_features, axis=0).tolist(),
                'feature_covariance': np.cov(regime_features.T).tolist(),
                
                # Regime type (data-driven classification)
                'regime_type': self._classify_regime_type(returns),
                
                # Persistence characteristics
                'avg_duration': self._calculate_average_duration(regime_mask),
                'transition_volatility': self._calculate_transition_volatility(regime_mask, returns)
            }
            
            regime_models[regime_id] = model
        
        return regime_models
    
    def _classify_regime_type(self, returns: pd.Series) -> str:
        """Classify regime type based on data characteristics."""
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Data-driven thresholds (no economic assumptions)
        high_return_threshold = returns.quantile(0.6)
        high_vol_threshold = returns.rolling(20).std().quantile(0.6)
        
        if mean_return > high_return_threshold:
            if volatility > high_vol_threshold:
                return DataDrivenRegimeType.HIGH_RETURN_HIGH_VOL.value
            else:
                return DataDrivenRegimeType.HIGH_RETURN_LOW_VOL.value
        else:
            if volatility > high_vol_threshold:
                return DataDrivenRegimeType.LOW_RETURN_HIGH_VOL.value
            else:
                return DataDrivenRegimeType.LOW_RETURN_LOW_VOL.value
    
    def _calculate_average_duration(self, regime_mask: np.ndarray) -> float:
        """Calculate average duration of regime episodes."""
        changes = np.diff(regime_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if regime_mask[0]:
            starts = np.concatenate([[0], starts])
        if regime_mask[-1]:
            ends = np.concatenate([ends, [len(regime_mask)]])
        
        if len(starts) == len(ends) and len(starts) > 0:
            durations = ends - starts
            return float(np.mean(durations))
        
        return float(np.sum(regime_mask))  # Fallback
    
    def _calculate_transition_volatility(self, regime_mask: np.ndarray, returns: pd.Series) -> float:
        """Calculate volatility around regime transitions."""
        changes = np.where(np.diff(regime_mask.astype(int)) != 0)[0]
        
        if len(changes) == 0:
            return 0.0
        
        transition_returns = []
        window = 5  # Look 5 periods around transitions
        
        for change_point in changes:
            start = max(0, change_point - window)
            end = min(len(returns), change_point + window + 1)
            transition_returns.extend(returns.iloc[start:end].tolist())
        
        return float(np.std(transition_returns)) if transition_returns else 0.0
    
    def _estimate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from regime sequence."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        
        # Map regime IDs to indices
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        
        # Count transitions
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for t in range(len(regimes) - 1):
            current_idx = regime_to_idx[regimes[t]]
            next_idx = regime_to_idx[regimes[t + 1]]
            transition_counts[current_idx, next_idx] += 1
        
        # Normalize to probabilities with smoothing
        smoothing = 0.01  # Small smoothing to avoid zero probabilities
        transition_matrix = transition_counts + smoothing
        
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def _build_forecasting_models(self, 
                                data: pd.DataFrame, 
                                regimes: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Build forecasting models for regime transitions and returns."""
        forecasting_models = {}
        
        for regime_id in np.unique(regimes):
            regime_mask = regimes == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) < 20:  # Need sufficient data for forecasting
                continue
            
            returns = regime_data['close'].pct_change().dropna()
            
            # Simple forecasting model for this regime
            forecasting_models[regime_id] = {
                'return_forecast': {
                    'mean': float(returns.mean()),
                    'std': float(returns.std()),
                    'distribution': 'normal'  # Could be enhanced with distribution fitting
                },
                'volatility_forecast': {
                    'mean': float(returns.rolling(10).std().mean()),
                    'persistence': float(returns.rolling(10).std().autocorr(lag=1))
                },
                'regime_persistence': {
                    'avg_duration': self._calculate_average_duration(regime_mask),
                    'exit_probability': 1.0 - self.transition_matrix[regime_id, regime_id] if hasattr(self, 'transition_matrix') else 0.1
                }
            }
        
        return forecasting_models
    
    def _analyze_regime_characteristics(self, 
                                     data: pd.DataFrame, 
                                     regimes: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of detected regimes."""
        characteristics = {}
        
        for regime_id in np.unique(regimes):
            regime_mask = regimes == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            returns = regime_data['close'].pct_change().dropna()
            
            characteristics[regime_id] = {
                'data_driven_metrics': {
                    'frequency': float(np.sum(regime_mask) / len(regimes)),
                    'avg_duration': self._calculate_average_duration(regime_mask),
                    'return_mean': float(returns.mean()),
                    'return_std': float(returns.std()),
                    'return_sharpe': float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
                    'max_drawdown': float(self._calculate_max_drawdown(regime_data['close'])),
                    'regime_type': self._classify_regime_type(returns)
                },
                'statistical_properties': {
                    'return_skewness': float(returns.skew()),
                    'return_kurtosis': float(returns.kurtosis()),
                    'jarque_bera_pvalue': float(stats.jarque_bera(returns)[1]) if len(returns) > 8 else 1.0,
                    'autocorrelation_lag1': float(returns.autocorr(lag=1)) if len(returns) > 1 else 0.0
                }
            }
        
        return characteristics
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _get_model_selection_info(self, features: np.ndarray, regimes: np.ndarray) -> Dict[str, Any]:
        """Get model selection information."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        n_obs, n_features = features.shape
        
        # Calculate information criteria
        # Simple approximation - could be enhanced with proper likelihood calculation
        within_cluster_variance = 0.0
        for regime_id in unique_regimes:
            regime_features = features[regimes == regime_id]
            if len(regime_features) > 1:
                within_cluster_variance += np.trace(np.cov(regime_features.T))
        
        # Rough approximation of log-likelihood
        log_likelihood = -0.5 * n_obs * np.log(2 * np.pi * within_cluster_variance / n_obs)
        
        # Number of parameters (rough estimate)
        n_params = n_regimes * (n_features + n_features * (n_features + 1) // 2) + n_regimes * (n_regimes - 1)
        
        return {
            'n_regimes': n_regimes,
            'n_parameters': n_params,
            'log_likelihood': float(log_likelihood),
            'aic': float(2 * n_params - 2 * log_likelihood),
            'bic': float(np.log(n_obs) * n_params - 2 * log_likelihood),
            'within_cluster_variance': float(within_cluster_variance)
        }
    
    def predict_regime_transitions(self, 
                                 current_regime: int, 
                                 horizon: int = 20) -> Dict[str, Any]:
        """Predict regime transitions over specified horizon."""
        if self.transition_matrix is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Multi-step transition probabilities
        transition_probs = np.linalg.matrix_power(self.transition_matrix, horizon)
        
        current_regime_idx = current_regime
        future_probs = transition_probs[current_regime_idx]
        
        return {
            'horizon': horizon,
            'current_regime': current_regime,
            'regime_probabilities': future_probs.tolist(),
            'most_likely_regime': int(np.argmax(future_probs)),
            'regime_uncertainty': float(1.0 - np.max(future_probs))
        }


class DataDrivenHiddenSemiMarkovModel:
    """
    Data-driven Hidden Semi-Markov Model with self-determined duration distributions.
    
    This model learns state durations and characteristics entirely from data
    without imposing economic constraints.
    """
    
    def __init__(self, config: DataDrivenHSMMConfig):
        self.config = config
        self.logger = system_logger.getChild('DataDrivenHSMM')
        
        # Model components
        self.duration_models: Dict[int, Dict[str, Any]] = {}
        self.emission_models: Dict[int, Dict[str, Any]] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.state_sequence: Optional[np.ndarray] = None
        
        # Duration learning components
        self.duration_distributions: Dict[int, str] = {}
        self.duration_parameters: Dict[int, Dict[str, float]] = {}
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit data-driven Hidden Semi-Markov Model.
        
        Args:
            data: Market data
            
        Returns:
            Comprehensive fitting results
        """
        self.logger.info("🔬 Fitting Data-Driven Hidden Semi-Markov Model")
        
        try:
            # Step 1: Prepare observations
            observations = self._prepare_observations(data)
            
            # Step 2: Initial state detection
            initial_states = self._detect_initial_states(observations)
            
            # Step 3: Optimize number of states if requested
            if self.config.automatic_state_number:
                optimized_states = self._optimize_state_number(observations, initial_states)
            else:
                optimized_states = initial_states
            
            # Step 4: Learn duration distributions from data
            duration_models = self._learn_duration_distributions(optimized_states)
            
            # Step 5: Learn emission models
            emission_models = self._learn_emission_models(observations, optimized_states)
            
            # Step 6: Estimate transition matrix (no self-transitions in HSMM)
            transition_matrix = self._estimate_hsmm_transition_matrix(optimized_states)
            
            # Step 7: Refine state sequence using learned models
            if self.config.enhanced_transition_detection:
                refined_states = self._refine_state_sequence(observations, optimized_states)
            else:
                refined_states = optimized_states
            
            # Store results
            self.duration_models = duration_models
            self.emission_models = emission_models
            self.transition_matrix = transition_matrix
            self.state_sequence = refined_states
            
            # Generate comprehensive results
            results = {
                'state_sequence': refined_states,
                'n_states': len(np.unique(refined_states)),
                'duration_models': duration_models,
                'emission_models': emission_models,
                'transition_matrix': transition_matrix,
                'state_characteristics': self._analyze_state_characteristics(data, refined_states),
                'duration_analysis': self._analyze_duration_patterns(refined_states),
                'transition_detection': self._analyze_transition_quality(refined_states),
                'model_diagnostics': self._get_model_diagnostics(observations, refined_states)
            }
            
            self.logger.info(f"✅ HSMM fitted: {results['n_states']} states with data-driven durations")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Data-driven HSMM fitting failed: {e}")
            raise
    
    def _prepare_observations(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare multi-dimensional observations for HSMM."""
        obs_data = pd.DataFrame()
        
        # Core market features
        obs_data['returns'] = data['close'].pct_change()
        obs_data['log_returns'] = np.log(data['close']).diff()
        
        # Volatility features (multiple windows)
        for window in [5, 10, 20]:
            obs_data[f'vol_{window}'] = obs_data['returns'].rolling(window).std()
        
        # Price momentum
        for window in [5, 10, 20]:
            obs_data[f'momentum_{window}'] = data['close'].pct_change(window)
        
        # Volume (if available)
        if 'volume' in data.columns:
            obs_data['volume_norm'] = np.log(data['volume'] / data['volume'].rolling(20).mean() + 1e-8)
        else:
            obs_data['volume_norm'] = np.zeros(len(data))
        
        # Technical features
        obs_data['price_position'] = (data['close'] - data['close'].rolling(50).min()) / (
            data['close'].rolling(50).max() - data['close'].rolling(50).min() + 1e-8
        )
        
        # Clean and standardize
        obs_data = obs_data.fillna(method='ffill').fillna(0)
        
        # Remove extreme outliers
        for col in obs_data.columns:
            q99 = obs_data[col].quantile(0.99)
            q01 = obs_data[col].quantile(0.01)
            obs_data[col] = obs_data[col].clip(q01, q99)
        
        # Standardize
        scaler = StandardScaler()
        return scaler.fit_transform(obs_data)
    
    def _detect_initial_states(self, observations: np.ndarray) -> np.ndarray:
        """Detect initial states using clustering."""
        if self.config.state_clustering_method == "gmm":
            clusterer = GaussianMixture(n_components=self.config.n_states, random_state=42)
        elif self.config.state_clustering_method == "spectral":
            from sklearn.cluster import SpectralClustering
            clusterer = SpectralClustering(n_clusters=self.config.n_states, random_state=42)
        else:  # kmeans
            clusterer = KMeans(n_clusters=self.config.n_states, random_state=42, n_init=10)
        
        initial_states = clusterer.fit_predict(observations)
        
        self.logger.info(f"📊 Initial state distribution: {np.bincount(initial_states)}")
        return initial_states
    
    def _optimize_state_number(self, observations: np.ndarray, initial_states: np.ndarray) -> np.ndarray:
        """Optimize number of states using model selection."""
        best_states = initial_states
        best_score = float('inf')  # Lower is better for AIC/BIC
        
        for n_states in range(2, self.config.max_states + 1):
            try:
                # Fit GMM with n_states
                gmm = GaussianMixture(n_components=n_states, random_state=42)
                states = gmm.fit_predict(observations)
                
                # Calculate selection criterion
                if self.config.duration_model_selection == "aic":
                    score = gmm.aic(observations)
                elif self.config.duration_model_selection == "bic":
                    score = gmm.bic(observations)
                else:  # cross_validation
                    score = gmm.bic(observations)  # Fallback to BIC
                
                if score < best_score:
                    best_score = score
                    best_states = states
                    
            except Exception as e:
                self.logger.warning(f"Failed to fit {n_states} states: {e}")
                continue
        
        self.logger.info(f"🎯 Optimal number of states: {len(np.unique(best_states))}")
        return best_states
    
    def _learn_duration_distributions(self, states: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Learn duration distributions for each state from data."""
        duration_models = {}
        
        for state in np.unique(states):
            # Extract durations for this state
            durations = self._extract_state_durations(states, state)
            
            if len(durations) < 3:  # Need at least 3 episodes
                continue
            
            # Find best duration distribution
            best_distribution = self._fit_best_duration_distribution(durations)
            
            duration_models[state] = {
                'empirical_durations': durations.tolist(),
                'n_episodes': len(durations),
                'mean_duration': float(np.mean(durations)),
                'median_duration': float(np.median(durations)),
                'std_duration': float(np.std(durations)),
                'min_duration': int(np.min(durations)),
                'max_duration': int(np.max(durations)),
                'best_distribution': best_distribution['distribution'],
                'distribution_parameters': best_distribution['parameters'],
                'distribution_fit_quality': best_distribution['fit_quality']
            }
        
        return duration_models
    
    def _extract_state_durations(self, states: np.ndarray, target_state: int) -> np.ndarray:
        """Extract durations of episodes for a specific state."""
        state_mask = states == target_state
        
        # Find state changes
        changes = np.diff(state_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if state_mask[0]:
            starts = np.concatenate([[0], starts])
        if state_mask[-1]:
            ends = np.concatenate([ends, [len(states)]])
        
        # Calculate durations
        if len(starts) == len(ends):
            durations = ends - starts
            return durations[durations > 0]  # Only positive durations
        
        return np.array([])
    
    def _fit_best_duration_distribution(self, durations: np.ndarray) -> Dict[str, Any]:
        """Fit best duration distribution to observed durations."""
        candidates = self.config.duration_distribution_candidates
        best_fit = None
        best_score = float('inf')
        
        for dist_name in candidates:
            try:
                if dist_name == 'gamma':
                    # Fit gamma distribution
                    shape, loc, scale = stats.gamma.fit(durations, floc=0)
                    params = {'shape': shape, 'scale': scale}
                    log_likelihood = np.sum(stats.gamma.logpdf(durations, shape, loc=0, scale=scale))
                    
                elif dist_name == 'weibull':
                    # Fit Weibull distribution (using Weibull_min from scipy)
                    c, loc, scale = stats.weibull_min.fit(durations, floc=0)
                    params = {'shape': c, 'scale': scale}
                    log_likelihood = np.sum(stats.weibull_min.logpdf(durations, c, loc=0, scale=scale))
                    
                elif dist_name == 'lognormal':
                    # Fit log-normal distribution
                    s, loc, scale = stats.lognorm.fit(durations, floc=0)
                    params = {'s': s, 'scale': scale}
                    log_likelihood = np.sum(stats.lognorm.logpdf(durations, s, loc=0, scale=scale))
                    
                elif dist_name == 'negative_binomial':
                    # Fit negative binomial (discrete)
                    # Convert to discrete values
                    discrete_durations = durations.astype(int)
                    mean_dur = np.mean(discrete_durations)
                    var_dur = np.var(discrete_durations)
                    
                    if var_dur > mean_dur:  # Overdispersed
                        p = mean_dur / var_dur
                        n = mean_dur * p / (1 - p)
                        params = {'n': max(1, n), 'p': min(0.99, max(0.01, p))}
                        log_likelihood = np.sum(stats.nbinom.logpmf(discrete_durations, params['n'], params['p']))
                    else:
                        continue  # Skip if not overdispersed
                    
                elif dist_name == 'poisson':
                    # Fit Poisson distribution
                    mu = np.mean(durations)
                    params = {'mu': mu}
                    log_likelihood = np.sum(stats.poisson.logpmf(durations.astype(int), mu))
                    
                else:
                    continue
                
                # Calculate AIC
                n_params = len(params)
                aic = 2 * n_params - 2 * log_likelihood
                
                if aic < best_score:
                    best_score = aic
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'log_likelihood': log_likelihood,
                        'aic': aic,
                        'fit_quality': 'good' if log_likelihood > -1000 else 'poor'
                    }
                    
            except Exception as e:
                self.logger.debug(f"Failed to fit {dist_name} distribution: {e}")
                continue
        
        # Fallback to empirical distribution if no parametric fit works
        if best_fit is None:
            best_fit = {
                'distribution': 'empirical',
                'parameters': {'values': durations.tolist()},
                'log_likelihood': 0.0,
                'aic': float('inf'),
                'fit_quality': 'empirical'
            }
        
        return best_fit
    
    def _learn_emission_models(self, observations: np.ndarray, states: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Learn emission models for each state."""
        emission_models = {}
        
        for state in np.unique(states):
            state_observations = observations[states == state]
            
            if len(state_observations) < 5:  # Need sufficient data
                continue
            
            # Fit multivariate Gaussian emission model
            mean = np.mean(state_observations, axis=0)
            cov = np.cov(state_observations.T)
            
            # Regularize covariance matrix
            cov = cov + 1e-6 * np.eye(cov.shape[0])
            
            emission_models[state] = {
                'type': 'multivariate_gaussian',
                'mean': mean.tolist(),
                'covariance': cov.tolist(),
                'n_observations': len(state_observations),
                'log_likelihood': float(stats.multivariate_normal.logpdf(state_observations, mean, cov).sum())
            }
        
        return emission_models
    
    def _estimate_hsmm_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Estimate transition matrix for HSMM (no self-transitions)."""
        unique_states = np.unique(states)
        n_states = len(unique_states)
        
        # Map state IDs to indices
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
        
        # Count transitions (excluding self-transitions)
        transition_counts = np.zeros((n_states, n_states))
        
        prev_state = states[0]
        for t in range(1, len(states)):
            current_state = states[t]
            if current_state != prev_state:  # Only count actual state changes
                prev_idx = state_to_idx[prev_state]
                current_idx = state_to_idx[current_state]
                transition_counts[prev_idx, current_idx] += 1
                prev_state = current_state
        
        # Set diagonal to zero (no self-transitions in HSMM)
        np.fill_diagonal(transition_counts, 0)
        
        # Normalize with smoothing
        smoothing = 0.01
        transition_matrix = transition_counts + smoothing
        
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def _refine_state_sequence(self, observations: np.ndarray, initial_states: np.ndarray) -> np.ndarray:
        """Refine state sequence using enhanced transition detection."""
        if not self.config.enhanced_transition_detection:
            return initial_states
        
        refined_states = initial_states.copy()
        
        # Apply duration-based smoothing
        for state in np.unique(initial_states):
            durations = self._extract_state_durations(initial_states, state)
            if len(durations) > 0:
                min_duration = max(1, int(np.percentile(durations, 25)))  # 25th percentile as min
                
                # Merge short episodes
                refined_states = self._merge_short_episodes(refined_states, state, min_duration)
        
        return refined_states
    
    def _merge_short_episodes(self, states: np.ndarray, target_state: int, min_duration: int) -> np.ndarray:
        """Merge episodes shorter than minimum duration."""
        result = states.copy()
        
        # Find episodes of target state
        state_mask = states == target_state
        changes = np.diff(state_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if state_mask[0]:
            starts = np.concatenate([[0], starts])
        if state_mask[-1]:
            ends = np.concatenate([ends, [len(states)]])
        
        # Merge short episodes
        if len(starts) == len(ends):
            for start, end in zip(starts, ends):
                if end - start < min_duration:
                    # Find most common neighboring state
                    neighbors = []
                    if start > 0:
                        neighbors.append(states[start - 1])
                    if end < len(states):
                        neighbors.append(states[end])
                    
                    if neighbors:
                        replacement_state = max(set(neighbors), key=neighbors.count)
                        result[start:end] = replacement_state
        
        return result
    
    def _analyze_state_characteristics(self, data: pd.DataFrame, states: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of detected states."""
        characteristics = {}
        
        for state in np.unique(states):
            state_mask = states == state
            state_data = data[state_mask]
            
            if len(state_data) == 0:
                continue
            
            returns = state_data['close'].pct_change().dropna()
            durations = self._extract_state_durations(states, state)
            
            characteristics[state] = {
                'basic_stats': {
                    'frequency': float(np.sum(state_mask) / len(states)),
                    'n_episodes': len(durations),
                    'total_observations': len(state_data),
                    'mean_return': float(returns.mean()) if len(returns) > 0 else 0.0,
                    'return_volatility': float(returns.std()) if len(returns) > 0 else 0.0,
                    'sharpe_ratio': float(returns.mean() / returns.std()) if len(returns) > 0 and returns.std() > 0 else 0.0
                },
                'duration_stats': {
                    'mean_duration': float(np.mean(durations)) if len(durations) > 0 else 0.0,
                    'median_duration': float(np.median(durations)) if len(durations) > 0 else 0.0,
                    'duration_variability': float(np.std(durations) / np.mean(durations)) if len(durations) > 0 and np.mean(durations) > 0 else 0.0,
                    'min_duration': int(np.min(durations)) if len(durations) > 0 else 0,
                    'max_duration': int(np.max(durations)) if len(durations) > 0 else 0
                },
                'transition_stats': {
                    'avg_entry_volatility': self._calculate_entry_exit_volatility(states, state, data, 'entry'),
                    'avg_exit_volatility': self._calculate_entry_exit_volatility(states, state, data, 'exit')
                }
            }
        
        return characteristics
    
    def _calculate_entry_exit_volatility(self, states: np.ndarray, target_state: int, 
                                       data: pd.DataFrame, mode: str) -> float:
        """Calculate volatility around state entries or exits."""
        state_mask = states == target_state
        changes = np.diff(state_mask.astype(int))
        
        if mode == 'entry':
            transition_points = np.where(changes == 1)[0] + 1
        else:  # exit
            transition_points = np.where(changes == -1)[0] + 1
        
        if len(transition_points) == 0:
            return 0.0
        
        returns = data['close'].pct_change()
        window = 5  # Look 5 periods around transitions
        
        transition_volatilities = []
        for point in transition_points:
            start = max(0, point - window)
            end = min(len(returns), point + window + 1)
            period_returns = returns.iloc[start:end].dropna()
            
            if len(period_returns) > 1:
                transition_volatilities.append(period_returns.std())
        
        return float(np.mean(transition_volatilities)) if transition_volatilities else 0.0
    
    def _analyze_duration_patterns(self, states: np.ndarray) -> Dict[str, Any]:
        """Analyze duration patterns across all states."""
        all_durations = []
        state_duration_stats = {}
        
        for state in np.unique(states):
            durations = self._extract_state_durations(states, state)
            if len(durations) > 0:
                all_durations.extend(durations.tolist())
                state_duration_stats[state] = {
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations)),
                    'cv': float(np.std(durations) / np.mean(durations)) if np.mean(durations) > 0 else 0.0
                }
        
        return {
            'overall_duration_stats': {
                'mean_duration': float(np.mean(all_durations)) if all_durations else 0.0,
                'median_duration': float(np.median(all_durations)) if all_durations else 0.0,
                'duration_range': float(np.max(all_durations) - np.min(all_durations)) if all_durations else 0.0,
                'duration_variability': float(np.std(all_durations) / np.mean(all_durations)) if all_durations and np.mean(all_durations) > 0 else 0.0
            },
            'state_duration_comparison': state_duration_stats,
            'duration_distribution_shape': {
                'skewness': float(stats.skew(all_durations)) if len(all_durations) > 2 else 0.0,
                'kurtosis': float(stats.kurtosis(all_durations)) if len(all_durations) > 2 else 0.0
            }
        }
    
    def _analyze_transition_quality(self, states: np.ndarray) -> Dict[str, Any]:
        """Analyze quality of transition detection."""
        # Count transitions
        total_transitions = np.sum(np.diff(states) != 0)
        
        # Calculate transition frequency
        transition_rate = total_transitions / len(states)
        
        # Analyze transition patterns
        transition_intervals = []
        last_transition = 0
        
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                interval = i - last_transition
                transition_intervals.append(interval)
                last_transition = i
        
        return {
            'transition_statistics': {
                'total_transitions': int(total_transitions),
                'transition_rate': float(transition_rate),
                'avg_interval_between_transitions': float(np.mean(transition_intervals)) if transition_intervals else float('inf'),
                'transition_regularity': float(1.0 / (1.0 + np.std(transition_intervals))) if len(transition_intervals) > 1 else 1.0
            },
            'transition_quality_assessment': 'high' if transition_rate < 0.1 else 'medium' if transition_rate < 0.2 else 'low'
        }
    
    def _get_model_diagnostics(self, observations: np.ndarray, states: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive model diagnostics."""
        unique_states = np.unique(states)
        n_states = len(unique_states)
        n_obs, n_features = observations.shape
        
        # State assignment quality
        state_counts = np.bincount(states)
        state_balance = np.std(state_counts) / np.mean(state_counts) if np.mean(state_counts) > 0 else float('inf')
        
        # Within-state variance
        total_within_variance = 0.0
        for state in unique_states:
            state_obs = observations[states == state]
            if len(state_obs) > 1:
                total_within_variance += np.trace(np.cov(state_obs.T))
        
        return {
            'model_complexity': {
                'n_states': n_states,
                'n_observations': n_obs,
                'n_features': n_features,
                'observations_per_state': float(n_obs / n_states)
            },
            'state_quality': {
                'state_balance': float(state_balance),
                'min_state_size': int(np.min(state_counts)),
                'max_state_size': int(np.max(state_counts)),
                'within_state_variance': float(total_within_variance)
            },
            'model_fit_assessment': 'good' if state_balance < 1.0 and np.min(state_counts) > 10 else 'needs_tuning'
        }


# Integration class for both models
class DataDrivenAdvancedMarkovIntegration:
    """Integration of data-driven advanced Markov models."""
    
    def __init__(self):
        self.logger = system_logger.getChild('DataDrivenAdvancedMarkov')
        self.msm_model = None
        self.hsmm_model = None
    
    def run_data_driven_analysis(self, 
                                data: pd.DataFrame,
                                include_msm: bool = True,
                                include_hsmm: bool = True) -> Dict[str, Any]:
        """Run comprehensive data-driven advanced Markov analysis."""
        
        self.logger.info("🔬 Starting Data-Driven Advanced Markov Analysis")
        
        results = {
            'analysis_type': 'data_driven_advanced_markov',
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': data.shape,
            'models_executed': []
        }
        
        # Run Markov-Switching Model
        if include_msm:
            try:
                self.logger.info("📊 Running Data-Driven Markov-Switching Model")
                msm_config = DataDrivenMSMConfig(
                    enable_break_detection=True,
                    adaptive_n_regimes=True,
                    transition_prediction=True
                )
                
                self.msm_model = DataDrivenMarkovSwitchingModel(msm_config)
                msm_results = self.msm_model.fit(data)
                
                results['markov_switching'] = msm_results
                results['models_executed'].append('data_driven_msm')
                
                self.logger.info(f"✅ MSM: {msm_results['n_regimes']} regimes, {len(msm_results['structural_breaks'])} breaks")
                
            except Exception as e:
                self.logger.error(f"❌ Data-driven MSM failed: {e}")
                results['markov_switching_error'] = str(e)
        
        # Run Hidden Semi-Markov Model
        if include_hsmm:
            try:
                self.logger.info("📊 Running Data-Driven Hidden Semi-Markov Model")
                hsmm_config = DataDrivenHSMMConfig(
                    learn_duration_from_data=True,
                    adaptive_durations=True,
                    automatic_state_number=True,
                    enhanced_transition_detection=True
                )
                
                self.hsmm_model = DataDrivenHiddenSemiMarkovModel(hsmm_config)
                hsmm_results = self.hsmm_model.fit(data)
                
                results['hidden_semi_markov'] = hsmm_results
                results['models_executed'].append('data_driven_hsmm')
                
                self.logger.info(f"✅ HSMM: {hsmm_results['n_states']} states with learned durations")
                
            except Exception as e:
                self.logger.error(f"❌ Data-driven HSMM failed: {e}")
                results['hidden_semi_markov_error'] = str(e)
        
        # Generate comparative analysis
        if len(results['models_executed']) > 1:
            results['comparative_analysis'] = self._compare_data_driven_models(results)
        
        # Generate data-driven insights
        results['insights'] = self._generate_data_driven_insights(results)
        
        self.logger.info(f"✅ Data-driven analysis completed: {len(results['models_executed'])} models")
        return results
    
    def _compare_data_driven_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from data-driven models."""
        comparison = {}
        
        if 'markov_switching' in results and 'hidden_semi_markov' in results:
            msm_regimes = results['markov_switching']['regime_assignments']
            hsmm_states = results['hidden_semi_markov']['state_sequence']
            
            # Calculate agreement
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            comparison['model_agreement'] = {
                'adjusted_rand_score': float(adjusted_rand_score(msm_regimes, hsmm_states)),
                'normalized_mutual_info': float(normalized_mutual_info_score(msm_regimes, hsmm_states)),
                'n_regimes_msm': len(np.unique(msm_regimes)),
                'n_states_hsmm': len(np.unique(hsmm_states))
            }
            
            # Compare duration characteristics
            msm_regimes_array = np.array(msm_regimes)
            hsmm_states_array = np.array(hsmm_states)
            
            msm_durations = []
            hsmm_durations = []
            
            # Calculate average durations for MSM
            for regime in np.unique(msm_regimes_array):
                regime_mask = msm_regimes_array == regime
                changes = np.diff(regime_mask.astype(int))
                starts = np.where(changes == 1)[0] + 1
                ends = np.where(changes == -1)[0] + 1
                
                if regime_mask[0]:
                    starts = np.concatenate([[0], starts])
                if regime_mask[-1]:
                    ends = np.concatenate([ends, [len(regime_mask)]])
                
                if len(starts) == len(ends):
                    msm_durations.extend((ends - starts).tolist())
            
            # Calculate average durations for HSMM
            for state in np.unique(hsmm_states_array):
                state_mask = hsmm_states_array == state
                changes = np.diff(state_mask.astype(int))
                starts = np.where(changes == 1)[0] + 1
                ends = np.where(changes == -1)[0] + 1
                
                if state_mask[0]:
                    starts = np.concatenate([[0], starts])
                if state_mask[-1]:
                    ends = np.concatenate([ends, [len(state_mask)]])
                
                if len(starts) == len(ends):
                    hsmm_durations.extend((ends - starts).tolist())
            
            comparison['duration_comparison'] = {
                'msm_avg_duration': float(np.mean(msm_durations)) if msm_durations else 0.0,
                'hsmm_avg_duration': float(np.mean(hsmm_durations)) if hsmm_durations else 0.0,
                'duration_correlation': float(np.corrcoef(
                    msm_durations[:min(len(msm_durations), len(hsmm_durations))],
                    hsmm_durations[:min(len(msm_durations), len(hsmm_durations))]
                )[0,1]) if len(msm_durations) > 1 and len(hsmm_durations) > 1 else 0.0
            }
        
        return comparison
    
    def _generate_data_driven_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from data-driven analysis."""
        insights = []
        
        # MSM insights
        if 'markov_switching' in results:
            msm_results = results['markov_switching']
            n_breaks = len(msm_results.get('structural_breaks', []))
            n_regimes = msm_results.get('n_regimes', 0)
            
            insights.append(f"🔍 Detected {n_breaks} structural breaks leading to {n_regimes} data-driven regimes")
            
            if 'forecasting_models' in msm_results and msm_results['forecasting_models']:
                insights.append("📈 Enhanced forecasting models built for regime transitions")
        
        # HSMM insights
        if 'hidden_semi_markov' in results:
            hsmm_results = results['hidden_semi_markov']
            n_states = hsmm_results.get('n_states', 0)
            
            duration_models = hsmm_results.get('duration_models', {})
            learned_distributions = set(
                model.get('best_distribution', 'unknown') 
                for model in duration_models.values()
            )
            
            insights.append(f"⏱️ Learned {len(learned_distributions)} types of duration distributions across {n_states} states")
            
            if learned_distributions:
                insights.append(f"📊 Data-determined duration types: {', '.join(learned_distributions)}")
        
        # Comparative insights
        if 'comparative_analysis' in results:
            comparison = results['comparative_analysis']
            agreement = comparison.get('model_agreement', {})
            ari = agreement.get('adjusted_rand_score', 0)
            
            if ari > 0.5:
                insights.append("🎯 Strong agreement between MSM and HSMM - data patterns are robust")
            elif ari > 0.3:
                insights.append("📊 Moderate agreement between models - complementary perspectives")
            else:
                insights.append("🔍 Low agreement suggests complex regime structure - investigate further")
        
        # Data-driven validation
        if not insights:
            insights.append("📝 Complete data-driven analysis to generate specific insights")
        
        return insights


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_obs = len(dates)
    
    # Create complex regime structure
    prices = np.zeros(n_obs)
    prices[0] = 100.0
    
    # Data-driven regime periods (no economic assumptions)
    for i in range(1, n_obs):
        # Dynamic regime switching based on data characteristics
        if i < 400:  # Period 1
            ret = np.random.normal(0.001, 0.015)
        elif i < 600:  # Period 2 - detected by structural break
            ret = np.random.normal(-0.002, 0.030)  
        elif i < 800:  # Period 3 - high volatility period
            ret = np.random.normal(0.0, 0.045)
        else:  # Period 4 - recovery
            ret = np.random.normal(0.0008, 0.012)
        
        prices[i] = prices[i-1] * (1 + ret)
    
    # Create test data
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_obs)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_obs))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_obs))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.3, n_obs)
    })
    
    print("🧪 Testing Data-Driven Advanced Markov Models")
    print(f"📊 Test data: {len(test_data)} observations")
    
    # Run data-driven analysis
    integration = DataDrivenAdvancedMarkovIntegration()
    results = integration.run_data_driven_analysis(test_data)
    
    print(f"\n✅ Models executed: {results['models_executed']}")
    
    print("\n💡 Data-Driven Insights:")
    for insight in results['insights']:
        print(f"  {insight}")
    
    if 'comparative_analysis' in results:
        agreement = results['comparative_analysis'].get('model_agreement', {})
        print(f"\n🤝 Model Agreement: ARI={agreement.get('adjusted_rand_score', 0):.3f}")