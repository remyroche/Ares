"""
NAS Integration Component

Integrates with the NAS (Neural Architecture Search) regime detection system
to provide regime detection capabilities for the hybrid system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time

from ..config.hybrid_config import HybridNASConfig


class NASIntegration:
    """
    Integration component for NAS regime detection.
    
    This component interfaces with the NAS regime detection system to:
    1. Extract regime detection results from NAS
    2. Process NAS regime predictions
    3. Convert NAS outputs to hybrid system format
    4. Provide NAS-specific regime analysis
    """
    
    def __init__(self, config: HybridNASConfig):
        """
        Initialize NAS Integration.
        
        Args:
            config: NAS-specific configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize NAS components (these would be imported from actual NAS modules)
        self.nas_engine = None  # Would be initialized with actual NAS engine
        self.nas_detector = None  # Would be initialized with actual NAS detector
        
        self.logger.info("✅ NAS Integration initialized")
        self.logger.info(f"🧠 NAS model types: {config.nas_model_types}")
        self.logger.info(f"🔍 NAS search strategy: {config.nas_search_strategy}")
        self.logger.info(f"📊 NAS regime detection: {config.nas_regime_detection_enabled}")
    
    def detect_regimes(self, 
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect regimes using NAS system.
        
        Args:
            market_data: Market data (OHLCV or features)
            timestamps: Optional timestamps
            
        Returns:
            Dictionary with NAS regime detection results
        """
        start_time = time.time()
        self.logger.info("🧠 Starting NAS regime detection")
        
        try:
            # Prepare data for NAS
            nas_data = self._prepare_nas_data(market_data, timestamps)
            
            # Perform NAS regime detection
            nas_results = self._perform_nas_regime_detection(nas_data)
            
            # Process NAS results
            processed_results = self._process_nas_results(nas_results)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ NAS regime detection completed in {execution_time:.2f}s")
            
            return {
                'success': True,
                'regime_predictions': processed_results['regime_predictions'],
                'regime_probabilities': processed_results['regime_probabilities'],
                'regime_labels': processed_results['regime_labels'],
                'regime_stability_scores': processed_results['regime_stability_scores'],
                'regime_transition_probabilities': processed_results['regime_transition_probabilities'],
                'economic_significance_scores': processed_results['economic_significance_scores'],
                'trading_viability_scores': processed_results['trading_viability_scores'],
                'uncertainty_estimates': processed_results['uncertainty_estimates'],
                'performance_score': processed_results['performance_score'],
                'confidence': processed_results['confidence'],
                'execution_time': execution_time,
                'metadata': {
                    'nas_model_types': self.config.nas_model_types,
                    'nas_search_strategy': self.config.nas_search_strategy,
                    'nas_architecture_types': self.config.nas_architecture_types
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS regime detection failed: {e}")
            
            return {
                'success': False,
                'regime_predictions': np.array([]),
                'regime_probabilities': np.array([]),
                'regime_labels': [],
                'regime_stability_scores': np.array([]),
                'regime_transition_probabilities': np.array([]),
                'economic_significance_scores': np.array([]),
                'trading_viability_scores': np.array([]),
                'uncertainty_estimates': np.array([]),
                'performance_score': 0.0,
                'confidence': 0.0,
                'execution_time': execution_time,
                'error_message': str(e)
            }
    
    def _prepare_nas_data(self, 
                          market_data: Union[pd.DataFrame, np.ndarray],
                          timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Prepare data for NAS regime detection."""
        self.logger.info("📊 Preparing data for NAS regime detection")
        
        # Convert to numpy array if needed
        if isinstance(market_data, pd.DataFrame):
            data_array = market_data.values
            feature_names = market_data.columns.tolist()
        else:
            data_array = market_data
            feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        # Prepare NAS-specific features
        nas_features = self._extract_nas_features(data_array)
        
        return {
            'data': data_array,
            'features': nas_features,
            'feature_names': feature_names,
            'timestamps': timestamps,
            'n_samples': len(data_array),
            'n_features': data_array.shape[1] if len(data_array.shape) > 1 else 1
        }
    
    def _extract_nas_features(self, data: np.ndarray) -> np.ndarray:
        """Extract NAS-specific features."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Neural network specific features
        features = []
        
        # Time series features for neural networks
        if data.shape[1] >= 4:  # OHLCV data
            # Price features
            features.append(data[:, 0])  # Open
            features.append(data[:, 1])  # High
            features.append(data[:, 2])  # Low
            features.append(data[:, 3])  # Close
            
            # Price ratios
            close = data[:, 3]
            open_price = data[:, 0]
            high = data[:, 1]
            low = data[:, 2]
            
            # Price ratios
            features.append(close / (open_price + 1e-8))  # Close/Open ratio
            features.append(high / (low + 1e-8))  # High/Low ratio
            features.append((high - low) / (close + 1e-8))  # Range/Close ratio
            
            # Returns
            returns = np.diff(close) / (close[:-1] + 1e-8)
            features.append(np.concatenate([[0], returns]))  # Returns
            
            # Volatility (rolling standard deviation)
            window_size = min(20, len(close) // 4)
            volatility = self._calculate_rolling_volatility(returns, window_size)
            features.append(volatility)
            
            # Technical indicators for neural networks
            # RSI
            rsi = self._calculate_rsi(close, 14)
            features.append(rsi)
            
            # MACD
            macd_line, macd_signal = self._calculate_macd(close)
            features.append(macd_line)
            features.append(macd_signal)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20)
            features.append(bb_upper)
            features.append(bb_middle)
            features.append(bb_lower)
            features.append((close - bb_lower) / (bb_upper - bb_lower + 1e-8))  # BB position
        
        # Sequence features for neural networks
        if len(data) > 1:
            # Moving averages
            ma_5 = self._calculate_moving_average(data[:, -1], 5)
            ma_20 = self._calculate_moving_average(data[:, -1], 20)
            ma_50 = self._calculate_moving_average(data[:, -1], 50)
            features.append(ma_5)
            features.append(ma_20)
            features.append(ma_50)
            
            # MA ratios
            features.append(ma_5 / (ma_20 + 1e-8))
            features.append(ma_20 / (ma_50 + 1e-8))
            
            # Momentum
            momentum_5 = self._calculate_momentum(data[:, -1], 5)
            momentum_20 = self._calculate_momentum(data[:, -1], 20)
            features.append(momentum_5)
            features.append(momentum_20)
            
            # Rate of change
            roc_5 = self._calculate_rate_of_change(data[:, -1], 5)
            roc_20 = self._calculate_rate_of_change(data[:, -1], 20)
            features.append(roc_5)
            features.append(roc_20)
        
        # Combine all features
        if features:
            nas_features = np.column_stack(features)
        else:
            nas_features = data
        
        return nas_features
    
    def _calculate_rolling_volatility(self, returns: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate rolling volatility."""
        volatility = np.zeros(len(returns))
        
        for i in range(len(returns)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_returns = returns[start_idx:end_idx]
            
            if len(window_returns) > 1:
                volatility[i] = np.std(window_returns)
            else:
                volatility[i] = 0.0
        
        return volatility
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        # Initial averages
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Smoothed averages
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        # RSI calculation
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow_period:
            return np.zeros(len(prices)), np.zeros(len(prices))
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        return macd_line, signal_line
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA (Exponential Moving Average)."""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        ema = np.zeros(len(prices))
        ema[period-1] = np.mean(prices[:period])
        
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
        
        # Calculate SMA
        sma = self._calculate_moving_average(prices, period)
        
        # Calculate standard deviation
        std = np.zeros(len(prices))
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        
        # Bollinger Bands
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average."""
        ma = np.zeros(len(data))
        
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_data = data[start_idx:end_idx]
            ma[i] = np.mean(window_data)
        
        return ma
    
    def _calculate_momentum(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate momentum."""
        momentum = np.zeros(len(data))
        
        for i in range(len(data)):
            if i >= window_size:
                momentum[i] = data[i] - data[i - window_size]
            else:
                momentum[i] = 0.0
        
        return momentum
    
    def _calculate_rate_of_change(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate rate of change."""
        roc = np.zeros(len(data))
        
        for i in range(len(data)):
            if i >= window_size:
                roc[i] = (data[i] - data[i - window_size]) / (data[i - window_size] + 1e-8)
            else:
                roc[i] = 0.0
        
        return roc
    
    def _perform_nas_regime_detection(self, nas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform NAS regime detection."""
        self.logger.info("🔍 Performing NAS regime detection")
        
        # This would integrate with actual NAS system
        # For now, simulate NAS regime detection using neural network approach
        
        data = nas_data['data']
        features = nas_data['features']
        n_samples = nas_data['n_samples']
        
        # Simulate neural network-based regime detection
        n_regimes = min(self.config.nas_regime_detection_enabled and 10 or 6, n_samples // 8)
        n_regimes = max(3, n_regimes)
        
        # Use Gaussian Mixture Model to simulate neural network clustering
        from sklearn.mixture import GaussianMixture
        
        if len(features.shape) == 1:
            features_2d = features.reshape(-1, 1)
        else:
            features_2d = features
        
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        regime_predictions = gmm.fit_predict(features_2d)
        
        # Calculate regime probabilities
        regime_probabilities = gmm.predict_proba(features_2d)
        
        # Calculate regime stability scores
        regime_stability_scores = self._calculate_nas_stability_scores(regime_predictions)
        
        # Calculate transition probabilities
        regime_transition_probabilities = self._calculate_nas_transition_probabilities(regime_predictions)
        
        # Calculate economic significance scores
        economic_significance_scores = self._calculate_nas_economic_significance(
            regime_predictions, data, features
        )
        
        # Calculate trading viability scores
        trading_viability_scores = self._calculate_nas_trading_viability(
            regime_predictions, regime_stability_scores, economic_significance_scores
        )
        
        # Calculate uncertainty estimates
        uncertainty_estimates = self._calculate_nas_uncertainty_estimates(regime_probabilities)
        
        return {
            'regime_predictions': regime_predictions,
            'regime_probabilities': regime_probabilities,
            'regime_stability_scores': regime_stability_scores,
            'regime_transition_probabilities': regime_transition_probabilities,
            'economic_significance_scores': economic_significance_scores,
            'trading_viability_scores': trading_viability_scores,
            'uncertainty_estimates': uncertainty_estimates,
            'gmm_means': gmm.means_,
            'gmm_covariances': gmm.covariances_,
            'aic': gmm.aic(features_2d),
            'bic': gmm.bic(features_2d)
        }
    
    def _calculate_nas_stability_scores(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate NAS regime stability scores."""
        if len(regime_predictions) < 2:
            return np.array([1.0] * len(regime_predictions))
        
        stability_scores = np.zeros(len(regime_predictions))
        
        for i in range(len(regime_predictions)):
            # Look at surrounding regimes for stability
            window_size = min(15, len(regime_predictions) // 3)
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(regime_predictions), i + window_size // 2 + 1)
            
            window_regimes = regime_predictions[start_idx:end_idx]
            current_regime = regime_predictions[i]
            
            # Stability is based on consistency within window
            consistency = np.sum(window_regimes == current_regime) / len(window_regimes)
            stability_scores[i] = consistency
        
        return stability_scores
    
    def _calculate_nas_transition_probabilities(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate NAS regime transition probabilities."""
        if len(regime_predictions) < 2:
            return np.array([0.0] * len(regime_predictions))
        
        transition_probabilities = np.zeros(len(regime_predictions))
        
        for i in range(1, len(regime_predictions)):
            current_regime = regime_predictions[i]
            previous_regime = regime_predictions[i-1]
            
            # Transition probability is 1 if regime changed, 0 if same
            transition_probabilities[i] = 1.0 if current_regime != previous_regime else 0.0
        
        return transition_probabilities
    
    def _calculate_nas_economic_significance(self, 
                                              regime_predictions: np.ndarray,
                                              data: np.ndarray,
                                              features: np.ndarray) -> np.ndarray:
        """Calculate NAS economic significance scores."""
        # Base economic significance on regime stability and neural network confidence
        stability_scores = self._calculate_nas_stability_scores(regime_predictions)
        
        # Adjust based on feature complexity (neural networks excel with complex features)
        if len(features.shape) > 1:
            feature_complexity = np.std(features, axis=1)
            complexity_factor = np.clip(feature_complexity / (np.mean(feature_complexity) + 1e-8), 0.5, 2.0)
        else:
            complexity_factor = np.ones(len(regime_predictions))
        
        # Adjust based on data volatility
        if len(data.shape) > 1 and data.shape[1] > 0:
            price_data = data[:, -1]
            if len(price_data) > 1:
                returns = np.diff(price_data) / (price_data[:-1] + 1e-8)
                volatility = np.std(returns) if len(returns) > 0 else 0.0
                volatility_factor = min(1.0, volatility * 10)
            else:
                volatility_factor = 0.5
        else:
            volatility_factor = 0.5
        
        # Combine factors
        economic_significance = 0.5 * stability_scores + 0.3 * complexity_factor + 0.2 * volatility_factor
        
        return economic_significance
    
    def _calculate_nas_trading_viability(self, 
                                          regime_predictions: np.ndarray,
                                          stability_scores: np.ndarray,
                                          economic_significance_scores: np.ndarray) -> np.ndarray:
        """Calculate NAS trading viability scores."""
        # Trading viability combines stability and economic significance
        trading_viability = 0.6 * stability_scores + 0.4 * economic_significance_scores
        
        # Apply NAS-specific thresholds
        trading_viability = np.where(
            trading_viability >= self.config.nas_trading_viability_threshold,
            trading_viability,
            trading_viability * 0.5  # Penalize low viability
        )
        
        return trading_viability
    
    def _calculate_nas_uncertainty_estimates(self, regime_probabilities: np.ndarray) -> np.ndarray:
        """Calculate NAS uncertainty estimates."""
        if len(regime_probabilities) == 0:
            return np.array([])
        
        uncertainty_scores = np.zeros(len(regime_probabilities))
        
        for i, probs in enumerate(regime_probabilities):
            if isinstance(probs, (list, np.ndarray)) and len(probs) > 1:
                # Calculate entropy
                probs = np.array(probs)
                probs = probs / (np.sum(probs) + 1e-8)  # Normalize
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                uncertainty_scores[i] = entropy
            else:
                # Single probability value
                prob = float(probs) if not isinstance(probs, (list, np.ndarray)) else probs[0]
                uncertainty_scores[i] = -prob * np.log(prob + 1e-8) - (1-prob) * np.log(1-prob + 1e-8)
        
        return uncertainty_scores
    
    def _process_nas_results(self, nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process NAS results for hybrid system."""
        self.logger.info("⚙️ Processing NAS results")
        
        # Extract regime predictions and probabilities
        regime_predictions = nas_results['regime_predictions']
        regime_probabilities = nas_results['regime_probabilities']
        
        # Generate regime labels
        regime_labels = self._generate_nas_regime_labels(regime_predictions)
        
        # Calculate performance score
        performance_score = self._calculate_nas_performance_score(nas_results)
        
        # Calculate confidence
        confidence = self._calculate_nas_confidence(nas_results)
        
        return {
            'regime_predictions': regime_predictions,
            'regime_probabilities': regime_probabilities,
            'regime_labels': regime_labels,
            'regime_stability_scores': nas_results['regime_stability_scores'],
            'regime_transition_probabilities': nas_results['regime_transition_probabilities'],
            'economic_significance_scores': nas_results['economic_significance_scores'],
            'trading_viability_scores': nas_results['trading_viability_scores'],
            'uncertainty_estimates': nas_results['uncertainty_estimates'],
            'performance_score': performance_score,
            'confidence': confidence
        }
    
    def _generate_nas_regime_labels(self, regime_predictions: np.ndarray) -> List[str]:
        """Generate NAS regime labels."""
        unique_regimes = np.unique(regime_predictions)
        regime_labels = []
        
        for regime_id in unique_regimes:
            if regime_id == 0:
                regime_labels.append("normal")
            elif regime_id == 1:
                regime_labels.append("bull_market")
            elif regime_id == 2:
                regime_labels.append("bear_market")
            elif regime_id == 3:
                regime_labels.append("high_volatility")
            elif regime_id == 4:
                regime_labels.append("low_volatility")
            elif regime_id == 5:
                regime_labels.append("trending_up")
            elif regime_id == 6:
                regime_labels.append("trending_down")
            elif regime_id == 7:
                regime_labels.append("mean_reverting")
            elif regime_id == 8:
                regime_labels.append("breakout")
            elif regime_id == 9:
                regime_labels.append("consolidation")
            else:
                regime_labels.append("unknown")
        
        return regime_labels
    
    def _calculate_nas_performance_score(self, nas_results: Dict[str, Any]) -> float:
        """Calculate NAS performance score."""
        # Combine multiple performance metrics
        stability_score = np.mean(nas_results['regime_stability_scores'])
        economic_score = np.mean(nas_results['economic_significance_scores'])
        trading_score = np.mean(nas_results['trading_viability_scores'])
        
        # Weighted combination
        performance_score = 0.4 * stability_score + 0.3 * economic_score + 0.3 * trading_score
        
        return float(performance_score)
    
    def _calculate_nas_confidence(self, nas_results: Dict[str, Any]) -> float:
        """Calculate NAS confidence score."""
        # Confidence based on stability and uncertainty
        stability_score = np.mean(nas_results['regime_stability_scores'])
        uncertainty_score = np.mean(nas_results['uncertainty_estimates'])
        
        # Higher stability and lower uncertainty = higher confidence
        confidence = stability_score * (1.0 - uncertainty_score)
        
        return float(confidence)
    
    def get_nas_summary(self) -> Dict[str, Any]:
        """Get summary of NAS integration."""
        return {
            "nas_model_types": self.config.nas_model_types,
            "nas_search_strategy": self.config.nas_search_strategy,
            "nas_regime_detection_enabled": self.config.nas_regime_detection_enabled,
            "nas_economic_significance_threshold": self.config.nas_economic_significance_threshold,
            "nas_trading_viability_threshold": self.config.nas_trading_viability_threshold
        }