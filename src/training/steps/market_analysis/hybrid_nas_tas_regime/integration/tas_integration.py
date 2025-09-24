"""
TAS Integration Component

Integrates with the TAS (Tree Architecture Search) regime detection system
to provide regime detection capabilities for the hybrid system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time

from ..config.hybrid_config import HybridTASConfig


class TASIntegration:
    """
    Integration component for TAS regime detection.
    
    This component interfaces with the TAS regime detection system to:
    1. Extract regime detection results from TAS
    2. Process TAS regime predictions
    3. Convert TAS outputs to hybrid system format
    4. Provide TAS-specific regime analysis
    """
    
    def __init__(self, config: HybridTASConfig):
        """
        Initialize TAS Integration.
        
        Args:
            config: TAS-specific configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize TAS components (these would be imported from actual TAS modules)
        self.tas_engine = None  # Would be initialized with actual TAS engine
        self.tas_clusterer = None  # Would be initialized with actual TAS clusterer
        
        self.logger.info("✅ TAS Integration initialized")
        self.logger.info(f"🌳 TAS model types: {config.tas_model_types}")
        self.logger.info(f"🔍 TAS search strategy: {config.tas_search_strategy}")
        self.logger.info(f"📊 TAS regime detection: {config.tas_regime_detection_enabled}")
    
    def detect_regimes(self, 
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect regimes using TAS system.
        
        Args:
            market_data: Market data (OHLCV or features)
            timestamps: Optional timestamps
            
        Returns:
            Dictionary with TAS regime detection results
        """
        start_time = time.time()
        self.logger.info("🌳 Starting TAS regime detection")
        
        try:
            # Prepare data for TAS
            tas_data = self._prepare_tas_data(market_data, timestamps)
            
            # Perform TAS regime detection
            tas_results = self._perform_tas_regime_detection(tas_data)
            
            # Process TAS results
            processed_results = self._process_tas_results(tas_results)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ TAS regime detection completed in {execution_time:.2f}s")
            
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
                    'tas_model_types': self.config.tas_model_types,
                    'tas_search_strategy': self.config.tas_search_strategy,
                    'tas_architecture_types': self.config.tas_architecture_types
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ TAS regime detection failed: {e}")
            
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
    
    def _prepare_tas_data(self, 
                          market_data: Union[pd.DataFrame, np.ndarray],
                          timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Prepare data for TAS regime detection."""
        self.logger.info("📊 Preparing data for TAS regime detection")
        
        # Convert to numpy array if needed
        if isinstance(market_data, pd.DataFrame):
            data_array = market_data.values
            feature_names = market_data.columns.tolist()
        else:
            data_array = market_data
            feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        # Prepare TAS-specific features
        tas_features = self._extract_tas_features(data_array)
        
        return {
            'data': data_array,
            'features': tas_features,
            'feature_names': feature_names,
            'timestamps': timestamps,
            'n_samples': len(data_array),
            'n_features': data_array.shape[1] if len(data_array.shape) > 1 else 1
        }
    
    def _extract_tas_features(self, data: np.ndarray) -> np.ndarray:
        """Extract TAS-specific features."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Basic statistical features
        features = []
        
        # Price-based features (if OHLCV data)
        if data.shape[1] >= 4:  # OHLCV data
            # Price features
            features.append(data[:, 0])  # Open
            features.append(data[:, 1])  # High
            features.append(data[:, 2])  # Low
            features.append(data[:, 3])  # Close
            
            # Price ratios
            if data.shape[1] >= 4:
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
        
        # Technical indicators
        if len(data) > 1:
            # Moving averages
            ma_5 = self._calculate_moving_average(data[:, -1], 5)
            ma_20 = self._calculate_moving_average(data[:, -1], 20)
            features.append(ma_5)
            features.append(ma_20)
            features.append(ma_5 / (ma_20 + 1e-8))  # MA ratio
            
            # Momentum
            momentum = self._calculate_momentum(data[:, -1], 10)
            features.append(momentum)
        
        # Combine all features
        if features:
            tas_features = np.column_stack(features)
        else:
            tas_features = data
        
        return tas_features
    
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
    
    def _perform_tas_regime_detection(self, tas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform TAS regime detection."""
        self.logger.info("🔍 Performing TAS regime detection")
        
        # This would integrate with actual TAS system
        # For now, simulate TAS regime detection
        
        data = tas_data['data']
        n_samples = tas_data['n_samples']
        
        # Simulate regime detection using clustering
        n_regimes = min(self.config.tas_regime_detection_enabled and 8 or 4, n_samples // 10)
        n_regimes = max(2, n_regimes)
        
        # Simple clustering-based regime detection
        from sklearn.cluster import KMeans
        
        if len(data.shape) == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_predictions = kmeans.fit_predict(data_2d)
        
        # Calculate regime probabilities (distance to cluster centers)
        distances = kmeans.transform(data_2d)
        regime_probabilities = 1.0 / (distances + 1e-8)
        regime_probabilities = regime_probabilities / np.sum(regime_probabilities, axis=1, keepdims=True)
        
        # Calculate regime stability scores
        regime_stability_scores = self._calculate_tas_stability_scores(regime_predictions)
        
        # Calculate transition probabilities
        regime_transition_probabilities = self._calculate_tas_transition_probabilities(regime_predictions)
        
        # Calculate economic significance scores
        economic_significance_scores = self._calculate_tas_economic_significance(
            regime_predictions, data
        )
        
        # Calculate trading viability scores
        trading_viability_scores = self._calculate_tas_trading_viability(
            regime_predictions, regime_stability_scores, economic_significance_scores
        )
        
        # Calculate uncertainty estimates
        uncertainty_estimates = self._calculate_tas_uncertainty_estimates(regime_probabilities)
        
        return {
            'regime_predictions': regime_predictions,
            'regime_probabilities': regime_probabilities,
            'regime_stability_scores': regime_stability_scores,
            'regime_transition_probabilities': regime_transition_probabilities,
            'economic_significance_scores': economic_significance_scores,
            'trading_viability_scores': trading_viability_scores,
            'uncertainty_estimates': uncertainty_estimates,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
    
    def _calculate_tas_stability_scores(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate TAS regime stability scores."""
        if len(regime_predictions) < 2:
            return np.array([1.0] * len(regime_predictions))
        
        stability_scores = np.zeros(len(regime_predictions))
        
        for i in range(len(regime_predictions)):
            # Look at surrounding regimes for stability
            window_size = min(10, len(regime_predictions) // 4)
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(regime_predictions), i + window_size // 2 + 1)
            
            window_regimes = regime_predictions[start_idx:end_idx]
            current_regime = regime_predictions[i]
            
            # Stability is based on consistency within window
            consistency = np.sum(window_regimes == current_regime) / len(window_regimes)
            stability_scores[i] = consistency
        
        return stability_scores
    
    def _calculate_tas_transition_probabilities(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate TAS regime transition probabilities."""
        if len(regime_predictions) < 2:
            return np.array([0.0] * len(regime_predictions))
        
        transition_probabilities = np.zeros(len(regime_predictions))
        
        for i in range(1, len(regime_predictions)):
            current_regime = regime_predictions[i]
            previous_regime = regime_predictions[i-1]
            
            # Transition probability is 1 if regime changed, 0 if same
            transition_probabilities[i] = 1.0 if current_regime != previous_regime else 0.0
        
        return transition_probabilities
    
    def _calculate_tas_economic_significance(self, 
                                              regime_predictions: np.ndarray,
                                              data: np.ndarray) -> np.ndarray:
        """Calculate TAS economic significance scores."""
        # Base economic significance on regime stability and data characteristics
        stability_scores = self._calculate_tas_stability_scores(regime_predictions)
        
        # Adjust based on data volatility (higher volatility = higher economic significance)
        if len(data.shape) > 1 and data.shape[1] > 0:
            # Use last column (usually close price) for volatility calculation
            price_data = data[:, -1]
            if len(price_data) > 1:
                returns = np.diff(price_data) / (price_data[:-1] + 1e-8)
                volatility = np.std(returns) if len(returns) > 0 else 0.0
                volatility_factor = min(1.0, volatility * 10)  # Scale volatility
            else:
                volatility_factor = 0.5
        else:
            volatility_factor = 0.5
        
        # Combine stability and volatility
        economic_significance = 0.7 * stability_scores + 0.3 * volatility_factor
        
        return economic_significance
    
    def _calculate_tas_trading_viability(self, 
                                         regime_predictions: np.ndarray,
                                         stability_scores: np.ndarray,
                                         economic_significance_scores: np.ndarray) -> np.ndarray:
        """Calculate TAS trading viability scores."""
        # Trading viability combines stability and economic significance
        trading_viability = 0.6 * stability_scores + 0.4 * economic_significance_scores
        
        # Apply TAS-specific thresholds
        trading_viability = np.where(
            trading_viability >= self.config.tas_trading_viability_threshold,
            trading_viability,
            trading_viability * 0.5  # Penalize low viability
        )
        
        return trading_viability
    
    def _calculate_tas_uncertainty_estimates(self, regime_probabilities: np.ndarray) -> np.ndarray:
        """Calculate TAS uncertainty estimates."""
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
    
    def _process_tas_results(self, tas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process TAS results for hybrid system."""
        self.logger.info("⚙️ Processing TAS results")
        
        # Extract regime predictions and probabilities
        regime_predictions = tas_results['regime_predictions']
        regime_probabilities = tas_results['regime_probabilities']
        
        # Generate regime labels
        regime_labels = self._generate_tas_regime_labels(regime_predictions)
        
        # Calculate performance score
        performance_score = self._calculate_tas_performance_score(tas_results)
        
        # Calculate confidence
        confidence = self._calculate_tas_confidence(tas_results)
        
        return {
            'regime_predictions': regime_predictions,
            'regime_probabilities': regime_probabilities,
            'regime_labels': regime_labels,
            'regime_stability_scores': tas_results['regime_stability_scores'],
            'regime_transition_probabilities': tas_results['regime_transition_probabilities'],
            'economic_significance_scores': tas_results['economic_significance_scores'],
            'trading_viability_scores': tas_results['trading_viability_scores'],
            'uncertainty_estimates': tas_results['uncertainty_estimates'],
            'performance_score': performance_score,
            'confidence': confidence
        }
    
    def _generate_tas_regime_labels(self, regime_predictions: np.ndarray) -> List[str]:
        """Generate TAS regime labels."""
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
            else:
                regime_labels.append("unknown")
        
        return regime_labels
    
    def _calculate_tas_performance_score(self, tas_results: Dict[str, Any]) -> float:
        """Calculate TAS performance score."""
        # Combine multiple performance metrics
        stability_score = np.mean(tas_results['regime_stability_scores'])
        economic_score = np.mean(tas_results['economic_significance_scores'])
        trading_score = np.mean(tas_results['trading_viability_scores'])
        
        # Weighted combination
        performance_score = 0.4 * stability_score + 0.3 * economic_score + 0.3 * trading_score
        
        return float(performance_score)
    
    def _calculate_tas_confidence(self, tas_results: Dict[str, Any]) -> float:
        """Calculate TAS confidence score."""
        # Confidence based on stability and uncertainty
        stability_score = np.mean(tas_results['regime_stability_scores'])
        uncertainty_score = np.mean(tas_results['uncertainty_estimates'])
        
        # Higher stability and lower uncertainty = higher confidence
        confidence = stability_score * (1.0 - uncertainty_score)
        
        return float(confidence)
    
    def get_tas_summary(self) -> Dict[str, Any]:
        """Get summary of TAS integration."""
        return {
            "tas_model_types": self.config.tas_model_types,
            "tas_search_strategy": self.config.tas_search_strategy,
            "tas_regime_detection_enabled": self.config.tas_regime_detection_enabled,
            "tas_economic_significance_threshold": self.config.tas_economic_significance_threshold,
            "tas_trading_viability_threshold": self.config.tas_trading_viability_threshold
        }