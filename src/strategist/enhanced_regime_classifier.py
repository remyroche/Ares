"""
Enhanced Regime Classifier for Strategist
Implements refined market regime detection with more granular regime types
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates
from src.core.decorators.errors import handles_errors


class EnhancedRegimeClassifier:
    """
    Enhanced regime classifier with refined regime types for strategy generation.
    
    Refined Regimes:
    1. STRONG_BULL: Strong uptrend with high momentum
    2. MODERATE_BULL: Steady uptrend with normal momentum
    3. WEAK_BULL: Weak uptrend, potential reversal
    4. STRONG_BEAR: Strong downtrend with high momentum
    5. MODERATE_BEAR: Steady downtrend with normal momentum
    6. WEAK_BEAR: Weak downtrend, potential reversal
    7. RANGING_HIGH: Sideways movement in upper price range
    8. RANGING_LOW: Sideways movement in lower price range
    9. VOLATILE_BULLISH: High volatility with bullish bias
    10. VOLATILE_BEARISH: High volatility with bearish bias
    11. BREAKOUT_UP: Breaking resistance levels
    12. BREAKOUT_DOWN: Breaking support levels
    """
    
    # Refined regime types
    REGIMES = [
        "STRONG_BULL", "MODERATE_BULL", "WEAK_BULL",
        "STRONG_BEAR", "MODERATE_BEAR", "WEAK_BEAR",
        "RANGING_HIGH", "RANGING_LOW",
        "VOLATILE_BULLISH", "VOLATILE_BEARISH",
        "BREAKOUT_UP", "BREAKOUT_DOWN"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedRegimeClassifier")
        
        # HMM configuration - 12 states for refined regimes
        self.n_states = 12
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.trained = False
        
        # Regime detection parameters
        self.momentum_threshold = config.get("momentum_threshold", 0.001)
        self.volatility_threshold = config.get("volatility_threshold", 0.02)
        self.volume_threshold = config.get("volume_threshold", 1.5)
        self.breakout_threshold = config.get("breakout_threshold", 0.03)
        
        # Feature windows
        self.short_window = config.get("short_window", 5)
        self.medium_window = config.get("medium_window", 20)
        self.long_window = config.get("long_window", 50)
        
    @handles_errors(fallback=None)
    async def initialize(self) -> bool:
        """Initialize the enhanced regime classifier."""
        try:
            self.logger.info("Initializing Enhanced Regime Classifier...")
            
            # Initialize HMM model
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            
            self.logger.info("✅ Enhanced Regime Classifier initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize regime classifier: {e}")
            return False
    
    def calculate_enhanced_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced features for refined regime detection.
        
        Features include:
        - Multi-timeframe momentum
        - Volatility patterns
        - Volume dynamics
        - Price position relative to moving averages
        - Trend strength indicators
        """
        features = pd.DataFrame(index=market_data.index)
        
        # Price returns at multiple scales
        features['return_1'] = market_data['close'].pct_change(1)
        features['return_5'] = market_data['close'].pct_change(self.short_window)
        features['return_20'] = market_data['close'].pct_change(self.medium_window)
        
        # Momentum indicators
        features['momentum_short'] = features['return_5'].rolling(self.short_window).mean()
        features['momentum_medium'] = features['return_20'].rolling(self.medium_window).mean()
        features['momentum_acceleration'] = features['momentum_short'] - features['momentum_short'].shift(self.short_window)
        
        # Volatility measures
        features['volatility_short'] = features['return_1'].rolling(self.short_window).std()
        features['volatility_medium'] = features['return_1'].rolling(self.medium_window).std()
        features['volatility_ratio'] = features['volatility_short'] / features['volatility_medium']
        
        # Volume analysis
        features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(self.medium_window).mean()
        features['volume_trend'] = market_data['volume'].rolling(self.short_window).mean() / market_data['volume'].rolling(self.medium_window).mean()
        
        # Price position
        sma_short = market_data['close'].rolling(self.short_window).mean()
        sma_medium = market_data['close'].rolling(self.medium_window).mean()
        sma_long = market_data['close'].rolling(self.long_window).mean()
        
        features['price_position_short'] = (market_data['close'] - sma_short) / sma_short
        features['price_position_medium'] = (market_data['close'] - sma_medium) / sma_medium
        features['price_position_long'] = (market_data['close'] - sma_long) / sma_long
        
        # Trend strength
        features['trend_strength'] = (sma_short - sma_long) / sma_long
        features['trend_consistency'] = features['return_5'].rolling(self.medium_window).apply(lambda x: np.sum(x > 0) / len(x))
        
        # High/Low analysis for breakout detection
        high_rolling = market_data['high'].rolling(self.medium_window).max()
        low_rolling = market_data['low'].rolling(self.medium_window).min()
        
        features['distance_from_high'] = (high_rolling - market_data['close']) / market_data['close']
        features['distance_from_low'] = (market_data['close'] - low_rolling) / market_data['close']
        features['range_position'] = (market_data['close'] - low_rolling) / (high_rolling - low_rolling)
        
        return features.fillna(0)
    
    def classify_regime(self, features: pd.Series, hmm_state: int = None) -> str:
        """
        Classify market regime based on enhanced features.
        
        Args:
            features: Current feature values
            hmm_state: Optional HMM state for additional context
            
        Returns:
            Refined regime classification
        """
        # Extract key metrics
        momentum_short = features['momentum_short']
        momentum_medium = features['momentum_medium']
        momentum_accel = features['momentum_acceleration']
        volatility_ratio = features['volatility_ratio']
        volume_ratio = features['volume_ratio']
        trend_strength = features['trend_strength']
        distance_from_high = features['distance_from_high']
        distance_from_low = features['distance_from_low']
        
        # Breakout detection
        if distance_from_high < 0.001 and volume_ratio > self.volume_threshold:
            return "BREAKOUT_UP"
        elif distance_from_low < 0.001 and volume_ratio > self.volume_threshold:
            return "BREAKOUT_DOWN"
        
        # Volatility-based regimes
        if volatility_ratio > 1.5:
            if momentum_short > self.momentum_threshold:
                return "VOLATILE_BULLISH"
            elif momentum_short < -self.momentum_threshold:
                return "VOLATILE_BEARISH"
        
        # Trend-based regimes
        if trend_strength > 0.02:  # Strong bullish trend
            if momentum_accel > 0 and volume_ratio > 1.2:
                return "STRONG_BULL"
            elif momentum_short > self.momentum_threshold:
                return "MODERATE_BULL"
            else:
                return "WEAK_BULL"
                
        elif trend_strength < -0.02:  # Strong bearish trend
            if momentum_accel < 0 and volume_ratio > 1.2:
                return "STRONG_BEAR"
            elif momentum_short < -self.momentum_threshold:
                return "MODERATE_BEAR"
            else:
                return "WEAK_BEAR"
        
        # Ranging regimes
        else:
            if features['range_position'] > 0.7:
                return "RANGING_HIGH"
            elif features['range_position'] < 0.3:
                return "RANGING_LOW"
            else:
                # Default based on short-term momentum
                if momentum_short > 0:
                    return "WEAK_BULL"
                else:
                    return "WEAK_BEAR"
    
    @handles_errors(fallback=("MODERATE_BULL", 0.5, {}))
    async def predict_regime(
        self, 
        market_data: pd.DataFrame
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict current market regime with confidence.
        
        Args:
            market_data: Recent market data
            
        Returns:
            Tuple of (regime, confidence, metadata)
        """
        try:
            # Calculate features
            features_df = self.calculate_enhanced_features(market_data)
            if features_df.empty or len(features_df) < self.long_window:
                return "MODERATE_BULL", 0.3, {"error": "Insufficient data"}
            
            current_features = features_df.iloc[-1]
            
            # Get HMM prediction if model is trained
            hmm_state = None
            hmm_confidence = 0.5
            
            if self.trained and self.hmm_model is not None:
                # Prepare features for HMM
                hmm_features = features_df[[
                    'return_1', 'return_5', 'return_20',
                    'volatility_short', 'volatility_medium',
                    'volume_ratio', 'trend_strength'
                ]].iloc[-self.medium_window:].values
                
                # Scale features
                hmm_features_scaled = self.scaler.transform(hmm_features)
                
                # Predict state
                hmm_state = self.hmm_model.predict(hmm_features_scaled)[-1]
                
                # Calculate confidence from HMM probabilities
                log_prob, posteriors = self.hmm_model.score_samples(hmm_features_scaled)
                hmm_confidence = np.max(posteriors[-1])
            
            # Classify regime
            regime = self.classify_regime(current_features, hmm_state)
            
            # Calculate rule-based confidence
            rule_confidence = self._calculate_regime_confidence(current_features, regime)
            
            # Combine confidences
            final_confidence = 0.7 * rule_confidence + 0.3 * hmm_confidence if self.trained else rule_confidence
            
            # Generate metadata
            metadata = {
                "hmm_state": int(hmm_state) if hmm_state is not None else None,
                "hmm_confidence": float(hmm_confidence),
                "rule_confidence": float(rule_confidence),
                "momentum_short": float(current_features['momentum_short']),
                "momentum_medium": float(current_features['momentum_medium']),
                "volatility_ratio": float(current_features['volatility_ratio']),
                "trend_strength": float(current_features['trend_strength']),
                "volume_ratio": float(current_features['volume_ratio']),
                "timestamp": datetime.now().isoformat()
            }
            
            return regime, final_confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Error predicting regime: {e}")
            return "MODERATE_BULL", 0.3, {"error": str(e)}
    
    def _calculate_regime_confidence(self, features: pd.Series, regime: str) -> float:
        """Calculate confidence for regime classification based on feature strength."""
        confidence = 0.5
        
        # Momentum-based confidence
        momentum_strength = abs(features['momentum_short'])
        if momentum_strength > self.momentum_threshold * 2:
            confidence += 0.2
        elif momentum_strength > self.momentum_threshold:
            confidence += 0.1
        
        # Trend consistency confidence
        if features['trend_consistency'] > 0.7:
            confidence += 0.15
        elif features['trend_consistency'] > 0.6:
            confidence += 0.1
        
        # Volume confirmation
        if features['volume_ratio'] > 1.5:
            confidence += 0.1
        
        # Breakout confidence boost
        if "BREAKOUT" in regime:
            if features['volume_ratio'] > 2.0:
                confidence += 0.2
            else:
                confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def train(self, historical_data: pd.DataFrame) -> bool:
        """Train HMM model on historical data."""
        try:
            self.logger.info("Training Enhanced Regime Classifier...")
            
            # Calculate features
            features_df = self.calculate_enhanced_features(historical_data)
            
            # Select features for HMM
            hmm_features = features_df[[
                'return_1', 'return_5', 'return_20',
                'volatility_short', 'volatility_medium',
                'volume_ratio', 'trend_strength'
            ]].dropna().values
            
            # Scale features
            hmm_features_scaled = self.scaler.fit_transform(hmm_features)
            
            # Train HMM
            self.hmm_model.fit(hmm_features_scaled)
            
            self.trained = True
            self.logger.info("✅ Enhanced Regime Classifier trained successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to train regime classifier: {e}")
            return False
    
    def get_regime_strategy_params(self, regime: str) -> Dict[str, Any]:
        """
        Get strategy parameters based on regime.
        
        Returns regime-specific parameters for:
        - Position sizing
        - Risk management
        - Entry/exit thresholds
        - Indicator weights
        """
        regime_params = {
            "STRONG_BULL": {
                "position_size_multiplier": 1.5,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 1.5,
                "entry_confidence_threshold": 0.6,
                "momentum_weight": 0.7,
                "mean_reversion_weight": 0.3
            },
            "MODERATE_BULL": {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.2,
                "entry_confidence_threshold": 0.65,
                "momentum_weight": 0.6,
                "mean_reversion_weight": 0.4
            },
            "WEAK_BULL": {
                "position_size_multiplier": 0.7,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 1.0,
                "entry_confidence_threshold": 0.7,
                "momentum_weight": 0.4,
                "mean_reversion_weight": 0.6
            },
            "STRONG_BEAR": {
                "position_size_multiplier": 1.5,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 1.5,
                "entry_confidence_threshold": 0.6,
                "momentum_weight": 0.7,
                "mean_reversion_weight": 0.3
            },
            "MODERATE_BEAR": {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.2,
                "entry_confidence_threshold": 0.65,
                "momentum_weight": 0.6,
                "mean_reversion_weight": 0.4
            },
            "WEAK_BEAR": {
                "position_size_multiplier": 0.7,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 1.0,
                "entry_confidence_threshold": 0.7,
                "momentum_weight": 0.4,
                "mean_reversion_weight": 0.6
            },
            "RANGING_HIGH": {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 0.8,
                "entry_confidence_threshold": 0.75,
                "momentum_weight": 0.3,
                "mean_reversion_weight": 0.7
            },
            "RANGING_LOW": {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 0.8,
                "entry_confidence_threshold": 0.75,
                "momentum_weight": 0.3,
                "mean_reversion_weight": 0.7
            },
            "VOLATILE_BULLISH": {
                "position_size_multiplier": 0.6,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 1.8,
                "entry_confidence_threshold": 0.8,
                "momentum_weight": 0.5,
                "mean_reversion_weight": 0.5
            },
            "VOLATILE_BEARISH": {
                "position_size_multiplier": 0.6,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 1.8,
                "entry_confidence_threshold": 0.8,
                "momentum_weight": 0.5,
                "mean_reversion_weight": 0.5
            },
            "BREAKOUT_UP": {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 0.7,
                "take_profit_multiplier": 2.0,
                "entry_confidence_threshold": 0.65,
                "momentum_weight": 0.8,
                "mean_reversion_weight": 0.2
            },
            "BREAKOUT_DOWN": {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 0.7,
                "take_profit_multiplier": 2.0,
                "entry_confidence_threshold": 0.65,
                "momentum_weight": 0.8,
                "mean_reversion_weight": 0.2
            }
        }
        
        return regime_params.get(regime, regime_params["MODERATE_BULL"])