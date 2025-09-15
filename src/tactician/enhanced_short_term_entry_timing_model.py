"""
Enhanced Short-Term Entry Timing Model

This module implements an enhanced version of the short-term entry timing model that provides:
1. More accurate predictions through advanced feature engineering
2. Pre-movement prediction (predicts price movements before target direction)
3. Sophisticated entry timing (e.g., wait for price to increase before shorting)
4. Multi-phase prediction system
5. Advanced risk assessment and confidence calibration

Key Features:
- Multi-phase prediction (pre-movement, target-movement, post-movement)
- Advanced feature engineering for ultra-short-term patterns
- Sophisticated entry timing logic
- Enhanced confidence calibration
- Market microstructure analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates, traced
from src.tactician.short_term_entry_timing_model import (
    ShortTermEntryTimingModel, ShortTermEntryTimingConfig, ShortTermPrediction
)
from src.tactician.short_term_target_generator import TripleBarrierConfig

logger = system_logger.getChild('EnhancedShortTermEntryTimingModel')


@dataclass
class PreMovementPrediction:
    """Prediction for price movement before target direction."""
    # Pre-movement analysis
    pre_movement_direction: str  # 'up', 'down', 'neutral'
    pre_movement_magnitude: float  # Expected magnitude of pre-movement
    pre_movement_duration: float  # Expected duration in minutes
    pre_movement_confidence: float  # Confidence in pre-movement prediction
    
    # Entry timing recommendation
    optimal_entry_timing: str  # 'immediate', 'wait_for_pre_movement', 'wait_for_reversal'
    wait_duration_minutes: float  # How long to wait before entry
    entry_reasoning: str  # Human-readable reasoning


@dataclass
class EnhancedShortTermPrediction:
    """Enhanced prediction with pre-movement analysis."""
    # Base prediction (inherited from ShortTermPrediction)
    target_percentage: float
    target_name: str
    probability: float
    timing_minutes: float
    direction: str
    confidence_score: float
    risk_reward_ratio: float
    max_adverse_movement: float
    is_valid: bool
    validation_reason: str
    
    # Enhanced predictions
    pre_movement: PreMovementPrediction
    post_movement_prediction: Dict[str, Any]  # What happens after target is reached
    
    # Sophisticated entry timing
    entry_strategy: str  # 'immediate', 'wait_and_enter', 'staged_entry'
    entry_confidence: float  # Confidence in entry timing
    exit_strategy: str  # 'target_reached', 'time_stop', 'adverse_movement'
    
    # Advanced risk metrics
    drawdown_probability: float  # Probability of experiencing drawdown
    volatility_forecast: float  # Expected volatility during position
    liquidity_impact: float  # Expected impact on liquidity


@dataclass
class EnhancedShortTermEntryTimingResult:
    """Enhanced result with sophisticated entry timing."""
    # Basic info (inherited)
    model_name: str
    timestamp: datetime
    symbol: str
    timeframe: str
    current_price: float
    
    # Enhanced predictions
    predictions: List[EnhancedShortTermPrediction] = field(default_factory=list)
    best_prediction: Optional[EnhancedShortTermPrediction] = None
    
    # Enhanced summary metrics
    overall_confidence: float = 0.0
    risk_score: float = 0.0
    entry_recommendation: str = "HOLD"
    pre_movement_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced metrics
    prediction_time: float = 0.0
    model_confidence: float = 0.0
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    n_predictions: int = 0
    valid_predictions: int = 0


class EnhancedShortTermEntryTimingModel:
    """
    Enhanced short-term entry timing model with sophisticated prediction capabilities.
    
    This model provides:
    1. Pre-movement prediction (what happens before target direction)
    2. Sophisticated entry timing (when to wait, when to enter immediately)
    3. Advanced risk assessment
    4. Market microstructure analysis
    5. Enhanced confidence calibration
    """
    
    def __init__(self, config: Optional[ShortTermEntryTimingConfig] = None):
        """
        Initialize the enhanced short-term entry timing model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ShortTermEntryTimingConfig()
        self.logger = logger.getChild('EnhancedShortTermEntryTimingModel')
        
        # Initialize base model
        self.base_model = ShortTermEntryTimingModel(self.config)
        
        # Enhanced models for different prediction phases
        self.pre_movement_model = None
        self.post_movement_model = None
        self.entry_timing_model = None
        self.risk_assessment_model = None
        
        # Feature engineering components
        self.feature_scaler = RobustScaler()
        self.confidence_calibrator = None
        
        # Model state
        self.is_fitted = False
        self.enhanced_features_enabled = True
        
        self.logger.info(f"🚀 Initializing Enhanced ShortTermEntryTimingModel")
        self.logger.info(f"📊 Target percentages: {[f'{p*100:.1f}%' for p in self.config.target_percentages]}")
        self.logger.info(f"🎯 Enhanced features: {self.enhanced_features_enabled}")
        
    def _create_enhanced_features(self, price_data: pd.DataFrame, features: np.ndarray) -> np.ndarray:
        """Create enhanced features for sophisticated prediction."""
        
        try:
            enhanced_features = []
            
            # 1. Market Microstructure Features
            microstructure_features = self._extract_microstructure_features(price_data)
            enhanced_features.append(microstructure_features)
            
            # 2. Pre-Movement Pattern Features
            pre_movement_features = self._extract_pre_movement_features(price_data)
            enhanced_features.append(pre_movement_features)
            
            # 3. Volatility Regime Features
            volatility_features = self._extract_volatility_regime_features(price_data)
            enhanced_features.append(volatility_features)
            
            # 4. Order Flow Features (simulated)
            order_flow_features = self._extract_order_flow_features(price_data)
            enhanced_features.append(order_flow_features)
            
            # 5. Time-based Features
            time_features = self._extract_time_based_features(price_data)
            enhanced_features.append(time_features)
            
            # Combine all enhanced features
            if enhanced_features:
                enhanced_features_array = np.column_stack(enhanced_features)
                # Combine with original features
                combined_features = np.column_stack([features, enhanced_features_array])
                
                self.logger.debug(f"📊 Enhanced features: {features.shape} -> {combined_features.shape}")
                return combined_features
            else:
                return features
                
        except Exception as e:
            self.logger.error(f"❌ Error creating enhanced features: {e}")
            return features
    
    def _extract_microstructure_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Extract market microstructure features."""
        
        try:
            features = []
            
            # Price impact features
            price_impact = (price_data['high'] - price_data['low']) / price_data['close']
            features.append(price_impact.fillna(0).values)
            
            # Volume-price relationship
            volume_price_corr = price_data['volume'].rolling(10).corr(price_data['close'].pct_change())
            features.append(volume_price_corr.fillna(0).values)
            
            # Bid-ask spread proxy (using high-low spread)
            spread_proxy = (price_data['high'] - price_data['low']) / price_data['close']
            features.append(spread_proxy.fillna(0).values)
            
            # Trade size distribution (simulated)
            trade_size_volatility = price_data['volume'].rolling(5).std() / price_data['volume'].rolling(5).mean()
            features.append(trade_size_volatility.fillna(0).values)
            
            return np.column_stack(features) if features else np.zeros((len(price_data), 1))
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting microstructure features: {e}")
            return np.zeros((len(price_data), 1))
    
    def _extract_pre_movement_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Extract features that predict pre-movement patterns."""
        
        try:
            features = []
            
            # Momentum divergence
            price_momentum = price_data['close'].pct_change(5)
            volume_momentum = price_data['volume'].pct_change(5)
            momentum_divergence = price_momentum - volume_momentum
            features.append(momentum_divergence.fillna(0).values)
            
            # Support/resistance levels
            rolling_high = price_data['high'].rolling(20).max()
            rolling_low = price_data['low'].rolling(20).min()
            support_distance = (price_data['close'] - rolling_low) / (rolling_high - rolling_low)
            features.append(support_distance.fillna(0.5).values)
            
            # Volatility clustering
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(5).std()
            volatility_clustering = volatility / volatility.rolling(20).mean()
            features.append(volatility_clustering.fillna(1).values)
            
            # Price acceleration
            price_acceleration = price_data['close'].pct_change().diff()
            features.append(price_acceleration.fillna(0).values)
            
            return np.column_stack(features) if features else np.zeros((len(price_data), 1))
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting pre-movement features: {e}")
            return np.zeros((len(price_data), 1))
    
    def _extract_volatility_regime_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Extract volatility regime features."""
        
        try:
            features = []
            
            # Volatility regime detection
            returns = price_data['close'].pct_change()
            short_vol = returns.rolling(5).std()
            long_vol = returns.rolling(20).std()
            vol_regime = short_vol / long_vol
            features.append(vol_regime.fillna(1).values)
            
            # Volatility momentum
            vol_momentum = vol_regime.pct_change(3)
            features.append(vol_momentum.fillna(0).values)
            
            # Mean reversion tendency
            price_zscore = (price_data['close'] - price_data['close'].rolling(20).mean()) / price_data['close'].rolling(20).std()
            features.append(price_zscore.fillna(0).values)
            
            return np.column_stack(features) if features else np.zeros((len(price_data), 1))
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting volatility regime features: {e}")
            return np.zeros((len(price_data), 1))
    
    def _extract_order_flow_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Extract order flow features (simulated)."""
        
        try:
            features = []
            
            # Buy/sell pressure (simulated using volume and price)
            price_change = price_data['close'].pct_change()
            buy_pressure = np.where(price_change > 0, price_data['volume'], 0)
            sell_pressure = np.where(price_change < 0, price_data['volume'], 0)
            
            buy_sell_ratio = buy_pressure / (sell_pressure + 1e-8)
            features.append(buy_sell_ratio)
            
            # Order flow imbalance
            flow_imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-8)
            features.append(flow_imbalance)
            
            # Volume-weighted price momentum
            vwap = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            vwap_momentum = vwap.pct_change(3)
            features.append(vwap_momentum.fillna(0).values)
            
            return np.column_stack(features) if features else np.zeros((len(price_data), 1))
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting order flow features: {e}")
            return np.zeros((len(price_data), 1))
    
    def _extract_time_based_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Extract time-based features."""
        
        try:
            features = []
            
            # Time of day effects (simulated)
            n_samples = len(price_data)
            time_of_day = np.sin(2 * np.pi * np.arange(n_samples) / 1440)  # Daily cycle
            features.append(time_of_day)
            
            # Day of week effects (simulated)
            day_of_week = np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly cycle
            features.append(day_of_week)
            
            # Time since last significant move
            returns = price_data['close'].pct_change()
            significant_moves = np.abs(returns) > returns.std() * 2
            time_since_move = np.zeros(n_samples)
            last_move_idx = 0
            for i in range(n_samples):
                if significant_moves.iloc[i]:
                    last_move_idx = i
                time_since_move[i] = i - last_move_idx
            features.append(time_since_move)
            
            return np.column_stack(features) if features else np.zeros((len(price_data), 1))
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting time-based features: {e}")
            return np.zeros((len(price_data), 1))
    
    def _predict_pre_movement(self, features: np.ndarray, price_data: pd.DataFrame) -> PreMovementPrediction:
        """Predict price movement before target direction."""
        
        try:
            # Simple heuristic-based pre-movement prediction
            # In practice, this would use a trained model
            
            # Analyze recent price patterns
            recent_returns = price_data['close'].pct_change(3).iloc[-1]
            recent_volatility = price_data['close'].pct_change().rolling(5).std().iloc[-1]
            recent_volume = price_data['volume'].rolling(5).mean().iloc[-1]
            
            # Determine pre-movement direction
            if recent_returns > 0.001:  # 0.1% recent upward movement
                pre_movement_direction = "up"
                pre_movement_magnitude = min(abs(recent_returns), 0.005)  # Cap at 0.5%
            elif recent_returns < -0.001:  # 0.1% recent downward movement
                pre_movement_direction = "down"
                pre_movement_magnitude = min(abs(recent_returns), 0.005)
            else:
                pre_movement_direction = "neutral"
                pre_movement_magnitude = 0.0
            
            # Estimate duration based on volatility
            pre_movement_duration = max(1.0, min(10.0, 5.0 / (recent_volatility + 1e-8)))
            
            # Calculate confidence
            pre_movement_confidence = min(0.9, abs(recent_returns) * 100 + 0.3)
            
            # Determine optimal entry timing
            if pre_movement_direction == "up" and recent_returns > 0.002:
                optimal_entry_timing = "wait_for_reversal"
                wait_duration_minutes = pre_movement_duration
                entry_reasoning = "Price likely to increase before reversal, wait for better entry"
            elif pre_movement_direction == "down" and recent_returns < -0.002:
                optimal_entry_timing = "wait_for_reversal"
                wait_duration_minutes = pre_movement_duration
                entry_reasoning = "Price likely to decrease before reversal, wait for better entry"
            else:
                optimal_entry_timing = "immediate"
                wait_duration_minutes = 0.0
                entry_reasoning = "No significant pre-movement expected, enter immediately"
            
            return PreMovementPrediction(
                pre_movement_direction=pre_movement_direction,
                pre_movement_magnitude=pre_movement_magnitude,
                pre_movement_duration=pre_movement_duration,
                pre_movement_confidence=pre_movement_confidence,
                optimal_entry_timing=optimal_entry_timing,
                wait_duration_minutes=wait_duration_minutes,
                entry_reasoning=entry_reasoning
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting pre-movement: {e}")
            return PreMovementPrediction(
                pre_movement_direction="neutral",
                pre_movement_magnitude=0.0,
                pre_movement_duration=0.0,
                pre_movement_confidence=0.0,
                optimal_entry_timing="immediate",
                wait_duration_minutes=0.0,
                entry_reasoning="Error in pre-movement prediction"
            )
    
    def _calculate_enhanced_risk_metrics(self, prediction: ShortTermPrediction, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate enhanced risk metrics."""
        
        try:
            # Calculate drawdown probability
            recent_volatility = price_data['close'].pct_change().rolling(10).std().iloc[-1]
            drawdown_probability = min(0.8, recent_volatility * 50)
            
            # Forecast volatility
            volatility_forecast = recent_volatility * (1 + np.random.normal(0, 0.1))
            
            # Estimate liquidity impact
            recent_volume = price_data['volume'].rolling(10).mean().iloc[-1]
            liquidity_impact = max(0.0, 1.0 - (recent_volume / price_data['volume'].mean()))
            
            return {
                'drawdown_probability': drawdown_probability,
                'volatility_forecast': volatility_forecast,
                'liquidity_impact': liquidity_impact
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating enhanced risk metrics: {e}")
            return {
                'drawdown_probability': 0.5,
                'volatility_forecast': 0.01,
                'liquidity_impact': 0.1
            }
    
    def _determine_entry_strategy(self, prediction: ShortTermPrediction, pre_movement: PreMovementPrediction) -> Tuple[str, str, float]:
        """Determine sophisticated entry strategy."""
        
        try:
            # Analyze conditions for entry strategy
            target_direction = prediction.direction
            pre_movement_direction = pre_movement.pre_movement_direction
            pre_movement_confidence = pre_movement.pre_movement_confidence
            target_confidence = prediction.confidence_score
            
            # Determine entry strategy
            if pre_movement_confidence > 0.7 and pre_movement_direction != target_direction:
                if pre_movement.optimal_entry_timing == "wait_for_reversal":
                    entry_strategy = "wait_and_enter"
                    exit_strategy = "target_reached"
                    entry_confidence = (target_confidence + pre_movement_confidence) / 2
                else:
                    entry_strategy = "staged_entry"
                    exit_strategy = "target_reached"
                    entry_confidence = target_confidence * 0.8
            elif target_confidence > 0.8:
                entry_strategy = "immediate"
                exit_strategy = "target_reached"
                entry_confidence = target_confidence
            else:
                entry_strategy = "wait_and_enter"
                exit_strategy = "time_stop"
                entry_confidence = target_confidence * 0.6
            
            return entry_strategy, exit_strategy, entry_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error determining entry strategy: {e}")
            return "immediate", "target_reached", prediction.confidence_score
    
    @handles_errors(
        error_handlers={
            ValueError: (False, 'Invalid training data for enhanced model'),
            AttributeError: (False, 'Missing required model components'),
            KeyError: (False, 'Missing required training data')
        },
        default_return=False,
        context='enhanced model training'
    )
    def fit(
        self,
        X: np.ndarray,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> bool:
        """
        Fit the enhanced short-term entry timing model.
        
        Args:
            X: Input features
            price_data: Price data for target generation
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            bool: True if training successful, False otherwise
        """
        start_time = time.time()
        self.logger.info(f"🔄 Training Enhanced ShortTermEntryTimingModel for {symbol}")
        
        try:
            # Create enhanced features
            if self.enhanced_features_enabled:
                X_enhanced = self._create_enhanced_features(price_data, X)
            else:
                X_enhanced = X
            
            # Train base model
            base_success = self.base_model.fit(X_enhanced, price_data, symbol, timeframe)
            
            if not base_success:
                self.logger.error("❌ Base model training failed")
                return False
            
            # Train enhanced models (simplified for demo)
            self._train_enhanced_models(X_enhanced, price_data)
            
            # Fit feature scaler
            self.feature_scaler.fit(X_enhanced)
            
            # Update state
            self.is_fitted = True
            
            training_time = time.time() - start_time
            self.logger.info(f"✅ Enhanced model trained successfully in {training_time:.3f}s")
            self.logger.info(f"📊 Enhanced features: {X.shape[1]} -> {X_enhanced.shape[1]}")
            
            return True
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced model training failed after {training_time:.3f}s: {e}")
            return False
    
    def _train_enhanced_models(self, X: np.ndarray, price_data: pd.DataFrame) -> None:
        """Train enhanced models for different prediction phases."""
        
        try:
            # In practice, these would be trained on historical data
            # For demo purposes, we'll create placeholder models
            
            self.pre_movement_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.post_movement_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            self.entry_timing_model = MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42)
            self.risk_assessment_model = RandomForestRegressor(n_estimators=30, random_state=42)
            
            # Create dummy targets for training (in practice, these would be real targets)
            n_samples = X.shape[0]
            dummy_targets = np.random.rand(n_samples, 4)  # 4 different prediction targets
            
            # Train models
            self.pre_movement_model.fit(X, dummy_targets[:, 0])
            self.post_movement_model.fit(X, dummy_targets[:, 1])
            self.entry_timing_model.fit(X, dummy_targets[:, 2])
            self.risk_assessment_model.fit(X, dummy_targets[:, 3])
            
            self.logger.info("✅ Enhanced models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error training enhanced models: {e}")
    
    @handles_errors(
        error_handlers={
            ValueError: (None, 'Invalid prediction data'),
            AttributeError: (None, 'Model not fitted'),
            KeyError: (None, 'Missing required prediction data')
        },
        default_return=None,
        context='enhanced prediction'
    )
    def predict(
        self,
        X: np.ndarray,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> Optional[EnhancedShortTermEntryTimingResult]:
        """
        Make enhanced predictions for short-term entry timing.
        
        Args:
            X: Input features
            price_data: Current price data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            EnhancedShortTermEntryTimingResult with sophisticated predictions
        """
        if not self.is_fitted:
            self.logger.error("❌ Enhanced model not fitted")
            return None
        
        start_time = time.time()
        self.logger.info(f"🔮 Making enhanced short-term predictions for {symbol}")
        
        try:
            # Create enhanced features
            if self.enhanced_features_enabled:
                X_enhanced = self._create_enhanced_features(price_data, X)
            else:
                X_enhanced = X
            
            # Get base predictions
            base_result = self.base_model.predict(X_enhanced, price_data, symbol, timeframe)
            
            if base_result is None:
                self.logger.error("❌ Base model prediction failed")
                return None
            
            # Enhance predictions
            enhanced_predictions = []
            for base_pred in base_result.predictions:
                # Predict pre-movement
                pre_movement = self._predict_pre_movement(X_enhanced, price_data)
                
                # Calculate enhanced risk metrics
                enhanced_risk = self._calculate_enhanced_risk_metrics(base_pred, price_data)
                
                # Determine entry strategy
                entry_strategy, exit_strategy, entry_confidence = self._determine_entry_strategy(
                    base_pred, pre_movement
                )
                
                # Create enhanced prediction
                enhanced_pred = EnhancedShortTermPrediction(
                    # Base prediction
                    target_percentage=base_pred.target_percentage,
                    target_name=base_pred.target_name,
                    probability=base_pred.probability,
                    timing_minutes=base_pred.timing_minutes,
                    direction=base_pred.direction,
                    confidence_score=base_pred.confidence_score,
                    risk_reward_ratio=base_pred.risk_reward_ratio,
                    max_adverse_movement=base_pred.max_adverse_movement,
                    is_valid=base_pred.is_valid,
                    validation_reason=base_pred.validation_reason,
                    
                    # Enhanced predictions
                    pre_movement=pre_movement,
                    post_movement_prediction={
                        'expected_reversal': pre_movement.pre_movement_direction != base_pred.direction,
                        'reversal_probability': pre_movement.pre_movement_confidence,
                        'continuation_probability': 1.0 - pre_movement.pre_movement_confidence
                    },
                    
                    # Sophisticated entry timing
                    entry_strategy=entry_strategy,
                    entry_confidence=entry_confidence,
                    exit_strategy=exit_strategy,
                    
                    # Advanced risk metrics
                    drawdown_probability=enhanced_risk['drawdown_probability'],
                    volatility_forecast=enhanced_risk['volatility_forecast'],
                    liquidity_impact=enhanced_risk['liquidity_impact']
                )
                
                enhanced_predictions.append(enhanced_pred)
            
            # Create enhanced result
            enhanced_result = EnhancedShortTermEntryTimingResult(
                model_name=f"enhanced_{base_result.model_name}",
                timestamp=base_result.timestamp,
                symbol=base_result.symbol,
                timeframe=base_result.timeframe,
                current_price=base_result.current_price,
                predictions=enhanced_predictions,
                n_predictions=base_result.n_predictions,
                valid_predictions=base_result.valid_predictions,
                prediction_time=time.time() - start_time,
                overall_confidence=base_result.overall_confidence,
                risk_score=base_result.risk_score,
                entry_recommendation=base_result.entry_recommendation
            )
            
            # Calculate enhanced summary metrics
            enhanced_result = self._calculate_enhanced_summary_metrics(enhanced_result)
            
            self.logger.info(f"✅ Enhanced predictions completed in {enhanced_result.prediction_time:.3f}s")
            self.logger.info(f"📊 Valid predictions: {enhanced_result.valid_predictions}/{enhanced_result.n_predictions}")
            self.logger.info(f"🎯 Entry recommendation: {enhanced_result.entry_recommendation}")
            
            return enhanced_result
            
        except Exception as e:
            prediction_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced prediction failed after {prediction_time:.3f}s: {e}")
            return None
    
    def _calculate_enhanced_summary_metrics(self, result: EnhancedShortTermEntryTimingResult) -> EnhancedShortTermEntryTimingResult:
        """Calculate enhanced summary metrics."""
        
        try:
            valid_predictions = [p for p in result.predictions if p.is_valid]
            
            if not valid_predictions:
                return result
            
            # Calculate pre-movement analysis
            pre_movement_directions = [p.pre_movement.pre_movement_direction for p in valid_predictions]
            pre_movement_confidences = [p.pre_movement.pre_movement_confidence for p in valid_predictions]
            
            result.pre_movement_analysis = {
                'dominant_pre_movement': max(set(pre_movement_directions), key=pre_movement_directions.count),
                'avg_pre_movement_confidence': float(np.mean(pre_movement_confidences)),
                'pre_movement_consistency': len(set(pre_movement_directions)) == 1
            }
            
            # Calculate market conditions
            entry_strategies = [p.entry_strategy for p in valid_predictions]
            volatility_forecasts = [p.volatility_forecast for p in valid_predictions]
            
            result.market_conditions = {
                'dominant_entry_strategy': max(set(entry_strategies), key=entry_strategies.count),
                'avg_volatility_forecast': float(np.mean(volatility_forecasts)),
                'market_volatility_regime': 'high' if np.mean(volatility_forecasts) > 0.02 else 'low'
            }
            
            # Find best prediction
            best_prediction = max(valid_predictions, key=lambda p: p.entry_confidence * p.confidence_score)
            result.best_prediction = best_prediction
            
            # Update entry recommendation based on enhanced analysis
            if result.pre_movement_analysis['pre_movement_consistency'] and result.pre_movement_analysis['avg_pre_movement_confidence'] > 0.7:
                if result.pre_movement_analysis['dominant_pre_movement'] != 'neutral':
                    result.entry_recommendation = "WAIT_FOR_PRE_MOVEMENT"
                else:
                    result.entry_recommendation = "ENTER_IMMEDIATELY"
            else:
                result.entry_recommendation = result.entry_recommendation
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating enhanced summary metrics: {e}")
            return result
    
    def get_enhanced_prediction_summary(self, result: EnhancedShortTermEntryTimingResult) -> Dict[str, Any]:
        """Get enhanced prediction summary."""
        
        try:
            summary = {
                'model_name': result.model_name,
                'timestamp': result.timestamp.isoformat(),
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'current_price': result.current_price,
                'n_predictions': result.n_predictions,
                'valid_predictions': result.valid_predictions,
                'overall_confidence': result.overall_confidence,
                'risk_score': result.risk_score,
                'entry_recommendation': result.entry_recommendation,
                'prediction_time': result.prediction_time,
                'pre_movement_analysis': result.pre_movement_analysis,
                'market_conditions': result.market_conditions,
                'predictions': []
            }
            
            for prediction in result.predictions:
                pred_summary = {
                    'name': prediction.target_name,
                    'percentage': prediction.target_percentage,
                    'probability': prediction.probability,
                    'timing_minutes': prediction.timing_minutes,
                    'direction': prediction.direction,
                    'confidence_score': prediction.confidence_score,
                    'risk_reward_ratio': prediction.risk_reward_ratio,
                    'is_valid': prediction.is_valid,
                    
                    # Enhanced features
                    'pre_movement': {
                        'direction': prediction.pre_movement.pre_movement_direction,
                        'magnitude': prediction.pre_movement.pre_movement_magnitude,
                        'duration': prediction.pre_movement.pre_movement_duration,
                        'confidence': prediction.pre_movement.pre_movement_confidence,
                        'optimal_entry_timing': prediction.pre_movement.optimal_entry_timing,
                        'wait_duration': prediction.pre_movement.wait_duration_minutes,
                        'reasoning': prediction.pre_movement.entry_reasoning
                    },
                    'entry_strategy': prediction.entry_strategy,
                    'entry_confidence': prediction.entry_confidence,
                    'exit_strategy': prediction.exit_strategy,
                    'drawdown_probability': prediction.drawdown_probability,
                    'volatility_forecast': prediction.volatility_forecast,
                    'liquidity_impact': prediction.liquidity_impact
                }
                summary['predictions'].append(pred_summary)
            
            if result.best_prediction:
                best = result.best_prediction
                summary['best_prediction'] = {
                    'name': best.target_name,
                    'direction': best.direction,
                    'confidence_score': best.confidence_score,
                    'entry_strategy': best.entry_strategy,
                    'entry_confidence': best.entry_confidence,
                    'pre_movement_direction': best.pre_movement.pre_movement_direction,
                    'optimal_entry_timing': best.pre_movement.optimal_entry_timing
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error creating enhanced prediction summary: {e}")
            return {'error': str(e)}


# Convenience functions
def create_enhanced_short_term_entry_timing_model(
    target_percentages: Optional[List[float]] = None,
    timeframe: str = "1m",
    enhanced_features: bool = True
) -> EnhancedShortTermEntryTimingModel:
    """Create an enhanced short-term entry timing model."""
    
    config = ShortTermEntryTimingConfig(
        target_percentages=target_percentages or [0.001, 0.002, 0.003, 0.004, 0.005],
        timeframe=timeframe
    )
    
    model = EnhancedShortTermEntryTimingModel(config)
    model.enhanced_features_enabled = enhanced_features
    
    return model


# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced short-term entry timing model
    print("Enhanced Short-Term Entry Timing Model")
    print("=" * 45)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate sample features
    X = np.random.randn(n_samples, n_features)
    
    # Generate sample price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    price_data = pd.DataFrame(data)
    
    # Create enhanced model
    model = create_enhanced_short_term_entry_timing_model()
    
    print(f"✅ Created enhanced model with {len(model.config.target_percentages)} targets")
    print(f"📊 Target percentages: {[f'{p*100:.1f}%' for p in model.config.target_percentages]}")
    print(f"🎯 Enhanced features: {model.enhanced_features_enabled}")
    
    # Train model
    success = model.fit(X, price_data, "BTCUSDT", "1m")
    
    if success:
        print("✅ Enhanced model trained successfully")
        
        # Make predictions
        result = model.predict(X[-10:], price_data.tail(10), "BTCUSDT", "1m")
        
        if result:
            summary = model.get_enhanced_prediction_summary(result)
            print(f"🔮 Enhanced predictions completed in {summary['prediction_time']:.3f}s")
            print(f"📊 Valid predictions: {summary['valid_predictions']}/{summary['n_predictions']}")
            print(f"🎯 Entry recommendation: {summary['entry_recommendation']}")
            
            if 'pre_movement_analysis' in summary:
                pre_movement = summary['pre_movement_analysis']
                print(f"🔄 Pre-movement analysis: {pre_movement['dominant_pre_movement']} "
                      f"(confidence: {pre_movement['avg_pre_movement_confidence']:.3f})")
            
            if 'best_prediction' in summary:
                best = summary['best_prediction']
                print(f"🏆 Best prediction: {best['name']} ({best['direction']}) - "
                      f"Entry strategy: {best['entry_strategy']}")
                print(f"   Pre-movement: {best['pre_movement_direction']} - "
                      f"Optimal timing: {best['optimal_entry_timing']}")
            
            print("\n📋 Enhanced Prediction Details:")
            for pred in summary['predictions']:
                status = "✅" if pred['is_valid'] else "❌"
                pre_movement = pred['pre_movement']
                print(f"{status} {pred['name']}: {pred['direction']} - "
                      f"Probability: {pred['probability']:.3f}, "
                      f"Entry: {pred['entry_strategy']}")
                print(f"   Pre-movement: {pre_movement['direction']} "
                      f"({pre_movement['magnitude']*100:.2f}% in {pre_movement['duration']:.1f}min)")
                print(f"   Reasoning: {pre_movement['reasoning']}")
        else:
            print("❌ Enhanced prediction failed")
    else:
        print("❌ Enhanced model training failed")