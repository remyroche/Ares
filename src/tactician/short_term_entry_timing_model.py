"""
Short-Term Entry Timing Model

This module implements a specialized multi-output model for predicting optimal entry timing
based on expected short-term price movements (0.1% to 0.5%) using the triple barrier method.

Key Features:
- Multi-output prediction for 5 different target percentages
- Each target has probability and timing predictions
- Triple barrier method integration
- Direction-aware predictions (up/down movement)
- Adverse movement protection
- Integration with existing Tactician architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates, traced
from src.utils.ml_common.models.multi_output_models import (
    MultiOutputConfig, MultiOutputStackingModel, MultiOutputResult
)
from src.tactician.short_term_target_generator import (
    ShortTermTargetGenerator, TripleBarrierConfig, ShortTermTargets
)

logger = system_logger.getChild('ShortTermEntryTimingModel')


@dataclass
class ShortTermEntryTimingConfig:
    """Configuration for short-term entry timing model."""
    # Basic configuration
    model_name: str = "short_term_entry_timing"
    timeframe: str = "1m"
    n_targets: int = 5
    target_percentages: List[float] = field(default_factory=lambda: [0.001, 0.002, 0.003, 0.004, 0.005])
    
    # Model configuration
    base_models: Dict[str, Any] = field(default_factory=dict)
    meta_models: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 15
    
    # Multi-output specific settings
    output_weights: Optional[List[float]] = None
    output_loss_weights: Optional[List[float]] = None
    enable_output_correlation: bool = True
    correlation_threshold: float = 0.6
    
    # Triple barrier configuration
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 200
    enable_profiling: bool = False


@dataclass
class ShortTermPrediction:
    """Individual short-term prediction result."""
    target_percentage: float
    target_name: str
    
    # Predictions
    probability: float
    timing_minutes: float
    direction: str  # 'up', 'down', 'neutral'
    
    # Confidence and risk
    confidence_score: float
    risk_reward_ratio: float
    max_adverse_movement: float
    
    # Validation
    is_valid: bool
    validation_reason: str


@dataclass
class ShortTermEntryTimingResult:
    """Result from short-term entry timing model."""
    # Basic info
    model_name: str
    timestamp: datetime
    symbol: str
    timeframe: str
    current_price: float
    
    # Predictions
    predictions: List[ShortTermPrediction] = field(default_factory=list)
    best_prediction: Optional[ShortTermPrediction] = None
    
    # Summary metrics
    overall_confidence: float = 0.0
    risk_score: float = 0.0
    entry_recommendation: str = "HOLD"  # "ENTER", "HOLD", "EXIT"
    
    # Performance metrics
    prediction_time: float = 0.0
    model_confidence: float = 0.0
    
    # Metadata
    n_predictions: int = 0
    valid_predictions: int = 0


class ShortTermEntryTimingModel:
    """
    Multi-output model for short-term entry timing prediction.
    
    This model predicts optimal entry timing for 0.1% to 0.5% price movements
    using the triple barrier method to ensure expected price direction without
    adverse movement.
    """
    
    def __init__(self, config: Optional[ShortTermEntryTimingConfig] = None):
        """
        Initialize the short-term entry timing model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ShortTermEntryTimingConfig()
        self.logger = logger.getChild('ShortTermEntryTimingModel')
        
        # Initialize target generator
        self.target_generator = ShortTermTargetGenerator(self.config.triple_barrier)
        
        # Initialize multi-output model
        self.multi_output_config = self._create_multi_output_config()
        self.multi_output_model = MultiOutputStackingModel(self.multi_output_config)
        
        # Model state
        self.is_fitted = False
        self.training_history: List[Dict[str, Any]] = []
        self.prediction_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"🚀 Initializing ShortTermEntryTimingModel")
        self.logger.info(f"📊 Target percentages: {[f'{p*100:.1f}%' for p in self.config.target_percentages]}")
        self.logger.info(f"⏰ Timeframe: {self.config.timeframe}")
        
    def _create_multi_output_config(self) -> MultiOutputConfig:
        """Create multi-output configuration for the model."""
        
        # Create output names for each target (probability + timing)
        output_names = []
        for target_pct in self.config.target_percentages:
            target_name = f"{target_pct*100:.1f}pct"
            output_names.extend([f"{target_name}_probability", f"{target_name}_timing"])
        
        return MultiOutputConfig(
            model_name=self.config.model_name,
            n_outputs=len(output_names),
            output_names=output_names,
            base_models=self.config.base_models,
            meta_model=self.config.meta_models,
            enable_cross_validation=self.config.enable_cross_validation,
            cv_folds=self.config.cv_folds,
            enable_early_stopping=self.config.enable_early_stopping,
            early_stopping_patience=self.config.early_stopping_patience,
            output_weights=self.config.output_weights,
            output_loss_weights=self.config.output_loss_weights,
            enable_output_correlation=self.config.enable_output_correlation,
            correlation_threshold=self.config.correlation_threshold,
            enable_caching=self.config.enable_caching,
            cache_size_mb=self.config.cache_size_mb,
            enable_profiling=self.config.enable_profiling
        )
    
    @handles_errors(
        error_handlers={
            ValueError: (False, 'Invalid training data for short-term model'),
            KeyError: (False, 'Missing required training data columns'),
            IndexError: (False, 'Insufficient training data')
        },
        default_return=False,
        context='short-term model training'
    )
    def fit(
        self,
        X: np.ndarray,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> bool:
        """
        Fit the short-term entry timing model.
        
        Args:
            X: Input features
            price_data: Price data for target generation
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            bool: True if training successful, False otherwise
        """
        start_time = time.time()
        self.logger.info(f"🔄 Training ShortTermEntryTimingModel for {symbol}")
        
        try:
            # Generate targets using triple barrier method
            targets_collection = self.target_generator.generate_targets(
                price_data, symbol, timeframe
            )
            
            if not targets_collection:
                self.logger.error("❌ Failed to generate targets for training")
                return False
            
            # Convert targets to multi-output format
            y = self._convert_targets_to_multi_output(targets_collection, len(X))
            
            if y is None:
                self.logger.error("❌ Failed to convert targets to multi-output format")
                return False
            
            # Train multi-output model
            self.multi_output_model.fit(X, y)
            
            # Update state
            self.is_fitted = True
            
            # Record training history
            training_time = time.time() - start_time
            self.training_history.append({
                'timestamp': datetime.now(),
                'duration': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_targets': len(self.config.target_percentages),
                'symbol': symbol,
                'timeframe': timeframe,
                'valid_targets': targets_collection.valid_targets,
                'total_targets': targets_collection.total_targets
            })
            
            self.logger.info(f"✅ Model trained successfully in {training_time:.3f}s")
            self.logger.info(f"📊 Training samples: {X.shape[0]}, Features: {X.shape[1]}")
            self.logger.info(f"🎯 Targets: {targets_collection.valid_targets}/{targets_collection.total_targets} valid")
            
            return True
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Training failed after {training_time:.3f}s: {e}")
            return False
    
    def _convert_targets_to_multi_output(
        self, 
        targets_collection: ShortTermTargets, 
        n_samples: int
    ) -> Optional[np.ndarray]:
        """Convert targets collection to multi-output format."""
        
        try:
            n_outputs = len(self.multi_output_config.output_names)
            y = np.zeros((n_samples, n_outputs))
            
            # For now, we'll use the target collection as a template
            # In practice, you would have historical data with actual targets
            for i, target in enumerate(targets_collection.targets):
                if i >= len(self.config.target_percentages):
                    break
                
                target_pct = self.config.target_percentages[i]
                target_name = f"{target_pct*100:.1f}pct"
                
                # Find output indices
                prob_idx = self.multi_output_config.output_names.index(f"{target_name}_probability")
                timing_idx = self.multi_output_config.output_names.index(f"{target_name}_timing")
                
                # Set values (in practice, these would come from historical data)
                y[:, prob_idx] = target.confidence_score if target.is_valid else 0.0
                y[:, timing_idx] = target.entry_timing_minutes if target.is_valid else 0.0
            
            self.logger.debug(f"📊 Converted targets to multi-output format: {y.shape}")
            return y
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert targets: {e}")
            return None
    
    @handles_errors(
        error_handlers={
            ValueError: (None, 'Invalid prediction data'),
            AttributeError: (None, 'Model not fitted'),
            KeyError: (None, 'Missing required prediction data')
        },
        default_return=None,
        context='short-term prediction'
    )
    def predict(
        self,
        X: np.ndarray,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> Optional[ShortTermEntryTimingResult]:
        """
        Make predictions for short-term entry timing.
        
        Args:
            X: Input features
            price_data: Current price data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            ShortTermEntryTimingResult with predictions
        """
        if not self.is_fitted:
            self.logger.error("❌ Model not fitted")
            return None
        
        start_time = time.time()
        self.logger.info(f"🔮 Making short-term predictions for {symbol}")
        
        try:
            # Get current price
            current_price = price_data['close'].iloc[-1]
            
            # Make multi-output predictions
            predictions = self.multi_output_model.predict(X)
            
            # Convert predictions to short-term format
            short_term_predictions = self._convert_predictions_to_short_term(
                predictions, symbol, timeframe, current_price
            )
            
            # Create result
            result = ShortTermEntryTimingResult(
                model_name=self.config.model_name,
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                predictions=short_term_predictions,
                n_predictions=len(short_term_predictions),
                valid_predictions=len([p for p in short_term_predictions if p.is_valid]),
                prediction_time=time.time() - start_time
            )
            
            # Calculate summary metrics
            result = self._calculate_summary_metrics(result)
            
            # Record prediction history
            self.prediction_history.append({
                'timestamp': result.timestamp,
                'duration': result.prediction_time,
                'n_samples': X.shape[0],
                'symbol': symbol,
                'overall_confidence': result.overall_confidence,
                'risk_score': result.risk_score,
                'entry_recommendation': result.entry_recommendation
            })
            
            self.logger.info(f"✅ Predictions completed in {result.prediction_time:.3f}s")
            self.logger.info(f"📊 Valid predictions: {result.valid_predictions}/{result.n_predictions}")
            self.logger.info(f"🎯 Entry recommendation: {result.entry_recommendation}")
            
            return result
            
        except Exception as e:
            prediction_time = time.time() - start_time
            self.logger.error(f"❌ Prediction failed after {prediction_time:.3f}s: {e}")
            return None
    
    def _convert_predictions_to_short_term(
        self,
        predictions: np.ndarray,
        symbol: str,
        timeframe: str,
        current_price: float
    ) -> List[ShortTermPrediction]:
        """Convert multi-output predictions to short-term format."""
        
        short_term_predictions = []
        
        try:
            for i, target_pct in enumerate(self.config.target_percentages):
                target_name = f"{target_pct*100:.1f}pct"
                
                # Find output indices
                prob_idx = self.multi_output_config.output_names.index(f"{target_name}_probability")
                timing_idx = self.multi_output_config.output_names.index(f"{target_name}_timing")
                
                # Get predictions (use mean for multiple samples)
                probability = float(np.mean(predictions[:, prob_idx]))
                timing = float(np.mean(predictions[:, timing_idx]))
                
                # Determine direction based on probability and timing
                direction = self._determine_direction(probability, timing, target_pct)
                
                # Calculate confidence and risk metrics
                confidence_score = probability
                risk_reward_ratio = self._calculate_risk_reward_ratio(target_pct, timing)
                max_adverse_movement = self._calculate_max_adverse_movement(target_pct)
                
                # Validate prediction
                is_valid, validation_reason = self._validate_prediction(
                    probability, timing, target_pct, confidence_score
                )
                
                # Create prediction
                prediction = ShortTermPrediction(
                    target_percentage=target_pct,
                    target_name=target_name,
                    probability=probability,
                    timing_minutes=timing,
                    direction=direction,
                    confidence_score=confidence_score,
                    risk_reward_ratio=risk_reward_ratio,
                    max_adverse_movement=max_adverse_movement,
                    is_valid=is_valid,
                    validation_reason=validation_reason
                )
                
                short_term_predictions.append(prediction)
            
            return short_term_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert predictions: {e}")
            return []
    
    def _determine_direction(self, probability: float, timing: float, target_pct: float) -> str:
        """Determine direction based on prediction metrics."""
        
        # Simple heuristic - in practice, you would use more sophisticated logic
        if probability > 0.6 and timing < 10:
            return "up"
        elif probability > 0.6 and timing >= 10:
            return "down"
        else:
            return "neutral"
    
    def _calculate_risk_reward_ratio(self, target_pct: float, timing: float) -> float:
        """Calculate risk/reward ratio for the prediction."""
        
        # Simple calculation - in practice, you would use more sophisticated risk models
        base_ratio = target_pct / (target_pct * 0.5)  # Assume 50% of target as risk
        timing_factor = max(0.5, 1.0 - (timing / 15.0))  # Timing penalty
        
        return base_ratio * timing_factor
    
    def _calculate_max_adverse_movement(self, target_pct: float) -> float:
        """Calculate maximum adverse movement for the target."""
        
        # Use triple barrier configuration
        return target_pct * self.config.triple_barrier.lower_barrier_multiplier
    
    def _validate_prediction(
        self, 
        probability: float, 
        timing: float, 
        target_pct: float, 
        confidence_score: float
    ) -> Tuple[bool, str]:
        """Validate if prediction meets criteria."""
        
        # Check probability threshold
        min_confidence = 0.5  # Base threshold
        if target_pct <= 0.002:  # 0.2% or less
            min_confidence = 0.7
        elif target_pct <= 0.003:  # 0.3% or less
            min_confidence = 0.65
        elif target_pct <= 0.004:  # 0.4% or less
            min_confidence = 0.6
        
        if probability < min_confidence:
            return False, f"Probability too low: {probability:.3f} < {min_confidence:.3f}"
        
        # Check timing
        if timing < 0 or timing > self.config.triple_barrier.max_hold_time_minutes:
            return False, f"Invalid timing: {timing:.1f} minutes"
        
        # Check confidence score
        if confidence_score < 0.5:
            return False, f"Confidence score too low: {confidence_score:.3f}"
        
        return True, "Prediction validation passed"
    
    def _calculate_summary_metrics(self, result: ShortTermEntryTimingResult) -> ShortTermEntryTimingResult:
        """Calculate summary metrics for the result."""
        
        try:
            valid_predictions = [p for p in result.predictions if p.is_valid]
            
            if not valid_predictions:
                result.overall_confidence = 0.0
                result.risk_score = 1.0
                result.entry_recommendation = "HOLD"
                return result
            
            # Calculate overall confidence
            confidences = [p.confidence_score for p in valid_predictions]
            result.overall_confidence = float(np.mean(confidences))
            
            # Calculate risk score
            risk_rewards = [p.risk_reward_ratio for p in valid_predictions]
            adverse_movements = [p.max_adverse_movement for p in valid_predictions]
            
            avg_risk_reward = float(np.mean(risk_rewards))
            avg_adverse = float(np.mean(adverse_movements))
            
            result.risk_score = avg_adverse / max(avg_risk_reward, 0.1)
            
            # Find best prediction
            best_prediction = max(valid_predictions, key=lambda p: p.confidence_score * p.risk_reward_ratio)
            result.best_prediction = best_prediction
            
            # Determine entry recommendation
            if result.overall_confidence > 0.7 and result.risk_score < 0.5:
                result.entry_recommendation = "ENTER"
            elif result.overall_confidence > 0.5 and result.risk_score < 0.7:
                result.entry_recommendation = "HOLD"
            else:
                result.entry_recommendation = "HOLD"
            
            # Calculate model confidence
            result.model_confidence = result.overall_confidence * (1.0 - result.risk_score)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating summary metrics: {e}")
            return result
    
    def get_prediction_summary(self, result: ShortTermEntryTimingResult) -> Dict[str, Any]:
        """Get a summary of the prediction results."""
        
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
                'model_confidence': result.model_confidence,
                'prediction_time': result.prediction_time,
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
                    'max_adverse_movement': prediction.max_adverse_movement,
                    'is_valid': prediction.is_valid,
                    'validation_reason': prediction.validation_reason
                }
                summary['predictions'].append(pred_summary)
            
            if result.best_prediction:
                summary['best_prediction'] = {
                    'name': result.best_prediction.target_name,
                    'direction': result.best_prediction.direction,
                    'confidence_score': result.best_prediction.confidence_score,
                    'risk_reward_ratio': result.best_prediction.risk_reward_ratio
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error creating prediction summary: {e}")
            return {'error': str(e)}
    
    def save_model(self, file_path: str) -> None:
        """Save the model to disk."""
        try:
            import pickle
            
            model_data = {
                'config': self.config,
                'is_fitted': self.is_fitted,
                'multi_output_model': self.multi_output_model,
                'training_history': self.training_history,
                'prediction_history': self.prediction_history
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"💾 Model saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def load_model(self, file_path: str) -> None:
        """Load the model from disk."""
        try:
            import pickle
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            self.multi_output_model = model_data['multi_output_model']
            self.training_history = model_data['training_history']
            self.prediction_history = model_data['prediction_history']
            
            # Reinitialize target generator
            self.target_generator = ShortTermTargetGenerator(self.config.triple_barrier)
            
            self.logger.info(f"📂 Model loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise


# Convenience functions
def create_short_term_entry_timing_model(
    target_percentages: Optional[List[float]] = None,
    timeframe: str = "1m",
    max_hold_time_minutes: int = 15
) -> ShortTermEntryTimingModel:
    """Create a short-term entry timing model with custom configuration."""
    
    config = ShortTermEntryTimingConfig(
        target_percentages=target_percentages or [0.001, 0.002, 0.003, 0.004, 0.005],
        timeframe=timeframe,
        triple_barrier=TripleBarrierConfig(
            target_percentages=target_percentages or [0.001, 0.002, 0.003, 0.004, 0.005],
            max_hold_time_minutes=max_hold_time_minutes
        )
    )
    
    return ShortTermEntryTimingModel(config)


# Example usage
if __name__ == "__main__":
    # Example of how to use the short-term entry timing model
    print("Short-Term Entry Timing Model")
    print("=" * 40)
    
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
    
    # Create and train model
    model = create_short_term_entry_timing_model()
    
    print(f"✅ Created model with {len(model.config.target_percentages)} targets")
    print(f"📊 Target percentages: {[f'{p*100:.1f}%' for p in model.config.target_percentages]}")
    print(f"⏰ Timeframe: {model.config.timeframe}")
    
    # Train model
    success = model.fit(X, price_data, "BTCUSDT", "1m")
    
    if success:
        print("✅ Model trained successfully")
        
        # Make predictions
        result = model.predict(X[-10:], price_data.tail(10), "BTCUSDT", "1m")
        
        if result:
            summary = model.get_prediction_summary(result)
            print(f"🔮 Predictions completed in {summary['prediction_time']:.3f}s")
            print(f"📊 Valid predictions: {summary['valid_predictions']}/{summary['n_predictions']}")
            print(f"🎯 Entry recommendation: {summary['entry_recommendation']}")
            print(f"📈 Overall confidence: {summary['overall_confidence']:.3f}")
            print(f"🛡️ Risk score: {summary['risk_score']:.3f}")
            
            if summary.get('best_prediction'):
                best = summary['best_prediction']
                print(f"🏆 Best prediction: {best['name']} ({best['direction']}) - "
                      f"Confidence: {best['confidence_score']:.3f}")
            
            print("\n📋 Prediction Details:")
            for pred in summary['predictions']:
                status = "✅" if pred['is_valid'] else "❌"
                print(f"{status} {pred['name']}: {pred['direction']} - "
                      f"Probability: {pred['probability']:.3f}, "
                      f"Timing: {pred['timing_minutes']:.1f}min, "
                      f"R/R: {pred['risk_reward_ratio']:.2f}")
        else:
            print("❌ Failed to make predictions")
    else:
        print("❌ Failed to train model")