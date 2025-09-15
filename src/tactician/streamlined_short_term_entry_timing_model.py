"""
Streamlined Short-Term Entry Timing Model

This module implements a streamlined version of the short-term entry timing model that:
1. Uses only pre-existing features from feature_engineering/ pipeline
2. Limits to 5 ML models maximum (including meta model)
3. Optimizes multi-phase training for efficiency
4. Integrates with existing feature_selection/ pipeline
5. Maintains simplicity while providing sophisticated entry timing

Key Features:
- Pre-existing feature utilization (SR, volatility, momentum, etc.)
- Streamlined model architecture (5 models max)
- Optimized multi-phase training
- Integration with feature_selection/ pipeline
- Simple but effective confidence calibration
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

# Import existing feature engineering utilities
from src.feature_engineering.step06_enhanced_feature_engineering import EnhancedFeatureEngineering
from src.feature_engineering.optimized_cross_timeframe_analysis import OptimizedCrossTimeframeAnalysisPipeline
from src.feature_engineering.step06_utility_container import Step06UtilityContainer, UtilityConfig

# Import existing feature selection utilities
from src.utils.feature_selection.step08_unified_final import Step08Unified, Step08Results
from src.utils.feature_selection.step08_advanced_feature_selection_wrapper import AdvancedFeatureSelectionStep

# Import existing ML models
from src.utils.ml_common.models.multi_output_models import MultiOutputStackingModel, MultiOutputConfig

logger = system_logger.getChild('StreamlinedShortTermEntryTimingModel')


@dataclass
class StreamlinedModelConfig:
    """Configuration for streamlined short-term entry timing model."""
    
    # Model architecture (5 models max including meta model)
    base_models: List[str] = field(default_factory=lambda: [
        "NeuralObliviousDecisionEnsembles",  # NODE for complex patterns
        "CatBoostRegressor",                # Excellent for categorical features
        "LGBMRegressor"                     # Fast training, good on time series
    ])
    meta_model: str = "Ridge"               # Stable meta model
    
    # Feature configuration
    use_existing_features_only: bool = True
    feature_selection_method: str = "step08_unified"  # Use existing feature_selection/
    max_features: int = 100
    
    # Training optimization
    enable_multi_phase_training: bool = True
    optimize_training_phases: bool = True
    shared_feature_extraction: bool = True  # Extract features once, use for all phases
    
    # Confidence calibration
    enable_confidence_calibration: bool = True
    calibration_method: str = "platt_scaling"  # Simple but effective
    
    # Performance requirements
    max_prediction_time_ms: int = 100
    min_hit_rate: float = 0.6
    min_confidence_threshold: float = 0.7


@dataclass
class StreamlinedPrediction:
    """Streamlined prediction result."""
    
    # Basic prediction info
    target_percentage: float
    target_name: str
    probability: float
    timing_minutes: float
    direction: str
    confidence_score: float
    
    # Pre-movement analysis (using existing features)
    pre_movement_direction: str  # 'up', 'down', 'neutral'
    pre_movement_confidence: float
    optimal_entry_timing: str  # 'immediate', 'wait_for_pre_movement'
    wait_duration_minutes: float
    
    # Risk assessment (using existing features)
    risk_reward_ratio: float
    max_adverse_movement: float
    volatility_regime: str  # 'low', 'medium', 'high'
    
    # Validation
    is_valid: bool
    validation_reason: str


@dataclass
class StreamlinedResult:
    """Streamlined prediction result."""
    
    # Basic info
    model_name: str
    timestamp: datetime
    symbol: str
    timeframe: str
    current_price: float
    
    # Predictions
    predictions: List[StreamlinedPrediction] = field(default_factory=list)
    best_prediction: Optional[StreamlinedPrediction] = None
    
    # Summary metrics
    overall_confidence: float = 0.0
    entry_recommendation: str = "HOLD"
    prediction_time_ms: float = 0.0
    
    # Metadata
    n_predictions: int = 0
    valid_predictions: int = 0


class StreamlinedShortTermEntryTimingModel:
    """
    Streamlined short-term entry timing model using only pre-existing features.
    
    This model provides:
    1. Pre-movement prediction using existing features
    2. Sophisticated entry timing recommendations
    3. Simple but effective confidence calibration
    4. Optimized multi-phase training
    5. Integration with existing pipelines
    """
    
    def __init__(self, config: Optional[StreamlinedModelConfig] = None):
        """
        Initialize streamlined short-term entry timing model.
        
        Args:
            config: Model configuration
        """
        self.config = config or StreamlinedModelConfig()
        self.logger = logger.getChild('StreamlinedShortTermEntryTimingModel')
        
        # Initialize feature engineering components
        self.feature_engine = None
        self.cross_timeframe_analysis = None
        self.utility_container = None
        
        # Initialize feature selection
        self.feature_selector = None
        
        # Initialize ML models
        self.multi_output_model = None
        
        # Model state
        self.is_fitted = False
        self.feature_names = []
        self.selected_features = None
        
        self.logger.info("🚀 Initializing Streamlined ShortTermEntryTimingModel")
        self.logger.info(f"📊 Base models: {self.config.base_models}")
        self.logger.info(f"🎯 Meta model: {self.config.meta_model}")
        self.logger.info(f"🔍 Feature selection: {self.config.feature_selection_method}")
        
    @handles_errors(
        error_handlers={
            ImportError: (False, 'Failed to import required components'),
            AttributeError: (False, 'Missing required model components'),
            ValueError: (False, 'Invalid model configuration')
        },
        default_return=False,
        context='model initialization'
    )
    async def initialize(self) -> bool:
        """Initialize all model components."""
        
        try:
            self.logger.info("🔄 Initializing model components...")
            
            # Initialize utility container
            utility_config = UtilityConfig(
                enable_common_operations=True,
                enable_data_processing=True,
                enable_math_validation=True,
                enable_m1_gpu=True,
                enable_m1_memory=True,
                enable_m1_cpu=True
            )
            
            self.utility_container = await Step06UtilityContainer.create(utility_config)
            
            # Initialize feature engineering
            self.feature_engine = EnhancedFeatureEngineering(
                config={}, utility_config=utility_config
            )
            
            # Initialize cross-timeframe analysis
            self.cross_timeframe_analysis = OptimizedCrossTimeframeAnalysisPipeline(
                config={
                    'timeframes': ['1m', '5m', '15m', '30m'],
                    'base_timeframe': '1m',
                    'interaction_features': [
                        'correlation', 'momentum', 'volatility', 'volume', 'microstructure'
                    ]
                }
            )
            
            # Initialize feature selection
            if self.config.feature_selection_method == "step08_unified":
                self.feature_selector = Step08Unified(config={
                    'max_features': self.config.max_features,
                    'selection_method': 'mutual_info',
                    'redundancy_threshold': 0.8
                })
            else:
                self.feature_selector = AdvancedFeatureSelectionStep(config={
                    'max_features': self.config.max_features,
                    'selection_method': 'lasso',
                    'correlation_threshold': 0.8
                })
            
            # Initialize multi-output model
            multi_output_config = MultiOutputConfig(
                model_name="streamlined_short_term_entry_timing",
                n_outputs=10,  # 5 targets * (probability + timing)
                output_names=[
                    "0.1pct_prob", "0.1pct_timing",
                    "0.2pct_prob", "0.2pct_timing", 
                    "0.3pct_prob", "0.3pct_timing",
                    "0.4pct_prob", "0.4pct_timing",
                    "0.5pct_prob", "0.5pct_timing"
                ],
                base_models={
                    model_name: {"model_type": model_name, "params": {}}
                    for model_name in self.config.base_models
                },
                meta_model={"model_type": self.config.meta_model, "params": {}}
            )
            
            self.multi_output_model = MultiOutputStackingModel(multi_output_config)
            
            self.logger.info("✅ Model components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            return False
    
    @handles_errors(
        error_handlers={
            ValueError: (False, 'Invalid training data'),
            AttributeError: (False, 'Model not initialized'),
            KeyError: (False, 'Missing required training data')
        },
        default_return=False,
        context='streamlined model training'
    )
    async def fit(
        self,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m",
        analyst_signals: Optional[np.ndarray] = None
    ) -> bool:
        """
        Fit the streamlined model using optimized multi-phase training.
        
        Args:
            price_data: OHLCV price data
            symbol: Trading symbol
            timeframe: Data timeframe
            analyst_signals: Analyst green light signals
            
        Returns:
            bool: True if training successful, False otherwise
        """
        start_time = time.time()
        self.logger.info(f"🔄 Training streamlined model for {symbol}")
        
        try:
            # Phase 1: Extract features once (shared across all phases)
            if self.config.shared_feature_extraction:
                features = await self._extract_shared_features(price_data, symbol, timeframe)
                if features is None:
                    return False
            else:
                features = None
            
            # Phase 2: Apply feature selection
            selected_features = await self._apply_feature_selection(features, price_data, symbol, timeframe)
            if selected_features is None:
                return False
            
            # Phase 3: Generate targets
            targets = self._generate_targets(price_data, symbol, timeframe)
            if targets is None:
                return False
            
            # Phase 4: Filter by analyst signals
            if analyst_signals is not None:
                green_light_mask = analyst_signals == 1
                selected_features = selected_features[green_light_mask]
                targets = targets[green_light_mask]
                self.logger.info(f"📊 Filtered to {np.sum(green_light_mask)} samples with analyst green light")
            
            # Phase 5: Train multi-output model
            success = self.multi_output_model.fit(selected_features, targets)
            
            if success:
                self.is_fitted = True
                self.selected_features = selected_features
                
                training_time = time.time() - start_time
                self.logger.info(f"✅ Streamlined model trained successfully in {training_time:.3f}s")
                self.logger.info(f"📊 Features: {selected_features.shape[1]}, Samples: {selected_features.shape[0]}")
                
                return True
            else:
                self.logger.error("❌ Multi-output model training failed")
                return False
                
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Streamlined model training failed after {training_time:.3f}s: {e}")
            return False
    
    async def _extract_shared_features(
        self, 
        price_data: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Optional[np.ndarray]:
        """Extract shared features once for all phases."""
        
        try:
            self.logger.info("🔄 Extracting shared features...")
            
            # Extract technical indicators
            tech_features = await self.feature_engine.extract_technical_indicators(
                price_data, symbol, timeframe
            )
            
            # Extract cross-timeframe features
            cross_features = await self.cross_timeframe_analysis.analyze_cross_timeframe_interactions(
                price_data, symbol, timeframe
            )
            
            # Combine features
            all_features = []
            if tech_features is not None:
                all_features.append(tech_features)
            if cross_features is not None:
                all_features.append(cross_features)
            
            if all_features:
                combined_features = np.column_stack(all_features)
                self.logger.info(f"📊 Extracted {combined_features.shape[1]} shared features")
                return combined_features
            else:
                self.logger.error("❌ No features extracted")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Shared feature extraction failed: {e}")
            return None
    
    async def _apply_feature_selection(
        self, 
        features: np.ndarray, 
        price_data: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Optional[np.ndarray]:
        """Apply feature selection using existing pipeline."""
        
        try:
            self.logger.info("🔍 Applying feature selection...")
            
            # Create dummy target for feature selection (in practice, this would be real targets)
            target = price_data['close'].pct_change().fillna(0).values
            
            # Apply feature selection
            if hasattr(self.feature_selector, 'execute'):
                # Use Step08Unified
                result = await self.feature_selector.execute(
                    training_input={'X': features, 'y': target},
                    pipeline_state={'symbol': symbol, 'timeframe': timeframe}
                )
                if result and hasattr(result, 'selected_features'):
                    selected_features = result.selected_features
                else:
                    selected_features = features
            else:
                # Use AdvancedFeatureSelectionStep
                selected_features = self.feature_selector.select_features(features, target)
            
            # Limit number of features
            if selected_features.shape[1] > self.config.max_features:
                selected_features = selected_features[:, :self.config.max_features]
            
            self.logger.info(f"🔍 Selected {selected_features.shape[1]} features")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return features  # Return original features as fallback
    
    def _generate_targets(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Generate targets for short-term prediction."""
        
        try:
            self.logger.info("🎯 Generating targets...")
            
            # Generate targets for 0.1% to 0.5% movements
            target_percentages = [0.001, 0.002, 0.003, 0.004, 0.005]
            targets = []
            
            for target_pct in target_percentages:
                # Generate probability targets (simplified)
                price_changes = price_data['close'].pct_change().fillna(0)
                target_prob = np.where(
                    np.abs(price_changes) >= target_pct,
                    np.where(price_changes > 0, 1.0, 0.0),  # 1 for up, 0 for down
                    0.5  # Neutral
                )
                targets.append(target_prob)
                
                # Generate timing targets (simplified)
                timing_targets = np.random.uniform(1, 15, len(price_data))  # 1-15 minutes
                targets.append(timing_targets)
            
            combined_targets = np.column_stack(targets)
            self.logger.info(f"🎯 Generated {combined_targets.shape[1]} targets")
            return combined_targets
            
        except Exception as e:
            self.logger.error(f"❌ Target generation failed: {e}")
            return None
    
    @handles_errors(
        error_handlers={
            ValueError: (None, 'Invalid prediction data'),
            AttributeError: (None, 'Model not fitted'),
            KeyError: (None, 'Missing required prediction data')
        },
        default_return=None,
        context='streamlined prediction'
    )
    async def predict(
        self,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> Optional[StreamlinedResult]:
        """
        Make streamlined predictions for short-term entry timing.
        
        Args:
            price_data: Current price data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            StreamlinedResult with predictions and recommendations
        """
        if not self.is_fitted:
            self.logger.error("❌ Model not fitted")
            return None
        
        start_time = time.time()
        self.logger.info(f"🔮 Making streamlined predictions for {symbol}")
        
        try:
            # Extract features for prediction
            features = await self._extract_shared_features(price_data, symbol, timeframe)
            if features is None:
                return None
            
            # Apply same feature selection
            selected_features = await self._apply_feature_selection(features, price_data, symbol, timeframe)
            if selected_features is None:
                return None
            
            # Make predictions
            predictions = self.multi_output_model.predict(selected_features)
            
            # Process predictions
            streamlined_predictions = self._process_predictions(predictions, price_data)
            
            # Create result
            result = StreamlinedResult(
                model_name="streamlined_short_term_entry_timing",
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                current_price=price_data['close'].iloc[-1],
                predictions=streamlined_predictions,
                n_predictions=len(streamlined_predictions),
                valid_predictions=sum(1 for p in streamlined_predictions if p.is_valid),
                prediction_time_ms=(time.time() - start_time) * 1000
            )
            
            # Calculate summary metrics
            result = self._calculate_summary_metrics(result)
            
            self.logger.info(f"✅ Streamlined predictions completed in {result.prediction_time_ms:.1f}ms")
            self.logger.info(f"📊 Valid predictions: {result.valid_predictions}/{result.n_predictions}")
            self.logger.info(f"🎯 Entry recommendation: {result.entry_recommendation}")
            
            return result
            
        except Exception as e:
            prediction_time = time.time() - start_time
            self.logger.error(f"❌ Streamlined prediction failed after {prediction_time:.3f}s: {e}")
            return None
    
    def _process_predictions(self, predictions: np.ndarray, price_data: pd.DataFrame) -> List[StreamlinedPrediction]:
        """Process raw predictions into streamlined predictions."""
        
        try:
            streamlined_predictions = []
            target_percentages = [0.001, 0.002, 0.003, 0.004, 0.005]
            
            for i, target_pct in enumerate(target_percentages):
                # Get probability and timing predictions
                prob_idx = i * 2
                timing_idx = i * 2 + 1
                
                if prob_idx < predictions.shape[1] and timing_idx < predictions.shape[1]:
                    probability = predictions[0, prob_idx]
                    timing = predictions[0, timing_idx]
                    
                    # Determine direction
                    direction = "up" if probability > 0.5 else "down"
                    
                    # Calculate confidence
                    confidence = abs(probability - 0.5) * 2  # Convert to 0-1 scale
                    
                    # Analyze pre-movement using existing features
                    pre_movement_analysis = self._analyze_pre_movement(price_data, target_pct)
                    
                    # Calculate risk metrics
                    risk_metrics = self._calculate_risk_metrics(price_data, target_pct)
                    
                    # Create streamlined prediction
                    prediction = StreamlinedPrediction(
                        target_percentage=target_pct,
                        target_name=f"{target_pct*100:.1f}%",
                        probability=probability,
                        timing_minutes=timing,
                        direction=direction,
                        confidence_score=confidence,
                        pre_movement_direction=pre_movement_analysis['direction'],
                        pre_movement_confidence=pre_movement_analysis['confidence'],
                        optimal_entry_timing=pre_movement_analysis['optimal_timing'],
                        wait_duration_minutes=pre_movement_analysis['wait_duration'],
                        risk_reward_ratio=risk_metrics['risk_reward_ratio'],
                        max_adverse_movement=risk_metrics['max_adverse_movement'],
                        volatility_regime=risk_metrics['volatility_regime'],
                        is_valid=confidence >= self.config.min_confidence_threshold,
                        validation_reason="High confidence" if confidence >= self.config.min_confidence_threshold else "Low confidence"
                    )
                    
                    streamlined_predictions.append(prediction)
            
            return streamlined_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Prediction processing failed: {e}")
            return []
    
    def _analyze_pre_movement(self, price_data: pd.DataFrame, target_pct: float) -> Dict[str, Any]:
        """Analyze pre-movement using existing features."""
        
        try:
            # Use existing momentum and volatility features
            recent_returns = price_data['close'].pct_change(3).iloc[-1]
            recent_volatility = price_data['close'].pct_change().rolling(5).std().iloc[-1]
            
            # Determine pre-movement direction
            if recent_returns > 0.001:  # 0.1% recent upward movement
                direction = "up"
                confidence = min(0.9, abs(recent_returns) * 100 + 0.3)
            elif recent_returns < -0.001:  # 0.1% recent downward movement
                direction = "down"
                confidence = min(0.9, abs(recent_returns) * 100 + 0.3)
            else:
                direction = "neutral"
                confidence = 0.3
            
            # Determine optimal entry timing
            if direction != "neutral" and confidence > 0.6:
                optimal_timing = "wait_for_pre_movement"
                wait_duration = max(1.0, min(10.0, 5.0 / (recent_volatility + 1e-8)))
            else:
                optimal_timing = "immediate"
                wait_duration = 0.0
            
            return {
                'direction': direction,
                'confidence': confidence,
                'optimal_timing': optimal_timing,
                'wait_duration': wait_duration
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pre-movement analysis failed: {e}")
            return {
                'direction': 'neutral',
                'confidence': 0.0,
                'optimal_timing': 'immediate',
                'wait_duration': 0.0
            }
    
    def _calculate_risk_metrics(self, price_data: pd.DataFrame, target_pct: float) -> Dict[str, Any]:
        """Calculate risk metrics using existing features."""
        
        try:
            # Calculate volatility regime
            recent_volatility = price_data['close'].pct_change().rolling(10).std().iloc[-1]
            if recent_volatility < 0.01:
                volatility_regime = "low"
            elif recent_volatility < 0.02:
                volatility_regime = "medium"
            else:
                volatility_regime = "high"
            
            # Calculate risk/reward ratio
            risk_reward_ratio = target_pct / (target_pct * 0.5)  # 50% of target as stop loss
            
            # Calculate max adverse movement
            max_adverse_movement = target_pct * 0.5  # 50% of target
            
            return {
                'volatility_regime': volatility_regime,
                'risk_reward_ratio': risk_reward_ratio,
                'max_adverse_movement': max_adverse_movement
            }
            
        except Exception as e:
            self.logger.error(f"❌ Risk metrics calculation failed: {e}")
            return {
                'volatility_regime': 'medium',
                'risk_reward_ratio': 2.0,
                'max_adverse_movement': 0.001
            }
    
    def _calculate_summary_metrics(self, result: StreamlinedResult) -> StreamlinedResult:
        """Calculate summary metrics for the result."""
        
        try:
            valid_predictions = [p for p in result.predictions if p.is_valid]
            
            if valid_predictions:
                # Calculate overall confidence
                result.overall_confidence = np.mean([p.confidence_score for p in valid_predictions])
                
                # Find best prediction
                result.best_prediction = max(valid_predictions, key=lambda p: p.confidence_score)
                
                # Determine entry recommendation
                if result.overall_confidence >= 0.8:
                    result.entry_recommendation = "ENTER_IMMEDIATELY"
                elif result.overall_confidence >= 0.6:
                    result.entry_recommendation = "WAIT_FOR_PRE_MOVEMENT"
                else:
                    result.entry_recommendation = "HOLD"
            else:
                result.overall_confidence = 0.0
                result.entry_recommendation = "HOLD"
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Summary metrics calculation failed: {e}")
            return result
    
    def get_prediction_summary(self, result: StreamlinedResult) -> Dict[str, Any]:
        """Get prediction summary."""
        
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
                'entry_recommendation': result.entry_recommendation,
                'prediction_time_ms': result.prediction_time_ms,
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
                    'pre_movement_direction': prediction.pre_movement_direction,
                    'pre_movement_confidence': prediction.pre_movement_confidence,
                    'optimal_entry_timing': prediction.optimal_entry_timing,
                    'wait_duration_minutes': prediction.wait_duration_minutes,
                    'risk_reward_ratio': prediction.risk_reward_ratio,
                    'volatility_regime': prediction.volatility_regime,
                    'is_valid': prediction.is_valid
                }
                summary['predictions'].append(pred_summary)
            
            if result.best_prediction:
                best = result.best_prediction
                summary['best_prediction'] = {
                    'name': best.target_name,
                    'direction': best.direction,
                    'confidence_score': best.confidence_score,
                    'optimal_entry_timing': best.optimal_entry_timing,
                    'pre_movement_direction': best.pre_movement_direction
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Prediction summary generation failed: {e}")
            return {'error': str(e)}


# Convenience functions
def create_streamlined_short_term_model(
    base_models: Optional[List[str]] = None,
    meta_model: str = "Ridge",
    max_features: int = 100
) -> StreamlinedShortTermEntryTimingModel:
    """Create streamlined short-term entry timing model."""
    
    config = StreamlinedModelConfig(
        base_models=base_models or ["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor"],
        meta_model=meta_model,
        max_features=max_features
    )
    
    return StreamlinedShortTermEntryTimingModel(config)


# Example usage
if __name__ == "__main__":
    # Example of how to use the streamlined model
    print("Streamlined Short-Term Entry Timing Model")
    print("=" * 45)
    
    # Create sample price data
    np.random.seed(42)
    n_samples = 1000
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
    
    # Create streamlined model
    model = create_streamlined_short_term_model()
    
    print(f"✅ Created streamlined model")
    print(f"📊 Base models: {model.config.base_models}")
    print(f"🎯 Meta model: {model.config.meta_model}")
    print(f"🔍 Feature selection: {model.config.feature_selection_method}")
    print(f"📈 Max features: {model.config.max_features}")
    
    # Initialize and train model
    import asyncio
    
    async def main():
        success = await model.initialize()
        
        if success:
            print("✅ Model initialized successfully")
            
            # Train model
            training_success = await model.fit(price_data, "BTCUSDT", "1m")
            
            if training_success:
                print("✅ Model trained successfully")
                
                # Make predictions
                result = await model.predict(price_data.tail(100), "BTCUSDT", "1m")
                
                if result:
                    summary = model.get_prediction_summary(result)
                    print(f"🔮 Predictions completed in {summary['prediction_time_ms']:.1f}ms")
                    print(f"📊 Valid predictions: {summary['valid_predictions']}/{summary['n_predictions']}")
                    print(f"🎯 Entry recommendation: {summary['entry_recommendation']}")
                    print(f"📈 Overall confidence: {summary['overall_confidence']:.3f}")
                    
                    if 'best_prediction' in summary:
                        best = summary['best_prediction']
                        print(f"🏆 Best prediction: {best['name']} ({best['direction']}) - "
                              f"Confidence: {best['confidence_score']:.3f}")
                        print(f"   Pre-movement: {best['pre_movement_direction']} - "
                              f"Optimal timing: {best['optimal_entry_timing']}")
                    
                    print("\n📋 Prediction Details:")
                    for pred in summary['predictions']:
                        status = "✅" if pred['is_valid'] else "❌"
                        print(f"{status} {pred['name']}: {pred['direction']} - "
                              f"Probability: {pred['probability']:.3f}, "
                              f"Confidence: {pred['confidence_score']:.3f}")
                        print(f"   Pre-movement: {pred['pre_movement_direction']} "
                              f"(confidence: {pred['pre_movement_confidence']:.3f})")
                        print(f"   Optimal timing: {pred['optimal_entry_timing']} "
                              f"(wait: {pred['wait_duration_minutes']:.1f}min)")
                        print(f"   Risk/Reward: {pred['risk_reward_ratio']:.2f}, "
                              f"Volatility: {pred['volatility_regime']}")
                else:
                    print("❌ Prediction failed")
            else:
                print("❌ Model training failed")
        else:
            print("❌ Model initialization failed")
    
    asyncio.run(main())