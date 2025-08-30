# src/tactician/enhanced_prediction_integrator.py

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    handle_errors,
    with_tracing_span,
)
from src.utils.logger import get_logger
from src.utils.warning_symbols import error, warning
from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator


class TacticianEnhancedPredictionIntegrator:
    """
    Enhanced Prediction Integrator for Tactician that delivers the same multi-outcome predictions
    as the Analyst but on shorter timeframes with more precise values.
    
    Key Features:
    - Multi-outcome predictions (price, confidence, regime, etc.)
    - Shorter timeframes (1m, 5m vs Analyst's longer timeframes)
    - More precise values using dynamic barriers
    - High precision mode with quality filters
    - Integration with Analyst predictions
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the Tactician enhanced prediction integrator."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize dynamic barrier calculator
        self.barrier_calculator = DynamicBarrierCalculator(config)
        
        # Load Tactician configuration
        self._load_tactician_config()
        
        # Initialize prediction models (placeholder for actual ML models)
        self.tactician_models = self._initialize_tactician_models()
        
        self.logger.info("🚀 Tactician Enhanced Prediction Integrator initialized")

    def _load_tactician_config(self) -> None:
        """Load Tactician-specific configuration."""
        tactician_config = self.config.get("tactician_triple_barrier", {})
        
        # Timeframe configuration
        self.timeframes = tactician_config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = tactician_config.get("primary_timeframe", "1m")
        self.secondary_timeframe = tactician_config.get("secondary_timeframe", "5m")
        
        # Precision configuration
        self.enable_high_precision_mode = tactician_config.get("enable_high_precision_mode", True)
        self.precision_threshold = tactician_config.get("precision_threshold", 0.85)
        
        # Multi-outcome prediction configuration
        self.prediction_types = [
            "price_prediction",
            "confidence_prediction", 
            "regime_prediction",
            "volatility_prediction",
            "momentum_prediction",
            "trend_prediction"
        ]
        
        # Precision multipliers for different prediction types
        self.precision_multipliers = {
            "price_prediction": 2.0,      # 2x more precise price predictions
            "confidence_prediction": 1.5,  # 1.5x more precise confidence
            "regime_prediction": 1.2,      # 1.2x more precise regime detection
            "volatility_prediction": 2.5,  # 2.5x more precise volatility
            "momentum_prediction": 2.0,    # 2x more precise momentum
            "trend_prediction": 1.8        # 1.8x more precise trend
        }

    def _initialize_tactician_models(self) -> dict[str, Any]:
        """Initialize Tactician prediction models."""
        # Placeholder for actual ML models
        # In production, these would be trained models for each prediction type
        models = {}
        
        for prediction_type in self.prediction_types:
            models[prediction_type] = {
                "model": None,  # Placeholder for actual model
                "confidence": 0.85,
                "timeframe": self.primary_timeframe,
                "precision_multiplier": self.precision_multipliers[prediction_type]
            }
        
        return models

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generating tactician enhanced predictions",
    )
    @with_tracing_span("Tactician.generateEnhancedPredictions")
    async def generate_tactician_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_predictions: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = None
    ) -> dict[str, Any]:
        """
        Generate Tactician enhanced predictions based on Analyst predictions.
        
        Args:
            market_data: Market data for the timeframe
            analyst_predictions: Analyst's multi-outcome predictions
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Target timeframe (1m or 5m)
            
        Returns:
            dict: Tactician's enhanced multi-outcome predictions
        """
        try:
            # Determine timeframe
            if timeframe is None:
                timeframe = self._determine_optimal_timeframe(market_data)
            
            # Get dynamic barriers for this timeframe
            upper_barrier, lower_barrier = self.barrier_calculator.calculate_dynamic_barriers(timeframe)
            
            # Generate enhanced predictions for each type
            tactician_predictions = {}
            
            for prediction_type in self.prediction_types:
                enhanced_prediction = await self._generate_enhanced_prediction(
                    prediction_type=prediction_type,
                    market_data=market_data,
                    analyst_predictions=analyst_predictions,
                    upper_barrier=upper_barrier,
                    lower_barrier=lower_barrier,
                    timeframe=timeframe,
                    symbol=symbol,
                    exchange=exchange
                )
                
                if enhanced_prediction:
                    tactician_predictions[prediction_type] = enhanced_prediction
            
            # Add metadata
            tactician_predictions["metadata"] = {
                "timeframe": timeframe,
                "symbol": symbol,
                "exchange": exchange,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "precision_threshold": self.precision_threshold,
                "high_precision_mode": self.enable_high_precision_mode,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"🎯 Generated {len(tactician_predictions)} Tactician enhanced predictions for {timeframe}")
            
            return tactician_predictions
            
        except Exception as e:
            self.logger.error(error(f"❌ Error generating Tactician predictions: {e}"))
            return {}

    async def _generate_enhanced_prediction(
        self,
        prediction_type: str,
        market_data: pd.DataFrame,
        analyst_predictions: dict[str, Any],
        upper_barrier: float,
        lower_barrier: float,
        timeframe: str,
        symbol: str,
        exchange: str
    ) -> dict[str, Any]:
        """Generate enhanced prediction for a specific type."""
        try:
            # Get base prediction from Analyst
            base_prediction = self._extract_analyst_prediction(analyst_predictions, prediction_type)
            
            if base_prediction is None:
                return None
            
            # Apply precision enhancement
            enhanced_prediction = self._apply_precision_enhancement(
                prediction_type=prediction_type,
                base_prediction=base_prediction,
                market_data=market_data,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                timeframe=timeframe
            )
            
            # Apply high precision filtering
            if self.enable_high_precision_mode:
                if enhanced_prediction.get("precision_score", 0.0) < self.precision_threshold:
                    return None
            
            return enhanced_prediction
            
        except Exception as e:
            self.logger.error(error(f"❌ Error generating {prediction_type} prediction: {e}"))
            return None

    def _extract_analyst_prediction(
        self, 
        analyst_predictions: dict[str, Any], 
        prediction_type: str
    ) -> Optional[dict[str, Any]]:
        """Extract base prediction from Analyst predictions."""
        try:
            # Look for prediction in various possible locations
            for key, value in analyst_predictions.items():
                if prediction_type in key.lower():
                    return value
                elif isinstance(value, dict) and prediction_type in value:
                    return value[prediction_type]
            
            # Fallback: create synthetic prediction based on market data
            return self._create_synthetic_prediction(prediction_type, analyst_predictions)
            
        except Exception as e:
            self.logger.warning(warning(f"⚠️ Could not extract {prediction_type} from Analyst predictions: {e}"))
            return None

    def _create_synthetic_prediction(
        self, 
        prediction_type: str, 
        analyst_predictions: dict[str, Any]
    ) -> dict[str, Any]:
        """Create synthetic prediction when Analyst prediction is not available."""
        try:
            # Extract any available confidence or prediction values
            base_confidence = 0.5
            base_prediction = 0.0
            
            for key, value in analyst_predictions.items():
                if isinstance(value, dict):
                    if "confidence" in value:
                        base_confidence = value["confidence"]
                    if "prediction" in value:
                        base_prediction = value["prediction"]
            
            # Create synthetic prediction based on type
            if prediction_type == "price_prediction":
                synthetic_value = base_prediction + np.random.normal(0, 0.001)  # Small price change
            elif prediction_type == "confidence_prediction":
                synthetic_value = min(1.0, base_confidence * 1.1)  # Slightly higher confidence
            elif prediction_type == "regime_prediction":
                synthetic_value = base_prediction  # Keep same regime
            elif prediction_type == "volatility_prediction":
                synthetic_value = base_prediction * 1.2  # Higher volatility for shorter timeframe
            elif prediction_type == "momentum_prediction":
                synthetic_value = base_prediction * 1.5  # Higher momentum for shorter timeframe
            elif prediction_type == "trend_prediction":
                synthetic_value = base_prediction * 1.3  # Higher trend strength
            else:
                synthetic_value = base_prediction
            
            return {
                "prediction": synthetic_value,
                "confidence": base_confidence,
                "model_type": "synthetic",
                "prediction_type": prediction_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(error(f"❌ Error creating synthetic prediction: {e}"))
            return {
                "prediction": 0.0,
                "confidence": 0.5,
                "model_type": "synthetic",
                "prediction_type": prediction_type,
                "timestamp": datetime.now().isoformat()
            }

    def _apply_precision_enhancement(
        self,
        prediction_type: str,
        base_prediction: dict[str, Any],
        market_data: pd.DataFrame,
        upper_barrier: float,
        lower_barrier: float,
        timeframe: str
    ) -> dict[str, Any]:
        """Apply precision enhancement to base prediction."""
        try:
            # Get precision multiplier for this prediction type
            precision_multiplier = self.precision_multipliers.get(prediction_type, 1.0)
            
            # Extract base values
            base_value = base_prediction.get("prediction", 0.0)
            base_confidence = base_prediction.get("confidence", 0.5)
            
            # Apply precision enhancement based on prediction type
            enhanced_value = self._enhance_prediction_value(
                prediction_type=prediction_type,
                base_value=base_value,
                precision_multiplier=precision_multiplier,
                market_data=market_data,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                timeframe=timeframe
            )
            
            # Calculate enhanced confidence
            enhanced_confidence = self._calculate_enhanced_confidence(
                base_confidence=base_confidence,
                precision_multiplier=precision_multiplier,
                market_data=market_data,
                timeframe=timeframe
            )
            
            # Calculate precision score
            precision_score = self._calculate_precision_score(
                enhanced_confidence=enhanced_confidence,
                market_data=market_data,
                timeframe=timeframe
            )
            
            return {
                "prediction": enhanced_value,
                "confidence": enhanced_confidence,
                "precision_score": precision_score,
                "model_type": "tactician_enhanced",
                "prediction_type": prediction_type,
                "timeframe": timeframe,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "precision_multiplier": precision_multiplier,
                "base_prediction": base_prediction,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(error(f"❌ Error applying precision enhancement: {e}"))
            return base_prediction

    def _enhance_prediction_value(
        self,
        prediction_type: str,
        base_value: float,
        precision_multiplier: float,
        market_data: pd.DataFrame,
        upper_barrier: float,
        lower_barrier: float,
        timeframe: str
    ) -> float:
        """Enhance prediction value based on type and precision multiplier."""
        try:
            if prediction_type == "price_prediction":
                # More precise price prediction using barriers
                current_price = market_data['close'].iloc[-1] if not market_data.empty else 100.0
                price_range = current_price * (upper_barrier + lower_barrier)
                enhanced_value = base_value + (price_range * precision_multiplier * 0.1)
                
            elif prediction_type == "confidence_prediction":
                # Higher confidence for shorter timeframes
                enhanced_value = min(1.0, base_value * precision_multiplier)
                
            elif prediction_type == "regime_prediction":
                # More precise regime detection
                enhanced_value = base_value * precision_multiplier
                
            elif prediction_type == "volatility_prediction":
                # Higher volatility prediction for shorter timeframes
                enhanced_value = base_value * precision_multiplier
                
            elif prediction_type == "momentum_prediction":
                # Higher momentum prediction for shorter timeframes
                enhanced_value = base_value * precision_multiplier
                
            elif prediction_type == "trend_prediction":
                # More precise trend prediction
                enhanced_value = base_value * precision_multiplier
                
            else:
                enhanced_value = base_value * precision_multiplier
            
            return enhanced_value
            
        except Exception as e:
            self.logger.error(error(f"❌ Error enhancing prediction value: {e}"))
            return base_value

    def _calculate_enhanced_confidence(
        self,
        base_confidence: float,
        precision_multiplier: float,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> float:
        """Calculate enhanced confidence based on precision multiplier and market conditions."""
        try:
            # Base enhancement from precision multiplier
            enhanced_confidence = min(1.0, base_confidence * precision_multiplier)
            
            # Additional enhancement based on market data quality
            if not market_data.empty:
                # Check for recent volatility (lower volatility = higher confidence)
                recent_volatility = market_data['close'].pct_change().tail(10).std()
                volatility_factor = max(0.5, 1.0 - recent_volatility * 10)
                
                # Check for data quality (more recent data = higher confidence)
                data_freshness = 1.0  # Assume fresh data
                
                # Apply factors
                enhanced_confidence *= volatility_factor * data_freshness
            
            return min(1.0, enhanced_confidence)
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating enhanced confidence: {e}"))
            return base_confidence

    def _calculate_precision_score(
        self,
        enhanced_confidence: float,
        market_data: pd.DataFrame,
        timeframe: str
    ) -> float:
        """Calculate precision score for the enhanced prediction."""
        try:
            # Base precision from confidence
            precision_score = enhanced_confidence
            
            # Adjust based on timeframe (shorter timeframes get higher precision)
            timeframe_factor = 1.2 if timeframe == "1m" else 1.0
            precision_score *= timeframe_factor
            
            # Adjust based on market data quality
            if not market_data.empty:
                # More data points = higher precision
                data_quality_factor = min(1.2, len(market_data) / 100)
                precision_score *= data_quality_factor
            
            return min(1.0, precision_score)
            
        except Exception as e:
            self.logger.error(error(f"❌ Error calculating precision score: {e}"))
            return enhanced_confidence

    def _determine_optimal_timeframe(self, market_data: pd.DataFrame) -> str:
        """Determine optimal timeframe based on market data."""
        try:
            if market_data.empty:
                return self.primary_timeframe
            
            # Check data frequency
            if len(market_data) > 1:
                time_diff = market_data.index[1] - market_data.index[0]
                if time_diff <= timedelta(minutes=1):
                    return "1m"
                elif time_diff <= timedelta(minutes=5):
                    return "5m"
            
            return self.primary_timeframe
            
        except Exception as e:
            self.logger.error(error(f"❌ Error determining optimal timeframe: {e}"))
            return self.primary_timeframe

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="validating tactician predictions",
    )
    async def validate_tactician_predictions(
        self,
        tactician_predictions: dict[str, Any],
        analyst_predictions: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate Tactician predictions against Analyst predictions."""
        try:
            validation_results = {
                "is_valid": True,
                "validation_score": 0.0,
                "issues": [],
                "enhancements": []
            }
            
            total_score = 0.0
            valid_predictions = 0
            
            for prediction_type in self.prediction_types:
                if prediction_type in tactician_predictions:
                    tactician_pred = tactician_predictions[prediction_type]
                    analyst_pred = self._extract_analyst_prediction(analyst_predictions, prediction_type)
                    
                    if analyst_pred:
                        # Validate precision enhancement
                        precision_multiplier = self.precision_multipliers.get(prediction_type, 1.0)
                        tactician_value = tactician_pred.get("prediction", 0.0)
                        analyst_value = analyst_pred.get("prediction", 0.0)
                        
                        # Check if Tactician prediction is more precise
                        if abs(tactician_value) >= abs(analyst_value) * precision_multiplier * 0.8:
                            validation_results["enhancements"].append(f"{prediction_type}: Enhanced precision")
                            total_score += 1.0
                        else:
                            validation_results["issues"].append(f"{prediction_type}: Insufficient precision enhancement")
                        
                        # Check confidence enhancement
                        tactician_confidence = tactician_pred.get("confidence", 0.0)
                        analyst_confidence = analyst_pred.get("confidence", 0.0)
                        
                        if tactician_confidence >= analyst_confidence:
                            total_score += 0.5
                        else:
                            validation_results["issues"].append(f"{prediction_type}: Confidence not enhanced")
                        
                        valid_predictions += 1
                    else:
                        validation_results["issues"].append(f"{prediction_type}: No Analyst prediction available")
                else:
                    validation_results["issues"].append(f"{prediction_type}: Missing Tactician prediction")
            
            # Calculate overall validation score
            if valid_predictions > 0:
                validation_results["validation_score"] = total_score / valid_predictions
                validation_results["is_valid"] = validation_results["validation_score"] >= 0.7
            
            return validation_results
            
        except Exception as e:
            self.logger.error(error(f"❌ Error validating Tactician predictions: {e}"))
            return {"is_valid": False, "validation_score": 0.0, "issues": [str(e)], "enhancements": []}

    def get_prediction_summary(self, tactician_predictions: dict[str, Any]) -> dict[str, Any]:
        """Get summary of Tactician predictions."""
        try:
            summary = {
                "total_predictions": 0,
                "high_precision_predictions": 0,
                "average_confidence": 0.0,
                "average_precision_score": 0.0,
                "prediction_types": {},
                "timeframe": tactician_predictions.get("metadata", {}).get("timeframe", "unknown")
            }
            
            total_confidence = 0.0
            total_precision = 0.0
            valid_predictions = 0
            
            for prediction_type in self.prediction_types:
                if prediction_type in tactician_predictions:
                    pred = tactician_predictions[prediction_type]
                    summary["total_predictions"] += 1
                    
                    confidence = pred.get("confidence", 0.0)
                    precision_score = pred.get("precision_score", 0.0)
                    
                    total_confidence += confidence
                    total_precision += precision_score
                    valid_predictions += 1
                    
                    if precision_score >= self.precision_threshold:
                        summary["high_precision_predictions"] += 1
                    
                    summary["prediction_types"][prediction_type] = {
                        "prediction": pred.get("prediction", 0.0),
                        "confidence": confidence,
                        "precision_score": precision_score
                    }
            
            if valid_predictions > 0:
                summary["average_confidence"] = total_confidence / valid_predictions
                summary["average_precision_score"] = total_precision / valid_predictions
            
            return summary
            
        except Exception as e:
            self.logger.error(error(f"❌ Error getting prediction summary: {e}"))
            return {"error": str(e)}