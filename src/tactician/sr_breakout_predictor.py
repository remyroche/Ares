"""Refactored SR Breakout Predictor using modular components."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger

# Import modular components
from .sr_modules import SRFeatureExtractor, SRLevelDetector, SRMetricsCalculator


class SRBreakoutPredictor:
    """
    Refactored SR Breakout Predictor with modular architecture.
    
    This is a simplified version that delegates functionality to specialized modules:
    - SRLevelDetector: Detects support/resistance levels
    - SRMetricsCalculator: Calculates various metrics
    - SRReportGenerator: Generates analysis reports
    - SRFeatureExtractor: Extracts ML features
    - SRAnalyzer: Performs advanced analysis
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize SR breakout predictor with modular components."""
        self.config = config
        self.logger = system_logger.getChild("SRBreakoutPredictor")
        
        # SR predictor state
        self.is_initialized = False
        self.sr_predictions = {}
        
        # Configuration
        self.sr_config = self.config.get("sr_breakout_predictor", {})
        self.enable_sr_breakout_tactics = self.sr_config.get(
            "enable_sr_breakout_tactics", True
        )
        self.sr_proximity_threshold = self.sr_config.get(
            "sr_proximity_threshold", 0.02
        )
        self.breakout_confidence_threshold = self.sr_config.get(
            "breakout_confidence_threshold", 0.6
        )
        
        # Initialize modular components
        self.level_detector = None
        self.metrics_calculator = None
        self.report_generator = None
        self.feature_extractor = None
        self.analyzer = None
        
        # Performance tracking
        self.performance_metrics = {
            "predictions_made": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "accuracy": 0.0
        }
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="initialize SR breakout predictor"
    )
    async def initialize(self) -> bool:
        """Initialize the SR breakout predictor and its components."""
        try:
            self.logger.info("Initializing SR Breakout Predictor...")
            
            # Initialize components
            self.level_detector = SRLevelDetector(self.config)
            self.metrics_calculator = SRMetricsCalculator(self.config)
            self.feature_extractor = SRFeatureExtractor(self.config)
            
            # Initialize optional components
            try:
                from .sr_modules import SRReportGenerator
                self.report_generator = SRReportGenerator(self.config)
            except ImportError:
                self.logger.warning("Report generator not available")
            
            try:
                from .sr_modules import SRAnalyzer
                self.analyzer = SRAnalyzer(self.config)
            except ImportError:
                self.logger.warning("SR analyzer not available")
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            self.is_initialized = True
            self.logger.info("✅ SR Breakout Predictor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SR breakout predictor: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Check threshold values
            if not 0 < self.sr_proximity_threshold < 1:
                self.logger.error("Invalid sr_proximity_threshold")
                return False
                
            if not 0 < self.breakout_confidence_threshold < 1:
                self.logger.error("Invalid breakout_confidence_threshold")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="get SR context"
    )
    async def get_sr_context(
        self, 
        market_data: pd.DataFrame, 
        current_price: float
    ) -> dict[str, Any]:
        """
        Get comprehensive S/R context for current market state.
        
        Args:
            market_data: Market data DataFrame
            current_price: Current price
            
        Returns:
            Dictionary containing S/R analysis context
        """
        try:
            if not self.is_initialized:
                self.logger.error("SR breakout predictor not initialized")
                return {}
            
            # Detect S/R levels
            sr_levels = self.level_detector.detect_sr_levels(
                market_data, current_price
            )
            
            # Create basic context
            sr_context = {
                "support": sr_levels.get("support", []),
                "resistance": sr_levels.get("resistance", []),
                "current_price": current_price,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add proximity information
            proximity_info = self._calculate_proximity_info(
                current_price, sr_context
            )
            sr_context.update(proximity_info)
            
            # Calculate metrics
            if self.metrics_calculator:
                metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                    market_data, sr_context
                )
                sr_context["metrics"] = metrics
            
            # Perform advanced analysis if available
            if self.analyzer:
                analysis = self.analyzer.analyze_sr_patterns(
                    market_data, sr_context
                )
                sr_context["analysis"] = analysis
            
            return sr_context
            
        except Exception as e:
            self.logger.error(f"Error getting SR context: {e}")
            return {}
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="extract ML features"
    )
    async def extract_ml_features(
        self, 
        market_data: pd.DataFrame, 
        current_price: float
    ) -> dict[str, float]:
        """
        Extract ML features for S/R analysis.
        
        Args:
            market_data: Market data DataFrame
            current_price: Current price
            
        Returns:
            Dictionary of ML features
        """
        try:
            if not self.is_initialized:
                self.logger.error("SR breakout predictor not initialized")
                return {}
            
            # Get SR context first
            sr_context = await self.get_sr_context(market_data, current_price)
            
            # Extract features
            features = self.feature_extractor.extract_ml_features(
                market_data, current_price, sr_context
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return {}
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="predict SR breakout"
    )
    async def predict_sr_breakout(
        self, 
        market_data: pd.DataFrame, 
        current_price: float,
        ml_predictions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Predict S/R breakout likelihood.
        
        Args:
            market_data: Market data DataFrame
            current_price: Current price
            ml_predictions: Optional ML predictions
            
        Returns:
            Dictionary containing breakout predictions
        """
        try:
            if not self.is_initialized:
                self.logger.error("SR breakout predictor not initialized")
                return self._get_default_prediction()
            
            # Get SR context
            sr_context = await self.get_sr_context(market_data, current_price)
            
            # Check proximity to S/R levels
            proximity_details = self.get_sr_proximity_details(
                current_price, sr_context
            )
            
            # Determine breakout type
            breakout_type = "none"
            confidence = 0.0
            target_level = None
            
            if proximity_details["is_near_resistance"]:
                # Potential resistance breakout
                breakout_type = "resistance"
                target_level = proximity_details["nearest_resistance"]
                confidence = self._calculate_breakout_confidence(
                    market_data, sr_context, "resistance"
                )
                
            elif proximity_details["is_near_support"]:
                # Potential support breakdown
                breakout_type = "support"
                target_level = proximity_details["nearest_support"]
                confidence = self._calculate_breakout_confidence(
                    market_data, sr_context, "support"
                )
            
            # Build prediction
            prediction = {
                "breakout_type": breakout_type,
                "confidence": confidence,
                "target_level": target_level,
                "proximity_details": proximity_details,
                "sr_context": sr_context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add ML predictions if available
            if ml_predictions:
                prediction["ml_predictions"] = ml_predictions
            
            # Update performance tracking
            self.performance_metrics["predictions_made"] += 1
            
            # Store prediction for later validation
            self.sr_predictions[datetime.now().isoformat()] = prediction
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting SR breakout: {e}")
            return self._get_default_prediction()
    
    def is_near_sr_level(
        self, 
        current_price: float, 
        sr_levels: list[dict[str, Any]],
        threshold: float | None = None
    ) -> bool:
        """Check if price is near any S/R level."""
        if not sr_levels:
            return False
            
        threshold = threshold or self.sr_proximity_threshold
        
        for level in sr_levels:
            distance = abs(current_price - level["price"]) / current_price
            if distance <= threshold:
                return True
                
        return False
    
    def get_sr_proximity_details(
        self, 
        current_price: float, 
        sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Get detailed proximity information to S/R levels."""
        support_levels = sr_context.get("support", [])
        resistance_levels = sr_context.get("resistance", [])
        
        # Check proximity
        is_near_support = self.is_near_sr_level(
            current_price, support_levels
        )
        is_near_resistance = self.is_near_sr_level(
            current_price, resistance_levels
        )
        
        # Find nearest levels
        nearest_support = self._find_nearest_level(
            current_price, support_levels
        )
        nearest_resistance = self._find_nearest_level(
            current_price, resistance_levels
        )
        
        return {
            "is_near_support": is_near_support,
            "is_near_resistance": is_near_resistance,
            "is_near_any": is_near_support or is_near_resistance,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "support_distance": self._calculate_proximity(
                current_price, nearest_support
            ) if nearest_support else float('inf'),
            "resistance_distance": self._calculate_proximity(
                current_price, nearest_resistance
            ) if nearest_resistance else float('inf')
        }
    
    def _calculate_proximity_info(
        self, 
        current_price: float, 
        sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate proximity information for SR context."""
        proximity_details = self.get_sr_proximity_details(
            current_price, sr_context
        )
        
        return {
            "proximity": proximity_details,
            "is_at_key_level": (
                proximity_details["is_near_support"] or 
                proximity_details["is_near_resistance"]
            )
        }
    
    def _find_nearest_level(
        self, 
        current_price: float, 
        levels: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Find nearest S/R level to current price."""
        if not levels:
            return None
            
        return min(
            levels,
            key=lambda x: abs(x["price"] - current_price)
        )
    
    def _calculate_proximity(
        self, 
        current_price: float, 
        level: dict[str, Any] | None
    ) -> float:
        """Calculate proximity to a level."""
        if not level:
            return float('inf')
            
        return abs(current_price - level["price"]) / current_price
    
    def _calculate_breakout_confidence(
        self, 
        market_data: pd.DataFrame,
        sr_context: dict[str, Any],
        breakout_type: str
    ) -> float:
        """Calculate confidence in breakout prediction."""
        try:
            # Base confidence on multiple factors
            confidence_factors = []
            
            # Momentum factor
            momentum = market_data["close"].pct_change(5).iloc[-1]
            if breakout_type == "resistance" and momentum > 0:
                confidence_factors.append(0.7)
            elif breakout_type == "support" and momentum < 0:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # Volume factor
            volume_ratio = (
                market_data["volume"].iloc[-1] / 
                market_data["volume"].rolling(20).mean().iloc[-1]
            )
            if volume_ratio > 1.5:
                confidence_factors.append(0.8)
            elif volume_ratio > 1.0:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            
            # S/R strength factor
            proximity = sr_context.get("proximity", {})
            if breakout_type == "resistance":
                level = proximity.get("nearest_resistance")
            else:
                level = proximity.get("nearest_support")
                
            if level and level.get("strength", 0) < 0.5:
                confidence_factors.append(0.7)  # Weak level easier to break
            else:
                confidence_factors.append(0.4)
            
            # Average confidence
            confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout confidence: {e}")
            return 0.5
    
    def _get_default_prediction(self) -> dict[str, Any]:
        """Get default prediction when error occurs."""
        return {
            "breakout_type": "none",
            "confidence": 0.0,
            "target_level": None,
            "proximity_details": {
                "is_near_support": False,
                "is_near_resistance": False,
                "is_near_any": False,
                "nearest_support": None,
                "nearest_resistance": None,
                "support_distance": float('inf'),
                "resistance_distance": float('inf')
            },
            "sr_context": {},
            "timestamp": datetime.now().isoformat()
        }
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="predict SR outcome"
    )
    async def predict_sr_outcome(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Predict S/R interaction outcome using optimized parameters.
        
        Args:
            market_data: Market data DataFrame
            current_price: Current price
            sr_context: S/R context
            
        Returns:
            Dictionary containing outcome prediction with probabilities
        """
        try:
            if not self.is_initialized:
                self.logger.error("SR breakout predictor not initialized")
                return self._get_default_outcome()
            
            # Use the optimized probability calculator
            from .sr_modules.sr_probability_calculator import SRProbabilityCalculator
            prob_calculator = SRProbabilityCalculator(self.config)
            
            # Calculate probabilities using optimized parameters
            probabilities = prob_calculator.calculate_probabilities(
                market_data, current_price, sr_context
            )
            
            # Determine most likely outcome
            outcome_map = {
                "breakout": "breakout",
                "rebounce": "rebounce", 
                "consolidation": "consolidation"
            }
            
            predicted_outcome = max(probabilities, key=lambda x: probabilities.get(x, 0))
            confidence = probabilities[predicted_outcome]
            
            # Build outcome prediction
            outcome = {
                "outcome": outcome_map.get(predicted_outcome, "consolidation"),
                "confidence": confidence,
                "probabilities": {
                    "breakout": probabilities["breakout"],
                    "rebounce": probabilities["rebounce"],
                    "consolidation": probabilities["consolidation"]
                },
                "is_near_sr_level": self._check_near_sr_level(current_price, sr_context),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update performance tracking
            self.performance_metrics["predictions_made"] += 1
            
            return outcome
            
        except Exception as e:
            self.logger.error(f"Error predicting SR outcome: {e}")
            return self._get_default_outcome()
    
    def _check_near_sr_level(self, current_price: float, sr_context: dict[str, Any]) -> bool:
        """Check if price is near any S/R level."""
        all_levels = sr_context.get("support", []) + sr_context.get("resistance", [])
        return self.is_near_sr_level(current_price, all_levels)
    
    def _get_default_outcome(self) -> dict[str, Any]:
        """Get default outcome prediction."""
        return {
            "outcome": "consolidation",
            "confidence": 0.33,
            "probabilities": {
                "breakout": 0.33,
                "rebounce": 0.33,
                "consolidation": 0.34
            },
            "is_near_sr_level": False,
            "timestamp": datetime.now().isoformat()
        }
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout stop"
    )
    async def stop(self) -> None:
        """Stop the SR breakout predictor."""
        try:
            self.logger.info("Stopping SR Breakout Predictor...")
            self.is_initialized = False
            self.logger.info("✅ SR Breakout Predictor stopped")
        except Exception as e:
            self.logger.error(f"Error stopping SR breakout predictor: {e}")
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout cleanup"
    )
    async def cleanup(self) -> None:
        """Cleanup SR breakout predictor resources."""
        try:
            self.logger.info("Cleaning up SR Breakout Predictor...")
            await self.stop()
            
            # Clear predictions
            self.sr_predictions.clear()
            
            # Reset performance metrics
            self.performance_metrics = {
                "predictions_made": 0,
                "successful_predictions": 0,
                "failed_predictions": 0,
                "accuracy": 0.0
            }
            
            self.logger.info("✅ SR Breakout Predictor cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up SR breakout predictor: {e}")