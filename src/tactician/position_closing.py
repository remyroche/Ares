# src/tactician/position_closing.py

"""
Position Closing Module for Tactician.
Handles position closure based on ML model predictions and step17-optimized parameters.
"""

from datetime import datetime
from typing import Any, Dict, Optional, List

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
)

class PositionCloser:
    """
    Position Closer that handles position closure based on ML model predictions
    and step17-optimized parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Position Closer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PositionCloser")

        # Configuration from step17 optimization results
        self.position_config = config.get("position_closing", {})
        
        # Load step17 optimized parameters
        step17_config = config.get("step17_optimization", {})
        tpsl_optimization = step17_config.get("tpsl", {})
        
        # Load optimized position closing parameters (step17-optimized)
        self.atr_multiplier = tpsl_optimization.get("atr_multiplier", 2.0)
        self.confidence_threshold = tpsl_optimization.get("confidence_threshold", 0.7)
        self.min_hold_time = tpsl_optimization.get("min_hold_time", 300)  # 5 minutes
        
        # Load additional optimized parameters (step17-optimized)
        self.stop_loss_multiplier = tpsl_optimization.get("stop_loss_multiplier", 1.5)
        self.take_profit_multiplier = tpsl_optimization.get("take_profit_multiplier", 2.0)
        self.trailing_stop_enabled = tpsl_optimization.get("trailing_stop_enabled", True)
        self.trailing_stop_distance = tpsl_optimization.get("trailing_stop_distance", 0.02)
        self.max_hold_time = tpsl_optimization.get("max_hold_time", 3600)  # 1 hour

        # Load step17 barrier confidence threshold for exit strategy (step17-optimized)
        step12_config = config.get("step12_confidence_optimization", {})
        position_opening_config = step12_config.get("position_opening", {})
        self.barrier_confidence_threshold = position_opening_config.get("min_barrier_confidence", 0.72)

        # ML Model Integration
        self.ml_models = {}
        self.ml_config = config.get("ml_models", {})
        
        # Step17-optimized ML model parameters
        ml_optimization = step17_config.get("ml_models", {})
        self.barrier_confidence_model_weight = ml_optimization.get("barrier_confidence_model_weight", 0.8)
        self.confidence_factor_model_weight = ml_optimization.get("confidence_factor_model_weight", 0.2)
        self.ml_confidence_threshold = ml_optimization.get("ml_confidence_threshold", 0.6)

        # State tracking
        self.closed_positions = []
        self.position_history = []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closer initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the position closer.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Position Closer...")

            # Initialize ML models for barrier confidence assessment
            await self._initialize_ml_models()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid position closer configuration"))
                return False

            self.logger.info("✅ Position Closer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Position Closer initialization failed: {e}"))
            return False

    async def _initialize_ml_models(self) -> None:
        """
        Initialize ML models for barrier confidence assessment.
        These models should be trained in step9 and optimized in step17.
        """
        try:
            # Load barrier confidence prediction model (step17-optimized)
            barrier_model_path = self.ml_config.get("barrier_confidence_model_path")
            if barrier_model_path:
                try:
                    import joblib
                    self.ml_models["barrier_confidence"] = joblib.load(barrier_model_path)
                    self.logger.info(f"✅ Loaded barrier confidence ML model: {barrier_model_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load barrier confidence model: {e}")

            # Load confidence factor prediction model (step17-optimized)
            confidence_factor_model_path = self.ml_config.get("confidence_factor_model_path")
            if confidence_factor_model_path:
                try:
                    import joblib
                    self.ml_models["confidence_factors"] = joblib.load(confidence_factor_model_path)
                    self.logger.info(f"✅ Loaded confidence factors ML model: {confidence_factor_model_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load confidence factors model: {e}")

            # Load price direction prediction model (step17-optimized)
            price_direction_model_path = self.ml_config.get("price_direction_model_path")
            if price_direction_model_path:
                try:
                    import joblib
                    self.ml_models["price_direction"] = joblib.load(price_direction_model_path)
                    self.logger.info(f"✅ Loaded price direction ML model: {price_direction_model_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load price direction model: {e}")

            self.logger.info(f"✅ Initialized {len(self.ml_models)} ML models for position closing")

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to initialize ML models: {e}"))

    def _validate_configuration(self) -> bool:
        """
        Validate position closer configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.atr_multiplier <= 0:
                self.logger.error(invalid("ATR multiplier must be positive"))
                return False

            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid("Confidence threshold must be between 0 and 1"))
                return False

            if self.min_hold_time < 0:
                self.logger.error(invalid("Minimum hold time must be non-negative"))
                return False

            if not 0 <= self.barrier_confidence_threshold <= 1:
                self.logger.error(invalid("Barrier confidence threshold must be between 0 and 1"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.
        
        Args:
            step17_results: Step17 optimization results
        """
        try:
            if "tpsl" in step17_results:
                tpsl_optimization = step17_results["tpsl"]
                
                # Update position closing parameters (step17-optimized)
                self.atr_multiplier = tpsl_optimization.get("atr_multiplier", self.atr_multiplier)
                self.confidence_threshold = tpsl_optimization.get("confidence_threshold", self.confidence_threshold)
                self.min_hold_time = tpsl_optimization.get("min_hold_time", self.min_hold_time)
                
                # Update additional parameters (step17-optimized)
                self.stop_loss_multiplier = tpsl_optimization.get("stop_loss_multiplier", self.stop_loss_multiplier)
                self.take_profit_multiplier = tpsl_optimization.get("take_profit_multiplier", self.take_profit_multiplier)
                self.trailing_stop_enabled = tpsl_optimization.get("trailing_stop_enabled", self.trailing_stop_enabled)
                self.trailing_stop_distance = tpsl_optimization.get("trailing_stop_distance", self.trailing_stop_distance)
                self.max_hold_time = tpsl_optimization.get("max_hold_time", self.max_hold_time)

            if "ml_models" in step17_results:
                ml_optimization = step17_results["ml_models"]
                
                # Update ML model parameters (step17-optimized)
                self.barrier_confidence_model_weight = ml_optimization.get("barrier_confidence_model_weight", self.barrier_confidence_model_weight)
                self.confidence_factor_model_weight = ml_optimization.get("confidence_factor_model_weight", self.confidence_factor_model_weight)
                self.ml_confidence_threshold = ml_optimization.get("ml_confidence_threshold", self.ml_confidence_threshold)

            if "position_opening" in step17_results:
                position_opening = step17_results["position_opening"]
                self.barrier_confidence_threshold = position_opening.get("min_barrier_confidence", self.barrier_confidence_threshold)
                
                self.logger.info("✅ Position closer configuration refreshed from step17 results")
                
        except Exception as e:
            self.logger.error(f"Error refreshing step17 configuration: {e}")

    def _get_ml_barrier_predictions(self, market_data: Dict[str, Any], position_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Get ML model predictions for barrier confidence assessment.
        
        Args:
            market_data: Current market data
            position_data: Position information
            
        Returns:
            Dict containing ML model predictions
        """
        try:
            predictions = {}
            
            # Prepare features for ML models
            features = self._prepare_ml_features(market_data, position_data)
            
            # Get barrier confidence prediction from ML model
            if "barrier_confidence" in self.ml_models:
                try:
                    barrier_model = self.ml_models["barrier_confidence"]
                    barrier_confidence = barrier_model.predict_proba([features])[0]
                    predictions["barrier_confidence"] = barrier_confidence[1]  # Probability of high confidence
                except Exception as e:
                    self.logger.warning(f"⚠️ Barrier confidence ML model prediction failed: {e}")
                    predictions["barrier_confidence"] = 0.5  # Fallback
            
            # Get confidence factors prediction from ML model
            if "confidence_factors" in self.ml_models:
                try:
                    confidence_model = self.ml_models["confidence_factors"]
                    confidence_factors = confidence_model.predict([features])[0]
                    predictions["price_direction_confidence"] = confidence_factors[0]
                    predictions["price_target_confidence"] = confidence_factors[1]
                except Exception as e:
                    self.logger.warning(f"⚠️ Confidence factors ML model prediction failed: {e}")
                    predictions["price_direction_confidence"] = 1.0  # Fallback
                    predictions["price_target_confidence"] = 1.0  # Fallback
            
            # Get price direction prediction from ML model
            if "price_direction" in self.ml_models:
                try:
                    direction_model = self.ml_models["price_direction"]
                    direction_proba = direction_model.predict_proba([features])[0]
                    predictions["price_direction_probability"] = direction_proba[1]  # Probability of upward movement
                except Exception as e:
                    self.logger.warning(f"⚠️ Price direction ML model prediction failed: {e}")
                    predictions["price_direction_probability"] = 0.5  # Fallback
            
            return predictions
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error getting ML barrier predictions: {e}"))
            return {
                "barrier_confidence": 0.5,
                "price_direction_confidence": 1.0,
                "price_target_confidence": 1.0,
                "price_direction_probability": 0.5
            }

    def _prepare_ml_features(self, market_data: Dict[str, Any], position_data: Dict[str, Any]) -> List[float]:
        """
        Prepare features for ML model prediction.
        
        Args:
            market_data: Current market data
            position_data: Position information
            
        Returns:
            List of feature values for ML model
        """
        try:
            features = []
            
            # Market features
            features.extend([
                market_data.get("current_price", 0),
                market_data.get("volume", 0),
                market_data.get("atr", 0),
                market_data.get("rsi", 50),
                market_data.get("momentum", 0),
                market_data.get("volatility", 0),
            ])
            
            # Position features
            features.extend([
                position_data.get("entry_price", 0),
                position_data.get("quantity", 0),
                position_data.get("unrealized_pnl", 0),
                1.0 if position_data.get("side", "").upper() == "LONG" else 0.0,
            ])
            
            # Time features
            entry_time = position_data.get("entry_time")
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                position_age = (datetime.now() - entry_time).total_seconds()
                features.append(position_age)
            else:
                features.append(0)
            
            # Ensure we have enough features (pad with zeros if needed)
            while len(features) < 20:  # Minimum feature count for ML models
                features.append(0.0)
            
            return features[:20]  # Limit to 20 features
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error preparing ML features: {e}"))
            return [0.0] * 20  # Return default features

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closure evaluation"
    )
    async def should_close_position(
        self,
        position_data: Dict[str, Any],
        model_confidence: float,
        atr_value: float,
        current_price: float,
        barrier_confidence: Optional[float] = None
    ) -> bool:
        """
        Determine if a position should be closed based on ML model predictions and step17-optimized parameters.

        Args:
            position_data: Position information
            model_confidence: Model confidence score (0-1)
            atr_value: Average True Range value
            current_price: Current market price
            barrier_confidence: ML-predicted confidence for meeting the two barriers (optional)

        Returns:
            bool: True if position should be closed
        """
        try:
            # Check ML-predicted barrier confidence threshold (NEW EXIT STRATEGY)
            if barrier_confidence is not None and barrier_confidence < self.barrier_confidence_threshold:
                self.logger.info(f"🚨 ML EXIT STRATEGY: Closing position due to low ML barrier confidence: {barrier_confidence:.3f} < {self.barrier_confidence_threshold}")
                return True

            # Check ML confidence threshold (step17-optimized)
            if model_confidence < self.ml_confidence_threshold:
                self.logger.info(f"Closing position due to low ML confidence: {model_confidence:.3f} < {self.ml_confidence_threshold}")
                return True

            # Check ATR-based exit (step17-optimized)
            if self._should_close_by_atr(position_data, atr_value, current_price):
                self.logger.info("Closing position due to ATR-based exit rule")
                return True

            # Check minimum hold time (step17-optimized)
            if self._should_close_by_time(position_data):
                self.logger.info("Closing position due to minimum hold time")
                return True

            return False

        except Exception as e:
            self.logger.error(failed(f"❌ Position closure evaluation failed: {e}"))
            return False

    def assess_barrier_confidence(
        self,
        tactician_predictions: Dict[str, Any],
        current_price: float,
        position_data: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Assess confidence for meeting the two barriers using ML models.
        
        This method uses ML models trained in step9 and optimized in step17
        to predict barrier confidence instead of using hardcoded formulas.
        
        Args:
            tactician_predictions: Tactician's predictions including barrier probabilities
            current_price: Current market price
            position_data: Position information including entry price and side
            market_data: Additional market data for ML model features
            
        Returns:
            float: ML-predicted confidence for meeting the two barriers (0-1)
        """
        try:
            # Prepare market data if not provided
            if market_data is None:
                market_data = {
                    "current_price": current_price,
                    "volume": 0,
                    "atr": 0,
                    "rsi": 50,
                    "momentum": 0,
                    "volatility": 0,
                }
            
            # Get ML model predictions (step17-optimized)
            ml_predictions = self._get_ml_barrier_predictions(market_data, position_data)
            
            # Use ML model predictions as primary source
            barrier_confidence = ml_predictions.get("barrier_confidence", 0.5)
            price_direction_confidence = ml_predictions.get("price_direction_confidence", 1.0)
            price_target_confidence = ml_predictions.get("price_target_confidence", 1.0)
            
            # Apply step17-optimized weights
            combined_confidence = (
                barrier_confidence * self.barrier_confidence_model_weight +
                (price_direction_confidence * price_target_confidence) * self.confidence_factor_model_weight
            )
            
            # Ensure confidence is within valid range
            combined_confidence = max(0.0, min(1.0, combined_confidence))
            
            # Log ML model predictions
            self.logger.info(f"🎯 ML Barrier Confidence Assessment:")
            self.logger.info(f"   Position: {position_data.get('side', 'UNKNOWN')} @ {position_data.get('entry_price', 0):.4f}, Current: {current_price:.4f}")
            self.logger.info(f"   ML Barrier Confidence: {barrier_confidence:.3f}")
            self.logger.info(f"   ML Price Direction Confidence: {price_direction_confidence:.3f}")
            self.logger.info(f"   ML Price Target Confidence: {price_target_confidence:.3f}")
            self.logger.info(f"   Combined ML Confidence: {combined_confidence:.3f}")
            self.logger.info(f"   Step17 Threshold: {self.barrier_confidence_threshold:.3f}")
            self.logger.info(f"   ML Model Weight: {self.barrier_confidence_model_weight:.3f}")
            
            return combined_confidence
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error assessing barrier confidence with ML models: {e}"))
            return 0.0

    def _should_close_by_atr(
        self,
        position_data: Dict[str, Any],
        atr_value: float,
        current_price: float
    ) -> bool:
        """
        Check if position should be closed based on ATR.

        Args:
            position_data: Position information
            atr_value: ATR value
            current_price: Current market price

        Returns:
            bool: True if should close by ATR
        """
        try:
            entry_price = position_data.get("entry_price", 0)
            if entry_price <= 0:
                return False

            # Calculate ATR-based exit levels
            atr_exit_distance = atr_value * self.atr_multiplier

            # For long positions
            if position_data.get("side", "").upper() == "LONG":
                stop_loss = entry_price - atr_exit_distance
                return current_price <= stop_loss

            # For short positions
            elif position_data.get("side", "").upper() == "SHORT":
                stop_loss = entry_price + atr_exit_distance
                return current_price >= stop_loss

            return False

        except Exception as e:
            self.logger.error(failed(f"❌ ATR-based closure check failed: {e}"))
            return False

    def _should_close_by_time(self, position_data: Dict[str, Any]) -> bool:
        """
        Check if position should be closed based on minimum hold time.

        Args:
            position_data: Position information

        Returns:
            bool: True if should close by time
        """
        try:
            entry_time = position_data.get("entry_time")
            if not entry_time:
                return False

            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

            hold_time = (datetime.now() - entry_time).total_seconds()
            return hold_time >= self.min_hold_time

        except Exception as e:
            self.logger.error(failed(f"❌ Time-based closure check failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position closure execution"
    )
    async def close_position(
        self,
        position_data: Dict[str, Any],
        close_reason: str
    ) -> Optional[Dict[str, Any]]:
        """
        Execute position closure.

        Args:
            position_data: Position information
            close_reason: Reason for closure

        Returns:
            Dict: Closure result or None if failed
        """
        try:
            self.logger.info(f"Closing position: {close_reason}")

            # Record closure
            closure_record = {
                "position_id": position_data.get("position_id"),
                "symbol": position_data.get("symbol"),
                "side": position_data.get("side"),
                "entry_price": position_data.get("entry_price"),
                "exit_price": position_data.get("current_price"),
                "quantity": position_data.get("quantity"),
                "close_reason": close_reason,
                "close_time": datetime.now().isoformat(),
                "pnl": self._calculate_pnl(position_data)
            }

            self.closed_positions.append(closure_record)
            self.position_history.append(closure_record)

            self.logger.info(f"✅ Position closed successfully: {closure_record['pnl']:.4f} PnL")
            return closure_record

        except Exception as e:
            self.logger.error(failed(f"❌ Position closure failed: {e}"))
            return None

    def _calculate_pnl(self, position_data: Dict[str, Any]) -> float:
        """
        Calculate position PnL.

        Args:
            position_data: Position information

        Returns:
            float: Calculated PnL
        """
        try:
            entry_price = position_data.get("entry_price", 0)
            current_price = position_data.get("current_price", 0)
            quantity = position_data.get("quantity", 0)
            side = position_data.get("side", "").upper()

            if entry_price <= 0 or current_price <= 0 or quantity <= 0:
                return 0.0

            if side == "LONG":
                return (current_price - entry_price) * quantity
            elif side == "SHORT":
                return (entry_price - current_price) * quantity
            else:
                return 0.0

        except Exception as e:
            self.logger.error(failed(f"❌ PnL calculation failed: {e}"))
            return 0.0

    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """
        Get list of closed positions.

        Returns:
            List[Dict[str, Any]]: Closed positions
        """
        return self.closed_positions.copy()

    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get complete position history.

        Returns:
            List[Dict[str, Any]]: Position history
        """
        return self.position_history.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for closed positions.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            if not self.closed_positions:
                return {
                    "total_positions": 0,
                    "winning_positions": 0,
                    "losing_positions": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "average_pnl": 0.0
                }

            total_positions = len(self.closed_positions)
            winning_positions = len([p for p in self.closed_positions if p.get("pnl", 0) > 0])
            losing_positions = len([p for p in self.closed_positions if p.get("pnl", 0) < 0])
            total_pnl = sum(p.get("pnl", 0) for p in self.closed_positions)

            return {
                "total_positions": total_positions,
                "winning_positions": winning_positions,
                "losing_positions": losing_positions,
                "win_rate": winning_positions / total_positions if total_positions > 0 else 0.0,
                "total_pnl": total_pnl,
                "average_pnl": total_pnl / total_positions if total_positions > 0 else 0.0
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Performance metrics calculation failed: {e}"))
            return {}

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info("Cleaning up Position Closer...")

            # Save position history if needed
            if self.position_history:
                self.logger.info(f"Saving {len(self.position_history)} position records")

            self.logger.info("✅ Position Closer cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ Position Closer cleanup failed: {e}"))
