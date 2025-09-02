# src/tactician/ml_target_updater.py

"""
ML Target Updater for continuously updating ML targets based on real-time conditions.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analyst.ml_dynamic_target_predictor import MLDynamicTargetPredictor
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    warning,
    invalid,
)
from src.utils.centralized_decorators import validate_data_quality


class MLTargetUpdater:
    """
    Continuously monitors active positions and updates their targets based on:
    - Real-time ML predictions
    - Changing market conditions
    - Position performance
    - Risk management rules

    This ensures targets are constantly optimized rather than being set once at entry.
    """

    def __init__(
        self,
        ml_target_predictor: MLDynamicTargetPredictor,
        exchange_client: Any,
        state_manager: Any,
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize the ML Target Updater.

        Args:
            ml_target_predictor: ML dynamic target predictor
            exchange_client: Exchange client for market data
            state_manager: State manager for position tracking
            config: Configuration dictionary
        """
        self.ml_target_predictor = ml_target_predictor
        self.exchange_client = exchange_client
        self.state_manager = state_manager
        self.config = config
        self.logger = system_logger.getChild("MLTargetUpdater")

        # Configuration
        self.updater_config = config.get("ml_target_updater", {})
        self.update_interval = self.updater_config.get("update_interval", 1)  # Real-time updates
        self.max_target_age = self.updater_config.get("max_target_age", 300)  # 5 minutes
        self.confidence_threshold = self.updater_config.get("confidence_threshold", 0.6)

        # Validation configuration
        self.validation_config = self.updater_config.get("validation", {})
        self.statistical_validation = self.validation_config.get("statistical_validation", True)
        self.domain_validation = self.validation_config.get("domain_validation", True)
        self.z_score_threshold = self.validation_config.get("z_score_threshold", 3.0)
        self.iqr_multiplier = self.validation_config.get("iqr_multiplier", 1.5)
        self.price_range_threshold = self.validation_config.get("price_range_threshold", 0.1)  # 10%

        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.target_history: List[Dict[str, Any]] = []
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="ML target updater initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the ML Target Updater."""
        try:
            self.logger.info("Initializing ML Target Updater...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid ML target updater configuration"))
                return False

            # Initialize target predictor
            if not await self.ml_target_predictor.initialize():
                self.logger.error("Failed to initialize ML target predictor")
                return False

            self.logger.info("✅ ML Target Updater initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ ML Target Updater initialization failed: {e}"))
            return False

    def _validate_configuration(self) -> bool:
        """Validate the configuration parameters."""
        try:
            # Validate update interval
            if self.update_interval <= 0:
                self.logger.error(invalid("Update interval must be positive"))
                return False

            # Validate confidence threshold
            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid("Confidence threshold must be between 0 and 1"))
                return False

            # Validate validation thresholds
            if self.z_score_threshold <= 0:
                self.logger.error(invalid("Z-score threshold must be positive"))
                return False

            if self.iqr_multiplier <= 0:
                self.logger.error(invalid("IQR multiplier must be positive"))
                return False

            if self.price_range_threshold <= 0:
                self.logger.error(invalid("Price range threshold must be positive"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    async def start_updating(self) -> bool:
        """Start the ML target updating process."""
        try:
            if self.is_running:
                self.logger.warning(warning("ML target updating already active"))
                return True

            self.is_running = True
            self.update_task = asyncio.create_task(self._updating_loop())

            self.logger.info(f"✅ ML target updating started (interval: {self.update_interval}s)")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to start ML target updating: {e}"))
            return False

    async def stop_updating(self) -> bool:
        """Stop the ML target updating process."""
        try:
            if not self.is_running:
                self.logger.warning(warning("ML target updating not active"))
                return True

            self.is_running = False

            if self.update_task:
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("✅ ML target updating stopped")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to stop ML target updating: {e}"))
            return False

    async def _updating_loop(self) -> None:
        """Main updating loop for ML targets."""
        try:
            while self.is_running:
                # Update targets for all active positions
                await self._update_targets()

                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)

        except asyncio.CancelledError:
            self.logger.info("ML target updating loop cancelled")
        except Exception as e:
            self.logger.error(failed(f"❌ Error in updating loop: {e}"))

    async def _update_targets(self) -> None:
        """Update targets for all active positions."""
        try:
            if not self.active_positions:
                return

            for position_id, position_data in self.active_positions.items():
                updated_target = await self._update_position_target(position_id, position_data)
                
                if updated_target:
                    # Store target update
                    self.target_history.append(updated_target)
                    
                    # Update position with new target
                    self.active_positions[position_id]["current_target"] = updated_target["target"]
                    self.active_positions[position_id]["target_confidence"] = updated_target["confidence"]
                    self.active_positions[position_id]["last_target_update"] = datetime.now()

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating targets: {e}"))

    async def _update_position_target(
        self, 
        position_id: str, 
        position_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update target for a specific position."""
        try:
            # Get current market data
            symbol = position_data["symbol"]
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                return None

            # Get ML prediction for new target
            ml_prediction = await self.ml_target_predictor.predict_target(
                symbol=symbol,
                current_price=current_price,
                position_data=position_data
            )

            if not ml_prediction:
                return None

            # Extract target and confidence
            new_target = ml_prediction.get("target_price")
            confidence = ml_prediction.get("confidence", 0.5)

            # Validate target
            if not self._validate_target(new_target, current_price, position_data):
                self.logger.warning(f"Target validation failed for position {position_id}")
                return None

            # Create target update record
            target_update = {
                "position_id": position_id,
                "symbol": symbol,
                "old_target": position_data.get("current_target"),
                "new_target": new_target,
                "current_price": current_price,
                "confidence": confidence,
                "timestamp": datetime.now(),
                "validation_passed": True,
                "ml_prediction": ml_prediction
            }

            return target_update

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating target for position {position_id}: {e}"))
            return None

    def _validate_target(
        self, 
        target: float, 
        current_price: float, 
        position_data: Dict[str, Any]
    ) -> bool:
        """Validate a target price."""
        try:
            if target is None or target <= 0:
                return False

            # Statistical validation
            if self.statistical_validation:
                if not self._validate_statistical(target, current_price, position_data):
                    return False

            # Domain-specific validation
            if self.domain_validation:
                if not self._validate_domain(target, current_price, position_data):
                    return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Error validating target: {e}"))
            return False

    def _validate_statistical(
        self, 
        target: float, 
        current_price: float, 
        position_data: Dict[str, Any]
    ) -> bool:
        """Validate target using statistical methods."""
        try:
            # Get historical targets for this position
            historical_targets = self._get_historical_targets(position_data["symbol"])
            
            if not historical_targets or len(historical_targets) < 10:
                # Not enough data for statistical validation
                return True

            # Z-score validation
            mean_target = np.mean(historical_targets)
            std_target = np.std(historical_targets)
            
            if std_target > 0:
                z_score = abs(target - mean_target) / std_target
                if z_score > self.z_score_threshold:
                    self.logger.warning(f"Target failed z-score validation: {z_score:.2f} > {self.z_score_threshold}")
                    return False

            # IQR validation
            q1 = np.percentile(historical_targets, 25)
            q3 = np.percentile(historical_targets, 75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - self.iqr_multiplier * iqr
                upper_bound = q3 + self.iqr_multiplier * iqr
                
                if target < lower_bound or target > upper_bound:
                    self.logger.warning(f"Target failed IQR validation: {target:.2f} outside [{lower_bound:.2f}, {upper_bound:.2f}]")
                    return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Error in statistical validation: {e}"))
            return False

    def _validate_domain(
        self, 
        target: float, 
        current_price: float, 
        position_data: Dict[str, Any]
    ) -> bool:
        """Validate target using domain-specific rules."""
        try:
            # Price range validation
            price_change = abs(target - current_price) / current_price
            
            if price_change > self.price_range_threshold:
                self.logger.warning(f"Target failed price range validation: {price_change:.2%} > {self.price_range_threshold:.2%}")
                return False

            # Position direction validation
            side = position_data.get("side", "long")
            entry_price = position_data.get("entry_price", current_price)

            if side == "long":
                # For long positions, target should be above entry price
                if target < entry_price * 0.95:  # Allow 5% below entry
                    self.logger.warning(f"Long position target too low: {target:.2f} < {entry_price * 0.95:.2f}")
                    return False
            else:
                # For short positions, target should be below entry price
                if target > entry_price * 1.05:  # Allow 5% above entry
                    self.logger.warning(f"Short position target too high: {target:.2f} > {entry_price * 1.05:.2f}")
                    return False

            # Market hours validation (if applicable)
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # Outside trading hours
                # Be more conservative with targets outside trading hours
                if price_change > self.price_range_threshold * 0.5:
                    self.logger.warning(f"Target too aggressive outside trading hours: {price_change:.2%}")
                    return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Error in domain validation: {e}"))
            return False

    def _get_historical_targets(self, symbol: str) -> List[float]:
        """Get historical targets for a symbol."""
        try:
            # Get recent targets from history
            recent_targets = []
            for target_record in self.target_history[-100:]:  # Last 100 targets
                if target_record["symbol"] == symbol:
                    recent_targets.append(target_record["new_target"])

            return recent_targets

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting historical targets: {e}"))
            return []

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Implement actual price fetching from exchange client - will be added in future updates
            # For now, return a placeholder
            return 50000.0  # Placeholder price

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting current price: {e}"))
            return None

    def add_position(self, position_data: Dict[str, Any]) -> None:
        """Add a position to the target updating system."""
        try:
            position_id = position_data["position_id"]
            self.active_positions[position_id] = position_data
            self.logger.info(f"✅ Added position {position_id} to target updating")

        except Exception as e:
            self.logger.error(failed(f"❌ Error adding position: {e}"))

    def remove_position(self, position_id: str) -> None:
        """Remove a position from the target updating system."""
        try:
            if position_id in self.active_positions:
                self.active_positions.pop(position_id)
                self.logger.info(f"✅ Removed position {position_id} from target updating")

        except Exception as e:
            self.logger.error(failed(f"❌ Error removing position: {e}"))

    def get_target_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get target update history."""
        try:
            if limit:
                return self.target_history[-limit:]
            return self.target_history.copy()

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting target history: {e}"))
            return []

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        return self.active_positions.copy()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Stop updating
            await self.stop_updating()

            # Cleanup component managers
            if self.ml_target_predictor:
                await self.ml_target_predictor.cleanup()

            self.logger.info("✅ ML Target Updater cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ ML Target Updater cleanup failed: {e}"))


# Setup function for easy integration
async def setup_ml_target_updater(
    ml_target_predictor: MLDynamicTargetPredictor,
    exchange_client: Any,
    state_manager: Any,
    config: Dict[str, Any]
) -> Optional[MLTargetUpdater]:
    """Setup and initialize ML Target Updater."""
    try:
        updater = MLTargetUpdater(ml_target_predictor, exchange_client, state_manager, config)
        if await updater.initialize():
            return updater
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup ML target updater: {e}")
        return None
