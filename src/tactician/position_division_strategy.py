# src/tactician/position_division_strategy.py

"""
Position Division Strategy for tactical position management.
Defines strategies for multiple positions, take profit, stop loss, and position closure.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.error_handler import (
    handle_errors, 
    handle_specific_errors,
    create_retry_strategy,
    create_fallback_strategy,
    create_graceful_degradation_strategy,
    RetryStrategy,
    FallbackStrategy,
    GracefulDegradationStrategy,
)
from src.utils.logger import system_logger


class PositionDivisionStrategy:
    """
    Position division strategy that manages multiple positions, take profit,
    stop loss, and position closure based on ML confidence and short-term analysis.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PositionDivisionStrategy")

        # Load configuration
        self.division_config: dict[str, Any] = self.config.get("position_division", {})
        
        # Position entry thresholds
        self.entry_confidence_threshold: float = self.division_config.get("entry_confidence_threshold", 0.7)
        self.additional_position_threshold: float = self.division_config.get("additional_position_threshold", 0.8)
        self.max_positions: int = self.division_config.get("max_positions", 3)
        
        # Position division parameters
        self.max_division_ratio: float = self.division_config.get("max_division_ratio", 1.0)  # 100% of original position size
        self.division_confidence_threshold: float = self.division_config.get("division_confidence_threshold", 0.85)  # Very high confidence for division
        
        # Load optimized parameters from HPO results if available
        self._load_optimized_parameters()
        
        # Position division parameters
        self.max_position_multiplier: float = self.division_config.get("max_position_multiplier", 1.5)  # 150% max
        self.high_confidence_threshold: float = self.division_config.get("high_confidence_threshold", 0.85)
        self.division_confidence_threshold: float = self.division_config.get("division_confidence_threshold", 0.75)
        
        # Take profit thresholds (confidence-based)
        self.take_profit_confidence_decrease: float = self.division_config.get("take_profit_confidence_decrease", 0.1)
        self.take_profit_short_term_decrease: float = self.division_config.get("take_profit_short_term_decrease", 0.08)
        
        # Stop loss thresholds (confidence-based with trailing stop)
        self.stop_loss_confidence_threshold: float = self.division_config.get("stop_loss_confidence_threshold", 0.3)
        self.stop_loss_short_term_threshold: float = self.division_config.get("stop_loss_short_term_threshold", 0.24)
        self.stop_loss_price_threshold: float = self.division_config.get("stop_loss_price_threshold", -0.05)  # Trailing stop
        
        # Position closure thresholds (confidence-based)
        self.full_close_confidence_threshold: float = self.division_config.get("full_close_confidence_threshold", 0.2)
        self.full_close_short_term_threshold: float = self.division_config.get("full_close_short_term_threshold", 0.16)
        
        # Position holding time limit (12 hours max)
        self.max_position_hold_hours: float = self.division_config.get("max_position_hold_hours", 12.0)
        
        # Enhanced position management thresholds
        self.dynamic_position_sizing = self.division_config.get("dynamic_position_sizing", True)
        self.kelly_criterion_enabled = self.division_config.get("kelly_criterion_enabled", True)
        self.volatility_targeting = self.division_config.get("volatility_targeting", True)
        
        # Advanced profit-taking mechanisms
        self.scaled_profit_taking = self.division_config.get("scaled_profit_taking", True)
        self.profit_targets = self.division_config.get("profit_targets", [0.01, 0.02, 0.03])  # 1%, 2%, 3%
        self.profit_scaling_factors = self.division_config.get("profit_scaling_factors", [0.3, 0.3, 0.4])  # 30%, 30%, 40%
        
        # Enhanced stop-loss mechanisms
        self.trailing_stop_enabled = self.division_config.get("trailing_stop_enabled", True)
        self.atr_multiplier = self.division_config.get("atr_multiplier", 2.0)
        self.confidence_based_stop = self.division_config.get("confidence_based_stop", True)
        

        
        self.is_initialized: bool = False
        self.position_division_history: List[dict[str, Any]] = []
        
    def _load_optimized_parameters(self) -> None:
        """Load optimized parameters from HPO results if available."""
        try:
            # Try to load HPO results from multiple possible locations
            hpo_results_paths = [
                "data/training/optimized_position_division_params.json",
                "data/training/hpo_results.json",
                "data/training/multi_stage_hpo_results.json"
            ]
            
            optimized_params = None
            for path in hpo_results_paths:
                try:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            hpo_data = json.load(f)
                            
                            # Extract position division parameters from HPO results
                            if "best_params" in hpo_data:
                                best_params = hpo_data["best_params"]
                                optimized_params = {
                                    "entry_confidence_threshold": best_params.get("entry_confidence_threshold", self.entry_confidence_threshold),
                                    "additional_position_threshold": best_params.get("additional_position_threshold", self.additional_position_threshold),
                                    "division_confidence_threshold": best_params.get("division_confidence_threshold", self.division_confidence_threshold),
                                    "max_division_ratio": best_params.get("max_division_ratio", self.max_division_ratio),
                                    "max_positions": best_params.get("max_positions", self.max_positions),
                                }
                                self.logger.info(f"✅ Loaded optimized position division parameters from {path}")
                                break
                except Exception as e:
                    self.logger.debug(f"Could not load from {path}: {e}")
                    continue
            
            # Apply optimized parameters if found
            if optimized_params:
                self.entry_confidence_threshold = optimized_params["entry_confidence_threshold"]
                self.additional_position_threshold = optimized_params["additional_position_threshold"]
                self.division_confidence_threshold = optimized_params["division_confidence_threshold"]
                self.max_division_ratio = optimized_params["max_division_ratio"]
                self.max_positions = optimized_params["max_positions"]
                self.logger.info("✅ Applied optimized position division parameters")
            else:
                self.logger.info("ℹ️ Using default position division parameters (no HPO results found)")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading optimized parameters: {e}, using defaults")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position division configuration"),
            AttributeError: (False, "Missing required division parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position division strategy initialization",
        recovery_strategies=[
            create_retry_strategy(max_retries=2, base_delay=1.0),
            create_graceful_degradation_strategy(default_return=False),
        ],
    )
    async def initialize(self) -> bool:
        """Initialize the position division strategy."""
        try:
            self.logger.info("🚀 Initializing position division strategy...")
            self.logger.info(f"📊 Configuration loaded: {len(self.division_config)} parameters")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Configuration validation failed")
                return False

            self.is_initialized = True
            self.logger.info("✅ Position division strategy initialized successfully")
            self.logger.info(f"📋 Key parameters:")
            self.logger.info(f"   - Entry threshold: {self.entry_confidence_threshold:.3f}")
            self.logger.info(f"   - Additional position threshold: {self.additional_position_threshold:.3f}")
            self.logger.info(f"   - Max positions: {self.max_positions}")
            self.logger.info(f"   - Max position hold time: {self.max_position_hold_hours:.1f} hours")

            self.logger.info(f"   - Take profit decrease: {self.take_profit_confidence_decrease:.3f}")
            self.logger.info(f"   - Stop loss threshold: {self.stop_loss_confidence_threshold:.3f}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing position division strategy: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate position division configuration."""
        try:
            self.logger.debug("🔍 Validating position division configuration...")
            
            required_keys = [
                "entry_confidence_threshold", "additional_position_threshold", "max_positions",
                "take_profit_confidence_decrease", "take_profit_short_term_decrease",
                "stop_loss_confidence_threshold", "stop_loss_short_term_threshold", "stop_loss_price_threshold",
                "full_close_confidence_threshold", "full_close_short_term_threshold",
                "max_position_hold_hours"
            ]
            
            self.logger.debug(f"📋 Checking {len(required_keys)} required configuration keys...")
            
            for key in required_keys:
                if key not in self.division_config:
                    self.logger.error(f"❌ Missing required configuration key: {key}")
                    return False
                else:
                    self.logger.debug(f"✅ Found configuration key: {key} = {self.division_config[key]}")

            # Validate numeric parameters
            if self.max_positions <= 0:
                self.logger.error(f"❌ max_positions must be positive, got: {self.max_positions}")
                return False
            
            if self.max_position_hold_hours <= 0:
                self.logger.error(f"❌ max_position_hold_hours must be positive, got: {self.max_position_hold_hours}")
                return False
            


            self.logger.info("✅ Position division configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for position division"),
            AttributeError: (None, "Strategy not properly initialized"),
        },
        default_return=None,
        context="position division analysis",
    )
    async def analyze_position_division(
        self,
        ml_predictions: dict[str, Any],
        current_positions: List[dict[str, Any]],
        current_price: float,
        short_term_analysis: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Analyze position division strategy based on ML confidence and short-term analysis.
        
        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            current_positions: List of current positions
            current_price: Current market price
            short_term_analysis: Short-term price/volume/market microstructure analysis
            
        Returns:
            dict[str, Any]: Position division analysis and recommendations
        """
        try:
            if not self.is_initialized:
                self.logger.error("Position division strategy not initialized")
                return None

            self.logger.info("🔄 Starting position division strategy analysis...")
            self.logger.info(f"📊 Input data: {len(current_positions)} positions, price: ${current_price:.2f}")

            # Extract ML confidence scores
            movement_confidence = ml_predictions.get("movement_confidence_scores", {})
            adverse_movement_risks = ml_predictions.get("adverse_movement_risks", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})
            
            self.logger.info(f"📈 ML confidence levels: {len(movement_confidence)} movement levels")
            self.logger.info(f"⚠️ Adverse movement risks: {len(adverse_movement_risks)} risk levels")
            self.logger.info(f"🎯 Directional confidence: {directional_confidence}")

            # Calculate average confidence
            avg_confidence = self._calculate_average_confidence(movement_confidence)
            self.logger.info(f"📊 Average confidence calculated: {avg_confidence:.3f}")
            
            # Analyze short-term indicators
            short_term_score = self._analyze_short_term_indicators(short_term_analysis)
            self.logger.info(f"⚡ Short-term score: {short_term_score:.3f}")
            
            # Generate position division recommendations
            self.logger.info("🎯 Generating position division recommendations...")
            
            entry_recommendation = self._analyze_entry_strategy(avg_confidence, len(current_positions), short_term_score, current_positions)
            take_profit_recommendation = self._analyze_take_profit_strategy(avg_confidence, current_positions, current_price, short_term_score)
            stop_loss_recommendation = self._analyze_stop_loss_strategy(avg_confidence, current_positions, current_price, short_term_score)
            full_close_recommendation = self._analyze_full_close_strategy(avg_confidence, current_positions, current_price, short_term_score)

            # Log key recommendations
            self.logger.info(f"📝 Entry recommendation: {entry_recommendation.get('should_enter', False)} - {entry_recommendation.get('reason', 'No reason')}")
            self.logger.info(f"💰 Take profit actions: {len(take_profit_recommendation.get('take_profit_actions', []))} positions")
            self.logger.info(f"🛑 Stop loss actions: {len(stop_loss_recommendation.get('stop_loss_actions', []))} positions")
            self.logger.info(f"🚪 Full close actions: {len(full_close_recommendation.get('full_close_actions', []))} positions")

            # Create position division analysis
            division_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "current_positions_count": len(current_positions),
                "average_confidence": avg_confidence,
                "short_term_score": short_term_score,
                "entry_recommendation": entry_recommendation,
                "take_profit_recommendation": take_profit_recommendation,
                "stop_loss_recommendation": stop_loss_recommendation,
                "full_close_recommendation": full_close_recommendation,
                "ml_confidence_scores": movement_confidence,
                "adverse_movement_risks": adverse_movement_risks,
                "directional_confidence": directional_confidence,
                "division_reason": self._generate_division_reason(
                    entry_recommendation, take_profit_recommendation, stop_loss_recommendation, full_close_recommendation, avg_confidence
                ),
            }

            # Store in history (efficient management)
            self.position_division_history.append(division_analysis)
            if len(self.position_division_history) > 100:  # Keep last 100 entries
                self.position_division_history = self.position_division_history[-100:]

            self.logger.info(f"✅ Position division analysis completed successfully")
            self.logger.info(f"📋 Analysis stored in history (total: {len(self.position_division_history)} entries)")
            return division_analysis

        except Exception as e:
            self.logger.error(f"❌ Error analyzing position division: {e}")
            # Return a safe default analysis instead of None
            return {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "current_positions_count": len(current_positions),
                "average_confidence": 0.5,
                "short_term_score": 0.5,
                "entry_recommendation": {"should_enter": False, "confidence": 0.5, "reason": "Error in analysis"},
                "take_profit_recommendation": {"take_profit_actions": [], "total_take_profit_size": 0.0},
                "stop_loss_recommendation": {"stop_loss_actions": [], "total_stop_loss_size": 0.0},
                "full_close_recommendation": {"full_close_actions": [], "total_full_close_size": 0.0},
                "ml_confidence_scores": {},
                "adverse_movement_risks": {},
                "directional_confidence": {},
                "division_reason": "Error in position division analysis",
                "error": str(e),
            }

    def _calculate_average_confidence(self, movement_confidence: dict[str, float]) -> float:
        """Calculate average confidence for key movement levels."""
        try:
            self.logger.debug(f"🔍 Calculating average confidence from {len(movement_confidence)} movement levels")
            
            if not movement_confidence:
                self.logger.warning("⚠️ No movement confidence data available, using default 0.5")
                return 0.5
            
            # Get average confidence for target levels (0.5% to 2.0%)
            target_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            
            for level in target_levels:
                try:
                    # Find closest available level
                    available_levels = [float(k) for k in movement_confidence.keys() if k.replace('.', '').isdigit()]
                    if not available_levels:
                        self.logger.warning(f"⚠️ No numeric levels found in movement_confidence, using default 0.5 for level {level}")
                        confidences.append(0.5)
                        continue
                    
                    closest_level = min(available_levels, key=lambda x: abs(x - level))
                    confidence = movement_confidence.get(str(closest_level), 0.5)
                    confidences.append(confidence)
                    self.logger.debug(f"📊 Target {level}% -> closest {closest_level}% -> confidence {confidence:.3f}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"⚠️ Error processing level {level}: {e}, using default 0.5")
                    confidences.append(0.5)
            
            if not confidences:
                self.logger.warning("⚠️ No valid confidences calculated, using default 0.5")
                return 0.5
            
            avg_confidence = sum(confidences) / len(confidences)
            self.logger.debug(f"📈 Average confidence calculated: {avg_confidence:.3f} from {len(confidences)} levels")
            
            return avg_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating average confidence: {e}")
            return 0.5

    def _analyze_short_term_indicators(self, short_term_analysis: Optional[Dict[str, Any]]) -> float:
        """Analyze short-term ML confidence scores for 1m-5m timeframes."""
        try:
            if not short_term_analysis:
                self.logger.debug("📊 No short-term analysis provided, using default score 0.5")
                return 0.5
            
            # Extract short-term ML confidence scores
            short_term_ml_scores = short_term_analysis.get("ml_confidence_scores", {})
            self.logger.debug(f"📈 Short-term ML scores available: {list(short_term_ml_scores.keys())}")
            
            # Get confidence scores for 1m and 5m timeframes
            confidence_1m = short_term_ml_scores.get("1m", {}).get("confidence", 0.5)
            confidence_5m = short_term_ml_scores.get("5m", {}).get("confidence", 0.5)
            
            self.logger.debug(f"⚡ 1m confidence: {confidence_1m:.3f}, 5m confidence: {confidence_5m:.3f}")
            
            # Calculate weighted short-term score (5m has more weight than 1m)
            short_term_score = (confidence_1m * 0.4 + confidence_5m * 0.6)
            short_term_score = max(0.0, min(1.0, short_term_score))
            
            self.logger.debug(f"📊 Weighted short-term score: {short_term_score:.3f}")
            
            return short_term_score
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing short-term indicators: {e}")
            return 0.5

    def _analyze_entry_strategy(self, avg_confidence: float, current_positions_count: int, short_term_score: float, current_positions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Analyze entry strategy for additional positions based on ML confidence scores."""
        try:
            self.logger.debug(f"🎯 Analyzing entry strategy: {current_positions_count}/{self.max_positions} positions")
            self.logger.debug(f"📊 Input scores - Short-term: {short_term_score:.3f}, Overall: {avg_confidence:.3f}")
            
            # Use short-term ML confidence (1m-5m) for entry decisions
            # Higher weight on short-term confidence for precise entry timing
            combined_score = (short_term_score * 0.7 + avg_confidence * 0.3)
            self.logger.debug(f"📈 Combined score: {combined_score:.3f} (threshold: {self.entry_confidence_threshold:.3f})")
            
            should_enter = False
            reason = ""
            
            # Check for positions that have exceeded the 12-hour holding limit
            if current_positions:
                current_time = datetime.now()
                for position in current_positions:
                    entry_timestamp = position.get("entry_timestamp")
                    if entry_timestamp:
                        try:
                            if isinstance(entry_timestamp, str):
                                entry_time = datetime.fromisoformat(entry_timestamp.replace('Z', '+00:00'))
                            else:
                                entry_time = entry_timestamp
                            
                            hours_in_position = (current_time - entry_time).total_seconds() / 3600
                            if hours_in_position >= self.max_position_hold_hours:
                                self.logger.warning(f"⚠️ Position {position.get('position_id', 'unknown')} has exceeded {self.max_position_hold_hours}h limit ({hours_in_position:.1f}h)")
                                reason = f"Position holding time limit exceeded ({hours_in_position:.1f}h >= {self.max_position_hold_hours}h)"
                                return {
                                    "should_enter": False,
                                    "position_size": 0.0,
                                    "confidence": combined_score,
                                    "short_term_confidence": short_term_score,
                                    "overall_confidence": avg_confidence,
                                    "reason": reason,
                                    "max_positions_reached": False,
                                    "holding_time_exceeded": True,
                                }
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"⚠️ Error parsing entry timestamp: {e}")
                            continue
            
            # Check if we should enter a new position
            if current_positions_count < self.max_positions:
                if combined_score >= self.entry_confidence_threshold:
                    should_enter = True
                    position_size = min(combined_score * 0.1, self.max_position_size)  # Cap at max position size
                    reason = f"High short-term confidence ({short_term_score:.2f}) and good overall confidence ({avg_confidence:.2f})"
                    self.logger.info(f"✅ Entry recommended: High confidence ({combined_score:.3f}) - Size: {position_size:.3f}")
                elif combined_score >= self.additional_position_threshold:
                    should_enter = True
                    position_size = min(combined_score * 0.05, self.max_position_size * 0.5)  # Smaller position size, capped
                    reason = f"Moderate short-term confidence ({short_term_score:.2f}) for additional position"
                    self.logger.info(f"⚠️ Entry recommended: Moderate confidence ({combined_score:.3f}) - Size: {position_size:.3f}")
                else:
                    self.logger.debug(f"❌ Entry rejected: Low confidence ({combined_score:.3f}) < threshold ({self.entry_confidence_threshold:.3f})")
            else:
                self.logger.debug(f"❌ Entry rejected: Max positions reached ({current_positions_count}/{self.max_positions})")
            
            result = {
                "should_enter": should_enter,
                "confidence": combined_score,
                "short_term_confidence": short_term_score,
                "overall_confidence": avg_confidence,
                "reason": reason,
                "max_positions_reached": current_positions_count >= self.max_positions,
            }
            
            self.logger.debug(f"📋 Entry analysis result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing entry strategy: {e}")
            return {"should_enter": False, "confidence": 0.0, "reason": "Error in analysis"}

    def _analyze_take_profit_strategy(self, avg_confidence: float, current_positions: List[Dict[str, Any]], current_price: float, short_term_score: float) -> Dict[str, Any]:
        """Analyze take profit strategy based on confidence decreases."""
        try:
            self.logger.debug(f"💰 Analyzing take profit strategy for {len(current_positions)} positions")
            self.logger.debug(f"📊 Current confidence: {avg_confidence:.3f}, Short-term: {short_term_score:.3f}")
            
            take_profit_actions = []
            total_take_profit_size = 0.0
            
            for i, position in enumerate(current_positions):
                entry_price = position.get("entry_price", current_price)
                position_size = position.get("position_size", 0.0)
                position_id = position.get("position_id", f"pos_{i}")
                entry_confidence = position.get("entry_confidence", avg_confidence)
                
                self.logger.debug(f"📋 Position {position_id}: Entry confidence {entry_confidence:.3f}, Size {position_size:.3f}")
                
                # Calculate confidence decrease
                confidence_decrease = entry_confidence - avg_confidence
                short_term_confidence_decrease = entry_confidence - short_term_score
                
                self.logger.debug(f"📉 Confidence decreases - Overall: {confidence_decrease:.3f}, Short-term: {short_term_confidence_decrease:.3f}")
                
                # Determine take profit action based on confidence decreases (gradual approach)
                should_take_profit = False
                take_profit_size = 0.0
                reason = ""
                
                # Calculate gradual take profit based on confidence decrease severity
                if confidence_decrease >= self.take_profit_confidence_decrease:
                    # Large confidence decrease - take profit on 50% of position
                    should_take_profit = True
                    take_profit_size = position_size * 0.5
                    reason = f"Large confidence decrease ({confidence_decrease:.2f}) - 50% take profit"
                    self.logger.info(f"💰 Take profit recommended for {position_id}: Large confidence decrease ({confidence_decrease:.3f}) - Size: {take_profit_size:.3f}")
                
                elif short_term_confidence_decrease >= self.take_profit_short_term_decrease:
                    # Short-term confidence decrease - take profit on 30% of position
                    should_take_profit = True
                    take_profit_size = position_size * 0.3
                    reason = f"Short-term confidence decreased ({short_term_confidence_decrease:.2f}) - 30% take profit"
                    self.logger.info(f"💰 Take profit recommended for {position_id}: Short-term decrease ({short_term_confidence_decrease:.3f}) - Size: {take_profit_size:.3f}")
                
                elif confidence_decrease >= self.take_profit_confidence_decrease * 0.7:
                    # Moderate confidence decrease - take profit on 25% of position
                    should_take_profit = True
                    take_profit_size = position_size * 0.25
                    reason = f"Moderate confidence decrease ({confidence_decrease:.2f}) - 25% take profit"
                    self.logger.info(f"💰 Take profit recommended for {position_id}: Moderate decrease ({confidence_decrease:.3f}) - Size: {take_profit_size:.3f}")
                
                elif confidence_decrease >= self.take_profit_confidence_decrease * 0.5:
                    # Small confidence decrease - take profit on 15% of position
                    should_take_profit = True
                    take_profit_size = position_size * 0.15
                    reason = f"Small confidence decrease ({confidence_decrease:.2f}) - 15% take profit"
                    self.logger.info(f"💰 Take profit recommended for {position_id}: Small decrease ({confidence_decrease:.3f}) - Size: {take_profit_size:.3f}")
                
                elif confidence_decrease >= self.take_profit_confidence_decrease * 0.3:
                    # Very small confidence decrease - take profit on 10% of position
                    should_take_profit = True
                    take_profit_size = position_size * 0.1
                    reason = f"Very small confidence decrease ({confidence_decrease:.2f}) - 10% take profit"
                    self.logger.info(f"💰 Take profit recommended for {position_id}: Very small decrease ({confidence_decrease:.3f}) - Size: {take_profit_size:.3f}")
                
                else:
                    self.logger.debug(f"❌ No take profit for {position_id}: Insufficient confidence decrease ({confidence_decrease:.3f})")
                
                take_profit_actions.append({
                    "position_id": position_id,
                    "should_take_profit": should_take_profit,
                    "take_profit_size": take_profit_size,
                    "confidence_decrease": confidence_decrease,
                    "short_term_confidence_decrease": short_term_confidence_decrease,
                    "reason": reason,
                })
                
                total_take_profit_size += take_profit_size
            
            result = {
                "take_profit_actions": take_profit_actions,
                "total_take_profit_size": total_take_profit_size,
            }
            
            self.logger.debug(f"📋 Take profit analysis: {len([a for a in take_profit_actions if a['should_take_profit']])} actions, Total size: {total_take_profit_size:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing take profit strategy: {e}")
            return {"take_profit_actions": [], "total_take_profit_size": 0.0}

    def _analyze_stop_loss_strategy(self, avg_confidence: float, current_positions: List[Dict[str, Any]], current_price: float, short_term_score: float) -> Dict[str, Any]:
        """Analyze stop loss strategy based on confidence and trailing stop."""
        try:
            self.logger.debug(f"🛑 Analyzing stop loss strategy for {len(current_positions)} positions")
            self.logger.debug(f"📊 Current confidence: {avg_confidence:.3f}, Short-term: {short_term_score:.3f}")
            self.logger.debug(f"💰 Current price: ${current_price:.2f}")
            
            stop_loss_actions = []
            total_stop_loss_size = 0.0
            
            for i, position in enumerate(current_positions):
                entry_price = position.get("entry_price", current_price)
                position_size = position.get("position_size", 0.0)
                position_id = position.get("position_id", f"pos_{i}")
                entry_confidence = position.get("entry_confidence", avg_confidence)
                
                # Calculate price change for trailing stop
                price_change = (current_price - entry_price) / entry_price
                
                self.logger.debug(f"📋 Position {position_id}: Entry ${entry_price:.2f}, Size {position_size:.3f}, Entry confidence {entry_confidence:.3f}")
                self.logger.debug(f"📉 Price change: {price_change:.2%}, Confidence thresholds: {self.stop_loss_confidence_threshold:.3f}/{self.stop_loss_short_term_threshold:.3f}")
                
                # Determine stop loss action based on confidence and trailing stop (gradual approach)
                should_stop_loss = False
                stop_loss_size = 0.0
                reason = ""
                
                # Full stop loss if confidence is very low
                if avg_confidence <= self.stop_loss_confidence_threshold:
                    should_stop_loss = True
                    stop_loss_size = position_size  # Full stop loss
                    reason = f"Very low confidence ({avg_confidence:.2f}) - full stop loss"
                    self.logger.warning(f"🛑 Full stop loss recommended for {position_id}: Very low confidence ({avg_confidence:.3f})")
                
                # Stop loss if short-term confidence is very low
                elif short_term_score <= self.stop_loss_short_term_threshold:
                    should_stop_loss = True
                    stop_loss_size = position_size  # Full stop loss
                    reason = f"Very low short-term confidence ({short_term_score:.2f}) - full stop loss"
                    self.logger.warning(f"🛑 Full stop loss recommended for {position_id}: Very low short-term confidence ({short_term_score:.3f})")
                
                # Trailing stop loss if price moved against us significantly
                elif price_change <= self.stop_loss_price_threshold:
                    should_stop_loss = True
                    stop_loss_size = position_size  # Full stop loss
                    reason = f"Trailing stop loss triggered: price change ({price_change:.2%})"
                    self.logger.warning(f"🛑 Trailing stop loss triggered for {position_id}: Price change {price_change:.2%}")
                
                # Gradual stop loss based on confidence severity
                elif avg_confidence <= self.stop_loss_confidence_threshold * 1.2:
                    should_stop_loss = True
                    stop_loss_size = position_size * 0.75  # 75% stop loss
                    reason = f"Very low confidence ({avg_confidence:.2f}) - 75% stop loss"
                    self.logger.warning(f"🛑 75% stop loss recommended for {position_id}: Very low confidence ({avg_confidence:.3f})")
                
                elif avg_confidence <= self.stop_loss_confidence_threshold * 1.5:
                    should_stop_loss = True
                    stop_loss_size = position_size * 0.5  # 50% stop loss
                    reason = f"Low confidence ({avg_confidence:.2f}) - 50% stop loss"
                    self.logger.info(f"🛑 50% stop loss recommended for {position_id}: Low confidence ({avg_confidence:.3f})")
                
                elif avg_confidence <= self.stop_loss_confidence_threshold * 2.0:
                    should_stop_loss = True
                    stop_loss_size = position_size * 0.25  # 25% stop loss
                    reason = f"Moderate confidence decrease ({avg_confidence:.2f}) - 25% stop loss"
                    self.logger.info(f"🛑 25% stop loss recommended for {position_id}: Moderate confidence ({avg_confidence:.3f})")
                
                elif avg_confidence <= self.stop_loss_confidence_threshold * 2.5:
                    should_stop_loss = True
                    stop_loss_size = position_size * 0.1  # 10% stop loss
                    reason = f"Small confidence decrease ({avg_confidence:.2f}) - 10% stop loss"
                    self.logger.info(f"🛑 10% stop loss recommended for {position_id}: Small confidence decrease ({avg_confidence:.3f})")
                
                else:
                    self.logger.debug(f"❌ No stop loss for {position_id}: Confidence and price within acceptable ranges")
                
                stop_loss_actions.append({
                    "position_id": position_id,
                    "should_stop_loss": should_stop_loss,
                    "stop_loss_size": stop_loss_size,
                    "price_change": price_change,
                    "confidence_decrease": entry_confidence - avg_confidence,
                    "short_term_confidence_decrease": entry_confidence - short_term_score,
                    "reason": reason,
                })
                
                total_stop_loss_size += stop_loss_size
            
            result = {
                "stop_loss_actions": stop_loss_actions,
                "total_stop_loss_size": total_stop_loss_size,
            }
            
            self.logger.debug(f"📋 Stop loss analysis: {len([a for a in stop_loss_actions if a['should_stop_loss']])} actions, Total size: {total_stop_loss_size:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing stop loss strategy: {e}")
            return {"stop_loss_actions": [], "total_stop_loss_size": 0.0}

    def _analyze_full_close_strategy(self, avg_confidence: float, current_positions: List[Dict[str, Any]], current_price: float, short_term_score: float) -> Dict[str, Any]:
        """Analyze full position closure strategy based on confidence."""
        try:
            self.logger.debug(f"🚪 Analyzing full close strategy for {len(current_positions)} positions")
            self.logger.debug(f"📊 Current confidence: {avg_confidence:.3f}, Short-term: {short_term_score:.3f}")
            self.logger.debug(f"💰 Current price: ${current_price:.2f}")
            
            full_close_actions = []
            total_full_close_size = 0.0
            
            for i, position in enumerate(current_positions):
                entry_price = position.get("entry_price", current_price)
                position_size = position.get("position_size", 0.0)
                position_id = position.get("position_id", f"pos_{i}")
                entry_confidence = position.get("entry_confidence", avg_confidence)
                
                # Calculate price change for reference
                price_change = (current_price - entry_price) / entry_price
                
                self.logger.debug(f"📋 Position {position_id}: Entry ${entry_price:.2f}, Size {position_size:.3f}, Entry confidence {entry_confidence:.3f}")
                self.logger.debug(f"📉 Price change: {price_change:.2%}")
                
                # Determine full close action based on confidence
                should_full_close = False
                reason = ""
                
                # Full close if both overall and short-term confidence are very low
                if (avg_confidence <= self.full_close_confidence_threshold and 
                    short_term_score <= self.full_close_short_term_threshold):
                    should_full_close = True
                    reason = f"Full close: very low overall confidence ({avg_confidence:.2f}) and short-term confidence ({short_term_score:.2f})"
                    self.logger.warning(f"🚪 Full close recommended for {position_id}: Very low confidence levels")
                
                # Full close if overall confidence dropped dramatically
                elif avg_confidence <= self.full_close_confidence_threshold * 0.5:
                    should_full_close = True
                    reason = f"Full close: extremely low overall confidence ({avg_confidence:.2f})"
                    self.logger.warning(f"🚪 Full close recommended for {position_id}: Extremely low overall confidence ({avg_confidence:.3f})")
                
                # Full close if short-term confidence dropped dramatically
                elif short_term_score <= self.full_close_short_term_threshold * 0.5:
                    should_full_close = True
                    reason = f"Full close: extremely low short-term confidence ({short_term_score:.2f})"
                    self.logger.warning(f"🚪 Full close recommended for {position_id}: Extremely low short-term confidence ({short_term_score:.3f})")
                
                # Full close when there's over 50% chance of price reversal
                # 70% weight on short-term, 30% on overall confidence
                reversal_probability = self._calculate_reversal_probability(entry_confidence, avg_confidence, short_term_score)
                if reversal_probability > 0.5:
                    should_full_close = True
                    reason = f"Full close: {reversal_probability:.1%} chance of price reversal (70% short-term, 30% overall)"
                    self.logger.warning(f"🚪 Full close recommended for {position_id}: High reversal probability ({reversal_probability:.1%})")
                else:
                    self.logger.debug(f"📊 Reversal probability for {position_id}: {reversal_probability:.1%} (below 50% threshold)")
                
                # Calculate reversal probability for all positions (for monitoring)
                reversal_probability = self._calculate_reversal_probability(entry_confidence, avg_confidence, short_term_score)
                
                full_close_actions.append({
                    "position_id": position_id,
                    "should_full_close": should_full_close,
                    "position_size": position_size if should_full_close else 0.0,
                    "price_change": price_change,
                    "confidence_decrease": entry_confidence - avg_confidence,
                    "short_term_confidence_decrease": entry_confidence - short_term_score,
                    "reversal_probability": reversal_probability,
                    "reason": reason,
                })
                
                if should_full_close:
                    total_full_close_size += position_size
            
            result = {
                "full_close_actions": full_close_actions,
                "total_full_close_size": total_full_close_size,
            }
            
            self.logger.debug(f"📋 Full close analysis: {len([a for a in full_close_actions if a['should_full_close']])} actions, Total size: {total_full_close_size:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing full close strategy: {e}")
            return {"full_close_actions": [], "total_full_close_size": 0.0}

    def _calculate_reversal_probability(self, entry_confidence: float, current_confidence: float, short_term_confidence: float) -> float:
        """
        Calculate probability of price reversal based on confidence changes.
        70% weight on short-term confidence, 30% on overall confidence.
        
        Args:
            entry_confidence: Confidence at position entry
            current_confidence: Current overall confidence
            short_term_confidence: Current short-term confidence (1m-5m)
            
        Returns:
            float: Probability of price reversal (0.0 to 1.0)
        """
        try:
            self.logger.debug(f"🔄 Calculating reversal probability - Entry: {entry_confidence:.3f}, Current: {current_confidence:.3f}, Short-term: {short_term_confidence:.3f}")
            
            # Calculate confidence changes (can be positive or negative)
            overall_confidence_change = entry_confidence - current_confidence
            short_term_confidence_change = entry_confidence - short_term_confidence
            
            self.logger.debug(f"📉 Confidence changes - Overall: {overall_confidence_change:.3f}, Short-term: {short_term_confidence_change:.3f}")
            
            # Only consider decreases for reversal probability (positive changes mean confidence improved)
            overall_confidence_decrease = max(0.0, overall_confidence_change)
            short_term_confidence_decrease = max(0.0, short_term_confidence_change)
            
            # Normalize confidence decreases to 0-1 scale
            max_confidence_decrease = 1.0  # Maximum possible decrease
            
            # Calculate reversal probability components
            overall_reversal_prob = min(1.0, overall_confidence_decrease / max_confidence_decrease)
            short_term_reversal_prob = min(1.0, short_term_confidence_decrease / max_confidence_decrease)
            
            self.logger.debug(f"📊 Raw reversal probabilities - Overall: {overall_reversal_prob:.3f}, Short-term: {short_term_reversal_prob:.3f}")
            
            # Weighted combination: 70% short-term, 30% overall
            reversal_probability = (
                short_term_reversal_prob * 0.7 + 
                overall_reversal_prob * 0.3
            )
            
            self.logger.debug(f"📈 Weighted reversal probability: {reversal_probability:.3f}")
            
            # Apply non-linear scaling for more realistic probability
            # Small confidence decreases = low reversal probability
            # Large confidence decreases = high reversal probability
            if reversal_probability <= 0.1:
                scaled_probability = reversal_probability * 0.5  # Very low probability
                self.logger.debug(f"📊 Very low probability scaling: {scaled_probability:.3f}")
            elif reversal_probability <= 0.3:
                scaled_probability = reversal_probability * 0.8  # Low probability
                self.logger.debug(f"📊 Low probability scaling: {scaled_probability:.3f}")
            elif reversal_probability <= 0.5:
                scaled_probability = reversal_probability * 1.2  # Moderate probability
                self.logger.debug(f"📊 Moderate probability scaling: {scaled_probability:.3f}")
            elif reversal_probability <= 0.7:
                scaled_probability = reversal_probability * 1.5  # High probability
                self.logger.debug(f"📊 High probability scaling: {scaled_probability:.3f}")
            else:
                scaled_probability = min(1.0, reversal_probability * 1.8)  # Very high probability
                self.logger.debug(f"📊 Very high probability scaling: {scaled_probability:.3f}")
            
            final_probability = max(0.0, min(1.0, scaled_probability))
            self.logger.debug(f"🎯 Final reversal probability: {final_probability:.3f}")
            
            return final_probability
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating reversal probability: {e}")
            return 0.0

    def _generate_division_reason(
        self,
        entry_recommendation: Dict[str, Any],
        take_profit_recommendation: Dict[str, Any],
        stop_loss_recommendation: Dict[str, Any],
        full_close_recommendation: Dict[str, Any],
        avg_confidence: float,
    ) -> str:
        """Generate reason for position division decisions."""
        try:
            self.logger.debug(f"📝 Generating division reason with confidence {avg_confidence:.3f}")
            
            actions = []
            
            if entry_recommendation.get("should_enter", False):
                actions.append("enter_new_position")
                self.logger.debug("📝 Adding 'enter_new_position' to actions")
            
            if take_profit_recommendation.get("total_take_profit_size", 0.0) > 0:
                actions.append("take_profit")
                self.logger.debug(f"📝 Adding 'take_profit' to actions (size: {take_profit_recommendation.get('total_take_profit_size', 0.0):.3f})")
            
            if stop_loss_recommendation.get("total_stop_loss_size", 0.0) > 0:
                actions.append("stop_loss")
                self.logger.debug(f"📝 Adding 'stop_loss' to actions (size: {stop_loss_recommendation.get('total_stop_loss_size', 0.0):.3f})")
            
            if full_close_recommendation.get("total_full_close_size", 0.0) > 0:
                actions.append("full_close")
                self.logger.debug(f"📝 Adding 'full_close' to actions (size: {full_close_recommendation.get('total_full_close_size', 0.0):.3f})")
            
            if not actions:
                reason = f"No position actions recommended with confidence {avg_confidence:.2f}"
                self.logger.debug(f"📝 Generated reason: {reason}")
                return reason
            else:
                reason = f"Position actions: {', '.join(actions)} based on confidence {avg_confidence:.2f}"
                self.logger.debug(f"📝 Generated reason: {reason}")
                return reason
                
        except Exception as e:
            self.logger.error(f"Error generating division reason: {e}")
            return "Position division analysis completed"

    def get_position_division_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get position division history."""
        if limit:
            return self.position_division_history[-limit:]
        return self.position_division_history.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position division strategy cleanup",
        recovery_strategies=[
            create_graceful_degradation_strategy(default_return=None),
        ],
    )
    async def stop(self) -> None:
        """Stop the position division strategy."""
        try:
            self.logger.info("Stopping position division strategy...")
            self.is_initialized = False
            self.logger.info("✅ Position division strategy stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping position division strategy: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position division strategy setup",
    recovery_strategies=[
        create_retry_strategy(max_retries=1, base_delay=0.5),
        create_graceful_degradation_strategy(default_return=None),
    ],
)
async def setup_position_division_strategy(
    config: Optional[Dict[str, Any]] = None,
) -> Optional[PositionDivisionStrategy]:
    """
    Setup position division strategy.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[PositionDivisionStrategy]: Initialized position division strategy or None
    """
    try:
        if config is None:
            config = {}

        position_division_strategy = PositionDivisionStrategy(config)
        
        if await position_division_strategy.initialize():
            return position_division_strategy
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up position division strategy: {e}")
        return None 