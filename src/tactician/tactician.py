import asyncio
import time
from typing import Dict, Any, Optional, Union, Tuple  # Ensure Union is imported
import uuid  # Import uuid for unique trade IDs
import datetime  # Import datetime for timestamps
import pandas as pd

from exchange.binance import BinanceExchange
from src.utils.logger import system_logger
from src.config import settings, CONFIG  # Import CONFIG for fees etc.
from src.utils.state_manager import StateManager
from src.supervisor.performance_reporter import (
    PerformanceReporter,
)  # Import PerformanceReporter
from src.utils.error_handler import (
    handle_errors,
)
from src.analyst.ml_dynamic_target_predictor import MLDynamicTargetPredictor
from src.tactician.ml_target_updater import MLTargetUpdater
from src.tactician.ml_target_validator import MLTargetValidator


class Tactician:
    """
    The Tactician translates the Analyst's rich intelligence into a high-level trading plan.
    It now uses detailed technical analysis (VWAP, MAs, etc.) to formulate its strategy.
    Also responsible for generating detailed trade logs.
    """

    def __init__(
        self,
        exchange_client: Optional[BinanceExchange] = None,
        state_manager: Optional[StateManager] = None,
        performance_reporter: Optional[PerformanceReporter] = None,
    ):  # Added performance_reporter
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.performance_reporter = performance_reporter  # Store reporter instance
        self.logger = system_logger.getChild("Tactician")
        self.config = settings.get("tactician", {})
        self.trade_symbol = settings.trade_symbol

        # Initialize ML Dynamic Target Predictor
        self.ml_target_predictor = MLDynamicTargetPredictor(settings)

        # Initialize ML Target Updater
        self.ml_target_updater = MLTargetUpdater(
            ml_target_predictor=self.ml_target_predictor,
            exchange_client=exchange_client,
            state_manager=state_manager,
            config=settings,
        )

        # Initialize ML Target Validator
        self.ml_target_validator = MLTargetValidator(settings)

        # Initialize current_position from state_manager if available, otherwise default
        # Ensure state_manager is not None before calling get_state
        self.current_position = (
            self.state_manager.get_state(
                "current_position", self._get_default_position()
            )
            if self.state_manager
            else self._get_default_position()
        )
        self.logger.info(
            f"Tactician initialized. Position: {self.current_position.get('direction')}"
        )
        self.last_analyst_timestamp = None

    def _get_default_position(self):
        """Returns the default structure for an empty position."""
        return {
            "direction": None,
            "size": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "leverage": 1,
            "entry_timestamp": 0.0,
            "trade_id": None,  # Add trade ID
            "entry_fees_usd": 0.0,  # Initialize entry fees
            "entry_context": {},  # Store all decision-making context at entry
        }

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="tactician_start"
    )
    async def start(self):
        """Starts the main tactician loop."""
        # Fixed: Explicitly check if state_manager and exchange are not None
        if self.state_manager is None or self.exchange is None:
            self.logger.error(
                "Tactician cannot start: StateManager or Exchange client not provided."
            )
            return

        self.logger.info("Tactician started. Waiting for new analyst intelligence...")

        # Start ML target updater in background
        asyncio.create_task(self.ml_target_updater.start_monitoring())

        while True:
            try:
                if self.state_manager.get_state("is_trading_paused", False):
                    await asyncio.sleep(10)
                    continue

                analyst_intelligence = self.state_manager.get_state(
                    "analyst_intelligence"
                )

                if (
                    analyst_intelligence
                    and analyst_intelligence.get("timestamp")
                    != self.last_analyst_timestamp
                ):
                    self.last_analyst_timestamp = analyst_intelligence.get("timestamp")
                    self.logger.info(
                        "New analyst intelligence detected. Running tactical assessment."
                    )

                    await self.run_tactical_assessment(analyst_intelligence)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Tactician task cancelled.")
                break
            except Exception as e:
                self.logger.error(
                    f"An error occurred in the Tactician loop: {e}", exc_info=True
                )
                await asyncio.sleep(10)

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="run_tactical_assessment"
    )
    async def run_tactical_assessment(self, analyst_intel: Dict):
        """
        Assesses the current market situation based on Analyst intelligence
        and decides whether to enter or exit a position.
        """
        self.logger.info("--- Starting Tactical Assessment ---")

        # 1. Check for exit conditions if a position is open
        if self.current_position.get("size", 0) > 0:
            exit_decision = self._check_exit_conditions(analyst_intel)
            if exit_decision:
                # Fixed: Ensure exit_price is a float before passing
                exit_price_val = exit_decision.get("exit_price")
                if exit_price_val is not None:
                    await self.execute_close_position(
                        exit_decision["reason"], analyst_intel, float(exit_price_val)
                    )
                else:
                    self.logger.warning(
                        f"Exit decision made but no valid exit_price provided for reason: {exit_decision['reason']}. Closing at current market price."
                    )
                    await self.execute_close_position(
                        exit_decision["reason"], analyst_intel
                    )  # Close at current market price
                return
            self.logger.info("Holding open position. No exit conditions met.")
            return

        # 2. If no position is open, assess for a new entry
        entry_decision = self._prepare_trade_decision(analyst_intel)

        if entry_decision:
            await self.execute_open_position(
                entry_decision, analyst_intel
            )  # Pass analyst_intel for entry context
        else:
            self.logger.info("Holding. No entry conditions met.")

    def _prepare_trade_decision(self, analyst_intel: Dict) -> Union[Dict, None]:
        """Prepares a trade decision based on analyst intelligence with micro-movement detection."""
        # Extract and validate signal data
        signal_data = self._extract_signal_data(analyst_intel)
        if not signal_data:
            return None

        signal, confidence, technical_analysis_data, current_price, current_atr = (
            signal_data
        )

        # Validate confidence threshold
        if not self._validate_confidence(confidence):
            return None

        # Detect micro-movements and market conditions for high-leverage opportunities
        market_conditions = self._detect_micro_movements(analyst_intel)

        # Log market conditions for debugging
        if market_conditions.get("opportunity_type") != "STANDARD":
            self.logger.info(f"Special market conditions detected: {market_conditions}")

        # Prepare order parameters based on signal
        order_params = self._prepare_order_parameters(
            signal, technical_analysis_data, current_atr
        )
        if not order_params:
            return None

        # Calculate enhanced position sizing and risk management with market conditions
        risk_params = self._calculate_enhanced_risk_parameters(
            current_price,
            order_params["stop_loss"],
            analyst_intel,
            confidence,
            market_conditions,
        )

        # Build final decision
        decision = self._build_trade_decision(
            signal, confidence, order_params, risk_params, current_price, current_atr
        )

        # Add market conditions to decision for logging
        decision["market_conditions"] = market_conditions

        self.logger.info(
            f"Trade decision prepared: {signal} {order_params['side']} @ {order_params['entry_price']} "
            f"(SL: {order_params['stop_loss']}, TP: {order_params['take_profit']}, "
            f"Size: {risk_params['position_size']:.4f}, Leverage: {risk_params['leverage']}x, "
            f"Opportunity: {market_conditions.get('opportunity_type', 'STANDARD')})"
        )
        return decision

    def _extract_signal_data(
        self, analyst_intel: Dict
    ) -> Optional[Tuple[str, float, Dict, float, float]]:
        """Extract and validate signal data from analyst intelligence."""
        signal = analyst_intel.get("ensemble_prediction", "HOLD")
        confidence = analyst_intel.get("ensemble_confidence", 0.0)
        technical_analysis_data = analyst_intel.get("technical_signals", {})

        # Essential data checks
        current_price = technical_analysis_data.get("current_price")
        current_atr = technical_analysis_data.get("ATR")

        if current_price is None or current_atr is None or current_atr <= 0:
            self.logger.warning(
                f"Cannot prepare trade decision: current price ({current_price}) or ATR ({current_atr}) data is missing or invalid from technical_signals."
            )
            return None

        return signal, confidence, technical_analysis_data, current_price, current_atr

    def _validate_confidence(self, confidence: float) -> bool:
        """Validate if confidence meets minimum threshold."""
        min_confidence = self.config.get("min_confidence_for_entry", 0.65)
        if confidence < min_confidence:
            self.logger.info(
                f"Signal confidence ({confidence:.2f}) is below threshold ({min_confidence}). No action."
            )
            return False
        return True

    def _detect_micro_movements(self, analyst_intel: Dict) -> Dict[str, Any]:
        """Detect micro price movements and market conditions for high-leverage opportunities."""
        current_price = analyst_intel.get("current_price", 0)
        technical_analysis = analyst_intel.get("technical_analysis", {})

        # Get recent price data from technical analysis
        recent_klines = technical_analysis.get("recent_klines", [])
        if not recent_klines or len(recent_klines) < 2:
            self.logger.warning(
                "Insufficient recent klines data for micro-movement detection"
            )
            return {
                "micro_movement": False,
                "movement_size": 0,
                "opportunity_type": "STANDARD",
            }

        # Calculate recent price movement
        prev_price = recent_klines[-2].get("close", current_price)
        price_change = abs(current_price - prev_price) / prev_price

        # Detect micro movements (less than 0.2% by default)
        micro_movement_threshold = self.config.get("micro_movement_threshold", 0.002)
        is_micro_movement = price_change <= micro_movement_threshold

        # Detect huge candles (more than 5% by default)
        huge_candle_threshold = self.config.get("huge_candle_threshold", 0.05)
        is_huge_candle = price_change >= huge_candle_threshold

        # Detect S/R zone proximity using S/R levels from analyst
        sr_levels = technical_analysis.get("sr_levels", [])
        is_near_sr = False
        nearest_sr_distance = float("inf")
        nearest_sr_level = None

        for sr_level in sr_levels:
            if isinstance(sr_level, dict):
                level_price = sr_level.get("level_price", 0)
            else:
                level_price = float(sr_level)

            distance = abs(current_price - level_price) / current_price
            sr_zone_proximity = self.config.get("sr_zone_proximity", 0.01)

            if distance <= sr_zone_proximity:
                is_near_sr = True
                if distance < nearest_sr_distance:
                    nearest_sr_distance = distance
                    nearest_sr_level = sr_level

        # Determine opportunity type
        opportunity_type = self._determine_opportunity_type(
            is_micro_movement, is_huge_candle, is_near_sr
        )

        # Enhanced logging for micro-movement opportunities
        if opportunity_type != "STANDARD":
            self.logger.info("🎯 MICRO-MOVEMENT OPPORTUNITY DETECTED:")
            self.logger.info(
                f"   Price Change: {price_change:.4f} ({price_change * 100:.2f}%)"
            )
            self.logger.info(
                f"   Micro Movement: {is_micro_movement} (threshold: {micro_movement_threshold})"
            )
            self.logger.info(
                f"   Huge Candle: {is_huge_candle} (threshold: {huge_candle_threshold})"
            )
            self.logger.info(
                f"   Near S/R Zone: {is_near_sr} (distance: {nearest_sr_distance:.4f})"
            )
            if nearest_sr_level:
                self.logger.info(
                    f"   Nearest S/R: {nearest_sr_level.get('type', 'Unknown')} at {nearest_sr_level.get('level_price', 0):.2f}"
                )
            self.logger.info(f"   Opportunity Type: {opportunity_type}")

        return {
            "micro_movement": is_micro_movement,
            "huge_candle": is_huge_candle,
            "near_sr_zone": is_near_sr,
            "movement_size": price_change,
            "nearest_sr_distance": nearest_sr_distance,
            "nearest_sr_level": nearest_sr_level,
            "opportunity_type": opportunity_type,
            "current_price": current_price,
            "prev_price": prev_price,
        }

    def _determine_opportunity_type(
        self, micro_movement: bool, huge_candle: bool, near_sr: bool
    ) -> str:
        """Determine the type of trading opportunity based on market conditions."""
        if huge_candle and near_sr:
            return "SR_BREAKOUT"
        elif micro_movement and near_sr:
            return "SR_FADE"
        elif huge_candle:
            return "MOMENTUM"
        elif micro_movement:
            return "MICRO_MOVEMENT"
        else:
            return "STANDARD"

    def _prepare_order_parameters(
        self, signal: str, technical_analysis_data: Dict, current_atr: float
    ) -> Optional[Dict]:
        """Prepare order parameters based on signal type."""
        if signal == "BUY":
            return self._prepare_buy_order(technical_analysis_data, current_atr)
        elif signal == "SELL":
            return self._prepare_sell_order(technical_analysis_data, current_atr)
        elif signal == "SR_FADE_LONG":
            return self._prepare_sr_fade_long_order(
                technical_analysis_data, current_atr
            )
        elif signal == "SR_FADE_SHORT":
            return self._prepare_sr_fade_short_order(
                technical_analysis_data, current_atr
            )
        elif signal == "SR_BREAKOUT_LONG":
            return self._prepare_sr_breakout_long_order(
                technical_analysis_data, current_atr
            )
        elif signal == "SR_BREAKOUT_SHORT":
            return self._prepare_sr_breakout_short_order(
                technical_analysis_data, current_atr
            )
        else:
            return None  # No action for HOLD or unknown signals

    def _prepare_buy_order(
        self, technical_analysis_data: Dict, current_atr: float
    ) -> Dict:
        """Prepare buy order parameters."""
        stop_loss = self._calculate_atr_stop_loss("buy", technical_analysis_data)
        take_profit = self._calculate_atr_take_profit(
            "buy", technical_analysis_data, stop_loss
        )

        return {
            "order_type": "MARKET",
            "side": "buy",
            "entry_price": None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _prepare_sell_order(
        self, technical_analysis_data: Dict, current_atr: float
    ) -> Dict:
        """Prepare sell order parameters."""
        stop_loss = self._calculate_atr_stop_loss("sell", technical_analysis_data)
        take_profit = self._calculate_atr_take_profit(
            "sell", technical_analysis_data, stop_loss
        )

        return {
            "order_type": "MARKET",
            "side": "sell",
            "entry_price": None,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _prepare_sr_fade_long_order(
        self, technical_analysis_data: Dict, current_atr: float
    ) -> Optional[Dict]:
        """Prepare SR fade long order parameters with ML-based dynamic targets."""
        entry_price = technical_analysis_data.get("low")
        if entry_price is None:
            self.logger.warning("Missing 'low' for SR_FADE_LONG entry.")
            return None

        # Get current features for ML prediction
        current_features = technical_analysis_data.get(
            "current_features", pd.DataFrame()
        )

        # Use ML-based dynamic targets with validation
        try:
            ml_targets = self.ml_target_predictor.predict_dynamic_targets(
                signal_type="SR_FADE_LONG",
                technical_analysis_data=technical_analysis_data,
                current_features=current_features,
                current_atr=current_atr,
            )

            # Validate and potentially correct the prediction
            validation_result = self.ml_target_validator.validate_prediction(
                prediction=ml_targets,
                signal_type="SR_FADE_LONG",
                current_atr=current_atr,
                market_data=technical_analysis_data,
            )

            validated_targets = validation_result["corrected_prediction"]
            stop_loss = validated_targets.get("stop_loss")
            take_profit = validated_targets.get("take_profit")
            confidence = validated_targets.get("prediction_confidence", 0.0)

            # Log validation results
            if validation_result["used_fallback"]:
                self.logger.warning(
                    f"SR_FADE_LONG used fallback: {', '.join(validation_result['validation_issues'][:2])}"
                )

            self.logger.info(
                f"SR_FADE_LONG ML targets: TP={validated_targets.get('tp_multiplier', 'N/A'):.2f}x ATR, "
                f"SL={validated_targets.get('sl_multiplier', 'N/A'):.2f}x ATR, Confidence={confidence:.2f}"
            )

        except Exception as e:
            self.logger.error(
                f"ML target prediction failed for SR_FADE_LONG: {e}. Using fallback."
            )
            # Fallback to original fixed multipliers
            sr_sl_multiplier = self.config.get("sr_sl_multiplier", 0.5)
            stop_loss = entry_price - (current_atr * sr_sl_multiplier)
            take_profit = entry_price + (
                current_atr * self.config.get("sr_tp_multiplier", 2.0)
            )

        return {
            "order_type": "LIMIT",
            "side": "buy",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _prepare_sr_fade_short_order(
        self, technical_analysis_data: Dict, current_atr: float
    ) -> Optional[Dict]:
        """Prepare SR fade short order parameters with ML-based dynamic targets."""
        entry_price = technical_analysis_data.get("high")
        if entry_price is None:
            self.logger.warning("Missing 'high' for SR_FADE_SHORT entry.")
            return None

        # Get current features for ML prediction
        current_features = technical_analysis_data.get(
            "current_features", pd.DataFrame()
        )

        # Use ML-based dynamic targets
        try:
            ml_targets = self.ml_target_predictor.predict_dynamic_targets(
                signal_type="SR_FADE_SHORT",
                technical_analysis_data=technical_analysis_data,
                current_features=current_features,
                current_atr=current_atr,
            )

            stop_loss = ml_targets.get("stop_loss")
            take_profit = ml_targets.get("take_profit")
            confidence = ml_targets.get("prediction_confidence", 0.0)

            self.logger.info(
                f"SR_FADE_SHORT ML targets: TP={ml_targets.get('tp_multiplier', 'N/A'):.2f}x ATR, "
                f"SL={ml_targets.get('sl_multiplier', 'N/A'):.2f}x ATR, Confidence={confidence:.2f}"
            )

        except Exception as e:
            self.logger.error(
                f"ML target prediction failed for SR_FADE_SHORT: {e}. Using fallback."
            )
            # Fallback to original fixed multipliers
            sr_sl_multiplier = self.config.get("sr_sl_multiplier", 0.5)
            stop_loss = entry_price + (current_atr * sr_sl_multiplier)
            take_profit = entry_price - (
                current_atr * self.config.get("sr_tp_multiplier", 2.0)
            )

        return {
            "order_type": "LIMIT",
            "side": "sell",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _prepare_sr_breakout_long_order(
        self, technical_analysis_data: Dict, current_atr: float
    ) -> Optional[Dict]:
        """Prepare SR breakout long order parameters with ML-based dynamic targets."""
        entry_price = technical_analysis_data.get("high")
        if entry_price is None:
            self.logger.warning("Missing 'high' for SR_BREAKOUT_LONG entry.")
            return None

        fallback_stop_loss = technical_analysis_data.get("low")
        if fallback_stop_loss is None:
            self.logger.warning("Missing 'low' for SR_BREAKOUT_LONG stop loss.")
            return None

        # Get current features for ML prediction
        current_features = technical_analysis_data.get(
            "current_features", pd.DataFrame()
        )

        # Use ML-based dynamic targets
        try:
            ml_targets = self.ml_target_predictor.predict_dynamic_targets(
                signal_type="SR_BREAKOUT_LONG",
                technical_analysis_data=technical_analysis_data,
                current_features=current_features,
                current_atr=current_atr,
            )

            stop_loss = ml_targets.get("stop_loss", fallback_stop_loss)
            take_profit = ml_targets.get("take_profit")
            confidence = ml_targets.get("prediction_confidence", 0.0)

            self.logger.info(
                f"SR_BREAKOUT_LONG ML targets: TP={ml_targets.get('tp_multiplier', 'N/A'):.2f}x ATR, "
                f"SL={ml_targets.get('sl_multiplier', 'N/A'):.2f}x ATR, Confidence={confidence:.2f}"
            )

        except Exception as e:
            self.logger.error(
                f"ML target prediction failed for SR_BREAKOUT_LONG: {e}. Using fallback."
            )
            # Fallback to original fixed multipliers
            stop_loss = fallback_stop_loss
            take_profit = entry_price + (
                current_atr * self.config.get("sr_tp_multiplier", 2.0)
            )

        return {
            "order_type": "STOP_MARKET",
            "side": "buy",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _prepare_sr_breakout_short_order(
        self, technical_analysis_data: Dict, current_atr: float
    ) -> Optional[Dict]:
        """Prepare SR breakout short order parameters with ML-based dynamic targets."""
        entry_price = technical_analysis_data.get("low")
        if entry_price is None:
            self.logger.warning("Missing 'low' for SR_BREAKOUT_SHORT entry.")
            return None

        fallback_stop_loss = technical_analysis_data.get("high")
        if fallback_stop_loss is None:
            self.logger.warning("Missing 'high' for SR_BREAKOUT_SHORT stop loss.")
            return None

        # Get current features for ML prediction
        current_features = technical_analysis_data.get(
            "current_features", pd.DataFrame()
        )

        # Use ML-based dynamic targets
        try:
            ml_targets = self.ml_target_predictor.predict_dynamic_targets(
                signal_type="SR_BREAKOUT_SHORT",
                technical_analysis_data=technical_analysis_data,
                current_features=current_features,
                current_atr=current_atr,
            )

            stop_loss = ml_targets.get("stop_loss", fallback_stop_loss)
            take_profit = ml_targets.get("take_profit")
            confidence = ml_targets.get("prediction_confidence", 0.0)

            self.logger.info(
                f"SR_BREAKOUT_SHORT ML targets: TP={ml_targets.get('tp_multiplier', 'N/A'):.2f}x ATR, "
                f"SL={ml_targets.get('sl_multiplier', 'N/A'):.2f}x ATR, Confidence={confidence:.2f}"
            )

        except Exception as e:
            self.logger.error(
                f"ML target prediction failed for SR_BREAKOUT_SHORT: {e}. Using fallback."
            )
            # Fallback to original fixed multipliers
            stop_loss = fallback_stop_loss
            take_profit = entry_price - (
                current_atr * self.config.get("sr_tp_multiplier", 2.0)
            )

        return {
            "order_type": "STOP_MARKET",
            "side": "sell",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _calculate_risk_parameters(
        self, current_price: float, stop_loss: float, analyst_intel: Dict
    ) -> Dict:
        """Calculate position sizing and risk management parameters."""
        leverage = self._determine_leverage(
            analyst_intel.get("liquidation_risk_score", 100),
            analyst_intel.get("max_allowable_leverage", 10),
        )
        position_size = self._calculate_position_size(
            current_price, stop_loss, leverage
        )

        return {"leverage": leverage, "position_size": position_size}

    def _calculate_enhanced_risk_parameters(
        self,
        current_price: float,
        stop_loss: float,
        analyst_intel: Dict,
        confidence: float,
        market_conditions: Dict[str, Any],
    ) -> Dict:
        """Calculate enhanced risk parameters with micro-movement and high-confidence adjustments."""
        # Get LSS from analyst intelligence
        lss = analyst_intel.get("liquidation_risk_score", 50)

        # Determine enhanced leverage based on LSS, confidence, and market conditions
        max_leverage_cap = self.config.get("max_leverage_cap", 100)
        leverage = self._determine_leverage(
            lss, max_leverage_cap, confidence, market_conditions
        )

        # Calculate enhanced position size with confidence and market condition multipliers
        position_size = self._calculate_position_size(
            current_price, stop_loss, leverage, confidence, market_conditions
        )

        # Calculate risk-reward ratio
        take_profit = analyst_intel.get("technical_signals", {}).get(
            "take_profit", current_price * 1.02
        )
        risk_reward_ratio = (
            abs(take_profit - current_price) / abs(stop_loss - current_price)
            if stop_loss != current_price
            else 0
        )

        return {
            "leverage": leverage,
            "position_size": position_size,
            "lss": lss,
            "confidence": confidence,
            "market_conditions": market_conditions,
            "risk_reward_ratio": risk_reward_ratio,
            "opportunity_type": market_conditions.get("opportunity_type", "STANDARD"),
        }

    def _build_trade_decision(
        self,
        signal: str,
        confidence: float,
        order_params: Dict,
        risk_params: Dict,
        current_price: float,
        current_atr: float,
    ) -> Dict:
        """Build the final trade decision structure."""
        return {
            "signal": signal,
            "confidence": confidence,
            "order_type": order_params["order_type"],
            "side": order_params["side"],
            "entry_price": order_params["entry_price"],
            "stop_loss": order_params["stop_loss"],
            "take_profit": order_params["take_profit"],
            "leverage": risk_params["leverage"],
            "position_size": risk_params["position_size"],
            "current_price": current_price,
            "current_atr": current_atr,
        }

    @handle_errors(
        exceptions=(KeyError, TypeError, AttributeError),
        default_return=None,
        context="execute_open_position",
    )
    async def execute_open_position(self, decision: Dict, analyst_intel: Dict):
        """Executes the logic to open a new position based on the prepared decision."""
        self.logger.info(
            f"Executing OPEN for {decision['side'].upper()} signal '{decision['signal']}': Qty={decision['quantity']:.3f}, Type={decision['order_type']}"
        )

        trade_id = str(uuid.uuid4())  # Generate unique trade ID
        entry_timestamp = time.time()  # Capture entry timestamp

        try:
            # Fixed: Ensure self.exchange is not None before calling create_order
            if self.exchange is None:
                self.logger.error(
                    "Exchange client is None. Cannot execute open position."
                )
                raise RuntimeError("Exchange client not initialized.")

            order_response = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side=decision["side"],
                type=decision["order_type"],
                quantity=decision["quantity"],
                price=decision["entry_price"],  # Used for LIMIT orders
                params={
                    "stopPrice": decision["entry_price"],  # Used for STOP_MARKET orders
                    "takeProfitPrice": decision["take_profit"],
                    "stopLossPrice": decision["stop_loss"],
                },
            )

            if order_response and order_response.get("status") == "failed":
                self.logger.error(
                    f"Order creation failed: {order_response.get('error')}"
                )
                raise Exception(f"Order creation failed: {order_response.get('error')}")

            self.logger.info(f"Entry order placed successfully: {order_response}")

            executed_qty = float(
                order_response.get("executedQty", decision["quantity"])
            )
            avg_entry_price = float(
                order_response.get(
                    "avgPrice",
                    decision["entry_price"] or analyst_intel["current_price"],
                )
            )

            entry_fees_usd = executed_qty * avg_entry_price * CONFIG["taker_fee"]

            entry_context = {
                "MarketRegimeAtEntry": analyst_intel.get("market_regime"),
                "TacticianSignal": decision["signal"],
                "EnsemblePredictionAtEntry": analyst_intel.get("ensemble_prediction"),
                "EnsembleConfidenceAtEntry": analyst_intel.get("ensemble_confidence"),
                "BaseModelPredictionsAtEntry": analyst_intel.get(
                    "base_model_predictions", {}
                ),
                "EnsembleWeightsAtEntry": analyst_intel.get("ensemble_weights", {}),
                "DirectionalConfidenceAtEntry": analyst_intel.get(
                    "directional_confidence_score"
                ),
                "MarketHealthScoreAtEntry": analyst_intel.get("market_health_score"),
                "LiquidationSafetyScoreAtEntry": analyst_intel.get(
                    "liquidation_risk_score"
                ),
                "TrendStrengthAtEntry": analyst_intel.get("trend_strength"),
                "ADXValueAtEntry": analyst_intel.get("adx"),
                "RSIValueAtEntry": analyst_intel.get("technical_signals", {}).get(
                    "rsi"
                ),  # Revert to technical_signals path
                "MACDHistogramValueAtEntry": analyst_intel.get("technical_signals", {})
                .get("macd", {})
                .get("histogram"),  # Revert
                "PriceVsVWAPRatioAtEntry": analyst_intel.get(
                    "technical_signals", {}
                ).get("price_to_vwap_ratio"),  # Revert
                "VolumeDeltaAtEntry": analyst_intel.get(
                    "volume_delta"
                ),  # This is now top-level in analyst_intel
                "GlobalRiskMultiplierAtEntry": self.state_manager.get_state(
                    "global_risk_multiplier"
                )
                if self.state_manager
                else None,  # Fixed: Check state_manager
                "AvailableAccountEquityAtEntry": self.state_manager.get_state(
                    "account_equity"
                )
                if self.state_manager
                else None,  # Fixed: Check state_manager
                "TradingEnvironment": settings.trading_environment,
                "IsTradingPausedAtEntry": self.state_manager.get_state(
                    "is_trading_paused"
                )
                if self.state_manager
                else None,  # Fixed: Check state_manager
                "KillSwitchActiveAtEntry": self.state_manager.is_kill_switch_active()
                if self.state_manager
                else None,  # Fixed: Check state_manager
                "ModelVersionID": self.state_manager.get_state(
                    "model_version_id", "champion"
                )
                if self.state_manager
                else None,  # Fixed: Check state_manager
            }

            # Fixed: Ensure self.state_manager is not None before setting state
            if self.state_manager:
                self.current_position = {
                    "direction": "LONG" if decision["side"] == "buy" else "SHORT",
                    "size": executed_qty,
                    "entry_price": avg_entry_price,
                    "stop_loss": decision["stop_loss"],
                    "take_profit": decision["take_profit"],
                    "leverage": decision["leverage"],
                    "entry_timestamp": entry_timestamp,
                    "trade_id": trade_id,
                    "entry_fees_usd": entry_fees_usd,
                    "entry_context": entry_context,
                }
                self.state_manager.set_state("current_position", self.current_position)
                self.logger.info(f"New position state saved: {self.current_position}")
            else:
                self.logger.error(
                    "State manager is None. Cannot save current position state."
                )

        except Exception as e:
            self.logger.error(
                f"Failed to execute trade entry or update state: {e}", exc_info=True
            )
            # Fixed: Ensure self.state_manager is not None before setting state
            if self.state_manager:
                self.current_position = self._get_default_position()
                self.state_manager.set_state("current_position", self.current_position)

    @handle_errors(
        exceptions=(KeyError, TypeError, AttributeError),
        default_return=None,
        context="check_exit_conditions",
    )
    def _check_exit_conditions(
        self, analyst_intel: Dict
    ) -> Union[Dict, None]:  # Fixed: Union syntax
        pos = self.current_position
        current_price = analyst_intel.get(
            "current_price"
        )  # Using top-level current_price from analyst_intel
        if current_price is None:
            self.logger.warning("Current price not available for exit condition check.")
            return None

        if pos.get("direction") == "LONG":
            if current_price >= pos.get("take_profit", float("inf")):
                return {
                    "reason": "Take Profit Hit",
                    "exit_price": pos.get("take_profit"),
                }
            if current_price <= pos.get("stop_loss", float("-inf")):
                return {"reason": "Stop Loss Hit", "exit_price": pos.get("stop_loss")}
        elif pos.get("direction") == "SHORT":
            if current_price <= pos.get("take_profit", float("-inf")):
                return {
                    "reason": "Take Profit Hit",
                    "exit_price": pos.get("take_profit"),
                }
            if current_price >= pos.get("stop_loss", float("inf")):
                return {"reason": "Stop Loss Hit", "exit_price": pos.get("stop_loss")}
        return None

    @handle_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=0.0,
        context="calculate_position_size",
    )
    def _calculate_position_size(
        self,
        current_price: float,
        stop_loss_price: float,
        leverage: int,
        confidence: float = 0.0,
        market_conditions: Dict[str, Any] = None,
    ) -> float:
        """Calculate position size with enhanced micro-movement handling."""
        # Fixed: Ensure self.state_manager is not None before calling get_state
        capital = (
            self.state_manager.get_state(
                "account_equity", settings.get("initial_equity", 10000)
            )
            if self.state_manager
            else settings.get("initial_equity", 10000)
        )
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 0.01)
        risk_multiplier = (
            self.state_manager.get_state("global_risk_multiplier", 1.0)
            if self.state_manager
            else 1.0
        )
        max_risk_usd = capital * risk_per_trade_pct * risk_multiplier

        # Handle potential division by zero if stop_loss_price is too close to current_price
        stop_loss_distance = abs(current_price - stop_loss_price)
        if stop_loss_distance == 0:
            self.logger.warning(
                "Stop loss distance is zero, cannot calculate position size. Returning 0."
            )
            return 0.0

        # Base position size calculation
        base_units = max_risk_usd / stop_loss_distance

        # Apply position size multipliers based on confidence and market conditions
        position_multiplier = 1.0

        if market_conditions:
            multipliers = self.config.get("position_size_multiplier", {})
            opportunity_type = market_conditions.get("opportunity_type", "STANDARD")

            # Enhanced micro-movement handling
            if opportunity_type in ["SR_FADE", "SR_BREAKOUT", "MICRO_MOVEMENT"]:
                # For micro-movements, we can be more aggressive with position sizing
                # since we're targeting small, precise moves
                micro_multiplier = multipliers.get("micro_movement", 1.5)
                position_multiplier *= micro_multiplier
                self.logger.info(
                    f"Micro-movement detected: Applying {micro_multiplier}x position multiplier"
                )

            # High confidence multiplier
            if confidence >= self.config.get("high_confidence_threshold", 0.85):
                high_conf_multiplier = multipliers.get("high_confidence", 1.5)
                position_multiplier *= high_conf_multiplier
                self.logger.info(
                    f"High confidence ({confidence:.2f}): Applying {high_conf_multiplier}x position multiplier"
                )

            # S/R zone multiplier
            if market_conditions.get("near_sr_zone", False):
                sr_multiplier = multipliers.get("sr_zone", 1.3)
                position_multiplier *= sr_multiplier
                self.logger.info(
                    f"Near S/R zone: Applying {sr_multiplier}x position multiplier"
                )

            # Huge candle multiplier
            if market_conditions.get("huge_candle", False):
                huge_candle_multiplier = multipliers.get("huge_candle", 1.4)
                position_multiplier *= huge_candle_multiplier
                self.logger.info(
                    f"Huge candle detected: Applying {huge_candle_multiplier}x position multiplier"
                )

            # Combined conditions multiplier (multiple conditions met)
            condition_count = sum(
                [
                    confidence >= self.config.get("high_confidence_threshold", 0.85),
                    market_conditions.get("near_sr_zone", False),
                    market_conditions.get("huge_candle", False),
                    opportunity_type in ["SR_FADE", "SR_BREAKOUT", "MICRO_MOVEMENT"],
                ]
            )

            if condition_count >= 2:
                combined_multiplier = multipliers.get("combined", 2.0)
                position_multiplier *= combined_multiplier
                self.logger.info(
                    f"Multiple conditions met ({condition_count}): Applying {combined_multiplier}x position multiplier"
                )

        final_units = base_units * position_multiplier

        # Log detailed position sizing information
        self.logger.info("Position size calculation:")
        self.logger.info(f"   Capital: ${capital:,.2f}")
        self.logger.info(f"   Max Risk: ${max_risk_usd:,.2f}")
        self.logger.info(f"   Stop Loss Distance: ${stop_loss_distance:.4f}")
        self.logger.info(f"   Base Units: {base_units:.3f}")
        self.logger.info(f"   Position Multiplier: {position_multiplier:.2f}")
        self.logger.info(f"   Final Units: {final_units:.3f}")

        return round(final_units, 3)

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=1,
        context="determine_leverage",
    )
    def _determine_leverage(
        self,
        lss: float,
        max_leverage_cap: int,
        confidence: float = 0.0,
        market_conditions: Dict[str, Any] = None,
    ) -> int:
        """Determine leverage with enhanced micro-movement handling."""
        # Base leverage calculation
        base_leverage = min(lss, max_leverage_cap)

        # Apply confidence-based adjustments
        if confidence >= self.config.get("high_confidence_threshold", 0.85):
            high_conf_boost = self.config.get("high_confidence_leverage_boost", 1.8)
            base_leverage = min(base_leverage * high_conf_boost, max_leverage_cap)
            self.logger.info(f"High confidence leverage boost: {high_conf_boost}x")

        # Apply market condition adjustments
        if market_conditions:
            opportunity_type = market_conditions.get("opportunity_type", "STANDARD")

            # Enhanced leverage for micro-movements
            if opportunity_type in ["SR_FADE", "SR_BREAKOUT", "MICRO_MOVEMENT"]:
                # For micro-movements, we can use higher leverage since we're targeting
                # small, precise moves with tight stops
                micro_leverage_boost = 2.0  # Higher leverage for micro-movements
                base_leverage = min(
                    base_leverage * micro_leverage_boost, max_leverage_cap
                )
                self.logger.info(
                    f"Micro-movement leverage boost: {micro_leverage_boost}x"
                )

            # S/R zone leverage boost
            if market_conditions.get("near_sr_zone", False):
                sr_boost = self.config.get("sr_zone_leverage_boost", 1.5)
                base_leverage = min(base_leverage * sr_boost, max_leverage_cap)
                self.logger.info(f"S/R zone leverage boost: {sr_boost}x")

            # Huge candle leverage boost
            if market_conditions.get("huge_candle", False):
                huge_candle_boost = self.config.get("huge_candle_leverage_boost", 2.0)
                base_leverage = min(base_leverage * huge_candle_boost, max_leverage_cap)
                self.logger.info(f"Huge candle leverage boost: {huge_candle_boost}x")

        # Ensure leverage doesn't exceed maximum cap
        final_leverage = min(int(base_leverage), max_leverage_cap)

        # Log leverage calculation details
        self.logger.info("Leverage calculation:")
        self.logger.info(f"   Base LSS: {lss}")
        self.logger.info(f"   Max Cap: {max_leverage_cap}")
        self.logger.info(f"   Confidence: {confidence:.2f}")
        if market_conditions:
            self.logger.info(
                f"   Opportunity Type: {market_conditions.get('opportunity_type', 'STANDARD')}"
            )
        self.logger.info(f"   Final Leverage: {final_leverage}x")

        return final_leverage

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="execute_close_position"
    )
    async def execute_close_position(
        self,
        reason: str,
        analyst_intel: Dict,
        exit_price_override: Optional[float] = None,
    ):
        """Executes the logic to close an open position and logs the trade."""

        pos_details = self.current_position
        if not pos_details.get("trade_id"):
            self.logger.error(
                "Attempted to close position but no trade_id found in current_position. Aborting log."
            )
            return  # Don't proceed with logging if essential data is missing

        trade_id = pos_details["trade_id"]
        entry_timestamp = pos_details["entry_timestamp"]
        entry_price = pos_details["entry_price"]
        quantity = pos_details["size"]
        direction = pos_details["direction"]
        entry_fees_usd = pos_details.get("entry_fees_usd", 0.0)

        exit_timestamp = time.time()

        exit_price = (
            exit_price_override
            if exit_price_override is not None
            else analyst_intel.get("current_price")
        )

        if exit_price is None:
            self.logger.error(
                f"Exit price not available for trade {trade_id}. Cannot close position or log accurately."
            )
            return  # Cannot proceed without exit price

        self.logger.warning(
            f"Executing CLOSE for {direction} position (ID: {trade_id}). Reason: {reason}. Exit Price: {exit_price:.2f}"
        )

        try:
            # Fixed: Ensure self.exchange is not None before calling create_order
            if self.exchange is None:
                self.logger.error(
                    "Exchange client is None. Cannot execute close position."
                )
                raise RuntimeError("Exchange client not initialized.")

            order_response = await self.exchange.create_order(
                symbol=self.trade_symbol,
                side="SELL" if direction == "LONG" else "BUY",
                type="MARKET",
                quantity=quantity,
            )

            if order_response and order_response.get("status") == "failed":
                self.logger.error(
                    f"Closing order failed: {order_response.get('error')}"
                )
                raise Exception(f"Closing order failed: {order_response.get('error')}")

            self.logger.info(f"Close order placed successfully: {order_response}")

            actual_exit_price = float(order_response.get("avgPrice", exit_price))
            exit_fees_usd = quantity * actual_exit_price * CONFIG["taker_fee"]

            net_pnl_usd = 0.0
            pnl_percentage = 0.0
            if direction == "LONG":
                net_pnl_usd = (
                    (actual_exit_price - entry_price) * quantity
                    - entry_fees_usd
                    - exit_fees_usd
                )
                pnl_percentage = (
                    (net_pnl_usd / (entry_price * quantity)) * 100
                    if (entry_price * quantity) != 0
                    else 0.0
                )
            elif direction == "SHORT":
                net_pnl_usd = (
                    (entry_price - actual_exit_price) * quantity
                    - entry_fees_usd
                    - exit_fees_usd
                )
                pnl_percentage = (
                    (net_pnl_usd / (entry_price * quantity)) * 100
                    if (entry_price * quantity) != 0
                    else 0.0
                )

            trade_duration_seconds = exit_timestamp - entry_timestamp

            trade_log = {
                "TradeID": trade_id,
                "Token": self.trade_symbol.replace(
                    settings.get("base_currency", "USDT"), ""
                ),
                "Exchange": "Binance",
                "Side": direction,
                "EntryTimestampUTC": datetime.datetime.fromtimestamp(
                    entry_timestamp
                ).isoformat(),
                "ExitTimestampUTC": datetime.datetime.fromtimestamp(
                    exit_timestamp
                ).isoformat(),
                "TradeDurationSeconds": trade_duration_seconds,
                "NetPnLUSD": net_pnl_usd,
                "PnLPercentage": pnl_percentage,
                "ExitReason": reason,
                "EntryPrice": entry_price,
                "ExitPrice": actual_exit_price,
                "QuantityBaseAsset": quantity,
                "NotionalSizeUSD": entry_price * quantity,
                "LeverageUsed": pos_details.get("leverage"),
                "IntendedStopLossPrice": pos_details.get("stop_loss"),
                "IntendedTakeProfitPrice": pos_details.get("take_profit"),
                "ActualStopLossPrice": pos_details.get("stop_loss")
                if reason == "Stop Loss Hit"
                else None,
                "ActualTakeProfitPrice": pos_details.get("take_profit")
                if reason == "Take Profit Hit"
                else None,
                "OrderTypeEntry": pos_details.get("entry_context", {}).get(
                    "OrderTypeEntry"
                ),
                "OrderTypeExit": "MARKET",
                "EntryFeesUSD": entry_fees_usd,
                "ExitFeesUSD": exit_fees_usd,
                "SlippageEntryPct": None,
                "SlippageExitPct": None,
                **pos_details.get("entry_context", {}),
            }

            # Fixed: Ensure self.performance_reporter is not None before calling record_detailed_trade_log
            if self.performance_reporter:
                await self.performance_reporter.record_detailed_trade_log(trade_log)
            else:
                self.logger.warning(
                    "Performance reporter is None. Cannot record detailed trade log."
                )

            # Fixed: Ensure self.state_manager is not None before setting state
            if self.state_manager:
                self.current_position = self._get_default_position()
                self.state_manager.set_state("current_position", self.current_position)
                self.logger.info("Position closed and state has been reset.")
            else:
                self.logger.error("State manager is None. Cannot reset position state.")

        except Exception as e:
            self.logger.error(
                f"Failed to execute close order or log trade: {e}", exc_info=True
            )
            # If close failed, don't reset position, allow supervisor to re-sync
            # or manual intervention.

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=0.0,
        context="calculate_atr_stop_loss",
    )
    def _calculate_atr_stop_loss(
        self, side: str, candle: Dict[str, Any], multiplier: float = 1.5
    ) -> float:  # Fixed: Type hint for candle
        atr = candle.get("ATR", 0)
        return (
            candle["current_price"] - (atr * multiplier)
            if side == "buy"
            else candle["current_price"] + (atr * multiplier)
        )

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=0.0,
        context="calculate_atr_take_profit",
    )
    def _calculate_atr_take_profit(
        self, side: str, candle: Dict[str, Any], stop_loss: float, rr_ratio: float = 2.0
    ) -> float:  # Fixed: Type hint for candle
        risk = abs(candle["current_price"] - stop_loss)
        return (
            candle["current_price"] + (risk * rr_ratio)
            if side == "buy"
            else candle["current_price"] - (risk * rr_ratio)
        )
