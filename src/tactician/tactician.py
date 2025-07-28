# src/tactician/tactician.py
import pandas as pd
import numpy as np
import os
import sys
import datetime
import json # For serializing dicts
import asyncio # For async Firestore operations

# Assume these are available in the same package or through sys.path
from config import CONFIG, SYMBOL # Use the main CONFIG dictionary and SYMBOL
from utils.logger import system_logger
from database.firestore_manager import FirestoreManager # New import

class Tactician:
    """
    The Tactician module, the "brain" of the system, deciding when, where, and how to place each order.
    It uses a sophisticated rule-based system, with parameters optimized by the Supervisor,
    to manage individual orders and implement dynamic laddering.
    """
    def __init__(self, config=CONFIG, firestore_manager: FirestoreManager = None):
        self.config = config.get("tactician", {})
        self.global_config = config # Store global config to access BEST_PARAMS etc.
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('Tactician') # Child logger for Tactician

        self.current_position = {
            "symbol": None,
            "direction": None, # "LONG" or "SHORT"
            "size": 0.0, # in units of asset
            "entry_price": 0.0,
            "unrealized_pnl": 0.0,
            "current_leverage": 0,
            "ladder_steps": 0, # How many additional orders have been placed
            "stop_loss": None,
            "take_profit": None,
            "liquidation_price": 0.0, # Added for consistency with live data
            "entry_confidence": 0.0, # Confidence at the time of initial entry
            "entry_lss": 0.0 # LSS at the time of initial entry
        }
        self.trade_id_counter = 0 # Simple counter for trade IDs

    def _calculate_position_size(self, capital: float, current_price: float, stop_loss_price: float, leverage: float):
        """
        Calculates position size based on risk per trade and stop loss distance.
        This is a critical risk management function.
        """
        if stop_loss_price is None or current_price == stop_loss_price:
            self.logger.warning("Cannot calculate position size: Stop loss is None or same as entry price.")
            return 0.0, 0.0 # units, notional_value

        risk_per_trade_pct = self.global_config["RISK_PER_TRADE_PCT"] # From main config
        
        max_risk_usd = capital * risk_per_trade_pct
        stop_loss_distance_per_unit = abs(current_price - stop_loss_price)

        if stop_loss_distance_per_unit == 0:
            self.logger.warning("Stop loss distance is zero, cannot calculate position size.")
            return 0.0, 0.0

        units = max_risk_usd / stop_loss_distance_per_unit
        notional_value = units * current_price
        
        required_margin = notional_value / leverage
        if required_margin > capital:
            # Adjust units down if required margin exceeds available capital
            units = (capital * leverage) / current_price
            notional_value = units * current_price
            self.logger.info(f"Adjusted position size due to capital limits. New units: {units:.4f}, Notional: ${notional_value:.2f}")

        return units, notional_value

    def _determine_leverage(self, lss: float, max_allowable_leverage_cap: int):
        """
        Determines leverage based on Liquidation Safety Score (LSS) and Strategist's cap.
        LSS is 0-100.
        """
        ladder_config = self.config.get("laddering", {})
        initial_leverage = ladder_config.get("initial_leverage", 25)
        
        # Scale leverage linearly from initial_leverage to max_allowable_leverage_cap
        # based on LSS. Assume LSS of 50 is base, 100 is max.
        # This mapping can be optimized by Supervisor.
        
        # Example: LSS 0-50 maps to initial_leverage. LSS 50-100 scales from initial to max.
        if lss <= 50:
            scaled_leverage = initial_leverage
        else:
            # Scale from initial_leverage to max_allowable_leverage_cap over LSS range 50-100
            scaled_leverage = initial_leverage + (lss - 50) / 50 * (max_allowable_leverage_cap - initial_leverage)

        determined_leverage = max(initial_leverage, int(scaled_leverage))
        determined_leverage = min(determined_leverage, max_allowable_leverage_cap)
        
        self.logger.info(f"Determined Leverage: LSS={lss:.2f}, Scaled={scaled_leverage:.2f}, Final={determined_leverage}x (Cap={max_allowable_leverage_cap}x)")
        return determined_leverage

    def _update_position(self, symbol, direction, size, entry_price, leverage, stop_loss, take_profit, entry_confidence, entry_lss):
        """Internal method to update the Tactician's current position state."""
        self.current_position = {
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "unrealized_pnl": 0.0, # Will be updated externally by real-time data
            "current_leverage": leverage,
            "ladder_steps": 0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "liquidation_price": 0.0, # Will be updated by pipeline from live data
            "entry_confidence": entry_confidence,
            "entry_lss": entry_lss
        }
        self.trade_id_counter += 1
        self.logger.info(f"Position opened/updated: {direction} {size:.4f} {symbol} at {entry_price:.2f} with {leverage}x leverage. Conf: {entry_confidence:.2f}, LSS: {entry_lss:.2f}")

    def _add_to_ladder(self, symbol, direction, current_price, additional_size, new_leverage, new_stop_loss, new_take_profit):
        """Internal method to add to an existing laddered position."""
        if self.current_position["symbol"] != symbol or self.current_position["direction"] != direction:
            self.logger.warning("Cannot add to ladder: Symbol or direction mismatch.")
            return

        total_notional_old = self.current_position["size"] * self.current_position["entry_price"]
        total_notional_new = additional_size * current_price
        new_total_size = self.current_position["size"] + additional_size
        
        # Calculate new average entry price
        if new_total_size > 0:
            new_avg_entry_price = (total_notional_old + total_notional_new) / new_total_size
        else:
            new_avg_entry_price = self.current_position["entry_price"] # Should not happen if additional_size > 0

        self.current_position["size"] = new_total_size
        self.current_position["entry_price"] = new_avg_entry_price
        self.current_position["current_leverage"] = new_leverage # Update to the new, higher leverage
        self.current_position["ladder_steps"] += 1
        self.current_position["stop_loss"] = new_stop_loss # Update SL/TP for the combined position
        self.current_position["take_profit"] = new_take_profit

        self.logger.info(f"Laddered up: Added {additional_size:.4f} {symbol} at {current_price:.2f}. "
              f"New total size: {self.current_position['size']:.4f}, Avg Entry: {self.current_position['entry_price']:.2f}, "
              f"Leverage: {self.current_position['current_leverage']}x, Ladder Steps: {self.current_position['ladder_steps']}")

    def _close_position(self, exit_price: float, exit_reason: str):
        """Internal method to close the current position."""
        if self.current_position["size"] == 0:
            self.logger.info("No open position to close.")
            return

        pnl_pct = (exit_price - self.current_position["entry_price"]) / self.current_position["entry_price"] \
                  if self.current_position["direction"] == "LONG" else \
                  (self.current_position["entry_price"] - exit_price) / self.current_position["entry_price"]
        
        simulated_pnl_usd = pnl_pct * (self.current_position["size"] * self.current_position["entry_price"])

        self.logger.info(f"Position closed: {self.current_position['direction']} {self.current_position['size']:.4f} {self.current_position['symbol']}. "
              f"Entry: {self.current_position['entry_price']:.2f}, Exit: {exit_price:.2f}. "
              f"P&L: {pnl_pct*100:.2f}% (${simulated_pnl_usd:.2f} approx). Reason: {exit_reason}")
        
        self.current_position = { # Reset position
            "symbol": None, "direction": None, "size": 0.0, "entry_price": 0.0,
            "unrealized_pnl": 0.0, "current_leverage": 0, "ladder_steps": 0,
            "stop_loss": None, "take_profit": None, "liquidation_price": 0.0,
            "entry_confidence": 0.0, "entry_lss": 0.0
        }

    async def _determine_action_rules(self, state: dict):
        """
        Determines the trading action based on sophisticated rules and current market intelligence.
        """
        self.logger.info("Tactician: Determining action based on rule-based engine...")
        
        # Extract relevant state variables
        current_equity = state.get("current_equity", self.global_config["INITIAL_EQUITY"]) # Use passed equity
        current_price = state.get("current_price", 0.0)
        current_atr = state.get("current_atr", 0.0)
        
        directional_prediction = state.get("directional_prediction", "HOLD")
        directional_confidence = state.get("directional_confidence_score", 0.0)
        lss = state.get("liquidation_safety_score", 0.0)
        market_regime = state.get("market_regime", "UNKNOWN")
        sr_interaction_signal = state.get("sr_interaction_signal")
        high_impact_candle_signal = state.get("high_impact_candle_signal", {}).get("is_high_impact", False)

        max_allowable_leverage_cap = state.get("Max_Allowable_Leverage_Cap", self.global_config["strategist"].get("max_leverage_cap_default", 100))
        trading_range = state.get("Trading_Range", {"low": 0.0, "high": float('inf')})
        positional_bias = state.get("Positional_Bias", "NEUTRAL")

        # Get parameters from BEST_PARAMS (optimized by Supervisor)
        trade_entry_threshold = self.global_config["BEST_PARAMS"]["trade_entry_threshold"]
        sl_atr_multiplier = self.global_config["BEST_PARAMS"]["sl_atr_multiplier"]
        take_profit_rr = self.global_config["BEST_PARAMS"]["take_profit_rr"]
        min_lss_for_ladder = self.config["laddering"].get("min_lss_for_ladder", 70)
        min_confidence_for_ladder = self.config["laddering"].get("min_confidence_for_ladder", 0.75)
        ladder_step_leverage_increase = self.config["laddering"].get("ladder_step_leverage_increase", 5)
        max_ladder_steps = self.config["laddering"].get("max_ladder_steps", 3)
        
        # Confidence thresholds for closing a position (e.g., if confidence drops significantly)
        confidence_reversal_threshold = self.global_config["confidence_wrong_direction_thresholds"][1] # e.g., 0.5% drop

        # --- Rule 1: Manage Existing Position (Exit Conditions) ---
        if self.current_position["size"] != 0:
            # Check for Take Profit or Stop Loss
            if self.current_position["stop_loss"] is not None and self.current_position["take_profit"] is not None:
                if self.current_position["direction"] == "LONG":
                    if current_price >= self.current_position["take_profit"]:
                        return {"action": "POSITION_CLOSED", "reason": "Take Profit Hit."}
                    elif current_price <= self.current_position["stop_loss"]:
                        return {"action": "POSITION_CLOSED", "reason": "Stop Loss Hit."}
                elif self.current_position["direction"] == "SHORT":
                    if current_price <= self.current_position["take_profit"]:
                        return {"action": "POSITION_CLOSED", "reason": "Take Profit Hit."}
                    elif current_price >= self.current_position["stop_loss"]:
                        return {"action": "POSITION_CLOSED", "reason": "Stop Loss Hit."}

            # Check for significant confidence reversal or LSS deterioration
            # Compare current confidence to entry confidence for reversal
            if self.current_position["direction"] == "LONG" and directional_prediction == "SELL" and directional_confidence > confidence_reversal_threshold:
                 return {"action": "POSITION_CLOSED", "reason": f"Directional reversal ({directional_prediction} signal)."}
            if self.current_position["direction"] == "SHORT" and directional_prediction == "BUY" and directional_confidence > confidence_reversal_threshold:
                 return {"action": "POSITION_CLOSED", "reason": f"Directional reversal ({directional_prediction} signal)."}
            
            # If LSS drops significantly below entry LSS or a critical threshold
            if lss < self.current_position["entry_lss"] * 0.8 and lss < 50: # LSS dropped 20% and is below 50
                return {"action": "POSITION_CLOSED", "reason": f"Liquidation Safety Score deteriorated (LSS: {lss:.2f})."}

            # Close if market regime becomes highly unfavorable and position is open
            if (self.current_position["direction"] == "LONG" and market_regime == "BEAR_TREND") or \
               (self.current_position["direction"] == "SHORT" and market_regime == "BULL_TREND"):
                # Only close if not already profitable or if confidence is low
                if (self.current_position["unrealized_pnl"] < 0) or (directional_confidence < 0.5):
                    return {"action": "POSITION_CLOSED", "reason": f"Market regime shift to {market_regime}."}
            
            # --- Dynamic Stop-Loss/Take-Profit Adjustment ---
            # If position is profitable, trail the stop loss
            is_profitable = (self.current_position["direction"] == "LONG" and current_price > self.current_position["entry_price"]) or \
                            (self.current_position["direction"] == "SHORT" and current_price < self.current_position["entry_price"])
            
            if is_profitable and current_atr > 0:
                new_stop_loss = None
                if self.current_position["direction"] == "LONG":
                    # New SL should be (current_price - ATR_multiplier * ATR) but not lower than current SL
                    potential_new_sl = current_price - (current_atr * sl_atr_multiplier)
                    if self.current_position["stop_loss"] is None or potential_new_sl > self.current_position["stop_loss"]:
                        new_stop_loss = potential_new_sl
                elif self.current_position["direction"] == "SHORT":
                    # New SL should be (current_price + ATR_multiplier * ATR) but not higher than current SL
                    potential_new_sl = current_price + (current_atr * sl_atr_multiplier)
                    if self.current_position["stop_loss"] is None or potential_new_sl < self.current_position["stop_loss"]:
                        new_stop_loss = potential_new_sl
                
                if new_stop_loss is not None:
                    self.logger.info(f"Trailing Stop Loss: Old SL {self.current_position['stop_loss']:.2f}, New SL {new_stop_loss:.2f}")
                    self.current_position["stop_loss"] = new_stop_loss
                    # No explicit action returned here, as it's an internal adjustment.
                    # In a live system, this would translate to an "AMEND_ORDER" or "MOVE_STOP_LOSS" action.
            
            # --- Rule 2: Laddering Up (Add to Position) ---
            # Conditions for laddering:
            # 1. Position is currently profitable (unrealized PnL > 0).
            # 2. Directional confidence has increased (or is very high) AND LSS has increased (or is very high).
            # 3. Max ladder steps not reached.
            # 4. Price is still within Strategist's trading range.
            
            # Re-check is_profitable as it might have been updated by trailing SL logic
            is_profitable = (self.current_position["direction"] == "LONG" and current_price > self.current_position["entry_price"]) or \
                            (self.current_position["direction"] == "SHORT" and current_price < self.current_position["entry_price"])
            
            # Check if confidence increased AND LSS increased OR both are very high
            confidence_increased = directional_confidence > self.current_position["entry_confidence"] + 0.1 # 10% increase
            lss_increased = lss > self.current_position["entry_lss"] + 10 # 10 point increase in LSS

            can_ladder = is_profitable and \
                         (confidence_increased or directional_confidence >= min_confidence_for_ladder) and \
                         (lss_increased or lss >= min_lss_for_ladder) and \
                         (self.current_position["ladder_steps"] < max_ladder_steps) and \
                         (trading_range["low"] < current_price < trading_range["high"])
            
            if can_ladder and current_atr > 0:
                new_leverage = min(self.current_position["current_leverage"] + ladder_step_leverage_increase, max_allowable_leverage_cap)
                
                # Calculate new SL/TP for the combined position based on current price
                if self.current_position["direction"] == "LONG":
                    new_stop_loss_ladder = current_price - (current_atr * sl_atr_multiplier)
                    new_take_profit_ladder = current_price + (current_atr * sl_atr_multiplier * take_profit_rr)
                else: # SHORT
                    new_stop_loss_ladder = current_price + (current_atr * sl_atr_multiplier)
                    new_take_profit_ladder = current_price - (current_atr * sl_atr_multiplier * take_profit_rr)

                additional_units, _ = self._calculate_position_size(current_equity, current_price, new_stop_loss_ladder, new_leverage)
                
                if additional_units > 0:
                    return {
                        "action": "ADD_TO_LADDER",
                        "symbol": SYMBOL,
                        "direction": self.current_position["direction"],
                        "order_type": "MARKET", # Laddering usually market orders
                        "quantity": additional_units,
                        "leverage": new_leverage,
                        "stop_loss": new_stop_loss_ladder,
                        "take_profit": new_take_profit_ladder,
                        "reason": f"Laddering up: Profit, Confidence increased ({directional_confidence:.2f}), LSS increased ({lss:.2f})."
                    }
            
            # If position is open but no action, then HOLD
            return {"action": "HOLD", "reason": "Position open, no laddering or exit conditions met."}

        # --- Rule 3: Initial Entry (No Open Position) ---
        
        # Check if within trading range
        if not (trading_range["low"] < current_price < trading_range["high"]):
            return {"action": "HOLD", "reason": "Price outside Strategist's trading range."}

        # Check positional bias
        if positional_bias != "NEUTRAL" and \
           ((positional_bias == "LONG" and directional_prediction == "SELL") or \
            (positional_bias == "SHORT" and directional_prediction == "BUY")):
            return {"action": "HOLD", "reason": f"Directional prediction ({directional_prediction}) contradicts Strategist's bias ({positional_bias})."}

        # Entry conditions:
        # 1. Strong directional signal (BUY/SELL)
        # 2. High directional confidence
        # 3. High Liquidation Safety Score (LSS)
        # 4. Not an SR_ZONE_ACTION (unless it's a confirmed breakout signal)
        # 5. Not a High-Impact Candle that signals reversal (unless it's a confirmed follow-through)
        
        can_enter = False
        reason = "No entry signal."

        if directional_prediction == "BUY" and directional_confidence >= trade_entry_threshold and lss >= self.config["risk_management"].get("min_lss_for_entry", 60):
            if market_regime == "SR_ZONE_ACTION":
                # Only enter SR_ZONE_ACTION if it's a "BREAKTHROUGH_UP" signal from ensemble
                if state.get("directional_prediction") == "BREAKTHROUGH_UP": # Assuming ensemble provides specific breakout signals
                    can_enter = True
                    reason = "Breakthrough UP from S/R zone."
                else:
                    reason = "SR_ZONE_ACTION, not a confirmed breakthrough."
            elif high_impact_candle_signal and state.get("directional_prediction") != "FOLLOW_THROUGH_UP": # Assuming ensemble provides follow-through signals
                reason = "High-impact candle detected, but not confirmed follow-through."
            elif market_regime == "BULL_TREND" or market_regime == "SIDEWAYS_RANGE":
                can_enter = True
                reason = f"Strong BUY signal in {market_regime}."
            else:
                reason = f"Unfavorable market regime ({market_regime}) for BUY."

        elif directional_prediction == "SELL" and directional_confidence >= trade_entry_threshold and lss >= self.config["risk_management"].get("min_lss_for_entry", 60):
            if market_regime == "SR_ZONE_ACTION":
                # Only enter SR_ZONE_ACTION if it's a "BREAKTHROUGH_DOWN" signal from ensemble
                if state.get("directional_prediction") == "BREAKTHROUGH_DOWN": # Assuming ensemble provides specific breakout signals
                    can_enter = True
                    reason = "Breakthrough DOWN from S/R zone."
                else:
                    reason = "SR_ZONE_ACTION, not a confirmed breakthrough."
            elif high_impact_candle_signal and state.get("directional_prediction") != "FOLLOW_THROUGH_DOWN": # Assuming ensemble provides follow-through signals
                reason = "High-impact candle detected, but not confirmed follow-through."
            elif market_regime == "BEAR_TREND" or market_regime == "SIDEWAYS_RANGE":
                can_enter = True
                reason = f"Strong SELL signal in {market_regime}."
            else:
                reason = f"Unfavorable market regime ({market_regime}) for SELL."

        if can_enter and current_atr > 0 and current_equity > 0:
            leverage = self._determine_leverage(lss, max_allowable_leverage_cap)
            
            if leverage > 0:
                # Calculate SL/TP
                if directional_prediction == "BUY":
                    stop_loss = current_price - (current_atr * sl_atr_multiplier)
                    take_profit = current_price + (current_atr * sl_atr_multiplier * take_profit_rr)
                else: # SELL
                    stop_loss = current_price + (current_atr * sl_atr_multiplier)
                    take_profit = current_price - (current_atr * sl_atr_multiplier * take_profit_rr)

                units, _ = self._calculate_position_size(current_equity, current_price, stop_loss, leverage)
                
                if units > 0:
                    return {
                        "action": "PLACE_ORDER",
                        "symbol": SYMBOL,
                        "direction": "LONG" if directional_prediction == "BUY" else "SHORT",
                        "order_type": "MARKET", # Initial entry usually market order
                        "quantity": units,
                        "leverage": leverage,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "reason": reason
                    }
                else:
                    return {"action": "HOLD", "reason": "Calculated position size is zero."}
            else:
                return {"action": "HOLD", "reason": "Determined leverage is zero."}
        
        return {"action": "HOLD", "reason": reason} # Default to HOLD if no conditions met

    async def process_intelligence(self, analyst_intelligence: dict, strategist_params: dict, current_market_data: dict):
        """
        Receives intelligence from the Analyst and parameters from the Strategist,
        then decides on a trading action using rule-based logic.
        :param analyst_intelligence: Dictionary of insights from the Analyst.
        :param strategist_params: Dictionary of macro parameters from the Strategist.
        :param current_market_data: Dictionary of real-time market data (e.g., current price, ATR, current_equity).
        """
        self.logger.info("\n--- Tactician: Processing Intelligence ---")

        # Combine inputs into a single state representation for the rule engine
        state = {
            "current_position_size": self.current_position["size"],
            "current_position_direction": self.current_position["direction"],
            "current_position_entry_price": self.current_position["entry_price"],
            "current_position_leverage": self.current_position["current_leverage"],
            "current_position_ladder_steps": self.current_position["ladder_steps"],
            "current_position_unrealized_pnl": self.current_position["unrealized_pnl"], # For profitability check
            "current_position_entry_confidence": self.current_position["entry_confidence"],
            "current_position_entry_lss": self.current_position["entry_lss"],
            **current_market_data, # Includes current_price, current_atr, current_equity
            **analyst_intelligence,
            **strategist_params
        }

        # Get action from the rule-based engine
        decision = await self._determine_action_rules(state)

        action = decision["action"]
        reason = decision.get("reason", "No specific reason.")

        self.logger.info(f"Tactician Decision: {action} - {reason}")

        # Execute the decided action
        if action == "PLACE_ORDER":
            symbol = decision["symbol"]
            direction = decision["direction"]
            order_type = decision["order_type"]
            quantity = decision["quantity"]
            leverage = decision["leverage"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]

            self.logger.info(f"Executing PLACE_ORDER: {direction} {quantity:.4f} {symbol} at current price ({current_market_data['current_price']:.2f}) with {leverage}x leverage. SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            # Update internal position state immediately upon "order placed" decision
            self._update_position(symbol, direction, quantity, current_market_data['current_price'], leverage, stop_loss, take_profit, state["directional_confidence_score"], state["liquidation_safety_score"])
            return {"action": "ORDER_PLACED", "details": decision}

        elif action == "ADD_TO_LADDER":
            symbol = decision["symbol"]
            direction = decision["direction"]
            quantity = decision["quantity"]
            leverage = decision["leverage"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]

            self.logger.info(f"Executing ADD_TO_LADDER: Add {quantity:.4f} {symbol} at current price ({current_market_data['current_price']:.2f}) with {leverage}x leverage. New SL: {stop_loss:.2f}, New TP: {take_profit:.2f}")
            # Update internal position state immediately upon "ladder updated" decision
            self._add_to_ladder(symbol, direction, current_market_data['current_price'], quantity, leverage, stop_loss, take_profit)
            return {"action": "LADDER_UPDATED", "details": decision}

        elif action == "POSITION_CLOSED":
            self.logger.info(f"Executing CLOSE_POSITION ({reason}) at {current_market_data['current_price']:.2f}")
            # _close_position will be called by the pipeline after confirming live close
            return {"action": "POSITION_CLOSED", "reason": reason, "current_price": current_market_data['current_price']}

        elif action == "CANCEL_ORDER":
            self.logger.info("Executing CANCEL_ORDER (Placeholder: No active orders to cancel in this demo).")
            return {"action": "ORDER_CANCELLED", "details": decision}

        elif action == "HOLD":
            self.logger.info("Executing HOLD: No action taken.")
            return {"action": "HOLD", "details": decision}
        
        return {"action": "UNKNOWN", "details": decision}
