# src/tactician/tactician.py
import pandas as pd
import numpy as np
import os
import sys
import datetime
import json # For serializing dicts
import asyncio # For async Firestore operations

# Placeholder for RL agent library
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# from stable_baselines3.common.policies import ActorCriticPolicy

# Assume these are available in the same package or through sys.path
from config import CONFIG # Use the main CONFIG dictionary
from utils.logger import system_logger
from database.firestore_manager import FirestoreManager # New import

class Tactician:
    """
    The Tactician module, the "brain" of the system, deciding when, where, and how to place each order.
    It uses a Reinforcement Learning agent to manage individual orders and implement laddering.
    """
    def __init__(self, config=CONFIG, firestore_manager: FirestoreManager = None):
        self.config = config.get("tactician", {})
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('Tactician') # Child logger for Tactician

        self.rl_agent = None # Placeholder for the RL agent
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
            "liquidation_price": 0.0 # Added for consistency with live data
        }
        self.trade_id_counter = 0 # Simple counter for trade IDs

        self.model_storage_path = self.config.get("rl_agent", {}).get("model_path", "models/tactician_ppo_model.zip")
        os.makedirs(os.path.dirname(self.model_storage_path), exist_ok=True) # Ensure model storage path exists

        # Load or initialize RL agent (async call in pipeline setup)
        # self._initialize_rl_agent()

    async def _initialize_rl_agent(self):
        """
        Initializes or loads the Reinforcement Learning agent.
        This is a placeholder for a full RL implementation.
        """
        self.logger.info("Tactician: Initializing RL Agent (Placeholder)...")
        
        l1_reg = self.config["rl_agent"].get("l1_regularization_strength", 0.001)
        l2_reg = self.config["rl_agent"].get("l2_regularization_strength", 0.001)
        self.logger.info(f"RL Agent Regularization: L1={l1_reg}, L2={l2_reg}")

        # Simulate loading a trained agent or training it
        if os.path.exists(self.model_storage_path):
            self.logger.info(f"Tactician: Simulating loading RL Agent from {self.model_storage_path}")
            # self.rl_agent = PPO.load(self.model_storage_path)
            # Load metadata from Firestore
            if self.firestore_manager and self.firestore_manager.firestore_enabled:
                metadata = await self.firestore_manager.get_document(
                    CONFIG['firestore']['model_metadata_collection'],
                    doc_id="Tactician_RLAgent_latest", # Assuming a 'latest' doc for RL agent
                    is_public=True
                )
                if metadata:
                    self.logger.info(f"Loaded RL Agent metadata from Firestore: {metadata.get('version')}")
        else:
            self.logger.info("Tactician: No pre-trained RL Agent found. Simulating training and saving.")
            # Simulate training
            # self.rl_agent = PPO(...)
            # self.rl_agent.save(self.model_storage_path)
            # Save metadata after simulated training
            await self.save_model_metadata("Tactician_RLAgent", "v1.0", {"reward": 1000, "episodes": 100}, self.model_storage_path)

    async def save_model_metadata(self, model_name: str, version: str, performance_metrics: dict, file_path_reference: str):
        """
        Saves model metadata to Firestore and CSV.
        """
        metadata = {
            "model_name": model_name,
            "version": version,
            "training_date": datetime.datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "file_path_reference": file_path_reference,
            "config_snapshot": json.dumps(self.config) # Snapshot Tactician config
        }
        
        # Save to Firestore
        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            await self.firestore_manager.set_document(
                CONFIG['firestore']['model_metadata_collection'],
                doc_id=f"{model_name}_{version}",
                data=metadata,
                is_public=True
            )
            await self.firestore_manager.set_document( # Also update a 'latest' document
                CONFIG['firestore']['model_metadata_collection'],
                doc_id=f"{model_name}_latest",
                data=metadata,
                is_public=True
            )
            self.logger.info(f"Model metadata for {model_name} saved to Firestore.")

        # Export to CSV
        try:
            with open(CONFIG['supervisor']['model_metadata_csv'], 'a') as f:
                f.write(f"{metadata['model_name']},{metadata['version']},{metadata['training_date']},"
                        f"{json.dumps(metadata['performance_metrics'])},{metadata['file_path_reference']},"
                        f"{metadata['config_snapshot']}\n")
            self.logger.info(f"Model metadata for {model_name} exported to CSV.")
        except Exception as e:
            self.logger.error(f"Error exporting model metadata for {model_name} to CSV: {e}")


    def _calculate_position_size(self, capital: float, current_price: float, stop_loss_price: float, leverage: float):
        """
        Calculates position size based on risk per trade and stop loss distance.
        This is a critical risk management function.
        """
        if stop_loss_price is None or current_price == stop_loss_price:
            self.logger.warning("Cannot calculate position size: Stop loss is None or same as entry price.")
            return 0.0, 0.0 # units, notional_value

        risk_per_trade_pct = CONFIG["RISK_PER_TRADE_PCT"] # From main config
        
        max_risk_usd = capital * risk_per_trade_pct
        stop_loss_distance_per_unit = abs(current_price - stop_loss_price)

        if stop_loss_distance_per_unit == 0:
            self.logger.warning("Stop loss distance is zero, cannot calculate position size.")
            return 0.0, 0.0

        units = max_risk_usd / stop_loss_distance_per_unit
        notional_value = units * current_price
        
        required_margin = notional_value / leverage
        if required_margin > capital:
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
        
        scaled_leverage = (lss / 100.0) * max_allowable_leverage_cap

        determined_leverage = max(initial_leverage, int(scaled_leverage))
        determined_leverage = min(determined_leverage, max_allowable_leverage_cap)
        
        self.logger.info(f"Determined Leverage: LSS={lss:.2f}, Scaled={scaled_leverage:.2f}, Final={determined_leverage}x (Cap={max_allowable_leverage_cap}x)")
        return determined_leverage

    def _update_position(self, symbol, direction, size, entry_price, leverage, stop_loss, take_profit):
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
            "liquidation_price": 0.0 # Will be updated by pipeline from live data
        }
        self.trade_id_counter += 1
        self.logger.info(f"Position opened/updated: {direction} {size:.4f} {symbol} at {entry_price:.2f} with {leverage}x leverage.")

    def _add_to_ladder(self, symbol, direction, current_price, additional_size, new_leverage, new_stop_loss, new_take_profit):
        """Internal method to add to an existing laddered position."""
        if self.current_position["symbol"] != symbol or self.current_position["direction"] != direction:
            self.logger.warning("Cannot add to ladder: Symbol or direction mismatch.")
            return

        total_notional_old = self.current_position["size"] * self.current_position["entry_price"]
        total_notional_new = additional_size * current_price
        new_total_size = self.current_position["size"] + additional_size
        new_avg_entry_price = (total_notional_old + total_notional_new) / new_total_size

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
            "stop_loss": None, "take_profit": None, "liquidation_price": 0.0
        }

    async def _get_rl_action(self, state: dict):
        """
        Placeholder for getting an action from the RL agent.
        """
        self.logger.info("Tactician: Querying RL Agent for action (Placeholder)...")
        
        # For now, simulate a decision based on simplified rules
        directional_prediction = state.get("directional_prediction", "HOLD")
        directional_confidence = state.get("directional_confidence_score", 0.0)
        lss = state.get("liquidation_safety_score", 0.0)
        current_price = state.get("current_price", 0.0)
        current_atr = state.get("current_atr", 0.0)

        max_allowable_leverage_cap = state.get("Max_Allowable_Leverage_Cap", self.config["laddering"].get("max_leverage_cap", 100))
        
        min_lss_for_ladder = self.config["laddering"].get("min_lss_for_ladder", 70)
        min_confidence_for_ladder = self.config["laddering"].get("min_confidence_for_ladder", 0.75)
        ladder_step_leverage_increase = self.config["laddering"].get("ladder_step_leverage_increase", 5)
        max_ladder_steps = self.config["laddering"].get("max_ladder_steps", 3)

        simulated_capital = CONFIG["INITIAL_EQUITY"] * self.firestore_manager.firestore_enabled # Placeholder for actual allocated capital

        # If no position, look for initial entry
        if self.current_position["size"] == 0:
            if directional_prediction in ["BUY", "SELL"] and directional_confidence >= CONFIG["analyst"]["regime_predictive_ensembles"]["min_confluence_confidence"]:
                leverage = self._determine_leverage(lss, max_allowable_leverage_cap)
                if leverage > 0 and current_atr > 0:
                    sl_atr_multiplier = CONFIG["atr"]["stop_loss_multiplier"]
                    take_profit_rr = CONFIG["BEST_PARAMS"]["take_profit_rr"] # Use BEST_PARAMS for TP_RR
                    
                    if directional_prediction == "BUY":
                        stop_loss = current_price - (current_atr * sl_atr_multiplier)
                        take_profit = current_price + (current_atr * sl_atr_multiplier * take_profit_rr)
                    else: # SELL
                        stop_loss = current_price + (current_atr * sl_atr_multiplier)
                        take_profit = current_price - (current_atr * sl_atr_multiplier * take_profit_rr)

                    units, _ = self._calculate_position_size(simulated_capital, current_price, stop_loss, leverage)
                    
                    if units > 0:
                        return {
                            "action": "PLACE_ORDER",
                            "symbol": SYMBOL,
                            "direction": "LONG" if directional_prediction == "BUY" else "SHORT",
                            "order_type": "MARKET",
                            "quantity": units,
                            "leverage": leverage,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "reason": f"Initial entry based on {directional_prediction} signal (Conf: {directional_confidence:.2f}, LSS: {lss:.2f})."
                        }
        else: # Position is open, consider laddering or closing
            # Check for Take Profit or Stop Loss conditions
            if self.current_position["stop_loss"] is not None and self.current_position["take_profit"] is not None:
                if self.current_position["direction"] == "LONG":
                    if current_price >= self.current_position["take_profit"]:
                        return {"action": "TAKE_PROFIT", "reason": "Take profit hit."}
                    elif current_price <= self.current_position["stop_loss"]:
                        return {"action": "CLOSE_POSITION", "reason": "Stop loss hit."}
                elif self.current_position["direction"] == "SHORT":
                    if current_price <= self.current_position["take_profit"]:
                        return {"action": "TAKE_PROFIT", "reason": "Take profit hit."}
                    elif current_price >= self.current_position["stop_loss"]:
                        return {"action": "CLOSE_POSITION", "reason": "Stop loss hit."}

            # Laddering logic: If trade is profitable and confidence/LSS are high
            if (self.current_position["direction"] == "LONG" and current_price > self.current_position["entry_price"]) or \
               (self.current_position["direction"] == "SHORT" and current_price < self.current_position["entry_price"]):
                
                if lss >= min_lss_for_ladder and directional_confidence >= min_confidence_for_ladder and \
                   self.current_position["ladder_steps"] < max_ladder_steps:
                    
                    new_leverage = min(self.current_position["current_leverage"] + ladder_step_leverage_increase, max_allowable_leverage_cap)
                    
                    if current_atr > 0:
                        sl_atr_multiplier = CONFIG["atr"]["stop_loss_multiplier"]
                        take_profit_rr = CONFIG["BEST_PARAMS"]["take_profit_rr"]

                        if self.current_position["direction"] == "LONG":
                            new_stop_loss = current_price - (current_atr * sl_atr_multiplier)
                            new_take_profit = current_price + (current_atr * sl_atr_multiplier * take_profit_rr)
                        else: # SHORT
                            new_stop_loss = current_price + (current_atr * sl_atr_multiplier)
                            new_take_profit = current_price - (current_atr * sl_atr_multiplier * take_profit_rr)

                        additional_units, _ = self._calculate_position_size(simulated_capital, current_price, new_stop_loss, new_leverage)
                        
                        if additional_units > 0:
                            return {
                                "action": "ADD_TO_LADDER",
                                "symbol": SYMBOL,
                                "direction": self.current_position["direction"],
                                "quantity": additional_units,
                                "leverage": new_leverage,
                                "stop_loss": new_stop_loss,
                                "take_profit": new_take_profit,
                                "reason": f"Laddering up based on increased confidence (Conf: {directional_confidence:.2f}, LSS: {lss:.2f})."
                            }

        return {"action": "HOLD", "reason": "No actionable signal or conditions not met."}

    async def process_intelligence(self, analyst_intelligence: dict, strategist_params: dict, current_market_data: dict):
        """
        Receives intelligence from the Analyst and parameters from the Strategist,
        then decides on a trading action.
        :param analyst_intelligence: Dictionary of insights from the Analyst.
        :param strategist_params: Dictionary of macro parameters from the Strategist.
        :param current_market_data: Dictionary of real-time market data (e.g., current price, ATR).
        """
        self.logger.info("\n--- Tactician: Processing Intelligence ---")

        # Combine inputs into a state representation for the RL agent (or rule-based logic)
        state = {
            "current_position_size": self.current_position["size"],
            "current_position_direction": self.current_position["direction"],
            "current_position_entry_price": self.current_position["entry_price"],
            "current_position_leverage": self.current_position["current_leverage"],
            "current_position_ladder_steps": self.current_position["ladder_steps"],
            "current_price": current_market_data.get("current_price", 0.0),
            "current_atr": current_market_data.get("current_atr", 0.0),
            **analyst_intelligence,
            **strategist_params
        }

        # Get action from RL agent (or rule-based simulation)
        decision = await self._get_rl_action(state)

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
            return {"action": "ORDER_PLACED", "details": decision}

        elif action == "ADD_TO_LADDER":
            symbol = decision["symbol"]
            direction = decision["direction"]
            quantity = decision["quantity"]
            leverage = decision["leverage"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]

            self.logger.info(f"Executing ADD_TO_LADDER: Add {quantity:.4f} {symbol} at current price ({current_market_data['current_price']:.2f}) with {leverage}x leverage. New SL: {stop_loss:.2f}, New TP: {take_profit:.2f}")
            return {"action": "LADDER_UPDATED", "details": decision}

        elif action == "TAKE_PROFIT":
            self.logger.info(f"Executing TAKE_PROFIT at {current_market_data['current_price']:.2f}")
            return {"action": "POSITION_CLOSED", "reason": "Take Profit", "current_price": current_market_data['current_price']}
        
        elif action == "CLOSE_POSITION":
            self.logger.info(f"Executing CLOSE_POSITION (Stop Loss/Reversal) at {current_market_data['current_price']:.2f}")
            return {"action": "POSITION_CLOSED", "reason": "Stop Loss/Reversal", "current_price": current_market_data['current_price']}

        elif action == "CANCEL_ORDER":
            self.logger.info("Executing CANCEL_ORDER (Placeholder: No active orders to cancel in this demo).")
            return {"action": "ORDER_CANCELLED", "details": decision}

        elif action == "HOLD":
            self.logger.info("Executing HOLD: No action taken.")
            return {"action": "HOLD", "details": decision}
        
        return {"action": "UNKNOWN", "details": decision}

# Example Usage (Main execution block for demonstration) - Removed as it's now part of pipeline
