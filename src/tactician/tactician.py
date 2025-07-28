# src/tactician/tactician.py
import pandas as pd
import numpy as np
import os
import sys
# Placeholder for RL agent library
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# from stable_baselines3.common.policies import ActorCriticPolicy

# Assume these are available in the same package or through sys.path
from config import CONFIG # Use the main CONFIG dictionary

class Tactician:
    """
    The Tactician module, the "brain" of the system, deciding when, where, and how to place each order.
    It uses a Reinforcement Learning agent to manage individual orders and implement laddering.
    """
    def __init__(self, config=CONFIG):
        self.config = config.get("tactician", {})
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
            "take_profit": None
        }
        self.trade_id_counter = 0 # Simple counter for trade IDs

        # Load or initialize RL agent
        self._initialize_rl_agent()

    def _initialize_rl_agent(self):
        """
        Initializes or loads the Reinforcement Learning agent.
        This is a placeholder for a full RL implementation.
        """
        print("Tactician: Initializing RL Agent (Placeholder)...")
        # In a real implementation, you would define your custom environment,
        # define your policy network, and then initialize or load the RL agent.

        # Example of how an RL agent might be initialized (requires stable_baselines3)
        # from stable_baselines3 import PPO
        # from stable_baselines3.common.policies import ActorCriticPolicy
        # from stable_baselines3.common.callbacks import BaseCallback
        # from tensorflow.keras.regularizers import l1_l2 # If using Keras for policy network

        # Define a dummy observation space and action space for illustration
        # obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32)
        # action_space = spaces.Discrete(4) # e.g., 0: HOLD, 1: BUY, 2: SELL, 3: CLOSE

        # This is where L1/L2 regularization would be applied if using a custom policy network
        # For stable_baselines3, regularization might be applied via custom network architectures
        # or by extending the policy class.
        l1_reg = self.config["rl_agent"].get("l1_regularization_strength", 0.001)
        l2_reg = self.config["rl_agent"].get("l2_regularization_strength", 0.001)
        print(f"RL Agent Regularization: L1={l1_reg}, L2={l2_reg}")

        # self.rl_agent = PPO(
        #     ActorCriticPolicy, # Or a custom policy
        #     env, # Your custom trading environment
        #     verbose=0,
        #     policy_kwargs=dict(
        #         # Example of how regularization might be passed to a custom network
        #         # net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        #         # activation_fn=nn.ReLU,
        #         # features_extractor_class=YourCustomFeatureExtractor,
        #         # features_extractor_kwargs=dict(regularizer=l1_l2(l1=l1_reg, l2=l2_reg))
        #     )
        # )
        
        # Simulate loading a trained agent
        # if os.path.exists(self.config["rl_agent"]["model_path"]):
        #     self.rl_agent = PPO.load(self.config["rl_agent"]["model_path"])
        #     print(f"Tactician: Loaded RL Agent from {self.config['rl_agent']['model_path']}")
        # else:
        #     print("Tactician: No pre-trained RL Agent found. Will use rule-based logic for now.")

    def _calculate_position_size(self, capital: float, current_price: float, stop_loss_price: float, leverage: float):
        """
        Calculates position size based on risk per trade and stop loss distance.
        This is a critical risk management function.
        """
        if stop_loss_price is None or current_price == stop_loss_price:
            print("Cannot calculate position size: Stop loss is None or same as entry price.")
            return 0.0, 0.0 # units, notional_value

        risk_per_trade_pct = self.config["risk_management"]["risk_per_trade_pct"]
        
        # Calculate the absolute dollar risk
        max_risk_usd = capital * risk_per_trade_pct

        # Calculate the stop loss distance in USD per unit of asset
        stop_loss_distance_per_unit = abs(current_price - stop_loss_price)

        if stop_loss_distance_per_unit == 0:
            print("Stop loss distance is zero, cannot calculate position size.")
            return 0.0, 0.0

        # Calculate units based on risk
        units = max_risk_usd / stop_loss_distance_per_unit
        notional_value = units * current_price
        
        # Adjust units based on leverage and available capital (margin required)
        required_margin = notional_value / leverage
        if required_margin > capital:
            # If required margin exceeds available capital, scale down units
            units = (capital * leverage) / current_price
            notional_value = units * current_price
            print(f"Adjusted position size due to capital limits. New units: {units:.4f}, Notional: ${notional_value:.2f}")

        return units, notional_value

    def _determine_leverage(self, lss: float, max_allowable_leverage_cap: int):
        """
        Determines leverage based on Liquidation Safety Score (LSS) and Strategist's cap.
        LSS is 0-100.
        """
        ladder_config = self.config.get("laddering", {})
        initial_leverage = ladder_config.get("initial_leverage", 25)
        
        # Scale leverage based on LSS:
        # If LSS is 0, leverage is 0 (or minimum).
        # If LSS is 100, leverage is max_allowable_leverage_cap.
        # This is a linear scaling, can be made non-linear (e.g., exponential)
        scaled_leverage = (lss / 100.0) * max_allowable_leverage_cap

        # Ensure minimum initial leverage if LSS is high enough, and respect max cap
        determined_leverage = max(initial_leverage, int(scaled_leverage))
        determined_leverage = min(determined_leverage, max_allowable_leverage_cap)
        
        print(f"Determined Leverage: LSS={lss:.2f}, Scaled={scaled_leverage:.2f}, Final={determined_leverage}x (Cap={max_allowable_leverage_cap}x)")
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
            "take_profit": take_profit
        }
        self.trade_id_counter += 1
        print(f"Position opened/updated: {direction} {size:.4f} {symbol} at {entry_price:.2f} with {leverage}x leverage.")

    def _add_to_ladder(self, symbol, direction, current_price, additional_size, new_leverage, new_stop_loss, new_take_profit):
        """Internal method to add to an existing laddered position."""
        if self.current_position["symbol"] != symbol or self.current_position["direction"] != direction:
            print("Cannot add to ladder: Symbol or direction mismatch.")
            return

        # Calculate new average entry price
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

        print(f"Laddered up: Added {additional_size:.4f} {symbol} at {current_price:.2f}. "
              f"New total size: {self.current_position['size']:.4f}, Avg Entry: {self.current_position['entry_price']:.2f}, "
              f"Leverage: {self.current_position['current_leverage']}x, Ladder Steps: {self.current_position['ladder_steps']}")

    def _close_position(self, exit_price: float, exit_reason: str):
        """Internal method to close the current position."""
        if self.current_position["size"] == 0:
            print("No open position to close.")
            return

        pnl_pct = (exit_price - self.current_position["entry_price"]) / self.current_position["entry_price"] \
                  if self.current_position["direction"] == "LONG" else \
                  (self.current_position["entry_price"] - exit_price) / self.current_position["entry_price"]
        
        # Simulate P&L in USD (very rough, doesn't account for leverage fully)
        simulated_pnl_usd = pnl_pct * (self.current_position["size"] * self.current_position["entry_price"])

        print(f"Position closed: {self.current_position['direction']} {self.current_position['size']:.4f} {self.current_position['symbol']}. "
              f"Entry: {self.current_position['entry_price']:.2f}, Exit: {exit_price:.2f}. "
              f"P&L: {pnl_pct*100:.2f}% (${simulated_pnl_usd:.2f} approx). Reason: {exit_reason}")
        
        self.current_position = { # Reset position
            "symbol": None, "direction": None, "size": 0.0, "entry_price": 0.0,
            "unrealized_pnl": 0.0, "current_leverage": 0, "ladder_steps": 0,
            "stop_loss": None, "take_profit": None
        }

    def _get_rl_action(self, state: dict):
        """
        Placeholder for getting an action from the RL agent.
        In a real scenario, `state` would be converted into an observation space
        for the RL agent.
        """
        print("Tactician: Querying RL Agent for action (Placeholder)...")
        # Example:
        # obs = self._convert_state_to_observation(state)
        # action, _states = self.rl_agent.predict(obs, deterministic=True)
        # return self._convert_action_to_decision(action)

        # For now, simulate a decision based on simplified rules
        # This is where the laddering logic would be "learned" by the RL agent
        # but is hardcoded for this placeholder.

        # Decision based on Analyst intelligence and current position
        directional_prediction = state.get("directional_prediction", "HOLD")
        directional_confidence = state.get("directional_confidence_score", 0.0)
        lss = state.get("liquidation_safety_score", 0.0)
        current_price = state.get("current_price", 0.0)
        current_atr = state.get("current_atr", 0.0) # From features or market health

        # Strategist parameters (simulated for now, would come from Strategist module)
        max_allowable_leverage_cap = self.config["laddering"].get("max_leverage_cap", 100)
        
        # Laddering parameters
        min_lss_for_ladder = self.config["laddering"].get("min_lss_for_ladder", 70)
        min_confidence_for_ladder = self.config["laddering"].get("min_confidence_for_ladder", 0.75)
        ladder_step_leverage_increase = self.config["laddering"].get("ladder_step_leverage_increase", 5)
        max_ladder_steps = self.config["laddering"].get("max_ladder_steps", 3)

        # If no position, look for initial entry
        if self.current_position["size"] == 0:
            if directional_prediction in ["BUY", "SELL"] and directional_confidence >= self.config["regime_predictive_ensembles"]["min_confluence_confidence"]:
                leverage = self._determine_leverage(lss, max_allowable_leverage_cap)
                if leverage > 0:
                    # Calculate initial stop loss and take profit based on ATR and RR
                    sl_atr_multiplier = self.config["risk_management"].get("sl_atr_multiplier", 1.5) # From main config
                    take_profit_rr = self.config["risk_management"].get("take_profit_rr", 2.0) # From main config
                    
                    if current_atr == 0:
                        print("ATR is zero, cannot calculate SL/TP dynamically.")
                        return {"action": "HOLD", "reason": "ATR zero for SL/TP."}

                    if directional_prediction == "BUY":
                        stop_loss = current_price - (current_atr * sl_atr_multiplier)
                        take_profit = current_price + (current_atr * sl_atr_multiplier * take_profit_rr)
                    else: # SELL
                        stop_loss = current_price + (current_atr * sl_atr_multiplier)
                        take_profit = current_price - (current_atr * sl_atr_multiplier * take_profit_rr)

                    # Simulate capital from Strategist or global config
                    simulated_capital = CONFIG["INITIAL_EQUITY"] 
                    units, _ = self._calculate_position_size(simulated_capital, current_price, stop_loss, leverage)
                    
                    if units > 0:
                        return {
                            "action": "PLACE_ORDER",
                            "symbol": "ETHUSDT", # Hardcoded for now
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
            # (RL agent would learn this, but here it's rule-based)
            if (self.current_position["direction"] == "LONG" and current_price > self.current_position["entry_price"]) or \
               (self.current_position["direction"] == "SHORT" and current_price < self.current_position["entry_price"]):
                
                if lss >= min_lss_for_ladder and directional_confidence >= min_confidence_for_ladder and \
                   self.current_position["ladder_steps"] < max_ladder_steps:
                    
                    # Calculate new leverage for the laddered position
                    new_leverage = min(self.current_position["current_leverage"] + ladder_step_leverage_increase, max_allowable_leverage_cap)
                    
                    # Calculate additional size (e.g., same initial risk amount, but with higher leverage)
                    # This is a simplified example; a real RL agent would determine optimal size
                    simulated_capital = CONFIG["INITIAL_EQUITY"]
                    # Recalculate units for the additional leg, assuming same risk basis but new leverage
                    sl_atr_multiplier = self.config["risk_management"].get("sl_atr_multiplier", 1.5)
                    if current_atr == 0:
                        print("ATR is zero, cannot calculate SL/TP dynamically for laddering.")
                        return {"action": "HOLD", "reason": "ATR zero for laddering SL/TP."}

                    # Recalculate stop loss based on current price and ATR for the new combined position
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
                            "symbol": "ETHUSDT",
                            "direction": self.current_position["direction"],
                            "quantity": additional_units,
                            "leverage": new_leverage,
                            "stop_loss": new_stop_loss,
                            "take_profit": new_take_profit,
                            "reason": f"Laddering up based on increased confidence (Conf: {directional_confidence:.2f}, LSS: {lss:.2f})."
                        }

        return {"action": "HOLD", "reason": "No actionable signal or conditions not met."}

    def process_intelligence(self, analyst_intelligence: dict, strategist_params: dict, current_market_data: dict):
        """
        Receives intelligence from the Analyst and parameters from the Strategist,
        then decides on a trading action.
        :param analyst_intelligence: Dictionary of insights from the Analyst.
        :param strategist_params: Dictionary of macro parameters from the Strategist.
        :param current_market_data: Dictionary of real-time market data (e.g., current price, ATR).
        """
        print("\n--- Tactician: Processing Intelligence ---")

        # Combine inputs into a state representation for the RL agent (or rule-based logic)
        state = {
            "current_position_size": self.current_position["size"],
            "current_position_direction": self.current_position["direction"],
            "current_position_entry_price": self.current_position["entry_price"],
            "current_position_leverage": self.current_position["current_leverage"],
            "current_position_ladder_steps": self.current_position["ladder_steps"],
            "current_price": current_market_data.get("current_price", 0.0),
            "current_atr": current_market_data.get("current_atr", 0.0), # Assuming ATR is passed in market data
            **analyst_intelligence, # Merge all analyst insights
            **strategist_params # Merge all strategist parameters
        }

        # Get action from RL agent (or rule-based simulation)
        decision = self._get_rl_action(state)

        action = decision["action"]
        reason = decision.get("reason", "No specific reason.")

        print(f"Tactician Decision: {action} - {reason}")

        # Execute the decided action
        if action == "PLACE_ORDER":
            symbol = decision["symbol"]
            direction = decision["direction"]
            order_type = decision["order_type"]
            quantity = decision["quantity"]
            leverage = decision["leverage"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]

            print(f"Executing PLACE_ORDER: {direction} {quantity:.4f} {symbol} at current price ({current_market_data['current_price']:.2f}) with {leverage}x leverage. SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            # In a real system, this would call an exchange API
            self._update_position(symbol, direction, quantity, current_market_data['current_price'], leverage, stop_loss, take_profit)
            return {"action": "ORDER_PLACED", "details": decision}

        elif action == "ADD_TO_LADDER":
            symbol = decision["symbol"]
            direction = decision["direction"]
            quantity = decision["quantity"]
            leverage = decision["leverage"]
            stop_loss = decision["stop_loss"]
            take_profit = decision["take_profit"]

            print(f"Executing ADD_TO_LADDER: Add {quantity:.4f} {symbol} at current price ({current_market_data['current_price']:.2f}) with {leverage}x leverage. New SL: {stop_loss:.2f}, New TP: {take_profit:.2f}")
            self._add_to_ladder(symbol, direction, current_market_data['current_price'], quantity, leverage, stop_loss, take_profit)
            return {"action": "LADDER_UPDATED", "details": decision}

        elif action == "TAKE_PROFIT":
            print(f"Executing TAKE_PROFIT at {current_market_data['current_price']:.2f}")
            self._close_position(current_market_data['current_price'], "Take Profit")
            return {"action": "POSITION_CLOSED", "reason": "Take Profit"}
        
        elif action == "CLOSE_POSITION":
            print(f"Executing CLOSE_POSITION (Stop Loss/Reversal) at {current_market_data['current_price']:.2f}")
            self._close_position(current_market_data['current_price'], "Stop Loss/Reversal")
            return {"action": "POSITION_CLOSED", "reason": "Stop Loss/Reversal"}

        elif action == "CANCEL_ORDER":
            print("Executing CANCEL_ORDER (Placeholder: No active orders to cancel in this demo).")
            # Logic to identify and cancel specific open orders
            return {"action": "ORDER_CANCELLED", "details": decision}

        elif action == "HOLD":
            print("Executing HOLD: No action taken.")
            return {"action": "HOLD", "details": decision}
        
        return {"action": "UNKNOWN", "details": decision}


# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    print("Running Tactician Module Demonstration...")

    # For demonstration, we'll need a dummy Analyst and Strategist output
    # In a real system, these would be actual objects/modules.

    # Dummy Analyst Intelligence (simplified from src/analyst/analyst.py output)
    dummy_analyst_intelligence = {
        "market_regime": "BULL_TREND",
        "trend_strength_score": 0.85,
        "adx_value": 70.5,
        "directional_prediction": "BUY", # This is the key signal from ensemble
        "directional_confidence_score": 0.80, # High confidence
        "liquidation_safety_score": 85.0, # High LSS
        "market_health_score": 90.0,
        "sr_interaction_signal": None,
        "high_impact_candle_signal": {"is_high_impact": False, "reason": "No high-impact candle detected."}
    }

    # Dummy Strategist Parameters (simplified)
    dummy_strategist_params = {
        "trading_range_low": 1900.0,
        "trading_range_high": 2200.0,
        "max_allowable_leverage_cap": 75, # Strategist sets this cap
        "positional_bias": "LONG"
    }

    # Simulate current market data
    current_price_sim = 2050.0
    current_atr_sim = 5.0 # Example ATR value

    current_market_data_sim = {
        "current_price": current_price_sim,
        "current_atr": current_atr_sim # Passed for SL/TP calculation
    }

    tactician = Tactician()

    print("\n--- Scenario 1: Initial Entry Opportunity (Bull Trend) ---")
    # Simulate first call to Tactician when no position is open
    action_result_1 = tactician.process_intelligence(
        dummy_analyst_intelligence,
        dummy_strategist_params,
        current_market_data_sim
    )
    print(f"Scenario 1 Result: {action_result_1}")
    print(f"Current Position State: {tactician.current_position}")

    print("\n--- Scenario 2: Laddering Opportunity (Trade moves favorably) ---")
    # Simulate price moving favorably, and Analyst providing high confidence/LSS
    current_market_data_sim["current_price"] = 2060.0 # Price moved up for a long
    dummy_analyst_intelligence["directional_confidence_score"] = 0.90 # Even higher confidence
    dummy_analyst_intelligence["liquidation_safety_score"] = 95.0 # Even higher LSS

    action_result_2 = tactician.process_intelligence(
        dummy_analyst_intelligence,
        dummy_strategist_params,
        current_market_data_sim
    )
    print(f"Scenario 2 Result: {action_result_2}")
    print(f"Current Position State: {tactician.current_position}")

    print("\n--- Scenario 3: Another Laddering Opportunity (Max steps not reached) ---")
    current_market_data_sim["current_price"] = 2070.0 # Price moved up further
    dummy_analyst_intelligence["directional_confidence_score"] = 0.92
    dummy_analyst_intelligence["liquidation_safety_score"] = 98.0

    action_result_3 = tactician.process_intelligence(
        dummy_analyst_intelligence,
        dummy_strategist_params,
        current_market_data_sim
    )
    print(f"Scenario 3 Result: {action_result_3}")
    print(f"Current Position State: {tactician.current_position}")

    print("\n--- Scenario 4: Take Profit Hit ---")
    # Simulate price hitting the take profit level
    if tactician.current_position["take_profit"]:
        current_market_data_sim["current_price"] = tactician.current_position["take_profit"] + 0.1 # Just above TP
    else:
        current_market_data_sim["current_price"] = 2100.0 # Fallback if TP not set

    action_result_4 = tactician.process_intelligence(
        dummy_analyst_intelligence, # Analyst intelligence might still be bullish
        dummy_strategist_params,
        current_market_data_sim
    )
    print(f"Scenario 4 Result: {action_result_4}")
    print(f"Current Position State: {tactician.current_position}")

    print("\n--- Scenario 5: No Signal / Hold ---")
    # Simulate a scenario where no strong signal or laddering condition is met
    dummy_analyst_intelligence["directional_prediction"] = "HOLD"
    dummy_analyst_intelligence["directional_confidence_score"] = 0.50
    dummy_analyst_intelligence["liquidation_safety_score"] = 60.0 # Lower LSS

    current_market_data_sim["current_price"] = 2000.0 # Reset price

    action_result_5 = tactician.process_intelligence(
        dummy_analyst_intelligence,
        dummy_strategist_params,
        current_market_data_sim
    )
    print(f"Scenario 5 Result: {action_result_5}")
    print(f"Current Position State: {tactician.current_position}")

    print("\nTactician Module Demonstration Complete.")
