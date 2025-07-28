# src/sentinel/sentinel.py
import requests
import time
import os
import sys
# Import the main CONFIG dictionary
from config import CONFIG, PIPELINE_PID_FILE # PIPELINE_PID_FILE is still needed for os.remove
from utils.logger import system_logger
from emails.ares_mailer import send_email
from src.utils.state_manager import StateManager

class Sentinel:
    """
    The Sentinel module is responsible for continuous monitoring of the system's health,
    API connectivity, and detecting unusual behavior or anomalies in model outputs/trade activity.
    It acts as a fail-safe to prevent catastrophic errors.
    """
    def __init__(self, config=CONFIG):
        self.config = config.get("sentinel", {})
        self.global_config = config # Store global config to access other sections
        self.logger = system_logger.getChild('Sentinel') # Child logger for Sentinel

        self.consecutive_api_errors = 0
        self.consecutive_model_errors = 0
        self.max_consecutive_errors = self.config.get("max_consecutive_errors", 3)
        self.alert_recipient_email = self.config.get("alert_recipient_email")

        self.state_manager = StateManager()

        # Store previous states for deviation monitoring
        self.previous_analyst_intelligence = {}
        self.previous_strategist_params = {}
        self.previous_tactician_decision = {}

    def _send_alert(self, subject: str, body: str):
        """Sends an email alert using the Ares Mailer."""
        if self.alert_recipient_email and self.global_config['EMAIL_CONFIG'].get('enabled', False): # Use global config for email enabled
            send_email(subject, body)
        else:
            self.logger.warning("Alert recipient email not configured or email sending disabled. Skipping email alert.")

    def _trigger_system_shutdown(self, reason: str):
        """Initiates a system-wide shutdown."""
        self.logger.critical(f"SYSTEM SHUTDOWN TRIGGERED: {reason}")
        alert_subject = f"Ares Critical Alert: System Shutdown - {reason}"
        alert_body = f"The Ares trading system is initiating a shutdown due to: {reason}\n\nPlease investigate immediately."
        self._send_alert(alert_subject, alert_body)
        
        # In a real system, this would involve:
        # 1. Closing all open positions on the exchange.
        # 2. Cancelling all open orders.
        # 3. Gracefully shutting down all other modules (Analyst, Tactician, Strategist, Supervisor).
        # 4. Removing PID files.
        
        # For demonstration, we'll just exit the process.
        if os.path.exists(self.global_config['PIPELINE_PID_FILE']): # Access from CONFIG
            os.remove(self.global_config['PIPELINE_PID_FILE']) # Clean up PID file
        sys.exit(1) # Exit with an error code

    def check_api_connectivity(self, api_endpoint: str = "https://fapi.binance.com/fapi/v1/ping"):
        """
        Checks connectivity and latency to a given API endpoint.
        :param api_endpoint: The URL to ping for connectivity.
        :return: True if API is responsive within latency threshold, False otherwise.
        """
        self.logger.info(f"Checking API connectivity to {api_endpoint}...")
        try:
            start_time = time.time()
            response = requests.get(api_endpoint, timeout=5) # 5-second timeout
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.logger.info(f"API is reachable. Latency: {latency_ms:.2f} ms.")
                if latency_ms > self.config.get("api_latency_threshold_ms", 500):
                    self.logger.warning(f"API latency ({latency_ms:.2f} ms) exceeds threshold ({self.config['api_latency_threshold_ms']} ms).")
                    self.consecutive_api_errors += 1
                else:
                    self.consecutive_api_errors = 0 # Reset counter on success
                return True
            else:
                self.logger.error(f"API returned status code {response.status_code}.")
                self.consecutive_api_errors += 1
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API connectivity check failed: {e}")
            self.consecutive_api_errors += 1
        
        if self.consecutive_api_errors >= self.max_consecutive_errors:
            self._trigger_system_shutdown("Consecutive API connectivity failures.")
        return False

    def monitor_model_output_deviation(self, current_analyst_intelligence: dict, current_strategist_params: dict, current_tactician_decision: dict):
        """
        Monitors for unusual deviations in model outputs (e.g., confidence scores, parameters).
        This is a simplified check; real deviation detection would use statistical methods.
        """
        self.logger.info("Monitoring model output deviation...")
        deviation_detected = False
        deviation_reason = []
        threshold = self.config.get("model_output_deviation_threshold", 0.2) # 20% deviation

        # Analyst Intelligence Check (e.g., directional confidence score)
        if self.previous_analyst_intelligence:
            prev_conf = self.previous_analyst_intelligence.get('directional_confidence_score', 0.0)
            curr_conf = current_analyst_intelligence.get('directional_confidence_score', 0.0)
            # Assuming confidence is a proportion (0-1), so threshold is also a proportion
            if abs(curr_conf - prev_conf) > threshold: 
                deviation_detected = True
                deviation_reason.append(f"Analyst confidence score jumped from {prev_conf:.2f} to {curr_conf:.2f}.")
        self.previous_analyst_intelligence = current_analyst_intelligence

        # Strategist Parameters Check (e.g., max leverage cap)
        if self.previous_strategist_params:
            prev_leverage_cap = self.previous_strategist_params.get('Max_Allowable_Leverage_Cap', 0)
            curr_leverage_cap = current_strategist_params.get('Max_Allowable_Leverage_Cap', 0)
            if prev_leverage_cap > 0 and abs(curr_leverage_cap - prev_leverage_cap) / prev_leverage_cap > threshold:
                deviation_detected = True
                deviation_reason.append(f"Strategist leverage cap changed by >{threshold*100:.0f}% from {prev_leverage_cap} to {curr_leverage_cap}.")
        self.previous_strategist_params = current_strategist_params

        # Tactician Decision Check (e.g., sudden change in action without strong reason)
        if self.previous_tactician_decision and current_tactician_decision:
            prev_action = self.previous_tactician_decision.get('action')
            curr_action = current_tactician_decision.get('action')
            # Example: If previous was HOLD, and current is PLACE_ORDER with very low confidence, might be an anomaly
            # Access min_lss_for_entry from CONFIG['tactician']['risk_management']
            min_confidence_for_action = self.global_config['tactician']['risk_management'].get("min_lss_for_entry", 60) / 100.0 # Convert LSS to 0-1 range for confidence comparison
            
            if prev_action == "HOLD" and curr_action == "PLACE_ORDER" and \
               current_analyst_intelligence.get('directional_confidence_score', 0.0) < min_confidence_for_action:
                deviation_detected = True
                deviation_reason.append(f"Tactician placed order from HOLD with low confidence ({current_analyst_intelligence.get('directional_confidence_score',0.0):.2f}).")
        self.previous_tactician_decision = current_tactician_decision

        if deviation_detected:
            self.consecutive_model_errors += 1
            self.logger.warning(f"Model output deviation detected: {'; '.join(deviation_reason)}")
            if self.consecutive_model_errors >= self.max_consecutive_errors:
                self._trigger_system_shutdown(f"Consecutive model output deviations: {'; '.join(deviation_reason)}")
        else:
            self.consecutive_model_errors = 0 # Reset counter on no deviation

        return deviation_detected

    def monitor_unusual_trade_activity(self, latest_trade_data: dict, average_trade_size: float):
        """
        Monitors for unusually large or frequent trades (e.g., if the system goes rogue).
        :param latest_trade_data: Dictionary of the latest executed trade.
        :param average_trade_size: The expected average trade size (e.g., from historical data).
        """
        if not latest_trade_data or average_trade_size == 0:
            return False

        trade_size = latest_trade_data.get('position_size', 0) * latest_trade_data.get('entry_price', 0) # Notional value
        unusual_multiplier = self.config.get("unusual_trade_volume_multiplier", 5.0)

        if average_trade_size > 0 and trade_size > average_trade_size * unusual_multiplier: # Added check for average_trade_size > 0
            reason = f"Unusually large trade detected: Notional value ${trade_size:,.2f} is >{unusual_multiplier}x average (${average_trade_size:,.2f})."
            self.logger.critical(reason)
            self._trigger_system_shutdown(reason)
            return True
        return False

    def check_performance_thresholds(self, historical_pnl_data: pd.DataFrame):
        """
        Monitors key performance metrics (Sharpe Ratio, Drawdown) against thresholds.
        If thresholds are breached, it pauses trading.
        """
        if historical_pnl_data.empty or len(historical_pnl_data) < self.config.get("performance_lookback_days", 30):
            return # Not enough data to make a decision

        self.logger.info("Checking performance thresholds...")
        lookback_data = historical_pnl_data.tail(self.config.get("performance_lookback_days", 30))
        
        # Calculate Sharpe Ratio
        daily_returns = lookback_data['NetPnL'] / self.global_config['INITIAL_EQUITY']
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

        # Calculate Max Drawdown
        equity_curve = (1 + daily_returns).cumprod()
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = -drawdown.min() * 100

        # Check against thresholds from config
        min_sharpe_threshold = self.config.get("min_sharpe_ratio_threshold", 0.5)
        max_drawdown_threshold = self.config.get("max_drawdown_threshold_pct", 20.0)

        if sharpe_ratio < min_sharpe_threshold:
            reason = f"Performance issue: Sharpe Ratio ({sharpe_ratio:.2f}) is below threshold ({min_sharpe_threshold})."
            self.logger.critical(reason)
            self._send_alert("Ares Critical Alert: Trading Paused Due to Low Sharpe Ratio", reason)
            self.state_manager.pause_trading()

        if max_drawdown > max_drawdown_threshold:
            reason = f"Performance issue: Max Drawdown ({max_drawdown:.2f}%) has exceeded threshold ({max_drawdown_threshold}%)."
            self.logger.critical(reason)
            self._send_alert("Ares Critical Alert: Trading Paused Due to High Drawdown", reason)
            self.state_manager.pause_trading()

    def run_checks(self, historical_pnl_data: pd.DataFrame):
        """
        Runs all Sentinel checks.
        """
        self.logger.info("Running Sentinel checks...")
        
        self.check_api_connectivity()
        self.check_performance_thresholds(historical_pnl_data)

        self.logger.info("Sentinel checks complete.")


# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    print("Running Sentinel Module Demonstration...")

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Re-initialize logger to ensure it's fresh for demo if run multiple times
    # (In a real system, logger is typically set up once at startup)
    from importlib import reload
    import utils.logger
    reload(utils.logger)
    system_logger = utils.logger.system_logger
    system_logger.info("Starting Sentinel demo.")

    # Pass the main CONFIG dictionary to Sentinel
    sentinel = Sentinel(config=CONFIG)

    # Simulate inputs for Sentinel
    analyst_intel_1 = {
        "market_regime": "BULL_TREND",
        "directional_confidence_score": 0.80, # Assuming 0-1 range
        "liquidation_safety_score": 85.0
    }
    strategist_params_1 = {
        "Trading_Range": {"low": 1900.0, "high": 2200.0},
        "Max_Allowable_Leverage_Cap": 75,
        "Positional_Bias": "LONG"
    }
    tactician_decision_1 = {
        "action": "PLACE_ORDER",
        "symbol": "ETHUSDT",
        "direction": "LONG",
        "quantity": 0.5,
        "leverage": 50
    }
    latest_trade_data_1 = { # For unusual trade activity check
        "position_size": 0.5, # Changed from 'quantity' to 'position_size' for consistency
        "entry_price": 2000.0,
        "symbol": "ETHUSDT"
    }
    average_expected_trade_size = 0.1 * 2000 # Example: $200 notional

    print("\n--- Scenario 1: Normal Operation ---")
    sentinel.run_checks(analyst_intel_1, strategist_params_1, tactician_decision_1, latest_trade_data_1, average_expected_trade_size)
    time.sleep(1) # Simulate time passing

    print("\n--- Scenario 2: API Latency Warning ---")
    # Temporarily reduce threshold to trigger warning
    sentinel.config["api_latency_threshold_ms"] = 10
    sentinel.check_api_connectivity() # Will log a warning
    sentinel.config["api_latency_threshold_ms"] = 500 # Reset for next checks
    time.sleep(1)

    print("\n--- Scenario 3: Model Output Deviation (Confidence Jump) ---")
    analyst_intel_2 = analyst_intel_1.copy()
    analyst_intel_2["directional_confidence_score"] = 0.10 # Sudden drop in confidence
    sentinel.run_checks(analyst_intel_2, strategist_params_1, tactician_decision_1)
    time.sleep(1)

    print("\n--- Scenario 4: Consecutive Model Output Deviations (Triggering Shutdown) ---")
    # Simulate multiple consecutive deviations
    for i in range(sentinel.max_consecutive_errors):
        print(f"  Attempt {i+1}/{sentinel.max_consecutive_errors} with deviation...")
        analyst_intel_3 = analyst_intel_1.copy()
        analyst_intel_3["directional_confidence_score"] = 0.05 + (i * 0.01) # Keep it low
        # This call should trigger shutdown on the last iteration
        try:
            sentinel.run_checks(analyst_intel_3, strategist_params_1, tactician_decision_1)
        except SystemExit as e:
            print(f"Caught SystemExit: {e}. Sentinel triggered shutdown as expected.")
            break # Exit loop as system would have shut down
        time.sleep(0.5)

    print("\nSentinel Module Demonstration Complete.")
    print(f"Check 'logs/ares_system.log' for detailed logs.")
