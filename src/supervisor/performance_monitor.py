import pandas as pd
import numpy as np
import json
import datetime
import logging

from src.config import CONFIG
from src.utils.logger import system_logger
from emails.ares_mailer import AresMailer # Assuming AresMailer is available in emails/ares_mailer.py

class PerformanceMonitor:
    """
    Monitors live trading performance against backtested expectations to detect model decay.
    Triggers alerts if performance degrades significantly.
    """
    def __init__(self, config=CONFIG, firestore_manager=None):
        """
        Initializes the PerformanceMonitor.

        Args:
            config (dict): The global configuration dictionary.
            firestore_manager: An instance of FirestoreManager for data persistence.
        """
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('PerformanceMonitor')
        self.ares_mailer = AresMailer(config=config) # Initialize AresMailer for sending alerts

        # Configuration for performance monitoring thresholds
        self.decay_threshold_profit_factor = self.config.get("decay_threshold_profit_factor", 0.8) # e.g., 20% drop
        self.decay_threshold_sharpe_ratio = self.config.get("decay_threshold_sharpe_ratio", 0.7) # e.g., 30% drop
        self.decay_threshold_max_drawdown_multiplier = self.config.get("decay_threshold_max_drawdown_multiplier", 1.5) # e.g., 50% increase
        self.min_trades_for_monitoring = self.config.get("min_trades_for_monitoring", 50) # Minimum trades before monitoring starts

        # Load backtested expectations (e.g., from a 'latest' document in Firestore or a local file)
        self.backtested_expectations = self._load_backtested_expectations()

    async def _load_backtested_expectations(self) -> dict:
        """
        Loads the expected performance metrics from the most recent backtest.
        This typically comes from the 'latest' optimized parameters document.

        Returns:
            dict: A dictionary of expected performance metrics, or an empty dict if not found.
        """
        self.logger.info("Loading backtested performance expectations...")
        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            try:
                latest_params_doc = await self.firestore_manager.get_document(
                    self.global_config['firestore']['optimized_params_collection'],
                    doc_id='latest',
                    is_public=True
                )
                if latest_params_doc and 'performance_metrics' in latest_params_doc:
                    self.logger.info("Backtested expectations loaded successfully from Firestore.")
                    return latest_params_doc['performance_metrics']
                else:
                    self.logger.warning("No 'latest' backtested performance metrics found in Firestore. Monitoring will be limited.")
                    return {}
            except Exception as e:
                self.logger.error(f"Error loading backtested expectations from Firestore: {e}", exc_info=True)
                return {}
        else:
            self.logger.warning("Firestore is not enabled or manager not provided. Cannot load backtested expectations.")
            # Fallback for local testing or if Firestore is not used
            # You might want to load from a local JSON file here if applicable
            return {}

    async def monitor_performance(self, live_metrics: dict):
        """
        Compares live trading results against backtested expectations and detects model decay.

        Args:
            live_metrics (dict): A dictionary of current live trading performance metrics.
                                 Expected keys: 'Final Equity', 'Sharpe Ratio', 'Max Drawdown (%)',
                                 'Profit Factor', 'Win Rate (%)', 'Total Trades'.
        """
        self.logger.info("\n--- Starting Performance Monitoring ---")
        self.logger.info(f"Live Metrics: {live_metrics}")

        if not self.backtested_expectations:
            self.logger.warning("Backtested expectations not loaded. Cannot perform detailed decay detection.")
            return

        # Check if enough trades have occurred for meaningful monitoring
        total_trades = live_metrics.get('Total Trades', 0)
        if total_trades < self.min_trades_for_monitoring:
            self.logger.info(f"Not enough trades ({total_trades}) for meaningful monitoring (min: {self.min_trades_for_monitoring}). Skipping decay detection.")
            return

        decay_detected = False
        alert_messages = ["Model Decay Detected! Immediate human intervention may be required.\n"]
        alert_messages.append("--- Performance Comparison ---")
        alert_messages.append(f"Backtested Expectations: {json.dumps(self.backtested_expectations, indent=2)}")
        alert_messages.append(f"Live Trading Metrics: {json.dumps(live_metrics, indent=2)}")
        alert_messages.append("\n--- Decay Details ---")

        # Compare Profit Factor
        expected_profit_factor = self.backtested_expectations.get('Profit Factor', 0)
        live_profit_factor = live_metrics.get('Profit Factor', 0)
        if expected_profit_factor > 0 and live_profit_factor < expected_profit_factor * self.decay_threshold_profit_factor:
            decay_detected = True
            alert_messages.append(f"  - Profit Factor degraded: Live ({live_profit_factor:.2f}) < Expected ({expected_profit_factor:.2f}) * {self.decay_threshold_profit_factor}")

        # Compare Sharpe Ratio
        expected_sharpe_ratio = self.backtested_expectations.get('Sharpe Ratio', 0)
        live_sharpe_ratio = live_metrics.get('Sharpe Ratio', 0)
        if expected_sharpe_ratio > 0 and live_sharpe_ratio < expected_sharpe_ratio * self.decay_threshold_sharpe_ratio:
            decay_detected = True
            alert_messages.append(f"  - Sharpe Ratio degraded: Live ({live_sharpe_ratio:.2f}) < Expected ({expected_sharpe_ratio:.2f}) * {self.decay_threshold_sharpe_ratio}")

        # Compare Max Drawdown (%) - higher drawdown is worse
        expected_max_drawdown = self.backtested_expectations.get('Max Drawdown (%)', 0)
        live_max_drawdown = live_metrics.get('Max Drawdown (%)', 0)
        # Only trigger if expected drawdown is not zero and live drawdown is significantly higher
        if expected_max_drawdown > 0 and live_max_drawdown > expected_max_drawdown * self.decay_threshold_max_drawdown_multiplier:
            decay_detected = True
            alert_messages.append(f"  - Max Drawdown increased: Live ({live_max_drawdown:.2f}%) > Expected ({expected_max_drawdown:.2f}%) * {self.decay_threshold_max_drawdown_multiplier}")
        elif expected_max_drawdown == 0 and live_max_drawdown > 0: # If expected was 0, any drawdown is a concern
              decay_detected = True
              alert_messages.append(f"  - Max Drawdown observed: Live ({live_max_drawdown:.2f}%), while expected was 0%.")

        if decay_detected:
            full_alert_message = "\n".join(alert_messages)
            self.logger.warning(full_alert_message)
            subject = "Ares Alert: Model Performance Decay Detected!"
            try:
                # Note: The original code had `send_email`, assuming it's `send_alert` from the mailer class
                await self.ares_mailer.send_alert(subject, full_alert_message)
                self.logger.info("Model decay alert email sent successfully.")
            except Exception as e:
                self.logger.error(f"Failed to send model decay alert email: {e}", exc_info=True)
        else:
            self.logger.info("No significant model decay detected. Performance is within expected bounds.")

