import pandas as pd
import numpy as np
import json
import datetime
import logging
import asyncio
from typing import Union, Dict, Any, Optional

from src.config import CONFIG
from src.utils.logger import system_logger
# Import both managers for type hinting, but use the one passed in __init__
from src.database.firestore_manager import FirestoreManager
from src.database.sqlite_manager import SQLiteManager
from emails.ares_mailer import AresMailer # Assuming AresMailer is available in emails/ares_mailer.py
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_file_operations,
    handle_network_operations,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies
)

class PerformanceMonitor:
    """
    Monitors live trading performance against backtested expectations to detect model decay.
    Triggers alerts if performance degrades significantly.
    """
    def __init__(self, config=CONFIG, db_manager: Union[FirestoreManager, SQLiteManager, None] = None): # Fixed: Accept generic db_manager
        """
        Initializes the PerformanceMonitor.

        Args:
            config (dict): The global configuration dictionary.
            db_manager: An instance of FirestoreManager or SQLiteManager for data persistence.
        """
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.db_manager = db_manager # Use the passed db_manager
        self.logger = system_logger.getChild('PerformanceMonitor')
        self.ares_mailer = AresMailer(config=config) # Initialize AresMailer for sending alerts

        # Configuration for performance monitoring thresholds
        self.decay_threshold_profit_factor = self.config.get("decay_threshold_profit_factor", 0.8) # e.g., 20% drop
        self.decay_threshold_sharpe_ratio = self.config.get("decay_threshold_sharpe_ratio", 0.7) # e.g., 30% drop
        self.decay_threshold_max_drawdown_multiplier = self.config.get("decay_threshold_max_drawdown_multiplier", 1.5) # e.g., 50% increase
        self.min_trades_for_monitoring = self.config.get("min_trades_for_monitoring", 50) # Minimum trades before monitoring starts

        # Load backtested expectations (e.g., from a 'latest' document in DB)
        self.backtested_expectations = {} # Initialize as empty dict
        # Call async method from sync __init__ using asyncio.run for initial load
        # This is generally not ideal, but acceptable for startup loading of config-like data.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g., in async context), schedule as a task
                loop.create_task(self._load_backtested_expectations_async())
            else:
                # If no loop running, run blocking
                loop.run_until_complete(self._load_backtested_expectations_async())
        except Exception as e:
            self.logger.error(f"Failed to load backtested expectations during init: {e}", exc_info=True)

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="load_backtested_expectations_async"
    )
    async def _load_backtested_expectations_async(self) -> dict:
        """Load backtested expectations from database."""
        try:
            if self.db_manager:
                # Try to load from database
                expectations = await self.db_manager.get_backtest_results()
                if expectations:
                    self.backtested_expectations = expectations
                    self.logger.info("Backtested expectations loaded from database")
                    return expectations
                else:
                    self.logger.warning("No backtested expectations found in database")
                    self.backtested_expectations = {}
                    return {}
            else:
                self.logger.warning("No database manager available for loading expectations")
                self.backtested_expectations = {}
                return {}
        except Exception as e:
            self.logger.error(f"Error loading backtested expectations: {e}")
            self.backtested_expectations = {}
            return {}

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return=None,
        context="monitor_performance"
    )
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
            alert_messages.append(f"❌ Profit Factor decay: Live {live_profit_factor:.2f} vs Expected {expected_profit_factor:.2f}")

        # Compare Sharpe Ratio
        expected_sharpe = self.backtested_expectations.get('Sharpe Ratio', 0)
        live_sharpe = live_metrics.get('Sharpe Ratio', 0)
        
        if expected_sharpe > 0 and live_sharpe < expected_sharpe * self.decay_threshold_sharpe_ratio:
            decay_detected = True
            alert_messages.append(f"❌ Sharpe Ratio decay: Live {live_sharpe:.2f} vs Expected {expected_sharpe:.2f}")

        # Compare Max Drawdown
        expected_drawdown = self.backtested_expectations.get('Max Drawdown (%)', 0)
        live_drawdown = live_metrics.get('Max Drawdown (%)', 0)
        
        if expected_drawdown > 0 and live_drawdown > expected_drawdown * self.decay_threshold_max_drawdown_multiplier:
            decay_detected = True
            alert_messages.append(f"❌ Max Drawdown increase: Live {live_drawdown:.2f}% vs Expected {expected_drawdown:.2f}%")

        # Compare Win Rate
        expected_win_rate = self.backtested_expectations.get('Win Rate (%)', 0)
        live_win_rate = live_metrics.get('Win Rate (%)', 0)
        
        if expected_win_rate > 0 and live_win_rate < expected_win_rate * 0.8:  # 20% drop threshold
            decay_detected = True
            alert_messages.append(f"❌ Win Rate decay: Live {live_win_rate:.2f}% vs Expected {expected_win_rate:.2f}%")

        if decay_detected:
            alert_messages.append("\n🚨 IMMEDIATE ACTION REQUIRED:")
            alert_messages.append("1. Review recent market conditions")
            alert_messages.append("2. Check for data quality issues")
            alert_messages.append("3. Consider model retraining")
            alert_messages.append("4. Adjust risk parameters if necessary")
            
            # Send alert
            try:
                await self.ares_mailer.send_alert(
                    subject="🚨 Model Decay Detected - Immediate Action Required",
                    message="\n".join(alert_messages),
                    priority="high"
                )
                self.logger.critical("Model decay alert sent via email")
            except Exception as e:
                self.logger.error(f"Failed to send model decay alert: {e}")
            
            # Log detailed decay information
            self.logger.critical("Model decay detected - detailed analysis:")
            for message in alert_messages:
                self.logger.critical(message)
        else:
            self.logger.info("✅ No model decay detected - performance within acceptable parameters")

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_performance_score"
    )
    def calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate a composite performance score based on multiple metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            float: Performance score between 0-100 (higher = better)
        """
        try:
            # Extract metrics with defaults
            profit_factor = metrics.get('Profit Factor', 1.0)
            sharpe_ratio = metrics.get('Sharpe Ratio', 0.0)
            win_rate = metrics.get('Win Rate (%)', 50.0)
            max_drawdown = metrics.get('Max Drawdown (%)', 0.0)
            
            # Calculate individual scores
            pf_score = min(100.0, profit_factor * 50)  # 2.0 PF = 100 score
            sharpe_score = min(100.0, max(0.0, sharpe_ratio * 20))  # 5.0 Sharpe = 100 score
            win_rate_score = win_rate  # Win rate is already 0-100
            drawdown_score = max(0.0, 100.0 - max_drawdown * 2)  # 50% drawdown = 0 score
            
            # Weighted average
            weights = {
                'profit_factor': 0.3,
                'sharpe_ratio': 0.3,
                'win_rate': 0.2,
                'drawdown': 0.2
            }
            
            performance_score = (
                pf_score * weights['profit_factor'] +
                sharpe_score * weights['sharpe_ratio'] +
                win_rate_score * weights['win_rate'] +
                drawdown_score * weights['drawdown']
            )
            
            return max(0.0, min(100.0, performance_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return 50.0

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="get_performance_summary"
    )
    def get_performance_summary(self, live_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary.
        
        Args:
            live_metrics: Current live trading metrics
            
        Returns:
            Dict: Performance summary with analysis
        """
        try:
            performance_score = self.calculate_performance_score(live_metrics)
            
            # Determine performance status
            if performance_score >= 80:
                status = "EXCELLENT"
            elif performance_score >= 60:
                status = "GOOD"
            elif performance_score >= 40:
                status = "FAIR"
            else:
                status = "POOR"
            
            # Calculate comparison with backtested expectations
            comparison = {}
            if self.backtested_expectations:
                for metric in ['Profit Factor', 'Sharpe Ratio', 'Win Rate (%)', 'Max Drawdown (%)']:
                    expected = self.backtested_expectations.get(metric, 0)
                    actual = live_metrics.get(metric, 0)
                    if expected > 0:
                        comparison[metric] = {
                            'expected': expected,
                            'actual': actual,
                            'ratio': actual / expected if expected > 0 else 0
                        }
            
            return {
                'performance_score': performance_score,
                'status': status,
                'live_metrics': live_metrics,
                'backtested_expectations': self.backtested_expectations,
                'comparison': comparison,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}

