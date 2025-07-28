import asyncio
import sys
import os
import time
import pandas as pd
from typing import Dict

from src.exchange.binance import exchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager
from src.emails.ares_mailer import send_email

class Sentinel:
    """
    The Sentinel module is responsible for two primary functions:
    1. Real-time Data Streaming: Manages WebSocket connections for live market data.
    2. System Monitoring: Continuously checks system health, API connectivity, 
       model performance, and trade activity to act as a fail-safe.
    """

    def __init__(self, exchange_client, state_manager: StateManager):
        """
        Initializes the Sentinel.

        Args:
            exchange_client: An instance of the BinanceExchange client.
            state_manager: An instance of the StateManager to get/set system state.
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Sentinel')
        self.config = settings.get("sentinel", {})
        self.trade_symbol = settings.trade_symbol
        self.timeframe = settings.timeframe
        self.alert_recipient_email = self.config.get("alert_recipient_email")
        self.consecutive_api_errors = 0
        self.consecutive_model_errors = 0
        self.max_consecutive_errors = self.config.get("max_consecutive_errors", 3)
        self.previous_analyst_intelligence = {}
        self.previous_strategist_params = {}
        self.previous_tactician_decision = {}

    async def start(self):
        """
        Starts all Sentinel operations: data streams and the monitoring loop.
        """
        self.logger.info(f"Sentinel starting all operations for {self.trade_symbol}...")
        data_stream_task = asyncio.create_task(self._start_data_streams())
        monitoring_loop_task = asyncio.create_task(self._run_monitoring_loop())
        await asyncio.gather(data_stream_task, monitoring_loop_task)

    async def _start_data_streams(self):
        """
        Starts and manages the WebSocket data streams from the exchange.
        """
        self.logger.info("Initiating WebSocket data streams...")
        kline_task = asyncio.create_task(self.exchange.start_kline_socket(self.trade_symbol, self.timeframe))
        depth_task = asyncio.create_task(self.exchange.start_depth_socket(self.trade_symbol))
        trade_task = asyncio.create_task(self.exchange.start_trade_socket(self.trade_symbol))
        user_data_task = asyncio.create_task(self.exchange.start_user_data_socket(self._handle_user_data))
        await asyncio.gather(kline_task, depth_task, trade_task, user_data_task)

    async def _handle_user_data(self, data: Dict):
        """
        Callback for handling user data messages (e.g., order updates, balance changes).
        """
        event_type = data.get('e')
        if event_type == 'ORDER_TRADE_UPDATE':
            order_data = data.get('o')
            self.logger.info(f"Received order update: {order_data}")
            await self.monitor_unusual_trade_activity(order_data)
        elif event_type == 'ACCOUNT_UPDATE':
            self.logger.info(f"Received account update: {data}")
        else:
            self.logger.debug(f"Received user data: {data}")

    async def _run_monitoring_loop(self):
        """
        Periodically runs all synchronous Sentinel checks.
        """
        check_interval = self.config.get("check_interval_seconds", 60)
        self.logger.info(f"Monitoring loop started. Checks will run every {check_interval} seconds.")
        while True:
            await asyncio.sleep(check_interval)
            self.logger.info("Running periodic Sentinel checks...")
            await self.check_api_connectivity()
            current_analyst_intel = self.state_manager.get_state("analyst_intelligence", {})
            current_strategist_params = self.state_manager.get_state("strategist_params", {})
            current_tactician_decision = self.state_manager.get_state("tactician_decision", {})
            await self.monitor_model_output_deviation(
                current_analyst_intel,
                current_strategist_params,
                current_tactician_decision
            )

    async def check_api_connectivity(self):
        """
        Checks API latency. WebSocket health is implicitly checked by their connection status.
        """
        self.logger.debug("Checking API REST endpoint latency...")
        try:
            start_time = time.time()
            await self.exchange.get_position_risk(self.trade_symbol)
            latency_ms = (time.time() - start_time) * 1000
            self.logger.info(f"API is responsive. Latency: {latency_ms:.2f} ms.")
            if latency_ms > self.config.get("api_latency_threshold_ms", 1000):
                self.logger.warning(f"API latency ({latency_ms:.2f} ms) exceeds threshold.")
                self.consecutive_api_errors += 1
            else:
                self.consecutive_api_errors = 0
        except Exception as e:
            self.logger.error(f"API connectivity check failed: {e}")
            self.consecutive_api_errors += 1
        if self.consecutive_api_errors >= self.max_consecutive_errors:
            await self._trigger_system_shutdown("Consecutive API connectivity failures.")

    async def monitor_model_output_deviation(self, current_analyst_intel, current_strategist_params, current_tactician_decision):
        """Monitors for unusual deviations in model outputs."""
        self.logger.debug("Monitoring model output deviation...")
        deviation_detected = False
        deviation_reason = []
        threshold = self.config.get("model_output_deviation_threshold", 0.5)
        if self.previous_analyst_intelligence and current_analyst_intel:
            prev_conf = self.previous_analyst_intelligence.get('directional_confidence_score', 0.0)
            curr_conf = current_analyst_intel.get('directional_confidence_score', 0.0)
            if abs(curr_conf - prev_conf) > threshold:
                deviation_detected = True
                deviation_reason.append(f"Analyst confidence jumped from {prev_conf:.2f} to {curr_conf:.2f}.")
        self.previous_analyst_intelligence = current_analyst_intel
        if deviation_detected:
            self.consecutive_model_errors += 1
            self.logger.warning(f"Model output deviation detected: {'; '.join(deviation_reason)}")
            if self.consecutive_model_errors >= self.max_consecutive_errors:
                await self._trigger_system_shutdown(f"Consecutive model output deviations: {'; '.join(deviation_reason)}")
        else:
            self.consecutive_model_errors = 0

    async def monitor_unusual_trade_activity(self, order_data: dict):
        """Monitors for unusually large trades based on real-time order updates."""
        if not order_data:
            return
        if order_data.get('X') in ['FILLED', 'PARTIALLY_FILLED']:
            trade_size_notional = float(order_data.get('p', 0)) * float(order_data.get('q', 0))
            unusual_multiplier = self.config.get("unusual_trade_volume_multiplier", 10.0)
            max_notional_allowed = self.state_manager.get_state("strategist_params", {}).get("max_notional_trade_size", 10000)
            if trade_size_notional > max_notional_allowed * unusual_multiplier:
                reason = f"Unusually large trade detected: Notional value ${trade_size_notional:,.2f} is >{unusual_multiplier}x allowed max (${max_notional_allowed:,.2f})."
                self.logger.critical(reason)
                await self._trigger_system_shutdown(reason)

    async def _send_alert(self, subject: str, body: str):
        """Sends an email alert."""
        self.logger.warning(f"Sending Alert: {subject} - {body}")
        if self.alert_recipient_email and settings.email_config.get('enabled', False):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, send_email, subject, body)
            except Exception as e:
                self.logger.error(f"Failed to send email alert: {e}")
        else:
            self.logger.warning("Email sending disabled or no recipient configured.")

    async def _trigger_system_shutdown(self, reason: str):
        """Initiates a system-wide shutdown."""
        self.logger.critical(f"SYSTEM SHUTDOWN TRIGGERED: {reason}")
        alert_subject = f"Ares Critical Alert: System Shutdown - {reason}"
        alert_body = f"The Ares trading system is initiating a shutdown due to: {reason}"
        await self._send_alert(alert_subject, alert_body)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        self.logger.info("Closing all open positions and cancelling orders...")
        try:
            # These functions would need to be implemented in the exchange client
            # await self.exchange.close_all_positions()
            # await self.exchange.cancel_all_orders(self.trade_symbol)
            pass
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown procedure: {e}")
        sys.exit(1)
