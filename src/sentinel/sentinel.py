import asyncio
import sys
import os
import time
import datetime
import json
import pandas as pd
import logging

from typing import Dict, Any
from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings, CONFIG # Import CONFIG for alert thresholds
from src.utils.state_manager import StateManager
from src.emails.ares_mailer import AresMailer # Import the AresMailer class


class Sentinel:
    """
    The Sentinel module is responsible for two primary functions:
    1. Real-time Data Streaming: Manages WebSocket connections for live market data.
    2. System Monitoring: Continuously checks system health, API connectivity, 
       model performance, and trade activity to act as a fail-safe.
       Now provides more granular and context-aware alerts.
    """

    # Define Alert Types and Severity Levels
    ALERT_TYPE_API_ERROR = "API_ERROR"
    ALERT_TYPE_MODEL_DEVIATION = "MODEL_DEVIATION"
    ALERT_TYPE_TRADE_ANOMALY = "TRADE_ANOMALY"
    ALERT_TYPE_MARKET_HEALTH = "MARKET_HEALTH_DEGRADATION"
    ALERT_TYPE_LIQUIDATION_RISK = "LIQUIDATION_RISK_CRITICAL"
    ALERT_TYPE_SYSTEM_STATE = "SYSTEM_STATE_CHANGE"
    ALERT_TYPE_SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"

    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_INFO = "INFO"

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
        self.config = settings.get("sentinel", {}) # Sentinel specific config
        self.global_config = CONFIG # Global config for shared thresholds
        self.trade_symbol = settings.trade_symbol
        self.timeframe = settings.timeframe
        
        # Initialize AresMailer for sending alerts
        self.ares_mailer = AresMailer(config=self.global_config)

        self.consecutive_api_errors = 0
        self.consecutive_model_errors = 0
        self.max_consecutive_errors = self.config.get("max_consecutive_errors", 3)

        # Store previous states for deviation monitoring
        self.previous_analyst_intelligence = {}
        self.previous_strategist_params = {}
        self.previous_tactician_decision = {}
        self.previous_trading_paused_state = self.state_manager.get_state("is_trading_paused", False)
        self.previous_kill_switch_state = self.state_manager.is_kill_switch_active()

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
            current_tactician_decision = self.state_manager.get_state("tactician_decision", {}) # Assuming tactician decision is stored

            await self.monitor_model_output_deviation(
                current_analyst_intel,
                current_strategist_params,
                current_tactician_decision
            )
            
            await self._monitor_market_health()
            await self._monitor_liquidation_risk()
            await self._monitor_system_state_changes()

    async def _dispatch_alert(self, alert_type: str, severity: str, message: str, context_data: Dict[str, Any] = None):
        """
        Centralized method to dispatch alerts.
        Formats the alert and sends it via the AresMailer.
        """
        context_data = context_data or {}
        timestamp = datetime.datetime.utcnow().isoformat()
        
        subject = f"[ARES {severity}] {alert_type} - {self.trade_symbol}"
        body = (
            f"Timestamp (UTC): {timestamp}\n"
            f"Alert Type: {alert_type}\n"
            f"Severity: {severity}\n"
            f"Message: {message}\n"
            f"Context: {json.dumps(context_data, indent=2)}\n"
            f"--- End Ares Alert ---"
        )
        
        self.logger.log(logging.getLevelName(severity), f"Dispatching alert: {subject} | {message}")
        await self.ares_mailer.send_alert(subject, body)

    async def check_api_connectivity(self):
        """
        Checks API latency. WebSocket health is implicitly checked by their connection status.
        Triggers alerts for high latency or prolonged failures.
        """
        self.logger.debug("Checking API REST endpoint latency...")
        try:
            start_time = time.time()
            # Use a lightweight API call, e.g., get_position_risk with a dummy symbol or just ping
            # For simplicity, using get_position_risk for the trade_symbol
            await self.exchange.get_position_risk(self.trade_symbol) 
            latency_ms = (time.time() - start_time) * 1000
            self.logger.info(f"API is responsive. Latency: {latency_ms:.2f} ms.")

            api_latency_threshold_ms = self.config.get("api_latency_threshold_ms", 1000)
            if latency_ms > api_latency_threshold_ms:
                self.consecutive_api_errors += 1
                message = f"API latency ({latency_ms:.2f} ms) exceeds threshold ({api_latency_threshold_ms} ms)."
                context = {"latency_ms": latency_ms, "threshold_ms": api_latency_threshold_ms}
                
                if self.consecutive_api_errors >= self.max_consecutive_errors:
                    await self._dispatch_alert(self.ALERT_TYPE_API_ERROR, self.SEVERITY_CRITICAL, 
                                               f"Consecutive API latency failures: {message}", context)
                    await self._trigger_system_shutdown("Consecutive API connectivity failures.")
                else:
                    await self._dispatch_alert(self.ALERT_TYPE_API_ERROR, self.SEVERITY_WARNING, message, context)
            else:
                self.consecutive_api_errors = 0

        except Exception as e:
            self.logger.error(f"API connectivity check failed: {e}")
            self.consecutive_api_errors += 1
            message = f"API connectivity check failed with error: {e}"
            context = {"error": str(e)}
            if self.consecutive_api_errors >= self.max_consecutive_errors:
                await self._dispatch_alert(self.ALERT_TYPE_API_ERROR, self.SEVERITY_CRITICAL, 
                                           f"Consecutive API connectivity failures: {message}", context)
                await self._trigger_system_shutdown("Consecutive API connectivity failures.")
            else:
                await self._dispatch_alert(self.ALERT_TYPE_API_ERROR, self.SEVERITY_WARNING, message, context)

    async def monitor_model_output_deviation(self, current_analyst_intel: Dict, current_strategist_params: Dict, current_tactician_decision: Dict):
        """Monitors for unusual deviations in model outputs."""
        self.logger.debug("Monitoring model output deviation...")
        deviation_detected = False
        deviation_reason = []
        
        # Threshold for confidence deviation (e.g., 50% change)
        confidence_deviation_threshold = self.config.get("model_output_deviation_threshold", 0.5) 
        
        # Analyst Intelligence Deviation
        if self.previous_analyst_intelligence and current_analyst_intel:
            prev_conf = self.previous_analyst_intelligence.get('directional_confidence_score', 0.0)
            curr_conf = current_analyst_intel.get('directional_confidence_score', 0.0)
            if abs(curr_conf - prev_conf) > confidence_deviation_threshold:
                deviation_detected = True
                deviation_reason.append(f"Analyst confidence jumped from {prev_conf:.2f} to {curr_conf:.2f}.")
        self.previous_analyst_intelligence = current_analyst_intel # Update for next cycle

        # Strategist Parameters Deviation (e.g., sudden large change in leverage cap)
        if self.previous_strategist_params and current_strategist_params:
            prev_leverage = self.previous_strategist_params.get('max_allowable_leverage', 0)
            curr_leverage = current_strategist_params.get('max_allowable_leverage', 0)
            if abs(curr_leverage - prev_leverage) > self.config.get("leverage_change_threshold", 10): # e.g., >10x change
                deviation_detected = True
                deviation_reason.append(f"Strategist leverage cap changed from {prev_leverage}x to {curr_leverage}x.")
        self.previous_strategist_params = current_strategist_params

        # Tactician Decision Deviation (e.g., unexpected action or size) - More complex to define
        # For simplicity, let's just check if a decision was made when it shouldn't have been, or vice versa
        # This requires more sophisticated state tracking. For now, we'll focus on Analyst/Strategist.
        self.previous_tactician_decision = current_tactician_decision # Update for next cycle

        if deviation_detected:
            self.consecutive_model_errors += 1
            message = f"Model output deviation detected: {'; '.join(deviation_reason)}"
            context = {"analyst_intel": current_analyst_intel, "strategist_params": current_strategist_params}
            
            if self.consecutive_model_errors >= self.max_consecutive_errors:
                await self._dispatch_alert(self.ALERT_TYPE_MODEL_DEVIATION, self.SEVERITY_CRITICAL, 
                                           f"Consecutive model output deviations: {message}", context)
                await self._trigger_system_shutdown(f"Consecutive model output deviations: {'; '.join(deviation_reason)}")
            else:
                await self._dispatch_alert(self.ALERT_TYPE_MODEL_DEVIATION, self.SEVERITY_WARNING, message, context)
        else:
            self.consecutive_model_errors = 0

    async def monitor_unusual_trade_activity(self, order_data: Dict):
        """
        Monitors for unusually large trades or unexpected trade outcomes
        based on real-time order updates.
        """
        if not order_data:
            return
        
        # Check for filled orders
        if order_data.get('X') in ['FILLED', 'PARTIALLY_FILLED']:
            trade_size_notional = float(order_data.get('L', 0)) # 'L' is last filled quantity
            current_price = float(order_data.get('L', 0)) # Use last filled price 'L'
            # Assuming 'L' is last trade quantity and 'p' is price from order_data
            # Binance order updates have 'p' as last price, 'q' as quantity, 'L' as last quantity.
            # Let's assume 'p' is the executed price and 'q' is the executed quantity for simplicity here.
            executed_qty = float(order_data.get('q', 0))
            executed_price = float(order_data.get('p', 0))
            trade_size_notional = executed_qty * executed_price

            unusual_multiplier = self.config.get("unusual_trade_volume_multiplier", 10.0)
            # Get max_notional_trade_size from strategist_params in state_manager
            max_notional_allowed = self.state_manager.get_state("strategist_params", {}).get("max_notional_trade_size", 10000)

            if trade_size_notional > max_notional_allowed * unusual_multiplier:
                message = f"Unusually large trade detected: Notional value ${trade_size_notional:,.2f} is >{unusual_multiplier}x allowed max (${max_notional_allowed:,.2f})."
                context = {
                    "symbol": order_data.get('s'),
                    "order_id": order_data.get('i'),
                    "executed_qty": executed_qty,
                    "current_price": current_price,
                    "executed_price": executed_price,
                    "trade_notional": trade_size_notional,
                    "max_allowed_notional": max_notional_allowed
                }
                await self._dispatch_alert(self.ALERT_TYPE_TRADE_ANOMALY, self.SEVERITY_CRITICAL, message, context)
                await self._trigger_system_shutdown(message)
        
        # Add checks for unexpected order statuses (e.g., order rejected, unexpected cancellation)
        if order_data.get('X') == 'REJECTED':
            message = f"Order rejected: {order_data.get('r', 'No reason provided')} for order ID {order_data.get('i')}."
            context = {"order_data": order_data}
            await self._dispatch_alert(self.ALERT_TYPE_TRADE_ANOMALY, self.SEVERITY_CRITICAL, message, context)
            await self._trigger_system_shutdown(message)

    async def _monitor_market_health(self):
        """
        Monitors the overall market health score from the Analyst.
        Triggers an alert if the market health degrades significantly.
        """
        analyst_intelligence = self.state_manager.get_state("analyst_intelligence", {})
        market_health_score = analyst_intelligence.get("market_health_score")
        
        if market_health_score is None:
            self.logger.warning("Market health score not available from Analyst intelligence.")
            return

        market_health_threshold = self.config.get("market_health_alert_threshold", 40.0) # e.g., below 40 is unhealthy

        if market_health_score < market_health_threshold:
            message = f"Market health score ({market_health_score:.2f}) is below critical threshold ({market_health_threshold:.2f})."
            context = {"market_health_score": market_health_score, "threshold": market_health_threshold}
            await self._dispatch_alert(self.ALERT_TYPE_MARKET_HEALTH, self.SEVERITY_WARNING, message, context)
            # Could escalate to CRITICAL or shutdown if prolonged or very low
        else:
            self.logger.debug(f"Market health is good: {market_health_score:.2f}")

    async def _monitor_liquidation_risk(self):
        """
        Monitors the Liquidation Safety Score (LSS) from the Analyst.
        Triggers a critical alert if LSS falls below a dangerous level.
        """
        analyst_intelligence = self.state_manager.get_state("analyst_intelligence", {})
        liquidation_risk_score = analyst_intelligence.get("liquidation_risk_score")
        liquidation_risk_reasons = analyst_intelligence.get("liquidation_risk_reasons")
        
        if liquidation_risk_score is None:
            self.logger.warning("Liquidation risk score not available from Analyst intelligence.")
            return

        lss_critical_threshold = self.config.get("lss_critical_threshold", 20.0) # e.g., below 20 is critical

        if liquidation_risk_score < lss_critical_threshold:
            message = f"Liquidation Safety Score (LSS) is CRITICAL ({liquidation_risk_score:.2f}). Immediate risk of liquidation."
            context = {
                "lss": liquidation_risk_score, 
                "threshold": lss_critical_threshold,
                "reasons": liquidation_risk_reasons
            }
            await self._dispatch_alert(self.ALERT_TYPE_LIQUIDATION_RISK, self.SEVERITY_CRITICAL, message, context)
            await self._trigger_system_shutdown(f"Critical LSS: {liquidation_risk_score:.2f}")
        else:
            self.logger.debug(f"LSS is healthy: {liquidation_risk_score:.2f}")

    async def _monitor_system_state_changes(self):
        """
        Monitors for changes in core system states like trading pause or kill switch.
        """
        current_paused_state = self.state_manager.get_state("is_trading_paused", False)
        current_kill_switch_active = self.state_manager.is_kill_switch_active()
        
        # Check for trading pause state changes
        if current_paused_state != self.previous_trading_paused_state:
            message = f"Trading state changed to {'PAUSED' if current_paused_state else 'RESUMED'}."
            context = {"is_trading_paused": current_paused_state}
            await self._dispatch_alert(self.ALERT_TYPE_SYSTEM_STATE, self.SEVERITY_INFO, message, context)
            self.previous_trading_paused_state = current_paused_state

        # Check for kill switch state changes
        if current_kill_switch_active != self.previous_kill_switch_state:
            if current_kill_switch_active:
                reason = self.state_manager.get_kill_switch_reason()
                message = f"KILL SWITCH ACTIVATED! Reason: {reason}"
                context = {"reason": reason}
                await self._dispatch_alert(self.ALERT_TYPE_SYSTEM_STATE, self.SEVERITY_CRITICAL, message, context)
            else:
                message = "KILL SWITCH DEACTIVATED."
                await self._dispatch_alert(self.ALERT_TYPE_SYSTEM_STATE, self.SEVERITY_INFO, message, {})
            self.previous_kill_switch_state = current_kill_switch_active
        
        # If kill switch is active, ensure repeated critical alerts (optional, but good for persistence)
        if current_kill_switch_active:
            reason = self.state_manager.get_kill_switch_reason()
            message = f"KILL SWITCH REMAINS ACTIVE. Reason: {reason}"
            context = {"reason": reason}
            await self._dispatch_alert(self.ALERT_TYPE_SYSTEM_STATE, self.SEVERITY_CRITICAL, message, context)


    async def _trigger_system_shutdown(self, reason: str):
        """Initiates a system-wide shutdown."""
        self.logger.critical(f"SYSTEM SHUTDOWN TRIGGERED: {reason}")
        
        # Dispatch a final critical alert for shutdown
        await self._dispatch_alert(self.ALERT_TYPE_SYSTEM_SHUTDOWN, self.SEVERITY_CRITICAL, 
                                   f"Ares trading system is initiating a shutdown due to: {reason}", 
                                   {"reason": reason})

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel() # Request cancellation of all other tasks
        
        self.logger.info("Attempting to close all open positions and cancel orders...")
        try:
            # These functions would need to be implemented in the exchange client
            # and should be awaited.
            await self.exchange.close_all_positions(self.trade_symbol)
            await self.exchange.cancel_all_orders(self.trade_symbol)
            self.logger.info("Emergency shutdown: Positions closed, orders cancelled.")
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown procedure (closing positions/orders): {e}", exc_info=True)
        
        # Exit the process
        sys.exit(1)
