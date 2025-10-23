"""
Auto Monitoring Launcher

This module provides automatic activation of the enhanced monitoring system
when the trading system is launched in any mode (BACKTEST, PAPER, LIVE).
"""

import os

from typing import Any, Dict, Optional
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.config.environment import get_environment_settings
from .trading_mode_monitoring_integration import TradingModeMonitoringIntegration
import logging
import time

class AutoMonitoringLauncher:
    """
    Automatically launches and manages the enhanced monitoring system
    based on the current trading mode and environment configuration.
    """

    def __init__(self):
        """Initialize auto monitoring launcher."""
        self.logger = system_logger.getChild('AutoMonitoringLauncher')
        self.monitoring_integration: Optional[TradingModeMonitoringIntegration] = None
        self.is_launched: bool = False
        self.trading_mode: str = 'PAPER'
        self.launch_time: Optional[datetime] = None

    @handles_errors(
        error_handlers={
            ValueError: (False, 'Invalid monitoring configuration'),
            AttributeError: (False, 'Missing required monitoring parameters'),
            KeyError: (False, 'Missing configuration keys')
        },
        default_return=False,
        context='auto monitoring launcher initialization'
    )
    async def launch(self) -> bool:
        """
        Launch the enhanced monitoring system automatically.

        Returns:
            bool: True if launch successful, False otherwise
        """
        try:
            self.logger.info('🚀 Launching Auto Enhanced Monitoring System...')

            # Get environment settings
            env_settings = get_environment_settings()

            # Get trading mode
            self.trading_mode = os.environ.get('TRADING_MODE', env_settings.trading_environment).upper()

            # Check if monitoring is enabled
            monitoring_config = env_settings.get_enhanced_monitoring_config()
            if not monitoring_config.get('enable_enhanced_monitoring', True):
                self.logger.info('⚠️ Enhanced monitoring is disabled in configuration')
                return True

            # Initialize monitoring integration
            self.monitoring_integration = TradingModeMonitoringIntegration()
            success = await self.monitoring_integration.initialize()

            if success:
                self.is_launched = True
                self.launch_time = datetime.now()

                self.logger.info('✅ Auto Enhanced Monitoring System launched successfully')
                self.logger.info(f'   📊 Trading Mode: {self.trading_mode}')
                self.logger.info(f'   🕐 Launch Time: {self.launch_time}')
                self.logger.info('   🔍 SHAP/LIME explanations active')
                self.logger.info('   📈 Ensemble and ML model tracking active')
                self.logger.info('   📋 Daily and monthly CSV exports configured')
                self.logger.info('   🎯 Automatic trade decision capture enabled')

                # Log configuration details
                self._log_monitoring_configuration(monitoring_config)

                return True
            else:
                self.logger.warning('⚠️ Failed to launch Enhanced Monitoring System')
                return False

        except Exception as e:
            self.logger.exception(f'❌ Auto Enhanced Monitoring System launch failed: {e}')
            return False

    def _log_monitoring_configuration(self, config: Dict[str, Any]) -> None:
        """Log monitoring configuration details."""
        try:
            self.logger.info('📋 Monitoring Configuration:')
            self.logger.info(f'   📁 Export Directory: {config.get("monitoring_export_directory", "monitoring_exports")}')
            self.logger.info(f'   📅 CSV Export Interval: {config.get("monitoring_csv_export_interval_days", 30)} days')
            self.logger.info(f'   💾 Max Decisions in Memory: {config.get("monitoring_max_decisions_in_memory", 10000)}')
            self.logger.info(f'   ⚡ Real-time Updates: {config.get("monitoring_enable_real_time_updates", True)}')
            self.logger.info(f'   🔍 SHAP Analysis: {config.get("monitoring_enable_shap", True)}')
            self.logger.info(f'   🔍 LIME Analysis: {config.get("monitoring_enable_lime", True)}')
        except Exception as e:
            self.logger.exception(f'Error logging monitoring configuration: {e}')

    @handles_errors(
        default_return=None,
        context='trade decision auto capture'
    )
    async def capture_trade_decision(
        self,
        trade_data: Dict[str, Any],
        trading_mode: Optional[str] = None
    ) -> None:
        """
        Automatically capture a trade decision.

        Args:
            trade_data: Trade decision data
            trading_mode: Override trading mode (optional)
        """
        try:
            if not self.is_launched or not self.monitoring_integration:
                self.logger.warning('⚠️ Auto monitoring not launched, skipping trade capture')
                return

            await self.monitoring_integration.record_trade_decision(trade_data, trading_mode)

        except Exception as e:
            self.logger.exception(f'Error capturing trade decision: {e}')

    @handles_errors(
        default_return=None,
        context='performance auto update'
    )
    async def update_performance(
        self,
        performance_data: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> None:
        """
        Automatically update performance metrics.

        Args:
            performance_data: Performance metrics data
            model_id: Optional model identifier
        """
        try:
            if not self.is_launched or not self.monitoring_integration:
                return

            await self.monitoring_integration.update_performance_metrics(performance_data, model_id)

        except Exception as e:
            self.logger.exception(f'Error updating performance: {e}')

    @handles_errors(
        default_return=None,
        context='ensemble auto update'
    )
    async def update_ensemble(
        self,
        ensemble_data: Dict[str, Any],
        ensemble_id: Optional[str] = None
    ) -> None:
        """
        Automatically update ensemble performance.

        Args:
            ensemble_data: Ensemble performance data
            ensemble_id: Optional ensemble identifier
        """
        try:
            if not self.is_launched or not self.monitoring_integration:
                return

            await self.monitoring_integration.update_ensemble_performance(ensemble_data, ensemble_id)

        except Exception as e:
            self.logger.exception(f'Error updating ensemble: {e}')

    def is_monitoring_active(self) -> bool:
        """
        Check if monitoring is active.

        Returns:
            bool: True if monitoring is active
        """
        return self.is_launched and self.monitoring_integration is not None

    def get_trading_mode(self) -> str:
        """
        Get current trading mode.

        Returns:
            str: Current trading mode
        """
        return self.trading_mode

    def get_launch_info(self) -> Dict[str, Any]:
        """
        Get launch information.

        Returns:
            Dict[str, Any]: Launch information
        """
        return {
            'is_launched': self.is_launched,
            'trading_mode': self.trading_mode,
            'launch_time': self.launch_time,
            'monitoring_active': self.is_monitoring_active()
        }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dict[str, Any]: System status information
        """
        status = {
            'launcher_status': {
                'is_launched': self.is_launched,
                'trading_mode': self.trading_mode,
                'launch_time': self.launch_time,
                'monitoring_active': self.is_monitoring_active()
            }
        }

        if self.monitoring_integration:
            try:
                integration_status = self.monitoring_integration.get_system_status()
                status['monitoring_integration_status'] = integration_status
            except Exception as e:
                status['monitoring_integration_status'] = {'error': str(e)}

        return status

    @handles_errors(
        default_return=None,
        context='auto monitoring launcher cleanup'
    )
    async def stop(self) -> None:
        """Stop the auto monitoring launcher."""
        try:
            self.logger.info('🛑 Stopping Auto Enhanced Monitoring System...')

            if self.monitoring_integration:
                await self.monitoring_integration.stop()
                self.logger.info('🔍 Monitoring integration stopped')

            self.is_launched = False
            self.logger.info('✅ Auto Enhanced Monitoring System stopped successfully')

        except Exception as e:
            self.logger.exception(f'Error stopping auto monitoring launcher: {e}')

# Global instance for easy access
_auto_monitoring_launcher: Optional[AutoMonitoringLauncher] = None

async def launch_auto_monitoring() -> bool:
    """
    Launch the auto monitoring system globally.

    Returns:
        bool: True if launch successful, False otherwise
    """
    global _auto_monitoring_launcher

    try:
        if _auto_monitoring_launcher is None:
            _auto_monitoring_launcher = AutoMonitoringLauncher()

        return await _auto_monitoring_launcher.launch()
    except Exception as e:
        system_logger.exception(f'Error launching auto monitoring: {e}')
        return False

async def get_auto_monitoring() -> Optional[AutoMonitoringLauncher]:
    """
    Get the global auto monitoring launcher instance.

    Returns:
        Optional[AutoMonitoringLauncher]: Global launcher instance
    """
    return _auto_monitoring_launcher

async def auto_capture_trade_decision(trade_data: Dict[str, Any]) -> None:
    """
    Automatically capture a trade decision using the global launcher.

    Args:
        trade_data: Trade decision data
    """
    try:
        launcher = await get_auto_monitoring()
        if launcher:
            await launcher.capture_trade_decision(trade_data)
    except Exception as e:
        system_logger.exception(f'Error in automatic trade decision capture: {e}')

async def auto_update_performance(performance_data: Dict[str, Any], model_id: Optional[str] = None) -> None:
    """
    Automatically update performance metrics using the global launcher.

    Args:
        performance_data: Performance metrics data
        model_id: Optional model identifier
    """
    try:
        launcher = await get_auto_monitoring()
        if launcher:
            await launcher.update_performance(performance_data, model_id)
    except Exception as e:
        system_logger.exception(f'Error in automatic performance update: {e}')

async def auto_update_ensemble(ensemble_data: Dict[str, Any], ensemble_id: Optional[str] = None) -> None:
    """
    Automatically update ensemble performance using the global launcher.

    Args:
        ensemble_data: Ensemble performance data
        ensemble_id: Optional ensemble identifier
    """
    try:
        launcher = await get_auto_monitoring()
        if launcher:
            await launcher.update_ensemble(ensemble_data, ensemble_id)
    except Exception as e:
        system_logger.exception(f'Error in automatic ensemble update: {e}')

async def stop_auto_monitoring() -> None:
    """Stop the global auto monitoring system."""
    try:
        global _auto_monitoring_launcher
        if _auto_monitoring_launcher:
            await _auto_monitoring_launcher.stop()
            _auto_monitoring_launcher = None
    except Exception as e:
        system_logger.exception(f'Error stopping auto monitoring: {e}')

def is_auto_monitoring_active() -> bool:
    """
    Check if auto monitoring is active.

    Returns:
        bool: True if auto monitoring is active
    """
    global _auto_monitoring_launcher
    return _auto_monitoring_launcher is not None and _auto_monitoring_launcher.is_monitoring_active()
