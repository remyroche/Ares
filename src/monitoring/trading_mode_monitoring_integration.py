"""
Trading Mode Monitoring Integration

This module provides automatic integration of the enhanced monitoring system
with different trading modes (BACKTEST, PAPER, LIVE).
"""

import os

from typing import Any, Dict, Optional
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.config.environment import get_environment_settings
from .enhanced_monitoring_orchestrator import EnhancedMonitoringOrchestrator
import logging
import time

class TradingModeMonitoringIntegration:
    """
    Integrates enhanced monitoring system with different trading modes.
    Automatically activates monitoring based on trading environment.
    """

    def __init__(self):
        """Initialize trading mode monitoring integration."""
        self.logger = system_logger.getChild('TradingModeMonitoringIntegration')
        self.enhanced_monitoring: Optional[EnhancedMonitoringOrchestrator] = None
        self.is_initialized: bool = False
        self.trading_mode: str = 'PAPER'
        self.monitoring_config: Dict[str, Any] = {}

    @handles_errors(
        error_handlers={
            ValueError: (False, 'Invalid monitoring configuration'),
            AttributeError: (False, 'Missing required monitoring parameters'),
            KeyError: (False, 'Missing configuration keys')
        },
        default_return=False,
        context='trading mode monitoring initialization'
    )
    async def initialize(self) -> bool:
        """
        Initialize trading mode monitoring integration.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info('🔍 Initializing Trading Mode Monitoring Integration...')

            # Get environment settings
            env_settings = get_environment_settings()

            # Get trading mode from environment
            self.trading_mode = os.environ.get('TRADING_MODE', env_settings.trading_environment).upper()

            # Get monitoring configuration
            self.monitoring_config = env_settings.get_enhanced_monitoring_config()

            # Check if monitoring is enabled
            if not self.monitoring_config.get('enable_enhanced_monitoring', True):
                self.logger.info('⚠️ Enhanced monitoring is disabled in configuration')
                return True

            # Initialize enhanced monitoring orchestrator
            self.enhanced_monitoring = EnhancedMonitoringOrchestrator()
            await self.enhanced_monitoring.initialize()

            if self.enhanced_monitoring:
                self.is_initialized = True
                self.logger.info('✅ Trading Mode Monitoring Integration initialized successfully')
                self.logger.info(f'   📊 Trading Mode: {self.trading_mode}')
                self.logger.info(f'   🔍 SHAP Enabled: {self.monitoring_config.get("monitoring_enable_shap", True)}')
                self.logger.info(f'   🔍 LIME Enabled: {self.monitoring_config.get("monitoring_enable_lime", True)}')
                self.logger.info(f'   📈 Real-time Updates: {self.monitoring_config.get("monitoring_enable_real_time_updates", True)}')
                self.logger.info(f'   📋 Export Directory: {self.monitoring_config.get("monitoring_export_directory", "monitoring_exports")}')
                return True
            else:
                self.logger.warning('⚠️ Failed to initialize Enhanced Monitoring Orchestrator')
                return False

        except Exception as e:
            self.logger.exception(f'❌ Trading Mode Monitoring Integration initialization failed: {e}')
            return False

    @handles_errors(
        default_return=None,
        context='trade decision recording'
    )
    async def record_trade_decision(
        self,
        trade_data: Dict[str, Any],
        trading_mode: Optional[str] = None
    ) -> None:
        """
        Record a trade decision in the monitoring system.

        Args:
            trade_data: Trade decision data
            trading_mode: Override trading mode (optional)
        """
        try:
            if not self.is_initialized or not self.enhanced_monitoring:
                self.logger.warning('⚠️ Monitoring system not initialized, skipping trade recording')
                return

            # Use provided trading mode or default
            mode = trading_mode or self.trading_mode

            # Enhance trade data with monitoring context
            enhanced_trade_data = {
                **trade_data,
                'trading_mode': mode,
                'timestamp': trade_data.get('timestamp', datetime.now()),
                'monitoring_metadata': {
                    'integration_version': '1.0.0',
                    'auto_captured': True,
                    'trading_mode_detected': self.trading_mode
                }
            }

            # Record the trade decision
            await self.enhanced_monitoring.record_comprehensive_trade_decision(enhanced_trade_data)

            self.logger.info(f'📊 Trade decision recorded for {mode} mode')

        except Exception as e:
            self.logger.exception(f'Error recording trade decision: {e}')

    @handles_errors(
        default_return=None,
        context='performance update'
    )
    async def update_performance_metrics(
        self,
        performance_data: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> None:
        """
        Update performance metrics in the monitoring system.

        Args:
            performance_data: Performance metrics data
            model_id: Optional model identifier
        """
        try:
            if not self.is_initialized or not self.enhanced_monitoring:
                return

            # Update performance metrics
            await self.enhanced_monitoring.update_model_performance(
                model_id or 'default_model',
                performance_data
            )

            self.logger.info(f'📈 Performance metrics updated for model: {model_id or "default_model"}')

        except Exception as e:
            self.logger.exception(f'Error updating performance metrics: {e}')

    @handles_errors(
        default_return=None,
        context='ensemble update'
    )
    async def update_ensemble_performance(
        self,
        ensemble_data: Dict[str, Any],
        ensemble_id: Optional[str] = None
    ) -> None:
        """
        Update ensemble performance in the monitoring system.

        Args:
            ensemble_data: Ensemble performance data
            ensemble_id: Optional ensemble identifier
        """
        try:
            if not self.is_initialized or not self.enhanced_monitoring:
                return

            # Update ensemble performance
            await self.enhanced_monitoring.update_ensemble_performance(
                ensemble_id or 'default_ensemble',
                ensemble_data
            )

            self.logger.info(f'🎯 Ensemble performance updated for: {ensemble_id or "default_ensemble"}')

        except Exception as e:
            self.logger.exception(f'Error updating ensemble performance: {e}')

    def get_trading_mode(self) -> str:
        """
        Get current trading mode.

        Returns:
            str: Current trading mode
        """
        return self.trading_mode

    def is_monitoring_enabled(self) -> bool:
        """
        Check if monitoring is enabled.

        Returns:
            bool: True if monitoring is enabled
        """
        return self.is_initialized and self.enhanced_monitoring is not None

    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring configuration.

        Returns:
            Dict[str, Any]: Monitoring configuration
        """
        return self.monitoring_config.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.

        Returns:
            Dict[str, Any]: System status information
        """
        status = {
            'is_initialized': self.is_initialized,
            'trading_mode': self.trading_mode,
            'monitoring_enabled': self.is_monitoring_enabled(),
            'monitoring_config': self.monitoring_config
        }

        if self.enhanced_monitoring:
            try:
                monitoring_status = self.enhanced_monitoring.get_system_status()
                status['enhanced_monitoring_status'] = monitoring_status
            except Exception as e:
                status['enhanced_monitoring_status'] = {'error': str(e)}

        return status

    @handles_errors(
        default_return=None,
        context='trading mode monitoring cleanup'
    )
    async def stop(self) -> None:
        """Stop the trading mode monitoring integration."""
        try:
            self.logger.info('🛑 Stopping Trading Mode Monitoring Integration...')

            if self.enhanced_monitoring:
                await self.enhanced_monitoring.stop()
                self.logger.info('🔍 Enhanced monitoring stopped')

            self.is_initialized = False
            self.logger.info('✅ Trading Mode Monitoring Integration stopped successfully')

        except Exception as e:
            self.logger.exception(f'Error stopping trading mode monitoring integration: {e}')

# Global instance for easy access
_trading_mode_monitoring: Optional[TradingModeMonitoringIntegration] = None

async def get_trading_mode_monitoring() -> TradingModeMonitoringIntegration:
    """
    Get or create the global trading mode monitoring instance.

    Returns:
        TradingModeMonitoringIntegration: Global monitoring instance
    """
    global _trading_mode_monitoring

    if _trading_mode_monitoring is None:
        _trading_mode_monitoring = TradingModeMonitoringIntegration()
        await _trading_mode_monitoring.initialize()

    return _trading_mode_monitoring

async def record_trade_decision_auto(trade_data: Dict[str, Any]) -> None:
    """
    Automatically record a trade decision using the global monitoring instance.

    Args:
        trade_data: Trade decision data
    """
    try:
        monitoring = await get_trading_mode_monitoring()
        await monitoring.record_trade_decision(trade_data)
    except Exception as e:
        system_logger.exception(f'Error in automatic trade decision recording: {e}')

async def update_performance_auto(performance_data: Dict[str, Any], model_id: Optional[str] = None) -> None:
    """
    Automatically update performance metrics using the global monitoring instance.

    Args:
        performance_data: Performance metrics data
        model_id: Optional model identifier
    """
    try:
        monitoring = await get_trading_mode_monitoring()
        await monitoring.update_performance_metrics(performance_data, model_id)
    except Exception as e:
        system_logger.exception(f'Error in automatic performance update: {e}')

async def update_ensemble_auto(ensemble_data: Dict[str, Any], ensemble_id: Optional[str] = None) -> None:
    """
    Automatically update ensemble performance using the global monitoring instance.

    Args:
        ensemble_data: Ensemble performance data
        ensemble_id: Optional ensemble identifier
    """
    try:
        monitoring = await get_trading_mode_monitoring()
        await monitoring.update_ensemble_performance(ensemble_data, ensemble_id)
    except Exception as e:
        system_logger.exception(f'Error in automatic ensemble update: {e}')
