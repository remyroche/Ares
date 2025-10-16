"""
NAS/TAS Trading Main Entry Point

This module provides the main entry point for live trading with NAS/TAS enhanced
Analyst and Tactician models.

Usage:
    python -m src.trading.nas_tas_trading_main
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_success, tprint_error
from src.trading.execution.trading_orchestrator import TradingOrchestrator
from src.config.trading import get_trading_config
from src.config.config import NASConfig, TASConfig

logger = system_logger.getChild('NASTASTradingMain')

async def main():
    """Main entry point for NAS/TAS enhanced trading."""
    try:
        tprint_info("🚀 Starting NAS/TAS Enhanced Trading...")

        # Get trading configuration
        trading_config = get_trading_config()

        # Initialize NAS/TAS configuration
        nas_config = NASConfig()
        tas_config = TASConfig()

        # Update trading config with NAS/TAS settings
        trading_config.update({
            'nas_tas_enabled': True,
            'nas_config': nas_config,
            'tas_config': tas_config,
            'nas_models': {},  # Will be loaded from saved models
            'tas_models': {},  # Will be loaded from saved models
            'analyst_signals': {
                'enable_nas_enhancement': True,
                'nas_confidence_threshold': nas_config.nas_confidence_threshold,
                'nas_timeframe': nas_config.nas_timeframe,
                'regime_timeframe': nas_config.regime_timeframe
            },
            'tactician_signals': {
                'enable_tas_enhancement': True,
                'tas_confidence_threshold': tas_config.tas_confidence_threshold,
                'tas_timeframe': tas_config.tas_timeframe
            }
        })

        # Initialize trading orchestrator
        orchestrator = TradingOrchestrator(trading_config)

        # Initialize all components
        tprint_info("🔄 Initializing trading components...")
        await orchestrator.initialize()

        # Start trading session
        tprint_info("📈 Starting trading session...")
        await orchestrator.start_trading_session()

        tprint_success("✅ NAS/TAS enhanced trading started successfully!")
        tprint_info("🎯 Trading with enhanced Analyst (NAS) and Tactician (TAS) models")

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            tprint_info("🛑 Stopping trading session...")
            await orchestrator.stop_trading_session()
            tprint_success("✅ Trading session stopped gracefully")

        return True

    except Exception as e:
        tprint_error(f"❌ NAS/TAS trading failed: {str(e)}")
        logger.error(f"NAS/TAS trading failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
