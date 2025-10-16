#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Auto Monitoring Demo

This script demonstrates the automatic activation of the enhanced monitoring system
when the trading system is launched in any mode (BACKTEST, PAPER, LIVE).
"""

import os
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import system_logger, setup_logging
import logging
import time

from src.monitoring.auto_monitoring_launcher import (
    launch_auto_monitoring, get_auto_monitoring, stop_auto_monitoring,
    auto_capture_trade_decision, auto_update_performance, auto_update_ensemble
)

async def demo_auto_monitoring():
    """Demonstrate automatic monitoring activation."""
    tprint("🚀 Auto Enhanced Monitoring System Demo")
    tprint("=" * 50)

    # Setup logging
    setup_logging()
    logger = system_logger.getChild('AutoMonitoringDemo')

    try:
        # Set trading mode (can be BACKTEST, PAPER, or LIVE)
        trading_mode = os.environ.get('TRADING_MODE', 'PAPER')
        tprint(f"📊 Trading Mode: {trading_mode}")

        # Launch auto monitoring system
        tprint("\n🔍 Launching Auto Enhanced Monitoring System...")
        success = await launch_auto_monitoring()

        if not success:
            tprint("❌ Failed to launch auto monitoring system")
            return

        tprint("✅ Auto Enhanced Monitoring System launched successfully!")

        # Get the launcher instance
        launcher = await get_auto_monitoring()
        if not launcher:
            tprint("❌ Failed to get auto monitoring launcher")
            return

        # Display system status
        tprint("\n📋 System Status:")
        status = launcher.get_system_status()
        tprint(f"   🚀 Launched: {status['launcher_status']['is_launched']}")
        tprint(f"   📊 Trading Mode: {status['launcher_status']['trading_mode']}")
        tprint(f"   🕐 Launch Time: {status['launcher_status']['launch_time']}")
        tprint(f"   🎯 Monitoring Active: {status['launcher_status']['monitoring_active']}")

        # Simulate trade decisions
        tprint("\n📈 Simulating Trade Decisions...")

        # Simulate multiple trade decisions
        for i in range(3):
            trade_data = {
                'timestamp': datetime.now(),
                'exchange': 'BINANCE',
                'symbol': 'ETHUSDT',
                'price': 2000.0 + (i * 10),
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.1,
                'confidence': 0.75 + (i * 0.05),
                'position_size': 0.1,
                'leverage': 1.0,
                'trade_metadata': {
                    'trade_id': f'DEMO_TRADE_{i+1}',
                    'commission': 0.001,
                    'slippage': 0.0005,
                    'model_weights': {'analyst': 0.6, 'tactician': 0.4},
                    'model_confidences': {'analyst': 0.8, 'tactician': 0.7},
                    'hmm_regime': f'regime_{i+1}',
                    'market_conditions': {'volatility': 0.15, 'trend': 'bullish'}
                }
            }

            tprint(f"   📊 Recording trade {i+1}: {trade_data['action']} {trade_data['symbol']} @ ${trade_data['price']}")
            await auto_capture_trade_decision(trade_data)

        # Simulate performance updates
        tprint("\n📊 Simulating Performance Updates...")

        performance_data = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'auc': 0.90,
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.08,
            'win_rate': 0.65,
            'profit_factor': 1.45
        }

        tprint("   📈 Updating model performance metrics...")
        await auto_update_performance(performance_data, 'analyst_model')
        await auto_update_performance(performance_data, 'tactician_model')

        # Simulate ensemble updates
        tprint("\n🎯 Simulating Ensemble Updates...")

        ensemble_data = {
            'ensemble_accuracy': 0.87,
            'ensemble_sharpe': 1.35,
            'model_weights': {'analyst': 0.6, 'tactician': 0.4},
            'consensus_score': 0.85,
            'disagreement_level': 0.12
        }

        tprint("   🎯 Updating ensemble performance...")
        await auto_update_ensemble(ensemble_data, 'dual_model_ensemble')

        # Display final status
        tprint("\n📋 Final System Status:")
        final_status = launcher.get_system_status()
        tprint(f"   🚀 Launched: {final_status['launcher_status']['is_launched']}")
        tprint(f"   📊 Trading Mode: {final_status['launcher_status']['trading_mode']}")
        tprint(f"   🎯 Monitoring Active: {final_status['launcher_status']['monitoring_active']}")

        if 'monitoring_integration_status' in final_status:
            integration_status = final_status['monitoring_integration_status']
            if 'enhanced_monitoring_status' in integration_status:
                monitoring_status = integration_status['enhanced_monitoring_status']
                tprint(f"   📈 Decisions Recorded: {monitoring_status.get('total_decisions_recorded', 0)}")
                tprint(f"   🔍 SHAP Enabled: {monitoring_status.get('shap_enabled', False)}")
                tprint(f"   🔍 LIME Enabled: {monitoring_status.get('lime_enabled', False)}")

        tprint("\n✅ Auto Enhanced Monitoring System Demo completed successfully!")
        tprint("\n📋 Key Features Demonstrated:")
        tprint("   🎯 Automatic trade decision capture")
        tprint("   📊 Performance metrics tracking")
        tprint("   🎯 Ensemble performance monitoring")
        tprint("   🔍 SHAP/LIME explanations ready")
        tprint("   📋 Daily and monthly CSV exports configured")
        tprint("   📈 Real-time monitoring active")

    except Exception as e:
        logger.exception(f"Error in auto monitoring demo: {e}")
        tprint(f"❌ Demo failed: {e}")

    finally:
        # Stop the monitoring system
        tprint("\n🛑 Stopping Auto Enhanced Monitoring System...")
        await stop_auto_monitoring()
        tprint("✅ Auto Enhanced Monitoring System stopped successfully!")

async def main():
    """Main entry point."""
    tprint("🚀 Starting Auto Enhanced Monitoring System Demo...")
    tprint("This demo shows how the monitoring system automatically activates")
    tprint("when the trading system is launched in any mode.\n")

    await demo_auto_monitoring()

    tprint("\n🎉 Demo completed! The enhanced monitoring system is now")
    tprint("automatically integrated with your trading system and will")
    tprint("activate whenever you launch the Ares pipeline in trading mode.")

if __name__ == "__main__":
    asyncio.run(main())
