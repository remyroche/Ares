#!/usr/bin/env python3
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
from src.monitoring.auto_monitoring_launcher import (
import logging
import time

    launch_auto_monitoring, get_auto_monitoring, stop_auto_monitoring,
    auto_capture_trade_decision, auto_update_performance, auto_update_ensemble
)

async def demo_auto_monitoring():
    """Demonstrate automatic monitoring activation."""
    print("🚀 Auto Enhanced Monitoring System Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = system_logger.getChild('AutoMonitoringDemo')
    
    try:
        # Set trading mode (can be BACKTEST, PAPER, or LIVE)
        trading_mode = os.environ.get('TRADING_MODE', 'PAPER')
        print(f"📊 Trading Mode: {trading_mode}")
        
        # Launch auto monitoring system
        print("\n🔍 Launching Auto Enhanced Monitoring System...")
        success = await launch_auto_monitoring()
        
        if not success:
            print("❌ Failed to launch auto monitoring system")
            return
        
        print("✅ Auto Enhanced Monitoring System launched successfully!")
        
        # Get the launcher instance
        launcher = await get_auto_monitoring()
        if not launcher:
            print("❌ Failed to get auto monitoring launcher")
            return
        
        # Display system status
        print("\n📋 System Status:")
        status = launcher.get_system_status()
        print(f"   🚀 Launched: {status['launcher_status']['is_launched']}")
        print(f"   📊 Trading Mode: {status['launcher_status']['trading_mode']}")
        print(f"   🕐 Launch Time: {status['launcher_status']['launch_time']}")
        print(f"   🎯 Monitoring Active: {status['launcher_status']['monitoring_active']}")
        
        # Simulate trade decisions
        print("\n📈 Simulating Trade Decisions...")
        
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
            
            print(f"   📊 Recording trade {i+1}: {trade_data['action']} {trade_data['symbol']} @ ${trade_data['price']}")
            await auto_capture_trade_decision(trade_data)
        
        # Simulate performance updates
        print("\n📊 Simulating Performance Updates...")
        
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
        
        print("   📈 Updating model performance metrics...")
        await auto_update_performance(performance_data, 'analyst_model')
        await auto_update_performance(performance_data, 'tactician_model')
        
        # Simulate ensemble updates
        print("\n🎯 Simulating Ensemble Updates...")
        
        ensemble_data = {
            'ensemble_accuracy': 0.87,
            'ensemble_sharpe': 1.35,
            'model_weights': {'analyst': 0.6, 'tactician': 0.4},
            'consensus_score': 0.85,
            'disagreement_level': 0.12
        }
        
        print("   🎯 Updating ensemble performance...")
        await auto_update_ensemble(ensemble_data, 'dual_model_ensemble')
        
        # Display final status
        print("\n📋 Final System Status:")
        final_status = launcher.get_system_status()
        print(f"   🚀 Launched: {final_status['launcher_status']['is_launched']}")
        print(f"   📊 Trading Mode: {final_status['launcher_status']['trading_mode']}")
        print(f"   🎯 Monitoring Active: {final_status['launcher_status']['monitoring_active']}")
        
        if 'monitoring_integration_status' in final_status:
            integration_status = final_status['monitoring_integration_status']
            if 'enhanced_monitoring_status' in integration_status:
                monitoring_status = integration_status['enhanced_monitoring_status']
                print(f"   📈 Decisions Recorded: {monitoring_status.get('total_decisions_recorded', 0)}")
                print(f"   🔍 SHAP Enabled: {monitoring_status.get('shap_enabled', False)}")
                print(f"   🔍 LIME Enabled: {monitoring_status.get('lime_enabled', False)}")
        
        print("\n✅ Auto Enhanced Monitoring System Demo completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("   🎯 Automatic trade decision capture")
        print("   📊 Performance metrics tracking")
        print("   🎯 Ensemble performance monitoring")
        print("   🔍 SHAP/LIME explanations ready")
        print("   📋 Daily and monthly CSV exports configured")
        print("   📈 Real-time monitoring active")
        
    except Exception as e:
        logger.exception(f"Error in auto monitoring demo: {e}")
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Stop the monitoring system
        print("\n🛑 Stopping Auto Enhanced Monitoring System...")
        await stop_auto_monitoring()
        print("✅ Auto Enhanced Monitoring System stopped successfully!")

async def main():
    """Main entry point."""
    print("🚀 Starting Auto Enhanced Monitoring System Demo...")
    print("This demo shows how the monitoring system automatically activates")
    print("when the trading system is launched in any mode.\n")
    
    await demo_auto_monitoring()
    
    print("\n🎉 Demo completed! The enhanced monitoring system is now")
    print("automatically integrated with your trading system and will")
    print("activate whenever you launch the Ares pipeline in trading mode.")

if __name__ == "__main__":
    asyncio.run(main())