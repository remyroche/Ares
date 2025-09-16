#!/usr/bin/env python3
"""
Example: Integrated Trading System

This script demonstrates the new integrated trading system that combines
Analyst, Tactician, Supervisor, and Strategist components with live trading
coordination and real-time monitoring.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the new integrated trading system
from src.trading import (
    # Core components
    TradingOrchestrator, TradingMode, OrchestratorStatus,
    LiveTradingScheduler, ModelType, ExecutionStatus,
    
    # Signal generators
    AnalystSignalGenerator, AnalystSignal, SignalType, SignalStrength,
    TacticianSignalGenerator, TacticianSignal, TimingSignal, TimingConfidence,
    
    # Data collection
    LiveDataCollector, LiveDataConfig, CollectionMode, DataQuality, CollectionInterval,
    
    # Monitoring
    TradeMonitor, Trade, TradeStatus, Alert, AlertLevel,
    
    # Convenience functions
    create_trading_orchestrator, start_trading_orchestrator,
    create_live_trading_scheduler, start_live_trading_scheduler,
    create_analyst_signal_generator, create_tactician_signal_generator,
    create_trade_monitor, start_trade_monitoring
)

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('IntegratedTradingExample')

class IntegratedTradingSystem:
    """
    Integrated Trading System that demonstrates the new unified architecture.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the integrated trading system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('IntegratedTradingSystem')
        
        # Core components
        self.orchestrator: TradingOrchestrator = None
        self.scheduler: LiveTradingScheduler = None
        self.trade_monitor: TradeMonitor = None
        
        # Signal generators
        self.analyst_signal_generator: AnalystSignalGenerator = None
        self.tactician_signal_generator: TacticianSignalGenerator = None
        
        # Data collector
        self.data_collector: LiveDataCollector = None
        
        # State
        self.is_running = False
        self.start_time: datetime = None

    async def initialize(self) -> bool:
        """
        Initialize all components of the integrated trading system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            tprint_info("🚀 Initializing Integrated Trading System...")
            
            # Initialize trade monitor
            self.trade_monitor = create_trade_monitor(self.config.get('trade_monitor', {}))
            await self.trade_monitor.initialize()
            
            # Initialize signal generators
            self.analyst_signal_generator = create_analyst_signal_generator(
                self.config.get('analyst_signals', {})
            )
            
            self.tactician_signal_generator = create_tactician_signal_generator(
                self.config.get('tactician_signals', {})
            )
            
            # Initialize data collector
            data_config = LiveDataConfig(
                symbol=self.config.get('symbol', 'ETH'),
                exchange=self.config.get('exchange', 'binance'),
                enable_ml_predictions=True,
                feature_engineering=True
            )
            self.data_collector = LiveDataCollector(data_config)
            
            # Initialize trading scheduler
            self.scheduler = create_live_trading_scheduler(
                symbol=self.config.get('symbol', 'ETH'),
                exchange=self.config.get('exchange', 'binance')
            )
            
            # Initialize orchestrator
            self.orchestrator = create_trading_orchestrator(self.config)
            await self.orchestrator.initialize()
            
            tprint_success("✅ Integrated Trading System initialized successfully")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Integrated Trading System: {e}")
            return False

    async def start_system(self) -> bool:
        """
        Start the integrated trading system.
        
        Returns:
            bool: True if system started successfully
        """
        try:
            if self.is_running:
                tprint_warning("⚠️ System already running")
                return False
            
            tprint_info("🚀 Starting Integrated Trading System...")
            
            # Start trade monitoring
            await self.trade_monitor.start_monitoring()
            
            # Start data collection
            await self.data_collector.start_collection()
            
            # Start trading scheduler
            await self.scheduler.start_scheduler()
            
            # Start orchestrator
            await self.orchestrator.start_trading_session()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            tprint_success("✅ Integrated Trading System started successfully")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to start Integrated Trading System: {e}")
            return False

    async def stop_system(self) -> bool:
        """
        Stop the integrated trading system.
        
        Returns:
            bool: True if system stopped successfully
        """
        try:
            if not self.is_running:
                return True
            
            tprint_info("🛑 Stopping Integrated Trading System...")
            
            # Stop orchestrator
            await self.orchestrator.stop_trading_session()
            
            # Stop scheduler
            await self.scheduler.stop_scheduler()
            
            # Stop data collection
            await self.data_collector.stop_collection()
            
            # Stop trade monitoring
            await self.trade_monitor.stop_monitoring()
            
            self.is_running = False
            
            tprint_success("✅ Integrated Trading System stopped successfully")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to stop Integrated Trading System: {e}")
            return False

    async def run_demo(self, duration_minutes: int = 5):
        """
        Run a demonstration of the integrated trading system.
        
        Args:
            duration_minutes: Duration to run the demo in minutes
        """
        try:
            tprint_info(f"🎯 Running demo for {duration_minutes} minutes...")
            
            # Set up periodic reporting
            asyncio.create_task(self._periodic_reporting())
            
            # Run for specified duration
            await asyncio.sleep(duration_minutes * 60)
            
            tprint_success("✅ Demo completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Demo failed: {e}")

    async def _periodic_reporting(self):
        """Periodic reporting of system status."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                # Get system statistics
                orchestrator_stats = self.orchestrator.get_orchestrator_stats()
                scheduler_stats = self.scheduler.get_scheduler_stats()
                monitor_stats = self.trade_monitor.get_monitoring_stats()
                data_stats = self.data_collector.get_stats()
                
                # Print status report
                tprint_info("📊 System Status Report:")
                tprint_info(f"   Orchestrator: {orchestrator_stats['status']}")
                tprint_info(f"   Scheduler: {scheduler_stats['is_running']}")
                tprint_info(f"   Data Collection: {data_stats['is_running']}")
                tprint_info(f"   Trade Monitoring: {monitor_stats['is_monitoring']}")
                tprint_info(f"   Active Trades: {monitor_stats['active_trades_count']}")
                tprint_info(f"   Total Trades: {monitor_stats['total_trades']}")
                
                # Performance metrics
                perf_metrics = monitor_stats['performance_metrics']
                if perf_metrics['total_trades'] > 0:
                    tprint_info(f"   Win Rate: {perf_metrics['win_rate']:.2%}")
                    tprint_info(f"   Total PnL: {perf_metrics['total_pnl']:.4f}")
                    tprint_info(f"   Current Drawdown: {perf_metrics['current_drawdown']:.2%}")
                
            except Exception as e:
                self.logger.error(f"❌ Periodic reporting error: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time,
            'uptime_seconds': (
                (datetime.now() - self.start_time).total_seconds() 
                if self.start_time else 0
            ),
            'orchestrator_stats': self.orchestrator.get_orchestrator_stats() if self.orchestrator else None,
            'scheduler_stats': self.scheduler.get_scheduler_stats() if self.scheduler else None,
            'monitor_stats': self.trade_monitor.get_monitoring_stats() if self.trade_monitor else None,
            'data_stats': self.data_collector.get_stats() if self.data_collector else None
        }

# Example usage and demonstration
async def main():
    """Main function demonstrating the integrated trading system."""
    
    # Configuration for the integrated system
    config = {
        'symbol': 'ETH',
        'exchange': 'binance',
        'trading_mode': 'paper',
        'account_balance': 10000.0,
        
        # Component configurations
        'analyst': {
            'confidence_threshold': 0.6,
            'enable_regime_detection': True
        },
        'tactician': {
            'confidence_threshold': 0.6,
            'risk_per_trade': 0.02,
            'max_leverage': 3.0
        },
        'strategist': {
            'enable_regime_detection': True,
            'use_vectorized_calculations': True
        },
        'analyst_signals': {
            'confidence_threshold': 0.6,
            'max_history': 1000
        },
        'tactician_signals': {
            'confidence_threshold': 0.6,
            'risk_per_trade': 0.02,
            'max_leverage': 3.0,
            'max_history': 1000
        },
        'signal_combiner': {
            'combination_method': 'weighted_average',
            'analyst_weight': 0.6,
            'tactician_weight': 0.4,
            'confidence_threshold': 0.6
        },
        'trade_monitor': {
            'max_drawdown_alert': 0.05,
            'min_win_rate_alert': 0.4,
            'max_loss_per_trade_alert': 0.02,
            'max_daily_loss_alert': 0.05
        }
    }
    
    # Create and initialize the system
    system = IntegratedTradingSystem(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        tprint_info("🛑 Received shutdown signal")
        asyncio.create_task(system.stop_system())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize the system
        success = await system.initialize()
        if not success:
            tprint_error("❌ Failed to initialize system")
            return
        
        # Start the system
        success = await system.start_system()
        if not success:
            tprint_error("❌ Failed to start system")
            return
        
        # Run demo for 5 minutes
        await system.run_demo(duration_minutes=5)
        
        # Get final statistics
        final_stats = system.get_system_stats()
        tprint_info("📊 Final System Statistics:")
        tprint_info(f"   Uptime: {final_stats['uptime_seconds']:.0f} seconds")
        
        if final_stats['monitor_stats']:
            perf_metrics = final_stats['monitor_stats']['performance_metrics']
            tprint_info(f"   Total Trades: {perf_metrics['total_trades']}")
            tprint_info(f"   Win Rate: {perf_metrics['win_rate']:.2%}")
            tprint_info(f"   Total PnL: {perf_metrics['total_pnl']:.4f}")
            tprint_info(f"   Max Drawdown: {perf_metrics['max_drawdown']:.2%}")
        
    except KeyboardInterrupt:
        tprint_info("🛑 Demo interrupted by user")
    except Exception as e:
        tprint_error(f"❌ Demo failed: {e}")
    finally:
        # Stop the system
        await system.stop_system()
        tprint_success("✅ Demo completed")

if __name__ == "__main__":
    # Run the integrated trading system demo
    asyncio.run(main())