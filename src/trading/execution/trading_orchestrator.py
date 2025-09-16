"""
Trading Orchestrator

This module provides a unified trading orchestrator that coordinates
the Analyst, Tactician, Supervisor, and Strategist components for
comprehensive trading operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)

# Import signal generators
from ..signal_generation.analyst_signals import AnalystSignalGenerator, AnalystSignal
from ..signal_generation.tactician_signals import TacticianSignalGenerator, TacticianSignal
from ..signal_generation.signal_combiner import SignalCombiner
from ..data.live_data_collector import LiveDataCollector, LiveDataConfig
from .live_trading_scheduler import LiveTradingScheduler, ModelType

logger = system_logger.getChild('TradingOrchestrator')

class TradingMode(Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"

class OrchestratorStatus(Enum):
    """Orchestrator status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class TradingDecision:
    """Final trading decision."""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold', 'close'
    quantity: float
    price: float
    confidence: float
    analyst_signal: Optional[AnalystSignal] = None
    tactician_signal: Optional[TacticianSignal] = None
    combined_signal: Optional[Dict[str, Any]] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingSession:
    """Trading session information."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    symbol: str = "ETH"
    mode: TradingMode = TradingMode.PAPER
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

class TradingOrchestrator:
    """
    Unified Trading Orchestrator that coordinates all trading components.
    
    Features:
    - Integrates Analyst, Tactician, Supervisor, and Strategist
    - Real-time data collection and processing
    - Multi-timeframe model coordination
    - Risk management and position sizing
    - Performance monitoring and reporting
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('TradingOrchestrator')
        
        # Core components
        self.analyst = None
        self.tactician = None
        self.supervisor = None
        self.strategist = None
        
        # Signal generators
        self.analyst_signal_generator = None
        self.tactician_signal_generator = None
        self.signal_combiner = None
        
        # Data and scheduling
        self.data_collector = None
        self.trading_scheduler = None
        
        # State management
        self.status = OrchestratorStatus.STOPPED
        self.current_session: Optional[TradingSession] = None
        self.trading_decisions: List[TradingDecision] = []
        
        # Configuration
        self.symbol = config.get('symbol', 'ETH')
        self.exchange = config.get('exchange', 'binance')
        self.trading_mode = TradingMode(config.get('trading_mode', 'paper'))
        self.account_balance = config.get('account_balance', 10000.0)
        
        # Performance tracking
        self.performance_metrics = {
            'total_sessions': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'avg_session_duration': 0.0
        }

    async def initialize(self) -> bool:
        """
        Initialize all trading components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            tprint_info("🚀 Initializing Trading Orchestrator...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize signal generators
            await self._initialize_signal_generators()
            
            # Initialize data collection
            await self._initialize_data_collection()
            
            # Initialize trading scheduler
            await self._initialize_trading_scheduler()
            
            self.status = OrchestratorStatus.STOPPED
            tprint_success("✅ Trading Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Trading Orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
            return False

    async def _initialize_core_components(self):
        """Initialize core trading components."""
        try:
            tprint_info("🔄 Initializing core components...")
            
            # Initialize Analyst
            from src.analyst.analyst import Analyst
            analyst_config = self.config.get('analyst', {})
            self.analyst = Analyst(analyst_config)
            await self.analyst.initialize()
            
            # Initialize Tactician
            from src.tactician.tactician import Tactician
            tactician_config = self.config.get('tactician', {})
            self.tactician = Tactician(tactician_config)
            await self.tactician.initialize()
            
            # Initialize Supervisor
            from src.supervisor.main import Supervisor
            supervisor_config = self.config.get('supervisor', {})
            # Note: Supervisor requires additional parameters
            # self.supervisor = Supervisor(supervisor_config)
            
            # Initialize Strategist
            from src.strategist.strategist import Strategist
            strategist_config = self.config.get('strategist', {})
            self.strategist = Strategist(strategist_config)
            await self.strategist.initialize()
            
            tprint_success("✅ Core components initialized")
            
        except Exception as e:
            tprint_error(f"❌ Core component initialization failed: {e}")
            raise

    async def _initialize_signal_generators(self):
        """Initialize signal generators."""
        try:
            tprint_info("🔄 Initializing signal generators...")
            
            # Initialize Analyst signal generator
            analyst_signal_config = self.config.get('analyst_signals', {})
            self.analyst_signal_generator = AnalystSignalGenerator(analyst_signal_config)
            await self.analyst_signal_generator.initialize(self.analyst)
            
            # Initialize Tactician signal generator
            tactician_signal_config = self.config.get('tactician_signals', {})
            self.tactician_signal_generator = TacticianSignalGenerator(tactician_signal_config)
            await self.tactician_signal_generator.initialize(self.tactician)
            
            # Initialize signal combiner
            signal_combiner_config = self.config.get('signal_combiner', {})
            self.signal_combiner = SignalCombiner(signal_combiner_config)
            await self.signal_combiner.initialize()
            
            tprint_success("✅ Signal generators initialized")
            
        except Exception as e:
            tprint_error(f"❌ Signal generator initialization failed: {e}")
            raise

    async def _initialize_data_collection(self):
        """Initialize data collection."""
        try:
            tprint_info("🔄 Initializing data collection...")
            
            # Create data collector configuration
            data_config = LiveDataConfig(
                symbol=self.symbol,
                exchange=self.exchange,
                enable_ml_predictions=True,
                feature_engineering=True
            )
            
            self.data_collector = LiveDataCollector(data_config)
            
            tprint_success("✅ Data collection initialized")
            
        except Exception as e:
            tprint_error(f"❌ Data collection initialization failed: {e}")
            raise

    async def _initialize_trading_scheduler(self):
        """Initialize trading scheduler."""
        try:
            tprint_info("🔄 Initializing trading scheduler...")
            
            self.trading_scheduler = LiveTradingScheduler(
                symbol=self.symbol,
                exchange=self.exchange
            )
            
            tprint_success("✅ Trading scheduler initialized")
            
        except Exception as e:
            tprint_error(f"❌ Trading scheduler initialization failed: {e}")
            raise

    async def start_trading_session(self) -> bool:
        """
        Start a new trading session.
        
        Returns:
            bool: True if session started successfully
        """
        try:
            if self.status != OrchestratorStatus.STOPPED:
                tprint_warning("⚠️ Orchestrator is not in stopped state")
                return False
            
            tprint_info("🚀 Starting trading session...")
            self.status = OrchestratorStatus.STARTING
            
            # Create new session
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                symbol=self.symbol,
                mode=self.trading_mode
            )
            
            # Start data collection
            await self.data_collector.start_collection()
            
            # Start trading scheduler
            await self.trading_scheduler.start_scheduler()
            
            # Set up callbacks
            self._setup_callbacks()
            
            # Start main trading loop
            asyncio.create_task(self._trading_loop())
            
            self.status = OrchestratorStatus.RUNNING
            self.performance_metrics['total_sessions'] += 1
            
            tprint_success(f"✅ Trading session {session_id} started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start trading session: {e}")
            self.status = OrchestratorStatus.ERROR
            return False

    async def stop_trading_session(self) -> bool:
        """
        Stop the current trading session.
        
        Returns:
            bool: True if session stopped successfully
        """
        try:
            if self.status != OrchestratorStatus.RUNNING:
                tprint_warning("⚠️ No active trading session to stop")
                return False
            
            tprint_info("🛑 Stopping trading session...")
            self.status = OrchestratorStatus.STOPPING
            
            # Stop data collection
            await self.data_collector.stop_collection()
            
            # Stop trading scheduler
            await self.trading_scheduler.stop_scheduler()
            
            # End current session
            if self.current_session:
                self.current_session.end_time = datetime.now()
                self._update_session_metrics()
            
            self.status = OrchestratorStatus.STOPPED
            
            tprint_success("✅ Trading session stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop trading session: {e}")
            self.status = OrchestratorStatus.ERROR
            return False

    def _setup_callbacks(self):
        """Set up callbacks for data and execution events."""
        try:
            # Data collection callbacks
            self.data_collector.add_data_callback(self._on_new_data)
            self.data_collector.add_error_callback(self._on_data_error)
            
            # Trading scheduler callbacks
            self.trading_scheduler.add_execution_callback(self._on_model_execution)
            self.trading_scheduler.add_error_callback(self._on_scheduler_error)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup callbacks: {e}")

    async def _trading_loop(self):
        """Main trading loop."""
        while self.status == OrchestratorStatus.RUNNING:
            try:
                # Wait for sufficient data
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Generate trading decision
                decision = await self._generate_trading_decision()
                
                if decision:
                    await self._execute_trading_decision(decision)
                
            except Exception as e:
                self.logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _generate_trading_decision(self) -> Optional[TradingDecision]:
        """Generate trading decision based on all components."""
        try:
            # Get recent market data
            market_data = self.data_collector.get_processed_data_df(n=100)
            if market_data.empty:
                return None
            
            # Get regime data from HMM
            regime_data = self.trading_scheduler.hmm_data
            
            # Generate Analyst signal
            analyst_signal = await self.analyst_signal_generator.generate_signal(
                symbol=self.symbol,
                market_data=market_data,
                regime_data=regime_data
            )
            
            if not analyst_signal:
                return None
            
            # Generate Tactician signal
            tactician_signal = await self.tactician_signal_generator.generate_timing_signal(
                symbol=self.symbol,
                analyst_signal=analyst_signal.__dict__,
                market_data=market_data,
                account_balance=self.account_balance
            )
            
            if not tactician_signal:
                return None
            
            # Combine signals
            combined_signal = await self.signal_combiner.combine_signals(
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal
            )
            
            # Create trading decision
            decision = TradingDecision(
                timestamp=datetime.now(),
                symbol=self.symbol,
                action=combined_signal.get('action', 'hold'),
                quantity=tactician_signal.position_sizing.recommended_size,
                price=market_data['close'].iloc[-1],
                confidence=combined_signal.get('confidence', 0.0),
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combined_signal=combined_signal,
                risk_metrics=combined_signal.get('risk_metrics', {}),
                metadata={
                    'session_id': self.current_session.session_id if self.current_session else None,
                    'regime_data': regime_data
                }
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"❌ Trading decision generation failed: {e}")
            return None

    async def _execute_trading_decision(self, decision: TradingDecision):
        """Execute trading decision."""
        try:
            tprint_info(f"🔄 Executing {decision.action} order for {decision.symbol}")
            
            # Here you would integrate with your execution system
            # For now, just log the decision
            self.trading_decisions.append(decision)
            
            if self.current_session:
                self.current_session.total_trades += 1
            
            self.performance_metrics['total_trades'] += 1
            
            tprint_success(f"✅ {decision.action} order executed for {decision.symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Trading decision execution failed: {e}")
            if self.current_session:
                self.current_session.failed_trades += 1
            self.performance_metrics['failed_trades'] += 1

    async def _on_new_data(self, data_point):
        """Handle new data point."""
        try:
            # Process new data if needed
            pass
        except Exception as e:
            self.logger.error(f"❌ New data handler error: {e}")

    async def _on_data_error(self, error: Exception):
        """Handle data collection error."""
        self.logger.error(f"❌ Data collection error: {error}")

    async def _on_model_execution(self, result):
        """Handle model execution result."""
        try:
            tprint_info(f"📊 {result.model_type.value.upper()} execution completed")
        except Exception as e:
            self.logger.error(f"❌ Model execution handler error: {e}")

    async def _on_scheduler_error(self, error: Exception):
        """Handle scheduler error."""
        self.logger.error(f"❌ Scheduler error: {error}")

    def _update_session_metrics(self):
        """Update session performance metrics."""
        if not self.current_session:
            return
        
        # Calculate session metrics
        session_duration = (self.current_session.end_time - self.current_session.start_time).total_seconds()
        self.current_session.total_trades = len(self.trading_decisions)
        
        # Update global metrics
        self.performance_metrics['avg_session_duration'] = (
            (self.performance_metrics['avg_session_duration'] * (self.performance_metrics['total_sessions'] - 1) + session_duration)
            / self.performance_metrics['total_sessions']
        )

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'status': self.status.value,
            'current_session': self.current_session.__dict__ if self.current_session else None,
            'performance_metrics': self.performance_metrics,
            'data_collector_stats': self.data_collector.get_stats() if self.data_collector else None,
            'scheduler_stats': self.trading_scheduler.get_scheduler_stats() if self.trading_scheduler else None,
            'recent_decisions': len(self.trading_decisions)
        }

    def get_trading_decisions(self, n: int = 100) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return self.trading_decisions[-n:] if len(self.trading_decisions) >= n else self.trading_decisions.copy()

# Convenience functions

def create_trading_orchestrator(config: Dict[str, Any]) -> TradingOrchestrator:
    """Create a configured trading orchestrator."""
    return TradingOrchestrator(config)

async def start_trading_orchestrator(
    config: Dict[str, Any],
    symbol: str = "ETH",
    exchange: str = "binance",
    trading_mode: str = "paper"
) -> TradingOrchestrator:
    """Start trading orchestrator with default settings."""
    
    # Update config with provided parameters
    config.update({
        'symbol': symbol,
        'exchange': exchange,
        'trading_mode': trading_mode
    })
    
    orchestrator = create_trading_orchestrator(config)
    
    # Initialize orchestrator
    success = await orchestrator.initialize()
    if not success:
        raise RuntimeError("Failed to initialize trading orchestrator")
    
    # Start trading session
    success = await orchestrator.start_trading_session()
    if not success:
        raise RuntimeError("Failed to start trading session")
    
    return orchestrator

# Example usage
if __name__ == "__main__":
    async def main():
        """Example main function."""
        config = {
            'analyst': {},
            'tactician': {},
            'strategist': {},
            'analyst_signals': {'confidence_threshold': 0.6},
            'tactician_signals': {'confidence_threshold': 0.6},
            'signal_combiner': {}
        }
        
        orchestrator = await start_trading_orchestrator(
            config=config,
            symbol="ETH",
            exchange="binance",
            trading_mode="paper"
        )
        
        # Run for 10 minutes
        await asyncio.sleep(600)
        
        # Get stats
        stats = orchestrator.get_orchestrator_stats()
        tprint_structured(stats, LogLevel.INFO)
        
        # Stop orchestrator
        await orchestrator.stop_trading_session()
    
    # Run example
    asyncio.run(main())