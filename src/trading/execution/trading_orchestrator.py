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
from ..monitoring.comprehensive_trade_monitor import comprehensive_trade_monitor, record_detailed_trade, update_trade_outcome
from ..monitoring.unified_trailing_manager import (
    UnifiedTrailingManager,
    TrailingAction,
    TrailingDecision,
)
from ..reporting.performance_reporter import performance_reporter, generate_trading_report
from ..reporting.dashboard_generator import dashboard_generator, create_trading_dashboard
from ..reporting.daily_recorder import daily_recorder, record_daily_trading_summary
from ..utils.helpers import prepare_trailing_feature_bundle, TrailingFeatureBundle

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
        
        # Cross-asset trade coordination
        self.trade_gate = self.config.get('trade_gate')

        # Trade decision callbacks (for cross-asset manager integration)
        self._on_trade_decision_callbacks: List[Any] = []

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

        # Unified trailing management
        self.trailing_manager = UnifiedTrailingManager(config.get('exit_strategy', {}))
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self._latest_signals: Dict[str, Any] = {}
        self._latest_market_snapshot: Optional[Dict[str, Any]] = None

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
            
            # Initialize comprehensive monitoring
            await self._initialize_comprehensive_monitoring()
            
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
        """Initialize signal generators with NAS/TAS enhancement."""
        try:
            tprint_info("🔄 Initializing enhanced signal generators with NAS/TAS...")
            
            # Initialize enhanced Analyst signal generator with NAS integration
            analyst_signal_config = self.config.get('analyst_signals', {})
            # Enable NAS enhancement by default
            analyst_signal_config.update({
                'enable_nas_enhancement': True,
                'nas_confidence_threshold': 0.7,
                'nas_timeframe': '5m',
                'regime_timeframe': '15m'
            })
            self.analyst_signal_generator = AnalystSignalGenerator(analyst_signal_config)
            
            # Load NAS models if available
            nas_models = self.config.get('nas_models', {})
            await self.analyst_signal_generator.initialize(self.analyst, nas_models=nas_models)
            
            # Initialize enhanced Tactician signal generator with TAS integration
            tactician_signal_config = self.config.get('tactician_signals', {})
            # Enable TAS enhancement by default
            tactician_signal_config.update({
                'enable_tas_enhancement': True,
                'tas_confidence_threshold': 0.7,
                'tas_timeframe': '1m'
            })
            self.tactician_signal_generator = TacticianSignalGenerator(tactician_signal_config)
            
            # Load TAS models if available
            tas_models = self.config.get('tas_models', {})
            await self.tactician_signal_generator.initialize(self.tactician, tas_models=tas_models)
            
            # Initialize signal combiner
            signal_combiner_config = self.config.get('signal_combiner', {})
            self.signal_combiner = SignalCombiner(signal_combiner_config)
            await self.signal_combiner.initialize()
            
            tprint_success("✅ Enhanced signal generators with NAS/TAS initialized")
            
        except Exception as e:
            tprint_error(f"❌ Enhanced signal generator initialization failed: {e}")
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
    
    async def _initialize_comprehensive_monitoring(self):
        """Initialize comprehensive trade monitoring."""
        try:
            tprint_info("🔄 Initializing comprehensive trade monitoring...")
            
            # Initialize comprehensive monitoring with configuration
            monitoring_config = {
                'enable_explanations': True,
                'enable_real_time_export': True,
                'export_directory': 'trading_reports',
                'max_trades_in_memory': 5000
            }
            
            # Initialize if not already initialized
            if not comprehensive_trade_monitor.is_initialized:
                success = await comprehensive_trade_monitor.initialize()
                if not success:
                    tprint_warning("⚠️ Failed to initialize comprehensive monitoring")
                else:
                    tprint_success("✅ Comprehensive monitoring initialized")
            else:
                tprint_success("✅ Comprehensive monitoring already initialized")
            
        except Exception as e:
            tprint_error(f"❌ Comprehensive monitoring initialization failed: {e}")
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
                
                # Record daily summary if this is end of day
                await self._record_daily_summary_if_needed()
            
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
        polling_interval = self.config.get('trading_interval', 30)
        while self.status == OrchestratorStatus.RUNNING:
            try:
                market_snapshot = await self._get_market_snapshot()
                if market_snapshot:
                    await self._evaluate_trailing_positions(market_snapshot)

                decision = await self._generate_trading_decision(market_snapshot)
                if decision and market_snapshot:
                    await self._execute_trading_decision(decision, market_snapshot)

                await asyncio.sleep(polling_interval)
            except Exception as e:
                self.logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(5)

    async def _generate_trading_decision(
        self, market_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[TradingDecision]:
        """Generate trading decision based on all components."""
        try:
            if market_snapshot is None:
                market_snapshot = await self._get_market_snapshot()
            if not market_snapshot:
                return None

            market_data: pd.DataFrame = market_snapshot['market_data']
            self._latest_market_snapshot = market_snapshot

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

            self._latest_signals = {
                'analyst': analyst_signal,
                'tactician': tactician_signal,
            }

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

    async def _execute_trading_decision(
        self,
        decision: TradingDecision,
        market_snapshot: Dict[str, Any],
    ):
        """Execute trading decision with comprehensive monitoring."""
        try:
            tprint_info(f"🔄 Executing {decision.action} order for {decision.symbol}")

            # Cross-asset trade gate: serialize trade execution across symbols
            gate_acquired = False
            if self.trade_gate is not None:
                gate_acquired = await self.trade_gate.acquire(self.symbol, decision)
                if not gate_acquired:
                    await self._trigger_trade_callbacks(decision, event="skipped_gate")
                    tprint_warning("⚠️ Trade skipped due to global gate or risk checks")
                    return
            
            feature_bundle: TrailingFeatureBundle = market_snapshot['feature_bundle']

            # Prepare comprehensive trade data
            trade_data = {
                'symbol': decision.symbol,
                'action': decision.action,
                'quantity': decision.quantity,
                'price': decision.price,
                'confidence': decision.confidence,
                'trading_mode': self.trading_mode.value,
                'exchange': self.exchange,
                'analyst_signal': decision.analyst_signal.__dict__ if decision.analyst_signal else None,
                'tactician_signal': decision.tactician_signal.__dict__ if decision.tactician_signal else None,
                'combined_signal': decision.combined_signal,
                'risk_metrics': decision.risk_metrics,
                'regime_data': decision.metadata.get('regime_data', {}),
                'position_sizing': {
                    'recommended_size': decision.quantity,
                    'leverage': 1.0,  # Default for paper trading
                    'risk_per_trade': 0.02
                }
            }
            
            # Prepare models used information
            models_used = {
                'analyst_model': {
                    'type': 'analyst',
                    'prediction': decision.confidence,
                    'confidence': decision.confidence,
                    'weight': 0.6,
                    'version': '1.0'
                },
                'tactician_model': {
                    'type': 'tactician',
                    'prediction': decision.confidence,
                    'confidence': decision.confidence,
                    'weight': 0.4,
                    'version': '1.0'
                }
            }
            
            # Get recent market data for context
            market_data = market_snapshot['market_data'] if market_snapshot else None
            
            # Record comprehensive trade decision
            trade_id = await record_detailed_trade(trade_data, models_used, market_data)
            
            # Store decision with trade ID for outcome tracking
            decision.metadata['trade_id'] = trade_id
            self.trading_decisions.append(decision)

            # Inform trade gate of active trade id
            if self.trade_gate is not None and trade_id:
                try:
                    self.trade_gate.set_active_trade_id(trade_id)
                except Exception:
                    pass

            # Trigger pre-execution callback
            await self._trigger_trade_callbacks(decision, event="pre_execute")
            
            # Simulate execution (in real trading, this would place actual orders)
            execution_success = await self._simulate_order_execution(decision)
            
            # Update trade outcome
            if execution_success:
                outcome_data = {
                    'exit_price': decision.price,  # For immediate execution
                    'pnl_absolute': 0.0,  # Will be updated when position is closed
                    'pnl_percentage': 0.0,
                    'duration_minutes': 0.0,
                    'execution_quality': 0.95,  # High quality for successful execution
                    'slippage': 0.001,  # 0.1% slippage
                    'commission': decision.quantity * decision.price * 0.001,  # 0.1% commission
                    'timing_quality': decision.confidence  # Use confidence as timing quality
                }
                
                await update_trade_outcome(trade_id, outcome_data)

                if self.current_session:
                    self.current_session.total_trades += 1

                self.performance_metrics['total_trades'] += 1

                tprint_success(f"✅ {decision.action} order executed for {decision.symbol} (ID: {trade_id})")

                self._update_active_positions(decision, trade_id, feature_bundle)
            else:
                tprint_error(f"❌ Failed to execute {decision.action} order for {decision.symbol}")
                if self.current_session:
                    self.current_session.failed_trades += 1
                self.performance_metrics['failed_trades'] += 1

            # Trigger post-execution callback
            await self._trigger_trade_callbacks(decision, event="post_execute", success=execution_success)
            
        except Exception as e:
            self.logger.error(f"❌ Trading decision execution failed: {e}")
            if self.current_session:
                self.current_session.failed_trades += 1
            self.performance_metrics['failed_trades'] += 1
        finally:
            # Ensure gate is released
            if self.trade_gate is not None:
                try:
                    await self.trade_gate.release(decision.metadata.get('trade_id'))
                except Exception:
                    pass

    async def _get_market_snapshot(self) -> Optional[Dict[str, Any]]:
        """Collect latest market data and derived trailing features."""

        try:
            if not self.data_collector:
                return None

            market_data = self.data_collector.get_processed_data_df(n=200)
            if market_data is None or market_data.empty:
                return None

            feature_bundle = prepare_trailing_feature_bundle(market_data)
            if feature_bundle is None:
                return None

            return {
                'market_data': market_data,
                'feature_bundle': feature_bundle,
            }
        except Exception as exc:
            self.logger.error(f"❌ Failed to collect market snapshot: {exc}")
            return None

    async def _evaluate_trailing_positions(self, market_snapshot: Dict[str, Any]) -> None:
        """Evaluate trailing actions for all active positions."""

        if not self.active_positions:
            return

        feature_bundle: TrailingFeatureBundle = market_snapshot.get('feature_bundle')
        if feature_bundle is None:
            return

        tact_metrics = feature_bundle.tactician
        price = feature_bundle.current_price
        atr = tact_metrics.get('atr', 0.0)
        sigma = tact_metrics.get('sigma', 0.0)
        momentum = tact_metrics.get('momentum', 0.0)
        rsi = tact_metrics.get('rsi', 50.0)
        vol_slope = tact_metrics.get('vol_slope', 0.0)

        for position_id, position in list(self.active_positions.items()):
            ml_context = self._build_ml_context(position)
            decision = self.trailing_manager.evaluate_position(
                position_id,
                price=price,
                atr=atr,
                sigma=sigma,
                momentum=momentum,
                rsi=rsi,
                vol_slope=vol_slope,
                timestamp=feature_bundle.timestamp,
                ml_context=ml_context,
            )
            self._apply_trailing_decision(position_id, position, decision)

    def _update_active_positions(
        self,
        decision: TradingDecision,
        trade_id: str,
        feature_bundle: TrailingFeatureBundle,
    ) -> None:
        action = decision.action.lower()

        if action in {"close", "exit"}:
            self._close_position_by_symbol(decision.symbol, reason=action)
            return

        if action not in {"buy", "sell"}:
            return

        side = "long" if action == "buy" else "short"
        existing = self._find_position_for_symbol(decision.symbol)
        if existing:
            position_id, position = existing
            if position["side"] != side:
                self._close_position(position_id, reason="opposite_signal")
            else:
                # Treat as scaling into existing position
                position["quantity"] += decision.quantity
                state = self.trailing_manager.positions.get(position_id)
                if state:
                    state.quantity = position["quantity"]
                return

        self._open_position(decision, trade_id, feature_bundle, side)

    def _open_position(
        self,
        decision: TradingDecision,
        trade_id: str,
        feature_bundle: TrailingFeatureBundle,
        side: str,
    ) -> None:
        tact_metrics = feature_bundle.tactician
        entry_atr = tact_metrics.get('atr', 0.0)
        entry_sigma = tact_metrics.get('sigma', 0.0)
        if entry_atr <= 0:
            self.logger.warning("⚠️ Skipping trailing registration due to invalid ATR")
            return

        ml_entry = {
            'analyst_confidence': getattr(decision.analyst_signal, 'confidence_score', None)
            if decision.analyst_signal
            else None,
            'tactician_confidence': getattr(decision.tactician_signal, 'confidence_score', None)
            if decision.tactician_signal
            else None,
            'tactician_momentum': (decision.tactician_signal.risk_metrics.get('momentum')
                                   if decision.tactician_signal and decision.tactician_signal.risk_metrics
                                   else None),
            'regime': getattr(decision.analyst_signal, 'regime_id', None)
            if decision.analyst_signal
            else None,
        }

        state = self.trailing_manager.register_position(
            trade_id,
            side=side,
            entry_price=decision.price,
            entry_time=decision.timestamp,
            quantity=decision.quantity,
            entry_atr=entry_atr,
            entry_sigma=entry_sigma,
            bar_duration=feature_bundle.bar_seconds,
            metadata={'symbol': decision.symbol, 'ml_entry': ml_entry},
        )

        self.active_positions[trade_id] = {
            'symbol': decision.symbol,
            'side': side,
            'quantity': decision.quantity,
            'entry_price': decision.price,
            'entry_time': decision.timestamp,
            'ml_entry': ml_entry,
            'trailing_state': state,
        }

    def _close_position(self, position_id: str, reason: str) -> None:
        position = self.active_positions.pop(position_id, None)
        if not position:
            return

        self.trailing_manager.remove_position(position_id)
        self.logger.info(
            "🚪 Closed position %s (%s) due to %s",
            position_id,
            position['symbol'],
            reason,
        )

    def _close_position_by_symbol(self, symbol: str, reason: str) -> None:
        existing = self._find_position_for_symbol(symbol)
        if existing:
            self._close_position(existing[0], reason)

    def _find_position_for_symbol(
        self, symbol: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        for position_id, position in self.active_positions.items():
            if position.get('symbol') == symbol:
                return position_id, position
        return None

    def _build_ml_context(self, position: Dict[str, Any]) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            'entry': position.get('ml_entry', {}),
        }

        analyst_signal = self._latest_signals.get('analyst')
        if analyst_signal:
            context['analyst_confidence'] = getattr(
                analyst_signal, 'confidence_score', None
            )
            entry_regime = position.get('ml_entry', {}).get('regime')
            current_regime = getattr(analyst_signal, 'regime_id', None)
            if entry_regime is not None and current_regime is not None:
                context['regime_changed'] = entry_regime != current_regime

        tactician_signal = self._latest_signals.get('tactician')
        if tactician_signal:
            context['tactician_confidence'] = getattr(
                tactician_signal, 'confidence_score', None
            )
            if tactician_signal.risk_metrics:
                context['tactician_momentum'] = tactician_signal.risk_metrics.get('momentum')

        return context

    def _apply_trailing_decision(
        self,
        position_id: str,
        position: Dict[str, Any],
        decision: TrailingDecision,
    ) -> None:
        if decision.action == TrailingAction.NONE:
            return

        if decision.action == TrailingAction.MOVE_STOP and decision.stop_price is not None:
            position['trailing_stop'] = decision.stop_price
            self.logger.info(
                "📉 Updated trailing stop for %s to %.4f (%s)",
                position_id,
                decision.stop_price,
                decision.reason,
            )
            return

        if decision.action == TrailingAction.PARTIAL_EXIT:
            fraction = decision.exit_fraction or self.trailing_manager.config['partial_take_fraction']
            position['quantity'] = max(0.0, position['quantity'] * (1.0 - fraction))
            state = self.trailing_manager.positions.get(position_id)
            if state:
                state.quantity = position['quantity']
            self.logger.info(
                "💰 Partial exit %.2f%% for %s (%s)",
                fraction * 100,
                position_id,
                decision.reason,
            )
            if position['quantity'] <= 1e-6:
                self._close_position(position_id, reason=decision.reason or 'partial_exit_complete')
            return

        if decision.action == TrailingAction.FULL_EXIT:
            self.logger.info(
                "🛑 Trailing exit triggered for %s (%s)",
                position_id,
                decision.reason,
            )
            self._close_position(position_id, reason=decision.reason or 'trailing_exit')

    async def _simulate_order_execution(self, decision: TradingDecision) -> bool:
        """Simulate order execution (replace with real execution in live trading)."""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Simulate execution success (95% success rate)
            import random
            return random.random() > 0.05
            
        except Exception as e:
            self.logger.error(f"❌ Order execution simulation failed: {e}")
            return False

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
            'recent_decisions': len(self.trading_decisions),
            'monitoring_stats': comprehensive_trade_monitor.get_monitor_stats() if comprehensive_trade_monitor.is_initialized else None
        }
    
    async def generate_live_dashboard(self) -> Dict[str, Any]:
        """Generate live trading dashboard."""
        try:
            if not comprehensive_trade_monitor.is_initialized:
                tprint_warning("⚠️ Comprehensive monitoring not initialized")
                return {}
            
            # Get completed trades
            completed_trades = comprehensive_trade_monitor.completed_trades
            
            # Get active trades
            active_trades = comprehensive_trade_monitor.active_trades
            
            # Get session metrics
            session_metrics = comprehensive_trade_monitor.current_session
            
            # Generate dashboard
            dashboard = await create_trading_dashboard(
                trades=completed_trades,
                session_metrics=session_metrics,
                active_trades=active_trades
            )
            
            return dashboard
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate live dashboard: {e}")
            return {}
    
    async def generate_performance_report(self, report_type: str = "session") -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            if not comprehensive_trade_monitor.is_initialized:
                tprint_warning("⚠️ Comprehensive monitoring not initialized")
                return {}
            
            # Get completed trades
            completed_trades = comprehensive_trade_monitor.completed_trades
            
            # Get session metrics
            session_metrics = comprehensive_trade_monitor.current_session
            
            # Generate report
            report = await generate_trading_report(
                trades=completed_trades,
                session_metrics=session_metrics,
                report_name=f"orchestrator_{report_type}_report"
            )
            
            return report
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate performance report: {e}")
            return {}
    
    async def _record_daily_summary_if_needed(self):
        """Record daily summary if this is the last session of the day."""
        try:
            current_time = datetime.now()
            
            # Check if this is late in the day (after 6 PM) or if explicitly requested
            if current_time.hour >= 18 or self.config.get('always_record_daily', False):
                tprint_info("📝 Recording daily trading summary...")
                
                # Get all completed trades for today
                today = current_time.date()
                todays_trades = [
                    t for t in comprehensive_trade_monitor.completed_trades
                    if t.timestamp.date() == today
                ]
                
                # Get today's sessions
                todays_sessions = [self.current_session] if self.current_session else []
                
                # Record daily summary
                success = await record_daily_trading_summary(
                    trades=todays_trades,
                    sessions=todays_sessions,
                    target_date=today
                )
                
                if success:
                    tprint_success(f"✅ Daily summary recorded for {today}")
                else:
                    tprint_warning(f"⚠️ Failed to record daily summary for {today}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to record daily summary: {e}")
    
    async def force_record_daily_summary(self, target_date: Optional[date] = None) -> bool:
        """Force recording of daily summary for specified date."""
        try:
            record_date = target_date or datetime.now().date()
            
            tprint_info(f"📝 Force recording daily summary for {record_date}")
            
            # Get trades for the specified date
            target_trades = [
                t for t in comprehensive_trade_monitor.completed_trades
                if t.timestamp.date() == record_date
            ]
            
            # Get sessions for the specified date
            target_sessions = []
            if self.current_session and self.current_session.start_time.date() == record_date:
                target_sessions = [self.current_session]
            
            # Record daily summary
            success = await record_daily_trading_summary(
                trades=target_trades,
                sessions=target_sessions,
                target_date=record_date
            )
            
            if success:
                tprint_success(f"✅ Daily summary force-recorded for {record_date}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Failed to force record daily summary: {e}")
            return False

    def get_trading_decisions(self, n: int = 100) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return self.trading_decisions[-n:] if len(self.trading_decisions) >= n else self.trading_decisions.copy()

    def add_trade_decision_callback(self, callback: Any) -> None:
        """Register a callback to observe trade decision lifecycle events.
        Callback signature: async|sync callback(decision: TradingDecision, event: str, **kwargs)
        Events: 'skipped_gate' | 'pre_execute' | 'post_execute'
        """
        self._on_trade_decision_callbacks.append(callback)

    async def _trigger_trade_callbacks(self, decision: TradingDecision, event: str, **kwargs):
        for cb in self._on_trade_decision_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(decision, event=event, **kwargs)
                else:
                    cb(decision, event=event, **kwargs)
            except Exception:
                # Swallow to not interrupt trading flow
                pass

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