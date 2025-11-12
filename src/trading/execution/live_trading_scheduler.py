"""
Live Trading Scheduler - Coordinates HMM, Analyst, and Tactician Execution

This module provides a comprehensive live trading scheduler that coordinates
the execution of the three-tier model system with different frequencies:

- HMM (1h timeframe): Runs every 15 minutes with partial-bar nowcasting
- Analyst (5m timeframe): Runs every 2 minutes
- Tactician (15m timeframe): Runs every 3 minutes

The scheduler ensures proper data flow between models and maintains
the hierarchical decision-making process. Now includes partial-bar nowcasting
to ensure regime evaluation always uses complete 1-hour bars.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

# Import partial-bar nowcasting
try:
    from .partial_bar_nowcasting import PartialBarNowcaster, create_partial_bar_nowcaster
except ImportError:
    # Fallback if partial_bar_nowcasting is not available
    tprint_warning("⚠️ Partial-bar nowcasting not available, using mock implementation")
    PartialBarNowcaster = None
    create_partial_bar_nowcaster = None

logger = system_logger.getChild('LiveTradingScheduler')

class ModelType(Enum):
    """Model types in the trading system."""
    HMM = "hmm"
    ANALYST = "analyst"
    TACTICIAN = "tactician"

class ExecutionStatus(Enum):
    """Execution status for models."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ModelConfig:
    """Configuration for each model in the trading system."""
    model_type: ModelType
    timeframe: str
    execution_interval_seconds: int
    enabled: bool = True
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Result of model execution."""
    model_type: ModelType
    execution_time: datetime
    status: ExecutionStatus
    execution_duration: float
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class LiveTradingScheduler:
    """
    Live Trading Scheduler for coordinating HMM, Analyst, and Tactician execution.

    Features:
    - HMM (1h timeframe): Runs every 15 minutes
    - Analyst (5m timeframe): Runs every 2 minutes
    - Tactician (15m timeframe): Runs every 3 minutes
    - Proper data flow between models
    - Hierarchical decision-making process
    - Error handling and recovery
    - Performance monitoring
    """

    def __init__(self, symbol: str = "ETH", exchange: str = "binance"):
        """
        tprint(f"🚀 ModelType.__init__: symbol={symbol}, exchange={exchange}", "INFO")

        Initialize the live trading scheduler.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
        """
        self.symbol = symbol
        self.exchange = exchange
        self.logger = logger.getChild(f'{symbol}_{exchange}')

        # Initialize partial-bar nowcaster for HMM
        if create_partial_bar_nowcaster:
            self.nowcaster = create_partial_bar_nowcaster(
                base_timeframe="1h",
                evaluation_interval=15 * 60,  # 15 minutes
                min_bar_completion=0.25,     # 25% minimum completion
                max_bar_completion=0.95      # 95% maximum completion
            )
        else:
            self.nowcaster = None
            tprint_warning("⚠️ Partial-bar nowcaster not available, HMM will use mock data")

        # Model configurations
        self.model_configs = {
            ModelType.HMM: ModelConfig(
                model_type=ModelType.HMM,
                timeframe="1h",
                execution_interval_seconds=15 * 60,  # 15 minutes
                custom_params={
                    'n_regimes': 20,
                    'n_features': 100,
                    'base_models': ['catboost', 'elastic_net'],
                    'meta_learner': 'xgboost',
                    'use_nowcasting': True  # Enable partial-bar nowcasting
                }
            ),
            ModelType.ANALYST: ModelConfig(
                model_type=ModelType.ANALYST,
                timeframe="5m",
                execution_interval_seconds=2 * 60,  # 2 minutes
                custom_params={
                    'n_features': 300,
                    'base_models': ['lgbm', 'tcn', 'catboost'],
                    'meta_learner': 'stacker_lgbm_calibrated',
                    'per_regime_training': True
                }
            ),
            ModelType.TACTICIAN: ModelConfig(
                model_type=ModelType.TACTICIAN,
                timeframe="15m",
                execution_interval_seconds=3 * 60,  # 3 minutes (matches tactician_base_config.yaml)
                custom_params={
                    'n_features': 100,
                    'base_models': ['lgbm', 'catboost', 'extratrees', 'gru'],
                    'meta_learner': 'stacker_lgbm_calibrated_gating',
                    'target_price_change': 0.005  # 0.5%
                }
            )
        }

        # State management
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.execution_history: List[ExecutionResult] = []

        # Model instances and data
        self.hmm_models = None
        self.analyst_models = None
        self.tactician_models = None
        self.hmm_data = None
        self.analyst_data = None
        self.tactician_data = None

        # Callbacks
        self.on_execution_callbacks: List[Callable[[ExecutionResult], None]] = []
        self.on_error_callbacks: List[Callable[[Exception], None]] = []

        tprint_info(f"🚀 Initialized Live Trading Scheduler for {symbol} on {exchange}")
        tprint_info("📊 Model Configuration:")
        for model_type, config in self.model_configs.items():
            tprint_info(f"  - {model_type.value.upper()}: {config.timeframe} timeframe, every {config.execution_interval_seconds}s")

    async def start_scheduler(self) -> bool:
        """Start the live trading scheduler."""
        tprint(f"🚀 ModelType.start_scheduler: Starting", "INFO")

        if self.is_running:
            tprint_warning("⚠️ Scheduler already running")
            return False

        try:
            tprint_info("🚀 Starting Live Trading Scheduler...")

            # Initialize partial-bar nowcaster
            if self.nowcaster:
                try:
                    await self.nowcaster.initialize()
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to initialize nowcaster: {e}")
                    self.nowcaster = None
            else:
                tprint_warning("⚠️ Skipping nowcaster initialization - not available")

            # Initialize models
            try:
                await self._initialize_models()
            except Exception as e:
                tprint_error(f"❌ Failed to initialize models: {e}")
                return False

            # Set initial execution times
            self._schedule_initial_executions()

            # Start scheduler loop
            self.is_running = True
            self.start_time = datetime.now()

            # Start the main scheduler loop
            asyncio.create_task(self._scheduler_loop())

            tprint_success("✅ Live Trading Scheduler started successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to start scheduler: {e}")
            self.is_running = False
            return False

    async def stop_scheduler(self) -> bool:
        """Stop the live trading scheduler."""
        tprint(f"🚀 ModelType.stop_scheduler: Starting", "INFO")

        if not self.is_running:
            return True

        tprint_info("🛑 Stopping Live Trading Scheduler...")
        self.is_running = False

        # Wait for cleanup
        await asyncio.sleep(0.1)

        tprint_success("✅ Live Trading Scheduler stopped")
        return True

    async def _initialize_models(self) -> None:
        """Initialize all models for live trading."""
        try:
            tprint_info("🔄 Initializing models for live trading...")

            # Initialize HMM models
            await self._initialize_hmm_models()

            # Initialize Analyst models
            await self._initialize_analyst_models()

            # Initialize Tactician models
            await self._initialize_tactician_models()

            tprint_success("✅ All models initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Model initialization failed: {e}")
            raise

    async def _initialize_hmm_models(self) -> None:
        """Initialize HMM models for regime detection."""
        try:
            # Create a mock HMM implementation since the actual training module is not available
            class MockHMMPipeline:
                def __init__(self, n_regimes=20, n_features=100):
                    tprint(f"🚀 ModelType.__init__: n_regimes={n_regimes}, n_features={n_features}", "INFO")

                    self.n_features = n_features
                    self.is_initialized = True
                
                async def predict(self, data: Any) -> np.ndarray:
                    """Mock prediction method."""
                    tprint(f"🚀 ModelType.predict: data={data}", "INFO")

                    return np.random.randint(0, self.n_regimes, len(data))
                
                async def predict_proba(self, data: Any) -> np.ndarray:
                    """Mock probability prediction."""
                    tprint(f"🚀 ModelType.predict_proba: data={data}", "INFO")

                    return np.random.rand(len(data), self.n_regimes)

            config = self.model_configs[ModelType.HMM]
            self.hmm_models = MockHMMPipeline(
                n_regimes=config.custom_params['n_regimes'],
                n_features=config.custom_params['n_features']
            )

            tprint_success("✅ HMM models initialized (mock implementation)")

        except Exception as e:
            tprint_error(f"❌ HMM model initialization failed: {e}")
            raise

    async def _initialize_analyst_models(self) -> None:
        """Initialize Analyst models for trade decisions."""
        try:
            # Try to import the actual module, fallback to mock if not available
            try:
                from src.training.steps.model_training.analyst_ensemble_training import AnalystEnsembleTrainingStep
                self.analyst_models = AnalystEnsembleTrainingStep()
                tprint_success("✅ Analyst models initialized")
            except ImportError:
                # Create mock implementation
                class MockAnalystEnsemble:
                    def __init__(self):
                        tprint(f"🚀 ModelType.__init__: Starting", "INFO")

                    
                    async def predict(self, data: Any) -> np.ndarray:
                        """Mock prediction method."""
                        tprint(f"🚀 ModelType.predict: data={data}", "INFO")

                        return np.random.choice([0, 1], len(data), p=[0.7, 0.3])
                    
                    async def predict_proba(self, data: Any) -> np.ndarray:
                        """Mock probability prediction."""
                        tprint(f"🚀 ModelType.predict_proba: data={data}", "INFO")

                        return np.random.rand(len(data), 2)
                
                self.analyst_models = MockAnalystEnsemble()
                tprint_success("✅ Analyst models initialized (mock implementation)")

        except Exception as e:
            tprint_error(f"❌ Analyst model initialization failed: {e}")
            raise

    async def _initialize_tactician_models(self) -> None:
        """Initialize Tactician models for timing decisions."""
        try:
            # Try to import the actual module, fallback to mock if not available
            try:
                from src.training.steps.model_training.tactician_ensemble_training import TacticianEnsembleTrainingStep
                self.tactician_models = TacticianEnsembleTrainingStep()
                tprint_success("✅ Tactician models initialized")
            except ImportError:
                # Create mock implementation
                class MockTacticianEnsemble:
                    def __init__(self):
                        tprint(f"🚀 ModelType.__init__: Starting", "INFO")

                    
                    async def predict(self, data: Any) -> np.ndarray:
                        """Mock prediction method."""
                        tprint(f"🚀 ModelType.predict: data={data}", "INFO")

                        return np.random.choice([0, 1], len(data), p=[0.8, 0.2])
                    
                    async def predict_proba(self, data: Any) -> np.ndarray:
                        """Mock probability prediction."""
                        tprint(f"🚀 ModelType.predict_proba: data={data}", "INFO")

                        return np.random.rand(len(data), 2)
                
                self.tactician_models = MockTacticianEnsemble()
                tprint_success("✅ Tactician models initialized (mock implementation)")

        except Exception as e:
            tprint_error(f"❌ Tactician model initialization failed: {e}")
            raise

    def _schedule_initial_executions(self) -> None:
        """Schedule initial execution times for all models."""
        now = datetime.now()

        for model_type, config in self.model_configs.items():
            if config.enabled:
                # Schedule first execution immediately
                config.next_execution = now
                tprint_info(f"📅 {model_type.value.upper()} scheduled for immediate execution")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                current_time = datetime.now()

                # Check which models need to be executed
                models_to_execute = []
                for model_type, config in self.model_configs.items():
                    if (config.enabled and
                        config.next_execution and
                        current_time >= config.next_execution):

                        # For HMM, check if regime evaluation should occur based on bar completion
                        if model_type == ModelType.HMM and config.custom_params.get('use_nowcasting', False) and self.nowcaster:
                            should_evaluate = await self.nowcaster.should_evaluate_regime(current_time)
                            if not should_evaluate:
                                tprint_debug("⏳ HMM evaluation skipped - insufficient bar completion")
                                continue

                        models_to_execute.append(model_type)

                # Execute models in order of priority (HMM -> Analyst -> Tactician)
                execution_order = [ModelType.HMM, ModelType.ANALYST, ModelType.TACTICIAN]
                for model_type in execution_order:
                    if model_type in models_to_execute:
                        await self._execute_model(model_type)

                # Brief pause to prevent excessive CPU usage
                await asyncio.sleep(1)

            except Exception as e:
                tprint_error(f"❌ Scheduler loop error: {e}")
                await self._handle_error(e)
                await asyncio.sleep(5)  # Wait before retrying

    async def _execute_model(self, model_type: ModelType) -> None:
        """Execute a specific model with error recovery."""
        config = self.model_configs[model_type]
        execution_start = time.time()
        
        # Track consecutive failures for exponential backoff
        max_consecutive_failures = config.failure_count - config.success_count
        
        try:
            tprint_info(f"🔄 Executing {model_type.value.upper()} model...")

            # Update status
            config.last_execution = datetime.now()
            config.execution_count += 1

            # Execute based on model type
            if model_type == ModelType.HMM:
                result_data = await self._execute_hmm()
            elif model_type == ModelType.ANALYST:
                result_data = await self._execute_analyst()
            elif model_type == ModelType.TACTICIAN:
                result_data = await self._execute_tactician()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Update execution time
            execution_duration = time.time() - execution_start
            config.avg_execution_time = (
                (config.avg_execution_time * (config.execution_count - 1) + execution_duration)
                / config.execution_count
            )

            # Reset failure count on success
            if config.failure_count > 0:
                config.failure_count = max(0, config.failure_count - 1)

            # Schedule next execution with normal interval
            config.next_execution = datetime.now() + timedelta(seconds=config.execution_interval_seconds)

            # Create execution result
            result = ExecutionResult(
                model_type=model_type,
                execution_time=config.last_execution,
                status=ExecutionStatus.COMPLETED,
                execution_duration=execution_duration,
                result_data=result_data,
                metrics={
                    'execution_count': config.execution_count,
                    'avg_execution_time': config.avg_execution_time,
                    'next_execution': config.next_execution
                }
            )

            # Update success count
            config.success_count += 1

            # Store result
            self.execution_history.append(result)

            # Trigger callbacks
            await self._trigger_execution_callbacks(result)

            tprint_success(f"✅ {model_type.value.upper()} execution completed in {execution_duration:.2f}s")

        except Exception as e:
            execution_duration = time.time() - execution_start
            config.failure_count += 1
            
            # Calculate backoff based on consecutive failures
            consecutive_failures = config.failure_count - config.success_count
            backoff_multiplier = min(2 ** consecutive_failures, 32)  # Max 32x backoff
            backoff_seconds = config.execution_interval_seconds * backoff_multiplier
            
            # Create error result
            result = ExecutionResult(
                model_type=model_type,
                execution_time=datetime.now(),
                status=ExecutionStatus.FAILED,
                execution_duration=execution_duration,
                error_message=str(e),
                metrics={
                    'execution_count': config.execution_count,
                    'failure_count': config.failure_count,
                    'consecutive_failures': consecutive_failures,
                    'backoff_multiplier': backoff_multiplier
                }
            )

            # Store result
            self.execution_history.append(result)

            # Trigger error callbacks
            await self._trigger_error_callbacks(e)

            tprint_error(f"❌ {model_type.value.upper()} execution failed: {e}")
            tprint_warning(f"⚠️ Applying {backoff_multiplier}x backoff: next execution in {backoff_seconds}s")

            # Schedule next execution with exponential backoff
            config.next_execution = datetime.now() + timedelta(seconds=backoff_seconds)
            
            # If too many failures, disable model temporarily
            if consecutive_failures >= 5:
                tprint_error(f"❌ Too many consecutive failures ({consecutive_failures}), disabling {model_type.value.upper()} temporarily")
                config.enabled = False
                config.next_execution = datetime.now() + timedelta(minutes=15)  # Re-enable after 15 minutes

    async def _execute_hmm(self) -> Dict[str, Any]:
        """Execute HMM model for regime detection with partial-bar nowcasting."""
        try:
            tprint_info("🔮 Executing HMM with partial-bar nowcasting...")

            if self.nowcaster:
                # Get complete hourly bars using nowcasting
                complete_bars = await self.nowcaster.get_complete_hourly_bars(n_bars=24)

                if len(complete_bars) == 0:
                    tprint_warning("⚠️ No complete bars available for HMM evaluation")
                    return {
                        'regime_states': [],
                        'regime_probabilities': [],
                        'regime_confidence': [],
                        'n_regimes': 0,
                        'n_features': 0,
                        'execution_time': datetime.now().isoformat(),
                        'error': 'No complete bars available'
                    }

                # Create bar split for this evaluation
                bar_split = await self.nowcaster.create_bar_split()

                # This would integrate with your HMM training pipeline
                # For now, return mock data with nowcasting information
                result = {
                    'regime_states': np.random.randint(0, 20, len(complete_bars)).tolist(),
                    'regime_probabilities': np.random.rand(len(complete_bars), 20).tolist(),
                    'regime_confidence': np.random.rand(len(complete_bars)).tolist(),
                    'n_regimes': 20,
                    'n_features': 100,
                    'execution_time': datetime.now().isoformat(),
                    'nowcasting_info': {
                        'bar_completion': bar_split.split_ratio,
                        'complete_bars_count': len(complete_bars),
                        'nowcasted_bars_count': len(complete_bars[complete_bars.get('is_nowcasted', False)]),
                        'bar_split_time': bar_split.end_time.isoformat()
                    }
                }

                # Update evaluation time
                await self.nowcaster.update_evaluation_time()

                tprint_success(f"✅ HMM execution completed with {len(complete_bars)} complete bars")
            else:
                # Fallback to mock data without nowcasting
                tprint_warning("⚠️ Using mock HMM data (nowcaster not available)")
                result = {
                    'regime_states': np.random.randint(0, 20, 24).tolist(),
                    'regime_probabilities': np.random.rand(24, 20).tolist(),
                    'regime_confidence': np.random.rand(24).tolist(),
                    'n_regimes': 20,
                    'n_features': 100,
                    'execution_time': datetime.now().isoformat(),
                    'nowcasting_info': {
                        'bar_completion': 1.0,
                        'complete_bars_count': 24,
                        'nowcasted_bars_count': 0,
                        'bar_split_time': datetime.now().isoformat()
                    }
                }

            # Store HMM data for other models
            self.hmm_data = result

            return result

        except Exception as e:
            tprint_error(f"❌ HMM execution failed: {e}")
            raise

    async def _execute_analyst(self) -> Dict[str, Any]:
        """Execute Analyst model for trade decisions."""
        try:
            # This would integrate with your Analyst training pipeline
            # For now, return mock data
            result = {
                'trade_signals': np.random.choice([0, 1], 100, p=[0.7, 0.3]).tolist(),
                'confidence_scores': np.random.rand(100).tolist(),
                'green_light_periods': np.random.choice([True, False], 100, p=[0.3, 0.7]).tolist(),
                'n_features': 300,
                'execution_time': datetime.now().isoformat()
            }

            # Store Analyst data for Tactician
            self.analyst_data = result

            return result

        except Exception as e:
            tprint_error(f"❌ Analyst execution failed: {e}")
            raise

    async def _execute_tactician(self) -> Dict[str, Any]:
        """Execute Tactician model for timing decisions."""
        try:
            # This would integrate with your Tactician training pipeline
            # For now, return mock data
            result = {
                'timing_signals': np.random.choice([0, 1], 100, p=[0.8, 0.2]).tolist(),
                'price_change_predictions': np.random.normal(0, 0.01, 100).tolist(),
                'confidence_scores': np.random.rand(100).tolist(),
                'n_features': 50,
                'execution_time': datetime.now().isoformat()
            }

            # Store Tactician data
            self.tactician_data = result

            return result

        except Exception as e:
            tprint_error(f"❌ Tactician execution failed: {e}")
            raise

    async def _trigger_execution_callbacks(self, result: ExecutionResult) -> None:
        """Trigger execution callbacks."""
        for callback in self.on_execution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                tprint_warning(f"⚠️ Execution callback failed: {e}")

    async def _trigger_error_callbacks(self, error: Exception) -> None:
        """Trigger error callbacks."""
        for callback in self.on_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                tprint_warning(f"⚠️ Error callback failed: {e}")

    async def _handle_error(self, error: Exception) -> None:
        """Handle scheduler errors."""
        tprint_error(f"❌ Scheduler error: {error}")
        await self._trigger_error_callbacks(error)

    def add_execution_callback(self, callback: Callable[[ExecutionResult], None]) -> None:
        """Add a callback for model executions."""
        tprint(f"🚀 ModelType.add_execution_callback: callback={callback}, None={None}", "INFO")

        self.on_execution_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for errors."""
        tprint(f"🚀 ModelType.add_error_callback: callback={callback}, None={None}", "INFO")

        self.on_error_callbacks.append(callback)

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        tprint(f"🚀 ModelType.get_scheduler_stats: Starting", "INFO")

        total_executions = sum(config.execution_count for config in self.model_configs.values())
        total_successes = sum(config.success_count for config in self.model_configs.values())
        total_failures = sum(config.failure_count for config in self.model_configs.values())

        return {
            'is_running': self.is_running,
            'start_time': self.start_time,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'total_executions': total_executions,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'success_rate': total_successes / total_executions if total_executions > 0 else 0,
            'model_stats': {
                model_type.value: {
                    'execution_count': config.execution_count,
                    'success_count': config.success_count,
                    'failure_count': config.failure_count,
                    'avg_execution_time': config.avg_execution_time,
                    'next_execution': config.next_execution,
                    'enabled': config.enabled
                }
                for model_type, config in self.model_configs.items()
            },
            'execution_history_count': len(self.execution_history)
        }

    async def get_nowcasting_stats(self) -> Dict[str, Any]:
        """Get partial-bar nowcasting statistics."""
        tprint(f"🚀 ModelType.get_nowcasting_stats: Starting", "INFO")

        try:
            if self.nowcaster:
                return await self.nowcaster.get_nowcasting_stats()
            else:
                return {'error': 'Nowcaster not available'}
        except Exception as e:
            tprint_error(f"❌ Failed to get nowcasting stats: {e}")
            return {'error': str(e)}

    def get_recent_executions(self, n: int = 10) -> List[ExecutionResult]:
        """Get recent execution results."""
        tprint(f"🚀 ModelType.get_recent_executions: n={n}", "INFO")

        return self.execution_history[-n:] if len(self.execution_history) >= n else self.execution_history.copy()

    def enable_model(self, model_type: ModelType, enabled: bool = True) -> None:
        """Enable or disable a model."""
        tprint(f"🚀 ModelType.enable_model: model_type={model_type}, enabled={enabled}", "INFO")

        if model_type in self.model_configs:
            self.model_configs[model_type].enabled = enabled
            if enabled:
                self.model_configs[model_type].next_execution = datetime.now()
            tprint_info(f"📊 {model_type.value.upper()} model {'enabled' if enabled else 'disabled'}")

    def update_model_interval(self, model_type: ModelType, interval_seconds: int) -> None:
        """Update execution interval for a model."""
        tprint(f"🚀 ModelType.update_model_interval: model_type={model_type}, interval_seconds={interval_seconds}", "INFO")

        if model_type in self.model_configs:
            self.model_configs[model_type].execution_interval_seconds = interval_seconds
            tprint_info(f"📊 {model_type.value.upper()} execution interval updated to {interval_seconds}s")

# Convenience functions

def create_live_trading_scheduler(
    symbol: str = "ETH",
    exchange: str = "binance"
) -> LiveTradingScheduler:
    """Create a configured live trading scheduler."""
    return LiveTradingScheduler(symbol=symbol, exchange=exchange)

async def start_live_trading_scheduler(
    symbol: str = "ETH",
    exchange: str = "binance",
    execution_callback: Optional[Callable] = None,
    error_callback: Optional[Callable] = None
) -> LiveTradingScheduler:
    """Start live trading scheduler with default settings."""

    scheduler = create_live_trading_scheduler(symbol=symbol, exchange=exchange)

    if execution_callback:
        scheduler.add_execution_callback(execution_callback)

    if error_callback:
        scheduler.add_error_callback(error_callback)

    success = await scheduler.start_scheduler()
    if success:
        tprint_success(f"✅ Live trading scheduler started for {symbol} on {exchange}")
    else:
        tprint_error(f"❌ Failed to start live trading scheduler")

    return scheduler

# Example usage
if __name__ == "__main__":
    async def example_execution_callback(result: ExecutionResult) -> None:
        """Example execution callback."""
        tprint_info(f"📊 {result.model_type.value.upper()} execution completed: {result.status.value}")
        if result.result_data:
            tprint_info(f"   Duration: {result.execution_duration:.2f}s")

    async def example_error_callback(error: Exception) -> None:
        """Example error callback."""
        tprint_error(f"❌ Scheduler error: {error}")

    async def main() -> None:
        """Example main function."""
        tprint(f"🚀 ModelType.main: Starting", "INFO")

        scheduler = await start_live_trading_scheduler(
            symbol="ETH",
            exchange="binance",
            execution_callback=example_execution_callback,
            error_callback=example_error_callback
        )

        # Run for 5 minutes
        await asyncio.sleep(300)

        # Get stats
        stats = scheduler.get_scheduler_stats()
        tprint_structured(stats, LogLevel.INFO)

        # Stop scheduler
        await scheduler.stop_scheduler()

    # Run example
    asyncio.run(main())
