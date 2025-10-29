"""
Signal Generation Pipeline

Implements proper data flow: HMM regime -> analyst -> tactician
with sequential model calls and confidence score optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from threading import Lock
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from ..config.regime_config import RegimeType
from ..config.trading_config import TradingConfig
from ..model_selection import get_model_selector_service, ModelSelectionResult
from .utils import (
    validate_market_data, validate_regime_probabilities, validate_signal_parameters,
    calculate_weighted_regime_multiplier, CircuitBreaker, RateLimiter, SignalDeduplicator
)

logger = system_logger.getChild('SignalGenerationPipeline')

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_REGIME_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_EXIT_CONFIDENCE_THRESHOLD = 0.5
MIN_MARKET_DATA_POINTS = 20
DEFAULT_MAX_HISTORY = 1000
DEFAULT_RATE_LIMIT_CALLS = 10
DEFAULT_RATE_LIMIT_WINDOW = 60.0
DEFAULT_CIRCUIT_BREAKER_FAILURES = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT = 60.0
DEFAULT_SIGNAL_DEDUP_WINDOW = 300.0

@dataclass
class RegimeOutput:
    """HMM regime detection output."""
    timestamp: datetime
    regime_probabilities: Dict[RegimeType, float]
    primary_regime: RegimeType
    confidence: float
    regime_strength: float
    transition_probability: float
    features_used: Dict[str, Any]

@dataclass
class AnalystBaseOutput:
    """Analyst base models output."""
    timestamp: datetime
    market_health: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    liquidity_analysis: Dict[str, Any]
    stress_analysis: Dict[str, Any]
    base_confidence: float
    features: Dict[str, Any]

@dataclass
class AnalystMetaOutput:
    """Analyst meta model output."""
    timestamp: datetime
    analyst_confidence: float
    market_health_score: float
    regime_adjusted_confidence: float
    meta_features: Dict[str, Any]
    base_outputs: List[AnalystBaseOutput]

@dataclass
class TacticianBaseOutput:
    """Tactician base models output."""
    timestamp: datetime
    scenario_predictions: Dict[str, Any]
    price_targets: Dict[str, float]
    adversarial_risks: Dict[str, float]
    base_confidence: float
    position_recommendations: Dict[str, Any]

@dataclass
class TacticianMetaOutput:
    """Tactician meta model output."""
    timestamp: datetime
    tactician_confidence: float
    combined_confidence: float
    final_signal: str  # 'buy', 'sell', 'hold'
    signal_strength: float
    meta_features: Dict[str, Any]
    base_outputs: List[TacticianBaseOutput]

@dataclass
class PositionState:
    """Current position state."""
    is_open: bool = False
    entry_timestamp: Optional[datetime] = None
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    direction: Optional[str] = None  # 'long' or 'short'
    entry_confidence: Optional[float] = None

@dataclass
class SignalGenerationResult:
    """Complete signal generation result."""
    timestamp: datetime
    symbol: str
    hmm_output: RegimeOutput
    analyst_output: AnalystMetaOutput
    tactician_output: TacticianMetaOutput
    final_signal: str
    final_confidence: float
    signal_strength: float
    optimization_parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    # Exit-specific fields
    exit_confidence: Optional[float] = None
    should_exit: bool = False
    exit_reason: Optional[str] = None
    position_state: Optional[PositionState] = None

class SignalGenerationPipeline:
    """
    Signal generation pipeline with proper data flow:
    HMM regime -> analyst base models -> analyst meta model -> tactician base models -> tactician meta model

    Implements confidence score optimization based on backtesting parameters.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('SignalGenerationPipeline')

        # Pipeline components
        self.regime_detector = None
        self.analyst_base_models = []
        self.analyst_meta_model = None
        self.tactician_base_models = []
        self.tactician_meta_model = None

        # Model selection
        self.model_selector_service = None

        # Optimization parameters (from backtesting)
        self.optimization_params = {
            'analyst_confidence_weight': 0.6,
            'tactician_confidence_weight': 0.4,
            'regime_confidence_threshold': 0.7,
            'signal_confidence_threshold': 0.6,
            'meta_model_weight': 0.8,
            'base_model_weight': 0.2,
            # Exit-specific parameters
            'exit_confidence_threshold': 0.5,
            'tactician_exit_confidence_weight': 0.6,
            'analyst_exit_confidence_weight': 0.4,
            'exit_confidence_combination_method': 'multiplicative'  # 'multiplicative', 'logarithmic', 'weighted_average'
        }

        # State management
        self.is_initialized = False
        self.signal_history: deque = deque(maxlen=getattr(config, 'max_history', DEFAULT_MAX_HISTORY))

        # Position state management (thread-safe)
        self.current_position: Optional[PositionState] = None
        self.position_history: deque = deque(maxlen=getattr(config, 'max_history', DEFAULT_MAX_HISTORY))
        self._position_lock = Lock()

        # Circuit breaker for failure handling
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, 'circuit_breaker_failures', DEFAULT_CIRCUIT_BREAKER_FAILURES),
            recovery_timeout=getattr(config, 'circuit_breaker_timeout', DEFAULT_CIRCUIT_BREAKER_TIMEOUT)
        )

        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_calls=getattr(config, 'rate_limit_calls', DEFAULT_RATE_LIMIT_CALLS),
            time_window=getattr(config, 'rate_limit_window', DEFAULT_RATE_LIMIT_WINDOW)
        )

        # Signal deduplicator
        self.signal_deduplicator = SignalDeduplicator(
            deduplication_window=getattr(config, 'signal_dedup_window', DEFAULT_SIGNAL_DEDUP_WINDOW)
        )

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize signal generation pipeline."""
        try:
            self.logger.info("Initializing Signal Generation Pipeline...")

            # Initialize regime detector
            await self._initialize_regime_detector()

            # Initialize analyst models
            await self._initialize_analyst_models()

            # Initialize tactician models
            await self._initialize_tactician_models()

            # Initialize model selector service
            await self._initialize_model_selector_service()

            # Load optimization parameters
            await self._load_optimization_parameters()

            self.is_initialized = True
            self.logger.info("✅ Signal Generation Pipeline initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Signal Generation Pipeline: {e}")
            return False

    async def _initialize_regime_detector(self):
        """Initialize regime detector (loads models from market_analysis training)."""
        try:
            # Regime detector will be initialized by Strategist
            # The Strategist handles loading models from market_analysis/regime_models_training
            # and regime_ensemble_training, and provides regime predictions
            self.regime_detector = None
            self.logger.info("ℹ️ Regime detector will be provided by Strategist")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime detector: {e}")
            raise

    async def _initialize_analyst_models(self):
        """Initialize analyst base and meta models."""
        try:
            # Initialize meta model (analyst orchestrator)
            from src.analyst.analyst import Analyst
            self.analyst_meta_model = Analyst(self.config)
            await self.analyst_meta_model.initialize()

            # Load trained analyst base models from training steps using unified model loader
            from src.trading.integration.unified_model_loader import get_unified_model_loader
            
            unified_loader = get_unified_model_loader()
            
            # Load analyst base models and ensemble model
            symbol = getattr(self.config, 'symbol', 'ETHUSDT')
            exchange = getattr(self.config, 'exchange', 'binance')
            analyst_timeframe = getattr(self.config, 'analyst_timeframe', '15m')
            direction = getattr(self.config, 'direction', 'long')
            
            try:
                analyst_base_models_dict = await unified_loader.load_analyst_base_models(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=analyst_timeframe,
                    direction=direction
                )
                analyst_ensemble_model = await unified_loader.load_analyst_ensemble_model(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=analyst_timeframe,
                    direction=direction
                )
                
                # Convert dict to list for base models
                self.analyst_base_models = list(analyst_base_models_dict.values())
                
                # Store ensemble model separately if needed
                if analyst_ensemble_model:
                    self.analyst_base_models.append(analyst_ensemble_model)
                
                self.logger.info(f"✅ Loaded {len(self.analyst_base_models)} trained analyst models")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load trained analyst models: {e}")
                self.analyst_base_models = []  # Fallback to empty list

            self.logger.info("✅ Analyst models initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize analyst models: {e}")
            raise

    async def _initialize_tactician_models(self):
        """Initialize tactician base and meta models."""
        try:
            # Initialize meta model (tactician orchestrator)
            from src.tactician.tactician import Tactician
            self.tactician_meta_model = Tactician(self.config)
            await self.tactician_meta_model.initialize()

            # Load trained tactician base models from training steps using unified model loader
            from src.trading.integration.unified_model_loader import get_unified_model_loader
            
            unified_loader = get_unified_model_loader()
            
            # Load tactician base models and ensemble model
            symbol = getattr(self.config, 'symbol', 'ETHUSDT')
            exchange = getattr(self.config, 'exchange', 'binance')
            tactician_timeframe = getattr(self.config, 'tactician_timeframe', '5m')
            direction = getattr(self.config, 'direction', 'long')
            
            try:
                tactician_base_models_dict = await unified_loader.load_tactician_base_models(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=tactician_timeframe,
                    direction=direction
                )
                tactician_ensemble_model = await unified_loader.load_tactician_ensemble_model(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=tactician_timeframe,
                    direction=direction
                )
                
                # Convert dict to list for base models
                self.tactician_base_models = list(tactician_base_models_dict.values())
                
                # Store ensemble model separately if needed
                if tactician_ensemble_model:
                    self.tactician_base_models.append(tactician_ensemble_model)
                
                self.logger.info(f"✅ Loaded {len(self.tactician_base_models)} trained tactician models")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load trained tactician models: {e}")
                self.tactician_base_models = []  # Fallback to empty list

            self.logger.info("✅ Tactician models initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tactician models: {e}")
            raise

    async def _initialize_model_selector_service(self):
        """Initialize model selector service."""
        try:
            # Initialize model selector service
            self.model_selector_service = get_model_selector_service()

            self.logger.info("✅ Model selector service initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model selector service: {e}")
            raise

    async def _load_optimization_parameters(self):
        """Load optimization parameters from final_parameters_optimization step."""
        try:
            # Load optimization parameters from final_parameters_optimization using unified model loader
            from src.trading.integration.unified_model_loader import get_unified_model_loader
            
            unified_loader = get_unified_model_loader()
            
            symbol = getattr(self.config, 'symbol', 'ETHUSDT')
            exchange = getattr(self.config, 'exchange', 'binance')
            timeframe = getattr(self.config, 'timeframe', '15m')
            direction = getattr(self.config, 'direction', 'long')
            
            try:
                optimized_params = await unified_loader.load_optimized_parameters(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction
                )
                
                if optimized_params:
                    # Update optimization parameters with optimized values
                    self.optimization_params.update({
                        'analyst_confidence_weight': optimized_params.get('analyst_confidence_weight', 0.6),
                        'tactician_confidence_weight': optimized_params.get('tactician_confidence_weight', 0.4),
                        'regime_confidence_threshold': optimized_params.get('regime_confidence_threshold', 0.7),
                        'signal_confidence_threshold': optimized_params.get('signal_confidence_threshold', 0.6),
                        'meta_model_weight': optimized_params.get('meta_model_weight', 0.8),
                        'base_model_weight': optimized_params.get('base_model_weight', 0.2),
                        # Exit-specific parameters
                        'exit_confidence_threshold': optimized_params.get('exit_confidence_threshold', 0.5),
                        'tactician_exit_confidence_weight': optimized_params.get('tactician_exit_confidence_weight', 0.6),
                        'analyst_exit_confidence_weight': optimized_params.get('analyst_exit_confidence_weight', 0.4),
                        'exit_confidence_combination_method': optimized_params.get('exit_confidence_combination_method', 'multiplicative'),
                        # Additional parameters from optimization
                        'confidence_threshold': optimized_params.get('confidence_threshold', 0.75),
                        'position_sizing_factor': optimized_params.get('position_sizing_factor', 0.02),
                        'leverage_multiplier': optimized_params.get('leverage_multiplier', 1.5),
                        'stop_loss_pct': optimized_params.get('stop_loss_pct', 0.03),
                        'take_profit_pct': optimized_params.get('take_profit_pct', 0.06),
                        'ensemble_weight_analyst': optimized_params.get('ensemble_weight_analyst', 0.6),
                        'ensemble_weight_tactician': optimized_params.get('ensemble_weight_tactician', 0.4)
                    })
                    
                    self.logger.info("✅ Loaded optimized parameters from final_parameters_optimization")
                else:
                    self.logger.warning("⚠️ No optimized parameters found, using defaults")

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load optimized parameters: {e}")

            self.logger.info("✅ Optimization parameters loaded")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load optimization parameters, using defaults: {e}")

    async def _generate_signal_internal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        additional_features: Optional[Dict[str, Any]],
        timestamp: datetime
    ) -> SignalGenerationResult:
        """
        Internal signal generation method (called by generate_signal wrapper).

        Args:
            symbol: Trading symbol
            market_data: Market data DataFrame
            additional_features: Additional features for analysis
            timestamp: Timestamp for signal generation

        Returns:
            SignalGenerationResult: Complete signal generation result
        """
        # Step 1: HMM Regime Detection
        regime_output = await self._detect_regime(market_data, timestamp)
        
        # Validate regime probabilities
        is_valid, error_msg = validate_regime_probabilities(regime_output.regime_probabilities)
        if not is_valid:
            self.logger.warning(f"⚠️ Invalid regime probabilities: {error_msg}")

        # Step 1.5: Model Selection
        model_selection_result = await self._select_models_for_trading(
            market_data, symbol, timestamp
        )

        # Step 2: Analyst Base Models
        analyst_base_outputs = await self._run_analyst_base_models(
            market_data, regime_output, additional_features, timestamp, model_selection_result
        )

        # Step 3: Analyst Meta Model
        analyst_meta_output = await self._run_analyst_meta_model(
            market_data, regime_output, analyst_base_outputs, timestamp
        )

        # Step 4: Tactician Base Models
        tactician_base_outputs = await self._run_tactician_base_models(
            market_data, regime_output, analyst_meta_output, timestamp
        )

        # Step 5: Tactician Meta Model
        tactician_meta_output = await self._run_tactician_meta_model(
            market_data, regime_output, analyst_meta_output, tactician_base_outputs, timestamp
        )

        # Step 6: Calculate Exit Confidence (for position management)
        exit_confidence = self._calculate_exit_confidence(
            analyst_meta_output.analyst_confidence,
            tactician_meta_output.tactician_confidence
        )

        # Step 7: Check Exit Conditions (if position is open)
        should_exit, exit_reason = self._check_exit_conditions(exit_confidence)

        # Step 8: Final Signal Generation (with position validation)
        final_signal = self._generate_final_signal(
            regime_output, analyst_meta_output, tactician_meta_output, should_exit, exit_reason
        )

        # Validate signal against current position
        validation_result = self._validate_signal_against_position(final_signal)
        if not validation_result['is_valid']:
            self.logger.warning(f"⚠️ Signal validation failed: {validation_result['reason']}")
            # Adjust signal if needed
            if validation_result.get('adjusted_signal'):
                final_signal = validation_result['adjusted_signal']

        # Update position state based on signal (thread-safe)
        self._update_position_state(final_signal, timestamp, should_exit)

        # Create result
        result = SignalGenerationResult(
            timestamp=timestamp,
            symbol=symbol,
            regime_output=regime_output,
            analyst_output=analyst_meta_output,
            tactician_output=tactician_meta_output,
            final_signal=final_signal['signal'],
            final_confidence=final_signal['confidence'],
            signal_strength=final_signal['strength'],
            optimization_parameters=self.optimization_params,
            metadata={
                'symbol': symbol,
                'data_points': len(market_data),
                'processing_time_ms': 0  # Will be set by decorator
            },
            # Exit-specific fields
            exit_confidence=exit_confidence,
            should_exit=should_exit,
            exit_reason=exit_reason,
            position_state=self.current_position
        )

        # Store in history (deque automatically handles maxlen)
        self.signal_history.append(result)

        self.logger.debug(f"Signal generated for {symbol}: {final_signal['signal']} (confidence: {final_signal['confidence']:.3f})")

        return result

    @handles_errors
    @log_execution_time()
    @traced(span_name="generate_signal")
    async def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        additional_features: Optional[Dict[str, Any]] = None
    ) -> SignalGenerationResult:
        """
        Generate trading signal with proper data flow.

        Args:
            symbol: Trading symbol
            market_data: Market data DataFrame
            additional_features: Additional features for analysis

        Returns:
            SignalGenerationResult: Complete signal generation result
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Signal Generation Pipeline not initialized")

            # Input validation
            is_valid, error_msg = validate_signal_parameters(symbol=symbol)
            if not is_valid:
                raise ValueError(f"Invalid symbol parameter: {error_msg}")

            is_valid, error_msg = validate_market_data(market_data)
            if not is_valid:
                raise ValueError(f"Invalid market data: {error_msg}")

            # Rate limiting check
            if not self.rate_limiter.acquire():
                wait_time = self.rate_limiter.wait_time()
                raise RuntimeError(f"Rate limit exceeded. Wait {wait_time:.1f}s before retrying.")

            timestamp = datetime.now()

            # Check for signal deduplication (before generation to avoid wasted work)
            # Generate signal
            try:
                result = await self._generate_signal_internal(
                    symbol, market_data, additional_features, timestamp
                )
            except Exception as e:
                # Circuit breaker will handle the failure
                self.circuit_breaker._on_failure()
                raise

            # Check for duplicate signal
            if self.signal_deduplicator.is_duplicate(symbol, result.final_signal, timestamp):
                self.logger.warning(f"⚠️ Duplicate signal detected: {symbol} {result.final_signal}")
                # Still return the signal but log it

            # Record signal for deduplication
            self.signal_deduplicator.record_signal(symbol, result.final_signal, timestamp)

            # Success - update circuit breaker
            self.circuit_breaker._on_success()

            return result

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed for {symbol}: {e}")
            raise

    async def _detect_regime(self, market_data: pd.DataFrame, timestamp: datetime) -> RegimeOutput:
        """Step 1: Detect regime using regime detector (loads models from market_analysis training)."""
        # Fast fail: regime detector must be provided by Strategist
        if self.regime_detector is None:
            error_msg = "Regime detector not initialized. Strategist must provide regime detector."
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        try:
            # Use regime detector to predict regime
            regime_prediction = await self.regime_detector.predict_regime(market_data)
            
            if regime_prediction is None:
                raise RuntimeError("Regime detector returned None prediction")
            
            # Convert to RegimeOutput format
            primary_regime_raw = regime_prediction.get('primary_regime')
            if primary_regime_raw is None:
                # Default to first regime type if not provided
                primary_regime = list(RegimeType)[0]
            elif isinstance(primary_regime_raw, RegimeType):
                primary_regime = primary_regime_raw
            elif isinstance(primary_regime_raw, (int, str)):
                # Try to convert to RegimeType
                try:
                    if isinstance(primary_regime_raw, int):
                        # If it's an integer index, get the regime by index
                        regime_list = list(RegimeType)
                        if 0 <= primary_regime_raw < len(regime_list):
                            primary_regime = regime_list[primary_regime_raw]
                        else:
                            primary_regime = RegimeType.SIDEWAYS  # Default fallback
                    else:
                        # If it's a string, try to match by value
                        primary_regime = RegimeType(primary_regime_raw)
                except (ValueError, IndexError):
                    self.logger.warning(f"⚠️ Invalid primary_regime value: {primary_regime_raw}, defaulting to SIDEWAYS")
                    primary_regime = RegimeType.SIDEWAYS
            else:
                self.logger.warning(f"⚠️ Unexpected primary_regime type: {type(primary_regime_raw)}, defaulting to SIDEWAYS")
                primary_regime = RegimeType.SIDEWAYS
            
            return RegimeOutput(
                timestamp=timestamp,
                regime_probabilities=regime_prediction.get('regime_probabilities', {}),
                primary_regime=primary_regime,
                confidence=regime_prediction.get('confidence', 0.5),
                regime_strength=regime_prediction.get('regime_strength', 0.5),
                transition_probability=regime_prediction.get('transition_probability', 0.5),
                features_used=regime_prediction.get('features_used', {})
            )

        except Exception as e:
            error_msg = f"Regime detection failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _select_models_for_trading(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        timestamp: datetime
    ) -> ModelSelectionResult:
        """Step 1.5: Select best models for trading based on current regime."""
        try:
            if not self.model_selector_service:
                # Fallback if model selector not available
                return ModelSelectionResult(
                    selected_models={'analyst': 'default', 'tactician': 'default'},
                    ensemble_weights={'analyst': {'default': 1.0}, 'tactician': {'default': 1.0}},
                    regime_id=0,
                    confidence_score=0.5,
                    selection_metadata={
                        'fallback': True,
                        'fallback_reason': 'model_selector_unavailable',
                        'combined_timestamp': timestamp.isoformat()
                    },
                    confirmation_status='unavailable',
                    confirmation_details={'reason': 'model_selector_service_missing'}
                )

            # Select models for both timeframes
            analyst_models = self.model_selector_service.select_models_for_trading(
                market_data=market_data,
                model_types=['random_forest', 'xgboost', 'lightgbm'],
                symbol=symbol,
                timeframe='15m'
            )

            tactician_models = self.model_selector_service.select_models_for_trading(
                market_data=market_data,
                model_types=['random_forest', 'xgboost', 'lightgbm'],
                symbol=symbol,
                timeframe='5m'
            )

            cross_config: Dict[str, Any] = {}
            raw_cross_config = getattr(self.config, 'cross_timeframe_confirmation', None)
            if isinstance(raw_cross_config, dict):
                cross_config = raw_cross_config.copy()
            elif hasattr(self.config, 'custom_params') and isinstance(self.config.custom_params, dict):
                candidate = self.config.custom_params.get('cross_timeframe_confirmation', {})
                if isinstance(candidate, dict):
                    cross_config = candidate.copy()

            analyst_selected_models = {
                k: v for k, v in (analyst_models.selected_models or {}).items() if v
            }
            tactician_selected_models = {
                k: v for k, v in (tactician_models.selected_models or {}).items() if v
            }

            analyst_values = set(analyst_selected_models.values())
            tactician_values = set(tactician_selected_models.values())
            shared_models = sorted(analyst_values.intersection(tactician_values))

            analyst_regime = analyst_models.regime_id if analyst_models.regime_id is not None else 0
            tactician_regime = tactician_models.regime_id if tactician_models.regime_id is not None else 0

            regime_difference = abs(analyst_regime - tactician_regime)
            confidence_delta = abs(analyst_models.confidence_score - tactician_models.confidence_score)

            max_regime_difference = cross_config.get('max_regime_difference', 0)
            max_confidence_delta = cross_config.get('max_confidence_delta', 0.2)

            confirmation_details: Dict[str, Any] = {
                'enabled': bool(cross_config.get('enabled', False)),
                'analyst': {
                    'regime_id': analyst_regime,
                    'confidence_score': analyst_models.confidence_score,
                    'selected_models': analyst_selected_models,
                    'error': analyst_models.error_message,
                },
                'tactician': {
                    'regime_id': tactician_regime,
                    'confidence_score': tactician_models.confidence_score,
                    'selected_models': tactician_selected_models,
                    'error': tactician_models.error_message,
                },
                'shared_models': shared_models,
                'regime_difference': regime_difference,
                'regime_match': analyst_regime == tactician_regime,
                'confidence_delta': confidence_delta,
                'thresholds': {
                    'max_regime_difference': max_regime_difference,
                    'max_confidence_delta': max_confidence_delta,
                },
                'disagreement_reasons': [],
                'confirmation_passed': True,
            }

            # Combine results
            base_confidence = (analyst_models.confidence_score + tactician_models.confidence_score) / 2
            combined_result = ModelSelectionResult(
                selected_models={
                    'analyst': analyst_models.selected_models.get('random_forest', 'default'),
                    'tactician': tactician_models.selected_models.get('random_forest', 'default')
                },
                ensemble_weights={
                    'analyst': analyst_models.ensemble_weights.get('random_forest', {'default': 1.0}),
                    'tactician': tactician_models.ensemble_weights.get('random_forest', {'default': 1.0})
                },
                regime_id=analyst_models.regime_id,
                confidence_score=base_confidence,
                selection_metadata={
                    'analyst_selection': analyst_models.selection_metadata,
                    'tactician_selection': tactician_models.selection_metadata,
                    'combined_timestamp': timestamp.isoformat()
                },
                confirmation_status='disabled' if not confirmation_details['enabled'] else 'confirmed',
                confirmation_details={}
            )

            if confirmation_details['enabled']:
                disagreement_reasons: List[str] = []
                if regime_difference > max_regime_difference:
                    disagreement_reasons.append('regime_mismatch')
                if confidence_delta > max_confidence_delta:
                    disagreement_reasons.append('confidence_delta_exceeded')

                confirmation_details['disagreement_reasons'] = disagreement_reasons
                confirmation_details['confirmation_passed'] = len(disagreement_reasons) == 0

                if disagreement_reasons:
                    confirmation_details['action'] = 'reject' if cross_config.get('reject_on_disagreement', False) else 'downgrade'
                    if cross_config.get('reject_on_disagreement', False):
                        rejection_confidence = cross_config.get('rejection_confidence', 0.0)
                        confirmation_details['rejection_confidence'] = rejection_confidence
                        confirmation_details['confirmation_passed'] = False
                        combined_metadata = {
                            'fallback': True,
                            'fallback_reason': 'cross_timeframe_disagreement',
                            'analyst_selection': analyst_models.selection_metadata,
                            'tactician_selection': tactician_models.selection_metadata,
                            'combined_timestamp': timestamp.isoformat(),
                            'original_models': {
                                'analyst': analyst_models.selected_models,
                                'tactician': tactician_models.selected_models
                            },
                            'cross_timeframe_confirmation': confirmation_details
                        }
                        self.logger.warning(
                            "⚠️ Cross-timeframe confirmation rejected due to %s",
                            ', '.join(disagreement_reasons)
                        )
                        return ModelSelectionResult(
                            selected_models={'analyst': 'default', 'tactician': 'default'},
                            ensemble_weights={'analyst': {'default': 1.0}, 'tactician': {'default': 1.0}},
                            regime_id=analyst_models.regime_id,
                            confidence_score=rejection_confidence,
                            selection_metadata=combined_metadata,
                            confirmation_status='rejected',
                            confirmation_details=confirmation_details
                        )

                    downgrade_factor = max(0.0, float(cross_config.get('downgrade_confidence_factor', 0.5)))
                    original_confidence = combined_result.confidence_score
                    downgraded_confidence = max(0.0, original_confidence * downgrade_factor)
                    confirmation_details['applied_downgrade_factor'] = downgrade_factor
                    confirmation_details['original_confidence'] = original_confidence
                    combined_result.selection_metadata['original_confidence'] = original_confidence
                    combined_result.confidence_score = downgraded_confidence
                    combined_result.confirmation_status = 'downgraded'
                    confirmation_details['confirmation_passed'] = False
                    self.logger.warning(
                        "⚠️ Cross-timeframe disagreement (%s). Confidence downgraded by factor %.2f",
                        ', '.join(disagreement_reasons),
                        downgrade_factor
                    )
                else:
                    confirmation_details['action'] = 'confirmed'
                    combined_result.confirmation_status = 'confirmed'
            else:
                confirmation_details['action'] = 'disabled'
                confirmation_details['confirmation_passed'] = True

            combined_result.confirmation_details = confirmation_details
            combined_result.selection_metadata['cross_timeframe_confirmation'] = confirmation_details

            self.logger.info(
                "✅ Model selection completed (%s): %s",
                combined_result.confirmation_status,
                combined_result.selected_models
            )
            return combined_result

        except Exception as e:
            self.logger.error(f"❌ Model selection failed: {e}")
            # Return fallback result
            return ModelSelectionResult(
                selected_models={'analyst': 'default', 'tactician': 'default'},
                ensemble_weights={'analyst': {'default': 1.0}, 'tactician': {'default': 1.0}},
                regime_id=0,
                confidence_score=0.5,
                selection_metadata={
                    'error': str(e),
                    'fallback': True,
                    'fallback_reason': 'model_selection_exception'
                },
                confirmation_status='failed',
                confirmation_details={'exception': str(e)}
            )

    async def _run_analyst_base_models(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        additional_features: Optional[Dict[str, Any]],
        timestamp: datetime,
        model_selection_result: Optional[ModelSelectionResult] = None
    ) -> List[AnalystBaseOutput]:
        """Step 2: Run analyst base models sequentially."""
        try:
            base_outputs = []

            # Run the trained analyst base models from training steps
            # These are the models trained in analyst_models_training_refactored.py
            # Use model selection result if available
            selected_analyst_model = None
            if model_selection_result and 'analyst' in model_selection_result.selected_models:
                selected_analyst_model = model_selection_result.selected_models['analyst']
                self.logger.info(f"🎯 Using selected analyst model: {selected_analyst_model}")

            for model in self.analyst_base_models:
                try:
                    # Use the trained model to make predictions
                    # The model should have a predict method
                    if hasattr(model, 'predict'):
                        prediction = model.predict(market_data)
                        confidence = getattr(prediction, 'confidence', 0.5) if hasattr(prediction, 'confidence') else 0.5
                        features = getattr(prediction, 'features', {}) if hasattr(prediction, 'features') else {}
                    else:
                        # Fallback for models without standard predict interface
                        self.logger.warning(f"⚠️ Analyst base model missing 'predict' method, using fallback confidence")
                        confidence = 0.5
                        features = {}

                    # Create base output (ignore market health and liquidation risk as requested)
                    base_output = AnalystBaseOutput(
                        timestamp=timestamp,
                        market_health={},  # Ignored as requested
                        volatility_analysis={},
                        liquidity_analysis={},
                        stress_analysis={},  # Ignored as requested
                        base_confidence=confidence,
                        features=features
                    )

                    base_outputs.append(base_output)

                except Exception as e:
                    self.logger.warning(f"⚠️ Analyst base model failed: {e}", exc_info=True)
                    # Create fallback output with explicit logging
                    self.logger.debug(f"Creating fallback output due to model failure")
                    base_outputs.append(AnalystBaseOutput(
                        timestamp=timestamp,
                        market_health={},
                        volatility_analysis={},
                        liquidity_analysis={},
                        stress_analysis={},
                        base_confidence=0.5,
                        features={}
                    ))

            return base_outputs

        except Exception as e:
            self.logger.error(f"❌ Analyst base models failed: {e}")
            raise

    async def _run_analyst_meta_model(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        base_outputs: List[AnalystBaseOutput],
        timestamp: datetime
    ) -> AnalystMetaOutput:
        """Step 3: Run analyst meta model."""
        try:
            # Use the existing analyst.analyze_regime method
            meta_result = await self.analyst_meta_model.analyze_regime(market_data)

            # Extract confidence from the result
            analyst_confidence = meta_result.get('confidence', 0.5)

            # Apply regime adjustment
            regime_adjusted_confidence = self._apply_regime_adjustment(
                analyst_confidence,
                    regime_output.regime_probabilities
            )

            return AnalystMetaOutput(
                timestamp=timestamp,
                analyst_confidence=regime_adjusted_confidence,
                market_health_score=meta_result.get('market_health_score', 0.5),
                regime_adjusted_confidence=regime_adjusted_confidence,
                meta_features=meta_result,
                base_outputs=base_outputs
            )

        except Exception as e:
            self.logger.error(f"❌ Analyst meta model failed: {e}")
            raise

    async def _run_tactician_base_models(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        analyst_output: AnalystMetaOutput,
        timestamp: datetime
    ) -> List[TacticianBaseOutput]:
        """Step 4: Run tactician base models sequentially."""
        try:
            base_outputs = []

            # Run the trained tactician base models from training steps
            # These are the models trained in tactician_models_training_refactored.py
            for model in self.tactician_base_models:
                try:
                    # Use the trained model to make predictions
                    # The model should have a predict method
                    if hasattr(model, 'predict'):
                        prediction = model.predict(market_data)
                        confidence = getattr(prediction, 'confidence', 0.5) if hasattr(prediction, 'confidence') else 0.5
                        scenario_predictions = getattr(prediction, 'scenario_predictions', {}) if hasattr(prediction, 'scenario_predictions') else {}
                        price_targets = getattr(prediction, 'price_targets', {}) if hasattr(prediction, 'price_targets') else {}
                        adversarial_risks = getattr(prediction, 'adversarial_risks', {}) if hasattr(prediction, 'adversarial_risks') else {}
                    else:
                        # Fallback for models without standard predict interface
                        self.logger.warning(f"⚠️ Tactician base model missing 'predict' method, using fallback confidence")
                        confidence = 0.5
                        scenario_predictions = {}
                        price_targets = {}
                        adversarial_risks = {}

                    # Create base output
                    base_output = TacticianBaseOutput(
                        timestamp=timestamp,
                        scenario_predictions=scenario_predictions,
                        price_targets=price_targets,
                        adversarial_risks=adversarial_risks,
                        base_confidence=confidence,
                        position_recommendations={}
                    )

                    base_outputs.append(base_output)

                except Exception as e:
                    self.logger.warning(f"⚠️ Tactician base model failed: {e}", exc_info=True)
                    # Create fallback output with explicit logging
                    self.logger.debug(f"Creating fallback output due to model failure")
                    base_outputs.append(TacticianBaseOutput(
                        timestamp=timestamp,
                        scenario_predictions={},
                        price_targets={},
                        adversarial_risks={},
                        base_confidence=0.5,
                        position_recommendations={}
                    ))

            return base_outputs

        except Exception as e:
            self.logger.error(f"❌ Tactician base models failed: {e}")
            raise

    async def _run_tactician_meta_model(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        analyst_output: AnalystMetaOutput,
        base_outputs: List[TacticianBaseOutput],
        timestamp: datetime
    ) -> TacticianMetaOutput:
        """Step 5: Run tactician meta model."""
        try:
            # Use the existing tactician.generate_enhanced_predictions method
            symbol = "ETHUSDT"  # Default symbol
            if hasattr(market_data, 'columns') and len(market_data.columns) > 0:
                symbol = market_data.columns[0]

            meta_result = await self.tactician_meta_model.generate_enhanced_predictions(
                market_data, {}, symbol, "1m", analyst_output.analyst_confidence
            )

            # Extract confidence from the result
            tactician_confidence = meta_result.get('confidence', 0.5)

            # Calculate combined confidence
            combined_confidence = self._calculate_combined_confidence(
                analyst_output.analyst_confidence, tactician_confidence
            )

            # Extract final signal from trading decisions
            trading_decisions = meta_result.get('trading_decisions', {})
            final_signal = 'hold'
            if trading_decisions.get('entry_signal', False):
                final_signal = 'buy' if trading_decisions.get('direction', '') == 'long' else 'sell'

            return TacticianMetaOutput(
                timestamp=timestamp,
                tactician_confidence=tactician_confidence,
                combined_confidence=combined_confidence,
                final_signal=final_signal,
                signal_strength=meta_result.get('signal_strength', 0.5),
                meta_features=meta_result,
                base_outputs=base_outputs
            )

        except Exception as e:
            self.logger.error(f"❌ Tactician meta model failed: {e}")
            raise

    def _apply_regime_adjustment(self, base_confidence: float, regime_probabilities: Dict[RegimeType, float]) -> float:
        """Apply regime-based confidence adjustment using weighted average."""
        try:
            # Regime confidence multipliers
            regime_multipliers = {
                RegimeType.TRENDING_UP: 1.2,
                RegimeType.TRENDING_DOWN: 1.2,
                RegimeType.SIDEWAYS: 0.8,
                RegimeType.HIGH_VOLATILITY: 0.7,
                RegimeType.LOW_VOLATILITY: 1.1,
                RegimeType.BREAKOUT: 1.3,
                RegimeType.REVERSAL: 0.9,
                RegimeType.MOMENTUM: 1.2,
                RegimeType.MEAN_REVERSION: 0.9,
            }

            # Calculate weighted regime multiplier (fixed: use weighted average instead of additive)
            regime_multiplier = calculate_weighted_regime_multiplier(
                regime_probabilities, regime_multipliers
            )

            # Apply adjustment
            adjusted_confidence = base_confidence * regime_multiplier
            return max(0.0, min(1.0, adjusted_confidence))

        except Exception as e:
            self.logger.warning(f"⚠️ Regime adjustment failed: {e}")
            return base_confidence

    def _calculate_combined_confidence(self, analyst_confidence: float, tactician_confidence: float) -> float:
        """Calculate combined confidence using optimization parameters."""
        try:
            analyst_weight = self.optimization_params['analyst_confidence_weight']
            tactician_weight = self.optimization_params['tactician_confidence_weight']

            combined = (analyst_confidence * analyst_weight +
                       tactician_confidence * tactician_weight)

            return max(0.0, min(1.0, combined))

        except Exception as e:
            self.logger.warning(f"⚠️ Combined confidence calculation failed: {e}")
            return (analyst_confidence + tactician_confidence) / 2

    def _calculate_exit_confidence(self, analyst_confidence: float, tactician_confidence: float) -> float:
        """
        Calculate exit confidence using multiplicative and logarithmic combinations.

        This method implements the requirement for optimal exit confidence calculation based on
        tactician's and analyst's confidence outputs, using different combination methods.

        Args:
            analyst_confidence: Current analyst confidence (regularly recalculated)
            tactician_confidence: Current tactician confidence (regularly recalculated)

        Returns:
            Combined exit confidence value
        """
        try:
            # Get exit-specific weights and combination method
            tactician_weight = self.optimization_params['tactician_exit_confidence_weight']
            analyst_weight = self.optimization_params['analyst_exit_confidence_weight']
            combination_method = self.optimization_params['exit_confidence_combination_method']

            # Ensure weights are normalized
            total_weight = tactician_weight + analyst_weight
            if total_weight > 0:
                tactician_weight = tactician_weight / total_weight
                analyst_weight = analyst_weight / total_weight
            else:
                tactician_weight = 0.6
                analyst_weight = 0.4

            # Calculate exit confidence based on selected method
            if combination_method == 'multiplicative':
                # Multiplicative combination: (tactician_conf^tactician_weight) * (analyst_conf^analyst_weight)
                exit_confidence = self._calculate_multiplicative_exit_confidence(
                    analyst_confidence, tactician_confidence, tactician_weight, analyst_weight
                )
            elif combination_method == 'logarithmic':
                # Logarithmic combination: exp(tactician_weight * log(tactician_conf) + analyst_weight * log(analyst_conf))
                exit_confidence = self._calculate_logarithmic_exit_confidence(
                    analyst_confidence, tactician_confidence, tactician_weight, analyst_weight
                )
            else:  # weighted_average or default
                # Weighted average combination
                exit_confidence = (analyst_confidence * analyst_weight +
                                 tactician_confidence * tactician_weight)

            # Ensure confidence is within valid range [0, 1]
            exit_confidence = max(0.0, min(1.0, exit_confidence))

            self.logger.debug(f"📊 Exit confidence calculation using {combination_method}:")
            self.logger.debug(f"   Analyst: {analyst_confidence:.4f} (weight: {analyst_weight:.3f})")
            self.logger.debug(f"   Tactician: {tactician_confidence:.4f} (weight: {tactician_weight:.3f})")
            self.logger.debug(f"   Exit confidence: {exit_confidence:.4f}")

            return exit_confidence

        except Exception as e:
            self.logger.error(f"❌ Error calculating exit confidence: {e}")
            # Fallback to simple weighted average
            return (analyst_confidence * 0.4 + tactician_confidence * 0.6)

    def _calculate_multiplicative_exit_confidence(self, analyst_confidence: float, tactician_confidence: float,
                                                tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate exit confidence using multiplicative operations.

        Formula: (tactician_confidence^tactician_weight) * (analyst_confidence^analyst_weight)
        """
        try:
            # Ensure confidences are positive for power operations
            analyst_conf = max(0.001, analyst_confidence)
            tactician_conf = max(0.001, tactician_confidence)

            # Multiplicative combination with weights as exponents
            multiplicative_conf = (
                (tactician_conf ** tactician_weight) *
                (analyst_conf ** analyst_weight)
            )

            # Normalize to [0, 1] range
            multiplicative_conf = min(1.0, multiplicative_conf)

            return multiplicative_conf

        except Exception as e:
            self.logger.error(f"❌ Error in multiplicative exit confidence calculation: {e}")
            return 0.5  # Default fallback

    def _calculate_logarithmic_exit_confidence(self, analyst_confidence: float, tactician_confidence: float,
                                             tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate exit confidence using logarithmic additions.

        Formula: exp(tactician_weight * log(tactician_confidence) + analyst_weight * log(analyst_confidence))
        """
        try:
            # Ensure confidences are positive for log operations
            analyst_conf = max(0.001, analyst_confidence)
            tactician_conf = max(0.001, tactician_confidence)

            # Logarithmic addition with weights
            log_combination = (
                tactician_weight * np.log(tactician_conf) +
                analyst_weight * np.log(analyst_conf)
            )

            # Convert back using exponential
            logarithmic_conf = np.exp(log_combination)

            # Normalize to [0, 1] range
            logarithmic_conf = min(1.0, max(0.0, logarithmic_conf))

            return logarithmic_conf

        except Exception as e:
            self.logger.error(f"❌ Error in logarithmic exit confidence calculation: {e}")
            return 0.5  # Default fallback

    def _check_exit_conditions(self, exit_confidence: float) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited based on exit confidence threshold.

        Args:
            exit_confidence: Combined exit confidence from analyst and tactician

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        try:
            # If no position is open, no need to exit
            if not self.current_position or not self.current_position.is_open:
                return False, None

            exit_threshold = self.optimization_params['exit_confidence_threshold']

            # Check if exit confidence drops below threshold
            if exit_confidence < exit_threshold:
                exit_reason = f"Exit confidence {exit_confidence:.3f} below threshold {exit_threshold:.3f}"
                self.logger.info(f"🚪 Exit condition triggered: {exit_reason}")
                return True, exit_reason

            return False, None

        except Exception as e:
            self.logger.error(f"❌ Error checking exit conditions: {e}")
            return False, f"Error checking exit conditions: {e}"

    def _validate_signal_against_position(self, final_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate signal against current position to avoid conflicts.
        
        Args:
            final_signal: Generated trading signal
        
        Returns:
            Dict with 'is_valid', 'reason', and optionally 'adjusted_signal'
        """
        try:
            with self._position_lock:
                current_pos = self.current_position
                
            signal = final_signal['signal']
            
            # If no position, any entry signal is valid
            if not current_pos or not current_pos.is_open:
                if signal in ['buy', 'sell']:
                    return {'is_valid': True, 'reason': 'No open position, entry signal valid'}
                return {'is_valid': True, 'reason': 'No open position'}
            
            # Check for conflicts
            if signal == 'buy' and current_pos.direction == 'long':
                return {
                    'is_valid': False,
                    'reason': 'Buy signal conflicts with existing long position',
                    'adjusted_signal': {'signal': 'hold', 'confidence': final_signal['confidence'] * 0.5, 'strength': 0.0}
                }
            
            if signal == 'sell' and current_pos.direction == 'short':
                return {
                    'is_valid': False,
                    'reason': 'Sell signal conflicts with existing short position',
                    'adjusted_signal': {'signal': 'hold', 'confidence': final_signal['confidence'] * 0.5, 'strength': 0.0}
                }
            
            # Close signals are always valid when position is open
            if signal == 'close':
                return {'is_valid': True, 'reason': 'Close signal valid for open position'}
            
            # Entry signals opposite to current position need higher confidence
            if signal == 'buy' and current_pos.direction == 'short':
                if final_signal['confidence'] < 0.8:
                    return {
                        'is_valid': False,
                        'reason': 'Buy signal to reverse short position requires higher confidence',
                        'adjusted_signal': {'signal': 'close', 'confidence': final_signal['confidence'], 'strength': final_signal['strength']}
                    }
            
            if signal == 'sell' and current_pos.direction == 'long':
                if final_signal['confidence'] < 0.8:
                    return {
                        'is_valid': False,
                        'reason': 'Sell signal to reverse long position requires higher confidence',
                        'adjusted_signal': {'signal': 'close', 'confidence': final_signal['confidence'], 'strength': final_signal['strength']}
                    }
            
            return {'is_valid': True, 'reason': 'Signal validated against position'}
            
        except Exception as e:
            self.logger.error(f"❌ Signal position validation failed: {e}")
            return {'is_valid': True, 'reason': f'Validation error: {e}'}  # Default to valid on error

    def _update_position_state(self, final_signal: Dict[str, Any], timestamp: datetime, should_exit: bool):
        """
        Update position state based on signal and exit conditions (thread-safe).

        Args:
            final_signal: Generated trading signal
            timestamp: Current timestamp
            should_exit: Whether position should be exited
        """
        try:
            signal = final_signal['signal']
            confidence = final_signal['confidence']

            with self._position_lock:
                # Handle exit conditions first
                if should_exit and self.current_position and self.current_position.is_open:
                    # Close current position
                    self.current_position.is_open = False
                    self.position_history.append(self.current_position)
                    self.logger.info(f"📉 Position closed: {self.current_position.direction} from {self.current_position.entry_timestamp}")
                    self.current_position = None
                    return

                # Handle new position entries
                if signal in ['buy', 'sell'] and (not self.current_position or not self.current_position.is_open):
                    # Open new position
                    self.current_position = PositionState(
                        is_open=True,
                        entry_timestamp=timestamp,
                        entry_price=None,  # Would be set by execution engine
                        position_size=None,  # Would be set by execution engine
                        direction='long' if signal == 'buy' else 'short',
                        entry_confidence=confidence
                    )
                    self.logger.info(f"📈 New position opened: {signal} at {timestamp} (confidence: {confidence:.3f})")

                # Handle position closes from signal
                elif signal == 'close' and self.current_position and self.current_position.is_open:
                    self.current_position.is_open = False
                    self.position_history.append(self.current_position)
                    self.logger.info(f"📉 Position closed by signal: {self.current_position.direction}")
                    self.current_position = None

        except Exception as e:
            self.logger.error(f"❌ Error updating position state: {e}")

    def _generate_final_signal(
        self,
        regime_output: RegimeOutput,
        analyst_output: AnalystMetaOutput,
        tactician_output: TacticianMetaOutput,
        should_exit: bool = False,
        exit_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate final trading signal with validation."""
        try:
            # Priority 1: Handle exit conditions first
            if should_exit:
                return {
                    'signal': 'close',
                    'confidence': 1.0,  # High confidence for exit
                    'strength': 1.0,
                    'reason': f'Exit condition triggered: {exit_reason}'
                }

            # Use optimization parameters for thresholds
            regime_threshold = self.optimization_params['regime_confidence_threshold']
            signal_threshold = self.optimization_params['signal_confidence_threshold']

            # Check regime confidence
            if regime_output.confidence < regime_threshold:
                return {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'reason': f'Low regime confidence: {regime_output.confidence:.3f} < {regime_threshold:.3f}'
                }

            # Check signal confidence
            if tactician_output.combined_confidence < signal_threshold:
                return {
                    'signal': 'hold',
                    'confidence': tactician_output.combined_confidence,
                    'strength': 0.0,
                    'reason': f'Low signal confidence: {tactician_output.combined_confidence:.3f} < {signal_threshold:.3f}'
                }

            # Validate signal based on analyst and tactician outputs
            validation_result = self._validate_signal(analyst_output, tactician_output)

            if not validation_result['is_valid']:
                return {
                    'signal': 'hold',
                    'confidence': tactician_output.combined_confidence,
                    'strength': 0.0,
                    'reason': f'Signal validation failed: {validation_result["reason"]}'
                }

            # Generate signal based on tactician output
            final_signal = tactician_output.final_signal
            final_confidence = tactician_output.combined_confidence
            signal_strength = tactician_output.signal_strength

            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'strength': signal_strength,
                'reason': f'Signal generated with confidence: {final_confidence:.3f}'
            }

        except Exception as e:
            self.logger.error(f"❌ Final signal generation failed: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'strength': 0.0,
                'reason': f'Error: {e}'
            }

    def _validate_signal(self, analyst_output: AnalystMetaOutput, tactician_output: TacticianMetaOutput) -> Dict[str, Any]:
        """
        Simple signal validation based on analyst and tactician outputs.
        This is optimized by the backtesting modules.
        """
        try:
            # Basic validation checks
            validation_checks = []

            # Check analyst confidence
            if analyst_output.analyst_confidence < 0.3:
                validation_checks.append("Low analyst confidence")

            # Check tactician confidence
            if tactician_output.tactician_confidence < 0.3:
                validation_checks.append("Low tactician confidence")

            # Check confidence consistency
            confidence_diff = abs(analyst_output.analyst_confidence - tactician_output.tactician_confidence)
            if confidence_diff > 0.5:
                validation_checks.append("High confidence difference between analyst and tactician")

            # Check signal strength
            if tactician_output.signal_strength < 0.2:
                validation_checks.append("Weak signal strength")

            # Check if signal is not hold
            if tactician_output.final_signal == 'hold':
                validation_checks.append("Tactician recommends hold")

            # Determine if signal is valid
            is_valid = len(validation_checks) == 0
            reason = "; ".join(validation_checks) if validation_checks else "Signal validation passed"

            return {
                'is_valid': is_valid,
                'reason': reason,
                'checks_failed': validation_checks
            }

        except Exception as e:
            self.logger.error(f"❌ Signal validation failed: {e}")
            return {
                'is_valid': False,
                'reason': f'Validation error: {e}',
                'checks_failed': ['Validation error']
            }

    def get_signal_history(self, limit: int = 100) -> List[SignalGenerationResult]:
        """Get recent signal generation history."""
        # Convert deque to list for return (deque is already bounded by maxlen)
        return list(self.signal_history)[-limit:] if len(self.signal_history) > limit else list(self.signal_history)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for signal generation."""
        try:
            if not self.signal_history:
                return {
                    'total_signals': 0,
                    'avg_confidence': 0.0,
                    'signal_distribution': {'buy': 0, 'sell': 0, 'hold': 0}
                }

            # Convert deque to list for slicing
            signal_list = list(self.signal_history)
            recent_signals = signal_list[-100:] if len(signal_list) > 100 else signal_list

            avg_confidence = sum(s.final_confidence for s in recent_signals) / len(recent_signals) if recent_signals else 0.0

            signal_distribution = {'buy': 0, 'sell': 0, 'hold': 0}
            for signal in recent_signals:
                signal_distribution[signal.final_signal] += 1

            return {
                'total_signals': len(self.signal_history),
                'avg_confidence': avg_confidence,
                'signal_distribution': signal_distribution,
                'optimization_parameters': self.optimization_params
            }

        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {}

    async def stop(self):
        """Stop signal generation pipeline."""
        try:
            self.logger.info("🛑 Stopping Signal Generation Pipeline...")

            # Stop all components
            if self.regime_detector:
                await self.regime_detector.stop()

            for model in self.analyst_base_models:
                if hasattr(model, 'stop'):
                    await model.stop()

            if self.analyst_meta_model and hasattr(self.analyst_meta_model, 'stop'):
                await self.analyst_meta_model.stop()

            for model in self.tactician_base_models:
                if hasattr(model, 'stop'):
                    await model.stop()

            if self.tactician_meta_model and hasattr(self.tactician_meta_model, 'stop'):
                await self.tactician_meta_model.stop()

            self.is_initialized = False
            self.logger.info("✅ Signal Generation Pipeline stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping Signal Generation Pipeline: {e}")

# Convenience function
async def setup_signal_generation_pipeline(config: TradingConfig) -> Optional[SignalGenerationPipeline]:
    """Setup and initialize signal generation pipeline."""
    try:
        pipeline = SignalGenerationPipeline(config)
        success = await pipeline.initialize()
        if success:
            return pipeline
        return None
    except Exception as e:
        logger.error(f"❌ Failed to setup signal generation pipeline: {e}")
        return None
