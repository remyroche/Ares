"""
Signal Generation Pipeline

Implements proper data flow: Regime -> Analyst Base Models -> Analyst Ensemble -> Tactician Base Models -> Tactician Ensemble
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

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.printing import tprint
from src.utils.logger import system_logger
from src.trading.config.regime_config import RegimeType
from src.trading.config.trading_config import TradingConfig
from src.trading.model_selection.model_selector_service import ModelSelectionResult, get_model_selector_service
from src.core.decorators import handles_errors, traced, log_execution_time
from .utils import (
    CircuitBreaker, RateLimiter, SignalDeduplicator, 
    validate_regime_probabilities, validate_market_data, validate_signal_parameters,
    calculate_weighted_regime_multiplier
)

# Shared feature engineering
from src.feature_generation.shared.feature_engineer import (
    AnalystFeatureEngineer,
    TacticianFeatureEngineer
)
from src.feature_generation.shared.feature_validator import FeatureValidator

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
    """Analyst ensemble model output (previously called meta output for compatibility)."""
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
    """Tactician ensemble model output (previously called meta output for compatibility)."""
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
    entry_analyst_confidence: Optional[float] = None  # Track analyst confidence at entry
    entry_tactician_confidence: Optional[float] = None  # Track tactician confidence at entry
    peak_profit_pct: Optional[float] = None  # Tracks highest profit achieved for trailing

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
    Regime -> Analyst Base Models -> Analyst Ensemble -> Tactician Base Models -> Tactician Ensemble

    Architecture:
    - Regime Detector: Provides regime probabilities
    - Analyst Base Models: Individual ML models making predictions
    - Analyst Ensemble Model: ML stacker that combines base predictions optimally
    - Tactician Base Models: Individual ML models making timing predictions (with analyst inputs)
    - Tactician Ensemble Model: ML stacker that combines base predictions optimally

    Implements confidence score optimization based on backtesting parameters.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        tprint(f"[SIGNAL_PIPELINE] __init__: Initializing signal generation pipeline")
        self.logger = logger.getChild('SignalGenerationPipeline')

        # Pipeline components
        self.regime_detector: Optional[Any] = None
        self.analyst_base_models: List[Any] = []
        self.analyst_ensemble_model: Optional[Any] = None
        self.tactician_base_models: List[Any] = []
        self.tactician_ensemble_model: Optional[Any] = None

        # Model selection
        self.model_selector_service: Optional[Any] = None

        # Optimization parameters (from backtesting)
        # These will be overridden by final_parameters_optimization if available
        self.optimization_params = {
            # Note: No confidence combination weights - we use only Tactician's Ensemble confidence
            # Keeping for backward compatibility but not used
            'analyst_confidence_weight': 0.6,  # Deprecated - not used
            'tactician_confidence_weight': 0.4,  # Deprecated - not used
            'regime_confidence_threshold': DEFAULT_REGIME_CONFIDENCE_THRESHOLD,
            'signal_confidence_threshold': DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD,
            # Exit-specific parameters (will be overridden by final_parameters_optimization)
            'exit_confidence_threshold': DEFAULT_EXIT_CONFIDENCE_THRESHOLD,
            # Note: Exit confidence weights removed - we use only Tactician Ensemble confidence
            'tactician_exit_confidence_weight': 0.6,  # Deprecated - not used
            'analyst_exit_confidence_weight': 0.4,  # Deprecated - not used
            # Note: exit_confidence_combination_method removed - we use only Tactician Ensemble confidence
            # Exit strategy parameters (loaded from final_parameters_optimization)
            'exit_strategy': {}
        }

        # State management
        self.is_initialized: bool = False
        self.signal_history: deque[SignalGenerationResult] = deque(maxlen=getattr(config, 'max_history', DEFAULT_MAX_HISTORY))

        # Position state management (thread-safe)
        self.current_position: Optional[PositionState] = None
        self.position_history: deque[PositionState] = deque(maxlen=getattr(config, 'max_history', DEFAULT_MAX_HISTORY))
        self._position_lock: Lock = Lock()

        # Circuit breaker for failure handling
        self.circuit_breaker: CircuitBreaker = CircuitBreaker(
            failure_threshold=getattr(config, 'circuit_breaker_failures', DEFAULT_CIRCUIT_BREAKER_FAILURES),
            recovery_timeout=getattr(config, 'circuit_breaker_timeout', DEFAULT_CIRCUIT_BREAKER_TIMEOUT)
        )

        # Rate limiter
        self.rate_limiter: RateLimiter = RateLimiter(
            max_calls=getattr(config, 'rate_limit_calls', DEFAULT_RATE_LIMIT_CALLS),
            time_window=getattr(config, 'rate_limit_window', DEFAULT_RATE_LIMIT_WINDOW)
        )

        # Signal deduplicator
        self.signal_deduplicator: SignalDeduplicator = SignalDeduplicator(
            deduplication_window=getattr(config, 'signal_dedup_window', DEFAULT_SIGNAL_DEDUP_WINDOW)
        )
        
        # Shared feature engineers (for consistency with training)
        self.analyst_feature_engineer = AnalystFeatureEngineer(logger=self.logger)
        self.tactician_feature_engineer = TacticianFeatureEngineer(logger=self.logger)
        self.feature_validator = FeatureValidator(logger=self.logger)
        tprint(f"[SIGNAL_PIPELINE] __init__ -> initialized (regime_threshold={self.optimization_params[\'regime_confidence_threshold\']}, signal_threshold={self.optimization_params[\'signal_confidence_threshold\']})")

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize signal generation pipeline."""
        try:
            tprint(f"[SIGNAL_PIPELINE] initialize: Initializing pipeline components")
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
            
            # Validate feature engineering setup
            await self._validate_feature_engineering()
            
            self.is_initialized = True
            self.logger.info("✅ Signal Generation Pipeline initialized successfully")
            tprint(f"[SIGNAL_PIPELINE] initialize -> True (all components initialized)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Signal Generation Pipeline: {e}")
            tprint(f"[SIGNAL_PIPELINE] initialize -> False (error: {e})", color="red")
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
        """Initialize analyst base models and ensemble model."""
        try:
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
                
                # Store ensemble model separately (not in base_models list)
                # The ensemble model will use base model predictions as features
                self.analyst_ensemble_model = analyst_ensemble_model
                
                self.logger.info(
                    f"✅ Loaded {len(self.analyst_base_models)} analyst base models"
                    f"{' and ensemble model' if analyst_ensemble_model else ''}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load trained analyst models: {e}")
                self.analyst_base_models = []  # Fallback to empty list
                self.analyst_ensemble_model = None

            self.logger.info("✅ Analyst models initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize analyst models: {e}")
            raise

    async def _initialize_tactician_models(self):
        """Initialize tactician base models and ensemble model."""
        try:
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
                
                # Store ensemble model separately (not in base_models list)
                # The ensemble model will use base model predictions as features
                self.tactician_ensemble_model = tactician_ensemble_model
                
                self.logger.info(
                    f"✅ Loaded {len(self.tactician_base_models)} tactician base models"
                    f"{' and ensemble model' if tactician_ensemble_model else ''}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load trained tactician models: {e}")
                self.tactician_base_models = []  # Fallback to empty list
                self.tactician_ensemble_model = None

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
            tprint_info("🔄 Loading optimization parameters from final_parameters_optimization...")
            
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
                    tprint_success(f"✅ Successfully loaded optimization parameters for {symbol} on {exchange} ({timeframe}, {direction})")
                    
                    # Update optimization parameters with optimized values (overriding defaults)
                    self.optimization_params.update({
                        # Note: confidence weights removed - we use only Tactician Ensemble confidence
                        'regime_confidence_threshold': optimized_params.get('regime_confidence_threshold', DEFAULT_REGIME_CONFIDENCE_THRESHOLD),
                        'signal_confidence_threshold': optimized_params.get('signal_confidence_threshold', DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD),
                        # Exit-specific parameters (override defaults)
                        'exit_confidence_threshold': optimized_params.get('exit_confidence_threshold', DEFAULT_EXIT_CONFIDENCE_THRESHOLD),
                        # Note: Exit confidence weights removed - we use only Tactician Ensemble confidence
                        'tactician_exit_confidence_weight': optimized_params.get('tactician_exit_confidence_weight', self.optimization_params.get('tactician_exit_confidence_weight', 0.6)),  # Deprecated
                        'analyst_exit_confidence_weight': optimized_params.get('analyst_exit_confidence_weight', self.optimization_params.get('analyst_exit_confidence_weight', 0.4)),  # Deprecated
                        # Note: exit_confidence_combination_method removed - we use only Tactician Ensemble confidence
                        # Additional parameters from optimization
                        'confidence_threshold': optimized_params.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD),
                        'position_sizing_factor': optimized_params.get('position_sizing_factor', 0.02),
                        'leverage_multiplier': optimized_params.get('leverage_multiplier', 1.5),
                        'stop_loss_pct': optimized_params.get('stop_loss_pct', 0.03),
                        'take_profit_pct': optimized_params.get('take_profit_pct', 0.06),
                        'ensemble_weight_analyst': optimized_params.get('ensemble_weight_analyst', 0.6),
                        'ensemble_weight_tactician': optimized_params.get('ensemble_weight_tactician', 0.4)
                    })
                    
                    # Load exit_strategy parameters from final_parameters_optimization
                    exit_strategy_results = optimized_params.get('exit_strategy', {})
                    if isinstance(exit_strategy_results, dict):
                        exit_strategy_params = exit_strategy_results.get('best_params', {})
                        if exit_strategy_params:
                            # Store exit strategy parameters for use in exit condition checking
                            self.optimization_params['exit_strategy'] = exit_strategy_params
                            tprint_success(f"✅ Loaded exit_strategy parameters: {len(exit_strategy_params)} parameters")
                            self.logger.info(f"✅ Loaded exit_strategy parameters: {len(exit_strategy_params)} parameters")
                        else:
                            # Try to get from position_monitor_exit_strategy format
                            position_monitor_exit = optimized_params.get('position_monitor_exit_strategy', {})
                            if position_monitor_exit:
                                self.optimization_params['exit_strategy'] = position_monitor_exit
                                tprint_success("✅ Loaded exit_strategy from position_monitor_exit_strategy format")
                                self.logger.info("✅ Loaded exit_strategy from position_monitor_exit_strategy format")
                    
                    # Also check for raw exit_strategy parameters directly in optimized_params
                    if 'exit_strategy' not in self.optimization_params or not self.optimization_params['exit_strategy']:
                        # Check if exit parameters are at top level
                        if any(key.startswith('exit_') or key.startswith('confidence_') for key in optimized_params.keys()):
                            # Extract exit-related parameters
                            exit_params = {k: v for k, v in optimized_params.items() 
                                         if k.startswith('exit_') or k.startswith('confidence_') or k in 
                                         ['base_profit_target', 'base_stop_loss', 'max_hold_time',
                                          'component_confidence_drop', 'profit_trailing_percent',
                                          'trailing_atr_multiplier', 'profit_buffer_ratio']}
                            if exit_params:
                                self.optimization_params['exit_strategy'] = exit_params
                                tprint_success(f"✅ Loaded exit_strategy parameters from top-level keys: {len(exit_params)} parameters")
                                self.logger.info(f"✅ Loaded exit_strategy parameters from top-level keys: {len(exit_params)} parameters")
                    
                    tprint_success("✅ Optimization parameters loaded successfully - using optimized values")
                    self.logger.info("✅ Loaded optimized parameters from final_parameters_optimization")
                    self.logger.debug(f"   Regime threshold: {self.optimization_params['regime_confidence_threshold']}")
                    self.logger.debug(f"   Signal threshold: {self.optimization_params['signal_confidence_threshold']}")
                    self.logger.debug(f"   Exit threshold: {self.optimization_params['exit_confidence_threshold']}")
                else:
                    tprint_warning("⚠️⚠️⚠️ OPTIMIZATION PARAMETERS NOT FOUND ⚠️⚠️⚠️")
                    tprint_warning("⚠️ No optimized parameters found in final_parameters_optimization!")
                    tprint_warning("⚠️ Using DEFAULT values - this may not be optimal for trading!")
                    tprint_warning("⚠️ Please ensure final_parameters_optimization has been run and parameters are available.")
                    self.logger.warning("⚠️ No optimized parameters found, using defaults")

            except Exception as e:
                tprint_warning("⚠️⚠️⚠️ FAILED TO LOAD OPTIMIZATION PARAMETERS ⚠️⚠️⚠️")
                tprint_warning(f"⚠️ Error loading optimization parameters: {e}")
                tprint_warning("⚠️ Using DEFAULT values - this may not be optimal for trading!")
                self.logger.warning(f"⚠️ Failed to load optimized parameters: {e}")

            self.logger.info("✅ Optimization parameters loaded")

        except Exception as e:
            tprint_warning("⚠️⚠️⚠️ CRITICAL: FAILED TO LOAD OPTIMIZATION PARAMETERS ⚠️⚠️⚠️")
            tprint_warning(f"⚠️ Critical error during parameter loading: {e}")
            tprint_warning("⚠️ Using DEFAULT values - this may not be optimal for trading!")
            self.logger.warning(f"⚠️ Failed to load optimization parameters, using defaults: {e}")
    
    async def _validate_feature_engineering(self):
        """Validate feature engineering setup and log expected features."""
        try:
            self.logger.info("🔍 Validating feature engineering setup...")
            
            # Log expected engineered features for Analyst
            analyst_features = self.analyst_feature_engineer.get_engineered_feature_names()
            self.logger.info(f"📊 Analyst engineered features: {analyst_features}")
            
            # Log expected engineered features for Tactician
            tactician_features = self.tactician_feature_engineer.get_engineered_feature_names()
            self.logger.info(f"📊 Tactician engineered features: {tactician_features}")
            
            # Note: Full feature validation will be done when models are loaded
            # if model metadata includes expected feature names
            self.logger.info("✅ Feature engineering setup validated")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineering validation warning: {e}")
    
    def validate_features_for_prediction(
        self,
        market_data: pd.DataFrame,
        role: str = "analyst"
    ) -> bool:
        """
        Validate that market_data has required columns for feature engineering.
        
        Args:
            market_data: Market data DataFrame
            role: Role name ('analyst' or 'tactician')
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            if role.lower() == "analyst":
                required = ['close']  # Minimum required for analyst features
            else:
                required = ['close']  # Minimum required for tactician features
            
            missing = [col for col in required if col not in market_data.columns]
            if missing:
                self.logger.warning(f"⚠️ Missing required columns for {role}: {missing}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Feature validation failed: {e}")
            return False
    
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

        # Step 3: Analyst Ensemble Model (combines base model predictions)
        analyst_output = await self._run_analyst_ensemble(
            market_data, regime_output, analyst_base_outputs, timestamp
        )

        # Step 4: Tactician Base Models
        tactician_base_outputs = await self._run_tactician_base_models(
            market_data, regime_output, analyst_output, timestamp
        )

        # Step 5: Tactician Ensemble Model (combines base model predictions)
        tactician_output = await self._run_tactician_ensemble(
            market_data, regime_output, analyst_output, tactician_base_outputs, timestamp
        )

        # Step 6: Calculate Exit Confidence (for position management)
        # Use only Tactician's Ensemble confidence for exit decisions
        exit_confidence = self._calculate_exit_confidence(
            analyst_output.analyst_confidence,  # Not used, kept for API compatibility
            tactician_output.tactician_confidence  # Tactician's Ensemble confidence
        )

        # Step 7: Check Exit Conditions (if position is open) - using comprehensive exit parameters
        should_exit, exit_reason = self._check_exit_conditions(
            exit_confidence,
            analyst_output.analyst_confidence,
            tactician_output.tactician_confidence,
            market_data,
            timestamp
        )

        # Step 8: Final Signal Generation (with position validation)
        final_signal = self._generate_final_signal(
            regime_output, analyst_output, tactician_output, should_exit, exit_reason
        )

        # Validate signal against current position
        validation_result = self._validate_signal_against_position(final_signal)
        if not validation_result['is_valid']:
            self.logger.warning(f"⚠️ Signal validation failed: {validation_result['reason']}")
            # Adjust signal if needed
            if validation_result.get('adjusted_signal'):
                final_signal = validation_result['adjusted_signal']

        # Update position state based on signal (thread-safe)
        self._update_position_state(
            final_signal, timestamp, should_exit,
            analyst_output.analyst_confidence,
            tactician_output.tactician_confidence
        )

        # Create result
        result = SignalGenerationResult(
            timestamp=timestamp,
            symbol=symbol,
            regime_output=regime_output,
            analyst_output=analyst_output,
            tactician_output=tactician_output,
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
            regime_prediction = await self.regime_detector.predict_regime(market_data, return_probabilities=True)
            
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
            
            # Convert regime_probabilities from {"regime_0": 0.5, ...} to {RegimeType: float}
            raw_probabilities = regime_prediction.get('regime_probabilities', {})
            regime_probabilities_dict: Dict[RegimeType, float] = {}
            
            if isinstance(raw_probabilities, dict):
                regime_list = list(RegimeType)
                for key, value in raw_probabilities.items():
                    try:
                        # Try to parse key as "regime_0", "regime_1", etc.
                        if isinstance(key, str) and key.startswith('regime_'):
                            regime_index = int(key.split('_')[1])
                            if 0 <= regime_index < len(regime_list):
                                regime_probabilities_dict[regime_list[regime_index]] = float(value)
                        # Try to match key directly to RegimeType value
                        elif isinstance(key, str):
                            try:
                                regime_type = RegimeType(key)
                                regime_probabilities_dict[regime_type] = float(value)
                            except ValueError:
                                # Try to find by value match
                                for rt in RegimeType:
                                    if rt.value == key:
                                        regime_probabilities_dict[rt] = float(value)
                                        break
                        # If key is already a RegimeType
                        elif isinstance(key, RegimeType):
                            regime_probabilities_dict[key] = float(value)
                        # If key is an integer index
                        elif isinstance(key, int):
                            if 0 <= key < len(regime_list):
                                regime_probabilities_dict[regime_list[key]] = float(value)
                    except (ValueError, IndexError, KeyError) as e:
                        self.logger.debug(f"Could not convert regime probability key {key}: {e}")
                        continue
            
            # If no probabilities were converted, create default distribution
            if not regime_probabilities_dict:
                self.logger.warning("⚠️ No regime probabilities could be converted, using default distribution")
                # Assign primary_regime a high probability, others low
                for regime in RegimeType:
                    if regime == primary_regime:
                        regime_probabilities_dict[regime] = 0.7
                    else:
                        regime_probabilities_dict[regime] = 0.3 / (len(RegimeType) - 1)
            else:
                # Normalize probabilities to sum to 1.0
                total_prob = sum(regime_probabilities_dict.values())
                if total_prob > 0:
                    regime_probabilities_dict = {
                        k: v / total_prob for k, v in regime_probabilities_dict.items()
                    }
            
            return RegimeOutput(
                timestamp=timestamp,
                regime_probabilities=regime_probabilities_dict,
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
        """Step 2: Run analyst base models sequentially with regime probabilities."""
        try:
            base_outputs = []

            # Prepare regime probabilities as features for base models
            # Convert RegimeType keys to a format models can use
            regime_probs_array = np.array([
                regime_output.regime_probabilities.get(rt, 0.0) 
                for rt in RegimeType
            ])

            # Run the trained analyst base models from training steps
            # These are the models trained in analyst_models_training_refactored.py
            # Use model selection result if available
            selected_analyst_model = None
            if model_selection_result and 'analyst' in model_selection_result.selected_models:
                selected_analyst_model = model_selection_result.selected_models['analyst']
                self.logger.info(f"🎯 Using selected analyst model: {selected_analyst_model}")

            for i, model in enumerate(self.analyst_base_models):
                try:
                    # Prepare input: features + regime probabilities
                    # Combine market_data features with regime probabilities
                    regime_probs_values = np.array([
                        regime_output.regime_probabilities.get(rt, 0.0) 
                        for rt in RegimeType
                    ])
                    
                    # Create enhanced features by combining market_data with regime probabilities
                    # Use shared feature engineering for consistency with training
                    if hasattr(model, 'predict'):
                        # Prepare market_data DataFrame (use last row for single prediction)
                        if isinstance(market_data, pd.DataFrame):
                            # Use last row for prediction
                            market_data_row = market_data.iloc[[-1]].copy()
                            
                            # Apply shared feature engineering (same as training)
                            # Extract primary regime probability for feature engineering
                            primary_regime_prob = max(regime_output.regime_probabilities.values()) if regime_output.regime_probabilities else 0.5
                            
                            # Engineer features using shared module
                            engineered_data = self.analyst_feature_engineer.engineer_features(
                                market_data_row,
                                regime_probability=primary_regime_prob
                            )
                            
                            # Extract all numeric features (including engineered ones)
                            numeric_data = engineered_data.select_dtypes(include=[np.number])
                            if len(numeric_data) > 0:
                                market_features = numeric_data.iloc[-1].values
                            else:
                                market_features = np.array([])
                        else:
                            market_features = np.array([])
                        
                        # Combine market features (now includes engineered features) with regime probabilities
                        combined_features = np.concatenate([market_features, regime_probs_values]) if len(market_features) > 0 else regime_probs_values
                        
                        # Reshape to (1, n_features) for single prediction
                        if combined_features.ndim == 1:
                            combined_features = combined_features.reshape(1, -1)
                        
                        # Predict with combined features (market_data + regime probabilities)
                        # Some models might accept the original market_data + regime_probs separately
                        # Try combined first, fallback to market_data if needed
                        try:
                            prediction = model.predict(combined_features)
                            self.logger.debug(
                                f"Analyst base model {i}: used combined features "
                                f"({len(market_features)} market + {len(regime_probs_values)} regime = {combined_features.shape[1]} total)"
                            )
                        except Exception:
                            # Fallback: try with market_data only (model might handle regime internally)
                            prediction = model.predict(market_data)
                            self.logger.debug(
                                f"Analyst base model {i}: fallback to market_data only "
                                f"(model may handle regime probabilities internally)"
                            )
                        
                        confidence = getattr(prediction, 'confidence', 0.5) if hasattr(prediction, 'confidence') else 0.5
                        if isinstance(prediction, np.ndarray) and prediction.size == 1:
                            confidence = float(prediction[0])
                        elif isinstance(prediction, (int, float)):
                            confidence = float(prediction)
                        
                        features = getattr(prediction, 'features', {}) if hasattr(prediction, 'features') else {}
                        
                        # Store regime information in features dict
                        features['regime_probabilities'] = dict(regime_output.regime_probabilities)
                        features['primary_regime'] = regime_output.primary_regime.value
                        features['regime_confidence'] = regime_output.confidence
                    else:
                        # Fallback for models without standard predict interface
                        self.logger.warning(f"⚠️ Analyst base model {i} missing 'predict' method, using fallback confidence")
                        confidence = 0.5
                        features = {
                            'regime_probabilities': regime_output.regime_probabilities,
                            'primary_regime': regime_output.primary_regime.value,
                            'regime_confidence': regime_output.confidence
                        }

                    # Create base output with regime information
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
                    self.logger.warning(f"⚠️ Analyst base model {i} failed: {e}", exc_info=True)
                    # Create fallback output with explicit logging
                    self.logger.debug(f"Creating fallback output due to model failure")
                    base_outputs.append(AnalystBaseOutput(
                        timestamp=timestamp,
                        market_health={},
                        volatility_analysis={},
                        liquidity_analysis={},
                        stress_analysis={},
                        base_confidence=0.5,
                        features={
                            'regime_probabilities': regime_output.regime_probabilities,
                            'primary_regime': regime_output.primary_regime.value,
                            'regime_confidence': regime_output.confidence
                        }
                    ))

            return base_outputs

        except Exception as e:
            self.logger.error(f"❌ Analyst base models failed: {e}")
            raise

    async def _run_analyst_ensemble(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        base_outputs: List[AnalystBaseOutput],
        timestamp: datetime
    ) -> AnalystMetaOutput:
        """
        Step 3: Run analyst ensemble model with base predictions.
        
        Architecture:
        - Base Models: Individual ML models (CatBoost, XGBoost, LightGBM, etc.) trained separately
        - Ensemble Model: Trained ML stacker model that learns optimal combination of base model predictions
        
        Flow:
        1. Base models produce predictions (already done in _run_analyst_base_models)
        2. Ensemble ML model receives: market features + regime probabilities + base model outputs
        3. Ensemble produces confidence directly (no meta model combination)
        
        Inputs to Analyst Ensemble:
        - Market data features (from market_data DataFrame)
        - Regime probabilities (from regime detector)
        - Base model outputs (confidences from all base models)
        """
        try:
            # Prepare inputs for ensemble model: features + regime probabilities + base model outputs
            # Use shared feature engineering (same as base models)
            # 1. Prepare market_data features with engineered features
            if isinstance(market_data, pd.DataFrame):
                # Use last row for prediction
                market_data_row = market_data.iloc[[-1]].copy()
                
                # Apply shared feature engineering (same as training and base models)
                primary_regime_prob = max(regime_output.regime_probabilities.values()) if regime_output.regime_probabilities else 0.5
                
                engineered_data = self.analyst_feature_engineer.engineer_features(
                    market_data_row,
                    regime_probability=primary_regime_prob
                )
                
                # Extract all numeric features (including engineered ones)
                numeric_data = engineered_data.select_dtypes(include=[np.number])
                if len(numeric_data) > 0:
                    market_features = numeric_data.iloc[-1].values
                else:
                    market_features = np.array([])
            else:
                market_features = np.array([])
            
            # 2. Prepare regime probabilities
            regime_probs_values = np.array([
                regime_output.regime_probabilities.get(rt, 0.0) 
                for rt in RegimeType
            ])
            
            # 3. Collect base model predictions/outputs
            base_confidences = [output.base_confidence for output in base_outputs]
            base_predictions_array = np.array(base_confidences).reshape(1, -1) if base_confidences else np.array([]).reshape(1, 0)
            
            # Combine all inputs: market_features + regime_probs + base_outputs
            ensemble_input_parts = []
            if len(market_features) > 0:
                ensemble_input_parts.append(market_features)
            ensemble_input_parts.append(regime_probs_values)
            if len(base_predictions_array.flatten()) > 0:
                ensemble_input_parts.append(base_predictions_array.flatten())
            
            ensemble_input = np.concatenate(ensemble_input_parts) if ensemble_input_parts else np.array([])
            ensemble_input = ensemble_input.reshape(1, -1) if ensemble_input.ndim == 1 else ensemble_input
            
            self.logger.debug(
                f"Analyst ensemble input prepared: "
                f"{len(market_features)} market + {len(regime_probs_values)} regime + "
                f"{len(base_predictions_array.flatten())} base outputs = {ensemble_input.shape[1]} total features"
            )
            
            ensemble_confidence = None
            ensemble_features = {}
            
            # If ensemble model is available, use combined features as input
            if self.analyst_ensemble_model is not None and len(ensemble_input.flatten()) > 0:
                try:
                    if hasattr(self.analyst_ensemble_model, 'predict'):
                        ensemble_prediction = self.analyst_ensemble_model.predict(ensemble_input)
                        if hasattr(ensemble_prediction, 'confidence'):
                            ensemble_confidence = float(ensemble_prediction.confidence)
                        elif isinstance(ensemble_prediction, np.ndarray):
                            # If array, use mean or max
                            ensemble_confidence = float(np.mean(ensemble_prediction))
                        elif isinstance(ensemble_prediction, (int, float)):
                            ensemble_confidence = float(ensemble_prediction)
                        
                        # Try to get probabilities if available
                        if hasattr(self.analyst_ensemble_model, 'predict_proba'):
                            ensemble_proba = self.analyst_ensemble_model.predict_proba(ensemble_input)
                            ensemble_features['ensemble_probabilities'] = ensemble_proba
                    
                    self.logger.debug(f"✅ Analyst ensemble model prediction: {ensemble_confidence}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Analyst ensemble model prediction failed: {e}")
            
            # Use ensemble confidence, or fallback to average of base confidences
            if ensemble_confidence is not None:
                analyst_confidence = ensemble_confidence
                self.logger.debug(f"Using analyst ensemble confidence: {analyst_confidence:.3f}")
            elif base_confidences:
                # Fallback: average of base model confidences
                analyst_confidence = float(np.mean(base_confidences))
                self.logger.debug(f"Using average base model confidence: {analyst_confidence:.3f}")
            else:
                # Ultimate fallback
                analyst_confidence = 0.5
                self.logger.warning("⚠️ No analyst predictions available, using default confidence 0.5")

            # Apply regime adjustment
            regime_adjusted_confidence = self._apply_regime_adjustment(
                analyst_confidence,
                regime_output.regime_probabilities
            )
            
            # Prepare meta features
            meta_features = {
                'regime_probabilities': regime_output.regime_probabilities,
                'primary_regime': regime_output.primary_regime.value,
                'regime_confidence': regime_output.confidence,
                'base_model_count': len(base_outputs),
                'ensemble_model_used': ensemble_confidence is not None
            }
            
            # Add ensemble features
            if ensemble_features:
                meta_features.update(ensemble_features)

            return AnalystMetaOutput(
                timestamp=timestamp,
                analyst_confidence=regime_adjusted_confidence,
                market_health_score=0.5,  # Not calculated without meta model
                regime_adjusted_confidence=regime_adjusted_confidence,
                meta_features=meta_features,
                base_outputs=base_outputs
            )

        except Exception as e:
            self.logger.error(f"❌ Analyst ensemble failed: {e}")
            raise

    async def _run_tactician_base_models(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        analyst_output: AnalystMetaOutput,
        timestamp: datetime
    ) -> List[TacticianBaseOutput]:
        """Step 4: Run tactician base models sequentially with regime probabilities and analyst outputs."""
        try:
            base_outputs = []

            # Prepare inputs for tactician models:
            # - Regime probabilities
            # - Analyst confidence and outputs
            analyst_confidence = analyst_output.analyst_confidence
            analyst_features = analyst_output.meta_features

            # Run the trained tactician base models from training steps
            # These are the models trained in tactician_models_training_refactored.py
            for i, model in enumerate(self.tactician_base_models):
                try:
                    # Prepare input: features + regime probabilities + analyst ensemble outputs
                    # Use shared feature engineering for consistency with training
                    # 1. Market data features (with engineered features)
                    if isinstance(market_data, pd.DataFrame):
                        # Use last row for prediction
                        market_data_row = market_data.iloc[[-1]].copy()
                        
                        # Apply shared feature engineering (same as training)
                        # Extract primary regime probability for feature engineering
                        primary_regime_prob = max(regime_output.regime_probabilities.values()) if regime_output.regime_probabilities else 0.5
                        
                        engineered_data = self.tactician_feature_engineer.engineer_features(
                            market_data_row,
                            regime_probability=primary_regime_prob,
                            timestamp=timestamp,
                            analyst_confidence=analyst_confidence,
                            analyst_outputs={
                                'analyst_confidence': analyst_confidence,
                                'market_health_score': analyst_output.market_health_score,
                                'regime_adjusted_confidence': analyst_output.regime_adjusted_confidence
                            }
                        )
                        
                        # Extract all numeric features (including engineered ones)
                        numeric_data = engineered_data.select_dtypes(include=[np.number])
                        if len(numeric_data) > 0:
                            market_features = numeric_data.iloc[-1].values
                        else:
                            market_features = np.array([])
                    else:
                        market_features = np.array([])
                    
                    # 2. Regime probabilities
                    regime_probs_values = np.array([
                        regime_output.regime_probabilities.get(rt, 0.0) 
                        for rt in RegimeType
                    ])
                    
                    # 3. Analyst ensemble outputs (confidence + features)
                    analyst_ensemble_inputs = np.array([
                        analyst_confidence,
                        analyst_output.market_health_score,
                        analyst_output.regime_adjusted_confidence
                    ])
                    
                    # Combine all inputs for tactician base models
                    combined_features_parts = []
                    if len(market_features) > 0:
                        combined_features_parts.append(market_features)
                    combined_features_parts.append(regime_probs_values)
                    combined_features_parts.append(analyst_ensemble_inputs)
                    
                    combined_features = np.concatenate(combined_features_parts) if combined_features_parts else np.array([])
                    if combined_features.ndim == 1:
                        combined_features = combined_features.reshape(1, -1)
                    
                    # Use the trained model to make predictions
                    if hasattr(model, 'predict'):
                        # Predict with combined features (market_data + regime probabilities + analyst ensemble outputs)
                        # Try combined features first, fallback to market_data if needed
                        try:
                            prediction = model.predict(combined_features)
                            self.logger.debug(
                                f"Tactician base model {i}: used combined features "
                                f"({len(market_features)} market + {len(regime_probs_values)} regime + "
                                f"{len(analyst_ensemble_inputs)} analyst = {combined_features.shape[1]} total)"
                            )
                        except Exception:
                            prediction = model.predict(market_data)
                            self.logger.debug(
                                f"Tactician base model {i}: fallback to market_data only "
                                f"(model may handle additional inputs internally)"
                            )
                        
                        confidence = getattr(prediction, 'confidence', 0.5) if hasattr(prediction, 'confidence') else 0.5
                        if isinstance(prediction, np.ndarray) and prediction.size == 1:
                            confidence = float(prediction[0])
                        elif isinstance(prediction, (int, float)):
                            confidence = float(prediction)
                        
                        scenario_predictions = getattr(prediction, 'scenario_predictions', {}) if hasattr(prediction, 'scenario_predictions') else {}
                        price_targets = getattr(prediction, 'price_targets', {}) if hasattr(prediction, 'price_targets') else {}
                        adversarial_risks = getattr(prediction, 'adversarial_risks', {}) if hasattr(prediction, 'adversarial_risks') else {}
                        
                        # Store regime and analyst information in scenario predictions
                        scenario_predictions['regime_probabilities'] = dict(regime_output.regime_probabilities)
                        scenario_predictions['primary_regime'] = regime_output.primary_regime.value
                        scenario_predictions['analyst_confidence'] = analyst_confidence
                        scenario_predictions['analyst_market_health_score'] = analyst_output.market_health_score
                        if analyst_features:
                            scenario_predictions['analyst_features'] = analyst_features
                    else:
                        # Fallback for models without standard predict interface
                        self.logger.warning(f"⚠️ Tactician base model {i} missing 'predict' method, using fallback confidence")
                        confidence = 0.5
                        scenario_predictions = {
                            'regime_probabilities': regime_output.regime_probabilities,
                            'primary_regime': regime_output.primary_regime.value,
                            'analyst_confidence': analyst_confidence
                        }
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
                    self.logger.warning(f"⚠️ Tactician base model {i} failed: {e}", exc_info=True)
                    # Create fallback output with explicit logging
                    self.logger.debug(f"Creating fallback output due to model failure")
                    base_outputs.append(TacticianBaseOutput(
                        timestamp=timestamp,
                        scenario_predictions={
                            'regime_probabilities': regime_output.regime_probabilities,
                            'primary_regime': regime_output.primary_regime.value,
                            'analyst_confidence': analyst_confidence
                        },
                        price_targets={},
                        adversarial_risks={},
                        base_confidence=0.5,
                        position_recommendations={}
                    ))

            return base_outputs

        except Exception as e:
            self.logger.error(f"❌ Tactician base models failed: {e}")
            raise

    async def _run_tactician_ensemble(
        self,
        market_data: pd.DataFrame,
        regime_output: RegimeOutput,
        analyst_output: AnalystMetaOutput,
        base_outputs: List[TacticianBaseOutput],
        timestamp: datetime
    ) -> TacticianMetaOutput:
        """
        Step 5: Run tactician ensemble model with base predictions.
        
        Architecture:
        - Base Models: Individual ML models (CatBoost, XGBoost, LightGBM, etc.) trained separately
        - Ensemble Model: Trained ML stacker model that learns optimal combination of base model predictions
        
        Flow:
        1. Base models produce predictions with analyst outputs (already done in _run_tactician_base_models)
        2. Ensemble ML model receives: market features + regime probabilities + analyst ensemble outputs + base model outputs
        3. Ensemble produces confidence directly (no meta model combination)
        4. Generate final signal based on confidence thresholds
        
        Inputs to Tactician Ensemble:
        - Market data features (from market_data DataFrame)
        - Regime probabilities (from regime detector)
        - Analyst ensemble outputs (confidence, market_health_score, regime_adjusted_confidence)
        - Base model outputs (confidences from all tactician base models)
        """
        try:
            # Prepare inputs for tactician ensemble: features + regime probabilities + analyst ensemble outputs + tactician base outputs
            # 1. Market data features
            if isinstance(market_data, pd.DataFrame):
                numeric_data = market_data.select_dtypes(include=[np.number])
                if len(numeric_data) > 0:
                    market_features = numeric_data.iloc[-1].values
                else:
                    market_features = np.array([])
            else:
                market_features = np.array([])
            
            # 2. Regime probabilities
            regime_probs_values = np.array([
                regime_output.regime_probabilities.get(rt, 0.0) 
                for rt in RegimeType
            ])
            
            # 3. Analyst ensemble outputs
            analyst_ensemble_inputs = np.array([
                analyst_output.analyst_confidence,
                analyst_output.market_health_score,
                analyst_output.regime_adjusted_confidence
            ])
            
            # 4. Tactician base model outputs
            base_confidences = [output.base_confidence for output in base_outputs]
            base_predictions_array = np.array(base_confidences).reshape(1, -1) if base_confidences else np.array([]).reshape(1, 0)
            
            # Combine all inputs: market_features + regime_probs + analyst_outputs + base_outputs
            ensemble_input_parts = []
            if len(market_features) > 0:
                ensemble_input_parts.append(market_features)
            ensemble_input_parts.append(regime_probs_values)
            ensemble_input_parts.append(analyst_ensemble_inputs)
            if len(base_predictions_array.flatten()) > 0:
                ensemble_input_parts.append(base_predictions_array.flatten())
            
            ensemble_input = np.concatenate(ensemble_input_parts) if ensemble_input_parts else np.array([])
            ensemble_input = ensemble_input.reshape(1, -1) if ensemble_input.ndim == 1 else ensemble_input
            
            self.logger.debug(
                f"Tactician ensemble input prepared: "
                f"{len(market_features)} market + {len(regime_probs_values)} regime + "
                f"{len(analyst_ensemble_inputs)} analyst + {len(base_predictions_array.flatten())} base = "
                f"{ensemble_input.shape[1]} total features"
            )
            
            ensemble_confidence = None
            ensemble_features = {}
            
            # If ensemble model is available, use combined features as input
            if self.tactician_ensemble_model is not None and len(ensemble_input.flatten()) > 0:
                try:
                    if hasattr(self.tactician_ensemble_model, 'predict'):
                        ensemble_prediction = self.tactician_ensemble_model.predict(ensemble_input)
                        if hasattr(ensemble_prediction, 'confidence'):
                            ensemble_confidence = float(ensemble_prediction.confidence)
                        elif isinstance(ensemble_prediction, np.ndarray):
                            # If array, use mean or max
                            ensemble_confidence = float(np.mean(ensemble_prediction))
                        elif isinstance(ensemble_prediction, (int, float)):
                            ensemble_confidence = float(ensemble_prediction)
                        
                        # Try to get probabilities if available
                        if hasattr(self.tactician_ensemble_model, 'predict_proba'):
                            ensemble_proba = self.tactician_ensemble_model.predict_proba(ensemble_input)
                            ensemble_features['ensemble_probabilities'] = ensemble_proba
                    
                    self.logger.debug(f"✅ Tactician ensemble model prediction: {ensemble_confidence}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Tactician ensemble model prediction failed: {e}")
            
            # Use ensemble confidence - REQUIRED, no fallbacks allowed
            if ensemble_confidence is not None:
                tactician_confidence = ensemble_confidence
                self.logger.debug(f"Using tactician ensemble confidence: {tactician_confidence:.3f}")
            else:
                # Tactician ensemble confidence is required - raise error if not available
                error_msg = "Tactician ensemble model confidence not available. This is required for trading decisions."
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Use Tactician confidence directly (no combination with Analyst)
            # combined_confidence is kept for backward compatibility but equals tactician_confidence
            combined_confidence = tactician_confidence

            # Generate final signal based on confidence and thresholds
            # Extract signal direction from base outputs if available
            final_signal = 'hold'
            signal_strength = 0.5
            
            # Try to infer signal from base outputs
            if base_outputs:
                # Look for scenario predictions that might indicate direction
                for output in base_outputs:
                    scenario_preds = output.scenario_predictions
                    if scenario_preds:
                        bullish = scenario_preds.get('bullish_probability', 0.0)
                        bearish = scenario_preds.get('bearish_probability', 0.0)
                        if bullish > 0.6 and bullish > bearish:
                            final_signal = 'buy'
                            signal_strength = bullish
                            break
                        elif bearish > 0.6 and bearish > bullish:
                            final_signal = 'sell'
                            signal_strength = bearish
                            break
            
            # If still hold and confidence is high, use confidence to determine signal
            if final_signal == 'hold' and tactician_confidence > 0.7:
                # Default to buy if high confidence (can be refined based on price action)
                final_signal = 'buy'
                signal_strength = tactician_confidence
            
            # Prepare meta features
            meta_features = {
                'regime_probabilities': regime_output.regime_probabilities,
                'primary_regime': regime_output.primary_regime.value,
                'regime_confidence': regime_output.confidence,
                'analyst_confidence': analyst_output.analyst_confidence,
                'base_model_count': len(base_outputs),
                'ensemble_model_used': ensemble_confidence is not None
            }
            
            # Add ensemble features
            if ensemble_features:
                meta_features.update(ensemble_features)

            return TacticianMetaOutput(
                timestamp=timestamp,
                tactician_confidence=tactician_confidence,
                combined_confidence=combined_confidence,
                final_signal=final_signal,
                signal_strength=signal_strength,
                meta_features=meta_features,
                base_outputs=base_outputs
            )

        except Exception as e:
            self.logger.error(f"❌ Tactician ensemble failed: {e}")
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

    # Note: _calculate_combined_confidence removed - we use only Tactician's Ensemble confidence
    # Tactician uses Analyst output as input, so combining confidences would cause overfitting.

    def _calculate_exit_confidence(self, analyst_confidence: float, tactician_confidence: float) -> float:
        """
        Return Tactician confidence for exit decisions (no combination with Analyst).
        
        Note: Tactician uses Analyst output as input, so combining confidences
        would cause overfitting. We use only Tactician's Ensemble confidence.

        Args:
            analyst_confidence: Current analyst confidence (not used, kept for API compatibility)
            tactician_confidence: Current tactician confidence (Tactician's Ensemble confidence)

        Returns:
            Tactician confidence value (no combination)
        """
        try:
            # Use only Tactician confidence (no combination)
            exit_confidence = max(0.0, min(1.0, tactician_confidence))

            self.logger.debug(f"📊 Exit confidence calculation (Tactician only):")
            self.logger.debug(f"   Tactician Ensemble: {tactician_confidence:.4f}")
            self.logger.debug(f"   Exit confidence: {exit_confidence:.4f}")

            return exit_confidence

        except Exception as e:
            self.logger.error(f"❌ Error calculating exit confidence: {e}")
            # If Tactician confidence is not available, return 0 and raise error
            raise ValueError(f"Tactician ensemble confidence required for exit decisions: {e}")

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

    def _check_exit_conditions(
        self,
        exit_confidence: float,
        analyst_confidence: float,
        tactician_confidence: float,
        market_data: pd.DataFrame,
        timestamp: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position should be exited based on comprehensive exit parameters from final_parameters_optimization.
        
        This matches the exit logic tested by final_parameters_optimization, checking:
        - Exit confidence thresholds
        - Stop-loss conditions
        - Time-based exit conditions (max hold time)
        - Confidence drop conditions (combined and individual component drops)
        - Profit trailing conditions (triggered after base_profit_target reached)
        
        Args:
            exit_confidence: Combined exit confidence from analyst and tactician
            analyst_confidence: Current analyst confidence
            tactician_confidence: Current tactician confidence
            market_data: Current market data for price/volatility checks
            timestamp: Current timestamp for time-based checks
        
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        try:
            with self._position_lock:
                current_pos = self.current_position
            
            # If no position is open, no need to exit
            if not current_pos or not current_pos.is_open:
                return False, None
            
            exit_strategy = self.optimization_params.get('exit_strategy', {})
            exit_reasons = []
            
            # Calculate current profit/loss percentage
            current_price = None
            profit_pct = None
            if current_pos.entry_price is not None and len(market_data) > 0:
                current_price = market_data['close'].iloc[-1]
                if current_pos.direction == 'long':
                    profit_pct = (current_price - current_pos.entry_price) / current_pos.entry_price
                else:  # short
                    profit_pct = (current_pos.entry_price - current_price) / current_pos.entry_price
                
                # Update peak profit tracking
                with self._position_lock:
                    if current_pos.peak_profit_pct is None or profit_pct > current_pos.peak_profit_pct:
                        current_pos.peak_profit_pct = profit_pct
            
            # 1. Exit confidence threshold check (primary check)
            exit_threshold = self.optimization_params.get('exit_confidence_threshold', DEFAULT_EXIT_CONFIDENCE_THRESHOLD)
            if exit_confidence < exit_threshold:
                exit_reasons.append(f"Exit confidence {exit_confidence:.3f} below threshold {exit_threshold:.3f}")
            
            # 2. Combined confidence drop check
            if current_pos.entry_confidence is not None:
                confidence_drop = current_pos.entry_confidence - exit_confidence
                # Check for confidence drop threshold
                exit_confidence_drop = exit_strategy.get('exit_confidence_drop') if isinstance(exit_strategy, dict) else None
                if exit_confidence_drop and confidence_drop >= exit_confidence_drop:
                    exit_reasons.append(f"Confidence drop {confidence_drop:.3f} >= threshold {exit_confidence_drop:.3f}")
            
            # 3. Time-based exit check (max hold time only - min hold time removed)
            if current_pos.entry_timestamp:
                elapsed_time = (timestamp - current_pos.entry_timestamp).total_seconds()
                
                if isinstance(exit_strategy, dict):
                    # Handle both formatted and raw formats
                    time_based = exit_strategy.get('time_based', {})
                    if time_based:
                        max_hold_time = time_based.get('max_hold_time', 10800)  # Default 3 hours
                    else:
                        # Check for raw format keys
                        max_hold_time = exit_strategy.get('max_hold_time', 10800)
                    
                    if elapsed_time >= max_hold_time:
                        exit_reasons.append(f"Maximum hold time exceeded: {elapsed_time:.0f}s >= {max_hold_time:.0f}s")
            
            # 4. Profit trailing check (triggered when base_profit_target reached)
            if profit_pct is not None and isinstance(exit_strategy, dict):
                base_profit_target = exit_strategy.get('base_profit_target')
                profit_trailing_percent = exit_strategy.get('profit_trailing_percent')
                
                if base_profit_target is not None and profit_trailing_percent is not None:
                    # Check if we've reached base target (once reached, trailing activates)
                    if current_pos.peak_profit_pct is not None and current_pos.peak_profit_pct >= base_profit_target:
                        # Calculate trailing stop: peak_profit - trailing_percent
                        trailing_stop_pct = current_pos.peak_profit_pct - profit_trailing_percent
                        
                        # Exit if current profit has dropped below trailing stop
                        if profit_pct < trailing_stop_pct:
                            exit_reasons.append(
                                f"Profit trailing stop triggered: current {profit_pct:.4f} < trailing stop {trailing_stop_pct:.4f} "
                                f"(peak: {current_pos.peak_profit_pct:.4f}, trailing: {profit_trailing_percent:.4f})"
                            )
            
            # 5. Stop-loss check (if position has loss)
            if current_pos.entry_price is not None and len(market_data) > 0:
                current_price = market_data['close'].iloc[-1]
                
                if current_pos.direction == 'long':
                    loss_pct = (current_pos.entry_price - current_price) / current_pos.entry_price
                else:  # short
                    loss_pct = (current_price - current_pos.entry_price) / current_pos.entry_price
                
                if isinstance(exit_strategy, dict):
                    # Handle both formatted and raw formats
                    stop_loss = exit_strategy.get('stop_loss', {})
                    if stop_loss:
                        base_stop_loss = abs(stop_loss.get('base_stop_loss', -0.05))
                    else:
                        # Check for raw format key
                        base_stop_loss = abs(exit_strategy.get('base_stop_loss', -0.05))
                    
                    if loss_pct >= base_stop_loss:
                        exit_reasons.append(f"Stop-loss triggered: {loss_pct:.3f} >= {base_stop_loss:.3f}")
            
            # 6. Tactician confidence drop check (using only Tactician Ensemble confidence)
            # Exit if Tactician confidence drops significantly from entry
            if current_pos.entry_tactician_confidence is not None:
                entry_tactician_conf = current_pos.entry_tactician_confidence
                tactician_drop = entry_tactician_conf - tactician_confidence if entry_tactician_conf > tactician_confidence else 0
                
                # Get component confidence drop threshold from optimization (backtested parameter)
                component_confidence_drop = exit_strategy.get('component_confidence_drop', 0.3) if isinstance(exit_strategy, dict) else 0.3
                
                # Exit if Tactician confidence drops significantly
                if tactician_drop >= component_confidence_drop:
                    exit_reasons.append(
                        f"Tactician ensemble confidence dropped significantly: {tactician_drop:.3f} >= threshold {component_confidence_drop:.3f}"
                    )
            
            # Determine if we should exit based on any exit reason
            if exit_reasons:
                # Primary exit reason is the first/most critical one
                exit_reason = exit_reasons[0]
                if len(exit_reasons) > 1:
                    exit_reason += f" (and {len(exit_reasons) - 1} other condition(s))"
                
                self.logger.info(f"🚪 Exit condition(s) triggered: {exit_reason}")
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

    def _update_position_state(
        self,
        final_signal: Dict[str, Any],
        timestamp: datetime,
        should_exit: bool,
        analyst_confidence: float,
        tactician_confidence: float
    ):
        """
        Update position state based on signal and exit conditions (thread-safe).

        Args:
            final_signal: Generated trading signal
            timestamp: Current timestamp
            should_exit: Whether position should be exited
            analyst_confidence: Current analyst confidence
            tactician_confidence: Current tactician confidence
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
                        entry_confidence=confidence,
                        entry_analyst_confidence=analyst_confidence,
                        entry_tactician_confidence=tactician_confidence
                    )
                    self.logger.info(
                        f"📈 New position opened: {signal} at {timestamp} "
                        f"(confidence: {confidence:.3f}, analyst: {analyst_confidence:.3f}, tactician: {tactician_confidence:.3f})"
                    )

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

            # Use optimization parameters for thresholds (overridden by final_parameters_optimization)
            regime_threshold = self.optimization_params.get('regime_confidence_threshold', DEFAULT_REGIME_CONFIDENCE_THRESHOLD)
            signal_threshold = self.optimization_params.get('signal_confidence_threshold', DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD)

            # Check regime confidence
            if regime_output.confidence < regime_threshold:
                return {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'reason': f'Low regime confidence: {regime_output.confidence:.3f} < {regime_threshold:.3f}'
                }

            # Check signal confidence (using only Tactician Ensemble confidence)
            tactician_confidence = tactician_output.tactician_confidence
            if tactician_confidence < signal_threshold:
                return {
                    'signal': 'hold',
                    'confidence': tactician_confidence,
                    'strength': 0.0,
                    'reason': f'Low Tactician ensemble confidence: {tactician_confidence:.3f} < {signal_threshold:.3f}'
                }

            # Validate signal based on analyst and tactician outputs
            validation_result = self._validate_signal(analyst_output, tactician_output)

            if not validation_result['is_valid']:
                return {
                    'signal': 'hold',
                    'confidence': tactician_confidence,
                    'strength': 0.0,
                    'reason': f'Signal validation failed: {validation_result["reason"]}'
                }

            # Generate signal based on tactician output (using only Tactician Ensemble confidence)
            final_signal = tactician_output.final_signal
            final_confidence = tactician_confidence  # Use Tactician Ensemble confidence directly
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

            for model in self.tactician_base_models:
                if hasattr(model, 'stop'):
                    await model.stop()

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
