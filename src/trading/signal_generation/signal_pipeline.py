"""
Signal Generation Pipeline (Refactored)

Architecture:
1.  **Specialist Layer:** Runs 7 specialist models (Risk, SMC, Liquidity, Breakout, Meso, Macro, Reversion).
2.  **Analyst Base Layer:**
    *   **LGBM:** Predicts signal direction/strength.
    *   **NGBoost:** Predicts uncertainty.
3.  **Decision Layer:** Combines signal and uncertainty for final trading decision.

No Tactician or Ensemble models.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from threading import Lock
import numpy as np
import pandas as pd

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.trading.config.trading_config import TradingConfig
from src.trading.config.regime_config import RegimeType
from src.core.decorators import handles_errors, traced, log_execution_time
from .utils import (
    CircuitBreaker, RateLimiter, SignalDeduplicator, 
    validate_market_data, validate_signal_parameters
)
from src.trading.integration.unified_model_loader import get_unified_model_loader
from src.trading.integration.regime_detector import HybridRegimeDetector
from src.feature_generation.shared.feature_engineer import AnalystFeatureEngineer
try:
    from src.trading.model_selection import ModelSelectionResult
except Exception:
    ModelSelectionResult = Any  # type: ignore

logger = system_logger.getChild('SignalGenerationPipeline')

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_REGIME_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_EXIT_CONFIDENCE_THRESHOLD = 0.5
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
    specialist_outputs: Dict[str, Any]

@dataclass
class SpecialistOutput:
    """Canonical scalar outputs from specialist models."""
    timestamp: datetime
    risk_score: float = 0.5
    smc_score: float = 0.5
    liquidity_score: float = 0.5
    breakout_score: float = 0.5
    meso_score: float = 0.5
    macro_score: float = 0.5
    reversion_score: float = 0.5
    raw_outputs: Dict[str, Any] = None

@dataclass
class AnalystOutput:
    """Analyst Layer output (LGBM Signal + NGBoost Uncertainty)."""
    timestamp: datetime
    signal_score: float  # Raw model output (can be logit or probability)
    signal_confidence: float  # Calibrated probability [0, 1]
    uncertainty: float  # NGBoost uncertainty [0, 1] (normalized)
    features: Dict[str, Any]

@dataclass
class PositionState:
    """Current position state."""
    is_open: bool = False
    entry_timestamp: Optional[datetime] = None
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    direction: Optional[str] = None  # 'long' or 'short'
    entry_confidence: Optional[float] = None
    entry_uncertainty: Optional[float] = None
    peak_profit_pct: Optional[float] = None
    max_hold_time_seconds: Optional[int] = None
    profit_target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None

@dataclass
class SignalGenerationResult:
    """Complete signal generation result."""
    timestamp: datetime
    symbol: str
    regime_output: RegimeOutput
    specialist_output: SpecialistOutput
    analyst_output: AnalystOutput
    final_signal: str  # 'buy', 'sell', 'hold', 'close'
    final_confidence: float
    signal_strength: float
    uncertainty: float
    optimization_parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    # Exit-specific fields
    exit_confidence: Optional[float] = None
    should_exit: bool = False
    exit_reason: Optional[str] = None
    position_state: Optional[PositionState] = None

class SignalGenerationPipeline:
    """
    Refactored Signal Generation Pipeline.

    Flow:
    1. Regime Detection (Hybrid)
    2. Specialist Models (7 types) -> Canonical Scalars
    3. Analyst Base Layer:
       - LGBM -> Signal/Confidence
       - NGBoost -> Uncertainty
    4. Decision Layer (Signal + Uncertainty + Exit Logic)
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('SignalGenerationPipeline')
        tprint("[SIGNAL_PIPELINE] Initializing refactored pipeline")

        # Components
        self.unified_loader = None
        self.regime_detector = None
        self.feature_engineer = AnalystFeatureEngineer(logger=self.logger)

        # Models
        self.specialist_models = {}
        self.analyst_lgbm = None
        self.analyst_ngboost = None

        # Optimization Parameters
        self.optimization_params = {
            'regime_confidence_threshold': DEFAULT_REGIME_CONFIDENCE_THRESHOLD,
            'signal_confidence_threshold': DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD,
            'exit_confidence_threshold': DEFAULT_EXIT_CONFIDENCE_THRESHOLD,
            'exit_strategy': {}
        }

        # State
        self.is_initialized = False
        self.signal_history: deque[SignalGenerationResult] = deque(maxlen=DEFAULT_MAX_HISTORY)
        self.current_position: Optional[PositionState] = None
        self.position_history: deque[PositionState] = deque(maxlen=DEFAULT_MAX_HISTORY)
        self._position_lock: Lock = Lock()

        # Utilities
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, 'circuit_breaker_failures', DEFAULT_CIRCUIT_BREAKER_FAILURES),
            recovery_timeout=getattr(config, 'circuit_breaker_timeout', DEFAULT_CIRCUIT_BREAKER_TIMEOUT)
        )
        self.rate_limiter = RateLimiter(
            max_calls=getattr(config, 'rate_limit_calls', DEFAULT_RATE_LIMIT_CALLS),
            time_window=getattr(config, 'rate_limit_window', DEFAULT_RATE_LIMIT_WINDOW)
        )
        self.signal_deduplicator = SignalDeduplicator(
            deduplication_window=getattr(config, 'signal_dedup_window', DEFAULT_SIGNAL_DEDUP_WINDOW)
        )

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize pipeline components and load models."""
        try:
            tprint("[SIGNAL_PIPELINE] initialize: Loading models...")
            
            # 1. Unified Loader
            self.unified_loader = get_unified_model_loader()
            
            symbol = getattr(self.config, 'symbol', 'ETHUSDT')
            exchange = getattr(self.config, 'exchange', 'binance')
            timeframe = getattr(self.config, 'timeframe', '15m')
            direction = getattr(self.config, 'direction', 'long')

            # 2. Load All Models
            models_data = await self.unified_loader.load_all_models(
                symbol=symbol,
                exchange=exchange,
                analyst_timeframe=timeframe,
                direction=direction
            )
            
            self.specialist_models = models_data.get('specialists', {})
            
            # Extract Analyst Base Models from generic dict
            analyst_base_models = models_data.get('analyst_base_models', {})
            self.analyst_lgbm = analyst_base_models.get('lgbm')
            self.analyst_ngboost = analyst_base_models.get('ngboost')
            
            # Warning if models missing
            if not self.analyst_lgbm:
                tprint("⚠️ LGBM model not found in analyst_base_models", "warning")
            if not self.analyst_ngboost:
                tprint("⚠️ NGBoost model not found in analyst_base_models", "warning")
            
            # 3. Load Optimization Parameters
            opt_params = models_data.get('optimized_parameters', {})
            if opt_params:
                self.optimization_params.update(opt_params)
                tprint("✅ Loaded optimized parameters")
            
            # 4. Initialize Regime Detector
            self.regime_detector = HybridRegimeDetector(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                base_timeframe=timeframe
            )
            
            self.is_initialized = True
            tprint("✅ Signal Generation Pipeline initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False

    @handles_errors
    @log_execution_time()
    @traced(span_name="generate_signal")
    async def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        additional_features: Optional[Dict[str, Any]] = None
    ) -> SignalGenerationResult:
        """Generate trading signal."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
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
                features_used=regime_prediction.get('features_used', {}),
                specialist_outputs=regime_prediction.get('specialist_outputs', {})
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
                regime_probs_values = np.array([
                    regime_output.regime_probabilities.get(rt, 0.0) 
                    for rt in RegimeType
                ])
                
                if hasattr(model, 'predict'):
                    # Prepare market_data DataFrame (use last row for single prediction)
                    if isinstance(market_data, pd.DataFrame):
                        market_data_row = market_data.iloc[[-1]].copy()

                        regime_probs_for_engineer: Dict[int, float] = {}
                        try:
                            regime_list = list(RegimeType)
                            for idx, rt in enumerate(regime_list[:4]):
                                regime_probs_for_engineer[idx] = float(
                                    regime_output.regime_probabilities.get(rt, 0.0)
                                )
                        except Exception:
                            regime_probs_for_engineer = {}

                        engineered_data = self.analyst_feature_engineer.engineer_features(
                            market_data_row,
                            regime_probabilities=regime_probs_for_engineer or None,
                        )

                        numeric_data = engineered_data.select_dtypes(include=[np.number])
                        market_features = numeric_data.iloc[-1].values if len(numeric_data) > 0 else np.array([])
                    else:
                        market_features = np.array([])
                    
                    combined_features = np.concatenate([market_features, regime_probs_values]) if len(market_features) > 0 else regime_probs_values
                    if combined_features.ndim == 1:
                        combined_features = combined_features.reshape(1, -1)
                    
                    try:
                        prediction = model.predict(combined_features)
                        self.logger.debug(
                            f"Analyst base model {i}: used combined features "
                            f"({len(market_features)} market + {len(regime_probs_values)} regime = {combined_features.shape[1]} total)"
                        )
                    except Exception:
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
                    
                    features['regime_probabilities'] = dict(regime_output.regime_probabilities)
                    features['primary_regime'] = regime_output.primary_regime.value
                    features['regime_confidence'] = regime_output.confidence
                else:
                    self.logger.warning(f"⚠️ Analyst base model {i} missing 'predict' method, using fallback confidence")
                    confidence = 0.5
                    features = {
                        'regime_probabilities': regime_output.regime_probabilities,
                        'primary_regime': regime_output.primary_regime.value,
                        'regime_confidence': regime_output.confidence
                    }

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
                self.logger.error(f"❌ Analyst base model {i} failed: {e}")
                continue

        # Validation & Rate Limiting
        validate_market_data(market_data)
        if not self.rate_limiter.acquire():
            raise RuntimeError("Rate limit exceeded")

        timestamp = datetime.now()

        try:
            # 1. Feature Engineering
            engineered_features = self.feature_engineer.engineer_features(market_data)
            
            # 2. Regime Detection
            regime_output_data = await self.regime_detector.predict_regime(market_data)
            regime_output = self._parse_regime_output(regime_output_data, timestamp)
            
            # 3. Specialist Inference
            specialist_output = self._run_specialist_models(market_data, engineered_features)
            
            # 4. Construct Analyst Input Vector
            analyst_input = self._prepare_analyst_input(
                engineered_features,
                specialist_output,
                regime_output
            )
            
            # 5. Analyst Inference (LGBM + NGBoost)
            analyst_output = self._run_analyst_models(analyst_input, timestamp)
 
            # Prepare specialist scalar features (SMC, Volume Force, Path Score, etc.)
            # These must match the order and presence of features used during training.
            # While the ensemble model likely expects a specific order, we can pass
            # a dictionary or dataframe if the model supports it, or construct the vector
            # if we know the order.
            # Assuming StandardizedXGBTrainer or similar was used which might rely on
            # feature names if input is DataFrame, or strict order if numpy.
            #
            # If the trained model expects specific specialist features that were present
            # in get_specialist_models_outputs, we should append them here.
            # Since we don't have the exact training column order here without the model metadata,
            # we rely on the fact that for many ML frameworks (like XGBoost/LightGBM with sklearn API),
            # passing a numpy array requires strict order, but we can try to extract
            # feature names from the model if available.

            # Extract scalar specialist features from RegimeOutput
            specialist_vals = []
            specialist_keys = []
            if regime_output.specialist_outputs:
                # We prioritize specific known scalar keys that match get_specialist_models_outputs
                # Priority list based on get_specialist_models_outputs.py:
                # 1. Risk Score (already partially covered by regime probs but often used as scalar)
                # 2. Liquidity (probs covered)
                # 3. Breakout/Bounce (resistance_scalar, support_scalar, breakout_success_prob)
                # 4. Path (path_risk_score)
                # 5. Meso (meso_trend_score_continuous)
                # 6. Macro (macro_trend_score_continuous)
                # 7. SMC (smc_predicted)
                # 8. Mean Reversion (mr_probability_dense or mr_probability)
                # 9. Volume Force (vol_force_scalar)

                # NOTE: The exact order MUST match training. If we append everything found,
                # it might be risky. Ideally, we should check model.feature_names_in_ if available.

                # For now, we will append them to a list but only if we can verify order or
                # if we are constructing a DataFrame.
                pass

            # Combine all inputs: market_features + regime_probs + base_outputs + meta_features
            ensemble_input_parts = []
            if len(market_features) > 0:
                ensemble_input_parts.append(market_features)
            ensemble_input_parts.append(regime_probs_values)
            if len(base_predictions_array.flatten()) > 0:
                ensemble_input_parts.append(base_predictions_array.flatten())
            if len(live_meta_features) > 0:
                ensemble_input_parts.append(live_meta_features)
            
            # Append specialist scalars if available and relevant (experimental)
            # This is a placeholder for where full wiring would happen if the ensemble
            # was retrained to expect them. Currently, to avoid breaking existing models
            # that expect a fixed input size, we do NOT blindly append them unless
            # we are sure the model handles named features (DataFrame input).

            ensemble_input = np.concatenate(ensemble_input_parts) if ensemble_input_parts else np.array([])

            # If the model supports named features (DataFrame), we should construct one.
            # This is much safer than guessing numpy array order.
            if hasattr(self.analyst_ensemble_model, "feature_names_in_"):
                try:
                    # Construct a dictionary of all available features
                    feature_dict = {}

                    # 1. Market features (we need names from feature engineer)
                    eng_names = self.analyst_feature_engineer.get_engineered_feature_names()
                    if len(market_features) == len(eng_names):
                        for i, name in enumerate(eng_names):
                            feature_dict[name] = market_features[i]

                    # 2. Regime probs
                    for i, rt in enumerate(RegimeType):
                        feature_dict[f"regime_prob_{rt.value}"] = regime_probs_values[i]
                        # Also add legacy/alternative names if needed by model
                        feature_dict[f"regime_{i}_prob"] = regime_probs_values[i]

                    # 3. Base predictions
                    for i, val in enumerate(base_confidences):
                        feature_dict[f"base_model_{i}"] = val

                    # 4. Meta features
                    # (Meta feature names need to be known. live_meta_features has 9 elements)
                    meta_names = [
                        "meta_disagreement_variance", "meta_disagreement_range", "meta_disagreement_std",
                        "meta_disagreement_entropy", "meta_disagreement_pairwise", "meta_disagreement_max",
                        "meta_disagreement_rate", "meta_bars_since_last_event", "meta_event_mean_return"
                    ]
                    for i, name in enumerate(meta_names):
                        if i < len(live_meta_features):
                            feature_dict[name] = live_meta_features[i]

                    # 5. Specialist Outputs
                    if regime_output.specialist_outputs:
                        feature_dict.update(regime_output.specialist_outputs)

                    # Create DataFrame with columns matching model's expected features
                    expected_features = self.analyst_ensemble_model.feature_names_in_
                    df_input = pd.DataFrame([feature_dict])

                    # Ensure all expected columns exist (fill 0 if missing)
                    for col in expected_features:
                        if col not in df_input.columns:
                            df_input[col] = 0.0

                    # Reorder to match model expectation
                    ensemble_input = df_input[expected_features]
                    self.logger.debug(f"constructed dataframe input with {len(expected_features)} features")

                except Exception as e:
                    self.logger.warning(f"Failed to construct named feature input: {e}; falling back to numpy array")
                    ensemble_input = ensemble_input.reshape(1, -1) if ensemble_input.ndim == 1 else ensemble_input
            else:
                ensemble_input = ensemble_input.reshape(1, -1) if ensemble_input.ndim == 1 else ensemble_input
            
            # 6. Ensemble voting / stacking
            ensemble_output = self._run_ensemble_models(
                analyst_output,
                specialist_output,
                regime_output,
                base_predictions_array,
            )
            
            # 6. Exit Logic (if position open)
            should_exit, exit_reason = self._check_exit_conditions(
                ensemble_output, market_data, timestamp
            )
            
            # 7. Final Signal Decision
            signal_output = self._prepare_signal_output(
                ensemble_output,
                regime_output,
                analyst_output,
                specialist_output,
            )

            self.signal_history.append(signal_output)
            self.circuit_breaker._on_success()
            return signal_output

        except Exception as e:
            self.circuit_breaker._on_failure()
            self.logger.error(f"Signal generation failed: {e}")
            raise

    def _parse_regime_output(self, data: Dict[str, Any], timestamp: datetime) -> RegimeOutput:
        """Convert raw regime detector output to RegimeOutput dataclass."""
        probs = data.get('regime_probabilities', {})
        # Ensure keys are RegimeType enums
        typed_probs = {}
        for k, v in probs.items():
            if isinstance(k, str):
                try:
                    # Handle "regime_0" format
                    if k.startswith("regime_"):
                        idx = int(k.split("_")[1])
                        regime_enum = list(RegimeType)[idx]
                    else:
                        regime_enum = RegimeType(k)
                    typed_probs[regime_enum] = float(v)
                except Exception:
                    continue

        # Default if empty
        if not typed_probs:
            typed_probs = {rt: 1.0/len(RegimeType) for rt in RegimeType}

        primary = data.get('primary_regime', 0)
        if isinstance(primary, int):
            primary_enum = list(RegimeType)[primary]
        else:
            primary_enum = primary

        return RegimeOutput(
            timestamp=timestamp,
            regime_probabilities=typed_probs,
            primary_regime=primary_enum,
            confidence=data.get('confidence', 0.5),
            regime_strength=data.get('regime_strength', 0.5),
            transition_probability=data.get('transition_probability', 0.0),
            features_used=data.get('features_used', {})
        )

    def _run_specialist_models(self, market_data: pd.DataFrame, features: pd.DataFrame) -> SpecialistOutput:
        """Run all specialist models and return canonical scalars."""
        timestamp = features.index[-1] if not features.empty else datetime.now()
        
        scores = {
            'risk_score': 0.5,
            'smc_score': 0.5,
            'liquidity_score': 0.5,
            'breakout_score': 0.5,
            'meso_score': 0.5,
            'macro_score': 0.5,
            'reversion_score': 0.5
        }
        
        raw_outputs = {}
        
        # Helper to safely predict
        def safe_predict(model_key, input_data, output_key, default=0.5):
            model = self.specialist_models.get(model_key)
            if model and hasattr(model, 'predict'):
                try:
                    # Handle input shape (1 sample)
                    if isinstance(input_data, pd.DataFrame):
                        # Ensure input data has same columns as training if model has feature_names_in_
                        if hasattr(model, 'feature_names_in_'):
                            # Filter/reorder columns to match model expectation
                            available_cols = [c for c in model.feature_names_in_ if c in input_data.columns]
                            if len(available_cols) < len(model.feature_names_in_):
                                # self.logger.warning(f"Feature mismatch for {model_key}: missing columns")
                                pass
                            X = input_data[available_cols].iloc[[-1]].values
                        else:
                            X = input_data.iloc[[-1]].values
                    else:
                        X = input_data
                    
                    pred = model.predict(X)
                    val = float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
                    return np.clip(val, 0.0, 1.0)
                except Exception as e:
                    self.logger.warning(f"Specialist {model_key} failed: {e}")
            return default

        # 1. Risk
        # Try to use pipeline features if available
        risk_pipe = self.specialist_models.get('risk_pipeline')
        if risk_pipe:
             try:
                 risk_feats = risk_pipe.transform(market_data.iloc[[-1]])
                 scores['risk_score'] = safe_predict('risk_model', risk_feats, 'risk')
             except Exception:
                 scores['risk_score'] = safe_predict('risk_model', features, 'risk')
        else:
            scores['risk_score'] = safe_predict('risk_model', features, 'risk')
        
        # 2. SMC
        scores['smc_score'] = safe_predict('smc_model', features, 'smc')

        # 3. Liquidity (Tree/Thresholds often don't have predict, skip if not model)
        scores['liquidity_score'] = safe_predict('liquidity_tree', features, 'liquidity')

        # 4. Breakout
        scores['breakout_score'] = safe_predict('breakout_model', features, 'breakout')

        # 5. Meso Trend
        meso_pipe = self.specialist_models.get('meso_pipeline')
        if meso_pipe:
            try:
                meso_feats = meso_pipe.transform(market_data.iloc[[-1]])
                scores['meso_score'] = safe_predict('meso_model', meso_feats, 'meso')
            except Exception:
                scores['meso_score'] = safe_predict('meso_model', features, 'meso')
        else:
            scores['meso_score'] = safe_predict('meso_model', features, 'meso')

        # 6. Macro Trend
        macro_pipe = self.specialist_models.get('macro_pipeline')
        if macro_pipe:
            try:
                macro_feats = macro_pipe.transform(market_data.iloc[[-1]])
                scores['macro_score'] = safe_predict('macro_model', macro_feats, 'macro')
            except Exception:
                scores['macro_score'] = safe_predict('macro_model', features, 'macro')
        else:
            scores['macro_score'] = safe_predict('macro_model', features, 'macro')

        # 7. Mean Reversion
        scores['reversion_score'] = safe_predict('reversion_model', features, 'reversion')

        return SpecialistOutput(
            timestamp=timestamp,
            **scores,
            raw_outputs=raw_outputs
        )

    def _prepare_analyst_input(
        self,
        base_features: pd.DataFrame,
        specialist_output: SpecialistOutput,
        regime_output: RegimeOutput
    ) -> np.ndarray:
        """Combine all features into a single input vector for Analyst models."""
        # 1. Base Features (last row)
        base_vec = base_features.iloc[-1].values

        # 2. Specialist Scalars (order matters if model trained on them)
        spec_vec = np.array([
            specialist_output.risk_score,
            specialist_output.smc_score,
            specialist_output.liquidity_score,
            specialist_output.breakout_score,
            specialist_output.meso_score,
            specialist_output.macro_score,
            specialist_output.reversion_score
        ])

        # 3. Regime Probabilities (order matters)
        regime_probs = [regime_output.regime_probabilities.get(rt, 0.0) for rt in RegimeType]
        regime_vec = np.array(regime_probs)

        # Concatenate
        combined = np.concatenate([base_vec, spec_vec, regime_vec])
        return combined.reshape(1, -1)

    def _run_analyst_models(self, input_vector: np.ndarray, timestamp: datetime) -> AnalystOutput:
        """Run LGBM and NGBoost models."""
        # LGBM -> Signal/Confidence
        signal_score = 0.0
        signal_confidence = 0.5

        if self.analyst_lgbm:
            try:
                # Predict
                raw_pred = self.analyst_lgbm.predict(input_vector)[0]
                signal_score = float(raw_pred)

                # If classifier with proba
                if hasattr(self.analyst_lgbm, 'predict_proba'):
                    probs = self.analyst_lgbm.predict_proba(input_vector)[0]
                    # Assuming binary classification [0, 1] or multi-class
                    # Use max prob as confidence
                    signal_confidence = float(np.max(probs))
                else:
                    # If regressor, treat score as confidence proxy (normalize if needed)
                    # For now assume score is [0,1] or similar
                    signal_confidence = np.clip(abs(signal_score), 0.0, 1.0)
            except Exception as e:
                self.logger.error(f"LGBM inference failed: {e}")

        # NGBoost -> Uncertainty
        uncertainty = 0.5
        if self.analyst_ngboost:
            try:
                # NGBoost predict_dist returns a distribution object
                if hasattr(self.analyst_ngboost, 'predict_dist'):
                    dist = self.analyst_ngboost.predict_dist(input_vector)
                    # Use variance or std as uncertainty
                    # NGBoost distributions usually have params method
                    # For Normal, params[1] is scale (log(sigma) or sigma depending on implementation)
                    # Let's try standard deviation directly if available on dist object
                    if hasattr(dist, 'params'):
                        # Assuming Normal distribution: loc, scale
                        scale = dist.params['scale'][0]
                        uncertainty = float(scale)
                    else:
                        # Fallback
                        uncertainty = 0.5
                else:
                    # Standard predict doesn't give uncertainty
                    uncertainty = 0.5

                # Normalize uncertainty to [0, 1] if possible, or clip
                # Sigmoid transform for unbounded variance?
                uncertainty = np.clip(uncertainty, 0.0, 1.0)
            except Exception as e:
                self.logger.error(f"NGBoost inference failed: {e}")

        return AnalystOutput(
            timestamp=timestamp,
            signal_score=signal_score,
            signal_confidence=signal_confidence,
            uncertainty=uncertainty,
            features={}
        )

    def _check_exit_conditions(
        self,
        analyst_output: AnalystOutput,
        market_data: pd.DataFrame,
        timestamp: datetime
    ) -> Tuple[bool, Optional[str]]:
        """Check exit conditions using Signal Confidence and Uncertainty."""
        with self._position_lock:
            if not self.current_position or not self.current_position.is_open:
                return False, None
            
            pos = self.current_position
            
            # 1. Uncertainty-based Exit
            # If uncertainty spikes, exit
            MAX_UNCERTAINTY = 0.8  # Threshold
            if analyst_output.uncertainty > MAX_UNCERTAINTY:
                return True, f"High uncertainty ({analyst_output.uncertainty:.2f})"
            
            # 2. Confidence Drop (adjusted by uncertainty)
            # Effective confidence = Signal Confidence * (1 - Uncertainty)
            eff_conf = analyst_output.signal_confidence * (1.0 - analyst_output.uncertainty)
            
            exit_thresh = self.optimization_params.get('exit_confidence_threshold', DEFAULT_EXIT_CONFIDENCE_THRESHOLD)
            if eff_conf < exit_thresh:
                return True, f"Low effective confidence ({eff_conf:.2f} < {exit_thresh})"
            
            # 3. Stop Loss / Take Profit (Standard logic)
            # ... (Reuse existing logic or simplified version)
            current_price = market_data['close'].iloc[-1]
            if pos.entry_price:
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                if pos.direction == 'short':
                    pnl_pct = -pnl_pct
                
                if pos.stop_loss_pct and pnl_pct < -abs(pos.stop_loss_pct):
                    return True, f"Stop loss triggered ({pnl_pct:.2%})"
                
                # Trail profit?
                if pos.peak_profit_pct and pos.peak_profit_pct > 0.02 and pnl_pct < (pos.peak_profit_pct * 0.8):
                     return True, "Trailing stop triggered"

            # Update peak profit
            if pnl_pct > (pos.peak_profit_pct or -999):
                pos.peak_profit_pct = pnl_pct

            return False, None

    def _decide_signal(
        self,
        analyst_output: AnalystOutput,
        regime_output: RegimeOutput,
        should_exit: bool,
        exit_reason: str
    ) -> Dict[str, Any]:
        """Combine inputs to make final decision."""
        if should_exit:
            return {'signal': 'close', 'confidence': 1.0, 'strength': 1.0}

        # Effective confidence considers uncertainty
        eff_conf = analyst_output.signal_confidence * (1.0 - analyst_output.uncertainty)

        signal_thresh = self.optimization_params.get('signal_confidence_threshold', 0.6)

        if eff_conf >= signal_thresh:
            # Respect configured direction
            configured_direction = getattr(self.config, 'direction', 'long')

            # Note: analyst_output.signal_score is typically the probability of the target class.
            # For a Long model, target=1 means "Long Profitable".
            # For a Short model, target=1 means "Short Profitable" (assuming separate models).

            # If score > 0.5 (or threshold), it's a signal to ENTER the configured direction.
            # If score <= 0.5, it's a signal to HOLD (not enter).

            # We strictly prevent "Long" model from signaling "Short" and vice versa.

            signal_direction = 'hold'

            if analyst_output.signal_score > 0.5:
                if configured_direction == 'long':
                    signal_direction = 'buy'
                elif configured_direction == 'short':
                    signal_direction = 'sell'

            # If signal_score is low, we hold. We do NOT reverse direction.

            return {
                'signal': signal_direction,
                'confidence': eff_conf,
                'strength': abs(analyst_output.signal_score)
            }

        return {'signal': 'hold', 'confidence': eff_conf, 'strength': 0.0}

    def _update_position_state(self, signal_data, analyst_output, market_data, timestamp):
        """Update position state."""
        signal = signal_data['signal']
        with self._position_lock:
            price = market_data['close'].iloc[-1]

            if signal == 'close' and self.current_position and self.current_position.is_open:
                self.current_position.is_open = False
                self.position_history.append(self.current_position)
                self.current_position = None
                self.logger.info(f"Position closed at {price}")

            elif signal in ['buy', 'sell'] and not self.current_position:
                self.current_position = PositionState(
                    is_open=True,
                    entry_timestamp=timestamp,
                    entry_price=price,
                    direction='long' if signal == 'buy' else 'short',
                    entry_confidence=signal_data['confidence'],
                    entry_uncertainty=analyst_output.uncertainty,
                    stop_loss_pct=self.optimization_params.get('stop_loss_pct', 0.03)
                )
                self.logger.info(f"Position opened: {signal} at {price}")

    async def stop(self):
        """Stop pipeline."""
        self.is_initialized = False
        tprint("[SIGNAL_PIPELINE] Stopped")

# Convenience function
async def setup_signal_generation_pipeline(config: TradingConfig) -> Optional[SignalGenerationPipeline]:
    pipeline = SignalGenerationPipeline(config)
    if await pipeline.initialize():
        return pipeline
    return None
