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

        # Validation & Rate Limiting
        validate_market_data(market_data)
        if not self.rate_limiter.acquire():
            raise RuntimeError("Rate limit exceeded")

        timestamp = datetime.now()

        try:
            # 1. Feature Engineering
            # Use last row context but need enough history for lookbacks
            engineered_features = self.feature_engineer.engineer_features(market_data)
            
            # 2. Regime Detection
            regime_output_data = await self.regime_detector.predict_regime(market_data)
            regime_output = self._parse_regime_output(regime_output_data, timestamp)
            
            # 3. Specialist Inference
            specialist_output = self._run_specialist_models(market_data, engineered_features)
            
            # 4. Construct Analyst Input Vector
            # Combine engineered features + specialist scalars + regime probs
            analyst_input = self._prepare_analyst_input(
                engineered_features,
                specialist_output,
                regime_output
            )
            
            # 5. Analyst Inference (LGBM + NGBoost)
            analyst_output = self._run_analyst_models(analyst_input, timestamp)
            
            # 6. Exit Logic (if position open)
            should_exit, exit_reason = self._check_exit_conditions(
                analyst_output, market_data, timestamp
            )
            
            # 7. Final Signal Decision
            final_signal_data = self._decide_signal(
                analyst_output,
                regime_output,
                should_exit,
                exit_reason
            )
            
            # 8. Update State
            self._update_position_state(final_signal_data, analyst_output, market_data, timestamp)
            
            # 9. Result
            result = SignalGenerationResult(
                timestamp=timestamp,
                symbol=symbol,
                regime_output=regime_output,
                specialist_output=specialist_output,
                analyst_output=analyst_output,
                final_signal=final_signal_data['signal'],
                final_confidence=final_signal_data['confidence'],
                signal_strength=final_signal_data['strength'],
                uncertainty=analyst_output.uncertainty,
                optimization_parameters=self.optimization_params,
                metadata={'data_points': len(market_data)},
                exit_confidence=analyst_output.signal_confidence, # Using signal confidence as proxy for now
                should_exit=should_exit,
                exit_reason=exit_reason,
                position_state=self.current_position
            )

            self.signal_history.append(result)
            self.circuit_breaker._on_success()
            return result

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
            # Determine direction from signal_score
            # Assuming score > 0 is Long, < 0 is Short (or use classes if probabilistic)
            # If LGBM is a regressor predicting return:
            direction = 'buy' if analyst_output.signal_score > 0 else 'sell'
            # If LGBM is binary classifier (1=Long, 0=Short? Or 1=Buy, 0=Hold?)
            # Adjust based on your model training target.
            # Assuming Regressor for now or Binary Target where 1=Buy.
            # If score is probability of BUY:
            if analyst_output.signal_score > 0.5:
                direction = 'buy'
            else:
                direction = 'sell' # or hold?

            return {
                'signal': direction,
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
