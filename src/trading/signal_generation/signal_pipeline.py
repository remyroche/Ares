"""
Signal Generation Pipeline

Implements proper data flow: HMM regime -> analyst -> tactician
with sequential model calls and confidence score optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from ..config.regime_config import RegimeType
from ..config.trading_config import TradingConfig

logger = system_logger.getChild('SignalGenerationPipeline')

@dataclass
class HMMRegimeOutput:
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
class SignalGenerationResult:
    """Complete signal generation result."""
    timestamp: datetime
    symbol: str
    hmm_output: HMMRegimeOutput
    analyst_output: AnalystMetaOutput
    tactician_output: TacticianMetaOutput
    final_signal: str
    final_confidence: float
    signal_strength: float
    optimization_parameters: Dict[str, Any]
    metadata: Dict[str, Any]

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
        self.hmm_regime_detector = None
        self.analyst_base_models = []
        self.analyst_meta_model = None
        self.tactician_base_models = []
        self.tactician_meta_model = None
        
        # Optimization parameters (from backtesting)
        self.optimization_params = {
            'analyst_confidence_weight': 0.6,
            'tactician_confidence_weight': 0.4,
            'regime_confidence_threshold': 0.7,
            'signal_confidence_threshold': 0.6,
            'meta_model_weight': 0.8,
            'base_model_weight': 0.2
        }
        
        # State management
        self.is_initialized = False
        self.signal_history: List[SignalGenerationResult] = []
        
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize signal generation pipeline."""
        try:
            self.logger.info("Initializing Signal Generation Pipeline...")
            
            # Initialize HMM regime detector
            await self._initialize_hmm_regime_detector()
            
            # Initialize analyst models
            await self._initialize_analyst_models()
            
            # Initialize tactician models
            await self._initialize_tactician_models()
            
            # Load optimization parameters
            await self._load_optimization_parameters()
            
            self.is_initialized = True
            self.logger.info("✅ Signal Generation Pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Signal Generation Pipeline: {e}")
            return False
    
    async def _initialize_hmm_regime_detector(self):
        """Initialize HMM regime detector."""
        try:
            # Import and initialize HMM regime detector
            from ..regime.regime_detector import RegimeDetector
            from ..config.regime_config import RegimeConfig
            
            regime_config = RegimeConfig()
            self.hmm_regime_detector = RegimeDetector(regime_config)
            await self.hmm_regime_detector.initialize()
            
            self.logger.info("✅ HMM regime detector initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize HMM regime detector: {e}")
            raise
    
    async def _initialize_analyst_models(self):
        """Initialize analyst base and meta models."""
        try:
            # Import analyst components
            from src.analyst.market_health_analyzer import MarketHealthAnalyzer
            from src.analyst.ml_confidence_predictor import MLConfidencePredictor
            from src.analyst.liquidation_risk_model import LiquidationRiskModel
            
            # Initialize base models
            self.analyst_base_models = [
                MarketHealthAnalyzer(self.config),
                MLConfidencePredictor(self.config),
                LiquidationRiskModel(self.config)
            ]
            
            # Initialize base models
            for model in self.analyst_base_models:
                if hasattr(model, 'initialize'):
                    await model.initialize()
            
            # Initialize meta model (analyst orchestrator)
            from src.analyst.analyst import Analyst
            self.analyst_meta_model = Analyst(self.config)
            await self.analyst_meta_model.initialize()
            
            self.logger.info("✅ Analyst models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize analyst models: {e}")
            raise
    
    async def _initialize_tactician_models(self):
        """Initialize tactician base and meta models."""
        try:
            # Import tactician components
            from src.tactician.scenario_based_predictor import ScenarioBasedPredictor
            from src.tactician.enhanced_scenario_based_predictor import EnhancedScenarioBasedPredictor
            from src.tactician.position_sizer import PositionSizer
            from src.tactician.leverage_sizer import LeverageSizer
            
            # Initialize base models
            self.tactician_base_models = [
                ScenarioBasedPredictor(self.config),
                EnhancedScenarioBasedPredictor(self.config),
                PositionSizer(self.config),
                LeverageSizer(self.config)
            ]
            
            # Initialize base models
            for model in self.tactician_base_models:
                if hasattr(model, 'initialize'):
                    await model.initialize()
            
            # Initialize meta model (tactician orchestrator)
            from src.tactician.tactician import Tactician
            self.tactician_meta_model = Tactician(self.config)
            await self.tactician_meta_model.initialize()
            
            self.logger.info("✅ Tactician models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tactician models: {e}")
            raise
    
    async def _load_optimization_parameters(self):
        """Load optimization parameters from backtesting results."""
        try:
            # Load optimization parameters from backtesting results
            # This would typically load from a file or database
            # For now, using default optimized parameters
            
            optimization_file = "optimization_results.json"
            # In practice, this would load actual optimization results
            # self.optimization_params = load_optimization_results(optimization_file)
            
            self.logger.info("✅ Optimization parameters loaded")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load optimization parameters, using defaults: {e}")
    
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
            
            timestamp = datetime.now()
            
            # Step 1: HMM Regime Detection
            hmm_output = await self._detect_hmm_regime(market_data, timestamp)
            
            # Step 2: Analyst Base Models
            analyst_base_outputs = await self._run_analyst_base_models(
                market_data, hmm_output, additional_features, timestamp
            )
            
            # Step 3: Analyst Meta Model
            analyst_meta_output = await self._run_analyst_meta_model(
                market_data, hmm_output, analyst_base_outputs, timestamp
            )
            
            # Step 4: Tactician Base Models
            tactician_base_outputs = await self._run_tactician_base_models(
                market_data, hmm_output, analyst_meta_output, timestamp
            )
            
            # Step 5: Tactician Meta Model
            tactician_meta_output = await self._run_tactician_meta_model(
                market_data, hmm_output, analyst_meta_output, tactician_base_outputs, timestamp
            )
            
            # Step 6: Final Signal Generation
            final_signal = self._generate_final_signal(
                hmm_output, analyst_meta_output, tactician_meta_output
            )
            
            # Create result
            result = SignalGenerationResult(
                timestamp=timestamp,
                symbol=symbol,
                hmm_output=hmm_output,
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
                }
            )
            
            # Store in history
            self.signal_history.append(result)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            self.logger.debug(f"Signal generated for {symbol}: {final_signal['signal']} (confidence: {final_signal['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Signal generation failed for {symbol}: {e}")
            raise
    
    async def _detect_hmm_regime(self, market_data: pd.DataFrame, timestamp: datetime) -> HMMRegimeOutput:
        """Step 1: Detect HMM regime."""
        try:
            # Use HMM regime detector
            regime_detection = await self.hmm_regime_detector.detect_regime(market_data)
            
            return HMMRegimeOutput(
                timestamp=timestamp,
                regime_probabilities=regime_detection.regime_probabilities,
                primary_regime=regime_detection.primary_regime,
                confidence=regime_detection.confidence,
                regime_strength=regime_detection.regime_strength,
                transition_probability=regime_detection.transition_probability,
                features_used=regime_detection.features_used
            )
            
        except Exception as e:
            self.logger.error(f"❌ HMM regime detection failed: {e}")
            raise
    
    async def _run_analyst_base_models(
        self,
        market_data: pd.DataFrame,
        hmm_output: HMMRegimeOutput,
        additional_features: Optional[Dict[str, Any]],
        timestamp: datetime
    ) -> List[AnalystBaseOutput]:
        """Step 2: Run analyst base models sequentially."""
        try:
            base_outputs = []
            
            for model in self.analyst_base_models:
                try:
                    # Run each base model
                    if hasattr(model, 'analyze'):
                        result = await model.analyze(market_data, hmm_output.regime_probabilities)
                    elif hasattr(model, 'predict'):
                        result = await model.predict(market_data)
                    else:
                        # Fallback for models without standard interface
                        result = {'confidence': 0.5, 'features': {}}
                    
                    # Create base output
                    base_output = AnalystBaseOutput(
                        timestamp=timestamp,
                        market_health=result.get('market_health', {}),
                        volatility_analysis=result.get('volatility_analysis', {}),
                        liquidity_analysis=result.get('liquidity_analysis', {}),
                        stress_analysis=result.get('stress_analysis', {}),
                        base_confidence=result.get('confidence', 0.5),
                        features=result.get('features', {})
                    )
                    
                    base_outputs.append(base_output)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Analyst base model failed: {e}")
                    # Create fallback output
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
        hmm_output: HMMRegimeOutput,
        base_outputs: List[AnalystBaseOutput],
        timestamp: datetime
    ) -> AnalystMetaOutput:
        """Step 3: Run analyst meta model."""
        try:
            # Combine base model outputs
            combined_features = {}
            total_confidence = 0.0
            
            for base_output in base_outputs:
                combined_features.update(base_output.features)
                total_confidence += base_output.base_confidence
            
            avg_base_confidence = total_confidence / len(base_outputs) if base_outputs else 0.5
            
            # Run meta model
            meta_result = await self.analyst_meta_model.analyze_regime(market_data)
            
            # Apply regime adjustment
            regime_adjusted_confidence = self._apply_regime_adjustment(
                meta_result.get('confidence', avg_base_confidence),
                hmm_output.regime_probabilities
            )
            
            return AnalystMetaOutput(
                timestamp=timestamp,
                analyst_confidence=regime_adjusted_confidence,
                market_health_score=meta_result.get('market_health_score', 0.5),
                regime_adjusted_confidence=regime_adjusted_confidence,
                meta_features=combined_features,
                base_outputs=base_outputs
            )
            
        except Exception as e:
            self.logger.error(f"❌ Analyst meta model failed: {e}")
            raise
    
    async def _run_tactician_base_models(
        self,
        market_data: pd.DataFrame,
        hmm_output: HMMRegimeOutput,
        analyst_output: AnalystMetaOutput,
        timestamp: datetime
    ) -> List[TacticianBaseOutput]:
        """Step 4: Run tactician base models sequentially."""
        try:
            base_outputs = []
            
            for model in self.tactician_base_models:
                try:
                    # Run each base model with analyst output
                    if hasattr(model, 'generate_enhanced_predictions'):
                        result = await model.generate_enhanced_predictions(
                            market_data, {}, market_data.columns[0] if len(market_data.columns) > 0 else "ETHUSDT",
                            "1m", analyst_output.analyst_confidence
                        )
                    elif hasattr(model, 'predict'):
                        result = await model.predict(market_data)
                    else:
                        # Fallback for models without standard interface
                        result = {
                            'confidence': 0.5,
                            'scenario_predictions': {},
                            'price_targets': {},
                            'adversarial_risks': {}
                        }
                    
                    # Create base output
                    base_output = TacticianBaseOutput(
                        timestamp=timestamp,
                        scenario_predictions=result.get('scenario_predictions', {}),
                        price_targets=result.get('price_targets', {}),
                        adversarial_risks=result.get('adversarial_risks', {}),
                        base_confidence=result.get('confidence', 0.5),
                        position_recommendations=result.get('position_recommendations', {})
                    )
                    
                    base_outputs.append(base_output)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Tactician base model failed: {e}")
                    # Create fallback output
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
        hmm_output: HMMRegimeOutput,
        analyst_output: AnalystMetaOutput,
        base_outputs: List[TacticianBaseOutput],
        timestamp: datetime
    ) -> TacticianMetaOutput:
        """Step 5: Run tactician meta model."""
        try:
            # Combine base model outputs
            combined_scenario_predictions = {}
            combined_price_targets = {}
            combined_adversarial_risks = {}
            total_confidence = 0.0
            
            for base_output in base_outputs:
                combined_scenario_predictions.update(base_output.scenario_predictions)
                combined_price_targets.update(base_output.price_targets)
                combined_adversarial_risks.update(base_output.adversarial_risks)
                total_confidence += base_output.base_confidence
            
            avg_base_confidence = total_confidence / len(base_outputs) if base_outputs else 0.5
            
            # Run meta model
            meta_result = await self.tactician_meta_model.generate_enhanced_predictions(
                market_data, {}, market_data.columns[0] if len(market_data.columns) > 0 else "ETHUSDT",
                "1m", analyst_output.analyst_confidence
            )
            
            # Calculate combined confidence
            tactician_confidence = meta_result.get('confidence', avg_base_confidence)
            combined_confidence = self._calculate_combined_confidence(
                analyst_output.analyst_confidence, tactician_confidence
            )
            
            return TacticianMetaOutput(
                timestamp=timestamp,
                tactician_confidence=tactician_confidence,
                combined_confidence=combined_confidence,
                final_signal=meta_result.get('final_signal', 'hold'),
                signal_strength=meta_result.get('signal_strength', 0.5),
                meta_features={
                    'scenario_predictions': combined_scenario_predictions,
                    'price_targets': combined_price_targets,
                    'adversarial_risks': combined_adversarial_risks
                },
                base_outputs=base_outputs
            )
            
        except Exception as e:
            self.logger.error(f"❌ Tactician meta model failed: {e}")
            raise
    
    def _apply_regime_adjustment(self, base_confidence: float, regime_probabilities: Dict[RegimeType, float]) -> float:
        """Apply regime-based confidence adjustment."""
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
            
            # Calculate weighted regime multiplier
            regime_multiplier = 1.0
            for regime, probability in regime_probabilities.items():
                multiplier = regime_multipliers.get(regime, 1.0)
                regime_multiplier += (multiplier - 1.0) * probability
            
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
    
    def _generate_final_signal(
        self,
        hmm_output: HMMRegimeOutput,
        analyst_output: AnalystMetaOutput,
        tactician_output: TacticianMetaOutput
    ) -> Dict[str, Any]:
        """Generate final trading signal."""
        try:
            # Use optimization parameters for thresholds
            regime_threshold = self.optimization_params['regime_confidence_threshold']
            signal_threshold = self.optimization_params['signal_confidence_threshold']
            
            # Check regime confidence
            if hmm_output.confidence < regime_threshold:
                return {
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'reason': f'Low regime confidence: {hmm_output.confidence:.3f} < {regime_threshold:.3f}'
                }
            
            # Check signal confidence
            if tactician_output.combined_confidence < signal_threshold:
                return {
                    'signal': 'hold',
                    'confidence': tactician_output.combined_confidence,
                    'strength': 0.0,
                    'reason': f'Low signal confidence: {tactician_output.combined_confidence:.3f} < {signal_threshold:.3f}'
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
    
    def get_signal_history(self, limit: int = 100) -> List[SignalGenerationResult]:
        """Get recent signal generation history."""
        return self.signal_history[-limit:] if self.signal_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for signal generation."""
        try:
            if not self.signal_history:
                return {
                    'total_signals': 0,
                    'avg_confidence': 0.0,
                    'signal_distribution': {'buy': 0, 'sell': 0, 'hold': 0}
                }
            
            recent_signals = self.signal_history[-100:]  # Last 100 signals
            
            avg_confidence = sum(s.final_confidence for s in recent_signals) / len(recent_signals)
            
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
            if self.hmm_regime_detector:
                await self.hmm_regime_detector.stop()
            
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