"""
Model Selection and Routing System

Intelligent model selection and routing system that automatically selects
the best model for each market regime based on performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Tuple as TypingTuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import regime detection systems
try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector, TASRegimeConfig
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector, PerfectNASConfig
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeConfig
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Model selection strategies."""
    BEST_PERFORMANCE = "best_performance"  # Select best performing model
    ENSEMBLE = "ensemble"                 # Use ensemble of models
    ADAPTIVE = "adaptive"                 # Adaptively select based on conditions
    META_LEARNING = "meta_learning"       # Use meta-learning for selection
    CONFIDENCE_BASED = "confidence_based" # Select based on prediction confidence


class RoutingMethod(Enum):
    """Model routing methods."""
    REGIME_BASED = "regime_based"         # Route based on detected regime
    PERFORMANCE_BASED = "performance_based" # Route based on recent performance
    HYBRID = "hybrid"                     # Combine regime and performance
    DYNAMIC = "dynamic"                   # Dynamic routing based on conditions


@dataclass
class ModelSelectionConfig:
    """Configuration for model selection and routing."""
    
    # Selection strategy
    selection_strategy: SelectionStrategy = SelectionStrategy.BEST_PERFORMANCE
    routing_method: RoutingMethod = RoutingMethod.REGIME_BASED
    
    # Performance tracking
    performance_window: int = 100  # Number of recent predictions to consider
    performance_metric: str = "f1_score"  # Metric to optimize
    min_performance_threshold: float = 0.5  # Minimum performance to consider model
    
    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_method: str = "voting"  # "voting", "stacking", "blending"
    ensemble_weights: Optional[List[float]] = None  # None for automatic weighting
    
    # Adaptive selection
    enable_adaptive_selection: bool = True
    adaptation_rate: float = 0.1  # Learning rate for adaptation
    confidence_threshold: float = 0.7  # Confidence threshold for selection
    
    # Meta-learning
    enable_meta_learning: bool = False
    meta_features: List[str] = field(default_factory=lambda: [
        'market_volatility', 'trend_strength', 'volume_ratio', 'time_of_day'
    ])
    
    # Fallback mechanisms
    enable_fallback: bool = True
    fallback_model: str = "xgboost"  # Fallback model type
    fallback_threshold: float = 0.3  # Performance threshold for fallback
    
    # Regime detection
    regime_detection_method: str = "hybrid"  # "tas", "nas", "hybrid"
    regime_confidence_threshold: float = 0.7
    enable_regime_validation: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_frequency: int = 50  # Update every N predictions
    save_selection_history: bool = True
    selection_history_path: str = "model_selection_history.json"

    # Top-K selection
    top_k: int = 3  # Number of top models to select per market/regime


@dataclass
class ModelSelectionResult:
    """Result from model selection."""
    
    # Selection results
    selected_model: Any
    selected_model_type: str
    selected_regime: int
    selection_confidence: float
    selection_reason: str
    
    # Performance metrics
    expected_performance: Dict[str, float]
    historical_performance: Dict[str, float]
    confidence_interval: Tuple[float, float]

    # Alternative options
    alternative_models: List[Dict[str, Any]]

    # Metadata
    selection_time: datetime
    ensemble_weights: Optional[Dict[str, float]] = None
    regime_probabilities: Optional[np.ndarray] = None
    model_rankings: Dict[str, int] = field(default_factory=dict)


class ModelSelector:
    """
    Intelligent model selection and routing system.
    
    Automatically selects the best model for each market regime based on
    performance metrics, confidence scores, and adaptive learning.
    """
    
    def __init__(self, config: ModelSelectionConfig):
        """Initialize model selector.
        
        Args:
            config: Model selection configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regime detection
        self._initialize_regime_detection()
        
        # Initialize ML common utilities
        self._initialize_ml_common()
        
        # Selection state
        self.available_models = {}  # regime_id -> {model_type: model_info}
        self.performance_history = {}  # model_id -> [performance_scores]
        self.selection_history = []
        self.model_rankings = {}
        self.ensemble_weights = {}
        
        # Adaptive learning state
        self.adaptation_weights = {}
        self.meta_features = {}
        
        self.logger.info("✅ Model Selector initialized")
        self.logger.info(f"   Selection strategy: {config.selection_strategy.value}")
        self.logger.info(f"   Routing method: {config.routing_method.value}")
        self.logger.info(f"   Ensemble enabled: {config.enable_ensemble}")
    
    def _initialize_regime_detection(self):
        """Initialize regime detection systems."""
        if not REGIME_DETECTION_AVAILABLE:
            self.logger.warning("⚠️ Regime detection systems not available")
            self.tas_detector = None
            self.nas_detector = None
            return
        
        try:
            if self.config.regime_detection_method in ["tas", "hybrid", "hybrid_nas_tas"]:
                tas_config = TASRegimeConfig(
                    n_regimes=8,
                    enable_economic_evaluation=True,
                    enable_uncertainty_quantification=True
                )
                self.tas_detector = TASRegimeDetector(tas_config)
                self.logger.info("✅ TAS regime detector initialized")

            if self.config.regime_detection_method in ["nas", "hybrid", "hybrid_nas_tas"]:
                nas_config = PerfectNASConfig.create_short_term_trading_config()
                self.nas_detector = PerfectNASRegimeDetector(nas_config)
                self.logger.info("✅ NAS regime detector initialized")

            if self.config.regime_detection_method == "hybrid_nas_tas":
                hybrid_config = HybridRegimeConfig(
                    combination_strategy="weighted",
                    tas_weight=0.4,
                    nas_weight=0.6,
                    enable_economic_evaluation=True,
                    enable_financial_relevance=True
                )
                self.hybrid_detector = HybridNASTASRegimeDetector(hybrid_config)
                self.logger.info("✅ Hybrid NAS-TAS regime detector initialized")

        except Exception as e:
            self.logger.warning(f"Regime detection initialization failed: {e}")
            self.tas_detector = None
            self.nas_detector = None
            self.hybrid_detector = None
    
    def _initialize_ml_common(self):
        """Initialize ML common utilities."""
        if not ML_COMMON_AVAILABLE:
            self.logger.warning("⚠️ ML common utilities not available")
            self.ml_common_ops = None
            self.validation_framework = None
            return
        
        try:
            self.ml_common_ops = get_ml_common_operations()
            self.validation_framework = get_validation_framework()
            self.logger.info("✅ ML common utilities initialized")
        except Exception as e:
            self.logger.warning(f"ML common initialization failed: {e}")
            self.ml_common_ops = None
            self.validation_framework = None
    
    def register_models(self,
                      regime_models: Dict[int, Dict[str, Any]],
                      ensemble_models: Optional[Dict[str, Any]] = None,
                      directional_models: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None):
        """
        Register trained models for selection.

        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
            directional_models: Optional directional models (direction -> regime -> models)
        """
        self.logger.info("📝 Registering models for selection")
        
        # Register regime models (handling both old and new structure)
        for regime_id, models in regime_models.items():
            self.available_models[regime_id] = {}

            if isinstance(models, dict) and any(isinstance(v, dict) for v in models.values()):
                # New structure: {regime: {'long': model_info, 'short': model_info}}
                for direction, model_info in models.items():
                    if isinstance(model_info, dict) and 'model' in model_info:
                        model_id = f"regime_{regime_id}_{direction}"
                        self.available_models[regime_id][direction] = {
                            'model': model_info['model'],
                            'model_id': model_id,
                            'performance': model_info.get('val_metrics', {}),
                            'feature_importance': model_info.get('feature_importance', {}),
                            'hyperparameters': model_info.get('hyperparameters', {}),
                            'direction': direction
                        }

                        # Initialize performance history
                        self.performance_history[model_id] = []

                        self.logger.info(f"   ✅ Registered {direction} model for regime {regime_id}")
            else:
                # Old structure: {regime: {'model_type': model_info}}
                for model_type, model_info in models.items():
                    model_id = f"regime_{regime_id}_{model_type}"
                    self.available_models[regime_id][model_type] = {
                        'model': model_info['model'],
                        'model_id': model_id,
                        'performance': model_info.get('val_metrics', {}),
                        'feature_importance': model_info.get('feature_importance', {}),
                        'hyperparameters': model_info.get('hyperparameters', {})
                    }

                    # Initialize performance history
                    self.performance_history[model_id] = []

                    self.logger.info(f"   ✅ Registered {model_type} for regime {regime_id}")
        
        # Register ensemble models
        if ensemble_models:
            for ensemble_name, ensemble_info in ensemble_models.items():
                ensemble_id = f"ensemble_{ensemble_name}"
                self.available_models['ensemble'] = self.available_models.get('ensemble', {})
                self.available_models['ensemble'][ensemble_name] = {
                    'model': ensemble_info['model'],
                    'model_id': ensemble_id,
                    'performance': ensemble_info.get('val_metrics', {}),
                    'base_models': ensemble_info.get('base_models', []),
                    'weights': ensemble_info.get('weights')
                }
                
                self.performance_history[ensemble_id] = []
                self.logger.info(f"   ✅ Registered ensemble {ensemble_name}")

        # Register directional models
        if directional_models:
            for direction, regime_models in directional_models.items():
                self.available_models[f"{direction}_models"] = {}

                for regime_id, models in regime_models.items():
                    direction_regime_key = f"{direction}_regime_{regime_id}"
                    self.available_models[f"{direction}_models"][direction_regime_key] = {}

                    for model_type, model_info in models.items():
                        model_id = f"{direction}_{regime_id}_{model_type}"
                        self.available_models[f"{direction}_models"][direction_regime_key][model_type] = {
                            'model': model_info['model'],
                            'model_id': model_id,
                            'performance': model_info.get('val_metrics', {}),
                            'feature_importance': model_info.get('feature_importance', {}),
                            'hyperparameters': model_info.get('hyperparameters', {}),
                            'direction': direction
                        }

                        # Initialize performance history
                        self.performance_history[model_id] = []

                        self.logger.info(f"   ✅ Registered {model_type} for {direction} regime {regime_id}")

        self.logger.info(f"📊 Total models registered: {sum(len(models) for models in self.available_models.values())}")
    
    def select_model(self,
                    market_data: pd.DataFrame,
                    current_regime: Optional[int] = None,
                    context: Optional[Dict[str, Any]] = None,
                    direction: Optional[str] = None) -> ModelSelectionResult:
        """
        Select the best model for given market conditions.

        Args:
            market_data: Current market data
            current_regime: Current regime (None for auto-detection)
            context: Additional context for selection
            direction: Trading direction ('long', 'short', or None for auto-detection)

        Returns:
            ModelSelectionResult with selected model and metadata
        """
        start_time = datetime.now()
        self.logger.info("🎯 Starting model selection")
        
        try:
            # Step 1: Detect regime if not provided
            if current_regime is None:
                regime_info = self._detect_regime(market_data)
                current_regime = regime_info['regime_id']
                regime_probabilities = regime_info.get('probabilities')
            else:
                regime_probabilities = None

            # Step 2: Detect direction if not provided
            if direction is None:
                direction = self._detect_direction(market_data, context)

            # Step 3: Get available models for regime and direction
            regime_models = self._get_models_for_regime_and_direction(current_regime, direction)

            if not regime_models:
                self.logger.warning(f"⚠️ No directional models found for regime {current_regime}, direction {direction}. Using standard regime models.")
                if current_regime not in self.available_models:
                    raise ValueError(f"No models available for regime {current_regime}")
                regime_models = self.available_models[current_regime]
            
            # Step 3: Select model based on strategy
            if self.config.selection_strategy == SelectionStrategy.BEST_PERFORMANCE:
                selection_result = self._select_best_performance_model(
                    regime_models, market_data, context
                )
            elif self.config.selection_strategy == SelectionStrategy.ENSEMBLE:
                selection_result = self._select_ensemble_model(
                    current_regime, regime_models, market_data, context
                )
            elif self.config.selection_strategy == SelectionStrategy.ADAPTIVE:
                selection_result = self._select_adaptive_model(
                    regime_models, market_data, context
                )
            elif self.config.selection_strategy == SelectionStrategy.META_LEARNING:
                selection_result = self._select_meta_learning_model(
                    regime_models, market_data, context
                )
            elif self.config.selection_strategy == SelectionStrategy.CONFIDENCE_BASED:
                selection_result = self._select_confidence_based_model(
                    regime_models, market_data, context
                )
            else:
                raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")
            
            # Step 4: Calculate performance metrics
            expected_performance = self._calculate_expected_performance(
                selection_result['model_id'], market_data
            )
            
            historical_performance = self._get_historical_performance(
                selection_result['model_id']
            )
            
            confidence_interval = self._calculate_confidence_interval(
                selection_result['model_id']
            )
            
            # Step 5: Get alternative models (top K-1 since we already have the selected model)
            alternative_models = self._get_alternative_models(
                regime_models, selection_result['model_id']
            )

            # Ensure we have top K total models (selected + alternatives)
            total_models_needed = getattr(self.config, 'top_k', 3)
            if len(alternative_models) < total_models_needed - 1:
                # If we don't have enough alternatives, pad with None or repeat selection
                tprint_warning(f"⚠️ Only {len(alternative_models) + 1}/{total_models_needed} models available for regime {current_regime}")
                result.metadata['insufficient_models'] = True
            
            # Step 6: Create result
            result = ModelSelectionResult(
                selected_model=selection_result['model'],
                selected_model_type=selection_result['model_type'],
                selected_regime=current_regime,
                selection_confidence=selection_result['confidence'],
                selection_reason=selection_result['reason'],
                expected_performance=expected_performance,
                historical_performance=historical_performance,
                confidence_interval=confidence_interval,
                alternative_models=alternative_models,
                ensemble_weights=selection_result.get('ensemble_weights'),
                selection_time=start_time,
                regime_probabilities=regime_probabilities,
                model_rankings=self._get_model_rankings(regime_models)
            )
            
            # Step 7: Update selection history
            if self.config.save_selection_history:
                self._update_selection_history(result)
            
            # Step 8: Update adaptive learning
            if self.config.enable_adaptive_selection:
                self._update_adaptive_learning(result, market_data)
            
            self.logger.info(f"✅ Model selected: {result.selected_model_type} for regime {result.selected_regime}")
            self.logger.info(f"   Confidence: {result.selection_confidence:.3f}")
            self.logger.info(f"   Expected F1: {expected_performance.get('f1_score', 0):.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model selection failed: {e}")
            
            # Return fallback model if available
            if self.config.enable_fallback:
                return self._get_fallback_model(market_data, current_regime)
            else:
                raise
    
    def _detect_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime."""
        try:
            if self.config.regime_detection_method == "tas" and self.tas_detector:
                result = self.tas_detector.detect_regimes(market_data)
                return {
                    'regime_id': result.regime_predictions[-1],
                    'probabilities': result.regime_probabilities[-1],
                    'confidence': np.max(result.regime_probabilities[-1])
                }
            
            elif self.config.regime_detection_method == "nas" and self.nas_detector:
                result = self.nas_detector.detect_regimes(market_data)
                return {
                    'regime_id': result.regime_predictions[-1],
                    'probabilities': result.regime_probabilities[-1],
                    'confidence': np.max(result.regime_probabilities[-1])
                }
            
            elif self.config.regime_detection_method == "hybrid":
                # Use both detectors and combine results
                tas_result = self.tas_detector.detect_regimes(market_data) if self.tas_detector else None
                nas_result = self.nas_detector.detect_regimes(market_data) if self.nas_detector else None

                if tas_result and nas_result:
                    # Combine predictions
                    combined_regime = tas_result.regime_predictions[-1]
                    combined_probs = (tas_result.regime_probabilities[-1] + nas_result.regime_probabilities[-1]) / 2
                    return {
                        'regime_id': combined_regime,
                        'probabilities': combined_probs,
                        'confidence': np.max(combined_probs)
                    }
                elif tas_result:
                    return {
                        'regime_id': tas_result.regime_predictions[-1],
                        'probabilities': tas_result.regime_probabilities[-1],
                        'confidence': np.max(tas_result.regime_probabilities[-1])
                    }
                elif nas_result:
                    return {
                        'regime_id': nas_result.regime_predictions[-1],
                        'probabilities': nas_result.regime_probabilities[-1],
                        'confidence': np.max(nas_result.regime_probabilities[-1])
                    }

            elif self.config.regime_detection_method == "hybrid_nas_tas" and self.hybrid_detector:
                # Use the advanced hybrid NAS-TAS detector
                result = self.hybrid_detector.detect_regimes(market_data)
                if result.success:
                    return {
                        'regime_id': result.regime_predictions[-1],
                        'probabilities': result.regime_probabilities[-1],
                        'confidence': np.max(result.regime_probabilities[-1]),
                        'economic_significance': result.economic_significance_scores[result.regime_predictions[-1]],
                        'financial_relevance': result.financial_relevance_scores[result.regime_predictions[-1]]
                    }
            
            # Fallback to simple regime detection
            return self._fallback_regime_detection(market_data)
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return {'regime_id': 0, 'probabilities': np.array([1.0, 0.0, 0.0]), 'confidence': 1.0}
    
    def _fallback_regime_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback regime detection using simple heuristics."""
        try:
            # Simple regime detection based on price volatility
            if 'close' in market_data.columns:
                prices = market_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                if volatility < 0.01:
                    regime_id = 0  # Low volatility regime
                elif volatility < 0.03:
                    regime_id = 1  # Medium volatility regime
                else:
                    regime_id = 2  # High volatility regime
            else:
                regime_id = 0  # Default regime
            
            return {
                'regime_id': regime_id,
                'probabilities': np.array([1.0 if i == regime_id else 0.0 for i in range(3)]),
                'confidence': 1.0
            }
            
        except Exception as e:
            self.logger.warning(f"Fallback regime detection failed: {e}")
            return {'regime_id': 0, 'probabilities': np.array([1.0, 0.0, 0.0]), 'confidence': 1.0}
    
    def _select_best_performance_model(self, 
                                     regime_models: Dict[str, Any],
                                     market_data: pd.DataFrame,
                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select model based on best historical performance."""
        best_model = None
        best_score = -1
        best_model_type = None
        best_model_id = None
        
        for model_type, model_info in regime_models.items():
            # Get performance score
            performance = model_info['performance']
            score = performance.get(self.config.performance_metric, 0.0)

            if score > best_score:
                best_score = score
                best_model = model_info['model']
                best_model_type = model_type
                best_model_id = model_info['model_id']

        if best_model is None:
            raise ValueError("No models available for selection")

        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': best_model_id,
            'confidence': best_score,
            'reason': f"Best {self.config.performance_metric}: {best_score:.3f}"
        }

    def _select_ensemble_model(self,
                             regime_id: int,
                             regime_models: Dict[str, Any],
                             market_data: pd.DataFrame,
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select ensemble model."""
        if not self.config.enable_ensemble:
            return self._select_best_performance_model(regime_models, market_data, context)

        try:
            from sklearn.ensemble import VotingClassifier

            # Create ensemble from available models
            base_models = []
            model_weights = []
            base_model_ids = []

            for model_type, model_info in regime_models.items():
                base_models.append((model_type, model_info['model']))
                # Weight by performance
                performance = model_info['performance']
                weight = performance.get(self.config.performance_metric, 0.5)
                model_weights.append(weight)
                base_model_ids.append(model_info['model_id'])

            if len(base_models) < 2:
                # Not enough models for ensemble, use best single model
                return self._select_best_performance_model(regime_models, market_data, context)

            # Normalize weights
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]

            # Try to find a matching pre-trained ensemble
            ensemble_registry = self.available_models.get('ensemble', {})
            for ensemble_name, ensemble_info in ensemble_registry.items():
                registered_base_models = ensemble_info.get('base_models', [])
                if registered_base_models and not set(registered_base_models).issubset(set(base_model_ids)):
                    continue

                performance = ensemble_info.get('performance', {})
                confidence = performance.get(self.config.performance_metric, np.mean(model_weights))

                return {
                    'model': ensemble_info['model'],
                    'model_type': 'ensemble',
                    'model_id': ensemble_info['model_id'],
                    'confidence': confidence,
                    'reason': f"Pre-trained ensemble {ensemble_name}",
                    'ensemble_weights': ensemble_info.get('weights')
                }

            # No pre-trained ensemble available - attempt to build one on the fly
            training_data = None
            if context:
                training_data = context.get('training_data')

            if training_data is None:
                self.logger.warning(
                    "No pre-trained ensemble available and no training data provided; falling back to best model"
                )
                return self._select_best_performance_model(regime_models, market_data, context)

            # Extract training data (supports tuple/list or dict with keys)
            if isinstance(training_data, (tuple, list)) and len(training_data) >= 2:
                X_train, y_train = training_data[0], training_data[1]
            elif isinstance(training_data, dict):
                X_train = training_data.get('X') or training_data.get('features')
                y_train = training_data.get('y') or training_data.get('labels')
            else:
                raise ValueError("Unsupported training_data format for ensemble fitting")

            if X_train is None or y_train is None:
                raise ValueError("Training data for ensemble fitting must include features (X) and labels (y)")

            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                weights=model_weights
            )

            ensemble.fit(X_train, y_train)

            # Calculate ensemble confidence
            ensemble_confidence = np.mean(model_weights)

            # Create deterministic ensemble identifier based on regime and base model ids
            sorted_ids = "_".join(sorted(base_model_ids))
            ensemble_name = f"auto_regime_{regime_id}_{hash(sorted_ids) & 0xffff:x}"
            ensemble_id = f"ensemble_{ensemble_name}"

            # Register the newly created ensemble for future reuse
            self.available_models.setdefault('ensemble', {})[ensemble_name] = {
                'model': ensemble,
                'model_id': ensemble_id,
                'performance': {},
                'base_models': base_model_ids,
                'weights': dict(zip([name for name, _ in base_models], model_weights))
            }
            self.performance_history.setdefault(ensemble_id, [])

            return {
                'model': ensemble,
                'model_type': 'ensemble',
                'model_id': ensemble_id,
                'confidence': ensemble_confidence,
                'reason': f"Auto-ensemble of {len(base_models)} models",
                'ensemble_weights': dict(zip([name for name, _ in base_models], model_weights))
            }

        except Exception as e:
            self.logger.warning(f"Ensemble selection failed: {e}")
            return self._select_best_performance_model(regime_models, market_data, context)
    
    def _select_adaptive_model(self, 
                            regime_models: Dict[str, Any],
                            market_data: pd.DataFrame,
                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select model using adaptive learning."""
        if not self.config.enable_adaptive_selection:
            return self._select_best_performance_model(regime_models, market_data, context)
        
        # Initialize adaptation weights if not exists
        if not self.adaptation_weights:
            for model_type in regime_models.keys():
                self.adaptation_weights[model_type] = 1.0
        
        # Calculate adaptive scores
        adaptive_scores = {}
        for model_type, model_info in regime_models.items():
            base_performance = model_info['performance'].get(self.config.performance_metric, 0.0)
            adaptation_weight = self.adaptation_weights.get(model_type, 1.0)
            adaptive_scores[model_type] = base_performance * adaptation_weight
        
        # Select best adaptive model
        best_model_type = max(adaptive_scores.keys(), key=lambda x: adaptive_scores[x])
        best_model = regime_models[best_model_type]['model']
        best_model_id = regime_models[best_model_type]['model_id']

        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': best_model_id,
            'confidence': adaptive_scores[best_model_type],
            'reason': f"Adaptive selection: {best_model_type} (weight: {self.adaptation_weights.get(best_model_type, 1.0):.3f})"
        }
    
    def _select_meta_learning_model(self, 
                                  regime_models: Dict[str, Any],
                                  market_data: pd.DataFrame,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select model using meta-learning."""
        # Extract meta-features
        meta_features = self._extract_meta_features(market_data, context)
        
        # Calculate meta-learning scores
        meta_scores = {}
        for model_type, model_info in regime_models.items():
            # Simple meta-learning: weight by feature similarity
            similarity_score = self._calculate_feature_similarity(meta_features, model_type)
            base_performance = model_info['performance'].get(self.config.performance_metric, 0.0)
            meta_scores[model_type] = base_performance * similarity_score
        
        # Select best meta-learning model
        best_model_type = max(meta_scores.keys(), key=lambda x: meta_scores[x])
        best_model = regime_models[best_model_type]['model']
        best_model_id = regime_models[best_model_type]['model_id']

        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': best_model_id,
            'confidence': meta_scores[best_model_type],
            'reason': f"Meta-learning selection: {best_model_type}"
        }
    
    def _select_confidence_based_model(self, 
                                     regime_models: Dict[str, Any],
                                     market_data: pd.DataFrame,
                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select model based on prediction confidence."""
        confidence_scores = {}
        
        for model_type, model_info in regime_models.items():
            model = model_info['model']
            
            # Get prediction confidence
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(market_data.iloc[-1:].values)
                    if len(proba[0]) > 0:
                        confidence = np.max(proba[0])
                    else:
                        confidence = 0.5
                        self.logger.warning(f"Model {model_type} returned empty probability array")
                except (ValueError, IndexError, TypeError) as e:
                    confidence = 0.5
                    self.logger.warning(f"Could not calculate confidence for {model_type}: {e}")
                except Exception as e:
                    confidence = 0.0
                    self.logger.error(f"Unexpected error calculating confidence for {model_type}: {e}")
            else:
                confidence = 0.5
                self.logger.debug(f"Model {model_type} doesn't support predict_proba")
            
            # Combine with historical performance
            base_performance = model_info['performance'].get(self.config.performance_metric, 0.0)
            confidence_scores[model_type] = base_performance * confidence
        
        # Select best confidence-based model
        best_model_type = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
        best_model = regime_models[best_model_type]['model']
        best_model_id = regime_models[best_model_type]['model_id']

        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': best_model_id,
            'confidence': confidence_scores[best_model_type],
            'reason': f"Confidence-based selection: {best_model_type}"
        }
    
    def _extract_meta_features(self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract meta-features for model selection."""
        meta_features = {}
        
        try:
            # Market volatility
            if 'close' in market_data.columns:
                prices = market_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                meta_features['market_volatility'] = np.std(returns)
                meta_features['trend_strength'] = np.mean(returns)
            else:
                meta_features['market_volatility'] = 0.0
                meta_features['trend_strength'] = 0.0
            
            # Volume ratio
            if 'volume' in market_data.columns:
                volumes = market_data['volume'].values
                meta_features['volume_ratio'] = volumes[-1] / np.mean(volumes) if len(volumes) > 0 else 1.0
            else:
                meta_features['volume_ratio'] = 1.0
            
            # Time of day (if timestamp available)
            if context and 'timestamp' in context:
                hour = context['timestamp'].hour
                meta_features['time_of_day'] = hour / 24.0
            else:
                meta_features['time_of_day'] = 0.5
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"Meta-feature extraction failed: {e}")
            # Default meta-features
            meta_features = {
                'market_volatility': 0.02,
                'trend_strength': 0.0,
                'volume_ratio': 1.0,
                'time_of_day': 0.5
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during meta-feature extraction: {e}")
            # Default meta-features
            meta_features = {
                'market_volatility': 0.02,
                'trend_strength': 0.0,
                'volume_ratio': 1.0,
                'time_of_day': 0.5
            }
        
        return meta_features
    
    def _calculate_feature_similarity(self, meta_features: Dict[str, float], model_type: str) -> float:
        """Calculate similarity between meta-features and model type."""
        # Simple similarity based on model type characteristics
        similarity_map = {
            'xgboost': 0.8,  # Good for complex patterns
            'lightgbm': 0.8,  # Good for large datasets
            'catboost': 0.7,  # Good for categorical features
            'random_forest': 0.6,  # Good for general purpose
            'logistic_regression': 0.4,  # Good for linear relationships
            'svm': 0.5  # Good for non-linear relationships
        }
        
        base_similarity = similarity_map.get(model_type, 0.5)
        
        # Adjust based on meta-features
        volatility = meta_features.get('market_volatility', 0.02)
        if volatility > 0.03:  # High volatility
            if model_type in ['xgboost', 'lightgbm']:
                base_similarity += 0.1
        elif volatility < 0.01:  # Low volatility
            if model_type in ['logistic_regression', 'svm']:
                base_similarity += 0.1
        
        return min(base_similarity, 1.0)
    
    def _calculate_expected_performance(self, model_id: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate expected performance for selected model."""
        # Get historical performance
        historical_perf = self._get_historical_performance(model_id)
        
        # Simple expected performance (could be enhanced with more sophisticated methods)
        expected_perf = {
            'f1_score': historical_perf.get('f1_score', 0.5),
            'accuracy': historical_perf.get('accuracy', 0.5),
            'precision': historical_perf.get('precision', 0.5),
            'recall': historical_perf.get('recall', 0.5)
        }
        
        return expected_perf
    
    def _get_historical_performance(self, model_id: str) -> Dict[str, float]:
        """Get historical performance for model."""
        if model_id not in self.performance_history:
            return {'f1_score': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5}
        
        recent_performance = self.performance_history[model_id][-self.config.performance_window:]
        
        if not recent_performance:
            return {'f1_score': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5}
        
        return {
            'f1_score': np.mean([p.get('f1_score', 0.5) for p in recent_performance]),
            'accuracy': np.mean([p.get('accuracy', 0.5) for p in recent_performance]),
            'precision': np.mean([p.get('precision', 0.5) for p in recent_performance]),
            'recall': np.mean([p.get('recall', 0.5) for p in recent_performance])
        }
    
    def _calculate_confidence_interval(self, model_id: str, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for model performance."""
        if model_id not in self.performance_history:
            return (0.0, 1.0)
        
        recent_performance = self.performance_history[model_id][-self.config.performance_window:]
        
        if len(recent_performance) < 2:
            return (0.0, 1.0)
        
        f1_scores = [p.get('f1_score', 0.5) for p in recent_performance]
        
        # Simple confidence interval calculation
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        # 95% confidence interval (approximate)
        margin_of_error = 1.96 * std_f1 / np.sqrt(len(f1_scores))
        
        return (max(0.0, mean_f1 - margin_of_error), min(1.0, mean_f1 + margin_of_error))
    
    def _get_alternative_models(self,
                               regime_models: Dict[str, Any],
                               selected_model_id: str) -> List[Dict[str, Any]]:
        """Get alternative models for the regime."""
        alternatives = []

        for model_type, model_info in regime_models.items():
            if model_type not in selected_model_id:  # Different from selected
                alternatives.append({
                    'model_type': model_type,
                    'performance': model_info['performance'],
                    'confidence': model_info['performance'].get(self.config.performance_metric, 0.0)
                })

        # Sort by performance
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)

        # Return top K alternatives based on configuration
        top_k = getattr(self.config, 'top_k', 3)
        return alternatives[:top_k]

    def select_top_k_models(self,
                           market_data: pd.DataFrame,
                           regime_id: int,
                           k: int = 3) -> List[ModelSelectionResult]:
        """
        Select top K models for a specific regime/market.

        Args:
            market_data: Market data for selection
            regime_id: Regime ID
            k: Number of top models to select

        Returns:
            List of ModelSelectionResult objects for top K models
        """
        tprint_info(f"🎯 Selecting top {k} models for regime {regime_id}")

        # Get models for this regime
        regime_models = self._get_models_for_regime_and_direction(regime_id, 'long')  # Default to long for selection

        if not regime_models:
            tprint_warning(f"⚠️ No models available for regime {regime_id}")
            return []

        top_models = []

        # Select top K models using different strategies
        for i in range(min(k, len(regime_models))):
            try:
                selection_result = self.select_model(
                    market_data=market_data,
                    current_regime=regime_id
                )

                if selection_result and selection_result.selected_model:
                    top_models.append(selection_result)

                    # Remove selected model from available models for next iteration
                    selected_type = selection_result.selected_model_type
                    if selected_type in regime_models:
                        del regime_models[selected_type]

                    tprint_info(f"✅ Selected model {i+1}/{k}: {selected_type}")
                else:
                    tprint_warning(f"⚠️ Failed to select model {i+1}/{k} for regime {regime_id}")
                    break

            except Exception as e:
                tprint_error(f"❌ Failed to select model {i+1}/{k} for regime {regime_id}: {e}")
                break

        tprint_success(f"✅ Selected {len(top_models)}/{k} top models for regime {regime_id}")
        return top_models
    
    def _get_model_rankings(self, regime_models: Dict[str, Any]) -> Dict[str, int]:
        """Get model rankings for the regime."""
        rankings = {}
        
        # Sort models by performance
        sorted_models = sorted(
            regime_models.items(),
            key=lambda x: x[1]['performance'].get(self.config.performance_metric, 0.0),
            reverse=True
        )
        
        for rank, (model_type, _) in enumerate(sorted_models, 1):
            rankings[model_type] = rank
        
        return rankings
    
    def _update_selection_history(self, result: ModelSelectionResult):
        """Update selection history."""
        history_entry = {
            'timestamp': result.selection_time.isoformat(),
            'selected_model': result.selected_model_type,
            'selected_regime': result.selected_regime,
            'confidence': result.selection_confidence,
            'expected_performance': result.expected_performance,
            'reason': result.selection_reason
        }
        
        self.selection_history.append(history_entry)
        
        # Save to file if configured
        if self.config.save_selection_history:
            try:
                history_path = Path(self.config.selection_history_path)
                history_path.parent.mkdir(parents=True, exist_ok=True)

                with open(history_path, 'w') as f:
                    json.dump(self.selection_history, f, indent=2)

                self.logger.debug(f"Selection history saved to {history_path}")
            except (IOError, OSError, json.JSONEncodeError) as e:
                self.logger.warning(f"Could not save selection history: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error saving selection history: {e}")
    
    def _update_adaptive_learning(self, result: ModelSelectionResult, market_data: pd.DataFrame):
        """Update adaptive learning weights."""
        if not self.config.enable_adaptive_selection:
            return
        
        # Update weights based on selection outcome
        selected_model_type = result.selected_model_type
        
        if selected_model_type not in self.adaptation_weights:
            self.adaptation_weights[selected_model_type] = 1.0
        
        # Increase weight for selected model
        self.adaptation_weights[selected_model_type] += self.config.adaptation_rate
        
        # Normalize weights to prevent explosion
        total_weight = sum(self.adaptation_weights.values())
        for model_type in self.adaptation_weights:
            self.adaptation_weights[model_type] /= total_weight
    
    def _get_fallback_model(self, market_data: pd.DataFrame, current_regime: Optional[int]) -> ModelSelectionResult:
        """Get fallback model when selection fails."""
        self.logger.warning("⚠️ Using fallback model")
        
        # Try to find any available model
        for regime_id, models in self.available_models.items():
            if models:
                model_type = list(models.keys())[0]
                model_info = models[model_type]
                
                return ModelSelectionResult(
                    selected_model=model_info['model'],
                    selected_model_type=model_type,
                    selected_regime=regime_id,
                    selection_confidence=0.5,
                    selection_reason="Fallback model",
                    expected_performance={'f1_score': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5},
                    historical_performance={'f1_score': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5},
                    confidence_interval=(0.0, 1.0),
                    alternative_models=[],
                    selection_time=datetime.now()
                )
        
        # No models available
        raise ValueError("No models available for fallback")
    
    def update_model_performance(self, 
                               model_id: str,
                               performance_metrics: Dict[str, float]):
        """Update model performance with new metrics."""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        # Add new performance metrics
        self.performance_history[model_id].append(performance_metrics)
        
        # Keep only recent performance (sliding window)
        if len(self.performance_history[model_id]) > self.config.performance_window:
            self.performance_history[model_id] = self.performance_history[model_id][-self.config.performance_window:]
        
        self.logger.info(f"📊 Updated performance for {model_id}: F1={performance_metrics.get('f1_score', 0):.3f}")
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about model selection."""
        if not self.selection_history:
            return {}
        
        # Calculate selection statistics
        total_selections = len(self.selection_history)
        model_counts = {}
        regime_counts = {}
        avg_confidence = 0.0
        
        for entry in self.selection_history:
            model_type = entry['selected_model']
            regime = entry['selected_regime']
            confidence = entry['confidence']
            
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            avg_confidence += confidence
        
        avg_confidence /= total_selections
        
        return {
            'total_selections': total_selections,
            'model_distribution': model_counts,
            'regime_distribution': regime_counts,
            'average_confidence': avg_confidence,
            'most_selected_model': max(model_counts.keys(), key=lambda x: model_counts[x]) if model_counts else None,
            'most_selected_regime': max(regime_counts.keys(), key=lambda x: regime_counts[x]) if regime_counts else None
        }
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance across all models."""
        summary = {}
        
        for model_id, performance_history in self.performance_history.items():
            if not performance_history:
                continue
            
            recent_performance = performance_history[-self.config.performance_window:]
            
            if recent_performance:
                f1_scores = [p.get('f1_score', 0) for p in recent_performance]
                summary[model_id] = {
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'min_f1': np.min(f1_scores),
                    'max_f1': np.max(f1_scores),
                    'n_predictions': len(recent_performance)
                }
        
        return summary

    def _detect_direction(self, market_data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> str:
        """Detect trading direction from market data."""
        try:
            # Check for explicit direction indicators in data
            direction_columns = [col for col in market_data.columns if any(keyword in col.lower() for keyword in ['direction', 'long', 'short'])]

            if direction_columns:
                for col in direction_columns:
                    if market_data[col].iloc[-1] == 1:  # Assuming binary indicator
                        if 'long' in col.lower():
                            return 'long'
                        elif 'short' in col.lower():
                            return 'short'

            # Check context for direction
            if context and 'direction' in context:
                return context['direction']

            # Infer direction from recent price movement
            if 'close' in market_data.columns and len(market_data) >= 2:
                recent_prices = market_data['close'].tail(5)
                price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

                if price_change > 0.001:  # 0.1% positive change
                    return 'long'
                elif price_change < -0.001:  # 0.1% negative change
                    return 'short'

            # Default to long if unclear
            return 'long'

        except Exception as e:
            self.logger.warning(f"Direction detection failed: {e}")
            return 'long'

    def _get_models_for_regime_and_direction(self, regime_id: int, direction: str) -> Dict[str, Any]:
        """Get models for specific regime and direction."""
        # Try directional models first
        direction_key = f"{direction}_models"
        if direction_key in self.available_models:
            direction_regime_key = f"{direction}_regime_{regime_id}"
            if direction_regime_key in self.available_models[direction_key]:
                return self.available_models[direction_key][direction_regime_key]

        # Fall back to standard regime models
        if regime_id in self.available_models:
            return self.available_models[regime_id]

        return {}

    def select_model_for_direction(self,
                                 market_data: pd.DataFrame,
                                 direction: str,
                                 current_regime: Optional[int] = None,
                                 context: Optional[Dict[str, Any]] = None) -> ModelSelectionResult:
        """
        Select model specifically for a trading direction.

        Args:
            market_data: Current market data
            direction: Trading direction ('long' or 'short')
            current_regime: Current regime (None for auto-detection)
            context: Additional context

        Returns:
            ModelSelectionResult with selected model for the direction
        """
        return self.select_model(market_data, current_regime, context, direction)

    def get_directional_model_summary(self) -> Dict[str, Any]:
        """Get summary of available directional models."""
        summary = {
            'directions_available': [],
            'directional_models': {},
            'total_directional_models': 0
        }

        for key, models in self.available_models.items():
            if '_models' in key and key != 'ensemble':
                direction = key.replace('_models', '')
                summary['directions_available'].append(direction)
                summary['directional_models'][direction] = len(models)
                summary['total_directional_models'] += len(models)

        return summary