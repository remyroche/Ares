"""
Model Selection and Routing System

Intelligent model selection and routing system that automatically selects
the best model for each market regime based on performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
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
    ensemble_weights: Optional[Dict[str, float]] = None
    
    # Metadata
    selection_time: datetime
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
            if self.config.regime_detection_method in ["tas", "hybrid"]:
                tas_config = TASRegimeConfig(
                    n_regimes=8,
                    enable_economic_evaluation=True,
                    enable_uncertainty_quantification=True
                )
                self.tas_detector = TASRegimeDetector(tas_config)
                self.logger.info("✅ TAS regime detector initialized")
            
            if self.config.regime_detection_method in ["nas", "hybrid"]:
                nas_config = PerfectNASConfig.create_short_term_trading_config()
                self.nas_detector = PerfectNASRegimeDetector(nas_config)
                self.logger.info("✅ NAS regime detector initialized")
                
        except Exception as e:
            self.logger.warning(f"Regime detection initialization failed: {e}")
            self.tas_detector = None
            self.nas_detector = None
    
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
                      ensemble_models: Optional[Dict[str, Any]] = None):
        """
        Register trained models for selection.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
        """
        self.logger.info("📝 Registering models for selection")
        
        # Register regime models
        for regime_id, models in regime_models.items():
            self.available_models[regime_id] = {}
            
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
                    'base_models': ensemble_info.get('base_models', [])
                }
                
                self.performance_history[ensemble_id] = []
                self.logger.info(f"   ✅ Registered ensemble {ensemble_name}")
        
        self.logger.info(f"📊 Total models registered: {sum(len(models) for models in self.available_models.values())}")
    
    def select_model(self, 
                    market_data: pd.DataFrame,
                    current_regime: Optional[int] = None,
                    context: Optional[Dict[str, Any]] = None) -> ModelSelectionResult:
        """
        Select the best model for given market conditions.
        
        Args:
            market_data: Current market data
            current_regime: Current regime (None for auto-detection)
            context: Additional context for selection
            
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
            
            # Step 2: Get available models for regime
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
                    regime_models, market_data, context
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
            
            # Step 5: Get alternative models
            alternative_models = self._get_alternative_models(
                regime_models, selection_result['model_id']
            )
            
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
        
        for model_type, model_info in regime_models.items():
            # Get performance score
            performance = model_info['performance']
            score = performance.get(self.config.performance_metric, 0.0)
            
            if score > best_score:
                best_score = score
                best_model = model_info['model']
                best_model_type = model_type
        
        if best_model is None:
            raise ValueError("No models available for selection")
        
        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': f"regime_{best_model_type}",
            'confidence': best_score,
            'reason': f"Best {self.config.performance_metric}: {best_score:.3f}"
        }
    
    def _select_ensemble_model(self, 
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
            
            for model_type, model_info in regime_models.items():
                base_models.append((model_type, model_info['model']))
                # Weight by performance
                performance = model_info['performance']
                weight = performance.get(self.config.performance_metric, 0.5)
                model_weights.append(weight)
            
            if len(base_models) < 2:
                # Not enough models for ensemble, use best single model
                return self._select_best_performance_model(regime_models, market_data, context)
            
            # Normalize weights
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]
            
            # Create ensemble
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                weights=model_weights
            )
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(model_weights)
            
            return {
                'model': ensemble,
                'model_type': 'ensemble',
                'model_id': 'ensemble_model',
                'confidence': ensemble_confidence,
                'reason': f"Ensemble of {len(base_models)} models",
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
        
        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': f"adaptive_{best_model_type}",
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
        
        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': f"meta_{best_model_type}",
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
                    confidence = np.max(proba[0])
                except:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            # Combine with historical performance
            base_performance = model_info['performance'].get(self.config.performance_metric, 0.0)
            confidence_scores[model_type] = base_performance * confidence
        
        # Select best confidence-based model
        best_model_type = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
        best_model = regime_models[best_model_type]['model']
        
        return {
            'model': best_model,
            'model_type': best_model_type,
            'model_id': f"confidence_{best_model_type}",
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
            
        except Exception as e:
            self.logger.warning(f"Meta-feature extraction failed: {e}")
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
        
        return alternatives[:3]  # Return top 3 alternatives
    
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
            except Exception as e:
                self.logger.warning(f"Failed to save selection history: {e}")
    
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