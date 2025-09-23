"""
Adaptive Regime Learner

Advanced meta-learning system for adaptive regime detection.
Combines few-shot learning, continual learning, and uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import time

from ..core.perfect_nas_config import MetaLearningConfig

logger = logging.getLogger(__name__)

@dataclass
class AdaptationResult:
    """Result from regime adaptation."""
    success: bool
    adaptation_accuracy: float
    uncertainty_estimate: float
    adaptation_time: float
    regime_confidence: float
    meta_learning_metrics: Dict[str, float]
    error_message: Optional[str] = None

class AdaptiveRegimeLearner:
    """
    Adaptive regime learner with meta-learning capabilities.
    
    Features:
    - Few-shot learning for new regimes
    - Continual learning for regime evolution
    - Uncertainty estimation for predictions
    - Memory replay for knowledge retention
    - Meta-optimization for architecture adaptation
    """
    
    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        """Initialize adaptive regime learner.
        
        Args:
            base_model: Base neural network model
            config: Meta-learning configuration
        """
        self.base_model = base_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Meta-learning components
        self._initialize_meta_learning_components()
        
        # Memory systems
        self.episodic_memory = deque(maxlen=config.memory_size)
        self.regime_memory = defaultdict(list)
        self.adaptation_history = []
        
        # Performance tracking
        self.performance_tracker = defaultdict(list)
        self.uncertainty_tracker = deque(maxlen=100)
        
        # Current state
        self.current_regime = None
        self.regime_confidence = 0.0
        self.adaptation_count = 0
        
        self.logger.info("✅ Adaptive Regime Learner initialized")
        self.logger.info(f"   Memory size: {config.memory_size}")
        self.logger.info(f"   Adaptation steps: {config.adaptation_steps}")
        self.logger.info(f"   Meta-learning rate: {config.meta_learning_rate}")
    
    def _initialize_meta_learning_components(self):
        """Initialize meta-learning components."""
        try:
            # Few-shot learning components
            self.few_shot_learner = FewShotRegimeLearner(self.config)
            
            # Uncertainty estimation
            self.uncertainty_estimator = UncertaintyEstimator(
                self.base_model, 
                dropout_rate=0.1, 
                num_samples=10
            )
            
            # Continual learning
            self.continual_learner = ContinualLearningModel(
                self.base_model, 
                memory_size=self.config.memory_size
            )
            
            # Meta-optimizer
            self.meta_optimizer = MetaNAS_Optimizer(self.config)
            
            self.logger.info("✅ Meta-learning components initialized")
            
        except Exception as e:
            self.logger.error(f"Meta-learning component initialization failed: {e}")
            raise
    
    def adapt_to_new_regime(self, market_data: torch.Tensor, regime_labels: torch.Tensor,
                          regime_type: str = "unknown", adaptation_method: str = "few_shot") -> AdaptationResult:
        """Adapt to a new regime using meta-learning.
        
        Args:
            market_data: Market data for adaptation
            regime_labels: Regime labels for adaptation
            regime_type: Type of regime being adapted to
            adaptation_method: Method for adaptation ("few_shot", "continual", "meta_optimization")
            
        Returns:
            Adaptation result with performance metrics
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"🧠 Adapting to new regime: {regime_type}")
            self.logger.info(f"   Data shape: {market_data.shape}")
            self.logger.info(f"   Adaptation method: {adaptation_method}")
            
            # Perform adaptation based on method
            if adaptation_method == "few_shot":
                adaptation_result = self._few_shot_adaptation(market_data, regime_labels, regime_type)
            elif adaptation_method == "continual":
                adaptation_result = self._continual_adaptation(market_data, regime_labels, regime_type)
            elif adaptation_method == "meta_optimization":
                adaptation_result = self._meta_optimization_adaptation(market_data, regime_labels, regime_type)
            else:
                raise ValueError(f"Unknown adaptation method: {adaptation_method}")
            
            # Update state
            self.current_regime = regime_type
            self.adaptation_count += 1
            self.adaptation_history.append({
                'regime_type': regime_type,
                'method': adaptation_method,
                'accuracy': adaptation_result.adaptation_accuracy,
                'uncertainty': adaptation_result.uncertainty_estimate,
                'timestamp': time.time()
            })
            
            # Update memory
            self._update_memory(market_data, regime_labels, regime_type)
            
            adaptation_time = time.time() - start_time
            adaptation_result.adaptation_time = adaptation_time
            
            self.logger.info(f"✅ Regime adaptation completed in {adaptation_time:.2f}s")
            self.logger.info(f"   Adaptation accuracy: {adaptation_result.adaptation_accuracy:.4f}")
            self.logger.info(f"   Uncertainty estimate: {adaptation_result.uncertainty_estimate:.4f}")
            
            return adaptation_result
            
        except Exception as e:
            adaptation_time = time.time() - start_time
            self.logger.error(f"❌ Regime adaptation failed: {e}")
            
            return AdaptationResult(
                success=False,
                adaptation_accuracy=0.0,
                uncertainty_estimate=1.0,
                adaptation_time=adaptation_time,
                regime_confidence=0.0,
                meta_learning_metrics={},
                error_message=str(e)
            )
    
    def _few_shot_adaptation(self, market_data: torch.Tensor, regime_labels: torch.Tensor,
                           regime_type: str) -> AdaptationResult:
        """Perform few-shot adaptation."""
        try:
            # Create support and query sets
            support_size = min(self.config.num_shots, len(market_data) // 2)
            support_data = market_data[:support_size]
            support_labels = regime_labels[:support_size]
            query_data = market_data[support_size:]
            query_labels = regime_labels[support_size:]
            
            # Perform few-shot adaptation
            adaptation_results = self.few_shot_learner.few_shot_adaptation(
                (support_data, support_labels),
                (query_data, query_labels),
                regime_type=regime_type
            )
            
            # Calculate adaptation accuracy
            adaptation_accuracy = adaptation_results.get('maml_accuracy', 0.0)
            if adaptation_accuracy == 0.0:
                adaptation_accuracy = adaptation_results.get('prototypical_accuracy', 0.0)
            
            # Get uncertainty estimate
            uncertainty_estimate = adaptation_results.get('uncertainty_score', 0.5)
            
            # Calculate regime confidence
            regime_confidence = 1.0 - uncertainty_estimate
            
            return AdaptationResult(
                success=True,
                adaptation_accuracy=adaptation_accuracy,
                uncertainty_estimate=uncertainty_estimate,
                adaptation_time=0.0,  # Will be set by caller
                regime_confidence=regime_confidence,
                meta_learning_metrics=adaptation_results
            )
            
        except Exception as e:
            self.logger.warning(f"Few-shot adaptation failed: {e}")
            return AdaptationResult(
                success=False,
                adaptation_accuracy=0.0,
                uncertainty_estimate=1.0,
                adaptation_time=0.0,
                regime_confidence=0.0,
                meta_learning_metrics={},
                error_message=str(e)
            )
    
    def _continual_adaptation(self, market_data: torch.Tensor, regime_labels: torch.Tensor,
                            regime_type: str) -> AdaptationResult:
        """Perform continual learning adaptation."""
        try:
            # Update continual learning model
            self.continual_learner.update_memory(market_data, regime_labels)
            
            # Perform adaptation with memory replay
            adaptation_accuracy = self._evaluate_continual_adaptation(market_data, regime_labels)
            
            # Get uncertainty estimate
            with torch.no_grad():
                predictions = self.base_model(market_data)
                uncertainty_estimate = self._calculate_prediction_uncertainty(predictions)
            
            # Calculate regime confidence
            regime_confidence = 1.0 - uncertainty_estimate
            
            return AdaptationResult(
                success=True,
                adaptation_accuracy=adaptation_accuracy,
                uncertainty_estimate=uncertainty_estimate,
                adaptation_time=0.0,
                regime_confidence=regime_confidence,
                meta_learning_metrics={
                    'memory_size': len(self.continual_learner.episodic_memory),
                    'adaptation_method': 'continual_learning'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Continual adaptation failed: {e}")
            return AdaptationResult(
                success=False,
                adaptation_accuracy=0.0,
                uncertainty_estimate=1.0,
                adaptation_time=0.0,
                regime_confidence=0.0,
                meta_learning_metrics={},
                error_message=str(e)
            )
    
    def _meta_optimization_adaptation(self, market_data: torch.Tensor, regime_labels: torch.Tensor,
                                    regime_type: str) -> AdaptationResult:
        """Perform meta-optimization adaptation."""
        try:
            # Create meta-learning tasks
            meta_tasks = self._create_meta_learning_tasks(market_data, regime_labels)
            
            # Perform meta-optimization
            meta_optimization_result = self.meta_optimizer.meta_optimize_architecture(
                self.base_model, meta_tasks, meta_tasks
            )
            
            # Calculate adaptation accuracy
            adaptation_accuracy = meta_optimization_result.get('final_performance', 0.0)
            
            # Get uncertainty estimate
            uncertainty_estimate = self._calculate_meta_uncertainty(meta_optimization_result)
            
            # Calculate regime confidence
            regime_confidence = 1.0 - uncertainty_estimate
            
            return AdaptationResult(
                success=True,
                adaptation_accuracy=adaptation_accuracy,
                uncertainty_estimate=uncertainty_estimate,
                adaptation_time=0.0,
                regime_confidence=regime_confidence,
                meta_learning_metrics=meta_optimization_result
            )
            
        except Exception as e:
            self.logger.warning(f"Meta-optimization adaptation failed: {e}")
            return AdaptationResult(
                success=False,
                adaptation_accuracy=0.0,
                uncertainty_estimate=1.0,
                adaptation_time=0.0,
                regime_confidence=0.0,
                meta_learning_metrics={},
                error_message=str(e)
            )
    
    def _evaluate_continual_adaptation(self, market_data: torch.Tensor, 
                                     regime_labels: torch.Tensor) -> float:
        """Evaluate continual learning adaptation."""
        try:
            with torch.no_grad():
                predictions = self.base_model(market_data)
                predicted_labels = torch.argmax(predictions, dim=1)
                accuracy = (predicted_labels == regime_labels).float().mean()
                return accuracy.item()
                
        except Exception as e:
            self.logger.warning(f"Continual adaptation evaluation failed: {e}")
            return 0.0
    
    def _calculate_prediction_uncertainty(self, predictions: torch.Tensor) -> float:
        """Calculate prediction uncertainty."""
        try:
            # Use entropy as uncertainty measure
            probs = F.softmax(predictions, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            uncertainty = entropy.mean().item()
            
            # Normalize to 0-1 range
            uncertainty = min(uncertainty / np.log(predictions.shape[1]), 1.0)
            
            return uncertainty
            
        except Exception as e:
            self.logger.warning(f"Uncertainty calculation failed: {e}")
            return 0.5
    
    def _calculate_meta_uncertainty(self, meta_result: Dict[str, Any]) -> float:
        """Calculate uncertainty from meta-optimization result."""
        try:
            # Extract uncertainty from meta-learning result
            if 'uncertainty_estimates' in meta_result:
                return np.mean(meta_result['uncertainty_estimates'])
            else:
                # Use performance as inverse uncertainty
                performance = meta_result.get('final_performance', 0.5)
                return 1.0 - performance
                
        except Exception as e:
            self.logger.warning(f"Meta uncertainty calculation failed: {e}")
            return 0.5
    
    def _create_meta_learning_tasks(self, market_data: torch.Tensor, 
                                  regime_labels: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Create meta-learning tasks from data."""
        try:
            tasks = []
            
            # Create multiple tasks by splitting data
            n_tasks = 5
            task_size = len(market_data) // n_tasks
            
            for i in range(n_tasks):
                start_idx = i * task_size
                end_idx = start_idx + task_size if i < n_tasks - 1 else len(market_data)
                
                task_data = market_data[start_idx:end_idx]
                task_labels = regime_labels[start_idx:end_idx]
                
                # Split into support and query sets
                support_size = len(task_data) // 2
                support_data = task_data[:support_size]
                support_labels = task_labels[:support_size]
                query_data = task_data[support_size:]
                query_labels = task_labels[support_size:]
                
                task = {
                    'support_x': support_data,
                    'support_y': support_labels,
                    'query_x': query_data,
                    'query_y': query_labels
                }
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            self.logger.warning(f"Meta-learning task creation failed: {e}")
            return []
    
    def _update_memory(self, market_data: torch.Tensor, regime_labels: torch.Tensor, 
                      regime_type: str):
        """Update memory systems."""
        try:
            # Update episodic memory
            for i in range(len(market_data)):
                sample = {
                    'data': market_data[i].clone(),
                    'label': regime_labels[i].clone(),
                    'regime_type': regime_type,
                    'timestamp': time.time()
                }
                self.episodic_memory.append(sample)
            
            # Update regime-specific memory
            self.regime_memory[regime_type].extend([
                {'data': market_data[i].clone(), 'label': regime_labels[i].clone()}
                for i in range(len(market_data))
            ])
            
            # Limit regime memory size
            max_regime_memory = 100
            if len(self.regime_memory[regime_type]) > max_regime_memory:
                self.regime_memory[regime_type] = self.regime_memory[regime_type][-max_regime_memory:]
            
        except Exception as e:
            self.logger.warning(f"Memory update failed: {e}")
    
    def predict_with_uncertainty(self, market_data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Make predictions with uncertainty estimation."""
        try:
            with torch.no_grad():
                # Get predictions
                predictions = self.base_model(market_data)
                
                # Calculate uncertainty
                uncertainty = self._calculate_prediction_uncertainty(predictions)
                
                return predictions, uncertainty
                
        except Exception as e:
            self.logger.warning(f"Prediction with uncertainty failed: {e}")
            # Return dummy predictions
            dummy_predictions = torch.zeros(len(market_data), 5)  # Assume 5 regimes
            return dummy_predictions, 1.0
    
    def detect_regime_change(self, market_data: torch.Tensor, threshold: float = 0.1) -> bool:
        """Detect if regime has changed significantly."""
        try:
            if self.current_regime is None:
                return False
            
            # Get current predictions
            predictions, uncertainty = self.predict_with_uncertainty(market_data)
            
            # Regime change if uncertainty is high
            regime_change = uncertainty > threshold
            
            if regime_change:
                self.logger.info(f"🔄 Regime change detected (uncertainty: {uncertainty:.4f})")
            
            return regime_change
            
        except Exception as e:
            self.logger.warning(f"Regime change detection failed: {e}")
            return False
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime adaptations."""
        try:
            if not self.adaptation_history:
                return {}
            
            # Calculate statistics
            accuracies = [h['accuracy'] for h in self.adaptation_history]
            uncertainties = [h['uncertainty'] for h in self.adaptation_history]
            
            statistics = {
                'total_adaptations': len(self.adaptation_history),
                'average_accuracy': np.mean(accuracies),
                'average_uncertainty': np.mean(uncertainties),
                'accuracy_std': np.std(accuracies),
                'uncertainty_std': np.std(uncertainties),
                'current_regime': self.current_regime,
                'regime_confidence': self.regime_confidence,
                'memory_utilization': len(self.episodic_memory) / self.config.memory_size,
                'unique_regimes': len(self.regime_memory),
                'adaptation_trend': self._calculate_adaptation_trend()
            }
            
            return statistics
            
        except Exception as e:
            self.logger.warning(f"Adaptation statistics calculation failed: {e}")
            return {}
    
    def _calculate_adaptation_trend(self) -> str:
        """Calculate adaptation performance trend."""
        try:
            if len(self.adaptation_history) < 3:
                return "insufficient_data"
            
            recent_accuracies = [h['accuracy'] for h in self.adaptation_history[-3:]]
            earlier_accuracies = [h['accuracy'] for h in self.adaptation_history[-6:-3]]
            
            if len(earlier_accuracies) == 0:
                return "insufficient_data"
            
            recent_avg = np.mean(recent_accuracies)
            earlier_avg = np.mean(earlier_accuracies)
            
            if recent_avg > earlier_avg + 0.05:
                return "improving"
            elif recent_avg < earlier_avg - 0.05:
                return "declining"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Get analysis of memory systems."""
        try:
            analysis = {
                'episodic_memory': {
                    'size': len(self.episodic_memory),
                    'capacity': self.config.memory_size,
                    'utilization': len(self.episodic_memory) / self.config.memory_size
                },
                'regime_memory': {
                    'regimes': list(self.regime_memory.keys()),
                    'regime_counts': {k: len(v) for k, v in self.regime_memory.items()},
                    'total_samples': sum(len(v) for v in self.regime_memory.values())
                },
                'memory_quality': self._assess_memory_quality()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Memory analysis failed: {e}")
            return {}
    
    def _assess_memory_quality(self) -> Dict[str, float]:
        """Assess quality of memory systems."""
        try:
            quality_metrics = {
                'diversity': 0.0,
                'recency': 0.0,
                'consistency': 0.0
            }
            
            if not self.episodic_memory:
                return quality_metrics
            
            # Calculate diversity (number of unique regimes)
            unique_regimes = set(sample['regime_type'] for sample in self.episodic_memory)
            quality_metrics['diversity'] = len(unique_regimes) / 10.0  # Normalize
            
            # Calculate recency (how recent are the samples)
            current_time = time.time()
            recent_samples = sum(1 for sample in self.episodic_memory 
                               if current_time - sample['timestamp'] < 3600)  # Last hour
            quality_metrics['recency'] = recent_samples / len(self.episodic_memory)
            
            # Calculate consistency (how consistent are the labels)
            if len(self.episodic_memory) > 1:
                labels = [sample['label'].item() for sample in self.episodic_memory]
                label_consistency = 1.0 - (np.std(labels) / (np.mean(labels) + 1e-8))
                quality_metrics['consistency'] = max(0.0, min(1.0, label_consistency))
            
            return quality_metrics
            
        except Exception:
            return {'diversity': 0.0, 'recency': 0.0, 'consistency': 0.0}

# Placeholder classes for meta-learning components
class FewShotRegimeLearner:
    """Placeholder for few-shot regime learner."""
    def __init__(self, config):
        self.config = config
    
    def few_shot_adaptation(self, support_set, query_set, regime_type):
        return {'maml_accuracy': 0.8, 'prototypical_accuracy': 0.7, 'uncertainty_score': 0.2}

class UncertaintyEstimator:
    """Placeholder for uncertainty estimator."""
    def __init__(self, model, dropout_rate, num_samples):
        self.model = model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples

class ContinualLearningModel:
    """Placeholder for continual learning model."""
    def __init__(self, model, memory_size):
        self.model = model
        self.episodic_memory = []

class MetaNAS_Optimizer:
    """Placeholder for meta-NAS optimizer."""
    def __init__(self, config):
        self.config = config
    
    def meta_optimize_architecture(self, model, train_tasks, test_tasks):
        return {'final_performance': 0.8, 'uncertainty_estimates': [0.2, 0.3, 0.1]}