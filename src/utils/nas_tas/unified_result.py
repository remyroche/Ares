"""
Unified Regime Detection Result

Provides a unified result structure for both TAS and NAS regime detection systems.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class UnifiedRegimeResult:
    """Unified result from regime detection systems."""
    
    # Core results
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    trading_viability_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    
    # Optional advanced results
    micro_regimes: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    uncertainty_estimates: Optional[np.ndarray] = None
    enhanced_features: Optional[np.ndarray] = None
    
    # Execution metadata
    execution_time: float = 0.0
    system_type: str = "unified"
    architecture_used: str = "hybrid"
    
    # Metadata and error handling
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate result after initialization."""
        self._validate_result()
    
    def _validate_result(self):
        """Validate result data."""
        if not self.success:
            return
        
        # Validate array shapes
        if len(self.regime_predictions) != len(self.regime_probabilities):
            raise ValueError("Mismatch between predictions and probabilities length")
        
        if len(self.economic_significance_scores) != len(self.regime_predictions):
            raise ValueError("Mismatch between predictions and economic scores length")
        
        if len(self.trading_viability_scores) != len(self.regime_predictions):
            raise ValueError("Mismatch between predictions and trading scores length")
        
        if len(self.regime_stability_scores) != len(self.regime_predictions):
            raise ValueError("Mismatch between predictions and stability scores length")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        if not self.success:
            return {
                'success': False,
                'error': self.error_message,
                'execution_time': self.execution_time
            }
        
        return {
            'success': True,
            'n_samples': len(self.regime_predictions),
            'n_regimes': len(np.unique(self.regime_predictions)),
            'mean_economic_significance': float(np.mean(self.economic_significance_scores)),
            'mean_trading_viability': float(np.mean(self.trading_viability_scores)),
            'mean_regime_stability': float(np.mean(self.regime_stability_scores)),
            'execution_time': self.execution_time,
            'system_type': self.system_type,
            'architecture_used': self.architecture_used,
            'has_micro_regimes': self.micro_regimes is not None,
            'has_uncertainty_estimates': self.uncertainty_estimates is not None,
            'has_enhanced_features': self.enhanced_features is not None
        }
    
    def get_regime_distribution(self) -> Dict[str, int]:
        """Get distribution of regime predictions."""
        if not self.success:
            return {}
        
        unique_regimes, counts = np.unique(self.regime_predictions, return_counts=True)
        return {f'regime_{regime}': int(count) for regime, count in zip(unique_regimes, counts)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.success or self.performance_metrics is None:
            return {}
        
        return {
            'accuracy': self.performance_metrics.get('accuracy', 0.0),
            'precision': self.performance_metrics.get('precision', 0.0),
            'recall': self.performance_metrics.get('recall', 0.0),
            'f1_score': self.performance_metrics.get('f1_score', 0.0),
            'method': self.performance_metrics.get('method', 'unknown')
        }
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics for the results."""
        if not self.success:
            return {}
        
        return {
            'mean_economic_significance': float(np.mean(self.economic_significance_scores)),
            'std_economic_significance': float(np.std(self.economic_significance_scores)),
            'mean_trading_viability': float(np.mean(self.trading_viability_scores)),
            'std_trading_viability': float(np.std(self.trading_viability_scores)),
            'mean_regime_stability': float(np.mean(self.regime_stability_scores)),
            'std_regime_stability': float(np.std(self.regime_stability_scores)),
            'regime_transitions': int(np.sum(np.diff(self.regime_predictions) != 0)) if len(self.regime_predictions) > 1 else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'regime_predictions': self.regime_predictions.tolist() if self.regime_predictions is not None else None,
            'regime_probabilities': self.regime_probabilities.tolist() if self.regime_probabilities is not None else None,
            'economic_significance_scores': self.economic_significance_scores.tolist() if self.economic_significance_scores is not None else None,
            'trading_viability_scores': self.trading_viability_scores.tolist() if self.trading_viability_scores is not None else None,
            'regime_stability_scores': self.regime_stability_scores.tolist() if self.regime_stability_scores is not None else None,
            'transition_probabilities': self.transition_probabilities.tolist() if self.transition_probabilities is not None else None,
            'micro_regimes': self.micro_regimes,
            'performance_metrics': self.performance_metrics,
            'uncertainty_estimates': self.uncertainty_estimates.tolist() if self.uncertainty_estimates is not None else None,
            'enhanced_features': self.enhanced_features.tolist() if self.enhanced_features is not None else None,
            'execution_time': self.execution_time,
            'system_type': self.system_type,
            'architecture_used': self.architecture_used,
            'metadata': self.metadata,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedRegimeResult':
        """Create result from dictionary."""
        return cls(
            success=data.get('success', False),
            regime_predictions=np.array(data.get('regime_predictions', [])),
            regime_probabilities=np.array(data.get('regime_probabilities', [])),
            economic_significance_scores=np.array(data.get('economic_significance_scores', [])),
            trading_viability_scores=np.array(data.get('trading_viability_scores', [])),
            regime_stability_scores=np.array(data.get('regime_stability_scores', [])),
            transition_probabilities=np.array(data.get('transition_probabilities', [])),
            micro_regimes=data.get('micro_regimes'),
            performance_metrics=data.get('performance_metrics'),
            uncertainty_estimates=np.array(data.get('uncertainty_estimates', [])) if data.get('uncertainty_estimates') else None,
            enhanced_features=np.array(data.get('enhanced_features', [])) if data.get('enhanced_features') else None,
            execution_time=data.get('execution_time', 0.0),
            system_type=data.get('system_type', 'unified'),
            architecture_used=data.get('architecture_used', 'hybrid'),
            metadata=data.get('metadata'),
            error_message=data.get('error_message')
        )