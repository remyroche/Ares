"""
TAS Result Classes

Result classes for the Tree Architecture Search system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from .tree_architecture import TreeArchitectureCandidate


@dataclass
class TASResult:
    """Base result class for TAS operations."""
    
    # Core results
    best_architecture: Optional[TreeArchitectureCandidate] = None
    best_score: float = 0.0
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    execution_time: float = 0.0
    n_evaluations: int = 0
    convergence_iteration: Optional[int] = None
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[Dict[str, Any]] = None
    
    # Advanced results
    uncertainty_estimates: Optional[Dict[str, float]] = None
    regime_analysis: Optional[Dict[str, Any]] = None
    multi_objective_results: Optional[Dict[str, Any]] = None
    robustness_analysis: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_architecture': self.best_architecture.to_dict() if self.best_architecture else None,
            'best_score': self.best_score,
            'search_history': self.search_history,
            'execution_time': self.execution_time,
            'n_evaluations': self.n_evaluations,
            'convergence_iteration': self.convergence_iteration,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'config': self.config,
            'uncertainty_estimates': self.uncertainty_estimates,
            'regime_analysis': self.regime_analysis,
            'multi_objective_results': self.multi_objective_results,
            'robustness_analysis': self.robustness_analysis
        }
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'TASResult':
        """Create result from dictionary."""
        # Reconstruct best architecture if available
        best_architecture = None
        if result_dict.get('best_architecture'):
            best_architecture = TreeArchitectureCandidate.from_dict(
                result_dict['best_architecture']
            )
        
        return cls(
            best_architecture=best_architecture,
            best_score=result_dict.get('best_score', 0.0),
            search_history=result_dict.get('search_history', []),
            execution_time=result_dict.get('execution_time', 0.0),
            n_evaluations=result_dict.get('n_evaluations', 0),
            convergence_iteration=result_dict.get('convergence_iteration'),
            success=result_dict.get('success', True),
            error_message=result_dict.get('error_message'),
            timestamp=result_dict.get('timestamp', datetime.now().isoformat()),
            config=result_dict.get('config'),
            uncertainty_estimates=result_dict.get('uncertainty_estimates'),
            regime_analysis=result_dict.get('regime_analysis'),
            multi_objective_results=result_dict.get('multi_objective_results'),
            robustness_analysis=result_dict.get('robustness_analysis')
        )


@dataclass
class TASSearchResult(TASResult):
    """Result class for TAS search operations."""
    
    # Search-specific results
    search_strategy: str = "unknown"
    optimization_mode: str = "single_objective"
    convergence_history: List[float] = field(default_factory=list)
    
    # Search statistics
    total_iterations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    
    # Performance breakdown
    evaluation_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    # Search quality metrics
    search_efficiency: float = 0.0
    exploration_vs_exploitation: float = 0.0
    diversity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'search_strategy': self.search_strategy,
            'optimization_mode': self.optimization_mode,
            'convergence_history': self.convergence_history,
            'total_iterations': self.total_iterations,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'evaluation_times': self.evaluation_times,
            'memory_usage': self.memory_usage,
            'search_efficiency': self.search_efficiency,
            'exploration_vs_exploitation': self.exploration_vs_exploitation,
            'diversity_score': self.diversity_score
        })
        return base_dict


@dataclass
class TASOptimizationResult(TASResult):
    """Result class for TAS optimization operations."""
    
    # Optimization-specific results
    optimization_method: str = "unknown"
    optimization_objectives: List[str] = field(default_factory=list)
    pareto_front: List[TreeArchitectureCandidate] = field(default_factory=list)
    
    # Multi-objective results
    pareto_front_scores: List[Dict[str, float]] = field(default_factory=list)
    hypervolume: float = 0.0
    spread: float = 0.0
    
    # Optimization statistics
    optimization_iterations: int = 0
    objective_improvements: List[Dict[str, float]] = field(default_factory=list)
    
    # Regime-specific results
    regime_specific_architectures: Dict[str, TreeArchitectureCandidate] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert optimization result to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'optimization_method': self.optimization_method,
            'optimization_objectives': self.optimization_objectives,
            'pareto_front': [arch.to_dict() for arch in self.pareto_front],
            'pareto_front_scores': self.pareto_front_scores,
            'hypervolume': self.hypervolume,
            'spread': self.spread,
            'optimization_iterations': self.optimization_iterations,
            'objective_improvements': self.objective_improvements,
            'regime_specific_architectures': {
                regime: arch.to_dict() for regime, arch in self.regime_specific_architectures.items()
            },
            'regime_performance': self.regime_performance
        })
        return base_dict


@dataclass
class TASEvaluationResult:
    """Result class for TAS evaluation operations."""
    
    # Evaluation results
    architecture: TreeArchitectureCandidate
    scores: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    
    # Evaluation metadata
    evaluation_time: float = 0.0
    evaluation_method: str = "holdout"
    data_size: Tuple[int, int] = (0, 0)
    
    # Performance breakdown
    train_score: float = 0.0
    validation_score: float = 0.0
    test_score: Optional[float] = None
    
    # Uncertainty estimates
    prediction_uncertainty: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary."""
        return {
            'architecture': self.architecture.to_dict(),
            'scores': self.scores,
            'predictions': self.predictions.tolist() if self.predictions is not None else None,
            'probabilities': self.probabilities.tolist() if self.probabilities is not None else None,
            'evaluation_time': self.evaluation_time,
            'evaluation_method': self.evaluation_method,
            'data_size': self.data_size,
            'train_score': self.train_score,
            'validation_score': self.validation_score,
            'test_score': self.test_score,
            'prediction_uncertainty': self.prediction_uncertainty,
            'confidence_interval': self.confidence_interval
        }


@dataclass
class TASRegimeResult:
    """Result class for TAS regime analysis."""
    
    # Regime detection results
    detected_regimes: List[str] = field(default_factory=list)
    regime_labels: Optional[np.ndarray] = None
    regime_centers: Optional[np.ndarray] = None
    
    # Regime characteristics
    regime_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_transitions: Optional[np.ndarray] = None
    
    # Regime-specific architectures
    regime_architectures: Dict[str, TreeArchitectureCandidate] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Regime quality metrics
    regime_stability: float = 0.0
    regime_separation: float = 0.0
    regime_consistency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert regime result to dictionary."""
        return {
            'detected_regimes': self.detected_regimes,
            'regime_labels': self.regime_labels.tolist() if self.regime_labels is not None else None,
            'regime_centers': self.regime_centers.tolist() if self.regime_centers is not None else None,
            'regime_statistics': self.regime_statistics,
            'regime_transitions': self.regime_transitions.tolist() if self.regime_transitions is not None else None,
            'regime_architectures': {
                regime: arch.to_dict() for regime, arch in self.regime_architectures.items()
            },
            'regime_performance': self.regime_performance,
            'regime_stability': self.regime_stability,
            'regime_separation': self.regime_separation,
            'regime_consistency': self.regime_consistency
        }


@dataclass
class TASUncertaintyResult:
    """Result class for TAS uncertainty estimation."""
    
    # Uncertainty estimates
    prediction_uncertainty: float = 0.0
    model_uncertainty: float = 0.0
    data_uncertainty: float = 0.0
    
    # Confidence measures
    confidence_score: float = 0.0
    reliability_score: float = 0.0
    
    # Uncertainty breakdown
    aleatoric_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    
    # Uncertainty sources
    ensemble_variance: float = 0.0
    prediction_entropy: float = 0.0
    calibration_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert uncertainty result to dictionary."""
        return {
            'prediction_uncertainty': self.prediction_uncertainty,
            'model_uncertainty': self.model_uncertainty,
            'data_uncertainty': self.data_uncertainty,
            'confidence_score': self.confidence_score,
            'reliability_score': self.reliability_score,
            'aleatoric_uncertainty': self.aleatoric_uncertainty,
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'ensemble_variance': self.ensemble_variance,
            'prediction_entropy': self.prediction_entropy,
            'calibration_score': self.calibration_score
        }


# Result utility functions
def save_result(result: TASResult, filepath: Union[str, Path]):
    """Save TAS result to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def load_result(filepath: Union[str, Path]) -> TASResult:
    """Load TAS result from file."""
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        result_dict = json.load(f)
    
    return TASResult.from_dict(result_dict)


def compare_results(results: List[TASResult]) -> Dict[str, Any]:
    """Compare multiple TAS results."""
    if not results:
        return {}
    
    comparison = {
        'n_results': len(results),
        'best_scores': [r.best_score for r in results],
        'execution_times': [r.execution_time for r in results],
        'n_evaluations': [r.n_evaluations for r in results],
        'success_rates': [r.success for r in results]
    }
    
    # Statistical summary
    comparison['score_statistics'] = {
        'mean': np.mean(comparison['best_scores']),
        'std': np.std(comparison['best_scores']),
        'min': np.min(comparison['best_scores']),
        'max': np.max(comparison['best_scores'])
    }
    
    comparison['time_statistics'] = {
        'mean': np.mean(comparison['execution_times']),
        'std': np.std(comparison['execution_times']),
        'min': np.min(comparison['execution_times']),
        'max': np.max(comparison['execution_times'])
    }
    
    # Best result
    best_idx = np.argmax(comparison['best_scores'])
    comparison['best_result'] = {
        'index': int(best_idx),
        'score': comparison['best_scores'][best_idx],
        'execution_time': comparison['execution_times'][best_idx]
    }
    
    return comparison