"""
Optimization results and data structures.

This module provides data structures for storing and managing optimization results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class HPOResult:
    """Comprehensive HPO result."""
    
    # Basic results
    best_params: Dict[str, Any]
    best_score: float
    best_trial: Optional[Any] = None
    
    # Optimization metadata
    n_trials: int = 0
    optimization_time: float = 0.0
    strategy: str = "unknown"
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed results
    trial_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    # Validation results
    cv_scores: Optional[List[float]] = None
    validation_score: float = 0.0
    overfitting_detected: bool = False
    
    # Metadata
    model_name: str = "unknown"
    optimization_timestamp: str = None
    
    def __post_init__(self):
        if self.optimization_timestamp is None:
            self.optimization_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'optimization_time': self.optimization_time,
            'strategy': self.strategy,
            'convergence_info': self.convergence_info,
            'trial_results': self.trial_results,
            'optimization_history': self.optimization_history,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'cv_scores': self.cv_scores,
            'validation_score': self.validation_score,
            'overfitting_detected': self.overfitting_detected,
            'model_name': self.model_name,
            'optimization_timestamp': self.optimization_timestamp
        }
    
    def save(self, filepath: str) -> None:
        """Save result to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'HPOResult':
        """Load result from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            'model_name': self.model_name,
            'strategy': self.strategy,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'optimization_time': self.optimization_time,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'overfitting_detected': self.overfitting_detected,
            'timestamp': self.optimization_timestamp
        }
    
    def get_best_params_summary(self) -> Dict[str, Any]:
        """Get summary of best parameters."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'parameter_count': len(self.best_params)
        }
    
    def get_trial_statistics(self) -> Dict[str, Any]:
        """Get statistics about all trials."""
        if not self.trial_results:
            return {}
        
        scores = [t.get('value', 0) for t in self.trial_results if t.get('value') is not None]
        
        if not scores:
            return {}
        
        import numpy as np
        
        return {
            'total_trials': len(self.trial_results),
            'valid_trials': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores)),
            'q25_score': float(np.percentile(scores, 25)),
            'q75_score': float(np.percentile(scores, 75))
        }