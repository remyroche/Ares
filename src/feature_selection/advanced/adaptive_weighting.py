"""
Adaptive Weighting System for Ensemble Feature Selection

This module implements performance-based adaptive weighting where weights
are determined by each method's cross-validation performance.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import deque
from dataclasses import dataclass

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

class AdaptiveWeightingSystem:
    """System for adaptive weighting based on method performance."""

    def __init__(self, config, hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize adaptive weighting system."""
        self.config = config
        self.hardware_manager = hardware_manager
        self.logger = logger.getChild('AdaptiveWeightingSystem')

        # Weight tracking
        self.method_weights = {}
        self.performance_history = {}
        self.weight_history = {}
        self.selection_count = 0

        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'weight_updates': 0,
            'avg_performance': {},
            'weight_stability': {}
        }

        tprint_success("🔧 AdaptiveWeightingSystem initialized")

    def initialize_weights(self, method_names: List[str]) -> Dict[str, float]:
        """Initialize equal weights for all methods."""
        initial_weight = 1.0 / len(method_names)
        self.method_weights = {method: initial_weight for method in method_names}

        # Initialize performance history
        for method in method_names:
            self.performance_history[method] = deque(maxlen=100)
            self.weight_history[method] = deque(maxlen=100)

        tprint_debug(f"🔧 Initialized weights: {self.method_weights}")
        return self.method_weights.copy()

    def update_weights(self, method_performances: Dict[str, float]) -> Dict[str, float]:
        """Update weights based on method performances."""
        if not self.config.enable_adaptive_weighting:
            return self.method_weights.copy()

        tprint_debug("🔧 Updating adaptive weights")

        try:
            # Store performance history
            for method, performance in method_performances.items():
                if method in self.performance_history:
                    self.performance_history[method].append(performance)

            # Calculate new weights based on performance
            new_weights = self._calculate_performance_based_weights(method_performances)

            # Apply smoothing
            if self.selection_count > 0:
                new_weights = self._apply_weight_smoothing(new_weights)

            # Apply constraints
            new_weights = self._apply_weight_constraints(new_weights)

            # Update weights
            self.method_weights.update(new_weights)

            # Store weight history
            for method, weight in self.method_weights.items():
                if method in self.weight_history:
                    self.weight_history[method].append(weight)

            # Update statistics
            self.selection_count += 1
            self.performance_stats['total_selections'] = self.selection_count

            if self.selection_count % self.config.weight_update_frequency == 0:
                self.performance_stats['weight_updates'] += 1
                self._update_performance_stats()

            tprint_debug(f"🔧 Updated weights: {self.method_weights}")
            return self.method_weights.copy()

        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
            return self.method_weights.copy()

    def _calculate_performance_based_weights(self, method_performances: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights based on method performances."""
        # Normalize performances to [0, 1] range
        performances = np.array(list(method_performances.values()))
        method_names = list(method_performances.keys())

        # Handle different performance metrics
        if self.config.performance_metric in ['mse', 'mae']:
            # Lower is better - invert the scale
            normalized_performances = 1.0 - (performances - performances.min()) / (performances.max() - performances.min() + 1e-8)
        else:
            # Higher is better - use as is
            normalized_performances = (performances - performances.min()) / (performances.max() - performances.min() + 1e-8)

        # Calculate weights proportional to performance
        weights = normalized_performances / (normalized_performances.sum() + 1e-8)

        return dict(zip(method_names, weights))

    def _apply_weight_smoothing(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply smoothing to prevent rapid weight changes."""
        smoothed_weights = {}

        for method, new_weight in new_weights.items():
            if method in self.method_weights:
                current_weight = self.method_weights[method]
                smoothed_weight = (1 - self.config.weight_smoothing) * current_weight + \
                                self.config.weight_smoothing * new_weight
                smoothed_weights[method] = smoothed_weight
            else:
                smoothed_weights[method] = new_weight

        return smoothed_weights

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        constrained_weights = {}

        for method, weight in weights.items():
            # Apply constraints
            constrained_weight = np.clip(weight, self.config.min_weight, self.config.max_weight)
            constrained_weights[method] = constrained_weight

        # Renormalize to ensure weights sum to 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {method: weight / total_weight for method, weight in constrained_weights.items()}

        return constrained_weights

    def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        for method in self.method_weights.keys():
            if method in self.performance_history and len(self.performance_history[method]) > 0:
                performances = list(self.performance_history[method])
                self.performance_stats['avg_performance'][method] = np.mean(performances)

            if method in self.weight_history and len(self.weight_history[method]) > 0:
                weights = list(self.weight_history[method])
                weight_std = np.std(weights)
                self.performance_stats['weight_stability'][method] = 1.0 / (1.0 + weight_std)

    def get_current_weights(self) -> Dict[str, float]:
        """Get current method weights."""
        return self.method_weights.copy()

    def get_weight_history(self, method: str) -> List[float]:
        """Get weight history for a specific method."""
        return list(self.weight_history.get(method, []))

    def get_performance_history(self, method: str) -> List[float]:
        """Get performance history for a specific method."""
        return list(self.performance_history.get(method, []))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        # Add current weights
        stats['current_weights'] = self.method_weights.copy()

        # Add weight stability metrics
        if self.weight_history:
            for method in self.method_weights.keys():
                if method in self.weight_history and len(self.weight_history[method]) > 1:
                    weights = list(self.weight_history[method])
                    weight_std = np.std(weights)
                    weight_mean = np.mean(weights)
                    stats['weight_stability'][method] = {
                        'mean': float(weight_mean),
                        'std': float(weight_std),
                        'coefficient_of_variation': float(weight_std / (weight_mean + 1e-8))
                    }

        return stats

    def reset_weights(self, method_names: List[str]) -> None:
        """Reset weights to equal values."""
        self.initialize_weights(method_names)
        self.selection_count = 0
        self.performance_history.clear()
        self.weight_history.clear()
        tprint_info("🔧 Reset adaptive weights to equal values")

    def get_weight_insights(self) -> Dict[str, Any]:
        """Get insights about weight behavior."""
        insights = {
            'total_selections': self.selection_count,
            'weight_updates': self.performance_stats['weight_updates'],
            'current_weights': self.method_weights.copy(),
            'weight_stability': {},
            'performance_trends': {}
        }

        # Analyze weight stability
        for method in self.method_weights.keys():
            if method in self.weight_history and len(self.weight_history[method]) > 1:
                weights = list(self.weight_history[method])
                weight_std = np.std(weights)
                weight_mean = np.mean(weights)

                insights['weight_stability'][method] = {
                    'stability_score': float(1.0 / (1.0 + weight_std)),
                    'weight_range': [float(min(weights)), float(max(weights))],
                    'trend': 'increasing' if weights[-1] > weights[0] else 'decreasing' if weights[-1] < weights[0] else 'stable'
                }

        # Analyze performance trends
        for method in self.method_weights.keys():
            if method in self.performance_history and len(self.performance_history[method]) > 1:
                performances = list(self.performance_history[method])
                recent_avg = np.mean(performances[-5:]) if len(performances) >= 5 else np.mean(performances)
                overall_avg = np.mean(performances)

                insights['performance_trends'][method] = {
                    'recent_avg': float(recent_avg),
                    'overall_avg': float(overall_avg),
                    'trend': 'improving' if recent_avg > overall_avg else 'declining' if recent_avg < overall_avg else 'stable'
                }

        return insights
