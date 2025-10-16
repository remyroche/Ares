#!/usr/bin/env python3
"""
Ensemble Monitor for Enhanced ML Monitoring

Monitors ensemble performance, model weights, and individual model contributions
with detailed tracking and analysis.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import datetime
import logging
import time
import typing

from src.utils.logger import system_logger
from src.core.decorators import handles_errors

@dataclass
class ModelWeight:
    """Model weight information for ensemble."""
    model_id: str
    model_type: str
    weight: float
    confidence: float
    performance_score: float
    last_updated: datetime
    weight_history: List[Tuple[datetime, float]] = None

@dataclass
class EnsembleState:
    """Current state of an ensemble."""
    ensemble_id: str
    timestamp: datetime
    model_weights: Dict[str, ModelWeight]
    total_models: int
    active_models: int
    weight_stability_score: float
    performance_trend: str  # "improving", "stable", "declining"
    last_rebalance: Optional[datetime] = None

@dataclass
class ModelContribution:
    """Individual model contribution to ensemble performance."""
    model_id: str
    model_type: str
    contribution_score: float
    accuracy_contribution: float
    profit_contribution: float
    risk_contribution: float
    prediction_agreement: float
    feature_diversity: float
    timestamp: datetime

@dataclass
class EnsemblePerformanceSnapshot:
    """Snapshot of ensemble performance at a point in time."""
    ensemble_id: str
    timestamp: datetime

    # Overall performance
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float

    # Ensemble-specific metrics
    model_diversity_score: float
    consensus_quality: float
    disagreement_level: float
    weight_stability: float

    # Individual model contributions
    model_contributions: List[ModelContribution]

    # Weight distribution
    weight_entropy: float
    dominant_model_share: float

class EnsembleMonitor:
    """
    Monitors ensemble performance, model weights, and individual contributions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the ensemble monitor."""
        self.config = config
        self.logger = system_logger.getChild("EnsembleMonitor")

        # Configuration
        self.ensemble_config = config.get("ensemble_monitoring", {})
        self.weight_update_frequency = self.ensemble_config.get("weight_update_frequency_hours", 24)
        self.performance_window_days = self.ensemble_config.get("performance_window_days", 30)
        self.min_weight_threshold = self.ensemble_config.get("min_weight_threshold", 0.01)
        self.max_weight_threshold = self.ensemble_config.get("max_weight_threshold", 0.8)
        self.rebalance_threshold = self.ensemble_config.get("rebalance_threshold", 0.1)

        # Storage
        self.ensemble_states: Dict[str, EnsembleState] = {}
        self.performance_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen = 1000))
        self.model_contributions: Dict[str, deque] = defaultdict(lambda: deque(maxlen = 1000))
        self.weight_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen = 1000))

        # Performance tracking
        self.last_weight_update: Dict[str, datetime] = {}
        self.ensemble_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

        self.logger.info("Ensemble Monitor initialized")

    @handles_errors(default_return = None, context="ensemble_monitor.update_ensemble_weights")
    async def update_ensemble_weights(self, ensemble_id: str,
                                    model_performances: Dict[str, Dict[str, float]],
                                    current_weights: Dict[str, float]) -> Dict[str, float]:
        """Update ensemble model weights based on recent performance."""
        try:
            current_time = datetime.now()

            # Check if we need to update weights
            last_update = self.last_weight_update.get(ensemble_id)
            if (last_update and
                (current_time - last_update).total_seconds() < self.weight_update_frequency * 3600):
                return current_weights

            # Calculate new weights based on performance
            new_weights = self._calculate_performance_based_weights(
                model_performances, current_weights
            )

            # Apply weight constraints
            new_weights = self._apply_weight_constraints(new_weights)

            # Update ensemble state
            await self._update_ensemble_state(ensemble_id, new_weights, model_performances)

            # Record weight history
            self._record_weight_history(ensemble_id, new_weights, current_time)

            # Update last update time
            self.last_weight_update[ensemble_id] = current_time

            self.logger.info(
                f"Updated weights for ensemble {ensemble_id}: "
                f"{dict(sorted(new_weights.items(), key = lambda x: x[1], reverse = True))}"
            )

            return new_weights

        except Exception as e:
            self.logger.error(f"Error updating ensemble weights for {ensemble_id}: {e}")
            return current_weights

    def _calculate_performance_based_weights(self, model_performances: Dict[str, Dict[str, float]],
                                        current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate new weights based on model performance metrics."""
        new_weights = {}

        # Performance metrics to consider
        performance_metrics = ['accuracy', 'win_rate', 'profit_factor', 'sharpe_ratio']

        for model_id, performance in model_performances.items():
            # Calculate composite performance score
            performance_scores = []

            for metric in performance_metrics:
                if metric in performance:
                    # Normalize metric (assuming higher is better)
                    score = performance[metric]
                    if metric == 'max_drawdown':
                        # For drawdown, lower is better
                        score = 1.0 - abs(score)

                    performance_scores.append(max(0.0, min(1.0, score)))

            if performance_scores:
                composite_score = np.mean(performance_scores)
            else:
                # Fallback to current weight if no performance data
                composite_score = current_weights.get(model_id, 0.1)

            new_weights[model_id] = composite_score

        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        else:
            # Equal weights if no performance data
            new_weights = {k: 1.0 / len(new_weights) for k in new_weights.keys()}

        return new_weights

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        constrained_weights = {}

        # Apply minimum threshold
        for model_id, weight in weights.items():
            if weight < self.min_weight_threshold:
                constrained_weights[model_id] = 0.0
            else:
                constrained_weights[model_id] = weight

        # Apply maximum threshold
        for model_id, weight in constrained_weights.items():
            if weight > self.max_weight_threshold:
                constrained_weights[model_id] = self.max_weight_threshold

        # Renormalize
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {k: v / total_weight for k, v in constrained_weights.items()}

        return constrained_weights

    async def _update_ensemble_state(self, ensemble_id: str,
                                new_weights: Dict[str, float],
                                model_performances: Dict[str, Dict[str, float]]):
        """Update the ensemble state with new weights and performance data."""
        try:
            current_time = datetime.now()

            # Create model weight objects
            model_weights = {}
            for model_id, weight in new_weights.items():
                performance = model_performances.get(model_id, {})
                performance_score = np.mean([
                    performance.get('accuracy', 0.5),
                    performance.get('win_rate', 0.5),
                    performance.get('profit_factor', 1.0) / 2.0,  # Normalize
                    performance.get('sharpe_ratio', 0.0) / 2.0 + 0.5  # Normalize
                ])

                # Get weight history
                weight_history = []
                if ensemble_id in self.weight_histories:
                    for timestamp, weights in self.weight_histories[ensemble_id]:
                        if model_id in weights:
                            weight_history.append((timestamp, weights[model_id]))

                model_weights[model_id] = ModelWeight(
                    model_id = model_id,
                    model_type = performance.get('model_type', 'unknown'),
                    weight = weight,
                    confidence = performance.get('confidence', 0.5),
                    performance_score = performance_score,
                    last_updated = current_time,
                    weight_history = weight_history[-10:]  # Keep last 10 updates
                )

            # Calculate weight stability
            weight_stability = self._calculate_weight_stability(ensemble_id, new_weights)

            # Determine performance trend
            performance_trend = self._determine_performance_trend(ensemble_id)

            # Create ensemble state
            ensemble_state = EnsembleState(
                ensemble_id = ensemble_id,
                timestamp = current_time,
                model_weights = model_weights,
                total_models = len(new_weights),
                active_models = sum(1 for w in new_weights.values() if w > 0),
                weight_stability_score = weight_stability,
                performance_trend = performance_trend,
                last_rebalance = current_time if weight_stability < self.rebalance_threshold else None
            )

            self.ensemble_states[ensemble_id] = ensemble_state

        except Exception as e:
            self.logger.error(f"Error updating ensemble state for {ensemble_id}: {e}")

    def _calculate_weight_stability(self, ensemble_id: str,
                                current_weights: Dict[str, float]) -> float:
        """Calculate weight stability score for the ensemble."""
        try:
            if ensemble_id not in self.weight_histories or len(self.weight_histories[ensemble_id]) < 2:
                return 1.0  # Perfect stability if no history

            # Get previous weights
            previous_weights = self.weight_histories[ensemble_id][-1][1]

            # Calculate weight changes
            weight_changes = []
            for model_id in set(current_weights.keys()) | set(previous_weights.keys()):
                current = current_weights.get(model_id, 0.0)
                previous = previous_weights.get(model_id, 0.0)
                change = abs(current - previous)
                weight_changes.append(change)

            # Stability is inverse of average change
            avg_change = np.mean(weight_changes) if weight_changes else 0.0
            stability = max(0.0, 1.0 - avg_change)

            return stability

        except Exception as e:
            self.logger.error(f"Error calculating weight stability: {e}")
            return 0.5

    def _determine_performance_trend(self, ensemble_id: str) -> str:
        """Determine the performance trend for the ensemble."""
        try:
            if ensemble_id not in self.performance_snapshots or len(self.performance_snapshots[ensemble_id]) < 3:
                return "stable"

            # Get recent performance snapshots
            recent_snapshots = list(self.performance_snapshots[ensemble_id])[-5:]

            # Calculate trend in accuracy
            accuracies = [snap.accuracy for snap in recent_snapshots]
            if len(accuracies) >= 3:
                # Simple linear trend
                x = np.arange(len(accuracies))
                slope = np.polyfit(x, accuracies, 1)[0]

                if slope > 0.01:
                    return "improving"
                elif slope < -0.01:
                    return "declining"
                else:
                    return "stable"

            return "stable"

        except Exception as e:
            self.logger.error(f"Error determining performance trend: {e}")
            return "stable"

    def _record_weight_history(self, ensemble_id: str, weights: Dict[str, float],
                            timestamp: datetime):
        """Record weight history for the ensemble."""
        self.weight_histories[ensemble_id].append((timestamp, weights.copy()))

    @handles_errors(default_return = None, context="ensemble_monitor.record_ensemble_performance")
    async def record_ensemble_performance(self, ensemble_id: str,
                                        performance_metrics: Dict[str, float],
                                        model_contributions: List[ModelContribution]):
        """Record ensemble performance snapshot."""
        try:
            current_time = datetime.now()

            # Create performance snapshot
            snapshot = EnsemblePerformanceSnapshot(
                ensemble_id = ensemble_id,
                timestamp = current_time,
                accuracy = performance_metrics.get('accuracy', 0.0),
                precision = performance_metrics.get('precision', 0.0),
                recall = performance_metrics.get('recall', 0.0),
                f1_score = performance_metrics.get('f1_score', 0.0),
                win_rate = performance_metrics.get('win_rate', 0.0),
                profit_factor = performance_metrics.get('profit_factor', 1.0),
                sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0),
                max_drawdown = performance_metrics.get('max_drawdown', 0.0),
                model_diversity_score = performance_metrics.get('model_diversity_score', 0.0),
                consensus_quality = performance_metrics.get('consensus_quality', 0.0),
                disagreement_level = performance_metrics.get('disagreement_level', 0.0),
                weight_stability = performance_metrics.get('weight_stability', 0.0),
                model_contributions = model_contributions,
                weight_entropy = self._calculate_weight_entropy(ensemble_id),
                dominant_model_share = self._calculate_dominant_model_share(ensemble_id)
            )

            # Store snapshot
            self.performance_snapshots[ensemble_id].append(snapshot)

            # Store model contributions
            for contribution in model_contributions:
                self.model_contributions[contribution.model_id].append(contribution)

            # Update ensemble metrics
            self.ensemble_metrics[ensemble_id].update(performance_metrics)

            self.logger.debug(
                f"Recorded performance for ensemble {ensemble_id}: "
                f"accuracy={snapshot.accuracy:.3f}, "
                f"win_rate={snapshot.win_rate:.3f}, "
                f"diversity={snapshot.model_diversity_score:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Error recording ensemble performance for {ensemble_id}: {e}")

    def _calculate_weight_entropy(self, ensemble_id: str) -> float:
        """Calculate weight entropy (diversity measure) for the ensemble."""
        try:
            if ensemble_id not in self.ensemble_states:
                return 0.0

            weights = [mw.weight for mw in self.ensemble_states[ensemble_id].model_weights.values()]
            weights = [w for w in weights if w > 0]  # Remove zero weights

            if not weights:
                return 0.0

            # Calculate entropy
            entropy = -sum(w * np.log(w) for w in weights if w > 0)
            return entropy

        except Exception as e:
            self.logger.error(f"Error calculating weight entropy: {e}")
            return 0.0

    def _calculate_dominant_model_share(self, ensemble_id: str) -> float:
        """Calculate the share of the dominant model in the ensemble."""
        try:
            if ensemble_id not in self.ensemble_states:
                return 0.0

            weights = [mw.weight for mw in self.ensemble_states[ensemble_id].model_weights.values()]
            if not weights:
                return 0.0

            return max(weights)

        except Exception as e:
            self.logger.error(f"Error calculating dominant model share: {e}")
            return 0.0

    @handles_errors(default_return = None, context="ensemble_monitor.get_ensemble_analysis")
    async def get_ensemble_analysis(self, ensemble_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis of ensemble performance and state."""
        try:
            analysis = {
                'ensemble_id': ensemble_id,
                'timestamp': datetime.now().isoformat(),
                'current_state': None,
                'performance_summary': {},
                'weight_analysis': {},
                'model_contributions': {},
                'recommendations': []
            }

            # Current state
            if ensemble_id in self.ensemble_states:
                state = self.ensemble_states[ensemble_id]
                analysis['current_state'] = {
                    'total_models': state.total_models,
                    'active_models': state.active_models,
                    'weight_stability': state.weight_stability_score,
                    'performance_trend': state.performance_trend,
                    'last_rebalance': state.last_rebalance.isoformat() if state.last_rebalance else None
                }

                # Weight analysis
                analysis['weight_analysis'] = {
                    'model_weights': {mw.model_id: {
                        'weight': mw.weight,
                        'confidence': mw.confidence,
                        'performance_score': mw.performance_score,
                        'last_updated': mw.last_updated.isoformat()
                    } for mw in state.model_weights.values()},
                    'weight_entropy': self._calculate_weight_entropy(ensemble_id),
                    'dominant_model_share': self._calculate_dominant_model_share(ensemble_id)
                }

            # Performance summary
            if ensemble_id in self.performance_snapshots and self.performance_snapshots[ensemble_id]:
                recent_snapshots = list(self.performance_snapshots[ensemble_id])[-10:]

                analysis['performance_summary'] = {
                    'recent_accuracy': np.mean([s.accuracy for s in recent_snapshots]),
                    'recent_win_rate': np.mean([s.win_rate for s in recent_snapshots]),
                    'recent_profit_factor': np.mean([s.profit_factor for s in recent_snapshots]),
                    'recent_sharpe_ratio': np.mean([s.sharpe_ratio for s in recent_snapshots]),
                    'model_diversity': np.mean([s.model_diversity_score for s in recent_snapshots]),
                    'consensus_quality': np.mean([s.consensus_quality for s in recent_snapshots]),
                    'weight_stability': np.mean([s.weight_stability for s in recent_snapshots])
                }

            # Model contributions
            if ensemble_id in self.ensemble_states:
                for model_id, mw in self.ensemble_states[ensemble_id].model_weights.items():
                    if model_id in self.model_contributions and self.model_contributions[model_id]:
                        recent_contributions = list(self.model_contributions[model_id])[-5:]
                        analysis['model_contributions'][model_id] = {
                            'avg_contribution': np.mean([c.contribution_score for c in recent_contributions]),
                            'avg_accuracy_contribution': np.mean([c.accuracy_contribution for c in recent_contributions]),
                            'avg_profit_contribution': np.mean([c.profit_contribution for c in recent_contributions]),
                            'avg_agreement': np.mean([c.prediction_agreement for c in recent_contributions])
                        }

            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(ensemble_id, analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error generating ensemble analysis for {ensemble_id}: {e}")
            return {'ensemble_id': ensemble_id, 'error': str(e)}

    def _generate_recommendations(self, ensemble_id: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for ensemble optimization."""
        recommendations = []

        try:
            # Check weight stability
            if 'current_state' in analysis and analysis['current_state']:
                weight_stability = analysis['current_state'].get('weight_stability', 1.0)
                if weight_stability < 0.7:
                    recommendations.append("Consider reducing weight update frequency due to high instability")

            # Check model diversity
            if 'performance_summary' in analysis and analysis['performance_summary']:
                diversity = analysis['performance_summary'].get('model_diversity', 0.0)
                if diversity < 0.5:
                    recommendations.append("Low model diversity detected - consider adding more diverse models")

            # Check dominant model
            if 'weight_analysis' in analysis and analysis['weight_analysis']:
                dominant_share = analysis['weight_analysis'].get('dominant_model_share', 0.0)
                if dominant_share > 0.7:
                    recommendations.append("Single model dominates ensemble - consider rebalancing weights")

            # Check performance trend
            if 'current_state' in analysis and analysis['current_state']:
                trend = analysis['current_state'].get('performance_trend', 'stable')
                if trend == 'declining':
                    recommendations.append("Performance declining - investigate model degradation")

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")

        return recommendations

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get statistics about ensemble monitoring."""
        return {
            'total_ensembles': len(self.ensemble_states),
            'total_performance_snapshots': sum(len(snapshots) for snapshots in self.performance_snapshots.values()),
            'total_model_contributions': sum(len(contributions) for contributions in self.model_contributions.values()),
            'weight_update_frequency_hours': self.weight_update_frequency,
            'performance_window_days': self.performance_window_days,
        }
