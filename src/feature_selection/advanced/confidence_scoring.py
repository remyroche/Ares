"""
Confidence Scoring System for Feature Selection

This module implements confidence scoring based on method agreement,
stability, and performance metrics.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from collections import defaultdict, Counter

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

class ConfidenceScoringSystem:
    """System for calculating confidence scores based on method agreement."""

    def __init__(self, config, hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize confidence scoring system."""
        self.config = config
        self.hardware_manager = hardware_manager
        self.logger = logger.getChild('ConfidenceScoringSystem')

        # Confidence tracking
        self.feature_confidence_history = defaultdict(list)
        self.method_agreement_history = defaultdict(list)
        self.stability_history = defaultdict(list)

        # Performance tracking
        self.performance_stats = {
            'total_scorings': 0,
            'high_confidence_features': 0,
            'consensus_features': 0,
            'avg_confidence': 0.0
        }

        tprint_success("🔧 ConfidenceScoringSystem initialized")

    def calculate_confidence_scores(self,
                                  method_results: Dict[str, Dict[str, Any]],
                                  feature_names: List[str],
                                  stability_scores: Optional[Dict[str, float]] = None,
                                  performance_scores: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate confidence scores for selected features."""
        if not self.config.enable_confidence_scoring:
            return self._create_default_confidence_scores(feature_names)

        tprint_debug("🔧 Calculating confidence scores")

        try:
            # Extract selected features from each method
            method_selections = self._extract_method_selections(method_results, feature_names)

            # Calculate agreement scores
            agreement_scores = self._calculate_agreement_scores(method_selections, feature_names)

            # Calculate consensus scores
            consensus_scores = self._calculate_consensus_scores(method_selections, feature_names)

            # Calculate stability scores
            if stability_scores is None:
                stability_scores = self._calculate_stability_scores(method_selections, feature_names)

            # Calculate performance scores
            if performance_scores is None:
                performance_scores = self._calculate_performance_scores(method_selections, feature_names)

            # Combine scores into confidence
            confidence_scores = self._combine_confidence_scores(
                agreement_scores, consensus_scores, stability_scores, performance_scores
            )

            # Apply confidence constraints
            confidence_scores = self._apply_confidence_constraints(confidence_scores)

            # Update statistics
            self._update_confidence_statistics(confidence_scores)

            # Store history
            self._store_confidence_history(confidence_scores, method_selections)

            tprint_debug(f"🔧 Calculated confidence scores for {len(confidence_scores)} features")
            return confidence_scores

        except Exception as e:
            self.logger.error(f"Confidence scoring failed: {e}")
            return self._create_default_confidence_scores(feature_names)

    def _extract_method_selections(self, method_results: Dict[str, Dict[str, Any]],
                                 feature_names: List[str]) -> Dict[str, Set[str]]:
        """Extract selected features from each method's results."""
        method_selections = {}

        for method_name, result in method_results.items():
            if result.get('success', False):
                selected_features = result.get('selected_features', [])
                method_selections[method_name] = set(selected_features)
            else:
                method_selections[method_name] = set()

        return method_selections

    def _calculate_agreement_scores(self, method_selections: Dict[str, Set[str]],
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate agreement scores for each feature."""
        agreement_scores = {}

        for feature in feature_names:
            # Count how many methods selected this feature
            selection_count = sum(1 for selections in method_selections.values() if feature in selections)
            total_methods = len(method_selections)

            # Calculate agreement score
            agreement_score = selection_count / total_methods if total_methods > 0 else 0.0
            agreement_scores[feature] = agreement_score

        return agreement_scores

    def _calculate_consensus_scores(self, method_selections: Dict[str, Set[str]],
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate consensus scores for each feature."""
        consensus_scores = {}

        for feature in feature_names:
            # Count how many methods selected this feature
            selection_count = sum(1 for selections in method_selections.values() if feature in selections)

            # Apply consensus bonus
            if selection_count >= self.config.consensus_min_methods:
                consensus_bonus = self.config.consensus_bonus
            else:
                consensus_bonus = 0.0

            consensus_scores[feature] = min(1.0, selection_count / len(method_selections) + consensus_bonus)

        return consensus_scores

    def _calculate_stability_scores(self, method_selections: Dict[str, Set[str]],
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate stability scores for each feature."""
        stability_scores = {}

        for feature in feature_names:
            # Check if feature has been consistently selected in history
            if feature in self.stability_history and len(self.stability_history[feature]) > 0:
                historical_selections = self.stability_history[feature]
                stability_score = np.mean(historical_selections)
            else:
                # For new features, use current selection status
                current_selection = sum(1 for selections in method_selections.values() if feature in selections)
                stability_score = current_selection / len(method_selections) if len(method_selections) > 0 else 0.0

            stability_scores[feature] = stability_score

        return stability_scores

    def _calculate_performance_scores(self, method_selections: Dict[str, Set[str]],
                                    feature_names: List[str]) -> Dict[str, float]:
        """Calculate performance scores for each feature."""
        performance_scores = {}

        for feature in feature_names:
            # Calculate performance score based on method performance
            total_performance = 0.0
            method_count = 0

            for method_name, selections in method_selections.items():
                if feature in selections:
                    # Get method performance (if available)
                    method_performance = self._get_method_performance(method_name)
                    total_performance += method_performance
                    method_count += 1

            if method_count > 0:
                performance_score = total_performance / method_count
            else:
                performance_score = 0.0

            performance_scores[feature] = performance_score

        return performance_scores

    def _get_method_performance(self, method_name: str) -> float:
        """Get performance score for a specific method."""
        # This would typically come from the method's validation results
        # For now, return a default value
        return 0.5

    def _combine_confidence_scores(self,
                                 agreement_scores: Dict[str, float],
                                 consensus_scores: Dict[str, float],
                                 stability_scores: Dict[str, float],
                                 performance_scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Combine all scores into final confidence scores."""
        confidence_scores = {}

        for feature in agreement_scores.keys():
            # Weighted combination of scores
            combined_score = (
                self.config.agreement_weight * agreement_scores[feature] +
                self.config.stability_weight * stability_scores[feature] +
                self.config.performance_weight * performance_scores[feature]
            )

            # Apply consensus bonus
            consensus_bonus = consensus_scores[feature] - agreement_scores[feature]
            combined_score += consensus_bonus * self.config.consensus_bonus

            confidence_scores[feature] = {
                'confidence_score': float(combined_score),
                'agreement_score': float(agreement_scores[feature]),
                'consensus_score': float(consensus_scores[feature]),
                'stability_score': float(stability_scores[feature]),
                'performance_score': float(performance_scores[feature]),
                'is_consensus': consensus_scores[feature] >= self.config.consensus_min_methods / len(agreement_scores),
                'is_high_confidence': combined_score >= self.config.agreement_threshold
            }

        return confidence_scores

    def _apply_confidence_constraints(self, confidence_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply minimum and maximum confidence constraints."""
        constrained_scores = {}

        for feature, scores in confidence_scores.items():
            constrained_score = np.clip(
                scores['confidence_score'],
                self.config.min_confidence,
                self.config.max_confidence
            )

            scores['confidence_score'] = float(constrained_score)
            constrained_scores[feature] = scores

        return constrained_scores

    def _create_default_confidence_scores(self, feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create default confidence scores when scoring is disabled."""
        default_scores = {}

        for feature in feature_names:
            default_scores[feature] = {
                'confidence_score': 0.5,
                'agreement_score': 0.5,
                'consensus_score': 0.5,
                'stability_score': 0.5,
                'performance_score': 0.5,
                'is_consensus': False,
                'is_high_confidence': False
            }

        return default_scores

    def _update_confidence_statistics(self, confidence_scores: Dict[str, Dict[str, Any]]) -> None:
        """Update confidence statistics."""
        self.performance_stats['total_scorings'] += 1

        if confidence_scores:
            confidences = [scores['confidence_score'] for scores in confidence_scores.values()]
            self.performance_stats['avg_confidence'] = np.mean(confidences)
            self.performance_stats['high_confidence_features'] = sum(
                1 for scores in confidence_scores.values()
                if scores['is_high_confidence']
            )
            self.performance_stats['consensus_features'] = sum(
                1 for scores in confidence_scores.values()
                if scores['is_consensus']
            )

    def _store_confidence_history(self, confidence_scores: Dict[str, Dict[str, Any]],
                                method_selections: Dict[str, Set[str]]) -> None:
        """Store confidence history for future stability calculations."""
        for feature, scores in confidence_scores.items():
            # Store confidence score
            self.feature_confidence_history[feature].append(scores['confidence_score'])

            # Store agreement score
            self.method_agreement_history[feature].append(scores['agreement_score'])

            # Store stability score
            self.stability_history[feature].append(scores['stability_score'])

    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get confidence scoring statistics."""
        stats = self.performance_stats.copy()

        # Add feature-level statistics
        if self.feature_confidence_history:
            all_confidences = []
            for confidences in self.feature_confidence_history.values():
                all_confidences.extend(confidences)

            if all_confidences:
                stats['confidence_distribution'] = {
                    'mean': float(np.mean(all_confidences)),
                    'std': float(np.std(all_confidences)),
                    'min': float(np.min(all_confidences)),
                    'max': float(np.max(all_confidences)),
                    'median': float(np.median(all_confidences))
                }

        return stats

    def get_feature_confidence_history(self, feature: str) -> List[float]:
        """Get confidence history for a specific feature."""
        return list(self.feature_confidence_history.get(feature, []))

    def get_high_confidence_features(self, confidence_scores: Dict[str, Dict[str, Any]],
                                   threshold: Optional[float] = None) -> List[str]:
        """Get features with high confidence scores."""
        if threshold is None:
            threshold = self.config.agreement_threshold

        return [
            feature for feature, scores in confidence_scores.items()
            if scores['confidence_score'] >= threshold
        ]

    def get_consensus_features(self, confidence_scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get features selected by consensus."""
        return [
            feature for feature, scores in confidence_scores.items()
            if scores['is_consensus']
        ]

    def get_confidence_insights(self, confidence_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get insights about confidence scoring."""
        insights = {
            'total_features': len(confidence_scores),
            'high_confidence_count': sum(1 for scores in confidence_scores.values() if scores['is_high_confidence']),
            'consensus_count': sum(1 for scores in confidence_scores.values() if scores['is_consensus']),
            'avg_confidence': np.mean([scores['confidence_score'] for scores in confidence_scores.values()]),
            'confidence_distribution': {},
            'method_agreement_analysis': {}
        }

        if confidence_scores:
            confidences = [scores['confidence_score'] for scores in confidence_scores.values()]
            insights['confidence_distribution'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }

            # Analyze method agreement
            agreement_scores = [scores['agreement_score'] for scores in confidence_scores.values()]
            insights['method_agreement_analysis'] = {
                'avg_agreement': float(np.mean(agreement_scores)),
                'high_agreement_count': sum(1 for score in agreement_scores if score >= 0.8),
                'low_agreement_count': sum(1 for score in agreement_scores if score < 0.3)
            }

        return insights
