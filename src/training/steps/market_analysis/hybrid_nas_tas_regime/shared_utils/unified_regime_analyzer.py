"""
Unified Regime Analyzer

This module provides a unified regime analysis system that combines
the best practices from both TAS and NAS regime detection systems.

Features:
- Unified regime stability calculations
- Common transition probability computation
- Shared uncertainty quantification
- Common meta-learning adaptation
- Support for both tree and neural architectures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class RegimeAnalysisType(Enum):
    """Types of regime analysis."""
    STABILITY = "stability"
    TRANSITIONS = "transitions"
    UNCERTAINTY = "uncertainty"
    META_LEARNING = "meta_learning"
    ADAPTATION = "adaptation"


@dataclass
class RegimeAnalysisConfig:
    """Configuration for unified regime analysis."""
    
    # Analysis types to perform
    analysis_types: List[RegimeAnalysisType] = field(default_factory=lambda: [
        RegimeAnalysisType.STABILITY,
        RegimeAnalysisType.TRANSITIONS,
        RegimeAnalysisType.UNCERTAINTY
    ])
    
    # Stability analysis
    stability_window: int = 20
    stability_threshold: float = 0.7
    
    # Transition analysis
    transition_window: int = 10
    min_transition_samples: int = 5
    
    # Uncertainty analysis
    uncertainty_method: str = "entropy"
    confidence_threshold: float = 0.8
    
    # Meta-learning
    enable_meta_learning: bool = True
    adaptation_rate: float = 0.1
    learning_threshold: float = 0.05
    
    # TAS-specific enhancements
    enable_tree_based_analysis: bool = True
    tree_importance_threshold: float = 0.1
    tree_depth_penalty: float = 0.1
    tree_interpretability_weight: float = 0.3
    
    # NAS-specific enhancements
    enable_neural_based_analysis: bool = True
    neural_confidence_threshold: float = 0.8
    neural_uncertainty_weight: float = 0.3
    neural_architecture_complexity: float = 0.1
    
    # Hybrid analysis
    enable_hybrid_analysis: bool = True
    hybrid_consensus_threshold: float = 0.7
    hybrid_ensemble_weight: float = 0.5


@dataclass
class RegimeAnalysisResult:
    """Result from unified regime analysis."""
    
    # Stability analysis
    regime_stability_scores: np.ndarray
    overall_stability: float
    stability_analysis: Dict[str, Any]
    
    # Transition analysis
    transition_probabilities: np.ndarray
    transition_matrix: np.ndarray
    transition_analysis: Dict[str, Any]
    
    # Uncertainty analysis
    uncertainty_estimates: np.ndarray
    confidence_scores: np.ndarray
    uncertainty_analysis: Dict[str, Any]
    
    # Meta-learning analysis
    adaptation_scores: np.ndarray
    learning_indicators: Dict[str, Any]
    meta_learning_analysis: Dict[str, Any]
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    data_shape: Tuple[int, int] = (0, 0)
    n_regimes: int = 0
    analysis_time: float = 0.0


class UnifiedRegimeAnalyzer:
    """
    Unified Regime Analyzer.
    
    Combines the best practices from both TAS and NAS regime detection systems
    to provide comprehensive regime analysis.
    """
    
    def __init__(self, config: RegimeAnalysisConfig):
        """Initialize unified regime analyzer."""
        tprint_info("🚀 Initializing Unified Regime Analyzer")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint_success("✅ Unified Regime Analyzer initialized")
        tprint_info(f"   Analysis types: {[t.value for t in config.analysis_types]}")
        tprint_info(f"   Meta-learning: {config.enable_meta_learning}")
        self.logger.info("✅ Unified Regime Analyzer initialized")
        self.logger.info(f"   Analysis types: {[t.value for t in config.analysis_types]}")
        self.logger.info(f"   Meta-learning: {config.enable_meta_learning}")
    
    def analyze(self, 
                regime_predictions: np.ndarray,
                regime_probabilities: Optional[np.ndarray] = None,
                market_data: Optional[np.ndarray] = None,
                timestamps: Optional[np.ndarray] = None,
                architecture_type: Optional[str] = None,
                model_metadata: Optional[Dict[str, Any]] = None) -> RegimeAnalysisResult:
        """
        Perform unified regime analysis.
        
        Args:
            regime_predictions: Regime predictions
            regime_probabilities: Optional regime probabilities
            market_data: Optional market data
            timestamps: Optional timestamps
            
        Returns:
            Comprehensive regime analysis result
        """
        start_time = time.time()
        
        try:
            tprint_info("🔍 Starting unified regime analysis...")
            tprint_debug(f"Predictions: {len(regime_predictions)}")
            tprint_debug(f"Regimes: {len(np.unique(regime_predictions))}")
            self.logger.info("🔍 Starting unified regime analysis...")
            self.logger.info(f"   Predictions: {len(regime_predictions)}")
            self.logger.info(f"   Regimes: {len(np.unique(regime_predictions))}")
            
            # Initialize result arrays
            tprint_debug("📊 Initializing result arrays...")
            stability_scores = np.zeros(len(regime_predictions))
            transition_probs = np.zeros((len(regime_predictions), len(regime_predictions)))
            uncertainty_estimates = np.zeros(len(regime_predictions))
            adaptation_scores = np.zeros(len(regime_predictions))
            tprint_success("✅ Result arrays initialized")
            
            # Perform analysis based on configuration
            tprint_info("🔍 Performing analysis based on configuration...")
            stability_analysis = {}
            transition_analysis = {}
            uncertainty_analysis = {}
            meta_learning_analysis = {}
            
            if RegimeAnalysisType.STABILITY in self.config.analysis_types:
                stability_scores, stability_analysis = self._analyze_regime_stability(regime_predictions)
            
            if RegimeAnalysisType.TRANSITIONS in self.config.analysis_types:
                transition_probs, transition_matrix, transition_analysis = self._analyze_regime_transitions(regime_predictions)
            
            if RegimeAnalysisType.UNCERTAINTY in self.config.analysis_types:
                uncertainty_estimates, uncertainty_analysis = self._analyze_uncertainty(regime_predictions, regime_probabilities)
            
            if RegimeAnalysisType.META_LEARNING in self.config.analysis_types and self.config.enable_meta_learning:
                adaptation_scores, meta_learning_analysis = self._analyze_meta_learning(regime_predictions, market_data)
            
            # Architecture-specific enhancements
            if architecture_type == "TAS" and self.config.enable_tree_based_analysis:
                tree_analysis = self._analyze_tree_based_regime_patterns(
                    regime_predictions, regime_probabilities, model_metadata
                )
                # Adjust scores based on tree analysis
                stability_scores = self._adjust_scores_with_tree_analysis(stability_scores, tree_analysis)
                uncertainty_estimates = self._adjust_scores_with_tree_analysis(uncertainty_estimates, tree_analysis)
                
            elif architecture_type == "NAS" and self.config.enable_neural_based_analysis:
                neural_analysis = self._analyze_neural_based_regime_patterns(
                    regime_predictions, regime_probabilities, model_metadata
                )
                # Adjust scores based on neural analysis
                stability_scores = self._adjust_scores_with_neural_analysis(stability_scores, neural_analysis)
                uncertainty_estimates = self._adjust_scores_with_neural_analysis(uncertainty_estimates, neural_analysis)
                
            elif architecture_type == "HYBRID" and self.config.enable_hybrid_analysis:
                hybrid_analysis = self._analyze_hybrid_regime_patterns(
                    regime_predictions, regime_probabilities, model_metadata
                )
                # Adjust scores based on hybrid analysis
                stability_scores = self._adjust_scores_with_hybrid_analysis(stability_scores, hybrid_analysis)
                uncertainty_estimates = self._adjust_scores_with_hybrid_analysis(uncertainty_estimates, hybrid_analysis)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(regime_predictions, regime_probabilities)
            
            # Calculate learning indicators
            learning_indicators = self._calculate_learning_indicators(regime_predictions, market_data)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = RegimeAnalysisResult(
                regime_stability_scores=stability_scores,
                overall_stability=np.mean(stability_scores),
                stability_analysis=stability_analysis,
                transition_probabilities=transition_probs,
                transition_matrix=self._calculate_transition_matrix(regime_predictions),
                transition_analysis=transition_analysis,
                uncertainty_estimates=uncertainty_estimates,
                confidence_scores=confidence_scores,
                uncertainty_analysis=uncertainty_analysis,
                adaptation_scores=adaptation_scores,
                learning_indicators=learning_indicators,
                meta_learning_analysis=meta_learning_analysis,
                data_shape=(len(regime_predictions), len(np.unique(regime_predictions))),
                n_regimes=len(np.unique(regime_predictions)),
                analysis_time=execution_time
            )
            
            self.logger.info(f"✅ Unified regime analysis completed in {execution_time:.2f}s")
            self.logger.info(f"   Overall stability: {result.overall_stability:.3f}")
            self.logger.info(f"   Average uncertainty: {np.mean(uncertainty_estimates):.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Unified regime analysis failed: {e}")
            
            return RegimeAnalysisResult(
                regime_stability_scores=np.zeros(len(regime_predictions)),
                overall_stability=0.0,
                stability_analysis={},
                transition_probabilities=np.zeros((len(regime_predictions), len(regime_predictions))),
                transition_matrix=np.zeros((len(np.unique(regime_predictions)), len(np.unique(regime_predictions)))),
                transition_analysis={},
                uncertainty_estimates=np.zeros(len(regime_predictions)),
                confidence_scores=np.zeros(len(regime_predictions)),
                uncertainty_analysis={},
                adaptation_scores=np.zeros(len(regime_predictions)),
                learning_indicators={},
                meta_learning_analysis={},
                analysis_time=execution_time
            )
    
    def _analyze_regime_stability(self, regime_predictions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Analyze regime stability."""
        try:
            tprint("🔒 Analyzing regime stability...", color="blue")
            stability_scores = np.zeros(len(regime_predictions))
            
            for i in range(len(regime_predictions)):
                # Calculate stability for current point
                lookback = min(self.config.stability_window, i)
                lookahead = min(self.config.stability_window, len(regime_predictions) - i - 1)
                
                if lookback > 0:
                    past_regimes = regime_predictions[i-lookback:i]
                    past_consistency = np.mean(past_regimes == regime_predictions[i])
                else:
                    past_consistency = 1.0
                
                if lookahead > 0:
                    future_regimes = regime_predictions[i+1:i+1+lookahead]
                    future_consistency = np.mean(future_regimes == regime_predictions[i])
                else:
                    future_consistency = 1.0
                
                stability_scores[i] = (past_consistency + future_consistency) / 2.0
            
            # Calculate stability analysis
            stability_analysis = {
                'mean_stability': np.mean(stability_scores),
                'std_stability': np.std(stability_scores),
                'min_stability': np.min(stability_scores),
                'max_stability': np.max(stability_scores),
                'stable_periods': np.sum(stability_scores >= self.config.stability_threshold),
                'unstable_periods': np.sum(stability_scores < self.config.stability_threshold)
            }
            
            tprint(f"✅ Regime stability analysis completed: {np.mean(stability_scores):.3f} average stability", color="green")
            return stability_scores, stability_analysis
            
        except Exception as e:
            self.logger.warning(f"Regime stability analysis failed: {e}")
            tprint(f"❌ Regime stability analysis failed: {e}", color="red")
            return np.zeros(len(regime_predictions)), {}
    
    def _analyze_regime_transitions(self, regime_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Analyze regime transitions."""
        try:
            tprint("🔄 Analyzing regime transitions...", color="blue")
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)
            tprint(f"📊 Found {n_regimes} unique regimes", color="cyan")
            
            # Calculate transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regime_predictions) - 1):
                current_regime = regime_predictions[i]
                next_regime = regime_predictions[i + 1]
                
                if current_regime in unique_regimes and next_regime in unique_regimes:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    next_idx = np.where(unique_regimes == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            # Calculate transition probabilities for each point
            transition_probs = np.zeros((len(regime_predictions), n_regimes))
            
            for i in range(len(regime_predictions)):
                current_regime = regime_predictions[i]
                if current_regime in unique_regimes:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    transition_probs[i] = transition_matrix[current_idx]
                else:
                    transition_probs[i] = np.ones(n_regimes) / n_regimes
            
            # Calculate transition analysis
            transition_analysis = {
                'transition_matrix': transition_matrix,
                'total_transitions': np.sum(transition_matrix),
                'transition_entropy': self._calculate_transition_entropy(transition_matrix),
                'most_likely_transitions': self._find_most_likely_transitions(transition_matrix, unique_regimes)
            }
            
            return transition_probs, transition_matrix, transition_analysis
            
        except Exception as e:
            self.logger.warning(f"Regime transition analysis failed: {e}")
            n_regimes = len(np.unique(regime_predictions))
            return np.zeros((len(regime_predictions), n_regimes)), np.zeros((n_regimes, n_regimes)), {}
    
    def _analyze_uncertainty(self, regime_predictions: np.ndarray, 
                           regime_probabilities: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Analyze uncertainty in regime predictions."""
        try:
            uncertainty_estimates = np.zeros(len(regime_predictions))
            
            if regime_probabilities is not None:
                # Use probability-based uncertainty
                if regime_probabilities.ndim > 1:
                    # Multi-class probabilities
                    entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-8), axis=1)
                    max_entropy = np.log(regime_probabilities.shape[1])
                    uncertainty_estimates = entropy / max_entropy
                else:
                    # Binary probabilities
                    uncertainty_estimates = 1.0 - np.abs(regime_probabilities - 0.5) * 2
            else:
                # Use regime stability as proxy for uncertainty
                stability_scores, _ = self._analyze_regime_stability(regime_predictions)
                uncertainty_estimates = 1.0 - stability_scores
            
            # Calculate uncertainty analysis
            uncertainty_analysis = {
                'mean_uncertainty': np.mean(uncertainty_estimates),
                'std_uncertainty': np.std(uncertainty_estimates),
                'high_uncertainty_periods': np.sum(uncertainty_estimates > 0.7),
                'low_uncertainty_periods': np.sum(uncertainty_estimates < 0.3),
                'uncertainty_trend': self._calculate_uncertainty_trend(uncertainty_estimates)
            }
            
            return uncertainty_estimates, uncertainty_analysis
            
        except Exception as e:
            self.logger.warning(f"Uncertainty analysis failed: {e}")
            return np.zeros(len(regime_predictions)), {}
    
    def _analyze_meta_learning(self, regime_predictions: np.ndarray, 
                             market_data: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Analyze meta-learning adaptation."""
        try:
            adaptation_scores = np.zeros(len(regime_predictions))
            
            # Simple meta-learning analysis based on regime changes
            for i in range(1, len(regime_predictions)):
                if regime_predictions[i] != regime_predictions[i-1]:
                    # Regime change detected
                    # Calculate adaptation score based on recent performance
                    lookback = min(10, i)
                    recent_regimes = regime_predictions[i-lookback:i]
                    
                    # Adaptation score based on regime diversity
                    regime_diversity = len(np.unique(recent_regimes)) / len(recent_regimes)
                    adaptation_scores[i] = regime_diversity
                else:
                    # No regime change
                    adaptation_scores[i] = adaptation_scores[i-1] * 0.9  # Decay
            
            # Calculate meta-learning analysis
            meta_learning_analysis = {
                'mean_adaptation': np.mean(adaptation_scores),
                'adaptation_rate': self._calculate_adaptation_rate(adaptation_scores),
                'learning_indicators': self._calculate_learning_indicators(regime_predictions, market_data),
                'adaptation_quality': self._assess_adaptation_quality(adaptation_scores)
            }
            
            return adaptation_scores, meta_learning_analysis
            
        except Exception as e:
            self.logger.warning(f"Meta-learning analysis failed: {e}")
            return np.zeros(len(regime_predictions)), {}
    
    def _calculate_confidence_scores(self, regime_predictions: np.ndarray, 
                                  regime_probabilities: Optional[np.ndarray]) -> np.ndarray:
        """Calculate confidence scores for regime predictions."""
        try:
            if regime_probabilities is not None:
                if regime_probabilities.ndim > 1:
                    # Multi-class probabilities - use max probability as confidence
                    confidence_scores = np.max(regime_probabilities, axis=1)
                else:
                    # Binary probabilities
                    confidence_scores = np.maximum(regime_probabilities, 1.0 - regime_probabilities)
            else:
                # Use stability as proxy for confidence
                stability_scores, _ = self._analyze_regime_stability(regime_predictions)
                confidence_scores = stability_scores
            
            return confidence_scores
            
        except Exception as e:
            self.logger.warning(f"Confidence score calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_learning_indicators(self, regime_predictions: np.ndarray, 
                                      market_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate learning indicators."""
        try:
            indicators = {
                'regime_diversity': len(np.unique(regime_predictions)) / len(regime_predictions),
                'transition_frequency': np.sum(np.diff(regime_predictions) != 0) / len(regime_predictions),
                'stability_trend': self._calculate_stability_trend(regime_predictions),
                'learning_rate': self._calculate_learning_rate(regime_predictions)
            }
            
            if market_data is not None:
                indicators['market_adaptation'] = self._calculate_market_adaptation(regime_predictions, market_data)
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f"Learning indicator calculation failed: {e}")
            return {}
    
    def _calculate_transition_matrix(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate transition matrix."""
        try:
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regime_predictions) - 1):
                current_regime = regime_predictions[i]
                next_regime = regime_predictions[i + 1]
                
                if current_regime in unique_regimes and next_regime in unique_regimes:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    next_idx = np.where(unique_regimes == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Normalize
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"Transition matrix calculation failed: {e}")
            n_regimes = len(np.unique(regime_predictions))
            return np.eye(n_regimes) / n_regimes
    
    def _calculate_transition_entropy(self, transition_matrix: np.ndarray) -> float:
        """Calculate transition entropy."""
        try:
            entropy = 0.0
            for row in transition_matrix:
                for prob in row:
                    if prob > 0:
                        entropy -= prob * np.log(prob)
            return entropy
            
        except Exception:
            return 0.0
    
    def _find_most_likely_transitions(self, transition_matrix: np.ndarray, 
                                    unique_regimes: np.ndarray) -> List[Dict[str, Any]]:
        """Find most likely transitions."""
        try:
            transitions = []
            for i, from_regime in enumerate(unique_regimes):
                for j, to_regime in enumerate(unique_regimes):
                    if transition_matrix[i, j] > 0.1:  # Threshold for significant transitions
                        transitions.append({
                            'from_regime': from_regime,
                            'to_regime': to_regime,
                            'probability': transition_matrix[i, j]
                        })
            
            # Sort by probability
            transitions.sort(key=lambda x: x['probability'], reverse=True)
            return transitions
            
        except Exception:
            return []
    
    def _calculate_uncertainty_trend(self, uncertainty_estimates: np.ndarray) -> float:
        """Calculate uncertainty trend."""
        try:
            if len(uncertainty_estimates) < 2:
                return 0.0
            
            # Linear trend
            x = np.arange(len(uncertainty_estimates))
            slope = np.polyfit(x, uncertainty_estimates, 1)[0]
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_adaptation_rate(self, adaptation_scores: np.ndarray) -> float:
        """Calculate adaptation rate."""
        try:
            if len(adaptation_scores) < 2:
                return 0.0
            
            # Calculate rate of change
            changes = np.diff(adaptation_scores)
            return np.mean(np.abs(changes))
            
        except Exception:
            return 0.0
    
    def _assess_adaptation_quality(self, adaptation_scores: np.ndarray) -> str:
        """Assess adaptation quality."""
        try:
            mean_adaptation = np.mean(adaptation_scores)
            
            if mean_adaptation > 0.7:
                return "excellent"
            elif mean_adaptation > 0.5:
                return "good"
            elif mean_adaptation > 0.3:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
    def _calculate_stability_trend(self, regime_predictions: np.ndarray) -> float:
        """Calculate stability trend."""
        try:
            if len(regime_predictions) < 10:
                return 0.0
            
            # Calculate rolling stability
            window_size = min(10, len(regime_predictions) // 2)
            stability_scores = []
            
            for i in range(window_size, len(regime_predictions)):
                window_regimes = regime_predictions[i-window_size:i]
                stability = 1.0 - np.sum(np.diff(window_regimes) != 0) / (len(window_regimes) - 1)
                stability_scores.append(stability)
            
            if len(stability_scores) < 2:
                return 0.0
            
            # Calculate trend
            x = np.arange(len(stability_scores))
            slope = np.polyfit(x, stability_scores, 1)[0]
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_learning_rate(self, regime_predictions: np.ndarray) -> float:
        """Calculate learning rate."""
        try:
            if len(regime_predictions) < 10:
                return 0.0
            
            # Calculate regime change frequency over time
            window_size = min(10, len(regime_predictions) // 2)
            change_rates = []
            
            for i in range(window_size, len(regime_predictions)):
                window_regimes = regime_predictions[i-window_size:i]
                change_rate = np.sum(np.diff(window_regimes) != 0) / (len(window_regimes) - 1)
                change_rates.append(change_rate)
            
            if len(change_rates) < 2:
                return 0.0
            
            # Calculate trend in change rates
            x = np.arange(len(change_rates))
            slope = np.polyfit(x, change_rates, 1)[0]
            return -slope  # Negative slope indicates learning (fewer changes over time)
            
        except Exception:
            return 0.0
    
    def _calculate_market_adaptation(self, regime_predictions: np.ndarray, 
                                  market_data: np.ndarray) -> float:
        """Calculate market adaptation."""
        try:
            if len(market_data) != len(regime_predictions):
                return 0.0
            
            # Calculate correlation between regime changes and market changes
            regime_changes = np.diff(regime_predictions) != 0
            
            if market_data.ndim > 1:
                market_changes = np.diff(market_data[:, 0])  # Use first column (price)
            else:
                market_changes = np.diff(market_data)
            
            if len(regime_changes) != len(market_changes):
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(regime_changes.astype(float), market_changes)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_tree_based_regime_patterns(self, regime_predictions: np.ndarray,
                                          regime_probabilities: Optional[np.ndarray],
                                          model_metadata: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Analyze tree-based regime patterns."""
        try:
            tree_analysis = {}
            
            if model_metadata is None:
                return {}
            
            # Extract tree-specific information
            tree_depth = model_metadata.get('tree_depth', 5)
            tree_importance = model_metadata.get('feature_importance', {})
            tree_interpretability = model_metadata.get('interpretability', 0.8)
            
            # Calculate tree-based regime patterns
            depth_penalty = max(0.0, 1.0 - (tree_depth - 3) * self.config.tree_depth_penalty)
            interpretability_score = tree_interpretability * self.config.tree_interpretability_weight
            
            # Feature importance based regime analysis
            importance_scores = np.zeros(len(regime_predictions))
            for i, regime in enumerate(regime_predictions):
                regime_importance = tree_importance.get(f'regime_{regime}', 0.5)
                importance_scores[i] = regime_importance
            
            tree_analysis = {
                'depth_penalty': np.full(len(regime_predictions), depth_penalty),
                'interpretability_score': np.full(len(regime_predictions), interpretability_score),
                'importance_score': importance_scores
            }
            
            return tree_analysis
            
        except Exception as e:
            self.logger.warning(f"Tree-based regime pattern analysis failed: {e}")
            return {}
    
    def _analyze_neural_based_regime_patterns(self, regime_predictions: np.ndarray,
                                            regime_probabilities: Optional[np.ndarray],
                                            model_metadata: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Analyze neural-based regime patterns."""
        try:
            neural_analysis = {}
            
            if model_metadata is None:
                return {}
            
            # Extract neural-specific information
            model_confidence = model_metadata.get('confidence', 0.8)
            architecture_complexity = model_metadata.get('architecture_complexity', 0.5)
            uncertainty_estimates = model_metadata.get('uncertainty', None)
            
            # Calculate neural-based regime patterns
            confidence_scores = np.full(len(regime_predictions), model_confidence)
            complexity_scores = np.full(len(regime_predictions), 1.0 - architecture_complexity)
            
            # Uncertainty-based regime analysis
            if uncertainty_estimates is not None:
                uncertainty_scores = 1.0 - uncertainty_estimates * self.config.neural_uncertainty_weight
            else:
                uncertainty_scores = np.ones(len(regime_predictions)) * 0.5
            
            neural_analysis = {
                'confidence_score': confidence_scores,
                'complexity_score': complexity_scores,
                'uncertainty_score': uncertainty_scores
            }
            
            return neural_analysis
            
        except Exception as e:
            self.logger.warning(f"Neural-based regime pattern analysis failed: {e}")
            return {}
    
    def _analyze_hybrid_regime_patterns(self, regime_predictions: np.ndarray,
                                     regime_probabilities: Optional[np.ndarray],
                                     model_metadata: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Analyze hybrid regime patterns."""
        try:
            hybrid_analysis = {}
            
            if model_metadata is None:
                return {}
            
            # Extract hybrid information
            tree_confidence = model_metadata.get('tree_confidence', 0.7)
            neural_confidence = model_metadata.get('neural_confidence', 0.8)
            consensus_score = model_metadata.get('consensus', 0.5)
            ensemble_weight = model_metadata.get('ensemble_weight', 0.5)
            
            # Calculate hybrid regime patterns
            weighted_confidence = (
                tree_confidence * (1.0 - ensemble_weight) +
                neural_confidence * ensemble_weight
            )
            
            consensus_scores = np.full(len(regime_predictions), consensus_score)
            confidence_scores = np.full(len(regime_predictions), weighted_confidence)
            ensemble_scores = np.full(len(regime_predictions), ensemble_weight)
            
            hybrid_analysis = {
                'consensus_score': consensus_scores,
                'confidence_score': confidence_scores,
                'ensemble_score': ensemble_scores
            }
            
            return hybrid_analysis
            
        except Exception as e:
            self.logger.warning(f"Hybrid regime pattern analysis failed: {e}")
            return {}
    
    def _adjust_scores_with_tree_analysis(self, base_scores: np.ndarray, 
                                        tree_analysis: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust scores based on tree analysis."""
        try:
            if not tree_analysis:
                return base_scores
            
            # Apply tree-specific adjustments
            adjusted_scores = base_scores.copy()
            
            if 'importance_score' in tree_analysis:
                adjusted_scores *= tree_analysis['importance_score']
            
            if 'depth_penalty' in tree_analysis:
                adjusted_scores *= tree_analysis['depth_penalty']
            
            if 'interpretability_score' in tree_analysis:
                adjusted_scores *= tree_analysis['interpretability_score']
            
            return np.clip(adjusted_scores, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Tree score adjustment failed: {e}")
            return base_scores
    
    def _adjust_scores_with_neural_analysis(self, base_scores: np.ndarray, 
                                           neural_analysis: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust scores based on neural analysis."""
        try:
            if not neural_analysis:
                return base_scores
            
            # Apply neural-specific adjustments
            adjusted_scores = base_scores.copy()
            
            if 'confidence_score' in neural_analysis:
                adjusted_scores *= neural_analysis['confidence_score']
            
            if 'uncertainty_score' in neural_analysis:
                adjusted_scores *= neural_analysis['uncertainty_score']
            
            if 'complexity_score' in neural_analysis:
                adjusted_scores *= neural_analysis['complexity_score']
            
            return np.clip(adjusted_scores, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Neural score adjustment failed: {e}")
            return base_scores
    
    def _adjust_scores_with_hybrid_analysis(self, base_scores: np.ndarray, 
                                          hybrid_analysis: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust scores based on hybrid analysis."""
        try:
            if not hybrid_analysis:
                return base_scores
            
            # Apply hybrid-specific adjustments
            adjusted_scores = base_scores.copy()
            
            if 'consensus_score' in hybrid_analysis:
                adjusted_scores *= hybrid_analysis['consensus_score']
            
            if 'confidence_score' in hybrid_analysis:
                adjusted_scores *= hybrid_analysis['confidence_score']
            
            if 'ensemble_score' in hybrid_analysis:
                adjusted_scores *= hybrid_analysis['ensemble_score']
            
            return np.clip(adjusted_scores, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Hybrid score adjustment failed: {e}")
            return base_scores


# Convenience functions
def create_unified_regime_analyzer(config: Optional[RegimeAnalysisConfig] = None) -> UnifiedRegimeAnalyzer:
    """Create a unified regime analyzer."""
    if config is None:
        config = RegimeAnalysisConfig()
    return UnifiedRegimeAnalyzer(config)


def quick_regime_analysis(regime_predictions: np.ndarray,
                        regime_probabilities: Optional[np.ndarray] = None,
                        config: Optional[RegimeAnalysisConfig] = None) -> RegimeAnalysisResult:
    """Quick regime analysis with default settings."""
    analyzer = create_unified_regime_analyzer(config)
    return analyzer.analyze(regime_predictions, regime_probabilities)