"""
NAS Regime Analyzer for comprehensive regime analysis and reporting.

This module provides detailed analysis of detected regimes and generates
comprehensive reports for ML model training and decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class RegimeAnalysisResult:
    """Result of regime analysis."""
    regime_statistics: Dict[str, Any]
    regime_characteristics: Dict[str, Any]
    regime_transitions: Dict[str, Any]
    regime_quality_metrics: Dict[str, Any]
    economic_analysis: Dict[str, Any]
    trading_analysis: Dict[str, Any]
    micro_regime_analysis: Dict[str, Any]
    ml_training_features: Dict[str, Any]
    execution_time: float
    timestamp: str


class NASRegimeAnalyzer:
    """Comprehensive regime analyzer for NAS clustering results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS regime analyzer.
        
        Args:
            config: Analyzer configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis settings
        self.enable_detailed_reporting = config.get('enable_detailed_reporting', True)
        self.enable_ml_training_features = config.get('enable_ml_training_features', True)
        self.enable_economic_analysis = config.get('enable_economic_analysis', True)
        self.enable_trading_analysis = config.get('enable_trading_analysis', True)
        self.enable_micro_regime_analysis = config.get('enable_micro_regime_analysis', True)
        
        # ML model support
        self.supported_ml_models = ['DeepScale', 'LGBM', 'XGBoost', 'RandomForest', 'SVM', 'NeuralNetwork']
        self.ml_feature_types = ['regime_features', 'transition_features', 'economic_features', 'trading_features', 'micro_regime_features']
        
        self.logger.info("✅ NAS Regime Analyzer initialized")
    
    def analyze_regimes(self, clustering_result: Any, feature_result: Any, 
                       micro_regime_result: Any, market_data: np.ndarray,
                       timestamps: np.ndarray) -> RegimeAnalysisResult:
        """Perform comprehensive regime analysis.
        
        Args:
            clustering_result: NAS clustering result
            feature_result: Feature extraction result
            micro_regime_result: Micro-regime detection result
            market_data: Market data array
            timestamps: Timestamps array
            
        Returns:
            RegimeAnalysisResult with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🔍 Starting comprehensive regime analysis")
            
            # Analyze regime statistics
            regime_statistics = self._analyze_regime_statistics(
                clustering_result, market_data, timestamps
            )
            
            # Analyze regime characteristics
            regime_characteristics = self._analyze_regime_characteristics(
                clustering_result, market_data, timestamps
            )
            
            # Analyze regime transitions
            regime_transitions = self._analyze_regime_transitions(
                clustering_result, market_data, timestamps
            )
            
            # Analyze regime quality metrics
            regime_quality_metrics = self._analyze_regime_quality_metrics(
                clustering_result, feature_result
            )
            
            # Analyze economic aspects
            economic_analysis = self._analyze_economic_aspects(
                clustering_result, market_data, timestamps
            )
            
            # Analyze trading aspects
            trading_analysis = self._analyze_trading_aspects(
                clustering_result, market_data, timestamps
            )
            
            # Analyze micro-regimes
            micro_regime_analysis = self._analyze_micro_regimes(
                micro_regime_result, market_data, timestamps
            )
            
            # Generate ML training features
            ml_training_features = self._generate_ml_training_features(
                clustering_result, feature_result, micro_regime_result,
                market_data, timestamps
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = RegimeAnalysisResult(
                regime_statistics=regime_statistics,
                regime_characteristics=regime_characteristics,
                regime_transitions=regime_transitions,
                regime_quality_metrics=regime_quality_metrics,
                economic_analysis=economic_analysis,
                trading_analysis=trading_analysis,
                micro_regime_analysis=micro_regime_analysis,
                ml_training_features=ml_training_features,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"✅ Comprehensive regime analysis completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Regime analysis failed: {e}")
            return RegimeAnalysisResult(
                regime_statistics={},
                regime_characteristics={},
                regime_transitions={},
                regime_quality_metrics={},
                economic_analysis={},
                trading_analysis={},
                micro_regime_analysis={},
                ml_training_features={},
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
    
    def _analyze_regime_statistics(self, clustering_result: Any, 
                                 market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze regime statistics."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            # Basic statistics
            n_regimes = len(unique_labels)
            n_samples = len(labels)
            
            # Regime distribution
            regime_distribution = {}
            regime_percentages = {}
            regime_durations = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                count = np.sum(regime_mask)
                percentage = (count / n_samples) * 100
                
                regime_distribution[f'regime_{label}'] = int(count)
                regime_percentages[f'regime_{label}'] = float(percentage)
                
                # Calculate regime duration
                regime_indices = np.where(regime_mask)[0]
                if len(regime_indices) > 0:
                    duration = len(regime_indices)
                    regime_durations[f'regime_{label}'] = int(duration)
            
            # Regime stability
            regime_changes = np.sum(np.diff(labels) != 0)
            stability_score = 1.0 - (regime_changes / (n_samples - 1)) if n_samples > 1 else 0.0
            
            # Regime persistence
            regime_persistence = self._calculate_regime_persistence(labels)
            
            return {
                'n_regimes': n_regimes,
                'n_samples': n_samples,
                'regime_distribution': regime_distribution,
                'regime_percentages': regime_percentages,
                'regime_durations': regime_durations,
                'stability_score': float(stability_score),
                'regime_changes': int(regime_changes),
                'regime_persistence': regime_persistence,
                'mean_regime_duration': float(np.mean(list(regime_durations.values()))),
                'std_regime_duration': float(np.std(list(regime_durations.values())))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime statistics analysis failed: {e}")
            return {}
    
    def _analyze_regime_characteristics(self, clustering_result: Any,
                                      market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze regime characteristics."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            regime_characteristics = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue
                
                regime_data = market_data[regime_mask]
                regime_timestamps = timestamps[regime_mask]
                
                # Price characteristics
                if regime_data.shape[1] >= 4:
                    open_prices = regime_data[:, 0]
                    high_prices = regime_data[:, 1]
                    low_prices = regime_data[:, 2]
                    close_prices = regime_data[:, 3]
                    
                    # Price statistics
                    price_range = np.max(high_prices) - np.min(low_prices)
                    price_change = (close_prices[-1] - open_prices[0]) / open_prices[0]
                    price_volatility = np.std(close_prices) / np.mean(close_prices)
                    
                    # Trend analysis
                    trend_direction = 1 if price_change > 0.01 else -1 if price_change < -0.01 else 0
                    trend_strength = abs(price_change)
                    
                    # Volatility analysis
                    volatility_level = 'high' if price_volatility > 0.05 else 'medium' if price_volatility > 0.02 else 'low'
                    
                    regime_characteristics[f'regime_{label}'] = {
                        'price_range': float(price_range),
                        'price_change': float(price_change),
                        'price_volatility': float(price_volatility),
                        'trend_direction': trend_direction,
                        'trend_strength': float(trend_strength),
                        'volatility_level': volatility_level,
                        'mean_price': float(np.mean(close_prices)),
                        'std_price': float(np.std(close_prices)),
                        'min_price': float(np.min(low_prices)),
                        'max_price': float(np.max(high_prices))
                    }
                
                # Volume characteristics
                if regime_data.shape[1] >= 5:
                    volumes = regime_data[:, 4]
                    volume_mean = np.mean(volumes)
                    volume_std = np.std(volumes)
                    volume_ratio = volume_mean / np.mean(market_data[:, 4])
                    
                    regime_characteristics[f'regime_{label}'].update({
                        'volume_mean': float(volume_mean),
                        'volume_std': float(volume_std),
                        'volume_ratio': float(volume_ratio),
                        'volume_level': 'high' if volume_ratio > 1.5 else 'medium' if volume_ratio > 0.8 else 'low'
                    })
            
            return regime_characteristics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime characteristics analysis failed: {e}")
            return {}
    
    def _analyze_regime_transitions(self, clustering_result: Any,
                                   market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            transition_counts = {}
            
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                
                if current_regime in unique_labels and next_regime in unique_labels:
                    current_idx = np.where(unique_labels == current_regime)[0][0]
                    next_idx = np.where(unique_labels == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
                    
                    transition_key = f'regime_{current_regime}_to_regime_{next_regime}'
                    transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
            
            # Calculate transition probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            # Calculate transition entropy
            transition_entropy = 0.0
            for i in range(n_regimes):
                for j in range(n_regimes):
                    if transition_probs[i, j] > 0:
                        transition_entropy -= transition_probs[i, j] * np.log2(transition_probs[i, j])
            
            # Calculate regime persistence
            regime_persistence = {}
            for i, label in enumerate(unique_labels):
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_indices = np.where(regime_mask)[0]
                    persistence = len(regime_indices) / n_regimes
                    regime_persistence[f'regime_{label}'] = float(persistence)
            
            return {
                'transition_matrix': transition_matrix.tolist(),
                'transition_probabilities': transition_probs.tolist(),
                'transition_counts': transition_counts,
                'transition_entropy': float(transition_entropy),
                'regime_persistence': regime_persistence,
                'n_transitions': int(np.sum(transition_matrix)),
                'transition_rate': float(np.sum(transition_matrix) / (len(labels) - 1))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transitions analysis failed: {e}")
            return {}
    
    def _analyze_regime_quality_metrics(self, clustering_result: Any, 
                                       feature_result: Any) -> Dict[str, Any]:
        """Analyze regime quality metrics."""
        try:
            quality_metrics = clustering_result.quality_metrics
            
            # Standard clustering metrics
            silhouette_score = quality_metrics.get('silhouette_score', 0.0)
            calinski_harabasz_score = quality_metrics.get('calinski_harabasz_score', 0.0)
            nas_score = quality_metrics.get('nas_score', 0.0)
            
            # Quality assessment
            quality_level = 'high' if nas_score > 0.7 else 'medium' if nas_score > 0.5 else 'low'
            
            # Regime quality breakdown
            regime_quality_breakdown = {}
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    # Calculate regime-specific quality metrics
                    regime_size = np.sum(regime_mask)
                    regime_quality = min(regime_size / len(labels), 1.0)
                    regime_quality_breakdown[f'regime_{label}'] = {
                        'size': int(regime_size),
                        'quality_score': float(regime_quality),
                        'quality_level': 'high' if regime_quality > 0.1 else 'medium' if regime_quality > 0.05 else 'low'
                    }
            
            return {
                'silhouette_score': float(silhouette_score),
                'calinski_harabasz_score': float(calinski_harabasz_score),
                'nas_score': float(nas_score),
                'quality_level': quality_level,
                'regime_quality_breakdown': regime_quality_breakdown,
                'overall_quality': quality_level
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime quality metrics analysis failed: {e}")
            return {}
    
    def _analyze_economic_aspects(self, clustering_result: Any,
                                market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze economic aspects of regimes."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            economic_analysis = {}
            
            # Overall economic significance
            economic_scores = clustering_result.economic_significance_scores
            mean_economic_significance = np.mean(economic_scores)
            std_economic_significance = np.std(economic_scores)
            
            # Regime-wise economic analysis
            regime_economic_analysis = {}
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_economic_scores = economic_scores[regime_mask]
                    regime_economic_analysis[f'regime_{label}'] = {
                        'mean_economic_significance': float(np.mean(regime_economic_scores)),
                        'std_economic_significance': float(np.std(regime_economic_scores)),
                        'min_economic_significance': float(np.min(regime_economic_scores)),
                        'max_economic_significance': float(np.max(regime_economic_scores)),
                        'economic_level': 'high' if np.mean(regime_economic_scores) > 0.7 else 'medium' if np.mean(regime_economic_scores) > 0.5 else 'low'
                    }
            
            # Economic regime ranking
            economic_ranking = sorted(
                regime_economic_analysis.items(),
                key=lambda x: x[1]['mean_economic_significance'],
                reverse=True
            )
            
            return {
                'overall_economic_significance': float(mean_economic_significance),
                'economic_significance_std': float(std_economic_significance),
                'regime_economic_analysis': regime_economic_analysis,
                'economic_ranking': economic_ranking,
                'top_economic_regimes': [item[0] for item in economic_ranking[:3]]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic aspects analysis failed: {e}")
            return {}
    
    def _analyze_trading_aspects(self, clustering_result: Any,
                               market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze trading aspects of regimes."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            trading_analysis = {}
            
            # Overall trading viability
            trading_scores = clustering_result.trading_viability_scores
            mean_trading_viability = np.mean(trading_scores)
            std_trading_viability = np.std(trading_scores)
            
            # Regime-wise trading analysis
            regime_trading_analysis = {}
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_trading_scores = trading_scores[regime_mask]
                    regime_trading_analysis[f'regime_{label}'] = {
                        'mean_trading_viability': float(np.mean(regime_trading_scores)),
                        'std_trading_viability': float(np.std(regime_trading_scores)),
                        'min_trading_viability': float(np.min(regime_trading_scores)),
                        'max_trading_viability': float(np.max(regime_trading_scores)),
                        'trading_level': 'high' if np.mean(regime_trading_scores) > 0.7 else 'medium' if np.mean(regime_trading_scores) > 0.5 else 'low'
                    }
            
            # Trading regime ranking
            trading_ranking = sorted(
                regime_trading_analysis.items(),
                key=lambda x: x[1]['mean_trading_viability'],
                reverse=True
            )
            
            return {
                'overall_trading_viability': float(mean_trading_viability),
                'trading_viability_std': float(std_trading_viability),
                'regime_trading_analysis': regime_trading_analysis,
                'trading_ranking': trading_ranking,
                'top_trading_regimes': [item[0] for item in trading_ranking[:3]]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading aspects analysis failed: {e}")
            return {}
    
    def _analyze_micro_regimes(self, micro_regime_result: Any,
                             market_data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze micro-regimes."""
        try:
            if micro_regime_result is None:
                return {}
            
            micro_regimes = micro_regime_result.micro_regimes
            micro_regime_types = micro_regime_result.micro_regime_types
            micro_regime_scores = micro_regime_result.micro_regime_scores
            detection_accuracy = micro_regime_result.detection_accuracy
            
            # Micro-regime distribution
            unique_micro_regimes = np.unique(micro_regimes)
            micro_regime_distribution = {}
            micro_regime_percentages = {}
            
            for regime_id in unique_micro_regimes:
                regime_mask = micro_regimes == regime_id
                count = np.sum(regime_mask)
                percentage = (count / len(micro_regimes)) * 100
                
                micro_regime_distribution[f'micro_regime_{regime_id}'] = int(count)
                micro_regime_percentages[f'micro_regime_{regime_id}'] = float(percentage)
            
            # Micro-regime type analysis
            micro_regime_type_analysis = {}
            for i, regime_type in enumerate(micro_regime_types):
                regime_mask = micro_regimes == i
                if np.any(regime_mask):
                    regime_scores = micro_regime_scores[regime_mask]
                    micro_regime_type_analysis[regime_type.value] = {
                        'count': int(np.sum(regime_mask)),
                        'mean_score': float(np.mean(regime_scores)),
                        'std_score': float(np.std(regime_scores)),
                        'quality_level': 'high' if np.mean(regime_scores) > 0.7 else 'medium' if np.mean(regime_scores) > 0.5 else 'low'
                    }
            
            return {
                'detection_accuracy': float(detection_accuracy),
                'n_micro_regimes': len(unique_micro_regimes),
                'micro_regime_distribution': micro_regime_distribution,
                'micro_regime_percentages': micro_regime_percentages,
                'micro_regime_type_analysis': micro_regime_type_analysis,
                'micro_regime_types': [t.value for t in micro_regime_types]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime analysis failed: {e}")
            return {}
    
    def _generate_ml_training_features(self, clustering_result: Any, feature_result: Any,
                                     micro_regime_result: Any, market_data: np.ndarray,
                                     timestamps: np.ndarray) -> Dict[str, Any]:
        """Generate features for ML model training (DeepScale, LGBM, XGBoost, etc.)."""
        try:
            labels = clustering_result.labels
            unique_labels = np.unique(labels)
            
            # Base regime features
            regime_features = {
                'regime_labels': labels.tolist(),
                'regime_centers': clustering_result.cluster_centers.tolist(),
                'regime_statistics': clustering_result.statistics,
                'regime_quality_metrics': clustering_result.quality_metrics
            }
            
            # Transition features
            transition_features = {
                'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                'regime_persistence': self._calculate_regime_persistence(labels),
                'regime_change_frequency': float(np.sum(np.diff(labels) != 0) / (len(labels) - 1))
            }
            
            # Economic features
            economic_features = {
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'mean_economic_significance': float(np.mean(clustering_result.economic_significance_scores)),
                'std_economic_significance': float(np.std(clustering_result.economic_significance_scores)),
                'economic_regime_ranking': self._calculate_economic_regime_ranking(labels, clustering_result.economic_significance_scores)
            }
            
            # Trading features
            trading_features = {
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                'mean_trading_viability': float(np.mean(clustering_result.trading_viability_scores)),
                'std_trading_viability': float(np.std(clustering_result.trading_viability_scores)),
                'trading_regime_ranking': self._calculate_trading_regime_ranking(labels, clustering_result.trading_viability_scores)
            }
            
            # Micro-regime features
            micro_regime_features = {}
            if micro_regime_result is not None:
                micro_regime_features = {
                    'micro_regime_labels': micro_regime_result.micro_regimes.tolist(),
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types],
                    'micro_regime_scores': micro_regime_result.micro_regime_scores.tolist(),
                    'micro_regime_detection_accuracy': micro_regime_result.detection_accuracy,
                    'micro_regime_type_distribution': self._calculate_micro_regime_type_distribution(micro_regime_result)
                }
            
            # Market data features
            market_features = {
                'price_features': self._extract_price_features(market_data),
                'volume_features': self._extract_volume_features(market_data),
                'volatility_features': self._extract_volatility_features(market_data),
                'momentum_features': self._extract_momentum_features(market_data)
            }
            
            # Combined ML training features
            ml_training_features = {
                'regime_features': regime_features,
                'transition_features': transition_features,
                'economic_features': economic_features,
                'trading_features': trading_features,
                'micro_regime_features': micro_regime_features,
                'market_features': market_features,
                'feature_metadata': {
                    'n_samples': len(labels),
                    'n_regimes': len(unique_labels),
                    'n_features': len(feature_result.feature_names) if feature_result else 0,
                    'supported_ml_models': self.supported_ml_models,
                    'feature_types': self.ml_feature_types,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return ml_training_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ ML training features generation failed: {e}")
            return {}
    
    def _calculate_regime_persistence(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate regime persistence."""
        try:
            unique_labels = np.unique(labels)
            persistence = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_indices = np.where(regime_mask)[0]
                    persistence[f'regime_{label}'] = float(len(regime_indices) / len(labels))
            
            return persistence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime persistence calculation failed: {e}")
            return {}
    
    def _calculate_economic_regime_ranking(self, labels: np.ndarray, economic_scores: np.ndarray) -> List[Tuple[str, float]]:
        """Calculate economic regime ranking."""
        try:
            unique_labels = np.unique(labels)
            ranking = []
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_economic_score = np.mean(economic_scores[regime_mask])
                    ranking.append((f'regime_{label}', float(regime_economic_score)))
            
            return sorted(ranking, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic regime ranking calculation failed: {e}")
            return []
    
    def _calculate_trading_regime_ranking(self, labels: np.ndarray, trading_scores: np.ndarray) -> List[Tuple[str, float]]:
        """Calculate trading regime ranking."""
        try:
            unique_labels = np.unique(labels)
            ranking = []
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_trading_score = np.mean(trading_scores[regime_mask])
                    ranking.append((f'regime_{label}', float(regime_trading_score)))
            
            return sorted(ranking, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading regime ranking calculation failed: {e}")
            return []
    
    def _calculate_micro_regime_type_distribution(self, micro_regime_result: Any) -> Dict[str, int]:
        """Calculate micro-regime type distribution."""
        try:
            if micro_regime_result is None:
                return {}
            
            micro_regime_types = micro_regime_result.micro_regime_types
            distribution = {}
            
            for regime_type in micro_regime_types:
                type_name = regime_type.value
                distribution[type_name] = distribution.get(type_name, 0) + 1
            
            return distribution
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime type distribution calculation failed: {e}")
            return {}
    
    def _extract_price_features(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Extract price-based features for ML training."""
        try:
            if market_data.shape[1] < 4:
                return {}
            
            close_prices = market_data[:, 3]
            high_prices = market_data[:, 1]
            low_prices = market_data[:, 2]
            
            return {
                'close_prices': close_prices.tolist(),
                'high_prices': high_prices.tolist(),
                'low_prices': low_prices.tolist(),
                'price_ranges': (high_prices - low_prices).tolist(),
                'price_returns': np.diff(close_prices).tolist(),
                'price_volatility': float(np.std(close_prices) / np.mean(close_prices))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price features extraction failed: {e}")
            return {}
    
    def _extract_volume_features(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Extract volume-based features for ML training."""
        try:
            if market_data.shape[1] < 5:
                return {}
            
            volumes = market_data[:, 4]
            
            return {
                'volumes': volumes.tolist(),
                'volume_returns': np.diff(volumes).tolist(),
                'volume_volatility': float(np.std(volumes) / np.mean(volumes)),
                'volume_trend': float(np.polyfit(range(len(volumes)), volumes, 1)[0])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume features extraction failed: {e}")
            return {}
    
    def _extract_volatility_features(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Extract volatility-based features for ML training."""
        try:
            if market_data.shape[1] < 4:
                return {}
            
            close_prices = market_data[:, 3]
            returns = np.diff(close_prices) / close_prices[:-1]
            
            return {
                'returns': returns.tolist(),
                'volatility': float(np.std(returns)),
                'volatility_trend': float(np.polyfit(range(len(returns)), returns, 1)[0]),
                'volatility_clusters': self._cluster_volatility(returns)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility features extraction failed: {e}")
            return {}
    
    def _extract_momentum_features(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Extract momentum-based features for ML training."""
        try:
            if market_data.shape[1] < 4:
                return {}
            
            close_prices = market_data[:, 3]
            
            # Calculate momentum indicators
            momentum_5 = close_prices[5:] - close_prices[:-5] if len(close_prices) > 5 else []
            momentum_10 = close_prices[10:] - close_prices[:-10] if len(close_prices) > 10 else []
            
            return {
                'momentum_5': momentum_5.tolist(),
                'momentum_10': momentum_10.tolist(),
                'momentum_trend': float(np.polyfit(range(len(momentum_5)), momentum_5, 1)[0]) if len(momentum_5) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum features extraction failed: {e}")
            return {}
    
    def _cluster_volatility(self, returns: np.ndarray) -> List[int]:
        """Cluster volatility levels."""
        try:
            if len(returns) < 3:
                return []
            
            # Simple volatility clustering
            volatility_levels = np.abs(returns)
            kmeans = KMeans(n_clusters=3, random_state=42)
            volatility_clusters = kmeans.fit_predict(volatility_levels.reshape(-1, 1))
            
            return volatility_clusters.tolist()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility clustering failed: {e}")
            return []
    
    def generate_detailed_report(self, analysis_result: RegimeAnalysisResult) -> Dict[str, Any]:
        """Generate detailed comprehensive report."""
        try:
            report = {
                'executive_summary': {
                    'analysis_timestamp': analysis_result.timestamp,
                    'execution_time': analysis_result.execution_time,
                    'n_regimes': analysis_result.regime_statistics.get('n_regimes', 0),
                    'n_samples': analysis_result.regime_statistics.get('n_samples', 0),
                    'overall_quality': analysis_result.regime_quality_metrics.get('overall_quality', 'unknown'),
                    'economic_significance': analysis_result.economic_analysis.get('overall_economic_significance', 0.0),
                    'trading_viability': analysis_result.trading_analysis.get('overall_trading_viability', 0.0),
                    'micro_regime_accuracy': analysis_result.micro_regime_analysis.get('detection_accuracy', 0.0)
                },
                
                'regime_analysis': {
                    'statistics': analysis_result.regime_statistics,
                    'characteristics': analysis_result.regime_characteristics,
                    'transitions': analysis_result.regime_transitions,
                    'quality_metrics': analysis_result.regime_quality_metrics
                },
                
                'economic_analysis': analysis_result.economic_analysis,
                'trading_analysis': analysis_result.trading_analysis,
                'micro_regime_analysis': analysis_result.micro_regime_analysis,
                
                'ml_training_features': {
                    'feature_summary': {
                        'n_regime_features': len(analysis_result.ml_training_features.get('regime_features', {})),
                        'n_transition_features': len(analysis_result.ml_training_features.get('transition_features', {})),
                        'n_economic_features': len(analysis_result.ml_training_features.get('economic_features', {})),
                        'n_trading_features': len(analysis_result.ml_training_features.get('trading_features', {})),
                        'n_micro_regime_features': len(analysis_result.ml_training_features.get('micro_regime_features', {})),
                        'n_market_features': len(analysis_result.ml_training_features.get('market_features', {}))
                    },
                    'supported_ml_models': self.supported_ml_models,
                    'feature_types': self.ml_feature_types,
                    'feature_metadata': analysis_result.ml_training_features.get('feature_metadata', {})
                },
                
                'recommendations': self._generate_recommendations(analysis_result),
                'performance_metrics': self._calculate_performance_metrics(analysis_result)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Detailed report generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, analysis_result: RegimeAnalysisResult) -> List[str]:
        """Generate recommendations based on analysis."""
        try:
            recommendations = []
            
            # Quality recommendations
            overall_quality = analysis_result.regime_quality_metrics.get('overall_quality', 'unknown')
            if overall_quality == 'low':
                recommendations.append("Consider adjusting NAS architecture parameters to improve regime quality")
            
            # Economic significance recommendations
            economic_significance = analysis_result.economic_analysis.get('overall_economic_significance', 0.0)
            if economic_significance < 0.6:
                recommendations.append("Consider enhancing economic significance features for better regime detection")
            
            # Trading viability recommendations
            trading_viability = analysis_result.trading_analysis.get('overall_trading_viability', 0.0)
            if trading_viability < 0.6:
                recommendations.append("Consider improving trading viability features for better trading decisions")
            
            # Micro-regime recommendations
            micro_regime_accuracy = analysis_result.micro_regime_analysis.get('detection_accuracy', 0.0)
            if micro_regime_accuracy < 0.6:
                recommendations.append("Consider adjusting micro-regime detection sensitivity for better micro-regime detection")
            
            # ML training recommendations
            if analysis_result.ml_training_features:
                recommendations.append("ML training features are ready for DeepScale, LGBM, XGBoost, and other ML models")
                recommendations.append("Consider using regime features as input features for downstream ML models")
                recommendations.append("Micro-regime features can provide additional context for ML model training")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Recommendations generation failed: {e}")
            return []
    
    def _calculate_performance_metrics(self, analysis_result: RegimeAnalysisResult) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            return {
                'execution_time': analysis_result.execution_time,
                'regime_detection_accuracy': analysis_result.regime_quality_metrics.get('nas_score', 0.0),
                'economic_significance_score': analysis_result.economic_analysis.get('overall_economic_significance', 0.0),
                'trading_viability_score': analysis_result.trading_analysis.get('overall_trading_viability', 0.0),
                'micro_regime_detection_accuracy': analysis_result.micro_regime_analysis.get('detection_accuracy', 0.0),
                'overall_performance': np.mean([
                    analysis_result.regime_quality_metrics.get('nas_score', 0.0),
                    analysis_result.economic_analysis.get('overall_economic_significance', 0.0),
                    analysis_result.trading_analysis.get('overall_trading_viability', 0.0),
                    analysis_result.micro_regime_analysis.get('detection_accuracy', 0.0)
                ])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {}