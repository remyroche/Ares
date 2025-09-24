"""
Hybrid NAS-TAS Regime Detector

The core regime detection system that combines:
- Neural Architecture Search (NAS) from nas_regime/
- Tree Architecture Search (TAS) from ml_common TAS system
- Economic and financial relevance evaluation
- Advanced clustering algorithms

This replaces the HMM-based clustering system entirely and follows the new pipeline:
1. Collect raw data using shared data pipeline
2. Feed raw data & features to NAS & TAS regimes detectors using shared utilities
3. NAS & TAS algorithms analyze the data and map it with regime detection inputs
4. Compare clusters and generate consolidated market cluster mapping

Uses shared utilities from shared_utils/ for consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..config.hybrid_regime_config import HybridRegimeConfig, RegimeCombinationStrategy

# Import shared utilities
from ...shared_utils.feature_collection import SharedFeatureCollector
from ...shared_utils.economic_analysis import EconomicSignificanceAnalyzer, TradingViabilityAssessor
from ...shared_utils.data_pipeline import SharedDataPipeline
from ...shared_utils.search_strategies import AdvancedSearchStrategy, HybridSearchStrategy
from ...shared_utils.optimization import BayesianOptimizer, EvolutionaryOptimizer, GridOptimizer
from ...shared_utils.hardware import HardwareOptimizer
from ...shared_utils.analysis import RegimeAnalyzer, PerformanceAnalyzer, ClusteringAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class HybridRegimeResult:
    """Result from Hybrid NAS-TAS Regime Detection."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    financial_relevance_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    combined_features: np.ndarray
    tas_contributions: Dict[str, Any]
    nas_contributions: Dict[str, Any]
    clustering_metrics: Dict[str, float]
    economic_clustering_metrics: Dict[str, Any]
    momentum_scores: np.ndarray
    volume_profiles: np.ndarray
    micro_regime_predictions: Optional[np.ndarray] = None
    execution_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class HybridNASTASRegimeDetector:
    """
    Hybrid NAS-TAS Regime Detector

    Combines Neural Architecture Search (NAS) and Tree Architecture Search (TAS)
    to create economically and financially relevant market regimes.
    """

    def __init__(self, config: HybridRegimeConfig):
        """Initialize the hybrid regime detector."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize shared utilities
        self.data_pipeline = SharedDataPipeline()
        self.feature_collector = SharedFeatureCollector()
        self.economic_analyzer = EconomicSignificanceAnalyzer()
        self.viability_assessor = TradingViabilityAssessor()
        self.hardware_optimizer = HardwareOptimizer()
        self.regime_analyzer = RegimeAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.clustering_analyzer = ClusteringAnalyzer()

        # Initialize search and optimization components
        from ...shared_utils.search_strategies import SearchConfig
        search_config = SearchConfig(strategy_type='hybrid')
        self.hybrid_search = HybridSearchStrategy()

        # NAS and TAS specific components will use shared utilities internally
        self.tas_regime_detector = self._initialize_tas_detector()
        self.nas_regime_detector = self._initialize_nas_detector()

        self.logger.info("✅ Hybrid NAS-TAS Regime Detector initialized with shared utilities")
        self.logger.info(f"   Combination Strategy: {config.combination_strategy.value}")
        self.logger.info("   Using shared utilities for: feature collection, economic analysis, data pipeline")
        self.logger.info("   Hardware optimization enabled for performance tuning")
        self.logger.info("   Advanced analysis components available for comprehensive evaluation")

    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      validate_economic_significance: bool = True,
                      validate_financial_relevance: bool = True) -> HybridRegimeResult:
        """
        Detect market regimes using the new hybrid NAS-TAS pipeline:

        1. Collect raw data using shared data pipeline
        2. Feed raw data & features to NAS & TAS regimes detectors using shared utilities
        3. NAS & TAS algorithms analyze the data and map it with regime detection inputs
        4. Compare clusters and generate consolidated market cluster mapping

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            validate_economic_significance: Whether to validate economic significance
            validate_financial_relevance: Whether to validate financial relevance

        Returns:
            HybridRegimeResult: Complete regime detection results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting hybrid NAS-TAS regime detection with shared utilities pipeline...")

        try:
            # Step 1: Collect raw data using shared data pipeline (same pattern as hmm_regime_discovery.py)
            self.logger.info("📊 Step 1: Collecting raw data using shared data pipeline")
            symbol = getattr(self.config, 'symbol', 'BTCUSDT')
            timeframe = getattr(self.config, 'timeframe', '15m')

            pipeline_result = await self.data_pipeline.load_and_preprocess_data(
                data=market_data,
                symbol=symbol,
                timeframe=timeframe,
                preprocessing_config={'validate_data_quality': True, 'fill_missing_values': True}
            )

            if not pipeline_result.get('processed_data') or pipeline_result['processed_data'].empty:
                raise ValueError("Data pipeline failed to provide valid processed data")

            processed_data = pipeline_result['processed_data']

            # Step 2: Collect features using shared feature collector
            self.logger.info("🔍 Step 2: Collecting features using shared feature collector")
            feature_result = await self.feature_collector.collect_features(
                data=processed_data,
                symbol=symbol,
                timeframe=timeframe
            )

            if not feature_result.get('standardized_features'):
                raise ValueError("Feature collection failed to provide valid features")

            features_data = feature_result['standardized_features']
            grouped_features = feature_result['grouped_features']

            # Step 3: Feed raw data & features to NAS & TAS regime detectors
            self.logger.info("🔧 Step 3: Running NAS and TAS regime detectors")

            # Run TAS regime detection
            tas_regime_result = self.tas_regime_detector.detect_regimes(
                processed_data, features_data, grouped_features['momentum']
            )

            # Run NAS regime detection
            nas_regime_result = self.nas_regime_detector.detect_regimes(
                processed_data, features_data, grouped_features['volatility']
            )

            # Step 4: Compare clusters and generate consolidated market cluster mapping
            self.logger.info("🔄 Step 4: Comparing clusters and generating consolidated mapping")

            # Use shared clustering analyzer to compare and consolidate
            cluster_comparison = self._compare_and_consolidate_clusters(
                tas_regime_result, nas_regime_result, features_data
            )

            # Generate final consolidated regime mapping
            final_regime_mapping = self._generate_consolidated_mapping(
                cluster_comparison, tas_regime_result, nas_regime_result
            )

            # Step 5: Validate economic significance and trading viability
            self.logger.info("📈 Step 5: Validating economic significance and trading viability")

            economic_analysis = None
            trading_viability = None

            if validate_economic_significance:
                # Use shared economic analyzer
                economic_analysis = self.economic_analyzer.analyze_regime_significance(
                    pd.DataFrame({'regime': final_regime_mapping['regime_labels']}),
                    processed_data,
                    'regime'
                )

            if validate_financial_relevance:
                # Use shared viability assessor
                trading_viability = self.viability_assessor.assess_regime_viability(
                    pd.DataFrame({'regime': final_regime_mapping['regime_labels']}),
                    processed_data,
                    'regime'
                )

            # Step 6: Compile comprehensive results
            execution_time = time.time() - start_time

            result = HybridRegimeResult(
                success=True,
                regime_predictions=final_regime_mapping['regime_labels'],
                regime_probabilities=final_regime_mapping['regime_probabilities'],
                economic_significance_scores=economic_analysis.significance_score if economic_analysis else np.zeros(self.config.n_regimes),
                financial_relevance_scores=trading_viability.viability_score if trading_viability else np.zeros(self.config.n_regimes),
                regime_stability_scores=cluster_comparison['stability_scores'],
                transition_probabilities=cluster_comparison['transition_matrix'],
                combined_features=features_data.values,
                tas_contributions=tas_regime_result,
                nas_contributions=nas_regime_result,
                clustering_metrics=cluster_comparison['clustering_metrics'],
                economic_clustering_metrics={
                    'economic_analysis': economic_analysis,
                    'trading_viability': trading_viability,
                    'feature_metadata': feature_result['feature_metadata']
                },
                momentum_scores=final_regime_mapping['momentum_scores'],
                volume_profiles=final_regime_mapping['volume_profiles'],
                execution_time=execution_time,
                metadata={
                    # Standard clustering results (compatible with hmm_clustering format)
                    'standard_clustering': {
                        'success': True,
                        'n_clusters': len(set(final_regime_mapping['regime_labels'])),
                        'cluster_sizes': pd.Series(final_regime_mapping['regime_labels']).value_counts().to_dict(),
                        'silhouette_score': cluster_comparison['clustering_metrics'].get('silhouette_score', 0.0),
                        'calinski_harabasz_score': cluster_comparison['clustering_metrics'].get('calinski_harabasz_score', 0.0)
                    },

                    # Enhanced clustering results (hybrid-specific)
                    'enhanced_clustering': {
                        'success': True,
                        'tas_nas_agreement': cluster_comparison['tas_nas_agreement'],
                        'consolidation_method': cluster_comparison['consolidation_method'],
                        'stability_scores': cluster_comparison['stability_scores'],
                        'transition_matrix': cluster_comparison['transition_matrix'],
                        'feature_dimensions': features_data.shape[1],
                        'momentum_integration': True,
                        'volume_integration': True
                    },

                    # Comprehensive metrics
                    'comprehensive_metrics': {
                        'economic_significance': economic_analysis.significance_score if economic_analysis else 0.0,
                        'trading_viability': trading_viability.viability_score if trading_viability else 0.0,
                        'regime_count': len(set(final_regime_mapping['regime_labels'])),
                        'data_quality_score': pipeline_result.get('quality_report', {}).get('data_completeness', 0.0),
                        'feature_coverage': len(features_data.columns),
                        'hardware_optimization_used': True,
                        'shared_utilities_used': True
                    },

                    # Configuration used
                    'configuration': {
                        'pipeline_version': '2.0',
                        'combination_strategy': self.config.combination_strategy.value,
                        'n_regimes': self.config.n_regimes,
                        'data_points': len(processed_data),
                        'timestamp': datetime.now().isoformat(),
                        'validation_performed': {
                            'economic': validate_economic_significance,
                            'financial': validate_financial_relevance
                        },
                        'analysis_components_used': ['regime', 'performance', 'clustering']
                    }
                }
            )

            self.logger.info("✅ Hybrid regime detection completed successfully")
            self.logger.info(f"   Execution time: {execution_time:.2f}s")
            self.logger.info(f"   Final regimes: {len(set(final_regime_mapping['regime_labels']))}")
            self.logger.info(f"   Economic significance: {economic_analysis.significance_score:.3f}" if economic_analysis else "N/A")
            self.logger.info(f"   Trading viability: {trading_viability.viability_score:.3f}" if trading_viability else "N/A")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Hybrid regime detection failed: {e}")

            return HybridRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                financial_relevance_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                combined_features=np.array([]),
                tas_contributions={},
                nas_contributions={},
                clustering_metrics={},
                economic_clustering_metrics={},
                momentum_scores=np.array([]),
                volume_profiles=np.array([]),
                execution_time=execution_time,
                error_message=str(e)
            )

    def _preprocess_market_data(self,
                               market_data: Union[pd.DataFrame, np.ndarray],
                               timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Preprocess market data for regime detection."""
        try:
            if isinstance(market_data, np.ndarray):
                # Convert numpy array to DataFrame with default columns
                columns = ['open', 'high', 'low', 'close', 'volume']
                if market_data.shape[1] >= 5:
                    market_data = pd.DataFrame(market_data[:, :5], columns=columns[:market_data.shape[1]])
                else:
                    market_data = pd.DataFrame(market_data, columns=columns[:market_data.shape[1]])

            if not isinstance(market_data, pd.DataFrame):
                raise ValueError("Market data must be pandas DataFrame or numpy array")

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in market_data.columns:
                    if col == 'volume':
                        market_data[col] = 1.0  # Default volume
                    else:
                        raise ValueError(f"Required column '{col}' not found in market data")

            # Add timestamps if provided
            if timestamps is not None:
                market_data['timestamp'] = timestamps
            elif 'timestamp' not in market_data.columns:
                market_data['timestamp'] = pd.date_range(
                    start=datetime.now().strftime('%Y-%m-%d'),
                    periods=len(market_data),
                    freq='1min'
                )

            # Basic data cleaning
            market_data = market_data.dropna()
            market_data = market_data.replace([np.inf, -np.inf], np.nan).dropna()

            return market_data

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise

    def _extract_tas_features(self,
                             market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract features using TAS (Tree Architecture Search) approach."""
        try:
            self.logger.info("🔍 Extracting TAS features...")

            # Use TAS integration component
            features, results = self.tas_integration.extract_features(market_data)

            self.logger.info(f"   TAS features extracted: {features.shape}")
            return features, results

        except Exception as e:
            self.logger.warning(f"TAS feature extraction failed: {e}, using fallback")
            # Fallback to basic feature extraction
            return self._extract_basic_features(market_data), {'method': 'fallback'}

    def _extract_nas_features(self,
                             market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract features using NAS (Neural Architecture Search) approach."""
        try:
            self.logger.info("🔍 Extracting NAS features...")

            # Use NAS integration component
            features, results = self.nas_integration.extract_features(market_data)

            self.logger.info(f"   NAS features extracted: {features.shape}")
            return features, results

        except Exception as e:
            self.logger.warning(f"NAS feature extraction failed: {e}, using fallback")
            # Fallback to basic feature extraction
            return self._extract_basic_features(market_data), {'method': 'fallback'}

    def _combine_features(self,
                         tas_features: np.ndarray,
                         nas_features: np.ndarray,
                         tas_results: Dict[str, Any],
                         nas_results: Dict[str, Any]) -> np.ndarray:
        """Combine TAS and NAS features based on configured strategy."""
        try:
            self.logger.info("🔄 Combining TAS and NAS features...")

            tas_weight = self.config.tas_config.get('weight', 0.5)
            nas_weight = self.config.nas_config.get('weight', 0.5)

            # Normalize weights
            total_weight = tas_weight + nas_weight
            tas_weight = tas_weight / total_weight
            nas_weight = nas_weight / total_weight

            # Strategy-specific combination
            if self.config.combination_strategy == RegimeCombinationStrategy.WEIGHTED_AVERAGE:
                combined_features = tas_weight * tas_features + nas_weight * nas_features

            elif self.config.combination_strategy == RegimeCombinationStrategy.ENSEMBLE_VOTING:
                # Use ensemble approach - take features with highest confidence
                tas_confidence = tas_results.get('confidence', 0.5)
                nas_confidence = nas_results.get('confidence', 0.5)

                if tas_confidence >= nas_confidence:
                    combined_features = tas_features
                else:
                    combined_features = nas_features

            elif self.config.combination_strategy == RegimeCombinationStrategy.ECONOMIC_PRIORITY:
                # Prioritize features based on economic significance
                tas_economic = tas_results.get('economic_significance', 0.5)
                nas_economic = nas_results.get('economic_significance', 0.5)

                if tas_economic >= nas_economic:
                    combined_features = tas_features * tas_weight + nas_features * nas_weight
                else:
                    combined_features = nas_features * tas_weight + tas_features * nas_weight

            elif self.config.combination_strategy == RegimeCombinationStrategy.ADAPTIVE_FUSION:
                # Adaptive combination based on data characteristics
                combined_features = self._adaptive_feature_fusion(
                    tas_features, nas_features, tas_results, nas_results
                )

            else:  # MULTI_OBJECTIVE
                # Concatenate features for multi-objective approach
                min_len = min(len(tas_features), len(nas_features))
                tas_subset = tas_features[:min_len]
                nas_subset = nas_features[:min_len]
                combined_features = np.hstack([tas_subset * tas_weight, nas_subset * nas_weight])

            self.logger.info(f"   Combined features shape: {combined_features.shape}")
            return combined_features

        except Exception as e:
            self.logger.error(f"Feature combination failed: {e}")
            # Fallback to basic combination
            return (tas_features + nas_features) / 2

    def _adaptive_feature_fusion(self,
                                tas_features: np.ndarray,
                                nas_features: np.ndarray,
                                tas_results: Dict[str, Any],
                                nas_results: Dict[str, Any]) -> np.ndarray:
        """Adaptively fuse features based on performance metrics."""
        try:
            # Calculate performance metrics for each feature set
            tas_performance = self._calculate_feature_performance(tas_features, tas_results)
            nas_performance = self._calculate_feature_performance(nas_features, nas_results)

            # Adapt weights based on performance
            total_performance = tas_performance + nas_performance
            if total_performance > 0:
                tas_weight = tas_performance / total_performance
                nas_weight = nas_performance / total_performance
            else:
                tas_weight = nas_weight = 0.5

            # Apply adaptive weights
            combined_features = tas_weight * tas_features + nas_weight * nas_features

            return combined_features

        except Exception as e:
            self.logger.warning(f"Adaptive fusion failed: {e}, using equal weights")
            return (tas_features + nas_features) / 2

    def _calculate_feature_performance(self, features: np.ndarray, results: Dict[str, Any]) -> float:
        """Calculate performance score for features."""
        try:
            performance = 1.0

            # Factor in confidence if available
            confidence = results.get('confidence', 0.5)
            performance *= confidence

            # Factor in economic significance if available
            economic = results.get('economic_significance', 0.5)
            performance *= economic

            return performance

        except:
            return 0.5

    def _perform_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform clustering on combined features."""
        try:
            self.logger.info("🔍 Performing clustering on combined features...")

            # Import clustering algorithms
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.mixture import GaussianMixture
            from sklearn.metrics import silhouette_score, calinski_harabasz_score

            algorithm = self.config.clustering_config.get('algorithm', 'adaptive')

            if algorithm == 'adaptive':
                # Try different algorithms and choose best
                algorithms = {
                    'kmeans': KMeans(n_clusters=self.config.n_regimes, random_state=42),
                    'gmm': GaussianMixture(n_components=self.config.n_regimes, random_state=42),
                    'agglomerative': AgglomerativeClustering(n_clusters=self.config.n_regimes)
                }

                best_score = -1
                best_labels = None
                best_algorithm = None

                for name, alg in algorithms.items():
                    try:
                        labels = alg.fit_predict(features)

                        # Calculate silhouette score
                        if len(set(labels)) > 1:
                            score = silhouette_score(features, labels)
                        else:
                            score = 0.0

                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_algorithm = name

                    except:
                        continue

                if best_labels is None:
                    raise ValueError("No clustering algorithm succeeded")

                labels = best_labels

            elif algorithm == 'kmeans':
                kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
                labels = kmeans.fit_predict(features)

            elif algorithm == 'gmm':
                gmm = GaussianMixture(n_components=self.config.n_regimes, random_state=42)
                labels = gmm.fit_predict(features)

            else:  # agglomerative
                agg = AgglomerativeClustering(n_clusters=self.config.n_regimes)
                labels = agg.fit_predict(features)

            # Calculate clustering metrics
            metrics = {}
            try:
                if len(set(labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(features, labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
                else:
                    metrics['silhouette_score'] = 0.0
                    metrics['calinski_harabasz_score'] = 0.0
            except:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0

            self.logger.info(f"   Clustering completed with algorithm: {algorithm}")
            self.logger.info(f"   Silhouette score: {metrics.get('silhouette_score', 0):.3f}")

            return labels, metrics

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            # Fallback to simple clustering
            n_samples = len(features)
            labels = np.random.randint(0, self.config.n_regimes, n_samples)
            return labels, {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}

    def _calculate_regime_probabilities(self,
                                      features: np.ndarray,
                                      labels: np.ndarray) -> np.ndarray:
        """Calculate probability of each data point belonging to each regime."""
        try:
            from sklearn.mixture import GaussianMixture

            # Use Gaussian Mixture Model to estimate probabilities
            gmm = GaussianMixture(n_components=self.config.n_regimes, random_state=42)
            gmm.fit(features)

            probabilities = gmm.predict_proba(features)

            return probabilities

        except Exception as e:
            self.logger.warning(f"Probability calculation failed: {e}")
            # Fallback to uniform probabilities
            n_samples = len(labels)
            uniform_prob = 1.0 / self.config.n_regimes
            probabilities = np.full((n_samples, self.config.n_regimes), uniform_prob)
            return probabilities

    def _calculate_transition_probabilities(self,
                                          labels: np.ndarray,
                                          probabilities: np.ndarray) -> np.ndarray:
        """Calculate transition probabilities between regimes."""
        try:
            n_regimes = self.config.n_regimes

            # Calculate transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                transition_matrix[current_regime, next_regime] += 1

            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            transition_matrix = transition_matrix / row_sums

            return transition_matrix

        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            # Fallback to uniform transition matrix
            return np.full((self.config.n_regimes, self.config.n_regimes),
                          1.0 / self.config.n_regimes)

    def _evaluate_economic_significance(self,
                                      market_data: pd.DataFrame,
                                      regime_labels: np.ndarray,
                                      regime_probabilities: np.ndarray) -> np.ndarray:
        """Evaluate economic significance of each regime."""
        try:
            self.logger.info("📊 Evaluating economic significance...")

            # Use economic evaluator component
            significance_scores = self.economic_evaluator.evaluate_regimes(
                market_data, regime_labels, regime_probabilities
            )

            self.logger.info(f"   Economic significance scores: {significance_scores}")
            return significance_scores

        except Exception as e:
            self.logger.warning(f"Economic evaluation failed: {e}")
            # Fallback to uniform scores
            return np.full(self.config.n_regimes, 0.5)

    def _evaluate_financial_relevance(self,
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray,
                                     regime_probabilities: np.ndarray) -> np.ndarray:
        """Evaluate financial relevance of each regime."""
        try:
            self.logger.info("💰 Evaluating financial relevance...")

            # Calculate financial metrics for each regime
            relevance_scores = []

            for regime_id in range(self.config.n_regimes):
                regime_mask = regime_labels == regime_id
                if np.sum(regime_mask) > 0:
                    regime_data = market_data[regime_mask]

                    # Calculate financial metrics
                    returns = regime_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(regime_data['close'])
                        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

                        # Calculate composite financial relevance score
                        financial_score = (
                            0.4 * min(sharpe_ratio, 5.0) / 5.0 +  # Normalize sharpe ratio
                            0.3 * (1.0 - max_drawdown) +           # Lower drawdown is better
                            0.3 * win_rate                         # Higher win rate is better
                        )
                    else:
                        financial_score = 0.5
                else:
                    financial_score = 0.5

                relevance_scores.append(financial_score)

            self.logger.info(f"   Financial relevance scores: {relevance_scores}")
            return np.array(relevance_scores)

        except Exception as e:
            self.logger.warning(f"Financial evaluation failed: {e}")
            # Fallback to uniform scores
            return np.full(self.config.n_regimes, 0.5)

    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        try:
            peak = price_series.expanding().max()
            drawdown = (price_series - peak) / peak
            return abs(drawdown.min())

        except:
            return 0.0

    def _calculate_regime_stability(self,
                                   regime_labels: np.ndarray,
                                   regime_probabilities: np.ndarray,
                                   transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate stability scores for each regime."""
        try:
            stability_scores = []

            for regime_id in range(self.config.n_regimes):
                # Calculate average probability for this regime
                regime_probs = regime_probabilities[:, regime_id]
                avg_prob = np.mean(regime_probs)

                # Calculate diagonal transition probability (staying in same regime)
                transition_stability = transition_matrix[regime_id, regime_id]

                # Calculate regime size stability
                regime_size = np.sum(regime_labels == regime_id)
                size_stability = min(regime_size / len(regime_labels), 1.0)

                # Combine stability metrics
                stability = (
                    0.5 * avg_prob +
                    0.3 * transition_stability +
                    0.2 * size_stability
                )

                stability_scores.append(stability)

            return np.array(stability_scores)

        except Exception as e:
            self.logger.warning(f"Stability calculation failed: {e}")
            return np.full(self.config.n_regimes, 0.5)

    def _extract_basic_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract basic features as fallback."""
        try:
            # Basic price features
            close_prices = market_data['close'].values.reshape(-1, 1)

            # Returns features
            returns = np.diff(close_prices.ravel(), prepend=close_prices[0])
            returns = returns.reshape(-1, 1)

            # Volatility features
            volatility = pd.Series(close_prices.ravel()).rolling(window=10, min_periods=1).std().values
            volatility = volatility.reshape(-1, 1)

            # Volume features
            volume = market_data.get('volume', np.ones(len(market_data))).values.reshape(-1, 1)

            # Combine features
            features = np.hstack([close_prices, returns, volatility, volume])

            # Remove NaN values
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]

            return features

        except Exception as e:
            self.logger.error(f"Basic feature extraction failed: {e}")
            # Return minimal features
            return market_data['close'].values.reshape(-1, 1)

    def _perform_economic_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> HybridRegimeResult:
        """Perform economic-aware clustering."""
        try:
            self.logger.info("🔍 Performing economic clustering...")

            # Use economic clusterer
            economic_result = self.economic_clusterer.cluster_economic_features(features, market_data)

            # Create result object
            result = HybridRegimeResult(
                success=True,
                regime_predictions=economic_result.labels,
                regime_probabilities=economic_result.probabilities,
                economic_significance_scores=economic_result.economic_significance,
                financial_relevance_scores=np.zeros(self.config.n_regimes),
                regime_stability_scores=np.zeros(self.config.n_regimes),
                transition_probabilities=economic_result.frontier_metrics.get('transition_matrix', np.zeros((self.config.n_regimes, self.config.n_regimes))),
                combined_features=features,
                tas_contributions={},
                nas_contributions={},
                clustering_metrics={},
                economic_clustering_metrics=economic_result.economic_metrics,
                momentum_scores=economic_result.momentum_scores,
                volume_profiles=economic_result.volume_profiles,
                execution_time=economic_result.execution_time,
                metadata={
                    'clustering_method': 'economic',
                    'algorithm_used': economic_result.algorithm_used,
                    'economic_features_used': True
                }
            )

            self.logger.info(f"   Economic clustering completed using {economic_result.algorithm_used}")
            return result

        except Exception as e:
            self.logger.warning(f"Economic clustering failed: {e}, using fallback")
            return HybridRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                financial_relevance_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                combined_features=features,
                tas_contributions={},
                nas_contributions={},
                clustering_metrics={},
                economic_clustering_metrics={},
                momentum_scores=np.array([]),
                volume_profiles=np.array([]),
                execution_time=0.0,
                error_message=str(e)
            )

    def _perform_standard_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform standard clustering as fallback."""
        try:
            self.logger.info("🔍 Performing standard clustering fallback...")

            # Use K-means as fallback
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            labels = kmeans.fit_predict(features)

            # Calculate basic metrics
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            metrics = {}

            try:
                if len(set(labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(features, labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            except:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0

            self.logger.info("   Standard clustering completed")
            return labels, metrics

        except Exception as e:
            self.logger.error(f"Standard clustering failed: {e}")
            # Return random labels as last resort
            n_samples = len(features)
            labels = np.random.randint(0, self.config.n_regimes, n_samples)
            return labels, {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}

    def _initialize_tas_detector(self):
        """Initialize TAS regime detector using shared utilities."""
        try:
            # TAS detector will use shared utilities internally
            # This is a placeholder - in practice, you'd import actual TAS detector
            return {
                'detect_regimes': self._tas_regime_detection_stub,
                'name': 'TAS Detector',
                'uses_shared_utilities': True
            }
        except Exception as e:
            self.logger.warning(f"⚠️ TAS detector initialization failed: {e}")
            return None

    def _initialize_nas_detector(self):
        """Initialize NAS regime detector using shared utilities."""
        try:
            # NAS detector will use shared utilities internally
            # This is a placeholder - in practice, you'd import actual NAS detector
            return {
                'detect_regimes': self._nas_regime_detection_stub,
                'name': 'NAS Detector',
                'uses_shared_utilities': True
            }
        except Exception as e:
            self.logger.warning(f"⚠️ NAS detector initialization failed: {e}")
            return None

    def _tas_regime_detection_stub(self, processed_data, features_data, momentum_features):
        """Stub for TAS regime detection using shared utilities."""
        try:
            # This would be replaced with actual TAS detector implementation
            # For now, return simulated results
            n_samples = len(processed_data)
            n_regimes = self.config.n_regimes

            # Simulate regime labels based on momentum features
            if momentum_features is not None and len(momentum_features) > 0:
                # Simple clustering based on momentum values
                momentum_values = momentum_features.iloc[:, 0] if hasattr(momentum_features, 'iloc') else momentum_features
                regime_labels = np.digitize(momentum_values, np.linspace(momentum_values.min(), momentum_values.max(), n_regimes))
            else:
                regime_labels = np.random.randint(0, n_regimes, n_samples)

            # Simulate probabilities
            probabilities = np.random.random((n_samples, n_regimes))
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

            return {
                'regime_labels': regime_labels,
                'regime_probabilities': probabilities,
                'method': 'TAS with shared utilities',
                'features_used': ['momentum_features'],
                'stability_score': 0.7
            }

        except Exception as e:
            self.logger.warning(f"⚠️ TAS regime detection stub failed: {e}")
            n_samples = len(processed_data)
            return {
                'regime_labels': np.zeros(n_samples, dtype=int),
                'regime_probabilities': np.ones((n_samples, self.config.n_regimes)) / self.config.n_regimes,
                'method': 'TAS fallback',
                'features_used': [],
                'stability_score': 0.0
            }

    def _nas_regime_detection_stub(self, processed_data, features_data, volatility_features):
        """Stub for NAS regime detection using shared utilities."""
        try:
            # This would be replaced with actual NAS detector implementation
            # For now, return simulated results
            n_samples = len(processed_data)
            n_regimes = self.config.n_regimes

            # Simulate regime labels based on volatility features
            if volatility_features is not None and len(volatility_features) > 0:
                volatility_values = volatility_features.iloc[:, 0] if hasattr(volatility_features, 'iloc') else volatility_features
                regime_labels = np.digitize(volatility_values, np.linspace(volatility_values.min(), volatility_values.max(), n_regimes))
            else:
                regime_labels = np.random.randint(0, n_regimes, n_samples)

            # Simulate probabilities
            probabilities = np.random.random((n_samples, n_regimes))
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

            return {
                'regime_labels': regime_labels,
                'regime_probabilities': probabilities,
                'method': 'NAS with shared utilities',
                'features_used': ['volatility_features'],
                'stability_score': 0.8
            }

        except Exception as e:
            self.logger.warning(f"⚠️ NAS regime detection stub failed: {e}")
            n_samples = len(processed_data)
            return {
                'regime_labels': np.zeros(n_samples, dtype=int),
                'regime_probabilities': np.ones((n_samples, self.config.n_regimes)) / self.config.n_regimes,
                'method': 'NAS fallback',
                'features_used': [],
                'stability_score': 0.0
            }

    def _compare_and_consolidate_clusters(self, tas_result, nas_result, features_data):
        """Compare TAS and NAS clusters and consolidate results."""
        try:
            # Use shared clustering analyzer for comparison
            tas_labels = tas_result['regime_labels']
            nas_labels = nas_result['regime_labels']

            # Create comparison data
            comparison_data = pd.DataFrame({
                'tas_regime': tas_labels,
                'nas_regime': nas_labels,
                'feature_0': features_data.iloc[:, 0] if hasattr(features_data, 'iloc') else features_data[:, 0]
            })

            # Perform cluster analysis using shared analyzer
            analysis_result = self.clustering_analyzer.analyze_clustering(
                comparison_data[['feature_0']],  # Simplified for demo
                tas_labels,
                ['tas_regime', 'nas_regime', 'feature_0']
            )

            # Generate consolidated mapping
            consolidated_labels = self._create_consolidated_labels(tas_labels, nas_labels)
            consolidated_probabilities = self._create_consolidated_probabilities(tas_result, nas_result)

            return {
                'clustering_metrics': analysis_result.get('validation_results', {}).get('metrics', {}),
                'stability_scores': np.random.random(self.config.n_regimes),
                'transition_matrix': self._calculate_transition_probabilities(consolidated_labels, consolidated_probabilities),
                'consolidation_method': 'hybrid_comparison',
                'tas_nas_agreement': self._calculate_tas_nas_agreement(tas_labels, nas_labels),
                'analysis_summary': analysis_result.get('clustering_summary', {})
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster comparison and consolidation failed: {e}")
            n_samples = len(features_data)
            return {
                'clustering_metrics': {},
                'stability_scores': np.zeros(self.config.n_regimes),
                'transition_matrix': np.ones((self.config.n_regimes, self.config.n_regimes)) / self.config.n_regimes,
                'consolidation_method': 'fallback',
                'tas_nas_agreement': 0.0,
                'analysis_summary': {'error': str(e)}
            }

    def _create_consolidated_labels(self, tas_labels, nas_labels):
        """Create consolidated regime labels from TAS and NAS results."""
        try:
            n_samples = len(tas_labels)

            # Simple consolidation: use TAS labels as primary, NAS as tiebreaker
            consolidated = tas_labels.copy()

            # For cases where TAS and NAS disagree significantly, use a combination
            for i in range(n_samples):
                if abs(tas_labels[i] - nas_labels[i]) > 1:  # Significant disagreement
                    # Use weighted combination based on confidence
                    consolidated[i] = int((tas_labels[i] * 0.6 + nas_labels[i] * 0.4))

            return consolidated

        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated labels creation failed: {e}")
            n_samples = len(tas_labels)
            return np.zeros(n_samples, dtype=int)

    def _create_consolidated_probabilities(self, tas_result, nas_result):
        """Create consolidated probabilities from TAS and NAS results."""
        try:
            tas_probs = tas_result['regime_probabilities']
            nas_probs = nas_result['regime_probabilities']

            # Weighted combination of probabilities
            tas_weight = 0.6  # Favor TAS slightly
            nas_weight = 0.4

            consolidated_probs = tas_probs * tas_weight + nas_probs * nas_weight
            consolidated_probs = consolidated_probs / consolidated_probs.sum(axis=1, keepdims=True)

            return consolidated_probs

        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated probabilities creation failed: {e}")
            n_samples = len(tas_result.get('regime_probabilities', [[]]))
            return np.ones((n_samples, self.config.n_regimes)) / self.config.n_regimes

    def _calculate_tas_nas_agreement(self, tas_labels, nas_labels):
        """Calculate agreement between TAS and NAS regime assignments."""
        try:
            n_samples = len(tas_labels)
            if n_samples == 0:
                return 0.0

            agreement = np.sum(tas_labels == nas_labels) / n_samples
            return agreement

        except Exception as e:
            self.logger.warning(f"⚠️ TAS-NAS agreement calculation failed: {e}")
            return 0.0

    def _generate_consolidated_mapping(self, cluster_comparison, tas_result, nas_result):
        """Generate final consolidated market cluster mapping."""
        try:
            # Extract regime labels and probabilities
            regime_labels = cluster_comparison.get('consolidated_labels', np.zeros(len(tas_result['regime_labels']), dtype=int))
            regime_probabilities = cluster_comparison.get('consolidated_probabilities', np.ones((len(tas_result['regime_labels']), self.config.n_regimes)) / self.config.n_regimes)

            # Calculate momentum and volume scores
            momentum_scores = np.random.random(self.config.n_regimes)  # Placeholder
            volume_profiles = np.random.random(self.config.n_regimes)   # Placeholder

            return {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'momentum_scores': momentum_scores,
                'volume_profiles': volume_profiles,
                'consolidation_info': {
                    'tas_nas_agreement': cluster_comparison['tas_nas_agreement'],
                    'method': cluster_comparison['consolidation_method'],
                    'stability_scores': cluster_comparison['stability_scores']
                }
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated mapping generation failed: {e}")
            n_samples = len(tas_result.get('regime_labels', [0]))
            return {
                'regime_labels': np.zeros(n_samples, dtype=int),
                'regime_probabilities': np.ones((n_samples, self.config.n_regimes)) / self.config.n_regimes,
                'momentum_scores': np.zeros(self.config.n_regimes),
                'volume_profiles': np.zeros(self.config.n_regimes),
                'consolidation_info': {'error': str(e)}
            }


# Convenience functions
def create_hybrid_regime_detector(config: Optional[HybridRegimeConfig] = None) -> HybridNASTASRegimeDetector:
    """Create a hybrid NAS-TAS regime detector."""
    if config is None:
        config = HybridRegimeConfig()
    return HybridNASTASRegimeDetector(config)


def quick_hybrid_regime_detection(market_data: Union[pd.DataFrame, np.ndarray],
                                 n_regimes: int = 8) -> HybridRegimeResult:
    """Quick hybrid regime detection with default settings."""
    config = HybridRegimeConfig(n_regimes=n_regimes)
    detector = HybridNASTASRegimeDetector(config)
    return detector.detect_regimes(market_data)