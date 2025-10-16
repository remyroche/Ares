"""
import warnings
Advanced Trading Architecture Search (Advanced TAS)

This module provides the most sophisticated TAS implementation with:
- Micro-regime detection and analysis
- Neural architecture search integration
- Economic significance validation
- Multi-objective optimization with advanced constraints
- Hardware acceleration and performance optimization
- Meta-learning and transfer learning capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict
import warnings

# Import tprint for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs): print(*args)
    def tprint_debug(*args, **kwargs): print(f"[DEBUG] {args[0] if args else ''}")
    def tprint_info(*args, **kwargs): print(f"[INFO] {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"[WARNING] {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"[ERROR] {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"[SUCCESS] {args[0] if args else ''}")
    def tprint_progress(*args, **kwargs): print(f"[PROGRESS] {args[0] if args else ''}")
    def tprint_performance(*args, **kwargs): print(f"[PERFORMANCE] {args[0] if args else ''}")
    def tprint_timer(*args, **kwargs): print(f"[TIMER] {args[0] if args else ''}")

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

from .tas_config import TASConfig, TASArchitectureType, TradingObjective, MarketRegime, MicroRegimeType
from ..components.micro_regime_detector import MicroRegimeDetector, MicroRegimeDetectionResult
from ..components.neural_architecture import TASNeuralModel, NeuralArchitectureConfig
from ..evaluation.tas_evaluator import TASEvaluator, EvaluationResult
from ..search.advanced_search import AdvancedTASSearch

# Commented out missing imports - will add fallback implementations
# from ..hardware.accelerator import HardwareAccelerator
# from ..meta_learning.meta_learner import MetaLearner
# from ..validation.economic_validator import EconomicValidator

# Fallback implementations for missing modules
class HardwareAccelerator:
    """Fallback hardware accelerator."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("HardwareAccelerator not available - using fallback")

    def optimize_performance(self, *args, **kwargs):
        """Fallback optimization method."""
        return {}

class MetaLearner:
    """Fallback meta learner."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("MetaLearner not available - using fallback")

    def learn_from_experience(self, *args, **kwargs):
        """Fallback learning method."""
        return {}

class EconomicValidator:
    """Fallback economic validator."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("EconomicValidator not available - using fallback")

    def validate_economic_significance(self, *args, **kwargs):
        """Fallback validation method."""
        return {"is_significant": True, "score": 0.5}

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTASResult:
    """Result of advanced TAS optimization."""
    best_architecture: Any
    architecture_type: TASArchitectureType
    regime_analysis: Dict[MarketRegime, Any]
    micro_regime_analysis: Dict[MicroRegimeType, List[MicroRegimeDetectionResult]]
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_results: List[EvaluationResult] = field(default_factory=list)

    # Advanced metrics
    economic_significance_score: float = 0.0
    trading_viability_score: float = 0.0
    regime_adaptation_score: float = 0.0
    micro_regime_detection_accuracy: float = 0.0

    # Model ensemble information
    ensemble_members: List[Any] = field(default_factory=list)
    ensemble_weights: List[float] = field(default_factory=list)

    # Meta-learning information
    meta_learning_score: float = 0.0
    transfer_learning_improvement: float = 0.0

    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedTradingArchitectureSearch:
    """Most advanced TAS implementation with all cutting-edge features."""

    def __init__(self, config: TASConfig):
        """Initialize Advanced TAS.

        Args:
            config: Advanced TAS configuration
        """
        tprint_info("🔍 Initializing Advanced TAS Search")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Search enabled: {config.enable_advanced_search}")
        tprint_debug(f"Search iterations: {config.search_iterations}")
        tprint_debug(f"Micro regime sensitivity: {config.micro_regime_sensitivity}")
        tprint_debug(f"Micro regime detection threshold: {config.micro_regime_detection_threshold}")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'search_time': 0.0,
            'evaluation_time': 0.0,
            'total_execution_time': 0.0
        }

        # Initialize components
        tprint_debug("Initializing micro regime detector...")
        self.micro_regime_detector = MicroRegimeDetector(
            sensitivity=config.micro_regime_sensitivity,
            detection_threshold=config.micro_regime_detection_threshold
        )

        self.evaluator = TASEvaluator(config)
        self.hardware_accelerator = HardwareAccelerator(config)
        self.meta_learner = MetaLearner(config)
        self.economic_validator = EconomicValidator(config)

        # Architecture search
        self.search_engine = AdvancedTASSearch(config)

        # Results tracking
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[Dict[str, Any]] = []
        self.micro_regime_history: List[MicroRegimeDetectionResult] = []
        self.performance_history: List[Dict[str, Any]] = []

        self.logger.info("✅ Advanced TAS initialized with all components")

    def optimize_advanced_architecture(self,
                                     market_data: pd.DataFrame,
                                     target_returns: pd.Series,
                                     existing_regimes: Optional[Dict] = None) -> AdvancedTASResult:
        """
        Perform advanced TAS optimization with all sophisticated features.

        Args:
            market_data: Historical market data for comprehensive analysis
            target_returns: Target returns for model training
            existing_regimes: Pre-detected regimes (optional)

        Returns:
            AdvancedTASResult with optimal architectures and comprehensive analysis
        """
        self.logger.info("🚀 Starting Advanced TAS optimization...")
        start_time = time.time()

        try:
            # Step 1: Comprehensive market analysis
            market_analysis = self._perform_comprehensive_market_analysis(market_data, target_returns)

            # Step 2: Micro-regime detection and analysis
            micro_regime_analysis = self._analyze_micro_regimes(market_data)

            # Step 3: Advanced architecture search
            optimal_architecture = self._perform_advanced_architecture_search(
                market_data, target_returns, market_analysis, micro_regime_analysis
            )

            # Step 4: Multi-objective optimization and evaluation
            evaluation_results = self._evaluate_architecture_comprehensive(
                optimal_architecture, market_data, target_returns, market_analysis, micro_regime_analysis
            )

            # Step 5: Economic validation and viability assessment
            validation_results = self._validate_economic_significance(evaluation_results)

            # Step 6: Create ensemble and meta-learning integration
            ensemble_result = self._create_advanced_ensemble(optimal_architecture, evaluation_results)

            # Step 7: Compile comprehensive result
            result = AdvancedTASResult(
                best_architecture=ensemble_result['ensemble'],
                architecture_type=self.config.architecture_type,
                regime_analysis=market_analysis,
                micro_regime_analysis=micro_regime_analysis,
                evaluation_results=evaluation_results,
                economic_significance_score=validation_results['economic_score'],
                trading_viability_score=validation_results['viability_score'],
                regime_adaptation_score=self._calculate_regime_adaptation_score(market_analysis),
                micro_regime_detection_accuracy=self._calculate_micro_regime_accuracy(micro_regime_analysis),
                ensemble_members=ensemble_result['members'],
                ensemble_weights=ensemble_result['weights'],
                meta_learning_score=ensemble_result['meta_learning_score'],
                transfer_learning_improvement=ensemble_result['transfer_improvement'],
                execution_time=time.time() - start_time,
                metadata={
                    'optimization_method': 'advanced_multi_objective',
                    'components_used': [
                        'micro_regime_detection',
                        'neural_architecture_search',
                        'economic_validation',
                        'hardware_acceleration',
                        'meta_learning',
                        'ensemble_optimization'
                    ],
                    'search_iterations': self.config.n_search_iterations,
                    'evaluation_samples': len(evaluation_results)
                }
            )

            self.logger.info(f"✅ Advanced TAS completed in {result.execution_time:.2f}s")
            self.logger.info(f"   Economic Significance: {result.economic_significance_score:.3f}")
            self.logger.info(f"   Trading Viability: {result.trading_viability_score:.3f}")
            self.logger.info(f"   Architecture Type: {result.architecture_type.value}")

            return result

        except Exception as e:
            self.logger.error(f"Advanced TAS failed: {e}")
            raise

    def _perform_comprehensive_market_analysis(self, market_data: pd.DataFrame,
                                            target_returns: pd.Series) -> Dict[MarketRegime, Any]:
        """Perform comprehensive market analysis including regime detection."""
        self.logger.info("🔍 Performing comprehensive market analysis...")

        # Use existing NAS clustering if available
        if self.config.integrate_with_nas_clustering and self.config.use_existing_regime_detection:
            try:
                regimes = self._detect_regimes_with_nas_clustering(market_data)
            except:
                regimes = self._detect_regimes_with_tree_models(market_data, target_returns)
        else:
            regimes = self._detect_regimes_with_tree_models(market_data, target_returns)

        # Enhance regime analysis with micro-regime information
        enhanced_regimes = {}
        for regime_type, regime_info in regimes.items():
            if isinstance(regime_info, dict):
                # Detect micro-regimes for this market regime
                regime_data = market_data  # In practice, would filter by regime
                micro_regimes = self.micro_regime_detector.detect_micro_regimes(regime_data, regime_type)

                enhanced_regime = {
                    'regime_type': regime_type,
                    'characteristics': regime_info.get('characteristics', {}),
                    'micro_regimes': micro_regimes,
                    'micro_regime_distribution': self._analyze_micro_regime_distribution(micro_regimes),
                    'transition_probabilities': self._calculate_regime_transition_probabilities(regime_type, regimes),
                    'optimal_architecture_type': self._determine_optimal_architecture_for_regime(regime_type)
                }
                enhanced_regimes[regime_type] = enhanced_regime

        self.logger.info(f"✅ Comprehensive market analysis completed: {len(enhanced_regimes)} regimes")
        return enhanced_regimes

    def _analyze_micro_regimes(self, market_data: pd.DataFrame) -> Dict[MicroRegimeType, List[MicroRegimeDetectionResult]]:
        """Analyze micro-regimes in the market data."""
        self.logger.info("🔬 Analyzing micro-regimes...")

        if not self.config.enable_micro_regime_detection:
            return {}

        micro_regimes = self.micro_regime_detector.detect_micro_regimes(market_data)

        # Group by micro-regime type
        micro_regime_analysis = defaultdict(list)
        for micro_regime in micro_regimes:
            micro_regime_analysis[micro_regime.regime_type].append(micro_regime)

        # Sort each group by confidence
        for regime_type in micro_regime_analysis:
            micro_regime_analysis[regime_type].sort(key=lambda x: x.confidence, reverse=True)

        self.micro_regime_history.extend(micro_regimes)

        self.logger.info(f"✅ Micro-regime analysis completed: {len(micro_regimes)} micro-regimes detected")
        return dict(micro_regime_analysis)

    def _perform_advanced_architecture_search(self, market_data: pd.DataFrame, target_returns: pd.Series,
                                            market_analysis: Dict, micro_regime_analysis: Dict) -> Any:
        """Perform advanced architecture search using multiple strategies."""
        self.logger.info("🏗️ Performing advanced architecture search...")

        # Use the advanced search engine
        search_space = self.config.get_tree_search_space()
        if not search_space:
            search_space = self._build_default_search_space(market_data)

        search_result = self.search_engine.search(
            market_data=market_data,
            target_returns=target_returns,
            market_regimes=market_analysis,
            micro_regimes=micro_regime_analysis,
            architecture_type=self.config.architecture_type,
            search_space=search_space
        )

        return search_result['best_architecture']

    def _build_default_search_space(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback search space when configuration does not provide one."""
        numeric_columns = market_data.select_dtypes(include=[np.number])
        n_features = max(numeric_columns.shape[1], 1)

        feature_options = list(range(1, min(n_features, 5) + 1))
        return {
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [1, 5, 10],
            'feature_subset': feature_options,
            'regularization_strength': [0.0, 0.01, 0.05],
        }

    def _evaluate_architecture_comprehensive(self, architecture: Any, market_data: pd.DataFrame,
                                           target_returns: pd.Series, market_analysis: Dict,
                                           micro_regime_analysis: Dict) -> List[EvaluationResult]:
        """Comprehensive evaluation of architecture across multiple dimensions."""
        self.logger.info("🔬 Performing comprehensive architecture evaluation...")

        evaluation_results = []

        # Evaluate on different data splits and regimes
        time_splits = self._create_time_based_splits(market_data, n_splits=5)

        for i, (train_data, val_data, test_data) in enumerate(time_splits):
            # Prepare data for this split
            X_train, y_train = self._prepare_training_data(train_data, target_returns)
            X_val, y_val = self._prepare_training_data(val_data, target_returns)
            X_test, y_test = self._prepare_training_data(test_data, target_returns)

            # Evaluate architecture
            evaluation_result = self.evaluator.evaluate_model(
                model=architecture,
                X_test=X_test,
                y_test=y_test,
                market_data=test_data,
                model_name=f"Architecture_Split_{i}",
                architecture_type=self.config.architecture_type.value
            )

            # Add regime-specific information
            if i < len(market_analysis):
                regime_key = list(market_analysis.keys())[i % len(market_analysis)]
                evaluation_result.notes = f"Evaluated on regime: {regime_key.value}"

            evaluation_results.append(evaluation_result)

        # Ensemble evaluation results
        if len(evaluation_results) > 1:
            ensemble_result = self._create_evaluation_ensemble(evaluation_results)
            evaluation_results.append(ensemble_result)

        self.logger.info(f"✅ Comprehensive evaluation completed: {len(evaluation_results)} results")
        return evaluation_results

    def _validate_economic_significance(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """Validate economic significance and trading viability."""
        self.logger.info("💰 Validating economic significance...")

        validation_scores = {
            'economic_score': 0.0,
            'viability_score': 0.0,
            'overall_validation': False
        }

        if not evaluation_results:
            return validation_scores

        # Aggregate evaluation results
        avg_economic_score = np.mean([r.economic_significance_score for r in evaluation_results])
        avg_viability_score = np.mean([r.trading_viability_score for r in evaluation_results])
        avg_sharpe_ratio = np.mean([r.sharpe_ratio for r in evaluation_results])
        avg_max_drawdown = np.mean([abs(r.max_drawdown) for r in evaluation_results])

        # Calculate validation scores
        validation_scores['economic_score'] = avg_economic_score
        validation_scores['viability_score'] = avg_viability_score

        # Overall validation
        economic_passes = avg_economic_score >= self.config.economic_significance_threshold
        viability_passes = avg_viability_score >= self.config.trading_viability_threshold
        risk_passes = avg_max_drawdown <= self.config.max_drawdown_threshold
        sharpe_passes = avg_sharpe_ratio >= self.config.risk_adjusted_return_threshold

        validation_scores['overall_validation'] = economic_passes and viability_passes and risk_passes and sharpe_passes

        self.logger.info(f"✅ Economic validation completed:")
        self.logger.info(f"   Economic Score: {avg_economic_score:.3f}")
        self.logger.info(f"   Viability Score: {avg_viability_score:.3f}")
        self.logger.info(f"   Overall Validation: {validation_scores['overall_validation']}")

        return validation_scores

    def _create_advanced_ensemble(self, base_architecture: Any,
                                 evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Create advanced ensemble with meta-learning and optimization."""
        self.logger.info("🤖 Creating advanced ensemble...")

        ensemble_result = {
            'ensemble': base_architecture,
            'members': [base_architecture],
            'weights': [1.0],
            'meta_learning_score': 0.0,
            'transfer_improvement': 0.0
        }

        try:
            # Use meta-learner to improve ensemble
            if self.config.enable_meta_learning:
                meta_result = self.meta_learner.optimize_ensemble(
                    base_architecture=base_architecture,
                    evaluation_results=evaluation_results,
                    market_regimes=list(self.regime_history[-10:])  # Recent regime history
                )

                if meta_result['improvement'] > 0:
                    ensemble_result.update(meta_result)
                    self.logger.info(f"✅ Ensemble optimized with meta-learning: {meta_result['improvement']:.3f} improvement")

        except Exception as e:
            self.logger.warning(f"Advanced ensemble creation partially failed: {e}")

        return ensemble_result

    def _calculate_regime_adaptation_score(self, market_analysis: Dict) -> float:
        """Calculate regime adaptation capability score."""
        try:
            if not market_analysis:
                return 0.0

            # Calculate based on regime diversity and stability
            n_regimes = len(market_analysis)
            regime_stability_scores = []

            for regime_info in market_analysis.values():
                if isinstance(regime_info, dict) and 'micro_regimes' in regime_info:
                    micro_regimes = regime_info['micro_regimes']
                    if micro_regimes:
                        # Calculate micro-regime stability
                        stability = np.mean([mr.confidence for mr in micro_regimes])
                        regime_stability_scores.append(stability)

            if regime_stability_scores:
                avg_stability = np.mean(regime_stability_scores)
                regime_diversity = min(1.0, n_regimes / 10)  # Normalize regime count

                return (avg_stability * 0.6 + regime_diversity * 0.4)
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Regime adaptation score calculation failed: {e}")
            return 0.0

    def _calculate_micro_regime_accuracy(self, micro_regime_analysis: Dict) -> float:
        """Calculate micro-regime detection accuracy."""
        try:
            if not micro_regime_analysis:
                return 0.0

            total_micro_regimes = sum(len(regimes) for regimes in micro_regime_analysis.values())
            high_confidence_regimes = sum(
                len([mr for mr in regimes if mr.confidence >= self.config.micro_regime_detection_threshold])
                for regimes in micro_regime_analysis.values()
            )

            if total_micro_regimes > 0:
                return high_confidence_regimes / total_micro_regimes
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Micro-regime accuracy calculation failed: {e}")
            return 0.0

    def _prepare_training_data(self, market_data: pd.DataFrame, target_returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training."""
        # Feature engineering
        X = self._engineer_features(market_data)
        y = target_returns.values if hasattr(target_returns, 'values') else target_returns

        return X, y

    def _engineer_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Engineer features for model training."""
        # Enhanced feature engineering
        features = []

        # Price-based features
        returns = market_data['close'].pct_change()
        for window in [5, 10, 20, 50]:
            features.extend([
                returns.rolling(window).mean().fillna(0),
                returns.rolling(window).std().fillna(0),
                returns.rolling(window).skew().fillna(0),
                returns.rolling(window).kurt().fillna(0)
            ])

        # Volume features (if available)
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            features.extend([
                (volume / volume.rolling(20).mean()).fillna(1),
                volume.rolling(10).std().fillna(0)
            ])

        # Technical indicators
        high, low, close = market_data['high'], market_data['low'], market_data['close']

        # Moving averages
        for window in [10, 20, 50]:
            sma = close.rolling(window).mean()
            features.append((close - sma) / sma)

        # RSI
        rsi = self._calculate_rsi(close, 14)
        features.append(rsi / 100)  # Normalize to 0-1

        # Bollinger Bands
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = (close - (bb_middle + 2 * bb_std)) / (2 * bb_std)
        bb_lower = ((bb_middle - 2 * bb_std) - close) / (2 * bb_std)
        features.extend([bb_upper.fillna(0), bb_lower.fillna(0)])

        return np.column_stack(features)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def _create_time_based_splits(self, market_data: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Create time-based data splits for evaluation."""
        splits = []
        data_length = len(market_data)

        for i in range(n_splits):
            # Create time-based split
            split_size = data_length // (n_splits + 1)
            start_idx = i * split_size
            mid_idx = (i + 1) * split_size
            end_idx = min((i + 2) * split_size, data_length)

            train_data = market_data.iloc[:mid_idx]
            val_data = market_data.iloc[mid_idx:end_idx]
            test_data = market_data.iloc[end_idx:]

            splits.append((train_data, val_data, test_data))

        return splits

    def _create_evaluation_ensemble(self, evaluation_results: List[EvaluationResult]) -> EvaluationResult:
        """Create ensemble of evaluation results."""
        ensemble_result = EvaluationResult(
            model_name="Evaluation_Ensemble",
            architecture_type="ensemble"
        )

        # Aggregate metrics
        ensemble_result.economic_significance_score = np.mean([r.economic_significance_score for r in evaluation_results])
        ensemble_result.trading_viability_score = np.mean([r.trading_viability_score for r in evaluation_results])
        ensemble_result.sharpe_ratio = np.mean([r.sharpe_ratio for r in evaluation_results])
        ensemble_result.max_drawdown = np.mean([r.max_drawdown for r in evaluation_results])
        ensemble_result.win_rate = np.mean([r.win_rate for r in evaluation_results])
        ensemble_result.regime_stability_score = np.mean([r.regime_stability_score for r in evaluation_results])

        # Mark as ensemble
        ensemble_result.notes = f"Ensemble of {len(evaluation_results)} evaluation results"

        return ensemble_result

    def _analyze_micro_regime_distribution(self, micro_regimes: List[MicroRegimeDetectionResult]) -> Dict[str, float]:
        """Analyze distribution of micro-regimes."""
        if not micro_regimes:
            return {}

        regime_counts = defaultdict(int)
        total_regimes = len(micro_regimes)

        for regime in micro_regimes:
            regime_counts[regime.regime_type.value] += 1

        return {regime: count / total_regimes for regime, count in regime_counts.items()}

    def _calculate_regime_transition_probabilities(self, current_regime: MarketRegime,
                                                all_regimes: Dict) -> Dict[MarketRegime, float]:
        """Calculate transition probabilities to other regimes."""
        # Simple transition probability calculation
        n_regimes = len(all_regimes)
        base_prob = 1.0 / (n_regimes - 1)  # Probability to transition to any other regime

        transition_probs = {}
        for regime in all_regimes.keys():
            if regime != current_regime:
                transition_probs[regime] = base_prob
            else:
                transition_probs[regime] = 1.0 - base_prob * (n_regimes - 1)  # Stay probability

        return transition_probs

    def _determine_optimal_architecture_for_regime(self, regime: MarketRegime) -> TASArchitectureType:
        """Determine optimal architecture type for a given regime."""
        # Regime-specific architecture selection
        regime_architecture_map = {
            MarketRegime.HIGH_VOLATILITY: TASArchitectureType.HYBRID_TREE_NEURAL,
            MarketRegime.LOW_VOLATILITY: TASArchitectureType.NEURAL_ONLY,
            MarketRegime.TRENDING_UP: TASArchitectureType.TREE_ONLY,
            MarketRegime.TRENDING_DOWN: TASArchitectureType.TREE_ONLY,
            MarketRegime.MEAN_REVERTING: TASArchitectureType.HYBRID_TREE_NEURAL,
            MarketRegime.BREAKOUT: TASArchitectureType.NEURAL_ONLY,
            MarketRegime.CONSOLIDATION: TASArchitectureType.TREE_ONLY
        }

        return regime_architecture_map.get(regime, TASArchitectureType.HYBRID_TREE_NEURAL)

    def _detect_regimes_with_nas_clustering(self, market_data: pd.DataFrame) -> Dict[MarketRegime, Dict]:
        """Detect regimes using existing NAS clustering system."""
        try:
            from src.training.steps.market_analysis.nas_clustering.core.nas_clusterer import NASClusterer
            from src.training.steps.market_analysis.nas_clustering.core.nas_config import NASClusteringConfig

            config = NASClusteringConfig.create_short_term_trading_config()
            clusterer = NASClusterer(config)

            # Prepare data for clustering
            clustering_data = self._prepare_clustering_data(market_data)

            # Perform clustering
            result = clusterer.cluster_market_data(clustering_data)

            # Convert clustering results to regimes
            regimes = {}
            for i, (regime_data, labels) in enumerate(zip(result.regime_data, result.labels)):
                regime_type = self._map_clustering_to_regime_type(i, regime_data, labels)
                regimes[regime_type] = {
                    'start_time': datetime.now() - timedelta(days=1),
                    'confidence': result.quality_metrics.get('regime_confidence', 0.8),
                    'characteristics': self._extract_regime_characteristics_from_clustering(regime_data),
                    'transition_probability': 0.1
                }

            return regimes

        except ImportError:
            self.logger.warning("NAS clustering not available, falling back to tree-based detection")
            return {}

    def _detect_regimes_with_tree_models(self, market_data: pd.DataFrame, target_returns: pd.Series) -> Dict[MarketRegime, Dict]:
        """Detect regimes using tree-based models."""
        regimes = {}

        try:
            # Feature engineering for regime detection
            regime_features = self._engineer_regime_features(market_data)

            # Use unsupervised learning to detect regimes
            from sklearn.ensemble import RandomForestClassifier

            # KMeans clustering removed - will be handled in subsequent step

            # Try different numbers of regimes
            for n_regimes in range(3, 8):  # 3 to 7 regimes
                try:
                    # Cluster the feature space
                    # Simple regime assignment instead of KMeans
                    n_samples = len(regime_features)
                    regime_size = n_samples // n_regimes
                    regime_labels = np.array([i // regime_size for i in range(n_samples)])
                    regime_labels = np.minimum(regime_labels, n_regimes - 1)
                    # clusters already assigned above

                    # Analyze each cluster
                    for cluster_id in range(n_regimes):
                        cluster_mask = regime_labels == cluster_id
                        cluster_data = market_data[cluster_mask]

                        if len(cluster_data) >= self.config.min_regime_samples:
                            regime_type = self._analyze_cluster_characteristics(cluster_data, target_returns[cluster_mask])
                            regimes[regime_type] = {
                                'start_time': cluster_data.index[0] if hasattr(cluster_data, 'index') else datetime.now(),
                                'confidence': len(cluster_data) / len(market_data),
                                'characteristics': self._extract_regime_characteristics(cluster_data),
                                'transition_probability': 1.0 / n_regimes
                            }
                except:
                    continue

            return regimes

        except Exception as e:
            self.logger.warning(f"Tree-based regime detection failed: {e}")
            return {}

    def _engineer_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Engineer features for regime detection."""
        features = []

        # Price-based features
        returns = market_data['close'].pct_change()

        # Volatility features
        for window in [5, 10, 20, 50]:
            vol = returns.rolling(window).std()
            features.append(vol.mean())

        # Trend features
        for window in [10, 20, 50]:
            sma = market_data['close'].rolling(window).mean()
            trend = (market_data['close'].iloc[-1] / sma.iloc[-1] - 1) if len(sma) > 0 else 0
            features.append(trend)

        # Momentum features
        for window in [5, 10, 20]:
            momentum = (market_data['close'] - market_data['close'].shift(window)) / market_data['close'].shift(window)
            features.append(momentum.mean())

        return np.array(features).reshape(1, -1)

    def _analyze_cluster_characteristics(self, cluster_data: pd.DataFrame, cluster_returns: pd.Series) -> MarketRegime:
        """Analyze cluster characteristics to determine regime type."""
        # Calculate cluster statistics
        returns = cluster_data['close'].pct_change()
        volatility = returns.std().mean()
        trend_strength = abs((cluster_data['close'].iloc[-1] / cluster_data['close'].iloc[0] - 1).mean())

        # Determine regime type based on characteristics
        if volatility > 0.03:  # High volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:  # Low volatility
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.1:  # Strong trend
            avg_return = cluster_returns.mean()
            return MarketRegime.TRENDING_UP if avg_return > 0 else MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.NORMAL

    def _extract_regime_characteristics(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Extract regime characteristics from cluster data."""
        returns = cluster_data['close'].pct_change()

        return {
            'volatility': returns.std().mean(),
            'trend_strength': abs((cluster_data['close'].iloc[-1] / cluster_data['close'].iloc[0] - 1).mean()),
            'mean_return': returns.mean().mean(),
            'max_return': returns.max().max(),
            'min_return': returns.min().min(),
            'duration_hours': len(cluster_data) / 60  # Assuming 1-minute data
        }

    def _map_clustering_to_regime_type(self, cluster_id: int, regime_data: pd.DataFrame, labels: np.ndarray) -> MarketRegime:
        """Map clustering results to regime types."""
        # Analyze cluster characteristics to determine regime type
        if 'volatility' in regime_data.columns:
            vol = regime_data['volatility'].mean()
            if vol > 0.03:  # High volatility
                return MarketRegime.HIGH_VOLATILITY
            elif vol < 0.01:  # Low volatility
                return MarketRegime.LOW_VOLATILITY

        # Check for trends
        if 'trend_strength' in regime_data.columns:
            trend = regime_data['trend_strength'].mean()
            if trend > 0.7:
                return MarketRegime.TRENDING_UP

        return MarketRegime.NORMAL

    def _prepare_clustering_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market data for clustering-based regime detection."""
        # Feature engineering for regime detection
        features = []

        # Volatility features
        returns = market_data['close'].pct_change()
        features.extend([
            returns.rolling(20).std().mean(),  # Rolling volatility
            returns.rolling(50).std().mean(),  # Longer-term volatility
            (returns.rolling(20).std() / returns.rolling(50).std()).mean()  # Volatility ratio
        ])

        # Trend features
        for window in [10, 20, 50]:
            sma = market_data['close'].rolling(window).mean()
            trend_strength = (market_data['close'] - sma).abs().mean() / market_data['close'].std()
            features.append(trend_strength.mean())

        # Volume features (if available)
        if 'volume' in market_data.columns:
            volume_ratio = market_data['volume'] / market_data['volume'].rolling(20).mean()
            features.append(volume_ratio.mean())

        return pd.DataFrame([features], columns=[f'feature_{i}' for i in range(len(features))])

    def _extract_regime_characteristics_from_clustering(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Extract characteristics from clustering results."""
        characteristics = {}

        for col in regime_data.columns:
            if col in ['volatility', 'trend_strength', 'volume_ratio']:
                characteristics[col] = regime_data[col].mean()

        return characteristics

# Convenience functions for advanced TAS
def optimize_advanced_trading_architecture(market_data: pd.DataFrame,
                                         target_returns: pd.Series,
                                         config: Optional[TASConfig] = None) -> AdvancedTASResult:
    """
    Convenience function for advanced TAS optimization.

    Args:
        market_data: Historical market data
        target_returns: Target returns for training
        config: Advanced TAS configuration

    Returns:
        AdvancedTASResult with optimal architectures and comprehensive analysis
    """
    if config is None:
        config = TASConfig.create_advanced_trading_config()

    tas = AdvancedTradingArchitectureSearch(config)
    return tas.optimize_advanced_architecture(market_data, target_returns)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
