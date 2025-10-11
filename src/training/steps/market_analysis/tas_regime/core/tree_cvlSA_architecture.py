"""
Tree-based CVLSA (Cascade Variable Length Selection Architecture) for TAS

This module implements a sophisticated tree-based architecture leveraging the existing
hierarchical ensemble capabilities with advanced cascade and variable-length selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from .tas_config import TASConfig, TASArchitectureType, TradingObjective, MarketRegime, MicroRegimeType
from ..components.micro_regime_detector import MicroRegimeDetector, MicroRegimeDetectionResult
from ..evaluation.tas_evaluator import TASEvaluator, EvaluationResult
from src.utils.ml_common.optimization.tree_architecture_search import TreeArchitectureSearch, TreeArchitectureConfig, TreeArchitectureCandidate

logger = logging.getLogger(__name__)


@dataclass
class CVLSAResult:
    """Result of CVLSA optimization."""
    best_architecture: TreeArchitectureCandidate
    architecture_type: str = "CVLSA_Tree"
    cascade_levels: List[Dict[str, Any]] = field(default_factory=list)
    variable_selection_config: Dict[str, Any] = field(default_factory=dict)
    regime_analysis: Dict[MarketRegime, Any] = field(default_factory=dict)
    micro_regime_analysis: Dict[MicroRegimeType, List[MicroRegimeDetectionResult]] = field(default_factory=dict)

    # Performance metrics
    economic_significance_score: float = 0.0
    trading_viability_score: float = 0.0
    cascade_efficiency: float = 0.0
    variable_selection_accuracy: float = 0.0

    # Ensemble information
    ensemble_members: List[TreeArchitectureCandidate] = field(default_factory=list)
    ensemble_weights: List[float] = field(default_factory=list)

    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TreeCVLSASearch:
    """Advanced tree-based CVLSA implementation."""

    def __init__(self, config: TASConfig):
        """Initialize Tree CVLSA.

        Args:
            config: TAS configuration
        """
        tprint_info("🌲 Initializing Tree CVLSA Architecture")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        tprint_info("🔧 Initializing components...")
        tprint_debug("🔍 Creating micro regime detector...")
        self.micro_regime_detector = MicroRegimeDetector(
            sensitivity=config.micro_regime_sensitivity,
            detection_threshold=config.micro_regime_detection_threshold
        )

        tprint_debug("📊 Creating TAS evaluator...")
        self.evaluator = TASEvaluator(config)
        tprint_debug("🌳 Creating base TAS search...")
        self.base_tas = TreeArchitectureSearch(config.get_tree_search_space())

        # CVLSA-specific parameters
        tprint_debug("⚙️ Setting CVLSA-specific parameters...")
        self.cascade_depth = 3
        self.variable_selection_methods = [
            'variance_threshold',
            'mutual_information',
            'tree_importance',
            'correlation_filter',
            'recursive_elimination'
        ]

        # Results tracking
        tprint_debug("📊 Initializing results tracking...")
        self.cascade_history: List[Dict[str, Any]] = []
        self.variable_selection_history: List[Dict[str, Any]] = []

        tprint_success("✅ Tree CVLSA initialized with cascade architecture")
        self.logger.info("✅ Tree CVLSA initialized with cascade architecture")

    def optimize_cvlSA_architecture(self,
                                   market_data: pd.DataFrame,
                                   target_returns: pd.Series,
                                   existing_regimes: Optional[Dict] = None) -> CVLSAResult:
        """
        Perform CVLSA optimization with advanced tree-based cascade architecture.

        Args:
            market_data: Historical market data
            target_returns: Target returns for training
            existing_regimes: Pre-detected regimes (optional)

        Returns:
            CVLSAResult with optimal CVLSA architecture
        """
        tprint_info("🌲 Starting Tree CVLSA optimization...")
        self.logger.info("🌲 Starting Tree CVLSA optimization...")
        start_time = time.time()

        try:
            # Step 1: Comprehensive market analysis
            tprint_info("📊 Step 1: Performing comprehensive market analysis...")
            market_analysis = self._perform_comprehensive_market_analysis(market_data, target_returns)
            tprint_success("✅ Market analysis completed")

            # Step 2: Micro-regime detection
            tprint_info("🔍 Step 2: Analyzing micro-regimes...")
            micro_regime_analysis = self._analyze_micro_regimes(market_data)
            tprint_success("✅ Micro-regime analysis completed")

            # Step 3: Feature engineering with variable selection
            tprint_info("🔧 Step 3: Performing advanced feature engineering...")
            engineered_features = self._perform_advanced_feature_engineering(market_data, target_returns)
            tprint_success("✅ Feature engineering completed")

            # Step 4: CVLSA cascade optimization
            tprint_info("🌊 Step 4: Optimizing cascade architecture...")
            cvlsa_architecture = self._optimize_cascade_architecture(
                engineered_features, target_returns, market_analysis, micro_regime_analysis
            )
            tprint_success("✅ Cascade optimization completed")

            # Step 5: Variable selection optimization
            tprint_info("🎯 Step 5: Optimizing variable selection...")
            variable_selection_config = self._optimize_variable_selection(
                engineered_features, target_returns, cvlsa_architecture
            )
            tprint_success("✅ Variable selection optimization completed")

            # Step 6: Comprehensive evaluation
            tprint_info("📈 Step 6: Performing comprehensive evaluation...")
            evaluation_results = self._evaluate_cvlSA_architecture(
                cvlsa_architecture, engineered_features, target_returns, market_analysis, micro_regime_analysis
            )
            tprint_success("✅ Comprehensive evaluation completed")

            # Step 7: Economic validation
            tprint_info("💰 Step 7: Performing economic validation...")
            validation_results = self._validate_cvlSA_performance(evaluation_results)
            tprint_success("✅ Economic validation completed")

            # Step 8: Create final CVLSA result
            tprint_info("📋 Step 8: Creating final CVLSA result...")
            result = CVLSAResult(
                best_architecture=cvlsa_architecture,
                cascade_levels=self._create_cascade_levels(cvlsa_architecture),
                variable_selection_config=variable_selection_config,
                regime_analysis=market_analysis,
                micro_regime_analysis=micro_regime_analysis,
                economic_significance_score=validation_results['economic_score'],
                trading_viability_score=validation_results['viability_score'],
                cascade_efficiency=self._calculate_cascade_efficiency(cvlsa_architecture),
                variable_selection_accuracy=self._calculate_variable_selection_accuracy(variable_selection_config),
                ensemble_members=[cvlsa_architecture],
                ensemble_weights=[1.0],
                execution_time=time.time() - start_time,
                metadata={
                    'optimization_method': 'CVLSA_tree_cascade',
                    'cascade_depth': self.cascade_depth,
                    'variable_selection_methods': self.variable_selection_methods,
                    'components_used': [
                        'micro_regime_detection',
                        'advanced_feature_engineering',
                        'cascade_architecture',
                        'variable_selection',
                        'economic_validation'
                    ]
                }
            )

            tprint_success(f"✅ Tree CVLSA completed in {result.execution_time:.2f}s")
            tprint_info(f"   Cascade Levels: {len(result.cascade_levels)}")
            tprint_info(f"   Economic Significance: {result.economic_significance_score:.3f}")
            tprint_info(f"   Cascade Efficiency: {result.cascade_efficiency:.3f}")
            self.logger.info(f"✅ Tree CVLSA completed in {result.execution_time:.2f}s")
            self.logger.info(f"   Cascade Levels: {len(result.cascade_levels)}")
            self.logger.info(f"   Economic Significance: {result.economic_significance_score:.3f}")
            self.logger.info(f"   Cascade Efficiency: {result.cascade_efficiency:.3f}")

            return result

        except Exception as e:
            tprint_error(f"❌ Tree CVLSA failed: {e}")
            self.logger.error(f"Tree CVLSA failed: {e}")
            raise

    def _perform_comprehensive_market_analysis(self, market_data: pd.DataFrame,
                                            target_returns: pd.Series) -> Dict[MarketRegime, Any]:
        """Perform comprehensive market analysis for CVLSA."""
        self.logger.info("🔍 Performing comprehensive market analysis for CVLSA...")

        # Use existing NAS clustering if available
        if self.config.integrate_with_nas_clustering and self.config.use_existing_regime_detection:
            try:
                regimes = self._detect_regimes_with_nas_clustering(market_data)
            except:
                regimes = self._detect_regimes_with_tree_models(market_data, target_returns)
        else:
            regimes = self._detect_regimes_with_tree_models(market_data, target_returns)

        # Enhance with CVLSA-specific analysis
        enhanced_regimes = {}
        for regime_type, regime_info in regimes.items():
            if isinstance(regime_info, dict):
                # Detect micro-regimes for this market regime
                regime_data = market_data
                micro_regimes = self.micro_regime_detector.detect_micro_regimes(regime_data, regime_type)

                # Calculate optimal cascade depth for this regime
                optimal_cascade_depth = self._calculate_optimal_cascade_depth(regime_type, micro_regimes)

                enhanced_regime = {
                    'regime_type': regime_type,
                    'characteristics': regime_info.get('characteristics', {}),
                    'micro_regimes': micro_regimes,
                    'micro_regime_distribution': self._analyze_micro_regime_distribution(micro_regimes),
                    'optimal_cascade_depth': optimal_cascade_depth,
                    'variable_selection_priority': self._calculate_variable_selection_priority(regime_type),
                    'cvlsa_architecture_type': self._determine_cvlSA_architecture_for_regime(regime_type)
                }
                enhanced_regimes[regime_type] = enhanced_regime

        self.logger.info(f"✅ CVLSA market analysis completed: {len(enhanced_regimes)} regimes")
        return enhanced_regimes

    def _perform_advanced_feature_engineering(self, market_data: pd.DataFrame,
                                            target_returns: pd.Series) -> Dict[str, Any]:
        """Perform advanced feature engineering with multiple selection methods."""
        self.logger.info("🔧 Performing advanced feature engineering for CVLSA...")

        # Base feature engineering
        base_features = self._engineer_base_features(market_data)

        # Apply variable selection methods
        selected_features = {}
        for method in self.variable_selection_methods:
            selected = self._apply_variable_selection_method(base_features, target_returns, method)
            selected_features[method] = selected

        # Create ensemble feature set
        ensemble_features = self._create_feature_ensemble(selected_features)

        return {
            'base_features': base_features,
            'selected_features': selected_features,
            'ensemble_features': ensemble_features,
            'feature_importance_scores': self._calculate_feature_importance_scores(ensemble_features, target_returns)
        }

    def _optimize_cascade_architecture(self, engineered_features: Dict[str, Any],
                                     target_returns: pd.Series,
                                     market_analysis: Dict,
                                     micro_regime_analysis: Dict) -> TreeArchitectureCandidate:
        """Optimize cascade architecture using hierarchical ensemble approach."""
        self.logger.info("🏗️ Optimizing CVLSA cascade architecture...")

        # Use the existing hierarchical search capability
        X = engineered_features['ensemble_features']
        y = target_returns.values if hasattr(target_returns, 'values') else target_returns

        # Create cascade-specific configuration
        cascade_config = TreeArchitectureConfig(
            min_depth=3,
            max_depth=12,
            min_trees=100,
            max_trees=500,
            objectives=['accuracy', 'efficiency', 'interpretability'],
            objective_weights=[0.4, 0.3, 0.3],
            enable_multi_fidelity=True,
            low_fidelity_fraction=0.5
        )

        # Perform hierarchical search
        cascade_tas = TreeArchitectureSearch(cascade_config)
        best_architecture = cascade_tas.search(X, y, search_method="hierarchical")

        # Optimize cascade structure
        optimized_architecture = self._optimize_cascade_structure(
            best_architecture, X, y, market_analysis
        )

        return optimized_architecture

    def _optimize_variable_selection(self, engineered_features: Dict[str, Any],
                                   target_returns: pd.Series,
                                   cascade_architecture: TreeArchitectureCandidate) -> Dict[str, Any]:
        """Optimize variable selection for CVLSA."""
        self.logger.info("🎯 Optimizing variable selection for CVLSA...")

        variable_selection_config = {
            'selected_methods': [],
            'feature_importance_threshold': 0.1,
            'redundancy_threshold': 0.8,
            'optimal_feature_count': 0,
            'selection_accuracy': 0.0
        }

        try:
            X = engineered_features['base_features']
            y = target_returns.values if hasattr(target_returns, 'values') else target_returns

            # Test each variable selection method
            method_performance = {}
            for method in self.variable_selection_methods:
                selected_X = engineered_features['selected_features'][method]

                # Evaluate performance with cascade architecture
                performance = self._evaluate_variable_selection_method(
                    selected_X, y, cascade_architecture, method
                )

                method_performance[method] = performance

            # Select best methods
            sorted_methods = sorted(method_performance.items(), key=lambda x: x[1]['score'], reverse=True)
            variable_selection_config['selected_methods'] = [method for method, _ in sorted_methods[:3]]

            # Optimize feature count
            optimal_count = self._optimize_feature_count(X, y, cascade_architecture, variable_selection_config)
            variable_selection_config['optimal_feature_count'] = optimal_count

            # Calculate selection accuracy
            variable_selection_config['selection_accuracy'] = np.mean([
                method_performance[method]['accuracy'] for method in variable_selection_config['selected_methods']
            ])

            self.logger.info(f"✅ Variable selection optimized: {len(variable_selection_config['selected_methods'])} methods selected")

        except Exception as e:
            self.logger.warning(f"Variable selection optimization failed: {e}")

        return variable_selection_config

    def _evaluate_cvlSA_architecture(self, architecture: TreeArchitectureCandidate,
                                   engineered_features: Dict[str, Any],
                                   target_returns: pd.Series,
                                   market_analysis: Dict,
                                   micro_regime_analysis: Dict) -> List[EvaluationResult]:
        """Evaluate CVLSA architecture comprehensively."""
        self.logger.info("🔬 Evaluating CVLSA architecture...")

        evaluation_results = []

        # Time-based evaluation
        time_splits = self._create_time_based_splits(
            pd.DataFrame(engineered_features['ensemble_features']),
            n_splits=5
        )

        for i, (X_train, X_val, X_test) in enumerate(time_splits):
            y_train = target_returns.iloc[:len(X_train)]
            y_val = target_returns.iloc[len(X_train):len(X_train)+len(X_val)]
            y_test = target_returns.iloc[-len(X_test):]

            # Evaluate architecture
            evaluation_result = self.evaluator.evaluate_model(
                model=architecture,
                X_test=X_test.values,
                y_test=y_test.values,
                market_data=pd.DataFrame(X_test),
                model_name=f"CVLSA_Architecture_Split_{i}",
                architecture_type="CVLSA_Tree"
            )

            # Add CVLSA-specific metrics
            evaluation_result.notes = f"CVLSA cascade evaluation - Split {i}"
            evaluation_results.append(evaluation_result)

        self.logger.info(f"✅ CVLSA evaluation completed: {len(evaluation_results)} results")
        return evaluation_results

    def _validate_cvlSA_performance(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """Validate CVLSA performance with economic significance."""
        self.logger.info("💰 Validating CVLSA economic significance...")

        validation_scores = {
            'economic_score': 0.0,
            'viability_score': 0.0,
            'cascade_efficiency': 0.0,
            'overall_validation': False
        }

        if not evaluation_results:
            return validation_scores

        # Aggregate evaluation results
        avg_economic_score = np.mean([r.economic_significance_score for r in evaluation_results])
        avg_viability_score = np.mean([r.trading_viability_score for r in evaluation_results])
        avg_sharpe_ratio = np.mean([r.sharpe_ratio for r in evaluation_results])
        avg_max_drawdown = np.mean([abs(r.max_drawdown) for r in evaluation_results])

        # Calculate CVLSA-specific metrics
        cascade_efficiency = self._calculate_cascade_efficiency_from_results(evaluation_results)

        # Validation scores
        validation_scores['economic_score'] = avg_economic_score
        validation_scores['viability_score'] = avg_viability_score
        validation_scores['cascade_efficiency'] = cascade_efficiency

        # Overall validation with CVLSA-specific criteria
        economic_passes = avg_economic_score >= self.config.economic_significance_threshold
        viability_passes = avg_viability_score >= self.config.trading_viability_threshold
        risk_passes = avg_max_drawdown <= self.config.max_drawdown_threshold
        sharpe_passes = avg_sharpe_ratio >= self.config.risk_adjusted_return_threshold
        cascade_passes = cascade_efficiency >= 0.7  # Minimum cascade efficiency

        validation_scores['overall_validation'] = (
            economic_passes and viability_passes and risk_passes and sharpe_passes and cascade_passes
        )

        self.logger.info(f"✅ CVLSA validation completed:")
        self.logger.info(f"   Economic Score: {avg_economic_score:.3f}")
        self.logger.info(f"   Cascade Efficiency: {cascade_efficiency:.3f}")
        self.logger.info(f"   Overall Validation: {validation_scores['overall_validation']}")

        return validation_scores

    def _engineer_base_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Engineer base features for CVLSA."""
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

        # Volume features
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
        features.append(rsi / 100)

        # Bollinger Bands
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = (close - (bb_middle + 2 * bb_std)) / (2 * bb_std)
        bb_lower = ((bb_middle - 2 * bb_std) - close) / (2 * bb_std)
        features.extend([bb_upper.fillna(0), bb_lower.fillna(0)])

        return np.column_stack(features)

    def _apply_variable_selection_method(self, features: np.ndarray, target: pd.Series,
                                       method: str) -> np.ndarray:
        """Apply specific variable selection method."""
        if method == 'variance_threshold':
            return self._variance_threshold_selection(features, threshold=0.01)
        elif method == 'mutual_information':
            return self._mutual_information_selection(features, target)
        elif method == 'tree_importance':
            return self._tree_importance_selection(features, target)
        elif method == 'correlation_filter':
            return self._correlation_filter_selection(features, threshold=0.8)
        elif method == 'recursive_elimination':
            return self._recursive_elimination_selection(features, target)
        else:
            return features

    def _variance_threshold_selection(self, features: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Apply variance threshold selection."""
        variances = np.var(features, axis=0)
        selected_mask = variances >= threshold
        return features[:, selected_mask]

    def _mutual_information_selection(self, features: np.ndarray, target: pd.Series) -> np.ndarray:
        """Apply mutual information-based selection."""
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

        if len(np.unique(target)) <= 10:
            mi_scores = mutual_info_classif(features, target)
        else:
            mi_scores = mutual_info_regression(features, target)

        # Select top 80% of features by mutual information
        n_features = int(len(mi_scores) * 0.8)
        top_indices = np.argsort(mi_scores)[-n_features:]
        return features[:, top_indices]

    def _tree_importance_selection(self, features: np.ndarray, target: pd.Series) -> np.ndarray:
        """Apply tree importance-based selection."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        if len(np.unique(target)) <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(features, target)

        # Select features with importance > threshold
        importance_mask = model.feature_importances_ > 0.01
        return features[:, importance_mask]

    def _correlation_filter_selection(self, features: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Apply correlation-based filtering."""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features.T)

        # Find highly correlated features
        n_features = features.shape[1]
        to_remove = set()

        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    to_remove.add(j)

        # Keep non-correlated features
        selected_mask = np.ones(n_features, dtype=bool)
        selected_mask[list(to_remove)] = False
        return features[:, selected_mask]

    def _recursive_elimination_selection(self, features: np.ndarray, target: pd.Series) -> np.ndarray:
        """Apply recursive feature elimination."""
        from sklearn.feature_selection import RFE

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

        if len(np.unique(target)) <= 10:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)

        # Select 80% of features
        n_features_to_select = int(features.shape[1] * 0.8)

        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(features, target)

        return features[:, rfe.support_]

    def _create_feature_ensemble(self, selected_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Create ensemble feature set from multiple selection methods."""
        if not selected_features:
            return np.array([])

        # Find common features across methods
        feature_sets = list(selected_features.values())
        if not feature_sets:
            return np.array([])

        # Use intersection of features from different methods
        common_features = feature_sets[0]
        for feature_set in feature_sets[1:]:
            common_features = np.intersect1d(
                np.arange(common_features.shape[1]),
                np.arange(feature_set.shape[1])
            )

        # If no common features, use all features from best method
        if len(common_features) == 0:
            best_method = max(selected_features.keys(),
                            key=lambda x: selected_features[x].shape[1])
            return selected_features[best_method]

        # Return common features (assuming first feature set has all features)
        return feature_sets[0][:, common_features]

    def _calculate_feature_importance_scores(self, features: np.ndarray, target: pd.Series) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:

            if len(np.unique(target)) <= 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(features, target)

            # Create feature importance dictionary
            importance_dict = {}
            for i, importance in enumerate(model.feature_importances_):
                importance_dict[f'feature_{i}'] = importance

            return importance_dict

        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}

    def _optimize_cascade_structure(self, base_architecture: TreeArchitectureCandidate,
                                  X: np.ndarray, y: np.ndarray,
                                  market_analysis: Dict) -> TreeArchitectureCandidate:
        """Optimize cascade structure for CVLSA."""
        self.logger.info("🔧 Optimizing cascade structure...")

        # Create cascade-specific architecture
        cascade_architecture = TreeArchitectureCandidate(
            n_trees=base_architecture.n_trees,
            max_depth=base_architecture.max_depth,
            min_samples_split=base_architecture.min_samples_split,
            min_samples_leaf=base_architecture.min_samples_leaf,
            max_features=base_architecture.max_features,
            splitting_strategy=base_architecture.splitting_strategy,
            search_method="CVLSA_cascade"
        )

        # Set hierarchical structure
        cascade_architecture.is_hierarchical = True
        cascade_architecture.ensemble_type = "cascade"

        # Create cascade levels based on market analysis
        cascade_levels = self._create_cvlSA_cascade_levels(base_architecture, market_analysis)
        cascade_architecture.hierarchy_levels = cascade_levels

        # Evaluate and optimize cascade
        optimized_architecture = self._optimize_cascade_levels(cascade_architecture, X, y)

        return optimized_architecture

    def _create_cvlSA_cascade_levels(self, base_architecture: TreeArchitectureCandidate,
                                   market_analysis: Dict) -> List[Dict[str, Any]]:
        """Create CVLSA cascade levels."""
        cascade_levels = []

        # Level 1: Base models with different configurations
        level1 = {
            'level': 1,
            'model_type': 'base',
            'n_models': base_architecture.n_trees // 4,
            'base_params': {
                'max_depth': base_architecture.max_depth,
                'min_samples_split': base_architecture.min_samples_split,
                'min_samples_leaf': base_architecture.min_samples_leaf,
                'max_features': base_architecture.max_features,
                'splitting_strategy': base_architecture.splitting_strategy
            },
            'aggregation_method': 'voting'
        }
        cascade_levels.append(level1)

        # Level 2: Meta-models that take predictions from level 1
        level2 = {
            'level': 2,
            'model_type': 'meta',
            'n_models': base_architecture.n_trees // 8,
            'base_params': {
                'max_depth': min(base_architecture.max_depth + 2, 20),
                'min_samples_split': base_architecture.min_samples_split,
                'min_samples_leaf': base_architecture.min_samples_leaf,
                'max_features': 'auto',
                'splitting_strategy': 'friedman_mse'
            },
            'input_from': [0],  # Take input from level 1
            'aggregation_method': 'stacking'
        }
        cascade_levels.append(level2)

        # Level 3: Final ensemble with regime-specific optimization
        level3 = {
            'level': 3,
            'model_type': 'final',
            'n_models': base_architecture.n_trees // 16,
            'base_params': {
                'max_depth': base_architecture.max_depth,
                'min_samples_split': base_architecture.min_samples_split,
                'min_samples_leaf': base_architecture.min_samples_leaf,
                'max_features': 'auto',
                'splitting_strategy': base_architecture.splitting_strategy
            },
            'input_from': [1],  # Take input from level 2
            'aggregation_method': 'weighted_voting',
            'regime_aware': True  # CVLSA-specific: regime-aware final layer
        }
        cascade_levels.append(level3)

        return cascade_levels

    def _optimize_cascade_levels(self, cascade_architecture: TreeArchitectureCandidate,
                               X: np.ndarray, y: np.ndarray) -> TreeArchitectureCandidate:
        """Optimize cascade levels for better performance."""
        self.logger.info("🔧 Optimizing cascade levels...")

        # Test different cascade configurations
        best_architecture = cascade_architecture
        best_score = 0.0

        # Try different model counts per level
        for model_multiplier in [0.5, 1.0, 1.5, 2.0]:
            test_architecture = TreeArchitectureCandidate(
                n_trees=cascade_architecture.n_trees,
                max_depth=cascade_architecture.max_depth,
                min_samples_split=cascade_architecture.min_samples_split,
                min_samples_leaf=cascade_architecture.min_samples_leaf,
                max_features=cascade_architecture.max_features,
                splitting_strategy=cascade_architecture.splitting_strategy,
                search_method="CVLSA_optimized"
            )

            # Adjust model counts
            for level in cascade_architecture.hierarchy_levels:
                if 'n_models' in level:
                    level['n_models'] = max(1, int(level['n_models'] * model_multiplier))

            test_architecture.hierarchy_levels = cascade_architecture.hierarchy_levels
            test_architecture.is_hierarchical = True
            test_architecture.ensemble_type = "cascade"

            # Evaluate performance
            score = self._evaluate_cascade_performance(test_architecture, X, y)

            if score > best_score:
                best_score = score
                best_architecture = test_architecture

        return best_architecture

    def _evaluate_cascade_performance(self, cascade_architecture: TreeArchitectureCandidate,
                                    X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate cascade architecture performance."""
        try:
            # Create cascade model
            cascade_model = self.base_tas._create_hierarchical_model(cascade_architecture)

            # Train and evaluate
            cascade_model.fit(X, y)

            if len(y.shape) > 1:
                predictions = cascade_model.predict(X)
                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            else:
                accuracy = cascade_model.score(X, y)

            # Calculate cascade efficiency (model complexity vs accuracy)
            complexity_penalty = len(cascade_architecture.hierarchy_levels) * 0.1
            efficiency_score = accuracy * (1 - complexity_penalty)

            return efficiency_score

        except Exception as e:
            self.logger.warning(f"Cascade performance evaluation failed: {e}")
            return 0.0

    def _calculate_cascade_efficiency(self, architecture: TreeArchitectureCandidate) -> float:
        """Calculate cascade efficiency."""
        if not architecture.is_hierarchical:
            return 1.0

        # Efficiency based on cascade structure
        n_levels = len(architecture.hierarchy_levels)
        total_models = sum(level.get('n_models', 0) for level in architecture.hierarchy_levels)

        # Base efficiency
        base_efficiency = 1.0

        # Penalty for complexity
        complexity_penalty = min(0.5, n_levels * 0.1 + total_models / 1000)

        return max(0.0, base_efficiency - complexity_penalty)

    def _calculate_variable_selection_accuracy(self, variable_selection_config: Dict[str, Any]) -> float:
        """Calculate variable selection accuracy."""
        if 'selection_accuracy' in variable_selection_config:
            return variable_selection_config['selection_accuracy']

        return 0.8  # Default high accuracy for CVLSA

    def _calculate_optimal_cascade_depth(self, regime_type: MarketRegime,
                                       micro_regimes: List[MicroRegimeDetectionResult]) -> int:
        """Calculate optimal cascade depth for a regime."""
        # Base depth
        base_depth = 2

        # Adjust based on regime complexity
        if regime_type in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            base_depth += 1
        elif regime_type in [MarketRegime.LOW_VOLATILITY, MarketRegime.CONSOLIDATION]:
            base_depth -= 1

        # Adjust based on micro-regime diversity
        if micro_regimes:
            diversity = len(set(mr.regime_type for mr in micro_regimes))
            base_depth += min(2, diversity // 3)

        return max(2, min(5, base_depth))

    def _calculate_variable_selection_priority(self, regime_type: MarketRegime) -> float:
        """Calculate variable selection priority for a regime."""
        priority_map = {
            MarketRegime.HIGH_VOLATILITY: 0.9,
            MarketRegime.LOW_VOLATILITY: 0.7,
            MarketRegime.TRENDING_UP: 0.8,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.MEAN_REVERTING: 0.6,
            MarketRegime.BREAKOUT: 0.9,
            MarketRegime.CRISIS: 0.95
        }

        return priority_map.get(regime_type, 0.7)

    def _determine_cvlSA_architecture_for_regime(self, regime_type: MarketRegime) -> str:
        """Determine CVLSA architecture type for a regime."""
        architecture_map = {
            MarketRegime.HIGH_VOLATILITY: "deep_cascade",
            MarketRegime.LOW_VOLATILITY: "shallow_cascade",
            MarketRegime.TRENDING_UP: "trend_optimized",
            MarketRegime.TRENDING_DOWN: "trend_optimized",
            MarketRegime.MEAN_REVERTING: "mean_reversion_optimized",
            MarketRegime.BREAKOUT: "breakout_optimized",
            MarketRegime.CONSOLIDATION: "consolidation_optimized",
            MarketRegime.CRISIS: "crisis_optimized"
        }

        return architecture_map.get(regime_type, "standard_cascade")

    def _optimize_feature_count(self, X: np.ndarray, y: np.ndarray,
                              cascade_architecture: TreeArchitectureCandidate,
                              variable_selection_config: Dict[str, Any]) -> int:
        """Optimize feature count for CVLSA."""
        try:
            # Test different feature counts
            n_features = X.shape[1]
            optimal_count = n_features

            for feature_count in [n_features // 4, n_features // 2, n_features * 3 // 4, n_features]:
                if feature_count == 0:
                    continue

                # Create test architecture with reduced features
                test_X = X[:, :feature_count]

                # Evaluate performance
                performance = self._evaluate_cascade_performance(cascade_architecture, test_X, y)

                # Track best feature count
                if performance > 0.8:  # Good performance threshold
                    optimal_count = feature_count
                    break

            return optimal_count

        except Exception as e:
            self.logger.warning(f"Feature count optimization failed: {e}")
            return X.shape[1]

    def _evaluate_variable_selection_method(self, X: np.ndarray, y: np.ndarray,
                                          cascade_architecture: TreeArchitectureCandidate,
                                          method: str) -> Dict[str, float]:
        """Evaluate variable selection method performance."""
        try:
            # Create cascade model
            cascade_model = self.base_tas._create_hierarchical_model(cascade_architecture)

            # Train and evaluate
            cascade_model.fit(X, y)

            if len(y.shape) > 1:
                predictions = cascade_model.predict(X)
                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            else:
                accuracy = cascade_model.score(X, y)

            return {
                'accuracy': accuracy,
                'n_features': X.shape[1],
                'score': accuracy * (X.shape[1] / 100)  # Balance accuracy and feature count
            }

        except Exception as e:
            self.logger.warning(f"Variable selection method evaluation failed: {e}")
            return {'accuracy': 0.0, 'n_features': X.shape[1], 'score': 0.0}

    def _create_cascade_levels(self, architecture: TreeArchitectureCandidate) -> List[Dict[str, Any]]:
        """Create cascade levels from architecture."""
        if not architecture.is_hierarchical:
            return []

        return architecture.hierarchy_levels

    def _calculate_cascade_efficiency_from_results(self, evaluation_results: List[EvaluationResult]) -> float:
        """Calculate cascade efficiency from evaluation results."""
        if not evaluation_results:
            return 0.0

        # Average performance across splits
        avg_accuracy = np.mean([r.accuracy for r in evaluation_results])
        avg_efficiency = np.mean([r.efficiency_score for r in evaluation_results if hasattr(r, 'efficiency_score')])

        # Calculate cascade efficiency
        return (avg_accuracy + avg_efficiency) / 2

    def _analyze_micro_regime_distribution(self, micro_regimes: List[MicroRegimeDetectionResult]) -> Dict[str, float]:
        """Analyze distribution of micro-regimes."""
        if not micro_regimes:
            return {}

        regime_counts = defaultdict(int)
        total_regimes = len(micro_regimes)

        for regime in micro_regimes:
            regime_counts[regime.regime_type.value] += 1

        return {regime: count / total_regimes for regime, count in regime_counts.items()}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def _create_time_based_splits(self, features: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Create time-based data splits for evaluation."""
        splits = []
        data_length = len(features)

        for i in range(n_splits):
            # Create time-based split
            split_size = data_length // (n_splits + 1)
            start_idx = i * split_size
            mid_idx = (i + 1) * split_size
            end_idx = min((i + 2) * split_size, data_length)

            train_data = features.iloc[:mid_idx]
            val_data = features.iloc[mid_idx:end_idx]
            test_data = features.iloc[end_idx:]

            splits.append((train_data, val_data, test_data))

        return splits

    def _analyze_micro_regimes(self, market_data: pd.DataFrame) -> Dict[MicroRegimeType, List[MicroRegimeDetectionResult]]:
        """Analyze micro-regimes in the market data."""
        self.logger.info("🔬 Analyzing micro-regimes for CVLSA...")

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


# Convenience functions for CVLSA
def optimize_cvlSA_architecture(market_data: pd.DataFrame,
                              target_returns: pd.Series,
                              config: Optional[TASConfig] = None) -> CVLSAResult:
    """
    Convenience function for CVLSA optimization.

    Args:
        market_data: Historical market data
        target_returns: Target returns for training
        config: TAS configuration

    Returns:
        CVLSAResult with optimal CVLSA architecture
    """
    if config is None:
        config = TASConfig.create_advanced_trading_config()
        config.architecture_type = TASArchitectureType.TREE_ONLY

    cvlsa = TreeCVLSASearch(config)
    return cvlsa.optimize_cvlSA_architecture(market_data, target_returns)

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
