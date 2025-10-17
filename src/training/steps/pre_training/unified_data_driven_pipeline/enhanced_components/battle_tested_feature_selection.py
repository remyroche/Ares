"""
Battle-Tested Feature Selection with Best Practices Integration

This module implements production-ready feature selection following battle-tested
guidelines for financial ML pipelines, integrating utilities from ml_common.

Key Features:
- Purged walk-forward CV with embargo
- Multi-objective Pareto ranking
- Stability selection with bootstrapped time blocks
- Redundancy pruning with hierarchical clustering
- Economic validation with OOF metrics
- Comprehensive logging and diagnostics
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    SelectKBest, f_regression, f_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Import ML Commons utilities
try:
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator, UnifiedCVResult
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, compute_pareto_front,
        select_knee_point, compute_hypervolume
    )
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    ML_COMMONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Commons not available: {e}")
    ML_COMMONS_AVAILABLE = False

# Import UnifiedVectorizationManager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OperationConfig
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OperationConfig = None

# Import purged K-fold
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class FeatureSelectionConfig:
    """Configuration for battle-tested feature selection."""
    
    # Core selection parameters
    max_features: int = 60
    min_features: int = 4
    min_ic_threshold: float = 0.01
    min_stability_threshold: float = 0.6
    max_correlation_threshold: float = 0.85
    
    # CV parameters
    n_splits: int = 5
    embargo_days: int = 7
    gap_days: int = 1
    
    # Multi-objective weights
    ic_weight: float = 0.4
    stability_weight: float = 0.3
    diversity_weight: float = 0.2
    cost_weight: float = 0.1
    
    # Stability selection
    n_bootstrap: int = 100
    bootstrap_fraction: float = 0.8
    
    # Redundancy pruning
    clustering_method: str = 'ward'  # 'ward', 'complete', 'average'
    distance_metric: str = 'correlation'  # 'correlation', 'euclidean'
    
    # Economic validation
    min_oof_ic: float = 0.005
    min_sharpe_improvement: float = 0.1
    
    # Logging
    enable_detailed_logging: bool = True
    save_artifacts: bool = True
    artifacts_dir: str = "outcomes"


@dataclass
class FeatureScore:
    """Individual feature score and metadata."""
    name: str
    ic_score: float
    stability_score: float
    diversity_score: float
    cost_score: float
    composite_score: float
    inclusion_probability: float
    oof_ic: float
    oof_sharpe: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSelectionResult:
    """Result of battle-tested feature selection."""
    selected_features: List[FeatureScore]
    feature_rankings: pd.DataFrame
    stability_plot_data: Dict[str, Any]
    correlation_heatmap_data: Dict[str, Any]
    pareto_front_data: Dict[str, Any]
    selection_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class BattleTestedFeatureSelector:
    """Production-ready feature selector with battle-tested best practices."""
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """Initialize the feature selector."""
        self.config = config or FeatureSelectionConfig()
        self.logger = logging.getLogger(__name__)
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize CV utilities
        if ML_COMMONS_AVAILABLE:
            self.cv_validator = UnifiedCrossValidator()
        else:
            self.cv_validator = None
            
        # Initialize Pareto utilities
        if ML_COMMONS_AVAILABLE:
            self.pareto_available = True
        else:
            self.pareto_available = False
            
        # Initialize purged K-fold
        if PURGED_KFOLD_AVAILABLE:
            self.purged_kfold = PurgedKFoldTime(
                n_splits=self.config.n_splits,
                embargo_td=pd.Timedelta(days=self.config.embargo_days)
            )
        else:
            self.purged_kfold = None
            
        # Initialize UnifiedVectorizationManager if available
        self.vectorization_manager = None
        if UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_info("✅ UnifiedVectorizationManager initialized for battle-tested feature selection")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize UnifiedVectorizationManager: {e}")
                self.vectorization_manager = None
            
    def select_features(self, 
                       data: pd.DataFrame, 
                       targets: pd.Series,
                       feature_columns: Optional[List[str]] = None) -> FeatureSelectionResult:
        """
        Perform battle-tested feature selection.
        
        Args:
            data: Input DataFrame with features
            targets: Target series
            feature_columns: Optional list of feature columns to consider
            
        Returns:
            FeatureSelectionResult with selected features and diagnostics
        """
        start_time = time.time()
        tprint_info("🎯 Starting battle-tested feature selection")
        
        try:
            # Step 1: Data validation and preparation
            tprint_info("📊 Step 1: Data validation and preparation")
            data, targets, feature_columns = self._validate_and_prepare_data(data, targets, feature_columns)
            
            # Use UnifiedVectorizationManager if available
            if self.vectorization_manager:
                tprint_info("🚀 Using UnifiedVectorizationManager for battle-tested feature selection")
                try:
                    with self.vectorization_manager.performance_monitoring("feature_selection"):
                        result = self.vectorization_manager.optimize_operation(
                            OperationType.FEATURE_SELECTION,
                            data,
                            targets=targets,
                            feature_columns=feature_columns,
                            selection_type="battle_tested"
                        )
                        if result:
                            tprint_success("✅ Vectorization manager optimization completed successfully")
                        else:
                            tprint_warning("⚠️ Vectorization manager returned no result, continuing with standard selection")
                except Exception as e:
                    tprint_warning(f"⚠️ Vectorization manager failed: {e}, continuing with standard selection")
            else:
                tprint_info("ℹ️ UnifiedVectorizationManager not available, using standard battle-tested selection")
            
            # Step 2: Fail-fast gates
            tprint_info("🚪 Step 2: Fail-fast validation gates")
            if not self._apply_fail_fast_gates(data, targets):
                return self._create_failure_result("Failed fail-fast validation gates")
            
            # Step 3: Multi-objective feature scoring
            tprint_info("📈 Step 3: Multi-objective feature scoring")
            feature_scores = self._calculate_multi_objective_scores(data, targets, feature_columns)
            
            # Step 4: Stability selection with bootstrapped time blocks
            tprint_info("🔄 Step 4: Stability selection with bootstrapped time blocks")
            stability_scores = self._stability_selection(data, targets, feature_columns)
            
            # Step 5: Redundancy pruning with hierarchical clustering
            tprint_info("🌳 Step 5: Redundancy pruning with hierarchical clustering")
            pruned_features = self._redundancy_pruning(data, feature_columns, feature_scores)
            
            # Step 6: Pareto-optimized final selection
            tprint_info("🎯 Step 6: Pareto-optimized final selection")
            final_features = self._pareto_optimized_selection(
                feature_scores, stability_scores, pruned_features
            )
            
            # Step 7: Economic validation
            tprint_info("💰 Step 7: Economic validation")
            validated_features = self._economic_validation(data, targets, final_features)
            
            # Step 8: Generate comprehensive results and artifacts
            tprint_info("📊 Step 8: Generating results and artifacts")
            result = self._generate_comprehensive_results(
                validated_features, feature_scores, stability_scores, 
                data, targets, start_time
            )
            
            tprint_success(f"✅ Battle-tested feature selection completed: {len(validated_features)} features selected")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return self._create_failure_result(str(e))
    
    def _validate_and_prepare_data(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series,
                                  feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Validate and prepare data for feature selection."""
        # Validate inputs
        if data is None or len(data) == 0:
            raise ValueError("Input data is None or empty")
        if targets is None or targets.empty:
            raise ValueError("Targets is None or empty")
        if len(data) != len(targets):
            raise ValueError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
        
        # Determine feature columns
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'timestamp']
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Filter data to feature columns only
        feature_data = data[feature_columns].copy()
        
        # Remove features with insufficient variance
        variance_threshold = 1e-8
        high_variance_features = feature_data.var() > variance_threshold
        feature_columns = feature_columns[high_variance_features]
        feature_data = feature_data[feature_columns]
        
        tprint_info(f"📊 Prepared {len(feature_columns)} features for selection")
        return feature_data, targets, feature_columns
    
    def _apply_fail_fast_gates(self, data: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates."""
        # Gate 1: Minimum data size
        if len(data) < 100:
            tprint_warning("⚠️ Insufficient data for reliable feature selection")
            return False
        
        # Gate 2: Target variance check
        if targets.var() < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False
        
        # Gate 3: Feature quality check
        nan_ratios = data.isnull().sum() / len(data)
        high_nan_features = nan_ratios > 0.5
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_features.sum()} features have >50% NaN values")
            return False
        
        # Gate 4: Memory check
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > 1000:  # 1GB limit
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False
        
        return True
    
    def _calculate_multi_objective_scores(self, 
                                        data: pd.DataFrame, 
                                        targets: pd.Series,
                                        feature_columns: List[str]) -> List[FeatureScore]:
        """Calculate multi-objective scores for all features."""
        tprint_info("📈 Calculating multi-objective scores")
        
        feature_scores = []
        
        for feature_name in feature_columns:
            try:
                feature_data = data[feature_name].dropna()
                if len(feature_data) == 0:
                    continue
                
                # Align targets with feature data
                aligned_targets = targets.loc[feature_data.index]
                
                # Calculate IC score (Information Coefficient)
                ic_score = self._calculate_ic_score(feature_data, aligned_targets)
                
                # Calculate stability score
                stability_score = self._calculate_stability_score(feature_data, aligned_targets)
                
                # Calculate diversity score based on feature uniqueness
                diversity_score = self._calculate_diversity_score(feature_data, feature_name)
                
                # Calculate cost score (feature complexity)
                cost_score = self._calculate_cost_score(feature_data)
                
                # Calculate composite score
                composite_score = (
                    self.config.ic_weight * ic_score +
                    self.config.stability_weight * stability_score +
                    self.config.diversity_weight * diversity_score +
                    self.config.cost_weight * cost_score
                )
                
                # Calculate OOF metrics
                oof_ic, oof_sharpe = self._calculate_oof_metrics(feature_data, aligned_targets)
                
                feature_score = FeatureScore(
                    name=feature_name,
                    ic_score=ic_score,
                    stability_score=stability_score,
                    diversity_score=diversity_score,
                    cost_score=cost_score,
                    composite_score=composite_score,
                    inclusion_probability=0.0,  # Will be updated in stability selection
                    oof_ic=oof_ic,
                    oof_sharpe=oof_sharpe,
                    metadata={
                        'feature_type': self._classify_feature_type(feature_name),
                        'data_points': len(feature_data),
                        'missing_ratio': data[feature_name].isnull().sum() / len(data)
                    }
                )
                
                feature_scores.append(feature_score)
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to score feature {feature_name}: {e}")
                continue
        
        # Sort by composite score
        feature_scores.sort(key=lambda x: x.composite_score, reverse=True)
        
        tprint_info(f"📊 Calculated scores for {len(feature_scores)} features")
        return feature_scores
    
    def _calculate_ic_score(self, feature: pd.Series, targets: pd.Series) -> float:
        """Calculate Information Coefficient score."""
        try:
            # Use Spearman correlation for IC
            correlation, _ = stats.spearmanr(feature, targets)
            if np.isnan(correlation):
                return 0.0
            return abs(correlation)
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, feature: pd.Series, targets: pd.Series) -> float:
        """Calculate stability score using purged walk-forward CV."""
        try:
            if self.purged_kfold is None:
                # Fallback to simple correlation stability
                return abs(safe_correlation(feature, targets))
            
            # Use purged walk-forward CV
            correlations = []
            for train_idx, val_idx in self.purged_kfold.split(feature.index):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                train_corr = safe_correlation(
                    feature.iloc[train_idx], targets.iloc[train_idx]
                )
                val_corr = safe_correlation(
                    feature.iloc[val_idx], targets.iloc[val_idx]
                )
                
                if not np.isnan(train_corr) and not np.isnan(val_corr):
                    correlations.append(val_corr)
            
            if not correlations:
                return 0.0
            
            # Stability is inverse of standard deviation
            stability = 1.0 / (1.0 + np.std(correlations))
            return min(stability, 1.0)
            
        except Exception:
            return 0.0

    def _calculate_diversity_score(self, feature_data: pd.Series, feature_name: str) -> float:
        """Calculate diversity score based on feature uniqueness and distribution."""
        try:
            if len(feature_data) < 2:
                return 0.0
                
            # Calculate coefficient of variation as diversity measure
            mean_val = feature_data.mean()
            std_val = feature_data.std()
            
            if mean_val == 0 or np.isnan(mean_val) or np.isnan(std_val):
                return 0.0
                
            cv = std_val / abs(mean_val)
            
            # Calculate unique value ratio
            unique_ratio = feature_data.nunique() / len(feature_data)
            
            # Calculate entropy-based diversity
            value_counts = feature_data.value_counts()
            probabilities = value_counts / len(feature_data)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Combine metrics (higher is more diverse)
            diversity_score = (cv * 0.3 + unique_ratio * 0.4 + normalized_entropy * 0.3)
            
            return min(diversity_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_cost_score(self, feature: pd.Series) -> float:
        """Calculate cost score (lower is better)."""
        try:
            # Cost based on computational complexity
            # Simple features get lower cost scores
            feature_name = feature.name.lower()
            
            if any(x in feature_name for x in ['sma', 'ema', 'rsi', 'macd']):
                return 0.2  # Simple technical indicators
            elif any(x in feature_name for x in ['rolling', 'expanding']):
                return 0.5  # Rolling calculations
            elif any(x in feature_name for x in ['interaction', 'cross', 'product']):
                return 0.8  # Complex interactions
            else:
                return 0.4  # Default medium complexity
                
        except Exception:
            return 0.5
    
    def _calculate_oof_metrics(self, feature: pd.Series, targets: pd.Series) -> Tuple[float, float]:
        """Calculate out-of-fold IC and Sharpe metrics."""
        try:
            if self.purged_kfold is None:
                return 0.0, 0.0
            
            oof_ics = []
            oof_returns = []
            
            for train_idx, val_idx in self.purged_kfold.split(feature.index):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                val_feature = feature.iloc[val_idx]
                val_targets = targets.iloc[val_idx]
                
                # Calculate IC
                ic = safe_correlation(val_feature, val_targets)
                if not np.isnan(ic):
                    oof_ics.append(ic)
                
                # Calculate returns (simplified)
                if len(val_targets) > 1:
                    returns = val_targets.pct_change().dropna()
                    if len(returns) > 0:
                        sharpe = safe_divide(returns.mean(), returns.std())
                        if not np.isnan(sharpe):
                            oof_returns.append(sharpe)
            
            oof_ic = np.mean(oof_ics) if oof_ics else 0.0
            oof_sharpe = np.mean(oof_returns) if oof_returns else 0.0
            
            return oof_ic, oof_sharpe
            
        except Exception:
            return 0.0, 0.0
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type for metadata."""
        feature_name_lower = feature_name.lower()
        
        if any(x in feature_name_lower for x in ['price', 'close', 'open', 'high', 'low']):
            return 'price'
        elif any(x in feature_name_lower for x in ['volume', 'vol']):
            return 'volume'
        elif any(x in feature_name_lower for x in ['rsi', 'macd', 'sma', 'ema', 'bb']):
            return 'technical'
        elif any(x in feature_name_lower for x in ['interaction', 'cross', 'product']):
            return 'interaction'
        elif any(x in feature_name_lower for x in ['htf', 'daily', 'hourly']):
            return 'htf'
        else:
            return 'other'
    
    def _stability_selection(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           feature_columns: List[str]) -> Dict[str, float]:
        """Perform stability selection with bootstrapped time blocks."""
        tprint_info("🔄 Performing stability selection")
        
        stability_scores = {}
        n_samples = len(data)
        bootstrap_size = int(n_samples * self.config.bootstrap_fraction)
        
        for _ in range(self.config.n_bootstrap):
            try:
                # Bootstrap sample with time awareness
                start_idx = np.random.randint(0, n_samples - bootstrap_size)
                end_idx = start_idx + bootstrap_size
                bootstrap_indices = np.arange(start_idx, end_idx)
                
                bootstrap_data = data.iloc[bootstrap_indices]
                bootstrap_targets = targets.iloc[bootstrap_indices]
                
                # Quick feature selection on bootstrap sample
                for feature_name in feature_columns:
                    if feature_name not in bootstrap_data.columns:
                        continue
                    
                    feature_data = bootstrap_data[feature_name].dropna()
                    if len(feature_data) < 10:
                        continue
                    
                    aligned_targets = bootstrap_targets.loc[feature_data.index]
                    ic = safe_correlation(feature_data, aligned_targets)
                    
                    if not np.isnan(ic) and abs(ic) > self.config.min_ic_threshold:
                        stability_scores[feature_name] = stability_scores.get(feature_name, 0) + 1
                        
            except Exception as e:
                tprint_warning(f"⚠️ Bootstrap iteration failed: {e}")
                continue
        
        # Convert counts to probabilities
        for feature_name in stability_scores:
            stability_scores[feature_name] /= self.config.n_bootstrap
        
        tprint_info(f"📊 Stability selection completed for {len(stability_scores)} features")
        return stability_scores
    
    def _redundancy_pruning(self, 
                           data: pd.DataFrame, 
                           feature_columns: List[str],
                           feature_scores: List[FeatureScore]) -> List[str]:
        """Perform redundancy pruning using hierarchical clustering."""
        tprint_info("🌳 Performing redundancy pruning")
        
        if len(feature_columns) < 2:
            return feature_columns
        
        try:
            # Calculate correlation matrix
            feature_data = data[feature_columns].dropna()
            if len(feature_data) < 10:
                return feature_columns
            
            corr_matrix = feature_data.corr().abs()
            
            # Convert to distance matrix
            distance_matrix = 1 - corr_matrix
            distance_matrix = distance_matrix.fillna(1.0)  # Handle NaN correlations
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(
                squareform(distance_matrix), 
                method=self.config.clustering_method
            )
            
            # Cluster features based on correlation threshold
            cluster_labels = fcluster(
                linkage_matrix, 
                1 - self.config.max_correlation_threshold, 
                criterion='distance'
            )
            
            # Select one feature per cluster (highest composite score)
            cluster_features = {}
            for i, feature_name in enumerate(feature_columns):
                cluster_id = cluster_labels[i]
                if cluster_id not in cluster_features:
                    cluster_features[cluster_id] = []
                
                # Find feature score
                feature_score = next(
                    (fs for fs in feature_scores if fs.name == feature_name), 
                    None
                )
                if feature_score:
                    cluster_features[cluster_id].append((feature_name, feature_score.composite_score))
            
            # Select best feature from each cluster
            pruned_features = []
            for cluster_id, features in cluster_features.items():
                if features:
                    # Sort by composite score and take the best
                    features.sort(key=lambda x: x[1], reverse=True)
                    pruned_features.append(features[0][0])
            
            tprint_info(f"🌳 Redundancy pruning: {len(feature_columns)} -> {len(pruned_features)} features")
            return pruned_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Redundancy pruning failed: {e}")
            return feature_columns
    
    def _pareto_optimized_selection(self, 
                                  feature_scores: List[FeatureScore],
                                  stability_scores: Dict[str, float],
                                  pruned_features: List[str]) -> List[FeatureScore]:
        """Perform Pareto-optimized feature selection."""
        tprint_info("🎯 Performing Pareto-optimized selection")
        
        # Filter to pruned features and add stability scores
        candidate_features = []
        for feature_score in feature_scores:
            if feature_score.name in pruned_features:
                # Update inclusion probability from stability selection
                feature_score.inclusion_probability = stability_scores.get(feature_score.name, 0.0)
                
                # Apply stability threshold
                if feature_score.inclusion_probability >= self.config.min_stability_threshold:
                    candidate_features.append(feature_score)
        
        # Sort by composite score
        candidate_features.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Apply Pareto optimization if available
        if self.pareto_available and len(candidate_features) > 10:
            try:
                # Create Pareto solutions
                solutions = []
                for i, feature in enumerate(candidate_features):
                    solution = Solution(
                        values=[feature.ic_score, feature.stability_score, feature.diversity_score],
                        metadata={'feature_name': feature.name, 'index': i}
                    )
                    solutions.append(solution)
                
                # Compute Pareto front
                pareto_front = compute_pareto_front(solutions)
                
                # Select features from Pareto front
                pareto_features = []
                for solution in pareto_front:
                    feature_name = solution.metadata['feature_name']
                    feature = next(fs for fs in candidate_features if fs.name == feature_name)
                    pareto_features.append(feature)
                
                # Limit to max_features
                final_features = pareto_features[:self.config.max_features]
                
            except Exception as e:
                tprint_warning(f"⚠️ Pareto optimization failed: {e}")
                final_features = candidate_features[:self.config.max_features]
        else:
            # Simple top-k selection
            final_features = candidate_features[:self.config.max_features]
        
        tprint_info(f"🎯 Selected {len(final_features)} features from Pareto optimization")
        return final_features
    
    def _economic_validation(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           features: List[FeatureScore]) -> List[FeatureScore]:
        """Perform economic validation of selected features."""
        tprint_info("💰 Performing economic validation")
        
        validated_features = []
        
        for feature in features:
            try:
                # Check OOF IC threshold
                if feature.oof_ic < self.config.min_oof_ic:
                    tprint_warning(f"⚠️ Feature {feature.name} failed OOF IC threshold: {feature.oof_ic:.4f}")
                    continue
                
                # Check OOF Sharpe improvement
                if feature.oof_sharpe < self.config.min_sharpe_improvement:
                    tprint_warning(f"⚠️ Feature {feature.name} failed OOF Sharpe threshold: {feature.oof_sharpe:.4f}")
                    continue
                
                validated_features.append(feature)
                
            except Exception as e:
                tprint_warning(f"⚠️ Economic validation failed for {feature.name}: {e}")
                continue
        
        tprint_info(f"💰 Economic validation: {len(features)} -> {len(validated_features)} features")
        return validated_features
    
    def _generate_comprehensive_results(self, 
                                      selected_features: List[FeatureScore],
                                      all_feature_scores: List[FeatureScore],
                                      stability_scores: Dict[str, float],
                                      data: pd.DataFrame,
                                      targets: pd.Series,
                                      start_time: float) -> FeatureSelectionResult:
        """Generate comprehensive results and artifacts."""
        tprint_info("📊 Generating comprehensive results")
        
        # Create feature rankings DataFrame
        rankings_data = []
        for feature in all_feature_scores:
            rankings_data.append({
                'feature_name': feature.name,
                'ic_score': feature.ic_score,
                'stability_score': feature.stability_score,
                'diversity_score': feature.diversity_score,
                'cost_score': feature.cost_score,
                'composite_score': feature.composite_score,
                'inclusion_probability': stability_scores.get(feature.name, 0.0),
                'oof_ic': feature.oof_ic,
                'oof_sharpe': feature.oof_sharpe,
                'selected': feature.name in [f.name for f in selected_features]
            })
        
        feature_rankings = pd.DataFrame(rankings_data)
        feature_rankings = feature_rankings.sort_values('composite_score', ascending=False)
        
        # Generate stability plot data
        stability_plot_data = {
            'inclusion_probabilities': list(stability_scores.values()),
            'feature_names': list(stability_scores.keys()),
            'threshold': self.config.min_stability_threshold
        }
        
        # Generate correlation heatmap data
        if len(selected_features) > 1:
            selected_feature_names = [f.name for f in selected_features]
            correlation_data = data[selected_feature_names].corr()
            correlation_heatmap_data = {
                'correlation_matrix': correlation_data.values.tolist(),
                'feature_names': selected_feature_names
            }
        else:
            correlation_heatmap_data = {'correlation_matrix': [], 'feature_names': []}
        
        # Generate Pareto front data
        pareto_front_data = {
            'ic_scores': [f.ic_score for f in selected_features],
            'stability_scores': [f.stability_score for f in selected_features],
            'diversity_scores': [f.diversity_score for f in selected_features],
            'feature_names': [f.name for f in selected_features]
        }
        
        # Calculate selection metrics
        selection_metrics = {
            'total_features_analyzed': len(all_feature_scores),
            'features_selected': len(selected_features),
            'selection_ratio': len(selected_features) / len(all_feature_scores) if all_feature_scores else 0,
            'average_ic': np.mean([f.ic_score for f in selected_features]) if selected_features else 0,
            'average_stability': np.mean([f.stability_score for f in selected_features]) if selected_features else 0,
            'average_oof_ic': np.mean([f.oof_ic for f in selected_features]) if selected_features else 0,
            'execution_time': time.time() - start_time
        }
        
        # Save artifacts if enabled
        artifacts = {}
        if self.config.save_artifacts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save feature rankings
            rankings_path = self.artifacts_dir / f"feature_rankings_{timestamp}.csv"
            feature_rankings.to_csv(rankings_path, index=False)
            artifacts['feature_rankings_path'] = str(rankings_path)
            
            # Save selection report
            report_path = self.artifacts_dir / f"selection_report_{timestamp}.json"
            report_data = {
                'selection_metrics': selection_metrics,
                'selected_features': [
                    {
                        'name': f.name,
                        'ic_score': f.ic_score,
                        'stability_score': f.stability_score,
                        'composite_score': f.composite_score,
                        'oof_ic': f.oof_ic,
                        'oof_sharpe': f.oof_sharpe
                    }
                    for f in selected_features
                ],
                'config': {
                    'max_features': self.config.max_features,
                    'min_ic_threshold': self.config.min_ic_threshold,
                    'min_stability_threshold': self.config.min_stability_threshold,
                    'max_correlation_threshold': self.config.max_correlation_threshold
                }
            }
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            artifacts['selection_report_path'] = str(report_path)
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_rankings=feature_rankings,
            stability_plot_data=stability_plot_data,
            correlation_heatmap_data=correlation_heatmap_data,
            pareto_front_data=pareto_front_data,
            selection_metrics=selection_metrics,
            artifacts=artifacts,
            success=True
        )
    
    def _create_failure_result(self, error_message: str) -> FeatureSelectionResult:
        """Create a failure result."""
        return FeatureSelectionResult(
            selected_features=[],
            feature_rankings=pd.DataFrame(),
            stability_plot_data={},
            correlation_heatmap_data={},
            pareto_front_data={},
            selection_metrics={},
            artifacts={},
            success=False,
            error_message=error_message
        )