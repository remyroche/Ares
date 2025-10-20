"""
SHAP Interaction Scorer for Feature Interaction Generation

Provides SHAP scoring with OOF aggregation, interaction centrality calculation,
and stability analysis for the three-phase interaction generation pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from dataclasses import dataclass
import logging
import time
from collections import defaultdict

from src.utils.tprint import tprint

# Import optimization utilities
from .advanced_memory_manager import AdvancedMemoryManager, MemoryConfig
from .enhanced_vectorbt_manager import EnhancedVectorBTManager, VectorBTConfig
from .m1_parallel_processor import M1ParallelProcessor, ParallelConfig
from .data_structure_optimizer import DataStructureOptimizer, OptimizationConfig
from .optimized_shap_computer import OptimizedSHAPComputer, SHAPConfig

# Import SHAP and LightGBM
try:
    import lightgbm as lgb
    import shap
    LGBM_SHAP_AVAILABLE = True
except ImportError:
    LGBM_SHAP_AVAILABLE = False
    warnings.warn("LightGBM/SHAP not available. Install with: pip install lightgbm shap")

# Import time series CV utilities
try:
    from src.utils.ml_common.validation.temporal_cross_validation import TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    warnings.warn("Time series CV utilities not available")

logger = logging.getLogger(__name__)

@dataclass
class SHAPScorerConfig:
    """Configuration for SHAP scoring."""
    
    # LGBM parameters
    lgbm_params: Dict[str, Any] = None
    
    # SHAP settings
    shap_sample_size: int = 1000
    shap_max_samples: int = 1000
    use_shap_interactions: bool = True
    interaction_pairs_limit: int = 25
    
    # Cross-validation settings
    n_folds: int = 3
    test_size: Optional[float] = None
    gap: int = 0
    
    # Scoring weights
    shap_weight: float = 0.5
    interaction_centrality_weight: float = 0.3
    stability_weight: float = 0.2
    
    # Stability analysis
    stability_method: str = 'coefficient_of_variation'  # 'coefficient_of_variation', 'std_ratio'
    stability_regime_aware: bool = True
    volatility_regime_percentiles: Tuple[float, float] = (0.33, 0.67)
    
    # Top-K filtering
    enable_top_k_filter: bool = True
    top_k_features: int = 60
    
    # Memory optimization
    enable_memory_optimization: bool = True
    chunk_size: int = 10000

class SHAPInteractionScorer:
    """
    Scores features using SHAP with interaction centrality and stability analysis.
    
    Provides:
    1. OOF SHAP aggregation across time series folds
    2. Interaction centrality calculation
    3. Stability analysis (within volatility regimes)
    4. Combined scoring with configurable weights
    """
    
    def __init__(self, config: Optional[SHAPScorerConfig] = None):
        """Initialize the SHAP scorer with advanced optimizations."""
        self.config = config or SHAPScorerConfig()
        self.logger = logger
        
        # Initialize optimization components
        self.memory_manager = AdvancedMemoryManager(MemoryConfig())
        self.vectorbt_manager = EnhancedVectorBTManager(VectorBTConfig())
        self.parallel_processor = M1ParallelProcessor(ParallelConfig())
        self.data_optimizer = DataStructureOptimizer(OptimizationConfig())
        self.optimized_shap_computer = OptimizedSHAPComputer(SHAPConfig())
        
        # Set default LGBM parameters
        if self.config.lgbm_params is None:
            self.config.lgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42,
                'force_col_wise': True,
            }
        
        # Cache for computed scores
        self._score_cache = {}
        self._interaction_cache = {}
        
    def _create_time_series_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Create time series splits for OOF validation."""
        if not CV_AVAILABLE:
            # Fallback to simple time-based split
            split_size = len(X) // (self.config.n_folds + 1)
            splits = []
            for i in range(self.config.n_folds):
                train_end = split_size * (i + 1)
                val_start = train_end
                val_end = split_size * (i + 2)
                
                if val_end > len(X):
                    val_end = len(X)
                    
                if val_start >= val_end:
                    break
                    
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                
                splits.append((X_train, y_train, X_val, y_val))
            return splits
        
        # Use proper time series split
        try:
            tss = TimeSeriesSplit(n_splits=self.config.n_folds, test_size=self.config.test_size, gap=self.config.gap)
        except:
            tss = SklearnTimeSeriesSplit(n_splits=self.config.n_folds)
            
        splits = []
        for train_idx, val_idx in tss.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            splits.append((X_train, y_train, X_val, y_val))
            
        return splits
    
    def _compute_volatility_regimes(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Compute volatility regimes for regime-aware stability analysis."""
        if not self.config.stability_regime_aware:
            return None
            
        try:
            # Use target volatility as proxy for market regime
            # Calculate rolling volatility of targets
            y_series = pd.Series(y)
            rolling_vol = y_series.rolling(window=20, min_periods=1).std()
            
            # Define regime boundaries
            low_percentile, high_percentile = self.config.volatility_regime_percentiles
            low_threshold = rolling_vol.quantile(low_percentile)
            high_threshold = rolling_vol.quantile(high_percentile)
            
            # Assign regimes
            regimes = np.zeros(len(y))
            regimes[rolling_vol <= low_threshold] = 0  # Low volatility
            regimes[rolling_vol >= high_threshold] = 2  # High volatility
            regimes[(rolling_vol > low_threshold) & (rolling_vol < high_threshold)] = 1  # Medium volatility
            
            return regimes
            
        except Exception as e:
            tprint(f"⚠️ Failed to compute volatility regimes: {e}")
            return None
    
    def _train_lgbm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         feature_names: List[str]) -> lgb.Booster:
        """Train LightGBM model with validation."""
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            self.config.lgbm_params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        return model
    
    def _compute_shap_values(self, model: lgb.Booster, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for features."""
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            
            # Sample data if too large
            if len(X) > self.config.shap_sample_size:
                sample_indices = np.random.choice(len(X), self.config.shap_sample_size, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
                
            shap_values = explainer.shap_values(X_sample, check_additivity=False)
            
            # Handle multi-class case
            if len(shap_values.shape) > 2:
                shap_values = shap_values[0]  # Take first class
                
            return shap_values
            
        except Exception as e:
            tprint(f"⚠️ Failed to compute SHAP values: {e}")
            return np.zeros((len(X), len(self.config.lgbm_params.get('feature_name', []))))
    
    def _compute_shap_interactions(self, model: lgb.Booster, X: np.ndarray,
                                 feature_names: List[str]) -> np.ndarray:
        """Compute SHAP interaction values."""
        if not self.config.use_shap_interactions:
            return np.array([])
            
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            
            # Sample data if too large
            if len(X) > self.config.shap_sample_size:
                sample_indices = np.random.choice(len(X), self.config.shap_sample_size, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
                
            # Compute interaction values (this is expensive)
            interaction_values = explainer.shap_interaction_values(X_sample)
            
            return interaction_values
            
        except Exception as e:
            tprint(f"⚠️ Failed to compute SHAP interactions: {e}")
            return np.array([])
    
    def _compute_interaction_centrality(self, interaction_values: np.ndarray,
                                      feature_names: List[str]) -> Dict[Tuple[str, str], float]:
        """Compute interaction centrality scores."""
        if len(interaction_values) == 0:
            return {}
            
        centrality_scores = {}
        
        try:
            # Compute absolute interaction values
            abs_interactions = np.abs(interaction_values)
            
            # Sum over samples to get interaction strengths
            interaction_strengths = np.sum(abs_interactions, axis=0)
            
            # Limit to top pairs
            n_features = len(feature_names)
            top_pairs = min(self.config.interaction_pairs_limit, n_features * (n_features - 1) // 2)
            
            # Get top interaction pairs
            interaction_pairs = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction_pairs.append((i, j, interaction_strengths[i, j]))
                    
            # Sort by interaction strength
            interaction_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Store top pairs
            for i, j, strength in interaction_pairs[:top_pairs]:
                centrality_scores[(feature_names[i], feature_names[j])] = float(strength)
                
        except Exception as e:
            tprint(f"⚠️ Failed to compute interaction centrality: {e}")
            
        return centrality_scores
    
    def _compute_stability_metrics(self, shap_values_by_fold: List[np.ndarray],
                                 regimes_by_fold: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Compute stability metrics for features."""
        if not shap_values_by_fold:
            return np.array([])
            
        n_features = shap_values_by_fold[0].shape[1]
        stability_scores = np.zeros(n_features)
        
        try:
            # Compute mean SHAP values per fold
            mean_shap_by_fold = [np.mean(np.abs(shap_vals), axis=0) for shap_vals in shap_values_by_fold]
            
            if self.config.stability_regime_aware and regimes_by_fold:
                # Compute stability within regimes
                for i in range(n_features):
                    feature_stability = []
                    
                    for fold_idx, (shap_vals, regimes) in enumerate(zip(shap_values_by_fold, regimes_by_fold)):
                        # Compute SHAP values within each regime
                        regime_shap = {}
                        for regime in [0, 1, 2]:  # Low, medium, high volatility
                            regime_mask = regimes == regime
                            if np.sum(regime_mask) > 0:
                                regime_shap[regime] = np.mean(np.abs(shap_vals[regime_mask, i]))
                                
                        # Compute stability across regimes
                        if len(regime_shap) > 1:
                            regime_values = list(regime_shap.values())
                            if self.config.stability_method == 'coefficient_of_variation':
                                stability = np.std(regime_values) / (np.mean(regime_values) + 1e-8)
                            else:  # std_ratio
                                stability = np.std(regime_values)
                            feature_stability.append(stability)
                    
                    # Average stability across folds
                    if feature_stability:
                        stability_scores[i] = np.mean(feature_stability)
                        
            else:
                # Simple stability across folds
                mean_shap_matrix = np.array(mean_shap_by_fold)
                
                for i in range(n_features):
                    feature_shap_across_folds = mean_shap_matrix[:, i]
                    
                    if self.config.stability_method == 'coefficient_of_variation':
                        stability = np.std(feature_shap_across_folds) / (np.mean(feature_shap_across_folds) + 1e-8)
                    else:  # std_ratio
                        stability = np.std(feature_shap_across_folds)
                        
                    stability_scores[i] = stability
                    
        except Exception as e:
            tprint(f"⚠️ Failed to compute stability metrics: {e}")
            
        return stability_scores
    
    def score_features(self, 
                      features_df: pd.DataFrame,
                      targets: pd.Series,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Score features using SHAP with OOF aggregation."""
        tprint("🔄 [SHAP] Starting SHAP feature scoring...")
        
        if not LGBM_SHAP_AVAILABLE:
            tprint("❌ [SHAP] LightGBM/SHAP not available")
            return {'error': 'LightGBM/SHAP not available'}
        
        # Prepare data
        if feature_names is None:
            feature_names = list(features_df.columns)
            
        # Align features and targets
        tprint("🔍 [SHAP] Aligning features and targets")
        aligned_data = features_df.join(targets.rename('target'), how='inner').dropna()
        if aligned_data.empty:
            tprint("❌ [SHAP] No aligned data after joining features and targets")
            return {'error': 'No aligned data'}
            
        X = aligned_data[feature_names].values
        y = aligned_data['target'].values
        
        tprint(f"📊 [SHAP] Data shape: {X.shape}, features: {len(feature_names)}")
        
        # Apply top-K filtering if enabled
        if self.config.enable_top_k_filter and len(feature_names) > self.config.top_k_features:
            tprint(f"🔍 [SHAP] Applying top-K filter: selecting top {self.config.top_k_features} features by LGBM gain")
            
            # Train initial model to get feature importance
            train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
            initial_model = lgb.train(
                self.config.lgbm_params,
                train_data,
                num_boost_round=50,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Get top features by gain
            importance = initial_model.feature_importance(importance_type='gain')
            top_indices = np.argsort(importance)[::-1][:self.config.top_k_features]
            
            # Update feature set
            feature_names = [feature_names[i] for i in top_indices]
            X = X[:, top_indices]
            
            tprint(f"📊 Filtered to {len(feature_names)} features")
        
        # Create time series splits
        splits = self._create_time_series_splits(X, y)
        tprint(f"📊 Created {len(splits)} time series splits")
        
        # Compute volatility regimes if enabled
        regimes = self._compute_volatility_regimes(X, y)
        regimes_by_fold = None
        if regimes is not None:
            regimes_by_fold = []
            for _, _, X_val, y_val in splits:
                val_start = len(X) - len(X_val)
                val_end = len(X)
                regimes_by_fold.append(regimes[val_start:val_end])
            tprint("📊 Computed volatility regimes for stability analysis")
        
        # OOF SHAP aggregation
        shap_values_by_fold = []
        interaction_centrality_by_fold = []
        
        for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(splits):
            tprint(f"📊 Processing fold {fold_idx + 1}/{len(splits)}")
            
            # Train model
            model = self._train_lgbm_model(X_train, y_train, X_val, y_val, feature_names)
            
            # Compute SHAP values
            shap_values = self._compute_shap_values(model, X_val)
            shap_values_by_fold.append(shap_values)
            
            # Compute SHAP interactions
            if self.config.use_shap_interactions:
                interaction_values = self._compute_shap_interactions(model, X_val, feature_names)
                if len(interaction_values) > 0:
                    centrality_scores = self._compute_interaction_centrality(interaction_values, feature_names)
                    interaction_centrality_by_fold.append(centrality_scores)
        
        # Aggregate SHAP values
        mean_shap_scores = np.mean([np.mean(np.abs(shap_vals), axis=0) for shap_vals in shap_values_by_fold], axis=0)
        
        # Aggregate interaction centrality
        aggregated_centrality = {}
        if interaction_centrality_by_fold:
            # Sum centrality scores across folds
            for centrality_dict in interaction_centrality_by_fold:
                for pair, score in centrality_dict.items():
                    aggregated_centrality[pair] = aggregated_centrality.get(pair, 0) + score
            
            # Average across folds
            n_folds = len(interaction_centrality_by_fold)
            aggregated_centrality = {pair: score / n_folds for pair, score in aggregated_centrality.items()}
        
        # Compute interaction centrality per feature
        feature_centrality = np.zeros(len(feature_names))
        for pair, score in aggregated_centrality.items():
            f1_name, f2_name = pair
            if f1_name in feature_names:
                feature_centrality[feature_names.index(f1_name)] += score
            if f2_name in feature_names:
                feature_centrality[feature_names.index(f2_name)] += score
        
        # Compute stability metrics
        stability_scores = self._compute_stability_metrics(shap_values_by_fold, regimes_by_fold)
        
        # Compute combined scores
        # Normalize scores to [0, 1] range
        norm_shap = mean_shap_scores / (np.max(mean_shap_scores) + 1e-8)
        norm_centrality = feature_centrality / (np.max(feature_centrality) + 1e-8)
        norm_stability = 1 - (stability_scores / (np.max(stability_scores) + 1e-8))  # Invert so lower is better
        
        combined_scores = (
            self.config.shap_weight * norm_shap +
            self.config.interaction_centrality_weight * norm_centrality +
            self.config.stability_weight * norm_stability
        )
        
        # Create results
        results = {
            'feature_names': feature_names,
            'shap_scores': mean_shap_scores,
            'interaction_centrality': feature_centrality,
            'stability_scores': stability_scores,
            'combined_scores': combined_scores,
            'interaction_centrality_pairs': aggregated_centrality,
            'n_folds': len(splits),
            'success': True
        }
        
        tprint(f"✅ SHAP scoring completed for {len(feature_names)} features across {len(splits)} folds")
        
        return results
    
    def get_top_features(self, 
                        results: Dict[str, Any],
                        top_k: int = 40,
                        score_type: str = 'combined') -> List[str]:
        """Get top features based on specified score type."""
        if not results.get('success', False):
            return []
            
        feature_names = results['feature_names']
        
        if score_type == 'combined':
            scores = results['combined_scores']
        elif score_type == 'shap':
            scores = results['shap_scores']
        elif score_type == 'centrality':
            scores = results['interaction_centrality']
        elif score_type == 'stability':
            scores = results['stability_scores']
        else:
            tprint(f"⚠️ Unknown score type: {score_type}")
            return []
        
        # Sort features by score
        sorted_indices = np.argsort(scores)[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:top_k]]
        
        return top_features
    
    def get_interaction_centrality_pairs(self, 
                                       results: Dict[str, Any],
                                       top_k: int = 25) -> List[Tuple[str, str, float]]:
        """Get top interaction centrality pairs."""
        if not results.get('success', False):
            return []
            
        centrality_pairs = results.get('interaction_centrality_pairs', {})
        
        # Sort pairs by centrality score
        sorted_pairs = sorted(centrality_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # Return top pairs
        top_pairs = [(pair[0], pair[1], score) for pair, score in sorted_pairs[:top_k]]
        
        return top_pairs

# Import tprint for logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
