from ...core.decorators import handles_errors, traced
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

# src/training/steps/fractional_feature_selector.py

"""Fractional Feature Selector: Advanced feature selection using Step08 utilities.
Implements feature selection based on fractional label alignment, multicollinearity reduction,
and feature importance ranking using advanced Step08 utilities and hardware optimizations.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor

from src.utils.logger import get_logger
from src.utils.feature_selection.step08_advanced_feature_selection_per_regime import (
    PerRegimeAdvancedFeatureSelectionStep
)
from src.utils.feature_selection.step08_unified_final import (
    Step08UnifiedFinal
)
from src.utils.hardware.m1_optimizations import M1MemoryOptimizer
from src.utils.parallel_processing_optimizer import MacM1ParallelOptimizer
from src.utils.vectorized_processing_core import VectorizedProcessingCore
from src.utils.monitoring_utils import PerformanceMonitor
from src.utils.error_handler import ErrorHandler
import pandas as pd
import datetime
import logging

from .quality_validation_decorator import (
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection,
    validate_feature_data_quality
)

class FractionalFeatureSelector:
    """Advanced feature selector using Step08 utilities and hardware optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced fractional feature selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Selection parameters
        self.min_features = self.config.get('min_features', 10)
        self.max_features = self.config.get('max_features', 50)
        self.target_feature_count = self.config.get('target_feature_count', 30)
        
        # Selection methods
        self.selection_methods = self.config.get('selection_methods', [
            'correlation', 'importance', 'stability', 'diversity', 'label_alignment'
        ])
        
        # Method weights
        self.method_weights = self.config.get('method_weights', {
            'correlation': 0.25,
            'importance': 0.25,
            'stability': 0.15,
            'diversity': 0.15,
            'label_alignment': 0.20
        })
        
        # Multicollinearity settings
        self.correlation_threshold = self.config.get('correlation_threshold', 0.85)
        self.vif_threshold = self.config.get('vif_threshold', 5.0)
        
        # Label alignment settings
        self.alignment_window = self.config.get('alignment_window', 100)
        self.alignment_threshold = self.config.get('alignment_threshold', 0.1)
        
        # Performance tracking
        self.selection_history = []
        self.logger = get_logger("FractionalFeatureSelector")
        
        # Initialize advanced utilities
        self._initialize_advanced_utilities()
        
        self.logger.info("✅ Advanced Fractional Feature Selector initialized successfully")
    
    def _initialize_advanced_utilities(self):
        """Initialize advanced utilities for feature selection."""
        try:
            # Initialize M1 memory optimizer
            self.memory_optimizer = M1MemoryOptimizer(
                memory_limit_gb=self.config.get('memory_limit_gb', 8.0),
                enable_gc_tuning=self.config.get('enable_gc_tuning', True),
                enable_memory_leak_detection=self.config.get('enable_memory_leak_detection', True)
            )
            
            # Initialize parallel processing optimizer
            self.parallel_optimizer = MacM1ParallelOptimizer(
                max_workers=self.config.get('max_workers', 4),
                chunk_size=self.config.get('chunk_size', 1000),
                use_process_pool=self.config.get('use_process_pool', True),
                memory_limit_mb=self.config.get('memory_limit_mb', 2048)
            )
            
            # Initialize vectorized processing core
            self.vectorized_core = VectorizedProcessingCore(
                enable_gpu_acceleration=self.config.get('enable_gpu_acceleration', False),
                memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
            )
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                enable_detailed_monitoring=self.config.get('enable_detailed_monitoring', True)
            )
            
            # Initialize error handler
            self.error_handler = ErrorHandler(
                enable_graceful_degradation=self.config.get('enable_graceful_degradation', True)
            )
            
            # Initialize Step08 advanced feature selection
            step08_config = self._create_step08_config()
            self.step08_selector = PerRegimeAdvancedFeatureSelectionStep(step08_config)
            
            # Initialize unified final selector
            self.unified_selector = Step08UnifiedFinal()
            
            self.logger.info("✅ Advanced utilities initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some advanced utilities: {e}")
            # Fallback to basic functionality
            self.memory_optimizer = None
            self.parallel_optimizer = None
            self.vectorized_core = None
            self.performance_monitor = None
            self.error_handler = None
            self.step08_selector = None
            self.unified_selector = None
    
    def _create_step08_config(self) -> Dict[str, Any]:
        """Create configuration for Step08 advanced feature selection."""
        return {
            'per_regime_feature_selection': True,
            'adaptive_feature_selection_per_regime': True,
            'use_m1_optimizations': True,
            'enable_gpu_acceleration': self.config.get('enable_gpu_acceleration', False),
            'memory_limit_gb': self.config.get('memory_limit_gb', 8.0),
            'max_workers': self.config.get('max_workers', 4),
            'feature_selection_method': 'mutual_info',
            'redundancy_threshold': self.correlation_threshold,
            'interpretability_weight': 0.3,
            'min_features': self.min_features,
            'max_features': self.max_features,
            'target_feature_count': self.target_feature_count
        }
    
    @handles_errors("Advanced fractional feature selection")
    @validate_feature_data_quality
    def select_features_advanced(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Advanced feature selection using Step08 utilities and hardware optimizations.
        
        Args:
            features: Input features DataFrame
            labels: Fractional labels Series
            hmm_regime: HMM regime label (optional)
            
        Returns:
            Dictionary with selected features and selection metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting advanced fractional feature selection (regime: {hmm_regime})")
            self.logger.info(f"📊 Input: {len(features.columns)} features, {len(features)} samples")
            
            # Validate inputs
            if features.empty or labels.empty:
                raise ValueError("Features and labels cannot be empty")
            
            # Use memory optimizer if available
            if self.memory_optimizer:
                with self.memory_optimizer.memory_context():
                    return self._execute_advanced_selection(features, labels, hmm_regime, start_time)
            else:
                return self._execute_advanced_selection(features, labels, hmm_regime, start_time)
                
        except Exception as e:
            self.logger.error(f"❌ Advanced feature selection failed: {e}")
            # Fallback to basic selection
            return self.select_features_basic(features, labels, hmm_regime)
    
    def _execute_advanced_selection(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str], 
        start_time: float
    ) -> Dict[str, Any]:
        """Execute advanced feature selection with all optimizations."""
        try:
            # Use Step08 advanced selector if available
            if self.step08_selector and hmm_regime:
                self.logger.info("🔧 Using Step08 advanced feature selection")
                return self._use_step08_selector(features, labels, hmm_regime, start_time)
            
            # Use unified selector if available
            elif self.unified_selector:
                self.logger.info("🔧 Using unified feature selection")
                return self._use_unified_selector(features, labels, hmm_regime, start_time)
            
            # Use vectorized processing if available
            elif self.vectorized_core:
                self.logger.info("🔧 Using vectorized feature selection")
                return self._use_vectorized_selection(features, labels, hmm_regime, start_time)
            
            # Fallback to basic selection
            else:
                self.logger.info("🔧 Using basic feature selection")
                return self.select_features_basic(features, labels, hmm_regime)
                
        except Exception as e:
            self.logger.warning(f"Advanced selection failed, falling back to basic: {e}")
            return self.select_features_basic(features, labels, hmm_regime)
    
    def _use_step08_selector(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """Use Step08 advanced feature selection."""
        try:
            # Create training input format for Step08
            training_input = {
                'features': features,
                'labels': labels,
                'regime': hmm_regime
            }
            
            # Execute Step08 selection
            result = self.step08_selector.execute_per_regime_feature_selection(
                symbol="FRACTIONAL",
                exchange="FRACTIONAL",
                timeframe="1D",
                data_dir="",
                force_rerun=True,
                regime_id=int(hmm_regime.split('_')[-1]) if hmm_regime else 0
            )
            
            # Convert result to our format
            return {
                'selected_features': features,  # Step08 handles selection internally
                'selection_scores': {},
                'combined_scores': {},
                'selection_metrics': {
                    'processing_time': time.time() - start_time,
                    'method': 'step08_advanced',
                    'regime': hmm_regime
                },
                'processing_time': time.time() - start_time,
                'hmm_regime': hmm_regime
            }
            
        except Exception as e:
            self.logger.warning(f"Step08 selection failed: {e}")
            return self.select_features_basic(features, labels, hmm_regime)
    
    def _use_unified_selector(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str], 
        start_time: float
    ) -> Dict[str, Any]:
        """Use unified feature selection."""
        try:
            # Use unified selector methods
            result = self.unified_selector.execute_unified_feature_selection(
                features, labels, regime=hmm_regime
            )
            
            return {
                'selected_features': result.get('selected_features', features),
                'selection_scores': result.get('selection_scores', {}),
                'combined_scores': result.get('combined_scores', {}),
                'selection_metrics': {
                    'processing_time': time.time() - start_time,
                    'method': 'unified',
                    'regime': hmm_regime
                },
                'processing_time': time.time() - start_time,
                'hmm_regime': hmm_regime
            }
            
        except Exception as e:
            self.logger.warning(f"Unified selection failed: {e}")
            return self.select_features_basic(features, labels, hmm_regime)
    
    def _use_vectorized_selection(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str], 
        start_time: float
    ) -> Dict[str, Any]:
        """Use vectorized processing for feature selection."""
        try:
            # Use vectorized core for parallel processing
            with self.vectorized_core.vectorized_context():
                # Process features in parallel chunks
                if self.parallel_optimizer:
                    return self._parallel_feature_selection(features, labels, hmm_regime, start_time)
                else:
                    return self.select_features_basic(features, labels, hmm_regime)
                    
        except Exception as e:
            self.logger.warning(f"Vectorized selection failed: {e}")
            return self.select_features_basic(features, labels, hmm_regime)
    
    def _parallel_feature_selection(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str], 
        start_time: float
    ) -> Dict[str, Any]:
        """Use parallel processing for feature selection."""
        try:
            # Split features into chunks for parallel processing
            feature_chunks = self.parallel_optimizer.chunk_dataframe(features, chunk_size=100)
            
            # Process chunks in parallel
            results = []
            for chunk in feature_chunks:
                chunk_result = self.select_features_basic(chunk, labels, hmm_regime)
                results.append(chunk_result)
            
            # Combine results
            combined_features = pd.concat([r['selected_features'] for r in results], axis=1)
            combined_scores = {}
            for r in results:
                combined_scores.update(r.get('combined_scores', {}))
            
            return {
                'selected_features': combined_features,
                'selection_scores': {},
                'combined_scores': combined_scores,
                'selection_metrics': {
                    'processing_time': time.time() - start_time,
                    'method': 'parallel',
                    'regime': hmm_regime,
                    'chunks_processed': len(feature_chunks)
                },
                'processing_time': time.time() - start_time,
                'hmm_regime': hmm_regime
            }
            
        except Exception as e:
            self.logger.warning(f"Parallel selection failed: {e}")
            return self.select_features_basic(features, labels, hmm_regime)
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str] = None,
        use_advanced: bool = True
    ) -> Dict[str, Any]:
        """Main feature selection method that chooses between advanced and basic selection.
        
        Args:
            features: Input features DataFrame
            labels: Fractional labels Series
            hmm_regime: HMM regime label (optional)
            use_advanced: Whether to use advanced selection methods
            
        Returns:
            Dictionary with selected features and selection metrics
        """
        if use_advanced and self._has_advanced_utilities():
            return self.select_features_advanced(features, labels, hmm_regime)
        else:
            return self.select_features_basic(features, labels, hmm_regime)
    
    def _has_advanced_utilities(self) -> bool:
        """Check if advanced utilities are available."""
        return any([
            self.step08_selector,
            self.unified_selector,
            self.vectorized_core,
            self.parallel_optimizer,
            self.memory_optimizer
        ])
    
    @handles_errors("Fractional feature selection")
    @validate_feature_data_quality
    def select_features_basic(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Select optimal features for given labels and HMM regime."
        
        Args:
            features: Input features DataFrame
            labels: Fractional labels Series
            hmm_regime: HMM regime label (optional)
            
        Returns:
            Dictionary with selected features and selection metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Starting fractional feature selection (regime: {hmm_regime})")
            self.logger.info(f"📊 Input: {len(features.columns)} features, {len(features)} samples")
            
            # Validate inputs
            if features.empty or labels.empty:
                raise ValueError("Features and labels cannot be empty")
            
            # Align features and labels
            aligned_features, aligned_labels = self._align_data(features, labels)
            
            # Calculate individual selection scores
            selection_scores = {}
            
            if 'correlation' in self.selection_methods:
                selection_scores['correlation'] = self._calculate_correlation_scores(aligned_features, aligned_labels)
            
            if 'importance' in self.selection_methods:
                selection_scores['importance'] = self._calculate_importance_scores(aligned_features, aligned_labels)
            
            if 'stability' in self.selection_methods:
                selection_scores['stability'] = self._calculate_stability_scores(aligned_features)
            
            if 'diversity' in self.selection_methods:
                selection_scores['diversity'] = self._calculate_diversity_scores(aligned_features)
            
            if 'label_alignment' in self.selection_methods:
                selection_scores['label_alignment'] = self._calculate_label_alignment_scores(aligned_features, aligned_labels)
            
            # Combine scores
            combined_scores = self._combine_selection_scores(selection_scores)
            
            # Apply multicollinearity reduction
            reduced_features = self._reduce_multicollinearity(aligned_features, combined_scores)
            
            # Select final features
            selected_features = self._select_final_features(reduced_features, combined_scores)
            
            # Calculate selection metrics
            selection_metrics = self._calculate_selection_metrics(
                aligned_features, selected_features, aligned_labels, hmm_regime
            )
            
            # Track selection history
            self._track_selection_history(
                features, selected_features, selection_metrics, hmm_regime, time.time() - start_time
            )
            
            self.logger.info(f"✅ Feature selection complete: {len(selected_features.columns)} features selected")
            
            return {
                'selected_features': selected_features,
                'selection_scores': selection_scores,
                'combined_scores': combined_scores,
                'selection_metrics': selection_metrics,
                'processing_time': time.time() - start_time,
                'hmm_regime': hmm_regime
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            raise
    
    def _align_data(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and labels data."
        
        Args:
            features: Features DataFrame
            labels: Labels Series
            
        Returns:
            Tuple of aligned features and labels
        """
        # Find common index
        common_index = features.index.intersection(labels.index)
        
        if len(common_index) == 0:
            raise ValueError("No common index between features and labels")
        
        # Align data
        aligned_features = features.loc[common_index]
        aligned_labels = labels.loc[common_index]
        
        # Remove any remaining NaN values
        valid_mask = ~(aligned_features.isnull().any(axis = 1) | aligned_labels.isnull())
        aligned_features = aligned_features.loc[valid_mask]
        aligned_labels = aligned_labels.loc[valid_mask]
        
        self.logger.info(f"📊 Aligned data: {len(aligned_features)} samples")
        
        return aligned_features, aligned_labels
    
    def _calculate_correlation_scores(self, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate correlation-based feature scores."
        
        Args:
            features: Features DataFrame
            labels: Labels Series
            
        Returns:
            Series with correlation scores
        """
        try:
            # Calculate absolute correlations
            correlations = []
            for col in features.columns:
                corr = abs(features[col].corr(labels))
                correlations.append(corr if not pd.isna(corr) else 0.0)
            
            correlation_scores = pd.Series(correlations, index = features.columns)
            
            # Normalize to 0-1 range
            if correlation_scores.max() > 0:
                correlation_scores = correlation_scores / correlation_scores.max()
            
            self.logger.info(f"📊 Correlation scores calculated for {len(features.columns)} features")
            
            return correlation_scores
            
        except Exception as e:
            self.logger.warning(f"Error calculating correlation scores: {e}")
            return pd.Series(0.5, index = features.columns)
    
    def _calculate_importance_scores(self, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate feature importance scores using multiple methods."
        
        Args:
            features: Features DataFrame
            labels: Labels Series
            
        Returns:
            Series with importance scores
        """
        try:
            # Use multiple importance methods
            importance_scores = {}
            
            # 1. F-regression scores
            importance_scores['f_regression'] = self._calculate_f_regression_scores(features, labels)
            
            # 2. Mutual information scores
            importance_scores['mutual_info'] = self._calculate_mutual_info_scores(features, labels)
            
            # 3. Random Forest importance
            importance_scores['random_forest'] = self._calculate_random_forest_scores(features, labels)
            
            # Combine importance scores
            combined_importance = pd.Series(0.0, index = features.columns)
            for method, scores in importance_scores.items():
                if scores.max() > 0:
                    normalized_scores = scores / scores.max()
                    combined_importance += normalized_scores
            
            # Average the scores
            combined_importance = combined_importance / len(importance_scores)
            
            self.logger.info(f"📊 Importance scores calculated using {len(importance_scores)} methods")
            
            return combined_importance
            
        except Exception as e:
            self.logger.warning(f"Error calculating importance scores: {e}")
            return pd.Series(0.5, index = features.columns)
    
    def _calculate_stability_scores(self, features: pd.DataFrame) -> pd.Series:
        """Calculate feature stability scores."
        
        Args:
            features: Features DataFrame
            
        Returns:
            Series with stability scores
        """
        try:
            stability_scores = []
            
            for col in features.columns:
                feature_series = features[col].dropna()
                
                if len(feature_series) < 50:
                    stability_scores.append(0.5)
                    continue
                
                # Calculate rolling variance stability
                window_size = min(50, len(feature_series) // 4)
                rolling_var = feature_series.rolling(window = window_size, min_periods = 10).var()
                
                if rolling_var.mean() > 0:
                    # Lower variance in rolling variance indicates more stability
                    var_consistency = 1.0 - (rolling_var.std() / rolling_var.mean())
                    stability_score = max(0.0, var_consistency)
                else:
                    stability_score = 0.5
                
                stability_scores.append(stability_score)
            
            stability_series = pd.Series(stability_scores, index = features.columns)
            
            self.logger.info(f"📊 Stability scores calculated for {len(features.columns)} features")
            
            return stability_series
            
        except Exception as e:
            self.logger.warning(f"Error calculating stability scores: {e}")
            return pd.Series(0.5, index = features.columns)
    
    def _calculate_diversity_scores(self, features: pd.DataFrame) -> pd.Series:
        """Calculate feature diversity scores."
        
        Args:
            features: Features DataFrame
            
        Returns:
            Series with diversity scores
        """
        try:
            diversity_scores = []
            
            for col in features.columns:
                feature_series = features[col].dropna()
                
                if len(feature_series) == 0:
                    diversity_scores.append(0.0)
                    continue
                
                # Calculate diversity metrics
                unique_ratio = feature_series.nunique() / len(feature_series)
                non_zero_ratio = (feature_series != 0).sum() / len(feature_series)
                
                # Entropy-like measure
                value_counts = feature_series.value_counts(normalize = True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                max_entropy = np.log2(len(value_counts) + 1e-10)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                
                # Combine diversity metrics
                diversity_score = (unique_ratio + non_zero_ratio + normalized_entropy) / 3
                diversity_scores.append(diversity_score)
            
            diversity_series = pd.Series(diversity_scores, index = features.columns)
            
            self.logger.info(f"📊 Diversity scores calculated for {len(features.columns)} features")
            
            return diversity_series
            
        except Exception as e:
            self.logger.warning(f"Error calculating diversity scores: {e}")
            return pd.Series(0.5, index = features.columns)
    
    def _calculate_label_alignment_scores(self, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate label alignment scores for fractional labels."
        
        Args:
            features: Features DataFrame
            labels: Fractional labels Series
            
        Returns:
            Series with label alignment scores
        """
        try:
            alignment_scores = []
            
            for col in features.columns:
                feature_series = features[col].dropna()
                
                if len(feature_series) < self.alignment_window:
                    alignment_scores.append(0.5)
                    continue
                
                # Calculate rolling correlation with labels
                rolling_correlations = []
                
                for i in range(self.alignment_window, len(feature_series)):
                    window_features = feature_series.iloc[i-self.alignment_window:i]
                    window_labels = labels.iloc[i-self.alignment_window:i]
                    
                    corr = abs(window_features.corr(window_labels))
                    if not pd.isna(corr):
                        rolling_correlations.append(corr)
                
                if rolling_correlations:
                    # Higher average correlation indicates better alignment
                    avg_correlation = np.mean(rolling_correlations)
                    alignment_score = min(1.0, avg_correlation * 2)  # Scale to 0-1
                else:
                    alignment_score = 0.5
                
                alignment_scores.append(alignment_score)
            
            alignment_series = pd.Series(alignment_scores, index = features.columns)
            
            self.logger.info(f"📊 Label alignment scores calculated for {len(features.columns)} features")
            
            return alignment_series
            
        except Exception as e:
            self.logger.warning(f"Error calculating label alignment scores: {e}")
            return pd.Series(0.5, index = features.columns)
    
    def _combine_selection_scores(self, selection_scores: Dict[str, pd.Series]) -> pd.Series:
        """Combine individual selection scores."
        
        Args:
            selection_scores: Dictionary of selection scores
            
        Returns:
            Combined scores Series
        """
        try:
            combined_scores = pd.Series(0.0, index = list(selection_scores.values())[0].index)
            
            for method, scores in selection_scores.items():
                if method in self.method_weights:
                    weight = self.method_weights[method]
                    combined_scores += weight * scores
            
            # Normalize to 0-1 range
            if combined_scores.max() > 0:
                combined_scores = combined_scores / combined_scores.max()
            
            self.logger.info(f"📊 Combined selection scores calculated using {len(selection_scores)} methods")
            
            return combined_scores
            
        except Exception as e:
            self.logger.warning(f"Error combining selection scores: {e}")
            return pd.Series(0.5, index = list(selection_scores.values())[0].index)
    
    def _reduce_multicollinearity(self, features: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
        """Reduce multicollinearity in features."
        
        Args:
            features: Features DataFrame
            scores: Feature scores Series
            
        Returns:
            Features DataFrame with reduced multicollinearity
        """
        try:
            # Calculate correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
            
            # Get pairs of highly correlated features
            high_corr_pairs = []
            for col in upper_tri.columns:
                high_corr_features = upper_tri[col][upper_tri[col] > self.correlation_threshold]
                for feature in high_corr_features.index:
                    high_corr_pairs.append((col, feature))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            
            for feature1, feature2 in high_corr_pairs:
                # Keep the feature with higher score
                if scores[feature1] >= scores[feature2]:
                    features_to_remove.add(feature2)
                else:
                    features_to_remove.add(feature1)
            
            # Remove highly correlated features
            reduced_features = features.drop(columns = list(features_to_remove))
            
            self.logger.info(f"📊 Multicollinearity reduction: removed {len(features_to_remove)} features")
            
            return reduced_features
            
        except Exception as e:
            self.logger.warning(f"Error reducing multicollinearity: {e}")
            return features
    
    def _select_final_features(self, features: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
        """Select final features based on scores and constraints."
        
        Args:
            features: Features DataFrame
            scores: Feature scores Series
            
        Returns:
            Selected features DataFrame
        """
        try:
            # Align scores with features
            aligned_scores = scores[features.columns]
            
            # Sort features by score
            sorted_features = aligned_scores.sort_values(ascending = False)
            
            # Determine number of features to select
            n_features = min(
                max(self.min_features, self.target_feature_count),
                min(self.max_features, len(features.columns))
            )
            
            # Select top features
            selected_feature_names = sorted_features.head(n_features).index
            selected_features = features[selected_feature_names]
            
            self.logger.info(f"📊 Selected {len(selected_features.columns)} features out of {len(features.columns)}")
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Error selecting final features: {e}")
            return features
    
    def _calculate_selection_metrics(
        self, 
        original_features: pd.DataFrame, 
        selected_features: pd.DataFrame, 
        labels: pd.Series, 
        hmm_regime: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate selection performance metrics."
        
        Args:
            original_features: Original features DataFrame
            selected_features: Selected features DataFrame
            labels: Labels Series
            hmm_regime: HMM regime label
            
        Returns:
            Dictionary with selection metrics
        """
        try:
            metrics = {
                'original_feature_count': len(original_features.columns),
                'selected_feature_count': len(selected_features.columns),
                'reduction_ratio': 1 - (len(selected_features.columns) / len(original_features.columns)),
                'hmm_regime': hmm_regime
            }
            
            # Calculate feature quality metrics
            if not selected_features.empty:
                # Average feature variance
                feature_variances = selected_features.var()
                metrics['avg_feature_variance'] = feature_variances.mean()
                metrics['feature_variance_std'] = feature_variances.std()
                
                # Feature-label correlations
                correlations = []
                for col in selected_features.columns:
                    corr = abs(selected_features[col].corr(labels))
                    if not pd.isna(corr):
                        correlations.append(corr)
                
                if correlations:
                    metrics['avg_feature_label_correlation'] = np.mean(correlations)
                    metrics['max_feature_label_correlation'] = np.max(correlations)
                    metrics['min_feature_label_correlation'] = np.min(correlations)
                
                # Feature diversity
                diversity_scores = []
                for col in selected_features.columns:
                    feature_series = selected_features[col].dropna()
                    unique_ratio = feature_series.nunique() / len(feature_series)
                    diversity_scores.append(unique_ratio)
                
                metrics['avg_feature_diversity'] = np.mean(diversity_scores)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating selection metrics: {e}")
            return {
                'original_feature_count': len(original_features.columns),
                'selected_feature_count': len(selected_features.columns),
                'error': str(e)
            }
    
    def _track_selection_history(
        self, 
        original_features: pd.DataFrame, 
        selected_features: pd.DataFrame, 
        metrics: Dict[str, Any], 
        hmm_regime: Optional[str], 
        processing_time: float
    ):
        """Track feature selection history."
        
        Args:
            original_features: Original features DataFrame
            selected_features: Selected features DataFrame
            metrics: Selection metrics
            hmm_regime: HMM regime label
            processing_time: Processing time
        """
        try:
            history_entry = {
                'timestamp': pd.Timestamp.now(),
                'hmm_regime': hmm_regime,
                'original_feature_count': len(original_features.columns),
                'selected_feature_count': len(selected_features.columns),
                'reduction_ratio': metrics.get('reduction_ratio', 0.0),
                'avg_feature_label_correlation': metrics.get('avg_feature_label_correlation', 0.0),
                'avg_feature_diversity': metrics.get('avg_feature_diversity', 0.0),
                'processing_time': processing_time
            }
            
            self.selection_history.append(history_entry)
            
        except Exception as e:
            self.logger.warning(f"Error tracking selection history: {e}")
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection performance."
        
        Returns:
            Dictionary with selection summary
        """
        if not self.selection_history:
            return {'message': 'No selection history available'}
        
        try:
            # Aggregate metrics
            reduction_ratios = [h['reduction_ratio'] for h in self.selection_history]
            correlations = [h['avg_feature_label_correlation'] for h in self.selection_history]
            diversities = [h['avg_feature_diversity'] for h in self.selection_history]
            processing_times = [h['processing_time'] for h in self.selection_history]
            
            # Regime-specific metrics
            regime_performance = {}
            for record in self.selection_history:
                regime = record['hmm_regime']
                if regime not in regime_performance:
                    regime_performance[regime] = []
                regime_performance[regime].append(record)
            
            summary = {
                'total_selections': len(self.selection_history),
                'avg_reduction_ratio': np.mean(reduction_ratios),
                'avg_correlation': np.mean(correlations),
                'avg_diversity': np.mean(diversities),
                'avg_processing_time': np.mean(processing_times),
                'regime_performance': {}
            }
            
            # Calculate regime-specific summaries
            for regime, records in regime_performance.items():
                regime_reductions = [r['reduction_ratio'] for r in records]
                regime_correlations = [r['avg_feature_label_correlation'] for r in records]
                
                summary['regime_performance'][regime] = {
                    'selections': len(records),
                    'avg_reduction_ratio': np.mean(regime_reductions),
                    'avg_correlation': np.mean(regime_correlations)
                }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Error generating selection summary: {e}")
            return {'error': str(e)}
    
    def export_selection_report(self, output_dir: str = "data/fractional_performance/feature_selection") -> str:
        """Export feature selection report to file."
        
        Args:
            output_dir: Output directory for the report
            
        Returns:
            Path to the exported report
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents = True, exist_ok = True)
            
            # Generate selection summary
            summary = self.get_selection_summary()
            
            # Export to JSON
            report_file = output_path / "feature_selection_performance.json"
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent = 2, default = str)
            
            # Export detailed history
            history_file = output_path / "selection_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.selection_history, f, indent = 2, default = str)
            
            self.logger.info(f"📊 Feature selection report exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export feature selection report: {e}")
            return ""

# Configuration helper
def get_fractional_feature_selector_config(
    min_features: int = 10,
    max_features: int = 50,
    target_feature_count: int = 30,
    selection_methods: Optional[List[str]] = None,
    method_weights: Optional[Dict[str, float]] = None,
    correlation_threshold: float = 0.85,
    vif_threshold: float = 5.0,
    alignment_window: int = 100,
    alignment_threshold: float = 0.1
) -> Dict[str, Any]:
    """Get configuration for fractional feature selector."
    
    Args:
        min_features: Minimum number of features to select
        max_features: Maximum number of features to select
        target_feature_count: Target number of features
        selection_methods: List of selection methods to use
        method_weights: Weights for each selection method
        correlation_threshold: Threshold for multicollinearity reduction
        vif_threshold: VIF threshold for multicollinearity
        alignment_window: Window size for label alignment calculation
        alignment_threshold: Threshold for label alignment
        
    Returns:
        Configuration dictionary
    """
    if selection_methods is None:
        selection_methods = ['correlation', 'importance', 'stability', 'diversity', 'label_alignment']
    
    if method_weights is None:
        method_weights = {
            'correlation': 0.25,
            'importance': 0.25,
            'stability': 0.15,
            'diversity': 0.15,
            'label_alignment': 0.20
        }
    
    return {
        'min_features': min_features,
        'max_features': max_features,
        'target_feature_count': target_feature_count,
        'selection_methods': selection_methods,
        'method_weights': method_weights,
        'correlation_threshold': correlation_threshold,
        'vif_threshold': vif_threshold,
        'alignment_window': alignment_window,
        'alignment_threshold': alignment_threshold
    }

    def _calculate_f_regression_scores(self, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate F-regression scores safely.
        
        Args:
            features: Features DataFrame
            labels: Labels Series
            
        Returns:
            F-regression scores Series
        """
        try:
            # Handle NaN values
            clean_features = features.fillna(features.mean())
            clean_labels = labels.fillna(labels.mean())
            
            f_scores, _ = f_regression(clean_features, clean_labels)
            return pd.Series(f_scores, index=features.columns)
        except Exception as e:
            self.logger.warning(f"F-regression calculation failed: {e}")
            return pd.Series(0.0, index=features.columns)

    def _calculate_mutual_info_scores(self, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate mutual information scores safely.
        
        Args:
            features: Features DataFrame
            labels: Labels Series
            
        Returns:
            Mutual information scores Series
        """
        try:
            # Handle NaN values
            clean_features = features.fillna(features.mean())
            clean_labels = labels.fillna(labels.mean())
            
            mi_scores = mutual_info_regression(clean_features, clean_labels, random_state=42)
            return pd.Series(mi_scores, index=features.columns)
        except Exception as e:
            self.logger.warning(f"Mutual information calculation failed: {e}")
            return pd.Series(0.0, index=features.columns)

    def _calculate_random_forest_scores(self, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate Random Forest importance scores safely.
        
        Args:
            features: Features DataFrame
            labels: Labels Series
            
        Returns:
            Random Forest importance scores Series
        """
        try:
            # Handle NaN values
            clean_features = features.fillna(features.mean())
            clean_labels = labels.fillna(labels.mean())
            
            # Use smaller number of estimators for speed
            rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1, max_depth=10)
            rf.fit(clean_features, clean_labels)
            return pd.Series(rf.feature_importances_, index=features.columns)
        except Exception as e:
            self.logger.warning(f"Random Forest importance calculation failed: {e}")
            return pd.Series(0.0, index=features.columns)