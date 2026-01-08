"""
TreeSHAP Interaction Analyzer for Feature Discovery

This module provides comprehensive TreeSHAP-based analysis for feature importance,
interaction discovery, and feature pruning in financial machine learning pipelines.

Key Features:
- TreeSHAP feature importance ranking
- Interaction feature discovery and ranking
- Feature redundancy detection and pruning
- Feature category diversity analysis
- Automatic feature selection based on SHAP importance
- Integration with GMM enhanced features pipeline

Usage:
    analyzer = SHAPInteractionAnalyzer()
    results = analyzer.analyze_features(features_df, target_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import warnings
from pathlib import Path
import time
import logging
from dataclasses import dataclass, asdict

# Try to import SHAP and LightGBM
try:
    import shap
    from shap import TreeExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    TreeExplainer = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = logging.getLogger(__name__)


@dataclass
class SHAPFeatureResult:
    """Container for SHAP feature analysis results."""
    feature_name: str
    shap_importance: float
    builtin_importance: float
    correlation_with_target: float
    feature_category: str
    redundancy_score: float
    diversity_score: float
    composite_score: float
    selected: bool = False
    interaction_partners: List[str] = None
    
    def __post_init__(self):
        if self.interaction_partners is None:
            self.interaction_partners = []


@dataclass
class SHAPInteractionResult:
    """Container for SHAP interaction analysis results."""
    feature_pair: str
    interaction_strength: float
    interaction_type: str  # 'synergistic', 'antagonistic', 'neutral'
    main_effect_ratio: float
    frequency: float
    significance: float


class SHAPInteractionAnalyzer:
    """
    Comprehensive TreeSHAP-based feature and interaction analyzer.
    
    This class provides advanced analysis capabilities for feature importance,
    interaction discovery, and feature selection using TreeSHAP methodology.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SHAP Interaction Analyzer."""
        self.config = config or {}
        
        # Model configuration
        self.n_estimators = self.config.get('n_estimators', 200)
        self.max_depth = self.config.get('max_depth', 8)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.num_leaves = self.config.get('num_leaves', 31)
        self.random_state = self.config.get('random_state', 42)
        
        # Analysis configuration
        self.max_samples = self.config.get('max_samples', 50000)
        self.interaction_sample_size = self.config.get('interaction_sample_size', 1000)
        self.importance_threshold = self.config.get('importance_threshold', 0.01)
        self.interaction_threshold = self.config.get('interaction_threshold', 0.005)
        self.redundancy_threshold = self.config.get('redundancy_threshold', 0.85)
        
        # Feature selection configuration
        self.max_features = self.config.get('max_features', 50)
        self.diversity_weight = self.config.get('diversity_weight', 0.2)
        self.importance_weight = self.config.get('importance_weight', 0.6)
        self.redundancy_weight = self.config.get('redundancy_weight', 0.2)
        
        # Results storage
        self.feature_results: List[SHAPFeatureResult] = []
        self.interaction_results: List[SHAPInteractionResult] = []
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_categories = {}
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        tprint_info("✅ SHAP and LightGBM dependencies verified")
    
    def analyze_features(self, 
                         features_df: pd.DataFrame, 
                         target_df: pd.DataFrame,
                         feature_categories: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive SHAP analysis of features.
        
        Args:
            features_df: Feature matrix (samples x features)
            target_df: Target matrix (samples x targets)
            feature_categories: Dictionary mapping feature names to categories
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        tprint_info("🔍 Starting comprehensive SHAP feature analysis...")
        
        try:
            # 1. Preprocess data
            X, y, feature_names = self._preprocess_data(features_df, target_df)
            
            # 2. Train model
            self.model = self._train_model(X, y)
            
            # 3. Calculate SHAP values
            self.shap_values = self._calculate_shap_values(X)
            
            # 4. Analyze feature importance
            self.feature_results = self._analyze_feature_importance(
                X, y, feature_names, feature_categories
            )
            
            # 5. Discover interactions
            self.interaction_results = self._discover_interactions(X, feature_names)
            
            # 6. Apply redundancy filtering
            filtered_features = self._apply_redundancy_filtering(X)
            
            # 7. Select final features
            selected_features = self._select_final_features(filtered_features)
            
            # 8. Prepare results
            execution_time = time.time() - start_time
            
            results = {
                'feature_analysis': {
                    'total_features': len(feature_names),
                    'selected_features': len(selected_features),
                    'feature_results': [asdict(fr) for fr in self.feature_results],
                    'top_features': selected_features[:20]  # Top 20 features
                },
                'interaction_analysis': {
                    'total_interactions': len(self.interaction_results),
                    'top_interactions': [asdict(ir) for ir in self.interaction_results[:20]],
                    'interaction_results': [asdict(ir) for ir in self.interaction_results]
                },
                'model_performance': {
                    'train_score': self.model.score(X, y),
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth
                },
                'execution_time': execution_time,
                'success': True
            }
            
            tprint_success(f"✅ SHAP analysis completed in {execution_time:.2f}s")
            tprint_info(f"📊 Analyzed {len(feature_names)} features, selected {len(selected_features)}")
            tprint_info(f"🔗 Discovered {len(self.interaction_results)} significant interactions")
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ SHAP analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _preprocess_data(self, 
                        features_df: pd.DataFrame, 
                        target_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess data for SHAP analysis."""
        # Align data
        common_index = features_df.index.intersection(target_df.index)
        X_aligned = features_df.loc[common_index]
        y_aligned = target_df.loc[common_index]
        
        # Handle multi-target case
        if len(y_aligned.shape) > 1 and y_aligned.shape[1] > 1:
            # Use first target or create composite
            y = y_aligned.iloc[:, 0].values
        else:
            y = y_aligned.values.flatten()
        
        # Convert to numpy
        X = X_aligned.values
        feature_names = list(X_aligned.columns)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Sample if too large
        if len(X) > self.max_samples:
            tprint_info(f"📊 Sampling {self.max_samples} rows from {len(X)} for efficiency")
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        tprint_info(f"📊 Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> lgb.LGBMModel:
        """Train LightGBM model for SHAP analysis."""
        tprint_info("🌳 Training LightGBM model for SHAP analysis...")
        
        # Determine problem type
        if len(np.unique(y)) <= 10:
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        
        model.fit(X, y)
        self.explainer = TreeExplainer(model)
        
        tprint_success(f"✅ Model trained with {self.n_estimators} estimators")
        
        return model
    
    def _calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values."""
        tprint_info("🔍 Calculating SHAP values...")
        
        # Sample for SHAP calculation if needed
        sample_size = min(len(X), self.interaction_sample_size)
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                shap_values = shap_values[0] + shap_values[1]
            else:  # Multi-class
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        
        tprint_success(f"✅ SHAP values calculated for {sample_size} samples")
        
        return shap_values
    
    def _analyze_feature_importance(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  feature_names: List[str],
                                  feature_categories: Optional[Dict[str, str]] = None) -> List[SHAPFeatureResult]:
        """Analyze feature importance using SHAP."""
        tprint_info("📊 Analyzing feature importance...")
        
        # Calculate SHAP importance
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        builtin_importance = self.model.feature_importances_
        
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        # Create feature results
        feature_results = []
        for i, name in enumerate(feature_names):
            category = feature_categories.get(name, self._infer_category(name)) if feature_categories else self._infer_category(name)
            
            result = SHAPFeatureResult(
                feature_name=name,
                shap_importance=float(shap_importance[i]),
                builtin_importance=float(builtin_importance[i]),
                correlation_with_target=float(correlations[i]),
                feature_category=category,
                redundancy_score=0.0,  # Will be calculated later
                diversity_score=0.0,   # Will be calculated later
                composite_score=0.0   # Will be calculated later
            )
            feature_results.append(result)
        
        # Calculate redundancy and diversity scores
        feature_results = self._calculate_redundancy_scores(feature_results, X)
        feature_results = self._calculate_diversity_scores(feature_results)
        
        # Calculate composite scores
        feature_results = self._calculate_composite_scores(feature_results)
        
        # Sort by composite score
        feature_results.sort(key=lambda x: x.composite_score, reverse=True)
        
        tprint_success(f"✅ Analyzed importance for {len(feature_results)} features")
        
        return feature_results
    
    def _infer_category(self, feature_name: str) -> str:
        """Infer feature category from name."""
        name_lower = feature_name.lower()
        
        if any(x in name_lower for x in ['price', 'close', 'open', 'high', 'low', 'return']):
            return 'price'
        elif any(x in name_lower for x in ['volume', 'vol']):
            return 'volume'
        elif any(x in name_lower for x in ['volatility', 'std', 'var', 'atr']):
            return 'volatility'
        elif any(x in name_lower for x in ['momentum', 'rsi', 'macd', 'cci']):
            return 'momentum'
        elif any(x in name_lower for x in ['trend', 'sma', 'ema', 'ma']):
            return 'trend'
        elif any(x in name_lower for x in ['gmm', 'regime', 'state', 'cluster']):
            return 'regime'
        elif any(x in name_lower for x in ['entropy', 'lz', 'complexity']):
            return 'entropy'
        elif any(x in name_lower for x in ['shock', 'jump', 'velocity', 'accel']):
            return 'kinematics'
        else:
            return 'other'
    
    def _calculate_redundancy_scores(self, 
                                   feature_results: List[SHAPFeatureResult], 
                                   X: np.ndarray) -> List[SHAPFeatureResult]:
        """Calculate redundancy scores based on feature correlations."""
        tprint_info("🔄 Calculating redundancy scores...")
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Calculate redundancy scores
        for i, result in enumerate(feature_results):
            # Average correlation with other features
            correlations_with_others = []
            for j in range(len(feature_results)):
                if i != j:
                    corr = abs(corr_matrix[i, j])
                    correlations_with_others.append(corr)
            
            # Redundancy score = average high correlation
            high_correlations = [c for c in correlations_with_others if c > self.redundancy_threshold]
            result.redundancy_score = np.mean(high_correlations) if high_correlations else 0.0
        
        return feature_results
    
    def _calculate_diversity_scores(self, feature_results: List[SHAPFeatureResult]) -> List[SHAPFeatureResult]:
        """Calculate diversity scores based on feature categories."""
        tprint_info("🎯 Calculating diversity scores...")
        
        # Count features per category
        category_counts = {}
        for result in feature_results:
            category_counts[result.feature_category] = category_counts.get(result.feature_category, 0) + 1
        
        # Calculate diversity scores (inverse of category frequency)
        for result in feature_results:
            category_count = category_counts[result.feature_category]
            result.diversity_score = 1.0 / max(category_count, 1)
        
        return feature_results
    
    def _calculate_composite_scores(self, feature_results: List[SHAPFeatureResult]) -> List[SHAPFeatureResult]:
        """Calculate composite scores combining all metrics."""
        tprint_info("⚡ Calculating composite scores...")
        
        # Normalize scores
        max_shap = max([r.shap_importance for r in feature_results]) if feature_results else 1.0
        max_corr = max([r.correlation_with_target for r in feature_results]) if feature_results else 1.0
        max_div = max([r.diversity_score for r in feature_results]) if feature_results else 1.0
        
        for result in feature_results:
            # Normalize scores
            norm_shap = result.shap_importance / max_shap if max_shap > 0 else 0.0
            norm_corr = result.correlation_with_target / max_corr if max_corr > 0 else 0.0
            norm_div = result.diversity_score / max_div if max_div > 0 else 0.0
            
            # Composite score (weighted combination)
            result.composite_score = (
                self.importance_weight * norm_shap +
                0.3 * norm_corr +  # Correlation weight
                self.diversity_weight * norm_div -
                self.redundancy_weight * result.redundancy_score
            )
        
        return feature_results
    
    def _discover_interactions(self, X: np.ndarray, feature_names: List[str]) -> List[SHAPInteractionResult]:
        """Discover feature interactions using SHAP interaction values."""
        tprint_info("🔗 Discovering feature interactions...")
        
        interaction_results = []
        
        try:
            # Sample for interaction calculation
            sample_size = min(len(X), 500)  # Smaller sample for interactions
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            
            # Calculate interaction values
            shap_interactions = self.explainer.shap_interaction_values(X_sample)
            
            # Handle different formats
            if isinstance(shap_interactions, list):
                shap_interactions = shap_interactions[0]  # Use first class
            
            # Calculate interaction strengths
            interaction_strength = np.abs(shap_interactions).mean(axis=0)
            
            # Find significant interactions
            n_features = len(feature_names)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    strength = interaction_strength[i, j]
                    
                    if strength > self.interaction_threshold:
                        # Determine interaction type
                        main_effect_i = np.abs(self.shap_values[:, i].mean())
                        main_effect_j = np.abs(self.shap_values[:, j].mean())
                        main_effect_total = main_effect_i + main_effect_j
                        
                        if main_effect_total > 0:
                            main_effect_ratio = strength / main_effect_total
                        else:
                            main_effect_ratio = 0.0
                        
                        # Classify interaction type
                        if main_effect_ratio > 0.5:
                            interaction_type = 'synergistic'
                        elif main_effect_ratio > 0.2:
                            interaction_type = 'moderate'
                        else:
                            interaction_type = 'weak'
                        
                        result = SHAPInteractionResult(
                            feature_pair=f"{feature_names[i]} * {feature_names[j]}",
                            interaction_strength=float(strength),
                            interaction_type=interaction_type,
                            main_effect_ratio=float(main_effect_ratio),
                            frequency=strength / np.sum(interaction_strength),
                            significance=min(strength * 10, 1.0)  # Normalized significance
                        )
                        interaction_results.append(result)
            
            # Sort by interaction strength
            interaction_results.sort(key=lambda x: x.interaction_strength, reverse=True)
            
            tprint_success(f"✅ Discovered {len(interaction_results)} significant interactions")
            
        except Exception as e:
            tprint_warning(f"⚠️ Interaction discovery failed: {e}")
        
        return interaction_results
    
    def _apply_redundancy_filtering(self, X: np.ndarray) -> List[SHAPFeatureResult]:
        """Apply redundancy filtering to feature results."""
        tprint_info("🔄 Applying redundancy filtering...")
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Filter redundant features
        filtered_results = []
        removed_indices = set()
        
        for i, result in enumerate(self.feature_results):
            if i in removed_indices:
                continue
            
            # Check for redundancy with already selected features
            is_redundant = False
            for j, selected_result in enumerate(filtered_results):
                if j in removed_indices:
                    continue
                
                # Find index of selected feature
                selected_idx = self.feature_results.index(selected_result)
                corr = abs(corr_matrix[i, selected_idx])
                
                if corr > self.redundancy_threshold:
                    # Keep the feature with higher composite score
                    if result.composite_score <= selected_result.composite_score:
                        is_redundant = True
                        removed_indices.add(i)
                        break
                    else:
                        # Remove the previously selected feature
                        filtered_results.remove(selected_result)
                        removed_indices.add(selected_idx)
                        break
            
            if not is_redundant:
                filtered_results.append(result)
        
        tprint_info(f"📊 Removed {len(removed_indices)} redundant features")
        tprint_info(f"📊 Remaining features: {len(filtered_results)}")
        
        return filtered_results
    
    def _select_final_features(self, filtered_features: List[SHAPFeatureResult]) -> List[str]:
        """Select final features based on composite scores and diversity."""
        tprint_info(f"🎯 Selecting final {self.max_features} features...")
        
        # Sort by composite score
        filtered_features.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Apply diversity enforcement
        selected_features = []
        category_counts = {}
        
        for result in filtered_features:
            if len(selected_features) >= self.max_features:
                break
            
            # Check category diversity
            category = result.feature_category
            category_count = category_counts.get(category, 0)
            
            # Allow more features from categories with high importance
            max_per_category = max(3, self.max_features // 5)  # At least 3 per category
            
            if category_count < max_per_category:
                selected_features.append(result.feature_name)
                category_counts[category] = category_count + 1
                result.selected = True
        
        tprint_success(f"✅ Selected {len(selected_features)} final features")
        
        # Log category distribution
        for category, count in category_counts.items():
            tprint_info(f"📊 {category}: {count} features")
        
        return selected_features
    
    def generate_interaction_features(self, 
                                   features_df: pd.DataFrame,
                                   top_n_interactions: int = 10) -> pd.DataFrame:
        """
        Generate interaction features based on discovered interactions.
        
        Args:
            features_df: Original feature matrix
            top_n_interactions: Number of top interactions to generate
            
        Returns:
            DataFrame with interaction features added
        """
        if not self.interaction_results:
            tprint_warning("⚠️ No interaction results available")
            return features_df.copy()
        
        tprint_info(f"🔗 Generating top {top_n_interactions} interaction features...")
        
        enhanced_features = features_df.copy()
        
        for i, interaction in enumerate(self.interaction_results[:top_n_interactions]):
            try:
                # Parse feature pair
                feature1, feature2 = interaction.feature_pair.split(' * ')
                
                if feature1 in features_df.columns and feature2 in features_df.columns:
                    # Generate interaction feature
                    interaction_name = f"int_{feature1}_{feature2}"
                    enhanced_features[interaction_name] = (
                        features_df[feature1] * features_df[feature2]
                    )
                    
                    tprint_info(f"✅ Generated: {interaction_name}")
                else:
                    tprint_warning(f"⚠️ Missing features for interaction: {interaction.feature_pair}")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Failed to generate interaction {i}: {e}")
        
        tprint_success(f"✅ Enhanced features: {len(enhanced_features.columns)} total columns")
        
        return enhanced_features
    
    def save_results(self, output_dir: str, timestamp: Optional[str] = None):
        """Save analysis results to files."""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature results
        feature_results_path = output_path / f"shap_feature_results_{timestamp}.json"
        feature_data = [asdict(fr) for fr in self.feature_results]
        with open(feature_results_path, 'w') as f:
            json.dump(feature_data, f, indent=2)
        
        # Save interaction results
        interaction_results_path = output_path / f"shap_interaction_results_{timestamp}.json"
        interaction_data = [asdict(ir) for ir in self.interaction_results]
        with open(interaction_results_path, 'w') as f:
            json.dump(interaction_data, f, indent=2)
        
        # Save summary
        summary_path = output_path / f"shap_analysis_summary_{timestamp}.json"
        summary = {
            'total_features_analyzed': len(self.feature_results),
            'total_interactions_discovered': len(self.interaction_results),
            'config': self.config,
            'timestamp': timestamp
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        tprint_success(f"💾 Results saved to {output_path}")
        
        return {
            'feature_results_path': str(feature_results_path),
            'interaction_results_path': str(interaction_results_path),
            'summary_path': str(summary_path)
        }


# Convenience function for quick analysis
def analyze_features_with_shap(features_df: pd.DataFrame,
                             target_df: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None,
                             output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for comprehensive SHAP feature analysis.
    
    Args:
        features_df: Feature matrix
        target_df: Target matrix
        config: Analysis configuration
        output_dir: Directory to save results
        
    Returns:
        Analysis results dictionary
    """
    analyzer = SHAPInteractionAnalyzer(config)
    results = analyzer.analyze_features(features_df, target_df)
    
    if output_dir and results.get('success'):
        analyzer.save_results(output_dir)
    
    return results


# Export main classes and functions
__all__ = [
    'SHAPInteractionAnalyzer',
    'SHAPFeatureResult',
    'SHAPInteractionResult',
    'analyze_features_with_shap'
]
