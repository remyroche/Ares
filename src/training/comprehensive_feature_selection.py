#!/usr/bin/env python3
"""Comprehensive Feature Selection System.

This module provides intelligent feature selection that uses ALL available features
from the feature engineering pipeline and selects the best features for multi-output
prediction (direction, profit, and price).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.training.steps.step4_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
    ProfitBasedFeatureEngineering
)
from src.utils.logger import system_logger


@dataclass
class FeatureSelectionConfig:
    """Configuration for comprehensive feature selection."""
    
    # Feature engineering settings
    use_profit_features: bool = True
    use_all_existing_features: bool = True
    use_interaction_features: bool = True
    use_polynomial_features: bool = False  # Can be computationally expensive
    
    # Selection methods
    selection_methods: List[str] = None  # Will be set to default methods
    max_features: int = 500
    min_features: int = 50
    
    # Statistical selection
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    mutual_info_threshold: float = 0.01
    
    # Model-based selection
    use_rf_selection: bool = True
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    
    # Dimensionality reduction
    use_pca: bool = False
    pca_explained_variance: float = 0.95
    
    # Multi-output specific
    direction_weight: float = 0.4
    profit_weight: float = 0.3
    price_weight: float = 0.3
    
    # Performance settings
    parallel_processing: bool = True
    memory_efficient: bool = True
    
    def __post_init__(self):
        if self.selection_methods is None:
            self.selection_methods = [
                "variance_threshold",
                "correlation_filter",
                "mutual_info",
                "rf_importance",
                "rfe",
                "ensemble_selection"
            ]


class ComprehensiveFeatureSelector:
    """Comprehensive feature selection system using all available features."""
    
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.logger = system_logger.getChild("ComprehensiveFeatureSelector")
        
        # Initialize profit-based feature engineering
        self.profit_feature_engine = ProfitBasedFeatureEngineering(
            profit_column="potential_profit_pct",
            use_numba=True,
            memory_efficient=True
        )
        
        # Feature selection results
        self.selected_features = []
        self.feature_scores = {}
        self.feature_importance = {}
        self.selection_history = []
        
        self.logger.info("🔧 Comprehensive feature selector initialized")
    
    def generate_all_features(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        use_autoencoder_features: bool = True,
        regime_name: str = "default"
    ) -> pd.DataFrame:
        """Generate ALL possible features from the data including autoencoder features.
        
        Args:
            data: Input DataFrame
            target_columns: Target columns to exclude from features
            use_autoencoder_features: Whether to use autoencoder feature generator
            regime_name: Market regime name for autoencoder features
            
        Returns:
            DataFrame with all generated features
        """
        self.logger.info("🔧 Generating all possible features...")
        
        # Start with original data
        all_features = data.copy()
        
        # Exclude target columns
        if target_columns:
            exclude_columns = target_columns + ["timestamp", "timeframe", "composite_cluster_id"]
        else:
            exclude_columns = ["direction", "potential_profit_pct", "target", "label", 
                             "timestamp", "timeframe", "composite_cluster_id"]
        
        feature_columns = [col for col in all_features.columns if col not in exclude_columns]
        self.logger.info(f"📊 Base features: {len(feature_columns)}")
        
        # Apply autoencoder feature generation (if enabled and available)
        if use_autoencoder_features:
            try:
                from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
                
                self.logger.info("🔧 Applying autoencoder feature generation...")
                autoencoder_generator = AutoencoderFeatureGenerator()
                
                # Create dummy labels for autoencoder (it needs labels for feature filtering)
                dummy_labels = np.zeros(len(all_features))
                if "direction" in all_features.columns:
                    dummy_labels = all_features["direction"].values
                
                # Generate autoencoder features
                autoencoder_features = autoencoder_generator.generate_features(
                    features_df=all_features[feature_columns],
                    regime_name=regime_name,
                    labels=dummy_labels,
                    enable_analysis=True
                )
                
                # If autoencoder features were generated, add them
                if not autoencoder_features.empty and len(autoencoder_features.columns) > 0:
                    # Add autoencoder features with prefix
                    autoencoder_features = autoencoder_features.add_prefix("ae_")
                    all_features = pd.concat([all_features, autoencoder_features], axis=1)
                    self.logger.info(f"📊 After autoencoder features: {len(all_features.columns)}")
                else:
                    self.logger.info("📊 No autoencoder features generated, continuing with base features")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Autoencoder feature generation failed: {e}")
                self.logger.info("📊 Continuing without autoencoder features")
        
        # Apply profit-based feature engineering
        if self.config.use_profit_features and "potential_profit_pct" in data.columns:
            self.logger.info("🔧 Applying profit-based feature engineering...")
            all_features = self.profit_feature_engine.apply_all_features(all_features)
            self.logger.info(f"📊 After profit features: {len(all_features.columns)}")
        
        # Generate interaction features
        if self.config.use_interaction_features:
            self.logger.info("🔧 Generating interaction features...")
            interaction_features = self._generate_interaction_features(all_features[feature_columns])
            all_features = pd.concat([all_features, interaction_features], axis=1)
            self.logger.info(f"📊 After interaction features: {len(all_features.columns)}")
        
        # Generate polynomial features (optional, can be expensive)
        if self.config.use_polynomial_features:
            self.logger.info("🔧 Generating polynomial features...")
            poly_features = self._generate_polynomial_features(all_features[feature_columns])
            all_features = pd.concat([all_features, poly_features], axis=1)
            self.logger.info(f"📊 After polynomial features: {len(all_features.columns)}")
        
        # Handle missing values
        all_features = all_features.fillna(0)
        
        self.logger.info(f"✅ Total features generated: {len(all_features.columns)}")
        return all_features
    
    def _generate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between important features."""
        interaction_features = {}
        
        # Get top features by variance
        feature_vars = features.var()
        top_features = feature_vars.nlargest(20).index.tolist()
        
        # Generate pairwise interactions
        for i, feat1 in enumerate(top_features[:10]):  # Limit to top 10 to avoid explosion
            for feat2 in top_features[i+1:11]:
                interaction_name = f"{feat1}_x_{feat2}"
                interaction_features[interaction_name] = features[feat1] * features[feat2]
        
        return pd.DataFrame(interaction_features)
    
    def _generate_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features for important features."""
        poly_features = {}
        
        # Get top features by variance
        feature_vars = features.var()
        top_features = feature_vars.nlargest(10).index.tolist()
        
        # Generate squared and cubed features
        for feat in top_features:
            poly_features[f"{feat}_squared"] = features[feat] ** 2
            poly_features[f"{feat}_cubed"] = features[feat] ** 3
        
        return pd.DataFrame(poly_features)
    
    def select_features_comprehensive(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None,
        use_pipeline_regularization: bool = True
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """Comprehensive feature selection using multiple methods with pipeline regularization.
        
        Args:
            features: Feature DataFrame
            direction_target: Direction target series
            profit_target: Profit target series
            price_target: Price target series (optional)
            use_pipeline_regularization: Whether to use pipeline regularization
            
        Returns:
            Tuple of (selected_features, selected_feature_names, feature_scores)
        """
        self.logger.info("🎯 Starting comprehensive feature selection...")
        
        # Initialize feature scores
        feature_scores = {}
        selected_features = features.copy()
        
        # Load regularization configuration from pipeline
        regularization_config = None
        if use_pipeline_regularization:
            try:
                from src.training.regularization import RegularizationManager
                reg_manager = RegularizationManager()
                regularization_config = reg_manager.regularization_config
                self.logger.info("🔧 Loaded pipeline regularization configuration")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load regularization config: {e}")
        
        # 1. Variance threshold
        if "variance_threshold" in self.config.selection_methods:
            self.logger.info("📊 Applying variance threshold...")
            selected_features, variance_scores = self._apply_variance_threshold(selected_features)
            feature_scores['variance'] = variance_scores
        
        # 2. Correlation filter
        if "correlation_filter" in self.config.selection_methods:
            self.logger.info("📊 Applying correlation filter...")
            selected_features, correlation_scores = self._apply_correlation_filter(selected_features)
            feature_scores['correlation'] = correlation_scores
        
        # 3. Mutual information
        if "mutual_info" in self.config.selection_methods:
            self.logger.info("📊 Applying mutual information selection...")
            selected_features, mi_scores = self._apply_mutual_info_selection(
                selected_features, direction_target, profit_target, price_target
            )
            feature_scores['mutual_info'] = mi_scores
        
        # 4. Random Forest importance with regularization
        if "rf_importance" in self.config.selection_methods:
            self.logger.info("📊 Applying Random Forest importance with regularization...")
            selected_features, rf_scores = self._apply_rf_importance_with_regularization(
                selected_features, direction_target, profit_target, price_target, regularization_config
            )
            feature_scores['rf_importance'] = rf_scores
        
        # 5. Recursive Feature Elimination with regularization
        if "rfe" in self.config.selection_methods:
            self.logger.info("📊 Applying Recursive Feature Elimination with regularization...")
            selected_features, rfe_scores = self._apply_rfe_with_regularization(
                selected_features, direction_target, profit_target, price_target, regularization_config
            )
            feature_scores['rfe'] = rfe_scores
        
        # 6. Ensemble selection
        if "ensemble_selection" in self.config.selection_methods:
            self.logger.info("📊 Applying ensemble selection...")
            selected_features, ensemble_scores = self._apply_ensemble_selection(
                selected_features, direction_target, profit_target, price_target
            )
            feature_scores['ensemble'] = ensemble_scores
        
        # 7. Pipeline feature selection integration
        if use_pipeline_regularization:
            self.logger.info("📊 Applying pipeline feature selection integration...")
            selected_features, pipeline_scores = self._apply_pipeline_feature_selection(
                selected_features, direction_target, profit_target, price_target, regularization_config
            )
            feature_scores['pipeline'] = pipeline_scores
        
        # Final feature selection based on scores
        final_features, final_scores = self._select_final_features(
            selected_features, feature_scores
        )
        
        self.logger.info(f"✅ Feature selection completed: {len(final_features.columns)} features selected")
        
        return final_features, list(final_features.columns), final_scores
    
    def _apply_variance_threshold(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply variance threshold selection."""
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        selected_features = selector.fit_transform(features)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_names = features.columns[selected_mask]
        
        # Calculate scores
        scores = {name: features[name].var() for name in selected_names}
        
        return pd.DataFrame(selected_features, columns=selected_names, index=features.index), scores
    
    def _apply_correlation_filter(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply correlation-based feature filtering."""
        corr_matrix = features.corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.config.correlation_threshold)]
        
        # Remove highly correlated features
        selected_features = features.drop(columns=to_drop)
        
        # Calculate scores (inverse of max correlation)
        scores = {}
        for col in selected_features.columns:
            max_corr = corr_matrix.loc[col, selected_features.columns].max()
            scores[col] = 1 - max_corr
        
        return selected_features, scores
    
    def _apply_mutual_info_selection(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply mutual information selection."""
        # Calculate mutual information for each target
        mi_direction = mutual_info_classif(features, direction_target, random_state=42)
        mi_profit = mutual_info_regression(features, profit_target, random_state=42)
        
        if price_target is not None:
            mi_price = mutual_info_regression(features, price_target, random_state=42)
        else:
            mi_price = mi_profit  # Use profit MI as proxy
        
        # Combine MI scores with weights
        combined_mi = (
            self.config.direction_weight * mi_direction +
            self.config.profit_weight * mi_profit +
            self.config.price_weight * mi_price
        )
        
        # Select features above threshold
        selected_mask = combined_mi > self.config.mutual_info_threshold
        selected_features = features.iloc[:, selected_mask]
        
        # Create scores dictionary
        scores = {features.columns[i]: combined_mi[i] for i in range(len(features.columns)) if selected_mask[i]}
        
        return selected_features, scores
    
    def _apply_rf_importance_with_regularization(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None,
        regularization_config: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply Random Forest importance selection with regularization."""
        # Apply regularization parameters if available
        rf_params = {
            'n_estimators': self.config.rf_n_estimators,
            'max_depth': self.config.rf_max_depth,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if regularization_config and 'lightgbm' in regularization_config:
            # Apply L1/L2 regularization equivalent for RF
            rf_params['max_features'] = 'sqrt'  # Reduce feature usage
            rf_params['min_samples_split'] = max(2, int(len(features) * 0.01))  # Increase split threshold
            rf_params['min_samples_leaf'] = max(1, int(len(features) * 0.005))  # Increase leaf threshold
        
        # Train RF models for each target
        rf_direction = RandomForestClassifier(**rf_params)
        rf_direction.fit(features, direction_target)
        
        rf_profit = RandomForestRegressor(**rf_params)
        rf_profit.fit(features, profit_target)
        
        # Combine importance scores
        importance_direction = rf_direction.feature_importances_
        importance_profit = rf_profit.feature_importances_
        
        if price_target is not None:
            rf_price = RandomForestRegressor(**rf_params)
            rf_price.fit(features, price_target)
            importance_price = rf_price.feature_importances_
        else:
            importance_price = importance_profit
        
        # Weighted combination
        combined_importance = (
            self.config.direction_weight * importance_direction +
            self.config.profit_weight * importance_profit +
            self.config.price_weight * importance_price
        )
        
        # Apply regularization penalty to importance scores
        if regularization_config:
            # Reduce importance of features with high correlation (L1-like effect)
            corr_matrix = features.corr().abs()
            for i, feature in enumerate(features.columns):
                max_corr = corr_matrix.loc[feature, features.columns].max()
                if max_corr > 0.8:  # High correlation penalty
                    combined_importance[i] *= 0.8
        
        # Select top features
        top_indices = np.argsort(combined_importance)[-self.config.max_features:]
        selected_features = features.iloc[:, top_indices]
        
        # Create scores dictionary
        scores = {features.columns[i]: combined_importance[i] for i in top_indices}
        
        return selected_features, scores
    
    def _apply_rf_importance(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply Random Forest importance selection (legacy method)."""
        return self._apply_rf_importance_with_regularization(
            features, direction_target, profit_target, price_target, None
        )
    
    def _apply_rfe_with_regularization(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None,
        regularization_config: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply Recursive Feature Elimination with regularization."""
        # Use Random Forest as base estimator with regularization
        rf_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
        
        if regularization_config and 'lightgbm' in regularization_config:
            # Apply regularization parameters
            rf_params['max_features'] = 'sqrt'
            rf_params['min_samples_split'] = max(2, int(len(features) * 0.01))
            rf_params['min_samples_leaf'] = max(1, int(len(features) * 0.005))
        
        base_estimator = RandomForestClassifier(**rf_params)
        
        # Apply RFE
        rfe = RFE(
            estimator=base_estimator,
            n_features_to_select=min(self.config.max_features, len(features.columns)),
            step=0.1
        )
        
        # Use direction target for RFE (can be modified to use combined target)
        rfe.fit(features, direction_target)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_features = features.iloc[:, selected_mask]
        
        # Create scores dictionary (ranking, lower is better)
        scores = {features.columns[i]: 1.0 / (rfe.ranking_[i] + 1) for i in range(len(features.columns)) if selected_mask[i]}
        
        return selected_features, scores
    
    def _apply_rfe(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply Recursive Feature Elimination (legacy method)."""
        return self._apply_rfe_with_regularization(
            features, direction_target, profit_target, price_target, None
        )
    
    def _apply_pipeline_feature_selection(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None,
        regularization_config: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply pipeline-specific feature selection methods."""
        self.logger.info("🔧 Applying pipeline feature selection methods...")
        
        # Initialize scores
        pipeline_scores = {}
        
        # 1. Apply regularization-aware feature selection
        if regularization_config:
            # Use regularization parameters to guide feature selection
            l1_alpha = regularization_config.get('l1_alpha', 0.01)
            l2_alpha = regularization_config.get('l2_alpha', 0.001)
            
            # Calculate feature stability scores
            stability_scores = self._calculate_feature_stability(features, direction_target)
            
            # Apply regularization penalty
            for feature in features.columns:
                base_score = stability_scores.get(feature, 0.0)
                # Higher regularization = lower feature scores
                regularization_penalty = 1.0 / (1.0 + l1_alpha + l2_alpha)
                pipeline_scores[feature] = base_score * regularization_penalty
        
        # 2. Apply feature importance from pipeline models
        try:
            # Try to get feature importance from existing pipeline models
            pipeline_importance = self._get_pipeline_feature_importance()
            if pipeline_importance:
                for feature in features.columns:
                    if feature in pipeline_importance:
                        pipeline_scores[feature] = pipeline_scores.get(feature, 0.0) + pipeline_importance[feature]
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get pipeline feature importance: {e}")
        
        # Select features based on pipeline scores
        if pipeline_scores:
            sorted_features = sorted(pipeline_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:self.config.max_features]]
            selected_features = features[top_features]
        else:
            selected_features = features
        
        return selected_features, pipeline_scores
    
    def _calculate_feature_stability(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, float]:
        """Calculate feature stability scores."""
        stability_scores = {}
        
        # Use cross-validation to assess feature stability
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        
        for feature in features.columns:
            try:
                # Use single feature for prediction
                X_single = features[[feature]]
                
                # Calculate cross-validation score
                cv_scores = cross_val_score(
                    LogisticRegression(random_state=42),
                    X_single,
                    target,
                    cv=3,
                    scoring='accuracy'
                )
                
                # Stability score is the mean CV score
                stability_scores[feature] = np.mean(cv_scores)
            except Exception:
                stability_scores[feature] = 0.0
        
        return stability_scores
    
    def _get_pipeline_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from existing pipeline models."""
        # This method can be extended to load feature importance from saved models
        # For now, return None to indicate no pipeline importance available
        return None
    
    def _apply_ensemble_selection(
        self,
        features: pd.DataFrame,
        direction_target: pd.Series,
        profit_target: pd.Series,
        price_target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Apply ensemble feature selection combining multiple methods."""
        # Get scores from previous methods
        ensemble_scores = {}
        
        # Combine scores from different methods
        for feature in features.columns:
            scores = []
            
            # Add variance score
            if 'variance' in self.feature_scores and feature in self.feature_scores['variance']:
                scores.append(self.feature_scores['variance'][feature])
            
            # Add mutual info score
            if 'mutual_info' in self.feature_scores and feature in self.feature_scores['mutual_info']:
                scores.append(self.feature_scores['mutual_info'][feature])
            
            # Add RF importance score
            if 'rf_importance' in self.feature_scores and feature in self.feature_scores['rf_importance']:
                scores.append(self.feature_scores['rf_importance'][feature])
            
            # Calculate ensemble score
            if scores:
                ensemble_scores[feature] = np.mean(scores)
            else:
                ensemble_scores[feature] = 0.0
        
        # Select top features
        sorted_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:self.config.max_features]]
        
        selected_features = features[top_features]
        
        return selected_features, ensemble_scores
    
    def _select_final_features(
        self,
        features: pd.DataFrame,
        feature_scores: Dict[str, Dict[str, float]]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select final features based on all scores."""
        # Combine all scores
        final_scores = {}
        
        for feature in features.columns:
            scores = []
            weights = []
            
            for method, scores_dict in feature_scores.items():
                if feature in scores_dict:
                    scores.append(scores_dict[feature])
                    weights.append(1.0)  # Equal weight for all methods
            
            if scores:
                # Calculate weighted average
                final_scores[feature] = np.average(scores, weights=weights)
            else:
                final_scores[feature] = 0.0
        
        # Select top features
        sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure minimum and maximum feature counts
        n_features = min(
            max(self.config.min_features, len(sorted_features)),
            self.config.max_features
        )
        
        top_features = [f[0] for f in sorted_features[:n_features]]
        selected_features = features[top_features]
        
        # Update final scores
        final_scores = {f[0]: f[1] for f in sorted_features[:n_features]}
        
        return selected_features, final_scores
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get summary of feature importance across all methods."""
        if not self.feature_scores:
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_data = []
        
        for feature in set().union(*[set(scores.keys()) for scores in self.feature_scores.values()]):
            row = {'feature': feature}
            
            for method, scores in self.feature_scores.items():
                row[f'{method}_score'] = scores.get(feature, 0.0)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate average score
        score_columns = [col for col in summary_df.columns if col.endswith('_score')]
        summary_df['average_score'] = summary_df[score_columns].mean(axis=1)
        
        # Sort by average score
        summary_df = summary_df.sort_values('average_score', ascending=False)
        
        return summary_df


def create_comprehensive_feature_selector(
    max_features: int = 500,
    use_profit_features: bool = True,
    use_all_existing_features: bool = True
) -> ComprehensiveFeatureSelector:
    """Factory function to create comprehensive feature selector.
    
    Args:
        max_features: Maximum number of features to select
        use_profit_features: Whether to use profit-based features
        use_all_existing_features: Whether to use all existing features
        
    Returns:
        Configured ComprehensiveFeatureSelector instance
    """
    config = FeatureSelectionConfig(
        max_features=max_features,
        use_profit_features=use_profit_features,
        use_all_existing_features=use_all_existing_features
    )
    
    return ComprehensiveFeatureSelector(config)