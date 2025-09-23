#!/usr/bin/env python3
"""
Model-Specific Feature Selection Strategy

This module provides adaptive feature selection strategies tailored for different
model types, particularly the advanced time series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import cross_val_score
import time

# Import existing utilities
from .feature_selection import FeatureSelectionFramework
from ..logger import get_logger

logger = get_logger("ModelSpecificFeatureSelection")

@dataclass
class ModelSpecificConfig:
    """Configuration for model-specific feature selection."""
    
    # Model type and target feature count
    model_type: str
    target_features: int
    
    # Selection strategy parameters
    use_regime_aware_selection: bool = True
    use_temporal_features: bool = True
    use_cross_timeframe_features: bool = True
    use_microstructure_features: bool = True
    
    # Feature category weights
    technical_indicators_weight: float = 1.0
    regime_features_weight: float = 1.0
    temporal_features_weight: float = 1.0
    microstructure_features_weight: float = 1.0
    interaction_features_weight: float = 0.8
    
    # Selection thresholds
    min_importance_threshold: float = 0.001
    correlation_threshold: float = 0.95
    stability_threshold: float = 0.7
    
    # Model-specific parameters
    mamba_state_expansion: int = 4
    resnet_blocks: List[int] = None
    nbeats_backcast_length: int = 100
    deepscaler_layers: int = 4

class ModelSpecificFeatureSelector:
    """Adaptive feature selection for different model types."""
    
    def __init__(self, config: ModelSpecificConfig):
        self.config = config
        self.logger = logger.getChild(f'ModelSpecific.{config.model_type}')
        
        # Initialize base feature selection framework
        self.base_selector = FeatureSelectionFramework()
        
        # Model-specific feature categories
        self.feature_categories = self._define_feature_categories()
        
        self.logger.info(f"🚀 ModelSpecificFeatureSelector initialized for {config.model_type}")
        self.logger.info(f"   🎯 Target features: {config.target_features}")
    
    def _define_feature_categories(self) -> Dict[str, List[str]]:
        """Define feature categories based on model type."""
        
        categories = {
            'technical_indicators': [
                'rsi', 'macd', 'bb_', 'atr', 'stoch', 'roc', 'momentum',
                'sma', 'ema', 'wma', 'vwap', 'obv', 'mfi', 'adx'
            ],
            'regime_features': [
                'regime', 'cluster', 'state', 'hmm', 'volatility_regime',
                'trend_regime', 'volume_regime', 'momentum_regime'
            ],
            'temporal_features': [
                'lag_', 'lead_', 'diff_', 'pct_change', 'rolling_',
                'expanding_', 'ewm_', 'seasonal_', 'trend_'
            ],
            'microstructure_features': [
                'spread', 'imbalance', 'flow', 'tick', 'orderbook',
                'bid_ask', 'volume_imbalance', 'trade_size'
            ],
            'cross_timeframe_features': [
                '1m_', '5m_', '15m_', '30m_', '1h_', '4h_', '1d_',
                'cross_tf_', 'multi_tf_'
            ],
            'interaction_features': [
                'momentum_volatility', 'volume_price', 'spread_volume',
                'regime_momentum', 'tf_interaction'
            ]
        }
        
        return categories
    
    def select_features_for_model(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 model_type: str = None) -> Dict[str, Any]:
        """Select features optimized for specific model type."""
        
        start_time = time.time()
        model_type = model_type or self.config.model_type
        
        self.logger.info(f"🔍 Starting model-specific feature selection for {model_type}")
        self.logger.info(f"   📊 Input: {len(X)} samples, {len(X.columns)} features")
        self.logger.info(f"   🎯 Target: {self.config.target_features} features")
        
        # Step 1: Categorize features
        categorized_features = self._categorize_features(X.columns)
        
        # Step 2: Apply model-specific selection strategy
        if model_type in ['advanced_mamba_hybrid', 'AdvancedMambaHybrid']:
            selected_features = self._select_for_mamba_hybrid(X, y, categorized_features)
        elif model_type in ['financial_resnet', 'FinancialResNet']:
            selected_features = self._select_for_financial_resnet(X, y, categorized_features)
        elif model_type in ['deepscaler', 'DeepScaler', 'deepscaler_1m', 'DeepScaler1m']:
            selected_features = self._select_for_deepscaler(X, y, categorized_features)
        elif model_type in ['nbeats', 'NBEATS']:
            selected_features = self._select_for_nbeats(X, y, categorized_features)
        else:
            # Fallback to general selection
            selected_features = self._select_general_features(X, y, categorized_features)
        
        # Step 3: Validate and refine selection
        final_features = self._validate_selection(X[selected_features], y, selected_features)
        
        # Step 4: Generate selection report
        selection_time = time.time() - start_time
        report = self._generate_selection_report(
            X, y, selected_features, final_features, selection_time
        )
        
        self.logger.info(f"✅ Model-specific feature selection completed in {selection_time:.3f}s")
        self.logger.info(f"   📊 Final features: {len(final_features)}")
        
        return {
            'selected_features': final_features,
            'selection_report': report,
            'model_type': model_type,
            'selection_time': selection_time
        }
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features based on naming patterns."""
        
        categorized = {category: [] for category in self.feature_categories.keys()}
        categorized['other'] = []
        
        for feature in feature_names:
            categorized_flag = False
            
            for category, patterns in self.feature_categories.items():
                if any(pattern.lower() in feature.lower() for pattern in patterns):
                    categorized[category].append(feature)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(feature)
        
        # Log categorization results
        for category, features in categorized.items():
            if features:
                self.logger.info(f"   📂 {category}: {len(features)} features")
        
        return categorized
    
    def _select_for_mamba_hybrid(self, X: pd.DataFrame, y: pd.Series, 
                                categorized_features: Dict[str, List[str]]) -> List[str]:
        """Select features optimized for Advanced Mamba Hybrid model."""
        
        self.logger.info("🎯 Selecting features for Advanced Mamba Hybrid")
        
        # Mamba models excel at temporal patterns and multi-timeframe fusion
        selected_features = []
        
        # Priority 1: Cross-timeframe features (40% of selection)
        cross_tf_count = int(self.config.target_features * 0.4)
        cross_tf_features = self._select_top_features(
            X[categorized_features['cross_timeframe_features']], y, cross_tf_count
        )
        selected_features.extend(cross_tf_features)
        
        # Priority 2: Temporal features (25% of selection)
        temporal_count = int(self.config.target_features * 0.25)
        temporal_features = self._select_top_features(
            X[categorized_features['temporal_features']], y, temporal_count
        )
        selected_features.extend(temporal_features)
        
        # Priority 3: Technical indicators (20% of selection)
        technical_count = int(self.config.target_features * 0.2)
        technical_features = self._select_top_features(
            X[categorized_features['technical_indicators']], y, technical_count
        )
        selected_features.extend(technical_features)
        
        # Priority 4: Regime features (10% of selection)
        regime_count = int(self.config.target_features * 0.1)
        regime_features = self._select_top_features(
            X[categorized_features['regime_features']], y, regime_count
        )
        selected_features.extend(regime_features)
        
        # Priority 5: Microstructure features (5% of selection)
        microstructure_count = self.config.target_features - len(selected_features)
        if microstructure_count > 0:
            microstructure_features = self._select_top_features(
                X[categorized_features['microstructure_features']], y, microstructure_count
            )
            selected_features.extend(microstructure_features)
        
        return selected_features[:self.config.target_features]
    
    def _select_for_financial_resnet(self, X: pd.DataFrame, y: pd.Series,
                                    categorized_features: Dict[str, List[str]]) -> List[str]:
        """Select features optimized for FinancialResNet model."""
        
        self.logger.info("🎯 Selecting features for FinancialResNet")
        
        # FinancialResNet is optimized for regime-aware classification
        selected_features = []
        
        # Priority 1: Regime features (35% of selection)
        regime_count = int(self.config.target_features * 0.35)
        regime_features = self._select_top_features(
            X[categorized_features['regime_features']], y, regime_count
        )
        selected_features.extend(regime_features)
        
        # Priority 2: Technical indicators (30% of selection)
        technical_count = int(self.config.target_features * 0.3)
        technical_features = self._select_top_features(
            X[categorized_features['technical_indicators']], y, technical_count
        )
        selected_features.extend(technical_features)
        
        # Priority 3: Temporal features (20% of selection)
        temporal_count = int(self.config.target_features * 0.2)
        temporal_features = self._select_top_features(
            X[categorized_features['temporal_features']], y, temporal_count
        )
        selected_features.extend(temporal_features)
        
        # Priority 4: Microstructure features (15% of selection)
        microstructure_count = self.config.target_features - len(selected_features)
        if microstructure_count > 0:
            microstructure_features = self._select_top_features(
                X[categorized_features['microstructure_features']], y, microstructure_count
            )
            selected_features.extend(microstructure_features)
        
        return selected_features[:self.config.target_features]
    
    def _select_for_deepscaler(self, X: pd.DataFrame, y: pd.Series,
                              categorized_features: Dict[str, List[str]]) -> List[str]:
        """Select features optimized for DeepScaler model."""
        
        self.logger.info("🎯 Selecting features for DeepScaler")
        
        # DeepScaler is optimized for scaling and normalization tasks
        selected_features = []
        
        # Priority 1: Technical indicators (50% of selection)
        technical_count = int(self.config.target_features * 0.5)
        technical_features = self._select_top_features(
            X[categorized_features['technical_indicators']], y, technical_count
        )
        selected_features.extend(technical_features)
        
        # Priority 2: Temporal features (30% of selection)
        temporal_count = int(self.config.target_features * 0.3)
        temporal_features = self._select_top_features(
            X[categorized_features['temporal_features']], y, temporal_count
        )
        selected_features.extend(temporal_features)
        
        # Priority 3: Other features (20% of selection)
        other_count = self.config.target_features - len(selected_features)
        if other_count > 0:
            other_features = self._select_top_features(
                X[categorized_features['other']], y, other_count
            )
            selected_features.extend(other_features)
        
        return selected_features[:self.config.target_features]
    
    def _select_for_nbeats(self, X: pd.DataFrame, y: pd.Series,
                          categorized_features: Dict[str, List[str]]) -> List[str]:
        """Select features optimized for N-BEATS model."""
        
        self.logger.info("🎯 Selecting features for N-BEATS")
        
        # N-BEATS is optimized for trend and seasonality decomposition
        selected_features = []
        
        # Priority 1: Temporal features (40% of selection)
        temporal_count = int(self.config.target_features * 0.4)
        temporal_features = self._select_top_features(
            X[categorized_features['temporal_features']], y, temporal_count
        )
        selected_features.extend(temporal_features)
        
        # Priority 2: Technical indicators (35% of selection)
        technical_count = int(self.config.target_features * 0.35)
        technical_features = self._select_top_features(
            X[categorized_features['technical_indicators']], y, technical_count
        )
        selected_features.extend(technical_features)
        
        # Priority 3: Regime features (15% of selection)
        regime_count = int(self.config.target_features * 0.15)
        regime_features = self._select_top_features(
            X[categorized_features['regime_features']], y, regime_count
        )
        selected_features.extend(regime_features)
        
        # Priority 4: Other features (10% of selection)
        other_count = self.config.target_features - len(selected_features)
        if other_count > 0:
            other_features = self._select_top_features(
                X[categorized_features['other']], y, other_count
            )
            selected_features.extend(other_features)
        
        return selected_features[:self.config.target_features]
    
    def _select_general_features(self, X: pd.DataFrame, y: pd.Series,
                                categorized_features: Dict[str, List[str]]) -> List[str]:
        """General feature selection for unknown model types."""
        
        self.logger.info("🎯 Using general feature selection strategy")
        
        # Use RandomForest importance for general selection
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = rf.feature_importances_
        feature_names = X.columns.tolist()
        
        # Sort by importance and select top features
        importance_pairs = list(zip(feature_names, importance_scores))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [name for name, _ in importance_pairs[:self.config.target_features]]
        
        return selected_features
    
    def _select_top_features(self, X_subset: pd.DataFrame, y: pd.Series, 
                           n_features: int) -> List[str]:
        """Select top features from a subset using RandomForest importance."""
        
        if len(X_subset.columns) == 0 or n_features <= 0:
            return []
        
        if len(X_subset.columns) <= n_features:
            return X_subset.columns.tolist()
        
        try:
            # Use RandomForest for feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_subset, y)
            
            # Get importance scores
            importance_scores = rf.feature_importances_
            feature_names = X_subset.columns.tolist()
            
            # Sort by importance
            importance_pairs = list(zip(feature_names, importance_scores))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected = [name for name, _ in importance_pairs[:n_features]]
            
            return selected
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection failed for subset: {e}")
            # Fallback: select first n_features
            return X_subset.columns.tolist()[:n_features]
    
    def _validate_selection(self, X_selected: pd.DataFrame, y: pd.Series, 
                          selected_features: List[str]) -> List[str]:
        """Validate and refine feature selection."""
        
        self.logger.info("🔍 Validating feature selection")
        
        # Check for high correlation
        correlation_matrix = X_selected.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.config.correlation_threshold:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        # Remove highly correlated features
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if feat1 in selected_features and feat2 in selected_features:
                # Keep the feature with higher variance
                if X_selected[feat1].var() > X_selected[feat2].var():
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)
        
        # Remove features with low variance
        for feature in selected_features:
            if X_selected[feature].var() < self.config.min_importance_threshold:
                features_to_remove.add(feature)
        
        # Final feature list
        final_features = [f for f in selected_features if f not in features_to_remove]
        
        if len(final_features) < self.config.target_features * 0.8:
            self.logger.warning(f"⚠️ Final feature count ({len(final_features)}) is below 80% of target")
        
        return final_features
    
    def _generate_selection_report(self, X: pd.DataFrame, y: pd.Series,
                                 selected_features: List[str], final_features: List[str],
                                 selection_time: float) -> Dict[str, Any]:
        """Generate comprehensive selection report."""
        
        # Categorize final features
        categorized_final = self._categorize_features(final_features)
        
        # Calculate feature importance scores
        if len(final_features) > 0:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X[final_features], y)
            importance_scores = dict(zip(final_features, rf.feature_importances_))
        else:
            importance_scores = {}
        
        report = {
            'model_type': self.config.model_type,
            'target_features': self.config.target_features,
            'selected_features': len(selected_features),
            'final_features': len(final_features),
            'selection_time': selection_time,
            'feature_categories': {
                category: len(features) for category, features in categorized_final.items()
            },
            'top_features': sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            'selection_quality': {
                'correlation_removed': len(selected_features) - len(final_features),
                'variance_filtered': sum(1 for f in final_features if X[f].var() > self.config.min_importance_threshold),
                'importance_distribution': {
                    'mean': np.mean(list(importance_scores.values())) if importance_scores else 0,
                    'std': np.std(list(importance_scores.values())) if importance_scores else 0,
                    'max': max(importance_scores.values()) if importance_scores else 0,
                    'min': min(importance_scores.values()) if importance_scores else 0
                }
            }
        }
        
        return report

def create_model_specific_selector(model_type: str, target_features: int = None) -> ModelSpecificFeatureSelector:
    """Factory function to create model-specific feature selector."""
    
    # Get target features from MODEL_FEATURE_TARGETS if not specified
    if target_features is None:
        from .feature_selection import FeatureSelectionFramework
        base_selector = FeatureSelectionFramework()
        target_features = base_selector.get_model_target_features(model_type)
    
    config = ModelSpecificConfig(
        model_type=model_type,
        target_features=target_features
    )
    
    return ModelSpecificFeatureSelector(config)