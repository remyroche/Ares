"""
Enhanced Feature Engineering for Tactician and Analyst Training

This module provides enhanced feature engineering that combines base features with
model outputs to create comprehensive feature sets for both Tactician and Analyst training.

Feature Architecture:
- Tactician: All base features + Analyst outputs + HMM outputs
- Analyst: All base features + HMM outputs (no circular dependency)

Key Features:
- Dynamic feature combination based on model type
- HMM regime outputs integration
- Analyst model outputs integration (for Tactician)
- Feature validation and quality checks
- Memory-efficient feature processing
- Configurable feature selection and scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance
)

logger = system_logger.getChild('EnhancedFeatureEngineering')


@dataclass
class FeatureEngineeringConfig:
    """Configuration for enhanced feature engineering."""
    
    # Feature sources
    include_base_features: bool = True
    include_hmm_features: bool = True
    include_analyst_features: bool = False  # Only for Tactician
    
    # Feature processing
    enable_feature_scaling: bool = True
    enable_feature_selection: bool = False
    feature_selection_threshold: float = 0.01
    
    # Memory management
    chunk_size: int = 10000
    enable_memory_optimization: bool = True
    
    # Validation
    validate_feature_quality: bool = True
    max_missing_ratio: float = 0.1
    min_feature_variance: float = 1e-8


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering component that combines base features with
    model outputs to create comprehensive feature sets for training.
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """Initialize the enhanced feature engineer."""
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger.getChild('EnhancedFeatureEngineer')
        
        # Model references
        self.hmm_models = {}
        self.analyst_model = None
        
        # Feature cache
        self.feature_cache = {}
        self.feature_stats = {}
        
        self.logger.info("🔧 Enhanced Feature Engineer initialized")
    
    def set_hmm_models(self, hmm_models: Dict[str, Any]):
        """Set HMM models for feature generation."""
        self.hmm_models = hmm_models
        self.logger.info(f"✅ Set {len(hmm_models)} HMM models")
    
    def set_analyst_model(self, analyst_model: Any):
        """Set Analyst model for feature generation."""
        self.analyst_model = analyst_model
        self.logger.info("✅ Set Analyst model")
    
    def generate_tactician_features(self, 
                                  data: pd.DataFrame,
                                  labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate features for Tactician training.
        
        Tactician features: All base features + Analyst outputs + HMM outputs
        
        Args:
            data: Base data with OHLC and technical features
            labels: Optional labels for feature validation
            
        Returns:
            Enhanced feature DataFrame for Tactician training
        """
        try:
            tprint_info("🎯 Generating Tactician features...")
            
            # Start with base features
            tactician_features = data.copy()
            
            # Add HMM features
            if self.config.include_hmm_features and self.hmm_models:
                hmm_features = self._generate_hmm_features(data)
                tactician_features = pd.concat([tactician_features, hmm_features], axis=1)
                tprint_info(f"   → Added {len(hmm_features.columns)} HMM features")
            
            # Add Analyst features
            if self.config.include_analyst_features and self.analyst_model:
                analyst_features = self._generate_analyst_features(data)
                tactician_features = pd.concat([tactician_features, analyst_features], axis=1)
                tprint_info(f"   → Added {len(analyst_features.columns)} Analyst features")
            
            # Apply feature processing
            tactician_features = self._process_features(tactician_features, labels)
            
            tprint_success(f"✅ Generated {len(tactician_features.columns)} Tactician features")
            return tactician_features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate Tactician features: {e}")
            raise
    
    def generate_analyst_features(self, 
                                data: pd.DataFrame,
                                labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate features for Analyst training.
        
        Analyst features: All base features + HMM outputs (no circular dependency)
        
        Args:
            data: Base data with OHLC and technical features
            labels: Optional labels for feature validation
            
        Returns:
            Enhanced feature DataFrame for Analyst training
        """
        try:
            tprint_info("🔍 Generating Analyst features...")
            
            # Start with base features
            analyst_features = data.copy()
            
            # Add HMM features
            if self.config.include_hmm_features and self.hmm_models:
                hmm_features = self._generate_hmm_features(data)
                analyst_features = pd.concat([analyst_features, hmm_features], axis=1)
                tprint_info(f"   → Added {len(hmm_features.columns)} HMM features")
            
            # Apply feature processing
            analyst_features = self._process_features(analyst_features, labels)
            
            tprint_success(f"✅ Generated {len(analyst_features.columns)} Analyst features")
            return analyst_features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate Analyst features: {e}")
            raise
    
    def _generate_hmm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate HMM model outputs as features."""
        try:
            hmm_features = pd.DataFrame(index=data.index)
            
            for regime_name, hmm_model in self.hmm_models.items():
                try:
                    # Get HMM predictions for this regime
                    regime_features = self._extract_hmm_regime_features(data, hmm_model, regime_name)
                    hmm_features = pd.concat([hmm_features, regime_features], axis=1)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to extract features for regime {regime_name}: {e}")
                    continue
            
            # Add composite HMM features
            composite_features = self._generate_composite_hmm_features(hmm_features)
            hmm_features = pd.concat([hmm_features, composite_features], axis=1)
            
            return hmm_features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate HMM features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _extract_hmm_regime_features(self, 
                                   data: pd.DataFrame,
                                   hmm_model: Any,
                                   regime_name: str) -> pd.DataFrame:
        """Extract features from a specific HMM regime model."""
        regime_features = pd.DataFrame(index=data.index)
        
        try:
            # Try to get regime predictions
            if hasattr(hmm_model, 'predict'):
                regime_states = hmm_model.predict(data.values)
                regime_features[f'hmm_{regime_name}_state'] = regime_states
            
            # Try to get regime probabilities
            if hasattr(hmm_model, 'predict_proba'):
                regime_probs = hmm_model.predict_proba(data.values)
                if regime_probs.ndim > 1:
                    for i in range(regime_probs.shape[1]):
                        regime_features[f'hmm_{regime_name}_prob_{i}'] = regime_probs[:, i]
                else:
                    regime_features[f'hmm_{regime_name}_prob'] = regime_probs
            
            # Try to get transition probabilities
            if hasattr(hmm_model, 'transmat_'):
                # Use current state to get transition probabilities
                if 'hmm_{regime_name}_state' in regime_features.columns:
                    current_states = regime_features[f'hmm_{regime_name}_state'].astype(int)
                    for i, state in enumerate(current_states.unique()):
                        if state < len(hmm_model.transmat_):
                            trans_probs = hmm_model.transmat_[state]
                            for j, prob in enumerate(trans_probs):
                                regime_features[f'hmm_{regime_name}_trans_{state}_{j}'] = prob
            
            # Add regime-specific technical features
            regime_features[f'hmm_{regime_name}_confidence'] = self._calculate_regime_confidence(
                regime_features, regime_name
            )
            
            return regime_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract regime features for {regime_name}: {e}")
            return pd.DataFrame(index=data.index)
    
    def _generate_composite_hmm_features(self, hmm_features: pd.DataFrame) -> pd.DataFrame:
        """Generate composite HMM features."""
        composite_features = pd.DataFrame(index=hmm_features.index)
        
        try:
            # Find all regime probability columns
            prob_columns = [col for col in hmm_features.columns if '_prob_' in col]
            
            if prob_columns:
                # Calculate regime diversity (entropy)
                regime_probs = hmm_features[prob_columns].values
                regime_probs = np.clip(regime_probs, 1e-8, 1.0)  # Avoid log(0)
                entropy = -np.sum(regime_probs * np.log(regime_probs), axis=1)
                composite_features['hmm_regime_entropy'] = entropy
                
                # Calculate dominant regime strength
                max_probs = np.max(regime_probs, axis=1)
                composite_features['hmm_dominant_regime_strength'] = max_probs
                
                # Calculate regime stability (variance of probabilities)
                prob_var = np.var(regime_probs, axis=1)
                composite_features['hmm_regime_stability'] = 1.0 / (1.0 + prob_var)
            
            # Find all confidence columns
            confidence_columns = [col for col in hmm_features.columns if '_confidence' in col]
            if confidence_columns:
                avg_confidence = hmm_features[confidence_columns].mean(axis=1)
                composite_features['hmm_avg_confidence'] = avg_confidence
                
                max_confidence = hmm_features[confidence_columns].max(axis=1)
                composite_features['hmm_max_confidence'] = max_confidence
            
            return composite_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate composite HMM features: {e}")
            return pd.DataFrame(index=hmm_features.index)
    
    def _calculate_regime_confidence(self, regime_features: pd.DataFrame, regime_name: str) -> pd.Series:
        """Calculate confidence score for a regime."""
        try:
            confidence = pd.Series(0.5, index=regime_features.index)  # Default confidence
            
            # Use probability-based confidence if available
            prob_columns = [col for col in regime_features.columns if f'hmm_{regime_name}_prob' in col]
            if prob_columns:
                max_prob = regime_features[prob_columns].max(axis=1)
                confidence = max_prob
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate confidence for {regime_name}: {e}")
            return pd.Series(0.5, index=regime_features.index)
    
    def _generate_analyst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Analyst model outputs as features for Tactician."""
        try:
            analyst_features = pd.DataFrame(index=data.index)
            
            if self.analyst_model is None:
                self.logger.warning("⚠️ No Analyst model available for feature generation")
                return analyst_features
            
            # Prepare features for Analyst prediction
            if hasattr(self.analyst_model, 'predict'):
                # Get Analyst predictions
                predictions = self.analyst_model.predict(data.values)
                if predictions.ndim > 1:
                    for i in range(predictions.shape[1]):
                        analyst_features[f'analyst_pred_{i}'] = predictions[:, i]
                else:
                    analyst_features['analyst_prediction'] = predictions
            
            # Get confidence scores if available
            if hasattr(self.analyst_model, 'predict_proba'):
                probabilities = self.analyst_model.predict_proba(data.values)
                if probabilities.ndim > 1:
                    for i in range(probabilities.shape[1]):
                        analyst_features[f'analyst_prob_{i}'] = probabilities[:, i]
                    
                    # Calculate confidence as max probability
                    max_prob = np.max(probabilities, axis=1)
                    analyst_features['analyst_confidence'] = max_prob
                else:
                    analyst_features['analyst_confidence'] = probabilities
            
            # Get ensemble weights if available
            if hasattr(self.analyst_model, 'get_ensemble_weights'):
                weights = self.analyst_model.get_ensemble_weights()
                for i, weight in enumerate(weights):
                    analyst_features[f'analyst_weight_{i}'] = weight
            
            # Add prediction-based features
            if 'analyst_prediction' in analyst_features.columns:
                pred = analyst_features['analyst_prediction']
                
                # Rolling statistics of predictions
                analyst_features['analyst_pred_mean_5'] = pred.rolling(5).mean()
                analyst_features['analyst_pred_std_5'] = pred.rolling(5).std()
                analyst_features['analyst_pred_mean_20'] = pred.rolling(20).mean()
                analyst_features['analyst_pred_std_20'] = pred.rolling(20).std()
                
                # Prediction momentum
                analyst_features['analyst_pred_momentum'] = pred.diff()
                analyst_features['analyst_pred_momentum_5'] = pred.diff(5)
            
            return analyst_features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate Analyst features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _process_features(self, 
                         features: pd.DataFrame,
                         labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply feature processing (scaling, selection, validation)."""
        processed_features = features.copy()
        
        try:
            # Feature validation
            if self.config.validate_feature_quality:
                processed_features = self._validate_features(processed_features)
            
            # Feature scaling
            if self.config.enable_feature_scaling:
                processed_features = self._scale_features(processed_features)
            
            # Feature selection
            if self.config.enable_feature_selection and labels is not None:
                processed_features = self._select_features(processed_features, labels)
            
            return processed_features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process features: {e}")
            return features
    
    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate feature quality and remove problematic features."""
        validated_features = features.copy()
        
        # Remove features with too many missing values
        missing_ratio = validated_features.isnull().sum() / len(validated_features)
        high_missing_cols = missing_ratio[missing_ratio > self.config.max_missing_ratio].index
        validated_features = validated_features.drop(columns=high_missing_cols)
        
        # Remove features with zero variance
        feature_var = validated_features.var()
        zero_var_cols = feature_var[feature_var < self.config.min_feature_variance].index
        validated_features = validated_features.drop(columns=zero_var_cols)
        
        # Remove features with infinite values
        inf_cols = []
        for col in validated_features.columns:
            if np.isinf(validated_features[col]).any():
                inf_cols.append(col)
        validated_features = validated_features.drop(columns=inf_cols)
        
        if high_missing_cols.any() or zero_var_cols.any() or inf_cols:
            self.logger.info(f"🧹 Removed {len(high_missing_cols) + len(zero_var_cols) + len(inf_cols)} problematic features")
        
        return validated_features
    
    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Select numeric columns only
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return features
            
            # Apply standard scaling
            scaler = StandardScaler()
            features_scaled = features.copy()
            features_scaled[numeric_cols] = scaler.fit_transform(features[numeric_cols])
            
            # Store scaler for later use
            self.feature_cache['scaler'] = scaler
            
            return features_scaled
            
        except ImportError:
            self.logger.warning("⚠️ sklearn not available for feature scaling")
            return features
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to scale features: {e}")
            return features
    
    def _select_features(self, 
                        features: pd.DataFrame,
                        labels: pd.Series) -> pd.DataFrame:
        """Apply feature selection based on correlation with labels."""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            
            # Select numeric columns only
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return features
            
            # Remove features with low correlation
            correlations = features[numeric_cols].corrwith(labels).abs()
            selected_cols = correlations[correlations > self.config.feature_selection_threshold].index
            
            if len(selected_cols) < len(numeric_cols):
                self.logger.info(f"🎯 Selected {len(selected_cols)}/{len(numeric_cols)} features based on correlation")
                return features[selected_cols]
            
            return features
            
        except ImportError:
            self.logger.warning("⚠️ sklearn not available for feature selection")
            return features
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to select features: {e}")
            return features
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature engineering statistics."""
        return self.feature_stats.copy()
    
    def save_feature_engineer(self, file_path: str):
        """Save the feature engineer configuration and models."""
        try:
            save_data = {
                'config': self.config,
                'hmm_models': self.hmm_models,
                'feature_stats': self.feature_stats
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"💾 Feature engineer saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save feature engineer: {e}")
    
    def load_feature_engineer(self, file_path: str):
        """Load the feature engineer configuration and models."""
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.config = save_data.get('config', FeatureEngineeringConfig())
            self.hmm_models = save_data.get('hmm_models', {})
            self.feature_stats = save_data.get('feature_stats', {})
            
            self.logger.info(f"📁 Feature engineer loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load feature engineer: {e}")


def create_enhanced_feature_engineer(
    include_hmm_features: bool = True,
    include_analyst_features: bool = False,
    enable_feature_scaling: bool = True
) -> EnhancedFeatureEngineer:
    """
    Create an enhanced feature engineer with specified configuration.
    
    Args:
        include_hmm_features: Whether to include HMM model outputs
        include_analyst_features: Whether to include Analyst model outputs (for Tactician)
        enable_feature_scaling: Whether to apply feature scaling
        
    Returns:
        Configured EnhancedFeatureEngineer instance
    """
    config = FeatureEngineeringConfig(
        include_hmm_features=include_hmm_features,
        include_analyst_features=include_analyst_features,
        enable_feature_scaling=enable_feature_scaling
    )
    
    return EnhancedFeatureEngineer(config)


if __name__ == '__main__':
    # Test the enhanced feature engineering
    print("🔧 Testing Enhanced Feature Engineering")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'rsi': np.random.uniform(0, 100, 1000),
        'macd': np.random.uniform(-1, 1, 1000)
    }, index=dates)
    
    # Create mock HMM model
    class MockHMM:
        def predict(self, X):
            return np.random.randint(0, 3, len(X))
        
        def predict_proba(self, X):
            probs = np.random.rand(len(X), 3)
            return probs / probs.sum(axis=1, keepdims=True)
        
        @property
        def transmat_(self):
            return np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    
    # Create mock Analyst model
    class MockAnalyst:
        def predict(self, X):
            return np.random.uniform(0, 1, len(X))
        
        def predict_proba(self, X):
            probs = np.random.rand(len(X), 2)
            return probs / probs.sum(axis=1, keepdims=True)
    
    # Test feature engineering
    hmm_models = {'regime_1': MockHMM(), 'regime_2': MockHMM()}
    analyst_model = MockAnalyst()
    
    # Test Analyst features (no circular dependency)
    print("\n🔍 Testing Analyst feature generation...")
    analyst_feature_engineer = create_enhanced_feature_engineer(
        include_hmm_features=True,
        include_analyst_features=False
    )
    analyst_feature_engineer.set_hmm_models(hmm_models)
    
    analyst_features = analyst_feature_engineer.generate_analyst_features(data)
    print(f"✅ Generated {len(analyst_features.columns)} Analyst features")
    
    # Test Tactician features (with Analyst outputs)
    print("\n🎯 Testing Tactician feature generation...")
    tactician_feature_engineer = create_enhanced_feature_engineer(
        include_hmm_features=True,
        include_analyst_features=True
    )
    tactician_feature_engineer.set_hmm_models(hmm_models)
    tactician_feature_engineer.set_analyst_model(analyst_model)
    
    tactician_features = tactician_feature_engineer.generate_tactician_features(data)
    print(f"✅ Generated {len(tactician_features.columns)} Tactician features")
    
    print('✅ Enhanced Feature Engineering test completed!')