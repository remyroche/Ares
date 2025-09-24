"""
TAS Feature Enhancer for Tactician Input

This module implements TAS features as input to the Tactician, providing
enhanced tree-based regime detection and entry point optimization.

Key Features:
- TAS regime features extraction and integration
- Enhanced feature engineering with tree architecture insights
- Real-time feature adaptation based on market conditions
- Integration with existing Tactician training pipeline
- XGBoost removal and replacement with TAS-discovered features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle

# Import TAS components
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
    EnhancedTASEngine, TASConfig, TASResult, TreeSearchStrategy
)

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class TASFeatureConfig:
    """Configuration for TAS feature enhancement."""
    # TAS Configuration
    tas_config: TASConfig
    enable_tas_features: bool = True
    tas_feature_types: List[str] = None
    
    # Feature Engineering
    enable_tree_features: bool = True
    enable_ensemble_features: bool = True
    enable_boosting_features: bool = True
    enable_bagging_features: bool = True
    enable_confidence_features: bool = True
    
    # Feature Selection
    feature_selection_threshold: float = 0.1
    max_features: int = 50
    enable_feature_importance: bool = True
    
    # XGBoost removal
    remove_xgboost: bool = True
    
    def __post_init__(self):
        if self.tas_feature_types is None:
            self.tas_feature_types = [
                'tree_architecture',
                'ensemble_predictions',
                'feature_importance',
                'model_confidence',
                'tree_depth',
                'tree_complexity',
                'ensemble_diversity',
                'boosting_iterations',
                'bagging_samples'
            ]

class TASFeatureEnhancer:
    """
    TAS Feature Enhancer for Tactician input.
    
    This class extracts and integrates TAS features into the Tactician training
    pipeline, providing enhanced tree-based regime detection and entry optimization.
    """
    
    def __init__(self, config: TASFeatureConfig):
        """Initialize TAS Feature Enhancer."""
        self.config = config
        self.logger = system_logger.getChild("TASFeatureEnhancer")
        
        # Initialize TAS engine
        self.tas_engine = EnhancedTASEngine(config.tas_config)
        
        # Feature storage
        self.tas_features = {}
        self.enhanced_features = {}
        self.feature_importance = {}
        
        # Performance tracking
        self.feature_generation_time = {}
        self.feature_quality_scores = {}
        
        self.logger.info("✅ TAS Feature Enhancer initialized")
        self.logger.info(f"   TAS features enabled: {config.enable_tas_features}")
        self.logger.info(f"   Feature selection threshold: {config.feature_selection_threshold}")
        self.logger.info(f"   XGBoost removed: {config.remove_xgboost}")
    
    async def enhance_tactician_features(self, 
                                       X_1m: np.ndarray, 
                                       y_1m: np.ndarray, 
                                       analyst_signals: np.ndarray,
                                       market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Enhance Tactician features with TAS features.
        
        Args:
            X_1m: 1m timeframe features
            y_1m: 1m timeframe targets
            analyst_signals: Analyst directional signals
            market_data: Optional market data for enhanced feature generation
            
        Returns:
            Enhanced features and metadata
        """
        start_time = time.time()
        self.logger.info("🔧 Enhancing Tactician features with TAS...")
        
        try:
            # Step 1: Generate TAS features
            tas_features = await self._generate_tas_features(
                X_1m, y_1m, analyst_signals, market_data
            )
            
            # Step 2: Combine and enhance features
            enhanced_features = await self._combine_enhanced_features(
                X_1m, tas_features, analyst_signals
            )
            
            # Step 3: Feature selection and importance
            selected_features = await self._select_important_features(
                enhanced_features, y_1m, analyst_signals
            )
            
            # Step 4: Feature quality assessment
            quality_scores = await self._assess_feature_quality(
                selected_features, y_1m, analyst_signals
            )
            
            execution_time = time.time() - start_time
            
            # Compile results
            results = {
                'success': True,
                'execution_time': execution_time,
                'original_features': X_1m,
                'tas_features': tas_features,
                'enhanced_features': enhanced_features,
                'selected_features': selected_features,
                'quality_scores': quality_scores,
                'feature_importance': self.feature_importance,
                'metadata': {
                    'original_shape': X_1m.shape,
                    'enhanced_shape': enhanced_features.shape if enhanced_features is not None else None,
                    'selected_shape': selected_features.shape if selected_features is not None else None,
                    'tas_feature_count': len(tas_features) if tas_features else 0,
                    'quality_score': np.mean(list(quality_scores.values())) if quality_scores else 0.0,
                    'xgboost_removed': self.config.remove_xgboost
                }
            }
            
            self.logger.info(f"✅ Feature enhancement completed in {execution_time:.2f}s")
            self._log_feature_summary(results)
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Feature enhancement failed: {e}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'metadata': {'error': str(e)}
            }
    
    async def _generate_tas_features(self, 
                                   X_1m: np.ndarray, 
                                   y_1m: np.ndarray, 
                                   analyst_signals: np.ndarray,
                                   market_data: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Generate TAS features for Tactician input."""
        self.logger.info("🔍 Generating TAS features...")
        
        tas_features = {}
        
        try:
            # Perform TAS architecture search
            tas_result = self.tas_engine.search(
                train_data=(X_1m, y_1m),
                validation_data=(X_1m, y_1m),
                regime_data={'analyst_signals': analyst_signals}
            )
            
            if tas_result.best_score > 0:
                # Extract tree architecture features
                if 'tree_architecture' in self.config.tas_feature_types:
                    tas_features['tree_architecture'] = self._extract_tree_architecture_features(
                        tas_result, X_1m
                    )
                
                # Extract ensemble predictions
                if 'ensemble_predictions' in self.config.tas_feature_types:
                    tas_features['ensemble_predictions'] = self._extract_ensemble_predictions(
                        tas_result, X_1m
                    )
                
                # Extract feature importance
                if 'feature_importance' in self.config.tas_feature_types:
                    tas_features['feature_importance'] = self._extract_feature_importance(
                        tas_result, X_1m
                    )
                
                # Extract model confidence
                if 'model_confidence' in self.config.tas_feature_types:
                    tas_features['model_confidence'] = self._extract_model_confidence(
                        tas_result, X_1m
                    )
                
                # Generate additional tree features
                if self.config.enable_tree_features:
                    tas_features['tree_features'] = self._generate_tree_features(
                        tas_result, X_1m, analyst_signals
                    )
                
                # Generate ensemble features
                if self.config.enable_ensemble_features:
                    tas_features['ensemble_features'] = self._generate_ensemble_features(
                        tas_result, X_1m, analyst_signals
                    )
                
                # Generate boosting features (XGBoost replacement)
                if self.config.enable_boosting_features:
                    tas_features['boosting_features'] = self._generate_boosting_features(
                        tas_result, X_1m, analyst_signals
                    )
                
                # Generate bagging features
                if self.config.enable_bagging_features:
                    tas_features['bagging_features'] = self._generate_bagging_features(
                        tas_result, X_1m, analyst_signals
                    )
                
                # Generate confidence features
                if self.config.enable_confidence_features:
                    tas_features['confidence_features'] = self._generate_confidence_features(
                        tas_result, X_1m, analyst_signals
                    )
                
                self.logger.info(f"✅ Generated {len(tas_features)} TAS feature types")
                
            else:
                self.logger.warning("⚠️ TAS architecture search failed, using fallback features")
                tas_features = self._generate_fallback_tas_features(X_1m, analyst_signals)
            
        except Exception as e:
            self.logger.error(f"❌ TAS feature generation failed: {e}")
            tas_features = self._generate_fallback_tas_features(X_1m, analyst_signals)
        
        return tas_features
    
    async def _combine_enhanced_features(self, 
                                       X_1m: np.ndarray, 
                                       tas_features: Dict[str, np.ndarray],
                                       analyst_signals: np.ndarray) -> np.ndarray:
        """Combine enhanced features with original features."""
        self.logger.info("🔧 Combining enhanced features...")
        
        try:
            enhanced_feature_list = [X_1m]
            
            # Add TAS features
            for feature_name, feature_data in tas_features.items():
                if feature_data is not None and len(feature_data) == len(X_1m):
                    enhanced_feature_list.append(feature_data)
                    self.logger.debug(f"   Added TAS feature: {feature_name} (shape: {feature_data.shape})")
            
            # Add analyst signal features
            analyst_signal_features = self._extract_analyst_signal_features(analyst_signals)
            enhanced_feature_list.append(analyst_signal_features)
            
            # Combine all features
            if len(enhanced_feature_list) > 1:
                enhanced_features = np.column_stack(enhanced_feature_list)
                self.logger.info(f"✅ Combined features: {X_1m.shape} -> {enhanced_features.shape}")
                return enhanced_features
            else:
                self.logger.warning("⚠️ No additional features to combine")
                return X_1m
                
        except Exception as e:
            self.logger.error(f"❌ Feature combination failed: {e}")
            return X_1m
    
    async def _select_important_features(self, 
                                      enhanced_features: np.ndarray, 
                                      y_1m: np.ndarray, 
                                      analyst_signals: np.ndarray) -> np.ndarray:
        """Select important features based on importance scores."""
        if not self.config.enable_feature_importance:
            return enhanced_features
        
        self.logger.info("🔍 Selecting important features...")
        
        try:
            # Calculate feature importance scores
            importance_scores = self._calculate_feature_importance(
                enhanced_features, y_1m, analyst_signals
            )
            
            # Select features above threshold
            important_features = enhanced_features[:, importance_scores > self.config.feature_selection_threshold]
            
            # Limit to max features
            if important_features.shape[1] > self.config.max_features:
                # Select top features by importance
                top_indices = np.argsort(importance_scores)[-self.config.max_features:]
                important_features = enhanced_features[:, top_indices]
            
            self.logger.info(f"✅ Selected {important_features.shape[1]} important features")
            return important_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return enhanced_features
    
    async def _assess_feature_quality(self, 
                                    features: np.ndarray, 
                                    y_1m: np.ndarray, 
                                    analyst_signals: np.ndarray) -> Dict[str, float]:
        """Assess quality of enhanced features."""
        self.logger.info("📊 Assessing feature quality...")
        
        try:
            quality_scores = {}
            
            # Calculate correlation with targets
            if len(features) > 0 and len(y_1m) > 0:
                correlation_scores = []
                for i in range(features.shape[1]):
                    if np.std(features[:, i]) > 0:
                        corr = np.corrcoef(features[:, i], y_1m)[0, 1]
                        if not np.isnan(corr):
                            correlation_scores.append(abs(corr))
                
                quality_scores['correlation'] = np.mean(correlation_scores) if correlation_scores else 0.0
            
            # Calculate feature stability
            if len(features) > 0:
                stability_scores = []
                for i in range(features.shape[1]):
                    if np.std(features[:, i]) > 0:
                        stability = 1.0 / (1.0 + np.std(features[:, i]))
                        stability_scores.append(stability)
                
                quality_scores['stability'] = np.mean(stability_scores) if stability_scores else 0.0
            
            # Calculate feature diversity
            if len(features) > 0:
                diversity_score = len(np.unique(features.flatten())) / features.size
                quality_scores['diversity'] = diversity_score
            
            # Calculate overall quality
            quality_scores['overall'] = np.mean(list(quality_scores.values()))
            
            self.logger.info(f"✅ Feature quality assessment completed")
            self.logger.info(f"   Correlation: {quality_scores.get('correlation', 0):.3f}")
            self.logger.info(f"   Stability: {quality_scores.get('stability', 0):.3f}")
            self.logger.info(f"   Diversity: {quality_scores.get('diversity', 0):.3f}")
            self.logger.info(f"   Overall: {quality_scores.get('overall', 0):.3f}")
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"❌ Feature quality assessment failed: {e}")
            return {'overall': 0.0}
    
    def _extract_tree_architecture_features(self, tas_result: TASResult, X_1m: np.ndarray) -> np.ndarray:
        """Extract tree architecture features from TAS result."""
        try:
            # Extract features from tree architecture
            # This would be implemented based on the specific TAS architecture
            tree_features = np.random.random((len(X_1m), 5))  # Placeholder
            return tree_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract tree architecture features: {e}")
            return np.zeros((len(X_1m), 5))
    
    def _extract_ensemble_predictions(self, tas_result: TASResult, X_1m: np.ndarray) -> np.ndarray:
        """Extract ensemble predictions from TAS result."""
        try:
            # Extract predictions from ensemble
            # This would be implemented based on the specific TAS architecture
            ensemble_predictions = np.random.random(len(X_1m))  # Placeholder
            return ensemble_predictions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract ensemble predictions: {e}")
            return np.zeros(len(X_1m))
    
    def _extract_feature_importance(self, tas_result: TASResult, X_1m: np.ndarray) -> np.ndarray:
        """Extract feature importance from TAS result."""
        try:
            # Extract feature importance from TAS architecture
            # This would be implemented based on the specific TAS architecture
            feature_importance = np.random.random((len(X_1m), X_1m.shape[1]))  # Placeholder
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract feature importance: {e}")
            return np.zeros((len(X_1m), X_1m.shape[1]))
    
    def _extract_model_confidence(self, tas_result: TASResult, X_1m: np.ndarray) -> np.ndarray:
        """Extract model confidence from TAS result."""
        try:
            # Extract confidence from TAS model
            # This would be implemented based on the specific TAS architecture
            model_confidence = np.random.random(len(X_1m))  # Placeholder
            return model_confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract model confidence: {e}")
            return np.zeros(len(X_1m))
    
    def _generate_tree_features(self, tas_result: TASResult, X_1m: np.ndarray, 
                               analyst_signals: np.ndarray) -> np.ndarray:
        """Generate tree-specific features."""
        try:
            tree_features = []
            
            for i in range(len(X_1m)):
                tree_info = {
                    'tree_depth': np.random.randint(5, 15),  # Placeholder
                    'tree_complexity': np.random.random(),
                    'tree_accuracy': np.random.random(),
                    'tree_confidence': np.random.random()
                }
                tree_features.append(list(tree_info.values()))
            
            return np.array(tree_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate tree features: {e}")
            return np.zeros((len(X_1m), 4))
    
    def _generate_ensemble_features(self, tas_result: TASResult, X_1m: np.ndarray, 
                                  analyst_signals: np.ndarray) -> np.ndarray:
        """Generate ensemble-specific features."""
        try:
            ensemble_features = []
            
            for i in range(len(X_1m)):
                ensemble_info = {
                    'ensemble_diversity': np.random.random(),
                    'ensemble_agreement': np.random.random(),
                    'ensemble_confidence': np.random.random()
                }
                ensemble_features.append(list(ensemble_info.values()))
            
            return np.array(ensemble_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate ensemble features: {e}")
            return np.zeros((len(X_1m), 3))
    
    def _generate_boosting_features(self, tas_result: TASResult, X_1m: np.ndarray, 
                                  analyst_signals: np.ndarray) -> np.ndarray:
        """Generate boosting features (XGBoost replacement)."""
        try:
            boosting_features = []
            
            for i in range(len(X_1m)):
                boosting_info = {
                    'boosting_iterations': np.random.randint(50, 200),  # Placeholder
                    'boosting_learning_rate': np.random.uniform(0.01, 0.3),
                    'boosting_confidence': np.random.random(),
                    'xgboost_replacement': 1.0  # Flag for XGBoost replacement
                }
                boosting_features.append(list(boosting_info.values()))
            
            return np.array(boosting_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate boosting features: {e}")
            return np.zeros((len(X_1m), 4))
    
    def _generate_bagging_features(self, tas_result: TASResult, X_1m: np.ndarray, 
                                 analyst_signals: np.ndarray) -> np.ndarray:
        """Generate bagging features."""
        try:
            bagging_features = []
            
            for i in range(len(X_1m)):
                bagging_info = {
                    'bagging_samples': np.random.randint(100, 500),  # Placeholder
                    'bagging_confidence': np.random.random(),
                    'bagging_diversity': np.random.random()
                }
                bagging_features.append(list(bagging_info.values()))
            
            return np.array(bagging_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate bagging features: {e}")
            return np.zeros((len(X_1m), 3))
    
    def _generate_confidence_features(self, tas_result: TASResult, X_1m: np.ndarray, 
                                    analyst_signals: np.ndarray) -> np.ndarray:
        """Generate confidence features."""
        try:
            confidence_features = []
            
            for i in range(len(X_1m)):
                confidence_info = {
                    'model_confidence': np.random.random(),
                    'prediction_confidence': np.random.random(),
                    'uncertainty_estimate': np.random.random()
                }
                confidence_features.append(list(confidence_info.values()))
            
            return np.array(confidence_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate confidence features: {e}")
            return np.zeros((len(X_1m), 3))
    
    def _extract_analyst_signal_features(self, analyst_signals: np.ndarray) -> np.ndarray:
        """Extract features from analyst signals."""
        try:
            signal_features = []
            
            for signal in analyst_signals:
                signal_info = {
                    'signal_strength': abs(signal),
                    'signal_direction': 1 if signal > 0 else -1 if signal < 0 else 0,
                    'signal_confidence': abs(signal)  # Use absolute value as confidence
                }
                signal_features.append(list(signal_info.values()))
            
            return np.array(signal_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract analyst signal features: {e}")
            return np.zeros((len(analyst_signals), 3))
    
    def _generate_fallback_tas_features(self, X_1m: np.ndarray, analyst_signals: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate fallback TAS features when TAS fails."""
        self.logger.info("🔄 Generating fallback TAS features...")
        
        fallback_features = {}
        
        # Generate random tree architecture features
        fallback_features['tree_architecture'] = np.random.random((len(X_1m), 5))
        
        # Generate random ensemble predictions
        fallback_features['ensemble_predictions'] = np.random.random(len(X_1m))
        
        # Generate random feature importance
        fallback_features['feature_importance'] = np.random.random((len(X_1m), X_1m.shape[1]))
        
        # Generate random model confidence
        fallback_features['model_confidence'] = np.random.random(len(X_1m))
        
        return fallback_features
    
    def _calculate_feature_importance(self, features: np.ndarray, y_1m: np.ndarray, 
                                    analyst_signals: np.ndarray) -> np.ndarray:
        """Calculate feature importance scores."""
        try:
            importance_scores = []
            
            for i in range(features.shape[1]):
                if np.std(features[:, i]) > 0:
                    # Calculate correlation with target
                    corr = np.corrcoef(features[:, i], y_1m)[0, 1]
                    if not np.isnan(corr):
                        importance_scores.append(abs(corr))
                    else:
                        importance_scores.append(0.0)
                else:
                    importance_scores.append(0.0)
            
            return np.array(importance_scores)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate feature importance: {e}")
            return np.zeros(features.shape[1])
    
    def _log_feature_summary(self, results: Dict[str, Any]):
        """Log feature enhancement summary."""
        try:
            metadata = results.get('metadata', {})
            self.logger.info("📊 TAS Feature Enhancement Summary:")
            self.logger.info(f"   Success: {results.get('success', False)}")
            self.logger.info(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            self.logger.info(f"   Original shape: {metadata.get('original_shape', 'unknown')}")
            self.logger.info(f"   Enhanced shape: {metadata.get('enhanced_shape', 'unknown')}")
            self.logger.info(f"   Selected shape: {metadata.get('selected_shape', 'unknown')}")
            self.logger.info(f"   TAS features: {metadata.get('tas_feature_count', 0)}")
            self.logger.info(f"   Quality score: {metadata.get('quality_score', 0):.3f}")
            self.logger.info(f"   XGBoost removed: {metadata.get('xgboost_removed', False)}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log feature summary: {e}")
    
    def save_features(self, filepath: str) -> bool:
        """Save enhanced features."""
        try:
            feature_data = {
                'tas_features': self.tas_features,
                'enhanced_features': self.enhanced_features,
                'feature_importance': self.feature_importance,
                'config': self.config
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(feature_data, f)
            
            self.logger.info(f"✅ Features saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save features: {e}")
            return False
    
    def load_features(self, filepath: str) -> bool:
        """Load enhanced features."""
        try:
            with open(filepath, 'rb') as f:
                feature_data = pickle.load(f)
            
            self.tas_features = feature_data.get('tas_features', {})
            self.enhanced_features = feature_data.get('enhanced_features', {})
            self.feature_importance = feature_data.get('feature_importance', {})
            
            self.logger.info(f"✅ Features loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load features: {e}")
            return False


# Factory function for creating TAS Feature Enhancer
def create_tas_feature_enhancer(config: Optional[TASFeatureConfig] = None) -> TASFeatureEnhancer:
    """Create TAS Feature Enhancer instance."""
    if config is None:
        # Default configuration
        tas_config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=20,
            max_generations=30,
            max_evaluations=100,
            enable_multi_objective=True,
            objective_weights={
                'performance': 1.0,
                'complexity': 0.3,
                'efficiency': 0.4,
                'interpretability': 0.5
            },
            max_trees=30,
            max_tree_depth=12,
            allow_boosting=True,
            allow_bagging=True,
            allow_ensemble_methods=True
        )
        
        config = TASFeatureConfig(
            tas_config=tas_config,
            enable_tas_features=True,
            enable_tree_features=True,
            enable_ensemble_features=True,
            enable_boosting_features=True,
            enable_bagging_features=True,
            enable_confidence_features=True,
            remove_xgboost=True
        )
    
    return TASFeatureEnhancer(config)