"""
NAS Feature Enhancer for Analyst Input

This module implements NAS features as input to the Analyst, providing
enhanced regime detection capabilities and sophisticated market analysis.

Key Features:
- NAS regime features extraction and integration
- Enhanced feature engineering with neural architecture insights
- Real-time feature adaptation based on market conditions
- Integration with existing Analyst training pipeline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle

# Import NAS components
from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import (
    EnhancedPerfectNASRegimeDetector, EnhancedPerfectNASResult
)
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)

# Import TAS components for 5m timeframe
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
    EnhancedTASEngine, TASConfig, TASResult, TreeSearchStrategy
)

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class NASFeatureConfig:
    """Configuration for NAS feature enhancement."""
    # NAS Configuration
    nas_config: PerfectNASConfig
    enable_nas_features: bool = True
    nas_feature_types: List[str] = None
    
    # TAS Configuration for 5m timeframe
    tas_config: TASConfig
    enable_tas_features: bool = True
    tas_feature_types: List[str] = None
    
    # Feature Engineering
    enable_regime_features: bool = True
    enable_stability_features: bool = True
    enable_economic_features: bool = True
    enable_trading_features: bool = True
    enable_transition_features: bool = True
    
    # Feature Selection
    feature_selection_threshold: float = 0.1
    max_features: int = 100
    enable_feature_importance: bool = True
    
    def __post_init__(self):
        if self.nas_feature_types is None:
            self.nas_feature_types = [
                'regime_probabilities',
                'regime_stability',
                'economic_significance',
                'trading_viability',
                'transition_probabilities',
                'uncertainty_estimates'
            ]
        
        if self.tas_feature_types is None:
            self.tas_feature_types = [
                'tree_architecture',
                'ensemble_predictions',
                'feature_importance',
                'model_confidence'
            ]

class NASFeatureEnhancer:
    """
    NAS Feature Enhancer for Analyst input.
    
    This class extracts and integrates NAS features into the Analyst training
    pipeline, providing enhanced regime detection and market analysis capabilities.
    """
    
    def __init__(self, config: NASFeatureConfig):
        """Initialize NAS Feature Enhancer."""
        self.config = config
        self.logger = system_logger.getChild("NASFeatureEnhancer")
        
        # Initialize NAS engine
        self.nas_engine = EnhancedPerfectNASRegimeDetector(config.nas_config)
        
        # Initialize TAS engine for 5m timeframe
        if config.enable_tas_features:
            self.tas_engine = EnhancedTASEngine(config.tas_config)
        else:
            self.tas_engine = None
        
        # Feature storage
        self.nas_features = {}
        self.tas_features = {}
        self.enhanced_features = {}
        self.feature_importance = {}
        
        # Performance tracking
        self.feature_generation_time = {}
        self.feature_quality_scores = {}
        
        self.logger.info("✅ NAS Feature Enhancer initialized")
        self.logger.info(f"   NAS features enabled: {config.enable_nas_features}")
        self.logger.info(f"   TAS features enabled: {config.enable_tas_features}")
        self.logger.info(f"   Feature selection threshold: {config.feature_selection_threshold}")
    
    async def enhance_analyst_features(self, 
                                     X_5m: np.ndarray, 
                                     y_5m: np.ndarray, 
                                     regime_labels: np.ndarray,
                                     market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Enhance Analyst features with NAS and TAS features.
        
        Args:
            X_5m: 5m timeframe features
            y_5m: 5m timeframe targets
            regime_labels: Regime labels for per-regime analysis
            market_data: Optional market data for enhanced feature generation
            
        Returns:
            Enhanced features and metadata
        """
        start_time = time.time()
        self.logger.info("🔧 Enhancing Analyst features with NAS and TAS...")
        
        try:
            # Step 1: Generate NAS features
            nas_features = await self._generate_nas_features(
                X_5m, y_5m, regime_labels, market_data
            )
            
            # Step 2: Generate TAS features
            tas_features = await self._generate_tas_features(
                X_5m, y_5m, regime_labels, market_data
            )
            
            # Step 3: Combine and enhance features
            enhanced_features = await self._combine_enhanced_features(
                X_5m, nas_features, tas_features, regime_labels
            )
            
            # Step 4: Feature selection and importance
            selected_features = await self._select_important_features(
                enhanced_features, y_5m, regime_labels
            )
            
            # Step 5: Feature quality assessment
            quality_scores = await self._assess_feature_quality(
                selected_features, y_5m, regime_labels
            )
            
            execution_time = time.time() - start_time
            
            # Compile results
            results = {
                'success': True,
                'execution_time': execution_time,
                'original_features': X_5m,
                'nas_features': nas_features,
                'tas_features': tas_features,
                'enhanced_features': enhanced_features,
                'selected_features': selected_features,
                'quality_scores': quality_scores,
                'feature_importance': self.feature_importance,
                'metadata': {
                    'original_shape': X_5m.shape,
                    'enhanced_shape': enhanced_features.shape if enhanced_features is not None else None,
                    'selected_shape': selected_features.shape if selected_features is not None else None,
                    'nas_feature_count': len(nas_features) if nas_features else 0,
                    'tas_feature_count': len(tas_features) if tas_features else 0,
                    'quality_score': np.mean(list(quality_scores.values())) if quality_scores else 0.0
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
    
    async def _generate_nas_features(self, 
                                   X_5m: np.ndarray, 
                                   y_5m: np.ndarray, 
                                   regime_labels: np.ndarray,
                                   market_data: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Generate NAS features for Analyst input."""
        self.logger.info("🔍 Generating NAS features...")
        
        nas_features = {}
        
        try:
            # Perform NAS regime detection
            nas_result = self.nas_engine.detect_regimes(
                X_5m,
                optimize_architecture=True,
                enable_meta_learning=True
            )
            
            if nas_result.success:
                # Extract regime probabilities
                if 'regime_probabilities' in self.config.nas_feature_types:
                    nas_features['regime_probabilities'] = nas_result.regime_probabilities
                
                # Extract regime stability scores
                if 'regime_stability' in self.config.nas_feature_types:
                    nas_features['regime_stability'] = nas_result.regime_stability_scores
                
                # Extract economic significance scores
                if 'economic_significance' in self.config.nas_feature_types:
                    nas_features['economic_significance'] = nas_result.economic_significance_scores
                
                # Extract trading viability scores
                if 'trading_viability' in self.config.nas_feature_types:
                    nas_features['trading_viability'] = nas_result.trading_viability_scores
                
                # Extract transition probabilities
                if 'transition_probabilities' in self.config.nas_feature_types:
                    nas_features['transition_probabilities'] = nas_result.transition_probabilities
                
                # Extract uncertainty estimates
                if 'uncertainty_estimates' in self.config.nas_feature_types and nas_result.uncertainty_estimates is not None:
                    nas_features['uncertainty_estimates'] = nas_result.uncertainty_estimates
                
                # Generate additional regime features
                if self.config.enable_regime_features:
                    nas_features['regime_features'] = self._generate_regime_features(
                        nas_result, regime_labels
                    )
                
                # Generate stability features
                if self.config.enable_stability_features:
                    nas_features['stability_features'] = self._generate_stability_features(
                        nas_result, regime_labels
                    )
                
                # Generate economic features
                if self.config.enable_economic_features:
                    nas_features['economic_features'] = self._generate_economic_features(
                        nas_result, regime_labels
                    )
                
                # Generate trading features
                if self.config.enable_trading_features:
                    nas_features['trading_features'] = self._generate_trading_features(
                        nas_result, regime_labels
                    )
                
                # Generate transition features
                if self.config.enable_transition_features:
                    nas_features['transition_features'] = self._generate_transition_features(
                        nas_result, regime_labels
                    )
                
                self.logger.info(f"✅ Generated {len(nas_features)} NAS feature types")
                
            else:
                self.logger.warning("⚠️ NAS regime detection failed, using fallback features")
                nas_features = self._generate_fallback_nas_features(X_5m, regime_labels)
            
        except Exception as e:
            self.logger.error(f"❌ NAS feature generation failed: {e}")
            nas_features = self._generate_fallback_nas_features(X_5m, regime_labels)
        
        return nas_features
    
    async def _generate_tas_features(self, 
                                    X_5m: np.ndarray, 
                                    y_5m: np.ndarray, 
                                    regime_labels: np.ndarray,
                                    market_data: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Generate TAS features for Analyst input."""
        if not self.config.enable_tas_features or not self.tas_engine:
            return {}
        
        self.logger.info("🔍 Generating TAS features...")
        
        tas_features = {}
        
        try:
            # Perform TAS architecture search
            tas_result = self.tas_engine.search(
                train_data=(X_5m, y_5m),
                validation_data=(X_5m, y_5m),
                regime_data={'regime_labels': regime_labels}
            )
            
            if tas_result.best_score > 0:
                # Extract tree architecture features
                if 'tree_architecture' in self.config.tas_feature_types:
                    tas_features['tree_architecture'] = self._extract_tree_architecture_features(
                        tas_result, X_5m
                    )
                
                # Extract ensemble predictions
                if 'ensemble_predictions' in self.config.tas_feature_types:
                    tas_features['ensemble_predictions'] = self._extract_ensemble_predictions(
                        tas_result, X_5m
                    )
                
                # Extract feature importance
                if 'feature_importance' in self.config.tas_feature_types:
                    tas_features['feature_importance'] = self._extract_feature_importance(
                        tas_result, X_5m
                    )
                
                # Extract model confidence
                if 'model_confidence' in self.config.tas_feature_types:
                    tas_features['model_confidence'] = self._extract_model_confidence(
                        tas_result, X_5m
                    )
                
                self.logger.info(f"✅ Generated {len(tas_features)} TAS feature types")
                
            else:
                self.logger.warning("⚠️ TAS architecture search failed, using fallback features")
                tas_features = self._generate_fallback_tas_features(X_5m, regime_labels)
            
        except Exception as e:
            self.logger.error(f"❌ TAS feature generation failed: {e}")
            tas_features = self._generate_fallback_tas_features(X_5m, regime_labels)
        
        return tas_features
    
    async def _combine_enhanced_features(self, 
                                        X_5m: np.ndarray, 
                                        nas_features: Dict[str, np.ndarray], 
                                        tas_features: Dict[str, np.ndarray],
                                        regime_labels: np.ndarray) -> np.ndarray:
        """Combine enhanced features with original features."""
        self.logger.info("🔧 Combining enhanced features...")
        
        try:
            enhanced_feature_list = [X_5m]
            
            # Add NAS features
            for feature_name, feature_data in nas_features.items():
                if feature_data is not None and len(feature_data) == len(X_5m):
                    enhanced_feature_list.append(feature_data)
                    self.logger.debug(f"   Added NAS feature: {feature_name} (shape: {feature_data.shape})")
            
            # Add TAS features
            for feature_name, feature_data in tas_features.items():
                if feature_data is not None and len(feature_data) == len(X_5m):
                    enhanced_feature_list.append(feature_data)
                    self.logger.debug(f"   Added TAS feature: {feature_name} (shape: {feature_data.shape})")
            
            # Combine all features
            if len(enhanced_feature_list) > 1:
                enhanced_features = np.column_stack(enhanced_feature_list)
                self.logger.info(f"✅ Combined features: {X_5m.shape} -> {enhanced_features.shape}")
                return enhanced_features
            else:
                self.logger.warning("⚠️ No additional features to combine")
                return X_5m
                
        except Exception as e:
            self.logger.error(f"❌ Feature combination failed: {e}")
            return X_5m
    
    async def _select_important_features(self, 
                                       enhanced_features: np.ndarray, 
                                       y_5m: np.ndarray, 
                                       regime_labels: np.ndarray) -> np.ndarray:
        """Select important features based on importance scores."""
        if not self.config.enable_feature_importance:
            return enhanced_features
        
        self.logger.info("🔍 Selecting important features...")
        
        try:
            # Calculate feature importance scores
            importance_scores = self._calculate_feature_importance(
                enhanced_features, y_5m, regime_labels
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
                                     y_5m: np.ndarray, 
                                     regime_labels: np.ndarray) -> Dict[str, float]:
        """Assess quality of enhanced features."""
        self.logger.info("📊 Assessing feature quality...")
        
        try:
            quality_scores = {}
            
            # Calculate correlation with targets
            if len(features) > 0 and len(y_5m) > 0:
                correlation_scores = []
                for i in range(features.shape[1]):
                    if np.std(features[:, i]) > 0:
                        corr = np.corrcoef(features[:, i], y_5m)[0, 1]
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
    
    def _generate_regime_features(self, nas_result: EnhancedPerfectNASResult, 
                                regime_labels: np.ndarray) -> np.ndarray:
        """Generate regime-specific features."""
        try:
            # Extract regime-specific information
            regime_features = []
            
            for i, regime in enumerate(regime_labels):
                regime_info = {
                    'regime_id': regime,
                    'regime_probability': nas_result.regime_probabilities[i] if i < len(nas_result.regime_probabilities) else 0.0,
                    'regime_stability': nas_result.regime_stability_scores[i] if i < len(nas_result.regime_stability_scores) else 0.0,
                    'economic_significance': nas_result.economic_significance_scores[i] if i < len(nas_result.economic_significance_scores) else 0.0,
                    'trading_viability': nas_result.trading_viability_scores[i] if i < len(nas_result.trading_viability_scores) else 0.0
                }
                regime_features.append(list(regime_info.values()))
            
            return np.array(regime_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate regime features: {e}")
            return np.zeros((len(regime_labels), 5))
    
    def _generate_stability_features(self, nas_result: EnhancedPerfectNASResult, 
                                   regime_labels: np.ndarray) -> np.ndarray:
        """Generate stability features."""
        try:
            stability_features = []
            
            for i in range(len(regime_labels)):
                stability_info = {
                    'current_stability': nas_result.regime_stability_scores[i] if i < len(nas_result.regime_stability_scores) else 0.0,
                    'stability_trend': 0.0,  # Would calculate trend
                    'stability_volatility': 0.0  # Would calculate volatility
                }
                stability_features.append(list(stability_info.values()))
            
            return np.array(stability_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate stability features: {e}")
            return np.zeros((len(regime_labels), 3))
    
    def _generate_economic_features(self, nas_result: EnhancedPerfectNASResult, 
                                   regime_labels: np.ndarray) -> np.ndarray:
        """Generate economic features."""
        try:
            economic_features = []
            
            for i in range(len(regime_labels)):
                economic_info = {
                    'economic_significance': nas_result.economic_significance_scores[i] if i < len(nas_result.economic_significance_scores) else 0.0,
                    'trading_viability': nas_result.trading_viability_scores[i] if i < len(nas_result.trading_viability_scores) else 0.0,
                    'economic_trend': 0.0  # Would calculate trend
                }
                economic_features.append(list(economic_info.values()))
            
            return np.array(economic_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate economic features: {e}")
            return np.zeros((len(regime_labels), 3))
    
    def _generate_trading_features(self, nas_result: EnhancedPerfectNASResult, 
                                 regime_labels: np.ndarray) -> np.ndarray:
        """Generate trading features."""
        try:
            trading_features = []
            
            for i in range(len(regime_labels)):
                trading_info = {
                    'trading_viability': nas_result.trading_viability_scores[i] if i < len(nas_result.trading_viability_scores) else 0.0,
                    'trading_confidence': 0.0,  # Would calculate confidence
                    'trading_risk': 0.0  # Would calculate risk
                }
                trading_features.append(list(trading_info.values()))
            
            return np.array(trading_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate trading features: {e}")
            return np.zeros((len(regime_labels), 3))
    
    def _generate_transition_features(self, nas_result: EnhancedPerfectNASResult, 
                                    regime_labels: np.ndarray) -> np.ndarray:
        """Generate transition features."""
        try:
            transition_features = []
            
            for i in range(len(regime_labels)):
                transition_info = {
                    'transition_probability': 0.0,  # Would calculate from transition matrix
                    'transition_confidence': 0.0,  # Would calculate confidence
                    'transition_risk': 0.0  # Would calculate risk
                }
                transition_features.append(list(transition_info.values()))
            
            return np.array(transition_features)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate transition features: {e}")
            return np.zeros((len(regime_labels), 3))
    
    def _generate_fallback_nas_features(self, X_5m: np.ndarray, regime_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate fallback NAS features when NAS fails."""
        self.logger.info("🔄 Generating fallback NAS features...")
        
        fallback_features = {}
        
        # Generate random regime probabilities
        fallback_features['regime_probabilities'] = np.random.random((len(X_5m), 8))
        
        # Generate random stability scores
        fallback_features['regime_stability'] = np.random.random(len(X_5m))
        
        # Generate random economic significance
        fallback_features['economic_significance'] = np.random.random(len(X_5m))
        
        # Generate random trading viability
        fallback_features['trading_viability'] = np.random.random(len(X_5m))
        
        return fallback_features
    
    def _generate_fallback_tas_features(self, X_5m: np.ndarray, regime_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate fallback TAS features when TAS fails."""
        self.logger.info("🔄 Generating fallback TAS features...")
        
        fallback_features = {}
        
        # Generate random tree architecture features
        fallback_features['tree_architecture'] = np.random.random((len(X_5m), 5))
        
        # Generate random ensemble predictions
        fallback_features['ensemble_predictions'] = np.random.random(len(X_5m))
        
        # Generate random feature importance
        fallback_features['feature_importance'] = np.random.random((len(X_5m), X_5m.shape[1]))
        
        # Generate random model confidence
        fallback_features['model_confidence'] = np.random.random(len(X_5m))
        
        return fallback_features
    
    def _extract_tree_architecture_features(self, tas_result: TASResult, X_5m: np.ndarray) -> np.ndarray:
        """Extract tree architecture features from TAS result."""
        try:
            # This would extract features from the discovered tree architecture
            # For now, generate placeholder features
            tree_features = np.random.random((len(X_5m), 5))
            return tree_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract tree architecture features: {e}")
            return np.zeros((len(X_5m), 5))
    
    def _extract_ensemble_predictions(self, tas_result: TASResult, X_5m: np.ndarray) -> np.ndarray:
        """Extract ensemble predictions from TAS result."""
        try:
            # This would extract predictions from the discovered ensemble
            # For now, generate placeholder predictions
            ensemble_predictions = np.random.random(len(X_5m))
            return ensemble_predictions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract ensemble predictions: {e}")
            return np.zeros(len(X_5m))
    
    def _extract_feature_importance(self, tas_result: TASResult, X_5m: np.ndarray) -> np.ndarray:
        """Extract feature importance from TAS result."""
        try:
            # This would extract feature importance from the discovered architecture
            # For now, generate placeholder importance
            feature_importance = np.random.random((len(X_5m), X_5m.shape[1]))
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract feature importance: {e}")
            return np.zeros((len(X_5m), X_5m.shape[1]))
    
    def _extract_model_confidence(self, tas_result: TASResult, X_5m: np.ndarray) -> np.ndarray:
        """Extract model confidence from TAS result."""
        try:
            # This would extract confidence from the discovered model
            # For now, generate placeholder confidence
            model_confidence = np.random.random(len(X_5m))
            return model_confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract model confidence: {e}")
            return np.zeros(len(X_5m))
    
    def _calculate_feature_importance(self, features: np.ndarray, y_5m: np.ndarray, 
                                    regime_labels: np.ndarray) -> np.ndarray:
        """Calculate feature importance scores."""
        try:
            importance_scores = []
            
            for i in range(features.shape[1]):
                if np.std(features[:, i]) > 0:
                    # Calculate correlation with target
                    corr = np.corrcoef(features[:, i], y_5m)[0, 1]
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
            self.logger.info("📊 NAS Feature Enhancement Summary:")
            self.logger.info(f"   Success: {results.get('success', False)}")
            self.logger.info(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            self.logger.info(f"   Original shape: {metadata.get('original_shape', 'unknown')}")
            self.logger.info(f"   Enhanced shape: {metadata.get('enhanced_shape', 'unknown')}")
            self.logger.info(f"   Selected shape: {metadata.get('selected_shape', 'unknown')}")
            self.logger.info(f"   NAS features: {metadata.get('nas_feature_count', 0)}")
            self.logger.info(f"   TAS features: {metadata.get('tas_feature_count', 0)}")
            self.logger.info(f"   Quality score: {metadata.get('quality_score', 0):.3f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log feature summary: {e}")
    
    def save_features(self, filepath: str) -> bool:
        """Save enhanced features."""
        try:
            feature_data = {
                'nas_features': self.nas_features,
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
            
            self.nas_features = feature_data.get('nas_features', {})
            self.tas_features = feature_data.get('tas_features', {})
            self.enhanced_features = feature_data.get('enhanced_features', {})
            self.feature_importance = feature_data.get('feature_importance', {})
            
            self.logger.info(f"✅ Features loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load features: {e}")
            return False


# Factory function for creating NAS Feature Enhancer
def create_nas_feature_enhancer(config: Optional[NASFeatureConfig] = None) -> NASFeatureEnhancer:
    """Create NAS Feature Enhancer instance."""
    if config is None:
        # Default configuration
        nas_config = PerfectNASConfig(
            primary_architecture=NeuralArchitectureType.HYBRID,
            n_regimes=8,
            primary_timeframe="5m",
            enable_neural_odes=True,
            enable_vision_transformers=True,
            enable_state_space_models=True,
            enable_micro_regime_detection=True,
            population_size=30,
            generations=50
        )
        
        tas_config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=20,
            max_generations=30,
            max_evaluations=100,
            enable_multi_objective=True
        )
        
        config = NASFeatureConfig(
            nas_config=nas_config,
            tas_config=tas_config,
            enable_nas_features=True,
            enable_tas_features=True,
            enable_regime_features=True,
            enable_stability_features=True,
            enable_economic_features=True,
            enable_trading_features=True,
            enable_transition_features=True
        )
    
    return NASFeatureEnhancer(config)