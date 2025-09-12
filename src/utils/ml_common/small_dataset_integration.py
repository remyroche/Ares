"""
Small Dataset Integration for SR ML Enhancer
Integrates all small dataset management techniques with existing SR ML Enhancer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

# Import our small dataset management modules
from .small_dataset_management import SmallDatasetMLManager, SmallDatasetConfig
from .transfer_learning_sr import SRTransferLearningEngine, TransferLearningConfig
from .regularized_ensemble_sr import RegularizedEnsembleSR, RegularizedEnsembleConfig

@dataclass
class SmallDatasetIntegrationConfig:
    """Configuration for small dataset integration."""
    enable_data_augmentation: bool = True
    enable_transfer_learning: bool = True
    enable_regularized_ensemble: bool = True
    min_samples_threshold: int = 50
    augmentation_factor: float = 2.0
    regularization_strength: float = 1.0
    feature_selection_ratio: float = 0.3
    cross_validation_folds: int = 3
    ensemble_methods: List[str] = None

class SmallDatasetSRIntegration:
    """Main integration class for small dataset management in SR ML."""
    
    def __init__(self, config: SmallDatasetIntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.small_dataset_manager = None
        self.transfer_learning_engine = None
        self.regularized_ensemble = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all small dataset management components."""
        try:
            # Small Dataset Manager
            if self.config.enable_data_augmentation:
                small_dataset_config = SmallDatasetConfig(
                    min_samples_for_training=self.config.min_samples_threshold,
                    augmentation_factor=self.config.augmentation_factor,
                    regularization_strength=self.config.regularization_strength
                )
                self.small_dataset_manager = SmallDatasetMLManager(small_dataset_config)
            
            # Transfer Learning Engine
            if self.config.enable_transfer_learning:
                transfer_config = TransferLearningConfig(
                    feature_similarity_threshold=0.6,
                    adaptation_rate=0.1
                )
                self.transfer_learning_engine = SRTransferLearningEngine(transfer_config)
            
            # Regularized Ensemble
            if self.config.enable_regularized_ensemble:
                ensemble_config = RegularizedEnsembleConfig(
                    n_estimators=50,
                    max_depth=3,
                    regularization_strength=self.config.regularization_strength,
                    feature_selection_ratio=self.config.feature_selection_ratio
                )
                self.regularized_ensemble = RegularizedEnsembleSR(ensemble_config)
            
            self.logger.info("✅ Small dataset management components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
    
    def enhance_sr_ml_training(self, market_data: pd.DataFrame, 
                             sr_levels: List[Dict[str, Any]], 
                             historical_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced SR ML training with small dataset management.
        
        Args:
            market_data: Market data DataFrame
            sr_levels: List of SR levels
            historical_performance: Historical performance data
            
        Returns:
            Enhanced training results with small dataset optimizations
        """
        results = {
            'original_dataset_size': len(sr_levels),
            'enhanced_dataset_size': len(sr_levels),
            'training_strategy': 'standard',
            'models_trained': {},
            'performance_metrics': {},
            'recommendations': [],
            'small_dataset_techniques_applied': []
        }
        
        try:
            # Check if we need small dataset management
            if len(sr_levels) < self.config.min_samples_threshold:
                self.logger.info(f"🔧 Small dataset detected ({len(sr_levels)} samples) - applying enhanced techniques")
                results['training_strategy'] = 'small_dataset_enhanced'
                
                # Apply small dataset techniques
                enhanced_results = self._apply_small_dataset_techniques(
                    market_data, sr_levels, historical_performance
                )
                
                # Merge results
                results.update(enhanced_results)
                
            else:
                self.logger.info(f"✅ Sufficient data ({len(sr_levels)} samples) - using standard training")
                results['training_strategy'] = 'standard'
                results['recommendations'].append("Sufficient data for standard ML training")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced training failed: {e}")
            results['error'] = str(e)
            return results
    
    def _apply_small_dataset_techniques(self, market_data: pd.DataFrame, 
                                      sr_levels: List[Dict[str, Any]], 
                                      historical_performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply all small dataset management techniques."""
        enhanced_results = {
            'enhanced_dataset_size': len(sr_levels),
            'models_trained': {},
            'performance_metrics': {},
            'recommendations': [],
            'small_dataset_techniques_applied': []
        }
        
        # 1. Data Augmentation
        if self.config.enable_data_augmentation and self.small_dataset_manager:
            try:
                self.logger.info("🔄 Applying data augmentation...")
                
                # Prepare training data (simplified version of existing SR ML Enhancer logic)
                X, y, feature_names = self._prepare_training_features(
                    market_data, sr_levels, historical_performance
                )
                
                if X is not None and len(X) > 0:
                    # Apply data augmentation
                    augmentation_results = self.small_dataset_manager.train_with_small_dataset(
                        X, y, feature_names, task_type='regression'
                    )
                    
                    enhanced_results['models_trained']['augmented_models'] = augmentation_results['models']
                    enhanced_results['performance_metrics']['augmentation'] = augmentation_results['performance']
                    enhanced_results['enhanced_dataset_size'] = augmentation_results['augmented_samples']
                    enhanced_results['small_dataset_techniques_applied'].append('data_augmentation')
                    
                    self.logger.info(f"✅ Data augmentation completed: {len(X)} -> {augmentation_results['augmented_samples']} samples")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Data augmentation failed: {e}")
                enhanced_results['recommendations'].append(f"Data augmentation failed: {e}")
        
        # 2. Transfer Learning
        if self.config.enable_transfer_learning and self.transfer_learning_engine:
            try:
                self.logger.info("🔄 Applying transfer learning...")
                
                X, y, feature_names = self._prepare_training_features(
                    market_data, sr_levels, historical_performance
                )
                
                if X is not None and len(X) > 0:
                    # Create synthetic source data for transfer learning
                    source_data = self.transfer_learning_engine.create_synthetic_source_data(X, y)
                    
                    # Apply transfer learning
                    transfer_results = self.transfer_learning_engine.transfer_from_similar_markets(
                        X, y, source_data, feature_names
                    )
                    
                    enhanced_results['models_trained']['transfer_models'] = transfer_results['transferred_models']
                    enhanced_results['performance_metrics']['transfer_learning'] = transfer_results['performance_gains']
                    enhanced_results['small_dataset_techniques_applied'].append('transfer_learning')
                    
                    self.logger.info("✅ Transfer learning completed")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Transfer learning failed: {e}")
                enhanced_results['recommendations'].append(f"Transfer learning failed: {e}")
        
        # 3. Regularized Ensemble
        if self.config.enable_regularized_ensemble and self.regularized_ensemble:
            try:
                self.logger.info("🔄 Creating regularized ensemble...")
                
                X, y, feature_names = self._prepare_training_features(
                    market_data, sr_levels, historical_performance
                )
                
                if X is not None and len(X) > 0:
                    # Create regularized ensemble
                    ensemble_results = self.regularized_ensemble.create_regularized_ensemble(
                        X, y, feature_names, task_type='regression'
                    )
                    
                    enhanced_results['models_trained']['ensemble_models'] = ensemble_results['ensemble_models']
                    enhanced_results['performance_metrics']['ensemble'] = ensemble_results['performance_scores']
                    enhanced_results['small_dataset_techniques_applied'].append('regularized_ensemble')
                    
                    self.logger.info("✅ Regularized ensemble created")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Regularized ensemble failed: {e}")
                enhanced_results['recommendations'].append(f"Regularized ensemble failed: {e}")
        
        # 4. Generate final recommendations
        enhanced_results['recommendations'].extend(self._generate_final_recommendations(enhanced_results))
        
        return enhanced_results
    
    def _prepare_training_features(self, market_data: pd.DataFrame, 
                                 sr_levels: List[Dict[str, Any]], 
                                 historical_performance: Optional[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare training features (simplified version of existing SR ML Enhancer logic)."""
        try:
            features = []
            targets = []
            
            # Extract features for each SR level (simplified version)
            for level in sr_levels:
                # Basic SR features (simplified)
                level_features = [
                    level.get('touch_count', 0),
                    level.get('strength', 0.5),
                    level.get('age_bars', 0),
                    level.get('avg_bounce_ratio', 0),
                    level.get('max_bounce_ratio', 0),
                    level.get('volume_confirmation_score', 0.5),
                    level.get('consistency_score', 0.5),
                    level.get('failure_count', 0),
                    level.get('confluence_score', 0.5),
                    level.get('hvn_strength', 0.5),
                    level.get('fib_confluence_count', 0),
                    level.get('psychological_level_type', 0.0),
                    level.get('pivot_strength', 0.5),
                    level.get('trendline_strength', 0.5),
                    level.get('sr_retest_success_rate', 0.5)
                ]
                
                # Add some market context features
                if len(market_data) > 0:
                    current_price = market_data['close'].iloc[-1]
                    level_price = level.get('price', 0)
                    
                    if level_price > 0:
                        proximity = abs(current_price - level_price) / level_price
                        level_features.append(proximity)
                    else:
                        level_features.append(1.0)
                    
                    # Add some basic technical indicators
                    if len(market_data) >= 20:
                        sma_20 = market_data['close'].rolling(window=20).mean().iloc[-1]
                        price_vs_sma = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
                        level_features.append(price_vs_sma)
                    else:
                        level_features.append(0.0)
                
                features.append(level_features)
                
                # Create target (simplified quality score)
                if historical_performance and level.get('id') in historical_performance:
                    target = historical_performance[level['id']].get('quality_score', 0.5)
                else:
                    # Simple target based on level properties
                    target = (
                        level.get('strength', 0.5) * 0.3 +
                        level.get('volume_confirmation_score', 0.5) * 0.2 +
                        level.get('consistency_score', 0.5) * 0.2 +
                        min(level.get('touch_count', 0) / 10.0, 1.0) * 0.3
                    )
                
                targets.append(target)
            
            if not features:
                return None, None, []
            
            feature_names = [
                'touch_count', 'strength', 'age_bars', 'avg_bounce_ratio', 'max_bounce_ratio',
                'volume_confirmation_score', 'consistency_score', 'failure_count', 'confluence_score',
                'hvn_strength', 'fib_confluence_count', 'psychological_level_type', 'pivot_strength',
                'trendline_strength', 'sr_retest_success_rate', 'proximity_to_level', 'price_vs_sma'
            ]
            
            return np.array(features), np.array(targets), feature_names
            
        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            return None, None, []
    
    def _generate_final_recommendations(self, enhanced_results: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on all applied techniques."""
        recommendations = []
        
        original_size = enhanced_results.get('original_dataset_size', 0)
        enhanced_size = enhanced_results.get('enhanced_dataset_size', original_size)
        techniques_applied = enhanced_results.get('small_dataset_techniques_applied', [])
        
        if original_size < 50:
            recommendations.append("⚠️ Very small dataset - monitor model performance closely")
            recommendations.append("📊 Consider collecting more SR level data over time")
        
        if enhanced_size > original_size:
            recommendations.append(f"✅ Dataset enhanced from {original_size} to {enhanced_size} samples")
        
        if techniques_applied:
            recommendations.append(f"🔧 Applied techniques: {', '.join(techniques_applied)}")
        
        # Performance-based recommendations
        performance_metrics = enhanced_results.get('performance_metrics', {})
        for technique, metrics in performance_metrics.items():
            if isinstance(metrics, dict) and 'improvement' in metrics:
                if metrics['improvement'] > 0:
                    recommendations.append(f"✅ {technique} improved performance by {metrics['improvement']:.3f}")
                else:
                    recommendations.append(f"⚠️ {technique} didn't improve performance - consider different approach")
        
        recommendations.append("🎯 Use ensemble predictions for better reliability")
        recommendations.append("📈 Implement online learning to update models with new SR levels")
        
        return recommendations
    
    def predict_with_enhanced_models(self, market_data: pd.DataFrame, 
                                   sr_levels: List[Dict[str, Any]], 
                                   model_type: str = 'ensemble') -> List[Dict[str, Any]]:
        """Make predictions using enhanced models."""
        predictions = []
        
        try:
            # Prepare features for prediction
            X, _, feature_names = self._prepare_training_features(market_data, sr_levels, None)
            
            if X is None or len(X) == 0:
                self.logger.warning("⚠️ No features available for prediction")
                return predictions
            
            # Make predictions based on model type
            if model_type == 'ensemble' and self.regularized_ensemble:
                # Use ensemble model
                ensemble_pred = self.regularized_ensemble.predict_ensemble(X, 'voting_regressor')
                
                for i, level in enumerate(sr_levels):
                    if i < len(ensemble_pred):
                        prediction = {
                            'level_id': level.get('id', f'level_{i}'),
                            'quality_score': float(ensemble_pred[i]),
                            'confidence': 0.8,  # Higher confidence for ensemble
                            'prediction_method': 'enhanced_ensemble',
                            'model_type': model_type
                        }
                        predictions.append(prediction)
            
            else:
                # Fallback to basic prediction
                for i, level in enumerate(sr_levels):
                    prediction = {
                        'level_id': level.get('id', f'level_{i}'),
                        'quality_score': level.get('strength', 0.5),
                        'confidence': 0.5,
                        'prediction_method': 'fallback',
                        'model_type': 'basic'
                    }
                    predictions.append(prediction)
            
            self.logger.info(f"✅ Generated {len(predictions)} predictions using {model_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
        
        return predictions

# Integration function for existing SR ML Enhancer
def integrate_small_dataset_management_with_sr_enhancer():
    """Main integration function for existing SR ML Enhancer."""
    
    config = SmallDatasetIntegrationConfig(
        enable_data_augmentation=True,
        enable_transfer_learning=True,
        enable_regularized_ensemble=True,
        min_samples_threshold=50,
        augmentation_factor=2.0,
        regularization_strength=1.0
    )
    
    integration_manager = SmallDatasetSRIntegration(config)
    
    return integration_manager

if __name__ == "__main__":
    # Test the integration
    config = SmallDatasetIntegrationConfig()
    integration = SmallDatasetSRIntegration(config)
    
    # Create dummy data
    np.random.seed(42)
    market_data = pd.DataFrame({
        'close': np.random.uniform(100, 200, 1000),
        'high': np.random.uniform(100, 200, 1000),
        'low': np.random.uniform(100, 200, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    sr_levels = []
    for i in range(91):  # 91 SR levels
        level = {
            'id': f'level_{i}',
            'price': np.random.uniform(100, 200),
            'touch_count': np.random.randint(1, 20),
            'strength': np.random.uniform(0.3, 0.9),
            'age_bars': np.random.randint(1, 100),
            'volume_confirmation_score': np.random.uniform(0.3, 0.9),
            'consistency_score': np.random.uniform(0.3, 0.9),
            'confluence_score': np.random.uniform(0.3, 0.9),
            'hvn_strength': np.random.uniform(0.3, 0.9),
            'fib_confluence_count': np.random.randint(0, 5),
            'psychological_level_type': np.random.uniform(0, 1),
            'pivot_strength': np.random.uniform(0.3, 0.9),
            'trendline_strength': np.random.uniform(0.3, 0.9),
            'sr_retest_success_rate': np.random.uniform(0.3, 0.9)
        }
        sr_levels.append(level)
    
    # Test enhanced training
    results = integration.enhance_sr_ml_training(market_data, sr_levels)
    
    print("Small Dataset Integration Results:")
    print(f"Original dataset size: {results['original_dataset_size']}")
    print(f"Enhanced dataset size: {results['enhanced_dataset_size']}")
    print(f"Training strategy: {results['training_strategy']}")
    print(f"Techniques applied: {results['small_dataset_techniques_applied']}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")
    
    # Test predictions
    predictions = integration.predict_with_enhanced_models(market_data, sr_levels[:5])
    print(f"\nGenerated {len(predictions)} predictions")