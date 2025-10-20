"""
Enhanced HDBSCAN Regime Discovery Integration

This module provides a unified interface for the enhanced HDBSCAN economic profiling system,
integrating all improvements and providing a single entry point for regime discovery.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from pathlib import Path

# Import enhanced components
from .enhanced_hdbscan_clusterer import EnhancedHDBSCANClusterer, EnhancedHDBSCANConfig
from .regime_feature_extractor import RegimeFeatureExtractor, RegimeFeatureConfig
from .feature_processor import FeatureProcessor, FeatureProcessorConfig
from .dimensionality_reducer import DimensionalityReducer, DimensionalityReducerConfig
from .economic_validator import EconomicValidator, EconomicValidatorConfig
from .temporal_stabilizer import TemporalStabilizer, TemporalStabilizerConfig
from .validation.statistical_validator import StatisticalValidator, ValidationConfig

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRegimeDiscoveryConfig:
    """Configuration for the enhanced regime discovery system."""
    # Feature extraction
    feature_config: Optional[RegimeFeatureConfig] = None
    
    # Feature processing
    processor_config: Optional[FeatureProcessorConfig] = None
    
    # Dimensionality reduction
    dr_config: Optional[DimensionalityReducerConfig] = None
    
    # HDBSCAN clustering
    hdbscan_config: Optional[EnhancedHDBSCANConfig] = None
    
    # Economic validation
    economic_config: Optional[EconomicValidatorConfig] = None
    
    # Temporal stabilization
    temporal_config: Optional[TemporalStabilizerConfig] = None
    
    # Statistical validation
    validation_config: Optional[ValidationConfig] = None
    
    # System settings
    enable_validation: bool = True
    enable_persistence: bool = True
    enable_optimization: bool = True
    enable_uncertainty_quantification: bool = True
    enable_ensemble_prediction: bool = True
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        if self.feature_config is None:
            self.feature_config = RegimeFeatureConfig()
        
        if self.processor_config is None:
            self.processor_config = FeatureProcessorConfig()
        
        if self.dr_config is None:
            self.dr_config = DimensionalityReducerConfig()
        
        if self.hdbscan_config is None:
            self.hdbscan_config = EnhancedHDBSCANConfig()
        
        if self.economic_config is None:
            self.economic_config = EconomicValidatorConfig()
        
        if self.temporal_config is None:
            self.temporal_config = TemporalStabilizerConfig()
        
        if self.validation_config is None:
            self.validation_config = ValidationConfig()

@dataclass
class EnhancedRegimeResult:
    """Result of enhanced regime discovery."""
    # Core results
    regime_labels: np.ndarray
    regime_probabilities: Optional[np.ndarray] = None
    uncertainty_measures: Optional[Dict[str, Any]] = None
    
    # Economic analysis
    economic_profiles: Optional[List[Dict[str, Any]]] = None
    trading_recommendations: Optional[Dict[str, Any]] = None
    
    # Validation results
    validation_results: Optional[Dict[str, Any]] = None
    overall_quality_score: float = 0.0
    
    # Processing information
    processing_time: float = 0.0
    pipeline_steps: Optional[Dict[str, Any]] = None
    model_metadata: Optional[Dict[str, Any]] = None
    
    # Success indicators
    success: bool = False
    error_message: Optional[str] = None

class EnhancedRegimeDiscovery:
    """
    Enhanced HDBSCAN Regime Discovery System
    
    This class provides a unified interface for the enhanced HDBSCAN economic profiling system,
    integrating all improvements and providing comprehensive regime discovery capabilities.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[EnhancedRegimeDiscoveryConfig] = None):
        """Initialize the enhanced regime discovery system."""
        self.config = config or EnhancedRegimeDiscoveryConfig()
        self.components = {}
        self.is_fitted = False
        self.training_data = None
        self.training_labels = None
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            tprint_info("Initializing enhanced regime discovery components")
            
            # Feature extractor
            self.components['feature_extractor'] = RegimeFeatureExtractor(
                self.config.feature_config
            )
            
            # Feature processor
            self.components['feature_processor'] = FeatureProcessor(
                self.config.processor_config
            )
            
            # Dimensionality reducer
            self.components['dimensionality_reducer'] = DimensionalityReducer(
                self.config.dr_config
            )
            
            # Enhanced HDBSCAN clusterer
            self.components['hdbscan_clusterer'] = EnhancedHDBSCANClusterer(
                self.config.hdbscan_config
            )
            
            # Economic validator
            self.components['economic_validator'] = EconomicValidator(
                self.config.economic_config
            )
            
            # Temporal stabilizer
            self.components['temporal_stabilizer'] = TemporalStabilizer(
                self.config.temporal_config
            )
            
            # Statistical validator
            if self.config.enable_validation:
                self.components['statistical_validator'] = StatisticalValidator(
                    self.config.validation_config
                )
            
            tprint_success("All components initialized successfully")
            
        except Exception as e:
            tprint_error(f"Component initialization failed: {e}")
            raise
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def fit(self, market_data: pd.DataFrame) -> EnhancedRegimeResult:
        """
        Fit the enhanced regime discovery system to market data.
        
        Args:
            market_data: Market data DataFrame with OHLCV data
            
        Returns:
            EnhancedRegimeResult with regime discovery results
        """
        try:
            tprint_info("Starting enhanced regime discovery fitting")
            start_time = time.perf_counter()
            
            result = EnhancedRegimeResult()
            pipeline_steps = {}
            
            # Step 1: Feature extraction
            with tprint_timer("Feature extraction"):
                features = self.components['feature_extractor'].extract_features(market_data)
                pipeline_steps['feature_extraction'] = {
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'success': True
                }
                tprint_success(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
            
            # Step 2: Feature processing
            with tprint_timer("Feature processing"):
                processed_result = self.components['feature_processor'].process_features(features)
                processed_features = processed_result.processed_features
                pipeline_steps['feature_processing'] = {
                    'n_features_before': features.shape[1],
                    'n_features_after': processed_features.shape[1],
                    'success': True
                }
                tprint_success(f"Processed features: {features.shape[1]} -> {processed_features.shape[1]}")
            
            # Step 3: Dimensionality reduction
            with tprint_timer("Dimensionality reduction"):
                dr_result = self.components['dimensionality_reducer'].reduce(processed_features)
                reduced_features = dr_result.reduced_features
                pipeline_steps['dimensionality_reduction'] = {
                    'n_features_before': processed_features.shape[1],
                    'n_features_after': reduced_features.shape[1],
                    'success': True
                }
                tprint_success(f"Dimensionality reduction: {processed_features.shape[1]} -> {reduced_features.shape[1]}")
            
            # Step 4: Enhanced HDBSCAN clustering
            with tprint_timer("Enhanced HDBSCAN clustering"):
                clustering_result = self.components['hdbscan_clusterer'].cluster_data(reduced_features)
                
                if clustering_result.get('success', False):
                    regime_labels = clustering_result['labels']
                    pipeline_steps['clustering'] = {
                        'n_clusters': clustering_result.get('n_clusters', 0),
                        'n_noise': clustering_result.get('n_noise', 0),
                        'silhouette_score': clustering_result.get('clustering_stats', {}).get('silhouette_score', -1),
                        'success': True
                    }
                    tprint_success(f"Clustering completed: {clustering_result.get('n_clusters', 0)} clusters found")
                else:
                    raise Exception(f"Clustering failed: {clustering_result.get('error', 'Unknown error')}")
            
            # Step 5: Temporal stabilization
            with tprint_timer("Temporal stabilization"):
                stabilization_result = self.components['temporal_stabilizer'].stabilize_regimes(
                    regime_labels, market_data
                )
                stabilized_labels = stabilization_result['stabilized_labels']
                pipeline_steps['temporal_stabilization'] = {
                    'stability_score': stabilization_result.get('stability_score', 0),
                    'n_transitions_before': stabilization_result.get('n_transitions_before', 0),
                    'n_transitions_after': stabilization_result.get('n_transitions_after', 0),
                    'success': True
                }
                tprint_success(f"Temporal stabilization completed: {stabilization_result.get('stability_score', 0):.3f} stability score")
            
            # Step 6: Economic validation and profiling
            with tprint_timer("Economic validation and profiling"):
                economic_result = self.components['economic_validator'].validate_and_profile(
                    market_data, stabilized_labels
                )
                pipeline_steps['economic_validation'] = {
                    'n_regime_profiles': len(economic_result.get('regime_profiles', [])),
                    'overall_quality_score': economic_result.get('overall_quality_score', 0),
                    'success': True
                }
                tprint_success(f"Economic validation completed: {len(economic_result.get('regime_profiles', []))} regime profiles")
            
            # Step 7: Statistical validation (if enabled)
            if self.config.enable_validation:
                with tprint_timer("Statistical validation"):
                    validation_result = self.components['statistical_validator'].validate_regime_profiling(
                        market_data, stabilized_labels, self.components['economic_validator']
                    )
                    pipeline_steps['statistical_validation'] = {
                        'overall_score': validation_result.get('overall_score', 0),
                        'success': True
                    }
                    tprint_success(f"Statistical validation completed: {validation_result.get('overall_score', 0):.3f} overall score")
            else:
                validation_result = None
                pipeline_steps['statistical_validation'] = {'success': False, 'skipped': True}
            
            # Store training data
            self.training_data = market_data.copy()
            self.training_labels = stabilized_labels.copy()
            self.is_fitted = True
            
            # Prepare result
            result.regime_labels = stabilized_labels
            result.economic_profiles = economic_result.get('regime_profiles', [])
            result.trading_recommendations = economic_result.get('trading_recommendations', {})
            result.validation_results = validation_result
            result.overall_quality_score = economic_result.get('overall_quality_score', 0)
            result.processing_time = time.perf_counter() - start_time
            result.pipeline_steps = pipeline_steps
            result.model_metadata = {
                'created_at': time.time(),
                'n_samples': len(market_data),
                'n_features': features.shape[1],
                'n_clusters': clustering_result.get('n_clusters', 0),
                'config': self.config
            }
            result.success = True
            
            tprint_success(f"Enhanced regime discovery fitting completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"Enhanced regime discovery fitting failed: {e}")
            result = EnhancedRegimeResult()
            result.error_message = str(e)
            result.success = False
            return result
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def predict(self, market_data: pd.DataFrame) -> EnhancedRegimeResult:
        """
        Predict regime labels for new market data.
        
        Args:
            market_data: New market data DataFrame
            
        Returns:
            EnhancedRegimeResult with predictions
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            tprint_info("Starting enhanced regime prediction")
            start_time = time.perf_counter()
            
            result = EnhancedRegimeResult()
            
            # Extract features
            features = self.components['feature_extractor'].extract_features(market_data)
            
            # Process features
            processed_result = self.components['feature_processor'].transform_features(features)
            processed_features = processed_result.processed_features
            
            # Reduce dimensionality
            dr_result = self.components['dimensionality_reducer'].reduce(processed_features)
            reduced_features = dr_result.reduced_features
            
            # Enhanced prediction with uncertainty
            if self.config.enable_uncertainty_quantification:
                prediction_result = self.components['hdbscan_clusterer'].enhanced_predict_with_uncertainty(reduced_features)
                
                if prediction_result.get('success', False):
                    result.regime_labels = prediction_result['labels']
                    result.regime_probabilities = prediction_result['probabilities']
                    result.uncertainty_measures = prediction_result['uncertainty_measures']
                else:
                    raise Exception(f"Enhanced prediction failed: {prediction_result.get('error', 'Unknown error')}")
            else:
                # Basic prediction
                labels, probabilities, method = self.components['hdbscan_clusterer'].approximate_predict_with_fallback(reduced_features)
                result.regime_labels = labels
                result.regime_probabilities = probabilities
            
            # Economic analysis for predictions
            economic_result = self.components['economic_validator'].validate_and_profile(
                market_data, result.regime_labels
            )
            result.economic_profiles = economic_result.get('regime_profiles', [])
            result.trading_recommendations = economic_result.get('trading_recommendations', {})
            result.overall_quality_score = economic_result.get('overall_quality_score', 0)
            
            result.processing_time = time.perf_counter() - start_time
            result.success = True
            
            tprint_success(f"Enhanced regime prediction completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"Enhanced regime prediction failed: {e}")
            result = EnhancedRegimeResult()
            result.error_message = str(e)
            result.success = False
            return result
    
    def save_model(self, filepath: str) -> bool:
        """Save the complete model to disk."""
        try:
            if not self.is_fitted:
                tprint_warning("Model not fitted, cannot save")
                return False
            
            model_data = {
                'config': self.config,
                'components': {},
                'training_data': self.training_data,
                'training_labels': self.training_labels,
                'is_fitted': self.is_fitted,
                'model_metadata': {
                    'created_at': time.time(),
                    'version': '1.0.0'
                }
            }
            
            # Save individual components
            for name, component in self.components.items():
                if hasattr(component, 'save_model'):
                    component_path = f"{filepath}_{name}.pkl"
                    if component.save_model(component_path):
                        model_data['components'][name] = component_path
                    else:
                        tprint_warning(f"Failed to save component {name}")
            
            # Save main model
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            tprint_success(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Model saving failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a complete model from disk."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load configuration
            self.config = model_data['config']
            
            # Reinitialize components
            self._initialize_components()
            
            # Load individual components
            for name, component_path in model_data['components'].items():
                if name in self.components and hasattr(self.components[name], 'load_model'):
                    if not self.components[name].load_model(component_path):
                        tprint_warning(f"Failed to load component {name}")
            
            # Load training data
            self.training_data = model_data['training_data']
            self.training_labels = model_data['training_labels']
            self.is_fitted = model_data['is_fitted']
            
            tprint_success(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Model loading failed: {e}")
            return False
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered regimes."""
        try:
            if not self.is_fitted or self.training_labels is None:
                return {'error': 'Model not fitted'}
            
            unique_regimes = np.unique(self.training_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]  # Remove noise
            
            summary = {
                'n_regimes': len(unique_regimes),
                'n_noise_points': np.sum(self.training_labels == -1),
                'noise_ratio': np.sum(self.training_labels == -1) / len(self.training_labels),
                'regime_durations': [],
                'regime_transitions': 0
            }
            
            # Calculate regime durations
            for regime in unique_regimes:
                regime_mask = self.training_labels == regime
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) > 0:
                    # Find consecutive periods
                    consecutive_periods = []
                    current_length = 1
                    for i in range(1, len(regime_indices)):
                        if regime_indices[i] == regime_indices[i-1] + 1:
                            current_length += 1
                        else:
                            consecutive_periods.append(current_length)
                            current_length = 1
                    consecutive_periods.append(current_length)
                    summary['regime_durations'].extend(consecutive_periods)
            
            # Calculate transitions
            for i in range(1, len(self.training_labels)):
                if self.training_labels[i] != self.training_labels[i-1]:
                    summary['regime_transitions'] += 1
            
            return summary
            
        except Exception as e:
            tprint_error(f"Regime summary generation failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of the regime discovery system."""
        try:
            if not self.is_fitted:
                return "Model not fitted. Run fit() first."
            
            report = []
            report.append("=" * 100)
            report.append("ENHANCED HDBSCAN REGIME DISCOVERY SYSTEM - COMPREHENSIVE REPORT")
            report.append("=" * 100)
            report.append("")
            
            # System configuration
            report.append("SYSTEM CONFIGURATION:")
            report.append(f"  Feature Extraction: {self.config.feature_config.__class__.__name__}")
            report.append(f"  Feature Processing: {self.config.processor_config.__class__.__name__}")
            report.append(f"  Dimensionality Reduction: {self.config.dr_config.__class__.__name__}")
            report.append(f"  HDBSCAN Clustering: {self.config.hdbscan_config.__class__.__name__}")
            report.append(f"  Economic Validation: {self.config.economic_config.__class__.__name__}")
            report.append(f"  Temporal Stabilization: {self.config.temporal_config.__class__.__name__}")
            report.append(f"  Statistical Validation: {'Enabled' if self.config.enable_validation else 'Disabled'}")
            report.append("")
            
            # Regime summary
            regime_summary = self.get_regime_summary()
            if 'error' not in regime_summary:
                report.append("REGIME SUMMARY:")
                report.append(f"  Number of Regimes: {regime_summary['n_regimes']}")
                report.append(f"  Noise Points: {regime_summary['n_noise_points']}")
                report.append(f"  Noise Ratio: {regime_summary['noise_ratio']:.3f}")
                report.append(f"  Regime Transitions: {regime_summary['regime_transitions']}")
                
                if regime_summary['regime_durations']:
                    report.append(f"  Min Regime Duration: {min(regime_summary['regime_durations'])}")
                    report.append(f"  Max Regime Duration: {max(regime_summary['regime_durations'])}")
                    report.append(f"  Avg Regime Duration: {np.mean(regime_summary['regime_durations']):.1f}")
                report.append("")
            
            # Model metadata
            if hasattr(self, 'model_metadata') and self.model_metadata:
                report.append("MODEL METADATA:")
                for key, value in self.model_metadata.items():
                    if key != 'config':
                        report.append(f"  {key}: {value}")
                report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            if regime_summary.get('n_regimes', 0) < 2:
                report.append("  - Consider adjusting clustering parameters to find more regimes")
            elif regime_summary.get('noise_ratio', 0) > 0.3:
                report.append("  - High noise ratio detected, consider feature engineering improvements")
            elif regime_summary.get('regime_transitions', 0) > 50:
                report.append("  - Many regime transitions detected, consider temporal stabilization")
            else:
                report.append("  - System appears to be working well")
                report.append("  - Consider real-time implementation")
                report.append("  - Add monitoring and alerting")
            
            report.append("")
            report.append("=" * 100)
            
            return "\n".join(report)
            
        except Exception as e:
            tprint_error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

# Convenience function for easy usage
def create_enhanced_regime_discovery(config: Optional[EnhancedRegimeDiscoveryConfig] = None) -> EnhancedRegimeDiscovery:
    """
    Create an enhanced regime discovery system with default or custom configuration.
    
    Args:
        config: Optional configuration object
        
    Returns:
        EnhancedRegimeDiscovery instance
    """
    return EnhancedRegimeDiscovery(config)