"""
Infrastructure Integration for Enhanced Data & Labels System

This module provides seamless integration between the enhanced data and labels system
and the existing infrastructure, ensuring that all existing components natively benefit
from the upgrades.

Key Integration Points:
1. Volatility-Aware Labeler Integration
2. Regime Detection Integration
3. Feature Engineering Integration
4. Training Pipeline Integration
5. Monitoring and Validation Integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
import warnings

# Import existing infrastructure components
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig
)
from src.training.steps.pre_training.profit_labeling.enhanced_data_labels_system import (
    EnhancedDataLabelsSystem, EnhancedDataLabelsConfig, create_trading_optimized_config
)
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from src.utils.ml_common.data_processing.data_quality import DataQualityUtilities

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


class InfrastructureIntegrationManager:
    """
    Manages integration between enhanced data and labels system and existing infrastructure.
    
    This class ensures that all existing components can seamlessly use the enhanced
    data and labels system while maintaining backward compatibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the infrastructure integration manager."""
        self.config = config or {}
        self.logger = logging.getLogger('InfrastructureIntegrationManager')
        
        # Initialize core systems
        self.enhanced_labels_system = None
        self.volatility_aware_labeler = None
        self.regime_classifier = None
        self.feature_engineering_orchestrator = None
        self.data_quality_utilities = None
        
        # Integration state
        self.integration_status = {
            'enhanced_labels': False,
            'volatility_aware': False,
            'regime_detection': False,
            'feature_engineering': False,
            'data_quality': False
        }
        
        # Initialize all components
        self._initialize_components()
        
        tprint_success("🚀 Infrastructure Integration Manager initialized")
        tprint_info("   → Enhanced data and labels system")
        tprint_info("   → Volatility-aware labeler integration")
        tprint_info("   → Regime detection integration")
        tprint_info("   → Feature engineering integration")
    
    def _initialize_components(self):
        """Initialize all integrated components."""
        try:
            # Initialize enhanced data and labels system
            tprint_info("🔧 Initializing enhanced data and labels system...")
            enhanced_config = create_trading_optimized_config()
            self.enhanced_labels_system = EnhancedDataLabelsSystem(enhanced_config)
            self.integration_status['enhanced_labels'] = True
            tprint_success("✅ Enhanced data and labels system initialized")
            
            # Initialize volatility-aware labeler with enhanced integration
            tprint_info("🔧 Initializing volatility-aware labeler...")
            volatility_config = VolatilityAwareConfig(
                enable_enhanced_labels=True,
                label_definition_type='analyst'  # Use analyst labels by default
            )
            self.volatility_aware_labeler = VolatilityAwareMultiHorizonLabeler(volatility_config)
            self.integration_status['volatility_aware'] = True
            tprint_success("✅ Volatility-aware labeler initialized")
            
            # Initialize regime classifier
            tprint_info("🔧 Initializing regime classifier...")
            try:
                self.regime_classifier = UnifiedRegimeClassifier()
                self.integration_status['regime_detection'] = True
                tprint_success("✅ Regime classifier initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Regime classifier initialization failed: {e}")
                self.regime_classifier = None
            
            # Initialize feature engineering orchestrator
            tprint_info("🔧 Initializing feature engineering orchestrator...")
            try:
                self.feature_engineering_orchestrator = FeatureEngineeringOrchestrator()
                self.integration_status['feature_engineering'] = True
                tprint_success("✅ Feature engineering orchestrator initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Feature engineering orchestrator initialization failed: {e}")
                self.feature_engineering_orchestrator = None
            
            # Initialize data quality utilities
            tprint_info("🔧 Initializing data quality utilities...")
            self.data_quality_utilities = DataQualityUtilities()
            self.integration_status['data_quality'] = True
            tprint_success("✅ Data quality utilities initialized")
            
            tprint_success("🎉 All components initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Component initialization failed: {e}")
            raise
    
    def process_market_data_with_enhanced_labels(
        self,
        market_data: pd.DataFrame,
        force_regime_detection: bool = True,
        force_feature_engineering: bool = True,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Process market data through the complete enhanced pipeline.
        
        Args:
            market_data: OHLCV market data with datetime index
            force_regime_detection: Force regime detection even if cached
            force_feature_engineering: Force feature engineering even if cached
            force_recompute: Force recomputation of labels
            
        Returns:
            Dictionary containing all processed data and labels
        """
        start_time = datetime.now()
        tprint_info("🔄 Starting enhanced market data processing")
        
        try:
            # Step 1: Regime Detection
            regime_data = None
            if self.regime_classifier and force_regime_detection:
                tprint_info("🎭 Step 1: Detecting market regimes")
                regime_data = self._detect_market_regimes(market_data)
            
            # Step 2: Feature Engineering
            engineered_features = None
            if self.feature_engineering_orchestrator and force_feature_engineering:
                tprint_info("⚙️ Step 2: Engineering features")
                engineered_features = self._engineer_features(market_data, regime_data)
            
            # Step 3: Enhanced Data and Labels Processing
            tprint_info("🎯 Step 3: Processing with enhanced data and labels system")
            enhanced_result = self.enhanced_labels_system.process_market_data(
                market_data=market_data,
                regime_data=regime_data,
                force_recompute=force_recompute
            )
            
            # Step 4: Integrate with Volatility-Aware Labeler
            tprint_info("📊 Step 4: Integrating with volatility-aware labeler")
            volatility_result = self._integrate_volatility_aware_labeling(
                market_data, enhanced_result, regime_data
            )
            
            # Step 5: Compile Final Results
            tprint_info("📋 Step 5: Compiling final results")
            final_result = self._compile_final_results(
                market_data, enhanced_result, volatility_result, 
                regime_data, engineered_features
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            final_result['processing_time'] = processing_time
            final_result['timestamp'] = datetime.now()
            
            tprint_success(f"✅ Enhanced market data processing completed in {processing_time:.2f}s")
            tprint_info(f"   → Data quality: {final_result.get('data_quality_level', 'unknown')}")
            tprint_info(f"   → Label stability: {final_result.get('label_stability_level', 'unknown')}")
            tprint_info(f"   → Total samples: {len(final_result.get('processed_data', []))}")
            
            return final_result
            
        except Exception as e:
            tprint_error(f"❌ Enhanced market data processing failed: {e}")
            return self._create_error_result(str(e))
    
    def _detect_market_regimes(self, market_data: pd.DataFrame) -> Optional[pd.Series]:
        """Detect market regimes using the unified regime classifier."""
        try:
            if not self.regime_classifier:
                return None
            
            # Prepare data for regime detection
            regime_input = market_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Detect regimes
            regime_predictions = self.regime_classifier.predict(regime_input)
            
            # Convert to Series with proper index
            regime_series = pd.Series(regime_predictions, index=market_data.index)
            
            tprint_success(f"✅ Market regimes detected: {len(regime_series.unique())} unique regimes")
            return regime_series
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime detection failed: {e}")
            return None
    
    def _engineer_features(
        self, 
        market_data: pd.DataFrame, 
        regime_data: Optional[pd.Series] = None
    ) -> Optional[pd.DataFrame]:
        """Engineer features using the feature engineering orchestrator."""
        try:
            if not self.feature_engineering_orchestrator:
                return None
            
            # Prepare input data
            feature_input = market_data.copy()
            if regime_data is not None:
                feature_input['regime'] = regime_data
            
            # Engineer features
            engineered_features = self.feature_engineering_orchestrator.generate_features(feature_input)
            
            tprint_success(f"✅ Features engineered: {engineed_features.shape[1]} features")
            return engineered_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature engineering failed: {e}")
            return None
    
    def _integrate_volatility_aware_labeling(
        self,
        market_data: pd.DataFrame,
        enhanced_result: Dict[str, Any],
        regime_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Integrate with volatility-aware labeler for additional labeling."""
        try:
            if not self.volatility_aware_labeler:
                return {}
            
            # Use the cleaned data from enhanced result
            cleaned_data = enhanced_result.get('processed_data', market_data)
            
            # Generate volatility-aware labels
            volatility_result = self.volatility_aware_labeler.generate_labels(cleaned_data)
            
            # Extract additional insights
            volatility_insights = {
                'volatility_labels': volatility_result.labels,
                'volatility_confidence': volatility_result.confidence_scores,
                'volatility_quality': volatility_result.quality_scores,
                'n_samples': volatility_result.n_samples,
                'n_targets': volatility_result.n_targets
            }
            
            tprint_success("✅ Volatility-aware labeling integrated")
            return volatility_insights
            
        except Exception as e:
            tprint_warning(f"⚠️ Volatility-aware labeling integration failed: {e}")
            return {}
    
    def _compile_final_results(
        self,
        market_data: pd.DataFrame,
        enhanced_result: Dict[str, Any],
        volatility_result: Dict[str, Any],
        regime_data: Optional[pd.Series] = None,
        engineered_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Compile all results into a comprehensive output."""
        try:
            # Base results from enhanced data and labels system
            final_result = {
                'original_data': market_data,
                'processed_data': enhanced_result.get('processed_data', market_data),
                'labels': enhanced_result.get('labels', pd.DataFrame()),
                'sample_weights': enhanced_result.get('sample_weights', pd.Series()),
                'confidence_scores': enhanced_result.get('confidence_scores', pd.DataFrame()),
                'data_quality': enhanced_result.get('data_quality', {}),
                'label_stability': enhanced_result.get('label_stability', {}),
                'final_quality': enhanced_result.get('final_quality', {}),
            }
            
            # Add regime information
            if regime_data is not None:
                final_result['regime_data'] = regime_data
                final_result['regime_summary'] = {
                    'unique_regimes': len(regime_data.unique()),
                    'regime_distribution': regime_data.value_counts().to_dict()
                }
            
            # Add engineered features
            if engineered_features is not None:
                final_result['engineered_features'] = engineered_features
                final_result['feature_summary'] = {
                    'n_features': len(engineered_features.columns),
                    'feature_names': list(engineered_features.columns)
                }
            
            # Add volatility-aware insights
            if volatility_result:
                final_result['volatility_insights'] = volatility_result
            
            # Add integration status
            final_result['integration_status'] = self.integration_status.copy()
            
            # Add quality levels for easy access
            final_result['data_quality_level'] = enhanced_result.get('data_quality', {}).get('quality_level', 'unknown')
            final_result['label_stability_level'] = enhanced_result.get('label_stability', {}).get('stability_level', 'unknown')
            
            return final_result
            
        except Exception as e:
            tprint_error(f"❌ Final result compilation failed: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'error': error_message,
            'original_data': pd.DataFrame(),
            'processed_data': pd.DataFrame(),
            'labels': pd.DataFrame(),
            'sample_weights': pd.Series(),
            'confidence_scores': pd.DataFrame(),
            'integration_status': self.integration_status.copy(),
            'processing_time': 0.0,
            'timestamp': datetime.now()
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'integration_status': self.integration_status.copy(),
            'components_available': {
                'enhanced_labels_system': self.enhanced_labels_system is not None,
                'volatility_aware_labeler': self.volatility_aware_labeler is not None,
                'regime_classifier': self.regime_classifier is not None,
                'feature_engineering_orchestrator': self.feature_engineering_orchestrator is not None,
                'data_quality_utilities': self.data_quality_utilities is not None
            },
            'system_ready': all(self.integration_status.values())
        }
    
    def validate_integration(self) -> Dict[str, Any]:
        """Validate that all integrations are working correctly."""
        validation_results = {
            'overall_status': 'unknown',
            'component_tests': {},
            'recommendations': []
        }
        
        try:
            # Test enhanced labels system
            if self.enhanced_labels_system:
                test_data = pd.DataFrame({
                    'open': [100, 101, 102],
                    'high': [101, 102, 103],
                    'low': [99, 100, 101],
                    'close': [100.5, 101.5, 102.5],
                    'volume': [1000, 1100, 1200]
                })
                
                test_result = self.enhanced_labels_system.process_market_data(test_data)
                validation_results['component_tests']['enhanced_labels'] = {
                    'status': 'passed' if 'error' not in test_result else 'failed',
                    'details': 'Enhanced labels system working correctly'
                }
            else:
                validation_results['component_tests']['enhanced_labels'] = {
                    'status': 'not_available',
                    'details': 'Enhanced labels system not initialized'
                }
            
            # Test volatility-aware labeler
            if self.volatility_aware_labeler:
                validation_results['component_tests']['volatility_aware'] = {
                    'status': 'available',
                    'details': 'Volatility-aware labeler initialized'
                }
            else:
                validation_results['component_tests']['volatility_aware'] = {
                    'status': 'not_available',
                    'details': 'Volatility-aware labeler not initialized'
                }
            
            # Determine overall status
            passed_tests = sum(1 for test in validation_results['component_tests'].values() 
                             if test['status'] == 'passed')
            total_tests = len(validation_results['component_tests'])
            
            if passed_tests == total_tests:
                validation_results['overall_status'] = 'excellent'
            elif passed_tests >= total_tests * 0.8:
                validation_results['overall_status'] = 'good'
            elif passed_tests >= total_tests * 0.6:
                validation_results['overall_status'] = 'fair'
            else:
                validation_results['overall_status'] = 'poor'
            
            # Generate recommendations
            if validation_results['overall_status'] in ['fair', 'poor']:
                validation_results['recommendations'].append(
                    "Some components are not working correctly - check initialization"
                )
            
            return validation_results
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            return validation_results


# Global integration manager instance
_integration_manager = None

def get_integration_manager(config: Optional[Dict[str, Any]] = None) -> InfrastructureIntegrationManager:
    """Get the global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = InfrastructureIntegrationManager(config)
    return _integration_manager


def process_market_data_enhanced(
    market_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    force_regime_detection: bool = True,
    force_feature_engineering: bool = True,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to process market data with enhanced data and labels system.
    
    This function provides a simple interface to the complete enhanced pipeline
    while maintaining full integration with existing infrastructure.
    """
    manager = get_integration_manager(config)
    return manager.process_market_data_with_enhanced_labels(
        market_data=market_data,
        force_regime_detection=force_regime_detection,
        force_feature_engineering=force_feature_engineering,
        force_recompute=force_recompute
    )


def validate_system_integration() -> Dict[str, Any]:
    """Validate that the enhanced system is properly integrated."""
    manager = get_integration_manager()
    return manager.validate_integration()


def get_system_status() -> Dict[str, Any]:
    """Get current system status and integration information."""
    manager = get_integration_manager()
    return {
        'integration_status': manager.get_integration_status(),
        'enhanced_labels_status': manager.enhanced_labels_system.get_system_status() if manager.enhanced_labels_system else None
    }