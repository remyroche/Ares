#!/usr/bin/env python3
"""
Complete Integration Demo for HMM Clustering with Common Utilities

This module demonstrates the complete integration of all HMM clustering
modules with all available common utilities for comprehensive market analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

# Import all HMM clustering modules
from enhanced_hmm_clustering import (
    EnhancedHMMClustering, 
    HMMClusteringConfig,
    create_enhanced_hmm_clustering
)
from hardware_optimized_hmm import (
    HardwareOptimizedHMM,
    HardwareOptimizedConfig,
    create_hardware_optimized_hmm
)
from matrix_operations_integration import (
    MatrixOperationsIntegration,
    MatrixOperationsConfig,
    create_matrix_operations_integration
)
from ml_utilities_integration import (
    MLUtilitiesIntegration,
    MLUtilitiesConfig,
    create_ml_utilities_integration
)
from comprehensive_validation import (
    ComprehensiveValidator,
    ValidationConfig,
    create_comprehensive_validator
)

# Import common utilities
from src.utils.common_operations import (
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_m1_cpu_optimizer,
    safe_dataframe_operation,
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import safe_divide, safe_log, validate_finite
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.logger import system_logger

# Setup logging
logger = system_logger.getChild('CompleteIntegrationDemo')

class CompleteHMMIntegration:
    """
    Complete HMM clustering integration with all common utilities.
    
    This class demonstrates how to use all HMM clustering modules
    together with all available common utilities for comprehensive
    market analysis workflows.
    """
    
    def __init__(self):
        """Initialize complete integration."""
        self.logger = logger.getChild('CompleteHMMIntegration')
        
        # Initialize all common utilities
        self._initialize_common_utilities()
        
        # Initialize all HMM clustering modules
        self._initialize_hmm_modules()
        
        # Performance tracking
        self.performance_metrics = {}
        self.integration_results = {}
        
        self.logger.info("🚀 Complete HMM Integration initialized with all utilities")
    
    def _initialize_common_utilities(self):
        """Initialize all common utilities."""
        self.logger.info("🔧 Initializing common utilities...")
        
        # Hardware utilities
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Data operations
        self.matrix_ops = UnifiedMatrixOperations()
        
        # ML utilities
        self.cv_validator = TimeSeriesCrossValidator()
        self.hpo_optimizer = HyperparameterOptimizer()
        self.hmm_regime_detector = HMMRegimeDetector()
        
        # Serialization
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        
        # Log utility status
        self.logger.info(f"✅ GPU Manager: {'Available' if self.gpu_manager else 'Not Available'}")
        self.logger.info(f"✅ Memory Optimizer: {'Available' if self.memory_optimizer else 'Not Available'}")
        self.logger.info(f"✅ CPU Optimizer: {'Available' if self.cpu_optimizer else 'Not Available'}")
        self.logger.info(f"✅ Matrix Operations: {'Available' if self.matrix_ops else 'Not Available'}")
    
    def _initialize_hmm_modules(self):
        """Initialize all HMM clustering modules."""
        self.logger.info("🔧 Initializing HMM clustering modules...")
        
        # Enhanced HMM clustering
        self.enhanced_hmm_config = HMMClusteringConfig(
            n_components=3,
            covariance_type='full',
            n_iter=100,
            random_state=42,
            use_gpu=True,
            enable_validation=True,
            enable_optimization=True
        )
        self.enhanced_hmm = create_enhanced_hmm_clustering(self.enhanced_hmm_config)
        
        # Hardware-optimized HMM
        self.hardware_hmm_config = HardwareOptimizedConfig(
            n_components=3,
            use_gpu_acceleration=True,
            use_memory_optimization=True,
            use_cpu_optimization=True,
            batch_size=1000,
            enable_profiling=True
        )
        self.hardware_hmm = create_hardware_optimized_hmm(self.hardware_hmm_config)
        
        # Matrix operations integration
        self.matrix_ops_config = MatrixOperationsConfig(
            scaling_method='standard',
            enable_dimensionality_reduction=True,
            reduction_method='pca',
            n_components=10,
            enable_feature_selection=True,
            selection_method='kbest',
            n_features=15,
            enable_matrix_optimization=True,
            memory_efficient=True
        )
        self.matrix_ops_integration = create_matrix_operations_integration(self.matrix_ops_config)
        
        # ML utilities integration
        self.ml_utils_config = MLUtilitiesConfig(
            enable_hpo=True,
            hpo_method='bayesian',
            n_trials=50,
            enable_ensemble=True,
            ensemble_methods=['hmm', 'kmeans'],
            enable_regime_processing=True
        )
        self.ml_utils_integration = create_ml_utilities_integration(self.ml_utils_config)
        
        # Comprehensive validation
        self.validation_config = ValidationConfig(
            enable_data_validation=True,
            enable_model_validation=True,
            enable_performance_validation=True,
            enable_error_recovery=True,
            enable_detailed_logging=True
        )
        self.validator = create_comprehensive_validator(self.validation_config)
        
        self.logger.info("✅ All HMM clustering modules initialized")
    
    def generate_sample_market_data(self, n_samples: int = 2000, n_features: int = 8) -> pd.DataFrame:
        """Generate realistic sample market data."""
        self.logger.info(f"📊 Generating sample market data: {n_samples} samples, {n_features} features")
        
        np.random.seed(42)
        
        # Generate time series data
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Generate price data with trends and volatility clusters
        returns = np.random.normal(0, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume data
        volume = np.random.lognormal(10, 0.5, n_samples)
        
        # Create OHLCV data
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': volume
        })
        
        # Add technical indicators
        market_data['returns'] = market_data['close'].pct_change()
        market_data['log_returns'] = safe_log(market_data['close'] / market_data['close'].shift(1))
        market_data['volatility'] = market_data['returns'].rolling(20).std()
        market_data['volume_ratio'] = safe_divide(
            market_data['volume'], 
            market_data['volume'].rolling(20).mean()
        )
        
        # Remove NaN values
        market_data = market_data.dropna()
        
        self.logger.info(f"✅ Sample market data generated: {market_data.shape}")
        return market_data
    
    def prepare_features_comprehensive(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Prepare features using comprehensive matrix operations."""
        self.logger.info("🔧 Preparing features with comprehensive matrix operations...")
        
        # Select numeric features
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = data[numeric_features]
        
        # Validate data quality
        quality_metrics = calculate_data_quality_metrics(feature_data)
        self.logger.info(f"📊 Data quality metrics: {quality_metrics}")
        
        # Use matrix operations integration
        transformed_data, metadata = self.matrix_ops_integration.fit_transform(feature_data.values)
        
        self.logger.info(f"✅ Features prepared: {feature_data.shape} → {transformed_data.shape}")
        
        return transformed_data, metadata
    
    def run_comprehensive_validation(self, data: np.ndarray, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Run comprehensive validation using all validation utilities."""
        self.logger.info("🔍 Running comprehensive validation...")
        
        # Validate data
        data_valid, data_results = self.validator.validate_data(data, "comprehensive analysis")
        
        # Validate parameters
        param_valid, param_results = self.validator.validate_model_parameters(params, "HMM parameters")
        
        # Validate HMM training
        hmm_valid, hmm_results = self.validator.validate_hmm_training(data, params)
        
        # Combine validation results
        validation_results = {
            'data_validation': data_results,
            'parameter_validation': param_results,
            'hmm_validation': hmm_results,
            'overall_valid': data_valid and param_valid and hmm_valid
        }
        
        self.logger.info(f"✅ Comprehensive validation completed: {validation_results['overall_valid']}")
        
        return validation_results['overall_valid'], validation_results
    
    def run_enhanced_hmm_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run enhanced HMM analysis with all utilities."""
        self.logger.info("🚀 Running enhanced HMM analysis...")
        
        start_time = time.time()
        
        try:
            # Run enhanced HMM clustering
            results = self.enhanced_hmm.fit(data)
            
            # Get performance summary
            performance_summary = self.enhanced_hmm.get_performance_summary()
            
            analysis_time = time.time() - start_time
            
            return {
                'results': results,
                'performance_summary': performance_summary,
                'analysis_time': analysis_time,
                'method': 'enhanced_hmm'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced HMM analysis failed: {e}")
            raise
    
    def run_hardware_optimized_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run hardware-optimized HMM analysis."""
        self.logger.info("🚀 Running hardware-optimized HMM analysis...")
        
        start_time = time.time()
        
        try:
            # Run hardware-optimized HMM clustering
            results = self.hardware_hmm.fit(data)
            
            # Get hardware performance summary
            hardware_summary = self.hardware_hmm.get_hardware_performance_summary()
            
            analysis_time = time.time() - start_time
            
            return {
                'results': results,
                'hardware_summary': hardware_summary,
                'analysis_time': analysis_time,
                'method': 'hardware_optimized'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hardware-optimized HMM analysis failed: {e}")
            raise
    
    def run_ml_utilities_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run ML utilities analysis with cross-validation and HPO."""
        self.logger.info("🚀 Running ML utilities analysis...")
        
        start_time = time.time()
        
        try:
            # Run comprehensive ML analysis
            results = self.ml_utils_integration.run_comprehensive_analysis(data)
            
            # Get performance summary
            performance_summary = self.ml_utils_integration.get_performance_summary()
            
            analysis_time = time.time() - start_time
            
            return {
                'results': results,
                'performance_summary': performance_summary,
                'analysis_time': analysis_time,
                'method': 'ml_utilities'
            }
            
        except Exception as e:
            self.logger.error(f"❌ ML utilities analysis failed: {e}")
            raise
    
    def run_complete_integration_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete integration analysis using all modules and utilities."""
        self.logger.info("🚀 Running complete integration analysis...")
        
        overall_start_time = time.time()
        
        try:
            # Prepare features
            features, feature_metadata = self.prepare_features_comprehensive(data)
            
            # Run comprehensive validation
            validation_params = {
                'n_components': 3,
                'covariance_type': 'full',
                'n_iter': 100,
                'random_state': 42
            }
            
            is_valid, validation_results = self.run_comprehensive_validation(features, validation_params)
            
            if not is_valid:
                self.logger.warning("⚠️ Validation failed, proceeding with warnings")
            
            # Run all analysis methods
            analysis_results = {}
            
            # Enhanced HMM analysis
            try:
                enhanced_results = self.run_enhanced_hmm_analysis(features)
                analysis_results['enhanced_hmm'] = enhanced_results
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced HMM analysis failed: {e}")
            
            # Hardware-optimized analysis
            try:
                hardware_results = self.run_hardware_optimized_analysis(features)
                analysis_results['hardware_optimized'] = hardware_results
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware-optimized analysis failed: {e}")
            
            # ML utilities analysis
            try:
                ml_results = self.run_ml_utilities_analysis(features)
                analysis_results['ml_utilities'] = ml_results
            except Exception as e:
                self.logger.warning(f"⚠️ ML utilities analysis failed: {e}")
            
            # Calculate overall metrics
            overall_analysis_time = time.time() - overall_start_time
            
            # Create comprehensive results
            comprehensive_results = {
                'feature_preparation': feature_metadata,
                'validation': validation_results,
                'analysis_results': analysis_results,
                'overall_analysis_time': overall_analysis_time,
                'common_utilities_status': {
                    'gpu_manager': self.gpu_manager is not None,
                    'memory_optimizer': self.memory_optimizer is not None,
                    'cpu_optimizer': self.cpu_optimizer is not None,
                    'matrix_ops': self.matrix_ops is not None,
                    'cv_validator': self.cv_validator is not None,
                    'hpo_optimizer': self.hpo_optimizer is not None
                },
                'timestamp': time.time()
            }
            
            self.logger.info("✅ Complete integration analysis completed!")
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"❌ Complete integration analysis failed: {e}")
            raise
    
    def save_comprehensive_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Save comprehensive results using serialization utilities."""
        self.logger.info(f"💾 Saving comprehensive results to {filepath}")
        
        try:
            # Prepare results for serialization
            serializable_results = self._prepare_results_for_serialization(results)
            
            # Save using appropriate serializer
            if filepath.endswith('.json'):
                success = self.json_serializer.save(serializable_results, filepath)
            else:
                success = self.pickle_serializer.save(serializable_results, filepath)
            
            if success:
                self.logger.info("✅ Comprehensive results saved successfully")
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save comprehensive results: {e}")
            return False
    
    def _prepare_results_for_serialization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for serialization by converting non-serializable objects."""
        serializable = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                serializable[key] = self._prepare_results_for_serialization(value)
            elif isinstance(value, (np.ndarray, np.generic)):
                serializable[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            elif hasattr(value, '__dict__'):
                # Convert objects to dict
                try:
                    serializable[key] = value.__dict__
                except:
                    serializable[key] = str(value)
            else:
                serializable[key] = value
        
        return serializable
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report."""
        self.logger.info("📊 Generating comprehensive analysis report...")
        
        try:
            report = f"""
# Complete HMM Clustering Integration Analysis Report

## Analysis Overview
- **Analysis Time**: {results['overall_analysis_time']:.2f} seconds
- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}

## Feature Preparation
- **Original Features**: {results['feature_preparation'].get('n_features_original', 'N/A')}
- **Processed Features**: {results['feature_preparation'].get('n_features_processed', 'N/A')}
- **Feature Reduction Ratio**: {results['feature_preparation'].get('feature_reduction_ratio', 'N/A'):.3f}

## Validation Results
- **Overall Valid**: {results['validation']['overall_valid']}
- **Data Validation**: {'✅ Passed' if results['validation']['data_validation']['is_valid'] else '❌ Failed'}
- **Parameter Validation**: {'✅ Passed' if results['validation']['parameter_validation']['is_valid'] else '❌ Failed'}
- **HMM Validation**: {'✅ Passed' if results['validation']['hmm_validation']['is_valid'] else '❌ Failed'}

## Analysis Results
"""
            
            # Add results for each analysis method
            for method, analysis in results['analysis_results'].items():
                report += f"\n### {method.replace('_', ' ').title()}\n"
                report += f"- **Analysis Time**: {analysis['analysis_time']:.2f} seconds\n"
                
                if 'results' in analysis:
                    if hasattr(analysis['results'], 'silhouette_score'):
                        report += f"- **Silhouette Score**: {analysis['results'].silhouette_score:.3f}\n"
                    if hasattr(analysis['results'], 'training_time'):
                        report += f"- **Training Time**: {analysis['results'].training_time:.2f} seconds\n"
                
                if 'performance_summary' in analysis:
                    perf = analysis['performance_summary']
                    if 'training_metrics' in perf:
                        report += f"- **Log Likelihood**: {perf['training_metrics'].get('log_likelihood', 'N/A')}\n"
                        report += f"- **AIC**: {perf['training_metrics'].get('aic', 'N/A')}\n"
                        report += f"- **BIC**: {perf['training_metrics'].get('bic', 'N/A')}\n"
            
            # Add common utilities status
            report += f"\n## Common Utilities Status\n"
            utils_status = results['common_utilities_status']
            for utility, status in utils_status.items():
                report += f"- **{utility.replace('_', ' ').title()}**: {'✅ Available' if status else '❌ Not Available'}\n"
            
            report += f"\n## Recommendations\n"
            if results['validation']['overall_valid']:
                report += "- ✅ All validations passed - results are reliable\n"
            else:
                report += "- ⚠️ Some validations failed - review warnings and errors\n"
            
            if len(results['analysis_results']) > 1:
                report += "- 📊 Multiple analysis methods completed - compare results\n"
            
            report += "- 💾 Results saved for future reference\n"
            report += "- 🔄 Consider running with different parameters for comparison\n"
            
            self.logger.info("✅ Comprehensive analysis report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            return f"Error generating report: {e}"


def run_complete_demo():
    """Run complete integration demo."""
    logger.info("🚀 Starting Complete HMM Clustering Integration Demo")
    
    try:
        # Create complete integration instance
        integration = CompleteHMMIntegration()
        
        # Generate sample market data
        market_data = integration.generate_sample_market_data(n_samples=2000, n_features=8)
        
        # Run complete integration analysis
        results = integration.run_complete_integration_analysis(market_data)
        
        # Generate comprehensive report
        report = integration.generate_comprehensive_report(results)
        print(report)
        
        # Save results
        integration.save_comprehensive_results(results, 'complete_integration_results.json')
        
        # Save report
        with open('complete_integration_report.md', 'w') as f:
            f.write(report)
        
        logger.info("✅ Complete integration demo finished successfully!")
        
    except Exception as e:
        logger.error(f"❌ Complete integration demo failed: {e}")
        raise


if __name__ == "__main__":
    run_complete_demo()