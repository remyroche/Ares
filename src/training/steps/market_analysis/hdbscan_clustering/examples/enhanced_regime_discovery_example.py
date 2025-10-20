"""
Enhanced HDBSCAN Regime Discovery Example

This example demonstrates the enhanced HDBSCAN economic profiling system with:
- Advanced probability calculation
- Model persistence
- Improved out-of-sample prediction
- Statistical validation
- Comprehensive regime analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import enhanced components
from ..enhanced_hdbscan_clusterer import EnhancedHDBSCANClusterer, EnhancedHDBSCANConfig
from ..regime_feature_extractor import RegimeFeatureExtractor, RegimeFeatureConfig
from ..feature_processor import FeatureProcessor, FeatureProcessorConfig
from ..dimensionality_reducer import DimensionalityReducer, DimensionalityReducerConfig
from ..economic_validator import EconomicValidator, EconomicValidatorConfig
from ..temporal_stabilizer import TemporalStabilizer, TemporalStabilizerConfig
from ..validation.statistical_validator import StatisticalValidator, ValidationConfig

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

class EnhancedRegimeDiscoveryExample:
    """
    Example demonstrating the enhanced HDBSCAN regime discovery system.
    """
    
    def __init__(self):
        """Initialize the example."""
        self.market_data = None
        self.features = None
        self.processed_features = None
        self.regime_labels = None
        self.regime_probabilities = None
        self.economic_profiles = None
        self.validation_results = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            tprint_info("Initializing enhanced regime discovery components")
            
            # Enhanced HDBSCAN clusterer
            hdbscan_config = EnhancedHDBSCANConfig(
                min_cluster_size=15,
                min_samples=5,
                probability_methods=['density_based', 'distance_based', 'knn_based', 'gmm_based', 'ensemble'],
                ensemble_weights={
                    'density_based': 0.3,
                    'distance_based': 0.2,
                    'knn_based': 0.2,
                    'gmm_based': 0.3
                },
                enable_persistence=True,
                auto_save=True
            )
            self.hdbscan_clusterer = EnhancedHDBSCANClusterer(hdbscan_config)
            
            # Feature extractor
            feature_config = RegimeFeatureConfig(
                enable_returns=True,
                enable_volatility=True,
                enable_volume=True,
                enable_entropy=True,
                enable_spectral=True,
                enable_regime_specific=True,
                window_sizes=[5, 10, 20, 50],
                enable_interactions=True,
                enable_polynomial=True,
                enable_ratios=True
            )
            self.feature_extractor = RegimeFeatureExtractor(feature_config)
            
            # Feature processor
            processor_config = FeatureProcessorConfig(
                enable_cleaning=True,
                enable_outlier_handling=True,
                enable_scaling=True,
                scaling_method='robust',
                enable_feature_selection=True,
                selection_method='mutual_info',
                enable_dimensionality_reduction=True,
                reduction_method='pca',
                n_components=0.95
            )
            self.feature_processor = FeatureProcessor(processor_config)
            
            # Dimensionality reducer
            dr_config = DimensionalityReducerConfig(
                method='pca',
                n_components=0.95,
                enable_correlation_removal=True,
                correlation_threshold=0.95
            )
            self.dimensionality_reducer = DimensionalityReducer(dr_config)
            
            # Economic validator
            econ_config = EconomicValidatorConfig(
                min_regime_duration=10,
                confidence_level=0.95,
                enable_trading_recommendations=True,
                enable_risk_assessment=True
            )
            self.economic_validator = EconomicValidator(econ_config)
            
            # Temporal stabilizer
            temporal_config = TemporalStabilizerConfig(
                enable_median_filter=True,
                enable_majority_vote=True,
                enable_temporal_smoothing=True,
                min_dwell_bars=5,
                cooldown_bars=3,
                max_transitions_per_period=10
            )
            self.temporal_stabilizer = TemporalStabilizer(temporal_config)
            
            # Statistical validator
            validation_config = ValidationConfig(
                n_splits=5,
                min_confidence_level=0.95,
                max_p_value=0.05,
                min_regime_duration=10,
                min_regime_stability=0.8
            )
            self.statistical_validator = StatisticalValidator(validation_config)
            
            tprint_success("All components initialized successfully")
            
        except Exception as e:
            tprint_error(f"Component initialization failed: {e}")
            raise
    
    def generate_sample_market_data(self, n_periods: int = 1000) -> pd.DataFrame:
        """Generate sample market data for demonstration."""
        try:
            tprint_info(f"Generating {n_periods} periods of sample market data")
            
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Generate time index
            start_date = datetime.now() - timedelta(days=n_periods)
            dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
            
            # Generate price data with regime changes
            base_price = 100.0
            prices = [base_price]
            
            # Define regime periods
            regime_periods = [
                (0, 200, 0.001, 0.02),    # Bull market: high return, low vol
                (200, 400, -0.0005, 0.03), # Bear market: negative return, high vol
                (400, 600, 0.0002, 0.015), # Sideways: low return, low vol
                (600, 800, 0.0008, 0.025), # Recovery: moderate return, moderate vol
                (800, 1000, 0.0001, 0.02)  # Consolidation: low return, low vol
            ]
            
            for i in range(1, n_periods):
                # Find current regime
                current_regime = None
                for start, end, mean_return, volatility in regime_periods:
                    if start <= i < end:
                        current_regime = (mean_return, volatility)
                        break
                
                if current_regime is None:
                    current_regime = (0.0001, 0.02)  # Default regime
                
                # Generate return
                daily_return = np.random.normal(current_regime[0], current_regime[1])
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
            
            # Generate volume data (correlated with volatility)
            volumes = []
            for i in range(n_periods):
                # Find current regime for volume calculation
                current_regime = None
                for start, end, mean_return, volatility in regime_periods:
                    if start <= i < end:
                        current_regime = (mean_return, volatility)
                        break
                
                if current_regime is None:
                    current_regime = (0.0001, 0.02)
                
                # Volume is inversely related to volatility (simplified)
                base_volume = 1000000
                volume_multiplier = 1.0 + (0.03 - current_regime[1]) * 10
                volume = base_volume * volume_multiplier * np.random.uniform(0.8, 1.2)
                volumes.append(max(volume, 100000))  # Minimum volume
            
            # Create DataFrame
            market_data = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
                'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            # Ensure high >= low
            market_data['high'] = np.maximum(market_data['high'], market_data['close'])
            market_data['low'] = np.minimum(market_data['low'], market_data['close'])
            
            tprint_success(f"Generated market data: {len(market_data)} periods")
            return market_data
            
        except Exception as e:
            tprint_error(f"Market data generation failed: {e}")
            raise
    
    def run_enhanced_regime_discovery(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run the enhanced regime discovery pipeline."""
        try:
            tprint_info("Starting enhanced regime discovery pipeline")
            start_time = time.perf_counter()
            
            results = {
                'pipeline_steps': {},
                'regime_labels': None,
                'regime_probabilities': None,
                'economic_profiles': None,
                'validation_results': None,
                'processing_time': 0.0,
                'success': False
            }
            
            # Step 1: Feature extraction
            with tprint_timer("Feature extraction"):
                features = self.feature_extractor.extract_features(market_data)
                results['pipeline_steps']['feature_extraction'] = {
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'success': True
                }
                tprint_success(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
            
            # Step 2: Feature processing
            with tprint_timer("Feature processing"):
                processed_result = self.feature_processor.process_features(features)
                processed_features = processed_result.processed_features
                results['pipeline_steps']['feature_processing'] = {
                    'n_features_before': features.shape[1],
                    'n_features_after': processed_features.shape[1],
                    'success': True
                }
                tprint_success(f"Processed features: {features.shape[1]} -> {processed_features.shape[1]}")
            
            # Step 3: Dimensionality reduction
            with tprint_timer("Dimensionality reduction"):
                dr_result = self.dimensionality_reducer.reduce(processed_features)
                reduced_features = dr_result.reduced_features
                results['pipeline_steps']['dimensionality_reduction'] = {
                    'n_features_before': processed_features.shape[1],
                    'n_features_after': reduced_features.shape[1],
                    'success': True
                }
                tprint_success(f"Dimensionality reduction: {processed_features.shape[1]} -> {reduced_features.shape[1]}")
            
            # Step 4: Enhanced HDBSCAN clustering
            with tprint_timer("Enhanced HDBSCAN clustering"):
                clustering_result = self.hdbscan_clusterer.cluster_data(reduced_features)
                
                if clustering_result.get('success', False):
                    regime_labels = clustering_result['labels']
                    results['pipeline_steps']['clustering'] = {
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
                stabilization_result = self.temporal_stabilizer.stabilize_regimes(
                    regime_labels, market_data
                )
                stabilized_labels = stabilization_result['stabilized_labels']
                results['pipeline_steps']['temporal_stabilization'] = {
                    'stability_score': stabilization_result.get('stability_score', 0),
                    'n_transitions_before': stabilization_result.get('n_transitions_before', 0),
                    'n_transitions_after': stabilization_result.get('n_transitions_after', 0),
                    'success': True
                }
                tprint_success(f"Temporal stabilization completed: {stabilization_result.get('stability_score', 0):.3f} stability score")
            
            # Step 6: Economic validation and profiling
            with tprint_timer("Economic validation and profiling"):
                economic_result = self.economic_validator.validate_and_profile(
                    market_data, stabilized_labels
                )
                results['pipeline_steps']['economic_validation'] = {
                    'n_regime_profiles': len(economic_result.get('regime_profiles', [])),
                    'overall_quality_score': economic_result.get('overall_quality_score', 0),
                    'success': True
                }
                tprint_success(f"Economic validation completed: {len(economic_result.get('regime_profiles', []))} regime profiles")
            
            # Step 7: Enhanced out-of-sample prediction
            with tprint_timer("Enhanced out-of-sample prediction"):
                # Use last 20% of data for out-of-sample testing
                test_size = int(len(reduced_features) * 0.2)
                train_features = reduced_features[:-test_size]
                test_features = reduced_features[-test_size:]
                
                # Retrain on training data
                train_result = self.hdbscan_clusterer.cluster_data(train_features)
                if train_result.get('success', False):
                    # Predict on test data
                    prediction_result = self.hdbscan_clusterer.enhanced_predict_with_uncertainty(test_features)
                    
                    if prediction_result.get('success', False):
                        results['pipeline_steps']['out_of_sample_prediction'] = {
                            'n_test_samples': len(test_features),
                            'uncertainty_measures': prediction_result.get('uncertainty_measures', {}),
                            'success': True
                        }
                        tprint_success(f"Out-of-sample prediction completed: {len(test_features)} test samples")
                    else:
                        tprint_warning("Out-of-sample prediction failed")
                        results['pipeline_steps']['out_of_sample_prediction'] = {
                            'success': False,
                            'error': prediction_result.get('error', 'Unknown error')
                        }
                else:
                    tprint_warning("Training for out-of-sample prediction failed")
                    results['pipeline_steps']['out_of_sample_prediction'] = {
                        'success': False,
                        'error': 'Training failed'
                    }
            
            # Step 8: Statistical validation
            with tprint_timer("Statistical validation"):
                validation_result = self.statistical_validator.validate_regime_profiling(
                    market_data, stabilized_labels, self.economic_validator
                )
                results['pipeline_steps']['statistical_validation'] = {
                    'overall_score': validation_result.get('overall_score', 0),
                    'success': True
                }
                tprint_success(f"Statistical validation completed: {validation_result.get('overall_score', 0):.3f} overall score")
            
            # Store results
            results['regime_labels'] = stabilized_labels
            results['economic_profiles'] = economic_result
            results['validation_results'] = validation_result
            results['processing_time'] = time.perf_counter() - start_time
            results['success'] = True
            
            tprint_success(f"Enhanced regime discovery pipeline completed in {results['processing_time']:.2f}s")
            return results
            
        except Exception as e:
            tprint_error(f"Enhanced regime discovery failed: {e}")
            return {'error': str(e), 'success': False}
    
    def demonstrate_enhanced_features(self, market_data: pd.DataFrame, 
                                    regime_labels: np.ndarray) -> Dict[str, Any]:
        """Demonstrate enhanced features of the system."""
        try:
            tprint_info("Demonstrating enhanced features")
            
            demo_results = {
                'probability_calculation': {},
                'model_persistence': {},
                'uncertainty_quantification': {},
                'ensemble_prediction': {},
                'success': False
            }
            
            # Demonstrate enhanced probability calculation
            tprint_info("Demonstrating enhanced probability calculation...")
            
            # Use last 100 samples for demonstration
            demo_features = market_data[['close', 'volume']].iloc[-100:].values
            
            # Scale features (simplified)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            demo_features_scaled = scaler.fit_transform(demo_features)
            
            # Get enhanced predictions
            prediction_result = self.hdbscan_clusterer.enhanced_predict_with_uncertainty(demo_features_scaled)
            
            if prediction_result.get('success', False):
                demo_results['probability_calculation'] = {
                    'method_breakdown': prediction_result.get('method_breakdown', {}),
                    'uncertainty_measures': prediction_result.get('uncertainty_measures', {}),
                    'n_samples': len(demo_features_scaled)
                }
                tprint_success("Enhanced probability calculation demonstrated")
            else:
                tprint_warning("Enhanced probability calculation demonstration failed")
            
            # Demonstrate model persistence
            tprint_info("Demonstrating model persistence...")
            
            # Save model
            save_success = self.hdbscan_clusterer.save_model("demo_model.pkl")
            if save_success:
                # Load model
                load_success = self.hdbscan_clusterer.load_model("demo_model.pkl")
                if load_success:
                    demo_results['model_persistence'] = {
                        'save_success': True,
                        'load_success': True,
                        'model_metadata': self.hdbscan_clusterer.model_metadata
                    }
                    tprint_success("Model persistence demonstrated")
                else:
                    tprint_warning("Model loading failed")
            else:
                tprint_warning("Model saving failed")
            
            # Demonstrate uncertainty quantification
            if prediction_result.get('success', False):
                uncertainty_measures = prediction_result.get('uncertainty_measures', {})
                demo_results['uncertainty_quantification'] = {
                    'method_agreement': uncertainty_measures.get('method_agreement', 0),
                    'probability_variance': uncertainty_measures.get('probability_variance', 0),
                    'low_confidence_ratio': uncertainty_measures.get('low_confidence_ratio', 0),
                    'noise_ratio': uncertainty_measures.get('noise_ratio', 0)
                }
                tprint_success("Uncertainty quantification demonstrated")
            
            # Demonstrate ensemble prediction
            if prediction_result.get('success', False):
                method_breakdown = prediction_result.get('method_breakdown', {})
                demo_results['ensemble_prediction'] = {
                    'n_methods': len(method_breakdown),
                    'methods_used': list(method_breakdown.keys()),
                    'ensemble_weights': self.hdbscan_clusterer.config.ensemble_weights
                }
                tprint_success("Ensemble prediction demonstrated")
            
            demo_results['success'] = True
            tprint_success("Enhanced features demonstration completed")
            return demo_results
            
        except Exception as e:
            tprint_error(f"Enhanced features demonstration failed: {e}")
            return {'error': str(e), 'success': False}
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report of the enhanced regime discovery."""
        try:
            report = []
            report.append("=" * 100)
            report.append("ENHANCED HDBSCAN ECONOMIC PROFILING SYSTEM - COMPREHENSIVE REPORT")
            report.append("=" * 100)
            report.append("")
            
            # Pipeline overview
            report.append("PIPELINE OVERVIEW:")
            report.append(f"  Processing Time: {results.get('processing_time', 0):.2f} seconds")
            report.append(f"  Success: {results.get('success', False)}")
            report.append("")
            
            # Pipeline steps
            report.append("PIPELINE STEPS:")
            pipeline_steps = results.get('pipeline_steps', {})
            for step_name, step_info in pipeline_steps.items():
                report.append(f"  {step_name.replace('_', ' ').title()}:")
                for key, value in step_info.items():
                    if key != 'success':
                        report.append(f"    {key}: {value}")
                report.append(f"    Success: {step_info.get('success', False)}")
                report.append("")
            
            # Regime analysis
            regime_labels = results.get('regime_labels')
            if regime_labels is not None:
                unique_regimes = np.unique(regime_labels)
                unique_regimes = unique_regimes[unique_regimes != -1]
                
                report.append("REGIME ANALYSIS:")
                report.append(f"  Number of Regimes: {len(unique_regimes)}")
                report.append(f"  Noise Points: {np.sum(regime_labels == -1)}")
                report.append(f"  Noise Ratio: {np.sum(regime_labels == -1) / len(regime_labels):.3f}")
                
                # Regime durations
                regime_durations = []
                for regime in unique_regimes:
                    regime_mask = regime_labels == regime
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
                        regime_durations.extend(consecutive_periods)
                
                if regime_durations:
                    report.append(f"  Min Regime Duration: {min(regime_durations)}")
                    report.append(f"  Max Regime Duration: {max(regime_durations)}")
                    report.append(f"  Avg Regime Duration: {np.mean(regime_durations):.1f}")
                report.append("")
            
            # Economic profiles
            economic_profiles = results.get('economic_profiles', {})
            if economic_profiles and 'regime_profiles' in economic_profiles:
                report.append("ECONOMIC PROFILES:")
                for i, profile in enumerate(economic_profiles['regime_profiles']):
                    report.append(f"  Regime {i+1} ({profile.get('regime_name', 'Unknown')}):")
                    report.append(f"    Mean Return: {profile.get('mean_return', 0):.4f}")
                    report.append(f"    Volatility: {profile.get('volatility', 0):.4f}")
                    report.append(f"    Sharpe Ratio: {profile.get('sharpe_ratio', 0):.4f}")
                    report.append(f"    Max Drawdown: {profile.get('max_drawdown', 0):.4f}")
                    report.append(f"    Quality Score: {profile.get('quality_score', 0):.3f}")
                report.append("")
            
            # Validation results
            validation_results = results.get('validation_results', {})
            if validation_results:
                report.append("VALIDATION RESULTS:")
                report.append(f"  Overall Score: {validation_results.get('overall_score', 0):.3f}")
                
                # Regime profiling validation
                regime_validation = validation_results.get('regime_profiling', {})
                report.append(f"  Regime Profiling Valid: {regime_validation.get('is_valid', False)}")
                report.append(f"  Regime Stability: {regime_validation.get('regime_stability', 0):.3f}")
                
                # Statistical validation
                stat_validation = validation_results.get('statistical_analysis', {})
                report.append(f"  Statistical Analysis Valid: {stat_validation.get('is_valid', False)}")
                
                # Economic validation
                econ_validation = validation_results.get('economic_validation', {})
                report.append(f"  Economic Validation Valid: {econ_validation.get('is_valid', False)}")
                
                # Cross-validation
                cv_validation = validation_results.get('cross_validation', {})
                report.append(f"  Cross-Validation Valid: {cv_validation.get('is_valid', False)}")
                report.append(f"  Mean CV Score: {cv_validation.get('mean_cv_score', 0):.3f}")
                report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            overall_score = validation_results.get('overall_score', 0) if validation_results else 0
            
            if overall_score < 0.5:
                report.append("  - System needs significant improvements")
                report.append("  - Consider adjusting clustering parameters")
                report.append("  - Review feature engineering process")
            elif overall_score < 0.7:
                report.append("  - System is functional but needs optimization")
                report.append("  - Fine-tune hyperparameters")
                report.append("  - Improve feature selection")
            elif overall_score < 0.9:
                report.append("  - System is good with minor improvements needed")
                report.append("  - Consider ensemble methods")
                report.append("  - Optimize temporal stabilization")
            else:
                report.append("  - System is excellent and ready for production")
                report.append("  - Consider real-time implementation")
                report.append("  - Add monitoring and alerting")
            
            report.append("")
            report.append("=" * 100)
            
            return "\n".join(report)
            
        except Exception as e:
            tprint_error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

def main():
    """Main function to run the enhanced regime discovery example."""
    try:
        tprint_info("Starting Enhanced HDBSCAN Regime Discovery Example")
        
        # Create example instance
        example = EnhancedRegimeDiscoveryExample()
        
        # Generate sample market data
        market_data = example.generate_sample_market_data(n_periods=1000)
        tprint_success(f"Generated market data: {len(market_data)} periods")
        
        # Run enhanced regime discovery
        results = example.run_enhanced_regime_discovery(market_data)
        
        if results.get('success', False):
            tprint_success("Enhanced regime discovery completed successfully")
            
            # Demonstrate enhanced features
            demo_results = example.demonstrate_enhanced_features(
                market_data, results['regime_labels']
            )
            
            if demo_results.get('success', False):
                tprint_success("Enhanced features demonstration completed")
            else:
                tprint_warning("Enhanced features demonstration failed")
            
            # Generate comprehensive report
            report = example.generate_comprehensive_report(results)
            print("\n" + report)
            
            # Save report to file
            with open("enhanced_regime_discovery_report.txt", "w") as f:
                f.write(report)
            tprint_success("Report saved to enhanced_regime_discovery_report.txt")
            
        else:
            tprint_error("Enhanced regime discovery failed")
            print(f"Error: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        tprint_error(f"Example execution failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()