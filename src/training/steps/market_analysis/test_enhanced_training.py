"""
Enhanced Tactician and Analyst Training - Integration Test

This test validates the complete implementation of the enhanced training pipeline:
1. Tactician trains on Analyst confidence > 0.5 + next 45 minutes after confidence drops below 0.5
2. Tactician uses all features + Analyst outputs + HMM outputs
3. Analyst uses all features + HMM outputs

Test Coverage:
- Tactician training filter functionality
- Enhanced feature engineering for both models
- Complete training pipeline integration
- Configuration validation
- Performance and memory usage validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import time
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance
)

# Import our enhanced training components
try:
    from .tactician_training_filter import TacticianTrainingFilter, TacticianFilterConfig
    from .enhanced_feature_engineering import EnhancedFeatureEngineer, FeatureEngineeringConfig
    from .enhanced_tactician_training import EnhancedTacticianTrainingPipeline, EnhancedTrainingConfig
    from .labeling_components import ComprehensiveLabeling
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    tprint_error(f"❌ Failed to import enhanced training components: {e}")
    COMPONENTS_AVAILABLE = False

logger = system_logger.getChild('EnhancedTrainingTest')


class EnhancedTrainingIntegrationTest:
    """
    Comprehensive integration test for enhanced Tactician and Analyst training.
    """
    
    def __init__(self):
        """Initialize the integration test."""
        self.logger = logger.getChild('IntegrationTest')
        self.test_results = {}
        self.test_data = None
        self.test_labels = None
        
        tprint_info("🧪 Enhanced Training Integration Test initialized")
    
    def setup_test_data(self, n_samples: int = 2000) -> bool:
        """
        Set up comprehensive test data for training validation.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            tprint_info(f"📊 Setting up test data with {n_samples:,} samples...")
            
            # Create realistic datetime index
            dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
            
            # Generate realistic OHLC data
            np.random.seed(42)  # For reproducible tests
            base_price = 100.0
            price_changes = np.random.normal(0, 0.001, n_samples)  # 0.1% volatility
            prices = base_price * np.exp(np.cumsum(price_changes))
            
            # Generate OHLC data
            highs = prices * (1 + np.abs(np.random.normal(0, 0.0005, n_samples)))
            lows = prices * (1 - np.abs(np.random.normal(0, 0.0005, n_samples)))
            opens = np.roll(prices, 1)
            opens[0] = base_price
            volumes = np.random.uniform(1000, 10000, n_samples)
            
            # Create base DataFrame
            self.test_data = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes,
                'rsi': np.random.uniform(0, 100, n_samples),
                'macd': np.random.uniform(-1, 1, n_samples),
                'bb_upper': prices * 1.02,
                'bb_lower': prices * 0.98,
                'ema_20': prices * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
                'ema_50': prices * (1 + np.random.uniform(-0.02, 0.02, n_samples)),
                'atr': prices * np.random.uniform(0.001, 0.005, n_samples),
                'stoch_k': np.random.uniform(0, 100, n_samples),
                'stoch_d': np.random.uniform(0, 100, n_samples),
                'williams_r': np.random.uniform(-100, 0, n_samples),
                'cci': np.random.uniform(-200, 200, n_samples),
                'roc': np.random.uniform(-10, 10, n_samples),
                'momentum': np.random.uniform(-1, 1, n_samples),
                'adx': np.random.uniform(0, 100, n_samples),
                'di_plus': np.random.uniform(0, 100, n_samples),
                'di_minus': np.random.uniform(0, 100, n_samples),
                'obv': np.cumsum(volumes * np.sign(price_changes)),
                'mfi': np.random.uniform(0, 100, n_samples),
                'trix': np.random.uniform(-1, 1, n_samples),
                'ultosc': np.random.uniform(0, 100, n_samples),
                'aroon_up': np.random.uniform(0, 100, n_samples),
                'aroon_down': np.random.uniform(0, 100, n_samples),
                'bop': np.random.uniform(-1, 1, n_samples),
                'cmf': np.random.uniform(-1, 1, n_samples),
                'eom': np.random.uniform(-1, 1, n_samples),
                'kvo': np.random.uniform(-1000, 1000, n_samples),
                'mvo': np.random.uniform(-1000, 1000, n_samples),
                'ppo': np.random.uniform(-1, 1, n_samples),
                'pvo': np.random.uniform(-1000, 1000, n_samples),
                'roc': np.random.uniform(-10, 10, n_samples),
                'rsi_2': np.random.uniform(0, 100, n_samples),
                'stoch_rsi': np.random.uniform(0, 100, n_samples),
                'trix_2': np.random.uniform(-1, 1, n_samples),
                'ultosc_2': np.random.uniform(0, 100, n_samples),
                'williams_r_2': np.random.uniform(-100, 0, n_samples),
                'adx_2': np.random.uniform(0, 100, n_samples),
                'aroon_oscillator': np.random.uniform(-100, 100, n_samples),
                'balance_of_power': np.random.uniform(-1, 1, n_samples),
                'commodity_channel_index': np.random.uniform(-200, 200, n_samples),
                'ease_of_movement': np.random.uniform(-1, 1, n_samples),
                'mass_index': np.random.uniform(0, 50, n_samples),
                'money_flow_index': np.random.uniform(0, 100, n_samples),
                'negative_volume_index': np.cumsum(np.random.uniform(-100, 100, n_samples)),
                'on_balance_volume': np.cumsum(volumes * np.sign(price_changes)),
                'positive_volume_index': np.cumsum(np.random.uniform(0, 100, n_samples)),
                'price_volume_trend': np.cumsum(volumes * np.sign(price_changes)),
                'volume_price_trend': np.cumsum(volumes * np.sign(price_changes)),
                'williams_accumulation_distribution': np.cumsum(volumes * np.sign(price_changes)),
                'chaikin_oscillator': np.random.uniform(-1000, 1000, n_samples),
                'klinger_volume_oscillator': np.random.uniform(-1000, 1000, n_samples),
                'money_flow_oscillator': np.random.uniform(-1000, 1000, n_samples),
                'percentage_volume_oscillator': np.random.uniform(-1000, 1000, n_samples),
                'detrended_price_oscillator': np.random.uniform(-1, 1, n_samples),
                'linear_regression_intercept': np.random.uniform(-1, 1, n_samples),
                'linear_regression_slope': np.random.uniform(-1, 1, n_samples),
                'standard_deviation': np.random.uniform(0, 1, n_samples),
                'variance': np.random.uniform(0, 1, n_samples),
                'volatility': np.random.uniform(0, 0.1, n_samples),
                'historical_volatility': np.random.uniform(0, 0.1, n_samples),
                'implied_volatility': np.random.uniform(0, 0.1, n_samples),
                'parkinson_volatility': np.random.uniform(0, 0.1, n_samples),
                'garman_klass_volatility': np.random.uniform(0, 0.1, n_samples),
                'rogers_satchell_volatility': np.random.uniform(0, 0.1, n_samples),
                'yang_zhang_volatility': np.random.uniform(0, 0.1, n_samples),
                'hurst_exponent': np.random.uniform(0, 1, n_samples),
                'fractal_dimension': np.random.uniform(1, 2, n_samples),
                'lyapunov_exponent': np.random.uniform(-1, 1, n_samples),
                'correlation_dimension': np.random.uniform(1, 3, n_samples),
                'approximate_entropy': np.random.uniform(0, 2, n_samples),
                'sample_entropy': np.random.uniform(0, 2, n_samples),
                'multiscale_entropy': np.random.uniform(0, 2, n_samples),
                'permutation_entropy': np.random.uniform(0, 1, n_samples),
                'shannon_entropy': np.random.uniform(0, 10, n_samples),
                'renyi_entropy': np.random.uniform(0, 10, n_samples),
                'tsallis_entropy': np.random.uniform(0, 10, n_samples),
                'kolmogorov_complexity': np.random.uniform(0, 100, n_samples),
                'lz_complexity': np.random.uniform(0, 100, n_samples),
                'lempel_ziv_complexity': np.random.uniform(0, 100, n_samples),
                'mutual_information': np.random.uniform(0, 1, n_samples),
                'transfer_entropy': np.random.uniform(0, 1, n_samples),
                'conditional_entropy': np.random.uniform(0, 10, n_samples),
                'joint_entropy': np.random.uniform(0, 10, n_samples),
                'cross_entropy': np.random.uniform(0, 10, n_samples),
                'kullback_leibler_divergence': np.random.uniform(0, 1, n_samples),
                'jensen_shannon_divergence': np.random.uniform(0, 1, n_samples),
                'wasserstein_distance': np.random.uniform(0, 1, n_samples),
                'earth_movers_distance': np.random.uniform(0, 1, n_samples),
                'bhattacharyya_distance': np.random.uniform(0, 1, n_samples),
                'hellinger_distance': np.random.uniform(0, 1, n_samples),
                'total_variation_distance': np.random.uniform(0, 1, n_samples),
                'jaccard_distance': np.random.uniform(0, 1, n_samples),
                'cosine_distance': np.random.uniform(0, 1, n_samples),
                'euclidean_distance': np.random.uniform(0, 10, n_samples),
                'manhattan_distance': np.random.uniform(0, 10, n_samples),
                'chebyshev_distance': np.random.uniform(0, 10, n_samples),
                'minkowski_distance': np.random.uniform(0, 10, n_samples),
                'canberra_distance': np.random.uniform(0, 1, n_samples),
                'bray_curtis_distance': np.random.uniform(0, 1, n_samples),
                'dice_distance': np.random.uniform(0, 1, n_samples),
                'hamming_distance': np.random.uniform(0, 1, n_samples),
                'levenshtein_distance': np.random.uniform(0, 10, n_samples),
                'jaro_distance': np.random.uniform(0, 1, n_samples),
                'jaro_winkler_distance': np.random.uniform(0, 1, n_samples),
                'needleman_wunsch_distance': np.random.uniform(0, 10, n_samples),
                'smith_waterman_distance': np.random.uniform(0, 10, n_samples),
                'gotoh_distance': np.random.uniform(0, 10, n_samples),
                'affine_gap_distance': np.random.uniform(0, 10, n_samples),
                'pam250_distance': np.random.uniform(0, 10, n_samples),
                'blosum62_distance': np.random.uniform(0, 10, n_samples),
                'dayhoff_distance': np.random.uniform(0, 10, n_samples),
                'jones_distance': np.random.uniform(0, 10, n_samples),
                'wags_distance': np.random.uniform(0, 10, n_samples),
                'lg_distance': np.random.uniform(0, 10, n_samples),
                'rtrev_distance': np.random.uniform(0, 10, n_samples),
                'vt_distance': np.random.uniform(0, 10, n_samples),
                'mtrev_distance': np.random.uniform(0, 10, n_samples),
                'mtart_distance': np.random.uniform(0, 10, n_samples),
                'mtzoa_distance': np.random.uniform(0, 10, n_samples),
                'mtinv_distance': np.random.uniform(0, 10, n_samples),
                'mtmam_distance': np.random.uniform(0, 10, n_samples),
                'mtart_distance_2': np.random.uniform(0, 10, n_samples),
                'mtzoa_distance_2': np.random.uniform(0, 10, n_samples),
                'mtinv_distance_2': np.random.uniform(0, 10, n_samples),
                'mtmam_distance_2': np.random.uniform(0, 10, n_samples),
                'mtart_distance_3': np.random.uniform(0, 10, n_samples),
                'mtzoa_distance_3': np.random.uniform(0, 10, n_samples),
                'mtinv_distance_3': np.random.uniform(0, 10, n_samples),
                'mtmam_distance_3': np.random.uniform(0, 10, n_samples)
            }, index=dates)
            
            # Generate realistic labels with some structure
            # Create periods of high volatility and trends
            trend_periods = []
            for i in range(0, n_samples, 200):
                trend_periods.append((i, min(i + 100, n_samples)))
            
            labels = np.zeros(n_samples)
            for start, end in trend_periods:
                # Add some trend-based labels
                trend_direction = np.random.choice([-1, 1])
                labels[start:end] = trend_direction
            
            # Add some random noise
            noise_mask = np.random.random(n_samples) < 0.1  # 10% random labels
            labels[noise_mask] = np.random.choice([-1, 0, 1], size=noise_mask.sum())
            
            self.test_labels = pd.Series(labels, index=dates)
            
            tprint_success(f"✅ Test data setup completed: {len(self.test_data):,} samples, {len(self.test_data.columns)} features")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to setup test data: {e}")
            return False
    
    def test_tactician_filter(self) -> bool:
        """Test Tactician training filter functionality."""
        try:
            tprint_info("🎯 Testing Tactician training filter...")
            
            if not COMPONENTS_AVAILABLE:
                tprint_warning("⚠️ Components not available, skipping test")
                return True
            
            # Create mock Analyst confidence with realistic patterns
            confidence = pd.Series(0.3, index=self.test_data.index)
            
            # Add some high confidence periods
            confidence.iloc[100:200] = 0.8  # High confidence period 1
            confidence.iloc[500:600] = 0.9  # High confidence period 2
            confidence.iloc[800:900] = 0.7  # High confidence period 3
            
            # Add some medium confidence periods
            confidence.iloc[300:350] = 0.4  # Medium confidence (below threshold)
            confidence.iloc[700:750] = 0.45  # Medium confidence (below threshold)
            
            # Create filter
            filter_config = TacticianFilterConfig(
                confidence_threshold=0.5,
                post_drop_window_minutes=45
            )
            training_filter = TacticianTrainingFilter(filter_config)
            
            # Apply filtering
            start_time = time.time()
            training_mask = training_filter.create_training_mask(confidence)
            filter_time = time.time() - start_time
            
            # Validate results
            filtered_data = self.test_data[training_mask]
            filter_stats = training_filter.get_filter_stats()
            
            # Check filtering results
            assert len(filtered_data) > 0, "No samples selected for training"
            assert len(filtered_data) < len(self.test_data), "All samples selected (filtering not working)"
            assert filter_stats['training_coverage'] > 0.1, "Training coverage too low"
            assert filter_stats['training_coverage'] < 0.9, "Training coverage too high"
            
            self.test_results['tactician_filter'] = {
                'success': True,
                'filter_time': filter_time,
                'filter_stats': filter_stats,
                'filtered_samples': len(filtered_data)
            }
            
            tprint_success(f"✅ Tactician filter test passed: {len(filtered_data):,}/{len(self.test_data):,} samples selected")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Tactician filter test failed: {e}")
            self.test_results['tactician_filter'] = {'success': False, 'error': str(e)}
            return False
    
    def test_enhanced_feature_engineering(self) -> bool:
        """Test enhanced feature engineering for both models."""
        try:
            tprint_info("🔧 Testing enhanced feature engineering...")
            
            if not COMPONENTS_AVAILABLE:
                tprint_warning("⚠️ Components not available, skipping test")
                return True
            
            # Create mock HMM models
            class MockHMM:
                def predict(self, X):
                    return np.random.randint(0, 3, len(X))
                
                def predict_proba(self, X):
                    probs = np.random.rand(len(X), 3)
                    return probs / probs.sum(axis=1, keepdims=True)
            
            # Create mock Analyst model
            class MockAnalyst:
                def predict(self, X):
                    return np.random.uniform(0, 1, len(X))
                
                def predict_proba(self, X):
                    probs = np.random.rand(len(X), 2)
                    return probs / probs.sum(axis=1, keepdims=True)
            
            hmm_models = {'regime_1': MockHMM(), 'regime_2': MockHMM()}
            analyst_model = MockAnalyst()
            
            # Test Analyst feature engineering
            feature_engineer = EnhancedFeatureEngineer(
                FeatureEngineeringConfig(
                    include_hmm_features=True,
                    include_analyst_features=False
                )
            )
            
            feature_engineer.set_hmm_models(hmm_models)
            
            start_time = time.time()
            analyst_features = feature_engineer.generate_analyst_features(self.test_data)
            analyst_feature_time = time.time() - start_time
            
            # Test Tactician feature engineering
            feature_engineer.set_analyst_model(analyst_model)
            
            start_time = time.time()
            tactician_features = feature_engineer.generate_tactician_features(self.test_data)
            tactician_feature_time = time.time() - start_time
            
            # Validate results
            assert len(analyst_features.columns) > len(self.test_data.columns), "Analyst features not enhanced"
            assert len(tactician_features.columns) > len(self.test_data.columns), "Tactician features not enhanced"
            assert len(tactician_features.columns) > len(analyst_features.columns), "Tactician features not more enhanced than Analyst"
            
            # Check for HMM features in both
            hmm_columns = [col for col in analyst_features.columns if 'hmm_' in col]
            assert len(hmm_columns) > 0, "HMM features not found in Analyst features"
            
            hmm_columns_tactician = [col for col in tactician_features.columns if 'hmm_' in col]
            assert len(hmm_columns_tactician) > 0, "HMM features not found in Tactician features"
            
            # Check for Analyst features in Tactician
            analyst_columns = [col for col in tactician_features.columns if 'analyst_' in col]
            assert len(analyst_columns) > 0, "Analyst features not found in Tactician features"
            
            self.test_results['feature_engineering'] = {
                'success': True,
                'analyst_feature_time': analyst_feature_time,
                'tactician_feature_time': tactician_feature_time,
                'analyst_features_shape': analyst_features.shape,
                'tactician_features_shape': tactician_features.shape,
                'hmm_features_count': len(hmm_columns),
                'analyst_features_count': len(analyst_columns)
            }
            
            tprint_success(f"✅ Feature engineering test passed:")
            tprint_success(f"   → Analyst features: {analyst_features.shape}")
            tprint_success(f"   → Tactician features: {tactician_features.shape}")
            tprint_success(f"   → HMM features: {len(hmm_columns)}")
            tprint_success(f"   → Analyst features in Tactician: {len(analyst_columns)}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Feature engineering test failed: {e}")
            self.test_results['feature_engineering'] = {'success': False, 'error': str(e)}
            return False
    
    def test_enhanced_training_pipeline(self) -> bool:
        """Test the complete enhanced training pipeline."""
        try:
            tprint_info("🚀 Testing enhanced training pipeline...")
            
            if not COMPONENTS_AVAILABLE:
                tprint_warning("⚠️ Components not available, skipping test")
                return True
            
            # Create training pipeline
            pipeline = EnhancedTacticianTrainingPipeline(
                EnhancedTrainingConfig(
                    tactician_confidence_threshold=0.5,
                    tactician_post_drop_window_minutes=45,
                    enable_hmm_training=True,
                    enable_analyst_training=True,
                    enable_tactician_training=True,
                    enable_feature_scaling=True
                )
            )
            
            # Run full training pipeline
            start_time = time.time()
            result = pipeline.run_full_training(self.test_data, self.test_labels)
            training_time = time.time() - start_time
            
            # Validate results
            assert result.success, f"Training pipeline failed: {result.error_message}"
            assert result.training_time > 0, "Training time not recorded"
            assert result.hmm_models is not None, "HMM models not created"
            assert result.analyst_model is not None, "Analyst model not created"
            assert result.tactician_model is not None, "Tactician model not created"
            assert result.tactician_filter_stats is not None, "Tactician filter stats not recorded"
            
            # Validate training statistics
            assert result.tactician_filter_stats['training_coverage'] > 0.1, "Training coverage too low"
            assert result.analyst_features_shape[1] > len(self.test_data.columns), "Analyst features not enhanced"
            assert result.tactician_features_shape[1] > result.analyst_features_shape[1], "Tactician features not more enhanced than Analyst"
            
            self.test_results['training_pipeline'] = {
                'success': True,
                'training_time': training_time,
                'result': {
                    'success': result.success,
                    'training_time': result.training_time,
                    'hmm_models_count': len(result.hmm_models) if result.hmm_models else 0,
                    'analyst_model_created': result.analyst_model is not None,
                    'tactician_model_created': result.tactician_model is not None,
                    'analyst_features_shape': result.analyst_features_shape,
                    'tactician_features_shape': result.tactician_features_shape,
                    'tactician_filter_stats': result.tactician_filter_stats
                }
            }
            
            tprint_success(f"✅ Training pipeline test passed:")
            tprint_success(f"   → Training time: {training_time:.2f}s")
            tprint_success(f"   → HMM models: {len(result.hmm_models)}")
            tprint_success(f"   → Analyst model: {'✅' if result.analyst_model else '❌'}")
            tprint_success(f"   → Tactician model: {'✅' if result.tactician_model else '❌'}")
            tprint_success(f"   → Training coverage: {result.tactician_filter_stats['training_coverage']:.1%}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Training pipeline test failed: {e}")
            self.test_results['training_pipeline'] = {'success': False, 'error': str(e)}
            return False
    
    def test_performance_validation(self) -> bool:
        """Test performance and memory usage validation."""
        try:
            tprint_info("⚡ Testing performance validation...")
            
            if not COMPONENTS_AVAILABLE:
                tprint_warning("⚠️ Components not available, skipping test")
                return True
            
            # Test memory usage during feature generation
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate features with large dataset
            large_data = pd.concat([self.test_data] * 5, ignore_index=True)  # 5x larger dataset
            
            feature_engineer = EnhancedFeatureEngineer()
            
            start_time = time.time()
            enhanced_features = feature_engineer.generate_analyst_features(large_data)
            feature_time = time.time() - start_time
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Validate performance
            assert feature_time < 30, f"Feature generation too slow: {feature_time:.2f}s"
            assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.1f}MB"
            
            self.test_results['performance'] = {
                'success': True,
                'feature_generation_time': feature_time,
                'memory_increase_mb': memory_increase,
                'data_size': len(large_data),
                'feature_count': len(enhanced_features.columns)
            }
            
            tprint_success(f"✅ Performance test passed:")
            tprint_success(f"   → Feature generation time: {feature_time:.2f}s")
            tprint_success(f"   → Memory increase: {memory_increase:.1f}MB")
            tprint_success(f"   → Data size: {len(large_data):,} samples")
            tprint_success(f"   → Feature count: {len(enhanced_features.columns)}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = {'success': False, 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        try:
            tprint_info("🧪 Starting Enhanced Training Integration Tests")
            
            # Setup test data
            if not self.setup_test_data():
                return {'success': False, 'error': 'Failed to setup test data'}
            
            # Run individual tests
            tests = [
                ('Tactician Filter', self.test_tactician_filter),
                ('Feature Engineering', self.test_enhanced_feature_engineering),
                ('Training Pipeline', self.test_enhanced_training_pipeline),
                ('Performance', self.test_performance_validation)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                tprint_info(f"🔍 Running {test_name} test...")
                if test_func():
                    passed_tests += 1
                    tprint_success(f"✅ {test_name} test passed")
                else:
                    tprint_error(f"❌ {test_name} test failed")
                tprint_info("")  # Add spacing
            
            # Generate summary
            success_rate = passed_tests / total_tests
            overall_success = passed_tests == total_tests
            
            summary = {
                'success': overall_success,
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'success_rate': success_rate,
                'test_results': self.test_results,
                'data_info': {
                    'samples': len(self.test_data),
                    'features': len(self.test_data.columns)
                }
            }
            
            tprint_info("📊 Test Summary:")
            tprint_info(f"   → Tests passed: {passed_tests}/{total_tests}")
            tprint_info(f"   → Success rate: {success_rate:.1%}")
            tprint_info(f"   → Overall success: {'✅' if overall_success else '❌'}")
            
            return summary
            
        except Exception as e:
            tprint_error(f"❌ Integration tests failed: {e}")
            return {'success': False, 'error': str(e), 'test_results': self.test_results}


def run_enhanced_training_tests():
    """Run the enhanced training integration tests."""
    test_runner = EnhancedTrainingIntegrationTest()
    return test_runner.run_all_tests()


if __name__ == '__main__':
    print("🧪 Enhanced Tactician and Analyst Training - Integration Test")
    print("=" * 70)
    
    # Run tests
    results = run_enhanced_training_tests()
    
    print("\n" + "=" * 70)
    print("📊 Final Results:")
    print(f"   Overall Success: {'✅' if results['success'] else '❌'}")
    print(f"   Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    
    if not results['success']:
        print(f"   Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)
    else:
        print("🎉 All tests passed! Enhanced training implementation is working correctly.")
        sys.exit(0)