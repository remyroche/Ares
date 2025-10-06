"""
Enhanced Labels Validation Suite

This module provides comprehensive validation and testing for the enhanced data and labels system,
ensuring that all components work correctly and produce high-quality results.

Key Validation Areas:
1. Label Quality Validation
2. Data Quality Validation
3. Stability Validation
4. Integration Validation
5. Performance Validation
6. Trading Objective Validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
import time
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import warnings

# Import the enhanced system
from .enhanced_data_labels_system import EnhancedDataLabelsSystem, EnhancedDataLabelsConfig
from .infrastructure_integration import get_integration_manager, process_market_data_enhanced

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


class EnhancedLabelsValidator:
    """
    Comprehensive validator for the enhanced data and labels system.
    
    This validator ensures that the enhanced system produces high-quality,
    stable, and trading-relevant labels that meet all requirements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced labels validator."""
        self.config = config or {}
        self.logger = logging.getLogger('EnhancedLabelsValidator')
        
        # Validation results storage
        self.validation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'processing_times': [],
            'quality_scores': [],
            'stability_scores': []
        }
        
        tprint_success("🚀 Enhanced Labels Validator initialized")
    
    def run_comprehensive_validation(
        self,
        test_data: Optional[pd.DataFrame] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation of the enhanced data and labels system.
        
        Args:
            test_data: Optional test data (will generate synthetic data if not provided)
            validation_config: Optional validation configuration
            
        Returns:
            Comprehensive validation results
        """
        start_time = time.time()
        tprint_info("🔍 Starting comprehensive validation of enhanced data and labels system")
        
        try:
            # Generate test data if not provided
            if test_data is None:
                test_data = self._generate_synthetic_test_data()
            
            # Run all validation tests
            validation_results = {
                'timestamp': datetime.now(),
                'test_data_info': {
                    'shape': test_data.shape,
                    'date_range': (test_data.index[0], test_data.index[-1]) if isinstance(test_data.index, pd.DatetimeIndex) else None
                },
                'validation_tests': {}
            }
            
            # Test 1: Data Quality Validation
            tprint_info("🧹 Test 1: Data Quality Validation")
            data_quality_results = self._validate_data_quality(test_data)
            validation_results['validation_tests']['data_quality'] = data_quality_results
            
            # Test 2: Label Generation Validation
            tprint_info("🎯 Test 2: Label Generation Validation")
            label_generation_results = self._validate_label_generation(test_data)
            validation_results['validation_tests']['label_generation'] = label_generation_results
            
            # Test 3: Label Quality Validation
            tprint_info("📊 Test 3: Label Quality Validation")
            label_quality_results = self._validate_label_quality(test_data)
            validation_results['validation_tests']['label_quality'] = label_quality_results
            
            # Test 4: Stability Validation
            tprint_info("🔍 Test 4: Stability Validation")
            stability_results = self._validate_stability(test_data)
            validation_results['validation_tests']['stability'] = stability_results
            
            # Test 5: Trading Objective Validation
            tprint_info("💰 Test 5: Trading Objective Validation")
            trading_objective_results = self._validate_trading_objectives(test_data)
            validation_results['validation_tests']['trading_objectives'] = trading_objective_results
            
            # Test 6: Integration Validation
            tprint_info("🔗 Test 6: Integration Validation")
            integration_results = self._validate_integration(test_data)
            validation_results['validation_tests']['integration'] = integration_results
            
            # Test 7: Performance Validation
            tprint_info("⚡ Test 7: Performance Validation")
            performance_results = self._validate_performance(test_data)
            validation_results['validation_tests']['performance'] = performance_results
            
            # Calculate overall validation score
            overall_score = self._calculate_overall_validation_score(validation_results['validation_tests'])
            validation_results['overall_score'] = overall_score
            validation_results['overall_status'] = self._determine_validation_status(overall_score)
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_validation_recommendations(
                validation_results['validation_tests']
            )
            
            # Store in history
            self.validation_history.append(validation_results)
            
            validation_time = time.time() - start_time
            validation_results['validation_time'] = validation_time
            
            tprint_success(f"✅ Comprehensive validation completed in {validation_time:.2f}s")
            tprint_info(f"   → Overall score: {overall_score:.3f} ({validation_results['overall_status']})")
            tprint_info(f"   → Tests passed: {sum(1 for test in validation_results['validation_tests'].values() if test.get('passed', False))}/{len(validation_results['validation_tests'])}")
            
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ Comprehensive validation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(),
                'overall_score': 0.0,
                'overall_status': 'failed'
            }
    
    def _generate_synthetic_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test data for validation."""
        try:
            tprint_info(f"📊 Generating synthetic test data ({n_samples} samples)")
            
            # Generate datetime index
            start_date = datetime.now() - timedelta(days=n_samples // 24)  # Assuming hourly data
            dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
            
            # Generate synthetic price data
            np.random.seed(42)  # For reproducibility
            
            # Start with base price
            base_price = 100.0
            returns = np.random.normal(0, 0.02, n_samples)  # 2% volatility
            
            # Add some trend and volatility clustering
            trend = np.linspace(0, 0.1, n_samples)  # Slight upward trend
            volatility_cluster = np.random.normal(0, 0.01, n_samples)
            volatility_cluster = np.convolve(volatility_cluster, np.ones(10)/10, mode='same')  # Smooth volatility
            
            returns = returns + trend + volatility_cluster
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, n_samples)
            }, index=dates)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
            
            tprint_success(f"✅ Synthetic test data generated: {data.shape}")
            return data
            
        except Exception as e:
            tprint_error(f"❌ Synthetic test data generation failed: {e}")
            # Return minimal test data
            return pd.DataFrame({
                'open': [100, 101, 102],
                'high': [101, 102, 103],
                'low': [99, 100, 101],
                'close': [100.5, 101.5, 102.5],
                'volume': [1000, 1100, 1200]
            })
    
    def _validate_data_quality(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality aspects."""
        try:
            tprint_info("🧹 Validating data quality...")
            
            # Test with enhanced data and labels system
            enhanced_config = EnhancedDataLabelsConfig()
            enhanced_system = EnhancedDataLabelsSystem(enhanced_config)
            
            result = enhanced_system.process_market_data(test_data)
            
            # Extract quality metrics
            data_quality = result.get('data_quality', {})
            quality_score = data_quality.get('quality_score', 0.0)
            quality_level = data_quality.get('quality_level', 'unknown')
            
            # Validate quality thresholds
            min_quality_threshold = 0.7
            quality_passed = quality_score >= min_quality_threshold
            
            validation_result = {
                'passed': quality_passed,
                'quality_score': quality_score,
                'quality_level': str(quality_level),
                'samples_removed': data_quality.get('samples_removed', 0),
                'features_removed': data_quality.get('features_removed', 0),
                'threshold': min_quality_threshold,
                'details': f"Data quality {quality_level} with score {quality_score:.3f}"
            }
            
            if quality_passed:
                tprint_success(f"✅ Data quality validation passed: {quality_score:.3f}")
            else:
                tprint_warning(f"⚠️ Data quality validation failed: {quality_score:.3f} < {min_quality_threshold}")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Data quality validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'quality_score': 0.0,
                'details': f"Data quality validation failed: {e}"
            }
    
    def _validate_label_generation(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate label generation functionality."""
        try:
            tprint_info("🎯 Validating label generation...")
            
            # Test with enhanced data and labels system
            enhanced_config = EnhancedDataLabelsConfig()
            enhanced_system = EnhancedDataLabelsSystem(enhanced_config)
            
            result = enhanced_system.process_market_data(test_data)
            
            # Check if labels were generated
            labels = result.get('labels', pd.DataFrame())
            confidence_scores = result.get('confidence_scores', pd.DataFrame())
            
            # Validate label structure
            required_columns = ['analyst_label', 'tactician_label']
            has_required_columns = all(col in labels.columns for col in required_columns)
            
            # Validate label values
            analyst_labels = labels.get('analyst_label', pd.Series())
            tactician_labels = labels.get('tactician_label', pd.Series())
            
            analyst_valid = analyst_labels.isin([0, 1]).all() if not analyst_labels.empty else False
            tactician_valid = tactician_labels.isin([0, 1]).all() if not tactician_labels.empty else False
            
            # Check for reasonable label distribution
            analyst_positive_ratio = analyst_labels.mean() if not analyst_labels.empty else 0
            tactician_positive_ratio = tactician_labels.mean() if not tactician_labels.empty else 0
            
            # Labels should not be all 0 or all 1
            analyst_balanced = 0.1 <= analyst_positive_ratio <= 0.9
            tactician_balanced = 0.1 <= tactician_positive_ratio <= 0.9
            
            # Overall validation
            generation_passed = (
                has_required_columns and 
                analyst_valid and 
                tactician_valid and 
                analyst_balanced and 
                tactician_balanced
            )
            
            validation_result = {
                'passed': generation_passed,
                'has_required_columns': has_required_columns,
                'analyst_valid': analyst_valid,
                'tactician_valid': tactician_valid,
                'analyst_balanced': analyst_balanced,
                'tactician_balanced': tactician_balanced,
                'analyst_positive_ratio': analyst_positive_ratio,
                'tactician_positive_ratio': tactician_positive_ratio,
                'total_labels': len(labels),
                'details': f"Generated {len(labels)} labels with analyst ratio {analyst_positive_ratio:.3f}"
            }
            
            if generation_passed:
                tprint_success(f"✅ Label generation validation passed")
            else:
                tprint_warning(f"⚠️ Label generation validation failed")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Label generation validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': f"Label generation validation failed: {e}"
            }
    
    def _validate_label_quality(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate label quality metrics."""
        try:
            tprint_info("📊 Validating label quality...")
            
            # Test with enhanced data and labels system
            enhanced_config = EnhancedDataLabelsConfig()
            enhanced_system = EnhancedDataLabelsSystem(enhanced_config)
            
            result = enhanced_system.process_market_data(test_data)
            
            # Extract labels and quality metrics
            labels = result.get('labels', pd.DataFrame())
            final_quality = result.get('final_quality', {})
            
            # Check final quality score
            overall_quality = final_quality.get('overall_score', 0.0)
            quality_grade = final_quality.get('quality_grade', 'F')
            is_acceptable = final_quality.get('is_acceptable', False)
            
            # Check component scores
            component_scores = final_quality.get('component_scores', {})
            data_quality_score = component_scores.get('data_quality', 0.0)
            label_stability_score = component_scores.get('label_stability', 0.0)
            class_balance_score = component_scores.get('class_balance', 0.0)
            
            # Validate quality thresholds
            min_overall_quality = 0.6
            min_component_quality = 0.5
            
            quality_passed = (
                overall_quality >= min_overall_quality and
                data_quality_score >= min_component_quality and
                label_stability_score >= min_component_quality and
                class_balance_score >= min_component_quality
            )
            
            validation_result = {
                'passed': quality_passed,
                'overall_quality': overall_quality,
                'quality_grade': quality_grade,
                'is_acceptable': is_acceptable,
                'component_scores': component_scores,
                'thresholds': {
                    'min_overall': min_overall_quality,
                    'min_component': min_component_quality
                },
                'details': f"Overall quality {quality_grade} with score {overall_quality:.3f}"
            }
            
            if quality_passed:
                tprint_success(f"✅ Label quality validation passed: {overall_quality:.3f}")
            else:
                tprint_warning(f"⚠️ Label quality validation failed: {overall_quality:.3f}")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Label quality validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'overall_quality': 0.0,
                'details': f"Label quality validation failed: {e}"
            }
    
    def _validate_stability(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate label stability."""
        try:
            tprint_info("🔍 Validating stability...")
            
            # Test with enhanced data and labels system
            enhanced_config = EnhancedDataLabelsConfig()
            enhanced_system = EnhancedDataLabelsSystem(enhanced_config)
            
            result = enhanced_system.process_market_data(test_data)
            
            # Extract stability metrics
            label_stability = result.get('label_stability', {})
            stability_level = label_stability.get('stability_level', 'unknown')
            overall_stability = label_stability.get('overall_stability', 0.0)
            
            # Check for leakage and drift
            leakage_results = label_stability.get('leakage_results', {})
            drift_results = label_stability.get('drift_results', {})
            autocorr_results = label_stability.get('autocorr_results', {})
            
            is_leakage_detected = leakage_results.get('is_leakage_detected', False)
            is_drift_detected = drift_results.get('is_drift_detected', False)
            is_high_autocorr = autocorr_results.get('is_high_autocorr', False)
            
            # Validate stability thresholds
            min_stability = 0.6
            stability_passed = (
                overall_stability >= min_stability and
                not is_leakage_detected and
                not is_drift_detected and
                not is_high_autocorr
            )
            
            validation_result = {
                'passed': stability_passed,
                'stability_level': str(stability_level),
                'overall_stability': overall_stability,
                'is_leakage_detected': is_leakage_detected,
                'is_drift_detected': is_drift_detected,
                'is_high_autocorr': is_high_autocorr,
                'threshold': min_stability,
                'details': f"Stability {stability_level} with score {overall_stability:.3f}"
            }
            
            if stability_passed:
                tprint_success(f"✅ Stability validation passed: {overall_stability:.3f}")
            else:
                tprint_warning(f"⚠️ Stability validation failed: {overall_stability:.3f}")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Stability validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'overall_stability': 0.0,
                'details': f"Stability validation failed: {e}"
            }
    
    def _validate_trading_objectives(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate that labels align with trading objectives."""
        try:
            tprint_info("💰 Validating trading objectives...")
            
            # Test with enhanced data and labels system
            enhanced_config = EnhancedDataLabelsConfig()
            enhanced_system = EnhancedDataLabelsSystem(enhanced_config)
            
            result = enhanced_system.process_market_data(test_data)
            
            # Extract labels and data
            labels = result.get('labels', pd.DataFrame())
            processed_data = result.get('processed_data', test_data)
            
            if labels.empty or processed_data.empty:
                return {
                    'passed': False,
                    'error': 'No labels or data available for trading objective validation',
                    'details': 'Cannot validate trading objectives without labels'
                }
            
            # Check analyst labels (Should we trade?)
            analyst_labels = labels.get('analyst_label', pd.Series())
            analyst_confidence = labels.get('analyst_confidence', pd.Series())
            
            # Analyst labels should be based on profitability
            # This is a simplified check - in practice, you'd validate against actual P&L
            analyst_positive_ratio = analyst_labels.mean()
            analyst_confidence_mean = analyst_confidence.mean()
            
            # Check tactician labels (Direction/magnitude)
            tactician_labels = labels.get('tactician_label', pd.Series())
            tactician_magnitude = labels.get('tactician_magnitude', pd.Series())
            
            tactician_positive_ratio = tactician_labels.mean()
            tactician_magnitude_mean = tactician_magnitude.mean()
            
            # Validate trading objective alignment
            # Labels should not be too extreme (all 0 or all 1)
            analyst_reasonable = 0.1 <= analyst_positive_ratio <= 0.9
            tactician_reasonable = 0.1 <= tactician_positive_ratio <= 0.9
            
            # Confidence should be reasonable
            confidence_reasonable = 0.3 <= analyst_confidence_mean <= 0.9
            
            # Magnitude should be reasonable
            magnitude_reasonable = 0.5 <= tactician_magnitude_mean <= 2.0
            
            trading_objectives_passed = (
                analyst_reasonable and
                tactician_reasonable and
                confidence_reasonable and
                magnitude_reasonable
            )
            
            validation_result = {
                'passed': trading_objectives_passed,
                'analyst_positive_ratio': analyst_positive_ratio,
                'tactician_positive_ratio': tactician_positive_ratio,
                'analyst_confidence_mean': analyst_confidence_mean,
                'tactician_magnitude_mean': tactician_magnitude_mean,
                'analyst_reasonable': analyst_reasonable,
                'tactician_reasonable': tactician_reasonable,
                'confidence_reasonable': confidence_reasonable,
                'magnitude_reasonable': magnitude_reasonable,
                'details': f"Analyst ratio {analyst_positive_ratio:.3f}, Tactician ratio {tactician_positive_ratio:.3f}"
            }
            
            if trading_objectives_passed:
                tprint_success(f"✅ Trading objectives validation passed")
            else:
                tprint_warning(f"⚠️ Trading objectives validation failed")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Trading objectives validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': f"Trading objectives validation failed: {e}"
            }
    
    def _validate_integration(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate integration with existing infrastructure."""
        try:
            tprint_info("🔗 Validating integration...")
            
            # Test with integration manager
            integration_result = process_market_data_enhanced(test_data)
            
            # Check if integration worked
            has_processed_data = 'processed_data' in integration_result
            has_labels = 'labels' in integration_result
            has_quality_metrics = 'data_quality' in integration_result
            has_stability_metrics = 'label_stability' in integration_result
            
            # Check integration status
            integration_status = integration_result.get('integration_status', {})
            components_available = sum(integration_status.values())
            total_components = len(integration_status)
            
            # Validate integration
            integration_passed = (
                has_processed_data and
                has_labels and
                has_quality_metrics and
                has_stability_metrics and
                components_available >= total_components * 0.8  # At least 80% of components available
            )
            
            validation_result = {
                'passed': integration_passed,
                'has_processed_data': has_processed_data,
                'has_labels': has_labels,
                'has_quality_metrics': has_quality_metrics,
                'has_stability_metrics': has_stability_metrics,
                'components_available': components_available,
                'total_components': total_components,
                'integration_status': integration_status,
                'details': f"Integration working with {components_available}/{total_components} components"
            }
            
            if integration_passed:
                tprint_success(f"✅ Integration validation passed")
            else:
                tprint_warning(f"⚠️ Integration validation failed")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Integration validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'details': f"Integration validation failed: {e}"
            }
    
    def _validate_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate performance characteristics."""
        try:
            tprint_info("⚡ Validating performance...")
            
            # Test processing time
            start_time = time.time()
            
            enhanced_config = EnhancedDataLabelsConfig()
            enhanced_system = EnhancedDataLabelsSystem(enhanced_config)
            result = enhanced_system.process_market_data(test_data)
            
            processing_time = time.time() - start_time
            
            # Check performance thresholds
            max_processing_time = 30.0  # 30 seconds max
            performance_passed = processing_time <= max_processing_time
            
            # Check memory usage (simplified)
            data_size_mb = test_data.memory_usage(deep=True).sum() / 1024 / 1024
            max_memory_mb = 1000.0  # 1GB max
            memory_passed = data_size_mb <= max_memory_mb
            
            # Overall performance
            overall_performance_passed = performance_passed and memory_passed
            
            validation_result = {
                'passed': overall_performance_passed,
                'processing_time': processing_time,
                'data_size_mb': data_size_mb,
                'max_processing_time': max_processing_time,
                'max_memory_mb': max_memory_mb,
                'performance_passed': performance_passed,
                'memory_passed': memory_passed,
                'details': f"Processing time {processing_time:.2f}s, Data size {data_size_mb:.2f}MB"
            }
            
            # Store performance metrics
            self.performance_metrics['processing_times'].append(processing_time)
            if 'final_quality' in result:
                self.performance_metrics['quality_scores'].append(result['final_quality'].get('overall_score', 0.0))
            if 'label_stability' in result:
                self.performance_metrics['stability_scores'].append(result['label_stability'].get('overall_stability', 0.0))
            
            if overall_performance_passed:
                tprint_success(f"✅ Performance validation passed: {processing_time:.2f}s")
            else:
                tprint_warning(f"⚠️ Performance validation failed: {processing_time:.2f}s")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Performance validation failed: {e}")
            return {
                'passed': False,
                'error': str(e),
                'processing_time': 0.0,
                'details': f"Performance validation failed: {e}"
            }
    
    def _calculate_overall_validation_score(self, validation_tests: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        try:
            scores = []
            weights = {
                'data_quality': 0.2,
                'label_generation': 0.2,
                'label_quality': 0.2,
                'stability': 0.15,
                'trading_objectives': 0.15,
                'integration': 0.05,
                'performance': 0.05
            }
            
            for test_name, test_result in validation_tests.items():
                if test_result.get('passed', False):
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            
            # Calculate weighted average
            if scores:
                overall_score = np.average(scores, weights=[weights.get(test_name, 1.0) for test_name in validation_tests.keys()])
            else:
                overall_score = 0.0
            
            return overall_score
            
        except Exception:
            return 0.0
    
    def _determine_validation_status(self, overall_score: float) -> str:
        """Determine validation status based on overall score."""
        if overall_score >= 0.9:
            return 'excellent'
        elif overall_score >= 0.8:
            return 'good'
        elif overall_score >= 0.7:
            return 'fair'
        elif overall_score >= 0.6:
            return 'poor'
        else:
            return 'failed'
    
    def _generate_validation_recommendations(self, validation_tests: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for test_name, test_result in validation_tests.items():
            if not test_result.get('passed', False):
                if test_name == 'data_quality':
                    recommendations.append("Improve data quality - check for missing values, outliers, and data consistency")
                elif test_name == 'label_generation':
                    recommendations.append("Fix label generation - ensure proper label structure and values")
                elif test_name == 'label_quality':
                    recommendations.append("Improve label quality - check quality thresholds and component scores")
                elif test_name == 'stability':
                    recommendations.append("Address stability issues - check for leakage, drift, and autocorrelation")
                elif test_name == 'trading_objectives':
                    recommendations.append("Align labels with trading objectives - ensure reasonable label distributions")
                elif test_name == 'integration':
                    recommendations.append("Fix integration issues - ensure all components are properly connected")
                elif test_name == 'performance':
                    recommendations.append("Optimize performance - reduce processing time and memory usage")
        
        if not recommendations:
            recommendations.append("All validation tests passed - system is working correctly")
        
        return recommendations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation runs."""
        try:
            if not self.validation_history:
                return {'message': 'No validation history available'}
            
            # Calculate summary statistics
            overall_scores = [run['overall_score'] for run in self.validation_history]
            validation_times = [run.get('validation_time', 0) for run in self.validation_history]
            
            summary = {
                'total_validations': len(self.validation_history),
                'avg_overall_score': np.mean(overall_scores),
                'max_overall_score': np.max(overall_scores),
                'min_overall_score': np.min(overall_scores),
                'avg_validation_time': np.mean(validation_times),
                'recent_status': self.validation_history[-1].get('overall_status', 'unknown'),
                'performance_metrics': {
                    'avg_processing_time': np.mean(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0,
                    'avg_quality_score': np.mean(self.performance_metrics['quality_scores']) if self.performance_metrics['quality_scores'] else 0,
                    'avg_stability_score': np.mean(self.performance_metrics['stability_scores']) if self.performance_metrics['stability_scores'] else 0
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}


# Convenience functions
def run_enhanced_labels_validation(
    test_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run comprehensive validation of the enhanced labels system."""
    validator = EnhancedLabelsValidator(config)
    return validator.run_comprehensive_validation(test_data)


def validate_system_integration() -> Dict[str, Any]:
    """Validate that the enhanced system is properly integrated."""
    try:
        # Test basic functionality
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Test enhanced processing
        result = process_market_data_enhanced(test_data)
        
        # Check if processing was successful
        success = 'error' not in result and 'processed_data' in result
        
        return {
            'integration_working': success,
            'test_result': result,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        return {
            'integration_working': False,
            'error': str(e),
            'timestamp': datetime.now()
        }