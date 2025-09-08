#!/usr/bin/env python3
"""
Comprehensive Fail-Fast Validation Example

This example demonstrates the enhanced fail-fast validation system that covers
all important aspects beyond just per-regime validation, including:

1. Data Quality Validation
2. Performance Validation  
3. Model Quality Validation
4. Feature Quality Validation
5. Execution Environment Validation
6. Business Logic Validation
7. Regime Quality Validation (for post-HMM steps)
8. Empty Running Detection

The system provides comprehensive validation to prevent empty running or
important degradation across all critical aspects of the training pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

# Import the enhanced fail-fast validation system
from src.utils.enhanced_financial_metrics_logger import (
import logging

    EnhancedFinancialMetricsLogger,
    FailFastValidationResult,
    RegimeValidationResult
)


class ComprehensiveFailFastValidationExample:
    """
    Example demonstrating comprehensive fail-fast validation across all important aspects.
    """
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        
        # Initialize enhanced financial logger with comprehensive fail-fast validation
        self.enhanced_logger = EnhancedFinancialMetricsLogger(
            fail_fast_enabled=True,
            regime_validation_enabled=True,
            min_regime_samples=100,
            max_regime_imbalance=0.8
        )
    
    async def demonstrate_comprehensive_validation(self):
        """Demonstrate comprehensive fail-fast validation across all aspects."""
        print("🚀 Comprehensive Fail-Fast Validation Demonstration")
        print("=" * 60)
        
        # Test scenarios for different validation categories
        test_scenarios = [
            ("Data Quality Issues", self._test_data_quality_validation),
            ("Performance Degradation", self._test_performance_validation),
            ("Model Quality Issues", self._test_model_quality_validation),
            ("Feature Quality Problems", self._test_feature_quality_validation),
            ("Execution Environment Issues", self._test_execution_environment_validation),
            ("Business Logic Violations", self._test_business_logic_validation),
            ("Regime Quality Issues", self._test_regime_quality_validation),
            ("Empty Running Detection", self._test_empty_running_detection),
            ("Comprehensive Validation", self._test_comprehensive_validation)
        ]
        
        for scenario_name, test_func in test_scenarios:
            print(f"\n🔍 Testing: {scenario_name}")
            print("-" * 40)
            await test_func()
        
        print("\n" + "=" * 60)
        print("✅ Comprehensive Fail-Fast Validation Demonstration Complete!")
    
    async def _test_data_quality_validation(self):
        """Test data quality validation."""
        print("Testing data quality validation...")
        
        # Test 1: Empty data
        empty_data = pd.DataFrame()
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=empty_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Empty data: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 2: High NaN ratio
        high_nan_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, np.nan, np.nan],
            'feature2': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'composite_cluster_id': ['regime_0', 'regime_1', 'regime_0', 'regime_1', 'regime_0']
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=high_nan_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  High NaN ratio: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 3: Constant columns
        constant_data = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],
            'feature2': [2, 2, 2, 2, 2],
            'composite_cluster_id': ['regime_0', 'regime_1', 'regime_0', 'regime_1', 'regime_0']
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=constant_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Constant columns: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 4: Good data quality
        good_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'composite_cluster_id': np.random.choice(['regime_0', 'regime_1', 'regime_2'], 1000)
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=good_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Good data quality: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        print(f"    Data quality score: {result.data_quality_score:.2f}")
    
    async def _test_performance_validation(self):
        """Test performance validation."""
        print("Testing performance validation...")
        
        # Test 1: Low model accuracy
        context = {
            'model_performance': {'accuracy': 0.3},
            'execution_time': 500
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Low model accuracy: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 2: Long execution time
        context = {
            'model_performance': {'accuracy': 0.8},
            'execution_time': 4000  # 1+ hours
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Long execution time: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
        
        # Test 3: Good performance
        context = {
            'model_performance': {'accuracy': 0.85},
            'execution_time': 300
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Good performance: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        print(f"    Performance score: {result.performance_score:.2f}")
    
    async def _test_model_quality_validation(self):
        """Test model quality validation."""
        print("Testing model quality validation...")
        
        # Test 1: Model did not converge
        context = {
            'model_convergence': False,
            'model_metrics': {'loss': 15.0}
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Model not converged: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 2: High model loss
        context = {
            'model_convergence': True,
            'model_metrics': {'loss': 12.0}
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  High model loss: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 3: Overfitting detection
        context = {
            'model_convergence': True,
            'model_metrics': {'loss': 2.0},
            'training_accuracy': 0.95,
            'validation_accuracy': 0.70
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Overfitting detection: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
        
        # Test 4: Good model quality
        context = {
            'model_convergence': True,
            'model_metrics': {'loss': 1.5},
            'training_accuracy': 0.85,
            'validation_accuracy': 0.82
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Good model quality: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        print(f"    Model quality score: {result.model_quality_score:.2f}")
    
    async def _test_feature_quality_validation(self):
        """Test feature quality validation."""
        print("Testing feature quality validation...")
        
        # Test 1: Low feature count
        low_features_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=low_features_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Low feature count: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
        
        # Test 2: High feature correlation
        high_corr_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'feature4': np.random.randn(100),
            'feature5': np.random.randn(100)
        })
        # Make feature2 highly correlated with feature1
        high_corr_data['feature2'] = high_corr_data['feature1'] + np.random.randn(100) * 0.1
        high_corr_data['feature3'] = high_corr_data['feature1'] + np.random.randn(100) * 0.1
        
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=high_corr_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  High feature correlation: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
        
        # Test 3: Good feature quality
        good_features_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'feature4': np.random.randn(1000),
            'feature5': np.random.randn(1000),
            'feature6': np.random.randn(1000),
            'composite_cluster_id': np.random.choice(['regime_0', 'regime_1', 'regime_2'], 1000)
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=good_features_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Good feature quality: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        print(f"    Feature quality score: {result.feature_quality_score:.2f}")
    
    async def _test_execution_environment_validation(self):
        """Test execution environment validation."""
        print("Testing execution environment validation...")
        
        # Test 1: High memory usage
        context = {
            'memory_usage_mb': 9000,  # 9GB
            'cpu_usage_percent': 85
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  High memory usage: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
        
        # Test 2: Low disk space
        context = {
            'memory_usage_mb': 4000,
            'cpu_usage_percent': 70,
            'disk_usage_percent': 95
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Low disk space: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 3: Execution errors
        context = {
            'memory_usage_mb': 4000,
            'cpu_usage_percent': 70,
            'errors': ['Error 1', 'Error 2', 'Error 3']
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Execution errors: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 4: Good execution environment
        context = {
            'memory_usage_mb': 3000,
            'cpu_usage_percent': 60,
            'disk_usage_percent': 70
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3]}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Good execution environment: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
    
    async def _test_business_logic_validation(self):
        """Test business logic validation."""
        print("Testing business logic validation...")
        
        # Test 1: Missing required columns
        missing_cols_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
            # Missing 'composite_cluster_id' for post-HMM step
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=missing_cols_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Missing required columns: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 2: Business rule violations
        context = {
            'business_rules': {
                'violations': ['Rule 1 violation', 'Rule 2 violation']
            }
        }
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=pd.DataFrame({'feature1': [1, 2, 3], 'composite_cluster_id': ['regime_0', 'regime_1', 'regime_0']}),
            step_name="Step09_HMM_Based_Training",
            additional_context=context
        )
        print(f"  Business rule violations: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 3: Negative prices
        negative_prices_data = pd.DataFrame({
            'price': [-100, 200, -50, 300, 150],
            'composite_cluster_id': ['regime_0', 'regime_1', 'regime_0', 'regime_1', 'regime_0']
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=negative_prices_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Negative prices: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.warnings:
            print(f"    Warnings: {result.warnings}")
        
        # Test 4: Good business logic
        good_business_data = pd.DataFrame({
            'price': [100, 200, 150, 300, 250],
            'volume': [1000, 2000, 1500, 3000, 2500],
            'composite_cluster_id': ['regime_0', 'regime_1', 'regime_0', 'regime_1', 'regime_0']
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=good_business_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Good business logic: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
    
    async def _test_regime_quality_validation(self):
        """Test regime quality validation."""
        print("Testing regime quality validation...")
        
        # Test 1: Missing regime column
        no_regime_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10]
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=no_regime_data,
            step_name="Step09_HMM_Based_Training",  # Post-HMM step
            additional_context={}
        )
        print(f"  Missing regime column: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 2: Insufficient regime diversity
        low_diversity_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'composite_cluster_id': ['regime_0', 'regime_0', 'regime_0', 'regime_0', 'regime_0']
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=low_diversity_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Insufficient regime diversity: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 3: Good regime quality
        good_regime_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'composite_cluster_id': np.random.choice(['regime_0', 'regime_1', 'regime_2'], 1000, p=[0.4, 0.35, 0.25])
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=good_regime_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Good regime quality: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
    
    async def _test_empty_running_detection(self):
        """Test empty running detection."""
        print("Testing empty running detection...")
        
        # Test 1: Empty data
        empty_data = pd.DataFrame()
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=empty_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Empty data: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 2: Too small dataset
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]  # Only 5 samples
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=small_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Too small dataset: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 3: No data variation
        no_variation_data = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],
            'feature2': [2, 2, 2, 2, 2]
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=no_variation_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  No data variation: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
        
        # Test 4: Good data (not empty running)
        good_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'composite_cluster_id': np.random.choice(['regime_0', 'regime_1', 'regime_2'], 1000)
        })
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=good_data,
            step_name="Step09_HMM_Based_Training",
            additional_context={}
        )
        print(f"  Good data (not empty running): {'❌ FAIL' if result.should_fail else '✅ PASS'}")
    
    async def _test_comprehensive_validation(self):
        """Test comprehensive validation with multiple issues."""
        print("Testing comprehensive validation...")
        
        # Test with multiple issues
        problematic_data = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],  # Constant column
            'feature2': [np.nan, np.nan, np.nan, np.nan, np.nan],  # All NaN
            'composite_cluster_id': ['regime_0', 'regime_0', 'regime_0', 'regime_0', 'regime_0']  # No diversity
        })
        
        problematic_context = {
            'model_performance': {'accuracy': 0.3},  # Low accuracy
            'model_convergence': False,  # Not converged
            'memory_usage_mb': 9000,  # High memory
            'errors': ['Error 1', 'Error 2'],  # Execution errors
            'business_rules': {'violations': ['Rule violation']}  # Business rule violation
        }
        
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=problematic_data,
            step_name="Step09_HMM_Based_Training",
            additional_context=problematic_context
        )
        
        print(f"  Multiple issues: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if result.should_fail:
            print(f"    Reason: {result.failure_reason}")
            print(f"    Critical issues: {len(result.critical_issues)}")
            print(f"    Warnings: {len(result.warnings)}")
            print(f"    Validation categories: {result.validation_categories}")
            print(f"    Recommendations: {len(result.recommendations)}")
        
        # Test with good comprehensive data
        good_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'feature4': np.random.randn(1000),
            'feature5': np.random.randn(1000),
            'composite_cluster_id': np.random.choice(['regime_0', 'regime_1', 'regime_2'], 1000, p=[0.4, 0.35, 0.25])
        })
        
        good_context = {
            'model_performance': {'accuracy': 0.85},
            'model_convergence': True,
            'model_metrics': {'loss': 1.5},
            'training_accuracy': 0.85,
            'validation_accuracy': 0.82,
            'memory_usage_mb': 3000,
            'cpu_usage_percent': 60,
            'execution_time': 300
        }
        
        result = self.enhanced_logger.validate_fail_fast_conditions(
            data=good_data,
            step_name="Step09_HMM_Based_Training",
            additional_context=good_context
        )
        
        print(f"  Good comprehensive data: {'❌ FAIL' if result.should_fail else '✅ PASS'}")
        if not result.should_fail:
            print(f"    Data quality score: {result.data_quality_score:.2f}")
            print(f"    Performance score: {result.performance_score:.2f}")
            print(f"    Model quality score: {result.model_quality_score:.2f}")
            print(f"    Feature quality score: {result.feature_quality_score:.2f}")
            print(f"    Validation categories: {result.validation_categories}")


async def main():
    """Main function to run the comprehensive fail-fast validation example."""
    example = ComprehensiveFailFastValidationExample('BTCUSDT', 'binance', '1h')
    await example.demonstrate_comprehensive_validation()


if __name__ == "__main__":
    asyncio.run(main())