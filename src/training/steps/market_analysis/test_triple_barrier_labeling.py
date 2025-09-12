"""
Comprehensive Test Suite for MARKET_ANALYSIS Triple Barrier Labeling

This module provides comprehensive tests for all triple barrier labeling components
including unit tests, integration tests, and performance tests.

Key Features:
- Unit tests for individual components
- Integration tests for full pipeline
- Performance benchmarks
- Edge case testing
- Validation testing
"""

import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil

# Import the triple barrier components
from .triple_barrier_labeling import (
    MarketAnalysisTripleBarrierLabeling,
    TripleBarrierConfig,
    create_triple_barrier_labeler,
    apply_triple_barrier_labeling,
    benchmark_triple_barrier_methods
)
from .regime_aware_triple_barrier_optimizer import (
    RegimeAwareTripleBarrierOptimizer,
    RegimeBarrierParams,
    RegimePerformanceMetrics,
    optimize_regime_barriers,
    apply_optimized_regime_labeling
)
from .triple_barrier_validator import (
    TripleBarrierValidator,
    ValidationReport,
    validate_triple_barrier_implementation,
    quick_validate_triple_barrier
)
from .enhanced_market_analysis_with_triple_barrier import (
    EnhancedMarketAnalysisWithTripleBarrier,
    MarketAnalysisTripleBarrierConfig,
    run_enhanced_market_analysis_with_triple_barrier,
    quick_triple_barrier_analysis
)

class TestDataGenerator:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_basic_market_data(n_samples: int = 1000, start_price: float = 100.0) -> pd.DataFrame:
        """Create basic market data for testing."""
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        # Generate realistic price data with some trend and volatility
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, n_samples)  # Small positive drift with 1% volatility
        prices = [start_price]
        
        for i in range(1, n_samples):
            prices.append(prices[-1] * (1 + returns[i]))
        
        prices = np.array(prices)
        
        # Generate OHLC data
        data = pd.DataFrame({
            'open': prices * np.random.uniform(0.999, 1.001, n_samples),
            'high': prices * np.random.uniform(1.001, 1.005, n_samples),
            'low': prices * np.random.uniform(0.995, 0.999, n_samples),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_samples)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    @staticmethod
    def create_regime_market_data(n_samples: int = 2000) -> pd.DataFrame:
        """Create market data with regime information."""
        data = TestDataGenerator.create_basic_market_data(n_samples)
        
        # Add regime information
        np.random.seed(42)
        regime_changes = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
        
        # Add some persistence to regimes
        for i in range(1, n_samples):
            if np.random.random() < 0.95:  # 95% chance to stay in same regime
                regime_changes[i] = regime_changes[i-1]
        
        data['hmm_regime'] = regime_changes
        
        # Adjust volatility based on regime
        for regime in [0, 1, 2]:
            regime_mask = data['hmm_regime'] == regime
            if regime == 0:  # Low volatility regime
                data.loc[regime_mask, 'high'] *= 1.001
                data.loc[regime_mask, 'low'] *= 0.999
            elif regime == 1:  # Medium volatility regime
                data.loc[regime_mask, 'high'] *= 1.003
                data.loc[regime_mask, 'low'] *= 0.997
            else:  # High volatility regime
                data.loc[regime_mask, 'high'] *= 1.005
                data.loc[regime_mask, 'low'] *= 0.995
        
        return data
    
    @staticmethod
    def create_problematic_data() -> pd.DataFrame:
        """Create data with various issues for testing edge cases."""
        data = TestDataGenerator.create_basic_market_data(100)
        
        # Add some problematic data
        data.iloc[10, 0] = np.nan  # Missing open price
        data.iloc[20, 1] = 0  # Zero high price
        data.iloc[30, 2] = -1  # Negative low price
        data.iloc[40, 3] = np.inf  # Infinite close price
        
        return data

class TestTripleBarrierLabeling:
    """Test suite for MarketAnalysisTripleBarrierLabeling."""
    
    def test_basic_initialization(self):
        """Test basic initialization of triple barrier labeler."""
        config = TripleBarrierConfig(
            profit_take_multiplier=0.002,
            stop_loss_multiplier=0.001,
            time_barrier_minutes=30,
            max_lookahead=100
        )
        
        labeler = MarketAnalysisTripleBarrierLabeling(config)
        
        assert labeler.config.profit_take_multiplier == 0.002
        assert labeler.config.stop_loss_multiplier == 0.001
        assert labeler.config.time_barrier_minutes == 30
        assert labeler.config.max_lookahead == 100
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid parameters
        with pytest.raises(Exception):
            config = TripleBarrierConfig(profit_take_multiplier=-0.001)
            MarketAnalysisTripleBarrierLabeling(config)
        
        with pytest.raises(Exception):
            config = TripleBarrierConfig(stop_loss_multiplier=0.1)
            MarketAnalysisTripleBarrierLabeling(config)
    
    def test_basic_labeling(self):
        """Test basic triple barrier labeling."""
        data = TestDataGenerator.create_basic_market_data(1000)
        
        config = TripleBarrierConfig(
            profit_take_multiplier=0.002,
            stop_loss_multiplier=0.001,
            binary_classification=True
        )
        
        labeler = MarketAnalysisTripleBarrierLabeling(config)
        labeled_data = labeler.apply_triple_barrier_labeling(data)
        
        # Basic checks
        assert len(labeled_data) > 0
        assert 'label' in labeled_data.columns
        assert 'potential_profit_pct' in labeled_data.columns
        assert 'transaction_cost' in labeled_data.columns
        
        # Check label values
        labels = labeled_data['label'].values
        assert all(label in [-1, 1] for label in labels)  # Binary classification
        
        # Check profit tracking
        profits = labeled_data['potential_profit_pct'].values
        assert not np.any(np.isnan(profits))
        assert not np.any(np.isinf(profits))
    
    def test_regime_aware_labeling(self):
        """Test regime-aware labeling."""
        data = TestDataGenerator.create_regime_market_data(1000)
        
        config = TripleBarrierConfig(
            profit_take_multiplier=0.002,
            stop_loss_multiplier=0.001,
            regime_aware=True,
            regime_column='hmm_regime'
        )
        
        labeler = MarketAnalysisTripleBarrierLabeling(config)
        labeled_data = labeler.apply_triple_barrier_labeling(data)
        
        # Basic checks
        assert len(labeled_data) > 0
        assert 'label' in labeled_data.columns
        assert 'labeling_method' in labeled_data.columns
        assert labeled_data['labeling_method'].iloc[0] == 'regime_aware'
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with insufficient data
        data = TestDataGenerator.create_basic_market_data(10)
        
        config = TripleBarrierConfig()
        labeler = MarketAnalysisTripleBarrierLabeling(config)
        
        # Should handle small datasets gracefully
        labeled_data = labeler.apply_triple_barrier_labeling(data)
        assert isinstance(labeled_data, pd.DataFrame)
        
        # Test with problematic data
        problematic_data = TestDataGenerator.create_problematic_data()
        
        # Should handle problematic data gracefully
        labeled_data = labeler.apply_triple_barrier_labeling(problematic_data)
        assert isinstance(labeled_data, pd.DataFrame)
    
    def test_column_name_standardization(self):
        """Test column name standardization."""
        data = TestDataGenerator.create_basic_market_data(100)
        
        # Rename columns to test standardization
        data = data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        })
        
        config = TripleBarrierConfig()
        labeler = MarketAnalysisTripleBarrierLabeling(config)
        labeled_data = labeler.apply_triple_barrier_labeling(data)
        
        # Should handle renamed columns
        assert len(labeled_data) > 0
        assert 'label' in labeled_data.columns

class TestRegimeAwareTripleBarrierOptimizer:
    """Test suite for RegimeAwareTripleBarrierOptimizer."""
    
    def test_basic_initialization(self):
        """Test basic initialization of regime optimizer."""
        optimizer = RegimeAwareTripleBarrierOptimizer()
        
        assert optimizer.optimization_params is not None
        assert 'profit_take_range' in optimizer.optimization_params
        assert 'stop_loss_range' in optimizer.optimization_params
    
    def test_regime_parameter_optimization(self):
        """Test regime parameter optimization."""
        data = TestDataGenerator.create_regime_market_data(500)
        
        # Use smaller dataset and simpler config for testing
        config = {
            'profit_take_range': (0.001, 0.005),
            'stop_loss_range': (0.0005, 0.003),
            'time_barrier_range': (20, 40),
            'max_lookahead_range': (50, 150),
            'max_iterations': 10  # Reduce for testing
        }
        
        optimizer = RegimeAwareTripleBarrierOptimizer(config)
        regime_parameters = optimizer.optimize_regime_parameters(data)
        
        # Check that parameters were optimized
        assert len(regime_parameters) > 0
        for regime, params in regime_parameters.items():
            assert isinstance(params, RegimeBarrierParams)
            assert params.profit_take_multiplier > 0
            assert params.stop_loss_multiplier > 0
    
    def test_optimized_labeling(self):
        """Test optimized regime labeling."""
        data = TestDataGenerator.create_regime_market_data(500)
        
        optimizer = RegimeAwareTripleBarrierOptimizer()
        optimizer.optimize_regime_parameters(data)
        
        labeled_data = optimizer.apply_optimized_labeling(data)
        
        # Basic checks
        assert len(labeled_data) > 0
        assert 'label' in labeled_data.columns
        assert 'labeling_method' in labeled_data.columns
        assert labeled_data['labeling_method'].iloc[0] == 'regime_optimized'
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create mock labeled data
        data = pd.DataFrame({
            'label': [1, -1, 1, -1, 1],
            'potential_profit_pct': [0.002, -0.001, 0.003, -0.002, 0.001]
        })
        
        optimizer = RegimeAwareTripleBarrierOptimizer()
        
        # Test performance calculation
        metrics = optimizer._calculate_regime_performance(
            data, 
            RegimeBarrierParams(), 
            0, 
            'regime'
        )
        
        assert isinstance(metrics, RegimePerformanceMetrics)
        assert metrics.total_samples == 5
        assert metrics.win_rate >= 0
        assert metrics.win_rate <= 1

class TestTripleBarrierValidator:
    """Test suite for TripleBarrierValidator."""
    
    def test_basic_initialization(self):
        """Test basic initialization of validator."""
        validator = TripleBarrierValidator()
        
        assert validator.validation_params is not None
        assert 'min_data_points' in validator.validation_params
        assert 'max_missing_ratio' in validator.validation_params
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        validator = TripleBarrierValidator()
        
        # Test with good data
        good_data = TestDataGenerator.create_basic_market_data(1000)
        result = validator._validate_data_quality(good_data)
        
        assert result.passed
        assert result.score > 0.8
        
        # Test with problematic data
        bad_data = TestDataGenerator.create_problematic_data()
        result = validator._validate_data_quality(bad_data)
        
        assert not result.passed
        assert result.score < 0.8
    
    def test_labeling_quality_validation(self):
        """Test labeling quality validation."""
        validator = TripleBarrierValidator()
        
        # Create test labeled data
        data = TestDataGenerator.create_basic_market_data(1000)
        labeled_data = data.copy()
        
        # Test with balanced labels
        labeled_data['label'] = np.random.choice([-1, 1], 1000, p=[0.5, 0.5])
        labeled_data['potential_profit_pct'] = np.random.normal(0.001, 0.005, 1000)
        
        result = validator._validate_labeling_quality(data, labeled_data)
        
        assert result.passed
        assert result.score > 0.7
    
    def test_performance_validation(self):
        """Test performance validation."""
        validator = TripleBarrierValidator()
        
        # Create test labeled data with good performance
        labeled_data = pd.DataFrame({
            'label': [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            'net_profit_pct': [0.002, -0.001, 0.003, -0.001, 0.002, -0.001, 0.003, -0.001, 0.002, -0.001]
        })
        
        result = validator._validate_performance(labeled_data)
        
        assert result.passed
        assert result.score > 0.7
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation."""
        validator = TripleBarrierValidator()
        
        # Create test data
        data = TestDataGenerator.create_basic_market_data(1000)
        labeled_data = data.copy()
        labeled_data['label'] = np.random.choice([-1, 1], 1000, p=[0.5, 0.5])
        labeled_data['potential_profit_pct'] = np.random.normal(0.001, 0.005, 1000)
        labeled_data['transaction_cost'] = 0.0008
        
        report = validator.validate_triple_barrier_implementation(data, labeled_data)
        
        assert isinstance(report, ValidationReport)
        assert report.total_checks > 0
        assert report.overall_score >= 0
        assert report.overall_score <= 1

class TestEnhancedMarketAnalysisWithTripleBarrier:
    """Test suite for EnhancedMarketAnalysisWithTripleBarrier."""
    
    def test_basic_initialization(self):
        """Test basic initialization of enhanced pipeline."""
        config = MarketAnalysisTripleBarrierConfig()
        pipeline = EnhancedMarketAnalysisWithTripleBarrier(config)
        
        assert pipeline.config is not None
        assert pipeline.triple_barrier_labeler is not None
    
    def test_full_pipeline_execution(self):
        """Test full pipeline execution."""
        data = TestDataGenerator.create_regime_market_data(500)
        
        config = MarketAnalysisTripleBarrierConfig(
            optimize_regime_parameters=False,  # Skip for faster testing
            enable_validation=True,
            save_intermediate_results=False
        )
        
        pipeline = EnhancedMarketAnalysisWithTripleBarrier(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = pipeline.run_market_analysis_with_triple_barrier(
                data, 'TEST', 'TEST', '1m', temp_dir
            )
            
            # Check results structure
            assert 'symbol' in results
            assert 'exchange' in results
            assert 'timeframe' in results
            assert 'triple_barrier_labeling' in results
            assert 'performance_metrics' in results
            
            # Check labeling results
            labeling_result = results['triple_barrier_labeling']
            assert labeling_result['success']
            assert 'labeled_data' in labeling_result
            
            # Check performance metrics
            performance = results['performance_metrics']
            assert 'total_trades' in performance
            assert 'win_rate' in performance
            assert 'sharpe_ratio' in performance
    
    def test_quick_analysis(self):
        """Test quick analysis function."""
        data = TestDataGenerator.create_regime_market_data(200)
        
        labeled_data = quick_triple_barrier_analysis(data)
        
        assert len(labeled_data) > 0
        assert 'label' in labeled_data.columns

class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks."""
    
    def test_triple_barrier_benchmark(self):
        """Test triple barrier benchmark."""
        data = TestDataGenerator.create_basic_market_data(1000)
        
        benchmark_results = benchmark_triple_barrier_methods(data)
        
        assert 'standard_time' in benchmark_results
        assert 'regime_aware_time' in benchmark_results
        assert 'data_size' in benchmark_results
        assert benchmark_results['data_size'] == 1000
    
    def test_performance_with_different_data_sizes(self):
        """Test performance with different data sizes."""
        data_sizes = [100, 500, 1000, 2000]
        
        for size in data_sizes:
            data = TestDataGenerator.create_basic_market_data(size)
            
            start_time = time.time()
            labeled_data = apply_triple_barrier_labeling(data)
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert execution_time < 10.0  # 10 seconds max
            assert len(labeled_data) > 0

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test data
        data = TestDataGenerator.create_regime_market_data(1000)
        
        # Test the complete workflow
        results = run_enhanced_market_analysis_with_triple_barrier(
            data, 'ETHUSDT', 'BINANCE', '1m'
        )
        
        # Verify all components worked
        assert 'triple_barrier_labeling' in results
        assert 'performance_metrics' in results
        assert 'execution_time' in results
        
        # Check that labeling was successful
        labeling_result = results['triple_barrier_labeling']
        assert labeling_result['success']
        assert labeling_result['total_samples'] > 0
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        data = TestDataGenerator.create_basic_market_data(500)
        
        # Test create_triple_barrier_labeler
        labeler = create_triple_barrier_labeler()
        assert isinstance(labeler, MarketAnalysisTripleBarrierLabeling)
        
        # Test apply_triple_barrier_labeling
        labeled_data = apply_triple_barrier_labeling(data)
        assert len(labeled_data) > 0
        assert 'label' in labeled_data.columns
        
        # Test validate_triple_barrier_implementation
        report = validate_triple_barrier_implementation(data, labeled_data)
        assert isinstance(report, ValidationReport)
        
        # Test quick_validate_triple_barrier
        is_valid = quick_validate_triple_barrier(data, labeled_data)
        assert isinstance(is_valid, bool)

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Test with empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):
            apply_triple_barrier_labeling(empty_data)
    
    def test_missing_columns(self):
        """Test with missing required columns."""
        data = pd.DataFrame({'price': [100, 101, 102]})
        
        with pytest.raises(Exception):
            apply_triple_barrier_labeling(data)
    
    def test_invalid_parameters(self):
        """Test with invalid parameters."""
        data = TestDataGenerator.create_basic_market_data(100)
        
        with pytest.raises(Exception):
            apply_triple_barrier_labeling(
                data, 
                profit_take_multiplier=-0.001
            )
    
    def test_extreme_values(self):
        """Test with extreme values."""
        data = TestDataGenerator.create_basic_market_data(100)
        
        # Add extreme values
        data.iloc[50, 3] = 1e10  # Very large price
        data.iloc[51, 3] = 1e-10  # Very small price
        
        # Should handle gracefully
        labeled_data = apply_triple_barrier_labeling(data)
        assert isinstance(labeled_data, pd.DataFrame)

# Performance tests
class TestPerformance:
    """Performance tests for the triple barrier system."""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        data = TestDataGenerator.create_basic_market_data(10000)
        
        start_time = time.time()
        labeled_data = apply_triple_barrier_labeling(data)
        execution_time = time.time() - start_time
        
        # Should handle large datasets efficiently
        assert execution_time < 30.0  # 30 seconds max
        assert len(labeled_data) > 0
    
    def test_memory_usage(self):
        """Test memory usage with large dataset."""
        data = TestDataGenerator.create_basic_market_data(5000)
        
        # Should not cause memory issues
        labeled_data = apply_triple_barrier_labeling(data)
        
        # Check that memory usage is reasonable
        memory_usage = labeled_data.memory_usage(deep=True).sum()
        assert memory_usage < 100 * 1024 * 1024  # Less than 100MB

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])