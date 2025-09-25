"""
Comprehensive Tests for Unified Backtesting Framework

This module provides comprehensive tests for the unified backtesting framework
to ensure all components work correctly and integration is successful.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Any, Optional

# Import unified backtesting framework
try:
    from src.utils.common_backtesting import (
        BacktestingEngine,
        BacktestingConfig,
        BacktestingMode,
        MonteCarloEngine,
        MonteCarloConfig,
        PerformanceAttribution,
        PerformanceAttributionConfig,
        WalkForwardAnalyzer,
        WalkForwardConfig,
        BacktestingDataManager,
        DataManagerConfig,
        RiskAnalyzer,
        RiskAnalysisConfig,
        UnifiedBacktestingOrchestrator,
        OrchestratorConfig
    )
    UNIFIED_BACKTESTING_AVAILABLE = True
except ImportError:
    UNIFIED_BACKTESTING_AVAILABLE = False


class TestUnifiedBacktestingFramework:
    """Test suite for unified backtesting framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate synthetic OHLCV data
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, len(dates))))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        data['returns'] = data['close'].pct_change()
        return data.dropna()
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple test model."""
        class SimpleModel:
            def __init__(self):
                self.is_fitted = False
            
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Simple momentum strategy
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                return np.random.choice([-1, 0, 1], size=X.shape[0])
        
        return SimpleModel()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_backtesting_engine_initialization(self):
        """Test backtesting engine initialization."""
        config = BacktestingConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            mode=BacktestingMode.HISTORICAL
        )
        
        engine = BacktestingEngine(config)
        assert engine is not None
        assert engine.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_backtesting_engine_validation(self):
        """Test backtesting engine configuration validation."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            config = BacktestingConfig(
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2020, 1, 1),  # End before start
                mode=BacktestingMode.HISTORICAL
            )
            BacktestingEngine(config)
        
        # Test negative capital
        with pytest.raises(ValueError):
            config = BacktestingConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=-1000,  # Negative capital
                mode=BacktestingMode.HISTORICAL
            )
            BacktestingEngine(config)
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_backtesting_engine_run(self, sample_data, sample_model):
        """Test backtesting engine execution."""
        config = BacktestingConfig(
            start_date=sample_data.index.min(),
            end_date=sample_data.index.max(),
            mode=BacktestingMode.HISTORICAL,
            save_results=False
        )
        
        engine = BacktestingEngine(config)
        result = engine.run_backtest(sample_model, sample_data)
        
        assert result is not None
        assert result.config == config
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.execution_time > 0
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_data_manager_initialization(self):
        """Test data manager initialization."""
        config = DataManagerConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        manager = BacktestingDataManager(config)
        assert manager is not None
        assert manager.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_data_manager_data_processing(self, sample_data, temp_dir):
        """Test data manager data processing."""
        config = DataManagerConfig(
            cache_directory=temp_dir,
            enable_validation=True,
            enable_preprocessing=True
        )
        
        manager = BacktestingDataManager(config)
        processed_data = manager._process_data(sample_data)
        
        assert processed_data is not None
        assert len(processed_data) > 0
        assert 'returns' in processed_data.columns
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_monte_carlo_engine_initialization(self):
        """Test Monte Carlo engine initialization."""
        config = MonteCarloConfig(
            n_simulations=100,
            confidence_level=0.95
        )
        
        engine = MonteCarloEngine(config)
        assert engine is not None
        assert engine.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_monte_carlo_engine_simulation(self, sample_data, sample_model):
        """Test Monte Carlo engine simulation."""
        config = MonteCarloConfig(
            n_simulations=50,  # Reduced for testing
            confidence_level=0.95
        )
        
        engine = MonteCarloEngine(config)
        result = engine.run_simulation(sample_model, sample_data)
        
        assert result is not None
        assert result.n_simulations == config.n_simulations
        assert result.mean_return is not None
        assert result.var_95 is not None
        assert result.probability_of_loss is not None
        assert result.execution_time > 0
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_performance_attribution_initialization(self):
        """Test performance attribution initialization."""
        config = PerformanceAttributionConfig(
            benchmark_symbol="SPY",
            risk_free_rate=0.02
        )
        
        attribution = PerformanceAttribution(config)
        assert attribution is not None
        assert attribution.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_performance_attribution_analysis(self, sample_data):
        """Test performance attribution analysis."""
        config = PerformanceAttributionConfig()
        
        attribution = PerformanceAttribution(config)
        
        # Create sample returns
        returns = sample_data['returns'].head(100)
        benchmark_returns = returns * 0.8  # Simulate benchmark
        
        metrics = attribution.analyze_performance(returns, benchmark_returns)
        
        assert metrics is not None
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.beta is not None
        assert metrics.alpha is not None
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_walk_forward_analyzer_initialization(self):
        """Test walk-forward analyzer initialization."""
        config = WalkForwardConfig(
            train_period_days=252,
            test_period_days=63,
            step_size_days=21
        )
        
        analyzer = WalkForwardAnalyzer(config)
        assert analyzer is not None
        assert analyzer.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_walk_forward_analyzer_analysis(self, sample_data, sample_model):
        """Test walk-forward analyzer execution."""
        config = WalkForwardConfig(
            train_period_days=100,  # Reduced for testing
            test_period_days=30,
            step_size_days=10,
            min_train_periods=50,
            min_test_periods=15
        )
        
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.analyze(sample_model, sample_data)
        
        assert result is not None
        assert result.n_periods > 0
        assert result.total_return is not None
        assert result.performance_stability is not None
        assert result.parameter_stability is not None
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_risk_analyzer_initialization(self):
        """Test risk analyzer initialization."""
        config = RiskAnalysisConfig(
            confidence_level=0.95,
            enable_var=True,
            enable_cvar=True
        )
        
        analyzer = RiskAnalyzer(config)
        assert analyzer is not None
        assert analyzer.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_risk_analyzer_analysis(self, sample_data):
        """Test risk analyzer execution."""
        config = RiskAnalysisConfig()
        
        analyzer = RiskAnalyzer(config)
        
        returns = sample_data['returns'].head(100)
        risk_metrics = analyzer.analyze(returns)
        
        assert risk_metrics is not None
        assert 'var_95' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'realized_volatility' in risk_metrics
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_unified_orchestrator_initialization(self):
        """Test unified orchestrator initialization."""
        config = OrchestratorConfig(
            enable_monte_carlo=True,
            enable_walk_forward=True,
            enable_performance_attribution=True,
            enable_risk_analysis=True
        )
        
        orchestrator = UnifiedBacktestingOrchestrator(config)
        assert orchestrator is not None
        assert orchestrator.config == config
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_unified_orchestrator_comprehensive_analysis(self, sample_data, sample_model, temp_dir):
        """Test unified orchestrator comprehensive analysis."""
        config = OrchestratorConfig(
            enable_monte_carlo=True,
            enable_walk_forward=False,  # Disabled for faster testing
            enable_performance_attribution=True,
            enable_risk_analysis=True,
            save_all_results=True,
            results_directory=temp_dir
        )
        
        orchestrator = UnifiedBacktestingOrchestrator(config)
        result = orchestrator.run_comprehensive_analysis(sample_model, sample_data)
        
        assert result is not None
        assert result.backtesting_result is not None
        assert result.monte_carlo_result is not None
        assert result.performance_metrics is not None
        assert result.risk_metrics is not None
        assert result.overall_score is not None
        assert result.execution_time > 0
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_integration_tas_system(self, sample_data, sample_model):
        """Test integration with TAS system."""
        try:
            from src.training.steps.market_analysis.tas_regime.backtesting.unified_backtesting_integration import (
                TASUnifiedBacktestingIntegration,
                TASBacktestingConfig
            )
            
            config = TASBacktestingConfig()
            integration = TASUnifiedBacktestingIntegration(config)
            
            result = integration.run_tas_backtest(sample_model, sample_data)
            
            assert result is not None
            assert hasattr(result, 'tas_metrics')
            
        except ImportError:
            pytest.skip("TAS integration not available")
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_integration_nas_system(self, sample_data, sample_model):
        """Test integration with NAS system."""
        try:
            from src.training.steps.market_analysis.nas_regime.backtesting.unified_backtesting_integration import (
                NASUnifiedBacktestingIntegration,
                NASBacktestingConfig
            )
            
            config = NASBacktestingConfig()
            integration = NASUnifiedBacktestingIntegration(config)
            
            result = integration.run_nas_backtest(sample_model, sample_data)
            
            assert result is not None
            assert hasattr(result, 'nas_metrics')
            
        except ImportError:
            pytest.skip("NAS integration not available")
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_error_handling(self, sample_data):
        """Test error handling in unified framework."""
        # Test with invalid model
        config = BacktestingConfig(save_results=False)
        engine = BacktestingEngine(config)
        
        with pytest.raises(Exception):
            engine.run_backtest(None, sample_data)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            engine.run_backtest(sample_model, empty_data)
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_performance_benchmarks(self, sample_data, sample_model):
        """Test performance benchmarks."""
        import time
        
        config = BacktestingConfig(save_results=False)
        engine = BacktestingEngine(config)
        
        start_time = time.time()
        result = engine.run_backtest(sample_model, sample_data.head(1000))
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30  # 30 seconds max
        assert result.execution_time > 0
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_memory_usage(self, sample_data, sample_model):
        """Test memory usage optimization."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        config = BacktestingConfig(save_results=False, enable_memory_optimization=True)
        engine = BacktestingEngine(config)
        
        # Run multiple backtests to test memory management
        for _ in range(5):
            result = engine.run_backtest(sample_model, sample_data.head(500))
            assert result is not None
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100


class TestUnifiedBacktestingIntegration:
    """Test suite for integration with existing systems."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, len(dates))))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        data['returns'] = data['close'].pct_change()
        return data.dropna()
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_backward_compatibility(self, sample_data):
        """Test backward compatibility with existing systems."""
        # Test that existing systems can still work
        from src.utils.common_backtesting import create_backtesting_engine, run_quick_backtest
        
        # Test quick backtest function
        class SimpleModel:
            def predict(self, X):
                return np.random.choice([-1, 0, 1], size=len(X))
        
        model = SimpleModel()
        result = run_quick_backtest(model, sample_data.head(100))
        
        assert result is not None
        assert result.total_return is not None
    
    @pytest.mark.skipif(not UNIFIED_BACKTESTING_AVAILABLE, reason="Unified backtesting framework not available")
    def test_configuration_compatibility(self):
        """Test configuration compatibility."""
        from src.utils.common_backtesting import create_quick_config, create_full_config
        
        quick_config = create_quick_config()
        assert quick_config is not None
        assert isinstance(quick_config, OrchestratorConfig)
        
        full_config = create_full_config()
        assert full_config is not None
        assert isinstance(full_config, OrchestratorConfig)
        assert full_config.enable_monte_carlo
        assert full_config.enable_walk_forward


if __name__ == "__main__":
    pytest.main([__file__])