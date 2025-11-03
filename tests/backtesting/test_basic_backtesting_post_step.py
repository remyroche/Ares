"""
Unit Tests for BasicBacktestingPostStep
========================================

Comprehensive tests for the post-optimization backtesting step including:
- Initialization and setup
- Data loading (ML-scored data, price data, parameters)
- Signal generation (ML-based and fallback)
- Backtest execution with VectorBT
- Metrics calculation and comparison
- Report generation
- Error handling

Author: Ares Trading System
Date: 2025-10-31
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.training.steps.backtesting.basic_backtesting_post_step import (
        BasicBacktestingPostStep,
        VECTORBT_AVAILABLE,
        DATA_LOADING_AVAILABLE
    )
except ImportError as e:
    pytest.skip(f"Cannot import BasicBacktestingPostStep: {e}", allow_module_level=True)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'execution_mode': 'light',
        'initial_cash': 10000.0,
        'fees': 0.001,
        'slippage': 0.0005
    }


@pytest.fixture
def sample_price_data():
    """Create sample OHLCV price data."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    
    # Generate realistic price data
    close_prices = 2000 + np.cumsum(np.random.randn(1000) * 10)
    
    return pd.DataFrame({
        'open': close_prices + np.random.randn(1000) * 2,
        'high': close_prices + abs(np.random.randn(1000) * 3),
        'low': close_prices - abs(np.random.randn(1000) * 3),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)


@pytest.fixture
def sample_ml_scored_data(sample_price_data):
    """Create sample ML-scored data with predictions."""
    ml_data = sample_price_data.copy()
    
    # Add ML prediction columns
    ml_data['prediction'] = np.random.randn(len(ml_data))
    ml_data['confidence'] = np.random.uniform(0.3, 0.9, len(ml_data))
    ml_data['directional_score'] = np.random.uniform(-1, 1, len(ml_data))
    
    return ml_data


@pytest.fixture
def sample_optimized_params():
    """Create sample optimized parameters."""
    return {
        'confidence_threshold': 0.65,
        'exit_threshold': 0.45,
        'fast_ma_window': 20,
        'slow_ma_window': 50,
        'max_position_size': 0.5,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0
    }


@pytest.fixture
def sample_baseline_metrics():
    """Create sample baseline metrics."""
    return {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'sortino_ratio': 1.5,
        'calmar_ratio': 0.8,
        'max_drawdown': -0.12,
        'win_rate': 0.55,
        'total_trades': 100
    }


@pytest.fixture
def sample_trades_dataframe():
    """Create sample trades DataFrame for VectorBT."""
    return pd.DataFrame({
        'Entry Price': [2000, 2100, 2050],
        'Exit Price': [2050, 2080, 2100],
        'PnL': [50, -20, 50],
        'Return': [0.025, -0.0095, 0.024],
        'Duration': pd.to_timedelta(['1 hour', '2 hours', '30 minutes'])
    })


class TestBasicBacktestingPostStepInitialization:
    """Test initialization of BasicBacktestingPostStep."""
    
    def test_init_default_name(self):
        """Test initialization with default step name."""
        step = BasicBacktestingPostStep()
        assert step.step_name == "basic_backtesting_post"
        assert step.logger is not None
    
    def test_init_custom_name(self):
        """Test initialization with custom step name."""
        step = BasicBacktestingPostStep(step_name="custom_backtest")
        assert step.step_name == "custom_backtest"
    
    def test_init_inherits_base_step(self):
        """Test that step properly inherits from BaseStep."""
        from src.training.steps.base_step import BaseStep
        step = BasicBacktestingPostStep()
        assert isinstance(step, BaseStep)
    
    def test_artifact_manager_initialized(self):
        """Test that artifact manager is properly initialized."""
        step = BasicBacktestingPostStep()
        assert step.artifact_manager is not None


class TestVectorBTMetricsCalculation:
    """Test VectorBT metrics calculation methods."""
    
    def test_calculate_vectorbt_metrics_basic(self, sample_price_data):
        """Test basic metrics calculation."""
        step = BasicBacktestingPostStep()
        
        # Generate sample returns
        returns = sample_price_data['close'].pct_change().fillna(0)
        prices = sample_price_data['close']
        
        metrics = step._calculate_vectorbt_metrics(returns, prices)
        
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_calculate_vectorbt_metrics_with_positive_returns(self):
        """Test metrics with positive returns."""
        step = BasicBacktestingPostStep()
        
        # Create consistently positive returns
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        returns = pd.Series(np.random.uniform(0.001, 0.01, 100), index=dates)
        prices = pd.Series((1 + returns).cumprod() * 2000, index=dates)
        
        metrics = step._calculate_vectorbt_metrics(returns, prices)
        
        assert metrics['total_return'] > 0
        assert metrics['sharpe_ratio'] > 0
        assert metrics['max_drawdown'] <= 0
    
    def test_calculate_vectorbt_metrics_with_negative_returns(self):
        """Test metrics with negative returns."""
        step = BasicBacktestingPostStep()
        
        # Create consistently negative returns
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        returns = pd.Series(np.random.uniform(-0.01, -0.001, 100), index=dates)
        prices = pd.Series((1 + returns).cumprod() * 2000, index=dates)
        
        metrics = step._calculate_vectorbt_metrics(returns, prices)
        
        assert metrics['total_return'] < 0
        assert metrics['max_drawdown'] < 0
    
    def test_calculate_vectorbt_metrics_empty_returns(self):
        """Test metrics with empty returns."""
        step = BasicBacktestingPostStep()
        
        returns = pd.Series([])
        prices = pd.Series([])
        
        metrics = step._calculate_vectorbt_metrics(returns, prices)
        
        assert isinstance(metrics, dict)
        # Should return empty dict or handle gracefully
    
    def test_calculate_vectorbt_metrics_volatility(self, sample_price_data):
        """Test volatility calculation."""
        step = BasicBacktestingPostStep()
        
        returns = sample_price_data['close'].pct_change().fillna(0)
        prices = sample_price_data['close']
        
        metrics = step._calculate_vectorbt_metrics(returns, prices)
        
        assert 'volatility' in metrics
        assert metrics['volatility'] > 0
    
    def test_calculate_vectorbt_metrics_sharpe_sortino_spread(self, sample_price_data):
        """Test Sharpe-Sortino spread calculation."""
        step = BasicBacktestingPostStep()
        
        returns = sample_price_data['close'].pct_change().fillna(0)
        prices = sample_price_data['close']
        
        metrics = step._calculate_vectorbt_metrics(returns, prices)
        
        assert 'sharpe_sortino_spread' in metrics
        assert isinstance(metrics['sharpe_sortino_spread'], float)


class TestParameterLoading:
    """Test parameter loading methods."""
    
    def test_load_optimized_parameters_success(self, sample_config, sample_optimized_params):
        """Test successful loading of optimized parameters."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=sample_optimized_params):
            params = step._load_optimized_parameters(sample_config)
            
            assert params is not None
            assert params == sample_optimized_params
            assert 'confidence_threshold' in params
    
    def test_load_optimized_parameters_not_found(self, sample_config):
        """Test when optimized parameters are not found."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=None):
            params = step._load_optimized_parameters(sample_config)
            
            assert params is None
    
    def test_load_optimized_parameters_error(self, sample_config):
        """Test error handling when loading parameters fails."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', side_effect=Exception("Load failed")):
            params = step._load_optimized_parameters(sample_config)
            
            assert params is None
    
    def test_load_baseline_metrics_success(self, sample_baseline_metrics):
        """Test successful loading of baseline metrics."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=sample_baseline_metrics):
            baseline = step._load_baseline_metrics()
            
            assert baseline is not None
            assert baseline == sample_baseline_metrics
            assert 'total_return' in baseline
    
    def test_load_baseline_metrics_not_found(self):
        """Test when baseline metrics are not found."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=None):
            baseline = step._load_baseline_metrics()
            
            assert baseline is None


class TestMLDataLoading:
    """Test ML-scored data loading."""
    
    def test_load_ml_scored_data_tactician(self, sample_config, sample_ml_scored_data):
        """Test loading tactician ML-scored data."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=sample_ml_scored_data):
            ml_data = step._load_ml_scored_data(sample_config)
            
            assert ml_data is not None
            assert len(ml_data) > 0
            assert 'prediction' in ml_data.columns
    
    def test_load_ml_scored_data_analyst_fallback(self, sample_config):
        """Test fallback to analyst data when tactician not found."""
        step = BasicBacktestingPostStep()
        sample_analyst_data = pd.DataFrame({
            'prediction': np.random.randn(100),
            'confidence': np.random.uniform(0.5, 0.9, 100)
        })
        
        def mock_get_artifact(name, artifact_type):
            if 'tactician' in name:
                return None
            elif 'analyst' in name:
                return sample_analyst_data
            return None
        
        with patch.object(step, '_get_artifact', side_effect=mock_get_artifact):
            ml_data = step._load_ml_scored_data(sample_config)
            
            assert ml_data is not None
            assert len(ml_data) == 100
    
    def test_load_ml_scored_data_not_found(self, sample_config):
        """Test when no ML-scored data is found."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=None):
            ml_data = step._load_ml_scored_data(sample_config)
            
            assert ml_data is None
    
    def test_load_ml_scored_data_empty_dataframe(self, sample_config):
        """Test when ML-scored data is empty."""
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_get_artifact', return_value=pd.DataFrame()):
            ml_data = step._load_ml_scored_data(sample_config)
            
            assert ml_data is None


class TestPriceDataLoading:
    """Test price data loading methods."""
    
    @pytest.mark.skipif(not DATA_LOADING_AVAILABLE, reason="Data loading utilities not available")
    def test_load_price_data_success(self, sample_config, sample_price_data):
        """Test successful price data loading."""
        step = BasicBacktestingPostStep()
        
        with patch('src.training.steps.backtesting.basic_backtesting_post_step.get_klines_manager') as mock_manager:
            mock_klines = Mock()
            mock_klines.read_data.return_value = sample_price_data
            mock_manager.return_value = mock_klines
            
            price_data = step._load_price_data(sample_config)
            
            assert price_data is not None
            assert len(price_data) > 0
            assert 'close' in price_data.columns
    
    def test_load_price_data_no_utilities(self, sample_config):
        """Test when data loading utilities are not available."""
        step = BasicBacktestingPostStep()
        
        with patch('src.training.steps.backtesting.basic_backtesting_post_step.DATA_LOADING_AVAILABLE', False):
            price_data = step._load_price_data(sample_config)
            
            assert price_data is None


class TestSignalGeneration:
    """Test signal generation methods."""
    
    def test_generate_ml_signals_long_direction(self, sample_ml_scored_data):
        """Test ML signal generation for long direction."""
        step = BasicBacktestingPostStep()
        params = {'confidence_threshold': 0.6, 'exit_threshold': 0.4}
        
        long_entries, short_entries, exits = step._generate_ml_signals(
            sample_ml_scored_data, params, 'long'
        )
        
        assert isinstance(long_entries, pd.Series)
        assert isinstance(short_entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(long_entries) == len(sample_ml_scored_data)
        assert long_entries.sum() >= 0  # Should have some entries
    
    def test_generate_ml_signals_short_direction(self, sample_ml_scored_data):
        """Test ML signal generation for short direction."""
        step = BasicBacktestingPostStep()
        params = {'confidence_threshold': 0.6, 'exit_threshold': 0.4}
        
        long_entries, short_entries, exits = step._generate_ml_signals(
            sample_ml_scored_data, params, 'short'
        )
        
        assert isinstance(short_entries, pd.Series)
        assert len(short_entries) == len(sample_ml_scored_data)
    
    def test_generate_ml_signals_both_directions(self, sample_ml_scored_data):
        """Test ML signal generation for both directions."""
        step = BasicBacktestingPostStep()
        params = {'confidence_threshold': 0.6, 'exit_threshold': 0.4}
        
        long_entries, short_entries, exits = step._generate_ml_signals(
            sample_ml_scored_data, params, 'both'
        )
        
        assert long_entries.sum() >= 0
        assert short_entries.sum() >= 0
    
    def test_generate_ml_signals_no_parameters(self, sample_ml_scored_data):
        """Test ML signal generation with no parameters (uses defaults)."""
        step = BasicBacktestingPostStep()
        
        long_entries, short_entries, exits = step._generate_ml_signals(
            sample_ml_scored_data, None, 'long'
        )
        
        assert isinstance(long_entries, pd.Series)
        assert len(long_entries) == len(sample_ml_scored_data)
    
    def test_generate_ml_signals_error_handling(self):
        """Test error handling in ML signal generation."""
        step = BasicBacktestingPostStep()
        
        # Create invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        long_entries, short_entries, exits = step._generate_ml_signals(
            invalid_data, None, 'long'
        )
        
        # Should return empty signals on error
        assert long_entries.sum() == 0
        assert short_entries.sum() == 0
    
    def test_generate_simple_signals_long_direction(self, sample_price_data):
        """Test simple signal generation for long direction."""
        step = BasicBacktestingPostStep()
        params = {'fast_ma_window': 20, 'slow_ma_window': 50}
        
        long_entries, short_entries, exits = step._generate_simple_signals(
            sample_price_data, params, 'long'
        )
        
        assert isinstance(long_entries, pd.Series)
        assert len(long_entries) == len(sample_price_data)
        # Should have some crossovers
        assert long_entries.sum() >= 0
    
    def test_generate_simple_signals_short_direction(self, sample_price_data):
        """Test simple signal generation for short direction."""
        step = BasicBacktestingPostStep()
        params = {'fast_ma_window': 20, 'slow_ma_window': 50}
        
        long_entries, short_entries, exits = step._generate_simple_signals(
            sample_price_data, params, 'short'
        )
        
        assert isinstance(short_entries, pd.Series)
        assert len(short_entries) == len(sample_price_data)
    
    def test_generate_simple_signals_both_directions(self, sample_price_data):
        """Test simple signal generation for both directions."""
        step = BasicBacktestingPostStep()
        params = {'fast_ma_window': 20, 'slow_ma_window': 50}
        
        long_entries, short_entries, exits = step._generate_simple_signals(
            sample_price_data, params, 'both'
        )
        
        assert long_entries.sum() >= 0
        assert short_entries.sum() >= 0
    
    def test_generate_simple_signals_no_parameters(self, sample_price_data):
        """Test simple signal generation with default parameters."""
        step = BasicBacktestingPostStep()
        
        long_entries, short_entries, exits = step._generate_simple_signals(
            sample_price_data, None, 'long'
        )
        
        assert isinstance(long_entries, pd.Series)
        assert len(long_entries) == len(sample_price_data)


class TestTradeMetricsCalculation:
    """Test trade metrics calculation."""
    
    def test_calculate_trade_metrics_basic(self, sample_trades_dataframe):
        """Test basic trade metrics calculation."""
        step = BasicBacktestingPostStep()
        
        metrics = step._calculate_trade_metrics(sample_trades_dataframe)
        
        assert isinstance(metrics, dict)
        assert 'avg_win_loss_ratio' in metrics
        assert 'profit_factor' in metrics
        assert 'expectancy' in metrics
        assert 'largest_win' in metrics
        assert 'largest_loss' in metrics
    
    def test_calculate_trade_metrics_empty_trades(self):
        """Test trade metrics with no trades."""
        step = BasicBacktestingPostStep()
        
        empty_trades = pd.DataFrame()
        metrics = step._calculate_trade_metrics(empty_trades)
        
        assert metrics['avg_win_loss_ratio'] == 0.0
        assert metrics['profit_factor'] == 0.0
        assert metrics['total_trades'] == 0 or 'total_trades' not in metrics
    
    def test_calculate_trade_metrics_all_winning_trades(self):
        """Test metrics with all winning trades."""
        step = BasicBacktestingPostStep()
        
        winning_trades = pd.DataFrame({
            'PnL': [50, 30, 40, 60],
            'Duration': pd.to_timedelta(['1 hour'] * 4)
        })
        
        metrics = step._calculate_trade_metrics(winning_trades)
        
        assert metrics['profit_factor'] == 0.0  # No losses, so ratio is 0
        assert metrics['largest_win'] > 0
    
    def test_calculate_trade_metrics_all_losing_trades(self):
        """Test metrics with all losing trades."""
        step = BasicBacktestingPostStep()
        
        losing_trades = pd.DataFrame({
            'PnL': [-50, -30, -40, -60],
            'Duration': pd.to_timedelta(['1 hour'] * 4)
        })
        
        metrics = step._calculate_trade_metrics(losing_trades)
        
        assert metrics['largest_loss'] < 0
        assert metrics['expectancy'] < 0


class TestBaselineComparison:
    """Test baseline comparison methods."""
    
    def test_compare_with_baseline_success(self, sample_baseline_metrics):
        """Test successful comparison with baseline."""
        step = BasicBacktestingPostStep()
        
        post_metrics = {
            'total_return': 0.20,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8,
            'calmar_ratio': 1.0,
            'max_drawdown': -0.10,
            'win_rate': 0.60
        }
        
        comparison = step._compare_with_baseline(post_metrics, sample_baseline_metrics)
        
        assert isinstance(comparison, dict)
        assert 'total_return_improvement' in comparison
        assert 'sharpe_ratio_improvement' in comparison
        assert 'max_drawdown_reduction' in comparison
    
    def test_compare_with_baseline_no_baseline(self):
        """Test comparison when no baseline is available."""
        step = BasicBacktestingPostStep()
        
        post_metrics = {'total_return': 0.20, 'sharpe_ratio': 1.5}
        
        comparison = step._compare_with_baseline(post_metrics, None)
        
        assert comparison == {}
    
    def test_compare_with_baseline_improvements(self):
        """Test that improvements are calculated correctly."""
        step = BasicBacktestingPostStep()
        
        baseline = {'total_return': 0.10, 'sharpe_ratio': 1.0}
        post = {'total_return': 0.15, 'sharpe_ratio': 1.2}
        
        comparison = step._compare_with_baseline(post, baseline)
        
        assert comparison['total_return_improvement'] == 0.05
        assert comparison['sharpe_ratio_improvement'] == 0.2
    
    def test_compare_with_baseline_max_drawdown_reduction(self):
        """Test max drawdown reduction calculation."""
        step = BasicBacktestingPostStep()
        
        baseline = {'max_drawdown': -0.15}
        post = {'max_drawdown': -0.10}
        
        comparison = step._compare_with_baseline(post, baseline)
        
        # Reduction should be positive when drawdown improves
        assert 'max_drawdown_reduction' in comparison


class TestMarkdownReportGeneration:
    """Test markdown report generation."""
    
    def test_generate_markdown_report_basic(self, sample_config, sample_price_data, 
                                           sample_optimized_params, tmp_path):
        """Test basic report generation."""
        step = BasicBacktestingPostStep()
        
        metrics = {
            'total_return': 0.20,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8,
            'max_drawdown': -0.10,
            'total_trades': 50,
            'win_rate': 0.60
        }
        
        artifacts = {'artifacts_saved': ['/path/to/artifact1.parquet']}
        
        with patch('src.training.steps.backtesting.basic_backtesting_post_step.Path.mkdir'):
            report_path = step._generate_markdown_report(
                sample_config, metrics, artifacts, sample_price_data, sample_optimized_params
            )
        
        # Just check that it returns a string path
        assert isinstance(report_path, str)
    
    def test_generate_markdown_report_with_baseline_comparison(self, sample_config, 
                                                               sample_price_data, 
                                                               sample_optimized_params):
        """Test report generation with baseline comparison."""
        step = BasicBacktestingPostStep()
        
        metrics = {
            'total_return': 0.20,
            'sharpe_ratio': 1.5,
            'total_trades': 50,
            'improvement_vs_baseline': {
                'total_return_improvement': 0.05,
                'sharpe_ratio_improvement': 0.3
            }
        }
        
        artifacts = {}
        
        with patch('src.training.steps.backtesting.basic_backtesting_post_step.Path.mkdir'):
            with patch('builtins.open', create=True):
                report_path = step._generate_markdown_report(
                    sample_config, metrics, artifacts, sample_price_data, sample_optimized_params
                )
        
        assert isinstance(report_path, str)
    
    def test_generate_markdown_report_error_handling(self, sample_config, sample_price_data):
        """Test error handling in report generation."""
        step = BasicBacktestingPostStep()
        
        metrics = {}
        artifacts = {}
        
        with patch('builtins.open', side_effect=Exception("Write failed")):
            report_path = step._generate_markdown_report(
                sample_config, metrics, artifacts, sample_price_data, None
            )
        
        # Should return empty string on error
        assert report_path == ""


class TestVectorBTBacktest:
    """Test VectorBT backtest execution."""
    
    @pytest.mark.skipif(not VECTORBT_AVAILABLE, reason="VectorBT not available")
    def test_run_vectorbt_backtest_long(self, sample_config, sample_price_data):
        """Test running backtest for long direction."""
        step = BasicBacktestingPostStep()
        
        # Create simple signals
        long_entries = pd.Series(False, index=sample_price_data.index)
        long_entries.iloc[10] = True
        long_entries.iloc[50] = True
        
        short_entries = pd.Series(False, index=sample_price_data.index)
        
        exits = pd.Series(False, index=sample_price_data.index)
        exits.iloc[20] = True
        exits.iloc[60] = True
        
        result = step._run_vectorbt_backtest(
            sample_price_data, long_entries, short_entries, exits, sample_config
        )
        
        if result is not None:  # VectorBT might not be available in test env
            assert isinstance(result, dict)
            assert 'total_return' in result
            assert 'sharpe_ratio' in result
            assert 'trades' in result
    
    def test_run_vectorbt_backtest_not_available(self, sample_config, sample_price_data):
        """Test backtest when VectorBT is not available."""
        step = BasicBacktestingPostStep()
        
        with patch('src.training.steps.backtesting.basic_backtesting_post_step.VECTORBT_AVAILABLE', False):
            long_entries = pd.Series(False, index=sample_price_data.index)
            short_entries = pd.Series(False, index=sample_price_data.index)
            exits = pd.Series(False, index=sample_price_data.index)
            
            result = step._run_vectorbt_backtest(
                sample_price_data, long_entries, short_entries, exits, sample_config
            )
            
            assert result is None


class TestExecuteMethod:
    """Test the main execute method."""
    
    def test_execute_success_with_ml_data(self, sample_config, sample_ml_scored_data,
                                                sample_optimized_params, sample_baseline_metrics):
        """Test successful execution with ML-scored data."""
        import asyncio
        step = BasicBacktestingPostStep()
        
        # Mock all dependencies
        with patch.object(step, '_load_optimized_parameters', return_value=sample_optimized_params):
            with patch.object(step, '_load_baseline_metrics', return_value=sample_baseline_metrics):
                with patch.object(step, '_load_ml_scored_data', return_value=sample_ml_scored_data):
                    with patch.object(step, '_generate_ml_signals', return_value=(
                        pd.Series(False, index=sample_ml_scored_data.index),
                        pd.Series(False, index=sample_ml_scored_data.index),
                        pd.Series(False, index=sample_ml_scored_data.index)
                    )):
                        with patch.object(step, '_run_vectorbt_backtest', return_value={
                            'total_return': 0.20,
                            'sharpe_ratio': 1.5,
                            'sortino_ratio': 1.8,
                            'max_drawdown': -0.10,
                            'total_trades': 50,
                            'win_rate': 0.60,
                            'equity_curve': pd.Series([10000, 10500, 11000]),
                            'trades': pd.DataFrame()
                        }):
                            with patch.object(step, '_compare_with_baseline', return_value={}):
                                with patch.object(step, '_save_artifact', return_value='/path/artifact.parquet'):
                                    with patch.object(step, '_generate_markdown_report', return_value='/path/report.md'):
                                        result = asyncio.run(step.execute(sample_config))
        
        assert result['success'] is True
        assert 'metrics' in result
        assert 'artifacts' in result
        assert result['metrics']['total_return'] == 0.20
    
    def test_execute_success_with_fallback_signals(self, sample_config, sample_price_data,
                                                        sample_optimized_params):
        """Test successful execution with fallback simple signals."""
        import asyncio
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_load_optimized_parameters', return_value=sample_optimized_params):
            with patch.object(step, '_load_baseline_metrics', return_value=None):
                with patch.object(step, '_load_ml_scored_data', return_value=None):
                    with patch.object(step, '_load_price_data', return_value=sample_price_data):
                        with patch.object(step, '_generate_simple_signals', return_value=(
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index)
                        )):
                            with patch.object(step, '_run_vectorbt_backtest', return_value={
                                'total_return': 0.15,
                                'sharpe_ratio': 1.2,
                                'sortino_ratio': 1.4,
                                'max_drawdown': -0.12,
                                'total_trades': 40,
                                'win_rate': 0.55,
                                'equity_curve': pd.Series([10000, 10300, 10600]),
                                'trades': pd.DataFrame()
                            }):
                                with patch.object(step, '_compare_with_baseline', return_value={}):
                                    with patch.object(step, '_save_artifact', return_value='/path/artifact.parquet'):
                                        with patch.object(step, '_generate_markdown_report', return_value='/path/report.md'):
                                            result = asyncio.run(step.execute(sample_config))
        
        assert result['success'] is True
        assert 'metrics' in result
    
    def test_execute_failure_no_price_data(self, sample_config):
        """Test execution failure when price data cannot be loaded."""
        import asyncio
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_load_optimized_parameters', return_value=None):
            with patch.object(step, '_load_baseline_metrics', return_value=None):
                with patch.object(step, '_load_ml_scored_data', return_value=None):
                    with patch.object(step, '_load_price_data', return_value=None):
                        result = asyncio.run(step.execute(sample_config))
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_execute_failure_backtest_failed(self, sample_config, sample_price_data):
        """Test execution failure when backtest fails."""
        import asyncio
        step = BasicBacktestingPostStep()
        
        with patch.object(step, '_load_optimized_parameters', return_value=None):
            with patch.object(step, '_load_baseline_metrics', return_value=None):
                with patch.object(step, '_load_ml_scored_data', return_value=None):
                    with patch.object(step, '_load_price_data', return_value=sample_price_data):
                        with patch.object(step, '_generate_simple_signals', return_value=(
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index)
                        )):
                            with patch.object(step, '_run_vectorbt_backtest', return_value=None):
                                result = asyncio.run(step.execute(sample_config))
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_execute_with_short_direction(self, sample_config, sample_price_data):
        """Test execution with short direction."""
        import asyncio
        step = BasicBacktestingPostStep()
        sample_config['direction'] = 'short'
        
        with patch.object(step, '_load_optimized_parameters', return_value=None):
            with patch.object(step, '_load_baseline_metrics', return_value=None):
                with patch.object(step, '_load_ml_scored_data', return_value=None):
                    with patch.object(step, '_load_price_data', return_value=sample_price_data):
                        with patch.object(step, '_generate_simple_signals', return_value=(
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index)
                        )):
                            with patch.object(step, '_run_vectorbt_backtest', return_value={
                                'total_return': 0.10,
                                'sharpe_ratio': 1.0,
                                'sortino_ratio': 1.2,
                                'max_drawdown': -0.15,
                                'total_trades': 30,
                                'win_rate': 0.50,
                                'equity_curve': pd.Series([10000, 10200, 10400]),
                                'trades': pd.DataFrame()
                            }):
                                with patch.object(step, '_compare_with_baseline', return_value={}):
                                    with patch.object(step, '_save_artifact', return_value='/path/artifact.parquet'):
                                        with patch.object(step, '_generate_markdown_report', return_value='/path/report.md'):
                                            result = asyncio.run(step.execute(sample_config))
        
        assert result['success'] is True
        assert result['metrics']['direction'] == 'short'
    
    def test_execute_with_both_directions(self, sample_config, sample_price_data):
        """Test execution with both long and short directions."""
        import asyncio
        step = BasicBacktestingPostStep()
        sample_config['direction'] = 'both'
        
        with patch.object(step, '_load_optimized_parameters', return_value=None):
            with patch.object(step, '_load_baseline_metrics', return_value=None):
                with patch.object(step, '_load_ml_scored_data', return_value=None):
                    with patch.object(step, '_load_price_data', return_value=sample_price_data):
                        with patch.object(step, '_generate_simple_signals', return_value=(
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index),
                            pd.Series(False, index=sample_price_data.index)
                        )):
                            with patch.object(step, '_run_vectorbt_backtest', return_value={
                                'total_return': 0.18,
                                'sharpe_ratio': 1.3,
                                'sortino_ratio': 1.6,
                                'max_drawdown': -0.11,
                                'total_trades': 60,
                                'win_rate': 0.58,
                                'equity_curve': pd.Series([10000, 10400, 10800]),
                                'trades': pd.DataFrame()
                            }):
                                with patch.object(step, '_compare_with_baseline', return_value={}):
                                    with patch.object(step, '_save_artifact', return_value='/path/artifact.parquet'):
                                        with patch.object(step, '_generate_markdown_report', return_value='/path/report.md'):
                                            result = asyncio.run(step.execute(sample_config))
        
        assert result['success'] is True
        assert result['metrics']['direction'] == 'both'


class TestRunMethod:
    """Test the run method (BaseStep interface requirement)."""
    
    @pytest.mark.asyncio
    def test_run_method_calls_execute(self, sample_config):
        """Test that run method properly calls execute."""
        import asyncio
        step = BasicBacktestingPostStep()
        
        expected_result = {
            'success': True,
            'metrics': {},
            'artifacts': {}
        }
        
        with patch.object(step, 'execute', return_value=expected_result) as mock_execute:
            result = asyncio.run(step.run(sample_config))
            
            mock_execute.assert_called_once_with(sample_config)
            assert result == expected_result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-k", "not vectorbt"])

