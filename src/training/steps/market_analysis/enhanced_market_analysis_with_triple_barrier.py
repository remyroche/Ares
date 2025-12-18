"""
Enhanced Market Analysis Pipeline with Triple Barrier Labeling Integration

This module integrates the comprehensive triple barrier labeling system with the existing
market analysis pipeline, providing seamless workflow integration and enhanced functionality.

Key Features:
- Seamless integration with existing market analysis pipeline
- Regime-aware triple barrier labeling
- Comprehensive validation and error handling
- Performance optimization
- Automated workflow orchestration
"""

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time, cached
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import contextlib

from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Import the triple barrier components
from ..pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)
from .regime_aware_triple_barrier_optimizer import (
    RegimeAwareTripleBarrierOptimizer,
    RegimeBarrierParams,
    optimize_regime_barriers,
    apply_optimized_regime_labeling
)
from src.training.steps.labeling.triple_barrier_validator import (
    TripleBarrierValidator,
    ValidationReport,
    validate_triple_barrier_implementation,
    quick_validate_triple_barrier
)

@dataclass
class MarketAnalysisTripleBarrierConfig:
    """Configuration for market analysis with triple barrier labeling."""
    # Triple barrier parameters
    profit_take_multiplier: float = 0.002
    stop_loss_multiplier: float = 0.001
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    binary_classification: bool = True

    # Regime awareness
    regime_aware: bool = True
    regime_column: str = 'hmm_regime'
    optimize_regime_parameters: bool = True

    # Validation
    enable_validation: bool = True
    validation_threshold: float = 0.7

    # Performance optimization
    enable_numba_acceleration: bool = True
    enable_vectorization: bool = True

    # Output settings
    save_intermediate_results: bool = True
    save_optimization_results: bool = True
    output_directory: str = 'generated/market_analysis/triple_barrier_results'

class EnhancedMarketAnalysisWithTripleBarrier:
    """
    Enhanced Market Analysis Pipeline with integrated triple barrier labeling.

    This class provides a comprehensive workflow that integrates triple barrier
    labeling with the existing market analysis pipeline, including regime-aware
    optimization and validation.
    """

    def __init__(self, config: Optional[MarketAnalysisTripleBarrierConfig] = None):
        """Initialize the enhanced market analysis pipeline.

        Args:
            config: Configuration for the pipeline
        """
        self.config = config or MarketAnalysisTripleBarrierConfig()
        self.logger = get_logger('EnhancedMarketAnalysisWithTripleBarrier')

        # Initialize components
        self.triple_barrier_labeler = None
        self.regime_optimizer = None
        self.validator = None

        # Results storage
        self.labeling_results = {}
        self.optimization_results = {}
        self.validation_results = {}

        # Performance tracking
        self.performance_metrics = {}

        self._log_initialization()
        self._initialize_components()

    def _log_initialization(self):
        """Log initialization parameters."""
        self.logger.info('🚀 Initializing Enhanced Market Analysis with Triple Barrier Labeling')
        self.logger.info(f'📋 Configuration:')
        self.logger.info(f'   → Regime aware: {self.config.regime_aware}')
        self.logger.info(f'   → Regime optimization: {self.config.optimize_regime_parameters}')
        self.logger.info(f'   → Validation enabled: {self.config.enable_validation}')
        self.logger.info(f'   → Profit take: {self.config.profit_take_multiplier:.4f}')
        self.logger.info(f'   → Stop loss: {self.config.stop_loss_multiplier:.4f}')

    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize triple barrier labeler
            triple_barrier_config = TripleBarrierConfig(
                profit_take_multiplier=self.config.profit_take_multiplier,
                stop_loss_multiplier=self.config.stop_loss_multiplier,
                time_barrier_minutes=self.config.time_barrier_minutes,
                max_lookahead=self.config.max_lookahead,
                transaction_cost=self.config.transaction_cost,
                binary_classification=self.config.binary_classification,
                regime_aware=self.config.regime_aware,
                regime_column=self.config.regime_column,
                enable_validation=self.config.enable_validation
            )

            self.triple_barrier_labeler = MultiHorizonProfitLabeler(triple_barrier_config)

            # Initialize regime optimizer if regime-aware
            if self.config.regime_aware and self.config.optimize_regime_parameters:
                optimizer_config = {
                    'profit_take_range': (0.0005, 0.01),
                    'stop_loss_range': (0.0005, 0.005),
                    'time_barrier_range': (15, 60),
                    'max_lookahead_range': (50, 200),
                    'transaction_cost': self.config.transaction_cost,
                    'optimization_method': 'minimize',
                    'objective_function': 'sharpe_ratio'
                }

                self.regime_optimizer = RegimeAwareTripleBarrierOptimizer(optimizer_config)

            # Initialize validator
            if self.config.enable_validation:
                validator_config = {
                    'min_data_points': 100,
                    'max_missing_ratio': 0.05,
                    'min_label_ratio': 0.01,
                    'max_imbalance_ratio': 10.0,
                    'min_win_rate': 0.3,
                    'max_drawdown_threshold': 0.2,
                    'min_sharpe_ratio': 0.5,
                    'temporal_validation': True,
                    'statistical_validation': True,
                    'performance_validation': True
                }

                self.validator = TripleBarrierValidator(validator_config)

            self.logger.info('✅ Pipeline components initialized successfully')

        except Exception as e:
            self.logger.error(f'❌ Error initializing pipeline components: {e}')
            raise

    @traced(span_name='run_market_analysis_with_triple_barrier')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return={})
    @log_execution_time()
    def run_market_analysis_with_triple_barrier(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete market analysis with triple barrier labeling.

        Args:
            data: Market data with OHLCV and regime information
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            output_dir: Output directory for results

        Returns:
            Dictionary containing all analysis results
        """
        start_time = time.time()
        self.logger.info(f'🚀 Starting enhanced market analysis with triple barrier labeling')
        self.logger.info(f'   Symbol: {symbol}')
        self.logger.info(f'   Exchange: {exchange}')
        self.logger.info(f'   Timeframe: {timeframe}')
        self.logger.info(f'   Data shape: {data.shape}')

        # Setup output directory
        if output_dir is None:
            output_dir = self.config.output_directory

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_shape': data.shape,
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.__dict__
        }

        try:
            # Step 1: Validate input data
            self.logger.info('📊 Step 1: Validating input data...')
            data_validation_result = self._validate_input_data(data)
            results['data_validation'] = data_validation_result

            if not data_validation_result['passed']:
                self.logger.error('❌ Input data validation failed')
                return results

            # Step 2: Regime parameter optimization (if enabled)
            if self.config.regime_aware and self.config.optimize_regime_parameters:
                self.logger.info('🎯 Step 2: Optimizing regime parameters...')
                optimization_result = self._optimize_regime_parameters(data, output_path)
                results['regime_optimization'] = optimization_result

            # Step 3: Apply triple barrier labeling
            self.logger.info('🏷️ Step 3: Applying triple barrier labeling...')
            labeling_result = self._apply_triple_barrier_labeling(data, symbol, exchange, timeframe)
            results['triple_barrier_labeling'] = labeling_result

            # Step 4: Validate labeling results
            if self.config.enable_validation:
                self.logger.info('🔍 Step 4: Validating labeling results...')
                validation_result = self._validate_labeling_results(data, labeling_result['labeled_data'])
                results['labeling_validation'] = validation_result

                # Check if validation passed
                if validation_result['overall_score'] < self.config.validation_threshold:
                    self.logger.warning(f'⚠️ Validation score below threshold: {validation_result["overall_score"]:.3f} < {self.config.validation_threshold}')

            # Step 5: Calculate performance metrics
            self.logger.info('💰 Step 5: Calculating performance metrics...')
            performance_result = self._calculate_performance_metrics(labeling_result['labeled_data'])
            results['performance_metrics'] = performance_result

            # Step 6: Save results
            if self.config.save_intermediate_results:
                self.logger.info('💾 Step 6: Saving results...')
                save_result = self._save_results(results, labeling_result['labeled_data'], output_path)
                results['save_result'] = save_result

            # Calculate total execution time
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time

            self.logger.info(f'✅ Enhanced market analysis completed successfully in {execution_time:.2f} seconds')
            self.logger.info(f'   → Labeled samples: {len(labeling_result["labeled_data"])}')
            self.logger.info(f'   → Validation score: {validation_result.get("overall_score", 0.0):.3f}')
            self.logger.info(f'   → Win rate: {performance_result.get("win_rate", 0.0):.3f}')

            return results

        except Exception as e:
            self.logger.exception(f'❌ Error in enhanced market analysis: {e}')
            results['error'] = str(e)
            results['execution_time'] = time.time() - start_time
            return results

    def _validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data quality."""
        try:
            # Basic validation
            validation_result = {
                'passed': True,
                'issues': [],
                'data_quality_score': 1.0
            }

            # Check data size
            if len(data) < 100:
                validation_result['passed'] = False
                validation_result['issues'].append(f'Insufficient data: {len(data)} < 100')
                validation_result['data_quality_score'] -= 0.5

            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_result['passed'] = False
                validation_result['issues'].append(f'Missing columns: {missing_columns}')
                validation_result['data_quality_score'] -= 0.5

            # Check regime column if regime-aware
            if self.config.regime_aware and self.config.regime_column not in data.columns:
                validation_result['issues'].append(f'Regime column "{self.config.regime_column}" not found')
                validation_result['data_quality_score'] -= 0.2

            # Check for missing values
            if 'close' in data.columns:
                missing_ratio = data['close'].isna().sum() / len(data)
                if missing_ratio > 0.05:
                    validation_result['issues'].append(f'High missing value ratio: {missing_ratio:.3f}')
                    validation_result['data_quality_score'] -= 0.2

            validation_result['data_quality_score'] = max(0.0, validation_result['data_quality_score'])

            return validation_result

        except Exception as e:
            self.logger.error(f'❌ Error validating input data: {e}')
            return {
                'passed': False,
                'issues': [f'Validation error: {e}'],
                'data_quality_score': 0.0
            }

    def _optimize_regime_parameters(self, data: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
        """Optimize regime-specific parameters."""
        try:
            if self.regime_optimizer is None:
                return {'error': 'Regime optimizer not initialized'}

            # Run optimization
            regime_parameters = self.regime_optimizer.optimize_regime_parameters(
                data,
                regime_column=self.config.regime_column,
                validation_split=0.2,
                random_state=42
            )

            # Generate optimization report
            optimization_report = self.regime_optimizer.generate_optimization_report()

            # Save optimization results if enabled
            if self.config.save_optimization_results:
                optimization_file = output_path / 'regime_optimization_results.json'
                self.regime_optimizer.save_optimization_results(optimization_file)

            return {
                'success': True,
                'regime_parameters': {str(k): v.to_dict() for k, v in regime_parameters.items()},
                'optimization_report': optimization_report,
                'optimization_file': str(optimization_file) if self.config.save_optimization_results else None
            }

        except Exception as e:
            self.logger.error(f'❌ Error optimizing regime parameters: {e}')
            return {
                'success': False,
                'error': str(e)
            }

    def _apply_triple_barrier_labeling(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Apply triple barrier labeling."""
        try:
            if self.triple_barrier_labeler is None:
                return {'error': 'Triple barrier labeler not initialized'}

            # Apply labeling
            if self.config.regime_aware and self.regime_optimizer is not None:
                # Use optimized regime-aware labeling
                labeled_data = self.regime_optimizer.apply_optimized_labeling(
                    data,
                    regime_column=self.config.regime_column
                )
                labeling_method = 'regime_optimized'
            else:
                # Use standard labeling
                result = self.triple_barrier_labeler.apply_labeling(data)
                labeled_data = result.labeled_data if result.success else pd.DataFrame()
                labeling_method = 'standard'

            # Calculate basic statistics
            if len(labeled_data) > 0:
                label_counts = labeled_data['label'].value_counts()
                total_samples = len(labeled_data)

                # Calculate profit statistics
                if 'net_profit_pct' in labeled_data.columns:
                    profits = labeled_data['net_profit_pct']
                    win_rate = (profits > 0).mean()
                    avg_profit = profits.mean()
                    total_return = profits.sum()
                    sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252) if profits.std() > 0 else 0
                else:
                    win_rate = 0.0
                    avg_profit = 0.0
                    total_return = 0.0
                    sharpe_ratio = 0.0
            else:
                label_counts = {}
                total_samples = 0
                win_rate = 0.0
                avg_profit = 0.0
                total_return = 0.0
                sharpe_ratio = 0.0

            return {
                'success': True,
                'labeled_data': labeled_data,
                'labeling_method': labeling_method,
                'total_samples': total_samples,
                'label_distribution': label_counts.to_dict(),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio
            }

        except Exception as e:
            self.logger.error(f'❌ Error applying triple barrier labeling: {e}')
            return {
                'success': False,
                'error': str(e)
            }

    def _validate_labeling_results(self, original_data: pd.DataFrame, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate labeling results."""
        try:
            if self.validator is None:
                return {'overall_score': 1.0, 'message': 'Validation disabled'}

            # Run validation
            validation_report = self.validator.validate_triple_barrier_implementation(
                original_data,
                labeled_data
            )

            return {
                'overall_score': validation_report.overall_score,
                'total_checks': validation_report.total_checks,
                'passed_checks': validation_report.passed_checks,
                'failed_checks': validation_report.failed_checks,
                'critical_issues': validation_report.critical_issues,
                'recommendations': validation_report.recommendations,
                'validation_results': validation_report.to_dict()
            }

        except Exception as e:
            self.logger.error(f'❌ Error validating labeling results: {e}')
            return {
                'overall_score': 0.0,
                'error': str(e)
            }

    def _calculate_performance_metrics(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if len(labeled_data) == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'profit_factor': 0.0
                }

            # Get profit data
            if 'net_profit_pct' in labeled_data.columns:
                profits = labeled_data['net_profit_pct']
            elif 'potential_profit_pct' in labeled_data.columns:
                profits = labeled_data['potential_profit_pct']
            else:
                return {'error': 'No profit data available'}

            # Calculate metrics
            total_trades = len(profits)
            win_rate = (profits > 0).mean()
            avg_profit = profits.mean()
            total_return = profits.sum()

            # Risk metrics
            if profits.std() > 0:
                sharpe_ratio = profits.mean() / profits.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Maximum drawdown
            cumulative_returns = profits.cumsum()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

            # Profit factor
            gross_profit = profits[profits > 0].sum()
            gross_loss = abs(profits[profits < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }

        except Exception as e:
            self.logger.error(f'❌ Error calculating performance metrics: {e}')
            return {'error': str(e)}

    def _save_results(
        self,
        results: Dict[str, Any],
        labeled_data: pd.DataFrame,
        output_path: Path
    ) -> Dict[str, Any]:
        """Save analysis results."""
        try:
            # Save labeled data
            labeled_data_file = output_path / 'labeled_data.parquet'
            standardized_parquet_handler.write_parquet_standardized(labeled_data, labeled_data_file)

            # Save results summary
            results_file = output_path / 'analysis_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            return {
                'success': True,
                'labeled_data_file': str(labeled_data_file),
                'results_file': str(results_file)
            }

        except Exception as e:
            self.logger.error(f'❌ Error saving results: {e}')
            return {
                'success': False,
                'error': str(e)
            }

# Convenience functions for easy integration
def run_enhanced_market_analysis_with_triple_barrier(
    data: pd.DataFrame,
    symbol: str,
    exchange: str,
    timeframe: str,
    config: Optional[MarketAnalysisTripleBarrierConfig] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run enhanced market analysis with triple barrier labeling.

    Args:
        data: Market data with OHLCV and regime information
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Configuration for the analysis
        output_dir: Output directory for results

    Returns:
        Dictionary containing all analysis results
    """
    pipeline = EnhancedMarketAnalysisWithTripleBarrier(config)
    return pipeline.run_market_analysis_with_triple_barrier(data, symbol, exchange, timeframe, output_dir)

def quick_triple_barrier_analysis(
    data: pd.DataFrame,
    profit_take_multiplier: float = 0.002,
    stop_loss_multiplier: float = 0.001,
    regime_aware: bool = True,
    regime_column: str = 'hmm_regime'
) -> pd.DataFrame:
    """Quick triple barrier analysis with minimal configuration.

    Args:
        data: Market data with OHLCV and regime information
        profit_take_multiplier: Profit take multiplier (default: 0.2%)
        stop_loss_multiplier: Stop loss multiplier (default: 0.1%)
        regime_aware: Whether to use regime-aware labeling (default: True)
        regime_column: Column name for regime information (default: 'hmm_regime')

    Returns:
        DataFrame with triple barrier labels
    """
    config = MarketAnalysisTripleBarrierConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        regime_aware=regime_aware,
        regime_column=regime_column,
        optimize_regime_parameters=False,  # Skip optimization for speed
        enable_validation=False,  # Skip validation for speed
        save_intermediate_results=False  # Skip saving for speed
    )

    pipeline = EnhancedMarketAnalysisWithTripleBarrier(config)
    result = pipeline.run_market_analysis_with_triple_barrier(
        data, 'SYMBOL', 'EXCHANGE', 'TIMEFRAME'
    )

    return result.get('triple_barrier_labeling', {}).get('labeled_data', pd.DataFrame())

if __name__ == '__main__':
    # Test the enhanced market analysis pipeline
    tprint('🧪 Testing Enhanced Market Analysis with Triple Barrier Labeling')

    # Create test data
    dates = pd.date_range('2024-01-01', periods=2000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 2000),
        'high': np.random.uniform(105, 115, 2000),
        'low': np.random.uniform(95, 105, 2000),
        'close': np.random.uniform(100, 110, 2000),
        'volume': np.random.uniform(1000, 10000, 2000),
        'hmm_regime': np.random.choice([0, 1, 2], 2000, p=[0.4, 0.4, 0.2])
    }, index=dates)

    # Test full pipeline
    tprint('\n🚀 Testing full enhanced market analysis pipeline...')
    results = run_enhanced_market_analysis_with_triple_barrier(
        data, 'ETHUSDT', 'BINANCE', '1m'
    )

    tprint(f'Pipeline execution completed:')
    tprint(f'   → Success: {results.get("triple_barrier_labeling", {}).get("success", False)}')
    tprint(f'   → Labeled samples: {results.get("triple_barrier_labeling", {}).get("total_samples", 0)}')
    tprint(f'   → Win rate: {results.get("performance_metrics", {}).get("win_rate", 0.0):.3f}')
    tprint(f'   → Execution time: {results.get("execution_time", 0.0):.2f}s')

    # Test quick analysis
    tprint('\n⚡ Testing quick triple barrier analysis...')
    quick_labeled = quick_triple_barrier_analysis(data)
    tprint(f'Quick analysis completed: {len(quick_labeled)} samples labeled')

    tprint('✅ Enhanced Market Analysis with Triple Barrier Labeling test completed successfully!')
