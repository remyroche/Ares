"""
Comprehensive Backtesting Example for TAS

Demonstrates the complete backtesting framework including:
- Historical backtesting
- Walk-forward analysis
- Performance attribution
- Risk analysis
- Scenario testing
- Monte Carlo simulation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import TAS backtesting components
from ..backtesting_engine import BacktestingEngine, BacktestingConfig, BacktestingResult
from src.utils.nas_tas.walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig, WalkForwardResult
from src.utils.nas_tas.performance_attribution import PerformanceAttributor, AttributionConfig, AttributionResult
from src.utils.nas_tas.risk_analysis import RiskAnalyzer, RiskConfig, RiskResult
from ..scenario_testing import ScenarioTester, ScenarioConfig, ScenarioResult
from ..monte_carlo import MonteCarloSimulator, MonteCarloConfig, MonteCarloResult
from ..data_manager import BacktestingDataManager, DataConfig, DataResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_days: int = 1000, start_date: datetime = None) -> pd.DataFrame:
    """Generate synthetic market data for backtesting."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_days)
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    # Generate returns with regime changes
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    
    # Add regime changes
    regime_changes = [200, 400, 600, 800]
    for change_point in regime_changes:
        if change_point < n_days:
            # High volatility regime
            returns[change_point:change_point+50] = np.random.normal(0.001, 0.03, 50)
            # Low volatility regime
            returns[change_point+50:change_point+100] = np.random.normal(0.0002, 0.01, 50)
    
    # Generate OHLCV data
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame(index=dates)
    data['open'] = prices
    data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    data['close'] = prices * (1 + np.random.normal(0, 0.005, n_days))
    data['volume'] = np.random.lognormal(10, 1, n_days)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data


def run_comprehensive_backtesting_example():
    """Run comprehensive backtesting example."""
    logger.info("🚀 Starting comprehensive backtesting example")
    
    try:
        # Step 1: Generate synthetic data
        logger.info("📊 Step 1: Generating synthetic market data...")
        market_data = generate_synthetic_data(n_days=1000)
        logger.info(f"✅ Generated {len(market_data)} data points")
        
        # Step 2: Set up data management
        logger.info("📊 Step 2: Setting up data management...")
        data_config = DataConfig(
            data_source=DataSource.MEMORY,
            enable_data_cleaning=True,
            enable_outlier_detection=True,
            enable_technical_indicators=True,
            enable_regime_features=True
        )
        
        data_manager = BacktestingDataManager(data_config)
        data_result = data_manager.load_data(market_data)
        logger.info(f"✅ Data management completed: {data_result.data_quality_score:.3f} quality score")
        
        # Step 3: Run historical backtesting
        logger.info("📊 Step 3: Running historical backtesting...")
        backtesting_config = BacktestingConfig(
            start_date=market_data.index[0],
            end_date=market_data.index[-1],
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        backtesting_engine = BacktestingEngine(backtesting_config)
        backtesting_result = backtesting_engine.run_backtest(
            market_data=data_result.processed_data,
            regime_data=data_result.regime_data
        )
        logger.info(f"✅ Historical backtesting completed: {backtesting_result.total_return:.2%} return")
        
        # Step 4: Run walk-forward analysis
        logger.info("📊 Step 4: Running walk-forward analysis...")
        walk_forward_config = WalkForwardConfig(
            training_window=252,
            testing_window=63,
            step_size=21,
            min_sharpe_ratio=0.5,
            max_drawdown_threshold=0.15
        )
        
        walk_forward_analyzer = WalkForwardAnalyzer(walk_forward_config)
        walk_forward_result = walk_forward_analyzer.run_analysis(
            market_data=data_result.processed_data,
            regime_data=data_result.regime_data
        )
        logger.info(f"✅ Walk-forward analysis completed: {walk_forward_result.success_rate:.2%} success rate")
        
        # Step 5: Run performance attribution
        logger.info("📊 Step 5: Running performance attribution...")
        attribution_config = AttributionConfig(
            attribution_methods=[AttributionMethod.REGIME_BASED, AttributionMethod.TIME_BASED],
            enable_regime_attribution=True,
            enable_time_attribution=True
        )
        
        performance_attributor = PerformanceAttributor(attribution_config)
        attribution_result = performance_attributor.run_attribution(
            returns_series=backtesting_result.returns_series,
            regime_data=data_result.regime_data
        )
        logger.info(f"✅ Performance attribution completed: {attribution_result.attribution_ratio:.2%} attribution ratio")
        
        # Step 6: Run risk analysis
        logger.info("📊 Step 6: Running risk analysis...")
        risk_config = RiskConfig(
            var_confidence_levels=[0.95, 0.99],
            enable_stress_testing=True,
            enable_scenario_analysis=True,
            enable_risk_attribution=True
        )
        
        risk_analyzer = RiskAnalyzer(risk_config)
        risk_result = risk_analyzer.run_analysis(
            returns_series=backtesting_result.returns_series,
            regime_data=data_result.regime_data
        )
        logger.info(f"✅ Risk analysis completed: {risk_result.var_95:.2%} VaR 95%")
        
        # Step 7: Run scenario testing
        logger.info("📊 Step 7: Running scenario testing...")
        scenario_config = ScenarioConfig(
            scenario_types=[ScenarioType.STRESS, ScenarioType.MONTE_CARLO, ScenarioType.SENSITIVITY],
            n_simulations=5000,
            enable_stress_testing=True,
            enable_scenario_analysis=True
        )
        
        scenario_tester = ScenarioTester(scenario_config)
        scenario_result = scenario_tester.run_scenario_testing(
            returns_series=backtesting_result.returns_series,
            regime_data=data_result.regime_data
        )
        logger.info(f"✅ Scenario testing completed: {scenario_result.scenario_risk_score:.3f} risk score")
        
        # Step 8: Run Monte Carlo simulation
        logger.info("📊 Step 8: Running Monte Carlo simulation...")
        monte_carlo_config = MonteCarloConfig(
            n_simulations=10000,
            simulation_horizon=252,
            method=MonteCarloMethod.PARAMETRIC,
            use_regime_data=True
        )
        
        monte_carlo_simulator = MonteCarloSimulator(monte_carlo_config)
        monte_carlo_result = monte_carlo_simulator.run_simulation(
            returns_series=backtesting_result.returns_series,
            regime_data=data_result.regime_data
        )
        logger.info(f"✅ Monte Carlo simulation completed: {monte_carlo_result.expected_return:.2%} expected return")
        
        # Step 9: Generate comprehensive report
        logger.info("📊 Step 9: Generating comprehensive report...")
        generate_comprehensive_report(
            backtesting_result=backtesting_result,
            walk_forward_result=walk_forward_result,
            attribution_result=attribution_result,
            risk_result=risk_result,
            scenario_result=scenario_result,
            monte_carlo_result=monte_carlo_result,
            data_result=data_result
        )
        
        logger.info("✅ Comprehensive backtesting example completed successfully!")
        
        return {
            'backtesting_result': backtesting_result,
            'walk_forward_result': walk_forward_result,
            'attribution_result': attribution_result,
            'risk_result': risk_result,
            'scenario_result': scenario_result,
            'monte_carlo_result': monte_carlo_result,
            'data_result': data_result
        }
        
    except Exception as e:
        logger.error(f"❌ Comprehensive backtesting example failed: {e}")
        raise


def generate_comprehensive_report(backtesting_result: BacktestingResult,
                                walk_forward_result: WalkForwardResult,
                                attribution_result: AttributionResult,
                                risk_result: RiskResult,
                                scenario_result: ScenarioResult,
                                monte_carlo_result: MonteCarloResult,
                                data_result: DataResult):
    """Generate comprehensive backtesting report."""
    
    logger.info("📊 COMPREHENSIVE BACKTESTING REPORT")
    logger.info("=" * 50)
    
    # Data Quality Report
    logger.info("📊 DATA QUALITY REPORT")
    logger.info(f"   Data Shape: {data_result.data_shape}")
    logger.info(f"   Data Quality Score: {data_result.data_quality_score:.3f}")
    logger.info(f"   Missing Values: {sum(data_result.missing_values.values())}")
    logger.info(f"   Outliers: {data_result.outlier_count}")
    logger.info(f"   Data Range: {data_result.data_range[0]} to {data_result.data_range[1]}")
    
    # Historical Backtesting Report
    logger.info("\n📈 HISTORICAL BACKTESTING REPORT")
    logger.info(f"   Total Return: {backtesting_result.total_return:.2%}")
    logger.info(f"   Annualized Return: {backtesting_result.annualized_return:.2%}")
    logger.info(f"   Volatility: {backtesting_result.volatility:.2%}")
    logger.info(f"   Sharpe Ratio: {backtesting_result.sharpe_ratio:.3f}")
    logger.info(f"   Sortino Ratio: {backtesting_result.sortino_ratio:.3f}")
    logger.info(f"   Max Drawdown: {backtesting_result.max_drawdown:.2%}")
    logger.info(f"   Calmar Ratio: {backtesting_result.calmar_ratio:.3f}")
    logger.info(f"   Total Trades: {backtesting_result.total_trades}")
    logger.info(f"   Win Rate: {backtesting_result.win_rate:.2%}")
    logger.info(f"   Profit Factor: {backtesting_result.profit_factor:.3f}")
    
    # Walk-Forward Analysis Report
    logger.info("\n🔄 WALK-FORWARD ANALYSIS REPORT")
    logger.info(f"   Number of Periods: {walk_forward_result.n_periods}")
    logger.info(f"   Successful Periods: {walk_forward_result.successful_periods}")
    logger.info(f"   Failed Periods: {walk_forward_result.failed_periods}")
    logger.info(f"   Success Rate: {walk_forward_result.success_rate:.2%}")
    logger.info(f"   Average Return: {walk_forward_result.average_return:.2%}")
    logger.info(f"   Average Sharpe: {walk_forward_result.average_sharpe:.3f}")
    logger.info(f"   Average Drawdown: {walk_forward_result.average_drawdown:.2%}")
    logger.info(f"   Total Return: {walk_forward_result.total_return:.2%}")
    logger.info(f"   Cumulative Return: {walk_forward_result.cumulative_return:.2%}")
    
    # Performance Attribution Report
    logger.info("\n🔍 PERFORMANCE ATTRIBUTION REPORT")
    logger.info(f"   Total Attribution: {attribution_result.total_attribution:.2%}")
    logger.info(f"   Unexplained Return: {attribution_result.unexplained_return:.2%}")
    logger.info(f"   Attribution Ratio: {attribution_result.attribution_ratio:.2%}")
    logger.info(f"   R-squared: {attribution_result.r_squared:.3f}")
    logger.info(f"   Adjusted R-squared: {attribution_result.adjusted_r_squared:.3f}")
    logger.info(f"   F-statistic: {attribution_result.f_statistic:.3f}")
    logger.info(f"   P-value: {attribution_result.p_value:.3f}")
    
    # Risk Analysis Report
    logger.info("\n⚠️ RISK ANALYSIS REPORT")
    logger.info(f"   VaR 95%: {risk_result.var_95:.2%}")
    logger.info(f"   VaR 99%: {risk_result.var_99:.2%}")
    logger.info(f"   CVaR 95%: {risk_result.cvar_95:.2%}")
    logger.info(f"   CVaR 99%: {risk_result.cvar_99:.2%}")
    logger.info(f"   Max Drawdown: {risk_result.max_drawdown:.2%}")
    logger.info(f"   Sharpe Ratio: {risk_result.sharpe_ratio:.3f}")
    logger.info(f"   Sortino Ratio: {risk_result.sortino_ratio:.3f}")
    logger.info(f"   Calmar Ratio: {risk_result.calmar_ratio:.3f}")
    logger.info(f"   Omega Ratio: {risk_result.omega_ratio:.3f}")
    logger.info(f"   Beta: {risk_result.beta:.3f}")
    logger.info(f"   Alpha: {risk_result.alpha:.2%}")
    logger.info(f"   Volatility: {risk_result.volatility:.2%}")
    logger.info(f"   Skewness: {risk_result.skewness:.3f}")
    logger.info(f"   Kurtosis: {risk_result.kurtosis:.3f}")
    
    # Scenario Testing Report
    logger.info("\n🎯 SCENARIO TESTING REPORT")
    logger.info(f"   Worst Case Scenario: {scenario_result.worst_case_scenario}")
    logger.info(f"   Best Case Scenario: {scenario_result.best_case_scenario}")
    logger.info(f"   Expected Scenario: {scenario_result.expected_scenario}")
    logger.info(f"   Scenario Risk Score: {scenario_result.scenario_risk_score:.3f}")
    logger.info(f"   Stress Scenarios: {len(scenario_result.stress_scenarios)}")
    logger.info(f"   Monte Carlo Results: {len(scenario_result.monte_carlo_results)}")
    logger.info(f"   Sensitivity Rankings: {len(scenario_result.sensitivity_rankings)}")
    
    # Monte Carlo Simulation Report
    logger.info("\n🎲 MONTE CARLO SIMULATION REPORT")
    logger.info(f"   Number of Simulations: {monte_carlo_result.config.n_simulations}")
    logger.info(f"   Simulation Horizon: {monte_carlo_result.config.simulation_horizon} days")
    logger.info(f"   Expected Return: {monte_carlo_result.expected_return:.2%}")
    logger.info(f"   Expected Volatility: {monte_carlo_result.expected_volatility:.2%}")
    logger.info(f"   VaR 95%: {monte_carlo_result.var_95:.2%}")
    logger.info(f"   VaR 99%: {monte_carlo_result.var_99:.2%}")
    logger.info(f"   CVaR 95%: {monte_carlo_result.cvar_95:.2%}")
    logger.info(f"   CVaR 99%: {monte_carlo_result.cvar_99:.2%}")
    logger.info(f"   Percentiles: {len(monte_carlo_result.percentiles)}")
    logger.info(f"   Confidence Intervals: {len(monte_carlo_result.confidence_intervals)}")
    
    # Execution Summary
    logger.info("\n⏱️ EXECUTION SUMMARY")
    logger.info(f"   Data Processing Time: {data_result.processing_time:.2f}s")
    logger.info(f"   Backtesting Time: {backtesting_result.execution_time:.2f}s")
    logger.info(f"   Walk-Forward Time: {walk_forward_result.execution_time:.2f}s")
    logger.info(f"   Attribution Time: {attribution_result.execution_time:.2f}s")
    logger.info(f"   Risk Analysis Time: {risk_result.execution_time:.2f}s")
    logger.info(f"   Scenario Testing Time: {scenario_result.execution_time:.2f}s")
    logger.info(f"   Monte Carlo Time: {monte_carlo_result.execution_time:.2f}s")
    
    total_time = (data_result.processing_time + backtesting_result.execution_time + 
                 walk_forward_result.execution_time + attribution_result.execution_time + 
                 risk_result.execution_time + scenario_result.execution_time + 
                 monte_carlo_result.execution_time)
    
    logger.info(f"   Total Execution Time: {total_time:.2f}s")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Run comprehensive backtesting example
    results = run_comprehensive_backtesting_example()
    
    # Export results
    logger.info("📁 Exporting results...")
    
    # Export backtesting results
    backtesting_engine = BacktestingEngine(BacktestingConfig())
    backtesting_engine.results = results['backtesting_result']
    backtesting_engine.export_results("backtesting_results.csv")
    
    # Export walk-forward results
    walk_forward_analyzer = WalkForwardAnalyzer(WalkForwardConfig())
    walk_forward_analyzer.results = results['walk_forward_result']
    walk_forward_analyzer.export_results("walk_forward_results.csv")
    
    # Export attribution results
    performance_attributor = PerformanceAttributor(AttributionConfig())
    performance_attributor.results = results['attribution_result']
    performance_attributor.export_results("attribution_results.csv")
    
    # Export risk analysis results
    risk_analyzer = RiskAnalyzer(RiskConfig())
    risk_analyzer.results = results['risk_result']
    risk_analyzer.export_results("risk_analysis_results.csv")
    
    # Export scenario testing results
    scenario_tester = ScenarioTester(ScenarioConfig())
    scenario_tester.results = results['scenario_result']
    scenario_tester.export_results("scenario_testing_results.csv")
    
    # Export Monte Carlo results
    monte_carlo_simulator = MonteCarloSimulator(MonteCarloConfig())
    monte_carlo_simulator.results = results['monte_carlo_result']
    monte_carlo_simulator.export_results("monte_carlo_results.csv")
    
    # Export data results
    data_manager = BacktestingDataManager(DataConfig())
    data_manager.processed_data = results['data_result'].processed_data
    data_manager.feature_data = results['data_result'].feature_data
    data_manager.regime_data = results['data_result'].regime_data
    data_manager.export_data("processed_data.csv")
    
    logger.info("✅ All results exported successfully!")
    logger.info("🎉 Comprehensive backtesting example completed!")