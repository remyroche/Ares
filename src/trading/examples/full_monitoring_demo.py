"""
Full Monitoring Demo

Complete demonstration of the comprehensive trading monitoring system
showing all features: detailed trade metrics, SHAP/LIME explanations,
performance reporting, and live dashboards.
"""

import asyncio
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error, tprint_structured, LogLevel

# Import all trading components
from ..execution.trading_orchestrator import TradingOrchestrator, create_trading_orchestrator
from ..monitoring.comprehensive_trade_monitor import (
    comprehensive_trade_monitor, initialize_comprehensive_monitoring,
    record_detailed_trade, update_trade_outcome, DetailedTradeMetrics
)
from ..reporting.performance_reporter import generate_trading_report
from ..reporting.dashboard_generator import create_trading_dashboard
from ..reporting.trade_analyzer import analyze_trade_performance

async def main():
    """
    Main demonstration of comprehensive trading monitoring.

    This example shows:
    1. Setting up comprehensive monitoring
    2. Recording detailed trade metrics
    3. Generating SHAP/LIME explanations
    4. Creating performance reports
    5. Building live dashboards
    6. Analyzing individual trades
    """

    tprint_info("🚀 Starting Full Trading Monitoring Demonstration")
    print("=" * 80)

    try:
        # Step 1: Initialize Comprehensive Monitoring
        tprint_info("📊 Step 1: Initializing Comprehensive Monitoring System")

        monitoring_config = {
            'enable_explanations': True,
            'enable_real_time_export': True,
            'export_directory': 'demo_trading_reports',
            'max_trades_in_memory': 1000,
            'enable_shap': True,
            'enable_lime': True
        }

        # Initialize comprehensive monitoring
        try:
            if not comprehensive_trade_monitor.is_initialized:
                success = await comprehensive_trade_monitor.initialize(monitoring_config)
                if not success:
                    tprint_error("❌ Failed to initialize monitoring system")
                    return
            else:
                tprint_success("✅ Comprehensive monitoring system already initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize monitoring system: {e}")
            return

        # Step 2: Create Trading Orchestrator
        tprint_info("🎯 Step 2: Setting up Trading Orchestrator")

        orchestrator_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'trading_mode': 'paper',
            'account_balance': 10000.0,
            'analyst': {
                'confidence_threshold': 0.6
            },
            'tactician': {
                'confidence_threshold': 0.6,
                'risk_per_trade': 0.02,
                'max_leverage': 3.0
            },
            'signal_combiner': {
                'analyst_weight': 0.6,
                'tactician_weight': 0.4,
                'confidence_threshold': 0.6
            }
        }

        try:
            orchestrator = create_trading_orchestrator(orchestrator_config)
            success = await orchestrator.initialize()

            if success:
                tprint_success("✅ Trading orchestrator initialized")
            else:
                tprint_error("❌ Failed to initialize trading orchestrator")
                return
        except Exception as e:
            tprint_error(f"❌ Failed to create trading orchestrator: {e}")
            return

        # Step 3: Simulate Detailed Trading Operations
        tprint_info("💹 Step 3: Simulating Trading Operations with Full Monitoring")

        # Create realistic market data
        market_data = create_realistic_market_data()

        # Simulate various trading scenarios
        trade_scenarios = [
            {
                'name': 'High Confidence Trend Following',
                'symbol': 'ETHUSDT',
                'action': 'buy',
                'quantity': 0.8,
                'price': 3000.0,
                'confidence': 0.92,
                'regime_type': 'trending_up',
                'models': {
                    'analyst_trend_model': {'confidence': 0.95, 'weight': 0.4},
                    'analyst_momentum_model': {'confidence': 0.88, 'weight': 0.2},
                    'tactician_timing_model': {'confidence': 0.91, 'weight': 0.3},
                    'hmm_regime_model': {'confidence': 0.89, 'weight': 0.1}
                },
                'expected_pnl': 250.0
            },
            {
                'name': 'Medium Confidence Mean Reversion',
                'symbol': 'ETHUSDT',
                'action': 'sell',
                'quantity': 0.4,
                'price': 3050.0,
                'confidence': 0.68,
                'regime_type': 'mean_reversion',
                'models': {
                    'analyst_reversion_model': {'confidence': 0.72, 'weight': 0.5},
                    'tactician_entry_model': {'confidence': 0.64, 'weight': 0.3},
                    'hmm_regime_model': {'confidence': 0.75, 'weight': 0.2}
                },
                'expected_pnl': 80.0
            },
            {
                'name': 'Low Confidence High Risk',
                'symbol': 'ETHUSDT',
                'action': 'buy',
                'quantity': 0.2,
                'price': 2950.0,
                'confidence': 0.52,
                'regime_type': 'high_volatility',
                'models': {
                    'analyst_volatility_model': {'confidence': 0.55, 'weight': 0.6},
                    'tactician_risk_model': {'confidence': 0.48, 'weight': 0.4}
                },
                'expected_pnl': -45.0
            },
            {
                'name': 'Regime Transition Trade',
                'symbol': 'ETHUSDT',
                'action': 'hold',
                'quantity': 0.0,
                'price': 3025.0,
                'confidence': 0.45,
                'regime_type': 'regime_transition',
                'models': {
                    'analyst_uncertainty_model': {'confidence': 0.48, 'weight': 0.5},
                    'tactician_wait_model': {'confidence': 0.42, 'weight': 0.5}
                },
                'expected_pnl': 0.0
            }
        ]

        executed_trade_ids = []

        for i, scenario in enumerate(trade_scenarios, 1):
            tprint_info(f"🎯 Executing Scenario {i}: {scenario['name']}")

            # Record detailed trade
            trade_id = await record_comprehensive_trade(scenario, market_data)
            executed_trade_ids.append(trade_id)

            # Simulate trade execution and outcome
            await simulate_trade_outcome(trade_id, scenario)

            # Brief pause between trades
            await asyncio.sleep(2)

        tprint_success(f"✅ Executed {len(trade_scenarios)} trading scenarios")

        # Step 4: Generate Comprehensive Reports
        tprint_info("📊 Step 4: Generating Comprehensive Reports")

        # Session performance report
        try:
            session_report = await comprehensive_trade_monitor.generate_performance_report("session", "json")
            if session_report:
                tprint_success("✅ Generated session performance report")

                # Display key metrics
                if 'executive_summary' in session_report:
                    tprint_info("📈 Session Performance Summary:")
                    exec_summary = session_report['executive_summary']

                    if 'performance_overview' in exec_summary:
                        perf = exec_summary['performance_overview']
                        tprint_structured({
                            'Total Trades': perf.get('total_trades', 0),
                            'Win Rate': f"{perf.get('win_rate', 0):.1%}",
                            'Total PnL': f"${perf.get('total_pnl', 0):.2f}",
                            'Profit Factor': f"{perf.get('profit_factor', 0):.2f}",
                            'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.3f}",
                            'Max Drawdown': f"{perf.get('max_drawdown', 0):.1%}"
                        }, LogLevel.INFO)

                # Model performance analysis
                if 'model_performance' in session_report:
                    tprint_info("🤖 Model Performance Analysis:")
                    model_perf = session_report['model_performance']['individual_model_performance']

                    for model_id, metrics in model_perf.items():
                        tprint_info(f"  {model_id}:")
                        tprint_info(f"    Usage: {metrics['usage_count']} trades")
                        tprint_info(f"    PnL: ${metrics['total_pnl']:.2f}")
                        tprint_info(f"    Accuracy: {metrics['accuracy']:.1%}")
                        tprint_info(f"    Avg Confidence: {metrics['avg_confidence']:.1%}")
            else:
                tprint_warning("⚠️ No session report generated")
        except Exception as e:
            tprint_error(f"❌ Failed to generate session report: {e}")

        # Step 5: Generate Live Dashboard
        tprint_info("📱 Step 5: Generating Live Dashboard")

        try:
            dashboard = await orchestrator.generate_live_dashboard()
            if dashboard:
                tprint_success("✅ Generated live trading dashboard")

                # Show live metrics
                if 'live_metrics' in dashboard:
                    live_metrics = dashboard['live_metrics']
                    tprint_info("📊 Live Trading Metrics:")

                    if 'current_performance' in live_metrics:
                        current_perf = live_metrics['current_performance']
                        tprint_structured({
                            'Total Trades': current_perf.get('total_trades', 0),
                            'Total PnL': f"${current_perf.get('total_pnl', 0):.2f}",
                            'Win Rate': f"{current_perf.get('win_rate', 0):.1%}",
                            'Current Drawdown': f"{current_perf.get('current_drawdown', 0):.1%}",
                            'Trades/Hour': f"{current_perf.get('trades_per_hour', 0):.1f}"
                        }, LogLevel.INFO)
            else:
                tprint_warning("⚠️ No dashboard generated")
        except Exception as e:
            tprint_error(f"❌ Failed to generate live dashboard: {e}")

        # Step 6: Individual Trade Analysis
        tprint_info("🔍 Step 6: Detailed Individual Trade Analysis")

        try:
            if comprehensive_trade_monitor.completed_trades:
                # Analyze the first completed trade
                trade = comprehensive_trade_monitor.completed_trades[0]

                analysis = await analyze_trade_performance(trade, include_explanations=True)

                if analysis:
                    tprint_success("✅ Completed detailed trade analysis")

                    # Show trade quality
                    if 'trade_quality_score' in analysis:
                        quality = analysis['trade_quality_score']
                        tprint_info(f"📊 Trade Quality Analysis:")
                        tprint_structured({
                            'Overall Score': f"{quality['overall_score']:.3f}",
                            'Quality Grade': quality['quality_grade'],
                            'Classification': quality['trade_classification'],
                            'Performance Score': f"{quality['component_scores']['performance']:.3f}",
                            'Model Score': f"{quality['component_scores']['model_effectiveness']:.3f}",
                            'Risk Score': f"{quality['component_scores']['risk_management']:.3f}",
                            'Timing Score': f"{quality['component_scores']['timing']:.3f}",
                            'Execution Score': f"{quality['component_scores']['execution']:.3f}"
                        }, LogLevel.INFO)

                    # Show feature importance
                    if 'explainability_analysis' in analysis:
                        exp_analysis = analysis['explainability_analysis']
                        if 'feature_consensus' in exp_analysis and 'top_features' in exp_analysis['feature_consensus']:
                            top_features = exp_analysis['feature_consensus']['top_features']
                            tprint_info("🎯 Top Important Features:")
                            for feature, importance in top_features[:5]:
                                tprint_info(f"  {feature}: {importance:.4f}")
                else:
                    tprint_warning("⚠️ No trade analysis generated")
            else:
                tprint_warning("⚠️ No completed trades available for analysis")
        except Exception as e:
            tprint_error(f"❌ Failed to analyze individual trades: {e}")

        # Step 7: Show Monitoring Statistics
        tprint_info("📈 Step 7: Final Monitoring Statistics")

        try:
            monitor_stats = comprehensive_trade_monitor.get_monitor_stats()
            tprint_structured(monitor_stats, LogLevel.INFO)
        except Exception as e:
            tprint_error(f"❌ Failed to get monitoring statistics: {e}")

        tprint_success("🎉 Full Trading Monitoring Demonstration Completed!")
        print("=" * 80)

        # Show file exports
        tprint_info("📁 Generated Files:")
        tprint_info("  📊 Session reports: demo_trading_reports/")
        tprint_info("  📱 Live dashboards: trading_dashboards/")
        tprint_info("  📈 Performance reports: trading_reports/")
        tprint_info("  🔍 Individual trade analysis: Available via API")

    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        raise

    finally:
        # Cleanup
        try:
            if comprehensive_trade_monitor.is_initialized:
                await comprehensive_trade_monitor.stop()

            if 'orchestrator' in locals():
                await orchestrator.stop_trading_session()

        except Exception as e:
            tprint_warning(f"⚠️ Cleanup warning: {e}")

async def record_comprehensive_trade(scenario: Dict[str, Any], market_data: pd.DataFrame) -> str:
    """Record a comprehensive trade with all monitoring features."""
    try:
        # Prepare detailed trade data
        trade_data = {
            'symbol': scenario['symbol'],
            'action': scenario['action'],
            'quantity': scenario['quantity'],
            'price': scenario['price'],
            'confidence': scenario['confidence'],
            'trading_mode': 'paper',
            'exchange': 'binance',
            'analyst_signal': {
                'signal_type': scenario['action'],
                'confidence_score': scenario['confidence'],
                'market_health_score': 0.8,
                'volatility_score': 0.3,
                'liquidation_risk_score': 0.15,
                'feature_importance': {
                    'close': 0.25,
                    'sma_20': 0.18,
                    'rsi': 0.12,
                    'volume': 0.10,
                    'volatility': 0.08
                }
            },
            'tactician_signal': {
                'timing_signal': f"enter_long" if scenario['action'] == 'buy' else 'enter_short' if scenario['action'] == 'sell' else 'hold',
                'confidence_score': scenario['confidence'] * 0.9,
                'position_sizing': {
                    'recommended_size': scenario['quantity'],
                    'leverage': 1.5,
                    'risk_per_trade': 0.02,
                    'kelly_fraction': 0.15,
                    'confidence_multiplier': scenario['confidence']
                },
                'risk_metrics': {
                    'volatility': 0.025,
                    'momentum': 0.15,
                    'volume_trend': 1.2
                }
            },
            'regime_data': {
                'primary_regime': scenario['regime_type'],
                'confidence': 0.82,
                'regime_probabilities': {
                    scenario['regime_type']: 0.75,
                    'secondary_regime': 0.25
                },
                'stability_score': 0.88
            },
            'risk_metrics': {
                'portfolio_risk': 0.025,
                'var_95': 0.035,
                'expected_shortfall': 0.045,
                'max_drawdown_risk': 0.15,
                'volatility_estimate': 0.4
            },
            'position_sizing': {
                'recommended_size': scenario['quantity'],
                'max_size': 1.0,
                'leverage': 1.5,
                'risk_per_trade': 0.02,
                'kelly_fraction': 0.15
            }
        }

        # Prepare models used with mock models for SHAP/LIME
        models_used = {}
        for model_name, model_info in scenario['models'].items():
            models_used[model_name] = {
                'type': model_name.split('_')[0],  # analyst/tactician/hmm
                'model': create_advanced_mock_model(model_name),
                'prediction': model_info['confidence'],
                'confidence': model_info['confidence'],
                'weight': model_info['weight'],
                'version': '1.2.0',
                'features_count': np.random.randint(20, 50),
                'training_date': '2024-01-15'
            }

        # Record the comprehensive trade using the actual function
        trade_id = await record_detailed_trade(trade_data, models_used, market_data)

        if trade_id:
            tprint_success(f"✅ Recorded comprehensive trade: {trade_id}")
            tprint_info(f"   Symbol: {scenario['symbol']}")
            tprint_info(f"   Action: {scenario['action']}")
            tprint_info(f"   Confidence: {scenario['confidence']:.1%}")
            tprint_info(f"   Regime: {scenario['regime_type']}")
            tprint_info(f"   Models Used: {len(models_used)}")
        else:
            tprint_warning("⚠️ Trade recording returned no trade ID")

        return trade_id or ""

    except Exception as e:
        tprint_error(f"❌ Failed to record comprehensive trade: {e}")
        return ""

async def simulate_trade_outcome(trade_id: str, scenario: Dict[str, Any]):
    """Simulate realistic trade outcome with detailed metrics."""
    try:
        # Simulate execution time
        execution_time = np.random.uniform(100, 500)  # 100-500ms
        await asyncio.sleep(execution_time / 1000)

        # Calculate realistic outcome based on scenario
        base_pnl = scenario['expected_pnl']

        # Add some randomness
        actual_pnl = base_pnl + np.random.normal(0, abs(base_pnl) * 0.2)

        # Calculate percentage PnL
        position_value = scenario['quantity'] * scenario['price']
        pnl_percentage = actual_pnl / position_value if position_value > 0 else 0.0

        # Simulate exit price
        exit_price = scenario['price'] * (1 + pnl_percentage)

        # Calculate execution quality based on confidence
        base_execution_quality = scenario['confidence']
        execution_quality = base_execution_quality + np.random.uniform(-0.1, 0.1)
        execution_quality = max(0.0, min(1.0, execution_quality))

        # Simulate other metrics
        slippage = np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2%
        commission = position_value * 0.001  # 0.1%
        duration_minutes = np.random.uniform(5, 45)

        # Maximum excursions
        if actual_pnl > 0:
            max_favorable_excursion = actual_pnl * np.random.uniform(1.1, 1.5)
            max_adverse_excursion = actual_pnl * np.random.uniform(0.1, 0.3)
        else:
            max_favorable_excursion = abs(actual_pnl) * np.random.uniform(0.1, 0.4)
            max_adverse_excursion = abs(actual_pnl) * np.random.uniform(1.0, 1.3)

        # Prepare outcome data
        outcome_data = {
            'exit_price': exit_price,
            'pnl_absolute': actual_pnl,
            'pnl_percentage': pnl_percentage,
            'duration_minutes': duration_minutes,
            'execution_quality': execution_quality,
            'slippage': slippage,
            'commission': commission,
            'timing_quality': scenario['confidence'],
            'max_favorable_excursion': max_favorable_excursion,
            'max_adverse_excursion': max_adverse_excursion
        }

        # Update trade outcome
        if trade_id:
            success = await update_trade_outcome(trade_id, outcome_data)

            if success:
                outcome_emoji = "📈" if actual_pnl > 0 else "📉" if actual_pnl < 0 else "➡️"
                tprint_success(f"✅ {outcome_emoji} Trade completed: {actual_pnl:+.2f} PnL ({pnl_percentage:+.2%})")
            else:
                tprint_warning("⚠️ Failed to update trade outcome")
        else:
            tprint_warning("⚠️ No trade ID available for outcome update")

    except Exception as e:
        tprint_error(f"❌ Failed to simulate trade outcome: {e}")

def create_realistic_market_data() -> pd.DataFrame:
    """Create realistic market data for comprehensive testing."""
    try:
        # Generate 200 candles of realistic market data
        np.random.seed(42)

        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=4), periods=200, freq='1T')

        # Generate realistic price movement with trends and volatility
        base_price = 3000.0
        trend = 0.0001  # Slight upward trend
        volatility = 0.002  # 0.2% volatility

        # Generate correlated price movements
        price_changes = []
        for i in range(200):
            # Add trend component
            trend_component = trend

            # Add volatility component
            vol_component = np.random.normal(0, volatility)

            # Add momentum component (correlation with previous moves)
            momentum_component = 0.0
            if i > 0:
                momentum_component = price_changes[-1] * 0.1  # 10% momentum

            total_change = trend_component + vol_component + momentum_component
            price_changes.append(total_change)

        # Calculate prices
        prices = base_price * np.cumprod(1 + np.array(price_changes))

        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC
            volatility_factor = abs(price_changes[i]) * 2

            high = close * (1 + volatility_factor)
            low = close * (1 - volatility_factor)
            open_price = prices[i-1] if i > 0 else close

            # Ensure OHLC consistency
            high = max(open_price, close, high)
            low = min(open_price, close, low)

            # Generate volume with correlation to price movement
            base_volume = 500
            volume_factor = 1 + abs(price_changes[i]) * 10  # Higher volume on big moves
            volume = base_volume * volume_factor * np.random.uniform(0.8, 1.2)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)

        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    except Exception as e:
        tprint_error(f"❌ Failed to create market data: {e}")
        return pd.DataFrame()

def create_advanced_mock_model(model_name: str):
    """Create advanced mock model with SHAP/LIME capabilities."""
    class AdvancedMockModel:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.model_type = model_name.split('_')[0]  # analyst/tactician/hmm

            # Define realistic feature names based on model type
            if 'analyst' in model_name:
                self.feature_names = [
                    'close', 'sma_20', 'sma_50', 'rsi', 'macd', 'bb_position',
                    'volume', 'volatility_20', 'returns_1', 'returns_5', 'returns_20',
                    'volume_ratio', 'price_momentum', 'trend_strength'
                ]
            elif 'tactician' in model_name:
                self.feature_names = [
                    'close', 'volume', 'volatility_20', 'rsi', 'macd',
                    'bb_position', 'support_level', 'resistance_level',
                    'momentum_score', 'timing_indicator'
                ]
            else:  # hmm
                self.feature_names = [
                    'returns_1', 'returns_5', 'volatility_5', 'volatility_20',
                    'volume_ratio', 'regime_features'
                ]

        def predict(self, X):
            """Mock prediction method."""
            if hasattr(X, '__len__'):
                return np.random.uniform(0.4, 0.9, len(X))
            else:
                return np.random.uniform(0.4, 0.9)

        def predict_proba(self, X):
            """Mock probability prediction."""
            pred = self.predict(X)
            if hasattr(pred, '__len__'):
                return [[1-p, p] for p in pred]
            else:
                return [[1-pred, pred]]

        def get_feature_importance(self):
            """Mock feature importance for SHAP."""
            importance = np.random.exponential(0.1, len(self.feature_names))
            importance = importance / importance.sum()  # Normalize
            return dict(zip(self.feature_names, importance))

    return AdvancedMockModel(model_name)

# Run the demonstration
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tprint_info("⏹️ Demonstration stopped by user")
    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        import traceback

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None
        traceback.print_exc()

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
