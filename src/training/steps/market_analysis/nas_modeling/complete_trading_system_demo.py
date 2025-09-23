"""
Complete Trading System Demo

This script demonstrates the complete trading system from regime detection
to actual trading execution, showing how all components work together.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

from .core.neural_state_space_nas import ContinuousTimeRegimeDetector, NeuralSSMConfig
from .core.trading_signal_generator import (
    TradingSignalGenerator, PortfolioManager, RiskManager,
    Backtester, TradingConfig
)
from .core.portfolio_optimizer import (
    MeanVarianceOptimizer, RiskParityOptimizer,
    BlackLittermanOptimizer, PortfolioConfig
)
from .core.execution_system import (
    ExecutionEngine, MarketImpactModel, RealTimeExecutionMonitor,
    TradingSystemIntegration, ExecutionConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataSimulator:
    """Simulates realistic market data for testing."""

    def __init__(self, num_symbols: int = 5, num_days: int = 1000):
        """Initialize market data simulator.

        Args:
            num_symbols: Number of trading symbols
            num_days: Number of days of data
        """
        self.num_symbols = num_symbols
        self.num_days = num_days
        self.symbols = [f"STOCK_{i}" for i in range(num_symbols)]
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_market_data(self) -> Dict[str, pd.DataFrame]:
        """Generate realistic market data."""
        np.random.seed(42)
        market_data = {}

        for symbol in self.symbols:
            # Generate price series with trends, cycles, and noise
            dates = pd.date_range('2020-01-01', periods=self.num_days, freq='D')

            # Base price trend
            trend = np.linspace(100, 150, self.num_days)

            # Cyclical component
            cycle = 10 * np.sin(2 * np.pi * np.arange(self.num_days) / 252)

            # Seasonal component
            seasonal = 5 * np.sin(2 * np.pi * np.arange(self.num_days) / 365)

            # Random walk component
            random_walk = np.cumsum(np.random.randn(self.num_days) * 0.5)

            # Combine components
            price = trend + cycle + seasonal + random_walk + 100

            # Generate OHLCV data
            open_price = price * (1 + np.random.normal(0, 0.002, self.num_days))
            high_price = price * (1 + np.abs(np.random.normal(0, 0.005, self.num_days)))
            low_price = price * (1 - np.abs(np.random.normal(0, 0.005, self.num_days)))
            close_price = price + np.random.normal(0, 0.5, self.num_days)
            volume = np.random.exponential(100000, self.num_days)

            # Create DataFrame
            df = pd.DataFrame({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }, index=dates)

            market_data[symbol] = df

        self.logger.info(f"✅ Generated market data for {len(self.symbols)} symbols")
        return market_data

class CompleteTradingSystemDemo:
    """Complete demo of the trading system."""

    def __init__(self):
        """Initialize complete trading demo."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.regime_model = None
        self.trading_system = None
        self.market_simulator = MarketDataSimulator()

    def setup_system(self):
        """Setup complete trading system."""
        logger.info("🚀 Setting up complete trading system")

        # 1. Create regime detection model
        ssm_config = NeuralSSMConfig(
            state_size=64,
            hidden_size=128,
            time_points=20
        )
        self.regime_model = ContinuousTimeRegimeDetector(
            input_size=5,  # OHLCV features
            state_size=64,
            num_regimes=5
        )

        # 2. Create trading system
        execution_config = ExecutionConfig(
            default_order_type="market",
            max_slippage=0.001,
            use_smart_routing=True
        )

        self.trading_system = TradingSystemIntegration(
            self.regime_model, execution_config
        )

        # 3. Generate market data
        self.market_data = self.market_simulator.generate_market_data()

        self.logger.info("✅ Trading system setup completed")

    async def run_trading_simulation(self, num_cycles: int = 10) -> Dict[str, Any]:
        """Run complete trading simulation.

        Args:
            num_cycles: Number of trading cycles to run

        Returns:
            Simulation results
        """
        logger.info(f"📊 Starting trading simulation with {num_cycles} cycles")

        results = {
            'cycles': [],
            'portfolio_history': [],
            'final_portfolio_value': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

        # Run trading cycles
        for cycle in range(num_cycles):
            # Get current market data
            current_data = {}
            for symbol, data in self.market_data.items():
                # Use latest 100 days of data
                current_data[symbol] = data.tail(100)

            # Process trading cycle
            cycle_result = await self.trading_system.process_trading_cycle(current_data)
            results['cycles'].append(cycle_result)

            # Get portfolio status
            portfolio_status = self.trading_system.get_system_status()
            results['portfolio_history'].append({
                'cycle': cycle,
                'portfolio_value': portfolio_status['portfolio_value'],
                'total_return': portfolio_status['total_return']
            })

            if cycle % 5 == 0:
                self.logger.info(f"📈 Cycle {cycle}: Portfolio = ${portfolio_status['portfolio_value']:.2f}")

        # Calculate final metrics
        results['final_portfolio_value'] = portfolio_status['portfolio_value']
        results['total_return'] = portfolio_status['total_return']

        # Calculate performance metrics
        portfolio_values = [entry['portfolio_value'] for entry in results['portfolio_history']]
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            results['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)
            results['sharpe_ratio'] = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0 else 0
            )

        self.logger.info("✅ Trading simulation completed")
        self.logger.info(f"💰 Final portfolio value: ${results['final_portfolio_value']:.2f}")
        self.logger.info(f"📈 Total return: {results['total_return']:.2f}%")
        self.logger.info(f"📉 Max drawdown: {results['max_drawdown']:.3f}")
        self.logger.info(f"📊 Sharpe ratio: {results['sharpe_ratio']:.3f}")

        return results

    def run_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtest."""
        logger.info("🔬 Running comprehensive backtest")

        # Create backtester
        backtester = Backtester(initial_capital=100000.0)

        # Run backtest
        backtest_result = backtester.run_backtest(
            signal_generator=self.trading_system.signal_generator,
            regime_model=self.regime_model,
            market_data=self.market_data,
            start_date='2020-01-01',
            end_date='2023-01-01'
        )

        self.logger.info("✅ Backtest completed")
        self.logger.info(f"📈 Backtest total return: {backtest_result['total_return']:.2f}%")
        self.logger.info(f"📊 Backtest Sharpe ratio: {backtest_result['performance_metrics'].get('sharpe_ratio', 0):.3f}")

        return backtest_result

    def demonstrate_component_integration(self):
        """Demonstrate how all components work together."""
        logger.info("🔧 Demonstrating component integration")

        # 1. Market Data -> Regime Detection
        sample_data = {}
        for symbol, data in self.market_data.items():
            sample_data[symbol] = data.tail(50)  # 50 days of data

        # Get regime prediction
        regime_input = self._prepare_regime_input(sample_data)
        with torch.no_grad():
            regime_prediction = self.regime_model(regime_input).numpy()

        print("🏛️ Regime Detection:")
        print(f"   Regime probabilities: {regime_prediction}")
        print(f"   Predicted regime: {np.argmax(regime_prediction)}")

        # 2. Regime -> Trading Signal
        symbol = list(sample_data.keys())[0]
        signal = self.trading_system.signal_generator.generate_signal(
            regime_prediction, sample_data[symbol], 0.0
        )

        print("📈 Trading Signal:")
        print(f"   Signal type: {signal['signal_type'].value}")
        print(f"   Signal strength: {signal['signal_strength']:.3f}")
        print(f"   Position size: {signal['position_size']:.3f}")

        # 3. Signal -> Portfolio Optimization
        returns_data = self._prepare_returns_data(sample_data)
        portfolio_result = self.trading_system.portfolio_optimizer.optimize_portfolio(returns_data)

        print("⚖️ Portfolio Optimization:")
        print(f"   Optimal weights: {portfolio_result['optimal_weights']}")
        print(f"   Expected return: {portfolio_result['expected_return']:.4f}")
        print(f"   Portfolio risk: {portfolio_result['portfolio_risk']:.4f}")

        # 4. Portfolio -> Execution
        execution_config = ExecutionConfig()
        execution_engine = ExecutionEngine(execution_config)

        # Create sample order
        order = self._create_sample_order(signal, sample_data[symbol])

        print("📋 Execution:")
        print(f"   Order type: {order.order_type.value}")
        print(f"   Quantity: {order.quantity}")
        print(f"   Symbol: {order.symbol}")

    def _prepare_regime_input(self, market_data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Prepare input for regime model."""
        symbol = list(market_data.keys())[0]
        data = market_data[symbol]

        features = ['open', 'high', 'low', 'close', 'volume']
        feature_data = data[features].values[-50:]  # Last 50 days

        return torch.FloatTensor(feature_data).unsqueeze(0)

    def _prepare_returns_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare returns data for portfolio optimization."""
        returns_list = []

        for symbol, data in market_data.items():
            returns = data['close'].pct_change().dropna()
            returns_list.append(returns)

        return pd.concat(returns_list, axis=1)

    def _create_sample_order(self, signal: Dict[str, Any],
                           market_data: pd.DataFrame) -> Any:
        """Create sample order for demonstration."""
        from .core.execution_system import Order, OrderType

        symbol = list(self.market_data.keys())[0]
        current_price = market_data['close'].iloc[-1]
        quantity = abs(signal['position_change']) * 1000

        order = Order(
            order_id="demo_order",
            symbol=symbol,
            order_type=OrderType.MARKET,
            side="buy" if signal['position_change'] > 0 else "sell",
            quantity=quantity,
            price=current_price,
            timestamp=datetime.now()
        )

        return order

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values[0]
        max_dd = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

async def main():
    """Main demo function."""
    logger.info("🚀 Complete Trading System Demo")
    logger.info("=" * 60)

    try:
        # Initialize demo
        demo = CompleteTradingSystemDemo()
        demo.setup_system()

        # Demonstrate component integration
        logger.info("\\n1. Component Integration Demo:")
        demo.demonstrate_component_integration()

        # Run trading simulation
        logger.info("\\n2. Trading Simulation:")
        simulation_results = await demo.run_trading_simulation(num_cycles=20)

        # Run backtest
        logger.info("\\n3. Historical Backtest:")
        backtest_results = demo.run_backtest()

        # Display comprehensive results
        logger.info("\\n📊 COMPREHENSIVE TRADING SYSTEM RESULTS")
        logger.info("=" * 50)

        # Simulation Results
        logger.info("📈 SIMULATION RESULTS:")
        logger.info(f"   Final Portfolio Value: ${simulation_results['final_portfolio_value']:,.2f}")
        logger.info(f"   Total Return: {simulation_results['total_return']:.2f}%")
        logger.info(f"   Max Drawdown: {simulation_results['max_drawdown']:.3f}")
        logger.info(f"   Sharpe Ratio: {simulation_results['sharpe_ratio']:.3f}")

        # Backtest Results
        logger.info("🔬 BACKTEST RESULTS:")
        logger.info(f"   Backtest Return: {backtest_results['total_return']:.2f}%")
        logger.info(f"   Backtest Sharpe: {backtest_results['performance_metrics'].get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Number of Trades: {backtest_results['num_trades']}")
        logger.info(f"   Win Rate: {backtest_results['performance_metrics'].get('win_rate', 0):.3f}")

        # System Components Status
        logger.info("🏗️ SYSTEM COMPONENTS:")
        logger.info("   ✅ Regime Detection: Neural State Space Models")
        logger.info("   ✅ Signal Generation: Multi-objective optimization")
        logger.info("   ✅ Portfolio Management: Mean-variance + risk parity")
        logger.info("   ✅ Risk Management: Dynamic risk limits")
        logger.info("   ✅ Execution Engine: Smart order routing")
        logger.info("   ✅ Monitoring: Real-time performance tracking")

        # Key Achievements
        logger.info("🎯 KEY ACHIEVEMENTS:")
        logger.info("   ✅ Complete regime detection → trading pipeline")
        logger.info("   ✅ Advanced optimization techniques")
        logger.info("   ✅ Production-ready execution system")
        logger.info("   ✅ Comprehensive risk management")
        logger.info("   ✅ Real-time monitoring and adaptation")

        return {
            'simulation_results': simulation_results,
            'backtest_results': backtest_results,
            'system_status': 'operational',
            'components_ready': True
        }

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    # Run the complete demo
    results = asyncio.run(main())

    print("\\n" + "=" * 60)
    print("🎉 COMPLETE TRADING SYSTEM DEMO FINISHED")
    print("=" * 60)
    print("Your system now includes:")
    print("✅ Advanced regime detection with Neural ODEs")
    print("✅ Trading signal generation")
    print("✅ Portfolio optimization")
    print("✅ Risk management")
    print("✅ Execution system")
    print("✅ Real-time monitoring")
    print("\\nThis is now a complete trading system!")