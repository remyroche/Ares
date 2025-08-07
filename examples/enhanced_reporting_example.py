#!/usr/bin/env python3
"""
Enhanced Reporting Example

This example demonstrates how to use the enhanced reporting system
for paper trading and backtesting with comprehensive metrics.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.launcher.enhanced_trading_launcher import setup_enhanced_trading_launcher
from src.config.enhanced_reporting_config import get_enhanced_reporting_config


async def example_paper_trading():
    """Example of paper trading with enhanced reporting."""
    print("🚀 Starting Paper Trading Example with Enhanced Reporting...")
    
    # Get configuration
    config = get_enhanced_reporting_config()
    
    # Setup enhanced trading launcher
    launcher = await setup_enhanced_trading_launcher(config)
    if not launcher:
        print("❌ Failed to setup enhanced trading launcher")
        return
    
    # Launch paper trading
    success = await launcher.launch_paper_trading()
    if not success:
        print("❌ Failed to launch paper trading")
        return
    
    print("✅ Paper trading launched successfully")
    
    # Execute some example trades with detailed metadata
    trades = [
        {
            "symbol": "ETHUSDT",
            "side": "buy",
            "quantity": 1.0,
            "price": 2000.0,
            "timestamp": datetime.now(),
            "trade_metadata": {
                "exchange": "paper",
                "leverage": 1.0,
                "duration": "swing",
                "strategy": "momentum",
                "order_type": "market",
                "portfolio_percentage": 0.1,
                "risk_percentage": 0.02,
                "max_position_size": 0.1,
                "position_ranking": 1,
                "execution_quality": 0.95,
                "risk_metrics": {
                    "var_95": 0.02,
                    "expected_shortfall": 0.03,
                },
                "notes": "Momentum breakout trade",
                "market_indicators": {
                    "rsi": 65.5,
                    "macd": 0.002,
                    "macd_signal": 0.001,
                    "bollinger_upper": 2050.0,
                    "bollinger_lower": 1950.0,
                    "bollinger_middle": 2000.0,
                    "atr": 50.0,
                    "volume_sma": 1000000.0,
                    "price_sma_20": 1980.0,
                    "price_sma_50": 1950.0,
                    "price_sma_200": 1900.0,
                    "volatility": 0.025,
                    "momentum": 0.02,
                    "support_level": 1950.0,
                    "resistance_level": 2050.0,
                },
                "market_health": {
                    "overall_health_score": 0.75,
                    "volatility_regime": "medium",
                    "liquidity_score": 0.8,
                    "stress_score": 0.3,
                    "market_strength": 0.7,
                    "volume_health": "healthy",
                    "price_trend": "uptrend",
                    "market_regime": "bullish",
                },
                "ml_confidence": {
                    "analyst_confidence": 0.8,
                    "tactician_confidence": 0.75,
                    "ensemble_confidence": 0.78,
                    "meta_learner_confidence": 0.82,
                    "individual_model_confidences": {
                        "xgboost": 0.8,
                        "lstm": 0.75,
                        "random_forest": 0.78,
                    },
                    "ensemble_agreement": 0.85,
                    "model_diversity": 0.7,
                    "prediction_consistency": 0.8,
                },
            },
        },
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 45000.0,
            "timestamp": datetime.now() + timedelta(hours=1),
            "trade_metadata": {
                "exchange": "paper",
                "leverage": 1.0,
                "duration": "day_trading",
                "strategy": "breakout",
                "order_type": "limit",
                "portfolio_percentage": 0.05,
                "risk_percentage": 0.01,
                "max_position_size": 0.05,
                "position_ranking": 2,
                "execution_quality": 0.9,
                "risk_metrics": {
                    "var_95": 0.015,
                    "expected_shortfall": 0.02,
                },
                "notes": "Breakout trade on BTC",
                "market_indicators": {
                    "rsi": 70.0,
                    "macd": 0.005,
                    "macd_signal": 0.002,
                    "bollinger_upper": 46000.0,
                    "bollinger_lower": 44000.0,
                    "bollinger_middle": 45000.0,
                    "atr": 1000.0,
                    "volume_sma": 5000000.0,
                    "price_sma_20": 44800.0,
                    "price_sma_50": 44500.0,
                    "price_sma_200": 44000.0,
                    "volatility": 0.022,
                    "momentum": 0.015,
                    "support_level": 44000.0,
                    "resistance_level": 46000.0,
                },
                "market_health": {
                    "overall_health_score": 0.8,
                    "volatility_regime": "low",
                    "liquidity_score": 0.9,
                    "stress_score": 0.2,
                    "market_strength": 0.8,
                    "volume_health": "very_healthy",
                    "price_trend": "strong_uptrend",
                    "market_regime": "very_bullish",
                },
                "ml_confidence": {
                    "analyst_confidence": 0.85,
                    "tactician_confidence": 0.8,
                    "ensemble_confidence": 0.83,
                    "meta_learner_confidence": 0.87,
                    "individual_model_confidences": {
                        "xgboost": 0.85,
                        "lstm": 0.8,
                        "random_forest": 0.83,
                    },
                    "ensemble_agreement": 0.9,
                    "model_diversity": 0.75,
                    "prediction_consistency": 0.85,
                },
            },
        },
    ]
    
    # Execute trades
    for trade in trades:
        success = await launcher.execute_trade(
            symbol=trade["symbol"],
            side=trade["side"],
            quantity=trade["quantity"],
            price=trade["price"],
            timestamp=trade["timestamp"],
            trade_metadata=trade["trade_metadata"],
        )
        
        if success:
            print(f"✅ Executed {trade['side']} trade: {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        else:
            print(f"❌ Failed to execute trade: {trade['symbol']}")
    
    # Get performance metrics
    metrics = launcher.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Total PnL: ${metrics.get('total_pnl', 0):.2f}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Get portfolio summary
    portfolio = launcher.get_portfolio_summary()
    print(f"\n💼 Portfolio Summary:")
    print(f"Total Value: ${portfolio.get('total_value', 0):.2f}")
    print(f"Positions Count: {portfolio.get('positions_count', 0)}")
    
    # Generate comprehensive report
    report = await launcher.generate_comprehensive_report("example")
    print(f"\n📋 Generated comprehensive report with {len(report)} sections")
    
    # Stop launcher
    await launcher.stop()
    print("✅ Paper trading example completed")


async def example_backtesting():
    """Example of backtesting with enhanced reporting."""
    print("\n🚀 Starting Backtesting Example with Enhanced Reporting...")
    
    # Get configuration
    config = get_enhanced_reporting_config()
    
    # Setup enhanced trading launcher
    launcher = await setup_enhanced_trading_launcher(config)
    if not launcher:
        print("❌ Failed to setup enhanced trading launcher")
        return
    
    # Generate sample historical data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    historical_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(200, 300, len(dates)),
        'low': np.random.uniform(50, 150, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.uniform(1000000, 5000000, len(dates)),
    }, index=dates)
    
    # Generate sample strategy signals
    strategy_signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], len(dates), p=[0.3, 0.4, 0.3]),
        'close': historical_data['close'],
        'symbol': ['ETHUSDT'] * len(dates),
    }, index=dates)
    
    # Add some trend to make it more realistic
    trend = np.cumsum(np.random.normal(0, 0.01, len(dates)))
    strategy_signals['close'] = 150 + trend * 50
    
    # Backtest configuration
    backtest_config = {
        "strategy": "example_strategy",
        "leverage": 1.0,
        "duration": "backtest",
        "order_type": "market",
        "portfolio_percentage": 0.1,
        "risk_percentage": 0.02,
        "max_position_size": 0.1,
        "position_ranking": 1,
        "execution_quality": 0.95,
        "risk_metrics": {
            "var_95": 0.02,
            "expected_shortfall": 0.03,
        },
        "notes": "Example backtest strategy",
        "market_indicators": {
            "rsi": 65.5,
            "macd": 0.002,
            "macd_signal": 0.001,
            "bollinger_upper": 2050.0,
            "bollinger_lower": 1950.0,
            "bollinger_middle": 2000.0,
            "atr": 50.0,
            "volume_sma": 1000000.0,
            "price_sma_20": 1980.0,
            "price_sma_50": 1950.0,
            "price_sma_200": 1900.0,
            "volatility": 0.025,
            "momentum": 0.02,
            "support_level": 1950.0,
            "resistance_level": 2050.0,
        },
        "market_health": {
            "overall_health_score": 0.75,
            "volatility_regime": "medium",
            "liquidity_score": 0.8,
            "stress_score": 0.3,
            "market_strength": 0.7,
            "volume_health": "healthy",
            "price_trend": "uptrend",
            "market_regime": "bullish",
        },
        "ml_confidence": {
            "analyst_confidence": 0.8,
            "tactician_confidence": 0.75,
            "ensemble_confidence": 0.78,
            "meta_learner_confidence": 0.82,
            "individual_model_confidences": {
                "xgboost": 0.8,
                "lstm": 0.75,
                "random_forest": 0.78,
            },
            "ensemble_agreement": 0.85,
            "model_diversity": 0.7,
            "prediction_consistency": 0.8,
        },
    }
    
    # Launch backtest
    results = await launcher.launch_backtest(
        historical_data=historical_data,
        strategy_signals=strategy_signals,
        backtest_config=backtest_config,
    )
    
    if results:
        print("✅ Backtest completed successfully")
        
        # Display results
        performance_metrics = results.get("performance_metrics", {})
        print(f"\n📊 Backtest Results:")
        print(f"Total Trades: {performance_metrics.get('total_trades', 0)}")
        print(f"Total PnL: ${performance_metrics.get('total_pnl', 0):.2f}")
        print(f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}")
        print(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
        print(f"Total Return: {performance_metrics.get('total_return', 0):.2%}")
        
        # Get trade history
        trade_history = launcher.get_trade_history()
        print(f"\n📈 Trade History: {len(trade_history)} trades")
        
        # Get portfolio summary
        portfolio = launcher.get_portfolio_summary()
        print(f"\n💼 Final Portfolio Value: ${portfolio.get('final_portfolio_value', 0):.2f}")
        
    else:
        print("❌ Backtest failed")
    
    # Stop launcher
    await launcher.stop()
    print("✅ Backtesting example completed")


async def main():
    """Main function to run examples."""
    print("🎯 Enhanced Reporting System Examples")
    print("=" * 50)
    
    try:
        # Run paper trading example
        await example_paper_trading()
        
        # Run backtesting example
        await example_backtesting()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())