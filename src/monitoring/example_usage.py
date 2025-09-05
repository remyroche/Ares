#!/usr/bin/env python3
"""
Example Usage of Enhanced ML Monitoring System

This example demonstrates how to use the enhanced monitoring system
for comprehensive ML model and ensemble monitoring with SHAP/LIME explanations.
"""

import asyncio
import json
from datetime import datetime

from .monitoring_orchestrator import (
    MonitoringOrchestrator, create_monitoring_orchestrator,
    TradeContext, TradingIndicator, MLModelDecision, EnsembleDecision,
    TradeDecision, TradingMode, ModelType, ModelPerformanceMetrics,
    EnsemblePerformanceMetrics
)


async def example_enhanced_monitoring():
    """Example of using the enhanced monitoring system."""
    
    # Load configuration
    config = {
        "enhanced_monitoring": {
            "enable_monitoring": True,
            "enable_explanations": True,
            "enable_ensemble_monitoring": True,
            "enable_csv_export": True,
            "export_interval_days": 1,  # Export daily for demo
            "max_memory_decisions": 1000,
            "export_directory": "example_monitoring_exports"
        },
        "enhanced_ml_monitoring": {
            "enable_shap": True,
            "enable_lime": True,
            "csv_export_interval_days": 1,
            "max_decisions_in_memory": 1000,
            "export_directory": "example_monitoring_exports"
        },
        "explainability_integration": {
            "enable_shap": True,
            "enable_lime": True,
            "max_features_explained": 10,
            "explanation_cache_size": 100
        },
        "ensemble_monitoring": {
            "weight_update_frequency_hours": 1,  # Update hourly for demo
            "performance_window_days": 7,
            "min_weight_threshold": 0.05,
            "max_weight_threshold": 0.7
        },
        "csv_export": {
            "export_directory": "example_monitoring_exports",
            "include_raw_data": True,
            "include_summary_stats": True,
            "decimal_precision": 4
        }
    }
    
    # Create and initialize monitoring orchestrator
    print("🚀 Initializing Enhanced ML Monitoring System...")
    orchestrator = await create_monitoring_orchestrator(config)
    
    if not orchestrator:
        print("❌ Failed to initialize monitoring orchestrator")
        return
    
    print("✅ Enhanced ML Monitoring System initialized successfully!")
    
    # Example 1: Record a trade decision
    print("\n📊 Recording example trade decision...")
    
    # Create trade context
    context = TradeContext(
        exchange="binance",
        token="BTCUSDT",
        timestamp=datetime.now(),
        price=45000.0,
        volume=0.1,
        timeframe="1h",
        regime="bullish",
        market_conditions={
            "volatility": 0.15,
            "trend": "upward",
            "volume_profile": "high"
        }
    )
    
    # Create trading indicators
    trading_indicators = [
        TradingIndicator(
            name="RSI",
            value=65.5,
            weight=0.3,
            confidence=0.8,
            risk_score=0.2,
            description="Relative Strength Index indicating overbought conditions"
        ),
        TradingIndicator(
            name="MACD",
            value=125.3,
            weight=0.4,
            confidence=0.9,
            risk_score=0.1,
            description="MACD showing bullish momentum"
        ),
        TradingIndicator(
            name="Bollinger Bands",
            value=0.85,
            weight=0.3,
            confidence=0.7,
            risk_score=0.3,
            description="Price near upper Bollinger Band"
        )
    ]
    
    # Create individual model decisions
    model_decisions = [
        MLModelDecision(
            model_id="hmm_model_1",
            model_type=ModelType.HMM,
            prediction=0.75,
            confidence=0.85,
            risk_score=0.2,
            feature_importance={
                "price_momentum": 0.4,
                "volume_trend": 0.3,
                "volatility": 0.2,
                "regime_stability": 0.1
            },
            processing_time_ms=15.2,
            model_version="v2.1"
        ),
        MLModelDecision(
            model_id="analyst_model_1",
            model_type=ModelType.ANALYST,
            prediction=0.68,
            confidence=0.78,
            risk_score=0.25,
            feature_importance={
                "technical_indicators": 0.5,
                "market_sentiment": 0.3,
                "fundamental_analysis": 0.2
            },
            processing_time_ms=22.1,
            model_version="v1.5"
        ),
        MLModelDecision(
            model_id="tactician_model_1",
            model_type=ModelType.TACTICIAN,
            prediction=0.82,
            confidence=0.92,
            risk_score=0.15,
            feature_importance={
                "entry_timing": 0.4,
                "risk_management": 0.3,
                "position_sizing": 0.3
            },
            processing_time_ms=8.7,
            model_version="v3.0"
        )
    ]
    
    # Create ensemble decision
    ensemble_decision = EnsembleDecision(
        ensemble_id="main_ensemble",
        final_prediction=0.75,
        final_confidence=0.85,
        final_risk_score=0.2,
        model_weights={
            "hmm_model_1": 0.4,
            "analyst_model_1": 0.35,
            "tactician_model_1": 0.25
        },
        model_decisions=model_decisions,
        voting_mechanism="weighted_average",
        consensus_score=0.8,
        disagreement_level=0.15
    )
    
    # Create complete trade decision
    trade_decision = TradeDecision(
        decision_id="trade_001",
        context=context,
        trading_mode=TradingMode.PAPER,
        timestamp=datetime.now(),
        trading_indicators=trading_indicators,
        overall_confidence=0.82,
        overall_risk_score=0.18,
        ensemble_decision=ensemble_decision,
        action="buy",
        position_size=0.1,
        stop_loss=43000.0,
        take_profit=48000.0,
        execution_time_ms=45.3
    )
    
    # Record the trade decision
    await orchestrator.record_trade_decision(trade_decision)
    print("✅ Trade decision recorded successfully!")
    
    # Example 2: Record model performance metrics
    print("\n📈 Recording model performance metrics...")
    
    model_performance = ModelPerformanceMetrics(
        model_id="hmm_model_1",
        model_type=ModelType.HMM,
        timestamp=datetime.now(),
        accuracy=0.78,
        precision=0.82,
        recall=0.75,
        f1_score=0.78,
        auc_score=0.85,
        win_rate=0.72,
        profit_factor=1.45,
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        prediction_confidence_std=0.12,
        feature_importance_stability=0.85,
        concept_drift_score=0.15,
        data_drift_score=0.08
    )
    
    await orchestrator.enhanced_monitor.record_model_performance(model_performance)
    print("✅ Model performance metrics recorded!")
    
    # Example 3: Update ensemble weights
    print("\n⚖️ Updating ensemble weights...")
    
    model_performances = {
        "hmm_model_1": {
            "accuracy": 0.78,
            "win_rate": 0.72,
            "profit_factor": 1.45,
            "sharpe_ratio": 1.2
        },
        "analyst_model_1": {
            "accuracy": 0.75,
            "win_rate": 0.68,
            "profit_factor": 1.32,
            "sharpe_ratio": 1.1
        },
        "tactician_model_1": {
            "accuracy": 0.82,
            "win_rate": 0.78,
            "profit_factor": 1.58,
            "sharpe_ratio": 1.35
        }
    }
    
    current_weights = {
        "hmm_model_1": 0.4,
        "analyst_model_1": 0.35,
        "tactician_model_1": 0.25
    }
    
    new_weights = await orchestrator.update_ensemble_weights(
        "main_ensemble", model_performances, current_weights
    )
    
    print(f"✅ Ensemble weights updated: {new_weights}")
    
    # Example 4: Get ensemble analysis
    print("\n🔍 Getting ensemble analysis...")
    
    analysis = await orchestrator.get_ensemble_analysis("main_ensemble")
    print(f"✅ Ensemble analysis: {json.dumps(analysis, indent=2, default=str)}")
    
    # Example 5: Force export monitoring data
    print("\n📤 Exporting monitoring data...")
    
    export_success = await orchestrator.export_monitoring_data()
    if export_success:
        print("✅ Monitoring data exported successfully!")
    else:
        print("❌ Failed to export monitoring data")
    
    # Example 6: Get comprehensive statistics
    print("\n📊 Getting comprehensive monitoring statistics...")
    
    stats = orchestrator.get_comprehensive_stats()
    print(f"✅ Monitoring statistics: {json.dumps(stats, indent=2, default=str)}")
    
    # Example 7: Simulate multiple trade decisions
    print("\n🔄 Simulating multiple trade decisions...")
    
    for i in range(5):
        # Create a simple trade decision
        simple_context = TradeContext(
            exchange="binance",
            token="ETHUSDT",
            timestamp=datetime.now(),
            price=3000.0 + i * 10,
            volume=0.05,
            timeframe="1h"
        )
        
        simple_ensemble = EnsembleDecision(
            ensemble_id="main_ensemble",
            final_prediction=0.6 + i * 0.05,
            final_confidence=0.7 + i * 0.02,
            final_risk_score=0.3 - i * 0.02,
            model_weights={"hmm_model_1": 0.5, "analyst_model_1": 0.5},
            model_decisions=[],
            voting_mechanism="weighted_average",
            consensus_score=0.8,
            disagreement_level=0.1
        )
        
        simple_decision = TradeDecision(
            decision_id=f"trade_{i+2:03d}",
            context=simple_context,
            trading_mode=TradingMode.PAPER,
            timestamp=datetime.now(),
            trading_indicators=[],
            overall_confidence=0.7 + i * 0.02,
            overall_risk_score=0.3 - i * 0.02,
            ensemble_decision=simple_ensemble,
            action="buy" if i % 2 == 0 else "sell",
            position_size=0.05
        )
        
        await orchestrator.record_trade_decision(simple_decision)
    
    print("✅ Multiple trade decisions recorded!")
    
    # Final export
    print("\n📤 Final export of all monitoring data...")
    await orchestrator.export_monitoring_data()
    
    # Shutdown
    print("\n🛑 Shutting down monitoring system...")
    await orchestrator.shutdown()
    
    print("✅ Enhanced ML Monitoring example completed successfully!")


async def example_trading_system_integration():
    """Example of integrating monitoring with trading systems."""
    
    print("\n🔗 Trading System Integration Example")
    
    # Mock trading system classes
    class MockBacktestingSystem:
        async def execute_trade(self, **kwargs):
            print(f"Backtesting: Executing trade with {kwargs}")
            return {"profit_loss": 100.0, "execution_price": 45000.0}
        
        async def get_prediction(self, **kwargs):
            print(f"Backtesting: Getting prediction with {kwargs}")
            return 0.75
    
    class MockPaperTradingSystem:
        async def execute_trade(self, **kwargs):
            print(f"Paper Trading: Executing trade with {kwargs}")
            return {"profit_loss": 50.0, "execution_price": 45000.0}
    
    class MockLiveTradingSystem:
        async def execute_trade(self, **kwargs):
            print(f"Live Trading: Executing trade with {kwargs}")
            return {"profit_loss": 200.0, "execution_price": 45000.0}
    
    # Create monitoring orchestrator
    config = {
        "enhanced_monitoring": {
            "enable_monitoring": True,
            "enable_backtesting_integration": True,
            "enable_paper_trading_integration": True,
            "enable_live_trading_integration": True
        }
    }
    
    orchestrator = await create_monitoring_orchestrator(config)
    if not orchestrator:
        print("❌ Failed to initialize monitoring orchestrator")
        return
    
    # Integrate with trading systems
    backtesting_system = MockBacktestingSystem()
    paper_trading_system = MockPaperTradingSystem()
    live_trading_system = MockLiveTradingSystem()
    
    # Integrate systems
    await orchestrator.integrate_trading_system(backtesting_system, "backtesting")
    await orchestrator.integrate_trading_system(paper_trading_system, "paper_trading")
    await orchestrator.integrate_trading_system(live_trading_system, "live_trading")
    
    print("✅ Trading systems integrated with monitoring!")
    
    # Simulate trades (these will be automatically monitored)
    print("\n🔄 Simulating monitored trades...")
    
    # Backtesting trade
    await backtesting_system.execute_trade(
        exchange="binance",
        token="BTCUSDT",
        price=45000.0,
        volume=0.1,
        action="buy",
        prediction=0.75,
        confidence=0.8
    )
    
    # Paper trading trade
    await paper_trading_system.execute_trade(
        exchange="binance",
        token="ETHUSDT",
        price=3000.0,
        volume=0.05,
        action="sell",
        prediction=0.65,
        confidence=0.7
    )
    
    # Live trading trade
    await live_trading_system.execute_trade(
        exchange="binance",
        token="ADAUSDT",
        price=0.5,
        volume=1000.0,
        action="buy",
        prediction=0.8,
        confidence=0.9
    )
    
    print("✅ Monitored trades executed!")
    
    # Get integration statistics
    stats = orchestrator.get_comprehensive_stats()
    print(f"📊 Integration stats: {json.dumps(stats['trading_integrator'], indent=2)}")
    
    # Shutdown
    await orchestrator.shutdown()
    print("✅ Trading system integration example completed!")


if __name__ == "__main__":
    print("🚀 Enhanced ML Monitoring System Examples")
    print("=" * 50)
    
    # Run examples
    asyncio.run(await example_enhanced_monitoring())
    asyncio.run(await example_trading_system_integration())
    
    print("\n🎉 All examples completed successfully!")
    print("\n📁 Check the 'example_monitoring_exports' directory for exported CSV files")