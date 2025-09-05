#!/usr/bin/env python3
"""
Enhanced Example Usage with HMM Regime and Daily Summary Features

This example demonstrates the enhanced monitoring system with HMM regime
information, daily summaries, and GUI integration.
"""

import asyncio
from datetime import datetime, date, timedelta

from .monitoring_orchestrator import (
    MonitoringOrchestrator, create_monitoring_orchestrator,
    TradeContext, TradingIndicator, MLModelDecision, EnsembleDecision,
    TradeDecision, TradingMode, ModelType, ModelPerformanceMetrics,
    EnsemblePerformanceMetrics, HMMRegimeInfo, DailyTradeSummary
)


async def example_enhanced_monitoring_with_hmm():
    """Example of using the enhanced monitoring system with HMM regime information."""
    
    # Load configuration
    config = {
        "enhanced_monitoring": {
            "enable_monitoring": True,
            "enable_explanations": True,
            "enable_ensemble_monitoring": True,
            "enable_csv_export": True,
            "export_interval_days": 1,  # Export daily for demo
            "max_memory_decisions": 1000,
            "export_directory": "enhanced_monitoring_exports"
        },
        "daily_summary_tracker": {
            "enable_real_time_updates": True,
            "summary_retention_days": 30,
            "export_directory": "daily_summaries"
        },
        "enhanced_ml_monitoring": {
            "enable_shap": True,
            "enable_lime": True,
            "csv_export_interval_days": 1,
            "max_decisions_in_memory": 1000,
            "export_directory": "enhanced_monitoring_exports"
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
            "export_directory": "enhanced_monitoring_exports",
            "include_raw_data": True,
            "include_summary_stats": True,
            "decimal_precision": 4
        }
    }
    
    # Create and initialize monitoring orchestrator
    print("🚀 Initializing Enhanced ML Monitoring System with HMM Regime Support...")
    orchestrator = await create_monitoring_orchestrator(config)
    
    if not orchestrator:
        print("❌ Failed to initialize monitoring orchestrator")
        return
    
    print("✅ Enhanced ML Monitoring System initialized successfully!")
    
    # Example 1: Record trade decisions with HMM regime information
    print("\n📊 Recording trade decisions with HMM regime information...")
    
    # Simulate multiple days of trading
    base_date = datetime.now() - timedelta(days=5)
    
    for day_offset in range(5):
        current_date = base_date + timedelta(days=day_offset)
        
        # Simulate 3-8 trades per day
        num_trades = np.random.randint(3, 9)
        
        for trade_num in range(num_trades):
            # Create HMM regime information
            regime_id = f"regime_{np.random.choice(['bull', 'bear', 'sideways'])}"
            regime_name = regime_id.replace('_', ' ').title()
            
            hmm_regime_info = HMMRegimeInfo(
                regime_id=regime_id,
                regime_name=regime_name,
                regime_probability=np.random.uniform(0.6, 0.95),
                regime_transition_probability=np.random.uniform(0.1, 0.3),
                regime_duration=np.random.randint(5, 50),
                regime_stability_score=np.random.uniform(0.7, 0.95),
                next_regime_probabilities={
                    "regime_bull": np.random.uniform(0.2, 0.4),
                    "regime_bear": np.random.uniform(0.2, 0.4),
                    "regime_sideways": np.random.uniform(0.2, 0.4)
                }
            )
            
            # Create trade context with HMM regime
            context = TradeContext(
                exchange="binance",
                token=np.random.choice(["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]),
                timestamp=current_date + timedelta(hours=trade_num * 2),
                price=np.random.uniform(20000, 50000) if "BTC" in context.token else np.random.uniform(1000, 4000),
                volume=np.random.uniform(0.01, 0.5),
                timeframe="1h",
                regime=regime_id,
                hmm_regime_info=hmm_regime_info,
                market_conditions={
                    "volatility": np.random.uniform(0.1, 0.3),
                    "trend": np.random.choice(["upward", "downward", "sideways"]),
                    "volume_profile": np.random.choice(["high", "medium", "low"])
                }
            )
            
            # Create trading indicators
            trading_indicators = [
                TradingIndicator(
                    name="RSI",
                    value=np.random.uniform(20, 80),
                    weight=0.3,
                    confidence=np.random.uniform(0.6, 0.9),
                    risk_score=np.random.uniform(0.1, 0.4),
                    description="Relative Strength Index"
                ),
                TradingIndicator(
                    name="MACD",
                    value=np.random.uniform(-100, 200),
                    weight=0.4,
                    confidence=np.random.uniform(0.7, 0.95),
                    risk_score=np.random.uniform(0.1, 0.3),
                    description="MACD Signal"
                ),
                TradingIndicator(
                    name="Bollinger Bands",
                    value=np.random.uniform(0.1, 0.9),
                    weight=0.3,
                    confidence=np.random.uniform(0.6, 0.8),
                    risk_score=np.random.uniform(0.2, 0.5),
                    description="Bollinger Band Position"
                )
            ]
            
            # Create individual model decisions
            model_decisions = [
                MLModelDecision(
                    model_id="hmm_model_1",
                    model_type=ModelType.HMM,
                    prediction=np.random.uniform(0.4, 0.9),
                    confidence=np.random.uniform(0.7, 0.95),
                    risk_score=np.random.uniform(0.1, 0.3),
                    feature_importance={
                        "price_momentum": np.random.uniform(0.3, 0.5),
                        "volume_trend": np.random.uniform(0.2, 0.4),
                        "volatility": np.random.uniform(0.1, 0.3),
                        "regime_stability": np.random.uniform(0.1, 0.2)
                    },
                    processing_time_ms=np.random.uniform(10, 30),
                    model_version="v2.1"
                ),
                MLModelDecision(
                    model_id="analyst_model_1",
                    model_type=ModelType.ANALYST,
                    prediction=np.random.uniform(0.3, 0.8),
                    confidence=np.random.uniform(0.6, 0.9),
                    risk_score=np.random.uniform(0.2, 0.4),
                    feature_importance={
                        "technical_indicators": np.random.uniform(0.4, 0.6),
                        "market_sentiment": np.random.uniform(0.2, 0.4),
                        "fundamental_analysis": np.random.uniform(0.1, 0.3)
                    },
                    processing_time_ms=np.random.uniform(15, 40),
                    model_version="v1.5"
                )
            ]
            
            # Create ensemble decision
            ensemble_decision = EnsembleDecision(
                ensemble_id="main_ensemble",
                final_prediction=np.random.uniform(0.4, 0.8),
                final_confidence=np.random.uniform(0.7, 0.9),
                final_risk_score=np.random.uniform(0.15, 0.35),
                model_weights={
                    "hmm_model_1": 0.6,
                    "analyst_model_1": 0.4
                },
                model_decisions=model_decisions,
                voting_mechanism="weighted_average",
                consensus_score=np.random.uniform(0.7, 0.9),
                disagreement_level=np.random.uniform(0.1, 0.3)
            )
            
            # Create complete trade decision
            trade_decision = TradeDecision(
                decision_id=f"trade_{day_offset}_{trade_num:03d}",
                context=context,
                trading_mode=TradingMode.PAPER,
                timestamp=context.timestamp,
                trading_indicators=trading_indicators,
                overall_confidence=np.random.uniform(0.7, 0.9),
                overall_risk_score=np.random.uniform(0.15, 0.35),
                ensemble_decision=ensemble_decision,
                action=np.random.choice(["buy", "sell", "hold"]),
                position_size=np.random.uniform(0.01, 0.2),
                stop_loss=context.price * np.random.uniform(0.95, 0.98) if np.random.choice([True, False]) else None,
                take_profit=context.price * np.random.uniform(1.02, 1.05) if np.random.choice([True, False]) else None,
                execution_time_ms=np.random.uniform(20, 60),
                success_metrics={
                    "profit_loss": np.random.uniform(-100, 200),
                    "execution_price": context.price * np.random.uniform(0.999, 1.001),
                    "slippage": np.random.uniform(0.0001, 0.001),
                    "commission": np.random.uniform(0.1, 0.5)
                }
            )
            
            # Record the trade decision
            await orchestrator.record_trade_decision(trade_decision)
        
        print(f"✅ Recorded {num_trades} trades for {current_date.date()}")
    
    # Example 2: Get daily summary for a specific date
    print("\n📈 Getting daily summary...")
    
    summary_date = date.today() - timedelta(days=1)
    daily_summary = await orchestrator.daily_summary_tracker.get_daily_summary(summary_date)
    
    if daily_summary:
        print(f"Daily Summary for {summary_date}:")
        print(f"  Total Trades: {daily_summary.total_trades}")
        print(f"  Long Trades: {daily_summary.long_trades}")
        print(f"  Short Trades: {daily_summary.short_trades}")
        print(f"  Dominant Regime: {daily_summary.dominant_regime}")
        print(f"  Total PnL: {daily_summary.total_pnl:.2f}")
        print(f"  Win Rate: {daily_summary.win_rate:.3f}")
        print(f"  Profit Factor: {daily_summary.profit_factor:.2f}")
        print(f"  Regime Distribution: {daily_summary.regime_distribution}")
    
    # Example 3: Get regime performance analysis
    print("\n🔍 Getting regime performance analysis...")
    
    regime_analysis = await orchestrator.get_ensemble_analysis("main_ensemble")
    if regime_analysis and 'current_state' in regime_analysis:
        print("Ensemble Analysis:")
        print(f"  Total Models: {regime_analysis['current_state']['total_models']}")
        print(f"  Active Models: {regime_analysis['current_state']['active_models']}")
        print(f"  Weight Stability: {regime_analysis['current_state']['weight_stability']:.3f}")
        print(f"  Performance Trend: {regime_analysis['current_state']['performance_trend']}")
    
    # Example 4: Export all monitoring data
    print("\n📤 Exporting comprehensive monitoring data...")
    
    export_success = await orchestrator.export_monitoring_data()
    if export_success:
        print("✅ Monitoring data exported successfully!")
        print("📁 Check the 'enhanced_monitoring_exports' directory for CSV files")
        print("📁 Check the 'daily_summaries' directory for daily summary files")
    else:
        print("❌ Failed to export monitoring data")
    
    # Example 5: Get comprehensive statistics
    print("\n📊 Getting comprehensive monitoring statistics...")
    
    stats = orchestrator.get_comprehensive_stats()
    print("Monitoring Statistics:")
    print(f"  Total Decisions Processed: {stats['orchestrator']['total_decisions_processed']}")
    print(f"  Total Exports Performed: {stats['orchestrator']['total_exports_performed']}")
    print(f"  Uptime Hours: {stats['orchestrator']['uptime_hours']:.2f}")
    
    if 'daily_summary_tracker' in stats:
        print(f"  Days Tracked: {stats['daily_summary_tracker']['total_days_tracked']}")
        print(f"  Regimes Tracked: {stats['daily_summary_tracker']['regimes_tracked']}")
    
    # Example 6: Export daily summaries CSV
    print("\n📊 Exporting daily summaries CSV...")
    
    if orchestrator.daily_summary_tracker:
        csv_success = await orchestrator.daily_summary_tracker.export_summary_csv()
        if csv_success:
            print("✅ Daily summaries CSV exported successfully!")
        else:
            print("❌ Failed to export daily summaries CSV")
    
    # Shutdown
    print("\n🛑 Shutting down monitoring system...")
    await orchestrator.shutdown()
    
    print("✅ Enhanced ML Monitoring example completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("  ✅ HMM regime information in trade decisions")
    print("  ✅ Daily summary tracking with regime analysis")
    print("  ✅ Comprehensive CSV exports with regime data")
    print("  ✅ Real-time monitoring and statistics")
    print("  ✅ Ensemble performance tracking")


async def example_gui_launch():
    """Example of launching the GUI dashboard."""
    print("\n🖥️ Launching Enhanced Monitoring Dashboard GUI...")
    
    try:
        from .monitoring.gui import launch_dashboard
        
        # Launch the GUI (this will block until GUI is closed)
        exit_code = launch_dashboard()
        
        if exit_code == 0:
            print("✅ GUI dashboard closed successfully")
        else:
            print(f"❌ GUI dashboard exited with code {exit_code}")
            
    except ImportError as e:
        print(f"❌ Could not import GUI components: {e}")
        print("Make sure matplotlib, seaborn, and tkinter are installed")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")


if __name__ == "__main__":
    print("🚀 Enhanced ML Monitoring System Examples")
    print("=" * 60)
    
    # Run enhanced monitoring example
    asyncio.run(await example_enhanced_monitoring_with_hmm())
    
    # Ask user if they want to launch GUI
    try:
        response = input("\n🖥️ Would you like to launch the GUI dashboard? (y/n): ")
        if response.lower() in ['y', 'yes']:
            asyncio.run(await example_gui_launch())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    print("\n🎉 All examples completed!")
    print("\n📁 Generated Files:")
    print("  📊 enhanced_monitoring_exports/ - Trade decisions with HMM regime data")
    print("  📈 daily_summaries/ - Daily summary CSV files")
    print("  📋 monitoring_exports/ - Comprehensive monitoring reports")