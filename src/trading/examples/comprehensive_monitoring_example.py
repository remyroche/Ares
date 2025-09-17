"""
Comprehensive Monitoring Example

Demonstrates the full trading system with comprehensive monitoring,
detailed trade metrics, SHAP/LIME explanations, and real-time reporting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_structured, LogLevel
from ..execution.trading_orchestrator import TradingOrchestrator, create_trading_orchestrator
from ..monitoring.comprehensive_trade_monitor import (
    comprehensive_trade_monitor, initialize_comprehensive_monitoring,
    record_detailed_trade, update_trade_outcome
)
from ..reporting.performance_reporter import generate_trading_report
from ..reporting.dashboard_generator import create_trading_dashboard
from ..reporting.trade_analyzer import analyze_trade_performance

async def run_comprehensive_monitoring_example():
    """
    Run a comprehensive example showing all monitoring and reporting features.
    """
    tprint_info("🚀 Starting Comprehensive Trading Monitoring Example")
    
    try:
        # Initialize comprehensive monitoring
        monitoring_config = {
            'enable_explanations': True,
            'enable_real_time_export': True,
            'export_directory': 'example_trading_reports',
            'max_trades_in_memory': 1000
        }
        
        success = await initialize_comprehensive_monitoring(monitoring_config)
        if not success:
            tprint_warning("⚠️ Failed to initialize comprehensive monitoring")
            return
        
        tprint_success("✅ Comprehensive monitoring initialized")
        
        # Create trading orchestrator configuration
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
                'risk_per_trade': 0.02
            },
            'strategist': {},
            'analyst_signals': {
                'confidence_threshold': 0.6,
                'max_history': 1000
            },
            'tactician_signals': {
                'confidence_threshold': 0.6,
                'max_history': 1000
            },
            'signal_combiner': {
                'analyst_weight': 0.6,
                'tactician_weight': 0.4,
                'combination_method': 'weighted_average'
            }
        }
        
        # Create and initialize trading orchestrator
        tprint_info("🔄 Creating trading orchestrator...")
        orchestrator = create_trading_orchestrator(orchestrator_config)
        
        success = await orchestrator.initialize()
        if not success:
            tprint_warning("⚠️ Failed to initialize trading orchestrator")
            return
        
        tprint_success("✅ Trading orchestrator initialized")
        
        # Simulate trading session with detailed monitoring
        await simulate_trading_with_monitoring(orchestrator)
        
        # Generate comprehensive reports
        await generate_example_reports()
        
        # Show live dashboard
        await show_live_dashboard(orchestrator)
        
        tprint_success("✅ Comprehensive monitoring example completed")
        
    except Exception as e:
        tprint_warning(f"❌ Example failed: {e}")
        raise

async def simulate_trading_with_monitoring(orchestrator: TradingOrchestrator):
    """Simulate trading operations with comprehensive monitoring."""
    try:
        tprint_info("🎯 Simulating trading operations with comprehensive monitoring...")
        
        # Start trading session
        success = await orchestrator.start_trading_session()
        if not success:
            tprint_warning("⚠️ Failed to start trading session")
            return
        
        # Simulate market data
        market_data = create_sample_market_data()
        
        # Simulate multiple trades with different scenarios
        trade_scenarios = [
            {
                'symbol': 'ETHUSDT',
                'action': 'buy',
                'quantity': 0.5,
                'price': 3000.0,
                'confidence': 0.85,
                'regime_type': 'trending_up',
                'expected_outcome': 'profit'
            },
            {
                'symbol': 'ETHUSDT',
                'action': 'sell',
                'quantity': 0.3,
                'price': 3050.0,
                'confidence': 0.72,
                'regime_type': 'high_volatility',
                'expected_outcome': 'profit'
            },
            {
                'symbol': 'ETHUSDT',
                'action': 'buy',
                'quantity': 0.2,
                'price': 2980.0,
                'confidence': 0.55,
                'regime_type': 'sideways',
                'expected_outcome': 'loss'
            },
            {
                'symbol': 'ETHUSDT',
                'action': 'sell',
                'quantity': 0.4,
                'price': 3020.0,
                'confidence': 0.78,
                'regime_type': 'momentum',
                'expected_outcome': 'profit'
            }
        ]
        
        trade_ids = []
        
        for i, scenario in enumerate(trade_scenarios):
            tprint_info(f"🔄 Executing trade scenario {i+1}/{len(trade_scenarios)}")
            
            # Prepare comprehensive trade data
            trade_data = {
                **scenario,
                'trading_mode': 'paper',
                'exchange': 'binance',
                'analyst_signal': {
                    'signal_type': scenario['action'],
                    'confidence_score': scenario['confidence'],
                    'market_health_score': 0.8,
                    'volatility_score': 0.3,
                    'liquidation_risk_score': 0.2
                },
                'tactician_signal': {
                    'timing_signal': f"enter_long" if scenario['action'] == 'buy' else 'enter_short',
                    'confidence_score': scenario['confidence'] * 0.9,
                    'position_sizing': {
                        'recommended_size': scenario['quantity'],
                        'leverage': 1.5,
                        'risk_per_trade': 0.02,
                        'kelly_fraction': 0.15
                    }
                },
                'regime_data': {
                    'primary_regime': scenario['regime_type'],
                    'confidence': 0.8,
                    'regime_probabilities': {
                        scenario['regime_type']: 0.7,
                        'other_regime': 0.3
                    }
                },
                'risk_metrics': {
                    'portfolio_risk': 0.025,
                    'var_95': 0.035,
                    'volatility_estimate': 0.4
                }
            }
            
            # Prepare models used
            models_used = {
                'analyst_ensemble': {
                    'type': 'analyst',
                    'model': create_mock_model('analyst'),
                    'prediction': scenario['confidence'],
                    'confidence': scenario['confidence'],
                    'weight': 0.6,
                    'version': '1.2.0',
                    'features_count': 45
                },
                'tactician_ensemble': {
                    'type': 'tactician',
                    'model': create_mock_model('tactician'),
                    'prediction': scenario['confidence'] * 0.9,
                    'confidence': scenario['confidence'] * 0.9,
                    'weight': 0.4,
                    'version': '1.1.0',
                    'features_count': 32
                },
                'hmm_regime_model': {
                    'type': 'hmm',
                    'model': create_mock_model('hmm'),
                    'prediction': 0.8,
                    'confidence': 0.8,
                    'weight': 1.0,
                    'version': '2.0.0',
                    'features_count': 15
                }
            }
            
            # Record detailed trade
            trade_id = await record_detailed_trade(trade_data, models_used, market_data)
            trade_ids.append(trade_id)
            
            # Simulate trade execution and outcome
            await asyncio.sleep(1)  # Simulate execution time
            
            # Update trade outcome based on scenario
            if scenario['expected_outcome'] == 'profit':
                pnl_absolute = np.random.uniform(50, 200)
                pnl_percentage = pnl_absolute / (scenario['quantity'] * scenario['price'])
                execution_quality = np.random.uniform(0.85, 0.95)
            else:
                pnl_absolute = np.random.uniform(-100, -20)
                pnl_percentage = pnl_absolute / (scenario['quantity'] * scenario['price'])
                execution_quality = np.random.uniform(0.6, 0.8)
            
            outcome_data = {
                'exit_price': scenario['price'] * (1 + pnl_percentage),
                'pnl_absolute': pnl_absolute,
                'pnl_percentage': pnl_percentage,
                'duration_minutes': np.random.uniform(5, 30),
                'execution_quality': execution_quality,
                'slippage': np.random.uniform(0.0005, 0.002),
                'commission': scenario['quantity'] * scenario['price'] * 0.001,
                'timing_quality': scenario['confidence'],
                'max_favorable_excursion': abs(pnl_absolute) * 1.2 if pnl_absolute > 0 else 0.0,
                'max_adverse_excursion': abs(pnl_absolute) * 0.3 if pnl_absolute < 0 else 0.0
            }
            
            await update_trade_outcome(trade_id, outcome_data)
            
            tprint_success(f"✅ Completed trade scenario {i+1}: {scenario['action']} {scenario['symbol']} -> {scenario['expected_outcome']}")
            
            # Brief pause between trades
            await asyncio.sleep(2)
        
        tprint_success(f"✅ Completed {len(trade_scenarios)} simulated trades")
        
        # Stop trading session
        await orchestrator.stop_trading_session()
        
    except Exception as e:
        tprint_warning(f"❌ Trading simulation failed: {e}")
        raise

def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for testing."""
    # Generate 100 candles of sample data
    np.random.seed(42)
    
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=2), periods=100, freq='1T')
    
    # Generate realistic price data
    base_price = 3000.0
    price_changes = np.random.normal(0, 0.001, 100)  # 0.1% volatility
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.0005)))
        low = close * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_mock_model(model_type: str):
    """Create a mock model for testing explanations."""
    class MockModel:
        def __init__(self, model_type: str):
            self.model_type = model_type
            self.feature_names = [
                'close', 'volume', 'sma_20', 'rsi', 'macd',
                'volatility_20', 'returns_5', 'bb_position'
            ]
        
        def predict(self, X):
            # Mock prediction
            return np.random.uniform(0.4, 0.9, len(X) if hasattr(X, '__len__') else 1)
        
        def predict_proba(self, X):
            # Mock probability prediction
            proba = np.random.uniform(0.3, 0.8)
            return [[1-proba, proba]]
    
    return MockModel(model_type)

async def generate_example_reports():
    """Generate example reports showing all monitoring features."""
    try:
        tprint_info("📊 Generating example reports...")
        
        # Generate session performance report
        session_report = await comprehensive_trade_monitor.generate_performance_report("session", "json")
        if session_report:
            tprint_success("✅ Generated session performance report")
            
            # Show key metrics
            if 'executive_summary' in session_report:
                tprint_structured(session_report['executive_summary'], LogLevel.INFO)
        
        # Generate daily report
        daily_report = await comprehensive_trade_monitor.generate_performance_report("daily", "json")
        if daily_report:
            tprint_success("✅ Generated daily performance report")
        
        tprint_success("✅ All example reports generated")
        
    except Exception as e:
        tprint_warning(f"❌ Failed to generate example reports: {e}")

async def show_live_dashboard(orchestrator: TradingOrchestrator):
    """Show live dashboard data."""
    try:
        tprint_info("📊 Generating live dashboard...")
        
        # Generate live dashboard
        dashboard = await orchestrator.generate_live_dashboard()
        
        if dashboard:
            tprint_success("✅ Live dashboard generated")
            
            # Show key dashboard metrics
            if 'live_metrics' in dashboard:
                tprint_info("📈 Live Performance Metrics:")
                tprint_structured(dashboard['live_metrics'], LogLevel.INFO)
            
            # Show model performance
            if 'model_dashboard' in dashboard:
                tprint_info("🤖 Model Performance Dashboard:")
                for model_id, metrics in dashboard['model_dashboard'].items():
                    tprint_info(f"  {model_id}: {metrics['performance_stats']['total_pnl']:.2f} PnL, {metrics['confidence_stats']['avg_confidence']:.1%} confidence")
        
        tprint_success("✅ Live dashboard displayed")
        
    except Exception as e:
        tprint_warning(f"❌ Failed to show live dashboard: {e}")

async def demonstrate_trade_analysis():
    """Demonstrate detailed individual trade analysis."""
    try:
        tprint_info("🔍 Demonstrating detailed trade analysis...")
        
        # Get a completed trade for analysis
        if comprehensive_trade_monitor.completed_trades:
            trade = comprehensive_trade_monitor.completed_trades[0]
            
            # Perform detailed analysis
            analysis = await analyze_trade_performance(trade, include_explanations=True)
            
            if analysis:
                tprint_success("✅ Detailed trade analysis completed")
                
                # Show trade quality score
                if 'trade_quality_score' in analysis:
                    quality = analysis['trade_quality_score']
                    tprint_info(f"📊 Trade Quality: {quality['quality_grade']} (Score: {quality['overall_score']:.3f})")
                
                # Show model contributions
                if 'model_analysis' in analysis:
                    tprint_info("🤖 Model Contributions:")
                    for model_id, contribution in analysis['model_analysis']['individual_models'].items():
                        tprint_info(f"  {model_id}: {contribution['performance_assessment']} (confidence: {contribution['confidence']:.1%})")
                
                # Show feature importance
                if 'explainability_analysis' in analysis and 'feature_consensus' in analysis['explainability_analysis']:
                    top_features = analysis['explainability_analysis']['feature_consensus'].get('top_features', [])
                    if top_features:
                        tprint_info("🎯 Top Important Features:")
                        for feature, importance in top_features[:5]:
                            tprint_info(f"  {feature}: {importance:.4f}")
        
    except Exception as e:
        tprint_warning(f"❌ Trade analysis demonstration failed: {e}")

# Main execution
if __name__ == "__main__":
    async def main():
        """Main example execution."""
        try:
            # Run comprehensive monitoring example
            await run_comprehensive_monitoring_example()
            
            # Demonstrate individual trade analysis
            await demonstrate_trade_analysis()
            
            # Keep running for a bit to show live updates
            tprint_info("⏰ Monitoring live updates for 30 seconds...")
            await asyncio.sleep(30)
            
            tprint_success("🎉 Comprehensive monitoring example completed successfully!")
            
        except KeyboardInterrupt:
            tprint_info("⏹️ Example stopped by user")
        except Exception as e:
            tprint_warning(f"❌ Example failed: {e}")
        finally:
            # Cleanup
            if comprehensive_trade_monitor.is_initialized:
                await comprehensive_trade_monitor.stop()
    
    # Run the example
    asyncio.run(main())