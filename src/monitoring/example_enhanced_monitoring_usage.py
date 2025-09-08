#!/usr/bin/env python3
"""
Example Usage of Enhanced Monitoring System

This example demonstrates how to use the enhanced monitoring system for
comprehensive trade decision tracking and analysis across backtesting,
paper trading, and live trading modes.
"""

import asyncio

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Import the enhanced monitoring components
from .enhanced_monitoring_orchestrator import (
    EnhancedMonitoringOrchestrator, 
    ComprehensiveTradeDecision,
    EnhancedMonitoringConfig
)
from .trade_decision_capture import (
    TradeDecisionContextCapture,
    ComprehensiveTradeContext,
    MarketConditions,
    HMMRegimeContext,
    TradingSignalContext,
    ModelDecisionContext,
    EnsembleDecisionContext
)
from .shap_lime_integration import (
    ExplainabilityIntegrator,
    ModelExplanationRequest
)
import json
import time

from .enhanced_ml_monitoring import (
    TradeContext, TradingIndicator, MLModelDecision, 
    EnsembleDecision, TradingMode, ModelType
)

class MockTradingSystem:
    """Mock trading system for demonstration purposes."""
    
    def __init__(self, system_type: str):
        self.system_type = system_type
        self.trades_executed = 0
        self.current_balance = 10000.0
    
    async def execute_trade(self, symbol: str, action: str, size: float, price: float) -> Dict[str, Any]:
        """Mock trade execution."""
        self.trades_executed += 1
        
        # Simulate trade result
        if action == "buy":
            self.current_balance -= size * price
            pnl = np.random.normal(0.01, 0.05) * size * price  # Random PnL
        elif action == "sell":
            self.current_balance += size * price
            pnl = np.random.normal(0.01, 0.05) * size * price  # Random PnL
        else:
            pnl = 0.0
        
        return {
            'trade_id': f"{self.system_type}_{self.trades_executed}",
            'executed_price': price * (1 + np.random.normal(0, 0.001)),  # Slippage
            'slippage': np.random.normal(0, 0.001),
            'commission': size * price * 0.001,  # 0.1% commission
            'profit_loss': pnl,
            'execution_time_ms': np.random.uniform(10, 100)
        }

class MockMLModel:
    """Mock ML model for demonstration purposes."""
    
    def __init__(self, model_id: str, model_type: str):
        self.model_id = model_id
        self.model_type = model_type
        self.version = "1.0"
    
    def predict(self, features: np.ndarray) -> float:
        """Mock prediction."""
        return np.random.normal(0.5, 0.2)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Mock probability prediction."""
        return np.array([0.3, 0.7])

class MockHMMModel:
    """Mock HMM model for demonstration purposes."""
    
    def __init__(self):
        self.regimes = ['bull', 'bear', 'sideways']
        self.current_regime = 0
    
    def predict(self, features: np.ndarray) -> int:
        """Mock regime prediction."""
        return self.current_regime
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Mock regime probabilities."""
        probs = np.random.dirichlet([1, 1, 1])
        return probs

class MockSignalGenerator:
    """Mock signal generator for demonstration purposes."""
    
    def generate_signals(self, market_data: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Generate mock trading signals."""
        return {
            'signal_strength': np.random.uniform(-1, 1),
            'signal_confidence': np.random.uniform(0.3, 0.9),
            'signal_quality': np.random.uniform(0.4, 0.8),
            'trend_signal': np.random.uniform(-1, 1),
            'momentum_signal': np.random.uniform(-1, 1),
            'mean_reversion_signal': np.random.uniform(-1, 1),
            'volatility_signal': np.random.uniform(-1, 1),
            'volume_signal': np.random.uniform(-1, 1),
            'signal_freshness': np.random.uniform(0.5, 1.0),
            'signal_persistence': np.random.uniform(0.0, 0.8),
            'risk_level': np.random.uniform(0.1, 0.8),
            'drawdown_risk': np.random.uniform(0.0, 0.3),
            'volatility_risk': np.random.uniform(0.0, 0.5),
            'liquidity_risk': np.random.uniform(0.0, 0.2)
        }

def create_mock_market_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Create mock market data for demonstration."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='15T')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    price = 100.0
    prices = [price]
    
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)  # 2% volatility
        price *= (1 + change)
        prices.append(price)
    
    # Generate volume data
    volumes = np.random.lognormal(10, 0.5, len(dates))
    
    # Generate OHLC data
    data = []
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Add technical indicators
    df['rsi_14'] = 50 + np.random.normal(0, 15, len(df))
    df['macd'] = np.random.normal(0, 0.5, len(df))
    df['bb_position'] = np.random.uniform(0, 1, len(df))
    df['atr_14'] = df['close'] * np.random.uniform(0.01, 0.05, len(df))
    df['adx_14'] = np.random.uniform(20, 50, len(df))
    
    return df

async def example_enhanced_monitoring_usage():
    """Example of how to use the enhanced monitoring system."""
    
    print("🚀 Starting Enhanced Monitoring Example")
    
    # 1. Load configuration
    config_path = Path(__file__).parent / "enhanced_monitoring_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ Configuration loaded")
    
    # 2. Initialize enhanced monitoring orchestrator
    orchestrator = EnhancedMonitoringOrchestrator(config)
    context_capture = TradeDecisionContextCapture(config)
    explainability_integrator = ExplainabilityIntegrator(config)
    
    print("✅ Enhanced monitoring components initialized")
    
    # 3. Create mock trading systems
    backtesting_system = MockTradingSystem("backtesting")
    paper_trading_system = MockTradingSystem("paper_trading")
    live_trading_system = MockTradingSystem("live_trading")
    
    # 4. Create mock models and data sources
    hmm_model = MockHMMModel()
    signal_generator = MockSignalGenerator()
    
    models = {
        'hmm_model': {
            'model': hmm_model,
            'type': 'hmm',
            'version': '1.0',
            'confidence': 0.8,
            'uncertainty': 0.2,
            'recent_accuracy': 0.75,
            'recent_sharpe_ratio': 1.2,
            'stability_score': 0.9,
            'health_score': 0.95
        },
        'analyst_model': {
            'model': MockMLModel('analyst_model', 'analyst'),
            'type': 'analyst',
            'version': '1.0',
            'confidence': 0.7,
            'uncertainty': 0.3,
            'recent_accuracy': 0.68,
            'recent_sharpe_ratio': 0.9,
            'stability_score': 0.8,
            'health_score': 0.9
        },
        'tactician_model': {
            'model': MockMLModel('tactician_model', 'tactician'),
            'type': 'tactician',
            'version': '1.0',
            'confidence': 0.75,
            'uncertainty': 0.25,
            'recent_accuracy': 0.72,
            'recent_sharpe_ratio': 1.1,
            'stability_score': 0.85,
            'health_score': 0.92
        }
    }
    
    print("✅ Mock systems and models created")
    
    # 5. Integrate with trading systems
    await orchestrator.integrate_trading_systems(
        backtesting_system=backtesting_system,
        paper_trading_system=paper_trading_system,
        live_trading_system=live_trading_system
    )
    
    print("✅ Trading systems integrated")
    
    # 6. Simulate trading decisions across different modes
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    trading_modes = [TradingMode.BACKTEST, TradingMode.PAPER, TradingMode.LIVE]
    
    for mode in trading_modes:
        print(f"\n📊 Simulating {mode.value} trading decisions...")
        
        for i in range(5):  # Simulate 5 decisions per mode
            symbol = symbols[i % len(symbols)]
            
            # Create mock market data
            market_data = create_mock_market_data(symbol, days=7)
            current_price = market_data['close'].iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            
            # Capture comprehensive trade context
            comprehensive_context = await context_capture.capture_trade_context(
                exchange="binance",
                symbol=symbol,
                trading_mode=mode,
                current_price=current_price,
                current_volume=current_volume,
                price_history=market_data['close'].tail(20).tolist(),
                volume_history=market_data['volume'].tail(20).tolist(),
                market_data=market_data,
                hmm_model=hmm_model,
                signal_generator=signal_generator,
                models=models,
                ensemble=None,  # Would be a real ensemble
                additional_context={
                    'session_id': f"session_{mode.value}_{i}",
                    'strategy_id': f"strategy_{symbol}",
                    'risk_parameters': {
                        'max_position_size': 0.1,
                        'stop_loss': 0.02,
                        'take_profit': 0.04
                    }
                }
            )
            
            if not comprehensive_context:
                print(f"❌ Failed to capture context for {symbol}")
                continue
            
            # Create trading indicators
            trading_indicators = [
                TradingIndicator(
                    name="RSI",
                    value=market_data['rsi_14'].iloc[-1],
                    weight=0.2,
                    confidence=0.8,
                    risk_score=0.3,
                    description="Relative Strength Index"
                ),
                TradingIndicator(
                    name="MACD",
                    value=market_data['macd'].iloc[-1],
                    weight=0.3,
                    confidence=0.7,
                    risk_score=0.4,
                    description="MACD Signal"
                ),
                TradingIndicator(
                    name="Bollinger Position",
                    value=market_data['bb_position'].iloc[-1],
                    weight=0.25,
                    confidence=0.75,
                    risk_score=0.35,
                    description="Position within Bollinger Bands"
                ),
                TradingIndicator(
                    name="Volume",
                    value=current_volume,
                    weight=0.25,
                    confidence=0.6,
                    risk_score=0.2,
                    description="Current volume"
                )
            ]
            
            # Create individual model decisions
            individual_model_decisions = []
            for model_id, model_info in models.items():
                model = model_info['model']
                features = np.array([current_price])
                
                # Get model prediction
                prediction = model.predict(features)
                
                # Create model decision
                model_decision = MLModelDecision(
                    model_id=model_id,
                    model_type=ModelType(model_info['type']),
                    prediction=prediction,
                    confidence=model_info['confidence'],
                    risk_score=1.0 - model_info['confidence'],
                    feature_importance={
                        'price': 0.6,
                        'volume': 0.2,
                        'rsi': 0.1,
                        'macd': 0.1
                    },
                    processing_time_ms=np.random.uniform(5, 50),
                    model_version=model_info['version']
                )
                
                individual_model_decisions.append(model_decision)
            
            # Create ensemble decision
            ensemble_decision = EnsembleDecision(
                ensemble_id="main_ensemble",
                final_prediction=np.mean([md.prediction for md in individual_model_decisions]),
                final_confidence=np.mean([md.confidence for md in individual_model_decisions]),
                final_risk_score=np.mean([md.risk_score for md in individual_model_decisions]),
                model_weights={md.model_id: 1.0/len(individual_model_decisions) for md in individual_model_decisions},
                model_decisions=individual_model_decisions,
                voting_mechanism="weighted_average",
                consensus_score=0.8,
                disagreement_level=0.2
            )
            
            # Model indicator weights (how each model weights different indicators)
            model_indicator_weights = {}
            for model_decision in individual_model_decisions:
                model_indicator_weights[model_decision.model_id] = {
                    'RSI': np.random.uniform(0.1, 0.4),
                    'MACD': np.random.uniform(0.2, 0.5),
                    'Bollinger Position': np.random.uniform(0.1, 0.3),
                    'Volume': np.random.uniform(0.1, 0.3)
                }
            
            # Determine action based on ensemble prediction
            if ensemble_decision.final_prediction > 0.6:
                action = "buy"
                position_size = 0.1
            elif ensemble_decision.final_prediction < 0.4:
                action = "sell"
                position_size = 0.1
            else:
                action = "hold"
                position_size = 0.0
            
            # Record comprehensive decision
            comprehensive_decision = await orchestrator.record_comprehensive_decision(
                context=TradeContext(
                    exchange="binance",
                    token=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    volume=current_volume,
                    timeframe="15m",
                    regime=comprehensive_context.hmm_regime_context.regime_id if comprehensive_context.hmm_regime_context else None,
                    hmm_regime_info=None,  # Would be populated from HMM context
                    market_conditions=comprehensive_context.market_conditions.__dict__ if comprehensive_context.market_conditions else None
                ),
                trading_mode=mode,
                trading_indicators=trading_indicators,
                ensemble_decision=ensemble_decision,
                individual_model_decisions=individual_model_decisions,
                model_indicator_weights=model_indicator_weights,
                action=action,
                position_size=position_size,
                stop_loss=current_price * 0.98 if action == "buy" else current_price * 1.02,
                take_profit=current_price * 1.04 if action == "buy" else current_price * 0.96,
                market_conditions=comprehensive_context.market_conditions.__dict__ if comprehensive_context.market_conditions else None,
                regime_analysis=comprehensive_context.hmm_regime_context.__dict__ if comprehensive_context.hmm_regime_context else None,
                execution_time_ms=np.random.uniform(10, 100)
            )
            
            if comprehensive_decision:
                print(f"✅ Recorded {mode.value} decision for {symbol}: {action} at {current_price:.2f}")
                
                # Simulate trade execution
                if action in ["buy", "sell"]:
                    trade_result = await (backtesting_system if mode == TradingMode.BACKTEST 
                                       else paper_trading_system if mode == TradingMode.PAPER 
                                       else live_trading_system).execute_trade(
                        symbol=symbol,
                        action=action,
                        size=position_size,
                        price=current_price
                    )
                    
                    # Update decision with trade results
                    comprehensive_decision.success_metrics = {
                        'profit_loss': trade_result['profit_loss'],
                        'execution_price': trade_result['executed_price'],
                        'slippage': trade_result['slippage'],
                        'commission': trade_result['commission']
                    }
            else:
                print(f"❌ Failed to record {mode.value} decision for {symbol}")
    
    print("\n📈 Trading simulation completed")
    
    # 7. Generate SHAP/LIME explanations for some decisions
    print("\n🔍 Generating model explanations...")
    
    for i, decision in enumerate(orchestrator.comprehensive_decisions[-3:]):  # Last 3 decisions
        for model_decision in decision.individual_model_decisions:
            # Create explanation request
            explanation_request = ModelExplanationRequest(
                model_id=model_decision.model_id,
                model_type=model_decision.model_type.value,
                features=np.array([decision.context.price]),
                feature_names=['price'],
                prediction=model_decision.prediction,
                model=models[model_decision.model_id]['model'],
                training_data=create_mock_market_data(decision.context.token, days=30)
            )
            
            # Generate explanations
            explanations = await explainability_integrator.explain_model_prediction(explanation_request)
            
            if explanations:
                print(f"✅ Generated explanations for {model_decision.model_id}")
            else:
                print(f"⚠️ No explanations generated for {model_decision.model_id}")
    
    # 8. Export monitoring data
    print("\n📊 Exporting monitoring data...")
    
    # Export monthly report
    monthly_success = await orchestrator.export_monthly_report()
    if monthly_success:
        print("✅ Monthly report exported")
    else:
        print("❌ Failed to export monthly report")
    
    # Export daily ongoing CSV
    daily_success = await orchestrator.export_daily_ongoing_csv()
    if daily_success:
        print("✅ Daily ongoing CSV exported")
    else:
        print("❌ Failed to export daily ongoing CSV")
    
    # Force export all data
    export_success = await orchestrator.force_export_all()
    if export_success:
        print("✅ All monitoring data exported")
    else:
        print("❌ Failed to export all monitoring data")
    
    # 9. Display monitoring statistics
    print("\n📊 Monitoring Statistics:")
    stats = orchestrator.get_monitoring_stats()
    
    print(f"Total comprehensive decisions: {stats['orchestrator_stats']['total_comprehensive_decisions']}")
    print(f"Decision count: {stats['orchestrator_stats']['decision_count']}")
    print(f"Monitoring duration: {stats['orchestrator_stats']['monitoring_duration_hours']:.2f} hours")
    print(f"Monthly reports generated: {stats['orchestrator_stats']['monthly_reports_generated']}")
    
    # Enhanced ML Monitor stats
    ml_stats = stats['enhanced_ml_monitor_stats']
    print(f"ML Monitor decisions: {ml_stats['total_decisions']}")
    print(f"ML Monitor model performances: {ml_stats['total_model_performances']}")
    print(f"ML Monitor ensemble performances: {ml_stats['total_ensemble_performances']}")
    
    # Daily Summary Tracker stats
    daily_stats = stats['daily_summary_tracker_stats']
    print(f"Daily summaries tracked: {daily_stats['total_days_tracked']}")
    print(f"Regimes tracked: {daily_stats['regimes_tracked']}")
    
    # Explainability stats
    explainability_stats = explainability_integrator.get_explanation_stats()
    print(f"SHAP available: {explainability_stats['shap_available']}")
    print(f"LIME available: {explainability_stats['lime_available']}")
    print(f"SHAP explanations generated: {explainability_stats['shap_explanations_generated']}")
    print(f"LIME explanations generated: {explainability_stats['lime_explanations_generated']}")
    print(f"Combined explanations generated: {explainability_stats['combined_explanations_generated']}")
    
    # Context capture stats
    context_stats = context_capture.get_capture_stats()
    print(f"Contexts captured: {context_stats['total_contexts_captured']}")
    print(f"Market conditions enabled: {context_stats['enable_market_conditions']}")
    print(f"HMM context enabled: {context_stats['enable_hmm_context']}")
    print(f"Signal context enabled: {context_stats['enable_signal_context']}")
    print(f"Model context enabled: {context_stats['enable_model_context']}")
    print(f"Ensemble context enabled: {context_stats['enable_ensemble_context']}")
    
    print("\n🎉 Enhanced monitoring example completed successfully!")
    
    # 10. Display export directory contents
    export_dir = Path(config['enhanced_monitoring']['export_directory'])
    if export_dir.exists():
        print(f"\n📁 Export directory contents ({export_dir}):")
        for file_path in export_dir.rglob("*"):
            if file_path.is_file():
                print(f"  📄 {file_path.relative_to(export_dir)}")
    
    return orchestrator, context_capture, explainability_integrator

async def example_backtesting_integration():
    """Example of integrating enhanced monitoring with backtesting."""
    
    print("\n🔄 Backtesting Integration Example")
    
    # Load configuration
    config_path = Path(__file__).parent / "enhanced_monitoring_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize monitoring
    orchestrator = EnhancedMonitoringOrchestrator(config)
    
    # Create mock backtesting system
    backtesting_system = MockTradingSystem("backtesting")
    
    # Integrate with backtesting
    await orchestrator.integrate_trading_systems(backtesting_system=backtesting_system)
    
    # Simulate backtesting run
    symbols = ['BTCUSDT', 'ETHUSDT']
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    current_date = start_date
    while current_date <= end_date:
        for symbol in symbols:
            # Simulate daily trading decision
            market_data = create_mock_market_data(symbol, days=1)
            current_price = market_data['close'].iloc[-1]
            
            # Create and record decision
            # ... (similar to main example but focused on backtesting)
            
            pass
        
        current_date += timedelta(days=1)
    
    print("✅ Backtesting integration example completed")

async def example_paper_trading_integration():
    """Example of integrating enhanced monitoring with paper trading."""
    
    print("\n📝 Paper Trading Integration Example")
    
    # Similar to backtesting but with real-time monitoring
    # ... (implementation would be similar to backtesting example)
    
    print("✅ Paper trading integration example completed")

async def example_live_trading_integration():
    """Example of integrating enhanced monitoring with live trading."""
    
    print("\n⚡ Live Trading Integration Example")
    
    # Similar to paper trading but with real market data and risk management
    # ... (implementation would be similar to other examples)
    
    print("✅ Live trading integration example completed")

if __name__ == "__main__":
    import yaml
    
    # Run the main example
    asyncio.run(example_enhanced_monitoring_usage())
    
    # Run integration examples
    asyncio.run(example_backtesting_integration())
    asyncio.run(example_paper_trading_integration())
    asyncio.run(example_live_trading_integration())