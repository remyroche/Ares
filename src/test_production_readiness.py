"""
Comprehensive test suite for production readiness of all implemented components.

This module tests all the concrete implementations to ensure they are production-ready
and fully functional for deployment.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all implementations
from interfaces.concrete_implementations import (
    InMemoryStateManager, FileBasedStateManager, EventBus, PerformanceReporter,
    ExchangeClient, Analyst, Strategist, Tactician, Supervisor, ModelManager
)

from protocols.concrete_protocols import (
    BinanceTradingDataProvider, MLTradingPredictor, AdvancedRiskManager
)

from utils.ml_common.optimization.concrete_optimization_classes import (
    TradingMultiFidelityObjective, TradingAdvancedMetrics, TradingEvolutionaryAlgorithm,
    TradingFeatureEngineer, TradingEvaluationMetrics
)

from training.steps.market_analysis.components.clustering_algorithms import (
    GaussianMixtureClustering, KMeansClustering, AgglomerativeClusteringAlgorithm,
    AdaptiveClusteringAlgorithm, ClusteringAlgorithmFactory
)

from training.steps.market_analysis.components.clustering_config import ClusteringConfig
from training.steps.market_analysis.components.memory_manager import MemoryManager


class ProductionReadinessTester:
    """Comprehensive tester for production readiness."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_results = {}
        self.logger.info("✅ ProductionReadinessTester initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all production readiness tests."""
        self.logger.info("🚀 Starting comprehensive production readiness tests")
        
        test_suites = [
            ("State Management", self.test_state_management),
            ("Event System", self.test_event_system),
            ("Performance Reporting", self.test_performance_reporting),
            ("Exchange Integration", self.test_exchange_integration),
            ("Trading Components", self.test_trading_components),
            ("ML Predictors", self.test_ml_predictors),
            ("Risk Management", self.test_risk_management),
            ("Clustering Algorithms", self.test_clustering_algorithms),
            ("Optimization Systems", self.test_optimization_systems),
            ("End-to-End Integration", self.test_end_to_end_integration)
        ]
        
        for suite_name, test_func in test_suites:
            try:
                self.logger.info(f"🧪 Running {suite_name} tests...")
                start_time = time.time()
                
                result = await test_func()
                execution_time = time.time() - start_time
                
                self.test_results[suite_name] = {
                    'status': 'PASSED' if result['success'] else 'FAILED',
                    'execution_time': execution_time,
                    'details': result
                }
                
                status_emoji = "✅" if result['success'] else "❌"
                self.logger.info(f"{status_emoji} {suite_name}: {result['message']} ({execution_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"❌ {suite_name} tests failed with exception: {e}")
                self.test_results[suite_name] = {
                    'status': 'FAILED',
                    'execution_time': 0.0,
                    'details': {'success': False, 'message': str(e), 'error': str(e)}
                }
        
        # Generate summary
        total_tests = len(test_suites)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        total_time = sum(result['execution_time'] for result in self.test_results.values())
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'total_execution_time': total_time,
            'production_ready': passed_tests == total_tests,
            'test_results': self.test_results
        }
        
        self.logger.info(f"🏁 Test Summary: {passed_tests}/{total_tests} passed ({summary['success_rate']:.1%})")
        self.logger.info(f"⏱️ Total execution time: {total_time:.2f}s")
        self.logger.info(f"🚀 Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
        
        return summary
    
    async def test_state_management(self) -> Dict[str, Any]:
        """Test state management implementations."""
        try:
            # Test InMemoryStateManager
            state_manager = InMemoryStateManager()
            
            # Basic operations
            state_manager.set_state('test_key', 'test_value')
            assert state_manager.get_state('test_key') == 'test_value'
            
            # Default value
            default_value = state_manager.get_state_if_not_exists('new_key', 'default')
            assert default_value == 'default'
            assert state_manager.get_state('new_key') == 'default'
            
            # Test FileBasedStateManager
            file_state_manager = FileBasedStateManager('test_state.json')
            file_state_manager.set_state('file_key', 'file_value')
            assert file_state_manager.get_state('file_key') == 'file_value'
            
            return {'success': True, 'message': 'State management tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'State management tests failed: {e}'}
    
    async def test_event_system(self) -> Dict[str, Any]:
        """Test event bus system."""
        try:
            event_bus = EventBus()
            
            # Test subscription and publishing
            received_events = []
            
            def test_callback(data):
                received_events.append(data)
            
            # Subscribe to event
            event_bus.subscribe('test_event', test_callback)
            assert event_bus.get_subscriber_count('test_event') == 1
            
            # Publish event
            await event_bus.publish('test_event', {'test': 'data'})
            await asyncio.sleep(0.1)  # Allow async processing
            
            assert len(received_events) == 1
            assert received_events[0]['test'] == 'data'
            
            # Test unsubscribe
            event_bus.unsubscribe('test_event', test_callback)
            assert event_bus.get_subscriber_count('test_event') == 0
            
            return {'success': True, 'message': 'Event system tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Event system tests failed: {e}'}
    
    async def test_performance_reporting(self) -> Dict[str, Any]:
        """Test performance reporting system."""
        try:
            state_manager = InMemoryStateManager()
            performance_reporter = PerformanceReporter(state_manager)
            
            # Test trade logging
            trade_data = {
                'symbol': 'BTCUSDT',
                'action': 'BUY',
                'quantity': 0.1,
                'price': 50000,
                'pnl': 100.0
            }
            
            await performance_reporter.log_trade(trade_data)
            
            # Test performance summary
            summary = await performance_reporter.get_performance_summary()
            assert summary['total_trades'] == 1
            assert summary['total_pnl'] == 100.0
            
            # Test report generation
            report = await performance_reporter.generate_report()
            assert 'TRADING PERFORMANCE REPORT' in report
            assert 'BTCUSDT' in report
            
            return {'success': True, 'message': 'Performance reporting tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Performance reporting tests failed: {e}'}
    
    async def test_exchange_integration(self) -> Dict[str, Any]:
        """Test exchange client integration."""
        try:
            exchange_client = ExchangeClient('binance')
            
            # Test connection
            connected = await exchange_client.connect()
            assert connected
            assert exchange_client.is_connected()
            
            # Test account info
            account_info = await exchange_client.get_account_info()
            assert 'balances' in account_info
            assert 'can_trade' in account_info
            
            # Test klines
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            klines = await exchange_client.get_klines('BTCUSDT', '1m', 60)
            assert len(klines) == 60
            assert all(hasattr(kline, 'symbol') for kline in klines)
            
            # Test order creation
            order_result = await exchange_client.create_order('BTCUSDT', 'BUY', 0.001, 50000)
            assert 'order_id' in order_result
            assert order_result['symbol'] == 'BTCUSDT'
            
            return {'success': True, 'message': 'Exchange integration tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Exchange integration tests failed: {e}'}
    
    async def test_trading_components(self) -> Dict[str, Any]:
        """Test trading system components."""
        try:
            state_manager = InMemoryStateManager()
            event_bus = EventBus()
            exchange_client = ExchangeClient()
            
            # Test Analyst
            analyst = Analyst(state_manager, event_bus)
            await analyst.start()
            
            # Create mock market data
            market_data = type('MarketData', (), {
                'symbol': 'BTCUSDT',
                'timestamp': datetime.now(),
                'open': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'close': 50500.0,
                'volume': 1000.0,
                'interval': '1m'
            })()
            
            analysis_result = await analyst.analyze_market_data(market_data)
            assert hasattr(analysis_result, 'symbol')
            assert hasattr(analysis_result, 'confidence')
            assert hasattr(analysis_result, 'signal')
            
            await analyst.stop()
            
            # Test Strategist
            strategist = Strategist(state_manager, event_bus)
            await strategist.start()
            
            strategy_result = await strategist.formulate_strategy(analysis_result)
            assert hasattr(strategy_result, 'position_bias')
            assert hasattr(strategy_result, 'leverage_cap')
            
            await strategist.stop()
            
            # Test Tactician
            tactician = Tactician(state_manager, event_bus, exchange_client)
            await tactician.start()
            
            trade_decision = await tactician.execute_trade_decision(strategy_result, analysis_result)
            # Trade decision might be None if conditions not met, which is valid
            
            await tactician.stop()
            
            return {'success': True, 'message': 'Trading components tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Trading components tests failed: {e}'}
    
    async def test_ml_predictors(self) -> Dict[str, Any]:
        """Test ML predictor implementations."""
        try:
            # Test ML Trading Predictor
            predictor = MLTradingPredictor()
            
            # Create mock input data
            features = np.random.randn(100, 10)
            model_input = type('ModelInput', (), {
                'features': features,
                'symbol': 'BTCUSDT',
                'timestamp': datetime.now(),
                'market_data': {}
            })()
            
            # Test market direction prediction
            prediction_result = await predictor.predict_market_direction(model_input)
            assert hasattr(prediction_result, 'prediction')
            assert hasattr(prediction_result, 'confidence')
            
            # Test regime classification
            regime_result = await predictor.classify_regime(model_input)
            assert hasattr(regime_result, 'regime')
            assert hasattr(regime_result, 'confidence')
            
            # Test signal generation
            signals = await predictor.generate_signals(model_input)
            assert isinstance(signals, list)
            
            # Test model status
            assert predictor.is_model_ready() in [True, False]
            
            return {'success': True, 'message': 'ML predictors tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'ML predictors tests failed: {e}'}
    
    async def test_risk_management(self) -> Dict[str, Any]:
        """Test risk management system."""
        try:
            risk_manager = AdvancedRiskManager()
            
            # Test trade validation
            trade_decision = type('TradeDecision', (), {
                'symbol': 'BTCUSDT',
                'action': 'BUY',
                'quantity': 0.1,
                'price': 50000.0,
                'leverage': 2.0,
                'stop_loss': 49000.0,
                'take_profit': 52000.0,
                'confidence': 0.8,
                'risk_score': 0.3,
                'timestamp': datetime.now()
            })()
            
            is_valid = await risk_manager.validate_trade(trade_decision)
            assert isinstance(is_valid, bool)
            
            # Test position size calculation
            account_info = {
                'balances': [{'asset': 'USDT', 'free': '10000.0'}]
            }
            risk_params = type('RiskParameters', (), {
                'max_position_size': 0.1,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_drawdown': 0.15,
                'risk_score': 0.3
            })()
            
            position_size = await risk_manager.calculate_position_size('BTCUSDT', account_info, risk_params)
            assert position_size >= 0
            
            # Test portfolio risk assessment
            positions = []
            portfolio_risk = await risk_manager.assess_portfolio_risk(positions)
            assert 'total_risk' in portfolio_risk
            
            return {'success': True, 'message': 'Risk management tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Risk management tests failed: {e}'}
    
    async def test_clustering_algorithms(self) -> Dict[str, Any]:
        """Test clustering algorithms."""
        try:
            # Create test data
            np.random.seed(42)
            test_data = np.random.randn(100, 5)
            
            # Test configuration
            config = ClusteringConfig(n_regimes=3, use_standardized_features=True)
            memory_manager = MemoryManager()
            
            # Test Gaussian Mixture
            gmm = GaussianMixtureClustering(config, memory_manager)
            gmm_result = gmm.fit_predict(test_data)
            assert gmm_result.n_clusters > 0
            assert len(gmm_result.labels) == len(test_data)
            
            # Test K-Means
            kmeans = KMeansClustering(config, memory_manager)
            kmeans_result = kmeans.fit_predict(test_data)
            assert kmeans_result.n_clusters > 0
            assert len(kmeans_result.labels) == len(test_data)
            
            # Test Agglomerative
            agg = AgglomerativeClusteringAlgorithm(config, memory_manager)
            agg_result = agg.fit_predict(test_data)
            assert agg_result.n_clusters > 0
            assert len(agg_result.labels) == len(test_data)
            
            # Test Adaptive
            adaptive = AdaptiveClusteringAlgorithm(config, memory_manager)
            adaptive_result = adaptive.fit_predict(test_data)
            assert adaptive_result.n_clusters > 0
            assert len(adaptive_result.labels) == len(test_data)
            
            # Test Factory
            factory_algorithm = ClusteringAlgorithmFactory.create_algorithm('gaussian_mixture', config, memory_manager)
            assert isinstance(factory_algorithm, GaussianMixtureClustering)
            
            return {'success': True, 'message': 'Clustering algorithms tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Clustering algorithms tests failed: {e}'}
    
    async def test_optimization_systems(self) -> Dict[str, Any]:
        """Test optimization systems."""
        try:
            # Test Multi-Fidelity Objective
            config = type('MultiFidelityConfig', (), {
                'min_resource': 1,
                'max_resource': 10,
                'resource_scaling_factor': 1.0,
                'early_stopping_threshold': 0.01,
                'min_improvement_threshold': 0.001
            })()
            
            mf_objective = TradingMultiFidelityObjective(config)
            test_params = {'learning_rate': 0.01, 'batch_size': 32}
            performance = mf_objective.evaluate(test_params, 5)
            assert isinstance(performance, float)
            
            # Test Advanced Metrics
            advanced_metrics = TradingAdvancedMetrics()
            predictions = np.random.randn(100)
            targets = np.random.randn(100)
            returns = np.random.randn(100) * 0.01
            
            metrics = advanced_metrics.calculate(predictions, targets, returns)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0
            
            # Test Evolutionary Algorithm
            evo_config = type('EvolutionaryConfig', (), {
                'population_size': 20,
                'max_generations': 5,
                'crossover_probability': 0.8,
                'mutation_probability': 0.1,
                'tournament_size': 3,
                'elitism_size': 2,
                'convergence_threshold': 1e-6,
                'random_state': 42
            })()
            
            def simple_objective(params):
                return sum(params.values()) / len(params)
            
            evo_algorithm = TradingEvolutionaryAlgorithm(evo_config, simple_objective)
            parameter_space = {
                'param1': (0, 1),
                'param2': (0, 1),
                'param3': (0, 1)
            }
            
            result = evo_algorithm.optimize([simple_objective], parameter_space)
            assert result.success
            assert len(result.best_individuals) > 0
            
            # Test Feature Engineer
            feature_engineer = TradingFeatureEngineer()
            test_data = np.random.randn(50, 3)
            feature_result = feature_engineer.generate_features(test_data)
            assert feature_result.features.shape[0] == test_data.shape[0]
            assert feature_result.features.shape[1] >= test_data.shape[1]
            
            # Test Evaluation Metrics
            eval_metrics = TradingEvaluationMetrics()
            eval_result = eval_metrics.calculate(predictions, targets, returns)
            assert isinstance(eval_result, dict)
            assert len(eval_result) > 0
            
            return {'success': True, 'message': 'Optimization systems tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Optimization systems tests failed: {e}'}
    
    async def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test end-to-end system integration."""
        try:
            # Initialize all components
            state_manager = InMemoryStateManager()
            event_bus = EventBus()
            exchange_client = ExchangeClient()
            
            # Create model manager
            model_manager = ModelManager(state_manager, event_bus)
            
            # Get trading components
            analyst = model_manager.get_analyst()
            strategist = model_manager.get_strategist()
            tactician = model_manager.get_tactician()
            
            # Start all components
            await analyst.start()
            await strategist.start()
            await tactician.start()
            
            # Create supervisor
            supervisor = Supervisor(state_manager, event_bus)
            supervisor.register_component('analyst', analyst)
            supervisor.register_component('strategist', strategist)
            supervisor.register_component('tactician', tactician)
            
            await supervisor.start()
            
            # Test full trading cycle
            # 1. Get market data
            market_data = type('MarketData', (), {
                'symbol': 'BTCUSDT',
                'timestamp': datetime.now(),
                'open': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'close': 50500.0,
                'volume': 1000.0,
                'interval': '1m'
            })()
            
            # 2. Analyze market
            analysis_result = await analyst.analyze_market_data(market_data)
            
            # 3. Formulate strategy
            strategy_result = await strategist.formulate_strategy(analysis_result)
            
            # 4. Execute trade (if conditions are met)
            trade_decision = await tactician.execute_trade_decision(strategy_result, analysis_result)
            
            # 5. Monitor performance
            performance = await supervisor.monitor_performance()
            assert 'system_status' in performance
            
            # 6. Manage risk
            risk_management = await supervisor.manage_risk()
            assert 'risk_level' in risk_management
            
            # Stop all components
            await supervisor.stop()
            await tactician.stop()
            await strategist.stop()
            await analyst.stop()
            
            return {'success': True, 'message': 'End-to-end integration tests passed'}
            
        except Exception as e:
            return {'success': False, 'message': f'End-to-end integration tests failed: {e}'}


async def main():
    """Main test execution function."""
    tester = ProductionReadinessTester()
    results = await tester.run_all_tests()
    
    # Print detailed results
    print("\n" + "="*80)
    print("PRODUCTION READINESS TEST RESULTS")
    print("="*80)
    
    for test_name, result in results['test_results'].items():
        status = result['status']
        execution_time = result['execution_time']
        message = result['details']['message']
        
        emoji = "✅" if status == "PASSED" else "❌"
        print(f"{emoji} {test_name:<25} {status:<8} {execution_time:>6.2f}s - {message}")
    
    print("="*80)
    print(f"SUMMARY: {results['passed_tests']}/{results['total_tests']} tests passed ({results['success_rate']:.1%})")
    print(f"EXECUTION TIME: {results['total_execution_time']:.2f}s")
    print(f"PRODUCTION READY: {'YES' if results['production_ready'] else 'NO'}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run the tests
    results = asyncio.run(main())
    
    # Exit with appropriate code
    exit(0 if results['production_ready'] else 1)