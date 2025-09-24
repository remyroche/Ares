"""
TPSL Parameter Optimization Example

This example demonstrates how to optimize Take Profit/Stop Loss (TPSL) parameters
for different models using grid search and advanced optimization techniques.

Key Features Demonstrated:
- TPSL parameter grid search optimization
- Multiple optimization strategies (Fixed, ATR-based, Volatility-based)
- Performance comparison of different TPSL configurations
- Statistical validation of TPSL effectiveness
- Automated parameter selection based on historical performance
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path
import time

from src.utils.leverage_constants import MIN_LEVERAGE, MAX_LEVERAGE

# Import enhanced framework components
from .enhanced_abc_testing_framework import (
    EnhancedABCTestingFramework, TPSLConfig, TPSLStrategy, TPSLMode,
    TPSLManager, TPSLOptimizationResult
)
from .abc_testing_framework import ABCTestingConfig
from .multi_model_orchestrator import MultiModelOrchestrator, OrchestrationConfig
from .paper_trading_engine import PaperTradingEngine, PaperTradingConfig
from .risk_management import RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
from .statistical_analysis import StatisticalAnalyzer, StatisticalTestConfig
from .performance_monitoring import PerformanceMonitor, MonitoringConfig
from .results_visualization import ResultsVisualizer, VisualizationConfig
from .configuration_management import ConfigurationManager

# Import model factory
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType

logger = logging.getLogger(__name__)


class TPSLOptimizationExample:
    """Comprehensive TPSL parameter optimization example."""
    
    def __init__(self, config_dir: str = "config/tpsl_optimization"):
        """Initialize the TPSL optimization example."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('TPSLOptimizationExample')
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🚀 TPSL Optimization Example initialized")
        self.logger.info(f"📁 Configuration directory: {self.config_dir}")
    
    def _initialize_components(self) -> None:
        """Initialize all framework components."""
        try:
            # Model Factory
            self.model_factory = EnhancedModelFactory()
            
            # Configuration Manager
            self.config_manager = ConfigurationManager(str(self.config_dir))
            
            # Risk Management
            self.risk_limits = RiskLimits(
                max_portfolio_risk=0.18,
                max_position_risk=0.035,
                max_correlation=0.65,
                max_drawdown=0.10,
                MAX_LEVERAGE,
                max_concurrent_positions=8,
                max_daily_loss=0.05,
                enable_circuit_breakers=True,
                circuit_breaker_threshold=0.08
            )
            
            self.position_sizing_config = PositionSizingConfig(
                method=PositionSizingMethod.FIXED_FRACTIONAL,
                base_risk_per_trade=0.02,
                max_position_size=0.08,
                min_position_size=0.005,
                volatility_lookback=20,
                correlation_lookback=60,
                kelly_fraction=0.25,
                atr_multiplier=2.0,
                enable_dynamic_sizing=True,
                enable_correlation_adjustment=True
            )
            
            self.risk_manager = RiskManager(self.risk_limits, self.position_sizing_config)
            
            # Paper Trading Configuration
            self.paper_trading_config = PaperTradingConfig(
                initial_capital=100000.0,
                max_position_size=0.08,
                risk_per_trade=0.02,
                commission_rate=0.001,
                slippage_model="sqrt",
                market_impact_model="sqrt",
                enable_slippage=True,
                enable_market_impact=True,
                enable_partial_fills=True,
                max_slippage_bps=20.0,
                latency_ms=(5, 30),
                volatility_multiplier=1.0,
                liquidity_factor=1.0
            )
            
            # Statistical Analysis
            self.statistical_config = StatisticalTestConfig(
                confidence_level=0.95,
                alpha=0.05,
                min_sample_size=100,
                enable_multiple_testing_correction=True,
                correction_method="bonferroni",
                effect_size_threshold=0.2,
                power_analysis=True,
                power_threshold=0.8
            )
            
            # Performance Monitoring
            self.monitoring_config = MonitoringConfig(
                monitoring_interval=30,
                enable_alerting=True,
                alert_cooldown_minutes=10,
                performance_thresholds={
                    "max_drawdown": 0.12,
                    "min_sharpe_ratio": 0.8,
                    "max_volatility": 0.25,
                    "min_win_rate": 0.45
                }
            )
            
            # Results Visualization
            self.visualization_config = VisualizationConfig(
                output_dir="outcomes/backtesting/tpsl_optimization",
                format="html",
                enable_interactive=True,
                include_statistical_tests=True,
                include_performance_metrics=True,
                include_risk_analysis=True,
                include_correlation_analysis=True,
                include_tpsl_analysis=True
            )
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def run_tpsl_optimization(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive TPSL parameter optimization."""
        try:
            self.logger.info("🚀 Starting TPSL parameter optimization")
            
            # Step 1: Create optimization configuration
            await self._create_optimization_configuration(optimization_config)
            
            # Step 2: Initialize models for optimization
            models = await self._initialize_models(optimization_config["models"])
            
            # Step 3: Generate historical data for optimization
            historical_data = await self._generate_historical_data(optimization_config)
            
            # Step 4: Run TPSL optimization for each model
            optimization_results = {}
            
            for model_id, model_data in models.items():
                self.logger.info(f"🔍 Optimizing TPSL parameters for model: {model_id}")
                
                # Create TPSL manager
                base_tpsl_config = TPSLConfig()
                tpsl_manager = TPSLManager(base_tpsl_config)
                
                # Define optimization parameters
                optimization_params = self._create_optimization_parameters(
                    optimization_config["optimization_strategies"][model_id]
                )
                
                # Run optimization
                start_time = time.time()
                optimization_result = tpsl_manager.optimize_tpsl_parameters(
                    historical_data=historical_data,
                    symbol=optimization_config["symbol"],
                    position_side=optimization_config.get("position_side", "buy"),
                    optimization_params=optimization_params
                )
                optimization_time = time.time() - start_time
                
                optimization_results[model_id] = {
                    "optimization_result": optimization_result,
                    "optimization_time": optimization_time,
                    "model_info": model_data["metadata"]
                }
                
                self.logger.info(f"✅ TPSL optimization completed for {model_id} in {optimization_time:.2f}s")
                self.logger.info(f"📊 Best score: {optimization_result.best_score:.4f}")
                self.logger.info(f"📊 Best TP: {optimization_result.best_config.take_profit_pct:.1%}")
                self.logger.info(f"📊 Best SL: {optimization_result.best_config.stop_loss_pct:.1%}")
            
            # Step 5: Compare optimization results
            comparison_analysis = await self._compare_optimization_results(optimization_results)
            
            # Step 6: Generate optimization reports
            await self._generate_optimization_reports(optimization_results, comparison_analysis)
            
            # Step 7: Save optimization results
            await self._save_optimization_results(optimization_results, comparison_analysis, optimization_config)
            
            self.logger.info("✅ TPSL parameter optimization completed successfully")
            
            return {
                "optimization_results": optimization_results,
                "comparison_analysis": comparison_analysis,
                "optimization_config": optimization_config
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error running TPSL optimization: {e}")
            raise
    
    async def _create_optimization_configuration(self, optimization_config: Dict[str, Any]) -> None:
        """Create and save optimization configuration."""
        try:
            config_entry = ConfigurationEntry(
                config_id=f"tpsl_optimization_{optimization_config['test_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=optimization_config["test_name"],
                scope=ConfigurationScope.TEST,
                format=ConfigurationFormat.JSON,
                content=optimization_config,
                schema_id="tpsl_optimization",
                description=f"TPSL optimization configuration for {optimization_config['test_name']}",
                tags=["tpsl", "optimization", "parameter_tuning", optimization_config["symbol"]],
                environment="production"
            )
            
            self.config_manager.save_configuration(config_entry)
            self.logger.info(f"✅ Optimization configuration saved: {optimization_config['test_name']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating optimization configuration: {e}")
            raise
    
    async def _initialize_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize models for optimization."""
        try:
            models = {}
            
            for model_config in model_configs:
                model_id = model_config["model_id"]
                model_name = model_config["model_name"]
                model_type = ModelType(model_config["model_type"])
                
                # Create model using factory
                model = self.model_factory.create_model(
                    model_type=model_type,
                    model_params=model_config.get("model_params", {}),
                    model_name=model_name
                )
                
                models[model_id] = {
                    "model": model,
                    "config": model_config,
                    "metadata": {
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_type": model_type.value
                    }
                }
                
                self.logger.info(f"✅ Model initialized: {model_name} ({model_type.value})")
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing models: {e}")
            raise
    
    async def _generate_historical_data(self, optimization_config: Dict[str, Any]) -> pd.DataFrame:
        """Generate or load historical data for optimization."""
        try:
            # In a real implementation, this would load actual historical data
            # For this example, we'll generate synthetic data
            
            start_date = datetime.fromisoformat(optimization_config["start_date"])
            end_date = datetime.fromisoformat(optimization_config["end_date"])
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
            
            # Generate synthetic price data
            np.random.seed(42)
            n_periods = len(date_range)
            
            # Generate price series with trend and volatility
            returns = np.random.normal(0.0001, 0.02, n_periods)  # 0.01% mean return, 2% volatility
            prices = 50000 * np.exp(np.cumsum(returns))  # Start at $50,000
            
            # Generate volume data
            volumes = np.random.lognormal(10, 0.5, n_periods)
            
            # Create DataFrame
            historical_data = pd.DataFrame({
                'timestamp': date_range,
                'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
                'close': prices,
                'volume': volumes
            })
            
            historical_data.set_index('timestamp', inplace=True)
            
            self.logger.info(f"✅ Historical data generated: {len(historical_data)} periods")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"❌ Error generating historical data: {e}")
            raise
    
    def _create_optimization_parameters(self, strategy: str) -> Dict[str, List[float]]:
        """Create optimization parameters based on strategy."""
        try:
            if strategy == "atr_based":
                return {
                    "atr_multiplier_tp": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                    "atr_multiplier_sl": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
                }
            elif strategy == "dynamic":
                return {
                    "take_profit_pct": [0.01, 0.015, 0.02, 0.025, 0.03],
                    "stop_loss_pct": [0.008, 0.01, 0.012, 0.015, 0.02],
                    "dynamic_adjustment_factor": [0.3, 0.5, 0.7, 0.9, 1.1]
                }
            elif strategy == "trailing":
                return {
                    "take_profit_pct": [0.015, 0.02, 0.025, 0.03, 0.035],
                    "stop_loss_pct": [0.01, 0.012, 0.015, 0.018, 0.02],
                    "trailing_start_pct": [0.008, 0.01, 0.012, 0.015, 0.018],
                    "trailing_step_pct": [0.003, 0.005, 0.007, 0.01, 0.012]
                }
            elif strategy == "scaling":
                return {
                    "take_profit_pct": [0.012, 0.015, 0.018, 0.02, 0.025],
                    "stop_loss_pct": [0.008, 0.01, 0.012, 0.015, 0.018],
                    "scale_out_levels": [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]
                }
            elif strategy == "momentum_based":
                return {
                    "take_profit_pct": [0.015, 0.02, 0.025, 0.03],
                    "stop_loss_pct": [0.01, 0.012, 0.015, 0.018],
                    "momentum_period": [5, 10, 15, 20],
                    "momentum_threshold": [0.3, 0.5, 0.7, 0.9]
                }
            elif strategy == "support_resistance":
                return {
                    "take_profit_pct": [0.015, 0.02, 0.025, 0.03],
                    "stop_loss_pct": [0.01, 0.012, 0.015, 0.018],
                    "sr_lookback": [10, 15, 20, 25],
                    "sr_buffer_pct": [0.001, 0.002, 0.003, 0.005]
                }
            elif strategy == "confidence_based":
                return {
                    "take_profit_pct": [0.015, 0.02, 0.025, 0.03],
                    "stop_loss_pct": [0.01, 0.012, 0.015, 0.018],
                    "confidence_threshold_high": [0.7, 0.8, 0.9],
                    "confidence_threshold_medium": [0.5, 0.6, 0.7],
                    "confidence_threshold_low": [0.3, 0.4, 0.5],
                    "analyst_confidence_weight": [0.4, 0.5, 0.6, 0.7],
                    "tactician_confidence_weight": [0.3, 0.4, 0.5, 0.6]
                }
            else:
                # Default to ATR-based strategy
                return {
                    "atr_multiplier_tp": [1.5, 2.0, 2.5, 3.0],
                    "atr_multiplier_sl": [0.8, 1.0, 1.2, 1.5]
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error creating optimization parameters: {e}")
            return {"take_profit_pct": [0.02], "stop_loss_pct": [0.01]}
    
    async def _compare_optimization_results(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare optimization results across models."""
        try:
            self.logger.info("📊 Comparing optimization results...")
            
            comparison_data = []
            
            for model_id, result_data in optimization_results.items():
                optimization_result = result_data["optimization_result"]
                
                comparison_data.append({
                    "model_id": model_id,
                    "model_name": result_data["model_info"]["model_name"],
                    "model_type": result_data["model_info"]["model_type"],
                    "best_score": optimization_result.best_score,
                    "best_tp": optimization_result.best_config.take_profit_pct,
                    "best_sl": optimization_result.best_config.stop_loss_pct,
                    "strategy": optimization_result.best_config.strategy.value,
                    "optimization_time": result_data["optimization_time"],
                    "total_tests": optimization_result.total_tests,
                    "parameter_importance": optimization_result.parameter_importance
                })
            
            # Sort by best score
            comparison_data.sort(key=lambda x: x["best_score"], reverse=True)
            
            # Calculate statistics
            scores = [data["best_score"] for data in comparison_data]
            optimization_times = [data["optimization_time"] for data in comparison_data]
            
            comparison_analysis = {
                "ranked_results": comparison_data,
                "best_model": comparison_data[0] if comparison_data else None,
                "statistics": {
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "min_score": np.min(scores),
                    "max_score": np.max(scores),
                    "mean_optimization_time": np.mean(optimization_times),
                    "total_optimization_time": np.sum(optimization_times)
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Comparison analysis completed. Best model: {comparison_analysis['best_model']['model_id'] if comparison_analysis['best_model'] else 'N/A'}")
            return comparison_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing optimization results: {e}")
            return {}
    
    async def _generate_optimization_reports(self, optimization_results: Dict[str, Any], 
                                           comparison_analysis: Dict[str, Any]) -> None:
        """Generate optimization reports."""
        try:
            # Create optimization report
            optimization_report = {
                "title": "TPSL Parameter Optimization Report",
                "timestamp": datetime.now().isoformat(),
                "optimization_results": {},
                "comparison_analysis": comparison_analysis,
                "summary": {
                    "total_models": len(optimization_results),
                    "best_model": comparison_analysis.get("best_model", {}).get("model_id", "N/A"),
                    "best_score": comparison_analysis.get("best_model", {}).get("best_score", 0.0),
                    "total_optimization_time": comparison_analysis.get("statistics", {}).get("total_optimization_time", 0.0)
                }
            }
            
            # Add detailed results for each model
            for model_id, result_data in optimization_results.items():
                optimization_result = result_data["optimization_result"]
                
                optimization_report["optimization_results"][model_id] = {
                    "model_info": result_data["model_info"],
                    "best_config": {
                        "strategy": optimization_result.best_config.strategy.value,
                        "take_profit_pct": optimization_result.best_config.take_profit_pct,
                        "stop_loss_pct": optimization_result.best_config.stop_loss_pct,
                        "atr_multiplier_tp": optimization_result.best_config.atr_multiplier_tp,
                        "atr_multiplier_sl": optimization_result.best_config.atr_multiplier_sl,
                        "volatility_multiplier_tp": optimization_result.best_config.volatility_multiplier_tp,
                        "volatility_multiplier_sl": optimization_result.best_config.volatility_multiplier_sl
                    },
                    "best_score": optimization_result.best_score,
                    "optimization_time": result_data["optimization_time"],
                    "total_tests": optimization_result.total_tests,
                    "parameter_importance": optimization_result.parameter_importance
                }
            
            # Save optimization report
            optimization_report_file = Path("outcomes/backtesting") / "tpsl_optimization_report.json"
            optimization_report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(optimization_report_file, 'w') as f:
                json.dump(optimization_report, f, indent=2, default=str)
            
            self.logger.info("✅ Optimization reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating optimization reports: {e}")
    
    async def _save_optimization_results(self, optimization_results: Dict[str, Any], 
                                       comparison_analysis: Dict[str, Any], 
                                       optimization_config: Dict[str, Any]) -> None:
        """Save comprehensive optimization results."""
        try:
            results_dir = Path("generated/backtesting") / "tpsl_optimization" / optimization_config["test_name"]
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = results_dir / f"tpsl_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results_data = {
                "optimization_config": optimization_config,
                "optimization_results": optimization_results,
                "comparison_analysis": comparison_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Optimization results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving optimization results: {e}")
            raise


async def run_tpsl_optimization_example():
    """Run a comprehensive TPSL parameter optimization example."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting TPSL Parameter Optimization Example")
    
    try:
        # Initialize the example
        example = TPSLOptimizationExample("config/tpsl_optimization_example")
        
        # Define comprehensive optimization configuration
        optimization_config = {
            "test_name": "Crypto_TPSL_Parameter_Optimization",
            "test_description": "Comprehensive TPSL parameter optimization for multiple crypto trading models",
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "timeframe": "1h",
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T23:59:59",
            "position_side": "buy",
            "models": [
                {
                    "model_id": "model_a",
                    "model_name": "RandomForest_Optimization",
                    "model_type": "random_forest",
                    "model_params": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_b",
                    "model_name": "LightGBM_Optimization",
                    "model_type": "lightgbm",
                    "model_params": {
                        "n_estimators": 200,
                        "max_depth": 8,
                        "learning_rate": 0.1,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_c",
                    "model_name": "XGBoost_Optimization",
                    "model_type": "xgboost",
                    "model_params": {
                        "n_estimators": 150,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_d",
                    "model_name": "TabNet_Optimization",
                    "model_type": "tabnet",
                    "model_params": {
                        "n_d": 64,
                        "n_a": 64,
                        "n_steps": 5,
                        "random_state": 42
                    }
                }
            ],
            "optimization_strategies": {
                "model_a": "atr_based",
                "model_b": "dynamic",
                "model_c": "confidence_based",
                "model_d": "trailing"
            }
        }
        
        # Run the optimization
        results = await example.run_tpsl_optimization(optimization_config)
        
        # Print summary
        logger.info("📊 TPSL Parameter Optimization Results Summary:")
        logger.info(f"   Test Name: {optimization_config['test_name']}")
        logger.info(f"   Symbol: {optimization_config['symbol']}")
        logger.info(f"   Models Optimized: {len(optimization_config['models'])}")
        logger.info(f"   Optimization Period: {optimization_config['start_date']} to {optimization_config['end_date']}")
        
        # Print optimization results
        comparison_analysis = results.get("comparison_analysis", {})
        if comparison_analysis:
            best_model = comparison_analysis.get("best_model", {})
            if best_model:
                logger.info(f"   Best Model: {best_model['model_id']} ({best_model['model_name']})")
                logger.info(f"   Best Score: {best_model['best_score']:.4f}")
                logger.info(f"   Best TP: {best_model['best_tp']:.1%}")
                logger.info(f"   Best SL: {best_model['best_sl']:.1%}")
                logger.info(f"   Strategy: {best_model['strategy']}")
            
            statistics = comparison_analysis.get("statistics", {})
            if statistics:
                logger.info(f"   Mean Score: {statistics.get('mean_score', 0):.4f}")
                logger.info(f"   Total Optimization Time: {statistics.get('total_optimization_time', 0):.2f}s")
        
        # Print ranked results
        ranked_results = comparison_analysis.get("ranked_results", [])
        if ranked_results:
            logger.info("   Model Rankings:")
            for i, result in enumerate(ranked_results[:3]):
                logger.info(f"     {i+1}. {result['model_id']}: {result['best_score']:.4f} ({result['strategy']})")
        
        logger.info("✅ TPSL Parameter Optimization Example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in TPSL optimization example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_tpsl_optimization_example())