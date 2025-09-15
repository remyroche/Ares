"""
A/B/C Testing Integration Example

This module demonstrates how to use the comprehensive A/B/C testing framework
for paper-trading multiple models simultaneously with all the integrated components.

Key Features Demonstrated:
- Complete A/B/C testing workflow
- Multi-model orchestration
- Realistic paper trading simulation
- Advanced risk management
- Statistical analysis and validation
- Performance monitoring and alerting
- Results visualization and reporting
- Configuration management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from pathlib import Path
import json

# Import all the framework components
from src.training.steps.backtesting.abc_testing_framework import (
    ABCTestingFramework, ABCTestingConfig, ABCTestingResults
)
from src.training.steps.backtesting.multi_model_orchestrator import (
    MultiModelOrchestrator, ModelConfig, OrchestrationConfig
)
from src.training.steps.backtesting.paper_trading_engine import (
    PaperTradingEngine, PaperTradingConfig, MarketData, OrderSide, OrderType
)
from src.training.steps.backtesting.risk_management import (
    RiskManager, RiskLimits, PositionSizingConfig, PositionSizingMethod
)
from src.training.steps.backtesting.statistical_analysis import (
    StatisticalAnalyzer, StatisticalTestConfig, StatisticalResults
)
from src.training.steps.backtesting.performance_monitoring import (
    PerformanceMonitor, MonitoringConfig, AlertConfig
)
from src.training.steps.backtesting.results_visualization import (
    ResultsVisualizer, VisualizationConfig
)
from src.training.steps.backtesting.configuration_management import (
    ConfigurationManager, ConfigurationEntry, ConfigurationScope, ConfigurationFormat
)

# Import model factory and registry
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType
from src.utils.ml_common.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ABCTestingIntegrationExample:
    """Comprehensive A/B/C testing integration example."""
    
    def __init__(self, config_dir: str = "config/abc_testing"):
        """Initialize the integration example."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('ABCTestingIntegrationExample')
        
        # Initialize all components
        self._initialize_components()
        
        self.logger.info("🚀 A/B/C Testing Integration Example initialized")
        self.logger.info(f"📁 Configuration directory: {self.config_dir}")
    
    def _initialize_components(self) -> None:
        """Initialize all framework components."""
        try:
            # Configuration Management
            self.config_manager = ConfigurationManager(str(self.config_dir))
            
            # Model Factory and Registry
            self.model_factory = EnhancedModelFactory()
            self.model_registry = ModelRegistry()
            
            # Risk Management
            self.risk_limits = RiskLimits(
                max_portfolio_risk=0.15,
                max_position_risk=0.03,
                max_correlation=0.60,
                max_drawdown=0.10,
                max_leverage=1.0,
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
            
            # Paper Trading Engine
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
                max_slippage_bps=30.0,
                latency_ms=(5, 50),
                volatility_multiplier=1.2,
                liquidity_factor=0.8
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
                },
                email_settings={
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "recipients": []
                }
            )
            
            # Results Visualization
            self.visualization_config = VisualizationConfig(
                output_dir=str(self.config_dir / "reports"),
                format="html",
                enable_interactive=True,
                include_statistical_tests=True,
                include_performance_metrics=True,
                include_risk_analysis=True,
                include_correlation_analysis=True
            )
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def run_complete_abc_test(self, test_config: Dict[str, Any]) -> ABCTestingResults:
        """Run a complete A/B/C test with all components integrated."""
        try:
            self.logger.info("🚀 Starting comprehensive A/B/C test")
            
            # Step 1: Create and save test configuration
            await self._create_test_configuration(test_config)
            
            # Step 2: Initialize models
            models = await self._initialize_models(test_config["models"])
            
            # Step 3: Set up paper trading engines for each model
            trading_engines = await self._setup_trading_engines(models)
            
            # Step 4: Initialize monitoring and visualization
            monitor = PerformanceMonitor(self.monitoring_config)
            visualizer = ResultsVisualizer(self.visualization_config)
            
            # Step 5: Create A/B/C testing framework
            abc_config = ABCTestingConfig(
                test_name=test_config["test_name"],
                test_description=test_config["test_description"],
                symbol=test_config["symbol"],
                exchange=test_config["exchange"],
                timeframe=test_config["timeframe"],
                start_date=datetime.fromisoformat(test_config["start_date"]),
                end_date=datetime.fromisoformat(test_config["end_date"]),
                model_configs=test_config["models"],
                statistical_testing=test_config.get("statistical_testing", {}),
                risk_management=test_config.get("risk_management", {}),
                monitoring_config=self.monitoring_config,
                visualization_config=self.visualization_config
            )
            
            abc_framework = ABCTestingFramework(abc_config)
            
            # Step 6: Create multi-model orchestrator
            orchestration_config = OrchestrationConfig(
                max_concurrent_models=len(models),
                enable_parallel_execution=True,
                enable_risk_management=True,
                enable_performance_monitoring=True,
                enable_real_time_alerts=True,
                risk_limits=self.risk_limits,
                position_sizing_config=self.position_sizing_config
            )
            
            orchestrator = MultiModelOrchestrator(
                models=models,
                trading_engines=trading_engines,
                risk_manager=self.risk_manager,
                monitor=monitor,
                config=orchestration_config
            )
            
            # Step 7: Run the A/B/C test
            self.logger.info("📊 Executing A/B/C test...")
            results = await abc_framework.execute(orchestrator)
            
            # Step 8: Generate comprehensive reports
            self.logger.info("📈 Generating reports...")
            await self._generate_reports(results, visualizer)
            
            # Step 9: Save results
            await self._save_test_results(results, test_config["test_name"])
            
            self.logger.info("✅ A/B/C test completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error running A/B/C test: {e}")
            raise
    
    async def _create_test_configuration(self, test_config: Dict[str, Any]) -> None:
        """Create and save test configuration."""
        try:
            # Create configuration entry
            config_entry = ConfigurationEntry(
                config_id=f"abc_test_{test_config['test_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=test_config["test_name"],
                scope=ConfigurationScope.TEST,
                format=ConfigurationFormat.JSON,
                content=test_config,
                schema_id="abc_testing",
                description=f"A/B/C test configuration for {test_config['test_name']}",
                tags=["abc_testing", "paper_trading", test_config["symbol"]],
                environment="production"
            )
            
            # Save configuration
            self.config_manager.save_configuration(config_entry)
            
            self.logger.info(f"✅ Test configuration saved: {test_config['test_name']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating test configuration: {e}")
            raise
    
    async def _initialize_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize all models for testing."""
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
                
                # Register model
                self.model_registry.save_model_with_metadata(
                    model=model,
                    model_name=model_name,
                    model_type=model_type.value,
                    metadata={
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_type": model_type.value,
                        "model_params": model_config.get("model_params", {}),
                        "initial_capital": model_config.get("initial_capital", 100000.0),
                        "max_position_size": model_config.get("max_position_size", 0.08),
                        "risk_per_trade": model_config.get("risk_per_trade", 0.02),
                        "created_at": datetime.now().isoformat(),
                        "test_config": model_config
                    }
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
    
    async def _setup_trading_engines(self, models: Dict[str, Any]) -> Dict[str, PaperTradingEngine]:
        """Set up paper trading engines for each model."""
        try:
            trading_engines = {}
            
            for model_id, model_data in models.items():
                # Create individual trading config for this model
                model_config = model_data["config"]
                trading_config = PaperTradingConfig(
                    initial_capital=model_config.get("initial_capital", 100000.0),
                    max_position_size=model_config.get("max_position_size", 0.08),
                    risk_per_trade=model_config.get("risk_per_trade", 0.02),
                    commission_rate=self.paper_trading_config.commission_rate,
                    slippage_model=self.paper_trading_config.slippage_model,
                    market_impact_model=self.paper_trading_config.market_impact_model,
                    enable_slippage=self.paper_trading_config.enable_slippage,
                    enable_market_impact=self.paper_trading_config.enable_market_impact,
                    enable_partial_fills=self.paper_trading_config.enable_partial_fills,
                    max_slippage_bps=self.paper_trading_config.max_slippage_bps,
                    latency_ms=self.paper_trading_config.latency_ms,
                    volatility_multiplier=self.paper_trading_config.volatility_multiplier,
                    liquidity_factor=self.paper_trading_config.liquidity_factor
                )
                
                # Create trading engine
                trading_engine = PaperTradingEngine(trading_config)
                trading_engines[model_id] = trading_engine
                
                self.logger.info(f"✅ Trading engine created for model: {model_id}")
            
            return trading_engines
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up trading engines: {e}")
            raise
    
    async def _generate_reports(self, results: ABCTestingResults, visualizer: ResultsVisualizer) -> None:
        """Generate comprehensive reports."""
        try:
            # Generate performance comparison report
            await visualizer.generate_performance_comparison_report(results)
            
            # Generate statistical analysis report
            await visualizer.generate_statistical_analysis_report(results)
            
            # Generate risk analysis report
            await visualizer.generate_risk_analysis_report(results)
            
            # Generate correlation analysis report
            await visualizer.generate_correlation_analysis_report(results)
            
            # Generate executive summary
            await visualizer.generate_executive_summary(results)
            
            # Generate comprehensive dashboard
            await visualizer.generate_comprehensive_dashboard(results)
            
            self.logger.info("✅ All reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating reports: {e}")
            raise
    
    async def _save_test_results(self, results: ABCTestingResults, test_name: str) -> None:
        """Save test results to disk."""
        try:
            # Create results directory
            results_dir = self.config_dir / "results" / test_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results as JSON
            results_file = results_dir / f"{test_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert results to serializable format
            results_data = {
                "test_name": test_name,
                "test_config": results.test_config.__dict__ if hasattr(results, 'test_config') else {},
                "model_results": {},
                "statistical_results": {},
                "performance_metrics": {},
                "risk_metrics": {},
                "recommendations": results.recommendations if hasattr(results, 'recommendations') else [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Add model results
            if hasattr(results, 'model_results'):
                for model_id, model_result in results.model_results.items():
                    results_data["model_results"][model_id] = {
                        "performance_metrics": model_result.performance_metrics if hasattr(model_result, 'performance_metrics') else {},
                        "trades": len(model_result.trades) if hasattr(model_result, 'trades') else 0,
                        "final_value": model_result.final_value if hasattr(model_result, 'final_value') else 0,
                        "total_return": model_result.total_return if hasattr(model_result, 'total_return') else 0
                    }
            
            # Save to file
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Test results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving test results: {e}")
            raise


async def run_example_abc_test():
    """Run an example A/B/C test with multiple models."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting A/B/C Testing Integration Example")
    
    try:
        # Initialize the integration example
        integration_example = ABCTestingIntegrationExample("config/abc_testing_example")
        
        # Define test configuration
        test_config = {
            "test_name": "Crypto_Trading_Models_ABC_Test",
            "test_description": "Comprehensive A/B/C test comparing multiple crypto trading models",
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-03-31T23:59:59",
            "models": [
                {
                    "model_id": "model_a",
                    "model_name": "RandomForest_Model",
                    "model_type": "random_forest",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.08,
                    "risk_per_trade": 0.02,
                    "model_params": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "min_samples_split": 5,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_b",
                    "model_name": "LightGBM_Model",
                    "model_type": "lightgbm",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.08,
                    "risk_per_trade": 0.02,
                    "model_params": {
                        "n_estimators": 200,
                        "max_depth": 8,
                        "learning_rate": 0.1,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_c",
                    "model_name": "XGBoost_Model",
                    "model_type": "xgboost",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.08,
                    "risk_per_trade": 0.02,
                    "model_params": {
                        "n_estimators": 150,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_d",
                    "model_name": "TabNet_Model",
                    "model_type": "tabnet",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.08,
                    "risk_per_trade": 0.02,
                    "model_params": {
                        "n_d": 64,
                        "n_a": 64,
                        "n_steps": 5,
                        "gamma": 1.5,
                        "lambda_sparse": 1e-3
                    }
                }
            ],
            "statistical_testing": {
                "enable_statistical_testing": True,
                "confidence_level": 0.95,
                "alpha": 0.05,
                "min_sample_size": 100,
                "enable_multiple_testing_correction": True,
                "correction_method": "bonferroni",
                "effect_size_threshold": 0.2,
                "power_analysis": True,
                "power_threshold": 0.8
            },
            "risk_management": {
                "global_risk_limit": 0.15,
                "max_concurrent_positions": 8,
                "correlation_threshold": 0.60,
                "enable_circuit_breakers": True,
                "circuit_breaker_threshold": 0.08
            }
        }
        
        # Run the complete A/B/C test
        results = await integration_example.run_complete_abc_test(test_config)
        
        # Print summary
        logger.info("📊 A/B/C Test Results Summary:")
        logger.info(f"   Test Name: {test_config['test_name']}")
        logger.info(f"   Symbol: {test_config['symbol']}")
        logger.info(f"   Timeframe: {test_config['timeframe']}")
        logger.info(f"   Models Tested: {len(test_config['models'])}")
        logger.info(f"   Test Period: {test_config['start_date']} to {test_config['end_date']}")
        
        if hasattr(results, 'model_results'):
            logger.info("   Model Performance:")
            for model_id, model_result in results.model_results.items():
                total_return = getattr(model_result, 'total_return', 0)
                final_value = getattr(model_result, 'final_value', 0)
                logger.info(f"     {model_id}: {total_return:.2%} return, ${final_value:,.2f} final value")
        
        logger.info("✅ A/B/C Testing Integration Example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in A/B/C testing example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example_abc_test())