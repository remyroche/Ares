"""
Multi-Model TPSL Testing Example

This example demonstrates how to use the enhanced A/B/C testing framework
to test 5+ models simultaneously with different TPSL (Take Profit/Stop Loss)
parameters and strategies.

Key Features Demonstrated:
- Testing 5+ models simultaneously (A/B/C/D/E/F testing)
- Multiple TPSL strategies and parameters
- TPSL parameter optimization
- Performance comparison across models and TPSL configurations
- Statistical analysis of TPSL effectiveness
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

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


class MultiModelTPSLExample:
    """Comprehensive example of multi-model testing with TPSL optimization."""
    
    def __init__(self, config_dir: str = "config/multi_model_tpsl"):
        """Initialize the multi-model TPSL example."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('MultiModelTPSLExample')
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🚀 Multi-Model TPSL Example initialized")
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
                max_portfolio_risk=0.20,
                max_position_risk=0.04,
                max_correlation=0.70,
                max_drawdown=0.12,
                max_leverage=1.0,
                max_concurrent_positions=12,  # Support more models
                max_daily_loss=0.06,
                enable_circuit_breakers=True,
                circuit_breaker_threshold=0.10
            )
            
            self.position_sizing_config = PositionSizingConfig(
                method=PositionSizingMethod.FIXED_FRACTIONAL,
                base_risk_per_trade=0.025,
                max_position_size=0.10,
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
                max_position_size=0.10,
                risk_per_trade=0.025,
                commission_rate=0.001,
                slippage_model="sqrt",
                market_impact_model="sqrt",
                enable_slippage=True,
                enable_market_impact=True,
                enable_partial_fills=True,
                max_slippage_bps=25.0,
                latency_ms=(5, 40),
                volatility_multiplier=1.1,
                liquidity_factor=0.9
            )
            
            # Statistical Analysis
            self.statistical_config = StatisticalTestConfig(
                confidence_level=0.95,
                alpha=0.05,
                min_sample_size=150,
                enable_multiple_testing_correction=True,
                correction_method="bonferroni",
                effect_size_threshold=0.25,
                power_analysis=True,
                power_threshold=0.85
            )
            
            # Performance Monitoring
            self.monitoring_config = MonitoringConfig(
                monitoring_interval=20,
                enable_alerting=True,
                alert_cooldown_minutes=8,
                performance_thresholds={
                    "max_drawdown": 0.15,
                    "min_sharpe_ratio": 1.0,
                    "max_volatility": 0.30,
                    "min_win_rate": 0.50
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
                include_correlation_analysis=True,
                include_tpsl_analysis=True  # New feature for TPSL analysis
            )
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def run_multi_model_tpsl_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a comprehensive multi-model TPSL test."""
        try:
            self.logger.info("🚀 Starting multi-model TPSL test")
            
            # Step 1: Create test configuration
            await self._create_test_configuration(test_config)
            
            # Step 2: Initialize models (5+ models)
            models = await self._initialize_models(test_config["models"])
            
            # Step 3: Create TPSL configurations for each model
            tpsl_configs = await self._create_tpsl_configurations(test_config["tpsl_configs"])
            
            # Step 4: Set up paper trading engines
            trading_engines = await self._setup_trading_engines(models)
            
            # Step 5: Initialize monitoring and visualization
            monitor = PerformanceMonitor(self.monitoring_config)
            visualizer = ResultsVisualizer(self.visualization_config)
            
            # Step 6: Create enhanced A/B/C testing framework
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
            
            # Create enhanced framework with TPSL support
            enhanced_framework = EnhancedABCTestingFramework(abc_config, tpsl_configs)
            
            # Step 7: Create multi-model orchestrator
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
            
            # Step 8: Run the enhanced A/B/C test
            self.logger.info("📊 Executing enhanced A/B/C test with TPSL...")
            results = await enhanced_framework.execute(orchestrator)
            
            # Step 9: Analyze TPSL performance
            tpsl_analysis = await self._analyze_tpsl_performance(enhanced_framework)
            
            # Step 10: Generate comprehensive reports
            self.logger.info("📈 Generating comprehensive reports...")
            await self._generate_comprehensive_reports(results, tpsl_analysis, visualizer)
            
            # Step 11: Save results
            await self._save_test_results(results, tpsl_analysis, test_config["test_name"])
            
            self.logger.info("✅ Multi-model TPSL test completed successfully")
            
            return {
                "test_results": results,
                "tpsl_analysis": tpsl_analysis,
                "test_config": test_config
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error running multi-model TPSL test: {e}")
            raise
    
    async def _create_test_configuration(self, test_config: Dict[str, Any]) -> None:
        """Create and save test configuration."""
        try:
            config_entry = ConfigurationEntry(
                config_id=f"multi_model_tpsl_{test_config['test_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=test_config["test_name"],
                scope=ConfigurationScope.TEST,
                format=ConfigurationFormat.JSON,
                content=test_config,
                schema_id="multi_model_tpsl",
                description=f"Multi-model TPSL test configuration for {test_config['test_name']}",
                tags=["multi_model", "tpsl", "abc_testing", test_config["symbol"]],
                environment="production"
            )
            
            self.config_manager.save_configuration(config_entry)
            self.logger.info(f"✅ Test configuration saved: {test_config['test_name']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating test configuration: {e}")
            raise
    
    async def _initialize_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize 5+ models for testing."""
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
            
            self.logger.info(f"📊 Total models initialized: {len(models)}")
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing models: {e}")
            raise
    
    async def _create_tpsl_configurations(self, tpsl_configs: Dict[str, Dict[str, Any]]) -> Dict[str, TPSLConfig]:
        """Create TPSL configurations for each model."""
        try:
            tpsl_configurations = {}
            
            for model_id, tpsl_params in tpsl_configs.items():
                # Create TPSL configuration
                tpsl_config = TPSLConfig(
                    strategy=TPSLStrategy(tpsl_params.get("strategy", "fixed")),
                    mode=TPSLMode(tpsl_params.get("mode", "immediate")),
                    take_profit_pct=tpsl_params.get("take_profit_pct", 0.02),
                    stop_loss_pct=tpsl_params.get("stop_loss_pct", 0.01),
                    atr_multiplier_tp=tpsl_params.get("atr_multiplier_tp", 2.0),
                    atr_multiplier_sl=tpsl_params.get("atr_multiplier_sl", 1.0),
                    volatility_multiplier_tp=tpsl_params.get("volatility_multiplier_tp", 1.5),
                    volatility_multiplier_sl=tpsl_params.get("volatility_multiplier_sl", 1.0),
                    enable_breakeven=tpsl_params.get("enable_breakeven", True),
                    enable_partial_tp=tpsl_params.get("enable_partial_tp", False),
                    enable_trailing_sl=tpsl_params.get("enable_trailing_sl", False),
                    max_risk_per_trade=tpsl_params.get("max_risk_per_trade", 0.025),
                    min_risk_reward_ratio=tpsl_params.get("min_risk_reward_ratio", 1.5)
                )
                
                tpsl_configurations[model_id] = tpsl_config
                self.logger.info(f"✅ TPSL config created for {model_id}: {tpsl_config.strategy.value}")
            
            return tpsl_configurations
            
        except Exception as e:
            self.logger.error(f"❌ Error creating TPSL configurations: {e}")
            raise
    
    async def _setup_trading_engines(self, models: Dict[str, Any]) -> Dict[str, PaperTradingEngine]:
        """Set up paper trading engines for each model."""
        try:
            trading_engines = {}
            
            for model_id, model_data in models.items():
                model_config = model_data["config"]
                trading_config = PaperTradingConfig(
                    initial_capital=model_config.get("initial_capital", 100000.0),
                    max_position_size=model_config.get("max_position_size", 0.10),
                    risk_per_trade=model_config.get("risk_per_trade", 0.025),
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
                
                trading_engine = PaperTradingEngine(trading_config)
                trading_engines[model_id] = trading_engine
                
                self.logger.info(f"✅ Trading engine created for model: {model_id}")
            
            return trading_engines
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up trading engines: {e}")
            raise
    
    async def _analyze_tpsl_performance(self, enhanced_framework: EnhancedABCTestingFramework) -> Dict[str, Any]:
        """Analyze TPSL performance across all models."""
        try:
            self.logger.info("📊 Analyzing TPSL performance...")
            
            # Get TPSL performance summary
            tpsl_summary = enhanced_framework.get_tpsl_performance_summary()
            
            # Calculate TPSL effectiveness metrics
            tpsl_effectiveness = {}
            
            for model_id, metrics in tpsl_summary.items():
                effectiveness_score = 0.0
                
                # Weighted effectiveness score
                if metrics.get("total_trades", 0) > 0:
                    win_rate = metrics.get("win_rate", 0.0)
                    avg_risk_reward = metrics.get("avg_risk_reward", 0.0)
                    tp_effectiveness = metrics.get("tp_effectiveness", 0.0)
                    sl_effectiveness = metrics.get("sl_effectiveness", 0.0)
                    
                    # Calculate composite effectiveness score
                    effectiveness_score = (
                        win_rate * 0.3 +
                        min(avg_risk_reward / 2.0, 1.0) * 0.3 +
                        tp_effectiveness * 0.2 +
                        sl_effectiveness * 0.2
                    )
                
                tpsl_effectiveness[model_id] = {
                    "effectiveness_score": effectiveness_score,
                    "metrics": metrics
                }
            
            # Rank models by TPSL effectiveness
            ranked_models = sorted(
                tpsl_effectiveness.items(),
                key=lambda x: x[1]["effectiveness_score"],
                reverse=True
            )
            
            analysis = {
                "tpsl_summary": tpsl_summary,
                "tpsl_effectiveness": tpsl_effectiveness,
                "ranked_models": ranked_models,
                "best_tpsl_model": ranked_models[0][0] if ranked_models else None,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ TPSL analysis completed. Best model: {analysis['best_tpsl_model']}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing TPSL performance: {e}")
            return {}
    
    async def _generate_comprehensive_reports(self, results: Any, tpsl_analysis: Dict[str, Any], 
                                            visualizer: ResultsVisualizer) -> None:
        """Generate comprehensive reports including TPSL analysis."""
        try:
            # Generate standard reports
            await visualizer.generate_performance_comparison_report(results)
            await visualizer.generate_statistical_analysis_report(results)
            await visualizer.generate_risk_analysis_report(results)
            await visualizer.generate_correlation_analysis_report(results)
            
            # Generate TPSL-specific reports
            await self._generate_tpsl_reports(tpsl_analysis, visualizer)
            
            # Generate executive summary
            await visualizer.generate_executive_summary(results)
            
            # Generate comprehensive dashboard
            await visualizer.generate_comprehensive_dashboard(results)
            
            self.logger.info("✅ All reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating reports: {e}")
            raise
    
    async def _generate_tpsl_reports(self, tpsl_analysis: Dict[str, Any], visualizer: ResultsVisualizer) -> None:
        """Generate TPSL-specific reports."""
        try:
            # Create TPSL performance report
            tpsl_report = {
                "title": "TPSL Performance Analysis Report",
                "timestamp": datetime.now().isoformat(),
                "summary": tpsl_analysis.get("tpsl_summary", {}),
                "effectiveness": tpsl_analysis.get("tpsl_effectiveness", {}),
                "rankings": tpsl_analysis.get("ranked_models", []),
                "best_model": tpsl_analysis.get("best_tpsl_model", "N/A")
            }
            
            # Save TPSL report
            tpsl_report_file = self.config_dir / "reports" / "tpsl_performance_analysis.json"
            tpsl_report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tpsl_report_file, 'w') as f:
                json.dump(tpsl_report, f, indent=2, default=str)
            
            self.logger.info("✅ TPSL reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating TPSL reports: {e}")
    
    async def _save_test_results(self, results: Any, tpsl_analysis: Dict[str, Any], test_name: str) -> None:
        """Save comprehensive test results."""
        try:
            results_dir = self.config_dir / "results" / test_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = results_dir / f"{test_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results_data = {
                "test_name": test_name,
                "test_results": {
                    "model_results": getattr(results, 'model_results', {}),
                    "statistical_results": getattr(results, 'statistical_results', {}),
                    "performance_metrics": getattr(results, 'performance_metrics', {}),
                    "recommendations": getattr(results, 'recommendations', [])
                },
                "tpsl_analysis": tpsl_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Test results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving test results: {e}")
            raise


async def run_multi_model_tpsl_example():
    """Run a comprehensive multi-model TPSL testing example."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Multi-Model TPSL Testing Example")
    
    try:
        # Initialize the example
        example = MultiModelTPSLExample("config/multi_model_tpsl_example")
        
        # Define comprehensive test configuration with 6 models
        test_config = {
            "test_name": "Crypto_6_Model_TPSL_Test",
            "test_description": "Comprehensive testing of 6 crypto trading models with different TPSL strategies",
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "timeframe": "1h",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-06-30T23:59:59",
            "models": [
                {
                    "model_id": "model_a",
                    "model_name": "RandomForest_Conservative",
                    "model_type": "random_forest",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.08,
                    "risk_per_trade": 0.02,
                    "model_params": {
                        "n_estimators": 150,
                        "max_depth": 12,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_b",
                    "model_name": "LightGBM_Aggressive",
                    "model_type": "lightgbm",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.12,
                    "risk_per_trade": 0.03,
                    "model_params": {
                        "n_estimators": 300,
                        "max_depth": 10,
                        "learning_rate": 0.08,
                        "num_leaves": 60,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_c",
                    "model_name": "XGBoost_Balanced",
                    "model_type": "xgboost",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.10,
                    "risk_per_trade": 0.025,
                    "model_params": {
                        "n_estimators": 250,
                        "max_depth": 8,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_d",
                    "model_name": "TabNet_Advanced",
                    "model_type": "tabnet",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.09,
                    "risk_per_trade": 0.025,
                    "model_params": {
                        "n_d": 64,
                        "n_a": 64,
                        "n_steps": 5,
                        "gamma": 1.5,
                        "lambda_sparse": 1e-3,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_e",
                    "model_name": "CatBoost_Stable",
                    "model_type": "catboost",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.07,
                    "risk_per_trade": 0.02,
                    "model_params": {
                        "iterations": 200,
                        "depth": 8,
                        "learning_rate": 0.1,
                        "l2_leaf_reg": 3,
                        "random_state": 42
                    }
                },
                {
                    "model_id": "model_f",
                    "model_name": "ExtraTrees_Fast",
                    "model_type": "extra_trees",
                    "initial_capital": 100000.0,
                    "max_position_size": 0.11,
                    "risk_per_trade": 0.03,
                    "model_params": {
                        "n_estimators": 200,
                        "max_depth": 15,
                        "min_samples_split": 3,
                        "min_samples_leaf": 1,
                        "random_state": 42
                    }
                }
            ],
            "tpsl_configs": {
                "model_a": {
                    "strategy": "atr_based",
                    "atr_multiplier_tp": 2.0,
                    "atr_multiplier_sl": 1.0,
                    "max_risk_per_trade": 0.02
                },
                "model_b": {
                    "strategy": "dynamic",
                    "take_profit_pct": 0.02,
                    "stop_loss_pct": 0.01,
                    "dynamic_adjustment_factor": 0.6,
                    "max_risk_per_trade": 0.03
                },
                "model_c": {
                    "strategy": "confidence_based",
                    "take_profit_pct": 0.02,
                    "stop_loss_pct": 0.01,
                    "confidence_threshold_high": 0.8,
                    "confidence_threshold_medium": 0.6,
                    "confidence_threshold_low": 0.4,
                    "analyst_confidence_weight": 0.6,
                    "tactician_confidence_weight": 0.4,
                    "max_risk_per_trade": 0.025
                },
                "model_d": {
                    "strategy": "trailing",
                    "take_profit_pct": 0.025,
                    "stop_loss_pct": 0.012,
                    "trailing_start_pct": 0.015,
                    "trailing_step_pct": 0.006,
                    "max_risk_per_trade": 0.025
                },
                "model_e": {
                    "strategy": "scaling",
                    "take_profit_pct": 0.018,
                    "stop_loss_pct": 0.009,
                    "scale_out_levels": [0.5, 0.3, 0.2],
                    "scale_out_sizes": [0.3, 0.3, 0.4],
                    "max_risk_per_trade": 0.02
                },
                "model_f": {
                    "strategy": "momentum_based",
                    "take_profit_pct": 0.02,
                    "stop_loss_pct": 0.01,
                    "momentum_period": 10,
                    "momentum_threshold": 0.5,
                    "max_risk_per_trade": 0.03
                }
            },
            "statistical_testing": {
                "enable_statistical_testing": True,
                "confidence_level": 0.95,
                "alpha": 0.05,
                "min_sample_size": 200,
                "enable_multiple_testing_correction": True,
                "correction_method": "bonferroni",
                "effect_size_threshold": 0.3,
                "power_analysis": True,
                "power_threshold": 0.85
            },
            "risk_management": {
                "global_risk_limit": 0.25,
                "max_concurrent_positions": 12,
                "correlation_threshold": 0.75,
                "enable_circuit_breakers": True,
                "circuit_breaker_threshold": 0.12
            }
        }
        
        # Run the comprehensive test
        results = await example.run_multi_model_tpsl_test(test_config)
        
        # Print summary
        logger.info("📊 Multi-Model TPSL Test Results Summary:")
        logger.info(f"   Test Name: {test_config['test_name']}")
        logger.info(f"   Symbol: {test_config['symbol']}")
        logger.info(f"   Timeframe: {test_config['timeframe']}")
        logger.info(f"   Models Tested: {len(test_config['models'])}")
        logger.info(f"   TPSL Strategies: {len(set(config['strategy'] for config in test_config['tpsl_configs'].values()))}")
        logger.info(f"   Test Period: {test_config['start_date']} to {test_config['end_date']}")
        
        # Print TPSL analysis summary
        tpsl_analysis = results.get("tpsl_analysis", {})
        if tpsl_analysis:
            best_model = tpsl_analysis.get("best_tpsl_model", "N/A")
            logger.info(f"   Best TPSL Model: {best_model}")
            
            ranked_models = tpsl_analysis.get("ranked_models", [])
            if ranked_models:
                logger.info("   Model TPSL Rankings:")
                for i, (model_id, effectiveness) in enumerate(ranked_models[:3]):
                    score = effectiveness.get("effectiveness_score", 0.0)
                    logger.info(f"     {i+1}. {model_id}: {score:.3f}")
        
        logger.info("✅ Multi-Model TPSL Testing Example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in multi-model TPSL testing example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_multi_model_tpsl_example())