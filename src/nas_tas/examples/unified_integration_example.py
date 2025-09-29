"""
Unified NAS/TAS Integration Example

This example demonstrates how to use the unified NAS/TAS tools to replace
duplicate code and ensure consistent behavior across NAS and TAS implementations.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Import unified NAS/TAS tools
from ..config.base_config import (
    UnifiedArchitectureConfig, 
    create_comprehensive_config,
    create_regime_aware_config,
    ArchitectureType,
    OptimizationMode,
    SearchStrategy
)
from ..data.data_processor import UnifiedDataProcessor, DataProcessingConfig
from ..evaluation.unified_evaluator import UnifiedEvaluator, EvaluationConfig
from ..training.training_orchestrator import UnifiedTrainingOrchestrator, TrainingConfig
from ..unified_pipeline import (
    UnifiedNASPipeline, 
    UnifiedTASPipeline, 
    UnifiedHybridPipeline,
    create_nas_pipeline,
    create_tas_pipeline,
    create_hybrid_pipeline
)
from ..results.result_manager import ResultManager
from ..error_handling import UnifiedErrorHandler
from ..logging import UnifiedLogger, LoggingConfig
from ..migration.config_migration_helper import ConfigMigrationHelper


class UnifiedIntegrationExample:
    """Example demonstrating unified NAS/TAS integration."""
    
    def __init__(self):
        """Initialize the integration example."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        
        # Initialize unified components
        self.setup_unified_components()
        
        # Example data
        self.sample_data = None
        self.sample_target = None
    
    def setup_logging(self):
        """Setup unified logging."""
        logging_config = LoggingConfig(
            log_level="INFO",
            enable_file_logging=True,
            enable_console_logging=True,
            log_format="detailed"
        )
        self.unified_logger = UnifiedLogger(logging_config)
        self.logger.info("✅ Unified logging setup completed")
    
    def setup_unified_components(self):
        """Setup all unified components."""
        # Unified error handler
        self.error_handler = UnifiedErrorHandler()
        
        # Unified data processor
        self.data_config = DataProcessingConfig(
            handle_missing_values=True,
            missing_value_strategy="median",
            handle_outliers=True,
            outlier_method="iqr",
            enable_scaling=True,
            scaling_method="standard",
            enable_feature_engineering=True,
            create_time_features=True,
            validate_data=True,
            min_data_quality_score=0.8
        )
        self.data_processor = UnifiedDataProcessor(self.data_config)
        
        # Unified evaluator
        self.eval_config = EvaluationConfig(
            evaluation_type="comprehensive",
            calculate_performance_metrics=True,
            calculate_financial_metrics=True,
            calculate_regime_metrics=True,
            calculate_risk_metrics=True,
            financial_validation=True,
            enable_parallel_evaluation=True,
            max_workers=4
        )
        self.evaluator = UnifiedEvaluator(self.eval_config)
        
        # Unified training orchestrator
        self.training_config = TrainingConfig(
            max_training_time_minutes=60,
            max_models=10,
            enable_parallel_training=True,
            max_parallel_models=3,
            model_selection_strategy="best_performance"
        )
        self.training_orchestrator = UnifiedTrainingOrchestrator(
            self.training_config,
            create_comprehensive_config()
        )
        
        # Result manager
        self.result_manager = ResultManager("unified_example_results")
        
        self.logger.info("✅ All unified components initialized")
    
    def create_sample_data(self, n_samples: int = 1000, n_features: int = 20) -> tuple:
        """Create sample data for demonstration."""
        np.random.seed(42)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target (binary classification)
        y = np.random.randint(0, 2, n_samples)
        
        # Add some missing values and outliers for demonstration
        X[10:20, 5] = np.nan  # Missing values
        X[100:110, 10] = X[100:110, 10] * 10  # Outliers
        
        self.sample_data = pd.DataFrame(
            X, 
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        self.sample_target = pd.Series(y, name="target")
        
        self.logger.info(f"✅ Sample data created: {X.shape}, target shape: {y.shape}")
        return self.sample_data, self.sample_target
    
    def demonstrate_unified_data_processing(self):
        """Demonstrate unified data processing capabilities."""
        self.logger.info("🔧 Demonstrating unified data processing...")
        
        # Process data
        processed_X, processed_y, validation_result = self.data_processor.process_data(
            self.sample_data, self.sample_target, fit=True
        )
        
        # Display results
        self.logger.info(f"✅ Data processing completed:")
        self.logger.info(f"   Original shape: {self.sample_data.shape}")
        self.logger.info(f"   Processed shape: {processed_X.shape}")
        self.logger.info(f"   Validation passed: {validation_result.validation_passed}")
        self.logger.info(f"   Quality score: {validation_result.validation_score:.3f}")
        self.logger.info(f"   Missing values: {validation_result.quality_metrics.missing_value_percentage:.1f}%")
        self.logger.info(f"   Complete rows: {validation_result.quality_metrics.complete_rows_percentage:.1f}%")
        
        if validation_result.warnings:
            self.logger.warning(f"⚠️ Warnings: {validation_result.warnings}")
        
        if validation_result.recommendations:
            self.logger.info(f"💡 Recommendations: {validation_result.recommendations}")
        
        return processed_X, processed_y, validation_result
    
    async def demonstrate_unified_evaluation(self, model, X, y):
        """Demonstrate unified evaluation capabilities."""
        self.logger.info("📊 Demonstrating unified evaluation...")
        
        # Create a simple model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Evaluate model
        eval_result = await self.evaluator.evaluate_model(model, X, y)
        
        # Display results
        self.logger.info(f"✅ Model evaluation completed:")
        self.logger.info(f"   Evaluation successful: {eval_result.evaluation_successful}")
        self.logger.info(f"   Overall score: {eval_result.evaluation_score:.3f}")
        self.logger.info(f"   Accuracy: {eval_result.metrics.accuracy:.3f}")
        self.logger.info(f"   F1 Score: {eval_result.metrics.f1_score:.3f}")
        self.logger.info(f"   Financial validation: {eval_result.financial_validation.passed_validation if eval_result.financial_validation else 'N/A'}")
        
        if eval_result.recommendations:
            self.logger.info(f"💡 Recommendations: {eval_result.recommendations}")
        
        return eval_result
    
    def demonstrate_configuration_management(self):
        """Demonstrate unified configuration management."""
        self.logger.info("⚙️ Demonstrating configuration management...")
        
        # Create configurations using presets
        quick_config = create_comprehensive_config()
        regime_config = create_regime_aware_config()
        
        # Customize configurations
        quick_config.n_regimes = 6
        quick_config.population_size = 50
        quick_config.architecture_type = ArchitectureType.HYBRID_NEURAL_TREE
        
        regime_config.optimization_mode = OptimizationMode.REGIME_AWARE
        regime_config.search_strategy = SearchStrategy.HYBRID
        
        # Save configurations
        quick_config.save_to_file("quick_config.json")
        regime_config.save_to_file("regime_config.json")
        
        # Load configurations
        loaded_config = UnifiedArchitectureConfig.load_from_file("quick_config.json")
        
        self.logger.info(f"✅ Configuration management completed:")
        self.logger.info(f"   Quick config: {quick_config.architecture_type.value}")
        self.logger.info(f"   Regime config: {regime_config.optimization_mode.value}")
        self.logger.info(f"   Loaded config: {loaded_config.architecture_type.value}")
        
        return quick_config, regime_config, loaded_config
    
    def demonstrate_configuration_migration(self):
        """Demonstrate configuration migration from legacy formats."""
        self.logger.info("🔄 Demonstrating configuration migration...")
        
        # Example legacy NAS config
        legacy_nas_config = {
            'n_regimes': 8,
            'population_size': 100,
            'generations': 200,
            'max_search_iterations': 500,
            'optimization_mode': 'regime_aware',
            'search_strategy': 'evolutionary',
            'validation_method': 'time_series_split',
            'random_state': 42,
            'verbose': True,
            'custom_param': 'legacy_value'
        }
        
        # Example legacy TAS config
        legacy_tas_config = {
            'n_regimes': 6,
            'regime_stability_threshold': 0.8,
            'data_driven_regimes': True,
            'population_size': 80,
            'generations': 150,
            'optimization_mode': 'regime_aware',
            'search_strategy': 'bayesian',
            'custom_tas_param': 'tas_value'
        }
        
        # Migrate configurations
        migrated_nas = ConfigMigrationHelper.migrate_nas_config_to_unified(
            legacy_nas_config, preserve_custom=True
        )
        
        migrated_tas = ConfigMigrationHelper.migrate_tas_config_to_unified(
            legacy_tas_config, preserve_custom=True
        )
        
        self.logger.info(f"✅ Configuration migration completed:")
        self.logger.info(f"   NAS migrated: {migrated_nas.architecture_type.value}")
        self.logger.info(f"   TAS migrated: {migrated_tas.architecture_type.value}")
        self.logger.info(f"   NAS custom params: {migrated_nas.custom_parameters}")
        self.logger.info(f"   TAS custom params: {migrated_tas.custom_parameters}")
        
        return migrated_nas, migrated_tas
    
    async def demonstrate_unified_pipelines(self, X, y):
        """Demonstrate unified pipeline capabilities."""
        self.logger.info("🚀 Demonstrating unified pipelines...")
        
        # Create pipelines
        nas_pipeline = create_nas_pipeline()
        tas_pipeline = create_tas_pipeline()
        hybrid_pipeline = create_hybrid_pipeline()
        
        # Configure pipelines
        pipeline_config = create_comprehensive_config()
        pipeline_config.n_regimes = 4
        pipeline_config.population_size = 30
        pipeline_config.max_search_iterations = 50
        
        # Execute pipelines (simplified for demonstration)
        self.logger.info("📊 Executing NAS pipeline...")
        try:
            nas_result = await nas_pipeline.execute_pipeline(X, y)
            self.logger.info(f"   NAS pipeline completed: {nas_result.execution_info.status}")
        except Exception as e:
            self.logger.warning(f"   NAS pipeline failed: {e}")
        
        self.logger.info("📊 Executing TAS pipeline...")
        try:
            tas_result = await tas_pipeline.execute_pipeline(X, y)
            self.logger.info(f"   TAS pipeline completed: {tas_result.execution_info.status}")
        except Exception as e:
            self.logger.warning(f"   TAS pipeline failed: {e}")
        
        self.logger.info("📊 Executing Hybrid pipeline...")
        try:
            hybrid_result = await hybrid_pipeline.execute_pipeline(X, y)
            self.logger.info(f"   Hybrid pipeline completed: {hybrid_result.execution_info.status}")
        except Exception as e:
            self.logger.warning(f"   Hybrid pipeline failed: {e}")
        
        self.logger.info("✅ Unified pipeline demonstration completed")
    
    async def demonstrate_error_handling(self):
        """Demonstrate unified error handling."""
        self.logger.info("⚠️ Demonstrating error handling...")
        
        # Simulate different types of errors
        test_cases = [
            ("Data Processing Error", lambda: self.data_processor.process_data(None, None)),
            ("Evaluation Error", lambda: self.evaluator.evaluate_model(None, None, None)),
            ("Configuration Error", lambda: UnifiedArchitectureConfig(n_regimes=-1))
        ]
        
        for error_type, error_func in test_cases:
            try:
                self.logger.info(f"   Testing {error_type}...")
                await error_func() if asyncio.iscoroutinefunction(error_func) else error_func()
            except Exception as e:
                # Handle error using unified error handler
                error_info = self.error_handler.handle_error(e, {
                    "component": "example",
                    "error_type": error_type,
                    "timestamp": datetime.now()
                })
                self.logger.info(f"   ✅ Error handled: {error_info['error_type']}")
        
        self.logger.info("✅ Error handling demonstration completed")
    
    def demonstrate_result_management(self, results: List[Any]):
        """Demonstrate unified result management."""
        self.logger.info("📁 Demonstrating result management...")
        
        # Store results
        for i, result in enumerate(results):
            if hasattr(result, 'to_dict'):
                self.result_manager.store_result(result)
                self.logger.info(f"   ✅ Stored result {i+1}")
        
        # Get summary
        summary = self.result_manager.get_results_summary()
        self.logger.info(f"✅ Result management completed:")
        self.logger.info(f"   Total results stored: {summary.get('total_results', 0)}")
        self.logger.info(f"   Storage directory: {self.result_manager.output_directory}")
        
        return summary
    
    async def run_complete_demonstration(self):
        """Run complete demonstration of unified NAS/TAS tools."""
        self.logger.info("🎯 Starting Unified NAS/TAS Integration Demonstration")
        self.logger.info("=" * 60)
        
        try:
            # 1. Create sample data
            self.logger.info("\n1️⃣ Creating sample data...")
            X, y = self.create_sample_data()
            
            # 2. Demonstrate data processing
            self.logger.info("\n2️⃣ Data processing demonstration...")
            processed_X, processed_y, validation_result = self.demonstrate_unified_data_processing()
            
            # 3. Demonstrate evaluation
            self.logger.info("\n3️⃣ Evaluation demonstration...")
            eval_result = await self.demonstrate_unified_evaluation(None, processed_X, processed_y)
            
            # 4. Demonstrate configuration management
            self.logger.info("\n4️⃣ Configuration management demonstration...")
            configs = self.demonstrate_configuration_management()
            
            # 5. Demonstrate configuration migration
            self.logger.info("\n5️⃣ Configuration migration demonstration...")
            migrated_configs = self.demonstrate_configuration_migration()
            
            # 6. Demonstrate unified pipelines
            self.logger.info("\n6️⃣ Unified pipeline demonstration...")
            await self.demonstrate_unified_pipelines(processed_X, processed_y)
            
            # 7. Demonstrate error handling
            self.logger.info("\n7️⃣ Error handling demonstration...")
            await self.demonstrate_error_handling()
            
            # 8. Demonstrate result management
            self.logger.info("\n8️⃣ Result management demonstration...")
            all_results = [validation_result, eval_result] + list(configs) + list(migrated_configs)
            summary = self.demonstrate_result_management(all_results)
            
            # Final summary
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🎉 Unified NAS/TAS Integration Demonstration Completed!")
            self.logger.info(f"✅ All components successfully demonstrated")
            self.logger.info(f"📊 Results summary: {summary}")
            
        except Exception as e:
            self.logger.error(f"❌ Demonstration failed: {e}")
            # Handle error using unified error handler
            error_info = self.error_handler.handle_error(e, {
                "component": "demonstration",
                "timestamp": datetime.now()
            })
            self.logger.info(f"🔧 Error handled: {error_info['error_type']}")


async def main():
    """Main function to run the demonstration."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run demonstration
    example = UnifiedIntegrationExample()
    await example.run_complete_demonstration()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())