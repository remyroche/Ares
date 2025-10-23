"""
OOF/OOS Usage Examples

This module provides comprehensive examples of how to use the enhanced consolidated
OOF/OOS utilities for various machine learning scenarios.

Examples include:
1. Basic OOF prediction generation
2. Advanced stacking ensemble with confidence intervals
3. OOS validation with nested Sharpe ratio optimization
4. Multi-output model training and evaluation
5. Leakage detection and temporal validation
6. Integration with existing training pipelines
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

# Import enhanced consolidated utilities
from .enhanced_consolidated_oof_oos import (
    EnhancedConsolidatedOOFGenerator,
    EnhancedConsolidatedOOSValidator,
    create_enhanced_oof_generator,
    create_enhanced_oos_validator,
    OOFStrategy,
    OOSValidationType,
    EnsembleType
)

# Import configuration classes
from .enhanced_consolidated_oof_oos import (
    EnhancedOOFConfig,
    EnhancedOOSConfig
)

# Import sklearn models for examples
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


class OOFOOSExamples:
    """Comprehensive examples of OOF/OOS utilities usage."""
    
    def __init__(self):
        """Initialize examples with sample data."""
        self.logger = logging.getLogger(f"{__name__}.OOFOOSExamples")
        self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample data for examples."""
        np.random.seed(42)
        
        # Generate sample features
        n_samples = 1000
        n_features = 20
        
        self.X = np.random.randn(n_samples, n_features)
        
        # Generate sample targets (regression)
        self.y_regression = np.random.randn(n_samples)
        
        # Generate sample targets (classification)
        self.y_classification = np.random.randint(0, 2, n_samples)
        
        # Generate sample returns for Sharpe ratio calculation
        self.returns = np.random.randn(n_samples) * 0.02  # 2% daily volatility
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=n_samples)
        self.timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1H')
        
        # Generate sample models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'lightgbm': LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        }
        
        self.logger.info("✅ Sample data generated for examples")
    
    def example_1_basic_oof_generation(self):
        """Example 1: Basic OOF prediction generation."""
        self.logger.info("🔄 Example 1: Basic OOF prediction generation")
        
        # Create OOF generator with basic configuration
        oof_generator = create_enhanced_oof_generator(
            strategy=OOFStrategy.MEAN,
            n_folds=5,
            enable_confidence_intervals=True,
            enable_diversity_metrics=True
        )
        
        # Generate OOF predictions
        oof_result = oof_generator.generate_oof_predictions(
            models=self.models,
            X=self.X,
            y=self.y_regression,
            timestamps=self.timestamps
        )
        
        # Display results
        self.logger.info(f"✅ OOF predictions generated for {len(oof_result.model_names)} models")
        self.logger.info(f"📊 Generation time: {oof_result.generation_time:.2f}s")
        self.logger.info(f"📈 OOF scores: {oof_result.oof_scores}")
        
        if oof_result.ensemble_diversity:
            self.logger.info(f"🎯 Ensemble diversity: {oof_result.ensemble_diversity}")
        
        if oof_result.confidence_intervals:
            self.logger.info(f"📊 Confidence intervals calculated for {len(oof_result.confidence_intervals)} models")
        
        return oof_result
    
    def example_2_advanced_stacking_ensemble(self):
        """Example 2: Advanced stacking ensemble with confidence intervals."""
        self.logger.info("🔄 Example 2: Advanced stacking ensemble")
        
        # Create OOF generator with stacking configuration
        oof_generator = create_enhanced_oof_generator(
            strategy=OOFStrategy.STACKING,
            n_folds=5,
            ensemble_type=EnsembleType.STACKING,
            enable_meta_learning=True,
            meta_model_type="ridge",
            enable_confidence_intervals=True,
            enable_diversity_metrics=True,
            enable_leakage_detection=True,
            enable_temporal_validation=True
        )
        
        # Generate OOF predictions with stacking
        oof_result = oof_generator.generate_oof_predictions(
            models=self.models,
            X=self.X,
            y=self.y_regression,
            timestamps=self.timestamps
        )
        
        # Display results
        self.logger.info(f"✅ Stacking ensemble OOF predictions generated")
        self.logger.info(f"📊 Meta-model performance: {oof_result.meta_model_performance}")
        self.logger.info(f"⚖️ Base model weights: {oof_result.base_model_weights}")
        
        if oof_result.leakage_detection:
            self.logger.info(f"🔍 Leakage detection: {oof_result.leakage_detection}")
        
        if oof_result.temporal_analysis:
            self.logger.info(f"⏰ Temporal analysis: {oof_result.temporal_analysis}")
        
        return oof_result
    
    def example_3_oos_validation_nested_sharpe(self):
        """Example 3: OOS validation with nested Sharpe ratio optimization."""
        self.logger.info("🔄 Example 3: OOS validation with nested Sharpe ratio")
        
        # First generate OOF predictions
        oof_generator = create_enhanced_oof_generator(
            strategy=OOFStrategy.STACKING,
            n_folds=5
        )
        oof_result = oof_generator.generate_oof_predictions(
            models=self.models,
            X=self.X,
            y=self.y_regression
        )
        
        # Create OOS validator with nested Sharpe ratio
        oos_validator = create_enhanced_oos_validator(
            validation_type=OOSValidationType.NESTED_SHARPE,
            enable_nested_sharpe=True,
            sharpe_optimization=True,
            sharpe_threshold=0.5,
            risk_free_rate=0.0,
            min_test_signals=100,
            enable_leakage_detection=True,
            enable_temporal_validation=True
        )
        
        # Perform OOS validation
        oos_result = oos_validator.validate_oos(
            predictions=oof_result.oof_predictions['ensemble'],
            targets=self.y_regression,
            returns=self.returns,
            timestamps=self.timestamps
        )
        
        # Display results
        self.logger.info(f"✅ OOS validation completed")
        self.logger.info(f"📊 Validation time: {oos_result.validation_time:.2f}s")
        self.logger.info(f"📈 Validation scores: {oos_result.validation_scores}")
        
        if oos_result.nested_sharpe_scores:
            self.logger.info(f"🎯 Nested Sharpe scores: {oos_result.nested_sharpe_scores}")
        
        if oos_result.leakage_detection:
            self.logger.info(f"🔍 Leakage detection: {oos_result.leakage_detection}")
        
        return oos_result
    
    def example_4_multi_output_training(self):
        """Example 4: Multi-output model training and evaluation."""
        self.logger.info("🔄 Example 4: Multi-output model training")
        
        # Generate multi-output targets
        n_outputs = 4
        y_multi = np.random.randn(len(self.X), n_outputs)
        output_names = [f"output_{i+1}" for i in range(n_outputs)]
        
        # Create OOF generator for multi-output
        oof_generator = create_enhanced_oof_generator(
            strategy=OOFStrategy.STACKING,
            n_folds=5,
            n_outputs=n_outputs,
            output_names=output_names,
            enable_confidence_intervals=True,
            enable_diversity_metrics=True
        )
        
        # Generate OOF predictions for multi-output
        oof_result = oof_generator.generate_oof_predictions(
            models=self.models,
            X=self.X,
            y=y_multi
        )
        
        # Display results
        self.logger.info(f"✅ Multi-output OOF predictions generated")
        self.logger.info(f"📊 Number of outputs: {oof_result.config.n_outputs}")
        self.logger.info(f"📈 Output names: {oof_result.config.output_names}")
        
        # Calculate per-output performance
        for i, output_name in enumerate(output_names):
            if output_name in oof_result.oof_scores:
                self.logger.info(f"📊 {output_name} OOF score: {oof_result.oof_scores[output_name]:.4f}")
        
        return oof_result
    
    def example_5_leakage_detection_integration(self):
        """Example 5: Leakage detection integration."""
        self.logger.info("🔄 Example 5: Leakage detection integration")
        
        # Create OOF generator with leakage detection
        oof_generator = create_enhanced_oof_generator(
            strategy=OOFStrategy.STACKING,
            n_folds=5,
            enable_leakage_detection=True,
            enable_temporal_validation=True
        )
        
        # Generate OOF predictions with leakage detection
        oof_result = oof_generator.generate_oof_predictions(
            models=self.models,
            X=self.X,
            y=self.y_regression,
            timestamps=self.timestamps
        )
        
        # Display leakage detection results
        if oof_result.leakage_detection:
            self.logger.info(f"🔍 Leakage detection results:")
            self.logger.info(f"   - Leakage detected: {oof_result.leakage_detection.get('leakage_detected', False)}")
            self.logger.info(f"   - Leakage count: {oof_result.leakage_detection.get('leakage_count', 0)}")
            if 'leakage_types' in oof_result.leakage_detection:
                self.logger.info(f"   - Leakage types: {oof_result.leakage_detection['leakage_types']}")
        
        # Display temporal analysis results
        if oof_result.temporal_analysis:
            self.logger.info(f"⏰ Temporal analysis results:")
            for model_name, analysis in oof_result.temporal_analysis.items():
                self.logger.info(f"   - {model_name}:")
                self.logger.info(f"     * Temporal correlation: {analysis.get('temporal_correlation', 0):.4f}")
                self.logger.info(f"     * Temporal stability: {analysis.get('temporal_stability', 0):.4f}")
                self.logger.info(f"     * Trend strength: {analysis.get('trend_strength', 0):.4f}")
        
        return oof_result
    
    def example_6_performance_comparison(self):
        """Example 6: Performance comparison between different strategies."""
        self.logger.info("🔄 Example 6: Performance comparison")
        
        strategies = [
            OOFStrategy.MEAN,
            OOFStrategy.MEDIAN,
            OOFStrategy.WEIGHTED_MEAN,
            OOFStrategy.STACKING
        ]
        
        results = {}
        
        for strategy in strategies:
            self.logger.info(f"🔄 Testing strategy: {strategy.value}")
            
            # Create OOF generator for this strategy
            oof_generator = create_enhanced_oof_generator(
                strategy=strategy,
                n_folds=5,
                enable_confidence_intervals=True,
                enable_diversity_metrics=True
            )
            
            # Generate OOF predictions
            oof_result = oof_generator.generate_oof_predictions(
                models=self.models,
                X=self.X,
                y=self.y_regression
            )
            
            # Store results
            results[strategy.value] = {
                'oof_scores': oof_result.oof_scores,
                'generation_time': oof_result.generation_time,
                'ensemble_diversity': oof_result.ensemble_diversity,
                'meta_model_performance': oof_result.meta_model_performance
            }
        
        # Display comparison
        self.logger.info("📊 Performance comparison results:")
        for strategy_name, result in results.items():
            self.logger.info(f"   - {strategy_name}:")
            self.logger.info(f"     * Generation time: {result['generation_time']:.2f}s")
            if result['ensemble_diversity']:
                self.logger.info(f"     * Diversity score: {result['ensemble_diversity'].get('diversity_score', 0):.4f}")
            if result['meta_model_performance']:
                self.logger.info(f"     * Meta-model performance: {result['meta_model_performance']}")
        
        return results
    
    def example_7_integration_with_training_pipeline(self):
        """Example 7: Integration with existing training pipeline."""
        self.logger.info("🔄 Example 7: Integration with training pipeline")
        
        # Simulate a training pipeline
        def train_model_pipeline(X, y, model_name, model):
            """Simulate model training pipeline."""
            # Train model
            model.fit(X, y)
            
            # Generate OOF predictions
            oof_generator = create_enhanced_oof_generator(
                strategy=OOFStrategy.STACKING,
                n_folds=5,
                enable_confidence_intervals=True,
                enable_diversity_metrics=True,
                enable_leakage_detection=True
            )
            
            oof_result = oof_generator.generate_oof_predictions(
                models={model_name: model},
                X=X,
                y=y,
                timestamps=self.timestamps
            )
            
            # Perform OOS validation
            oos_validator = create_enhanced_oos_validator(
                validation_type=OOSValidationType.NESTED_SHARPE,
                enable_nested_sharpe=True,
                sharpe_optimization=True
            )
            
            oos_result = oos_validator.validate_oos(
                predictions=oof_result.oof_predictions[model_name],
                targets=y,
                returns=self.returns
            )
            
            return {
                'model': model,
                'oof_result': oof_result,
                'oos_result': oos_result
            }
        
        # Train multiple models
        pipeline_results = {}
        for model_name, model in self.models.items():
            self.logger.info(f"🔄 Training {model_name}")
            result = train_model_pipeline(self.X, self.y_regression, model_name, model)
            pipeline_results[model_name] = result
        
        # Display pipeline results
        self.logger.info("📊 Training pipeline results:")
        for model_name, result in pipeline_results.items():
            oof_score = result['oof_result'].oof_scores[model_name]
            oos_score = result['oos_result'].validation_scores.get('sharpe_ratio', 0)
            self.logger.info(f"   - {model_name}:")
            self.logger.info(f"     * OOF score: {oof_score:.4f}")
            self.logger.info(f"     * OOS Sharpe ratio: {oos_score:.4f}")
        
        return pipeline_results
    
    def run_all_examples(self):
        """Run all examples."""
        self.logger.info("🚀 Running all OOF/OOS examples")
        
        examples = [
            self.example_1_basic_oof_generation,
            self.example_2_advanced_stacking_ensemble,
            self.example_3_oos_validation_nested_sharpe,
            self.example_4_multi_output_training,
            self.example_5_leakage_detection_integration,
            self.example_6_performance_comparison,
            self.example_7_integration_with_training_pipeline
        ]
        
        results = {}
        
        for i, example_func in enumerate(examples, 1):
            try:
                self.logger.info(f"🔄 Running Example {i}: {example_func.__name__}")
                result = example_func()
                results[f"example_{i}"] = result
                self.logger.info(f"✅ Example {i} completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Example {i} failed: {e}")
                results[f"example_{i}"] = {"error": str(e)}
        
        self.logger.info("🎉 All examples completed")
        return results


def main():
    """Main function to run examples."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create examples instance
    examples = OOFOOSExamples()
    
    # Run all examples
    results = examples.run_all_examples()
    
    # Display summary
    print("\n" + "="*50)
    print("OOF/OOS EXAMPLES SUMMARY")
    print("="*50)
    
    for example_name, result in results.items():
        if "error" in result:
            print(f"❌ {example_name}: FAILED - {result['error']}")
        else:
            print(f"✅ {example_name}: SUCCESS")
    
    print("="*50)


if __name__ == "__main__":
    main()