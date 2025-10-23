"""
Comprehensive Example: Using Abstract Base Classes

This example demonstrates how to use all the abstract base classes
in a complete machine learning pipeline for production use.

Pipeline Steps:
1. Data Validation - Validate input data quality
2. Data Preprocessing - Clean and prepare data
3. Pattern Discovery - Discover patterns in the data
4. Labeling Strategy - Generate labels for training
5. Clustering Analysis - Cluster similar data points
6. Multi-Output Training - Train models for multiple outputs
7. Performance Evaluation - Evaluate model performance
8. Model Persistence - Save and load models

This example shows how the abstract base classes provide a consistent
interface while allowing for flexible implementations.
"""

import numpy as np
import pandas as pd
import asyncio
import time
from typing import Dict, Any, List
import logging
from pathlib import Path

# Import base classes
from src.core.abstract_base_classes import (
    ValidationLevel, TrainingStatus, ClusteringAlgorithm,
    PatternType, LabelingStrategy
)

# Import concrete implementations
from src.core.concrete_implementations import (
    DataValidator, MLTrainingStep, KMeansClustering,
    MultiOutputRandomForest, MomentumPatternDiscoverer, ProfitBasedLabeling
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMLPipeline:
    """
    Complete production ML pipeline using abstract base classes.
    
    This class demonstrates how to use all the abstract base classes
    together in a real-world machine learning pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the production ML pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config
        self.logger = logger.getChild('ProductionMLPipeline')
        
        # Initialize components
        self.validator = None
        self.training_step = None
        self.clustering_algorithm = None
        self.multi_output_model = None
        self.pattern_discoverer = None
        self.labeling_strategy = None
        
        # Pipeline state
        self.is_initialized = False
        self.training_data = None
        self.test_data = None
        self.results = {}
        
        self.logger.info("Production ML Pipeline initialized")

    def initialize_components(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")
        
        # Initialize validator
        self.validator = DataValidator(
            name="pipeline_validator",
            validation_level=ValidationLevel.PRODUCTION,
            config={
                'required_columns': ['price', 'volume', 'returns'],
                'min_samples': 100,
                'max_missing_ratio': 0.05,
                'value_ranges': {
                    'price': (0, 10000),
                    'volume': (0, 1000000),
                    'returns': (-1, 1)
                }
            }
        )
        
        # Initialize training step
        self.training_step = MLTrainingStep(
            name="pipeline_training",
            model_type="random_forest",
            config={
                'n_estimators': 200,
                'max_depth': 10,
                'scale_features': True
            }
        )
        
        # Initialize clustering algorithm
        self.clustering_algorithm = KMeansClustering(
            name="pipeline_clustering",
            n_clusters=5,
            config={
                'random_state': 42,
                'n_init': 10
            }
        )
        
        # Initialize multi-output model
        self.multi_output_model = MultiOutputRandomForest(
            name="pipeline_multi_output",
            n_outputs=3,
            output_names=['signal_strength', 'confidence', 'risk_score'],
            config={
                'n_estimators': 150,
                'max_depth': 8
            }
        )
        
        # Initialize pattern discoverer
        self.pattern_discoverer = MomentumPatternDiscoverer(
            name="pipeline_pattern_discoverer",
            config={
                'lookback_period': 20,
                'momentum_threshold': 0.03,
                'confidence_threshold': 0.7
            }
        )
        
        # Initialize labeling strategy
        self.labeling_strategy = ProfitBasedLabeling(
            name="pipeline_labeling",
            config={
                'profit_threshold': 0.02,
                'lookforward_period': 5,
                'min_confidence': 0.6
            }
        )
        
        self.is_initialized = True
        self.logger.info("All pipeline components initialized successfully")

    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample financial data for the pipeline.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample financial data
        """
        self.logger.info(f"Generating {n_samples} samples of financial data...")
        
        np.random.seed(42)
        
        # Generate price data (random walk with trend)
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume data (correlated with price volatility)
        volume = np.random.lognormal(8, 0.5, n_samples)
        volume = volume * (1 + np.abs(returns) * 10)
        
        # Generate returns
        returns_series = np.diff(prices) / prices[:-1]
        returns_series = np.concatenate([[0], returns_series])
        
        # Create DataFrame
        data = pd.DataFrame({
            'price': prices,
            'volume': volume,
            'returns': returns_series,
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        })
        
        self.logger.info(f"Generated data with shape: {data.shape}")
        return data

    async def run_validation_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run data validation pipeline.
        
        Args:
            data: Input data to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Running data validation pipeline...")
        
        start_time = time.time()
        
        # Validate data
        validation_result = await self.validator.validate(data)
        
        # Get validation summary
        validation_summary = self.validator.get_validation_summary()
        
        results = {
            'validation_result': validation_result,
            'validation_summary': validation_summary,
            'execution_time': time.time() - start_time
        }
        
        self.logger.info(f"Validation completed in {results['execution_time']:.2f}s")
        return results

    async def run_pattern_discovery_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run pattern discovery pipeline.
        
        Args:
            data: Input data for pattern discovery
            
        Returns:
            Dictionary with pattern discovery results
        """
        self.logger.info("Running pattern discovery pipeline...")
        
        start_time = time.time()
        
        # Extract price data
        prices = data['price'].values
        
        # Discover patterns
        pattern_result = self.pattern_discoverer.discover_pattern(prices)
        
        # Get pattern definition
        pattern_definition = self.pattern_discoverer.get_pattern_definition()
        
        # Get pattern summary
        pattern_summary = self.pattern_discoverer.get_pattern_summary()
        
        results = {
            'pattern_result': pattern_result,
            'pattern_definition': pattern_definition,
            'pattern_summary': pattern_summary,
            'execution_time': time.time() - start_time
        }
        
        self.logger.info(f"Pattern discovery completed in {results['execution_time']:.2f}s")
        return results

    async def run_labeling_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run labeling pipeline.
        
        Args:
            data: Input data for labeling
            
        Returns:
            Dictionary with labeling results
        """
        self.logger.info("Running labeling pipeline...")
        
        start_time = time.time()
        
        # Extract price data
        prices = data['price'].values
        
        # Generate labels
        labeling_result = self.labeling_strategy.generate_labels(prices)
        
        # Get labeling summary
        labeling_summary = self.labeling_strategy.get_labeling_summary()
        
        results = {
            'labeling_result': labeling_result,
            'labeling_summary': labeling_summary,
            'execution_time': time.time() - start_time
        }
        
        self.logger.info(f"Labeling completed in {results['execution_time']:.2f}s")
        return results

    async def run_clustering_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run clustering pipeline.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Dictionary with clustering results
        """
        self.logger.info("Running clustering pipeline...")
        
        start_time = time.time()
        
        # Prepare features for clustering
        features = data[['price', 'volume', 'returns']].values
        
        # Perform clustering
        clustering_result = self.clustering_algorithm.fit_predict(features)
        
        # Get clustering summary
        clustering_summary = self.clustering_algorithm.get_clustering_summary()
        
        results = {
            'clustering_result': clustering_result,
            'clustering_summary': clustering_summary,
            'execution_time': time.time() - start_time
        }
        
        self.logger.info(f"Clustering completed in {results['execution_time']:.2f}s")
        return results

    async def run_training_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run training pipeline.
        
        Args:
            data: Input data for training
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Running training pipeline...")
        
        start_time = time.time()
        
        # Prepare training data
        X = data[['price', 'volume', 'returns']].values
        y = data['returns'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        training_result = await self.training_step.execute_training(
            (X_train, y_train),
            test_data=(X_test, y_test)
        )
        
        # Get training summary
        training_summary = self.training_step.get_training_summary()
        
        results = {
            'training_result': training_result,
            'training_summary': training_summary,
            'execution_time': time.time() - start_time
        }
        
        self.logger.info(f"Training completed in {results['execution_time']:.2f}s")
        return results

    async def run_multi_output_training_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run multi-output training pipeline.
        
        Args:
            data: Input data for multi-output training
            
        Returns:
            Dictionary with multi-output training results
        """
        self.logger.info("Running multi-output training pipeline...")
        
        start_time = time.time()
        
        # Prepare multi-output data
        X = data[['price', 'volume', 'returns']].values
        
        # Create multi-output targets
        signal_strength = np.random.rand(len(data))  # Mock signal strength
        confidence = np.random.rand(len(data))       # Mock confidence
        risk_score = np.random.rand(len(data))       # Mock risk score
        
        y = np.column_stack([signal_strength, confidence, risk_score])
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train multi-output model
        self.multi_output_model.fit(X_train, y_train)
        
        # Evaluate performance
        performance_results = self.multi_output_model.evaluate_performance(X_test, y_test)
        
        # Get model summary
        model_summary = self.multi_output_model.get_model_summary()
        
        results = {
            'performance_results': performance_results,
            'model_summary': model_summary,
            'execution_time': time.time() - start_time
        }
        
        self.logger.info(f"Multi-output training completed in {results['execution_time']:.2f}s")
        return results

    async def run_complete_pipeline(self, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with all pipeline results
        """
        self.logger.info("Starting complete ML pipeline...")
        
        pipeline_start_time = time.time()
        
        # Initialize components if not already done
        if not self.is_initialized:
            self.initialize_components()
        
        # Generate sample data
        data = self.generate_sample_data(n_samples)
        
        # Run all pipeline stages
        pipeline_results = {}
        
        # 1. Data Validation
        pipeline_results['validation'] = await self.run_validation_pipeline(data)
        
        # 2. Pattern Discovery
        pipeline_results['pattern_discovery'] = await self.run_pattern_discovery_pipeline(data)
        
        # 3. Labeling
        pipeline_results['labeling'] = await self.run_labeling_pipeline(data)
        
        # 4. Clustering
        pipeline_results['clustering'] = await self.run_clustering_pipeline(data)
        
        # 5. Single-output Training
        pipeline_results['training'] = await self.run_training_pipeline(data)
        
        # 6. Multi-output Training
        pipeline_results['multi_output_training'] = await self.run_multi_output_training_pipeline(data)
        
        # Calculate total execution time
        total_execution_time = time.time() - pipeline_start_time
        pipeline_results['total_execution_time'] = total_execution_time
        
        self.logger.info(f"Complete pipeline finished in {total_execution_time:.2f}s")
        
        return pipeline_results

    def save_models(self, output_dir: str = "models") -> Dict[str, bool]:
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary with save results
        """
        self.logger.info(f"Saving models to {output_dir}...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        save_results = {}
        
        # Save training step model
        if self.training_step and self.training_step.current_model is not None:
            success = self.training_step.save_model(f"{output_dir}/training_model.pkl")
            save_results['training_model'] = success
        
        # Save multi-output model
        if self.multi_output_model and self.multi_output_model.is_fitted:
            success = self.multi_output_model.save_model(f"{output_dir}/multi_output_model.pkl")
            save_results['multi_output_model'] = success
        
        self.logger.info(f"Model saving completed: {save_results}")
        return save_results

    def load_models(self, model_dir: str = "models") -> Dict[str, bool]:
        """
        Load all saved models.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Dictionary with load results
        """
        self.logger.info(f"Loading models from {model_dir}...")
        
        load_results = {}
        
        # Load training step model
        if self.training_step:
            success = self.training_step.load_model(f"{model_dir}/training_model.pkl")
            load_results['training_model'] = success
        
        # Load multi-output model
        if self.multi_output_model:
            success = self.multi_output_model.load_model(f"{model_dir}/multi_output_model.pkl")
            load_results['multi_output_model'] = success
        
        self.logger.info(f"Model loading completed: {load_results}")
        return load_results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report of pipeline results.
        
        Args:
            results: Pipeline results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("PRODUCTION ML PIPELINE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall execution time
        total_time = results.get('total_execution_time', 0)
        report.append(f"Total Execution Time: {total_time:.2f} seconds")
        report.append("")
        
        # Validation results
        if 'validation' in results:
            val_results = results['validation']
            val_summary = val_results['validation_summary']
            report.append("DATA VALIDATION:")
            report.append(f"  Success Rate: {val_summary['success_rate']:.2%}")
            report.append(f"  Total Validations: {val_summary['total_validations']}")
            report.append(f"  Execution Time: {val_results['execution_time']:.2f}s")
            report.append("")
        
        # Pattern discovery results
        if 'pattern_discovery' in results:
            pattern_results = results['pattern_discovery']
            pattern_summary = pattern_results['pattern_summary']
            report.append("PATTERN DISCOVERY:")
            report.append(f"  Discovered Patterns: {pattern_summary['discovered_patterns']}")
            report.append(f"  Average Confidence: {pattern_summary['avg_confidence']:.3f}")
            report.append(f"  Execution Time: {pattern_results['execution_time']:.2f}s")
            report.append("")
        
        # Labeling results
        if 'labeling' in results:
            labeling_results = results['labeling']
            labeling_summary = labeling_results['labeling_summary']
            report.append("LABELING:")
            report.append(f"  Total Samples Labeled: {labeling_summary['total_samples_labeled']}")
            report.append(f"  Average Confidence: {labeling_summary['avg_confidence']:.3f}")
            report.append(f"  Execution Time: {labeling_results['execution_time']:.2f}s")
            report.append("")
        
        # Clustering results
        if 'clustering' in results:
            clustering_results = results['clustering']
            clustering_summary = clustering_results['clustering_summary']
            report.append("CLUSTERING:")
            report.append(f"  Number of Clusters: {clustering_summary['latest_n_clusters']}")
            report.append(f"  Samples Processed: {clustering_summary['total_samples_processed']}")
            report.append(f"  Execution Time: {clustering_results['execution_time']:.2f}s")
            report.append("")
        
        # Training results
        if 'training' in results:
            training_results = results['training']
            training_summary = training_results['training_summary']
            report.append("SINGLE-OUTPUT TRAINING:")
            report.append(f"  Successful Runs: {training_summary['successful_runs']}")
            report.append(f"  Total Training Time: {training_summary['total_training_time']:.2f}s")
            report.append(f"  Execution Time: {training_results['execution_time']:.2f}s")
            report.append("")
        
        # Multi-output training results
        if 'multi_output_training' in results:
            multi_results = results['multi_output_training']
            model_summary = multi_results['model_summary']
            report.append("MULTI-OUTPUT TRAINING:")
            report.append(f"  Number of Outputs: {model_summary['n_outputs']}")
            report.append(f"  Model Fitted: {model_summary['is_fitted']}")
            report.append(f"  Execution Time: {multi_results['execution_time']:.2f}s")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main function to run the complete example."""
    print("Starting Production ML Pipeline Example...")
    print("=" * 80)
    
    # Configuration
    config = {
        'n_samples': 1000,
        'output_dir': 'models',
        'enable_logging': True
    }
    
    # Create pipeline
    pipeline = ProductionMLPipeline(config)
    
    try:
        # Run complete pipeline
        results = await pipeline.run_complete_pipeline(n_samples=config['n_samples'])
        
        # Save models
        save_results = pipeline.save_models(config['output_dir'])
        print(f"Models saved: {save_results}")
        
        # Generate and print report
        report = pipeline.generate_report(results)
        print("\n" + report)
        
        # Demonstrate model loading
        print("\nDemonstrating model loading...")
        load_results = pipeline.load_models(config['output_dir'])
        print(f"Models loaded: {load_results}")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())