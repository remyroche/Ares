"""
Unified Artifact Integration Examples

This module provides comprehensive examples showing how to use the unified
artifact management system that integrates KlinesParquetManager, serialization_utils,
and artifact_manager with BaseStep workflows.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

# Import the unified system
from src.utils.unified_artifact_system import (
    UnifiedArtifactSystem, UnifiedConfig, EnhancedBaseStep
)
from src.training.enhanced_base_step import EnhancedBaseStep, StepArtifactManager
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_info, tprint_warning, tprint_error


# Example 1: Basic Unified System Usage
def example_basic_unified_usage():
    """Example showing basic usage of the unified artifact system."""
    tprint_info("📚 EXAMPLE 1: Basic Unified System Usage")
    
    # Create unified system
    config = UnifiedConfig(
        base_dir="examples/unified_artifacts",
        enable_klines_optimization=True,
        enable_compression=True,
        enable_caching=True
    )
    
    system = UnifiedArtifactSystem(config)
    
    # Set context
    system.set_context(
        step_name="data_processing",
        symbol="ETHUSDT",
        exchange="binance",
        interval="1m",
        direction="long",
        model="Analyst"
    )
    
    # Create sample klines data
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=1440, freq='1min')
    klines_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(3000, 3100, 1440),
        'high': np.random.uniform(3100, 3200, 1440),
        'low': np.random.uniform(2900, 3000, 1440),
        'close': np.random.uniform(3000, 3100, 1440),
        'volume': np.random.uniform(100, 1000, 1440)
    })
    
    # Store klines data (automatically uses specialized klines manager)
    klines_id = system.store_klines(
        df=klines_data,
        symbol="ETHUSDT",
        exchange="binance",
        interval="1m"
    )
    tprint_success(f"✅ Stored klines data with ID: {klines_id}")
    
    # Store generic data (uses artifact manager)
    generic_data = {"features": ["rsi", "macd", "bollinger"], "values": [0.5, 0.3, 0.8]}
    generic_id = system.store_artifact(
        data=generic_data,
        artifact_name="feature_data",
        artifact_type="metadata"
    )
    tprint_success(f"✅ Stored generic data with ID: {generic_id}")
    
    # Load data back
    loaded_klines = system.load_klines("ETHUSDT", "binance", "1m")
    loaded_generic = system.load_artifact("feature_data", "metadata")
    
    tprint_info(f"📊 Loaded klines: {len(loaded_klines)} records")
    tprint_info(f"📊 Loaded generic: {loaded_generic}")
    
    # Get performance metrics
    metrics = system.get_performance_metrics()
    tprint_info(f"📈 Performance metrics: {metrics}")
    
    # Cleanup
    system.cleanup()


# Example 2: Enhanced BaseStep Usage
class DataProcessingStep(EnhancedBaseStep):
    """Example step that processes klines data."""
    
    async def _execute_step(self, data: Any) -> Any:
        """Process the input data."""
        tprint_info(f"🔄 Processing data in {self.step_name}")
        
        # Load klines data if not provided
        if data is None:
            klines_df = self.artifacts.load_klines()
        else:
            klines_df = data
        
        # Store input data
        input_id = self.artifacts.store_input(klines_df, "raw_klines")
        tprint_info(f"📥 Stored input data: {input_id}")
        
        # Process data (example: calculate moving averages)
        processed_df = klines_df.copy()
        processed_df['sma_20'] = processed_df['close'].rolling(window=20).mean()
        processed_df['sma_50'] = processed_df['close'].rolling(window=50).mean()
        processed_df['rsi'] = self._calculate_rsi(processed_df['close'])
        
        # Store intermediate results
        intermediate_id = self.artifacts.store_intermediate(processed_df, "processed_features")
        tprint_info(f"🔄 Stored intermediate data: {intermediate_id}")
        
        # Create final output
        output_data = {
            'processed_df': processed_df,
            'summary_stats': {
                'total_records': len(processed_df),
                'sma_20_mean': processed_df['sma_20'].mean(),
                'sma_50_mean': processed_df['sma_50'].mean(),
                'rsi_mean': processed_df['rsi'].mean()
            }
        }
        
        # Store output
        output_id = self.artifacts.store_output(output_data, "final_results")
        tprint_info(f"📤 Stored output data: {output_id}")
        
        return output_data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def example_enhanced_basestep_usage():
    """Example showing enhanced BaseStep usage."""
    tprint_info("📚 EXAMPLE 2: Enhanced BaseStep Usage")
    
    # Create unified system
    system = UnifiedArtifactSystem()
    
    # Create step configuration
    config = {
        'step_name': 'data_processing',
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'interval': '1m',
        'direction': 'long',
        'model': 'Analyst'
    }
    
    # Create step instance
    step = DataProcessingStep(config, system)
    
    # Validate configuration
    step.validate_config()
    
    # Create sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=1440, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(3000, 3100, 1440),
        'high': np.random.uniform(3100, 3200, 1440),
        'low': np.random.uniform(2900, 3000, 1440),
        'close': np.random.uniform(3000, 3100, 1440),
        'volume': np.random.uniform(100, 1000, 1440)
    })
    
    # Execute step
    async def run_step():
        result = await step.execute(sample_data)
        return result
    
    # Run the step
    result = asyncio.run(run_step())
    
    # Get step status
    status = step.get_status()
    tprint_info(f"📊 Step status: {status}")
    
    # List artifacts created by the step
    artifacts = step.get_step_artifacts()
    tprint_info(f"📁 Step artifacts: {len(artifacts)} created")
    
    # Get execution summary
    summary = step.get_execution_summary()
    tprint_info(f"📈 Execution summary: {summary}")
    
    # Cleanup step artifacts
    step.cleanup_step()


# Example 3: Multi-Step Workflow
class DataCollectionStep(EnhancedBaseStep):
    """Step that collects and stores klines data."""
    
    async def _execute_step(self, data: Any) -> Any:
        """Collect klines data."""
        tprint_info(f"📥 Collecting klines data in {self.step_name}")
        
        # Simulate data collection
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=10080, freq='1min')
        klines_df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(3000, 3100, 10080),
            'high': np.random.uniform(3100, 3200, 10080),
            'low': np.random.uniform(2900, 3000, 10080),
            'close': np.random.uniform(3000, 3100, 10080),
            'volume': np.random.uniform(100, 1000, 10080)
        })
        
        # Store klines data
        klines_id = self.artifacts.store_klines(klines_df)
        tprint_success(f"✅ Collected {len(klines_df)} klines records")
        
        return klines_df


class FeatureEngineeringStep(EnhancedBaseStep):
    """Step that engineers features from klines data."""
    
    async def _execute_step(self, data: Any) -> Any:
        """Engineer features from klines data."""
        tprint_info(f"🔧 Engineering features in {self.step_name}")
        
        # Load klines data
        klines_df = self.artifacts.load_klines()
        
        # Engineer features
        features_df = klines_df.copy()
        features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
        features_df['sma_50'] = features_df['close'].rolling(window=50).mean()
        features_df['rsi'] = self._calculate_rsi(features_df['close'])
        features_df['bollinger_upper'] = features_df['close'].rolling(window=20).mean() + 2 * features_df['close'].rolling(window=20).std()
        features_df['bollinger_lower'] = features_df['close'].rolling(window=20).mean() - 2 * features_df['close'].rolling(window=20).std()
        
        # Store engineered features
        features_id = self.artifacts.store_output(features_df, "engineered_features")
        tprint_success(f"✅ Engineered features for {len(features_df)} records")
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class ModelTrainingStep(EnhancedBaseStep):
    """Step that trains a model on engineered features."""
    
    async def _execute_step(self, data: Any) -> Any:
        """Train a model."""
        tprint_info(f"🤖 Training model in {self.step_name}")
        
        # Load engineered features
        features_df = self.artifacts.load_output("engineered_features", "output")
        
        # Simulate model training
        model = {
            'type': 'RandomForest',
            'features': ['sma_20', 'sma_50', 'rsi', 'bollinger_upper', 'bollinger_lower'],
            'accuracy': 0.85,
            'trained_at': datetime.now().isoformat()
        }
        
        # Store model
        model_id = self.artifacts.store_model(model, "trained_model")
        tprint_success(f"✅ Trained model with accuracy: {model['accuracy']}")
        
        return model


def example_multi_step_workflow():
    """Example showing a multi-step workflow."""
    tprint_info("📚 EXAMPLE 3: Multi-Step Workflow")
    
    # Create unified system
    system = UnifiedArtifactSystem()
    
    # Define workflow steps
    steps_config = [
        {
            'step_name': 'data_collection',
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'interval': '1m',
            'direction': 'long',
            'model': 'Analyst'
        },
        {
            'step_name': 'feature_engineering',
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'interval': '1m',
            'direction': 'long',
            'model': 'Analyst'
        },
        {
            'step_name': 'model_training',
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'interval': '1m',
            'direction': 'long',
            'model': 'Analyst'
        }
    ]
    
    # Create step instances
    steps = [
        DataCollectionStep(steps_config[0], system),
        FeatureEngineeringStep(steps_config[1], system),
        ModelTrainingStep(steps_config[2], system)
    ]
    
    # Execute workflow
    async def run_workflow():
        results = []
        for i, step in enumerate(steps):
            tprint_info(f"🔄 Executing step {i+1}/{len(steps)}: {step.step_name}")
            
            # Validate step
            step.validate_config()
            
            # Execute step
            result = await step.execute(None)
            results.append(result)
            
            # Get step status
            status = step.get_status()
            tprint_info(f"📊 Step {step.step_name} status: {status['execution_status']['last_execution_success']}")
        
        return results
    
    # Run the workflow
    workflow_results = asyncio.run(run_workflow())
    
    # Get overall workflow metrics
    tprint_info("📈 WORKFLOW COMPLETED")
    for i, step in enumerate(steps):
        summary = step.get_execution_summary()
        tprint_info(f"Step {i+1} ({step.step_name}): {summary['success_rate']*100:.1f}% success rate")
    
    # List all artifacts created
    all_artifacts = system.list_artifacts()
    tprint_info(f"📁 Total artifacts created: {len(all_artifacts)}")
    
    # Get system performance metrics
    system_metrics = system.get_performance_metrics()
    tprint_info(f"📊 System performance: {system_metrics}")


# Example 4: Error Handling and Recovery
class FaultyStep(EnhancedBaseStep):
    """Step that demonstrates error handling."""
    
    def __init__(self, config: Dict[str, Any], artifact_system=None, fail_on_attempt: int = 2):
        super().__init__(config, artifact_system)
        self.fail_on_attempt = fail_on_attempt
        self.attempt_count = 0
    
    async def _execute_step(self, data: Any) -> Any:
        """Step that fails on specific attempts."""
        self.attempt_count += 1
        tprint_info(f"🔄 Attempting execution {self.attempt_count} in {self.step_name}")
        
        if self.attempt_count == self.fail_on_attempt:
            raise Exception(f"Simulated failure on attempt {self.attempt_count}")
        
        # Store some data
        result = {"attempt": self.attempt_count, "status": "success"}
        self.artifacts.store_output(result, "attempt_result")
        
        return result


def example_error_handling():
    """Example showing error handling and recovery."""
    tprint_info("📚 EXAMPLE 4: Error Handling and Recovery")
    
    # Create unified system
    system = UnifiedArtifactSystem()
    
    # Create faulty step
    config = {
        'step_name': 'faulty_step',
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'interval': '1m'
    }
    
    step = FaultyStep(config, system, fail_on_attempt=2)
    
    # Execute step multiple times to demonstrate error handling
    async def run_with_retries():
        for attempt in range(3):
            try:
                tprint_info(f"🔄 Attempt {attempt + 1}")
                result = await step.execute(None)
                tprint_success(f"✅ Success on attempt {attempt + 1}")
                return result
            except Exception as e:
                tprint_warning(f"⚠️ Failed on attempt {attempt + 1}: {str(e)}")
                if attempt < 2:  # Not the last attempt
                    tprint_info("🔄 Retrying...")
                else:
                    tprint_error("❌ All attempts failed")
                    raise
    
    # Run with retries
    try:
        result = asyncio.run(run_with_retries())
        tprint_success(f"✅ Final result: {result}")
    except Exception as e:
        tprint_error(f"❌ Workflow failed: {str(e)}")
    
    # Get step status
    status = step.get_status()
    tprint_info(f"📊 Step status: {status['execution_status']}")
    
    # Get execution summary
    summary = step.get_execution_summary()
    tprint_info(f"📈 Execution summary: {summary}")


# Example 5: Performance Monitoring
def example_performance_monitoring():
    """Example showing performance monitoring capabilities."""
    tprint_info("📚 EXAMPLE 5: Performance Monitoring")
    
    # Create unified system with performance tracking
    config = UnifiedConfig(
        base_dir="examples/performance_monitoring",
        enable_compression=True,
        enable_caching=True,
        enable_memory_optimization=True
    )
    
    system = UnifiedArtifactSystem(config)
    
    # Set context
    system.set_context(
        step_name="performance_test",
        symbol="BTCUSDT",
        exchange="binance",
        interval="5m"
    )
    
    # Create large dataset for performance testing
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=8640, freq='5min')
    large_dataset = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, 8640),
        'high': np.random.uniform(50000, 60000, 8640),
        'low': np.random.uniform(30000, 40000, 8640),
        'close': np.random.uniform(40000, 50000, 8640),
        'volume': np.random.uniform(1000, 10000, 8640)
    })
    
    # Store large dataset
    start_time = datetime.now()
    klines_id = system.store_klines(
        df=large_dataset,
        symbol="BTCUSDT",
        exchange="binance",
        interval="5m"
    )
    store_time = (datetime.now() - start_time).total_seconds()
    tprint_success(f"✅ Stored {len(large_dataset)} records in {store_time:.2f}s")
    
    # Load dataset
    start_time = datetime.now()
    loaded_data = system.load_klines("BTCUSDT", "binance", "5m")
    load_time = (datetime.now() - start_time).total_seconds()
    tprint_success(f"✅ Loaded {len(loaded_data)} records in {load_time:.2f}s")
    
    # Store multiple artifacts
    artifacts_created = []
    for i in range(10):
        artifact_data = {
            'id': f'artifact_{i}',
            'data': np.random.random(1000).tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        artifact_id = system.store_artifact(
            data=artifact_data,
            artifact_name=f"test_artifact_{i}",
            artifact_type="test"
        )
        store_time = (datetime.now() - start_time).total_seconds()
        
        artifacts_created.append({
            'id': artifact_id,
            'store_time': store_time
        })
    
    # Get performance metrics
    metrics = system.get_performance_metrics()
    tprint_info(f"📊 Performance Metrics:")
    tprint_info(f"  - Total operations: {metrics['total_operations']}")
    tprint_info(f"  - Klines operations: {metrics['klines_operations']}")
    tprint_info(f"  - Generic operations: {metrics['generic_operations']}")
    tprint_info(f"  - Cache hits: {metrics['cache_hits']}")
    tprint_info(f"  - Cache misses: {metrics['cache_misses']}")
    
    # List all artifacts
    all_artifacts = system.list_artifacts()
    tprint_info(f"📁 Total artifacts: {len(all_artifacts)}")
    
    # Cleanup
    system.cleanup()


def main():
    """Run all examples."""
    tprint_info("🚀 RUNNING UNIFIED ARTIFACT INTEGRATION EXAMPLES")
    
    try:
        # Run examples
        example_basic_unified_usage()
        print("\n" + "="*80 + "\n")
        
        example_enhanced_basestep_usage()
        print("\n" + "="*80 + "\n")
        
        example_multi_step_workflow()
        print("\n" + "="*80 + "\n")
        
        example_error_handling()
        print("\n" + "="*80 + "\n")
        
        example_performance_monitoring()
        
        tprint_success("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        tprint_error(f"❌ EXAMPLES FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    main()