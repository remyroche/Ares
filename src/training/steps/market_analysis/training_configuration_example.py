"""
Training Configuration Example

This module demonstrates how to configure and use the refined tactician and analyst training data.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import the training components
from .multi_horizon_profit_labeler import MultiHorizonProfitLabeler, MultiHorizonConfig
from .pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
from .feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .final_feature_selection_step import FinalFeatureSelectionStep
from .tactician_training_adapter import TacticianTrainingAdapter, run_tactician_training

# Import base components
from .components.base_component import ComponentConfig


class TrainingMode(Enum):
    """Training mode enumeration."""
    ANALYST = "analyst"  # 5m timeframe, no long/short differentiation
    TACTICIAN = "tactician"  # 1m timeframe, long/short differentiation


@dataclass
class TrainingConfiguration:
    """Configuration for training setup."""
    mode: TrainingMode
    symbol: str
    exchange: str
    timeframe: str
    data_path: str
    output_path: str
    analyst_mode: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode == TrainingMode.ANALYST and self.timeframe != '5m':
            raise ValueError("Analyst mode requires 5m timeframe")
        if self.mode == TrainingMode.TACTICIAN and self.timeframe != '1m':
            raise ValueError("Tactician mode requires 1m timeframe")
        
        self.analyst_mode = self.mode == TrainingMode.ANALYST


class TrainingPipeline:
    """Training pipeline for both Analyst and Tactician models."""
    
    def __init__(self, config: TrainingConfiguration):
        """Initialize training pipeline."""
        self.config = config
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize training components based on mode."""
        base_config = ComponentConfig(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe
        )
        
        if self.config.mode == TrainingMode.ANALYST:
            # ANALYST MODE: No long/short differentiation
            self.components['multi_horizon_labeler'] = MultiHorizonProfitLabeler(
                MultiHorizonConfig(analyst_mode=True)
            )
            self.components['pid_generator'] = PIDBasedFeatureGenerationComponent(base_config)
            self.components['lookback_optimizer'] = FeatureLookbackOptimizationComponent(base_config)
            self.components['feature_selector'] = FinalFeatureSelectionStep({
                'timeframe': self.config.timeframe,
                'model_type': 'analyst'
            })
            
        else:  # TACTICIAN MODE
            # TACTICIAN MODE: Long/short differentiation with adapter
            self.components['tactician_adapter'] = TacticianTrainingAdapter(base_config)
    
    async def run_analyst_training(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run analyst training (5m timeframe, no long/short differentiation)."""
        if self.config.mode != TrainingMode.ANALYST:
            raise ValueError("Analyst training requires analyst mode")
        
        results = {}
        
        # Step 1: Multi-horizon profit labeling (analyst mode)
        self.logger.info("📊 Running multi-horizon profit labeling (analyst mode)...")
        labeled_data = self.components['multi_horizon_labeler'].generate_labels(data)
        results['labeled_data'] = labeled_data
        
        # Step 2: Feature lookback optimization
        self.logger.info("⚙️ Running feature lookback optimization...")
        lookback_result = await self.components['lookback_optimizer'].execute(data, pipeline_state)
        results['lookback_optimization'] = lookback_result.artifacts
        
        # Step 3: PID-based feature generation
        self.logger.info("🚀 Running PID-based feature generation...")
        pid_result = await self.components['pid_generator'].execute(data, pipeline_state)
        results['pid_generation'] = pid_result.artifacts
        
        # Step 4: Final feature selection
        self.logger.info("📊 Running final feature selection...")
        selection_success = await self.components['feature_selector'].execute_final_feature_selection(
            self.config.symbol,
            self.config.exchange,
            self.config.timeframe,
            self.config.data_path
        )
        results['feature_selection'] = selection_success
        
        return results
    
    async def run_tactician_training(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run tactician training (1m timeframe, long/short differentiation)."""
        if self.config.mode != TrainingMode.TACTICIAN:
            raise ValueError("Tactician training requires tactician mode")
        
        # Use the tactician adapter for long/short separation
        self.logger.info("🔧 Running tactician training with long/short separation...")
        tactician_result = await self.components['tactician_adapter'].execute(data, pipeline_state)
        
        return tactician_result.artifacts


# Example usage functions
async def run_analyst_training_example():
    """Example of running analyst training."""
    config = TrainingConfiguration(
        mode=TrainingMode.ANALYST,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="5m",
        data_path="historical_data",
        output_path="outcomes/analyst_training"
    )
    
    pipeline = TrainingPipeline(config)
    
    # Load your data here
    data = None  # Your market data
    
    # Run analyst training
    results = await pipeline.run_analyst_training(data, {})
    
    print("✅ Analyst training completed")
    print(f"📊 Labeled data shape: {results['labeled_data'].shape if hasattr(results['labeled_data'], 'shape') else 'N/A'}")
    print(f"⚙️ Lookback optimization: {'Success' if results['lookback_optimization'] else 'Failed'}")
    print(f"🚀 PID generation: {'Success' if results['pid_generation'] else 'Failed'}")
    print(f"📊 Feature selection: {'Success' if results['feature_selection'] else 'Failed'}")


async def run_tactician_training_example():
    """Example of running tactician training."""
    config = TrainingConfiguration(
        mode=TrainingMode.TACTICIAN,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1m",
        data_path="historical_data",
        output_path="outcomes/tactician_training"
    )
    
    pipeline = TrainingPipeline(config)
    
    # Load your data here
    data = None  # Your market data
    
    # Run tactician training
    results = await pipeline.run_tactician_training(data, {})
    
    tactician_result = results.get('tactician_training_result', {})
    long_model = tactician_result.get('long_model', {})
    short_model = tactician_result.get('short_model', {})
    
    print("✅ Tactician training completed")
    print(f"🚀 Long model: {len(long_model.get('features', []))} features")
    print(f"🚀 Short model: {len(short_model.get('features', []))} features")
    print(f"📊 Long training successful: {long_model.get('training_successful', False)}")
    print(f"📊 Short training successful: {short_model.get('training_successful', False)}")


async def run_both_training_example():
    """Example of running both analyst and tactician training."""
    # First, run analyst training on 5m data
    analyst_config = TrainingConfiguration(
        mode=TrainingMode.ANALYST,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="5m",
        data_path="historical_data",
        output_path="outcomes/analyst_training"
    )
    
    analyst_pipeline = TrainingPipeline(analyst_config)
    analyst_results = await analyst_pipeline.run_analyst_training(None, {})
    
    print("✅ Analyst training completed")
    
    # Then, run tactician training on 1m data with analyst results
    tactician_config = TrainingConfiguration(
        mode=TrainingMode.TACTICIAN,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1m",
        data_path="historical_data",
        output_path="outcomes/tactician_training"
    )
    
    tactician_pipeline = TrainingPipeline(tactician_config)
    
    # Pass analyst results to tactician training
    pipeline_state = {
        'analyst_results': analyst_results
    }
    
    tactician_results = await tactician_pipeline.run_tactician_training(None, pipeline_state)
    
    print("✅ Tactician training completed")
    print("🎯 Both models trained successfully!")


if __name__ == "__main__":
    import asyncio
    
    # Run examples
    print("🔧 Running training configuration examples...")
    
    # Example 1: Analyst training only
    print("\n📊 Example 1: Analyst Training (5m timeframe, no long/short differentiation)")
    asyncio.run(run_analyst_training_example())
    
    # Example 2: Tactician training only
    print("\n🚀 Example 2: Tactician Training (1m timeframe, long/short differentiation)")
    asyncio.run(run_tactician_training_example())
    
    # Example 3: Both training modes
    print("\n🎯 Example 3: Both Analyst and Tactician Training")
    asyncio.run(run_both_training_example())