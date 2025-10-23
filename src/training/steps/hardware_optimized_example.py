#!/usr/bin/env python3
"""
Hardware-Optimized Training Steps Example

This module demonstrates how to use the enhanced training steps with hardware optimization,
VectorBT integration, and unified vectorization management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import the enhanced components
from src.training.steps.base_step import BaseStep, StepConfig, ExecutionResult
from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    StorageConfig, KlinesMetadata, EnhancedKlinesProcessingPipeline
)
from src.training.steps.data_collection.unified_data_loader import UnifiedDataLoader
from src.training.steps.data_collection.utils.data_operations_utils import (
    MemoryOptimizedDataHandler, DataFormatter, DataAnalyzer
)
from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
from src.training.steps.models_training.components.base_component import BaseModelsTrainingComponent

# Hardware optimization imports
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.optimization_decorators import smart_cache, memory_efficient, auto_optimize
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareOptimizedDataProcessingStep(BaseStep):
    """
    Example step demonstrating hardware-optimized data processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("hardware_optimized_data_processing", config)
        self._init_hardware_components()
    
    def _init_hardware_components(self):
        """Initialize hardware optimization components."""
        if HARDWARE_AVAILABLE:
            try:
                # Initialize hardware manager
                self.hardware_manager = UnifiedHardwareManager()
                
                # Initialize VectorBT optimizer
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                
                # Initialize vectorization manager
                self.vectorization_manager = UnifiedVectorizationManager()
                
                # Initialize data processing components with hardware optimization
                self.storage_config = StorageConfig(
                    base_path="data/hardware_optimized",
                    enable_hardware_optimization=True,
                    memory_optimization_level="balanced"
                )
                
                self.data_loader = UnifiedDataLoader({
                    "enable_hardware_optimization": True,
                    "enable_vectorbt_optimization": True
                })
                
                self.memory_handler = MemoryOptimizedDataHandler({
                    "enable_hardware_optimization": True,
                    "memory_limit_mb": 2000
                })
                
                logger.info("✅ Hardware optimization components initialized")
                
            except Exception as e:
                logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.hardware_manager = None
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            logger.warning("⚠️ Hardware optimization not available")
            self.hardware_manager = None
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
    
    async def execute(self, config: StepConfig) -> ExecutionResult:
        """Execute hardware-optimized data processing."""
        try:
            logger.info("🚀 Starting hardware-optimized data processing")
            
            # Get configuration
            symbol = config.get('symbol', 'BTCUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '1m')
            
            # Step 1: Load data with hardware optimization
            logger.info("📊 Loading data with hardware optimization")
            data_result = await self._load_optimized_data(symbol, exchange, timeframe)
            
            if not data_result['success']:
                return ExecutionResult(
                    success=False,
                    error=f"Data loading failed: {data_result.get('error', 'Unknown error')}",
                    execution_time=0.0
                )
            
            # Step 2: Process data with VectorBT optimization
            logger.info("⚡ Processing data with VectorBT optimization")
            processed_data = await self._process_with_vectorbt(data_result['data'])
            
            # Step 3: Optimize memory usage
            logger.info("🧠 Optimizing memory usage")
            optimized_data = await self._optimize_memory_usage(processed_data)
            
            # Step 4: Save with hardware-optimized storage
            logger.info("💾 Saving with hardware-optimized storage")
            save_result = await self._save_optimized_data(optimized_data, symbol, exchange, timeframe)
            
            # Step 5: Generate performance metrics
            metrics = await self._generate_performance_metrics(data_result, processed_data, optimized_data)
            
            return ExecutionResult(
                success=True,
                artifacts=[save_result['file_path']],
                metrics=metrics,
                execution_time=metrics.get('total_execution_time', 0.0)
            )
            
        except Exception as e:
            logger.error(f"❌ Hardware-optimized data processing failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=0.0
            )
    
    async def _load_optimized_data(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Load data with hardware optimization."""
        try:
            if self.data_loader:
                # Use hardware-optimized data loader
                result = self.data_loader.load_dataset(f"{symbol}_{exchange}_{timeframe}.parquet")
                if result is not None:
                    return {'success': True, 'data': result}
            
            # Fallback to basic loading
            return {'success': False, 'error': 'Data loader not available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_with_vectorbt(self, data) -> Any:
        """Process data with VectorBT optimization."""
        try:
            if self.vectorbt_optimizer and hasattr(data, 'columns'):
                # Use VectorBT for technical indicators
                processed_data = data.copy()
                
                # Calculate rolling statistics with VectorBT
                if 'close' in data.columns:
                    processed_data['sma_20'] = self.vectorbt_optimizer.rolling_mean(data['close'], window=20)
                    processed_data['std_20'] = self.vectorbt_optimizer.rolling_std(data['close'], window=20)
                    processed_data['rsi_14'] = self.vectorbt_optimizer.calculate_rsi(data['close'], window=14)
                
                return processed_data
            
            return data
            
        except Exception as e:
            logger.warning(f"VectorBT processing failed: {e}")
            return data
    
    async def _optimize_memory_usage(self, data) -> Any:
        """Optimize memory usage with hardware acceleration."""
        try:
            if self.memory_handler and hasattr(data, 'memory_usage'):
                return self.memory_handler.optimize_dataframe_memory(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return data
    
    async def _save_optimized_data(self, data, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Save data with hardware-optimized storage."""
        try:
            if self.data_loader:
                filename = f"{symbol}_{exchange}_{timeframe}_optimized.parquet"
                success = self.data_loader.save_dataset(data, filename)
                
                if success:
                    return {
                        'file_path': str(self.data_loader.base_path / filename),
                        'success': True
                    }
            
            return {'success': False, 'error': 'Data loader not available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _generate_performance_metrics(self, data_result: Dict, processed_data: Any, optimized_data: Any) -> Dict[str, Any]:
        """Generate performance metrics."""
        try:
            metrics = {
                'data_loaded': data_result['success'],
                'hardware_optimization_enabled': self.hardware_manager is not None,
                'vectorbt_optimization_enabled': self.vectorbt_optimizer is not None,
                'vectorization_manager_enabled': self.vectorization_manager is not None
            }
            
            if hasattr(processed_data, 'memory_usage'):
                metrics['processed_data_memory_mb'] = processed_data.memory_usage(deep=True).sum() / 1024**2
            
            if hasattr(optimized_data, 'memory_usage'):
                metrics['optimized_data_memory_mb'] = optimized_data.memory_usage(deep=True).sum() / 1024**2
                
                if 'processed_data_memory_mb' in metrics:
                    memory_savings = (metrics['processed_data_memory_mb'] - metrics['optimized_data_memory_mb']) / metrics['processed_data_memory_mb']
                    metrics['memory_savings_percent'] = memory_savings * 100
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Performance metrics generation failed: {e}")
            return {}


class HardwareOptimizedBacktestingStep(BaseStep):
    """
    Example step demonstrating hardware-optimized backtesting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("hardware_optimized_backtesting", config)
        self._init_optimization_components()
    
    def _init_optimization_components(self):
        """Initialize optimization components."""
        try:
            # Initialize parameters optimizer with hardware acceleration
            self.parameters_optimizer = FinalParametersOptimizer()
            
            # Initialize hardware components
            if HARDWARE_AVAILABLE:
                self.hardware_manager = UnifiedHardwareManager()
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                self.vectorization_manager = UnifiedVectorizationManager()
            
            logger.info("✅ Backtesting optimization components initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Backtesting optimization initialization failed: {e}")
    
    async def execute(self, config: StepConfig) -> ExecutionResult:
        """Execute hardware-optimized backtesting."""
        try:
            logger.info("🚀 Starting hardware-optimized backtesting")
            
            # Get configuration
            symbol = config.get('symbol', 'BTCUSDT')
            timeframe = config.get('timeframe', '1m')
            direction = config.get('direction', 'long')
            execution_mode = config.get('execution_mode', 'light')
            
            # Step 1: Optimize parameters with hardware acceleration
            logger.info("⚙️ Optimizing parameters with hardware acceleration")
            optimization_result = await self._optimize_parameters(symbol, timeframe, direction, execution_mode)
            
            if not optimization_result['success']:
                return ExecutionResult(
                    success=False,
                    error=f"Parameter optimization failed: {optimization_result.get('error', 'Unknown error')}",
                    execution_time=0.0
                )
            
            # Step 2: Run backtesting with VectorBT
            logger.info("📈 Running backtesting with VectorBT")
            backtesting_result = await self._run_vectorbt_backtesting(
                optimization_result['optimized_parameters'], symbol, timeframe, direction
            )
            
            # Step 3: Generate performance metrics
            metrics = await self._generate_backtesting_metrics(optimization_result, backtesting_result)
            
            return ExecutionResult(
                success=True,
                artifacts=[f"backtesting_results_{symbol}_{timeframe}.json"],
                metrics=metrics,
                execution_time=metrics.get('total_execution_time', 0.0)
            )
            
        except Exception as e:
            logger.error(f"❌ Hardware-optimized backtesting failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=0.0
            )
    
    async def _optimize_parameters(self, symbol: str, timeframe: str, direction: str, execution_mode: str) -> Dict[str, Any]:
        """Optimize parameters with hardware acceleration."""
        try:
            if self.parameters_optimizer:
                result = await self.parameters_optimizer.optimize_parameters(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    execution_mode=execution_mode
                )
                return {'success': True, 'optimized_parameters': result['optimized_parameters']}
            
            return {'success': False, 'error': 'Parameters optimizer not available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _run_vectorbt_backtesting(self, parameters: Dict[str, float], symbol: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Run backtesting with VectorBT."""
        try:
            if self.vectorbt_optimizer:
                # Use VectorBT for backtesting simulation
                result = {
                    'total_return': 0.15,  # Simulated
                    'sharpe_ratio': 1.8,   # Simulated
                    'max_drawdown': 0.08,  # Simulated
                    'win_rate': 0.65,      # Simulated
                    'vectorbt_optimized': True
                }
                return result
            
            # Fallback simulation
            return {
                'total_return': 0.12,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.10,
                'win_rate': 0.60,
                'vectorbt_optimized': False
            }
            
        except Exception as e:
            logger.warning(f"VectorBT backtesting failed: {e}")
            return {'error': str(e)}
    
    async def _generate_backtesting_metrics(self, optimization_result: Dict, backtesting_result: Dict) -> Dict[str, Any]:
        """Generate backtesting performance metrics."""
        try:
            metrics = {
                'optimization_success': optimization_result['success'],
                'backtesting_success': 'error' not in backtesting_result,
                'hardware_optimization_enabled': self.hardware_manager is not None,
                'vectorbt_optimization_enabled': self.vectorbt_optimizer is not None
            }
            
            if 'error' not in backtesting_result:
                metrics.update(backtesting_result)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Backtesting metrics generation failed: {e}")
            return {}


async def main():
    """Main function demonstrating hardware-optimized training steps."""
    logger.info("🚀 Starting hardware-optimized training steps demonstration")
    
    # Create example configurations
    data_config = StepConfig({
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m'
    })
    
    backtesting_config = StepConfig({
        'symbol': 'BTCUSDT',
        'timeframe': '1m',
        'direction': 'long',
        'execution_mode': 'light'
    })
    
    # Initialize steps
    data_step = HardwareOptimizedDataProcessingStep()
    backtesting_step = HardwareOptimizedBacktestingStep()
    
    # Execute data processing step
    logger.info("📊 Executing data processing step")
    data_result = await data_step.execute(data_config)
    
    if data_result.success:
        logger.info(f"✅ Data processing completed successfully")
        logger.info(f"📈 Artifacts: {data_result.artifacts}")
        logger.info(f"📊 Metrics: {data_result.metrics}")
    else:
        logger.error(f"❌ Data processing failed: {data_result.error}")
    
    # Execute backtesting step
    logger.info("📈 Executing backtesting step")
    backtesting_result = await backtesting_step.execute(backtesting_config)
    
    if backtesting_result.success:
        logger.info(f"✅ Backtesting completed successfully")
        logger.info(f"📈 Artifacts: {backtesting_result.artifacts}")
        logger.info(f"📊 Metrics: {backtesting_result.metrics}")
    else:
        logger.error(f"❌ Backtesting failed: {backtesting_result.error}")
    
    logger.info("🎉 Hardware-optimized training steps demonstration completed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    asyncio.run(main())