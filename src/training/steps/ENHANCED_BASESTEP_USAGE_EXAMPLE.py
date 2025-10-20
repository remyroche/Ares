"""
Enhanced BaseStep Usage Example

This file demonstrates how to use the enhanced BaseStep class with all its advanced features:
- Step-category organization
- Multiple fallback mechanisms
- Enhanced artifact management
- Performance monitoring
- Memory optimization
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

from src.training.steps.base_step import BaseStep


class ExampleEnhancedStep(BaseStep):
    """
    Example step demonstrating enhanced BaseStep functionality.
    
    This step shows how to use all the advanced features:
    - Context setting for enhanced file naming
    - Multiple fallback mechanisms for data retrieval
    - Enhanced artifact storage and retrieval
    - Performance monitoring
    - Memory analytics
    """
    
    def __init__(self, step_name: str = "example_enhanced_step"):
        super().__init__(step_name)
        self.logger.info(f"🚀 ExampleEnhancedStep initialized: {step_name}")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the example step with enhanced functionality.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'BTCUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - information: Information type (e.g., 'klines')
                - direction: Trading direction ('long' or 'short')
                - model: Model type ('Analyst' or 'Tactician')
        
        Returns:
            Execution result with artifacts and metrics
        """
        try:
            # 1. Set context for enhanced file naming and path management
            self._set_context(
                symbol=config.get('symbol', 'BTCUSDT'),
                exchange=config.get('exchange', 'binance'),
                information=config.get('information', 'klines'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # 2. Demonstrate data loading with fallback mechanisms
            self.logger.info("📊 Loading market data with fallback support...")
            market_data = self._load_dataframe('market_data')
            
            if market_data is None:
                # Create sample data if not found
                self.logger.info("📊 No existing data found, creating sample data...")
                market_data = self._create_sample_data()
            
            # 3. Process the data
            self.logger.info("⚙️ Processing data...")
            processed_data = self._process_data(market_data)
            
            # 4. Save processed data with enhanced features
            self.logger.info("💾 Saving processed data...")
            self._save_dataframe(processed_data, 'processed_data', {
                'processing_timestamp': datetime.now().isoformat(),
                'original_rows': len(market_data),
                'processed_rows': len(processed_data),
                'processing_method': 'enhanced_example'
            })
            
            # 5. Demonstrate model saving and loading
            self.logger.info("🤖 Creating and saving model...")
            model = self._create_sample_model()
            self._save_model(model, 'sample_model', {
                'model_type': 'example_model',
                'created_at': datetime.now().isoformat(),
                'parameters': {'learning_rate': 0.01, 'epochs': 100}
            })
            
            # 6. Demonstrate metadata saving
            metadata = {
                'step_name': self.step_name,
                'execution_time': datetime.now().isoformat(),
                'config': config,
                'artifacts_created': ['processed_data', 'sample_model'],
                'performance_metrics': self._get_performance_metrics()
            }
            self._save_metadata(metadata, 'execution_metadata')
            
            # 7. Get performance and memory analytics
            performance_metrics = self._get_performance_metrics()
            memory_analytics = self._get_memory_analytics()
            
            self.logger.info(f"📊 Performance Metrics: {performance_metrics}")
            self.logger.info(f"🧠 Memory Analytics: {memory_analytics}")
            
            # 8. Demonstrate fallback retrieval
            self.logger.info("🔄 Testing fallback retrieval...")
            retrieved_data = self._load_dataframe('processed_data')
            if retrieved_data is not None:
                self.logger.info(f"✅ Successfully retrieved data: {retrieved_data.shape}")
            
            # 9. Demonstrate different fallback levels
            self.logger.info("🔄 Testing different fallback levels...")
            
            # Test primary (step-category) retrieval
            primary_data = self._load_dataframe('processed_data')
            if primary_data is not None:
                self.logger.info("✅ Primary fallback (step-category) successful")
            
            # Test generic search (without model/direction)
            generic_data = self._load_dataframe('some_other_data')
            if generic_data is not None:
                self.logger.info("✅ Generic search fallback successful")
            
            return {
                'success': True,
                'artifacts': ['processed_data', 'sample_model', 'execution_metadata'],
                'metrics': {
                    'performance': performance_metrics,
                    'memory': memory_analytics,
                    'data_shape': processed_data.shape if processed_data is not None else None
                },
                'message': 'Enhanced step completed successfully with all features demonstrated'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Step execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample market data for demonstration."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        np.random.seed(42)
        
        data = {
            'timestamp': dates,
            'open': 50000 + np.random.randn(len(dates)) * 1000,
            'high': 50000 + np.random.randn(len(dates)) * 1000 + 500,
            'low': 50000 + np.random.randn(len(dates)) * 1000 - 500,
            'close': 50000 + np.random.randn(len(dates)) * 1000,
            'volume': np.random.randint(100, 10000, len(dates))
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the market data."""
        processed = data.copy()
        
        # Add some technical indicators
        processed['sma_20'] = processed['close'].rolling(window=20).mean()
        processed['rsi'] = self._calculate_rsi(processed['close'])
        processed['price_change'] = processed['close'].pct_change()
        
        # Remove NaN values
        processed = processed.dropna()
        
        return processed
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_sample_model(self) -> Dict[str, Any]:
        """Create a sample model for demonstration."""
        return {
            'model_type': 'sample_classifier',
            'parameters': {
                'learning_rate': 0.01,
                'epochs': 100,
                'batch_size': 32,
                'hidden_layers': [64, 32, 16]
            },
            'weights': np.random.randn(100, 10).tolist(),  # Sample weights
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }


async def main():
    """Main function to demonstrate the enhanced BaseStep usage."""
    print("🚀 Enhanced BaseStep Usage Example")
    print("=" * 50)
    
    # Create the step
    step = ExampleEnhancedStep("example_enhanced_step")
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'information': 'klines',
        'direction': 'long',
        'model': 'Analyst',
        'timeframe': '1h'
    }
    
    print(f"📋 Configuration: {config}")
    print()
    
    # Run the step
    print("🔄 Running enhanced step...")
    result = await step.run(config)
    
    # Display results
    print("\n📊 Results:")
    print(f"Success: {result['success']}")
    print(f"Artifacts: {result.get('artifacts', [])}")
    
    if 'metrics' in result:
        print("\n📈 Performance Metrics:")
        for key, value in result['metrics'].get('performance', {}).items():
            print(f"  {key}: {value}")
        
        print("\n🧠 Memory Analytics:")
        for key, value in result['metrics'].get('memory', {}).items():
            print(f"  {key}: {value}")
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    
    print(f"\n⏱️ Execution Time: {result.get('execution_time', 0):.2f} seconds")
    
    # Demonstrate cache clearing
    print("\n🧹 Clearing cache...")
    step._clear_cache()
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())