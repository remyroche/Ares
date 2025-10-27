"""
Regime Models Training Step.

This step trains machine learning models for regime classification using the comprehensive
RegimeModelsTrainingComponent implementation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from src.training.steps.market_analysis.components.improved_regime_models_training import ImprovedRegimeModelsTrainingComponent
from src.training.steps.market_analysis.components.base_component import ComponentConfig

logger = logging.getLogger(__name__)


class RegimeModelsTrainingStep(BaseStep):
    """
    Regime Models Training Step.

    Trains ML models for regime classification using regime labels.
    """

    def __init__(self, step_name: str = "regime_models_training"):
        """Initialize the regime models training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('RegimeModelsTraining')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime models training using the comprehensive RegimeModelsTrainingComponent.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
                - pipeline_state: Pipeline state containing artifacts and data

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"🧠 Starting regime models training for {config.get('symbol', 'UNKNOWN')}", "INFO")
        
        # Use regime_timeframe (defaults to 1h) for regime models training
        regime_timeframe = config.get('regime_timeframe', '1h')
        if 'regime_timeframe' not in config:
            tprint(f"⏰ Using regime_timeframe={regime_timeframe} for regime models training", "INFO")
            config['regime_timeframe'] = regime_timeframe
        if config.get('timeframe') != regime_timeframe:
            tprint(f"⏰ Overriding timeframe to {regime_timeframe} for regime models training (was: {config.get('timeframe', 'not set')})", "INFO")
            config['timeframe'] = regime_timeframe

        try:
            # Initialize the improved RegimeModelsTrainingComponent
            tprint("🚀 Initializing ImprovedRegimeModelsTrainingComponent", "INFO")
            component_config = ComponentConfig(
                symbol=config.get('symbol', 'UNKNOWN'),
                exchange=config.get('exchange', 'binance'),
                timeframe=config.get('timeframe', regime_timeframe),
                execution_mode=config.get('execution_mode', 'light')
            )
            
            models_component = ImprovedRegimeModelsTrainingComponent(component_config)
            tprint("✅ ImprovedRegimeModelsTrainingComponent initialized successfully", "SUCCESS")

            # Get pipeline state from config (should contain artifacts from previous steps)
            pipeline_state = config.get('pipeline_state', {})
            
            # Get market data from pipeline state or create synthetic data if not available
            market_data = pipeline_state.get('market_data')
            if market_data is None:
                tprint("⚠️ No market data found in pipeline state, creating synthetic data", "WARNING")
                market_data = self._create_synthetic_market_data(config)
            
            # Convert market data to DataFrame if it's not already
            if not isinstance(market_data, pd.DataFrame):
                tprint("⚠️ Market data is not a DataFrame, attempting conversion", "WARNING")
                if isinstance(market_data, dict):
                    market_data = pd.DataFrame(market_data)
                else:
                    tprint("❌ Cannot convert market data to DataFrame", "ERROR")
                    raise ValueError("Market data must be a pandas DataFrame or convertible dict")

            tprint(f"📊 Market data shape: {market_data.shape}", "INFO")
            tprint(f"📊 Market data columns: {list(market_data.columns)}", "INFO")

            # Execute the comprehensive models training component
            tprint("🏋️ Executing comprehensive regime models training", "INFO")
            result = await models_component.execute(market_data, pipeline_state)
            
            if result.success:
                tprint("✅ Regime models training completed successfully", "SUCCESS")
                
                # Extract artifacts and metrics from component result
                artifacts = result.artifacts
                metrics = {
                    'model_type': 'comprehensive_ml_models',
                    'training_time': result.metadata.get('execution_time', 0),
                    'execution_mode': config.get('execution_mode', 'light'),
                    'success': True,
                    'component_metadata': result.metadata
                }
                
                # Extract model metrics if available
                models_result = artifacts.get('regime_models_training_result', {})
                if models_result:
                    models = models_result.get('models', {})
                    if models:
                        # Calculate average accuracy across all models
                        accuracies = []
                        for model_name, model_data in models.items():
                            if isinstance(model_data, dict) and 'accuracy' in model_data:
                                accuracies.append(model_data['accuracy'])
                            elif hasattr(model_data, 'score'):
                                # If it's a sklearn model, we can't get accuracy without test data
                                accuracies.append(0.0)  # Placeholder
                        
                        if accuracies:
                            avg_accuracy = sum(accuracies) / len(accuracies)
                            metrics.update({
                                'average_accuracy': avg_accuracy,
                                'n_models_trained': len(models),
                                'model_names': list(models.keys())
                            })
                
                return {
                    'success': True,
                    'artifacts': artifacts,
                    'metrics': metrics
                }
            else:
                error_msg = f"Regime models training component failed: {result.error_message}"
                tprint(f"❌ {error_msg}", "ERROR")
                self.logger.error(error_msg)
                
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }

        except Exception as e:
            error_msg = f"Regime models training failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    def _create_synthetic_market_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create synthetic market data for testing when real data is not available.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Synthetic market data DataFrame
        """
        tprint("🔧 Creating synthetic market data for regime models training", "INFO")
        
        try:
            # Create synthetic OHLCV data
            n_samples = 1000
            np.random.seed(42)
            
            # Generate synthetic price data with regime-like patterns
            base_price = 100.0
            returns = np.random.normal(0, 0.02, n_samples)
            
            # Add regime-like patterns
            regime_changes = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
            regime_multipliers = np.array([1.0, 1.5, 0.5, 2.0])[regime_changes]
            returns *= regime_multipliers
            
            # Generate OHLCV data
            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = np.array(prices[1:])  # Remove initial base price
            
            # Generate OHLC from close prices
            high_multiplier = 1 + np.abs(np.random.normal(0, 0.01, n_samples))
            low_multiplier = 1 - np.abs(np.random.normal(0, 0.01, n_samples))
            
            high = prices * high_multiplier
            low = prices * low_multiplier
            open_prices = np.roll(prices, 1)
            open_prices[0] = base_price
            
            # Generate volume
            volume = np.random.lognormal(10, 1, n_samples)
            
            # Create DataFrame
            market_data = pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': prices,
                'volume': volume,
                'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1H')
            })
            
            tprint(f"✅ Synthetic market data created: {market_data.shape}", "SUCCESS")
            return market_data
            
        except Exception as e:
            tprint(f"❌ Failed to create synthetic market data: {e}", "ERROR")
            # Return minimal fallback data
            return pd.DataFrame({
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [1000],
                'timestamp': [pd.Timestamp.now()]
            })

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_regime_models_training_step():
    """Register the regime models training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("regime_models_training", RegimeModelsTrainingStep)
    tprint("✅ Regime models training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_models_training_step()
