"""
Regime Ensemble Training Step.

This step trains ensemble models for regime classification using the comprehensive
RegimeEnsembleTrainingComponent implementation.
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
from src.training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent
from src.training.steps.market_analysis.components.base_component import ComponentConfig

logger = logging.getLogger(__name__)


class RegimeEnsembleTrainingStep(BaseStep):
    """
    Regime Ensemble Training Step.

    Trains ensemble models for regime classification using meta-learning approaches.
    """

    def __init__(self, step_name: str = "regime_ensemble_training"):
        """Initialize the regime ensemble training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('RegimeEnsembleTraining')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime ensemble training using the comprehensive RegimeEnsembleTrainingComponent.

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
        tprint(f"🎯 Starting regime ensemble training for {config.get('symbol', 'UNKNOWN')}", "INFO")
        
        # Use regime_timeframe (defaults to 1h) for regime ensemble training
        regime_timeframe = config.get('regime_timeframe', '1h')
        if 'regime_timeframe' not in config:
            tprint(f"⏰ Using regime_timeframe={regime_timeframe} for regime ensemble training", "INFO")
            config['regime_timeframe'] = regime_timeframe
        if config.get('timeframe') != regime_timeframe:
            tprint(f"⏰ Overriding timeframe to {regime_timeframe} for regime ensemble training (was: {config.get('timeframe', 'not set')})", "INFO")
            config['timeframe'] = regime_timeframe

        try:
            # Initialize the comprehensive RegimeEnsembleTrainingComponent
            tprint("🚀 Initializing RegimeEnsembleTrainingComponent", "INFO")
            component_config = ComponentConfig(
                symbol=config.get('symbol', 'UNKNOWN'),
                exchange=config.get('exchange', 'binance'),
                timeframe=config.get('timeframe', regime_timeframe),
                execution_mode=config.get('execution_mode', 'light')
            )
            
            ensemble_component = RegimeEnsembleTrainingComponent(component_config)
            tprint("✅ RegimeEnsembleTrainingComponent initialized successfully", "SUCCESS")

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

            # Execute the comprehensive ensemble training component
            tprint("🏋️ Executing comprehensive regime ensemble training", "INFO")
            result = await ensemble_component.execute(market_data, pipeline_state)
            
            if result.success:
                tprint("✅ Regime ensemble training completed successfully", "SUCCESS")
                
                # Extract artifacts and metrics from component result
                artifacts = result.artifacts
                metrics = {
                    'ensemble_type': 'stacker_lgbm_calibrated',
                    'training_time': result.metadata.get('execution_time', 0),
                    'execution_mode': config.get('execution_mode', 'light'),
                    'success': True,
                    'component_metadata': result.metadata
                }
                
                # Extract ensemble metrics if available
                ensemble_metrics = artifacts.get('regime_ensemble_training_result', {}).get('ensemble_metrics', {})
                if ensemble_metrics:
                    stacker_metrics = ensemble_metrics.get('stacker_lgbm_calibrated', {})
                    if stacker_metrics:
                        metrics.update({
                            'ensemble_accuracy': stacker_metrics.get('accuracy', 0),
                            'prediction_confidence': stacker_metrics.get('prediction_confidence', {}),
                            'calibration_method': stacker_metrics.get('calibration_method', 'none')
                        })
                
                return {
                    'success': True,
                    'artifacts': artifacts,
                    'metrics': metrics
                }
            else:
                error_msg = f"Regime ensemble training component failed: {result.error_message}"
                tprint(f"❌ {error_msg}", "ERROR")
                self.logger.error(error_msg)
                
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }

        except Exception as e:
            error_msg = f"Regime ensemble training failed: {str(e)}"
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
        tprint("🔧 Creating synthetic market data for regime ensemble training", "INFO")
        
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
def register_regime_ensemble_training_step():
    """Register the regime ensemble training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
    tprint("✅ Regime ensemble training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_ensemble_training_step()
