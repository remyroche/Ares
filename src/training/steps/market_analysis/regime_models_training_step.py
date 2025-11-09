"""
Regime Models Training Step.

This step trains machine learning models for regime classification using the comprehensive
RegimeModelsTrainingComponent implementation.
"""

import asyncio  # type: ignore  # noqa: F401
import logging
from typing import Any, Dict  # type: ignore  # noqa: F401
from datetime import datetime  # type: ignore  # noqa: F401

# Handle optional dependencies gracefully
try:
    import numpy as np  # noqa: F401
    NUMPY_AVAILABLE = True  # type: ignore
except ImportError:
    NUMPY_AVAILABLE = False  # type: ignore
    np = None  # type: ignore

try:
    import pandas as pd
    PANDAS_AVAILABLE = True  # type: ignore
except ImportError:
    PANDAS_AVAILABLE = False  # type: ignore
    pd = None  # type: ignore

from src.training.steps.base_step import BaseStep  # type: ignore
from src.utils.logger import system_logger  # type: ignore
from src.utils.tprint import tprint  # type: ignore
from src.training.steps.market_analysis.components.regime_models_training import RegimeModelsTrainingComponent  # type: ignore
from src.training.steps.market_analysis.components.base_component import ComponentConfig  # type: ignore

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
        
        # DEBUG: Confirm execute method is being called
        tprint("=" * 80, "INFO")
        tprint("🔥 REGIME_MODELS_TRAINING_STEP.EXECUTE() CALLED!", "SUCCESS")
        tprint(f"🔥 Symbol: {config.get('symbol')}, Execution Mode: {config.get('execution_mode')}", "SUCCESS")
        tprint("=" * 80, "INFO")
        
        # Use regime_timeframe (defaults to 1h) for regime models training
        regime_timeframe = config.get('regime_timeframe', '1h')
        if 'regime_timeframe' not in config:
            tprint(f"⏰ Using regime_timeframe={regime_timeframe} for regime models training", "INFO")
            config['regime_timeframe'] = regime_timeframe
        if config.get('timeframe') != regime_timeframe:
            tprint(f"⏰ Overriding timeframe to {regime_timeframe} for regime models training (was: {config.get('timeframe', 'not set')})", "INFO")
            config['timeframe'] = regime_timeframe

        try:
            # Initialize the RegimeModelsTrainingComponent (now with improvements)
            tprint("🚀 Initializing RegimeModelsTrainingComponent", "INFO")
            component_config = ComponentConfig(
                symbol=config.get('symbol', 'UNKNOWN'),
                exchange=config.get('exchange', 'binance'),
                timeframe=config.get('timeframe', regime_timeframe),
                execution_mode=config.get('execution_mode', 'light')
            )
            
            models_component = RegimeModelsTrainingComponent(component_config)
            tprint("✅ RegimeModelsTrainingComponent initialized successfully", "SUCCESS")

            # Get pipeline state from config (should contain artifacts from previous steps)
            pipeline_state = config.get('pipeline_state', {})

            # For blank mode, prioritize loading fresh data from historical storage
            execution_mode = config.get('execution_mode', 'light')
            symbol = config.get('symbol', 'UNKNOWN')

            # CRITICAL: Set execution_mode in context so BaseStep uses correct mode
            self.set_context(execution_mode=execution_mode)
            tprint(f"🔧 STEP CONTEXT: Set execution_mode={execution_mode} in context", "INFO")

            # Ensure blank_mode_days is in context for BaseStep to use
            if execution_mode == 'blank' and 'blank_mode_days' not in self._current_context:
                self.set_context(blank_mode_days=config.get('blank_mode_days', 180))
                tprint(f"🔧 STEP CONTEXT: Set blank_mode_days={config.get('blank_mode_days', 180)} in context", "INFO")
            elif execution_mode == 'light' and 'light_mode_days' not in self._current_context:
                self.set_context(light_mode_days=config.get('light_mode_days', 20))
                tprint(f"🔧 STEP CONTEXT: Set light_mode_days={config.get('light_mode_days', 20)} in context", "INFO")
            
            # Show current context after setting
            tprint(f"🔧 STEP CONTEXT: Current context keys: {list(self._current_context.keys())}", "INFO")
            tprint(f"🔧 STEP CONTEXT: blank_mode_days in context: {self._current_context.get('blank_mode_days', 'NOT FOUND')}", "INFO")
            
            if execution_mode == 'blank':
                tprint(f"📥 Blank mode: Loading fresh data for {symbol} from historical storage", "INFO")
                # Try to load directly from historical storage first using BaseStep method
                market_data = self._load_market_data_from_historical_storage(
                    symbol=symbol,
                    exchange=config.get('exchange', 'binance'),
                    timeframe=config.get('timeframe', '1h'),
                    start_date=None,  # Will use execution mode defaults
                    end_date=None
                )
                
                if market_data is not None and len(market_data) > 0:
                    market_data_source = "historical_storage"
                    tprint(f"✅ Loaded {len(market_data):,} rows from historical storage", "SUCCESS")
                else:
                    tprint("⚠️ Failed to load from historical storage, falling back to artifacts", "WARNING")
                    # Fallback to artifact loading
                    market_data, market_data_source = self.ensure_market_data_in_pipeline_state(
                        config,
                        pipeline_state,
                        allow_config_override=True,
                    )
            else:
                # For other modes, use standard loading
                market_data, market_data_source = self.ensure_market_data_in_pipeline_state(
                    config,
                    pipeline_state,
                    allow_config_override=True,
                )

            tprint(
                f"✅ Using market data from {market_data_source}",
                "SUCCESS"
            )
            
            # Get timeframe for validation (execution_mode and symbol already defined above)
            timeframe = config.get('timeframe', '1h')
            
            # Check if we have enough data for blank mode (180 days)
            if execution_mode == 'blank' and market_data is not None:
                expected_samples_per_day = 24 if timeframe == '1h' else (24 * 4 if timeframe == '15m' else 24)
                expected_samples = 180 * expected_samples_per_day
                actual_samples = len(market_data)
                
                tprint(f"📊 Data validation: Expected ~{expected_samples:,} samples for 180 days of {timeframe} data", "INFO")
                tprint(f"📊 Data validation: Actual samples: {actual_samples:,}", "INFO")
                
                # If we have significantly less data than expected, warn the user
                if actual_samples < expected_samples * 0.5:  # Less than 50% of expected
                    tprint(f"⚠️ WARNING: Only {actual_samples:,} samples available (expected ~{expected_samples:,})", "WARNING")
                    tprint(f"⚠️ This may indicate:", "WARNING")
                    tprint(f"   • Cached data from wrong symbol (check if {symbol} data exists)", "WARNING")
                    tprint(f"   • Incomplete historical data", "WARNING")
                    tprint(f"   • Need to run klines_downloading_processing first", "WARNING")
            
            # Convert market data to DataFrame if it's not already
            if not isinstance(market_data, pd.DataFrame):  # type: ignore
                tprint("⚠️ Market data is not a DataFrame, attempting conversion", "WARNING")
                if isinstance(market_data, dict):
                    market_data = pd.DataFrame(market_data)  # type: ignore
                else:
                    tprint("❌ Cannot convert market data to DataFrame", "ERROR")
                    raise ValueError("Market data must be a pandas DataFrame or convertible dict")

            # NOTE: Data filtering for execution modes is now handled by the component itself
            # The component loads fresh data directly from historical storage in blank/light modes
            # This ensures we get the correct amount of data (180 days for blank, 20 days for light)
            # without relying on potentially stale data from pipeline_state

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
                        for _model_name, model_data in models.items():
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

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_regime_models_training_step():
    """Register the regime models training step."""
    from src.training.steps.base_step import step_registry  # type: ignore

    step_registry.register("regime_models_training", RegimeModelsTrainingStep)
    tprint("✅ Regime models training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_models_training_step()
