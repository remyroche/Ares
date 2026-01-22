"""
Regime Ensemble Training Step.

This step trains ensemble models for regime classification using the comprehensive
RegimeEnsembleTrainingComponent implementation.
"""

import logging
from typing import Any, Dict

import pandas as pd

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
        super().__init__(step_name, use_versioned_artifacts=True)  # Enable HDF5 storage
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

            # Load regime probabilities from versioned artifacts (HDF5)
            # The ensemble needs regime probabilities as the data input
            tprint("📥 Loading regime probabilities from versioned artifacts", "INFO")
            
            # Enable versioned artifacts for loading
            self.use_versioned_artifacts = True
            self.set_context(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                timeframe=regime_timeframe,
                direction='long',
                model='regime'
            )
            
            regime_probs = self._get_artifact(
                'rolling_hmm_regime_probabilities',
                artifact_type='data',
                data_category='features'
            )
            
            if regime_probs is None:
                error_msg = (
                    "❌ No regime probabilities found in versioned artifacts!\n"
                    "   Please run rolling_hmm_regime_discovery first:\n"
                    f"   python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery --symbol {config.get('symbol')} --timeframe {regime_timeframe} --execution-mode blank"
                )
                tprint(error_msg, "ERROR")
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }
            
            tprint(f"✅ Loaded regime probabilities: {regime_probs.shape}", "SUCCESS")

            # Load Causal Surprise Events (Layer 2 Output) if available
            # Used to reinforce regime detection with causal signals (max_severity)
            tprint("📥 Loading Layer 2 Causal Events for feature reinforcement...", "INFO")
            layer2_events = self._get_artifact(
                'layer2_events', # or 'events', check logic later
                artifact_type='data',
                data_category='features'
            )
            # Fallback to search in versioned artifacts by pattern if named differently
            if layer2_events is None:
                # Try explicit path from pipeline_state if available or common names
                pass 

            # Integrate features
            market_data = regime_probs
            if layer2_events is not None and not layer2_events.empty:
                tprint(f"   ✅ Found Causal Events: {layer2_events.shape}", "SUCCESS")
                # Align and merge relevant columns
                # We want 'max_severity' (Weight) and 'causal_surprise' (Binary)
                cols_to_use = [c for c in ['max_severity', 'causal_surprise'] if c in layer2_events.columns]
                if cols_to_use:
                    aligned_events = layer2_events[cols_to_use].reindex(market_data.index).fillna(0.0) # Fill missing/non-events with 0
                    market_data = market_data.join(aligned_events)
                    tprint(f"   🔗 Injected Causal Features: {cols_to_use}", "INFO")
                else:
                    tprint("   ⚠️ Events found but missing 'max_severity' column.", "WARNING")
            else:
                tprint("   ⚠️ No Causal Events found. Training Regime Ensemble without surprise features.", "WARNING")

            
            # Use regime probabilities (and surprises) as the data input for the component

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



# Register the step
def register_regime_ensemble_training_step():
    """Register the regime ensemble training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
    tprint("✅ Regime ensemble training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_ensemble_training_step()
