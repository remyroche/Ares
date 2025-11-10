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

            # Log loaded regime probabilities with comprehensive preview
            from src.utils.tprint import tprint_data_preview
            tprint("=" * 80, "INFO")
            tprint("📥 DATA LOADED: Regime Probabilities from HMM Discovery", "INFO")
            tprint("=" * 80, "INFO")
            tprint_data_preview(
                regime_probs,
                name="Regime Probabilities",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint("=" * 80, "INFO")

            # Load OHLCV data for temporal analysis (returns calculation)
            tprint("📥 Loading OHLCV data from versioned artifacts for temporal analysis", "INFO")
            ohlcv_data = self._get_artifact(
                'ohlcv_data',
                artifact_type='data',
                data_category='raw'
            )

            # Combine regime probabilities with OHLCV data
            if ohlcv_data is not None:
                tprint(f"✅ Loaded OHLCV data: {ohlcv_data.shape}", "SUCCESS")

                # Align indices: regime_probs has integer index, OHLCV has datetime index
                # Use the last N rows of OHLCV to match regime_probs length
                n_regime_samples = len(regime_probs)
                ohlcv_aligned = ohlcv_data.tail(n_regime_samples).reset_index(drop=True)
                regime_probs_aligned = regime_probs.reset_index(drop=True)

                tprint(f"   Aligned OHLCV to last {n_regime_samples} samples", "INFO")
                tprint(f"   OHLCV close sample (aligned): {ohlcv_aligned['close'].head(3).tolist() if 'close' in ohlcv_aligned.columns else 'N/A'}", "INFO")

                # Merge on aligned integer index
                market_data = regime_probs_aligned.join(ohlcv_aligned[['open', 'high', 'low', 'close', 'volume']], how='left')
                tprint(f"✅ Combined regime probabilities with OHLCV: {market_data.shape}", "SUCCESS")
                tprint(f"   Combined close sample: {market_data['close'].head(3).tolist() if 'close' in market_data.columns else 'N/A'}", "INFO")
                tprint(f"   Close NaN count: {market_data['close'].isna().sum() if 'close' in market_data.columns else 'N/A'}", "INFO")

                # Log combined market data with comprehensive preview
                tprint("=" * 80, "INFO")
                tprint("🔗 DATA MODIFICATION: Combined Regime Probabilities with OHLCV", "INFO")
                tprint("=" * 80, "INFO")
                tprint_data_preview(
                    market_data,
                    name="Combined Regime + OHLCV Data",
                    max_rows=5,
                    max_cols=10,
                    show_dtypes=True,
                    show_shape=True
                )
                tprint("=" * 80, "INFO")
            else:
                tprint("⚠️ No OHLCV data found, temporal analysis will be skipped", "WARNING")
                market_data = regime_probs

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
