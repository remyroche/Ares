"""
Analyst-Tactician Training Integration

This module provides a comprehensive integration example showing how to use the
directional adapters for both Analyst and Tactician training:

1. Analyst Training (5m timeframe):
   - Combined signals without directional differentiation
   - Uses combined opportunity scores for unified training

2. Tactician Training (1m timeframe):
   - Separates long and short signals
   - Trains separate models for each direction
   - Uses optimized feature selection and labeling for each direction

This integration demonstrates the complete workflow for creating optimized
training data for both model types.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.tprint import tprint
from src.utils.math_validation import safe_divide, validate_finite

# Import the adapters we created
from .analyst_tactician_adapter import (
    AnalystTacticianConfig, TrainingMode,
    prepare_data_for_analyst, prepare_data_for_tactician
)

from .analyst_tactician_pid_adapter import (
    PIDAnalystTacticianConfig, PIDTrainingMode,
    generate_features_for_analyst, generate_features_for_tactician as generate_pid_features
)

from .analyst_tactician_labeler_adapter import (
    LabelerAnalystTacticianConfig, LabelerTrainingMode,
    generate_labels_for_analyst, generate_labels_for_tactician as generate_labeler_labels
)

from .analyst_tactician_feature_selection_adapter import (
    FeatureSelectionAnalystTacticianConfig, FeatureSelectionTrainingMode,
    select_features_for_analyst, select_features_for_tactician as select_final_features
)


@dataclass
class TrainingIntegrationConfig:
    """Configuration for the complete training integration."""
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    analyst_timeframe: str = "5m"
    tactician_timeframe: str = "1m"
    data_directory: str = "historical_data"

    # Feature targets
    analyst_combined_features: int = 80
    tactician_long_features: int = 50
    tactician_short_features: int = 50

    # Directional weighting
    long_weight: float = 0.5
    short_weight: float = 0.5

    # Optimization settings
    enable_optimization: bool = True
    save_intermediate_results: bool = True
    output_directory: str = "generated/training_data"

    # Quality settings
    min_feature_quality: float = 0.7
    max_correlation_threshold: float = 0.95
    enable_quality_validation: bool = True


class AnalystTacticianTrainingIntegration:
    """
    Complete integration for Analyst and Tactician training data preparation.

    This class orchestrates the entire workflow for creating optimized training data
    for both Analyst (5m, combined signals) and Tactician (1m, directional) models.
    """

    def __init__(self, config: Optional[TrainingIntegrationConfig] = None):
        self.config = config or TrainingIntegrationConfig()
        self.logger = get_logger('AnalystTacticianTrainingIntegration')

        # Track training results
        self.training_results = {
            'analyst': {},
            'tactician_long': {},
            'tactician_short': {}
        }

        self.logger.info("🚀 Analyst-Tactician Training Integration initialized")
        self.logger.info(f"   📊 Symbol: {self.config.symbol}")
        self.logger.info(f"   🏢 Exchange: {self.config.exchange}")
        self.logger.info(f"   ⏰ Analyst timeframe: {self.config.analyst_timeframe}")
        self.logger.info(f"   ⏰ Tactician timeframe: {self.config.tactician_timeframe}")

    async def run_complete_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline for both Analyst and Tactician models.

        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info("🚀 Starting complete training pipeline")

        try:
            # Step 1: Prepare Analyst training data (5m, combined)
            self.logger.info("📊 Step 1: Preparing Analyst training data (5m)")
            analyst_results = await self._prepare_analyst_training_data()

            # Step 2: Prepare Tactician training data (1m, directional)
            self.logger.info("📊 Step 2: Preparing Tactician training data (1m)")
            tactician_results = await self._prepare_tactician_training_data()

            # Step 3: Generate training reports
            self.logger.info("📊 Step 3: Generating training reports")
            await self._generate_training_reports()

            # Compile final results
            final_results = {
                'success': True,
                'analyst': analyst_results,
                'tactician': tactician_results,
                'pipeline_config': self.config.__dict__,
                'execution_time': datetime.now().isoformat(),
                'status_summary': self._generate_status_summary()
            }

            self.logger.info("✅ Complete training pipeline completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_config': self.config.__dict__,
                'execution_time': datetime.now().isoformat()
            }

    async def _prepare_analyst_training_data(self) -> Dict[str, Any]:
        """
        Prepare training data for Analyst model (5m timeframe, combined signals).

        Returns:
            Dictionary with Analyst training results
        """
        self.logger.info("🔄 Preparing Analyst training data")

        results = {
            'timeframe': self.config.analyst_timeframe,
            'training_mode': 'combined',
            'feature_count': 0,
            'sample_count': 0,
            'steps': []
        }

        try:
            # Load base market data
            market_data = await self._load_market_data(self.config.symbol, self.config.exchange,
                                                     self.config.analyst_timeframe)
            if market_data is None or market_data.empty:
                raise Exception(f"No market data available for {self.config.symbol} {self.config.analyst_timeframe}")

            results['sample_count'] = len(market_data)
            self.logger.info(f"📂 Loaded {len(market_data)} samples of market data")

            # Step 1: Generate features using PID-based generation
            step_result = await self._run_analyst_feature_generation(market_data)
            results['steps'].append(step_result)

            if step_result['success']:
                feature_data = step_result['data']

                # Step 2: Generate labels using multi-horizon profit labeler
                step_result = await self._run_analyst_labeling(feature_data)
                results['steps'].append(step_result)

                if step_result['success']:
                    labeled_data = step_result['data']

                    # Step 3: Select optimal features
                    step_result = await self._run_analyst_feature_selection(labeled_data)
                    results['steps'].append(step_result)

                    if step_result['success']:
                        final_data = step_result['data']
                        results['feature_count'] = len(final_data.columns)

                        # Save results if requested
                        if self.config.save_intermediate_results:
                            await self._save_training_data(final_data, 'analyst')

                        self.logger.info(f"✅ Analyst training data preparation completed: {len(final_data)} samples, {len(final_data.columns)} features")
                    else:
                        raise Exception("Analyst feature selection failed")
                else:
                    raise Exception("Analyst labeling failed")
            else:
                raise Exception("Analyst feature generation failed")

        except Exception as e:
            self.logger.error(f"❌ Analyst training data preparation failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

        results['success'] = True
        return results

    async def _prepare_tactician_training_data(self) -> Dict[str, Any]:
        """
        Prepare training data for Tactician models (1m timeframe, directional).

        Returns:
            Dictionary with Tactician training results
        """
        self.logger.info("🔄 Preparing Tactician training data")

        results = {
            'timeframe': self.config.tactician_timeframe,
            'training_modes': ['long', 'short'],
            'long_results': {'feature_count': 0, 'sample_count': 0},
            'short_results': {'feature_count': 0, 'sample_count': 0},
            'steps': []
        }

        try:
            # Load base market data for 1m timeframe
            market_data = await self._load_market_data(self.config.symbol, self.config.exchange,
                                                     self.config.tactician_timeframe)
            if market_data is None or market_data.empty:
                raise Exception(f"No market data available for {self.config.symbol} {self.config.tactician_timeframe}")

            results['sample_count'] = len(market_data)
            self.logger.info(f"📂 Loaded {len(market_data)} samples of 1m market data")

            # Step 1: Generate directional features
            step_result = await self._run_tactician_feature_generation(market_data)
            results['steps'].append(step_result)

            if step_result['success']:
                long_features = step_result['long_data']
                short_features = step_result['short_data']

                # Step 2: Generate directional labels
                step_result = await self._run_tactician_labeling(long_features, short_features)
                results['steps'].append(step_result)

                if step_result['success']:
                    long_labeled = step_result['long_data']
                    short_labeled = step_result['short_data']

                    # Step 3: Select optimal features for each direction
                    long_step_result = await self._run_tactician_feature_selection(long_labeled, 'long')
                    short_step_result = await self._run_tactician_feature_selection(short_labeled, 'short')

                    if long_step_result['success'] and short_step_result['success']:
                        results['long_results'] = {
                            'feature_count': len(long_step_result['data'].columns),
                            'sample_count': len(long_step_result['data']),
                            'success': True
                        }
                        results['short_results'] = {
                            'feature_count': len(short_step_result['data'].columns),
                            'sample_count': len(short_step_result['data']),
                            'success': True
                        }

                        # Save results if requested
                        if self.config.save_intermediate_results:
                            await self._save_training_data(long_step_result['data'], 'tactician_long')
                            await self._save_training_data(short_step_result['data'], 'tactician_short')

                        self.logger.info("✅ Tactician training data preparation completed")
                        self.logger.info(f"   📈 Long: {results['long_results']['feature_count']} features, {results['long_results']['sample_count']} samples")
                        self.logger.info(f"   📉 Short: {results['short_results']['feature_count']} features, {results['short_results']['sample_count']} samples")
                    else:
                        raise Exception(f"Tactician feature selection failed: long={long_step_result['success']}, short={short_step_result['success']}")
                else:
                    raise Exception("Tactician labeling failed")
            else:
                raise Exception("Tactician feature generation failed")

        except Exception as e:
            self.logger.error(f"❌ Tactician training data preparation failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

        results['success'] = True
        return results

    async def _run_analyst_feature_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run feature generation for Analyst training."""
        try:
            self.logger.info("🔧 Running Analyst feature generation")

            # Configure PID-based feature generation for Analyst
            pid_config = PIDAnalystTacticianConfig(
                training_mode=PIDTrainingMode.ANALYST,
                max_interaction_features=100,
                max_polynomial_features=50,
                max_cross_timeframe_features=50,
                long_weight=self.config.long_weight,
                short_weight=self.config.short_weight,
                enable_interaction_features=True,
                enable_polynomial_features=True,
                enable_cross_timeframe_features=True,
                enable_parallel_processing=True,
                enable_gpu_acceleration=True
            )

            # Generate combined features
            features_data = generate_features_for_analyst(data, pid_config)

            return {
                'step': 'feature_generation',
                'mode': 'analyst',
                'success': True,
                'data': features_data,
                'feature_count': len(features_data.columns),
                'sample_count': len(features_data)
            }

        except Exception as e:
            self.logger.error(f"❌ Analyst feature generation failed: {e}")
            return {
                'step': 'feature_generation',
                'mode': 'analyst',
                'success': False,
                'error': str(e)
            }

    async def _run_analyst_labeling(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run labeling for Analyst training."""
        try:
            self.logger.info("🔧 Running Analyst labeling")

            # Configure multi-horizon labeling for Analyst
            labeler_config = LabelerAnalystTacticianConfig(
                training_mode=LabelerTrainingMode.ANALYST,
                long_weight=self.config.long_weight,
                short_weight=self.config.short_weight,
                analyst_profit_targets={
                    'micro': 0.003, 'small': 0.005, 'medium': 0.007, 'good': 0.010
                },
                analyst_time_horizons={'immediate': 2, 'short': 4},
                enable_quality_scoring=True,
                enable_quality_validation=True
            )

            # Generate combined labels
            labeled_data = generate_labels_for_analyst(data, labeler_config)

            return {
                'step': 'labeling',
                'mode': 'analyst',
                'success': True,
                'data': labeled_data,
                'feature_count': len(labeled_data.columns),
                'sample_count': len(labeled_data)
            }

        except Exception as e:
            self.logger.error(f"❌ Analyst labeling failed: {e}")
            return {
                'step': 'labeling',
                'mode': 'analyst',
                'success': False,
                'error': str(e)
            }

    async def _run_analyst_feature_selection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run feature selection for Analyst training."""
        try:
            self.logger.info("🔧 Running Analyst feature selection")

            # Configure feature selection for Analyst
            selection_config = FeatureSelectionAnalystTacticianConfig(
                training_mode=FeatureSelectionTrainingMode.ANALYST,
                combined_features_target=self.config.analyst_combined_features,
                long_features_target=self.config.tactician_long_features,
                short_features_target=self.config.tactician_short_features,
                long_weight=self.config.long_weight,
                short_weight=self.config.short_weight,
                rf_n_estimators=100,
                cv_folds=5,
                enable_quality_validation=True,
                enable_outlier_detection=True,
                min_sample_quality_score=self.config.min_feature_quality
            )

            # Select optimal features
            selected_data = await select_features_for_analyst(
                self.config.symbol, self.config.exchange, self.config.data_directory, selection_config
            )

            return {
                'step': 'feature_selection',
                'mode': 'analyst',
                'success': True,
                'data': selected_data,
                'feature_count': len(selected_data.columns),
                'sample_count': len(selected_data)
            }

        except Exception as e:
            self.logger.error(f"❌ Analyst feature selection failed: {e}")
            return {
                'step': 'feature_selection',
                'mode': 'analyst',
                'success': False,
                'error': str(e)
            }

    async def _run_tactician_feature_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run feature generation for Tactician training."""
        try:
            self.logger.info("🔧 Running Tactician feature generation")

            # Configure PID-based feature generation for Tactician
            pid_config = PIDAnalystTacticianConfig(
                training_mode=PIDTrainingMode.TACTICIAN_LONG,  # Will be used for both directions
                max_interaction_features=100,
                max_polynomial_features=50,
                max_cross_timeframe_features=50,
                long_weight=self.config.long_weight,
                short_weight=self.config.short_weight,
                enable_interaction_features=True,
                enable_polynomial_features=True,
                enable_cross_timeframe_features=True,
                enable_parallel_processing=True,
                enable_gpu_acceleration=True
            )

            # Generate directional features
            long_data, short_data = generate_pid_features(data, 'both', pid_config)

            return {
                'step': 'feature_generation',
                'mode': 'tactician',
                'success': True,
                'long_data': long_data,
                'short_data': short_data,
                'long_feature_count': len(long_data.columns) if long_data is not None else 0,
                'short_feature_count': len(short_data.columns) if short_data is not None else 0,
                'sample_count': len(data)
            }

        except Exception as e:
            self.logger.error(f"❌ Tactician feature generation failed: {e}")
            return {
                'step': 'feature_generation',
                'mode': 'tactician',
                'success': False,
                'error': str(e)
            }

    async def _run_tactician_labeling(self, long_data: pd.DataFrame, short_data: pd.DataFrame) -> Dict[str, Any]:
        """Run labeling for Tactician training."""
        try:
            self.logger.info("🔧 Running Tactician labeling")

            # Configure multi-horizon labeling for Tactician
            labeler_config = LabelerAnalystTacticianConfig(
                training_mode=LabelerTrainingMode.TACTICIAN_LONG,  # Will be used for both directions
                long_weight=self.config.long_weight,
                short_weight=self.config.short_weight,
                tactician_profit_targets={
                    'micro': 0.001, 'small': 0.002, 'medium': 0.003, 'good': 0.005
                },
                tactician_time_horizons={'immediate': 5, 'short': 10},
                enable_quality_scoring=True,
                enable_quality_validation=True
            )

            # Generate directional labels
            long_labeled, short_labeled = generate_labeler_labels(long_data, 'both', labeler_config)

            return {
                'step': 'labeling',
                'mode': 'tactician',
                'success': True,
                'long_data': long_labeled,
                'short_data': short_labeled,
                'long_feature_count': len(long_labeled.columns) if long_labeled is not None else 0,
                'short_feature_count': len(short_labeled.columns) if short_labeled is not None else 0,
                'sample_count': len(long_data) if long_data is not None else len(short_data)
            }

        except Exception as e:
            self.logger.error(f"❌ Tactician labeling failed: {e}")
            return {
                'step': 'labeling',
                'mode': 'tactician',
                'success': False,
                'error': str(e)
            }

    async def _run_tactician_feature_selection(self, data: pd.DataFrame, direction: str) -> Dict[str, Any]:
        """Run feature selection for Tactician training in specific direction."""
        try:
            self.logger.info(f"🔧 Running Tactician {direction} feature selection")

            # Configure feature selection for Tactician direction
            selection_config = FeatureSelectionAnalystTacticianConfig(
                training_mode=getattr(FeatureSelectionTrainingMode, f'TACTICIAN_{direction.upper()}'),
                combined_features_target=self.config.analyst_combined_features,
                long_features_target=self.config.tactician_long_features,
                short_features_target=self.config.tactician_short_features,
                long_weight=self.config.long_weight,
                short_weight=self.config.short_weight,
                rf_n_estimators=100,
                cv_folds=5,
                enable_quality_validation=True,
                enable_outlier_detection=True,
                min_sample_quality_score=self.config.min_feature_quality
            )

            # Select optimal features for direction
            selected_data = await select_final_features(
                self.config.symbol, self.config.exchange, self.config.data_directory, direction, selection_config
            )

            return {
                'step': 'feature_selection',
                'mode': f'tactician_{direction}',
                'success': True,
                'data': selected_data,
                'feature_count': len(selected_data.columns),
                'sample_count': len(selected_data)
            }

        except Exception as e:
            self.logger.error(f"❌ Tactician {direction} feature selection failed: {e}")
            return {
                'step': 'feature_selection',
                'mode': f'tactician_{direction}',
                'success': False,
                'error': str(e)
            }

    async def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load market data for training."""
        try:
            from pathlib import Path

            data_file = Path(self.config.data_directory) / f"{exchange}/{symbol}/{timeframe}/data.parquet"
            if data_file.exists():
                data = pd.read_parquet(data_file)
                self.logger.info(f"✅ Loaded market data from {data_file}")
                return data
            else:
                self.logger.warning(f"⚠️ Market data file not found: {data_file}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load market data: {e}")
            return None

    async def _save_training_data(self, data: pd.DataFrame, dataset_name: str) -> None:
        """Save training data to disk."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{dataset_name}_training_data.parquet"
            data.to_parquet(output_file)

            self.logger.info(f"💾 Saved {dataset_name} training data to {output_file}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save training data: {e}")

    async def _generate_training_reports(self) -> None:
        """Generate training reports and summaries."""
        try:
            self.logger.info("📊 Generating training reports")

            # Create summary report
            report = {
                'training_integration_summary': {
                    'execution_time': datetime.now().isoformat(),
                    'config': self.config.__dict__,
                    'analyst_summary': self.training_results['analyst'],
                    'tactician_summary': {
                        'long': self.training_results['tactician_long'],
                        'short': self.training_results['tactician_short']
                    }
                }
            }

            # Save report
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            import json
            report_file = output_dir / "training_integration_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"📋 Training report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate training reports: {e}")

    def _generate_status_summary(self) -> Dict[str, Any]:
        """Generate status summary for the training pipeline."""
        return {
            'analyst_ready': self.training_results['analyst'].get('success', False),
            'tactician_long_ready': self.training_results['tactician_long'].get('success', False),
            'tactician_short_ready': self.training_results['tactician_short'].get('success', False),
            'total_features_analyst': self.training_results['analyst'].get('feature_count', 0),
            'total_features_long': self.training_results['tactician_long'].get('feature_count', 0),
            'total_features_short': self.training_results['tactician_short'].get('feature_count', 0),
            'samples_processed': self.training_results['analyst'].get('sample_count', 0)
        }


# Convenience function for easy integration
async def run_training_integration(config: Optional[TrainingIntegrationConfig] = None) -> Dict[str, Any]:
    """
    Run the complete Analyst-Tactician training integration.

    Args:
        config: Optional configuration for the integration

    Returns:
        Dictionary with training results and metrics
    """
    integration = AnalystTacticianTrainingIntegration(config)
    return await integration.run_complete_training_pipeline()


if __name__ == "__main__":
    # Example usage
    async def main():
        tprint("🧪 Testing Analyst-Tactician Training Integration")

        # Create configuration
        config = TrainingIntegrationConfig(
            symbol="BTCUSDT",
            exchange="binance",
            analyst_timeframe="5m",
            tactician_timeframe="1m",
            analyst_combined_features=80,
            tactician_long_features=50,
            tactician_short_features=50,
            save_intermediate_results=True,
            output_directory="generated/training_test"
        )

        # Run training integration
        results = await run_training_integration(config)

        if results['success']:
            tprint("✅ Training integration completed successfully!")
            tprint(f"   📊 Analyst: {results['analyst']['feature_count']} features, {results['analyst']['sample_count']} samples")
            tprint(f"   📈 Tactician Long: {results['tactician']['long_results']['feature_count']} features")
            tprint(f"   📉 Tactician Short: {results['tactician']['short_results']['feature_count']} features")
        else:
            tprint(f"❌ Training integration failed: {results.get('error', 'Unknown error')}")

    # Run the example
    asyncio.run(main())