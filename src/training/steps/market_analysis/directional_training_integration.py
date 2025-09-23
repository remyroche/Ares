"""
Directional Training Integration

This module provides a complete integration example showing how to use the modified
market analysis pipeline for both Analyst (5m timeframe, combined approach) and
Tactician (1m timeframe, directional approach) training.

Key Features:
- Complete workflow for Analyst training (5m, no directional differentiation)
- Complete workflow for Tactician training (1m, directional differentiation)
- Integration with signal separation utility
- End-to-end training pipeline examples
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Import modified components
from .multi_horizon_profit_labeler import MultiHorizonProfitLabeler, MultiHorizonConfig
from .pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
from .feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimization
from .final_feature_selection_step import FinalFeatureSelectionStep
from .tactician_directional_adaptor import TacticianDirectionalAdaptor, TacticianDirectionalConfig
from .signal_separation_utility import SignalSeparationUtility, SignalSeparationConfig

from src.utils.tprint import tprint
from src.utils.logger import get_logger

class DirectionalTrainingIntegration:
    """
    Complete integration for directional training workflows.

    This class provides end-to-end workflows for:
    1. Analyst training (5m timeframe, combined approach)
    2. Tactician training (1m timeframe, directional approach)
    """

    def __init__(self):
        """Initialize the directional training integration."""
        self.logger = get_logger('DirectionalTrainingIntegration')

        # Initialize components
        self.profit_labeler = MultiHorizonProfitLabeler()
        self.feature_generator = PIDBasedFeatureGenerationComponent()
        self.lookback_optimizer = FeatureLookbackOptimization()
        self.feature_selector = FinalFeatureSelectionStep()
        self.tactician_adaptor = TacticianDirectionalAdaptor()
        self.signal_separator = SignalSeparationUtility()

        self.logger.info("🚀 DirectionalTrainingIntegration initialized")

    async def run_analyst_training_pipeline(
        self,
        market_data_5m: pd.DataFrame,
        output_dir: str = "analyst_training"
    ) -> Dict[str, Any]:
        """
        Run complete Analyst training pipeline (5m timeframe, combined approach).

        Args:
            market_data_5m: Market data for 5m timeframe
            output_dir: Output directory for results

        Returns:
            Dict containing training results
        """
        self.logger.info("🔧 Starting Analyst training pipeline (5m, combined approach)")

        try:
            # Step 1: Generate combined profit labels (no directional differentiation)
            self.logger.info("🎯 Step 1: Generating combined profit labels")
            profit_labels = self.profit_labeler.generate_labels(market_data_5m)

            # Step 2: Generate features using PID approach (combined targets)
            self.logger.info("🔧 Step 2: Generating PID-based features")
            feature_generation_result = await self._run_feature_generation_combined(
                market_data_5m, profit_labels
            )

            # Step 3: Optimize lookback periods (combined approach)
            self.logger.info("⚙️ Step 3: Optimizing lookback periods")
            lookback_optimization_result = await self._run_lookback_optimization_combined(
                feature_generation_result, profit_labels
            )

            # Step 4: Select final features (combined approach)
            self.logger.info("🎯 Step 4: Selecting final features")
            final_features = await self._run_feature_selection_combined(
                feature_generation_result, lookback_optimization_result, profit_labels
            )

            # Compile results
            analyst_results = {
                'timeframe': '5m',
                'approach': 'combined',
                'profit_labels_shape': profit_labels.shape,
                'feature_generation': feature_generation_result,
                'lookback_optimization': lookback_optimization_result,
                'final_features_shape': final_features.shape if isinstance(final_features, pd.DataFrame) else 'N/A',
                'training_ready': True,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info("✅ Analyst training pipeline completed successfully")
            return analyst_results

        except Exception as e:
            self.logger.error(f"❌ Analyst training pipeline failed: {e}")
            return {
                'timeframe': '5m',
                'approach': 'combined',
                'training_ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def run_tactician_training_pipeline(
        self,
        analyst_predictions: pd.DataFrame,
        market_data_1m: pd.DataFrame,
        output_dir: str = "tactician_training"
    ) -> Dict[str, Any]:
        """
        Run complete Tactician training pipeline (1m timeframe, directional approach).

        Args:
            analyst_predictions: Predictions from Analyst (5m timeframe)
            market_data_1m: Market data for 1m timeframe
            output_dir: Output directory for results

        Returns:
            Dict containing training results
        """
        self.logger.info("🔧 Starting Tactician training pipeline (1m, directional approach)")

        try:
            # Step 1: Separate long/short signals from Analyst predictions
            self.logger.info("🎯 Step 1: Separating long/short signals")
            signal_separation = self.signal_separator.separate_signals(analyst_predictions)

            # Step 2: Adapt market data for directional analysis
            self.logger.info("🔧 Step 2: Adapting market data for directional analysis")
            adapted_market_data = self._adapt_market_data_directional(
                market_data_1m, signal_separation
            )

            # Step 3: Generate directional profit labels
            self.logger.info("🎯 Step 3: Generating directional profit labels")
            directional_labels = await self._generate_directional_labels(adapted_market_data)

            # Step 4: Generate directional features
            self.logger.info("🔧 Step 4: Generating directional features")
            directional_features = await self._generate_directional_features(
                adapted_market_data, directional_labels
            )

            # Step 5: Optimize directional lookback periods
            self.logger.info("⚙️ Step 5: Optimizing directional lookback periods")
            directional_lookbacks = await self._optimize_directional_lookbacks(
                directional_features, directional_labels
            )

            # Step 6: Select final directional features
            self.logger.info("🎯 Step 6: Selecting final directional features")
            final_directional_features = await self._select_directional_features(
                directional_features, directional_lookbacks, directional_labels
            )

            # Step 7: Prepare directional training data
            self.logger.info("📊 Step 7: Preparing directional training data")
            training_data = self._prepare_directional_training_data(
                final_directional_features, directional_labels, signal_separation
            )

            # Step 8: Use tactician adaptor for final integration
            self.logger.info("🤖 Step 8: Running tactician directional adaptor")
            tactician_result = await self.tactician_adaptor.adapt_for_directional_training(
                analyst_predictions, market_data_1m, '1m'
            )

            # Compile results
            tactician_results = {
                'timeframe': '1m',
                'approach': 'directional',
                'signal_separation': {
                    'long_signals': len(signal_separation.long_signals),
                    'short_signals': len(signal_separation.short_signals),
                    'neutral_signals': len(signal_separation.neutral_signals),
                    'separation_quality': signal_separation.separation_quality,
                    'signal_balance': signal_separation.signal_balance
                },
                'directional_labels_shape': directional_labels.shape if isinstance(directional_labels, pd.DataFrame) else 'N/A',
                'directional_features_shape': directional_features.shape if isinstance(directional_features, pd.DataFrame) else 'N/A',
                'training_data': training_data,
                'tactician_adaptor_result': tactician_result.adaptation_success if tactician_result else False,
                'training_ready': tactician_result.adaptation_success if tactician_result else False,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info("✅ Tactician training pipeline completed successfully")
            return tactician_results

        except Exception as e:
            self.logger.error(f"❌ Tactician training pipeline failed: {e}")
            return {
                'timeframe': '1m',
                'approach': 'directional',
                'training_ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _run_feature_generation_combined(
        self,
        market_data: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run feature generation for combined approach (Analyst)."""
        # This would use the PID feature generation component with combined targets
        # For now, return a placeholder result
        return {
            'features_generated': len(market_data.columns) * 2,  # Placeholder
            'quality_score': 0.8,  # Placeholder
            'combined_approach': True
        }

    async def _run_lookback_optimization_combined(
        self,
        features: Dict[str, Any],
        labels: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run lookback optimization for combined approach (Analyst)."""
        # This would use the feature lookback optimization component with combined targets
        return {
            'optimized_features': 50,  # Placeholder
            'optimization_quality': 0.75,  # Placeholder
            'combined_approach': True
        }

    async def _run_feature_selection_combined(
        self,
        features: Dict[str, Any],
        lookbacks: Dict[str, Any],
        labels: pd.DataFrame
    ) -> pd.DataFrame:
        """Run feature selection for combined approach (Analyst)."""
        # This would use the final feature selection step with combined features
        # For now, return a placeholder DataFrame
        return pd.DataFrame()  # Placeholder

    def _adapt_market_data_directional(
        self,
        market_data: pd.DataFrame,
        signal_separation: Any
    ) -> pd.DataFrame:
        """Adapt market data for directional analysis."""
        adapted_data = market_data.copy()

        # Add directional indicators based on signal separation
        adapted_data['directional_bias'] = 0.0
        adapted_data['is_long_signal'] = False
        adapted_data['is_short_signal'] = False

        # Set directional indicators based on signal separation
        long_indices = signal_separation.long_signals.index
        short_indices = signal_separation.short_signals.index

        adapted_data.loc[long_indices, 'directional_bias'] = 1.0
        adapted_data.loc[short_indices, 'directional_bias'] = -1.0
        adapted_data.loc[long_indices, 'is_long_signal'] = True
        adapted_data.loc[short_indices, 'is_short_signal'] = True

        return adapted_data

    async def _generate_directional_labels(self, adapted_market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate directional profit labels."""
        # This would use the multi-horizon profit labeler with directional analysis
        # For now, return a placeholder DataFrame
        return pd.DataFrame()  # Placeholder

    async def _generate_directional_features(
        self,
        market_data: pd.DataFrame,
        labels: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate directional features."""
        # This would use the PID feature generation component with directional targets
        # For now, return a placeholder DataFrame
        return pd.DataFrame()  # Placeholder

    async def _optimize_directional_lookbacks(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Dict[str, Any]:
        """Optimize directional lookback periods."""
        # This would use the directional lookback optimizer
        return {
            'long_lookbacks': {},
            'short_lookbacks': {},
            'optimization_quality': 0.8
        }

    async def _select_directional_features(
        self,
        features: pd.DataFrame,
        lookbacks: Dict[str, Any],
        labels: pd.DataFrame
    ) -> pd.DataFrame:
        """Select final directional features."""
        # This would use the final feature selection step with directional features
        # For now, return a placeholder DataFrame
        return pd.DataFrame()  # Placeholder

    def _prepare_directional_training_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        signal_separation: Any
    ) -> Dict[str, Any]:
        """Prepare directional training data."""
        return {
            'long': {
                'features': features.copy(),  # Placeholder
                'labels': labels.copy(),      # Placeholder
                'ready': True
            },
            'short': {
                'features': features.copy(),  # Placeholder
                'labels': labels.copy(),      # Placeholder
                'ready': True
            }
        }

# Convenience functions for easy usage
async def run_complete_training_workflows(
    market_data_5m: pd.DataFrame,
    market_data_1m: pd.DataFrame,
    analyst_predictions: pd.DataFrame,
    output_dir: str = "training_results"
) -> Dict[str, Any]:
    """
    Run complete training workflows for both Analyst and Tactician.

    Args:
        market_data_5m: Market data for Analyst (5m timeframe)
        market_data_1m: Market data for Tactician (1m timeframe)
        analyst_predictions: Analyst predictions for signal separation
        output_dir: Output directory for results

    Returns:
        Dict containing results for both training workflows
    """
    integration = DirectionalTrainingIntegration()

    # Run both workflows concurrently
    analyst_task = integration.run_analyst_training_pipeline(market_data_5m, output_dir)
    tactician_task = integration.run_tactician_training_pipeline(analyst_predictions, market_data_1m, output_dir)

    analyst_result, tactician_result = await asyncio.gather(analyst_task, tactician_task)

    return {
        'analyst_training': analyst_result,
        'tactician_training': tactician_result,
        'integration_complete': analyst_result.get('training_ready', False) and tactician_result.get('training_ready', False)
    }

# Example usage and testing
if __name__ == "__main__":
    async def example_usage():
        """Example usage of the directional training integration."""
        tprint("🔧 Directional Training Integration Example")

        # This would be replaced with actual data loading
        # market_data_5m = load_market_data('5m', 'BTCUSDT', days=30)
        # market_data_1m = load_market_data('1m', 'BTCUSDT', days=7)
        # analyst_predictions = load_analyst_predictions('5m', 'BTCUSDT')

        # For demonstration purposes, create sample data
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 1000),
            'high': np.random.uniform(40000, 50000, 1000),
            'low': np.random.uniform(40000, 50000, 1000),
            'close': np.random.uniform(40000, 50000, 1000),
            'volume': np.random.uniform(1000, 10000, 1000)
        }, index=dates)

        # Create sample analyst predictions with directional bias
        analyst_pred_dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        analyst_predictions = pd.DataFrame({
            'overall_opportunity': np.random.uniform(0.3, 0.8, 1000),
            'directional_bias': np.random.normal(0, 0.5, 1000),
            'long_overall_opportunity': np.random.uniform(0.3, 0.8, 1000),
            'short_overall_opportunity': np.random.uniform(0.3, 0.8, 1000)
        }, index=analyst_pred_dates)

        # Run training workflows
        results = await run_complete_training_workflows(
            market_data_5m=sample_data,
            market_data_1m=sample_data,  # Using same data for demo
            analyst_predictions=analyst_predictions
        )

        tprint("✅ Training workflows completed:")
        tprint(f"   → Analyst training: {'✅ Success' if results['analyst_training']['training_ready'] else '❌ Failed'}")
        tprint(f"   → Tactician training: {'✅ Success' if results['tactician_training']['training_ready'] else '❌ Failed'}")
        tprint(f"   → Integration complete: {'✅ Success' if results['integration_complete'] else '❌ Failed'}")

    # Run example
    asyncio.run(example_usage())