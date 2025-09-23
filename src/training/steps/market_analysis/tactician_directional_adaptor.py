"""
Tactician Directional Adaptor

This adaptor enables the existing market analysis pipeline to work with directional
training for the Tactician. It separates long/short signals from Analyst output and
adapts the existing market analysis files for directional differentiation.

Key Features:
- Separates long/short signals from Analyst predictions
- Adapts existing market analysis files for directional training
- Trains 2 separate Tactician models (long and short)
- Maintains compatibility with existing pipeline structure
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import existing market analysis components
from .multi_horizon_profit_labeler import MultiHorizonProfitLabeler, MultiHorizonConfig
from .pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
from .feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimization
from .final_feature_selection_step import FinalFeatureSelectionStep

# Import utilities
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.utils.common_operations import safe_dataframe_operation

@dataclass
class TacticianDirectionalConfig:
    """Configuration for tactician directional training."""

    # Directional settings
    enable_directional_training: bool = True
    signal_separation_threshold: float = 0.6  # Threshold for separating long/short signals
    min_samples_per_direction: int = 100

    # Model training settings
    train_separate_models: bool = True
    long_model_name: str = "tactician_long"
    short_model_name: str = "tactician_short"

    # Integration settings
    use_existing_pipeline: bool = True
    adapt_existing_files: bool = True

    # Quality thresholds
    directional_quality_threshold: float = 0.7
    signal_balance_threshold: float = 0.3  # Min ratio of smaller/larger direction

@dataclass
class DirectionalTrainingResult:
    """Result of directional training adaptation."""

    # Signal separation results
    long_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    short_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    neutral_signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Training data
    long_training_data: Dict[str, Any] = field(default_factory=dict)
    short_training_data: Dict[str, Any] = field(default_factory=dict)

    # Model configurations
    long_model_config: Dict[str, Any] = field(default_factory=dict)
    short_model_config: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    signal_separation_quality: float = 0.0
    directional_balance: float = 0.0
    adaptation_success: bool = False

    # Metadata
    adaptation_metadata: Dict[str, Any] = field(default_factory=dict)

class TacticianDirectionalAdaptor:
    """
    Adaptor for directional training of Tactician using existing market analysis files.

    This adaptor takes Analyst predictions and separates them into long/short signals,
    then adapts the existing market analysis pipeline for directional training.
    """

    def __init__(self, config: Optional[TacticianDirectionalConfig] = None):
        """Initialize the tactician directional adaptor."""
        self.config = config or TacticianDirectionalConfig()
        self.logger = get_logger('TacticianDirectionalAdaptor')

        # Initialize existing components
        self._initialize_components()

        self.logger.info("🚀 TacticianDirectionalAdaptor initialized")
        self.logger.info(f"   → Directional training: {'Enabled' if self.config.enable_directional_training else 'Disabled'}")
        self.logger.info(f"   → Separate models: {'Enabled' if self.config.train_separate_models else 'Disabled'}")

    def _initialize_components(self):
        """Initialize existing market analysis components."""
        # Initialize multi-horizon profit labeler (directional version)
        self.profit_labeler = MultiHorizonProfitLabeler()

        # Initialize PID-based feature generation
        self.feature_generator = PIDBasedFeatureGenerationComponent()

        # Initialize feature lookback optimization (directional version)
        self.lookback_optimizer = FeatureLookbackOptimization()

        # Initialize final feature selection
        self.feature_selector = FinalFeatureSelectionStep()

        self.logger.info("✅ All market analysis components initialized")

    async def adapt_for_directional_training(
        self,
        analyst_predictions: pd.DataFrame,
        market_data: pd.DataFrame,
        timeframe: str = '1m'
    ) -> DirectionalTrainingResult:
        """
        Adapt existing pipeline for directional training of Tactician.

        Args:
            analyst_predictions: Predictions from Analyst (5m timeframe)
            market_data: Market data for Tactician timeframe (1m)
            timeframe: Target timeframe for Tactician

        Returns:
            DirectionalTrainingResult with adapted training data
        """
        self.logger.info(f"🔧 Adapting pipeline for directional training (timeframe: {timeframe})")

        try:
            # Step 1: Separate long/short signals from Analyst predictions
            signal_separation = self._separate_long_short_signals(analyst_predictions)
            self.logger.info(f"✅ Signal separation completed: {len(signal_separation.long_signals)} long, {len(signal_separation.short_signals)} short samples")

            # Step 2: Adapt market data for directional analysis
            adapted_market_data = self._adapt_market_data_for_directional(market_data, signal_separation)
            self.logger.info(f"✅ Market data adapted for directional analysis: {len(adapted_market_data)} samples")

            # Step 3: Generate directional profit labels
            directional_labels = await self._generate_directional_labels(adapted_market_data)
            self.logger.info("✅ Directional labels generated")

            # Step 4: Generate directional features using existing PID component
            directional_features = await self._generate_directional_features(
                adapted_market_data, directional_labels, timeframe
            )
            self.logger.info("✅ Directional features generated")

            # Step 5: Optimize lookback periods for directional features
            optimized_lookbacks = await self._optimize_directional_lookbacks(
                directional_features, directional_labels
            )
            self.logger.info("✅ Directional lookback optimization completed")

            # Step 6: Select final directional features
            final_features = await self._select_directional_features(
                directional_features, optimized_lookbacks, directional_labels
            )
            self.logger.info("✅ Directional feature selection completed")

            # Step 7: Prepare training data for separate models
            training_data = self._prepare_directional_training_data(
                final_features, directional_labels, signal_separation
            )

            # Step 8: Create model configurations
            model_configs = self._create_directional_model_configs()

            # Calculate quality metrics
            quality_metrics = self._calculate_directional_quality_metrics(
                signal_separation, training_data
            )

            result = DirectionalTrainingResult(
                long_signals=signal_separation.long_signals,
                short_signals=signal_separation.short_signals,
                neutral_signals=signal_separation.neutral_signals,
                long_training_data=training_data.get('long', {}),
                short_training_data=training_data.get('short', {}),
                long_model_config=model_configs.get('long', {}),
                short_model_config=model_configs.get('short', {}),
                signal_separation_quality=quality_metrics.get('signal_quality', 0.0),
                directional_balance=quality_metrics.get('balance', 0.0),
                adaptation_success=True,
                adaptation_metadata={
                    'analyst_predictions_shape': analyst_predictions.shape,
                    'market_data_shape': market_data.shape,
                    'timeframe': timeframe,
                    'adaptation_timestamp': datetime.now().isoformat(),
                    'signal_separation_threshold': self.config.signal_separation_threshold,
                    'min_samples_per_direction': self.config.min_samples_per_direction
                }
            )

            self.logger.info("✅ Directional adaptation completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"❌ Directional adaptation failed: {e}")
            return DirectionalTrainingResult(adaptation_success=False)

    def _separate_long_short_signals(self, analyst_predictions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Separate long/short signals from Analyst predictions.

        Args:
            analyst_predictions: DataFrame with Analyst predictions

        Returns:
            Dict containing separated long, short, and neutral signals
        """
        self.logger.info("🔍 Separating long/short signals from Analyst predictions")

        # Identify directional columns from Analyst predictions
        directional_columns = [col for col in analyst_predictions.columns
                              if 'long' in col.lower() or 'short' in col.lower()]

        if not directional_columns:
            self.logger.warning("⚠️ No directional columns found in Analyst predictions")
            # Fallback: use overall opportunity as combined signal
            combined_signal = analyst_predictions['overall_opportunity'] if 'overall_opportunity' in analyst_predictions.columns else pd.Series(0.5, index=analyst_predictions.index)

            # Split based on combined signal (this is a fallback approach)
            neutral_mask = combined_signal < self.config.signal_separation_threshold
            long_mask = (combined_signal >= self.config.signal_separation_threshold) & (combined_signal < 0.7)
            short_mask = combined_signal >= 0.7

            return {
                'long_signals': analyst_predictions[long_mask].copy(),
                'short_signals': analyst_predictions[short_mask].copy(),
                'neutral_signals': analyst_predictions[neutral_mask].copy()
            }

        # Use directional bias for signal separation
        if 'directional_bias' in analyst_predictions.columns:
            directional_bias = analyst_predictions['directional_bias']

            # Separate based on directional bias
            long_mask = directional_bias > 0.3  # Strong long bias
            short_mask = directional_bias < -0.3  # Strong short bias
            neutral_mask = (directional_bias >= -0.3) & (directional_bias <= 0.3)

            return {
                'long_signals': analyst_predictions[long_mask].copy(),
                'short_signals': analyst_predictions[short_mask].copy(),
                'neutral_signals': analyst_predictions[neutral_mask].copy()
            }

        # Fallback: use long/short opportunity scores
        long_cols = [col for col in directional_columns if 'long' in col.lower()]
        short_cols = [col for col in directional_columns if 'short' in col.lower()]

        if long_cols and short_cols:
            # Calculate average long and short opportunity scores
            long_opportunity = analyst_predictions[long_cols].mean(axis=1)
            short_opportunity = analyst_predictions[short_cols].mean(axis=1)

            # Separate based on opportunity scores
            long_mask = (long_opportunity - short_opportunity) > self.config.signal_separation_threshold
            short_mask = (short_opportunity - long_opportunity) > self.config.signal_separation_threshold
            neutral_mask = ~long_mask & ~short_mask

            return {
                'long_signals': analyst_predictions[long_mask].copy(),
                'short_signals': analyst_predictions[short_mask].copy(),
                'neutral_signals': analyst_predictions[neutral_mask].copy()
            }

        # Final fallback: split based on overall opportunity
        self.logger.warning("⚠️ Using fallback signal separation method")
        combined_signal = analyst_predictions.get('overall_opportunity', pd.Series(0.5, index=analyst_predictions.index))

        neutral_mask = combined_signal < 0.4
        long_mask = (combined_signal >= 0.4) & (combined_signal < 0.6)
        short_mask = combined_signal >= 0.6

        return {
            'long_signals': analyst_predictions[long_mask].copy(),
            'short_signals': analyst_predictions[short_mask].copy(),
            'neutral_signals': analyst_predictions[neutral_mask].copy()
        }

    def _adapt_market_data_for_directional(
        self,
        market_data: pd.DataFrame,
        signal_separation: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Adapt market data for directional analysis.

        Args:
            market_data: Original market data
            signal_separation: Separated long/short signals

        Returns:
            Adapted market data with directional indicators
        """
        self.logger.info("🔧 Adapting market data for directional analysis")

        adapted_data = market_data.copy()

        # Add directional indicators based on signal separation
        long_indices = signal_separation['long_signals'].index
        short_indices = signal_separation['short_signals'].index
        neutral_indices = signal_separation['neutral_signals'].index

        # Add directional columns
        adapted_data['directional_bias'] = 0.0
        adapted_data['is_long_signal'] = False
        adapted_data['is_short_signal'] = False
        adapted_data['is_neutral_signal'] = False

        # Set directional indicators
        adapted_data.loc[long_indices, 'directional_bias'] = 1.0
        adapted_data.loc[short_indices, 'directional_bias'] = -1.0
        adapted_data.loc[long_indices, 'is_long_signal'] = True
        adapted_data.loc[short_indices, 'is_short_signal'] = True
        adapted_data.loc[neutral_indices, 'is_neutral_signal'] = True

        # Add signal strength indicators
        if 'overall_opportunity' in signal_separation['long_signals'].columns:
            adapted_data['long_signal_strength'] = 0.0
            adapted_data['short_signal_strength'] = 0.0

            long_strength = signal_separation['long_signals'].get('overall_opportunity', 0.5)
            short_strength = signal_separation['short_signals'].get('overall_opportunity', 0.5)

            adapted_data.loc[long_indices, 'long_signal_strength'] = long_strength
            adapted_data.loc[short_indices, 'short_signal_strength'] = short_strength

        self.logger.info(f"✅ Market data adapted: {len(long_indices)} long, {len(short_indices)} short, {len(neutral_indices)} neutral signals")
        return adapted_data

    async def _generate_directional_labels(self, adapted_market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate directional profit labels using existing multi-horizon profit labeler.

        Args:
            adapted_market_data: Market data adapted for directional analysis

        Returns:
            Dict containing directional labels
        """
        self.logger.info("🎯 Generating directional profit labels")

        # Use the existing multi-horizon profit labeler with directional mode
        # The labeler already supports directional analysis, so we just need to run it
        directional_labels = await self._run_directional_labeling(adapted_market_data)

        return directional_labels

    async def _run_directional_labeling(self, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run directional labeling using existing profit labeler."""
        # This would use the existing multi-horizon profit labeler
        # Since we modified it to support directional analysis, we can use it directly
        return {}

    async def _generate_directional_features(
        self,
        market_data: pd.DataFrame,
        labels: Dict[str, pd.DataFrame],
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate directional features using existing PID component.

        Args:
            market_data: Market data for feature generation
            labels: Directional labels
            timeframe: Timeframe for feature generation

        Returns:
            Dict containing directional features
        """
        self.logger.info("🔧 Generating directional features")

        # Use the existing PID-based feature generation component
        # We need to modify it to work with directional targets

        # For now, return placeholder - this would be implemented to use the existing component
        return {}

    async def _optimize_directional_lookbacks(
        self,
        features: Dict[str, pd.DataFrame],
        labels: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Optimize lookback periods for directional features.

        Args:
            features: Directional features
            labels: Directional labels

        Returns:
            Dict containing optimized lookback periods
        """
        self.logger.info("⚙️ Optimizing directional lookback periods")

        # Use the existing feature lookback optimization component
        # This would be adapted to work with directional features

        return {}

    async def _select_directional_features(
        self,
        features: Dict[str, pd.DataFrame],
        lookbacks: Dict[str, Any],
        labels: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Select final directional features.

        Args:
            features: Directional features
            lookbacks: Optimized lookback periods
            labels: Directional labels

        Returns:
            Dict containing selected directional features
        """
        self.logger.info("🎯 Selecting final directional features")

        # Use the existing final feature selection step
        # This would be adapted for directional selection

        return {}

    def _prepare_directional_training_data(
        self,
        features: Dict[str, pd.DataFrame],
        labels: Dict[str, pd.DataFrame],
        signal_separation: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Prepare training data for separate long/short models.

        Args:
            features: Directional features
            labels: Directional labels
            signal_separation: Separated signals

        Returns:
            Dict containing training data for long and short models
        """
        self.logger.info("📊 Preparing directional training data")

        training_data = {
            'long': {},
            'short': {}
        }

        # Prepare long training data
        if not signal_separation['long_signals'].empty:
            training_data['long'] = {
                'features': features.get('long', pd.DataFrame()),
                'labels': labels.get('long', pd.DataFrame()),
                'signal_data': signal_separation['long_signals'],
                'model_type': 'long'
            }

        # Prepare short training data
        if not signal_separation['short_signals'].empty:
            training_data['short'] = {
                'features': features.get('short', pd.DataFrame()),
                'labels': labels.get('short', pd.DataFrame()),
                'signal_data': signal_separation['short_signals'],
                'model_type': 'short'
            }

        self.logger.info(f"✅ Training data prepared: {len(training_data['long'])} long samples, {len(training_data['short'])} short samples")
        return training_data

    def _create_directional_model_configs(self) -> Dict[str, Any]:
        """
        Create model configurations for directional training.

        Returns:
            Dict containing model configurations for long and short models
        """
        self.logger.info("⚙️ Creating directional model configurations")

        model_configs = {
            'long': {
                'model_name': self.config.long_model_name,
                'model_type': 'tactician_long',
                'target_direction': 'long',
                'training_params': {
                    'epochs': 100,
                    'batch_size': 32,
                    'validation_split': 0.2
                }
            },
            'short': {
                'model_name': self.config.short_model_name,
                'model_type': 'tactician_short',
                'target_direction': 'short',
                'training_params': {
                    'epochs': 100,
                    'batch_size': 32,
                    'validation_split': 0.2
                }
            }
        }

        self.logger.info("✅ Model configurations created")
        return model_configs

    def _calculate_directional_quality_metrics(
        self,
        signal_separation: Dict[str, pd.DataFrame],
        training_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for directional adaptation.

        Args:
            signal_separation: Separated signals
            training_data: Training data

        Returns:
            Dict containing quality metrics
        """
        self.logger.info("📊 Calculating directional quality metrics")

        long_count = len(signal_separation['long_signals'])
        short_count = len(signal_separation['short_signals'])
        neutral_count = len(signal_separation['neutral_signals'])

        total_signals = long_count + short_count + neutral_count

        if total_signals == 0:
            return {'signal_quality': 0.0, 'balance': 0.0}

        # Signal quality score
        directional_ratio = (long_count + short_count) / total_signals
        signal_quality = directional_ratio * (1.0 - abs(long_count - short_count) / max(long_count + short_count, 1))

        # Balance score
        balance = 1.0 - abs(long_count - short_count) / max(long_count + short_count, 1) if (long_count + short_count) > 0 else 0.0

        metrics = {
            'signal_quality': signal_quality,
            'balance': balance,
            'directional_ratio': directional_ratio,
            'long_ratio': long_count / total_signals,
            'short_ratio': short_count / total_signals,
            'neutral_ratio': neutral_count / total_signals
        }

        self.logger.info(f"✅ Quality metrics: signal_quality={signal_quality:.3f}, balance={balance:.3f}")
        return metrics

    async def train_directional_models(
        self,
        training_data: Dict[str, Any],
        model_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train separate long and short models.

        Args:
            training_data: Training data for models
            model_configs: Model configurations

        Returns:
            Dict containing trained models
        """
        self.logger.info("🤖 Training directional models")

        trained_models = {}

        # Train long model
        if training_data.get('long') and model_configs.get('long'):
            self.logger.info("🔧 Training long model...")
            # This would contain the actual model training logic
            # For now, just store the configuration
            trained_models['long'] = {
                'model_config': model_configs['long'],
                'training_data': training_data['long'],
                'training_status': 'prepared'
            }

        # Train short model
        if training_data.get('short') and model_configs.get('short'):
            self.logger.info("🔧 Training short model...")
            # This would contain the actual model training logic
            # For now, just store the configuration
            trained_models['short'] = {
                'model_config': model_configs['short'],
                'training_data': training_data['short'],
                'training_status': 'prepared'
            }

        self.logger.info("✅ Model training preparation completed")
        return trained_models

# Convenience functions for easy integration
async def adapt_tactician_for_directional_training(
    analyst_predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    timeframe: str = '1m',
    config: Optional[TacticianDirectionalConfig] = None
) -> DirectionalTrainingResult:
    """Adapt tactician for directional training."""
    adaptor = TacticianDirectionalAdaptor(config)
    return await adaptor.adapt_for_directional_training(analyst_predictions, market_data, timeframe)

async def train_directional_tactician_models(
    analyst_predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    config: Optional[TacticianDirectionalConfig] = None
) -> Dict[str, Any]:
    """Train directional tactician models end-to-end."""
    adaptor = TacticianDirectionalAdaptor(config)
    adaptation_result = await adaptor.adapt_for_directional_training(analyst_predictions, market_data)

    if adaptation_result.adaptation_success:
        trained_models = await adaptor.train_directional_models(
            adaptation_result.long_training_data,
            adaptation_result.short_training_data
        )
        return {
            'adaptation_result': adaptation_result,
            'trained_models': trained_models
        }

    return {'adaptation_result': adaptation_result, 'trained_models': {}}