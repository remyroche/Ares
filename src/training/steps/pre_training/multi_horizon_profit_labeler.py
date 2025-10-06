"""
Multi-Horizon Profit Labeler Component for Pre-Training Pipeline.

This component integrates the VolatilityAwareMultiHorizonLabeler with regime data splitting
to create differentiated profit labels for different market regimes.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger

# Import the volatility-aware multi-horizon labeler
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig,
    LabelingResult
)

# Import base component
from .components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult


@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon profit labeling."""

    # Timeframe settings
    timeframe: str = "15m"
    base_period_minutes: float = 15.0

    # Volatility-aware labeling settings
    enable_volatility_normalization: bool = True
    enable_noise_gating: bool = True
    enable_quality_scoring: bool = True
    enable_multi_target_scheme: bool = True

    # Regime integration settings
    enable_regime_aware_labeling: bool = True
    regime_column: str = "regime_state"

    # Output settings
    min_data_points: int = 1000
    save_intermediate_results: bool = True
    generate_reports: bool = True

    # Quality thresholds
    min_auc_threshold: float = 0.55
    max_auc_std_threshold: float = 0.03
    min_psi_threshold: float = 0.1
    max_flip_rate_threshold: float = 0.15
    min_balance_threshold: float = 0.35
    max_balance_threshold: float = 0.65


class MultiHorizonProfitLabeler:
    """
    Multi-Horizon Profit Labeler that integrates volatility-aware labeling with regime data.

    This class creates differentiated profit labels for different market regimes,
    ensuring that the labeling process accounts for regime-specific behaviors.
    """

    def __init__(self, config: MultiHorizonConfig = None):
        """Initialize multi-horizon profit labeler."""
        self.config = config or MultiHorizonConfig()
        self.logger = logging.getLogger('MultiHorizonProfitLabeler')

        # Initialize the volatility-aware labeler
        self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(self._create_volatility_config())

        tprint_success("🚀 Multi-Horizon Profit Labeler initialized")
        tprint_info(f"   → Timeframe: {self.config.timeframe}")
        tprint_info(f"   → Regime-aware: {self.config.enable_regime_aware_labeling}")
        tprint_info(f"   → Volatility normalization: {self.config.enable_volatility_normalization}")

    def _create_volatility_config(self) -> VolatilityAwareConfig:
        """Create volatility-aware configuration from multi-horizon config."""
        return VolatilityAwareConfig(
            min_data_points=self.config.min_data_points,
            generate_reports=self.config.generate_reports,
            save_intermediate_results=self.config.save_intermediate_results,
            min_auc_threshold=self.config.min_auc_threshold,
            max_auc_std_threshold=self.config.max_auc_std_threshold
        )

    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "historical_data"
    ) -> Dict[str, Any]:
        """
        Execute multi-horizon profit labeling.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe for labeling (e.g., '15m')
            data_dir: Directory containing historical data

        Returns:
            Dictionary containing labeling results and metadata
        """
        try:
            tprint_info(f"🏷️ Starting multi-horizon profit labeling for {symbol} on {exchange}")
            tprint_info(f"⏰ Timeframe: {timeframe}")

            # Load market data
            market_data = await self._load_market_data(symbol, exchange, timeframe, data_dir)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {symbol} {timeframe}")

            # Apply regime-aware labeling if enabled
            if self.config.enable_regime_aware_labeling:
                labeling_result = await self._execute_regime_aware_labeling(market_data)
            else:
                labeling_result = self.volatility_labeler.generate_labels(market_data)

            # Generate comprehensive report
            report = self._generate_labeling_report(labeling_result, symbol, exchange, timeframe)

            # Create artifacts
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labels': labeling_result.labels,
                    'confidence_scores': labeling_result.confidence_scores,
                    'eligibility_masks': labeling_result.eligibility_masks,
                    'quality_scores': labeling_result.quality_scores,
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': self.config.enable_regime_aware_labeling,
                        'processing_time': labeling_result.processing_time,
                        'n_samples': labeling_result.n_samples,
                        'n_targets': labeling_result.n_targets,
                        'n_horizons': labeling_result.n_horizons
                    }
                },
                'labeling_report': report
            }

            tprint_success(f"✅ Multi-horizon labeling completed for {symbol}")
            tprint_info(f"   → Samples: {labeling_result.n_samples}")
            tprint_info(f"   → Targets: {labeling_result.n_targets}")
            tprint_info(f"   → Processing time: {labeling_result.processing_time:.2f}s")

            return artifacts

        except Exception as e:
            tprint_error(f"❌ Multi-horizon labeling failed: {e}")
            return {
                'multi_horizon_labeling_result': {},
                'labeling_report': {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    async def _load_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load market data for the specified symbol and timeframe."""
        try:
            # This would typically load from your data management system
            # For now, return a placeholder that indicates data loading is needed
            tprint_warning("⚠️ Market data loading not implemented - would need data management integration")
            return None

        except Exception as e:
            tprint_error(f"❌ Error loading market data: {e}")
            return None

    async def _execute_regime_aware_labeling(self, market_data: pd.DataFrame) -> LabelingResult:
        """
        Execute regime-aware labeling that creates differentiated labels for different regimes.

        Args:
            market_data: Market data with regime information

        Returns:
            LabelingResult with regime-differentiated labels
        """
        try:
            tprint_info("🎭 Executing regime-aware labeling")

            # Check if regime column exists
            if self.config.regime_column not in market_data.columns:
                tprint_warning(f"⚠️ Regime column '{self.config.regime_column}' not found, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

            # Get unique regimes
            regimes = market_data[self.config.regime_column].unique()
            tprint_info(f"📊 Found {len(regimes)} distinct regimes")

            # Create regime-specific labels
            regime_labels = {}
            regime_quality_scores = {}

            for regime in regimes:
                if pd.isna(regime):
                    continue

                tprint_info(f"🏷️ Processing regime {regime}")

                # Filter data for this regime
                regime_data = market_data[market_data[self.config.regime_column] == regime].copy()

                if len(regime_data) < self.config.min_data_points:
                    tprint_warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data)} samples")
                    continue

                # Generate labels for this regime
                regime_result = self.volatility_labeler.generate_labels(regime_data)

                if not regime_result.labels.empty:
                    # Add regime suffix to column names
                    regime_labels[regime] = regime_result.labels.add_suffix(f'_regime_{regime}')
                    regime_quality_scores.update({
                        f"{target}_regime_{regime}": quality_score
                        for target, quality_score in regime_result.quality_scores.items()
                    })

            # Combine regime-specific labels
            if regime_labels:
                combined_labels = pd.concat(regime_labels.values(), axis=1)
                combined_quality_scores = regime_quality_scores

                # Create combined result
                combined_result = LabelingResult(
                    labels=combined_labels,
                    confidence_scores=pd.DataFrame(index=combined_labels.index),  # Placeholder
                    eligibility_masks=pd.DataFrame(index=combined_labels.index),   # Placeholder
                    quality_scores=combined_quality_scores,
                    n_samples=len(combined_labels),
                    n_targets=len([col for col in combined_labels.columns if 'target' in col]),
                    processing_time=sum(
                        quality_score.processing_time
                        for quality_score in combined_quality_scores.values()
                    )
                )

                tprint_success(f"✅ Regime-aware labeling completed for {len(regime_labels)} regimes")
                return combined_result
            else:
                tprint_warning("⚠️ No valid regime-specific labels generated, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

        except Exception as e:
            tprint_error(f"❌ Regime-aware labeling failed: {e}")
            # Fall back to standard labeling
            return self.volatility_labeler.generate_labels(market_data)

    def _generate_labeling_report(
        self,
        labeling_result: LabelingResult,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate comprehensive labeling report."""
        try:
            report = {
                'status': 'completed',
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'processing_time': labeling_result.processing_time,
                'statistics': {
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                    'label_distribution': labeling_result.label_distribution
                },
                'quality_summary': {}
            }

            # Add quality scores summary
            if labeling_result.quality_scores:
                quality_summary = {}
                for target_name, quality_score in labeling_result.quality_scores.items():
                    quality_summary[target_name] = {
                        'overall_quality': quality_score.overall_quality,
                        'predictability': quality_score.predictability,
                        'stability': quality_score.stability,
                        'balance': quality_score.balance,
                        'auc_mean': quality_score.auc_mean,
                        'class_balance': quality_score.class_balance
                    }
                report['quality_summary'] = quality_summary

            return report

        except Exception as e:
            tprint_warning(f"⚠️ Error generating report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class MultiHorizonProfitLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Multi-Horizon Profit Labeler.

    This component integrates with the pre-training pipeline and handles
    regime-aware profit labeling with proper error handling and reporting.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the multi-horizon profit labeler component."""
        super().__init__(config)
        self.labeler = None

        # Create configuration from component config
        mh_config = MultiHorizonConfig()

        # Override with custom parameters if provided
        if config and config.custom_params:
            for key, value in config.custom_params.items():
                if hasattr(mh_config, key):
                    setattr(mh_config, key, value)

        self.labeler = MultiHorizonProfitLabeler(mh_config)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute multi-horizon profit labeling as a component.

        Args:
            data: Input data (typically None for this component)
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with labeling results
        """
        try:
            # Extract parameters from pipeline state
            symbol = pipeline_state.get('symbol', 'ETHUSDT')
            exchange = pipeline_state.get('exchange', 'binance')
            timeframe = pipeline_state.get('timeframe', '15m')
            data_dir = pipeline_state.get('data_dir', 'historical_data')

            # Execute labeling
            labeling_result = await self.labeler.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )

            return ComponentResult(
                success=True,
                artifacts=labeling_result,
                metadata={
                    'component_type': 'multi_horizon_profit_labeler',
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe
                }
            )

        except Exception as e:
            tprint_error(f"❌ Multi-horizon profit labeler component failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={
                    'multi_horizon_labeling_result': {},
                    'labeling_report': {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                },
                error_message=str(e),
                metadata={'component_type': 'multi_horizon_profit_labeler'}
            )