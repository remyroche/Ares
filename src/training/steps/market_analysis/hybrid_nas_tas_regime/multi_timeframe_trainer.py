"""
Multi-Timeframe Training System for Hybrid NAS-TAS Regime Detection

This module enables training NAS and TAS systems on multiple timeframes (1m, 5m)
while detecting regimes on the primary timeframe (15m). This allows the systems
to learn from higher-resolution data while making regime decisions at the
appropriate resolution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from src.utils.nas_tas.config.hybrid_regime_config import HybridRegimeConfig
from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
    HybridOrchestrator,
    HybridOrchestratorConfig,
)
from src.utils.nas_tas.shared_utils.search_strategies import (
    SearchStrategyConfig,
    SearchStrategyManager,
)
from src.utils.nas_tas.shared_utils.analysis_components import SharedClusteringUtilities

logger = logging.getLogger(__name__)


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe training."""
    primary_timeframe: str = "15m"  # Timeframe for regime detection
    training_timeframes: List[str] = None  # Timeframes for training (1m, 5m)
    enable_cross_timeframe_learning: bool = True
    transfer_learning_weight: float = 0.7
    regime_alignment_threshold: float = 0.8
    minimum_training_samples: int = 1000
    maximum_training_samples: int = 10000
    enable_regime_consistency_check: bool = True
    enable_timeframe_aggregation: bool = True

    def __post_init__(self):
        if self.training_timeframes is None:
            self.training_timeframes = ['1m', '5m']


@dataclass
class MultiTimeframeResult:
    """Result from multi-timeframe training."""
    success: bool
    primary_regime_results: Dict[str, Any]
    training_results: Dict[str, Dict[str, Any]]
    cross_timeframe_analysis: Dict[str, Any]
    regime_consistency_scores: Dict[str, float]
    transfer_learning_metrics: Dict[str, Any]
    execution_time: float = 0.0
    error_message: Optional[str] = None


class MultiTimeframeTrainer:
    """
    Multi-Timeframe Training System

    Enables training NAS and TAS systems on multiple timeframes while
    detecting regimes on the primary timeframe.
    """

    def __init__(self, config: MultiTimeframeConfig, hybrid_config: HybridRegimeConfig):
        """Initialize the multi-timeframe trainer."""
        self.config = config
        self.hybrid_config = hybrid_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.hybrid_orchestrator = None
        self.search_manager = None
        self.clustering_utilities = None

        self._initialize_components()
        self.logger.info("✅ Multi-Timeframe Trainer initialized")

    def _initialize_components(self):
        """Initialize trainer components."""
        try:
            # Initialize hybrid orchestrator
            orchestrator_config = HybridOrchestratorConfig(
                symbol="BTCUSDT",
                timeframe=self.config.primary_timeframe,
                start_date=None,
                end_date=None,
                use_standardized_features=True,
                feature_categories=['momentum', 'volatility', 'volume', 'trend'],
                significance_threshold=0.5,
                min_regime_duration=10,
                viability_threshold=0.5,
                minimum_regime_duration=5,
                max_iterations=100,
                use_bayesian_optimization=True,
                population_size=100,
                max_generations=50,
                use_nsga2=True,
                use_spea2=True,
                use_gpu_acceleration=True,
                memory_limit_gb=8.0,
                include_detailed_metrics=True,
                save_to_file=True
            )
            self.hybrid_orchestrator = HybridOrchestrator(orchestrator_config)

            # Initialize search strategy manager
            search_config = SearchStrategyConfig(
                max_iterations=50,
                n_initial_points=10,
                acquisition_function="expected_improvement",
                exploration_weight=0.1,
                convergence_threshold=1e-6,
                parallel_evaluations=1,
                random_state=42,
                use_bayesian_optimization=True,
                use_grid_optimization=True
            )
            self.search_manager = SearchStrategyManager(search_config)

            # Initialize clustering utilities
            self.clustering_utilities = SharedClusteringUtilities()

            self.logger.info("✅ All trainer components initialized")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def train_multi_timeframe(self,
                            market_data: Union[pd.DataFrame, np.ndarray],
                            timestamps: Optional[np.ndarray] = None,
                            training_timeframes: Optional[List[str]] = None) -> MultiTimeframeResult:
        """Train systems on multiple timeframes while detecting on primary timeframe."""
        start_time = time.time()
        self.logger.info("🚀 Starting multi-timeframe training...")

        try:
            # Use configured training timeframes if not specified
            if training_timeframes is None:
                training_timeframes = self.config.training_timeframes

            # Prepare data for different timeframes
            timeframe_data = {}
            for timeframe in training_timeframes + [self.config.primary_timeframe]:
                timeframe_data[timeframe] = self._prepare_timeframe_data(market_data, timeframe)

            # Step 1: Train on each timeframe individually
            training_results = {}
            for timeframe in training_timeframes:
                self.logger.info(f"🧠 Training on {timeframe} timeframe...")

                result = self._train_on_timeframe(
                    timeframe_data[timeframe],
                    timestamps,
                    timeframe
                )
                training_results[timeframe] = result

            # Step 2: Perform regime detection on primary timeframe
            self.logger.info(f"📊 Detecting regimes on {self.config.primary_timeframe} timeframe...")

            primary_result = self._detect_primary_regimes(
                timeframe_data[self.config.primary_timeframe],
                timestamps
            )

            # Step 3: Perform cross-timeframe analysis
            self.logger.info("🔄 Performing cross-timeframe analysis...")

            cross_timeframe_analysis = self._analyze_cross_timeframe(
                training_results,
                primary_result
            )

            # Step 4: Calculate regime consistency scores
            self.logger.info("📈 Calculating regime consistency scores...")

            regime_consistency_scores = self._calculate_regime_consistency(
                training_results,
                primary_result
            )

            # Step 5: Apply transfer learning if enabled
            transfer_learning_metrics = {}
            if self.config.enable_cross_timeframe_learning:
                self.logger.info("🔄 Applying cross-timeframe transfer learning...")

                transfer_learning_metrics = self._apply_transfer_learning(
                    training_results,
                    primary_result
                )

            execution_time = time.time() - start_time

            self.logger.info("✅ Multi-timeframe training completed successfully")
            self.logger.info(f"   Execution time: {execution_time:.2f}s")
            self.logger.info(f"   Training timeframes: {training_timeframes}")
            self.logger.info(f"   Primary timeframe: {self.config.primary_timeframe}")

            return MultiTimeframeResult(
                success=True,
                primary_regime_results=primary_result,
                training_results=training_results,
                cross_timeframe_analysis=cross_timeframe_analysis,
                regime_consistency_scores=regime_consistency_scores,
                transfer_learning_metrics=transfer_learning_metrics,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Multi-timeframe training failed: {e}")

            return MultiTimeframeResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _prepare_timeframe_data(self, market_data: Union[pd.DataFrame, np.ndarray],
                               timeframe: str) -> Union[pd.DataFrame, np.ndarray]:
        """Prepare data for specific timeframe."""
        try:
            if isinstance(market_data, np.ndarray):
                # For numpy arrays, resample based on timeframe
                if timeframe == '1m':
                    return market_data  # Assume input is already 1m
                elif timeframe == '5m':
                    if len(market_data) >= 5:
                        # Take every 5th sample for 5m data
                        indices = range(0, len(market_data), 5)
                        return market_data[indices]
                    else:
                        return market_data
                elif timeframe == '15m':
                    if len(market_data) >= 15:
                        # Take every 15th sample for 15m data
                        indices = range(0, len(market_data), 15)
                        return market_data[indices]
                    else:
                        return market_data
                else:
                    return market_data

            elif isinstance(market_data, pd.DataFrame):
                # For DataFrame, resample based on timeframe
                if 'timestamp' in market_data.columns:
                    market_data = market_data.set_index('timestamp')

                # Resample to the target timeframe
                if timeframe == '1m':
                    return market_data  # Assume input is already 1m
                else:
                    # Resample to higher timeframes
                    resampled = market_data.resample(timeframe).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    return resampled.reset_index()

            return market_data

        except Exception as e:
            tprint_error(f"⚠️ Timeframe data preparation failed: {e}")
            tprint_debug(f"Timeframe data preparation error context: {locals()}")
            tprint_error("CRITICAL: Timeframe data preparation is required for multi-timeframe training")
            tprint_error("Cannot proceed without proper timeframe data preparation")
            self.logger.error(f"⚠️ Timeframe data preparation failed: {e}")
            raise ValueError(f"Timeframe data preparation failed: {e}") from e

    def _train_on_timeframe(self, market_data: Union[pd.DataFrame, np.ndarray],
                           timestamps: Optional[np.ndarray],
                           timeframe: str) -> Dict[str, Any]:
        """Train systems on a specific timeframe."""
        try:
            self.logger.info(f"🎯 Training TAS/NAS systems on {timeframe} data...")

            # Use hybrid orchestrator to train on this timeframe
            if self.hybrid_orchestrator is not None:
                # Temporarily update orchestrator timeframe
                original_timeframe = self.hybrid_orchestrator.config.timeframe
                self.hybrid_orchestrator.config.timeframe = timeframe

                # Run orchestration on this timeframe
                result = self.hybrid_orchestrator.orchestrate_tas_nas_detection(
                    market_data,
                    timestamps,
                    timeframes=[timeframe]
                )

                # Restore original timeframe
                self.hybrid_orchestrator.config.timeframe = original_timeframe

                return {
                    'success': result.get('error', None) is None,
                    'tas_results': result.get('tas_results', {}),
                    'nas_results': result.get('nas_results', {}),
                    'execution_time': result.get('execution_time', 0.0),
                    'timeframe': timeframe,
                    'samples': len(market_data)
                }
            else:
                return {
                    'success': False,
                    'error': 'Hybrid orchestrator not available',
                    'timeframe': timeframe
                }

        except Exception as e:
            tprint_error(f"⚠️ Training failed for {timeframe}: {e}")
            tprint_debug(f"Training error context for {timeframe}: {locals()}")
            tprint_error(f"CRITICAL: Training on {timeframe} is required for multi-timeframe training")
            tprint_error(f"Cannot proceed without proper training on {timeframe}")
            self.logger.error(f"⚠️ Training failed for {timeframe}: {e}")
            raise ValueError(f"Training failed for {timeframe}: {e}") from e

    def _detect_primary_regimes(self, market_data: Union[pd.DataFrame, np.ndarray],
                               timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Detect regimes on primary timeframe."""
        try:
            self.logger.info(f"📊 Detecting regimes on primary timeframe {self.config.primary_timeframe}...")

            # Use hybrid orchestrator for primary regime detection
            if self.hybrid_orchestrator is not None:
                result = self.hybrid_orchestrator.orchestrate_tas_nas_detection(
                    market_data,
                    timestamps,
                    timeframes=[self.config.primary_timeframe]
                )

                return {
                    'success': result.get('error', None) is None,
                    'hybrid_results': result.get('hybrid_analysis', {}),
                    'tas_results': result.get('tas_results', {}).get(self.config.primary_timeframe, {}),
                    'nas_results': result.get('nas_results', {}).get(self.config.primary_timeframe, {}),
                    'execution_time': result.get('execution_time', 0.0),
                    'timeframe': self.config.primary_timeframe
                }
            else:
                return {
                    'success': False,
                    'error': 'Hybrid orchestrator not available'
                }

        except Exception as e:
            tprint_error(f"⚠️ Primary regime detection failed: {e}")
            tprint_debug(f"Primary regime detection error context: {locals()}")
            tprint_error("CRITICAL: Primary regime detection is required for multi-timeframe training")
            tprint_error("Cannot proceed without proper primary regime detection")
            self.logger.error(f"⚠️ Primary regime detection failed: {e}")
            raise ValueError(f"Primary regime detection failed: {e}") from e

    def _analyze_cross_timeframe(self, training_results: Dict[str, Dict[str, Any]],
                                primary_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-timeframe relationships."""
        try:
            self.logger.info("🔄 Analyzing cross-timeframe relationships...")

            # Extract regime predictions from training results
            training_regimes = {}
            for timeframe, result in training_results.items():
                if result.get('success', False):
                    # Get regime predictions from TAS or NAS
                    tas_result = result.get('tas_results', {}).get(timeframe, {})
                    nas_result = result.get('nas_results', {}).get(timeframe, {})

                    if tas_result.get('success', False):
                        training_regimes[timeframe] = {
                            'system': 'TAS',
                            'predictions': tas_result.get('regime_predictions', np.array([])),
                            'probabilities': tas_result.get('regime_probabilities', np.array([]))
                        }
                    elif nas_result.get('success', False):
                        training_regimes[timeframe] = {
                            'system': 'NAS',
                            'predictions': nas_result.get('regime_predictions', np.array([])),
                            'probabilities': nas_result.get('regime_probabilities', np.array([]))
                        }

            # Extract primary regime predictions
            primary_tas = primary_result.get('tas_results', {})
            primary_nas = primary_result.get('nas_results', {})

            primary_regimes = {
                'TAS': primary_tas.get('regime_predictions', np.array([])),
                'NAS': primary_nas.get('regime_predictions', np.array([]))
            }

            # Calculate alignment between training and primary regimes
            alignment_scores = {}
            for timeframe, training_regime in training_regimes.items():
                if len(training_regime['predictions']) > 0 and len(primary_regimes['TAS']) > 0:
                    # Simple alignment score based on correlation
                    min_len = min(len(training_regime['predictions']), len(primary_regimes['TAS']))
                    if min_len > 0:
                        correlation = np.corrcoef(
                            training_regime['predictions'][:min_len],
                            primary_regimes['TAS'][:min_len]
                        )[0, 1]
                        alignment_scores[timeframe] = abs(correlation)

            return {
                'training_regimes': training_regimes,
                'primary_regimes': primary_regimes,
                'alignment_scores': alignment_scores,
                'cross_timeframe_enabled': self.config.enable_cross_timeframe_learning
            }

        except Exception as e:
            tprint_error(f"⚠️ Cross-timeframe analysis failed: {e}")
            tprint_debug(f"Cross-timeframe analysis error context: {locals()}")
            tprint_error("CRITICAL: Cross-timeframe analysis is required for multi-timeframe training")
            tprint_error("Cannot proceed without proper cross-timeframe analysis")
            self.logger.error(f"⚠️ Cross-timeframe analysis failed: {e}")
            raise ValueError(f"Cross-timeframe analysis failed: {e}") from e

    def _calculate_regime_consistency(self, training_results: Dict[str, Dict[str, Any]],
                                    primary_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate regime consistency scores across timeframes."""
        try:
            self.logger.info("📈 Calculating regime consistency scores...")

            consistency_scores = {}

            # Get primary regime predictions
            primary_tas = primary_result.get('tas_results', {})
            primary_nas = primary_result.get('nas_results', {})

            primary_tas_preds = primary_tas.get('regime_predictions', np.array([]))
            primary_nas_preds = primary_nas.get('regime_predictions', np.array([]))

            if len(primary_tas_preds) == 0 and len(primary_nas_preds) == 0:
                return {'error': 'No primary regime predictions available'}

            # Calculate consistency with each training timeframe
            for timeframe, training_result in training_results.items():
                if not training_result.get('success', False):
                    consistency_scores[timeframe] = 0.0
                    continue

                # Get training regime predictions
                tas_result = training_result.get('tas_results', {}).get(timeframe, {})
                nas_result = training_result.get('nas_results', {}).get(timeframe, {})

                training_preds = None
                if tas_result.get('success', False):
                    training_preds = tas_result.get('regime_predictions', np.array([]))
                elif nas_result.get('success', False):
                    training_preds = nas_result.get('regime_predictions', np.array([]))

                if training_preds is not None and len(training_preds) > 0:
                    # Calculate consistency score
                    if len(primary_tas_preds) > 0:
                        # Use TAS as primary reference
                        min_len = min(len(training_preds), len(primary_tas_preds))
                        if min_len > 0:
                            correlation = np.corrcoef(
                                training_preds[:min_len],
                                primary_tas_preds[:min_len]
                            )[0, 1]
                            consistency_scores[timeframe] = abs(correlation)
                        else:
                            consistency_scores[timeframe] = 0.0
                    else:
                        consistency_scores[timeframe] = 0.0
                else:
                    consistency_scores[timeframe] = 0.0

            return consistency_scores

        except Exception as e:
            tprint_error(f"⚠️ Regime consistency calculation failed: {e}")
            tprint_debug(f"Regime consistency calculation error context: {locals()}")
            tprint_error("CRITICAL: Regime consistency calculation is required for multi-timeframe training")
            tprint_error("Cannot proceed without proper regime consistency calculation")
            self.logger.error(f"⚠️ Regime consistency calculation failed: {e}")
            raise ValueError(f"Regime consistency calculation failed: {e}") from e

    def _apply_transfer_learning(self, training_results: Dict[str, Dict[str, Any]],
                                primary_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transfer learning from training timeframes to primary timeframe."""
        try:
            self.logger.info("🔄 Applying transfer learning...")

            # Extract knowledge from training timeframes
            transferred_knowledge = {}
            total_weight = 0.0

            for timeframe, training_result in training_results.items():
                if training_result.get('success', False):
                    # Get consistency score for this timeframe
                    consistency_score = self.config.regime_consistency_scores.get(timeframe, 0.0)

                    if consistency_score >= self.config.regime_alignment_threshold:
                        # Transfer knowledge from this timeframe
                        weight = consistency_score * self.config.transfer_learning_weight

                        # Extract regime patterns from TAS/NAS results
                        tas_result = training_result.get('tas_results', {}).get(timeframe, {})
                        nas_result = training_result.get('nas_results', {}).get(timeframe, {})

                        if tas_result.get('success', False):
                            transferred_knowledge[timeframe] = {
                                'system': 'TAS',
                                'patterns': tas_result.get('regime_predictions', np.array([])),
                                'probabilities': tas_result.get('regime_probabilities', np.array([])),
                                'weight': weight,
                                'execution_time': tas_result.get('execution_time', 0.0)
                            }

                        if nas_result.get('success', False):
                            transferred_knowledge[timeframe] = {
                                'system': 'NAS',
                                'patterns': nas_result.get('regime_predictions', np.array([])),
                                'probabilities': nas_result.get('regime_probabilities', np.array([])),
                                'economic_scores': nas_result.get('economic_significance_scores', np.array([])),
                                'viability_scores': nas_result.get('trading_viability_scores', np.array([])),
                                'weight': weight,
                                'execution_time': nas_result.get('execution_time', 0.0)
                            }

                        total_weight += weight

            # Normalize weights
            if total_weight > 0:
                for timeframe in transferred_knowledge:
                    transferred_knowledge[timeframe]['weight'] /= total_weight

            return {
                'transferred_knowledge': transferred_knowledge,
                'total_transfer_weight': total_weight,
                'transfer_learning_applied': self.config.enable_cross_timeframe_learning,
                'alignment_threshold': self.config.regime_alignment_threshold
            }

        except Exception as e:
            tprint_error(f"⚠️ Transfer learning failed: {e}")
            tprint_debug(f"Transfer learning error context: {locals()}")
            tprint_error("CRITICAL: Transfer learning is required for multi-timeframe training")
            tprint_error("Cannot proceed without proper transfer learning")
            self.logger.error(f"⚠️ Transfer learning failed: {e}")
            raise ValueError(f"Transfer learning failed: {e}") from e


# Convenience functions
def create_multi_timeframe_trainer(hybrid_config: HybridRegimeConfig) -> MultiTimeframeTrainer:
    """Create a multi-timeframe trainer instance."""
    config = MultiTimeframeConfig()
    return MultiTimeframeTrainer(config, hybrid_config)


def quick_multi_timeframe_training(market_data: Union[pd.DataFrame, np.ndarray],
                                  primary_timeframe: str = "15m",
                                  training_timeframes: Optional[List[str]] = None) -> MultiTimeframeResult:
    """Quick multi-timeframe training with default settings."""
    if training_timeframes is None:
        training_timeframes = ['1m', '5m']

    config = MultiTimeframeConfig(
        primary_timeframe=primary_timeframe,
        training_timeframes=training_timeframes
    )

    hybrid_config = HybridRegimeConfig()
    trainer = MultiTimeframeTrainer(config, hybrid_config)

    return trainer.train_multi_timeframe(market_data)