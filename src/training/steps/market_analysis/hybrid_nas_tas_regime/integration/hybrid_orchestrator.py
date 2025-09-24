"""
Hybrid Regime Orchestrator

Main orchestrator for the hybrid NAS-TAS regime system that replaces HMM clustering.
This provides a complete replacement for the HMM clustering functionality with
enhanced economic and financial relevance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle

from ..config.hybrid_regime_config import HybridRegimeConfig
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeResult
from ..tagging.regime_tagger import RegimeTagger

logger = logging.getLogger(__name__)


class HybridRegimeOrchestrator:
    """
    Hybrid Regime Orchestrator

    Main orchestrator that replaces HMM clustering functionality with hybrid NAS-TAS
    regime detection. Provides complete regime discovery, detection, evaluation,
    and tagging capabilities with economic and financial relevance.
    """

    def __init__(self, config: HybridRegimeConfig):
        """Initialize hybrid regime orchestrator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.hybrid_detector = HybridNASTASRegimeDetector(config)
        self.regime_tagger = RegimeTagger(config.tagging_config)

        # State tracking
        self.last_detection_result = None
        self.detected_regimes = {}
        self.performance_history = []

        self.logger.info("✅ Hybrid Regime Orchestrator initialized")
        self.logger.info(f"   Combination strategy: {config.combination_strategy.value}")
        self.logger.info(f"   Economic evaluation: {config.economic_evaluation.get('enabled', True)}")
        self.logger.info(f"   Financial relevance: {config.financial_relevance.get('enabled', True)}")

    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      symbol: str = "UNKNOWN",
                      exchange: str = "unknown",
                      timeframe: str = "1h",
                      save_results: bool = True,
                      output_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect regimes using hybrid NAS-TAS approach.

        Args:
            market_data: Market data for regime detection
            timestamps: Optional timestamps
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            save_results: Whether to save results to disk
            output_directory: Directory to save results

        Returns:
            Complete detection results and metadata
        """
        try:
            self.logger.info(f"🚀 Starting hybrid regime detection for {symbol} on {exchange} ({timeframe})")

            # Detect regimes
            detection_result = self.hybrid_detector.detect_regimes(
                market_data,
                timestamps,
                validate_economic_significance=True,
                validate_financial_relevance=True
            )

            if not detection_result.success:
                raise ValueError("Hybrid regime detection failed")

            # Store result
            self.last_detection_result = detection_result

            # Create comprehensive results
            results = {
                'success': True,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'regime_data': self._extract_regime_data(detection_result),
                'economic_analysis': self._perform_economic_analysis(detection_result),
                'financial_analysis': self._perform_financial_analysis(detection_result),
                'performance_metrics': self._calculate_performance_metrics(detection_result),
                'metadata': detection_result.metadata
            }

            # Save results if requested
            if save_results:
                self._save_detection_results(results, output_directory)

            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'n_regimes': len(set(detection_result.regime_predictions)),
                'avg_economic_significance': np.mean(detection_result.economic_significance_scores),
                'avg_financial_relevance': np.mean(detection_result.financial_relevance_scores),
                'execution_time': detection_result.execution_time
            })

            self.logger.info(f"✅ Hybrid regime detection completed for {symbol}")
            self.logger.info(f"   Detected {len(set(detection_result.regime_predictions))} regimes")
            self.logger.info(f"   Average economic significance: {np.mean(detection_result.economic_significance_scores):.3f}")
            self.logger.info(f"   Average financial relevance: {np.mean(detection_result.financial_relevance_scores):.3f}")

            return results

        except Exception as e:
            self.logger.error(f"Hybrid regime detection failed: {e}")

            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def tag_existing_data(self,
                         data_directory: str,
                         output_directory: Optional[str] = None,
                         file_pattern: str = "*.csv") -> Dict[str, Any]:
        """
        Tag existing market data with regime information.

        Args:
            data_directory: Directory containing historical data
            output_directory: Directory to save tagged data
            file_pattern: Pattern to match data files

        Returns:
            Tagging results summary
        """
        try:
            self.logger.info(f"🏷️ Starting regime tagging for directory: {data_directory}")

            # Use regime tagger
            tagging_results = self.regime_tagger.tag_historical_directory(
                data_directory, output_directory, file_pattern
            )

            self.logger.info(f"✅ Regime tagging completed: {tagging_results}")
            return tagging_results

        except Exception as e:
            self.logger.error(f"Regime tagging failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def create_regime_aware_dataset(self,
                                   market_data: pd.DataFrame,
                                   regime_results: Dict[str, Any],
                                   split_by_regime: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Create regime-aware dataset from market data and regime results.

        Args:
            market_data: Original market data
            regime_results: Results from regime detection
            split_by_regime: Whether to split data by regime

        Returns:
            Regime-aware dataset (single DataFrame or dict of DataFrames)
        """
        try:
            if not regime_results.get('success', False):
                raise ValueError("Invalid regime results")

            # Create regime-aware data
            regime_data = market_data.copy()

            # Add regime information
            regime_predictions = regime_results['regime_data']['predictions']
            regime_probabilities = regime_results['regime_data']['probabilities']

            regime_data['regime_id'] = regime_predictions
            regime_data['regime_confidence'] = np.max(regime_probabilities, axis=1)
            regime_data['economic_significance'] = regime_results['economic_analysis']['significance_scores'][regime_predictions]
            regime_data['financial_relevance'] = regime_results['financial_analysis']['relevance_scores'][regime_predictions]

            # Add regime metadata
            regime_data['regime_detection_method'] = 'hybrid_nas_tas'
            regime_data['regime_detection_timestamp'] = regime_results['timestamp']

            if split_by_regime:
                # Split into separate DataFrames by regime
                regime_datasets = {}
                unique_regimes = set(regime_predictions)

                for regime_id in unique_regimes:
                    regime_mask = regime_data['regime_id'] == regime_id
                    regime_datasets[f'regime_{regime_id}'] = regime_data[regime_mask].copy()

                return regime_datasets
            else:
                # Return single unified dataset
                return regime_data

        except Exception as e:
            self.logger.error(f"Regime-aware dataset creation failed: {e}")
            return market_data

    def _extract_regime_data(self, detection_result: HybridRegimeResult) -> Dict[str, Any]:
        """Extract regime data from detection result."""
        try:
            return {
                'predictions': detection_result.regime_predictions,
                'probabilities': detection_result.regime_probabilities,
                'transition_matrix': detection_result.transition_probabilities,
                'stability_scores': detection_result.regime_stability_scores,
                'n_regimes': len(set(detection_result.regime_predictions)),
                'n_samples': len(detection_result.regime_predictions)
            }

        except Exception as e:
            self.logger.error(f"Regime data extraction failed: {e}")
            return {}

    def _perform_economic_analysis(self, detection_result: HybridRegimeResult) -> Dict[str, Any]:
        """Perform economic analysis of detected regimes."""
        try:
            return {
                'significance_scores': detection_result.economic_significance_scores,
                'average_significance': np.mean(detection_result.economic_significance_scores),
                'max_significance': np.max(detection_result.economic_significance_scores),
                'min_significance': np.min(detection_result.economic_significance_scores),
                'significant_regimes': np.sum(detection_result.economic_significance_scores > 0.7),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Economic analysis failed: {e}")
            return {'error': str(e)}

    def _perform_financial_analysis(self, detection_result: HybridRegimeResult) -> Dict[str, Any]:
        """Perform financial analysis of detected regimes."""
        try:
            return {
                'relevance_scores': detection_result.financial_relevance_scores,
                'average_relevance': np.mean(detection_result.financial_relevance_scores),
                'max_relevance': np.max(detection_result.financial_relevance_scores),
                'min_relevance': np.min(detection_result.financial_relevance_scores),
                'relevant_regimes': np.sum(detection_result.financial_relevance_scores > 0.6),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Financial analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_performance_metrics(self, detection_result: HybridRegimeResult) -> Dict[str, Any]:
        """Calculate performance metrics for regime detection."""
        try:
            # Clustering metrics
            clustering_metrics = detection_result.clustering_metrics

            # Regime quality metrics
            regime_sizes = np.bincount(detection_result.regime_predictions, minlength=len(detection_result.economic_significance_scores))
            regime_balance = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if np.mean(regime_sizes) > 0 else 0

            # Transition stability
            transition_matrix = detection_result.transition_probabilities
            transition_stability = np.mean(np.diag(transition_matrix))

            # Confidence metrics
            avg_confidence = np.mean(np.max(detection_result.regime_probabilities, axis=1))

            return {
                'clustering_quality': clustering_metrics,
                'regime_balance': regime_balance,
                'transition_stability': transition_stability,
                'average_confidence': avg_confidence,
                'execution_time': detection_result.execution_time,
                'total_regimes': len(detection_result.economic_significance_scores),
                'total_samples': len(detection_result.regime_predictions)
            }

        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}

    def _save_detection_results(self, results: Dict[str, Any], output_directory: Optional[str] = None):
        """Save detection results to disk."""
        try:
            if output_directory is None:
                output_directory = self.config.output_config.get('output_directory', 'generated/market_analysis/hybrid_regime')

            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save comprehensive results as JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = output_path / f"hybrid_regime_results_{timestamp}.json"

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Save regime predictions as CSV
            regime_data = results['regime_data']
            predictions_df = pd.DataFrame({
                'regime_id': regime_data['predictions'],
                'max_probability': np.max(regime_data['probabilities'], axis=1),
                'regime_confidence': np.max(regime_data['probabilities'], axis=1)
            })

            predictions_file = output_path / f"regime_predictions_{timestamp}.csv"
            predictions_df.to_csv(predictions_file, index=False)

            # Save performance metrics
            metrics_file = output_path / f"performance_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(results['performance_metrics'], f, indent=2, default=str)

            self.logger.info(f"💾 Detection results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save detection results: {e}")

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of last regime detection."""
        try:
            if self.last_detection_result is None:
                return {'error': 'No detection results available'}

            result = self.last_detection_result

            return {
                'n_regimes': len(set(result.regime_predictions)),
                'n_samples': len(result.regime_predictions),
                'avg_economic_significance': np.mean(result.economic_significance_scores),
                'avg_financial_relevance': np.mean(result.financial_relevance_scores),
                'avg_stability': np.mean(result.regime_stability_scores),
                'execution_time': result.execution_time,
                'timestamp': result.metadata.get('timestamp', 'unknown')
            }

        except Exception as e:
            return {'error': str(e)}

    def save_model(self, filepath: str):
        """Save orchestrator state for later use."""
        try:
            state = {
                'config': self.config.__dict__,
                'performance_history': self.performance_history,
                'last_detection_result': self.last_detection_result.metadata if self.last_detection_result else None
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"✅ Orchestrator state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str):
        """Load orchestrator state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Restore state
            self.performance_history = state.get('performance_history', [])
            self.logger.info(f"✅ Orchestrator state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")


def create_hybrid_orchestrator(config: Optional[HybridRegimeConfig] = None) -> HybridRegimeOrchestrator:
    """Create hybrid regime orchestrator."""
    if config is None:
        config = HybridRegimeConfig()
    return HybridRegimeOrchestrator(config)


def quick_hybrid_detection(market_data: Union[pd.DataFrame, np.ndarray],
                          symbol: str = "UNKNOWN",
                          exchange: str = "unknown",
                          timeframe: str = "1h") -> Dict[str, Any]:
    """Quick hybrid regime detection with default settings."""
    config = HybridRegimeConfig()
    orchestrator = HybridRegimeOrchestrator(config)
    return orchestrator.detect_regimes(market_data, symbol=symbol, exchange=exchange, timeframe=timeframe)