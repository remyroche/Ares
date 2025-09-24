"""
Regime Tagger

Tags existing market data with regime information from the hybrid NAS-TAS system.
This allows existing data to be processed with regime-aware training steps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
from pathlib import Path

from ..config.hybrid_regime_config import HybridRegimeConfig
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeResult

logger = logging.getLogger(__name__)


class RegimeTagger:
    """
    Regime Tagger

    Tags existing market data with regime information from the hybrid NAS-TAS system.
    This enables regime-aware processing of historical data without requiring
    full retraining.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize regime tagger."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hybrid regime detector
        self.hybrid_detector = None
        self._initialize_detector()

        # Tag columns to add
        self.tag_columns = config.get('tag_columns', [
            'regime_id', 'regime_confidence', 'economic_significance', 'financial_relevance'
        ])

        self.preserve_original = config.get('preserve_original_data', True)
        self.tag_historical = config.get('tag_historical_data', True)

        self.logger.info("✅ Regime Tagger initialized")
        self.logger.info(f"   Tag columns: {self.tag_columns}")
        self.logger.info(f"   Preserve original: {self.preserve_original}")

    def _initialize_detector(self):
        """Initialize the hybrid regime detector."""
        try:
            # Create configuration
            hybrid_config = HybridRegimeConfig(
                n_regimes=8,  # Default, will be updated based on data
                combination_strategy=self.config.get('combination_strategy', 'adaptive_fusion')
            )

            self.hybrid_detector = HybridNASTASRegimeDetector(hybrid_config)

            self.logger.info("✅ Hybrid detector initialized for tagging")

        except Exception as e:
            self.logger.warning(f"Could not initialize hybrid detector: {e}")
            self.hybrid_detector = None

    def tag_market_data(self,
                       market_data: pd.DataFrame,
                       output_path: Optional[str] = None,
                       batch_size: int = 1000) -> pd.DataFrame:
        """
        Tag market data with regime information.

        Args:
            market_data: Market data to tag
            output_path: Optional path to save tagged data
            batch_size: Batch size for processing large datasets

        Returns:
            Tagged market data
        """
        try:
            if self.hybrid_detector is None:
                self.logger.error("Hybrid detector not available for tagging")
                return self._add_empty_tags(market_data)

            self.logger.info(f"🏷️ Starting regime tagging for {len(market_data)} data points")

            # Process in batches for large datasets
            if len(market_data) > batch_size:
                tagged_data = self._process_in_batches(market_data, batch_size)
            else:
                tagged_data = self._tag_single_batch(market_data)

            # Save if requested
            if output_path:
                self._save_tagged_data(tagged_data, output_path)

            self.logger.info(f"✅ Regime tagging completed for {len(tagged_data)} data points")
            return tagged_data

        except Exception as e:
            self.logger.error(f"Regime tagging failed: {e}")
            return self._add_empty_tags(market_data)

    def _tag_single_batch(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Tag a single batch of market data."""
        try:
            # Detect regimes using hybrid detector
            regime_result = self.hybrid_detector.detect_regimes(market_data)

            if not regime_result.success:
                self.logger.warning("Regime detection failed, using fallback tagging")
                return self._add_empty_tags(market_data)

            # Create tagged data
            tagged_data = market_data.copy() if self.preserve_original else market_data

            # Add regime tags
            n_samples = len(tagged_data)
            n_regimes = len(regime_result.economic_significance_scores)

            # Regime ID
            regime_ids = regime_result.regime_predictions
            if len(regime_ids) != n_samples:
                regime_ids = self._fallback_regime_assignment(market_data, n_regimes)
            tagged_data['regime_id'] = regime_ids

            # Regime confidence (probability)
            if regime_result.regime_probabilities.size > 0:
                max_probs = np.max(regime_result.regime_probabilities, axis=1)
                if len(max_probs) == n_samples:
                    tagged_data['regime_confidence'] = max_probs
                else:
                    tagged_data['regime_confidence'] = 0.5  # Default confidence

            # Economic significance
            if len(regime_result.economic_significance_scores) > 0:
                # Map economic significance to each data point based on regime
                econ_significance = regime_result.economic_significance_scores[regime_ids]
                tagged_data['economic_significance'] = econ_significance
            else:
                tagged_data['economic_significance'] = 0.5

            # Financial relevance
            if len(regime_result.financial_relevance_scores) > 0:
                # Map financial relevance to each data point based on regime
                fin_relevance = regime_result.financial_relevance_scores[regime_ids]
                tagged_data['financial_relevance'] = fin_relevance
            else:
                tagged_data['financial_relevance'] = 0.5

            # Add additional metadata
            tagged_data['regime_detection_method'] = 'hybrid_nas_tas'
            tagged_data['regime_detection_timestamp'] = datetime.now()
            tagged_data['n_regimes_detected'] = n_regimes

            return tagged_data

        except Exception as e:
            self.logger.error(f"Single batch tagging failed: {e}")
            return self._add_empty_tags(market_data)

    def _process_in_batches(self, market_data: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """Process market data in batches."""
        try:
            self.logger.info(f"📊 Processing {len(market_data)} points in batches of {batch_size}")

            tagged_batches = []
            total_batches = (len(market_data) + batch_size - 1) // batch_size

            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(market_data))

                batch_data = market_data.iloc[start_idx:end_idx]
                self.logger.info(f"   Processing batch {i+1}/{total_batches} ({len(batch_data)} points)")

                tagged_batch = self._tag_single_batch(batch_data)
                tagged_batches.append(tagged_batch)

            # Combine all batches
            tagged_data = pd.concat(tagged_batches, ignore_index=True)
            return tagged_data

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return self._add_empty_tags(market_data)

    def _fallback_regime_assignment(self, market_data: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Fallback regime assignment when detection fails."""
        try:
            # Simple fallback based on volatility clusters
            returns = market_data['close'].pct_change().fillna(0).values
            volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0.01).values

            # Simple clustering based on volatility percentiles
            vol_percentiles = np.percentile(volatility, np.linspace(0, 100, n_regimes + 1))

            regime_ids = np.zeros(len(market_data), dtype=int)
            for i in range(n_regimes):
                mask = (volatility >= vol_percentiles[i]) & (volatility < vol_percentiles[i + 1])
                regime_ids[mask] = i

            return regime_ids

        except Exception as e:
            self.logger.warning(f"Fallback regime assignment failed: {e}")
            # Return sequential assignment
            n_samples = len(market_data)
            return np.arange(n_samples) % n_regimes

    def _add_empty_tags(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add empty/default tags when detection fails."""
        try:
            tagged_data = market_data.copy() if self.preserve_original else market_data

            # Add default tags
            n_samples = len(tagged_data)
            n_regimes = 8  # Default

            tagged_data['regime_id'] = np.arange(n_samples) % n_regimes
            tagged_data['regime_confidence'] = 0.5
            tagged_data['economic_significance'] = 0.5
            tagged_data['financial_relevance'] = 0.5
            tagged_data['regime_detection_method'] = 'fallback'
            tagged_data['regime_detection_timestamp'] = datetime.now()
            tagged_data['n_regimes_detected'] = n_regimes

            return tagged_data

        except Exception as e:
            self.logger.error(f"Empty tag addition failed: {e}")
            return market_data

    def _save_tagged_data(self, tagged_data: pd.DataFrame, output_path: str):
        """Save tagged data to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as CSV with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tagged_regime_data_{timestamp}.csv"
            full_path = output_path / filename

            tagged_data.to_csv(full_path, index=False)
            self.logger.info(f"💾 Tagged data saved to {full_path}")

            # Also save as Parquet for efficiency
            parquet_path = output_path / f"tagged_regime_data_{timestamp}.parquet"
            tagged_data.to_parquet(parquet_path, index=False)
            self.logger.info(f"💾 Tagged data saved to {parquet_path}")

        except Exception as e:
            self.logger.error(f"Failed to save tagged data: {e}")

    def tag_historical_directory(self,
                                data_directory: str,
                                output_directory: Optional[str] = None,
                                file_pattern: str = "*.csv") -> Dict[str, Any]:
        """
        Tag all historical data files in a directory.

        Args:
            data_directory: Directory containing historical data files
            output_directory: Directory to save tagged files
            file_pattern: Pattern to match data files

        Returns:
            Summary of tagging results
        """
        try:
            data_path = Path(data_directory)
            output_path = Path(output_directory) if output_directory else data_path / "tagged"

            output_path.mkdir(parents=True, exist_ok=True)

            # Find all data files
            data_files = list(data_path.glob(file_pattern))

            if not data_files:
                self.logger.warning(f"No files found matching pattern {file_pattern}")
                return {'success': False, 'message': 'No files found'}

            self.logger.info(f"📁 Found {len(data_files)} files to tag")

            results = {
                'total_files': len(data_files),
                'successful_tags': 0,
                'failed_tags': 0,
                'tagged_files': []
            }

            for data_file in data_files:
                try:
                    self.logger.info(f"🏷️ Tagging {data_file.name}")

                    # Load data
                    if data_file.suffix.lower() == '.parquet':
                        market_data = pd.read_parquet(data_file)
                    else:
                        market_data = pd.read_csv(data_file)

                    # Tag data
                    tagged_data = self.tag_market_data(market_data)

                    # Save tagged data
                    output_file = output_path / f"tagged_{data_file.name}"
                    self._save_tagged_data(tagged_data, str(output_file))

                    results['successful_tags'] += 1
                    results['tagged_files'].append(str(output_file))

                except Exception as e:
                    self.logger.error(f"Failed to tag {data_file.name}: {e}")
                    results['failed_tags'] += 1

            results['success'] = results['successful_tags'] > 0
            self.logger.info(f"✅ Historical tagging completed: {results['successful_tags']}/{results['total_files']} successful")

            return results

        except Exception as e:
            self.logger.error(f"Historical tagging failed: {e}")
            return {'success': False, 'message': str(e)}


def create_regime_tagger(config: Dict[str, Any]) -> RegimeTagger:
    """Create regime tagger."""
    return RegimeTagger(config)