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

        # Add comprehensive regime tags
        n_samples = len(tagged_data)
        n_regimes = len(regime_result.economic_significance_scores)

        # Regime ID
        regime_ids = regime_result.regime_predictions
        if len(regime_ids) != n_samples:
            regime_ids = self._fallback_regime_assignment(market_data, n_regimes)
        tagged_data['regime_id'] = regime_ids

        # Enhanced regime confidence with validation
        confidence_scores = self._calculate_confidence_scores(
            regime_result, market_data, regime_ids
        )
        tagged_data['regime_confidence'] = confidence_scores

        # Economic significance with validation
        economic_scores = self._calculate_economic_scores(
            regime_result, market_data, regime_ids
        )
        tagged_data['economic_significance'] = economic_scores

        # Financial relevance with validation
        financial_scores = self._calculate_financial_scores(
            regime_result, market_data, regime_ids
        )
        tagged_data['financial_relevance'] = financial_scores

        # Micro-regime detection
        micro_regime_ids = self._detect_micro_regimes(market_data, regime_ids)
        tagged_data['micro_regime_id'] = micro_regime_ids

        # Regime stability and transition analysis
        stability_scores = self._calculate_regime_stability(market_data, regime_ids)
        transition_scores = self._calculate_transition_scores(market_data, regime_ids)
        tagged_data['regime_stability'] = stability_scores
        tagged_data['transition_probability'] = transition_scores

        # Tag validation score
        validation_scores = self._validate_regime_tags(
            market_data, regime_ids, confidence_scores, economic_scores, financial_scores
        )
        tagged_data['tag_validation_score'] = validation_scores

        # Regime duration tracking
        duration_scores = self._calculate_regime_duration(market_data, regime_ids)
        tagged_data['regime_duration'] = duration_scores

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

    def _calculate_confidence_scores(self,
                                   regime_result,
                                   market_data: pd.DataFrame,
                                   regime_ids: np.ndarray) -> np.ndarray:
        """Calculate enhanced confidence scores with validation."""
        try:
            n_samples = len(market_data)

            # Base confidence from regime probabilities
            if regime_result.regime_probabilities.size > 0:
                base_confidence = np.max(regime_result.regime_probabilities, axis=1)
                if len(base_confidence) != n_samples:
                    base_confidence = np.full(n_samples, 0.5)
            else:
                base_confidence = np.full(n_samples, 0.5)

            # Economic confidence factor
            if len(regime_result.economic_significance_scores) > 0:
                economic_confidence = regime_result.economic_significance_scores[regime_ids]
                economic_confidence = np.where(economic_confidence > 0.7, 1.0,
                                             np.where(economic_confidence > 0.4, 0.7, 0.4))
            else:
                economic_confidence = np.full(n_samples, 0.7)

            # Financial confidence factor
            if len(regime_result.financial_relevance_scores) > 0:
                financial_confidence = regime_result.financial_relevance_scores[regime_ids]
                financial_confidence = np.where(financial_confidence > 0.7, 1.0,
                                              np.where(financial_confidence > 0.4, 0.7, 0.4))
            else:
                financial_confidence = np.full(n_samples, 0.7)

            # Data quality factor
            data_quality = self._calculate_data_quality_factor(market_data)

            # Combine confidence factors
            combined_confidence = (
                0.5 * base_confidence +
                0.2 * economic_confidence +
                0.2 * financial_confidence +
                0.1 * data_quality
            )

            # Apply confidence threshold
            confidence_threshold = self.config.get('confidence_threshold', 0.7)
            combined_confidence = np.where(combined_confidence < confidence_threshold,
                                         combined_confidence * 0.8, combined_confidence)

            return np.clip(combined_confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return np.full(len(market_data), 0.7)

    def _calculate_economic_scores(self,
                                 regime_result,
                                 market_data: pd.DataFrame,
                                 regime_ids: np.ndarray) -> np.ndarray:
        """Calculate enhanced economic significance scores."""
        try:
            n_samples = len(market_data)

            # Base economic scores from regime analysis
            if len(regime_result.economic_significance_scores) > 0:
                base_economic = regime_result.economic_significance_scores[regime_ids]
            else:
                base_economic = np.full(n_samples, 0.5)

            # Market condition factor
            market_condition_factor = self._calculate_market_condition_factor(market_data)

            # Volatility regime factor
            volatility_factor = self._calculate_volatility_regime_factor(market_data, regime_ids)

            # Trend strength factor
            trend_factor = self._calculate_trend_regime_factor(market_data, regime_ids)

            # Combine economic factors
            combined_economic = (
                0.4 * base_economic +
                0.3 * market_condition_factor +
                0.2 * volatility_factor +
                0.1 * trend_factor
            )

            return np.clip(combined_economic, 0.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Economic score calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _calculate_financial_scores(self,
                                  regime_result,
                                  market_data: pd.DataFrame,
                                  regime_ids: np.ndarray) -> np.ndarray:
        """Calculate enhanced financial relevance scores."""
        try:
            n_samples = len(market_data)

            # Base financial scores from regime analysis
            if len(regime_result.financial_relevance_scores) > 0:
                base_financial = regime_result.financial_relevance_scores[regime_ids]
            else:
                base_financial = np.full(n_samples, 0.5)

            # Trading viability factor
            trading_factor = self._calculate_trading_viability_factor(market_data, regime_ids)

            # Risk-return factor
            risk_factor = self._calculate_risk_return_factor(market_data, regime_ids)

            # Liquidity factor
            liquidity_factor = self._calculate_liquidity_factor(market_data, regime_ids)

            # Combine financial factors
            combined_financial = (
                0.4 * base_financial +
                0.3 * trading_factor +
                0.2 * risk_factor +
                0.1 * liquidity_factor
            )

            return np.clip(combined_financial, 0.0, 1.0)

        except Exception as e:
            self.logger.warning(f"Financial score calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _detect_micro_regimes(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Detect micro-regimes within macro regimes."""
        try:
            n_samples = len(market_data)

            # Simple micro-regime detection based on volatility clusters
            returns = market_data['close'].pct_change().fillna(0).values

            # Detect local volatility patterns
            rolling_volatility = pd.Series(np.abs(returns)).rolling(window=10, min_periods=5).std().fillna(0.01).values

            # Identify micro-regime breaks
            micro_regime_ids = np.zeros(n_samples, dtype=int)

            current_micro = 0
            for i in range(n_samples):
                if i > 0 and abs(rolling_volatility[i] - rolling_volatility[i-1]) > 0.02:
                    current_micro += 1
                micro_regime_ids[i] = current_micro

            # Limit number of micro-regimes
            max_micro_regimes = min(10, n_samples // 20)
            micro_regime_ids = micro_regime_ids % max_micro_regimes

            return micro_regime_ids

        except Exception as e:
            self.logger.warning(f"Micro-regime detection failed: {e}")
            return np.zeros(len(market_data), dtype=int)

    def _calculate_regime_stability(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate regime stability scores."""
        try:
            n_samples = len(market_data)

            # Calculate rolling regime consistency
            stability_scores = np.zeros(n_samples)

            for i in range(n_samples):
                start_idx = max(0, i - 10)
                end_idx = min(n_samples, i + 11)

                window_regimes = regime_ids[start_idx:end_idx]
                current_regime = regime_ids[i]

                # Calculate stability as fraction of same regime in window
                stability = np.mean(window_regimes == current_regime)
                stability_scores[i] = stability

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return np.full(len(market_data), 0.7)

    def _calculate_transition_scores(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability scores."""
        try:
            n_samples = len(market_data)

            # Calculate transition probabilities
            transition_scores = np.zeros(n_samples)

            for i in range(n_samples):
                if i == 0:
                    transition_scores[i] = 0.5  # No previous regime
                    continue

                current_regime = regime_ids[i]
                previous_regime = regime_ids[i-1]

                if current_regime == previous_regime:
                    # Same regime - high transition score (stability)
                    transition_scores[i] = 0.9
                else:
                    # Different regime - calculate transition probability
                    # Simplified: based on regime frequency
                    regime_counts = np.bincount(regime_ids, minlength=np.max(regime_ids) + 1)
                    total_samples = len(regime_ids)

                    if total_samples > 0:
                        current_freq = regime_counts[current_regime] / total_samples
                        transition_prob = min(current_freq * 2, 0.8)  # Scale frequency to probability
                        transition_scores[i] = transition_prob
                    else:
                        transition_scores[i] = 0.5

            return transition_scores

        except Exception as e:
            self.logger.warning(f"Transition score calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _validate_regime_tags(self,
                            market_data: pd.DataFrame,
                            regime_ids: np.ndarray,
                            confidence_scores: np.ndarray,
                            economic_scores: np.ndarray,
                            financial_scores: np.ndarray) -> np.ndarray:
        """Validate regime tags and calculate validation scores."""
        try:
            n_samples = len(market_data)

            # Validation based on consistency checks
            validation_scores = np.zeros(n_samples)

            for i in range(n_samples):
                validation_score = 1.0

                # Check confidence threshold
                confidence_threshold = self.config.get('confidence_threshold', 0.7)
                if confidence_scores[i] < confidence_threshold:
                    validation_score *= 0.8

                # Check economic significance threshold
                if economic_scores[i] < 0.4:
                    validation_score *= 0.9

                # Check financial relevance threshold
                if financial_scores[i] < 0.4:
                    validation_score *= 0.9

                # Check data quality
                if i < len(market_data) - 1:
                    price_change = abs(market_data.iloc[i]['close'] - market_data.iloc[i+1]['close']) / market_data.iloc[i]['close']
                    if price_change > 0.5:  # Extreme price movement
                        validation_score *= 0.7

                validation_scores[i] = validation_score

            return validation_scores

        except Exception as e:
            self.logger.warning(f"Tag validation failed: {e}")
            return np.full(len(market_data), 0.8)

    def _calculate_regime_duration(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate regime duration scores."""
        try:
            n_samples = len(market_data)

            # Calculate regime duration in periods
            duration_scores = np.zeros(n_samples)

            for i in range(n_samples):
                if i == 0:
                    duration_scores[i] = 1  # First period of regime
                    continue

                current_regime = regime_ids[i]
                previous_regime = regime_ids[i-1]

                if current_regime == previous_regime:
                    # Continuing regime - increment duration
                    duration_scores[i] = duration_scores[i-1] + 1
                else:
                    # New regime - reset duration
                    duration_scores[i] = 1

            return duration_scores

        except Exception as e:
            self.logger.warning(f"Regime duration calculation failed: {e}")
            return np.full(len(market_data), 1)

    def _calculate_data_quality_factor(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate data quality factor for confidence scoring."""
        try:
            n_samples = len(market_data)

            # Check for missing values
            missing_ratio = market_data.isnull().sum().sum() / (market_data.shape[0] * market_data.shape[1])
            missing_factor = max(0, 1 - missing_ratio * 2)  # Penalize missing data

            # Check for extreme values
            price_data = market_data['close']
            extreme_ratio = np.sum(np.abs(price_data.pct_change()) > 0.1) / len(price_data)
            extreme_factor = min(1.0, 1 - extreme_ratio)  # Penalize extreme movements

            # Check for volume consistency
            if 'volume' in market_data.columns:
                volume_data = market_data['volume']
                volume_missing = volume_data.isnull().sum() / len(volume_data)
                volume_factor = max(0, 1 - volume_missing)
            else:
                volume_factor = 0.8  # Neutral for missing volume

            # Combine factors
            quality_factor = np.full(n_samples, (missing_factor + extreme_factor + volume_factor) / 3)

            return quality_factor

        except Exception as e:
            self.logger.warning(f"Data quality factor calculation failed: {e}")
            return np.full(len(market_data), 0.8)

    def _calculate_market_condition_factor(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate market condition factor for economic scoring."""
        try:
            n_samples = len(market_data)

            # Market volatility factor
            returns = market_data['close'].pct_change().fillna(0).values
            rolling_volatility = pd.Series(np.abs(returns)).rolling(window=20, min_periods=10).mean().fillna(0.01).values

            # Normalize volatility to 0-1 scale
            max_volatility = np.max(rolling_volatility) if np.max(rolling_volatility) > 0 else 1
            volatility_factor = np.clip(rolling_volatility / max_volatility, 0, 1)

            # Market trend factor
            trend_factor = np.ones(n_samples) * 0.5  # Neutral default

            # Liquidity factor
            spreads = (market_data['high'] - market_data['low']) / market_data['close']
            liquidity_factor = 1 - np.clip(spreads.values, 0, 0.1) / 0.1  # Lower spread = higher liquidity

            # Combine factors
            market_factor = np.mean([volatility_factor, trend_factor, liquidity_factor], axis=0)

            return market_factor

        except Exception as e:
            self.logger.warning(f"Market condition factor calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _calculate_volatility_regime_factor(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate volatility regime factor."""
        try:
            n_samples = len(market_data)

            # Calculate rolling volatility by regime
            volatility_factor = np.zeros(n_samples)

            unique_regimes = set(regime_ids)

            for regime_id in unique_regimes:
                regime_mask = regime_ids == regime_id
                if np.sum(regime_mask) > 10:
                    regime_returns = market_data.loc[regime_mask, 'close'].pct_change().fillna(0).values
                    regime_volatility = np.std(regime_returns)

                    # Calculate relative volatility
                    overall_volatility = np.std(market_data['close'].pct_change().fillna(0).values)
                    relative_volatility = regime_volatility / (overall_volatility + 1e-8)

                    volatility_factor[regime_mask] = min(relative_volatility, 2.0) / 2.0  # Normalize to 0-1

            return volatility_factor

        except Exception as e:
            self.logger.warning(f"Volatility regime factor calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _calculate_trend_regime_factor(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate trend regime factor."""
        try:
            n_samples = len(market_data)

            # Calculate trend strength by regime
            trend_factor = np.zeros(n_samples)

            unique_regimes = set(regime_ids)

            for regime_id in unique_regimes:
                regime_mask = regime_ids == regime_id
                if np.sum(regime_mask) > 20:
                    regime_prices = market_data.loc[regime_mask, 'close'].values

                    # Simple linear trend R-squared
                    x = np.arange(len(regime_prices))
                    if len(x) > 1:
                        from scipy.stats import linregress
                        slope, intercept, r_value, p_value, std_err = linregress(x, regime_prices)
                        trend_strength = r_value ** 2

                        trend_factor[regime_mask] = min(trend_strength, 1.0)

            return trend_factor

        except Exception as e:
            self.logger.warning(f"Trend regime factor calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _calculate_trading_viability_factor(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate trading viability factor."""
        try:
            n_samples = len(market_data)

            # Calculate trading metrics by regime
            viability_factor = np.zeros(n_samples)

            unique_regimes = set(regime_ids)

            for regime_id in unique_regimes:
                regime_mask = regime_ids == regime_id
                if np.sum(regime_mask) > 10:
                    regime_returns = market_data.loc[regime_mask, 'close'].pct_change().fillna(0).values

                    if len(regime_returns) > 5:
                        # Win rate
                        win_rate = np.sum(regime_returns > 0) / len(regime_returns)

                        # Profit factor approximation
                        gains = np.sum(regime_returns[regime_returns > 0])
                        losses = abs(np.sum(regime_returns[regime_returns < 0]))
                        profit_factor = gains / (losses + 1e-8)

                        # Trading viability score
                        viability = (0.6 * win_rate + 0.4 * min(profit_factor / 2, 1.0))
                        viability_factor[regime_mask] = viability

            return viability_factor

        except Exception as e:
            self.logger.warning(f"Trading viability factor calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _calculate_risk_return_factor(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate risk-return factor."""
        try:
            n_samples = len(market_data)

            # Calculate risk-return metrics by regime
            risk_factor = np.zeros(n_samples)

            unique_regimes = set(regime_ids)

            for regime_id in unique_regimes:
                regime_mask = regime_ids == regime_id
                if np.sum(regime_mask) > 10:
                    regime_returns = market_data.loc[regime_mask, 'close'].pct_change().fillna(0).values

                    if len(regime_returns) > 5:
                        mean_return = np.mean(regime_returns)
                        volatility = np.std(regime_returns)

                        # Sharpe ratio approximation (assuming 0 risk-free rate)
                        sharpe_ratio = mean_return / (volatility + 1e-8)

                        # Risk-return score
                        risk_return_score = min(sharpe_ratio * 10, 1.0)  # Scale and cap
                        risk_factor[regime_mask] = risk_return_score

            return risk_factor

        except Exception as e:
            self.logger.warning(f"Risk-return factor calculation failed: {e}")
            return np.full(len(market_data), 0.5)

    def _calculate_liquidity_factor(self, market_data: pd.DataFrame, regime_ids: np.ndarray) -> np.ndarray:
        """Calculate liquidity factor."""
        try:
            n_samples = len(market_data)

            # Calculate liquidity metrics by regime
            liquidity_factor = np.zeros(n_samples)

            unique_regimes = set(regime_ids)

            for regime_id in unique_regimes:
                regime_mask = regime_ids == regime_id
                if np.sum(regime_mask) > 10:
                    regime_data = market_data[regime_mask]

                    # Spread-based liquidity
                    spreads = (regime_data['high'] - regime_data['low']) / regime_data['close']
                    avg_spread = np.mean(spreads)

                    # Volume-based liquidity (if available)
                    if 'volume' in regime_data.columns:
                        volume = regime_data['volume']
                        avg_volume = np.mean(volume)
                        volume_factor = min(avg_volume / 1000, 1.0)  # Scale volume
                    else:
                        volume_factor = 0.5

                    # Liquidity score (lower spread = higher liquidity)
                    spread_score = max(0, 1 - avg_spread * 10)  # Scale spread
                    liquidity_score = 0.7 * spread_score + 0.3 * volume_factor

                    liquidity_factor[regime_mask] = liquidity_score

            return liquidity_factor

        except Exception as e:
            self.logger.warning(f"Liquidity factor calculation failed: {e}")
            return np.full(len(market_data), 0.7)


def create_regime_tagger(config: Dict[str, Any]) -> RegimeTagger:
    """Create regime tagger."""
    return RegimeTagger(config)