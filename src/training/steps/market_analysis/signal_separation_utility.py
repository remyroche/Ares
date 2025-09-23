"""
Signal Separation Utility

This utility provides functions to separate long/short signals from Analyst predictions
for directional training of the Tactician. It analyzes the output from the Analyst
and creates separate datasets for long and short signals.

Key Features:
- Extracts directional signals from Analyst predictions
- Handles various signal formats and confidence thresholds
- Provides quality metrics for signal separation
- Supports fallback methods for signal extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.utils.math_validation import safe_divide

@dataclass
class SignalSeparationConfig:
    """Configuration for signal separation."""

    # Signal extraction settings
    directional_bias_threshold: float = 0.3
    opportunity_threshold: float = 0.6
    confidence_threshold: float = 0.7

    # Quality settings
    min_signal_quality: float = 0.5
    balance_threshold: float = 0.3

    # Fallback settings
    use_fallback_separation: bool = True
    fallback_method: str = 'opportunity_split'  # 'opportunity_split', 'random', 'combined'

@dataclass
class SignalSeparationResult:
    """Result of signal separation."""

    # Separated signals
    long_signals: pd.DataFrame
    short_signals: pd.DataFrame
    neutral_signals: pd.DataFrame

    # Quality metrics
    separation_quality: float
    signal_balance: float
    confidence_scores: Dict[str, float]

    # Metadata
    separation_metadata: Dict[str, Any]

class SignalSeparationUtility:
    """
    Utility for separating long/short signals from Analyst predictions.

    This utility analyzes Analyst output and creates separate datasets
    for directional training of the Tactician.
    """

    def __init__(self, config: Optional[SignalSeparationConfig] = None):
        """Initialize the signal separation utility."""
        self.config = config or SignalSeparationConfig()
        self.logger = get_logger('SignalSeparationUtility')

    def separate_signals(
        self,
        analyst_predictions: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> SignalSeparationResult:
        """
        Separate long/short signals from Analyst predictions.

        Args:
            analyst_predictions: DataFrame with Analyst predictions
            market_data: Optional market data for context

        Returns:
            SignalSeparationResult with separated signals
        """
        self.logger.info("🔍 Starting signal separation from Analyst predictions")

        try:
            # Step 1: Identify directional signals
            directional_signals = self._identify_directional_signals(analyst_predictions)

            # Step 2: Apply separation thresholds
            separated_signals = self._apply_separation_thresholds(
                analyst_predictions, directional_signals
            )

            # Step 3: Calculate quality metrics
            quality_metrics = self._calculate_separation_quality(separated_signals)

            # Step 4: Create separation result
            result = SignalSeparationResult(
                long_signals=separated_signals['long'],
                short_signals=separated_signals['short'],
                neutral_signals=separated_signals['neutral'],
                separation_quality=quality_metrics['quality'],
                signal_balance=quality_metrics['balance'],
                confidence_scores=quality_metrics['confidence'],
                separation_metadata={
                    'total_samples': len(analyst_predictions),
                    'separation_method': 'directional_bias',
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config.__dict__
                }
            )

            self.logger.info(f"✅ Signal separation completed: {len(result.long_signals)} long, {len(result.short_signals)} short, {len(result.neutral_signals)} neutral")
            return result

        except Exception as e:
            self.logger.error(f"❌ Signal separation failed: {e}")
            # Return fallback result
            return self._create_fallback_separation(analyst_predictions)

    def _identify_directional_signals(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify directional signals in the predictions.

        Args:
            predictions: Analyst predictions DataFrame

        Returns:
            Dict containing identified signal types and columns
        """
        self.logger.info("🔍 Identifying directional signals")

        # Look for directional columns
        directional_info = {
            'long_columns': [],
            'short_columns': [],
            'bias_columns': [],
            'opportunity_columns': [],
            'confidence_columns': []
        }

        for col in predictions.columns:
            col_lower = col.lower()

            # Long columns
            if 'long' in col_lower and 'prob' in col_lower:
                directional_info['long_columns'].append(col)
            elif 'long' in col_lower and 'opportunity' in col_lower:
                directional_info['long_columns'].append(col)

            # Short columns
            elif 'short' in col_lower and 'prob' in col_lower:
                directional_info['short_columns'].append(col)
            elif 'short' in col_lower and 'opportunity' in col_lower:
                directional_info['short_columns'].append(col)

            # Bias columns
            elif 'bias' in col_lower or 'directional' in col_lower:
                directional_info['bias_columns'].append(col)

            # Opportunity columns
            elif 'opportunity' in col_lower or 'overall' in col_lower:
                directional_info['opportunity_columns'].append(col)

            # Confidence columns
            elif 'confidence' in col_lower or 'conf' in col_lower:
                directional_info['confidence_columns'].append(col)

        self.logger.info(f"✅ Found directional columns: {directional_info}")
        return directional_info

    def _apply_separation_thresholds(
        self,
        predictions: pd.DataFrame,
        directional_info: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply separation thresholds to create signal datasets.

        Args:
            predictions: Analyst predictions
            directional_info: Identified directional columns

        Returns:
            Dict containing separated signal datasets
        """
        self.logger.info("🔧 Applying separation thresholds")

        # Initialize signal masks
        long_mask = pd.Series(False, index=predictions.index)
        short_mask = pd.Series(False, index=predictions.index)
        neutral_mask = pd.Series(False, index=predictions.index)

        # Method 1: Use directional bias if available
        if directional_info['bias_columns']:
            bias_col = directional_info['bias_columns'][0]  # Use first bias column
            if bias_col in predictions.columns:
                directional_bias = predictions[bias_col]

                long_mask = directional_bias > self.config.directional_bias_threshold
                short_mask = directional_bias < -self.config.directional_bias_threshold
                neutral_mask = (directional_bias >= -self.config.directional_bias_threshold) & \
                              (directional_bias <= self.config.directional_bias_threshold)

                self.logger.info(f"✅ Used directional bias column: {bias_col}")
                return {
                    'long': predictions[long_mask].copy(),
                    'short': predictions[short_mask].copy(),
                    'neutral': predictions[neutral_mask].copy()
                }

        # Method 2: Use long/short opportunity comparison
        if directional_info['long_columns'] and directional_info['short_columns']:
            long_cols = directional_info['long_columns']
            short_cols = directional_info['short_columns']

            long_opportunity = predictions[long_cols].mean(axis=1)
            short_opportunity = predictions[short_cols].mean(axis=1)

            # Calculate directional strength
            directional_strength = long_opportunity - short_opportunity

            long_mask = directional_strength > self.config.opportunity_threshold
            short_mask = directional_strength < -self.config.opportunity_threshold
            neutral_mask = (directional_strength >= -self.config.opportunity_threshold) & \
                          (directional_strength <= self.config.opportunity_threshold)

            self.logger.info("✅ Used long/short opportunity comparison")
            return {
                'long': predictions[long_mask].copy(),
                'short': predictions[short_mask].copy(),
                'neutral': predictions[neutral_mask].copy()
            }

        # Method 3: Use overall opportunity with fallback split
        if directional_info['opportunity_columns']:
            opportunity_cols = directional_info['opportunity_columns']

            # Use the highest confidence opportunity column
            best_opportunity_col = None
            best_confidence = 0

            for col in opportunity_cols:
                if col in predictions.columns:
                    confidence = self._calculate_column_confidence(predictions[col])
                    if confidence > best_confidence:
                        best_opportunity_col = col
                        best_confidence = confidence

            if best_opportunity_col:
                opportunity_values = predictions[best_opportunity_col]

                # Split based on opportunity values
                high_opportunity = opportunity_values > 0.7
                medium_opportunity = (opportunity_values > 0.4) & (opportunity_values <= 0.7)
                low_opportunity = opportunity_values <= 0.4

                # For fallback, assign high to long, medium to neutral, low to short
                # This is a simplified approach - in practice, this would be more sophisticated
                long_mask = high_opportunity
                short_mask = low_opportunity
                neutral_mask = medium_opportunity

                self.logger.info(f"✅ Used opportunity-based separation: {best_opportunity_col}")
                return {
                    'long': predictions[long_mask].copy(),
                    'short': predictions[short_mask].copy(),
                    'neutral': predictions[neutral_mask].copy()
                }

        # Fallback method: Split based on index (for testing)
        self.logger.warning("⚠️ Using fallback signal separation")
        return self._fallback_signal_separation(predictions)

    def _calculate_column_confidence(self, series: pd.Series) -> float:
        """
        Calculate confidence score for a column.

        Args:
            series: Data series to analyze

        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Remove NaN values
            valid_values = series.dropna()

            if len(valid_values) == 0:
                return 0.0

            # Calculate variance-based confidence
            variance = valid_values.var()
            mean_val = valid_values.mean()

            # Higher variance and reasonable mean = higher confidence
            variance_confidence = min(1.0, variance * 10)  # Scale variance
            mean_confidence = 1.0 if 0.1 <= mean_val <= 0.9 else 0.5

            confidence = (variance_confidence + mean_confidence) / 2
            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating column confidence: {e}")
            return 0.5  # Default confidence

    def _fallback_signal_separation(self, predictions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Fallback signal separation method.

        Args:
            predictions: Analyst predictions

        Returns:
            Dict containing separated signals using fallback method
        """
        n_samples = len(predictions)

        if self.config.fallback_method == 'opportunity_split':
            # Split based on opportunity if available
            opportunity_cols = [col for col in predictions.columns if 'opportunity' in col.lower()]

            if opportunity_cols:
                opportunity_values = predictions[opportunity_cols[0]].fillna(0.5)
                long_mask = opportunity_values > 0.7
                short_mask = opportunity_values < 0.3
                neutral_mask = (opportunity_values >= 0.3) & (opportunity_values <= 0.7)

                return {
                    'long': predictions[long_mask].copy(),
                    'short': predictions[short_mask].copy(),
                    'neutral': predictions[neutral_mask].copy()
                }

        # Random split fallback
        np.random.seed(42)  # For reproducibility
        random_values = np.random.random(n_samples)

        long_mask = random_values > 0.7
        short_mask = random_values < 0.3
        neutral_mask = (random_values >= 0.3) & (random_values <= 0.7)

        self.logger.info("⚠️ Used random split fallback method")
        return {
            'long': predictions[long_mask].copy(),
            'short': predictions[short_mask].copy(),
            'neutral': predictions[neutral_mask].copy()
        }

    def _calculate_separation_quality(self, separated_signals: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate quality metrics for signal separation.

        Args:
            separated_signals: Separated signal datasets

        Returns:
            Dict containing quality metrics
        """
        self.logger.info("📊 Calculating separation quality metrics")

        long_count = len(separated_signals['long'])
        short_count = len(separated_signals['short'])
        neutral_count = len(separated_signals['neutral'])

        total_count = long_count + short_count + neutral_count

        if total_count == 0:
            return {'quality': 0.0, 'balance': 0.0, 'confidence': {'long': 0.0, 'short': 0.0}}

        # Calculate directional ratio (higher = better separation)
        directional_ratio = (long_count + short_count) / total_count

        # Calculate balance (closer to 0.5 = better balance)
        balance = 1.0 - abs(long_count - short_count) / max(long_count + short_count, 1)

        # Calculate confidence scores
        long_confidence = self._calculate_signal_confidence(separated_signals['long'])
        short_confidence = self._calculate_signal_confidence(separated_signals['short'])

        # Overall quality score
        quality = directional_ratio * balance * (long_confidence + short_confidence) / 2

        quality_metrics = {
            'quality': quality,
            'balance': balance,
            'directional_ratio': directional_ratio,
            'confidence': {
                'long': long_confidence,
                'short': short_confidence,
                'overall': (long_confidence + short_confidence) / 2
            },
            'counts': {
                'long': long_count,
                'short': short_count,
                'neutral': neutral_count,
                'total': total_count
            }
        }

        self.logger.info(f"✅ Quality metrics: quality={quality:.3f}, balance={balance:.3f}")
        return quality_metrics

    def _calculate_signal_confidence(self, signal_data: pd.DataFrame) -> float:
        """
        Calculate confidence score for separated signals.

        Args:
            signal_data: Separated signal dataset

        Returns:
            Confidence score between 0 and 1
        """
        if signal_data.empty:
            return 0.0

        try:
            # Look for confidence indicators
            confidence_cols = [col for col in signal_data.columns
                              if 'confidence' in col.lower() or 'conf' in col.lower()]

            if confidence_cols:
                # Use existing confidence column
                confidence_values = signal_data[confidence_cols[0]].dropna()
                if len(confidence_values) > 0:
                    return confidence_values.mean()

            # Calculate based on signal strength
            opportunity_cols = [col for col in signal_data.columns
                               if 'opportunity' in col.lower() or 'prob' in col.lower()]

            if opportunity_cols:
                opportunity_values = signal_data[opportunity_cols].mean(axis=1).dropna()
                if len(opportunity_values) > 0:
                    # Higher opportunity values = higher confidence
                    return opportunity_values.mean()

            # Default confidence
            return 0.5

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating signal confidence: {e}")
            return 0.5

    def _create_fallback_separation(self, predictions: pd.DataFrame) -> SignalSeparationResult:
        """
        Create a fallback separation result when normal separation fails.

        Args:
            predictions: Analyst predictions

        Returns:
            SignalSeparationResult with fallback separation
        """
        self.logger.warning("⚠️ Creating fallback separation result")

        # Create empty DataFrames for each category
        long_signals = pd.DataFrame(columns=predictions.columns)
        short_signals = pd.DataFrame(columns=predictions.columns)
        neutral_signals = predictions.copy()  # Put all in neutral

        return SignalSeparationResult(
            long_signals=long_signals,
            short_signals=short_signals,
            neutral_signals=neutral_signals,
            separation_quality=0.0,
            signal_balance=0.0,
            confidence_scores={'long': 0.0, 'short': 0.0, 'overall': 0.0},
            separation_metadata={
                'total_samples': len(predictions),
                'separation_method': 'fallback',
                'fallback_reason': 'signal_separation_failed',
                'timestamp': datetime.now().isoformat()
            }
        )

# Convenience functions
def separate_analyst_signals(
    analyst_predictions: pd.DataFrame,
    config: Optional[SignalSeparationConfig] = None
) -> SignalSeparationResult:
    """Separate long/short signals from Analyst predictions."""
    utility = SignalSeparationUtility(config)
    return utility.separate_signals(analyst_predictions)

def extract_directional_signals(
    analyst_predictions: pd.DataFrame,
    directional_bias_threshold: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract directional signals from Analyst predictions.

    Args:
        analyst_predictions: Analyst predictions DataFrame
        directional_bias_threshold: Threshold for directional bias

    Returns:
        Tuple of (long_signals, short_signals, neutral_signals)
    """
    config = SignalSeparationConfig(directional_bias_threshold=directional_bias_threshold)
    result = separate_analyst_signals(analyst_predictions, config)

    return result.long_signals, result.short_signals, result.neutral_signals