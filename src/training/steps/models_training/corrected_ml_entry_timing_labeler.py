"""
Corrected ML-Based Entry Timing Labeler for Tactician

This module implements ML-based entry timing labeling that:
1. Creates labels based on peak/bottom detection (not timing)
2. Positive points for entries at optimal peaks/bottoms
3. Negative points for entries too early (adversarial movement) or too late (missed opportunity)
4. Trains ML models to predict entry quality based on this objective

The approach follows this workflow:
Peak/Bottom Detection → Entry Quality Scoring → ML Model Training → Refined Entry Prediction
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks, argrelextrema
import joblib
import warnings

try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_timer
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import utilities: {e}")

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import VectorBT Rolling Optimizer for enhanced performance
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_apply, optimized_rolling_corr, optimized_rolling_cov
    )
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT Rolling Optimizer not available: {e}")
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False

except ImportError:

    cp = None
    UTILS_AVAILABLE = False

@dataclass
class CorrectedMLEntryTimingConfig:
    """Configuration for corrected ML-based entry timing labeling."""
    # Peak/bottom detection
    peak_detection_window: int = 20  # Window for peak/bottom detection
    min_peak_prominence: float = 0.5  # Minimum prominence for peak detection
    min_peak_distance: int = 5  # Minimum distance between peaks

    # Entry quality scoring
    max_adverse_movement_pct: float = 2.0  # Max adverse movement before negative scoring
    opportunity_capture_weight: float = 0.6  # Weight for opportunity capture
    adverse_movement_weight: float = 0.4  # Weight for avoiding adverse movement

    # ML model configuration
    models: List[str] = field(default_factory=lambda: ['random_forest', 'gradient_boosting', 'ridge'])
    test_size: float = 0.2
    random_state: int = 42

    # Feature engineering
    feature_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    technical_indicators: bool = True
    price_action_features: bool = True
    volume_features: bool = True
    volatility_features: bool = True

    # Training configuration
    max_iterations: int = 3
    min_improvement_threshold: float = 0.01
    cross_validation_folds: int = 5

    # Quality thresholds
    min_r2_score: float = 0.3
    min_correlation: float = 0.5

class CorrectedMLEntryTimingLabeler:
    """Corrected ML-based entry timing labeler focused on peak/bottom detection."""

    def __init__(self, config: CorrectedMLEntryTimingConfig):
        self.config = config
        self.logger = system_logger.getChild('CorrectedMLEntryTimingLabeler')
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        self.peak_bottom_data = {}

        # Initialize VectorBT Rolling Optimizer for enhanced performance
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,  # Conservative for ML labeling
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=500,  # Smaller chunks for ML operations
                fast_fail=False,  # Use fallbacks for robustness
                enable_logging=True
            )
            tprint_success("✅ VectorBT Rolling Optimizer initialized for ML Entry Timing Labeler")
        else:
            self.vectorbt_optimizer = None
            tprint_warning("⚠️ VectorBT Rolling Optimizer not available for ML Entry Timing Labeler")

    def create_corrected_ml_labels(
        self,
        data: pd.DataFrame,
        analyst_signals: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Create corrected ML-based entry timing labels.

        Args:
            data: Market data with OHLCV columns
            analyst_signals: Analyst green light signals
            regime_assignments: Optional regime assignments

        Returns:
            Tuple of (ml_labels, training_metrics)
        """
        tprint_info("🎯 Creating corrected ML-based entry timing labels...")

        # Step 1: Detect peaks and bottoms within Analyst signal periods up to 0.7% move
        peaks, bottoms = self._detect_peaks_and_bottoms(data, analyst_signals)
        tprint_info(f"📊 Detected {len(peaks)} peaks and {len(bottoms)} bottoms within Analyst signal periods")

        # Step 2: Create entry quality labels based on peak/bottom proximity
        entry_labels = self._create_entry_quality_labels(data, peaks, bottoms, analyst_signals)
        tprint_info(f"📊 Created entry quality labels: {len(entry_labels[entry_labels > 0])} positive entries")

        # Step 3: Generate features for ML training
        features = self._generate_ml_features(data, analyst_signals, regime_assignments)
        tprint_info(f"📊 Generated {len(features.columns)} features for ML training")

        # Step 4: Prepare training data
        X, y, valid_indices = self._prepare_training_data(features, entry_labels)
        tprint_info(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")

        if len(X) < 100:
            tprint_warning("⚠️ Insufficient training data for ML labeling")
            return entry_labels, {'error': 'Insufficient training data'}

        # Step 5: Train ML models
        training_metrics = self._train_models(X, y)
        tprint_info(f"📊 Model training completed. Best R²: {training_metrics.get('best_r2', 0):.3f}")

        # Step 6: Generate ML-based labels
        ml_labels = self._generate_ml_labels(features, valid_indices)

        # Step 7: Calculate quality metrics
        quality_metrics = self._calculate_ml_quality_metrics(
            entry_labels, ml_labels, training_metrics
        )

        tprint_success(f"✅ Corrected ML-based labeling completed")
        tprint_info(f"📊 ML label quality: {quality_metrics.get('overall_quality', 0):.3f}")

        return ml_labels, {**training_metrics, **quality_metrics}

    def _detect_peaks_and_bottoms(self, data: pd.DataFrame, analyst_signals: pd.Series) -> Tuple[List[int], List[int]]:
        """Detect peaks and bottoms only within Analyst signal periods up to 0.7% price move."""
        peaks = []
        bottoms = []

        # Find Analyst green light periods
        green_periods = self._find_green_periods(analyst_signals)

        for period in green_periods:
            period_start = period['start']
            period_end = period['end']
            period_data = data.iloc[period_start:period_end]

            # Determine trend direction from Analyst signal
            trend_direction = self._determine_trend_from_analyst_signal(analyst_signals.iloc[period_start:period_end])

            # Find the 0.7% price move in the right direction
            target_price_move = 0.007  # 0.7%
            start_price = period_data['close'].iloc[0]

            if trend_direction == 'long':
                # For long signals, find 0.7% upward move
                target_price = start_price * (1 + target_price_move)
                move_end_idx = None
                for i, price in enumerate(period_data['close']):
                    if price >= target_price:
                        move_end_idx = period_start + i
                        break
            else:  # short
                # For short signals, find 0.7% downward move
                target_price = start_price * (1 - target_price_move)
                move_end_idx = None
                for i, price in enumerate(period_data['close']):
                    if price <= target_price:
                        move_end_idx = period_start + i
                        break

            # If no 0.7% move found, use the entire period
            if move_end_idx is None:
                move_end_idx = period_end

            # Detect peaks/bottoms only within this limited period
            limited_period_data = data.iloc[period_start:move_end_idx]
            limited_prices = limited_period_data['close'].values

            if len(limited_prices) < 5:  # Need minimum data for peak detection
                continue

            # Detect peaks (local maxima) - only for short signals
            if trend_direction == 'short':
                period_peaks, peak_properties = find_peaks(
                    limited_prices,
                    prominence=self.config.min_peak_prominence,
                    distance=self.config.min_peak_distance
                )
                # Adjust indices to global data index
                global_peaks = [period_start + p for p in period_peaks]
                peaks.extend(global_peaks)

            # Detect bottoms (local minima) - only for long signals
            if trend_direction == 'long':
                period_bottoms, bottom_properties = find_peaks(
                    -limited_prices,  # Invert to find minima
                    prominence=self.config.min_peak_prominence,
                    distance=self.config.min_peak_distance
                )
                # Adjust indices to global data index
                global_bottoms = [period_start + b for b in period_bottoms]
                bottoms.extend(global_bottoms)

        # Store peak/bottom data for analysis
        self.peak_bottom_data = {
            'peaks': peaks,
            'bottoms': bottoms,
            'peak_properties': {},
            'bottom_properties': {}
        }

        return peaks, bottoms

    def _create_entry_quality_labels(
        self,
        data: pd.DataFrame,
        peaks: List[int],
        bottoms: List[int],
        analyst_signals: pd.Series
    ) -> pd.Series:
        """
        Create entry quality labels based on peak/bottom proximity.

        Positive points: Entry at peak (for shorts) or bottom (for longs)
        Negative points: Entry too early (adversarial movement) or too late (missed opportunity)
        """
        labels = pd.Series(0.0, index=data.index)

        # Only consider entries within analyst green light periods
        green_periods = self._find_green_periods(analyst_signals)

        for period in green_periods:
            period_start = period['start']
            period_end = period['end']
            period_data = data.iloc[period_start:period_end]

            # Determine trend direction from Analyst signal
            period_trend = self._determine_trend_from_analyst_signal(analyst_signals.iloc[period_start:period_end])

            if period_trend == 'long':
                # For long opportunities, look for bottoms (buy at bottom)
                relevant_bottoms = [b for b in bottoms if period_start <= b < period_end]
                labels = self._score_long_entries(
                    labels, period_data, relevant_bottoms, period_start
                )
            elif period_trend == 'short':
                # For short opportunities, look for peaks (sell at peak)
                relevant_peaks = [p for p in peaks if period_start <= p < period_end]
                labels = self._score_short_entries(
                    labels, period_data, relevant_peaks, period_start
                )

        return labels

    def _find_green_periods(self, analyst_signals: pd.Series) -> List[Dict[str, int]]:
        """Find continuous green light periods from Analyst signals."""
        green_periods = []
        in_green = False
        start_idx = 0

        for i, signal in enumerate(analyst_signals):
            if signal > 0 and not in_green:
                # Start of green period
                in_green = True
                start_idx = i
            elif signal == 0 and in_green:
                # End of green period
                in_green = False
                if i - start_idx >= 3:  # Minimum period length
                    green_periods.append({
                        'start': start_idx,
                        'end': i,
                        'length': i - start_idx
                    })

        # Handle case where period extends to end
        if in_green and len(analyst_signals) - start_idx >= 3:
            green_periods.append({
                'start': start_idx,
                'end': len(analyst_signals),
                'length': len(analyst_signals) - start_idx
            })

        return green_periods

    def _determine_trend_from_analyst_signal(self, analyst_signals: pd.Series) -> str:
        """Determine trend direction from Analyst signal."""
        # For now, assume Analyst signals are always long (buy signals)
        # In a real implementation, this would depend on the Analyst signal format
        return 'long'

    def _determine_period_trend(self, period_data: pd.DataFrame) -> str:
        """Determine if period is trending up (long) or down (short)."""
        start_price = period_data['close'].iloc[0]
        end_price = period_data['close'].iloc[-1]
        price_change = (end_price - start_price) / start_price

        # Use threshold to determine trend
        if price_change > 0.01:  # 1% up
            return 'long'
        elif price_change < -0.01:  # 1% down
            return 'short'
        else:
            # Use volatility to determine trend for sideways markets
            volatility = period_data['close'].pct_change().std()
            if volatility > 0.02:  # High volatility, use recent trend
                recent_change = period_data['close'].iloc[-5:].pct_change().mean()
                return 'long' if recent_change > 0 else 'short'
            else:
                return 'long'  # Default to long for low volatility

    def _score_long_entries(
        self,
        labels: pd.Series,
        period_data: pd.DataFrame,
        bottoms: List[int],
        period_start: int
    ) -> pd.Series:
        """Score long entries based on proximity to bottoms."""
        for i, row in period_data.iterrows():
            current_idx = i - period_data.index[0] + period_start
            current_price = row['close']

            # Find nearest bottom
            if bottoms:
                distances_to_bottoms = [abs(b - current_idx) for b in bottoms]
                nearest_bottom_idx = bottoms[np.argmin(distances_to_bottoms)]
                nearest_bottom_price = period_data.iloc[nearest_bottom_idx - period_start]['close']

                # Calculate entry quality score
                score = self._calculate_long_entry_score(
                    current_price, nearest_bottom_price, current_idx, nearest_bottom_idx, period_data
                )
                labels.iloc[current_idx] = score

        return labels

    def _score_short_entries(
        self,
        labels: pd.Series,
        period_data: pd.DataFrame,
        peaks: List[int],
        period_start: int
    ) -> pd.Series:
        """Score short entries based on proximity to peaks."""
        for i, row in period_data.iterrows():
            current_idx = i - period_data.index[0] + period_start
            current_price = row['close']

            # Find nearest peak
            if peaks:
                distances_to_peaks = [abs(p - current_idx) for p in peaks]
                nearest_peak_idx = peaks[np.argmin(distances_to_peaks)]
                nearest_peak_price = period_data.iloc[nearest_peak_idx - period_start]['close']

                # Calculate entry quality score
                score = self._calculate_short_entry_score(
                    current_price, nearest_peak_price, current_idx, nearest_peak_idx, period_data
                )
                labels.iloc[current_idx] = score

        return labels

    def _calculate_long_entry_score(
        self,
        current_price: float,
        bottom_price: float,
        current_idx: int,
        bottom_idx: int,
        period_data: pd.DataFrame
    ) -> float:
        """Calculate entry quality score for long positions."""
        # Distance to bottom (closer is better)
        distance_to_bottom = abs(current_idx - bottom_idx)
        distance_score = max(0, 1 - distance_to_bottom / 10)  # Normalize distance

        # Price proximity to bottom (closer is better)
        price_proximity = 1 - abs(current_price - bottom_price) / bottom_price
        price_proximity = max(0, min(1, price_proximity))

        # Check for adverse movement (price going down after entry)
        future_data = period_data.iloc[current_idx - period_data.index[0]:]
        if len(future_data) > 1:
            future_low = future_data['low'].iloc[1:].min()
            adverse_movement = max(0, (current_price - future_low) / current_price)
            adverse_penalty = max(0, 1 - adverse_movement / (self.config.max_adverse_movement_pct / 100))
        else:
            adverse_penalty = 1  # No adverse movement if no future data

        # Check for opportunity capture (price going up after entry)
        if len(future_data) > 1:
            future_high = future_data['high'].iloc[1:].max()
            opportunity_capture = (future_high - current_price) / current_price
            opportunity_score = min(1, opportunity_capture / 0.05)  # Normalize to 5% gain
        else:
            opportunity_score = 0  # No opportunity if no future data

        # Combine scores
        entry_score = (
            distance_score * 0.3 +
            price_proximity * 0.3 +
            adverse_penalty * self.config.adverse_movement_weight +
            opportunity_score * self.config.opportunity_capture_weight
        )

        return max(0, min(1, entry_score))

    def _calculate_short_entry_score(
        self,
        current_price: float,
        peak_price: float,
        current_idx: int,
        peak_idx: int,
        period_data: pd.DataFrame
    ) -> float:
        """Calculate entry quality score for short positions."""
        # Distance to peak (closer is better)
        distance_to_peak = abs(current_idx - peak_idx)
        distance_score = max(0, 1 - distance_to_peak / 10)  # Normalize distance

        # Price proximity to peak (closer is better)
        price_proximity = 1 - abs(current_price - peak_price) / peak_price
        price_proximity = max(0, min(1, price_proximity))

        # Check for adverse movement (price going up after entry)
        future_data = period_data.iloc[current_idx - period_data.index[0]:]
        if len(future_data) > 1:
            future_high = future_data['high'].iloc[1:].max()
            adverse_movement = max(0, (future_high - current_price) / current_price)
            adverse_penalty = max(0, 1 - adverse_movement / (self.config.max_adverse_movement_pct / 100))
        else:
            adverse_penalty = 1  # No adverse movement if no future data

        # Check for opportunity capture (price going down after entry)
        if len(future_data) > 1:
            future_low = future_data['low'].iloc[1:].min()
            opportunity_capture = (current_price - future_low) / current_price
            opportunity_score = min(1, opportunity_capture / 0.05)  # Normalize to 5% gain
        else:
            opportunity_score = 0  # No opportunity if no future data

        # Combine scores
        entry_score = (
            distance_score * 0.3 +
            price_proximity * 0.3 +
            adverse_penalty * self.config.adverse_movement_weight +
            opportunity_score * self.config.opportunity_capture_weight
        )

        return max(0, min(1, entry_score))

    def _generate_ml_features(
        self,
        data: pd.DataFrame,
        analyst_signals: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Generate features for ML training - only peak/bottom and analyst signals."""
        features = pd.DataFrame(index=data.index)

        # Peak/bottom proximity features (primary features)
        peak_bottom_features = self._generate_peak_bottom_features(data)
        features = pd.concat([features, peak_bottom_features], axis=1)

        # Analyst signal features (primary features)
        analyst_features = self._generate_analyst_signal_features(analyst_signals)
        features = pd.concat([features, analyst_features], axis=1)

        # Technical indicators, volume, volatility (for ML models only, not for labeling)
        if self.config.technical_indicators:
            tech_features = self._generate_technical_indicator_features(data)
            features = pd.concat([features, tech_features], axis=1)

        if self.config.volume_features:
            volume_features = self._generate_volume_features(data)
            features = pd.concat([features, volume_features], axis=1)

        if self.config.volatility_features:
            vol_features = self._generate_volatility_features(data)
            features = pd.concat([features, vol_features], axis=1)

        return features

    def _generate_peak_bottom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features based on peak/bottom proximity."""
        features = pd.DataFrame(index=data.index)

        peaks = self.peak_bottom_data.get('peaks', [])
        bottoms = self.peak_bottom_data.get('bottoms', [])

        # Distance to nearest peak
        if peaks:
            peak_distances = []
            for i in range(len(data)):
                distances = [abs(p - i) for p in peaks]
                peak_distances.append(min(distances) if distances else len(data))
            features['distance_to_nearest_peak'] = peak_distances
            features['peak_proximity'] = 1 / (1 + features['distance_to_nearest_peak'])
        else:
            features['distance_to_nearest_peak'] = len(data)
            features['peak_proximity'] = 0

        # Distance to nearest bottom
        if bottoms:
            bottom_distances = []
            for i in range(len(data)):
                distances = [abs(b - i) for b in bottoms]
                bottom_distances.append(min(distances) if distances else len(data))
            features['distance_to_nearest_bottom'] = bottom_distances
            features['bottom_proximity'] = 1 / (1 + features['distance_to_nearest_bottom'])
        else:
            features['distance_to_nearest_bottom'] = len(data)
            features['bottom_proximity'] = 0

        # Peak/bottom density in recent window
        window = 20
        peak_density = []
        bottom_density = []
        for i in range(len(data)):
            start_idx = max(0, i - window)
            recent_peaks = [p for p in peaks if start_idx <= p <= i]
            recent_bottoms = [b for b in bottoms if start_idx <= b <= i]
            peak_density.append(len(recent_peaks) / window)
            bottom_density.append(len(recent_bottoms) / window)

        features['peak_density'] = peak_density
        features['bottom_density'] = bottom_density

        return features

    def _generate_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price action features."""
        features = pd.DataFrame(index=data.index)

        # Basic price features
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['close'] = data['close']

        # Price ratios
        features['hl_ratio'] = data['high'] / data['low']
        features['oc_ratio'] = data['open'] / data['close']
        features['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)

        # Price changes
        for window in self.config.feature_windows:
            features[f'price_change_{window}'] = data['close'].pct_change(window)
            features[f'price_volatility_{window}'] = data['close'].pct_change().rolling(window).std()

        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            features[f'ma_{window}'] = ma
            features[f'price_vs_ma_{window}'] = (data['close'] - ma) / ma

        return features

    def _generate_technical_indicator_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        features = pd.DataFrame(index=data.index)

        # RSI
        for window in [14, 21]:
            rsi = self._calculate_rsi(data['close'], window)
            features[f'rsi_{window}'] = rsi

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(data['close'])
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = histogram

        # Bollinger Bands
        for window in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['close'], window)
            features[f'bb_upper_{window}'] = bb_upper
            features[f'bb_middle_{window}'] = bb_middle
            features[f'bb_lower_{window}'] = bb_lower
            features[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_position_{window}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        return features

    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        features = pd.DataFrame(index=data.index)

        # Volume ratios
        for window in self.config.feature_windows:
            avg_volume = data['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = data['volume'] / (avg_volume + 1e-8)
            features[f'volume_change_{window}'] = data['volume'].pct_change(window)

        # Volume-price relationship
        features['volume_price_trend'] = (data['volume'] * data['close'].pct_change()).rolling(20).sum()

        # VWAP
        vwap = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
        features['vwap'] = vwap
        features['price_vs_vwap'] = (data['close'] - vwap) / vwap

        return features

    def _generate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features."""
        features = pd.DataFrame(index=data.index)

        # Rolling volatility
        for window in self.config.feature_windows:
            returns = data['close'].pct_change()
            vol = returns.rolling(window).std()
            features[f'volatility_{window}'] = vol

            # Volatility of volatility
            vol_of_vol = vol.rolling(window).std()
            features[f'vol_of_vol_{window}'] = vol_of_vol

        # GARCH-like features
        returns = data['close'].pct_change()
        features['returns'] = returns
        features['abs_returns'] = abs(returns)
        features['squared_returns'] = returns ** 2

        return features

    def _generate_analyst_signal_features(self, analyst_signals: pd.Series) -> pd.DataFrame:
        """Generate analyst signal features."""
        features = pd.DataFrame(index=analyst_signals.index)

        features['analyst_signal'] = analyst_signals

        # Signal strength over time
        for window in [3, 5, 10]:
            features[f'analyst_signal_strength_{window}'] = analyst_signals.rolling(window).mean()
            features[f'analyst_signal_consistency_{window}'] = analyst_signals.rolling(window).std()

        return features

    def _generate_time_features(self, index: pd.Index) -> pd.DataFrame:
        """Generate time-based features."""
        features = pd.DataFrame(index=index)

        # Time components
        features['hour'] = index.hour
        features['day_of_week'] = index.dayofweek
        features['day_of_month'] = index.day
        features['month'] = index.month

        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        return features

    def _prepare_training_data(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Prepare training data for ML models."""
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]

        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        valid_indices = X_clean.index

        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())

        return X_clean.values, y_clean.values, valid_indices

    def _train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train ML models for entry timing prediction."""
        training_metrics = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        # Train models
        model_configs = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.config.random_state),
            'ridge': Ridge(alpha=1.0)
        }

        best_model = None
        best_score = -np.inf

        for model_name in self.config.models:
            if model_name not in model_configs:
                continue

            tprint_info(f"🤖 Training {model_name}...")

            model = model_configs[model_name]
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.config.cross_validation_folds)

            training_metrics[model_name] = {
                'r2_score': r2,
                'mse': mse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            # Store model if it's the best
            if r2 > best_score:
                best_score = r2
                best_model = model_name
                self.models['best'] = model

            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_

            tprint_info(f"   R²: {r2:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        training_metrics['best_model'] = best_model
        training_metrics['best_r2'] = best_score

        return training_metrics

    def _generate_ml_labels(
        self,
        features: pd.DataFrame,
        valid_indices: pd.Index
    ) -> pd.Series:
        """Generate ML-based labels using trained models."""
        if 'best' not in self.models:
            tprint_error("❌ No trained model available for label generation")
            return pd.Series(0, index=features.index)

        # Prepare features
        X = features.loc[valid_indices]
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Scale features
        X_scaled = self.scalers['main'].transform(X_clean)

        # Generate predictions
        predictions = self.models['best'].predict(X_scaled)

        # Create labels series
        ml_labels = pd.Series(0, index=features.index, dtype=float)
        ml_labels.loc[valid_indices] = predictions

        # Apply quality threshold
        quality_threshold = np.percentile(predictions[predictions > 0], 70) if (predictions > 0).any() else 0.5
        ml_labels = ml_labels.where(ml_labels >= quality_threshold, 0)

        return ml_labels

    def _calculate_ml_quality_metrics(
        self,
        initial_labels: pd.Series,
        ml_labels: pd.Series,
        training_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality metrics for ML-based labeling."""
        metrics = {}

        # Basic metrics
        initial_positive = (initial_labels > 0).sum()
        ml_positive = (ml_labels > 0).sum()

        metrics['initial_positive_count'] = initial_positive
        metrics['ml_positive_count'] = ml_positive
        metrics['label_change_ratio'] = ml_positive / initial_positive if initial_positive > 0 else 0

        # Correlation with initial labels
        common_index = initial_labels.index.intersection(ml_labels.index)
        if len(common_index) > 0:
            correlation = initial_labels.loc[common_index].corr(ml_labels.loc[common_index])
            metrics['correlation_with_initial'] = correlation if not np.isnan(correlation) else 0

        # Model performance
        metrics['best_r2_score'] = training_metrics.get('best_r2', 0)
        metrics['best_model'] = training_metrics.get('best_model', 'unknown')

        # Overall quality
        metrics['overall_quality'] = (
            metrics.get('correlation_with_initial', 0) * 0.4 +
            metrics.get('best_r2_score', 0) * 0.6
        )

        return metrics

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    def save_models(self, filepath: str) -> None:
        """Save trained models and scalers."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'peak_bottom_data': self.peak_bottom_data
        }
        joblib.dump(model_data, filepath)
        tprint_success(f"✅ Models saved to {filepath}")

    def load_models(self, filepath: str) -> None:
        """Load trained models and scalers."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.peak_bottom_data = model_data.get('peak_bottom_data', {})
        tprint_success(f"✅ Models loaded from {filepath}")

    def _optimized_rolling_operation(self, data: pd.Series, operation: str,
                                   window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT Rolling Optimizer."""
        if self.vectorbt_optimizer is not None:
            try:
                if operation == 'mean':
                    return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.vectorbt_optimizer.rolling_apply(data, func, window=window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT Rolling Optimizer failed for {operation}: {e}, using fallback")
                return self._fallback_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _fallback_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses optimized rolling operations."""
        return self._optimized_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses fallback rolling operations."""
        return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses optimized rolling operations."""
        return self._optimized_rolling_operation(data, 'apply', window, func=func, **kwargs)
