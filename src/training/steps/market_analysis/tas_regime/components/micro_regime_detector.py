"""
Micro-Regime Detector for Advanced TAS

This module provides sophisticated micro-regime detection capabilities for:
- Breakout detection
- Consolidation patterns
- Reversal signals
- Acceleration/deceleration detection
- Volume and volatility spikes
- Momentum shifts
- Liquidity changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
# DBSCAN clustering removed - will be handled in subsequent step
from sklearn.mixture import GaussianMixture

from ..core.tas_config import MicroRegimeType, MarketRegime

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class MicroRegimeDetectionResult:
    """Result of micro-regime detection."""
    regime_type: MicroRegimeType
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    characteristics: Dict[str, float] = field(default_factory=dict)
    signal_strength: float = 0.0
    duration_minutes: float = 0.0
    transition_probability: float = 0.0


class MicroRegimeDetector:
    """Advanced micro-regime detector for short-term trading."""

    def __init__(self, sensitivity: float = 0.7, detection_threshold: float = 0.6):
        """Initialize micro-regime detector.

        Args:
            sensitivity: Detection sensitivity (0-1)
            detection_threshold: Minimum confidence for detection
        """
        tprint_info("🔬 Initializing Micro Regime Detector")
        tprint_debug(f"Sensitivity: {sensitivity}")
        tprint_debug(f"Detection threshold: {detection_threshold}")
        
        self.sensitivity = sensitivity
        self.detection_threshold = detection_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'detection_time': 0.0,
            'analysis_time': 0.0,
            'total_execution_time': 0.0
        }

        # Detection parameters
        self.breakout_params = {
            'price_threshold': 2.0,  # Standard deviations for breakout
            'volume_multiplier': 1.5,  # Volume increase for confirmation
            'min_duration': 5  # Minimum breakout duration in minutes
        }

        self.consolidation_params = {
            'volatility_threshold': 0.3,  # Max volatility for consolidation
            'price_range_threshold': 0.02,  # Max price range (2%)
            'min_duration': 15  # Minimum consolidation duration
        }

        self.reversal_params = {
            'momentum_threshold': 0.7,  # Momentum change threshold
            'volume_confirmation': 1.2,  # Volume confirmation
            'rsi_threshold': 30  # RSI threshold for reversal
        }

        self.acceleration_params = {
            'momentum_acceleration': 0.1,  # Rate of momentum change
            'volume_trend': 1.3,  # Volume trend confirmation
            'duration_threshold': 10
        }

        self.volume_spike_params = {
            'volume_multiplier': 2.0,
            'price_confirmation': 0.01,  # 1% price movement
            'isolation_threshold': 0.8
        }

        self.volatility_spike_params = {
            'volatility_multiplier': 2.5,
            'duration_threshold': 5,
            'price_impact_threshold': 0.02
        }

    def detect_micro_regimes(self, market_data: pd.DataFrame,
                           current_regime: Optional[MarketRegime] = None) -> List[MicroRegimeDetectionResult]:
        """Detect micro-regimes in market data.

        Args:
            market_data: Market data with OHLCV and indicators
            current_regime: Current market regime context

        Returns:
            List of detected micro-regimes
        """
        self.logger.info("🔍 Starting micro-regime detection...")
        tprint("🔍 Starting micro-regime detection...", color="blue")

        detected_regimes = []

        try:
            # Preprocess data
            tprint("📊 Preprocessing data for micro-regime detection...", color="cyan")
            processed_data = self._preprocess_data(market_data)
            tprint(f"✅ Data preprocessed: {len(processed_data)} data points", color="green")

            # Detect different micro-regime types
            tprint("🚀 Detecting breakouts...", color="yellow")
            breakout_regimes = self._detect_breakouts(processed_data)
            detected_regimes.extend(breakout_regimes)
            tprint(f"✅ Breakouts detected: {len(breakout_regimes)}", color="green")

            tprint("📊 Detecting consolidations...", color="yellow")
            consolidation_regimes = self._detect_consolidations(processed_data)
            detected_regimes.extend(consolidation_regimes)
            tprint(f"✅ Consolidations detected: {len(consolidation_regimes)}", color="green")

            tprint("🔄 Detecting reversals...", color="yellow")
            reversal_regimes = self._detect_reversals(processed_data)
            detected_regimes.extend(reversal_regimes)
            tprint(f"✅ Reversals detected: {len(reversal_regimes)}", color="green")

            tprint("⚡ Detecting accelerations...", color="yellow")
            acceleration_regimes = self._detect_accelerations(processed_data)
            detected_regimes.extend(acceleration_regimes)
            tprint(f"✅ Accelerations detected: {len(acceleration_regimes)}", color="green")

            tprint("📈 Detecting volume spikes...", color="yellow")
            volume_spikes = self._detect_volume_spikes(processed_data)
            detected_regimes.extend(volume_spikes)
            tprint(f"✅ Volume spikes detected: {len(volume_spikes)}", color="green")

            tprint("📊 Detecting volatility spikes...", color="yellow")
            volatility_spikes = self._detect_volatility_spikes(processed_data)
            detected_regimes.extend(volatility_spikes)
            tprint(f"✅ Volatility spikes detected: {len(volatility_spikes)}", color="green")

            # Filter and rank by confidence
            tprint("🔍 Filtering and ranking micro-regimes...", color="cyan")
            filtered_regimes = [r for r in detected_regimes if r.confidence >= self.detection_threshold]
            filtered_regimes.sort(key=lambda x: x.confidence, reverse=True)
            tprint(f"✅ Filtered micro-regimes: {len(filtered_regimes)}", color="green")

            self.logger.info(f"✅ Detected {len(filtered_regimes)} micro-regimes")
            tprint(f"✅ Detected {len(filtered_regimes)} micro-regimes", color="green")
            return filtered_regimes

        except Exception as e:
            self.logger.error(f"Micro-regime detection failed: {e}")
            tprint(f"❌ Micro-regime detection failed: {e}", color="red")
            return []

    def _preprocess_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess market data for micro-regime detection."""
        returns = market_data['close'].pct_change()
        log_returns = np.log(market_data['close']).diff()

        # Calculate technical indicators
        sma_20 = market_data['close'].rolling(20).mean()
        sma_50 = market_data['close'].rolling(50).mean()
        volatility_20 = returns.rolling(20).std()
        volatility_50 = returns.rolling(50).std()

        # Calculate momentum
        momentum_5 = market_data['close'] / market_data['close'].shift(5) - 1
        momentum_10 = market_data['close'] / market_data['close'].shift(10) - 1

        # Calculate RSI
        rsi = self._calculate_rsi(market_data['close'], 14)

        # Calculate volume metrics
        volume_ma_20 = market_data.get('volume', 1).rolling(20).mean()
        volume_ratio = market_data.get('volume', 1) / volume_ma_20

        # Calculate price derivatives
        price_velocity = returns.rolling(5).mean()
        price_acceleration = price_velocity.diff()

        return {
            'price': market_data['close'],
            'returns': returns,
            'log_returns': log_returns,
            'volume': market_data.get('volume', 1),
            'high': market_data['high'],
            'low': market_data['low'],
            'sma_20': sma_20,
            'sma_50': sma_50,
            'volatility_20': volatility_20,
            'volatility_50': volatility_50,
            'momentum_5': momentum_5,
            'momentum_10': momentum_10,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'price_velocity': price_velocity,
            'price_acceleration': price_acceleration,
            'timestamp': market_data.index
        }

    def _detect_breakouts(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect breakout micro-regimes."""
        breakouts = []

        try:
            # Price-based breakout detection
            price = data['price']
            returns = data['returns']

            # Calculate rolling statistics
            rolling_mean = price.rolling(20).mean()
            rolling_std = price.rolling(20).std()

            # Detect price breakouts
            price_breakout_mask = (
                (price > rolling_mean + self.breakout_params['price_threshold'] * rolling_std) |
                (price < rolling_mean - self.breakout_params['price_threshold'] * rolling_std)
            )

            # Find breakout periods
            breakout_periods = self._find_contiguous_periods(price_breakout_mask)

            for start_idx, end_idx in breakout_periods:
                if end_idx - start_idx >= self.breakout_params['min_duration']:

                    # Calculate breakout characteristics
                    breakout_price = price.iloc[start_idx:end_idx]
                    breakout_returns = returns.iloc[start_idx:end_idx]
                    breakout_volume = data['volume'].iloc[start_idx:end_idx]

                    # Check volume confirmation
                    avg_volume = breakout_volume.mean()
                    baseline_volume = data['volume'].iloc[:start_idx].tail(20).mean()
                    volume_confirmation = avg_volume > baseline_volume * self.breakout_params['volume_multiplier']

                    if volume_confirmation:
                        confidence = self._calculate_breakout_confidence(
                            breakout_price, breakout_returns, breakout_volume
                        )

                        characteristics = {
                            'breakout_strength': abs(breakout_returns).mean(),
                            'volume_multiplier': avg_volume / baseline_volume,
                            'price_range': (breakout_price.max() - breakout_price.min()) / breakout_price.iloc[0],
                            'duration': len(breakout_price)
                        }

                        breakout_regime = MicroRegimeDetectionResult(
                            regime_type=MicroRegimeType.BREAKOUT,
                            confidence=confidence,
                            start_time=data['timestamp'][start_idx],
                            end_time=data['timestamp'][end_idx],
                            characteristics=characteristics,
                            signal_strength=abs(breakout_returns).mean(),
                            duration_minutes=(end_idx - start_idx) * 5,  # Assuming 5-minute bars
                            transition_probability=self._calculate_transition_probability(
                                MicroRegimeType.BREAKOUT, data, start_idx
                            )
                        )

                        breakouts.append(breakout_regime)

        except Exception as e:
            self.logger.warning(f"Breakout detection failed: {e}")

        return breakouts

    def _detect_consolidations(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect consolidation micro-regimes."""
        consolidations = []

        try:
            price = data['price']
            volatility = data['volatility_20']

            # Detect consolidation periods (low volatility)
            consolidation_mask = volatility < self.consolidation_params['volatility_threshold']

            # Add price range constraint
            price_range = (price - price.shift(1)).abs()
            max_range = price * self.consolidation_params['price_range_threshold']
            consolidation_mask &= price_range <= max_range

            # Find consolidation periods
            consolidation_periods = self._find_contiguous_periods(consolidation_mask)

            for start_idx, end_idx in consolidation_periods:
                if end_idx - start_idx >= self.consolidation_params['min_duration']:

                    consolidation_price = price.iloc[start_idx:end_idx]
                    consolidation_volatility = volatility.iloc[start_idx:end_idx]

                    confidence = self._calculate_consolidation_confidence(
                        consolidation_price, consolidation_volatility
                    )

                    characteristics = {
                        'avg_volatility': consolidation_volatility.mean(),
                        'price_range': (consolidation_price.max() - consolidation_price.min()) / consolidation_price.iloc[0],
                        'duration': len(consolidation_price),
                        'volatility_trend': consolidation_volatility.iloc[-1] / consolidation_volatility.iloc[0]
                    }

                    consolidation_regime = MicroRegimeDetectionResult(
                        regime_type=MicroRegimeType.CONSOLIDATION,
                        confidence=confidence,
                        start_time=data['timestamp'][start_idx],
                        end_time=data['timestamp'][end_idx],
                        characteristics=characteristics,
                        signal_strength=1 - consolidation_volatility.mean(),
                        duration_minutes=(end_idx - start_idx) * 5,
                        transition_probability=self._calculate_transition_probability(
                            MicroRegimeType.CONSOLIDATION, data, start_idx
                        )
                    )

                    consolidations.append(consolidation_regime)

        except Exception as e:
            self.logger.warning(f"Consolidation detection failed: {e}")

        return consolidations

    def _detect_reversals(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect reversal micro-regimes."""
        reversals = []

        try:
            price = data['price']
            momentum = data['momentum_5']
            volume = data['volume']
            rsi = data['rsi']

            # Detect momentum reversals
            momentum_change = momentum - momentum.shift(1)
            reversal_mask = abs(momentum_change) > self.reversal_params['momentum_threshold']

            # Add RSI confirmation for oversold/overbought
            rsi_reversal = (rsi < self.reversal_params['rsi_threshold']) | (rsi > (100 - self.reversal_params['rsi_threshold']))

            # Add volume confirmation
            volume_baseline = volume.rolling(20).mean()
            volume_reversal = volume > volume_baseline * self.reversal_params['volume_confirmation']

            reversal_mask &= (rsi_reversal | volume_reversal)

            # Find reversal periods
            reversal_periods = self._find_contiguous_periods(reversal_mask)

            for start_idx, end_idx in reversal_periods:
                if end_idx - start_idx >= 3:  # Minimum 3 periods for reversal

                    reversal_momentum = momentum.iloc[start_idx:end_idx]
                    reversal_price = price.iloc[start_idx:end_idx]
                    reversal_rsi = rsi.iloc[start_idx:end_idx]
                    reversal_volume = volume.iloc[start_idx:end_idx]

                    confidence = self._calculate_reversal_confidence(
                        reversal_momentum, reversal_price, reversal_rsi, reversal_volume
                    )

                    characteristics = {
                        'momentum_change': abs(reversal_momentum).mean(),
                        'price_change': (reversal_price.iloc[-1] / reversal_price.iloc[0] - 1),
                        'rsi_level': reversal_rsi.iloc[-1],
                        'volume_multiplier': reversal_volume.iloc[-1] / volume_baseline.iloc[start_idx],
                        'reversal_duration': len(reversal_price)
                    }

                    reversal_regime = MicroRegimeDetectionResult(
                        regime_type=MicroRegimeType.REVERSAL,
                        confidence=confidence,
                        start_time=data['timestamp'][start_idx],
                        end_time=data['timestamp'][end_idx],
                        characteristics=characteristics,
                        signal_strength=abs(reversal_momentum).mean(),
                        duration_minutes=(end_idx - start_idx) * 5,
                        transition_probability=self._calculate_transition_probability(
                            MicroRegimeType.REVERSAL, data, start_idx
                        )
                    )

                    reversals.append(reversal_regime)

        except Exception as e:
            self.logger.warning(f"Reversal detection failed: {e}")

        return reversals

    def _detect_accelerations(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect acceleration/deceleration micro-regimes."""
        accelerations = []

        try:
            momentum = data['momentum_5']
            acceleration = data['price_acceleration']
            volume_trend = data['volume_ratio']

            # Detect acceleration periods
            acceleration_mask = abs(acceleration) > self.acceleration_params['momentum_acceleration']
            volume_trend_mask = volume_trend > self.acceleration_params['volume_trend']

            acceleration_mask &= volume_trend_mask

            # Find acceleration periods
            acceleration_periods = self._find_contiguous_periods(acceleration_mask)

            for start_idx, end_idx in acceleration_periods:
                if end_idx - start_idx >= self.acceleration_params['duration_threshold']:

                    acc_momentum = momentum.iloc[start_idx:end_idx]
                    acc_acceleration = acceleration.iloc[start_idx:end_idx]
                    acc_volume = volume_trend.iloc[start_idx:end_idx]

                    # Determine if acceleration or deceleration
                    avg_acceleration = acc_acceleration.mean()
                    if avg_acceleration > 0:
                        regime_type = MicroRegimeType.ACCELERATION
                    else:
                        regime_type = MicroRegimeType.DECELERATION

                    confidence = self._calculate_acceleration_confidence(
                        acc_momentum, acc_acceleration, acc_volume
                    )

                    characteristics = {
                        'avg_acceleration': avg_acceleration,
                        'momentum_trend': acc_momentum.iloc[-1] - acc_momentum.iloc[0],
                        'volume_trend': acc_volume.iloc[-1] / acc_volume.iloc[0],
                        'acceleration_duration': len(acc_momentum)
                    }

                    acc_regime = MicroRegimeDetectionResult(
                        regime_type=regime_type,
                        confidence=confidence,
                        start_time=data['timestamp'][start_idx],
                        end_time=data['timestamp'][end_idx],
                        characteristics=characteristics,
                        signal_strength=abs(avg_acceleration),
                        duration_minutes=(end_idx - start_idx) * 5,
                        transition_probability=self._calculate_transition_probability(
                            regime_type, data, start_idx
                        )
                    )

                    accelerations.append(acc_regime)

        except Exception as e:
            self.logger.warning(f"Acceleration detection failed: {e}")

        return accelerations

    def _detect_volume_spikes(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect volume spike micro-regimes."""
        spikes = []

        try:
            volume = data['volume']
            price = data['price']

            # Detect volume spikes
            volume_ma = volume.rolling(20).mean()
            volume_spike_mask = volume > volume_ma * self.volume_spike_params['volume_multiplier']

            # Price confirmation
            price_change = abs(price - price.shift(1)) / price.shift(1)
            price_confirmation = price_change > self.volume_spike_params['price_confirmation']

            volume_spike_mask &= price_confirmation

            # Find spike periods
            spike_periods = self._find_contiguous_periods(volume_spike_mask)

            for start_idx, end_idx in spike_periods:
                spike_volume = volume.iloc[start_idx:end_idx]
                spike_price = price.iloc[start_idx:end_idx]

                confidence = self._calculate_volume_spike_confidence(spike_volume, volume_ma.iloc[start_idx:end_idx])

                characteristics = {
                    'volume_multiplier': spike_volume.iloc[-1] / volume_ma.iloc[start_idx],
                    'price_impact': abs(spike_price.iloc[-1] / spike_price.iloc[0] - 1),
                    'spike_duration': len(spike_volume),
                    'volume_trend': spike_volume.iloc[-1] / spike_volume.iloc[0]
                }

                spike_regime = MicroRegimeDetectionResult(
                    regime_type=MicroRegimeType.VOLUME_SPIKE,
                    confidence=confidence,
                    start_time=data['timestamp'][start_idx],
                    end_time=data['timestamp'][end_idx],
                    characteristics=characteristics,
                    signal_strength=spike_volume.iloc[-1] / volume_ma.iloc[start_idx],
                    duration_minutes=(end_idx - start_idx) * 5,
                    transition_probability=self._calculate_transition_probability(
                        MicroRegimeType.VOLUME_SPIKE, data, start_idx
                    )
                )

                spikes.append(spike_regime)

        except Exception as e:
            self.logger.warning(f"Volume spike detection failed: {e}")

        return spikes

    def _detect_volatility_spikes(self, data: Dict[str, Any]) -> List[MicroRegimeDetectionResult]:
        """Detect volatility spike micro-regimes."""
        spikes = []

        try:
            volatility = data['volatility_20']
            price = data['price']

            # Detect volatility spikes
            vol_ma = volatility.rolling(20).mean()
            vol_spike_mask = volatility > vol_ma * self.volatility_spike_params['volatility_multiplier']

            # Price impact confirmation
            price_impact = abs(price - price.shift(1)) / price.shift(1)
            price_impact_mask = price_impact > self.volatility_spike_params['price_impact_threshold']

            vol_spike_mask &= price_impact_mask

            # Find spike periods
            spike_periods = self._find_contiguous_periods(vol_spike_mask)

            for start_idx, end_idx in spike_periods:
                if end_idx - start_idx >= self.volatility_spike_params['duration_threshold']:

                    spike_volatility = volatility.iloc[start_idx:end_idx]
                    spike_price = price.iloc[start_idx:end_idx]

                    confidence = self._calculate_volatility_spike_confidence(
                        spike_volatility, vol_ma.iloc[start_idx:end_idx]
                    )

                    characteristics = {
                        'volatility_multiplier': spike_volatility.iloc[-1] / vol_ma.iloc[start_idx],
                        'price_impact': abs(spike_price.iloc[-1] / spike_price.iloc[0] - 1),
                        'volatility_duration': len(spike_volatility),
                        'max_volatility': spike_volatility.max()
                    }

                    spike_regime = MicroRegimeDetectionResult(
                        regime_type=MicroRegimeType.VOLATILITY_SPIKE,
                        confidence=confidence,
                        start_time=data['timestamp'][start_idx],
                        end_time=data['timestamp'][end_idx],
                        characteristics=characteristics,
                        signal_strength=spike_volatility.iloc[-1] / vol_ma.iloc[start_idx],
                        duration_minutes=(end_idx - start_idx) * 5,
                        transition_probability=self._calculate_transition_probability(
                            MicroRegimeType.VOLATILITY_SPIKE, data, start_idx
                        )
                    )

                    spikes.append(spike_regime)

        except Exception as e:
            self.logger.warning(f"Volatility spike detection failed: {e}")

        return spikes

    def _calculate_breakout_confidence(self, price: pd.Series, returns: pd.Series, volume: pd.Series) -> float:
        """Calculate confidence for breakout detection."""
        # Multi-factor confidence calculation
        price_confidence = min(1.0, abs(returns).mean() * 10)
        volume_confidence = min(1.0, (volume.iloc[-1] / volume.iloc[0] - 1) * 2)
        duration_confidence = min(1.0, len(price) / 20)

        return (price_confidence * 0.4 + volume_confidence * 0.4 + duration_confidence * 0.2)

    def _calculate_consolidation_confidence(self, price: pd.Series, volatility: pd.Series) -> float:
        """Calculate confidence for consolidation detection."""
        volatility_confidence = max(0.0, 1 - volatility.mean() / self.consolidation_params['volatility_threshold'])
        range_confidence = max(0.0, 1 - (price.max() - price.min()) / price.iloc[0] / self.consolidation_params['price_range_threshold'])
        duration_confidence = min(1.0, len(price) / 30)

        return (volatility_confidence * 0.4 + range_confidence * 0.4 + duration_confidence * 0.2)

    def _calculate_reversal_confidence(self, momentum: pd.Series, price: pd.Series,
                                    rsi: pd.Series, volume: pd.Series) -> float:
        """Calculate confidence for reversal detection."""
        momentum_confidence = min(1.0, abs(momentum).mean() / self.reversal_params['momentum_threshold'])
        rsi_confidence = min(1.0, abs(rsi.iloc[-1] - 50) / 50)
        volume_confidence = min(1.0, (volume.iloc[-1] / volume.iloc[0] - 1) / (self.reversal_params['volume_confirmation'] - 1))
        price_confidence = min(1.0, abs(price.iloc[-1] / price.iloc[0] - 1) * 10)

        return (momentum_confidence * 0.3 + rsi_confidence * 0.3 + volume_confidence * 0.2 + price_confidence * 0.2)

    def _calculate_acceleration_confidence(self, momentum: pd.Series, acceleration: pd.Series, volume: pd.Series) -> float:
        """Calculate confidence for acceleration detection."""
        acceleration_confidence = min(1.0, abs(acceleration).mean() / self.acceleration_params['momentum_acceleration'])
        momentum_confidence = min(1.0, abs(momentum).mean() * 5)
        volume_confidence = min(1.0, (volume.iloc[-1] / volume.iloc[0] - 1) / (self.acceleration_params['volume_trend'] - 1))

        return (acceleration_confidence * 0.5 + momentum_confidence * 0.3 + volume_confidence * 0.2)

    def _calculate_volume_spike_confidence(self, volume: pd.Series, baseline_volume: pd.Series) -> float:
        """Calculate confidence for volume spike detection."""
        volume_ratio = volume.iloc[-1] / baseline_volume.iloc[-1]
        isolation_factor = 1 - (1 / (1 + len(volume)))  # Confidence increases with duration

        return min(1.0, volume_ratio / self.volume_spike_params['volume_multiplier'] * isolation_factor)

    def _calculate_volatility_spike_confidence(self, volatility: pd.Series, baseline_volatility: pd.Series) -> float:
        """Calculate confidence for volatility spike detection."""
        volatility_ratio = volatility.iloc[-1] / baseline_volatility.iloc[-1]
        duration_factor = min(1.0, len(volatility) / 10)

        return min(1.0, volatility_ratio / self.volatility_spike_params['volatility_multiplier'] * duration_factor)

    def _calculate_transition_probability(self, regime_type: MicroRegimeType, data: Dict[str, Any], start_idx: int) -> float:
        """Calculate probability of transitioning to this micro-regime."""
        # Base transition probabilities based on market conditions
        base_probabilities = {
            MicroRegimeType.BREAKOUT: 0.15,
            MicroRegimeType.CONSOLIDATION: 0.25,
            MicroRegimeType.REVERSAL: 0.10,
            MicroRegimeType.ACCELERATION: 0.20,
            MicroRegimeType.DECELERATION: 0.15,
            MicroRegimeType.VOLUME_SPIKE: 0.10,
            MicroRegimeType.VOLATILITY_SPIKE: 0.05
        }

        base_prob = base_probabilities.get(regime_type, 0.1)

        # Adjust based on recent market conditions
        if start_idx > 20:
            recent_volatility = data['volatility_20'].iloc[start_idx-20:start_idx].mean()
            recent_volume = data['volume_ratio'].iloc[start_idx-20:start_idx].mean()

            # Higher volatility increases breakout and volatility spike probability
            if regime_type in [MicroRegimeType.BREAKOUT, MicroRegimeType.VOLATILITY_SPIKE]:
                base_prob *= (1 + recent_volatility)

            # Higher volume increases volume spike probability
            if regime_type == MicroRegimeType.VOLUME_SPIKE:
                base_prob *= recent_volume

        return min(1.0, base_prob)

    def _find_contiguous_periods(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Find contiguous periods where mask is True."""
        periods = []
        start_idx = None

        for i, val in enumerate(mask):
            if val and start_idx is None:
                start_idx = i
            elif not val and start_idx is not None:
                periods.append((start_idx, i))
                start_idx = None

        if start_idx is not None:
            periods.append((start_idx, len(mask)))

        return periods

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Fill NaN with neutral RSI

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
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
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
